"""Uncertainty and reproducibility analysis on the framework's raw per-sample logs.

This script complements make_revision_outputs.py, which works from aggregated
values only. Given the sample-level CSV files written by the benchmarking
framework (one row per measured request), it produces:

1. accuracy per configuration with an item-level percentile bootstrap and a
   Wilson interval (items are the unit of analysis; repeated presentations of
   the same item across runs are first averaged per item);
2. paired exact McNemar tests between all configurations evaluated on the same
   items within a context and benchmark, with Holm adjustment;
3. 95% intervals for mean TTFT, end-to-end latency, tokens/s and (if present)
   energy per request: a t-interval over the ten run-level means and a
   two-stage cluster bootstrap (runs, then requests within runs) that preserves
   within-run correlation; the cluster bootstrap also gives a p95 interval;
4. a run manifest (configuration, run, seed if logged, ordered item IDs) and the
   pairwise item-overlap (Jaccard) between runs and between configurations.

Usage:
    python run_level_uncertainty.py --samples "results/**/*samples*.csv" --out derived_raw
    python run_level_uncertainty.py --help      # column-name mapping options
"""
from __future__ import annotations

import argparse
import glob
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest, t

from stats_utils import holm, wilson

DEFAULT_COLUMNS = {
    "context": "platform", "model": "model", "dataset": "dataset", "run": "run",
    "item": "item_id", "correct": "correct", "ttft": "ttft_ms", "latency": "latency_ms",
    "tps": "tokens_per_sec", "energy": "energy_j", "seed": "seed",
}


def load_samples(pattern: str, cols: dict) -> pd.DataFrame:
    files = sorted(glob.glob(pattern, recursive=True))
    if not files:
        raise SystemExit(f"no files match {pattern!r}")
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    rename = {v: k for k, v in cols.items() if v in df.columns}
    df = df.rename(columns=rename)
    missing = {"context", "model", "dataset", "run", "item", "latency"} - set(df.columns)
    if missing:
        raise SystemExit(f"missing required columns: {sorted(missing)}; use --col-* options")
    return df


def cluster_bootstrap(df: pd.DataFrame, value: str, b: int, rng: np.random.Generator):
    """Two-stage bootstrap: resample runs, then requests within each run."""
    groups = [g[value].dropna().to_numpy() for _, g in df.groupby("run")]
    groups = [g for g in groups if len(g)]
    means, p95s = np.empty(b), np.empty(b)
    for k in range(b):
        picks = rng.integers(0, len(groups), len(groups))
        sample = np.concatenate([rng.choice(groups[i], len(groups[i])) for i in picks])
        means[k], p95s[k] = sample.mean(), np.percentile(sample, 95)
    return np.percentile(means, [2.5, 97.5]), np.percentile(p95s, [2.5, 97.5])


def run_t_interval(df: pd.DataFrame, value: str):
    m = df.groupby("run")[value].mean().dropna().to_numpy()
    if len(m) < 2:
        return (np.nan, np.nan), np.nan
    half = t.ppf(0.975, len(m) - 1) * m.std(ddof=1) / np.sqrt(len(m))
    return (m.mean() - half, m.mean() + half), m.std(ddof=1)


def item_accuracy(df: pd.DataFrame) -> pd.Series:
    return df.groupby("item")["correct"].mean()


def mcnemar_exact(a: pd.Series, b: pd.Series) -> tuple[int, int, float]:
    common = a.index.intersection(b.index)
    a_, b_ = a.loc[common].round().astype(int), b.loc[common].round().astype(int)
    n01 = int(((a_ == 1) & (b_ == 0)).sum())
    n10 = int(((a_ == 0) & (b_ == 1)).sum())
    p = 1.0 if n01 + n10 == 0 else binomtest(n01, n01 + n10, 0.5).pvalue
    return n01, n10, p


def analyse(df: pd.DataFrame, b: int, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    acc_rows, sys_rows, pair_rows, manifest, overlap_rows = [], [], [], [], []
    for (ctx, model, ds), g in df.groupby(["context", "model", "dataset"]):
        for run, gr in g.groupby("run"):
            manifest.append({"context": ctx, "model": model, "dataset": ds, "run": run,
                             "seed": gr["seed"].iloc[0] if "seed" in gr else None,
                             "items": list(map(str, gr["item"]))})
        runs = {r: set(gr["item"]) for r, gr in g.groupby("run")}
        for r1, r2 in itertools.combinations(sorted(runs), 2):
            inter = len(runs[r1] & runs[r2])
            overlap_rows.append({"context": ctx, "model": model, "dataset": ds, "run_a": r1,
                                 "run_b": r2, "shared_items": inter,
                                 "jaccard": inter / len(runs[r1] | runs[r2])})
        if "correct" in g and g["correct"].notna().any():
            items = item_accuracy(g.dropna(subset=["correct"]))
            n, p = len(items), items.mean()
            boot = [items.sample(n, replace=True, random_state=rng.integers(1 << 31)).mean()
                    for _ in range(b)]
            lo, hi = wilson(p, n)
            per_run = g.groupby("run")["correct"].mean()
            acc_rows.append({"context": ctx, "model": model, "dataset": ds, "n_items": n,
                             "accuracy": p, "wilson_low": lo, "wilson_high": hi,
                             "boot_low": np.percentile(boot, 2.5),
                             "boot_high": np.percentile(boot, 97.5),
                             "run_sd": per_run.std(ddof=1), "run_min": per_run.min(),
                             "run_max": per_run.max()})
        for col in ("ttft", "latency", "tps", "energy"):
            if col not in g or g[col].isna().all():
                continue
            (tl, th), sd = run_t_interval(g, col)
            (ml, mh), (pl, ph) = cluster_bootstrap(g, col, b, rng)
            sys_rows.append({"context": ctx, "model": model, "dataset": ds, "metric": col,
                             "mean": g[col].mean(), "run_sd": sd, "t_low": tl, "t_high": th,
                             "boot_low": ml, "boot_high": mh,
                             "p95": np.percentile(g[col].dropna(), 95),
                             "p95_low": pl, "p95_high": ph})
    if "correct" in df:
        for (ctx, ds), g in df.dropna(subset=["correct"]).groupby(["context", "dataset"]):
            per_model = {m: item_accuracy(gm) for m, gm in g.groupby("model")}
            pairs = list(itertools.combinations(sorted(per_model), 2))
            stats = [mcnemar_exact(per_model[a], per_model[b_]) for a, b_ in pairs]
            adj = holm([s[2] for s in stats]) if stats else []
            for (a, b_), (n01, n10, p), pa in zip(pairs, stats, adj):
                pair_rows.append({"context": ctx, "dataset": ds, "model_a": a, "model_b": b_,
                                  "a_only_correct": n01, "b_only_correct": n10,
                                  "p_exact": p, "p_holm": pa})
    return {"accuracy": pd.DataFrame(acc_rows), "system": pd.DataFrame(sys_rows),
            "pairs": pd.DataFrame(pair_rows), "overlap": pd.DataFrame(overlap_rows),
            "manifest": manifest}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--samples", required=True, help="glob of sample-level CSV files")
    ap.add_argument("--out", default="derived_raw", help="output directory")
    ap.add_argument("--bootstrap", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=20260101)
    for k, v in DEFAULT_COLUMNS.items():
        ap.add_argument(f"--col-{k}", default=v, help=f"column holding '{k}' (default: {v})")
    a = ap.parse_args(argv)
    cols = {k: getattr(a, f"col_{k}") for k in DEFAULT_COLUMNS}
    res = analyse(load_samples(a.samples, cols), a.bootstrap, a.seed)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    for name in ("accuracy", "system", "pairs", "overlap"):
        res[name].to_csv(out / f"{name}.csv", index=False)
    (out / "run_manifest.json").write_text(json.dumps(res["manifest"], indent=1, default=str))
    print(f"wrote results to {out}/")
    return res


if __name__ == "__main__":
    main()
