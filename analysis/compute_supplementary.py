"""Run-level uncertainty and supplementary tables from the raw benchmark results.

Reads the per-run sample files of the benchmarking framework (repository
TB-2026-SLM-benchmarking, folder ``results/``) and produces:

* 95% intervals for mean TTFT, end-to-end latency, and per-request throughput
  (two-stage cluster bootstrap: runs, then requests within runs; resampling is
  stratified by measurement session and sessions are weighted equally, which is
  the estimator behind the published means);
* run-level 95% t-intervals for run-level sustained throughput and request rate
  (from the per-session mean and standard deviation over runs);
* 95% bootstrap intervals for mean GPU energy per request;
* run-to-run spread of accuracy;
* paired exact McNemar comparisons against the highest observed mean
  (sensitivity analysis for the unpaired comparisons in the article).

Every published mean is reproduced first and the maximum deviation is printed.

Usage:
    python compute_supplementary.py --repo /path/to/TB-2026-SLM-benchmarking
"""
from __future__ import annotations

import argparse
import glob
import itertools
import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest, t

from stats_utils import holm, wilson

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
SUPP = ROOT.parent / "supplementary"

CONTEXT = {"pi5": "Edge", "cpu": "Local CPU", "vllm": "GPU server"}
LABEL = {
    "gemma-3-1b-it-Q8_0": "Gemma 3 1B (Q8)",
    "llama-3.2-1b-instruct-q4_k_m": "Llama 3.2 1B (Q4)",
    "Llama-3.2-1B-Instruct-Q8_0": "Llama 3.2 1B (Q8)",
    "Phi-4-mini-instruct-Q4_K_M": "Phi-4 Mini 3.8B (Q4)",
    "SmolLM3-3B-Q4_K_M": "SmolLM3 3B (Q4)",
    "Ministral-3-3B-Instruct-2512-Q4_K_M": "Ministral 3 3B (Q4)",
    "apertus_70b_instruct_bf16": "Apertus 70B (bf16)",
    "granite4_htiny_bf16": "Granite 4 H-Tiny (bf16)",
    "kimi_linear_48b_bf16": "Kimi Linear 48B-A3B (bf16)",
    "llama3_2_1b_bf16": "Llama 3.2 1B (bf16)",
    "llama3_3_70b_fp8": "Llama 3.3 70B (FP8)",
    "ministral3_14b": "Ministral 3 14B",
    "ministral3_3b_awq": "Ministral 3 3B (AWQ)",
    "ministral3_8b_awq": "Ministral 3 8B (AWQ)",
    "phi4_14b_bf16": "Phi-4 14B (bf16)",
    "phi4mini_bf16": "Phi-4 Mini 3.8B (bf16)",
    "phi4mini_gptq_4bit": "Phi-4 Mini 3.8B (GPTQ-4)",
    "qwen2_5_1_5b_bf16": "Qwen 2.5 1.5B (bf16)",
    "qwen2_5_7b_awq": "Qwen 2.5 7B (AWQ)",
    "qwen3_30b_instruct_bf16": "Qwen 3 30B-A3B (bf16)",
    "qwen3_4b_awq": "Qwen 3 4B (AWQ)",
}
DATASET = {"mmlu": "MMLU", "gsm8k": "GSM8K", "hellaswag": "HellaSwag", "dolly": "Dolly"}
CTX_ORDER = ["Edge", "Local CPU", "GPU server"]
DS_ORDER = ["MMLU", "GSM8K", "HellaSwag", "Dolly"]
B = 2000
RNG = np.random.default_rng(20261020)


def is_lfs(path: str) -> bool:
    with open(path) as fh:
        return fh.readline().startswith("version https://git-lfs")


def load_samples(repo: Path) -> pd.DataFrame:
    cols = ["run_idx", "sample_idx", "question", "choices", "correct", "ttft_ms",
            "end_to_end_latency_ms", "tokens_per_sec", "output_tokens"]
    frames = []
    for f in sorted(glob.glob(str(repo / "results" / "*" / "*" / "*" / "*_samples.csv"))):
        ctx, model, session, fname = Path(f).parts[-4:]
        if ctx not in CONTEXT or model not in LABEL or is_lfs(f):
            continue
        d = pd.read_csv(f, usecols=lambda c: c in cols, low_memory=False)
        d["context"], d["model"] = CONTEXT[ctx], LABEL[model]
        d["dataset"], d["session"] = DATASET[fname.replace("_samples.csv", "")], session
        frames.append(d)
    return pd.concat(frames, ignore_index=True)


def load_session_summaries(repo: Path) -> pd.DataFrame:
    rows = []
    for f in sorted(glob.glob(str(repo / "results" / "*" / "*" / "*" / "*.csv"))):
        ctx, model, session, fname = Path(f).parts[-4:]
        ds = fname.replace(".csv", "")
        if ctx not in CONTEXT or model not in LABEL or ds not in DATASET or is_lfs(f):
            continue
        s = pd.read_csv(f).iloc[0]
        n_runs = len(str(s.get("All served model ids", "")).split("|"))
        rows.append({"context": CONTEXT[ctx], "model": LABEL[model], "dataset": DATASET[ds],
                     "session": session, "n_runs": n_runs,
                     "sps_mean": s.get("Sequential sustained samples/sec mean"),
                     "sps_sd": s.get("Sequential sustained samples/sec sd"),
                     "stps_mean": s.get("Sequential sustained tokens/sec mean"),
                     "stps_sd": s.get("Sequential sustained tokens/sec sd")})
    return pd.DataFrame(rows)


def session_weighted_mean(g: pd.DataFrame, col: str) -> float:
    return float(np.mean([s[col].mean() for _, s in g.groupby("session")]))


def bootstrap_ci(g: pd.DataFrame, col: str) -> tuple[float, float]:
    """Two-stage cluster bootstrap stratified by session (equal session weights)."""
    reps = np.zeros(B)
    sessions = list(g.groupby("session"))
    for _, s in sessions:
        runs = [r[col].to_numpy(dtype=float) for _, r in s.groupby("run_idx")]
        n = max(len(r) for r in runs)
        X = np.full((len(runs), n), np.nan)
        for i, r in enumerate(runs):
            X[i, :len(r)] = r
        counts = np.array([len(r) for r in runs])
        ir = RNG.integers(0, len(runs), (B, len(runs)))
        iq = (RNG.random((B, len(runs), n)) * counts[ir][..., None]).astype(int)
        vals = X[ir[..., None], iq]
        reps += np.nanmean(vals, axis=(1, 2))
    reps /= len(sessions)
    return tuple(np.percentile(reps, [2.5, 97.5]))


def t_interval_sessions(g: pd.DataFrame, mean_col: str, sd_col: str):
    k = len(g)
    m = g[mean_col].mean()
    var = (g[sd_col] ** 2 / g.n_runs).sum() / k ** 2
    df = max(int(g.n_runs.sum()) - k, 1)
    half = t.ppf(0.975, df) * math.sqrt(var)
    return m, m - half, m + half


def timing_intervals(samples: pd.DataFrame, summaries: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (ctx, model, ds), g in samples.groupby(["context", "model", "dataset"]):
        rec = {"context": ctx, "model": model, "dataset": ds,
               "sessions": g.session.nunique(),
               "runs": int(g.groupby("session").run_idx.nunique().sum())}
        for col, key in (("ttft_ms", "ttft"), ("end_to_end_latency_ms", "latency"),
                         ("tokens_per_sec", "tps"), ("output_tokens", "words")):
            rec[key] = session_weighted_mean(g, col)
            rec[key + "_lo"], rec[key + "_hi"] = bootstrap_ci(g, col)
        s = summaries[(summaries.context == ctx) & (summaries.model == model)
                      & (summaries.dataset == ds)]
        if len(s):
            rec["sps"], rec["sps_lo"], rec["sps_hi"] = t_interval_sessions(s, "sps_mean", "sps_sd")
            rec["stps"], rec["stps_lo"], rec["stps_hi"] = t_interval_sessions(s, "stps_mean", "stps_sd")
        if g.correct.notna().any():
            per_run = g.groupby(["session", "run_idx"]).correct.mean()
            rec["accuracy"] = session_weighted_mean(g, "correct")
            rec["acc_run_min"], rec["acc_run_max"] = per_run.min(), per_run.max()
            rec["acc_run_sd"] = per_run.std(ddof=1) if len(per_run) > 1 else 0.0
        rows.append(rec)
    return pd.DataFrame(rows)


def validate(iv: pd.DataFrame) -> None:
    pub = pd.read_csv(DATA / "aggregated_metrics.csv")
    worst = {}
    for _, p in pub.iterrows():
        r = iv[(iv.context == p.context) & (iv.model == p.model) & (iv.dataset == p.dataset)]
        if r.empty:
            continue
        r = r.iloc[0]
        for col, pub_col, src in (("latency", "latency_ms", "latency_ms_source"),
                                  ("ttft", "ttft_ms", "ttft_ms_source"),
                                  ("accuracy", "accuracy", "accuracy_source")):
            if p.get(src) in ("tables", "figure") and not pd.isna(p[pub_col]) and col in r:
                dev = abs(r[col] / p[pub_col] - 1) if p[pub_col] else 0
                worst[col] = max(worst.get(col, 0), dev)
    print("reproduction of published means (max relative deviation): "
          + ", ".join(f"{k} {100 * v:.2f}%" for k, v in worst.items()))


def energy_intervals(repo: Path) -> pd.DataFrame:
    ev = repo / "results" / "manuscript_evidence_pack_v53" / "tables"
    frames = [pd.read_csv(f) for f in sorted(ev.glob("figure_03b*_measurement_values.csv"))]
    e = pd.concat(frames, ignore_index=True)
    rows = []
    for (model, ds), g in e.groupby(["model_configuration", "dataset"]):
        x = g.energy_j.to_numpy(dtype=float)
        reps = RNG.choice(x, (B, len(x))).mean(axis=1)
        rows.append({"context": "GPU server", "model": model, "dataset": DATASET[ds],
                     "requests": len(x), "energy_j": x.mean(),
                     "energy_lo": np.percentile(reps, 2.5), "energy_hi": np.percentile(reps, 97.5)})
    return pd.DataFrame(rows)


def mcnemar(samples: pd.DataFrame) -> pd.DataFrame:
    s = samples[samples.dataset.isin(["MMLU", "GSM8K", "HellaSwag"]) & samples.correct.notna()].copy()
    s["item"] = s.question.astype(str) + "||" + s.choices.astype(str)
    s["correct"] = s.correct.astype(float)
    rows = []
    for (ctx, ds), g in s.groupby(["context", "dataset"]):
        item_acc = {m: gm.groupby("item").correct.mean() for m, gm in g.groupby("model")}
        acc = {m: v.mean() for m, v in item_acc.items()}
        leader = max(acc, key=acc.get)
        others = [m for m in acc if m != leader]
        stats = []
        for m in others:
            a, b = item_acc[leader], item_acc[m]
            common = a.index.intersection(b.index)
            a, b = (a.loc[common] >= 0.5).astype(int), (b.loc[common] >= 0.5).astype(int)
            n10 = int(((a == 1) & (b == 0)).sum())
            n01 = int(((a == 0) & (b == 1)).sum())
            p = 1.0 if n10 + n01 == 0 else binomtest(n10, n10 + n01, 0.5).pvalue
            stats.append((m, len(common), n10, n01, p))
        adj = holm([x[4] for x in stats])
        for (m, n, n10, n01, p), pa in zip(stats, adj):
            rows.append({"context": ctx, "dataset": ds, "leader": leader, "model": m,
                         "items": n, "leader_only": n10, "model_only": n01,
                         "p_exact": p, "p_holm": pa, "resolved": pa < 0.05})
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ LaTeX output
def tex(s: str) -> str:
    return s.replace("_", r"\_").replace("%", r"\%").replace("&", r"\&")


def fmt_ci(m, lo, hi, nd):
    if m is None or (isinstance(m, float) and math.isnan(m)):
        return "--"
    return f"{m:.{nd}f} [{lo:.{nd}f}, {hi:.{nd}f}]"


def fp(p: float) -> str:
    return r"$<$0.001" if p < 0.001 else f"{p:.3f}"


def write_supplementary(iv, en, mc):
    SUPP.mkdir(exist_ok=True)
    key = lambda df: df.assign(c=df.context.map(CTX_ORDER.index), d=df.dataset.map(DS_ORDER.index)) \
        .sort_values(["c", "model", "d"])
    iv = key(iv)
    lines = [r"\begin{longtable}{@{}l l l c r r r r@{}}",
             r"\caption{Mean time to first token (TTFT), end-to-end latency, and per-request "
             r"throughput with 95\% intervals (two-stage cluster bootstrap over runs and requests, "
             r"stratified by measurement session, sessions weighted equally). Throughput is "
             r"generated words per second of the decoding interval. S/R: number of measurement "
             r"sessions and measured runs pooled.}\label{tab:s1}\\",
             r"\toprule", r"Context & Configuration & Workload & S/R & TTFT (ms) & Latency (ms) & "
             r"Throughput (words/s) & Words \\", r"\midrule\endfirsthead",
             r"\toprule", r"Context & Configuration & Workload & S/R & TTFT (ms) & Latency (ms) & "
             r"Throughput (words/s) & Words \\", r"\midrule\endhead", r"\bottomrule\endfoot"]
    prev = None
    for _, r in iv.iterrows():
        if prev and prev != r.context:
            lines.append(r"\midrule")
        prev = r.context
        lines.append(f"{r.context} & {tex(r.model)} & {r.dataset} & {r.sessions}/{r.runs} & "
                     f"{fmt_ci(r.ttft, r.ttft_lo, r.ttft_hi, 0)} & "
                     f"{fmt_ci(r.latency, r.latency_lo, r.latency_hi, 0)} & "
                     f"{fmt_ci(r.tps, r.tps_lo, r.tps_hi, 2)} & {r.words:.1f} \\\\")
    lines.append(r"\end{longtable}")
    s1 = "\n".join(lines)

    lines = [r"\begin{longtable}{@{}l l l r r@{}}",
             r"\caption{Run-level request rate and sustained throughput (mean over runs with 95\% "
             r"$t$-intervals; sessions weighted equally). Both include all inter-request overheads "
             r"and, on the Raspberry Pi~5, the cooling pause (1\,s for MMLU and HellaSwag, 2\,s "
             r"for GSM8K and Dolly). One sequential request stream.}\label{tab:s2}\\",
             r"\toprule", r"Context & Configuration & Workload & Requests/s & Sustained words/s \\",
             r"\midrule\endfirsthead", r"\toprule",
             r"Context & Configuration & Workload & Requests/s & Sustained words/s \\",
             r"\midrule\endhead", r"\bottomrule\endfoot"]
    prev = None
    for _, r in iv.iterrows():
        if pd.isna(r.get("sps")):
            continue
        if prev and prev != r.context:
            lines.append(r"\midrule")
        prev = r.context
        lines.append(f"{r.context} & {tex(r.model)} & {r.dataset} & "
                     f"{fmt_ci(r.sps, r.sps_lo, r.sps_hi, 3)} & {fmt_ci(r.stps, r.stps_lo, r.stps_hi, 2)} \\\\")
    lines.append(r"\end{longtable}")
    s2 = "\n".join(lines)

    en = key(en)
    lines = [r"\begin{longtable}{@{}l l r r@{}}",
             r"\caption{Mean GPU energy per request on the GPU server with 95\% bootstrap intervals "
             r"over requests. Energy is the trapezoidal integral of the board power of the first GPU "
             r"reported by \texttt{nvidia-smi} over the request; all models were served with tensor "
             r"parallelism across both GPUs, so values cover one of the two GPUs.}\label{tab:s3}\\",
             r"\toprule", r"Configuration & Workload & Requests & Energy (J) \\",
             r"\midrule\endfirsthead", r"\toprule", r"Configuration & Workload & Requests & Energy (J) \\",
             r"\midrule\endhead", r"\bottomrule\endfoot"]
    for _, r in en.iterrows():
        lines.append(f"{tex(r.model)} & {r.dataset} & {r.requests} & "
                     f"{fmt_ci(r.energy_j, r.energy_lo, r.energy_hi, 0)} \\\\")
    lines.append(r"\end{longtable}")
    s3 = "\n".join(lines)

    acc = iv[iv.dataset != "Dolly"]
    lines = [r"\begin{longtable}{@{}l l l r r r r@{}}",
             r"\caption{Run-to-run spread of accuracy. Accuracy is the session-weighted mean; the "
             r"spread is computed over all measured runs. Because the same 100 items are used in "
             r"every run with greedy decoding, the spread reflects execution non-determinism, not "
             r"item sampling; item-level uncertainty is given by the Wilson intervals in the "
             r"article.}\label{tab:s4}\\",
             r"\toprule", r"Context & Configuration & Workload & Accuracy & Min & Max & SD \\",
             r"\midrule\endfirsthead", r"\toprule",
             r"Context & Configuration & Workload & Accuracy & Min & Max & SD \\",
             r"\midrule\endhead", r"\bottomrule\endfoot"]
    prev = None
    for _, r in acc.iterrows():
        if prev and prev != r.context:
            lines.append(r"\midrule")
        prev = r.context
        lines.append(f"{r.context} & {tex(r.model)} & {r.dataset} & {r.accuracy:.3f} & "
                     f"{r.acc_run_min:.2f} & {r.acc_run_max:.2f} & {r.acc_run_sd:.3f} \\\\")
    lines.append(r"\end{longtable}")
    s4 = "\n".join(lines)

    mc = mc.assign(c=mc.context.map(CTX_ORDER.index), d=mc.dataset.map(DS_ORDER.index)) \
        .sort_values(["c", "d", "p_holm"])
    lines = [r"\begin{longtable}{@{}l l l l r r r c@{}}",
             r"\caption{Paired sensitivity analysis: exact McNemar tests of the configuration with "
             r"the highest observed accuracy against every other configuration in the same context "
             r"and benchmark, on the identical 100 items (item correct if correct in at least half "
             r"of its runs), Holm-adjusted within each cell. L/O: items answered correctly only by "
             r"the leader / only by the other configuration.}\label{tab:s5}\\",
             r"\toprule", r"Context & Bench. & Leader & Compared configuration & L/O & $p$ & "
             r"$p_{\mathrm{Holm}}$ & Resolved \\", r"\midrule\endfirsthead", r"\toprule",
             r"Context & Bench. & Leader & Compared configuration & L/O & $p$ & "
             r"$p_{\mathrm{Holm}}$ & Resolved \\", r"\midrule\endhead", r"\bottomrule\endfoot"]
    prev = None
    for _, r in mc.iterrows():
        cell = (r.context, r.dataset)
        if prev and prev != cell:
            lines.append(r"\midrule")
        prev = cell
        lines.append(f"{r.context} & {r.dataset} & {tex(r.leader)} & {tex(r.model)} & "
                     f"{r.leader_only}/{r.model_only} & {fp(r.p_exact)} & {fp(r.p_holm)} & "
                     f"{'yes' if r.resolved else 'no'} \\\\")
    lines.append(r"\end{longtable}")
    s5 = "\n".join(lines)
    for name, body in (("s1_timing", s1), ("s2_rates", s2), ("s3_energy", s3),
                       ("s4_accuracy_spread", s4), ("s5_mcnemar", s5)):
        (SUPP / f"{name}.tex").write_text(body + "\n")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--repo", required=True, help="path to the TB-2026-SLM-benchmarking repository")
    a = ap.parse_args(argv)
    repo = Path(a.repo)
    samples = load_samples(repo)
    summaries = load_session_summaries(repo)
    iv = timing_intervals(samples, summaries)
    validate(iv)
    en = energy_intervals(repo)
    mc = mcnemar(samples)
    iv.to_csv(DATA / "derived_run_level_intervals.csv", index=False)
    en.to_csv(DATA / "derived_energy_intervals.csv", index=False)
    mc.to_csv(DATA / "derived_mcnemar.csv", index=False)
    write_supplementary(iv, en, mc)
    print(f"{len(iv)} configuration-workload rows, {len(en)} energy rows, {len(mc)} paired tests")


if __name__ == "__main__":
    main()
