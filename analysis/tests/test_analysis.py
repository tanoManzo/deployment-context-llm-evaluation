"""Unit tests for the revision analysis. Run: python -m pytest tests -q"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from stats_utils import holm, newcombe, wilson  # noqa: E402
import run_level_uncertainty as rlu  # noqa: E402


def test_wilson_reference_values():
    # Newcombe (1998), Table I: 81/263 -> 0.2553 to 0.3662
    lo, hi = wilson(81 / 263, 263)
    assert lo == pytest.approx(0.2553, abs=5e-4)
    assert hi == pytest.approx(0.3662, abs=5e-4)
    lo, hi = wilson(0.5, 100)
    assert (lo, hi) == pytest.approx((0.4038, 0.5962), abs=5e-4)


def test_newcombe_reference_values():
    # Newcombe (1998), method 10, example 56/70 vs 48/80 -> 0.0524 to 0.3339
    lo, hi = newcombe(56 / 70, 70, 48 / 80, 80)
    assert lo == pytest.approx(0.0524, abs=5e-4)
    assert hi == pytest.approx(0.3339, abs=5e-4)


def test_holm_monotone_and_bounded():
    adj = holm([0.01, 0.04, 0.03, 0.005])
    assert adj == pytest.approx([0.03, 0.06, 0.06, 0.02])
    assert all(0 <= p <= 1 for p in adj)


def _synthetic(tmp_path: Path) -> str:
    rng = np.random.default_rng(0)
    rows = []
    items = [f"q{i}" for i in range(100)]
    skill = {"A": 0.8, "B": 0.5}
    for model, p in skill.items():
        truth = rng.random(100) < p
        for run in range(10):
            order = rng.permutation(items)
            for it in order:
                k = items.index(it)
                rows.append({"platform": "gpu", "model": model, "dataset": "mmlu", "run": run,
                             "seed": run, "item_id": it, "correct": int(truth[k]),
                             "ttft_ms": 100 + rng.normal(0, 5),
                             "latency_ms": 250 + 10 * run + rng.normal(0, 20),
                             "tokens_per_sec": 9 + rng.normal(0, 0.3)})
    f = tmp_path / "samples.csv"
    pd.DataFrame(rows).to_csv(f, index=False)
    return str(f)


def test_raw_pipeline_on_synthetic_data(tmp_path):
    res = rlu.main(["--samples", _synthetic(tmp_path), "--out", str(tmp_path / "o"),
                    "--bootstrap", "200"])
    acc = res["accuracy"].set_index("model")
    assert (acc.n_items == 100).all()
    assert acc.run_sd.to_numpy() == pytest.approx(0.0, abs=1e-12)  # same items, deterministic
    assert acc.loc["A", "accuracy"] > acc.loc["B", "accuracy"]
    assert (res["overlap"].jaccard == 1.0).all()         # identical item sets per run
    pair = res["pairs"].iloc[0]
    assert pair.p_holm < 0.05
    lat = res["system"].query("metric == 'latency'").iloc[0]
    assert lat.boot_low < lat["mean"] < lat.boot_high
    assert lat.t_low < lat["mean"] < lat.t_high
    assert (tmp_path / "o" / "run_manifest.json").exists()
