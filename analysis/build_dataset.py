"""Assemble the aggregated per-configuration dataset used in the revision.

Precedence of sources (highest first), recorded per field in ``*_source``:
  1. ``tables``  -- LaTeX tables exported by the results pipeline (../tables/*.tex)
  2. ``figure``  -- values recovered from the pipeline's vector figures
                    (data/recovered_from_figures.csv, see recover_figure_values.py)
  3. ``thesis``  -- Tables 8.1 and 8.2 of the B.Sc. thesis (cached in
                    data/thesis_tables_8_1_8_2.csv; re-parsed if the PDF is present)

Output: data/aggregated_metrics.csv (one row per context x configuration x dataset).

The raw per-sample and per-run CSV files produced by the framework supersede
this file; ``run_level_uncertainty.py`` operates on those files directly.
"""
from __future__ import annotations

import csv
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent
TABLES = ROOT.parent / "tables"
DATA = ROOT / "data"
THESIS_PDF = ROOT.parent.parent / "Rapport final - Stephane Nascimento Santos.pdf"
THESIS_CACHE = DATA / "thesis_tables_8_1_8_2.csv"
RECOVERED = DATA / "recovered_from_figures.csv"
OUT = DATA / "aggregated_metrics.csv"

CTX = {"pi5": "Edge", "cpu": "Local CPU", "vllm": "GPU server"}
DS = {"mmlu": "MMLU", "gsm8k": "GSM8K", "hellaswag": "HellaSwag", "dolly": "Dolly"}

# Thesis identifiers -> display labels used by the pipeline tables.
LABELS = {
    "Llama-3.2-1B-Instruct-Q8_0": "Llama 3.2 1B (Q8)",
    "llama-3.2-1b-instruct-q4_k_m": "Llama 3.2 1B (Q4)",
    "Ministral-3-3B-Instruct-2512-Q4_K_M": "Ministral 3 3B (Q4)",
    "Phi-4-mini-instruct-Q4_K_M": "Phi-4 Mini 3.8B (Q4)",
    "gemma-3-1b-it-Q8_0": "Gemma 3 1B (Q8)",
    "SmolLM3-3B-Q4_K_M": "SmolLM3 3B (Q4)",
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

FIELDS = ["accuracy", "ttft_ms", "latency_ms", "tokens_per_s", "reported_tokens_per_s",
          "throughput_basis", "samples_per_s", "peak_ram_mb", "peak_ram_gib", "gpu_energy_j"]


def _clean(s: str) -> str:
    return s.replace(".gguf", "").replace(" ", "").replace("-in-struct", "-instruct") \
            .replace("in-struct", "instruct").replace("aper-tus", "apertus") \
            .replace("gran-ite", "granite").replace("kimi_lin-ear", "kimi_linear") \
            .replace("minis-tral", "ministral").replace("|", "")


def parse_thesis_pdf() -> list[dict]:
    import fitz  # PyMuPDF
    doc = fitz.open(THESIS_PDF)

    def flat(pages):
        toks = [t.strip() for i in pages for t in doc[i].get_text().split("\n") if t.strip()]
        return " ".join(toks).replace("- ", "-").replace("hel-laswag", "hellaswag")

    rows = []
    acc = re.compile(r"(cpu|pi5|vllm)\s+\|?\s*(\S+?(?:\s\S+?)?)\s+(gsm8k|hellaswag|mmlu)\s+(0?\.\d+|1\.0|0)\b")
    for m in acc.finditer(flat(range(40, 42))):
        rows.append({"platform": m[1], "model": _clean(m[2]), "dataset": m[3],
                     "accuracy": m[4]})
    tim = re.compile(r"(cpu|pi5|vllm)\s+(.+?)\s+(dolly|gsm8k|hellaswag|mmlu)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)")
    for m in tim.finditer(flat(range(42, 46))):
        rows.append({"platform": m[1], "model": _clean(m[2]), "dataset": m[3],
                     "ttft_ms": m[4], "latency_ms": m[5], "tokens_per_s": m[6]})
    return rows


def load_thesis() -> dict:
    if THESIS_PDF.exists():
        raw = parse_thesis_pdf()
        with THESIS_CACHE.open("w", newline="") as fh:
            w = csv.DictWriter(fh, ["platform", "model", "dataset", "accuracy", "ttft_ms",
                                    "latency_ms", "tokens_per_s"])
            w.writeheader()
            w.writerows(raw)
    out: dict = {}
    with THESIS_CACHE.open() as fh:
        for r in csv.DictReader(fh):
            key = (CTX[r["platform"]], LABELS[r["model"].replace(".gguf", "")], DS[r["dataset"]])
            for f in ("accuracy", "ttft_ms", "latency_ms", "tokens_per_s"):
                if r.get(f):
                    out.setdefault(key, {})[f] = float(r[f])
    return out


def _rows(tex: Path):
    for line in tex.read_text().splitlines():
        if "&" in line and "\\textbf" not in line:
            yield [c.strip().rstrip("\\").strip() for c in line.split("&")]


def _f(x: str):
    return None if x.strip() in {"--", "---", ""} else float(x)


def load_tables() -> dict:
    out: dict = {}

    def put(key, field, val):
        if val is not None:
            out.setdefault(key, {})[field] = val

    for c in _rows(TABLES / "table_04_task_accuracy_selected.tex"):
        for ds, v in zip(("MMLU", "GSM8K", "HellaSwag"), c[2:5]):
            put((c[0], c[1], ds), "accuracy", _f(v))
    for c in _rows(TABLES / "table_05_timing_throughput_selected.tex"):
        key = (c[0], c[1], c[2])
        put(key, "ttft_ms", _f(c[3]))
        put(key, "latency_ms", _f(c[4]))
        put(key, "reported_tokens_per_s", _f(c[5]))
        if c[0] != "GPU server":
            put(key, "tokens_per_s", _f(c[5]))
        put(key, "throughput_basis", c[6])
    for c in _rows(TABLES / "table_06_peak_resources_selected.tex"):
        put((c[0], c[1], c[2]), "peak_ram_mb", _f(c[4]))
    for c in _rows(TABLES / "table_07_cost_effectiveness_mmlu.tex"):
        key = (c[0], c[1], "MMLU")
        put(key, "latency_ms", _f(c[3]) * 1000)
        put(key, "samples_per_s", _f(c[4]))
    return out


def load_recovered(acc_lookup: dict) -> dict:
    """Assign recovered GSM8K latency markers to configurations by accuracy."""
    out: dict = {}
    lat_pts: dict = {}
    with RECOVERED.open() as fh:
        rows = list(csv.DictReader(fh))
    i = 0
    while i < len(rows):
        r = rows[i]
        if r["metric"] == "gsm8k_latency_s":
            lat_pts.setdefault(r["context"], []).append(
                (float(r["value"]), float(rows[i + 1]["value"])))
            i += 2
            continue
        if r["metric"].startswith("reported_tokens_per_s:"):
            ds = r["metric"].split(":")[1]
            basis = "Run-level sustained" if r["context"] == "GPU server" else "Active generation"
            out.setdefault((r["context"], r["model"], ds), {}).update(
                {"reported_tokens_per_s": float(r["value"]), "throughput_basis": basis})
            i += 1
            continue
        field = "peak_ram_gib" if r["metric"] == "peak_ram_gib" else "gpu_energy_j"
        out.setdefault((r["context"], r["model"], "GSM8K"), {})[field] = float(r["value"])
        i += 1
    # Match each marker to the configuration with the closest (accuracy, latency).
    for ctx, pts in lat_pts.items():
        cands = {k: v for k, v in acc_lookup.items() if k[0] == ctx and k[2] == "GSM8K"}
        for lat, acc in pts:
            def dist(item):
                k, v = item
                d = abs(v.get("accuracy", 9) - acc)
                if "latency_ms" in v:
                    d += abs(v["latency_ms"] / 1000 - lat) / max(lat, 1)
                return d
            key, _ = min(cands.items(), key=dist)
            out.setdefault(key, {})["latency_ms"] = lat * 1000
            cands.pop(key)
    return out


def cross_check(tables: dict, recovered: dict) -> None:
    """Report agreement between recovered figure values and exported tables."""
    errs = []
    for k, v in tables.items():
        r = recovered.get(k, {})
        for f in ("reported_tokens_per_s", "latency_ms"):
            if f in v and f in r and v[f]:
                errs.append(abs(r[f] / v[f] - 1))
        if "peak_ram_mb" in v and "peak_ram_gib" in r:
            errs.append(abs(r["peak_ram_gib"] * 1024 / v["peak_ram_mb"] - 1))
    if errs:
        print(f"cross-check: {len(errs)} overlapping values, max relative deviation "
              f"{100 * max(errs):.2f}%")


def main():
    thesis = load_thesis()
    tables = load_tables()
    acc_view = {k: dict(thesis.get(k, {}), **tables.get(k, {}))
                for k in set(thesis) | set(tables)}
    recovered = load_recovered(acc_view)
    cross_check(tables, recovered)
    keys = sorted(set(thesis) | set(tables) | set(recovered),
                  key=lambda k: (["Edge", "Local CPU", "GPU server"].index(k[0]), k[1], k[2]))
    with OUT.open("w", newline="") as fh:
        cols = ["context", "model", "dataset"]
        for f in FIELDS:
            cols += [f, f + "_source"]
        w = csv.DictWriter(fh, cols)
        w.writeheader()
        for k in keys:
            row = {"context": k[0], "model": k[1], "dataset": k[2]}
            for f in FIELDS:
                for name, src in (("tables", tables), ("figure", recovered), ("thesis", thesis)):
                    if f in src.get(k, {}):
                        row[f], row[f + "_source"] = src[k][f], name
                        break
            w.writerow(row)
    print(f"wrote {OUT} with {len(keys)} rows")


if __name__ == "__main__":
    main()
