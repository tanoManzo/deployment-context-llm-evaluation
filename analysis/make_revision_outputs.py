"""Generate every derived table, figure, and number used in the revised manuscript.

Input : data/aggregated_metrics.csv   (built by build_dataset.py)
Output: ../figures/rev_*.pdf, ../tables/rev_*.tex, data/derived_*.csv

Run   : python make_revision_outputs.py

Design facts used here (Section III of the manuscript):
* every benchmark uses one fixed subset of N_ITEMS = 100 questions; the ten
  runs present the same questions in a run-specific order, and all models
  receive the same questions. Item-level sampling uncertainty therefore has an
  effective sample size of 100 per configuration and benchmark.
"""
from __future__ import annotations

import csv
import math
from dataclasses import dataclass, field
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

from stats_utils import holm, newcombe, two_prop_p, wilson  # noqa: E402

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
FIG = ROOT.parent / "figures"
TAB = ROOT.parent / "tables"

N_ITEMS = 100
ALPHA = 0.05
TAU = 0.10          # practical-significance tolerance for latency / cost
PI_PAUSE_S = 1.0    # inter-request cooling pause on the Raspberry Pi 5
CONTEXTS = ["Edge", "Local CPU", "GPU server"]
COLORS = {"Edge": "#E69F00", "Local CPU": "#0072B2", "GPU server": "#009E73"}
DATASETS = ["MMLU", "GSM8K", "HellaSwag"]

# Values printed in Table 7 of the submitted manuscript ("Acc/s"), kept only to
# document the audit requested by Reviewer 1.
SUBMITTED_TABLE7 = {
    ("Edge", "Llama 3.2 1B (Q4)"): (0.395, 12.70, 0.300),
    ("Edge", "Gemma 3 1B (Q8)"): (0.315, 5.41, 0.262),
    ("Edge", "SmolLM3 3B (Q4)"): (0.520, 4.64, 0.120),
    ("Edge", "Phi-4 Mini 3.8B (Q4)"): (0.564, 3.55, 0.116),
    ("Local CPU", "Llama 3.2 1B (Q8)"): (0.380, 18.14, 0.288),
    ("Local CPU", "Gemma 3 1B (Q8)"): (0.310, 13.57, 0.419),
    ("Local CPU", "Phi-4 Mini 3.8B (Q4)"): (0.540, 9.76, 0.199),
    ("Local CPU", "Ministral 3 3B (Q4)"): (0.620, 8.26, 0.139),
    ("GPU server", "Llama 3.2 1B (bf16)"): (0.430, 9.08, 1.800),
    ("GPU server", "Qwen 2.5 1.5B (bf16)"): (0.520, 9.23, 2.147),
    ("GPU server", "Ministral 3 3B (AWQ)"): (0.640, 9.02, 2.654),
    ("GPU server", "Phi-4 Mini 3.8B (bf16)"): (0.630, 23.89, 2.508),
    ("GPU server", "Ministral 3 8B (AWQ)"): (0.660, 9.04, 2.707),
    ("GPU server", "Ministral 3 14B"): (0.700, 8.62, 2.555),
    ("GPU server", "Phi-4 14B (bf16)"): (0.820, 11.41, 3.149),
    ("GPU server", "Qwen 3 30B-A3B (bf16)"): (0.816, 8.83, 3.081),
    ("GPU server", "Llama 3.3 70B (FP8)"): (0.802, 8.70, 2.735),
}


@dataclass
class Scenario:
    """Illustrative deployment requirements (Section V-E of the manuscript)."""
    name: str
    context: str
    request: str
    max_mem: float | None        # GiB of peak system RAM (Edge / Local CPU)
    max_latency_s: float
    min_accuracy: float
    cost_field: str              # third objective
    cost_label: str
    notes: list[str] = field(default_factory=list)


SCENARIOS = [
    Scenario("E1", "Edge", "on-device math helper", 6.0, 30.0, 0.45,
             "peak_ram_gib", "RAM (GiB)"),
    Scenario("E2", "Edge", "offline batch grading", 7.5, 45.0, 0.45,
             "peak_ram_gib", "RAM (GiB)"),
    Scenario("C1", "Local CPU", "desktop assistant", 16.0, 20.0, 0.50,
             "peak_ram_gib", "RAM (GiB)"),
    Scenario("C2", "Local CPU", "tight-latency assistant", 16.0, 12.0, 0.45,
             "peak_ram_gib", "RAM (GiB)"),
    Scenario("G1", "GPU server", "single-user analyst assistant", None, 5.0, 0.60,
             "gpu_energy_j", "GPU energy (J)"),
    Scenario("G2", "GPU server", "accuracy-oriented single-user assistant", None, 8.0, 0.60,
             "gpu_energy_j", "GPU energy (J)"),
]


# --------------------------------------------------------------------------- utils
def load() -> pd.DataFrame:
    df = pd.read_csv(DATA / "aggregated_metrics.csv")
    df["latency_s"] = df["latency_ms"] / 1000.0
    return df


def fmt(x, nd=3):
    return "--" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{x:.{nd}f}"


def tex_escape(s: str) -> str:
    return s.replace("_", r"\_").replace("%", r"\%").replace("&", r"\&")


def write_tex(name: str, body: str) -> None:
    (TAB / name).write_text(body)
    print("wrote", TAB / name)


def setup_mpl():
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "Nimbus Roman", "DejaVu Serif"],
        "font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 8,
        "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7,
        "pdf.fonttype": 42, "ps.fonttype": 42, "axes.spines.top": False,
        "axes.spines.right": False, "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
    })


# ------------------------------------------------------------- 1. accuracy + CIs
def accuracy_table(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, r in df[df.dataset.isin(DATASETS) & df.accuracy.notna()].iterrows():
        lo, hi = wilson(r.accuracy, N_ITEMS)
        rows.append({"context": r.context, "model": r.model, "dataset": r.dataset,
                     "accuracy": r.accuracy, "ci_low": lo, "ci_high": hi})
    acc = pd.DataFrame(rows)
    # Leader set: configurations not distinguishable from the highest observed
    # mean within (context, dataset); Holm-adjusted two-proportion tests.
    acc["leader"] = False
    acc["indistinguishable"] = False
    acc["p_holm_vs_leader"] = float("nan")
    for (ctx, ds), g in acc.groupby(["context", "dataset"]):
        top = g.accuracy.max()
        lead_idx = g.index[g.accuracy == top]
        acc.loc[lead_idx, "leader"] = True
        others = g.drop(lead_idx)
        ps = [two_prop_p(top, N_ITEMS, a, N_ITEMS) for a in others.accuracy]
        adj = holm(ps) if ps else []
        for idx, p in zip(others.index, adj):
            acc.loc[idx, "p_holm_vs_leader"] = p
            acc.loc[idx, "indistinguishable"] = p >= ALPHA
    acc.to_csv(DATA / "derived_accuracy_ci.csv", index=False)
    return acc


def accuracy_table_tex(acc: pd.DataFrame) -> None:
    lines = [r"\begin{table*}[!t]", r"\revstart\centering",
             r"\caption{Task accuracy with 95\% Wilson score intervals ($n=100$ distinct items per "
             r"benchmark; identical items for all configurations). Bold marks the highest observed "
             r"mean within a context and benchmark; $^{\dagger}$ marks configurations whose difference "
             r"from that mean is not statistically resolved (Holm-adjusted two-proportion tests, "
             r"$\alpha=0.05$). Entries marked $^{\dagger}$ should be read as tied with the bold entry, "
             r"not as ranked below it.}",
             r"\label{tab:accuracy-ci}", r"\footnotesize", r"\setlength{\tabcolsep}{3.5pt}",
             r"\begin{tabularx}{\textwidth}{@{}l L c c c@{}}", r"\toprule",
             r"\textbf{Context} & \textbf{Configuration} & \textbf{MMLU} & \textbf{GSM8K} & "
             r"\textbf{HellaSwag} \\", r"\midrule"]
    for ci, ctx in enumerate(CONTEXTS):
        if ci:
            lines.append(r"\midrule")
        sub = acc[acc.context == ctx]
        for model in sorted(sub.model.unique()):
            cells = []
            for ds in DATASETS:
                r = sub[(sub.model == model) & (sub.dataset == ds)]
                if r.empty:
                    cells.append("--")
                    continue
                r = r.iloc[0]
                val = f"{r.accuracy:.3f}"
                if r.leader:
                    val = r"\textbf{" + val + "}"
                elif r.indistinguishable:
                    val += r"$^{\dagger}$"
                cells.append(f"{val} [{r.ci_low:.2f}, {r.ci_high:.2f}]")
            lines.append(f"{ctx} & {tex_escape(model)} & " + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabularx}", r"\end{table*}", ""]
    write_tex("rev_table_accuracy_ci.tex", "\n".join(lines))


def accuracy_figure(acc: pd.DataFrame) -> None:
    fig, axes = plt.subplots(3, 1, figsize=(7.16, 6.4), sharex=True)
    order = []
    for ctx in CONTEXTS:
        order += [(ctx, m) for m in sorted(acc[acc.context == ctx].model.unique())]
    xpos, x, ticks = {}, 0.0, []
    for i, key in enumerate(order):
        if i and key[0] != order[i - 1][0]:
            x += 0.9
        xpos[key] = x
        ticks.append(x)
        x += 1
    for ax, ds in zip(axes, DATASETS):
        sub = acc[acc.dataset == ds]
        for _, r in sub.iterrows():
            xx = xpos[(r.context, r.model)]
            ax.bar(xx, r.accuracy, width=0.72, color=COLORS[r.context], alpha=0.85,
                   edgecolor="black", linewidth=0.4)
            ax.errorbar(xx, r.accuracy, yerr=[[r.accuracy - r.ci_low], [r.ci_high - r.accuracy]],
                        fmt="none", ecolor="black", elinewidth=0.7, capsize=1.8)
        ax.set_ylim(0, 1)
        if ds in ("MMLU", "HellaSwag"):
            ax.axhline(0.25, color="0.35", linestyle="--", linewidth=0.6)
        ax.set_ylabel(f"{ds} accuracy")
        ax.grid(axis="y", linewidth=0.3, alpha=0.6)
        ax.set_axisbelow(True)
        for ctx in CONTEXTS[1:]:
            first = min(v for k, v in xpos.items() if k[0] == ctx)
            ax.axvline(first - 0.95, color="0.5", linewidth=0.5, linestyle=":")
    axes[-1].set_xticks(ticks)
    axes[-1].set_xticklabels([m for _, m in order], rotation=65, ha="right")
    handles = [plt.Rectangle((0, 0), 1, 1, color=COLORS[c], alpha=0.85) for c in CONTEXTS]
    handles.append(plt.Line2D([], [], color="0.35", linestyle="--", linewidth=0.6))
    axes[0].legend(handles, CONTEXTS + ["4-way chance (MMLU, HellaSwag)"], ncol=4,
                   loc="upper left", frameon=False)
    fig.align_ylabels(axes)
    fig.savefig(FIG / "rev_fig_accuracy_ci.pdf")
    plt.close(fig)
    print("wrote", FIG / "rev_fig_accuracy_ci.pdf")


def leader_table_tex(acc: pd.DataFrame) -> None:
    lines = [r"\begin{table}[!t]", r"\revstart\centering",
             r"\caption{Highest observed mean accuracy per context and benchmark, and the "
             r"configurations that are not statistically distinguishable from it (Holm-adjusted "
             r"two-proportion tests over the comparisons within each cell, $\alpha=0.05$, $n=100$).}",
             r"\label{tab:leaders}", r"\footnotesize", r"\setlength{\tabcolsep}{3pt}",
             r"\begin{tabularx}{\columnwidth}{@{}l l L@{}}", r"\toprule",
             r"\textbf{Context} & \textbf{Bench.} & \textbf{Highest mean; not distinguishable from it} \\",
             r"\midrule"]
    for ctx in CONTEXTS:
        for ds in DATASETS:
            g = acc[(acc.context == ctx) & (acc.dataset == ds)]
            if g.empty:
                continue
            lead = g[g.leader]
            tied = g[g.indistinguishable].sort_values("accuracy", ascending=False)
            lead_s = "; ".join(f"{tex_escape(m)} ({a:.3f})" for m, a in zip(lead.model, lead.accuracy))
            tied_s = ", ".join(tex_escape(m) for m in tied.model) or "none"
            lines.append(f"{ctx} & {ds} & \\textbf{{{lead_s}}}; {tied_s} \\\\")
        if ctx != CONTEXTS[-1]:
            lines.append(r"\midrule")
    lines += [r"\bottomrule", r"\end{tabularx}", r"\end{table}", ""]
    write_tex("rev_table_leaders.tex", "\n".join(lines))


# ------------------------------------------------------- 2. Pareto + decisions
def resolved_acc(a1: float, a2: float) -> bool:
    lo, hi = newcombe(a1, N_ITEMS, a2, N_ITEMS)
    return lo > 0 or hi < 0


def dominates(x, y, cost) -> bool:
    ge = x.accuracy >= y.accuracy and x.latency_s <= y.latency_s and x[cost] <= y[cost]
    gt = x.accuracy > y.accuracy or x.latency_s < y.latency_s or x[cost] < y[cost]
    return ge and gt


def confidently_dominates(x, y, cost) -> bool:
    """Dominance whose improvement exceeds measurement uncertainty or tolerance."""
    if not dominates(x, y, cost):
        return False
    return (resolved_acc(x.accuracy, y.accuracy)
            or y.latency_s >= (1 + TAU) * x.latency_s
            or y[cost] >= (1 + TAU) * x[cost])


def pareto(cands: pd.DataFrame, cost: str):
    point, robust = [], []
    for i, y in cands.iterrows():
        if not any(dominates(x, y, cost) for j, x in cands.iterrows() if j != i):
            point.append(i)
        if not any(confidently_dominates(x, y, cost) for j, x in cands.iterrows() if j != i):
            robust.append(i)
    return point, robust


def select(front: pd.DataFrame, cost: str):
    """Algorithm 1, steps 3-4: tie-aware selection on the feasible frontier."""
    if front.empty:
        return None, None, []
    point_choice = front.sort_values(["accuracy", "latency_s"], ascending=[False, True]).index[0]
    top = front.accuracy.max()
    leader = front[front.accuracy == top]
    others = front.drop(leader.index)
    ps = holm([two_prop_p(top, N_ITEMS, a, N_ITEMS) for a in others.accuracy]) if len(others) else []
    tied = list(leader.index) + [i for i, p in zip(others.index, ps) if p >= ALPHA]
    t = front.loc[tied]
    fastest = t.latency_s.min()
    near = t[t.latency_s <= (1 + TAU) * fastest]
    tie_choice = near.sort_values([cost, "accuracy"], ascending=[True, False]).index[0]
    return point_choice, tie_choice, tied


def gsm8k_view(df: pd.DataFrame) -> pd.DataFrame:
    g = df[df.dataset == "GSM8K"].copy()
    return g.set_index(g.context + " | " + g.model)


def run_decisions(df: pd.DataFrame):
    g = gsm8k_view(df)
    results, frontier_rows = [], []
    for ctx in CONTEXTS:
        cost = "gpu_energy_j" if ctx == "GPU server" else "peak_ram_gib"
        c = g[g.context == ctx]
        point, robust = pareto(c, cost)
        for i, r in c.iterrows():
            frontier_rows.append({"context": ctx, "model": r.model, "accuracy": r.accuracy,
                                  "latency_s": r.latency_s, "cost_field": cost, "cost": r[cost],
                                  "point_frontier": i in point,
                                  "uncertainty_aware_frontier": i in robust})
    pd.DataFrame(frontier_rows).to_csv(DATA / "derived_gsm8k_frontiers.csv", index=False)

    for sc in SCENARIOS:
        c = g[g.context == sc.context].copy()
        reasons = {}
        for i, r in c.iterrows():
            # Sequential filter (memory -> latency -> accuracy); each configuration
            # is reported under the first hard constraint it violates.
            why = []
            if sc.max_mem is not None and r[sc.cost_field] > sc.max_mem:
                why.append(f"RAM {r[sc.cost_field]:.1f}\\,GiB")
            elif r.latency_s > sc.max_latency_s:
                why.append(f"latency {r.latency_s:.1f}\\,s")
            elif r.accuracy < sc.min_accuracy:
                why.append(f"accuracy {r.accuracy:.3f}")
            reasons[i] = why
        feas = c[[not reasons[i] for i in c.index]]
        point, robust = pareto(feas, sc.cost_field) if len(feas) else ([], [])
        front = feas.loc[robust]
        point_choice, tie_choice, tied = select(front, sc.cost_field)
        results.append({"scenario": sc, "data": c, "reasons": reasons, "feasible": list(feas.index),
                        "point_front": point, "robust_front": robust,
                        "point_choice": point_choice, "tie_choice": tie_choice, "tied": tied})
    with (DATA / "derived_decisions.csv").open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["scenario", "context", "max_mem_gib", "max_latency_s", "min_accuracy",
                    "excluded", "feasible", "uncertainty_aware_frontier", "accuracy_tied_set",
                    "selected_point_estimate", "selected_tie_aware"])
        for res in results:
            sc = res["scenario"]
            name = lambda i: res["data"].loc[i, "model"]  # noqa: E731
            w.writerow([sc.name, sc.context, sc.max_mem, sc.max_latency_s, sc.min_accuracy,
                        "; ".join(f"{name(i)} ({', '.join(v)})" for i, v in res["reasons"].items() if v),
                        "; ".join(map(name, res["feasible"])),
                        "; ".join(map(name, res["robust_front"])),
                        "; ".join(map(name, res["tied"])),
                        name(res["point_choice"]) if res["point_choice"] else "none",
                        name(res["tie_choice"]) if res["tie_choice"] else "none"])
    return results


def decision_table_tex(results) -> None:
    lines = [r"\begin{table*}[!t]", r"\revstart\centering",
             r"\caption{Worked deployment decisions on GSM8K-type requests (multi-step arithmetic word "
             r"problems, one sequential request stream). Thresholds are illustrative and are not taken "
             r"from a documented application. Step~1 removes configurations that violate a hard "
             r"constraint; Step~2 retains the uncertainty-aware Pareto set of the feasible "
             r"configurations; Step~3 applies the tie-aware rule of Algorithm~\ref{alg:selection}. "
             r"The point-estimate choice (highest observed accuracy) is shown for comparison.}",
             r"\label{tab:decisions}", r"\scriptsize", r"\setlength{\tabcolsep}{3pt}",
             r"\renewcommand{\arraystretch}{1.12}",
             r"\begin{tabularx}{\textwidth}{@{}P{0.9cm} P{3.0cm} L L P{2.3cm} P{2.3cm}@{}}",
             r"\toprule",
             r"\textbf{ID} & \textbf{Requirements} & \textbf{Step 1: excluded (violated limit)} & "
             r"\textbf{Step 2: feasible uncertainty-aware Pareto set} & "
             r"\textbf{Step 3: tie-aware choice} & \textbf{Point-estimate choice} \\", r"\midrule"]
    prev = None
    for res in results:
        sc = res["scenario"]
        if prev and prev != sc.context:
            lines.append(r"\midrule")
        prev = sc.context
        mem = (f"RAM $\\le$ {sc.max_mem:.1f}\\,GiB; " if sc.max_mem is not None
               else "fits server memory; ")
        req = (f"{sc.context}, {sc.request}: {mem}latency $\\le$ {sc.max_latency_s:.0f}\\,s; "
               f"accuracy $\\ge$ {sc.min_accuracy:.2f}")
        d = res["data"]

        def group(kind):
            names = [tex_escape(d.loc[i, "model"]) + f" ({v[0].split(' ', 1)[1]})"
                     for i, v in res["reasons"].items() if v and v[0].startswith(kind)]
            return names
        parts = []
        for kind, label in (("RAM", "memory"), ("latency", "latency"), ("accuracy", "accuracy")):
            names = group(kind)
            if names:
                parts.append(f"\\emph{{{label}:}} " + ", ".join(names))
        excl = "; ".join(parts) or "none"

        def desc(i):
            r = d.loc[i]
            cost = (f"{r[sc.cost_field]:.1f}\\,GiB" if sc.cost_field == "peak_ram_gib"
                    else f"{r[sc.cost_field]:.0f}\\,J")
            return f"{tex_escape(r.model)} ({r.accuracy:.3f}; {r.latency_s:.1f}\\,s; {cost})"
        front = "; ".join(desc(i) for i in sorted(res["robust_front"],
                                                   key=lambda i: -d.loc[i, "accuracy"])) or "none"
        tie = tex_escape(d.loc[res["tie_choice"], "model"]) if res["tie_choice"] else "none"
        pt = tex_escape(d.loc[res["point_choice"], "model"]) if res["point_choice"] else "none"
        lines.append(f"{sc.name} & {req} & {excl} & {front} & \\textbf{{{tie}}} & {pt} \\\\")
    lines += [r"\bottomrule", r"\end{tabularx}", r"\end{table*}", ""]
    write_tex("rev_table_decisions.tex", "\n".join(lines))


SHORT = {
    "Gemma 3 1B (Q8)": "Gemma-1B", "Llama 3.2 1B (Q4)": "Llama-1B", "Llama 3.2 1B (Q8)": "Llama-1B",
    "Llama 3.2 1B (bf16)": "Llama-1B", "Phi-4 Mini 3.8B (Q4)": "Phi-4-mini",
    "Phi-4 Mini 3.8B (bf16)": "Phi-4-mini", "Phi-4 Mini 3.8B (GPTQ-4)": "Phi-4-mini GPTQ",
    "SmolLM3 3B (Q4)": "SmolLM3-3B", "Ministral 3 3B (Q4)": "Ministral-3B",
    "Ministral 3 3B (AWQ)": "Ministral-3B", "Ministral 3 8B (AWQ)": "Ministral-8B",
    "Ministral 3 14B": "Ministral-14B", "Phi-4 14B (bf16)": "Phi-4-14B",
    "Qwen 2.5 1.5B (bf16)": "Qwen2.5-1.5B", "Qwen 2.5 7B (AWQ)": "Qwen2.5-7B",
    "Qwen 3 4B (AWQ)": "Qwen3-4B", "Qwen 3 30B-A3B (bf16)": "Qwen3-30B-A3B",
    "Granite 4 H-Tiny (bf16)": "Granite-H-Tiny", "Kimi Linear 48B-A3B (bf16)": "Kimi-48B-A3B",
    "Llama 3.3 70B (FP8)": "Llama-70B", "Apertus 70B (bf16)": "Apertus-70B",
}


LABEL_OFFSETS = {
    ("Edge", "SmolLM3-3B"): (-4, 5, "right"),
    ("GPU server", "Kimi-48B-A3B"): (-4, 7, "right"),
    ("GPU server", "Llama-70B"): (2, 5, "left"),
    ("GPU server", "Granite-H-Tiny"): (4, -6, "left"),
    ("GPU server", "Phi-4-mini GPTQ"): (6, -10, "left"),
}


def decision_figure(results) -> None:
    primary = [r for r in results if r["scenario"].name in ("E1", "C1", "G1")]
    fig, axes = plt.subplots(1, 3, figsize=(7.16, 2.75))
    for ax, res in zip(axes, primary):
        sc, d = res["scenario"], res["data"]
        xmax = d.latency_s.max() * 1.12
        ax.axvspan(0, sc.max_latency_s, ymin=0, ymax=1, color="#E8F5E9", zorder=0)
        ax.axhspan(0, sc.min_accuracy, color="white", zorder=0.5)
        ax.axvline(sc.max_latency_s, color="0.3", linestyle="--", linewidth=0.7)
        ax.axhline(sc.min_accuracy, color="0.3", linestyle="--", linewidth=0.7)
        lat_acc = d.sort_values("latency_s")
        # Accuracy-latency envelope of the context's (unconstrained) Pareto set.
        env, best = [], -1.0
        for _, r in lat_acc.iterrows():
            if r.accuracy > best:
                env.append((r.latency_s, r.accuracy))
                best = r.accuracy
        if len(env) > 1:
            xs, ys = zip(*env)
            ax.step(list(xs) + [xmax], list(ys) + [ys[-1]], where="post",
                    color=COLORS[sc.context], linewidth=0.9, zorder=2)
        for i, r in d.iterrows():
            lo, hi = wilson(r.accuracy, N_ITEMS)
            why = res["reasons"][i]
            infeasible = bool(why)
            mem_out = any(w.startswith("RAM") for w in why)
            ax.errorbar(r.latency_s, r.accuracy, yerr=[[r.accuracy - lo], [hi - r.accuracy]],
                        fmt="none", ecolor="0.55" if infeasible else "0.15", elinewidth=0.6,
                        capsize=1.5, zorder=3)
            if i == res["tie_choice"]:
                ax.scatter(r.latency_s, r.accuracy, marker="*", s=120, color=COLORS[sc.context],
                           edgecolor="black", linewidth=0.6, zorder=6)
            elif mem_out:
                ax.scatter(r.latency_s, r.accuracy, marker="x", s=22, color="0.45", zorder=4)
            elif infeasible:
                ax.scatter(r.latency_s, r.accuracy, marker="o", s=16, facecolor="white",
                           edgecolor="0.45", linewidth=0.7, zorder=4)
            else:
                ax.scatter(r.latency_s, r.accuracy, marker="o", s=18, color=COLORS[sc.context],
                           edgecolor="black", linewidth=0.4, zorder=5)
            short = SHORT.get(r.model, r.model)
            if not infeasible or sc.context != "GPU server" or r.accuracy >= 0.69:
                dx, dy, ha = LABEL_OFFSETS.get((sc.context, short), (3, 3, "left"))
                ax.annotate(short, (r.latency_s, r.accuracy), xytext=(dx, dy),
                            textcoords="offset points", fontsize=5.8, ha=ha,
                            color="0.1" if not infeasible else "0.45", zorder=7)
        mem = (f"RAM$\\leq${sc.max_mem:.0f} GiB, " if sc.max_mem else "")
        ax.set_title(f"({'abc'[primary.index(res)]}) {sc.context} [{sc.name}]: {mem}"
                     f"L$\\leq${sc.max_latency_s:.0f} s, A$\\geq${sc.min_accuracy:.2f}", fontsize=7.2)
        ax.set_xlim(0, xmax)
        ax.set_ylim(0.0 if sc.context == "Local CPU" else 0.25, 0.9)
        ax.set_xlabel("Mean end-to-end latency (s)")
        ax.grid(linewidth=0.3, alpha=0.5)
    axes[0].set_ylabel("GSM8K accuracy (95% Wilson CI)")
    from matplotlib.lines import Line2D
    handles = [
        Line2D([], [], marker="*", linestyle="", markersize=9, markerfacecolor="0.7",
               markeredgecolor="black", label="selected (tie-aware rule)"),
        Line2D([], [], marker="o", linestyle="", markersize=4, markerfacecolor="0.7",
               markeredgecolor="black", label="feasible"),
        Line2D([], [], marker="o", linestyle="", markersize=4, markerfacecolor="white",
               markeredgecolor="0.45", label="violates latency or accuracy limit"),
        Line2D([], [], marker="x", linestyle="", markersize=4, color="0.45",
               label="violates memory limit"),
        Line2D([], [], color="0.3", linewidth=0.9, label="accuracy-latency Pareto envelope"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=5, frameon=False,
               bbox_to_anchor=(0.5, -0.06), fontsize=6.5)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.savefig(FIG / "rev_fig_decision_frontiers.pdf")
    plt.close(fig)
    print("wrote", FIG / "rev_fig_decision_frontiers.pdf")


# ------------------------------------------------- 3. matched-checkpoint subset
MATCHED = ["Gemma 3 1B (Q8)", "Phi-4 Mini 3.8B (Q4)"]


def matched_table(df: pd.DataFrame) -> None:
    rows, out = [], []
    for m in MATCHED:
        for ds in DATASETS:
            e = df[(df.context == "Edge") & (df.model == m) & (df.dataset == ds)].iloc[0]
            c = df[(df.context == "Local CPU") & (df.model == m) & (df.dataset == ds)].iloc[0]
            lo, hi = newcombe(e.accuracy, N_ITEMS, c.accuracy, N_ITEMS)
            lat_ok = ds in ("MMLU", "GSM8K")
            ratio = e.latency_s / c.latency_s if lat_ok else float("nan")
            rows.append(dict(model=m, dataset=ds, acc_edge=e.accuracy, acc_cpu=c.accuracy,
                             diff=e.accuracy - c.accuracy, diff_lo=lo, diff_hi=hi,
                             lat_edge=e.latency_s if lat_ok else float("nan"),
                             lat_cpu=c.latency_s if lat_ok else float("nan"), ratio=ratio))
    pd.DataFrame(rows).to_csv(DATA / "derived_matched_subset.csv", index=False)
    out = [r"\begin{table}[!t]", r"\revstart\centering",
           r"\caption{Matched-checkpoint comparison between the Edge and Local CPU contexts. "
           r"Both configurations use the identical GGUF file, llama.cpp backend, prompts, item set, "
           r"and greedy decoding; only the host hardware (and the Podman container on the laptop) "
           r"differs. Diff.\ is Edge minus Local CPU accuracy with a 95\% Newcombe interval. "
           r"Latency ratios are reported for the workloads re-measured in the current pipeline.}",
           r"\label{tab:matched}", r"\footnotesize", r"\setlength{\tabcolsep}{3pt}",
           r"\begin{tabular*}{\columnwidth}{@{\extracolsep{\fill}}l c c c c@{}}", r"\toprule",
           r"\textbf{Bench.} & \textbf{Edge} & \textbf{CPU} & "
           r"\textbf{Diff. [95\% CI]} & \textbf{Latency (s), ratio} \\", r"\midrule"]
    for i, r in enumerate(rows):
        if r["dataset"] == "MMLU":
            if i:
                out.append(r"\midrule")
            label = {"Gemma 3 1B (Q8)": "Gemma 3 1B (Q8\\_0)",
                     "Phi-4 Mini 3.8B (Q4)": "Phi-4 Mini 3.8B (Q4\\_K\\_M)"}[r["model"]]
            out.append(rf"\multicolumn{{5}}{{@{{}}l}}{{\textit{{{label}}}}} \\")
        ratio = "--" if math.isnan(r["ratio"]) else (
            f"{r['lat_edge']:.1f}/{r['lat_cpu']:.1f} ({r['ratio']:.2f}$\\times$)")
        out.append(f"{r['dataset']} & {r['acc_edge']:.3f} & {r['acc_cpu']:.3f} & "
                   f"${r['diff']:+.3f}$ [${r['diff_lo']:+.2f}$, ${r['diff_hi']:+.2f}$] & {ratio} \\\\")
    out += [r"\bottomrule", r"\end{tabular*}", r"\end{table}", ""]
    write_tex("rev_table_matched.tex", "\n".join(out))


# --------------------------------------------- 4. Raspberry Pi sustained rates
def pi_rates(df: pd.DataFrame) -> None:
    """Active-generation rate vs. measured request rate including the cooling pause.

    Request rates are the measured run-level rates (session-weighted means over
    runs) from data/derived_run_level_intervals.csv, written by
    compute_supplementary.py from the raw per-run results.
    """
    pause = {"MMLU": 1.0, "GSM8K": 2.0}
    iv = pd.read_csv(DATA / "derived_run_level_intervals.csv")
    e = df[df.context == "Edge"]
    rows = []
    for m in sorted(e.model.unique()):
        rec = {"model": m}
        for ds in ("MMLU", "GSM8K"):
            r = e[(e.model == m) & (e.dataset == ds)].iloc[0]
            q = iv[(iv.context == "Edge") & (iv.model == m) & (iv.dataset == ds)].iloc[0]
            rec[ds] = dict(tok=r.reported_tokens_per_s, lat=r.latency_s, rate=q.sps,
                           rate_lo=q.sps_lo, rate_hi=q.sps_hi, pause_share=pause[ds] * q.sps)
        rows.append(rec)
    flat = [{"model": r["model"], "dataset": ds, **r[ds]} for r in rows for ds in ("MMLU", "GSM8K")]
    pd.DataFrame(flat).to_csv(DATA / "derived_pi_rates.csv", index=False)
    out = [r"\begin{table}[!t]", r"\revstart\centering",
           r"\caption{Raspberry Pi~5: active-generation rate (generated words per second of the "
           r"decoding interval) versus measured end-to-end request rate, which includes the "
           r"inter-request cooling pause of 1\,s (MMLU) or 2\,s (GSM8K). Request rates are run-level "
           r"means; their 95\% intervals are within $\pm$1\% (Supplementary Table~S2). Pause is the "
           r"fraction of wall-clock time spent in the pause.}",
           r"\label{tab:pi-rates}", r"\footnotesize", r"\setlength{\tabcolsep}{2.2pt}",
           r"\begin{tabularx}{\columnwidth}{@{}L l r r r r@{}}", r"\toprule",
           r"\textbf{Config.} & \textbf{Bench.} & \textbf{Tok/s} & \textbf{Lat. (s)} & "
           r"\textbf{Req/h} & \textbf{Pause} \\", r"\midrule"]
    for r in rows:
        for ds in ("MMLU", "GSM8K"):
            v = r[ds]
            name = tex_escape(r["model"]) if ds == "MMLU" else ""
            out.append(f"{name} & {ds} & {v['tok']:.2f} & {v['lat']:.1f} & "
                       f"{3600 * v['rate']:.0f} & {100 * v['pause_share']:.0f}\\% \\\\")
    out += [r"\bottomrule", r"\end{tabularx}", r"\end{table}", ""]
    write_tex("rev_table_pi_rates.tex", "\n".join(out))
    widths = [(v["rate_hi"] - v["rate_lo"]) / 2 / v["rate"] for r in rows for v in (r["MMLU"], r["GSM8K"])]
    print(f"Pi request-rate interval half-width: max {100 * max(widths):.2f}%")


# ---------------------------------------- 5. corrected cost-effectiveness (T7)
def cost_effectiveness(df: pd.DataFrame) -> None:
    m = df[(df.dataset == "MMLU") & df.samples_per_s.notna()].copy()
    m["ecr"] = m.accuracy * m.samples_per_s
    m["ecr_lo"] = [wilson(a, N_ITEMS)[0] * s for a, s in zip(m.accuracy, m.samples_per_s)]
    m["ecr_hi"] = [wilson(a, N_ITEMS)[1] * s for a, s in zip(m.accuracy, m.samples_per_s)]
    audit = []
    for _, r in m.iterrows():
        sub = SUBMITTED_TABLE7.get((r.context, r.model))
        if sub:
            acc_s, tok_s, reported = sub
            audit.append({"context": r.context, "model": r.model, "submitted_accuracy": acc_s,
                          "submitted_tokens_per_s": tok_s, "submitted_acc_per_s": reported,
                          "stated_formula_acc_over_tokens_per_s": acc_s / tok_s,
                          "acc_over_latency_s": acc_s / r.latency_s,
                          "corrected_accuracy": r.accuracy, "samples_per_s": r.samples_per_s,
                          "expected_correct_per_s": r.ecr})
    pd.DataFrame(audit).to_csv(DATA / "derived_table7_audit.csv", index=False)
    m.to_csv(DATA / "derived_cost_effectiveness.csv", index=False)

    lines = [r"\begin{table}[!t]", r"\revstart\centering",
             r"\caption{Corrected short-output cost-effectiveness on MMLU: expected correct responses "
             r"per second, $E_c$, i.e., the product of accuracy $A_c$ and the measured "
             r"run-level request rate $R_c$ (requests/s) of one sequential stream, including inter-request "
             r"overheads and, on the Raspberry Pi~5, the 1-s cooling pause. Intervals propagate the "
             r"95\% Wilson interval of $A_c$; $R_c$ is treated as fixed. Rankings are within-context only.}",
             r"\label{tab:cost}", r"\footnotesize", r"\setlength{\tabcolsep}{2.4pt}",
             r"\begin{tabularx}{\columnwidth}{@{}L r r r c@{}}", r"\toprule",
             r"\textbf{Configuration} & \textbf{Acc.} & \textbf{Lat. (s)} & "
             r"\textbf{Rate (req/s)} & \textbf{Correct/s [95\% CI]} \\", r"\midrule"]
    for ci, ctx in enumerate(CONTEXTS):
        sub = m[m.context == ctx].sort_values("ecr", ascending=False)
        if sub.empty:
            continue
        lines.append(rf"\multicolumn{{5}}{{@{{}}l}}{{\textit{{{ctx}}}}} \\")
        for _, r in sub.iterrows():
            lines.append(f"{tex_escape(r.model)} & {r.accuracy:.3f} & {r.latency_s:.3f} & "
                         f"{r.samples_per_s:.3f} & {r.ecr:.3f} [{r.ecr_lo:.3f}, {r.ecr_hi:.3f}] \\\\")
    lines += [r"\bottomrule", r"\end{tabularx}", r"\end{table}", ""]
    write_tex("rev_table_cost.tex", "\n".join(lines))

    a = [r"\begin{tabular}{@{}l l r r r r@{}}", r"\toprule",
         r"Context & Configuration & Submitted Acc/s & Stated formula $A/\mathrm{TPS}$ & "
         r"$A/\bar{L}$ & Corrected $E_c$ \\", r"\midrule"]
    for r in audit:
        a.append(f"{r['context']} & {tex_escape(r['model'])} & {r['submitted_acc_per_s']:.3f} & "
                 f"{r['stated_formula_acc_over_tokens_per_s']:.4f} & {r['acc_over_latency_s']:.3f} & "
                 f"{r['expected_correct_per_s']:.3f} \\\\")
    a += [r"\bottomrule", r"\end{tabular}", ""]
    (DATA / "table7_audit_for_response.tex").write_text("\n".join(a))


# --------------------------------------------------- 6. workload characterisation
def workload(df: pd.DataFrame) -> None:
    """Approximate generated tokens per request = tokens/s x (E2E - TTFT).

    Uses internally consistent triples only: the original thesis runs (Table 8.2)
    and the rows of the exported timing table.
    """
    triples = []
    th = pd.read_csv(DATA / "thesis_tables_8_1_8_2.csv").dropna(subset=["latency_ms"])
    for _, r in th.iterrows():
        triples.append((r.platform, r.dataset, r.ttft_ms, r.latency_ms, r.tokens_per_s, "thesis"))
    for _, r in df.iterrows():
        if (r.ttft_ms_source == r.latency_ms_source == r.tokens_per_s_source == "tables"):
            triples.append((r.context, r.dataset.lower(), r.ttft_ms, r.latency_ms,
                            r.tokens_per_s, "tables"))
    ctx = {"pi5": "Edge", "cpu": "Local CPU", "vllm": "GPU server"}
    rows = [{"context": ctx.get(p, p), "dataset": d, "source": s,
             "est_generated_tokens": tps * (lat - ttft) / 1000.0}
            for p, d, ttft, lat, tps, s in triples]
    w = pd.DataFrame(rows)
    w.to_csv(DATA / "derived_workload_tokens.csv", index=False)
    summ = w.groupby(["dataset", "context"]).est_generated_tokens.agg(["min", "median", "max", "count"])
    summ.to_csv(DATA / "derived_workload_summary.csv")
    print(summ.round(1))


def main():
    setup_mpl()
    df = load()
    acc = accuracy_table(df)
    accuracy_table_tex(acc)
    leader_table_tex(acc)
    accuracy_figure(acc)
    results = run_decisions(df)
    decision_table_tex(results)
    decision_figure(results)
    matched_table(df)
    pi_rates(df)
    cost_effectiveness(df)
    workload(df)


if __name__ == "__main__":
    main()
