"""Collect Phase A (edge split) and Phase B (cold-start) outputs into one table
and figure.

Usage (from repo root):
    python Code/summarize_phaseAB.py
Writes Results/phaseAB_summary.csv and Results/Figure_3_Benchmark_PhaseAB.png/.pdf
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
RES = REPO / "Results"

FEATURE_ORDER = ["Biomni only", "ESM only", "Node2Vec only", "Biomni + ESM",
                 "Biomni + Node2Vec", "ESM + Node2Vec", "Biomni + ESM + Node2Vec"]
MODEL_ORDER = ["GCN", "GAT", "GraphSAGE", "MLP (no graph)"]
TOPOLOGY = ["Common neighbors", "Adamic-Adar", "Resource allocation", "Preferential attachment"]
COLORS = {"GCN": "#4C72B0", "GAT": "#DD8452", "GraphSAGE": "#55A868", "MLP (no graph)": "#8C8C8C"}

PANELS = [
    ("phaseA_fixed", "auc_mean", "auc_std", "Edge split (transductive)\nuniform negatives"),
    ("phaseB_coldstart_uniform", "C2_auc_mean", "C2_auc_std", "Held-out proteins (cold start)\nuniform negatives"),
    ("phaseB_coldstart_degree", "C2_auc_mean", "C2_auc_std", "Held-out proteins (cold start)\ndegree-matched negatives"),
]


def load():
    rows = []
    for run, mcol, scol, label in PANELS:
        f = RES / run / "summary.csv"
        if not f.exists():
            continue
        s = pd.read_csv(f)
        rows.append(pd.DataFrame({"protocol": label.replace("\n", ", "), "run": run,
                                  "feature_set": s["feature_set"], "model": s["model"],
                                  "auc_mean": s[mcol], "auc_std": s[scol], "n_seeds": s["n"]}))
    return pd.concat(rows, ignore_index=True)


def main():
    df = load()
    df.to_csv(RES / "phaseAB_summary.csv", index=False)
    runs = [p for p in PANELS if (RES / p[0] / "summary.csv").exists()]

    fig, axes = plt.subplots(len(runs), 1, figsize=(11, 3.4 * len(runs)), sharex=True)
    axes = np.atleast_1d(axes)
    x = np.arange(len(FEATURE_ORDER))
    for ax, (run, _, _, label) in zip(axes, runs):
        d = df[df["run"] == run]
        models = [m for m in MODEL_ORDER if m in set(d["model"])]
        w = 0.8 / len(models)
        for k, m in enumerate(models):
            dm = d[d["model"] == m].set_index("feature_set").reindex(FEATURE_ORDER)
            ax.bar(x + (k - (len(models) - 1) / 2) * w, dm["auc_mean"], w, yerr=dm["auc_std"],
                   color=COLORS[m], label=m, capsize=2, error_kw={"lw": 0.8})
        base = d[~d["model"].isin(MODEL_ORDER)]
        topo = base[base["model"].isin(TOPOLOGY)]
        lines = [("Best topology heuristic", topo["auc_mean"].max(), "--")] if len(topo) else []
        lines += [(b["model"], b["auc_mean"], ":") for _, b in base[~base["model"].isin(TOPOLOGY)].iterrows()]
        for name, val, ls in lines:
            ax.axhline(val, color="k", lw=0.8, ls=ls)
            ax.text(len(FEATURE_ORDER) - 0.45, val, f" {name} ({val:.2f})", va="center", fontsize=7)
        ax.axhline(0.5, color="grey", lw=0.6)
        ax.set_ylim(0.3, 1.0)
        ax.set_ylabel("Test ROC-AUC")
        ax.set_title(label, fontsize=9, loc="left")
        ax.spines[["top", "right"]].set_visible(False)
    handles = [plt.Rectangle((0, 0), 1, 1, color=COLORS[m]) for m in MODEL_ORDER]
    fig.legend(handles, MODEL_ORDER, ncol=4, fontsize=8, frameon=False, loc="upper center", bbox_to_anchor=(0.5, 1.02))
    axes[-1].set_xticks(x)
    axes[-1].set_xticklabels(FEATURE_ORDER, rotation=20, ha="right", fontsize=8)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(RES / f"Figure_3_Benchmark_PhaseAB.{ext}", dpi=300, bbox_inches="tight")
    print(df.pivot_table(index=["feature_set", "model"], columns="run", values="auc_mean").round(3).to_string())


if __name__ == "__main__":
    main()
