"""Figures for the resubmitted manuscript.

  fig_benchmark            three evaluation protocols x feature sets x models (final graph)
  fig_external_validation  ranking methods on the never-submitted proteins (development graph)
  fig_biomni_tiers         (a) surface/secretion families by Biomni tier,
                           (b) tier pairs: model top-5 partners vs real recovered links
  fig_coherence            functional coherence of predicted pairs vs random pairs and real edges

Usage (from repo root):
    python Code/make_manuscript_figures.py [--only external tiers benchmark coherence]
Writes PDF + PNG to Flova_Reverse_Vac_Biomni/resubmission/figures/
"""

import argparse
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
REPO = Path(__file__).resolve().parents[1]
RES = REPO / "Results"
OUT = REPO / "Flova_Reverse_Vac_Biomni" / "resubmission" / "figures"

# validated categorical palette (dataviz reference, slots 1-4), text and grid tokens
C = {"GCN": "#2a78d6", "GAT": "#eb6834", "GraphSAGE": "#1baf7a", "MLP (no graph)": "#eda100"}
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
FEATURES = ["Biomni only", "ESM only", "Node2Vec only", "Biomni + ESM",
            "Biomni + Node2Vec", "ESM + Node2Vec", "Biomni + ESM + Node2Vec"]
TOPO = ["Common neighbors", "Adamic-Adar", "Resource allocation", "Preferential attachment"]

plt.rcParams.update({"font.size": 8, "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2,
                     "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False,
                     "font.family": "DejaVu Sans"})


def save(fig, name):
    OUT.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"{name}.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {name}")


def fig_benchmark(root):
    panels = [("phaseA_fixed_esm650M", "auc_mean", "auc_std", "a  Edge split (both proteins in the training graph), uniform negatives"),
              ("phaseB_coldstart_uniform", "C2_auc_mean", "C2_auc_std", "b  Held-out proteins, uniform negatives"),
              ("phaseB_coldstart_degree", "C2_auc_mean", "C2_auc_std", "c  Held-out proteins, degree-matched negatives")]
    fig, axes = plt.subplots(3, 1, figsize=(7.2, 7.6), sharex=True)
    x = np.arange(len(FEATURES))
    for ax, (run, m, s, title) in zip(axes, panels):
        d = pd.read_csv(root / run / "summary.csv")
        models = [k for k in C if k in set(d["model"])]
        w = 0.8 / len(models)
        for i, mod in enumerate(models):
            dm = d[d["model"] == mod].set_index("feature_set").reindex(FEATURES)
            ax.bar(x + (i - (len(models) - 1) / 2) * w, dm[m] - 0.3, w * 0.9, bottom=0.3, yerr=dm[s],
                   color=C[mod], label=mod, error_kw={"lw": 0.7, "ecolor": INK2, "capsize": 1.5})
        base = d[~d["model"].isin(list(C))]
        lines = []
        topo = base[base["model"].isin(TOPO)]
        if len(topo):
            lines.append(("best topology heuristic", topo[m].max(), "--"))
        for name in ["Partner degree", "ESM cosine"]:
            if name in set(base["model"]):
                lines.append((name.lower(), base.loc[base["model"] == name, m].iloc[0], ":"))
        last = -1.0
        for name, val, ls in sorted(lines, key=lambda t: t[1]):
            ax.axhline(val, color=INK, lw=0.8, ls=ls, zorder=0)
            ypos = max(val, last + 0.04)  # keep labels of nearby baselines apart
            last = ypos
            ax.annotate(f"{name} {val:.2f}", (len(FEATURES) - 0.45, ypos), xytext=(2, 0),
                        textcoords="offset points", va="center", fontsize=6.5, color=INK2, annotation_clip=False)
        ax.axhline(0.5, color=GRID, lw=0.8, zorder=0)
        ax.set_ylim(0.3, 1.0)
        ax.set_ylabel("Test ROC-AUC")
        ax.set_title(title, loc="left", fontsize=8, color=INK)
        ax.yaxis.grid(True, color=GRID, lw=0.5)
        ax.set_axisbelow(True)
    axes[-1].set_xticks(x, [f.replace("ESM", "ESM-2") for f in FEATURES], rotation=20, ha="right")
    h = [plt.Rectangle((0, 0), 1, 1, color=C[k]) for k in C]
    fig.legend(h, list(C), ncol=4, frameon=False, loc="upper center", bbox_to_anchor=(0.5, 1.01))
    fig.tight_layout(rect=(0, 0, 0.9, 0.98))
    save(fig, "fig_benchmark")


def fig_external_validation():
    e = pd.read_csv(RES / "phaseE_rerank_biomni_esm" / "external_summary.csv")
    eo = pd.read_csv(RES / "phaseE_rerank_esm" / "external_summary.csv")
    label = {"esm": "ESM-2 cosine (no training)", "dot": "Model score (Phase C)", "cos": "Model embedding cosine",
             "csls": "Model cosine, hubness-corrected", "blend(cos,0.5)": "Blend: model cosine + ESM (0.5)",
             "blend(cos,0.25)": "Blend: model cosine + ESM (0.25)", "blend(cos,0.75)": "Blend: model cosine + ESM (0.75)",
             "blend(csls,0.5)": "Blend: corrected cosine + ESM (0.5)", "partner degree": "Partner degree"}
    keep = ["esm", "blend(cos,0.25)", "blend(cos,0.5)", "blend(cos,0.75)", "cos", "dot", "partner degree"]
    e = e.set_index("variant").loc[keep]
    eo = eo.set_index("variant").loc[keep]
    base = 0.0122
    metrics = [("per_protein_P@5", "Precision of each\nprotein's top 5", base),
               ("per_protein_hit@10", "Proteins with a true\npartner in the top 10", None),
               ("pooled_AUC", "Pooled ROC-AUC\n(all pairs)", 0.5)]
    fig, axes = plt.subplots(1, 3, figsize=(7.4, 3.2), sharey=True)
    y = np.arange(len(keep))[::-1]
    for ax, (col, title, ref) in zip(axes, metrics):
        ax.hlines(y, 0, e[col], color=GRID, lw=1.2, zorder=1)
        ax.scatter(e[col], y, s=28, color=C["GCN"], zorder=3, label="Biomni + ESM model")
        ax.scatter(eo[col], y, s=28, facecolor="white", edgecolor=C["GAT"], lw=1.4, zorder=4, label="ESM-only model")
        if ref is not None:
            ax.axvline(ref, color=INK2, lw=0.8, ls=":")
            ax.annotate("random", (ref, y.min() - 0.45), fontsize=6, color=INK2, ha="left", va="center",
                        xytext=(2, 0), textcoords="offset points", annotation_clip=False)
        for yy, v in zip(y, e[col]):
            ax.annotate(f"{v:.2f}" if v >= 0.1 else f"{v:.3f}", (v, yy), xytext=(5, 3),
                        textcoords="offset points", fontsize=6, color=INK2)
        ax.set_title(title, fontsize=7.5, loc="left", color=INK)
        ax.xaxis.grid(True, color=GRID, lw=0.5)
        ax.set_axisbelow(True)
    axes[0].set_yticks(y, [label[k] for k in keep])
    axes[0].get_yticklabels()[0].set_fontweight("bold")
    axes[2].set_xlim(0.5, 0.78)
    axes[0].set_ylim(y.min() - 0.8, y.max() + 0.5)
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, frameon=False, fontsize=7, ncol=2, loc="lower center", bbox_to_anchor=(0.6, -0.04))
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    save(fig, "fig_external_validation")


def tier_pair_bias():
    """Tier-pair shares of real recovered links vs model top-5 partners (same proteins)."""
    from phaseA_rerun import build_graph
    from phaseC_rescore_candidates import tier_pair
    cache = RES / "phaseD_external_validation" / "tier_pairs_model_vs_real.csv"
    if cache.exists():
        return pd.read_csv(cache, index_col=0)
    G, b = build_graph(source="string_web")
    tier = b["priority"].to_dict()
    pp = pd.read_csv(RES / "phaseD_external_validation" / "per_protein_validation.csv")
    sc = np.load(RES / "phaseC_candidates" / "ensemble_scores.npz", allow_pickle=True)
    iso = list(sc["isolated"])
    pos = {p: i for i, p in enumerate(iso)}
    part = list(sc["connected"]) + iso
    nc = len(sc["connected"])
    rows = []
    for p in pp["protein"]:
        a = pos[p]
        s = np.r_[sc["S_isolated_connected"][a], sc["S_isolated_isolated"][a]]
        s[nc + a] = -1
        rows += [tier_pair(tier[p], tier[part[j]]) for j in np.argsort(-s)[:5]]
    real = pd.read_csv(RES / "phaseD_external_validation" / "tier_pairs_recovered_links.csv", index_col=0)
    t = pd.DataFrame({"real_links": real["observed"] / real["observed"].sum(),
                      "model_top5": pd.Series(rows).value_counts(normalize=True)}).fillna(0)
    t.to_csv(cache)
    return t


def fig_biomni_tiers():
    tv = pd.read_csv(RES / "phaseD_biology" / "tier_validation.csv").set_index("family_set")
    tiers = ["Very Low", "Low", "Medium", "High"]
    anyrow = tv.loc["Any surface/secretion family"]
    fig, (a, b) = plt.subplots(1, 2, figsize=(7.2, 2.8), gridspec_kw={"width_ratios": [1, 1.4]})
    vals = [anyrow[f"frac_{t}"] * 100 for t in tiers]
    a.bar(tiers, vals, color=C["GCN"], width=0.6)
    for i, (t, v) in enumerate(zip(tiers, vals)):
        a.annotate(f"{v:.1f}%\n({int(anyrow[f'n_{t}'])})", (i, v), xytext=(0, 2), textcoords="offset points",
                   ha="center", fontsize=6.5, color=INK2)
    a.set_ylabel("Proteins with a surface/secretion\nPfam family (%)")
    a.set_xlabel("Biomni priority tier")
    a.set_title(f"a  High vs rest: OR {anyrow['odds_ratio_high_vs_rest']:.2f}, "
                f"BH p = {anyrow['fisher_p_bh']:.2f}", loc="left", fontsize=7.5)
    a.set_ylim(0, max(vals) * 1.3)

    t = tier_pair_bias()
    order = ["Very Low–Very Low", "Very Low–Low", "Very Low–Medium", "Very Low–High", "Low–Low",
             "Low–Medium", "Low–High", "Medium–Medium", "Medium–High", "High–High"]
    t = t.reindex(order)
    ratio = t["model_top5"] / t["real_links"]
    yy = np.arange(len(order))[::-1]
    b.axvline(1, color=INK2, lw=0.8, ls=":")
    b.hlines(yy, 1, ratio, color=GRID, lw=1.2)
    b.scatter(ratio, yy, s=26, color=[C["GAT"] if r > 1 else C["GCN"] for r in ratio], zorder=3)
    for y_, r in zip(yy, ratio):
        b.annotate(f"{r:.2f}×", (r, y_), xytext=(0, 5), textcoords="offset points",
                   ha="center", va="bottom", fontsize=6.5, color=INK2)
    b.set_xscale("log", base=2)
    b.set_xlim(0.4, 4)
    b.set_ylim(yy.min() - 0.6, yy.max() + 0.8)
    b.set_xticks([0.5, 1, 2, 4], ["0.5×", "1×", "2×", "4×"])
    b.set_yticks(yy, order)
    b.set_xlabel("Share in model top-5 partners / share in real STRING links")
    b.set_title("b  Tier pairs: model predictions vs recovered real links (728 proteins)", loc="left", fontsize=7.5)
    fig.tight_layout()
    save(fig, "fig_biomni_tiers")


def fig_coherence(root):
    d = pd.read_csv(root / "phaseD_biology" / "functional_coherence.csv").set_index("pair_set")
    groups = [("Isolated–connected", [k for k in d.index if "isolated-connected" in k]),
              ("Isolated–isolated", [k for k in d.index if "isolated-isolated" in k])]
    real = [k for k in d.index if "STRING" in k][0]
    metrics = [("share_go_term", "Share a GO term"), ("share_pfam_clan", "Share a Pfam clan")]
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.6), sharey=False)
    for ax, (col, title) in zip(axes, metrics):
        labels, vals, cols = [], [], []
        for gname, keys in groups:
            top = [k for k in keys if k.startswith("Top")][0]
            rnd = [k for k in keys if k.startswith("Random")][0]
            short = "I–C" if "connected" in gname else "I–I"
            labels += [f"{short}\ntop", f"{short}\nrandom"]
            vals += [d.loc[top, col] * 100, d.loc[rnd, col] * 100]
            cols += [C["GCN"], GRID]
        labels.append("Network\nedges")
        vals.append(d.loc[real, col] * 100)
        cols.append(C["GraphSAGE"])
        xs = np.arange(len(vals))
        ax.bar(xs, vals, color=cols, width=0.65, edgecolor=[INK2 if c == GRID else c for c in cols], lw=0.5)
        for xx, v in zip(xs, vals):
            ax.annotate(f"{v:.1f}%", (xx, v), xytext=(0, 2), textcoords="offset points", ha="center",
                        fontsize=6.5, color=INK2)
        ax.set_xticks(xs, labels, fontsize=6.5)
        ax.set_ylabel(f"{title} (%)")
        ax.set_title(title, loc="left", fontsize=7.5)
        ax.set_ylim(0, max(vals) * 1.2)
    fig.tight_layout()
    save(fig, "fig_coherence")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="+", default=["external", "tiers", "benchmark", "coherence"])
    ap.add_argument("--final_root", type=Path, default=RES / "final_v12")
    args = ap.parse_args()
    if "external" in args.only:
        fig_external_validation()
    if "tiers" in args.only:
        fig_biomni_tiers()
    if "benchmark" in args.only:
        fig_benchmark(args.final_root)
    if "coherence" in args.only:
        fig_coherence(args.final_root)


if __name__ == "__main__":
    main()
