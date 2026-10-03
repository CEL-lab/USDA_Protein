"""Re-ranking variants for per-protein partner lists, selected internally and then
tested once on the external (never-submitted) proteins.

The Phase C dot-product scores favour a few hub partners (large embedding norms),
and in the external test raw ESM cosine gave better per-protein top-5 lists.
Candidate fixes are declared here up front:

  dot            sigmoid(z_i . z_j)                       (Phase C)
  cos            cosine of model embeddings               (removes norm-driven hubness)
  csls           2 cos - r_i - r_j, r = mean cosine to the k=10 nearest of the other
                 side (Conneau et al. 2018 hubness correction)
  esm            cosine of raw ESM-2 embeddings
  blend(a,w)     w * rowrank(a) + (1-w) * rowrank(esm), a in {cos, csls}

Selection: the variant with the highest mean per-protein precision@5 on the internal
cold-start test proteins (10 degree-matched splits). The external validation is then
reported for all variants, with the pre-selected one marked.

Usage (from repo root):
    python Code/phaseE_rerank.py [--features biomni_esm|esm]
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score
from torch_geometric.utils import to_undirected

sys.path.insert(0, str(Path(__file__).resolve().parent))
from phaseA_rerun import REPO, biomni_matrix, build_graph, esm_matrix  # noqa: E402
from phaseB_coldstart import make_cold_split  # noqa: E402
from phaseC_rescore_candidates import sequence_groups, train_model  # noqa: E402
from phaseD_external_validation import external_test_set  # noqa: E402

K_CSLS = 10
BLENDS = [("cos", 0.25), ("cos", 0.5), ("cos", 0.75), ("csls", 0.5)]


def unit(x):
    return x / np.linalg.norm(x, axis=1, keepdims=True)


def topk_mean(M, k, axis):
    k = min(k, M.shape[axis])
    return -np.mean(np.partition(-M, k - 1, axis=axis).take(range(k), axis=axis), axis=axis)


def row_pct(M):
    return np.vstack([rankdata(r) / len(r) for r in M])


def variants(z_q, z_p, e_q, e_p):
    """Score matrices (queries x partners) for every variant."""
    dot = 1 / (1 + np.exp(-(z_q @ z_p.T)))
    cos = unit(z_q) @ unit(z_p).T
    csls = 2 * cos - topk_mean(cos, K_CSLS, 1)[:, None] - topk_mean(cos, K_CSLS, 0)[None, :]
    esm = unit(e_q) @ unit(e_p).T
    out = {"dot": dot, "cos": cos, "csls": csls, "esm": esm}
    pct = {k: row_pct(out[k]) for k in ("cos", "csls", "esm")}
    for a, w in BLENDS:
        out[f"blend({a},{w})"] = w * pct[a] + (1 - w) * pct["esm"]
    return out


def evaluate(rows, label):
    """rows: list of (score_vector, label_vector) per query protein."""
    rec = {"variant": label}
    auc, p5, h10 = [], [], []
    for s, y in rows:
        auc.append(roc_auc_score(y, s))
        top = np.argsort(-s, kind="stable")
        p5.append(y[top[:5]].mean())
        h10.append(float(y[top[:10]].any()))
    s_all = np.concatenate([r[0] for r in rows])
    y_all = np.concatenate([r[1] for r in rows])
    order = np.argsort(-s_all, kind="stable")
    rec.update({"per_protein_AUC": np.mean(auc), "per_protein_P@5": np.mean(p5),
                "per_protein_hit@10": np.mean(h10), "pooled_AUC": roc_auc_score(y_all, s_all)})
    for f in (0.001, 0.01):
        k = max(1, int(round(f * len(s_all))))
        rec[f"pooled_precision_top{f:g}"] = y_all[order[:k]].mean()
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", choices=["biomni_esm", "esm"], default="biomni_esm")
    ap.add_argument("--seeds", type=int, nargs="+", default=list(range(10)))
    ap.add_argument("--esm", type=Path, default=REPO / "Data" / "esm_embeddings_esm2_t33_650M_UR50D_all.npz")
    ap.add_argument("--db", type=Path, default=Path.home() / ".cache" / "string_v12")
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--no_external", action="store_true",
                    help="skip the external test (use with USDA_GRAPH=v12_full, where those proteins are in the graph)")
    args = ap.parse_args()
    out = REPO / (args.out or Path(f"Results/phaseE_rerank_{args.features}"))
    out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)

    G, biomni_df = build_graph()
    nodes = list(G.nodes)
    idx = {n: i for i, n in enumerate(nodes)}
    N = len(nodes)
    E = esm_matrix(nodes, args.esm)
    X_esm = torch.tensor(E)
    X = X_esm if args.features == "esm" else torch.cat(
        [torch.tensor(biomni_matrix(biomni_df, nodes, "fixed")[0], dtype=torch.float), X_esm], dim=1)
    edge_index = to_undirected(
        torch.tensor([[idx[u], idx[v]] for u, v in G.edges()], dtype=torch.long).t(), num_nodes=N)
    deg = np.array([G.degree(n) for n in nodes])
    degd = dict(zip(nodes, deg))
    connected, isolated = np.where(deg > 0)[0], np.where(deg == 0)[0]
    group = dict(zip(nodes, sequence_groups(nodes)[0]))
    adj = np.zeros((N, N), dtype=bool)
    adj[edge_index[0].numpy(), edge_index[1].numpy()] = True
    partner_order = np.r_[connected, isolated]  # Phase C layout: con + iso

    internal = []
    ext_sum = None
    for seed in args.seeds:
        mp, splits, role = make_cold_split(edge_index, N, connected, seed, "degree")
        z = train_model(X, mp, splits, seed)[0].numpy()
        test = np.where(role == 2)[0]
        partners = np.where((role == 0) & (deg > 0))[0]
        V = variants(z[test], z[partners], E[test], E[partners])
        Y = adj[np.ix_(test, partners)].astype(int)
        has = Y.sum(1) > 0
        for name, M in V.items():
            rec = evaluate([(M[i], Y[i]) for i in np.where(has)[0]], name)
            internal.append({"seed": seed, **rec})
        print(f"seed {seed}: internal P@5 " + ", ".join(
            f"{r['variant']}={r['per_protein_P@5']:.3f}" for r in internal if r["seed"] == seed), flush=True)
        if args.no_external:
            continue
        # external: raw cosine / csls / dot accumulated over seeds, blends recomputed after averaging
        Ve = variants(z[isolated], z[partner_order], E[isolated], E[partner_order])
        base = {k: Ve[k] for k in ("dot", "cos", "csls")}
        ext_sum = base if ext_sum is None else {k: ext_sum[k] + base[k] for k in base}

    internal = pd.DataFrame(internal)
    internal.to_csv(out / "internal_per_seed.csv", index=False)
    isum = internal.drop(columns="seed").groupby("variant").agg(["mean", "std"])
    isum.columns = [f"{a}_{b}" for a, b in isum.columns]
    isum = isum.sort_values(["per_protein_P@5_mean", "per_protein_hit@10_mean"], ascending=False)
    isum.to_csv(out / "internal_summary.csv")
    selected = isum.index[0]
    print("\nInternal (cold-start, degree-matched) summary:")
    print(isum[["per_protein_P@5_mean", "per_protein_P@5_std", "per_protein_hit@10_mean",
                "per_protein_AUC_mean", "pooled_AUC_mean"]].round(4).to_string())
    print(f"Selected variant (highest internal per-protein P@5): {selected}")
    if args.no_external:
        return

    # ---- external test, all variants, ensemble over seeds
    ens = {k: v / len(args.seeds) for k, v in ext_sum.items()}
    e_iso, e_par = E[isolated], E[partner_order]
    ens["esm"] = unit(e_iso) @ unit(e_par).T
    pct = {k: row_pct(ens[k]) for k in ("cos", "csls", "esm")}
    for a, w in BLENDS:
        ens[f"blend({a},{w})"] = w * pct[a] + (1 - w) * pct["esm"]
    iso_names = [nodes[i] for i in isolated]
    con_names = [nodes[i] for i in connected]
    tests, _, _ = external_test_set(nodes, degd, group, iso_names, con_names, args.db)
    ext = []
    for name, M in ens.items():
        rec = evaluate([(M[t["iso_index"], t["keep"]], t["y"]) for t in tests], name)
        rec["selected_internally"] = name == selected
        ext.append(rec)
    deg_rows = [(deg[partner_order][t["keep"]].astype(float), t["y"]) for t in tests]
    ext.append({**evaluate(deg_rows, "partner degree"), "selected_internally": False})
    ext = pd.DataFrame(ext).sort_values("per_protein_P@5", ascending=False)
    ext.to_csv(out / "external_summary.csv", index=False)
    print(f"\nExternal validation ({len(tests)} never-submitted proteins):")
    print(ext.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
