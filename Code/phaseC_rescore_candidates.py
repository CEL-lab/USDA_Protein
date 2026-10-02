"""Re-score candidate links for STRING-isolated proteins with the cold-start model.

The published candidate list was scored with Node2Vec-only + GCN, which is at
chance for proteins without edges (Results/PHASE_AB_NOTES.md). Here the best
cold-start configuration (Biomni + ESM, MLP encoder, degree-matched negatives)
is trained on each of the 10 cold-start splits and the 10 models are averaged.

Outputs (Results/phaseC_candidates/):
  heldout_precision.csv       full-ranking precision of each split's model on its
                              held-out proteins (all test x train-partner pairs,
                              true STRING density as the prior)
  candidates_<set>_top.csv    top-ranked isolated-connected / isolated-isolated
                              pairs with annotations
  class_pair_enrichment_<set>_top<K>.csv
                              observed vs expected tier-pair counts in the top K
  high_priority_isolated_partners.csv
                              top partners for each High-priority isolated protein
  Figure_classpair_enrichment.png/.pdf

Usage (from repo root):
    python Code/phaseC_rescore_candidates.py
"""

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from scipy import stats
from sklearn.metrics import roc_auc_score
from torch_geometric.utils import to_undirected

sys.path.insert(0, str(Path(__file__).resolve().parent))
from phaseA_rerun import (  # noqa: E402
    CONFIG, REPO, bce_loss, bh_fdr, biomni_matrix, build_graph, esm_matrix, eval_auc_ap, set_global_seed,
)
from phaseB_coldstart import MLPLinkPredictor, as_data, make_cold_split  # noqa: E402

TIERS = ["Very Low", "Low", "Medium", "High"]
FRACTIONS = [0.0005, 0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1]


def train_model(X, mp, splits, seed):
    set_global_seed(seed)
    tr = as_data(X, mp, *splits["train"])
    va = as_data(X, mp, *splits["val"])
    model = MLPLinkPredictor(X.shape[1], CONFIG["hidden_dim"], CONFIG["out_dim"], CONFIG["dropout"])
    opt = torch.optim.Adam(model.parameters(), lr=CONFIG["lr"])
    best_val, best_state, no_improve = 0.0, None, 0
    for _ in range(CONFIG["max_epochs"]):
        model.train()
        opt.zero_grad()
        bce_loss(model(tr.x, tr.edge_index, tr.edge_label_index), tr.edge_label).backward()
        opt.step()
        val_auc, _, _ = eval_auc_ap(model, va)
        if val_auc > best_val + 1e-4:
            best_val, no_improve = val_auc, 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            no_improve += 1
            if no_improve >= CONFIG["patience"]:
                break
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        z = model.encode(X, mp)
    return z, best_val


def pair_scores(z, rows, cols):
    """Sigmoid dot-product scores for the full rows x cols block."""
    return torch.sigmoid(z[rows] @ z[cols].T).numpy()


def precision_at_fractions(scores, labels):
    order = np.argsort(-scores, kind="stable")
    hits = np.cumsum(labels[order])
    out = {}
    for f in FRACTIONS:
        k = max(1, int(round(f * len(scores))))
        out[f] = (k, hits[k - 1] / k)
    return out


def enrichment(top, pool_counts, K, label):
    """Observed vs expected unordered tier-pair counts among the top K pairs."""
    obs = top.groupby("tier_pair").size()
    p = pool_counts / pool_counts.sum()
    rows = []
    for tp, prob in p.items():
        o = int(obs.get(tp, 0))
        e = prob * K
        pval = stats.binomtest(o, K, prob).pvalue
        rows.append({"set": label, "K": K, "tier_pair": tp, "observed": o, "expected": e,
                     "obs_over_exp": o / e if e > 0 else np.nan, "binom_p": pval})
    df = pd.DataFrame(rows)
    df["binom_p_bh"] = bh_fdr(df["binom_p"])
    return df


def sequence_groups(nodes):
    seqs, pid = {}, None
    for line in open(REPO / "Data" / "Flavobacterium sp. MO_0223_q7L1K.faa"):
        line = line.strip()
        if line.startswith(">"):
            pid = line[1:].split()[0]
            seqs[pid] = []
        elif pid:
            seqs[pid].append(line)
    seqs = {k: "".join(v) for k, v in seqs.items()}
    gid = {}
    groups = np.array([gid.setdefault(seqs[n], len(gid)) for n in nodes])
    return groups, np.bincount(groups)[groups]


def tier_pair(a, b):
    ia, ib = TIERS.index(a), TIERS.index(b)
    lo, hi = sorted((ia, ib))
    return f"{TIERS[lo]}–{TIERS[hi]}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=list(range(10)))
    ap.add_argument("--esm", type=Path, default=REPO / "Data" / "esm_embeddings_esm2_t33_650M_UR50D_all.npz")
    ap.add_argument("--top", type=int, nargs="+", default=[2000, 5000])
    ap.add_argument("--features", choices=["biomni_esm", "esm"], default="biomni_esm",
                    help="esm drops the Biomni block, to check whether tier enrichment is circular")
    ap.add_argument("--cap", type=int, default=5, help="max pairs per protein in the capped list")
    ap.add_argument("--out", type=Path, default=Path("Results/phaseC_candidates"))
    args = ap.parse_args()
    out = REPO / args.out
    out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)

    G, biomni_df = build_graph()
    nodes = list(G.nodes)
    idx = {n: i for i, n in enumerate(nodes)}
    N = len(nodes)
    X_esm = torch.tensor(esm_matrix(nodes, args.esm))
    if args.features == "esm":
        X = X_esm
    else:
        X = torch.cat([torch.tensor(biomni_matrix(biomni_df, nodes, "fixed")[0], dtype=torch.float), X_esm], dim=1)
    edge_index = to_undirected(
        torch.tensor([[idx[u], idx[v]] for u, v in G.edges()], dtype=torch.long).t(), num_nodes=N)
    deg = np.array([G.degree(n) for n in nodes])
    connected = np.where(deg > 0)[0]
    isolated = np.where(deg == 0)[0]
    adj = np.zeros((N, N), dtype=bool)
    adj[edge_index[0].numpy(), edge_index[1].numpy()] = True
    print(f"nodes={N} connected={len(connected)} isolated={len(isolated)} X dim={X.shape[1]}", flush=True)

    # ---- train one model per cold-start split; evaluate full-ranking precision on held-out proteins
    sum_ic = np.zeros((len(isolated), len(connected)), dtype=np.float64)
    sum_ii = np.zeros((len(isolated), len(isolated)), dtype=np.float64)
    prec_rows = []
    for seed in args.seeds:
        mp, splits, role = make_cold_split(edge_index, N, connected, seed, "degree")
        z, val_auc = train_model(X, mp, splits, seed)
        test_nodes = np.where(role == 2)[0]
        partners = np.where((role == 0) & (deg > 0))[0]
        s = pair_scores(z, test_nodes, partners).ravel()
        y = adj[np.ix_(test_nodes, partners)].ravel().astype(int)
        rec = {"seed": seed, "val_auc": val_auc, "n_pairs": len(s), "n_pos": int(y.sum()),
               "base_rate": y.mean(), "full_ranking_auc": roc_auc_score(y, s)}
        S, Y = s.reshape(len(test_nodes), -1), y.reshape(len(test_nodes), -1)
        has_pos = Y.sum(1) > 0
        top5 = np.argsort(-S, axis=1)[:, :5]
        rec["per_protein_precision_at5"] = np.take_along_axis(Y, top5, 1)[has_pos].mean()
        rec["per_protein_random_at5"] = Y[has_pos].mean()
        for f, (k, p) in precision_at_fractions(s, y).items():
            rec[f"precision_top{f:g}"] = p
            rec[f"enrichment_top{f:g}"] = p / y.mean()
        prec_rows.append(rec)
        print(f"seed {seed}: val={val_auc:.3f} full-ranking AUC={rec['full_ranking_auc']:.3f} "
              f"base={y.mean():.4f} P@0.1%={rec['precision_top0.001']:.3f} P@1%={rec['precision_top0.01']:.3f}",
              flush=True)
        sum_ic += pair_scores(z, isolated, connected)
        sum_ii += pair_scores(z, isolated, isolated)

    prec = pd.DataFrame(prec_rows)
    prec.to_csv(out / "heldout_precision.csv", index=False)
    summary = prec.drop(columns="seed").agg(["mean", "std"]).T
    summary.to_csv(out / "heldout_precision_summary.csv")
    print(summary.round(4).to_string())

    # ---- ensemble scores for the candidate pairs
    S_ic = sum_ic / len(args.seeds)
    S_ii = sum_ii / len(args.seeds)
    iu = np.triu_indices(len(isolated), k=1)
    np.savez_compressed(out / "ensemble_scores.npz", S_isolated_connected=S_ic.astype(np.float32),
                        S_isolated_isolated=S_ii.astype(np.float32),
                        isolated=np.array(nodes)[isolated], connected=np.array(nodes)[connected])

    ann = biomni_df.reindex(nodes)
    prot = np.array(nodes)
    tier = ann["priority"].values
    # The Biomni "description" column only repeats the protein id; use the annotation of the
    # matched F. columnare STRING protein where one exists.
    mapping = pd.read_csv(REPO / "Data" / "string_mapping.tsv", sep="\t")
    string_ann = dict(zip(mapping["queryItem"], mapping["annotation"]))
    desc = np.array([string_ann.get(n, "") for n in nodes], dtype=object)
    reason = np.array(["connected" if deg[i] > 0 else
                       ("secondary STRING hit (STRING protein assigned to a better match)" if n in string_ann else "no STRING match")
                       for i, n in enumerate(nodes)], dtype=object)
    # identical sequences share ESM and Biomni features, so they get identical scores
    seq_group, n_copies = sequence_groups(nodes)

    tier_code = np.array([TIERS.index(t) for t in tier])
    CAP_POOL = 1_000_000  # ranked pairs kept in memory; the per-protein cap is applied within these
    top_n = max(args.top)

    def table(rows_idx, cols_idx, scores, label):
        order = np.argsort(-scores, kind="stable")
        ta, tb = tier_code[rows_idx], tier_code[cols_idx]
        code = np.minimum(ta, tb) * 4 + np.maximum(ta, tb)
        counts = np.bincount(code, minlength=16)
        pool = pd.Series({f"{TIERS[c // 4]}–{TIERS[c % 4]}": counts[c] for c in range(16) if counts[c] > 0})
        o = order[:CAP_POOL]
        df = pd.DataFrame({"rank": np.arange(1, len(o) + 1),
                           "protein_a": prot[rows_idx[o]], "protein_b": prot[cols_idx[o]], "score": scores[o]})
        df["percentile_top"] = df["rank"] / len(scores)
        df["tier_a"], df["tier_b"] = tier[rows_idx[o]], tier[cols_idx[o]]
        df["tier_pair"] = [tier_pair(a, b) for a, b in zip(df["tier_a"], df["tier_b"])]
        df["desc_a"], df["desc_b"] = desc[rows_idx[o]], desc[cols_idx[o]]
        df["degree_b"] = deg[cols_idx[o]]
        df["isolated_reason_a"] = reason[rows_idx[o]]
        df["isolated_reason_b"] = reason[cols_idx[o]]
        df["seq_group_a"], df["seq_group_b"] = seq_group[rows_idx[o]], seq_group[cols_idx[o]]
        df["n_identical_a"], df["n_identical_b"] = n_copies[rows_idx[o]], n_copies[cols_idx[o]]
        print(f"{label}: {len(scores):,} candidate pairs; top score {df['score'].iloc[0]:.3f}, "
              f"score at rank 2000 {df['score'].iloc[1999]:.3f}", flush=True)
        return df, pool

    ri, ci = np.meshgrid(isolated, connected, indexing="ij")
    ic_df, ic_pool = table(ri.ravel(), ci.ravel(), S_ic.ravel(), "isolated-connected")
    ii_df, ii_pool = table(isolated[iu[0]], isolated[iu[1]], S_ii[iu], "isolated-isolated")

    ic_df.head(top_n).to_csv(out / "candidates_isolated_connected_top.csv", index=False)
    ii_df.head(top_n).to_csv(out / "candidates_isolated_isolated_top.csv", index=False)

    def capped(df):
        """Collapse identical-sequence copies and keep at most args.cap pairs per sequence."""
        seen, pairs, keep = {}, set(), []
        for a, b in zip(df["seq_group_a"], df["seq_group_b"]):
            key = (min(a, b), max(a, b))
            ok = a != b and key not in pairs and seen.get(a, 0) < args.cap and seen.get(b, 0) < args.cap
            keep.append(ok)
            if ok:
                pairs.add(key)
                seen[a] = seen.get(a, 0) + 1
                seen[b] = seen.get(b, 0) + 1
        return df[keep].reset_index(drop=True)

    ic_cap, ii_cap = capped(ic_df), capped(ii_df)
    ic_cap.to_csv(out / "candidates_isolated_connected_capped.csv", index=False)
    ii_cap.to_csv(out / "candidates_isolated_isolated_capped.csv", index=False)
    print(f"capped lists (<= {args.cap} pairs per protein): {len(ic_cap)} / {len(ii_cap)} pairs "
          f"from the top {top_n}", flush=True)

    enr = []
    for K in args.top:
        enr.append(enrichment(ic_df.head(K), ic_pool, K, "isolated-connected"))
        enr.append(enrichment(ii_df.head(K), ii_pool, K, "isolated-isolated"))
    for K in [k for k in args.top if k <= min(len(ic_cap), len(ii_cap))]:
        enr.append(enrichment(ic_cap.head(K), ic_pool, K, "isolated-connected (capped)"))
        enr.append(enrichment(ii_cap.head(K), ii_pool, K, "isolated-isolated (capped)"))
    enr = pd.concat(enr, ignore_index=True)
    enr.to_csv(out / "class_pair_enrichment.csv", index=False)
    print(enr[enr["K"] == 2000][["set", "tier_pair", "observed", "expected", "obs_over_exp", "binom_p_bh"]]
          .round(4).to_string())

    # ---- top partners for each High-priority isolated protein (any partner type)
    rows = []
    for a, i in enumerate(isolated):
        if tier[i] != "High":
            continue
        cand = np.r_[S_ic[a], np.delete(S_ii[a], a)]
        partner = np.r_[connected, np.delete(isolated, a)]
        best, used = [], {seq_group[i]}
        for b in np.argsort(-cand):
            if seq_group[partner[b]] not in used:
                used.add(seq_group[partner[b]])
                best.append(b)
            if len(best) == 5:
                break
        for r, b in enumerate(best, 1):
            rows.append({"isolated_protein": prot[i], "isolated_reason": reason[i], "isolated_desc": desc[i],
                         "n_identical": n_copies[i],
                         "vaccine_score": ann["vaccine_score"].values[i], "partner_rank": r,
                         "partner": prot[partner[b]], "partner_tier": tier[partner[b]],
                         "partner_desc": desc[partner[b]], "partner_degree": deg[partner[b]],
                         "score": cand[b]})
    hp = pd.DataFrame(rows)
    hp.to_csv(out / "high_priority_isolated_partners.csv", index=False)

    # ---- figure: observed/expected tier-pair heatmaps for the top 2000
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), gridspec_kw={"wspace": 0.45})
    for ax, label in zip(axes, ["isolated-connected (capped)", "isolated-isolated (capped)"]):
        d = enr[(enr["set"] == label) & (enr["K"] == 2000)].set_index("tier_pair")
        M = np.full((4, 4), np.nan)
        A = [["" for _ in range(4)] for _ in range(4)]
        for i in range(4):
            for j in range(i, 4):
                tp = f"{TIERS[i]}–{TIERS[j]}"
                if tp in d.index:
                    r = d.loc[tp]
                    M[i, j] = M[j, i] = np.log2(max(r["obs_over_exp"], 1e-3))
                    star = "*" if r["binom_p_bh"] < 0.05 else ""
                    A[i][j] = A[j][i] = f"{int(r['observed'])}\n({r['obs_over_exp']:.2f}){star}"
        im = ax.imshow(M, cmap="RdBu_r", vmin=-3, vmax=3)
        for i in range(4):
            for j in range(4):
                ax.text(j, i, A[i][j], ha="center", va="center", fontsize=7)
        ax.set_xticks(range(4), TIERS, rotation=30)
        ax.set_yticks(range(4), TIERS)
        ax.set_title(f"Top 2,000 {label.replace(' (capped)', '')} pairs\n(≤ {args.cap} per protein)", fontsize=9)
    fig.colorbar(im, ax=axes, shrink=0.8, label="log2(observed / expected)")
    fig.suptitle("Biomni priority tier pairs among top-ranked predictions "
                 "(count, O/E; * BH-adjusted binomial p < 0.05)", fontsize=9)
    for ext in ("png", "pdf"):
        fig.savefig(out / f"Figure_classpair_enrichment.{ext}", dpi=300, bbox_inches="tight")
    print(f"Saved outputs to {out}")


if __name__ == "__main__":
    main()
