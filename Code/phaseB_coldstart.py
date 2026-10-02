"""Cold-start (node-held-out) link-prediction benchmark.

The repeated random *edge* split in phaseA_rerun.py only scores pairs whose
endpoints both keep other edges in the training graph. The intended use case
of the paper is the opposite: proteins with no observed interactions. Here a
fraction of connected proteins is held out together with all of their edges,
so at evaluation time those proteins are isolated in the message-passing graph
exactly like the 1,650 STRING-isolated proteins.

Per seed:
  * 10% of connected nodes -> validation nodes, 10% -> test nodes, rest train.
  * Message-passing graph and Node2Vec walks use train-train edges only.
  * Positives: train-train (training), val-train (validation), test-train
    (test, one protein unseen; "C2" in Park & Marcotte 2012). Test-test edges
    (both unseen, "C3") are scored separately.
  * Negatives: non-edges with the same node-type constraint, 1:1 with
    positives. --neg uniform samples endpoints uniformly; --neg degree samples
    each endpoint in proportion to how often it appears among the positives,
    so node degree alone cannot separate positives from negatives.

Models: GCN, GAT, GraphSAGE and a graph-free MLP encoder, all with the
dot-product decoder. Baselines: topology heuristics (computed on the training
graph) and raw ESM cosine similarity.

Usage (from repo root):
    python Code/phaseB_coldstart.py --seeds 0 1 2 3 4 5 6 7 8 9 --out Results/phaseB_coldstart
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import average_precision_score, roc_auc_score
from torch_geometric.data import Data
from torch_geometric.utils import negative_sampling, to_undirected

sys.path.insert(0, str(Path(__file__).resolve().parent))
from phaseA_rerun import (  # noqa: E402
    CONFIG, FEATURE_SETS, MODEL_MAP, REPO, DotDecoder, bce_loss, biomni_matrix, bh_fdr,
    build_graph, esm_matrix, eval_auc_ap, heuristic_scores, node2vec_from_train_edges, set_global_seed,
)
from scipy import stats  # noqa: E402


class MLPLinkPredictor(DotDecoder):
    """Graph-free control: same capacity, ignores edge_index."""

    def __init__(self, in_dim, hidden_dim, out_dim, dropout):
        super().__init__()
        self.lin1 = nn.Linear(in_dim, hidden_dim)
        self.lin2 = nn.Linear(hidden_dim, out_dim)
        self.dropout = dropout

    def encode(self, x, edge_index):
        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=self.dropout, training=self.training)
        return self.lin2(x)


MODELS = {**MODEL_MAP, "MLP (no graph)": MLPLinkPredictor}


def endpoint_weights(pos, pool, side):
    counts = np.bincount(pos[side].numpy(), minlength=int(pool.max()) + 1)[pool].astype(float)
    return counts / counts.sum() if counts.sum() > 0 else None


def sample_negatives(rng, src_pool, dst_pool, n, forbidden, p_src=None, p_dst=None):
    out = set()
    while len(out) < n:
        u = int(rng.choice(src_pool, p=p_src))
        v = int(rng.choice(dst_pool, p=p_dst))
        if u == v:
            continue
        key = (min(u, v), max(u, v))
        if key in forbidden or key in out:
            continue
        out.add(key)
    return torch.tensor(sorted(out), dtype=torch.long).t()


def orient(e, role, held):
    """Put the held-out endpoint in row 0 so endpoint weights are side-specific."""
    flip = role[e[0].numpy()] != held
    e = e.clone()
    e[:, flip] = e.flip(0)[:, flip]
    return e


def make_cold_split(edge_index, num_nodes, connected, seed, neg_mode):
    rng = np.random.default_rng(seed)
    perm = rng.permutation(connected)
    n_hold = int(round(0.1 * len(connected)))
    val_nodes, test_nodes = perm[:n_hold], perm[n_hold:2 * n_hold]
    role = np.zeros(num_nodes, dtype=int)  # 0 train (incl. STRING-isolated), 1 val, 2 test
    role[val_nodes], role[test_nodes] = 1, 2

    und = edge_index[:, edge_index[0] < edge_index[1]]
    r0, r1 = role[und[0].numpy()], role[und[1].numpy()]
    tt = und[:, (r0 == 0) & (r1 == 0)]
    vt = und[:, ((r0 == 1) & (r1 == 0)) | ((r0 == 0) & (r1 == 1))]
    xt = und[:, ((r0 == 2) & (r1 == 0)) | ((r0 == 0) & (r1 == 2))]
    xx = und[:, (r0 == 2) & (r1 == 2)]

    vt, xt = orient(vt, role, 1), orient(xt, role, 2)
    forbidden = set(map(tuple, und.t().tolist()))
    train_pool = np.where(role == 0)[0]
    train_mp = to_undirected(tt, num_nodes=num_nodes)

    def neg(pos, src_pool, dst_pool, n):
        if neg_mode == "uniform":
            return sample_negatives(rng, src_pool, dst_pool, n, forbidden)
        both = torch.cat([pos, pos.flip(0)], dim=1) if src_pool is dst_pool else pos
        return sample_negatives(rng, src_pool, dst_pool, n, forbidden,
                                endpoint_weights(both, src_pool, 0), endpoint_weights(both, dst_pool, 1))

    if neg_mode == "uniform":
        tr_neg = negative_sampling(train_mp, num_nodes=num_nodes, num_neg_samples=tt.shape[1])
    else:
        tr_neg = neg(tt, train_pool, train_pool, tt.shape[1])
    splits = {
        "train": (tt, tr_neg),
        "val": (vt, neg(vt, val_nodes, train_pool, vt.shape[1])),
        "test_C2": (xt, neg(xt, test_nodes, train_pool, xt.shape[1])),
        "test_C3": (xx, neg(xx, test_nodes, test_nodes, max(xx.shape[1], 1))),
    }
    return train_mp, splits, role


def as_data(x, mp, pos, neg):
    return Data(x=x, edge_index=mp,
                edge_label_index=torch.cat([pos, neg], dim=1),
                edge_label=torch.cat([torch.ones(pos.shape[1]), torch.zeros(neg.shape[1])]))


def train_eval(model_name, X, mp, splits, seed):
    set_global_seed(seed)
    tr = as_data(X, mp, *splits["train"])
    va = as_data(X, mp, *splits["val"])
    tests = {k: as_data(X, mp, *splits[k]) for k in ("test_C2", "test_C3")}
    model = MODELS[model_name](X.shape[1], CONFIG["hidden_dim"], CONFIG["out_dim"], CONFIG["dropout"])
    opt = torch.optim.Adam(model.parameters(), lr=CONFIG["lr"])
    best_val, best_state, best_epoch, no_improve = 0.0, None, 0, 0
    for epoch in range(1, CONFIG["max_epochs"] + 1):
        model.train()
        opt.zero_grad()
        loss = bce_loss(model(tr.x, tr.edge_index, tr.edge_label_index), tr.edge_label)
        loss.backward()
        opt.step()
        val_auc, _, _ = eval_auc_ap(model, va)
        if val_auc > best_val + 1e-4:
            best_val, best_epoch, no_improve = val_auc, epoch, 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            no_improve += 1
            if no_improve >= CONFIG["patience"]:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    out = {"best_val_auc": best_val, "best_epoch": best_epoch}
    for k, d in tests.items():
        auc, ap, sat = eval_auc_ap(model, d)
        out[f"{k}_auc"], out[f"{k}_ap"] = auc, ap
    return out


def score_baseline(scores, pos, neg):
    y = np.r_[np.ones(pos.shape[1]), np.zeros(neg.shape[1])]
    return roc_auc_score(y, scores), average_precision_score(y, scores)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--esm", type=Path, default=CONFIG["esm_path"])
    ap.add_argument("--neg", choices=["uniform", "degree"], default="uniform")
    args = ap.parse_args()
    out_dir = REPO / args.out if not args.out.is_absolute() else args.out
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)

    G, biomni_df = build_graph()
    nodes = list(G.nodes)
    idx = {n: i for i, n in enumerate(nodes)}
    N = len(nodes)
    X_biomni = torch.tensor(biomni_matrix(biomni_df, nodes, "fixed")[0], dtype=torch.float)
    X_esm = torch.tensor(esm_matrix(nodes, args.esm))
    edge_index = to_undirected(
        torch.tensor([[idx[u], idx[v]] for u, v in G.edges()], dtype=torch.long).t(), num_nodes=N)
    connected = np.array([idx[n] for n in nodes if G.degree(n) > 0])
    esm_unit = F.normalize(X_esm, dim=1)
    print(f"nodes={N} connected={len(connected)} ESM dim={X_esm.shape[1]} ({args.esm.name})", flush=True)

    records = []
    for seed in args.seeds:
        t0 = time.time()
        mp, splits, role = make_cold_split(edge_index, N, connected, seed, args.neg)
        sizes = {k: int(v[0].shape[1]) for k, v in splits.items()}
        print(f"=== seed {seed} | positives {sizes}", flush=True)
        n2v = node2vec_from_train_edges(N, mp, seed)
        blocks = {"Biomni": X_biomni, "ESM": X_esm, "Node2Vec": n2v}

        for fs in FEATURE_SETS:
            X = torch.cat([blocks[p.strip()] for p in fs.replace(" only", "").split("+")], dim=1)
            for m in MODELS:
                r = train_eval(m, X, mp, splits, seed)
                records.append({"seed": seed, "feature_set": fs, "model": m, **sizes, **r})
                print(f"  {fs:<26s} {m:<15s} val={r['best_val_auc']:.3f} C2={r['test_C2_auc']:.3f} "
                      f"C3={r['test_C3_auc']:.3f}", flush=True)

        for k in ("test_C2", "test_C3"):
            pos, neg = splits[k]
            pairs = torch.cat([pos, neg], dim=1)
            base = heuristic_scores(N, mp, pairs)
            base["ESM cosine"] = (esm_unit[pairs[0]] * esm_unit[pairs[1]]).sum(1).numpy()
            # one-sided degree: the held-out endpoint has degree 0, so score by the partner's degree
            deg = torch.bincount(mp[0], minlength=N).float()
            base["Partner degree"] = (deg[pairs[0]] + deg[pairs[1]]).numpy()
            for name, s in base.items():
                auc, apv = score_baseline(s, pos, neg)
                rec = next((r for r in records if r["seed"] == seed and r["model"] == name), None)
                if rec is None:
                    rec = {"seed": seed, "feature_set": "Baseline (no training)", "model": name, **sizes}
                    records.append(rec)
                rec[f"{k}_auc"], rec[f"{k}_ap"] = auc, apv
        print(f"  baselines C2: " + ", ".join(
            f"{r['model']}={r['test_C2_auc']:.3f}" for r in records
            if r["seed"] == seed and r["feature_set"] == "Baseline (no training)"), flush=True)
        pd.DataFrame(records).to_csv(out_dir / "results_per_seed.csv", index=False)
        print(f"  seed {seed} done in {time.time() - t0:.0f}s", flush=True)

    res = pd.DataFrame(records)
    summ = res.groupby(["feature_set", "model"]).agg(
        n=("test_C2_auc", "size"),
        C2_auc_mean=("test_C2_auc", "mean"), C2_auc_std=("test_C2_auc", "std"),
        C2_ap_mean=("test_C2_ap", "mean"), C2_ap_std=("test_C2_ap", "std"),
        C3_auc_mean=("test_C3_auc", "mean"), C3_auc_std=("test_C3_auc", "std"),
    ).reset_index().sort_values("C2_auc_mean", ascending=False).reset_index(drop=True)
    summ.to_csv(out_dir / "summary.csv", index=False)

    piv = res.pivot_table(index="seed", columns=["feature_set", "model"], values="test_C2_auc")
    best = tuple(summ.loc[0, ["feature_set", "model"]])
    rows = []
    for col in piv.columns:
        if col == best:
            continue
        d = (piv[best] - piv[col]).values
        rows.append({"best": " | ".join(best), "comp_feature_set": col[0], "comp_model": col[1],
                     "mean_diff_auc": d.mean(), "paired_t_p": stats.ttest_rel(piv[best], piv[col]).pvalue,
                     "wilcoxon_p": stats.wilcoxon(d).pvalue if np.any(d != 0) else 1.0})
    st = pd.DataFrame(rows)
    st["paired_t_p_bh"], st["wilcoxon_p_bh"] = bh_fdr(st["paired_t_p"]), bh_fdr(st["wilcoxon_p"])
    st.sort_values("mean_diff_auc").to_csv(out_dir / "statistical_tests.csv", index=False)
    (out_dir / "run_manifest.json").write_text(json.dumps(
        {"seeds": args.seeds, "esm_file": str(args.esm), "neg_mode": args.neg, "holdout_frac_val": 0.1, "holdout_frac_test": 0.1,
         "config": {k: str(v) for k, v in CONFIG.items()}}, indent=2))
    print(summ[["feature_set", "model", "C2_auc_mean", "C2_auc_std", "C3_auc_mean"]].to_string())


if __name__ == "__main__":
    main()
