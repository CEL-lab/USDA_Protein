"""Phase A rerun of the repeated-split link-prediction benchmark.

Mirrors Code/revised_link_prediction_experiment.ipynb (same graph construction,
models, RandomLinkSplit protocol, per-split Node2Vec refit, early stopping), with
the following changes, controlled by --mode:

  raw    reproduces the notebook exactly (unscaled Biomni features) so the
         degenerate AUC=0.5 GCN/GAT results can be checked.
  fixed  Biomni descriptors are log-transformed where heavy-tailed, molecular
         weight is dropped (collinear with length), and all Biomni columns are
         z-scored before use.

Additions in both modes: topological heuristic baselines on the same splits,
per-split edge indices saved for release, and BH-FDR-corrected paired tests.

Usage (from repo root):
    python Code/phaseA_rerun.py --mode raw   --seeds 42 43 44 45 46 --out Results/phaseA_raw
    python Code/phaseA_rerun.py --mode fixed --seeds 0 1 2 3 4 5 6 7 8 9 --out Results/phaseA_fixed
"""

import argparse
import gzip
import json
import os
import platform
import random
import time
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
import scipy
import sklearn
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric
from scipy import stats
from sklearn.metrics import average_precision_score, roc_auc_score
from torch_geometric.data import Data
from torch_geometric.nn import GATConv, GCNConv, SAGEConv
from torch_geometric.transforms import RandomLinkSplit
from torch_geometric.utils import to_undirected

REPO = Path(__file__).resolve().parents[1]
DATA = REPO / "Data"

CONFIG = {
    "string_path": DATA / "string_interactions.tsv",
    "mapping_path": DATA / "string_mapping.tsv",
    "biomni_path": DATA / "complete_vaccine_analysis_all_3257_proteins.csv",
    "esm_path": DATA / "esm_embeddings_esm2_t12_35M_UR50D_all.npz",
    "num_val": 0.1,
    "num_test": 0.1,
    "max_epochs": 60,
    "patience": 12,
    "lr": 1e-3,
    "hidden_dim": 128,
    "out_dim": 64,
    "dropout": 0.5,
    "node2vec_dim": 128,
    "n2v_walk_length": 10,
    "n2v_num_walks": 80,
    "n2v_window": 5,
    "n2v_epochs": 5,
    "n2v_workers": 1,  # single worker so Word2Vec is deterministic for a given seed
}

FEATURE_SETS = [
    "Biomni only",
    "ESM only",
    "Node2Vec only",
    "Biomni + ESM",
    "Biomni + Node2Vec",
    "ESM + Node2Vec",
    "Biomni + ESM + Node2Vec",
]
MODELS = ["GCN", "GAT", "GraphSAGE"]
HEURISTICS = ["Common neighbors", "Adamic-Adar", "Resource allocation", "Preferential attachment"]

# Graph source, read by build_graph(). "string_web": the original STRING web export, built from
# the 2,000 uploaded sequences. "v12_full": all 3,257 proteins mapped to STRING v12
# (Code/map_to_string_v12.py), links with combined score >= 400. Set via the USDA_GRAPH variable.
GRAPH_SOURCE = os.environ.get("USDA_GRAPH", "string_web")
STRING_V12_DIR = Path(os.environ.get("STRING_V12_DIR", Path.home() / ".cache" / "string_v12"))
V12_MIN_SCORE = 400

RAW_BIOMNI_COLS = [
    "signal_peptide_strength",
    "number_of_tm_domains",
    "cysteine_content",
    "gravy_hydrophobicity",
    "protein_length",
    "molecular_weight_da",
    "instability_index",
    "isoelectric_point",
    "vaccine_score",
]


def set_global_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


# --------------------------------------------------------------------------
# Data
# --------------------------------------------------------------------------
def build_graph(mapping="best", source=None):
    """STRING graph relabelled to proteome ids.

    mapping="best": each STRING protein is assigned to the proteome protein with the
    highest bitscore (ties: higher identity). mapping="last" reproduces the original
    notebook, where dict(zip(preferredName, queryItem)) kept whichever row came last.
    source: "string_web" (default, the original export) or "v12_full" (see GRAPH_SOURCE).
    """
    source = source or GRAPH_SOURCE
    if source == "v12_full":
        mapping_df = pd.read_csv(REPO / "Data" / "string_v12_mapping_all.tsv", sep="\t")
        mapping_df = mapping_df[mapping_df["accepted"]]
        links = pd.read_csv(gzip.open(STRING_V12_DIR / "1041826.protein.links.detailed.v12.0.txt.gz"), sep=" ",
                            usecols=["protein1", "protein2", "combined_score"])
        links = links[(links["combined_score"] >= V12_MIN_SCORE) & (links["protein1"] < links["protein2"])]
        string_df = pd.DataFrame({"a": links["protein1"], "b": links["protein2"],
                                  "w": links["combined_score"] / 1000.0})
        best = mapping_df.sort_values(["bitscore", "identity"], ascending=False).drop_duplicates("stringId")
        relabel = dict(zip(best["stringId"], best["queryItem"]))
        string_df = string_df[string_df["a"].isin(relabel) & string_df["b"].isin(relabel)]
    elif source != "string_web":
        raise ValueError(f"unknown graph source: {source}")
    else:
        string_df = pd.read_csv(CONFIG["string_path"], sep="\t")
        mapping_df = pd.read_csv(CONFIG["mapping_path"], sep="\t")
        if mapping == "last":
            string_df = string_df[["#node1", "node2", "combined_score"]]
            relabel = dict(zip(mapping_df["preferredName"], mapping_df["queryItem"]))
        elif mapping == "best":
            string_df = string_df[["node1_string_id", "node2_string_id", "combined_score"]]
            best = mapping_df.sort_values(["bitscore", "identity"], ascending=False).drop_duplicates("stringId")
            relabel = dict(zip(best["stringId"], best["queryItem"]))
        else:
            raise ValueError(f"unknown mapping rule: {mapping}")

    G = nx.Graph()
    for a, b, w in string_df.itertuples(index=False):
        G.add_edge(a, b, weight=float(w))
    G = nx.relabel_nodes(G, relabel, copy=True)

    biomni_df = pd.read_csv(CONFIG["biomni_path"]).set_index("protein_id")
    for pid in biomni_df.index:
        if pid not in G:
            G.add_node(pid)
    return G, biomni_df


def biomni_matrix(biomni_df, nodes, mode):
    df = biomni_df.reindex(nodes)
    if df[RAW_BIOMNI_COLS].isna().any().any():
        raise ValueError("Missing Biomni rows for some graph nodes")
    if mode == "raw":
        return df[RAW_BIOMNI_COLS].astype(float).values, list(RAW_BIOMNI_COLS)

    out = pd.DataFrame(index=df.index)
    out["signal_peptide_strength"] = df["signal_peptide_strength"]
    out["log1p_tm_domains"] = np.log1p(df["number_of_tm_domains"])
    out["cysteine_content"] = df["cysteine_content"]
    out["gravy_hydrophobicity"] = df["gravy_hydrophobicity"]
    out["log_protein_length"] = np.log(df["protein_length"])
    out["instability_index"] = df["instability_index"]
    out["isoelectric_point"] = df["isoelectric_point"]
    out["vaccine_score"] = df["vaccine_score"]
    X = out.values.astype(float)
    # Node attributes are label-free, so standardizing over all nodes does not leak edge labels.
    X = (X - X.mean(axis=0)) / X.std(axis=0)
    return X, list(out.columns)


def esm_matrix(nodes, path):
    npz = np.load(path)
    missing = [n for n in nodes if n not in npz.files]
    if missing:
        raise ValueError(f"{len(missing)} nodes have no ESM embedding")
    return np.vstack([npz[n] for n in nodes]).astype(np.float32)


# --------------------------------------------------------------------------
# Node2Vec (p = q = 1, i.e. the karateclub default) refit on training edges
# --------------------------------------------------------------------------
def node2vec_from_train_edges(num_nodes, train_edge_index, seed):
    from gensim.models import Word2Vec

    rng = random.Random(seed)
    adj = [[] for _ in range(num_nodes)]
    for u, v in train_edge_index.t().tolist():
        adj[u].append(v)
    nodes = list(range(num_nodes))
    walks = []
    for _ in range(CONFIG["n2v_num_walks"]):
        rng.shuffle(nodes)
        for start in nodes:
            walk = [start]
            for _ in range(CONFIG["n2v_walk_length"] - 1):
                nbrs = adj[walk[-1]]
                if not nbrs:
                    break
                walk.append(rng.choice(nbrs))
            walks.append([str(x) for x in walk])

    w2v = Word2Vec(
        sentences=walks,
        vector_size=CONFIG["node2vec_dim"],
        window=CONFIG["n2v_window"],
        min_count=1,
        sg=1,
        workers=CONFIG["n2v_workers"],
        epochs=CONFIG["n2v_epochs"],
        seed=seed,
    )
    return torch.tensor(np.vstack([w2v.wv[str(i)] for i in range(num_nodes)]), dtype=torch.float)


# --------------------------------------------------------------------------
# Models (identical to the notebook)
# --------------------------------------------------------------------------
class DotDecoder(nn.Module):
    def decode(self, z, edge_label_index):
        return (z[edge_label_index[0]] * z[edge_label_index[1]]).sum(dim=-1)

    def forward(self, x, edge_index, edge_label_index):
        return self.decode(self.encode(x, edge_index), edge_label_index)


class GCNLinkPredictor(DotDecoder):
    def __init__(self, in_dim, hidden_dim, out_dim, dropout):
        super().__init__()
        self.conv1 = GCNConv(in_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, out_dim)
        self.dropout = dropout

    def encode(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, p=self.dropout, training=self.training)
        return self.conv2(x, edge_index)


class GATLinkPredictor(DotDecoder):
    def __init__(self, in_dim, hidden_dim, out_dim, dropout, heads=4):
        super().__init__()
        self.conv1 = GATConv(in_dim, hidden_dim // heads, heads=heads, dropout=dropout)
        self.conv2 = GATConv(hidden_dim, out_dim, heads=1, dropout=dropout)
        self.dropout = dropout

    def encode(self, x, edge_index):
        x = F.elu(self.conv1(x, edge_index))
        x = F.dropout(x, p=self.dropout, training=self.training)
        return self.conv2(x, edge_index)


class GraphSAGELinkPredictor(DotDecoder):
    def __init__(self, in_dim, hidden_dim, out_dim, dropout):
        super().__init__()
        self.conv1 = SAGEConv(in_dim, hidden_dim)
        self.conv2 = SAGEConv(hidden_dim, out_dim)
        self.dropout = dropout

    def encode(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, p=self.dropout, training=self.training)
        return self.conv2(x, edge_index)


MODEL_MAP = {"GCN": GCNLinkPredictor, "GAT": GATLinkPredictor, "GraphSAGE": GraphSAGELinkPredictor}
bce_loss = nn.BCEWithLogitsLoss()


@torch.no_grad()
def eval_auc_ap(model, data):
    model.eval()
    logits = model(data.x, data.edge_index, data.edge_label_index)
    y_true = data.edge_label.numpy()
    y_score = torch.sigmoid(logits).numpy()
    if not np.isfinite(y_score).all():
        return 0.5, 0.5, 1.0
    # share of scores saturated at exactly 0 or 1 flags numerical collapse
    saturated = float(np.mean((y_score == 0.0) | (y_score == 1.0)))
    return roc_auc_score(y_true, y_score), average_precision_score(y_true, y_score), saturated


def run_single_model(model_name, in_dim, tr, va, te, seed):
    set_global_seed(seed)
    model = MODEL_MAP[model_name](in_dim, CONFIG["hidden_dim"], CONFIG["out_dim"], CONFIG["dropout"])
    optimizer = torch.optim.Adam(model.parameters(), lr=CONFIG["lr"])
    best = {"best_val_auc": 0.0, "test_auc": 0.0, "test_ap": 0.0, "best_epoch": 0, "test_saturated": np.nan}
    no_improve = 0
    for epoch in range(1, CONFIG["max_epochs"] + 1):
        model.train()
        optimizer.zero_grad()
        loss = bce_loss(model(tr.x, tr.edge_index, tr.edge_label_index), tr.edge_label.float())
        loss.backward()
        optimizer.step()

        val_auc, _, _ = eval_auc_ap(model, va)
        if val_auc > best["best_val_auc"] + 1e-4:
            test_auc, test_ap, sat = eval_auc_ap(model, te)
            best.update(best_val_auc=val_auc, test_auc=test_auc, test_ap=test_ap,
                        best_epoch=epoch, test_saturated=sat)
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= CONFIG["patience"]:
                break
    return best


# --------------------------------------------------------------------------
# Heuristic baselines, scored on the same message-passing graph the GNN sees
# --------------------------------------------------------------------------
def heuristic_scores(num_nodes, edge_index, pairs):
    g = nx.Graph()
    g.add_nodes_from(range(num_nodes))
    g.add_edges_from(edge_index.t().tolist())
    ebunch = [tuple(p) for p in pairs.t().tolist()]
    deg = dict(g.degree())
    out = {
        "Common neighbors": np.array([len(list(nx.common_neighbors(g, u, v))) for u, v in ebunch], dtype=float),
        "Adamic-Adar": np.array([s for _, _, s in nx.adamic_adar_index(g, ebunch)], dtype=float),
        "Resource allocation": np.array([s for _, _, s in nx.resource_allocation_index(g, ebunch)], dtype=float),
        "Preferential attachment": np.array([deg[u] * deg[v] for u, v in ebunch], dtype=float),
    }
    return out


# --------------------------------------------------------------------------
# Experiment
# --------------------------------------------------------------------------
def run_split(seed, full_data, X_biomni, X_esm, out_dir):
    set_global_seed(seed)
    splitter = RandomLinkSplit(
        num_val=CONFIG["num_val"],
        num_test=CONFIG["num_test"],
        is_undirected=True,
        add_negative_train_samples=True,
        split_labels=False,
    )
    tr0, va0, te0 = splitter(full_data)
    np.savez_compressed(
        out_dir / "splits" / f"split_seed{seed}.npz",
        train_edge_index=tr0.edge_index.numpy(),
        train_edge_label_index=tr0.edge_label_index.numpy(), train_edge_label=tr0.edge_label.numpy(),
        val_edge_label_index=va0.edge_label_index.numpy(), val_edge_label=va0.edge_label.numpy(),
        test_edge_index=te0.edge_index.numpy(),
        test_edge_label_index=te0.edge_label_index.numpy(), test_edge_label=te0.edge_label.numpy(),
    )

    t0 = time.time()
    n2v = node2vec_from_train_edges(full_data.num_nodes, tr0.edge_index, seed)
    print(f"  Node2Vec refit on train edges: {time.time() - t0:.1f}s", flush=True)

    blocks = {"Biomni": X_biomni, "ESM": X_esm, "Node2Vec": n2v}
    records = []
    for fs in FEATURE_SETS:
        X = torch.cat([blocks[p.strip()] for p in fs.replace(" only", "").split("+")], dim=1)
        tr, va, te = tr0.clone(), va0.clone(), te0.clone()
        tr.x = va.x = te.x = X
        for m in MODELS:
            best = run_single_model(m, X.shape[1], tr, va, te, seed)
            records.append({"seed": seed, "feature_set": fs, "model": m, **best})
            print(f"  {fs:<26s} {m:<10s} val={best['best_val_auc']:.3f} test={best['test_auc']:.3f} "
                  f"ap={best['test_ap']:.3f} ep={best['best_epoch']} sat={best['test_saturated']:.2f}", flush=True)

    y = te0.edge_label.numpy()
    for name, s in heuristic_scores(full_data.num_nodes, te0.edge_index, te0.edge_label_index).items():
        records.append({"seed": seed, "feature_set": "Topology heuristic", "model": name,
                        "best_val_auc": np.nan, "test_auc": roc_auc_score(y, s),
                        "test_ap": average_precision_score(y, s), "best_epoch": 0, "test_saturated": np.nan})
        print(f"  {'Topology heuristic':<26s} {name:<24s} test={records[-1]['test_auc']:.3f}", flush=True)
    return records


def bh_fdr(p):
    p = np.asarray(p, dtype=float)
    order = np.argsort(p)
    ranked = p[order] * len(p) / (np.arange(len(p)) + 1)
    adj = np.minimum.accumulate(ranked[::-1])[::-1].clip(max=1.0)
    out = np.empty_like(adj)
    out[order] = adj
    return out


def summarize(results, out_dir):
    g = results.groupby(["feature_set", "model"])
    summary = g.agg(
        n=("test_auc", "size"),
        auc_mean=("test_auc", "mean"), auc_std=("test_auc", "std"),
        ap_mean=("test_ap", "mean"), ap_std=("test_ap", "std"),
        val_auc_mean=("best_val_auc", "mean"), val_auc_std=("best_val_auc", "std"),
        best_epoch_mean=("best_epoch", "mean"), test_saturated_mean=("test_saturated", "mean"),
    ).reset_index()
    tcrit = stats.t.ppf(0.975, summary["n"] - 1)
    summary["auc_ci95"] = tcrit * summary["auc_std"] / np.sqrt(summary["n"])
    summary["ap_ci95"] = tcrit * summary["ap_std"] / np.sqrt(summary["n"])
    summary = summary.sort_values(["auc_mean", "ap_mean"], ascending=False).reset_index(drop=True)
    summary.to_csv(out_dir / "summary.csv", index=False)

    pivot = results.pivot_table(index="seed", columns=["feature_set", "model"], values="test_auc")
    best_key = tuple(summary.loc[0, ["feature_set", "model"]])
    rows = []
    for col in pivot.columns:
        if col == best_key:
            continue
        x, y = pivot[best_key].values, pivot[col].values
        d = x - y
        t_p = stats.ttest_rel(x, y).pvalue if np.std(d) > 0 else 0.0
        w_p = stats.wilcoxon(x, y).pvalue if np.any(d != 0) else 1.0
        rows.append({"best_feature_set": best_key[0], "best_model": best_key[1],
                     "comp_feature_set": col[0], "comp_model": col[1],
                     "mean_diff_auc": d.mean(), "sd_diff_auc": d.std(ddof=1),
                     "paired_t_p": t_p, "wilcoxon_p": w_p, "n_pairs": len(d)})
    st = pd.DataFrame(rows)
    st["paired_t_p_bh"] = bh_fdr(st["paired_t_p"])
    st["wilcoxon_p_bh"] = bh_fdr(st["wilcoxon_p"])
    st.sort_values("mean_diff_auc").to_csv(out_dir / "statistical_tests.csv", index=False)
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["raw", "fixed"], required=True)
    ap.add_argument("--seeds", type=int, nargs="+", required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--esm", type=Path, default=CONFIG["esm_path"], help="npz of per-protein ESM embeddings")
    args = ap.parse_args()

    out_dir = (REPO / args.out) if not args.out.is_absolute() else args.out
    (out_dir / "splits").mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)

    G, biomni_df = build_graph("last" if args.mode == "raw" else "best")
    nodes = list(G.nodes)
    node_to_idx = {n: i for i, n in enumerate(nodes)}
    Xb, biomni_cols = biomni_matrix(biomni_df, nodes, args.mode)
    X_biomni = torch.tensor(Xb, dtype=torch.float)
    X_esm = torch.tensor(esm_matrix(nodes, args.esm))

    edge_index = torch.tensor([[node_to_idx[u], node_to_idx[v]] for u, v in G.edges()], dtype=torch.long).t()
    edge_index = to_undirected(edge_index, num_nodes=len(nodes))
    full_data = Data(x=torch.zeros((len(nodes), 1)), edge_index=edge_index, num_nodes=len(nodes))

    n_iso = sum(1 for n in nodes if G.degree(n) == 0)
    r_len_mw = np.corrcoef(biomni_df["protein_length"], biomni_df["molecular_weight_da"])[0, 1]
    print(f"Mode={args.mode} | nodes={len(nodes)} undirected edges={G.number_of_edges()} isolated={n_iso}")
    print(f"Biomni cols={biomni_cols}")
    print(f"Biomni column SDs: {np.round(Xb.std(axis=0), 3).tolist()}")
    print(f"corr(protein_length, molecular_weight)={r_len_mw:.4f} | ESM dim={X_esm.shape[1]}")
    pd.Series(nodes, name="protein_id").to_csv(out_dir / "node_order.csv", index=False)

    records = []
    for seed in args.seeds:
        print(f"=== seed {seed} ===", flush=True)
        t0 = time.time()
        records.extend(run_split(seed, full_data, X_biomni, X_esm, out_dir))
        pd.DataFrame(records).to_csv(out_dir / "results_per_seed.csv", index=False)
        print(f"  seed {seed} done in {time.time() - t0:.0f}s", flush=True)

    summary = summarize(pd.DataFrame(records), out_dir)
    print(summary[["feature_set", "model", "auc_mean", "auc_std", "ap_mean", "test_saturated_mean"]].to_string())

    manifest = {
        "graph_source": GRAPH_SOURCE, "mode": args.mode, "mapping": "last" if args.mode == "raw" else "best", "seeds": args.seeds, "config": {k: str(v) for k, v in CONFIG.items()},
        "biomni_cols": biomni_cols, "n_nodes": len(nodes), "n_edges": G.number_of_edges(), "n_isolated": n_iso,
        "esm_dim": int(X_esm.shape[1]), "esm_file": str(args.esm), "python": platform.python_version(), "platform": platform.platform(),
        "torch": torch.__version__, "torch_geometric": torch_geometric.__version__,
        "numpy": np.__version__, "scipy": scipy.__version__, "sklearn": sklearn.__version__,
        "pandas": pd.__version__, "networkx": nx.__version__,
    }
    (out_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
