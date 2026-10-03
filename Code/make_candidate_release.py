"""Release table of candidate partners for proteins without edges in the final network.

For each interaction-orphan protein (no edge in the final STRING v12 network) the five best
partners are listed under the two rankings reported in the manuscript:
  esm_nn   ESM-2 cosine nearest neighbours (pre-selected on internal splits)
  blend    0.5 * rowrank(model cosine) + 0.5 * rowrank(ESM-2 cosine)
Model cosine and the global model score are averaged over the ten P3 models
(Biomni + ESM-2 650M, MLP encoder, degree-matched cold-start splits). Identical
sequences are collapsed. Pfam, GO and Biomni annotations are attached.

Usage (from repo root):
    USDA_GRAPH=v12_full python Code/make_candidate_release.py
Writes Results/final_v12/candidate_partners_orphan_proteins.tsv
"""

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch_geometric.utils import to_undirected

sys.path.insert(0, str(Path(__file__).resolve().parent))
from phaseA_rerun import REPO, biomni_matrix, build_graph, esm_matrix  # noqa: E402
from phaseB_coldstart import make_cold_split  # noqa: E402
from phaseC_rescore_candidates import sequence_groups, train_model  # noqa: E402
from phaseE_rerank import row_pct, unit  # noqa: E402

TOP = 5


def main():
    if os.environ.get("USDA_GRAPH") != "v12_full":
        raise SystemExit("run with USDA_GRAPH=v12_full")
    torch.set_num_threads(4)
    G, biomni_df = build_graph()
    nodes = list(G.nodes)
    idx = {x: i for i, x in enumerate(nodes)}
    N = len(nodes)
    E = esm_matrix(nodes, REPO / "Data" / "esm_embeddings_esm2_t33_650M_UR50D_all.npz")
    X = torch.cat([torch.tensor(biomni_matrix(biomni_df, nodes, "fixed")[0], dtype=torch.float),
                   torch.tensor(E)], dim=1)
    ei = to_undirected(torch.tensor([[idx[u], idx[v]] for u, v in G.edges()]).t(), num_nodes=N)
    deg = np.array([G.degree(x) for x in nodes])
    connected, orphans = np.where(deg > 0)[0], np.where(deg == 0)[0]

    cos_sum = np.zeros((len(orphans), N))
    score_sum = np.zeros((len(orphans), N))
    for seed in range(10):
        mp, splits, _ = make_cold_split(ei, N, connected, seed, "degree")
        z = train_model(X, mp, splits, seed)[0].numpy()
        cos_sum += unit(z[orphans]) @ unit(z).T
        score_sum += 1 / (1 + np.exp(-(z[orphans] @ z.T)))
        print(f"model {seed} done", flush=True)
    cos_m, score_m = cos_sum / 10, score_sum / 10
    esm_cos = unit(E[orphans]) @ unit(E).T

    group = sequence_groups(nodes)[0]
    # exclude self and identical sequences, then rank
    mask = group[orphans][:, None] == group[None, :]
    esm_cos[mask] = -np.inf
    cos_m[mask] = -np.inf
    blend = 0.5 * row_pct(cos_m) + 0.5 * row_pct(esm_cos)
    blend[mask] = -np.inf

    ann = pd.read_csv(REPO / "Data" / "annotation" / "protein_annotation.tsv", sep="\t").fillna("").set_index("protein_id")
    tier = biomni_df["priority"]
    vs = biomni_df["vaccine_score"]
    rows = []
    for a, p in enumerate(orphans):
        for ranking, M in [("esm_nn", esm_cos), ("blend", blend)]:
            used, r = set(), 0
            for j in np.argsort(-M[a]):
                if group[j] in used or not np.isfinite(M[a, j]):
                    continue
                used.add(group[j])
                r += 1
                q = nodes[j]
                rows.append({"orphan_protein": nodes[p], "orphan_tier": tier[nodes[p]],
                             "orphan_vaccine_score": vs[nodes[p]],
                             "orphan_pfam": ann.loc[nodes[p], "pfam_architecture"],
                             "ranking": ranking, "rank": r, "partner": q, "partner_connected": bool(deg[j] > 0),
                             "partner_tier": tier[q], "partner_pfam": ann.loc[q, "pfam_architecture"],
                             "partner_pfam_desc": ann.loc[q, "pfam_descriptions"],
                             "esm_cosine": esm_cos[a, j], "model_cosine": cos_m[a, j],
                             "model_score": score_m[a, j], "blend_score": blend[a, j]})
                if r == TOP:
                    break
    out = pd.DataFrame(rows)
    # global confidence: mean model score of the orphan's top-5 blend partners, as a percentile across orphans
    conf = out[out["ranking"] == "blend"].groupby("orphan_protein")["model_score"].mean()
    out["orphan_confidence_percentile"] = out["orphan_protein"].map(conf.rank(pct=True))
    path = REPO / "Results" / "final_v12" / "candidate_partners_orphan_proteins.tsv"
    out.to_csv(path, sep="\t", index=False)
    print(f"{out['orphan_protein'].nunique()} orphan proteins, {len(out)} rows -> {path}")


if __name__ == "__main__":
    main()
