"""External validation of the Phase C candidate scores on proteins never sent to STRING.

Only 2,000 of the 3,257 proteins were uploaded to STRING (web limit), so 1,257 of
the 1,650 "isolated" proteins had never been queried. Code/map_to_string_v12.py maps
all proteins to STRING v12 (taxon 1041826). For the never-submitted isolated
proteins that now map, their STRING v12 links (combined score >= 400, the threshold of
the original export) to other candidate proteins are interactions that no model saw.
The Phase C ensemble scores were computed before these links were known, so they are
scored here as a held-out test.

Partners are restricted to the Phase C candidate universe (connected proteins and the
other isolated proteins); pairs of identical sequences or the same STRING protein are
skipped.

Usage (from repo root):
    python Code/phaseD_external_validation.py
"""

import argparse
import gzip
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
from phaseA_rerun import REPO, build_graph, esm_matrix  # noqa: E402
from phaseC_rescore_candidates import TIERS, sequence_groups, tier_pair  # noqa: E402

FRACTIONS = [0.001, 0.005, 0.01, 0.05]


def load_links(db, min_score):
    links = pd.read_csv(gzip.open(db / "1041826.protein.links.detailed.v12.0.txt.gz"), sep=" ",
                        usecols=["protein1", "protein2", "combined_score"])
    links = links[links["combined_score"] >= min_score]
    return set(zip(links["protein1"], links["protein2"]))


def external_test_set(nodes, deg, group, iso, con, db, min_score=400):
    """Never-submitted isolated proteins with >=1 STRING v12 link to a candidate partner.

    Partners are indexed in the order con + iso (the Phase C score layout). Returns a list of
    {"protein", "iso_index", "keep", "y"} plus the counts of never-submitted and newly mapped
    proteins.
    """
    # proteome protein -> STRING v12 id: connected proteins keep the mapping that defined the graph,
    # all other proteins use the phmmer best hit
    web = pd.read_csv(REPO / "Data" / "string_mapping.tsv", sep="\t")
    web_best = web.sort_values(["bitscore", "identity"], ascending=False).drop_duplicates("stringId")
    sid = {q: s for q, s in zip(web_best["queryItem"], web_best["stringId"]) if deg[q] > 0}
    ph = pd.read_csv(REPO / "Data" / "string_v12_mapping_all.tsv", sep="\t")
    ph = ph[ph["accepted"]]
    for q, s in zip(ph["queryItem"], ph["stringId"]):
        sid.setdefault(q, s)
    submitted = {l[1:].split()[0] for l in open(REPO / "Data" / "top_2000_sequences.faa") if l.startswith(">")}
    links = load_links(db, min_score)

    partners_all = list(con) + list(iso)
    mapped = [(a, p) for a, p in enumerate(iso) if p not in submitted and p in sid]
    tests = []
    for a, p in mapped:
        keep = [j for j, q in enumerate(partners_all)
                if q != p and group[q] != group[p] and q in sid and sid[q] != sid[p]]
        y = np.array([(sid[p], sid[partners_all[j]]) in links for j in keep], dtype=int)
        if y.sum() > 0:
            tests.append({"protein": p, "iso_index": a, "keep": keep, "y": y})
    return tests, sum(p not in submitted for p in iso), len(mapped)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", type=Path, default=Path.home() / ".cache" / "string_v12")
    ap.add_argument("--min_score", type=int, default=400)
    ap.add_argument("--out", type=Path, default=Path("Results/phaseD_external_validation"))
    args = ap.parse_args()
    out = REPO / args.out
    out.mkdir(parents=True, exist_ok=True)

    G, biomni_df = build_graph()
    nodes = list(G.nodes)
    deg = dict(G.degree())
    tier = biomni_df["priority"].to_dict()
    group = dict(zip(nodes, sequence_groups(nodes)[0]))

    esm_unit = esm_matrix(nodes, REPO / "Data" / "esm_embeddings_esm2_t33_650M_UR50D_all.npz")
    esm_unit /= np.linalg.norm(esm_unit, axis=1, keepdims=True)
    eidx = {n: i for i, n in enumerate(nodes)}

    models = {}
    for label, d in [("Biomni + ESM (Phase C ensemble)", "phaseC_candidates"),
                     ("ESM only (Phase C ensemble)", "phaseC_candidates_esm_only")]:
        models[label] = np.load(REPO / "Results" / d / "ensemble_scores.npz", allow_pickle=True)
    ref = next(iter(models.values()))
    iso, con = list(ref["isolated"]), list(ref["connected"])
    partners_all = con + iso

    tests, n_never, n_mapped = external_test_set(nodes, deg, group, iso, con, args.db, args.min_score)
    print(f"isolated proteins never submitted to STRING: {n_never}; now mapped to STRING v12: {n_mapped}",
          flush=True)

    per_protein, pooled = [], {k: ([], []) for k in list(models) + ["ESM cosine", "Partner degree"]}
    true_pairs = []
    for t in tests:
        p, a, keep, y = t["protein"], t["iso_index"], t["keep"], t["y"]
        qs = [partners_all[j] for j in keep]
        true_pairs += [(p, q) for q, t in zip(qs, y) if t]
        scores = {}
        for label, sc in models.items():
            full = np.r_[sc["S_isolated_connected"][a], sc["S_isolated_isolated"][a]]
            scores[label] = full[keep]
        scores["ESM cosine"] = esm_unit[[eidx[q] for q in qs]] @ esm_unit[eidx[p]]
        scores["Partner degree"] = np.array([deg[q] for q in qs], dtype=float)
        rec = {"protein": p, "tier": tier[p], "n_candidates": len(y), "n_true": int(y.sum())}
        for label, s in scores.items():
            rec[f"{label} | AUC"] = roc_auc_score(y, s)
            top = np.argsort(-s, kind="stable")
            rec[f"{label} | P@5"] = y[top[:5]].mean()
            rec[f"{label} | hit@10"] = float(y[top[:10]].any())
            pooled[label][0].append(s)
            pooled[label][1].append(y)
        rec["random | P@5"] = y.mean()
        per_protein.append(rec)

    pp = pd.DataFrame(per_protein)
    pp.to_csv(out / "per_protein_validation.csv", index=False)
    print(f"proteins with >=1 recovered STRING v12 link to a candidate partner: {len(pp)}; "
          f"recovered links: {int(pp['n_true'].sum())}; base rate {pp['n_true'].sum() / pp['n_candidates'].sum():.4f}")

    rows = []
    base = pp["Biomni + ESM (Phase C ensemble) | AUC"]
    for label in pooled:
        s = np.concatenate(pooled[label][0]); y = np.concatenate(pooled[label][1])
        order = np.argsort(-s, kind="stable")
        rec = {"method": label, "per_protein_AUC_mean": pp[f"{label} | AUC"].mean(),
               "per_protein_AUC_median": pp[f"{label} | AUC"].median(),
               "per_protein_P@5": pp[f"{label} | P@5"].mean(),
               "per_protein_hit@10": pp[f"{label} | hit@10"].mean(), "pooled_AUC": roc_auc_score(y, s)}
        for f in FRACTIONS:
            k = max(1, int(round(f * len(s))))
            rec[f"pooled_precision_top{f:g}"] = y[order[:k]].mean()
        if label != "Biomni + ESM (Phase C ensemble)":
            rec["wilcoxon_p_vs_BiomniESM (per-protein AUC)"] = stats.wilcoxon(base, pp[f"{label} | AUC"]).pvalue
        rows.append(rec)
    rows.append({"method": "Random", "per_protein_AUC_mean": 0.5, "per_protein_P@5": pp["random | P@5"].mean(),
                 "pooled_AUC": 0.5,
                 **{f"pooled_precision_top{f:g}": pp["n_true"].sum() / pp["n_candidates"].sum() for f in FRACTIONS}})
    summ = pd.DataFrame(rows)
    summ.to_csv(out / "validation_summary.csv", index=False)
    print(summ.round(4).T.to_string())

    # tier pairs among the recovered (real) links vs all candidate pairs of the same proteins
    obs = pd.Series([tier_pair(tier[p], tier[q]) for p, q in true_pairs]).value_counts()
    pool = pd.Series(dtype=float)
    for p in pp["protein"]:
        for t in TIERS:
            n_t = sum(1 for q in partners_all if q != p and q in sid and tier[q] == t)
            key = tier_pair(tier[p], t)
            pool[key] = pool.get(key, 0) + n_t
    exp = pool / pool.sum() * obs.sum()
    tp = pd.DataFrame({"observed": obs, "expected": exp}).fillna(0)
    tp["obs_over_exp"] = tp["observed"] / tp["expected"]
    tp["binom_p"] = [stats.binomtest(int(o), int(obs.sum()), e / obs.sum()).pvalue for o, e in
                     zip(tp["observed"], tp["expected"])]
    tp.sort_values("obs_over_exp", ascending=False).to_csv(out / "tier_pairs_recovered_links.csv")
    print("\nTier pairs among recovered real links (O/E vs candidate pool):")
    print(tp.sort_values("obs_over_exp", ascending=False).round(3).to_string())


if __name__ == "__main__":
    main()
