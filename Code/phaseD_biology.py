"""Biological evaluation of Biomni tiers and re-scored candidate links (reviewer comment 6).

Uses Pfam 38.2 annotations (Code/annotate_pfam.py) and the Phase C ensemble scores
(Code/phaseC_rescore_candidates.py). Vaccine-relevant protein families are fixed
a priori by Pfam accession (FAMILY_SETS below), not chosen after seeing results.

Analyses
  1. tier_validation.csv         family-set membership by Biomni tier (Fisher High vs
                                 rest, Spearman trend over the four tiers)
  2. functional_coherence.csv    share of pairs with a common Pfam family / clan / GO term:
                                 top predicted pairs vs random candidate pairs vs real
                                 STRING edges
  3. family_enrichment_top.csv   family sets among isolated proteins that appear in the
                                 top predictions vs all isolated proteins
  4. go_enrichment_top.csv       GO terms (pfam2go) of those proteins vs all isolated
  5. surface_isolated_partners.csv
                                 top partners of every isolated protein in a family set

Usage (from repo root):
    python Code/phaseD_biology.py
"""

import argparse
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))
from phaseA_rerun import REPO, bh_fdr, build_graph  # noqa: E402
from phaseC_rescore_candidates import TIERS, sequence_groups  # noqa: E402

FAMILY_SETS = {
    "T9SS cargo (CTD)": ["PF18962", "PF13585"],
    "Outer membrane protein": ["PF00691", "PF16961", "PF14905", "PF13568", "PF13505", "PF03938", "PF02321",
                               "PF01103", "PF19143", "PF07244", "PF11551", "PF03349", "PF19838", "PF03968",
                               "PF13488", "PF13525"],
    "TonB / SusCD uptake": ["PF07715", "PF00593", "PF07980", "PF12741", "PF12771", "PF14322", "PF03544"],
    "T9SS / gliding machinery": ["PF11751", "PF19572", "PF17116", "PF14349", "PF13573", "PF21602", "PF21601",
                                 "PF12080", "PF12081", "PF22827", "PF19841", "PF25594", "PF25593", "PF14109",
                                 "PF19937"],
    "Adhesin / surface repeat": ["PF10988", "PF07675", "PF13205", "PF00801", "PF18911", "PF00041", "PF13385",
                                 "PF05593"],
    "T6SS (type VI secretion)": ["PF17642", "PF17555", "PF17561", "PF17541"],
}
ANY = "Any surface/secretion family"


def load_annotations(db):
    hits = pd.read_csv(REPO / "Data" / "annotation" / "pfam_domain_hits.tsv", sep="\t")
    fams = hits.groupby("protein_id")["pfam_acc"].apply(set).to_dict()
    clans = pd.read_csv(db / "Pfam-A.clans.tsv.gz", sep="\t", header=None,
                        names=["pfam_acc", "clan_acc", "clan_name", "pfam_name", "pfam_desc"])
    clan_of = {a: (c if isinstance(c, str) else a) for a, c in zip(clans["pfam_acc"], clans["clan_acc"])}
    clan_sets = {p: {clan_of.get(a, a) for a in s} for p, s in fams.items()}
    go = pd.read_csv(REPO / "Data" / "annotation" / "protein_to_go.tsv", sep="\t")
    go_sets = go.groupby("protein_id")["go_term"].apply(set).to_dict()
    go_name = {}
    pat = re.compile(r"^Pfam:PF\d+ .* > GO:(.*) ; (GO:\d+)$")
    for line in open(db / "pfam2go"):
        m = pat.match(line.strip())
        if m:
            go_name[m.group(2)] = m.group(1)
    return fams, clan_sets, go_sets, go_name


def family_membership(nodes, fams):
    member = {}
    for name, accs in FAMILY_SETS.items():
        accs = set(accs)
        member[name] = np.array([bool(fams.get(n, set()) & accs) for n in nodes])
    member[ANY] = np.any(np.vstack([member[k] for k in FAMILY_SETS]), axis=0)
    return member


def tier_validation(nodes, tier, member):
    rows = []
    order = np.array([TIERS.index(t) for t in tier])
    for name, m in member.items():
        rec = {"family_set": name, "n_proteins": int(m.sum())}
        for t in TIERS:
            sel = tier == t
            rec[f"n_{t}"] = int(m[sel].sum())
            rec[f"frac_{t}"] = m[sel].mean()
        high = tier == "High"
        table = [[m[high].sum(), (~m[high]).sum()], [m[~high].sum(), (~m[~high]).sum()]]
        rec["odds_ratio_high_vs_rest"], rec["fisher_p"] = stats.fisher_exact(table)
        rec["spearman_rho_tier"], rec["spearman_p"] = stats.spearmanr(order, m.astype(int))
        rows.append(rec)
    df = pd.DataFrame(rows)
    df["fisher_p_bh"] = bh_fdr(df["fisher_p"])
    df["spearman_p_bh"] = bh_fdr(df["spearman_p"])
    return df


def coherence(pairs, fams, clan_sets, go_sets, member_of, label):
    """Fraction of pairs sharing a Pfam family / clan / GO term, among pairs where both are annotated."""
    rec = {"pair_set": label, "n_pairs": len(pairs)}
    both_pf = [(a, b) for a, b in pairs if a in fams and b in fams]
    both_go = [(a, b) for a, b in pairs if a in go_sets and b in go_sets]
    rec["n_both_pfam"] = len(both_pf)
    rec["share_pfam_family"] = np.mean([bool(fams[a] & fams[b]) for a, b in both_pf]) if both_pf else np.nan
    rec["share_pfam_clan"] = np.mean([bool(clan_sets[a] & clan_sets[b]) for a, b in both_pf]) if both_pf else np.nan
    rec["n_both_go"] = len(both_go)
    rec["share_go_term"] = np.mean([bool(go_sets[a] & go_sets[b]) for a, b in both_go]) if both_go else np.nan
    rec["both_surface_family"] = np.mean([member_of[a] and member_of[b] for a, b in pairs])
    rec["any_surface_family"] = np.mean([member_of[a] or member_of[b] for a, b in pairs])
    return rec, both_pf, both_go


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", type=Path, default=Path.home() / ".cache" / "pfam")
    ap.add_argument("--cand", type=Path, default=Path("Results/phaseC_candidates"))
    ap.add_argument("--top", type=int, default=2000)
    ap.add_argument("--out", type=Path, default=Path("Results/phaseD_biology"))
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    out = REPO / args.out
    out.mkdir(parents=True, exist_ok=True)
    cand = REPO / args.cand
    rng = np.random.default_rng(args.seed)

    G, biomni_df = build_graph()
    nodes = np.array(list(G.nodes))
    tier = biomni_df.reindex(nodes)["priority"].values
    deg = np.array([G.degree(n) for n in nodes])
    fams, clan_sets, go_sets, go_name = load_annotations(args.db)
    member = family_membership(nodes, fams)
    member_of = dict(zip(nodes, member[ANY]))
    seq_group, _ = sequence_groups(list(nodes))
    group_of = dict(zip(nodes, seq_group))

    # 1. Biomni tier validation against a-priori family sets
    tv = tier_validation(nodes, tier, member)
    tv.to_csv(out / "tier_validation.csv", index=False)
    print(tv[["family_set", "n_proteins", "frac_Very Low", "frac_Low", "frac_Medium", "frac_High",
              "odds_ratio_high_vs_rest", "fisher_p_bh", "spearman_rho_tier", "spearman_p_bh"]].round(4).to_string())

    # 2. functional coherence of predicted pairs
    ic = pd.read_csv(cand / "candidates_isolated_connected_capped.csv").head(args.top)
    ii = pd.read_csv(cand / "candidates_isolated_isolated_capped.csv").head(args.top)
    sc = np.load(cand / "ensemble_scores.npz", allow_pickle=True)
    iso, con = sc["isolated"], sc["connected"]

    def random_pairs(A, B, n, distinct_groups=True):
        out_pairs = []
        while len(out_pairs) < n:
            a, b = A[rng.integers(len(A))], B[rng.integers(len(B))]
            if a != b and (not distinct_groups or group_of[a] != group_of[b]):
                out_pairs.append((a, b))
        return out_pairs

    string_edges = [(a, b) for a, b in G.edges() if group_of[a] != group_of[b]]
    sets = {
        f"Top {args.top} isolated-connected (capped)": list(zip(ic["protein_a"], ic["protein_b"])),
        "Random isolated-connected pairs": random_pairs(iso, con, 50_000),
        f"Top {args.top} isolated-isolated (capped)": list(zip(ii["protein_a"], ii["protein_b"])),
        "Random isolated-isolated pairs": random_pairs(iso, iso, 50_000),
        "Real STRING edges (positive control)": string_edges,
    }
    recs, detail = [], {}
    for label, pairs in sets.items():
        rec, bp, bg = coherence(pairs, fams, clan_sets, go_sets, member_of, label)
        recs.append(rec)
        detail[label] = rec
    coh = pd.DataFrame(recs)

    # Fisher tests: each top set vs its random counterpart
    tests = []
    for top_label, rnd_label in [(f"Top {args.top} isolated-connected (capped)", "Random isolated-connected pairs"),
                                 (f"Top {args.top} isolated-isolated (capped)", "Random isolated-isolated pairs")]:
        t, r = detail[top_label], detail[rnd_label]
        for metric, n_col in [("share_pfam_family", "n_both_pfam"), ("share_pfam_clan", "n_both_pfam"),
                              ("share_go_term", "n_both_go"), ("any_surface_family", "n_pairs")]:
            a = round(t[metric] * t[n_col]); b = t[n_col] - a
            c = round(r[metric] * r[n_col]); d = r[n_col] - c
            orr, p = stats.fisher_exact([[a, b], [c, d]])
            tests.append({"comparison": f"{top_label} vs random", "metric": metric,
                          "top": t[metric], "random": r[metric], "odds_ratio": orr, "fisher_p": p})
    tests = pd.DataFrame(tests)
    tests["fisher_p_bh"] = bh_fdr(tests["fisher_p"])
    coh.to_csv(out / "functional_coherence.csv", index=False)
    tests.to_csv(out / "functional_coherence_tests.csv", index=False)
    print(coh.round(4).to_string())
    print(tests.round(4).to_string())

    # 3. family sets among isolated proteins appearing in the top predictions
    in_top = set(ic["protein_a"]) | set(ii["protein_a"]) | set(ii["protein_b"])
    iso_mask = np.isin(nodes, iso)
    top_mask = np.isin(nodes, list(in_top)) & iso_mask
    rows = []
    for name, m in member.items():
        a = int((m & top_mask).sum()); b = int((~m & top_mask).sum())
        c = int((m & iso_mask & ~top_mask).sum()); d = int((~m & iso_mask & ~top_mask).sum())
        orr, p = stats.fisher_exact([[a, b], [c, d]])
        rows.append({"family_set": name, "in_top_with_family": a, "in_top_total": a + b,
                     "rest_with_family": c, "rest_total": c + d, "odds_ratio": orr, "fisher_p": p})
    fe = pd.DataFrame(rows)
    fe["fisher_p_bh"] = bh_fdr(fe["fisher_p"])
    fe.to_csv(out / "family_enrichment_top.csv", index=False)
    print(f"\nIsolated proteins in top lists: {top_mask.sum()} of {iso_mask.sum()}")
    print(fe.round(4).to_string())

    # 4. GO enrichment (pfam2go) for the same foreground vs all isolated proteins with GO
    bg = [n for n in nodes[iso_mask] if n in go_sets]
    fg = [n for n in bg if n in in_top]
    term_bg, term_fg = defaultdict(int), defaultdict(int)
    for n in bg:
        for t in go_sets[n]:
            term_bg[t] += 1
    for n in fg:
        for t in go_sets[n]:
            term_fg[t] += 1
    rows = []
    for t, K in term_bg.items():
        if K < 3:
            continue
        k = term_fg.get(t, 0)
        p = stats.hypergeom.sf(k - 1, len(bg), K, len(fg))
        rows.append({"go_term": t, "go_name": go_name.get(t, ""), "fg_count": k, "fg_size": len(fg),
                     "bg_count": K, "bg_size": len(bg), "fold": (k / len(fg)) / (K / len(bg)), "p": p})
    go_df = pd.DataFrame(rows)
    go_df["p_bh"] = bh_fdr(go_df["p"])
    go_df = go_df.sort_values("p")
    go_df.to_csv(out / "go_enrichment_top.csv", index=False)
    print(f"\nGO enrichment: {len(fg)} foreground / {len(bg)} background isolated proteins with GO")
    print(go_df.head(12).round(4).to_string())

    # 5. top partners of isolated proteins in a family set
    ann = pd.read_csv(REPO / "Data" / "annotation" / "protein_annotation.tsv", sep="\t").set_index("protein_id")
    ann = ann.fillna("")
    S_ic, S_ii = sc["S_isolated_connected"], sc["S_isolated_isolated"]
    cat_of = {n: ", ".join(k for k in FAMILY_SETS if member[k][i]) for i, n in enumerate(nodes)}
    tier_of = dict(zip(nodes, tier))
    rows = []
    for a, p in enumerate(iso):
        if not member_of[p]:
            continue
        scores = np.r_[S_ic[a], np.delete(S_ii[a], a)]
        partners = np.r_[con, np.delete(iso, a)]
        pct = scores.argsort().argsort() / (len(scores) - 1)
        used, r = {group_of[p]}, 0
        for b in np.argsort(-scores):
            q = partners[b]
            if group_of[q] in used:
                continue
            used.add(group_of[q])
            r += 1
            rows.append({"isolated_protein": p, "isolated_tier": tier_of[p], "isolated_families": cat_of[p],
                         "isolated_pfam": ann.loc[p, "pfam_architecture"], "partner_rank": r, "partner": q,
                         "partner_tier": tier_of[q], "partner_connected": bool(deg[nodes == q][0] > 0),
                         "partner_families": cat_of[q], "partner_pfam": ann.loc[q, "pfam_architecture"],
                         "partner_pfam_desc": ann.loc[q, "pfam_descriptions"], "score": scores[b],
                         "within_protein_percentile": pct[b]})
            if r == 5:
                break
    case = pd.DataFrame(rows)
    case.to_csv(out / "surface_isolated_partners.csv", index=False)
    top1 = case[case["partner_rank"] == 1]
    print(f"\nSurface-family isolated proteins: {case['isolated_protein'].nunique()}; "
          f"top-1 partner also in a surface family: {(top1['partner_families'] != '').mean():.0%} "
          f"(base rate among all proteins {member[ANY].mean():.0%})")


if __name__ == "__main__":
    main()
