"""Generate every number and results table quoted in the manuscript from the result files.

Writes Flova_Reverse_Vac_Biomni/resubmission/numbers.tex (LaTeX macros) and the
tab_*.tex table bodies, so the text never carries hand-copied values.

Usage (from repo root, after all runs have finished):
    python Code/make_manuscript_numbers.py
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))
from phaseA_rerun import build_graph  # noqa: E402
from phaseD_biology import FAMILY_SETS  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
RES = REPO / "Results"
FIN = RES / "final_v12"
OUT = REPO / "Flova_Reverse_Vac_Biomni" / "resubmission"
FEATURES = ["Biomni only", "ESM only", "Node2Vec only", "Biomni + ESM", "Biomni + Node2Vec",
            "ESM + Node2Vec", "Biomni + ESM + Node2Vec"]
MODELS = ["GCN", "GAT", "GraphSAGE", "MLP (no graph)"]
TEX_FEAT = {f: f.replace("ESM", "ESM-2").replace(" only", "").replace(" + ", "+") for f in FEATURES}

macros = {}


def m(name, value):
    macros[name] = value


def f3(x):
    return f"{x:.3f}"


def pm(mean, sd):
    return f"{mean:.3f}\\,$\\pm$\\,{sd:.3f}"


def pct(x, d=1):
    return f"{100 * x:.{d}f}\\%"


def n(x):
    return f"{int(x):,}".replace(",", "{,}")


def pval(p):
    return "<0.001" if p < 0.001 else f"{p:.3f}"


def label(fs, model):
    return f"{TEX_FEAT[fs]} with {model.replace(' (no graph)', '')}"


def summary(path):
    return pd.read_csv(path)


def main():
    OUT.mkdir(parents=True, exist_ok=True)

    # ---------------- data
    seqs, pid = {}, None
    for line in open(REPO / "Data" / "Flavobacterium sp. MO_0223_q7L1K.faa"):
        line = line.strip()
        if line.startswith(">"):
            pid = line[1:].split()[0]
            seqs[pid] = []
        elif pid:
            seqs[pid].append(line)
    seqs = {k: "".join(v) for k, v in seqs.items()}
    groups = pd.Series(list(seqs.values())).value_counts()
    m("nProteins", n(len(seqs)))
    m("nIdentGroups", n((groups > 1).sum()))
    m("nIdentProteins", n(groups[groups > 1].sum()))

    ann = pd.read_csv(REPO / "Data" / "annotation" / "protein_annotation.tsv", sep="\t").fillna("")
    k = (ann["pfam_architecture"] != "").sum()
    m("nPfamAnnotated", n(k))
    m("pctPfamAnnotated", pct(k / len(ann), 0))
    v = pd.read_csv(REPO / "Data" / "complete_vaccine_analysis_all_3257_proteins.csv")
    m("rMwLen", f"{np.corrcoef(v['protein_length'], v['molecular_weight_da'])[0, 1]:.3f}")

    submitted = {l[1:].split()[0] for l in open(REPO / "Data" / "top_2000_sequences.faa") if l.startswith(">")}
    web = pd.read_csv(REPO / "Data" / "string_mapping.tsv", sep="\t")
    ph = pd.read_csv(REPO / "Data" / "string_v12_mapping_all.tsv", sep="\t")
    acc = ph[ph["accepted"]]
    m("nSubmitted", n(len(submitted)))
    m("nWebMapped", n(web["queryItem"].nunique()))
    m("nNeverSubmitted", n(len(seqs) - len(submitted)))
    w1 = web.drop_duplicates("queryItem").merge(ph, on="queryItem", suffixes=("_web", "_ph"))
    m("mapAgree", pct((w1["stringId_web"] == w1["stringId_ph"]).mean()))
    m("nPhmmerNew", n(acc["queryItem"].isin(set(seqs) - submitted).sum()))
    m("nMappedFinal", n(len(acc)))
    import gzip
    m("nStringProteins", n(sum(1 for l in gzip.open(Path.home() / ".cache" / "string_v12" /
                                                     "1041826.protein.sequences.v12.0.fa.gz", "rt") if l.startswith(">"))))

    Gd, _ = build_graph(source="string_web")
    Gf, _ = build_graph(source="v12_full")
    m("nEdgesDev", n(Gd.number_of_edges()))
    m("nEdgesFinal", n(Gf.number_of_edges()))
    iso_d = [x for x in Gd if Gd.degree(x) == 0]
    iso_f = [x for x in Gf if Gf.degree(x) == 0]
    m("nIsoDev", n(len(iso_d)))
    m("nIsoFinal", n(len(iso_f)))
    ed = {frozenset(e) for e in Gd.edges()}
    ef = {frozenset(e) for e in Gf.edges()}
    m("vTwelveRecovered", pct(len(ed & ef) / len(ed)))
    web_ids = set(web["queryItem"])
    d_not = sum(x not in submitted for x in iso_d)
    d_nomatch = sum(x in submitted and x not in web_ids for x in iso_d)
    d_para = len(iso_d) - d_not - d_nomatch
    acc_ids = set(acc["queryItem"])
    best = acc.sort_values(["bitscore", "identity"], ascending=False).drop_duplicates("stringId")
    assigned = set(best["queryItem"])
    f_nomatch = sum(x not in acc_ids for x in iso_f)
    f_para = sum(x in acc_ids and x not in assigned for x in iso_f)
    f_nolink = len(iso_f) - f_nomatch - f_para
    m("isoDevNotSubmitted", n(d_not)); m("isoDevNoMatch", n(d_nomatch)); m("isoDevParalog", n(d_para))
    m("isoDevNoLink", "0")
    m("isoFinalNotSubmitted", "0"); m("isoFinalNoMatch", n(f_nomatch)); m("isoFinalParalog", n(f_para))
    m("isoFinalNoLink", n(f_nolink))

    man = json.loads((FIN / "phaseA_fixed_esm650M" / "run_manifest.json").read_text())
    m("pyVersion", man["python"]); m("torchVersion", man["torch"].split("+")[0])
    m("pygVersion", man["torch_geometric"]); m("sklearnVersion", man["sklearn"])
    m("nSeeds", str(len(man["seeds"])))

    # ---------------- benchmark, final network
    p1 = summary(FIN / "phaseA_fixed_esm650M" / "summary.csv")
    gnn = p1[p1["model"].isin(MODELS)].sort_values("auc_mean", ascending=False).iloc[0]
    heur = p1[p1["feature_set"] == "Topology heuristic"].sort_values("auc_mean", ascending=False).iloc[0]
    m("pOneBestGNN", pm(gnn["auc_mean"], gnn["auc_std"])); m("pOneBestGNNlabel", label(gnn["feature_set"], gnn["model"]))
    m("pOneBestHeur", pm(heur["auc_mean"], heur["auc_std"])); m("pOneBestHeurLabel", heur["model"].lower())
    bo = p1[p1["feature_set"] == "Biomni only"].sort_values("auc_mean", ascending=False).iloc[0]
    m("pOneBiomni", pm(bo["auc_mean"], bo["auc_std"]))

    p2 = summary(FIN / "phaseB_coldstart_uniform" / "summary.csv")
    pdg = p2[p2["model"] == "Partner degree"].iloc[0]
    m("pTwoPartnerDeg", pm(pdg["C2_auc_mean"], pdg["C2_auc_std"]))
    b2 = p2[p2["model"].isin(MODELS)].sort_values("C2_auc_mean", ascending=False).iloc[0]
    m("pTwoBest", pm(b2["C2_auc_mean"], b2["C2_auc_std"])); m("pTwoBestLabel", label(b2["feature_set"], b2["model"]))

    p3 = summary(FIN / "phaseB_coldstart_degree" / "summary.csv")
    pdg3 = p3[p3["model"] == "Partner degree"].iloc[0]
    m("pThreePartnerDeg", pm(pdg3["C2_auc_mean"], pdg3["C2_auc_std"]))
    b3 = p3[p3["model"].isin(MODELS)].sort_values("C2_auc_mean", ascending=False).iloc[0]
    m("pThreeBest", pm(b3["C2_auc_mean"], b3["C2_auc_std"])); m("pThreeBestLabel", label(b3["feature_set"], b3["model"]))
    esm_same = p3[(p3["feature_set"] == "ESM only") & (p3["model"] == b3["model"])].iloc[0]
    m("pThreeEsmMLP", pm(esm_same["C2_auc_mean"], esm_same["C2_auc_std"]))
    bo3 = p3[(p3["feature_set"] == "Biomni only") & p3["model"].isin(MODELS)].sort_values("C2_auc_mean").iloc[-1]
    m("pThreeBiomniOnly", pm(bo3["C2_auc_mean"], bo3["C2_auc_std"]))
    per = pd.read_csv(FIN / "phaseB_coldstart_degree" / "results_per_seed.csv")
    a = per[(per["feature_set"] == "Biomni + ESM") & (per["model"] == b3["model"])].set_index("seed")["test_C2_auc"]
    b = per[(per["feature_set"] == "ESM only") & (per["model"] == b3["model"])].set_index("seed")["test_C2_auc"]
    m("pThreeBiomniGain", f"{(a - b).mean():+.3f}")
    m("pThreeBiomniGainP", pval(stats.wilcoxon(a, b).pvalue))

    esm_mlp = p3[p3["feature_set"].str.contains("ESM") & (p3["model"] == "MLP (no graph)")]["C2_auc_mean"]
    m("pThreeEsmRangeLow", f3(esm_mlp.min())); m("pThreeEsmRangeHigh", f3(esm_mlp.max()))
    m("pThreeNodeVecMax", f3(p3[(p3["feature_set"] == "Node2Vec only") & p3["model"].isin(MODELS)]["C2_auc_mean"].max()))
    st3 = pd.read_csv(FIN / "phaseB_coldstart_degree" / "statistical_tests.csv")
    ties = st3[st3["comp_feature_set"].str.contains("ESM") & (st3["comp_model"] == "MLP (no graph)")]
    m("pThreeTiesMinP", f"{ties['wilcoxon_p_bh'].min():.2f}")
    p1g = p1[p1["model"].isin(MODELS)]
    m("pOneRangeLow", f3(p1g["auc_mean"].min()))

    cells = []
    for fs in FEATURES:
        row = p3[p3["feature_set"] == fs].set_index("model")
        vals = [row.loc[mo, "C2_auc_mean"] if mo in row.index else np.nan for mo in MODELS]
        best_i = int(np.nanargmax(vals))
        out = []
        for i, mo in enumerate(MODELS):
            s = pm(row.loc[mo, "C2_auc_mean"], row.loc[mo, "C2_auc_std"])
            out.append(f"\\textbf{{{s}}}" if i == best_i else s)
        cells.append(f"{TEX_FEAT[fs]} & " + " & ".join(out) + " \\\\")
    base = p3[p3["feature_set"] == "Baseline (no training)"].set_index("model")
    topo = base.loc[["Common neighbors", "Adamic-Adar", "Resource allocation", "Preferential attachment"], "C2_auc_mean"]
    brows = [("Best topology heuristic", topo.max(), 0.0),
             ("Partner degree", base.loc["Partner degree", "C2_auc_mean"], base.loc["Partner degree", "C2_auc_std"]),
             ("ESM-2 cosine similarity", base.loc["ESM cosine", "C2_auc_mean"], base.loc["ESM cosine", "C2_auc_std"])]
    tab = ["\\begin{tabular}{lcccc}", "\\toprule",
           "Feature set & GCN & GAT & GraphSAGE & MLP \\\\", "\\midrule", *cells, "\\midrule",
           *[f"{nm} & \\multicolumn{{4}}{{c}}{{{pm(mu, sd)}}} \\\\" for nm, mu, sd in brows],
           "\\bottomrule", "\\end{tabular}"]
    (OUT / "tab_coldstart.tex").write_text("\n".join(tab) + "\n")

    # ---------------- internal estimate and external validation (development network)
    cv = pd.read_csv(RES / "phaseC_candidates" / "heldout_precision_summary.csv", index_col=0)
    m("cvFullRankAUC", f3(cv.loc["full_ranking_auc", "mean"]))
    vs = pd.read_csv(RES / "phaseD_external_validation" / "validation_summary.csv").set_index("method")
    pp = pd.read_csv(RES / "phaseD_external_validation" / "per_protein_validation.csv")
    m("nExtProteins", n(len(pp))); m("nExtProteinsWithLinks", n(len(pp))); m("nExtLinks", n(pp["n_true"].sum()))
    m("extBaseRate", f3(pp["n_true"].sum() / pp["n_candidates"].sum()))
    bm, eo = "Biomni + ESM (Phase C ensemble)", "ESM only (Phase C ensemble)"
    m("extPooledAUC", f3(vs.loc[bm, "pooled_AUC"]))
    m("extPerProtAUC", f3(vs.loc[bm, "per_protein_AUC_mean"]))
    m("extPerProtAUCesm", f3(vs.loc[eo, "per_protein_AUC_mean"]))
    m("extBiomniP", f"{vs.loc[eo, 'wilcoxon_p_vs_BiomniESM (per-protein AUC)']:.2f}")
    m("extDegAUC", f3(vs.loc["Partner degree", "pooled_AUC"]))
    ex = pd.read_csv(RES / "phaseE_rerank_biomni_esm" / "external_summary.csv").set_index("variant")
    m("extDotPfive", f3(ex.loc["dot", "per_protein_P@5"]))
    m("extEsmPfive", f3(ex.loc["esm", "per_protein_P@5"]))
    m("extEsmHit", pct(ex.loc["esm", "per_protein_hit@10"], 0))
    m("extEsmPooled", f3(ex.loc["esm", "pooled_AUC"]))
    m("extBlendAUC", f3(ex.loc["blend(cos,0.5)", "per_protein_AUC"]))
    m("extBlendPfive", f3(ex.loc["blend(cos,0.5)", "per_protein_P@5"]))
    m("extBlendPooled", f3(ex.loc["blend(cos,0.5)", "pooled_AUC"]))

    rows = [("Model, Biomni+ESM-2", ex.loc["dot"]), ("Model, ESM-2 only", None),
            ("ESM-2 cosine (pre-selected)", ex.loc["esm"]), ("Blend (0.5)", ex.loc["blend(cos,0.5)"]),
            ("Partner degree", ex.loc["partner degree"])]
    eo_row = vs.loc[eo]
    lines = ["\\begin{tabular}{lccccc}", "\\toprule",
             "Ranking & \\multicolumn{3}{c}{Per protein} & \\multicolumn{2}{c}{Pooled} \\\\",
             "\\cmidrule(lr){2-4}\\cmidrule(lr){5-6}",
             " & AUC & P@5 & Hit@10 & AUC & P@0.1\\% \\\\", "\\midrule"]
    for nm, r in rows:
        if r is None:
            vals = [eo_row["per_protein_AUC_mean"], eo_row["per_protein_P@5"], eo_row["per_protein_hit@10"],
                    eo_row["pooled_AUC"], eo_row["pooled_precision_top0.001"]]
        else:
            vals = [r["per_protein_AUC"], r["per_protein_P@5"], r["per_protein_hit@10"], r["pooled_AUC"],
                    r["pooled_precision_top0.001"]]
        lines.append(nm + " & " + " & ".join(f3(x) for x in vals) + " \\\\")
    br = pp["n_true"].sum() / pp["n_candidates"].sum()
    lines += [f"Random & 0.500 & {f3(br)} & -- & 0.500 & {f3(br)} \\\\", "\\bottomrule", "\\end{tabular}"]
    (OUT / "tab_external.tex").write_text("\n".join(lines) + "\n")

    # ---------------- re-ranking table (internal final, external dev)
    ifin = pd.read_csv(FIN / "phaseE_rerank_biomni_esm" / "internal_summary.csv").set_index("variant")
    names = {"esm": "ESM-2 cosine", "dot": "Model score", "cos": "Model cosine", "csls": "Model cosine, CSLS",
             "blend(cos,0.25)": "Blend cosine/ESM-2 (0.25)", "blend(cos,0.5)": "Blend cosine/ESM-2 (0.5)",
             "blend(cos,0.75)": "Blend cosine/ESM-2 (0.75)", "blend(csls,0.5)": "Blend CSLS/ESM-2 (0.5)"}
    lines = ["\\begin{tabular}{lcccccc}", "\\toprule",
             "Variant & \\multicolumn{3}{c}{Internal (final network)} & \\multicolumn{3}{c}{External} \\\\",
             "\\cmidrule(lr){2-4}\\cmidrule(lr){5-7}",
             " & P@5 & Prot.\\ AUC & Pooled AUC & P@5 & Prot.\\ AUC & Pooled AUC \\\\", "\\midrule"]
    for key in ifin.index:
        i, e = ifin.loc[key], ex.loc[key]
        lines.append(f"{names[key]} & {f3(i['per_protein_P@5_mean'])} & {f3(i['per_protein_AUC_mean'])} & "
                     f"{f3(i['pooled_AUC_mean'])} & {f3(e['per_protein_P@5'])} & {f3(e['per_protein_AUC'])} & "
                     f"{f3(e['pooled_AUC'])} \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    (OUT / "tab_rerank.tex").write_text("\n".join(lines) + "\n")

    # ---------------- ESM size table (development network, P3)
    s35 = summary(RES / "phaseB_coldstart_degree" / "summary.csv").set_index(["feature_set", "model"])
    s650 = summary(RES / "phaseB_coldstart_degree_esm650M" / "summary.csv").set_index(["feature_set", "model"])
    lines = ["\\begin{tabular}{llcc}", "\\toprule", "Feature set & Encoder & ESM-2 35M & ESM-2 650M \\\\", "\\midrule"]
    for fs in ["ESM only", "Biomni + ESM", "ESM + Node2Vec", "Biomni + ESM + Node2Vec"]:
        for mo in ["GraphSAGE", "MLP (no graph)"]:
            a_, b_ = s35.loc[(fs, mo)], s650.loc[(fs, mo)]
            lines.append(f"{TEX_FEAT[fs]} & {mo.replace(' (no graph)', '')} & {pm(a_['C2_auc_mean'], a_['C2_auc_std'])}"
                         f" & {pm(b_['C2_auc_mean'], b_['C2_auc_std'])} \\\\")
    a_, b_ = s35.loc[("Baseline (no training)", "ESM cosine")], s650.loc[("Baseline (no training)", "ESM cosine")]
    lines += [f"ESM-2 cosine similarity & -- & {pm(a_['C2_auc_mean'], a_['C2_auc_std'])} & "
              f"{pm(b_['C2_auc_mean'], b_['C2_auc_std'])} \\\\", "\\bottomrule", "\\end{tabular}"]
    (OUT / "tab_esm_size.tex").write_text("\n".join(lines) + "\n")

    # ---------------- tiers and families
    tv = pd.read_csv(FIN / "phaseD_biology" / "tier_validation.csv").set_index("family_set")
    anyr = tv.loc["Any surface/secretion family"]
    m("nSurfaceProteins", n(anyr["n_proteins"]))
    m("tierFracMedium", pct(anyr["frac_Medium"])); m("tierFracHigh", pct(anyr["frac_High"]))
    m("tierOR", f"{anyr['odds_ratio_high_vs_rest']:.2f}"); m("tierORp", f"{anyr['fisher_p_bh']:.2f}")
    m("tierRho", f"{anyr['spearman_rho_tier']:.2f}")
    lines = ["\\begin{tabular}{lrccccc}", "\\toprule",
             "Family set & $n$ & Very Low & Low & Medium & High & OR High (BH $p$) \\\\", "\\midrule"]
    for fam, r in tv.iterrows():
        orv = "0" if r["odds_ratio_high_vs_rest"] == 0 else f"{r['odds_ratio_high_vs_rest']:.2f}"
        lines.append(f"{fam} & {int(r['n_proteins'])} & {pct(r['frac_Very Low'])} & {pct(r['frac_Low'])} & "
                     f"{pct(r['frac_Medium'])} & {pct(r['frac_High'])} & {orv} ({r['fisher_p_bh']:.2f}) \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    (OUT / "tab_tiers.tex").write_text("\n".join(lines) + "\n")

    hits = pd.read_csv(REPO / "Data" / "annotation" / "pfam_domain_hits.tsv", sep="\t")
    name_of = dict(zip(hits["pfam_acc"], hits["pfam_name"]))
    lines = ["\\begin{tabular}{lp{0.62\\textwidth}}", "\\toprule", "Family set & Pfam families \\\\", "\\midrule"]
    for fam, accs in FAMILY_SETS.items():
        items = ", ".join(f"{name_of.get(x, '').replace('_', chr(92) + '_')} ({x})" for x in accs)
        lines.append(f"{fam} & {items} \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    (OUT / "tab_families.tex").write_text("\n".join(lines) + "\n")

    bias = pd.read_csv(RES / "phaseD_external_validation" / "tier_pairs_model_vs_real.csv", index_col=0)
    ratio = bias["model_top5"] / bias["real_links"]
    m("biasMedMed", f"{ratio['Medium–Medium']:.1f}"); m("biasHighHigh", f"{ratio['High–High']:.1f}")
    m("biasVLVL", f"{ratio['Very Low–Very Low']:.2f}$\\times$")

    coh = pd.read_csv(FIN / "phaseD_biology" / "functional_coherence.csv").set_index("pair_set")
    ic = [x for x in coh.index if x.startswith("Top") and "isolated-connected" in x][0]
    ii = [x for x in coh.index if x.startswith("Top") and "isolated-isolated" in x][0]
    m("cohICgo", pct(coh.loc[ic, "share_go_term"])); m("cohIIgo", pct(coh.loc[ii, "share_go_term"]))
    m("cohICclan", pct(coh.loc[ic, "share_pfam_clan"])); m("cohIIclan", pct(coh.loc[ii, "share_pfam_clan"]))
    m("cohICgoRand", pct(coh.loc["Random isolated-connected pairs", "share_go_term"]))
    m("cohIIgoRand", pct(coh.loc["Random isolated-isolated pairs", "share_go_term"]))
    real = [x for x in coh.index if "STRING" in x][0]
    m("cohRealGo", pct(coh.loc[real, "share_go_term"])); m("cohRealClan", pct(coh.loc[real, "share_pfam_clan"]))

    body = ["% Generated by Code/make_manuscript_numbers.py -- do not edit by hand."]
    body += [f"\\newcommand{{\\{k}}}{{{v}}}" for k, v in sorted(macros.items())]
    (OUT / "numbers.tex").write_text("\n".join(body) + "\n")
    print(f"wrote {len(macros)} macros and 6 tables to {OUT}")


if __name__ == "__main__":
    main()
