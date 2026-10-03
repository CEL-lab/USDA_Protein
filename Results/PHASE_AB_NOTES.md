# Phase A/B reruns (2026-10-01)

Reruns made in response to the Aquaculture International review. Every run below uses
10 seeds (0–9), except `phaseA_raw`, which reuses the published seeds 42–46.
Each run writes `results_per_seed.csv`, `summary.csv`, `statistical_tests.csv`
(paired t-test and Wilcoxon against the best configuration, BH-FDR adjusted) and
`run_manifest.json`. The phaseA runs also save the split indices (`splits/`).

| Run | Script | What changed |
|---|---|---|
| `phaseA_raw` | `Code/phaseA_rerun.py --mode raw` | Reproduces the published notebook: unscaled Biomni features. |
| `phaseA_fixed` | `Code/phaseA_rerun.py --mode fixed` | Biomni features log-transformed where skewed, molecular weight dropped (r = 0.999 with length), all columns z-scored. |
| `phaseA_fixed_esm650M` | same, `--esm ..._650M_...npz` | ESM-2 650M (1280-d) instead of 35M (480-d). |
| `phaseB_coldstart_uniform` | `Code/phaseB_coldstart.py --neg uniform` | 10% val / 10% test proteins held out with all their edges. |
| `phaseB_coldstart_degree` | `Code/phaseB_coldstart.py --neg degree` | Same, but negatives are degree-matched. |
| `phaseB_coldstart_degree_esm650M` | same, `--esm ..._650M_...npz` | 650M embeddings. |

ESM embeddings for all 3,257 proteins come from `Code/compute_esm_embeddings.py`. The older
`Data/esm_embeddings_flavobacterium.npz` covers only 1,644 proteins. It was made with
ESM-2 **35M**, not the 650M model the manuscript names: its vectors match the new 35M
embeddings with median cosine similarity 1.000.

Overview figure: `Figure_3_Benchmark_PhaseAB.png/.pdf` (`Code/summarize_phaseAB.py`).

## Key results (test ROC-AUC, mean ± SD over seeds)

1. **The published Biomni results come from a scaling bug.** The raw rerun reproduces every
   Biomni row of the published table to 3 decimals. All of those predictions are saturated
   (sigmoid output exactly 0 or 1), and every GCN/GAT configuration that uses Biomni sits
   at 0.500 ± 0.000. After scaling, Biomni-only reaches 0.936 ± 0.004 (GCN and GraphSAGE),
   and no predictions are saturated.
2. **Edge split (the paper's protocol).** Node2Vec + GCN = 0.971 ± 0.002, but parameter-free
   heuristics reach 0.956–0.960 (resource allocation 0.960 ± 0.004). Under this protocol the
   GNNs add about 0.01 AUC over counting common neighbours.
3. **Cold start with uniform negatives** (held-out protein, partner in the training graph).
   Ranking pairs by the training partner's degree alone gives 0.890 ± 0.006, which beats
   every trained model. Uniform negatives mainly reward choosing hub partners.
4. **Cold start with degree-matched negatives** (the recommended primary protocol).
   - Node2Vec ≈ 0.51 and topology heuristics = 0.50: graph structure says nothing about
     proteins that have no edges.
   - Best: Biomni + ESM with an MLP that ignores the graph: 0.733 ± 0.020 (35M) and
     0.742 ± 0.017 (650M). The GNNs do not beat the MLP.
   - Biomni + ESM beats ESM-only: +0.007 (35M, BH p = 0.010) and +0.018 (650M, BH p = 0.005).
   - ESM-2 650M over 35M: a small, consistent gain (about +0.01–0.03).
   - Both endpoints unseen ("C3", only about 127 positives per seed): best about 0.72–0.73.

## Implications for the manuscript

- Fix the Methods to match the code: ESM-2 35M (or switch to 650M), seeds, hidden size 128,
  repeated random splits (not 3-fold CV), no hyperparameter grid.
- Report the cold-start protocol with degree-matched negatives as the main evaluation. That
  is the setting the paper applies the model to (1,650 isolated proteins). Node2Vec-based
  scores for isolated proteins are uninformative and should not be used to rank candidates.
- Re-score the candidate links for isolated proteins with the best cold-start model
  (Biomni + ESM, MLP or GraphSAGE) before any class-pair or biological interpretation.

# Phase C: re-scoring the isolated-protein candidates (2026-10-01)

Script: `Code/phaseC_rescore_candidates.py`. Outputs are in `Results/phaseC_candidates/`; the
ESM-only control is in `Results/phaseC_candidates_esm_only/`.

**Model.** Biomni + ESM-2 650M with the MLP encoder, trained with degree-matched
negatives on each of the 10 cold-start splits. The 10 models' scores are averaged.

**Candidates.** 2,651,550 isolated–connected and 1,360,425 isolated–isolated pairs.

## How good is the ranking on held-out proteins?

Each model ranks every pair of (held-out protein × training protein) against the true
STRING edges. The base rate is 1.2%.

| Metric | Biomni + ESM | ESM only |
|---|---|---|
| Full-ranking AUC | 0.715 ± 0.026 | 0.701 ± 0.028 |
| Precision of each protein's top 5 partners | 7.5% ± 1.8% (random 1.2%, about 6×) | 6.2% ± 2.0% |
| Precision of the top 0.1% of all pairs | 16.7% ± 11.3% (about 14×) | 18.0% ± 10.3% |
| Precision of the top 1% of all pairs | 9.7% ± 3.3% (about 8×) | n/a |

Scores are uncalibrated sigmoid outputs (top scores are about 0.99). Report ranks or
percentiles, not a probability threshold. The paper's "calibrated p ≥ 0.60" claim
should be dropped.

## Data problems found

1. **Identical sequences.** 320 proteins form 111 groups of identical sequence (largest:
   15 copies of a 40-aa peptide). They get identical scores. The capped lists collapse
   them and report `n_identical_*`. No group has more than one connected member, so
   there is no train/test leakage.
2. **Mapping collision, a graph-construction bug (now fixed).** The notebook relabelled STRING
   nodes with `dict(zip(preferredName, queryItem))`. When several proteome proteins match the
   same STRING protein, the last row won, not the best hit. 52 STRING proteins have more than
   one match, and in 30 of them a weaker match got the edges. `build_graph()` now assigns each
   STRING protein to its highest-bitscore match (`mapping="best"`); `--mode raw` keeps the old
   rule so the published numbers can still be reproduced. 29 proteins swap connected/isolated
   status. Phases A–C were rerun with the fixed graph; results from the old rule are archived
   in `Results/archive_lastwins_mapping/`. No AUC changed by more than 0.009, and every
   conclusion stands. The ~150 lower-scoring matches stay isolated by design (giving paralogs
   copies of the same edges would leak between train and test) and are flagged as
   `secondary STRING hit` in `isolated_reason_*`.
3. **Concentration.** One protein (peg.856) appears in about 10% of the uncapped top 2,000.
   Use the capped lists (≤ 5 pairs per unique sequence) for interpretation.

## Priority tier pairs (top 2,000 capped pairs; observed/expected from the candidate pool)

| Tier pair | Isolated–connected | Isolated–isolated |
|---|---|---|
| Medium–Medium | 6.6× | 6.2× |
| High–High | 5.9× | 6.5× |
| Medium–High | 3.3× | 3.3× |
| Low–High | 1.9× | 2.4× |
| Very Low–High | 0.54× | 0.32× |

All of these are BH-adjusted p < 0.001. The ESM-only model gives the same pattern
(Medium–Medium 7.0× / 5.8×), so it is not caused by the model seeing the tiers.
Real STRING edges show the same direction but weaker (Medium–Medium 1.8×, High–High 1.9×,
under a degree-preserving expectation).

This reverses the published claim that the top tail is dominated by Very Low/Low
pairings. That claim came from the Node2Vec model, which is at chance on isolated
proteins.

## Functional annotation (Pfam 38.2)

`Code/annotate_pfam.py` runs pyhmmer hmmsearch with Pfam-A gathering thresholds and maps
domains to GO terms with pfam2go. Results are in `Data/annotation/`
(`pfam_domain_hits.tsv`, `protein_annotation.tsv`, `protein_to_go.tsv`).

- 2,300 of 3,257 proteins have at least one domain (isolated proteins 52%, connected 90%);
  1,231 have GO terms.
- Vaccine-relevant families in the proteome: 34 proteins with the type IX secretion
  C-terminal sorting domain (`Por_Secre_tail`), 18 TonB-dependent receptors, 11 OmpA,
  9 CHU_C, 7 PorP/SprF.

`Code/annotate_candidates.py` adds the annotations to the capped candidate lists and the
High-priority partner table (`*_annotated.tsv`). The most frequently predicted proteins
(e.g. peg.856, peg.779, peg.2548) have no Pfam domain but strong signal-peptide scores,
so they are likely secreted proteins of unknown function.

## Still missing

- Known-antigen recovery and GO enrichment (reviewer comment 6): test whether High-tier
  proteins and top-ranked candidates are enriched for T9SS cargo, OmpA, TonB-dependent
  receptors and other surface families, against literature antigen lists from co-authors.

# Phase D: biological evaluation (2026-10-02)

## D1. Biomni tiers vs a-priori surface/secretion families (`Code/phaseD_biology.py`)

The Pfam family sets were fixed by accession before analysis (`FAMILY_SETS`). Results are in
`Results/phaseD_biology/tier_validation.csv`.

- Any surface/secretion family: 2.6% of Very Low proteins, 6.2% of Low, 12.4% of Medium,
  **3.0% of High**.
- High vs rest: odds ratio 0.62, BH p = 0.85. There is a weak positive monotone trend
  (Spearman ρ = 0.11, p < 0.001), but it is driven by the Medium tier.
- TonB/SusCD receptors: 0 in High. T9SS cargo: 0.4% in High vs 3.5% in Medium.

So the Biomni "High" tier does **not** pick out classic outer-membrane or secreted antigen
families. This is a direct, honest answer to reviewer comment 1 and should be reported.

## D2. Functional coherence of predicted pairs

Top 2,000 capped pairs, compared with 50,000 random candidate pairs and with all real STRING
edges:

| Pair set | Share a GO term | Share a Pfam clan | Share a Pfam family |
|---|---|---|---|
| Top isolated–connected | 18.1% | 5.2% | 1.1% |
| Random isolated–connected | 3.9% | 1.4% | 0.1% |
| Top isolated–isolated | 44.6% | 10.2% | 6.9% |
| Random isolated–isolated | 8.2% | 2.0% | 0.5% |
| Real STRING edges | 19.4% | 10.0% | 3.8% |

All differences are BH p < 0.001. Part of this is sequence similarity (ESM), so it shows
plausibility, not independent proof.

## D3. Pipeline problem: only 2,000 proteins were ever sent to STRING

`Data/top_2000_sequences.faa`, the longest-skewed set (mean 430 aa vs 116 aa for the
rest), was the only input to the STRING web mapping, which caps uploads at 2,000
sequences. As a result:

- 1,257 of the 1,650 "isolated" proteins (76%) were never queried.
- 240 were queried with no match, and 153 are secondary hits.
- 33 of the 38 ribosomal proteins are "isolated" only for this reason; this is why ribosome
  GO terms top the enrichment in `go_enrichment_top.csv`.

The paper's framing of isolated proteins as poorly characterized long-tail antigens does not
hold for most of them.

`Code/map_to_string_v12.py` maps all 3,257 proteins to STRING v12 *F. columnare* ATCC 49512
(taxon 1041826, 2,642 proteins) with pyhmmer phmmer. It agrees with the original web mapping
for 98.6% of the 1,760 proteins mapped there, and maps 745 of the 1,257 never-submitted
proteins (median identity 94%). Output: `Data/string_v12_mapping_all.tsv`.

## D4. External validation on never-submitted proteins (`Code/phaseD_external_validation.py`)

728 of those proteins have STRING v12 links (combined score ≥ 400) to candidate partners:
22,500 links that no model had seen. The base rate is 1.22%.

| Method | Per-protein AUC | Pooled AUC | Precision, top 0.1% | Per-protein precision@5 | Hit within top 10 |
|---|---|---|---|---|---|
| Biomni + ESM (Phase C) | 0.653 | 0.708 | 19.5% | 8.9% | 22% |
| ESM only (Phase C) | 0.650 | 0.721 | 35.6% | 9.1% | 22% |
| ESM cosine similarity | 0.638 | 0.595 | 5.9% | 13.5% | 40% |
| Partner degree | 0.547 | 0.558 | 10.5% | 9.0% | 32% |
| Random | 0.5 | 0.5 | 1.2% | 1.2% | n/a |

- The pooled AUC on truly new data (0.71) matches the cold-start estimate (0.715), so the
  degree-matched cold-start protocol gives an honest performance estimate.
- On this set Biomni adds nothing over ESM (per-protein AUC p = 0.99).
- For per-protein top-5 lists, raw ESM cosine beats the trained model. The dot-product
  model's hub partners (e.g. peg.856) dilute per-protein lists. A per-protein normalisation
  or a cosine/model blend is worth testing.
- Tier pairs among the **real** recovered links: Very Low–Very Low 1.75×, Medium–Medium 0.97×,
  High–High 0.77× (relative to the candidate pool of the same proteins). This does **not**
  support the Medium/High homophily in the model's top predictions (Phase C), which
  therefore reflects model preference and should not be presented as biology.

Tier pairs in the model's top-5 partners vs the real recovered links, for the same 728 proteins:

| Tier pair | Share of real links | Share of model top 5 | Model / real |
|---|---|---|---|
| Medium–Medium | 1.3% | 3.8% | 2.9× |
| High–High | 0.6% | 1.4% | 2.3× |
| Low–Medium | 3.9% | 8.5% | 2.2× |
| Very Low–Very Low | 51.8% | 30.7% | 0.59× |

The model over-predicts Medium/High pairings and under-predicts Very Low–Very Low ones,
which confirms that the Phase C tier enrichment is model bias.

## Consequences for the manuscript

- Report the external validation (D4) as the main biological and falsifiable test. It
  directly answers reviewer comment 6.
- Do not claim Biomni-tier homophily of predicted links; report D1 and D4 honestly.
- Truly unmapped proteins (no STRING v12 hit at all): 240 queried without a match plus
  512 never-submitted proteins that phmmer also cannot map, ≈ 752. Only these are genuine
  "no interaction evidence" proteins. The 745 newly mapped proteins already have STRING
  evidence and serve as the external test set.

# Phase E: re-ranking per-protein partner lists (2026-10-02)

Script: `Code/phaseE_rerank.py`. Outputs: `Results/phaseE_rerank_biomni_esm/` and
`Results/phaseE_rerank_esm/`.

Variants were declared in advance:
- **dot:** Phase C sigmoid of the embedding dot product.
- **cos:** cosine of the model embeddings.
- **csls:** hubness-corrected cosine, k = 10.
- **esm:** raw ESM-2 650M cosine.
- **blend:** row-rank blends of cos or csls with esm.

The selection rule, also fixed in advance, was the highest mean per-protein precision@5 on the
internal cold-start test proteins (10 degree-matched splits). External validation uses the
728 never-submitted proteins (D4), scored once for all variants.

| Variant (Biomni + ESM model) | Internal P@5 | External P@5 | External hit@10 | External per-protein AUC | External pooled AUC | External precision, top 0.1% |
|---|---|---|---|---|---|---|
| esm (**selected**) | 11.5% | **13.5%** | **40%** | 0.638 | 0.595 | 5.9% |
| blend(cos, 0.5) | 10.7% | 11.8% | 35% | **0.670** | 0.734 | 19.9% |
| blend(cos, 0.75) | 9.9% | 11.1% | 35% | 0.668 | **0.746** | 18.6% |
| cos | 8.0% | 10.8% | 33% | 0.660 | 0.698 | 23.1% |
| dot (Phase C) | 7.5% | 8.9% | 22% | 0.653 | 0.708 | 19.5% |

The ESM-only model gives the same ordering. Its dot variant has the best global
top-0.1% precision (35.6%).

## Interpretation

- Internal cold-start rankings of the variants predict the external ordering, which
  supports using the cold-start protocol for model choice.
- For a single protein's partner short-list, nearest neighbours in ESM-2 space (no training)
  are as good as or better than any trained model. Neither the GNNs nor Biomni features add to
  per-protein partner suggestion.
- The trained model is still useful for ranking pairs across proteins (pooled AUC 0.71–0.75
  vs 0.60 for raw ESM cosine), i.e. for deciding which proteins' predictions to trust most.
  Blending gives the best per-protein AUC.

**Recommended outputs:** per-protein partner lists from blend(cos, 0.5) (balanced) or
ESM nearest neighbours (best P@5); a global priority list from the model score.
Report both, together with the pre-declared selection, transparently.

# Final network and manuscript (2026-10-02)

## Final network (`USDA_GRAPH=v12_full`)

All 3,257 proteins are mapped with phmmer to STRING v12 (taxon 1041826) and links with combined
score ≥ 400 are kept: 31,926 edges and 993 proteins without edges. The original export's edges are
99.7% contained in v12, so the export was STRING v12, not v11.5 as the old manuscript stated.

Results are in `Results/final_v12/`:
- phaseA: P1, ESM-2 650M
- phaseB: P2 and P3
- phaseC: candidates
- phaseE: internal re-ranking only
- phaseD_biology
- `candidate_partners_orphan_proteins.tsv`: the release table, top-5 partners per orphan
  protein under the esm_nn and blend rankings

Key final numbers:

| Protocol | Result |
|---|---|
| P1 | best heuristic 0.969 vs best GNN 0.956 |
| P2 | partner degree 0.860 vs best model 0.835 |
| P3 | ESM-2-containing MLP configurations 0.767–0.777, statistically tied (BH p ≥ 0.07); Node2Vec ≤ 0.524 |
| Internal re-ranking | ESM-2 cosine pre-selected again |

## Manuscript

`Flova_Reverse_Vac_Biomni/resubmission/` contains:
- `manuscript.tex` / `manuscript.pdf` (16 pp.)
- `figures/`
- `numbers.tex` and `tab_*.tex`, generated by `Code/make_manuscript_numbers.py`
- `REVIEWER_RESPONSE.md`

Figures are produced by `Code/make_manuscript_figures.py`.

To rebuild:

```
python Code/make_manuscript_numbers.py
USDA_GRAPH=string_web python Code/make_manuscript_figures.py
cd Flova_Reverse_Vac_Biomni/resubmission && latexmk -pdf manuscript.tex
```
