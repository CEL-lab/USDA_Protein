# How the resubmission addresses the previous review

The previous version was rejected by *Aquaculture International* after a single review
(`comments_for_sn_article_gb.docx`, 7 major comments). The resubmitted manuscript
(`manuscript.tex`) is a substantially new study. Each comment is mapped to the change below.
Section, figure and table numbers refer to `manuscript.tex`.

Several problems in the original analysis came to light during the revision. They are not
described in the manuscript, which reports only the corrected analysis, but co-authors should
know about them:

| Problem in the original analysis | Effect | Fix |
|---|---|---|
| Biomni descriptors were used unscaled (molecular weight ≈ 35,000 next to 0–1 features) | Every Biomni + GCN/GAT run produced saturated outputs and ROC-AUC = 0.500 ± 0.000. This drove the "Biomni weakest" conclusion of the March draft. | Log-transform, drop molecular weight, z-score |
| ESM-2 embeddings came from the 35M model (480-d), but the text said 650M; the repo file covered only 1,644 of 3,257 proteins | Wrong method description, and degraded ESM results | Recomputed 35M and 650M embeddings for all proteins; 650M is used throughout |
| Only 2,000 sequences (the STRING web upload cap) were ever mapped to STRING | 1,257 of 1,650 "isolated" proteins were never queried | Full phmmer mapping to STRING v12; the withheld proteins became the external test set |
| STRING mapping collisions resolved by row order, not best hit | Edges given to a weaker match in 30 of 52 collisions | Best-bitscore assignment |
| Text and code disagreed (3-fold CV vs repeated splits, seeds, hidden size, "calibrated 0.60" threshold, STRING v11.5 vs v12) | Unreproducible methods | Methods rewritten from the code; all numbers generated from result files |

## Comment 1: predictive signal dominated by Biomni; Biomni is a black box used as static features

- **Analysis.** Biomni descriptors are now one of three input families, evaluated alone and in
  combination (Methods 2.2; Table 2). Under the held-out-protein protocol with degree-matched
  negatives, Biomni alone is weaker than ESM-2, and adding Biomni to ESM-2 gives a small internal
  gain that does not replicate externally (Results 3.2, 3.3).
- **Independent audit.** Biomni priority tiers were tested against Pfam surface and secretion
  families fixed in advance (Results 3.4, Fig. 4a, Supplementary Table S4). The High tier is not
  enriched for them (OR 0.61).
- **Transparency.** The descriptors and tier thresholds are listed explicitly, and the run is to
  be documented in Supplementary File S1. **Author input needed:** Biomni version, run date,
  language-model backbone, prompts and the generated code.
- **Framing.** The manuscript no longer claims that Biomni drives prediction or that the GNN
  "interacts" with Biomni. Its role is evaluated, not assumed.

## Comment 2: single graph, unsupported claims of transferability

- All "modular / transferable / species-agnostic" claims are removed. The Limitations section
  states that one proteome is analysed and that no transfer is claimed.
- **External validation instead.** 728 proteins withheld from the development network, with
  22,500 STRING v12 links unseen during development (Methods 2.8; Results 3.3; Table 3; Fig. 3).
  This is a genuinely independent test, although within one species.
- **Optional addition, if a reviewer insists:** a cross-species test (e.g. *F. psychrophilum*
  JIP02/86 in STRING). It is feasible with the existing code because ESM-2 and Biomni features
  carry over between species. It is not done yet.

## Comment 3: single split, no variance or significance testing

- Ten repeated splits for every protocol, with mean ± s.d. throughout.
- Paired t-tests and Wilcoxon signed-rank tests against the best configuration, with
  Benjamini–Hochberg correction (Methods 2.7).
- Per-split indices are released.

## Comment 4: frozen ESM-2 unjustified; ablations implausibly high (leakage)

- **Leakage.** The high ablation values were caused by the evaluation protocol. Three protocols
  are now compared (Methods 2.7; Fig. 2):
  - Random edge splits let simple heuristics reach about 0.96.
  - With whole proteins held out, Node2Vec falls to chance, and with uniform negatives partner
    degree alone wins.
  - Node2Vec is refitted per split on training edges only.
- **ESM-2.** The 650M and 35M models are compared (Supplementary Table S1). Frozen embeddings with
  a trainable encoder are justified by recent leakage-controlled studies showing that PLM
  embeddings account for most of the attainable signal (Reim et al. 2025, cited).
- **Conclusions are now consistent with the ablations.** Sequence embeddings carry the signal;
  topology carries none for unseen proteins.

## Comment 5: no methodological novelty

- The paper no longer presents a new model. Its contribution is stated as:
  1. a leakage-aware benchmark showing that the evaluation protocol, not the architecture,
     determines the method ranking;
  2. external validation of that protocol;
  3. an audit of agent-derived vaccine priorities;
  4. a released, annotated candidate list.
- The GraphSAGE-superiority claim is gone. No GNN beats a graph-free encoder for unseen proteins.
- **Target journal.** It fits a venue that publishes benchmark/application studies (e.g. BMC
  Bioinformatics or Computational and Structural Biotechnology Journal) rather than a methods
  venue.

## Comment 6: biological analysis descriptive, not falsifiable

- **Falsifiable test.** External validation against STRING v12 links that no model had seen
  (Results 3.3).
- **Antigen families.** Pfam 38.2 annotation of all proteins; family sets for surface and
  secretion proteins (T9SS cargo, OMPs, TonB/SusCD, T9SS/gliding machinery, adhesins, T6SS) fixed
  in advance (Supplementary Table S3).
- **Functional coherence.** Predicted pairs share GO terms and Pfam clans far more often than
  random pairs (Results 3.5; Fig. 5).
- **Negative result reported.** The model's tier homophily is shown to be model bias (Fig. 4b).
- **Still open (would strengthen the paper):** a curated list of experimentally supported
  *Flavobacterium* antigens from the co-authors (e.g. LaFrentz, Loch, Soto), to test recovery
  directly.

## Comment 7: overplotted network figures

- All whole-network drawings are removed.
- The figures are now: study design (Fig. 1), benchmark bar charts (Fig. 2), external validation
  dot plots (Fig. 3), tier audit (Fig. 4) and coherence bars (Fig. 5).
- Every number in the text and tables is generated from the result files by
  `Code/make_manuscript_numbers.py`.

## Before submission: author input needed

All marked in red as `[AUTHOR INPUT: ...]` in the PDF:

1. Genome accession / BioProject for *Flavobacterium* sp. MO_0223_q7L1K, and its isolation source.
2. Biomni version, date, backbone, prompts and generated code (Supplementary File S1).
3. Archived code/data release DOI (e.g. Zenodo).
4. Funding, acknowledgements, CRediT author contributions.
5. Confirm the target journal and reformat if needed (the sn-jnl template is kept).
6. Optional: the antigen list (Comment 6) and the cross-species test (Comment 2).
