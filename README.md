<p align="center">
  <img src="https://raw.githubusercontent.com/placeholder-user/placeholder-repo/main/landing_graphic_gnn_vaccine.svg" width="95%" alt="Multimodal GNN Vaccine Discovery Pipeline">
</p>

<h2 align="center">🧬 Multimodal GNN Framework for Vaccine Candidate Discovery</h2>

<p align="center">
  <strong>Biomni Prioritization</strong> • 
  <strong>ESM-2 Protein Language Models</strong> • 
  <strong>Node2Vec Graph Embeddings</strong> • 
  <strong>STRING PPI Networks</strong> • 
  <strong>GNN-based Link Prediction</strong>
</p>
<svg width="1250" height="420" viewBox="0 0 1250 420" xmlns="http://www.w3.org/2000/svg">

  <!-- Background -->
  <rect width="1250" height="420" fill="#f8f9fb" rx="14" />

  <!-- Title -->
  <text x="625" y="50" font-size="32" text-anchor="middle" font-family="Arial" fill="#222">
    Multimodal GNN Pipeline for Vaccine Target Discovery
  </text>

  <!-- Biomni Block -->
  <rect x="60" y="120" width="240" height="90" rx="10" fill="#ffdfdf" stroke="#cc6666" stroke-width="2"/>
  <text x="180" y="155" text-anchor="middle" font-size="18" font-family="Arial" fill="#8b0000">
    Biomni Features
  </text>
  <text x="180" y="180" text-anchor="middle" font-size="14" font-family="Arial" fill="#8b0000">
    Antigenicity • Virulence • Priority
  </text>

  <!-- ESM-2 Block -->
  <rect x="60" y="240" width="240" height="90" rx="10" fill="#dff0ff" stroke="#5b8cbf" stroke-width="2"/>
  <text x="180" y="275" text-anchor="middle" font-size="18" font-family="Arial" fill="#004a80">
    ESM-2 Embeddings
  </text>
  <text x="180" y="300" text-anchor="middle" font-size="14" font-family="Arial" fill="#004a80">
    Sequence Context (480-D)
  </text>

  <!-- Node2Vec Block -->
  <rect x="60" y="360" width="240" height="90" rx="10" fill="#e6ffe6" stroke="#4a8a4a" stroke-width="2"/>
  <text x="180" y="395" text-anchor="middle" font-size="18" font-family="Arial" fill="#1b5e20">
    Node2Vec Embeddings
  </text>
  <text x="180" y="420" text-anchor="middle" font-size="14" font-family="Arial" fill="#1b5e20">
    Graph Structure (128-D)
  </text>

  <!-- Arrows to Unified -->
  <line x1="300" y1="165" x2="420" y2="165" stroke="#444" stroke-width="2" marker-end="url(#arrow)" />
  <line x1="300" y1="285" x2="420" y2="225" stroke="#444" stroke-width="2" marker-end="url(#arrow)" />
  <line x1="300" y1="405" x2="420" y2="265" stroke="#444" stroke-width="2" marker-end="url(#arrow)" />

  <!-- Arrow Definition -->
  <defs>
    <marker id="arrow" markerWidth="10" markerHeight="10" refX="5" refY="5" orient="auto">
      <polygon points="0 0, 10 5, 0 10" fill="#444" />
    </marker>
  </defs>

  <!-- Unified Feature Matrix -->
  <rect x="420" y="180" width="260" height="100" rx="10" fill="#fff7d1" stroke="#c4a200" stroke-width="2"/>
  <text x="550" y="215" text-anchor="middle" font-size="20" font-family="Arial" fill="#8a6d00">
    Unified Feature Matrix
  </text>
  <text x="550" y="240" text-anchor="middle" font-size="14" font-family="Arial" fill="#8a6d00">
    Biomni + ESM-2 + Node2Vec
  </text>

  <!-- Arrow to GNN -->
  <line x1="680" y1="230" x2="780" y2="230" stroke="#444" stroke-width="2" marker-end="url(#arrow)" />

  <!-- GNN Models Block -->
  <rect x="780" y="160" width="260" height="140" rx="10" fill="#efe6ff" stroke="#7b5bc1" stroke-width="2"/>
  <text x="910" y="200" text-anchor="middle" font-size="20" font-family="Arial" fill="#4b2b99">
    GNN Models
  </text>
  <text x="910" y="230" text-anchor="middle" font-size="15" font-family="Arial" fill="#4b2b99">
    GCN • GAT • GraphSAGE
  </text>
  <text x="910" y="255" text-anchor="middle" font-size="13" font-family="Arial" fill="#4b2b99">
    Link Prediction (Inductive)
  </text>

  <!-- Arrow to Outputs -->
  <line x1="1040" y1="230" x2="1140" y2="230" stroke="#444" stroke-width="2" marker-end="url(#arrow)" />

  <!-- Output Block -->
  <rect x="1140" y="180" width="260" height="100" rx="10" fill="#ffeaea" stroke="#c44" stroke-width="2"/>
  <text x="1270" y="215" text-anchor="middle" font-size="20" font-family="Arial" fill="#880000">
    Predicted PPIs
  </text>
  <text x="1270" y="240" text-anchor="middle" font-size="14" font-family="Arial" fill="#880000">
    Priority-Aware Interaction Map
  </text>

</svg>

This repository implements a multimodal graph learning framework that integrates:

Biomni vaccine-priority and immunological features

ESM-2 protein language model embeddings

Node2Vec structural graph embeddings

STRING protein–protein interaction (PPI) networks

to predict novel protein–protein interactions in Flavobacterium using Graph Neural Networks (GNNs).
The workflow supports priority-aware link prediction, enabling the identification of underexplored high-priority vaccine candidates.

---

## 🧬 STRING Network Visualization

Below is the original STRING PPI network obtained from the organism proteome and mapped to FASTA identifiers:

![STRING PPI Network](string_normal_image.png)

> **Note:** This network only captures known interactions.
The GNN models infer new, high-confidence PPIs, especially for isolated and poorly understood proteins..

---
## 📁 Repository Structure
```
│
├── Data/
│   ├── flavobacterium_proteome.faa
│   ├── string_interactions.tsv
│   ├── string_mapping.tsv
│   ├── complete_vaccine_analysis_all_3257_proteins.csv
│   ├── esm2_embeddings.h5
│   ├── node2vec_flavobacterium.npz
│   └── ...
│
├── Code/
│   ├── link_prediction_experiment.ipynb
│   ├── esm_embedding_generation.ipynb
│   ├── node2vec_training.ipynb
│   └── utils/
│
├── Results/
│   ├── Ablations/
│   │   ├── esm_only_results.json
│   │   ├── n2v_only_results.json
│   │   └── biomni_n2v_results.json
│   ├── GraphSAGE/
│   │   ├── best_model.pt
│   │   ├── embeddings.pt
│   │   ├── priority_link_table.csv
│   │   └── heatmap_testAUC.png
│   ├── FinalFigures/
│   │   ├── Figure_2_ROC_PR_Curves.png
│   │   ├── Figure_3_Bar_Comparison.png
│   │   ├── Figure_4_Priority_Link_Heatmap.png
│   │   ├── Figure_6_Predicted_Network.png
│   │   └── PCA_of_GNN_Embeddings.png
│   └── ...
│
└── README.md
```
## 📊 Objective

This project investigates whether multimodal GNNs can recover missing PPIs and prioritize vaccine-relevant proteins by integrating:

Biological evidence (Biomni)

Evolutionary sequence context (ESM-2)

Graph structure (Node2Vec)

We specifically analyze:

Interactions involving isolated proteins

Whether predicted links are homogeneous (High–High) or heterogeneous (High–Low)

How each GNN architecture behaves under multimodal input

Priority-class mixing patterns in predicted edges

---

## ⚙️ Workflow Summary

**Step 1 — Data Integration**

i- STRING PPIs + FASTA mapping

ii- Biomni features (antigenicity, localization, virulence)

iii- ESM-2 embeddings (480-D)

iv- Node2Vec embeddings (128-D)

Multimodal feature matrix construction

**Step 2 — GNN Models** 

We train and evaluate:

i- GCN

ii- GAT

iii- GraphSAGE (**best model**)


**Under three regimes:**

1- Biomni-only

2- Biomni + ESM-2

3- Biomni + ESM-2 + Node2Vec

Plus ablation experiments to test independent feature performance.

**Step 3 — Link Prediction**

Score all candidate edges

Filter high-confidence predictions (≥ 0.90)

Focus on isolated–connected and isolated–isolated links

**Step 4 — Priority-Aware Class Pairing**

Build a 4×4 class-pair matrix

Visualize using heatmaps & barplots

Interpret class mixing and high-priority involvement
---

## 📈 Key Findings

GraphSAGE is the most effective GNN

   * Achieved best test AUC = 0.750 (Biomni-only)

   * Higher-dimensional features (ESM-2 + Node2Vec) did not improve generalization

   * Biomni provides the most stable inductive signal in low-homophily graphs

**Ablation results reveal high latent predictive capacity**

   i- Node2Vec-only AUC: 0.963

   ii- ESM-2-only AUC: 0.941

These represent upper bounds on the informativeness of each modality.

**Priority-aware insights**

High–Medium and High–Low edges appear more frequently than High–High

Many isolated proteins gain high-confidence predicted links

## Predicted Network Visualization

High-confidence predicted interactions (≥ 0.90) using GraphSAGE + Biomni:
![Predicted Distribution](new_r.png)

---
Node color legend:

Red — High-priority proteins

Orange — Medium-priority proteins

Blue — Low-priority proteins

Grey — Very Low priority proteins

This network illustrates cross-tier connectivity and the emergence of medium/high-priority hubs.
## 🛠 Dependencies

- Python 3.8+
- PyTorch
- PyTorch Geometric
- NetworkX
- Biopython
- Pandas
- Seaborn / Matplotlib
- scikit-learn

---

## 📚 Acknowledgments

- [STRING DB](https://string-db.org/) – PPI data source  
- [Biomni Toolkit](https://github.com/snap-stanford/biomni) – Feature generation & priority scoring  
- Developed as part of the **CELL Lab** research initiative at NDSU.


---

## 📬 Contact

For questions or collaborations, please contact:
**Muhammad Kazim** – *Graduate Research Assistant*  
CELL Lab  
Muhammad.kazim@ndsu.edu

