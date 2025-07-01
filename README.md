# 🔗 GNN-based Link Prediction of Vaccine Candidate Proteins

This project explores graph neural network (GNN) models for predicting novel protein-protein interactions (PPIs) using biological features derived from vaccine candidate analysis. Specifically, we evaluate whether newly predicted links occur between proteins of the same or different priority classes (e.g., High–High vs. High–Low).

---

## 📁 Repository Structure

│
├── Data/
│   │── top_2000_sequences.faa # Input protein sequences
│   ├── string_interactions.tsv # Known STRING PPI network
│   ├── string_mapping.tsv # Mapping from STRING IDs to FASTA IDs
│   └── complete_vaccine_analysis_all_3257_proteins.csv # Feature + priority annotations
├── Code/
│   ├── link_prediction_experiment.ipynb # Main Jupyter notebook
│
├── Results/
│ ├── predicted_links_table.csv # Priority class pair table
│ └── heatmap_predicted_links.png # Visualization of class-wise link patterns
└── README.md       


---

## 📊 Objective

To investigate the nature of **newly predicted links** between proteins, particularly:

- Do **isolated proteins** (those without known STRING edges) form connections?
- Are the new links **homogeneous** (e.g., High–High) or **heterogeneous** (e.g., High–Low)?
- What insights can be drawn from priority-class pairing patterns in predictions?

---

## ⚙️ Workflow Summary

1. **Data Loading & Graph Construction**
   - Load FASTA sequences and STRING PPI data.
   - Build a NetworkX graph, then relabel nodes to FASTA IDs.
   - Add protein features and priority labels from Biomni vaccine analysis.

2. **Graph Neural Network (GNN) Modeling**
   - Use a 2-layer GraphSAGE encoder in PyTorch Geometric.
   - Train a link prediction model using dot-product scoring.
   - Evaluate with binary cross-entropy and ROC-AUC.

3. **Prediction & Analysis**
   - Predict new links involving **isolated proteins**.
   - Filter high-confidence links (sigmoid ≥ 0.90, top 1000).
   - Analyze the **distribution of priority class pairs** for these predicted edges.
   - Visualize with a **heatmap**.

---

## 📈 Key Findings

- The majority of high-confidence predicted links **span across different priority levels**.
- **Very Low–High** and **Very Low–Medium** are the most frequent link types.
- Homogeneous High–High links are rare, suggesting potential for uncovering novel cross-functional associations.

---

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

- STRING Database for interaction data.
- Biomni Toolkit for feature extraction and priority scoring.
- Developed as part of a research project on vaccine candidate network inference.

---

## 📬 Contact

For questions or collaborations, please contact:
**Muhammad Kazim** – *Graduate Research Assistant*  
CELL Lab  
Muhammad.kazim@ndsu.edu

