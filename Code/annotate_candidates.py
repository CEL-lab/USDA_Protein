"""Attach Pfam/GO annotations (Code/annotate_pfam.py) to the Phase C candidate tables.

Usage (from repo root):
    python Code/annotate_candidates.py --dir Results/phaseC_candidates
Writes *_annotated.tsv next to each input table.
"""

import argparse
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[1]
TABLES = {
    "candidates_isolated_connected_capped.csv": ["protein_a", "protein_b"],
    "candidates_isolated_isolated_capped.csv": ["protein_a", "protein_b"],
    "high_priority_isolated_partners.csv": ["isolated_protein", "partner"],
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", type=Path, default=Path("Results/phaseC_candidates"))
    args = ap.parse_args()
    d = REPO / args.dir
    ann = pd.read_csv(REPO / "Data" / "annotation" / "protein_annotation.tsv", sep="\t").set_index("protein_id")
    ann = ann.fillna("")

    for name, cols in TABLES.items():
        df = pd.read_csv(d / name)
        for c in cols:
            for field in ["pfam_architecture", "pfam_descriptions", "go_terms"]:
                df[f"{c}_{field}"] = df[c].map(ann[field])
        df.to_csv(d / name.replace(".csv", "_annotated.tsv"), sep="\t", index=False)
        a, b = cols
        both = ((df[f"{a}_pfam_architecture"] != "") & (df[f"{b}_pfam_architecture"] != "")).mean()
        print(f"{name}: {len(df)} rows; both proteins Pfam-annotated in {both:.0%}")


if __name__ == "__main__":
    main()
