"""Map every proteome protein to the STRING v12 F. columnare ATCC 49512 proteome.

The original STRING mapping (Data/string_mapping.tsv) came from the STRING web
"multiple sequences" upload, which accepts at most 2,000 sequences; only
Data/top_2000_sequences.faa was submitted. Here all 3,257 proteins are searched
with phmmer against the 2,642 STRING v12 proteins of taxon 1041826. The best hit
per query is kept if its bitscore reaches the weakest hit STRING itself accepted
(51.2 bits).

STRING files (from stringdb-downloads.org, taxon 1041826, v12.0) are read from --db.

Usage (from repo root):
    python Code/map_to_string_v12.py --db ~/.cache/string_v12
Writes Data/string_v12_mapping_all.tsv
"""

import argparse
import gzip
import io
from pathlib import Path

import pandas as pd
import pyhmmer

REPO = Path(__file__).resolve().parents[1]
FASTA = REPO / "Data" / "Flavobacterium sp. MO_0223_q7L1K.faa"
MIN_BITS = 51.2


def as_str(x):
    return x.decode() if isinstance(x, bytes) else x


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", type=Path, default=Path.home() / ".cache" / "string_v12")
    ap.add_argument("--cpus", type=int, default=4)
    args = ap.parse_args()

    abc = pyhmmer.easel.Alphabet.amino()
    with pyhmmer.easel.SequenceFile(str(FASTA), digital=True, alphabet=abc) as f:
        queries = f.read_block()
    raw = gzip.open(args.db / "1041826.protein.sequences.v12.0.fa.gz").read()
    with pyhmmer.easel.SequenceFile(io.BytesIO(raw), format="fasta", digital=True, alphabet=abc) as f:
        targets = f.read_block()
    print(f"{len(queries)} queries vs {len(targets)} STRING v12 proteins", flush=True)

    rows = []
    for hits in pyhmmer.phmmer(queries, targets, cpus=args.cpus):
        q = as_str(hits.query.name)
        qlen = len(hits.query.sequence) if hasattr(hits.query, "sequence") else None
        best = None
        for h in hits:
            if h.included and (best is None or h.score > best.score):
                best = h
        if best is None:
            rows.append({"queryItem": q, "stringId": None})
            continue
        dom = best.best_domain.alignment
        ident = sum(c not in " +" for c in dom.identity_sequence) / len(dom.identity_sequence)
        rows.append({"queryItem": q, "stringId": as_str(best.name), "bitscore": best.score,
                     "evalue": best.evalue, "identity": 100 * ident,
                     "aln_len": len(dom.identity_sequence), "query_len": qlen})
    df = pd.DataFrame(rows)
    df["accepted"] = df["bitscore"].fillna(0) >= MIN_BITS
    out = REPO / "Data" / "string_v12_mapping_all.tsv"
    df.to_csv(out, sep="\t", index=False)
    print(f"accepted best hits: {df['accepted'].sum()} / {len(df)} -> {out}")


if __name__ == "__main__":
    main()
