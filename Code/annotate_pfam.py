"""Pfam domain and GO annotation of the full proteome with pyhmmer.

Searches every Pfam-A profile against the 3,257 proteins using the Pfam
gathering (GA) thresholds, then maps domains to clans and to GO terms via
pfam2go. Needs Pfam-A.hmm.gz, Pfam-A.clans.tsv.gz and pfam2go in --db
(downloaded from ftp.ebi.ac.uk/pub/databases/Pfam/current_release and
current.geneontology.org/ontology/external2go/pfam2go).

Usage (from repo root):
    python Code/annotate_pfam.py --db ~/.cache/pfam
Writes Data/annotation/pfam_domain_hits.tsv, Data/annotation/protein_annotation.tsv
and Data/annotation/protein_to_go.tsv.
"""

import argparse
import gzip
import re
import time
from collections import defaultdict
from pathlib import Path

import pandas as pd
import pyhmmer

REPO = Path(__file__).resolve().parents[1]
FASTA = REPO / "Data" / "Flavobacterium sp. MO_0223_q7L1K.faa"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", type=Path, default=Path.home() / ".cache" / "pfam")
    ap.add_argument("--cpus", type=int, default=4)
    args = ap.parse_args()
    out = REPO / "Data" / "annotation"
    out.mkdir(parents=True, exist_ok=True)

    alphabet = pyhmmer.easel.Alphabet.amino()
    with pyhmmer.easel.SequenceFile(str(FASTA), digital=True, alphabet=alphabet) as sf:
        seqs = sf.read_block()
    names = [s.name.decode() if isinstance(s.name, bytes) else s.name for s in seqs]
    print(f"{len(seqs)} sequences", flush=True)

    t0 = time.time()
    rows = []
    with gzip.open(args.db / "Pfam-A.hmm.gz", "rb") as fh, pyhmmer.plan7.HMMFile(fh) as hmms:
        for top in pyhmmer.hmmsearch(hmms, seqs, cpus=args.cpus, bit_cutoffs="gathering"):
            acc = top.query.accession
            acc = (acc.decode() if isinstance(acc, bytes) else acc).split(".")[0]
            qname = top.query.name.decode() if isinstance(top.query.name, bytes) else top.query.name
            desc = top.query.description
            desc = desc.decode() if isinstance(desc, bytes) else (desc or "")
            for hit in top:
                if not hit.included:
                    continue
                hname = hit.name.decode() if isinstance(hit.name, bytes) else hit.name
                for dom in hit.domains.included:
                    rows.append({"protein_id": hname, "pfam_acc": acc, "pfam_name": qname,
                                 "pfam_desc": desc, "env_from": dom.env_from, "env_to": dom.env_to,
                                 "domain_score": dom.score, "domain_ievalue": dom.i_evalue})
    hits = pd.DataFrame(rows).sort_values(["protein_id", "env_from"])
    hits.to_csv(out / "pfam_domain_hits.tsv", sep="\t", index=False)
    print(f"hmmsearch: {len(hits)} domain hits in {hits['protein_id'].nunique()} proteins "
          f"({time.time() - t0:.0f}s)", flush=True)

    clans = pd.read_csv(args.db / "Pfam-A.clans.tsv.gz", sep="\t", header=None,
                        names=["pfam_acc", "clan_acc", "clan_name", "pfam_name", "pfam_desc"])
    clan_of = dict(zip(clans["pfam_acc"], clans["clan_name"].fillna("")))

    go_of = defaultdict(set)
    pat = re.compile(r"^Pfam:(PF\d+) .* ; (GO:\d+)$")
    for line in open(args.db / "pfam2go"):
        m = pat.match(line.strip())
        if m:
            go_of[m.group(1)].add(m.group(2))

    by_prot = hits.groupby("protein_id")
    ann = pd.DataFrame({"protein_id": names})
    arch = by_prot["pfam_name"].apply(lambda s: ";".join(s))
    descs = by_prot["pfam_desc"].apply(lambda s: "; ".join(dict.fromkeys(s)))
    accs = by_prot["pfam_acc"].apply(lambda s: list(dict.fromkeys(s)))
    ann["pfam_architecture"] = ann["protein_id"].map(arch).fillna("")
    ann["pfam_descriptions"] = ann["protein_id"].map(descs).fillna("")
    ann["pfam_clans"] = ann["protein_id"].map(
        accs.apply(lambda a: ";".join(sorted({clan_of.get(x, "") for x in a} - {""})))).fillna("")
    go = accs.apply(lambda a: sorted(set().union(*[go_of[x] for x in a])))
    ann["go_terms"] = ann["protein_id"].map(go.apply(";".join)).fillna("")
    ann.to_csv(out / "protein_annotation.tsv", sep="\t", index=False)

    long = [(p, g) for p, gs in go.items() for g in gs]
    pd.DataFrame(long, columns=["protein_id", "go_term"]).to_csv(out / "protein_to_go.tsv", sep="\t", index=False)
    print(f"proteins with >=1 Pfam domain: {(ann['pfam_architecture'] != '').sum()} / {len(ann)}; "
          f"with >=1 GO term: {(ann['go_terms'] != '').sum()}", flush=True)


if __name__ == "__main__":
    main()
