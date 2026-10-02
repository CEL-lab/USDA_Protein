"""Mean-pooled ESM-2 embeddings for every protein in the proteome FASTA.

Per-residue representations from the final layer are averaged (BOS/EOS
excluded). Sequences longer than the model context (1022 residues) are split
into consecutive 1022-residue chunks and the residue embeddings of all chunks
are averaged, so every residue is weighted equally.

Usage (from repo root):
    python Code/compute_esm_embeddings.py --model esm2_t12_35M_UR50D
    python Code/compute_esm_embeddings.py --model esm2_t33_650M_UR50D
"""

import argparse
import time
from pathlib import Path

import esm
import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
FASTA = REPO / "Data" / "Flavobacterium sp. MO_0223_q7L1K.faa"
MAX_RES = 1022


def read_fasta(path):
    seqs, pid = {}, None
    for line in open(path):
        line = line.strip()
        if line.startswith(">"):
            pid = line[1:].split()[0]
            seqs[pid] = []
        elif pid:
            seqs[pid].append(line)
    return {k: "".join(v).rstrip("*") for k, v in seqs.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="esm2_t12_35M_UR50D")
    ap.add_argument("--token_budget", type=int, default=8000)
    args = ap.parse_args()

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model, alphabet = getattr(esm.pretrained, args.model)()
    model = model.eval().to(device)
    layer = model.num_layers
    converter = alphabet.get_batch_converter()

    seqs = read_fasta(FASTA)
    # chunk long sequences; remember which protein each chunk belongs to
    chunks = []
    for pid, s in seqs.items():
        for start in range(0, len(s), MAX_RES):
            chunks.append((pid, s[start:start + MAX_RES]))
    chunks.sort(key=lambda c: len(c[1]))

    sums = {pid: None for pid in seqs}
    counts = {pid: 0 for pid in seqs}
    t0 = time.time()
    i = 0
    while i < len(chunks):
        batch = [chunks[i]]
        while i + len(batch) < len(chunks) and (len(batch) + 1) * len(chunks[i + len(batch)][1]) <= args.token_budget:
            batch.append(chunks[i + len(batch)])
        _, _, toks = converter(batch)
        with torch.no_grad():
            reps = model(toks.to(device), repr_layers=[layer])["representations"][layer].float().cpu()
        for j, (pid, s) in enumerate(batch):
            r = reps[j, 1:len(s) + 1].sum(dim=0).numpy()
            sums[pid] = r if sums[pid] is None else sums[pid] + r
            counts[pid] += len(s)
        i += len(batch)
        if (i // len(batch)) % 20 == 0:
            print(f"{i}/{len(chunks)} chunks, {time.time() - t0:.0f}s", flush=True)

    emb = {pid: (sums[pid] / counts[pid]).astype(np.float32) for pid in seqs}
    out = REPO / "Data" / f"esm_embeddings_{args.model}_all.npz"
    np.savez_compressed(out, **emb)
    dim = next(iter(emb.values())).shape[0]
    print(f"Saved {len(emb)} embeddings (dim={dim}) to {out} in {time.time() - t0:.0f}s on {device}")


if __name__ == "__main__":
    main()
