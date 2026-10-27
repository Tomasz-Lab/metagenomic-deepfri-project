"""Build the nested MAG protein sets for the runtime benchmark.

The 100,000-protein set holds all 99,847 proteins of the 50 UHGG MAGs plus
153 random proteins of MGYG000002020, the 51st species drawn by the MAG
selection in marimo/mag__analysis.py (sample(n=51, random_state=42) keeps
the first 50). Smaller sets are prefixes of one seeded shuffle, so
10 < 100 < 1,000 < 10,000 < 100,000 are nested.
"""

import hashlib
import random
from pathlib import Path

SEED = 42
SIZES = [10, 100, 1_000, 10_000, 100_000]
MAGS_50 = Path("data/source/uhgp_mags/uhgp_50_mags/uhgp_50_mags.faa")
TOPUP = Path("data/source/uhgp_mags/runtime_topup/MGYG000002020.faa")
OUTPUT = Path("data/generated/runtime_benchmark/queries")


def read_fasta(path):
    records, name = {}, None
    for line in path.read_text().splitlines():
        if line.startswith(">"):
            name = line[1:].split()[0]
            records[name] = []
        elif line:
            records[name].append(line.strip())
    return {name: "".join(seq) for name, seq in records.items()}


def main():
    proteins = read_fasta(MAGS_50)
    topup = read_fasta(TOPUP)
    rng = random.Random(SEED)
    for name in rng.sample(sorted(topup), SIZES[-1] - len(proteins)):
        proteins[name] = topup[name]

    order = sorted(proteins)
    rng.shuffle(order)

    OUTPUT.mkdir(parents=True, exist_ok=True)
    checksums = []
    for size in SIZES:
        path = OUTPUT / f"mag_proteins_{size}.faa"
        path.write_text(
            "".join(f">{name}\n{proteins[name]}\n" for name in order[:size])
        )
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        checksums.append(f"{digest}  {path.name}\n")
    (OUTPUT / "SHA256SUMS").write_text("".join(checksums))


if __name__ == "__main__":
    main()
