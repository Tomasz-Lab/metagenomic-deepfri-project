"""Compare sequence identity definitions for the landscape benchmark pairs.

MMseqs2 searches in mdF run without backtrace (--alignment-mode 2), so the
reported fident is estimated from the alignment score. This script realigns
every query-template pair with the same MMseqs2 build twice, once with the
paper's settings (estimate) and once with backtrace (exact), and derives
identity and coverage variants from the PyOpal global alignments.

PyOpal alignment strings: M match, X mismatch, D query residue against a
target gap (a synthetic residue in the contact map), I target residue against
a query gap.
"""

import subprocess
import sys
from pathlib import Path

import pandas as pd

MMSEQS = "/home/FilipS/software/Metagenomic-DeepFRI-upstream/mmseqs/bin/mmseqs"
PAIRS = "data/generated/gc_benchmark_with_tm.csv"
QUERIES = "data/source/landscape/landscape_afdb_50k.fasta"
OUTPUT = "data/generated/gc_benchmark_identity_definitions.csv"
FIELDS = (
    "query,target,fident,nident,alnlen,qstart,qend,tstart,tend,qlen,tlen,bits"
)


def read_fasta(path):
    records, name = {}, None
    for line in Path(path).read_text().splitlines():
        if line.startswith(">"):
            name = line[1:].split()[0]
            records[name] = []
        elif line:
            records[name].append(line.strip())
    return {name: "".join(seq) for name, seq in records.items()}


def mmseqs(*args):
    subprocess.run([MMSEQS, *map(str, args), "-v", "1"], check=True)


def write_fasta(path, sequences):
    path.write_text("".join(f">{k}\n{v}\n" for k, v in sequences.items()))


def realign(pairs, workdir, threads=8):
    """Align exactly the given pairs with MMseqs2, estimated and exact."""
    workdir.mkdir(parents=True, exist_ok=True)
    query_seqs = dict(zip(pairs["query"], pairs["query_seq"]))
    target_seqs = dict(zip(pairs["target"], pairs["target_seq"]))
    write_fasta(workdir / "query.fasta", query_seqs)
    write_fasta(workdir / "target.fasta", target_seqs)
    mmseqs("createdb", workdir / "query.fasta", workdir / "qdb")
    mmseqs("createdb", workdir / "target.fasta", workdir / "tdb")

    def keys(db):
        lookup = pd.read_csv(
            f"{db}.lookup",
            sep="\t",
            header=None,
            usecols=[0, 1],
            names=["key", "name"],
        )
        return dict(zip(lookup["name"], lookup["key"]))

    qkeys, tkeys = keys(workdir / "qdb"), keys(workdir / "tdb")
    prefilter = pd.DataFrame(
        {
            "q": pairs["query"].map(qkeys),
            "t": pairs["target"].map(tkeys),
            "score": 0,
            "diagonal": 0,
        }
    )
    prefilter.sort_values(["q", "t"]).to_csv(
        workdir / "pairs.tsv", sep="\t", header=False, index=False
    )
    mmseqs(
        "tsv2db",
        workdir / "pairs.tsv",
        workdir / "prefdb",
        "--output-dbtype",
        7,
    )

    results = {}
    for name, flags in [
        ("est", ["--alignment-mode", 2, "-a", 0]),
        ("exact", ["--alignment-mode", 3, "-a", 1]),
    ]:
        aln = workdir / f"aln_{name}"
        mmseqs(
            "align",
            workdir / "qdb",
            workdir / "tdb",
            workdir / "prefdb",
            aln,
            "-e",
            "inf",
            "--threads",
            threads,
            *flags,
        )
        mmseqs(
            "convertalis",
            workdir / "qdb",
            workdir / "tdb",
            aln,
            f"{aln}.tsv",
            "--format-output",
            FIELDS,
        )
        table = pd.read_csv(
            f"{aln}.tsv", sep="\t", header=None, names=FIELDS.split(",")
        )
        results[name] = table.drop_duplicates(["query", "target"])
    return results


def pyopal_metrics(alignment):
    m, x = alignment.count("M"), alignment.count("X")
    d, i = alignment.count("D"), alignment.count("I")
    span = alignment.strip("D")
    return {
        "pyopal_matches": m,
        "pyopal_mismatches": x,
        "pyopal_query_gaps": d,
        "pyopal_target_gaps": i,
        "pyopal_id_aligned": m / (m + x),
        "pyopal_id_columns": m / len(alignment),
        "pyopal_id_query": m / (m + x + d),
        "pyopal_cov_span": (len(span) - span.count("I")) / (m + x + d),
        "pyopal_cov_aligned": (m + x) / (m + x + d),
        "pyopal_synthetic_in_span": span.count("D") / (m + x + d),
    }


def main():
    workdir = Path(sys.argv[1])
    pairs = pd.read_csv(
        PAIRS,
        usecols=[
            "query",
            "target",
            "idbin",
            "fident",
            "alnlen",
            "qcov",
            "tcov",
            "bits",
            "pyopal_identity",
            "pyopal_coverage",
            "pyopal_alignment_string",
            "pyopal_query_aln",
            "pyopal_target_aln",
        ],
    )
    pairs["query_seq"] = pairs["pyopal_query_aln"].str.replace("-", "")
    pairs["target_seq"] = pairs["pyopal_target_aln"].str.replace("-", "")
    queries = read_fasta(QUERIES)
    assert (pairs["query_seq"] == pairs["query"].map(queries)).all()

    metrics = pd.DataFrame(
        [pyopal_metrics(a) for a in pairs["pyopal_alignment_string"]]
    )
    assert (
        metrics["pyopal_matches"]
        + metrics["pyopal_mismatches"]
        + metrics["pyopal_query_gaps"]
        == pairs["query_seq"].str.len()
    ).all()
    assert (
        metrics["pyopal_matches"]
        + metrics["pyopal_mismatches"]
        + metrics["pyopal_target_gaps"]
        == pairs["target_seq"].str.len()
    ).all()
    pairs = pd.concat([pairs, metrics], axis=1)

    results = realign(pairs, workdir)
    for name, table in results.items():
        table = table.rename(
            columns={
                c: f"mm_{name}_{c}"
                for c in table.columns
                if c not in ("query", "target")
            }
        )
        pairs = pairs.merge(table, on=["query", "target"], how="left")

    pairs.drop(
        columns=[
            "pyopal_alignment_string",
            "pyopal_query_aln",
            "pyopal_target_aln",
            "query_seq",
            "target_seq",
        ]
    ).to_csv(OUTPUT, index=False)


if __name__ == "__main__":
    main()
