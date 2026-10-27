"""Look up TED domains and their CATH labels for the landscape benchmark.

TED (Lau et al., Science 2024) chopped every AFDB v4 model into domains and
assigned each a CATH superfamily (level H) or fold (level T). This script
extracts the TED domains of all benchmark queries and templates from
ted_365m.domain_summary.cath.globularity.taxid.tsv.gz (Zenodo 13908086).

Usage: python scripts/ted_assignment.py
"""

import io
import subprocess
import tempfile

import pandas as pd

TED = (
    "/mnt/storage_10T/software/ted/"
    "ted_365m.domain_summary.cath.globularity.taxid.tsv.gz"
)
PAIRS = "data/generated/gc_benchmark_with_tm.csv"
OUTPUT = "data/generated/landscape_ted_domains.csv"
COLUMNS = [
    "ted_id",
    "md5_domain",
    "consensus_level",
    "chopping",
    "nres_domain",
    "num_segments",
    "plddt",
    "num_helix_strand_turn",
    "num_helix",
    "num_strand",
    "num_helix_strand",
    "num_turn",
    "proteome_id",
    "cath_label",
    "cath_assignment_level",
    "cath_assignment_method",
    "packing_density",
    "norm_rg",
    "tax_common_name",
    "tax_scientific_name",
    "tax_lineage",
]


def main():
    pairs = pd.read_csv(PAIRS, usecols=["query", "target"])
    models = set(pairs["query"]) | set(pairs["target"])
    with tempfile.NamedTemporaryFile("w", suffix=".txt") as patterns:
        patterns.write("".join(f"{m}_TED\n" for m in sorted(models)))
        patterns.flush()
        matches = subprocess.run(
            f"pigz -dc {TED} | grep -F -f {patterns.name}",
            shell=True,
            check=True,
            capture_output=True,
            text=True,
        ).stdout
    domains = pd.read_csv(
        io.StringIO(matches),
        sep="\t",
        header=None,
        names=COLUMNS,
        na_values="-",
    )
    domains.insert(1, "model", domains["ted_id"].str.rsplit("_", n=1).str[0])
    domains[domains["model"].isin(models)].to_csv(OUTPUT, index=False)


if __name__ == "__main__":
    main()
