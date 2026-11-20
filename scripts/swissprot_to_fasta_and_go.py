"""Turn SwissProt entries into the test set for the mdF score calibration.

Keeps proteins of 60-1000 residues that have at least one experimental GO
annotation and writes
  - a FASTA file with the accession as sequence identifier (mdF input),
  - a table with their experimental GO annotations (ground truth).

Usage:
    python swissprot_to_fasta_and_go.py ENTRIES.dat.gz OUT.fasta OUT_go.tsv
"""

import gzip
import sys

from Bio import SwissProt

# Evidence codes counted as experimental by the DeepFRI paper and by CAFA3.
EXPERIMENTAL_CODES = {"EXP", "IDA", "IPI", "IMP", "IGI", "IEP", "TAS", "IC"}
# DeepFRI builds contact maps only for chains of this length.
MIN_LENGTH, MAX_LENGTH = 60, 1000
ASPECTS = {"F": "MF", "P": "BP", "C": "CC"}
PROTEIN_BINDING = "GO:0005515"


def experimental_go_terms(record):
    """Return the experimental GO annotations of an entry as
    (go_term, aspect, evidence_code) tuples."""
    terms = []
    for database, *fields in record.cross_references:
        if database != "GO":
            continue
        # e.g. ("GO:0005524", "F:ATP binding", "IDA:UniProtKB")
        go_term, name, evidence = fields
        evidence_code = evidence.split(":")[0]
        if evidence_code in EXPERIMENTAL_CODES:
            terms.append((go_term, ASPECTS[name[0]], evidence_code))

    # CAFA rule: "protein binding" as the only MF term is not informative.
    mf_terms = {go_term for go_term, aspect, _ in terms if aspect == "MF"}
    if mf_terms == {PROTEIN_BINDING}:
        terms = [term for term in terms if term[0] != PROTEIN_BINDING]
    return terms


def main(entries_path, fasta_path, go_path):
    n_entries = n_kept = 0
    with (
        gzip.open(entries_path, "rt") as entries,
        open(fasta_path, "w") as fasta,
        open(go_path, "w") as go_table,
    ):
        go_table.write("protein\tgo_term\taspect\tevidence\n")
        for record in SwissProt.parse(entries):
            n_entries += 1
            terms = experimental_go_terms(record)
            if not terms or not MIN_LENGTH <= record.sequence_length <= MAX_LENGTH:
                continue
            accession = record.accessions[0]
            fasta.write(f">{accession}\n{record.sequence}\n")
            for go_term, aspect, evidence_code in terms:
                go_table.write(f"{accession}\t{go_term}\t{aspect}\t{evidence_code}\n")
            n_kept += 1
    print(f"{n_entries} entries read, {n_kept} kept "
          f"({MIN_LENGTH}-{MAX_LENGTH} aa with experimental GO terms)")


if __name__ == "__main__":
    main(*sys.argv[1:4])
