#!/bin/bash
# Build the test set for the mdF score calibration: a temporal holdout of
# prokaryotic SwissProt proteins that DeepFRI v1.0 cannot have seen with
# their experimental annotations.
#
#   1. Download the SwissProt entries of Bacteria and Archaea that entered
#      SwissProt after DeepFRI v1.0 was trained.
#   2. Keep proteins of 60-1000 aa with experimental GO terms.
#   3. Cluster them so that no two representatives share more than 50%
#      identity over 80% of both sequences; the representatives are the
#      mdF input (clusters_rep_seq.fasta).
#
# Run from the repository root (light enough for the login node):
#   bash scripts/swissprot_calibration_prepare.sh
set -euo pipefail

# DeepFRI v1.0 was trained on annotations of UniProtKB release 2019_06,
# published on 2019-07-03. Entries created from the next day on are new.
FIRST_DAY=2019-07-04
ENTRIES=data/external/swissprot_prokaryotes_since_${FIRST_DAY}.dat.gz
OUT=data/source/swissprot_calibration
# MMseqs2 build that ships with mdF 1.3.0
MMSEQS=data/source/runtime_benchmark/envs/mdf-gpu/lib/python3.11/site-packages/mmseqs/bin/mmseqs
THREADS=4

mkdir -p "$OUT"

# date_created of a reviewed entry is the day it entered SwissProt.
# Taxonomy 2 is Bacteria, 2157 is Archaea.
QUERY="reviewed:true AND date_created:[${FIRST_DAY} TO *] AND (taxonomy_id:2 OR taxonomy_id:2157)"
curl --fail --silent --show-error --get "https://rest.uniprot.org/uniprotkb/stream" \
    --data-urlencode "query=${QUERY}" --data "format=txt" --data "compressed=true" \
    --dump-header "$OUT/download_headers.txt" --output "$ENTRIES"
# Keep a record of the UniProt release the entries come from.
grep -i "^x-uniprot-release" "$OUT/download_headers.txt" > "$OUT/uniprot_release.txt"
rm "$OUT/download_headers.txt"

uv run python scripts/swissprot_to_fasta_and_go.py \
    "$ENTRIES" "$OUT/holdout.fasta" "$OUT/holdout_go.tsv"

# --cov-mode 0: the alignment must cover 80% of both sequences.
"$MMSEQS" easy-cluster "$OUT/holdout.fasta" "$OUT/clusters" "$OUT/mmseqs_tmp" \
    --min-seq-id 0.5 -c 0.8 --cov-mode 0 --threads "$THREADS" -v 2
rm -r "$OUT/mmseqs_tmp" "$OUT/clusters_all_seqs.fasta"

echo "test set: $(grep -c '>' "$OUT/clusters_rep_seq.fasta") representatives" \
     "of $(grep -c '>' "$OUT/holdout.fasta") proteins"
