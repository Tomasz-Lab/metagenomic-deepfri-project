#!/bin/bash
# Run mdF on the SwissProt calibration set, allowing only structural
# templates below 50% sequence identity to the query.
#
# `mDeepFRI benchmark --identity-bin 0 0.5` keeps, for every query and
# database, the best MMseqs2 hit with identity in [0, 0.5). Templates are
# taken from PDB100 first, then AFDB v4, then ESM Atlas highquality_clust30.
# Queries without a template are predicted by the sequence-only CNN.
# All other parameters are the mdF defaults.
#
# Submit from the repository root:
#   sbatch scripts/swissprot_calibration_mdf.sbatch                                 # one GPU
#   sbatch --qos=cpu_s --gres=scratch:300 scripts/swissprot_calibration_mdf.sbatch   # CPU only
#
#SBATCH -J swissprot_calibration_mdf
#SBATCH -p gpu
#SBATCH --qos=gpu2
#SBATCH --gres=gpu:1,scratch:300
#SBATCH -c 16
#SBATCH --mem=200G
#SBATCH -t 12:00:00
#SBATCH -o data/source/swissprot_calibration/mdf_%j.log
set -euo pipefail

ROOT=$SLURM_SUBMIT_DIR
QUERY=$ROOT/data/source/swissprot_calibration/clusters_rep_seq.fasta
OUT=$ROOT/data/source/swissprot_calibration/mdf
# mdF 1.3.0 environments and DeepFRI v1.0 weights of the runtime benchmark
BENCH=$ROOT/data/source/runtime_benchmark
WEIGHTS=$BENCH/weights
AFDB=/cephfs/team/structural_and_functional_genomics/sciprog-kg1/shared/afdb_foldcomp/afdb_uniprot_v4
ESM=$BENCH/data/foldcomp/highquality_clust30/highquality_clust30

if [ -n "${SLURM_JOB_GPUS:-}" ]; then
    ENV=$BENCH/envs/mdf-gpu
    # onnxruntime finds the pip-installed CUDA libraries only on the library
    # path. Without them mdF silently falls back to the CPU, so check first.
    export LD_LIBRARY_PATH=$(ls -d "$ENV"/lib/python3.11/site-packages/nvidia/*/lib | paste -sd:)${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
    "$ENV/bin/python" "$BENCH/scripts/tests/ort_cuda_test.py" "$WEIGHTS"/DeepFRI-MERGED_*_mf.onnx \
        | grep "^session providers: \['CUDAExecutionProvider'" \
        || { echo "onnxruntime cannot use the GPU" >&2; exit 1; }
else
    ENV=$BENCH/envs/mdf-cpu
fi

# MMseqs2 writes its temporary files to the node-local scratch disk.
export TMPDIR=$SCRATCH_JOB/tmp
mkdir -p "$TMPDIR"

echo "node: $(hostname -s), threads: $SLURM_CPUS_PER_TASK, environment: $ENV"

# The environments were moved after they were created, which broke the
# `mDeepFRI` launcher, so the command-line module is called directly.
"$ENV/bin/python" -m mDeepFRI.cli benchmark \
    --input "$QUERY" \
    --output "$OUT" \
    --weights "$WEIGHTS" \
    --db-path "$AFDB" \
    --db-path "$ESM" \
    --identity-bin 0 0.5 \
    --generate-contacts 2 \
    --processing-modes bp --processing-modes cc --processing-modes mf \
    --threads "$SLURM_CPUS_PER_TASK" \
    --tmpdir "$TMPDIR"
