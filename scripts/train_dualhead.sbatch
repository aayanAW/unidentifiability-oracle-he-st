#!/bin/bash
# SLURM launcher for the dual-head oracle end-to-end fine-tune (multi-GPU, torchrun/DDP).
#
# This is the CLUSTER step only (plan.md "Direction update"): end-to-end backbone fine-tune + scaling to
# 5 organs + Xenium-5K. The frozen-embedding result runs cluster-free on a Mac
# (experiments/trained_oracle_breast.py). NOT Modal -- user runs on a SLURM cluster.
#
# Submit:  sbatch scripts/train_dualhead.sbatch
# Override:  DATA=... CONFIG=... OUT=... ENSEMBLE=5 EPOCHS=300 sbatch scripts/train_dualhead.sbatch
#
# One task per node so srun launches a single torchrun per node, which then spawns one worker per GPU
# (the standard multi-node torchrun-under-SLURM idiom -- avoids the srun x torchrun double-spawn).

#SBATCH --job-name=uorc-dualhead
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --time=08:00:00
#SBATCH --output=runs/%x-%j.out

set -euo pipefail

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
GPUS_PER_NODE="${GPUS_PER_NODE:-4}"
MASTER_ADDR="$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n1)"
export MASTER_ADDR
export MASTER_PORT="${MASTER_PORT:-29500}"

DATA="${DATA:-data/cache/dualhead_breast.npz}"
CONFIG="${CONFIG:-configs/train_breast.yaml}"
OUT="${OUT:-runs/dualhead_breast}"
ENSEMBLE="${ENSEMBLE:-5}"
EPOCHS="${EPOCHS:-300}"

mkdir -p runs "$OUT"

# record reproducibility stamp (git SHA + config hash) alongside the run
git rev-parse HEAD > "$OUT/git_sha.txt" 2>/dev/null || true
sha256sum "$CONFIG" > "$OUT/config_hash.txt" 2>/dev/null || true

srun torchrun \
  --nnodes="$SLURM_NNODES" \
  --nproc-per-node="$GPUS_PER_NODE" \
  --rdzv-backend=c10d \
  --rdzv-endpoint="$MASTER_ADDR:$MASTER_PORT" \
  -m src.train \
  --data "$DATA" \
  --config "$CONFIG" \
  --out "$OUT" \
  --ensemble "$ENSEMBLE" \
  --epochs "$EPOCHS"
