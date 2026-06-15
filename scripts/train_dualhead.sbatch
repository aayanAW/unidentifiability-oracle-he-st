#!/bin/bash
# SLURM launcher for the dual-head oracle (multi-GPU, torchrun/DDP). NOT Modal -- user runs on a cluster.
#
# DEFAULT MODE = frozen-embedding ensemble scale-up: trains the dual head on the pre-cached DINOv2-S
# embeddings (data/cache/dualhead_breast.npz from experiments/trained_oracle_breast.py), but at scale
# (multi-GPU deep ensemble, more epochs) than the Mac run. This is EXPLORATORY (frozen ungated encoder).
#
# END-TO-END MODE (the pre-registered cluster step, plan.md "Direction update"): pass BACKBONE=<timm name>
# AND a DATA .npz of image patches (X shape (N,3,H,W)); src/train.py then builds the backbone and
# fine-tunes the whole stack. Without BACKBONE the `backbone` yaml key is intentionally ignored (frozen).
#
# Submit (frozen scale-up):  sbatch scripts/train_dualhead.sbatch
# End-to-end:  BACKBONE=vit_small_patch14_dinov2.lvd142m DATA=data/cache/patches_breast.npz sbatch scripts/train_dualhead.sbatch
# Override:  DATA=... CONFIG=... OUT=... ENSEMBLE=5 EPOCHS=300 [BACKBONE=...] sbatch scripts/train_dualhead.sbatch
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
BACKBONE="${BACKBONE:-}"  # set -> end-to-end fine-tune; unset -> frozen-embedding scale-up

mkdir -p runs "$OUT"

# record reproducibility stamp (git SHA + config hash) alongside the run
git rev-parse HEAD > "$OUT/git_sha.txt" 2>/dev/null || true
sha256sum "$CONFIG" > "$OUT/config_hash.txt" 2>/dev/null || true

BACKBONE_ARG=()
[ -n "$BACKBONE" ] && BACKBONE_ARG=(--backbone "$BACKBONE")

# --rdzv-id pins each SLURM job to a unique rendezvous so two jobs on the same node never collide on the port.
srun torchrun \
  --nnodes="$SLURM_NNODES" \
  --nproc-per-node="$GPUS_PER_NODE" \
  --rdzv-backend=c10d \
  --rdzv-id="$SLURM_JOB_ID" \
  --rdzv-endpoint="$MASTER_ADDR:$MASTER_PORT" \
  -m src.train \
  --data "$DATA" \
  --config "$CONFIG" \
  --out "$OUT" \
  --ensemble "$ENSEMBLE" \
  --epochs "$EPOCHS" \
  "${BACKBONE_ARG[@]}"
