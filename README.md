# Unidentifiability Oracle for H&E→Spatial Transcriptomics

A **selective-risk (abstention) layer** for virtual spatial transcriptomics that separates _biological_
unpredictability (morphology genuinely cannot determine a gene) from _technical_ dropout (measurement
noise), using a cleaner Xenium reference as the truth check.

> Scope honesty (rollout 16): the post-hoc **conformal coverage layer** is now implemented
> (`src/conformal.py`: split + spatial-Mondrian) and gives a **marginal** distribution-free coverage
> guarantee — empirically valid + well-calibrated on real breast (0.902 at α=0.10). It does **not** give a
> per-spot _conditional_ guarantee (impossible, Foygel-Barber 2021); naive coverage is spatially
> heterogeneous (H1: worst block 0.839, miscoverage Moran's I p=0.005) and Mondrian only partially closes
> it. `U` remains an **upper bound** on intrinsic unidentifiability conditional on the predictor's
> recoverable signal (`preregistration.md` §11-B). All current results are **exploratory** (frozen ungated
> DINOv2-S, breast only); the confirmatory end-to-end / UNI run is the GPU-cluster step.

See the evidence chain and design docs:

- `novelty.md` — Phase B / B-delta novelty gate (oracle claim 7.5/10, verified open)
- `preregistration.md` — frozen hypotheses, estimand, kill criteria, plan-audit amendments
- `architecture.md` — staged C1→HYBRID design (judge-panel chosen)
- `feasibility.md` — the cheapest decisive experiment (this repo's first gate)
- `plan.md` — phased plan with go/no-go gates
- `decision-ledger.jsonl` — append-only Phase A→C decision trail

## Status

Cluster-free build **COMPLETE** (rollouts 1–20): real GSE243280 breast loader, the §6 Xenium-replicate
noise floor, the trained dual-head oracle + independent calibration, the post-hoc conformal layer (H1),
and the selective×conformal product are all built, audited (ultracode rollout 18 → fixed 19 → reviewed
20), and green (8 test suites). **No real data is committed** (gitignored — fetch it, see below). All
current results are **exploratory** (frozen ungated DINOv2-S, breast only); the confirmatory end-to-end +
UNI + multi-organ run is the GPU step. The real-data swap point is `src/loaders.py`.

## Quickstart — real breast results (Xenium + H&E)

These steps reproduce the corrected rollout-18/19 numbers on real data. Steps 3–6 need **only** Xenium +
H&E (fetched in step 1) — **no Visium, no UNI required**.

```bash
python experiments/feasibility_breast.py --synthetic              # 0) plumbing + logic check (no download)
bash scripts/fetch_data.sh data full                              # 1) Rep1/Rep2 Xenium + post-Xenium H&E (~4 GB)
python scripts/check_panel.py data/rep1 data/rep2                 # 2) confirm identical 313-gene panel
python experiments/noise_floor_breast.py data --bin-um 300        # 3) §6 control (K4) -- the first real result
python experiments/trained_oracle_breast.py data --bin-um 300     # 4) trained oracle + indep. calibration 0.601
python experiments/conformal_breast.py data --bin-um 300          # 5) H1 spatial coverage (Moran p=0.005)
python experiments/selective_conformal_breast.py data --bin-um 300 # 6) the selective×conformal product
```

On a GPU box pass `--device cuda` to steps 4/6. Encoders fall back to ungated DINOv2-S (run = EXPLORATORY)
unless UNI/Virchow2 access is configured (`hf auth login` with an **institutional-email** HF account),
in which case pass `--encoder uni` for a confirmatory morphology substrate.

### Full-triad separation gate (H2 — needs Visium, not in the fetch script)

`python experiments/feasibility_breast.py --real data/` additionally tests **H2** (deferrals concentrate
on high-`U`, not high-dropout), which requires the paired breast **Visium** (CytAssist FFPE) as the
dropout target under `data/visium/` (`*filtered_feature_bc_matrix.h5` + spatial positions). `fetch_data.sh`
does **not** download Visium yet — source it separately before running this gate, or it raises a
`FileNotFoundError` naming the missing dir. Pre-registered verdict (`feasibility.md`): **CONFIRM** if ≥20%
panel genes show structured `U` (FDR-corrected) AND deferral↔U − deferral↔dropout AUROC ≥ 0.10 under
**both** `f` and an independent `f'`; **KILL** if <5%, `U` spatially white, or AUROC gap <0.05.

## Gate scripts (the test suite)

```bash
python tests/test_gate.py     # ranking-based separation + a NEGATIVE CONTROL (a broken linear oracle
                              # must fail) -- so the gate can actually fail, not rubber-stamp the method
```

The gate uses a nonlinear (RandomForest) substrate `f`, a different-architecture `f'` (KNN), a
spatial-block CV split, and an explicit Xenium-replicate noise floor; the negative control shows a linear
substrate inflates `U` on identifiable genes ~2× (audit rollout 9→10 hardening).

## Running on the GPU cluster (the end-to-end fine-tune)

All code is in this repo; **data is not** (gitignored, as research data should be). To run the end-to-end
dual-head fine-tune on a SLURM cluster:

```bash
# 1. env -- install torch matching your cluster CUDA FIRST, then the rest
pip install torch --index-url https://download.pytorch.org/whl/cu121   # pick your CUDA
pip install -r requirements.txt

# 2. training data -- NOT in git (gitignored derived data). If you ALREADY built it locally
#    (experiments/build_patches.py, which needs the full GSE243280 download), transfer the 37 MB tensor:
rsync -avP data/cache/patches_breast.npz <cluster>:<repo>/data/cache/
#    OR regenerate it on the cluster from the raw GSE243280 breast data:
python3 experiments/build_patches.py data --bin-um 300      # needs data/rep1 + data/rep2

# 3. submit the multi-GPU end-to-end fine-tune (torchrun/DDP)
BACKBONE=vit_small_patch14_dinov2.lvd142m \
DATA=data/cache/patches_breast.npz ENSEMBLE=5 EPOCHS=300 \
sbatch scripts/train_dualhead.sbatch
```

Omit `BACKBONE=` for the cheaper frozen-embedding scale-up (`data/cache/dualhead_breast.npz`). Cluster
bugs from the rollout-18 audit (DDP `device_ids`, shared-backbone ensemble, seed-pinning) are fixed; the
checkpoints are stamped with `encoder_name` + `is_confirmatory` so a frozen run is never mistaken for UNI.
