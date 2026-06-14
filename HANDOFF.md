# HANDOFF — Unidentifiability Oracle for H&E→ST (WACV 2026)

Single entry point for resuming in a fresh session. Read this first, then `decision-ledger.jsonl`.

## ⟶ RESUME HERE (rollout 16, 2026-06-13) — all cluster-free work DONE; only GPU/data-gated work left

**Branch `feat/hybrid-build`** (10 commits, pushed to public `aayanAW/unidentifiability-oracle-he-st`; PR not yet opened). Off `main` @ `2f50a08`. Every cluster-free component is built + green; what remains needs the GPU cluster or more data.

**Rollout 16 (cluster-free completion) added:**

- **Conformal layer** (`src/conformal.py`, the missing Phase-3 PDC): split (marginal) + spatial-Mondrian (group-conditional) + σ-normalized scores. **First real H1 result** (`experiments/conformal_breast.py`): marginal coverage valid + well-calibrated (0.902/0.952/0.804), but naive coverage is spatially heterogeneous (worst block **0.839 < 0.90**, miscoverage **Moran's I=0.263, p=0.005**); spatial-Mondrian tightens it (spread 0.114→0.087, worst 0.839→0.854). H1 SUPPORTED.
- **End-to-end de-risked**: ImageNet-normalization added inside the backbone path; full `--backbone` CLI smoke validated (DINOv2-S builds, normalizes, fits, checkpoints with `encoder_name`/`is_confirmatory`). `experiments/build_patches.py` → `data/cache/patches_breast.npz` (399×3×224×224) so the cluster run is one submit.
- **Figures**: `experiments/make_figures.py` (selective-risk-coverage + H1 coverage).
- **README honesty** updated (conformal now implemented; marginal-not-conditional).

**What remains (GPU- or data-gated only):**

1. **End-to-end backbone fine-tune** (GPU): `BACKBONE=vit_small_patch14_dinov2.lvd142m DATA=data/cache/patches_breast.npz sbatch scripts/train_dualhead.sbatch`. The open scientific bet — does a jointly-trained `f` lift DDH-U above the RF baseline + clear K2?
2. **Deep ensemble across architectures** (epistemic term + independent-f′ arm; `fit_ensemble` built + tested). Planned cluster mechanism; a frozen 3-member run won't flip the DDH conclusion (floor-clipping, not seed variance).
3. **5 organs + Xenium-5K** (GPU + data: liver/HCC/ovary have NO Xenium technical replicate; disk tight).
4. **TISSUE baseline** comparison (can be added cluster-free later if wanted).
5. **Open the PR** for `feat/hybrid-build`.

---

### Prior: rollout 15 (Tier-2 HYBRID build) — superseded header kept for context

The Tier-2 cluster-free HYBRID build is DONE; the first trained-oracle result is in.

**What landed this session (rollout 15):**

1. **SB4 registration ported** into the real pipeline (`src/embeddings.py` + `src/loaders.py`): `inv(H) @ (centers_um/um_per_px)` + a memory-safe lazy zarr window reader. Validated **399/399 niches in-bounds** on real breast (the rollout-14 invariant).
2. **`src/predictor.py`** — `DualHeadOracle`: ungated DINOv2-S backbone (or frozen embeddings) → shared trunk → mean head + log-variance head; **β-NLL** loss (Seitzer 2022).
3. **`src/train.py`** — `fit_dual_head` (AMP cuda-only, DDP when `WORLD_SIZE>1`, deep ensemble, checkpointing) + CLI (frozen-embedding and `--backbone` end-to-end modes) + **`scripts/train_dualhead.sbatch`** (torchrun-under-SLURM).
4. **`src/oracle.py`** — `risk_coverage_curve` + `aurc` (the selective-risk-coverage utility metric).
5. **`experiments/trained_oracle_breast.py`** — the FIRST trained-oracle result.

**First trained-oracle result (breast @300 µm, frozen DINOv2-S, EXPLORATORY — `python3 experiments/trained_oracle_breast.py data --bin-um 300`):**

- The **frozen-RF raw-variance oracle is a STRONG real selective system**: efficiency **0.784**, Spearman(U,error)=**0.916**. The C1 oracle machinery works on real tissue.
- The **trained dual-head on FROZEN embeddings does NOT beat it**: head-to-head RF-U eff **0.524** vs DDH-U eff **−0.074**; DDH raw-variance diagnostic **0.386**. `U_ddh` collapses under floor-subtraction; mean head R² max 0.396, 130/313 genes.
- **Verdict MIXED, reported straight (not a kill).** Frozen embeddings + ~399 niches favor the RF ensemble variance; the dual-head's advantage is the **pre-registered end-to-end cluster bet** (jointly-trained backbone + 5 organs). No hyperparameter chasing.

**Review:** 4-dim adversarial Workflow → 14 confirmed findings (2-skeptic verified) → **all fixed** (channels-first all-zero patches, SB4 in-bounds invariant, real `--backbone` wiring, artifact labeling, GradScaler, efficiency normalizer, atomic gunzip, device-safe OOF, `--rdzv-id`, RF-defined C1 mask, …). 6 test suites green, ruff clean.

**Next actions (the cluster end-to-end bet — needs cluster access):**

1. Build the **image-patch dataset** `data/cache/patches_breast.npz` (X=(N,3,224,224) uint8 via `embeddings._extract_patches`, Y=z_est) for true end-to-end mode.
2. Run `BACKBONE=vit_small_patch14_dinov2.lvd142m DATA=data/cache/patches_breast.npz sbatch scripts/train_dualhead.sbatch` on the cluster (end-to-end fine-tune).
3. Scale to **5 organs + Xenium-5K**; re-evaluate: does a jointly-trained `f` lift DDH-U above the RF baseline and clear the 15% utility floor (K2)?

**Then:** open the PR for `feat/hybrid-build` once the cluster result lands (or sooner if merging the build is wanted). Append rollout-N to `decision-ledger.jsonl` per decision; commit + push per logical unit.

**Cluster boundary:** steps 1–3 are the cluster step (SLURM/DDP — NOT Modal). Everything in rollout 15 ran cluster-free on this Mac. Deps: `tifffile`, `timm`, `zarr<3`, `torch 2.0.0` (MPS).

---

## How to resume (do this)

1. Open Claude Code in this directory: `/Users/aayanalwani/Computer Vision`.
2. Paste:
   > Read HANDOFF.md, decision-ledger.jsonl, preregistration.md, and plan.md. We are mid-Phase-C build on Seed A2. Continue from the "Next actions" in HANDOFF.md.
3. That's it — the ledger (rollouts 1–7) + the 5 design docs + git history are the full state. No prior chat needed.

## Where the project stands (as of 2026-06-12)

- **Idea (Seed A2):** a cleaner-reference (Xenium) _unidentifiability oracle_ for H&E→spatial-transcriptomics — separates biological unpredictability from technical dropout, powering a coverage-guaranteed selective-risk (abstention) layer. Novelty gate **PASSED** at 7.5/10.
- **Phase chain (all in `decision-ledger.jsonl`):** A-ideate → B novelty gate (SHARPEN) → A re-entry (3 seeds) → picked A2 → B-delta (PROCEED-conditional) → pre-registration → C plan+audit → #0 data precondition verified.
- **Architecture chosen:** staged **C1 → HYBRID** (`architecture.md`). C1 = feasibility vehicle + fallback paper; HYBRID = dual-head oracle + post-hoc conformal.
- **Build so far:** Phase 0/1 harness + **real GSE243280 loader wired** (`8b2716f`, branch `feat/real-loader-phase1`). Synthetic gate GREEN; new binning/registration tests GREEN; ruff clean. torch 2.8 + scanpy present. **Still NO real data downloaded — zero real results yet.**
- **Noise-floor source LOCKED:** GSE243280 GSM7780153 (Rep1) + GSM7780154 (Rep2), breast, true Xenium technical replicate.
- **Two user-side blockers (rollout 8 recon):** (1) the expression matrix is inside each **9.2 GB `outs.zip`** → the noise floor needs a ~18 GB download (disk-safe via stream-extract-delete, peak ~9 GB; disk is tight at 22 GB free). (2) **UNI/Virchow2 are gated and auto-deny `@gmail` primary emails** — HF acct `aayan1234` + gmail will be rejected; needs a `.edu` (Weill Cornell / Stony Brook) primary email + `hf auth login`. **The prereg §6 noise-floor control needs NEITHER** (Xenium matrices only) — it is the next decisive step and is UNI-free.

## Key files

| File                                                            | What                                                                |
| --------------------------------------------------------------- | ------------------------------------------------------------------- |
| `decision-ledger.jsonl`                                         | append-only Phase A→C decision trail (the source of truth)          |
| `preregistration.md`                                            | frozen hypotheses H1–H4, kill criteria K1–K4, plan-audit amendments |
| `novelty.md` / `architecture.md` / `feasibility.md` / `plan.md` | the gate + design docs                                              |
| `src/`                                                          | simulator, oracle, metrics, loaders (real-data swap point)          |
| `experiments/feasibility_breast.py`                             | the decisive test (synthetic now, `--real` later)                   |
| `tests/test_gate.py`                                            | gate suite — run `python3 tests/test_gate.py` (must stay green)     |

## Next actions (Phase 1 — real gate)

**Step 0 (loader): DONE** — `src/loaders.py` wired against the confirmed GEO schema (`8b2716f`).

UNI-free path — do this FIRST (the §6 noise-floor control; answers the #1 circularity objection, K4):

1. `bash scripts/fetch_data.sh data noise` — pulls both 9.2 GB `outs.zip`, extracts only `cell_feature_matrix.h5` + `cells.parquet` + `gene_panel.json` per replicate, deletes each zip (peak ~9 GB). ~18 GB total transfer on your bandwidth.
2. `python3 scripts/check_panel.py data/rep1 data/rep2` — confirm identical 313-gene panel.
3. `python3 experiments/noise_floor_breast.py data` — **first REAL result.** Reports per-gene σ²_xen (=1−replicate concordance) and how many panel genes clear the floor. FLOOR-CLEARED → U worth computing; FLOOR-DOMINATES → K4 risk, coarsen bin or report negative.

U path (needs UNI, currently blocked by the gmail-gating issue above):

4. Fix HF: set a `.edu` primary email on account `aayan1234`, request `MahmoodLab/UNI` + `paige-ai/Virchow2`, `hf auth login`.
5. `bash scripts/fetch_data.sh data full` — adds the 1.3 GB post-Xenium H&E + homography per replicate.
6. `python3 experiments/feasibility_breast.py --real data` — **the real CONFIRM/KILL gate.** (Falls back to an ungated encoder if UNI still pending → run is EXPLORATORY, not confirmatory.) CONFIRM → build HYBRID; KILL → report negative or flip to virtual-staining.

## Gotchas

- **Model:** runs on Opus 4.8 (Fable 5 unavailable; every doc notes this).
- **Env:** system `python3` (3.9) has numpy/scipy/sklearn — enough for synthetic. Real data needs torch + HF auth.
- **git:** local repo only (no remote). Commits exist; nothing pushed. `feat/` branch off `main` for new experiments per the ML git-first rule.
- **Don't overclaim:** the synthetic CONFIRM validates _logic_, not biology. Real data decides.
- **Scoop clock:** UTOPIA (bioRxiv) is one revision from a selective-risk curve — Phase 1–4 speed matters.
- **Workflow prompts** (`~/Desktop/Claude Workflow/A,B,C`) were updated to mandate named tools + `ultracode`; Exa MCP is now configured.
