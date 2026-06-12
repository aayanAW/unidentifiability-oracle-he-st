# HANDOFF — Unidentifiability Oracle for H&E→ST (WACV 2026)

Single entry point for resuming in a fresh session. Read this first, then `decision-ledger.jsonl`.

## ⟶ RESUME HERE (rollout 14, 2026-06-12) — build the trainable HYBRID

Branch `feat/audit-fixes-rollout10` (PR #1 open to `main`, not merged). Public repo `aayanAW/unidentifiability-oracle-he-st`. State since the original handoff below:

- **Cross-model audit done** (rollout 9, `audit_report.md`): Claude + Codex gpt-5.5. 5 confirmed criticals.
- **Audit fixes done** (rollout 10): nonlinear RF substrate + failable gate w/ negative control, σ²_reg explicit, 2× noise-floor, spatial-block CV, KNN f′, bash-3.2 fetch, adaptive `grid_weights`, zip hardening. Gate green, ruff clean.
- **Real data downloaded** (public GEO): breast Xenium Rep1/Rep2 matrices + H&E (1.4 GB ea) + homography. `data/` (gitignored). Disk ~12 GB free — **other 4 organs NOT downloaded (won't fit; not needed yet)**.
- **Real noise floor** (rollout 11): K4 clears at **≥300 µm niches** (18.5% of 313 genes), dominates <250 µm. Resolution-constrained.
- **Exploratory real U** (rollout 14, frozen DINOv2-S, EXPLORATORY not confirmatory): after the **SB4 registration fix** (`Hinv @ Xenium-px`), morph→ST predicts **58% of genes** (R² max 0.38, median +0.073), biologically coherent, U spatially structured. **Vacuity risk de-risked. GREEN.** Run: `python3 experiments/exploratory_u_breast.py data --bin-um 300 --encoder vit_small_patch14_dinov2.lvd142m`.
- **Direction (rollout 12):** move to a **GPU-trained HYBRID** `f` on an **HPC cluster (SLURM/DDP — NOT Modal)**; fine-tune an **ungated** backbone (removes the UNI blocker). Not a novelty pivot. Proposal (`proposal.pdf`) reflects it.

**Next actions (Tier-2 HYBRID build — cluster-independent to write + smoke-test; full run on cluster):**

1. Port the SB4 registration fix (`experiments/exploratory_u_breast.py:_patches_lazy`) into `src/embeddings.py` + `src/loaders.py` (real pipeline).
2. `src/predictor.py` — trainable `f`: ungated backbone (`vit_small_patch14_dinov2.lvd142m` or CTransPath) + H&E→ST head + dual variance head (β-NLL).
3. `src/train.py` — AMP, checkpointing, `torchrun`/DDP-ready, deep-ensemble; SLURM `sbatch`. Smoke-test on CPU/MPS (tiny), full run on cluster.
4. Wire trained `f` as the `run_oracle` substrate; recompute U vs the frozen baseline (does training tighten it?).
5. Compute the **selective-risk-coverage curve** (the real utility metric) — the headline result.

- Deps installed this session: `tifffile`, `timm`, `zarr<3`. Awaiting cluster access (user runs on cluster, not Modal).

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
