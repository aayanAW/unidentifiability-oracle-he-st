# Feasibility — Cheapest Decisive Experiment (Seed A2)

> Phase C, Step 2. The single smallest experiment that proves or kills the **core** hypothesis before any infrastructure. Build is gated on this returning first. No infra before it. C1-shaped on purpose (the cheapest test rides the cheapest architecture).

## Core hypothesis under test

**H2 (the load-bearing claim):** a non-zero, spatially-structured intrinsic unidentifiability `U` survives subtraction of the Xenium-replicate noise floor, and abstention concentrates on high-`U` (biological) rather than high-dropout (technical) cells. If `U` collapses to the noise floor or is spatially white, the oracle is dead regardless of C1/C2/C3/HYBRID (kill criterion **K4**, then **K1**).

## Exact protocol

- **Dataset/split:** ONE organ — **breast, Janesick 2023, GEO `GSE243280`** (most replicates; gold-standard registration). Same/serial block, Visium + Xenium + H&E. Restrict to the **Xenium panel-intersection genes** (~280) and to regions below a registration-residual threshold. Calibration/test split by spatial block (no spot leakage).
- **Model:** frozen `MahmoodLab/UNI` embeddings (pre-cached) → 2-layer MLP `f` predicting Xenium-clean `z` from morphology. No fancy regressor needed for this test.
- **Procedure:**
  1. Estimate per-gene **Xenium-replicate noise floor** `σ̂²_xen` (variance across matched-spot replicate Xenium sections) and registration variance `σ̂²_reg` (registration-perturbation model).
  2. Estimate `Var[z|x]` via held-out cross-fitted residual of `f`.
  3. `U = Var[z|x] − σ̂²_xen − σ̂²_reg` per gene/region; bootstrap 95% CI.
  4. Compute Moran's I of `U` (spatial structure) and AUROC of {deferral ↔ high-`U`} vs {deferral ↔ high-dropout}.

## Confirm / kill numbers (pre-registered)

| Verdict                                       | Condition                                                                                                                                                                                                               |
| --------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **CONFIRM → build HYBRID**                    | ≥20% of panel genes have `U` with bootstrap 95% CI excluding 0 after floor subtraction, **AND** `U` spatially structured (Moran's I significant, p<0.05), **AND** deferral↔U AUROC − deferral↔dropout AUROC **≥ 0.10**. |
| **KILL → report negative, stop / flip to VS** | <5% genes with `U`>0 after subtraction, **OR** `U` spatially white (Moran's I n.s.), **OR** AUROC gap **< 0.05**.                                                                                                       |
| **AMBIGUOUS → limited HYBRID probe**          | 5–20% genes, or AUROC gap 0.05–0.10: build the DDH oracle on breast only (its exogenous-target heads may sharpen separation vs. C1's self-residual), re-evaluate the same numbers before scaling to 5 organs.           |

## Compute cost (inside 192 GB)

Trivial. UNI embeddings for one breast block (~few-thousand to ~50k spots) pre-cached once; one MLP ~50 epochs in **<5 min on a single GPU**; replicate-variance + Moran's I + AUROC are CPU-minutes. **Total < 1 GPU-hour incl. sweeps, ≪ 192 GB** (runs on a 24–48 GB card). No data infrastructure, no distributed anything, before this returns.

## Why this is decisive

It isolates the one claim everything else depends on (non-circular, structured `U`) on the cleanest-registration organ, using the cheapest model, and reuses the exact code as the C1 fallback paper. A CONFIRM unlocks the full build; a KILL costs <1 GPU-hour and is reported honestly (not reframed). This is the failure mode the researcher flagged — surfaced first, not after infrastructure.

## Plan-audit amendments to the gate (2026-06-11, mandatory)

- **σ²_xen source — RESOLVED (2026-06-12).** A true Xenium technical replicate exists: **LOCKED to GSE243280 GSM7780153 (Rep1) + GSM7780154 (Rep2)** — breast Sample #1, same block, two serial 5 µm sections, same 313-gene panel. K4′ (estimable floor) expected to pass on breast. Confirm both replicates' `panel.json` identical on download. Floor is serial-section+run-to-run → conservatively inflates `σ²_xen` (shrinks `U`, safe). Within-niche substitute demoted to fallback; published-model fallback = Spatial Touchstone (Plummer 2025) / Wang 2025.
- **Independent-`f′` arm (anti-circularity).** Recompute `U` from a different-seed/architecture `f′`. CONFIRM requires deferral↔U AUROC gap ≥ 0.10 under **both** `f` and `f′`; collapse under `f′` → KILL (U is f-specific noise).
- **Estimand fix.** Replace `Var[z|x]` with an ensemble / MC-dropout aleatoric estimate (epistemic removed) before subtracting the floor.
- **Resolution.** Compute `U` at Xenium cell/niche resolution, aggregate to region; run the ±1-spot registration-perturbation sensitivity.
- **Multiplicity.** Moran's-I structured-`U` count is FDR-corrected; the ≥20% / <5% thresholds apply to the corrected set.
