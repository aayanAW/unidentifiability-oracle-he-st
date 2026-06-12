# Plan — Seed A2: Unidentifiability Oracle for H&E→ST (WACV 2026)

> Phase C, Step 3. Phased execution with explicit go/no-go gates. 192 GB VRAM hard ceiling. No code until the user approves the build in a separate session.

## Compute reasoning (192 GB)

The architecture is **frozen encoder + pre-cached embeddings + light heads + post-hoc conformal** → VRAM is a non-issue (5–10× over-provisioned; runs on 24–48 GB). The binding resources are **(1)** one-time WSI tiling + UNI/Virchow2 embedding extraction (disk/throughput, ~1.5M HEST tiles), and **(2)** the statistical fragility of the oracle, not GPU memory. Fit strategy: frozen backbones, embedding cache, MLP/light heads, CPU-side conformal. No LoRA/distillation/progressive-resolution needed.

## Phased plan with gates

| Phase                                | Work                                                                                                                                                                         | Go/no-go gate                                                                                                                                 |
| ------------------------------------ | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------- |
| **0 · Repo**                         | `git init` private repo; commit `preregistration.md` + these docs as initial commit; `.gitignore` (data/, \*.ckpt, wandb/).                                                  | Repo + prereg committed before any modeling code (CLAUDE.md ML git-first).                                                                    |
| **1 · Feasibility**                  | Run `feasibility.md` exactly (breast, GSE243280, frozen-UNI MLP, noise-floor + U + AUROC).                                                                                   | **CONFIRM** (≥20% genes structured U, Moran's I sig, AUROC gap ≥0.10) → proceed. **KILL** → negative result, stop or flip to VS.              |
| **2 · Data + baselines**             | Register all 5 triads (breast/colon/lung/liver/ovary); cache UNI + Virchow2 embeddings; wrap TRIPLEX/STFlow as `f`; implement **raw-variance-U baseline** + TISSUE baseline. | Coverage of naive split-conformal measured; H1 (under-coverage on autocorrelated spots) testable; ≥4 organs registrable confirmed end-to-end. |
| **3 · Core method**                  | Build DDH disjoint dual-head oracle + PDC post-hoc spatial-Mondrian CRC; tune β-NLL/gradient-surgery for head stability.                                                     | DDH U beats raw-variance-U on selective-risk efficiency on breast; coverage holds within ±2% on calibration. Else fall back to C1 paper.      |
| **4 · Main experiments**             | Selective-risk-coverage curves across 5 organs (H3); spatial-CP vs naive (H1); deferral↔U separation (H2) at scale.                                                          | H2+H3 hold on ≥3 organs; retained-fraction×utility ≥ floor.                                                                                   |
| **5 · Ablations / robustness**       | H4 scaling (v1-panel → Xenium-5K; UNI vs Virchow2); registration-sensitivity; noise-floor ablation; raw-U vs DDH-U efficiency.                                               | Scaling gap within δ; results stable to registration threshold.                                                                               |
| **6 · Analysis / writing / release** | Paper, figures (selective-risk-coverage as design-system plots), code + benchmark release.                                                                                   | Pre-registered claims only; deviations logged; negatives reported.                                                                            |

## Reproducibility

Per CLAUDE.md ML git-first: experiment on `feat/<slug>` branches off `main`; record **git SHA + config-hash + artifact URI** per run; re-run gate scripts green before each commit; pre-registered confirmatory tests (H1–H4) separated from exploratory.

## Timeline (~2 months)

Wk 1: Phase 0–1 (feasibility verdict). Wk 2–3: Phase 2. Wk 4–6: Phase 3. Wk 7–8: Phase 4. Wk 9: Phase 5. Wk 10+: Phase 6. The scoop clock (UTOPIA one step from a selective-risk curve) makes Phase 1–4 speed the dominant variable.

## Top execution risks

1. **Circular oracle (highest).** `U` re-labels Xenium noise/registration error as biology. → Mitigated by the Phase-1 noise-floor gate + the exogenous-target DDH heads; K1/K4 kill it cheaply.
2. **Vacuity.** Pearson≈0.2 → abstain on ~everything. → retained-fraction×utility is a Phase-4 gate; K2 kills at <15% useful retained.
3. **DDH training instability.** Variance head collapses/explodes. → C1 (raw-variance-U) is the always-available fallback paper; HYBRID is an upgrade, not a dependency.
4. **Scoop.** UTOPIA/Zou ship abstention first. → Phase 1–4 prioritized; oracle (harder to scoop) is the headline, not the conformal layer.
5. **Panel-gene selection bias.** Oracle audited on the easiest genes. → disclosed, characterized not claimed; scaling test (H4) is the breadth argument.

## Oral odds (carried from gate)

Accept **40–45%**, oral **5–8%** if clean + pre-scoop. The Phase-1 noise-floor gate and the raw-U-vs-DDH-U efficiency headline are the two results that move a reviewer from accept toward champion.

## Plan-audit amendments (2026-06-11)

Plan status from audit: **Risky until patched** — sound architecture, but two CRITICAL construct/data fixes gate the build. Now folded in:

- **Phase 0 (precondition — DONE 2026-06-12):** Xenium technical-replicate source verified and **`σ²_xen` LOCKED to Janesick GSE243280 GSM7780153+GSM7780154** (breast, same block, serial 5 µm, same 313-gene panel). Remaining Phase-0 task: diff the two `panel.json` on download to confirm identical panel. Liver/HCC/ovary (Nat Commun 16:9232) have no Xenium replicates → published-model floor (Spatial Touchstone/Wang 2025) or evaluation-only.
- **Phase 1 gate (strengthened):** add (a) independent-`f′` circularity arm, (b) estimable-`σ²_xen` check (K4′), (c) FDR-corrected Moran's I, (d) `U` at cell/niche resolution + ±1-spot perturbation sensitivity.
- **Phase 3 (method):** use a **deep ensemble / MC-dropout of `f`** for epistemic/aleatoric separation — `U` is the aleatoric residual conditional on the ensemble's recoverable signal, not raw `Var[ẑ]`.
- **~~Top risk #0 (CRITICAL)~~ → RESOLVED 2026-06-12:** the replicate noise-floor source DOES exist (GSE243280 Rep1/Rep2, true technical replicate). Risk closed; only the `panel.json` identity check remains.
- **Construct-validity risk (CRITICAL):** `Var[ẑ]` conflates spatial heterogeneity with unpredictability for non-optimal `f` → ensemble-based aleatoric isolation + claim downgraded to a conditional upper bound.

## Direction update (2026-06-12) — move to a GPU-trained HYBRID (rollout 12)

The frozen-encoder + RF/Ridge substrate was the **C1 feasibility vehicle** (deliberately light, for a cheap decisive gate). The audit (rollout 9) confirmed its load-bearing weakness — **C1: a low-capacity `f` under-fits and inflates `U`** (identifiable genes keep `U≈0.4`, not ≈0). The principled fix is **more model capacity, which means compute**, so we now execute the planned **HYBRID** stage as the main method, not a frozen baseline:

- **Trained `f`:** fine-tune an **ungated** pathology backbone (CTransPath / DINOv2) + a high-capacity H&E→ST head (TRIPLEX/BLEEP/STFlow-class), end-to-end on WSI tiles. A stronger `f` leaves only irreducible residual → tightens `U` toward true unidentifiability (directly addresses audit C1). **Side benefit: removes the UNI/Virchow2 gated-license blocker.**
- **HYBRID dual-head oracle:** learned prediction + variance head (β-NLL + gradient surgery), calibrated by post-hoc spatial-Mondrian conformal.
- **Deep ensemble across architectures** for the epistemic term (replaces the RF bootstrap); also serves the independent-`f′` arm (audit C3).
- **Compute:** GPU-intensive — cloud **H100 via Modal** (tens of GPU-hours over 5 organs + Xenium-5K embedding extraction). Mixed precision + checkpointing + embedding cache fit each backbone on one H100; ensembling/organs scale out elastically. Within the 192 GB budget.
- **Staging unchanged / fallback intact:** the frozen post-hoc oracle (C1) remains an always-available, reproducible fallback paper, so HYBRID is an upgrade, not a dependency (mitigates training-instability risk, plan risk #3).
- **Scope note:** this is **not a novelty pivot** — same oracle claim (still 7.5/10), same pre-registration; only the substrate moves from frozen to trained. The contribution remains the oracle; the predictor class is wrapped, not claimed.
