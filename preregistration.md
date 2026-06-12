# Pre-registration — Seed A2: Unidentifiability Oracle for H&E→Spatial Transcriptomics

> Written 2026-06-11, before any modeling code. Satisfies the three Phase-B-delta gating conditions (`novelty.md` § Phase B-delta → Verdict). Confirmatory analyses are frozen here; anything added later is labeled exploratory in the deviations log. Reproducibility stamp per run: git SHA + config-hash + artifact URI (repo to be `git init`-ed as the first Phase-C action — see § 9).

## 1. One-line claim

A cleaner-reference (Xenium) **unidentifiability oracle** can separate _biological_ unpredictability of gene expression from morphology from _technical_ dropout/measurement noise, and this separation yields a coverage-guaranteed selective-risk (abstention) layer on H&E→spatial-transcriptomics whose deferrals concentrate on intrinsically unidentifiable genes/regions — not on noisy ones.

## 2. Estimand (frozen definition)

For spot/region `s`, gene `g`, H&E patch `x(s)`:

- `y_obs(s,g)` — Visium observed expression (dropout-corrupted target the predictor is trained on).
- `z(s,g)` — Xenium near-clean expression on the **panel-intersection** gene set `G∩` (the cleaner reference).
- `f` — any H&E→ST regressor (TRIPLEX / BLEEP / STFlow class), frozen as the substrate.

Define the **intrinsic unidentifiability** of `(g, region r)` as the conditional variance of the _clean_ signal not explained by morphology, in excess of the reference noise floor:

```
U(g,r) = Var_s∈r[ z(s,g) | x(s) ]  −  σ²_xen(g,r)  −  σ²_reg(g,r)
```

where `σ²_xen` = Xenium technical noise (from replicate sections, § 6) and `σ²_reg` = variance injected by adjacent-section registration error (§ 7.2). `U > 0` after both subtractions = genuine morphology→molecule non-identifiability; `U ≈ 0` with large Visium–Xenium gap = dropout, not unidentifiability.

**Why this is not circular (principled backbone):** the aleatoric/epistemic (noise vs. non-identifiability) split is provably **not** recoverable from the predictor's own predictive distribution (Hüllermeier & Waegeman 2019; Decoupled PFNs 2026; Credal CBM 2026). It requires _external_ labels for the noiseless target and the noise variance. Xenium supplies the first; the Xenium replicate supplies the second. The oracle label `U` is therefore defined from sources independent of `f`'s residuals.

## 3. Hypotheses (confirmatory, directional)

- **H1 — spatial under-coverage.** Naive split-conformal calibrated on Visium spots under-covers the Xenium-clean target on spatially autocorrelated regions (Moran's I above the per-slide median); a weighted/spatial-block conformal scheme restores marginal coverage to nominal `1−α` within ±2%. _Falsifies the spatial-non-exchangeability contribution if null._
- **H2 — non-circular separation.** After subtracting `σ²_xen + σ²_reg`, residual `U` is non-zero and gene/region-structured, and selective-layer **deferrals concentrate on high-`U` cells, not high-dropout cells**: AUROC(deferral ↔ high-`U`) − AUROC(deferral ↔ high-dropout) ≥ 0.10. _This is the core claim; if null, the oracle is measuring reference artifacts (the Critic's kill)._
- **H3 — non-vacuity.** At a pre-set target selective risk `α*`, retained-fraction × utility ≥ the floor (§ 5) on ≥3 of the 5 organs.
- **H4 — scaling.** An oracle calibrated on v1-panel organs (~hundreds of genes) holds on the Xenium-5K organ (~5k genes): coverage within ±2% of nominal and deferral↔`U` AUROC within δ=0.05. _Converts the panel-gene ceiling into a scaling result._

## 4. Data + scope (frozen)

| Organ     | Visium/Visium HD + Xenium + H&E, same/serial block | Source                                  |
| --------- | -------------------------------------------------- | --------------------------------------- |
| Breast    | ✓ (gold; explicit keypoint registration)           | Janesick 2023, GEO **GSE243280**        |
| Colon     | ✓ (post-Xenium CytAssist, same section)            | 10x Post-Xenium CytAssist               |
| Lung      | ✓ (post-Xenium, Xenium-5K)                         | 10x Visium HD Post-Xenium               |
| Liver/HCC | ✓ (4-platform serial)                              | Nat Commun 16:9232 (s41467-025-64292-3) |
| Ovary     | ✓ (4-platform serial)                              | Nat Commun 16:9232                      |

- **Auditable universe = the Xenium panel-gene intersection per organ** (~280–377 genes for v1-panel organs; ~5k for Xenium-5K). The claim is **panel-conditional**, explicitly NOT genome-wide.
- **HEST-1k** is used only as an H&E/featurization supplement, **not** as a triad source (it aggregates independent, non-co-registrable samples — the "HEST trap").
- Encoders frozen: UNI / Virchow2 (pre-cached embeddings). Conformal layer: MAPIE / crepes / TISSUE scaffold.

## 5. Metrics (primary / secondary)

- **Primary:** selective-risk-coverage curve (selective risk vs. retained fraction); marginal coverage at nominal `1−α` within each Mondrian (gene × region) group; deferral↔`U` AUROC (the separation metric, H2).
- **Secondary:** Moran's-I-stratified coverage (H1); retained-fraction × utility (H3); per-organ registration residual; cross-panel scaling gap (H4).
- **Utility floor (H3):** ≥15% of panel genes retained at target selective risk `α*` with retained-set error below the marginal predictor's error — set now, not after seeing results.

## 6. Gating control — Xenium-replicate noise floor (run FIRST)

Before any `U` is reported: estimate `σ²_xen(g,r)` from replicate Xenium sections of the same block (variance of `z` across replicates), and `σ²_reg` from a registration-perturbation model (§ 7.2). **No unidentifiability claim is admissible until this noise floor is subtracted.** If the noise floor consumes essentially all of `Var[z|x]` (i.e., `U ≈ 0` everywhere after subtraction), H2 is dead and the project stops — reported as a negative result. This control is the gate, not an optional robustness check.

## 7. Confounds + mitigations (pre-committed)

1. **Detectability-selection bias (the 1% catch):** panel-intersection genes are selected for being detectable/morphology-correlated → the oracle is audited on the _easiest_ genes. We **characterize, do not claim**, the oracle's coverage of the hard regime, and report the selection explicitly. No generalization claim to non-panel genes.
2. **Registration error:** quantify per-organ registration residual; propagate as `σ²_reg`; restrict confirmatory analysis to regions below a residual threshold. Same-section pairs (colon) weighted over serial pairs (lung).
3. **Xenium is cleaner-not-clean:** treat Xenium as lower-dropout, not zero-dropout; off-target probe binding handled via the replicate noise floor, never assumed away.
4. **Conditional-coverage impossibility (Barber 2021):** guarantees are **Mondrian group-conditional marginal** only. The words "conditional guarantee" do not appear in the paper.

## 8. Kill / stopping criteria (falsifiable, frozen)

The project is reported as a **negative result** (not reframed) if any fires:

- **K1 (circularity):** H2 null — deferrals do not concentrate on high-`U` over high-dropout after noise-floor correction (AUROC gap < 0.10).
- **K2 (vacuity):** H3 null — <15% useful retained at `α*` on ≥3 organs.
- **K3 (empty spatial contribution):** H1 null — naive split-conformal does not under-cover on autocorrelated spots (spatial CP buys nothing).
- **K4 (noise floor dominates):** § 6 control leaves `U ≈ 0` after subtraction.

## 9. Analysis plan + reproducibility

- Calibration/test split frozen by tissue block (no spot leakage across split). Confirmatory tests = H1–H4 above; everything else is exploratory and labeled so.
- **Git-first (CLAUDE.md ML):** `git init` private repo as the first Phase-C action; commit this pre-registration as the initial commit before any modeling code. Record git SHA + config-hash + artifact URI per experiment run.
- No peeking: § 6 noise-floor control and the H2 separation test are run before any metric tuning.

## 10. Deviations log

| Date       | Change                                                                                | Confirmatory→Exploratory? | Reason        |
| ---------- | ------------------------------------------------------------------------------------- | ------------------------- | ------------- |
| 2026-06-11 | Estimand §2: `Var[z\|x]` → ensemble/MC-dropout aleatoric estimate (epistemic removed) | confirmatory              | plan audit C2 |
| 2026-06-11 | `σ²_xen` source contingency (replicate vs disclosed within-niche substitute)          | confirmatory              | plan audit C1 |

## 11. Plan-audit amendments (2026-06-11) — mandatory before Phase 0

Adversarial plan audit surfaced two CRITICAL construct/data issues + four refinements; folded in as pre-registered conditions.

- **A — Replicate noise-floor source (CRITICAL) → RESOLVED 2026-06-12.** Verification found a **true Xenium technical replicate**: `σ²_xen` is **LOCKED** to Janesick GSE243280 — **GSM7780153 (Rep1) + GSM7780154 (Rep2)**, breast Sample #1, same FFPE block, two serial 5 µm sections, same 313-gene panel (280 base + 33 add-on). GSM7780155 = Sample #2 = different block = biological, **excluded**. Replication check: 10x "Entire Sample Area" breast Rep1/Rep2. Published-model fallback: Spatial Touchstone per-gene SNR (Plummer 2025, Nat Biotechnol) + Wang 2025 per-gene Spearman. The within-niche pooled-variance substitute is **demoted to fallback only** (not needed). Caveats carried: (i) the floor is serial-section + run-to-run (no same-section re-scan exists publicly for any tissue) → it conservatively _inflates_ `σ²_xen`, shrinking `U` (safe direction); (ii) confirm both replicates' `panel.json` are identical on download before locking. K4′ (estimable `σ²_xen`) stays a Phase-1 check but is expected to pass on breast. Nat Commun 16:9232 (liver/HCC/ovary) has **no** Xenium replicates ("each experiment performed once") → those organs use the published-model floor or are evaluation-only.
- **B — Epistemic/aleatoric conflation (CRITICAL).** A non-Bayes-optimal `f` makes `Var[z|x]` (held-out residual) mix irreducible aleatoric with _unlearned_ biological structure. Fix: estimate recoverable signal with a deep ensemble / MC-dropout of `f`; epistemic = variance **across models at fixed input**; residual aleatoric (epistemic removed) − noise floor = `U`. Headline downgraded to "`U` **upper-bounds** intrinsic unidentifiability conditional on the ensemble's recoverable signal" — not an unconditional separation.
- **C — Independent-`f′` circularity control (CRITICAL → gate).** `U` and the deferral rule both derive from `f`. Add a Phase-1 arm: recompute `U` from an independently-trained `f′` (different seed + architecture); if deferral↔U AUROC collapses under `f′`, `U` is f-specific noise → KILL.
- **D — Registration resolution (HIGH).** Compute `U` at Xenium **cell/niche** resolution, aggregate to region; never at raw 55 µm Visium-spot resolution. Pre-register a ±1-spot alignment-perturbation sensitivity; unstable structured-`U` fraction = artifact.
- **E — Mondrian groups (HIGH).** Groups frozen from an **independent** partition (not calibration data); report worst-group coverage with finite-sample beta intervals.
- **F — Effect size + multiplicity (MED).** Pre-register the minimum selective-risk-efficiency gain of DDH-`U` over raw-variance-`U` that counts as a win. Moran's-I "structured `U`" counts are **FDR-corrected**; the ≥20% / <5% thresholds apply to the corrected set.
