# Architecture — Seed A2: Unidentifiability Oracle for H&E→Spatial Transcriptomics

> Phase C, Step 1. No code. Implementation-ready spec. Routing: judge-panel via 3 parallel scoring agents (no `ultracode`/Workflow invoked). Winner chosen below with reasons. Grounds the pre-registration (`preregistration.md`) in concrete components.

## Design space (3 mechanisms + hybrid)

| Cand                                            | Mechanism for the oracle/abstention                                                                                            | Rigor judge (cov / non-circ / wrong-target) | Acceptance (novelty / defensibility / accept) | Feasibility (time / failure-surface / 192GB) |
| ----------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------ | ------------------------------------------- | --------------------------------------------- | -------------------------------------------- |
| **C1 PDC** — Post-hoc Decoupled Conformal       | frozen FM → light regressor `f`; `U` via post-hoc variance decomposition; post-hoc spatial-Mondrian split-conformal abstention | 9 / 5 / 4                                   | 4 / 8 / 6                                     | 9 / 8 / 10                                   |
| **C2 DDH** — Disjoint Dual-Head Heteroscedastic | disjoint-gradient clean-signal + noise-variance heads (Decoupled-PFN backbone); `U` read from heads                            | 6 / 6 / 6                                   | 7 / 6 / 5                                     | 4 / 4 / 10                                   |
| **C3 NSR** — Null-Space Residual Probe          | `U` = residual orthogonal to any high-capacity FM head (statistical Bhadra-Anastasio)                                          | 5 / 7 / 7                                   | 8 / 5 / 4                                     | 3 / 3 / 9                                    |
| **HYBRID** — DDH-oracle + PDC-guarantee         | C2 disjoint heads **estimate** `U`; C1 post-hoc split-conformal **guarantees** coverage                                        | 9 / 7 / 5                                   | 7 / 8 / 7                                     | 5 / 6 / 10                                   |

All three judges' verdicts reconcile to one **staged** design, not a single pick.

## Winner: **Staged C1 → (gate) → HYBRID**

- **C1 is the feasibility vehicle and the fallback paper.** The cheapest decisive experiment (`feasibility.md`) is C1-shaped, so the go/no-go and a publishable fallback come from the same code in week 1 (feasibility judge).
- **HYBRID is the target architecture.** It is the only design that keeps the marginal guarantee on a frozen, never-calibration-trained predictor (clean exchangeability) **while** sourcing `U` from an _exogenous_ Xenium-measured noise target rather than the predictor's own residual (rigor + acceptance judges). Its asymmetry — novel where it's safe (the estimator), guaranteed where it's attacked (coverage holds _regardless_ of whether `U` is "correct") — is the highest-expected-acceptance configuration.
- **C3's framing is grafted as related-work/interpretation only** (statistical null space), not built — its "orthogonal to _any_ head" estimand is capacity-confounded (all three judges flagged it).

**Mandatory baseline (acceptance judge):** a **trivial raw-variance `U`** (held-out residual variance / raw Visium variance). The dual-head oracle must beat it on selective-risk _efficiency_ — that gain is the headline result and the answer to "are the heads decoration?"

## Winning spec (HYBRID, implementation-ready)

### Components + exact artifacts

- **Encoder (frozen):** `MahmoodLab/UNI` (ViT-L/16, 1024-d) primary; `paige-ai/Virchow2` (ViT-H/14, 2560-d) as a second backbone for model-agnosticism. Embeddings **pre-computed and cached to disk** → encoder leaves VRAM.
- **Data/loaders:** `MahmoodLab/hest` toolkit (patching, UNI/Virchow featurizers, ST/Visium/Xenium loaders). Triads from `feasibility.md` § data.
- **Substrate regressor `f`:** for feasibility, frozen-UNI + 2-layer MLP. For main experiments, wrap ≥1 strong public H&E→ST regressor (TRIPLEX, CVPR 2024; or STFlow, ICLR 2025) to show the oracle is **model-agnostic** (post-hoc layer sits on any `f`).
- **Conformal library:** MAPIE (`/scikit-learn-contrib/mapie`) for split-conformal + conformal risk control (CRC); `crepes` for Mondrian/normalized regressors; `torchcp` if GPU conformal is needed. TISSUE repo (`sunericd/TISSUE`) as the ST-specific scaffold/baseline.

### Oracle estimator (DDH)

Two heads on cached embeddings `h(x(s))`, **non-overlapping gradient paths**:

- **Clean-signal head** `μ_θ(h) → ẑ`, target = Xenium clean expression `z` (panel-intersection genes).
- **Noise-variance head** `σ²_φ(h)`, target = exogenous noise = Xenium-replicate variance + Visium–Xenium dropout gap.
- **Intrinsic unidentifiability** `U(g,r) = Var_{s∈r}[ẑ(s,g)] − σ̂²_xen(g,r) − σ̂²_reg(g,r)` (estimand frozen in `preregistration.md` §2). Disjoint objectives + β-NLL/gradient-surgery for stability; the noise target is _measured_, not self-derived → less circular than C1's self-residual.

### Coverage guarantee (PDC, post-hoc)

- Split-conformal **risk** control on a calibration fold **disjoint from both heads' training and `f`**. Nonconformity score = `|z − ẑ|` normalized by an identifiability estimate from `U`; abstain above the CRC threshold for target selective risk `α*`.
- **Spatial non-exchangeability:** weighted/block conformal (NexCP-style weights from a spatial kernel; block-by-region calibration) — restores marginal coverage on autocorrelated spots.
- **Groups:** Mondrian = (gene-predictability tier × tissue region). **Marginal-per-group only** — never "conditional guarantee" (Barber 2021).

### Data flow

`WSI → tile → UNI/Virchow2 (cached) → [f → ŷ] + [DDH: ẑ, σ̂²] → U(g,r) → PDC spatial-Mondrian CRC → {predict | abstain} + selective-risk-coverage curve`. Xenium replicate → `σ̂²_xen` (gating control, run first).

## Exact metrics (named)

- **Primary:** selective-risk-coverage curve (selective risk vs. retained fraction); marginal coverage at nominal `1−α` per Mondrian group (target within ±2%); **deferral↔U AUROC** (separation).
- **Efficiency headline:** retained fraction at fixed `α*` for **dual-head U vs. raw-variance-U** baseline (the "heads aren't decoration" result).
- **Secondary:** Moran's-I-stratified coverage (spatial-CP vs. naive); retained-fraction × utility; per-organ registration residual; cross-panel scaling gap (UNI vs. Virchow2; v1-panel → Xenium-5K).

## Why not the alternatives (one line each)

- **C1 alone:** reviewers read it as TISSUE + a variance split (novelty 4) — kept as feasibility probe + fallback, not the headline.
- **C2 alone:** disjoint heteroscedastic heads are training-unstable and the bio-vs-technical split is an _unguaranteed_ identifiability claim — sinks if a head underperforms.
- **C3:** null space shrinks as the FM grows → measures capacity, not biology (capacity-confounded). Framing only.
