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

Phase 0 (repo) + Phase 1 (feasibility) harness. **No real data committed.** The feasibility experiment
runs on a synthetic simulator with _planted_ ground truth by default (`--synthetic`), so the gate script
validates the method machinery before any download. The real-data swap point is `src/loaders.py`.

## The decisive test (Phase 1)

```bash
python experiments/feasibility_breast.py --synthetic        # plumbing + logic check (no download)
bash scripts/fetch_data.sh                                  # download GSE243280 Rep1/Rep2 (large)
python scripts/check_panel.py data/rep1 data/rep2           # confirm identical Xenium panel
python experiments/feasibility_breast.py --real data/       # the real gate (after wiring loaders.py)
```

Pre-registered verdict (`feasibility.md`): **CONFIRM** if ≥20% panel genes show structured `U`
(FDR-corrected) AND deferral↔U AUROC − deferral↔dropout AUROC ≥ 0.10 under **both** `f` and an
independent `f'`. **KILL** if <5%, or `U` spatially white, or AUROC gap <0.05.

## Gate scripts (the test suite)

```bash
python tests/test_gate.py     # ranking-based separation + a NEGATIVE CONTROL (a broken linear oracle
                              # must fail) -- so the gate can actually fail, not rubber-stamp the method
```

The gate uses a nonlinear (RandomForest) substrate `f`, a different-architecture `f'` (KNN), a
spatial-block CV split, and an explicit Xenium-replicate noise floor; the negative control shows a linear
substrate inflates `U` on identifiable genes ~2× (audit rollout 9→10 hardening).
