# Unidentifiability Oracle for H&E→Spatial Transcriptomics

A coverage-guaranteed **selective-risk (abstention) layer** for virtual spatial transcriptomics that
separates _biological_ unpredictability (morphology genuinely cannot determine a gene) from _technical_
dropout (measurement noise), using a cleaner Xenium reference as the truth check.

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
python tests/test_gate.py     # asserts the method recovers planted unidentifiable genes, not dropout genes
```
