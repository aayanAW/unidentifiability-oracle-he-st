# HANDOFF — Unidentifiability Oracle for H&E→ST (WACV 2026)

Single entry point for resuming in a fresh session. Read this first, then `decision-ledger.jsonl`.

## How to resume (do this)

1. Open Claude Code in this directory: `/Users/aayanalwani/Computer Vision`.
2. Paste:
   > Read HANDOFF.md, decision-ledger.jsonl, preregistration.md, and plan.md. We are mid-Phase-C build on Seed A2. Continue from the "Next actions" in HANDOFF.md.
3. That's it — the ledger (rollouts 1–7) + the 5 design docs + git history are the full state. No prior chat needed.

## Where the project stands (as of 2026-06-12)

- **Idea (Seed A2):** a cleaner-reference (Xenium) _unidentifiability oracle_ for H&E→spatial-transcriptomics — separates biological unpredictability from technical dropout, powering a coverage-guaranteed selective-risk (abstention) layer. Novelty gate **PASSED** at 7.5/10.
- **Phase chain (all in `decision-ledger.jsonl`):** A-ideate → B novelty gate (SHARPEN) → A re-entry (3 seeds) → picked A2 → B-delta (PROCEED-conditional) → pre-registration → C plan+audit → #0 data precondition verified.
- **Architecture chosen:** staged **C1 → HYBRID** (`architecture.md`). C1 = feasibility vehicle + fallback paper; HYBRID = dual-head oracle + post-hoc conformal.
- **Build so far:** Phase 0/1 harness committed. Synthetic gate GREEN (method separates planted unidentifiable genes from dropout). **No real data, no torch yet.**
- **Noise-floor source LOCKED:** GSE243280 GSM7780153 (Rep1) + GSM7780154 (Rep2), breast, true Xenium technical replicate.

## Key files

| File                                                            | What                                                                |
| --------------------------------------------------------------- | ------------------------------------------------------------------- |
| `decision-ledger.jsonl`                                         | append-only Phase A→C decision trail (the source of truth)          |
| `preregistration.md`                                            | frozen hypotheses H1–H4, kill criteria K1–K4, plan-audit amendments |
| `novelty.md` / `architecture.md` / `feasibility.md` / `plan.md` | the gate + design docs                                              |
| `src/`                                                          | simulator, oracle, metrics, loaders (real-data swap point)          |
| `experiments/feasibility_breast.py`                             | the decisive test (synthetic now, `--real` later)                   |
| `tests/test_gate.py`                                            | gate suite — run `python3 tests/test_gate.py` (must stay green)     |

## Next actions (Phase 2)

1. `bash scripts/fetch_data.sh` — download GSE243280 Rep1/Rep2 (multi-GB) + paired Visium + H&E.
2. `python3 scripts/check_panel.py data/rep1 data/rep2` — confirm identical Xenium panel (the one open Phase-0 task).
3. Accept the `MahmoodLab/UNI` (and `paige-ai/Virchow2`) licenses on Hugging Face (account aayan1234), install torch.
4. Wire `src/loaders.py` (`load_breast_triad`) against the real files: UNI/Virchow2 cached embeddings + Visium↔Xenium registration at cell/niche resolution + panel intersection.
5. `python3 experiments/feasibility_breast.py --real data/` — **the real CONFIRM/KILL gate.** CONFIRM → build HYBRID; KILL → report negative or flip to the virtual-staining sibling.

## Gotchas

- **Model:** runs on Opus 4.8 (Fable 5 unavailable; every doc notes this).
- **Env:** system `python3` (3.9) has numpy/scipy/sklearn — enough for synthetic. Real data needs torch + HF auth.
- **git:** local repo only (no remote). Commits exist; nothing pushed. `feat/` branch off `main` for new experiments per the ML git-first rule.
- **Don't overclaim:** the synthetic CONFIRM validates _logic_, not biology. Real data decides.
- **Scoop clock:** UTOPIA (bioRxiv) is one revision from a selective-risk curve — Phase 1–4 speed matters.
- **Workflow prompts** (`~/Desktop/Claude Workflow/A,B,C`) were updated to mandate named tools + `ultracode`; Exa MCP is now configured.
