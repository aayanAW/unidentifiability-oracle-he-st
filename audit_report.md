# Cross-Model Project Audit — Unidentifiability Oracle (H&E→ST)

**Audited SHA:** `4975b2bc1a3bd32c16c28b4b018a2dd8b31bbe60` (branch `main`, **tree CLEAN**)
**Config-hash** (`configs/feasibility_breast.yaml`): `795543fe7bbf7ec6...a4bd44e6`
**Date:** 2026-06-12 · **Scope:** whole repo, 1618 LOC · **Mode:** READ-ONLY (zero fixes applied)

## Model families

| Family            | Engine                                          | Mechanism                                                                        | Findings                                    |
| ----------------- | ----------------------------------------------- | -------------------------------------------------------------------------------- | ------------------------------------------- |
| **A — Claude**    | Opus 4.8 (claim/gate) + Sonnet 4.6 (mechanical) | Workflow `wja6ryry5`, 7 agents, 466k subagent tokens, 6 dimension finders        | **60** (5 CRIT / 20 HIGH / 23 MED / 12 LOW) |
| **B — Codex/GPT** | gpt-5.5 (xhigh reasoning), authed via ChatGPT   | `codex exec` full audit (`bifchxen6`) + `codex review` native diff (`borj3s3vv`) | **29** (verdict: _fundamental rework_)      |

**Family-B availability — REAL** (codex gpt-5.5 live). **Deviation (disclosed):** the `/codex:*` slash-commands are not agent-invocable in this session (plugin installed, commands unregistered), so Family B was run by calling the **`codex` CLI directly** via Bash — same engine the slash-commands wrap, same concurrency, blind to Family A. No degraded intra-family fallback was needed.

Both families received only the shared repo as input; neither saw the other's findings before synthesis.

---

## Verdict: **FIX-THEN-SHIP (harness-level)** — not "ship", not "rework the idea"

The two families **converge** that the project does not currently hold up _as a method demonstration_, but they frame severity differently — this is the most informative disagreement (resolved below):

- **Codex:** blunt — _"fundamental rework."_
- **Claude:** graded — 5 CRITICAL clustered in claim-validity + gate-honesty, the rest fixable.

**Resolution (with evidence):** the _idea's_ integrity guards are pre-registered and sound (`preregistration.md` already names σ²_reg, the conditional-upper-bound downgrade, the independent-architecture f′, region-level U). The failure is that **the current CODE under-implements its own prereg, and the synthetic gate is partly a rubber stamp** — so the gap is _immature harness_, not _fatal idea_. Therefore: **do not cite the synthetic CONFIRM as method validation; harden the gate and implement the prereg's own mandated controls before any real-data run.** That is fix-then-ship, sharper than codex's "rework."

**The cross-model design earned its cost — concretely:** each family's code-reviewer caught a CRITICAL real-data bug the other was blind to (see DISAGREEMENT-BY-COVERAGE).

---

## Agreement matrix

### CONFIRMED — both families, independent + blind (highest confidence)

| #   | Sev          | Where                                 | Finding                                                                                                                                                                                                                                                                                                                                                |
| --- | ------------ | ------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| C1  | **CRITICAL** | `oracle.py:78-80`                     | **U mislabels unlearned structure as biology.** `aleatoric = resid2 − epi` removes ensemble _variance_ but not _bias²_; with a **linear Ridge** substrate, any nonlinear morphology→expression gene yields large positive U. Claude numerically: a deterministic _nonlinear_ gene (zero true aleatoric) → **reported U = 0.989**. The K1 attack lands. |
| C2  | **CRITICAL** | `oracle.py:73-74`                     | **σ²_reg ≡ 0 on every code path** (defaults to zeros; never passed by `feasibility_breast.py` or `test_gate.py`). The registration-noise subtraction `preregistration.md` calls _mandatory and non-optional_ (§6) is silently absent → registration jitter counts as biology.                                                                          |
| C3  | **CRITICAL** | `feasibility_breast.py:67`            | **The anti-circularity f′ arm is fake** — same Ridge ensemble, different seed; prereg §11-C requires a _different architecture_. Does not test f-specificity.                                                                                                                                                                                          |
| C4  | **CRITICAL** | `simulator.py:88-95` + `oracle.py:71` | **Synthetic gate is partly guaranteed by construction.** Class-B "dropout" is injected only into `visium`, but the oracle fits the **Xenium** target — so "deferrals don't track dropout" is true by construction, not earned. The gate largely validates the simulator, not the oracle.                                                               |
| C5  | **CRITICAL** | `xenium_io.py:306-315`                | **`matched_bins` cross-section key mismatch** — each section bins from its _own_ origin, then matches rounded absolute keys → registered replicates land on different integer grids → the noise floor itself is corrupted.                                                                                                                             |
| H1  | HIGH         | `oracle.py:97-107`                    | `separation_auroc` scores deferral = U _against U itself_ → circular metric, not an independent selective-layer eval.                                                                                                                                                                                                                                  |
| H2  | HIGH         | `xenium_io.py:335`                    | `_normalize_pair` averages per-replicate stds instead of true pooled std → "σ²_xen = 1 − concordance" not generally valid.                                                                                                                                                                                                                             |
| H3  | HIGH         | `embeddings.py:76-85`                 | Bare `except` silently downgrades a confirmatory (UNI) run to exploratory fallback via a print.                                                                                                                                                                                                                                                        |
| H4  | HIGH         | `embeddings.py:207-216`               | `_load_homography` returns `None` on malformed input → H&E patches extracted in the wrong frame, no error.                                                                                                                                                                                                                                             |
| H5  | HIGH         | `oracle.py:35`                        | OOF uses random `KFold` over spots, not tissue/spatial blocks → spatial leakage, violates `preregistration.md` §9.                                                                                                                                                                                                                                     |
| H6  | HIGH         | `feasibility_breast.py`               | No git SHA / config-hash / artifact URI / dep versions stamped at run time — the reproducibility discipline is promised in docs, absent in code.                                                                                                                                                                                                       |
| H7  | HIGH         | `embeddings.py:219-223`               | Embedding cache key omits image content, model revision, `um_per_px` → stale/mismatched embeddings reused silently. (Codex B2: fallback also cached under the `morph_uni_*` key → confirmatory runs stay on fallback after access is fixed.)                                                                                                           |
| H8  | HIGH         | `oracle.py:1-9`                       | Docstring: U is "the irreducible part NOT explained by morphology" — overstates the prereg §11-B downgrade to a conditional upper bound.                                                                                                                                                                                                               |
| H9  | HIGH         | `simulator.py:11`                     | "proving it tracks biology" — overclaims from a _planted_ synthetic gate.                                                                                                                                                                                                                                                                              |
| H10 | HIGH         | `fetch_data.sh`                       | Downloads have no checksum/signature verification → corrupted/substituted GEO artifacts enter analysis undetected.                                                                                                                                                                                                                                     |
| H11 | HIGH         | `xenium_io.py:70-77`                  | Zip handling: symlink members unfiltered + duplicate-basename overwrite + unbounded `src.read()` (Zip-Slip mitigated only by basename flattening).                                                                                                                                                                                                     |
| M1  | MED          | `oracle.py:82-91`                     | Bootstrap p-value not null-centered → BH flags do not validly test U ≤ 0.                                                                                                                                                                                                                                                                              |
| M2  | MED          | `test_gate.py:50-53`                  | Gate fraction band `[0.15,0.45]` reverse-engineered to the planted 1/3 → the gate cannot fail on that axis.                                                                                                                                                                                                                                            |
| M3  | MED          | `noise_floor_breast.py:7`             | Serial-section replicate variance described as pure technical noise — also contains serial-section biology + registration residual.                                                                                                                                                                                                                    |

### SINGLE-MODEL — only one family flagged (investigate; both verified real where checked)

**Codex-only (Claude's reviewers were blind):**

| #   | Sev                | Where              | Finding                                                                                                                                | Status                                                             |
| --- | ------------------ | ------------------ | -------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------ |
| SB1 | **CRITICAL**       | `metrics.py:27`    | `grid_weights(radius=1.5)` is unit-grid-only; on 100 µm Xenium niches → zero neighbors → degenerate Moran's I + spurious p.            | **VERIFIED real** by re-check                                      |
| SB2 | **HIGH (blocker)** | `fetch_data.sh:21` | `declare -A` fails on macOS **bash 3.2** (the default here) → with `set -e` the entry-point script aborts before any download.         | **VERIFIED** (`/bin/bash` = 3.2.57; `declare: -A: invalid option`) |
| SB3 | HIGH               | `fetch_data.sh:63` | `MODE=full` never fetches `visium/`, but `load_breast_triad` requires it → the documented `full` path fails before the real gate.      | plausible/real                                                     |
| SB4 | HIGH               | `loaders.py:126`   | Visium `tissue_positions` are **pixel** coords, Xenium centers are **microns**; `rigid_register` has no scale term → spots mis-binned. | real correctness gap                                               |
| SB5 | MED                | `xenium_io.py:75`  | `src.read()` loads a whole zip member into RAM → OOM risk on the tight disk it targets.                                                | real                                                               |

**Claude-only (Codex was blind):**

| #   | Sev      | Where                             | Finding                                                                                                                                                         | Status                                      |
| --- | -------- | --------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------- |
| SA1 | **HIGH** | `oracle.py:71-72`                 | **2× noise-floor/target mismatch**: target `z_est = 0.5(r1+r2)` has noise σ²/2, but `noise_floor` estimates the full single-replicate σ² → U over-subtracts 2×. | **numerically verified** (0.0585 vs 0.1192) |
| SA2 | HIGH     | `xenium_io.py:225-233`            | `rigid_register`: 2 of 4 flip candidates are reflections (det=−1); a reflection can win → mirror-image registration.                                            | real (ICP may partly correct)               |
| SA3 | MED      | `xenium_io.py:232`                | `best=None` → `TypeError` crash if all four flip candidates return −inf density overlap.                                                                        | real edge case                              |
| SA4 | HIGH     | `configs/feasibility_breast.yaml` | Config is **decorative** — no code loads it; experiment constants are hardcoded duplicates.                                                                     | real repro gap                              |
| SA5 | MED      | `README.md:3`                     | "coverage-guaranteed selective-risk" asserted, but **no conformal/coverage machinery exists** in the repo yet.                                                  | real overclaim                              |
| SA6 | MED      | `oracle.py:114-116`               | Per-spot `clip(...,0).mean` inflates apparent Moran's spatial structure (asymmetric clip on a smooth epistemic field).                                          | real                                        |

(+ a long tail of Claude-only silent-failure items: `_first_existing(None)` → cryptic errors in `_read_mex`/`_load_visium_binned`/`_read_visium_positions`, `_align_cells` silent cell drops, `extract_members` empty-on-no-match, `df -g` Linux-incompat, unbounded recursive panel-JSON parse, etc.)

### DISAGREEMENT-BY-COVERAGE (the cross-model payoff)

No hard _contradictions_ (neither family declared "fine" what the other called "broken"). The informative split is **differential blindness**:

- **Codex caught, Claude's code-reviewer missed:** `grid_weights` nan (SB1, CRITICAL) and the bash-3.2 entry-point break (SB2, blocker).
- **Claude caught, Codex missed:** the 2× noise-floor mismatch (SA1, numerically proven) and the reflection-flip registration bug (SA2).

→ Either family alone would have shipped a CRITICAL real-data bug. The two-family design is justified by exactly these four findings.

---

## Six-dimension summary

1. **Research / claim validity — FAIL (as-coded).** U is dominated by linear-substrate under-fitting bias (C1) and an unsubtracted registration term (C2); the separation can't be trusted until f is expressive and σ²_reg is real.
2. **Gate / eval honesty — FAIL.** The synthetic gate is partly a construction artifact (C4), the f′ arm is not independent (C3), the band is reverse-engineered (M2), and the AUROC is circular (H1). The synthetic CONFIRM must not be cited as validation.
3. **Code correctness — multiple real bugs**, worst in the _unexercised real-data path_: `matched_bins` key mismatch (C5), `grid_weights` nan (SB1), Visium scale (SB4), reflection flip (SA2), `_normalize_pair` (H2).
4. **Reproducibility — weak.** No runtime SHA/hash/URI stamp (H6), decorative config (SA4), no tissue-block split (H5), collision-prone cache (H7).
5. **Claim-honesty — several overstatements** vs the prereg's own (more honest) downgrades: docstring "irreducible" (H8), "proving biology" (H9), README "coverage-guaranteed" (SA5).
6. **Security / IO — medium.** No download integrity (H10), zip symlink/overwrite/unbounded read (H11), curl redirect/proto unbounded, recursive-JSON / parquet parse without guards.

## Top fix order (when the separate fix pass is approved)

1. **C4 + C3 + M2** — rebuild the synthetic gate so it can _fail_: inject the failure modes (nonlinear class-C, dropout into the fitted target if claiming dropout-robustness), make f′ a different architecture, derive the band from a null not the planted truth.
2. **C1** — replace/augment the linear Ridge substrate (or restrict the claim to the linear-recoverable regime and _test_ the nonlinear failure).
3. **C2 + H5** — implement σ²_reg (±1-bin perturbation) and a tissue-block split before any real run.
4. **SB2 + SB1 + C5 + SB4** — fix the broken/incorrect real-data path (bash-3.2 fetch, `grid_weights` radius, bin-key frame, Visium scale) — these block or corrupt the very first real run.
5. **H6 + SA4** — wire the config + per-run SHA/hash/URI stamp.
6. Docstring/README honesty (H8/H9/SA5); IO hardening (H10/H11).

> READ-ONLY audit. No code was modified. Fixing is a separate, approved pass.
> Heaviest independent Claude-cloud pass available on request: run `/code-review ultra` yourself (billed, cloud — the agent cannot launch it).
