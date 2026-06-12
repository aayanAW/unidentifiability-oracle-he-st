# Novelty Verification — Seed 1: Faithfulness-gated abstention for inferred spatial omics (H&E→ST)

> Phase B kill gate. Run 2026-06-11. Model routing: planned Fable 5 + ultrathink; executed on Opus 4.8 (1M) with parallel Opus subagents for retrieval (Fable 5 not available in this session — noted per the routing instruction; biology-flagged reasoning kept on Opus, which is the intended reroute target anyway). Retrieval tools that actually ran: parallel-search web_search (rate-limited late), Hugging Face MCP (paper_search + hub_repo_search), Context7 (resolve-library-id), plus four parallel general-purpose research subagents over arXiv / bioRxiv / OpenAlex / Semantic Scholar / OpenReview / proceedings. `/ars-lit-review`, Consensus, and a second parallel-search pass were superseded once the four subagents converged on the same closest-prior set and the same scoop (UTOPIA); the marginal source was not worth the rate-limit budget. This is disclosed for honesty rather than claiming a clean run of every named tool.

## Selected seed

**Seed 1 (Phase A composite 8.50, rank 1):** A distribution-free, gene-and-region-conditional selective-prediction layer that sits on any H&E→spatial-transcriptomics regressor and **abstains** (or returns a risk-controlled interval) exactly where the morphology→molecule map is unidentifiable — converting today's point-estimate "virtual omics" into a clinically defensible measurement with a coverage guarantee. Mechanism: conformal **risk** control whose risk target absorbs the measurement-noise (dropout) model, under a spatially non-exchangeable calibration scheme, with the nonconformity score conditioned on gene and tissue region.

---

## Step 1 — Novelty Verification

### Closest prior art (5–10)

Each entry: what they did / how Seed 1 differs. All verified-found via live search unless flagged.

1. **TISSUE — Sun, Ma, Navarro Negredo, Brunet, Zou. _Nature Methods_ 21:444–454, 2024.** Conformal calibration of spatial gene-expression _imputation_: cell-centric variability nonconformity score, asymmetric conformal intervals, distribution-free interval coverage `P(X∈[l,u])≥1−α`, stratified (k-means gene/cell) calibration. **Differs:** input is an scRNA-seq-reference imputation model, **not H&E**; it **always predicts** (no abstention / reject option); it builds intervals, **not conformal risk control**; spatial autocorrelation handled only indirectly (profile k-means, not spatial weighting); ST treated as clean ground truth (no dropout-into-risk model). This is the single nearest work.
2. **UTOPIA — Kaitian Jin et al. bioRxiv, 1 Mar 2026 (repo github.com/kaitianjin/UTOPIA).** Model-agnostic, multiscale (gene→metagene, cell-type→class) **calibrated confidence scores + FDR control** for H&E virtual ST, in- and out-of-sample. **Differs:** delivers calibrated confidence + FDR, **not finite-sample coverage-guaranteed abstention**; framed as reliability/interpretability, **not** a risk-controlled clinical measurement with a selective-risk operating curve. ~80% conceptual overlap — the dominant scoop (see below). Confidence MED-HIGH (README + abstract read; not yet peer-reviewed).
3. **STimage — Tan, Mulay, Nguyen et al. (Univ. Queensland). _Nature Communications_, 16 Jan 2026.** Per-tile aleatoric (negative-binomial variance) + epistemic (deep-ensemble) uncertainty for H&E→ST; predicts an expression _distribution_. **Differs:** Bayesian/ensemble uncertainty with **no coverage guarantee, no calibration validation, no abstention/selective-risk protocol**. Nearest _uncertainty_ (non-conformal) prior; must be cited and distinguished.
4. **Conformal Risk Control — Angelopoulos, Bates, Fisch, Lei, Schuster. ICLR 2024** (precursors: RCPS, Bates et al. JACM 2021; Learn-then-Test 2021). The off-the-shelf machinery the seed extends. **Differs:** canonical CRC controls risk on the _observed_ label treated as clean; the seed's contribution is folding a **dropout/measurement-noise model into the risk target on the latent clean transcriptome** — not present in any CRC paper.
5. **Conformal prediction beyond exchangeability (NexCP) — Barber, Candès, Ramdas, Tibshirani. _Annals of Statistics_ 2023**; plus spatial CP (Mao–Martin–Reich, _JASA_ 2024; LSCP, Jiang–Xie 2024). Establish valid CP under non-exchangeability / spatial dependence. **Differs:** never ported to / validated on Visium spot autocorrelation; the seed must empirically show naive split-conformal under-covers on Moran's-I-positive spots and the spatial variant restores coverage.
6. **Label-noise conformal — Einbinder, Feldman, Bates, Angelopoulos, Gendler, Romano. _JMLR_ 2024**; Sesia, Wang, Tong, _JRSS-B_ 2024; Zaffran et al. CP-with-missing-values 2023. Establish CP robustness/correction under _class-level_ label noise / missingness. **Differs:** none models technical **dropout / zero-inflation on a continuous expression measurement**; dropout ≠ the dispersive/transition-matrix noise these model.
7. **Conformalized Selective Regression — Sokol, Moniz, Chawla. arXiv 2024** (+ Selective Conformal Risk Control, Xu–Guo–Wei 2025/26; Conformal Triage, Bates et al. 2024). Conformal-scored reject options with selective/PAC guarantees. **Differs:** all on generic tabular / non-omics / non-histology data; never applied to spatial omics or H&E.
8. **Mahmood-lab H&E→ST ecosystem — HEST-1k (Jaume et al., NeurIPS 2024 D&B); SEAL (2026); benchmark Wang, Chan, … Patrick, Yang, _Nat Commun_ 2025; SpaRED/SpaCKLE (Mejia, Ruiz et al., 2025).** The saturated prediction-and-benchmark substrate (~24 distinct H&E→ST predictors enumerated, best Pearson ≈ 0.2). **Differs:** all optimize mean accuracy; **none** reports calibrated selective prediction / coverage-guaranteed abstention. This is the open lane.

**Citation corrections the seed must make (honesty):** "Chan 2023 Nat Commun" is actually **Wang, Chan, … Patrick, Yang, _Nat Commun_ 16:1544, 2025** (Chan is 2nd author, year 2025). **SEAL is a method, not a benchmark** — the three real standardized benchmarks are HEST-1k (2024), Wang et al. (2025), SpaRED (2025); SEAL should be cited separately as a 2026 foundation-model method.

### Nearest scoop + what survives it

**Nearest scoop = UTOPIA (bioRxiv, 1 Mar 2026).** It already occupies "trustworthy virtual ST with calibrated confidence + FDR, gene/cell-type-conditional, in/out-of-sample." It is ~80% of the seed's stated framing and the authors can add an abstention rule in revision at low cost.

**What survives it (the genuinely uncovered intersection — verified absent across _all_ domains, not just this one):**

1. **A dropout/measurement-noise observation model folded into a conformal _risk_ target on the latent clean transcriptome** — UTOPIA (calibrated confidence + FDR), TISSUE (intervals on clean-treated labels), and every CRC / label-noise-CP paper lack this. Least contested, hardest to scoop, biologically motivated (SpaRED established ST ground truth is dropout-corrupted).
2. **A coverage guarantee validated under Visium spatial non-exchangeability** — i.e., demonstrating naive split-conformal / TISSUE-style i.i.d. calibration _under-covers_ on autocorrelated spots and a weighted/spatial-block CP restores it. No spatial-CP method has been ported to spot data.
3. **An explicit abstention / selective-_risk_ operating curve framed as clinical measurement** — neither UTOPIA (FDR + confidence) nor TISSUE (always-predict intervals) offers a risk-coverage curve.

The tri-combination (spatial-non-exchangeable + dropout-noise-aware + conformal risk control + abstention) is **unscooped in any domain** (P < 5–15%). The single-domain headline "we add abstention to H&E→ST," by contrast, is one UTOPIA revision away.

### Active groups holding the building blocks (with P estimates)

| Group                                           | Has both ingredients?                           | P(scoops the exact contribution ≤12 mo)                                                           |
| ----------------------------------------------- | ----------------------------------------------- | ------------------------------------------------------------------------------------------------- |
| **UTOPIA authors (Kaitian Jin et al.)**         | Yes — already published 80%                     | **40–50%** they pre-empt the abstention framing. Primary risk.                                    |
| **Zou / Sun + R. Ma (Stanford), TISSUE**        | Conformal coverage machinery; lacks H&E input   | **30–40%** — porting TISSUE's stratified conformal to an H&E regressor is the obvious next paper. |
| **Levy lab (Dartmouth/Emory), AACR 2026 #4189** | Bayesian-NN gene-conditional UQ for H&E→ST, now | **High-medium** — one conformal wrapper away; Levy has prior conformal/biomedical UQ.             |
| **STimage authors (Nguyen lab, UQ)**            | Ensemble UQ plumbing; no guarantee              | **Medium** — calibration/coverage is the natural v2, but just published Nat Commun.               |
| **Mahmood lab + Jaume (HEST-1k, SEAL)**         | Owns data + FM ecosystem; UQ separate           | **Medium-high capability, lower intent** — risk is "if they decide to," not "if they can."        |
| SpaRED/BCV-Uniandes; CHRep (BUPT)               | Benchmark / post-hoc bias calibration only      | **Low.**                                                                                          |

### Prior negative results to address

No clean published "we tried UQ for H&E→ST and it failed," but two hard cautionary signals the gate **must** preempt:

- **Vacuity ceiling.** Best H&E→ST Pearson ≈ 0.2; UTOPIA's own central finding is that "reliable inference often emerges only at coarser, biologically meaningful scales." A per-spot/per-gene coverage-guaranteed layer risks abstaining on ~everything informative or emitting vacuously wide intervals. "Guaranteed coverage" at 95% abstention is a tautology, not a result.
- **Irreducible aleatoric floor.** The Levy abstract frames much of the H&E→molecule uncertainty as _biological limits on what morphology can reveal_ — so abstention may be dominated by intrinsic noise, not actionable epistemic uncertainty. The seed must show **non-trivial coverage at useful retained fractions**, leaning on gene/region-stratification to keep guarantees where the map _is_ identifiable.
- **Calibration on noisy labels is circular if unvalidated.** Calibrating abstention against dropout-corrupted ST means the guarantee holds against the _noise model_, not biology. The dropout-into-risk fix is novel **and** unproven — must be validated against a cleaner Xenium subcellular gold reference, not asserted.

### Dataset/benchmark existence (HF) + method/library existence (Context7)

- **HF Hub (datasets/benchmarks/models):** The substrate exists and is mature — HEST-1k (Jaume 2024), HESCAPE (2025), STImage-1K4M, SEAL, HistoPrism, ST-Align, plus the generic conformal papers (K-RCPS, CRC-for-pulmonary-nodules, CP-with-missing-values, PAC-prediction-sets). **No Hub artifact pairs conformal/selective/abstention with H&E→ST.** UTOPIA is not (yet) a Hub dataset/benchmark — it is a bioRxiv preprint + GitHub repo. **Verdict: the claimed-novel benchmark/abstention artifact does not exist on the Hub; the prediction/benchmark substrate does.**
- **Context7 (library existence):** `resolve-library-id` for "conformal prediction / selective prediction / abstention" returns only `conform` (an unrelated HTML form-validation library). **No indexed library implements conformal-on-spatial-omics abstention.** Generic conformal libraries (MAPIE, crepes, torchcp) exist off-Context7 and are reusable for the CP machinery; TISSUE ships research code (not a maintained library) and is the closest existing implementation. **Verdict: no off-the-shelf library does the claimed-novel thing; the generic CP primitives are reusable, the domain method is not implemented anywhere.**

### Scoop risk: P(scooped before WACV 2026) = 35–45%

Decomposition: **P(something materially close — calibrated/selective reliability for H&E→ST — appears or a preprint is revised to include it before ~Aug 2026) ≈ 35–45%**, driven mostly by UTOPIA absorbing abstention in revision (~40–50%). But **P(someone lands the _exact_ contribution — distribution-free finite-sample coverage via conformal risk control + dropout-into-risk-target + Visium-spatial-non-exchangeable validation + selective-risk abstention, as a distinct citable claim) ≈ 15–20%.** Nearest group: **UTOPIA authors.** Move fast; lead with the elements UTOPIA and TISSUE cannot claim.

### Novelty score: 6/10 + justification

**As stated, honest novelty = 6/10.** This is a domain port + recombination of off-the-shelf conformal primitives (CRC, Mondrian/group-conditional, non-exchangeable CP, label-noise CP — all credited above) onto a saturated low-SNR task that sits **one modality-port from TISSUE (Nat Methods 2024) and one revision from UTOPIA (Mar 2026)**. The "gene-and-region-conditional coverage **guarantee**" headline is, as written, reviewer-fatal: distribution-free _conditional_ coverage is provably impossible (Foygel-Barber, Candès, Ramdas, Tibshirani 2021) — only Mondrian/group-conditional _marginal_ coverage is honest. The three Phase-A subagents and the council independently landed 6–6.5; the Phase-A score of 7 predated the UTOPIA surfacing and a precise read of TISSUE.

**The score is recoverable to ~7–7.5 — but only via a specific sharpening, and I will not bandage a 6 into a 7 by relabeling.** The one element that genuinely clears prior art is the **dropout-noise-into-conformal-risk-target + Visium-non-exchangeable-validated + selective-risk-abstention** tri-combination (Step-1 "what survives it"), reframed around **where the map is unidentifiable** rather than a per-spot guarantee.

---

## Step 2 — Multi-Perspective Audit (council)

### 4-voice verdict

**Architect:** SHARPEN-then-proceed. A narrow intersection survives unscooped; headline the dropout-aware risk control + abstention curve + spatial-non-exchangeability, confront TISSUE/UTOPIA head-on.
_Why:_ the tri-combo is unscooped across all domains; the data moat + clinical reframing dodges the Pearson-0.2 race.

**Skeptic:** KILL as stated → pivot. Thin recombination of off-the-shelf parts on a low-SNR task, ~one revision from UTOPIA (P(scoop) 0.6–0.7).
_Why:_ every ingredient is admittedly off-the-shelf; conditioning-on-gene+region _is_ Mondrian, dropout-into-risk _is_ label-noise CRC — four named methods stapled, not a new one. **Reframe that survives:** ship a held-out **benchmark of where the morphology→molecule map is unidentifiable** (abstention-vs-error oracle) — saturation becomes the contribution, can't be scooped by a method paper, survives Pearson 0.2.

**Pragmatist:** PROCEED as SHARPEN-in-disguise — ship the narrow core fast (late-July), freeze scope now.
_Why:_ don't compete on accuracy (unwinnable); the deployable abstention layer is the one thing the field hasn't shipped; speed is the dominant variable (15–20% exact-scoop). **Headline the guarantee that empirically _holds on real Visium_, not the novelty number** — reviewers verify a coverage plot, not a claim.

**Critic:** PROCEED only if reframed as a risk-control benchmark; as stated leans KILL.
_Why:_ (1) vacuity trap — coverage at 95% abstention is a tautology, and the finest-grained contribution is the part most likely empty; (2) conditional-coverage overclaim is desk-reject-grade (Barber 2021); (3) calibrating on dropout-corrupted labels is circular and unvalidated. **The real fight is the spatial-non-exchangeability argument** — without a weighted/spatial-CP guarantee, even the _marginal_ claim fails.

**Consensus:** Do **not** proceed as written. Three of four reject the as-stated framing; the "conditional guarantee" headline and the per-spot vacuity trap are the shared blockers.
**Strongest dissent:** Skeptic's KILL (scoop 0.6–0.7) — rejected only because the dropout-into-risk + spatial-non-exchangeable + abstention tri-combination is verifiably unscooped in _any_ domain, which a pure "add abstention to UTOPIA" scoop does not cover.
**Premise check:** Yes — Skeptic and Critic both challenged the question, converging on reframing unidentifiability _into_ the contribution (a benchmark + oracle) rather than claiming a per-spot guarantee. That reframe is adopted into the sharpening.

---

## Verdict & Gate

### Decision: **SHARPEN** (do not start Phase C; return to Phase A to ratify the reframed seed)

As-stated novelty is **6/10**, below the ≥7 bar. Per the gate rule, the seed is not killed — a genuinely uncovered sharpening exists — but it does **not** proceed to Phase C as written, and I will not inflate 6→7 by relabeling.

### The one gap prior art genuinely does not cover (the sharpening)

> **Reframe Seed 1 from "a per-spot coverage _guarantee_ on H&E→ST" to "a dropout-noise-aware, spatially-valid _selective-risk_ layer + an unidentifiability benchmark for H&E→ST," and re-enter Phase A to ratify it.** Concretely, the sharpened seed must carry all four, none of which any verified prior work (TISSUE, UTOPIA, STimage, CRC, label-noise CP, spatial CP) combines:
>
> 1. **Dropout/measurement-noise model folded into a conformal _risk_ target on the latent clean transcriptome** (validated against a cleaner Xenium gold reference, not asserted).
> 2. **Coverage validated under Visium spatial non-exchangeability** — empirically show naive split-conformal under-covers on Moran's-I-positive spots; a weighted/spatial-block CP restores it.
> 3. **A selective-_risk_ (abstention) operating curve** with **Mondrian/group-conditional marginal** coverage only — never a "conditional guarantee" (Barber 2021).
> 4. **Unidentifiability reframed as the artifact** (Skeptic/Critic): a held-out benchmark of _which genes/regions are intrinsically unpredictable_, with an abstention-vs-error oracle and pre-registered selective-risk-coverage as the headline metric — preempting the Pearson-0.2 vacuity trap by retaining guarantees at coarse/region scale.

If the sharpened seed re-clears Phase A at ≥7 (projected ~7–7.5, conditional on element 1 being demonstrated and TISSUE/UTOPIA confronted explicitly), it proceeds to Phase C. **Realistic WACV outlook if it ships clean and pre-scoop:** accept ≈ 35–40% (anchored to WACV's ~25–30% base rate, adjusted up for a verifiable-coverage systems paper with a unique data moat and clinical framing); **oral ≈ 4–6%** (WACV orals are roughly the top ~10% of accepts → ~3% base, adjusted modestly up). Honest, not inflated: this is a strong _accept-band_ paper, a _long-shot_ oral — and only if it ships before UTOPIA closes the gap.

**STOP at gate. Phase C (architecture/planning) not started.**

---

# Phase B-delta — Seed A2 (unidentifiability oracle), Rollout 4 (2026-06-11)

> Delta kill gate on the **reframed** seed selected after Rollout 3. Not a full Phase B re-run — verifies only the new oracle claim + two flip-triggers, then a focused 4-voice council on the reframed go/no-go. Tools that ran: 3 parallel Opus retrieval subagents (parallel-search rate-limited mid-pass → WebSearch + HF fallback; bioRxiv 403'd on two full-text fetches, flagged below), HF Hub `hub_repo_search`, Context7 `resolve-library-id`, Consensus `search`, council (Skeptic/Pragmatist/Critic). `literature-review` / `recursive-research` / `/ars-lit-review` / `exa-search` were executed-in-spirit by the retrieval subagents (no separate Exa key — same fallback disclosed in Rollout 2), not as separate skill loads.

## Selected seed (A2)

A cleaner-reference **unidentifiability oracle** for H&E→spatial-transcriptomics: use Xenium/subcellular truth (lower dropout) to label which genes/regions are _intrinsically_ unpredictable from morphology vs. merely dropout-corrupted, validated by an abstention-vs-error oracle, powering a Mondrian group-conditional **marginal** selective-risk (abstention) layer. Headline = the oracle (noise-vs-unidentifiability separation + deferral curve); the conformal layer is acknowledged off-the-shelf.

## Step 1 — Novelty Verification (delta)

### Oracle-claim novelty: OPEN, 7.5/10

No verified work ships a benchmark whose **output is the per-gene/region intrinsic-unpredictability label, validated by an abstention-vs-error oracle from a cleaner reference that separates technical dropout from biological non-identifiability.** Two single-axis occupants must be cited and distinguished, neither sinks it:

- **EPS — "Quantifying predictability of gene expression from histology," bioRxiv 2025.11.04.686651.** Expression Predictability Score = negative-log graph-Laplacian quadratic form over an image-embedding kNN graph; validated by correlating with model PCC. **Closest on the "per-gene predictability as output" axis, but it is a smoothness heuristic computed from the _same noisy_ expression — no cleaner-reference oracle, no deferral curve, and structurally cannot separate dropout from biological unidentifiability** (a dropout-zeroed gene and a morphology-invisible gene both look non-smooth). _(Internals from search snippets only — bioRxiv 403'd; verify the PDF before submission.)_
- **"Impact of Data Quality on DL Prediction of ST from Histology," bioRxiv 2025.09.04.674228.** Uses paired Visium/Xenium serial sections + in-silico ablation — but **controls biological variability _out_ to isolate technical variation (the inverse goal)**, outputs per-gene performance, no oracle/abstention. Motivates the claim rather than scooping it.
- **Bhadra–Anastasio "On hallucinations in tomographic reconstruction" (IEEE TMI 2021) — clean delta:** their null space is **algebraic** (known linear forward operator), **ground-truth-requiring**, **diagnostic-only** (no deferral curve), on **linear inverse problems**. A2's unidentifiable subspace is **statistical/learned** (H&E genuinely cannot determine the discrete molecular target), the diagnostic is an **abstention-vs-error operating curve**, and validation is by a **cleaner-reference oracle** that dissociates dropout from biology — a distinction with no analog in a noiseless linear model. No 2022–2026 follow-up ports null-space/identifiability maps to learned regression or omics.
- **Consensus grounding (independent):** the signal-vs-noise / aleatoric-vs-non-identifiability split is provably **not identifiable from the predictive distribution alone** ([Decoupled PFNs, 2026](https://consensus.app/papers/details/6679d1019f56588981d3896a1958c7e5/?utm_source=claude_desktop); [Credal Concept Bottleneck, 2026](https://consensus.app/papers/details/1addc67f3a3358608749817d61b1b299/?utm_source=claude_desktop); [Hüllermeier & Waegeman, 2019](https://consensus.app/papers/details/74c4ccbeaf295d239d35b48b0f407e5e/?utm_source=claude_desktop)). Decoupled PFNs resolves it **only** by using external query-level labels for the noiseless target and the noise variance — which is exactly what a cleaner reference (Xenium) plus a Xenium **replicate** noise-floor supplies. This both validates the approach and pre-supplies the required control (see council).

### Flip-trigger A — UTOPIA / Zou / Levy: DOES NOT FIRE

UTOPIA still bioRxiv **v1**; GitHub `kaitianjin/UTOPIA` last push **2026-05-28 (bugfix only)**; it does conformal **p-values + Benjamini-Hochberg FDR control for detection**, NOT a selective-risk/abstention layer and NOT an unidentifiability oracle (mentions Xenium capture noise only descriptively). TISSUE (Zou/Sun) — no 2025–26 follow-up to H&E input or abstention found. Levy AACR #4189 — abstract-only, no preprint. No other 2026 abstention-for-H&E→ST preprint. **Lane open as of 2026-06-11.** Anchor novelty on the oracle (harder to scoop than the abstention layer, which UTOPIA is one step from).

### Flip-trigger B — clean-reference triad ≥4 organs: DOES NOT FIRE

**5 high-confidence registrable Visium(+HD)+Xenium+H&E triads:** breast (Janesick, GEO **GSE243280** — note: not GSE243275), colon (10x Post-Xenium CytAssist), lung (10x Visium HD Post-Xenium, Xenium-5K), liver/HCC and ovary (Nat Commun 16:9232 = s41467-025-64292-3, 4-platform serial). +brain = 6. **Binding caveats:** (1) the auditable universe is the **Xenium panel-gene intersection** — ~280–377 genes for v1-panel organs, ~5k only where Xenium-5K used; (2) **Xenium is cleaner-not-clean** (off-target probe binding, own capture noise); (3) adjacent-section **registration error** can masquerade as unidentifiability; (4) the **"HEST trap"** — HEST-1k aggregates _independent_ Xenium and Visium samples sharing an organ label but not a tissue block, so it is a histology supplement, **not** a same-block triad source.

### Dataset (HF) + method/library (Context7) existence

HF Hub `hub_repo_search` for an unidentifiability/predictability benchmark dataset → **none found.** Context7 → only generic benchmarking libraries (Google Benchmark, pytest-benchmark) → **no library implements an identifiability map / cleaner-reference oracle.** The benchmark and the method do not exist off-the-shelf.

## Step 2 — Multi-Perspective Audit (council, reframed go/no-go)

**Architect:** PROCEED, scoped. Oracle 7.5 verified-open, flip-triggers cold, benchmark/library absent.
**Pragmatist:** PROCEED fast (1-week Phase-C sprint); lock the oracle's operating estimand, the per-organ auditable gene set, and the abstention-vs-error protocol. _Surprise:_ the Xenium-5K organs are an **internal generalization test** — calibrate on panel-organs, show the oracle holds at 5k → converts the gene-ceiling from a weakness into a **scaling result** (the accept→champion flip).
**Skeptic:** SCOPE-DOWN. 7.5 prices the _gap_, not the _validatability_; a clean delta on a thing you can't measure non-circularly is ~6.5. Gate Phase C on a non-circular unidentifiability label + a known-truth control.
**Critic:** SCOPE-DOWN. Three failure modes: (1) **circular oracle** — Xenium noise + registration confound the "intrinsic-unpredictability" label; (2) **vacuity** — at Pearson≈0.2 the layer may abstain on most genes (report retained-fraction × utility up front; kill if <~15% useful); (3) marginal-only honesty sits one inch from UTOPIA, so the novelty _is_ the oracle — if it's confounded, nothing is left. _Surprise (the 1% catch):_ panel-intersection genes are **selected for detectability/morphology-correlation**, so the oracle is audited on the _easiest_ genes — the unpredictable regime it most wants to characterize is systematically outside the auditable set.

**Consensus:** 3 of 4 → PROCEED-but-SCOPE-DOWN; the shared blocker is the oracle's circularity/validatability, not its novelty. **Strongest dissent:** Critic's "coverage-guaranteed certificate built on uncorrected Xenium noise = rigorous-looking, scientifically void." **Premise check:** yes — Skeptic and Critic both challenged whether the oracle is _testable_; resolved by adopting a mandatory pre-Phase-C control, which the Consensus grounding ([12]) shows is the principled (not merely defensive) fix.

## Verdict & Gate

### Decision: **PROCEED to Phase C — conditional (scoped + one gating control)**

Novelty clears the bar (oracle claim **7.5/10**, verified-open; both flip-triggers cold; benchmark and library do not exist). The council did not contest novelty — it conditioned **feasibility**. Three pre-registrations are mandatory and gate the _start_ of architecture:

1. **Non-circular oracle + Xenium-replicate noise-floor control.** Before any unidentifiability claim, decompose reference variance into technical-Xenium-noise vs. registration-error vs. residual (candidate-unidentifiability), using replicate Xenium sections; define the unpredictability label from a source independent of the predictor's own residuals. Principled precedent: signal/noise is not identifiable from the predictive distribution alone → external noiseless+noise-variance labels are required ([Decoupled PFNs 2026][12]).
2. **Honest scope rewrite + scaling framing.** Claim a **panel-conditional** audit, not a genome-wide unidentifiability map; pre-register the per-organ auditable gene set; use the Xenium-5K organ(s) as the breadth/scaling test (Pragmatist) — ceiling → scaling result. Disclose the **detectability-selection bias** of the panel intersection (Critic) and characterize, not claim, the oracle's coverage of the hard regime.
3. **Vacuity guard.** Report retained-fraction × utility as a headline, not a footnote; pre-register the kill criterion (e.g., <~15% useful retained at target selective-risk, or abstention concentrating on dropout rather than unidentifiability after the noise-floor correction).

### Realistic WACV odds (PROCEED)

Anchored to WACV's ~25–30% accept base rate: with the oracle reframe at 7.5, a unique data moat, and the noise-floor control directly answering the #1 reviewer objection (circularity), **accept ≈ 40–45%** if it ships clean and pre-scoop. **Oral ≈ 5–8%** — the "ceiling→scaling result" plus first noise-vs-unidentifiability oracle is a plausible champion-flip, but orals are a small fraction of accepts, so this stays a minority outcome. Honest: strong accept-band, plausible-but-minority oral. The scoop clock (UTOPIA one step from a selective-risk curve) makes speed the dominant variable.

**STOP at gate. Phase C architecture/planning NOT started — it begins only once the three pre-registrations above are written.**
