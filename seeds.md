# Phase A Seeds — CV / WACV 2027 Oral (Novelty-First)

**Run:** Phase A ideate, 2026-06-11. **Method:** 5-area parallel landscape scan (Opus subagents; parallel-search + Consensus + bioRxiv + Hugging Face saturation check) → scientific-brainstorming shaping → scientific-critical-thinking + what-if-oracle stress-test → recursive-decision-ledger ranking (`decision-ledger.jsonl`).
**Compute ceiling assumed:** 192 GB VRAM total. **Data reality:** medical/scientific imaging = both compute + data; conformal = high compute; VLM = probe→method; diffusion = fine-tune scale; mech-interp = full-scale SAEs.
**Bar:** WACV oral; novelty × impact first, acceptance a floor.

---

## The convergence (read this first)

The same gap is Top-1 in **3 of 5** scanned areas (medical, conformal, diffusion) and shapes the VLM Top-1 too: **in medical/scientific imaging the dangerous failure is not average error — it is _confident_ error on the subset where the answer is unidentifiable or fabricable, and SOTA everywhere ships an unguarded point estimate.** Turning "I don't know here" into a _calibrated, sometimes-trainable, sometimes-mechanistically-grounded_ output — reported by selective risk-coverage, not accuracy — is the open lane that fits this researcher's GCS-Conformal track record × medical-data access × 192 GB better than anyone competing on raw prediction accuracy. All three seeds below are independent WACV papers **and** rungs of one coherent program.

---

## Seed 1 — Faithfulness-gated abstention for inferred spatial omics _(anchor; lowest risk, highest impact)_

**Core idea (one sentence):** A distribution-free, gene-and-region-conditional selective-prediction layer that sits on any H&E→spatial-transcriptomics regressor and abstains (or returns a risk-controlled interval) exactly where the morphology→molecule map is unidentifiable — converting today's point-estimate "virtual omics" into a clinically defensible measurement with a coverage guarantee.

**Gap it attacks:** H&E→spatial-omics _prediction_ is a graveyard — 20+ methods, 3 standardized benchmarks (Chan 2023; SpaRED/SpaCKLE 2025; SEAL 2026) — yet **none ship validated, gene/region-conditional abstention**. STimage (2026) estimates aleatoric+epistemic uncertainty but never calibrates it to a coverage guarantee or reports selective risk-coverage.

**Why the obvious fix fails:** Bolting MC-dropout or generic split-conformal on top breaks twice. (1) The ground truth is itself dropout-corrupted (SpaRED) → you'd calibrate abstention against label noise, not biological unidentifiability. (2) Visium spots are spatially autocorrelated → split-conformal exchangeability is violated and the "guarantee" is void. **The fix to both _is_ the contribution:** conformal risk control whose risk target absorbs the measurement-noise model, under a spatially non-exchangeable calibration scheme, with the nonconformity score conditioned on gene and tissue region.

**First-pass impact:** High (9). Reframes a saturated, un-orallable task into a trustworthiness result — you can finally tell a pathologist _which_ virtual-omics calls to trust. Directly clinical, directly fundable (Regeneron/ISEF narrative).

**Stress-test verdict:** Survives the "conformal-on-a-new-dataset" attack because it does **not** compete on prediction accuracy (it would lose to 20+ incumbents) — it changes the problem. **Δ-risk:** the abstention signal tracks dropout noise rather than true unidentifiability → mitigate by anchoring the risk definition on a higher-quality Xenium subcellular gold reference and pre-registering selective risk-coverage as the headline metric.
**Scores:** Novelty 7 · Impact 9 · Feasibility@192GB 9 · DataAccess **have** · Scoop **low** (data moat is unique to you) · Composite **8.50**.

---

## Seed 2 — End-to-end trainable _worst-group conditional_ conformal coverage _(method frontier; mandated direction)_

**Core idea (one sentence):** Train the backbone end-to-end through a differentiable _group-conditional_ conformal quantile so that subgroup miscoverage (across scanner / site / demographic / lesion subtype) is equalized **by shaping the representation**, not by a post-hoc patch — making conditional validity a property the learned features induce.

**Gap it attacks:** The prompt's flagged frontier — conformal is "now end-to-end trainable." But the naive framings just closed: CRT (NeurIPS 2025) owns generic end-to-end conformal _risk_; AT (ICLR 2026) learns image-adaptive thresholds for segmentation with a **frozen backbone**; Kandinsky / Rectifying-Scores (ICML 2025) improve conditional coverage **post-hoc on a fixed model**. Nobody backprops the _group-conditional_ quantile _into the encoder_.

**Why the obvious fix fails:** Mondrian / per-group calibration is the obvious answer and it starves on rare strata — the exact groups where marginal CP hides catastrophic undercoverage get the largest, useless sets. End-to-end feature sharing is what lets data-starved groups borrow strength; that is a clean, testable mechanism no post-hoc method can reach.

**First-pass impact:** High (9) if the delta holds; venue-general (applies beyond medical, with multi-site medical imaging as the killer app). Highest method-prestige of the three.

**Stress-test verdict:** **Hard constraint to honor:** distribution-free conditional coverage is _provably impossible_ (Vovk; Foygel-Barber) → the claim must be **worst-group efficiency at fixed marginal validity**, never a conditional guarantee. That honesty is reviewer-rewarded if framed precisely, reviewer-fatal if sloppy. **Δ-risk + highest scoop (~40%):** a fast-follower ships "group-conditional ConfTr" first, or the end-to-end gain fails to beat a CRT+Mondrian stack → mitigate by benchmarking against that stack head-on, targeting rare-stratum efficiency explicitly, and moving fast.
**Scores:** Novelty 7 · Impact 9 · Feasibility@192GB 9 · DataAccess **have** · Scoop **high** · Composite **7.60**.

---

## Seed 3 — VLM modality-gap abstention _(highest novelty, cleanest statistics; a single empirical bet)_

**Core idea (one sentence):** A lightweight, calibrated abstention signal for any VLM that runs the same query with the image present vs. masked and uses the answer divergence — text-only-answerable ⇒ ungrounded — as a conformal nonconformity score, turning the documented "VLMs answer from language priors" failure into a deployment-time selective-prediction guarantee.

**Gap it attacks:** The phenomenon is now rigorously documented (VLind-Bench 2024; two 2026 visual-ignorance diagnostics) and the counterfactual-masking primitive exists — but every prior use is a **benchmark or a steering intervention**, never a _calibrated, guaranteed abstention score_. Conformal-for-VLM work (ConfLVLM, Ye SCP) all scores answer-logits, which collapse when the model is confidently wrong about an ungrounded region.

**Why the obvious fix fails:** Logit-confidence abstention fails precisely on the dangerous case (confident + ungrounded). The masking-divergence signal is orthogonal to logit confidence — but only if divergence actually correlates with correctness, which is the whole bet.

**First-pass impact:** Medium-high (7). Method not probe; fully feasible on one 7B VLM within 192 GB; pairs with your medical imaging as a domain-grounded instantiation (acquisition-shift-robust clinical-triage VLM).

**Stress-test verdict:** Cleanest statistics of the three (post-hoc calibration → exchangeability holds, guarantee is sound _if the score is informative_). **Δ-risk:** image-present-vs-masked divergence correlates too weakly with correctness → coverage achievable only near 100% abstention (useless). This is a falsifiable empirical bet — pre-register a divergence↔correctness AUROC gate before committing. **Scoop ~30%** (AI2 reframes its diagnostic as a UQ method first) → move fast.
**Scores:** Novelty 8 · Impact 7 · Feasibility@192GB 9 · DataAccess **partial** · Scoop **moderate** · Composite **7.65**.

---

## Watch bench (not promoted — kept visible)

- **D · SAE features as the nonconformity signal** (mechanistically-grounded abstention in a pathology FM). Highest ceiling (Novelty ~9, no prior wires SAE features into conformal scores) but **three stacked load-bearing premises**, and the base one — vision SAE features are causally meaningful — is _actively under attack_ (Sanity-Checks 2026; AxBench shows simple baselines beat SAEs). De-risk only by benchmarking SAE-vs-PCA-vs-diff-in-means as the spurious-feature detector _first_. High variance; revisit if Seed 1/2 stalls.
- **E · Conformal FDR over hallucinated _structures_ (not pixels) in diffusion reconstruction** (Novelty 7, pretrain-free). Strongest single backup but conceptually adjacent to Seed 1 (both = conformal-on-generative-medical).
- **F · SAE dissection of a pathology FM (Virchow/UNI) with pathologist-validated morphology + biomarker steering** (Novelty 7). WACV-fit marginal (interp-venue pull); heating fast (3 medical-SAE papers in 12 mo).

---

## Honesty notes (research-grade)

- No novelty score here exceeds 8; none is a paradigm shift, and none is claimed as one. Seeds 1–3 are **new framings/regimes of known tools** (conformal risk control, selective prediction) onto unaddressed high-stakes events — which is exactly the defensible band for a WACV oral.
- All closest-prior works cited were surfaced and existence-checked via the scan tools; verify each against the primary source before any draft.
- Feasibility@192GB confirmed for all three: frozen-encoder + conformal calibration (Seed 1), multi-seed ViT-L conformal training (Seed 2), single 7B VLM inference + post-hoc calibration (Seed 3). None needs from-scratch foundation-model pretraining.

---

# Rollout 3 — Sharpened seeds (post-Phase-B re-entry, 2026-06-11)

**Why this rollout:** Phase B gated Seed 1 at **SHARPEN** (as-stated novelty 6/10 — UTOPIA bioRxiv Mar 2026 and TISSUE Nat Methods 2024 compress it; the "gene/region-conditional coverage _guarantee_" headline is reviewer-fatal per Foygel-Barber 2021). This rollout ran the Phase-A machinery (4 focused area subagents → scientific-brainstorming → what-if-oracle + scientific-critical-thinking → ledger) to ratify the sharpened seed and test whether a _sibling rung of the same program_ now beats it. See `novelty.md` and `decision-ledger.jsonl` (rollouts 2–3) for the evidence chain.

## The reframe (read this first)

The Phase-A convergence was "coverage-guaranteed abstention on inferred/generative medical measurement is the moat." Phase B showed the **H&E→ST rung is crowding** (UTOPIA, TISSUE, Zou lab). The rollout-3 scan found the genuinely ownable, still-open object is not the rung — it is the **primitive underneath all of them: a cleaner-reference _unidentifiability oracle_ that separates measurement noise from biological unidentifiability, validated by an abstention-vs-error oracle.** No verified prior work ships this. Elements 1–3 of the original Seed 1 (dropout-into-risk, spatial non-exchangeability, Mondrian selective risk) are domain ports of occupied methods (~5–6 each); **element 4 — unidentifiability-as-the-measured-object with a deferral oracle — is the only ≥7 path**, and it is the same primitive that instantiates on virtual staining and diffusion reconstruction. Own the primitive; instantiate it on the rung where the data moat is real.

## Seed A2 — Unidentifiability oracle for inferred molecular measurement (anchor; the sharpened Seed 1)

**Core idea (one sentence):** A cleaner-reference _unidentifiability oracle_ — using Xenium/subcellular ground truth to label which genes/regions are intrinsically unpredictable from morphology vs. merely dropout-corrupted — that powers a selective-risk (abstention) layer on any H&E→spatial-transcriptomics regressor, making _where the morphology→molecule map is unidentifiable_ the measured object, not an afterthought.

**Gap it attacks:** Across ~24 H&E→ST predictors and 3 benchmarks, plus UTOPIA (calibrated confidence + FDR) and TISSUE (conformal intervals on imputation), **no one separates measurement-dropout from biological unidentifiability, and none ships a benchmark whose output is the per-gene/region unpredictability label validated against a deferral oracle.** The nearest relative anywhere is Bhadra–Anastasio null-space hallucination maps (IEEE TMI 2021) — but that is ground-truth-needing, diagnostic-only, no abstention curve, and for _linear_ inverse problems, not learned morphology→molecule regression where the null space is statistical.

**Why the obvious fix fails:** Bolting conformal intervals on an H&E→ST regressor (TISSUE-style) calibrates against dropout-corrupted Visium → the "abstention" tracks label noise, not biology, and at best Pearson ≈ 0.2 it abstains on ~everything (vacuity trap). The fix _is_ the contribution: a paired-platform clean reference (Visium has zero where Xenium confirms a transcript ⇒ dropout, not unidentifiability; both platforms + morphology can't resolve ⇒ true unidentifiability) gives a defensible oracle, with Mondrian group-conditional **marginal** coverage only (never a conditional guarantee).

**First-pass impact:** High (8). Unique private + public data moat (HEST Xenium n≈65; Janesick GSE243275 Visium+Xenium+H&E; 2025 subcellular benchmarks). Feasibility 7 (TISSUE + HEST + MAPIE scaffold ~70%; 192 GB over-provisioned). **Binding constraint (pre-register):** clean reference is small-N, breast-dominated, targeted-panel-genes, adjacent-section registration error → scope the headline to a breast-led, few-organ proof-of-concept, not a pan-cancer transcriptome-wide claim. Novelty **7–7.5 if the oracle is the headline**, 6–6.5 if bundled as four conformal tweaks. **Scoop high (UTOPIA/Zou 40–50%) — move fast.**

## Seed VS — Conformal per-region hallucination abstention for virtual staining (hedge; less-scooped sibling)

**Core idea (one sentence):** The same oracle primitive on a less-crowded rung — conformal risk control over per-region virtual-stain trustworthiness for H&E→IHC (HER2/ER/PR/Ki-67), abstaining with a clinician-set false-trust rate, using paired real IHC as the cleaner reference.

**Gap it attacks:** Virtual staining is saturated/commercial, but the best hallucination detector — AQuA (Ozcan, _Nature BME_ 2025) — is **heuristic, image-level, H&E-only, with no coverage guarantee**, and the authors explicitly defer IHC/IF and formal guarantees. No conformal/risk-controlled per-region "don't trust this stain here" exists.

**Why the obvious fix fails:** Image-level hallucination scores (AQuA) can't tell a clinician _which region_ of a synthetic IHC to distrust at a guaranteed error rate; a per-region conformal-risk layer with paired-IHC calibration can. You beat AQuA on guarantees rather than being scooped by it.

**First-pass impact:** High (7). Scoop ~30% (Ozcan owns the detection framing, not the conformal/IHC instantiation). **Feasibility UNVERIFIED this round** (BCI/HER2 paired sets plausible; needs a confirm pass) — that gap is the reason it is the hedge, not the anchor.

## Seed E — Structure-level conformal FDR over diffusion-reconstruction hallucinations (second hedge; cleanest framing, highest dominant-group scoop)

**Core idea (one sentence):** Distribution-free FDR control over _fabricated anatomical structures_ (lesion/vessel/organ-level, not pixels) in diffusion-based medical image reconstruction — certifying "is this structure real or invented by the prior?" at a user-set false-discovery rate.

**Gap it attacks:** K-RCPS (Teneggi 2023) gives pixel intervals; sem-CRC (2025) gives organ-grouped pixel intervals; the Hallucination Index (2024) is uncalibrated; Dai et al. (2025) does instance-FDR for _discriminative_ segmentation. **Structure-level conformal FDR over _generative_ hallucinations is the open cell.**

**Why the obvious fix fails:** Pixel-interval coverage and image-quality metrics don't answer the discrete clinical question ("did the model invent this lesion?"); a structure-level FDR target does. **Risk:** the cell is open because it's the logical product of two public guarantees from one lab (Sulab/Teneggi/Tivnan, JHU) — they hold every building block → **scoop 30–40%, highest dominant-group exposure of the three.** 192 GB most comfortable of all (pretrain-free).

## Dropped this rollout

- **SAE-as-abstention (old watch-bench D): NO-GO.** Base premise (vision SAE features causally meaningful enough to gate clinical abstention) is under active attack — Sanity-Checks-for-SAEs (2026), AxBench (simple baselines beat SAEs), Peng/Movva position paper. A hostile reviewer's rejection is pre-written. Keep SAEs only as a _discovery_ layer atop a guarantee from elsewhere, never as the guarantee.
- **VLM masking-divergence abstention (old Seed 3 / C): demote.** The core score was published Feb 2026 (Visual Information Gain) for selective training, and the conformal-abstention machinery (CAP, ConfLVLM) is public → snap-together novelty, scoop 45–55%. Viable only with a _theoretical_ claim that masking-divergence is the correct conformal score.

## Ranking (rollout 3)

1. **Seed A2** — anchor. Highest evidence quality (multi-source converged + feasibility-verified + unique data moat), program-coherent primitive. Novelty 7–7.5 scoped; scoop high but survivable; the only seed whose feasibility was de-risked this round.
2. **Seed VS** — strongest hedge. Genuinely less scooped, same primitive transfers; needs one feasibility confirm pass before it can anchor.
3. **Seed E** — second hedge. Cleanest one-line differentiation and most compute-comfortable, but highest single-group scoop risk (JHU/Sulam).

**No-regret move across all three:** make the **cleaner-reference unidentifiability oracle** the contribution — it is the transferable asset whichever rung wins. **STOP — pick one before the tight Phase B-delta (re-verify the chosen reframe's oracle claim) → Phase C.**
