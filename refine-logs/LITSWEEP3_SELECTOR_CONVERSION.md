# LITSWEEP-3 / L1 — Selector-Conversion Lens

**Agent:** litsweep-3 L1 (ZERO GPU, ZERO test-touch). **Date:** 2026-07-25.
**Lens (unique):** *how does the literature convert per-item selection-locked oracle headroom using only training labels?*
**Deliverable:** this file + one local commit (no push). All citations verified via web search (arXiv ID + venue/year confirmed).

---

## 0. BOTTOM LINE (read first)

**My lens is pre-closed by three banked results; the literature corroborates the closure, it does not open it.**

- **F47 (ROUTER_GATE_RECORD.md, 30d0ee1):** a trained per-item selector over the two deployed channels is DEAD at
  **all three supervision sources** — unsupervised (K9 zeros), train-supervised (target degenerate, CLIP LOO 0.998),
  dev-supervised (**−0.046** at the maximally-favorable dev-CV ceiling, below the permutation null p95 = +0.0042).
  Decision-level meta-features (vote margins, purity, sub-votes, confidence differential, transcript stats) carry
  **no per-item routing signal**, GBM or linear. Re-open requires a selector input that is *a genuinely NEW
  information source not derivable from banked features/votes.*
- **F49 (MJ_FORENSIC_RECON.md):** any such new source must first demonstrate **alignment > 0.663** with the oracle
  routing decision from banked evidence. The only new source tested (MLLM modality-locus judgment) came in at
  **0.588 < 0.663**.
- **F66 (ISR_PREGATE_RECORD.md, a6e41f8):** the conversion is **arithmetically selection-locked**. Oracle headroom
  decomposes into a *symmetric/legal* slice (reachable by non-selecting operators) and a *selection* slice (reachable
  only by banned per-item gold selection). Measured: HateMM **+0.0776 = +0.0012 legal + +0.0764 selection (98%)**;
  EN **+0.0700 = +0.0064 + +0.0636 (91%)**. Router headroom (F47 §3.1) is the same shape: MHC oracle **+0.108** exists
  but is per-item **unpredictable**.

**The deepest obstacle — specific to our pipeline, and the one every candidate below dies on:** the deployed predictor
is a **kNN vote over the train memory**, so on the train split it **memorizes** (CLIP LOO acc **0.998**, F47 §3.2).
A selector trained on "which operator/segment errs" therefore sees a **degenerate, near-error-free** target on the only
split I may train on (train-disagreement "Qwen correct" = **0/109, 0/102, 0/92**), and that train base rate is the
**inverse** of the dev/test base rate (~0.55). **Training labels cannot supervise the test-time selection decision in
this pipeline, no matter how good the selector's input signal is.** This is a data-generating-process obstacle, not a
model-capacity one — it is upstream of every learning-to-select method in the literature.

**Verdict:** **no candidate in this lens carries a prior above ~3% of clearing the house bar** (+0.030 acc AND +0.030
mF1, 3/3, both protocols, ≥1 new dataset). The literature's value here is **PAPER-VALUE** (13 items, §3) — it lets the
paper *name and formalize* Law-I (the selection-lock) with the field's own vocabulary, which strengthens the negative
into a contribution. Two moves are worth surfacing as **new result axes / diagnostics** (not accuracy levers), both
requiring a **user protocol ruling** (§2 shortlist #1, #2).

---

## 1. LENS SWEEP — each mechanism family vs our constraints

Format per family: **(a)** test-time selector input (must be label-free) · **(b)** published evidence it recovers a
meaningful fraction of oracle-SELECTION gains (numbers) · **(c)** honest transplant · **(d)** strongest reason it fails
here · **(e)** prior of clearing the house bar.

### 1.1 Learning-to-defer / learning-to-reject (L2D)
Mozannar & Sontag 2020 (arXiv **2006.01862**, ICML 2020); Verma & Nalisnick 2022 (arXiv **2202.03673**, ICML 2022);
softmax-parametrization follow-up (arXiv **2311.01106**, NeurIPS 2023).
- **(a)** classifier confidence + features → a learned rejector `r(x)` that defers to an expert.
- **(b)** L2D beats *both* classifier-alone and expert-alone when the expert is **complementary** (human–AI teaming);
  the reported system gains are a few points over `max(classifier, expert)` **only when a genuine expert is queried at
  test time**. The gain is a function of expert complementarity, not of the rejector recovering an *oracle* selection.
- **(c)** map "experts" → our CLIP-arm and Qwen-arm (or segments); "defer" → route. This is **exactly F47**: the
  rejector consumes the same decision features F47 measured at zero signal.
- **(d)** Two kills. (i) **No test-time expert exists in-box** — the only candidate "expert" is another frozen operator
  we already hold, and F47 measured routing to it at −0.046 (below perm-null); the stronger experts (72B / API) are
  banned. Deferral target is empty. (ii) The rejector is trained on train labels, but our memorization (LOO 0.998)
  makes the train deferral target degenerate and **train→dev inverted** (F47 §3.2); L2D consistency theory assumes
  iid train/deploy error distributions, which our memory violates.
- **(e)** **~0.** Structurally isomorphic to F47 (per-item selection over banked channels).

### 1.2 Selective prediction / rejection learning
SelectiveNet — Geifman & El-Yaniv 2019 (arXiv **1901.09192**, ICML 2019); Deep Gamblers — Ziyin et al. 2019
(arXiv **1907.00208**, NeurIPS 2019); cost-based rejection — Charoenphakdee et al. 2021 (arXiv **2010.11748**, ICML 2021).
- **(a)** a learned selection score `g(x)`; abstain below threshold.
- **(b)** improved **risk–coverage tradeoff**: accuracy **on the covered subset** rises as coverage drops. At 100%
  coverage the selective classifier **equals the base classifier** — no full-set gain, by construction.
- **(c)** abstain on low-confidence kNN-vote items; the abstained items still need a prediction to produce a full-test
  number → the fallback is a second operator (**F47-routing, dead**) or a default (loses those items).
- **(d)** **Definitional:** selective prediction is not a full-set-accuracy lever; it trades coverage for
  covered-accuracy. The house bar is +0.030 on the **full 150-sample test**. The selection score `g(x)` is a
  confidence/margin — the exact banked feature F47 nulled.
- **(e)** **0 as an accuracy lever.** BUT a legitimate **new result axis** (risk–coverage curve) and PAPER-VALUE — see
  shortlist #2. Needs a **calibration-split protocol ruling** (FLAG).

### 1.3 Per-instance operator/expert routing (MoE gating over classifiers)
Expert-Choice Routing — Zhou et al. 2022 (arXiv **2202.09368**, NeurIPS 2022); oracle-routing gap — "Routing Manifold
Alignment" 2025 (arXiv **2511.07419**).
- **(a)** a gating net `π(x)` over operators.
- **(b)** the MoE literature itself reports a **10–20% oracle-routing gap** in the *data-rich LLM* regime (2511.07419):
  even there, learned routing leaves 10–20 accuracy points of the oracle unrecovered. That is the *good* case.
- **(c)** = **F47** verbatim (gating over CLIP-arm / Qwen-arm).
- **(d)** In our regime (n≈2–4k train, memorizing memory, 150-test) the recoverable fraction measured at **~0** (F47),
  vs the field's 80–90%-recovered in the data-rich case. The literature **corroborates** the closure quantitatively.
- **(e)** **0** (= F47).

### 1.4 Instance-difficulty estimation
V-usable info / PVI — Ethayarajh, Choi & Swayamdipta 2022 (arXiv **2110.08420**, ICML 2022 outstanding);
Dataset Cartography — Swayamdipta et al. 2020 (arXiv **2009.10795**, EMNLP 2020).
- **(a)** training-dynamics difficulty (confidence/variability across epochs) or PVI; at test time, a difficulty
  predictor ≈ margin/confidence = banked.
- **(b)** these are **diagnostic** tools (identify easy/hard/ambiguous/mislabeled items) for data selection and
  curriculum — **not** test-time accuracy levers. No claim of recovering oracle-selection gains.
- **(c)** the *valuable* transplant is **diagnostic**: use **PVI ($0, on banked caches)** to quantify how much of the
  selection-locked headroom is V-extractable from label-free features — formalizing F66's arithmetic (Law-I). See
  shortlist #1 + PAPER-VALUE §3.
- **(d)** difficulty estimation reorders training / flags data; at test time it does not pick the right operator or
  segment. Our curriculum-SFT (cand-2, F56) already exercised difficulty-ordering → **ZH TIE**.
- **(e)** **0 as a lever; high as PAPER-VALUE / $0 diagnostic.**

### 1.5 Agreement / disagreement-based selection
(Query-by-committee lineage; the prompt flags "operators' agreement patterns are label-free features.")
- **(a)** cross-operator agreement / sub-vote disagreement indicators.
- **(b)/(c)** these are precisely the features **F47 already included** (per-modality sub-votes, purity, agreement
  indicator, `vote_CLIP − vote_Qwen`, `|vote_CLIP| − |vote_Qwen|`) — measured **zero** routing signal, GBM and linear.
- **(d)** agreement over the **existing** operators is banked-derivable = dead. Genuinely-diverse **new** committee
  members would need a fresh independent draw → **cross-seed ensemble (vetoed)**, and F47's null extends to sub-vote
  agreement anyway.
- **(e)** **0.**

### 1.6 Conformal / margin-based per-item switching
Conformal intro — Angelopoulos & Bates 2021 (arXiv **2107.07511**); Conformal Risk Control — Angelopoulos, Bates et al.
2022 (arXiv **2208.02814**, ICLR 2024).
- **(a)** nonconformity score / conformal set-size per item.
- **(b)** conformal delivers coverage-**guaranteed** sets / a bounded monotone risk (e.g., FNR) — a *guarantee*, not a
  new routing signal.
- **(c)** "switch to operator B when A's conformal set is ambiguous" = F47-routing with a conformal trigger replacing
  the GBM trigger — same target, same banked margin.
- **(d)** the switch trigger is a monotone function of the margin F47 already nulled; conformal adds rigor, not signal.
  Standalone, conformal is an **abstention/uncertainty axis** (= §1.2) needing a **calibration split (FLAG)**.
- **(e)** **0 as a lever; PAPER-VALUE** for a rigorous uncertainty / FNR-control table (deployment-relevant for a
  moderation system).

### 1.7 Test-time-augmentation (TTA) consistency — the ONLY F47-carve-out-eligible signal
Pitfalls of in-domain uncertainty — Ashukha et al. 2020 (arXiv **2002.06470**, ICLR 2020).
- **(a)** re-encode the query under K augmentations (crop / frame-jitter / temporal-subsample) through Qwen; per-item
  prediction variance/entropy. This is **genuinely NOT in the bank** — the only candidate that clears the *letter* of
  the F47 carve-out.
- **(b)** Ashukha 2020: TTA is **competitive with deep ensembles for uncertainty**, but TTA-derived uncertainty is
  **highly correlated with softmax confidence** (largely redundant with margin). **No published evidence** that
  TTA-consistency recovers a meaningful fraction of oracle-**selection** gains.
- **(c)** honest transplant: (i) re-extract Qwen embeddings under K views for train (memory) + dev — **NOT $0**, ~K×
  the extraction cost, several GPU-h, **user-gated download/compute**; (ii) run an **F49-style $0 alignment gate**: does
  TTA-variance agree with the oracle routing decision on the dev-disagreement subset at **>0.663**? Only then does it
  clear the carve-out.
- **(d)** four intact obstacles: (i) the routing **target is train-non-transferable** (memorization) regardless of
  signal quality — F47 §3.2 kills it upstream of any input; (ii) Ashukha's redundancy → TTA-variance ≈ confidence →
  likely **alignment < 0.663** (probably worse than the MLLM's 0.588); (iii) needs GPU re-extraction (violates $0);
  (iv) even if aligned, it targets the per-item **selection** slice F66 caps and F47 non-transferability blocks.
- **(e)** **~2–3%** — the least-dead accuracy-lever in the lens *only* because it injects genuinely new information;
  every downstream obstacle survives. **Recommend NOT spend** (GPU-gated, prior < 3%).

### 1.8 Snapshot / hypothesis selection trained on train labels
- Per-instance selection among checkpoints/snapshots = F47 over snapshot-votes (banked). Adjacent kills already banked:
  cross-seed ensemble **vetoed**; single-trajectory SWA **dead** (F62/F62b); grad-norm checkpoint selection **dead,
  sign-inverted** (F69). **(e) 0.**

---

## 2. RANKED SHORTLIST (max 3) — honest priors, decisive-cell sketches, kill-switches

> Framing: **none of these clears the house bar.** They are the three highest-EV moves the selector-conversion lens can
> produce, ranked by (value delivered × feasibility). Priors are the probability of a **+0.030/+0.030 3/3 both-protocol**
> pass; where that is 0 by construction I say so and label what the move *does* deliver.

**#1 — $0 PVI selection-lock diagnostic (PAPER-VALUE / Law-I formalization).**
Prior of clearing house bar: **0** (it is a measurement, not a lever). EV: **highest in the lens.**
- *One-line rationale:* turns F66's arithmetic selection-lock into a citeable **V-usable-information** statement — "the
  selection-locked headroom is information not V-extractable by any label-free predictor family V" — using the field's
  own outstanding-paper vocabulary (Ethayarajh 2022, PVI).
- *Minimal decisive cell ($0, no GPU, banked caches only):* on the banked train/dev feature caches, fit a small
  predictor family V (linear + shallow MLP) and compute **PVI** of the label given (i) the pooled key vs (ii) the
  per-segment / per-channel *selection target*; show the selection-target PVI ≈ 0 for label-free V while the
  gold-selection oracle PVI is large. Cross-reference to the F47 dev-CV −0.046 and F66 91–98% decomposition.
- *Kill-switch:* if label-free-V PVI on the selection target is **> 0** and monotone with a realizable selector gain,
  the diagnostic would *contradict* F47 → escalate (would mean F47/F66 mis-measured; do not expect this).
- *Ruling needed:* none ($0, analysis-chapter addition). **Serves the user's oracle-queue ruling directly.**

**#2 — Conformal / selective-prediction risk–coverage as a NEW RESULT AXIS.**
Prior of clearing house bar: **0** (abstention does not move full-set accuracy). EV: new table + pillar-4 tie-in.
- *One-line rationale:* a moderation system's deployable value includes a **rigorous abstention / false-negative-control
  curve** (SelectiveNet / Deep Gamblers risk–coverage; Conformal Risk Control FNR bound) — a legitimate new axis that
  ties to pillar-4 (auditable/editable archive) without touching the accuracy claim.
- *Minimal decisive cell:* fit `g(x)` (or a conformal nonconformity score) on a **calibration split**, plot the
  risk–coverage curve on the covered dev set; report FNR bound at fixed coverage (Angelopoulos–Bates 2022).
- *Kill-switch:* the curve is a *description*, not a bar — there is no pass/fail; it is reported as a deployment axis,
  never as a +0.03 accuracy claim.
- *Ruling needed (FLAG):* requires a **calibration split distinct from the 150-sample test** — a **protocol change**
  (carving calibration from dev shrinks the already-underpowered dev; carving from test is out-of-box). **User must
  rule** before any calibration-split experiment. Do not assume allowed.

**#3 — TTA-consistency alignment gate (only carve-out-eligible accuracy lever).**
Prior of clearing house bar: **~2–3%.** EV: low; GPU-gated.
- *One-line rationale:* the only signal in the lens **not derivable from banked features/votes** (F47 carve-out letter),
  hence the only thing that could *even be gated* for a new selector input.
- *Minimal decisive cell:* an **F49-style $0-once-extracted alignment gate** — after K-view Qwen re-extraction, test
  whether TTA-variance agrees with the oracle routing decision on the dev-disagreement subset at **>0.663**. **The
  re-extraction is NOT $0** (~several GPU-h, user-gated download/compute).
- *Kill-switch:* alignment **< 0.663** (F49 bar) **OR** corr(TTA-variance, banked margin) **> 0.9** (redundancy, the
  Ashukha-2020 prior) **OR** the train-non-transferability obstacle (F47 §3.2) unresolved → KILL.
- *Recommendation:* **do not queue.** GPU-gated + prior < 3% + three intact obstacles. Record as user-gated option only.

**Also-considered, killed pre-shortlist:** L2D-to-a-trained-student (§1.1, expert-empty in-box); agreement-based
selection (§1.5, banked-derivable = F47); conformal per-item switching as a *lever* (§1.6, banked margin trigger);
snapshot/hypothesis selection (§1.8, cross-seed veto + F62/F69).

---

## 3. PAPER-VALUE LIST (13 items — worth citing regardless of transplant)

Formalize/characterize the negative (Law-I = selection-lock). All arXiv IDs + venue/year verified via web search.

1. **Ethayarajh, Choi & Swayamdipta 2022** — *Understanding Dataset Difficulty with V-Usable Information* — arXiv
   **2110.08420**, ICML 2022 (outstanding paper). **[TOP]** Formalizes Law-I; PVI = $0 quantifiable on our caches (#1).
2. **Swayamdipta et al. 2020** — *Dataset Cartography* — arXiv **2009.10795**, EMNLP 2020. Easy/hard/ambiguous map;
   characterizes the MHC label-limited datasets (F44/F74 label-limit story).
3. **Mozannar & Sontag 2020** — *Consistent Estimators for Learning to Defer to an Expert* — arXiv **2006.01862**,
   ICML 2020. Canonical L2D; cite to frame why *no-test-time-expert* forecloses deferral here.
4. **Verma & Nalisnick 2022** — *Calibrated Learning to Defer with One-vs-All Classifiers* — arXiv **2202.03673**,
   ICML 2022. Modern calibrated-L2D line.
5. **(softmax-parametrization L2D)** — *In Defense of Softmax Parametrization for Calibrated and Consistent L2D* —
   arXiv **2311.01106**, NeurIPS 2023. Companion to #4.
6. **Geifman & El-Yaniv 2019** — *SelectiveNet* — arXiv **1901.09192**, ICML 2019. Canonical selective prediction;
   risk–coverage framing for the abstention axis (#2).
7. **Ziyin et al. 2019** — *Deep Gamblers: Learning to Abstain with Portfolio Theory* — arXiv **1907.00208**,
   NeurIPS 2019. Alternative abstention objective.
8. **Charoenphakdee et al. 2021** — *Classification with Rejection Based on Cost-sensitive Classification* — arXiv
   **2010.11748**, ICML 2021. Theoretically-grounded rejection surrogate / rejection-cost framing.
9. **Angelopoulos & Bates 2021** — *A Gentle Introduction to Conformal Prediction and Distribution-Free UQ* — arXiv
   **2107.07511**. Uncertainty-set framing for a rigorous abstention table.
10. **Angelopoulos, Bates et al. 2022** — *Conformal Risk Control* — arXiv **2208.02814**, ICLR 2024. Bound a monotone
    risk (e.g., FNR) with finite-sample guarantee — deployment-relevant moderation-system table (#2).
11. **Zhou et al. 2022** — *Mixture-of-Experts with Expert Choice Routing* — arXiv **2202.09368**, NeurIPS 2022.
    Canonical learned-routing reference.
12. **"Routing Manifold Alignment…" 2025** — arXiv **2511.07419**. Reports a **10–20% oracle-routing gap** in data-rich
    MoE LLMs — external, quantitative corroboration that learned routing under-recovers oracle (supports F47 negative).
13. **Ashukha et al. 2020** — *Pitfalls of In-Domain Uncertainty Estimation and Ensembling* — arXiv **2002.06470**,
    ICLR 2020. TTA≈ensembles for UQ **and** UQ≈confidence redundancy — the prior against any confidence-like new signal.

*(Methodological companion, optional 14th):* *Overcoming Common Flaws in the Evaluation of Selective Classification
Systems* — arXiv **2407.01032** — cite if a risk–coverage table (#2) is added.*

---

## 4. FLAGS REQUIRING A USER RULING

- **Calibration-split protocol (shortlist #2):** conformal / selective-prediction needs a calibration set **distinct
  from the 150-sample test**. Carving it from dev shrinks an already-underpowered dev; carving from test is out-of-box.
  **Protocol change → user ruling required.**
- **TTA re-extraction (shortlist #3):** needs K-view Qwen re-extraction (GPU, user-gated compute/download). Not $0.
- **Transductive / test-time adaptation:** **not proposed** here (TTA-consistency does not update weights). If any
  future variant *updates on the test distribution*, that is transductive → out-of-box → **FLAG, do not assume allowed.**

---

## 5. PROVENANCE / HYGIENE

- **ZERO GPU, ZERO test-touch, no Modal.** Read-only: `state/directions_tried.json`, `state/findings.jsonl` (F61–F76),
  `ROUTER_GATE_RECORD.md` (F47), `ISR_PREGATE_RECORD.md` (F66), `MJ_FORENSIC_RECON.md` (F49).
- **Citations:** all 13 (+1 optional) arXiv IDs + venue/year confirmed by live web search 2026-07-25; no unverified refs.
- No `state/`, prereg, config, or frozen artifact mutated. One local commit on `main`, **not pushed.**
