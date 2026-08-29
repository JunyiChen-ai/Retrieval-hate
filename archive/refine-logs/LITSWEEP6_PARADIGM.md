# LITSWEEP-6 — PARADIGM-level alternatives to "encode → compare → one-shot per-item decision"

Date: 2026-07-27. Cost: $0 (WebSearch/WebFetch + read-only repo forensics; zero GPU, zero SLURM,
zero Modal, zero test contact). Lens ordered by the team lead: **data-to-output FLOW novelty**, not
another operator inside the existing flow.

Every paper below was fetched from its arXiv abstract page and confirmed to exist with the stated
title/authors/date. Where the abstract page did not state code availability, that is said explicitly
rather than guessed.

---

## 0. The one sentence that reorganises this sweep

Across three independent measurements this project has now shown the **same** structural fact:

| measurement | ranking quality | decision quality |
|---|---|---|
| F95 pair-verify (2026-07-27) | verifier beats cosine at the relation level by **+0.13 to +0.27 pair-AUC**, 18/18 cells, 4.3–8.8× over bar | end-to-end **0/36** cells |
| W4 temporal (research-wiki/EVAL_temporal_memory_W4.md) | EN temporal-split **ROC 0.8484**, *higher* than the random-split reference 0.7175 | macro-F1 **drops −0.084**; only 8.7% of scores clear t=0.5 against a 24.2% true positive rate |
| F88 ERRPAT | correct analogue present at median rank ~1.5 | it is out-voted; errors are *confident* inversions, ~90% seed-invariant |
| F50/F48 | AUC 0.898 | "unconvertible" (the original wording) |

**Our system ranks much better than it decides, everywhere we have measured it.** Every dead direction
in `directions_tried.json` tried to close that gap by finding a better *uniform decision rule* over the
same scores (vote operators F89, k F94, thresholds F88, losses F75, verifiers F95). That axis is now
closed from five directions.

The paradigm move this sweep recommends is therefore not another rule: it is to **change the output
object so that ranking quality is what gets cashed in** — a three-way output with a certified risk
budget (R1), and an operating point that is a policy rather than a constant (R2). Both are exactly the
literatures the lead asked for, and both consume our measured negatives as *premises* instead of being
blocked by them.

---

## R1 — Risk-controlled selective adjudication with a RELATIONAL audit score  ★ RANK 1

### (a) Verified papers

1. **Improving Selective Classification with Pairwise Queries for Binary Classification** — Harsh
   Vardhan, Sunav Choudhary, Natwar Modani, Arya Mazumdar. arXiv:2605.30615, submitted 28 May 2026
   (cs.LG). Venue not stated on the abstract page; no code link visible.
   *Their finding, verbatim in substance:* confidence estimates "might be inconsistent with model
   predictions, leading to high error" on accepted samples; **pairwise queries to the same model detect
   those high-error samples** and improve the rejection decision. Evaluated on synthetic + 4 in-context
   -learning binary datasets.
2. **Selective Conformal Risk Control** — Yunpeng Xu, Wenge Guo, Zhi Wei. arXiv:2512.12844, v1 14 Dec
   2025, v2 27 Apr 2026 (cs.LG). Two-stage: select confident samples, then apply conformal risk control
   *on the selected subset*. SCRC-T computes thresholds jointly over calibration+test and gives **exact
   finite-sample** guarantees; SCRC-I is calibration-only with PAC-style guarantees.
3. **The Confidence Gate Theorem: When Should Ranked Decision Systems Abstain?** — Ronald Doku.
   arXiv:2603.09947, 10 Mar 2026 (cs.AI). *This is the constraint, not the method.* Abstention improves
   decision quality monotonically only under **C1 rank-alignment and C2 no inversion zones**;
   structurally-grounded confidence signals "fail under contextual drift, producing as many monotonicity
   violations as random abstention."
4. **LLM Performance Predictors: Learning When to Escalate in Hybrid Human-AI Moderation Systems** —
   Bachar, Levi, Mishra, Levi, Minhas, Miller, Ben-Porat, Sheetrit, Morra. arXiv:2601.07006, 11 Jan
   2026, **AAMAS 2026**. Cost-aware selective classification for real moderation workflows; escalation
   driven by log-probs/entropy/uncertainty-attribution. *Venue precedent that this framing is publishable
   in moderation.*

### (b) The paradigm in one data-flow sentence

Theirs (ours today): encode → cosine kNN over memory → rank-weighted vote → **a binary label for every
item, always**. Ours becomes: encode → memory **nominates** precedents → a **relational audit** asks
"are this query's stated relations to its neighbours consistent with those neighbours' labels?" →
conformal calibration on a held-out split → output ∈ {hate, non-hate, **⊥ refer**} with a
**distribution-free bound on the error rate of the auto-decided subset**.

### (c) Which CP it solves + the novelty sentence

Primarily **CP-A**, and it solves it by accepting CP-A's own diagnosis. CP-A says the information to fix
the errors exists but the exchange rate into a uniform decision rule is < 1 and the residual is
per-item-conditional. Selective prediction is the paradigm whose success metric *is* per-item-conditional:
you never have to convert the signal into a global rule, you only have to **rank items by how likely the
decision is wrong**. Secondarily **CP-C**: conformal validity is finite-sample and distribution-free —
n = 78–161 calibration items is the regime it was built for, not a regime it merely tolerates.

> Novelty sentence a reviewer reads: *"On hateful video, retrieval-memory detectors fail in a specific
> way — their errors are confident neighbourhood inversions, and we show a relational signal that
> detects them is measurably stronger than the similarity the system decides with, yet provably cannot
> be converted into full-coverage accuracy. We therefore change the output: the detector abstains and
> refers, and the relational signal is spent on a certified risk budget instead of on accuracy. This is
> the first risk-controlled formulation of hateful-video detection."*

Mechanism novelty inside the paradigm (this is the part that is ours, not the literature's): **the
nonconformity score is relational**, computed from the F95 verifier's judgements on (query, neighbour)
pairs, not from a softmax/margin. The Confidence Gate Theorem is why this matters and not a detail:
our errors *are* inversion zones (F88), so C2 is violated and a confidence-based gate is predicted to
fail. A relational score is the documented escape route (paper 1), and we already measured that our
relational scorer is 4.3–8.8× better than the cosine at the relation level (F95 control 1).

### (d) Transplant sketch

- Reuse the **frozen** `scripts/analysis/mechnov_pairverify.py` (sha256 `77b0defd…b7240d`) verifier —
  trained on train-split pairs only, never on dev/test.
- Audit statistic per item: `a(x) = Σ_j w_j · v(x, n_j) · (2·y_{n_j} − 1)` over the banked top-k
  neighbours, where `v` is the verifier's same-class-likelihood. The *selective* score is the
  **inconsistency** between `a(x)` and the deployed cosine vote — i.e. exactly paper-1's "confidence
  inconsistent with prediction" detector, instantiated on retrieval relations.
- Calibrate a threshold on the audit statistic by split conformal on a clean calibration split; report a
  risk-coverage curve and, at a fixed target risk, the achieved coverage.
- Deployment framing per paper 4 (AAMAS): referred items go to a human queue, so the headline is the
  accuracy-cost trade-off, not accuracy alone.

### (e) Cheapest pregate — **$0, fully banked, no new test contact**

Everything needed already exists on disk:
`scripts/analysis/p2_out/cache_MHC_s{0,1,2,3}.json` (EN: per-item predictions + top-60 neighbour lists,
4 deployed seeds, EXACT — F94 reproduced the k=20 vote bit-exactly), `errpat_remint_dumps/*.pkl`
(HateMM/ZH CPU proxies validated to 4dp), `mechnov_pairverify_{en,hatemm,zh}_OUT.json`, and
`errpat_hatemm_peritem.csv`.

Pregate: compute risk-coverage curves for (A) baseline `|vote margin|` and (B) the relational audit
statistic, paired per seed per dataset. Pre-register: **B must beat A at ≥2 of the 3 coverage points
{0.70, 0.80, 0.90} on ≥2 datasets, sign-consistent across all seeds**, against a permutation null that
shuffles verifier scores within-query to price the curve's noise floor. Runtime: minutes on ≤8 CPU
threads.

**What CANNOT be pregated cheaply, stated plainly:**
1. **Whether an abstention deliverable satisfies the goal clause.** At 100% coverage this changes
   nothing; "+3 acc on ≥2 datasets" is unmet *by construction*. That is a **user ruling on the
   deliverable**, not an experiment, and it must be obtained before any GPU.
2. **Exchangeability of the calibration split.** Our dev sets have been used for val-selection, which
   breaks the exchangeability that the guarantee rests on. Two honest fixes: run under the
   **final-epoch protocol** (no val selection ⇒ dev is free and exchangeable), or split dev
   select/calibrate and halve n. *Note the synergy:* this gives an independent, method-level reason to
   settle the pending ZH val-selection retirement ruling — retirement makes dev a legal calibration set.
3. **Deferral cost model.** We have no data on human moderator accuracy or cost. Reporting "system
   accuracy with human = gold" is the standard L2D protocol but is a known inflation; it needs explicit
   cost accounting or it will be read as a trick.
4. **Granularity.** 1 test item = 0.47–0.67% on n = 149–215, so risk-coverage curves are coarse step
   functions. Per-bin item counts must be printed, and paired sign tests must carry the claim, not a
   smooth AUC difference.

### (f) Honest kill-risk vs our measured laws

- **Highest risk, and it is real:** F88 says errors are *confident*, and F89 T1 was **degenerate**
  (identical predictions on 215/215 HateMM, 149/149 ZH) — the local class prior is not separable from
  the retrieval signal in the cone-collapsed space. If the verifier's ranking is also uninformative on
  the ~22–28 stable-core errors, the risk-coverage curve is flat and this dies at the $0 pregate. That
  is the whole point of running the pregate first.
- **Structural distinctness from F95** (a reviewer will ask): F95 measured the verifier's **argmax and
  aggregation** as a decision replacement and it failed 0/36. R1 reads the verifier's **ranking of an
  audit statistic** and never lets it decide a label. Different functional (ranking vs argmax),
  different output space ({0,1} vs {0,1,⊥}), different success metric (selective risk vs accuracy).
  F95's control-1 pass is the *evidence for* R1; its control-2 failure is the *reason* R1 does not try
  to decide.
- **Confidence Gate C1/C2 must be validated on held-out data before deployment** — the paper says so
  explicitly, and our F88 says C2 is violated for the *confidence* signal. The pregate must therefore
  report C1/C2 diagnostics for the relational signal, not assume them.
- Not blocked by: law-I (no representation change), F63 (no transduction), F94 (k unchanged), F89
  (no vote-operator swap — the vote is untouched; a gate is added downstream), any banned constraint.

---

## R2 — Deployment-as-a-process: an anytime-valid, drift-adaptive operating point  ★ RANK 2

### (a) Verified papers

1. **Efficient Online Conformal Selection with Limited Feedback** — Sreenivas Gollapudi, Kostas Kollias,
   Kamesh Munagala, Ali Sinop. arXiv:2605.14953, 14 May 2026 (cs.LG). Applies the **ACI update rule to
   control parameters** under **bandit feedback**, with Lyapunov analysis giving (i) **adversarial
   validity** — the target is met on average *for any input sequence, hence under distribution shift* —
   and (ii) **sublinear efficiency regret** for i.i.d. inputs.
2. **Distribution-informed Online Conformal Prediction** — arXiv:2512.07770, **ICLR 2026** (stated on
   the PDF header). Target coverage with tighter sets than prior online CP under shift.
3. Classical anchor: Gibbs & Candès, *Adaptive Conformal Inference Under Distribution Shift* (NeurIPS
   2021) — the update rule the 2026 work builds on.

### (b) The paradigm in one data-flow sentence

Theirs (ours today): train once → freeze a threshold at 0.5 → evaluate on one i.i.d. test split. Ours
becomes: the detector is a **policy over a time-ordered stream**, its operating point is updated by an
**anytime-valid rule** consuming an O(1) label budget per interval, and the reported object is the
**risk/coverage trajectory over the stream** with a guarantee that holds for *any* sequence.

### (c) Which CP it solves + novelty sentence

**CP-B**, and — this is the part I did not expect to find — **we already own the measurement that makes
it a paper.** `research-wiki/EVAL_temporal_memory_W4.md` (2026-07-03/04) reports, on real upload-date
temporal splits that exist on disk (`data/gt/MHC_temporal` 549/80/161, `data/gt/MHC_zh_temporal`
579/78/149, plus `MHC{,_zh}_upload_dates.jsonl`):

- EN temporal split costs **−0.084 macro-F1** (0.7113 → 0.6273) while **ROC rises to 0.8484**. The drop
  is an **operating-point failure, not a separability failure** (their words, §1).
- **Memory augmentation — the obvious fix — is flat-to-negative at every k on both languages** (adding
  the whole val period *hurts*: 0.5923). "Adaptation-gain per sample ≤ 0 at every k."
- **Threshold recalibration with k=20 new-period labels fully recovers the drop** (0.7336 ± 0.0190 ≥ the
  random-split floor 0.7113); threshold-only oracle 0.7646.
- **ZH is a clean negative control with no drift — and there the ad-hoc rule HURTS at small k**
  (k=5: 0.7114 ± 0.0393 vs static 0.7779; all-78: 0.7351). "Where there is no drift, moving the
  threshold on tiny calibration sets is pure noise."

That last line is the opening. Our current adaptation rule is "argmax macro-F1 on k labels", which is
**unsafe by measurement**: it loses 6.6 macro-F1 points when there is nothing to adapt to. An
anytime-valid update is precisely the object that is valid for *any* sequence including the null one.

> Novelty sentence: *"Hate evolves, and on video we measure that the evolution is calibration drift, not
> representation drift: the ranking survives (ROC rises) while the decisions collapse. We therefore make
> the operating point a first-class, anytime-valid policy with an O(1) label budget — recovering the
> drift with a guarantee, and provably not paying for it when there is no drift, which is exactly where
> the natural heuristic loses."*

The architectural claim already drafted in W4 §3 is the paper's spine and is genuinely ours: **a
retrieval-memory system exposes its operating point as a first-class, O(1), reversible knob, whereas a
trained head hides it inside the weights.**

### (d) Transplant sketch

Replace the "maximise macro-F1 on the k labelled samples" step in
`scripts/analysis/temporal_recalibration.py` with an ACI-style update on the decision threshold
(`t ← t + η(α − 1{miscovered})` in the conformal parameterisation), driven by the same k labels arriving
in time order. Report the coverage/risk trajectory on EN (drift) and ZH (no drift), against three
baselines: static t=0.5, the W4 argmax-macro-F1 rule, and the threshold-only oracle 0.7646/0.7845.

### (e) Cheapest pregate — **$0 to ~0.05 GPU-h**

The $0 part: the temporal heads are frozen-CLIP heads that train in **~1 minute on CPU** (F88 recorded
the HateMM head at 52 s on 8 CPUs), the temporal split files and upload-date files are on disk, and
`temporal_recalibration.py` already exists CPU-only with an internal consistency check that reproduces
the static numbers exactly. The whole EN-vs-ZH comparison of {static, argmax, ACI} is a CPU replay.

**What cannot be pregated cheaply:** whether the effect survives on the *current* encoders. W4 ran on
frozen CLIP; the deployed stack is Qwen-LoRA. Re-minting temporal heads on the already-banked Qwen
feature caches is cheap (features exist; head training is minutes of CPU), but it is a new arm and needs
its own parity gate against the W4 numbers before any claim moves.

### (f) Honest kill-risk

- **Test-budget hygiene:** the temporal test splits were already consumed on 2026-07-03/04. Any new
  headline on them is a second draw and must be declared as such.
- **One dataset carries the effect.** ZH is a negative control by measurement and HateMM has no upload
  dates (`data/gt/` has no `HateMM_temporal`), so the drift claim is EN-only, n=161. Under CP-C that is
  thin, and W4 itself lists the caveats: 17–19 positives in the val pool, survivor bias compressing the
  measurable drift, k=5 unstable with 2/5 seeds guarded.
- **The headroom is already known and small.** k=20 already recovers the drop; the honest contribution
  is *safety and validity*, not a bigger number. A reviewer who wants a bigger number will not be
  satisfied, and we should not pretend otherwise.
- Not blocked by any banned constraint; labels used are val-period labels, not test.

---

## R3 — Precedent-based adjudication: verdict + citations + an explicit distinguishing test  ★ RANK 3

### (a) Verified papers

1. **Reasoning over Precedents Alongside Statutes: Case-Augmented Deliberative Alignment for LLM Safety**
   — Can Jin, Rui Wu, Tong Che, Qixin Zhang, Hongwu Peng, Jiahui Zhao, Zhenting Wang, Wenqi Wei, Ligong
   Han, Zhao Zhang, Yuan Cao, Ruixiang Tang, Dimitris N. Metaxas. arXiv:2601.08000, 12 Jan 2026
   (cs.AI/cs.SE). Pairs **policies with illustrative cases** ("case-augmented simple codes") instead of
   rules alone; RL-trained safety reasoning chains; reports more robust and generalised safety behaviour
   with less over-refusal. No code/case-bank link visible on the abstract page.
2. **Thinking Longer, Not Always Smarter: Evaluating LLM Capabilities in Hierarchical Legal Reasoning**
   — arXiv:2510.08710. Formalises **"significant distinctions"**: cases are modelled by factual
   predicates (factors) organised into a hierarchy, with **verifiable rules** for identifying a
   distinction, analysing its argumentative support, and evaluating its significance.
3. Context: *Review of Case-Based Reasoning for LLM Agents* — arXiv:2504.06943.

### (b) Paradigm in one sentence

Theirs (ours today): memory returns neighbours, neighbours vote, **the neighbours are then discarded**.
Ours becomes: memory returns **precedents**; each precedent is explicitly **followed or distinguished**
by a verifiable test; the output is a verdict **plus its citation set and the distinguishing factors**;
precedent conflict that no distinguishing test resolves routes to ⊥ (composes with R1).

### (c) CP + novelty

**CP-D** and pillar-④. A distinguishing factor can be exactly *"harmful, but no protected-group target"*
— the 41% EN cluster stops being an error and becomes a named, auditable outcome. And editing one
precedent has a traceable effect on every decision that cited it, which is the auditable/editable-memory
pillar upgraded from a property to the output format.

> Novelty sentence: *"A retrieval-memory moderator should output what a moderator actually owes: a
> verdict, the precedents it rests on, and why the nearest contrary precedent does not control. We give
> the first precedent-cited formulation of hateful-video moderation and show the citation set is
> auditable — deleting a precedent changes exactly the decisions that cited it."*

### (d)+(e) Transplant + $0 pregate

Neighbours are already banked with labels, so the citation set is free. **The $0 pregate that must run
first is a falsification test:** does precedent *conflict* predict error? Run it on
`errpat_hatemm_peritem.csv` + the banked top-60 EN lists.

**I can already predict this pregate is at serious risk, from F88:** on error items the median top-20
purity toward the truth is **0.12–0.22**, i.e. the neighbourhood is 78–88% *wrong-label* — our errors
are not ambiguous precedent sets, they are **confidently wrong** ones. A "conflict ⇒ abstain" trigger
will therefore mostly fire on items we already get right. This is the same C2-inversion-zone problem the
Confidence Gate Theorem names, and it is the reason R1 uses a *relational* statistic rather than
neighbourhood agreement.

### (f) Kill-risk

The distinguishing test wants a reasoner, and **MLLM-reasoner-in-the-accuracy-path is dead 13 ways at
7B–72B** and banned. R3 is therefore honest **only** as a justification/audit layer — which is precisely
the guard-rail/audit role the settled MLLM campaign *earned*. Treat R3 as a companion contribution to
R1, not a standalone performance play. Also: `gold annotations inside method` is banned, so
"factors" must be derived from the memory's own train labels, never from target/time-span annotations.

---

## R4 — Anytime / sequential evidence accumulation over segments  ★ RANK 4 (efficiency deliverable)

### (a) Verified papers

1. **EFlow: Learning Evidence Flow for Long-Video Reasoning with Adaptive Reflection** — Wenhao Zhang,
   Kuanwei Lin, Xuyi Yang, Wei Gao, Ge Li. arXiv:2607.00867, v1 1 Jul 2026, v3 15 Jul 2026 (cs.CV).
   Temporal grounding → reasoning → **confidence-aware reflection** that re-examines the video when
   retrieved evidence is insufficient; explicitly targets avoiding "premature semantic commitment".
   Built on **Qwen3-VL**, trained with SFT + RL. No code link visible on the abstract page.
2. Theory anchor: anytime-valid sequential testing / e-processes (SPRT lineage); *Anytime validity is
   free: inducing sequential tests*, JRSS-B advance article, 2026.
3. In-domain neighbour to be aware of: **MultiHateLoc: Towards Temporal Localisation of Multimodal Hate
   Content in Online Videos**, arXiv:2512.10408.

### (b) Paradigm

Theirs (ours today): 8 uniformly sampled frames → **one mean-pooled vector** → one decision. Ours
becomes: segments are consumed in order, an e-value/likelihood ratio accumulates, and the system stops
as soon as an anytime-valid boundary is crossed — **evidence is never pooled**.

### (c) Why it is not just another decision rule

It changes the aggregation, not the readout, and it attacks a *named mechanism*: F65/F67 both died
through **mean-pool attenuation**, and F67 measured 16 frames saturating at 8. Sequential accumulation
is the one formulation in which a strong signal in one segment cannot be averaged away. We also have the
campaign's **only positive MLLM role** to build on: P6 localization (wv-AUC 0.5435 vs memory 0.5140,
paired p = 0.007) and P10-b (0.5755, CI [0.5581, 0.5933]).

### (d)–(f) Sketch, pregate, and the honest verdict

$0 pregate: on the banked HateClipSeg segment scores, ask whether a sequential stopping rule reaches the
video-level decision after fewer segments at equal accuracy. That is a pure replay.

**Honest kill-risk, and it is high on the accuracy axis.** The chain still terminates in a video-level
binary decision, and **law-I now has 9 certified data** — the cleanest being F91, where the image stream
genuinely improved (+0.0558, best ever on HateMM) and the conversion was *negative*. Nothing in the
sweep suggests sequential aggregation escapes that. The deliverable law-I does **not** govern is
**efficiency**: "we watch 30% of the video for the same accuracy" is a real claim on a real axis, and
EFlow's own framing is a latency/quality trade-off. Also: a full EFlow transplant is an MLLM reasoner in
the accuracy path (banned) and needs SFT+RL on Qwen3-VL, so only the *statistical* stopping-rule half is
transplantable within our constraints. Rank 4 because the honest deliverable is efficiency, and the goal
clause is about accuracy.

---

## R5 — Decomposed / checklist verdict for the label-semantics boundary  ★ RANK 5

**Verified paper (and it is prior art that constrains us):** **xList-Hate: A Checklist-Based Framework
for Interpretable and Generalizable Hate Speech Detection** — Adrián Girón, Pablo Miralles, Javier
Huertas-Tato, Sergio D'Antonio, David Camacho. arXiv:2602.05874, 5 Feb 2026 (cs.CL). An LLM answers a
checklist of concept-level questions; a **lightweight interpretable decision tree** aggregates the binary
diagnostic signals. Claims improved cross-dataset robustness under domain shift and **less sensitivity
to annotation inconsistency and contextual ambiguity** — i.e. someone has already published CP-D's cure
in text hate speech, five months ago. Companion: *Compositional Generalisation for Explainable Hate
Speech Detection*, arXiv:2506.03916.

Paradigm: single binary → a **conjunctive predicate output** (harmful ∧ group-targeted), so the EN
annotation boundary becomes explicit instead of appearing as 41% of the hard errors.

**Why it ranks last despite hitting CP-D squarely:** (i) the predicates want an LLM to answer them —
banned in the accuracy path; (ii) supervising them from the Offensive class is the **F82 graded-label**
object, parked, revivable only by a user ruling *with a new mechanism argument*; (iii) xList-Hate
already occupies the interpretable-checklist claim in hate speech, so our novelty would have to be
"first on video", which is thin on its own. Listed for completeness because it is the only candidate
that attacks CP-D structurally.

---

## PRE-KILLS — two candidates this sweep closes at $0 so no future round spends on them

**PK-1. Prior-matched transductive batch assignment (Sinkhorn/OT over the test batch).** Superficially a
real paradigm change (per-item independent argmax → one joint assignment under a class-prior constraint)
and it is an active 2026 area. **It is arithmetically already measured.** For a 1-D score, constraining
the batch to match a class prior *is* choosing a quantile threshold on that score. F88 already measured
the **test-fitted threshold oracle**: ZH **+0.0201, below bar**; HateMM threshold-recalibration dead; EN
dev-selected threshold dead deployably 0/6 arms. Any prior-matched assignment is therefore **capped
below a measured-dead oracle**. F89's T1 class-balanced quota was additionally *degenerate* (identical
predictions 215/215 HateMM, 149/149 ZH). **Do not spend on this family.**

**PK-2. Pairwise Difference Learning as a decode variant.** *Pairwise Difference Learning for
Classification* — Mohamed Karim Belaid, Maximilian Rabus, Eyke Hüllermeier, arXiv:2406.20031, 28 Jun
2024; predicts the label *difference* for a pair and decodes by **averaging over every training
example**; a Python package is released. This is the canonical form of F95's dead route and a reviewer
will name it. F95 used a top-10-per-class shortlist with max / mean-top-3 aggregation, so PDL's exact
decode (unweighted average over *all* n train anchors) was not literally run. It is a **decode variant
inside a family whose control-2 failed 0/36**, so the prior is very low — but if the family is ever to
be closed by name in the paper, it is a ~10-minute CPU addition to the already-frozen
`mechnov_pairverify.py` harness, not a new experiment.

---

## Ranking, and what I would actually do next

| rank | candidate | paradigm novelty | survival vs our laws | lifts performance? |
|---|---|---|---|---|
| 1 | **R1 selective adjudication, relational nonconformity** | high — output space changes to {0,1,⊥} with a certified budget; **no conformal or selective-prediction work exists in hateful meme/video** (searched, empty) | best: consumes F95/F88 as premises; blocked by no ban; $0 falsifiable | **not at full coverage.** New axis (selective risk / accuracy-cost). Needs a user ruling on the deliverable |
| 2 | **R2 anytime-valid drift-adaptive operating point** | high — train-freeze-test → policy over a stream | good; anchored on our own W4 measurement and real temporal splits | yes on the temporal split (EN 0.6273 → 0.7336 already measured), but the honest contribution is **safety/validity**, not a bigger number; EN-only, n=161 |
| 3 | **R3 precedent-cited adjudication** | high on presentation, medium on mechanism | reasoner must stay out of the accuracy path; the "conflict ⇒ abstain" trigger is at real risk from F88's purity 0.12–0.22 | no — auditability contribution; composes with R1 |
| 4 | **R4 sequential/anytime evidence** | medium-high | **low on accuracy** (law-I ×9); survives only as an efficiency claim | efficiency yes, accuracy unlikely |
| 5 | **R5 decomposed/checklist verdict** | medium — already published in text hate speech Feb 2026 | needs the F82 ruling + LLM predicates are banned in-path | unknown; blocked before it can be measured |

**Recommendation.** Run the **R1 $0 pregate** and the **R2 $0 CPU replay** in parallel — they share no
machinery, both run on banked artifacts, both are minutes of CPU, and between them they cover CP-A and
CP-B. R1 is the one that could carry a paper; R2 is the one most likely to produce a defensible number.

**The one thing that must come from the user before either can go past pregate:** R1 and R2 both change
what the deliverable *is* (a risk-coverage curve; a coverage trajectory). Neither satisfies "+3 acc on
≥2 datasets" at full coverage, and no amount of experimental cleverness will make them. That is a
deliverable ruling, and the pregates are worth running before it is made only because they cost nothing
and would tell us whether the ruling is even worth asking for.

---

### Sources

- [Improving Selective Classification with Pairwise Queries for Binary Classification (arXiv:2605.30615)](https://arxiv.org/abs/2605.30615)
- [Selective Conformal Risk Control (arXiv:2512.12844)](https://arxiv.org/abs/2512.12844)
- [The Confidence Gate Theorem: When Should Ranked Decision Systems Abstain? (arXiv:2603.09947)](https://arxiv.org/abs/2603.09947)
- [LLM Performance Predictors: Learning When to Escalate in Hybrid Human-AI Moderation Systems (arXiv:2601.07006, AAMAS 2026)](https://arxiv.org/abs/2601.07006)
- [Efficient Online Conformal Selection with Limited Feedback (arXiv:2605.14953)](https://arxiv.org/abs/2605.14953)
- [Distribution-informed Online Conformal Prediction (arXiv:2512.07770, ICLR 2026)](https://arxiv.org/pdf/2512.07770)
- [Reasoning over Precedents Alongside Statutes: Case-Augmented Deliberative Alignment for LLM Safety (arXiv:2601.08000)](https://arxiv.org/abs/2601.08000)
- [Thinking Longer, Not Always Smarter: Evaluating LLM Capabilities in Hierarchical Legal Reasoning (arXiv:2510.08710)](https://arxiv.org/abs/2510.08710)
- [Review of Case-Based Reasoning for LLM Agents (arXiv:2504.06943)](https://arxiv.org/pdf/2504.06943)
- [EFlow: Learning Evidence Flow for Long-Video Reasoning with Adaptive Reflection (arXiv:2607.00867)](https://arxiv.org/abs/2607.00867)
- [MultiHateLoc: Towards Temporal Localisation of Multimodal Hate Content in Online Videos (arXiv:2512.10408)](https://arxiv.org/html/2512.10408v3)
- [xList-Hate: A Checklist-Based Framework for Interpretable and Generalizable Hate Speech Detection (arXiv:2602.05874)](https://arxiv.org/abs/2602.05874)
- [Compositional Generalisation for Explainable Hate Speech Detection (arXiv:2506.03916)](https://arxiv.org/pdf/2506.03916)
- [Pairwise Difference Learning for Classification (arXiv:2406.20031)](https://arxiv.org/html/2406.20031v1)
- [Learning to Defer in Congested Systems: The AI-Human Interplay (arXiv:2402.12237)](https://arxiv.org/abs/2402.12237)
- [TANDEM: Temporal-Aware Neural Detection for Multimodal Hate Speech (arXiv:2601.11178)](https://arxiv.org/html/2601.11178)
