# R9 — candidate slate (2026-08-17/18)

Round 9 of idea discovery. New hard constraint from the principal: **no human annotation may be
introduced** (any candidate requiring new human labels is out of scope this round). Standing
constraints unchanged: no new datasets, four datasets only, method paper with a real gain,
API ≤ ¥15 this round (cumulative ¥0 of ¥60), local RTX 5090 only, the four red lines, and the
new measurement protocol (MHC ≥30 seeds / HateMM ≥15 seeds, dev macro-F1 epoch selection,
paired bootstrap CI, ≥ +0.005 with CI excluding zero).

---

## 0. Zero-cost diagnostics run before any candidate was scored

Code `idea-stage/r9_diag/diag.py`, raw `idea-stage/r9_diag/{d1,d2}.json`,
logs `logging/runs/r9_diag/`. Both are descriptive; no candidate arm is scored in either.

### D1 — encoder adaptation is the only intervention in this project that moves ≥9 test items

Arms are two **already-deployed** encoders from the contrast-line table (frozen Qwen2.5-VL-7B vs
its own LoRA), 15 seeds each, `r4_harness` protocol (val macro-F1 epoch selection). Test labels
were read for the error-set comparison; declared as a disclosed diagnostic (same standing as
`IDEA_REPORT` §10.6). Nothing was selected or tuned on them.

| dataset | frozen | adapted | Δ | frozen errors | adapted errors | **fixed** | **broken** | Jaccard | prob corr |
|---|---|---|---|---|---|---|---|---|---|
| HateMM (LoRA) | 0.8575 | 0.8744 | +0.0169 | 30 | 25 | **11** | **6** | 0.53 | 0.952 |
| HateMM (LoRA-curric) | 0.8575 | 0.8696 | +0.0121 | 30 | 27 | **12** | **9** | 0.46 | 0.927 |
| MHC-EN (LoRA) | 0.7281 | 0.7220 | −0.0062 | 37 | 38 | **9** | **10** | 0.60 | 0.841 |
| MHC-ZH (LoRA) | 0.7681 | 0.8013 | +0.0332 | 29 | 24 | **9** | **4** | 0.61 | 0.934 |

Three facts follow.

1. **The error populations of the frozen and the adapted encoder genuinely differ.** Overlap is
   18-28 items against 3.5-8.7 expected under independence, so they are strongly correlated — but
   9-12 items per dataset flip in each direction. For comparison, every head-level / objective-level
   / fusion-level mechanism searched in rounds 6-8 moved 0-2 items.
2. **Adaptation is not monotone: it breaks 4-10 items it previously got right.** This "adaptation
   forgetting" is not a hypothetical oracle over unknown information — it is a loss the project
   already pays, on items the *same model family* already answered correctly before adaptation.
   Its size (6 / 9 / 10 / 4 net items) is +2.4 to +5.5 macro-F1 points, comparable to the S-bucket
   prize (+3.23 recoverable) and, unlike it, blocked on neither money nor annotation.
3. **MHC-EN is where adaptation costs more than it buys** (9 fixed, 10 broken, net −0.0062), and it
   is also the smallest train split (549). The forgetting is worst exactly where n is smallest.

**Caveat carried forward (F66/AGGNET law):** a large oracle is a precondition every failed
candidate has already met. The 6/9/10/4 figure is a *headroom*, not evidence for any candidate.
Also, suppressing the breaks per test item at inference is per-item selection and is banned by
Law III; only a training-time mechanism is legal here.

### D2 — the "labels contradict visible content" population is 2-4 % of train, and it is not noise-shaped

Frozen Qwen features, 5-fold stratified CV over train+val, 5 seeds, out-of-fold probabilities,
no test contact.

| dataset | n | OOF macro-F1 | OOF ROC | wrong | conf-wrong > 0.9 | > 0.8 | > 0.7 | positives among conf-wrong |
|---|---|---|---|---|---|---|---|---|
| HateMM | 851 | 0.8503 | 0.9102 | 123 | 34 (4.0 %) | 62 | 84 | 65 % |
| MHC-EN | 629 | 0.7864 | 0.8549 | 117 | 19 (3.0 %) | 25 | 30 | **100 %** |
| MHC-ZH | 657 | 0.7990 | 0.8849 | 116 | 12 (1.8 %) | 25 | 54 | **100 %** |
| ImpliHateVid | 1608 | 0.9254 | 0.9761 | 120 | 45 (2.8 %) | 64 | 79 | 42 % |

The S-bucket repricing (`S_PRIZE_DECOMP.md`) found that 21 of 49 stance errors are items where an
independent content-only panel unanimously reads the opposite of gold. If that were a *label*
property with train-side mass, a label-noise / robust-loss family would have something to attack.
D2 prices that family: at most 1.8-4.0 % of train items are confidently contradicted by the model,
and on both MultiHateClip splits **100 % of them are positives the model calls normal** — hard
positives, not symmetric annotation noise. Any small-loss-pruning / robust-loss mechanism is
therefore bounded by a 12-45 item train-side population per dataset.

### D3 — restatement of the S-prize with the circularity flagged (no run)

`S_PRIZE_DECOMP.md` defines "recoverable" as *the four-judge content-only panel agrees with gold*,
and prices the recoverable oracle at +3.23 mean macro-F1. The `CLAUDE_STANCE_GATE` result is that
the same panel scores 18/32 on stance rows — a tie with the constant-DISTANCED baseline. The
recoverable subset is therefore **defined by the panel's agreement with gold**, so +3.23 is a
pricing figure for *human* annotation, not evidence that a content-only method can reach those
items. Recorded here because round 9's brief asked what mechanism could carry "information beyond
the content"; see §3.

---

## 1. Candidate table

Composite = 0.3·premise + 0.3·novelty + 0.3·expected gain + 0.1·cost, scored 0-10 before external
review (self-scored; the hostile external review is §2).

| # | candidate | one-line mechanism | premise | novelty | gain | cost | comp |
|---|---|---|---|---|---|---|---|
| R9-A | **ANCHOR-INT** | interpolate the frozen and the LoRA-adapted feature of the *same* model, α on val — a zero-GPU proxy for anchored adaptation | D1 (4-10 broken items) | low (WiSE-FT family) | med | free | — |
| R9-B | **ANCHOR-TRAIN** | retrain the LoRA with an anchor term (L2-SP / feature distillation to the frozen encoder) so adaptation cannot move items the frozen model already gets right | D1 | low-med | med | 2-6 GPU-h + re-extraction | — |
| R9-C | **DONOHARM** | adaptation objective = labels on items the frozen head gets wrong + frozen-model ordering preserved on the rest | D1 | med | med | as B | — |
| R9-D | **PREFIX-PEFT** | adapt only a soft prompt (~10⁴ params) instead of LoRA, on the hypothesis that forgetting scales with adapted parameter count and n=549 is the failure point | D1 (MHC-EN worst, smallest n) | low | low-med | 2-4 GPU-h | — |
| R9-E | **ACT-STEER** | inject a train-derived difference-in-means direction into an intermediate layer during extraction (intervene in the computation instead of the read-out) | none measured | med | unknown | 1 extraction pass per arm | — |
| R9-F | **RLOSS** | robust-loss / small-loss pruning family screen (GCE, label smoothing, confidence pruning) | D2 bounds it to 12-45 items | ~0 | low | free | — |
| R9-G | **FEATMIX** | mixup / manifold augmentation in frozen feature space (R8 C8 FROFA, never piloted) | none | ~0 | low | free | — |
| R9-H | **DIAR-PROV** | speaker/source provenance from off-the-shelf diarization as a use-vs-mention carrier the transcript lacks | S bucket, but audio family is 0/4 and segments are closed | med | unknown | ~1 GPU-h gate | — |
| R9-I | **SS-CONSIST** | self-supervised consistency across frame samplings as an adaptation regulariser (label-free) | none | low | low | 2-4 GPU-h | — |
| R9-J | **XDATA-GUIDE** | factor the label into shared hatefulness + dataset-specific guideline offset, trained across the four splits | S_PRIZE §4 (guideline mismatch is the contested-item cause) | med | med-high | free | **policy-blocked**: `banned_constraints[8]` |
| R9-K | **AUX-LABELFIELD** | multi-task on `Target_Victim` / `Component` / 3-way severity already in the datasets | real | low | low | free | **policy-blocked**: `banned_constraints[1]` |
| R9-L | **STANCE-FUND** | buy the 750 stance judgements | best measured | — | +3.23 (panel-defined) | ¥ + labour | **out of scope this round** (no human annotation) |

---

## 2. External review

See §2 of this file after the review is folded in.

---

## 3. The brief's direct question — what can carry "information beyond the content"?

Round 9's brief asked: the S-bucket repricing says gold on 21 of 49 stance errors depends on
information outside the visible content; what mechanism can carry that signal, given that MLLM
judgement as training signal is banned (`banned_constraints[5]`), human annotation is banned this
round, and there is no metadata channel (`IDEA_REPORT` §10.9.1 — title is already inside the
encoder's text on both MultiHateClip splits and does not exist on HateMM/ImpliHateVid)?

The honest enumeration of carriers is:

1. **New human judgements** — banned this round, and the only route the project has priced.
2. **A model's judgement** — banned (`banned_constraints[5]`), and measured negative at all five
   MLLM access points.
3. **Another corpus's labels** — banned (`banned_constraints[8]`), and no new datasets.
4. **Platform / uploader / community metadata** — does not exist on any of the four datasets.
5. **The dataset's own train labels** — the only remaining carrier, and it is already fully
   consumed by the head. The contested-item examples in `S_PRIZE_DECOMP` §4 are *labelling
   convention* differences (ImpliHateVid counts objectification as hate; MHC annotators reject a
   speaker's disclaimer), i.e. properties of the labelling function of that same dataset. Learning
   them better is not a new information channel; it is sample efficiency.

**Conclusion recorded before scoring:** the S-bucket repricing does not open a new mechanism family
under round-9 constraints. It sharpens a *pricing* question for the principal (fund annotation, or
lift `banned_constraints[8]`), and it removes the "information beyond content" framing as a source
of candidates.
