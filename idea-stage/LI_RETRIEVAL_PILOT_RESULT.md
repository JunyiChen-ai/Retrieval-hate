# Late-interaction segment-level retrieval pilot — results

Run 2026-08-09. Arms, key construction, similarity, endpoints, splits and the decision rule were
frozen **before** any candidate number was computed, in `idea-stage/LI_RETRIEVAL_PILOT_FREEZE.md`.
Single submission: four arms x three splits x both endpoints in one process; no re-run, no tuning
after seeing numbers.

- Script: `idea-stage/li_retrieval_pilot.py`
- Log: `logging/runs/li_retrieval_pilot/run.log` (PID file `run.pid`); smoke log
  `logging/runs/li_retrieval_pilot/smoke_permuted.log`
- Raw results: `idea-stage/li_retrieval_pilot.json`
- New derived cache (reusable): `data/OCR/HateMM/pilot_ocr_window_vecs.npz` — the 6565 unique OCR
  window texts and their 768-d CLIP text-tower vectors, which `pilot_ocr_blocks.npz` did not store.
- Data boundary: HateMM-train only, 744 videos (298 hateful). `dev_seen` and `test` never opened.
  `ocr_windows_K30.jsonl` SHA-256 verified against `data/OCR/SHA256SUMS.json` at load.
- Total wall time: 6 s for the scored run (it reused the window-vector cache built during the
  smoke), 16 s end-to-end including the one-off GPU pass for the CLIP text tower.

## → **NO-GO**

`arm2 - arm0` is negative on **both** frozen endpoints, on **every** split, with a bootstrap CI
that excludes zero. Late interaction over retained segments does not beat the whole-video key; it
is measurably worse, and adding OCR to the segment key makes it worse again.

---

## Arms (recap)

Similarity is the **sum of per-block cosines** (dot product of concatenated, individually
L2-normalized blocks); a missing block is the frozen zero vector and contributes exactly 0.

| arm | name | key(s) | `S(q,m)` |
|---|---|---|---|
| 0 | baseline (current method) | 1 whole-video key `[l2(img) ‖ l2(txt)]` | `cos_img + cos_txt` |
| 0v | whole-video visual only *(control)* | 1 key `l2(img)` | `cos_img` |
| 1 | LI-visual | 30 keys `l2(S[v,k])` | `mean_k max_j cos_vis` |
| 2 | LI-visual+OCR | 30 keys `[l2(S[v,k]) ‖ l2(O[v,k])]` | `mean_k max_j (cos_vis + cos_ocr)` |

OCR coverage in the segment key: **9645/22320 windows (43.2%) are empty** after the frozen
`conf>=0.5, len>=2` filter and carry the zero block; **150/744 videos (20.2%) have no OCR anywhere**
and are therefore scored identically in arm 1 and arm 2.

---

## Endpoint 1 (mechanism) — neighbour label purity @ k=10

Split 0 (frozen Gate-0 folds). Per-query chance = the fold memory's same-label fraction;
mean chance 0.5198 across all arms.

| arm | purity@10 | lift over chance | lift CI95 (2000 boot) |
|---|---|---|---|
| **0 baseline** | **0.6984** | **+0.1786** | [+0.1576, +0.1989] |
| 0v whole-video visual | 0.6827 | +0.1629 | [+0.1437, +0.1813] |
| 1 LI-visual | 0.6638 | +0.1441 | [+0.1258, +0.1624] |
| **2 LI-visual+OCR** | **0.6472** | **+0.1274** | [+0.1095, +0.1452] |

Paired per-video differences on split 0 (same bootstrap draws for every arm):

| contrast | Δ purity | CI95 |
|---|---|---|
| **2 − 0 (gating)** | **−0.0512** | **[−0.0679, −0.0345]** |
| 0v − 0 (transcript cost) | −0.0157 | [−0.0305, +0.0001] |
| 1 − 0v (multi-segment retention) | −0.0188 | [−0.0293, −0.0090] |
| 2 − 1 (OCR into the key) | −0.0167 | [−0.0258, −0.0078] |
| 1 − 0 | −0.0345 | [−0.0513, −0.0177] |

The ordering `0 > 0v > 1 > 2` reproduces on splits 1 and 2 (purities 0.6930/0.6794/0.6632/0.6448
and 0.6956/0.6829/0.6634/0.6469), so it is not a property of the Gate-0 fold assignment.

## Endpoint 2 (performance) — kNN classification macro-F1, 744 OOF

k=10, similarity-weighted vote, threshold selected per fold on memory-side leave-one-out only.

| arm | split 0 (gate0) | split 1 (skf 20260901) | split 2 (skf 20260902) | **mean** |
|---|---|---|---|---|
| **0 baseline** | 0.7961 | 0.7709 | 0.7709 | **0.7793** |
| 0v whole-video visual | 0.7596 | 0.7398 | 0.7504 | **0.7499** |
| 1 LI-visual | 0.7547 | 0.7517 | 0.7563 | **0.7542** |
| **2 LI-visual+OCR** | 0.7479 | 0.7255 | 0.7353 | **0.7362** |

Split-paired deltas:

| contrast | per split | mean |
|---|---|---|
| **2 − 0 (gating)** | −0.0482, −0.0454, −0.0357 | **−0.0431** |
| 0v − 0 (transcript cost) | −0.0365, −0.0310, −0.0205 | −0.0293 |
| 1 − 0v (multi-segment retention) | −0.0049, +0.0119, +0.0059 | +0.0043 |
| 2 − 1 (OCR into the key) | −0.0068, −0.0262, −0.0211 | −0.0180 |
| 1 − 0 | −0.0414, −0.0191, −0.0146 | −0.0250 |

## Verdict against the frozen rule (transcribed unedited from the script)

- **Criterion A (performance)**: needs mean `F1_arm2 − F1_arm0 >= +0.005` and positive on all 3
  splits. Observed **−0.0431**, negative on 3/3. **Not met.**
- **Criterion B (mechanism)**: needs split-0 purity `Δ >= +0.020` with bootstrap 95% LB `> 0`.
  Observed **−0.0512**, CI95 [−0.0679, −0.0345]. **Not met.**
- **NO-GO clause**: mean F1 delta `<= 0` **and** split-0 purity delta `<= 0`. Both hold.

## → **NO-GO**

---

## Attribution decomposition

The headline −0.0431 macro-F1 / −0.0512 purity splits into three parts:

| step | Δ macro-F1 (mean of 3 splits) | Δ purity@10 (split 0) | reading |
|---|---|---|---|
| dropping the transcript (`0v − 0`) | −0.0293 | −0.0157 [−0.0305, +0.0001] | the transcript block is carrying most of arm 0's advantage |
| **multi-segment retention** (`1 − 0v`) | **+0.0043** (signs −,+,+) | **−0.0188** [−0.0293, −0.0090] | at best a wash on classification, a **reliable loss** on the mechanism endpoint |
| **OCR into the segment key** (`2 − 1`) | **−0.0180** (signs −,−,−) | **−0.0167** [−0.0258, −0.0078] | actively harmful, consistently |

Two things follow, and they are the substantive content of this pilot.

**1. "Retaining all 30 segments" is not the missing ingredient.** With the transcript confound
removed (`arm1` vs `arm0v`, both visual-only), MaxSim over 30 retained segments is worse than the
mean-pooled whole-video key on neighbour purity by −0.019 with a CI excluding zero, and is
indistinguishable on macro-F1 (+0.004, sign flips across splits). The pilot's motivating hypothesis
— that mean pooling is destroying retrievable structure that late interaction would recover — is
not supported. Whatever mean pooling destroys, MaxSim does not recover it, and MaxSim gives up
something of its own.

**2. Putting OCR in the retrieval key is worse than putting it in the classifier input.** The OCR
fusion pilot got **+0.0094** macro-F1 from mean-pooled OCR concatenated to a *classifier* input.
The same embeddings, same filter, bit-identical vectors (the self-check below), used as an
*additive term inside the retrieval similarity*, cost **−0.0180** macro-F1 and **−0.0167** purity.
The I5 COMPLEMENTARY finding (`ov@10 = 0.048` vs chance 0.017) says the OCR neighbourhood is
different from the transcript neighbourhood; this pilot says *different is not better* when the
difference enters through a `max`.

## Interpretation (labelled: not measured here)

The most likely mechanism for both losses is that `max_j` is an extreme-order statistic over
~17850 memory segments. For each query segment it returns the single best match anywhere in the
memory, so the score is dominated by whichever memory segments are generic enough to be everyone's
nearest neighbour — title cards, black/near-black frames, talking-head crops, and on the OCR side
the watermarks, channel handles and UI chrome that recur across unrelated videos. That predicts
exactly the observed pattern: the visual `max` costs a little, and the OCR `max` — over a channel
where 43% of windows are empty and much of the remainder is boilerplate — costs more. This is a
hypothesis consistent with the numbers, **not** something this pilot measured; the diagnostic that
would settle it is a hubness count (how concentrated the `argmax_j` distribution is over memory
segments, and whether the hub segments are boilerplate). It is deliberately not run here, because
this pilot's numbers are a single frozen submission.

## Reconciliation with P2 forensic H4 (important — H4 does not contradict this)

P2 forensic H4 reported segment kNN purity lift **+0.181** against whole-video **+0.138** and
concluded that "cutting the granularity finer improved label purity". That is not in conflict with
the result above, because the two numbers measure different objects, and reading H4 as a statement
about *video retrieval* is a mistake this pilot rules out.

Reading `idea-stage/p2_forensic2.py` lines 124-144: H4's `segment_mean_top20_purity` is, for each
video, the average over its **30 independent segment queries** of the parent-label purity of each
query's top-20 **segment** neighbours. Those 30 retrievals are never combined; no ranked list of
*videos* is ever produced. Its comparator `whole_video_top20_purity` uses `l2(W_img)` — visual
only, i.e. this pilot's **arm 0v**, not arm 0 (H4 +0.138 @ k=20 vs this pilot's arm0v +0.163 @
k=10; the gap is the k and the chance convention).

So H4 establishes that *a segment query is a purer query than a whole-video query*. This pilot
asks the next question — whether that survives aggregation into one video-level ranking — and
answers **no**: arm 1 (+0.144) sits below arm 0v (+0.163), CI on the paired difference excluding
zero. The segment-level advantage does not survive MaxSim. Under the frozen sum-of-cosines /
MaxSim rule the correct summary is: **segment queries are purer, segment-keyed video retrieval is
not.** Any future route citing H4 must cite this bound alongside it.

## Validity checks

- **OCR embedding identity.** Re-aggregating the newly built per-window vectors under the OCR
  fusion pilot's own rule reproduces `pilot_ocr_blocks.npz` `o3` and `o30` with `max|Δ| = 0.0`
  (exactly, both blocks). The two pilots are provably in the same embedding space; arm 2's OCR
  block is the same object the +0.0094 fusion number was computed from.
- **Null control.** The label-permuted smoke (real features, shuffled labels, full pipeline) gives
  purity 0.5220 / 0.5138 / 0.5130 / 0.5222 for arms 0/0v/1/2 against chance ~0.520 — lift ≈ 0
  everywhere — and macro-F1 0.496 / 0.487 / 0.471 / 0.474. The pipeline has no leakage path that
  manufactures purity or accuracy from nothing.
- **Determinism.** No model training, no random init, no sampling in the estimates; the only RNG is
  the purity bootstrap (seed 20260903). Seed-consistency is therefore carried by three independent
  5-fold splits rather than model seeds, as registered.
- **Tie handling.** Neighbours are ordered by `(-similarity, video_id)` with a lexicographic id
  tie-break, so no result can be produced by cache-position ordering (P2 forensic H5's failure).
- **Red lines.** Zero test/`dev_seen` contact (id guard + HALT); decision rule frozen first;
  implementation smoke-tested only on synthetic and label-permuted data; real numbers submitted
  once.

## Caveats, stated against this result rather than for it

1. **This bounds one aggregator, not late interaction as a family.** `mean_k max_j` with equal
   weight on all 30 query segments and no learned projection is the cheapest possible late
   interaction. A learned linear map into the interaction space, a soft aggregator (log-sum-exp,
   top-m mean instead of top-1), or query-segment weighting are not measured here and are not
   excluded by this result. What *is* excluded is the free-lunch version: the frozen-feature,
   untrained MaxSim drop-in does not beat the whole-video key, so any late-interaction route now
   has to pay for a trained component and justify that cost.
2. **The OCR weight is fixed at 1.0.** A weight sweep could make arm 2 approach arm 1 from below,
   but arm 2 → arm 1 is the *ceiling* of that sweep (weight 0 = arm 1), and arm 1 already loses to
   arm 0. Down-weighting OCR cannot rescue the gating contrast; no sweep was run and none is
   needed.
3. **The classifier is retrieval-only.** macro-F1 here (arm 0 = 0.779) is a pure kNN read-out, well
   below the ~0.81 linear-head figure in the OCR fusion pilot and the ~0.82 A0 figure. That is
   expected and intended — the endpoint has to be *sensitive to the retrieval key*, which a strong
   trained head would mask. The deltas are the meaningful quantity, not the absolute level.
4. **The AMBIGUOUS band of the frozen rule is weak.** Applied to the permuted-label null the rule
   returns AMBIGUOUS, because the null's purity delta happened to land at +0.0001 > 0 while the F1
   delta was negative, and the NO-GO clause requires both to be ≤ 0. This does not affect the
   present verdict — the real result is negative on both endpoints with margin — but a future
   pilot reusing this rule should require the NO-GO clause on either endpoint, not both.

## What this licenses

- **Close the "mean pooling is the culprit, late interaction is the fix" line as stated.** The
  proposition was tested at its cheapest and most direct form and came back negative on both a
  mechanism and a performance endpoint, consistently across three splits.
- **Do not put OCR into a retrieval key with a `max` aggregator.** OCR's measured value in this
  project remains where the OCR fusion pilot found it: a small, sub-threshold (+0.0094), positive
  contribution as a *classifier input*. This pilot adds a bound in the other direction, and the two
  together say the useful OCR operator is an average into the input, not a max into the metric.
- **Correct the standing reading of P2 H4.** "Segment retrieval is more label-pure than whole-video
  retrieval" is true only for segment-level queries evaluated segment-wise. As a video retrieval
  claim it is now falsified. Anything downstream that inherited the +0.181-vs-+0.138 framing needs
  this qualification.
- It does **not** license a claim that late interaction cannot work with trained representations,
  nor any statement about `dev_seen`/`test`, nor any change to a previously frozen verdict.
