# P2 forensic memo — why segment-keyed retrieval purity landed below chance

**Date** 2026-08-09 · **Scope** HateMM-train only (744 videos), zero `dev_seen` / `test` contact ·
**Type** diagnosis, not method development. Nothing was repaired and no repaired metric is reported
as a result.

**Artifacts.** `idea-stage/p2_forensic.py`, `idea-stage/p2_forensic2.py` (analysis scripts),
`idea-stage/p2_forensic.json`, `idea-stage/p2_forensic2.json` (all numbers below),
`idea-stage/p2_forensic_cache.npz` (the 744×30 purity matrix),
`logging/runs/p2_forensic/run.log`, `run2.log`.
The frozen P2 selector was re-implemented from `idea-stage/pilots.py` and reproduces
`hit_rate = 0.5436241610738255` bit-for-bit, so everything here is measured on the same object the
pilot measured.

**Headline.** The anomaly does not exist. The reported 0.544-vs-chance-0.762 is not a retrieval
result; it is an `np.argmax` tie-breaking artifact interacting with a positional prior in HateMM's
gold spans. Under a random tie-break the same selector scores **0.768** against chance **0.762** —
exactly at chance, not below it. What survives the forensic is a different and cleaner negative:
the neighbourhood-purity statistic carries **no within-video localization signal at all**
(within-video AUROC 0.511, CI [0.488, 0.533]) while carrying strong **video-level** label signal
(AUROC 0.782). The frozen NO-GO verdict is unaffected — it stands under every tie-break rule.

---

## H5 (not in the original list, and it dominates) — argmax tie-break degeneracy

**Judgment: SUPPORTED, decisively. This is the cause of the "below chance" number.**

`pilots.py:175` — `jstar[i] = int(np.argmax(mem_lab[nb].mean(axis=1)))`. With K=20 neighbours,
`p_j` can only take 21 values. Over the fold memory (~595 videos × 30 = ~17.8k segments) the
statistic saturates:

| quantity (298 hateful eval videos) | value |
|---|---|
| median number of **distinct** `p_j` values across the 30 segments | **2** |
| median tie multiplicity at the argmax | **16 of 30** |
| videos with a unique argmax | 24.2% |
| videos with `max_j p_j` exactly 1.0 | 52.3% |

`np.argmax` returns the **lowest index**, so ties resolve to the earliest segment. Result:
**51.3%** of hateful videos select segment k=0 and **63.4%** select k≤2. (Over all 744: 37.5% at
k=0. The artifact is class-correlated — 51.3% of hateful vs 28.3% of non-hateful videos land on
k=0.)

Meanwhile HateMM gold spans have a strong positional prior — annotators exclude intros and outros:

| position k | 0 | 1 | 2 | 10–19 (mean) | 27 | 28 | 29 |
|---|---|---|---|---|---|---|---|
| P(segment in gold span) | **0.339** | 0.527 | 0.681 | **0.816** | 0.674 | 0.594 | **0.379** |

So a selector that resolves ties to index 0 is forced onto the single position least likely to be
inside a span. Counterfactual audit of the identical `p_j` matrix, varying only the tie-break:

| selector | hit rate | vs chance 0.762 |
|---|---|---|
| uniform random segment | 0.762 (sd 0.014) | — |
| **frozen `np.argmax`, first index** | **0.544** | −0.218 |
| random tie-break | **0.768** (sd 0.014) | **+0.006** |
| last-index tie-break | 0.527 | −0.235 |
| subset with a *unique* argmax (n=72) | 0.681 | chance for that subset = 0.697 |
| subset with `p_max < 1.0` (n=142), random tie-break | 0.717 | chance for that subset = 0.712 |

Tie-break-free (exact expectation over the tied set, all 298 videos): hit 0.768, chance 0.762,
paired lift **+0.005, CI95 [−0.024, +0.035]**.

Two consequences.

1. "Segment-keyed retrieval lands *below* chance" is false and should not be repeated. The
   selector is *at* chance. There is no anti-correlation with evidence to explain.
2. The frozen decision is nevertheless untouched: the bar was `hit ≥ 2× chance and ≥ 0.35 with
   LB > chance`, NO-GO if `< 1.3× chance`. Under random tie-break the ratio is 1.008. **NO-GO
   under every tie-break rule.** Nothing needs to be procedurally reopened.

A third consequence for metric 2: the block appended in the classification arm was the video's
**opening segment** for 51% of hateful and 28% of non-hateful videos, so metric 2 measured
"whole-video features + opening frames", not "whole-video features + a purity-selected evidence
segment". It never tested its intended construct either (see the closing note on metric 2).

---

## H1 — benign segments dominate, diluting label purity

**Judgment: REFUTED as stated. The real finding is stronger and different: there is no
segment-level structure in `p_j` to dilute.**

H1 predicts that in-span segments have higher neighbourhood purity than out-of-span segments, with
the argmax drowned by the benign majority. Measured over the 249 hateful videos whose gold mask is
mixed (not all-30 / none):

| quantity | value |
|---|---|
| mean `p_j` for segments **inside** gold span | 0.6369 |
| mean `p_j` for segments **outside** gold span | 0.6247 |
| difference (in − out) | **+0.0121, CI95 [−0.0065, +0.0306]** |
| within-video AUROC of `p_j` predicting gold membership | **0.5114, CI95 [0.4878, 0.5333]** |

Both are indistinguishable from null. The positional profile makes it unmissable: `p_j` averaged
over hateful videos is flat at 0.62–0.65 across **all 30 positions** (min 0.624, max 0.653) while
gold membership swings from 0.339 to 0.859 over the same positions; correlation across positions
**−0.074**.

The neighbourhood composition confirms it from the memory side. For the selected segment, 79.5% of
its top-20 neighbours have a hateful parent, and of those hateful-parent neighbours 76.6% sit
inside *their own* gold span — but the unconditional base rate for a hateful memory segment being
inside its own span is 76.2%, and the average over all 30 query segments is 74.2%. The neighbours
are no more "on evidence" than random hateful segments.

What `p_j` *does* encode is the parent video's label, not the segment's role:

| quantity | value |
|---|---|
| mean `p_j` over segments, hateful videos | 0.640 |
| mean `p_j` over segments, non-hateful videos | 0.259 |
| memory hateful-segment fraction (null) | 0.401 |
| AUROC of **video label** from mean `p_j` | **0.782** |
| AUROC of **within-video gold membership** from `p_j` | **0.511** |

So a frozen CLIP visual segment key retrieves *videos that look like this video* — production
style, channel, format — and that is a strong video-level label cue and a useless within-video
evidence cue. This is the substantive null P2 actually produced, and it is not an artifact.

---

## H2 — segment features are visual-only, the failures need OCR/speech

**Judgment: NOT SUPPORTED, and underpowered. No differential is detectable.**

Gate-C adjudicated census rows (`logging/runs/gate_c_annotation/`, c1 overwritten by adj) intersect
the 298-video eval set on **103** videos. Stratified by `required_modalities`, using the
tie-break-free hit (exact expectation over the tied set) and the tie-break-immune within-video
AUROC:

| stratum | n | within-video AUROC | diff vs complement (CI95) | tie-break-free hit − chance | diff vs complement (CI95) |
|---|---|---|---|---|---|
| `on_screen_text` required | 49 | 0.523 | **−0.001 [−0.098, +0.097]** | −0.004 | **−0.083 [−0.197, +0.039]** |
| `on_screen_text` not required | 54 | 0.524 | — | +0.079 | — |
| speech (`transcript`/`audio`) required | 62 | 0.506 | **−0.043 [−0.141, +0.056]** | +0.016 | **−0.059 [−0.181, +0.056]** |
| speech not required | 41 | 0.549 | — | +0.075 | — |

Every interval straddles zero. Only **one** census video in the eval set is `visual`-only, so the
clean contrast H2 asks for cannot be formed at all.

Read honestly: H2 is not refuted, it is **untestable at this n**. With 49 vs 54 videos the design
could only detect an AUROC gap of roughly ≥0.10, and the point estimates are 0.001 and 0.043. But
note the shape of the null — the within-video AUROC is ≈0.51–0.55 in *every* stratum, including
the videos whose evidence the coders judged to be visual/on-screen. Even where the modality is
present in the feature, the segment key does not find the evidence. A modality argument therefore
cannot by itself explain H1's flat profile.

---

## H3 — official gold spans are so coarse that "hit rate" is a contaminated metric

**Judgment: PARTIALLY SUPPORTED — the metric is genuinely contaminated, but contamination is not
what produced the P2 number, and the sharper target does not rescue the idea.**

Two separate contaminations exist and should not be conflated.

*Coverage.* Official spans cover mean 0.717 / median 0.829 of the video, so top-1 chance is 0.762
and the metric has almost no dynamic range. On the 99 videos where blinded coders also gave
minimal sufficient intervals, coverage drops to 0.361.

*Position.* Documented under H5 — the metric additionally rewards mid-video selection irrespective
of evidence. Any selector with a positional bias is scored on its bias.

Re-scored on the 99 videos that have coder minimal intervals:

| target | selector | hit | chance | lift (CI95) |
|---|---|---|---|---|
| official spans, all 298 | tie-break-free | 0.768 | 0.762 | +0.005 [−0.024, +0.035] |
| official spans, the 99 | frozen argmax | 0.646 | 0.696 | — |
| coder minimal intervals, the 99 | frozen argmax | 0.444 | 0.361 | — |
| coder minimal intervals, the 99 | **tie-break-free** | **0.410** | **0.361** | **+0.050 [−0.009, +0.109]** |
| coder minimal intervals, the 99 | uniform random segment control | 0.363 | 0.361 | — |
| within-video AUROC vs coder minimal intervals | — | **0.544** | — | CI95 [0.496, 0.593] |

So the *sign* of the reported effect is target-dependent: ratio 0.71 against official spans with
the frozen tie-break, ratio 1.14 against coder minimal intervals with the tie-break removed. That
is a real indictment of the metric. But the corrected effect is +0.050 with a CI touching zero, and
the corrected AUROC is 0.544 with a CI touching 0.5. **The idea is not rescued by the sharper
target; it is moved from "significantly below chance" to "indistinguishable from chance".**

---

## H4 — segment vectors are noisy, whole-video pooling has better SNR

**Judgment: REFUTED. Segment-level retrieval is *more* label-pure than whole-video retrieval.**

Class separation (mean within-class cosine minus mean between-class cosine, L2-normalised):

| representation | within | between | separation |
|---|---|---|---|
| whole-video visual | 0.4015 | 0.3738 | **0.0277** |
| single segment (one random segment per video) | 0.3313 | 0.3080 | **0.0233** |
| mean-pooled normalised segments | 0.3982 | 0.3708 | **0.0274** |

Segments are ~16% less separated than whole-video vectors — real, but far too small to explain a
0.22-point hit-rate gap, and it disappears under pooling.

Retrieval label purity (top-20, fraction of neighbours matching the query's own label; the class
prior gives 0.520 for random neighbours):

| retrieval | purity | lift over prior |
|---|---|---|
| whole-video visual kNN | 0.658 | +0.138 |
| segment kNN, averaged over the 30 segments | **0.701** | **+0.181** |
| segment kNN, max over the 30 segments | 0.854 | — |

The direction is opposite to H4. Cutting the granularity finer *improved* label purity. What it
did not do — and this is the whole story — is make the purity *segment-specific*.

---

## Note on metric 2 (classification, −0.59 pt)

Reported as `macro-F1 0.8231 → 0.8172, −0.59 pt, bootstrap CI95 [−2.16, +0.99]`. The interval
contains zero. **No harm was demonstrated**; the correct description is a null, and the report's
phrase "costs 0.59 macro-F1 points" overstates it. Combined with H5 — the appended 1024-d block was
the opening segment for 51% of hateful and 28% of non-hateful videos — metric 2 is best read as:
concatenating a class-correlated positional artifact to an already-saturated 1792-d feature vector
changed nothing measurable. It is not evidence about purity-selected segments, in either direction.

---

## Implications for the mechanism-first route

Ordered by strength of the evidence behind each statement.

**1. (Strongest — deterministic, reproducible.) Discard the claim that segment-keyed retrieval is
anti-correlated with evidence.** There is no anomaly to build a paper around, and no "the retriever
actively avoids evidence" story. Any narrative, in `IDEA_REPORT.md` §0/§3/§4 or downstream, that
rests on "0.544 < 0.762 chance" must be corrected to "at chance (0.768 vs 0.762)". The NO-GO
verdict itself does not move.

**2. (Very strong — n=249, CI [0.488, 0.533].) Mechanisms that localize evidence by the label
purity of a frozen-CLIP-visual segment key's neighbourhood have no signal to work with.** The
statistic is a video-level style/channel detector (AUROC 0.782 for the video label) and a coin
flip within a video (AUROC 0.511 official, 0.544 coder-minimal). Any mechanism — hard top-1, hard
top-K, differentiable top-K, MIL, purity-guided attention — whose *selection score* is derived from
retrieved-neighbour labels under these features inherits that 0.51. The mechanism cannot fix a
signal that is absent from its input statistic. This kills the purity-closed-loop family (I4) on
mechanism grounds, independently of the buggy pilot number.

**3. (Very strong — median 2 distinct values across 30 segments.) Any selection criterion built on
a small-support discrete count is degenerate by construction and must be treated as a design red
flag, not a hyperparameter.** With K=20 neighbours the score has 21 levels; 52% of hateful videos
saturate at 1.0 across half their segments. Selection then reduces to whatever the tie-break is —
here, video position. Concretely: a mechanism route is *not* viable if its selection score is a
bounded vote/count over a memory this size, unless it also specifies a non-arbitrary tie
resolution. Continuous, non-saturating scores (margins, similarity-weighted quantities, calibrated
probabilities) do not have this failure mode. This is the one transferable lesson and it applies to
every remaining discrete-selection idea in the portfolio, including I3.

**4. (Strong — direction reversed with tight numbers.) Do not build a mechanism whose premise is
"segment features are too noisy, so pool/denoise them".** H4 is refuted: segment retrieval is more
label-pure than whole-video retrieval (+0.181 vs +0.138 lift). Denoising/pooling branches address a
problem that is not there.

**5. (Moderate.) Localization metrics on HateMM official spans are close to unusable and must not
be the primary target of any mechanism claim.** Chance is 0.762; the positional prior alone spans
0.34→0.86, so a metric difference can be manufactured by a selector's positional bias with zero
evidence-finding ability. Anything claiming localization on this dataset needs (a) coder minimal
sufficient intervals or an equivalently sharp target, (b) an explicit uniform-random-selector
control run on the same videos, (c) a declared tie-break policy, and (d) a rank-based statistic
(within-video AUROC) alongside top-1 hit, since AUROC is immune to both contaminations. This is a
constraint on evaluation design, and it applies whatever mechanism is chosen.

**6. (Weak — underpowered, 49 vs 54.) The modality-gap explanation for P2 is neither supported nor
excluded here.** The Gate-C `on_screen_text` enrichment (OR 2.29) that justified unblocking the OCR
channel is a statement about *classification failures*, and this forensic finds no corresponding
differential in *segment retrieval* (AUROC diff −0.001 [−0.098, +0.097]). The OCR decision does not
depend on P2 and is not disturbed by it — but P2 gives it no support either, and a route may not
cite P2 as evidence that adding OCR would have fixed the selector. Notably the within-video AUROC
is ≈0.51–0.55 in every stratum including visually-evidenced videos, so adding a modality is not on
its own a reason to expect within-video localization to appear.

**What would still have a chance, stated only as what the evidence permits.** A mechanism is not
ruled out by this forensic if its selection signal is (i) continuous and non-saturating, (ii) not
derived from retrieved-neighbour *labels* under frozen visual segment features, and (iii) evaluated
against a sharp evidence target with a random-selector control. What is ruled out is the specific
combination P2 tested — discrete argmax over a neighbour-label vote on frozen CLIP visual segment
keys — and, more generally, any route that assumes retrieval neighbourhoods contain within-video
evidence information under these features. On the current measurements they do not.

---

### Red-line compliance

HateMM-train only (744 videos); `dev_seen` and `test` were never opened. Gate-C census rows were
used for audit stratification only (H2, H3) and enter no training or inference path. No frozen
verdict is revised by this memo; the P2 NO-GO stands. Nothing was repaired — every corrected number
above is a re-scoring of the frozen selector's own `p_j` matrix under a different tie-break or a
different evaluation target, reported as diagnosis.
