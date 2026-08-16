# CN_VOTE_RECON — can MultiHateClip's discarded `Counter Narrative` votes be stance supervision?

**Date** 2026-08-17 · **Cost** zero API calls, zero GPU, CPU only
**Script** `idea-stage/cn_vote_recon.py` → `idea-stage/cn_vote_recon.json`
**Verdict: NO — STOP at step 3. Neither branch of the frozen bar is cleared, and one of the
two fails in the opposite direction from the hypothesis. Step 4 (discriminator training)
was NOT run.**

---

## 0. Question and frozen decision rule

`idea-stage/STANCE_LIT_RECON.md` §2.2 records that MultiHateClip's released per-annotator
`Label` field contains an undocumented `Counter Narrative` (CN) value used 139 times, which is
discarded when the three-way majority is collapsed to the project's binary label. The question
commissioned here: **is that residue usable as stance supervision for the stance/use-vs-mention
error bucket (`S`) of the round-4 detector?**

**Decision bar — frozen in the commissioning brief before any number below was computed.**
Step 4 (train a logistic-regression-level CN-vs-non-CN discriminator on cached text features,
5-fold CV, report AUC) runs **only if** at least one of:

- **(B1)** among the detector's `S`-bucket false positives, the fraction carrying ≥1 CN vote is **≥ 25 %**; or
- **(B2)** the detector's error rate on CN-voted videos is **significantly higher** than on
  non-CN hateful videos.

Otherwise: stop, report, no training.

**Test-set discipline.** The `S` bucket lives in the test split, so every test-side number below
is a **read-only** overlap count. Nothing was fitted, tuned, or selected on test. Step 4, had it
run, would have used train+val only.

**Sources (all pre-existing, none created here).**
`data/gt/mhc_votes/mhc_{English,Chinese}_{train,valid,test}.tsv` (per-annotator votes, SHA-256'd
official release); `data/gt/{MHC,MHC_zh}/{train,val,test}.jsonl` (project binary labels);
`idea-stage/r5_buckets.json` (manual error buckets); `idea-stage/r4_pilot1.json` (round-4 ensemble
test scores, per-item predictions recovered by the same threshold inversion used in
`idea-stage/r5_xbucket_recon.py`); `GOLD_VOICE` table inside `idea-stage/voice_field_analysis.py`.

Detector = the round-4 best ensemble comparator per dataset (MHC-EN `mean_logit`, MHC-ZH
`logistic`), 3 seeds, prediction = seed majority — identical to the object that produced
`r5_buckets.json`.

---

## 1. Step 1 — how much CN signal actually exists

Per-annotator vote vocabulary across all 2001 MHC videos:
`Normal 2699, Offensive 1082, Hateful 520, Counter Narrative 139, No 1`.

| | EN | ZH | total |
|---|---|---|---|
| videos | 1001 | 1000 | 2001 |
| videos with ≥1 CN vote | 63 | 76 | **139 (6.9 %)** |
| videos with ≥2 CN votes | 0 | 0 | **0** |
| videos where CN is the **majority** label | 0 | 0 | **0** |

Annotators per video: 2 for 1576 videos, 3 for 411, 4 for 14. **Every one of the 139 CN-flagged
videos carries exactly one CN vote** — the maximum over the corpus is 1. So CN is never a majority
and never even a tie; it is always a single dissenting annotator. This is a stronger constraint
than `STANCE_LIT_RECON.md` stated (it reported the majority-label composition, not the per-video
vote multiplicity).

Split and label composition of the 139:

| split | EN CN videos | ZH CN videos |
|---|---|---|
| train | 50 | 54 |
| val | 6 | 8 |
| test | 7 | 14 |

Three-way majority label of the 139: **Normal 90, Hateful 26, Offensive 23** (i.e. 49 videos where
one annotator read counter-narrative and the majority read hate/offensive — matches the earlier
report). Project binary label (positive = Hateful ∪ Offensive): **0 → 75, 1 → 39, not present in
the project splits → 25** (MHC's released annotation subset drops some videos).

**Usable training population for a CN-vs-non-CN discriminator: 104 positives in train
(50 EN + 54 ZH), minus those absent from the project splits — under 100 items, all
single-annotator, against ~1400 negatives.**

---

## 2. Step 2 — contingency: detector errors × CN votes (test, read-only)

Project test splits: MHC-EN 161 items, MHC-ZH 149 items (the vote TSVs cover 200 each; the
remainder are outside the project splits and are excluded).

**Pooled CN × detector-outcome table (both datasets, all test items with a vote row, n = 310):**

| | detector correct | detector error | total |
|---|---|---|---|
| **≥1 CN vote** | 14 | 1 | 15 |
| **no CN vote** | 241 | 54 | 295 |
| total | 255 | 55 | 310 |

Fisher exact two-sided: **OR = 0.319, p = 0.485**. Error rate 6.7 % on CN videos vs 18.3 % on
non-CN videos — CN-voted videos are, if anything, *easier* for the detector, and the difference is
not significant.

Per dataset:

| | MHC-EN | MHC-ZH |
|---|---|---|
| CN correct / CN error | 6 / 1 | 8 / 0 |
| non-CN correct / non-CN error | 124 / 30 | 117 / 24 |

**B1 — the decisive number.** `S`-bucket errors and how many carry a CN vote:

| | S total | S false positives (gold 0, pred 1) | S_FP with ≥1 CN vote |
|---|---|---|---|
| MHC-EN | 16 | 10 | **1** (`ga1r2cweP80`) |
| MHC-ZH | 12 | 5 | **0** |
| **combined** | **28** | **15** | **1 → 6.7 %** |

Over all 28 S-bucket errors (FP + FN) the fraction with a CN vote is **1/28 = 3.6 %**.
Against the frozen bar of ≥ 25 %: **B1 FAILS by a factor of ~4.**

**B2.** Detector error rate on gold-positive (hateful/offensive) test videos:

| | CN-voted positives | error rate | non-CN positives | error rate |
|---|---|---|---|---|
| MHC-EN | 5 | 0.000 | 44 | 0.227 |
| MHC-ZH | 3 | 0.000 | 42 | 0.310 |
| combined | 8 | **0.000** | 86 | **0.267** |

The hypothesis was that CN videos would be *harder*. They are **easier** — 0/8 errors. Direction is
reversed and n = 8 is far too small for significance in either direction. **B2 FAILS.**

---

## 3. Step 3 — CN votes vs the hand-coded `GOLD_VOICE` gold standard

`GOLD_VOICE` (in `idea-stage/voice_field_analysis.py`) hand-codes the utterance source of sampled
S-bucket items as OWN / NOT_OWN / UNDET. 28 of its rows are MHC-EN or MHC-ZH; 19 of those are
determinate (OWN or NOT_OWN) and have a vote row.

| | ≥1 CN vote | no CN vote |
|---|---|---|
| gold **NOT_OWN** (quotation / archive / third party) | **0** | 8 |
| gold **OWN** (uploader's own voice) | **1** | 10 |

The single overlap point is `ga1r2cweP80`, hand-coded **OWN** ("vlogger's own first-person
narration"). So on the only items where the two annotations can be compared, CN votes have **zero**
hits on the NOT_OWN class the stance channel is meant to capture, and their one hit is on the
opposite class. No agreement statistic is meaningful at this n (a κ on a 19-item table with one
positive is noise); reported as counts only.

---

## 4. Step 4 — NOT RUN

Both branches of the frozen bar fail, so by the rule fixed before the analysis, no discriminator
was trained and no AUC is reported. Recording why this is the right call on the merits as well,
not only by the letter of the bar:

1. **No majority signal exists.** Every CN video has exactly one CN vote out of 2–4 annotators.
   There is no video the annotation pool collectively called counter-narrative. A label built from
   a single dissenting annotator is a model of *that annotator*, not of stance.
2. **The residue does not sit where the errors are.** 1 of 15 stance-class false positives, 1 of 28
   stance-class errors. Even a perfect CN detector, used as a suppression rule, would address
   ≤ 1 test error.
3. **It does not agree with our own stance coding.** 0/8 on NOT_OWN.
4. **It is confounded with easiness.** CN-voted test videos have a 6.7 % detector error rate vs
   18.3 % elsewhere; a CN discriminator trained on this would partly learn "easy video", which is
   the opposite of useful.

This closes the loop on `STANCE_LIT_RECON.md`'s own caveat ("139 minority votes is not stance
supervision … use these as evidence that the phenomenon exists, **never** as a training signal").
That caveat is now measured rather than asserted: the overlap with the detector's stance errors is
3.6 %, and the overlap with our stance gold coding is zero.

**What survives.** The 139 votes remain valid *existence* evidence that trained annotators
occasionally read counter-narrative in hateful-video material, and that the released binary label
destroys that reading. They are citable as motivation. They are not a supervision source, at any
scale available in MHC.

---

## 5. Reproduction

```
/home/jehc223/miniconda3/envs/HateVideo/bin/python idea-stage/cn_vote_recon.py
```
Writes `idea-stage/cn_vote_recon.json` (census, contingency, gold-voice rows). Deterministic;
no randomness, no network, no GPU.
