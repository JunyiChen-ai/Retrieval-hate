# ARBITER — results

**Verdict: KILL.** Calling `qwen3-vl-plus` for a binary hateful judgement on the videos where
the frozen head is unconfident, and fusing that judgement back in, loses **−0.0135** test
macro-F1 on average, negative on 3/3 seeds. The premise fails at its own precondition: inside
the head's uncertainty band the MLLM is **less** accurate than the head in 21 of the 24
(seed × split × w) cells measured and exactly equal in the other 3. It is never better.

Design, band grid, fusion rules, final prompt and decision rule were frozen in
`idea-stage/ARBITER_FREEZE.md` (commit `564e70f`) **before** any band-set API call and before
any candidate metric existed. Single submission: 170 videos in one background process, 0
errors, 0 parse retries, no re-run, no top-up.

- Freeze: `idea-stage/ARBITER_FREEZE.md`
- Baseline runner: `idea-stage/arbiter/run_a0.sh`; probability dump `dump_probs.py` →
  `head_probs.json`; logs `logging/runs/arbiter/{run.log,logs/A0_s*.trainlog}`
- MLLM calls: `idea-stage/arbiter/mllm_judge.py` → `judgements.jsonl`;
  log `logging/runs/arbiter/mllm.log`
- Fusion + frozen decision rule: `idea-stage/arbiter/fuse.py` → `fuse_results.json`
- Wall clock: head training 31 s (3 seeds), band API run 105 s, fusion < 1 s.

---

## 1. Baseline — reproduced exactly

The head was never modified, retrained differently, or re-tuned. Same command line as
`desc_channel/run_arms.sh` arm A0, same frozen I1 epoch-selection rule.

| seed | selected epoch | val macro-F1 | test macro-F1 |
|---|---|---|---|
| 0 | 19 | 0.8466 | 0.8817 |
| 1 | 23 | 0.8456 | 0.8771 |
| 2 | 14 | 0.8482 | 0.8735 |
| **mean ± std** | | 0.8468 ± 0.0013 | **0.8774 ± 0.0041** |

Identical to four decimals to `DESC_CHANNEL_RESULT.md` §3 arm A0 and `RGCL_ABLATION_RESULT.md`
§3. All six per-video probability dumps were cross-checked against their trainlog macro-F1 and
matched to < 5e-4, so the probabilities the fusion consumes are provably the same model the
trainlog scored.

## 2. Band and API run

Call set = union over 3 seeds × {val, test} of `|p_head − 0.5| < 0.4` = **170 videos**.

| | n |
|---|---|
| videos called | 170 |
| parsed judgements | **162** (95.3 %) |
| refused by DashScope input moderation (`DataInspectionFailed`) | **8** (4.7 %) |
| transport errors | 0 |
| responses needing the one permitted parse retry | 0 |

Refusals: `hate_video_{114,184,282,340,368,408}`, `non_hate_video_{167,399}` — 6 gold-positive,
2 gold-negative. Each keeps the head's original probability, exactly as frozen.

**Cost (measured tokens).**

| pass | items | input tok | output tok | ¥ at 0.002 / 0.008 per 1 K |
|---|---|---|---|---|
| smoke (train videos, prompt only) | 8 | 17,626 | 126 | 0.036 |
| band run | 170 | 310,202 | 2,564 | 0.641 |
| **total** | | **327,828** | **2,690** | **≈ ¥0.68** |

Against the ¥8 cap. Token counts are measured and exact; DashScope does not report billed
cost, so substitute the real unit price for an exact figure. No thinking tokens were produced
(`enable_thinking=false`; 15–16 completion tokens per call).

## 3. Selection (val only) and the single test readout

Each seed picked its own combination by val macro-F1 over the frozen 4 w × 3 rules. Test
labels were read once, after selection.

| seed | selected w | selected rule | val fused | val baseline | test baseline | test fused | **test Δ** | n videos changed on test |
|---|---|---|---|---|---|---|---|---|
| 0 | 0.1 | a (hard replace) | 0.8375 | 0.8466 | 0.8817 | 0.8678 | **−0.0139** | 6 |
| 1 | 0.1 | a (hard replace) | 0.8375 | 0.8456 | 0.8771 | 0.8724 | **−0.0046** | 5 |
| 2 | 0.3 | c (agree-only) | 0.8494 | 0.8482 | 0.8735 | 0.8517 | **−0.0219** | 92 |
| **mean** | | | | | **0.8774** | **0.8640** | **−0.0135** | |

### Frozen verdict

| clause | requirement | measured | pass? |
|---|---|---|---|
| 1 | `mean(Δ) ≥ +0.005` | **−0.0135** | ✗ |
| 2 | 3/3 seeds `Δ > 0` | 0/3 | ✗ |

### → **KILL**

### Full val grid (selection surface, no test labels involved)

Baselines: s0 0.8466, s1 0.8456, s2 0.8482.

| combo | seed 0 | seed 1 | seed 2 |
|---|---|---|---|
| w0.1 a | 0.8375 (n=4) | **0.8375** (n=5) | 0.8300 (n=4) |
| w0.1 b | 0.8375 | 0.8375 | 0.8300 |
| w0.1 c | **0.8375** | 0.8375 | 0.8300 |
| w0.2 a | 0.8274 (n=7) | 0.8284 (n=7) | 0.8293 (n=8) |
| w0.2 b | 0.8274 | 0.8284 | 0.8202 |
| w0.2 c | 0.8375 | 0.8284 | 0.8300 |
| w0.3 a | 0.8284 (n=14) | 0.8193 (n=14) | 0.8489 (n=54) |
| w0.3 b | 0.8193 | 0.8193 | 0.8397 |
| w0.3 c | 0.8293 | 0.8293 | **0.8494** |
| w0.4 a | 0.8193 (n=20) | 0.8284 (n=17) | 0.8489 (n=58) |
| w0.4 b | 0.8103 | 0.8193 | 0.8397 |
| w0.4 c | 0.8202 | 0.8293 | 0.8494 |

**Design flaw, reported rather than patched.** The frozen grid contains no null arm
(w = 0 / "no fusion"), so every seed is forced to select *some* fusion even when all 12 are
worse than its baseline on val. That is what happened to seeds 0 and 1: their best val cell
(0.8375) is below their baseline (0.8466 / 0.8456), and the selection rule picked the
least-bad combination anyway. Had a null arm been in the grid, seeds 0 and 1 would have
selected it (Δ = 0 by construction, no additional test read) and only seed 2 would have
fused, since it is the one seed whose best val cell (0.8494) actually beats its baseline
(0.8482, by +0.0012). The resulting mean Δ would be (0 + 0 − 0.0219)/3 = **−0.0073**, still
negative, still not 3/3 positive, still **KILL**. The missing null arm changes the size of
the loss, not the verdict. This counterfactual is descriptive and consumes no test label
beyond the one already read under the frozen path.

Also visible in the grid: the single val cell that beats a baseline (seed 2, w0.3 c, +0.0012)
is the one that loses the most on test (−0.0219). Val, at 107 videos, cannot resolve a
difference this small.

## 4. Why it fails — in-band accuracy and error overlap (frozen §7, descriptive)

Restricted to band videos with a usable judgement. `head` = the head's own decision at 0.5,
`mllm` = the MLLM's binary judgement, both scored against gold.

| seed | split | w | n in band | n judged | head acc | MLLM acc | both wrong | head wrong only | MLLM wrong only | both right | error Jaccard |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | val | 0.1 | 4 | 4 | 1.000 | 0.750 | 0 | 0 | 1 | 3 | 0.000 |
| 0 | val | 0.2 | 7 | 7 | 0.857 | 0.571 | 1 | 0 | 2 | 4 | 0.333 |
| 0 | val | 0.3 | 14 | 14 | 0.714 | 0.571 | 2 | 2 | 4 | 6 | 0.250 |
| 0 | val | 0.4 | 20 | 20 | 0.650 | 0.500 | 5 | 2 | 5 | 8 | 0.417 |
| 0 | test | 0.1 | 7 | 6 | 0.833 | 0.333 | 1 | 0 | 3 | 2 | 0.250 |
| 0 | test | 0.2 | 11 | 10 | 0.800 | 0.400 | 2 | 0 | 4 | 4 | 0.333 |
| 0 | test | 0.3 | 19 | 17 | 0.824 | 0.647 | 2 | 1 | 4 | 10 | 0.286 |
| 0 | test | 0.4 | 36 | 32 | 0.781 | 0.688 | 4 | 3 | 6 | 19 | 0.308 |
| 1 | val | 0.1 | 5 | 5 | 0.800 | 0.600 | 0 | 1 | 2 | 2 | 0.000 |
| 1 | val | 0.2 | 7 | 7 | 0.857 | 0.571 | 0 | 1 | 3 | 3 | 0.000 |
| 1 | val | 0.3 | 14 | 14 | 0.786 | 0.571 | 1 | 2 | 5 | 6 | 0.125 |
| 1 | val | 0.4 | 17 | 17 | 0.647 | 0.529 | 3 | 3 | 5 | 6 | 0.273 |
| 1 | test | 0.1 | 5 | 5 | 0.600 | 0.400 | 2 | 0 | 1 | 2 | 0.667 |
| 1 | test | 0.2 | 9 | 8 | 0.750 | 0.500 | 2 | 0 | 2 | 4 | 0.500 |
| 1 | test | 0.3 | 16 | 14 | 0.786 | 0.714 | 2 | 1 | 2 | 9 | 0.400 |
| 1 | test | 0.4 | 30 | 27 | 0.741 | 0.741 | 4 | 3 | 3 | 17 | 0.400 |
| 2 | val | 0.1 | 4 | 4 | 1.000 | 0.500 | 0 | 0 | 2 | 2 | 0.000 |
| 2 | val | 0.2 | 8 | 8 | 0.750 | 0.500 | 1 | 1 | 3 | 3 | 0.200 |
| 2 | val | 0.3 | 55 | 54 | 0.741 | 0.741 | 11 | 3 | 3 | 37 | 0.647 |
| 2 | val | 0.4 | 59 | 58 | 0.759 | 0.759 | 11 | 3 | 3 | 41 | 0.647 |
| 2 | test | 0.1 | 6 | 5 | 0.600 | 0.200 | 2 | 0 | 2 | 1 | 0.500 |
| 2 | test | 0.2 | 16 | 14 | 0.643 | 0.429 | 3 | 2 | 5 | 4 | 0.300 |
| 2 | test | 0.3 | 99 | 92 | 0.837 | 0.772 | 13 | 2 | 8 | 69 | 0.565 |
| 2 | test | 0.4 | 110 | 103 | 0.816 | 0.748 | 15 | 4 | 11 | 73 | 0.500 |

**The MLLM is never more accurate than the head inside the band.** In 21 of 24 cells it is
strictly worse; the 3 ties are seed 1 test w0.4 and seed 2 val w0.3 / w0.4. There is no cell
where it is better. The gap is largest exactly where the head is least confident: at w = 0.1,
where |p_head − 0.5| < 0.1, the head still gets 0.600–1.000 right while the MLLM gets
0.200–0.750. This is the direct refutation of the experiment's precondition — the band does
not contain a population where the MLLM is the better judge.

**The errors are also not independent.** Error Jaccard rises with band width to 0.5–0.65 in
the wide bands, i.e. in the region where enough videos exist to matter, half to two thirds of
the union of the two error sets are videos both get wrong. This is consistent with the
previously measured whole-set agreement (0.94 / 0.43) that motivated the doubt in the freeze.

**Mechanism, pooled over the 162 judged band videos:**

| | value |
|---|---|
| gold positive rate in the band | 0.716 |
| MLLM positive rate | 0.889 |
| MLLM accuracy | 0.753 |
| confusion (gold, MLLM) | (1,1) 110 · (1,0) 6 · (0,1) 34 · (0,0) 12 |

The MLLM calls 34 of the 46 gold-negative band videos hateful — a 74 % false-positive rate on
the negatives it sees — while missing only 6 of 116 positives. It is a high-recall,
low-precision detector, and the band is already 72 % positive, so a rule that hands it the
decision converts most of the head's surviving negatives into false positives. Fusion rules
(a) and (b) are near-identical in effect because the model's `confidence` is saturated:
159 / 162 judgements report ≥ 0.95, and the full histogram is
{0.6: 1, 0.7: 1, 0.85: 1, 0.95: 36, 0.98: 88, 0.99: 9, 1.0: 26}. The stated confidence carries
essentially no information about whether the judgement is right.

## 5. 与现有工作的区分

Evidence base: `research-wiki/MLLM_USAGE_LANDSCAPE.md` (2026-07-02, mechanisms verified
against paper text and official code).

| axis | HVGuard (EMNLP 2025) | RAMF (TMLR) | MARS / LELA / IARE / TANDEM | Filter-And-Refine (TikTok, 2507.17204) | this experiment |
|---|---|---|---|---|---|
| what the MLLM produces | 3-step CoT rationale | 3 rationale views (T_O/T_H/T_N) | judgement + rationale | Yes/No moderation token | binary hateful/not + confidence |
| how the output is consumed | rationale embedded as a 4th modality feature, MoE fusion | rationale embedded as a 4th modality feature | fused / debated | is the ranker's decision | fused with the head's probability, **decision level only, no feature, no retraining** |
| who gets the call | every video | every video | every video (LELA: every frame × modality) | every video hits the router, 2.5 % reach the MLLM | **only videos whose head probability is inside a frozen uncertainty band** |
| what decides the call | nothing (always-on) | nothing | nothing | embedding similarity to a hand-picked high-risk seed bank | **the downstream classifier's own output probability** |
| why the gate exists | — | — | — | compute cost | test whether MLLM errors are independent of head errors where the head is unsure |
| does the base detector change | trained jointly with rationale features | trained with rationale features | — | — | **not at all — frozen weights, frozen features, frozen epoch selection** |

The distinction is real and, per the landscape file, unoccupied: on academic hateful-video
benchmarks (HateMM / MultiHateClip / ImpliHateVid) every reasoning-VLM method is always-on,
and the only prior gate in this literature — Filter-And-Refine — routes on *similarity to a
seed bank* to save GPU time, so its survivors are the high-risk items rather than the
*uncertain* ones. Confidence-gated deferral to an MLLM, consumed only inside the band, had not
been measured here before.

**And it does not pay.** The honest statement required by the freeze: **inside the band the
MLLM is not more accurate than the head** — 21 of 24 cells strictly worse, 3 ties, 0 better —
so there is no version of the fusion rule that could have helped. The gate is not the part
that failed; the gate did its job and selected the head's genuinely hard cases. What failed is
the assumption that a strong general-purpose MLLM asked a direct binary hate question is a
better judge than a small trained head on that dataset's own hard cases. It is not: it is a
high-recall, low-precision detector whose stated confidence is uninformative, and whose errors
overlap the head's at Jaccard 0.5–0.65 in the wide bands.

This also closes the last untested consumption mode for the MLLM's *judgement* in this project:
whole-set five-class stance, masked stance and binary-choice stance all failed
(`STANCE_PILOT_RESULT.md`, `MASK_STANCE_PILOT_RESULT.md`, `CONTRAST_STANCE_RESULT.md`), the
*generated description* as a feature channel failed (`DESC_CHANNEL_RESULT.md`,
`A0_OCR_E2E_RESULT.md`), and the judgement restricted to the uncertainty band fails here.

## 6. What this kills and what it does not

**Killed.**
- Uncertainty-gated MLLM deferral with `qwen3-vl-plus` on HateMM, in all three frozen fusion
  forms (hard replace / probability average / agreement-only), on the `LORA/HateMM/L1/I1`
  backbone, at every band width in {0.1, 0.2, 0.3, 0.4}.
- The associated novelty story. Since the MLLM is *less* accurate than the head inside the
  band, the gate cannot be defended as the contribution even if some other fusion rule were
  tried — the ceiling of any decision-level fusion is set by the in-band accuracies in §4.

**Not killed** (stated so the kill is not over-read).
- **A different judge.** One MLLM, one prompt, one temperature. The in-band accuracies are a
  property of `qwen3-vl-plus` answering a direct binary hate question, not of MLLMs in general.
  Anything that would rescue this direction has to first beat 0.65–0.84 in-band head accuracy,
  which is a measurable precondition that costs ~¥0.7 to check for any new judge.
- **A different question.** The MLLM was asked the target label directly. Asking it something
  it is better at — a perceptual or evidentiary sub-question whose answer the head cannot
  compute — is a different experiment; §4 says nothing about it.
- **Confidence gating in general** as a compute-saving device. Nothing here contradicts
  Filter-And-Refine-style routing when the goal is throughput rather than accuracy; the
  measurement here is only that this gate does not buy accuracy.
- **The 8 moderation refusals** (4.7 % of the band, 6 of them gold-positive) are an operational
  finding worth carrying: input moderation on this vendor removes hateful items preferentially,
  and the rate is nearly double the 2.7 % measured over the full 1066-video set in
  `DESC_CHANNEL_RESULT.md`, consistent with the band being 72 % positive.
