# EXP: P10 — amplify the MLLM localization role (calibrate on HateMM spans, test once on HateClipSeg)

> **Status: PRE-REGISTERED (design frozen before the HateClipSeg test pass).**
> P6 (`research-wiki/EXP_p6_mllm_localization.md`) is the campaign's one positive MLLM
> method-role: reading frames+ASR, the MLLM localizes hate WITHIN videos significantly better
> than the CLIP-visual memory scorer (HateClipSeg within-video AUC 0.5435, CI [0.533,0.554],
> p=5.4e-8; paired over memory +0.030, p=0.007) — but MODEST. With all nine accuracy routes
> refuted, P10 asks whether the localization gain can be made **substantial** and honest, by
> tuning the scorer on a **calibration dataset** and testing the single promoted config **once**
> on the held-out P6 harness.

## Two-dataset protocol (this is what makes free iteration legal)

- **CALIBRATION = HateMM gold `hate_snippet` spans** (`data/gt/HateMM/hate_spans.json`; 427
  hateful videos, 391 with both-class seconds; span median 32s, ~46% coverage — non-trivial;
  provenance `EVAL_localization_hatemm.md` §1). A P6-style within-video localization eval
  (`scripts/analysis/p10_eval_hatemm.py`): windows → 1-fps seconds (label = second-midpoint in a
  gold span), within-video AUC over hateful videos with both-class seconds + AP-hateonly + a
  random control. **On this set I iterate freely and exhaustively** (window K, prompt wording,
  few-shot exemplars, ASR-weighting/aggregation, scorer model) — **every config is logged**
  (no silent shopping). No HateClipSeg contact.
- **HELD-OUT TEST = HateClipSeg**, ONE pass with the single promoted config on the frozen P6
  harness (`p6_eval_localization.py`; same 395-video split, same within-video AUC + CI + AP,
  same controls incl. the memory-scores row and random).

## Anchor (compute FIRST — sets the bar)

The frozen P6 config = the P3 scorer (`score_segments_mllm.py`, Qwen2.5-VL-7B, frames + K-window
ASR → integer 0–3), K=30/M=120. Its HateMM-calibration within-video AUC is the **anchor**; the
promotion bar is stated relative to it. (Data point already in hand: the same scorer at K=4 gives
HateMM wv-AUC 0.5478, CI [0.533,0.563], p=3.9e-8, n=389.)

**ANCHOR (P6 config, K=30, HateMM TRAIN hateful): wv-AUC = 0.5387** (CI [0.5244, 0.5534],
sign-p 5.6e-11, n=266 partial-coverage videos; random 0.494; job 12474).
→ **promotion bar = paired wv-AUC ≥ 0.5387 + 0.04 = 0.5787, CI(Δ) excluding 0.**

> **Calibration-set note:** a comma in the `SPLITS=train,val,test` sbatch `--export` value collided
> with the `--export` comma separator, so all calibration scoring collapsed to `SPLITS=train`. This
> is benign and if anything cleaner — the calibration set is the **HateMM train hateful videos**
> (298 scored, 266 both-class), a large labeled span set with **zero** val/test/HateClipSeg contact.
> All configs are compared on this identical train set (apples-to-apples paired deltas).

## Iteration grid (HateMM calibration — logged, cheap→expensive)

CPU-only (re-aggregate existing scores, no re-scoring):
- **A-gate**: zero-weight windows with no speech (localization is speech-borne per P6 mechanism).
- **A-lex**: weight each window score by an ASR hate-lexicon hit count (`HateClipSeg/lexicons.json`
  is EN; used read-only as a generic cue, NOT a HateClipSeg label).
- **A-fuse**: combine K=4 + K=30 anchor scores (coarse×fine).

GPU (one scoring pass each; `p10_score_segments.py`, prompt/model variants):
- **K60**: K=60/M=120 — finer localization windows.
- **fewshot**: K=30 + in-context 0–3 rating exemplars in the prompt.
- **speech**: K=30 + a speech-focused prompt (rate the SPOKEN hate in this window).
- **32B**: K=30 + Qwen2.5-VL-32B-Instruct (bf16, 1×A100-80G) — stronger scorer, only if a 7B
  variant is close to the bar.

## Pre-registered promotion bar

A config is promoted to the single HateClipSeg test **iff**, on the HateMM calibration set, its
paired within-video AUC beats the **anchor** (P6 config) by **≥ +0.04** with the paired
bootstrap 95% CI **excluding 0**. If several clear, the highest paired Δ is promoted. **If none
clears, P10 dies calibration-side — the HateClipSeg test is never touched and P6 stands as-is.**

## Pre-registered substantial bar (the goal's bar, on the HateClipSeg test)

- **HateClipSeg wv-AUC ≥ 0.60** (vs P6's 0.5435, memory 0.514) = **clear success** — substantial,
  novel MLLM localization role.
- **0.56 ≤ wv-AUC < 0.60** with CI excluding P6's 0.5435 = **modest amplification** (honest
  report; user decides if it's enough).
- **wv-AUC < 0.56** = amplification did not transfer; **P6 stands as-is**.

One test touch total. HateClipSeg controls (memory row, random) recomputed unchanged.

## Hard rules

SLURM only (no `--time`, `HF_HUB_OFFLINE=1`, `WANDB_MODE=disabled`), foreground `sacct` polling;
calibration scoring uses hateful-only gt (`data/gt_p10hate/`, 427 vids) to bound GPU; ASR re-binned
on CPU from the stored word timestamps (`p10_rebin_asr.py`, no Whisper re-run). No `.pt` in git;
32B cache deleted after use; quota watch. Report the full HateMM calibration leaderboard BEFORE
the test pass.

---

## HateMM CALIBRATION LEADERBOARD

Run 2026-07-08. All configs scored on the **HateMM train hateful** calibration set (298 scored,
**n=266** both-class videos); eval `scripts/analysis/p10_eval_hatemm.py` (within-video AUC primary,
paired bootstrap 10k 95% CI on the per-video Δ vs anchor over the common video set, sign-test).
Anchor reproduces the pre-registered number bit-for-bit (0.5387). Random control wv 0.4940 for all.
Machine JSON: `scripts/analysis/loc_out/p10_hatemm_leaderboard.jsonl`.

| variant | source | K | HateMM wv-AUC | paired Δ vs anchor | paired Δ 95% CI | AP-hateonly | promoted? |
|---|---|---|---|---|---|---|---|
| **anchor** (P6 cfg, Qwen-7B, frames+ASR→0–3) | 12474 | 30 | **0.5387** | — (bar: ≥+0.04) | — | 0.7321 | — |
| A-gate (zero no-speech windows) | CPU re-agg | 30 | 0.5314 | −0.0074 | [−0.0195, +0.0045] | 0.7127 | no |
| K60 (K=60/M=120, finer windows) | 12475 | 60 | 0.5319 | −0.0068 | [−0.0156, +0.0019] | 0.7295 | no |
| fewshot (K=30 + 0–3 exemplars) | 12476 | 30 | 0.5359 | −0.0028 | [−0.0090, +0.0034] | 0.7291 | no |
| A-lex (ASR hate-lexicon weight) | CPU re-agg | 30 | 0.5450 | +0.0062 | [−0.0000, +0.0123] | 0.7345 | no |
| **A-fuse** (K4×K30 coarse×fine) | CPU re-agg | 30 | **0.5693** | **+0.0305** | **[+0.0175, +0.0437]** | 0.7441 | **no (Δ<+0.04)** |

**Reading of the bar (Δ ≥ +0.04 AND paired Δ CI excluding 0):**
- The two **GPU** scoring variants (K60, fewshot) and A-gate all land **at or slightly below** the
  anchor (paired Δ negative, CI straddling 0) — finer windows, in-context exemplars, and
  no-speech gating do not amplify localization; if anything they dilute it.
- A-lex nudges up (+0.0062) but its CI touches 0 and Δ is 6× short of the bar.
- **A-fuse is the single closest config**: it *does* clear the CI-excludes-0 half of the bar
  (paired Δ +0.0305, CI [+0.0175, +0.0437], sign-p 7e-7) — a real, significant improvement over the
  anchor — but its magnitude **+0.0305 is below the pre-registered +0.04 threshold** (equivalently
  0.5693 < 0.5787). Per the frozen bar it is **not promoted**. The bar was not moved to admit it.

### VERDICT vs the pre-registered promotion bar — **FAIL (no promotion)**

No config reaches paired wv-AUC ≥ 0.5787 with the paired Δ CI excluding 0. The best amplifier
(A-fuse) is significant but only +0.0305 over anchor — modest, below the +0.04 gate. **P10 dies
calibration-side: the HateClipSeg held-out test is NEVER touched, and P6 stands as-is** (HateClipSeg
within-video AUC 0.5435, CI [0.533, 0.554], p=5.4e-8; paired over memory +0.030, p=0.007 — a modest
but statistically solid MLLM localizer). The localization gain does not amplify to *substantial* on
the calibration set, so there is no honest basis to spend the single HateClipSeg test pass.

**Bottom line:** P10 = FAIL / no promotion. The MLLM-localization role remains at its P6 magnitude
(modest, significant). MLLM's earned roles across the campaign stay: encoder + localizer +
guard-rail/audit — no substantial amplification of the localizer at Qwen-7B, no HateClipSeg test
consumed. Coarse×fine fusion (A-fuse) is the only lever that even moved the needle significantly on
calibration (+0.03) and is the natural starting point if this is ever revisited with a stronger
scorer — but under the frozen P10 protocol it does not clear the bar.

## HateClipSeg TEST (promoted config only)

**Not run.** No config cleared the calibration-side promotion bar, so per the pre-registration the
single HateClipSeg test pass was never spent. P6's HateClipSeg result (wv-AUC 0.5435) stands as the
final localization number.

---

# P10-b — scale-ladder second calibration round (32B / 72B scorer × A-fuse aggregation)

> **Status: PRE-REGISTERED (design frozen before the round-2 evaluation).** This is the **last
> in-register path** of the whole MLLM-method-role campaign (`novelty-scope-and-plan.md`): does a
> **stronger localization scorer** amplify the P6 localizer past the pre-registered substantial bar?
> Round 1 (above, commit 7194ee2) FAILed: the anchor (Qwen2.5-VL-7B, K=30) HateMM-calibration
> wv-AUC = **0.5387**; the single significant lever was **A-fuse** (K4×K30 coarse×fine)
> wv-AUC 0.5693, **paired Δ +0.0305, CI [+0.0175, +0.0437]** — real but below the **+0.04** gate.
> P10-b climbs the scorer up the Qwen2.5-VL scale ladder (7B→32B→72B), pairs it with the round-1
> winning aggregation (A-fuse), and re-tests the **unchanged** promotion bar. No bar was moved.

## Frozen candidate list (exactly 5 rows — no additions)

Calibration set, harness, anchor, and paired protocol are **identical to round 1** (HateMM
train-hateful, 298 scored / n=266 both-class; `p10_eval_hatemm.py`; paired bootstrap 10k 95% CI on
the per-video Δ vs the frozen 7B anchor over the common video set; sign-test). Each stronger scorer
is scored at **K=30 (fine)** and **K=4 (coarse)** so its A-fuse channel is the SAME model
(matching the 7B A-fuse recipe exactly; no cross-model fusion). "Anchor aggregation" = the raw K=30
scores evaluated directly.

| # | candidate | scorer | aggregation | source |
|---|---|---|---|---|
| R2-1 | 32B · anchor-agg | Qwen2.5-VL-32B (bf16) | raw K=30 | GPU (`p10_score_ladder.sbatch`) |
| R2-2 | 32B · A-fuse | Qwen2.5-VL-32B (bf16) | 0.5·K30 + 0.5·K4(map), same model | GPU + CPU re-agg |
| R2-3 | 72B · anchor-agg | Qwen2.5-VL-72B (bnb4-nf4) | raw K=30 | GPU (if feasible) |
| R2-4 | 72B · A-fuse | Qwen2.5-VL-72B (bnb4-nf4) | 0.5·K30 + 0.5·K4(map), same model | GPU + CPU re-agg |
| R2-5 | 7B · A-fuse×A-lex | Qwen2.5-VL-7B (round-1 scores) | 0.5·K30 + 0.5·K4(map) + min(lex_hits,3) | **CPU only, no GPU** |

R2-5 is a pure re-aggregation stacking the two round-1 winners (A-fuse coarse×fine + A-lex ASR
hate-lexicon boost) on the already-landed 7B scores; it is explicitly a **multiple-comparison
extension** (`p10_aggregate_b.py --mode fuselex7b`). The other four are the {32B,72B}×{anchor,A-fuse}
scale grid the brief fixes.

## Infrastructure inventory & feasibility (measured, not guessed)

- **Anchor rate (measured):** 7B, 298 vids × 30 windows = 8,940 multimodal gens in **52 min**
  (job 12474), ≈ 0.34 s/gen.
- **32B-VL — FEASIBLE.** `Qwen/Qwen2.5-VL-32B-Instruct` = **68.3 GB** (bf16), fits 1×A100-80G
  (device_map=auto; 8-token gen → negligible KV). Est single-job wall ≈ **3–4 h** (K30) + ≈ 0.5 h
  (K4). Transformers 4.49 loads it via `Qwen2_5_VLForConditionalGeneration`. Downloaded on the
  login node (P2c pattern), offline in-job.
- **72B-VL — memory-feasible via bnb4; disk/throughput risk documented.** `Qwen/Qwen2.5-VL-72B-
  Instruct` bf16 = **146.8 GB** → **exceeds 80 G VRAM**, so it cannot run bf16 on 1 GPU. Load path =
  on-the-fly **4-bit nf4 (double-quant, bf16 compute)** — the exact P2c 72B recipe (`--quant bnb4`,
  ~40 G resident on 1×A100-80G, zero env mutation). Two residual risks: (a) the **147 GB checkpoint
  download** lands on a **shared /data at 97 % full (≈552 G free)**; (b) bnb4 multimodal throughput
  → est single-job wall ≈ **9–13 h**. Plan: attempt the 72B download; **if it completes within the
  disk budget, submit R2-3/R2-4; if the download/disk fails, downgrade to 32B-only and record R2-3/
  R2-4 as "not run — 72B infeasible" here.** No bar change either way.

## Promotion bar — UNCHANGED from round 1 (not moved to admit any round-2 config)

A candidate is promoted to the single HateClipSeg test **iff**, on the HateMM calibration set, its
**paired wv-AUC beats the frozen 7B anchor (0.5387) by ≥ +0.04** (equivalently wv-AUC ≥ **0.5787**)
**AND** the paired bootstrap 95% CI on the per-video Δ **excludes 0**. If several clear, the single
**highest paired Δ** is promoted. If none clears, **P10-b = FAIL, the HateClipSeg test is never
touched, and P6 stands as-is** (wv-AUC 0.5435).

## Multiple-comparison / round accounting (reported honestly)

This is the **second** calibration round. Configs compared against the frozen anchor so far:
**round 1 = 5** (A-gate, K60, fewshot, A-lex, A-fuse) + **round 2 = 5** (R2-1..R2-5) = **10 total**.
The +0.04 gate was **not** loosened for round 2, and round-2 positives (if any) are reported as
second-round, 10-comparison results. The A-fuse magnitude (+0.0305) is the round-1 reference the
scale ladder must exceed.

## Substantial bar on the HateClipSeg test (unchanged; one touch total)

Promoted config only, single pass, frozen P6 harness (`p6_eval_localization.py`, 395-video split,
within-video AUC + CI + memory/random controls): **wv-AUC ≥ 0.60 = substantial** / **0.56–0.60 with
CI excluding P6's 0.5435 = modest** / **< 0.56 = P6 stands**.

## Hard rules (carry over)

SLURM only (no `--time`, `HF_HUB_OFFLINE=1`, `WANDB_MODE=disabled`), foreground `sacct` polling (no
background waiter); calibration scoring uses hateful-only gt (`data/gt_p10hate/`) at K=30 and K=4;
no `.pt` in git; 32B/72B caches **deleted after use**; quota watch on the shared FS. Report the full
two-round leaderboard **before** any test pass.

## P10-b HateMM CALIBRATION LEADERBOARD

Run 2026-07-08/09. Same calibration set and harness as round 1 (HateMM train hateful, 298 scored,
**n=266** both-class; `p10_eval_hatemm.py`, paired bootstrap 10k 95% CI vs the frozen 7B anchor).
Jobs: 32B = 12562 (K30 complete; K4 OOM) + 12570 (K4/M16 done after fixes; 12568 = intermediate
fail); 72B = 12571 (bnb4-nf4, 4h04). Two execution bugs were hit and fixed **before** any round-2
number was computed (commits c5c47ee, e69065f): (a) 32B bf16 OOM on the coarse pass —
`expandable_segments` + per-video `empty_cache()`, both score-neutral; (b) the ladder's coarse pass
initially passed M=120; the round-1 recipe (P3-default `train_segscoreK4_qwen.jsonl`) is **K=4/M=16**,
so the pass was corrected to M=16 (also removes the OOM: 4-frame windows). Scoring health uniform
across models (1 undecodable video; parse fallbacks ≤0.35%). Machine rows appended to
`scripts/analysis/loc_out/p10_hatemm_leaderboard.jsonl`.

Full two-round table (11 comparisons vs the anchor; bar: paired Δ ≥ +0.04 AND CI(Δ) excl. 0):

| round | variant | HateMM wv-AUC | paired Δ vs anchor | paired Δ 95% CI | clears bar |
|---|---|---|---|---|---|
| — | **anchor** (7B, raw K30) | 0.5387 | — | — | — |
| 1 | A-gate | 0.5314 | −0.0074 | [−0.0195, +0.0045] | no |
| 1 | K60 | 0.5319 | −0.0068 | [−0.0156, +0.0019] | no |
| 1 | fewshot | 0.5359 | −0.0028 | [−0.0090, +0.0034] | no |
| 1 | A-lex | 0.5450 | +0.0062 | [−0.0000, +0.0123] | no |
| 1 | A-fuse (7B) | 0.5693 | +0.0305 | [+0.0175, +0.0437] | no (Δ<+0.04) |
| 2 | R2-5 · 7B A-fuse×A-lex (CPU) | 0.5752 | +0.0365 | [+0.0223, +0.0506] | no (Δ<+0.04) |
| 2 | R2-1 · 32B anchor-agg | 0.5512 | +0.0125 | [−0.0006, +0.0257] | no |
| 2 | R2-2 · 32B A-fuse | 0.5825 | +0.0437 | [+0.0240, +0.0631] | **yes** |
| 2 | R2-3 · 72B anchor-agg | 0.5593 | +0.0206 | [+0.0065, +0.0347] | no (Δ<+0.04) |
| 2 | **R2-4 · 72B A-fuse** | **0.5913** | **+0.0526** | **[+0.0333, +0.0721]** | **yes — highest Δ, PROMOTED** |

Two clean gradients separate on the calibration set:
- **Raw scorer scale alone does not clear the bar.** Anchor aggregation improves monotonically
  7B 0.5387 → 32B 0.5512 → 72B 0.5593, but even the 72B's Δ (+0.0206) is half the gate.
- **A-fuse × scale is the lever.** The coarse×fine fusion gains grow with the scorer:
  7B +0.0305 → 32B +0.0437 → 72B +0.0526. Both 32B and 72B A-fuse clear the unchanged bar;
  per the frozen rule (highest paired Δ) **R2-4 (72B A-fuse) is the single promoted config**.
- R2-5 (stacking the two round-1 CPU winners on 7B scores) lands at +0.0365 — the best 7B-only
  number, still short of the gate: the missing ingredient was scorer strength, not aggregation.

## P10-b HateClipSeg TEST (single pass, promoted R2-4)

Run 2026-07-09. Scoring job 12585 (72B bnb4, HateClipSeg test 395 videos, K30/M120 + K4/M16,
5h50; K4 ASR re-binned on CPU from the stored chunk timestamps, no Whisper re-run). Fuse on CPU →
`test_seen_segscoreK30_p10-p6-72b-bnb4-fuse.jsonl`; eval = the **frozen P6 harness**
(`p6_eval_localization.py --mllm_tag p10-p6-72b-bnb4-fuse`, same 395-video split, same estimators;
harness integrity pre-verified by reproducing the published P6 numbers bit-for-bit with the default
tag). Controls (memory row, random) recomputed **unchanged** — they reproduce P6 exactly. Machine
JSON: `loc_out_hcs/results_p10b_test.json`, paired stats `loc_out_hcs/p10b_test_paired.json`.

| condition | frame-full AP / AUC | seg AP / AUC | **within-video AUC** |
|---|---|---|---|
| a — memory `knn_hatemm_subclip` | 0.5329 / 0.5754 | 0.5246 / 0.5839 | 0.5140 |
| **b — R2-4 (72B A-fuse, promoted)** | 0.5929 / 0.6488 | 0.5948 / 0.6561 | **0.5755** |
| d — random | 0.4699 / 0.5084 | 0.4507 / 0.5065 | 0.5088 |
| e — broadcast of b's video mean | 0.6198 / 0.6598 | 0.6002 / 0.6595 | 0.5000\* |
| *(P6 reference: b at 7B)* | 0.5421 / 0.6034 | 0.5599 / 0.6353 | 0.5435 |

\* by construction. c (rank-avg with memory) = 0.5578 — again dilutes the MLLM; fusion with the
weaker memory scorer still hurts.

**Within-video significance (primary):** R2-4 wv-AUC **0.5755**, bootstrap 95% CI
**[0.5581, 0.5933]**, sign-p 1.4e-9 (n=329 both-class videos).
- **paired vs memory** (0.5140): Δ **+0.0615**, CI **[+0.0359, +0.0869]**, sign-p 4.9e-5.
- **paired vs P6's 7B scorer** (0.5435): Δ **+0.0319**, CI **[+0.0170, +0.0474]**, sign-p 0.0024 —
  the calibration-side promise (+0.0526 over the 7B anchor on HateMM) transfers at ~60% strength.

### VERDICT vs the pre-registered three-band test bar — **MODEST amplification**

- **wv-AUC ≥ 0.60 (substantial): NOT met** (0.5755 < 0.60).
- **0.56 ≤ wv-AUC < 0.60 with CI excluding P6's 0.5435: MET** — 0.5755 ∈ [0.56, 0.60) and the CI
  lower bound 0.5581 > 0.5435 (and the paired-vs-P6 CI excludes 0). Second-round /
  11-comparison caveat stated as pre-registered.
- Supporting metrics move the same way: frame AUC 0.6034→0.6488, segment AUC 0.6353→0.6561 vs P6;
  the broadcast control (e) remains the best pooled AP (0.6198), so video-level density is still
  the MLLM's dominant competence — but the within-video increment is now larger and CI-separated
  from the P6 baseline.

**Bottom line: P10-b = MODEST amplification, honestly reported.** The scale ladder × coarse×fine
fusion (Qwen2.5-VL-72B, A-fuse) lifts the MLLM localizer from P6's wv-AUC 0.5435 to **0.5755**
(CI [0.5581, 0.5933]; paired over P6 +0.0319, over memory +0.0615, both CIs excluding 0) on the
held-out HateClipSeg test — a real, statistically solid improvement that **does not reach the
0.60 substantial bar**. The campaign's MLLM localization role is upgraded from "modest (7B)" to
"modest-plus (72B A-fuse)": the earned-roles verdict (encoder + localizer + guard-rail/audit, no
main-table accuracy role) is **unchanged in kind, strengthened in degree**. The single HateClipSeg
test touch is now **spent**; P10-b closes the last in-register path of the MLLM-method-role
campaign. Whether "modest-plus" is worth the 72B inference cost is a framing decision for the
user, not a statistical one.

---

# P10-c — new-generation open-VLM scorer (third calibration round; OPEN execution of Kit-B)

> **Status: PRE-REGISTERED (design frozen before any HateMM scoring or aggregation).** This is the
> **open-source** execution of `OPTION_KITS_terminus.md` Kit-B (the closed-API P10-c draft): instead
> of GPT-5/Gemini/Claude over a third-party API (data-exfiltration + non-reproducible), we run a
> **new-generation OPEN VLM (non-Qwen2.5-VL) fully on-cluster** as the localization scorer. No data
> leaves the node; the result is reproducible and open-pipeline-legal. P10-b climbed the *same-family*
> scale ladder (Qwen2.5-VL 7B→32B→72B) and promoted 72B A-fuse (calib wv-AUC **0.5913** → test
> **0.5755 MODEST**). P10-c asks the one remaining question the reaggregation ceiling (0.5932, commit
> 93e82fa) leaves open: **does a genuinely stronger, newer scorer generation clear the substantial
> gate?** Only a stronger scorer can — aggregation is exhausted.

## Reconnaissance (measured, not guessed) — candidate new-gen open VLMs

HF metadata pulled 2026-07-09 (`huggingface.co` reachable from the login node). Env constraint: the
HateVideo env (transformers **4.49**) is **not** touched; Qwen3-VL needs the native
`Qwen3VL*ForConditionalGeneration` classes (config `transformers_version` 4.57.0.dev0), so a
**separate cloned env `HateVideoVLM`** was built (`conda create --clone HateVideo` → `pip install -U
transformers` → **transformers 4.57.6**, torch **2.6.0+cu124** and decord/PyAV **preserved** by the
clone; Qwen3VL classes + video backends import-verified). ms-swift's `transformers<4.50` pin conflict
is benign (ms-swift is not on the scoring path).

| candidate | HF repo | size (bf16) | load plan (A100-80G) | throughput est | wallclock est (calib) | key risk |
|---|---|---|---|---|---|---|
| **C1 Qwen3-VL-32B (dense)** | `Qwen/Qwen3-VL-32B-Instruct` | 66.7 GB / 14 shards | bf16, 1 GPU, `expandable_segments` (~13 GB headroom — proven for 32B in P10-b) | ~0.35–0.5 s/gen (dense-32B class) | K30 (8,940 gens) ≈ 3–4.5 h + K4 (1,192) ≈ 0.5 h | processor video-kwarg adaptation (necessary eng.) |
| **C2 Qwen3-VL-30B-A3B (MoE)** | `Qwen/Qwen3-VL-30B-A3B-Instruct` | 62.1 GB / 13 shards | bf16, 1 GPU | fast (3B active) ~0.15–0.3 s/gen | K30 ≈ 1.5–2.5 h + K4 ≈ 0.3 h | MoE routing; same HF path as C1 |

**Rejected candidates (recorded, so the choice is auditable):**
- `Qwen/Qwen3-VL-235B-A22B-Instruct` — 471 GB bf16 (96 shards); on-the-fly bnb4 still needs the full
  bf16 download (infeasible on a shared /data at 97 %, 548 G free), and its FP8 release cannot run
  natively on **A100 (Ampere = no FP8)**. **Infeasible.**
- `OpenGVLab/InternVL3_5-38B` (76.8 GB) / `OpenGVLab/InternVL3-78B` (156.8 GB) / `InternVL3_5-30B-A3B`
  (61.7 GB) — all use the **custom `InternVLChatModel`** (`model_type=internvl_chat`,
  `trust_remote_code`) with a **different dynamic-tiling image pipeline + `model.chat()` API**, not the
  HF processor/`generate` contract. That is a real deviation from the **frozen frame-window recipe**
  ("only swap the model") → comparability + bug risk; 78B also carries a 156 GB download disk risk.
  **Rejected in favor of Qwen3-VL's drop-in HF processor/generate path** (byte-identical recipe except
  the model class). If Qwen3-VL fails the gate, InternVL is the natural *next* family to try — but it
  is out of scope for this pre-registered round.

**Why Qwen3-VL is in-recipe.** `p10c_score_segments.py` reuses `p10_score_segments.py` verbatim
(SYSTEM_P6 prompt, RUBRIC, USER_TAIL, greedy `do_sample=False`/`max_new_tokens=8`, K30/M120 + K4/M16
windows, window-level ASR, `[0-3]` parse, output contract) — the **only** change is loading via the
generic `AutoModelForImageTextToText` (→ `Qwen3VL*ForConditionalGeneration`) instead of the hard-wired
Qwen2.5-VL class. The video message `{"type":"video","video":<PIL frames>}` and
`apply_chat_template → processor(text,videos) → generate` path are the same Qwen-VL convention.

## Frozen candidate list (exactly 4 rows — no additions)

Calibration set / harness / anchor / paired protocol **identical to rounds 1–2** (HateMM
train-hateful, 298 scored / **n=266** both-class; `p10_eval_hatemm.py`, paired bootstrap 10k 95% CI
on per-video Δ vs the frozen **7B anchor 0.5387**; sign-test). Each scorer is run at **K=30 (fine)**
and **K=4 (coarse)** so its A-fuse channel is the SAME model (no cross-model fusion). "anchor-agg" =
raw K=30.

| # | candidate | scorer | aggregation | score tag | source |
|---|---|---|---|---|---|
| C1a | Qwen3-VL-32B · anchor-agg | Qwen3-VL-32B (bf16) | raw K=30 | `segscoreK30_p10c-qwen3vl-32b` | GPU (`p10c_score_qwen3vl.sbatch`) |
| C1b | Qwen3-VL-32B · A-fuse | Qwen3-VL-32B (bf16) | 0.5·K30 + 0.5·K4(map), same model | `segscoreK30_p10c-qwen3vl-32b-fuse` | GPU + CPU re-agg |
| C2a | Qwen3-VL-30B-A3B · anchor-agg | Qwen3-VL-30B-A3B (bf16) | raw K=30 | `segscoreK30_p10c-qwen3vl-30ba3b` | GPU |
| C2b | Qwen3-VL-30B-A3B · A-fuse | Qwen3-VL-30B-A3B (bf16) | 0.5·K30 + 0.5·K4(map), same model | `segscoreK30_p10c-qwen3vl-30ba3b-fuse` | GPU + CPU re-agg |

Version strings are frozen as of this commit (repos above; env HateVideoVLM / transformers 4.57.6) to
prevent post-hoc model/version shopping.

## Promotion bar — STRICTER than rounds 1–2 (third round, sequential-testing control)

A candidate is promoted to the single HateClipSeg test **iff**, on the HateMM calibration set:
1. **calibration wv-AUC ≥ 0.616**, **AND**
2. paired Δ vs the frozen 7B anchor (0.5387) has its bootstrap 95% CI **excluding 0**.

(Condition 1 subsumes the rounds-1/2 "+0.04 over anchor" gate: 0.616 − 0.5387 = **+0.0773** ≥ +0.04,
so 0.616 is the binding threshold.) If several clear, the single **highest** calibration wv-AUC is
promoted. **If none clears, P10-c = FAIL, the HateClipSeg test is NEVER touched, and P10-b's 0.5755
stands as the final localization number.**

**Round / multiple-comparison accounting (honest).** This is the **third** calibration round.
Configs compared against the frozen anchor: round 1 = **5** (A-gate/K60/fewshot/A-lex/A-fuse) + round 2
= **5** (R2-1..R2-5) + round 3 = **4** (C1a/C1b/C2a/C2b) = **14 total**. The bar is **tightened**, not
loosened, for round 3.

**Why the 0.616 gate (not the +0.04 anchor gate) governs round 3.** Two rounds already spent the
**single** HateClipSeg test touch on the 72B champion (calib 0.5913 → test 0.5755). Spending nothing
more is the default; a third test touch is only justified if a candidate beats the *already-tested*
champion by a margin that plausibly lands ≥ 0.60 on test. The **two observed calibration→test points**
— (0.5387 → 0.5435) and (0.5913 → 0.5755) — extrapolate linearly to **test 0.60 ⇔ calibration ≈
0.616** (same waterline as the TERMINUS exploratory ceiling analysis). The **reaggregation ceiling is
proven at 0.5932 < 0.616** (commit 93e82fa), so **only a stronger scorer**, not any aggregation knob,
can reach 0.616 — which is exactly what P10-c tests. Requiring wv-AUC ≥ 0.616 (well above the champion
0.5913) is the sequential-testing control: it stops a candidate that merely ties the champion (and
would only re-deliver ~0.5755 MODEST on test) from consuming the final test touch.

## Substantial bar on the HateClipSeg test — UNCHANGED (one touch total, promoted config only)

Frozen P6 harness (`p6_eval_localization.py`, 395-video split, within-video AUC + CI + memory/random
controls): **wv-AUC ≥ 0.60 = substantial** / **0.56 ≤ wv-AUC < 0.60 with CI excluding P6's 0.5435 =
modest** / **< 0.56 = P6/P10-b stands**. Scoring the test pass reuses the same K30/M120 + K4/M16 recipe
(`p10c_score_qwen3vl.sbatch DS=HateClipSeg SPLITS=test GTDIR=./data/gt`); ASR is the stored
`test_seen_asrK{30,4}_whisper-large-v3` (no Whisper re-run); A-fuse on CPU (`p10_aggregate_b.py --mode
fuse --split test_seen`).

## Calibration→test transfer risk (must stay attached to any verdict)

The whole gate rests on a **two-point, zero-degree-of-freedom** calibration→test line. Three
fragilities carry over from Kit-B B.5: (a) two points define the line with no slope uncertainty, and
the observed transfer is only ~60 % strength (calib +0.0526 → test +0.0319); (b) the 0.5913→0.616
region is **extrapolation beyond the observed champion**; (c) HateMM (n=266) and HateClipSeg (n=329)
are **cross-corpus**. Therefore **even a candidate clearing 0.616 does not guarantee test ≥ 0.60** —
the final test-touch decision is returned to the user, not auto-triggered by the calibration number.

## Hard rules (carry over)

SLURM only (no `--time`, `HF_HUB_OFFLINE=1` in-job, `WANDB_MODE=disabled`), foreground `sacct`
polling; calibration scoring uses hateful-only gt (`data/gt_p10hate/`) at K30 and K4; no `.pt`/weights
in git; **Qwen3-VL caches deleted after use** (quota watch on the shared 97 %-full /data); the
HateVideo env is never mutated (all P10-c runs in `HateVideoVLM`). Report the full three-round
leaderboard **before** any test pass.

## P10-c HateMM CALIBRATION LEADERBOARD (round 3)

Run 2026-07-09. Same calibration set and harness as rounds 1–2 (HateMM train hateful, 298 scored,
**n=266** both-class; `p10_eval_hatemm.py`, paired bootstrap 10k 95% CI vs the frozen 7B anchor
0.5387; random control wv 0.4940). Jobs: smoke 12604 (32B, LIMIT=3, path-validated: within-video
discrimination confirmed, `video_grid_thw` t=2 encodes all 4 frames/window); full runs 12605
(30B-A3B, 1h50) + 12606 (32B, 1h34). Scoring health uniform and identical to previous rounds
(1 undecodable video; parse fallbacks 0.34% at both K). A-fuse on CPU (`p10_aggregate_b.py --mode
fuse`). Machine rows appended to `scripts/analysis/loc_out/p10_hatemm_leaderboard.jsonl`.

Full three-round table (14 comparisons vs the anchor; round-3 gate: **wv-AUC ≥ 0.616 AND CI(Δ)
excl. 0**):

| round | variant | HateMM wv-AUC | paired Δ vs anchor | paired Δ 95% CI | clears round-3 bar |
|---|---|---|---|---|---|
| — | **anchor** (7B, raw K30) | 0.5387 | — | — | — |
| 1 | A-gate | 0.5314 | −0.0074 | [−0.0195, +0.0045] | — |
| 1 | K60 | 0.5319 | −0.0068 | [−0.0156, +0.0019] | — |
| 1 | fewshot | 0.5359 | −0.0028 | [−0.0090, +0.0034] | — |
| 1 | A-lex | 0.5450 | +0.0062 | [−0.0000, +0.0123] | — |
| 1 | A-fuse (7B) | 0.5693 | +0.0305 | [+0.0175, +0.0437] | — |
| 2 | R2-5 · 7B A-fuse×A-lex | 0.5752 | +0.0365 | [+0.0223, +0.0506] | — |
| 2 | R2-1 · 32B anchor-agg | 0.5512 | +0.0125 | [−0.0006, +0.0257] | — |
| 2 | R2-2 · 32B A-fuse | 0.5825 | +0.0437 | [+0.0240, +0.0631] | — |
| 2 | R2-3 · 72B anchor-agg | 0.5593 | +0.0206 | [+0.0065, +0.0347] | — |
| 2 | R2-4 · 72B A-fuse (**champion, test-spent 0.5755**) | **0.5913** | +0.0526 | [+0.0333, +0.0721] | — |
| 3 | C2a · Qwen3-VL-30B-A3B anchor-agg | 0.5469 | +0.0082 | [−0.0058, +0.0222] | no |
| 3 | C1a · Qwen3-VL-32B anchor-agg | 0.5594 | +0.0207 | [+0.0077, +0.0339] | no (0.5594 < 0.616) |
| 3 | C2b · Qwen3-VL-30B-A3B A-fuse | 0.5821 | +0.0433 | [+0.0227, +0.0644] | no (0.5821 < 0.616) |
| 3 | **C1b · Qwen3-VL-32B A-fuse** (round-3 best) | **0.5866** | **+0.0479** | **[+0.0287, +0.0677]** | **no (0.5866 < 0.616)** |

**Reading (three clean facts):**
- **The new generation reproduces the P10-b structure almost exactly, at ~half the parameter cost.**
  Qwen3-VL-32B lands within noise of Qwen2.5-VL-32B on both aggregations (anchor-agg 0.5594 vs
  0.5512; A-fuse 0.5866 vs 0.5825), and A-fuse × scorer strength is again the only lever
  (30B-A3B +0.0433, 32B +0.0479 — both significant, both far above their anchor-agg rows).
- **Generation upgrade ≠ scale upgrade.** Qwen3-VL-32B A-fuse (0.5866) does **not** surpass the
  Qwen2.5-VL-**72B** A-fuse champion (0.5913): a newer 32B ties the two-generations-older 32B tier
  and stays below the 72B tier. Under the frozen recipe, within-video localization tracks
  *capacity* more than *generation*.
- The MoE (30B-A3B, 3B active) anchor-agg row (0.5469, CI incl. 0) is the weakest, consistent with
  active-parameter count (~3B) rather than total (30B) governing per-window rating quality; its
  A-fuse row still clears the old +0.04 bar — A-fuse's robustness across scorers is now shown on
  **five** models (7B/32B/72B/Qwen3-32B/Qwen3-30B-A3B).

### VERDICT vs the pre-registered round-3 gate — **FAIL (no promotion, test never touched)**

No candidate reaches calibration wv-AUC ≥ 0.616 (best: Qwen3-VL-32B A-fuse **0.5866**, below even
the already-tested 72B champion 0.5913). Per the pre-registration, **P10-c dies calibration-side:
the third HateClipSeg test touch is NOT spent**, and P10-b's result stands as the final localization
number (HateClipSeg wv-AUC **0.5755**, MODEST). The ceiling argument extends: neither aggregation
knobs (≤0.5932, commit 93e82fa) **nor a one-generation-newer open scorer at the ~32B tier** reaches
the 0.616 waterline the two-point calibration→test mapping demands for a substantial (≥0.60) test
result. What remains untried within open weights is a genuinely *larger* new-gen scorer
(Qwen3-VL-235B-A22B — infeasible on this cluster's A100/FP8 + disk budget) — recorded as out of
reach, not as an open in-register path.

**Bottom line: P10-c = FAIL, honestly closed.** Round 3 adds 4 comparisons (14 total vs anchor),
zero promotions. The MLLM-localization role remains at its P10-b magnitude: **modest-plus
(HateClipSeg wv-AUC 0.5755, CI [0.5581, 0.5933])**. The campaign's earned-roles verdict is
unchanged: encoder + localizer + guard-rail/audit, no main-table accuracy role; the localization
substantial line (0.60) is not reached by any open-weights scorer feasible on this cluster.
Qwen3-VL caches deleted after the runs per quota policy.

