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

_(to be appended after the round-2 scoring completes.)_

