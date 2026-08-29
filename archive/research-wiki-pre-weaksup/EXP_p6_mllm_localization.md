# EXP: P6 — MLLM evidence scores for span-free temporal localization

> **Status: PRE-REGISTERED (design frozen before any MLLM score is evaluated).**
> Motivation, pipeline, conditions, metrics, and the success bar are committed before running
> the scoring/eval. Results are appended in `## RESULTS`. Numbers by
> `scripts/analysis/p6_eval_localization.py`; MLLM scores by the frozen P3 scorer
> `scripts/analysis/score_segments_mllm.py` (prompt/model/greedy UNCHANGED); ASR by
> `src/utils/generate_segment_asr_HF.py`.

## Motivation

The existing zero-training localization capability (`research-wiki/EVAL_localization_hateclipseg.md`)
is an **existence proof only**: cross-dataset consensus-kNN scoring of CLIP-visual windows gives
best within-video mean-AUC 0.526 (CI [0.505, 0.547], sign-test p=0.0066 — significant in exactly
1 of 4 cells) and +0.088 frame-AP over random. It is weak because **CLIP-visual memory keys are
blind to speech-borne hate**, and HateClipSeg hate is largely spoken. P6 asks: does an MLLM that
**reads frames + ASR per window** localize substantially better? If yes, the MLLM earns a real,
removable role in the localization capability; if not, the capability stays existence-proof grade
and the MLLM-localization role is an honest kill.

## Data & harness (reuse EVAL_localization_hateclipseg.md EXACTLY)

- HateClipSeg alive subset: **395 videos**, cleaned gold segments (`data/gt/HateClipSeg/
  gold_segments.json`, 10,572 kept segments tiling [0,D); gold spans are **VALIDATION ONLY** — no
  HateClipSeg label enters any scoring path). Our declared split = all 395 (zero-training, no
  leakage). Same broadcast control, same 1-fps second→window mapping `min(K−1,⌊mK/D⌋)`, same
  within-video / frame / segment protocols and estimators, imported read-only from
  `scripts/analysis/eval_localization_hateclipseg.py`.
- **Granularity = K=30, M=120** (window median ≈ 8s, density-matched to the gold-segment median
  8.12s; ≈ 11,850 windows). This is the granularity where within-video localization is measurable
  (30 rankable windows/video) and where ASR is localized to ~8s; it matches the existing doc's
  K=30 memory numbers for a same-granularity head-to-head. Choosing K=30 (not P3's default K=4) is
  the **window-alignment adaptation** anticipated in the brief — it changes only the window count,
  not the P3 prompt/model/decoding, which are frozen.

## Pipeline (two GPU jobs, then CPU eval)

1. **ASR** — `generate_segment_asr_HF.py --dataset HateClipSeg --splits test --num_frames 120
   --num_subclips 30 --timestamps word` → `data/ASR/HateClipSeg/test_seen_asrK30_whisper-large-v3.jsonl`
   (Whisper large-v3, word timestamps binned to the 30 windows; language=en).
2. **MLLM scoring** — the **frozen** P3 scorer `score_segments_mllm.py --dataset HateClipSeg
   --splits test --num_frames 120 --num_subclips 30 --asr_tag asrK30_whisper-large-v3`
   → `data/MLLM_scores/HateClipSeg/test_seen_segscoreK30_qwen.jsonl` (Qwen2.5-VL-7B, greedy, each
   window scored IN ISOLATION on its ≤4 frames + its window ASR → integer hate-evidence density
   0..3). Prompt/model/decoding identical to P3 — **not retuned for HateClipSeg** (zero-shot; the
   rubric says "hate evidence density", so only a **threshold-free ranking** metric is valid, no
   operating point to tune).
3. **Eval** (CPU) — `p6_eval_localization.py` builds the [395, 30] window-score matrix for each
   condition and runs the SAME metric functions.

## Conditions (one evaluation pass, no metric shopping)

| id | condition | window-score source |
|----|-----------|---------------------|
| **a** | memory (baseline, reproduce) | consensus-kNN, `knn_hatemm_subclip` @ K=30 (existing doc's best memory config at this K; `knn_hatemm_video` also reported) — cached `loc_out_hcs/scores_knn_*_K30.npz` |
| **b** | **MLLM (ours)** | P3 scorer integer scores 0..3 |
| **c** | combination (pre-registered, ONE rule) | **per-video rank-average of (a) and (b)** — within each video, average the two windows' ranks (normalised rank/(K−1)); frozen now, no alternatives |
| **d** | random control | `np.random.RandomState(0)` |
| **e** | broadcast control | per-video mean of (b) broadcast to all windows (pooled-metric control; within-video AUC = 0.5 by construction) |

## Metrics (same estimators as the existing doc)

- **within-video mean AUC** over videos with both classes (the sharp localization diagnostic) +
  **bootstrap 10k 95% CI** + one-sided sign-test vs 0.5. **This is the PRIMARY metric.**
- frame-level AP/AUC (protocol-full + toxiconly) and segment-level AP/AUC (duration-weighted),
  reported as supporting/pooled evidence. Per the existing doc, pooled AP mostly reflects
  video-level toxicity **density**, not within-video localization — so it is secondary here.

## Pre-registered success bar

**PRIMARY (MLLM earns a removable localization role) — ALL of:**
1. within-video mean-AUC(**b**) > within-video mean-AUC(**a**) AND > (**d**);
2. **b**'s 95% bootstrap CI **excludes 0.5** and sign-test p < 0.05.

**SECONDARY (supporting, NOT required):**
3. frame-full AP(**b**) − AP(**a**) ≥ **+0.176** (= 2× the existing +0.088 headline delta) — the
   pre-registered "substantial" AP bar; stated now, acknowledged as demanding (pooled AP is
   density-dominated, so within-video AUC is the real localization test);
4. within-video AUC(**c**) ≥ max(**a**,**b**).

**KILL:** if (1)/(2) fail — the MLLM does not beat memory + random on within-video AUC with CI
excluding null — the localization capability **stays existence-proof grade**; that is an honest
kill of the MLLM-localization role, reported with the mechanism.

## Hard rules

GPU via SLURM (no `--time`, `HF_HUB_OFFLINE=1`, `WANDB_MODE=disabled`); one ASR job + one scoring
job; poll `sacct`; resume-safe, FORCE=False, no cache overwrite; no `.pt` in git; disk under quota;
commit (no push). The scorer is byte-frozen from P3 (only `--num_frames/--num_subclips/--asr_tag`
differ, i.e. window alignment).

---

## RESULTS

Run 2026-07-06/07. ASR = SLURM job 12394 (Whisper large-v3, K=30 word-timestamps; 395 videos,
394 with speech, 0 no-audio). MLLM scores = job 12395 (frozen P3 scorer, K=30/M=120; 395 videos,
394 decodable, 11,820/11,850 windows parsed; **1 undecodable video** yt_NzvfkIYS5Yg → zero row,
kept). MLLM score distribution is near-binary: **0 = 77.8%, 3 = 21.7%** (1/2 almost unused) — the
rater flags a minority of windows as explicit hate and the rest as none. Eval =
`p6_eval_localization.py` (CPU); condition (a) reproduces the existing K=30 memory numbers
bit-identically. Machine JSON: `loc_out_hcs/results_p6_mllm_loc.json`.

### K=30 — all conditions (395 videos; wv over the 329 both-class videos)

| condition | frame-full AP / AUC | frame-tox AP / AUC | segment AP / AUC | **within-video AUC** |
|---|---|---|---|---|
| a — memory `knn_hatemm_subclip` | 0.5329 / 0.5754 | 0.6074 / 0.5850 | 0.5246 / 0.5839 | 0.5140 |
| a′ — memory `knn_hatemm_video` | 0.5247 / 0.5656 | 0.6020 / 0.5732 | 0.5120 / 0.5688 | 0.5134 |
| **b — MLLM (ours)** | 0.5421 / **0.6034** | 0.6023 / 0.6017 | 0.5599 / **0.6353** | **0.5435** |
| c — per-video rank-avg(a,b) | 0.4971 / 0.5268 | 0.5624 / 0.5314 | 0.4867 / 0.5393 | 0.5371 |
| d — random | 0.4699 / 0.5084 | 0.5360 / 0.5090 | 0.4507 / 0.5065 | 0.5088 |
| e — broadcast of MLLM mean | **0.6297 / 0.6831** | 0.6778 / 0.6545 | 0.6158 / 0.6817 | 0.5000\* |

\* broadcast wv-AUC = 0.5 by construction (constant within video).

### Within-video significance (the primary metric)

- **MLLM (b) vs its own null:** mean wv-AUC **0.5435**, bootstrap 95% CI **[0.5330, 0.5544]**
  (excludes 0.5), sign-test **p = 5.4e-8** (vs the memory best cell's p=0.0066 — three orders
  tighter).
- **MLLM (b) vs memory (a), PAIRED per video (n=329):** mean Δ = **+0.0296**, bootstrap 95% CI
  **[+0.0088, +0.0504]** (excludes 0), sign-test **p = 0.0071** (b>a on 184 videos, b<a on 139).
- Segment-level AUC 0.6353 vs memory 0.5839 (+0.051) and frame AUC 0.6034 vs 0.5754 (+0.028)
  move the same way; the threshold-free **ranking** metrics all favour the MLLM.

### Verdict vs the pre-registered bar

1. **PRIMARY — MET.** (1) wv-AUC(b)=0.5435 > a=0.5140 AND > d=0.5088 ✓; (2) b's 95% CI excludes
   0.5 and sign-p<0.05 ✓ (and the paired b>a test is also significant, p=0.007). **The MLLM
   earns a real, removable localization role.**
2. **SECONDARY — not met (as anticipated).** frame-full AP(b)−AP(a) = **+0.0092** ≪ the +0.176
   bar. But pooled AP is **density-dominated**: the broadcast control (e) — the MLLM's per-video
   MEAN broadcast to all windows — is the single best pooled number (AP 0.6297 / AUC 0.6831),
   i.e. the MLLM's strongest signal is **video-level toxicity density**, not fine localization.
   Adding the (imperfect) within-video variation to that mean actually lowers pooled AP from
   0.63 (broadcast) to 0.54 (b), which is why the pooled-AP bar can't be met even though the
   within-video ranking is significantly better. This is exactly why the pre-registration made
   within-video AUC — not pooled AP — the primary metric.
3. **Combination (c) — not required, does not help.** rank-averaging with the weaker memory
   scorer dilutes the MLLM (wv 0.5371 < 0.5435). The MLLM stands alone; fusion hurts.

### Plain-language bottom line

**P6 is the campaign's first clearly-positive MLLM method-role result: reading frames + ASR, the
MLLM localizes hate WITHIN videos significantly better than the CLIP-visual memory scorer and
than chance.** It upgrades the localization capability from an *existence proof* (memory:
within-video AUC 0.526, significant in only 1 of 4 cells at p=0.0066) to a **significant
single-config MLLM localizer** (wv-AUC 0.5435, CI [0.533, 0.554], p=5.4e-8; and paired over
memory +0.030, p=0.007). The honest magnitude caveat: this is still a **modest** localizer —
0.5435 is ~3.5 points of AUC above chance and ~3 points above memory, not a strong 0.7-style
signal — and the MLLM's *dominant* competence here is video-level density detection (broadcast
AP 0.63), with fine within-window localization the smaller, though now statistically solid,
increment. The mechanism matches the prior doc's diagnosis: the memory scorer's CLIP-visual keys
are blind to speech-borne hate, and the MLLM's window ASR is exactly what closes part of that
gap. The MLLM-localization role is **kept** (not existence-proof-only), reported at its true
modest-but-significant strength; the natural next lever is finer/segment-native windows and
speech-weighted scoring, out of P6's frozen-scorer scope.

*(Numbers by `scripts/analysis/p6_eval_localization.py` / paired test in the run log; condition
(a) reproduces `EVAL_localization_hateclipseg.md` K=30 exactly. ASR + MLLM score caches under
`data/ASR/HateClipSeg/` and `data/MLLM_scores/HateClipSeg/`; no `.pt` in git.)*
