# Weakly-supervised frame-level baselines (imported from Hate-follow-up), 2026-08-23

> **OUTDATED (2026-08-30).** This is a 2026-08-23 snapshot covering only 4 reproduced methods.
> The authoritative baseline table is now `docs/duplex/OFFICIAL_VAL_RESULTS.md`
> (7 methods × 4 corpora, 3-seed official validation). See `research-wiki/STATUS.md`.

## 0. Provenance and comparability warning — read before quoting any number

**Source repository.** Every number in this file is transcribed from
`/home/jehc223/Hate-follow-up/docs/duplex/BASELINE_RESULTS.md`.
Repository HEAD at transcription time:

```
git -C /home/jehc223/Hate-follow-up log -1 --format=%H
14d5fa1a710da1530c81634c22d4191b65a4f717   (2026-08-23, "Add LAVAD baseline adapter and evaluation plumbing")
```

Working tree was clean at transcription time (`git status --short` empty).

**The raw JSON is not on this machine.** BASELINE_RESULTS.md states each row comes from
`results/reproduction/baselines/<method>/<corpus>/frame_eval.json`
(BASELINE_RESULTS.md:12-14). That directory does not exist in the checkout on this
workstation — `/home/jehc223/Hate-follow-up/results/` contains only `scale_emergence/`.
**The markdown is currently the sole surviving record of these numbers.** Nothing below
has been re-derived from evaluator output; this file is a transcription of a transcription.

**Protocol of the imported numbers** (`Hate-follow-up/docs/duplex/FRAME_EVAL_PROTOCOL.md`,
frozen 2026-08-18, amended 2026-08-19 to add HateClipSeg):

- **1 fps frame grid.** Frame timestamps are integers `t = 0,1,2,...` while `t < duration`;
  duration is the wav duration of the extracted 16 kHz mono audio, not the container duration
  (FRAME_EVAL_PROTOCOL.md:36-45). The rate is 1 fps because upstream annotations are given at
  whole-second resolution (FRAME_EVAL_PROTOCOL.md:47-52).
- **Span → frame:** positive iff `start <= t < end` (half-open), overlapping spans union,
  degenerate spans (`end <= start`) dropped (FRAME_EVAL_PROTOCOL.md:56-62).
- **Training supervision:** video-level labels only for every method in §2.
- **Checkpoint selection:** seeded, label-stratified **10 % validation carve-out of the train
  split**, selected by video-level average precision. **Upstream's test-selected checkpointing
  was removed** (patch V3 / M7). The test split is never opened during training
  (BASELINE_RESULTS.md:74-76, 144-149, 366-368). This makes these numbers *lower* than the
  corresponding upstream protocol would produce (BASELINE_RESULTS.md:146-149).

**These numbers cannot be placed in the same table as this repository's label-free campaign
results.** This repo's campaign (`idea-stage/repro_campaign/summary_test.csv`,
`research-wiki/LABELFREE_FRAMELEVEL_BASELINES_2026-08-18.md`) is built on a **dense 4 fps**
grid with its own ground-truth arrays (`data/gt/frame_gt_4fps/`, see
`idea-stage/repro_campaign/PHASE_A_STATUS.md:12,42-44,63-67`). Different grid, different gold
rasterization, different splits, different cohort. **Do not merge these two tables.** They are
imported here as a supervision-regime reference point, not as a comparison row.

**Column definitions** (BASELINE_RESULTS.md:20-24):
- *pooled ROC-AUC / pooled PR-AUC* — over every frame of every scored video.
- *within-hate macro (n)* — mean per-video ROC-AUC restricted to hateful videos whose gold
  array contains both classes; `n` = how many videos the mean covers. This is the column that
  measures **where inside a hateful video** the score peaks.
- *video AUC* — frame scores max-pooled per video, ranked against the video label.

**Cohorts and base rates:**

| corpus | gold videos | frames | frame positive rate | within-macro n | source lines |
|---|---|---|---|---|---|
| HateMM | 214 | 29,266 | 0.2419 | 85 / 214 (40 %) | :44-47, :1101 |
| MHC EN | 158 | 5,600 | 0.2505 | 44 / 158 (28 %) | :44-47, :1101 |
| MHC ZH | 153 | 4,817 | 0.2327 | 7 / 153 (5 %) | :44-47, :1101 |
| HateClipSeg | 79 (of 315 train / 79 test) | 18,839 | **0.5255** | 67 / 79 (85 %) | :1099-1122 |

HateClipSeg's chance PR-AUC is **0.5255**, not ~0.24 (BASELINE_RESULTS.md:1117-1118), and 69 of
79 test videos are video-level positive (87 %), so its video AUC rests on 10 negatives and its
video-AP selection criterion is close to uninformative (BASELINE_RESULTS.md:1118-1122).

---

## 1. One-glance summary: pooled frame ROC-AUC, primary branch only

Primary branch = the branch each paper headlines (`score_mlp` for VadCLIP/DSANet,
`score_av` for MACIL-SD, `score_fused` for MultiHateLoc, `score_mil` for the uni-modal MILs).

| Method | Supervision | HateMM | MHC EN | MHC ZH | HateClipSeg |
|---|---|---|---|---|---|
| **VadCLIP** (AAAI'24) | video labels | 0.6855 | 0.6281 | 0.5676 | 0.5328 |
| **DSANet** (AAAI'26) | video labels | 0.7063 | 0.6684 | 0.5749 | 0.5387 |
| **MACIL-SD** (MM'22, av) | video labels | 0.7282 | 0.6764 | 0.7757 | 0.5220 |
| **MACIL-SD audio-only MIL** | video labels | **0.7667** | **0.7142** | 0.6320 | 0.5166 |
| **MACIL-SD visual-only MIL** | video labels | 0.6398 | 0.6340 | 0.6860 | 0.5195 |
| **MultiHateLoc-reimpl** (WWW'26, fused) | video labels | 0.7504 | 0.6740 | 0.6749 | 0.4993 |
| *— reference, NOT weakly supervised —* | | | | | |
| *Ours (masked locator, 1 fwd/video)* | *zero labels* | *0.7451* | *0.6198* | *0.6004* | *0.5414* |
| *Vad-R1* (NeurIPS'25) | *none; ckpt trained on its own VAD corpus* | *0.5696* | *0.5427* | *0.5987* | ***0.6382*** |
| *EventVAD* (MM'25) | *none (training-free)* | *0.5174* | *0.5041* | *0.5202* | *not run* |

Best per corpus among **weakly supervised** methods: HateMM 0.7667 (audio-only MIL),
MHC EN 0.7142 (audio-only MIL), MHC ZH 0.7757 (MACIL-SD av), HateClipSeg 0.5387 (DSANet).

And the same table for the column that actually measures localization,
**within-hate macro** (chance = 0.5):

| Method | HateMM (n=85) | MHC EN (n=44) | MHC ZH (n=7) | HateClipSeg (n=67) |
|---|---|---|---|---|
| VadCLIP `score_mlp` | 0.4848 | 0.3331 | 0.3562 | 0.5530 |
| DSANet `score_mlp` | 0.5453 | 0.3844 | 0.3557 | **0.5583** |
| DSANet `score_align` | 0.5689 | **0.7230** | 0.5279 | 0.5015 |
| MACIL-SD `score_av` | 0.5383 | 0.5383 | 0.4588 | 0.5329 |
| audio-only MIL | 0.5966 | 0.5142 | 0.5269 | 0.5266 |
| visual-only MIL | 0.4966 | 0.5104 | 0.4995 | 0.5394 |
| MultiHateLoc `score_fused` | **0.6008** | 0.4611 | 0.4126 | 0.5085 |
| *Ours (zero labels)* | *0.5706* | *0.6154* | *0.6076* | *0.5324* |
| *Vad-R1* | *0.5000* | *0.5000* | *0.5000* | *0.5001* |

**Six of the seven weakly-supervised primary-branch cells on MHC EN / MHC ZH are at or below
chance**, i.e. the MIL score systematically peaks on the *non-hateful* seconds of hateful
videos (BASELINE_RESULTS.md:61-65).

---

## 2. Weakly supervised methods, full transcription

All rows: training supervision = **video-level label only**; output = per-frame (or per-snippet,
rasterized) score; checkpoint by 10 % val carve, test never opened.

### 2.1 VadCLIP — AAAI 2024

- Features: CLIP ViT-B/16 per-frame, precomputed.
- Branches: `score_mlp` = the MIL branch (paper's headline); `score_align` = text-alignment
  branch (BASELINE_RESULTS.md:49-50).
- Run settings: published XD-Violence defaults except `classes-num 2` and per-corpus
  `visual-length`/`attn-window`; seed 234, lr 1e-5, batch 96, 10 epochs
  (BASELINE_RESULTS.md:69-76).

| corpus | branch | pooled ROC | pooled PR | within-hate macro (n) | video AUC | line |
|---|---|---|---|---|---|---|
| HateMM | score_mlp | 0.6855 | 0.4457 | 0.4848 (85) | 0.7242 | :28 |
| HateMM | score_align | 0.5685 | 0.3359 | 0.5037 (85) | 0.6473 | :29 |
| MHC EN | score_mlp | 0.6281 | 0.3611 | 0.3331 (44) | 0.6405 | :30 |
| MHC EN | score_align | 0.4791 | 0.2347 | 0.4621 (44) | 0.5306 | :31 |
| MHC ZH | score_mlp | 0.5676 | 0.2705 | 0.3562 (7) | 0.5537 | :32 |
| MHC ZH | score_align | 0.3880 | 0.1806 | 0.3225 (7) | 0.3981 | :33 |
| HateClipSeg | score_mlp | 0.5328 | 0.5447 | 0.5530 (67) | 0.5072 | :1134 |
| HateClipSeg | score_align | 0.5531 | 0.5860 | 0.5063 (67) | 0.6275 | :1135 |

`VadCLIP / MHC ZH / score_align` at **0.3880 pooled is the worst cell in the study**, below
chance by a clear margin (BASELINE_RESULTS.md:64-65).

### 2.2 DSANet — AAAI 2026

- Features: consumes VadCLIP's exact CLIP ViT-B/16 features.
- Branches: `score_mlp` (MIL), `score_refined` (hierarchical refinement), `score_align`
  (text alignment).
- **`score_refined` equals `score_mlp` to eight decimal places** on every corpus — the expected
  consequence of binary collapse: with one non-normal class the hierarchical refinement
  redistributes the MLP score over a single column and returns it unchanged
  (BASELINE_RESULTS.md:50-55). The two branches are kept separate only so the equality is on
  record.

| corpus | branch | pooled ROC | pooled PR | within-hate macro (n) | video AUC | line |
|---|---|---|---|---|---|---|
| HateMM | score_mlp | 0.7063 | 0.4824 | 0.5453 (85) | 0.7470 | :34 |
| HateMM | score_refined | 0.7063 | 0.4824 | 0.5453 (85) | 0.7470 | :35 |
| HateMM | score_align | 0.6828 | 0.4540 | 0.5689 (85) | 0.7259 | :36 |
| MHC EN | score_mlp | 0.6684 | 0.4354 | 0.3844 (44) | 0.6768 | :37 |
| MHC EN | score_refined | 0.6684 | 0.4354 | 0.3844 (44) | 0.6768 | :38 |
| MHC EN | score_align | 0.5602 | 0.3596 | **0.7230** (44) | 0.5860 | :39 |
| MHC ZH | score_mlp | 0.5749 | 0.2921 | 0.3557 (7) | 0.5588 | :40 |
| MHC ZH | score_refined | 0.5749 | 0.2921 | 0.3557 (7) | 0.5588 | :41 |
| MHC ZH | score_align | 0.5904 | 0.3082 | 0.5279 (7) | 0.5792 | :42 |
| HateClipSeg | score_mlp | 0.5387 | 0.5502 | **0.5583** (67) | 0.5493 | :1136 |
| HateClipSeg | score_refined | 0.5387 | 0.5502 | 0.5583 (67) | 0.5493 | :1137 |
| HateClipSeg | score_align | 0.5559 | 0.5507 | 0.5015 (67) | 0.6145 | :1138 |

**Branch-level dissociation worth noting:** DSANet's alignment branch scores 0.7230 within-hate
macro on MHC EN — the single highest localization cell of any method on that corpus, including
the zero-label locator's 0.6154 — while its *pooled* number on the same branch is only 0.5602
and its own MIL branch's macro is 0.3844 (BASELINE_RESULTS.md:39, 528-530, 928-930).

### 2.3 MACIL-SD — ACM MM 2022, plus its two uni-modal ablations

- Features: I3D (visual) + VGGish (audio), precomputed.
- The `MACIL-SD` rows are the **audio-visual model**, one training run exposing three readouts:
  `score_av` (the fused score the paper headlines), `score_audio` and `score_visual`
  (the two branches of that same fused model).
- The **`MACIL-SD audio-only` and `MACIL-SD visual-only` rows are separate trainings** of
  upstream's own `Single_Model` on one modality alone, at upstream's lr/5 (patch M11). These
  are the honest uni-modal comparators, not branches of the fused model
  (BASELINE_RESULTS.md:372-378).
- Run settings: published defaults — seed 2333, lr 4e-4, batch 128, 50 epochs, `max-seqlen`
  200, EMA momentum 0.91, CMA lambdas 1.5/1.5/0.1, `--grid snippet`, `--crop-repeat 5`
  (BASELINE_RESULTS.md:361-368).

| corpus | row | branch | pooled ROC | pooled PR | within-hate macro (n) | video AUC | line |
|---|---|---|---|---|---|---|---|
| HateMM | MACIL-SD (av model) | score_av | 0.7282 | 0.5127 | 0.5383 (85) | 0.7611 | :382 |
| HateMM | MACIL-SD (av model) | score_audio | 0.7290 | 0.4501 | 0.5419 (85) | 0.7379 | :383 |
| HateMM | MACIL-SD (av model) | score_visual | 0.6552 | 0.4447 | 0.5012 (85) | 0.7059 | :384 |
| MHC EN | MACIL-SD (av model) | score_av | 0.6764 | 0.4664 | 0.5383 (44) | 0.7112 | :385 |
| MHC EN | MACIL-SD (av model) | score_audio | 0.6575 | 0.4453 | 0.5284 (44) | 0.7240 | :386 |
| MHC EN | MACIL-SD (av model) | score_visual | 0.6759 | 0.4530 | 0.5397 (44) | 0.6858 | :387 |
| MHC ZH | MACIL-SD (av model) | score_av | 0.7757 | 0.5233 | 0.4588 (7) | 0.7685 | :388 |
| MHC ZH | MACIL-SD (av model) | score_audio | **0.7774** | 0.5301 | 0.5256 (7) | 0.7808 | :389 |
| MHC ZH | MACIL-SD (av model) | score_visual | 0.7387 | 0.4834 | 0.4258 (7) | 0.7321 | :390 |
| HateMM | **audio-only MIL (VGGish)** | score_mil | **0.7667** | 0.4939 | 0.5966 (85) | 0.7814 | :391 |
| MHC EN | **audio-only MIL** | score_mil | **0.7142** | 0.4987 | 0.5142 (44) | 0.7141 | :392 |
| MHC ZH | **audio-only MIL** | score_mil | 0.6320 | 0.3254 | 0.5269 (7) | 0.6725 | :393 |
| HateMM | **visual-only MIL (I3D)** | score_mil | 0.6398 | 0.4073 | 0.4966 (85) | 0.7046 | :394 |
| MHC EN | **visual-only MIL** | score_mil | 0.6340 | 0.3670 | 0.5104 (44) | 0.6632 | :395 |
| MHC ZH | **visual-only MIL** | score_mil | 0.6860 | 0.4085 | 0.4995 (7) | 0.7262 | :396 |
| HateClipSeg | MACIL-SD (av model) | score_av | 0.5220 | 0.5412 | 0.5329 (67) | 0.5101 | :1139 |
| HateClipSeg | MACIL-SD (av model) | score_audio | 0.5132 | 0.5314 | 0.5123 (67) | 0.5159 | :1140 |
| HateClipSeg | MACIL-SD (av model) | score_visual | 0.5215 | 0.5471 | 0.5363 (67) | 0.5087 | :1141 |
| HateClipSeg | **audio-only MIL** | score_mil | 0.5166 | 0.5568 | 0.5266 (67) | 0.6275 | :1142 |
| HateClipSeg | **visual-only MIL** | score_mil | 0.5195 | 0.5581 | 0.5394 (67) | 0.6232 | :1143 |

**Audio carries the signal and fusion does not add to it.** The standalone audio-only model
beats the full audio-visual model on HateMM (0.7667 vs 0.7282 pooled; 0.7814 vs 0.7611 video)
and on MHC EN (0.7142 vs 0.6764), and beats the visual-only model on both. The exception is
MHC ZH, where audio-only drops to 0.6320 while the fused model reaches 0.7757
(BASELINE_RESULTS.md:410-418).

### 2.4 MultiHateLoc-reimplementation — WWW 2026 (arXiv 2512.10408)

**These numbers do not come from the authors' code.** The announced repository
`github.com/mmilabuk/multihateloc` contains a LICENSE file and nothing else, so these rows are
a **from-scratch reimplementation from the paper text** under the frozen protocol; every
architectural detail the paper leaves unstated was filled with the simplest working reading and
is marked `INFERRED` in `scripts/reproduction_baselines/multihateloc/DESIGN.md`
(BASELINE_RESULTS.md:151-161, and BASELINE_REPRODUCTION_LIST.md:74-77, 100-112).

Three caveats stated at the source (BASELINE_RESULTS.md:163-180):
1. **The paper does not state its frame rate.** T is only "the number of frames". The port
   freezes 1 fps; on any other grid the MIL pool size `ceil(T/3)` changes and a frame-level AUC
   means something else.
2. **The published 0.645 frame mAP / 0.799 AUC on HateMM is not a target these rows can hit or
   miss** — grid, gold rasterization, splits, cohort, model selection and metric all differ,
   four of the six because the paper does not specify them.
3. The largest invention is where the Dynamic Modality Selection weights enter the network: the
   paper uses them only in final frame selection (where they receive no gradient), so here they
   also scale each modality's contribution to the fused branch.

Branches (BASELINE_RESULTS.md:209-216): `score_fused` = primary, the paper's headline;
`score_visual` / `score_audio` / `score_text` = the three modality branches, each supervised by
the same MIL loss; `score_union` = **the paper's literal output**, the importance-gated union of
four top-K frame sets; `score_dms` = **not in the paper**, this port's continuous reading of the
same importance weights (convex combination of the three modality probabilities).

Features (BASELINE_RESULTS.md:271-281): visual = ImageNet ViT-B/16 (`google/vit-base-patch16-224`
CLS, 768-d — the encoder the paper cites, **not CLIP**); audio = VGGish 128-d; text =
`bert-base-uncased` (`bert-base-chinese` for ZH) over frozen whisper-large-v3 fragments,
repeat-padded across each fragment's interval. Published hyperparameters used verbatim: Adam
lr 1e-4, batch 32, 100 epochs, K = 3 (top third), smoothness λ 0.1, contrastive λ 0.2
(BASELINE_RESULTS.md:263-269).

| corpus | branch | pooled ROC | pooled PR | within-hate macro (n) | video AUC | line |
|---|---|---|---|---|---|---|
| HateMM | score_fused | 0.7504 | 0.4856 | **0.6008** (85) | 0.8622 | :190 |
| HateMM | score_dms | 0.7595 | 0.5165 | 0.6029 (85) | 0.8625 | :191 |
| HateMM | score_visual | 0.6434 | 0.4053 | 0.5495 (85) | 0.7126 | :192 |
| HateMM | score_audio | **0.7777** | 0.5115 | 0.6106 (85) | 0.8156 | :193 |
| HateMM | score_text | 0.6777 | 0.4137 | 0.5398 (85) | 0.8006 | :194 |
| HateMM | score_union | 0.5249 | 0.2517 | 0.5283 (85) | 0.5000 | :195 |
| MHC EN | score_fused | 0.6740 | 0.3700 | 0.4611 (44) | 0.6498 | :196 |
| MHC EN | score_dms | 0.6832 | 0.3890 | 0.4553 (44) | 0.6543 | :197 |
| MHC EN | score_visual | 0.6378 | 0.4110 | 0.4902 (44) | 0.6557 | :198 |
| MHC EN | score_audio | 0.6711 | 0.3528 | 0.5206 (44) | 0.7124 | :199 |
| MHC EN | score_text | 0.6219 | 0.3050 | 0.4902 (44) | 0.5312 | :200 |
| MHC EN | score_union | 0.4916 | 0.2474 | 0.4661 (44) | 0.5000 | :201 |
| MHC ZH | score_fused | 0.6749 | 0.4032 | 0.4126 (7) | 0.7382 | :202 |
| MHC ZH | score_dms | 0.7022 | 0.4085 | 0.4299 (7) | 0.7233 | :203 |
| MHC ZH | score_visual | 0.7011 | 0.4004 | 0.3697 (7) | 0.7156 | :204 |
| MHC ZH | score_audio | 0.6487 | 0.3314 | 0.5254 (7) | 0.6622 | :205 |
| MHC ZH | score_text | 0.6659 | 0.4134 | 0.4404 (7) | 0.6444 | :206 |
| MHC ZH | score_union | 0.4978 | 0.2319 | 0.3503 (7) | 0.5000 | :207 |
| HateClipSeg | score_fused | 0.4993 | 0.5208 | 0.5085 (67) | 0.6246 | :1144 |
| HateClipSeg | score_dms | 0.5393 | 0.5684 | 0.4903 (67) | 0.6493 | :1145 |
| HateClipSeg | score_visual | 0.5520 | 0.5767 | 0.4977 (67) | 0.6797 | :1146 |
| HateClipSeg | score_audio | 0.4792 | 0.4997 | **0.4709** (67) | 0.5210 | :1147 |
| HateClipSeg | score_text | 0.5346 | 0.5736 | 0.4990 (67) | 0.5558 | :1148 |
| HateClipSeg | score_union | 0.4837 | 0.5176 | 0.4942 (67) | 0.5000 | :1149 |

**`score_union` — the paper's literal output — is degenerate under a ranking metric.** Its
video AUC is exactly 0.5000 on all four corpora, and that is not a coincidence: the union always
contains at least the fused branch's top third of frames, so every video has at least one frame
set to 1, max-pooling gives every video the identical score, and the ranking is one giant tie.
Its pooled frame ROC of 0.48–0.52 is the same fact at frame level — a 0/1 array carries one
operating point and no ranking. **This is a property of the paper's stated output, not of the
port**, and it is why `score_dms` exists (BASELINE_RESULTS.md:241-248).

**Fusion buys almost nothing over the best single modality.** `score_dms` edges out
`score_fused` on pooled ROC on all three original corpora, and on HateMM the **audio branch
alone (0.7777) beats the fused branch (0.7504)**. Learned importance weights averaged over test
videos are 0.414/0.188/0.398 (visual/audio/text) on HateMM, 0.309/0.263/0.428 on MHC EN,
0.526/0.157/0.318 on MHC ZH — on HateMM the block puts its *lowest* weight on the modality whose
branch scores best, so the Dynamic Modality Selection block is not selecting well
(BASELINE_RESULTS.md:250-259).

The cross-method comparison on the three original corpora is **not clean, and the reason is the
input, not the method**: MultiHateLoc sees audio and text; VadCLIP and DSANet see only CLIP
frames. A three-modal model beating two visual-only models is mostly a statement about VGGish
and Whisper (BASELINE_RESULTS.md:232-239).

---

## 3. Reference rows — NOT weakly supervised, included only for context

**These three do not train on video labels and must never be described as weakly-supervised
baselines.**

| Method | Actual supervision | corpus | pooled ROC | pooled PR | within-hate macro (n) | video AUC | line |
|---|---|---|---|---|---|---|---|
| Vad-R1 (NeurIPS'25) zero-shot arm | none on our data; **released 7B ckpt trained on its own VAD corpus** | HateMM | 0.5696 | 0.2722 | 0.5000 (85) | 0.5288 (balanced acc) | :663, :890 |
| " | " | MHC EN | 0.5427 | 0.2699 | 0.5000 (44) | 0.5247 | :664 |
| " | " | MHC ZH | 0.5987 | 0.2838 | 0.5000 (7) | 0.6427 | :665 |
| " | " | HateClipSeg | **0.6382** | 0.6115 | 0.5001 (67) | 0.6826 | :1150 |
| Vad-R1 term-adaptation arm ("hateful" swapped for "abnormal") | same ckpt, owner-approved ablation | HateMM | 0.5750 | 0.2740 | 0.5000 (85) | 0.5451 (bal. acc) | :777 |
| " | " | MHC EN | 0.6053 | 0.3046 | 0.5005 (44) | 0.5778 | :779 |
| " | " | MHC ZH | 0.7262 | 0.3721 | 0.5000 (7) | 0.7272 | :781 |
| " | " | HateClipSeg | 0.5655 | 0.5610 | 0.5001 (67) | 0.6196 | :1151 |
| EventVAD (MM'25) | none, training-free (reimplemented from paper) | HateMM | 0.5174 | 0.2519 | 0.4988 (85) | 0.4519 | :973 |
| " | " | MHC EN | 0.5041 | 0.2568 | 0.4784 (44) | 0.5179 | :974 |
| " | " | MHC ZH | 0.5202 | 0.2440 | 0.4923 (7) | 0.5623 | :975 |
| " | " | HateClipSeg | — | — | — | **not run** | :1152 |
| Ours (masked packed locator, 1 fwd/video) | **zero labels, no training** | HateMM | 0.7451 | 0.5601 | 0.5706 (85) | 0.9010 | :885 |
| " | " | MHC EN | 0.6198 | 0.4141 | 0.6154 (44) | 0.7015 | :507, :902 |
| " | " | MHC ZH | 0.6004 | 0.3813 | 0.6076 (7) | 0.6153 | :508, :914 |
| " | " | HateClipSeg | 0.5414 | 0.6149 | 0.5324 (67) | 0.7536 | :1133 |

Vad-R1's video-level "AUC" is the **balanced accuracy `(TPR+TNR)/2`, not a ranking statistic**,
because its verdict is binary (BASELINE_RESULTS.md:670-673). Its frame scores are likewise
binary, so its ROC curve has a single interior operating point and its AUC is coarse by
construction and **not resolution-comparable to a continuous scorer's AUC**
(BASELINE_RESULTS.md:650-657).

---

## 4. Known negative results and warnings

**(a) On HateClipSeg — the only corpus that can actually measure localization — nothing
localizes.** All **16 supervised branch-cells** score within-hate macro in the band
**0.4709 – 0.5583** (0.4709 = MultiHateLoc `score_audio`, :1147; 0.5583 = DSANet
`score_mlp`/`score_refined`, :1136-1137). Five of the sixteen fall *below* chance, all five of
them MultiHateLoc's. The zero-label locator sits at 0.5324, inside the same band
(BASELINE_RESULTS.md:1161-1170). The source calls this the study's cleanest negative: on the one
corpus whose annotation supports a within-video measurement over 85 % of its cohort, **no method
— trained or zero-label, audio, visual, text or fused — rises meaningfully above chance at
saying where inside a hateful video the hate is** (BASELINE_RESULTS.md:1193-1199).

**(b) Vad-R1 emits a whole-clip interval 100 % of the time.** Across all 525 videos of the three
original corpora, every positive answer placed the event over the entire clip: 126/126 on
HateMM, 46/46 on MHC EN, 54/54 on MHC ZH, all `[0.0, 1.0]` up to a rounding that occasionally
writes `0.999`. Its within-hate macro is therefore pinned at the tie value 0.5000 with sd
0.0000, and its frame row is a **video-level verdict broadcast across the timeline**
(BASELINE_RESULTS.md:687-708). The term-adaptation arm does not recover a single sub-interval:
152/152, 65/65, 71/71 whole-clip (BASELINE_RESULTS.md:801-812). On HateClipSeg the same
degeneracy holds, macro 0.5001 in both arms (BASELINE_RESULTS.md:1212, 1217-1222).
**Consequence:** Vad-R1 topping HateClipSeg's pooled column at 0.6382 is the clearest single
demonstration that pooled frame ROC on such a corpus is a video-level metric wearing a
frame-level name (BASELINE_RESULTS.md:1180-1185).

**(c) EventVAD was not run on HateClipSeg** — owner default, not oversight: it read the floor
(0.50–0.52 pooled) on all three prior corpora while spending 6.4k MLLM calls and 13.4 GPU-hours,
with 40 % of events returning no parseable score under the paper's own prompt
(BASELINE_RESULTS.md:1154-1157). On the corpora where it did run, 301 of 525 videos carry a
**constant score array** (BASELINE_RESULTS.md:1030-1036, 1060).

**(d) LAVAD has adapter code only and was never run.** `scripts/reproduction_baselines/lavad/`,
`run_lavad.sh`, `smoke_cpu_lavad.py` and `DESIGN_LAVAD.md` exist in the Hate-follow-up checkout
(HEAD commit is literally "Add LAVAD baseline adapter and evaluation plumbing"), but **no LAVAD
row appears anywhere in BASELINE_RESULTS.md**. It is classified training-free and excluded from
retraining in BASELINE_REPRODUCTION_LIST.md:67-68, 88-89.

**(e) MHC ZH's within-hate macro rests on 7 videos and must not be leaned on.** 36 of the 43
hateful ZH gold videos are annotated hateful for their entire duration, leaving no within-video
ranking to score; the macro's sd is around 0.4 (BASELINE_RESULTS.md:136-142, 906).

**(f) Two rows are one-epoch models.** MACIL-SD on HateMM selects epoch 1 — validation AP peaks
at 0.8586 on the first epoch and never recovers it across the remaining 49
(BASELINE_RESULTS.md:435, 458-462). MultiHateLoc on HateClipSeg selects epoch 1 of 100
(BASELINE_RESULTS.md:1328, 1333-1339). Neither should be read as a converged result. On
HateClipSeg the selection criterion is barely discriminating at all, since 87 % of videos are
positive so validation video AP starts near 0.9.

**(g) MACIL-SD's fused MIL head barely trains on MultiHateClip:** `cls` loss 0.6035 → 0.5624 on
EN and 0.6097 → 0.6029 on ZH over 50 epochs, against 0.6788 → 0.1752 for the uni-modal model on
the same features. Reported as measured; no rerun, no hyperparameter change
(BASELINE_RESULTS.md:448-456).

**(h) One internal inconsistency in the source, flagged not silently fixed.** The consolidated
MHC ZH table gives VadCLIP `video AUC = 0.3981` (BASELINE_RESULTS.md:917), but 0.3981 is the
**`score_align`** branch's video AUC (:33); the `score_mlp` branch — whose pooled 0.5676,
PR 0.2705 and macro 0.3562 that same consolidated row quotes — has video AUC **0.5537** (:32).
The consolidated row therefore mixes two branches in one line. Every other consolidated row
checked (:883-889, :897-901, :910-916) matches its per-method table exactly. **Use the
per-method tables in §2 as authoritative.**

**(i) The consolidated tables pick the primary branch, which is not always the best branch.**
E.g. DSANet MHC ZH is listed at 0.5749 (`score_mlp`, :916) while its `score_align` branch reads
0.5904 (:42) — and the narrative at :406-408 quotes 0.5904 as DSANet's MHC ZH number. Both are
correct for their branch; state which branch when quoting.

**(j) The primary source's own caveat about upstream comparability:** because test-selected
checkpointing was removed, these numbers are systematically *lower* than what the upstream
protocols would report. `--val-frac 0 --select last` restores upstream behaviour if a
strictly-as-published number is ever needed (BASELINE_RESULTS.md:144-149).

---

## 5. Unreproduced general WSVAD candidates — zero implemented

From `Hate-follow-up/docs/duplex/BASELINE_REPRODUCTION_LIST.md` (2026-08-18), ranked by
reproduction priority = code health × task closeness × influence. Scope of that list: video-level
labels only, no frame/timestamp supervision, venue CCF-A or CORE A*, open source with training
code, every repo URL fetched and HTTP-200 verified on 2026-08-18
(BASELINE_REPRODUCTION_LIST.md:1-8, 22-24).

**None of the following has been implemented or run.** The four weakly-supervised methods
actually reproduced (VadCLIP #4, MACIL-SD #6, plus DSANet and the MultiHateLoc reimplementation
from the addenda) are struck from this list; everything below remains open.

| rank | Method | Venue | Features | Blocker / note | line |
|---|---|---|---|---|---|
| 1 | **UR-DMU** | AAAI'23 | I3D 10-crop | Top pick: only modern MIL repo *with* feature-extraction code included; shortest path | :26 |
| 2 | **RTFM** | ICCV'21 | I3D 10-crop | De-facto standard; basis of UR-DMU / S3R / MACIL-SD; porting it unlocks half the table | :27 |
| 3 | **HL-Net / XDVioDet** | ECCV'20 A* | I3D RGB + VGGish | Only A* audio-visual weakly-sup per-segment scorer; matches the audio-carries-hate finding; sparse README | :28 |
| 5 | **PEL4VAD** | TIP'24 | I3D + CLIP text prompts | Best engineering (ckpts + logs); intended as pipeline correctness anchor | :30 |
| 7 | **MGFN** | AAAI'23 | I3D 10-crop | Train/test complete; scrappy README | :32 |
| 8 | **DeepMIL** | CVPR'18 | C3D/I3D/R3D | ekosman fork = only full video→features→train chain | :33 |
| 9 | **S3R** | ECCV'22 A* | I3D | Dictionary-learning extra step; old env | :34 |
| 10 | **DELU** | ECCV'22 A* | THUMOS I3D two-stream | Healthiest WTAL repo; **blocker = optical flow** | :35 |
| 11 | **P-MIL** | CVPR'23 | two-stream + proposals | Needs upstream CO2-Net proposals first — double porting | :36 |
| 12 | **CO2-Net** | MM'21 | two-stream | WTAL base; oldest env (torch 1.3) | :37 |
| 13 | **UMIL** | CVPR'23 | none (end-to-end X-CLIP) | Skips feature extraction; needs apex→torch.amp surgery | :38 |
| 14 | **VERA** | CVPR'25 | raw frames + InternVL2-8B | Closest to an MLLM narrative; per-segment VLM calls are expensive | :39 |
| 15 | **MIST** | CVPR'21 | h5py features | Author warns "may have unknown bugs"; high friction | :40 |
| 16 | **W-TALC** | ECCV'18 A* | I3D/UNT | torch 0.4.1; **historical anchor only, do not spend porting time** | :41 |
| — | **PVLR** | MM'24 | I3D two-stream + CLIP RN50 | Video-level confirmed, but its mechanism aligns action *class names*; a binary hate label collapses the text side and the paper's story mostly evaporates | :91-98 |
| — | **Fed-WSVAD** | AAAI'25 | same as DSANet | Near-free (`--clients_num 1` degenerates to centralized) but the client partition is a free parameter reviewers will attack | :125 |
| — | **VADTree** | NeurIPS'25 | training-free, 4-model stack | Complete but multi-day setup — explicitly not recommended | :128 |

Interchangeable WTAL extras, all verified, pick at most one: CoLA (CVPR'21), ASM-Loc (CVPR'22),
DDG-Net (ICCV'23) (:43-45).

**Cross-cutting blocker for this whole list** (BASELINE_REPRODUCTION_LIST.md:10-20): model bodies
are tiny (0.3M–20M params, minutes to train), so the real cost is **feature extraction** — I3D
10-crop ≈ 4–8 h for HateMM's 43 h of video; I3D+TV-L1 two-stream (the whole WTAL line) adds tens
of hours of optical flow, which is that family's main practical blocker. Environment trap: most
repos pin torch 1.3–1.8 / CUDA 10–11, so everything needs a port to torch ≥ 2.7 + cu128 for
Blackwell (sm_120); apex-dependent repos (MIST, UMIL) need apex→torch.amp surgery.

**Verified exclusions** (do not re-litigate): empty-shell repos PE-MIL (CVPR'24), MSL (AAAI'22),
TDSD (MM'24), CU-Net (CVPR'23); no official repo STPrompt (MM'24), TPWNG (CVPR'24), LEC-VAD
(ICML'25); **stronger-than-video-level supervision** HolmesVAU (CVPR'25, segment/event
instructions), Hawk (NeurIPS'24), GlanceVAD (point), TE-TAD / UniMD / DiGIT / DyFADet
(timestamp); training-free (nothing to retrain) LAVAD (CVPR'24), LELA; framework-dead
UntrimmedNet (Caffe+Matlab), GCN label-noise cleaner (CVPR'19)
(BASELINE_REPRODUCTION_LIST.md:58-72, 84-89). 2025–2026 empty shells: LAVIDA (CVPR'26), PANDA
(NeurIPS'25), DualExplore (CVPR'26), RefineVAD (AAAI'26), PI-VAD (CVPR'25), Anomize (CVPR'25),
TLMA (CVPR'26) (:131-139). Note for the WSVAD line generally: **no A-venue WTAL entry with code
exists for 2025–2026; the freshest with code is PVLR (MM'24)** (:138-139).

---

## 6. Verification pass

Five numbers were re-checked against `BASELINE_RESULTS.md` after this file was written:

| # | Claim in this file | Source line | Verbatim source text | Match |
|---|---|---|---|---|
| 1 | VadCLIP MHC ZH `score_align` pooled ROC = 0.3880 | :33 | `| VadCLIP | mhclip_zh | score_align | 0.3880 | 0.1806 | 0.3225 (7) | 0.3981 |` | ✓ |
| 2 | MultiHateLoc HateMM `score_audio` pooled ROC = 0.7777 | :193 | `| MultiHateLoc-reimpl | hatemm | score_audio | 0.7777 | 0.5115 | 0.6106 (85) | 0.8156 |` | ✓ |
| 3 | MACIL-SD audio-only HateMM = 0.7667 / 0.4939 / 0.5966 (85) / 0.7814 | :391 | `| MACIL-SD audio-only | hatemm | score_mil | 0.7667 | 0.4939 | 0.5966 (85) | 0.7814 |` | ✓ |
| 4 | DSANet HateClipSeg `score_mlp` within-hate macro = 0.5583 (67), the best of the 16 supervised cells | :1136, :1164 | `| DSANet | video labels | score_mlp | 0.5387 | 0.5502 | **0.5583** (67) | 0.5493 |` and "The best cell in the study is DSANet's MIL branch at 0.5583" | ✓ |
| 5 | HateClipSeg supervised within-hate macro band = 0.4709–0.5583 across 16 branch-cells | :1161-1166 | "runs from 0.4709 to 0.5583 across the sixteen supervised branch-cells" — independently confirmed by counting rows :1134-1149 (VadCLIP 2 + DSANet 3 + MACIL-SD 5 + MultiHateLoc 6 = 16) and by min/max over that column | ✓ |

One discrepancy found in the source and recorded in §4(h) rather than corrected: the
consolidated MHC ZH VadCLIP row (:917) reports the `score_align` video AUC (0.3981) alongside
the `score_mlp` pooled/PR/macro figures.
