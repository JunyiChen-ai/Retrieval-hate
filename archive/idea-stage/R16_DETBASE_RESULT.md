# R16-DETBASE — result

**Date** 2026-08-18 · **Freeze** `idea-stage/R16_DETBASE_FREEZE.md`, commit **`94142ef`**,
committed before `scripts/r16_detbase/run_af.py` had ever been run on the full feature set ·
**Cost** ¥0 (no cloud, no API, no annotation), local RTX 5090 · **Wall** 4-FPS CLIP extraction
17 min (3 shards), dense audio+text 2 min, each ActionFormer arm 12-14 min for 3 seeds ·
**Seeds** 5100/5101/5102, 3 per arm, mean ± sd throughout.

## VERDICT

**The 29-point gap is not a gap in our detector. It is mostly a gap in what "one ground-truth
instance" means, and the rest is corpus and split.**

- Reproducing the paper's ActionFormer under **the project's historical ground-truth convention**
  (adjacent offensive segments merged into one block) gives test **F1@tIoU 0.5 = 21.00 ± 1.47** —
  barely above our own per-window score curve's 19.71 ± 0.91. Under that convention nothing
  reproduces, and the frozen S1 bar (≥ 40) fails.
- Reproducing it under **the convention the paper's own numbers imply** (every offensive segment
  is its own instance) gives test **38.22 ± 1.63** visual-only and **42.02 ± 0.73** with audio and
  text added, against the published 52.65 — same regime, on a 90.8% subset with 60% of the corpus
  in training instead of 80%. Our own score curve on that same convention scores **10.68 ± 0.54**.
- So on matched ground truth the detector is worth **+27.5 to +31.3 F1 points** over the test bed
  this project has been using, and the reason is measurable: the detector's proposal pool recalls
  **90.7%** of gold segments at tIoU 0.5, the score-curve decoder's entire reachable pool recalls
  **25.6%**.

**S1 fails on the frozen primary convention; S2 is met — the mechanism is identified and measured.**

---

## 1. What the paper actually specifies (arXiv 2508.01712v2, §3.1-3.2, §4.1-4.2, Table 4)

| item | paper |
|---|---|
| foreground classes | **one**. "All segments originally labeled as *hateful, insulting, sexual, violent* or *self-harm* are merged into a single *offensive* category, resulting in a binary classification scheme used throughout all downstream tasks." Normal = background. |
| visual features | frozen ViT-Large per timestamp; **moment rate 4 FPS**; ActionFormer trained **30 epochs** |
| text features | frozen BERT-Base over `[t−2s, t]` |
| audio features | frozen Wav2Vec-Emotion over `[t−4s, t]` |
| multimodal | **late fusion only** — "ActionFormer supports only unimodal visual features; thus we conduct unimodal experiments and derive multimodal results via late fusion", aligning non-visual predictions to their nearest visual counterparts, averaging boundaries, majority-voting labels |
| split | 80% train / 20% test, **no val split**, 30 epochs fixed |
| metric | tIoU ∈ {0.3, 0.5, 0.7}; Accuracy, Precision, Recall, F1 **for the offensive class only** |

**The metric is confirmed to be the same object we compute.** Table 4's four columns are not
independent: for visual at tIoU 0.5, P = 40.52 and R = 75.14 give F1 = 2PR/(P+R) = **52.65**
(printed 52.65) and Jaccard tp/(tp+fp+fn) = **35.73** (printed as Acc 35.73). The same identity
holds at 0.3 (45.70 / 84.74 → F1 59.38, Acc 42.22) and at 0.7. So their "Acc" is the Jaccard of
the same one-to-one proposal match, and their F1@tIoU is exactly
`scripts/r16_detbase/eval_f1.py:match_prf`. No metric-definition ambiguity remains.

**What the paper does not specify, and what turns out to decide everything:** whether two
*adjacent* offensive segments are one ground-truth instance or two. It merges the five offensive
*labels*; it never says it merges neighbouring segments. §3 below shows the answer is *two*.

## 2. Protocol differences, stated plainly

| item | paper | ours |
|---|---|---|
| corpus | 435 videos / 11,714 segments | **395 / 10,572** — the 90.8% surviving subset, non-random attrition (`DATASET_hateclipseg.md §4`) |
| split | 80 / 20, no val | **237 / 39 / 119** (60/10/30), frozen `p11_split.json`, reused unchanged |
| training videos | ~348 | **237** |
| visual encoder | "ViT-Large", checkpoint unnamed | `openai/clip-vit-large-patch14-336`, `pooler_output`, 1024-d — the tower this project uses everywhere else |
| epoch / threshold | fixed 30 epochs, no threshold stated | epoch and one global proposal-score threshold selected on the 39-video val split, then applied unchanged to test |
| multimodal | late fusion of per-modality detectors | **early fusion** — V ⊕ Wav2Vec-Emotion ⊕ BERT concatenated into one ActionFormer (2816-d). A different system; labelled as such |
| audio windowing | model re-run per `[t−4s, t]` window | encoder run once per video, hidden states mean-pooled per window (16× cheaper, equivalent up to per-input normalisation) |
| text windowing | words in `[t−2s, t]` | Whisper gives chunk-level timestamps, so the ASR chunks *overlapping* `[t−2s, t]` |

One video, `yt_NzvfkIYS5Yg` (test split), is a truncated download with zero decodable frames. The
project's existing K=30 cache already stores it as an all-zero row, so it is given an all-zero
dense feature array too and both systems see the identical 119 test videos. It contributes 6 gold
blocks / 9 gold segments that neither system can ever recall.

## 3. The ground-truth instance convention is the main term in the gap

HateClipSeg's annotation tiles each video with contiguous segments of mean 8.88 s. Two readings:

| convention | test instances | mean length | who uses it |
|---|---|---|---|
| `blocks` — maximal runs of adjacent offensive segments merged | 359 (3.02/video) | 37.5 s | **every number this project has ever reported** (`recon_decode.py:blocks_of`) |
| `rawseg` — every offensive segment its own instance | 1474 (12.39/video) | 9.1 s | the paper (evidence below) |

Both were pre-declared in the freeze, `blocks` as primary. The measurement:

**Test split, 119 videos, F1@tIoU, mean ± sd over 3 seeds.**

| system | GT | tIoU 0.3 | tIoU 0.5 | tIoU 0.7 |
|---|---|---|---|---|
| per-window score curve, V⊕T⊕O⊕A (project test bed) | blocks | 34.99 ± 1.29 | 19.71 ± 0.91 | 8.85 ± 0.32 |
| per-window score curve, visual only | blocks | 16.39 ± 6.35 | 9.12 ± 3.11 | 3.72 ± 1.55 |
| **ActionFormer, visual only** | blocks | 39.40 ± 1.03 | **21.00 ± 1.47** | 7.60 ± 0.65 |
| per-window score curve, V⊕T⊕O⊕A | rawseg | 20.55 ± 2.23 | 10.68 ± 0.54 | 3.36 ± 0.61 |
| **ActionFormer, visual only** | rawseg | 50.28 ± 1.82 | **38.22 ± 1.63** | 19.95 ± 0.09 |
| **ActionFormer, V ⊕ audio ⊕ text (early fusion)** | rawseg | 54.72 ± 0.69 | **42.02 ± 0.73** | 21.11 ± 0.86 |
| *paper, ActionFormer visual, 435 videos, 80/20* | *?* | *59.38* | *52.65* | *30.99* |

Val split (39 videos), same runs, for reference: ActionFormer visual `rawseg` 57.59 / 45.73 /
24.08; V⊕A⊕T `rawseg` 60.81 / 48.76 / 23.88.

**Three independent reasons the paper is on `rawseg`:**

1. **Magnitude.** Only `rawseg` puts a faithful ActionFormer in the paper's range. On `blocks`
   the same code, features and hyper-parameters land at 21.
2. **Operating point.** The paper runs precision 40.52 against recall 75.14 at tIoU 0.5 — about
   1.85 predictions per gold instance, recall far above precision. Our `rawseg` runs reproduce
   that shape (P 31.30, R 51.94, 1.71 predictions per gold). Our `blocks` runs cannot: at
   3.02 gold instances per video the detector emits 4.71 and still only recalls 26.9%.
3. **Prediction geometry.** ActionFormer trained on `rawseg` emits 21.1 intervals per video with
   median length 9.3 s against a gold median of 8.4 s; trained on `blocks` it emits intervals of
   median 39.0 s against a gold median of 21.0 s and never fits the long tail.

**Consequence for this project's record.** Every localization number this project has published —
21.6 / 23.8 / 25.5 F1@tIoU0.5, the 16.0 matched-protocol figure, the "87.6 grid ceiling" — is on
`blocks`, i.e. on a *different and harder task* than the 52.65 it was being compared against.
The comparison that motivated three rounds of method work was not like-for-like.

## 4. Where the remaining gap is (42.02 vs 52.65)

Post-hoc, descriptive, `scripts/r16_detbase/run_af_trainval.py`.

- **Training-set size — measured, and it is not the explanation.** Retraining on train+val
  (276 videos, +16%) moves F1@tIoU0.5 from 38.22 to 37.63, inside seed noise (§7).
- **Corpus attrition** — 395/435 videos, and attrition is worst in the rarest and most severe
  strata (harm 72%, sexual 86% video coverage). Not separable without the missing 40 videos.
- **Encoder** — the paper names only "ViT-Large"; we use CLIP ViT-L/14-336's pooler output.
- **Threshold transfer is not a factor.** Sweeping the score threshold directly on test raises
  `blocks` F1@0.5 from 21.00 to 21.30 and nothing more, so the val-chosen operating point is
  essentially optimal already.

## 5. Diagnostics: what the detector buys, and what it does not

`scripts/r16_detbase/diagnose.py`, test split, 3 seeds, post-hoc and descriptive.

### 5.1 The whole advantage is candidate generation

**Proposal-pool recall** — the fraction of gold instances that *some* proposal in the pool
localizes at that tIoU, ignoring precision entirely. For the score curve, "the pool" is every
interval its decoder could emit anywhere on its whole threshold × smoothing × merge-gap grid.

| pool | GT | tIoU 0.3 | tIoU 0.5 | tIoU 0.7 | pool size |
|---|---|---|---|---|---|
| score-curve decoder | blocks | 72.79 | 53.30 | 30.36 | 3 393 |
| ActionFormer | blocks | 96.10 | **84.49** | 46.24 | 23 800 |
| score-curve decoder | rawseg | 49.93 | **25.62** | 8.96 | 3 393 |
| ActionFormer | rawseg | 97.42 | **90.68** | 64.97 | 23 800 |

At the segment level the 30-window grid is not merely worse, it is **structurally incapable**: it
can reach a quarter of the gold segments, and no scoring improvement on top of it can ever exceed
that. This is the hard reason the project's test bed was never going to produce a competitive
localization number.

### 5.2 Boundary precision is worth nothing at block level and ~5 points at segment level

Snapping ActionFormer's own proposals to the same 30-window grid edges the score curve is forced
onto:

| GT | ActionFormer as-is | grid-snapped | cost |
|---|---|---|---|
| blocks | 21.00 | 21.00 | **0.00** |
| rawseg | 38.22 | 32.67 | **−5.55** |

So sub-window boundary regression is not where the detector's advantage lives. Its advantage is
*which* spans it proposes and how many.

### 5.3 The score curve's intervals mostly do not overlap the right thing

Snapping each of the score curve's decoded intervals to its best-overlapping gold instance
(oracle boundary repair — the ceiling of any post-processing that only moves boundaries):

| GT | curve as-is | oracle boundaries | ceiling |
|---|---|---|---|
| blocks | 19.71 | 52.47 | perfect boundaries would triple it |
| rawseg | 10.68 | 27.56 | still less than ActionFormer's 38.22 |

### 5.4 Ranking, not localization, is the residual bottleneck for both systems

Hold the pool fixed, keep the top-k proposals per video, and compare ranking by the model's own
score against ranking by oracle tIoU (F1@tIoU 0.5):

| pool | GT | rank by | k=1 | k=2 | k=3 | k=5 |
|---|---|---|---|---|---|---|
| ActionFormer | blocks | model | 11.2 | 16.0 | 19.2 | 20.2 |
| ActionFormer | blocks | **oracle** | 38.1 | 51.1 | **56.1** | 53.2 |
| ActionFormer | rawseg | model | 4.4 | 8.3 | 11.8 | 17.5 |
| ActionFormer | rawseg | **oracle** | 12.9 | 23.1 | 30.8 | **42.6** |
| score curve | blocks | model | 8.3 | 9.3 | 11.3 | 11.8 |
| score curve | blocks | **oracle** | 37.4 | 43.4 | 41.8 | 37.0 |

Read the `blocks` rows: ActionFormer's proposal pool already **contains** a 56-point solution, and
its own classifier extracts 19-21 of it. On `blocks`, localization is solved and *deciding which
span is offensive* is the entire remaining problem — which is the same conclusion rounds 11-15
reached from within-video AUC (0.59), now reproduced from the detector side on a competent test bed.

### 5.5 Prediction geometry

| system | GT | intervals/video | median length | gold median |
|---|---|---|---|---|
| gold | blocks | 3.02 | 21.0 s | — |
| score curve | blocks | 2.10 | 36.7 s | 21.0 s |
| ActionFormer | blocks | 4.71 | 39.0 s | 21.0 s |
| gold | rawseg | 12.39 | 8.4 s | — |
| score curve | rawseg | 3.16 | 9.7 s | 8.4 s |
| ActionFormer | rawseg | 21.14 | 9.3 s | 8.4 s |

## 6. What is now established

1. **The test bed was not merely weak, it was scoring a different task.** The project's `blocks`
   convention is a harder problem than the paper's, and the 29-point comparison that framed
   rounds 13-15 was between two different ground truths.
2. **On matched ground truth a real detector is worth +27 to +31 F1 points** over the per-window
   score curve, and §5.1 says exactly why: candidate coverage, 90.7% vs 25.6% pool recall at
   tIoU 0.5.
3. **The K=30 window grid is a hard ceiling for segment-level localization**, not a tunable one.
   Nothing built on it can pass 25.6 recall at tIoU 0.5, whatever the scores.
4. **On the harder `blocks` task, localization is not the bottleneck at all** — an oracle *ranking*
   of ActionFormer's own proposals scores 56.1 where its classifier scores 19.2.
5. **Fusion helps on this substrate, contrary to the paper's late-fusion result.** Early fusion of
   audio and text into the detector is +3.8 F1@tIoU0.5 over visual-only (42.02 vs 38.22), where the
   paper's late fusion *lost* 1.7 points (50.92 vs 52.65). The project's own audio and text channels
   are informative; the paper's fusion mechanism, not the modalities, is what failed there.
6. **The published 52.65 is now bracketed rather than mysterious.** A faithful re-implementation on
   a 90.8% subset with 60% of the corpus in training reaches 42.02 with the same metric and the same
   4 FPS moment rate; the residue is corpus, split and encoder.

## 7. Post-hoc arm: 70% training share

`scripts/r16_detbase/run_af_trainval.py`, rawseg, visual only, train+val = 276 videos, epoch count
and threshold carried over per seed from the train-only run (no new selection surface).

| training videos | tIoU 0.3 | tIoU 0.5 | tIoU 0.7 |
|---|---|---|---|
| 237 (train only) | 50.28 ± 1.82 | 38.22 ± 1.63 | 19.95 ± 0.09 |
| 276 (train + val) | 48.59 ± 2.10 | **37.63 ± 1.09** | 21.30 ± 0.30 |

**A 16% larger training set buys nothing** at tIoU 0.3 and 0.5 (both changes are inside seed
noise) and about +1.3 at tIoU 0.7. Extrapolating, the paper's 47% larger training set is not a
plausible explanation for a 10-point residual. Caveat: this arm reuses the epoch count selected
on val by the train-only run and cannot re-select it, which is the honest price of not having a
second held-out split; it slightly disadvantages the larger-data arm.

## 7b. Is the method line worth reopening on this base?

Stated as a recommendation, not a decision.

**Yes for the segment-level (`rawseg`) task, and the entry point is now specific.** The base is
competent: 42.02 F1@tIoU0.5 on test with three seeds and sd 0.73, inside the published regime, on
a substrate we own end to end and can retrain in 4 minutes per seed. Two things about it are
concrete openings rather than hopes:

- **Fusion is live again on this base.** Early fusion is +3.8 over visual-only here, where the
  paper's late fusion was −1.7. The modality-combination question that rounds 11-15 could not move
  on a per-window head *does* move on a detector. Everything the project learned about its audio,
  OCR and ASR channels is directly reusable, and none of it has been tried inside a detector.
- **The bottleneck is now named and it is not localization.** §5.4: with ActionFormer's own
  proposals, oracle *ranking* scores 42.6 where its classifier scores 17.5 (rawseg, k=5), and
  56.1 vs 19.2 on blocks. A method that improves *span classification* — which is what this
  project's retrieval, stance and memory ideas are actually about — now has a place to attach
  where the gain is 20+ points wide instead of the 2-4 points the decode axis was priced at.

**No for anything built on the 30-window score curve.** §5.1 is a hard bound, not a difficulty:
that grid's entire reachable interval set covers 25.6% of gold segments at tIoU 0.5. Rounds 11-15
were optimizing scores inside a container that caps out below a quarter of the task. That
explains their nulls without needing any of the mechanisms those rounds proposed and rejected —
and it also means those nulls do **not** transfer as evidence about the mechanisms themselves.
Three families (temporal architecture, within-video objectives, nuisance suppression) were killed
on a substrate that could not have shown a gain regardless. Whether they deserve a re-test on the
detector base is a scope question for the user, not something this round decides.

**One caution before any method round.** The reproduction only lands in the paper's regime under
`rawseg`, whose boundaries are Whisper sentence boundaries inside otherwise homogeneous offensive
stretches. Part of what a detector is being rewarded for there is predicting *speech pauses*, not
hate. That should be measured (e.g. an audio-onset-only baseline) before a method claim rests on
`rawseg` numbers.

## 8. Deviations

- **D1 — a plumbing smoke run touched test.** Before the real run, `run_af.py` was executed once
  with `--epochs 1 --seeds 9999 --touch-test` on the partially extracted feature set, purely to
  verify that the post-selection test branch does not crash after 40 minutes of training. It
  produced test F1@tIoU0.5 = 7.30 from a 6-epoch untrained model. No threshold, epoch, arm, seed
  or decision rule was derived from it, and it is reported here rather than buried. The project's
  own rule ("the cheapest check before submitting is the submission itself") is what motivated it.
- **D2 — one test video has no decodable frames.** `yt_NzvfkIYS5Yg` is a truncated download
  (135 KB for 274 s of declared content). It is given an all-zero dense feature array so that both
  systems evaluate on the identical 119 videos, matching the pre-existing all-zero row in the K=30
  cache. It costs both systems the same recall.
- **D3 — the multimodal arm is early fusion, not the paper's late fusion.** Declared in the freeze
  in advance. It is a different system and is labelled as one everywhere above.
- **D4 — audio windowing is pooled, not re-encoded per window.** Declared in
  `scripts/r16_detbase/extract_dense_at.py`; running the emotion model on 380k overlapping 4 s
  windows costs 16× the audio for a quantity that differs only by the model's per-input
  normalisation.
- **D5 — the `blocks` primary was the wrong primary.** The freeze designated `blocks` as the
  convention on which S1 is judged, because it is the one every prior project number uses. That
  choice is honoured: **S1 fails.** The `rawseg` arm, pre-declared in the same freeze as a
  protocol-sensitivity arm, is what carries the reproduction, and §3 gives the three independent
  reasons it is the paper's convention. No threshold, epoch or hyper-parameter was changed between
  the two arms.
- **D6 — `h5py` was missing from the environment** and had to be installed for the vendored
  ActionFormer's ActivityNet dataset module to import. No other torch 2.7.1+cu128 incompatibility
  appeared: the repo's `nms_1d_cpu` C++ extension compiles unmodified.

## 9. Reproduction

| artifact | path |
|---|---|
| freeze (pre-code, commit `94142ef`) | `idea-stage/R16_DETBASE_FREEZE.md` |
| vendored ActionFormer (`61ea7eb`, 2024-04-10) | `third_party/actionformer/` |
| dataset adapter / config | `third_party/actionformer/libs/datasets/hateclipseg.py`, `configs/hateclipseg_clip.yaml` |
| 4-FPS CLIP-L/336 features | `scripts/r16_detbase/extract_dense_clip.py` → `data/CLIP_Embedding/HateClipSeg/dense4fps_clipL336/` (1.5 GB) |
| 4-FPS audio / text features | `scripts/r16_detbase/extract_dense_at.py` → `dense4fps_w2vemo/`, `dense4fps_bertbase/`, fused `dense4fps_vat/` |
| annotation builder | `scripts/r16_detbase/make_af_json.py` |
| detector runner | `scripts/r16_detbase/run_af.py` → `idea-stage/r16_detbase/out/res_{v_blocks,v_rawseg,vat_rawseg}.json` |
| score-curve test bed on test | `scripts/r16_detbase/curve_baseline.py` → `curve_baseline_{ALL,VIS}_{blocks,rawseg}.json` |
| metric | `scripts/r16_detbase/eval_f1.py` |
| diagnostics | `scripts/r16_detbase/diagnose.py` → `diagnostics_{blocks,rawseg}.json` |
| logs | `logging/runs/r16_{extract,extract_at,af,curve}/` |
