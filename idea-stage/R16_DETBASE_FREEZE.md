# R16-DETBASE — freeze (baseline reproduction, not a method verdict)

**Date** 2026-08-18 · **Scope** reproduce the HateClipSeg paper's ActionFormer temporal
localization baseline on our 395-video subset, to establish whether this project's
"per-window score curve + threshold decode" test bed was ever a competent detector test bed.
**This is a baseline reproduction round.** It carries no method claim, no candidate, no KILL
rule against any idea. The ceremony below is deliberately light, per the 2026-08-05 ruling
that ceremony scales with the cost of a wasted run.

Committed **before** any number from `scripts/r16_detbase/run_af.py` exists.

---

## 1. Why

Every temporal-localization number this project has produced sits on one substrate: 30 uniform
windows per video, a 2-layer per-window head on frozen features, a threshold-plus-merge
decoder. Best F1@tIoU 0.5 ever recorded on that substrate is **23.8**; the round-13/14/15
matched-protocol figure is **16.0**. The dataset paper (arXiv 2508.01712, ACM MM '25) reports
**52.65** with ActionFormer on visual features. A 29-point gap between our test bed and the
published baseline means that every "the method does not move the number" verdict this project
has recorded on localization was measured on a test bed that may simply be incapable. The bench
is checked before any further method work is priced.

## 2. The paper's protocol, and where ours differs

Read from arXiv 2508.01712v2 §3.2, §4.1, §4.2, Table 4. Differences are recorded, not hidden;
the target is the **same order of magnitude (40-55 F1@tIoU 0.5)**, not the exact digit.

| item | paper | ours | why |
|---|---|---|---|
| corpus | 435 videos, 11,714 segments | **395 / 10,572** (90.8% surviving subset) | non-random platform attrition; `DATASET_hateclipseg.md §4` |
| split | 80% train / 20% test, no val | **237 / 39 / 119** (60/10/30), frozen `p11_split.json` | our split predates this round and is reused unchanged; we need a val split for epoch and threshold selection, the paper needs none because it fixes 30 epochs |
| classes | 5 offensive labels merged into **one** foreground class `offensive`; normal = background | identical | paper §3.1: "all segments originally labeled as hateful, insulting, sexual, violent, or self-harm are merged into a single offensive category" |
| visual features | frozen ViT-Large at each timestamp | frozen **CLIP ViT-L/14-336**, `CLIPVisionModel.pooler_output`, 1024-d | same tower this project uses everywhere else, so detector and score-curve features are the same substrate; the paper does not name its ViT-L checkpoint |
| moment rate | 4 FPS | 4 FPS (`ffmpeg fps=4`) | identical |
| detector | ActionFormer, 30 epochs, per-modality models, multimodal by late fusion | official ActionFormer (`happyharrycn/actionformer_release` @ `61ea7eb`), hyper-parameters copied verbatim from `configs/thumos_i3d.yaml` (30 epochs + 5 warmup, lr 1e-4, wd 0.05, bs 2, `n_mha_win_size 19`, `fpn_type identity`, soft-NMS test cfg) | THUMOS is the untrimmed-long-video config; the paper does not publish its config |
| metric | tIoU 0.3/0.5/0.7, offensive class only, P / R / F1 | identical (`scripts/r16_detbase/eval_f1.py`) | paper Table 4 satisfies F1 = 2PR/(P+R) exactly, so it is one-to-one matched-proposal P/R |
| GT instance | not stated | **primary = merged offensive blocks**; secondary = raw offensive segments | see §3 |

**Known-in-advance deviations.** (a) The paper's audio/text branches use windowed BERT-Base
`[t-2s,t]` and Wav2Vec-Emotion `[t-4s,t]` per timestamp and fuse *late*; if a multimodal arm is
run this round it will be **early fusion** (channel concatenation into one ActionFormer), which
is a different system and will be labelled as such. (b) The paper reports an "Acc" column for
localization whose definition it never gives; we do not attempt it.

## 3. Ground-truth instance convention (pre-declared, both reported)

HateClipSeg's segments tile the whole video (mean 8.88 s). The paper merges the five offensive
*labels*; it never says whether two adjacent offensive segments are one instance or two.

- **PRIMARY — `blocks`:** maximal contiguous runs of offensive segments (train/val/test =
  691/134/359 instances, mean 36.6 s). This is the convention behind every number this project
  has ever reported (`scripts/r14_loc/recon_decode.py:blocks_of`), so it is the only one on
  which the detector and the 16.0 / 23.8 score-curve figures are comparable. **All decisions in
  §5 are made on this convention.**
- **SECONDARY — `rawseg`:** every offensive segment is its own instance (2732/531/1474, mean
  9.1 s). Reported as a protocol-sensitivity arm; no decision hangs on it.

## 4. Hyper-parameter and code provenance (no tuning on test)

1. **Architecture and optimization:** `third_party/actionformer/configs/hateclipseg_clip.yaml`
   is `configs/thumos_i3d.yaml` with only the entries the substrate forces changed
   (`dataset_name`, paths, `num_classes: 1`, `input_dim: 1024`, `feat_stride: 1`,
   `num_frames: 0`, `default_fps: 4`). Nothing is swept.
2. **Selectable on val only:** the training epoch (of the 35 run) and the single global
   proposal-score threshold that turns scored proposals into a decision. Both are chosen to
   maximize **val F1@tIoU 0.5** and then applied unchanged at tIoU 0.3 and 0.7 and on test.
3. **Test is opened once per arm**, after selection, by the `--touch-test` path of
   `scripts/r16_detbase/run_af.py`. No test-derived quantity may re-enter any choice.
4. **Evaluation code:** `scripts/r16_detbase/eval_f1.py`, matcher semantics carried over from
   `scripts/r14_loc/recon_decode.py:match_f1`, with the one declared change that proposals are
   matched in descending score order (a detector emits scored proposals; the score-curve
   decoder emits unscored intervals, for which the two rules coincide).
5. **Seeds:** `5100, 5101, 5102` (3 seeds, unused by any prior round; prior rounds consumed
   42, 4299-4310, 4399). Report mean ± sd. Known GPU noise ±0.5.

## 5. What counts as success

This round succeeds if **either**:

- **(S1)** test **F1@tIoU 0.5 ≥ 40** on the primary convention, i.e. the published baseline is
  reproduced to the same order of magnitude on our subset; **or**
- **(S2)** it is not, and the round identifies the **mechanism** of the residual gap with a
  measurement, not a conjecture.

Failing both — a number well below 40 with no mechanism attached — is a failed round and must
be written up as one.

## 6. Post-hoc diagnostic (descriptive, no gate)

After the primary number exists, and labelled post-hoc, we decompose the gap against the
per-window score curve by swapping the two systems' parts:

- **D-a** ActionFormer proposals scored / kept, versus the score-curve decoder's intervals, on
  the identical GT and matcher — the total gap.
- **D-b** the score curve's decoded intervals *rescored* by ActionFormer's classification head
  (proposal scoring, no boundary regression) — isolates "which intervals to keep".
- **D-c** ActionFormer's proposals with boundary regression **disabled** at inference (each
  proposal snapped back to the grid segment that generated it) — isolates boundary regression.
- **D-d** an oracle that takes the score curve's intervals and snaps each to its best-matching
  gold block — the ceiling of pure boundary repair on our substrate.

These are diagnostics of a reproduction, not gates, and they may be defined after seeing the
primary number.

## 7. Artifacts

| artifact | path |
|---|---|
| this freeze | `idea-stage/R16_DETBASE_FREEZE.md` |
| vendored ActionFormer (`61ea7eb`, 2024-04-10) | `third_party/actionformer/` |
| dataset adapter | `third_party/actionformer/libs/datasets/hateclipseg.py` |
| config | `third_party/actionformer/configs/hateclipseg_clip.yaml` |
| 4-FPS CLIP feature extractor | `scripts/r16_detbase/extract_dense_clip.py` |
| annotation builder | `scripts/r16_detbase/make_af_json.py` |
| runner | `scripts/r16_detbase/run_af.py` |
| metric | `scripts/r16_detbase/eval_f1.py` |
| result | `idea-stage/R16_DETBASE_RESULT.md` |
