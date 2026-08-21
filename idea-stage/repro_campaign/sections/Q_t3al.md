## Q. Method as run — T3AL

T3AL, *Test-Time Zero-Shot Temporal Action Localization* (Liberatori, Conti, Rota,
Wang, Ricci; CVPR 2024). Repo `benedettaliberatori/T3AL` @ `dfbbbc1c`, on disk at
`third_party/T3AL`. Wave 2, supervision class **label-free**.

Mechanism under test: a frozen vision–language model (CoCa ViT-L/14) is adapted on
each individual **unlabelled** test video with a self-supervised objective, and the
adapted model's frame-wise similarity to a class name is thresholded into
intervals. Three stages per video — video-level pseudo-label by aggregating the
whole video; adaptation of the text projection, the image projection and the logit
scale by a BYOL-style loss between the class embedding and the top/bottom scoring
frames; refinement of the resulting proposals by the mutual similarity of CoCa
captions inside them.

Reading test videos without reading test **labels** is the method's own published
mechanism, so the row is label-free. Upstream's single point of contact with
annotations, `T3ALNet.get_segments_gt`, is overridden to return an empty list, so
the label path is severed in code, not by convention.

Run record, frozen and committed before any metric existed:
`idea-stage/repro_t3al/RUN_RECORD.md` (commit `bfa181d`).

### Q.1 What was ported

Nothing in T3AL's mechanism was rewritten. `scripts/repro_campaign/run_t3al.py`
imports `src.models.components.tt_method.T3ALNet` from the upstream tree and runs
`T3ALNet.forward` **verbatim** — pseudo-label inference, the adaptation loop, the
moving average, `select_segments`, the caption-refinement filter and the final
segment classification are all upstream code executing unmodified. The subclass
replaces exactly four hooks:

| hook | upstream | ours | why |
|---|---|---|---|
| `__init__` | loads a THUMOS/ActivityNet split dictionary and an annotation JSON | sets the same attributes from a campaign preset and the frozen class list | upstream's constructor hard-fails on any dataset name other than `thumos`/`anet` |
| `get_video_fps` | reads the container fps with OpenCV, because features are one vector per native frame | returns 4.0, our feature rate | see deviation T-2 |
| `get_segments_gt` | reads the gold segments of the video being localised | returns `([], set())` | severs the only annotation read; the return value feeds a visualisation and an unused `gt_mask` |
| `plot_visualize` | draws a matplotlib figure of the final similarity signal | records that signal and returns `None` | this is the only place upstream exposes the continuous localisation score; capturing it changes no value |

The other two artefacts T3AL consumes are produced by
`scripts/repro_campaign/extract_coca_4fps.py`:

* **features** — CoCa ViT-L/14 (`mscoco_finetuned_laion2B-s13B-b90k`, the repo's own
  checkpoint) visual embeddings taken *before* the visual projection, because
  `T3ALNet.forward` applies `@ self.model.visual.proj` itself. Extraction skips that
  one line by setting `visual.proj = None` for the call, so the stored tensor is
  exactly the tensor upstream's released `.npy` files hold, shape `(T, 768)`.
  Preprocessing is the repo's own `src/data/components/utils.py::transform` — an
  aspect-squashing resize to 224×224 then CLIP mean/std — reproduced with an ffmpeg
  `scale=224:224` filter.
* **captions** — CoCa captions written in the exact filename and line format the
  refinement step parses (`./captions/<video>.txt`, lines `"<k>.jpg-<caption>"`,
  where upstream reads back the feature index as `int(k) * 3`).

The published pre-extracted features cover THUMOS14 and ActivityNet-v1.3 only, which
is why both artefacts are extracted here rather than downloaded.

Install: `open-clip-torch==2.24.0` (the repo's pin) into the shared `HateVideo`
env with `--no-deps`, so no torch/transformers version moved. Two version-skew
shims were needed and are in `extract_coca_4fps.patch_isin_device`; both are pure
device/type plumbing and change no value (deviation T-6).

### Q.2 The frozen class list, verbatim

T3AL's interface is a closed list of bare class names — it builds its own prompt,
`"a video of action" + " " + name`. The six campaign categories are therefore given
as bare names, in the freeze §4 HateClipSeg class order, worded exactly as
`HCS_CLASS_QUERIES` in `scripts/repro_campaign/run_unitime.py` with that file's
fixed sentence frame removed:

```python
CLASS_NAMES = [
    "normal, non-toxic content",   # c0  <- "the moment containing normal, non-toxic content"
    "hateful content",             # c1  <- "the moment containing hateful content"
    "insulting content",           # c2  <- "the moment containing insulting content"
    "sexual content",              # c3  <- "the moment containing sexual content"
    "violent content",             # c4  <- "the moment containing violent content"
    "harmful content",             # c5  <- "the moment containing harmful content"
]
MAIN_LABEL = "hateful content targeting a person or group"
#   <- run_unitime.MAIN_QUERY, "the moment containing hateful content targeting a
#      person or group", reduced the same way
```

so the six prompts CoCa actually encodes are `a video of action normal, non-toxic
content`, `a video of action hateful content`, `a video of action insulting
content`, `a video of action sexual content`, `a video of action violent content`,
`a video of action harmful content`. The list is a module-level constant in
`scripts/repro_campaign/run_t3al.py` with a comment recording that it was fixed
before the run; it is unchanged since commit `bfa181d`.

### Q.3 Variants

| variant | what it is |
|---|---|
| `main` | the published pipeline end to end. The class is the one T3AL's own pseudo-label step picks from the six. Frame curve = the covering predicted segment's toxic probability (softmax mass on classes 1–5), 0 outside any segment; the interval file carries the same intervals and scores, so the frame numbers and F1@tIoU describe the same object. |
| `mainq_sim` | adaptation target forced to `MAIN_LABEL`, i.e. T3AL is asked the same question Wave 0's Qwen2.5-VL and Wave 1's UniTime were asked. Frame curve = the adapted model's continuous post-adaptation similarity signal (upstream's `similarity`, after its own moving average and min–max normalisation). |
| `c0_normal` … `c5_harm` | adaptation target forced to class *k*; frame curve = the same continuous signal. On HateClipSeg the evaluator scores each against that released class's own frame labels (`load_gt_hcs_class`); `c1_hateful` is also written on the other three datasets, where it is scored against the binary hate gold. |

`main` is the faithful localiser row. The forced-class rows exist because T3AL's
pseudo-label step commits to one class per video, so without them there is no
per-class score to build the HateClipSeg appendix from.

### Q.4 Deviations

**T-1 — features and captions are ours, not the released ones.** T3AL ships
pre-extracted CoCa features for THUMOS14 and ActivityNet-v1.3 only. Ours come from
the same checkpoint through the repo's own preprocessing, and are taken at the same
point in the network (pre-projection), so the interface is identical; only the
corpus differs.

**T-2 — feature rate is 4 fps, not the video's native frame rate.** Upstream stores
one vector per decoded frame and converts a predicted feature index to seconds by
dividing by the container fps. We extract on the campaign's canonical 4 fps grid
(freeze §1), so `native_rate` is 4 fps and the evaluator's piecewise-constant
broadcast is the identity. Cost: T3AL's index-unit knobs (`kernel_size`, `stride`)
no longer mean what they meant at 30 fps, which is why the val sweep carries a
rescaled preset (`B_thumos_rescaled`) alongside the verbatim one. Benefit: the
whole val+test corpus is 26.7 h of video, which at native rate would be ~2.9 M
frames instead of ~385 k.

*A mechanical consequence of T-2, recorded here because it is not a choice we
made and it is not visible in the numbers.* Upstream's `get_indices` has a guard:
when `100 × n` is at least the number of feature vectors in the video, it stops
sampling and takes `torch.arange(T)` for **both** the positive and the negative
index set, i.e. the adaptation loss for that video compares a set against itself.
With `n = 4` the guard fires for any video with fewer than 400 feature vectors. At
THUMOS' 30 fps that is a 13 s video and it never happens; on our 4 fps grid it is a
100 s video, so it fires for essentially every MHC-EN and MHC-ZH video (both
corpora are capped at 60 s, T ≈ 120–240) and for the shorter half of HateMM. It
does not fire on HateClipSeg (T ≈ 960) or on long HateMM videos. Preset `D_anet`
uses `n = 20`, so its guard fires below 2000 vectors, i.e. on almost everything.
Nothing was patched — this is upstream's code path for short inputs — but any
reading of the MHC rows has to know that T3AL's adaptation step is degenerate
there, and that the row therefore measures the pseudo-label plus the segment
selector rather than the adaptation.

**T-3 — captions are decoded greedily, not with 6-beam search.** `CoCa.generate`'s
default in the repo's open_clip version is `generation_type="beam_search",
num_beams=6`. Measured on this card, that is 1.5 captions/s against 61 captions/s
for greedy decoding (`generation_type="top_k", top_k=1`, which keeps a single token
and so is deterministic) — 18 GPU-hours against ~10 minutes for the ~96 k captions
the corpus needs. The captions are consumed only as CoCa *text embeddings* inside a
segment-consistency filter, never read as text, so the decoding rule affects which
proposals survive refinement but not what the refinement measures.

**T-4 — captions at 1 fps, features at 4 fps.** Upstream captioned THUMOS at 10 fps
(every third frame of 30 fps video) and ActivityNet at 1 fps; we caption at 1 fps
everywhere. Upstream's parser recovers the feature index as `3 ×` the line prefix,
so we write the prefix as the feature index divided by three, which places each
caption within ±2 feature indices (±0.5 s) of the frame it describes.

**T-5 — the checkpoint is restored fully between videos.** Upstream restores the
weights at the *start of the last adaptation step* rather than the pristine state,
and keeps one Adam instance for the whole test loop, so adaptation state leaks from
one test video into the next and the result depends on the order videos are visited
and on whether the run was ever resumed. We snapshot the pre-trained state once and
reload it before every video, and build the optimiser per video. This is what the
paper describes ("adapts a pre-trained VLM at inference time **on a sample basis**"),
and it is what makes a resumed corpus run reproduce an uninterrupted one. Each
video is additionally seeded from `seed ^ crc32(video_id)`, the convention already
used by the AV²A row. Upstream's typo `self.model.locit_scale = ...`, which
registers an unused parameter of that name the first time refinement fires, is left
in place and simply not restored.

**T-6 — two version-skew shims in `CoCa.generate`.** open_clip 2.24 against
transformers 4.49: `MinLengthLogitsProcessor` keeps `eos_token_id` on the CPU while
the vocabulary tensor is built on the scores' device, and
`StoppingCriteriaList.__call__` now returns a per-sequence bool tensor where
open_clip uses a scalar. The shims move the one tensor onto the other's device and
reduce the other with `.all()`, which is the semantics open_clip was written
against. Neither changes a value.

**T-7 — feature extraction runs the encoder in fp16, features stored fp32.** Same
convention as the campaign's existing dense CLIP cache
(`scripts/r16_detbase/extract_dense_clip.py`), which also extracts in fp16. The
adaptation itself runs in fp32, as upstream's `.float()` requires.

**T-8 — val+test only; the train split has no features.** The headline table is the
test split (freeze §5) and the knob choice needs val, so extraction covers those two
splits on all four datasets. There is no full-corpus T3AL row, which is why this
section has no §3-style full-corpus table.

### Q.5 Frozen knobs and where they came from

Fixed by the protocol or by upstream, never tuned: 4 fps features; CoCa ViT-L/14
`mscoco_finetuned_laion2B-s13B-b90k`; lr 1e-5, weight decay 1e-4, the `lr × 0.001`
group scaling on `visual.proj`; loss `BYOLfeat`; `randper` 10; `topk` 3; `m` 0.7;
`text_projection` / `image_projection` / `logit_scale` adapted, `text_encoder` not;
`refine_with_captions` on. Every one of these is identical in both released
configs, or is the value the selected config carries.

Chosen on **val**, one choice for all four datasets, from four presets built only
out of values that appear in the released configs:

| preset | source | steps | kernel_size | stride | normalize | remove_background | p | n | segment rule |
|---|---|---|---|---|---|---|---|---|---|
| `A_thumos` | `configs/model/tt_thumos.yaml` verbatim | 60 | 20 | 20 | on | on | 0.75 | 4 | moving average, threshold at the signal mean |
| `B_thumos_rescaled` | A with the index-unit knobs converted 30 fps → 4 fps | 60 | 3 | 3 | on | on | 0.75 | 4 | moving average, threshold at the signal mean |
| `C_thumos_15steps` | A with ActivityNet's adaptation budget | 15 | 20 | 20 | on | on | 0.75 | 4 | moving average, threshold at the signal mean |
| `D_anet` | `configs/model/tt_anet.yaml` verbatim | 15 | 50 | 200 | off | off | 0.8 | 20 | no moving average, threshold at 0.7 |

Selection rule, frozen in `RUN_RECORD.md` and implemented in
`scripts/repro_campaign/t3al_select.py` before the sweep ran: variant `main` only,
seed 20250819, val split of all four datasets; take the preset with the highest
**mean over the four datasets of the val pooled frame PR-AUC**; a dataset a preset
produced no row for scores that dataset's base rate; ties break by table order.

<!-- FILL: val sweep table from idea-stage/repro_t3al/preset_chosen.json -->

### Q.6 Seeds

T3AL's adaptation is stochastic — `get_indices` jitters the sampled positive and
negative frame indices by `torch.randint(-randper, randper)` at every one of the
adaptation steps — so freeze §6 applies and the row carries **three seeds,
20250819 / 20250820 / 20250821**, reported mean ± sd.

### Q.7 Headline test-split table

<!-- FILL: idea-stage/repro_t3al/eval/test_agg.md -->

### Q.8 HateClipSeg 6-class appendix

<!-- FILL: the c0..c5 rows of idea-stage/repro_t3al/eval/test_agg.md -->

### Q.9 Stratified sub-tables — single-span vs multi-span (HateMM, MHC)

<!-- FILL: strat_single_span / strat_multi_span of idea-stage/repro_t3al/eval/test_agg.json -->

### Q.10 What the numbers say

<!-- FILL -->
