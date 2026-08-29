# T3AL — run record (frozen before the run)

Written and committed **before any T3AL metric was computed**, per freeze §5 and
§10 red lines 2–4. Nothing below may be changed after the commit that introduces
it; a later change is a numbered deviation in the section file.

Method: T3AL, *Test-Time Zero-Shot Temporal Action Localization*, Liberatori et al.,
CVPR 2024. Repo `benedettaliberatori/T3AL` @ `dfbbbc1c`, on disk at
`third_party/T3AL`. Backbone: `open_clip` CoCa ViT-L-14,
`mscoco_finetuned_laion2B-s13B-b90k` (the repo's own, not a choice of ours).

## Supervision class: label-free

T3AL adapts a frozen vision–language checkpoint on each **unlabelled test video**.
That is the method's published mechanism, not a use of supervision: no annotation
of any split is read. Upstream's single point of contact with annotations,
`T3ALNet.get_segments_gt`, is overridden in `scripts/repro_campaign/run_t3al.py` to
return an empty list, so the label path is severed in code rather than by
convention.

## Frozen class list

Six campaign hate categories, in the freeze §4 HateClipSeg class order, wording
copied from `HCS_CLASS_QUERIES` in `scripts/repro_campaign/run_unitime.py` with
that file's fixed sentence frame `"the moment containing ..."` removed, because
T3AL's interface takes bare class names and builds its own prompt
(`"a video of action" + " " + name`).

| index | run_unitime.py source string | T3AL class name |
|---|---|---|
| 0 | `the moment containing normal, non-toxic content` | `normal, non-toxic content` |
| 1 | `the moment containing hateful content` | `hateful content` |
| 2 | `the moment containing insulting content` | `insulting content` |
| 3 | `the moment containing sexual content` | `sexual content` |
| 4 | `the moment containing violent content` | `violent content` |
| 5 | `the moment containing harmful content` | `harmful content` |

Adaptation target of the `mainq_sim` variant, from `MAIN_QUERY` in the same file
(`the moment containing hateful content targeting a person or group`):
`hateful content targeting a person or group`.

The list lives as the module-level constant `CLASS_NAMES` in
`scripts/repro_campaign/run_t3al.py` with the same table in a comment.

## Variants written

| variant | what it is | datasets |
|---|---|---|
| `main` | published pipeline end to end: video-level pseudo-label over the six classes, test-time adaptation, segment selection, caption refinement, segment classification. Frame curve = the predicted segment's toxic probability (softmax mass on classes 1–5), 0 outside any segment; intervals carry the same score. | all four |
| `mainq_sim` | adaptation target forced to the main-query label above; frame curve = the adapted model's post-adaptation similarity signal; intervals from the same `select_segments` call, scored by toxic probability. | all four |
| `c0_normal` … `c5_harm` | adaptation target forced to class *k*; frame curve = the post-adaptation similarity signal; intervals scored by class *k*'s probability. | `c1_hateful` on all four (scored against binary hate gold outside HateClipSeg); the full six on HateClipSeg, where the evaluator scores each against that released class's own frame labels. |

## Free knobs and how they are fixed

Fixed by the protocol, not tuned: feature rate 4 fps (freeze §1); backbone and its
pretrained tag (the repo's); learning rate 1e-5, weight decay 1e-4, the
`lr * 0.001` group scaling for `visual.proj`, loss `BYOLfeat`, `randper` 10, `topk`
3, `m` 0.7, `text_projection`/`image_projection`/`logit_scale` on, `text_encoder`
off, `refine_with_captions` on — all upstream's, identical across both released
configs or taken from the config the sweep selects.

Chosen on **val**, over four presets built only from values that already appear in
the released configs, one selection for all four datasets:

| preset | source | steps | kernel_size | stride | normalize | remove_background | p | n | segment rule |
|---|---|---|---|---|---|---|---|---|---|
| `A_thumos` | `configs/model/tt_thumos.yaml` verbatim | 60 | 20 | 20 | on | on | 0.75 | 4 | moving average, threshold at the mean |
| `B_thumos_rescaled` | as A, with the two index-unit knobs converted from THUMOS' 30 fps native feature rate to our 4 fps grid | 60 | 3 | 3 | on | on | 0.75 | 4 | moving average, threshold at the mean |
| `C_thumos_15steps` | as A with ActivityNet's adaptation budget | 15 | 20 | 20 | on | on | 0.75 | 4 | moving average, threshold at the mean |
| `D_anet` | `configs/model/tt_anet.yaml` verbatim | 15 | 50 | 200 | off | off | 0.8 | 20 | no moving average, threshold at 0.7 |

**Selection rule, frozen here:** run variant `main` only, seed 20250819, on the
**val** split of all four datasets; choose the preset with the highest **mean over
the four datasets of the val pooled frame PR-AUC**; a dataset for which a preset
produced no row scores that dataset's positive base rate; ties break by the order
in the table above. Implemented in `scripts/repro_campaign/t3al_select.py`, which
writes `preset_chosen.json` before stage 4 starts.

## Seeds

T3AL's adaptation loop is stochastic: `get_indices` jitters the sampled positive
and negative frame indices by `torch.randint(-randper, randper)` at every step. So
freeze §6 applies: **three seeds, 20250819 / 20250820 / 20250821**, mean ± sd.
Each video is seeded from `seed ^ crc32(video_id)` so a resumed run reproduces an
uninterrupted one.

## One test call

Stage 4 of `scripts/repro_campaign/t3al_stage.sh` is the single test-split call
(three seeds of the one frozen configuration). No test label is read by anything
before it; the val sweep reads val labels only, which freeze §5 provides for.

## Native output rate

T3AL's released features are one vector per **native video frame** — its evaluator
converts a predicted feature index to seconds by dividing by the video's own fps.
We extract on the campaign's 4 fps grid instead, so the `native_rate` column reads
**4 fps** and the evaluator's piecewise-constant broadcast is the identity.
