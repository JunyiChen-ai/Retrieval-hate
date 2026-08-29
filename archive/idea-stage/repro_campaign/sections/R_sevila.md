## R. Method as run — SeViLA Localizer

**Status at the time this section was written: the port is finished, validated and queued; the
corpus numbers are not in yet.** The GPU is held by the Wave 1 UniTime run (see §R.7), so the
single test call is parked in the campaign GPU queue behind it. Everything the freeze requires to
be fixed *before* the run — the question text, the prompt template, the sampling rate, the
read-out, the missing-video policy — is fixed in code and is transcribed verbatim below. Nothing
in §R.1–§R.5 can change once the run starts; §R.6 is the table shell the run fills.

Wave 2. Supervision class **aux-temporal-pretrain**: the Localizer Q-Former in
`sevila_pretrained.pth` was pre-trained with temporal supervision on QVHighlights, a corpus
disjoint from all four hate datasets. No hate-dataset label of any split touches this method.

### R.1 What the method is, and which code path is used

SeViLA (Yui010206/SeViLA @ `419e7281`, NeurIPS 2023) is a two-stage video-QA model: a **Localizer**
scores every candidate frame with a yes/no question and keeps the top-k, and an **Answerer** then
answers from the kept frames. Only the Localizer is a temporal-localisation mechanism, so only the
Localizer is run.

The code path is the repo's own QVHighlights *moment-retrieval* path, which is exactly "the
Localizer used as a frame scorer":

* model class `lavis/models/blip2_models/blip2_fmr.py::Blip2FMR` — the `arch:` the repo's own
  `lavis/projects/sevila/eval/qvh_eval.yaml` evaluates. It contains the ViT-g/14 vision tower, the
  Localizer Q-Former (`Qformer_loc`, `ln_vision_loc`, `t5_proj_loc`) and Flan-T5-XL, and nothing of
  the Answerer.
* read-out `Blip2FMR.generate` → `outputs.scores[0][:, [no_id, yes_id]]` → `yes_score`, i.e. the
  **raw logit of the token `yes` at the first decoding step**, one number per frame. Greedy,
  `num_beams=1`, `do_sample=False` → deterministic, one run, no error bar (freeze §6).
* checkpoint `third_party/_ckpt/sevila/sevila_pretrained.pth` (814 MB). It carries only the
  trainable parts (both Q-Formers, both projections, both LayerNorms); the ViT-g weights come from
  BLIP-2's `eva_vit_g.pth` and Flan-T5-XL from `google/flan-t5-xl`, both fetched for this run.
* processor `blip_video_eval` at `image_size=224` with the repo's CLIP mean/std — the processor
  `qvh_eval.yaml` names.

### R.2 The venv recipe

Verdict in `MODEL_ASSETS_STATUS.md` row 9 was NEEDS-OWN-VENV: SeViLA pins `timm==0.4.12` while the
shared `HateVideo` env is on `timm 1.0.15`, which ImageBind and others need. The venv layers the
repo's pins on top of the shared env instead of rebuilding a 3 GB CUDA stack:

```bash
source /home/jehc223/miniconda3/etc/profile.d/conda.sh && conda activate HateVideo
python -m venv --system-site-packages third_party/_venv/sevila
V=third_party/_venv/sevila/bin
$V/pip install "timm==0.4.12" --no-deps          # shadows the shared env's 1.0.15
$V/pip install webdataset contexttimer iopath pycocoevalcap "fairscale==0.4.4" --no-deps
$V/pip install braceexpand --no-deps
$V/pip install spacy pycocotools                 # lavis/__init__ imports both at package load
```

Assets fetched once, outside the venv:

```bash
python scripts/repro_campaign/hf_fetch.py google/flan-t5-xl \
    --exclude "*.h5" "*.msgpack" "*.onnx" "*tf_model*" "*flax*"     # ~22 GB into the HF cache
curl -L -C - -o ~/.cache/torch/hub/checkpoints/eva_vit_g.pth \
    https://storage.googleapis.com/sfr-vision-language-research/LAVIS/models/BLIP2/eva_vit_g.pth
```

Run with `PYTHONPATH=third_party/SeViLA` and `HF_HUB_OFFLINE=1`. The shared env is untouched: it
still has `timm 1.0.15`, and nothing was installed into it.

### R.3 The frozen question text, verbatim

The template is the repo's own, from the `'QVHighlight' in qid` branch of
`lavis/datasets/datasets/mc_video_vqa_datasets.py` — the branch that built the Localizer's
pre-training and evaluation prompts:

```
loc_prompt = 'Question: ' + q + ' ' + 'Option A: yes. Option B: no.' +
             ' Does the information within the frame provide the necessary details to accurately answer the given question?'
```

The **only** substitution is `q`. The main question is the yes/no phrasing of `MAIN_QUERY` in
`scripts/repro_campaign/run_unitime.py`, so SeViLA, UniTime and Qwen2.5-VL all answer the same
thing. Full string fed to the model, verbatim:

```
Question: Does this frame contain hateful content targeting a person or group? Option A: yes. Option B: no. Does the information within the frame provide the necessary details to accurately answer the given question?
```

The six HateClipSeg per-class questions, derived the same way from `HCS_CLASS_QUERIES`, each
substituted into the same template:

| variant | `q` |
|---|---|
| `c0_normal` | `Does this frame contain normal, non-toxic content?` |
| `c1_hateful` | `Does this frame contain hateful content?` |
| `c2_insulting` | `Does this frame contain insulting content?` |
| `c3_sexual` | `Does this frame contain sexual content?` |
| `c4_violence` | `Does this frame contain violent content?` |
| `c5_harm` | `Does this frame contain harmful content?` |

Every per-class row is scored against **that class's own frame labels**, rebuilt from
`gold_segments.json` on the 4 fps grid by the shared evaluator's `load_gt_hcs_class`.

### R.4 Frozen sampling rate and knobs

* **native_rate = 1.0 fps.** Frames come from the campaign's frozen 1 fps JPEG cache
  `data/frames_1fps/<DS>/<vid>/%06d.jpg` (ffmpeg `fps=1`, so file `000123.jpg` is the content at
  `t = 123 s`) — the same frames the LAVAD chain reads. The evaluator broadcasts sample `k`
  piecewise-constant over `[k, k+1)` on the 4 fps grid, which is what the frozen JPEG grid means.
  Per video the sample count is `floor(D)`, and `rate = 1.0` is written into every npz.
  Freeze §1 prefers a fixed 1 fps grid where the repo allows one, and it does: the Localizer scores
  each frame independently, so the repo's `n_frms=64`-per-video default is a budget, not a
  constraint of the model.
* **Cost, measured before committing.** The test split is 67,647 frames of vision
  (HateMM 29,243 / MHC-EN 5,604 / MHC-ZH 4,558 / HateClipSeg 28,242) and 237,099 prompt passes
  (HateClipSeg carries seven prompts per frame). Because the Localizer Q-Former takes no text
  input, one visual forward serves all seven prompts (deviation S4), so the vision tower runs once
  per frame. This is a low single-digit GPU-hour job at chunk 32, well inside the 12 h ceiling, so
  1 fps needed no reduction.
* **Free knobs: none.** The prompt template, the frame count per video, the precision, the decoding
  and the read-out are all the repo's. Nothing was selected on val; no val run was performed, and
  no test label is read by anything but the evaluator.
* **Test calls: one per dataset** (freeze §10 red line 4), a single
  `eval_frame.py --method curves --split test` invocation covering all variants.

### R.5 Adaptations, each a named deviation

**S1 — own venv layered on the shared env, with the repo's `timm` pin.** `python -m venv
--system-site-packages` keeps torch `2.7.1+cu128` from `HateVideo` (the RTX 5090 is `sm_120` and
needs cu128; the repo asks only for `torch>=1.10`), and `timm==0.4.12` is installed into the venv
where it shadows the shared 1.0.15. `transformers 4.49.0` is inherited rather than the repo-era
4.31: LAVIS vendors its own `modeling_t5.py` and `Qformer.py`, and both import cleanly and produce
the checkpoint's own numbers under 4.49 (verified by S4's equality check).

**S2 — `Blip2FMR`, not the `SeViLA` wrapper.** Both classes contain the identical Localizer
arithmetic; `Blip2FMR` is the repo's own moment-retrieval arch and stops after the frame scores,
where the `SeViLA` class would go on to run the Answerer on the top-k frames. We want the frame
scores, so the Answerer is not built and not run.

**S3 — the template's trailing-period normalisation is not applied.** The QVHighlights branch does
`if q[-1] != '.': q += '.'`, because QVHighlights queries are declarative fragments with no final
punctuation. Our question already ends in `?`; appending a period would produce `group?.`. The rule
is skipped and the question ends at its question mark.

**S4 — the published `generate` body is split at the Q-Former boundary.** `Blip2FMR.generate`
recomputes the whole vision tower for every prompt, which would mean seven ViT-g passes per
HateClipSeg frame. The body is split into `vision_feats` (vision tower → `ln_vision_loc` →
`Qformer_loc` → `t5_proj_loc`) and `prompt_logits` (frame prefix → T5 → `scores[0]`), copied line
for line, so one visual forward serves all seven prompts. `run_sevila.py --selftest` runs both the
split path and the **unmodified** `generate` on the same 8-frame chunk and prints the largest
absolute difference; on the CPU validation it was **0.000000** — bit-exact — and the check reruns on
GPU as the first step of the queued job.

**S5 — `max_new_tokens=1` instead of 30.** Only `outputs.scores[0]` is read, and under greedy
decoding the first-step distribution does not depend on how many further tokens are generated. The
S4 equality check compares against the unmodified call at `max_length=30`, so this is verified, not
assumed.

**S6 — frames from the campaign's 1 fps JPEG cache, not a decord read of the container.** Reasons:
the same frames then feed SeViLA and the LAVAD chain, so a rate difference cannot explain a
difference between them; and the two HateMM containers with no video stream and the handful of
truncated files are already characterised on that cache. JPEGs are decoded with PIL and resized to
224×224 bilinear, where the repo resizes inside decord — both squash the aspect ratio, neither
crops. Where a video has no cache entry the driver extracts it with the campaign's own
`extract_1fps` (identical ffmpeg command), so the full-corpus extension needs no new convention.

**S7 — `*_margin` is ours.** Each npz also carries `yes − no` from the same forward pass, as
`main_margin` and `c*_margin`. The base row is always the repo's own `yes_score`; the margin rows
are reported separately and clearly marked, per freeze §8's rule for our own variants, and never
replace the base row.

**S8 — crash handling follows deviation D3.** The driver writes an in-flight marker, but an id is
retired only after it has taken the process down **twice** (`inflight.crashcount.json`), and a
caught SIGTERM/SIGINT clears the marker and exits without retiring anything. A CUDA OOM halves the
chunk and retries the video rather than recording a method failure.

**Pre-run validation, all of it shape/range/agreement only (freeze §10 red line 3 — no metric of
this method has been computed).**

1. venv builds; `lavis.models.blip2_models.blip2_fmr` imports; the checkpoint loads
   (`Qformer_loc`, `ln_vision_loc`, `t5_proj_loc`, `query_tokens_loc` all matched; the Answerer keys
   in the file are unused by `Blip2FMR` and ignored).
2. `--selftest --device cpu` on 8 frames of a HateClipSeg video: clip `(8, 3, 224, 224)`, range
   `[-1.792, 2.146]`; `main` shape `(8,)`, values in `[-1.858, -1.699]`; the split path matches the
   unmodified `Blip2FMR.generate` with `max|Δ| = 0.000000`.
3. npz round-trip: exactly one file per video, keys `main`, `main_margin`, `rate`, atomic
   `tmp → os.replace`.
4. Evaluator plumbing exercised end-to-end on the **val** split with synthetic random curves in the
   npz layout above: every variant key resolves, the six `c*` names route to their own per-class
   gold (base rates 0.4910 / 0.2136 / … / 0.0111, i.e. class-specific as intended), `rate` is
   honoured, coverage 1.0, and random input lands at ROC ≈ 0.5. No test label was read.
5. The D3 crash handling was exercised: a `SIGTERM` to the driver mid-video cleared the in-flight
   marker, wrote no crash count and retired nothing.

**Missing videos.** `hate_video_147` and `hate_video_292` have no video stream (freeze D2); neither
is in any split. `yt_NzvfkIYS5Yg` (HateClipSeg test) is one of the four genuinely undecodable files
of D3 and has no 1 fps cache entry. Missing videos are reported and dropped from the pool; **no
zero array is fabricated for any of them.**

### R.6 Results

The single test call, once the queued job lands:

```bash
source /home/jehc223/miniconda3/etc/profile.d/conda.sh && conda activate HateVideo
python scripts/repro_campaign/eval_frame.py --method curves \
  --curve-dir idea-stage/repro_sevila/curves \
  --variants main,main_margin,c0_normal,c1_hateful,c2_insulting,c3_sexual,c4_violence,c5_harm \
  --method-name "SeViLA Localizer" --wave 2 --supervision aux-temporal-pretrain --split test
# tables:
python scripts/repro_campaign/make_results_rows.py \
  --json idea-stage/repro_campaign/eval_SeViLA\ Localizer_test.json --controls --split test \
  --run-dir '`idea-stage/repro_sevila/`' --native-rate '1 fps' \
  --variant-map 'main=base,main_margin=+margin (ours)'
python scripts/repro_campaign/make_results_rows.py --json ... --strata     # R.6.3
```

`F1@0.3 / F1@0.5 / F1@0.7` read **n/a** for every SeViLA row. The Localizer emits a score per
frame and no intervals; freeze §2 forbids inventing a threshold to manufacture proposals for a
score-curve method.

#### R.6.1 Headline — main question, test split

Control rows are the frozen §3 values for the test pool and are already final; the SeViLA rows are
pending the run.

| method | wave | dataset | split | supervision | variant | query_set | native_rate | frame_ROC_AUC | frame_PR_AUC | F1@0.3 | F1@0.5 | F1@0.7 | AP_norm | n_frames | base_rate | seeds | transplant | gt_convention | run_dir | notes |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| GOLD_BROADCAST | — | HateMM | test | control | control | n/a | video | 0.8857 | 0.5829 | n/a | n/a | n/a | 1.0000 | 116975 | 0.2421 | 1 | n/a | §4+D1 | idea-stage/repro_campaign/ | zero-temporal-resolution ceiling, full GT pool |
| RANDOM_UNIFORM | — | HateMM | test | control | control | n/a | 4 fps | 0.5003 ± 0.0019 | 0.2423 ± 0.0013 | n/a | n/a | n/a | 0.0000 | 116975 | 0.2421 | 20 | n/a | §4 | idea-stage/repro_campaign/ | U(0,1) per frame, 20 seeds, full GT pool |
| SeViLA Localizer | 2 | HateMM | test | aux-temporal-pretrain | base | main | 1 fps | *pending* | *pending* | n/a | n/a | n/a | *pending* | *pending* | 0.2421 | 1 | n/a | §4 | idea-stage/repro_sevila/ | yes-logit curve, 1 fps broadcast to 4 fps |
| GOLD_BROADCAST | — | MHC-EN | test | control | control | n/a | video | 0.9427 | 0.7664 | n/a | n/a | n/a | 1.0000 | 22337 | 0.2734 | 1 | n/a | §4+D1 | idea-stage/repro_campaign/ | zero-temporal-resolution ceiling, full GT pool |
| RANDOM_UNIFORM | — | MHC-EN | test | control | control | n/a | 4 fps | 0.5004 ± 0.0034 | 0.2737 ± 0.0026 | n/a | n/a | n/a | 0.0000 | 22337 | 0.2734 | 20 | n/a | §4 | idea-stage/repro_campaign/ | U(0,1) per frame, 20 seeds, full GT pool |
| SeViLA Localizer | 2 | MHC-EN | test | aux-temporal-pretrain | base | main | 1 fps | *pending* | *pending* | n/a | n/a | n/a | *pending* | *pending* | 0.2734 | 1 | n/a | §4 | idea-stage/repro_sevila/ | yes-logit curve, 1 fps broadcast to 4 fps |
| GOLD_BROADCAST | — | MHC-ZH | test | control | control | n/a | video | 0.9842 | 0.9191 | n/a | n/a | n/a | 1.0000 | 18199 | 0.2648 | 1 | n/a | §4+D1 | idea-stage/repro_campaign/ | zero-temporal-resolution ceiling, full GT pool |
| RANDOM_UNIFORM | — | MHC-ZH | test | control | control | n/a | 4 fps | 0.4985 ± 0.0052 | 0.2646 ± 0.0038 | n/a | n/a | n/a | 0.0000 | 18199 | 0.2648 | 20 | n/a | §4 | idea-stage/repro_campaign/ | U(0,1) per frame, 20 seeds, full GT pool |
| SeViLA Localizer | 2 | MHC-ZH | test | aux-temporal-pretrain | base | main | 1 fps | *pending* | *pending* | n/a | n/a | n/a | *pending* | *pending* | 0.2648 | 1 | n/a | §4 | idea-stage/repro_sevila/ | yes-logit curve, 1 fps broadcast to 4 fps |
| GOLD_BROADCAST | — | HateClipSeg | test | control | control | n/a | video | 0.6260 | 0.5437 | n/a | n/a | n/a | 1.0000 | 114097 | 0.4712 | 1 | n/a | §4+D1 | idea-stage/repro_campaign/ | zero-temporal-resolution ceiling, full GT pool |
| RANDOM_UNIFORM | — | HateClipSeg | test | control | control | n/a | 4 fps | 0.5009 ± 0.0021 | 0.4721 ± 0.0016 | n/a | n/a | n/a | 0.0000 | 114097 | 0.4712 | 20 | n/a | §4 | idea-stage/repro_campaign/ | U(0,1) per frame, 20 seeds, full GT pool |
| SeViLA Localizer | 2 | HateClipSeg | test | aux-temporal-pretrain | base | main | 1 fps | *pending* | *pending* | n/a | n/a | n/a | *pending* | 0.4712 | 1 | n/a | §4 | idea-stage/repro_sevila/ | yes-logit curve; 1 missing (`yt_NzvfkIYS5Yg`, undecodable) dropped, not interpolated |

The `+margin (ours)` rows (variant `main_margin`, the yes − no logit from the same forward pass)
are appended below the base row for each dataset by the same command.

#### R.6.2 HateClipSeg 6-class appendix

One question per released class, each scored against that class's own frame labels. Rows pending;
the table shape is the §I.3 one, with `variant` = `c0_normal … c5_harm` and `native_rate` = 1 fps.

#### R.6.3 Stratified sub-tables — single-span vs multi-span (HateMM / MHC)

Freeze §14 requires these because the coverage degeneracy differs sharply between the strata
(HateMM single-span 72.8%, MHC-EN 95.8%, MHC-ZH 98.2%). Produced by the same evaluator JSON with
`make_results_rows.py --strata`; rows pending. HateClipSeg has no stratification (the evaluator
emits strata only for the three span-annotated corpora).

### R.7 Run status, and how to finish it

| item | value |
|---|---|
| launcher | `scripts/repro_campaign/run_sevila_wave2.sh` |
| driver | `scripts/repro_campaign/run_sevila.py` |
| log | `logging/runs/repro_sevila/run.log` |
| pid file | `logging/runs/repro_sevila/run.pid` |
| queue | `scripts/repro_campaign/gpu_queue.sh sevila …`, parked behind `unitime` |
| provenance | `idea-stage/repro_sevila/run_meta_test.json` (written at the end of the run) |

The launcher is already detached and waiting on the campaign GPU lock. It takes the lock, runs the
GPU self-test (shapes, score range, and the S4 equality check against the unmodified `generate`),
then scores the test split of all four datasets. It is resume-safe: a video whose npz exists is
skipped, so a restart costs at most one video. Nothing else is needed to finish the section except
running the two commands in §R.6 and pasting their output.

### R.8 What the numbers say

*To be written from the run.* The single sentence this subsection must end with — **does the
literature mechanism (frame-wise yes/no VQA keyframe localisation) work in the hate domain, yes or
no, and on what evidence** — is deliberately left unwritten rather than guessed, because the only
admissible evidence is the test-split table above and it does not exist yet. Freeze §10 red line 3
forbids computing any metric for this method before the frozen run, and §15's reporting stance
forbids reporting a number that was not measured.
