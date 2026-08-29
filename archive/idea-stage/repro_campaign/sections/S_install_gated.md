## S. Install-gated methods — the try-install group

Wave 2 lists eight install-gated repos (§9 of the freeze). Three were already written off in
`MODEL_ASSETS_STATUS.md §1` (ZS-STVG, DASM, LAVIDA). This section covers the remaining five,
each given a 30-minute wall-clock budget for install, dependency resolution, checkpoint fetch
and a smoke run on one video. The deliverable per repo is a verdict, not a number: no metric
was computed on any of them (freeze §10 red line 3).

Smoke video: `/home/jehc223/data/HateClipSeg/videos/bit_0dcMcI6hYjhw.mp4` (222.6 s, 6678
frames at 30 fps). Audio for the two audio-only models:
`/home/jehc223/Retrieval-hate/data/AV2A_wav/HateClipSeg/bit_0dcMcI6hYjhw.wav` (pre-demuxed,
nothing new was demuxed).

GPU state during this work: the RTX 5090 was held by the UniTime corpus run (~23–24 GiB) and,
part of the time, by `run_mulde.py` (~1.9 GiB). Every smoke here ran on CPU or inside a
`set_per_process_memory_fraction(0.09)` cap. Nothing was killed and the shared `HateVideo`
conda env's torch / torchvision / transformers were not touched (verified after the fact:
still `2.7.1+cu128 / 0.22.1+cu128 / 4.49.0`).

### S.1 Verdict table

| Repo | Commit | Attempted | Where it stopped | Verdict |
|---|---|---|---|---|
| **OV-AVEL** | `b5fe1d6` (2025-03-07) | ckpt path repointed; vendored ImageBind loaded; training-free v0 method (ImageBind A/V/T + argmax-agreement) run on 10 s of our video, CPU | ran to completion | **RAN** |
| **OmniVTG** | `7d67b82` (2026-05-28) | own venv `third_party/_venv/omnivtg` (torch 2.8.0+cu128, vllm 0.11.0, transformers 4.57.1); `zhengmh/OmniVTG-7B` fetched (16.6 GB); `demo.py::prepare_inputs` run end-to-end on our video | input pipeline works; the vLLM engine forward was not attempted — 15.5 GiB of bf16 weights against ~6 GiB free VRAM | **NEEDS-FREE-CARD** |
| **FLAM** (`openflam`) | `855d9e2` (2026-06-03) | own venv `third_party/_venv/flam` (transformers 4.56.1 over the shared torch 2.7.1); `kechenadobe/OpenFLAM` v1-base auto-fetched; framewise similarity run on 10 s of our wav, CPU | ran to completion | **RAN** |
| **FineLAP** | `ad82bc1` (2026-04-20) | HF `AutoModel` route only; `AndreasXi/FineLAP` fetched (1.87 GB); all five demo API calls run on our wav, CPU | ran to completion (the `scripts/infer.sh` / fairseq route was not attempted and stays a DROP) | **RAN** |
| **BaGLM** (`baglm`) | `118f864` (2025-10-26) | own venv `third_party/_venv/baglm`; torchcodec 0.4.0 (not the pinned 0.2.1+cu126) decodes our mp4 with the stock system ffmpeg; flash-attn disabled at runtime; the VSG stage (`score.forward_vsg` + `prompts/vsg/question.txt`) run on 8 segments of our video with `internvl2.5-1b` substituted for the published `internvl2.5-8b` | ran to completion, on CPU | **RAN** (published 8B backbone still needs the free card) |

### S.2 OV-AVEL — RAN

Three blockers were named in the earlier triage and all three turned out to be cheap.

1. The ImageBind checkpoint path is hardcoded at
   `proposed_method/ImageBind-main/imagebind/models/imagebind_model.py:507` to
   `/root/autodl-tmp/OV_AVEL/proposed_method/ImageBind-main/.checkpoints/imagebind_huge.pth`.
   Repointed at the campaign's existing `third_party/_ckpt/imagebind_huge.pth`, with
   `$IMAGEBIND_CKPT` as an override. This is the only on-disk edit and is exported as
   `scripts/repro_campaign/patches/OV-AVEL.patch`.
2. `pytorchvideo` missing from `requirements.txt`: already present in the shared env with the
   `torchvision.transforms.functional_tensor` re-export shim (`MODEL_ASSETS_STATUS §3.9`), so
   `imagebind/data.py` imports unchanged.
3. The OneDrive dataset link: not needed. What is dataset-bound is
   `baseline_v0_training_free.py`'s plumbing (`configs/opts.py`, `dataloader.py::OVAVE_Dataset`,
   a meta CSV + annotation JSON + a preprocessed 10-frame directory tree, all under
   `/root/autodl-tmp`), not the method.

`scripts/repro_campaign/smoke_ovavel.py` keeps the method and replaces only that plumbing. It
imports the repo's vendored ImageBind, calls `imagebind.data.load_and_transform_{text,audio,vision}`
exactly as `dataloader.py` does, and copies `compute_cross_modal_similarity` and
`postprocess_simm` verbatim from `baseline_v0_training_free.py` (only `.cuda()` → `.to(device)`,
so the smoke can stay off the shared card).

Smoke output, CPU, under two minutes including loading the 4.5 GiB checkpoint:

```
[smoke] 10 frames at 1 fps
[smoke] audio (1, 10, 1, 128, 204) visual (10, 3, 224, 224) text (6, 77)
[smoke] emb audio (1, 10, 1024) vision (1, 10, 1024) text (6, 1024)
[smoke] simm_at (1, 10, 6) range -0.0642..0.1632
[smoke] simm_vt (1, 10, 6) range 0.1025..0.3530
[smoke] is_event_flag (per second) = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
[smoke] event_flag argmax = 6   (== len(text_list), i.e. background)
```

Two facts a corpus run has to work around, both properties of the published method rather than
of the port:

- **The native output is a hard 0/1 flag, not a score.** `postprocess_simm` reduces both
  similarity matrices to their argmax class and emits `1` only where the audio argmax and the
  vision argmax coincide. A binary curve gives a degenerate ROC/PR curve (three points). To
  produce something the shared evaluator can rank, the continuous quantity underneath —
  `simm_at` / `simm_vt`, or their agreement-gated maximum over the hate rows — has to be dumped
  alongside the flag and reported as our adaptation, exactly as the LaGoVAD row already reports
  both the binary head and the similarity rows.
- **The audio loader is hardcoded to exactly 10 seconds.** `imagebind/data.py:140`
  (`audio_duration = 10`) truncates or stereo-zero-pads every waveform to 10 s and emits one
  1 s mel clip per second. Our videos run 20–600 s, so a corpus adapter has to loop the loader
  over consecutive 10 s chunks and concatenate. Note also `new_waveform = torch.zeros([2, tmp])`
  at line 163 assumes two channels — mono wavs must be up-mixed before the call.

To take it to the corpus through the shared evaluator: write
`idea-stage/repro_ovavel/curves/<DS>/<vid>.npz` with `rate = 1.0` (1 fps native, the
evaluator's `broadcast_to_4fps` handles the rest) and one array key per variant, e.g.
`is_event` (the published binary flag) and `simm_max_hate` (the continuous variant), then run
`eval_frame.py --curve-dir idea-stage/repro_ovavel/curves --variants is_event,simm_max_hate`.
Cost estimate from the smoke: ImageBind-Huge is 4.5 GiB of fp32 weights and the run is
~12 s per 10 s chunk on CPU, so this wants the GPU — it is the same backbone as the Wave 0
ZS-ImageBind row and would reuse its memory profile (peak 4.67 GiB).

### S.3 OmniVTG — NEEDS-FREE-CARD

The venv builds cleanly and everything except the language-model forward runs on our video.

Built at `third_party/_venv/omnivtg` (python 3.11, plain venv, no system site-packages) with:

```
third_party/_venv/omnivtg/bin/pip install 'vllm==0.11.0' 'transformers==4.57.1' \
    'qwen_vl_utils==0.0.14' 'decord==0.6.0'
third_party/_venv/omnivtg/bin/pip install 'gradio==6.5.1'   # demo.py imports gradio at line 1
```

That resolves to torch 2.8.0+cu128 / vllm 0.11.0 / transformers 4.57.1 / xformers 0.0.32.post1,
none of which touches the shared env. `torch.cuda.get_device_capability(0)` returns `(12, 0)`
inside the venv, so sm_120 is supported by this torch build. Two harmless resolver complaints
were emitted and ignored: `prometheus-fastapi-instrumentator 8.1.0 requires starlette<2.0.0,
but you have starlette 0.52.1`.

Weights: `zhengmh/OmniVTG-7B`, fetched in 465 s through `scripts/repro_campaign/hf_fetch.py`
(one attempt, no stall) with `--exclude '*.pth' '*.bin'`. Four safetensors shards, 16.6 GB /
15.45 GiB on disk. (`hf_fetch.py`'s own "31663 MiB" line double-counts the blob store and the
snapshot symlinks; `du` on the blobs reads 16 G.) `AutoConfig` reports
`model_type = qwen2_5_vl` and the processor loads as `Qwen2_5_VLProcessor`, i.e. it is a
Qwen2.5-VL-7B fine-tune.

The published input pipeline runs on our video with no modification. `demo.py::prepare_inputs`
was called directly with `demo.py`'s own constants (`TOTAL_VIDEO_TOKENS=3584`, `FPS=2`,
`MAX_FRAMES=768`, `PATCH_SIZE=14`) and its own `PROMPT` template:

```
qwen-vl-utils using decord to read video.
[smoke] prompt_token_ids len 2554
[smoke] pixel_values_videos (10656, 1176)
[smoke] video_grid_thw (222, 3)
```

222 sampled frames with per-frame `<xx.x seconds>` timestamps interleaved into a 2554-token
prompt — the whole coarse-to-fine timestamp convention that makes this method a temporal
grounder is in that number, and it works.

**Where it stopped and why:** the remaining step is `LLM(model=..., gpu_memory_utilization=0.8,
max_model_len=32768, limit_mm_per_prompt={"image":0,"video":768})` from `demo.py`'s `__main__`.
15.5 GiB of bf16 weights plus a vLLM KV cache against roughly 6 GiB free on the shared card is
not a marginal fit, and vLLM's `gpu_memory_utilization` is a fraction of *total* memory, so
there is no setting that both holds the weights and stays inside the 0.09 (~3 GiB) cap. It was
not attempted, deliberately: no OOM was provoked on a card another run is using. 4-bit
bitsandbytes loading would bring the weights to ~5.5 GiB, still above the cap and no longer the
published pipeline.

To finish this one: on a free card, `python demo.py --model zhengmh/OmniVTG-7B` reproduces the
authors' setup; for the corpus, drive `prepare_inputs` + `llm.generate` in a loop and parse the
`<answer>From X seconds to Y seconds</answer>` block. Note that the output is an **interval**,
not a curve, so it belongs on the F1@tIoU path the Wave 0 Qwen2.5-VL row already uses
(`eval_frame.py::interval_metrics`), not on the `<vid>.npz` + `rate` curve path.

### S.4 FLAM (openflam) — RAN

Venv at `third_party/_venv/flam`, built with `--system-site-packages` over the shared
`HateVideo` env so it inherits torch 2.7.1+cu128 (FLAM's pin is `torch >= 2.6.0, < 2.8.0`,
which the shared torch already satisfies) and overrides only what conflicts:

```
/home/jehc223/miniconda3/envs/HateVideo/bin/python -m venv --system-site-packages third_party/_venv/flam
third_party/_venv/flam/bin/pip install 'transformers==4.56.1' torchlibrosa lightning soundfile librosa
cd third_party/openflam && ../_venv/flam/bin/pip install --no-deps -e .
```

Result inside the venv: `2.7.1+cu128 4.56.1`; the shared env stays on 4.49.0. One ignorable
resolver complaint: `ms-swift 3.2.0 requires transformers<4.50` (ms-swift is not used here).

Weights auto-fetch on first `openflam.OpenFLAM(model_name="v1-base", default_ckpt_path="/tmp/openflam")`
— roberta-base for the text branch plus the `kechenadobe/OpenFLAM` checkpoint, 1.3 GB total,
through plain `huggingface_hub` (~2.5 MB/s here, about 6 minutes; a corpus run should pre-fetch
it with `hf_fetch.py` instead).

`scripts/repro_campaign/smoke_flam.py` follows `test/local_example.py` and changes only the
audio path and the text list. Smoke output, CPU:

```
[smoke] audio (1, 480000) @ 48000 Hz
[smoke] local similarity map shape (1, 5, 32) range 0.0000..0.5596
[smoke] native frame rate = 3.20 Hz (32 frames over 10.0 s)
[smoke]   'hateful speech targeting a group of people': mean 0.0004 max 0.0048
[smoke]   'a person shouting angrily': mean 0.0000 max 0.0000
[smoke]   'music': mean 0.0000 max 0.0000
[smoke]   'female speaker': mean 0.0000 max 0.0000
[smoke]   'male speaker': mean 0.0571 max 0.5596
```

Shape and range are right and the ordering is plausible for a talking-head clip. Two properties
to carry forward: **FLAM requires 48 kHz** (`SR = 48000`; our `data/AV2A_wav` files are resampled
by librosa on load, which is fine but costs time on a corpus run), and the **native rate is
3.2 Hz**, i.e. below the campaign's 4 fps grid, so the evaluator's piecewise-constant broadcast
applies. FLAM is audio-only — it is an audio floor in the same sense the freeze already records
for AV²A, not a hate detector.

Corpus route: `idea-stage/repro_flam/curves/<DS>/<vid>.npz`, one key per text query (the
`method="unbiased"` cross-product map row for that query), `rate = 3.2`. FLAM has no fixed
window limit in the API — it embeds whatever tensor it is handed — but memory grows with
duration, so long videos should be chunked and concatenated along the time axis.

### S.5 FineLAP — RAN

Only the HF `AutoModel` route was attempted, as instructed. `AndreasXi/FineLAP` fetched with
`hf_fetch.py` (1870 MiB in 105 s, one attempt). The repo files it ships
(`modeling_finelap.py`, `modeling_eat.py`, `eat_model{,_core}.py`, `configuration_*.py`) import
only `torch`, `torchaudio`, `numpy`, `timm.models.layers` and
`transformers.{PreTrainedModel, RobertaModel, RobertaTokenizer}` — **no fairseq**. That
confirms the split in the earlier triage: the fairseq dependency and the missing
`weights/EAT-base_epoch30_ft.pt` belong to `scripts/infer.sh` only, and that route stays a
DROP.

Run in the FLAM venv (transformers 4.56.1; FineLAP pins no version, and the shared env's 4.49.0
would very likely work too). `scripts/repro_campaign/smoke_finelap.py` follows `demo.py` and
changes only the audio path and the phrase list. Smoke output, CPU:

```
[smoke] global_text_embeds (1, 1024)
[smoke] global_audio_embeds (1, 1024)
[smoke] dense_audio_embeds (1, 64, 1024)
[smoke] clip_level_score (1,) = [2.0733007204398746e-06]
[smoke] frame_level_score (1, 5, 64) range 0.0000..0.9195
[smoke]   Speech: mean 0.0262 max 0.1211
[smoke]   Music: mean 0.0243 max 0.0499
[smoke]   Shouting: mean 0.0002 max 0.0006
[smoke]   Laughter: mean 0.1039 max 0.9195
[smoke]   Crowd: mean 0.0009 max 0.0037
```

All five demo API calls work. The one thing a corpus run must handle: **FineLAP has a fixed
input window**. `config.json` gives the EAT audio tower `img_size: [1024, 128]`, i.e. 1024 mel
frames at 100 Hz = 10.24 s, which is why a 222 s wav still produced exactly 64 output frames
(6.25 Hz). `modeling_finelap.py::load_audio` mixes to mono, resamples to 16 kHz, computes a
10 ms-shift kaldi fbank and then does `mel = mel[:target_len, :]` with `target_len = 1024` —
a hard truncation, not a resample. The 64 frames above therefore describe the *first 10.24 s
only*, not the whole video, and that is also why the `clip_level_score` is near-zero: the
caption described the whole clip and the model saw ten seconds of it. Sliding 10.24 s windows
with concatenation (or overlap-and-average) are required; this is the same shape of adaptation
OV-AVEL needs. One implementation detail for that: `load_audio` takes **file paths**, not
arrays, so windowing means either writing temporary wav chunks or patching `load_audio` to
accept a tensor.

Corpus route: `idea-stage/repro_finelap/curves/<DS>/<vid>.npz`, one key per phrase, `rate = 6.25`.
Audio-only, same caveat as FLAM.

### S.6 BaGLM — RAN (with a substituted backbone)

Both named blockers dissolved, and the third — "no single-video API" — is the real one.

**torchcodec.** The README pins `torchcodec==0.2.1+cu126` on torch 2.6.0+cu126 and asks for a
prebuilt ffmpeg-7.1 shared build wrapped in `LD_LIBRARY_PATH` shims. Neither is needed here:
`torchcodec==0.4.0` installs against the shared torch 2.7.1+cu128 and decodes our mp4 with the
stock system ffmpeg —

```
third_party/_venv/baglm/bin/pip install 'torchcodec==0.4.0'
>>> VideoDecoder('/home/jehc223/data/HateClipSeg/videos/bit_0dcMcI6hYjhw.mp4')
metadata 6678 30.00030000300003 222.597774
frames (3, 3, 480, 220)
```

`src/utils/video_utils.py` uses only `VideoDecoder(...)`, `.metadata`, `.get_frames_at(indices=)`
and `decoder[idx]`, all of which are unchanged in 0.4.0.

**flash-attn.** `t2v_metrics/.../internvl_model.py` sets `"use_flash_attn": True` on all
seventeen InternVL entries and there is no sm_120 wheel for the pinned `2.7.4.post1`. InternVL's
vision tower degrades on its own (`FlashAttention2 is not installed.` and it continues), but the
LLM half would ask transformers for `flash_attention_2` and raise. Flipping the flag to `False`
in the loaded dict at runtime is enough; no on-disk edit was made, so there is no BaGLM patch
file. Attention falls back to eager, with the expected warning
`Sliding Window Attention is enabled but not implemented for eager`.

Two dependencies the requirements do not make obvious and that will stop a fresh attempt cold:

- `src/utils/text_utils.py:7` calls `spacy.load("en_core_web_sm")` at **module import**, so
  every entry point dies with `OSError: [E050] Can't find model 'en_core_web_sm'` unless
  `python -m spacy download en_core_web_sm` was run first.
- `t2v_metrics/__init__.py` imports every backbone eagerly, including
  `llavaov_model.py`, which does `from llava.constants import ...` →
  `ModuleNotFoundError: No module named 'llava'`. Fixed with
  `pip install --no-deps 'git+https://github.com/LLaVA-VL/LLaVA-NeXT.git#egg=llava'`
  (`--no-deps` matters; with deps it drags its own torch/transformers pins in).

**The real blocker is the entry point, and it is structural.** `src/{coin,htstep,crosstask,
ego4d_goalstep}_eval.py` each bind a dataset class to the scorer, and the method's own semantics
are procedural: `prompts/vsg/question.txt` asks *"You are watching a video segment of someone
attempting to {goal}. What is the main action being performed in this exact moment? Options:
{choices}"*, and the Bayesian filter downstream (`src/bayes_filter.py`) uses **step-prerequisite
probabilities** — a prior that step B follows step A — to smooth the per-segment posteriors.
That prior has no analogue in hate video: there is no ordered procedure whose steps a video
walks through. The VSG stage transfers; the filter that gives the paper its name does not,
without inventing a step ordering we would then have to defend.

`scripts/repro_campaign/smoke_baglm.py` keeps the scorer path exactly
(`t2v_metrics.get_score_model` → `score.forward_vsg(video_data, 1, question_template=...)`,
`prompts/vsg/question.txt`, `segment_duration=2`, `sampling_fps=2`) and replaces only
`dataset.COIN.__getitem__`, building the same `{"video_uid", "videos", "task_id", "texts"}` dict
from our mp4, with the campaign's hate categories as the "steps". Smoke output:

```
[smoke] video metadata {'num_frames': 6678, 'fps': 30.0003, 'duration': 222.597774}
[smoke] 32 frames at 2 fps -> 8 segments of 2 s
FlashAttention2 is not installed.
[smoke] 8 segments, each (4, 3, 448, 448)
[smoke] vsg scores (16, 7)  (segments x (6 steps + 'None of the above'))  range 0.0012..0.7383
[smoke] row sums [0.9949687123298645, 0.9949687123298645, 0.9961512088775635, 0.9961512088775635]
[smoke] per-second argmax [3, 3, 1, 1, 3, 3, 3, 3, 0, 0, 0, 0, 0, 0, 1, 1]
```

Rows are proper distributions over the seven options, and the argmax moves across the video, so
the stage is producing a real per-segment posterior rather than a constant.

**Two substitutions, both recorded rather than hidden.** (1) The backbone is `internvl2.5-1b`,
not the published `internvl2.5-8b`; 1B is in the repo's own `INTERNVL2_MODELS` table, so this is
a supported configuration, but any reported number must come from the 8B. (2) It ran on CPU and
took about 8 minutes for 8 segments. That is not a memory decision — `InternVL2Model.load_model`
has its `.to(self.device)` commented out at `internvl_model.py:315` and then sets
`self.device = next(self.model.parameters()).device`, so the model silently stays wherever
`from_pretrained` put it, which without a `device_map` is CPU. A GPU run needs that one line
restored (or `device_map` passed), which is a patch a future attempt should make and export.

At 8 min per 8 segments on CPU with a 1B model, the corpus is out of reach on CPU; on the free
card with the 8B this is the most expensive of the five by a wide margin. Corpus route if it is
ever wanted: `idea-stage/repro_baglm/curves/<DS>/<vid>.npz` with `rate = 0.5` (2 s segments) and
one key per hate category from the VSG posterior, plus a decision about whether to run
`bayes_filter.py` at all given the prerequisite-prior mismatch above.

### S.7 Files produced

New, tracked:

```
scripts/repro_campaign/smoke_ovavel.py
scripts/repro_campaign/smoke_flam.py
scripts/repro_campaign/smoke_finelap.py
scripts/repro_campaign/smoke_baglm.py
scripts/repro_campaign/patches/OV-AVEL.patch
```

New, untracked (under `third_party/`, per §4 of `MODEL_ASSETS_STATUS.md`):

```
third_party/_venv/omnivtg          torch 2.8.0+cu128, vllm 0.11.0, transformers 4.57.1, gradio 6.5.1
third_party/_venv/flam             --system-site-packages over HateVideo, transformers 4.56.1, openflam -e
third_party/_venv/baglm            --system-site-packages over HateVideo, torchcodec 0.4.0, spacy 3.8.7
                                   + en_core_web_sm, llava (LLaVA-NeXT, --no-deps)
```

Downloads (HF cache, ~/.cache/huggingface): `zhengmh/OmniVTG-7B` 16.6 GB,
`AndreasXi/FineLAP` 1.87 GB, `OpenGVLab/InternVL2_5-1B` 1.8 GB, plus
`kechenadobe/OpenFLAM` + roberta-base 1.3 GB under `/tmp/openflam` (that path is FLAM's own
default and should be moved somewhere durable before a corpus run).

Fetch logs: `logging/runs/repro_tryinstall/{omnivtg_fetch,omnivtg_pip,finelap_fetch,baglm_pip,internvl1b_fetch}.log`.
