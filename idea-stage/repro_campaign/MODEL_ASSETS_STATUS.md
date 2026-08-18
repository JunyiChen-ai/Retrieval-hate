# Label-free frame-level repro campaign — Phase A route 2: model assets status

Scope: clone every method repo with official code, install its dependencies,
download its checkpoints, and smoke-test each one on a single HateClipSeg video.
Plan: `/home/jehc223/.claude/plans/fluffy-napping-blum.md`.
Method list source: `research-wiki/LABELFREE_FRAMELEVEL_BASELINES_2026-08-18.md`.

Machine: single RTX 5090 (32 GB, sm_120), python 3.11.8, torch 2.7.1+cu128,
conda env `HateVideo`. The GPU is shared with the parallel feature-extraction
job, so every smoke below was run short and small.

Smoke video for all methods: `~/data/HateClipSeg/videos/bit_0dcMcI6hYjhw.mp4`
(222.6 s, 6678 frames at ~30 fps, 5.0 MB).

---

## 1. Verdict table

| # | Method | Repo @ commit | Dependency plan | Checkpoints | Smoke result | Verdict |
|---|---|---|---|---|---|---|
| 1 | **Qwen2.5-VL-7B native grounding** | no repo — our harness `scripts/repro_campaign/qwen25vl_grounding.py` | shared `HateVideo` | `Qwen/Qwen2.5-VL-7B-Instruct` (already local, 16 GB) | 1 video, 32 frames, 4-bit: emitted `"The event happens in 15.8 - 21.9 seconds."` → parsed span `[15.8, 21.9]`; 2.5 s/video; **peak 16.6 GiB** | **READY** |
| 2 | **UniTime** (NeurIPS 2025) | `lzq5/UniTime` @ `a557bbd5` (2026-05-20) | shared `HateVideo` (transformers 4.49.0 matches its pin exactly) | LoRA `zeqianli/UniTime` (162 MB) on local `Qwen/Qwen2-VL-7B-Instruct` | 1 video: `pred_relevant_windows = [[0.5, 146.4]]`, `pred_relevant_windows_mr_seg = [[0.2, 32.1, 63.9, 95.8, 127.6]]`; 26.6 s/video | **READY** |
| 3 | **LaGoVAD** (ICLR 2026) | `Kamino666/LaGoVAD-PreVAD` @ `e2b93f85` (2026-05-07) | shared `HateVideo` + `lightning==2.3.3`, `hydra-core`, `ftfy` (installed, no torch/transformers change) | `ckpts/best.ckpt` 218 MB from the repo's Google Drive link + `openai/clip-vit-base-patch16` (1.1 GB) | 1 video → 835 frames (every 8th): `mul_score` shape `(2, 835)`, range 0.0073–0.9897 with free-text hate queries; ~7 s/video | **READY** (see caveat §3.1) |
| 4 | **MULDE** (CVPR 2024) | `jakubmicorek/MULDE-...` @ `f821b965` (2024-06-19) | shared `HateVideo`, nothing extra | none — trains on our own feature vectors | ran its bundled toy pipeline for 3 epochs, train + noise-free log-density passes complete | **READY** |
| 5 | **LAVAD** (CVPR 2024) | `lucazanella/lavad` @ `1ad46c66` (2024-07-15) | shared `HateVideo` + `llama_hf` shim (§3.2) + `pytorchvideo` | BLIP-2 `Salesforce/blip2-opt-6.7b-coco` 31 GB, ImageBind-Huge 4.8 GB, Llama-2-13b-chat via `NousResearch/Llama-2-13b-chat-hf` 26 GB — **all local** | stage-by-stage: frame extraction 6678 JPEGs; BLIP-2 captioner produced 14 captions (e.g. `"a person is texting on a cell phone while driving"`) at 1.3 batch/s; ImageBind text/vision embeddings verified; Llama-2-13b-chat via the shim in NF4 scored 4 test captions `[0.2, 0.3, 0.3, 0.8]` — the gun caption highest — all parsed by LAVAD's own `_parse_score` regex. Weights 6.88 GiB, peak 7.66 GiB, 0.9 s for 4 dialogs | **READY** (full 7-stage chain not yet run end-to-end, §3.11) |
| 6 | **URF-HVAA** (NeurIPS 2025) | `Rathgrith/URF-HVAA` @ `ea993487` (2025-12-06) | shared `HateVideo` + `llama_hf` shim + `ffmpeg-python` | `DAMO-NLP-SG/VideoLLaMA3-7B` (31 GB incl. revisions) + `VL3-SigLIP-NaViT` 1.6 GB + Llama-3.1-8B via `NousResearch/Meta-Llama-3.1-8B-Instruct` — **all local** | `video_pre_caption.py` produced **418 caption entries** (keyed by frame index, step 16) for the 222 s video; the captions transcribe on-screen text, which is exactly the modality gap the OCR ruling identified. Llama-3.1-8B via the shim scored `[0.1, 0.8, -1]`; bf16 weights 14.96 GiB, peak 15.08 GiB | **READY** |
| 7 | **AV²A** (CVPR 2025) | `eitan159/AV2A` @ `b0d6db8b` (2025-10-21) | **own venv** `third_party/_venv/av2a` (§3.3) — its vendored LanguageBind needs `transformers==4.31.0`, incompatible with the shared 4.49.0 | LanguageBind Video_FT / Audio_FT / Image + `lb203/LanguageBind_Image` (6.8 GB, **done**); laion-CLAP default ckpt auto-fetched at first use | both backbones import cleanly in the venv (`LanguageBind` + `laion_clap`) | **READY-TO-ADAPT** (§3.4) |
| 8 | **ZS-ImageBind** (LAVAD's baseline) | ImageBind ships inside `lavad/libs/ImageBind` | shared `HateVideo` + `pytorchvideo` (installed `--no-deps`; needed a `functional_tensor` re-export shim, §3.9) | `imagebind_huge.pth` 4.8 GB from `dl.fbaipublicfiles.com` (**done**, ~100 MB/s) | 8 frames + 2 prompts: text `(2, 1024)`, vision `(8, 1024)`, softmax similarity spans 0.0–1.0; peak 4.67 GiB | **READY** |
| 9 | **SeViLA** | `Yui010206/SeViLA` @ `419e7281` (2024-01-14) | **own venv required** — pins `timm==0.4.12` (shared env is on 1.0.15, which ImageBind and others depend on) + `fairscale`, LAVIS-based | `sevila_pretrained.pth` 814 MB (**done**) | — | **NEEDS-OWN-VENV** |
| 10 | **CLAP** (CVPR 2024) | `AnasEmad11/CLAP` @ `3dcaadc1` (2024-09-30) | shared env; `pynvml` installed, **`visdom` fails to build** (only used for plots) | none — trains on our features | import probe only | **NEEDS-ADAPTATION** (§3.5) |
| 11 | **T3AL** | `benedettaliberatori/T3AL` @ `dfbbbc1c` (2024-09-11) | needs `open_clip` (CoCa); not yet installed | pre-extracted CoCa features via Google Drive (THUMOS/ANet only — we'd extract our own) | — | **PENDING** |
| 12 | **VADTree** | `wenlongli10/VADTree` @ `04dc1df3` (2026-06-09) | not yet attempted | needs **three** extra models: EfficientGEBD weights, `lmms-lab/LLaVA-Video-7B-Qwen2`, `deepseek-ai/DeepSeek-R1-Distill-Qwen-14B` (~45 GB more) | — | **DEFERRED** (heaviest of the set) |
| 13 | **EventVAD** | `YihuaJerry/EventVAD` @ `25cacd88` (2025-07-09) | **two separate envs by design**, both incompatible with sm_120 as pinned (§3.10) | needs `raft-things.pth` **and** the RAFT package itself, which is *not* vendored | — | **NEEDS-OWN-VENV**, high risk |
| 14 | **ZS-CLIP** (LAVAD's baseline) | **no code in the lavad repo** — the paper reports the number but ships only the LAVAD pipeline | our own harness (cosine similarity of frame CLIP features vs. prompts) | `openai/clip-vit-large-patch14-336` already local | — | **TO WRITE** (trivial; Phase B, reuses the existing CLIP cache) |

### Try-install group (30-min budget each, triaged read-only)

| Repo | Inference entry point | Weights | Blocker | Verdict |
|---|---|---|---|---|
| **OmniVTG** | `demo.py` (single video) | `zhengmh/OmniVTG-7B`, public | `vllm==0.11.0` forces torch 2.8.0 + transformers ≥4.55 | **NEEDS-OWN-VENV**, otherwise clean |
| **OV-AVEL** | `proposed_method/ImageBind-main/baseline_v0_training_free.py` | ImageBind-Huge (URL verified HTTP 200) | ckpt path hardcoded to `/root/autodl-tmp/...`; `pytorchvideo` missing from its requirements; dataset only via a OneDrive link | **LIKELY-INSTALLABLE**, lowest friction of the group |
| **FLAM** (`openflam`) | `test/global_example.py` | `kechenadobe/OpenFLAM`, public, auto-fetched | `transformers==4.56.1` hard pin. **Audio-only** model | **NEEDS-OWN-VENV** |
| **FineLAP** | `demo.py` via `AutoModel.from_pretrained("AndreasXi/FineLAP", trust_remote_code=True)` | HF repo public; the `scripts/infer.sh` route wants `weights/EAT-base_epoch30_ft.pt`, which is neither in the repo nor linked | the `infer.sh` route needs `fairseq` (unbuildable on py3.11/torch 2.7). **Audio-only** | **NEEDS-OWN-VENV** via the HF route only; the local-ckpt route is a **DROP** |
| **BAGLM** (`baglm`) | `src/{coin,htstep,...}_eval.py`; no single-video API | `OpenGVLab/InternVL2_5-8B` public; precomputed scores only via 4 Google Drive links | `torchcodec==0.2.1+cu126` + a custom prebuilt ffmpeg-7.1 with `LD_LIBRARY_PATH` hooks; `flash-attn==2.7.4.post1` pin | **NEEDS-OWN-VENV**, high effort |
| **ZS-STVG** (`LLaVA_Next_STVG`) | `stvg/hc_llava_next_video.py`, HC-STVG2 only | `lmms-lab/LLaVA-NeXT-Video-7B-DPO` public; needs 14 GB of precomputed proposals | three mutually inconsistent dep specs; `requirements.txt` pins `byted-torch-monitor`, an internal ByteDance package not on PyPI; proposal-extraction code never released | **DROP** for this campaign (effort ≫ value) |
| **DASM** (`ADSM`) | none | — | repo on disk is 3 files: `LICENSE`, `README.md`, `img/pipeline.png`. **Zero Python.** | **DROP** — stub |
| **LAVIDA** | none — only `train_acc.py` | **no checkpoint released anywhere** | README says "Data Preparation: *(Wait for further updates)*" | **DROP** — training-only, no weights |

---

## 2. Disk and download accounting

All downloads complete (2026-08-19 02:13). **HF cache went 60 GB → 161 GB, i.e.
101 GB downloaded**, plus 5.8 GB of direct-URL checkpoints in `third_party/_ckpt`
and 11 GB of repos + venv. Disk: 401 GB → ~500 GB used of 1.8 T; **1.3 T free.**

| Asset | Size | Location |
|---|---|---|
| BLIP-2 opt-6.7b-coco (safetensors only) | 31 GB | `~/.cache/huggingface/hub` |
| `NousResearch/Llama-2-13b-chat-hf` (safetensors) | 26 GB | `~/.cache/huggingface/hub` |
| `DAMO-NLP-SG/VideoLLaMA3-7B` + `VL3-SigLIP-NaViT` | 32 GB | `~/.cache/huggingface/hub` |
| `NousResearch/Meta-Llama-3.1-8B-Instruct` | 16 GB | `~/.cache/huggingface/hub` |
| ImageBind-Huge | 4.8 GB | `third_party/_ckpt/imagebind_huge.pth` |
| `Qwen/Qwen2.5-0.5B-Instruct` (shim smoke stand-in) | 1.9 GB | `~/.cache/huggingface/hub` |
| LanguageBind Video_FT / Audio_FT / Image | 5.2 GB | `~/.cache/huggingface/hub` |
| `lb203/LanguageBind_Image` (AV²A tokenizer) | 1.6 GB | `~/.cache/huggingface/hub` |
| `openai/clip-vit-base-patch16` (LaGoVAD) | 1.1 GB | `~/.cache/huggingface/hub` |
| LaGoVAD `best.ckpt` + config | 218 MB (+210 MB zip) | `third_party/_ckpt/lagovad` |
| UniTime LoRA | 162 MB | `third_party/_ckpt/unitime` |
| SeViLA `sevila_pretrained.pth` | 814 MB | `third_party/_ckpt/sevila` |
| 19 method repos (git) | 11 GB incl. venv | `third_party/` |
| AV²A venv (own torch copy) | 7.5 GB | `third_party/_venv/av2a` |
| LAVAD smoke frames (1 video) | 288 MB | `idea-stage/repro_campaign/smoke/lavad_ds` |

Not downloaded (deliberate, decision pending): the four other BLIP-2 variants
LAVAD's ensemble script lists (~120 GB, §3.11b); VADTree's LLaVA-Video-7B-Qwen2 +
DeepSeek-R1-Distill-Qwen-14B + EfficientGEBD (~45 GB); T3AL's CoCa features;
EventVAD's RAFT weights. No disk risk either way.

**Storage warning for Phase C:** LAVAD works on pre-extracted JPEG frames. One
222 s video at native fps gives 6678 frames / 283 MB. At that rate the 2689
HateMM/MHC videos plus HateClipSeg would need roughly **700–800 GB of JPEGs**.
LAVAD's own pipeline captions at 1 fps, so extract at reduced fps (or stream
frames) rather than replicating `00_extract_frames.sh` verbatim.

### Network finding (this cost ~40 minutes, worth recording)

Raw internet here is ~64 MB/s (cachefly), but a single connection to the HF CDN
gets **0.9 MB/s**. `hf_transfer` fixes the throughput (~20 MB/s) but **hangs on
the tail of nearly every file** — it stopped 800 KB short of a 599 MB blob and
never returned. Neither mode alone works. `scripts/repro_campaign/hf_fetch.py`
wraps `huggingface-cli` with a stall watchdog that kills and restarts on no
progress, **alternating hf_transfer on/off** between attempts: hf_transfer does
the bulk fast, plain mode resumes byte-exactly and finishes the tail. Every
large download in this campaign should go through it. Google Drive, by contrast,
gives a clean 35 MB/s (LaGoVAD's 210 MB ckpt in 6 s).

---

## 3. Adaptations and caveats (each is a deviation to carry into Phase B/C)

### 3.1 LaGoVAD binary head is constant across frames
With `verbalizer_type=None` (forced by the upstream demo script) `bin_score` is
identical at every frame — 0.0023 with UCF class names, 0.0037 with our
free-text hate queries. The per-frame signal lives entirely in the similarity
matrix (`cap_sim_mat` / `cls_sim_mat`), which does vary (0.0073–0.9897). Phase C
must score off the hate-query row of the similarity matrix, not `bin_score`.
Added `--queries` to `src/end2end_inference.py` so the free-text definition — the
whole point of LaGoVAD — is reachable from the CLI, plus an `.npz` dump of both
curves (upstream only rendered a PNG). Also symlinked `simhei.ttf` to matplotlib's
bundled DejaVuSans; without it the script crashes at the plotting stage after
inference has already succeeded.

### 3.2 `llama_hf` shim replaces Meta's `llama` package
`third_party/_shim/llama_hf/`. Both LAVAD and URF-HVAA call
`llama.Llama.build(...)` on **original-format Meta checkpoints**, which are gated
on HF; `llama-2-13b-chat` is additionally sharded model-parallel MP=2 and needs
two GPUs under Meta's own loader. The shim exposes exactly `Dialog` and `Llama`,
reproduces `chat_completion`'s return contract, maps `ckpt_dir` basenames to
ungated mirrors (`NousResearch/Llama-2-13b-chat-hf`,
`NousResearch/Meta-Llama-3.1-8B-Instruct`), and keeps Meta's sampling semantics
(temperature == 0 → greedy argmax). `LLAMA_HF_4BIT=1` switches to NF4 (~7.5 GB)
for the 13B on a shared card. The import line in
`lavad/src/models/llm_anomaly_scorer.py` and
`URF-HVAA/src/llm_anomaly_scorer.py` was redirected to it; both edits carry an
`# ADAPTED` comment.

### 3.3 AV²A needs its own venv
`third_party/_venv/av2a`. Its vendored LanguageBind imports `_expand_mask` from
`transformers.models.clip.modeling_clip`, removed after transformers ~4.36; the
shared env is on 4.49.0 and cannot move (UniTime pins 4.49.0 exactly). The venv
has torch 2.7.1+cu128 with `transformers==4.31.0`, `tokenizers==0.13.3`,
`peft==0.5.0`, `laion-clap==1.1.6`. One further fix: `pytorchvideo==0.1.5` imports
`torchvision.transforms.functional_tensor`, deleted in torchvision ≥0.17, so a
re-export shim was written into the venv's torchvision.

### 3.4 AV²A input format
`main.py` takes `--video_dir_path` + `--audio_dir_path` and expects LLP/AVE
layout with a candidates file; it does not read raw mp4 directly. Phase D needs a
demux + candidate-label adapter for HateMM/MHC/HateClipSeg. Also note the
landscape doc's own caveat: AV²A localises *acoustic events*, so it is an audio
floor, not a hate detector.

### 3.5 CLAP is a federated-learning codebase
`src/server/fedavg.py` expects UCF-Crime concatenated features plus scene
partition `.pkl` files. Feeding our own features requires writing a partition
+ dataset adapter. `visdom` (plot-only) fails to build on py3.11 and should be
stubbed rather than installed.

### 3.6 UniTime: two patches
1. Its vision tower hardcodes `attn_implementation="flash_attention_2"`; no
   flash-attn wheel exists for torch 2.7/cu128/sm_120 and building 2.7.2 from
   source is an hour, so it was switched to `sdpa`.
2. `models/qwen2_vl.py::forward` materialises `[B, T, 152k]` fp32 logits. On a
   222 s video at 2 fps that is a 9.4 GiB allocation and OOMs a 32 GB card. It
   now slices to the last position when `labels is None` (generation only) —
   mathematically identical, since `generate()` consumes only that row.
3. The released `adapter_config.json` hardcodes the authors' base-model path
   (`/raid/haoningwu/...`); repointed at the local Qwen2-VL-7B-Instruct snapshot.

### 3.7 Qwen2.5-VL harness fidelity and memory
Prompt and parser are copied verbatim from the lmms-eval
`temporal_grounding_charades` task (`charades.yaml` `pre_prompt`/`post_prompt`,
`utils.py::extract_time`), which is the harness TempSamp-R1 reports its
Qwen2.5-VL-7B zero-shot row under; the only substitution is the query sentence.
Greedy, `max_new_tokens=50`, 32 frames — all lmms-eval defaults.
**Memory:** the 4-bit weights are only 5.8 GiB but the run peaks at **16.6 GiB**,
because the ViT sees 32 frames × 771 patches under sdpa. For a shared GPU, lower
`--max-pixels` (default 151200 ≈ 192 merged tokens/frame). In bf16 the same run
would be ~26 GiB — it fits a free 32 GB card but not a shared one.

### 3.8 Gated-model substitutions
No HF token is configured on this machine, and every `meta-llama/*` repo is
gated. Substituted ungated mirrors: `NousResearch/Llama-2-13b-chat-hf` and
`NousResearch/Meta-Llama-3.1-8B-Instruct` (the latter also ships
`original/consolidated.00.pth`, so URF-HVAA's official loader stays an option —
8B is MP=1 and fits one card). If a token is ever added, both can be swapped back
to the canonical repos without touching code.

### 3.9 `pytorchvideo` needs a torchvision shim in the shared env too
ImageBind's `data.py` imports `pytorchvideo.transforms`, which imports
`torchvision.transforms.functional_tensor` — deleted in torchvision ≥0.17. The
same re-export shim used for the AV²A venv (§3.3) was written into the shared
env's torchvision. `pytorchvideo` itself was installed `--no-deps` so it could
not drag old pins into the shared env. ImageBind also needs
`PYTHONPATH=lavad/libs/ImageBind` because its `__init__.py` does an absolute
`from imagebind import data`.

### 3.10 EventVAD is split across two mutually incompatible envs
Its own README asks for `conda create -n event_seg` (torch 2.1.0+cu121,
transformers 4.33.2, `salesforce-lavis`) and a separate `conda create -n score`
(torch **2.2.0+cu118**, `flash-attn==2.5.8`, VideoLLaMA2 from git). Neither
pinned torch supports sm_120 — cu118 will not run on Blackwell at all — and
flash-attn 2.5.8 has no sm_120 wheel. Additionally `feature_extractor.py` does
`from RAFT.core.raft import RAFT` and hardcodes `model='/path/raft-things.pth'`,
but no `RAFT/` directory exists in the repo, so `princeton-vl/RAFT` must be
cloned separately. Runnable only by rebuilding both envs on torch 2.7.1+cu128
with flash-attn dropped.

### 3.11 The two things Phase C must decide before running LAVAD at scale

**(a) Llama-3.1-8B-Instruct refuses to score violent/hateful captions.**

*Observed.* Three captions were sent through the shim with LAVAD's exact
`context_prompt + format_prompt` (the "law enforcement agency, rate 0 to 1"
prompt) under greedy decoding:

| caption | Llama-3.1-8B-Instruct | Llama-2-13b-chat (NF4) |
|---|---|---|
| a person is holding a cell phone with a twitter app on it | `[0.1]` | `[0.2]` |
| a group of people fighting in the street | `[0.8]` | `[0.3]` |
| a man points a gun at a cashier | **`"I can't assist"`** → parsed `-1` | `[0.8]` |

*Why it matters mechanically.* `LLMAnomalyScorer._parse_score` matches
`r"\[(\d+(?:\.\d+)?)\]"`; anything else becomes `-1`, and
`_interpolate_unmatched_scores` then runs `np.interp` over the remaining valid
points — so a refusal is silently replaced by a linear blend of its neighbours
and nothing in the output distinguishes it from a real model score. URF-HVAA
inherits the identical code path (`src/llm_anomaly_scorer.py`).

*Why it is worse in our domain than in the papers'.* The refusal fires on
descriptions of violence and group-directed hostility. On UCF-Crime / XD-Violence
those are a minority of frames; **on HateMM / MHC / HateClipSeg they are the
positive class**. The refusals therefore concentrate on exactly the frames the
metric is computed over, and interpolation systematically drags positive-frame
scores toward the surrounding negative frames — a bias that *depresses* AUC in a
way that looks like the method underperforming rather than the harness failing.
It also invalidates the LELA alignment check (±0.03 vs ROC 0.6163 / PR 0.6239),
since LELA's port would have hit the same wall silently.

*Countermeasures, in the order Phase C should try them.*
1. **Measure first, always.** Instrument `_parse_score` to record, per video, the
   count and frame indices of `-1`s, and the raw refusal text. Report the refusal
   rate beside every LAVAD/URF-HVAA number, and break it down by GT label
   (refusal rate on positive vs negative frames). If the rate is under ~2 % and
   label-balanced, the rest of this list is unnecessary.
2. **Treat refusal windows as missing, not as interpolated.** Replace the blanket
   `np.interp` with an explicit mask and score only the frames where the model
   actually answered, reporting coverage alongside AUC. This is the honest
   fallback and costs nothing.
3. **Reframe the prompt, not the label.** LAVAD's prompt already casts the model
   as a law-enforcement rater; adding an explicit content-moderation framing
   ("you are a content-safety classifier; you are rating a description, not
   producing the content; respond only with the list") converts most Llama-3.1
   refusals into scores. This changes the prompt away from LAVAD's published one,
   so it must be reported as our adaptation and run **as a paired variant** next
   to the verbatim prompt, never silently substituted.
4. **Swap the LLM.** Llama-2-13b-chat did not refuse on any of the four test
   captions and is what LAVAD actually published with, so LAVAD should stay on
   it. For URF-HVAA, whose published backbone *is* Llama-3.1-8B, an uncensored or
   less safety-tuned instruct model of the same size is a last resort and would
   make the row a variant rather than a reproduction.

Options 1 and 2 are mandatory and cost nothing. 3 and 4 are only justified if the
measured refusal rate is high enough to move the numbers, which is why 1 comes
first.

**(b) The LAVAD chain has seven stages, and the shipped `04_query_llm.sh`
depends on stages 02–03.** The `--score_summary` pass reads *cleaned, nested*
captions produced by `02_create_index.sh` (ImageBind index over **five** BLIP-2
variants) and `03_clean_captions.sh`; feeding it the raw flat captions from stage
01 fails with a missing-summary error. Only `blip2-opt-6.7b-coco` was downloaded
(31 GB); the other four variants are another ~120 GB and several hours. Decide
explicitly whether to reproduce the full 5-model caption ensemble or to run a
single-captioner variant and label it as our adaptation — the LELA alignment
target was presumably computed with the full ensemble.

### 3.12 Llama-2 mirrors ship no chat template
`NousResearch/Llama-2-13b-chat-hf` has no `tokenizer.chat_template`, so
`apply_chat_template` raises. The shim now falls back to the canonical Llama-2
`[INST] <<SYS>>…` layout, which is what Meta's own `ChatFormat` emits for a
`[system, user]` pair — so prompt formatting stays faithful to LAVAD's original.


---

## 4. Git status

`third_party/` is excluded from git (`.gitignore` rule `third_party/*` with a
`!third_party/actionformer` negation so the previously-vendored ActionFormer stays
tracked). Method repos, downloaded checkpoints and the AV²A venv therefore never
enter the ledger; their exact commits are pinned in the table above. Committed
from this route: `scripts/repro_campaign/` (harness + downloader) and this file.

The `# ADAPTED` edits described in §3 live inside `third_party/`, which is
untracked, so they are exported as tracked patches:

```
scripts/repro_campaign/patches/{UniTime,LaGoVAD-PreVAD,lavad,URF-HVAA}.patch
scripts/repro_campaign/shim/llama_hf/          # the shim itself, symlinked in as third_party/_shim
```

To rebuild the environment from scratch: clone the repos at the commits in §1,
`git apply` the four patches, `ln -s scripts/repro_campaign/shim third_party/_shim`,
then run `scripts/repro_campaign/download_assets{,2}.sh`. Weight files are kept
out of the ledger by the `third_party/*` rule plus the pre-existing `*.pt`/`*.pth`
patterns and added `*.ckpt`/`*.safetensors` patterns; the HF cache lives outside
the repo at `~/.cache/huggingface`.
