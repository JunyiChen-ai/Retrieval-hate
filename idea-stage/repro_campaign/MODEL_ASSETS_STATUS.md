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
| 5 | **LAVAD** (CVPR 2024) | `lucazanella/lavad` @ `1ad46c66` (2024-07-15) | shared `HateVideo` + `llama_hf` shim (§3.2) | BLIP-2 `Salesforce/blip2-opt-6.7b-coco` (**downloading**, 25/31 GB), ImageBind-Huge 4.5 GB (**queued**), Llama-2-13b-chat via `NousResearch/Llama-2-13b-chat-hf` (**queued**, 26 GB) | frame extraction verified: 6678 JPEGs / 283 MB from one video. Captioner + LLM + ImageBind stages not yet smoked (weights still downloading) | **PENDING-DOWNLOAD** |
| 6 | **URF-HVAA** (NeurIPS 2025) | `Rathgrith/URF-HVAA` @ `ea993487` (2025-12-06) | shared `HateVideo` + `llama_hf` shim | `DAMO-NLP-SG/VideoLLaMA3-7B` 16 GB (**queued**), Llama-3.1-8B via `NousResearch/Meta-Llama-3.1-8B-Instruct` (**queued**) | not yet smoked (weights downloading) | **PENDING-DOWNLOAD** |
| 7 | **AV²A** (CVPR 2025) | `eitan159/AV2A` @ `b0d6db8b` (2025-10-21) | **own venv** `third_party/_venv/av2a` (§3.3) — its vendored LanguageBind needs `transformers==4.31.0`, incompatible with the shared 4.49.0 | LanguageBind Video_FT / Audio_FT / Image + `lb203/LanguageBind_Image` (6.8 GB, **done**); laion-CLAP default ckpt auto-fetched at first use | both backbones import cleanly in the venv (`LanguageBind` + `laion_clap`) | **READY-TO-ADAPT** (§3.4) |
| 8 | **SeViLA** | `Yui010206/SeViLA` @ `419e7281` (2024-01-14) | not yet attempted | `sevila_pretrained.pth` 814 MB (**downloading**) | — | **PENDING** |
| 9 | **CLAP** (CVPR 2024) | `AnasEmad11/CLAP` @ `3dcaadc1` (2024-09-30) | shared env; `pynvml` installed, **`visdom` fails to build** (only used for plots) | none — trains on our features | import probe only | **NEEDS-ADAPTATION** (§3.5) |
| 10 | **T3AL** | `benedettaliberatori/T3AL` @ `dfbbbc1c` (2024-09-11) | needs `open_clip` (CoCa); not yet installed | pre-extracted CoCa features via Google Drive (THUMOS/ANet only — we'd extract our own) | — | **PENDING** |
| 11 | **VADTree** | `wenlongli10/VADTree` @ `04dc1df3` (2026-06-09) | not yet attempted | needs **three** extra models: EfficientGEBD weights, `lmms-lab/LLaVA-Video-7B-Qwen2`, `deepseek-ai/DeepSeek-R1-Distill-Qwen-14B` (~45 GB more) | — | **DEFERRED** (heaviest of the set) |
| 12 | **EventVAD** | `YihuaJerry/EventVAD` @ `25cacd88` (2025-07-09) | not yet attempted | needs RAFT weights | — | **PENDING** |

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

Downloaded so far (2026-08-19 01:45):

| Asset | Size | Location |
|---|---|---|
| BLIP-2 opt-6.7b-coco (safetensors only) | 25 GB of ~31 GB, in flight | `~/.cache/huggingface/hub` |
| LanguageBind Video_FT / Audio_FT / Image | 5.2 GB | `~/.cache/huggingface/hub` |
| `lb203/LanguageBind_Image` (AV²A tokenizer) | 1.6 GB | `~/.cache/huggingface/hub` |
| `openai/clip-vit-base-patch16` (LaGoVAD) | 1.1 GB | `~/.cache/huggingface/hub` |
| LaGoVAD `best.ckpt` + config | 218 MB (+210 MB zip) | `third_party/_ckpt/lagovad` |
| UniTime LoRA | 162 MB | `third_party/_ckpt/unitime` |
| SeViLA `sevila_pretrained.pth` | 814 MB, in flight | `third_party/_ckpt/sevila` |
| 19 method repos (git) | 11 GB incl. venv | `third_party/` |
| AV²A venv (own torch copy) | 7.5 GB | `third_party/_venv/av2a` |
| LAVAD smoke frames (1 video) | 288 MB | `idea-stage/repro_campaign/smoke/lavad_ds` |

Still queued: VideoLLaMA3-7B 16 GB, VL3-SigLIP-NaViT ~1.5 GB, Llama-3.1-8B-Instruct
HF-format 16 GB, Llama-2-13b-chat-hf 26 GB, ImageBind-Huge 4.5 GB. **Projected
campaign total ≈ 115 GB of weights + 11 GB repos/venv.** Disk went 401 G → 440 G
used of 1.8 T; 1.3 T free. No disk risk.

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

---

## 4. Git status

`third_party/` is excluded from git (`.gitignore` rule `third_party/*` with a
`!third_party/actionformer` negation so the previously-vendored ActionFormer stays
tracked). Method repos, downloaded checkpoints and the AV²A venv therefore never
enter the ledger; their exact commits are pinned in the table above. Committed
from this route: `scripts/repro_campaign/` (harness + downloader) and this file.

Reproducibility caveat: the `# ADAPTED` edits described in §3 live inside
`third_party/`, which is untracked. They are all described here precisely enough
to re-apply, but if Phase C results depend on them, promote the patches to
tracked `.patch` files under `scripts/repro_campaign/patches/`.
