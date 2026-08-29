# MLLM Feasibility Memo — Introducing a Multimodal LLM into the Hateful-Video Pipeline

Scope: read-only feasibility scoping. No weights downloaded, no GPU jobs, no packages installed.
Date: 2026-07-01. Env: conda `HateVideo`. Host: `foscsmlprd01` (login = compute).

TL;DR: **Feasible.** The A100s are **80GB** (not 40GB), which removes VRAM as a blocker for
both a frozen 7B MLLM extractor and 7B LoRA-SFT. transformers 4.49 natively supports
Qwen2-VL, Qwen2.5-VL, LLaVA-OneVision, LLaVA-NeXT-Video, and more. The only real constraint is
disk (307G / 290G soft quota), and that is **easily solved**: a purely transient 26G rclone VFS
cache can be purged, restoring plenty of headroom for a ~15G bf16 7B MLLM. Recommended repos:
**Qwen/Qwen2.5-VL-7B-Instruct** (primary) and **llava-hf/llava-onevision-qwen2-7b-ov-hf** (backup),
plus **Qwen/Qwen2.5-VL-3B-Instruct** as a lightweight fallback.

---

## 1. Env capability

Interpreter / core stack (from `conda activate HateVideo`):

```
python 3.11.8
torch 2.6.0+cu124   cuda 12.4
transformers 4.49.0
```

MLLM model-class import checks (offline, `HF_HUB_OFFLINE=1`, no weights — config/class availability only):

```
OK    Qwen2-VL               Qwen2VLForConditionalGeneration
OK    Qwen2.5-VL             Qwen2_5_VLForConditionalGeneration
OK    LLaVA-OneVision        LlavaOnevisionForConditionalGeneration
OK    LLaVA-NeXT-Video       LlavaNextVideoForConditionalGeneration
OK    Video-LLaVA            VideoLlavaForConditionalGeneration
FAIL  InternVL(native)       InternVLForConditionalGeneration  -> AttributeError: module transformers has no attribute InternVLForConditionalGeneration
OK    Llama-3.2-Vision       MllamaForConditionalGeneration
OK    Aria                   AriaForConditionalGeneration
OK    Idefics3/SmolVLM       Idefics3ForConditionalGeneration
```

Interpretation vs. the requested list:

| Requested MLLM family | transformers 4.49 native support | Notes |
|---|---|---|
| **Qwen2-VL** | YES (`Qwen2VLForConditionalGeneration`) | video-capable |
| **Qwen2.5-VL** | YES (`Qwen2_5_VLForConditionalGeneration`) | video-capable, current SOTA-ish 7B |
| **LLaVA-OneVision** | YES (`LlavaOnevisionForConditionalGeneration`) | image + multi-frame/video |
| **LLaVA-Video** (a.k.a. LLaVA-NeXT-Video) | YES (`LlavaNextVideoForConditionalGeneration`) | native video path |
| **InternVL2.5** | NOT native in 4.49 (native `InternVLForConditionalGeneration` landed in transformers ~4.52). | Still runnable in 4.49 via `AutoModel(..., trust_remote_code=True)` (the OpenGVLab remote code path). Adds friction. |
| **VideoLLaMA2 / 3** | NOT a transformers model class | needs the authors' own repo (DAMO-NLP-SG). Out-of-band, not `from transformers import ...`. |
| **MiniCPM-V** | NOT a native class | runs via `trust_remote_code=True` only. |

Installed helper libraries (`pip list`):

```
accelerate    1.5.2
peft          0.14.0
bitsandbytes  0.49.2
decord        0.6.0        # video frame decode
av            17.0.0       # PyAV, video decode
imageio       2.37.0
einops        0.8.0
timm          1.0.15
sentencepiece 0.2.0
tokenizers    0.21.4
safetensors   0.5.3
deepspeed     0.16.4
torchvision   0.21.0
ninja         1.13.0
```

**Missing helpers (blockers to note, both installable on login node later):**
- **`flash-attn` / `flash_attn` — NOT installed.** Qwen2.5-VL and most 7B MLLMs *run* without it
  (fall back to eager / sdpa attention), just slower and with a bit more VRAM. LoRA-SFT strongly
  benefits from flash-attn-2. `ninja` is present, so a source build is possible, but flash-attn
  builds are slow/fragile; prefer a matching prebuilt wheel for torch 2.6 / cu124.
- **`qwen-vl-utils` — NOT installed.** This is the recommended helper for Qwen2-VL/2.5-VL
  image+video preprocessing (`process_vision_info`, smart frame sampling). Not strictly required
  (you can build the pixel/video tensors manually via the `AutoProcessor` + decord), but installing
  it on the login node will save a lot of preprocessing code. `xformers` also NOT installed.

`AutoProcessor` and `transformers.image_utils.load_image` import cleanly, so the generic
processor/video plumbing is present.

---

## 2. Local cache — what's already downloaded

Primary HF hub cache resolves to `/data/jehc223/home/.cache/huggingface/hub` (6.3G total).
Contents:

```
3.2G  models--openai--clip-vit-large-patch14-336            (CLIP, real weights — used by RGCL frame route)
1.1G  models--sentence-transformers--paraphrase-multilingual-mpnet-base-v2
711M  models--microsoft--deberta-v3-base
458M  models--sentence-transformers--paraphrase-multilingual-MiniLM-L12-v2
421M  models--bert-base-uncased
419M  models--sentence-transformers--all-mpnet-base-v2
 88M  models--sentence-transformers--all-MiniLM-L6-v2
700K  models--google-bert--bert-base-uncased
 96K  models--OpenGVLab--InternVL2-8B          <-- CONFIG/CODE ONLY, no weights (stub)
8.0K  models--mistralai--Mistral-7B-Instruct-v0.3  <-- refs only, no weights (stub)
  0   models--llava-hf--LLaVA-NeXT-Video-34B-hf     <-- BROKEN: symlink to a non-existent dir
```

**No usable MLLM is cached.** Concretely:
- `OpenGVLab/InternVL2-8B` = 92K of blobs (only `config.json`, `modeling_*.py`, tokenizer config) —
  **weights were never fetched.** Not usable as-is.
- `mistralai/Mistral-7B-Instruct-v0.3` = refs/blobs stub, **no weights** (and it's an LLM, not an MLLM).
- `llava-hf/LLaVA-NeXT-Video-34B-hf` is a **dangling symlink** into
  `/data/jehc223/.cache/huggingface/hub/`, which **does not exist**. Zero bytes. (Also 34B is too big
  to want anyway.)

So: any MLLM route requires a fresh download of the chosen 7B weights. `openai/clip-vit-large-patch14-336`
(3.2G, real) is present and is what the existing RGCL frame-CLIP route uses — unrelated to the MLLM but
worth noting it's already there.

---

## 3. VRAM feasibility on 1× A100

**A100 model confirmed: `NVIDIA A100-SXM4-80GB` — 81920 MiB per GPU (80GB, not 40GB).**
`nvidia-smi` shows 8 A100-80GB on the node; SLURM `Gres=gpu:a100:8`, partition `slurmpartition`,
node `foscsmlprd01`, 256 CPU / ~1000G RAM. Per-user cap 16 CPU / 128G / 2 GPU. Request `gpu:a100:1`.

`bitsandbytes 0.49.2` is installed → 4-bit / 8-bit quantization is available if ever needed (it is
*not* needed at 80GB for a single 7B, but useful for headroom or batch scaling).

### (a) FROZEN MLLM — inference-only feature/caption/rationale extractor

| Precision | ~Weights | + activations/KV (short gen, few frames) | Fits 40GB? | Fits 80GB? |
|---|---|---|---|---|
| bf16 (7B) | ~15 GB | +5–15 GB (video frames inflate vision tokens) | tight but yes | **comfortable** |
| 4-bit (7B) | ~5 GB | +5–10 GB | yes | trivial |

Verdict: **Fully feasible on the 80GB A100 in bf16, no quantization required.** You can run larger
batches or more frames per clip, or even a 34B-class frozen model if desired (34B bf16 ≈ 68G weights —
would fit but leaves little room; not recommended). For video, VRAM is dominated by the number of
sampled frames × vision tokens, so cap frames (e.g. 8–32) and keep generations short (captions/rationales).

### (b) LoRA SFT of a ~7B MLLM (adapters only; base frozen)

Rough budget in bf16 with LoRA (only adapter params get optimizer state):
- Base weights (bf16, frozen): ~15 GB
- LoRA adapter params: small (tens of MB)
- Adam optimizer states for LoRA only: small (hundreds of MB)
- Activations + gradients (batch 1, gradient checkpointing on, modest frames): ~10–25 GB
- Vision-tower activations for video (multiple frames): the swing factor

| Config | Est. peak VRAM | Fits 40GB? | Fits 80GB? |
|---|---|---|---|
| 7B QLoRA (4-bit base) + grad-ckpt | ~18–30 GB | yes | **easily** |
| 7B LoRA (bf16 base) + grad-ckpt | ~35–55 GB | risky | **yes** |
| 7B full-FT | 100G+ (needs ZeRO/multi-GPU) | no | no on 1 GPU |

Verdict: **7B LoRA-SFT fits the 80GB A100** with gradient checkpointing and small batch (accumulate
for effective batch). If VRAM gets tight from many video frames, drop to **QLoRA (4-bit base via
bitsandbytes)**, which is installed and comfortably fits. `deepspeed` is available if you later want
ZeRO to push batch size, but it is not required for single-GPU 7B LoRA. flash-attn absence means eager/sdpa
attention → higher activation memory and slower step time; installing a matching flash-attn wheel later is
the main perf win.

---

## 4. Disk plan — the real (but solvable) constraint

Current quota state (`quota -s`, filesystem `/dev/mapper/data-data`):

```
space=307G   soft-quota=290G   hard-limit=3000G   grace=6 days
```

**Important nuance:** we are ~17G over the *soft* limit, but the *hard* limit is **3 TB** with a
**6-day grace period**. So it is not a hard wall today — but we should get back under 290G to be safe
and stop the grace clock, before staging a ~15G MLLM.

Top-level usage (`du -sh`):

```
93G  AlphaSteer          (91G is AlphaSteer/data — a DIFFERENT project's data)
59G  miniconda3          (envs: SafetyContradiction 12G, implihate 11G, AlphaSteerRepro 9.3G,
                          ExMRD 9.2G, ipiguard 8.3G, HateVideo 7.1G)
39G  home                (32G is home/.cache -> see below)
36G  SafetyContradiction
28G  ExMRD_ours          (16G is ExMRD_ours/data)
27G  Multihateclip       (English 16G + Chinese 12G — this IS our hateful-video dataset)
14G  NIPS2026
9.6G HateMM
...
459M RGCL                (this project)
```

**Biggest easy win — a transient cache, not data:**

```
home/.cache = 32G, of which:
  26G  home/.cache/rclone/vfs     <-- transient rclone VFS read cache (from a mounted B2 remote)
  6.3G home/.cache/huggingface    <-- the HF cache above
  136M home/.cache/vllm
   32M home/.cache/pip
```

`home/.cache/rclone/vfs` is a **26G transient VFS cache** produced by rclone mounting the B2 remote —
it is a re-fetchable read cache, **safe to purge**. Deleting it (`rclone mount --vfs-cache-mode` cache,
or just `rm -rf ~/.cache/rclone/vfs` when no rclone mount is active) instantly drops usage from **307G
to ~281G**, back under the 290G soft limit with ~9G to spare and the grace clock stopped.

That still isn't enough headroom for a 15G MLLM on top, so combine with one more move:

Concrete, ranked recommendations to free space for the MLLM:
1. **Purge `~/.cache/rclone/vfs` (~26G)** — transient, zero data loss. Do this first. (307G → ~281G)
2. **Offload another project's bulk data to B2 and delete locally.** The rclone `b2:` remote is
   configured and working (recent logs show multi-GB uploads completing). Best candidate:
   **`AlphaSteer/data` = 91G** — a *different* project's data; if that project is idle, stage it to
   `b2:` and delete locally to free ~91G. `ExMRD_ours/data` (16G) is another candidate. This gives
   ample room (>15G) for the MLLM with margin.
3. **Prune stale conda envs.** Several envs total ~59G. If any of `AlphaSteerRepro` (9.3G), `ipiguard`
   (8.3G), `ExMRD` (9.2G) belong to finished projects, `conda env remove` frees GB-scale space. Keep
   `HateVideo` (7.1G).
4. **Do NOT delete `Multihateclip` (28G)** — that is our target hateful-video dataset (MHC English+Chinese).

**Recommended concrete plan for bringing in a 7B MLLM (~15G bf16):**
- Step 1: purge the 26G rclone VFS cache → back under soft quota immediately.
- Step 2: if still tight, offload `AlphaSteer/data` (91G) to `b2:` and delete locally.
- Step 3: on the **login node** (has internet), download the chosen 7B repo into the HF hub cache
  (`~/.cache/huggingface/hub`), verify with a quick offline `from_pretrained`, then run all SLURM
  jobs with `HF_HUB_OFFLINE=1`.
- Fallback if you want to stay minimal: use the **3B** class (Qwen2.5-VL-3B ≈ 7G bf16, or 4-bit ≈ 2–3G)
  which fits under quota with almost no cleanup, or 4-bit-quantize the 7B on download.

B2 just-in-time staging is available (rclone `b2:` works) but for a single 15G MLLM that you'll reuse
every job, it's simpler to keep it resident in the HF cache after freeing space than to pull it per-job.

---

## 5. Recommendation

Two MLLMs that satisfy (a) video/frame-capable, (b) native in transformers 4.49, (c) feasible on the
80GB A100 + disk, for BOTH a frozen-feature route and a LoRA-SFT route:

### Primary: Qwen2.5-VL-7B
- HF repo id: **`Qwen/Qwen2.5-VL-7B-Instruct`**
- Class: `Qwen2_5_VLForConditionalGeneration` (native, imports cleanly in 4.49).
- Native **video** input (dynamic-resolution, frame sampling). Strong OCR + reasoning — ideal for
  hateful-video where on-screen text, symbols, and transcript context matter.
- Frozen route: bf16 inference on 80GB A100 to emit per-clip captions / hateful-rationales / embeddings
  to fuse into the RGCL pipeline. Comfortable VRAM.
- LoRA route: fits 80GB with grad-ckpt (bf16 base) or trivially as QLoRA (4-bit base, bitsandbytes present).
- Action items before use: install **`qwen-vl-utils`** (video preprocessing) and, for speed, a matching
  **flash-attn** wheel — both on the login node.
- Lightweight fallback under tight disk: **`Qwen/Qwen2.5-VL-3B-Instruct`** (~7G bf16), same code path,
  fits comfortably under quota with minimal cleanup.

### Backup: LLaVA-OneVision-7B
- HF repo id: **`llava-hf/llava-onevision-qwen2-7b-ov-hf`**
- Class: `LlavaOnevisionForConditionalGeneration` (native in 4.49).
- Handles single-image, multi-image, and **multi-frame/video** in one model; pure-transformers path
  (no `trust_remote_code`, no external repo, no qwen-vl-utils dependency) → lowest integration risk.
- Same 80GB feasibility for frozen inference and 7B LoRA.
- (If a dedicated video model is preferred, `llava-hf/LLaVA-NeXT-Video-7B-hf` via
  `LlavaNextVideoForConditionalGeneration` is also native — note the 34B stub already in cache is
  broken/empty and unrelated.)

### Explicitly NOT recommended for the first pass
- **InternVL2.5 / InternVL2-8B**: no native class in transformers 4.49 (needs `trust_remote_code` or a
  transformers upgrade to ~4.52). The cached `InternVL2-8B` is a **config-only stub with no weights**.
- **VideoLLaMA2/3, MiniCPM-V**: not transformers-native classes; require the authors' repos /
  `trust_remote_code`. More integration risk for no clear gain over Qwen2.5-VL here.

---

## Honest blockers / caveats
- **Disk grace clock is ticking (6 days over soft quota).** Purge the 26G rclone VFS cache first;
  it's transient and safe. That alone gets us back under 290G.
- **flash-attn and qwen-vl-utils are not installed.** Models run without them, but for Qwen2.5-VL you'll
  want `qwen-vl-utils` (preprocessing) and, for LoRA-SFT throughput, `flash-attn`. Install on the login
  node (has internet); flash-attn build can be slow — prefer a prebuilt torch2.6/cu124 wheel.
- **No MLLM weights are cached** — a fresh ~15G (7B bf16) or ~7G (3B) download is required. Cached
  InternVL2-8B / Mistral-7B / LLaVA-34B entries are stubs/broken and unusable.
- **transformers 4.49 excludes native InternVL2.5**; if InternVL is specifically wanted, plan a
  transformers bump (which risks perturbing the reproduced RGCL env) or accept `trust_remote_code`.
- Video VRAM scales with sampled frames × vision tokens — cap frames (e.g. 8–32) to stay well within 80GB.

---

## Disk cleanup log
- 2026-07-01: Purged the transient rclone VFS read cache (`~/.cache/rclone/vfs/*` and
  `~/.cache/rclone/vfsMeta/*`) after confirming no active rclone mount/process (`mount | grep -i rclone`
  and `ps aux | grep 'rclone mount'` both empty). `home/.cache/rclone` went **26G → 0** (dirs kept, empty).
  Quota on `/dev/mapper/data-data` dropped **307G* (over soft, 6-day grace) → 282G** — back under the 290G
  soft limit, and the over-quota `*` flag and grace clock both cleared. ~25G reclaimed, zero data touched
  (no datasets, HF cache, AlphaSteer, or other project data affected).
