# MOLMO2 — forensic recon (2026-07-27)

**Status:** zero-GPU recon. No GPU job submitted, no feature cache written, no prereg, no
promotion. Bars in §7 are **pre-declared before any extraction** and are binding on the probe.

**Lane scope (from the gate):** Molmo2-8B frozen-encoder swap, **HateMM only** (F44 cap — on
MHC-EN encoder identity rotates rather than improves; ZH is text-dominant). Probe ends at the
$0 CPU-head verdict; any formal 3-seed GPU cell goes through prereg ceremony in the main loop.

**Honest prior.** F65 killed vision-side LoRA adaptation (image MOVED, zero conversion = 8th
law-I datum). ERRPAT-HateMM stream forensics show the HateMM image stream contributes ~nothing
net (text-only ≥ fused). The Molmo2 bet is specifically that a **video-native tower** changes the
*fused* geometry, not merely the image stream. Expect a kill; measure cleanly.

---

## 1. MODEL IDENTITY — VERIFIED, IT IS THE SigLIP2 GENERATION

The gate required: stop if only the earlier 2024 Molmo (non-SigLIP2) exists. It does not — the
2025 Molmo2 family is real and distinct.

| field | value |
|---|---|
| HF repo id | **`allenai/Molmo2-8B`** |
| license | **apache-2.0** |
| created | 2025-12-14 |
| downloads (at recon) | 252,343 |
| LLM backbone | **`Qwen/Qwen3-8B`** |
| vision backbone | **`google/siglip-so400m-patch14-384`** (SigLIP 2) |
| `model_type` | `molmo2`, `Molmo2ForConditionalGeneration`, `custom_code` (trust_remote_code) |
| checkpoint dtype | **float32** on the Hub |
| repo size | **34.66 GB** (8 shards) |
| text hidden size | **4096** (Qwen2.5-VL-7B was 3584) |
| vision tower | 27 layers, hidden 1152, patch 14, 378x378 native |

Confirmed from the model card: *"Molmo2-8B is based on Qwen3-8B and uses SigLIP 2 as vision
backbone. It outperforms others in the class of open weight and data models on short videos,
counting, and captioning."* This matches the F81 litsweep description exactly (SigLIP2 tower,
short-video strength). **No substitution was made.**

Disambiguation — these are NOT our model and were not used: `allenai/Molmo-7B-{D,O}-0924` and
`allenai/Molmo-72B-0924` (the 2024 generation, OpenAI-CLIP tower), `allenai/MolmoAct2*`
(robotics/action), `allenai/MolmoPoint-*`, `allenai/MolmoWeb-*`, `allenai/Molmo2-ER`.
`allenai/Molmo2-VideoPoint-4B` and `Molmo2-O-7B` exist as siblings but the gate named 8B.

---

## 2. ENVIRONMENT — HateVideoVLM (4.57.6), HateVideo IS TOO OLD

| env | transformers | verdict |
|---|---|---|
| `HateVideo` (project default) | **4.49.0** | **Cannot run Molmo2.** Model card and `config.json` both require **4.57.1**. |
| `HateVideoVLM` | **4.57.6** | **USE THIS.** Same pattern as P10-c. |

`HateVideo` is **not** mutated. Verified present in `HateVideoVLM`: torch 2.6.0+cu124,
einops 0.8.0, torchvision 0.21.0, decord 0.6.0, av 17.0.0, numpy 1.26.4, accelerate 1.14.0,
safetensors 0.5.3, huggingface_hub 0.36.2.

`molmo_utils` (named in the model card's quick-start) is **not needed** — it is imported by none
of the four remote-code modules; it serves the pointing/demo utilities only. `decord2` likewise
is not needed because we pass pre-decoded frames (§4).

**Version landmine, found and solved.** The remote code reads `metadata.timestamps`, but in
transformers 4.57.6 `VideoMetadata` has **no `timestamps` field** — it is a *derived property*
`[frame_idx / fps for frame_idx in frames_indices]`. Passing `timestamps=` raises
`TypeError`. Correct call passes **`fps` + `frames_indices`** and lets the property derive true
wall-clock times. This is a benefit, not a workaround: our sampler already produces real source
frame indices, so the injected timestamps are genuine.

---

## 3. DISK — the binding constraint, resolved WITHOUT deleting anything

At recon: **270 G used / 290 G soft quota** (hard 3000 G, 6-day grace) = **~20 G headroom**.
The gate's estimate of 18-20 GB assumed a bf16 checkpoint; upstream ships **fp32 = 34.66 GB**,
which would have blown the soft quota outright.

**Resolution — shard-streamed fp32 to bf16** (`scripts/analysis/molmo2_download_bf16.py`):
download one shard, cast floating tensors to bf16, write, delete the fp32 source, repeat.

| | GB |
|---|---|
| naive fp32 snapshot | 34.66 (**over quota**) |
| streamed peak on-disk | **~18.6** |
| final footprint | **~17.4** |

**bf16 is dtype-matched, not a concession:** the deployed Qwen recipe already loads the encoder
with `torch_dtype=torch.bfloat16`, so the swap compares like with like. Non-floating buffers stay
bit-identical. `float32_attention: true` in the vit/adapter configs still upcasts attention at
runtime. A `CONVERSION_NOTE.md` is written beside the weights.

**No deletion was requested or performed.** Model location `/data/jehc223/models/Molmo2-8B-bf16`
(a converted copy, so deliberately not placed in the pristine HF cache alongside
`models--Qwen--Qwen2.5-VL-7B-Instruct`).

---

## 4. EXTRACTOR MAPPING — Qwen recipe onto Molmo2

Reference: `src/utils/generate_VideoMLLM_embedding_HF.py` (Config MLLM-A). The cache contract
(`ids` as ONE sublist, `img_feats`, `text_feats`, `labels`) is unchanged, so
`src/data_loader/dataset.py:load_feats_MHC` consumes Molmo2 features with **zero code change**.
Head dims are inferred at runtime (`run_rac.py:1256-1257`), so 4096-d needs no config edit.

**Validated on CPU with no weights** (processor-only, 16 MB of config/tokenizer files):

```
chat template -> '<|video|><|im_start|>user\n{INSTRUCTION}<|im_end|>\n<|im_start|>assistant\n'
seq_len 732 | image_patch(151938) count 648 = 8 frames x 81 tokens
<|im_start|> id 151644 at positions [708, 729]
prefix span  [0:729]   -> 729 tokens (video + instruction)   == img_feats
response span [729:732] -> '<|im_start|>assistant\n'          == text_feats
```

Vision tokens merge **in place**: `modeling_molmo2.py:1447`
`x.view(-1, x.shape[-1])[is_image_patch] += image_features`, with
`assert is_image_patch.sum() == len(image_features)`. Sequence length is **not** expanded, so the
Qwen extractor's `hidden.shape[0] == input_ids.numel()` invariant and its span indexing both hold
verbatim.

### Deviation ledger

| aspect | Qwen2.5-VL recipe | Molmo2 | deviation |
|---|---|---|---|
| frame sampler | 8 uniform, decord -> PyAV fallback | **byte-identical sampler reused** | **none** |
| frame delivery | `videos=[frames]` (PIL list) | `videos=<(8,H,W,3) uint8>` + `video_metadata=[{fps, frames_indices, ...}]`, `do_sample_frames=False` | API shape only; frames pass through unresampled |
| instructions | `IMG_INSTRUCTION`, `TEXT_INSTRUCTION`, `Title: `, `Transcript: `, `(none)` | **byte-identical strings** | **none** |
| img span | mean over `[0 : last <\|im_start\|>)` | identical | **none** |
| text span | mean over `[last <\|im_start\|> : end]` | identical | **none** |
| pooling | mean -> L2-normalise | identical | **none** |
| dtype | bf16 | bf16 (cast from fp32 ckpt, §3) | **documented** |
| hidden dim | 3584 | 4096 | none (inferred) |
| **resolution** | `max_pixels=360*420`, dynamic ViT grid, ~1540 vision tokens | **fixed 378x378/frame, 3x3 pool, 648 tokens** | **FORCED — architecture** |
| **media position** | video tokens inside the user turn | `<\|video\|>` emitted **before** `<\|im_start\|>user` | **FORCED — chat template** |
| **timestamps** | none injected | per-frame wall-clock timestamps injected into the video string | **FORCED — video-native format** |

The three FORCED rows are exactly what "video-native tower" means and cannot be removed without
defeating the purpose of the swap. They are the reason this is an encoder-swap probe and not a
controlled single-variable ablation; the probe measures the encoder as shipped.

---

## 5. COST ESTIMATE

HateMM = 744 train + 107 val + 215 test = **1066 videos**, 2 forwards each = 2132 forwards.
Reference: Qwen2.5-VL-7B HateMM extraction (job 13240) = **26 min** on 1 A100. Molmo2-8B is
comparable in parameters and carries **fewer** vision tokens (648 vs ~1540), offset by
`float32_attention`. Budget **~30-60 min, 1 A100, 1 GPU-h ceiling.**

CPU-head probe: **52 s/seed on 8 CPUs**, 3 seeds per arm, **$0 GPU** (ERRPAT §8 infra finding).

**Measured (job 13648, added after the fact; the estimate above stands).** Extraction proper ran
at **60 videos/min** (clean 60 s window; cross-check 360 train items in the 351 s after the
"extraction START" marker, less ~50 s model load + smoke). 1066 videos therefore cost **~18 min
of GPU**, inside the 1 GPU-h ceiling. `float32_attention` is real but not materially costly.

A rate alarm was raised mid-run from 34 min wall / 180 items and projected ~3 GPU-h. That
projection double-counted a **30 min 45 s `disk_guard` phase** (08:36:08 -> 09:06:53) that runs
before extraction: `/data` was already at 270 G against the guard's 250 G threshold *before* this
lane's download, so the guard would have fired on whichever job submitted first that day. It
pruned only under `logging/` (B2-verified before each delete; `data/CLIP_Embedding` is
unreachable by the only destructive stage) and exhausted its candidate pool rather than reaching
target. No deviation to record against §5.

---

## 6. PROBE DESIGN

Arms, all trained with the byte-identical `run_rac.py` command from
`scripts/slurm/enc3seed_lora_curric.sbatch`, changing only `--device cpu --save_embed True
--model <tag> --group_name/--output_path`, 3 seeds (0/1/2), HateMM:

| arm | feature tag | role |
|---|---|---|
| **A. Molmo2 (treatment)** | `Molmo2-8B_HF` | the swap |
| **B. LoRA-curric (strongest floor)** | `Qwen2.5-VL-7B-Instruct-LoRA-curric_HF` | promotion bar reference |
| **C. frozen Qwen (identity control)** | `Qwen2.5-VL-7B-Instruct_HF` | frozen-vs-frozen mechanism read |

**Same-path discipline (F87, ERRPAT §8):** CPU-trained heads are NOT bit-exact to the CUDA floor
(-0.0031 final-epoch acc). Arm A is compared **only** against CPU-trained B and C, never against
the banked GPU floor. B and C are **re-run in this session** rather than quoted, so all three arms
share one machine state. Banked CPU-proxy reference for B (ERRPAT 0.2):
**val-sel 0.8775 / 0.8715, final 0.8760 / 0.8699** — re-derivation should reproduce it.

Arm A is **frozen**; arm B is **adapted**. That asymmetry is deliberate and is why both B and C
are carried: B answers "does it beat what we deploy", C answers "is the encoder itself better".

**Geometry diagnostics** (patterns from `scripts/analysis/mechfix_diag.py`), reported for all
three arms regardless of the accuracy verdict:
- top-1 cosine saturation (cone collapse)
- length-organisation rho(query volume, median top-20 bank-neighbour volume)
- raw-space (pre-head) kNN vote accuracy
- key-covariance variance share of the leading direction

---

## 7. PRE-DECLARED BARS (binding, fixed before extraction)

**PROMOTE to formal prereg ceremony** iff, on 3-seed CPU-paired HateMM:

1. **Delta >= +0.0200 on acc AND +0.0200 on mF1** vs the **strongest same-path CPU floor**
   (arm B), on the **same protocol**, and
2. **sign-consistent 3/3 seeds** (standing project discipline), and
3. the win survives on **both** protocols (val-selected and final-epoch), or if it splits, the
   split is reported as MARGINAL and does **not** auto-promote.

**KILL / PARK** otherwise. A delta below +0.0200 on either metric is a kill regardless of how the
geometry reads.

**The geometry read does not gate promotion.** It is recorded as a mechanism datum either way:
if accuracy is flat but the geometry genuinely moved (cone saturation down, length-organisation
broken), that is a *negative-with-mechanism* result — the representation changed and the box still
did not open — which is a stronger closure of the encoder-swap entry point than a flat number
alone. If accuracy is flat AND geometry is unchanged, the swap simply did not bite.

**No test-set quantity selects anything.** Val-selection uses val only (warmup>=5); the
final-epoch protocol selects nothing. Test numbers are single-draw reads.

---

## 8. RISKS

1. **Disk (~1.4 G margin at peak).** Other live lanes (CLAP, MNTP) write concurrently. Crossing
   290 G starts a 6-day grace, not a hard failure, so a small overshoot is recoverable. Not
   deleting anything; will report if it binds.
2. **Frozen-vs-adapted asymmetry** — mitigated by carrying arm C, not by re-tuning.
3. **fp32 -> bf16 cast** — matched to the deployed compute dtype, but it is a real (documented)
   difference from running the checkpoint as shipped.
4. **transformers 4.57.6 vs pinned 4.57.1** — processor path verified working; the model forward
   is verified at smoke time before the full run.
5. **Prompt-format drift** is unavoidable (§4 FORCED rows) and is the honest content of an
   encoder-swap comparison.
