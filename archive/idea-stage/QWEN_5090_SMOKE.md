# Qwen2.5-VL-7B on the RTX 5090 — trainability smoke test

Date: 2026-08-10/11. Machine: single RTX 5090 32 GB (Blackwell, sm_120), conda env `HateVideo`.
Question: can this box train Qwen2.5-VL-7B, and what does one LoRA-SFT seed cost?

> **Read this first — the card is not ours alone.** Throughout this whole test another user
> (`lhu861`, PID 1563224, `exp_a_interp.py`) has held **20.3 GiB and ~97 % SM utilisation**
> continuously; at the last check it had been running **10 h 07 m**. That leaves ~10.4 GiB usable.
> Every arm that needs more than ~10 GiB **could not be run at all**, and every timing below is
> measured while sharing SMs with that job, so all seconds/step are **pessimistic upper bounds**.
> This is itself the first answer: *the memory is there in principle, but on this box you do not
> currently get the card.*

---

## 1. Environment health check

| item | value | verdict |
|---|---|---|
| GPU | RTX 5090, 31.36 GiB usable, 170 SMs, driver 595.71.05 (CUDA 13.2) | |
| `torch.cuda.get_device_capability()` | **(12, 0)** = sm_120 | |
| torch | **2.7.1+cu128**, arch list contains `sm_120` + `compute_120` | OK — native kernels, no PTX JIT |
| cuDNN | 9.7.1 | OK |
| `is_bf16_supported()` | True | OK |
| transformers | 4.49.0 (Qwen2.5-VL landed in 4.49) | OK |
| peft | 0.14.0 | OK |
| **bitsandbytes** | **0.49.2 — works on sm_120** | `Linear4bit` NF4 fwd+bwd **OK**; `PagedAdamW8bit.step()` **OK**; full 4-bit 7B load + train step **OK**. QLoRA is viable. |
| accelerate / trl / datasets | 1.5.2 / 0.16.1 / 3.2.0 | OK |
| **flash-attn** | **not installed, and not needed** | PyPI ships sdist only (no sm_120 wheel; source build ≈1–2 h). Unnecessary: torch's own SDPA **FLASH / MEM_EFFICIENT / MATH / CUDNN backends all pass fwd+bwd on sm_120** (verified), so `attn_implementation="sdpa"` already dispatches the fused flash kernel. All runs below use `sdpa`. |
| LLaMA-Factory | `RA-HMD/LLAMA-FACTORY/` and `…-Ver202512/` are **empty directories** | The historical SFT harness did **not** survive the migration to this box. Re-running the old pipeline verbatim requires re-cloning LF. Smoke test therefore drives transformers+peft directly. |
| bf16 matmul throughput | 89.1 TFLOPS (8192³, **contended**) | free-card figure will be ~2× this |

Weights: `Qwen/Qwen2.5-VL-7B-Instruct` **download complete** — 16 files / 16 GB in the default HF
cache, snapshot `cc594898137f460bfe9f0759e9844b3ce807cfb5`.

Parameter breakdown (read from the safetensors index):

| block | params | bf16 bytes |
|---|---|---|
| vision tower (`visual.*`) | 0.677 B | 1.35 GB |
| 28 LM decoder layers | 6.526 B | 13.05 GB |
| `embed_tokens` + `lm_head` | 1.090 B | 2.18 GB |
| **total** | **8.292 B** | **16.6 GB (15.5 GiB)** |

Config: hidden 3584, 28 layers, 28 heads / 4 KV heads (GQA), intermediate 18944, **vocab 152 064**.

---

## 2. The operating point being replicated

The historical LoRA-SFT config is pinned by `refine-logs/LORA_HATEMM_PREREG.md` and
`refine-logs/FRAME_BUDGET_FORENSIC_RECON.md` (the yaml itself is gone with LF):

`lora_rank 16`, `lora_alpha 32`, dropout 0.0, targets `q,k,v,o,gate,up,down`,
**`freeze_vision_tower: true`**, `lr 1e-4`, 3 epochs, `per_device_train_batch_size 1`,
`gradient_accumulation_steps 8` (eff bs 8), `bf16`, `gradient_checkpointing: true`,
8-frame multi-image ShareGPT, **`image_max_pixels 262144`**, **`cutoff_len 4096`**.

Reproduced exactly against the real cached frames (`data/lora_frames/HateMM/`, 1065 videos × 8 JPGs,
854×480 — the same JPGs the historical run used):

| frames | vision tokens | **seq_len** | cutoff needed |
|---|---|---|---|
| 2 | 624 | 904 | 2048 |
| 4 | 1248 | 1532 | 3072 |
| **8 (historical)** | **2496** | **2788** | 4096 |
| 16 | 4808 | 5116 | 8192 |

312 tokens/frame (26×48 patch grid ÷ 2×2 merge). The 8-frame figure matches the recon doc's
"~2496–2520" exactly, so this is the historical operating point, not an invented one.
LoRA trainable params = **40.37 M**.

CPU-side data prep (JPEG decode + resize + processor) = **0.064 s/sample, single worker** —
negligible, and fully hidden by dataloader workers. Data pipeline is *not* the bottleneck.

---

## 3. Smoke matrix (real fwd + bwd + optimizer step)

Batch size 1, gradient checkpointing on, `sdpa`, real HateMM frames, 5 steps
(step 0 reported separately — it carries warmup). Memory = `torch.cuda.max_memory_allocated`.

| # | arm | frames | seq_len | weights (GiB) | **peak (GiB)** | **s/step (steady)** | result |
|---|---|---|---|---|---|---|---|
| 1 | **QLoRA-lean** NF4 + LoRA r16 | 2 | 938 | 5.49 | **8.02** | **3.21** (step0 3.85) | **PASS** — loss 14.79→10.54 over 5 steps |
| 2 | **QLoRA-lean** NF4 + LoRA r16 | 3 | 1252 | 5.49 | **8.80** | **4.37** (step0 5.16) | **PASS** — loss 15.95→11.94 |
| 3 | QLoRA-lean NF4 + LoRA r16 | 4 | 1532 | 5.49 | **9.55** (step 0) | ~5.7 (step0 6.7, ÷1.19 warmup) | step 0 completed, **OOM on step 1** (382 MiB short), reproduced twice |
| 4 | QLoRA (standard `prepare_model_for_kbit_training`) | 2 | 904 | 7.70 | — | — | **OOM** (needed >10.4 GiB free) |
| 5 | QLoRA (standard) | 4 | 1532 | 7.70 | — | — | **OOM** |
| 6 | QLoRA (standard) | 8 | 2788 | 7.70 | — | — | **OOM** — 8.67 GiB resident, then a single **5.94 GiB** allocation failed |
| 7 | QLoRA (standard) | 16 | 5116 | 7.70 | — | — | **OOM** |
| 8 | **bf16 LoRA** (no quant) | 8 | 2788 | 15.49 | — | — | **NOT RUNNABLE** — 15.5 GiB of weights alone exceeds the 10.4 GiB left by the other user |
| 9 | bf16 LoRA, 4 of 28 layers (truncated probe) | 8 | 2788 | 5.40 | — | — | **OOM** on the 5.94 GiB logits allocation |

Rows 1–3 are full 28-layer models on real HateMM frames — three genuine points on the scaling curve.

"QLoRA-lean" = identical to standard QLoRA except it skips `prepare_model_for_kbit_training`'s
**fp32 upcast of `embed_tokens`/`lm_head`/norms**, which costs a flat **+2.21 GiB** (5.49 → 7.70 GiB
of resident weights). That upcast is the single easiest saving on this box.

### 3.1 The dominant allocation is the loss head, not the transformer

Rows 6 and 9 both died on the *same* **5.94 GiB single allocation**, independent of model depth.
Cause, confirmed by reading `transformers/models/qwen2_5_vl/modeling_qwen2_5_vl.py` (4.49):

```
logits = self.lm_head(hidden_states)      # bf16   T × 152064 × 2
logits = logits.float()                   # fp32   T × 152064 × 4
shift_logits = logits[..., :-1, :].contiguous()   # fp32 copy, another × 4
loss = CrossEntropyLoss()(shift_logits.view(-1, 152064), shift_labels)  # log_softmax, another × 4
```

There is **no `logits_to_keep` / chunked-CE path** in this version — full-vocab fp32 logits are
always materialised. Cost ≈ `seq_len × 152064 × 14 bytes`:

| frames | seq_len | logits+CE spike |
|---|---|---|
| 2 | 904 | 1.79 GiB |
| 4 | 1532 | 3.04 GiB |
| **8** | **2788** | **5.53 GiB** |
| 16 | 5116 | **10.14 GiB** |

At 16 frames the loss head alone wants 10 GiB. LLaMA-Factory uses this same modeling file, so the
historical A100 run paid the same tax — but it had 80 GB to pay it with.

### 3.2 Memory model fitted to the three measured points

Subtracting the known 5.49 GiB of resident weights from rows 1–3 gives non-weight memory
2.53 / 3.31 / 4.06 GiB at seq_len 938 / 1252 / 1532. Least-squares fit:

> **non-weight peak (GiB) ≈ 0.11 + 0.002574 × seq_len**

Of that slope, `152064 × 14 bytes/token = 0.001982 GiB/token` is the fp32 logits+CE spike and the
remaining **0.00059 GiB/token (≈0.59 GiB per 1000 tokens)** is checkpointed activations + LoRA
optimizer state. The fit is not an assumption: it **predicted row 3's peak as 9.50 GiB against a
measured 9.55 GiB** (0.5 % error) before that row was run. Extrapolating to the real operating points:

| arm | frames | weights | non-weight | **predicted peak** | fits a *free* 31.36 GiB card? |
|---|---|---|---|---|---|
| QLoRA-lean | 8 | 5.49 | 7.28 | **≈ 12.8 GiB** | yes, huge headroom |
| QLoRA standard | 8 | 7.70 | 7.28 | **≈ 15.0 GiB** | yes |
| QLoRA standard | 16 | 7.70 | 13.28 | **≈ 21.0 GiB** | yes |
| **bf16 LoRA** | **8** | **15.49** | 7.50 | **≈ 23.0 GiB** | **yes, ~8 GiB headroom** |
| bf16 LoRA | 16 | 15.49 | 13.50 | **≈ 29.0 GiB** | **marginal — expect OOM once fragmentation is counted** |
| **full fine-tune** (8.29 B, bf16 + AdamW) | 8 | 15.5 weights + 15.5 grads + 62 optimizer | 7.3 | **≈ 100 GiB** | **no — >3× the card** |

Full fine-tuning does not become feasible with any 8-bit-optimizer trick either: bf16 weights +
bf16 grads + 8-bit Adam states is still ≈ 47 GiB. **Full fine-tune is out on a 32 GB card, full stop.**

Both terms grow linearly in seq_len, so going 8 → 16 frames costs **+6.0 GiB** — and that is exactly
what pushes bf16 LoRA over the edge at 16 frames while leaving QLoRA comfortable.

---

## 4. Hours per seed (HateMM, 744 train samples)

Step count is fixed by the frozen config: `per_device_train_batch_size 1`, `grad_accum 8`, 3 epochs.

* optimizer steps = ⌈744/8⌉ × 3 = **279**
* **micro-steps (= real fwd+bwd) = 744 × 3 = 2232**

**A100 anchor (real, not estimated):** `refine-logs/LORA_HATEMM_VERDICT_REVIEW.md` records the
completed HateMM LoRA-SFT with `train_runtime` **10 254.7 s = 2.85 h**, eval_loss 0.1084, epoch 2.97.
That is **4.59 s per micro-step** on one A100 at 8 frames.

**RTX 5090 timings**, all measured **while contending with the other user's 97 %-util job**:
3.21 s @ seq 938, 4.37 s @ seq 1252, ~5.7 s @ seq 1532. Least-squares:

> **s/micro-step (contended) ≈ −0.70 + 0.004116 × seq_len**

At the historical 8-frame point (seq 2788) that gives **≈ 10.8 s/micro-step contended**, i.e.
2232 × 10.8 = **≈ 6.7 h per seed on the card as it is right now**.

For a *free* card, the contention discount is bounded by the matmul measurement: 89.1 TFLOPS
contended vs ~209 TFLOPS spec bf16 dense ⇒ 1.5–2.3× recovery. QLoRA also carries 4-bit dequant
overhead that bf16 LoRA does not (typically 1.2–1.4×):

| scenario | s/micro-step | **hours / seed (2232 micro-steps)** |
|---|---|---|
| QLoRA, 8 frames, **contended** (measured conditions) | ≈ 10.8 | **≈ 6.7 h** |
| QLoRA, 8 frames, free card | ≈ 4.7–7.2 | **≈ 2.9–4.5 h** |
| **bf16 LoRA, 8 frames, free card** | ≈ 3.4–6.0 | **≈ 2.1–3.7 h** |
| *A100, same config — **measured**, `train_runtime` 10 254.7 s* | *4.59* | ***2.85 h*** |
| QLoRA, 16 frames, contended | ≈ 20.4 | ≈ 12.6 h |

So a free 5090 lands **roughly at A100 parity**: **~2–4 h per HateMM seed**, 3 seeds ≈ **7–12 h**.

⚠️ Confidence: **memory numbers high** — three measured points, model validated to 0.5 % on a
held-out row, mechanism understood. **Hour numbers medium** — the seconds/step are measured but
only up to seq 1532, then extrapolated 1.8× to seq 2788, and the free-card discount is inferred
rather than measured, because the card never freed up during this test (§0). The 8- and 16-frame
arms remain queued and will run automatically if a window opens
(`idea-stage/qwen_smoke_tmp/orch3.sh`, logging to `logging/runs/qwen_smoke/orch3.log`).

---

## 5. Verdict (one line)

**Full fine-tuning is impossible (~100 GiB needed vs 32 GB); bf16 LoRA at the historical 8-frame
config fits comfortably (~23 GiB peak) and QLoRA fits easily (~15 GiB) — so yes, this box can train
Qwen2.5-VL-7B at roughly A100 parity (~2–4 h per HateMM seed vs the A100's measured 2.85 h) — but
only once the card is actually free, which it was not at any point during this test (~6.7 h/seed
under today's contention).**

Corollaries: 16 frames is a QLoRA-only regime (bf16 LoRA at 16 frames is ~29 GiB = marginal);
and nothing here needs flash-attn.

---

## 6. Gotcha list

1. **The card is shared and currently occupied.** `lhu861`'s `exp_a_interp.py` has held 20.3 GiB /
   97 % SM for 10 h+. Anything needing >10 GiB simply cannot run. Check `nvidia-smi` before
   promising a schedule; the CLAUDE.md assumption of a private single-GPU box does not hold today.
2. **flash-attn is not installed and should stay that way.** No sm_120 wheel exists on PyPI (sdist
   only → 1–2 h source build, uncertain outcome). torch 2.7.1+cu128's SDPA already provides a
   working FLASH backend on sm_120 — use `attn_implementation="sdpa"`.
3. **The fp32 logits + CE spike is the real memory wall**, not the transformer. `seq_len × 152064 ×
   14 bytes` = 5.5 GiB at 8 frames, 10.1 GiB at 16. transformers 4.49's Qwen2.5-VL has no
   `logits_to_keep`/chunked-CE escape hatch. If memory ever gets tight, fixing this (chunked CE, or
   computing `lm_head` only on labelled positions — legitimate, since SFT masks the prompt) buys
   more than any other single change.
4. **`prepare_model_for_kbit_training` silently costs +2.21 GiB** by upcasting
   `embed_tokens`/`lm_head`/norms to fp32 (5.49 → 7.70 GiB resident). Worth skipping or scoping if
   memory is tight — but note the smoke's "lean" rows skipped it, so its numerical-stability
   benefit for a *real* run was not validated here.
5. **bitsandbytes 0.49.2 works on sm_120** — 4-bit NF4 fwd/bwd and PagedAdamW8bit both verified.
   QLoRA is not blocked on Blackwell.
6. **The LLaMA-Factory harness is gone.** `RA-HMD/LLAMA-FACTORY*/` are empty dirs; the historical
   `hatemm_qwen25vl_lora_sft.yaml` does not exist on this box. Reproducing the old run means
   re-cloning LF (and re-applying the `trl` version-check shims), or porting to plain peft+trl.
   The frame cache (`data/lora_frames/`) and the split CSVs *did* survive.
7. **`scripts/slurm/lora_sft.sbatch` is dead as written** — it hardcodes `/data/jehc223/RGCL`,
   which does not exist here, and there is no SLURM on this box.
8. **16 frames needs `cutoff_len 8192`**, not 4096 (4808 vision tokens alone overflow 4096). Any
   16-frame comparison is therefore ≥2 changed variables vs the 8-frame baseline — the same point
   `FRAME_BUDGET_FORENSIC_RECON.md` already made.
9. Qwen2.5-VL's vision tower emits a "None of the inputs have requires_grad" checkpointing warning
   under `freeze_vision_tower: true`. Benign — LoRA params still receive gradients (loss fell
   14.79 → 10.54 across 5 steps).

---

## 7. Reproduction

Driver script: `idea-stage/qwen_smoke_tmp/smoke.py`
(`--mode {qlora,bf16} --frames N --cutoff-len L [--lean] [--keep-layers K]`).
Raw JSON results: `idea-stage/qwen_smoke_tmp/res_*.json`. Logs: `logging/runs/qwen_smoke/`.
