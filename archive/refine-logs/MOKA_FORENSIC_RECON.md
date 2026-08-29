# MokA MODALITY-ROUTED LoRA — FORENSIC RECON (zero-GPU)

**Agent:** MokA forensic recon (read-only source recon; **ZERO GPU / SLURM / Modal / test-touch / `state/` mutation**;
one local commit, no push). **Date:** 2026-07-26 NZST.
**Trigger:** `refine-logs/REPRO_SURVEY_2025.md` rank-3 candidate (§4.3, §5 #3) + **USER RULING 2026-07-26: MokA is
UNGATED** — the licence concern is waived, the code may be used directly, and MokA will be acknowledged/credited in
the paper (`autoresearch/goal_mllm_plus3/state/progress.json`, commit `6c4766e`).
**Clone under recon:** `external/baselines/MokA` @ `b28e834` ("update", 2025-12-27 code refresh), 616 files, no LICENSE.
**House-style precedent:** `refine-logs/FUSIONSWAP_FORENSIC_RECON.md` (`934bc9a`), `refine-logs/VISION_UNFREEZE_FORENSIC_RECON.md`.

---

## 0. BOTTOM LINE — **GO (staged, ZH-first, LOW prior)**

**GO** for a **single arm** (`A`-split routing only, **cross-attention NOT ported**) on **`MHC_zh` first**, with a
mid-family kill switch that auto-defunds the HateMM leg. **EN is NOT re-opened** (§4.1 — the closure covers it, and
F65 supplies the datum that GAP-5b demanded).

- **First decisive cell (ZH): ~4.6 A100-h** (§5). Family max (both datasets): **~8.8 A100-h**.
- **Integration: ~420–450 new lines, ZERO lines edited in vendored LLaMA-Factory, 1 in-repo file modified (+25).**
- **Codex gate: MANDATORY** (custom PEFT layer + forward-pre-hook + mask plumbing + an unmerged extraction path).
- **Honest prior P(goal: ≥+0.030 acc AND +0.030 mF1, 3/3 sign, BOTH protocols, on ≥1 dataset) = 5–8 %.**
- The GO rests on **one** premise that is *not* priced dead (§5.1): MokA's arithmetic side-effect gives the
  **dominant TEXT stream its own undiluted down-projection**, and text is the stream that carries **both** measured
  passes (F45 ZH, F58 HateMM). MokA's *advertised* premise (protect the weak visual modality) is **already priced
  dead** by F58 + F65 and is explicitly **not** what this cell bets on. That distinction is load-bearing and must
  survive into the prereg verbatim.

---

## 1. THEIR IMPLEMENTATION — SOURCE-LEVEL (file:line; the paper was not consulted)

MokA ships **two independent, non-shared implementations**. Only the `VisualText/` one is PEFT-modern and relevant.

### 1.1 `VisualText/` — the routed-LoRA layer (the liftable core)

**`external/baselines/MokA/VisualText/modified_peft/tuners/lora/layer.py:548`**

```python
def forward(self, x, my_text_mask, my_image_mask, question_mask, *args, **kwargs) -> torch.Tensor:
```

- **:589** routing is active only when `my_text_mask is not None`; otherwise **:672–678** falls back to a plain
  single-adapter LoRA using the `'text'` adapter (this is the KV-cache decode path).
- **:600** `lora_A_output_flat = torch.zeros(x_flat.shape[0], self.r['text'], ...)` — the rank buffer is sized from
  the **`'text'`** adapter's rank; a different image rank would silently break.
- **:603–611** `idx_text = (text_mask_flat==1).nonzero()` → `lora_A['text'](dropout['text'](x_text))`, scattered back.
- **:613–621** same for `idx_image` → `lora_A['image']`, `lora_dropout['image']`.
- **:655–657** `lora_B = self.lora_B['text']` — **the up-projection is SHARED**, taken from the `'text'` adapter.
- **:660–669** per-modality **scaling** (`self.scaling['text']` vs `self.scaling['image']`) then `result_flat[idx] +=`.

**How modality is identified — `VisualText/train/train.py:210-211`:**

```python
my_image_mask = input_ids == image_pad_id      # image_pad_id = tokenizer.convert_tokens_to_ids("<image>")
my_text_mask  = input_ids != image_pad_id
```

**Pure token-id equality on `input_ids`.** Not a processor position map, not token-type ids. Collator pads both masks
with `False` (`train.py:283-296`), so pad positions receive **no LoRA delta at all** (base layer only). A third mask,
`question_mask` (**:221–231**) = `~image_mask & (labels == -100) & (position > last_image_token)` = the instruction
text *after* the image block.

**How the mask reaches every `Linear` — by HARD-FORKING the modeling file:**
`VisualText/modified_models/modeling_llama.py` threads 3 extra **positional** args through every projection:
- **:150–159** `LlamaMLP.forward(x, latest_my_text_mask, latest_my_image_mask, latest_question_mask)` →
  `gate_proj/up_proj/down_proj`
- **:241–279** attention → `q_proj/k_proj/v_proj/o_proj`
- **:310–329** decoder layer reads the masks out of `kwargs` and **nulls them when `past_key_value.get_seq_length() > 0`**
  (post-first-token decode).
- `modelling_llava.py:580` overrides `_validate_model_kwargs` to let the 3 custom kwargs through `generate()`.

**Coverage: attention AND MLP** — `train/train.py:531-541`,
`lora_trainable = "q_proj,k_proj,v_proj,o_proj,gate_proj,down_proj,up_proj"`, filtered to modules whose name contains
`language_model` (vision tower and projector excluded; the projector is fully fine-tuned separately, `train.py:525-529`).

**Defaults (`train/train.py:546-561`):** `r=4`, `lora_alpha=16`, `lora_dropout=0.05`, `task_type=CAUSAL_LM`.
Adapters are created as `PeftMixedModel(model, cfg, adapter_name='image')` then `.add_adapter('text', cfg)` then
`.set_adapter(['image','text'])`.
**Recipe (`VisualText/shell/train.sh`):** 2 epochs, lr `1e-4`, cosine, warmup 0.03, bs 2 × grad-accum 2 × 8 GPUs, bf16,
DeepSpeed ZeRO-2.

### 1.2 Cross-attention: **YES, it is actually implemented** (the survey's "??" is resolved)

`layer.py:627-653`, inside the same forward, **before** the shared `B`:

- per sample, **image tokens (post-`A_v`) are the query**, **question tokens (post-`A_t`) are key and value**;
- scaled dot-product **in rank space** (`d_k = r`), softmax, `attention_output`;
- **:653** `lora_A_output[i, image_idx] += self.attn_weight * attention_output`.

`self.attn_weight` — **config/code mismatch**: `tuners/lora/config.py:166` defaults it to **0.5**;
`tuners/lora/layer.py:408` `kwargs.get('attn_weight', 1)` defaults to **1**. The config value wins in practice
(`tuners/lora/model.py:199` passes it), so **0.5** unless overridden. The `AudioVisualText` shell script passes
`--blc_weight 1`, i.e. the shipped AVT recipe uses **1.0**, not the config default 0.5.

The loop is **per-sample Python** (`for i in range(batch)`) inside **every** LoRA layer of **every** decoder layer —
196 invocations per forward step at 28 layers × 7 projections. This is a material throughput cost and a second
manipulated variable. **Recommendation: do NOT port it in the first bite** (§3.2).

### 1.3 `AudioVisualText/` — the older, separate implementation

`external/baselines/MokA/AudioVisualText/peft_hyper/tuners/lora.py` (531 lines, a fork of PEFT ~0.2-era).

- **:309–322** the rank is encoded as the **decimal digits of an integer**: `lora_r = 444` → `rr = [4,4,4]`
  (`configs/unified_config.py:91`). `lora_A0/1/2` per modality; **`for i in range(1): lora_B0`** — a single shared `B`
  (no dead twin here, unlike VisualText).
- **:460–532** train branch. Routing is **multiplicative** (`only_inputs = [x*text_mask, x*video_mask, x*audio_mask]`)
  and the three `A` outputs are **summed** before the shared `B` — mathematically identical to VisualText's
  gather/scatter given disjoint masks, but it runs all 3 `A`s on the full sequence (3× the `A` FLOPs).
- **:480–521** the same question-conditioned cross-attention, for video and audio, weighted by `blc_weight`.
- Scaling is applied to the `A` output (`*self.scaling[0]`), not after `B` — numerically the same.
- Recipe (`scripts/finetune/ft_musicavqa.sh`): 3 epochs, lr `1e-4`, cosine, warmup 0.03, bf16, `--ddp_find_unused_parameters True`.

### 1.4 DEAD CODE / CONFIG-vs-CODE MISMATCHES (SynIB precedent — trusting nothing)

| # | Finding | Evidence |
|---|---|---|
| D1 | **`lora_B['image']` is created and trained but NEVER used.** `update_layer` (`layer.py:109-110`) builds `lora_A[name]` **and** `lora_B[name]` for every adapter; `forward:657` hard-codes `lora_B['text']`. `grep "lora_B\['image'\]"` over `layer.py` → **zero hits**. `train.py:570-576` marks every param containing `lora` trainable. ⇒ **27.5 % of MokA's LoRA parameters receive zero gradient signal by construction** (and inflate the shipped param count to exactly 2.0× standard LoRA). | `layer.py:109-110,:657`; `train.py:570-576` |
| D2 | `update_layer` hard-codes `dtype=torch.bfloat16` on both `A` and `B` (`layer.py:109-110`) — an unconditional divergence from upstream PEFT. | `layer.py:109-110` |
| D3 | `print('train mode')` fires **per decoder layer, per step** in the training path. | `modeling_llama.py:316` |
| D4 | `attn_weight` default is **0.5** in the config and **1** in the layer fallback. | `config.py:166` vs `layer.py:408` |
| D5 | AVT: `reserved_modality` and `blc_alpha` are plumbed through the config, stored on the layer (`lora.py:301,:304`), and **never read in any forward** — dead ablation knobs. | `AudioVisualText/peft_hyper/tuners/lora.py:301,304` |
| D6 | AVT: `finetune.py:82-83` **hard-codes** `lora_alpha = 16` and `lora_dropout = 0.05`, silently overriding the `--lora_alpha` / `--lora_dropout` CLI flags. | `scripts/finetune/finetune.py:82-83` |
| D7 | `layer.py:581` `torch_result_dtype` is commented out and **:609/:619** the `x.to(lora_A.weight.dtype)` casts are commented out — dtype safety removed relative to upstream PEFT. | `layer.py:581,609,619` |
| D8 | The rank-as-decimal-digits encoding (`lora_r=444`) breaks silently for any rank ≥ 10. | `AudioVisualText/.../lora.py:309-312` |

**Nothing beyond `A`-split + shared `B` + question→visual cross-attention is implemented.** There is no balance
loss, no gating, no router — `blc_alpha` (the only thing named like a loss) is dead (D5).

---

## 2. OUR SFT STACK — RECON (file:line, ground truth from the deployed artefacts)

### 2.1 What actually runs

| stage | entry point | notes |
|---|---|---|
| data build | `src/utils/build_lora_sft_data.py` | 8 frame JPGs/video → multi-image ShareGPT; `IMG_TOKENS = "<image>"*8` prepended to the instruction, which embeds the transcript. |
| curriculum build | `src/utils/build_curriculum_sft_data.py` | **multiset re-weighting, NOT ordering** — see §3.4. |
| SFT | `scripts/slurm/lora_sft.sbatch` → `python src/train.py <yaml>` in `RA-HMD/LLAMA-FACTORY-Ver202512` | `DISABLE_VERSION_CHECK=1`, nvcc shim, `HF_HUB_OFFLINE=1`. |
| extraction | `scripts/slurm/gen_embed_lora.sbatch` → `src/utils/generate_VideoMLLM_embedding_lora_HF.py` | **`merge_and_unload()`** — see §2.4, this is the blocker. |
| head | `scripts/slurm/enc3seed*.sbatch` → `src/run_rac.py` | 3 head-seeds, cached features. |

### 2.2 Pinned LoRA facts (verified against `logging/lora/HateMM/adapter_model.safetensors`, not the yaml)

- **PEFT `0.14.0`**, transformers `4.49.0`, torch `2.6.0+cu124`, accelerate `1.5.2` (conda `HateVideo`).
  LLaMA-Factory `requirements.txt:6` pins `peft>=0.14.0,<=0.17.1`.
- Recipe (`my_configs/hatevideo/hatemm_qwen25vl_lora_sft.yaml`): `lora_rank: 16`, `lora_alpha: 32`,
  `lora_target: q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj`, `freeze_vision_tower: true`,
  `freeze_multi_modal_projector: true`, `lr 1.0e-4`, `3.0` epochs, bs 1 × grad-accum 8, cosine, warmup 0.05, bf16,
  gradient checkpointing, `cutoff_len 4096`, `save_strategy: epoch`. **`lora_dropout` unset ⇒ 0.0.**
- **Adapter ground truth:** 392 tensors = 28 layers × 7 projections × {A,B}. **Zero `visual.*` keys** — the adapter is
  **LLM-decoder only**. Total LoRA params **40,370,176**, byte-matching the trainer's
  `trainable params: 40,370,176 || all params: 8,332,536,832 || trainable%: 0.4845`
  (`logging/slurm/lora_sft_13233.out:308`) ⇒ **nothing else is trainable**.
- Deployed adapters: `logging/lora/MHC_zh` (job **12143**, 2026-07-02, 02:39:49),
  `logging/lora/HateMM` (job **13233**, 2026-07-18, 03:11:18),
  `logging/lora/HateMM_curric` (job **13238**, 2026-07-18), `HateMM_curric_rep2` (13244, 05:00:26).
  All three configs are **byte-identical apart from `dataset:` and `output_dir:`** (verified by `diff`).

### 2.3 Where PEFT is injected (the insertion point)

`RA-HMD/LLAMA-FACTORY-Ver202512/src/llamafactory/model/adapter.py`:
- **:19** `from peft import LoraConfig, ..., get_peft_model`
- **:214–233** target-module resolution → `patch_target_modules` (`model/model_utils/visual.py:182-199`), which for
  `qwen2_5_vl` (registered at `visual.py:344-352`, `vision_model_keys=["visual.patch_embed","visual.blocks"]`,
  `projector_key="visual.merger"`) expands the 7 short names into explicit module paths **excluding** the vision tower.
- **:253–262** `peft_kwargs`, **:301–305** `LoraConfig(...)`, **:312** `model = get_peft_model(model, peft_config)`.
- Called from `model/loader.py:190` (`init_adapter`).

⇒ **`llamafactory.model.adapter.get_peft_model` is a module attribute** and can be monkey-patched from a wrapper
entry point **without editing a single line of the vendored LLaMA-Factory tree**.

### 2.4 Can we derive a per-token modality mask inside that stack? **YES — exactly, and cheaply.**

- SFT records carry 8 `<image>` placeholders; the Qwen2.5-VL processor expands each into `<|image_pad|>` tokens
  (**id 151655**; `<|video_pad|>` = **151656**, `<|vision_start|>` 151652, `<|vision_end|>` 151653 — read from the
  local tokenizer).
- `transformers/models/qwen2_5_vl/modeling_qwen2_5_vl.py:1803-1809,:1821-1827` performs
  `inputs_embeds.masked_scatter(image_mask, image_embeds)` — **vision embeddings replace the pad tokens 1:1 in place;
  the sequence is never shifted or re-lengthened.** Our own extractor already asserts this invariant
  (`src/utils/generate_VideoMLLM_embedding_lora_HF.py:347`).
- ⇒ `image_mask = (input_ids == 151655) | (input_ids == 151656)` is **positionally valid at the input of every one of
  the 7 targeted projections in all 28 decoder layers** (`q/k/v/o` see `[B,S,3584]`; `gate/up` see `[B,S,3584]`;
  `down` sees `[B,S,18944]` — all share the same sequence axis `S`).

**Consequence: we do NOT need MokA's modeling-file fork.** A single `register_forward_pre_hook(with_kwargs=True)` on
the base `Qwen2_5_VLForConditionalGeneration` computes the mask once per batch from `kwargs["input_ids"]` and stashes
it; each routed layer reads it. This is the single largest cost saving versus a faithful port — and it removes the
`generate()`/KV-cache path complexity entirely, since we never generate (we take hidden states, `use_cache=False`).

**Gradient-checkpointing note (must be in the codex gate):** the stash is written in the pre-forward hook and must
**not** be cleared at end-of-forward, because checkpoint recomputation replays the block during backward. Overwrite
per batch; never clear.

### 2.5 The **real** blocker: extraction merges the adapter

`src/utils/generate_VideoMLLM_embedding_lora_HF.py:485-487`:

```python
model = PeftModel.from_pretrained(model, lora_dir)
model = model.merge_and_unload()
```

**MokA has no merged form** — the delta is token-dependent (`B·A_v·x` vs `B·A_t·x`), so there is no single `ΔW`.
The extractor needs a `--moka` branch that keeps the adapter **live** and registers the mask hook.

**Correcting the survey.** `REPRO_SURVEY_2025.md` §4.3(b)/§5 #3 states MokA "requires switching extraction to a
joint multimodal forward (currently image and text are separate forwards) — that is the load-bearing engineering
step and the main risk." **This is wrong on the facts.** `_build_messages` (**:305–315**), used by the single `_encode` helper (**:318–387**), *always* emits
`[{"type":"video", ...}, {"type":"text", ...}]`; **both** the `img_feats` call (**:409-412**, `span="prefix"`) and
the `text_feats` call (**:425-428**, `span="response"`) run the **same joint video+text forward**, differing only in
the pooling span and the instruction string. The two streams are two **pooling windows over joint forwards**, not
two unimodal forwards. The real engineering risk is the **merge**, not the forward.

---

## 3. INTEGRATION DESIGN (minimal diff)

### 3.1 Files and line estimate

| file | status | ~lines | content |
|---|---|---|---|
| `src/moka/routed_lora.py` | NEW | ~150 | `MokaLinear(peft.tuners.lora.layer.Linear)` adding `lora_A_v: nn.ModuleDict`; routed `forward`; `install_moka(peft_model)` swapping every `lora.Linear` in place and registering the mask pre-hook; `_MASK` module-level stash. |
| `src/moka/train_moka.py` | NEW | ~40 | monkey-patch `llamafactory.model.adapter.get_peft_model` → `get_peft_model` + `install_moka`, then `run_exp()`. **Zero edits to the vendored tree.** |
| `src/utils/generate_VideoMLLM_embedding_lora_HF.py` | MODIFIED | +25 | `--moka` flag: skip `merge_and_unload()`, call `install_moka`, load the `lora_A_v` tensors explicitly. |
| `scripts/slurm/lora_sft_moka.sbatch` | NEW | ~60 | clone of `lora_sft.sbatch`, entry point `src/moka/train_moka.py`. |
| `scripts/slurm/enc3seed_moka.sbatch` | NEW | ~90 | clone of `enc3seed_zh_b3.sbatch` with `LORA=Qwen2.5-VL-7B-Instruct-LoRA-moka_HF`. |
| `my_configs/hatevideo/{mhc_zh,hatemm}_qwen25vl_lora_moka*_sft.yaml` | NEW | ~2×45 | copies of the deployed yaml with `output_dir` changed **only**. |
| `scripts/analysis/moka_smoke.py` | NEW | ~90 | CPU equivalence / mask-coverage / grad-flow asserts (§3.5). |

**Total ≈ 420–450 new lines, 1 in-repo file modified (+25), 0 vendored lines touched.**

### 3.2 Scope decision — **`A`-split ONLY; cross-attention OUT**

Porting `layer.py:627-653` would (i) add a second manipulated variable, (ii) add an untuned hyper-parameter
(`attn_weight`, whose own default is inconsistent, D4), (iii) require a "question" span our SFT records do not have
(our prompt is one user turn: 8 `<image>` + instruction), and (iv) cost a per-sample Python loop in 196 call sites
per step. **Out of scope; adding it re-costs a bite.**

### 3.3 Rank budget and parameter delta — **DECISION: `r_v = r_t = 16` (unsplit)**

The shared `B` forces `r_v = r_t` (both `A`s feed the same up-projection), so the only two options are:

| variant | per-layer params | total (28 layers) | ratio vs deployed | per-token rank |
|---|---|---|---|---|
| **deployed LoRA** (`A`+`B`, r=16) | 1,441,792 | **40,370,176** ✅ matches safetensors | 1.000× | 16 |
| **MokA r=16** (`A_t`+`A_v`+shared `B`) | 2,088,960 | **58,490,880** | **1.4489×** | **16** |
| MokA r=8 (halved) | 1,044,480 | 29,245,440 | 0.7244× | 8 |
| MokA **as GeWu ships it** (dead `B_v`, D1) | 2,883,584 | 80,740,352 | 2.000× | 16 |

**Choose r=16 and disclose +44.89 % parameters.** Justification, and it is stronger than any prior comparability
story in this campaign: **at any given token the effective rank is 16 and the FLOPs are identical to the deployed
adapter** — exactly one `A` fires per token. The extra parameters are *inactive* per token; the manipulated variable
is *which* `A` a token is routed through. Halving to r=8 would confound routing with a capacity **reduction**.
Precedent: FUSIONCAT accepted **2.0×** as "within the ~2× comparability guidance" for a change that (unlike this one)
altered per-token capacity. **1.449× with identical per-token rank/FLOPs is strictly inside that precedent.**

**Init:** `A_v` and `A_t` both Kaiming (PEFT's `reset_lora_parameters`), `B` zero-init ⇒ `ΔW = 0` at step 0, exactly
as standard LoRA. Same recipe ⇒ same optimizer, same lr, same schedule.

### 3.4 Curriculum compatibility — **TRIVIAL, verified**

`src/utils/build_curriculum_sft_data.py` docstring + `:172-174,:209` : the curriculum is a **deterministic
largest-remainder re-apportionment of example multiplicity** (`w_i = 1 + LAMBDA*c_i`, `CAP_RATIO = 1.0`), emitting
records that are **byte-identical** to the generic arm's, with the total capped to `N_train` so the 3-epoch step count
is unchanged. It is **not** an ordering (`disable_shuffling` exists at `finetuning_args.py:555` and is **not set**).
⇒ MokA composes with the curriculum by construction: the routed adapter is orthogonal to which records appear how
often, and the HateMM-curric cell reduces to "same `train_curric.json`, `--moka on`".

### 3.5 What must be proven before GPU (codex-gate items)

1. **CPU forward equivalence.** With `A_v` weights tied to `A_t`, `MokaLinear.forward` must equal upstream PEFT
   `Linear.forward` with **max-abs-diff 0.0** in fp32 on random tensors, for all-text, all-image and mixed masks.
2. **Mask coverage.** On a real batch: `image_mask.sum()` must equal the processor's vision-token count, and
   `image_mask | text_mask | pad_mask` must cover the sequence exactly once.
3. **Grad flow.** After one backward, `A_t.grad` and `A_v.grad` must both be non-zero and `B.grad` non-zero; no
   parameter may have `requires_grad=True` and `grad is None` (the D1 trap).
4. **Save/load round-trip.** `get_peft_model_state_dict` keeps any key containing `"lora_"`
   (`peft/utils/save_and_load.py:85`), so `...lora_A_v.default.weight` **is** saved — but
   `set_peft_model_state_dict` would **silently drop** it if the layer class is not installed before loading.
   Install the custom class in **both** the train and extract paths (via `LoraModel._create_new_module` /
   `install_moka` before weight load) and assert `len(missing) == len(unexpected) == 0`.
5. **Checkpoint-recompute safety** (§2.4 note).
6. **Merge guard.** `merge_and_unload()` must **raise** on a MokA model, never silently produce a wrong `ΔW`.

### 3.6 Pre-declared comparability caveat — **merged vs unmerged extraction precision**

The banked floors' features came from a **merged** encoder (`W + BA` folded, one bf16 matmul); MokA's must come from
an **unmerged** encoder (`Wx + B(A_m x)`, different bf16 accumulation order). A routing-OFF MokA therefore **cannot**
reproduce the floor cache bit-exactly, and the G-repro must be at the **layer** level (fp32 CPU, item 1 above), not
the pipeline level. **Pre-declare `KS-MOKA-0b`:** extract the *already-banked* generic ZH adapter through the
unmerged path (~0.6 GPU-h, **zero test-touch — features carry no labels**) and report mean per-item cosine against
the banked merged cache. If mean cosine < 0.9999, a same-path unmerged floor head run (3 seeds, +0.05 GPU-h,
+3 test evaluations) becomes **mandatory** before any verdict.

---

## 4. DATASET ORDER + BAN ADJUDICATION

### 4.1 MHC-EN — **CLOSED. MokA does NOT re-open it. No EN arm in scope.**

Verbatim closure record (`state/directions_tried.json`, findings F53/F55; `PREMISE_D_GATE_RECORD.md:231`):

> "MHC-EN now closed at frozen (F50), collapsed-adapted-deployed (B4/F53), and healthy-img+adapted-text composition
> (F55) levels simultaneously."
> "**No representation lever — adapted or not — clears +0.03 on EN** … **EN is closed to the entire family**"
> (`WAVE5_CANDIDATES.md` §0.2)

**Honest adjudication.**
- `REDTEAM_BAN_SCOPE_AUDIT.md` GAP-5 correctly ruled the "all levels / entire family" wording an **INDUCTIVE LEAP**
  and named exactly two unmeasured levels: **(a) EN + audio**, **(b) EN + vision-obligatory SFT**. MokA is **neither**.
- MokA is a **different adaptation structure at level (b)-adjacent**, i.e. an alternative *operator* inside the
  already-measured "adapted representation" level — and the closure language explicitly says "**adapted or not**".
  **By the letter, EN is covered.**
- More decisively, the closure is now **measured, not asserted**: **F65** (`VISION_UNFREEZE_VERDICT_REVIEW.md`,
  `09d02f8`) discharged GAP-5b — EN's collapsed image stream **MOVED** (+0.0320 trLOO / +0.0065 dev, "first lever ever
  to move the collapsed EN image stream") and **K-V2 was a TIE on both datasets and both protocols**, EN FAIL both.
  MokA's EN pitch in the survey was precisely "is F44's EN image-stream collapse an adaptation-subspace artefact?" —
  **F65 answered the operative half: the collapse is fixable, and fixing it converts nothing** (8th law-I instance).
- Unlike the F50 letter-overreach that F83 found for trained fusion, **this is a measured cap, not a base-rate prior.**

**Ruling: EN stays closed. Proposing an EN MokA cell would be a re-burn.** Record this in the prereg so the
0-context reviewer does not have to re-derive it.

### 4.2 Decisive order — **ZH (`MHC_zh`) FIRST, HateMM conditional**

| | `MHC_zh` | `HateMM` |
|---|---|---|
| floor | job **13150** (generic LoRA_HF) — val-sel **0.8322/0.8015**, final **0.8456/0.8173** | job **13241** (curric LoRA) — val-sel **0.8775/0.8711**, final **0.8791/0.8726** |
| status | goal leg: B3 **MARGINAL** final-epoch pass, val-sel FAIL (78-dev noise, F45) | **PASS both protocols** (F53 +0.0573/+0.0682); project-best cell |
| is adaptation the load-bearing lever? | **YES** — F58: "ZH's is **LoRA-SPECIFIC** (frozen fails)" | **NO** — F58: "HateMM's is **frozen-SUFFICIENT** (LoRA inherits)"; image stream LoRA−frozen **+0.0045/+0.0062 = FLAT** |
| headroom to a FORMAL pass | +0.030 → 0.8756 (val-sel) / 0.8756 (final) — reachable | +0.030 → **0.909**, near ceiling — **arithmetically implausible** |
| SFT cost | `train_runtime` **8,636 s** (2.40 h), job wall 02:39:49 | **10,255 s** (2.85 h), job wall 03:11:18 (curric 10,414 s / 03:14:44) |
| arm shape | MokA-generic vs generic floor — **single clean variable** | MokA+curric vs curric floor — two non-deployed features stacked |

**ZH first**, on four independent grounds: it is the **binding goal leg** (HateMM already passes and cannot
plausibly clear +0.030 from 0.879); it is the **only dataset where the adaptation operator is measurably
load-bearing** (F58) so a null there is informative rather than insensitive; it is **16 % cheaper**; and its floor is
the **generic** adapter so the diff is single-variable. HateMM is a *hold-the-pass* leg, sequenced second and
auto-defunded by `KS-MOKA-1`.

### 4.3 Standing-veto compliance

Single-dataset own-train-split only (no cross-dataset mixing) ✅; no OCR channel ✅; no gold spans/attributes ✅;
no cross-seed ensembles ✅; no MLLM-scores-as-training-signal ✅; raw videos never leave the machine ✅
(SFT reads local frames; only derived `.pt` → B2). MokA licence: **user-waived, credit in paper** (`6c4766e`).

---

## 5. COST + HONEST PRIOR

### 5.1 Prior

**Against (all measured, all binding on MokA's *advertised* premise):**
- **F58** — HateMM's pass is **text-carried and frozen-sufficient**; image stream LoRA−frozen = **+0.0045/+0.0062
  (FLAT)**. **F45** — ZH gain lives **entirely** in the text stream; image stream flat (0.718/0.721/0.714).
  ⇒ On **both** converting datasets the image stream contributes ≈0 to the adaptation gain.
- **F65** — the image stream is **movable** (+0.0320 trLOO on EN) and moving it **converts nothing** (K-V2 TIE
  everywhere). **8th law-I instance.** MokA's stated goal — protect the dominated visual modality — is priced at ~0.
- `LITSWEEP2_FRESH_2026` HUNT-3 priced "richer PEFT adapters (MoLE, Task-Adapter++)" at ~0 as tactics on a measured axis.
- ZH's binding val-sel leg must clear +0.030 through the **78-item dev** selection wall (F45/F63), whose head-seed
  noise band is ±0.014.

**For (the one premise not priced dead — this is what the cell actually bets on):**
- MokA's arithmetic side-effect is that the **dominant TEXT stream gets its own `A_t`, undiluted by image-token
  gradient**. Text is the stream that carries **both** measured passes. F58/F65 price *image-stream improvements*;
  **no finding prices a sharpened text-side adaptation subspace.** F45 already shows ZH text train-LOO AUC moving
  0.847 → 0.925 under a *shared* `A`; whether an undiluted `A_t` moves it further is **unmeasured**.
- Encoder-SFT is the **only** axis in the campaign that ever formally converted (F53, F45/B3).
- Every banked adapter — B3, F53, curric, vis, bidir — used **one shared `A` across modalities**. Modality-routed
  adaptation has **never been run on this project**.
- Per-token rank and FLOPs identical ⇒ the cleanest single-variable comparability available.

**P(goal: ≥+0.030 acc AND +0.030 mF1, 3/3 sign, BOTH protocols, ≥1 dataset) = 5–8 %.**
**P(any KS-surviving movement worth reporting) = 25–30 %.**
**Novelty class: D7-relevant but weak** — a published PEFT variant transplanted onto our encoder. A PASS is a
performance row plus a mechanism sentence; it is **not** a novelty mechanism. Say so in the prereg.

### 5.2 GPU budget (all A100-h; wall times read from `sacct` / `train_results.json`, not estimated)

**Stage 1 — ZH (the first decisive cell):**

| item | basis | GPU-h |
|---|---|---|
| GPU smoke (20-step SFT + 5-item extract) | — | 0.2 |
| `KS-MOKA-0b` merge-drift extract (ZH, 3 splits, unmerged banked adapter) | `lora_embed` 13234/13239/13240/13245/13302 = 26–37 min | 0.6 |
| MokA-ZH SFT | `train_runtime` 8,636 s × ~1.2 routing overhead + ~0.25 h build/load | **3.1** |
| MokA-ZH extraction (unmerged + routed, 3 splits) | 0.6 × ~1.15 | 0.7 |
| 3 head-seeds (`enc3seed`) | job 13150 = 00:02:46 for exactly 3 runs (`enc3seed_zh_b3.sbatch:36-40`) | 0.05 |
| **Stage-1 total** | | **≈ 4.6** |

**Stage 2 — HateMM (conditional on `KS-MOKA-1`):** SFT 3.3 + extract 0.75 + heads 0.05 ≈ **+4.2**.
**Family maximum ≈ 8.8 GPU-h**; expected ≈ 4.6 (the ZH kill switch fires in the modal case).
Reference points: vision-unfreeze (F65) spent ~15 GPU-h; NCA family (F75) ~0.33; FUSIONCAT ~0.1.
**Plus ~1 person-day of integration and a mandatory codex gate** — the campaign's largest code risk since bidir.

---

## 6. FAMILY + KS / FORMAL DRAFT (for the prereg author)

**Family `MOKA`.** One arm — **`moka` = modality-routed-`A` LoRA-SFT encoder** (`A_v` for `<|image_pad|>`/`<|video_pad|>`
positions, `A_t` for all other non-pad positions, **shared `B`**, `r_v = r_t = 16`, `alpha 32`, dropout 0.0,
same 7 projections × 28 decoder layers, **no cross-attention**) × **{`MHC_zh`, then `HateMM`}** × **3 head-seeds**,
**sequenced**, paired **within head-seed** against each dataset's **own banked floor** (ZH **13150** generic-LoRA,
HateMM **13241** curric-LoRA), **dual protocol** (val-selected AND final-epoch), each dataset trained on its **own
train split only**. Encoder tags `Qwen2.5-VL-7B-Instruct-LoRA-moka_HF` (ZH) and
`Qwen2.5-VL-7B-Instruct-LoRA-curric-moka_HF` (HateMM). **One bite.**

**Single-encoder-draw declaration (mandatory, B3/F53 precedent).** One SFT run per dataset; `--seed` varies the
**head** only. The encoder is a single draw and its seed variance is **not** estimated — declare it exactly as F53 did.

**Test-touch budget:** Stage 1 = **3** (ZH × 3 seeds); Stage 2 = **3**; `KS-MOKA-0b` = **0** (feature cosine only,
no labels). Contingent +3 if `KS-MOKA-0b` forces the same-path floor. **Zero test-touch before the independent verdict.**

**Kill switches (frozen wording to be finalised by the prereg author):**

- **`KS-MOKA-0` (pre-GPU, mandatory, codex-gated).** All six §3.5 asserts pass, incl. tied-`A_v` fp32 max-abs-diff
  **0.0** vs upstream PEFT. Any failure ⇒ **no submission**.
- **`KS-MOKA-0b` (pre-verdict, §3.6).** Mean per-item cosine(unmerged, merged) on the banked ZH cache ≥ **0.9999**,
  else the same-path unmerged floor head run becomes mandatory.
- **`KS-MOKA-1` (ZH decisive, mid-family).** If on **both** protocols mean paired Δacc ≤ 0 **or** the acc sign is not
  3/3 positive, the ZH arm is **DEAD** and the **HateMM leg is AUTO-DEFUNDED** (saves ~4.2 GPU-h). Law-I / FLAT.
- **`KS-MOKA-2` (routing-is-real, $0).** Report per-layer `‖A_v − A_t‖_F / ‖A_t‖_F`. If the two down-projections
  converge (< 5 % relative difference at the median layer) the arm is a **null-op** and any observed delta is head-seed
  noise, not routing — report it as such and do **not** claim a routing effect.
- **`KS-MOKA-3` (stream decomposition, $0, mandatory before any claim).** Re-run the F45/F58 train-LOO img/text-AUC
  machinery on the MokA cache vs its floor. Three pre-declared readings: **text moved** ⇒ the §5.1 bet is confirmed
  and must be reported as a *text-side* mechanism, **never** as "MokA protected the visual modality"; **image moved,
  head flat** ⇒ the **9th law-I instance**, report as such; **neither moved** ⇒ null-op, cross-check `KS-MOKA-2`.
- **`KS-regression`.** MokA below floor by ≥ 0.030 on either protocol ⇒ report as a measured REGRESSION finding.

**FORMAL bar (house standard, unchanged).** Mean paired Δ ≥ **+0.030 acc AND +0.030 mF1**, sign **3/3**, on **BOTH**
protocols, for a dataset-level PASS. **Goal** = that on **≥ 2** datasets.

**Codex gate: MANDATORY** (model-internals: custom PEFT layer, forward-pre-hook, mask plumbing, unmerged extraction
path, save/load key survival). This is not a flag-only arm; the FUSIONCAT §4.5 gate exemption does **not** apply.

---

## 7. PROVENANCE

Read-only. **Zero** GPU / SLURM / Modal / test-touch. **No `state/` mutation, no push.** Numbers in §2.2/§3.3 were
re-derived from `logging/lora/HateMM/adapter_model.safetensors` (392 keys, 40,370,176 params) and cross-checked
against `logging/slurm/lora_sft_13233.out:308`; wall times from `sacct` (12143, 13233, 13237, 13238, 13244, 13150,
13234/13239/13240/13245/13302) and `logging/lora/*/train_results.json`; floor metrics quoted from
`refine-logs/BIDIR_STAGE1_VERDICT_REVIEW.md` §3.1/§3.2; findings F44/F45/F53/F55/F58/F65/F66/F70/F75/F83 read from
`autoresearch/goal_mllm_plus3/state/findings.jsonl`; GAP-5 from `refine-logs/REDTEAM_BAN_SCOPE_AUDIT.md:217-260`.
Every MokA `file:line` was read in `external/baselines/MokA` @ `b28e834`.

**One survey correction is recorded (§2.5):** the "extraction runs image and text through separate forwards" claim in
`REPRO_SURVEY_2025.md` §4.3(b)/§5 #3 is **false** — both extraction calls are joint video+text forwards differing
only in pooling span. The load-bearing engineering step is the **`merge_and_unload()` incompatibility**, not the
forward topology. This *lowers* MokA's integration risk relative to the survey's rank-3 rationale.
