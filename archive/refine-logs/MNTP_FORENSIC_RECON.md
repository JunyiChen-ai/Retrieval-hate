# MNTP STAGE-2 — forensic recon (zero-GPU, review-ready design)

**Author:** MNTP forensic-recon agent. **ZERO GPU / ZERO SLURM / ZERO Modal / ZERO downloads / ZERO
`src/` edits / ZERO test-touch.** `autoresearch/.../state/` untouched. No prereg, no job submission, no
promotion. Repo HEAD at recon: `6c8929d` (working tree dirty; this analysis commits only this file plus
`scripts/analysis/mntp_rawkey_devscreen{.py,_OUT.json}`).
**Date:** 2026-07-27 NZST. **Gate:** user opened the F72-routed MNTP stage-2 funding decision.

---

## 0. VERDICT UP FRONT

**GO — but NOT to MNTP first.** The recon produced a new $0 measurement that changes the shape of the cell.

> **The F72 crater is not uniform across the two streams. It is almost entirely confined to the TEXT
> stream, which is the one stream read with a LAST-TOKEN-class readout. The mean-pooled IMAGE stream —
> same weights, same mask flip, same forward, same everything except which span is pooled — is
> UNHARMED and slightly BETTER under bidirectional attention (HateMM +0.0093, ZH +0.0128 dev acc).**

LLM2Vec's own ablation says mean pooling is best and **EOS pooling is worst**; our deployed text readout is
EOS-class. So a second hypothesis competes with F72's "causally-trained weights break" diagnosis, and it was
never separated: **the readout is mismatched to the topology.** The two hypotheses prescribe different
spends (H2 costs 0.5 GPU-h and no training and no corpus ruling; H1 costs 2-4 GPU-h, a corpus ruling, and
new training code), so the cheapest-kill discipline says **split them before funding MNTP**.

**Staged plan (each stage gated, each gate cheap):**

| stage | what | GPU-h | needs user ruling? |
|---|---|---|---|
| **S0** | $0 CPU belts + recon (this document) | 0 | no — done |
| **S1** | **bidir + MEAN-POOL readout**, no training | ~1.0 (both ds) | no |
| **S2a** | **published McGill MNTP adapter transplant**, no training | ~1.0 (both ds) | download gate only |
| **S2b** | MNTP trained by us | 2-4 /ds | **YES — corpus ruling** |
| **S3** | full ceremony, 3-seed, both protocols, single test-touch | ~0.2 | only on S1/S2 dev MOVE |

**NO-GO to funding S2b before S1 reports.** S1 is a strictly cheaper experiment that can dissolve the
premise S2b is built on.

---

## 1. THE NEW MEASUREMENT (the load-bearing part)

### 1.1 What was run

`scripts/analysis/mntp_rawkey_devscreen.py` (sha256 `8bc009e68833d8bad3aecb531c7c8b9879e05a2e00430465e0b2b4f05f9dede0`),
output `scripts/analysis/mntp_rawkey_devscreen_OUT.json`. CPU-only, seconds of wall time.

Operator = **the deployed one, verbatim** (`src/utils/metrics.py:229-284`, `use_sim=True`,
`majority_voting='arithmetic'`): L2-normalised keys → top-20 by cosine over the **own-train** memory →
labels mapped to ±1 → multiplied by cosine → rank-weighted `[20..1]` → normalised by Σw → decision
`sigmoid(v) ≥ 0.5 ⟺ v ≥ 0`.

Inputs = **banked caches only**, all present and id-aligned:
`data/CLIP_Embedding/HateMM/{train,dev_seen}_Qwen2.5-VL-7B-Instruct-LoRA-curric{,-bidir}_HF.pt` and
`data/CLIP_Embedding/MHC_zh/{train,dev_seen}_Qwen2.5-VL-7B-Instruct-LoRA{,-bidir}_HF.pt`.

**Scope discipline:** this is the **raw untrained key space** (no head trained) on **DEV only**. It is the
ERRPAT §3(a) lens, applied to the bidir arm for the first time. **No test file was opened.** It is a
*proxy* for the head, used here to calibrate a kill bar — not a verdict.

### 1.2 Result

| dataset | stream (readout) | causal | bidir | Δ acc | Δ mF1 |
|---|---|---|---|---|---|
| **HateMM** | **img** (prefix **mean-pool**) | 0.7570 | 0.7664 | **+0.0093** | +0.0049 |
| | **text** (assistant-tail = **last-token**) | 0.8037 | 0.7570 | **−0.0467** | −0.0626 |
| | concat | 0.8505 | 0.7944 | −0.0561 | −0.0627 |
| **MHC_zh** | **img** (prefix **mean-pool**) | 0.7436 | 0.7564 | **+0.0128** | +0.0117 |
| | **text** (assistant-tail = **last-token**) | 0.8462 | 0.6282 | **−0.2179** | −0.3151 |
| | concat | 0.8590 | 0.6410 | −0.2179 | −0.3081 |

Feature drift, mean per-item cosine(bidir, causal), id-matched:

| dataset | train img | train text | dev img | dev text |
|---|---|---|---|---|
| HateMM | 0.7902 | **0.3767** | 0.7946 | **0.4052** |
| MHC_zh | 0.6786 | **0.3622** | 0.6762 | **0.3568** |

The text representation is displaced ~2× further than the image representation (cos ~0.36-0.41 vs
~0.68-0.79), and it is the only one whose downstream vote collapses.

### 1.3 Why this is mechanistically coherent

Read `src/utils/generate_VideoMLLM_embedding_lora_HF.py:372-400`. The two streams differ in **which span is
mean-pooled**:

- `img_feats`, `span="prefix"` (`:385`) — `last_hidden[:end].mean(0)` where `end` = the last `<|im_start|>`.
  This is a **mean pool over the whole vision+instruction prefix** (~770 vision tokens + instruction).
- `text_feats`, `span="response"` (`:400`) — `last_hidden[start:].mean(0)` where `start` = the **last**
  `<|im_start|>`. This pools only the trailing `<|im_start|>assistant\n` header — **3-4 format tokens at
  the very end**. The code comment says it plainly: *"the canonical causal-LM sentence embedding"*.

Under **causal** attention the final tokens are the *only* ones that have seen the whole input, so
last-token pooling is privileged and correct. Under **bidirectional** attention that privilege evaporates —
every token sees everything — and the readout is left pooling content-free format tokens whose bidirectional
representations are dominated by local formatting, not by the transcript. Mean pooling, by contrast, is
topology-robust: it was already an aggregate under causal and remains one under bidir.

**This is exactly what LLM2Vec prescribes.** From the paper: *"LLM2Vec is compatible with all three
approaches and works best with mean pooling"*; **EOS pooling performs worst**. The README: *"By default the
LLM2Vec model uses the `mean` pooling strategy."* Every published LLM2Vec bidir result uses mean pooling.
**F72 flipped the mask while keeping an EOS-class readout — the one pooling LLM2Vec's own ablation names as
worst under exactly this topology.**

### 1.4 Honest limits of this evidence (read before relying on it)

This is a **strong suggestive contrast, not a controlled ablation.** The two streams differ in more than the
readout span:

1. **Different prompts.** The img prompt carries no title/transcript; the text prompt does. So "readout" and
   "input content" are confounded across the two streams.
2. **Different information content.** The text stream is the one carrying the project's measured passes
   (F45, F58); it has more to lose.
3. **Raw key space ≠ head space.** F72's *head-level* crater was −0.12; my raw-space concat crater is −0.056
   (HateMM). The head made it worse, not better — consistent, but the mapping is not 1:1.
4. **Dev, single draw, no seeds.** The raw-key-space vote has no training and therefore no seed variance,
   but it is one split and one operator.

What the contrast *does* establish, tightly: **a mask flip that leaves a mean-pooled stream intact (indeed
slightly improved) on both datasets is not a global "the weights are broken" event.** Whatever else is true,
H1 in its strong form ("bidirectional attention destroys these weights") is refuted by the img row. That is
enough to justify spending S1 before S2b.

### 1.5 The extraction sequence is vision-dominated (measured)

Processor run on CPU with dummy frames at the deployed `max_pixels=360*420=151200`, `num_frames=8`, over 60
real HateMM train prompts:

- **video tokens: 768, constant** (8 frames → 4 temporal groups × 192 merged tokens)
- **text tokens: median 162.5**
- **total: median 930.5 → vision is 82.5 % of the sequence**

Load-bearing for the corpus ruling (§3): a text-only MNTP corpus adapts the decoder to bidirectional
attention over **pure-text 512-token** sequences, while at extraction the decoder must handle **930-token
sequences that are 82.5 % vision tokens**. That is a real distribution gap in option (b), and it is the one
thing option (a′) gets right.

---

## 2. THE RECIPE (confirmed against the LLM2Vec paper and repo)

### 2.1 What stage 2 actually is

LLM2Vec = (1) enable bidirectional attention, (2) **MNTP**, (3) unsupervised SimCSE. Step 2 verbatim
(`experiments/run_mntp.py`):

- Choose a fraction `mlm_probability` of **text** positions; replace each with the mask token. With
  `mask_token_type: "blank"` the repo sets `tokenizer.mask_token = "_"` — Qwen has no `[MASK]`, so an
  underscore is used.
- Compute the standard **next-token shift**: `preds = preds[:, :-1]; labels = labels[:, 1:]`. So the hidden
  state at position **i−1** predicts the original token at masked position **i**. Loss is CE on masked
  positions only (`-100` elsewhere).
- The elegance: it **reuses the pretrained LM head unchanged** (no new head), and because position *i*
  holds `_` rather than the answer, bidirectional attention leaks nothing. The only new information
  position *i−1* gains is **the future beyond *i*** — i.e. precisely the bidirectional signal.

`Qwen2_5_VLForConditionalGeneration` has an untied `lm_head` (`tie_word_embeddings=False`), so this works
directly on our model.

### 2.2 Reference hyperparameters (McGill-NLP `train_configs/mntp/MetaLlama3.json`, fetched verbatim)

```json
{"dataset_name":"wikitext","dataset_config_name":"wikitext-103-raw-v1",
 "per_device_train_batch_size":32,"gradient_accumulation_steps":1,"max_seq_length":512,
 "mask_token_type":"blank","mlm_probability":0.2,"stop_after_n_steps":1000,
 "lora_r":16,"gradient_checkpointing":true,"torch_dtype":"bfloat16",
 "attn_implementation":"flash_attention_2"}
```

`initialize_peft` sets `lora_alpha = 2 * lora_r = 32`, `lora_dropout=0.05`, `bias="none"`, `task_type=None`,
targets `q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj`. `learning_rate` is unset → HF default
**5e-5**. Paper: masking **20 %** for S-LLaMA/LLaMA-2, **80 %** for Mistral (chosen by hyperparameter
search); **1000 steps**; cost **"90 minutes on a single 80GB A100" for 7B models**.

### 2.3 The remarkable coincidence: our task LoRA is already the same shape

`RA-HMD/LLAMA-FACTORY-Ver202512/my_configs/hatevideo/hatemm_qwen25vl_lora_sft.yaml`:

```yaml
lora_rank: 16
lora_alpha: 32
lora_target: q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj
freeze_vision_tower: true
freeze_multi_modal_projector: true
```

**Identical rank, alpha and target set to LLM2Vec's MNTP LoRA.** And our SFT already freezes the vision
tower and projector, so "MNTP scoped to the LLM trunk" is *the same scope the deployed adapter already has*.
No new scoping argument is needed.

### 2.4 Attention-enablement point — already built, already frozen, already reviewed

**Do not write new mask code.** `src/utils/bidir_patch.py` (frozen sha
`36cedbac365b2b13c945adbe3437efdc61d8be15ecc85878eb9614225abe367b`, F72 artifact A) is exactly the
enablement point, and it already handles the SDPA re-causalization trap (nulling the mask silently stays
causal via `modeling_qwen2_5_vl.py:989`; the fix is a forced non-None all-zeros 4D additive mask). It also
asserts `attn_implementation == "sdpa"` and defensively clears `is_causal` on all 28 decoder modules.

**Env note:** `flash_attn` is **ABSENT** from `HateVideo` (checked: transformers 4.49.0, peft 0.14.0, torch
2.6.0, datasets 3.2.0, accelerate 1.5.2; `llm2vec` also absent). The reference config's
`flash_attention_2` is therefore unavailable — which is *fortunate*, because our patch is SDPA-correct and
the assert prevents a silent flash fallback from re-causalizing training.

**Do not pip-install `llm2vec`.** It ships text-only `*BiForMNTP` classes; there is no
`Qwen2_5_VLBiForMNTP`, and bending its classes to a VL model is more work and more risk than a ~150-line
MNTP loop on top of our own already-frozen, already-codex-reviewed patch.

### 2.5 Does the vision path stay causal? — evidence says NO, and that is fine

Two separate objects, do not conflate:

- The **vision tower** (`model.visual`) is untouched and was never causal: block-diagonal `cu_seqlens`
  full-within-window attention (`modeling_qwen2_5_vl.py:265-269`). The patch binds only to `model.model`.
- The **decoder over vision tokens** — this is the real question, because `img_feats` is a decoder readout.

The brief's prior was that the img stream must stay causal. **§1.2 measures the opposite:** bidir img is
*better* on both datasets (+0.0093 / +0.0128). Keeping the streams on different topologies would also
require two forwards with different patches (doubling extraction cost) and would break the single-encoder
story. **Recommendation: apply bidir uniformly at the decoder.** This is an evidence-based reversal of the
brief's prior, and it is cheap to revisit — the img row is a $0 re-read on any new cache.

---

## 3. TRAINING-CORPUS RULING — SURFACED, NOT DECIDED

### 3.1 The veto text, verbatim from the ledger

`autoresearch/goal_mllm_plus3/state/directions_tried.json` → `banned_constraints`:

> `"TRAINING DATA = single-dataset train split ONLY (user veto 2026-07-14): no cross-dataset split mixing
> (trivial trick, not a contribution); conservatively also bans external unlabeled-pool training (C5)"`

**The conservative clause is already on the record and it already reaches option (b).** The earlier
`BIDIR_SURGERY_FORENSIC_RECON.md` §4 asserted MNTP is "own-split → LEGAL" — true for option (a), but it
never considered the wikitext option, so it does not license (b).

**Scope nuance the user may want to weigh:** per `C5_FORENSIC_RECON.md` §0, the "(C5)" the veto names is
*external unlabeled **video** + MLLM **pseudo-labels***, representation-training only. A generic unlabeled
**text** corpus used for **architecture adaptation with no labels of any kind** is a different object from
"an unlabeled pool of in-domain examples that enlarges the effective training set". Whether the veto's
*rationale* ("trivial trick, not a contribution") reaches it is the user's call, not mine.

### 3.2 Option (a): own-train transcripts only — LEGAL, but quantitatively inadequate ALONE

Measured with the Qwen2.5-VL tokenizer on `data/gt/*/train.jsonl`:

| dataset | n | transcript tokens (Σ) | median | mean | % of LLM2Vec budget | steps/epoch @32×512 | epochs to reach 1000 steps |
|---|---|---|---|---|---|---|---|
| **HateMM** | 744 | **239,382** | 170.5 | 321.8 | **1.46 %** | 14.6 | **~68** |
| **MHC_zh** | 579 | **52,351** | 76 | 90.4 | **0.32 %** | 3.2 | **~313** |

Reference budget = 1000 × 32 × 512 = **16,384,000 tokens**.

**Answer to the brief's question "is MNTP on ~100k tokens meaningful?" — no, not on its own.** HateMM's own
train split supplies **1.5 %** of the token budget LLM2Vec used; ZH supplies **0.3 %**. Reaching the
reference step count means ~68 (HateMM) / ~313 (ZH) passes over a few hundred transcripts with an r=16 LoRA
on all 7 projections — that is memorization of the memory bank's own text, not architecture adaptation, and
it would contaminate the retrieval geometry in a way the kNN vote is maximally sensitive to (the bank *is*
the train split). ZH is additionally hopeless on its own terms: median 76 tokens against `max_seq_length`
512, so most sequences would be padding.

I found no literature basis for MNTP at the 10⁵-token scale; every LLM2Vec-family result I could verify uses
wikitext-103 at ~10⁷ tokens.

### 3.3 Option (a′) — the variant the brief did not list, and the one I would actually run

**MNTP on our own train split in the DEPLOYED MULTIMODAL FORMAT**: the actual extraction sequences (8 frames
+ title + transcript + instruction), masking **only text positions**, vision tokens serving as frozen
context. Legality is identical to (a) — same data, same split, no labels — so **no user ruling is needed**.

Why it dominates (a): the *loss* budget is unchanged (~239k HateMM text tokens), but the *context
distribution* becomes the right one — 82.5 % vision, 930-token sequences (§1.5) — which is the regime the
extractor actually runs in and the one wikitext never touches. It is the only option that adapts the decoder
to bidirectional attention **over vision-token-heavy sequences**.

Its weakness is the same token budget as (a). It is a distribution fix, not a budget fix.

### 3.4 Option (b): generic unlabeled corpus (wikitext-103) — literature-faithful, **USER-GATED**

The published recipe. 1000 steps, ~90 min/A100 for 7B. Solves the budget problem completely and is the only
option with a direct published precedent to cite.

Two costs: **(i)** it needs the explicit veto relaxation in §3.1 — I am not deciding this; **(ii)** it has
the inverse distribution gap of (a′) — pure text at 512 tokens, never the 82.5 %-vision regime.

### 3.5 Option (c): published MNTP weights — **THEY EXIST, and this changes the ranking**

The brief asked me to check. **`McGill-NLP/LLM2Vec-Qwen25-7B-Instruct-mntp` exists on HuggingFace**, along
with 0.5B/1.5B/3B siblings and a Qwen3 family. Its `adapter_config.json`, fetched:

```
base_model_name_or_path = "Qwen/Qwen2.5-7B-Instruct"
r = 16 ; lora_alpha = 32 ; lora_dropout = 0.05 ; peft_type = "LORA" ; task_type = null
target_modules = [gate_proj, q_proj, down_proj, v_proj, k_proj, up_proj, o_proj]
```

**Shape compatibility verified against the local weights.** Qwen2.5-VL-7B's LLM trunk
(`.../models--Qwen--Qwen2.5-VL-7B-Instruct/snapshots/cc59489.../config.json`) vs Qwen2.5-7B-Instruct:

| field | Qwen2.5-VL-7B trunk | Qwen2.5-7B-Instruct | match |
|---|---|---|---|
| hidden_size | 3584 | 3584 | ✔ |
| num_hidden_layers | 28 | 28 | ✔ |
| num_attention_heads | 28 | 28 | ✔ |
| num_key_value_heads | 4 | 4 | ✔ |
| intermediate_size | 18944 | 18944 | ✔ |
| vocab_size | 152064 | 152064 | ✔ |
| rope_theta / rms_norm_eps | 1e6 / 1e-6 | 1e6 / 1e-6 | ✔ |

Every LoRA A/B matrix matches; PEFT matches `target_modules` by name suffix, which the VL decoder layers
carry. `max_position_embeddings` differs (128000 vs 32768) but LoRA does not touch positional encoding.
Download is ~80 MB (r=16 on 7 modules × 28 layers ≈ 40 M params in bf16).

**The caveat, stated plainly: this is a TRANSPLANT.** The adapter is a low-rank delta fitted to
Qwen2.5-7B-Instruct's weight point; Qwen2.5-VL's trunk was *initialised* from Qwen2.5 and then further
trained during VL pretraining, so the base has drifted. The delta may still carry the generic "use
bidirectional context" adaptation, or it may be noise at the new weight point. **That is an empirical
question answerable for ~0.5 GPU-h and zero training.**

**Veto analysis:** (c) involves **no training by us on any corpus**. It is an off-the-shelf pretrained
artifact — the same class of object as the Qwen2.5-VL base itself, or CLIP. A veto reading that blocked (c)
would also block the base encoder, so that reading is untenable. What (c) *does* need is the project's
standard **download gate** (the procedural gate that governs Molmo2-8B and CLAP), which at 80 MB is trivial
in cost but is still the user's to open.

### 3.6 Ranking and recommendation

| rank | option | legality | budget | distribution match | cost | recommendation |
|---|---|---|---|---|---|---|
| **1** | **(c) transplant** | off-the-shelf artifact; **download gate** only | n/a (0 training) | text-pretrained, VL-untouched | **0 train + 0.5 GPU-h/ds** | **run first** — near-free, decides a lot |
| **2** | **(b) wikitext** | **NEEDS USER RULING** (§3.1) | full 16.8 M ✔ | pure text ✘ | 1.5-2.5 GPU-h/ds | run if user opens the gate |
| **3** | **(a′) own-split multimodal** | **legal, no ruling** | 1.5 % ✘ | **82.5 %-vision ✔** | 2-3 GPU-h/ds | best *legal-without-ruling* training option |
| 4 | (a) own-split transcripts only | legal | 0.3-1.5 % ✘ | text-only ✘ | 1-2 GPU-h/ds | **not recommended alone** — dominated by (a′) at equal cost |

**My recommendation:** run **S1 (mean-pool readout, no training) before any of these**, because it can
dissolve the premise. Then **(c)** as the zero-training MNTP probe. Only if both under-deliver does the
corpus ruling become live — and at that point the strongest scientific package is **(b) then (a′)
sequentially** (generic corpus for architecture adaptation, own-split multimodal for distribution match),
which mirrors LLM2Vec's own "generic pretext, then task" ordering.

**(b) is the single item requiring a user decision.** Marked as such; not decided here.

---

## 4. WHERE MNTP PLUGS INTO OUR EXTRACTION

### 4.1 The deployed tap, exactly (verified in code, not from memory)

`src/utils/generate_VideoMLLM_embedding_lora_HF.py`: base Qwen2.5-VL loaded → `PeftModel.from_pretrained` →
**`merge_and_unload`** → SDPA (`:399` in the sibling frozen extractor) → one `torch.no_grad()` forward per
stream with `output_hidden_states=True` → **`out.hidden_states[-1]`** (final layer, `:361`) → span mean
(`:385` prefix / `:400` response) → **L2-norm** (`:404`) → banked as `img_feats` / `text_feats`, 3584-d.

**Correction to the brief:** the deployed text stream is **not** "mean-pooled under causal attention" over
the sequence. It is a mean over the trailing `<|im_start|>assistant\n` header — an **EOS-class last-token
readout**. The *image* stream is the mean-pooled one. This distinction is the whole finding of §1.

### 4.2 Arm construction (flag-controlled, single code path — F87 discipline)

The MNTP arm differs from the causal floor by exactly two things, and every arm must reach the head through
**identical extraction code**:

```
base Qwen2.5-VL-7B  →  PeftModel(task LoRA) → merge_and_unload      [unchanged, all arms]
                    →  [MNTP arm only] PeftModel(MNTP LoRA) → merge_and_unload
                    →  [MNTP arm only] bidir_patch.apply_bidir_mask(model)
                    →  process_split(...)                            [imported VERBATIM, all arms]
```

Follow the F72 artifact-A2 pattern exactly: `generate_VideoMLLM_embedding_bidir_HF.py` is a **thin fork**
that imports `read_gt` / `process_split` / `SPLIT_TO_OUTNAME` from the causal extractor and re-implements
only ~20 lines of `main()`. The MNTP runner should be the same fork plus one adapter load. The causal
extractor's sha stays byte-unchanged, so every provenance chain that pins it stays valid.

**Adapter-order question (must be pre-declared, not discovered):** stacking MNTP *on top of* the merged task
LoRA (above) is the cheap order and preserves the deployed encoder identity — but it asks a low-rank adapter
to repair both the base's and the task adapter's causal habits. The scientifically faithful LLM2Vec order is
**MNTP first, then task adaptation** (base → bidir+MNTP → merge → task LoRA-SFT under bidir → extract),
which costs an extra ~3.2 GPU-h of SFT per dataset and changes the encoder identity. **Recommendation:**
declare the cheap order for S2, and pre-commit that the faithful order is funded **only** if the cheap order
shows partial recovery (a signal that the direction is right but under-powered). Do not run both blind.

### 4.3 Same-path floor design (F87 KS-MOKA-0b generalized)

F87's lesson: a bf16 merge/unmerge path drifted enough (worst mean per-item cosine 0.99954879 < the 0.9999
bar) to manufacture a −0.0268 "method effect" that was pure path artifact. The MNTP arm adds **a second
merge**, so it is strictly more exposed.

**Binding rule for the prereg:** the floor for every MNTP arm must be produced by **the same number of
merges through the same code**. Concretely, the arm's control is not the banked causal cache but a
**same-path causal floor**: identical runner, identical double-`merge_and_unload`, with the MNTP adapter
replaced by a **zero-initialised LoRA of identical shape** (a true null-op adapter), bidir patch **off**.
If `KS-MNTP-0b` (below) shows the extra merge alone moves features by ≥ the 0.9999 cosine bar, that
same-path floor becomes **mandatory** and binding, exactly as `KS-MOKA-0b` made the unmerged floor binding.

Second F87 item carried: PEFT hooks belong on the **outer `PeftModel` wrapper**, not the inner module. Our
path calls `merge_and_unload` and then holds a plain `Qwen2_5_VLForConditionalGeneration`, so
`apply_bidir_mask(model)` binding to `model.model` is correct **only post-merge** — the F72 runner already
sequences it that way ("right after the LoRA merge and BEFORE any forward pass"). Preserve that order; a
pre-merge bind would attach to a `PeftModel` whose `.model` is the wrapper, not the decoder.

Third: **CPU-trained heads must be paired against CPU-trained floors.** ERRPAT measured CPU-vs-CUDA head
drift at −0.0031 final-epoch acc. Any CPU dev screen (§5) compares CPU-to-CPU, never CPU-to-banked-GPU.

---

## 5. CELL DESIGN, KILL BARS, COST

### 5.1 Datasets

- **HateMM — primary.** FN3 is the target cluster: 5 speech-rich, span-covered hate videos where the
  frames land inside the span and the text stream is simply wrong at top-20 purity 0.197. Ceiling
  **+0.0233**. This is the only cluster ERRPAT lists as "the text representation itself is the failure",
  and MNTP is the only remaining lever that changes it.
- **MHC_zh — secondary/hold.** Text-dominant, and it shows by far the largest bidir text crater (−0.2179),
  so it has the most dynamic range for the repair signal — but its own-split corpus is 0.32 % of budget
  (§3.2), so it is a poor host for options (a)/(a′).
- **MHC-EN — excluded.** Closed at all 3 supervision levels (F55).

**Ceiling honesty — a correction to the brief.** The brief cites "~+0.0465 if it also converts text-stream
failures inside FN1/FN2". Re-reading ERRPAT §5.1: **FN1 is defined by ≤25 transcript words** (median 6;
`hate_video_329` has **zero**; `hate_video_10`'s transcript is `"🎼 In and and.🎼And."`). A better text
*representation* cannot recover items with essentially **no text input** — that is an absence-of-signal
failure, not a representation failure, and it is what the CLAP gate targets. **The defensible MNTP ceiling
is FN3's +0.0233**, plus at most the 3 FN2 items if their dilution is representational rather than temporal
(ERRPAT calls FN2 "LOCKED — completely"). I would price the honest ceiling at **+0.0233, stretch +0.0372**,
not +0.0465.

### 5.2 Kill switches (KS analogues, ordered cheapest-first)

**`KS-MNTP-0a` — installation belt ($0 CPU, before any GPU).** Reuse F72's belts verbatim:
`bidir_patch.bidir_self_test()` PASS; the runtime line `[BIDIR] ... is_causal=False on 28 decoder attention
module(s)`; SDPA assert passes. **Fail ⇒ abort, no GPU.**

**`KS-MNTP-0b` — adapter-is-not-a-null-op ($0 CPU, immediately post-extraction).** Mean per-item
cosine(MNTP-bidir, plain-bidir) must be **< 0.9999** on all 6 split×stream cells. **If ≥ 0.9999 the MNTP
adapter silently failed to load** — the F87 failure shape — **⇒ ABORT before spending head time.**
Symmetrically, cosine(same-path-causal-floor, banked causal) **≥ 0.9999** on all 6 cells, else the same-path
floor becomes binding (§4.3).

**`KS-MNTP-1` — raw-key dev screen ($0 CPU, seconds; THE EARLIEST DISCRIMINATOR).** Re-run
`scripts/analysis/mntp_rawkey_devscreen.py` with the new cache tag. This is the answer to "the earliest
cheapest observable that distinguishes repair-working from crater-persisting", and §1.2 has already
**measured its dynamic range** — the crater is 3-15× the ±0.014 seed band, so the test is very
well-conditioned. Pre-set bars on the **text stream, dev, raw key space**:

| dataset | causal | bidir (crater) | 50 %-recovery bar | 25 %-recovery floor |
|---|---|---|---|---|
| HateMM | 0.8037 | 0.7570 | **≥ 0.7804** | 0.7687 |
| MHC_zh | 0.8462 | 0.6282 | **≥ 0.7372** | 0.6827 |

- **≥ 50 % recovery on ≥1 dataset ⇒ CONTINUE** to `KS-MNTP-2`.
- **< 25 % recovery on BOTH ⇒ KILL the arm** (crater persists; that MNTP variant is dead). No head time.
- Between ⇒ partial; continue only if the *sign* is consistent across both datasets.

**`KS-MNTP-2` — CPU head dev screen ($0, ~5 CPU-minutes).** ERRPAT's infra finding: the align head trains
and evaluates end-to-end in **52 s on 8 CPUs**. 3 seeds × 2 datasets ≈ 5 minutes, zero GPU, **dev only, zero
test-touch**, paired against a **CPU-trained same-path causal floor** (§4.3).
- **Continue iff mean paired Δdev-acc ≥ −0.014** vs the CPU causal floor on ≥1 dataset (crater closed into
  the banked seed-noise band).
- **Δdev-acc ≤ −0.05 ⇒ KILL** (crater materially persists).

**`KS-MNTP-3` — the goal gate (the one that matters).** Escalate to full ceremony **iff mean paired
Δdev-acc ≥ +0.020** over the same-path causal floor on ≥1 dataset.
**State this plainly in the prereg:** F81 already flagged that *"our +3-over-causal target sits above mere
recovery"*. **Recovering to the causal floor is a mechanism result, not a goal result.** An arm that lands
at Δ≈0 has refuted the Llama-pattern attribution and is publishable as such, but it does **not** advance the
+0.03-on-≥2-datasets clause and must not be dressed as if it did.

**`KS-MNTP-formal` — the verdict bar (unchanged house bar).** +0.030 acc **AND** +0.030 macro-F1, 3/3 seeds
positive, under **each** protocol independently, vs the same-path causal floor. Headline requires FORMAL
PASS on ≥2 datasets under a stated protocol.

### 5.3 Multiplicity and sequential-testing note

The family can contain up to **4 arms** (S1 mean-pool readout; S2a transplant; S2b-wikitext; S2b-own-split)
× 2 datasets. Discipline:

1. **Every arm is dev-gated.** Test is touched **once per dataset**, for the **single** promoted arm, at
   S3 only. Effective test multiplicity = 1 per dataset.
2. **The dev selection across ≤4 arms must be declared in the prereg** with the arm list frozen in advance —
   this is the forking-path exposure, and it is on dev, where the readout grid (F70) already established the
   precedent.
3. **Stages are sequential and each may terminate the family**, so this is a group-sequential design, not 4
   parallel tests. Report the stage at which each arm died.
4. **No S1×S2 grid.** Per `BIDIR_SURGERY_FORENSIC_RECON.md` §6, run the readout arm and the MNTP arms as
   single arms; pay for the interaction cell **only if both independently clear `KS-MNTP-2`**. Family
   multiplicity stays additive (arms + 1), never multiplicative.
5. n=3 seeds is too small for a bootstrap; report paired-t as an **effect-size descriptor only**, no
   significance claim (the standing house rule).

### 5.4 Cost ceiling

Calibrated against **measured** wall-clock, not estimates:

| item | evidence | cost |
|---|---|---|
| re-extraction, **both** datasets | job **13470 = 01:01:44** (1872 videos × 2 forwards) | **~1.0 GPU-h** |
| head 3-seed, both datasets, GPU | job **13471 = 00:06:43** | 0.11 GPU-h |
| head 3-seed, both datasets, **CPU** | ERRPAT: 52 s/seed | **$0** |
| smoke | job 13469 = 00:01:03 | ~0.02 GPU-h |
| MNTP (b) wikitext, 1000 steps | LLM2Vec: 90 min/A100 for 7B; SDPA not flash ⇒ discount | **1.5-2.5 GPU-h** |
| MNTP (a′) own-split multimodal | our LoRA-SFT, same LoRA shape, same 744 items × 3 ep = **3.2 GPU-h** (jobs 13237/13238) | **2-3 GPU-h/ds** |

**Stage totals:**

- **S1** (mean-pool readout, no training): **~1.0 GPU-h** both datasets + $0 CPU gates.
- **S2a** (transplant, no training): **~1.0 GPU-h** both datasets + 80 MB download + $0 CPU gates.
- **S2b** (we train): **+2-4 GPU-h/dataset** + ~1.0 GPU-h re-extraction.
- **S3** (ceremony, only on a MOVE): **~0.2 GPU-h**.

**Recommended cost ceiling: 2.5 GPU-h for S1+S2a combined**, which buys the answer to "is the bidirectional
axis alive at all, and is it a readout problem or a weights problem". **Do not authorise S2b's 4-8 GPU-h
until S1 and S2a have reported.** Total family ceiling if everything escalates: **~12 GPU-h**, against an
honest FN3 ceiling of +0.0233 and an F81 prior of 8-12 %.

### 5.5 Risks, ranked

1. **The crater is over-attributed (the main risk to F72's own conclusion).** §1 shows the mean-pooled
   stream is fine. If S1 closes most of the gap, "Llama-pattern, MNTP-motivated" was a **readout artifact
   wearing a weights-failure costume**, and the DEGRADE-branch rationale that routed this to the user
   partially dissolves. This is a *good* outcome scientifically and an important erratum — but it means the
   MNTP spend was justified on a shakier premise than the record states.
2. **Recovery ≠ goal.** The most likely non-null outcome is landing near the causal floor. `KS-MNTP-3`
   exists precisely to stop that being mis-sold. Base rate against conversion in this project is brutal:
   ~9 "new signal, no conversion" data points.
3. **Transplant is noise (S2a).** Detected by `KS-MNTP-1` at $0. Cheap to find out.
4. **Token-budget starvation / memorization (a, a′).** 1.5 % of reference budget over the memory bank's own
   text; the kNN vote is maximally sensitive to bank contamination. Mitigate with a held-out MNTP eval loss
   on the dataset's own **val** transcripts (free, in-training) and early stopping on it.
5. **Distribution gap (b).** Pure-text 512-token training vs 930-token, 82.5 %-vision inference (§1.5).
6. **Double-merge path drift.** `KS-MNTP-0b` + the same-path floor (§4.3). This is the F87 failure shape and
   the MNTP arm is *more* exposed than MokA was.
7. **Attribution collapse if S1 and S2 are bundled.** Changing the readout and adding MNTP in one arm makes
   the result uninterpretable. Keep them separate arms; this is non-negotiable.

---

## 6. STAGED PLAN (each stage gated, each gate cheap)

**S0 — $0, DONE (this document).** Recon + the §1 measurement + `KS-MNTP-0a` self-test is already banked
from F72.

**S1 — bidir + LLM2Vec mean-pool readout, NO training. ~1.0 GPU-h, no user ruling.**
Re-extract both datasets with the bidir patch **and** a mean-over-all-tokens text readout (plus the existing
prefix-mean img readout, unchanged). Gates: `KS-MNTP-0a` → `KS-MNTP-1` → `KS-MNTP-2`.
*Rationale:* the cheapest experiment that discriminates H1 (weights) from H2 (readout), and it needs no
corpus ruling, no download, and no new training code. **This is the recommended first spend.**
*Kill:* < 25 % text-stream recovery on both datasets ⇒ H2 refuted, H1 stands, proceed to S2a with the
readout question closed.

**S2a — published MNTP transplant, NO training. ~1.0 GPU-h + 80 MB download gate.**
Load `McGill-NLP/LLM2Vec-Qwen25-7B-Instruct-mntp` onto the merged VL trunk, bidir on, re-extract, same
gates. Runs on the **best readout S1 identified** (or the deployed one if S1 killed the readout hypothesis).
*Kill:* `KS-MNTP-0b` null-op ⇒ abort; `KS-MNTP-1` < 25 % ⇒ transplant dead.

**S2b — we train MNTP. 2-4 GPU-h/dataset. FUNDED ONLY IF (i) S1+S2a show partial recovery or a MOVE, AND
(ii) the corpus ruling is in hand.**
Corpus per §3.6: (b) if the user relaxes the veto, else (a′). Same gates, plus the held-out MNTP eval-loss
early stop.
*Kill:* `KS-MNTP-2` Δdev ≤ −0.05 ⇒ dead.

**S3 — full ceremony. Only on `KS-MNTP-3` (Δdev ≥ +0.020 on ≥1 dataset).**
prereg → independent 0-context review → freeze-hash → local SLURM 3-seed, both protocols, **single
test-touch per dataset**, paired against the same-path causal floor. ~0.2 GPU-h.

**Nothing in this document authorises GPU.** S1 requires the team lead's go; S2a additionally requires the
download gate; S2b additionally requires the §3.4 corpus ruling.

---

## 7. THE SINGLE MOST LOAD-BEARING UNCERTAINTY

**Whether F72's crater is a weights failure or a readout failure — because the two prescribe different
spends, and the project's record currently asserts the first without having tested the second.**

The evidence I could gather at $0 leans toward "substantially readout": under an identical mask flip with
identical weights, the mean-pooled stream is *better* (+0.0093 / +0.0128) while the last-token stream
collapses (−0.0467 / −0.2179), and LLM2Vec's own ablation independently names EOS pooling as the worst
pooling under bidirectional attention while every published LLM2Vec result uses mean pooling.

But the contrast is **not** clean (§1.4): the two streams differ in prompt and in information content as
well as in readout span, so I cannot separate "readout span" from "what that stream encodes" on banked data
alone. **Only S1 separates them** — same prompt, same weights, same topology, only the pooling span
changed — and it costs ~1.0 GPU-h with no ruling and no training.

If S1 closes most of the gap, the MNTP cell shrinks from "the designed repair for a catastrophic failure" to
"an optional enhancement on a working bidirectional axis", and the corpus ruling may never need to be made.
If S1 does not, F72's attribution is vindicated, H1 stands, and the corpus ruling becomes the live decision
with the transplant probe (S2a) as the cheapest next test.

**Either way, S1 is the next thing to run, and it is cheaper than the thing the gate opened.**

---

## 8. PROVENANCE

**Primary records read:** `refine-logs/BIDIR_STAGE1_VERDICT_REVIEW.md`,
`BIDIR_SURGERY_FORENSIC_RECON.md`, `BIDIR_STAGE1_PREREG.md`, `ERRPAT_HateMM_2026-07-26.md`,
`MOKA_VERDICT_REVIEW.md`, `LITSWEEP5_COMPLETENESS.md` (F81 §116), `C5_FORENSIC_RECON.md` §0,
`READOUT_PREREG.md`, `REPRO_SURVEY_2025.md`;
`autoresearch/goal_mllm_plus3/state/directions_tried.json` (`banned_constraints`).

**Code read (ground truth, not memory):** `src/utils/generate_VideoMLLM_embedding_lora_HF.py:320-473`
(the tap), `src/utils/generate_VideoMLLM_embedding_bidir_HF.py` (thin-fork pattern),
`src/utils/bidir_patch.py` (frozen artifact A), `src/utils/metrics.py:229-320` (the vote operator),
`RA-HMD/LLAMA-FACTORY-Ver202512/my_configs/hatevideo/hatemm_qwen25vl_lora_sft.yaml`,
`scripts/slurm/lora_sft.sbatch`.

**External (WebFetch, 2026-07-27):** LLM2Vec repo `README.md`, `experiments/run_mntp.py`,
`train_configs/mntp/MetaLlama3.json`; ar5iv HTML of arXiv 2404.05961; HuggingFace model listings and
`McGill-NLP/LLM2Vec-Qwen25-7B-Instruct-mntp/adapter_config.json`.

**Measured by this recon (all CPU, all $0):** the §1.2 raw-key dev screen + feature drift
(`scripts/analysis/mntp_rawkey_devscreen{.py,_OUT.json}`); §3.2 transcript token counts (Qwen2.5-VL
tokenizer over `data/gt/*/train.jsonl`); §1.5 vision/text token split (Qwen2.5-VL processor, dummy frames,
60 real prompts); §2.3/§3.5 config comparisons; §5.4 wall-clock from `sacct` (jobs 13469/13470/13471,
12142/12143/13233/13237/13238/13244); env versions in `HateVideo`.

**Required statements:** ZERO GPU / SLURM / Modal / downloads / training / test-touch spent. **No held-out
test file was opened or produced** — every metric above is train-memory + DEV. No `state/`, prereg, config,
`research-wiki/`, or frozen artifact mutated. No `src/` edit. Committed on `main`, not pushed.
