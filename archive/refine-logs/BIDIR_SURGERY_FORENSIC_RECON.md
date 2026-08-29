# BIDIRECTIONAL-ATTENTION SURGERY — forensic recon (F68-P3)

**Author:** forensic-recon agent (CPU-only; ZERO GPU/SLURM/Modal; `autoresearch/.../state/` untouched; no prereg; no job submission).
**Date:** 2026-07-25 NZST.
**Cell:** F68 ledger candidate **P3** — remove the causal mask from the Qwen2.5-VL **language decoder** when harvesting embeddings (LLM2Vec / NV-Embed recipe), optionally + MNTP adaptation. Source: `refine-logs/LITSURVEY_MLLM_EMBEDDING.md` §B3 (b807722). Motivating diagnosis: **F35** identified `is_causal=True` as the structural reason Qwen frame-group vectors are cumulative causal-prefix summaries — but no run ever removed the mask. This cell IS that removal.

---

## 0. VERDICT UP FRONT

**GO-IF** — GO to a cheap **STAGE-1 training-free mask-flip screen**, conditioned on three gates, none of which cost GPU:

1. **Codex-review of the ~10-line mask monkey-patch** before any GPU (project doctrine: code touching attention masks is codex-gated; `codex-code-review` skill).
2. **A one-line D7 user sub-ruling** acknowledging this is an *architecture-level third structural object* (attention topology), not a generic encoder swap — legal to *measure* under the **F65 precedent** (F65 already refuted F51's "two-object closure" wording by measuring the vision-reach third object), and worth the ~0.5 GPU-h even at a ~10-15% perf prior *because the D7-novelty payoff is the highest in the entire litsurvey*.
3. **Sequencing**: run bidir stage-1 as its OWN single arm on the BASELINE (prefix-mean) readout — NOT a combinatorial bidir × readout grid with the parallel readout-recon cell.

**NO-GO to skipping straight to Stage-2 MNTP.** LLM2Vec's Llama-precedent argues MNTP *may* be required, but the cheap-kill discipline says spend $0.5 GPU-h (stage-1) before $2-4 GPU-h (stage-2 MNTP). Stage-1's *outcome shape* (flat vs degrade vs move) is itself the signal that funds or kills stage-2 — see §5.

**Full ceremony (prereg → 0-context review → freeze-hash → local SLURM) triggers only if Stage-1 moves the dev screen.** Nothing here authorizes GPU.

---

## 1. MECHANICS — the load-bearing part (verified against the installed package)

**Installed package:** `transformers 4.49.0`, env `HateVideo` (python 3.11):
`/data/jehc223/miniconda3/envs/HateVideo/lib/python3.11/site-packages/transformers/models/qwen2_5_vl/modeling_qwen2_5_vl.py` (2112 lines). (Env `ExMRD` also carries 4.49.0 — identical code; extraction runs under `HateVideo`.)

### 1.1 Is there a config flag? NO.
`Qwen2_5_VLAttention.__init__` **hard-codes** `self.is_causal = True` (line **723**) — it is NOT read from config. There is no `is_causal` / `bidirectional` field in `Qwen2_5_VLConfig`, and all three attention implementations (`eager` / `flash_attention_2` / `sdpa`, registered at lines **1008-1012**) default to causal. **A targeted monkey-patch is mandatory; no flag exists.**

### 1.2 Our extraction path = SDPA. Where causality actually comes from.
Our canonical extractor `src/utils/generate_VideoMLLM_embedding_HF.py` loads with `attn_implementation="sdpa"` (line **399**), runs one frozen forward with `output_hidden_states=True` (line **278**), reads the **last layer** (`out.hidden_states[-1]`, line **279**), and **mean-pools the prefix span then L2-norms** (lines **303, 322**). The pooled vector is the banked `img_feats`. (`s2s_extract.py:534-535` confirms the same SDPA load; `sav_f0_common.py:20` documents `attn_implementation="sdpa"`.)

For **SDPA**, causality is enforced in TWO places, and **you must defeat both**:
- **(a)** the 4D additive mask returned by `Qwen2_5_VLModel._update_causal_mask` (lines **1244-1325**), consumed at `Qwen2_5_VLSdpaAttention.forward` as `attn_mask=causal_mask` (line **995**);
- **(b)** the fallback flag `is_causal = True if causal_mask is None and q_len > 1 else False` (line **989**).

**Critical gotcha:** for a single unpadded sample, `_update_causal_mask` returns **None** (via `AttentionMaskConverter._ignore_causal_mask_sdpa`, line **1278**), whereupon line 989 sets `is_causal=True` and **SDPA applies causal masking internally**. So **simply nulling the mask does NOT give bidirectional attention** — it silently stays causal. You must **force a non-None all-zeros 4D additive mask**, which (i) makes line 989 evaluate `is_causal=False` and (ii) adds a zero bias = attend-everywhere = bidirectional.

### 1.3 The patch (≤10 lines, verified sufficient for SDPA + eager)
Override `_update_causal_mask` on the **decoder instance** (`model.model`, the `Qwen2_5_VLModel`; vision is `model.visual`, lines **1519-1520**):

```python
import types, torch
def _bidir_mask(self, attention_mask, input_tensor, cache_position, past_key_values, output_attentions):
    dtype = input_tensor.dtype
    bsz, seq_len = input_tensor.shape[0], input_tensor.shape[1]
    mask = torch.zeros((bsz, 1, seq_len, seq_len), dtype=dtype, device=input_tensor.device)
    if attention_mask is not None and attention_mask.dim() == 2:   # fold padding (bsz=1 -> no-op)
        pad = (1.0 - attention_mask[:, None, None, :].to(dtype)) * torch.finfo(dtype).min
        mask = mask + pad
    return mask
model.model._update_causal_mask = types.MethodType(_bidir_mask, model.model)
```

**Why it is correct:**
- **SDPA** (line 989/995): returned mask is non-None → `is_causal=False`; `attn_mask` = all-zeros additive → no masking → **bidirectional**. ✓
- **Eager** (`Qwen2_5_VLAttention.forward`, lines **775-777**): `attn_weights + causal_mask` where `causal_mask` is all-zeros → no masking → bidirectional. ✓ (Eager only runs if `output_attentions=True`; not our path, but the patch covers it.)
- **Padding**: bsz=1 unpadded extraction → the padding fold is a no-op; kept for correctness if batched.
- It completely **replaces** `_update_causal_mask`, so `_ignore_causal_mask_sdpa` / `_unmask_unattended` (lines 1278, 1323) never run — no interaction hazard.

### 1.4 Vision tower + merger — UNTOUCHED and already bidirectional (verified)
The patch binds only to `model.model` (LLM decoder). The vision attention builds a **block-diagonal** mask from `cu_seqlens` (`Qwen2_5_VLVisionAttention.forward`, lines **265-269**: full attention *within* each window block, never causal); `Qwen2_5_VLVisionSdpaAttention` (line 284) is the same topology. So the vision tower and merger are already bidirectional-within-window and are **not** modified — exactly as F35 noted ("only vision encoder block-diagonal"). ✓

### 1.5 Flash-attention-2 caveat (off our path, but stated)
`Qwen2_5_VLFlashAttention2.forward` passes `is_causal=self.is_causal` (line **904**) and uses the 4D mask **only for padding**, not causality — so **the `_update_causal_mask` patch does NOT flip flash**. If flash were ever used you would ALSO need:
```python
for m in model.model.modules():
    if isinstance(m, Qwen2_5_VLAttention):
        m.is_causal = False
```
**Our extractor uses SDPA (line 399), so flash is off-path** and the single mask patch suffices. Recommendation: pin `attn_implementation="sdpa"` for the bidir arm and assert it at load, to avoid a silent flash fallback re-causalizing.

**Patch size verdict: one function, ~9 lines, one bind line. Small, inference-only, no weights changed at Stage-1.** Load-bearing verified facts: (i) no config flag; (ii) SDPA needs the *forced non-None all-zeros* mask (nulling is a trap); (iii) vision untouched; (iv) flash needs the extra `is_causal=False` loop it will not otherwise get.

---

## 2. STAGE-1 — training-free mask flip (the cheap kill)

Re-extract embeddings with the mask flipped, **nothing retrained** (encoder frozen, LoRA adapter still loaded — the adapter sits on the attention *linear* layers and is orthogonal to the mask topology).

- **Datasets:** ZH (LoRA adapter, **primary** — targets the F45 78-dev val-sel selection tax, the only live perf target) + HateMM (curric adapter, **hold**).
- **Cost:** ~0.5 GPU-h/dataset (one frozen forward per video, same as any re-extraction; F67/S2S precedent).
- **$0 dev screen (pin this one):** the **encoder-swap dev-screen machinery** — retrain the frozen-feature head on `-bidir` **train** features, evaluate dev-acc, compare to the head on causal (banked) features. Train+dev only, **zero test-touch**. This is the right screen because bidir *replaces* the representation content (it does not add a channel), so the apples-to-apples question is "does the head on bidir features beat the head on causal features on dev?" — not the conditional-info gate (`c3_fusion_probe.py` / `ctf_g0cond_gate.py`), which measures a NEW channel's info *over* Z_best and is the wrong shape here. (Machinery all present: `scripts/analysis/{lp_gate.py, laud_g0cond_gate.py, ctf_g0cond_gate.py, c3_fusion_probe.py}` — reuse the head-retrain path, not the add-a-channel gate.)
- **Decision:** if bidir dev degrades or ties on **ZH** → see §5 outcome logic (tie = cheap kill; degrade = the Llama-pattern signal for a *conditional* stage-2). If bidir dev *moves* on ZH → stage-2 / full ceremony.

---

## 3. STAGE-2 — MNTP adaptation (conditional)

Masked-next-token-prediction LoRA on the **OWN train split** transcripts+frames (self-supervised — no labels), then re-extract under the flipped mask.

- **Single-dataset veto check:** the veto is `TRAINING DATA = single-dataset own train split ONLY, no cross-dataset mixing`. MNTP is **self-supervised pretraining on the dataset's own train split** — no external corpus, no cross-dataset pool, no gold labels. **It is own-split → LEGAL.** State this explicitly in any prereg: MNTP consumes only the same own-train videos already used for LoRA-SFT, under a masked-token objective.
- **Cost estimate:** ~2-4 GPU-h/dataset (short MNTP LoRA + re-extraction), same order as the existing LoRA-SFT.
- **Funding rule:** conditional on Stage-1 outcome (§5). Do NOT pre-declare skip-to-stage-2. The LLM2Vec precedent (Mistral works training-free; Llama needs MNTP) means Qwen2.5-VL's behavior IS the experiment — and Stage-1's outcome *shape* is exactly the discriminator that tells us whether MNTP is the designed repair or a dead spend.

---

## 4. BAN CHECK (quoted scopes)

- **NOT F24 / encoder-swap.** F24: *"encoder-class levers do not satisfy novelty clause (D7 resolved-negative)."* Stage-1 uses the **same encoder, same frozen weights**, changing only attention **topology at inference** — it is not a weight-space lever at all. (Stage-2 MNTP adapts weights, but under a new *pretext objective on a new topology* — see F51 below, not a generic encoder LoRA.)
- **NOT F51's two adapted objects.** F51: *"adaptation has exactly two adapted objects — encoder (generic LoRA…) and joint encoder+decision (retrieval-loss-into-LoRA = P9b)… No third object exists."* Mask topology is **neither** generic-LoRA-weight-space **nor** joint-head-training — it is a **third structural object** (attention topology). **Precedent is decisive and favorable:** F65 already *"refut[ed] the F51/GAP-5b wording at mechanism level"* by *measuring* the vision-reach third object (image MOVED, then K-V2 tied). Bidir is a fourth such structural object; the F51 "closure" is a wording claim already breached, so measuring bidir is in-doctrine.
- **NOT P9b.** P9b = the **RGCL retrieval loss (triplet+BCE on the fused head key) coupled JOINTLY into the encoder LoRA** → redistribution law (head↔memory ±1.8pt, net 0). Stage-1 has **no loss, no head coupling, no training**. Stage-2 MNTP is **masked-token self-supervision, decoupled from the head, no retrieval loss** → not P9b.
- **NOT F35-F39 (don't-pool family).** Those **pool/match/supervise OVER** the cumulative-causal vectors: F37 (S2S set-pooling KILL), F39 (CTF supervised temporal-pool, $0 G0-cond KILL — zero conditional info over the pooled key). Bidir **ATTACKS the F35 mechanism** — it removes the `is_causal=True` that *creates* the cumulative-causal vectors — rather than pooling over their output. F35 is the diagnosis (*"structural (LLM is_causal=True, no bidirectional vision unmasking in transformers 4.49.0)"*); bidir is the mechanistic repair F35 sets up. Distinct family.

---

## 5. KILL BARS + priors + D7 story

### 5.1 Stage-1 outcome logic (pre-declared shape; bars finalized at prereg)
Compare `-bidir` vs causal head-on-frozen-features **dev-acc**, ZH primary (F45 target), HateMM hold:
- **FLAT** (|Δdev| within the noise band, no consistent sign across seeds) → **KILL cheap.** This is the Law-I outcome (F37/F39): the pooled prefix-mean already integrates the whole sequence, so a topology change that carries no *new convertible* dev signal is dead. No stage-2, no GPU beyond the $0.5 screen.
- **DEGRADE** (bidir dev clearly below causal) → **Llama-pattern signal**, NOT a full kill: the causally-trained weights break under bidir attention (distribution shift). This is precisely when MNTP is the *designed repair*. Fund **one** stage-2 MNTP shot **only if the prereg pre-commits** the LLM2Vec-Llama-precedent argument; otherwise KILL.
- **MOVE** (bidir dev > causal on ZH by the pre-set margin) → escalate to full ceremony (prereg → 0-context review → freeze-hash → local SLURM 3-seed on both protocols, test untouched until frozen).

### 5.2 Collision naming
Cache tag **`Qwen2.5-VL-7B-Instruct_HF-bidir`** (matches the existing `-16f` suffix convention seen in `logging/.../RAC_video_fb16/..._HF-16f`; base tag is `Qwen2.5-VL-7B-Instruct_HF`, extractor arg `--out_model_tag`, filename `{outname}_{tag}.pt` at `generate_VideoMLLM_embedding_HF.py:437`). Never overwrites the banked causal caches.

### 5.3 Honest priors
- **Cost:** MED (stage-1 ~0.5 GPU-h/dataset training-free; stage-2 ~2-4 GPU-h/dataset conditional).
- **Perf prior:** **~10-15%** for ≥+1pt on ≥1 dataset. Discounted by **Law-I** (F37/F39: pooling is effectively lossless over the causal reps; the *pooled* readout already sees the whole sequence via the mean). The one thing that keeps the prior above zero: bidir changes the **content** of each token vector (every token computed with full past+future context), whereas F37/F39 only showed that *re-pooling/matching the same causal vectors* adds nothing — bidir is the sole lever that changes the vectors themselves, not how they are pooled. Realistic target = **ZH val-sel hardening** (EN is label-limited at 5 proven levels; HateMM already passes F53).
- **D7-novelty:** **HIGHEST in the litsurvey.**

### 5.4 D7 story value (stated plainly)
This is an **architecture-level, named, cited mechanism** (bidirectional-attention surgery; LLM2Vec 2404.05961 / NV-Embed 2405.17428) **aimed directly at our OWN diagnosed pathology** (F35's cumulative causal-prefix representations). That triple — (i) named/cited in the SOTA MLLM-embedding literature, (ii) an attention-*topology* change rather than a weight swap, (iii) motivated by our own diagnostic finding rather than borrowed blind — is the **strongest "not a generic encoder swap" argument available in the project.** Even at a low perf prior, the D7-integration-story value is why this cell is worth the $0.5 GPU-h Stage-1 screen: a *measured* bidir result (pass OR clean kill) converts F35 from a passive diagnosis into an actively-tested named mechanism in the paper's method-novelty narrative.

---

## 6. INTERACTION with the readout cell + sequencing

Flipping the mask changes what **every layer and token** contains, so bidir × readout (the parallel readout-recon cell: layer-index / one-word / echo arms) **can interact** — a naive grid would be (readout arms) × (causal, bidir) = combinatorial multiplicity.

**Recommended sequencing (avoid the grid):**
1. **Readout grid on CAUSAL first.** The readout cell is pure-read (which layer / which token / which prompt) over the existing causal representation — cheapest, no topology change, and it establishes the baseline readout.
2. **Bidir Stage-1 as its OWN single arm on the BASELINE (prefix-mean) readout.** ONE pre-registered comparison per dataset (`bidir-prefix-mean` vs `causal-prefix-mean`) × 2 protocols. Do **not** multiply by readout arms.
3. **Interaction cell only if BOTH independently show life** — i.e. only pay for bidir × best-readout after each has separately passed its own dev screen. This keeps the family multiplicity at (readout arms) + 1 (bidir), not ×.

**Multiplicity plan:** bidir contributes exactly **1 primary dev comparison per dataset** (ZH primary + HateMM hold), dev-gated before any test-touch; test is touched only under §5.1-MOVE full ceremony. No bidir×readout combinatorics unless both survive alone.

**Note for the readout-recon agent:** bidir requires a NEW forward pass (different attention), so bidir arms are NOT composable-for-free with cached-all-layer readout arms — keep them as separate single arms, not a joint cache sweep.

---

## 7. VERDICT + STAGE PLAN

**GO-IF.** Proceed to **Stage-1 training-free mask-flip** on ZH (primary) + HateMM (hold), gated on: (1) codex-review of the ~10-line mask patch; (2) a one-line D7 user sub-ruling (third-object, F65 precedent, D7-payoff-worth-$0.5); (3) run as its own baseline-readout arm, sequenced after the causal readout grid, no bidir×readout grid.

- **Stage-1** (~0.5 GPU-h/dataset, training-free): flip mask via §1.3 patch → re-extract `-bidir` caches (train+dev, no test) → encoder-swap head dev screen → outcome logic §5.1.
- **Stage-2** (~2-4 GPU-h/dataset, conditional, self-supervised own-split MNTP, veto-legal): funded ONLY on Stage-1 MOVE (confirm/convert) or DEGRADE-with-precommit (Llama-repair); KILLED on Stage-1 FLAT.
- **NO-GO** to skipping straight to Stage-2. **NO GPU authorized here** — full ceremony triggers only on a Stage-1 dev MOVE.

Exact mask-flip mechanics verified: **no config flag exists** (`self.is_causal=True` hard-coded, `modeling_qwen2_5_vl.py:723`); **SDPA needs the forced non-None all-zeros 4D mask** (nulling silently stays causal via line 989 — the trap); patch is **one ~9-line function bound to `model.model._update_causal_mask`**, covers SDPA (our path) + eager; **vision tower/merger untouched and already bidirectional** (block-diagonal `cu_seqlens`, lines 265-269); **flash-attention would additionally need `is_causal=False` on every decoder attention module** (line 904) but is off our SDPA path. Stage-1 cost ~0.5 GPU-h/dataset. Sequence bidir as its own baseline-readout arm AFTER the causal readout grid.
