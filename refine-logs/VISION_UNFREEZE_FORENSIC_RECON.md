# VISION-SIDE ADAPTATION — FORENSIC RECON (GAP-5b / red-team cell C1)

**Author:** vision-unfreeze forensic-recon subagent (CPU-only; **ZERO GPU / SLURM / Modal / download /
test-touch**; no `state/` mutation; NO prereg authored, NO job submitted). **Date:** 2026-07-20 NZST.
**Mission:** zero-GPU GO/NO-GO + execution skeleton for the one lever the whole LoRA-SFT family never
pulled — **LoRA reaching the Qwen2.5-VL vision tower** (and/or projector), the untested cell that
targets EN's upstream image collapse and refutes the *wording* of F51's two-object closure.
**Reads (verified this recon):** `REDTEAM_UNTESTED_CELLS.md` (cell C1 + §0 CPU-probe appendix),
`REDTEAM_BAN_SCOPE_AUDIT.md` (GAP-5b), `HATEMM_LORA_STREAM_DECOMP.md` (F58),
`LORA_HATEMM_PREREG.md` (the generic recipe this forks), `CAND2_CURRICULUM_PREREG.md` §1.2 (config-diff
house style). On-disk code: `RA-HMD/LLAMA-FACTORY-Ver202512/src/llamafactory/{model,hparams}`.

---

## VERDICT (one line)

**GO** — run **LoRA-on-vision+LLM** (`freeze_vision_tower: false` + `lora_target: all`) on **MHC-EN
(the refutation target — the whole ballgame) + HateMM (mechanism-aligned hold/upside)**, ZH an optional
third; **~9–12 GPU-h** for the two primary legs (~14–18 GPU-h with ZH). This is a **PERFORMANCE lever +
an F51-wording refutation, NOT a novelty escape** — same D7 collision as generic LoRA (F0.3). No cheaper
$0 pre-GPU kill exists: the red-team upstream-collapse probe already *is* the $0 motivating evidence, and
the vision-adapted features cannot exist until the SFT runs — the cell is irreducibly a training experiment.

---

## 1. MECHANICS — exactly how LoRA reaches the Qwen2.5-VL vision path (verified in the on-disk code)

### 1.1 The composite-model registration (the load-bearing table)

`src/llamafactory/model/model_utils/visual.py:344-352` registers, for `model_type == "qwen2_5_vl"`:

| key | value |
|---|---|
| `vision_model_keys` | `["visual.patch_embed", "visual.blocks"]` (the ViT tower) |
| `projector_key` | `"visual.merger"` (the multimodal projector / patch-merger MLP) |
| `lora_conflict_keys` | `["patch_embed"]` (**always** excluded from LoRA — the Conv3d patch embed) |

Confirmed ViT internal module names (transformers 4.49.0 `modeling_qwen2_5_vl`): each of the **32**
`visual.blocks.N` has `attn.qkv` (fused, `nn.Linear(1280→3840)`), `attn.proj` (`1280→1280`), and a
**gated MLP** `mlp.{gate_proj,up_proj,down_proj}` (`1280↔3420`). `visual.patch_embed.proj` is a
**Conv3d** (not a Linear, and `patch_embed` ∈ `lora_conflict_keys`) ⇒ **never** LoRA-able.

### 1.2 `freeze_vision_tower: false` semantics under LoRA (NOT full-FT)

Two distinct machines, both verified:
- **Full/Freeze tuning** (`_setup_full_tuning`/`_setup_freeze_tuning`, `adapter.py:40-140`): `freeze_vision_tower:
  false` would make the **entire ~675M-param ViT trainable** (`get_forbidden_modules` drops the vision keys
  from the frozen set, `visual.py:159-179`). **This is the overfit-doomed variant — do NOT use it.**
- **LoRA tuning** (this cell): `freeze_vision_tower: false` does **not** full-FT anything. It only widens the
  LoRA target search: `find_all_linear_modules(model, freeze_vision_tower=False)` (`misc.py:28-52`) stops
  adding `["visual.patch_embed","visual.blocks"]` to the forbidden set (guard at `misc.py:40-41`), so the ViT
  block Linears become LoRA-eligible. **"Unfreeze the vision tower" here means "let rank-16 LoRA reach the
  ViT," ~11M params — not a 675M full-FT.** This distinction is the entire overfit case (§ below).

### 1.3 Can `lora_target` name visual blocks? YES — but `all` is the clean choice

- `lora_target: all` (`adapter.py:215-216`) → `find_all_linear_modules` collects all Linear **leaf** names
  except `lm_head` and `projector_key` (`visual.merger`, forbidden **unconditionally** at `misc.py:37-38`).
  With vision unfrozen it collects LLM `{q,k,v,o,gate,up,down}_proj` **plus** ViT `{qkv, proj}` (attn) and
  ViT `{gate,up,down}_proj` (mlp). `patch_target_modules` (`visual.py:182-197`) then expands to full paths,
  dropping `patch_embed` (lora_conflict) and `visual.merger` (projector). **Net LoRA footprint: all 28 LLM
  decoder layers (identical to generic) + all 32 ViT blocks (attn qkv+proj, mlp gate+up+down).**
- **Naming the 7 LLM modules + unfreezing vision is a TRAP** (rejected): the substring match in
  `patch_target_modules` (`"target in name"`) would hit the ViT `mlp.{gate,up,down}_proj` (they share names
  with the LLM) but **miss** ViT `attn.{qkv,proj}` (names not in the list) ⇒ a lopsided "vision-MLP-only"
  adaptation. `all` is the only clean, complete ViT-inclusive target.
- **The vision arm is therefore a CLEAN SUPERSET of generic:** under `lora_target: all` the LLM coverage is
  byte-identical to the generic 7-named list (verified: the on-disk generic adapter targets exactly
  `q/k/v/o/gate/up/down_proj` × 28 layers, **zero** `visual.*`), so **the only delta vs the banked generic
  arm is the ViT LoRA.** K-V2 (§4) then isolates the vision contribution exactly.

### 1.4 DoRA / rsLoRA / PiSSA / LoRA+ — all present and wired (red-team claim verified)

`finetuning_args.py`: `use_rslora` (L99), `use_dora` (L103), `pissa_init` (L107), `loraplus_lr_ratio`
(L91) — all togglable, default off. Wired into `peft_kwargs` at `adapter.py:259-260` (`use_rslora`,
`use_dora`) and `:292-298` (PiSSA). **Not recommended for the first draw** — they are C5 recipe-knobs
(orthogonal, forking-path); keep the arm a single clean flip so K-V2 attributes any gain to *vision reach*,
not a recipe change.

### 1.5 Recommended registered arm (ONE arm)

**`Qwen2.5-VL-7B-Instruct-LoRA-vis`** = the generic `hatemm_qwen25vl_lora_sft.yaml` recipe with **exactly two
changed lines**: `freeze_vision_tower: true → false` and `lora_target: q_proj,…,down_proj → all`. Everything
else byte-identical to generic (r16 / α32 / dropout 0.0 / 3 epochs / lr 1e-4 / cosine / warmup 0.05 / bs 1 /
grad-accum 8 / bf16 / grad-checkpointing / 8-frame ShareGPT / cutoff 4096). `freeze_multi_modal_projector`
stays **true** — under LoRA the merger is forbidden from LoRA anyway (`misc.py:37-38`), and unfreezing it
would require `additional_target: visual.merger` (full-FT `modules_to_save`), an extra overfit surface not
worth it for the first draw. **Justification vs full-ViT-FT:** full-FT = ~675M trainable on 743/549/579
videos = catastrophic overfit + ~8 GB optimizer state; rank-16 LoRA-on-vision = ~11M, bounded, the only
defensible variant.

---

## 2. RESOURCES — trainable params, VRAM, wall-clock (numbers verified, not guessed)

### 2.1 Trainable-param count (the recon's "~20M" was wrong; measured exact)

Counted **bit-exact from the on-disk generic adapter safetensors header** (`logging/lora/HateMM/
adapter_model.safetensors`) and reproduced analytically:

| component | modules | trainable params (r16) |
|---|---|---|
| **LLM-LoRA** (generic, unchanged) | 7 × 28 layers = 196 Linears | **40,370,176** (measured, = 40.37M) |
| **+ ViT-LoRA** (new) | (qkv+proj+gate+up+down) × 32 blocks = 160 Linears | **11,151,360** (analytic, +11.15M) |
| **vis-LoRA total** | 356 Linears | **≈ 51.5M** (r16), **+27.6% over generic** |

(Analytic formula `r·(in+out)` reproduces the LLM count 40,370,176 **exactly**, validating the +11.15M ViT
figure. The 161.5 MB file is fp32 storage of 40.37M values, not 80M params.) So the vision reach adds ~11M
trainable — **more than the recon's "a few M," but still bounded by rank-16 and small vs the LLM half.**

### 2.2 VRAM — comfortably feasible on 1×A100-80G

Generic encoder-SFT config (bs 1, grad-accum 8, cutoff 4096, 8 frames, grad-checkpointing) already runs the
ViT **forward** every step; the vision arm adds only the ViT **backward** + optimizer state for +11.15M
params (AdamW fp32 states ≈ +130 MB, negligible) + grad-checkpoint recompute of a **small** ViT (hidden
1280, ~168 vision tokens/video at `video_max_pixels 16384`). Peak stays well under 80 G — estimated +2–5 GB
over the generic run. **The banked OOMs in the logs are the r128 P9 `sft_classifier` regime (322M
trainable) and the frozen-32B extraction — NOT this r16 encoder SFT**, which completed with headroom.

### 2.3 Wall-clock — ~4–5.5 GPU-h SFT / dataset

Generic HateMM SFT (`logging/lora/HateMM/all_results.json`): **train_runtime 10,254.7 s (~2.85 h)** + eval
368.7 s, eval_loss 0.108, ~279 steps (ckpt-276). Vision backward + ViT recompute add roughly **+30–60%** →
**~4–5.5 h SFT/dataset**; + re-extraction ~0.4 h + 3-seed head ~2 min ⇒ **~4.5–6 GPU-h/dataset** end-to-end.

---

## 3. DATASETS + ARMS — the minimal decisive design

| dataset | role | vis-LoRA? | banked comparators (NO re-run) |
|---|---|---|---|
| **MHC-EN** (549 train) | **refutation target — image collapse is upstream, whole ballgame** | **YES** | generic-LoRA-EN (banked job 13235, adapter `logging/lora/MHC` + cache + 3 head-logs) · frozen-CLIP floor (12850) · **frozen-Qwen floor** (12850, the honesty bar) |
| **HateMM** (743 train) | **mechanism-aligned hold/upside** | **YES** | generic-LoRA (banked 13235) · frozen-CLIP (12850) · frozen-Qwen (12850) |
| **MHC-ZH** (579 train) | optional 3rd (off-mechanism — ZH is text-borne) | GO-IF EN moves | generic-LoRA (banked 13150) · frozen-CLIP (13115) |

- **3 head-seeds, dual protocol** (val-selected AND final-epoch, judged independently), decision rule verbatim
  from `exp-encoder-3seed.md:73-85` — identical machinery to LoRA-HateMM / cand-2.
- **Single-encoder-draw caveat carried (house F0.2):** the 3 head-seeds read ONE vis-LoRA SFT draw/dataset;
  the ±band is head-seed variance, not SFT-draw variance. Symmetric with the single-draw banked controls.
- **All comparators are banked** — the vis-LoRA arm is the only new SFT+extract+head; every floor and the
  generic arm are re-paired from existing trainlogs (numeric-provenance discipline).

---

## 4. KILL BARS (SKELETON — the prereg freezes exact numbers)

1. **K-V1 — house performance conjunct** (vs frozen-CLIP floor, primary): mean Δacc ≥ **+0.030** AND mean
   ΔmF1 ≥ **+0.030** AND sign **3/3**, judged independently per protocol. Below → NEGATIVE on that protocol.
2. **K-V2 — ADD-OVER-GENERIC** (the decisive bar; vis-LoRA − banked generic-LoRA, paired by head-seed):
   mean Δacc ≥ **+0.010** AND sign **3/3** AND mean ΔmF1 ≥ **0**. Because vis-LoRA = generic ⊕ ViT-LoRA, K-V2
   isolates the vision contribution **exactly**. TIE (|Δ|<0.010 or sign not 3/3) ⇒ "vision reach adds nothing
   over LLM-only LoRA" — bank the negative. (Same shape as cand-2 K-C2-2 / the recon's K-style bar.)
3. **EN-specific honesty flag:** on EN the vis-LoRA arm must **also beat the FROZEN-QWEN floor** (EN
   frozen-Qwen final 0.7847/0.7425, val-sel 0.7805/0.7219; `LORA_HATEMM_PREREG.md §2.2`) — the bar the
   *generic* EN LoRA could **not** clear (B4-EN expected-FAIL, below both frozen floors). If vis-LoRA still
   can't clear frozen-Qwen, the vision adaptation did not repair the collapse in a decision-relevant way ⇒
   it is **rearranging a dead cell**, not opening EN.
4. **Overfit / mechanism tripwire (two prongs):**
   - **image-stream-MOVED** (the F58 machinery, cite `HATEMM_LORA_STREAM_DECOMP.md` + `scripts/analysis/
     encoder_swap_geometry.py`; the red-team stream probe reproduced F58 to ~0.01 AUC): the **adapted image
     stream** must move vs the LLM-only-LoRA image stream — `img ΔAUC(vis-LoRA − generic-LoRA) ≥ +0.010`
     train-LOO **and** `≥ +0.005` dev (F58's MOVED rule). If the EN image stream stays flat (as it did under
     LLM-only LoRA: 0.653→0.659), the vision LoRA is inert ⇒ diagnostic auto-kill, independent of accuracy.
   - **eval_loss band** ~0.10–0.18 (generic HateMM 0.108); a much lower vis-LoRA eval_loss + a widening
     val-sel↘final-epoch gap = overfit warning on <750 videos.

---

## 5. $0 PRE-GATE — honest: none beyond what's already in hand

There is **no fully-$0 pre-GPU kill** for this cell, and I will not manufacture one. The vision-adapted
features do not exist until the SFT runs, so nothing cached can screen them. The **strongest cheap prior-mover
is already banked**: the red-team §0 CPU probe (banked caches only) shows LLM-only LoRA leaves the EN image
train-LOO AUC **flat (0.653→0.659 train, 0.695→0.695 dev)** while CLIP's healthy EN image is 0.745 — i.e. the
collapse is **upstream of the LLM, in the vision tower/merger**, exactly the parameters this arm is the only
lever that reaches. That probe *is* the $0 motivation. The only cheaper-than-full gate is a **$0-after-a-GPU-
extraction** screen: run the vis-LoRA SFT + extraction, then apply the F58 image-MOVED check (§4.4) to the
adapted cache **before** spending the (trivial) head budget — but that is cheap-after-GPU, not $0-pre-GPU.
**This cell is irreducibly a training experiment.**

---

## 6. GOVERNANCE

- **D7 status — plainly:** **encoder-class, SAME D7 collision as generic LoRA** (`LORA_HATEMM_PREREG.md`
  F0.3). LoRA-on-ViT is a 2024-25-standard technique; a pass is a **performance/ablation row**, not a novelty
  win. What this cell *does* buy on the method-space ledger is a **refutation of the WORDING of F51's
  two-object closure** and of `REDTEAM_BAN_SCOPE_AUDIT` GAP-5b / `REDTEAM_UNTESTED_CELLS` C1: every banked
  "encoder adaptation" adapted the **LLM only** (verified: generic adapter target-modules = 88 entries, **zero**
  `visual.*`), so "we adapted the encoder / EN is closed to the entire representation family" was asserted
  over a vision path that **was never adapted**. This measures it. **Not a novelty escape.**
- **Collision safety (all verified free this recon):** new adapter dirs `logging/lora/<DS>_vis`; new cache tag
  `Qwen2.5-VL-7B-Instruct-LoRA-vis_HF` (distinct from frozen / `-LoRA_HF` / `-LoRA-curric_HF`); new head group
  `RAC_video_lora_vis`. Extraction runner is **adapter-generic — no edit** (`generate_VideoMLLM_embedding_
  lora_HF.py:419` loads the full VLM, `:439-441` `PeftModel.from_pretrained` + `merge_and_unload` merges the
  ViT LoRA automatically; pass the new tag as `gen_embed_lora.sbatch` arg 3). New config = 2-line diff vs the
  frozen-vision generic config (§1.5), leaving the generic path byte-untouched.
- **In-box legality:** LEGAL — Qwen2.5-VL-7B local, single-dataset own-train (549/743/579), no gold, no OCR,
  no external API, no download; raw videos never leave; LoRA weights stay on disk. No standing veto touched.
- **Cost ledger:** EN + HateMM = **~9–12 GPU-h**; + ZH optional = **~14–18 GPU-h**. Mining/$0 CPU: n/a.
- **Honest prior per dataset (with mechanism):**
  - **MHC-EN ~10–15%** to clear K-V1 — the highest-value leg because it is the only untested lever aimed at
    EN's *upstream* collapse. Damper: F55's EN oracle ceiling (+0.025) is real but was measured on the
    **cross-encoder healthy-CLIP image**, so it does not fully subsume a **same-encoder** vision-repaired-then-
    co-trained Qwen. Most-likely EN outcome: **image stream MOVES but the conjunct still fails** (F44
    label-limited residual) — *still an informative refutation* of "EN closed to the entire family / no vision
    lever tried," and the honest performance kill for the vision axis.
  - **HateMM ~10–15%** to clear K-V2 (add-over-generic) — F58 showed HateMM's image is already strong/
    swap-neutral and LLM-LoRA left it flat while the pass is **text-carried & frozen-sufficient**; vision
    adaptation on an already-converted dataset likely sharpens a passing leg without adding a dataset.
  - **MHC-ZH ~5–8%** — ZH is **text-borne** (F45); vision adaptation is off-mechanism. Lowest priority.

---

## 7. RECOMMENDATION

- **GO.** Registered arm: **LoRA-on-vision+LLM** (`freeze_vision_tower: false`, `lora_target: all`,
  projector frozen, r16/α32, else byte-identical to generic).
- **Dataset order:** **EN first, HateMM alongside** (the two primary legs, ~9–12 GPU-h); read the **F58
  image-MOVED $0-after-extract diagnostic on EN before spending the head budget** — if the EN image stream is
  flat, the arm is inert and the axis is closed cheaply. **ZH only GO-IF EN's image stream moves** (else ZH is
  off-mechanism spend).
- **GO-IF conditions:** (a) EN-first sequencing with the image-MOVED gate; (b) carry the F0.2 single-encoder-
  draw caveat on any K-V2 pass; (c) K-V2 (add-over-banked-generic) is the decisive novelty-relevant statistic,
  since generic already passes HateMM — a K-V1 pass alone that merely equals generic earns nothing.
- **Total GPU-h:** ~9–12 (EN+HateMM) / ~14–18 (all three).

---

## PROVENANCE
- Mechanics: `RA-HMD/LLAMA-FACTORY-Ver202512/src/llamafactory/model/model_utils/visual.py:159-199,344-352`;
  `model_utils/misc.py:28-52`; `model/adapter.py:40-140,214-262`; `hparams/finetuning_args.py:99,103,107,539-549`;
  transformers 4.49.0 `modeling_qwen2_5_vl` (ViT `attn.qkv/proj`, `mlp.{gate,up,down}_proj`).
- Param count: on-disk `logging/lora/HateMM/adapter_model.safetensors` header (40,370,176, all LLM-side) +
  analytic ViT (+11,151,360) from `config.json` vision_config (hidden 1280 / inter 3420 / depth 32 / heads 16).
- Footprint: `logging/lora/HateMM/all_results.json` (train_runtime 10,254.7 s, eval_loss 0.108).
- Evidence chain: `REDTEAM_UNTESTED_CELLS.md` §0 + C1; `REDTEAM_BAN_SCOPE_AUDIT.md` GAP-5b;
  `HATEMM_LORA_STREAM_DECOMP.md` (F58 image-MOVED rule); `LORA_HATEMM_PREREG.md` §2.2 (EN floors, F0.x house
  clauses); `CAND2_CURRICULUM_PREREG.md` §1.2 (config-diff house style).
- **Required statements:** ZERO GPU / SLURM / Modal / download / test-touch spent by this recon; no held-out
  metric produced; no `state/`, prereg, config, `research-wiki/`, or frozen artifact mutated; NO job submitted.
  Committed on `main`, not pushed.
