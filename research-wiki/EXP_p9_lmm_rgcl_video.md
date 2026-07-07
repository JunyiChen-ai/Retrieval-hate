# P9 — LMM-RGCL stage-2 adapted to video (MLLM as trainable backbone + our retrieval memory)

Front: P9 (campaign goal, user-hardlocked: MLLM meaningfully+novelly integrated AND a
SUBSTANTIAL performance improvement). The last untested architectural locus: the DECISION level
with full LoRA-SFT of the LMM.

## Load-bearing recon corrections (verified in code, before any training)
1. **RA-HMD's LMM-RGCL is embedding-space, NOT in-context.** The classifier prompt is query-only
   (frames + transcript → label); retrieval-augmentation is an RGCL contrastive loss shaping the
   LMM's last-token hidden state + (in the original) a kNN vote. No neighbors in the prompt.
2. **This fork's (`Ver202512`) RGCL contrastive path is UNWIRED — `rgcl: true` crashes.** Two
   grep-verified breaks: `rgcl_loss.py:27` consumes `batch["sample_ids"]` which is produced
   nowhere; `trainer.py:273` calls `model(**inputs, classification_mode=True, output_embeds=True)`
   → `(outputs,pred,embeds)`, but nothing patches the Qwen forward to accept those kwargs / return
   3 values. Every checked-in meme config runs **rgcl-OFF**. There is also **no eval-time kNN vote**
   in the fork (eval = the MLP classifier head, sigmoid≥0.5). *(Paper note: the released RA-HMD
   stage-2, as shipped, runs rgcl-OFF — a one-line finding.)*

Consequence (team-lead decision): run the **runnable rgcl-OFF joint LM+classifier-head LoRA-SFT**,
but restore the method integration by evaluating **two read-outs** — the fork's native MLP head,
AND **our kNN-vote over the SFT'd LMM's last-token embeddings** (post-hoc feature extraction, no
training-time surgery), so the decision mechanism stays our updatable memory. Scope (not
implement) the RGCL fix in parallel; wire it only if C3's dev signal is competitive.

---

## PRE-REGISTRATION (locked before training; written 2026-07-07)

### Data
- ShareGPT multi-image records: 8 uniform frames (`<image>`×8) + Title+Transcript (≤1500 chars)
  → **Yes/No** answer (the sft_classifier stage derives the binary target from the "Yes" token;
  hateful→`Yes`, normal→`No`). Built by `src/utils/build_lora_sft_data.py --answer yesno`;
  registered `{mhc,mhc_zh,hatemm}_lora_yn_{train,val,test}`. Frames `data/lora_frames/<DS>/<id>/`.
- LOO / leak discipline: the C3-knn read-out builds its kNN memory from TRAIN embeddings and
  excludes the query's self-match (LOO) for train queries; test queries vote over the full train
  memory. Train labels only; test labels touch only the final metric.

### Conditions (per dataset)
- **A — frozen-head floor** (our current CLIP/Qwen RGCL kNN system; known numbers, no LoRA).
- **C3-mlp** — joint LM + binary classifier-head LoRA-SFT (RA-HMD `sft_classifier` stage,
  rgcl OFF, `loss_ratio [1,1]`), prediction = the in-LMM MLP head (sigmoid≥0.5). *Fork-native.*
- **C3-knn** — SAME trained checkpoint; extract the SFT'd LMM last-token embeddings for
  train+test, build our kNN memory, similarity-weighted vote → prediction. *Our retrieval-memory
  decision on the LoRA'd backbone.*

### Hyperparameters (RA-HMD meme defaults + documented video deviations)
- LoRA `r128 α256 dropout0.05 lora_target=all`, lr 4e-5, classifier_lr 1e-4, 3 epochs, cosine,
  warmup 0.1, template qwen2_vl, `stage: sft_classifier`, `loss_ratio [1,1]`.
- **Video-forced deviations (documented):** 8 frames/video (meme=1 image); `cutoff_len 2048`
  (meme 1024 — 8 frames + transcript is long); **bf16 full (no QLoRA-4bit)** since the A100-80GB
  has room and it avoids a bitsandbytes variable (meme used 4-bit to fit small GPUs); vision tower
  + mm-projector frozen; per_device_bs 2 × grad_accum 8 = eff. 16; output_dir contains "qwen"
  (required for `yes_token_id` resolution). Adapters → B2, not git.

### Staging (GPU budget protection — pre-registered)
- Run **seed 0** for every (dataset × {C3}) cell first (A is free/known). Report DEV accuracy
  (MLP + kNN read-outs) per cell.
- **Dev-gated expansion:** expand a cell to seeds {1,2} ONLY if its seed-0 **DEV acc ≥ floor DEV
  acc** (either read-out) on that dataset. No test-based adaptivity.
- **TEST is touched exactly once per cell**, only after all planned seeds for that cell finish.
- Datasets order & expectation: **HateMM (floor 0.870) and MHC_zh (LoRA helped ZH before) are the
  primary bets; MHC-EN is included but a-priori UNLIKELY** — the iter-2 LoRA-SFT of this same
  Qwen-VL encoder already regressed EN (0.7516/0.6916, below both frozen floors) and crossed 0.85
  on neither MHClip split.

### RGCL-fix decision rule (pre-registered; scoped below, NOT yet implemented)
Wire the true rgcl-ON arm ONLY if C3's dev signal shows the joint-SFT embedding space is
competitive: **C3 dev acc ≥ floor dev acc on ≥1 primary dataset (HateMM or MHC_zh)**. No point
doing model surgery on a space already losing at dev.

### Success criteria (pre-registered)
1. Smoke: the sft_classifier stage runs on video multi-image data (20-step smoke) — report walltime
   for the seed budget before the wave.
2. C3 (either read-out) beats the frozen-head floor by **mean >1pt** with **≥2/3 seeds** on **≥1
   dataset** (dev-gated), test touched once/cell. Honest magnitude — the bar is SUBSTANTIAL.
3. Integration read: report C3-knn vs C3-mlp — whether our retrieval-memory read-out matches/beats
   the fork-native head (i.e. the memory decision carries over on the LoRA'd backbone).
Anything weaker = within-noise / honest kill, reported as the verdict.

---

## RGCL-FIX SCOPING (patch plan only — do NOT implement until the dev gate opens)

To make `rgcl: true` runnable (the true embedding-contrastive arm), two changes:
- **(a) `sample_ids` collator field.** `rgcl_loss.py:27` needs a per-sample stable index for FAISS
  self-exclusion. Add `sample_ids` (the dataset row index) in the sft_classifier data collator /
  dataset `__getitem__`, threaded into the batch dict. Anchor: the collator used by
  `sft_classifier/workflow.py` + `trainer.py` `compute_loss` (where `batch` is consumed). Low risk,
  ~localized.
- **(b) Qwen2.5-VL forward patch** to accept `classification_mode`/`output_embeds` and return
  `(outputs, pred, embeds)` (`trainer.py:273-274`). This is the hard, high-risk part: wrap/patch
  `Qwen2_5_VLForConditionalGeneration.forward` so that with `output_embeds=True` it (i) runs the
  base forward with `output_hidden_states=True`, (ii) pools the last-prompt-token hidden state
  (`get_embeds_from_last_layer` logic, `trainer.py:337-386`), (iii) runs `self.classifier` for the
  logit, (iv) returns the triple. Must survive gradient-checkpointing + LoRA (PeftModel wrapping) +
  bf16. Realistic effort: **~0.5–1 day** incl. a numeric-equivalence smoke; risk MEDIUM-HIGH
  (touches the LMM forward under PEFT). Only worth it if the dev gate opens.

---

## RESULTS

<!-- SMOKE_PLACEHOLDER -->
<!-- RESULTS_PLACEHOLDER -->

### Jobs / artifacts
- Data: `src/utils/build_lora_sft_data.py` (word + yesno), `scripts/slurm/p9_build_data.sbatch`
  (frames). Config: `RA-HMD/LLAMA-FACTORY-Ver202512/my_configs/hatevideo/p9_mhc_c3_classifier.yaml`.
  Train: `scripts/slurm/p9_train.sbatch`. C3-knn read-out reuses `scripts/slurm/gen_embed_lora.sbatch`
  + `generate_VideoMLLM_embedding_lora_HF.py` → our kNN eval.
