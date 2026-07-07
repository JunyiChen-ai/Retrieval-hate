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

### Fork readiness — 5 half-finished spots found + fixed (one-line paper finding)
The released `Ver202512` sft_classifier stage does NOT run out of the box; the RA-HMD stage-2 as
shipped runs **rgcl-OFF** (its RGCL contrastive path is unwired). To run the classifier at all
required: (1) the unwired RGCL (deferred behind the dev gate; patch plan above); (2) a YAML
duplicate-key crash from my smoke step (append only `max_steps`); (3) vision-tower CUDA OOM on
8-frame video (fix: `image_max_pixels 65536`/256²/frame, bs1×accum16, `expandable_segments`);
(4) `evaluate()`/`predict()` in `workflow.py:165/173` pass generation `gen_kwargs` to the custom
regression trainer that rejects them (fix: drop gen_kwargs — classifier eval is head-based, not
generative); (5) **the fork saves `classifier.bin` at train time (`trainer.save_model`) but never
reloads it** — a predict/eval-only run inits a *fresh* (random) head, so any post-hoc test pass
would score garbage. Fix (`workflow.py`, after `add_module`): when `not do_train`, load the trained
head from the adapter dir. Verified: predict-only on ZH-s0 val **reproduces eval_accuracy 0.8974
exactly** (= the training-run final-epoch val). Smoke walltime: **17.2 s/step**; full 3-epoch ≈
30–46 min/cell; predict-only eval ≈ 3 min/cell.

### Floor definition (corrected — load-bearing)
Two candidate floors were considered; the honest one is the **trained-RGCL baseline** = our actual
current method (align-fusion head + retrieval kNN, `Test_Retrieval`) on the SAME frozen-Qwen
features, matched test splits (EN 161 / ZH 149):
- **EN** — recent 3-seed `arcbase_MHC_Qwen2.5-VL-7B-Instruct_HF_seed{1,2,3}` (final-epoch):
  test **0.7847** (0.7702/0.7826/0.8012), dev **0.7750** (0.7875/0.7750/0.7625).
- **ZH** — `rgcl_MHC_zh_Qwen2.5-VL-7B-Instruct_HF`: test **0.8188** (corroborated: raw-kNN of the
  same frozen features also gives 0.8188). ZH dev floor is taken as the matched raw-kNN 0.8590.

A *headless raw-kNN* floor (EN test 0.7453 / ZH 0.8188) was used in interim reports — for EN this is
a **strawman ~4pt below our real system**, so all "+4pt EN" interim deltas are withdrawn. ZH test
floors coincide (0.8188) so ZH deltas are unaffected by the correction.

### C3 TEST accuracy vs the trained-RGCL floor (the deciding metric), 3 seeds
| read-out | EN test (floor 0.7847) | ZH test (floor 0.8188) |
|---|---|---|
| **C3-mlp** (LoRA-SFT LMM + its own classifier head) | 0.7909 (.783/.795/.795) = **+0.6pt, 2/3** | 0.8635 (.846/.879/.866) = **+4.5pt, 3/3** |
| **C3-knn** (OUR retrieval memory on the SFT'd backbone) | 0.7578 (.758/.739/.776) = **−2.7pt, 0/3** | 0.7964 (.792/.752/.846) = **−2.2pt, 1/3** |

Supporting DEV (3 seeds): C3-mlp EN 0.7792 / ZH 0.8761; C3-knn EN 0.7708 / ZH 0.8632 (the ZH-kNN
s0=0.8333 that looked "below floor" was the low seed — multi-seed dev ≈ floor).
ZH C3-mlp per-seed test: **0.8456 / 0.8792 / 0.8658** (F1 .768/.830/.796), final-checkpoint
(no val-selection). **2/3 seeds cross 0.85; s0 (0.8456) is just under.**

### ZH floor reconciliation (protocol-matched — the +4.5 shrinks to +1.0)
The "+4.5pt" is vs the **frozen** RGCL floor (a), which attributes the *entire LoRA benefit* to C3.
But we already have LoRA systems on ZH, so the fair floor is a LoRA system at the **same
no-selection protocol**:
| ZH floor | value | what it is | C3-mlp 0.8635 Δ | matched? |
|---|---|---|---|---|
| (a) arcbase/rgcl frozen-Qwen RGCL | 0.8188 | frozen encoder + our kNN head, final-ep | +4.5pt | no (frozen vs LoRA) |
| (b) LoRA-encoder+head, val-sel | 0.8322 | LoRA enc + our head, val-selected, 1 seed | +3.1pt | no (val-sel, 1 seed) |
| **(c) LoRA final-epoch, multi-seed** | **0.8537±0.012** | LoRA enc + our head, no-selection, 3-seed | **+1.0pt** | **yes** |

**Defensible headline = +1.0pt vs (c), WITHIN NOISE** (bands overlap: 0.8635−0.017=0.8465 vs
0.8537+0.012=0.8657). The LMM's own head does **not substantially beat** our existing LoRA-encoder +
RGCL-kNN route on ZH — it *matches* it. And (c) **already crosses 0.85** (0.8537) at no-selection, so
C3-mlp does **not newly** satisfy the ZH 0.85 target.

### Verdict
1. **EN — no gain.** C3-mlp +0.6pt vs frozen best 0.7847 is inside the 3-pt arcbase seed spread;
   C3-knn −2.7pt below floor. (EN low-prior; iter-2 encoder-LoRA also regressed EN, so frozen IS EN's
   best existing config.)
2. **ZH — matches, does not beat, our best existing system.** vs the protocol-matched LoRA floor (c)
   C3-mlp is **+1.0pt, within noise** (the +4.5 vs frozen is the LoRA benefit we already had). And
   **C3-knn — our retrieval-memory read-out, the method's novelty pillar — is −2.2pt BELOW floor on
   ZH (and −2.7 on EN).** SFT-ing the backbone for our kNN *hurts* on both; the MLLM's own head only
   *matches* the LoRA route on ZH.
3. **Campaign bar NOT met, under every honest framing.** No substantial improvement over our own best
   config on either dataset (EN +0.6 noise, ZH +1.0 noise); the retrieval-memory decision actively
   regresses. The MLLM's own head **displaces** rather than enhances the memory pillar — the
   integration clause fails.
4. **RGCL-fix: not worth it.** Its purpose is to improve the embedding-space kNN — below floor on
   both datasets. Do not spend the ~0.5–1 day forward-patch.

### Paper-usable findings (independent of the verdict)
- The released RA-HMD stage-2 (`Ver202512`) runs **rgcl-OFF** and needs **5 fixes** to run on video
  (§Fork readiness), incl. it **never reloads its own trained classifier head** on eval/predict.
- Decision-level LMM-SFT **matches** our LoRA route on ZH (+1.0pt vs protocol-matched floor, noise)
  and our retrieval read-out on the SFT'd embedding space **loses on both** datasets — a clean "the
  gain is not from the memory, and not beyond our existing LoRA" negative for the method story.

### HateMM — DEFERRED per team-lead ruling (verbatim, still standing)
> The gate opens on a literal tie (0.8411 ≥ 0.8411); expansion is deferred behind the two cells
> with real dev signal because (a) queue budget, (b) HateMM's role in this experiment is
> no-harm/completeness for the paper table, not the goal-carrying claim; the rule's expansion set
> is unchanged.

Team-lead confirmed the rule-literal expansion; HateMM completion now **RUNNING** for paper-table
completeness: s0 already trained (dev tie 0.8411), s1/s2 training = jobs 12463/12464; then kNN
extraction (s0/s1/s2) + single test pass (test_yn built, 215 vids / 40% hateful). Matched frozen
floor ≈ 0.870 (the campaign's HateMM SOTA). Test touched once/cell; no HateMM test run yet.

### Jobs / artifacts
- Data: `src/utils/build_lora_sft_data.py` (word + yesno), `scripts/slurm/p9_build_data.sbatch`
  (frames). Config: `RA-HMD/LLAMA-FACTORY-Ver202512/my_configs/hatevideo/p9_mhc_c3_classifier.yaml`.
  Train: `scripts/slurm/p9_train.sbatch`. C3-knn read-out reuses `scripts/slurm/gen_embed_lora.sbatch`
  + `generate_VideoMLLM_embedding_lora_HF.py` → our kNN eval.
