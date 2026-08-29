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

HateMM completion (paper-table completeness) — nearly done; **confirms the P9 pattern on the 3rd
dataset.** Matched frozen floors: raw-kNN test 0.786; **trained-RGCL Test_Retrieval 0.8605** (val-sel
0.870). All 3 seeds trained + kNN-extracted.
| read-out | HateMM test | vs trained-RGCL floor 0.8605 |
|---|---|---|
| **C3-mlp** (in-LMM head) | **0.8698** (s0 only; s1/s2 predicts GPU-blocked) | +0.9pt (≈ floor) |
| **C3-knn** (our memory) | **0.814** (s0 .823 / s1 .814 / s2 .805) | **−4.7pt (BELOW)** |

Same shape as EN/ZH: the LMM's own head ≈ the trained-RGCL floor, while our retrieval read-out on the
rgcl-OFF SFT'd space loses by ~5pt. This is exactly the C3-knn regression that P9b's rgcl-ON arm (D3)
is built to repair. (HateMM s1/s2 MLP predicts 12470/12472 pending behind GPU contention.)

### Jobs / artifacts
- Data: `src/utils/build_lora_sft_data.py` (word + yesno), `scripts/slurm/p9_build_data.sbatch`
  (frames). Config: `RA-HMD/LLAMA-FACTORY-Ver202512/my_configs/hatevideo/p9_mhc_c3_classifier.yaml`.
  Train: `scripts/slurm/p9_train.sbatch`. C3-knn read-out reuses `scripts/slurm/gen_embed_lora.sbatch`
  + `generate_VideoMLLM_embedding_lora_HF.py` → our kNN eval.

---

# P9b — RGCL-ON arm (train the MLLM by our retrieval-contrastive loss; decide by our memory)

**Directive (team-lead, supersedes the earlier "RGCL-fix not worth it" EV call).** The locked goal
(meaningful+novel MLLM integration AND substantial main-table improvement) is primary. P9's own
finding motivates this precisely. **Motivating observation (all 3 datasets, C3 rgcl-OFF):** the
LMM's own head lands ≈ the trained-RGCL floor while OUR retrieval read-out on the same SFT'd space
regresses BELOW it — **C3-knn vs floor: EN −2.7, ZH −2.2, HateMM −4.7** (head: EN +0.6, ZH +1.0,
HateMM +0.9). So rgcl-OFF SFT reshaped the embedding space *for the MLP head and AGAINST our kNN*.
The **rgcl contrastive term is the missing objective that trains the embedding space FOR the memory
vote** — the literal form of the goal (MLLM trained by our retrieval-contrastive loss, decision by
our updatable kNN memory). RA-HMD's release never ran this.

## The patch (implemented 2026-07-08; Codex-reviewed; no Qwen forward surgery)
Recon overturned the scoped "hard forward patch": the trainer's RGCL orchestration is **already
wired** (feature bank + reindex, `compute_rgcl_loss` call, `loss_ratio[2]` combine, rgcl logging).
Only two things were broken, both fixable without touching the Qwen forward:
- **(a) `sample_ids`** — `compute_rgcl_loss` only uses `len(sample_ids)=batch_size` on the dense
  path (self-exclusion is label-based), so `inputs["sample_ids"] = torch.arange(bs)` suffices.
- **(b) the never-implemented `model(classification_mode=True, output_embeds=True) → (outputs, pred,
  embeds)` contract** at two call sites (trainer query batch + `dense_retrieve` reindex) — both
  routed through the existing `get_embeds_from_last_layer` (the exact pooling+classifier the MLP head
  uses), via a `_rgcl_embed_fn` threaded into `compute_rgcl_loss`/`dense_retrieve`.
- **CPU-faiss fixes** (Codex-found; RA-HMD only ever ran GPU faiss, `faiss.get_num_gpus()==0` here):
  `train_feats is None` (was `== None`, crashes on the cached numpy bank at step 1); numpy `index.add
  /search` branch (was torch `.to(float32)`); `torch.zeros` device/dtype captured pre-numpy-conversion.
- **Files:** `sft_classifier/trainer.py` (compute_loss rgcl branch + `_rgcl_embed_fn`),
  `sft_classifier/rgcl_loss.py` (`embed_fn` thread + CPU-faiss fixes).

## PRE-REGISTRATION (locked before D3 training)
- **Condition D3 = rgcl-ON**: identical recipe to C3 (r128 α256, lr 4e-5, classifier_lr 1e-4, 3
  epochs, 8 frames, bf16, final-checkpoint no-selection) EXCEPT `rgcl: true`, `loss_ratio: [1,1,1]`
  (lm,cls,rgcl — RA-HMD default equal weighting), `Faiss_GPU: false`, RGCL defaults (contrastive/cos,
  1 hard-neg + 1 pseudo-gold, in-batch), `rgcl_reindex_every: 50` (video deviation: short training).
- **Read-outs (BOTH, one test touch/cell):** D3-mlp (in-LMM head) and **D3-knn** (our kNN vote over
  the SFT'd LMM's last-token embeddings, SAME pipeline as C3-knn → comparable). *Note: the rgcl loss
  shapes the proj_dim=1024 classifier-embed; D3-knn reads the 3584-d hidden state, shaped indirectly
  through the classifier — the experiment tests whether that transfer helps our memory.*
- **Datasets/seeds:** ZH (primary — head showed life) + EN, seeds 0/1/2, dev-gated as C3.
- **Floors (protocol-matched, from the P9 reconciliation):** EN test 0.7847, ZH test **0.8537**
  (LoRA final-epoch, NOT the frozen 0.8188).
- **Guards:** (1) **λ_rgcl=0 bit-for-bit** — `loss_ratio:[1,1,0]` must reproduce the rgcl-OFF C3 run
  exactly (same seed). (2) 20-step smoke: rgcl loss logged nonzero + decreasing, no NaN, ckpt saves.
- **Success bar (the goal's):** **D3-knn beats the protocol-matched floor by >1.5pt mean, ≥2/3
  seeds, AND D3-knn ≥ D3-mlp − 1pt** (the memory read-out must carry the story, not be a casualty).
  Anything weaker = honest kill.
- Configs: `my_configs/hatevideo/p9_{mhc,mhc_zh}_d3_s{0,1,2}.yaml`. Smoke: job 12485.

## P9b RESULTS — smoke (job 12485) + guards

**Patch validated (runs end-to-end):** rgcl_loss logged nonzero, **no NaN**, checkpoint saves
(adapter + classifier.bin), gradient flows, lm/cls losses decrease. The 5 fork bugs + 4 Codex
CPU-faiss bugs all clear at runtime.

**BLOCKER — the rgcl loss is degenerate in the bs=1 video regime.** Logged components (ZH s0, 20 steps):
- `in_batch_negative_loss = 0.0` at **every** step: `per_device_train_batch_size=1` (forced by the
  8-frame vision-tower OOM) → one sample per batch → **no in-batch positives/negatives**, so RGCL's
  primary contrastive signal is identically zero.
- The only remaining signal (1 hard-neg + 1 pseudo-gold retrieved from the bank) collapses:
  `negative_loss` 2.25→2.49, `positive_loss` 2.69→2.50 (gap 0.44 → 0.006) → hard-neg ≈ pseudo-pos →
  **`rgcl_loss` pinned at ln(2)=0.692** (0.58/0.6916/0.6923/0.6919) = ~zero discriminative gradient.
- Cause: (1) bs=1 removes in-batch negatives; (2) the bank is reindexed only at step 0 (579 forwards
  ≈5 min — too expensive to refresh often), so as lm/cls training drifts the space, all bank features
  become equidistant from the query.
- **Consequence:** as configured (RA-HMD defaults, bs=1), the rgcl term contributes ~no gradient →
  **D3 ≈ C3 by construction.** RGCL does not port to the bs=1 video regime without restoring the
  in-batch-negative signal. Pending team-lead config call (raise effective batch ≥4–8 via fewer
  frames / lower pixels = RA-HMD-faithful fix; or reindex-often; or more retrieved pairs).

**Bit-for-bit (resolved):** the λ=0 guard (12487) matches a **fresh rgcl-OFF C3-repro** (12491, same
current code) to within ~0.001 early (guard lm 0.3291 / cls 0.84 vs repro 0.3286 / 0.8393),
diverging chaotically like any two GPU runs. **Both** differ from the old C3 run 12438 (lm 0.3178) →
12438 was a **stale reference**, not a patch corruption. The RNG-isolation discipline holds; the
plumbing is transparent (not provably bit-identical without deterministic-CUDA mode, which we don't
run). So loss_ratio=[1,1,0] ≈ rgcl-OFF as intended.

## PRE-REGISTRATION AMENDMENT (option a — team-lead ruling, justified by the bs=1 degeneracy)
The bs=1 config makes `in_batch_negative_loss ≡ 0` and pins `rgcl_loss` at ln(2) (pos/neg gap
0.44→0.006) — RGCL's primary in-batch term is dead. **Fix = restore in-batch contrastive by raising
the physical batch:** 4 frames (subsampled [0,2,4,6] from the already-extracted 8 — uniform, no
re-decode) at 256², **per_device_train_batch_size 4 × grad_accum 4 = effective 16**. Data registered
`*_lora_yn4_*` (`scripts/analysis/p9b_make_4frame_data.py`).
- **4-frame smoke (12492) PASSES:** `in_batch_negative_loss` nonzero+moving (0.79/0.65/0.89/0.73),
  `rgcl_loss` off the ln2 pin (~1.1), no NaN, no OOM (bs4 fits), bs4 eval fits, ckpt saves.
- **reindex_every=25** for the wave (fix #2 direction — the retrieved pos/neg still converges, but
  that's a 20-step-smoke artifact since a 2nd reindex only fires at step 25; 25 gives 5 reindexes vs
  3 in the full run → fresher bank). The in-batch term is now the primary signal regardless.
- **Matched control C3′** (team-lead, load-bearing): rerun rgcl-OFF at the SAME 4-frame config but
  **through the rgcl branch with loss_ratio `[1,1,0]`** (not the else-branch). So **D3 − C3′ = pure
  rgcl-term effect**, sidestepping the λ=0 branch-divergence (12438 = stale ref; the branch itself
  matches a fresh else-branch C3-repro within GPU noise — footnote, off the critical path). The
  original 8-frame C3 stays as the original condition; the amended comparison is **C3′ vs D3**.
- **Wave:** {D3, C3′} × {ZH, EN} × seeds 0/1/2 = 12 runs (jobs 12494–12505). Read-outs both; floors
  EN 0.7847 / ZH 0.8537; bar D3-knn > floor +1.5pt (≥2/3) AND D3-knn ≥ D3-mlp−1; **+ D3-knn vs
  C3′-knn = the mechanism claim** (does the rgcl term specifically help our memory read-out).
- **Honest caveat:** 4 frames is a real evidence cut (esp. ZH visual-borne hate); frames-vs-batch is
  a follow-up if D3 works at 4 frames.

## P9b WAVE RESULTS (12494–12505 + chains; harvested 2026-07-08)

**Wave execution: clean.** All 12 trainings COMPLETED (~39–44 min each), all 12 embedding
extractions (p9bx, jobs 12506–12528 even, ~16–18 min each), all 12 test-MLP predicts (p9bp, odd,
~2 min each), and the kNN read-out (12536, `scripts/slurm/p9b_knn_eval.sbatch` →
`scripts/analysis/p9b_knn_out/*.json`) — zero failures, zero requeues. Test touched once per
run via the pre-committed predict configs + the single kNN pass (final-checkpoint, no selection).

*Pipeline note (held constant, not a knob):* embeddings for the kNN read-out are extracted at the
standard 8-frame setting (`gen_embed_lora.sbatch` default) even though D3/C3′ trained on 4-frame
inputs — identical for both arms, and identical to the C3-knn/floor extraction pipeline, so
read-out comparability is preserved; only the training objective differs between arms.

### Per-seed accuracy (dev = final-epoch eval; test = single touch)
| cell | seed | mlp_dev | mlp_test | knn_dev | knn_test |
|---|---|---|---|---|---|
| D3_ZH | s0 | 0.8590 | 0.8255 | 0.8333 | 0.8322 |
| D3_ZH | s1 | 0.8718 | 0.8389 | 0.8718 | 0.8456 |
| D3_ZH | s2 | 0.8718 | 0.8591 | 0.8718 | 0.8389 |
| C3′_ZH | s0 | 0.8974 | 0.8658 | 0.8590 | 0.8322 |
| C3′_ZH | s1 | 0.8590 | 0.8591 | 0.8974 | 0.8322 |
| C3′_ZH | s2 | 0.8590 | 0.8523 | 0.9231 | 0.7987 |
| D3_EN | s0 | 0.8000 | 0.7826 | 0.8125 | 0.7640 |
| D3_EN | s1 | 0.7625 | 0.7764 | 0.7625 | 0.7764 |
| D3_EN | s2 | 0.8375 | 0.7826 | 0.7750 | 0.7826 |
| C3′_EN | s0 | 0.8125 | 0.7888 | 0.7875 | 0.7640 |
| C3′_EN | s1 | 0.7875 | 0.8012 | 0.7750 | 0.7640 |
| C3′_EN | s2 | 0.8375 | 0.7888 | 0.7875 | 0.7888 |

### Cell means (mean±std, n=3)
| cell | mlp_dev | mlp_test | knn_dev | knn_test |
|---|---|---|---|---|
| **D3_ZH** | 0.8675±0.006 | 0.8412±0.014 | 0.8590±0.018 | **0.8389±0.005** |
| C3′_ZH | 0.8718±0.018 | 0.8591±0.005 | 0.8932±0.026 | 0.8210±0.016 |
| **D3_EN** | 0.8000±0.031 | 0.7805±0.003 | 0.7833±0.021 | **0.7743±0.008** |
| C3′_EN | 0.8125±0.020 | 0.7930±0.006 | 0.7833±0.006 | 0.7723±0.012 |

### Verdict vs the pre-registered bar — **FAIL (honest kill)**
Bar: D3-knn > protocol-matched floor by **+1.5pt mean with ≥2/3 seeds**, AND **D3-knn ≥
D3-mlp − 1pt**. Floors: EN test 0.7847, ZH test 0.8537.
1. **C1 (beat floor) — FAIL on both datasets.**
   ZH: D3-knn 0.8389 = **−1.5pt vs floor**, 0/3 seeds ≥ floor+1.5 (best seed 0.8456).
   EN: D3-knn 0.7743 = **−1.0pt vs floor**, 0/3 seeds (best 0.7826).
2. **C2 (memory read-out not a casualty) — PASS on both.**
   ZH: knn 0.8389 vs mlp 0.8412 (−0.2pt); EN: 0.7743 vs 0.7805 (−0.6pt). In rgcl-OFF C3 the gap
   was −2 to −5pt; with rgcl-ON the knn read-out now *matches* the LMM's own head.
3. Overall = C1 ∧ C2 = **FAIL**. The memory read-out is repaired *relative to the head* but the
   whole D3 system sits below the protocol-matched floor on both datasets.

### Mechanism read (D3 − C3′ = the pure rgcl-term effect, same branch, same recipe)
- **On the memory read-out (the claim): positive.** D3-knn − C3′-knn = **ZH +1.8pt**
  (0.8389 vs 0.8210), **EN +0.2pt** (0.7743 vs 0.7723). The rgcl contrastive term does train the
  LMM embedding space *toward* the kNN memory vote — sign confirmed, magnitude modest, EN within
  noise.
- **On the MLP head: the mirror image.** D3-mlp − C3′-mlp = **ZH −1.8pt** (0.8412 vs 0.8591),
  **EN −1.2pt** (0.7805 vs 0.7930). At equal loss weighting [1,1,1] the rgcl term *redistributes*
  accuracy from the head to the memory read-out (ZH: an almost exact ±1.8pt swap) rather than
  adding any.
- **Net mechanism conclusion:** the RGCL loss works as designed (shapes the space for the memory),
  but in this regime it buys no system-level accuracy — no cell of the wave (either arm, either
  read-out) beats its protocol-matched floor; the best cell overall is C3′-ZH-mlp 0.8591 ≈ floor
  0.8537 (+0.5pt, within the ±1.2pt seed band).
- Side observation (confounded, non-load-bearing): at 4-frame/bs4 the rgcl-OFF control's knn
  read-out already improves on the old 8-frame/bs1 C3-knn (ZH 0.8210 vs 0.7964, EN 0.7723 vs
  0.7578) — batch/frames/branch all changed, so no attribution claimed.

### Campaign consequence
P9b was the last open architectural locus (decision-level LMM training BY our retrieval loss).
The pre-registered bar fails; combined with P9 (rgcl-OFF SFT) and the settled MLLM method-role
campaign, the conclusion stands: **the MLLM earns no main-table accuracy role in this system at
7B; the rgcl term's effect is a head↔memory redistribution, not a gain.** Paper-usable positives
from P9b: (i) first working port of RA-HMD's (released-broken) LMM-RGCL stage-2 to video, incl.
the bs=1 in-batch-degeneracy finding + 4-frame/bs4 fix; (ii) the clean D3−C3′ mechanism pair
showing the retrieval-contrastive term specifically re-shapes the LMM space for kNN memory
(+1.8pt ZH) at the head's expense — evidence the memory pillar and the LMM head compete for the
same capacity in this regime.

### Artifacts
- Harvest/verdict: `scripts/analysis/p9b_harvest.py` (reads training `eval_results.json`, predict
  dirs `logging/lora_p9/predict/*f4*_test/`, and `scripts/analysis/p9b_knn_out/*.json`).
- kNN read-out job: `scripts/slurm/p9b_knn_eval.sbatch` (12536). Extraction caches
  `data/CLIP_Embedding/{MHC,MHC_zh}/{train,dev_seen,test_seen}_p9{d3,c3p}_{zh,en}_s{0,1,2}.pt`
  (pushed to B2 `embeddings/`).
- Fork commits (submodule): 4ade9754 (rgcl wiring), 63d52fd1/c00cd96d (guards), 9b409c0b (wave
  configs); predict configs `my_configs/hatevideo/p9_predict/test_{d3,c3prime}_{zh,en}_s{0,1,2}.yaml`.
