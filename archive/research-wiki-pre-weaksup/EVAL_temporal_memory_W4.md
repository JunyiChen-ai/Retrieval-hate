# W4 — Evolving-memory protocol on the temporal splits (DESIGN_iter3 §3)

Date: 2026-07-03/04 (overnight wave W4).
Boundary respected: **no existing src/ file modified**; only new standalone
scripts + sbatch files (list at bottom). No git commit.

## Protocol

- Splits: `data/gt/{MHC,MHC_zh}_temporal/{train,val,test}.jsonl`
  (by upload_date; val period lies strictly between train and test periods).
- Head: frozen-CLIP RGCL head (same config as random-split `RAC_video_CLIP`
  heads: align fusion, triplet+hybrid BCE, topk=20, 30 ep, seed 0,
  Faiss_GPU=False), trained on **temporal-train only**, val-selected with
  **warmup>=5**, group `RAC_video_temporal`. Trained via
  `src/train_temporal_head.py` (reuses `run_rac.main` verbatim through a
  runtime loader injection; temporal features are the SAME cached CLIP
  embeddings re-indexed by video id — verified label-consistent by assert).
- Inference: arithmetic (rank-weighted) kNN vote over the memory, decision
  `score >= 0.5`; headline vote = `use_sim=False` (matches the quoted
  cross-matrix floors).
- Leak discipline: temporal-test NEVER enters the memory; k never tuned on
  test (full curves over k reported); asserts in code.

## 1. Random vs temporal split (same head-type, same protocol, mem=train)

| dataset | split | macro-F1 | acc | ROC | pos-F1 | majority acc / macro-F1 | test pos-rate |
|---|---|---|---|---|---|---|---|
| MHC (EN) | random  | 0.7113 | 0.7826 | 0.7175 | 0.5679 | 0.6957 / 0.4103 | 30.4% |
| MHC (EN) | temporal| **0.6273** | 0.7950 | **0.8484** | 0.3774 | 0.7578 / 0.4311 | 24.2% |
| MHC_zh   | random  | 0.7641 | 0.7987 | 0.8225 | 0.6739 | 0.6980 / 0.4111 | 30.2% |
| MHC_zh   | temporal| **0.7779** | 0.8255 | 0.8616 | 0.6750 | 0.7517 / 0.4291 | 24.8% |

- **EN: temporal split costs −0.084 macro-F1** (0.7113 → 0.6273) — evidence
  that EN hate content drifts within the MHClip window (2024-01→2024-05 test
  vs ≤2023-06 train).
- **ZH: NO temporal drop** (+0.014). Within MHClip's ZH window (test
  2023-11→2024-05), evolution is not measurable → the B-story must be
  scoped to EN (consistent with the known EN/ZH asymmetry of this model).
- Honest note: temporal test pos-rate is lower (24% vs 34% train;
  survivor+temporal bias), majority baselines recomputed and shown above.
- **Key diagnostic (EN)**: temporal ROC 0.8484 is *higher* than the random
  ref (0.7175) — the ranking survives the shift; only 8.7% of test scores
  clear the 0.5 vote threshold vs a 24.2% true pos-rate. The drop is an
  **operating-point (calibration) failure, not a separability failure**.

## 2. Adaptation mechanism A — memory augmentation (add k labelled new-period samples to the kNN memory; no retraining)

Pool = temporal-val (EN: 80 samples / 17 pos; ZH: 78 / 19 pos). Metric =
macro-F1 (arith vote); random = 5 seeds (mean±std); latest = newest by
upload_date; uncertain = smallest kNN vote margin |2p−1| vs static memory.

**MHC_temporal (EN), static = 0.6273:**

| k | random | latest | uncertain |
|---|---|---|---|
| 5  | 0.6213±0.0119 | 0.6273 | 0.5975 |
| 10 | 0.6154±0.0146 | 0.6273 | 0.5923 |
| 20 | 0.6180±0.0105 | 0.5975 | 0.5923 |
| 50 | 0.5944±0.0025 | 0.5923 | 0.5923 |
| all(80) | 0.5923 | — | — |

**MHC_zh_temporal, static = 0.7779:** same picture (random k=50
0.7584±0.0101; latest/uncertain 0.746–0.768; all(78) 0.7595).

**Result: memory augmentation is flat-to-NEGATIVE everywhere** — it does not
recover the EN drop, and adding the whole val period *hurts* (0.5923).
Adaptation-gain per sample ≤ 0 at every k on both languages. (Mechanism:
new-period positives' neighbours in the old memory are mostly negatives;
inserting a few new samples cannot flip rank-weighted 20-NN votes, and the
low-pos-rate val pool dilutes the memory's positive density.)

## 3. Adaptation mechanism B — threshold recalibration (same k labelled samples; memory untouched, no retraining)

Recalibrate only the vote threshold t (default 0.5) by maximizing macro-F1 on
the k labelled val samples' scores vs the static train memory (single-class
calibration set → keep 0.5, guarded). Same seeds as mechanism A.
Script: `scripts/analysis/temporal_recalibration.py` (CPU-only).
Baseline t=0.5 reproduces the static numbers exactly (internal consistency
check passed on both datasets).

**MHC_temporal (EN), static = 0.6273, random-split floor = 0.7113:**

| k | macro-F1 | acc | chosen t (per seed) |
|---|---|---|---|
| 5  | 0.6428±0.0432 | 0.7267±0.0789 | 2/5 guarded |
| 10 | 0.6839±0.0614 | 0.7205±0.0817 | 0.035–0.195 |
| **20** | **0.7336±0.0190** | 0.7864±0.0314 | 0.085–0.235 |
| 50 | 0.7106±0.0449 | 0.7578±0.0594 | 0.055–0.195 |
| all(80) | 0.7401 (t=0.185) | 0.7950 | — |
| ORACLE (test labels, diagnostic ceiling only) | 0.7646 (t=0.215) | 0.8199 | — |

**→ 20 labelled new-period samples FULLY recover the temporal drop**
(0.7336 ≥ random floor 0.7113; +0.0053 macro-F1 per sample), where the same
20 samples added to the memory recover nothing (0.6180). Threshold-only
ceiling 0.7646 > random floor — nearly the entire EN temporal drop is
**score/prior calibration drift**, not lost separability.

**MHC_zh_temporal (negative control), static = 0.7779, oracle 0.7845:**
recalibration has essentially nothing to gain and at small k it *hurts*
(k=5: 0.7114±0.0393; all(78): 0.7351) — where there is no drift, moving the
threshold on tiny calibration sets is pure noise. A deployed system should
recalibrate only when a drift signal exists (e.g. monitored score-vs-label
mismatch on the incoming stream).

## Honest verdict for the B chapter

1. "Hate evolves" is measurable on **EN only** within MHClip's window
   (−0.084 macro-F1); ZH shows no drop — say so explicitly.
2. The evolving-memory story in its original form ("add new samples to the
   memory to track drift") is **NOT supported** at MHClip scale: the
   recovery curve is flat-to-negative for all k ≤ 80 and all 3 selection
   strategies, on both languages.
3. The supported, weaker-but-real claim: **the dominant component of the
   EN temporal drop is calibration drift, and the correct k-shot lightweight
   adaptation is threshold recalibration (k=20 fully recovers the drop),
   which the retrieval architecture exposes as a first-class, O(1),
   reversible knob — a trained MoE/classifier head hides its operating
   point inside the weights** (adapting it requires fine-tuning). The
   memory bank itself remains useful for the cross-dataset swap capability
   (W-earlier result), but within-window drift tracking via memory growth
   is not demonstrated.
4. Caveats: val pool is small (17–19 positives); k=5 calibration is
   unstable (2/5 seeds single-class → guarded); temporal-val doubles as
   model-selection val (standard, pre-test period, test untouched);
   survivor bias (higher dead-link rate for Hateful/Offensive uploads)
   compresses the measurable drift.

## Jobs, artifacts, repro

- Jobs: smoke **12197** (COMPLETED, 3-ep + reduced eval); full wave
  **12214** (temporal train+eval for BOTH datasets COMPLETED; job state
  FAILED only at the final random-ref step because disk_guard had pruned
  the RAC_video_CLIP ckpts to B2 mid-day); repair refs **12253**
  (COMPLETED, self-healing B2 restore). Recalibration run CPU-side
  (scores-on-disk analysis, coordinator-approved).
- Results (JSON):
  `logging/temporal_memory/{MHC,MHC_zh}_temporal_evolving_memory.json`,
  `logging/temporal_memory/{MHC,MHC_zh}_random_ref.json`,
  `logging/temporal_memory/{MHC,MHC_zh}_temporal_recalibration.json`,
  logs `slurm/logs/tmp_{smoke_12197,mem_12214,refs_12253}.out`.
- Checkpoints (val-selected, warmup>=5):
  - `logging/Retrieval/MHC_temporal/RAC_video_temporal/RAC_lr0.0001_Bz64_Ep30_.../ckpt/best_model_16_0.8125.pt`
  - `logging/Retrieval/MHC_zh_temporal/RAC_video_temporal/RAC_lr0.0001_Bz64_Ep30_.../ckpt/best_model_27_0.8589....pt`
  - NOTE: disk_guard prunes oldest logging/*.pt to B2 when quota >250G; both
    best ckpts (and the two RAC_video_CLIP reference ckpts) were pruned and
    have been restored from `b2:junyi-data/RGCL_video/logs/...` with
    freshened mtimes. If missing again, restore the same way.
- New code (no src/ edits): `src/eval_temporal_memory.py`,
  `src/train_temporal_head.py`, `scripts/analysis/temporal_recalibration.py`,
  `scripts/slurm/{temporal_smoke,temporal_memory,temporal_memory_refs}.sbatch`.
