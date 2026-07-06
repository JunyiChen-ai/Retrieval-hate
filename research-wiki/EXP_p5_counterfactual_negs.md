# P5 — MLLM counterfactual hard-negative twins

Front: P5 (campaign goal = give the MLLM a *real* method role). Last pre-planned backup,
upgraded from "prototype injection" using what the campaign learned.

Rationale chain: EN hate is speech / on-screen-text-borne (EXP_mm_segment_keys attribution);
the EN failure mode is the hate-vs-offensive/benign *boundary*; contrastive training currently
mines hard negatives from OTHER videos only. The MLLM can manufacture the perfect boundary
negative: for each hateful/offensive TRAIN video, rewrite its title+transcript into a
sanitized counterfactual — same topic/style/length, hate removed. The twin = the anchor's
REAL img_feats (unchanged) + sanitized text_feats (CLIP text tower, same chunk-mean pipeline)
= a *benign confounder with identical visuals*, used ONLY as a per-anchor extra hard negative
in the contrastive loss. **Removing the MLLM = removing these negatives = the ablation.**

**Explicitly EXCLUDED (by design):** injecting twins into the eval-time kNN memory as benign
entries. If hate persists visually, a benign label there would poison votes. Training-time
hard-negative use has no such risk (the twin is only ever a repulsion target, never a
voting neighbour).

---

## PRE-REGISTRATION (locked before training; written 2026-07-06)

### Generation
- Model: Qwen2.5-VL-7B-Instruct, greedy, fixed prompt. Anchors = TRAIN label==1 videos of
  each dataset (MHC 168, MHC_zh 180). Source text = the `text` field of
  `data/gt/<DS>/train.jsonl` (title+transcript). Store
  `data/Counterfactual/<DS>/train_twins.jsonl`.
- Twin text_feats: encode the sanitized text with the EXACT chunk-mean CLIP-text pipeline
  used to build the whole-video caches (`generate_VideoCLIP_embedding_HF.encode_text`:
  ≤75-content-token windows → per-chunk pooler_output → mean). Twin img_feats = the anchor's
  REAL cached img_feats (no re-decode). Cache: `data/CLIP_Embedding/<DS>/train_cftwin_<model>.pt`.

### Quality gate (mandatory, before training)
- (a) **Self-verdict flip**: the SAME MLLM judges the ORIGINAL text HARMFUL and the SANITIZED
  text BENIGN. A twin is a VERIFIED FLIP iff orig==HARMFUL and sanitized==BENIGN. Failures get
  at most ONE regeneration. Require conditional flip rate P(san=BENIGN | orig=HARMFUL) ≥ 0.80.
  Non-flipping twins are DROPPED from training (masked). Report the rate + overall retention.
- (b) **Hardness**: each verified twin's text embedding must be closer to its anchor's text
  embedding than the MEDIAN benign TRAIN video is (a NEAR miss, not just another easy
  negative). Report, per anchor, cos(twin, anchor) vs median_{benign} cos(benign, anchor), and
  the fraction of twins that are near misses.

### Training integration
- Extend the negative set in the contrastive (triplet) loss with the anchor's twin at weight
  1.0 — ONE extra negative, same treatment as a mined hard negative (twin_fused =
  model(anchor_img, sanitized_text); its cosine to the anchor is added into `hard_loss` inside
  the triplet relu). NO new tuned hyperparameter.
- Flag `--cf_negs`. **Flag-off is bit-for-bit identical to the floor** (no cache load, loss
  block skipped, no new params → optimizer/model init untouched). For ON-vs-floor clean
  isolation the twin forward is wrapped in CPU+CUDA RNG save/restore, so the main forward +
  hard-neg mining draws stay identical to the floor and the ONLY difference is the added
  negative's gradient.
- Control `--cf_negs_random`: each anchor gets a randomly chosen OTHER anchor's sanitized twin
  text as its extra negative (seeded derangement) — tests whether the per-anchor pairing
  matters vs merely adding a benign-text negative on the anchor's own visuals.

### Conditions (one test measurement per cell)
Per dataset (both): floor / cf_negs / cf_negs_random, CLIP space, exact RAC_video_CLIP recipe,
GROUP `RAC_video_p5cf`, seeds {0,1,2}, 30 ep. Report BOTH protocols (val-selected + final-
epoch), macro-F1 and accuracy, paired per-seed deltas. Known floors (val-sel): EN 0.7826 acc /
0.7113 maF1; ZH 0.8054 / 0.7706.

### Success criteria (pre-registered)
1. Flag-off (floor) reproduces the floor bit-for-bit.
2. Quality gate passes: conditional flip rate ≥ 0.80 AND twins are near misses (hardness).
3. cf_negs beats floor with mean ΔmacroF1 > 0.01 (noise ≈ 1.6 videos), ≥2/3 seeds positive, on
   ≥1 dataset under BOTH protocols, no >0.01 harm elsewhere; and ideally cf_negs > cf_negs_random
   (the pairing matters). Weaker ⇒ within-noise, honest kill.

---

## QUALITY-GATE RESULTS

_(filled after the generation job; JSON `scripts/analysis/p5_out/quality_gate.json`)_

<!-- GATE_PLACEHOLDER -->

## RESULTS

_(filled after training; JSON `scripts/analysis/p5_out/p5_results.json`)_

<!-- RESULTS_PLACEHOLDER -->

### Jobs / artifacts / repro
- Generation: `scripts/analysis/p5_generate_twins.py` + `scripts/slurm/p5_generate_twins.sbatch`.
- Quality gate: `scripts/analysis/p5_quality_gate.py` (CPU). Loss + flag: `--cf_negs` /
  `compute_cf_negative_sim` in `src/model/loss.py`, `src/run_rac.py`.
- Training: `scripts/slurm/train_p5cf.sbatch`, parser `scripts/analysis/p4_collect.py`
  (reused, GROUP RAC_video_p5cf).
