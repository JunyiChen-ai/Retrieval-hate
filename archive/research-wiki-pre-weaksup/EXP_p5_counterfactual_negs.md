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

## QUALITY-GATE RESULTS — BOTH DATASETS GATE-CLOSED (flip criterion fails)

Generation job **12377** COMPLETED (24 min, 348 anchors). JSON:
`scripts/analysis/p5_out/quality_gate.json`.

| dataset | anchors | orig HARMFUL | verified flips | **cond. flip rate** (≥0.80?) | retention | near-miss frac (hardness) | twin_sim vs median-benign |
|---|---|---|---|---|---|---|---|
| MHC (EN) | 168 | 149 | 75 | **0.503 — FAIL** | 0.446 | 0.987 PASS | 0.732 vs 0.296 |
| MHC_zh | 180 | 169 | 57 | **0.337 — FAIL** | 0.317 | 0.614 PASS | 0.795 vs 0.793 |

- **(a) Self-verdict flip FAILS on both** (0.503 / 0.337 ≪ 0.80). The SAME MLLM still judges
  ~half (EN) / two-thirds (ZH) of its OWN sanitized rewrites HARMFUL, even after one
  regeneration — i.e. it cannot reliably manufacture a clean counterfactual. Those
  non-flipping rewrites are DROPPED, leaving only 75 (EN) / 57 (ZH) verified-clean twins.
- **(b) Hardness PASSES.** The verified twins are genuine near misses: EN twin↔anchor cosine
  0.732 vs median-benign 0.296 (98.7% are nearer than the median benign); ZH 0.795 vs 0.793
  (61.4% — barely, because ZH benign transcripts already crowd the anchor in CLIP-text space).
- **Gate decision (pre-registered): CLOSED for both datasets** (a fails). Per success
  criterion (2) the method as specified is not viable — its premise (the MLLM makes clean
  boundary counterfactuals) holds only ~50%/34% of the time.

**Diagnostic training (transparently gate-failed):** because the KEPT twins (verified flips)
are clean and hard, we still run floor/cf/cfrand on the verified-flip subset (the cache masks
to flipped=True) to answer the complementary question — *does the clean subset help despite
partial coverage (twin on only 45%/32% of positives)?* This is reported as a diagnostic, NOT
as a gate pass.

## RESULTS (gate-failed DIAGNOSTIC — not a gate pass)

Training job **12392** COMPLETED (18 runs = floor/cf/cfrand × 3 seeds × 2 ds, 30 ep). The cf
runs verified they loaded the twin bank (EN 75 / ZH 57 valid flipped twins). JSON:
`scripts/analysis/p5_out/p5_results.json`.

- **Bit-for-bit PASS (both):** floor seed0 val-sel = MHC 0.7826/0.7113, MHC_zh 0.8054/0.7706,
  exact. The `--cf_negs False` no-op holds.

Floor vs cf (ours) and cfrand (random-pairing control), macro-F1, 3 seeds paired:
| dataset | protocol | floor | cf | Δ cf (per-seed) | cfrand | Δ cfrand |
|---|---|---|---|---|---|---|
| MHC (EN) | val-selected | 0.6715 | 0.6080 | **−0.0635** [−.105, +.058, −.144] | 0.5984 | −0.0731 |
| MHC (EN) | final-epoch | 0.7202 | 0.6931 | **−0.0271** [−.020, −.032, −.030] (0/3+) | 0.6730 | −0.0473 |
| MHC_zh | val-selected | 0.7676 | 0.7167 | **−0.0509** [−.016, +.013, −.150] | 0.7237 | −0.0439 |
| MHC_zh | final-epoch | 0.7720 | 0.7733 | **+0.0013** [−.007, +.010, .000] | 0.7746 | +0.0025 |

### What happened
- **The counterfactual twin as a hard negative HURTS EN and is flat on ZH.** On EN it is
  net-negative under BOTH protocols (final −0.027 with 0/3 seeds positive; val-sel −0.064);
  on ZH it is flat (final +0.0013). No dataset comes near the pre-registered +0.01 bar.
- **The per-anchor pairing does not matter.** cfrand (random OTHER twin) is about the same as
  cf — also hurting EN and flat on ZH. So the specific counterfactual pairing gives no benefit
  over merely adding a random benign-text negative on the anchor's visuals.
- **Mechanism (why it hurts EN):** the twin shares the anchor's REAL visuals and differs only
  in (sanitized) text, so it sits extremely close to the anchor in the align-fusion space
  (cosine 0.73). Repelling the anchor from a point that carries its own visual signal fights
  that signal and destabilises the positive cluster; with a twin on only ~45% of EN positives
  the pressure is also asymmetric. Net: harmful. On ZH the twins are barely near-misses (0.795
  vs 0.793) and fewer (57), so the effect washes out to flat.

### Verdict (plain language)
**MLLM counterfactual hard-negative twins do NOT earn the MLLM a method role.** Two independent
failures, both trustworthy (bit-for-bit floor holds):
1. **Primary — quality gate CLOSED:** the MLLM cannot reliably manufacture the clean boundary
   counterfactual the method needs — its self-verdict flip rate is only 0.503 (EN) / 0.337 (ZH),
   far below 0.80, so half (EN) / two-thirds (ZH) of "sanitized" rewrites are still judged
   harmful. The method's premise does not hold.
2. **Diagnostic — even the clean subset does not help:** trained on the verified-clean, verified-
   hard twins, the extra negative HURTS EN (−0.027 final) and is flat on ZH, and does not beat a
   random-pairing control. The mechanism itself (repel the anchor from a visually-identical
   text-sanitized twin) is counterproductive in this fused space.

Hardness alone (the one gate that passed) is not enough: a negative that is *too* close because
it shares the anchor's visuals is a distractor, not a useful boundary. Honest kill.

### Jobs / artifacts / repro
- Generation: `scripts/analysis/p5_generate_twins.py` + `scripts/slurm/p5_generate_twins.sbatch`.
- Quality gate: `scripts/analysis/p5_quality_gate.py` (CPU). Loss + flag: `--cf_negs` /
  `compute_cf_negative_sim` in `src/model/loss.py`, `src/run_rac.py`.
- Training: `scripts/slurm/train_p5cf.sbatch`, parser `scripts/analysis/p4_collect.py`
  (reused, GROUP RAC_video_p5cf).
