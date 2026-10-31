# Backbone candidate: scale-aware cross-modal attention

Status (2026-10-31): [rule-4 proposal review: GO](../../docs/reviews/20261031_scale_aware_cma_proposal.md); [rule-6 code review: PASS](../../docs/reviews/20261031_scale_aware_cma_code.md); HateClipSeg 3 seeds done: 3-seed mean
.6858 / .6710, below r5 (.6865 / .6737), so goal 2 is not met by this candidate; over 30 same-hyperparameter trial
pairs the bias has no effect (section 5). HateMM seed 234 still running (kept for information); seeds 2025 / 3407 not run. Part of the user's two-goal autonomous iteration
(2026-10-31). Goal 2 asks for the project's own backbone to reach r5's 3-seed AP and ROC on both main corpora at
fixed 8 calls. r5: HateMM .6909 / .8849, HateClipSeg .6865 / .6737.

## 1. Evidence that motivates the change

These are developmental test analyses (rule 10). The script is `experiments/20261031_two_goal_diagnostics/backbone_gap.py`.
The outputs are `runs/20261031_two_goal_diagnostics/backbone_gap/{hatemm,hateclipseg}.json`. The input was the
test-selected trial of each seed for r5 and for the inside-outside backbone (IO), at fixed 0 and 8 calls.

| Fixed 8 calls, 3-seed mean | AP / ROC / within | Video ROC (mean score) |
|---|---|---|
| HateMM r5 | .691 / .885 / .773 | .928 |
| HateMM IO | .676 / .879 / .738 | .921 |
| HateMM, IO video level with r5 within-video shape | .685 / .883 / .773 | .922 |
| HateClipSeg r5 | .686 / .674 / .560 | .865 |
| HateClipSeg IO | .668 / .643 / .560 | .795 |
| HateClipSeg, IO shape with r5 video level | .677 / .665 / .560 | .838 |

- HateMM: the gap is in the within-video shape. When IO keeps its own video level but uses r5's within-video
  logit shape, it recovers .009 of the .015 AP gap and all of the within gap.
- HateClipSeg: the gap is mostly video-level discrimination (video ROC .795 vs .865), plus part of the shape.
- All three tree-composition encoders (initial, additive, residual) stayed below r5. Recursive interval
  composition and the root readout lost video-level discrimination. Adding the interval context as a residual on
  top of r5 did not close the gap either.
- The r5 project history shows that within-video localisation comes mainly from the content model, not from the
  VLM answers (`experiments/20260925_query_paradigm/README.md`, decomposition and the point-prompt pilot).

The r5 encoder is MACIL-SD's cross-modal layer. In MACIL-SD's layer, each second of one stream attends to all
seconds of the other stream, with no positional term (`src/qtl/model.py` `MultiHeadAttention`). The layer is
therefore permutation-equivariant in time:
- A second's representation is its own projection plus a mixture of the other stream that is the same whether the
  attended seconds are its neighbours or the far end of the video.
- The encoder cannot use temporal neighbourhood, even though hate in these corpora comes in contiguous runs and
  speech and visuals that belong together are close in time.
- Temporal structure enters only downstream of the encoder, through the chain prior and the node prior's interval pooling along the tree. MACIL-SD's released code has an unused hard local-window option (`masksize`, default off), a different mechanism (a single hard cut-off, not learned).

This is the specific weakness this candidate addresses, without removing what r5 does well: global cross-modal
retrieval and attention-pooled video head.

## 2. Mechanism

One change to the r5 content prior network. Everything else is r5 unchanged: inputs, VLM answers, answer model,
chain, node prior, losses, EIG policy, evaluation.

- **Scale-aware cross-modal attention.** The attention score between query second t and key second t' gets an
  additive learned bias `b[h, k(|t - t'|)]` per head h.
  - Scale bucket: `k(0) = 0` and `k(d) = min(1 + floor(log2 d), K - 1)` for d >= 1.
  - K = 11, so the buckets are distances 0, 1, 2-3, 4-7, ..., 256-511, and >= 512 seconds.
  - The buckets approximate the dyadic scales of the question tree's nodes: a node at depth j spans floor or ceil of
    T / 2^j seconds, so the scales match exactly only when T is a power of two. The attention can therefore learn how much context to take from each scale, from the same second up
    to the whole video.
- **Shared across directions.** The bias table is shared by both directions (visual to audio+text and audio+text
  to visual), as the attention module itself already is.
- **Initialisation.** The table is initialised at 0, so at initialisation the network equals r5 exactly. Training
  decides whether locality helps. With uniform bias over scales the network stays the order-blind r5 layer.
- **Size.** 4 heads x 11 scales = 44 new scalars, all learned. The method adds no hand-set constant. The dyadic
  bucket edges are the tree's own scales, not a tuned window.

Hypothesis: making cross-modal retrieval scale-aware lets a second weigh nearby evidence differently from distant
evidence. This should improve the within-video shape, the HateMM gap, without losing r5's global retrieval and
video head. Whether it also helps the HateClipSeg video-level gap is an open question for the experiment.

## 3. Novelty scope (rule 4)

- Relative position biases come from NLP: Shaw et al. 2018; T5's log-bucketed relative bias, Raffel et al. 2020.
  This candidate transfers a log-scale relative bias into MACIL-SD's cross-modal attention for hateful video
  localisation, with buckets equal to the question tree's scales.
- The user's 2026-09-03 ruling (`docs/20260903_three_module_program.md`) counts fixing a MACIL-SD implementation
  choice that is clearly bad for this task as backbone novelty. Order-blind attention for temporal localisation is
  such a choice.
- No claim of inventing relative position bias. The reviewer must check whether hateful video
  detection/localisation work already uses relative-position cross-modal attention.

## 4. Protocol (fixed before the search)

- Same as r5:
  - Corpora HateMM and HateClipSeg; 8-level single-question soft answers (`soft_both`); `node_prior=true`.
  - Anchored answer model, learned closed zero-inflated chain, CMAL; hidden 128, ffn 128, 4 heads.
  - batch 32, 50 epochs, crop_repeat 5, long_T 512.
- Optuna TPE with sampler seed = training seed, same four scalars as r5:
  - `lr` log [1e-4, 1e-3]
  - `lamda_cma` uniform [.5, 2]
  - `dropout` uniform [.1, .5]
  - `lr_answer` log [1e-4, .1]
- Each trial trains fully. Validation (AP + ROC) / 2 at fixed 8 selects the checkpoint; then test.
- The Optuna objective is test (AP + ROC) / 2 (rule 7). The validation-selected trial is also recorded.
- Trial budget: 20 if the first trial takes <= 1 h, else 5. Locked in `budget.json`.
- Screening: seed 234 on both corpora, compared with r5 seed 234 (HateMM .6944 / .8867 / .7887, HateClipSeg
  .6885 / .6748 / .5561).
  - If both corpora are at least level with r5 within single-seed noise (.005), run seeds 2025 and 3407 for the
    3-seed comparison against r5's 3-seed means.
- Ablation for rule 14(g): the same locked trial with the bias table frozen at 0, which is exactly r5, on 3 seeds.
- Compute: Slurm on uoa-lab3 (HateMM) and uoa-lab1 (HateClipSeg), `launch/sbatch_search.sh`.
  - r5 reference: about 740 s per HateMM trial and 300-1200 s per HateClipSeg trial (with contention).
  - The added cost is one T x T bias gather per attention call.
- Output: `runs/20261031_scale_aware_cma/<corpus>/seed<seed>/`.
- Code: the network is `src/qtl/scale_cma.py` (moved from this directory's `backbone.py` on 2026-10-31, no change, so
  that the goal-1 evaluator can load these checkpoints through `src/qtl/backbones.py`); `train.py` and `search.py` here
  are the entry points.

## 5. Results

Best trial of each 20-trial search (test-selected, rule 7), fixed 8 calls, AP / ROC / within.
Source: `runs/20261031_scale_aware_cma/<corpus>/seed<seed>/search.log` and `trial<k>/summary.json`; r5 from
`runs/20260929_query_paradigm_r5/<corpus>/seed<seed>/trial<k>/summary.json` (`fixed.8`).

| Corpus, seed | Scale-aware CMA | r5 | Difference |
|---|---|---|---|
| HateClipSeg 234 | .6853 / .6885 / .5859 (trial 7) | .6885 / .6748 / .5561 (trial 13) | -.0032 / +.0137 / +.0298 |
| HateClipSeg 2025 | .6921 / .6749 / .5757 (trial 18) | .6896 / .6798 / .5461 (trial 11) | +.0025 / -.0049 / +.0296 |
| HateClipSeg 3407 | .6799 / .6495 / .5762 (trial 4) | .6813 / .6666 / .5785 (trial 19) | -.0014 / -.0171 / -.0023 |
| HateClipSeg, 3-seed mean | .6858 / .6710 / .5793 | .6865 / .6737 / .5602 | -.0007 / -.0027 / +.0191 |
| HateMM 234 | running | .6944 / .8867 / .7887 (trial 9) | |

Seeds 2025 and 3407 of HateClipSeg were started before the HateMM seed-234 screening finished, to use the free GPU;
the decision rule above is unchanged.

HateClipSeg conclusion: goal 2 (3-seed mean AP and ROC >= r5) is not met (-.0007 / -.0027). The mechanism has no
measurable effect on HateClipSeg:
- Mean over all 20 trials per seed, scale-aware minus r5: AP / ROC -.004 / -.006 (234), +.009 / +.005 (2025),
  -.007 / -.006 (3407).
- The first trials of each search have the same hyperparameters as r5's (same sampler seed). Over these 30 pairs the
  objective difference is -.004 +- .004 (mean +- s.e.) and within +.005; the size of the learned table (max |b| from
  .01 to 12.7) does not correlate with the difference (r = .19 for the objective, -.06 for within). Pairs whose table
  stayed at about 0 still differ by -.04 to +.01: one training run varies by about .02 with fixed hyperparameters,
  so a candidate needs an expected gain clearly above .01 to pass goal 2 reliably.
- Video-level / within-video split (`backbone_gap.py`-style, CPU on lab1): video ROC .875 vs .865 and within .579
  vs .560, but positive-only pooled ROC .6115 vs .6187; r5's video level with the scale-aware within-video shape gives
  .6874 / .6729, still level with r5. Better per-video ranking did not turn into pooled gains.
- Where r5's remaining error is (`experiments/20261031_two_goal_diagnostics/headroom.py`): on HateClipSeg perfect
  within-video ranking would give .806 / .796 and perfect rejection of negative videos .705 / .717; on HateMM .749 /
  .902 and .751 / .929.

- HateClipSeg is level with r5 within .005 (AP -.003, ROC +.014). The learned table of trial 7 is small (|b| <= .35):
  the third head adds about .35 to distances up to 16 s and -.31 to 128-255 s; the >= 512 s column stays at its initial 0 (no training
  pair of seconds was that far apart).
- Validation-selected trials: HateClipSeg 234 trial 12 (.6688 / .6337 / .5573), 2025 trial 3 (.6663 / .6325 / .6040).
- HateMM seed 234, first two trials against the r5 trials with the same hyperparameters (same sampler seed):
  .6866 / .8798 / .7544 vs .6862 / .8824 / .7726, and .6720 / .8736 / .7468 vs .6781 / .8755 / .7570. Within is lower by
  .010-.018, the opposite of the hypothesis so far. Trial 0 (lr_answer .08) learned a large table: one head +9.8 at
  distance 0 and about -7.5 at 32-511 s (local attention), another +5 at 32-255 s.

## 6. Run-time note (2026-10-31)

- Seed-234 HateMM trial 0 took 2169 s on uoa-lab3. The r5 trial with the same hyperparameters took 741 s.
  HateClipSeg (short videos) took about 153 s per trial.
- Suspected cause: autograd's backward of the index lookup `table[:, buckets]` scatter-adds 4 x T x T values into 44
  slots per attention call. HateMM training batches reach T = 5809 s (batch size 1 for the longest videos).
- From seeds 2025 and 3407 on, the table gradient is computed as a dense product (`_BucketBias`, now in `src/qtl/scale_cma.py`):
  - the forward values are identical (same lookup);
  - the gradient equals the lookup's gradient up to float summation order (equal in a float64 CPU check).
  Seed-234 trials used the lookup's own backward. The mathematics of the method is unchanged.
- GPU timing and equality check: `bench_bias.py` (`launch/sbatch_bench.sh`), output
  `runs/20261031_scale_aware_cma/bench_bias_1303.out` (uoa-lab1, RTX 5090). Code review PASS:
  `docs/reviews/20261031_scale_aware_cma_grad_code.md`.
  - Forward identical at all four shapes; table gradient max relative difference 1.6e-7 to 1.2e-6 (float32).
  - One attention forward + backward: lookup vs dense 0.7 vs 0.6 ms (B 32, T 120), 2.6 vs 1.4 ms (32, 300),
    23.8 vs 15.5 ms (8, 1000), 157.9 vs 111.0 ms (4, 2000).
  - The dense gradient is 1.1-1.9x faster on this layer. That alone does not explain the 3x longer HateMM trial;
    the per-trial times of seeds 2025 / 3407 will show the effect on a full trial.
