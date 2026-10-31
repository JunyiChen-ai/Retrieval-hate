# Backbone candidate: scale-aware cross-modal attention

Status (2026-10-31): [rule-4 proposal review: GO](../../docs/reviews/20261031_scale_aware_cma_proposal.md); [rule-6 code review: PASS](../../docs/reviews/20261031_scale_aware_cma_code.md); seed-234 searches submitted (Slurm, lab3 HateMM, lab1 HateClipSeg). Part of the user's two-goal autonomous iteration
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

## 6. Run-time note (2026-10-31)

- Seed-234 HateMM trial 0 took 2169 s on uoa-lab3. The r5 trial with the same hyperparameters took 741 s.
  HateClipSeg (short videos) took about 153 s per trial.
- Suspected cause: autograd's backward of the index lookup `table[:, buckets]` scatter-adds 4 x T x T values into 44
  slots per attention call. HateMM batches reach T of several hundred seconds.
- From seeds 2025 and 3407 on, the table gradient is computed as a dense product (`_BucketBias` in `backbone.py`):
  - the forward values are identical (same lookup);
  - the gradient equals the lookup's gradient up to float summation order (equal in a float64 CPU check).
  Seed-234 trials used the lookup's own backward. The mathematics of the method is unchanged.
- GPU timing and equality check: `bench_bias.py` (`launch/sbatch_bench.sh`), output
  `runs/20261031_scale_aware_cma/bench_bias_<job>.out`.
