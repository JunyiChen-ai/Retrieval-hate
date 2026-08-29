# Pilot C5: leave-one-corpus-out span pretraining + rank-preserving weak adaptation

Date: 2026-08-30. Motivated by the cross-corpus probe
(`../20260830_xneg_mil_pilot/probe_cross_corpus.md`): a TemporalConv trained on
HateMM train spans transfers to MHC-EN at within-ROC .7819 and MHC-ZH at .7498
(above those corpora's own supervised ceilings), while every pointwise semantic
scorer (VLM windows, appearance kNN, external text classifier) and every weak
objective sits at .41-.63. The ordering signal must enter through span-level
supervision somewhere; the target corpus itself can stay weak.

## Setting (disclosed, fixed)

For each target corpus X: the model may use (a) the OTHER three corpora's
train-split span annotations (auxiliary supervision), and (b) X's train-split
VIDEO-level labels only. X's train spans are never loaded; X's val frame labels
only for checkpoint/hyperparameter selection (existing protocol); X's test only
for evaluation. This is a leave-one-corpus-out transfer setting and every
reported row states it.

## Mechanism (single core mechanism this round)

1. **Aux-span pretraining**: TemporalConv (diagnosis architecture, frozen
   CLIP+VGGish+BERT 1-fps features) trained with frame BCE on the union of the
   three auxiliary corpora's train-span rasterizations (single pos_weight
   computed over the aux union; amended pre-run 2026-08-30 to match the
   implementation — arm comparisons are unaffected since all arms share it).
2. **Rank-preserving weak adaptation** on the target train split: top-k MIL on
   video labels PLUS a pairwise order-distillation term from the frozen
   pretrained model's own scores (margin ranking loss on within-video pairs
   whose pretrained scores differ by >tau, tau = 0.5 logits). The constraint
   exists because naive MIL finetuning is expected to destroy transferred
   ordering (that failure is the attribution ablation).

## Arms (all 3 seeds 234/2025/3407, all corpora, TEST eval, standard evaluator)

- `loo_zero`: aux-span pretrained, no target adaptation (zero-shot transfer).
- `loo_adapt`: pretrained + rank-preserving weak adaptation (the method).
- `loo_naive`: pretrained + naive MIL finetune (no rank term) — attribution.
- `shuf_span`: pretraining on aux corpora with each video's span mask randomly
  rotated (circularly shifted by a uniform offset) — keeps per-video positive
  mass, destroys span placement; isolates whether span POSITIONS (not corpus
  frame statistics) carry the transfer.
- Reference rows from prior runs: weak-MIL control, best reproduced baseline.

## Gates (frozen before any arm runs)

- **Success**: `loo_adapt` test within-ROC macro beats the best reproduced
  baseline (hatemm .6315 / mhclip_en .6004 / mhclip_zh .5482 / hcs .5619) on
  >= 3 of 4 corpora, AND `loo_adapt` >= `loo_zero` - 0.01 on every corpus
  (adaptation must not destroy transfer).
- **Attribution**: `shuf_span` must NOT reach `loo_zero` - 0.02 on the corpora
  where transfer wins (otherwise the gain is corpus statistics, not spans, and
  the mechanism claim dies even if numbers are high).
- **Kill**: success criterion fails -> negative result, back to step 5.
- Secondary reporting (no gate): pooled frame AP/ROC, video ROC, hi-pos
  stratum, within-AP macro.

## Amendment A1 (2026-08-30, after first run; frozen before A1's run)

First run outcome vs the frozen gates: success clause PASSED (adapt beats the
best baseline on HateMM .6659 / EN .6610 / ZH .6263 vs .6315/.6004/.5482; HCS
fails as anticipated); no-destruction clause FAILED (EN .6610 vs zero-shot
.7289 = -.068; ZH -.011). Attribution controls all clean (naive < adapt
everywhere; shuf_span far below zero-shot everywhere). Per iteration rule 11
branch 3 (mechanism holds, implementation hurts strong-transfer corpora), one
implementation adjustment, no new mechanism:

- `loo_adapt_valsel`: identical training to loo_adapt, but the adaptation
  DEPTH is selected per corpus on the validation split's within-ROC macro
  (frame-labelled val is already sanctioned for checkpoint selection):
  checkpoints at epochs {0, 1, 2, 4, 8, 15}, pick the epoch with best val
  within-ROC (epoch 0 = pure zero-shot transfer is in the candidate set).
- Gate for A1 (frozen): loo_adapt_valsel beats the best reproduced baseline on
  >=3 of 4 corpora AND >= loo_zero - 0.01 on every corpus (the val selection
  makes this achievable by construction up to val-test noise; failing it means
  val does not track test ordering and the candidate dies).

## Known risks (stated pre-run)

- HateMM-as-target is the weak cell (aux sources EN/ZH/HCS individually
  transfer .55-.60); the >= 3 of 4 gate absorbs this.
- MHC-ZH within-n = 8: direction informative only, never load-bearing alone.
- Aux spans use the SAME rasterization built for diagnosis
  (runs/20260830_powa_within_diagnosis/gt_train_diagnosis_only) — that data is
  auxiliary supervision here, NOT diagnosis: the target corpus's own file is
  excluded from training by construction (verified in code review).
