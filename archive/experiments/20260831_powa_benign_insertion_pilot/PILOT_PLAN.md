# Pilot plan: negative-bag-certified benign insertion in POWA

Frozen: 2026-08-31, before implementation or training. Novelty evidence:
`NOVELTY_SCOUT.md`. Starting-point authority:
`runs/20260831_powa_starting_point/summary.json` and its per-seed evaluator
outputs.

## Research question

Can a negative video's reliable bag-level implication—every temporal instance
is benign under the dataset labeling protocol—supply the missing local negative
supervision inside a positive temporal context, improving POWA's within-video
ordering without sacrificing its pooled discrimination?

The method remains weakly supervised: each corpus is trained independently and
the only human training annotation is its video label. POWA's existing Qwen
teacher also produces sparse machine pseudo-targets directly from train-video
content; this is local machine supervision and must be disclosed as such. No
human frame/span annotation enters a gradient. Validation frame labels select
a checkpoint; test labels are evaluation-only.

## One core mechanism

For a positive recipient, sample one continuous multimodal feature window from
a negative donor in the same corpus and train split. Insert it into the
recipient sequence without deleting recipient content. The composite remains a
positive bag. Apply:

1. the unchanged POWA loss to the original sample;
2. positive-bag MIL to the composite;
3. dense benign BCE only on the donor interior, excluding a fixed boundary
   buffer;
4. stop-gradient prediction consistency only on unchanged recipient positions,
   excluding recipient positions adjacent to the insertion boundary.

The dense donor target is justified only by the negative-bag assumption.
Recipient positions remain latent and never receive dense hate/benign targets.
Consistency is a protection against global score drift, not a second novelty
claim.

Intervention layer is frozen to POWA's aligned I3D/VGGish/BERT feature streams.
All three modalities move together. The recipient is first represented by the
same at-most-200 uniformly sampled real rows used by POWA; a raw consecutive
donor window is then inserted, so all recipient rows remain present. The
composite is padded to a fixed augmented width and uses an explicit attention
padding mask.

Frozen settings:

- donor duration: uniformly sampled integer from 12 to 36 feature rows
  (approximately 8–24 seconds on the I3D snippet grid), capped by donor length;
- insertion location: uniform over all gaps in the real recipient rows;
- boundary buffer: 3 rows at each donor end for donor BCE and 3 recipient rows
  on each side for consistency;
- donor BCE weight: 1.0;
- composite MIL weight: 1.0;
- recipient consistency weight: 0.5, probability-space MSE;
- original POWA loss weights, optimizer, five-crop training, five epochs, and
  seed otherwise unchanged from the corpus-specific starting point;
- learning rate is `2e-4` for both pilot corpora, matching the authoritative
  seed-234 `final_maskfix_finetune_hatemm` and the corpus-only
  `final5crop_teacher005_finetune_hateclipseg` `train_meta.json` files;
- checkpoint selection: highest validation within-video macro ROC; ties within
  `1e-6` break by pooled AP, then earlier epoch.

No duration, buffer, loss weight, branch, or checkpoint may be changed after a
test result is observed in this round.

## Stage P: performance gate

Pilot corpora are HateMM and HateClipSeg, seed 234. They cover the objective-gap
and feature-gap diagnoses and prevent advancement on the easy MHC-EN cell
alone. Run only:

- `A_matched_powa`: current code, explicit padding mask, no insertion;
- `E_full`: complete mechanism above.

Each arm is validation-selected independently, then evaluated once on TEST by
the shared evaluator. `E_full` passes only if its single frozen `score_powa`
branch strictly exceeds every current reproduced-table SOTA threshold below on
both corpora:

| Corpus | pooled AP | pooled ROC | within-video ROC |
|---|---:|---:|---:|
| HateMM | .5938316 | .8161838 | .6315317 |
| HateClipSeg | .6193711 | .6050225 | .5619079 |

It must also exceed its matched A arm on within-video ROC on both corpora. Any
failed metric kills this candidate; pooled degradation cannot be repaired by
calibration, ensemble, branch switching, or a second mechanism.

## Stage M: attribution gate, run only if Stage P passes

Using the same corpora/seed/settings:

- `B_splice_only`: composite positive-bag MIL, no dense donor or consistency;
- `C_original_negative_dense`: dense benign loss on original negative videos,
  no insertion;
- `D_insertion_benign`: insertion + donor BCE, no consistency;
- `F_positive_donor`: same as E but donor windows come from positive videos and
  are intentionally assigned the benign target; invalid supervision control,
  never a candidate method.

Required attribution:

- E within-video ROC exceeds B and C on both corpora;
- negative-donor E exceeds positive-donor F on both corpora;
- D→E reduces recipient prediction drift and does not reduce any pooled test
  metric; if consistency changes neither metrics nor drift it must be deleted;
- donor-score reduction is present beyond the boundary buffer;
- original-test improvement is not concentrated in videos whose predicted
  peaks lie near sequence edges.

Failure means the gain is ordinary augmentation, generic dense-negative
training, boundary detection, or arbitrary foreign-window suppression; the
novelty mechanism is rejected.

## Controls and integrity

- Persist donor/recipient ids, corpus, split, insertion index, duration, and
  boundary mask for every augmented training item needed for audit.
- Abort on cross-corpus, non-train, same-video, or non-negative donor.
- Full score arrays and the evaluator-written `metrics.json` are mandatory.
- Report extended within-video AP only as a diagnostic; it is not one of the
  three frozen project metrics and has no complete baseline SOTA table.
- Formal training cannot start until an independent reviewer passes model,
  data, alignment, evaluator, leakage, and control semantics.

## Later promotion gate

Passing this one-seed two-corpus pilot is not SOTA evidence. Promotion requires
all four corpora, at least three seeds, the same frozen primary branch, all 12
strict metric gates, paired per-video tests, complete controls, integrity audit,
and a second independent novelty search focused on the ablation-proven part.
