# Step-14 scale-up plan (frozen 2026-08-30, before runs)

Method name (working): **LOCO-ST** (leave-one-corpus-out span transfer with
rank-preserving adaptation). Framing per NOVELTY_DEEP.md: carrier + domain +
ordering-repair; LaGoVAD/AMR/TANDEM/LELA cited up front; "target-span-free",
never bare "weakly supervised".

## Runs (all TEST-reported, shared evaluator, dense scores saved per run)

1. **Headline arms, 5 seeds** (234/2025/3407/42/20260830): `valsel` (method),
   `loo_zero`. Dense test scores.jsonl written per corpus/seed/arm — the
   authoritative artifacts.
2. **Joint-multitask comparator** (OSAD-style, 3 seeds): single model trained
   from scratch on aux-span frame BCE + target MIL jointly (loss sum, no
   pretrain/adapt split, no distillation). The reviewer-expected alternative.
3. **Source ablations, 3 seeds, zero-shot**: each aux corpus alone as source;
   union minus HateMM. Quantifies source value (HateMM expected dominant).
4. **Sensitivity on HateMM only, seed 234**: tau in {.25,.5,1}, lambda in
   {.3,1,3}, top-k in {T//4,T//8,T//16} (one-at-a-time around the frozen
   config). Reported as a table; no re-selection of the main config.
5. **Significance**: per-video paired bootstrap (10k resamples over hate test
   videos with both classes) on within-ROC: valsel vs best reproduced baseline
   per corpus; valsel vs loo_zero; valsel vs loo_naive (from pilot). Report
   95% CI of the mean difference.

## Frozen judgment rules

- The pilot verdict stands; scale-up cannot un-pass it. Scale-up FLAGS:
  (a) 5-seed valsel mean drops below the best baseline on a corpus that passed
  -> that corpus's win is seed-fragile, reported as such;
  (b) joint-multitask >= valsel - .005 within-ROC on the corpora where the
  method wins -> the pretrain/adapt+distill structure is not load-bearing and
  the contribution reduces to the transfer finding (paper reshapes, method
  section shrinks);
  (c) bootstrap CI of (valsel - best baseline) includes 0 on HateMM -> HateMM
  win reported as non-significant; EN/ZH wins carry the claim.
- No hyperparameter changes to the method based on any scale-up number.

## Not in scope here

CDL-style uncertainty weighting (unlabeled-external setting differs), OCR/HCS
feature enrichment (separate future iteration), 4 fps grids, TANDEM IoU-metric
crosswalk (paper-writing stage).
