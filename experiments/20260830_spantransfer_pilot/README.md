# C5 span-transfer pilot — PASSED (2026-08-30)

Plan/gates: `PILOT_PLAN.md` (frozen; amendment A1 frozen pre-run; both reviews
PASS). Artifacts: `runs/20260830_spantransfer_pilot/` (results.json/md,
valsel.md, logs). All numbers TEST split, within-hate macro ROC primary,
3 seeds (234/2025/3407), shared evaluator.

## Method (as piloted)

Leave-one-corpus-out span transfer: for target X, pretrain the TemporalConv
(frozen CLIP+VGGish+BERT, 1 fps) with frame BCE on the OTHER three corpora's
train spans; adapt to X with top-k MIL on X's video labels plus pairwise
margin-ranking distillation from the pretrained model (tau .5, lambda 1);
adaptation depth in {0,1,2,4,8,15} epochs selected on X's val within-ROC
(epoch 0 = zero-shot). X's own spans never touched.

## Final table (within-ROC macro, TEST)

| target | valsel (method) | loo_zero | loo_adapt e15 | loo_naive | shuf_span | best baseline | weak-MIL ctrl | skyline |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| hatemm | **.6801±.016** | .6283 | .6659 | .5840 | .5273 | .6315 (MultiHateLoc) | .5777 | .7495 |
| mhclip_en | **.7326±.001** | .7289 | .6610 | .5798 | .4726 | .6004 (CMHKF align) | .4592 | .7692 |
| mhclip_zh | **.6420±.113** | .6373 | .6263 | .5761 | .5458 | .5482 (MultiHateLoc union) | .4126 | .6217 |
| hateclipseg | .5431±.016 | .5389 | .5297 | .5383 | .5022 | .5619 (VERA) | .5234 | .5989 |

Selected depths (per seed): hatemm 4/1/15, en 15/0/0, zh 2/0/1, hcs 1/15/2.

## Gate verdict

- Beats the best reproduced baseline on 3 of 4 corpora (hatemm/en/zh) ✓
- >= loo_zero - .01 on every corpus ✓ (first fixed-depth run FAILED this on
  EN -.068 / ZH -.011; A1's val-selected depth fixed it as designed)
- Attribution: naive MIL finetune degrades ordering everywhere (rank term is
  load-bearing); shuf_span collapses transfer everywhere (span POSITIONS carry
  it, not corpus statistics) ✓

## Honest notes

- MHC-ZH within-n = 8, sd .113 — direction only, never load-bearing.
- HCS remains unsolved (feature deficit corpus: its supervised skyline is .599;
  no arm moves it; VERA .5619 keeps the corpus lead).
- EN frame AP drops for the epoch-0-selected seeds (zero-shot is uncalibrated
  to the corpus); within-video ordering is the primary metric, but the pooled
  column must be reported honestly in any paper table.
- Setting is target-span-free, not classically weakly-supervised: auxiliary
  corpora contribute span supervision. Every table row must disclose this
  (framing per NOVELTY_C5.md: cross-corpus LOCO protocol).

## Next (iteration steps 12-15)

Deep novelty check on the effective mechanism; refinement (module diet);
scale-up validation (significance, extra baselines: OSAD-style joint
multitask, CDL-style variant, per-source ablations); integrity audit.
