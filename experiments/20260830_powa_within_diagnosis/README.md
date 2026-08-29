# POWA-MACIL within-video diagnosis (iteration step 4)

Date: 2026-08-30. All numbers TEST split, 1 fps, evaluator
`scripts/reproduction_baselines/eval_baseline_scores.py`. Outputs in
`runs/20260830_powa_within_diagnosis/` (summary.md, skyline.md, per_video.csv).

## Question

POWA-MACIL is pooled-best on all four corpora but the research target is
within-video localization. Where and why does it fail?

## Findings

1. **POWA does not lead within-video ordering.** Test within-hate macro ROC:
   HateMM .5905 (MultiHateLoc fused: .6315), MHC-EN .5762 (CMHKF align .6004,
   unstable ±.175), MHC-ZH .4322 (below chance), HCS .5107 (VERA .5619).
2. **Failure concentrates in high-positive-fraction videos.** HCS pos>0.6
   (34 videos): .4806; MHC-ZH pos>0.6: .3332; while HCS pos<=0.2 reaches .7051.
   Correlation within-ROC vs pos_frac: HCS -.386, ZH -.505.
3. **MHC-ZH failure is an inversion, not noise.** In high-pos ZH videos the
   benign seconds sit at the edges (66% in the first 20% / last 10% of the
   timeline) and POWA ranks them at the top: mean score-rank .818 for benign
   vs .456 for hateful seconds. On HCS the ranking is simply uninformative
   (.500/.500).
4. **Supervised skyline (val-frame-trained, test-evaluated) caps the current
   features.** Best within-ROC per corpus: HateMM .6715, MHC-EN .6608,
   MHC-ZH .6756 (n=8), **HCS .5786**. On HCS even full frame supervision on
   frozen CLIP+VGGish+BERT barely beats chance: the 1 Hz features lack the
   within-video discriminative signal there. On the other three corpora the
   ceiling is ~.66-.68, leaving a real objective gap over the weak methods.

## Conclusions for the next iteration

- Two orthogonal deficits: (a) an objective deficit on HateMM/EN/ZH
  (weak methods sit .04-.15 under a small-data supervised ceiling), and
  (b) a feature deficit on HCS (ceiling itself near chance).
- Any candidate must be scored first on within-video AP/ROC (test), with the
  high-pos-fraction stratum reported separately; pooled numbers are secondary.
- Prior failed attempts to respect: V25 negative-reference density-ratio MIL
  and V26 counterfactual replacement both failed within-video gates —
  "reference/counterfactual against benign" as a family is exhausted at the
  feature level it was tried on.
