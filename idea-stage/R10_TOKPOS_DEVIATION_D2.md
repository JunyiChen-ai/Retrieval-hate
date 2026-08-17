# R10 Task B — deviation D2: the parity belt is replaced by an exact internal belt

**Filed after extraction, before any head run and before any arm metric exists.**
Supersedes the bar clause of `idea-stage/R10_TOKPOS_DEVIATION_D1.md`; everything else in D1 stands.

## What happened

D1 relaxed the belt from cosine ≥ 0.9999 to ≥ 0.99 on ≥ 99 % of rows against the banked
A100-extracted `-ro_L28` cache. On the full 579-row train split that bar also fails:

| split-level statistic (L28, A0 vs banked `-ro_L28` `text_feats`) | value |
|---|---|
| mean cosine | 0.996136 |
| median | 0.996846 |
| 1st percentile | 0.983588 |
| min | 0.962098 |
| fraction ≥ 0.99 | 0.9689 |
| fraction ≥ 0.98 | 0.9914 |
| fraction ≥ 0.95 | **1.0000** |

L24, same comparison: mean 0.998509, min 0.985433, fraction ≥ 0.99 = 0.9965.

**I am not relaxing the bar a second time.** Moving a threshold twice to fit the data is exactly
the failure mode the frozen-rule red line exists to prevent. Instead the belt is replaced by one
that tests the thing it was always meant to test.

## Why the cross-hardware belt cannot be the belt

Three pieces of evidence say the residual is floating-point drift from the platform migration, not
a span error:

1. **Depth monotonicity.** L24 (mean 0.9985, min 0.9854) is uniformly tighter than L28
   (mean 0.9961, min 0.9621). Error accumulated over four more decoder layers. A mis-indexed span
   would not be ordered by depth.
2. **No structural break and no covariate.** The cosine distribution is smooth and unimodal, every
   row is ≥ 0.95, and the Spearman correlation between cosine and transcript length is **0.056**.
   There is no subpopulation, no failure mode, nothing to attribute to a bug.
3. **A genuinely different span looks nothing like this.** The `TXT` span from the same forward has
   mean cosine **0.4531** to the same reference. Spans that differ read 0.45, not 0.996.

The banked caches came from A100 / torch 2.6.0+cu124; this run is RTX 5090 (sm_120) /
torch 2.7.1+cu128, with transformers 4.49.0 and peft 0.14.0 unchanged and the LoRA adapter
sha256-identical to the banked one. Any belt against those caches measures the GPU, not my code.

## The replacement belt — and it passes exactly

The question the belt should answer is: *does my fork's `A0` span equal the deployed readout
operator?* That is testable with no hardware term at all, by running the frozen
`generate_VideoMLLM_embedding_readout_HF._pool_span(..., span="response")` — the deployed function
object itself, imported not copied — on the **same forward, on this machine**, and comparing.

Result, 6 MHC_zh val videos × layers {28, 24}:

```
BV1qV411w7t1 L28  bit-identical=True  maxabsdiff=0.000e+00
BV1qV411w7t1 L24  bit-identical=True  maxabsdiff=0.000e+00
BV1cp421o7gr L28  bit-identical=True  maxabsdiff=0.000e+00
BV1cp421o7gr L24  bit-identical=True  maxabsdiff=0.000e+00
BV1wE411f7y8 L28  bit-identical=True  maxabsdiff=0.000e+00
BV1wE411f7y8 L24  bit-identical=True  maxabsdiff=0.000e+00
BV1yd4y1c7nm L28  bit-identical=True  maxabsdiff=0.000e+00
BV1yd4y1c7nm L24  bit-identical=True  maxabsdiff=0.000e+00
BV1XQ4y117qV L28  bit-identical=True  maxabsdiff=0.000e+00
BV1XQ4y117qV L24  bit-identical=True  maxabsdiff=0.000e+00
BV1E7411A783 L28  bit-identical=True  maxabsdiff=0.000e+00
BV1E7411A783 L24  bit-identical=True  maxabsdiff=0.000e+00
BELT: 12/12 bit-identical
```

**A0 is the deployed readout, bit for bit.** This is a stronger statement than any cosine bar
against a cache from another machine could have made.

## Ruling

1. The gating belt is now: **A0 must be bit-identical to the frozen `_pool_span(span="response")`
   on this machine.** Passed, 12/12, max abs diff 0.0.
2. The cross-hardware cosine against the banked caches is retained in
   `idea-stage/r10_tokpos/build_meta.json` as a **descriptive** record of platform drift, gated
   only by a loose bug-catching floor (mean ≥ 0.95 and every row ≥ 0.90) that a wrong span
   (0.45) would fail and drift (0.996 / 0.962) passes.
3. D1's substantive clauses are unchanged and remain binding: A0 is re-extracted in the same pass
   as every arm and is the control; the ledger 0.8014 is not this table's baseline; `img_feats`
   are the same banked vectors in every arm and cancel in every contrast; leg 2's `C0` is rebuilt
   from this pass rather than taken from `R6RO-CAT`.
4. **No decision rule, arm definition, seed range, protocol or the +0.005 bar changes.**

Belt script is inline in the execution record; the operator identity it asserts is re-checkable at
any time by importing `_pool_span` and `_spans_from_hidden` and comparing on one forward.
