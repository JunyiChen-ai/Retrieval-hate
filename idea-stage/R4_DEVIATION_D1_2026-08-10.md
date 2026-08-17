# Round-4 deviation D1 — the frozen R4-1 null is misspecified (found in smoke, before any primary run)

**Status: raised BEFORE the primary run, by the smoke test that exists to catch exactly this.**
No primary result has been produced. Disclosure of partial unblinding is in §4.

## 1. What the freeze says

`idea-stage/R4_PILOT_FREEZE_2026-08-10.md`, pilot R4-1, "Null (frozen)":

> Within each split and each hard-label stratum, independently permute every **non-reference**
> encoder's logit rows while holding the validation-best reference encoder fixed; refit the identical
> lattice. **This preserves each encoder's class-conditional score distribution and ROC while
> destroying item-level complementarity.**

The stated *intent* is to destroy item-level complementarity while holding each member's marginal
quality fixed, so that `Null95` measures how much lattice gain is obtainable with no real
complementarity present.

## 2. What it actually does — the opposite

Within-label permutation preserves each encoder's class-conditional distribution (as claimed) but it
also makes the permuted encoder's errors **conditionally independent of the reference encoder's,
given the label**. Conditional independence given the label is the *best possible* case for score
combination, not the worst. The permutation therefore **manufactures idealised complementarity**
rather than removing it.

Minimal demonstration (synthetic, two encoders with realistically correlated errors, n = 2000):

| | encoder A ROC | encoder B ROC | mean-of-two ROC |
|---|---|---|---|
| real (correlated errors) | 0.7981 | 0.7885 | **0.8266** |
| B permuted within label | 0.7981 | 0.7885 (unchanged, as the freeze predicts) | **0.8785** |

The ensemble gets **+0.052 ROC better** under the null. Confirmed on real data in a 2-rep smoke on
MHC-ZH: null `DeltaROC` values were **+0.0549 and +0.0692**, versus a primary-run `DeltaROC` of
`−0.0017` for the same cell/seed.

## 3. Why this is a blocking defect and not a cosmetic one

`Null95` is the 95th percentile of `max(0, MeanDeltaROC_null)`. Under the misspecified null it will
land somewhere around **+0.05 or higher**, so freeze clause 2 —

> `MeanDeltaROC >= 3 * Null95`

— demands `MeanDeltaROC >= ~0.15`, i.e. a 15-point ROC gain from an ensemble head. **No mechanism of
this kind can pass that.** The rule as written guarantees a KILL independent of the mechanism's
merit, which is a **false-KILL generator**.

This is the one category of defect the project's own process rules say must block: *"only defects
that would produce a wrong verdict / touch the test set may block"*. A rule that returns KILL for
every possible input produces a wrong verdict by construction and destroys the round's information
value. It is the mirror image of the R3-2 literal-rule note (where the literal rule was applied
unchanged **because** both readings failed and the verdict could not turn on it — here the verdict
turns on it entirely).

## 4. Disclosure — partial unblinding

The smoke run was executed on **MHC-ZH, seed 0 only**, and printed primary numbers before the defect
was identified. I have therefore seen:

- MHC-ZH seed 0 per-method test ROC / macro-F1 (single 0.8707/0.7712, mean-logit 0.9122/0.7890,
  mean-prob 0.9128/0.8156, weighted 0.9126/0.7980, logistic 0.9141/0.8039, MLP 0.9154/0.7997,
  MDL 0.9109/0.8089), frozen comparator `weighted`, `DeltaROC −0.0017`, `DeltaF1 +0.0110`.

That is 1 of 12 (dataset × seed) primary cells. It is recorded here so that no later claim can be
made that the round was fully blind. **No threshold, bar, comparator rule, or mechanism definition
has been altered, and none will be as a result of having seen it.** The re-specification requested
below concerns the null only, and the replacement is being chosen by the cross-model jury, not by the
executor.

## 5. What is requested

The **jury**, not the executor, re-specifies the null for R4-1 clause 2. Everything else in the
freeze — scope, model, comparators, clauses 1/3/4, the GO/KILL structure, the confirmation
interpretation — stands unchanged.

## 6. Jury ruling

Ruling delivered and adopted in full — see `idea-stage/R4_DEVIATION_D1_RULING.md` and the AMENDMENT appended to `idea-stage/R4_PILOT_FREEZE_2026-08-10.md`. Summary: (1) the null IS a blocking false-KILL generator; (2) no permutation can hold the required properties simultaneously, so clause 2 becomes `MeanDeltaROC >= +0.010 AND LCB95 > 0` under a paired stratified joint-row bootstrap (10,000 reps, rng 20260810); (3) the round survives the partial unblinding with disclosure, no restart.
