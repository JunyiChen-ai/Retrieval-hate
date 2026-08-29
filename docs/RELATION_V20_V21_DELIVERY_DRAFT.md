# Relation V20/V21 delivery draft

Status: internal method hand-off, 2026-08-29. This document does not turn
test-informed development artifacts into confirmatory results.

## Unified method specification

The deployable score starts from a frozen, same-corpus expert consensus
`b_v(t)`. Expert scores are calibrated with reference-split ECDFs and combined
without using test labels. The residual method separates two relations:

`s_v(t) = b_v(t) + alpha * g_v + beta * l_v(t)`

`g_v` is a cross-video prior correction: the ECDF-calibrated mean judgment of a
global causal ASR branch minus the ECDF-calibrated video mean of the frozen
consensus. `l_v(t)` is a within-video locator: an isolated ASR branch score is
assigned to its synchronized time span, centered to zero mean inside the video,
and scaled by a validation-reference RMS. All timestamp branches share one
judge policy. In the packed implementation, branches have isolated attention
and reset positions; a global branch is causal. `alpha=beta=0` is the exact
identity fallback and is always a validation candidate.

V20 changes the judge language for MHClip-ZH to native Chinese while retaining
the same binary decision and algebra. V21 asks whether the isolated local branch
should additionally see one synchronized center frame and the nearest OCR
window. Because Qwen3-VL dynamically expands visual tokens and uses 3-axis
mRoPE, V21 was evaluated sequentially first; multimodal packing was not claimed
or implemented without an exact sequential-equivalence proof.

## Load-bearing modules

1. **Frozen expert consensus (`b`)** is the performance-bearing backbone on all
   corpora. Removing it is not a valid ablation of the delivered method; it is a
   different model.
2. **Cross-video prior (`g`)** is the only selected correction on HateMM and
   MHClip-ZH. It improves pooled frame metrics but cannot improve any exactly
   constant-invariant within-video ranking metric.
3. **Isolated locator (`l`)** is selected only on MHClip-EN. It is the only
   component that improves within-video ordering in the current chain.
4. **Identity fallback** is load-bearing for safe generalization: HateClipSeg
   selects exact fallback, and all nonzero HateClipSeg local variants must be
   treated as failed ablations.
5. **Native-language policy** matters operationally for MHClip-ZH provenance,
   but the selected V20 result still has `beta=0`; it supplies no evidence that
   the Chinese local locator works.
6. **V21 frame/OCR input is not load-bearing.** Its 12-video evidence is weakly
   positive on HateMM and negative for HCS within-video localization.

## Incremental results

The machine-readable companion is
`results/reproduction/relation_v21/v20_v21_delivery_table.json`. Frame AP/ROC
are pooled 1-fps metrics. “Identity” below is the frozen consensus used by the
V19/V20 chain, not necessarily the older V8 equal-transport score.

| Corpus | V8 equal transport AP / ROC | Chain identity AP / ROC | Selected V20-chain AP / ROC | Increment | Selected alpha,beta | Within macro AP / ROC increment |
|---|---:|---:|---:|---:|---:|---:|
| HateMM | .6360 / .8331 | .6449 / .8398 | .6509 / .8441 | +.0060 / +.0042 | .025, 0 | 0 / 0 |
| MHClip-EN | .5277 / .7710 | .5592 / .7867 | .5932 / .7911 | +.0339 / +.0043 | .20, .01 | +.0421 / +.0420 |
| MHClip-ZH | .5061 / .7896 | .5186 / .7917 | .5214 / .7967 | +.0029 / +.0050 | .025, 0 | 0 / 0 |
| HateClipSeg | .6662 / .6140 | .6650 / .6130 | .6650 / .6130 | 0 / 0 | 0, 0 | 0 / 0 |

V21 is a val-only information gate, not an increment to this test table. On 12
mechanically selected validation videos, HM multimodal minus ASR-only within
macro ROC/AP was `+.0307/+.0331` with only four eligible mixed videos and both
95% bootstrap intervals crossing zero. HCS was `-.0091/-.0470` over eleven
eligible videos. HCS pooled AP/ROC improved, which is consistent with a stronger
video prior, not a stronger locator.

## Negative ablations and controls

- HateClipSeg validation selected `alpha=beta=0`; no relation correction beat
  the identity under the non-regression rule.
- MHClip-ZH V20 rejected every `beta>0` grid point. The selected `beta=0` makes
  time shuffle exactly invariant and leaves stable centered metrics identical.
- HateMM selected `beta=0`; its gain is prior-only and supplies no local-evidence
  claim.
- V21 HCS multimodal input reduced within macro AP by 4.70 points. This is a
  negative ablation, not a hidden success based on its pooled gain.
- V21 HM bootstrap intervals cross zero. It motivates no full-validation or
  packed implementation by itself.
- Duplicate/noise-expert invariance is not a property of the V20 judge layer;
  it belongs, where verified, to the frozen consensus implementation.
- A text-only 4D attention mask is insufficient evidence for multimodal packed
  fidelity because visual-token expansion and mRoPE indexing differ.

## Disclosure and limitations

The V19/V20 line is explicitly a **test-informed development version**: test
labels were not used to select `alpha,beta`, but test diagnostics from earlier
iterations influenced the research path. Report it as such until a fresh held-
out corpus or untouched split confirms it. V21 did not access test at all.

Known limitations are small mixed-video counts (especially ZH and the HM V21
pilot), corpus-dependent expert pools, ASR dependence, coarse/long spans that
inflate pooled localization metrics, no stable local gain on three of four
corpora, no multimodal packed-fidelity proof, OCR noise, one-frame sampling, and
the distinction between better video discrimination and better temporal
ordering. The honest current story is: robust consensus plus a safe decomposed
prior/locator interface, with convincing local gain only on MHClip-EN.

## DeHate first pilot

The frozen proposal is in
`results/reproduction/relation_v21/dehate_pilot_frozen_proposal.json`. It is a
four-video, validation-only, sequential ASR-versus-frame/OCR/ASR information
gate. Cohort selection is label-free and mechanical. No DeHate temporal GT,
hate class, explicit/implicit label, target group, or modality-contribution
label may be opened before raw scores and hashes are frozen. Because raw DeHate
media is gated and is not currently registered in this repository, the proposal
is blocked at media provenance rather than silently substituting annotation
rows or another split.
