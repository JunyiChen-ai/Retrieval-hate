# POWA corpus-independent starting-point audit

Date: 2026-08-31.

Purpose: freeze the actual POWA-MACIL starting point under the current rule
that each corpus is trained independently. This is an audit, not a new method.
It re-evaluates the existing corpus-specific, three-seed dense score files with
the repository's single shared evaluator on the 1 fps test grid.

The earlier HateClipSeg headline POWA row came from a four-corpus joint model
and is ineligible under the current protocol. This audit replaces it with the
three corpus-specific HateClipSeg runs initialized from corpus-specific
MACIL-SD checkpoints. HateMM, MHC-EN, and MHC-ZH already use corpus-specific
runs.

Run:

```bash
bash experiments/20260831_powa_starting_point/evaluate.sh
```

Artifacts are written to
`runs/20260831_powa_starting_point/<corpus>_seed<seed>/`. Each run contains the
training config snapshot, source-score path, code commit, log, PID, and the
shared evaluator's `metrics.json`. The aggregate summary is generated only
after all 12 evaluator outputs exist.

No training data or test labels are modified. No cross-corpus model is used.

The archived HCS-only checkpoints predate the masked-Sinkhorn correction. A
current-code three-seed HCS-only reproduction is therefore run separately:

```bash
bash experiments/20260831_powa_starting_point/rerun_hcs_maskfix.sh
```

This uses only HateClipSeg train/video labels, its validation split for
checkpoint selection, and its matching corpus-specific MACIL-SD initialization.

## Result

Authoritative aggregate: `runs/20260831_powa_starting_point/summary.json`;
per-seed authority is each run's evaluator-written `metrics.json`.

| Corpus | pooled AP | pooled ROC | within-video ROC |
|---|---:|---:|---:|
| HateMM | .593832 | .816184 | .590457 |
| MHC-EN | .468906 | .747812 | .576223 |
| MHC-ZH | .506032 | .766274 | .432221 |
| HateClipSeg, current-code corpus-only rerun | .577550 | .535191 | .521148 |

The HCS mask-fixed rerun is effectively identical to the archived HCS-only
reference, so the correction itself does not recover the invalid joint row.
Under the independent-corpus protocol, POWA holds the pooled AP/ROC reproduced
table leads on HateMM, MHC-EN, and MHC-ZH, but none of the four within-video
ROC leads and none of the three HateClipSeg leads.

## Starting-point risks that the next iteration must not hide

- Do not reuse the four-corpus HCS checkpoint or LOCO-ST cross-corpus spans.
- Do not choose a branch, dataset-specific regime, or hyperparameter on test.
- Do not claim ensemble, calibration, input replacement, or a video-prior gain
  as localization progress.
- POWA training pads or uniformly subsamples to 200 rows while attention is
  unmasked and inference uses full length; this is a reliability issue, not a
  novelty claim.
- The two sparse teacher chunks are endpoint samples, which can reinforce the
  diagnosed MHC-ZH edge bias; this must be controlled in mechanism analysis.
