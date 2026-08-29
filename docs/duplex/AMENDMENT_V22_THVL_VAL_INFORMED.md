# V22 THVL Validation-Informed Amendment

Status: **FROZEN BEFORE V22 GRID EVALUATION**  
Disclosure: **VAL-INFORMED**. THVL validation GT had already been opened for the terminal V21 audit. THVL test GT remains sealed and has not been opened.

## Motivation and fixed boundary

V21 terminated before selection because speech timestamps cover only part of long videos and packed logits were not numerically identical to sequential logits. V22 changes only the treatment of speech support and the packed-fidelity activation criterion. It does not change the taxonomy, prompt, model, raw scores, split, target, or test discipline.

## Scoped GT

- A frame is positive if it overlaps a `Hate`, `Bias`, or `Verbal Abuse` segment.
- A frame overlapping only another harm leaf is ignored.
- A frame outside every annotated harm segment is background negative.
- Positive takes precedence over ignored when annotations overlap.
- The grid is evaluated on the frozen 1-Hz media-duration timeline.

## Frozen score construction

For each video, global raw score is the duration-weighted mean of its valid `causal_continuous` chunk margins. Convert video global scores to a bounded robust reference using the tie-aware validation mid-ECDF, then subtract `0.5`:

`G_v = midECDF_val(global_raw_v) - 0.5`.

For the local stream, project `masked_branch_reset` chunk margins to frames by overlap-weighted mean. Frames with no valid speech-chunk overlap receive local correction exactly zero and are not treated as negative local evidence. On covered frames only, subtract that video's covered-frame mean. Divide all covered centered values by their validation RMS; uncovered values remain exact zero. Zero ECDF/RMS support fails closed.

The frozen grid is:

`S_v(t) = alpha G_v + beta L_v(t)`, with `alpha,beta ∈ {0,.25,.5,1,2}` except `(0,0)`.

Mandatory reports are global-only `(1,0)`, local-only `(0,1)`, and equal `(1,1)`. Selection maximizes pooled Frame AP, breaking ties by pooled ROC, within-video macro AP, smaller `alpha+beta`, then lexicographic `(alpha,beta)`. Metrics ignore other-harm frames.

## Packed-fidelity amendment

Absolute packed/sequential error is report-only. Activation requires:

1. Spearman correlation between packed and sequential masked-reset chunk margins at least `.99` on the frozen reference subset.
2. A score–metric fidelity audit on that same subset. Independently convert packed and sequential chunk margins to tie-aware mid-ECDF values, project each to the frozen speech support, center covered frames, and compute pooled AP/ROC plus within macro AP/ROC under the same scoped GT. The maximum absolute packed-versus-sequential metric delta must be at most `.01`.

No threshold, affine calibration, prompt, subset, or taxonomy item may be chosen from this audit.

## Controls and activation

- Within-video time shuffle: `B=200`, permuting covered local values within each video while preserving the zero uncovered mask. Report 97.5th percentiles.
- Paired source-group cluster bootstrap: `B=2000`, full minus global-only, fixed seed `22022`, percentile 95% intervals.

Test opening is authorized only if all hold:

- packed Spearman >= `.99` and maximum metric delta <= `.01`;
- selected pooled AP minus global-only AP >= `.01`;
- selected pooled ROC minus global-only ROC >= `-.005`;
- bootstrap AP lower 95% bound >= `0`;
- selected within macro ROC >= `.55` and exceeds the shuffle 97.5th percentile;
- selected within macro AP is no lower than global-only.

ASR video/time coverage is reported but is not an activation gate in V22. If any activation condition fails, freeze the negative validation result and keep test sealed. No further V22 configuration is selected.

