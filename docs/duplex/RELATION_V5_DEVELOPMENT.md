# Relation-V5 hierarchical performance record

V5 explicitly decomposes every calibrated expert score as

`r_e(v,t) = mean_t r_e(v,t) + (r_e(v,t) - mean_t r_e(v,t))`.

It independently selects a video-prior simplex, a centered-residual simplex,
and a residual amplitude. The fixed grids are simplex step 0.1 and amplitude
`[0,.25,.5,.75,1,1.25,1.5,2]`. Selection maximizes validation pooled Frame AP
subject to ROC being no lower than the validation AP-best expert's ROC. Expert,
static, and frozen V4 outputs are included as fallbacks. Test is not opened
during selection.

## HateClipSeg

| Pool / candidate | Validation AP/ROC | Test AP/ROC |
|---|---:|---:|
| VERA + VAD, best hierarchy: prior `1/0`, residual `.7/.3`, amp `1.25` | `.65165/.66719` | `.64132/.60284` |
| VERA + VAD + Fed1, best hierarchy: prior `1/0/0`, residual `.5/.3/.2`, amp `1.5` | `.65221/.66682` | `.63800/.60790` |
| frozen V4 fallback, formally selected in both pools | `.65634/.68006` | `.63885/.60977` |

The three-expert run enumerated all 34,848 hierarchical configurations plus
fallbacks. Fed1 receives residual weight on validation but does not transfer to
test. Neither hierarchy reaches the `.6494` target. Static V5 development is
therefore stopped; the next performance stage requires a train-only reliability
model using train sparse K16 supervision and frozen VAD/Fed train dense scores.
