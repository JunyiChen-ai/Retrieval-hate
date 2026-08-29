# Relation-V4 performance-first development record

## Method frozen for the current pilot

For one corpus, each expert produces a dense score `s_e(t)`. No test statistic
is used to normalize it. Expert `e` is calibrated by the empirical CDF fitted
only on validation scores:

`r_e(t) = ECDF_e,val(s_e(t))`.

The validation-selected static lower bound is

`z_static(t) = sum_e w_e r_e(t)`.

V4 treats expert agreement and disagreement as open evidence rather than
assigning frames to a closed semantic ontology. For target expert `q`, source
expert `e`, and local times `t,s`, it builds directed transport

`A_qe(t,s) = softmax_s(-(Delta r_q(t)-Delta r_e(s))^2 / tau)`,

restricted to `|t-s| <= W`, and obtains the aligned endpoint

`a_qe(t) = sum_s A_qe(t,s) r_e(s)`.

Cross-expert support and the per-frame gate are

`u_q(t) = -sum_e w_e |r_q(t)-a_qe(t)|`,

`g_q(t) = softmax_q(log w_q + beta u_q(t))`.

The gated and transported scores are

`z_gate(t) = sum_q g_q(t) r_q(t)`,

`z_transport(t) = sum_qe w_q w_e a_qe(t)`,

and the final rank correction is

`z(t) = z_gate(t) + gamma (z_transport(t)-z_gate(t))`.

`beta=gamma=0` is bit-exact static fusion. `beta` and `gamma` are selected only
on validation. A relation configuration may replace the static fallback only
if both validation Frame AP and Frame ROC are non-decreasing. Test is opened
only after this configuration is frozen.

## Seed-ensemble pilot results

All expert raw scores are averaged over seeds 234, 2025, and 3407 before the
validation-frozen ECDF is applied.

| Corpus | Variant selected on validation | Test Frame AP | Test ROC |
|---|---:|---:|---:|
| MHC-EN | full transport, beta=-2, gamma=0.5 | 0.49007 | 0.73698 |
| MHC-ZH | full transport, beta=-2, gamma=0.5 | 0.50120 | 0.78285 |
| HateClipSeg | full transport, beta=8, gamma=-0.5 | 0.63885 | 0.60977 |

For MHC-EN, the fixed two-expert pool is MACIL-SD audio and CMHKF. The complete
`0.00:0.01:1.00` audio-weight grid was selected by maximum validation Frame AP
with ROC used only as an exact tie-break. It selected audio/CMHKF weights
`0.02/0.98`; the full archived grid is in
`results/reproduction/relation_v4/final_mhclip_en_a02_provenance/weight_grid.json`.
With those immutable weights, static fusion gives `0.4808209/0.7331733` on
test. Validation then selects `beta=-2, gamma=0.5`, raising test performance to
`0.4900735/0.7369793`. This exceeds the preregistered `0.4819` AP target and is
`0.0382` above the `0.4519` reference.

MHC-ZH isolates the source of the gain:

| Ablation | Test Frame AP | Test ROC | AP delta vs static |
|---|---:|---:|---:|
| static three-expert fusion | 0.4863567 | 0.7718822 | -- |
| endpoint support gate only, beta held at -2 | 0.4879108 | 0.7753328 | +0.0015541 |
| transport only, gamma held at 0.5 | 0.4991352 | 0.7798624 | +0.0127785 |
| full, beta=-2 and gamma=0.5 | 0.5012044 | 0.7828548 | +0.0148477 |

The earlier `0.48597/0.77153` gate-only number came from a separately selected
beta and is retained only as development history; it is not the controlled
ablation. Holding the full model's beta at -2 shows a small gate contribution,
whereas transport propagation accounts for most of the gain. Thus the ZH
improvement is not attributable to adding a static ensemble or a generic
per-frame gate. Relative to the 0.4614 MACIL-SD AV reference, full V4 improves
Frame AP by 0.0398.

## Pending

For HateClipSeg, the fixed VERA/VADCLIP pool and complete 0.01 validation grid
select weights `0.99/0.01` (not the anticipated `1.00/0.00`). VERA contains
many tied scores, so the small VADCLIP contribution acts as a validation-chosen
tie breaker. Static test performance is `0.6360232/0.5965253`; the unchanged
V4 grid selects `beta=8, gamma=-0.5` and reaches `0.6388459/0.6097738`. This is
below the `0.6494` target, so the grid was not expanded.

The archived shift diagnostic separates video means from within-video
residuals. Applying only V4's video-mean correction to the static residual
gives `0.6366913` AP, while applying only V4's centered residual to the static
video mean gives `0.6338248`. Thus the current within-video correction does not
transfer; its gain depends on interaction with a small video-prior change.
Independent expert decomposition likewise finds the unattainable test-only
fusion upper bound is driven mainly by cross-video prior rather than local
boundary ranking. The next HCS design must explicitly separate a video-prior
gate from a centered within-video locator; more beta/gamma search is not
justified.

The identical frozen grid remains pending only for HateMM validation dense
scores. No grid expansion is planned.
