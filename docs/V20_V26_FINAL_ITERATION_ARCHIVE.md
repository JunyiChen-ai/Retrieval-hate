# Weakly supervised hateful-video localization: V20--V26 archive

Status: **V26 iteration complete; stop after this iteration** (2026-08-30).

This archive separates pooled frame metrics from within-video temporal ordering.
The latter is the decisive localization evidence. Test-informed development is
explicitly disclosed; V26 did not open test.

## Frozen baseline and V20-chain performance

| Corpus | Frozen baseline AP / ROC | Best V19/V20 chain AP / ROC | AP gain |
|---|---:|---:|---:|
| HateMM | .573299 / .806769 | .650886 / .844059 | +.077587 |
| MHClip-EN | .451900 / .727200 | .593155 / .791056 | +.141255 |
| MHClip-ZH | .461400 / .752100 | .521450 / .796731 | +.060050 |
| HateClipSeg | .619371 / .605020 | .664956 / .613014 | +.045585 |

The V20-chain increment relative to its immediate identity backbone was much
smaller: HateMM +.0060 AP, MHClip-EN +.0339, MHClip-ZH +.0029, and HCS +0.
Only MHClip-EN selected a nonzero local term and improved within-video AP/ROC
by about +.042/+.042. The other selected gains were video-prior effects or
exact fallback. External method review rated this line 5/10, weak reject:
the local module was load-bearing on only one corpus.

## Iteration ledger

| Iteration | Main idea | Outcome | Decisive evidence |
|---|---|---|---|
| V20 | Frozen consensus plus decomposed global prior and centered local ASR judge | Partial success | Strong pooled metrics, but genuine local gain only on MHClip-EN |
| V21 | Add synchronized center frame/OCR to the local judge | Rejected | HateMM small uncertain gain; HCS within AP -0.0470 and within ROC -0.0091 |
| V22 | Repair partial speech support and packed-fidelity activation | Did not establish a reusable locator | Validation-informed diagnostic only; test stayed sealed |
| V23 | Synchronized multimodal 30 s windows | Did not establish robust temporal ordering | Preserved as a development artifact, not a confirmed gain |
| V24 | Train-weak small fusion/calibrator | Failed | Formal selection returned epoch0/global fallback |
| V25 | Negative-reference density-ratio MIL | Failed formal validation | Selected development epoch2 was not statistically better than fallback/reference and temporal CI crossed zero |
| V26 | Counterfactual Temporal Witnesses (CTW): exact per-second replacement effects around a finite receptive field | Failed; iteration stopped | Very strong video classification, but no localization gain and negative-reference control was not necessary |

## V25 authoritative validation diagnostics

| Quantity | Value |
|---|---:|
| Real epoch2 video AP / ROC | .766394 / .785425 |
| Permuted control AP / ROC | .725280 / .750337 |
| Negative-reference-only AP / ROC | .756941 / .780027 |
| Epoch0/global AP / ROC | .708006 / .728745 |
| Frame AP / ROC | .275313 / .679356 |
| Within-video macro AP / ROC | .345356 / .566782 |
| Shuffled within ROC | .487198 |
| Paired within-ROC CI | [-.061121, .198219] |

V25 therefore froze the exact global fallback and did not authorize test.

## V26 method and execution

V26 consumes frozen 1 Hz visual (CLIP), audio (VGGish), and text (BERT)
features. A negative-only decoder supplies an out-of-fold counterfactual
background for every training video. A four-layer dilated temporal network
(receptive-field radius 15 s) predicts video evidence. The local output at
second `t` is the exact change caused by replacing that second's available
multimodal features with its negative reference and locally recomputing the
finite receptive field. Training uses only train video labels.

The kill pilot used seed 234, 314 videos, 115,088 seconds, 8 epochs, batch size
4, and three matched arms: real, permuted, and zero/negative-mean replacement.
Each arm completed 632 optimizer steps from the same epoch0 state.

The first exact-length permutation control was structurally unidentifiable:
222 distinct lengths and 144 singleton-length groups allowed only 54.14% of
videos / 42.98% of seconds to move. Those partial runs are quarantined. The
final control uses a canonical cyclic derangement: recipient `id/G/y` stays,
while the donor's complete variable-length `(T, X, masks, own-OOF-background)`
tuple moves without interpolation. Full preflight proved 100% nonself coverage,
donor bijection, and preservation of length, availability, and tuple multisets.
An independent fundamental code review returned PASS.

## V26 video-level validation (selected epoch 8)

| Arm | Video AP | Video ROC |
|---|---:|---:|
| Epoch0 frozen global | .708006 | .728745 |
| CTW real | **.888242** | **.882591** |
| Permuted | .677461 | .672065 |
| Negative-mean | .869906 | .878543 |

Real CTW gained .180236 AP over epoch0, paired 95% CI
[.022445, .381401], and strongly beat the permuted control. It beat the
negative-mean arm by only .018336 AP with CI [-.039394, .098337]. Therefore
the formal video gate failed: the learned negative reference was not shown to
be necessary, despite excellent video discrimination.

## V26 temporal validation (selected epoch 8)

| Quantity | Value |
|---|---:|
| Frame AP / ROC | .191365 / .709498 |
| Within-video macro AP / ROC | .279882 / .559938 |
| Within ROC 95% CI | [.462730, .668412] |
| CTW minus V25 within ROC | -.010671, CI [-.175633, .149745] |
| CTW minus V25 within AP | -.065510, CI [-.163444, .019977] |
| Constant-G within AP | .246046 |
| CTW gain over constant-G within AP | +.033837, CI [-.010066, .091252] |
| Faithfulness top-minus-random deletion | .123802, CI [-.207960, .474470] |
| Real-minus-shuffle paired CI | [-.032173, .164076] |
| Coverage / videos with nonzero variance | 1.0 / 1.0 |

Only coverage and variance gates passed. Within ROC, both within-AP gates,
V25 comparison, shuffle, faithfulness, and duration gates failed. The signed
report has `all_gates_pass=false` and `test_opened=false`.

## Final research conclusion

V26 supplies a useful negative result, not a SOTA candidate. The finite-RF
counterfactual mechanism and label/feature pairing clearly learn video-level
hatefulness (the derangement control collapses), but the per-second replacement
effect is not aligned with annotated hateful intervals. The zero-reference arm
nearly matches the learned-reference arm, and CTW is worse than V25 on both
within-video AP and ROC. Thus the bottleneck is not video discrimination; it is
an objective that makes temporal ordering identifiable from weak labels.

Do not extend V26 to more seeds/corpora or test. A future iteration should begin
from a strong video-level backbone but introduce a genuinely temporal learning
signal/constraint whose success is judged first by within-video AP/ROC and
shuffle contrast, not pooled Frame AP or video AP.

## Authoritative artifacts

- Training manifest: `results/steward_private/thvl_bench/train314/v26_finite_rf_seed234_killpilot_v3/manifest.json`
  (`6208296832c2d21ba8c589101bdc99959e6fd55f496a1c3fff007e74cdeece0f`)
- Permutation: `results/steward_private/thvl_bench/train314/v26_finite_rf_seed234_killpilot_v3/permuted/permutation.json`
  (`5f5ded6405e3e2739b9906c6ba61b5b7758c506e252002b9c567ac50624d4c9f`)
- Prediction manifest: `results/steward_private/thvl_bench/v26_val_predictions_seed234_v3/manifest.json`
  (`6d4b28931165d124b259c942a05ce5cccfab3ed376bab21c3a6cce6b587743b9`)
- Signed video selection: `results/steward_private/thvl_bench/v26_video_selection_seed234_v3.json`
  (`2751bfb3896c8b118d43c234cfb057c4251570ac60ede6b0892e795db39c34bf`)
- Signed temporal report: `results/steward_private/thvl_bench/v26_temporal_report_seed234_v3.json`
  (`b7c2d95affde733d746472cb771f1fe1b14a82c19e198f32b168675e448f1d9c`)
