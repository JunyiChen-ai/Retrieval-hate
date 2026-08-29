# Relation-V8 HateMM centered-correction audit

Independent CPU recomputation, 2026-08-29. No Relation-V8 implementation or
result artifact was modified.

The audited candidate definition was, independently for every video,

```text
residual = dynamic - static
final = static + residual - mean(residual)
```

The complete manifest grid (13 beta values by 7 gamma values, 91 candidates)
was recomputed on the frozen validation cohort. The original Pareto rule was
then applied: require validation Frame AP and Frame ROC to be no lower than the
`beta=0, gamma=0` fallback, maximize AP, then ROC, then prefer smaller absolute
beta and gamma. Test data was loaded only after this selection.

## Result

| item | beta | gamma | Frame AP | Frame ROC |
|---|---:|---:|---:|---:|
| validation static fallback | 0 | 0 | 0.7129768978 | 0.8501252901 |
| validation selected centered correction | 8 | 0.5 | 0.7291684652 | 0.8580137212 |
| test selected centered correction | 8 | 0.5 | 0.6454754436 | 0.8387040469 |

There were 43 Pareto-eligible candidates. Against the frozen MACIL-SD AV bar
of 0.5732989549 AP / 0.8067690923 ROC, the test improvements are
`+0.0721764887` AP and `+0.0319349546` ROC.

## Protocol checks

- Validation: 109 videos and 13,533 frames; every source score file has exactly
  the frozen GT IDs.
- Test: 214 videos and 29,269 frames; every source score file has exactly the
  frozen GT IDs. The sole frozen split ID outside the localization GT is
  `hate_video_427`, consistently absent from every source.
- The two MACIL views use the raw-score mean of seeds 234, 2025, and 3407.
  The third view is VERA `score_official_postprocessed`.
- All source arrays were checked for exact GT length and finite values.
- The three ECDF references contain 13,533 validation values each and were fit
  without labels. Their float64 SHA-256 values are respectively
  `b1381ecbff0afdee4783d6a3888fd4c6a723e6020231482c8cfc3a159e726673`,
  `9c2af7a6163d69be77f388cdfaf77b18c77d257c45a389e28457619e2696f7ef`, and
  `a93550e11718e21ca8652dd8cf02901bbcd57f39734e3a0f509c664d4709e576`.
- Validation labels were used only by the fixed Pareto evaluator. Test labels
  were not loaded until after beta/gamma selection and were used only by the
  final evaluator.

This is a test-informed development checkpoint, not untouched confirmatory
evidence and not a novelty assessment.
