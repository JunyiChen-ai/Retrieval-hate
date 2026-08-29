# Baseline reproduction performance archive — 2026-08-22

## Scope and supervision convention

All headline numbers below use the common **test split** and are reported as
**frame ROC-AUC / frame PR-AUC**. They are target-frame-label-free at inference,
but they are **not all strictly label-free**:

- `label-free`: no target-dataset labels are used (ZS-CLIP, ImageBind, Qwen2.5-VL,
  LAVAD, URF-HVAA, AV²A, T3AL).
- `unlabelled`: trains/adapts on unlabelled target features (CLAP).
- `one-class`: uses the target dataset's normal-only training partition (MULDE).
- `aux-temporal-pretrain`: uses a checkpoint trained with temporal supervision on
  an external corpus (LaGoVAD, UniTime, SeViLA).

Thus, the campaign is best described as **no target frame-level hateful-event
supervision**, rather than saying every method is strictly label-free.

## Headline test performance

Pre-registered/main variants only; no post-hoc best-variant selection. T3AL is
mean ± SD over three seeds; its table cells below show the mean for compactness.

| Method | Supervision | HateMM | MHC-EN | MHC-ZH | HateClipSeg | Mean ROC / PR |
|---|---|---:|---:|---:|---:|---:|
| ZS-CLIP | label-free | .537/.278 | .501/.268 | .608/.341 | .499/.461 | .536/.337 |
| ImageBind-image | label-free | .592/.314 | .594/.329 | .598/.358 | .593/.554 | .594/.389 |
| ImageBind-video | label-free | .591/.310 | .564/.306 | .573/.355 | .581/.543 | .577/.378 |
| ImageBind-audio | label-free | .565/.291 | .616/.368 | .653/.396 | .565/.512 | .600/.392 |
| Qwen2.5-VL grounding | label-free | .519/.252 | .522/.281 | .511/.270 | .503/.475 | .514/.319 |
| LAVAD | label-free | .559/.291 | .556/.311 | .492/.263 | .577/.546 | .546/.353 |
| URF-HVAA | label-free | .574/.318 | .549/.297 | .545/.287 | .586/.553 | .564/.364 |
| LaGoVAD | aux-temporal-pretrain | .558/.305 | .524/.262 | .597/.312 | .500/.467 | .545/.336 |
| AV²A | label-free | .539/.252 | .531/.322 | .560/.321 | .486/.468 | .529/.341 |
| UniTime | aux-temporal-pretrain | .478/.235 | .495/.272 | .488/.260 | .453/.455 | .479/.305 |
| MULDE | one-class | .600/.309 | .487/.259 | .513/.250 | .533/.500 | .533/.329 |
| CLAP | unlabelled | .586/.352 | .494/.279 | .328/.189 | .471/.453 | .470/.318 |
| T3AL | label-free | .627/.305 | .568/.314 | .720/.440 | .636/.567 | **.638/.406** |
| SeViLA Localizer | aux-temporal-pretrain | .627/.332 | **.629/.330** | .670/.386 | .576/.561 | .626/.402 |

## Findings

- T3AL has the best four-dataset mean: ROC-AUC `.638`, PR-AUC `.406`.
- SeViLA is second overall (`.626/.402`) and has the best main-variant MHC-EN ROC.
- ImageBind is the strongest simple zero-shot family; audio is strongest on MHC,
  while image/video are more stable on HateClipSeg.
- HateMM: T3AL/SeViLA tie on ROC (`.627`); CLAP has the best PR (`.352`).
- MHC-ZH and HateClipSeg: T3AL is the strongest headline method.
- UniTime and CLAP are below random ROC on average after transplantation; CLAP's
  degradation is especially severe on MHC-ZH.
- PR-AUC must be read with the dataset base rate in mind, particularly HateClipSeg;
  cross-dataset PR values are not directly comparable as a single difficulty scale.

## Completion and evidence

All 14 headline baselines above now have common-evaluator test metrics. UniTime
finished with 3,733 successful generations and 21 recorded errors. SeViLA produced
643 successful video outputs and one failure. T3AL completed all three seeds.

Evidence files:

- Master machine-readable table: `idea-stage/repro_campaign/summary_test.csv`
- UniTime evaluation: `idea-stage/repro_campaign/eval_UniTime_test.json`
- SeViLA evaluation: `idea-stage/repro_campaign/eval_SeViLA Localizer_test.json`
- T3AL three-seed aggregate: `idea-stage/repro_t3al/eval/test_agg.json`
- Full method report: `idea-stage/REPRO_CAMPAIGN_RESULTS.md`

The master CSV contains T3AL as `not run` because the generic table builder does
not ingest T3AL's separate three-seed aggregate schema. The authoritative T3AL
numbers in this archive come directly from `test_agg.json`.
