# UOT normal-reference witness pilot：独立 post-run audit

日期：2026-08-31  
范围：`runs/20260831_uot_normal_reference_witness/pilot_seed234/{hatemm,hateclipseg}/`
的 config、training completion、checkpoint/history、predictions、metrics/evaluation log，以及冻结的
protocol、train/evaluate 结果链。  
结果链结论：**PASS。** 两个 run 完整且独立复算一致；实验结论为 **FAIL**，SOTA gate 与机制
gate 均未通过，按冻结计划应淘汰本轮且不扩展 MHC-EN/ZH。

## Config、completion 与 checkpoint selection

- 两个 config 均为 corpus-specific seed 234、60 epochs、batch 4、LR `2e-4`、hidden 128、embed
  32、8 atoms、temperature `.1`、reject cost `1`、8 transport steps、pool power 8、temporal
  weight `.1`、negative-cost weight `.2`；其余 frozen split/feature/model 描述和可读输入路径均与
  pre-run protocol 一致。
- HateMM `training_complete.json` 为 `prediction_complete`、60 epochs、selected epoch 57、214
  test videos；HateClipSeg 为 `prediction_complete`、60 epochs、selected epoch 26、79 test videos。
  两份 run log 都先记录 `training_complete`，随后才记录 test `complete`。
- 两份 `train_history.json` 都有连续 epoch 1–60，训练项和 validation video AP 全部 finite。
  HateMM epoch 57、HCS epoch 26 分别是各自 history 中最高 validation video AP；completion 与
  history 的 selected epoch/value 一致。
- 只读加载 training state 与导出的 `model.pt` 后，19 个参数张量逐项等于所存 best state，而不是
  最后 epoch state。Checkpoint selection 因此确实只由本次训练内的 validation video AP 决定。

## Test isolation、prediction cohort 与 alignment

- `train.py` 只加载同一 corpus 的 frozen train/validation manifests 和对应 scoped video labels。
  Test 阶段发生在 best state 固定并写出 training-complete 边界之后，只读取 test membership 与
  local features，给全部 test items 使用值为 0 的 placeholder label。
- Test video labels、temporal GT 与 evaluator 不进入梯度或 checkpoint selection；只有
  `evaluate.py` 在 prediction complete 后读取 test GT。Config 和 metrics 中的 isolation 声明与
  实际调用链一致。
- HateMM predictions 为 214 个唯一 ID、精确按 frozen evaluator-test manifest 排序，覆盖 29,269
  帧；HCS 为 79 个唯一 ID、覆盖 18,839 帧。两者均无 missing、extra 或 duplicate。
- 每行严格包含 core、independent-transport control 和 nearest-normal control 三个 branch。三个
  branch 的每个 score vector 均为一维、finite、位于 `[0,1]`，并逐视频与 GT 长度完全一致。

## Shared evaluator 独立复算

在内存中从两份 `predictions.jsonl` 重建三个 branch，并对每个 branch 只调用仓库唯一
`eval_baseline_scores.evaluate_scores`。六份完整 shared-evaluator reports 与各自
`metrics.json` 逐字段一致，`evaluate.log` 的最终 payload 也与 `metrics.json` 一致：

| corpus | branch | pooled AP | pooled ROC-AUC | within-video ROC-AUC | within n |
|---|---|---:|---:|---:|---:|
| HateMM | core | 0.5183327793857767 | 0.7518682909763656 | 0.5828968461651707 | 85 |
| HateMM | independent transport | 0.5032519078632682 | 0.7317412296213347 | 0.6089604978248351 | 85 |
| HateMM | nearest normal | 0.48521753240751214 | 0.7261078522136356 | 0.5957843003297822 | 85 |
| HateClipSeg | core | 0.5817230628313732 | 0.5378779403838135 | 0.5196328659491805 | 67 |
| HateClipSeg | independent transport | 0.5628532494305437 | 0.5343665935561002 | 0.5322468497966201 | 67 |
| HateClipSeg | nearest normal | 0.5583993299893579 | 0.5250449285335738 | 0.5141713619649899 | 67 |

## SOTA gate、机制 gate 与裁定

- HateMM core 的 AP/ROC/within 全部低于固定 SOTA
  `0.5938315566328208 / 0.8161837922270064 / 0.631531717970362`，所以三指标 SOTA gate
  FAIL。Core within `0.5828968462` 还同时低于 independent `0.6089604978` 与 nearest
  `0.5957843003`，机制 gate FAIL。
- HateClipSeg core 的 AP/ROC/within 全部低于固定 SOTA
  `0.6193710949898349 / 0.6050224699167533 / 0.5619078936355938`，所以三指标 SOTA gate
  FAIL。Core within `0.5196328659` 高于 nearest `0.5141713620`，但低于 independent
  `0.5322468498`；冻结条件要求同时超过两个 attribution controls，因此机制 gate 仍 FAIL。
- 两个 `metrics.json` 中的 `core_all_fixed_metrics_sota=false` 与
  `core_exceeds_both_attribution_controls_within=false` 均计算正确。任一语料失败就停止的外层
  pilot gate 更不可能通过；不能围绕 test 数字调参、挑 corpus 或扩展 MHC-EN/ZH。
- 本轮 test 结果属于 iterative/developmental evidence，不得表述为未揭盲确认性结果。

**最终 verdict：PASS（result-chain integrity）；pilot：FAIL。** 正式结论应为淘汰 UOT
normal-reference witness 本轮实现，不继续扩展。本审查未启动训练，也未修改 predictions 或
metrics。
