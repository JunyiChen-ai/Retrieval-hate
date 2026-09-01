# Independent post-run integrity review

截至 2026-08-31。审查对象为本实验当前实现，以及
`runs/20260831_multimodal_pmil_baseline/pilot_seed234/` 下 HateMM、HateClipSeg 两个完整运行。

## 裁定

**Integrity PASS；performance/SOTA FAIL。** 两个 corpus 的训练、official-validation checkpoint
selection、blind test score 导出与共享 evaluator 评测链均完整且可复算。这个运行只能作为
multimodal binary P-MIL 的跨任务 baseline，不是 novelty candidate，也没有在任一 corpus 通过三项
SOTA 门；HateClipSeg 尤其明显失败。

## 1. 进程、日志与产物完整性

- root `run.pid` 记录的进程已经结束；`run.log` 无 traceback、abort 或 failure marker。
- 日志按冻结顺序完整记录 HateMM 15 epochs、blind test inference、共享 evaluator 输出，随后完整记录
  HateClipSeg 的同一链；末尾是 HateClipSeg `metrics.json` 成功写出，不是中断或 partial run。
- 两个 corpus 均具备非空 `config.json`、`model.pt`、`train_log.json`、`scores.jsonl`、
  `metrics.json` 与 `proposal_diagnostics.json`。所有 checkpoint tensor 均可解析且 finite。
- 正式入口固定 seed 234、15 epochs、Adam learning rate `5e-5`、bag batch size 10；运行中的 config
  与入口参数和 `run.py` 实际训练 recipe 一致。

## 2. 训练与 official validation selection

- 每个 corpus 独立训练；gradient loop 只遍历 official train IDs及其 scoped video labels。
  official validation IDs和 labels只进入逐 epoch video-level AP与本 arm checkpoint selection。
- 两份 history 均包含连续的 15 个 epochs，全部 loss项与 validation AP均 finite。保存的 selected epoch
  与 history 中 validation AP 的严格最大值一致：
  - HateMM：epoch 15，validation video AP `0.8250433307318501`；
  - HateClipSeg：epoch 5，validation video AP `0.9050234426842143`。
- frozen proposal producer 指向各 corpus 的 archived official-val MultiHateLoc seed-234 checkpoint。
  对应 source train logs也显示其保存 epoch是各自 history中 official-validation video AP 的最大值：
  HateMM epoch 47，HateClipSeg epoch 64。当前 baseline只复用这两个冻结 checkpoint。
- `run.py` 不调用 temporal GT或 corpus-wide label loader；只有 train、val 两次 scoped label调用。
  test IDs在最佳 state恢复后才进入 blind proposal generation与 scoring。test labels/GT只由 scores关闭后启动的
  共享 evaluator读取，不参与 gradient或 checkpoint selection。

## 3. Test coverage、score contract 与 evaluator复算

逐行解析 `scores.jsonl` 并与共享 evaluator test GT核对：ID集合和顺序精确、无 duplicate、无 missing或
extra；每条只有 `video_id` 与 `score_pmil`，score为有限的一维数组，长度逐视频与1 fps GT精确一致，且
全部落在 `[0,1]`。

| Corpus | Videos | Frames | Within-valid videos | Pooled AP | Pooled ROC-AUC | Within-video macro ROC-AUC |
|---|---:|---:|---:|---:|---:|---:|
| HateMM | 214 | 29,269 | 85 | 0.5892557205 | 0.8142208576 | 0.5898977996 |
| HateClipSeg | 79 | 18,839 | 67 | 0.4605069285 | 0.4230704517 | 0.4766100594 |

以上结果使用 `scripts/reproduction_baselines/eval_baseline_scores.py::evaluate_scores` 从正式 scores和
test GT重新计算，与两个 `metrics.json` 的完整 `score_pmil` 结果结构逐项相等。stored metadata也固定为
对应 corpus、`split=test`、正式 scores路径；missing/extra均为0。因此不存在实验目录内自建指标、换
split或只评部分 cohort的问题。

## 4. Proposal diagnostics 的边界

- `proposal_diagnostics.json` 精确覆盖 train/validation/test 三个互斥 manifest的并集：HateMM
  `744/109/214`，共1067个视频；HateClipSeg `251/63/79`，共393个视频。
- per-video proposal counts均为正整数且不超过冻结上限256；重新统计的 min/median/max与文件一致：
  HateMM `3/110/256`，HateClipSeg `68/100/248`。
- `train_log.json` 的 blind test summary精确覆盖各自 test cohort；video score和 proposal score
  min/mean/max全部 finite、位于 `[0,1]`，且顺序关系成立。正式逐帧 scores的 exact coverage进一步验证
  whole-video proposal/readout没有留下未覆盖秒。
- 这些 diagnostics只描述 proposal数量与模型自身置信度，不读取 temporal GT，也不能被解释为 proposal
  recall、oracle coverage、机制有效性或新方法证据。当前 run目录没有额外 test-error-analysis产物；若后续
  使用 test GT做 developmental error analysis，必须明确与本 baseline结果和训练选择隔离。

## 5. SOTA门与研究结论

按当前冻结 SOTA references比较，两个 corpus三项均未严格超过：

| Corpus | AP delta | Pooled ROC delta | Within ROC delta | Verdict |
|---|---:|---:|---:|---|
| HateMM | -0.0045758361 | -0.0019629346 | -0.0416339184 | 三项全 FAIL |
| HateClipSeg | -0.1588641665 | -0.1819520182 | -0.0852978342 | 三项全 FAIL，明显退化 |

HateMM 的两个 pooled指标虽接近门槛，定位主指标 within-video ROC仍低 `0.04163`，不能称为 SOTA。
HateClipSeg 的 pooled AP、pooled ROC与 within ROC均显著落后，不能用 validation AP较高、proposal
diagnostics或video-level结果抵消。也不得按 corpus改 threshold、readout、modality或 checkpoint规则后将
结果组合为一个方法。

最终结论：**结果链完整，独立 integrity audit PASS；P-MIL port只成立为已完成的 baseline，HateMM与
HateClipSeg均未过SOTA，尤其HateClipSeg明确失败。**
