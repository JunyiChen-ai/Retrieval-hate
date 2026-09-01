# Formal post-run result-chain audit

截至 2026-08-31。审查对象为
`runs/20260831_owner_abstaining_its2clr/pilot_seed234/` 的 HateMM/HateClipSeg
`anchor`、`broadcast`、`core` 六条正式结果链，以及冻结的训练、预测、共享评测与提前停止规则。
本审查未启动训练或模型推理，未修改任何正式 prediction、checkpoint、metrics 或 verdict。

## 最终裁定

**Integrity PASS。** 六个已完成 arm 的 blind test score、共享 evaluator 输出、训练 checkpoint、
per-arm OOF cache 与最终 `verdict.json` 一致且完整。`advance=false` 与停止该候选的裁定正确。

这不是八 arms 的完整 mechanism attribution。原顺序 run 在 HateMM core 完成共享评测并触发科学硬门失败后
主动停止；后续 completion 只补齐了 HateClipSeg 的三个主 arms。五个 attribution controls 未完成，报告只能
裁定候选不满足 performance gate 和最小 core-vs-broadcast 效果量门，不能声称这些 controls 证明或否定了
具体机制。

## 1. 运行完成状态与提前停止边界

- 根 `run.log` 依次完成 HateMM `anchor`、`broadcast`、`core` 的 metrics 写出，结尾无 traceback 或运行错误。
  这里的“hard failure”是冻结研究门失败，不是训练/评测进程崩溃。
- `rejection_completion.log` 依次完成 HateClipSeg 三个主 arms，并写出最终 verdict；当前根 `run.pid` 已结束，
  系统中没有该实验的存活训练或 supervisor 进程。
- HateMM `branch_selector/` 及其 `oof/` 只是空目录，没有 checkpoint、score、metrics 或可被当作完成结果的
  内容。`shuffled_carrier`、`abstain_negative`、`nonpositive_background`、`projection_only` 也没有完成
  artifacts。
- `verdict.json` 把完成项限制为 `anchor/broadcast/core`，并把上述五个 controls 明列为
  `uncompleted_arms_not_claimed`。因此没有把提前停止误述成完整 attribution。

## 2. Blind score、coverage 与共享 evaluator 复算

对六个 `scores.jsonl` 逐行解析，并与共享 evaluator 的固定 test GT 对齐。每个文件均满足：video ID 唯一、
顺序与 cohort 完全一致、无 missing/extra；每个 `score_core` 为一维数组，长度逐视频精确等于 1 fps GT，
全部 finite 且在 `[0,1]`。覆盖情况为：

| corpus | videos | frames | within-ROC eligible videos | missing / extra |
|---|---:|---:|---:|---:|
| HateMM | 214 | 29,269 | 85 | 0 / 0 |
| HateClipSeg | 79 | 18,839 | 67 | 0 / 0 |

六个文件均通过仓库唯一共享 evaluator
`scripts/reproduction_baselines/eval_baseline_scores.py::evaluate_scores` 重新评测；返回的完整 `results`
与对应 `metrics.json` 一致。复算三指标如下：

| corpus | arm | pooled AP | pooled ROC-AUC | within-video macro ROC-AUC |
|---|---|---:|---:|---:|
| HateMM | anchor | 0.4858992701 | 0.7412718020 | 0.6246711994 |
| HateMM | broadcast | 0.4990344005 | 0.7588693136 | 0.6152395361 |
| HateMM | core | 0.4823332626 | 0.7353273801 | 0.6183685712 |
| HateClipSeg | anchor | 0.5295705516 | 0.5154747780 | 0.5135475540 |
| HateClipSeg | broadcast | 0.5208985328 | 0.5103316926 | 0.5107843033 |
| HateClipSeg | core | 0.5213747652 | 0.5111428809 | 0.5118321365 |

所有 `metrics.json` 均明确记录 `split=test`、branch 为 `score_core`，并引用对应 arm 的 blind score 文件。

## 3. 训练与 validation checkpoint selection

六个 `train_log.jsonl` 都有连续 60 epochs，loss 与 validation video-AP 均 finite。每个 checkpoint 保存的
selected epoch/value 与该 arm 日志内 validation video-AP 最大值精确一致：

| corpus | arm | selected epoch | selected validation video-AP |
|---|---|---:|---:|
| HateMM | anchor | 10 | 0.8473530361 |
| HateMM | broadcast | 8 | 0.8540729298 |
| HateMM | core | 10 | 0.8507962836 |
| HateClipSeg | anchor | 1 | 0.9413103780 |
| HateClipSeg | broadcast | 1 | 0.9406498439 |
| HateClipSeg | core | 1 | 0.9425931698 |

训练配置均为 corpus-specific、seed 234、60 epochs，并保持冻结的 batch/model/optimizer 参数。训练日志只把
official validation video-AP 用于本 arm 内 checkpoint selection；blind test score 在 checkpoint 固定后生成，
没有 test 指标参与梯度、epoch selection 或跨 arm selection。

## 4. OOF cache、fold isolation 与 iterative refresh

`broadcast` 和 `core` 四个非anchor arm 各自拥有独立 `oof/pseudo.pt`、`oof_log.json` 与 diagnostics；cache
内 corpus/arm 与消费者配置一致，official train IDs 精确覆盖且无 test 使用。三折 assignment 与冻结的
stratified split 一致：HateMM 每折 496 fit / 248 held；HateClipSeg 两折 167 fit / 84 held，另一折
168 fit / 83 held。

每折日志均包含完整 40-epoch seed history 和 15-epoch refine history，loss finite；refine 每 5 epochs 后刷新，
因此三次 refresh 均真实执行。结合当前 `oof.py` 的同步实现核查：某 fit video 使用的 relation 来自把该视频
置于 held 集的另一 fold model；所有 fold 先在旧 cache 上更新，再统一刷新各自 held evidence，不存在 held
video 进入自身 seed fit、异步同轮污染或跨 arm cache 复用。cache 每行的 fused/branch/deletion/shuffle arrays
均与该视频 train 长度一致且 finite。

这些证据确认 iterative OOF 链实际运行，但不证明 pseudo relation 本身正确，也不扩展为 teacher independence
或因果 owner claim。

## 5. 冻结 gates 与 verdict 复核

冻结 SOTA 门为 HateMM AP/ROC/within
`0.5938315566 / 0.8161837922 / 0.6315317180`，HateClipSeg
`0.6193710950 / 0.6050224699 / 0.5619078936`。core 在两个语料的三项指标都严格低于对应门，因此
`performance_gate_by_corpus=false`、`all_metrics_sota_both=false` 正确。

core 相对 capacity-matched broadcast 的 within-video ROC 增益为 HateMM `+0.0031290352`、HateClipSeg
`+0.0010478332`。两边符号都为正，所以 `core_beats_broadcast_both=true` 正确；但最大增益远低于冻结的
“至少一语料 `>=0.020`”条件，所以 `at_least_one_gain_ge_020=false` 正确。

由 performance 硬门和最小效果量门已经失败，未完成 controls 不可能把该候选变成 eligible；最终
`advance=false` 和 `STOP` 正确。严格来说，完整 attribution gate 没有被评测，不能把它写成由全部 controls
得出的 `mechanism_gate=false`。

## 可用结论

六个已完成主 arms 的结果链可作为可信的 iterative/developmental test evidence。可以使用的结论只有：

1. 当前 owner-abstaining ItS2CLR core 在 HateMM 与 HateClipSeg 均未达到冻结 SOTA；
2. 相对 broadcast 的 within-video 改善只有约 `0.00313` 与 `0.00105`，不足以支持预注册机制效果量；
3. 因而停止候选、不扩 MHC-EN/ZH、也不围绕本轮 test 结果继续调当前机制是正确决策；
4. 未运行的 selector、shuffle、abstention、background 与 projection controls 不支持任何完整归因结论。

最终结论：**结果完整性 PASS；候选晋级 FAIL；完整机制归因未执行且未声称执行。**
