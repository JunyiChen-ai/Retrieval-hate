# Qwen3 dense pointwise teacher test diagnostic：独立 post-run audit

日期：2026-08-31  
范围：`runs/20260831_qwen3_test_teacher_diagnostic/formal/` 下两个 producer 产物、
`metrics.json`、`evaluate.log`，以及已审查的 `protocol.py` / `evaluate.py` 和共享 evaluator。  
结果链结论：**PASS。** 正式产物完整，独立复算与权威输出一致。Teacher premise 本身
**FAIL**，冻结结论 `STOP_BEFORE_STUDENT` 正确。

## Producer cohort、窗口与 failure

- HateMM 有 85 行、85 个唯一 ID；HateClipSeg 有 69 行、69 个唯一 ID。两者均按固定 test
  positive evaluator cohort 的完整顺序精确覆盖，无 missing、extra、duplicate 或 HMM no-GT
  exclusion 漂移。
- 两个 config 均匹配冻结 contract：同一 `Qwen/Qwen3-VL-8B-Instruct`、同一 prompt template、
  16 秒窗口、8 秒 stride、每窗最多 4 张 1 fps frame、deterministic decoding，并明确 test labels
  未用于梯度或 checkpoint selection。可读代码版本说明与正式 contract 一致。
- 正式 loader 重新验证了每行 model/corpus/split、当前 feature length、完整窗口序列、
  generation-to-score 解析及行级状态。HMM 共 1480 个窗口，HCS 共 2025 个窗口；所有窗口均为
  `ok`，parse failure 与 inference failure 都是 0。
- Producer 日志分别包含 85 与 69 个按序 progress events；每个 event 的 ID、index、窗口数和状态
  都与对应 JSONL 行一致。两个 producer 均从 `already_complete=0` 开始并以
  `cohort_complete=85/69` 结束。

## Shared evaluator 独立复算

- 逐行按冻结 overlap-window mean 重建 1 fps scores，全部 score/GT shape、finite、coverage 和
  within cohort count 检查通过；shared evaluator 报告零 missing/extra。
- 复算只调用仓库唯一 `eval_baseline_scores.evaluate_scores`。所得 corpus reports 与
  `formal/metrics.json` 逐字段一致，`evaluate.log` 中的最终 combined payload 也与
  `metrics.json` 完全一致：

| corpus | pooled AP | pooled ROC-AUC | within-video ROC-AUC | within n | failed windows |
|---|---:|---:|---:|---:|---:|
| HateMM | 0.6386555044753672 | 0.6042479331312027 | 0.5617599320908959 | 85 | 0 |
| HateClipSeg | 0.6340507489646093 | 0.5556474341970871 | 0.5396281638638373 | 67 | 0 |

Pooled AP/ROC 是 positive-only cohort diagnosis，不是 full-test SOTA 数字。Within-video ROC 的
定义与 full test 正例集合一致，是本轮唯一 gate 指标。

## Dual gate 与结论

- HateMM within `0.5617599320908959` 低于固定 test SOTA
  `0.631531717970362`，差 `-0.06977178587946609`，单语料 gate FAIL。
- HateClipSeg within `0.5396281638638373` 低于固定 test SOTA
  `0.5619078936355938`，差 `-0.022279729771756518`，单语料 gate FAIL。
- 冻结 protocol 要求两个语料同时严格超过各自 gate，并禁止单语料选择或 corpus routing。
  因此 `teacher_premise_pass_both=false`、`continue_to_student_design=false` 与
  `verdict=STOP_BEFORE_STUDENT` 均正确，不能进入 student design。
- 本结果是 test-informed iterative/developmental teacher evidence。Test GT 只进入最终评测，
  不进入 producer、梯度或 checkpoint selection；Qwen3 teacher 不能表述为项目方法或 SOTA。

**最终 verdict：PASS（result-chain integrity）；teacher premise：FAIL，必须
`STOP_BEFORE_STUDENT`。** 本审查未启动模型，也未修改 predictions、metrics 或冻结 gate。
