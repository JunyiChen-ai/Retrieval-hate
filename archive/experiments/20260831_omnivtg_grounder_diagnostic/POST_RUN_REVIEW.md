# OmniVTG formal teacher diagnosis：独立 post-run audit

日期：2026-08-31  
审查范围：`runs/20260831_omnivtg_grounder_diagnostic/formal/` 下两个 producer
产物、combined evaluation 产物，以及冻结的 `protocol.py`、`evaluate.py` 和 pre-run protocol。  
结果链结论：**PASS。** 产物完整、评测复算一致，冻结 gate 的结果确为
`teacher_premise_pass_both=false`、`continue_to_student_design=false`、
`STOP_BEFORE_STUDENT`。这里的 PASS 只表示结果链可信；teacher premise 本身未通过。

## Cohort、行契约与断点续跑

- HateMM `predictions.jsonl` 有 85 行、85 个唯一 ID，按固定 test split 顺序精确覆盖
  positive evaluator cohort；HateClipSeg 有 69 行、69 个唯一 ID，同样精确覆盖。无 missing、
  extra 或 duplicate。每行的 corpus、test split、固定 model/query、schema、completion/interval/error
  关系均通过正式 loader；`source_video` 也逐项等于当前 cohort 唯一解析出的实际媒体路径。
- HateMM 日志有 85 个按序 `video_complete`，末尾为 `cohort_complete`。HateClipSeg 首进程完成
  cohort 前 61 项后退出；清理孤儿 engine 后，严格 resume 明确报告
  `already_complete=61, pending=8`。最终文件的前 61 行正好是 cohort 前缀，新增 8 行正好是剩余
  后缀；日志共有 69 个按序且唯一的 completion events，并以 `cohort_complete` 结束。
- `evaluate.log` 中两次 `rows=61` handoff 均在 incomplete cohort 时 fail closed，没有生成或接受
  partial evaluation；只有 69 行完整后才产生最终 combined payload。
- 两个 producer config 都固定为 test、同一 `zhengmh/OmniVTG-7B`、同一 query、同一 runtime
  recipe 和 eager engine mode；记录明确说明 test labels 未参与梯度或 checkpoint selection。

## Failure accounting 与共享 evaluation

- HateMM：81 条成功解析、0 parse failure、4 inference failures、0 interval outside grid。
  HateClipSeg：66 条成功解析、0 parse failure、3 inference failures、0 interval outside grid。
  七个 inference failures 都保留在固定 denominator，并按冻结实现转为与各自 GT 等长的全零
  score，没有删除失败样本。
- Interval 转 1 fps score 唯一调用共享
  `frame_eval_common.spans_to_frame_scores`，使用冻结的半开区间规则；三项指标唯一调用共享
  `eval_baseline_scores.evaluate_scores`。复算时全部 score/GT shape 和 finite 检查通过，shared
  evaluator 的 missing/extra 均为零。
- 在内存中从两份 JSONL 重新构造逐帧 score，并调用上述共享 evaluator；所得完整 payload 与
  `formal/metrics.json` 及 `formal/evaluate.log` 的最终 payload 逐字段一致：

| corpus | pooled AP | pooled ROC-AUC | within-video ROC-AUC | within n |
|---|---:|---:|---:|---:|
| HateMM | 0.62571244946881 | 0.5780953327700183 | 0.6263799583430667 | 85 |
| HateClipSeg | 0.6264736216802578 | 0.5490441846780513 | 0.5390104457550855 | 67 |

这些数值均只针对固定 positive test cohort；pooled AP/ROC 不是 full-test SOTA 数字，不能作
SOTA 比较。Within-video ROC 是本 premise 唯一预注册 gate。

## Gate 与研究结论

- HateMM within ROC `0.6263799583430667` 低于固定 structured control
  `0.633766135171972`，差 `-0.0073861768289053`，所以该 corpus 明确 FAIL。
- HateClipSeg within ROC `0.5390104457550855` 高于固定 control
  `0.5365185532909721`，差 `+0.00249189246411341`，所以该 corpus 单独 PASS。
- 冻结 protocol 要求两个 corpus 同时严格超过 control，并禁止挑选单 corpus 或 corpus routing。
  因此 combined `teacher_premise_pass_both=false`、`continue_to_student_design=false` 和
  `STOP_BEFORE_STUDENT` 均计算正确；HateClipSeg 的单独通过不能授权 student。
- 本轮是 test-only developmental teacher diagnosis。Test GT 只进入最终 evaluation，不进入
  producer、梯度或 checkpoint selection；结果不能把外部 teacher 当作项目方法。

**最终 verdict：PASS（result-chain integrity）；teacher premise：FAIL，必须
`STOP_BEFORE_STUDENT`。** 本审查没有重新运行模型，也没有修改冻结 gate 或正式结果产物。
