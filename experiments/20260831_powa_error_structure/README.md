# POWA test-prediction error-structure diagnosis

截至 2026-08-31。该目录是只读诊断，不是候选方法，不训练、不选 checkpoint。
按 `RESEARCH_ITERATION_RULES.md` Rule 10，显式使用 test predictions 与 test GT
分析 POWA 的误差，并允许结果影响下一轮机制设计。权威输出：
`runs/20260831_powa_error_structure/analysis.json`。

## Test exposure

分析固定 seed 234 的四个 corpus-specific POWA score artifact；HateMM、MHC-EN、
MHC-ZH 使用 `runs/20260831_powa_starting_point/*_seed234/scores_source.txt`
记录的归档路径，HateClipSeg 使用合规 HCS-only
`runs/20260831_powa_starting_point/hcs_maskfix_seed234/scores.jsonl`。读取四个主数据集
test GT。输出只记录每个输入的可读绝对路径、覆盖率、shape 与解析检查；不计算或记录
任何哈希、checksum 或 digest。

因此，从本轮开始，受该诊断影响的后续 test 结果都属于
iterative/developmental evidence，不表述为未揭盲 confirmatory 结果。test labels
不会进入梯度训练或 checkpoint selection。

## 问题与受限 probes

只问 POWA 的绝对视频证据与视频内排序证据是否发生冲突。probes 限于已有 score
branches 的只读分解，以及固定窗口 `{3,5,9,15,31}` 的平滑/innovation；所有数字
仍由共享 evaluator 计算。它们是 error probes，不能作为 calibration、ensemble 或
post-processing 主方法。

## 结论及对下一轮的影响

- 平滑在 HateMM/MHC-EN 同时改善 pooled 与 within，但在 MHC-ZH/HCS 降低
  within；不存在一个跨四语料成立的 temporal post-processing 修复。
- `score_powa - score_base` 的 policy residual 在 MHC-ZH seed-234 test 把 within
  ROC 从 `.4442` 提到 `.5828`，却把 pooled AP/ROC 从 `.4963/.7618` 降到
  `.4006/.6605`。这说明局部 policy evidence 含有被最终绝对分数压住的排序信号，
  但直接替换 score mass 会破坏跨视频判别。
- HCS 的 POWA、base、audio、visual 与上述 probes 都接近 chance within；下一轮
  不能只做 score 重排或平滑，必须改变训练时局部表示/监督信号。

下一候选必须是训练时的 global/local evidence separation：保留 POWA 的绝对
video/frame evidence通道，同时让 local 通道在结构上不能使用 video-constant
context，并用同语料 train/val 冻结选择。这里的 probe 不直接决定 corpus-specific
分支、窗口或 inference calibration。
