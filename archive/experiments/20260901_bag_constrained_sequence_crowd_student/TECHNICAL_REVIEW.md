# Pre-run technical review

截至日期：2026-09-01。正式方法运行前唯一一次独立基础 technical review 最终裁定：**PASS**。Reviewer 未参与 novelty 或 process review，未修改代码，未运行训练。

审查严格限定于会改变实验观察或结论的问题。初审确认 sequence transition 与 source-specific boundary edge emission 实际进入 forward-backward posterior，core posterior 与 bag loss 实际进入 student loss，posterior/bins 只读 train，validation 只选择超参数与 checkpoint，test 调用统一 evaluator 并要求完整覆盖。

初审发现并完成最小修复的三项：

1. HMM test manifest 比冻结 GT 多一个视频；test inference 现按冻结 GT cohort 过滤并要求集合完全一致。定向确认 HMM/HCS 分别严格覆盖 214/79 个 test 视频。
2. HMM VERA sparse starts 原可越过媒体结束；现按 feature length 与 media duration 的有效交集生成 starts，并检查完整 expected starts 与 `start < end <= duration`。两个已知样例最后 start 为 223/175，均小于媒体 duration；HCS 既有 238 个 raw record 仍通过新校验。
3. Mechanism gate 原比 README 宽；现要求 core 相对两个 matched controls 的 AP、pooled ROC、within ROC 均非负，且两个 within 严格改善。每语料至少一个主要 gap 达到 `+.02` 也实际进入最终 mechanism gate。

同时定向确认 posterior-ordering 诊断只统计 positive 且 core/token-DS posterior 都非恒定的视频，并记录有效视频数。修复后 reviewer 给出 targeted `PASS`；没有重新展开泛化审查。
