# Pre-run review

截至 2026-08-31。独立 reviewer：`metrics_audit`。

最终 verdict：PASS。

正式运行前发现并修复了三个 blocker：ASR 的缺失/非法 endpoint 会导致解析
崩溃；HateMM test manifest 比 evaluator cohort 多一个无 GT 视频；文档与实现对
circular-shift 数量的定义不一致。修复后 reviewer 确认：

- 非法 ASR span fail-closed 跳过并分类计数；
- producer 仅用 GT archive member names 定义 evaluator cohort，不访问 GT 数值；
- control 固定为每视频最多 16 个均匀、唯一、非零 shift；
- scoped labels 严格属于当前语料 train split，train/test 无交集；
- lexical、speech、shift 的 AUC 与三项固定指标全部调用共享 evaluator；
- gate 与 README 一致，语法与 synthetic contract 检查通过。

reviewer 未运行正式 producer 或 test evaluation。
