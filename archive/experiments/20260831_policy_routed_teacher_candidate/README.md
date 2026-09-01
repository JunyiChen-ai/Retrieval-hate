# 淘汰：Certified-negative local teacher-error routing

淘汰原因：核心 transfer assumption 在两语料都失败，权威 verdict
`STOP_BEFORE_STUDENT`；不生成 train teacher、不训练 student、不读 test。

截至 2026-08-31。查新给 proposal-stage conditional GO（约 6/10），条件是先过
risk-transfer Gate A；结果未过。查新见 `NOVELTY_SCOUT.md`，权威输出：
`runs/20260831_policy_routed_teacher_candidate/gate_a.json`。

## Frozen Gate A

HateMM 与 HateClipSeg 分开。用 validation 中 negative-video 的 video label 推出所有
秒均为 benign；其中 VERA `score_raw=1` 的秒是 certified teacher false positives。
对 audio/visual/text/concat feature 的这些秒分别拟合固定最多 64 个 one-class error
prototypes。普通 normal-density control 用所有 negative-video 秒拟合同预算 prototypes。

在从未进入 prototype 拟合的 positive videos 上，只审计 VERA-positive 秒。frame GT
仅计算 teacher-error ROC/AP 与 accepted precision/coverage，不进入 prototype、阈值、
feature 或 checkpoint 选择。核心 concat FP-proximity risk 必须在两语料的 error ROC
都严格超过 teacher-confidence 与 concat normal-density，并且在 accepted coverage
`.25/.50/.75` 的 per-video macro precision 均不劣于两者，且必须胜过视频内 risk
shuffle；否则 `STOP_BEFORE_STUDENT`。error ROC 固定为逐 positive video 计算后 macro，
pooled seconds 只作诊断，防止 video identity 冒充 local error transfer。

该 validation diagnostic 模拟未来 train-negative→val-positive transfer；若通过，正式
student 仍须改为只用 train 构建 risk，并在实现后接受独立 Rule-9 review。

## 结果

初版 pooled-seconds error ROC 错把 video identity 当 local transfer；发现 HMM
within-video shuffle 更高后，按项目主指标语义修为 per-positive-video ROC macro，并把
shuffle 加为硬门。修正后 HMM concat FP-risk `.42525`、shuffle `.49292`；HCS
`.47640`、shuffle `.59779`。HCS teacher-confidence control `.68407`，明显胜过
FP-risk；两语料 accepted precision gates 也失败。说明 negative-video teacher FP
模式不能覆盖 positive-video hard negatives，继续做 policy gate 或 student 只会堆模块。
