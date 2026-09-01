# RESET7 跨候选失败矩阵

截至日期：2026-09-01。依据结果：`runs/20260901_reset7_cross_candidate_failure_matrix/main/matrix.json`；构建代码：`experiments/20260901_reset7_failure_matrix/build.py`。

六项候选前目标 gap 为 HMM AP/pooled ROC/within ROC `+.100831/+.077925/+.003076`，HCS `+.066350/+.060950/+.038207`。这些是下一候选 gain budget 的最低参照，不是 validation selection target。

| 正式候选 | 实际可用 correction signal | HMM core−matched control (AP/ROC/within) | HCS core−matched control (AP/ROC/within) | 聚焦 test 证据 | 裁定 |
|---|---|---:|---:|---|---|
| Lexically Anchored DCC | train/inference 可得的 lexical anchor 与跨视频 region memory | `-.008100/-.002821/+.010128` | `+.071535/+.093386/+.019294` | HMM 各正例占比组 within 全降且分数波动被压缩；HCS 分组方向交错 | 只有 HCS 对 shifted control 的大幅收益，HMM pooled 与 anchor 均退化；关闭 lexical region-memory/shared-representation family |
| Policy-Constrained Cluster Transport | train/inference 可得的 policy-state cluster transport constraint | `+.007514/-.003370/-.022346` | `-.025098/-.016846/-.017111` | HMM 所有占比组均负；HCS 仅最低组正、最高组 `-.070092`，harmful mass `.8741` | 正确 policy 不胜 binary；关闭 policy cluster transport family |
| Active-speaker-bound utterance MIL | 冻结 TalkNet active-speaker identity 与 face feature | `+.000149/-.000615/+.003067` | `-.000301/-.000847/-.000008` | eligible 秒仅 HMM `2.33%`、HCS `4.78%`；eligible-video within delta `-.001277/+.000249` | 覆盖低且正确 assignment 无共同收益；关闭 active-speaker/source-bound family |

## 共同失败链

三个来源方法不同，但都保留 POWA raw scorer 作为主要决策路径，再加入 semantic auxiliary constraint/adapter。三个 matched-control 结果均没有在 HMM/HCS 的 AP、pooled ROC、within ROC 上形成同方向、load-bearing 的共同收益；已有证据的量级也不足以覆盖主要 `.04–.10` gap。这里的问题不是 validation 或 test 执行，而是候选 admission 时没有要求 correction observation 已在两个语料中显示与目标 gap 直接相关的纠错证据。

## RESET7 生成约束

这些 family 及无独立新证据的变体保持关闭。后续不得默认 `POWA scorer不变 + auxiliary`；可以改变 representation/backbone。任何新 brief 在 novelty review 前必须引用 HMM/HCS 双语料 observed correction artifact，证明 signal 在 train/inference 可用、非 test oracle、非当前 scorer 自确认，并给出与六项 gap 同量级的数值 gain budget。若证据只支持千分量级，不进入正式训练。
