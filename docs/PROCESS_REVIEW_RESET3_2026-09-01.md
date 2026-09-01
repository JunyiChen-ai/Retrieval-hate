# Process Review RESET3

截至 2026-09-01。触发条件：RESET2 epoch 内 Witness-DGM、Temporal Expert-Choice 与 Marked Temporal Splat 三个正式方法均完成 HateMM/HateClipSeg test，但未通过双数据集 performance gate。审查范围仅为研究流程、状态与失败记录，不包含代码 review。

## Verdict

**RESET**。不停止研究，也不继续生成新 candidate。保留 marked temporal splat 已成立的 duration-field 机制证据，停止新 premise、新 teacher、新 raw statistic 与无关方向；下一阶段集中做现有 test prediction 的 error analysis、一个受约束的方法内改动、正常 validation 配置选择，以及立即 HMM/HCS test evaluation。

## Diagnosis

三次正式失败不是同一种失败：

- Witness-DGM 的 final ranking 几乎没有改变，gradient modulation 没有转化为定位。
- Temporal Expert-Choice 把固定负载误当 local competence，强制弱模态进入 final score，两个语料 within 均下降。
- Marked Temporal Splat 的机制实际成立：HMM/HCS core 都胜 anchor 和 point control，HMM within 相对 anchor 提升 `.099246`，HCS 提升 `.013493`。失败发生在最终 performance gate，而不是 duration-field premise。

停滞来自执行上仍把“首版没有全指标 SOTA”近似当成“机制方向关闭”，同时在前两次正式失败后穿插了无效 raw-statistic premise 与不相关 novelty candidate。HCS 已知还有 feature limitation，一个首版 temporal mechanism 同时解决 HMM pooled、HCS feature/generalization 与 within localization并不现实；但该事实不降低最终 performance gate。

## Mandatory process corrections

1. 不生成新 candidate，不做新 premise、teacher、raw statistic 或数学包装。
2. 当前 marked-splat run 可以归档，但 duration-field 记录为“机制成立、performance 未完成”，不得写成机制淘汰。
3. 只用已有 anchor、point、splat test predictions 与 GT 做一次集中 error analysis，回答 pooled separation、视频/跨度/正例率/模态可用性分层、HCS score spreading，以及 time-shuffle/position/carrier controls。它是正式方法后的机制分析，不是 premise，不扩展成 statistic sweep。
4. 对已经定义的方法，validation 可且应正常选择训练超参数、配置与 checkpoint；validation 不用于生成新机制或跨方法 performance 裁决。
5. 下一次正式运行只允许一个由 error-analysis artifact 直接支持、会影响结果的改动；必须同时针对 HMM pooled 缺口、HCS 剩余缺口，并保留 duration-field local-order gain。不得只扫描 kernel、top-K 或更换 loss 名称。
6. 沿用已审实现或只调配置不重做 technical review；只有 result-affecting 实现路径改变时做一次 review。Novelty claim 未变化时不重做 novelty review。
7. 训练后立即 HMM/HCS test，不设 premise gate 或 validation performance gate。通过后才扩 MHC-EN/ZH 与多 seed。
8. 保留累计连续 performance failure=`3`，只有 performance gate 通过才清零；同时新建 `formal failures since RESET3=0/3` 作为下一次 process-review 触发窗口，避免每次后续失败都重复开 reviewer。

## Evidence disposition

关闭 gradient-only modality balancing、fixed-capacity ownership、benign cross-modal surprise、generic smoothing/diffusion、coalition/Möbius 同族及继续寻找 raw statistic。继续 duration-field：HMM within `.727709`，HCS 三项均胜 anchor，point control 低于 splat，已经足以证明它进入并改善 final local ranking。

仍缺：现有 test prediction 的完整 error decomposition；HMM pooled 缺口来自跨视频 separation、score saturation 还是 bag separation；HCS 增益弱的原因；合法 validation 配置选择后的正式结果；达到竞争水平后的跨 seed 稳定性。

## Direction decisions

- **CONTINUE**：marked temporal duration-field 机制。
- **PAUSE**：新信息源、teacher、routing、ownership、cross-modal statistic 与全新跨任务 adaptation。
- **STOP**：Witness-DGM、Temporal Expert-Choice、benign surprise、generic temporal smoothing/diffusion、coalition-dividend，以及只换 head/loss/kernel 名称的重开。

最终单一裁定：**RESET**。
