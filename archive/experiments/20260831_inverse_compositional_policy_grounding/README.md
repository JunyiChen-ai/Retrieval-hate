# Inverse-compositional policy-relation grounding

**淘汰原因：binary bag label 与固定 policy bank 移除了原 ICL 每视频具体、变化 relation query 的监督锚点；query/role 可恒定重构，topic 可整段广播，第三项 novelty 门与可识别性均失败。**

截至日期：2026-08-31。当前阶段：双独立审查后停止；未实现、未训练、未生成 prediction。

## 来源与任务 adaptation

来源方法是 Li et al., *Inverse Compositional Learning for Weakly-supervised Relation Grounding* (ICCV 2023)。原方法把视频关系表示为 subject–predicate–object 的 holistic/partial 两级结构，用 inverse attention、compositional encoder、inverse loss 和 partial grounding 联合学习弱监督视频 relation grounding。

本候选不把外部方法缩成一个 relation scalar。每个 hate-policy clause 保留三个有身份的角色：accountable source、hostile predicate、protected target；visual/audio/text 是每个时间点的观测，不预先等同于某个角色。对每个 temporal proposal：

1. role-conditioned attention 分别产生 source、predicate、target partial grounding；
2. compositional encoder从三个 partial states构造完整有向 relation；
3. inverse decoder必须从 grounded proposal重构同一 clause 的三个 role identity，并在 role swap、target swap、predicate-only、target-only negatives上区分原关系；
4. positive video只要求 clause bank 中至少一个 proposal形成完整 relation，negative video压低所有完整 relation；最终 frame score只由覆盖该秒的完整 relation posterior产生，不允许 general-hate scalar fallback。

所有语料各自独立训练，只使用该语料 train video labels。固定 policy clause text和冻结通用 encoder属于外部先验，不使用任何主数据集 span/frame 标注。

## 与近邻的必要区别

- POWA target–predicate transport：已占用异步 hostile/target evidence transport；本候选只有在 inverse reconstruction 与 role-swap discrimination 对最终 temporal ordering 是 load-bearing 时才不同。
- Dense primitive teacher：已证明 typed attribution本身不足；本候选不能把 frozen primitive confidence当 pseudo span。
- Policy-complete P-MIL：已因 policy scalar接 proposal completeness而淘汰；本候选必须端到端学习 partial-to-holistic relation grounding，不能先做 scalar再送入普通 MIL。
- 旧 LB-SCGP：已占用 whole-video source/endorsement/target-predicate structural states；本候选不能以相同 semantic atoms 的时间展开作为 novelty。
- LELA/MATCH/IARE 等目标领域方法：必须独立检索其 composition matching、object/evidence attribution 是否已经覆盖本候选真正有效的部分。

## 必须先通过的审查问题

1. 原 ICL 或等价 inverse-compositional relation grounding 是否已经用于 hateful-video detection/localization。
2. binary video label + fixed policy clause bank 是否足以避免 fixed-query reconstruction、topic、whole-video 与 unary-clause collapse。
3. HCS 中缺少显式 source 或 target role 时，显式 null state是否会使完整关系不可用；禁止为单一语料增加 unary fallback。
4. 删除 inverse reconstruction、置乱 role identity、同容量 untyped composition、predicate-only、target-only controls能否严格区分所声称机制。

任一 novelty 硬门失败即归档，不运行 premise、不训练正式模型。

## 若审查通过的两语料最小门

先在 HateMM 与 HateClipSeg 各自独立训练。Validation仅在固定 core/control arm内部选择 checkpoint；训练后立即在 test 输出 pooled AP、pooled ROC、within-video macro ROC。core必须在两语料三个指标全部超过固定 SOTA gate，并胜过 role-shuffled、inverse-loss removed、untyped composition controls；否则淘汰，不扩 MHC-EN/ZH。
