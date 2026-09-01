# Novelty review: target-conditioned proposal MIL

**截至日期：2026-08-31**  
**审查对象：** `experiments/20260831_target_conditioned_proposal_mil/README.md`  
**裁定：PASS（窄 adaptation novelty，6.2/10）；作为可识别的 likelihood-ratio 方法则 FAIL。**

## 结论先行

没有找到“用同语料、同 target/topic 的 benign temporal proposals 定义局部参照，再以 proposal MIL 从 binary video labels 学 hateful temporal localization”的直接先例，也没有找到该机制已用于 hateful-video detection/localization。按照当前标准，跨任务 adaptation 可以构成贡献，因此本候选可以进入 pilot。

它能通过 novelty gate 的部分不是 proposal、inside-versus-surrounding feature、retrieval、normal prototype、MIL、sparsemax、noisy union 或 density ratio 中任何一个组件。唯一可保留的机制 claim 是：**hate 是相对于相同讨论对象/主题的 benign use 才成立的关系性 residual，因此 negative proposal 必须按 target/topic 条件匹配；这个 matched benign contrast 同时定义 proposal instance score，并由 proposal-level bag objective 学习。** 这比把 retrieval 输出接到 P-MIL classifier 后面更窄，也有明确的任务混淆因素和直接证伪 control。

但 README 当前公式

```text
A_I = log p(r_I | z_I, hateful) - log p(r_I | z_I, normal)
```

在只有 binary bag labels 时并不可识别。negative bags 可提供 `p(r|z,normal)` 的样本；positive bags 中 proposal 是 unknown mixture，至多知道每个 positive bag 含有至少一个 hateful instance。没有 instance prevalence、anchor instances、可分性或 mixture-proportion 假设，`p(r|z,hateful)` 与每个 topic 下的 mixture weight 可以相互补偿。sparsemax 只是 latent instance selection，不会使真实 class-conditional density 自动可识别。

因此：

- 若实现为 matched-negative contrastive energy，经 MIL 学一个 proposal discriminator，它是可实现、可证伪的 non-trivial adaptation，**PASS**。
- 若论文声称从 binary bags 估计了真实或校准的 conditional likelihood ratio，或识别了 hostile residual distribution，**FAIL**。

## 直接先例及其占位范围

### Proposal MIL 与 surrounding residual

Ren et al., [Proposal-Based Multiple Instance Learning for Weakly-Supervised Temporal Action Localization, CVPR 2023](https://openaccess.thecvf.com/content/CVPR2023/html/Ren_Proposal-Based_Multiple_Instance_Learning_for_Weakly-Supervised_Temporal_Action_Localization_CVPR_2023_paper.html) 已经在训练和测试都直接分类 temporal proposals，并使用 Surrounding Contrastive Feature Extraction：扩张 proposal 左右边界，计算 outer–inner contrastive feature，以抑制过短的判别性 proposals。它还用 video labels 聚合 proposal scores。

所以 proposal-level MIL、训练/测试观测单元一致和 inside-versus-surrounding residual 都不能作为新意。当前候选的 `r_I` 与 P-MIL 的 SCFE 高度重合；把 action proposal 换成 hate proposal也不够。P-MIL 未用于 hateful-video detection/localization，这满足“被 adaptation 的来源方法未进入目标任务”条件，但 adaptation 必须由 target-matched benign reference 承担。

### Normal-reference WSVAD

Park et al., [Normality Guided Multiple Instance Learning for Weakly Supervised Video Anomaly Detection, WACV 2023](https://openaccess.thecvf.com/content/WACV2023/papers/Park_Normality_Guided_Multiple_Instance_Learning_for_Weakly_Supervised_Video_Anomaly_WACV_2023_paper.pdf) 已从无噪声 normal videos 编码多样 normal prototypes，并用 similarity classifier/refinement 改善 positive bag 中错误 instance selection。Xu et al., [Discriminative Score Suppression for WSVAD, WACV 2025](https://openaccess.thecvf.com/content/WACV2025/papers/Xu_Discriminative_Score_Suppression_for_Weakly_Supervised_Video_Anomaly_Detection_WACV_2025_paper.pdf) 也使用 normal-prototype memory refinement。UR-DMU 等工作同样已占据 normal/abnormal memory + MIL。

所以“只从 negative videos 学 normal bank，再用与 normal 的不相似度帮助 MIL”已被直接占位。候选只可能在 **按 protected target/topic 条件匹配 normal proposals** 以及 proposal-level relational residual 上形成差异。

### Contextual anomaly / conditional comparison

Contextual anomaly detection 的标准定义就是：行为变量只有相对于共享 context 的 reference group 才是异常；条件 normal model、conditional compatibility 和 density-ratio scoring 都不是新概念。2026 年的 [When Anomalies Depend on Context](https://arxiv.org/abs/2601.22868) 也明确针对“同一对象/行为在不同 context 下正常性不同”的视觉异常，并学习 subject–context compatibility。该线没有 hateful temporal proposals 和 binary proposal MIL，但它占据通用的 `context z + behavioral residual r + conditional normality` 原理。

### 2025–2026 hateful-video 方法

- Lang et al., [Biting Off More Than You Can Detect: Retrieval-Augmented Multimodal Experts for Short Video Hate Detection, WWW 2025](https://jianlang.org/assets/papers/WWW-2025-MoRE.pdf) 已使用 joint multimodal video retriever 获取相关实例，为多模态 experts 提供 topic/context knowledge，并同时观察 hateful 与 non-hateful retrieved videos。它是当前“retrieval + hate + contextual knowledge”最近先例，但只做 video-level detection，不做 target-matched benign proposal likelihood，也不输出 temporal localization。
- Sun et al., [MultiHateLoc, WWW 2026](https://arxiv.org/abs/2512.10408) 使用 modality-aware temporal encoders、dynamic fusion、cross-modal contrastive alignment 和 top-style modality-aware MIL；没有 interval proposals、matched benign bank 或 conditional residual comparison。
- Sun et al., [LELA, 2026](https://arxiv.org/abs/2602.09637) 是 training-free frame localization，使用五模态 caption/prompt 与 composition matching；没有同语料 weak training、proposal MIL 或 negative reference。
- Zhang et al., [CLARA, 2026](https://arxiv.org/abs/2608.15905) 使用 utterance-aligned clips、MoE、local-global segment contrast 与 VLM rationales进行 video detection；它占据 clip modeling 和 local-global contrast，未使用 matched benign proposal reference 或 temporal localization likelihood。
- Li et al., [MATCH, TCSVT 2026](https://jianlang.org/papers/MATCH.html) 用 hate/non-hate dual-perspective LMM proposers 和 spatiotemporal verifier生成解释，再融合 video features做 detection；其“proposal”是 evidential clue，不是 temporal interval MIL，也没有 conditional normal density。
- 项目内 POWA 已使用 protected-target 与 hostile predicate 的跨时间 transport；它占据 target–hostility relational semantics。项目内 negative-reference UOT 和 V26 conditional background 又分别占据 negative-only reference 与 contextual residual。当前候选必须证明 **topic-matched benign proposal comparison** 本身 load-bearing，不能把这些既有语义重新组合后声称整个系统新。

综合判断：直接完整机制尚未被占位，但相邻空间非常拥挤；这是一个窄 task adaptation，而不是新方法族。

## 为什么不是自动等于“P-MIL + retrieval”

若 `z_I` 只用于取回附加 feature，再与 P-MIL logit 拼接，这就是 MoRE-style retrieval augmentation 与 P-MIL 的组件拼接，应判 FAIL。

可以通过标准的版本必须满足：

1. `z_I` 仅定义 comparison stratum，即“哪些 benign proposals 是这个 interval 的有效反事实”；
2. `r_I` 是在该 stratum 内被检验的 hostile relational behavior，而不是另一个随意 embedding；
3. proposal instance energy 明确定义为相对于 matched benign distribution/reference 的差异；
4. 同一个 energy 被 MIL bag likelihood训练并直接产生 localization score；
5. 去掉 conditioning 或破坏 target–reference 配对后，机制必须失败。

这五点使 target conditioning 改变监督语义，而不是增加一个输入模块。README 的故事基本满足 1–5，但数学定义仍需在训练前补全。

## Binary labels 下的实现与不可识别性

### 可实现部分

可以用 binary video labels训练：negative video 的所有 proposals 是可靠 negative；positive video 用 sparse MIL aggregator选择至少一个高能 proposal；matched negative proposals向 proposal energy 提供 topic-controlled comparison。这个目标与 ordinary discriminative MIL 一样可优化，也可通过 shuffled-target、unconditional reference 和 plain P-MIL controls直接证伪。

### 不可识别部分

设同一 topic 下 positive-bag proposal marginal 为：

```text
p_pos(r | z) = pi(z) p_hate(r | z) + (1-pi(z)) p_normal(r | z)
```

negative data只识别 `p_normal`。binary bag label只约束每个 positive bag至少有一个 positive proposal，不给出 `pi(z)`。多个不同的 `pi(z)` 和 `p_hate(r|z)` 可产生相同 observed bags。因此真实的 `p_hate/p_normal` 不可由现有监督唯一恢复。

必须二选一：

- 把方法诚实定义为 **target-matched benign contrastive energy proposal MIL**，不声称概率密度或校准 ratio；或
- 在 README 训练定义中明确给出 latent mixture、proposal prevalence/anchor 或 density-ratio estimation 假设，并用可恢复性实验验证。仅把 classifier logit 写成 log likelihood ratio 不够。

另一个识别风险是 frozen semantic anchor 同时编码 hostility。如果 `z` 已含 label signal，matching 会对 outcome conditioning，或者模型直接靠 target/topic预测 bag label；此时“控制 benign identity mention”的解释不成立。

## 主要退化

1. **Topic shortcut：** protected target 在 positive videos更常见，`z` 单独完成 bag classification，`r` 与 benign matching失去作用。
2. **Unconditional collapse：** conditional kernel带宽过大，所有 normal proposals权重近似相同，退化为 NG-MIL-style global normal bank。
3. **Nearest-neighbor memorization：** 带宽过小或 reference bank稀疏，分数主要反映 retrieval distance/视频身份，而不是 hostile residual。
4. **Self-confirming positive density：** 当前 classifier选中的 positive proposals又用于拟合 numerator，错误 topic/boundary会被强化。
5. **Proposal-length shortcut：** inside-versus-surrounding residual随 interval长度、边界扩张和语音切分系统变化；sparsemax可能只选最长或最短 proposals。
6. **Noisy-union coverage bias：** 若 frame score为 `1-∏(1-p_I)`，被更多 proposals覆盖的秒即使各 proposal probability相同也会得到更高分。proposal枚举和边缘位置会直接决定 score，必须归一化或证明 constant-logit control为平坦。
7. **Video-level broadcast：** 所有高分 proposals覆盖相近的大段，最终只是把 video label广播到整段，pooled指标可能好但 within-video ranking无效。

## 最低必要 controls

所有 trainable arms需独立重训；validation只用于该次训练选择 checkpoint，随后立即在 test报告固定三指标。同一 checkpoint 上删模块只能作诊断，不能证明训练机制归因。

1. **Faithful P-MIL port：** 保留与 core 相同 proposals、encoder、SCFE/inside-surrounding feature、参数量和 aggregator，仅移除 normal reference。不能用弱化的 pointwise classifier冒充 P-MIL control。
2. **Unconditional normal reference：** 同一 bank、同一容量和 kernel，只去除 `z` conditioning。
3. **Target-shuffled matching：** 在 normal bank内破坏 `z↔r` 配对，保持邻居数、温度和计算量。
4. **Proposal-level NG-MIL control：** 多 normal prototypes/similarity refinement，但不按 topic匹配。区分条件 reference与已有 normality-guided MIL。
5. **Retrieval augmentation control：** 取回同样 neighbors，将其 pooled feature拼入 P-MIL classifier，但不形成 matched contrast。区分统一监督语义与 P-MIL + MoRE式检索拼接。
6. **Target-only 与 residual-only：** 分别只让 `z_I`、只让未条件化 `r_I`预测 proposal/video。core必须超过两者，且 `z` 单独不应解释大部分收益。
7. **Matched benign label control：** 比较 same-topic benign neighbors、random benign neighbors、same-topic mixed-label neighbors。只有第一项成立才支持“benign target counterfactual”，而非普通语义检索。
8. **Reference coverage：** 每个语料报告 effective neighbor count、topic距离、无有效邻居比例及其按视频长度/label分层结果。任一主语料普遍无 matched benign support，机制前提即失败；不能退回 global bank后保留 core claim。
9. **Density claim control：** 若坚持 likelihood-ratio措辞，用有已知 instance labels/prevalence的合成 mixture验证 ratio recovery，并比较直接 bag classifier logit。否则改名为 contrastive energy。
10. **Proposal controls：** 相同 proposal budget下比较固定多尺度网格、P-MIL proposal generator与随机边界；报告 active proposal count、长度、overlap、coverage及 sparsemax entropy。
11. **Union-bias control：** constant proposal logits、coverage-count score、mean/max overlap readout。core不能由每秒覆盖 proposal数解释。
12. **Localization shortcut tests：** temporal shuffle、inside/surrounding swap、context置换、全视频 proposal；并在 test developmental error analysis中报告预测 span长度与 GT、target presence和视频长度的关系。

首轮 README 已有的四个 controls方向正确，但缺少 faithful SCFE-matched P-MIL、NG-MIL、retrieval-only、target-only、coverage/noisy-union 和 density-identifiability controls。这些是 attribution 必需项，不应作为失败后追加的可选消融。

## 最窄可主张机制

若按当前监督实现，建议删除“likelihood ratio”强表述，改为：

> We adapt proposal-based MIL to weakly supervised hateful-video localization by scoring each temporal proposal with a hostile relational energy contrasted against same-corpus benign proposals matched on protected-target/topic context. This target-matched benign comparison defines the MIL instance score, aiming to distinguish hostile use from benign mentions of the same target.

只有在补充可识别 mixture假设并验证 ratio recovery后，才可将 `energy` 改回 `conditional likelihood ratio`。

不得主张：首次 proposal MIL、首次 contextual anomaly detection、首次 normal-reference WSVAD、首次 retrieval-augmented hate detection、首次 relational target modeling，或从 binary labels识别真实 hateful proposal density。

## 补充裁定：改为 conditional normal flow `p0(r|z)`

补充候选不再估计 `p(r|z,hateful)`，而是采用 CFLOW-AD/ContextFlow++ 式 conditional density：只用 negative-train proposals 最大化

```text
log p0(r_I | z_I)
```

并以 `a_I = -log p0(r_I|z_I)` 作为 proposal abnormality；negative bag 压低所有 `a_I`，positive bag 的 MIL 只要求至少一个高 `a_I` interval。

### 裁定

**这比 README 原来的双密度比更可辩护，也更统一，但 novelty 更窄；仍为 PASS 的 cross-task adaptation，不是新的 conditional density/MIL 方法。**

更可辩护的原因是：negative bags确实给出全部 normal proposals，因此 `p0(r|z)` 在所选模型族、固定 representation 和 observed negative support上可由 maximum likelihood估计；不再需要从 contaminated positive bags分离不可识别的 `p_hate` 与 `pi(z)`。严格说它是 **conditional normal negative log-likelihood/one-class energy**，不是 likelihood ratio。

机制也比 bank retrieval更统一：`z` 直接是条件变量，`r` 是被建模变量，不需人为规定邻居个数或 kernel bandwidth；同一 `-log p0` 同时承担 local score和 bag instance score。对于“protected target本身常见且 benign，hostile use才应偏离同 target normal residual”的任务故事，这是一个合理、可证伪的 adaptation。

但相邻先例已经很近：

- Gudovskiy et al., [CFLOW-AD, WACV 2022](https://openaccess.thecvf.com/content/WACV2022/papers/Gudovskiy_CFLOW-AD_Real-Time_Unsupervised_Anomaly_Detection_With_Localization_via_Conditional_Normalizing_WACV_2022_paper.pdf) 已用 conditional normalizing flow在 anomaly-free training data上学习局部 feature density，并以 likelihood进行 anomaly localization。
- Gudovskiy et al., [ContextFlow++, UAI 2024](https://proceedings.mlr.press/v244/gudovskiy24a.html) 已占据 mixed-variable context encoding、context-conditioned specialist density和 exact likelihood。
- Cho et al., [Unsupervised Video Anomaly Detection via Normalizing Flows with Implicit Latent Features](https://arxiv.org/abs/2010.07524) 已在 normal-video latent features上最大化 flow likelihood，并以 clip/frame NLL做 video anomaly localization。
- Zhu et al., [Towards Open Set Video Anomaly Detection, ECCV 2022](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136940387.pdf) 已把 normalizing flow与 weakly supervised video MIL放在同一框架；其 flow用于生成 pseudo anomalies和改善 instance selection，而不是本候选的 topic-conditional proposal NLL，但它排除了“首次 NF + weak video MIL”的 claim。
- 2026 年 NF-DCL 也先用 normal videos训练 normalizing flow，再与 weakly supervised MIL/contrastive learning结合；同样不是 hateful topic-conditioned proposal density，但说明“normal flow + WSVAD”已是拥挤组合。

未找到上述工作把 `z` 定义为 protected-target/topic、把 `r` 定义为 hostile relational proposal residual，或用于 hateful temporal localization。因此剩余贡献仍只可能是：**将 conditional one-class proposal density适配为 same-topic benign counterfactual，并让 positive video label通过 proposal MIL选择低-normal-likelihood interval。**

### 新的可实现性分岔

必须在实现前明确 positive MIL 的梯度流向，否则该表述存在二选一问题：

1. 若 `z`、`r` encoder和 `p0` 全部只由 negative likelihood训练并冻结，则 `-log p0` 是干净的 one-class locator；positive MIL没有可训练对象或只学单调 calibration，方法实质上不是 weakly supervised proposal MIL。
2. 若 positive MIL反向更新 `r` encoder，同时 flow由 negative likelihood约束，则 positive labels确实参与 localization学习；但 encoder可把整段 positive video的全局 label shortcut编码进每个 proposal，使所有 positive proposals远离 `p0`。这重新引入 video-level broadcast，且 learned feature-space likelihood不再具有简单生成语义。
3. 若 positive MIL也直接更新 flow使 positive proposals低 likelihood，则它是 discriminative energy training，不应再称“只由 negative bags学习的 normal density”。

最可辩护的版本是：frozen target anchor；proposal-local `r` encoder受 negative MLE与positive bag MIL共同训练；conditional flow参数只接收 negative MLE；明确阻断 proposal外的 video identity/global pooled feature进入 `r`。这仍不是可识别的 causal hostility，但至少使 normal model和weak positive orientation职责清楚。

### OOD 退化风险

“low conditional normal likelihood = hate”只是 task hypothesis，不是由 flow保证的事实：

- rare benign reporting、少数语言/口音、罕见剪辑或低质量音频可能低 likelihood；常见仇恨口号反而可能位于高-density区域。
- protected topic的 negative support可能极稀疏，conditional flow会外推；高 NLL测到的是 topic coverage，而不是 hostile relation。
- Kirichenko et al., [Why Normalizing Flows Fail to Detect Out-of-Distribution Data, NeurIPS 2020](https://proceedings.neurips.cc/paper/2020/hash/ecb9fe2fbb99c31f567e9823e884dbec-Abstract.html) 表明 flow likelihood可能受低层统计主导，并给真正 OOD 样本更高 likelihood。使用 frozen high-level residual features减轻但不消除该问题。
- flow可通过 Jacobian/feature scale获得 likelihood差异；raw NLL也随 residual dimension、proposal length和feature norm变化。
- positive MIL只要求一个低-likelihood proposal时，最罕见的 benign editing artifact会成为稳定 shortcut。

因此该 reformulation不是“不可取”，但必须把它定位为高风险假设并优先做 premise test，不能因为 exact likelihood就声称语义更可靠。

### Flow reformulation 的新增必做 controls

除前述 P-MIL、unconditional、target-shuffled、NG-MIL和 proposal coverage controls外，还必须加入：

1. **Conditional flow vs matched kNN/kernel energy**：相同 `z/r` 和 negative data，证明收益来自 conditional density而非任意 nearest-normal distance。
2. **Unconditional flow 与 context-only flow**：区分 topic conditioning、normal density和单纯 target rarity。
3. **Frozen one-class vs positive-oriented encoder**：量化 positive MIL实际提供的增益；若两者相同，不得称 weakly supervised adaptation。
4. **Stop-gradient matrix**：flow仅 negative更新；`r` encoder为 negative-only或 negative+positive MIL；不得让定义不清的联合更新掩盖贡献。
5. **Typicality/simple baselines**：Mahalanobis、Gaussian mixture、normal prototypes、feature norm、kNN。flow必须超过这些，否则只是高容量 normal scorer。
6. **Feature scale/Jacobian control**：对 `r` 固定归一化，报告 latent norm、log-Jacobian和base-density各自对 ranking的贡献。
7. **Rare-benign hard set**：同 topic的 reporting/quotation、少数语言、低信噪音频与罕见剪辑；检查低 likelihood是否主要预测 rarity。
8. **Positive broadcast control**：禁止 full-video pooled feature进入 proposal encoder；做 temporal shuffle、proposal外 context置换和全视频 interval control。
9. **Topic-support stratification**：按 negative-train topic support/effective sample count报告 test score；性能若只在高/低 support组由 rarity决定，机制失败。

### 对总 verdict 的影响

该 reformulation **不改变 PASS/FAIL总裁定，但把推荐 claim进一步收窄**：

> A target-conditioned one-class proposal MIL adaptation for hateful-video localization, where a conditional normal model learned from benign training proposals scores hostile relational residuals relative to the same protected-target/topic context, and positive video labels orient at least one low-normal-likelihood temporal proposal.

它比原 `p_hate/p_normal` ratio更适合作为 pilot，因为 observable negative proposals足以训练 `p0`；但它比 retrieval版更接近 CFLOW-AD、ContextFlow++、flow-based VAD和 OpenVAD 的已知组合。只有 **protected-target conditionalization + relational residual + proposal MIL** 三者作为一个不可拆的 hate-specific adaptation 可以主张。若 conditional flow不超过 matched kNN、unconditional flow和 frozen one-class controls，应直接判定这只是 CFLOW/WSVAD 的任务移植，novelty转为 **FAIL**。

## 最终决策

**PASS，6.2/10：** 允许把 P-MIL 迁移到 hateful localization，并以 target-matched benign proposal comparison作为 load-bearing、task-specific adaptation。当前检索没有发现完整机制的目标任务先例，且 `unconditional_reference`、`target_shuffled`、faithful P-MIL 与 retrieval-only controls可直接证伪它是否超过组件拼接。

**硬限制：** 进入实现前必须把 numerator、reference weighting、leave-one-video-out bank isolation、无邻居行为和 overlap-to-frame readout写成确定公式。若最终实现只是检索 feature + P-MIL classifier，或无法在 matched controls中证明 target-conditioned benign comparison有贡献，novelty verdict自动转为 **FAIL**。即使性能通过，没有额外假设也只能声称 contrastive energy，不能声称已估计真实 conditional likelihood ratio。
