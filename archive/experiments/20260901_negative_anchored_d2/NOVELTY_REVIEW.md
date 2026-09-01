# Novelty review: Negative-anchored D2 denoising MIL

截至 2026-09-01。独立 reviewer 只审候选 brief、研究规则与 `STATUS.md` failure ledger，未审代码。

## 裁定

**STOP，4.8/10。** Rule 12 Gate 1、Gate 2 通过，Gate 3 失败。D2-Net 可以作为跨任务来源，窄检索也未发现 D2-Net/pDMI bottom-up denoising 已用于 hateful-video detection/localization；但本候选相对 source D2 的唯一任务改造，是把 binary MIL 中“negative bag 的所有 instance 均为 negative”这一标准逻辑写成 bottom-up head 的逐帧 BCE。其余承担定位作用的 foreground/background embedding、bottom-up attention、snippet/video pDMI 以及乘积式 test score 均来自 D2-Net。因此当前形式仍是 **D2-Net 直接换到 hateful-video binary MIL，再加 ordinary dense-negative BCE**，不满足 non-trivial task adaptation 硬门。

## 实际检索与来源核对

检索组合包括 `D2-Net + hateful/hate video`、`pseudo-Determinant/DMI + hateful video`、`bottom-up foreground attention + hateful video`、`negative video + bottom-up attention + hateful localization`，并核对以下原始论文或作者发布页：

- [D2-Net, ICCV 2021](https://openaccess.thecvf.com/content/ICCV2021/html/Narayan_D2-Net_Weakly-Supervised_Action_Localization_via_Discriminative_Embeddings_and_Denoised_Activations_ICCV_2021_paper.html)：来源任务是 video-label-only weakly supervised temporal action localization。原方法已经包含 top-down activation 加权的 foreground/background embeddings、独立 bottom-up attention、intra-video 与 inter-video pDMI denoising，以及 foreground/background discrimination。
- [D2-Net arXiv record](https://arxiv.org/abs/2012.06440)：标题、任务和方法归属与 ICCV 版本一致；检索未出现其在 hateful-video detection/localization 上的应用。
- [DMI loss, NeurIPS 2019 preprint](https://arxiv.org/abs/1909.03388)：DMI 本身是面向 noisy-label classification 的通用 loss；D2-Net 已经完成了把它改造成弱监督时序定位 pDMI 的关键跨任务 adaptation。
- [ImpliHateVid, ACL 2025](https://aclanthology.org/2025.acl-long.842/)：contrastive representation learning 已明确用于 hateful-video detection，并在 HateMM 上评估；所以不能把一般的 representation separation、contrast 或 MI regularization 当作本候选的新颖核心。
- [CLARA](https://arxiv.org/abs/2608.15905)：local-global segment contrastive objective 也已用于 clip-level hateful-video detection，进一步限定了可主张范围。
- [HateClipSeg](https://www.researchgate.net/publication/394293029_HateClipSeg_A_Segment-Level_Annotated_Dataset_for_Fine-Grained_Hate_Video_Detection)：其 temporal hateful-video localization 基线采用 ActionFormer，并明确把 TAL 迁移到该任务；未见 D2-Net、pDMI 或 negative-video anchored bottom-up attention。该结果支持 Gate 2 的窄 PASS，但并不自动使任意 TAL adaptation 满足 Gate 3。

这是有边界的公开检索结论，不表述为对所有未公开工作或所有数据库的绝对不存在证明。

## 三项硬门

### Gate 1：允许 adaptation 已有方法

**PASS。** 候选明确承认 D2-Net 来源，没有把 source-owned 部件表述为从零开发。

### Gate 2：来源方法未被 hateful-video task 占用

**PASS（窄口径）。** 未检出 D2-Net、其 pDMI denoising，或“negative-video anchored bottom-up attention”这一精确组合用于 hateful-video detection/localization。目标领域已经占用一般 contrastive learning，也已使用其他 TAL 模型，但这不等于 D2-Net 精确来源已被占用。

### Gate 3：non-trivial hateful-task adaptation

**FAIL。** 原因不是一般性的 broadcast shortcut，也不是要求实现前证明完整 identifiability，而是候选 brief 自身把 task delta 定义成了一个标准 MIL supervision 展开：

1. D2-Net 已经提供 top-down foreground/background separation、bottom-up attention、snippet/video pDMI 与最终 activation denoising；这些不是当前 adaptation 的新机制。
2. Binary MIL 的标准假设本来就是 positive bag 至少含一个 positive instance，而 negative bag 的全部 instances 均为 negative。对 negative-video 的全部有效帧施加 background BCE，只是把这个已知 bag implication 直接用作 dense instance loss。
3. `positive-unselected abstention` 在这里等价于“不对 positive bag 内未知 instance 施加 dense label”；这是避免错误伪标签的必要实现选择，但没有增加新的可学习约束。
4. 因而“asymmetric certified-negative anchor”在当前目标式中不是一个独立、非平凡的 hateful-localization mechanism；它是 ordinary dense-negative supervision 的命名。任务故事合理，但合理性不等于 novelty。

## 与 failure ledger 的逐项比较

| 近邻 | 既有核心 | 当前候选的关系 | 裁定影响 |
|---|---|---|---|
| benign insertion | 从 negative bag 取 certified benign 内容，放入 positive recipient，并施加局部 benign 约束 | 当前不做 transplant，但使用同一 certified-negative local supervision，且约束更简单 | 没有提供超出既有 negative certificate 的新干预或不变性 |
| dense-negative marked splat | 用 negative evidence 形成 dense local negative learning signal；已完成双 test 后失败并关闭 dense-negative 变体 | 当前把相同信息源接到 D2 bottom-up head，再换成 BCE/DMI 表达 | 触发 Rule 16：不能只换 head/loss/数学工具续命 |
| background prototype | 通过 background representation/prototype 组织 foreground-background separation | D2-Net 自身已经跨视频组织 foreground/background embeddings；当前没有新增 hateful-specific prototype constraint | background geometry 属于 source-owned 或既有 family，不是 task delta |
| binary CASE | 二元 latent foreground/background clustering，相对 wider policy 更强但仍未过性能门 | 当前改用 bottom-up sigmoid 与 DMI，不新增独立 correction signal | 数学形式不同，监督结构仍是 binary foreground/background separation |
| contrast / DMI 链 | 目标领域已有 contrastive hateful-video detection；项目 lexical DCC 已关闭。DMI 的 localization adaptation 又由 D2-Net 自身完成 | 当前不重复 lexical producer，但不能把 generic separation/pDMI 当 novel 部分 | 可主张范围只剩 negative BCE，而该部分是标准 MIL |

Rule 16 在这里是实质问题：candidate 没有新增独立信息源或新 premise，只是用 D2 的 head 与 DMI 重新承载已失败的 certified dense-negative signal。Rule 21 的 available correction signal 条件倒是满足——train negative-video label 确实可用，也不是 test oracle 或 self-credit——但“信号可用”只证明合规性，不证明 adaptation non-trivial。

## `source_d2` control 是否足够

**对局部因果归因足够，对 novelty 归因不足。** 将 negative-anchor BCE 权重置零，同时保持 head、D2 losses、训练量、最终乘积 score 和 validation 选择一致，能够回答：在这个实现里，新增 negative BCE 是否对 test 结果 load-bearing。若 core 在 HMM/HCS 都胜 `source_d2`，可以说明 BCE 有增量效果。

但该 control 不能回答两个决定 novelty 的问题：

- 增益是否需要 D2 的 pDMI/discriminative mechanism，还是 `POWA + bottom-up head + dense-negative BCE` 就能得到；
- 该增益是否超出了标准 MIL negative-bag supervision 和项目既有 dense-negative/certified-benign 链。

理论上，前一个问题需要同容量的 `negative-BCE-only` factorial control；但即使这个额外 control 显示 D2 与 negative BCE 有 interaction，当前 brief 仍没有定义该 interaction 所对应的新 hateful-specific constraint。因此不建议为了补 control 而进入训练：Gate 3 已经失败，control 不能把普通监督项变成 non-trivial adaptation。

## 最终可执行结论

- **不得实现或进入 HMM/HCS 正式训练。**
- 不把 D2-Net、pDMI 或 bottom-up attention 本身记为已被 hateful-video task 占用；Gate 2 结论仍保留。
- 关闭当前“D2 source path + negative-video dense BCE”候选。不得仅通过改 BCE 权重、DMI 形式、background head、prototype 或乘积 readout 重开。
- 若未来重开，必须先出现一个不等价于“negative bags are all negative”的新约束或新 correction signal，并明确证明它在机制上改变 D2，而不是给 D2 追加标准 MIL loss。
