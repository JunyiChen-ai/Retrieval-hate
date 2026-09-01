# Novelty and anti-pattern review: typed REBA adaptation

截至 2026-08-31。审查对象：本目录 `README.md`、`model.py`，官方
`third_party/REBA-WSVAD` 实现，以及 CVPR Findings 2026 REBA 论文。未运行训练或评测。

## Verdict

**FAIL**。

失败的不是“来源任务已经做过 hateful localization”。公开论文和官方仓库只在
UCF-Crime 与 XD-Violence 上报告，跨到 hateful video temporal localization 的前提成立。
失败原因是当前新增部分仍是三个常见组件的串接，而且其中两个新增机制与声称要解决的
modality ownership 问题相冲突：

1. modality-typed RMoE 是把 REBA 的 residual multi-scale gating 分别复制到三个模态，随后做
   普通 learned late fusion；
2. 所谓 class-aware multi-positive BiAlign 实际只使用二元 video label，把所有 hateful 视频视为
   同一语义类，并无 hate target、攻击类型或 modality ownership 类别；
3. occupancy pooling 在当前公式下会退化成按 video label 在 mean 与 peak-like pooling 之间选边，
   不是可辨识的事件占比机制。

因此本实现可以保留为 **REBA-inspired multimodal adaptation baseline**，但不满足当前标准下的
non-trivial task-mechanism claim。

## Primary-source comparison

REBA 原论文将贡献定义为视觉 CLIP snippet 上的 RMoE temporal adapter，以及 video representation
与 anomaly-class text prompts 之间的双向对齐；训练仍使用 top-k MIL。官方 RMoE 对同一视觉序列做
1/2/3 倍时间降采样、分别经过 temporal/graph expert、上采样后以 residual gate 融合。官方 BiAlign
用 anomaly logits 对视频特征做时间池化，再与对应类别的 text prompt 对齐。

来源：

- [REBA, CVPR Findings 2026](https://openaccess.thecvf.com/content/CVPR2026F/html/Chu_REBA_Residual_Mixture-of-Experts_and_Bidirectional_Video-Text_Alignment_for_Better_Fine-grained_CVPRF_2026_paper.html)
- 官方实现：`third_party/REBA-WSVAD/model.py`、`third_party/REBA-WSVAD/BiAlign.py`
- multi-positive supervised contrastive grouping 本身已有标准先例：[Supervised Contrastive Learning](https://arxiv.org/abs/2004.11362)
- learned attention pooling 也不是新机制：[Attention-based Deep Multiple Instance Learning](https://proceedings.mlr.press/v80/ilse18a.html)

本候选确实没有逐行复制官方模型。它把官方降采样 Transformer/graph experts 换成全分辨率的
dilated depthwise temporal convolutions，并为 audio、visual、text 建立不共享参数的三个 expert
families。这是实质性的工程改写，但“给每个模态各放一套 temporal encoder，再用 gate 融合”本身
不足以构成 hateful-specific mechanism。

## Blocking findings

### 1. BiAlign 重复了错误的 label semantics

`model.py:153-158` 先无条件平均 audio 与 visual embedding，再把这个 AV representation 与
transcript representation 对齐。`model.py:160` 的 positive mask 仅由二元 video labels 相等得到。

这意味着：

- 只在 transcript 中出现 hate 的视频，仍被要求让 benign audio/visual representation 对齐 hateful
  transcript；
- 只在视觉或音频中出现 hate 时，text 分支同样被要求承担该视频标签；
- 所有 hateful 视频互为 positives，即使其 target、攻击形式、语言和实际 witness modality 完全不同；
- 一个 batch 若只有同一二元类别，loss 恰为零。现有单测还把这一点当作期望行为。

这不是 ownership-aware alignment，而是对所有模态重新复制 video label。它直接违背 README
所引用的 modality ownership error analysis。避免官方 instance-pair false negatives 是合理修改，
但把所有同一二元标签样本设成 positives 并没有足够的语义依据。

此外，alignment 使用 detached frame logits 产生时间权重，因此 alignment loss 不能修正“哪些时间点
应作为 witness”的选择；modality gate 也不进入 alignment 计算。该 loss 对 ownership gate 没有直接
监督路径。

### 2. `class-aware` claim 过宽

当前数据传入的是 `labels` 的 0/1 video label，不是 hate class、target class 或 modality type。
准确名称只能是 **binary-label multi-positive AV-transcript contrastive regularization**。不能声称
class-aware hateful alignment，也不能声称它识别了 modality ownership。

### 3. Occupancy mixture 在当前目标下结构性退化

令 `m` 为 frame probability 的均值，`q` 为用同一 logits softmax 加权后的均值。因为 sigmoid
probability 随 logits 单调增加，通常有 `q >= m`。当前 bag probability 为：

`b = o * m + (1 - o) * q`。

对正 bag，BCE 会推动 `o` 减小，从而选择较大的 `q`；对负 bag，BCE 会推动 `o` 增大，从而选择较小的
`m`。所以 `o` 最容易学习成 video-label-dependent mean/peak selector，而不是事件 occupancy。
模型没有任何约束让 `o` 对应 hateful frames 的真实比例，也没有 duration、count 或 latent witness
likelihood 语义。

这比固定 top-k 少了显式 quota，但没有消除 peak-selection bias，只是把 mean/softmax 的选择交给了
另一个 video classifier。因而 README 中“由训练标签学习的 occupancy”表述过强。

### 4. Occupancy 不能改善主定位排序

最终输出为 `score = bag_probability * frame_probability`。对同一视频，`bag_probability` 是严格为正的
常数，因此不会改变任何 frame pair 的次序，within-video ROC 完全不变。它只能改变跨视频标度，影响
pooled ROC/AP；从功能上看更接近 video-global score scaling，而不是 localization mechanism。

README 所称“同一 frame probability 产生最终 score”也不准确：实际 score 额外乘了 video-global
bag probability。若项目禁止 calibration-style global rescaling，应直接用 `frame_probability` 评测，
或把该操作明确登记为模型内 video-level scaling，不能把它归因于边界定位。

### 5. Typed RMoE 与 ownership story 没有形成可归因耦合

三个模态有独立 projectors 和 temporal experts，这允许不同时间尺度，但 learned modality gate 只从
最终 bag BCE 获得间接信号。没有 witness exclusivity、latent modality responsibility、modality-specific
negative evidence 或防 gate collapse 的机制。当前 BiAlign 反而要求 AV 与 text 一致。

所以完整模型是：REBA-style temporal residual experts + standard modality attention + supervised
contrastive regularizer + adaptive MIL pooling。三者可以分别移除，彼此没有一个由 hateful
multimodal label semantics 推导出的联合约束，属于 component composition。

## Narrow claim that remains defensible

如果不改当前机制，只能主张：

> A REBA-inspired multimodal baseline for hateful temporal localization that applies separate
> residual dilated temporal expert families to audio, visual, and transcript features, performs
> learned late fusion, and adds binary-label multi-positive AV-transcript contrastive regularization.

不能主张：

- 新的 RMoE、BiAlign、MIL pooling 或 supervised contrastive objective；
- class-aware hateful alignment；
- modality ownership learning；
- occupancy estimation；
- occupancy 对 within-video localization 的贡献；
- 对 REBA 的忠实复现。候选 experts 是 dilation-based 改写，而非官方 downsampled
  Transformer/graph experts。

## What would be required for PASS

至少需要一个 load-bearing、由任务 label semantics 推导的结构化修改，而不是继续增加组件。例如：

1. alignment 必须允许 hate witness 只属于一个或部分模态，不能对正视频无条件强迫 AV 与 transcript
   一致；需要显式 latent modality responsibility 或 witness-conditioned agreement；
2. modality responsibility 必须实际进入 bag likelihood，并能在训练时阻止把 positive video label
   复制到所有 modality branches；
3. pooling 需要可解释的 latent witness/occupancy likelihood，或删除 occupancy claim。仅学习
   mean/softmax convex gate 不足；
4. 最终 frame score不得依赖只改变视频间尺度、却不改变视频内排序的模块来声称 localization 改进。

最低 attribution controls 应包括：共享单个 RMoE 对比 typed RMoE、无 modality gate、无 BiAlign、
官方 instance-pair BiAlign、当前 binary multi-positive BiAlign、mean/softmax/fixed top-k pooling、直接
`frame_probability` 对比乘 bag score，以及 modality-gate collapse 和 test-GT ownership 匹配分析。
这些 controls 可以判断模块贡献，但不能修复当前 claim 的语义问题。

## Final decision

**FAIL as a novel task adaptation; acceptable only as a clearly labeled cross-task baseline.**

在重新设计 ownership-aware training semantics 之前，不建议按当前 README 的 novelty claim 启动正式
训练。该结论只审查机制与代码，不使用 validation 或 test performance 作判断。
