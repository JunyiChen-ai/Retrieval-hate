# 可加统计量 inside–outside backbone：独立 proposal review

日期：2026-10-01。评审人：独立 Codex agent。本候选唯一一次规则 4 提案评审；依据 `experiments/20261001_associative_io_backbone/README.md` 与 `RESEARCH_ITERATION_RULES.md` 第 4 条。within 按最新 AGENTS.md 与 AP / ROC 并列主指标。本次没有实现、训练、修改规则或改动在跑实验。

**结论：GO。** 实际联网检索并核对原论文后，未发现本候选完整机制已经直接用于 hateful video detection/localization 的证据；其余三类 STOP 条件也未触发。此结论只允许实现检验，不等于已建立有效的 backbone novelty，也不支持世界首次主张。

## 1. 评审对象与来源

候选从已查看 test 的开发诊断出发，保留目标区间与区间外上下文的比较，改用点式正权重形成 `S=sum(w*x)`、`Z=sum(w)`。树内及树外传播只合并这些统计量，在节点读出时归一化、融合模态，输出既有问题树模型所需的内容先验。四个 attention head 属于同一可训练骨干。

新方法并不发明 attention pooling、可加统计量或 inside–outside。应明确引用以下来源：

| 原始来源与核对位置 | 已有机制及本候选边界 |
|---|---|
| [IORNN，Le & Zuidema，EMNLP 2014，§4.2](https://aclanthology.org/D14-1081.pdf) | 子节点形成 inside、父 outside 与兄弟 inside 形成 outside，以及可学习根上下文均已有先例。本候选改为固定时间区间和可加状态，不可把双向树表示本身当作发明。 |
| [Attention-based Deep Multiple Instance Learning，Ilse et al.，ICML 2018，§2.4](https://proceedings.mlr.press/v80/ilse18a/ilse18a.pdf) | 神经网络产生实例权重并进行归一化加权池化已有先例。候选节点的 `S/Z` 属于这类加权汇聚，单独使用它不构成贡献。 |
| [Transformers are RNNs，Katharopoulos et al.，ICML 2020，§3.3、式10–12与18–20](https://proceedings.mlr.press/v119/katharopoulos20a/katharopoulos20a.pdf) | 已通过可累计分子状态和归一化状态实现线性 attention。候选不是完整 query–key 线性 Transformer，但不能主张发明累计 attention 状态或延迟归一化。 |
| [Deep Sets，Zaheer et al.，NeurIPS 2017](https://arxiv.org/abs/1703.06114) | 点式映射、集合求和与后续读出是已有集合表示范式。候选的可加性与集合汇聚没有独立首次性。 |

提案诊断支持“值得检验聚合方式”，不证明递归非线性压缩已被识别为性能下降的原因。尤其去 outside 的 AP 差异主要由一个 seed 驱动，不能写成各 seed 稳定有效。本评审不重复转录结果；数值及其原始评测输出位置见候选 README。

## 2. 实际检索及 hateful-video 方法核对

本次重新执行联网 search/open/find，未仅沿用初版评审。检索词包括：

- `"hateful video" "attention pooling"`、`"hateful video" "sufficient statistics"`、`"hateful video" "associative"`。
- `"hateful video" "inside outside"`、`"hateful video" "inside-outside"`、`"hateful video" "inside" "outside" neural`。
- `"hateful video" "linear attention"`、`"HateMM" "linear attention"`、`"hateful video" "Deep Sets"`。
- `"hateful video" "complement" attention`、`"hateful video localization" backbone attention`。

精确组合词未返回所提完整机制的直接应用；部分搜索返回无关页面或二手摘要，未以这些摘要作结论。随后直接打开下列原论文并核对方法段：

| Hateful-video 原始来源 | 相关机制与判定 |
|---|---|
| [MultiHateGNN，作者论文，§3.3–3.5](https://arxiv.org/html/2509.13515v1) | 全局权重图产生经 softmax 的重要性，局部实例子图产生表示，再作加权聚合。已有结构化局部/全局建模及加权池化；没有本候选的逐区间互补 outside 或可加内外状态。不能声称 hateful-video 首次采用加权聚合。 |
| [SAGE，ACL 2026，§3.2–3.3](https://aclanthology.org/2026.acl-long.817.pdf) | 自注意力、全模态上下文跨注意力、attention pooling 和模态决策仲裁。其全局上下文包含当前模态内容，没有排除本区间缓存行的 outside。该论文进一步否定“首次 attention pooling”的宽泛主张。 |
| [MultiHateLoc，作者论文 v3，§3.2–3.5](https://arxiv.org/html/2512.10408v3) | 模态专属时间 Transformer、跨模态对比与动态融合、MIL；未采用可加统计量的区间 inside–outside 编码。 |
| [RAMF，作者论文 v2，§3.5–3.6](https://arxiv.org/html/2512.02743v2) | Conv1D / max pooling 局部分支和全局平均分支经门控融合，再作跨模态注意力。全局分支包含局部，不能等同于区间补集。 |
| [CLARA，作者论文，§3.3–3.4](https://arxiv.org/html/2608.15905v1) | 短长片段对比与 rationale-gated Transformer；局部、全局片段独立采样，不要求包含，也不要求互补。未采用本候选的可加内外状态。 |
| [LELA，作者论文，§3.2–3.4](https://arxiv.org/html/2602.09637v1) | 多模态描述与多阶段 LLM 提示产生逐帧分数；不是可训练的区间互补统计编码器。 |
| [TANDEM，作者论文，§3.1–3.3](https://arxiv.org/html/2601.11178v1) | 分块音视频语言模型生成时间戳并交替强化学习；不是候选所述单骨干内外统计表示。 |

“未找到”限于上述检索与原文核对范围，不应转写成全领域不存在。

## 3. 四项 STOP 判定

| 规则 4 条件 | 结论 | 依据 |
|---|---|---|
| 来源方法已用于 hateful-video detection/localization | 未触发 | 一般算子已有使用，但未检索到“点式可合并统计量 + 严格区间补集 outside + 问题树答案似然训练”的完整来源机制直接应用。 |
| 纯 training/test ensemble | 未触发 | 一个共同训练的骨干内部四个 head，不组合多个独立模型结果。 |
| 纯 calibration / 后处理 / 平滑 | 未触发 | 改变内容编码、节点势和训练计算图，最终仍调用现有答案后验，没有事后分数变换。 |
| 纯工程技巧 | 未触发 | 这是从开发诊断提出的可训练表征假设，改变递归内容压缩及区间内外表示；不只是优化相同数学输出的求和实现、换超参、换特征或训练配置。 |

不以可能退化、shortcut 或可识别性推理阻断提案。

## 4. 可主张范围与非阻断限制

建议主张范围是：**在弱监督 hateful video localization 的问题树上，以可合并 attention 统计量形成区间内与区间外表示，并通过既有多尺度答案似然学习内容先验。** 是否构成有效贡献，要由两主数据集的完整搜索、三项主指标及机制消融决定。

- 可加传播在实数算术中等价于直接对节点区间及其补集做加权汇聚。树提供区间集合和计算组织；不能再声称该部分具有初版左右有序的非线性组合表达力，也不能把不同求和顺序当成新的模型家族。浮点计算仍可能有舍入差异。
- “保留统计量”只指所选择的 `S/Z` 累计量，不代表保留全部原始信息；归一化池化仍是压缩。避免“无损表征”“保证消除信息损失”等表述。
- outside 排除仅针对当前输入缓存行；上游 I3D/BERT 感受野可能跨行，outside 也不等于无害背景。权重点式生成、根空上下文处理和最终读出是否符合声明，交给唯一一次 code review 核验。
- 四个 head 不自动证明更有效。README 已提出去 outside、常数权重及与递归组合的比较，可用于区分互补上下文、可加汇聚和参数容量带来的变化；本评审不增加晋级门或 matched-control 要求。
- 新增 VLM 调用与缓存抽取均为 0，新视频沿用原查询成本。可加状态的额外显存与训练时间仍须首个完整 trial 实测，不能由线性复杂度推断一定更快。

GO 后可进入实现和规则 6 的一次 code review；本评审未启动任何训练，也未更改 Pursuing goal 状态。
