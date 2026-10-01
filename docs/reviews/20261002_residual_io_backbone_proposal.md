# 区间内外残差 backbone：独立 proposal review

日期：2026-10-02。审稿人：独立 Codex agent。本候选唯一一次规则 4 提案评审；完整阅读 `experiments/20261002_residual_io_backbone/README.md` 和 `RESEARCH_ITERATION_RULES.md`，前两版评审只作背景，本次重新联网检索并阅读原文。within 按最新 AGENTS.md 与 pooled AP / ROC 并列主指标。

**结论：GO。** 本次检索未找到所提完整机制已经直接用于 hateful video detection/localization 的证据，另外三类 STOP 条件也不成立。无必须先澄清才能实现的提案问题。GO 允许实现和一次 code review，不代表性能提升、有效机制贡献或世界首次已成立。

## 1. 评审对象

一个共同训练的网络保留第 5 版模态投影、跨模态时间编码和视频头。从同一组 pre-CMA 投影行计算固定时间树上的区间与补集 sum/count，经共享非线性读出产生内容残差，进入秒级表示及内部节点势，由原视频监督和问题答案似然共同训练。最后一层零初始化用于保留原表达能力，不作为新意。视频头不直接读取新增残差，但共享参数联合训练仍会改变视频输出。

提案与前两版不同之处是内容条件化的接入位置，以及保留原跨模态时间编码与视频读出；它没有新发明 inside–outside、集合均值、残差连接或归一化。不能把前两版消融贡献转移成本版证据。

## 2. 本次实际检索

工具：联网 search / open / find；日期同上。执行的主要查询如下：

- `hateful video localization inside outside context temporal tree HateClipSeg MultiHateLoc`
- `hateful video detection inside outside residual context tree neural network`
- `"hateful video" "inside-outside"`、`"hateful video" "inside outside" neural tree`
- `"hateful video" "complement" residual`、`"hateful video" "interval" "complement"`
- `"hateful video" "context" "residual"`、`"HateMM" "residual" network context`
- `"HateMM" "inside" "outside"`
- `DIORA inside outside recursive autoencoder Drozdov 2019`
- `"Deep Residual Learning for Image Recognition" arxiv`

精确内外/补集组合检索没有找到候选完整机制的直接应用；部分返回无关页面。残差查询确实返回已有 hateful-video 实现，因此本评审不作“首次将残差用于 hateful video”的主张。只将下列原论文或作者代码作为技术判断依据，不依据二手自动摘要。未找到仅限本次检索范围，不表示穷尽全领域。

## 3. 来源与最近先例

| 已阅读的原始来源 | 原文机制和本候选边界 |
|---|---|
| [IORNN，Le & Zuidema，EMNLP 2014，§4.2](https://aclanthology.org/D14-1081.pdf) | inside 从子节点自下而上组合；outside 从父 outside 与兄弟 inside 自上而下形成。这是本候选的结构来源，不能主张双向树表示为新发明。本候选使用固定时间区间和 sum/count，不采用其依存句法预测目标。 |
| [DIORA，Drozdov et al.，NAACL 2019](https://aclanthology.org/N19-1116/) | 在候选二叉句法树上学习 inside/outside 表示，通过其余词重构当前词。提供上下文排除与树表示的迁移背景；本候选没有潜在句法树诱导和词重构。 |
| [Deep Sets，Zaheer et al.，§2.2](https://arxiv.org/html/1703.06114v3) | 点式映射、求和、后续读出是既有集合表示范式。sum/count 形成区间及补集均值不能独立构成新意；树上的传递与直接对对应行集合求均值在实数算术中等价。 |
| [Deep Residual Learning，He et al.，原论文](https://arxiv.org/html/1512.03385v1) | 残差函数与原表示相加是成熟设计。本候选只能把残差当作接入方式，不能靠保留原网络和零初始化本身主张科研贡献。 |
| [MultiHateLoc，作者论文 v3，§3.2–3.5](https://arxiv.org/html/2512.10408v3) | 模态专属 Transformer、动态跨模态融合、对比与 top-K MIL；式 (1) 明确包含残差相加。没有所提固定区间补集状态及问题树答案似然条件化。残差和跨模态时间建模已用于 hateful-video，不能单独作为本候选新意。 |
| [RAMF，作者论文 v2，§3.5–3.6](https://arxiv.org/html/2512.02743v2) | 局部 Conv1D/max-pooling 与全局平均池化经门控融合，再作语义跨注意力。其全局均值包含局部输入，不是每个时间区间的严格缓存行补集。局部/全局融合本身已有 hateful-video 先例。 |
| [CLARA，作者论文，§3.3–3.4](https://arxiv.org/html/2608.15905v1) | 独立采样短、长片段，均值池化后做局部/全局对比，并用 rationale-gated Transformer 编码视频。原文不要求两个片段互补，亦无本候选的问题树条件化残差。均值片段与上下文建模均非空白方向。 |
| [MultiHateGNN，作者论文，§3.3–3.5](https://arxiv.org/html/2509.13515v1) | 全局图产生实例权重，局部子图产生实例表示，再加权分类；不是逐节点排除自身区间的 outside 或问题树答案似然。结构化局部/全局 hateful-video 建模本身已有先例。 |
| [作者公开 hateful-video 分类实现](https://github.com/sha9189/multimodal-hate-video-classification) | README 明列 `USE_RESIDUAL_BLOCKS`，并引用 HateMM。它足以进一步否定“残差首次用于 hateful-video”的宽泛说法，但不是本候选完整机制的直接先例，也不将它当作正式论文贡献证据。 |

补充访问记录：CVF 的 ResNet HTML 地址返回抓取错误，改读上述作者 arXiv 原文；SAGE 的 ACL 原 PDF 首次可取、后续定位返回错误，故未据其未核对的方法段作判断。

## 4. 四项 STOP 判定

| 规则 4 条件 | 判定 | 理由 |
|---|---|---|
| 来源方法已用于 hateful-video detection/localization | 未触发 | 组成算子早已使用；未找到“pre-CMA 区间及严格补集统计 + 秒级和问题节点内容条件化 + 答案似然联合训练”的完整机制直接应用。提案也未将残差或 local/global 融合作为唯一来源方法。 |
| 纯 training/test ensemble | 未触发 | 共享同一投影和 encoder，共同从头训练，残差进入同一表示和同一读出；没有多个独立模型的输出、特征或训练目标聚合。将单个线性头拆成两项相加的代数写法不构成 multi-model ensemble。 |
| 纯 calibration / 后处理 / 平滑 | 未触发 | 改动在可训练秒级及节点表示内，进入训练计算图；不是对最终分数作事后校准或平滑。 |
| 纯工程技巧 | 未触发 | 完整候选检验区间与补集内容如何共同条件化秒级和多尺度节点先验，超出了只调初始化、超参或优化求和实现。零初始化和保留旧视频头本身不算新意，但不抹去整个可检验的表征假设。 |

不以可能忽略残差、退化、shortcut 或可识别性推理增加提案阻断条件。

## 5. 新意边界与后续证据

可检验的主张限于：**在弱监督 hateful-video 问题树中，利用投影缓存行的区间/补集内容，通过共享非线性残差共同条件化秒级与节点先验，并由答案似然学习。** 不是新一类通用 inside–outside 算法，也不是首次残差、多尺度融合、均值汇聚或结构化 hateful-video 表征。

- outside 的排除只针对 pre-CMA 分支的输入缓存行；CMA 已混合全视频信息，且上游缓存可能有跨行感受野。不能声称整个节点表示对本区间隔离，也不能把 outside 自动解释为无害背景。
- 零初始化只约束初始残差输出，不保证训练轨迹或最终性能等于原版；保留视频头结构也不保证视频分数不变。此前同时更换多个部件的结果不足以证明损失由其中某个部件单独造成。
- 本版 `io_outside=false` 与 `io_residual=false` 的消融是区分补集信息和整体内容条件化的已有计划；各自的作用须由本版两语料完整结果判断，不借用前两版数字。within 与 AP/ROC 必须一起报告。这些说明不增加研究规则以外的新门槛。
- 复用原缓存，新增 VLM 调用和特征抽取均为 0；新视频仍承担原提问策略成本。新增树统计和小 MLP 的时间估计须由首个完整 trial 修正；不能由线性节点数推断一定更快或保证涨点。

本评审只新增此冻结记录，没有实现、训练、改动活动代码或更改 Pursuing Goal 状态。可以进入实现和规则 6 的唯一一次 code review。
