# 候选9独立 proposal review（规则4，一次）

日期：2026-09-06。审阅者：独立 agent `proposal_review_c9`。依据：完整阅读 `AGENTS.md`、`RESEARCH_ITERATION_RULES.md` 及本目录 README；最终裁定针对本次 review 内已修订的 **latent ordinal observation channel** 版本，不针对最初累计码版本。本记录只作提案评审，不作代码评审或效果认证。

## 结论：GO

未发现规则4四种 STOP 的适用证据。该提案有可明确检验的整体机制：区间观察更新产生 query，同一个区间—内容分配承担双向模态传播及最终条件读出的边缘化；同时给定视频监督和更新前观察似然。它改变 forward、监督目标和读出，而不只是换特征、超参或训练配置。**GO 不等于已经实现三模块 novelty、有效性或整体 novel paradigm。**

| 规则4检查 | 裁定及依据 |
| --- | --- |
| 来源方法已用于 hateful video detection/localization | 未找到上述完整结构已被采用的证据；邻近方法确有 VLM、cross-attention、MoE、局部/全局交互，不能泛称这些首次用于 hateful video。检索边界见下。 |
| 纯 training/test ensemble | 否：一套共享内容网络和条件分类器；等级与区间是同一模型的潜在变量，不是独立模型的预测聚合。此结论以实现维持共享单模型为前提。 |
| 纯 calibration/后处理/平滑 | 否：观察更新在特征 query 和可训练骨干之前发生，参与视频损失梯度；最终读出是网络 forward，不是对已产生分数作外部修正。 |
| 纯工程技巧 | 否：完整学习机制改变观察表达、区间内归因、跨模态传播和预测组合，且有分别针对机制的替换实验；不以数学术语数量作为创新证据。 |

## 实际检索与 primary sources

检索式包含 `hateful video localization ordinal attention`、`hateful video interval attention`、`"hateful video" "ordinal regression"`、`"hateful video" "bipartite"`、`"hateful video" "marginalization"`、`"hateful video" "interval" "latent"`、`"hateful video" "Set Transformer"`、`"hateful video" "CORAL"`、`"hateful video" "confusion matrix"`、`"hateful video" "label noise" "transition matrix"`，并打开以下作者论文/正式出版页面核查。检索未命中不是对全部文献不存在的证明。

- [MultiHateLoc](https://arxiv.org/html/2512.10408v2)，§3.3–3.5：已有跨模态对比、模态重加权、拼接特征的 Q/K/V 注意力及 modality-aware top-K MIL。其全时间注意力和模态分支监督不是本提案的“VLM 区间观察—同一受覆盖约束分配—共享条件读出”。因此 attention、MIL、跨模态融合本身都不是 C9 的新贡献。
- [LELA](https://arxiv.org/html/2602.09637v1)，§3：多模态 caption 分解、composition matching 和多阶段 LLM 打分；已说明利用语言模型做 hate localization 并非首次。未见本提案的端到端观察通道及区间分配学习。
- [RAMF](https://arxiv.org/html/2512.02743v1)，§3：local-global fusion、semantic cross-attention，以及描述/hate/non-hate 三阶段推理；不是同一序数区间分配的传播与边缘化。
- [CLARA](https://arxiv.org/html/2608.15905v2)，§3 和附录 B：2026-08 的邻近工作，已有 clip-level MoE、local-global segment contrast、视频级 VLM rationale 和 gated Transformer。故“clip-level + VLM + attention”不可当作整体 novelty；检查到的结构不是 C9 的共享区间分配与观察更新。
- [HVGuard](https://aclanthology.org/2025.emnlp-main.456/)，作者摘要及论文：已有 MLLM CoT 和 MoE 融合；不能把 VLM reasoning 或自适应融合直接当新来源。
- [TANDEM](https://arxiv.org/html/2601.11178v1)：跨模态上下文和 VL/AL tandem RL 的结构化时间定位，不是本提案单模型的潜在区间条件分类器。
- [Revealing Temporal Label Noise](https://arxiv.org/html/2508.04900v1)，§5：研究裁切/完整视频的标签噪声影响，固定 baseline 架构对照数据组成；观察到噪声问题不等于已经采用 C9 的序数发射与后验 query。
- [HateClipSeg](https://arxiv.org/html/2508.01712v2)：核查数据集/任务论文，不能将提出细粒度任务本身作为 C9 创新。

机制基础须如实引用：

- [Patrini et al., CVPR 2017](https://openaccess.thecvf.com/content_cvpr_2017/html/Patrini_Making_Deep_Neural_CVPR_2017_paper.html) 已有由类条件噪声转移得到观测预测的 forward loss correction。C9 的 `sum_z pi(z) E(g|z)` 属于相关的既有观察模型思想，不应声称新概率公式；内容条件后验作为区间分配输入才是当前组合假设的一部分。
- [Set Transformer, ICML 2019](https://proceedings.mlr.press/v97/lee19d.html) 已有通过中间 inducing representation 降低注意力复杂度。区间往返注意力不是凭“线性于 T”就新；C9 的具体差别是观测驱动、时间覆盖约束及分配复用于读出。
- [CORAL](https://arxiv.org/abs/1901.07884) 已有累计等级监督；只把四等级换成累计码不能构成独立方法。本次修订已不再使用原阈值重构头。

## M1 主张边界与本次修订

原始 `e(g)=b+sum_{r<=g} d_r` 若增量向量完全自由，任意四类别 embedding 都可写成 `b=e(0), d_r=e(r)-e(r-1)`。因此原 `categorical_observation` 替换没有隔离实质机制，最多测量重参数化/优化差异，不能作为 M1 创新证据。这是具体主张边界说明，不以“可能退化”阻断完整候选。

修订后的 M1 增加 `pi(z|content)`、序数发射 `E(g|z)` 和后验条件区间 query，不再只是上述编码替换；主替换 `hard_observation` 与辅助 `categorical_noise` 分别检验软观察更新和序数距离约束。仍不可声称 tau 是真实可辨识噪声，或 latent z 是被证明恢复的真实 hate 等级。Bayes 更新与噪声通道的数学不是新发明；能否构成有效任务迁移由 test 消融判断。

## 非阻断的实现/表述提醒

- 观察 NLL 必须来自未读取该 grade 的内容 prior 边缘概率；不得用输入 grade 条件化的 posterior 再重构同一 grade 冒充监督。`hard_observation` 保留相同观察 loss；`no_vlm` 不读 grade 且不保留其监督。分别检查全部 split，不把 test GT 用于任何 prior 训练。
- M2 的 `uniform_assignment` 会同时影响传播及 R，这是共享分配机制的整体替换，不是仅骨干某层的独立定位。M3 的 additive 替换同时去掉条件交互与概率边缘化；若有效，只能支持联合读出这一整项，不能分别声称两个效果。
- 新增 VLM 抽取为 0；部署 34 次是沿用原 M1 成本，不是零成本，也不是候选7的每视频120次。分配点积的 `O(34 T H)` 不含稠密条件 MLP 的潜在 `O(34 T H²)`；实现可仅计算真实覆盖的 `(t,j)` 对，不改变模型。完整首 trial 记录实际耗时，不能先承诺明显提速。
- 三模块有效性、两语料三 seed 和规则14(g)仍未完成。当前不得汇报 SOTA 完成或三个 novelty 已成立；不用额外理论门阻止实现和固定预算筛选。
