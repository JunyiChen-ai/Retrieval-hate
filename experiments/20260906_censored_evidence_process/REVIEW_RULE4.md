# 候选8独立 Proposal Review：GO

审查日期：2026-09-06。审查者：独立 agent `proposal_review_c8`。对象：本目录 README 中的带噪窗口观测、总强度/位置分离、窗口事件监督方案。已阅读现行研究规则；本次仅作规则4要求的一次 proposal review，不是性能或最终 novelty 认证。未运行训练或修改研究规则。

## 1. 四项裁定

| STOP 条件 | 核查结论 |
|---|---|
| 具体来源已用于 hateful video detection/localization | 本次实际检索未发现“尺度噪声通道＋内容总强度/局部分配＋嵌套窗口删失事件监督”这一具体迁移已用于该任务的证据；这是有限检索结论，不是全球首创证明。 |
| 纯 ensemble | 否。单一冻结 Qwen 只产生 train 观测；同一内容模型学习局部强度，部署不组合多个独立模型预测。 |
| 纯 calibration/后处理 | 否。噪声通道是训练观察模型，窗口损失反传到内容模型；部署直接读出同一模型局部强度，不作事后校准。 |
| 纯工程技巧 | 否。方案改变观察假设、强度参数化及弱监督目标，不是仅换缓存、超参或运行方式。 |

结论：**GO，允许实现并按规则6做一次独立 code review 后正式训练。** 不以可识别性、可能退化或 shortcut 推理新增 STOP 门。

## 2. 实际检索的一手来源及其边界

检索词覆盖 `hateful video Poisson/noisy-or/transition matrix`、`learning from aggregate observations`、`noisy label observation model`、`censored Poisson binary`、`Poisson intensity total density factorization`，并逐项检索当前 hateful-video 方法。仅以下论文/作者实现用于结论，未把搜索摘要聚合站当作证据。

- **噪声观察已有通用来源**：[Patrini 等，CVPR 2017，Making Deep Neural Networks Robust to Label Noise](https://openaccess.thecvf.com/content_cvpr_2017/html/Patrini_Making_Deep_Neural_CVPR_2017_paper.html) 建立预测经噪声转移矩阵后求观察损失的 forward correction。C8 的 `r+(q-r)P_W` 是二分类非对称观察通道的特例，不能声称发明噪声矩阵或 forward correction。C8 的适配点是 train-only VLM 的窗口事件裁定、每尺度单调通道及与局部事件质量共同学习。
- **时间事件与噪声标签已有来源**：[Adams 等，AISTATS 2017，Learning Time Series Detection Models from Temporally Imprecise Labels](https://proceedings.mlr.press/v54/adams17a.html) 对事件时间和标签值的噪声建立概率观察过程，并优化边缘似然；应用为移动健康。C8 不得笼统声称首次用噪声标签学习时间定位；其观察是窗口是否含事件而非不精确事件时间戳。
- **聚合观察学习已有完整框架**：[Zhang 等，NeurIPS 2020，Learning from Aggregate Observations](https://arxiv.org/abs/2004.06316) 以概率框架从集合级观察学习个体预测，MIL 是其中一种。C8 可引用该迁移方向，但不能把“只有窗口标签却输出细粒度定位”本身作为新范式首创。
- **总强度与归一化位置分离已有直接来源**：[Schluck、Wu、Srivastava，2021，Intensity Estimation for Poisson Process With Compositional Noise](https://www.frontiersin.org/journals/applied-mathematics-and-statistics/articles/10.3389/fams.2021.648984/full) 显式将强度分解为总强度与密度，应用于带时间扭曲的神经脉冲过程。C8 不是这篇论文的时间配准算法；这里只支持 `Lambda × normalized density` 并非新数学分解。候选主张只能是其内容驱动、弱标签与窗口监督下的具体参数化。
- **Noisy-OR MIL 是既有操作**：[Kraus 等，2016，Classifying and segmenting microscopy images with deep multiple instance learning](https://pmc.ncbi.nlm.nih.gov/articles/PMC4908336/) 明列并讨论 `1-product(1-p_t)`。令 `p_t=1-exp(-lambda_t)`，C8 的窗口事件概率正好得到这一形式；不能把改写成强度称为新 MIL 聚合函数。
- **二值化 Poisson 事件观察已有来源**：[Hyperevent network modelling of partially observed gossip data，JRSS C](https://academic.oup.com/jrsssc/advance-article/doi/10.1093/jrsssc/qlag041/8736898) 将至少一次事件的二值观察写为删失 Poisson/互补 log-log 二项模型。该论文不是视频模型，仅确认 `1-exp(-integrated intensity)` 与二值事件模型是既有统计结构。

### Hateful-video 已用方法核查

- [MultiHateLoc 原文 §3.5](https://arxiv.org/html/2512.10408v3) 与其[作者仓库](https://github.com/Multimodal-Intelligence-Lab-MIL/MultiHateLoc)：多模态时序编码、动态融合、模态/融合 top-K MIL 与训练平滑；没有找到上述尺度噪声观察/总量位置分离/嵌套事件监督的具体组合。它已经采用至少一个有害片段的任务定义，因此不能把该定义当作 novelty。
- [Revealing Temporal Label Noise in Multimodal Hateful Video Classification 作者仓库](https://github.com/Multimodal-Intelligence-Lab-MIL/HatefulVideoLabelNoise)：以人工时间戳裁剪及受控噪声实验研究粗标签影响。已在 hateful video 讨论 temporal label noise；不能声称首次发现该问题。公开方法描述不是 C8 的可学习 VLM 窗口误报/灵敏度通道。
- [RAMF 原文](https://arxiv.org/abs/2512.02743)：局部/全局融合、语义交叉注意力及 VLM 多阶段正反解释；与 C8 的 train-only 单次裁定观察模型不同。
- [MARS 原文](https://arxiv.org/abs/2601.15115) 及[作者实现](https://github.com/Multimodal-Intelligence-Lab-MIL/MARS)：训练免费、多阶段对抗理由生成与综合判断，不是内容模型的窗口事件似然训练。
- [MATCH 作者原文](https://jianlang.org/papers/MATCH.pdf)：多个 LMM proposer/verifier 产生和核验时空证据，再结合视频特征；没有提供本候选的具体强度监督方案。
- [CLARA 原文 §3.3–3.5](https://arxiv.org/html/2608.15905v1)：MoE clip encoder、local/global contrastive loss、VLM rationale gated Transformer；通用局部/全局与 VLM 引导已有，具体机制与 C8 不同。
- [TANDEM 原文](https://arxiv.org/abs/2601.11178)：视听语言模型的 tandem RL 和结构化时间证据，不是 C8 的观察通道或复合事件损失。

## 3. 可以检验的三个模块边界

1. **M1**：主替换 `hard_observation` 只能验证可学习非对称窗口噪声通道的贡献，不能验证 VLM 自身 novelty；`no_vlm` 用于单 teacher 整体贡献。通道数值不得被称为已辨识真实误报率/灵敏度。
2. **M2**：`unfactorized` 检验总量/位置显式分离这一整体参数化，而非卷积、softmax 或“局部全局建模”首创。基础骨干与输入相同，参数数量或初始化变化需如实披露。
3. **M3**：`topk_event` 在其它项固定下检验窗口观察的事件并集建模；它没有移除视频事件监督，因而不能单独证明“整个 Poisson 过程框架”有效。`fine_only` 是两尺度观察的辅助分析，多尺度本身不作新贡献。

整体最多先称“面向 hateful localization 的带噪聚合观察监督、总量/位置分离的局部事件强度学习”；三个模块及整体是否有可主张的贡献，仍取决于两语料三 seed 的完整主实验与对应消融。不能把三个已有通用公式直接包装为三项已成立的新理论。

## 4. 必须诚实描述，但不新增阻断门

- K30、K4 与视频观察具有交叠/包含关系。逐项 BCE 求和是**多尺度复合似然（composite likelihood）**，不是同一潜在 Poisson 过程全部观察的精确联合似然；窗口标签条件独立也不能仅从 Poisson 独立增量推得。更正此表述不改变训练方案。
- `lambda_t` 是对应区间的积分强度/质量，不是无单位的每秒率。重采样、窗口交叠与原始 snippet 分数必须使用一致单位；由 code review 检查，不能靠术语掩盖采样密度效应。
- factorized 视频损失主要决定总量，定位需依靠窗口观察及共享内容表示；观察通道是否收缩到低信息、within 是否不足是正式实验问题，不据此预先 STOP。
- 原始缓存已有并不让最初训练数据的 VLM 成本消失。应分别披露：原训练观测每视频34次已有单窗口调用、当前新增抽取0、部署VLM调用0，I3D/VGGish/BERT预处理仍存在。该方案未恢复用户拒绝的 C7 四路方案；不得在实验中偷偷改用四路缓存或重新抽取。

本文件仅变更本候选的审查记录；未触碰 CLAUDE.md、AGENTS.md、研究规则或其它候选代码。
