# 外部审稿（GPT6-Astra，reasoning ultra，Codex MCP，2026-09-07；threadId 01a07735-f1b1-78b2-9655-6dfc28a95b45）

Brief：`REVIEW_NOVELTY_REQUEST.md`。审稿人以只读方式打开了仓库文件并核对了原始 metrics。以下为审稿原文（未改动；文内绝对路径为审稿人所写）。

---

**候选 3 回应了"骨干完全没改"的批评，但仍不足以支持顶会方法论文。** 结构收益真实；"证据只决定聚合位置、内容表示始终纯净"的机制表述却超出了代码和消融证据。

全程只读，未修改文件、运行训练或执行 Git 命令。我读了[本次 brief](/home/jehc223/Retrieval-hate/experiments/20260904_evidence_guided_attention/REVIEW_NOVELTY_REQUEST.md)、[上次评审](/home/jehc223/Retrieval-hate/experiments/20260906_hier_evidence_clean/REVIEW_NOVELTY_GPT6ASTRA.md)、所引代码，并逐一读取六个完整模型及 72 个消融的原始 `metrics.json`，重新计算均值、标准差和差值。

**1．哪些旧批评被回答了，哪些没有**

核算结果如下，顺序为 AP／ROC／within：

| 语料 | 完整模型，三 seed 均值 ± 样本标准差 | full 相对同配置 AVCE 的差值 |
|---|---|---|
| HateMM | .667777±.009692／.850412±.004856／.623284±.015019 | +.044420／+.019416／−.008418 |
| HateClipSeg | .697610±.007562／.684304±.009455／.548838±.011085 | +.009315／+.025286／+.003423 |

六个 full trial 为 HateMM 的 17／8／13、HateClipSeg 的 13／12／16；完整原始路径见[消融汇总](/home/jehc223/Retrieval-hate/runs/20260904_evidence_guided_attention_rev2_noprune/ablations/three_seed_summary_both_corpora.json)，其中列出的 78 个指标文件与汇总一致。

- **"MACIL-SD 原封不动"：已回答。** [model.py](/home/jehc223/Retrieval-hate/experiments/20260904_evidence_guided_attention/model.py) 确实改变注意力输入、注意力分数和头前表示；同证据 AVCE 对照也补强了结构贡献。不过，AVCE 使用 full 选出的超参，因此证明的是这些配置下的收益，不能直接解释为独立优化后的架构优势。
- **"融合主要是既有标签模型"：未回答。** [verdict_hmm.py](/home/jehc223/Retrieval-hate/src/verdict_hmm.py) 仍是全局参数 HMM、粗块 OR 因子与标准 EM。最近先例仍包括 Snorkel、尤其处理多分辨率序列弱监督的 Dugong，以及 CHMM；具体观测结构可以是区别，尚不是充分的方法创新。
- **"HateClipSeg 训练必要性弱"：未回答。** [HMM-only 原评测](/home/jehc223/Retrieval-hate/runs/20260903_hier_evidence_mil/verdict_hmm_only/hateclipseg/test/metrics.json) 为 **.698152／.661037／.553658**；候选 3 的增量是 **−.000542／+.023267／−.004820**。ROC 增益值得报告，AP 与视频内排序没有改善。
- **"固定 30/4 缺乏依据"：未回答。**
- 按约定，搜索问题只说一句：[search.py](/home/jehc223/Retrieval-hate/experiments/20260904_evidence_guided_attention/search.py) 仍将 test 指标反馈给 TPE，因此这些数字属于开发上界，三 seed 标准差也不是锁定配置的训练不确定性。

A、B、C 的最近先例应分别讨论，不能全部归入泛泛的"提示注意力"：

| 部件 | 最近结构先例 | 当前区别与新颖性边界 |
|---|---|---|
| **A：证据进入 q/k，value 不加证据** | DETR 在注意力中将位置编码加入 q/k、保留内容 value；Graphormer 的离散属性嵌入；领域相关的是 VadCLIP、MLLM4WTAL 的语义知识引导 | 两粒度裁定组合及后验成为共享跨模态条件，是具体区别；**q/k 与 value 分开承载信息本身不新**。VadCLIP 不是这个算子的完全同构先例。 |
| **B：逐头 key 偏置** | Graphormer 的逐头结构偏置、T5／ALiBi 的加性注意力偏置；MLLM4WTAL 的外部知识调制注意力 | 新的是偏置来源于 VLM/HMM 证据。数学上仍是注意力乘以 `exp(β_h(e_j))` 后归一化；它不依赖 query，不能独立证明位置特定的证据选择。 |
| **C：视频级证据上下文** | GIG-VAD 的全局信息引导、PEL4VAD 的全局局部聚合，以及 MIL 的 bag 摘要广播 | 当前由裁定分布产生，区别明确；但在线性头下只是视频级加性 logit 项，没有新的内容交互。 |

相关工作比较结合了已有知识和本地[文献检索记录](/home/jehc223/Retrieval-hate/experiments/20260904_evidence_guided_attention/REVIEW_RULE4.md)，本次未在线复核论文全文。该记录主要针对修订 1，不能将其对输入编码的判断直接当作修订 2 的新颖性证明。

**2．机制主张是否成立**

**可以主张"限制证据直接进入局部 value 输入"，不能主张"内容表示始终纯内容"。**

代码中：

1. `q_in`、`k_in` 加入证据，value 输入及原始残差输入不直接加；
2. 注意力输出已经随证据改变；
3. `v_out`、`a_out` 随后都加上 `c`；
4. [train.py](/home/jehc223/Retrieval-hate/experiments/20260904_evidence_guided_attention/train.py) 将这两个**含 c 的表示**交给 CMAL；块 MIL 读取的所谓 `content_logit` 同样含证据。

设共享分类头权重为 \(w\)，则每个裁剪的最终 logit 可写为：

\[
\widetilde z_t
=z_{\text{attention},t}(X,E)+2w^\top c(E)+\alpha\ell_t/L .
\]

其中视频级项直接改变分数，且 `z_attention` 本身也是证据条件化的。这不是内容与证据的概率独立分解。

消融支持一个较窄的结论：**当前训练设置下，q/k 注入优于直接向残差流注入。** 它不支持统一的局部定位机制：

- 去 q/k：HateMM AP／ROC 降 **.03885／.02189**，within 只降 **.00107**；HateClipSeg 三项约降 **.001**。因此 q/k 的主要可见收益也不是视频内排序。
- HateMM full 比原候选 1 within 低 **.02321**，比同配置 AVCE 低 **.00842**。两种对照不能混写，但方向一致。
- "注意力让不同位置的表示趋同"是合理假设，尚未被当前三 seed 的注意力或表示分析证实。
- **92%／79% 来自候选 1 单 seed 的 z**，不是当前模型最终分数；我核对了两份原始[HateMM 分析](/home/jehc223/Retrieval-hate/runs/20260903_hier_evidence_mil/analysis/backbone_mechanism/hatemm_seed234.txt)和[HateClipSeg 分析](/home/jehc223/Retrieval-hate/runs/20260903_hier_evidence_mil/analysis/backbone_mechanism/hateclipseg_seed234.txt)。

最有价值的补充是不训练地分解当前三 seed 的视频间／视频内变化，并在固定 checkpoint 下，仅打乱 A/B 使用的证据时间位置、保持 C 和最终先验不变。若时间对应关系被破坏后收益仍保留，"证据指导局部聚合"的解释会受到直接反证。注意力热图本身不够。

**3．审稿人会重点攻击的证据**

- **均值成立不等于稳定机制。** HateMM seed 3407 去 bias 反而提升 AP **.00315**、ROC **.00204**。HateClipSeg seed 234 的 AP 不降实际是 **7 臂，不是 brief 的 8 臂**。这些不推翻均值，但要求报告配对视频不确定性；项目规定的 `.01` 是效果阈值，不是显著性检验。

- **两个关键消融被过度解释。** `scalar_bias` 把完整 `e_j→每头偏置` 换成 `γℓ_j`，同时改变输入信息、表达能力和头共享方式，不能隔离"逐头"的价值。`no_cell` 实际是**四列线性映射**，并非 brief 表格所写的"两列"。`mean_prior` 仍保留 HMM 块后验监督，因此也不是彻底去 HMM；见 [hier_evidence_common.py](/home/jehc223/Retrieval-hate/src/hier_evidence_common.py)。

- **HateClipSeg 基线弱需要解释。** 原评测阳性帧比例为 **.525506**；VadCLIP AP／ROC 为 **.524703／.503482**，MACIL-SD 为 **.515896／.476528**。我核对了对应三个 seed 的 `frame_eval.json`，包括 [VadCLIP](/home/jehc223/Retrieval-hate/results/reproduction/official_val/final/vadclip/hateclipseg/seed_234/frame_eval.json)、[MACIL-SD](/home/jehc223/Retrieval-hate/results/reproduction/official_val/final/macilsd/hateclipseg/seed_234/frame_eval.json)，与[基线表](/home/jehc223/Retrieval-hate/docs/duplex/OFFICIAL_VAL_RESULTS.md)一致。数值真实，却不能排除额外语义输入和任务适配是主要优势；同输入、合理优化的简单对照仍重要。

- **层次与调用成本的必要性不足。** HateClipSeg 历史 fine-only 已有 **.696501／.662025／.564042**。HateMM 历史 coarse-only 为 **.590814／.850611／.665842**，ROC 与当前完整模型持平，within 更高，AP 较低。后者见[原评测](/home/jehc223/Retrieval-hate/runs/20260903_hier_evidence_mil/verdict_hmm_only/hatemm/test/metrics.json)。这些是完整 HMM 拟合后改变推断的臂，不能冒充独立拟合的单粒度方法，但足以要求认真比较。

- **30/4 不只是数字任意，还有观测范围近似。** `_block_map` 用细窗左端归属粗块，第一粗块的 OR 实际覆盖前 **8/30** 段，而 VLM 粗观测覆盖 **1/4** 视频。这需要解释；同一 VLM 的相关错误、四帧对整窗的代表性，以及正视频 EM 仍允许全零路径的问题也未解决。

- **三标量不算很多，统计角色却不清楚。** `w_fine` 是按权重 1 拟合 HMM 后再修改推断似然；α 又调同一证据的输出强度，λ_block 再调其监督强度。需要敏感性和简化证据，不能用"只有三个"替代解释。将当前融合称为严格贝叶斯融合仍不成立。

**4．按可信度或新颖性收益／成本排序的四个改进方向**

以下是建议，本次均未执行。

**① 先验证收益是否需要局部内容聚合。无需新搜索，优先级最高。**

- **机制检验：** 对当前预测做视频均值替代、视频内排序分析和按视频配对 bootstrap；固定模型分别扰动 A/B 的时间对应、移除 C，保留其他路径。
- **一个廉价反证实验：** 在 HateClipSeg 固定一个已有配置，只训练"裁定四格频率＋平均后验→视频标量"，再加原 HMM 先验，保留 bag／块监督。如果追平 full，复杂内容聚合的必要性就受到强反证；落后则仍不足以单独证明注意力机制。
- **与先例的区别：** 不产生新方法，目的是排除普通 bag 上下文和标签模型足以解释结果。
- **风险：** 固定模型扰动有分布偏移；必须与上述训练对照区分解读。无需恢复失败的序列目标。
- **成本：** 新增 VLM **0**；分析与前向通常低于一次训练 trial；附加一个完整固定配置训练，无 20-trial 搜索。

**② 补真正隔离部件的对照，并减少无效搜索量。主要提高可辩护性。**

- **精确改动：** 增加 `β_j=Linear(e_j→1)`、所有头共享的偏置臂，其他输入与初始化尽量对齐；它才能检验逐头差异。四格嵌入则与 `常数＋b_f＋b_c＋b_fb_c` 对照，明确贡献是否只是二元交互项。
- **与先例的区别：** 直接回答相对普通加性注意力先验、多变量线性模型还增加了什么；不能预设新颖性。
- **证明方式：** 两语料三 seed 的配对差值，加表示与训练轨迹解释；不要把多个变化合成一个消融。
- **风险：** 简单臂追平 full，意味着应删除对应主张，而不是继续加结构。
- **成本：** 新增 VLM **0**；两个结构臂共 12 个训练 trial，约一个 20-trial 搜索以内，先不重搜。另可优先固定 λ_cma 上限：六个 full 选中 checkpoint 都尚未触及该上限；`w_fine=1` 若作为正式方法变更，则需要新搜索确认。

**③ 自适应查询值得做，但目标应是减少定位不确定性。需要新机制搜索。**

- **精确机制：** 先查询四个粗块，再在尚未查询的细窗中，根据"预期视频内排序不确定性下降／实际调用耗时"选择下一次查询；融合显式边缘化未观测裁定，绝不能把未查询当阴性。共享骨干提供内容信息，不新增第二个预测模型。
- **与先例的区别：** 相对一般主动特征获取或粗到细采样，贡献必须落在弱监督定位的查询目标及缺失观测融合，而非"先粗后细"本身。
- **证明方式：** 相同调用预算比较均匀、随机、粗块阳性触发、分类熵和定位目标；分别去掉内容条件查询与缺失观测处理，报告 AP／ROC—调用数—GPU 时间曲线。
- **风险：** 只细化粗块阳性会永久漏掉粗判断假阴性；只在骨干"自信"时跳过，也可能保留系统性错误。它可能形成效率贡献，而不提高性能。
- **成本：** 首阶段严格遮蔽并回放现有 34 条缓存，新增 VLM **0**；部署每视频 **4 到 34 次**。每语料 seed 的搜索暂按 **1–3 小时**预算，完整确认加消融约半天起，策略开销需实测。新窗口若不同于缓存窗口，必须重新计入抽取成本。

**④ 用真实区间和时间长度替代索引层次。与③结合更有意义。**

- **精确机制：** 在所有实际查询边界的并集上定义隐状态，以 `exp(QΔt)` 表示转移；每条裁定只对其真实区间产生一次 OR 观测因子，允许非嵌套及缺失区间。先只替换融合，保留已有效的骨干和 top-k MIL。
- **与先例的区别：** 连续时间 HMM、Dugong 式区间弱监督都不是新发明；可争取的贡献是任意查询区间下的统一推断，以及**仅细化计算网格、未增加观测时预测不变**的性质。
- **证明方式：** 时间长度转移对索引转移、真实重叠对左端归属、任意区间对固定网格；再验证查询预算变化时是否稳定。单纯修正边界只是正确性改进。
- **风险：** 接近已尝试的 censored process／interval transport，可能增加潜变量复杂度却无收益；应避免同时替换 MIL 目标和整个骨干。四帧是否代表整窗的问题仍需单独说明。
- **成本：** 复用现有窗口时新增 VLM **0**，部署仍最多 34 次；初估搜索约现有 **1–2 倍**，确认与消融半天至一天，尚无实测速率。

另外，[context-witness 记录](/home/jehc223/Retrieval-hate/archive/experiments/20260906_context_witness/README.md)明确是成本叫停、没有训练结果；[interval transport 记录](/home/jehc223/Retrieval-hate/archive/experiments/20260906_interval_evidence_transport/README.md)已有完整 seed 234 搜索、确认被中止。不能将二者统一写成已经验证的完整性能失败。

**5．模拟审稿：今天原样投稿**

**Summary：** 本文用分层 HMM 处理两粒度冻结 VLM 裁定，将证据加入跨模态注意力的 q/k、key 偏置、视频级偏移和块监督，用于弱监督仇恨视频定位。

**Strengths：** 两语料 pooled 指标提升真实；三 seed 结构消融充分；相对上一版确有可测量的骨干改进；实现和原始结果可核查。

**Weaknesses：** 结构与既有条件注意力、加性偏置和 bag 上下文方法的区别仍有限；"内容始终纯净"的核心解释与实现不符；关键消融未隔离逐头贡献；HateClipSeg AP 几乎由 HMM-only 达到；HateMM pooled 上升伴随视频内排序下降；融合的新颖性、观测假设和调用必要性没有得到充分论证。

**Questions：** 同证据的视频标量模型能达到多少收益？共享的完整证据偏置是否足以替代逐头偏置？保持视频摘要不变、破坏证据时间对应后，模型损失多少性能？任意查询区间与缺失裁定能否在同一融合模型中得到一致处理？

**Confidence：4/5。结论：拒稿。评分：4/10。**
