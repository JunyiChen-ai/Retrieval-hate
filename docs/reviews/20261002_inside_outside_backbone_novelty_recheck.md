# 初版 inside–outside backbone：独立 novelty 复查

日期：2026-10-02。审稿人：独立 Codex agent。范围为 `experiments/20261001_inside_outside_backbone/` 初版及其自身结果；按研究规则第9条做完成后的独立复查，不重开 proposal/code review，不新增门或训练。依据当前 `backbone.py`、共享 `src/qtl/{content,train}.py`、原 proposal/code review、实验 README、STATUS，以及下列本机原始统一评测输出。within 按最新 AGENTS.md 与 AP、ROC 并列为主指标。

**结论：现有证据支持保留一项范围有限的迁移贡献：把 inside–outside 内容编码放到 hateful video localization 的固定时间问题树上，用区间外上下文形成秒级和节点级内容先验，并纳入既有问题答案似然训练。初版自己的 outside 消融满足两语料平均 AP 贡献要求；完整三 seed 搜索同时显示，相比 r5 的主要性能代价仍在。不能把这个结论写成新树算法、相对 r5 涨点、世界首次或稳定普适的优势。**

本次没有发现阻止上述窄主张成立的硬阻塞。它不等于整个自动迭代目标已完成：残差版尚在运行的既定完整搜索仍应跑完，再由主 agent 综合性能优先目标收尾；本复查不停止任何 monitor，不改 Goal 状态。

## 1. 实现支持的主张边界

初版直接投影已有逐秒视觉和音频文本缓存，自下而上合并子区间 inside；自上而下以父 outside 和兄弟 inside 生成当前区间 outside。`ContentLayouts` 使用标签推断同一 `ctree.Forest` 拓扑。`ContextFusion` 读取同模态 inside/outside、差、乘积与另一模态的 inside/outside；叶子产生秒级先验，内部节点产生势，根 inside 产生视频头。共享 trainer 的树边际似然与 CMAL 对该表示反向传播。骨干 forward 不接收 VLM 答案，答案仍经原答案模型参与后验；不能称为“答案直接调制 backbone”或本轮新发明了答案似然。

可区分于现有 r5 的是内容编码计算图及其与问题树的结构对应，而非单纯更名节点池化。原 code review 已验证 outside 的目标区间缓存行梯度为零及新参数进入训练/推断，本次静态复核与其一致。这里的排除严格限定在**缓存行索引**：I3D 感受野、上游文本编码可能跨区间，不能扩大为原始媒体隔离、统计独立或无任何信息重叠；outside 也不是已知无害背景。

`OrderedCompose` 是有序门控 MLP 与子向量均值的混合，没有 Tree-LSTM 的记忆单元。它受既有树组合思想启发，既不宜称为完整 Tree-LSTM 实现，也没有证据可单列“新门控组合器”贡献。答案树结构与训练目标是已有框架；本轮增加的是在该结构上学习内容先验的具体适配，尚无独立实验证明两者存在额外协同增益。

## 2. 本次实际联网检索与原论文核对

检索日期为2026-10-02。执行了以下查询，不以未返回相关结果证明不存在：

- `"inside-outside" "hateful" video`；`"hateful video" "inside–outside"`；`"HateClipSeg" "inside-outside"`；`"HateMM" "DIORA"`。
- `"hateful video" "Tree-LSTM"`；`"Tree-LSTM" "hateful video detection"`；`"hateful video" "recursive neural"`；`"hateful video" "binary tree"`。
- `"hateful video" localization 2026 tree backbone`；`"hateful video" "tree" "2026" neural`。
- `"inside-outside" "hate" neural video site:arxiv.org`；`"inside-outside" "hateful" site:aclanthology.org`；`"hateful video" "recursive" site:arxiv.org`；`"hateful video" "tree" site:openaccess.thecvf.com`。

本次重新打开并核对的 primary sources 如下；方法差异是对论文所述计算结构与本地实现的比较判断。没有直接采用其报告的性能数字作本项目对照。

| 原论文及核对位置 | 已有工作与本候选的关系 |
|---|---|
| [IORNN，Le & Zuidema，EMNLP 2014，§4.2](https://aclanthology.org/D14-1081.pdf) | 子节点构成 inside、父 outside 加兄弟 inside 构成 outside、可学习根 outside 均已有明确先例。这是主要迁移来源，不是本候选发明。 |
| [DIORA，NAACL 2019，§2](https://aclanthology.org/N19-1116.pdf) | 对候选句法树做 chart 内外计算，并以 outside 重构叶子；本候选采用固定时间树和已有视频标签/问题答案似然，没有潜在句法树诱导或词重构。 |
| [Tree-LSTM，ACL 2015，§3.2](https://aclanthology.org/P15-1150.pdf) | 有序子节点的门控树组合已存在，不能主张首次有序组合或首次树形门控。 |
| [RAMF，作者论文 v2，§3.5–3.6](https://arxiv.org/html/2512.02743v2) | 局部分支为 Conv1D/最大池化，全局分支为整序列均值，再门控融合及语义跨注意力。其全局量并非每个区间的排除式互补 outside；“局部与全局结合”已非空白。 |
| [MultiHateGNN，作者论文 v1，§3.3–3.5](https://arxiv.org/html/2509.13515v1) | 全局权重图配合局部实例子图并加权分类；已有结构化 hateful-video 内容建模，未使用所述固定时间树的父 outside/兄弟 inside 递归。 |
| [CLARA，作者论文 v1，§3](https://arxiv.org/html/2608.15905v1) | utterance-aligned clips、MoE、局部/全局片段对比和 rationale-gated Transformer。§3.3 两段独立选起点，不要求包含，也不要求互斥；不是按每个目标区间排除其缓存行的 outside 递归。不能主张首次细粒度上下文或多尺度关系建模。 |
| [MultiHateLoc，作者论文 v3，§3.2–3.5](https://arxiv.org/html/2512.10408v3) | 模态专属 Transformer、动态跨模态融合及 top-K MIL；没有本候选的内外内容递归。弱监督帧定位及跨模态融合本身已有先例。 |
| [TANDEM，作者论文 v1，§3](https://arxiv.org/html/2601.11178v1) | 30秒块、视觉/音频模型的结构化输出、交替上下文条件化与 RL；不是单一内容 prior network 的区间内外表示。 |
| [LELA，作者论文 v1，§3](https://arxiv.org/html/2602.09637v1) | 训练外的多模态文本化、多阶段提示、逐模态打分和逐帧最大值组合；没有所述可训练 inside–outside backbone。此次扩检新增核对。 |
| [IARE，作者论文 v1，§4](https://arxiv.org/html/2606.11953v1) | 信息扩充后以 SFT/DPO 增强解释推理；不是时间区间互补内容表示或问题树边际训练。此次扩检新增核对。 |
| [WINNER，CVPR 2023，§3](https://openaccess.thecvf.com/content/CVPR2023/papers/Li_WINNER_Weakly-Supervised_hIerarchical_decompositioN_and_aligNment_for_Spatio-tEmporal_Video_gRounding_CVPR_2023_paper.pdf) | 视频 grounding 已用语言分解树、自下而上多模态组合及自上而下回传；因此不能声称首次把树组合/双向传播用于视频。该论文引用 DIORA 不等于采用其完整 encoder，也不是 hateful-video 直接先例。Web open 失败后从同一 CVF 原始 PDF 读取并在内存中用 pdftotext 核对，未落地额外文件。 |

限定检索还返回了 [JL-Hate 的 IO 标签规范](https://aclanthology.org/2024.lrec-main.834.pdf)，它是文本 span 的 Inside/Outside 标注，不是神经 inside–outside 内容递归，不作为直接先例。

**截至本次实际查询及上述原论文核对，未找到 hateful-video detection/localization 已使用这一具体内外递归的直接先例。** 可以按项目规则保留迁移 novelty，不能升级为排他性的全球首创证明。已有局部/全局和结构化方法必须在 related work 中正面比较。

## 3. 从本机原始输出抽核的证据

本次直接解析统一评测文件的 `results.score_av.{pr_auc,roc_auc,per_video.macro_auc}`，重新计算下表均值、总体标准差（ddof=0）及逐 seed 配对差。未重做已通过的120个 trial 训练审计，未复制评测器或重新计算预测指标。搜索完成/每 trial 50 epoch/validation checkpoint 的既有全量核验见 `analysis/{hatemm,hateclipseg}_three_seed_search.json`；本次核对其关键结论与原始评测一致。

### 3.1 完整三 seed Optuna 性能

固定8次、test、1fps；每 seed 独立20 trial、各50 epoch。trial 内 validation 选 checkpoint；开发期由 test(AP+ROC)/2 排序 trial。下表均为均值±总体 seed 标准差，AP / ROC / within 同列报告。

| 选择方式 | HateMM | HateClipSeg |
|---|---|---|
| 初版，test 选 trial | .676491±.010167 / .879176±.001587 / .737845±.014984 | .668398±.003274 / .642595±.005721 / .560326±.004923 |
| 初版相对 r5 的均值差 | −.014380 / −.005681 / −.035070 | −.018057 / −.031130 / +.000100 |
| 初版，仅 validation 选 trial | .640230±.018754 / .868892±.001408 / .681005±.060963 | .652083±.002717 / .620540±.009350 / .561490±.012623 |

本次抽核的原始来源为 `runs/20261001_inside_outside_backbone/<corpus>/seed<seed>/trial<k>/metrics_test_fixed8.json`，按 seed234/2025/3407：HateMM test选17/11/11、validation选10/2/9；HCS test选11/2/10、validation选6/17/1。r5来源为 `runs/20260929_query_paradigm_r5/<corpus>/seed<seed>/trial<k>/metrics_test_fixed8.json`，HateMM9/4/5、HCS13/11/19。选择记录在对应 `study_summary.json`。

两主数据固定8次 pooled 均值低于 r5；HateMM within 也下降，HCS within 的+.000100不构成收益。通过既定历史 baseline 门不等于超过当前 r5，更不能直接写成对最新文献的全面 SOTA。已有0/32次汇总也没有提供替换主结论的依据：0次相对r5两语料三项均下降；32次HateMM pooled 均值小幅升但 within 降，HCS三项均降。0/32来源与逐项数字保留在 `analysis/two_corpus_three_seed_search.json` 和其单语料审计，不在这里另立操作点。

这些是已用 test 排序并影响设计的**开发期证据**；多 seed 完整搜索不使它变成未揭盲验证。仅 validation 选 trial 数字也来自同一受 test 目标驱动的候选搜索，不能称为完全独立的 validation-only 调参试验。

### 3.2 初版自身 outside 消融

HateMM锁定初版seed234 trial17，HCS锁定初版seed234 trial11；各 full/nooutside 三 seed 完整50 epoch，validation选checkpoint。本次逐组比较6对原始 `config.json`，差异均**仅 `io_outside: true → false`**。full seed234复用各自源trial，2025/3407为固定同一超参的训练；不是每个seed重新Optuna的最优配置。

下表为去outside减full，固定8次，原始评测重新配对计算：

| 语料/seed | ΔAP | ΔROC | Δwithin |
|---|---:|---:|---:|
| HateMM / 234 | −.058272 | −.016040 | −.055956 |
| HateMM / 2025 | −.004597 | −.006297 | +.029252 |
| HateMM / 3407 | −.011432 | +.003986 | −.002636 |
| HateMM / 均值±配对标准差 | −.024767±.023856 | −.006117±.008177 | −.009780±.035151 |
| HCS / 234 | −.050199 | −.030968 | −.023017 |
| HCS / 2025 | +.007295 | −.010908 | +.001976 |
| HCS / 3407 | +.006568 | +.013911 | −.028830 |
| HCS / 均值±配对标准差 | −.012112±.026933 | −.009322±.018356 | −.016624±.013365 |

原始来源：`runs/20261001_inside_outside_backbone/diagnostics/<corpus>/{full,nooutside}/seed<seed>/metrics_test_fixed8.json`；full seed234例外分别为 `hatemm/seed234/trial17/metrics_test_fixed8.json`、`hateclipseg/seed234/trial11/metrics_test_fixed8.json`。对照配置在各评测同目录 `config.json`。与 `analysis/two_corpus_outside_diagnosis.json` 一致。

**两语料移除outside后平均AP下降均≥.01，满足现行14(g)的均值要求。** HateMM三个seed的AP均下降，HCS平均效应由seed234主导而另外两个seed反向；方差较大，不主张统计显著、逐seed稳定或所有指标普遍改善。within的均值贡献可以报告，方向不完全一致也必须报告。这些固定配置诊断不能冒充Optuna确认，更不能借用加性/残差修订版的消融来增强初版证据。

消融支持的是“此已实现分支在这些配置中的平均贡献”。关闭outside也停用了该分支的有效计算，因此它尚不能把收益进一步唯一归因于排除式设计相对同容量普通全局上下文的优势；无需为当前窄主张引入新的匹配对照门。

另从HCS原始 `diagnostics/hateclipseg/mean/seed<seed>/metrics_test_fixed8.json` 与相应full抽核，mean−full为 **+.001006/+.002575/+.002760**。它只替换inside组合，outside仍保留原有组合器。因此没有证据证明学习到的inside门控优于递归均值，更没有两语料的独立gated-composition贡献；不能把该消融说成已经证明“所有树非线性都不必要”。

## 4. 可直接使用的窄主张与尚缺证据

论文方法主张建议：

> 我们将 inside–outside 表示迁移到弱监督 hateful video localization 的时间问题树上：在同一固定二分拓扑中，区间内部内容与排除该区间缓存行的外部上下文共同生成秒级先验和节点势，并由视频标签及既有带噪问题答案的边际似然训练。该结构不向内容编码器直接输入问题答案，也不新增 VLM 观测。固定8次查询下，移除outside使HateMM和HateClipSeg的三seed平均AP分别下降2.48和1.21个百分点，支持该分支在锁定配置下的贡献；HateClipSeg效应存在明显seed差异。完整开发期搜索仍低于r5的pooled性能，不主张整体性能提升。

方法段应引用IORNN/DIORA/Tree-LSTM并明确是任务适配；related work应区分RAMF、MultiHateGNN、CLARA等已有上下文建模。实验段需同时给出上面的主指标代价、方差及test选trial说明，不能仅放outside消融的正面结论。

对这个窄主张，**无需增加训练或新流程门**。不支持的更强主张及原因是：

- 新的通用树算法、首次结构化视频表征或世界首次：已有明确跨任务/视频先例，当前检索不提供排他证明。
- 门控组合更强、排除式outside优于所有同容量上下文编码器、树与答案似然有独立协同增益：当前对照不支持这些归因。原README提及但未完成的同容量逐秒对照不被写成已完成；收窄主张即可，不自动启动补实验。
- 相对r5涨点、普遍改善within、未揭盲泛化优势：原始性能及开发期选择协议不支持。保留有限有效贡献不等于兑现优先涨点目标。
- 更快或更省端到端推理：只确定新增特征抽取与新增VLM调用均为0，仍需原新视频输入处理和查询成本。首trial耗时及树的线性节点数不能代替受控吞吐/端到端成本测量；没有该测量就不做速度主张。

冻结范围：本次仅新增本记录，未修改研究规则、实现、评测器或其他文档；未启动训练/smoke，未计算内容哈希，未将后续修订版的结果混作初版证据。
