# 规则 4 复核：候选 3「证据引导的跨模态注意力骨干」（2026-09-04，独立 fable agent）

审的对象：本目录 `README.md` 第 1 节的三个部件 A/B/C，代码 `model.py`（`EvidenceEncoder`、`BiasedMultiHeadAttention`、`EvidenceGuidedCMA`、`EGCA`）与 `train.py`。只按 `RESEARCH_ITERATION_RULES.md` 第 4 条的四种 STOP 情形判定；不按"可能退化""可能有 shortcut""可识别性"判定。检索记录与打开过的页面全部列在英文附录。

## 结论

**放行（GO），6/10。** 四种 STOP 情形均不触发。扣分原因：A、B 两个部件合起来与 Graphormer（NeurIPS 2021）的"结构属性嵌入加到节点特征 + 结构属性做逐头注意力偏置"是同一个模板，只是把"图结构属性"换成"VLM 裁定"；B 只在 key 侧加偏置，数学上等于按 exp(β_h(e_j)) 对注意力权重逐 key 重加权，是 ALiBi/T5/Yang 2018 这一支"加性注意力先验"的直接实例；C 因为头是线性层，等价于每视频一个由裁定均值线性映射出来的 logit 偏移。新的部分是：证据来源（冻结 VLM 两粒度裁定 + HMM 后验）、"两粒度一致格子"作为离散嵌入索引、以及三条入口共用同一个证据编码进入一个弱监督跨模态 MIL 骨干；这些在 hateful video 文献里没有先例，在 WSVAD/WTAL 文献里也没有以同一形式出现。

## 四项检查

| 检查 | 结论 | 依据 |
|---|---|---|
| (1) 来源方法已用于 hateful video detection / localization？ | **否** | 逐篇核对（见下"领域内检查"）：MultiHateLoc、LELA、TANDEM、HateClipSeg 基线、SafeLens、HVGuard、MARS、IARE（SIGIR'26）、CMFusion、MM-HSD、Cross-Modal Transfer（WWW'25）、Lightweight Explainable HVD（ACL Findings'26）。没有一篇把 VLM 逐段裁定做成 token 级编码加到模态流、做注意力偏置、或做视频级证据向量。针对领域的查询 1/2/3/4/20/26/30/31/32/33 无相关命中。 |
| (2) 纯 training/test ensemble？ | **否** | 单一模型、单一 VLM 裁定源（候选 1 已有的输入），无多模型预测/特征聚合。 |
| (3) 纯 calibration / 后处理 / 平滑？ | **否** | A、B 在头之前、与骨干联合训练，改变的是表示与注意力分布；C 虽然等价于视频级线性 logit 偏移，但它是从**输入**（裁定分布）算出的、端到端训练的模型部件，不是对训练后输出做的事后校准。规则 3 禁止的是推理时后处理，这里不是。 |
| (4) 纯工程（只调超参/换特征/加增强/改训练配置）？ | **否** | 新增三组带参数的结构部件（格子嵌入表 + 线性映射、逐头 key 偏置投影、上下文投影），每个部件有对应消融臂与可证伪主张。去掉 EMA、合并 λ_cma 属于训练配置改动，但提案没有把它们作为 novelty 主张。 |

## 三个部件的最近先例与差别

### A. 模态共享的证据编码（格子嵌入 + 后验线性映射，加到两个模态流）

最近先例：
- **Graphormer centrality encoding**（Ying et al., NeurIPS 2021, arXiv 2106.05234）：按节点度查一张可学习嵌入表，加到节点输入特征。与 A 的 `Emb[cell_t]` 形式相同（离散属性 → 嵌入 → 加到 token）。
- **外部 tagger 预测作为输入嵌入**（NLP 里把预测 POS/supertag 的嵌入拼/加到词向量，查询 34 的结果）：把另一模型的离散判定当作 token 级输入特征，是标准做法。
- 本项目候选 1（修订 1）：四列裁定拼进音频流过 `fc_a`。A 与它的差别只有两点：拼接→加性嵌入；只进音频流→两流共用。
- 领域内：MultiHateLoc 无任何外部先验输入；LELA/TANDEM 直接把 VLM 输出当结果。

差别：A 的索引是"两粒度裁定的一致格子"（同一标注者两个时间尺度的组合），这个索引轴没有先例；"同一编码加到两个模态流"是位置编码的用法，用于外部裁定没有先例。但是从"离散侧信息 → 嵌入 → 加到 token"这个操作看，A 是 Graphormer 中心性编码的直接套用。

### B. 证据偏置的共享跨模态注意力（score(i,j) = q_i·k_j/√d + β_h(e_j)）

最近先例：
- **Graphormer spatial/edge encoding**：结构量 → 逐头标量偏置加到注意力 logit。B 的 `Linear(hid→nhead)` 零初始化偏置与它形式一致，差别是 B 只依赖 key（e_j），不依赖 (i,j) 对。
- **ALiBi / T5 相对位置偏置 / DAPE**（查询 5）：加性 logit 偏置，但都是位置量。
- **Yang et al., "Modeling Localness for Self-Attention Networks"**（EMNLP 2018）：可学习高斯偏置加到注意力分布——"注意力 + 先验"的标准形式；Lin et al. Transformer 综述（查询 6）把这类做法归为 "attention with prior"，先验分布与内容分数在 softmax 前加权求和。
- **SISA**（Shin & Yang, 2026-06, arXiv 2606.02332）：把 SSM 算出的重要性项直接加进注意力分数；**GOAT**（Litman & Guo, ICML 2026, arXiv 2601.15380）：把可训练先验放进注意力核心计算。两者都是"内部信号"，不是外部裁定。
- **Efficient Attention via Pre-Scoring**（Li et al., 2025-05）：给 key 一个与 query 无关的全局重要性先验——与 B "只看 key" 的结构最接近，但用于选 key 子集，不加进 logit。
- WSVAD/WTAL 里外部 VLM 信号进注意力的先例：**MLLM4WTAL**（Zhang et al., arXiv 2411.08466，2025）用 MLLM 文本先验与视频特征内积得到相似度矩阵调制注意力（KSM）并做蒸馏；**Snippet Anomalous Attention**（Fan et al., 2023）的注意力是单独的 T×1 权重乘到分数上、由自身分数监督，不是 logit 偏置，不用外部先验；VadCLIP / TPWNG / PEL4VAD 的文本提示进的是分类器或伪标签，不是注意力偏置；LAVAD 是 training-free 的分数细化。
- 视频时刻检索里 **SA-DETR**（COLING 2025）用 span anchor 生成高斯 mask 调制 cross-attention，**BA-SAM**（CVPR 2024）用 bias-mode mask——都是位置型偏置。
- 领域内：MultiHateLoc 的 CMA 是标准 softmax(QKᵀ)V，无偏置；DMS 是模态级 sigmoid 门。

差别：B 的偏置由**外部冻结 VLM 裁定**（经 A 的编码）算出，只依赖 key 秒，两个跨模态方向共用；没有先例把"另一模型对该时刻的判定"做成跨模态注意力的 key 偏置。必须如实说明：key-only 加性偏置 = 对每个 key 的注意力权重乘 exp(β_h(e_j)) 再归一化，即"证据引导"是逐 key 重加权，不是 query 依赖的引导。

### C. 视频级证据上下文（c = Linear(mean_t e_t)，加到两流每一行再进头）

最近先例：
- **Global Information Guided VAD**（Lv, Xu, Cui, 2021, arXiv 2104.06813）：用视频级弱标签监督一个全局模式向量（GPC），再用它调制帧级特征。
- **PEL4VAD 的 TCA**（Pu et al., TIP 2024）、**UR-DMU 的 GL-MHSA**（AAAI 2023）：全局-局部时间聚合，全局量由内容算出。
- MIL 里的 bag 级上下文向量广播回实例（查询 35 的 "bag encoding strategies" 等）：把 bag 摘要拼/加到实例特征是已知策略。
- 领域内：MultiHateLoc、TANDEM 明确无视频级向量（TANDEM 上下文限于 30 s 块）。

差别：C 的输入只有裁定编码的均值（不含内容），与本骨干 9.5 的"视频级密度"机制分析对应。结构上要注意：头 `fc` 是线性层，`fc(a_out + c) + fc(v_out + c) = fc(a_out) + fc(v_out) + 2·W_fc·c + const`，所以 C 恰好是每视频一个由 mean(e_t) 线性映射出的 logit 偏移，不产生与内容的交互。它不是后处理（输入侧、联合训练），但论文不能把它写成"上下文建模"。

## 领域内检查（是否已用于 hateful video）

| 论文 | 核对到的结构 | 有无 A/B/C 中任何一个 |
|---|---|---|
| MultiHateLoc（Sun et al., WWW 2026, arXiv 2512.10408v3，全文 HTML） | 逐模态 Transformer 编码器 + DMS 逐帧模态 sigmoid 门 + 标准 CMA + top-k 模态感知 MIL + 同时间戳跨模态对比；无外部先验、无视频级向量 | 无 |
| LELA（arXiv 2602.09637，全文 HTML） | training-free，逐帧多阶段提示打分，跨模态取 max；无训练、无注意力、无视频级聚合 | 无 |
| TANDEM（Koushik et al., arXiv 2601.11178v3，全文 HTML） | VLM/ALM 交替 RL，30 s 块内输出 XML 含时间戳；无注意力偏置、无视频级向量 | 无 |
| HateClipSeg（arXiv 2508.01712） | 定位基线 ActionFormer / LSTR | 无 |
| SafeLens（AAAI 2026；PDF 未能解析，依据检索摘要与候选 2 预检记录） | 段级流水线：Whisper 转写、OCR、视觉线索、结构化输出 | 无 |
| HVGuard（EMNLP 2025；PDF 未能解析，依据摘要与候选 2 预检记录） | MLLM CoT + MoE 融合，视频级分类 | 无 |
| MARS（Yang, Zhang, Fu, ICASSP 2026, arXiv 2601.15115） | training-free 三阶段对抗推理，视频级 | 无 |
| IARE / Ex-HateMM（Lu et al., SIGIR 2026, arXiv 2606.11953） | 多模态 CoT + DPO 的可解释分类 | 无 |
| Cross-Modal Transfer from Memes（WWW 2025）、CMFusion、MM-HSD、Lightweight Explainable HVD（ACL Findings 2026） | 视频级分类 / 融合 / 蒸馏 | 无 |

## 必须执行的修改（REQUIRED）

1. **引用并对照 Graphormer**（Ying et al., NeurIPS 2021）作为 A+B 的结构模板；论文对 novelty 的表述必须落在"证据来源 + 两粒度一致格子索引 + 同一证据编码经三条入口进入弱监督跨模态 MIL 骨干"，不能落在"注意力偏置"或"嵌入加到 token"本身。
2. **B 的表述**：正文必须写明 key-only 加性偏置等价于逐 key 注意力重加权 exp(β_h(e_j))；相关工作必须引 ALiBi、T5 相对位置偏置、Yang et al. EMNLP 2018（localness 高斯偏置）、Lin et al. 综述的 "attention with prior"，以及 MLLM4WTAL（arXiv 2411.08466，MLLM 先验调制注意力的最近 WTAL 先例）。
3. **预注册补一个臂 `scalar_bias`**：β_h(e_j) 换成单标量 γ·ℓ_j/L（所有头共用，γ 可学习）。没有这个臂，B 的主张与"固定的 ALiBi 式先验偏置"区分不开；B 只有在 `full` 三 seed 两语料 pooled 均高于 `scalar_bias` 时才能作为"学习到的证据偏置"主张，否则只能报告为一个标量先验。
4. **C 的表述**：正文必须写出上面的线性等价式（C = 每视频一个由 mean(e_t) 线性映射出的 logit 偏移），命名为"视频级证据偏移"而非"上下文向量/上下文建模"；相关工作引 Lv et al. 2021（GIG-VAD）与 PEL4VAD TCA。若想保留"上下文"说法，注入点必须移到 CMA 层之前（与内容产生非线性交互），并作为预注册的设计改动写明。
5. **相关工作必须包含"VLM 逐段分数进 WSVAD/WTAL 的三种方式"的对照**：作伪标签损失（TPWNG、TFPLG、LAVAD 系）、作注意力调制（MLLM4WTAL）、作输入编码（本候选）；并写明本候选与候选 1 的区别只在 A（加性、两流共用）、B、C 三处，训练目标不变。
6. **主表对照行不变**：仍须报候选 1 骨干（`avce` 臂，同训练同超参）、HMM 后验单独、MultiHateLoc（同评测器复现）。

以上第 3 条是预注册必须新增的臂；第 1/2/4/5 条是论文写法与引用要求。其余实现细节（padding 屏蔽、去 EMA、合并 λ）不构成 novelty，也不需要额外过审。

## 评分理由（6/10）

- +：四类 STOP 均不触发；三个部件各有消融臂与可证伪预期；设计由候选 1 的机制分析（9.2/9.5/9.7）直接导出；hateful video 文献无同类结构。
- −：A+B 是 Graphormer 模板换证据源；B 是 key 侧加性先验，与位置偏置族只差输入；C 在线性头下退化为视频级 logit 偏移；相对候选 1 的改动量小（参数 +5%），提升上限取决于 `avce` 对照臂的结果。

---

# English appendix: search log

All queries were executed 2026-09-04 via the WebSearch tool, verbatim, in this order.

1. `hateful video localization weakly supervised 2025 2026 arXiv`
2. `HateClipSeg segment-level hateful video localization benchmark`
3. `MultiHateLoc hateful video temporal localization`
4. `TANDEM hateful video detection LELA`
5. `attention bias from side information additive attention logits ALiBi non-positional cue`
6. `prior-guided attention bias saliency prior added to attention scores transformer`
7. `VadCLIP text prompt guided attention weakly supervised video anomaly detection`
8. `LAVAD training-free video anomaly detection large language model temporal scores`
9. `TPWNG text prompt normality guidance video anomaly detection CVPR 2024`
10. `PEL4VAD prompt-enhanced learning weakly supervised video anomaly detection`
11. `Graphormer centrality encoding added to node features spatial encoding attention bias`
12. `label embedding added to input token features noisy pseudo-label embedding transformer weak supervision`
13. `video-level context vector added to snippet features weakly supervised temporal action localization global context`
14. `global-local multi-head self-attention video-level feature weakly supervised video anomaly detection MSL RTFM UR-DMU`
15. `confidence-guided attention pseudo-label guided attention weakly supervised temporal action localization`
16. `cross-modal attention additive bias external prior audio-visual event localization`
17. `LLM anomaly score guided attention weakly supervised video anomaly detection VLM pseudo label attention`
18. `"evidence-guided attention" OR "evidence-aware attention" video`
19. `learned key-dependent attention bias from auxiliary features transformer "attention bias" side information`
20. `HateMM hateful video detection cross-modal attention VLM verdict guided`
21. `MACIL-SD modality-aware contrastive instance learning self-distillation audio-visual violence detection AVCE`
22. `knowledge-guided attention mask external knowledge bias attention logits vision language`
23. `Modeling Localness for Self-Attention Networks Gaussian bias attention scores Yang 2018`
24. `MLLM frame-level scores as input features weakly supervised temporal localization "input" prior feature concatenation video anomaly`
25. `video-level feature broadcast added to frame features global context anomaly detection "video-level" representation injected snippet`
26. `hateful video 2026 "attention" prior VLM guided segment localization HateMM MultiHateClip new method`
27. `text-guided attention bias cross-attention "attention prior" audio-visual video anomaly detection multimodal`
28. `off-the-shelf detector score soft mask additive attention bias transformer video moment retrieval saliency guided attention`
29. `"attention bias" "anomaly score" OR "pseudo label" keys transformer weakly supervised video`
30. `Cross-Modal Transfer from Memes to Videos hateful video detection architecture attention`
31. `SafeLens segment-level hate speech detection online videos arXiv architecture`
32. `HVGuard multimodal large language models hateful video detection mixture of experts arXiv`
33. `hateful video detection 2025 2026 LLM rationale reasoning feature fusion attention HateMM MultiHateClip ImpliHateVid`
34. `external tagger predictions as input features embedding added to word embeddings sequence labeling "stacked" predictions input feature transformer`
35. `bag-level context vector broadcast to instance features multiple instance learning global context instance classifier`
36. `pseudo-label conditioned transformer "label embedding" as positional encoding added to tokens video temporal localization prior score embedding`

## Pages actually opened (WebFetch)

| URL | Paper | Result |
|---|---|---|
| https://arxiv.org/html/2512.10408v3 | MultiHateLoc (WWW 2026) | full HTML read; architecture confirmed (no prior, no bias, no video-level vector) |
| https://arxiv.org/html/2601.11178v3 | TANDEM (ICWSM 2027 / arXiv 2601.11178) | full HTML read |
| https://arxiv.org/html/2602.09637 | LELA (arXiv 2602.09637) | full HTML read |
| https://arxiv.org/abs/2601.15115 | MARS (ICASSP 2026) | abstract |
| https://arxiv.org/abs/2606.11953v1 | IARE / Ex-HateMM (SIGIR 2026) | abstract |
| https://ojs.aaai.org/index.php/AAAI/article/download/42390/46351 | SafeLens (AAAI 2026) | PDF fetched but not parsed; architecture taken from search snippets and the candidate-2 precheck |
| https://aclanthology.org/2025.emnlp-main.456.pdf | HVGuard (EMNLP 2025) | PDF fetched but not parsed; taken from abstract snippets and the candidate-2 precheck |
| https://arxiv.org/abs/2106.05234 | Graphormer (NeurIPS 2021) | abstract; centrality encoding + attention-bias encodings confirmed |
| https://arxiv.org/abs/2606.02332 | SISA: SSM-Informed Softmax Attention (2026-06) | abstract |
| https://arxiv.org/abs/2601.15380 | GOAT: You Need Better Attention Priors (ICML 2026) | abstract |
| https://arxiv.org/abs/2505.11040 | Efficient Attention via Pre-Scoring (2025/2026) | abstract |
| https://arxiv.org/abs/2402.16790 | SyntaGuid (2024) | abstract |
| https://arxiv.org/abs/2209.03745 (and /pdf/, not parsed) | SPAN: Prior Knowledge-Guided Attention in SSL ViT (2022) | abstract; prior enters as attention-mask regularization |
| https://arxiv.org/abs/2602.10549 | Text-guided multimodal WSVAD (TMM 2026) | abstract; bottleneck-token fusion, no attention bias |
| https://arxiv.org/abs/2309.16309 and https://arxiv.org/html/2309.16309 | Snippet Anomalous Attention (2023) | full HTML read; T×1 attention multiplied into scores, self-supervised, no external prior |
| https://arxiv.org/html/2411.08466v2 | MLLM4WTAL: Dual-Prior Collaborative Learning Guided by MLLMs (2025) | full HTML read; MLLM priors modulate attention via similarity matrix + distillation |
| https://arxiv.org/abs/2104.06813 | Global Information Guided VAD (Lv et al., 2021) | abstract |

Not opened but identified from search snippets and cited above: ALiBi (Press et al.), T5 relative bias, DAPE, Yang et al. EMNLP 2018 (arXiv 1810.10182), Lin et al. Transformer survey (arXiv 2106.04554), PEL4VAD (TIP 2024, arXiv 2306.14451), UR-DMU (AAAI 2023, arXiv 2302.05160), VadCLIP (arXiv 2308.11681), TPWNG (CVPR 2024), LAVAD (CVPR 2024, arXiv 2404.01014), SA-DETR (COLING 2025), BA-SAM (CVPR 2024), HateClipSeg (arXiv 2508.01712), Cross-Modal Transfer from Memes to Videos (WWW 2025, arXiv 2501.15438), MM-HSD (MM 2025, arXiv 2508.20546), CMFusion (arXiv 2505.12051), Lightweight Explainable HVD (ACL Findings 2026), MACIL-SD (MM 2022, arXiv 2207.05500).
