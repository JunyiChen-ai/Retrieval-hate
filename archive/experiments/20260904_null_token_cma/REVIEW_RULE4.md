# 规则 4 复核：候选 4「空 token 跨模态注意力骨干」（2026-09-04，独立 fable agent）

审的对象：本目录 `README.md` 第 0–4 节，代码 `model.py`（`MultiHeadAttention.key_mask`、`NullTokenCMA.null_token/one`、`NTCA.forward` 里的 c 计算）与 `train.py`。只按 `RESEARCH_ITERATION_RULES.md` 第 4 条的四种 STOP 情形判定；不按"可能退化""可能有 shortcut""可识别性"判定。检索记录（43 条查询原文）与打开过的页面全部列在英文附录。候选 3 的 `REVIEW_RULE4.md` 已逐篇核对过的 hateful video 论文（MultiHateLoc、LELA、TANDEM、HateClipSeg、HVGuard、SafeLens、MARS、IARE/Ex-HateMM、CMFusion、MM-HSD、meme 迁移、Lightweight Explainable HVD）本次只针对 null / sink / register / padding 重查，不重复整体核对。

## 结论

**放行（GO），5/10。** 四种 STOP 情形均不触发。扣分原因：模块的两个组成部分各自都有已发表的、形式相同的先例——(i) "在跨注意力的 key/value 序列前拼一个可学习的空 key/value"就是 Mask-Align 的 leaky attention（Chen, Sun, Liu, ACL 2021：k_NULL、v_NULL 拼到 encoder 输出的 K、V 之前，用来吸收本该落到句号等"垃圾收集"token 上的注意力），同一形式还独立出现在 Sun et al. 2024（Massive Activations 的 explicit attention bias k′、v′）、Sukhbaatar et al. 2019（persistent memory，n=1 的特例）和 PyTorch `nn.MultiheadAttention(add_bias_kv=True)`；(ii) "token = 可学习基向量 + 线性映射(上下文摘要)"是 CoCoOp（Zhou et al., CVPR 2022）的 v_m + h_θ(x) 形式。本候选的 `const_token` 臂与 (i) 完全相同；`full` 臂 = (i) + (ii)。新的部分是：把这个 token 放进弱监督跨模态 MIL 骨干（MACIL-SD AVCE）、用冻结 VLM 裁定的视频级均值作条件、并由一个实测的诊断（MACIL-SD 不屏蔽 padding，训练时 .254 注意力落在 padding 上、测试时没有 padding）导出设计。这个诊断在 WSVAD / 音视频暴力检测 / hateful video 文献里没有先例（检索 12、24、34、38：唯一相近的是 Himelstein et al. 2025 "Silent Tokens, Loud Effects"，讨论 LLM 推理期 padding 未屏蔽的影响，不涉及训练/测试不一致，也不涉及弱监督视频）。

对"证据条件化的空 token 是否只是部件拼接"的明确判断：**从结构上看是两个已知部件的拼接**（leaky-attention 式空 KV + CoCoOp 式条件化）；**从科研陈述上看不只是拼接**，因为它有一个可独立验证的观察（padding 充当偶然 sink 且训练/测试不一致，屏蔽后变差）和一个由此导出的、可证伪的修正（显式 token 应优于偶然 sink 与屏蔽两者）。规则 4 不以"部件拼接"为 STOP 理由；论文的 novelty 表述必须落在诊断 + 条件来源 + 在弱监督跨模态 MIL 里的验证，不能落在"空 token"或"条件化 token"本身。

## 四项检查

| 检查 | 结论 | 依据 |
|---|---|---|
| (1) 来源方法已用于 hateful video detection / localization？ | **否** | 针对领域的查询 19、20、36 无任何 null / sink / register / padding-mask 命中；MultiHateLoc 全文（arXiv 2512.10408v3 HTML）重读：无 padding 屏蔽描述、无额外可学习 token、无视频级摘要 token，跨模态注意力是三模态拼接后的标准 softmax(QKᵀ)V。LELA、TANDEM、MARS 为 training-free 或 RL 方案，无注意力层可谈；HVGuard / SafeLens / IARE 为 MLLM 推理 + 融合的视频级分类；HateClipSeg 基线是 ActionFormer / LSTR。MACIL-SD（本候选的骨干来源）本身也没有空 token；候选 1 的 AVCE 照搬其"不屏蔽 padding"实现。 |
| (2) 纯 training/test ensemble？ | **否** | 单一模型、单一 VLM 裁定源（候选 1 已有输入），无多模型聚合。 |
| (3) 纯 calibration / 后处理 / 平滑？ | **否** | 空 token 在注意力层内、与骨干联合训练，改变每行的聚合分布与表示；推理流程无后处理。 |
| (4) 纯工程（只调超参 / 换特征 / 加增强 / 改训练配置）？ | **否** | 新增带参数的结构部件（b_m、W_m，1,280 参数）并附五个结构臂；但要如实指出，"把 padding 屏蔽掉"本身是工程修正，不得计入 novelty；EMA 删除是训练配置改动，提案也未把它作为主张。 |

## 最近先例与差别

### 1. 空 key/value token（对应 `const_token` 臂，也是 `full` 的骨架）

| 先例 | 形式 | 与本候选的关系 |
|---|---|---|
| **Mask-Align leaky attention**（Chen, Sun, Liu, ACL 2021, arXiv 2012.07162；ar5iv 全文读）| K = Concat(k_NULL, H_enc W_K)，V = Concat(v_NULL, H_enc W_V)；k_NULL、v_NULL 为可学习向量，小方差初始化；目的是给跨注意力一个 NULL 位置吸收"垃圾收集"式的偏置注意力 | **最近的结构先例**：同是跨注意力、同是拼在 key/value 序列前、同是可学习常量。本候选 `const_token` 臂 = 此形式（差别只在 Mask-Align 直接参数化投影后的 k/v，本候选参数化投影前的 token 再过 lin_k/lin_v，数学上等价于受限参数化）。 |
| **Massive Activations 的 explicit attention bias**（Sun et al., 2024, arXiv 2402.17762 §4.3；HTML 读）| Attention(Q,K,V;k′,v′) = softmax(Q[Kᵀ k′]/√d)[V; v′ᵀ]，每头一对可学习常量；训练后大部分注意力落在 k′/v′ 上，massive activation 消失 | 同一形式（自注意力）；给出"模型本来就需要一个常量 bias 位置"的证据，与本候选"padding 充当偶然 sink"的诊断同向。 |
| **Persistent memory**（Sukhbaatar et al., 2019, arXiv 1907.01470）| 可学习 KV 向量集拼进自注意力 | n=1 即空 token。 |
| **PyTorch `MultiheadAttention(add_bias_kv=True)` / `add_zero_attn=True`** | bias_k、bias_v 沿序列维拼接；或拼一行零 | 现成实现；`add_zero_attn` 恰是"零向量经投影后的常量 key"，即 MACIL-SD padding 行的行为。 |
| **StreamingLLM sink token**（Xiao et al., ICLR 2024, arXiv 2309.17453 §"Pre-Training with a Sink Token"；HTML 读）| 预训练时给每个样本前置一个可学习占位 token 作为专用 sink | 自注意力、语言模型；证明"显式 sink 优于依赖偶然 sink"。 |
| **Registers**（Darcet et al., ICLR 2024, arXiv 2309.16588）| ViT 加若干可学习 token 吸收高范数 artifact | 同类；Jiang et al. NeurIPS 2025（test-time registers）说明该作用可事后转移。 |
| **gpt-oss attention sink**（OpenAI 2025 model card, arXiv 2508.10925）| 每头一个可学习标量加进 softmax 分母，"允许一个头不看任何 token"，**value 为零** | "只当 sink、不注入内容"的极简形式；与本候选的差别在于本候选的 token 有 value（带内容进残差流）。 |
| Bondarenko et al. NeurIPS 2023（clipped softmax / gated attention）、Miller 2023（softmax₁）、Vig & Belinkov 2019（null attention）、Barbero et al. COLM 2025（sink 防止 over-mixing）| 解释与替代方案 | 相关工作里说明"attend to nothing"的谱系即可。 |

### 2. 条件化（对应 `full` 对 `const_token` 的差别）

- **CoCoOp**（Zhou et al., CVPR 2022, arXiv 2203.05557）：v_m(x) = v_m + π，π = h_θ(x)，可学习 token 加上由实例特征算出的条件向量。本候选 n_m = b_m + W_m c 是同一形式，h_θ 退化为 4→128 的线性层，条件量 c 是视频级裁定摘要而非图像特征。
- **Video-Specific Query-Key Attention for WTAL**（arXiv 2305.04186）：弱监督时序动作定位里按视频生成 query。说明"视频条件化的可学习 token"在 WTAL 已有先例，但用于 query、用于类别检测，不是空 key。
- 视频级摘要广播回实例（候选 3 复核已引 GIG-VAD、PEL4VAD TCA、bag-level context）：本候选与它们的差别是摘要**不加到每一行**，而是作为一个可选的 key/value，由每行自己的注意力权重 p_{t,0} 决定拿多少。这一点应写清楚：第 t 行从空 token 拿到的贡献是 p_{t,0}·(W_o W_v n_m)，即"一个每视频常量向量 × 一个每行标量"，条件进入是 c 的秩 ≤ 4 线性像；它不是候选 3 C 部件那种线性头下的纯视频级 logit 偏移（这里经过残差 + FFN + 每行权重，与内容有交互）。

### 3. 同任务的功能等价物（弱监督多模态定位里"允许一帧不用另一模态"）

- **Leaky Gated Cross-Attention**（Lee, Yun, Jain, WACV 2022）：弱监督多模态时序动作定位（RGB/flow，THUMOS14/ActivityNet1.2），多阶段跨注意力为基线，每帧一个门决定用"跨模态注意后的特征"还是"自身特征"，非选中支路以小强度泄漏作正则。**这是与本候选动机最接近的同任务先例**（"模态互补关系弱时融合有害，需要逐帧决定是否依赖另一模态"），机制不同：门作用在注意力输出上，本候选把"不依赖"做成注意力内部的一个候选 key。论文必须引用并对照；见必须修改第 3 条。
- Missing-aware prompts（Lee et al., CVPR 2023, arXiv 2303.03369）等：为缺失模态设置可学习 token，是"模态整体缺失"，不是逐帧。

### 4. padding 作偶然 sink、训练/测试不一致

- Himelstein et al. 2025 "Silent Tokens, Loud Effects: Padding in LLMs"（arXiv 2510.01238）：LLM 推理期未屏蔽 padding 会改变激活与输出。最近的相关分析，但对象是推理期，不涉及训练用 padding、测试无 padding 的结构性不一致。
- NMT 跨注意力 sink（arXiv 2605.01229）、diffusion LM sink（arXiv 2510.15731）、attention sink 综述（arXiv 2604.10098）：sink 现象跨架构普遍；无一涉及弱监督视频或 padding 作 sink。
- WSVAD / WTAL / 音视频暴力检测（查询 34、35、38）：没有找到任何分析"padding 行参与注意力"的工作。本候选 README 第 0.2 节的测量（.254 注意力落在 padding；屏蔽后 HateMM ROC −.017）如果在 MACIL-SD 原始设置（XD-Violence）上也成立，本身就是可发表的观察；至少应在两语料的候选 1 骨干上作为论文的诊断图给出。

## 必须执行的修改（REQUIRED）

1. **引用并对照 Mask-Align leaky attention（ACL 2021）作为空 token 的直接先例**，同时引 Sun et al. 2024 explicit attention bias、Sukhbaatar et al. 2019、PyTorch `add_bias_kv`；正文必须写明 `const_token` 臂就是这一形式。novelty 表述只能落在：(a) 诊断（MACIL-SD 族弱监督骨干里 padding 充当偶然 sink 且训练/测试不一致，屏蔽反而变差），(b) 用冻结 VLM 裁定的视频级摘要作 token 条件，(c) 在弱监督跨模态 MIL 上的验证。不得把"空 token"或"条件化 token"本身写成贡献。
2. **相关工作补 attention-sink / register 谱系**：Xiao et al. ICLR 2024（sink token）、Darcet et al. ICLR 2024（registers）、gpt-oss 2025（分母标量 sink）、Bondarenko et al. NeurIPS 2023、Miller 2023、Vig & Belinkov 2019、Barbero et al. 2025；padding 部分引 Himelstein et al. 2025。条件化形式引 CoCoOp（CVPR 2022）。
3. **预注册补一个臂 `gated_cma`**（同任务的功能等价物）：去掉空 token、屏蔽 padding，在跨模态注意力输出上加一个每行 sigmoid 门 g_t = σ(w·[x_t; attn_t] + b)，h = x + g_t·attn_t（Leaky Gated Cross-Attention 的最小形式，不带泄漏项）。没有这个臂，"把'不依赖另一模态'做成注意力内部的空 key"与"在输出上加门"区分不开，审稿人必问。判据：`full` 三 seed 两语料 pooled 不低于 `gated_cma`，否则只能主张诊断，不能主张空 token 形式。
4. **预注册补一个臂 `zero_value_sink`**：空 token 只有 key（可学习）、value 强制为零（gpt-oss / softmax₁ 形式），padding 屏蔽。它与 `const_token` 的差别只在"sink 是否带内容进残差流"，与 `full` 的差别在"内容是否由裁定摘要条件化"。没有它，"attending to nothing 等于拿到视频级证据上下文"这句主张无法与"只是需要一个 sink"区分。主张链固定为：`no_token_masked` < `zero_value_sink`（sink 必要）；`zero_value_sink` ≤ `const_token`（value 有用）；`const_token` < `full`（条件化有用）；任一环不成立，主张就止于前一环。
5. **主表对照行不变**：`no_token_unmasked`（候选 1 骨干原样）仍是主对照；同时报 MultiHateLoc（同评测器复现）。padding 屏蔽单独不作主张。

以上第 3、4 条是预注册必须新增的臂；第 1、2、5 条是写法与引用要求。其它实现细节（token 拼在位置 0、mask 拼接、EMA 删除）不构成 novelty，也不需要额外过审。

## 评分理由（5/10）

- +：四类 STOP 均不触发；设计由一个可复现的测量导出（候选 3 README 7.1）且预期可证伪；五个结构臂已在代码里；hateful video 与 WSVAD 文献里无同类诊断与部件；条件来源（VLM 裁定视频级摘要经空 token 进入）无先例。
- −：核心部件与 Mask-Align leaky attention 形式相同，且至少四个独立来源都已给出同一构造；条件化形式是 CoCoOp；参数增量 1,280（+0.4%）；单独看"空 token"没有 novelty，能否成立取决于第 3、4 条新增臂的结果。比候选 3（6/10）低一分，因为候选 3 的三个部件至少各自的输入轴（两粒度格子）没有先例，而本候选的部件在同一位置（跨注意力 key/value 序列前）以同一目的（吸收无处可去的注意力）已被发表过。

---

# English appendix: search log

All queries were executed 2026-09-04 via the WebSearch tool, verbatim, in this order.

1. `Efficient Streaming Language Models with Attention Sinks Xiao 2023 attention sink learnable sink token`
2. `Vision Transformers Need Registers Darcet 2023 register tokens attention artifacts`
3. `Quantizable Transformers removing outliers by helping attention heads do nothing Bondarenko clipped softmax gated attention`
4. `Attention Is Off By One Evan Miller softmax1 no-op attention`
5. `Vig Belinkov analyzing attention BERT attends to [SEP] no-op null attention`
6. `null token cross-attention "attend to nothing" learnable null key value dummy slot`
7. `Why do LLMs attend to the first token Barbero 2025 attention sink over-mixing`
8. `Massive Activations in Large Language Models Sun 2024 explicit attention bias learnable key value augmenting attention`
9. `Augmenting Self-attention with Persistent Memory Sukhbaatar 2019 persistent memory vectors key value`
10. `PyTorch MultiheadAttention add_bias_kv add_zero_attn learnable bias appended to key value sequence`
11. `gpt-oss learned attention sink per head scalar softmax denominator sink token 2025`
12. `padding tokens unmasked attention sink train test mismatch padding attention transformer analysis`
13. `DETR no-object class learnable object queries "no object" background slot set prediction`
14. `Set Transformer inducing points learnable ISAB Lee 2019`
15. `CoCoOp conditional context optimization instance-conditioned prompt tokens meta-net image feature added to learnable token`
16. `missing modality learnable token prompt multimodal transformer "missing-aware prompts" placeholder token absent modality`
17. `UR-DMU dual memory units learnable normal abnormal memory weakly supervised video anomaly detection`
18. `VadCLIP learnable prompt tokens PEL4VAD MGFN CLIP-TSA weakly supervised video anomaly detection learnable global token`
19. `hateful video detection "attention sink" OR "register token" OR "null token" HateMM MultiHateClip`
20. `hateful video localization padding mask cross-modal attention MultiHateLoc MACIL-SD padding unmasked`
21. `cross-modal attention learnable "null" key value audio-visual "no corresponding" modality absent attention fallback token`
22. `audio-visual event localization background token learnable "background" query temporal action localization transformer`
23. `"sink token" conditioned on input OR context-dependent sink OR "conditional register" transformer learnable`
24. `Silent Tokens Loud Effects padding in LLMs 2025 unmasked padding effect`
25. `Attention Sink in Transformers survey utilization interpretation mitigation 2026 learnable sink cross-attention`
26. `multiple instance learning transformer weakly supervised video anomaly detection "global token" OR "video-level token" OR "class token" cross-modal fusion audio visual`
27. `"attention sink" video transformer OR "video-language" OR multimodal cross-attention sink register 2025 2026`
28. `learnable "no-match" token OR "null slot" cross-attention retrieval "abstain" attention machine translation "null word" alignment`
29. `Global Information Guided video anomaly detection global pattern learnable video-level token weakly supervised`
30. `weakly supervised temporal action localization learnable "context token" OR "summary token" OR "prototype token" attention 2024 2025`
31. `Vision Transformers Don't Need Trained Registers test-time registers Jiang 2025 register neurons`
32. `input-dependent register tokens OR "dynamic register" OR "adaptive sink token" generated from global feature transformer 2025`
33. `Memory Transformer Burtsev memory tokens learnable prepended tokens 2020`
34. `"attention sink" OR "sink token" weakly supervised video anomaly detection OR temporal action localization MIL`
35. `audio-visual violence detection XD-Violence cross-modal attention learnable token "global" OR "null" OR "memory" MACIL-SD follow-up 2024 2025`
36. `HVGuard OR SafeLens OR "HateClipSeg" OR LELA hateful video "learnable token" OR "sink" OR "register" cross-modal attention`
37. `multiple instance learning attention pooling learnable "background" OR "null" instance dummy instance bag attention 2023`
38. `zero padding attention effect weakly supervised video anomaly detection variable length videos padding mask ablation RTFM MGFN`
39. `Attention Sinks in Massively Multilingual Neural Machine Translation cross-attention EOS sink mitigation 2026`
40. `context-conditioned learnable key value "summary vector" appended to keys cross-attention video-level conditioning weakly supervised localization`
41. `Mask-Align self-supervised neural word alignment leaky attention NULL token extra leak position Chen Sun Liu ACL 2021`
42. `"leaky attention" OR "leak attention" cross-attention NULL token learnable parameter alignment transformer`
43. `Leaky Gated Cross-Attention weakly supervised multi-modal temporal action localization Lee WACV 2022 gate cross-attended feature`

## Pages actually opened (WebFetch)

| URL | Paper | Result |
|---|---|---|
| https://ar5iv.labs.arxiv.org/html/2012.07162 | Mask-Align (ACL 2021), leaky attention | full HTML read; K = Concat(k_NULL, H_enc W_K), V = Concat(v_NULL, H_enc W_V); learnable, small-norm init; purpose = absorb biased attention |
| https://arxiv.org/pdf/2012.07162 and https://aclanthology.org/2021.acl-long.369.pdf | Mask-Align PDF | fetched, not parsed (binary); superseded by the ar5iv read |
| https://arxiv.org/html/2402.17762 | Massive Activations (Sun et al. 2024) | §4.3 read: k′, v′ constant learnable per head, concatenated to K/V; GPT-2 parity, massive activations disappear |
| https://arxiv.org/abs/2402.17762 | same | abstract only |
| https://arxiv.org/html/2309.17453v3 | StreamingLLM (ICLR 2024) | "Pre-Training with a Sink Token" read: prepended learnable placeholder; Table 3/4 results |
| https://arxiv.org/abs/2309.17453 | same | abstract |
| https://arxiv.org/html/2512.10408v3 | MultiHateLoc (WWW 2026) | full HTML re-read for null/sink/register/padding: none; attention on concatenated modality features, no masking described |
| https://arxiv.org/abs/2305.04186 | Video-Specific Query-Key Attention for WTAL | abstract: per-video, per-class learnable queries |
| https://arxiv.org/html/2508.10925v1 | gpt-oss model card | sink = learned bias in softmax denominator, "pay no attention to any tokens" |
| https://openaccess.thecvf.com/content/WACV2022/papers/Lee_Leaky_Gated_Cross-Attention_..._paper.pdf | Leaky Gated Cross-Attention (WACV 2022) | HTTP 403; mechanism taken from the CVF abstract page and search snippets (per-frame gate between cross-attended and own feature, leaky non-selected path) |

Identified from search snippets and cited above without opening: Darcet et al. (arXiv 2309.16588), Bondarenko et al. (arXiv 2306.12929), Miller 2023 (evanmiller.org), Vig & Belinkov 2019, Barbero et al. (arXiv 2504.02732), Sukhbaatar et al. (arXiv 1907.01470), PyTorch MultiheadAttention docs, Jiang et al. (arXiv 2506.08010), CoCoOp (arXiv 2203.05557), missing-aware prompts (arXiv 2303.03369), Himelstein et al. (arXiv 2510.01238), attention-sink survey (arXiv 2604.10098), NMT cross-attention sinks (arXiv 2605.01229), Memory Transformer (Burtsev et al. 2020), UR-DMU (arXiv 2302.05160), VadCLIP (arXiv 2308.11681), PEL4VAD (arXiv 2306.14451), DETR (arXiv 2005.12872), Set Transformer (Lee et al. ICML 2019), HVGuard (EMNLP 2025), SafeLens (AAAI 2026), HateClipSeg (arXiv 2508.01712), LELA (arXiv 2602.09637), TANDEM (arXiv 2601.11178), MARS (arXiv 2601.15115).
