# NOVELTY RECON — 两条候选机制轴的占位普查

Date: 2026-08-09 · 类型:纯文献 + 本地资料侦察(零 GPU,零 test 触碰)
目标档次:NeurIPS / ICML / ICLR / CVPR / ACL 主会 / ACM MM 及以上,**方法论文,机制必须 novel**
方法:WebSearch + arXiv API + 本地 `research-wiki/`(papers / experiments / PAPER_MASTER_TABLES / ITERATION_LOG / gap_map)

> **本文件的立场:诚实优先。** 下面两条轴我都给了"不够"的直判,并给出了导致该判决的具体证据
> (外部占位 + **我们自己已经跑过并证伪的实验**)。缝隙不存在的地方直接写不存在。

---

## 结论速览

| | 轴 1 — 流式/持续适应的检索 memory | 轴 2 — 跨语言检索对比(EN memory 救 ZH) |
|---|---|---|
| 核心机制被占? | **是,而且是最坏情况**:有一篇 EMNLP 2023(2209.05706)几乎一一对应,一篇 ICML 2023(AdaNPC)就是这个机制本身,还有一篇 EMNLP 2023(2305.13034)证明"零梯度 ≠ 梯度"这个 framing 是错的 | **是**,三重占位(Ghorbanpour EMNLP'25 主会 + Stap&Monz EMNLP'23 + MoRE WWW'25 / CRAVE ICCV'25) |
| 领域内已有直接前作? | SCANNER(AAAI 2026,但做的是跨平台不是时间) | SCANNER 已发表 MHClip 的 EN↔ZH 迁移数字 |
| 我们的资产支持? | **不支持** — 我方 W4 实验已把该机制证伪(memory 插入 flat-to-negative) | **不支持** — 我方 Phase-3b 实测 EN→ZH −0.138 macro-F1,且 motivation 方向是反的 |
| 顶会够不够? | **不够** | **不够**(前提本身站不住) |
| 排序 | 1(残值 = 一篇诚实的 negative-result 短文) | 2(零残值) |

**两条轴我都判"不够"。这不是留有余地的措辞 —— 我没有为了交差造缝隙。**

---

# 轴 1 — 流式 / 持续适应的检索 memory(evolving hate)

拟议 claim:*"第一个把 continual / streaming adaptation 形式化为 memory 插入问题的 hateful video
方法"*,配时间切分评估。

## 1.1 占位地图

### (A) 领域内(hateful / harmful content)

| 工作 | 出处 | 机制 | 占了什么 |
|---|---|---|---|
| **Class-RAG: Real-Time Content Moderation with RAG** | arXiv **2410.14881**(Meta GenAI, 2024-10) | DRAGON-RoBERTa 双编码器 + Faiss + safe/unsafe 双子库 + Llama-3-8B 分类器;检索库可热更新 | **直接占掉轴 1 的核心 claim。** 原文自述:"extends the capability of its base LLM through access to a retrieval library which can be **dynamically updated to enable semantic hotfixing** for immediate, flexible risk mitigation";"**low-cost adaptation … without requiring model retraining**";"performance **scales with the size of the retrieval library**"。§5.3 用"往库里加外部样本 → AUPRC 0.229→0.791"证明了"插样本即适应"。 |
| **SCANNER: Shedding the Facades, Connecting the Domains** | arXiv **2602.00132**, **AAAI 2026** | source-free TTA;K-means 求模态级 centroid + centroid-guided alignment + 样本级自适应 + intra-cluster 多样性正则 | **占掉"hateful VIDEO 的漂移适应"这个位子**,自称首个 HVD 的 TTA 框架。用 HateMM / MHC-YouTube / MHC-Bilibili 六个 A→B 设置,平均 +4.69 macro-F1。**无检索、无 memory bank。** |
| **MoRE: Biting Off More Than You Can Detect** | **WWW 2025** | 联合多模态视频检索器 + 双极 top-K 记忆 + MoE 专家 | **占掉"检索 = 应对 evolving hate"这套叙事本身**:MoRE 的动机第 1 条就是"hate 随时间演化 → 静态分类器失效",并主张检索增强提供 adaptability。 |
| **DriftGuard** | arXiv **2606.28725**(2026-06) | 五路 drift 监测(global text / identity-harm / uncertainty / toxic-risk / FN-risk)+ hard-mix 选择性再训练 | 占掉"**drift 监测 → 选择性适应**"这条路;并且**已经用了真正的时间切分**(Civil Comments 自然时间推进 + Jigsaw→DynaHate)。 |
| **Hatevolution: What Static Benchmarks Don't Tell Us** | arXiv **2506.12148**, **Findings of ACL 2025** | 20 个 LM × 两个 evolving-hate 实验,量化 static vs time-sensitive 评估的错位 | 占掉"**时间敏感评估协议**"这个贡献点(在 text hate 上)。 |
| HateDebias | arXiv **2406.04876** | continual-learning setting 下的 hate-debias benchmark | 占掉"hate + continual learning benchmark"。 |
| Evolving Hate Speech Online | arXiv **2502.10921** | 词嵌入驱动的 lexicon 增量更新 + 混合模型 | 占掉最朴素版本的"增量适应新黑话"。 |
| **CRAVE — Borrowing Eyes for the Blind Spot** | **ICCV 2025** | Pseudo-Pair Retriever,从资源丰富的 image-text 域跨域检索,救数据稀缺的恶意视频检测 | 占掉"**在恶意视频检测里,用检索从更丰富的池子借证据**"—— 而且是在 CVPR-tier 主会上。 |
| StreamSense | arXiv **2601.22738**(2026-01) | 轻量流式编码器 + 选择性路由到 VLM 专家 | 占掉"streaming hate detection"这个词;但无 memory、无 drift 适应。 |

### (B) 通用机制(与领域无关,"搬来即可"的部分)

> **先看三篇决定性的:下面这三篇单独任何一篇都足以让轴 1 的机制 claim 被 desk-reject。**

| 工作 | 出处 | 机制 | 判定 |
|---|---|---|---|
| **AdaNPC: Exploring Non-Parametric Classifier for Test-Time Adaptation** | arXiv **2304.12566**, **ICML 2023** | 摘要原句:"given a test instance, AdaNPC first **recalls K closed samples from the memory to vote for the prediction**, and then the **test feature and predicted label are added to the memory**." 零梯度,**带理论保证** | **☠️ 这就是轴 1 提议的机制本身,ICML 2023 已发表,还带理论。** |
| **Simple Temporal Adaptation to Changing Label Sets: Hashtag Prediction via Dense KNN** — Mireshghallah, Vogler, He, Florez, El-Kishky, Berg-Kirkpatrick | arXiv **2209.05706**, **EMNLP 2023**(`2023.emnlp-main.452`) | Twitter 全年数据,**交替 3 周训练 / 1 周测试、测试永远在未来**,12 个时间桶;**编码器冻结一次后永不重训**("freeze and re-use it for encoding every subsequent train/test week … we simply **swap out the encoded datastores** during updating");FAISS,K=1024,直接引 kNN-LM/kNN-MT。基线含 **12 个逐桶重训的参数化模型**,dense-KNN **+64%** 胜出。附完整 12×12 时间桶迁移矩阵 + 数据删除(datastore 编辑)能力 | **☠️ 与轴 1 的整套设计几乎一一对应**:冻结编码器 + kNN 记忆 + 时间切分 + 打赢重训基线。**这是提这条轴时的 desk-reject 引用。** 唯一区别:它是**换**最新桶的 datastore,不是**累积增长**。 |
| **kNN-MT is Meta-Optimizer on Output Projection Layer** | arXiv **2305.13034**, **EMNLP 2023** | 证明 kNN-MT **隐式地在输出投影层上执行梯度下降**,是 model fine-tuning 的一个特例 | **☠️ 堵死"记忆插入 vs 梯度更新是两种本质不同的机制"这个 framing。** 它们是同一件事的算力/延迟权衡。 |

| 工作(其余) | 出处 | 机制 | 判定 |
|---|---|---|---|
| **kNN-LM**(Generalization through Memorization) | arXiv **1911.00172**, **ICLR 2020** | 预训练 LM + kNN 插值,datastore 可换 | **CORE MECHANISM ALREADY PUBLISHED.** 摘要原句:"allows for effective **domain adaptation, by simply varying the nearest neighbor datastore, again without further training**." |
| **kNN-MT**(Nearest Neighbor MT) | arXiv **2010.00710**, **ICLR 2021** | 对 datastore 的 kNN 分类器,零额外训练 | **CORE MECHANISM ALREADY PUBLISHED.** 摘要原句:"allows a single model to be **adapted to diverse domains by using a domain-specific datastore**, improving results by an average of **9.2 BLEU** … **without training on these domains**." 换库适应的效应量比我们能拿到的大一个数量级。 |
| **TDA: Efficient Test-Time Adaptation of VLMs** | arXiv **2403.18293**, **CVPR 2024** | "a lightweight key-value cache that maintains a **dynamic queue with few-shot pseudo labels as values and the corresponding test-sample features as keys**";另加 **negative pseudo labeling** 压伪标签噪声 | **把"插入 = 适应"和"插入污染控制"一起占了。** 我原本设想的"插入污染控制"缺口就是它的第二个组件。 |
| **TT-RAA: Test-Time Retrieval-Augmented Adaptation for VLMs** | **ICCV 2025** | 检索最相似数据库中心以 training-free 改进预测 | 占掉"测试时检索式适应"。 |
| Online Learning via Memory: Retrieval-Augmented Detector Adaptation | arXiv **2409.10716**, **ECCV 2024 HCV workshop**(注:workshop 非主会) | 检索增强分类模块 + 可更新 memory bank,每类 ~10 张即可换域,detector 权重不动 | 占掉"少量新样本进 memory 即在线适应";档次低(workshop),但**先占了 idea**。 |
| **HippoRAG 2 / From RAG to Memory: Non-Parametric Continual Learning for LLMs** | arXiv **2502.14802**, **ICML 2025** | 非参数化持续学习 | **"non-parametric continual learning" 这个词组本身已被顶会占用。** |
| StreamingQA(⚠️ **ID 2205.11388 未独立核实**) | ICML 2022 | 14 年新闻;每季度把新文章渐进加入 DPR 检索空间 | 协议占位。**引用前须核 ID。** |
| Atlas §5.2(⚠️ **ID 2208.03299 未独立核实**) | — | 在 2017 索引上微调,换成 2020 索引不重训 | 占掉"换索引即时间适应"(单次 swap 版)。**引用前须核 ID。** |
| **RA-TTA: Retrieval-Augmented Test-Time Adaptation for VLMs**(Lee, Kim, Kang, Bang, Song, Lee) | **ICLR 2025**(经 proceedings PDF 核实) | 为每个测试样本检索外部 web 规模图像来适配 CLIP | **CORE。** |
| **T³AR: Train/Test-Time Adaptation with Retrieval**(Zancato et al.) | arXiv **2303.14333**, **CVPR 2023** | 通过对外部可检索池的检索做测试时适配 | **CORE。** |
| **Memory-Modular Classification**(Kang, Iscen, Jo, Choi, Cho, Schmid) | arXiv **2504.06021**, **TMLR** | **靠替换 memory 内容适应新类别,零重训** | ADJACENT(非漂移 framing),但把"换库即换任务"又占了一次。 |
| **NOTE** | arXiv **2208.05117**, **NeurIPS 2022** | prediction-balanced reservoir 记忆采样 | 占掉"记忆的类别平衡采样/替换"。 |
| **Efficient Cluster-Based kNN-MT** | arXiv **2204.06175**, **ACL 2022** | 聚类压缩 + 冗余剪枝 | datastore 压缩已解决。 |
| 检索库投毒一族:**PoisonedRAG**(2402.07867, **USENIX Security 2025**)、Zhong et al. corpus poisoning(2310.19156, **EMNLP 2023**)、**AgentPoison**(2407.12784, **NeurIPS 2024**)、TTA 投毒(2301.12576, ICML 2023;**RTTDP** 2410.04682, ICLR 2025) | 2023–2025 | 对抗性地污染检索库 / 测试时适配流 | **☠️ 占掉"检索 memory 的污染/投毒"整块地**(对抗侧)。 |
| 梯度式 TTA 崩溃经典:**CoTTA**(2203.13591, CVPR 2022)、**EATA**(2204.02610, ICML 2022)、**SAR**(2302.12400, ICLR 2023)、**PeTTA**(2311.18193, **NeurIPS 2024**,反复流下的崩溃诊断) | 2022–2024 | 长时程稳定性 | 占掉梯度侧的"长时程崩溃"。**非梯度侧见附录 C 第 1 条。** |
| Online Continual Learning Without the Storage Constraint | arXiv **2305.09253**(Prabhu, Cai, Dokania, Torr, Koltun, Sener) | kNN 分类器 + 冻结特征抽取器,显式 `Memory.Insert`/`Memory.Retrieve` 循环;CLOC(39M 图)与 CGLM 上比梯度式 CL **高 >20%** | 占掉"**在冻结特征上持续更新 kNN = 在线持续学习**"—— 正是轴 1 的架构。**⚠️ 更正:此文无正式发表 venue** —— 从 ICLR 2024 撤稿(OpenReview `E83OzFbNQ6`),以技术报告流通。 |
| **RoTTA** | arXiv **2303.13899**, **CVPR 2023** | CSTU memory bank,**显式的"年龄 + 不确定性"淘汰打分** `H = λ_t·sigmoid(A/N) + λ_u·U/log C`(A = 样本年龄) | **☠️ 占掉"漂移下按 staleness 淘汰记忆"** —— 我原本设想的第三个缺口。 |
| Lu, Lu, Zhang & López de Mántaras, *A concept drift-tolerant case-base editing technique* | **Artificial Intelligence 230:108–133, 2016** | 概念漂移下的 kNN 记忆编辑 | 同一件事,**十年前就做过**。 |
| **DeDrift: Robust Similarity Search under Content Drift** | arXiv **2308.02752**, **ICCV 2023** | 在线更新量化器维护大规模索引对抗内容漂移,比全量重建快 100× | 占掉"索引在漂移下的维护"。 |
| **KoK: Non-Parametric Online Learning from Human Feedback for NMT** | arXiv **2109.11136**, **AAAI 2022** | 把人工 post-edit **在线插入 datastore**,零梯度 | 占掉"流式插入新标注即适应"。 |
| cache 自污染 / 误差累积一族:**CRG**(2503.18334, ICME 2025)、**ACE**(2508.07570)、**DOTA**(2409.19375, NeurIPS 2025)、SCA(NeurIPS 2025) | 2025 | "cache noise"、"error accumulation within the cache"、cache-drop 遗忘;熵排序淘汰、负 cache、置信度门控准入 | **☠️ 占掉"插入污染控制"的所有一阶做法。** 置信度过滤准入**不能**当新意主张。 |
| **GradNormIR** | arXiv **2506.01877**, **Findings of ACL 2025** | 无监督 gradient-norm OOD 分数,作为"何时该更新稠密检索器"的触发器 | 占掉"何时更新检索器"(仅检测/时机,不含淘汰策略交互)。 |
| **TiC-CLIP**(**2310.16226**, **ICLR 2024**,已核实);同族 TiC-LM(2504.02107?)/ FoMo-in-Flux(2408.14471?)/ Wild-Time(2211.14238?)/ CLOC(2108.09020?)—— **⚠️ 这四个的 ID 未独立核实** | 2021–2025 | 时间连续基准与协议 | 占掉**时间切分评估协议**。**已核实:TiC-CLIP(及 TiC-LM)的续训基线全部是梯度式**(warm-start / cumulative / sequential / restart / patching / replay / distillation / cyclic-cosine),**没有任何检索 / kNN / cache 基线** —— 见附录 C 第 3 条。 |
| Gama, Sebastião & Rodrigues, *On evaluating stream learning algorithms* | **Machine Learning 90(3), 2013** | prequential(test-then-train)评估 + 滑窗/衰减因子遗忘 | 流式评估协议的**最早占位者**,比上面全部早十年。 |
| Learning to Query History | arXiv **2604.07027**(venue 未确认,疑 ICLR 2026 TSALM workshop) | 学习式离散检索策略采样历史样本,端到端训练 | 占掉"**学出来的**检索策略 for 非平稳分类"。(注:其 memory 在部署时**不增长**,基线也很弱。) |
| kNN-LM/MT 适应支线:Adaptive kNN-MT(**2105.13022**, ACL 2021)、Efficient kNN-LM(**2109.04212**, EMNLP 2021)、RetoMaton(**2201.12431**, ICML 2022)、SK-MT(**2302.12188**, ICLR 2023)、Adaptation Approaches for kNN-LM(**2211.07828**) | 2021–2023 | 学 k / 学插值权重、datastore 剪枝、按输入现建 datastore | 全部 ADJACENT,但把"调 datastore 的各种旋钮"这块地占满了。注意 2211.07828 的结论:**只改 datastore 会被"datastore + adapter"打败。** |
| Continual Test-Time Adaptation in CV: Methods, Benchmarks, Future Directions | arXiv **2607.08164** | 综述 | 说明 CTTA 已经成熟到有综述。 |
| The Illusion of Progress? A Critical Look at TTA for VLMs | arXiv **2506.24000**, **NeurIPS 2025 D&B Track** | 批判性复评:现有 TTA 相对开创工作增益有限,且"**accuracy gains frequently come at the cost of reduced model trustworthiness**"(鲁棒性/校准/可靠性) | 该子领域已经进入"批判现有进展"阶段 = **强拥挤信号**。新进入者必须先回答"你的增益是不是也只是幻觉"。 |
| BBSE / Saerens-EM 系 | Lipton et al. **ICML 2018**(arXiv 1802.03916);Saerens et al. 2002;Alexandari et al.;NeurIPS 2020 unified view | label / prior shift 估计与阈值修正 | **占掉"漂移下的阈值再校准"** —— 见 §1.2,这是我方唯一存活的正结果所在的位置,而它是 20 年前就解决的经典问题。 |

### (C) 天然实验协议(时间切分 / wave 增量插入)在本领域有人做过吗?

- **hateful VIDEO:上传时间切分 = 没有人做过。** 我用 arXiv 全库检索 + 定向 web 搜索都没找到任何
  hateful-video 论文按 upload date 切 train/test。SCANNER 虽然通篇讲 "evolving / shifting",
  **它的 shift 是跨平台跨语言的 dataset-to-dataset,不是时间**(原文:"each dataset originates
  from a distinct platform (BitChute, YouTube, Bilibili), differs in language … leading to
  substantial semantic drift")。这是个**真空**。
- **hate text:已经有人做**(Hatevolution ACL-F 2025 的 time-sensitive 评估;DriftGuard 的
  Civil Comments 时间流;"No Time Like the Present" arXiv 2207.04003)。
- **通用 ML:"wave 式增量插入 memory + 在后续时间片上评估"这个精确协议已经发表:**
  - **☠️ Mireshghallah et al.(EMNLP 2023, arXiv 2209.05706)—— 全文核实,最致命的一篇。**
    交替 3 周训练 / 1 周**未来**测试,覆盖 2021 全年 12 桶;BART 只在第一桶微调一次然后**永久冻结**;
    逐月建 FAISS datastore;**有"更新 datastore vs 保留第一桶"的消融**(+19% 第一桶预测不出的 hashtag);
    **有完整的 train-bucket × test-bucket 迁移矩阵**(其 Fig. 4);对比对象包含**持续重训的参数化模型**,
    胜出 **+64%**;显式引 kNN-LM / kNN-MT;还附 datastore 删除(隐私合规)能力。
    **机制、协议、消融,一篇全做完了。**
    (注:arXiv 标题是 *Non-Parametric Temporal Adaptation for Social Media Topic Classification*,
    与发表标题不同 —— 这就是它容易被漏检的原因。)
  - CLOC(39M 图)/ CGLM 上的在线流版本见 §1.1-B 的 `2305.09253`。
  - 更早:**Gama, Sebastião & Rodrigues, Machine Learning 90(3), 2013** 的
    prequential(test-then-train)评估 + 滑窗/衰减遗忘。
  - 时间连续基准群:**TiC-CLIP(2310.16226, ICLR 2024,已核实)**;
    ⚠️ StreamingQA / Wild-Time / TemporalWiki / CLOC / TiC-LM / FoMo-in-Flux
    **确实存在但其 arXiv ID 未独立核实,引用前须自查**(见文末声明)。

  ⇒ 所以"我们把它引进 hateful video" = **把一个 2013 年就有、2023 年被 EMNLP 精确实例化过的
  协议,搬到一个新数据集上。**

## 1.2 剩余缝隙的精确表述 —— 以及为什么它已经被我们自己关掉了

把上面减一减,**形式上**剩下的缝隙只有一句话:

> 「在 hateful **video** 上,用**上传时间**构造真实漂移,并证明**往 kNN memory 插入新时段样本**
> 能零梯度地恢复漂移损失。」

这句话的每个部件单独都不新,而且**整句话在别的域已经被完整写过一遍**:
Mireshghallah et al.(EMNLP 2023)= 冻结编码器 + kNN 记忆 + 未来时间片测试 + 打赢逐桶重训基线;
AdaNPC(ICML 2023)= 检索投票后把样本插回记忆的零梯度适应,带理论;
StreamingQA(ICML 2022,**ID 未核实**)= wave 式插入 + 后续时间片评估。
我们能加的只有"载体是视频、内容是仇恨"。它唯一可能的价值在于**实证结论**。

**而这个实证结论我们已经拿到了,是负的。** `research-wiki/EVAL_temporal_memory_W4.md`
(2026-07-03/04,jobs 12197/12214/12253):

- 时间切分**造出来了**,漂移**在 EN 上是真的**:MHC-EN 随机切分 macro-F1 0.7113 → 时间切分
  **0.6273**(−0.084)。ZH **没有漂移**(0.7641 → 0.7779,+0.014)—— 只有一条腿。
- **机制 A(往 memory 加新时段样本)= flat-to-negative,全灭**:
  k ∈ {5,10,20,50,80} × 三种选样策略(random / latest / uncertain)× 两种语言,
  **每一格的 per-sample adaptation gain ≤ 0**;把整个新时段 val 池(80 条)全加进去反而掉到
  **0.5923**(比静态 0.6273 更差)。
- 真正的失败模式是**校准漂移不是可分性损失**:时间切分下 ROC 反而更高(0.8484 > 随机切分参考
  0.7175),只有 8.7% 的测试分数越过 0.5 而真实正例率是 24.2%。
- 唯一有效的修复是**阈值再校准**:k=20 条新时段标注样本把 0.6273 拉回 **0.7336**(≥ 随机切分
  floor 0.7113,全额收复)。

也就是说:**轴 1 的机制在我们自己的数据上已经被证伪了一次**,而幸存下来的那条(阈值再校准)
恰好落在 BBSE / Saerens-EM(§1.1-B 最后一行)这个 1998–2018 年间就被解决透的经典问题里。
更糟的是,我们连"零标注版本的再校准"都试过并失败了:**P1 零标注先验重校准**
(`PAPER_MASTER_TABLES.md` T4 行 1)先验估计误差 0.22 EN / 0.18 ZH(判据要 ≤0.07),FAIL。

## 1.3 顶会够不够 —— **不够**

理由,按杀伤力排序:

1. **机制在通用 ML 里已被完整占位,且有一篇近乎一一对应的论文。**
   **Mireshghallah et al., EMNLP 2023(2209.05706)** 做的是:冻结编码器一次 → FAISS kNN 记忆 →
   交替 3 周训练 / 1 周**未来**测试 × 12 桶 → 逐周换 datastore → **打赢 12 个逐桶重训的
   参数化模型 +64%** → 附 12×12 时间迁移矩阵 + datastore 编辑(数据删除)能力。
   我们的设计与它的差别只有"载体是视频"和"换库 vs 累积增长"。**这是 desk-reject 引用。**
2. **零梯度插入式适应本身是 ICML 2023 的已发表机制,还带理论。**
   **AdaNPC(2304.12566, ICML 2023)**:检索 K 个近邻投票 → 把测试特征 + 预测标签插回记忆。
   这就是轴 1 的机制,一字不差。
3. **"插入 ≠ 梯度"这个 framing 已被证伪。** **kNN-MT is Meta-Optimizer on Output Projection
   Layer(2305.13034, EMNLP 2023)** 证明 kNN-MT 隐式在输出投影层上做梯度下降,是 fine-tuning
   的特例。所以"零梯度"不是机制上的不同,只是算力/延迟的权衡 —— 一个工程卖点,不是科学贡献。
4. **我原本准备当缺口的三件事,三个都不空:**
   - *插入污染控制* → **TDA 的 negative pseudo-labeling(CVPR 2024)**,加上 CRG / ACE / DOTA / SCA
     这一整族 2025 年工作(cache noise、error accumulation within the cache、cache-drop 遗忘)。
     **置信度门控准入、熵排序淘汰、负 cache 全部已发表。**
   - *memory 淘汰 / staleness 策略* → **RoTTA(CVPR 2023)** 的 CSTU 记忆库有显式的
     "年龄 + 不确定性"淘汰打分;更早还有 **Lu et al., Artificial Intelligence 2016** 的
     概念漂移下 case-base 编辑。
   - *漂移下的索引/空间维护* → **DeDrift(ICCV 2023)**;编码器动了那一侧还有 BCT/FCT/FastFill/SDC。
5. **领域内也已被占,且占位者是 Meta 的 Class-RAG。** "可更新检索库 → 零重训适应新出现的 harm"
   这句话 Class-RAG 用几乎相同的措辞写过("semantic hotfixing")。
   而 **"non-parametric continual learning" 这个词组本身已被 HippoRAG 2(ICML 2025)占用。**
6. **我们的实证结论是负的。** 顶会方法论文需要机制 work。我们手上的证据是它不 work
   (flat-to-negative,80 条全加更差)。
7. **幸存的正结果(阈值再校准)不是新机制。** 它是 label-shift adaptation 的一个实例,
   而且只在一条腿(EN)上成立,ZH 是负对照(k=5 时 −0.067)。
8. **这个子领域已经进入"批判现有进展"阶段。** *The Illusion of Progress?*
   (2506.24000, **NeurIPS 2025 D&B**)结论是 TTA 相对最早工作增益有限、与微调组合性差、
   且"accuracy gains frequently come at the cost of reduced model trustworthiness"。
   任何"我们的 cache 适应得更好"的主张都会正面撞上它。
9. **规模不够支撑一个 continual/streaming claim。** 我们的时间测试集是 **EN n=161 / ZH n=149**,
   EN 测试窗口只有 2024-01→2024-05,正例率还从 train 的 34% 掉到 test 的 24%
   (先验漂移与语义漂移混淆)。加上死链存活偏差(Hateful/Offensive 死链率显著高于 Normal:
   ZH-Hateful 18.8% vs Normal 8.6%),时间测试片**系统性地少了那个时代最有害的内容**。
   这个规模下画不出 wave-wise 的 streaming 曲线。
10. HateMM / ImpliHateVid **无法定年**(ID 已匿名化,官方发布物无 URL / 平台 ID / 日期,
   见 `TEMPORAL_SPLIT_FEASIBILITY.md`)。**只有 MultiHateClip 两个子集能做时间切分**,
   而它们正好是我们最小的两个库。没有扩规模的路径。

**可能的降级出路(仅供记录,我不推荐):** 把它写成一篇 *analysis / negative-result* 论文——
"检索 memory 的插入式适应在 hateful video 上不 work,漂移的主成分是校准而非可分性"。
这是诚实且有信息量的,但它是 findings / workshop / 短文档次,不是 NeurIPS 主会方法论文。

## 1.4 我们资产的匹配度:**低**

| 资产 | 对轴 1 的价值 |
|---|---|
| 时间切分基础设施(exp-temporal-split-infra, verdict=yes) | ✅ 真资产,而且**领域内独有**(没人做过 hateful video 的上传时间切分)。但它已经被用掉,并给出了负结果。 |
| 跨数据集可更新 kNN memory(优于 MoRE) | ⚠️ 是 **capability demo 不是 performance win**:6 个 informative cross cell 里 5/6 above-majority,但**跨库从不超过 in-domain**,并且以 MHC-EN 为目标时**塌到 majority baseline**(HateMM→MHC 0.548 F1 / 0.696 acc ≈ 多数类 0.6957)。 |
| RGCL 管线 / OCR cache / 多语言资产 | 与本轴无关。 |

## 1.5 最快的杀死性实验

**已经跑完了,已经杀掉了。** `EVAL_temporal_memory_W4.md` 就是那个 kill experiment
(CPU 级 + 一次 SLURM 训练,~1 GPU-h),判决 = memory 插入 flat-to-negative。

如果要再补最后一刀(把"也许是我们 memory 插入策略太笨"这个辩护也堵死),**最便宜的一发**是:

> **Oracle-insertion ceiling($0,纯 CPU,分钟级)。** 用**测试集标签**(仅作诊断天花板,
> 不进任何 claim)去选"最理想的 k 条新时段样本"插入 memory,扫 k=1…80。
> **杀死判据(先冻结):若 oracle 插入的 macro-F1 天花板 < 阈值再校准的 0.7336,
> 则"memory 插入"这条路在信息论上就没有可挖的余量 —— 关闭轴 1,永不再提。**
> 预期:会杀掉。因为 rank-weighted 20-NN 投票下,80 条样本改不动 549 条 memory 的邻域结构,
> 而 W4 已经显示"全加(80 条)"是所有 k 里最差的一格。

---

# 轴 2 — 跨语言检索对比(EN memory 救 ZH)

拟议 claim:*"用语言无关的检索空间让 EN 训练数据 / 邻居直接支援 ZH 推理"*,
故事前提是"低资源语言的仇恨检测差(ZH 0.83 vs EN 0.87)"。

## 2.1 先纠正前提 —— **这个前提在我们自己的数字上不成立**

任务书里的 "ZH 0.83 vs EN 0.87" 是**跨数据集比较**:0.87 是 **HateMM**(英文,BitChute,
另一个数据集)的数字,0.83 是 **MHC-ZH**。在**同一个 benchmark(MultiHateClip)**上,
`PAPER_MASTER_TABLES.md` T1.1 的多 seed 数字是:

| 数据集 | 最优栈 | val-sel acc | val-sel macro-F1 | final-ep acc | final-ep macro-F1 | seeds |
|---|---|---|---|---|---|---|
| **MHC-EN** (161) | frozen-Qwen + archive-kNN | 0.7935 ± 0.0205 | **0.7497 ± 0.0250** | 0.7826 | 0.7430 | 4 |
| **MHC-ZH** (149) | LoRA-Qwen | 0.8282 ± 0.0139 | **0.7962 ± 0.0167** | **0.8537 ± 0.0120** | **0.8259 ± 0.0124** | 5 |

**在 MultiHateClip 上,我们的 ZH 比 EN 高 5–8 个 macro-F1 点。** MoRE 同场对比表(T1.2)也是
同一方向(ours ZH 0.8023 F1 vs ours EN 0.7378 F1)。

⇒ **"用 EN memory 救低资源的 ZH" 这个故事方向是反的。** 在我们的系统里 ZH 是强的那一边。
审稿人会拿我们自己主表里的数字打这个 motivation。

**更糟:连 benchmark 原论文的那个 gap 也不支持这个故事。** MultiHateClip 的 EN>ZH 差距
**只存在于多分类**(macro-F1 EN 0.63 GPT-4V vs ZH 0.50 M1);**在二分类上是 EN 0.79 vs ZH 0.78,
几乎为零** —— 而**我们做的正是二分类**(offensive 折叠进 hateful,与 MoRE 同协议)。
所以在我们的评估口径下,"低资源语言更差"这个前提在 benchmark 层面就不成立。

**次生问题:** "CLIP 英文文本塔对中文是短板"是真的
(`PAPER_MASTER_TABLES.md` T4 行 9:冻结 English-centric CLIP text tower 把中文 byte-fragment
97% 截断),但**我们的 ZH 最优栈根本不用 CLIP text tower**,它用 LoRA-Qwen。这个短板已经被绕开了,
不能再当 motivation 用。

## 2.2 占位地图

### (A) 领域内 — 跨语言 hateful content 的检索 / memory 迁移

| 工作 | 出处 | 机制 | 判定 |
|---|---|---|---|
| **Data-Efficient Hate Speech Detection via Cross-Lingual Nearest Neighbor Retrieval** — Ghorbanpour, Dementieva, Fraser | arXiv **2505.14272**, **EMNLP 2025 Main** | bge-m3 多语言嵌入;目标语言的小 seed 集去查一个 **265,671 条 / 14 数据集 / 9 语言(67% 英文)** 的池子;MMR 去重;检索到的样本**并入微调集**。检索**显式跨语言**,并分析哪个源语言帮哪个目标语言 | **直接占位,占的就是轴 2 的核心 claim。** 8 个目标语言;只取 20 条即超过 target-only 训练;<50 标注时 >10 F1-macro 增益。**唯一差别是它是 text-only。** |
| **SCANNER** | arXiv **2602.00132**, **AAAI 2026** | source-free TTA;同语言用 CLIP,**跨语言时换成 multilingual Sentence-BERT "to ensure effective cross-lingual representation extraction"** | **占掉"MultiHateClip 上的 EN↔ZH 迁移数字"。** 六个 transfer pair 全跑:MHY→HMM 64.63、**MHY→MHB 60.57**、HMM→MHB 56.49、HMM→MHY 62.90、**MHB→MHY 60.10**、MHB→HMM 58.59。⇒ "**没人量过 MultiHateClip 的 EN↔ZH 迁移**"这句话**已经不成立**。 |
| **CRAVE — Borrowing Eyes for the Blind Spot: Overcoming Data Scarcity in Malicious Video Detection via Cross-Domain Retrieval Augmentation** | **ICCV 2025** | Pseudo-Pair Retriever,从资源丰富的 image-text 域检索,救数据稀缺的恶意视频检测 | **关键先例。**"从资源丰富的池子借邻居来救数据稀缺的目标"这句话**已经在 ICCV 主会上、就在这个任务族里发表了**。轴 2 = 同一句话把 "modality/domain" 换成 "language"。 |
| **RAG and Recall: Multilingual Hate Speech Detection with Semantic Memory** | **WOAH 2025**(`2025.woah-1.20`) | Chroma 向量库覆盖 En+Fr+Ar 合并语料 + semantic cache memory,喂 LLaMA-3-8B | OCCUPIES(弱):同一格子,workshop 档次,未隔离跨语言效应。 |
| **Retrieval Augmented Enhanced Dual Co-Attention … Bengali Hateful Meme** — Tanvir & Alam | arXiv **2602.19212**(2026-02) | **CLIP ViT-B/32 + XLM-R-Large + FAISS kNN + 检索加权标签融合** | **架构上最像我们的东西**(多语言编码器 + FAISS kNN + 融合),而且它**主动没走跨语言**(检索池纯孟加拉语)。⇒ 我们那个"显而易见的下一步"在审稿人眼里 = 对一个已发表架构改一行。 |
| **MoRE** | **WWW 2025** | 检索增强 MoE | ADJACENT:三库(含中文 MHC-B)都跑,但**每库单语**;其 "cross-dataset generalization" 实验是 **HateMM↔MHClip-Y 即 EN↔EN**。 |
| **AHA-Memes**(阿拉伯语 meme) | arXiv **2607.27393** | jina-clip-v2 多模态多语言嵌入 + RRF 检索式 few-shot ICL | ADJACENT:检索池仅阿拉伯语训练集(单语)。K=5 把 binary macro-F1 0.643→0.708,**仍低于微调的 0.768**。 |
| **From Native Memes to Global Moderation** | arXiv **2602.07497**, **WWW 2026** | 跨文化 VLM 评测;翻译后检测**反而更差**,母语 prompt + one-shot 更好 | ADJACENT,但**在顶会上抢先占了"英文中心模型在非英文仇恨内容上失效"这个 framing**。 |
| **MultiHateClip** 本身 | arXiv **2408.03468**, **ACM MM 2024** | 数据集 + off-the-shelf baseline | 占掉 benchmark 与协议。**关键更正:该论文跑了零个 train-on-one-language / test-on-the-other 实验**,其 EN>ZH 是"在中文上表现更差"的 per-language gap,不是 transfer 结论。**而且这个 gap 只存在于多分类(0.63 vs 0.50);二分类是 EN 0.79 vs ZH 0.78,几乎没有差距** —— 而我们做的正是二分类。 |
| HVGuard / RAMF / MARS / 知识增强中文 MHSD 等 | EMNLP 2025 / TMLR 2025 / ICASSP 2026 / ESWA 2025 | EN+ZH 都评,CoT / 多视角 / 训练无关推理 | ADJACENT:覆盖中文,不是检索式跨语言迁移。 |
| Few-Shot Contrastive Adaptation for Audio Abuse in Low-Resource Indic | arXiv **2604.09094** | CLAP 共享音频-语言空间 + 少样本适配 | ADJACENT,且结论对我们不利:"further adaptation with a handful of labelled examples per language yields **little extra benefit**"。 |

> **MultiHateClip 的 47 篇引用文献已被逐条枚举**(Semantic Scholar 全量)。**触碰跨语言角度的只有 SCANNER 一篇**;
> 检索式的只有 MoRE 与 CRAVE,两者都不跨语言。其余 40+ 篇是分割/定位/可解释/新基准方向。

### (B) 通用机制 — "搬来即可"的跨语言检索(**全部已发表**)

| 工作 | 出处 | 确立了什么 |
|---|---|---|
| **Multilingual k-Nearest-Neighbor Machine Translation** — Stap & Monz | arXiv **2310.14644**, **EMNLP 2023** | **跨语言 datastore 的 canonical 论文。** 把多语言表征合并进**一个** datastore,低资源 query 直接检索高资源邻居;低资源 +3.6 BLEU。**这就是轴 2 声称的机制,三年前就发表了。** |
| **CORA: One QA Model for Many Languages with Cross-lingual Dense Passage Retrieval** — Asai et al. | arXiv **2107.11976**, **NeurIPS 2021** | mDPR **跨语言**检索段落;确立"一个语言无关检索空间把监督从高资源迁到低资源",26 语言 / 9 个未见语言。 |
| **PARC: Cross-Lingual Retrieval Augmented Prompt for Low-Resource Languages** | arXiv **2212.09651**, **Findings of ACL 2023** | 检索语义相似的**高资源语言**句子来增强低资源语言**分类**;10 个低资源语言 +5.1%/+16.3%。**与仇恨检测同形状的任务。** |
| **XRICL: Cross-lingual Retrieval-Augmented In-Context Learning** | arXiv **2210.13693**, **Findings of EMNLP 2022** | "跨语言 demonstration 检索"的起源:学出来的 retrieve-and-rerank 取**英文** exemplar 服务非英文 query。 |
| **XAMPLER: Learning to Retrieve Cross-Lingual In-Context Examples** | arXiv **2405.05116**, **Findings of NAACL 2025** | 专门训练 retriever 为目标语言 ICL 取**英文** demonstration;SIB200(176 语言)。 |
| **CREA-ICL** | arXiv **2311.06595** | 系统研究:跨语言检索增强 ICL **在分类上稳定有增益**,在生成上吃力。直接刻画了我们要落的那个 regime。 |
| **LaBSE** / bge-m3 / jina-clip-v2 | arXiv **2007.01852** 等 | 字面意义上的"语言无关检索空间",109 语言,现成可用。 |
| XRAG(2505.10089)、Multilingual RAG for Culturally-Sensitive Tasks(2410.01171, Findings ACL 2025) | 2025 | 跨语言 RAG 已是有自己 benchmark 与鲁棒性文献的子领域。 |
| C2KD(2210.03625)、NLLB-CLIP(2309.01859)、CL2CM(2312.08984)、MLLM-Enhanced CLCMR(2409.19961, ACM MM 2024) | 2022–2024 | **跨语言跨模态检索是成熟领域**,多语言 video-text 检索 + 对未见语言的零样本迁移已经能work。 |

⇒ **"用语言无关空间检索另一语言的邻居"零机制新意。** 它是 2021–2025 年间跨语言 IR / X-ICL 的默认做法,
且已被 Ghorbanpour et al.(EMNLP 2025 主会)直接搬进 hate speech。

## 2.3 剩余缝隙的精确表述

拟议 claim 可以分解成三个部件,**每一个都被一篇已发表论文单独占了**:

1. *跨语言检索把仇恨检测的监督从高资源语言迁到低资源语言* → **Ghorbanpour et al., EMNLP 2025 主会**。
2. *合并的多语言 datastore 让低资源 query 在共享空间里检索高资源邻居* → **Stap & Monz, EMNLP 2023**;
   **CORA, NeurIPS 2021**;demonstration 变体是 PARC / XRICL / XAMPLER。
3. *在训练样本 memory bank 上检索能提升 hateful **VIDEO** 检测* → **MoRE, WWW 2025**
   (我们那三个数据集、我们那个指标);外加 **CRAVE, ICCV 2025**("从资源丰富的池子借邻居救数据稀缺的
   视频审核目标")。

减完之后**形式上**剩下的是:

> 「在 hateful **video** 上,把 EN 与 ZH 的标注样本放进**同一个学出来的、检索引导对比的**
> 多模态空间,证明跨语言邻居对低资源语言推理有净增益。」

这只是"把 SCANNER 已建立的迁移设定上的 centroid alignment 换成检索"—— **一次方法替换,不是新机制。**
**而且它在实证上已经被我们自己关掉了。**

`ITERATION_LOG.md` Phase-3b(jobs 12136/12137,`src/eval_cross_dataset.py`),
warmup-consistent CLIP-RGCL transfer matrix,cell = macro-F1 / acc:

| trained-on ↓ \ memory=test= → | MHC (EN, maj .696) | MHC_zh (ZH, maj .698) |
|---|---|---|
| **MHC (EN)** | **0.711 / 0.783**(in-domain) | **0.633 / 0.758** |
| **MHC_zh (ZH)** | **0.645 / 0.739** | **0.771 / 0.805**(in-domain) |

- **EN 头 + ZH memory = 0.633,比 ZH in-domain 的 0.771 低 0.138 macro-F1。**
  ZH 的 acc 0.758 只是勉强越过 ZH 多数类基线 0.698。
- 反方向同样退化:ZH 头 + EN memory = 0.645 vs EN in-domain 0.711(−0.066)。
- Qwen 矩阵同向(EN→ZH 0.707/0.752)。
- 原文自评:"cross-lingual retrieval-memory transfer is **above-chance but clearly degraded**;
  it is a real signal, **not a strong one**"。

⇒ **EN 邻居不会"救"ZH,它会拖 ZH。** 唯一还没直接测的变体是"EN+ZH **合并** memory"
(而不是整库替换),但先验很差:既然纯 EN memory 比纯 ZH memory 差 13.8 点,把 EN 混进去
最好的情况也就是被 ZH 部分稀释后接近 ZH-only。

## 2.4 顶会够不够 —— **不够**(而且比轴 1 更糟)

1. **Motivation 站不住,两层都不成立。** 我们自己表里 ZH > EN(0.826 vs 0.743 macro-F1);
   benchmark 原论文在二分类口径下也只有 0.79 vs 0.78。一篇论文的第一句话就被自己的主表反驳。
2. **机制没有新意,且已经被搬进本领域了。** 跨语言检索(Stap & Monz EMNLP 2023 的合并 datastore、
   CORA NeurIPS 2021)、语言无关嵌入空间(LaBSE / bge-m3 / jina-clip-v2)、跨语言 demonstration
   检索(PARC / XRICL / XAMPLER)—— 全是标准件;而 **Ghorbanpour et al.(EMNLP 2025 主会)
   已经把"跨语言 kNN 检索救低资源仇恨检测"整套搬进 hate speech 并发表了**,只差一个模态。
3. **领域内已有多重占位者。** SCANNER(AAAI 2026)已经发表了 MultiHateClip 的 EN↔ZH 迁移数字;
   MoRE(WWW 2025)占了"hateful video 的检索 memory";**CRAVE(ICCV 2025)占了
   "从资源丰富的池子借邻居救数据稀缺的恶意视频检测"这句话本身**。
   我们要么被当成 SCANNER 的检索版增量,要么得同场打赢它 —— 而它是 source-free TTA(零目标标签)
   设定,我们不同设定,连公平对比表都难搭。
4. **实证证据是负的**(§2.3):−0.138 macro-F1。而且"语言无关检索空间"这个组件替换
   (拿多语言编码器换掉 CLIP 英文文本塔)已经被 Tanvir & Alam(2602.19212)做过了
   —— 他们做了 CLIP+XLM-R+FAISS kNN,却**主动没有把检索池做成跨语言的**。
   我们要做的就是对一个已发表架构改一行。
5. **数据规模。** ZH test n=149,EN test n=161。本项目的噪声地板约定是
   1 acc 点 ≈ 1.6 个视频;跨语言增益要越过多 seed 噪声带需要的效应量在这个规模上几乎不可能干净拿到。
6. **顶会对"新语言/新数据集"的容忍度已经很低。** ACL 主会对纯 cross-lingual application
   论文的门槛已经在 "what's the new mechanism" 上;NeurIPS/ICML/ICLR/CVPR 对这类题材基本不接。

## 2.5 我们资产的匹配度:**很低**

| 资产 | 对轴 2 的价值 |
|---|---|
| MultiHateClip EN+ZH | ✅ 有数据,但 ❌ 是全项目最小最噪的两个库,且 benchmark + 协议 + "EN>ZH" 观察都是原论文的。 |
| LoRA-Qwen ZH 0.826 | ❌ **反向证据**:它把 ZH 变成了强的那一边,拆掉了 motivation。 |
| CLIP 英文文本塔短板 | ❌ 已被 LoRA-Qwen 绕过,不能再当动机。 |
| 跨数据集可更新 kNN memory | ⚠️ EN↔ZH cell 是这个矩阵里**最弱的**几格之一。 |

## 2.6 最快的杀死性实验

**大部分已经跑完并杀掉了**(Phase-3b transfer matrix)。补最后一刀,**$0 纯 CPU,分钟级**:

> **合并-memory 剂量曲线。** 固定 ZH 训练好的头,memory = ZH-train ∪ (ρ × EN-train),
> ρ ∈ {0, 0.25, 0.5, 1.0},只在 **dev** 上扫(test 不碰),5 seed。
> **杀死判据(先冻结):若在任何 ρ > 0 下,ZH dev macro-F1 的 5-seed 均值增益 < +0.014
> (本项目噪声带),则关闭轴 2。**
> 预期:会杀掉。EN-only memory 已经比 ZH-only 低 13.8 点,混入只可能稀释。
> 成本:复用已有的 `src/eval_cross_dataset.py` + 已缓存的特征,零 GPU,零新标注。

---

# 两轴排序

**排序:轴 1 > 轴 2 > (两条都不该投)**

| 排名 | 轴 | 顶会判决 | 一句话理由 |
|---|---|---|---|
| 1 | **轴 1(流式/持续 memory)** | **不够** | 唯一真资产是"hateful video 上传时间切分"这个领域真空,但**整条机制在别的域已被一字不差写过**:Mireshghallah et al.(EMNLP 2023)= 冻结编码器 + kNN 记忆 + 未来时间片 + 打赢重训基线;AdaNPC(ICML 2023)= 插回记忆的零梯度适应带理论;而 2305.13034(EMNLP 2023)证明"零梯度"根本不是机制上的不同。领域内还有 Class-RAG 占位。**加上我们自己的 W4 实验已经把它证伪**(memory 插入 flat-to-negative)。**唯一残值:一篇诚实的 negative-result / analysis 短文,非主会方法论文。** |
| 2 | **轴 2(跨语言检索)** | **不够,更糟** | Motivation 被我们自己的主表反驳(ZH 0.826 > EN 0.743),连 benchmark 原论文的二分类口径也只有 0.79 vs 0.78;机制的三个部件分别被 Ghorbanpour et al.(EMNLP 2025 主会)、Stap & Monz(EMNLP 2023)、MoRE(WWW 2025)+ CRAVE(ICCV 2025)占满;SCANNER(AAAI 2026)已发表同 benchmark 同语言对的迁移数字;我们的 Phase-3b 实测 EN→ZH **−0.138 macro-F1**。**零残值。** |

## 给主对话的建议(超出任务范围但必须说)

这两条轴都是**已经被本项目自己的实验关掉的方向的重新包装**
(`ideas/evolving-memory-protocol.md` 已标 "validated-as-calibration";
`gap_map.md` G3 已标 "cross-lingual sub-case 真实但弱")。
再往这两条轴投入 GPU 之前,建议先做上面两个 **$0 CPU 杀死性实验**(oracle-insertion ceiling
与合并-memory 剂量曲线)把它们**正式钉死**,然后把弹药转向本文件没有覆盖的方向。

**仍然空着的位子见附录 C**(四个,每个都附了"为什么仍不够"的理由)。**没有一个是本次两条轴的救援方案。**

一条与本次两条轴无关、但本次普查顺带确认的事实:**MoRE 的 retriever 是冻结的 weighted-cosine
启发式,监督全是 BCE,其 "contrastive" 的 BHAN 实际上是 attention 而非对比目标** ——
即"**学出来的、对比训练的 hateful video 检索空间**"在文献上仍无人发表。
这正是本项目 G1/G2 早就锁定的主线(`gap_map.md`),**是既有主线的再确认,不是新发现,
也不是这两条轴的替代品。**

---

## 附录 A — 除 MultiHateClip 外的多语言 / 非英语仇恨视频基准(轴 2 的"换数据集"退路评估)

| 基准 | 出处 | 内容 | 对我们有用吗 |
|---|---|---|---|
| **ADIMA: Abuse Detection In Multilingual Audio** | arXiv **2202.07991**, **ICASSP 2022** | 11,775 clip / 65h / **10 种印度语言**,**自带单语 + 零样本跨语言 baseline** | 唯一"真·多语言 + 有跨语言协议"的滥用检测基准,但**纯音频**。而且 2412.01408 与 2604.09094 已经在它上面做过跨语言 few-shot(且后者结论是"每语言加几条标注收益很小")。 |
| **MuSeD**(西班牙语性别歧视视频) | arXiv **2504.11169** | ~11h TikTok+BitChute,西班牙语,文本/音频/视觉分阶段标注 | 单语。 |
| **CH-SV**(中文有害短视频) | **ACM MM 2025**, DOI 10.1145/3746027.3758279 | 6,728 视频,6 类危害 | 纯中文单语。 |
| **NCSV** | Springer 2025 | 中文负面短视频 + 社交上下文 | 纯中文单语。 |
| Multimodal Hate Content Video Detection in Regional Languages | Springer ICCCN 2025 | 印度地方语言,MuRIL/XLM-R/IndicBERT | 低可见度会议。 |
| HateMM / HateClipSeg / ImpliHateVid / DeHate / HarmVideoBench | — | 全部英文 | 无跨语言可能。 |

⇒ **没有"换个更大的多语言视频基准"的退路。** 唯一带跨语言协议的是纯音频的 ADIMA,
而它上面已经有两篇跨语言 few-shot 工作。

---

## 附录 B — **绝对不要主张新意的清单**(轴 1,均已查到占位者)

写任何相关文字之前先读这一条。下面每一项都有上面列出的具体引用:

1. 增长或替换 datastore 以零梯度做域适应 —— kNN-LM(ICLR 2020)/ kNN-MT(ICLR 2021)。
2. 把模型自己的测试时预测插回记忆 —— AdaNPC(ICML 2023)、TDA(CVPR 2024)。
3. 按年龄 / staleness 打分淘汰记忆 —— RoTTA(CVPR 2023)、Lu et al.(AIJ 2016)。
4. 置信度 / 熵门控的 cache 准入 —— CRG / ACE / DOTA / SCA(2025)。
5. 指出伪标签错误会累积 —— 上同,已命名为 "cache noise" / "error accumulation within the cache"。
6. wave 式记忆插入 + 在后续时间片上评估(含 train×test 桶迁移矩阵、"更新 vs 不更新 datastore"消融)
   —— **Mireshghallah et al.(EMNLP 2023)一篇全包**。
7. "non-parametric continual learning" 这个提法 —— HippoRAG 2(**ICML 2025**, PMLR v267:21497)。
8. "零梯度 vs 梯度是两种本质不同的机制" —— 被 2305.13034(EMNLP 2023)证伪。
9. 靠替换 memory 内容适应新类别/新任务 —— Memory-Modular Classification(**TMLR**, 2504.06021)。
10. 检索库被对抗性污染 —— PoisonedRAG(**USENIX Security 2025**)、AgentPoison(**NeurIPS 2024**)、
    corpus poisoning(**EMNLP 2023**)、RTTDP(**ICLR 2025**)。
11. datastore 压缩 / 剪枝 —— Efficient kNN-LM(EMNLP 2021)、Cluster-Based kNN-MT(ACL 2022)。

## 附录 C — 本次普查中**仍未找到占位者**的几个位子(仅记录,不构成推荐)

诚实记录:下面这些没找到直接占位者,但每一条都附了它为什么**仍然不足以支撑本项目投顶会**的理由。

| 位子 | 状态 | 为什么仍不够 |
|---|---|---|
| **纯非梯度检索记忆的长时程崩溃诊断 / 理论 / 检测器** | 真空(梯度侧的对应物是 **PeTTA, 2311.18193, NeurIPS 2024**;非梯度侧只有一篇未评审单作者预印本 2607.21673) | 这是四条里最像顶会贡献的,但**必须是理论或诊断协议**;而且它与仇恨视频无关 —— 我们在这条上没有任何资产优势,是纯 ML 理论题。 |
| **刻画"记忆插入何时不再吸收漂移、必须刷新编码器"** —— 冻结编码器 + 漂移 datastore + **自插入伪标签**这个三元组 | 边缘真空:GradNormIR(ACL 2025 F)覆盖"冻结编码器 + 漂移语料"但语料是**干净**的;cache-TTA 一族覆盖自插入伪标签但**不看编码器空间退化**。交集确实很薄 | 仍是**分析性贡献**,不是机制;位子窄。 |
| **staleness 淘汰 + 自污染的联合预算化处理** | 半真空:现有方法各打一个轴且用手调线性打分(**RoTTA 直接令 λ_t = λ_u = 1.0**) | 审稿人会读成 RoTTA + TDA 的增量。**低到中等**接受概率。 |
| **在 TiC 式前向/后向迁移矩阵里补一条检索记忆基线** | 真空(**已核实** TiC-CLIP 的续训基线全是梯度式:warm-start / replay / distillation / patching / cyclic-cosine;TiC-LM 同) | **benchmark 贡献,零机制新意**,D&B track 封顶;且 Mireshghallah et al. 已在另一模态做过半参数版本。 |
| **插入"什么"而非"是否插入"**(分型 / 逐模态 / 逐片段的子样本证据插入) | 未找到占位者 | 最弱的一条;审稿人会读成架构变体,除非附上"整样本插入在原理上做不到"的结果。**而且本项目的"段级检索 key""类型硬分区 memory"已经是死机制。** |
| **hateful video 按上传时间的 train/test 切分** | 领域内真空 | 见 §1.3 第 9–10 条:n≈150、只有 MHClip 可定年、存活偏差 —— 规模上不去。 |

---

## 方法与可信度声明

- 检索途径:WebSearch、arXiv API 全库检索(`all:"hate" AND all:"video"`,按提交日期倒序 60 条)、
  arXiv HTML 全文抓取(SCANNER / Class-RAG 逐句核对)、本地 `research-wiki/`。
- 所有外部论文都给了 arXiv 号或 venue+year;凡未能独立核实的,**均在下方"引用前必须自查"里显式标注**。
- 所有"我方数字"都来自 `research-wiki/` 的已冻结记录并标注了出处文件,**没有重跑、没有触碰测试集**。
- 已知盲点:(1) 非英文文献(中文期刊 / CCF 中文会议)未检索;(2) 2026-08 之后的预印本未覆盖;
  (3) 闭源工业系统(平台内部的 moderation memory)不可见,但这不影响顶会 novelty 判定。
### 引用前必须自查的条目(诚实标注)

**ID 未独立核实**(论文确实存在,但 arXiv 号是从检索片段读来的,未逐个抓取核对):
StreamingQA(2205.11388?)、Atlas(2208.03299?)、Wild-Time(2211.14238?)、
TemporalWiki、CLOC(2108.09020?,中等置信)、TiC-LM(2504.02107?)、FoMo-in-Flux(2408.14471?)、
TESSERACT、Lazaridou "Mind the Gap"、RetoMaton / kNN-Prompt / kNN-Adapter、
T3A / GDumb / FCT / SDC 的 arXiv 号,FastFill 的 venue。
`2604.07027` 的 venue 未确认(疑 ICLR 2026 TSALM workshop)。
`2211.07828` 的 venue 未确认。

**已核实的更正:**
- `2305.09253`(Online CL Without the Storage Constraint)**无正式发表 venue** ——
  从 ICLR 2024 撤稿(OpenReview `E83OzFbNQ6`),按技术报告引用。
- `2608.02738` 已于 2026-08-06 撤稿,**不可引用**。
- "Auto-Intoxication Feedback Loop"(仅见 ResearchGate),**不可引用**。
- 附录 C 第 1 条提到的 `2607.21673` 是**未经评审的单作者预印本**,不可当既有工作压自己。

**核实为不存在**(不要凭印象引):题为 "Rethinking kNN-MT" 的论文;题为
"Retrieval-augmented models are continual learners" 或
"Continual learning with retrieval augmentation" 的论文。

### 覆盖度的诚实边界
轴 1 的通用机制普查由并行的定向 sweep 完成,其自述:部分子线程未按时返回,
canonical 论文已由主线独立复核,但标注为"仅检索片段"的条目未抓取原文;
且 WebSearch 预算(200 次/会话)在收尾前已用尽,限制了进一步核验。
**本节列出的每一条不确定性都是真实的,不要在写作时把它们当成已确认事实。**
