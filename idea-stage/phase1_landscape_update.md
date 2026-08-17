# Phase 1 — 文献景观更新(2026-08-09)

> **性质:** idea-discovery(composed 模式)Phase 1 的工作笔记。后续折叠进 `idea-stage/IDEA_REPORT.md`。
> **成本:** 纯文献,零 GPU,零 test 触碰。
> **方向:** hateful video detection 的方法论文机制(RGCL 检索管线为基座,允许大偏离)。
> **目标档次:** NeurIPS / ICML / ICLR / CVPR / ACL 主会。novelty 必须机制级。
>
> **本文件的立场:诚实优先。** 找不到够顶会的开放位就直说。第 3 节的最终判决是:
> **本轮扫新没有找到任何一个"现成可写、机制级、够 NeurIPS/CVPR 主会"的位子。**
> 有两个位子(#2 / #3)**有可能**成为这样的位子,但都卡在一个我们还没做的数据可用性核查上;
> 其余五个位子的诚实上限是 ACL 主会评估类 / D&B / ACM MM / ICMR 档。

---

## 0. 方法与可信度声明

**检索途径(本轮):**
- arXiv Atom API 全库检索(**必须走 `https://export.arxiv.org`**;`http://` 返回空),18 条查询,
  按 `submittedDate` 倒序,窗口 2025-11 → 2026-08。
- 两个并行 subagent(Opus 5):一条扫"检索增强分类 / 对比检索机制"(域外),
  一条扫"仇恨/有害视频 + 视频审核"(域内)。它们额外用了 Semantic Scholar / OpenAlex /
  Crossref / ACL Anthology API + 定向 WebFetch。
- 本地 `research-wiki/`(37 篇论文笔记、`gap_map.md`、`PAPER_MASTER_TABLES.md`、
  `NOVELTY_RECON_2026-08-09.md`、`idea-stage/IDEA_REPORT.md`)。

**已知盲点(全部诚实标注,不要在写作时忘记):**
1. **WebSearch 全会话额度(200)在本轮早期即耗尽**,两个 subagent 与主线均无法用网页搜索。
   会议"接收论文列表"页面(未被 DBLP/OpenAlex 索引的)因此**没有被扫到**。
   ACM MM 2026(11 月,Rio)尚无接收列表;ICME 2026 / INTERSPEECH 2026 / ECCV 2026 / EACL 2026
   尚未进 DBLP/OpenAlex,**不能声称这些会上没有竞品**。
2. arXiv API 中途 HTTP 429 限流,`abs:"hateful"` 那一路 sweep 未返回。用 S2/OpenAlex 补了,
   但可能漏掉少量 arXiv-only 预印本。
3. 四篇最有威胁的新论文(SAGE 除外的 MATCH / HCG-MPB / TIHD / ACE-HVD)**正文全部付费墙**
   (IEEE Xplore / ACM DL 403),摘要只写 "state-of-the-art" 不给数字。
   **HCG-MPB 与 TIHD 在同一个 ICMR 2026 proceedings 里对重叠 benchmark 各自声称 SOTA,
   两者不可能都对** —— 这需要拿到 PDF 才能解决。
4. `papers-with-code` 已下线(重定向到 HuggingFace),**没有 leaderboard 可交叉核对**任何数字。
5. 非英文文献(中文期刊 / CCF 中文会议)未检索。

**核实等级标注约定(全文使用):**
- **[A]** = 我本人通过 arXiv API 拿到标题+日期+摘要,ID 可信。
- **[B]** = subagent 通过 Crossref/S2/OpenAlex/ACL Anthology 的 DOI 元数据核实,我未独立复核。
- **[C]** = venue 未核实 / 数字取自摘要或仓库,**引用前必须自查**。

---

## 1. 增量扫新(2025-12 → 2026-08)—— 变了什么

上一轮(`IDEA_REPORT` §1,2026-08-08)的景观图现在有**四处实质性变化**。

### 1.1 HateMM 的性能前沿易主,而且新占位者的机制正对着我们

**SAGE — Synergistic Adaptive Gating of Experts for Hateful Video Detection**,
ACL **2026 主会 Long**(`10.18653/v1/2026.acl-long.817`, pp. 17950–17966,2026-07)**[B]**。
无 arXiv。

- **机制:** 放弃特征级融合,改**决策级仲裁** —— 每模态一个专家模型保留模态特有语义,
  一个 global expert 处理上下文,一个 instance-level "tribunal" 逐样本权衡各模态证据强度。
  明确对着 **feature dilution**(占主导的良性模态淹没稀疏局部的仇恨线索)这个问题写的。
- **数字:** HateMM **0.8710 acc / 0.8628 macro-F1**;MHC-EN 0.8375/0.7962;MHC-ZH 0.7901/0.7484。
- **对我们的意义(两条,方向相反):**
  1. **旧的"MM-HSD 0.874 天花板"叙事需要改写。** SAGE 自己复现的 MM-HSD 只有 0.8203/0.8054,
     与 MM-HSD 自报的 0.878/0.874 差 5 个点。**两个数字属于不同协议**
     (MM-HSD 是自报 5-fold CV)。写作时必须挑一个并声明,不能混。
  2. **我方 frozen-Qwen 的 HateMM 0.870/0.861(单 seed)与 SAGE 的 0.8710/0.8628 实质持平。**
     这不是好消息也不是坏消息 —— 它说明 HateMM 的 video-level accuracy 赛道已经挤到
     "各家在同一个小数点后第三位互咬"的阶段,**再靠 accuracy 讲故事的空间已经关闭**。
  3. SAGE 的 framing("feature dilution")与 TERA Gate-0 的 union 路标(83.6% 短时局部/跨模态)
     指向同一个现象,而 SAGE 用**决策级仲裁**而不是**片段选择**去解 —— 这独立佐证了
     Gate-0 判 NO-GO-C 之后"单段选择"路线的价值下降。

### 1.2 有人在顶会级 proceedings 里公开论证"实例检索是错的设计"

**HCG-MPB — Hierarchical Complementary Gating Mechanism with Multimodal Pattern Bank
for Hateful Video Detection**,Sun / Yang / Yin / Wei / Wu / Chen(新疆大学),
**ICMR 2026**(`10.1145/3805622.3810724`,2026-06-15)**[B]**。无 arXiv。

- **机制:**(a)Hierarchical Complementary Gating —— 文本作语义锚,动态门控**选择性**接纳
  audio-visual 特征(非对称融合,抗模态竞争,抑噪);(b)**Multimodal Pattern Bank** ——
  论文**点名 "instance-based retrieval" 为有缺陷的基线**(语义歧义 + 原始数据的存储/延迟代价),
  用 **LLM 蒸馏出的紧凑可解释 prototype** 取而代之。
- **对我们的意义:这是本轮扫新对本项目最有威胁的一条。** 它同时:
  (i)占掉了"hateful video 的检索记忆"这个位子的**第二个**版本(MoRE 之后),
  (ii)并且**把我们的核心设计(逐实例 kNN memory)写成了它的 motivation 里要反对的东西**。
  今后任何 RGCL-系的 hateful video 论文,related work 第一段必须正面回应它。
- **无法核实:** 数据集只写 "two public datasets"(大概率 HateMM + MultiHateClip),无 headline 数字。

### 1.3 "跨模态不一致 = 隐式仇恨的信号"这个位子被占了

**TIHD / QGC-Net — Query-Guided Conflict Inference and Incongruity-Aware Alignment
for Implicit Hate Speech Detection in Videos**,**ICMR 2026**
(`10.1145/3805622.3810673`,2026-06-15)**[B]**。无 arXiv。
*(勘误:此文常被误引为 ACM MM 2025;Crossref 核实为 ICMR 2026。)*

- **机制:** 命名 "Alignment Trap" —— 对称的、追求一致性的融合把跨模态**不一致**抹平了,
  而不一致**恰恰**是隐式仇恨的信号。以文本为锚,用学出来的 conflict gate **检索并放大**
  矛盾的 audio-visual 特征;Incongruity-Aware Alignment 做可微软对齐扫描(无需帧级监督)。
- **数据集:** ImpliHateVid、HateMM。
- **对我们的意义:** 这条**独立地、在本领域内**占掉了"跨流矛盾 = 证据"这条叙事。
  我们那条已死的 **OCR−ASR 残差**其实是它的一个特例(残差 = 一种矛盾度量);
  TIHD 的存在使得任何"跨模态矛盾"重新包装都变得**更**不可写。
  同族的还有 **MAGIC3**(`2603.14992`,2026-03,短视频假新闻的成对+全局三模态一致性建模)**[A]**。

### 1.4 多智能体证据核验成为本领域的新常规

- **MATCH: Multi-Agentic Evidence Grounding for Explainable Hate Video Detection**,
  IEEE **TCSVT**(`10.1109/TCSVT.2026.3672052`,2026-03-09)**[B]** —— 两个 LMM Proposer 从
  仇恨/非仇恨两个对立角度独立取证,第三个 LMM Verifier 拿视频的时空证据去验。
- **ACE-HVD**,IEEE **ICC 2026**(`10.1109/ICC59461.2026.11587024`)**[B]** —— 同组的会议前身。
- **An Interpretable Agentic Framework for Multimodal Hate Video Analysis with
  Explicit Evidence Attribution**,**WWW 2026 Companion**(`10.1145/3774905.3796488`)**[B]** ——
  把视觉/音频/文本/上下文保持为**独立证据源而非融合特征**,抽取物体、**屏幕文字(OCR)**、
  语音转写、毒性分数、命名实体进统一证据结构。**这是 OCR 通道解禁的独立外部佐证。**
- **UNIVID**(`2606.05748`,**ACL 2026 Industry**)**[A/B]** —— policy-aware caption 作可解释中间表示。
- **IPS: In-Prompt Process Supervision for Short Video Content Moderation**,
  **ACL 2026 Industry**(`10.18653/v1/2026.acl-industry.89`)**[B]**。

### 1.5 其他值得记的新入场者(不改变主图,但改变某些论证的可写性)

| 工作 | 出处 | 为什么重要 |
|---|---|---|
| **Failures to Surface Harmful Contents in Video Large Language Models** | **AAAI 2026**(`10.1609/aaai.v40i42.40841`)**[B]** | 五个 SOTA VideoLLM 的**遗漏率 >90%**;归因于**稀疏均匀帧采样** + 激进 visual-token 下采样 + encoder↔decoder 耦合弱。**对本项目是承重的效度威胁**:我们全线用 8 帧 / K=30 均匀采样。 |
| **HarmVideoBench** | `2606.27187`(2026-06)**[A]** | 1,379 视频 / 4,137 MCQ,三层(可观察证据 / 片内含义 / 片外推理);自带 **BCR** —— 预测推理边界、**仅在需要时才检索上下文**。 |
| **PaSBench-Video** | `2606.02443`(2026-06)**[A/C]** | 740 视频,**帧级风险起始点 + 事故边界**,因果流式协议,**显式测 false positive**;最严指标上无模型超过 20%。近期最干净的"起始点标注 + FP 受控"视频协议范例。 |
| **Beyond Hate: Differentiating Uncivil and Intolerant Speech** | `2603.22985`(2026-03,under review)**[A/C]** | 把 2,030 条 Hateful Memes 沿**两条可分离轴**重标:**incivility(语气)vs intolerance(内容)**。联合粗+细训练把 **FNR−FPR gap 减半**(0.74→0.42 LLaVA-1.6;0.54→0.28 Qwen2.5-VL)。**"拆标签而不是拆模型"的最佳近期范例 —— 也是我下面 #7 位子的最强反驳者。** |
| **UniSafe: Modality-Agnostic Hateful Content Detection via Shared-Space Projection** | **WWW 2026 Companion**(`10.1145/3774905.3795455`)**[B]** | 冻结编码器 + 轻投影进共享安全空间,顺序无关聚合 → **任意非空模态子集免重训推理**;消融说 **modality dropout 是鲁棒性主因**。**与我们已死的 silence-route / 缺模态填补重叠。** |
| **SenBen: Sensitive Scene Graphs for Explainable Content Moderation** | `2604.08819`,**CVPRW 2026**(v2 2026-05)**[A/B]** | 13,999 帧 / 157 电影的 Visual-Genome 式场景图(25 物体 / 28 属性含 pain/fear/aggression / 14 谓词 / 16 敏感度标签)。**modality-attribution 的空间接地版本。** |
| **CH-SV: A Benchmark for Multi-Type Chinese Harmful Short Video Detection** | **ACM MM 2025**(`10.1145/3746027.3758279`,2025-10)**[B]** | 6,728 视频 / 6 类。窗口外但**不在我们旧图上**,应补进语料。 |
| **PHTV-Scout / When Youth Enter the Algorithmic Wild** | `2605.23598`(2026-05)**[A/C]** | 186,727 条抖音/快手视频 + 51,287 条评论,6 个月;**有害内容占比 6.11%**;记录规避手法。 |
| **Old Tricks, New Models** | `2607.28187`(2026-07)**[A]** | 七种平凡图像变换(**连灰度化和反色都算**)在三家商用审核 API 上把 unsafe 翻成 safe;多模态内容与自伤类最脆弱。 |
| **TraRA: Trajectory-level Recognition Aggregation for Video Text Spotting** | `2606.07161`,IEEE AVSS 2026 **[A/B]** | 非审核领域,但是与我们 OCR cache 直接相关的工程结论:**逐帧 OCR 在模糊/遮挡/尺度变化下不稳定**,需按整条文字轨迹做时序聚类 + 聚合。 |
| **SIGNPOST-Bench** `2608.04244` **[A]** · **When Text Hijacks Vision** `2604.17375` **[A]** | 2026-08 / 2026-04 | 屏幕文字与画面**冲突**时 MLLM 的行为:系统性偏向覆盖文字的语义而幻觉。**正是"仇恨字幕盖在良性画面上"这个失败模式。** |
| **IARE** → **SIGIR 2026**(`10.1145/3805712.3809637`)**[B]**;**TANDEM** `2601.11178` v3 → **AAAI-ICWSM 2027** **[B]**;**MARS** → ICASSP 2026;**MultiHateLoc** → WWW 2026(`10.1145/3774904.3793032`);**LEAF** → ACL **Findings** 2026 | — | venue 更新,写作时用这些。 |
| ⚠️ **SafeLens 是两篇不同的论文** | (a)*SafeLens: Segment-Level Hate Speech Detection in Online Videos*,**AAAI 2026 demo/system**(`10.1609/aaai.v40i48.42390`)**[B]**;(b)*SafeLens: Deliberate and Efficient Video Guardrails*,`2605.17610` **[A]** | 同名同年、互不相关。`DRAFT_intro_related_limitations.md` 目前引的是 (b),**需核对是否引错**。 |
| ⚠️ **RASR** `2604.06687` **[A]** | v2 作者注:"The paper needs revision, and the experiments need to be expanded" | 唯一一篇真正做"检索增强的视频分类"(假新闻视频)的论文**自我作废了实验**。不要当占位者引,也不要当支持证据引。 |
| ⚠️ **The Ghost Annotator** `2606.02911` **[A]** | 作者**明确撤稿** | 不可引用。 |

### 1.6 域外机制:两条"framing 杀手",必须记住

> 这两条不是机会,是**约束**。任何未来的检索机制论文如果不先绕开它们,会被当场打回。

1. **Context-Adaptive Inference: A Unified Statistical and Foundation-Model View**,
   `2607.23304`(2026-07,90 页,"living version",**venue 未核实 [C]**)**[A]** ——
   证明在**平方损失 + 线性预测头 + 固定特征**下,**显式参数适应与隐式路由都等价于
   在 (input, context) 联合特征上的 kernel ridge regression**。
   ⇒ **"我们的检索模块是一种适应"这个概念性贡献已经被形式化地收编。**
   (与 `NOVELTY_RECON` 附录 B 第 8 条 `2305.13034` 同族,但更强、更一般。)
2. **RAG without Forgetting: Continual Query-Infused Key Memory (ERM)**,`2602.05152`
   (2026-02,**venue 未核实 [C]**)**[A]** —— 证明在标准相似度下 **query expansion ≡ key expansion**。
   ⇒ **"我们改进了 query/key 的构造方式"这一类主张的新意空间被形式化地压缩了。**
   这条直接打在"检索键该是什么"这个我们最爱的位子上。
3. 附带一条 confound 杀手:**The Structural Attention Tax**,`2606.11198` **[A/C]** ——
   检索内容的**格式**独立于相关性地扭曲注意力(demonstration attention 最多被压缩 42%)。
   任何"检索有用是因为检索到的内容相关"的因果主张都需要控制这一项。

---

## 2. 重画的占位地图 —— 按机制位置组织

约定:**"域内"= hateful / harmful video;"域外"= 其他领域(搬运需超额贡献)。**
「剩什么缝」一栏只写**我判断仍然空着**的东西,并附我为什么这么判断。

### 2.1 位置 A — 检索 / 记忆

| 谁占着 | 出处 | 占到什么程度 |
|---|---|---|
| **MoRE** | WWW 2025 | 域内唯一发表的检索增强方法。整视频键、**冻结**的 weighted-cosine 检索器、全 BCE 监督、trained-MoE 决策头。占掉"hateful video 有检索"这句话,**没有**占掉"学出来的检索空间"。 |
| **HCG-MPB** | ICMR 2026 **[B]** | **新(1.2)**。用 LLM 蒸馏 prototype **取代**逐实例检索,并**论证实例检索是错的**。占掉"hateful video 的压缩式记忆"整块地,并把我们的设计变成了它的靶子。 |
| **CRAVE** | ICCV 2025 | 跨域检索增强**训练**(从资源丰富的 image-text 域借邻居救数据稀缺的恶意视频检测)。占掉"从更富的池子借证据"。**注意它是重训,不是零重训换库。** |
| **Class-RAG** | `2410.14881`(Meta) | 可热更新检索库 + "semantic hotfixing" —— 占掉"换库即适应新出现的 harm"。 |
| **Now You See the Hate** | `2607.19061` **[A]** | 图像域(HatefulIllusion)。complementary view bank 上的 retrieve-and-calibrate;93.2% balanced acc vs 审核分类器 ≤24.5%。占掉"检索多视角再校准"。 |
| **Adaptive View Retrieval / AHA-Memes / Tanvir&Alam(孟加拉 meme,CLIP+XLM-R+FAISS)** | 见 `NOVELTY_RECON` §2.2 | meme 域的检索式仇恨检测已相当拥挤。 |
| 域外 · **学出来的检索器 + 分类器端到端** | `2604.07027`(ICLR 2026 **workshop**)**[A/B]**、TabSieve `2602.11700` **[C]** | 机制**作为技术已被证明可行**(score-based 梯度估计器),但只在非平稳时序 / 表格上,且是 workshop 档。**在 video / multimodal 分类上没有主会占位者。** |
| 域外 · **检索键该是什么** | LaPR `2604.03657`(CVPR 2026 **[A/B]**,image-label 联合键 + query-adaptive MoE 路由);CIRCLES `2603.16737`(CVPR 2026 **[A/B]**,属性分解键);ERM `2602.05152`(键侧持续更新 + **query≡key 等价定理**);`2606.05931`(INTERSPEECH 2026,逐 query 主动模态检测) | 2026 上半年这块被**大量**填充。**所有这些键都由"存在的信号"构造。** |
| 域外 · **邻域共识去噪** | AAAI-26 `2512.24064`、ICML-26 IN2R `2606.04061`、CVPR-26 ConeSep `2604.20358` **[A/B]** | **作为机制主张已经关闭**(八个月内落了三个顶会)。可以当工具用,不能当新意。**但全部是 noisy correspondence(配对错误,存在唯一正确答案),不是标注主观性。** |
| 域外 · **代价感知获取 / 检索要不要做** | ICLR-26 `2601.22570`(检索式 selective prediction)、ACL-26 `2605.13277`(information-gain 证据效用,带证明)、VOILA `2602.03007`、`2607.05438`、`2606.29959`、`2606.11907`(ECML 2026)… 12+ 篇 | **本轮扫新中最拥挤的一块。** 这独立地**确认了 CVoI 方向被杀是正确的**:该位子在 2026 上半年被 ICLR/ACL 级工作填满,且全部以**算力/延迟/token 为代价轴**。 |

**剩什么缝(检索/记忆):**
1. **对"标注不一致 / 争议标签"鲁棒的检索。** 两条独立检索(arXiv + Semantic Scholar)在窗口内
   返回**零**结果。现有 noisy-retrieval 文献的机制前提是"存在唯一正确配对,邻域几何可恢复它";
   **主观分歧没有可恢复的正确答案,这个前提不迁移。** 见 §3 #2。
   ⚠️ 这是**从检索静默推断的**,且我的查询次数被限流压缩过。
2. **端到端可微检索器 + 分类器,在视频/多模态上。** 技术已占(workshop 档),应用未占。
   但这只是"occupied-as-technique, unoccupied-as-application" —— **单靠它不构成机制新意。**
3. **残差键 / "缺了什么"作为键。** 形式上仍空,但:(a)我们自己的外审已判 `r = o − a` 是
   `[I, −I]` 的固定线性投影(2/10);(b)ERM 的 query≡key 等价定理进一步压缩了纯键重参数化的
   新意空间;(c)TIHD 在域内占掉了"矛盾即证据"的叙事。**综合判定:这条已死,不要复活。**

### 2.2 位置 B — 时序

| 谁占着 | 出处 | 程度 |
|---|---|---|
| **MultiHateLoc** | WWW 2026 / `2512.10408` | modality-aware **hard top-K MIL**,离散、非注意力的段选择。HateMM 帧级 mAP 0.645 / AUC 0.799;MHC 0.445 / 0.750。**域内定位的主占位者。** |
| **LELA** | `2602.09637` | training-free、五模态(图像/语音/**OCR**/音乐/上下文)、逐帧 prompt LLM。⚠️ **subagent 核实:尽管标题写 localisation,它报告的表全是 video-level**(HateMM 71.48/70.43),**没有在 HateClipSeg 或 MultiHateLoc 协议上评过**。 |
| **TANDEM** | `2601.11178` v3 → AAAI-ICWSM 2027 **[B]** | RL 出时间戳 + target identity;HateMM 目标识别 0.73 F1;**无 IoU/mAP**。 |
| **HateClipSeg** | ACM MM 2025 / `2508.01712` | 段级标注数据 + 三任务 baseline。定位随 tIoU 急剧退化(42.22@0.3 → 18.34@0.7),**且 V+T+A 在每个 tIoU 上都比 visual-only 差**。 |
| **MultiHateGNN** | BMVC 2025 | 软注意力加权段聚合 —— 这是"任何新时序机制必须是 hard/discrete 才可发"的原因。 |
| **时间标签噪声(Revealing Temporal Label Noise)** | `2508.04900`(MUWS@MM 2025) | oracle-trim 后 98.64 vs 粗标 79.30 macro-F1。**这是天花板估计,不是检测结果。** |
| **PaSBench-Video** | `2606.02443` **[A/C]** | 域外(安全隐患),但占掉了"帧级起始点 + FP 受控的流式协议"这个协议形状。 |

**剩什么缝(时序):基本没有。**
- 本项目自己的死方向清单里,**多段互补**(Gate-0 NO-GO-C,6/73=8.2%)、**单段选择**、
  **段级检索键**(`gap_map` G4 UPDATE:sign-flips by language)、**视觉纯度选段**(P2 forensic 的 within-video AUROC 仅 0.511)
  已经全部实测关闭。
- 外部这一块由 MultiHateLoc(离散 top-K)+ MultiHateGNN(软注意力)+ LELA(training-free)
  三点夹住,剩下的形状要么是它们的插值,要么是我们已经证伪的。
- **唯一没被占的是"评估侧"**(见 2.7),不是建模侧。

### 2.3 位置 C — 模态融合

| 谁占着 | 出处 | 程度 |
|---|---|---|
| **SAGE** | ACL 2026 主会 **[B]** | **决策级专家仲裁 + instance-level tribunal**,对着 feature dilution。HateMM 0.8710/0.8628。**新前沿。** |
| **HCG-MPB** | ICMR 2026 **[B]** | 文本作锚 + 层级互补门控的**非对称**融合(抗模态竞争)。 |
| **TIHD / QGC-Net** | ICMR 2026 **[B]** | **不一致放大**(Alignment Trap)。占掉"跨模态矛盾即信号"。 |
| **MM-HSD** | `2508.20546` | OCR 作 cross-modal attention 的 **query**,always-on。自报 HateMM 0.878/0.874(5-fold CV)。 |
| **UniSafe** | WWW 2026 Companion **[B]** | 冻结编码器 + 共享安全空间投影 + 顺序无关聚合 → 任意模态子集免重训;**modality dropout 是主因**。 |
| **RAMF / MARS / HVGuard / MATCH / ACE-HVD** | TMLR / ICASSP26 / EMNLP25 / TCSVT / ICC26 | 推理式 / 多智能体融合,已饱和。 |
| **MAGIC3** | `2603.14992` **[A]** | 短视频假新闻的成对+全局三模态一致性 + 不确定性感知的选择性 VLM 路由。 |

**剩什么缝(融合):**
- **只剩"屏幕文字的来源分型"这一条,而且很窄。** 见 §3 #7。所有现有工作把 OCR 当**一个**
  无差别的 "on-screen text" 模态(MM-HSD 作 query;CLaMR 作独立索引流;LELA 作五分之一;
  WWW26 agentic 作证据源之一)。**没有人区分"上传者叠加的文字"(可归因于作者的言语行为)
  与"画面里本来就有的文字"(被拍摄的内容)。** 但这条的先验风险很高(见 §3)。
- 我们已死的 **silence-route / 跨塔填补**现在还被 UniSafe 从另一侧占了(modality dropout)。

### 2.4 位置 D — 监督信号

| 谁占着 | 出处 | 程度 |
|---|---|---|
| **LEAF** | ACL Findings 2026 | LMM teacher → 推理期无 LMM 的 student(蒸馏**解释**)。MHC-ZH 81.41/77.14 **[B]**。 |
| **DeHate** | ACM MM 2025 | 人工**段级 contributing-modality** 标注。占掉"证据类型监督"。 |
| **IARE / Ex-HateMM / Ex-ImpliHateVid** | SIGIR 2026 **[B]** | 细粒度有害元素 + gold rationale 标注。 |
| **SenBen** | CVPRW 2026 **[A/B]** | 敏感场景图 —— 空间接地的 who/what/where 取代二值标签。 |
| **IPS** | ACL 2026 Industry **[B]** | 在 prompt 里注入顺序辅助推理问题作为过程监督;用模型生成的标注而非人工。 |
| **Beyond Hate**(incivility vs intolerance) | `2603.22985` **[A/C]** | **把标签本身拆成两条可分离轴**,联合粗+细训练把 FNR−FPR gap 减半。 |
| 域外 · **learning-with-disagreement 整个纲领** | LeWiDi-2025;DiADEM `2604.08425`;EDO `2607.08493`;soft-label `2511.14117`;`2605.24773`;Socio-Contrastive `2604.18069`;RGPO `2607.20515`;STABLEVAL `2605.02122`;NEC `2605.03135` … **[A/C]** | **本轮扫新中"域外最拥挤"的一块。** 软标签、annotator 建模、demographic 残差、可靠性加权 DPO、分歧损失 —— 全部已发表。**全部是文本,零篇视频。** |
| 域内 · 分歧诊断 | `2606.28772`(HateXplain,42.6% 分歧集中在 hate/offensive 边界);`2604.16654`(reclaimed slur);MultiPRIDE@EVALITA 2026(`2606.01298`/`2602.12818`) **[A]** | **诊断已充分,方法在视频上为零。** |

**剩什么缝(监督信号):**
- **"标注分歧作为视频仇恨检测的一等监督信号"整块空着。** subagent 两条独立检索都确认:
  learning-with-disagreement 的方法侧**没有一篇是视频**。
- **更窄也更值钱的一条:把"买标注"而不是"买算力"作为获取式框架的代价轴。**
  §2.1 的代价感知获取文献(12+ 篇,ICLR/ACL/ECML)**清一色**把代价定义为算力/延迟/token,
  收益定义为"对固定 gold label 的准确率"。**没有一篇把被购买的资源定义为标注、
  把收益定义为分歧下降。** 见 §3 #3。
- ⚠️ **两条缝都卡在同一个我们还没做的核查上:我们的视频数据集到底有没有 per-annotator 标签?**
  已知的:HateClipSeg 三阶段标注、Krippendorff α = 0.817(**只公布了聚合值,是否释出逐标注者
  标签未核实**);MultiHateClip 是 3 类(Hateful / Offensive / Normal)+ 段时间戳 + target +
  contributing modality;**我们自己的二分类协议正是把 offensive 折进 hateful** —— 也就是说
  **我们的标签边界恰好落在 `2606.28772` 量到的"42.6% 分歧集中处"**。这给了一个不依赖
  per-annotator 标签的**代理**:3 类标签里的 Offensive 类 = 争议带。**这个代理是否够用,
  是 #2/#3 能不能立项的唯一门。**

### 2.5 位置 E — 训练目标

| 谁占着 | 出处 | 程度 |
|---|---|---|
| **ImpliHateVid / TCL** | ACL 2025 | 两阶段 SupCon(**按类标签**选正负)。ImpliHateVid 87.53/87.73 二分类 **[B]**。 |
| **MultiHateLoc** | WWW 2026 | 帧级跨模态对比(**同视频同时间戳**为正)。 |
| **IARE** | SIGIR 2026 | DPO 偏好对比(正确 vs 故意错误的 rationale 路径)。 |
| **SCANNER** | AAAI 2026 | centroid alignment 式对比(source-free TTA)。 |
| 域外 · hard negative | `2606.01304`(KDD 2026,**When Hard Negatives Hurt**)**[A/B]**;`2603.25722`;`2604.13313` | hard-negative 合成/挖掘的失效条件已被系统研究。 |

**剩什么缝(训练目标):**
- `gap_map` G1/G2 的判断**在本轮扫新后仍然成立**:域内的对比目标全部按
  **类标签**或**同时间戳**选正负,**没有一篇按"检索到的最近邻标注样本"选**。
  这仍然是 RGCL 移植的唯一真正的机制立足点。
- **但它单独不够顶会** —— 这在 `RESEARCH_BRIEF` 里已经写明("bare port has NO methodological
  novelty by itself"),本轮没有任何证据改变它。而且 §1.6 的 `2607.23304` 现在还额外压了一道:
  "我们的检索是一种适应"这个概念叙事已被形式化收编。
- **真正剩下的是"用什么定义正负对"。** 我们已死的:段级键、类型分区、跨语言、残差。
  仍未被占的一个具体形状:**用检索挖出来的"自然发生的最小对"**(同一段素材的仇恨原贴
  vs 报道/反言论转贴)。见 §3 #6 —— 但它**必须先过一个 $0 的存在性测量**。

### 2.6 位置 F — 推理策略

| 谁占着 | 出处 | 程度 |
|---|---|---|
| **SCANNER** | AAAI 2026 | source-free TTA(centroid 对齐 + 样本级自适应)。占掉"漂移下的推理期适应"。 |
| **MARS / HVGuard / LELA / MATCH** | ICASSP26 / EMNLP25 / — / TCSVT | training-free 推理、CoT、多智能体核验。**饱和。** |
| **SafeLens (b)** `2605.17610`、**StreamSense** `2601.22738`、**Filter-And-Refine** `2507.17204`、**ResponseGuard** `2607.21401` | — | fast-slow / 级联 / 选择性路由。⚠️ **ResponseGuard 是反向结果**:一个**不做 CoT** 的 2B guard 在 ~150× 更低时延下打赢 3B 推理型 guard,并指出推理型 guard 几乎不把 verdict 注意力放到图像上,残余差距归因于**冻结的视觉编码器**而非缺推理。 |
| **HarmVideoBench / BCR** | `2606.27187` **[A]** | **预测推理边界、仅在需要时检索上下文** —— 这是"按需检索"在本领域的占位。 |
| 域外 · 检索式 selective prediction | ICLR 2026 `2601.22570` **[A/B]** | **占掉"用检索邻域稳定嵌入以改进 abstention"。** |

**剩什么缝(推理策略):几乎没有。** "要不要检索 / 要不要升级 / 要不要弃权"这三个决策
在 2026 上半年被 ICLR/ACL 级工作全部填满。**这独立确认了 CVoI 之死。**

### 2.7 位置 G — 评估协议(**本轮唯一确认为空的整块**)

| 谁占着 | 出处 | 程度 |
|---|---|---|
| **NExT-GQA (Acc@GQA)** CVPR 2024 · **EG-VQA (EG-F1)** `2606.24797` **[A]** | 域外 video QA | 占掉"联合 correct-and-grounded 指标"这个**形状**。 |
| **NEC (Normalized Excess Cost)** `2605.03135` **[A/C]** | 域外(动机是内容审核) | 用 annotator vote margin / 阈值距离 / 置信度导出的**逐样本代价**加权错误。⚠️ 它自己报告:**代价敏感的训练收益不稳定,贡献在评估侧**。 |
| **PaSBench-Video** `2606.02443` | 域外(安全隐患) | 起始点标注 + FP 受控的流式协议。 |
| **V-DEAL** `2607.21151` **[A/C]** | Video LLM 安全 | 识别准确率 >81% 却仍有 48.33% 攻击成功率;**理解 ≠ 决策**。 |
| **AAAI 2026 遗漏率论文** | **[B]** | >90% 遗漏,归因帧采样。 |

**剩什么缝(评估):整块。**
subagent 跑了专门针对"chance-corrected localization / 联合 classify-and-ground / 校准"
的定向查询,**在窗口内没有找到任何一篇为 hateful video 提出评估协议的论文。**
这与我们 2026-08-08 的 Phase-3 结论一致,且现在有了**八个月的额外确认**。

而我们手上恰好有这个位子上**别人没有的测量**:
- HateMM 官方 gold span 覆盖率(298 个仇恨训练视频):**mean 0.717 / median 0.829**,
  34.6% 的仇恨视频被标注为 ≥90% 仇恨 ⇒ **top-1 命中的机会率 = 0.762**。
- 同一批盲标注者的 **minimal sufficient evidence** 区间中位覆盖 **0.131**;
  99 个配对视频上,官方覆盖是最小证据覆盖的 **2.0×**。
- 受控 trim 分解:full 0.8196 / 长度匹配随机窗 0.8155 / gold-span 窗 0.8203 ⇒
  **generic-trim −0.41pt,oracle-alignment +0.48pt**(bootstrap 95% CI [−0.79, +1.76])。
  对照 `2508.04900` 报告同一操作 **+19.34 / +30.45 macro-F1**(无随机窗对照、无分解)。

---

## 3. 开放问题清单(7 条)

每条给:为什么现有工作没解决 · 我们哪个资产能打 · 顶会够不够初判 · 最可能被哪篇论文当场反驳。
**已明确排除死方向清单及其换皮:** 多段互补、单段选择、OCR−ASR 残差、CVoI 获取式、
段级检索 key、视觉纯度选段、类型硬分区 memory、流式 memory、跨语言 EN 救 ZH。

---

### #1 — 仇恨视频的接地/定位评估:机会率修正,以及"HateMM 的 gold span 根本不局部"

**问题。** 本领域所有定位数字(MultiHateLoc mAP 0.645/AUC 0.799、LELA、HateClipSeg baseline、
我们自己的 P6/P10-b wv-AUC 0.54–0.58)都在**没有对该视频自身可达命中概率做修正**的情况下报告。
在 HateMM 上这个机会率是 **0.762**。同时,`2508.04900` 报告的 "trim 到 gold span 得 +19.34/+30.45"
在我们的受控分解下是 **+0.48pt(CI 含 0)** —— 因为 trim 只切掉约 18% 的视频,没有东西可以被浓缩。

**为什么现有工作没解决。** 定位在本领域是"任务优先"的一条线:大家在比 mAP,没有人问
"随机选一段能得多少"。video QA 那边的 Acc@GQA / EG-F1 占掉了"联合正确+接地"这个**形状**,
但**都不做逐视频的机会率修正**(它们的 gold 区间短,机会率低,不构成问题)。
hateful video 的 gold span 中位覆盖 0.829,把这个问题从"可忽略"变成"支配性"。

**我们的资产。** 上面 §2.7 的四组测量已经在手(零 GPU、零 test 触碰);
HateClipSeg 段级标注 + 我们 5 个 scorer 规模档的逐窗打分梯(7B/32B/72B/Qwen3-32B/Qwen3-30B-A3B,
全部带配对 CI)是一个**现成的重打分语料**,可以把"修正前 vs 修正后"的排名变化直接画出来。

**顶会够不够 —— 不够(作机制论文);够(作评估/资源类)。**
- **够不上** NeurIPS/ICML/ICLR/CVPR 主会的机制论文档:它没有机制。
- **够得上** ACL 主会(Resources & Evaluation track)/ NeurIPS D&B / ACM MM 的评估类论文,
  尤其如果 headline 是**数据集事实**(0.829 中位覆盖 + 2.0× 比值)而不是指标定义。
- 统计上必须先修好:逐视频 κᵢ=(hᵢ−pᵢ)/(1−pᵢ) 在 pᵢ→1 时发散、pᵢ=1 时无定义;
  必须并报聚合形式 Σ(hᵢ−pᵢ)/Σ(1−pᵢ),且 pᵢ 必须由**实际离散候选段 + 命中规则**
  经 Monte-Carlo 求出,不能用原始时长覆盖率。

**最可能当场反驳它的论文。**
1. **EG-VQA**(`2606.24797`)/ **NExT-GQA** —— "联合 grounded 指标是我们的,你只是加了个归一化"。
2. **NEC**(`2605.03135`)—— 同一个"用标注信息重新加权评估"的思路,而且已经写在内容审核语境下。
3. 一个直白的审稿人:"κ 式修正是对异常长的 gold span 的常规校正,不足以独立成篇。"
   ⇒ **缓解只有一条:让数据集事实当 headline,指标当附属。**

---

### #2 — 在**争议标签**上做检索:当邻居的标签本身没有正确答案

**问题。** 检索式分类的整套机制(kNN 投票、邻域共识去噪、pseudo-gold positive 挖掘)
默认"每个记忆条目有唯一正确标签"。仇恨检测**结构性地**不满足这个前提:
`2606.28772` 量到 HateXplain 中 **42.6% 的标注分歧集中在 hate/offensive 边界**;
而**我们的二分类协议正是把 offensive 折进 hateful** —— 我们把最大的分歧带整个吞进了正类。

**为什么现有工作没解决。**
- **检索侧:** 两条独立索引(arXiv + Semantic Scholar)在 2025-12→2026-08 窗口内返回**零**篇
  "标注分歧感知的检索"。同期落地的三篇顶会 noisy-retrieval 工作
  (AAAI-26 `2512.24064`、ICML-26 IN2R `2606.04061`、CVPR-26 ConeSep `2604.20358`)
  **全部是 noisy correspondence** —— 配对错了,存在唯一正确配对,邻域几何可以把它找回来。
  **主观分歧没有可恢复的正确答案,它们的机制前提不迁移。**
- **分歧侧:** learning-with-disagreement 的方法文献在 2026 上半年极度拥挤
  (DiADEM / EDO / soft-label training / cSG-MCMC / Socio-Contrastive / RGPO …),
  **但零篇是视频,也零篇把分歧接到检索上。**

**我们的资产。**
- ZH 上已验证的 **consensus 去噪**机制(`exp-consensus-zh-seeds`:λ=0.5 vs λ=0 floor,
  5 seed,final-epoch 0.7841 vs 0.7594),以及 EN 上归因清楚的失败
  (`EXP_mm_segment_keys`:片段监督通道对语音承载的仇恨无增益)。
- **memory 编辑取证**:两票 AND 规则的 guard-rail 之所以有价值(C−D +0.47pt EN / +0.40pt ZH),
  正是因为**语义票否决了 Cleanlab 式 embedding-only 对"真仇恨但 embedding-hard"记忆
  (虐待证词 / 性侵报道 / 含 slur)的过删** —— 这批条目就是争议带本身。
- MHC 的 3 类标签(Hateful / Offensive / Normal)**可直接当分歧代理**,零新标注。
- Claude 帧读取豁免已扩为通用(CLAUDE.md),**可以补标 per-item 分歧**而不需要新的 DUA 审批。

**顶会够不够 —— 有可能够,但卡在一个未做的核查上。**
- 机制形状是对的:**记忆存的是标签分布而不是硬标签,检索投票产生的是预测分布,
  abstention/路由由邻域分歧而非模型置信度驱动。** 这既不是 noisy-correspondence 去噪,
  也不是 annotator modeling(那需要 per-annotator 标签,我们大概率没有)。
- **门:** 我们必须先确定手上到底有什么分歧信号。已知 HateClipSeg 只公布了聚合 Krippendorff α;
  MHC 的 3 类是**聚合后**的类别而非分歧度。**如果最终只有"Offensive 类作二值代理"这一个信号,
  这条会退化成"在一个子集上加软标签",不够顶会。**
- 反面风险:ICLR 2026 `2601.22570` 已占"检索邻域稳定嵌入 → 改进 abstention"。

**最可能当场反驳它的论文。**
1. **ICLR 2026 `2601.22570`**(Memory Augmented Plug-and-Play Selective Prediction)——
   "用检索邻域改进弃权是我们的"。
2. **ICML 2026 IN2R `2606.04061`** —— "邻域共识合成软 prototype 是我们的"(尽管是 noisy correspondence)。
3. **`2511.14117`(Distributions In, Distributions Out)** —— "软标签训练全面优于硬标签已被证明,
   你只是把它用在检索头上"。
4. **`2603.22985`(Beyond Hate)** —— "拆标签(incivility vs intolerance)我们已经做了,
   而且把 FNR−FPR gap 减半了"。

---

### #3 — 把"买标注"而不是"买算力"作为价值-of-information 的代价轴

**问题。** 2026 上半年爆发的代价感知获取文献(12+ 篇,含 ICLR 2026 / ACL 2026 / ECML 2026)
**清一色**把被购买的资源定义为**算力/延迟/token**,把收益定义为**对固定 gold label 的准确率**。
在内容审核里,**真正稀缺且昂贵的是标注,而不是 GPU**;而且被买下来的不确定性
**不是模型的,是标注者之间的**。

**为什么现有工作没解决。** 这是两块拥挤区域的**交集**,而交集是空的:
- 代价感知获取(§2.1 最后一行,12+ 篇)—— 代价轴全是算力。
- learning-with-disagreement(§2.4)—— 从不把标注当作要用预算去买的动作。
- 经典 active learning 会买标注,但**目标是降低模型不确定性,不是降低标注者分歧**,
  也不做 value-of-information 的显式定价。

**我们的资产。**
- **W4 的一个已知正结果正好是这个形状**:时间切分下 EN 掉 −0.084 macro-F1,而
  **k=20 条新时段标注样本做阈值再校准把 0.6273 拉回 0.7336**(≥ 随机切分 floor 0.7113)。
  这是一条真实的"少量标注买回大量性能"的曲线,已经跑完、在手上。
- Gate-C 的 73 FN + 30 TP 分层审计,是一个现成的"标注预算怎么花"的实例。
- Claude 帧读取豁免 ⇒ 补标成本近乎为零,可以真正跑出 acquisition curve。

**顶会够不够 —— 有可能够,但这是三条里最"新框架、最少现成证据"的一条。**
- 优点:交集确实空;而且它把 CVoI 的尸体**倒过来用** —— CVoI 死于"OCR 已经预算完了,
  没有真实的推理期代价";而**标注代价是项目永远真实付的**,而且不可能被预计算掉。
  这直接回应了 `EXP_cvoi_acquisition_KILL_2026-08-09.md` §5 写下的复活条件的**前半条**。
- 风险:它需要一个"分歧下降"的可测量收益函数 —— 与 #2 同一道门。而且如果最终收益还是
  用 macro-F1 衡量,审稿人会说"这就是 active learning"。

**最可能当场反驳它的论文。**
1. **ACL 2026 `2605.13277`**(Utility-Oriented Visual Evidence Selection)——
   "以 information gain 定义证据效用、并证明其与答案空间效用一致,是我们做的"。
2. **`2607.05438`**(Modality Relevance is not Modality Utility)—— "relevance ≠ utility 是我们的口号"。
3. **经典 active learning 全家桶** —— "买标注降不确定性是 1990 年代的题"。
4. **NEC `2605.03135`** —— 它自己诚实报告"代价敏感的**训练**收益不稳定,贡献在评估侧",
   这会被拿来预测我们的结果。

---

### #4 — 韵律 / 副语言作为检索键(不是作为第四个模态)

**问题。** 仇恨常常在**表达方式**里(嘲讽、模仿、居高临下的语调),而不在字面。
本领域把音频当**模态**已经做透(HateMM MFCC/VGG19、CMFusion、MM-HSD wav2vec2-xlsr、
Koushik CLAP、TANDEM Qwen2-Audio、PCLMM),但**从来没有人把韵律当检索键**。

**为什么现有工作没解决。** subagent 用 `prosody`/`prosodic`/`paralinguistic` ×
`hate`/`abusive`/`toxic`/`aggression` 做了穷举查询,**窗口内零篇**。
最接近的是 `2604.09094`(低资源印度语音频滥用检测的少样本对比适配,ADIMA 10 语言),
它跳过 ASR 直接用 CLAP —— 而它自己的结论对我们不利:
"每语言加几条标注**收益很小**",冻结 CLAP 线性分类器已经到全监督的 1–3 点以内。

**我们的资产。** `data/audio/` 已有 HateMM / MHC / MHC_zh 的音频缓存(247 MB);
RGCL 头重训 ~52s;检索管线现成。**这是四条里实现成本最低的一条。**

**顶会够不够 —— 位子是空的,但先验很差,我判"大概率不够"。**
- 空位是真的(两条独立检索确认)。
- 但**我们自己的证据在反对它**:`EXP_mm_segment_keys` 的 W2 归因显示,
  MHC-EN 的仇恨证据 **65.5% 由语音/屏幕文字承载,纯 visual-only 只有 15/168 = 8.9%** ——
  也就是说仇恨主要在**说了什么**里,不在**怎么说**里;而且本项目的 T4/T5/T6 已经积累了
  十几条"模态杠杆不转化"的记录。
- 判定:**值得花一个 $0 的探针**(用现成音频特征做一次 prosody-keyed kNN 的邻域纯度测量),
  **不值得在探针之前写任何预注册。**

**最可能当场反驳它的论文。**
1. **Koushik HCC1** —— "CLAP 已经是 HateMM 上音频对 F1 贡献最大的组件,你只是把它换了个位置"。
2. **`2604.09094`** —— "音频少样本适配收益很小,已被报告"。
3. **ADIMA**(ICASSP 2022)—— "纯音频滥用检测和跨语言协议我们四年前就有了"。

---

### #5 — 逐实例记忆 vs 蒸馏 prototype:HCG-MPB 刚刚打开的争议

**问题。** HCG-MPB(ICMR 2026)**公开论证**逐实例检索在 hateful video 上是错的设计
(语义歧义 + 存储/延迟),并用 LLM 蒸馏 prototype 取代之。这是一个**可证伪的经验主张**,
而且**没有人跑过公平的对照**。同时它默认忽略了逐实例记忆独有的能力:
**测试时换库 / 外科编辑 / 阈值再校准都是 O(1) 且零重训的,prototype bank 结构上做不到。**

**为什么现有工作没解决。** 这个争议是 2026-06 才被制造出来的,而且提出方
**没有公开代码、没有公开数字**(摘要不给 headline)。MoRE 那边我们已经知道其
released code 有 7 项缺陷。**没有第三方做过 instance-vs-prototype 的同表对照。**

**我们的资产(这是我们唯一的"结构性优势"资产)。**
- **MoRE 官方代码全量复跑**:同 split(逐行 diff)、同 clean test、双 variant + 5-seed 敏感性,
  7 项代码缺陷全部文档化处置。**这是本项目最难被别人复制的东西。**
- 跨数据集 memory swap 矩阵(6 个 informative cross cell,5/6 above-majority,
  跨库落后 in-domain 0.04–0.09 macro-F1,**零重训**)。
- 时间切分下的阈值再校准曲线(0.6273 → 0.7336,k=20)。
- human-in-the-loop 记忆编辑 + guard-rail(⚠️ **单 seed,口径必须写成 capability demonstration,
  不是 accuracy claim** —— F88 更正:4-seed 均值 +0.0031)。

**顶会够不够 —— 不够作机制论文;作为实证研究是 ICMR/ACM MM 档,或作为 #1/#2 的一节。**
- 它是**研究(study),不是机制**。NeurIPS D&B 有可能,但需要它自己是一个 benchmark 贡献。
- 它的真实作用更可能是**防守性的**:任何未来的 RGCL-系论文都必须在 related work 里
  回应 HCG-MPB,而我们是唯一有资产做这个回应的人。
- ⚠️ **前置阻塞:HCG-MPB 的 PDF 在 ACM DL 付费墙后,数据集名和数字都拿不到。**
  在拿到 PDF 之前不要基于它做任何设计决策。

**最可能当场反驳它的论文。** HCG-MPB 自己(如果它的数字确实碾压);
以及 `2607.23304`(Context-Adaptive Inference)—— "在线性头下检索和参数适应本来就等价,
你们比的是同一个估计量的两种实现"。

---

### #6 — 重贴/近重复污染,以及"自然发生的最小对"是否存在(**门控**)

**问题。** HateMM(BitChute)、MultiHateClip(YouTube/Bilibili)、ImpliHateVid 全部采自
**转贴是常态**的平台。两件事从来没有人查过:
(a)train/test 之间有多少近重复泄漏(⇒ 已发表的数字有多少是记忆而非泛化);
(b)语料里有多少**同素材、反标签**的对(仇恨原贴 vs 新闻报道/反言论转贴)。
(b) 如果存在且规模可观,它就是一个**天然的、免费的最小对语料** ——
而 RGCL 的 hard-negative 挖掘正好是挖它的机器。

**为什么现有工作没解决。** 定向查询在 arXiv 摘要层面找不到任何针对仇恨视频基准的
近重复/污染审计。领域内所有论文都直接用官方 split。

**我们的资产。** 四个数据集的 CLIP 嵌入全部已缓存(`data/CLIP_Embedding/`,2.2 GB);
FAISS 管线现成;MoRE 官方复跑在手(可以在去重后的 split 上**重新给一个已发表基线打分**)。
**整件事是 $0 CPU、分钟级。**

**顶会够不够 —— 取决于测量结果,现在不能判。**
- 若泄漏严重 ⇒ 这是一篇 benchmark-integrity 论文(ACL 主会 / D&B 档),而且会改写领域的读数。
- 若泄漏轻微但**反标签最小对存在** ⇒ 升级为 §2.5 说的机制路线
  ("内容匹配、framing 对比的检索"),可能够 ACM MM / ACL 主会。
- 若两者都为空 ⇒ **零残值,直接关闭。**
- ⇒ **这条的正确动作是:先花那 $0 做测量,再决定要不要写。不要先写预注册。**

**最可能当场反驳它的论文。**
1. **`2603.22985`(Beyond Hate)** —— "把标签沿 tone/content 两轴拆开我们做了,而且有数字"。
2. **RGCL 本身**(`mei2023`)—— "检索挖 hard negative 就是我们的机制,你只是换了个说法"。
3. **TIHD**(ICMR 2026)—— 如果最小对被表述成"矛盾/不一致",它当场占位。
4. 一个直白的审稿人:"近重复审计是数据清洗,不是研究。"

---

### #7 — 屏幕文字的**来源分型**:上传者叠加的文字 vs 画面里本来就有的文字

**问题。** 所有用 OCR 的方法(MM-HSD 作 CMA query、CLaMR 作独立索引流、LELA 作五分之一、
WWW26 agentic 作证据源之一)把屏幕文字当**一个无差别的模态**。但在审核语境里,
这两类文字的**归责性完全不同**:
**叠加文字(字幕、标题条、meme 文字)是上传者自己的言语行为**(意图可归因);
**画面内文字(标牌、衣服、书页)是被拍摄的内容**(可能是关于仇恨的证据,而不是仇恨本身)。
这条区分恰好落在本领域最痛的失败模式上:新闻报道 / 反言论 / 引述 被判成仇恨。

**为什么现有工作没解决。** 本领域没有人做这个区分。域外倒是有两条相关但不同的线:
video text spotting 的经典 "graphic text vs scene text" 分类(几十年历史),
以及 2026 新出的 MLLM 侧研究(`2604.17375` When Text Hijacks Vision;
`2608.04244` SIGNPOST-Bench)—— 它们研究的是**冲突时模型怎么错**,不是**冲突时该归责给谁**。

**我们的资产。** **OCR cache 是本项目独有的**:PaddleOCR PP-OCRv6,K=30 中点窗,
**1246 视频**(HateMM 851 / 25,530 窗;HateClipSeg 395 / 11,850 窗),
`data/OCR/SHA256SUMS.json` 逐文件 sha256。统计:HateMM **80.85%** 视频有过滤后文字,
**57.85%** 的窗有文字,每视频中位 390 字符。
**每个检测都带框和置信度 ⇒ 跨窗的框位稳定性可以 $0 算出来**,
而叠加文字正是"位置固定、跨帧持续"的那一类。
另有 Gate-C census:`on_screen_text required` 在 FN 中 53.4% vs TP 中 33.3%(Fisher OR **2.29**)。

**顶会够不够 —— 我判不够(独立成篇);可能够作某篇论文里的一个机制组件。**
- 分型的**分类学不新**(video OCR 领域几十年前就有 graphic/scene text)。
  新的只能是**"归责性是审核的正确轴,且它可从 OCR 几何学出来"**这个主张。
- 审稿人极易读成 feature engineering。
- 而且 MM-HSD 用**无差别**的 OCR 就拿到了 0.874 —— 举证责任在我们:必须证明分型
  在 OCR-required 分层上有真实增益,且在语音承载分层上无回归。

**最可能当场反驳它的论文。**
1. **MM-HSD**(`2508.20546`)—— "无差别用全部 OCR 已经是 SOTA,你的拆分没有超过它"。
2. **`2604.17375` / `2608.04244`** —— "屏幕文字与画面冲突的现象我们已经 benchmark 了"。
3. **TraRA**(`2606.07161`)—— "跨帧文字轨迹聚合是我们的工程贡献"。
4. **WWW 2026 Companion agentic framework** —— "把 OCR 当独立证据源我们已经做了"。

---

### 3.8 排序与总判决

| 排名 | 位子 | 顶会档次初判 | 阻塞项 |
|---|---|---|---|
| 1 | **#1 机会率修正的接地评估** | **ACL 主会(Eval track)/ NeurIPS D&B / ACM MM。不够 NeurIPS 主会机制论文。** | 无(证据已在手);只需统计口径修复 |
| 2 | **#2 争议标签上的检索** | **可能够顶会机制论文 —— 唯一一条形状对的。** | **per-annotator / 分歧信号可用性核查(未做)** |
| 3 | **#3 标注-VoI(买标注而非买算力)** | **可能够,但证据最少、框架最新。** | 同 #2 的门 + 一个"分歧下降"收益函数 |
| 4 | **#6 近重复污染 + 自然最小对** | **取决于 $0 测量结果,现在不能判。** | 测量本身(CPU 分钟级,应该先做) |
| 5 | **#5 实例记忆 vs prototype** | 实证研究,ICMR/ACM MM 档;D&B 需再加 benchmark 贡献。 | **HCG-MPB PDF 在付费墙后** |
| 6 | **#4 韵律检索键** | 位子空,先验差。先做 $0 探针。 | 我们自己的 65.5% 语音承载证据在反对它 |
| 7 | **#7 OCR 来源分型** | 不够独立成篇;可作组件。 | 举证责任重(要打过无差别 OCR) |

**总判决(诚实版):**

> **本轮扫新没有找到任何一个"现成可写、机制级、够 NeurIPS / ICML / ICLR / CVPR 主会"的位子。**
>
> 唯一**确认为整块空着**的位置是**评估协议**(§2.7),而它按定义不是机制。
> 唯一**形状对**的机制位子是 **#2 / #3**(争议标签 × 检索 / 标注-VoI),
> 而它们卡在同一道我们**还没走过**的门上:我们的视频数据集到底有没有可用的分歧信号。
> 在走这道门之前,任何"这是我们的下一篇 NeurIPS"的说法都是没有依据的。
>
> 与此同时,2026 上半年有**三件事在收紧空间**,必须记在心里:
> (a)SAGE 把 HateMM 的 accuracy 赛道咬到小数点后第三位;
> (b)HCG-MPB 在会议论文里**公开论证逐实例检索是错的设计**;
> (c)`2607.23304` 在线性头假设下把**检索/路由与参数适应形式化地收编为同一个估计量**。
>
> **建议的下一步动作,按性价比排序(全部 $0 或近 $0,全部零 test 触碰):**
> 1. **分歧信号可用性核查**(读 HateClipSeg / MultiHateClip 的发布物,确认是否有 per-annotator
>    或 per-item 分歧字段)—— 这一步决定 #2/#3 生死,成本 = 读几个 README。
> 2. **近重复/反标签最小对测量**(#6)—— 用已缓存的 CLIP 嵌入 + FAISS,CPU 分钟级。
> 3. **韵律键邻域纯度探针**(#4)—— 用已缓存音频特征,CPU 级。
> 4. **拿 HCG-MPB / TIHD / MATCH 的 PDF**(#5 的前置)。
> 5. #1 的统计口径修复(Monte-Carlo 求 pᵢ + 聚合式 κ)—— 这条无论最后写不写方法论文都该做,
>    因为它是任何选择类主张的量尺。

---

## 4. 附录 C 复评(`NOVELTY_RECON_2026-08-09.md` 附录 C 的五个位子)

原附录记录了五个"没找到占位者但仍不够"的位子。逐个用本轮新证据重评。

| 位子 | 原判 | 本轮新证据 | **复评** |
|---|---|---|---|
| **C-1** 纯非梯度检索记忆的长时程崩溃诊断 / 理论 / 检测器 | 真空,但"必须是理论或诊断协议",且与仇恨视频无关、我们零资产 | **`2607.23304`(Context-Adaptive Inference)**在线性头 + 平方损失下把检索/路由与参数适应统一为 kernel ridge regression。这**给了理论工作一个现成的形式化把手**,但同时**杀掉了"非参数化本质不同"这个 framing**(与附录 B 第 8 条同向、更强)。 | **仍然不够,且理由更硬。** 现在不但我们零资产,而且这条要做,得先在 `2607.23304` 的假设之外立足。**关闭。** |
| **C-2** 冻结编码器下"表征空间有效性"作为记忆管理信号 | 边缘真空,位子窄且偏工程味 | **ERM `2602.05152` 证明 query expansion ≡ key expansion**;**IN2R(ICML 2026)** 占掉"动态跨模态记忆 + 图推理合成软 prototype"。 | **比原判更不够。** 剩余空间被两侧同时压缩。**关闭。** |
| **C-3** 在 TiC 式前向/后向迁移矩阵里补一条检索记忆基线 | 真空,但是 benchmark/实证贡献,属 D&B | **HCG-MPB(ICMR 2026)公开论证实例检索是错的设计**,给了这条一个**它原本没有的用途** —— 它不再是"补一条基线",而是"裁决一个刚被制造出来的争议"。 | **档次不变(仍是实证研究、不是机制),但优先级上升。** 不单独成篇;**并入 §3 #5**。 |
| **C-4** 插入"什么"而非"是否插入"(分型/逐模态/逐片段的子样本证据插入) | 四条里最弱;审稿人会读成架构变体;且段级键 / 类型分区已死 | **新角度出现了:插入的不是"证据的一部分",而是"标签的分布"。** §2.4 确认 learning-with-disagreement 的方法侧零篇视频、零篇接检索;§2.1 确认分歧感知的检索在两条索引上零结果。这把 C-4 从"往记忆里塞哪一块特征"(架构变体)变成"记忆条目该携带什么样的标签对象"(机制)。 | **唯一一个判决改变的位子:从"不够"改为"条件性可能够"。** 条件 = §3 #2 的那道门。**这就是 §3 #2。** |
| **C-5** hateful video 按上传时间的 train/test 切分 | 领域内真空,但 n≈150、只有 MHClip 可定年、存活偏差 | 无新证据改变规模问题。`2605.23598`(PHTV-Scout,186,727 条抖音/快手视频、6 个月、6.11% 有害率)显示**存在**规模够的时间语料,但**不是公开释出的、也不是我们能拿到的**。 | **仍然不够,理由不变(规模)。** 保留为 §3 #6 的一个可选维度(时间也是一种近重复来源),不单独立项。 |

---

## 5. 写作时的强制约束(从本轮扫新新增)

追加到 `NOVELTY_RECON` 附录 B「绝对不要主张新意的清单」之后:

9. **"我们的检索模块是一种(测试时)适应"** —— 被 `2607.23304` 在线性头假设下形式化收编。
10. **"我们改进了检索 query / key 的构造"作为独立机制主张** —— 被 ERM `2602.05152` 的
    query≡key 等价定理压缩。
11. **"邻域共识去噪"作为机制新意** —— AAAI 2026 / ICML 2026 / CVPR 2026 三个顶会八个月内落地。
12. **"逐样本判断检索/升级/弃权值不值得"** —— ICLR 2026 `2601.22570` + ACL 2026 `2605.13277`
    + VOILA / `2607.05438` 已填满,且带证明。
13. **"跨模态不一致/矛盾是隐式仇恨的信号"** —— TIHD(ICMR 2026)域内占位,MAGIC3 域外占位。
14. **"我们是第一个把标注分歧引入仇恨检测"** —— 文本侧极度拥挤(LeWiDi / DiADEM / EDO /
    soft-label / `2606.28772` / MultiPRIDE@EVALITA 2026)。只能写"**在视频上**",并引全这批。
15. **"我们首次把 on-screen text 用于仇恨视频"** —— MM-HSD / LELA / WWW26 agentic 全都用了。
16. 数字纪律:**SAGE 0.8710/0.8628(ACL 2026 主会)与 MM-HSD 自报 0.878/0.874(5-fold CV)
    不同协议**;SAGE 自己复现的 MM-HSD 是 0.8203/0.8054。**HVGuard 的数字是过滤子集
    (EN 丢 ~11% / ZH 丢 ~10%,FunASR 重转写),不可与全集数字同表。**
    TCL 的 HateMM 97.58 与 IARE 的 Ex-* 数字均不可比。
    `2508.04900` 的 98.64 是 **oracle-trim 天花板,不是检测结果**。
17. ⚠️ 不可引用:`2606.02911`(The Ghost Annotator,作者撤稿);`2604.06687`(RASR,
    作者自述实验需重做);`2608.02738`(已于 2026-08-06 撤稿,沿用旧记录)。
18. ⚠️ **SafeLens 是两篇同名论文**(AAAI 2026 demo/system vs `2605.17610`)——
    `DRAFT_intro_related_limitations.md` 现引的那条需核对。
19. ⚠️ **TIHD 常被误引为 ACM MM 2025;Crossref 核实为 ICMR 2026。**

---

## 6. 本轮新增、应当补进 `research-wiki/papers/` 的条目

按重要性排序(全部需要先拿到正文才能写深读记录):

1. **SAGE**(ACL 2026 主会 Long 817)—— 新 HateMM 前沿 + 决策级仲裁机制。**最高优先。**
2. **HCG-MPB**(ICMR 2026,`10.1145/3805622.3810724`)—— 公开反对实例检索。**最高优先。**
3. **TIHD / QGC-Net**(ICMR 2026,`10.1145/3805622.3810673`)—— Alignment Trap / 不一致放大。
4. **MATCH**(TCSVT,`10.1109/TCSVT.2026.3672052`)+ **ACE-HVD**(ICC 2026)。
5. **Failures to Surface Harmful Contents in Video LLMs**(AAAI 2026,`10.1609/aaai.v40i42.40841`)
   —— 对我们的帧采样是效度威胁。
6. **UniSafe**(WWW 2026 Companion)、**WWW 2026 Companion agentic framework**。
7. **Beyond Hate**(`2603.22985`)—— 拆标签的范例,#2/#6 的主要反驳者。
8. **CH-SV**(ACM MM 2025)—— 旧图缺失的中文有害短视频基准。
9. **HarmVideoBench**(`2606.27187`)、**PaSBench-Video**(`2606.02443`)、**SenBen**(`2604.08819`)。
10. 域外三条 framing 杀手:`2607.23304`、`2602.05152`、`2606.11198`。
11. 域外 slot 参考:`2601.22570`(ICLR 26)、`2605.13277`(ACL 26)、`2606.04061`(ICML 26 IN2R)、
    `2512.24064`(AAAI 26)、`2604.03657` LaPR / `2603.16737` CIRCLES(CVPR 26)。
