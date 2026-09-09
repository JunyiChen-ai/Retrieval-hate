# 规则 4 复核：机制 T（文本证据流）（2026-09-10）

审稿人：独立 agent（Claude Fable 5.1），只读，不跑训练；依据 `RESEARCH_ITERATION_RULES.md` 第 4 条（四类阻断：(a) 来源方法已用于 hateful video detection / localization；(b) 纯 ensemble；(c) 纯 calibration / 后处理 / 平滑；(d) 纯工程技巧：只调超参、只换特征、只加增强、只改训练配置）。上一份复核 `REVIEW_RULE4.md` 已审 A–D，本份只审 README 第 7 节的 T，不重复 A–D 的先例。

读过：README 第 0–2、7 节；`src/interval_evidence_hmm.py`（docstring、`TEXT_BINS`、`text_counts`、`text_observation`、`_text_loglik`、`_emissions`、`fit` 的文本计数与 M 步、`_neg_loglik_z`）；`src/hier_evidence_common.py`（`COL_TEXT`、`LLR_SCALE`、`text_llr_seconds`、`text_llr_rows`）；`experiments/20260910_online_query_within/model.py`（`EvidenceEncoder.text_input`、第 204–215 行的 LLR 读入与 `no_text_input` 臂）；`scripts/build_text_hate_scores.py` 与 `data/text_hate/PROVENANCE.md`；`scripts/reproduction_baselines/multihateloc/extract_bert_sentence_features.py`。

## 结论：PASS-with-phrasing（放行，带措辞要求，第 5 节）；无须改代码

T 由三件事组成，代码与 README 一致：(T1) 区间 HMM 新增 ASR / OCR 两个观测族——冻结文本仇恨分类器的逐秒 P(HATE) 分 10 档，按段以 s_g 为条件做分类发射，2×10 表由 EM 闭式更新，负例视频计入 s = 0 行，每条分块 / 每窗按权重 w 只计一次；(T2) 骨干 scaffold 第 7 列 = 用拟合表算出的逐秒对数似然比，经线性映射进证据编码器；(T3) 窗损失目标改为含文本的 HMM 后验。三者都改动生成模型或监督目标，不是"只换特征"（d 类不成立）；无第二个任务模型的预测被平均（b 类不成立）；不动输出分数（c 类不成立）。(a) 类：检索到的 hateful video 工作没有一篇用外部文本仇恨分类器的逐句 / 逐秒分数作为时间定位证据（第 1 节）。**但 T 的每个组成部分在弱监督标签模型、代价敏感特征获取、HMM 观测融合三条线上都有明确先例，论文不能把 T 写成"文本与 VLM 证据融合"的首创，只能写成"异质代价证据模型 + 免费证据后的剩余不确定性获取"这一组合，且必须引第 2–4 节的先例。**

另记一条不属于规则 4 的边界问题（不阻断，请主 agent 提交用户裁定）：规则 3 允许"单一预训练编码器 / VLM 作特征来源"，T 引入第二个冻结模型（`cardiffnlp/twitter-roberta-base-hate-latest`），且送进 HMM 的是它的**预测分数** P(HATE)，不是嵌入（MultiHateLoc 用 BERT 嵌入，MM-HSD 用 Detoxify 编码器输出的向量）。HMM 把它当观测、按 s 条件学发射表，不是平均两个模型的预测；本审稿人认为不属于规则 3 的 ensemble，但字面上"组合多个独立模型的 prediction"可被读作命中，建议在 README 写明理由并让用户裁定一次。

## 1. 检索记录（2026-09-10，WebSearch / WebFetch）与问题 (3) 的答案

| 工作 | 文本怎么用 | 是否逐句 / 逐秒仇恨分数 | 是否时间定位 |
|---|---|---|---|
| HateMM（Das et al., ICWSM 2023, arXiv 2305.03915） | 转录文本 BERT 特征，视频级融合 | 否 | 否 |
| MultiHateLoc（WWW 2026, arXiv 2512.10408；抓取 html 核对） | Whisper 句级片段 → BERT 768 维 → 按时间戳重复填充到帧；无 OCR、无词典、无分类器 | 否 | 是（嵌入，不是分数） |
| HateClipSeg（ACM MM 2025, arXiv 2508.01712；抓取核对） | 分类：LLaMA-3.2-11B 读转录；定位：ActionFormer 单模态 + 晚期融合 | 否 | 是（无文本分数） |
| HVGuard（EMNLP 2025） | BERT 文本嵌入 + LLM 推理，视频级 MoE | 否 | 否 |
| MM-HSD（ACM MM 2025, arXiv 2508.20546；抓取核对） | 转录与 OCR 分开，用 Detoxify（RoBERTa 仇恨模型）取**嵌入**，视频级；作者明写"无帧级定位" | 否（嵌入） | 否 |
| LELA（arXiv 2602.09637，training-free；抓取核对） | LLM 对 speech / OCR / 图像 / 音乐描述逐帧打 0–1 分，取各模态最大值 | 是，但打分者是 LLM，不是文本分类器；无训练、无生成模型 | 是 |
| TANDEM（AAAI-ICWSM 2027, arXiv 2601.11178） | VLM + 音频语言模型 RL，输出时间戳 | 摘要未见文本分类器 | 是 |
| CLARA（2608.15905）、MARS（2601.15115）、ToxVidLM（2405.20628） | VLM rationale / 对抗推理 / 代码混合视频分类 | 否 | 否 |
| DSANet、VERA、CMHKF | VAD 基线，无文本或只有类名 prompt | 否 | 是 |
| 音频侧近邻：Explainable Audio Hate Speech Detection（arXiv 2408.06065, 2024） | WhisperX 词级时间戳 + BERT 仇恨分类器，把文本 rationale 映回音频时间 | **是**（词级） | 是，但纯音频、合成数据（AudioHateXplain），无视觉、无生成模型 |

问题 (3) 的答案：**没有** hateful video 基线把外部文本仇恨分类器的逐句分数当逐秒定位证据；最近的是 LELA（LLM 逐帧打分、取最大）和 2408.06065（纯音频、词级 BERT 分数映回时间）。(a) 类阻断不成立。同时这也说明"逐秒文本仇恨分数当证据"本身是一个显而易见的输入选择（规则 5：换输入不是 novelty），claim 必须落在融合与获取的结构上。

## 2. 问题 (1) 的先例：免费分类器分数与付费裁定在同一生成证据模型里融合

1. **Snorkel / 数据编程**（Ratner et al., NeurIPS 2016；VLDB 2018）：多个异质、准确率未知的弱源，生成式标签模型无真标签估计各源准确率。T1 的"文本分类器 = 一个连续值弱源，发射表由 EM 学"就是这一族；差别是我们的隐变量是时间序列且各源覆盖区间不同。
2. **Dugong**（Sala, Varma, Fries, Fu et al., NeurIPS 2019, arXiv 1910.09505）：**多分辨率**弱源作用在视频 / 传感器序列上（有的源给单帧标签、有的给整段标签），生成模型无标签恢复各源准确率与相关。这与"细窗 / 粗块 VLM 裁定 + 逐秒文本分数"在结构上最近：三种分辨率的源进同一序列生成模型。论文必须引，并说明差别：我们的区间 HMM 是连续时间二态链、发射是"区间 OR"因子（VLM）与逐秒分类表（文本），推断精确；Dugong 是因子图 + 方法矩。
3. **linked HMM**（Safranchik, Luo, Bach, AAAI 2020）与 **skweak 的 HMM 聚合**（Lison et al., ACL 2021 demo）：HMM 隐状态 = 真标签、观测 = 各标注函数输出，无标签 EM，含可识别性证明。T1 的"分类发射表按 s 条件"与其同构。
4. **tandem HMM**（Hermansky, Ellis, Sharma, ICASSP 2000）：把外部判别模型的后验概率当 HMM 观测。这是"分类器分数作为 HMM 发射"的经典来源，T1 的分箱做法属于此类。

相对这些工作 T1 的新点只有：(i) 两类源代价不同（免费、付费）且付费源由主动获取决定哪些被观测，标签模型在"部分观测的付费源 + 全观测的免费源"上做精确推断；(ii) 同一个 HMM 同时产出骨干输入（LLR 列、ell）和窗级监督目标（T3）。这两点才是可写的。

## 3. 问题 (2) 的先例：免费证据先看，付费证据只补不确定处

1. **代价敏感主动分类 / 特征获取**：Greiner, Grove, Roth, "Learning cost-sensitive active classifiers", Artificial Intelligence 139(2) 2002；Ji & Carin, "Cost-sensitive feature acquisition and classification", Pattern Recognition 40 2007（决定下一个买哪个特征、何时停）；Kanani & Melville, "Prediction-time active feature-value acquisition", NeurIPS 2008 workshop（预测期只对不确定实例买特征）；Contardo, Denoyer, Artières, "Sequential cost-sensitive feature acquisition", S+SSPR 2016（明写"低成本测量先做，必要时再做高成本测量"）；Shim et al. NeurIPS 2018、Janisch et al. AAAI 2019（上轮已录）。"免费特征已观测、付费特征按剩余不确定性买"是这一族的标准设定，不是新原则。
2. **价值信息**：Howard, "Information value theory", IEEE TSSC 1966——EOC 是其模型输出版本（上轮已裁：EMOC / DIME）。
3. **LLM 级联 / 人机分工**：FrugalGPT（Chen et al. 2023，上轮已录）；CoAnnotating（Li et al., EMNLP 2023）按不确定性把标注分给 LLM 或人。"便宜模型先答、不确定再升级"与 T 的"文本先看、VLM 再问"同一逻辑，但它们是实例级路由，T 是同一视频内的时间区间级。
4. **多模态时序主动获取** A2MT（Kossen et al., TMLR 2023，上轮已录）：逐时刻决定获取哪个模态，含代价。

T 相对这些的新点：获取对象是**结构化时间区间**的裁定，免费证据以逐秒分辨率进同一生成模型，"不确定性"由该模型在部分观测下精确给出，输出是弱监督逐秒定位。不能写"提出免费优先原则"。

## 4. 阻断类别逐条核对

- (a) 无 hateful video 先例（第 1 节）。
- (b) 文本分类器与 VLM 都是冻结观测源，被同一个 HMM 按 s 条件建模，不是对两个任务模型的预测取平均；骨干仍是一个。规则 3 边界见"结论"段。
- (c) T 改的是生成模型、输入列与损失目标，不动推理输出。
- (d) T2 单独看接近"只加一列特征"，但它是 T1 拟合表的派生量，随 T1 放行；T1、T3 是模型与监督结构的改变，臂 `no_text` / `no_text_input` / `window_target_verdict` 可证伪各部分。

## 5. 论文措辞要求（写作前必须遵守，不阻断开跑）

1. T 命名为 "heterogeneous-cost evidence model"：一个区间 HMM 同时容纳免费的逐秒文本源（ASR、OCR 各一张 EM 拟合的分类发射表）与付费的区间 VLM 裁定；引 Snorkel、Dugong（结构最近）、linked HMM / skweak、tandem HMM；说明与 Dugong 的三点差别（连续时间链、区间 OR 因子、精确推断）。
2. 获取部分写 "the acquisition criterion (EMOC) is evaluated on the posterior that already conditions on the free evidence, so paid calls go where free evidence leaves the output uncertain"；引 Greiner 2002、Ji & Carin 2007、Contardo 2016、Kanani & Melville 2008、CoAnnotating；不写"提出 free-first 原则"。
3. 文本分数来源写成实现细节（规则 5：换输入不是 novelty）："a frozen off-the-shelf text hate classifier (Cardiff NLP twitter-roberta-base-hate-latest) on Whisper chunks and PaddleOCR windows"；明写 LELA 用 LLM 逐帧打分、2408.06065 用 BERT 词级分数做纯音频定位，作为相关工作。
4. T3（后验窗目标）承接上一份复核对 C 的措辞："masked verdict prediction toward the HMM posterior that fuses the free evidence"；仍不写 self-supervised。
5. 不写"首个把文本证据用于 hateful video 定位"（MultiHateLoc 已有帧对齐文本嵌入）；只能写"首个把外部文本分类器分数作为生成证据模型的观测并据此决定 VLM 调用位置"。

## 6. 书目补录（并入 `docs/20260908_adaptive_query_survey.md` 补录表）

| 短名 | 题目 | 第一作者 | 出处 | 标识 | 核实 |
|---|---|---|---|---|---|
| Snorkel | Snorkel: Rapid Training Data Creation with Weak Supervision | Alexander Ratner | VLDB 2018（数据编程 NeurIPS 2016） | arXiv 1711.10160 | 已核实 |
| Dugong | Multi-Resolution Weak Supervision for Sequential Data | Frederic Sala | NeurIPS 2019 | arXiv 1910.09505 | 已核实 |
| linked HMM | Weakly Supervised Sequence Tagging from Noisy Rules | Esteban Safranchik | AAAI 2020 | doi 10.1609/aaai.v34i04.6009 | 已核实 |
| skweak | skweak: Weak Supervision Made Easy for NLP | Pierre Lison | ACL 2021 demo | arXiv 2104.09683 | 已核实 |
| tandem HMM | Tandem Connectionist Feature Extraction for Conventional HMM Systems | Hynek Hermansky | ICASSP 2000 | — | 已核实 |
| Greiner 2002 | Learning Cost-Sensitive Active Classifiers | Russell Greiner | Artificial Intelligence 139(2) 2002 | — | 已核实 |
| Ji & Carin 2007 | Cost-Sensitive Feature Acquisition and Classification | Shihao Ji | Pattern Recognition 40 2007 | — | 已核实 |
| Kanani 2008 | Prediction-time Active Feature-value Acquisition for Cost-Effective Customer Targeting | Pallika Kanani | NeurIPS 2008 workshop (Cost-Sensitive Learning) | ciir-publications id 860 | 已核实存在；workshop 名凭记忆 |
| Contardo 2016 | Sequential Cost-Sensitive Feature Acquisition | Gabriella Contardo | S+SSPR 2016 | arXiv 1607.03691 | 已核实 |
| Howard 1966 | Information Value Theory | Ronald A. Howard | IEEE Trans. Systems Science and Cybernetics 1966 | — | 经典，未再核 |
| CoAnnotating | CoAnnotating: Uncertainty-Guided Work Allocation between Human and LLMs for Data Annotation | Minzhi Li | EMNLP 2023 | arXiv 2310.15638 | 已核实 |
| MM-HSD | MM-HSD: Multi-Modal Hate Speech Detection in Videos | — | ACM MM 2025 | arXiv 2508.20546 | 已核实 |
| LELA | Towards Training-free Multimodal Hate Localisation with Large Language Models | — | arXiv 2026-02 | arXiv 2602.09637 | 已核实（上轮已录，本轮核对文本用法） |
| AudioHateXplain | An Investigation Into Explainable Audio Hate Speech Detection | — | arXiv 2024 | arXiv 2408.06065 | 已核实 |
| HateMM | HateMM: A Multi-Modal Dataset for Hate Video Classification | Mithun Das | ICWSM 2023 | arXiv 2305.03915 | 已核实 |
| MARS | Training-Free and Interpretable Hateful Video Detection via Multi-stage Adversarial Reasoning | — | arXiv 2026-01 | arXiv 2601.15115 | 已核实存在，只读摘要 |
| WWW'26 agentic | An Interpretable Agentic Framework for Multimodal Hate Video Analysis with Explicit Evidence Attribution | — | WWW 2026 Companion | doi 10.1145/3774905.3796488 | 页面 403，未读正文 |
