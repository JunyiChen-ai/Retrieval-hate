# Idea Discovery Report — 骨干（module 2）与融合（module 3）新候选

**方向**：弱监督 hateful video localization，在候选 1（hier_evidence_mil 修订 1）之上，为骨干模块与融合模块各找一个可 claim 的设计。
**日期**：2026-09-05。
**输入**：`docs/20260905_research_brief_backbone_fusion.md`（主上下文）、`RESEARCH_ITERATION_RULES.md`、`research-wiki/STATUS.md`、候选 1/3/4 README 的机制分析节。
**流程**：research-lit（4 个并行文献切片，fable agent）→ idea-creator（Codex gpt-5.6-sol xhigh 头脑风暴 11 条 + 3 个 Claude 分析视角各 6 条 + 执行者 7 条种子）→ 机械去重为 14 个机制簇 → Codex 交叉模型 devil's-advocate 分诊 → novelty-check（3 个并行检索 agent + Codex 交叉核对）→ research-review（Codex）→ research-refine-pipeline → 本报告。
**约束（覆盖 skill 默认）**：无 pilot、不用 GPU（MAX_PILOT_IDEAS = 0，所有候选标 "needs manual pilot"）；不渲染 HTML；输出只进 `docs/idea-stage-20260905/`。

## 执行摘要

跑了什么：四个并行文献切片（约 150 篇核对）、Codex 头脑风暴 11 条 + 三个 Claude 视角 18 条 + 执行者 7 条种子，去重为 14 个机制簇；Codex 交叉模型分诊淘汰 8 个、降级 2 个；对前三名做多源 novelty 复核（C1 6/10、C13 5/10、C7 增量 4/10 / 整体模块 6/10，均 PROCEED WITH CAUTION，无 ABANDON）；组合提案经 Codex ultra 两轮评审，从 3/10 修到 5/10（Weak Reject），主要数学缺陷已修（发射结构改为 coarse-first 有向模型、比例损失改作用于最终概率、去掉 target-copy、加精确重参数化对照）。
出了什么：推荐一个组合方法"裁定条件化密度估计"——模块 3 = coarse-first 条件标注模型（细裁定发射以粗裁定与前一细裁定为条件，负例识别误报侧），模块 2 = scale–rank 分解头（学习的视频级比例标量 + 中心化排序项），文本 between/within 路由作子开关。它有项目内测得的杠杆（K4-only 后验 .591/.851 对全证据 .541/.818；no_text ROC +.027 但 within −.036；注意力只作用于 pooled）。
对决策意味着什么：(1) 没有 pilot，所有候选"needs manual pilot"；(2) 第一步是 **0 GPU 的 CPU 门**（条件后验必须同时胜过 K4-only 与 4-cell lookup；q_v 必须是可用的比例估计），不过门就不实现网络；(3) 评审的诚实算术表明三个开关的预期增量都可能落在 ±.02 噪声内，按预注册标准删开关，不硬撑；(4) 顶会 novelty 为应用级，规则 4 不阻断，但论文定位必须正面引用 CHMM / FABLE / Dugong / MSL / ARMS / 2026 两篇 pooled-vs-within 审计。

## 1. 文献图景（Phase 1，四个切片，约 150 篇，均经 arXiv/出版页核对，未核对者标注）

### 1.1 WS-VAD / 音视频骨干（2022–2026）
MACIL-SD（MM 2022）共享跨模态层 + semi-bag 对比 + EMA 伙伴、padding 不屏蔽；UR-DMU（AAAI 2023）global + 固定局部窗口双分支、双记忆库、NUL 高斯正常隐变量；MGFN（AAAI 2023）glance/focus、特征范数放大、magnitude contrast；PEL4VAD（TIP 2024）global/local 用**一个**可学习标量融合；BN-WVAD（TCSVT 2024）batch 级自适应 k；DSANet（AAAI 2026）原型正常重建 + 用自身分数切 event/background；RefineVAD（AAAI 2026）逐 snippet 运动显著性门 + 类别原型交叉注意力；GS-MoE / GlanceVAD 时间高斯核；LAS-VAD（CVPR 2026）；VadCLIP / TPWNG / STPrompt / PE-MIL / LEC-VAD（ICML 2025）/ Fed-WSVAD / LAP / SteerVAD（ICLR 2026）；音视频：HyperVD、DSRL（NeurIPS 2024）/ PiercingEye（TPAMI 2025）、MSBT bottleneck token、CFA 逐 snippet 音频相关门、AVadCLIP、AViS-Mamba、IEF-VAD。
结论：文献只在三处变化——global 上下文从哪里进（永远是固定结构或一个标量，从无逐秒门）、什么先验调制特征（类别 prompt、原型库）、模态怎么合。**没有任何 WS-VAD/hate 论文让骨干读两粒度外部冻结标注器的逐窗裁定**。

### 1.2 外部噪声标注器与内容模型的融合
VLM/LLM 作标注器：LAVAD（CVPR 2024）相似度加权平滑、LAP、VERA（CVPR 2025）、AnomalyRuler、Holmes-VAU、SlowFastVAD（置信门控的测试期融合）、MuST-VAD（2026，双向循环但要改 LVLM）、MLLM4WTAL（CVPR 2025，MLLM 时间先验经 matching/distillation 损失进入，无噪声模型）、NoCo（AAAI 2025）、ZEAL、TFPLG。带时间结构的 label model：Dugong（NeurIPS 2019，多分辨率源、可识别性）、Linked HMM（AAAI 2020）、CHMM（ACL 2021，BERT 条件化转移/发射 + CHMM-ALT 交替）、Neural-Hidden-CRF（KDD 2023）、WeaSEL（NeurIPS 2021）。logit 偏置/PoE：Tip-Adapter、AMU-Tuning（CVPR 2024，按不确定性定融合权重）、logit adjustment（ICLR 2021，先验必须进训练损失才 Bayes 一致）。证据/不确定性融合：GUEF、IEF-VAD、DELU（ECCV 2022）。实例依赖噪声：CoNAL、part-dependent label noise（NeurIPS 2020）。结构推断入环：NN-Viterbi、Set-Constrained Viterbi、Neural HMM E2E（ASRU 2023）、Structured Attention Networks。
直接核对：**没有工作把 EM 拟合的 HMM 后验（对 VLM 逐窗裁定）作为 MIL 训练期先验**；最近的是 MLLM4WTAL（无噪声模型）与 NLP 的 CHMM/Neural-Hidden-CRF（从未用于视频裁定）。

### 1.3 Hateful video 与 WTAL
已用于 hateful video（规则 4 第一类阻断清单）：逐模态 transformer 编码器 + 逐时刻模态 sigmoid 门 + 跨模态注意力 + 同时刻对比 + top-k MIL + 平滑（MultiHateLoc, WWW 2026；HateMM mAP .645 / AUC .799，其自身 pipeline）；ActionFormer 全监督（HateClipSeg, MM 2025）；MoE 门控融合 + LLM 理由嵌入（HVGuard, EMNLP 2025，视频级）；实例图 + 权重图（MultiHateGNN, BMVC 2025）；OCR-as-query 跨模态注意力（MM-HSD, MM 2025）；时间 V–A 交叉注意力 + 通道/模态融合（CMFusion）；两阶段对比（ImpliHateVid, ACL 2025）；training-free LLM 逐帧打分（LELA）；tandem RL（TANDEM, ICWSM 2027）；分段 + LoRA policy LLM（SafeLens, AAAI-26 demo）；测试期适应（SCANNER, AAAI 2026）。
hateful video 工作中缺席：证据/不确定性建模、类无关前景分支、聚类/OT 自标注、歧义类型化 snippet 图、proposal 级 MIL、边界难样本挖掘、上下文拼接一致性、weak-to-full 头。
WTAL 可迁移：CoLA、ASM-Loc、DELU、P-MIL、DDG-Net（歧义 snippet 只收不发）、CASE（聚类 + OT 先验边际）、Bi-SCC、HSLA、PVLR、PseudoFormer。

### 1.4 注意力/融合机制变体
sink 与空 token：Registers（ICLR 2024）= 广播槽（即候选 4 的空 token）；StreamingLLM；gpt-oss 逐头标量 sink 偏置；Gated Attention（Qiu et al., NeurIPS 2025）query 条件 sigmoid 门消除 sink；sink 统一视角（2606.08105）：NOP sink 用输出门修、broadcast sink 用 register 修。逐 token 门控：Flamingo tanh 门（逐层标量）、Leaky Gated Cross-Attention（WACV 2022, WTAL）、Predictive Dynamic Fusion（ICML 2024）、ModDrop、MMP、FuseMoE（NeurIPS 2024）。尺度/排序：RTFM、MGFN、UR-DMU、VadCLIP（video prompt 广播）、BN-WVAD。LLP/基数势：LLP-ROT、cardinality potential（CVPR 2015 / UAI 2013）。AVEL/AVVP：PSP（阈值化配对，无配对的段不更新）、CMBS（跨模态分歧门）、JoMoLD、VALOR。图：HL-Net、DDG-Net。
结构空缺：(1) query 侧输出门代替 null key；(2) softmax 分母偏置的零参数 sink；(3) 显式 scale × rank 分解，s_v 由比例/基数目标训练、r_t 由视频内损失训练，within 下限按构造安全；(4) 非对称可靠性传播；(5) 以局部裁定模式为路由输入的逐秒专家路由。

### 1.5 图景对项目诊断的重述
项目自己测得：注意力层只作用于 pooled（跨视频尺度），within 不动；padding 是训练期才有的 NOP sink；候选 3 的视频级向量是 broadcast，破坏 HateMM within；候选 4 的空 token 是 register 型修法，在 HCS 只拿 1/T 注意力。四个切片一致指向两条未试过的路线：把"视频级密度"从注意力的副产品变成显式、顺序保持的标量项；把"两粒度可靠性校正"从骨干隐式学习搬进证据模型本身。

## 2. 候选池（Phase 2，机械去重后的 14 个机制簇）

| 簇 | 机制 | 模块 | 来源 |
|---|---|---|---|
| C1 | scale–rank 分解头：score_t = s_v + r_t，s_v 视频级密度标量（裁定分布统计 + 池化内容，比例目标来自 HMM 期望仇恨比例），r_t 逐秒排序项按视频去均值；按构造顺序保持 | 骨干 | Codex 1、regime 4、transfer 2、diagnosis 1、seed 1 |
| C2 | query 门控共享跨模态注意力（leaky），padding 屏蔽，门读 query + 局部裁定 | 骨干 | Codex 2、transfer 1、diagnosis 2、regime 3 |
| C3 | 多分辨率 / 块内受限注意力（+ 块级证据 register） | 骨干 | Codex 3、regime 5、seed 2 |
| C4 | 只收不发的可靠性注意力（逐 key 可靠性偏置，DDG-Net 非对称） | 骨干 | Codex 4、regime 3 |
| C5 | 长度稳定的 NOP sink（a_h(q_t) + log T 进 softmax 分母） | 骨干 | Codex 5 |
| C6 | 裁定格子条件化变换（低秩 Q/K/V 调制 或 FFN 专家） | 骨干 | Codex 6、transfer 3、seed 7 |
| C7 | 实例依赖发射的两级 HMM：发射参数按局部裁定上下文变化（逐窗 logistic 发射 / part 矩阵混合 / 与粗裁定链接 + 视频级可靠性隐变量），负例识别误报侧，正例 EM，可选经可微前向后向由 MIL 损失精调 | 融合 | Codex 7、transfer 5、regime 2、diagnosis 4、seed 3 |
| C8 | 精度加权 PoE 融合（逐秒 τ_e、τ_c） | 融合 | Codex 8、regime 1、seed 4 |
| C9 | 内容条件化时间结构（HSMM 内容 hazard / 秒级 IOHMM，裁定作两粒度 OR 发射） | 融合 | Codex 9、transfer 6、seed 5 |
| C10 | 基数：块计数替代 OR（cardinality CRF）/ HMM 计数后验定 bag 大小 | 融合 | Codex 10、regime 6、seed 6 |
| C11 | 冲突感知证据 HMM（subjective logic） | 融合 | Codex 11 |
| C12 | 后验监督的双原型记忆（UR-DMU 迁移） | 骨干 | transfer 4 |
| C13 | 文本流 between/within 分解：逐视频均值走标量头，逐秒偏差进内容流 | 骨干 | diagnosis 3 |
| C14 | 视频密度协变量进 HMM 的 p0 与转移（IOHMM 视频级输入） | 融合 | diagnosis 5 |

可行性门（客观）：全部在预算内（≤ 5 天实现、≤ 9 GPU-h/候选），无不可得数据；此阶段不淘汰。

## 3. 交叉模型分诊（Phase 2 jury，Codex gpt-5.6-sol xhigh，与头脑风暴同线程）

逐簇结论（三 seed 相对候选 1 的预期变化区间；within 风险）：

| 簇 | 预期 pooled 增益 | HateMM within 风险 | 最强反对 / 真正威胁 | 分诊 |
|---|---|---|---|---|
| C1 | HateMM AP +.005~+.025；HCS ROC 0~+.012 | 按构造为零（D6 块级项升为中） | "联合训练的视频分类器 / 逐视频校准"；PEL4VAD | 骨干第 1 |
| C2 | HateMM AP −.010~+.010；HCS AP −.020~+.003 | 低 | 项目内 gated_cma 臂已在 HCS 变差（.690 对 .706）；Leaky Gated CA | 淘汰 |
| C3 | HateMM AP −.030~+.015；HCS ROC +.005~+.025 | 高 | 块内广播 = 候选 3 的局部版本；UR-DMU | R5 淘汰，X3 备选 |
| C4 | HateMM AP −.020~+.015；HCS ROC +.005~+.020 | 高 | DDG-Net 几乎相同；候选 3 key 偏置先例 | 骨干第 3（须 matched-stream 消融） |
| C5 | ≈ 0 | 低 | 候选 4 zero_value_sink 已为零结果；padding 工程 | 淘汰 |
| C6 | HateMM AP −.005~+.015 | 中 | 原始裁定列已完成校正；T3 专家触 HVGuard MoE | T3 淘汰，低秩版备选 |
| C7 | HateMM AP +.015~+.040 / ROC +.008~+.025；HCS AP 0~+.012 / ROC +.003~+.018 | 中 | 正例侧可识别性；HateMM val/test 对 K30 方向相反；CHMM | 融合第 1 |
| C8 | HateMM AP +.004~+.020；HCS ROC +.006~+.020 | 中 | 两专家都读裁定，PoE 条件独立不成立；被判 learned calibration；IEF-VAD | 融合第 4 |
| C9 | HateMM AP −.020~+.010；HCS AP +.005~+.025 / ROC +.010~+.030 | 高 | 内容变化检测的是剪辑不是仇恨边界；IOHMM/Neural HMM E2E | 融合第 2 |
| C10 | HateMM AP 0~+.015；HCS AP +.003~+.020 | 中 | R6 是 pooling 配置；X10 若 K4 ≈ OR 则无信息 | X10 备选，R6 淘汰 |
| C11 | ≈ 0 | 中 | mass 不可标定；被 C7/C8 覆盖 | 淘汰 |
| C12 | ≈ 0 | 高 | UR-DMU 直接迁移、循环伪标签 | 淘汰 |
| C13 | HateMM ROC +.012~+.030 / AP +.003~+.015；HCS ≈ 0 | 低 | 单独看是输入变化（规则 4 第四类）；无直接先例 | 骨干第 2（作 C1 的组成部分） |
| C14 | HateMM ≈ 0；HCS AP/ROC +.005~+.020 | 中 | 裁定计数双重计入；视频级先验再校准；候选 2 负先例 | 淘汰 |

排序：骨干 C1 > C13 > C4 > C3 > C6 > C2 > C5 > C12；融合 C7 > C9 > C10(X10) > C8 > C14 > C11。
jury 建议的组合：**C7 + C1 + C13 作为一个方法**，三个互不重叠的消融开关（`conditional_emission` / `density_rank` / `text_decompose`）；C7 首版只用裁定局部上下文（不用视频比例、不用内容门），避免与 C1 的密度路径重叠。

## 4. Novelty 复核（Phase 3；三个并行检索 agent，WebSearch 额度中途耗尽，改用 arXiv API / OpenAlex / Crossref / ACL Anthology，所有引用已核对；trace `.aris/traces/novelty-check/20260905_run01/`；Codex 交叉核对并入第 5 节评审）

| 候选 | 评分 | 结论 | 最近先例（威胁） | 未被 claim 的部分 |
|---|---|---|---|---|
| C1 scale–rank 分解头 | 6/10 | PROCEED WITH CAUTION | MSL（Li, Liu, Jiao, AAAI 2022）视频级异常概率在推理时抑制片段分数波动；ARMS（Shi et al., IEEE TMM 2024）自估异常比例的 ratio-MIL 损失；3C-Net（ICCV 2019）计数损失；VadCLIP coarse/fine；LLP 理论（Quadrianto JMLR 2009；∝SVM ICML 2013）；两篇 2026 审计（arXiv 2608.21854 Song & Lee：Micro-AUROC = w·Within + (1−w)·Cross，视频均值替换保留 98.6% 的 margin；arXiv 2608.11985）已发表"pooled AUC 主要度量跨视频排序"的诊断 | 可识别的加性视频级项 + 来自外部标注模型的比例目标 + 中心化排序项；hateful video 定位器中没有任何视频级分支 |
| C13 文本 between/within 分解 | 5/10 | PROCEED WITH CAUTION | cepstral mean normalization（HLT 1993）；RUBi（NeurIPS 2019）/ Clark et al.（EMNLP 2019）bias-only 分支；2026 审计；SAGE（ACL 2026）hateful video 分类的模态解耦专家 | 无先例做逐视频均值中心化并把均值路由到顺序保持的标量头；无人报告 ASR 文本 pooled 有害 / within 有益。风险：单独看是输入归一化（规则 4 第四类），线性首层可重参数化 |
| C7 条件发射标注模型 | 增量 4/10；整个"VLM 裁定上的多分辨率标注模型"约 6/10 | PROCEED WITH CAUTION（作模块内被消融的设计选择） | CHMM（ACL 2021）token 条件化发射；FABLE（AISTATS 2023, arXiv 2210.02724）实例特征依赖混合权重的标注模型混合；Dugong（NeurIPS 2019）跨分辨率源相关；Hyper Label Model（ICLR 2023）；Linked HMM（AAAI 2020）；Lasserre–Bishop–Minka（CVPR 2006）混合生成判别；WeaSEL | 粗裁定作为细裁定可靠性协变量的跨分辨率条件发射；误报侧由负袋识别；**hateful video 文献中没有任何对 VLM 输出的标注模型** |

三者都不触发规则 4 的四类 STOP（未用于 hateful video；非 ensemble；非纯校准/后处理；非纯工程——C13 单独会触发第四类，故只作 C1 的子开关或独立标量头，且以分解 claim + 视觉/音频对照臂支撑）。

## 5. 外部批判性评审（Phase 4；Codex gpt-5.6-sol，ultra，新线程；trace `.aris/traces/research-review/20260905_run01/`；全文摘要 `refine-logs/REVIEW_SUMMARY.md`）

第 1 轮（原始 C7 + C1 + C13 组合）：**3/10 Reject**。决定性缺陷：条件发射把邻窗细裁定与块内计数同时放进两级发射，是循环伪似然；比例损失作用在 sigmoid(s_v) 而非最终逐秒概率且 mean P(s_t) 既是输入又是目标；先中心化再加先验；块级 MIL 监督中心化后的相对量；缺精确重参数化对照；三开关不独立；文本门槛低于 no_text 臂；预注册门槛低于噪声（HateMM 差值 SEM ≈ .0155）；CPU 预检只要求复制 K4-only；test 驱动搜索使确认性失效（项目裁定，记录不改）。
第 2 轮（修订提案，见 `refine-logs/FINAL_PROPOSAL.md`）：**5/10 Weak Reject**。接受 coarse-first 有向分解、先验后中心化、最终概率比例损失、去 target-copy、总分块级 MIL、精确重参数化对照。仍要求：q_v 比例有效性作 go/no-go 门（最能提分的单一改变，过则 6/10）；OOF、label-free 训练期后验；`no_b4` / `no_bprev` 隔离机制；CPU 门须有预定最小优势且胜过 4-cell lookup；λ_prop = 0 与 block-relocation-only 臂；完整 2^3；并指出按作者自己的预期，三个开关的增量都可能低于采用门槛。以上除协议裁定外全部写入最终提案与实验计划。

## 6. 排序后的候选（含 novelty 与 review 结果）

### 推荐：R1 = C7 + C1（C13 作子开关）"裁定条件化密度估计"（Verdict-Conditioned Density Estimation）
- **做什么（先说方法）**：(1) 重拟合两级 HMM，把细裁定发射改成 P(b30_t | s_t, b4_block(t), b30_{t−1}) 的 logistic 发射（coarse-first 有向模型），θ_0 只在负例视频上用 logistic 回归拟合，其余参数正例 EM；(2) 骨干的逐秒分数改为 score_t = s_v + r_t：a_t = z_t + α ℓ_t/L 按视频去均值得 r_t，s_v 是一个读裁定分布统计（触发比例、4 格直方图、最长 run）、detach 后的池化内容与 BERT 视频均值的 MLP 标量；(3) 损失 = bag BCE + 比例损失（最终逐秒概率均值 → HMM 期望仇恨比例，负例 0）+ 块内总分的块级 MIL + CMAL；(4) 文本只以逐秒偏差进入内容流。
- **替换什么**：候选 1 的四个全局发射率；候选 1 的单一 logit + 顶 k bag；EMA 伙伴删除。
- **预期增益与原因**：HateMM——K4-only 后验单独 .591/.851 对全证据 .541/.818 说明发射参数错估 .05 AP 的量，网络现在靠原始裁定列补回 .036，把校正搬进标注模型后先验路径可承担更大 α；密度项对应 92% 的视频间方差，no_text 显示 ROC 有 +.027 的空间。HCS——后验单独 ≈ 完整模型，只有模块 3 与 ROC 有空间（+.005~+.010）。**诚实预期**：端到端每个开关 +.00~+.02，可能落在噪声内。
- **消融臂**：CPU 门 A1（global / K4-only / K30-only / lookup / conditional / no_b4 / no_bprev）与 A2（q_v 比例 MAE/校准）；精确重参数化、固定 logit q_v、λ_prop = 0、block-relocation-only、只裁定统计、不中心化、删注意力保 s_v、MSL 乘性门、K4-only 先验；2^3 八格 × 3 随机数流。
- **HateMM within 风险**：推理时 s_v 是视频常数，不能重排；训练耦合（∂L/∂r_t 依赖 s_v）监控；条件发射改变后验的格间排序，方向与 GT 一致（(1,0) 应低于 (0,1)）。
- **最近先例与差异**：MSL（AAAI 2022，乘性视频门，推理期、二值标签）；ARMS（TMM 2024，自估比例 ratio 损失，作用于片段分数）；CHMM（ACL 2021，BERT 条件化发射，NLP）；FABLE（AISTATS 2023，实例依赖混合）；Dugong（NeurIPS 2019，多分辨率源相关）；2026 两篇审计（诊断，无方法）。均未用于 hateful video。差异：比例目标来自外部标注模型；加性可识别标量 + 中心化排序项 + 精确重参数化对照；跨分辨率条件发射 + 负袋识别。
- **与候选 2/3/4 的区别**：链仍是先验不是输出、发射上下文只读裁定不读内容（候选 2 的门因内容驱动 + 边际似然而卡死）；无视频级向量进内容表示（候选 3）；不动注意力 key（候选 4）。
- **Novelty**：C1 6/10、C7 4/10（模块 6/10）、C13 5/10。**Reviewer 评分**：3/10 → 修订后 5/10。**Pilot**：needs manual pilot；先跑 0 GPU 的 CPU 门。
- **下一步**：用户裁定 → `experiments/<date>_vcde/README.md`（规则 4 复核）→ 实现 → 规则 6 → Block A。

### 备选：R2 = C9-T6 秒级 IOHMM（内容变化驱动转移，裁定作两粒度 OR 发射）
- 方法：把链放到 1 fps，K30/K4 裁定作窗/块内 OR 的观测（8 态增广链），P(switch) = σ(a_s + softplus(β)·d_τ)，d_τ = 1 − cos(v_τ, v_{τ−1})；EM 拟合。替换 HMM 的窗分辨率与常数转移。预期：HCS 子窗边界（窗 ≈ 12 s）是 HMM 即整个方法的语料上唯一未碰的误差源，AP +.005~+.025 / ROC +.010~+.030；HateMM within 风险高（变化点是剪辑）。消融：β = 0、d 打乱、窗分辨率 vs 秒分辨率。先例 IOHMM / CHMM / Neural HMM E2E / HSLA。**有 0 GPU 预检**（posterior-alone HCS > .698）。Reviewer 分诊：融合第 2。needs manual pilot。
### 备选：R3 = C4 只收不发的可靠性注意力
- 方法：逐 key 可靠性偏置 log ρ_s（两粒度一致、HMM 熵、正下界），歧义秒只收不发。预期 HateMM AP −.020~+.015、HCS ROC +.005~+.020；within 风险高（候选 3 的 key 偏置先例）。先例 DDG-Net（ICCV 2023，几乎相同的非对称）。只在 R1 失败且 matched-stream 消融可负担时考虑。
### 备选：R4 = C10-X10 基数 CRF（块计数替代 OR）、R5 = C8 精度加权 PoE
- 记录为备选：X10 若 K4 ≈ OR 则无信息；C8 有校准暴露且两专家都读裁定。

## 7. 淘汰清单

| 候选 | 淘汰阶段 | 原因 |
|---|---|---|
| C2 query 门控跨模态注意力 | 分诊 | 项目内同机制臂（候选 4 `gated_cma`：query sigmoid 门 + 屏蔽 padding）已在 HCS 变差（.690 对 .706）、HateMM 持平；与 Leaky Gated Cross-Attention（WACV 2022）差异只剩局部裁定输入 |
| C5 长度稳定 NOP sink | 分诊 | 候选 4 `zero_value_sink` 臂在 HCS 等于屏蔽 padding（.685 对 .686）；被判 padding 工程 |
| C11 冲突感知证据 HMM | 分诊 | hate / non-hate / unknown mass 无法用视频标签标定；作用被 C7、C8 覆盖 |
| C12 后验监督双记忆 | 分诊 | UR-DMU 直接迁移；HMM 后验产生循环伪标签（项目已证明链/自蒸馏目标无益）；HCS 内容无信号 |
| C14 密度协变量 HMM 先验 | 分诊 | 同一裁定计数同时控制发射与 p0/A，双重计入；输出主要是逐视频后验平移（校准）；候选 2 密度条件链是负先例 |
| C3-R5 块级证据 register | 分诊 | 块内共享 register = 候选 3 的广播机制在块尺度重现，HateMM within 风险高 |
| C6-T3 裁定路由 FFN 专家 | 分诊 | HVGuard（EMNLP 2025）已在 hateful video 用 MoE 门控融合，规则 4 第一类暴露 |
| C10-R6 计数后验定 bag 大小 | 分诊 | 只改 pooling 的 bag 大小，属训练配置（规则 4 第四类） |
| C8 精度加权 PoE 融合 | 分诊（降为备选） | 两个专家都读裁定列，PoE 条件独立解释不成立；若精度收敛为常数则退化为重新搜索 prior_scale（learned calibration） |
| C9 内容条件化时间结构 | 分诊（降为备选） | HateMM within 风险高（内容变化点主要是剪辑）；HSMM 时长参数在 30 个格子上不可识别；只有 HCS 秒级边界一项有据可依 |

## 8. 试点实验

无（用户约束：全部 GPU 被预注册链占用；MAX_PILOT_IDEAS = 0）。所有推荐候选标 **needs manual pilot**。对 C7 与 C9 存在**不需要训练、不需要 GPU 的预检**（只重新拟合 HMM 并用统一评测器评 posterior-alone），本轮按约束未执行，写入实验计划作为第一步。

## 9. 精炼提案与实验计划

- 提案：`docs/idea-stage-20260905/refine-logs/FINAL_PROPOSAL.md`
- 实验计划：`docs/idea-stage-20260905/refine-logs/EXPERIMENT_PLAN.md`
- 跟踪表：`docs/idea-stage-20260905/refine-logs/EXPERIMENT_TRACKER.md`
- 研究契约：`docs/idea-stage-20260905/docs/research_contract.md`
- 精简候选表：`docs/idea-stage-20260905/IDEA_CANDIDATES.md`

## 10. 下一步

- [ ] 用户裁定是否进入实现（本报告无 pilot 证据；评审第 2 轮 5/10）。
- [ ] 写 `experiments/<date>_vcde/README.md`（机制、来源、搜索空间、预注册），规则 4 proposal review。
- [ ] Block A CPU 门（0 GPU）：A1 条件后验 vs K4-only / lookup；A2 q_v 比例有效性。不过门则模块 3 归档、模块 2 去比例损失。
- [ ] 实现（`src/verdict_hmm.py` 扩展条件发射与 OOF 拟合；候选 1 `train.py` 新头与臂）→ 规则 6 code review → Block B bit-match → Block C 搜索 → Block D 消融。
- [ ] 回传 `runs/` 后更新 `research-wiki/STATUS.md`。
