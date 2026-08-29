# LITERATURE: 跨领域顶会文献扫描 — MLLM 集成新路线(2026-07-13)

方法:3 个独立 Opus 文献 agent 并行扫描(视频理解 / 学习方法 / 跨领域应用),2023–2026 顶会,WebSearch 逐条核验 arXiv id 与 venue;未确认 venue 者带诚实标记。每个 agent 均给定 14 条死路线清单并要求非同构论证。配套 reflection 见 `REFLECTION_mllm_integration_failures.md`。
**用户裁定(2026-07-13,先于本文档定稿):不加 OCR 通道("没啥用")。** OCR 相关证据仅保留为 SOTA 校准,不作为候选。

## 1. 统一候选排序(三路合并,已应用 OCR 否决)

### C1 — RA-HMD 式 QLoRA 两阶段端到端(检索对比目标下微调 MLLM encoder 本体)【⚠️ 2026-07-13 当日证据 triage 后降级,见 §7——预期增量低于判决线,倾向 G0 处死】
- 锚文献:RA-HMD (Mei et al., **EMNLP 2025 Oral**, arXiv 2502.13061 — RGCL 同一血统的官方后续);CLIP-LoRA (CVPR**W** 2024, arXiv 2405.18541);DST (NeurIPS 2022 Oral, arXiv 2202.07136);DHO (preprint, arXiv 2505.07675, venue 未确认)。
- 机制:冻结 Qwen 特征 → **QLoRA 可训练** Qwen。Stage 1: QLoRA 任务适配;Stage 2: 冻结 backbone,训 MLP + triplet 对比 head;解耦双头避免监督/对比梯度打架(即 P9b 分数再分配的文献级解法)。
- 证据强度:RA-HMD 在 HatefulMemes 上 **+3.3 acc (82.1 vs 78.82,Δ=+3.28;scout 原报 +3.0 系舍入不一致,归档审计已标记,待原文核实口径) / +4.1 AUC** 对冻结特征 RGCL——正是我们管线的 meme 类比,达标线以上的直接先例。我们的 encoder swap(冻结)已 +5.3 HateMM,可训练化是其自然加深。
- ⚠️ 两个 scout 的口径分歧(预注册前必须读原文核实):methods-scout 报整条 QLoRA 管线 +3 acc vs 冻结 RGCL;cross-domain-scout 报 Stage-2 对比增量偏 OOD/跨集(in-domain FHM ≈ 冻结)。两者可同真(总增益 vs 阶段增量),需原文裁决。
- 资产:repo 内已有 `RA-HMD/LLAMA-FACTORY-Ver202512`(用户已在铺设);7B QLoRA 2×GPU 可行。
- 非同构:14 条死路线全部冻结 encoder 注决策侧;此路编辑表征流形本身 = 项目唯一 +3 杠杆的直接延伸。

### C2 — SAV 稀疏注意力头特征挖掘(冻结、便宜)
- 锚文献:Sparse Attention Vectors (**ICCV 2025**, arXiv 2412.00142)。
- 机制:弃用 Qwen 末层 pooled hidden state,挖出 <5% 对 hate 判别的注意力头,以其特征向量进 RGCL head + kNN。特征选择 = 表征级,单次冻结前向。
- 证据:小样本 + 安全域最强——比 LoRA 平均 +7%,VLGuard(有害内容安全基准)+62.9%,~20 样本/类即 SOTA,抗噪。
- 与本项目的独特契合:**给出了 encoder swap 在 MHC-EN 失败的可检验假说**(mean-pooling 稀释判别子空间)→ 修好即满足「≥2 数据集」。成本最低的高价值 pilot。

### C3 — MLLM 世界知识/推理密集文本作为新增表征通道(gate-first;非 OCR、非转写复述)
- 锚文献:Pro-Cap (ACM MM 2023, arXiv 2308.08088, meme +3~6 pts);Mr.Harm (Findings EMNLP 2023, arXiv 2312.05434);LaCLIP (NeurIPS 2023, arXiv 2305.20088);冗余对照:**"Does VLM Classification Benefit from LLM Description Semantics?" (AAAI 2025, arXiv 2412.11917)** — 证明描述文本增益大半是 semantic-agnostic ensembling(= P4 冗余的文献级形式化),真语义分量仅当文本区分嵌入邻域内实例时存活。
- 机制:MLLM 输出段落级推理(实体、暗语/dog-whistle、隐含 target、符号语境)→ 独立文本 encoder → 全维向量进融合表征。带宽 = 数百 token,输入通道而非决策信号。
- 风险:P4 陷阱的最大暴露面(transcript+title 已在输入中,OCR 已被否决,增量只能来自世界知识/推理)。**必须先过 §3 条件信息 gate,gate 不过零 GPU 处死。**

### C4 — 72B→小模型的对比表征蒸馏(CRD)
- 锚文献:CRD (ICLR 2020, arXiv 1910.10699,feature-KD > logit-KD);VLM-KD (arXiv 2408.16930, venue 未确认);反例警示 (arXiv 2511.17886, venue 未确认:更大 teacher ≠ 更好 student,须用关系/对比蒸馏)。
- 机制:72B 4-bit 离线抽特征(含无标注视频),小 head/LoRA-student 以 CRD 损失匹配 teacher 表征几何,再训 RGCL。非同构于 P1(logit/先验 = 决策侧;此为特征几何)。

### C5 — 无标注外部视频 + MLLM 伪标签:只喂表征训练,不进投票池
- 锚文献:kNN-LM (ICLR 2020, 1911.00172);"You can't pick your neighbors" (Findings EMNLP 2022, 2210.15859,污染警示);"Great Memory, Shallow Reasoning" (2408.11815);SuS-X (ICCV 2023, 2211.16198);TDA (CVPR 2024, 2403.18293);DST/FixMatch。
- 三路 scout 一致结论:直接扩 kNN 投票池 = 决策侧 + 低单位带宽 + 伪标签污染,预期 1–3 pt 且易沉入噪声地板 → **只作为 C1 的放大器**(表征训练数据扩充,置信度+自一致性过滤),不做独立 +3 赌注。需先解决域内无标注视频池的可得性。

### C6 — LaCLIP 式文本改写对比视图(正则化器,末位)
- 机制:MLLM 生成 K 个改写视图作对比正对,预训/正则融合表征。1k 规模下预期以正则化为主。

## 2. SOTA 校准表(cross-domain scout,协议差异需重点警惕)

### HateMM(~1,083 视频)
| 方法 | 成绩 | 家族 | 违禁信息? |
|---|---|---|---|
| ImpliHateVid (2508.06570) | 0.976 F1 | 对比+OFA caption+audio | 疑似随机切分,**离群值,不可比** |
| MM-HSD (2508.20546) | **0.878 acc / 0.874 M-F1** | Whisper+wav2vec2+ViT+PaddleOCR 四通道 CMA,无 LLM | 无(诚实监督 SOTA) |
| TCE-DBF (2024) | 0.876 micro-F1 | 四模态 CMA | 无 |
| HCC1 (2025) | 0.854 acc / 0.848 M-F1 | 晚融合 | 无 |
| TANDEM (2601.11178) | 0.78 acc | Qwen2.5-VL+Qwen2-Audio SFT+RL(LMM-as-reasoner) | 本地权重 OK,**落后融合 frontier** |
| **我们** | **~0.82 acc CLIP / 0.870 acc Qwen**(3-seed 均值 0.873 val-sel / 0.868 final-ep,exp-encoder-3seed.md:155-159;审计修正,原稿误写 ~0.88) | RGCL | 无 |

### MultiHateClip(1k EN + 1k ZH,test ~200/语言)
| 方法 | EN | ZH | 备注 |
|---|---|---|---|
| GPT-4V zero-shot (2408.03468) | 0.81 acc / 0.79 M-F1 | 更弱 | **外部 API,我们禁用** |
| 最佳多模态融合 (mBERT⊙MFCC⊙ViViT) | 0.75 / 0.74 | **0.80 / 0.78** | 诚实 frontier |
| LLaMA-3.2-11B | ~0.78 M-F1 | — | 本地 LMM |
| **我们** | **~0.79–0.81 acc** | **~0.85 acc**(归属修正 2026-07-14:0.8537±0.012 是 **LoRA-Qwen** 配置的 final-epoch 均值,非 frozen-CLIP floor;frozen-CLIP ZH floor = 0.8027–0.8143,≈ published frontier。见 refine-logs/B1_PREREG_REVIEW.md Task A) | LoRA 配置超 frontier;CLIP floor ≈ frontier |

**校准结论(审计修正后):** MHC-ZH 我们已超已发表最好成绩;MHC-EN 在 frontier 上;HateMM Qwen floor 0.870 **略低于** MM-HSD 0.878(差距 ~0.8 pt,MM-HSD 的增量来自音频 wav2vec2 + OCR 两个通道,后者已被用户否决,前者我们管线目前没有音频通道——这是一个被文献标定的、未被 14 条死路覆盖的表征缺口)。+3 acc ≈ 设立新 SOTA——目标在 frontier 之外,难度校准清楚。**结构性佐证:2024–2026 每一个诚实 HateMM 增益都来自加表征通道;LMM-as-reasoner 分类器(TANDEM 0.78)反而落后监督融合。** 与 reflection D2 独立吻合。RecSys 文献同构结论:LLM-as-ranker 在 warm/大数据 regime 对强协同特征冗余,LLM 只在表征注入时有效 (A-LLMRec 2404.11343; 2505.20730)。

## 3. 条件信息 Gate 配方(G0-cond,制度化,零 GPU)

1. **估计器**:conditional V-information / conditional probing (Hewitt et al., EMNLP 2021, 2021.emnlp-main.122;V-info: ICLR 2020, arXiv 2002.10689)。训 g(Z) 与 g'([Z,A]),A 的可用条件信息 = codelength 差。≈0 → 冗余,处死。
2. **四项校准**(修正历史 probe 说谎的原因):(i) 探针容量与实际部署 head 匹配(过强探针 = "probe 过、训练平"的直接机理);(ii) 用 **MDL codelength** 而非 accuracy (Voita & Titov, EMNLP 2020);(iii) bits→acc(Fano/经验斜率)换算,投影增益须 > +3 acc + ±1–2 pt 噪声带;(iv) 无选择协议 + 多 seed bootstrap CI 排除 0。
3. **oracle 上限杀开关**:先用 gold 版信号(合规:gold 仅限 probing)测上限;oracle 条件增益 < +3 → 信号族整体处死,MLLM 质量无关。
4. 决策规则:两条同时满足才允许花 GPU。

## 4. 三个自研想法的文献裁决

- **(a) QLoRA 端到端检索对比微调 — 三路一致 SUPPORT(最强)**,即 C1;附 RA-HMD in-domain/OOD 口径分歧待原文核实;SAV 论文警示 tiny-data LoRA 过拟合风险 → 低 rank、Stage-2 冻结 backbone、无选择协议监控。
- **(b) 密集 caption/rationale 新增通道 — QUALIFIED SUPPORT**,即 C3;AAAI 2025 (2412.11917) 给出增益大半是 ensembling 的一般性证明(且 ensemble 类增益撞我们的「禁 ensemble」规则),OCR 已被用户否决 → 只剩世界知识/推理分量,gate-first 或死。
- **(c) 伪标签外部视频扩 kNN 记忆 — 三路一致 REFUTE-as-posed / REDIRECT**,即 C5:决策侧投票池扩充 = 低带宽 + 污染风险 + 噪声地板;重定向为表征训练扩充。

## 5. 反推荐(合并,永久记录)

- LAVAD 式 training-free LLM 打分聚合;SlowFastVAD 慢速 VLM 复判 → P1/P2/P3/P10 同构。
- Evolver 式 LMM-as-classifier over 检索邻居 → P2+P10 同构;视频域实证落后融合 frontier(TANDEM)。
- MIL 段伪标签作弱监督 → P11 同构。Holmes-VAD 式指令微调出结论 → 低带宽结论 + 需 span 标注(违禁)。
- 概念瓶颈/属性打分 (LaBo, 2211.11158) → few-shot 专属优势,1k 下 = P4 同构风险。
- 描述符 prompting (Menon-Vondrick) → 增益 = ensembling (2412.11917)。
- logit/软标签蒸馏 → P1 同构;蒸馏要蒸表征 (CRD)。
- 跨域 TTA headline 数字 (SCANNER +3~5) → 域偏移协议,不可比;ImpliHateVid 0.976 → 切分离群。
- 关键帧重采样单独下注 (AKS, 2502.21271) → 源增益 1.6–2.5 pt < 噪声地板,除非 hate 强局部化。

## 6. 建议的 C 线执行顺序(GPU 与 A 线 M2/M3 交错)

1. **C2 SAV pilot**(最便宜,冻结前向 + 头选择;同时检验 MHC-EN 失败假说)与 **C1 预注册起草 + RA-HMD 原文口径核实**(零 GPU)先行;
2. C1 QLoRA 两阶段 = 头号 GPU 赌注(预注册 → 冻结 → 单提);
3. C3 密集推理文本:先 G0-cond gate(零 GPU),gate 过才排队;
4. C5/C6 仅作 C1 放大器。
A 线 lb_scgp_global 继续走完 M2→M3 干净判决,失败即切 C 线,不空转。

## 7. 当日增补:C1 证据 triage(prep agent 读 RA-HMD 原文 + repo 史比对,2026-07-13)

**Scout 口径分歧已裁决(arXiv 2502.13061v1 全文,Qwen2VL-7B/HatefulMemes,Table 1 + Table 3):** 两个 scout 都对。全管线 in-domain 对冻结 RGCL = +4.06 AUC / +3.28 acc;但消融显示 **增益几乎全部来自 Stage-1 LoRA-SFT 表征适配**(去掉 Stage-1:−6.7 AUC/−7.9 acc),**Stage-2 检索对比 in-domain 只值 +0.9 AUC / +0.7 acc**(其真实价值在跨域:cross-domain 去掉 Stage-2 −3.7 AUC/−7.6 acc)。(WebFetch HTML 读数,复审时对 PDF 再核一遍小数位。)

**与本项目历史对撞(关键):** 视频上的 LMM-RGCL 我们已经跑过两次并杀掉——**P9**(Stage-1 LoRA-SFT):MLP head ≈ 冻结 floor(+0.6/+1.0/+0.9,噪声内),kNN 反而**低于** floor(−2.7/−2.2/−4.7);**P9b**(RGCL 联训):0/12 cell 过 floor。C1 唯一真正未测的 cell = Stage-1 LoRA → 抽适配特征 → **顺序**训 RGCL 对比 head(P9 用的是无 Stage-2 head 的裸 kNN;P9b 是联训非顺序)。而 RA-HMD 自己的消融给这个 cell 的 in-domain 定价 ≈ **+0.7 acc**(还是 8.5k memes 规模;我们 549–744 视频)——低于 +3 判决线,也低于 ±1–2 pt 噪声地板。

**Triage 结论(待 fresh 审稿确认):C1 在 G0 阶段处死**,理由 = 自家 P9/P9b 负结果 + 锚论文自己的消融天花板;不花 GPU。**C 线头号候选变更为 C2(SAV)**,C3(gate-first 密集文本)次之。此 triage 正是 G0-cond 制度的第一次实战执行。

**资产事实(prep agent 实核):** 本地唯一完整 VL checkpoint = **Qwen2.5-VL-7B-Instruct(16 GB)**;32B/72B 只有锁文件无权重(此前下载日志有欺骗性),Qwen3-VL 仅元数据 stub → C 线所有候选当前都是 7B-only;C4(72B 蒸馏)在补下载(290G 配额内)之前不可行。SAV 所需的 per-(layer,head) 特征不在现有缓存中(现缓存仅末层 mean-pooled 3584-d),需一次新的冻结前向抽取。P9 已验证 8 帧 Qwen2.5-VL-7B LoRA 单张 80GB A100 可跑。
