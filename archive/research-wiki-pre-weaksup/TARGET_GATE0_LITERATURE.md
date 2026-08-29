# Target-driven Gate 0：MLLM 机制级整合的文献地图与查新

**检索日期：** 2026-07-10（Pacific/Auckland）  
**阶段：** Gate 0，文献与查新；未运行 GPU、未改实验代码、未接触测试集  
**目标：** 让 MLLM/VLM 不只是冻结编码器、额外特征或最终裁判，而是以可消融的机制进入 RGCL/RA-HMD hateful-video classifier，并在同一协议下带来 **substantial final accuracy 与 macro-F1 增益**。

## 0. 结论先行

Gate 0 后保留 **恰好三个**候选。初检曾给 A 更高 expected value；第二位架构审阅者作 adversarial novelty attack 后，最终优先级改为 **B > A ≫ C**：

| 候选 | MLLM 的不可替代作用 | 直接改变什么 | 查新判断 | Gate 1 建议 |
|---|---|---|---|---|
| **A. CCGC-RGCL：反事实模态联盟控制的检索梯度** | 对同一视频的模态干预作相对因果判别 | RGCL listwise retrieval gradient、将写入 memory 的表示几何及融合交互 | **中等新颖，风险中等偏高** | 第二顺位；必须保持 retrieval-specific |
| **B. SR-RGCL：语义关系签名的对比图** | 从跨视频对中抽取 target/stance/mechanism 关系；gold label 决定边符号 | RGCL 的正负样本拓扑与关系条件度量 | **中等新颖，三者中最可守** | **首选** |
| **C. RuleRoute：可执行规则条件的低秩专家路由** | 归纳、验证并标注可组合仇恨谓词/规则 | 样本经过哪一个融合适配器及规则一致性 | **中等偏低新颖，风险高** | 仅作第三顺位高风险方向 |

最重要的判断是：**不要再给现有分类器添加一个 MLLM 语义通道；要让 MLLM 改变训练时“哪些模态交互应该产生梯度、哪些跨样本关系应该塑造几何、或哪条可执行规则应该选择参数路径”。** 这三者才与已失败的 score、concat、memory reallocation、localizer、guard-rail 路线有机制差异。

### 本轮对“substantial”的保守操作化

项目以往将约 1 acc point 视为约 1.6 个测试视频的噪声地板。Gate 0 暂以更严格的 **同协议绝对提升 ≥2.0 percentage points，accuracy 与 macro-F1 两者都达到，且 ≥2/3 seeds 同号、无其他主数据集 >1 point 伤害** 作为最低 substantial 线；最终仍应由主循环统一预注册。

当前权威 floor 来自本地 [PAPER_MASTER_TABLES.md](./PAPER_MASTER_TABLES.md) 与 [DRAFT_experiments_chapter.md](./DRAFT_experiments_chapter.md)：

- HateMM：`acc/macro-F1 = 0.870/0.861`（clean n=215 口径）；
- MHC-EN final-epoch：`0.7888/0.7488`；
- MHC-ZH final-epoch：`0.8537/0.8259`。

因此任何新方向不能靠亚百分点波动宣称成功。

---

## 1. 权威本地证据：什么已被项目自己证伪

本轮先读了本地 campaign 总结、P1–P11 实验文档、主表与既有查新，并将以下路线设为硬排除区：

1. **最终分数/先验/裁判注入失败。** P1、P7 说明 MLLM verdict 随 era 漂移，且 semantic score 与强通道正相关而更弱；不能再做 score calibration、rank averaging、veto/boost 的换名版本。
2. **简单 embedding concat / 文本摘要 / 语义字段蒸馏失败。** P3、P4、P8 表明真实且可解码的语义信号会被已监督的融合头吸收，或与 gold label 冗余；“多给一条 embedding/auxiliary label”不是机制创新。
3. **邻居 rerank / KNN memory 再分配失败。** P2/P2b 中 topical comparability 与 vote correctness 几乎正交；P9/P9b 中 LMM head 与 kNN memory 只发生精度再分配而无净增益。
4. **文本反事实 hard negative 失败。** P5 的 sanitized twin 质量门不通过，且共享原视觉使 repulsion 反而伤害表示；候选不能再依赖自由文本改写造“干净负样本”。
5. **定位器与弱监督定位不是主目标。** P6/P10 只获得 modest localization，P11 又证明 teacher edge 很大部分来自 coarse×fine aggregation；本轮候选必须以 final video-level acc/macro-F1 为因变量。
6. **guard-rail/audit 不是成功替代品。** 它们可保留为论文贡献，但不能充当本目标要求的 final classifier 增益。

详细证据见 [CAMPAIGN_mllm_method_role.md](./CAMPAIGN_mllm_method_role.md)、[EXP_p9_lmm_rgcl_video.md](./EXP_p9_lmm_rgcl_video.md)、[EXP_p11_weaksup_localization.md](./EXP_p11_weaksup_localization.md) 与 [PAPER_MASTER_TABLES.md](./PAPER_MASTER_TABLES.md)。

---

## 2. 2024–2026 直接竞争地图

这里只列原论文、ACL Anthology、OpenReview 正式页面或作者 arXiv；不以二手新闻/博客支撑方法判断。

| 工作 | 状态与时间 | MLLM/VLM 的角色 | 对本项目的含义 |
|---|---|---|---|
| [RGCL (Mei et al., ACL 2024)](https://aclanthology.org/2024.acl-long.291/) | ACL 2024 | 检索引导对比学习；非视频 | 本项目的几何与 memory 原点，但没有 MLLM 因果/关系/规则教师 |
| [RA-HMD / Robust Adaptation (Mei et al., EMNLP 2025)](https://aclanthology.org/2025.emnlp-main.1215/) | EMNLP 2025 Main Oral | 整体 LMM SFT + retrieval-guided contrastive | 本地 P9/P9b 已证明直接 port 到视频只会 head↔memory 再分配 |
| [RAMF (Yang et al.)](https://openreview.net/forum?id=U9KnNiuMu1) | TMLR 2026-06 accepted | Qwen2.5-VL-32B 生成 objective / hate-assumed / non-hate-assumed 三路推理文本，再以 LGCF+SCA 融合 | **最直接竞争者。** 证明结构化对立推理可能增益，但仍把 reasoning 当额外语义输入；候选 A/B/C 必须超越这一用法 |
| [MARS (Yang et al.)](https://arxiv.org/abs/2601.15115) | arXiv 2026-01 | 训练免费四阶段 adversarial reasoning，直接输出判决 | 占据“多视角 prompt 判决”路线；本轮不再做 inference-time judge |
| [IARE (Lu et al.)](https://arxiv.org/abs/2606.11953) | arXiv 2026-06（文中为 SIGIR 2026 工作） | MLLM 以 SFT 学 contextual rationale，再用正确/错误 reasoning path 做 DPO | 其 Qwen2.5-VL-7B 相对 SFT 报告 `+4.28 macro-F1`（Ex-HateMM）；说明 reasoning supervision 有量级，但它训练生成式 backbone，未控制 RGCL 梯度/关系/路由 |
| [SCANNER (Li et al.)](https://arxiv.org/abs/2602.00132) | AAAI 2026 accepted | 无 MLLM；以 invariant hateful cores 做 source-free TTA centroid alignment | 报告相对最佳 baseline 平均 `+4.69 macro-F1`；说明“稳定核心+结构约束”可能有 substantial 量级，但不是本轮机制 |
| [TANDEM (Koushik et al.)](https://arxiv.org/abs/2601.11178) | arXiv 2026-01 | VLM 与 audio-LM 串联 RL，输出 timestamp/target | 结构化 reasoning + RL；主要目标是 grounding/target，不是 RGCL final classifier 的轻量机制 |
| [LELA (Sun et al.)](https://arxiv.org/abs/2602.09637) | arXiv 2026-02 | 五模态 caption + 多阶段 prompting + composition matching | 占据 training-free localization；不能把 localization 换名当主表创新 |
| [CGO: Controlled Gradient Optimization](https://openreview.net/forum?id=Z51RWOPKQQ) | ACL ARR 2026-01 submission | 无 MLLM；方向对齐、扰动可靠性加权、跨模态收敛协调 | **A 的最近优化邻居。** 它控制梯度但可靠性不来自 MLLM 的同样本因果干预差分 |
| [BridgeVLM](https://openreview.net/forum?id=NOoAIwF6bV) | ICML 2026 OpenReview entry | 从多图输入诱导 causal graph，转为模型内部 causal tokens / RAMP 消息传递 | **A/C 的相邻机制。** 任务为通用多图因果推理，不是 hateful-video 分类或 RGCL；提醒不能泛称“首次 causal token” |
| [Diagnosing Modality Interference](https://openreview.net/forum?id=0Cv0whP7l8) | ICLR 2026 submission | 通过因果干预诊断 modality interference，并用扰动一致性缓解 | **A 的一般领域祖先。** 差异必须落在 MLLM teacher 的 coalition interaction 和 RGCL gradient controller |
| [VC-STaR](https://openreview.net/forum?id=ZymCPON45y) | ICLR 2026 Oral | 用视觉相似对比 pair 提高 VLM rationale 质量并 SFT | **B 的相邻思想。** 它用 pair 帮 VLM 自提升，不用语义关系签名塑造下游 RGCL 几何 |
| [HateSieve (Su et al., Findings NAACL 2025)](https://aclanthology.org/2025.findings-naacl.289/) | peer-reviewed 2025 | Contrastive Meme Generator 造语义相关 triplet，训练 CLIP 对齐 | **B 的重要威胁。** 已占据“LMM/生成式 contrastive triplet”在 meme 的大类；B 必须坚持“不生成样本、gold-label signed typed edges” |
| [Multimodal Representation Learning Conditioned on Semantic Relations](https://openreview.net/forum?id=zAtrBcGsyf) | ICLR 2026 submission | 多对多语义关系 + relation-guided cross-attention | **B 的最接近通用工作。** 因而 B 只能主张 hateful-video/RGCL 中具体 signed stance-target mechanism，不可主张一般 relation-conditioned representation 首创 |
| [BPDMoE-Hate](https://openreview.net/forum?id=lMDKFriaJc) | ACL ARR 2026-01 submission | VLM 生成二元对立视角，adaptive viewpoint gating + 双空间 MoE | **C 的重大威胁。** 已覆盖“VLM perspectives 路由 MoE”在 harmful meme；C 必须依靠可验证谓词逻辑与低秩参数路由，而非观点 embedding gate |
| [SyRHM](https://openreview.net/forum?id=6tKlqJaBAQ) | ACL ARR 2026-01 submission | harmful meme 的 retrieval + symbolic translator/planner/solver | **C 的重大威胁。** 已占据 inference-time symbolic harmful-content reasoning；C 不能主张 harmful-content 神经符号第一 |
| [LangCBM](https://openreview.net/forum?id=aswvIu1Vgw) | ICLR 2026 workshop | VLM 文本概念作为 concept bottleneck | **C 的相邻工作。** 简单“概念→分类器”已不新；必须是规则可执行且改变 adapter path |
| [NS-Mem](https://arxiv.org/abs/2603.15280) | arXiv 2026-03 | 三层 neuro-symbolic memory + rule DAG + hybrid retrieval | 报告相对纯神经 memory 平均 `+4.35 accuracy`，但为 multimodal agent reasoning；只提供机制可行性旁证，不是可迁移结果保证 |

额外的可靠性警告来自 [MuCR (Li et al., Findings ACL 2025)](https://aclanthology.org/2025.findings-acl.288/) 与 [ACL 2026 “Look Light, Think Heavy”](https://aclanthology.org/2026.acl-long.387/)：MLLM 的跨模态因果推理和视觉反思并不天然可靠。因此三个候选都必须使用 **相对干预、gold-label 定符号、或可执行 verifier** 限制自由文本 hallucination，不能把 CoT 当真值。

不同论文使用不同 split、fold 与数据可用性，以上外部绝对数值 **不可** 与本项目 test cell 直接横比；它们只用于判断机制量级与近邻覆盖。

---

## 3. 候选 A：CCGC-RGCL（Counterfactual Coalition Gradient Control for RGCL）

### 核心可检验 claim

**同一视频内，MLLM 对确定性模态干预的“相对证据变化”比它的绝对 hate score 更稳定；把该 train-only privileged interaction signature 直接用于 RGCL 的 listwise retrieval gradient 与 memory-writing geometry（而非只调 fusion 权重），会减少 spurious-neighbour attraction，并带来 ≥2 points 的 final acc 与 macro-F1 增益。**

### Mechanism

对训练视频只构造确定性、无生成内容的 modality coalitions，例如：

- full `V+A+ASR/OCR`；
- leave-one-out：`-V`、`-A`、`-ASR/OCR`；
- 少量双删除；
- temporal shuffle 作为“内容还在但同步关系破坏”的交互干预。

冻结 MLLM 在同一个视频的这些版本上，以对立假设提示输出 **证据支持度**，但不采用其绝对标签。定义 teacher 的一阶必要性与二阶 synergy：

`N_m^T = h_T(x) - h_T(x\m)`；  
`I_mn^T = h_T(x)-h_T(x\m)-h_T(x\n)+h_T(x\{m,n})`。

学生仍是 RGCL/RA-HMD classifier。新增三项紧耦合机制：

1. **interaction ranking loss**：学生 coalition logits 的 `N_m^S`、`I_mn^S` 只匹配 teacher 的符号/排序，不蒸馏绝对 label；
2. **coalition-conditioned listwise RGCL**：对 query–neighbour 的 retrieval list，只有 causal signature 相容的 same-label pair 获得完整 attraction；signature 冲突的 neighbour 即使表面相似也被降权，权重仅作用于训练期 RGCL 梯度，绝不在测试时重排投票；
3. **memory-write consistency**：full-view 表示与 teacher 判为 sufficient 的最小 coalition view 必须写入同一局部几何，而 irrelevant/spurious coalition 不得主导最终写入 memory bank 的 full-view embedding。gold classification loss 始终是最终监督。

训练后可丢弃 MLLM；消融为 `λ_interaction=0`、listwise weight shuffle 与 memory-write consistency off。MLLM 的移除会失去“哪种因果 coalition 应塑造 retrieval geometry”的训练信号，而不是只少一条 embedding。

### 与最近工作的精确差异

- **vs RAMF/MARS/IARE：** 它们生成 reasoning text 或直接生成 label；CCGC 不把文本接入 classifier，也不做最终 judge，而从 **同样本 deterministic interventions 的差分** 得到训练约束。
- **vs CGO：** CGO 从学生扰动、梯度一致性和不确定性控制普通 multimodal optimization；CCGC-RGCL 的目标由外部 MLLM coalition 差分给出，并直接控制 **retrieval list 中每个 neighbour 对 embedding/memory geometry 的梯度贡献**。
- **vs BridgeVLM / modality-interference work：** 它们研究通用 VLM 内部 causal token/干预鲁棒性；CCGC 将该思想用于 **小样本 hateful-video 的 RGCL fusion gradient**，目标是 final acc/macro-F1。
- **vs 本地 P3/P4/P7：** 不做 segment score pooling、field auxiliary labels 或 score fusion；监督对象是 fusion interaction 和 gradient geometry。

### Novelty 风险

**评级：中等偏高风险，约 6.5/10；建议 PROCEED WITH CAUTION。** 未检到“MLLM coalition delta → hateful-video RGCL listwise/memory gradient”的直接重合。风险来自 CGO、BridgeVLM 与通用 modality-interference 文献已经分别占据 gradient control 和 causal intervention 两半；若实现退化成“用 delta 调 fusion weight”，会被合理视为 RAMF/IARE/动态模态融合的变体。论文必须证明 retrieval-specific 组合不是拼装，关键证据是：

1. 相对 coalition teacher 比绝对 score 更稳定；
2. teacher interaction 与 baseline **wrong-neighbour attraction/listwise gradient conflict** 有统计对应；
3. gain 在关掉 teacher 或打乱 teacher interaction 后消失。

### 预期 final 增益（假设，不是既有结果）

若因果故事成立，预期在语音/视觉互补且 shortcut 明显的 MHC-EN 上最高：

- acc `+2.0–3.5 pt`，macro-F1 `+2.5–4.5 pt`；
- MHC-ZH / HateMM acc 与 macro-F1 各 `+1.5–3.0 pt`，但 substantial pass 仍要求实际两指标都 ≥2.0 pt；
- 对应 EN final 目标区间约 `0.809–0.824 acc / 0.774–0.794 macro-F1`。

这些范围由 IARE 相对 SFT 的 +4.28 F1、SCANNER 的 +4.69 F1 和本地 1-point 噪声地板共同约束；不是把外部 split 数字直接迁移。

### Fast-fail 观察

在任何完整三 seed 训练前，必须看到：

1. 约 80–120 个 train-only 视频上，prompt paraphrase / 两次独立解析的 `sign(N_m^T)` 一致率 ≥70%，且 absolute verdict 不稳定时 relative ordering 仍稳定；
2. teacher 指为“必要”的删除比随机删除产生更大的 gold-margin 降幅，至少在两个 modality/error strata 同号；
3. 最小 seed-0/短训练中，interaction-on 相对 interaction-shuffled control 的 dev macro-F1 至少 `+1.0 pt`，并降低 baseline 错例上的 wrong-neighbour attraction 与 retrieval-gradient conflict。

任一失败即标 `not_working`，不靠换更大 MLLM、更多 frames 或更多 epoch 挽救。

### 为什么不是规模工程

干预集合固定且很小，teacher 冻结、只离线标 train，学生新增的是低维 interaction loss/listwise controller；无模型扩容、无额外数据、无 ensemble。科学变量是 **相对因果监督是否能纠正 RGCL retrieval gradient 与写入 memory 的表示几何**。

---

## 4. 候选 B：SR-RGCL（Semantically Signed Relational RGCL）

### 核心可检验 claim

**本地 P2 失败不是“关系无用”，而是让 MLLM 猜 vote correctness；若 gold label 固定边的正负号，MLLM 只负责抽取跨视频的 stance/target/mechanism 关系类型，则 typed hard pairs 能向 RGCL 提供普通 label-SupCon 没有的条件不变性，并实质改善 hate-vs-offensive/benign 边界。**

### Mechanism

对 train split 中现有近邻和 baseline hard-error pairs，让 MLLM 输出受限结构：

- `target_group`；
- `harmful_proposition / mechanism`；
- `speaker_stance ∈ {endorse, quote, condemn, satire, unclear}`；
- `evidence_binding`（哪种模态承载 proposition、哪种模态确定 stance）。

**MLLM 不决定边符号。** 边符号由 gold label 与关系模板确定：

- 同 target/相似表述但 stance 相反且 gold label 不同 → `counter-stance hard negative`；
- 同 mechanism、gold label 相同、surface/target 不同 → `mechanism-invariant positive`；
- 同 topic 但 target/intent 不同 → `topic-confound negative`；
- 解析不一致/unclear → 不建边。

在原 RGCL loss 上增加 relation-conditioned metric `W_r` 或小型 relation adapters，使不同关系拥有不同 margin；kNN memory 仍只作原本 readout，不让 MLLM 在测试时 rerank。消融包含 edge type shuffle、只用 gold label 的普通 SupCon、以及去掉 MLLM typed edges。

### 与最近工作的精确差异

- **vs RGCL/RA-HMD：** 原方法按标签和 embedding retrieval 建正负样本；SR-RGCL 进一步区分“为何相似/为何应分开”，在同标签/异标签内部加入 typed semantic geometry。
- **vs P2/P2b：** P2 让 MLLM 对测试邻居判 comparable 并删票，且 MLLM 需隐式猜 vote correctness；SR-RGCL 只在 train 建边，**gold label 决定 sign**，MLLM 只提供关系类型。
- **vs P4：** P4 每样本预测 schema field，容易与 label 冗余；SR-RGCL 的监督是跨样本的 conditional relation，不能由单样本 label 还原。
- **vs HateSieve：** HateSieve 生成新的 contrastive meme triplets；SR-RGCL 不生成任何视频/文本，只从真实训练样本中建立 verified typed edges。
- **vs relation-conditioned multimodal representation learning：** 通用工作已覆盖 relation-conditioned representation；本候选的新颖点只能是 **hateful-video 中 stance-target-mechanism 的 gold-signed graph 与 RGCL memory geometry**。
- **vs VC-STaR：** VC-STaR 用视觉相似 pair 帮 VLM 生成更可靠 rationale；本候选用 MLLM 关系来训练专用 classifier 的边界，不训练 reasoning model。

### Novelty 风险

**评级：中等偏高，约 6/10；PROCEED WITH CAUTION。** 没检到完全相同的 hateful-video 方法，但“关系条件表示”“rationale-guided contrastive”“生成 hard triplet”都已有近邻。能够守住的核心 delta 很窄：**真实视频、无内容生成、gold-label signed、typed stance relation、直接改 RGCL loss**。

若最后只剩 `target_group/mechanism` 字段做 supervised contrastive，它会退化成 P4 的关系化版本，novelty 与效果都不够。

### 预期 final 增益（假设）

- MHC-EN：acc `+2.0–3.0 pt`，macro-F1 `+2.0–3.5 pt`；
- MHC-ZH：两指标 `+1.5–3.0 pt`；
- HateMM：由于 explicit cases 已高，预计两指标 `+1.0–2.5 pt`。

其上界主要来自 P2 oracle membership editor 的大 headroom（EN +7.5 acc、ZH +10.6 acc），但这里不假设拿满 oracle；只假设 typed training geometry 能回收其中约四分之一到三分之一。

### Fast-fail 观察

1. 在约 200 个 train-only hard pairs 上，受限双提示/双向 pair order 的 edge-type agreement ≥80%；
2. `counter-stance` 和 `topic-confound` 边必须显著富集 baseline 错误，且人工抽审 precision ≥80%；覆盖率若 <15% hard pairs，则机制难以产生 substantial gain；
3. 冻结表示 probe 中，typed-edge score 对错误近邻的识别必须显著优于 P2 的 generic comparability；
4. seed-0 最小训练须使 dev macro-F1 相对 `label-only SupCon` 至少 +1.0 pt，且 `edge-type shuffle` 收回该增益。

任何一项失败即停止，不通过把 K、margin、边数扫成大网格追分。

### 为什么不是规模工程

只在现有 train pair 上增加稀疏关系标注与少量 relation parameters；数据、encoder、epoch、memory size 不变。因变量是 **typed relation 是否修复边界几何**，不是“多生成数据/多跑模型”。

---

## 5. 候选 C：RuleRoute（Verified Rule-conditioned Adapter Routing）

### 核心可检验 claim

**仇恨标签依赖可组合关系（protected target × derogatory proposition × endorsement/context），而非独立 schema 字段；让 MLLM 归纳并验证小型逻辑 clause，再用学生 predicate 激活结果选择融合 adapter，能形成多个条件决策边界，避免单一 head 吸收掉 MLLM 语义。**

### Mechanism

MLLM 只在 train split 完成两件受限任务：

1. 从 gold-labeled examples 中归纳最多 4–6 条 clause，例如：
   - `protected_target ∧ derogation/dehumanization ∧ endorse → hate`；
   - `quoted_or_condemned ∧ no_endorse → non-hate/offensive`；
   - `slur ∧ no_protected_target → offensive/non-hate`；
   - `visual_target ∧ speech_proposition ∧ temporal_binding → cross-modal hate`。
2. 为训练样本抽取 clause predicates 与 evidence binding；规则只有在 train-only support、precision、反例覆盖都过阈值时才保留，无法验证的输出进入 residual expert。

学生学习 predicate detectors。可微 rule layer 计算 clause activation；activation 只选择/混合 3–4 个小型 low-rank fusion adapters，随后仍由 RGCL loss 与 gold classification loss训练。推理时使用学生 predicate/router，不需要 MLLM 自己给 final score。消融为 rule shuffle、uniform router、无 predicate consistency、无 MLLM rule induction。

### 与最近工作的精确差异

- **vs P4 schema distillation：** P4 预测独立字段作为辅助目标；RuleRoute 执行字段间 conjunction/negation，并改变实际参数路径，`target` 或 `mechanism` 单独不等于 label。
- **vs RAMF/BPDMoE-Hate：** 它们将对立观点 embedding 融合或 gate MoE；RuleRoute 的 gate 来自受 verifier 约束的原子谓词和可执行 clause，不是 viewpoint embedding。
- **vs SyRHM/MemOracle/NS-Mem：** 它们在 inference-time 做 symbolic retrieval/reasoning；RuleRoute 将规则蒸馏成小学生的 **trainable adapter routing**，最终仍是 RGCL classifier。
- **vs LangCBM：** 普通 concept bottleneck 是 concepts→classifier；这里是 verified logical composition→parameter route，并要求 rule intervention consistency。
- **vs SCANNER：** SCANNER 用 stable core centroid 做 TTA；RuleRoute 是训练期 clause-conditioned decision functions，无 target-domain adaptation。

### Novelty 风险

**评级：高风险，约 5/10；仅保留为候选 C。** 2026 年 neural-symbolic、concept bottleneck、harmful-meme symbolic reasoning 和 viewpoint MoE 已很拥挤。只有“verified hate clauses + student predicate + low-rank fusion route + RGCL”这一完整组合尚未检到直接重合，但 reviewer 很可能评价为已有模块组合。

要提升新颖性，Gate 1 必须给出一个明确的机制命题，例如：**单 head 无法同时实现 endorsement 与 quotation 两种条件边界，而 rule-conditioned low-rank subspaces 可以；router intervention 会按预测方向改变 margin。** 若不能形成此类可证伪机制，应优先放弃 C。

### 预期 final 增益（假设）

- MHC-EN：acc `+2.0–4.0 pt`，macro-F1 `+2.5–5.0 pt`；
- MHC-ZH：两指标 `+2.0–3.5 pt`；
- HateMM：两指标 `+1.0–3.0 pt`。

范围较宽反映高方差：若错误确实来自 quotation/satire/target composition，收益可能大；若 predicate extraction 重复 gold label 或 clause coverage 低，则应接近 0。

### Fast-fail 观察

1. train-only 规则必须同时达到：support ≥10% 样本、precision ≥80%、至少覆盖 20% baseline errors；否则不具备主表影响面；
2. predicate paraphrase/双模型或双提示一致率 ≥80%，且 `stance` 不能退化成直接复制 gold label；
3. 简单 rule-only probe 必须优于独立 schema-field probe，并在 quotation/satire/implicit strata 上有净增益；
4. seed-0 最小训练中，rule router 相对 uniform router 的 dev macro-F1 ≥+1.0 pt，且 rule shuffle 破坏增益；同时每个 expert 有非退化 usage。

未达到即停止；不增加 expert 数、不换更大模型、不用 test 调 clause。

### 为什么不是规模工程

规则数、predicate 数、adapter rank 与 expert 数全部预先小规模封顶；MLLM 只离线读训练样本。核心实验是 **逻辑组合是否需要条件参数路径**，不是靠大 MoE 容量或额外训练数据。

---

## 6. 排序、决策与下一 Gate 的建议

### Expected-value 排序

1. **先做 B。** 它利用 P2 oracle headroom，但从根本上取消让 MLLM 猜 vote correctness 的任务；第二位审阅者认为 `gold-label signed nuisance-matched hard-pair topology` 是三者中最可辩护的 novelty。主要风险是通用 relation-conditioned representation 与 HateSieve 已使窗口收窄，因此必须守住 signed stance graph 的精确定义。
2. **B fast-fail 后做 A。** A 直接利用“MLLM semantic competence 真实但 final score 不可靠”：丢掉绝对 verdict，只保留同样本干预差分；但必须直接控制 RGCL listwise/memory gradient，若只调普通 fusion 即失去 novelty。
3. **C 只作第三顺位。** 它的潜在效果最大但 literature crowding 与实现自由度也最高，最容易变成模块堆叠；必须先过严格 rule coverage/precision gate。

### 与代码审计候选 1：`evidence-directed memory writing` 的关系

- 它与 **A（CCGC-RGCL）是同一机制族**：两者都试图让 MLLM evidence 决定什么表示可以塑造/写入 retrieval memory；它 **不是** 当前最高 EV 的 B。
- **可接受的等价实现：** evidence 必须是 train-only、由同一样本 deterministic modality interventions 得到的 relative coalition signature，并通过 listwise RGCL gradient / sufficient-view consistency 改变最终 full-view embedding 的 memory geometry。
- **不可接受的退化实现：** 若只是把 MLLM evidence score、rationale embedding、segment weight 或 schema field 用来选/加权 memory entry，它分别退化为 P2 rerank、P3 pooling、P4 semantic auxiliary supervision 的近邻，既难 substantial，也难守 novelty。
- **与 B 的关系：** B 决定跨样本 memory graph 中“谁与谁应形成哪类 signed hard edge”；A/审计候选 1 决定单样本“哪种因果 coalition 应对写入 memory 的表示产生梯度”。二者理论上互补，但 **Gate 2 的首轮不得合并**，否则涨跌无法归因。先单独验证最高 EV 的 B；若 B fast-fail，再以审计候选 1 作为 A 的具体实现。

### Gate 1 必须锁定的共同反事实

对每个候选都不能只比 `method vs floor`，还必须有能否证伪 MLLM 机制的 control：

- A：teacher interaction shuffle / absolute-score distill；
- B：edge-type shuffle / label-only SupCon / generic comparability；
- C：rule shuffle / uniform router / independent-field auxiliary loss。

只有 method 赢 floor 且赢这些 mechanism controls，才可说 MLLM meaningfully/novelly integrated；否则即使偶然涨分，也不能归因。

---

## 7. 查新检索记录（每个核心 claim ≥3 组；含最近六个月）

**最近六个月窗口：** 2026-01-10 至 2026-07-10。检索覆盖 arXiv、OpenReview（ICLR/ICML/ACL ARR/TMLR）与 ACL Anthology；另回溯 2024–2025 的 RGCL/RA-HMD、HateSieve 与通用 causal/rationale 工作。下面保留实际核心检索式，便于复查。

### Claim A：MLLM coalition delta → hateful-video RGCL retrieval/memory gradient control

1. `site:arxiv.org 2026 multimodal causal intervention distillation modality ablation vision language model`
2. `site:openreview.net 2025 2026 multimodal causal intervention distillation VLM`
3. `site:arxiv.org 2025 2026 hateful video causal multimodal fusion distillation`
4. `site:aclanthology.org 2025 multimodal rationale distillation causal intervention`
5. `site:arxiv.org 2026 "hateful video detection" MLLM reasoning`
6. `site:openreview.net 2026 hateful video multimodal`

命中最近六个月：BridgeVLM（2026-04/06）、CGO（2026-01）、Diagnosing Modality Interference（2026 revision）、RAMF（TMLR 2026-06）、IARE（2026-06）、MARS（2026-01）。**未命中直接相同组合。**

### Claim B：gold-signed semantic relation graph → relation-conditioned RGCL

1. `site:arxiv.org 2026 rationale guided contrastive learning multimodal hate detection graph`
2. `site:aclanthology.org 2025 2026 rationale guided contrastive multimodal classification`
3. `site:openreview.net 2026 signed relational contrastive learning multimodal reasoning`
4. `site:arxiv.org 2025 "hateful video" contrastive rationale relation`
5. `site:arxiv.org 2025 2026 LLM-guided hard negative contrastive multimodal classification`
6. `site:openreview.net 2026 rationale-guided hard pair contrastive multimodal`

命中：VC-STaR（ICLR 2026 Oral）、Multimodal Representation Learning Conditioned on Semantic Relations（ICLR 2026 submission）、HateSieve（NAACL Findings 2025）、RGCL/RA-HMD。**一般 relation-conditioned / generated triplet 已存在；具体 gold-signed stance-target relation for hateful-video RGCL 未命中。**

### Claim C：MLLM verified clauses → rule-conditioned adapter routing

1. `site:arxiv.org 2026 MLLM neuro-symbolic rule induction multimodal classification router`
2. `site:openreview.net 2025 2026 multimodal concept bottleneck rule routing vision language model`
3. `site:aclanthology.org 2025 2026 LLM induced rules multimodal classification neuro symbolic`
4. `site:arxiv.org 2025 2026 hateful video mixture of experts reasoning router`
5. `site:arxiv.org 2026 "hate video" multimodal large language model classification`
6. `site:openreview.net 2026 harmful meme symbolic reasoning mixture experts`

命中：BPDMoE-Hate、SyRHM、LangCBM、NS-Mem、2026 neural-symbolic rule work。**大类高度拥挤；verified clause 驱动小型 RGCL adapter path 未命中，但只属组合级空白。**

### 领域全量补查

1. `site:arxiv.org 2026 "hateful video detection" MLLM reasoning`
2. `site:arxiv.org 2026 "hate video" multimodal large language model classification`
3. `site:aclanthology.org 2026 "hateful video" OR "hate video detection"`
4. `site:openreview.net 2026 hateful video multimodal`

该轮补出 2026-06 IARE、TMLR 2026 RAMF、MARS、LELA、SCANNER、CGO、SyRHM、BPDMoE-Hate 等，因而本报告没有沿用 2025 年底以前的旧 landscape 直接下结论。

---

## 8. 查新边界与诚实声明

1. **查新不是数学证明。** arXiv/OpenReview 的索引可能延迟，ACL ARR 中还有匿名或标题变化稿；投稿前必须按同一 query pack 再扫一次。
2. OpenReview 中 CGO、BPDMoE-Hate、SyRHM、relation-conditioned representation 等是 submission/entry，除非页面明确 accepted，本报告不把它们写成 peer-reviewed SOTA；但查新时仍必须视为 prior-art threat。
3. 已收到第二位架构 agent 的 adversarial novelty attack：其 verdict 为 **B > A（仅当 A 是 retrieval/memory-gradient controller）≫ C**；本报告已据此改写 A 并调整执行顺序。该复核仍不是 novelty-check 技能指定的独立外部模型 verdict，因此进入 Gate 1/外审前还应针对 RAMF/CGO、relation-conditioned MRL/HateSieve、BPDMoE/SyRHM 三组近邻做一次独立模型审计。
4. 本报告只提出科学候选，不把预期增益当结果。真正成功必须由同 split、同 protocol、multi-seed final acc/macro-F1 与机制消融共同证明。

## 9. 本轮使用的本地材料

- [CAMPAIGN_mllm_method_role.md](./CAMPAIGN_mllm_method_role.md)
- [PAPER_MASTER_TABLES.md](./PAPER_MASTER_TABLES.md)
- [MLLM_USAGE_LANDSCAPE.md](./MLLM_USAGE_LANDSCAPE.md)
- [EXP_p2_neighbor_rerank.md](./EXP_p2_neighbor_rerank.md)
- [EXP_p3_evidence_pooling.md](./EXP_p3_evidence_pooling.md)
- [EXP_p4_schema_distill.md](./EXP_p4_schema_distill.md)
- [EXP_p5_counterfactual_negs.md](./EXP_p5_counterfactual_negs.md)
- [EXP_p9_lmm_rgcl_video.md](./EXP_p9_lmm_rgcl_video.md)
- [EXP_p10_loc_amplify.md](./EXP_p10_loc_amplify.md)
- [EXP_p11_weaksup_localization.md](./EXP_p11_weaksup_localization.md)
- [NOVELTY_CHECK_dirA.md](./NOVELTY_CHECK_dirA.md)
- `research-wiki/papers/` 中 20 篇最相关的 2025–2026 论文笔记，以及 `multihateloc_v3.pdf` 的前 3 页/元数据。
