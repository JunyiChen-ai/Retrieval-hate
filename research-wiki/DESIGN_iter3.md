# DESIGN — Iteration 3: Self-maintaining (auditable) hate memory — retrieval-consensus segment denoising (A) + evolving-memory protocol (B) + MLLM 三角色

_起草:2026-07-02。Status: **DRAFT — 待用户拍板**(所有 [USER-DECISION] 处不预设结论)。
Idea slug: `self-maintaining-hate-memory-iter3`。本文件不取代 `DESIGN_iter1.md`(iter1 的
已验证结论——updatable cross-dataset kNN memory 为 headline、multi-granularity 为 honest
negative、MLLM/LoRA 为 lever——全部继承,不重写)。_

_输入材料:`NOVELTY_CHECK_dirA.md`(方向 A 查新,机制 SAFE / 任务 THREATENED)、
`TEMPORAL_SPLIT_FEASIBILITY.md`(MHClip EN/ZH 时间戳 HIGH,HateMM/ImpliHateVid LOW)、
`MLLM_USAGE_LANDSCAPE.md`(三角色占位:角色1 OPEN / 角色2 PARTIAL / 角色3 领域内 OPEN)、
`experiments/exp-seg-mode-ablation.md`(Phase-3 negative 的病因诊断 = 方向 A 的动机)、
`gap_map.md` + `RESEARCH_BRIEF.md`(背景)。_

---

## 1. 故事与贡献声明

### 1.1 总故事(one story)

**A self-maintaining, auditable hate memory.** 检测本身由**检索记忆**完成(学习到的
retrieval-guided contrastive embedding 空间上,对带标签记忆库做 kNN vote——iter1 已验证的
骨架);**MLLM 不判案,为记忆打工**(生成可审计的结构化条目、在标签分歧时裁决片段);记忆
**自己维护自己**:(A)自己洗片段标签——retrieval-consensus segment denoising 把 video-level
标签继承产生的 MIL 噪声正样本剔除(直接治疗 Phase-3 确诊的病:auto sub-clip 继承标签 →
噪声 MIL 正样本 → seg-mode 语言间 sign-flip);(B)自己随仇恨演化更新——evolving-memory
协议在真实上传时间戳上形式化并系统评测"加样本不重训"的适应能力(把 iter1 已验证的
memory-swap 能力从静态 capability demo 升级为动态协议)。

三个部件不是拼盘:A 是记忆的**自清洁**(记忆当标注器,清洗喂给它自己的训练信号),B 是
记忆的**自更新**(记忆当分类器,免重训吸收新时期样本),MLLM 三角色是记忆的**雇员**
(条目生成/分歧裁决/可选的低置信升级)。判决权自始至终在记忆的 kNN vote 上。

### 1.2 Contributions(每条 "Unlike [最近邻竞品], we ..." 句式)

1. **[系统] Unlike MoRE**(记忆=不可读特征三元组、决策烧死在 trained MoE head、零 LLM)
   **and unlike SafeLens**(结构化 JSON 只是**即弃审计日志**,不入库不复用),**we** 把
   hateful-video 检测放进一个 **self-maintaining、human-auditable 的持久记忆**:判决 = 对带
   标签记忆的 kNN vote,记忆条目含 MLLM 生成的固定 schema 档案,**审计对象是持久记忆本身
   ——条目可读、可增删、可追责到每一次 kNN 判决,而非用后即弃的推理日志**。
2. **[方法 A] Unlike MultiHateLoc**(纯 MIL-ranking + attention,无伪标签、无检索、无去噪)
   **and unlike Yang et al. 2025**(用**金标 span** 诊断时序标签噪声、只诊断不治疗),**we**
   提出 **retrieval-consensus segment denoising**——hateful-video 领域内第一个
   **retrieval-as-annotator** 机制:用带标签跨视频记忆库的 kNN 邻居投票 × 自身 video 标签的
   一致性给 sub-clip 定伪标签、剔除 MIL 噪声正样本、EM 自训练;监督只到 video-level
   (**span-free supervision**),时序定位是 free byproduct(在 HateClipSeg 空赛道上外部验证)。
3. **[方法 B] Unlike MoRE**(结构上不能把决策重指向新支持集)**and unlike CRAVE**
   (检索增强**训练**,库在训练期固定),**we** **首个在 hateful-video 上形式化并系统评测
   evolving-hate 协议**:真实上传时间戳 temporal split + 阶段性增补带标签新时期样本(零重训)
   + 记忆维护策略(加什么/怎么权/删什么),指标为适应速度、保持性、更新成本。(注意:我们
   **不主张"首个拥有换库能力"**——RA-HMD 的 cross-dataset 协议事实上具备换库机制、只是未
   主张;我们主张的是**协议的形式化与系统评测**,并向 Class-RAG / Contextual Policy Engine
   的可更新库思想引用划界:它们分别是文生图 prompt 审核与纯文本政策库,非多模态视频。)
4. **[MLLM 三角色] Unlike 领域内全部 reasoning-VLM 方法**(MARS/HVGuard/RAMF/LELA/IARE/
   TANDEM:always-on、无门控、生成文本即用即弃、MLLM 即判决者),**we** 让 MLLM 严格为记忆
   打工且**从不输出最终判决**:角色1 = 固定 schema 结构化档案持久化入库作检索键;角色2 =
   仅在共识分歧时裁决片段(调用率作为指标报告);角色3(可选,[USER-DECISION])= kNN 低置信
   才唤醒的选择性推理。

### 1.3 措辞红线(硬约束,写作期逐条自查)

| 红线 | 原因(占位方) | 我们的合规措辞 |
|---|---|---|
| A 不得说 "annotation-free" | LELA 已占 training-free/无标注定位的话语空间(且我们用了 video-level 标签) | **"span-free supervision"** |
| A 不得说 "without dense (frame-level) supervision" 作为卖点 | TANDEM 明确声称 "不需要 dense frame-level supervision" | 同上,"span-free";机制卖点是 **retrieval-consensus / retrieval-as-annotator** |
| 不得说 "first weakly-supervised (temporal) hate localization" | MultiHateLoc(WWW 2026)已抢注任务第一 | 主 claim 定位在**去噪机制**;定位 = "free byproduct + 在 HateClipSeg 上首个弱监督评测**结果**"(先到先得的是结果,不是任务) |
| B 不得说 "first framework capable of memory swap / test-time update" | RA-HMD 换库机制事实存在(未主张);CRAVE 是检索增强训练需划界 | **"first to FORMALIZE and systematically EVALUATE an evolving-hate protocol"** |
| "auditable" 必须钉死审计对象 | SafeLens 宣称 "auditable via reproducible JSON logs"(同 Roy Lee 组生态,撞车风险最高) | **"auditable = the PERSISTENT memory itself"**(条目可读可增删、判决可溯源到具体邻居条目),正文显式对比 "即弃日志 vs 持久记忆" |
| 角色1 不得表述成 "generate caption/description then classify" | Pro-Cap(meme)已占;"rationale as feature/distillation" 被 HVGuard/RAMF/Mr.Harm 占;"debate/judge" 被 ExplainHM/MARS 占 | "structured, human-auditable memory entries under a fixed schema, **persisted as retrieval keys**"(schema 化生成 + 持久化入库 + 作检索键,三合取) |

---

## 2. 方法 A:retrieval-consensus segment denoising

### 2.1 动机(闭环叙事)

Phase-3(`exp-seg-mode-ablation.md`,verdict=no,confidence=high)确诊:auto sub-clip
**继承 video 标签**产生噪声 MIL 正样本,导致 segment 项语言间 sign-flip(full:EN +0.015 /
ZH −0.066;milmax 反转;driftneg 近无作用),无一 seg_mode 双语同向 ≥ baseline。Yang et al.
2025(arXiv:2508.04900)在领域内系统诊断了同一病因(用金标 span 裁剪后 +19–30% headroom),
但**不提供任何自动去噪方法**。方向 A = "Yang 诊断的自动化治疗 + 我们自己 G4 负结果的对症
闭环":用记忆库自己当标注器,替代金标 span。

### 2.2 机制

记 video $v$、标签 $Y_v\in\{0,1\}$、auto sub-clips $\{s_1..s_K\}$(K=4 均匀窗,零标注),
记忆库 $\mathcal{M}$ = train(+val) 带标签样本在**当前学习到的 fused 空间**中的 embedding。

**共识伪标签(retrieval-as-annotator)。** 对每个 sub-clip $s$:在 $\mathcal{M}$ 上做 kNN
(邻居粒度 = whole-video、sub-clip、或两者,消融决定),邻居的 **video-level 标签**做相似度
加权投票得 $\hat{y}_s$ 与 margin $m_s$。伪标签 = **agreement(自身 video 标签 × 邻居投票)**:

| $Y_v$ | 邻居投票 $\hat{y}_s$ | 判定 | 用途 |
|---|---|---|---|
| hate | hate(一致) | **高置信 hateful 片段** | sub-clip 对比项的正样本 |
| hate | benign(分歧) | **噪声正样本嫌疑 = 良民背景片段** | **忽略**(不当正样本);兼作同视频 **drifting hard-negative** 候选 |
| benign | benign(一致) | 高置信 benign 片段 | 负样本 |
| benign | hate(分歧) | 可疑 confusable(记忆噪声或近仇恨良民) | 默认忽略;"当 hard negative" 作消融([次级决策]) |

仅 margin $m_s \ge \tau$ 的高置信片段进入 sub-clip 对比损失(pseudo-gold positive / hard
negative 都从共识过滤后的集合里挖);其余片段只参与 whole-video 项。

**EM 迭代(2–3 轮)。** Round 0 用 frozen encoder 的初始 embedding 起共识;E-step:用当前头
的投影 embedding 重建 FAISS(与现有 per-epoch refresh 同一机制)并重算共识伪标签;M-step:
以新伪标签重训头。轮数 2–3(EM-MIL/MIST 的通行区间),防伪标签漂移:每轮报告伪标签翻转率,
翻转率不收敛即停。

**保留 drifting hard-negative。** Phase-3 的诊断把失败归因于**噪声 MIL 正样本**而非负例侧
(driftneg 分支在 EN 上近乎无损、机制本身未被判死);共识框架给 drifting negative 一个更干净
的来源——"$Y_v$=hate 且共识=benign"的片段正是语义上漂移的良民片段,替代原来 MIL 打分挖负例
的启发式。(诚实备注:driftneg 在 ZH 上曾低于 baseline,"健全"指机制未被证伪、非已证有益;
共识来源能否救活它正是本消融要回答的。)

### 2.3 工程挂点(对着 `/data/jehc223/RGCL/src`,增量最小)

- **Sub-clip 缓存已在**:`data/CLIP_Embedding/{MHC,MHC_zh}/{train,dev_seen,test_seen}_subclipK4_openai_clip-vit-large-patch14-336_HF.pt`
  (已核实在盘)。**HateMM / ImpliHateVid 的 subclipK4 缓存不存在,需新抽**(GPU 批任务,
  用现成 `src/utils/generate_subclip_embedding_HF.py`);Qwen 版 sub-clip 缓存亦无(可选)。
- **seg_mode 钩子已在**:`src/model/loss.py`(seg_mode 分支 full/driftneg/milmax,
  ~L578–803)+ `src/run_rac.py --seg_mode`(L236)+ `src/utils/retrieval.py`
  (`_encode_subclip_fused`,L487,sub-clip 融合编码/第二 FAISS 索引)。
- **新增**:(a) `seg_mode=consensus` 分支(邻居投票 + agreement 表 + margin 阈值 τ);
  (b) EM 外层循环(每轮 = 一次头训练 + 一次伪标签重算;脚本级,不动核心);(c) 自打分
  baseline(MIST 式:用头自身 hate-score 给 sub-clip 打伪标签,同一过滤/训练管线,只换
  标签来源——保证消融只差一个变量)。

### 2.4 定生死消融(make-or-break)

同一 split、同一头、同一 val-selection(warmup≥5 + max Val_Retrieval acc,tie-break roc):

| 伪标签来源 | 对应文献线 | 预期角色 |
|---|---|---|
| **consensus**(邻居投票 × 自标签一致性) | 我们(retrieval-as-annotator) | 主张的机制 |
| 自打分(模型自身 hate-score,MIST/C2FPL 式) | WSVAD 伪标签线的移植 | 机制对照:检索共识 vs self-labeled |
| 继承标签(seg_mode=full,Phase-3 复现) | Phase-3 negative | 已知失败下界 |

**判据(Phase-3 失败判据的镜像):consensus 必须在 MHC-EN 与 MHC_zh 双语同向 ≥ whole-video
baseline**。赢了 → novelty 从"组合新"升级为"机制优"(NOVELTY_CHECK §4.3.3 的一票定生死);
输给自打分 → 方向 A 只剩组合新颖性,故事降级为 honest analysis。补充机制消融(reviewer 必问,
NOVELTY_CHECK §4.3.2):Zhong 2019 式图传播清洗 vs 检索共识清洗;邻居粒度(video/sub-clip/
两者);τ 敏感性;EM 轮数(0/1/2/3)。

### 2.5 评测

- **主表:video-level 二分类**(MoRE 协议,macro-F1/P/R + acc,4 数据集)——A 的第一价值是
  修复 segment 项、把 Yang 的 headroom 兑现到分类上,并消除语言间 sign-flip。
- **HateClipSeg 弱监督定位(空赛道)**:435 视频 / 11,714 segments 金标,该数据集定位任务
  目前只有全监督 baseline、引用者中无人做弱监督(已穷尽核查)。我们只用 video-level 标签
  训练、金标只用于评测。**MultiHateLoc 当 baseline**——注意其发表数字在 HateMM
  (frame mAP 0.645 / AUC 0.799)与 MHC(mAP 0.445),**不在 HateClipSeg**,需移植复跑
  (代码可得性未核实,风险见 §6)。
- **HateMM frame-level 定位(可选,[次级决策])**:官方 Zenodo 标注 CSV 自带 `hate_snippet`
  列(TEMPORAL_SPLIT §3 已核实拉取),可直接按 MultiHateLoc 的 HateMM frame-level 协议评测
  ——**免移植即可与其发表数字同协议对比**。(与 iter1 "本地无金标 span" 的记录冲突,见报告;
  需先核实该 CSV 的 span 覆盖与格式。)

---

## 3. 方法 B:evolving-memory 协议

### 3.1 数据与 temporal split(TEMPORAL_SPLIT_FEASIBILITY 已验证 HIGH)

- **MHClip EN**(YouTube,891 条,`yt-dlp --print upload_date`,抽样存活 25/30 ≈ 83%)+
  **MHClip ZH**(Bilibili,897 条,web API `pubdate` 秒级,抽样存活 23/30 ≈ 77%);全量采集
  <1.5h 登录节点轻活,可断点续跑,结果落 `data/gt/<DS>_upload_dates.json` 一次采集永久复用。
  日期主体 2022–2024(EN 跨 2010–2024,ZH 跨 2018–2024),足够切时期。
- **Split 规则**:可定年子集按 upload_date 排序,切 $T_0$(旧期:train + 初始记忆)/
  $T_1..T_n$(新期:阶段测试流);**不可定年样本固定进 train/记忆库,永不进 test**(避免
  污染"未来"集)。EN、ZH 各自独立切,做跨语言时间泛化对照。
- **必须报告 survivor bias**:死链显著偏向 Hateful/Offensive(30 样本探针:ZH Hateful 死链
  ~60%,EN Hateful 50%、Offensive 43%——小样本估计,全量采集后重估),可定年子集的标签分布
  会向 Normal 偏移;主文报告可定年子集标签分布 + 该偏差声明;可选 Wayback/biliplus 补链。

### 3.2 协议(形式化)

每阶段 $t=1..n$:(i)**静态测**:用截至 $t-1$ 的记忆对 $T_t$ 测试(演化差距);
(ii)**增补**:向记忆加入 $k$ 个 $T_t$ 期带标签样本(embed + FAISS append,**零重训**);
(iii)**复测** $T_t$ 剩余样本与全部旧期 test。指标:

- **适应速度**:每 $k$ 个新样本带来的 ΔmacroF1;达到 in-period 上界(该期样本充分入库)所需
  的 $k$;
- **保持性**:更新后旧期 test 性能(纯 add 理论无遗忘;**删除/加权策略下才有风险**,这正是
  维护策略消融的看点);
- **零重训成本**:每次更新的 wall-clock / GPU 成本(embed+append,秒级)vs 全量重训 / 头微调
  基线(GPU 分钟级)——报告 性能-成本 曲线。

Baselines:static memory(不更新,下界)、全量重训(性能与成本上界)、头微调(中间)、
以及"傻塞"(见下)。

### 3.3 记忆维护策略(加什么 / 怎么权 / 删什么)

| 维护维度 | 策略 | 对照(傻塞) |
|---|---|---|
| **加什么** | 不确定性主动选择:kNN margin 最小 / 邻居标签熵最大的新期样本优先入库(标注预算固定为 $k$) | 随机 $k$ 个 |
| **怎么权** | 时间加权投票:邻居票权随年龄指数衰减(近期条目权重高) | uniform 票权 |
| **删什么** | 预算 $B$ 上限 + 去重(库内 cosine>τ 近重复合并/驱逐) | 不删 / FIFO |

消融矩阵 = {傻塞, +主动选择, +时间加权, +预算去重, 全开};看点:策略能否用**更少标注**达到
同等适应速度、且保持性不塌。

### 3.4 与静态 cross-dataset 矩阵的关系

HateMM / ImpliHateVid **官方发布已匿名化、无时间元数据**(恢复路径 ~0%,除联系作者),
不做 temporal split;二者保持 iter1/Phase-3b 的**静态 cross-dataset swap 矩阵**角色,作为
本节的互补小节(静态换库 = 演化协议的极限情形:一次性换整库)。划界引用:Class-RAG
(Meta,文生图 prompt 审核的 semantic hotfixing)、Contextual Policy Engine(纯文本政策库)
——可更新库思想首次带入多模态 hateful video;RA-HMD(机制潜在具备、未形式化未评测)。

---

## 4. MLLM 三角色(全部为记忆打工,MLLM 从不输出最终判决)

### 4.1 角色 1 — 结构化档案入库作检索键(占位判定:**OPEN**)

**机制**:离线、每视频一次,frozen Qwen2.5-VL-7B 在固定 schema 下生成结构化档案条目:
`{攻击目标 target, 攻击机制 mechanism, 载体模态 modality carrier, 显隐性 implicitness}`
(+ 一句话 evidence);条目**持久化入记忆库**,与 fused embedding 并列作为**检索键之一**
(档案文本再嵌入)与**审计证据**(kNN 判决可展示"命中了哪些条目、其档案说了什么")。
**ZH 用英文枢轴**:ZH 视频的档案字段统一用英文生成——规避 CLIP/文本塔中文短板、统一跨语言
检索键空间(EN↔ZH 记忆互检索时键同语言)。

**Claim 措辞锚点**(MLLM_USAGE_LANDSCAPE 总表 + 扫描点1):"structured, human-auditable
memory entries generated by an MLLM under a **fixed schema**, **persisted** as **retrieval
keys**" —— schema 化生成、持久化入库、作检索键,**三件事的合取**在 video+meme 两域均无先例。
Related work 必须逐一点名划界:SafeLens(结构化 JSON = 即弃审计日志)、TANDEM(XML 输出
schema,无库无检索)、IARE(harmful elements = 推理中间物)、Pro-Cap(meme,probing caption
喂分类器)、MoRE(有库,条目=不可读特征、零 LLM)。红线:不得写成 "generate
caption/description then classify"。

**消融**:检索键 = fused embedding only vs +档案嵌入 vs 档案 only;审计性以案例研究 +
条目人检一致率呈现。

### 4.2 角色 2 — 片段分歧裁决(占位判定:**PARTIAL,窄口径**)

**机制**:仅在方法 A 的共识循环里,当 sub-clip 出现**标签分歧**($Y_v$ 与邻居投票不一致,
即 agreement 表的两个分歧格)时,才调用 MLLM 对**该片段**裁决(hateful/benign/uncertain),
裁决结果决定该片段进正样本、当 drifting negative、还是继续忽略。**调用率(% sub-clips
adjudicated)作为指标随主表报告**——这是"选择性"主张的可量化证据。

**Claim 措辞锚点**(扫描点2):"**selective, disagreement-triggered adjudication at SEGMENT
level**" —— 三要素合取(片段级 × 分歧触发 × 裁决者)无人做。必须划界:IPS(全量视频级辅助
标签替代人工,工业;**投稿前复查 v3+ 演化**)、MetaHarm(GPT-4 为平级标注者非裁决者)、
HateClipSeg(LLM 只做预筛,segment 冲突仲裁是纯人工)、TANDEM/IARE(数据构建期 silver
标注,人工兜底)。窄口径 = 不宣称"MLLM 弱监督标注"(已拥挤),只宣称分歧裁决这个空格。

**消融**:consensus-only vs consensus+裁决;裁决预算敏感性(只裁 margin 最小的前 p%)。

### 4.3 角色 3 — kNN 置信门控的选择性推理(占位判定:**领域内 OPEN;缓办 [USER-DECISION]**)

**机制(若上)**:推理时以 kNN 邻居标签熵 / 相似度 margin 做门控,仅低置信样本唤醒 MLLM,
对**检索到的结构化条目(角色1 产物,非原始样本)**做 in-context 推理后回填判决;调用率与
性能-成本曲线一并报告。

**Claim 措辞锚点**(扫描点3):"**confidence-gated deferral** over the kNN memory, reasoning
**over retrieved structured entries**" —— 领域内全部 reasoning-VLM 均 always-on(MARS 明文
置信度"只作解释不做阈值");RA-HMD 判决头切换为协议固定、无任何动态门控(正交,引用时不可
说它有门控)。必须划界:Filter-And-Refine(TikTok,**相似度路由**级联而非置信 deferral、非
hate 学术基准)、Google 2406.12800(LLM→human escalation)、meme LMM Agents
(arXiv:2411.05383,检索**原始标注样本**进 prompt 且 always-on)。

**缓办理由**:A+B+角色1+角色2 已构成完整闭环故事;角色3 增加实现面与查新面(工业先例划界),
且与角色1 有依赖(需先有结构化条目可推理)。**上不上由用户拍板(§7-1)**。

---

## 5. 实验矩阵

标记:[login] 登录节点轻活;[SLURM-gpu] sbatch GPU 批任务(不设 `--time`);[cpu] faiss-cpu
头训练(SLURM CPU,分钟级/run)。一切执行走 subagent/workflow。

| # | 实验 | 数据集 | 依赖 | 成本(粗估) | 先后 |
|---|---|---|---|---|---|
| E0a | **时间戳采集脚本**(yt-dlp + Bilibili API,断点续跑 → `data/gt/*.json`) | MHC, MHC_zh | 无 | [login] <1.5h,0 GPU | **开工即做(并行①)** |
| E0b | **MHClip 结构化档案生成**(角色1,固定 schema,ZH 英文枢轴) | MHC, MHC_zh(~1.8k 视频) | 无(Qwen2.5-VL-7B 已在盘) | [SLURM-gpu] 单卡 半天级 | **开工即做(并行②)** |
| E1 | `seg_mode=consensus` 实现 + **定生死消融**(consensus vs 自打分 vs 继承) | MHC, MHC_zh(subclipK4 缓存已有) | 共识循环实现 | [cpu] 每 run 分钟级 × EM 2–3 轮 × 3 来源 × 2 语言 × 多种子 | E0 后立刻;**gate:双语同向 ≥ baseline** |
| E2 | HateMM / ImpliHateVid **subclipK4 缓存抽取** → 4 数据集主表 | HateMM, ImpliHateVid | E1 过 gate | [SLURM-gpu] 每集 1–3 GPU·h(CLIP;Qwen 版另计) | E1 之后 |
| E3 | **HateClipSeg 弱监督定位**:下载+嵌入+span-free 训练+金标评测;MultiHateLoc baseline 移植 | HateClipSeg(435 视频) | E1 机制;MultiHateLoc 代码可得性(未核实,§6) | [SLURM-gpu] 下载+嵌入数 GPU·h;移植人力为主 | E1 之后,与 E2 并行 |
| E4 | (可选)HateMM `hate_snippet` frame-level 定位(MultiHateLoc 同协议直接比) | HateMM | E2 的 HateMM sub-clip 缓存;核实 CSV span 覆盖 | [cpu] | E2 之后;[次级决策] |
| E5 | **Temporal split 构建 + evolving-memory 主协议**(§3.2,含三条指标) | MHC, MHC_zh | E0a 完成 | [cpu](embed 已缓存,更新=index append) | E0a 之后即可,**与 A 线独立并行** |
| E6 | 记忆维护策略消融(傻塞 vs 主动选择/时间加权/预算去重) | MHC, MHC_zh | E5 协议就绪 | [cpu] | E5 之后 |
| E7 | 角色1 检索键消融(embedding vs +档案 vs 档案 only)+ 审计案例 | 先 MHC/MHC_zh,后全集 | E0b | [cpu] + 档案嵌入 [SLURM-gpu] 小时级 | E0b 之后 |
| E8 | 角色2 分歧裁决消融(consensus±裁决;调用率、裁决预算) | MHC, MHC_zh | E1 过 gate + MLLM 批推理 | [SLURM-gpu] 只对分歧片段调用(量 = 调用率 × sub-clip 数,预期远小于全量) | E1 之后 |
| E9 | (缓办)角色3 门控推理 | 全集 | E7(需结构化条目)+ §7-1 拍板 | [SLURM-gpu] 低置信子集 | [USER-DECISION] |
| E10 | 多种子/CI + 最终 head-to-head(MoRE / MM-HSD / MultiHateLoc / RAMF / HVGuard) | 全集 | 上游全部 | [cpu] 为主 | 收尾 |

关键依赖链:E0a→E5→E6(B 线);E1→{E2,E3,E8}(A 线);E0b→E7(→E9)(MLLM 线)。三条线
**除 E1 的 gate 外互不阻塞**;开工日即可并行 E0a + E0b。

---

## 6. 风险与投稿前复查清单

1. **Exeter 组(Zeyu Fu)速度** —— Yang2025→MultiHateLoc→LELA→RAMF 连发,自动去噪是他们
   显而易见的下一步;方向 A 的最大非技术风险是被抢发。缓解:E1 定生死消融最先跑,过 gate
   即尽快挂 arXiv;持续盯 Exeter + SUTD(Roy Lee)两组新帖。
2. **HCG-MPB(ICMR 2026)全文未读** —— 查新中唯一未闭环的领域内 retrieval 系工作("Pattern
   Bank" 有记忆库味道);投稿前必须补查全文,确认无伪标签/无 segment 成分。
3. **IPS v3 演化中**(TikTok,arXiv:2412.15251)—— 角色2 最近的占位者,工业持续迭代;投稿前
   复查其最新版是否伸入"片段级/分歧触发"。
4. **SafeLens "auditable" 措辞撞车**(Roy Lee 组生态)—— 正文必须显式对比"持久记忆 vs 即弃
   JSON 日志",摘要/引言里 auditable 一词必须紧跟审计对象限定。
5. **MHC-EN 161 小测试集统计功效** —— 任何小幅优势在 161 样本上不显著;全部主张配多种子 +
   bootstrap CI(+McNemar);不在该 split 上单独立 claim。
6. **MultiHateLoc baseline 可得性** —— 其数字不在 HateClipSeg 上、代码可得性未核实;若无法
   移植,退路 = HateMM frame-level 同协议对比(E4)+ WSVAD 移植(MIST/RTFM)作定位 baseline。
7. **Temporal split survivor bias** —— 死链偏 Hateful(ZH ~60%,30 样本探针);全量采集后
   重估并在主文报告;不可定年样本固定进 train 的规则写死在脚本里。
8. **预印本索引滞后** —— 2026-05 之后的 arXiv 可能未被本轮查新覆盖(NOVELTY_CHECK 置信
   ~75–80%);投稿前一个月重跑方向 A 查新 + 重扫 HateClipSeg 新增引用。

---

## 7. [USER-DECISION] 汇总(留给用户拍板,本稿不替用户决定)

1. **打包范围:角色 3 上不上?**
   - 选项 A:缓办(默认草案)——A + B + 角色1 + 角色2 成稿;角色3 留作 future work 一段。
   - 选项 B:上——故事更完整("记忆的雇员"三角色齐),代价 = 实现面 + Filter-And-Refine 等
     工业先例的划界负担 + E9 的 GPU/时间。
2. **0.85 目标降级确认。** iter3 的故事重心从"MHClip acc≥0.85"(LoRA 后 EN gap 0.098 / ZH
   gap 0.018,双语均未过)转向"去噪机制 + 演化协议 + 可审计记忆";0.85 保留为 HateMM /
   ImpliHateVid 已达成的支撑数字,MHClip 目标改述为"beat MoRE + 双语**同向**增益 +(A 若过
   gate)兑现 Yang headroom 的一部分"。**是否接受该降级,请确认。**
3. **目标 venue / 时间线。** 候选:ACM MM 2027 / WWW 2027 / ACL-EMNLP 2027(领域主战场为
   MM/WWW;B 的协议叙事亦适合 WWW)。鉴于风险 §6-1(Exeter 速度),越早的 deadline 越优;
   是否以"E1 gate 通过即挂 arXiv 占位"为既定策略,亦请一并拍板。
4. (次级)HateMM `hate_snippet` 定位评测(E4)是否纳入主文(需先核实本地 CSV 的 span
   覆盖与格式;若可用,是与 MultiHateLoc 免移植同协议对比的最短路径)。
5. (次级)共识邻居粒度(whole-video / sub-clip / 两者)与"benign×邻居hate"格的处理
   (忽略 vs hard negative)——默认按 §2.4 作为消融跑,不需前置拍板;若想砍消融量再定。
