# OPTION KITS — terminus 后「裁决即开工」预备包

> **性质声明(读前必看)。** 本文件是 **决策辅助 (decision-support)**,面向用户在 TERMINUS 三选项
> (a 接受现状 / b 闭源 API 攻定位 / c 换方法族)之间拍板。**本文件不含任何已批准的行动**:所有内容
> 是「裁决后第一周的具体动作清单」草案,Kit-B 的预注册整体标 **DRAFT-INACTIVE(未获批不生效)**。
> 全程不提交 SLURM、不碰 test、不外发任何数据 —— 本文件本身只是文档。
>
> **依据基线(全部已 commit)。** TERMINUS 终局(`TERMINUS_mllm_campaign_DRAFT.md`,FINAL):11 路线主表
> accuracy 角色被证伪;MLLM 挣得三角色 = **encoder + 定位打分器(P6 modest 0.5435 → P10-b 72B A-fuse
> modest-plus 0.5755)+ guard-rail/审计**。P10-b 校准冠军 = 72B A-fuse HateMM wv-AUC **0.5913**(commit
> 03880f2);单次 HateClipSeg test = **0.5755(MODEST,已花掉)**;探索附录重聚合天花板 **0.5932 < 0.616**
> 目标线(commit 93e82fa)。主表终局:同场 MoRE 三库全胜(+5.6~+8.7 acc),EN ≈0.78–0.80 近天花板,
> ZH 双口径(val-选点 0.827 / final-epoch 0.8537)。

---

## Kit-A —— 接受现状(改文档即可定稿,零额外计算)

裁决为 (a) 后,第一周全部是**文档定稿动作**,无任何 GPU/SLURM/test。三块:改定位「三件套」措辞、
落地方法章的 MLLM 三角色框架、给实验表分配正文/附录。末尾列与两个旧决策的耦合。

### A.1 MORNING_REPORT 定位「三件套」怎么改(建议文案,**不直接改 §4**)

现状问题:MORNING_REPORT 有三处定位表述停在 **campaign 之前**的「视觉-only 记忆键」口径(wv-AUC 0.526,
只作能力演示、不做显著性主张),与终局的 P6→P10-b **统计稳固 modest-plus 定位器**不一致。三处需同向更新,
**核心 reframe = 区分两个东西**:记忆键定位(仍弱,0.526,能力演示)vs **MLLM 逐窗打分器定位(挣得的可移除
角色,P6 0.5435 → P10-b 0.5755,对 memory/random 均配对显著)**。三处均**保留诚实幅度警告(仍 modest,<0.60;
主导能力仍是 video-level density)**。

- **① §3「定位评测」第 2 条(HateClipSeg)。** 现文钉在记忆键「wv-AUC 0.526,仅 1/4 cell 过 Bonferroni …
  只作能力演示」。**建议追加一句(不删旧句,旧句作 memory baseline 保留):**
  > 「上述为视觉-only 记忆键口径。campaign 另立 **MLLM 逐窗打分器定位器**(读帧+ASR 打 0–3):P6-7B
  > within-video AUC **0.5435**(CI[.533,.554],对 memory 配对 **+0.030,p=0.007**;对空 p=5.4e-8);
  > P10-b 72B A-fuse 放大到 **0.5755**(CI[.5581,.5933],对 memory **+0.0615** CI 排除 0、对 P6-7B **+0.0319**
  > CI 排除 0)。判定 **modest / modest-plus**(0.56≤·<0.60,未达 substantial);主导仍是 video-level
  > density(broadcast AP 0.62),within-window 是更小但统计稳固、随 scorer 规模单调增长的增量。」

- **② §4④ 那句「定位 = 能力演示,不做切片级显著性主张(n 太小)」。** 该句只对**记忆键**成立;对 MLLM
  打分器已被 P6/P10-b 证伪(现有配对显著性)。**建议改为:**
  > 「**记忆键**定位不做切片级显著性主张(视觉-only,wv-AUC 0.526,n 小);但 **MLLM 逐窗打分器**是一条
  > **独立的、可移除的、统计稳固的 modest-plus 定位角色**(P6 0.5435 → P10-b 0.5755,对 memory 配对显著),
  > 见第 5 章 / §3。档案的付费点仍在审计/编辑,不在检测 accuracy。」

- **③ §5 撤回清单第 13 行「'定位能力强'类主张(HateClipSeg/HateMM)」。** 该行**保留但缩范围**:被撤的是
  **池化级「定位能力强」**(broadcast 对照几乎追平);新证据支持一条**更窄的**幸存主张。**建议把撤回依据
  末句改为:**
  > 「… 池化指标主体=毒性密度视频间排序(broadcast 追平),within-video 记忆键仅 1/4 cell 显著(0.526)
  > → **池化级「定位强」仍撤回**;但 **MLLM 逐窗打分器**挣得一条**窄而稳**的 modest-plus 定位角色
  > (P6 0.5435 → P10-b 0.5755,对 memory 配对显著,CI 排除 0),如实标 modest、不外推为 substantial。」

> 主会话若采纳,把上述三段落地到 §3/§4④/§5,并把 MORNING_REPORT §9(已引 TERMINUS DRAFT)与 §3「第 5 章
> 素材」的定位段对齐到 P10-b 数字。**本 Kit 不代改**,只给可粘贴文案。

### A.2 方法章的 MLLM 三角色叙述框架

论文方法章 MLLM 一节按**「三个挣得的角色 + 一条明确的非角色」**组织(全部可消融、全部诚实标幅度):

1. **encoder(已入主表)。** frozen Qwen 特征 HateMM 比 CLIP +4.2 macro-F1 且跨 0.85 —— 但这是「冻结
   encoder」,非本 campaign 追求的新方法角色,明说。
2. **定位打分器(唯一挣得的可移除方法角色)。** P6→P10-b:逐窗证据分把 HateClipSeg 时序定位从「存在性
   证明」升为**统计稳固 modest-plus 定位器**;写清「scale 在 localization 上起作用(A-fuse×规模 7B +0.0305
   → 32B +0.0437 → 72B +0.0526 单调)」是这套方法里**唯一** scale 能移针的赛道(与重排线「scale 改
   calibration 不改 selectivity」成对照)。
3. **guard-rail / 审计(可控性,非 raw acc)。** auto-repair 语义票否决 embedding-only 过删(C>D);可编辑
   档案(人审 2-entry 删除改善 EN —— **⚠ F88 更正:SINGLE-SEED**,seed 0 +0.0124 / seed 1-3 零翻转 /
   4-seed 均值 +0.0031,统一措辞 *capability demonstration, single-seed; not an accuracy claim*,
   `ERRPAT_MHC-EN_2026-07-26.md` §6.5 · `ad56a62`);标签盲审计重找到人审噪声 id。付费点 = 完整性/可控性。
4. **明确的非角色(强负结果 = ruled-out map)。** 主表 accuracy 角色被 11 路线机制级证伪;附**两条方法学
   定论**:`comparability ⊥ vote-correctness`(P2/P2b 7B→72B 规模梯,全 8 配置)与`probe 必要非充分`
   (P3-HateMM 最干净 probe / P8-EN 最强 probe 均训练不过)。此节是方法章的**负结果贡献**。

### A.3 哪些表进正文 / 哪些进附录

| 材料 | 位置 | 依据 |
|---|---|---|
| 同场 MoRE 三库全胜表 | **正文主表** | MORNING_REPORT §2(论文主对比) |
| 四支柱 claim(①核心 ②时间协议 ③共识去噪 ④可审计档案) | **正文** | MORNING_REPORT §4 |
| P6→P10-b 定位器结果(0.5435→0.5755 + CI) | **正文(第 5 章能力节)** | TERMINUS §6.2 |
| 11 路线全负记分板(1 行/路线) | **附录(方法学章 ruled-out map)** | TERMINUS §2 |
| 两条方法学定论(规模梯 selectivity 平坦 / probe 必要非充分) | **正文方法章一小节 + 附录数据** | TERMINUS §3 |
| P10-b 校准 leaderboard(11 比较)+ 探索天花板附录 | **附录** | TERMINUS §6.1 / EXPLORATORY |
| selection-robustness / sha1 / 双口径并排 | **附录(方法学纪律)** | MORNING_REPORT §6 |

### A.4 与两个旧决策的耦合(裁决 (a) 不豁免这两项,仍需用户拍板)

- **旧决策① Headline 协议:val-选点(ZH 0.827)vs final-epoch(0.8537)并排。** 与 (a) **弱耦合但需先定**:
  接受 (a) = 主表定稿,而 ZH 主表数字取哪口径直接决定 §1 记分板呈现。现建议案(MORNING_REPORT §7.1)=
  **两口径并排**,主表用预注册 val-选点,附录放五规则鲁棒性全表 + 已成文说明段;若改 final-epoch 必须以
  「未来预注册」名义全线统一并自曝时序(否则 rule-shopping,rebuttal 必死)。**耦合点:定稿前必须锁这口径,
  否则定位/四支柱都改完了主表口径还悬着。**
- **旧决策② EN 近天花板定位确认。** 与 (a) **强耦合**:(a) 的叙事骨架就是「EN 未达 0.85 但近天花板 + 归因
  分析」。需用户确认接受:EN ≈0.78–0.80 双口径、同场 MoRE 仅 0.69–0.72、CRAVE 发表 79.81 F1 为场上最高
  (全量 split);0.85 作为该 split 未被本方法族达到的公开目标如实报告,EN 章主体 = §4③ 归因链 + oracle
  复活条件(role-3 门控留 0.857–0.888 空间)。**耦合点:此确认是 (a) 成立的前提,若用户不接受「近天花板」
  叙事,等于否掉 (a) 转向 (b)/(c)。**

---

## Kit-B —— 闭源 API 攻定位:P10-c 预注册草案(**DRAFT-INACTIVE,未获批不生效**)

> **本节整体是草案,不构成已批准行动。** 生效前置条件(缺一不可):① 用户批准**数据外发**(仇恨内容送
> 第三方商业 API);② 用户接受「闭源不可复现、不能写进可开源 pipeline」的代价;③ 明确愿意为「把定位从
> modest-plus(0.5755)推到 substantial(0.60+)」这一**单一子目标**花第三次 test 触碰。P10-c 是 P10-b
> scale ladder 的**自然续段**:开源 7B→32B→72B 已单调爬到 0.5913(校准)/0.5755(test),闭源只剩
> 「0.5755 → 0.60+」这一段增量。

### B.1 候选清单(冻结在预注册,每模型两行:anchor-agg + A-fuse)

沿用 P10-b 的**同模型自融合**配方(NO cross-model ensemble,守红线):每候选在 **K=30(fine)** 与
**K=4(coarse)** 各打一遍,A-fuse = `0.5·K30 + 0.5·K4(map)` 用**同一模型**分数。anchor-agg = raw K=30。

| # | 候选(当前旗舰,视觉能力) | K30 anchor-agg 行 | K30+K4 A-fuse 行 |
|---|---|---|---|
| C1 | GPT-5(OpenAI 旗舰多模态) | raw K30 | 同模型 0.5·K30+0.5·K4 |
| C2 | Gemini(Google 当前旗舰,如 2.5 Pro / 3.x) | raw K30 | 同模型 0.5·K30+0.5·K4 |
| C3 | Claude(Anthropic 当前旗舰 Opus 级) | raw K30 | 同模型 0.5·K30+0.5·K4 |

> 每候选 2 行 × 3 候选 = **6 个校准配置**(第三轮)。旗舰型号在预注册当日**冻结具体版本串**(避免中途换版
> 造成事后择模)。prompt 复用 P3/P6 冻结模板(帧+窗级 ASR → 整数 0–3,max_new_tokens=8),**不改 prompt**
> (否则混入 prompt 工程,破坏与 7B/32B/72B 的可比)。

### B.2 晋级线设计(vs 新冠军 0.5913;三轮多重比较记账;test-touch 功效分析)

- **校准晋级线(不为第三轮松动):** 某候选配置晋级 test **iff** 在 HateMM 校准集(train-hateful,298 scored /
  n=266 both-class,`p10_eval_hatemm.py`,paired bootstrap 10k)上 **paired Δ vs 冻结 7B anchor(0.5387)
  ≥ +0.04 且 95% CI 排除 0**(与 P10/P10-b **同一未改 bar**),**且** 绝对 wv-AUC 超过现冠军 R2-4 **0.5913**
  (否则纵然过 anchor bar,也不比已花过 test 的 72B A-fuse 更值得再花一次 test)。
- **三轮多重比较记账:** round-1 = 5 配置(A-gate/K60/fewshot/A-lex/A-fuse);round-2 = 5 配置(R2-1..R2-5);
  **round-3 = 6 配置(C1–C3 × 2)**。累计 **16 个校准比较**,预注册明写:test 触碰的是**最高 paired Δ 且
  超冠军 0.5913** 的单一候选;晋级判据对多重比较的 caveat 随判定一并声明(与 P10-b 的「第二轮/11 比较 caveat」
  同纪律)。
- **需要多大 Δ 才值得花第三次 test —— 功效/映射分析:**
  - 现有**两点校准→test 映射**:(anchor 0.5387 → test 0.5435)与(冠军 0.5913 → test 0.5755)。两点连线外推,
    **test 达 0.60 需校准 wv-AUC ≈ 0.616**(TERMINUS 探索附录同一水位)。
  - 即需第三轮候选校准 **≥ 0.616**,对现冠军 0.5913 的 **paired Δ ≳ +0.025 且 CI 下界 > 0.5913**。低于此
    水位,期望 test 仍 <0.60(仍 modest),不值得花唯一剩余的 test 触碰。
  - **重聚合封顶反证:** 探索附录已证现有 7B/32B/72B 分数的任何合法 re-aggregation 封顶 **0.5932 < 0.616**;
    故 0.616 只能靠**更强的 scorer(闭源旗舰)本身**,不能靠聚合旋钮 —— 这正是 P10-c 唯一的立项理由。
  - **n=266 校准集功效:** paired bootstrap 下 P10-b 已能把 +0.04 级 Δ 的 CI 稳定排除 0(如 72B A-fuse
    +0.0526 CI[+.0333,+.0721]),故 +0.025 级 Δ 在 n=266 上**可测但 CI 更贴 0**,预注册要求 CI 下界 > 0
    **且** 绝对值过 0.616,双条件都满足才晋级。

### B.3 数据外发清单(精确列出会离开集群的内容)

> **这是 (b) 的核心决策依赖项 —— 用户批准数据外发是 P10-c 生效的硬前置。**

- **主体(校准):HateMM train-hateful 校准集 298 视频。** 每视频送:
  - **帧:** M=120 均匀采样帧(K30 用)+ M=16 帧(K4 用),每帧 ≤ **360×420 px**(`max_pixels=360*420`)。
    每次 generation 只送该窗 ~4 帧;跨全部窗的并集 ≈ 每视频 **120 + 16 = 136 帧**。全校准集 ≈ **298 × 136
    ≈ 4.05 万帧 JPEG**(每帧 ≤151k px,粗估 ~0.6–1.2 GB 图像)。
  - **窗级 ASR 文本:** 每窗的 Whisper 英文转写(hateful 视频的**口语仇恨内容**)——这是**最敏感**的外发项。
    体量小(文本,数 MB 量级),但内容是仇恨言论明文。
- **次体(仅晋级配置的单次 test):HateClipSeg test 395 视频**(K30/M120 + K4/M16),仅当某候选晋级后**一次性**
  外发;未晋级则**不外发 test 数据**。
- **许可与伦理注意:**
  - HateMM / HateClipSeg 均为**仇恨视频研究数据集**,其 DUA/license 通常限制再分发;送第三方商业 API =
    第三方处理/潜在再分发,须先核对**各数据集 DUA** 与**各 API 的数据留存/训练政策**(是否用于训练、留存
    窗口、能否 opt-out)。
  - 仇恨内容可能触发 API 内容审核 → **拒答/空返回**是功能性风险(需在预注册写明「parse_ok / 拒答率」为
    一等诊断,拒答样本按缺失处理、不猜分)。
  - 送出前建议:数据最小化(只送打分必需的帧+窗 ASR)、记录外发清单与时间、遵循机构 IRB/伦理审查口径。
  - **此清单本身不构成外发授权;外发须用户显式批准。**

### B.4 粗略成本估计(量级,非报价;每模型一次完整校准 = 6 配置里的 2 行)

- **generation 计数(每模型一次校准):** K30 = 298 vids × 30 windows = **8,940 gens**;K4 = 298 × 4 =
  **1,192 gens**;合计 **~10,132 gens/模型**。A-fuse 是 CPU 重聚合,不额外调 API。三候选 = **~30.4k gens**。
  test(仅晋级 1 配置,一次)= 395 × (30+4) = **~13,430 gens**。
- **每 gen 载荷(粗估):** ~4 帧 @ ≤360×420 px + 短窗 ASR 文本 + 输出 ≤8 token。按各家视觉 token 计费,
  4 帧粗估 **~1–4k 图像 token** + ~50–150 文本 token,输出可忽略。→ 每 gen ≈ **1–5k input token**。
- **量级估算(务必以当日官方定价复核):** 单模型校准 ~10k gens × ~1–5k token ≈ **1e7–5e7 input token**。
  按主流旗舰多模态 input 价（**量级** $1–$15 / 1M token，逐家差异大、须复核）→ **单模型校准约 $10^1–$10^2
  量级**;三候选合计 **约 $10^2 量级**;test 单次 ~13k gens 再加 **$10^1 量级**。**结论:P10-c 全程 API 费用
  大概率在低-中三位数美元量级**,金额上不是障碍;**真正的门槛是数据外发合规 + 闭源不可复现,而非钱。**
  (以上为量级判断,任何实际预算须按各家当日 vision pricing 与实测 token 逐配置重算。)

### B.5 「校准→test 映射不确定性」诚实一段(须写进预注册的风险声明)

> 整个 P10-c 立项依赖一条**只由两个点确定的**校准→test 映射:(校准 0.5387 → test 0.5435)与
> (校准 0.5913 → test 0.5755)。两点连成一条**无误差带的直线**,据此外推「test 0.60 需校准 ≈0.616」。
> 这条外推有三重脆弱性:① **两点定线、零自由度**,无法估斜率不确定性,真实 transfer 可能非线性(P10-b
> 实测 transfer 仅 ~60% 强度:校准 +0.0526 → test +0.0319);② 外推区间(0.5913→0.616)**在观测点之外**,
> 越过冠军后行为未知;③ 校准集(HateMM,n=266)与 test 集(HateClipSeg,n=329)**跨库**,分布差异本身
> 就是 transfer 折损的来源。**因此即便某闭源候选校准过 0.616,test 达 0.60 仍非保证。** 预注册须把这段
> 风险明写,并把「唯一一次 test 触碰」的决策交回用户,而非由校准数字自动触发。

---

## Kit-C —— 换方法族(不做方案设计,只给判断素材)

> (c) 的方案设计是裁决**之后**的事。本 Kit 只回答用户「要不要走这条路」需要的两件事:**现有资产哪些可迁移**、
> **红线下有哪些方向 + 各自最大风险**。

### C.1 可迁移资产(换族不必从零)

- **数据管线:** 4 数据集帧-CLIP + transcript 抽取、Whisper ASR(`data/ASR/`,注意 word-ts DTW bug 致 EN 41%
  降级,复用前修或换 whisperX)、子片段/窗级切分与 gt(`data/gt_p10hate/`)。
- **嵌入缓存:** frozen CLIP / frozen Qwen / LoRA-Qwen 视频嵌入、子片段 mm 嵌入生成器
  (`generate_subclip_embedding_HF.py` / `generate_subclip_mm_embedding_HF.py`)。
- **评估 harness:** `p6_eval_localization.py`(within-video AUC + CI + AP,395-video split)、
  `p10_eval_hatemm.py`(paired bootstrap 10k)、MoRE 复跑 harness(每视频预测落盘可审计)。
- **纪律/模板:** 预注册模板(冻结候选 + 晋级 bar + 多轮多重比较记账)、selection-robustness 五规则网格、
  sha1/bit-for-bit 身份审计、双口径(val-选点 + final-epoch)并排、probe-before-train 双闸门。
- **红线约束:** **NO 重造 codeless baseline**(MultiHateLoc/LELA/TANDEM 仓库空,已标 ABANDONED 留档);
  自建新方法、缺件自补 = 相容;换族若等于「复刻某无代码 baseline」= 触红线。

### C.2 红线下的 2–3 个方向(一句话素描 + 最大风险,供判断)

1. **可训练-MLLM 分类器 + 检索作 in-context(而非 last-token kNN)。** 让 MLLM 本体可训、检索证据以
   in-context 喂入,避开「LMM 头与记忆支柱争容量」。**最大风险:P9/P9b 已警示** —— 决策级把整个 LMM 训成
   分类器只**匹配**、不超我方现有 LoRA-encoder+RGCL(ZH +1.0 vs floor,noise),且 rgcl 只把精度在 head↔memory
   间**再分配**、非净增益;换 in-context 未必绕开该容量争夺。
2. **换中文能力文本塔的检索族(multilingual/CN CLIP、mpnet-zh)。** 直击 P8c 定论的 ZH 真瓶颈(English-centric
   CLIP text tower 把中文 byte-fragment 97% 截断)。**最大风险:这是「换冻结编码器」的另一实验族,大概率
   只救 ZH、不动 EN**,且**不给 MLLM 任何方法角色**(与本 campaign 的 MLLM 目标正交)—— 更像修 ZH 主表的
   工程项,而非新贡献故事。
3. **定位为主线的 reframe(span-free hate localization 作 headline)。** 把论文重心从「主表 accuracy」转到
   **唯一 scale 起作用的定位赛道**(P6→P10-b),可能配一个**可训练定位头**再放大。**最大风险:仍 modest
   (0.5755<0.60)**;且定位 benchmark(MultiHateLoc/LELA/TANDEM)**codeless**,无法在红线内做 head-to-head
   对比,headline 化后审稿人会要同场定位 baseline —— 而那正是红线禁区。

---

*(本文件为 terminus 后决策辅助包,三 Kit 均为「裁决即开工」的第一周动作草案。不含已批准行动;Kit-B 预注册
DRAFT-INACTIVE。生成:2026-07-09。)*
