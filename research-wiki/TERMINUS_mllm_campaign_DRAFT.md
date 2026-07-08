# TERMINUS — MLLM 方法角色 campaign 战略终局报告(定稿)

> **状态:FINAL(定稿,2026-07-09)。** 面向用户决策。campaign 全部在册路线(含最后一条 **P10-b
> scale ladder**,见 §6)判定完毕。本文件只整理**已 commit 的事实与数字**,引用各前沿 EXP 文档的判定
> 原文,不新增测量、不四舍五入到误导。生成:2026-07-08;定稿:2026-07-09(P10-b 落地,§6 占位符补完)。
> 素材:`CAMPAIGN_mllm_method_role.md` + 11 份 `EXP_*.md` + `MORNING_REPORT.md §9` + `EXP_p10_loc_amplify.md`(commit 03880f2)。

---

## 1. 目标原文与判定现状

**用户锁定目标(user-hardlocked):** 让 MLLM **meaningfully and novelly** 地集成进方法,并带来
**substantial performance improvement** —— 即除做冻结 encoder 之外,MLLM 需挣得一个**可被消融的方法角色**:
移除它会在**主表 accuracy** 上可测量地掉点(超过这些 ~150 样本测试集的 ~1.6 视频 ≈ 1 acc 点噪声地板)。
**判定现状:主表 accuracy 角色被彻底证伪。** 11 条预注册路线(auto-repair, P1–P10;含 P2b/P2c 的
7B→32B→72B 规模梯、P3 的 EN/ZH/HateMM 三库、P8b/P8c、P9b 的 rgcl-ON 臂)全部为**诚实 kill 或 within-noise**,
且每条都有**复现 / bit-for-bit / probe 护栏背书**(非 harness 假象)。最后一条在册路线 P10-b 已落地(见 §6):
stronger-scorer scale ladder(Qwen2.5-VL 7B→32B→72B × A-fuse)把定位角色从 modest **放大为 modest-plus**
(72B A-fuse,HateClipSeg 单次 test wv-AUC 0.5755,对 memory +6.2pt / 对 P6-7B +3.2pt 配对显著),但**未达
substantial(0.60)bar**,主表 accuracy 已被证伪的终局不变。MLLM 挣得的、可入论文的角色为:**encoder + 定位
打分器(P6 modest → P10-b modest-plus,显著)+ guard-rail/审计**(详见 §4)。

---

## 2. 路线总表(11 条,各一行)

判定口径统一:1 acc 点 ≈ 1.6 视频;sub-1pt 效应记为 **within-noise, no claim**,看配对 delta 符号而非 p 值;
全程无 cross-seed ensemble。「关键数字」引各前沿 EXP 文档;「免死原因」= 独立于该路线 kill 之外仍存活、可入论文的价值。

| # | 路线 | 机制假设(MLLM 的方法职责) | 关键数字 | 判定 | 免死原因(如有) | commit · doc |
|---|---|---|---|---|---|---|
| 0 | **auto-repair** | 两票 AND 规则(embedding LOO 反对率 ≥0.80 **AND** MLLM 判 CONTRADICT)自动删噪记忆,无人复现手工 2-entry 编辑增益(EN 0.8075→0.8199) | C−A = **+0.0000**(0/4 EN seed);手工删的 2 个 id embedding 反对率仅 0.50/0.60 <0.80,AND 规则**结构性删不到**;C−D = **+0.47pt EN / +0.40pt ZH** | **FAIL**(不复现手工增益,不过 floor) | ✅ **guard-rail:** 语义票**否决** Cleanlab 式 embedding-only 对「真仇恨但 embedding-hard」记忆(虐待证词/性侵报道/含 slur)的过删,是 C>D 唯一来源;并**可审计**(标签盲重找到人审的 2 个噪声 id 且理由正确) | `d4e58aa` · EXP_auto_memory_repair |
| 1 | **P1** 零标注先验重校准 | 读档案→无标注 HARMFUL/BENIGN→adjusted classify-and-count 估先验 p̂→分位重设漂移门控投票阈值(时间协议) | p̂ 误差 **0.22 EN / 0.18 ZH**(criterion ≤0.07);corrected recal 0.48 < static 0.63(EN);ZH forced 0.778→0.723(**−0.055**) | **FAIL**(criteria 2/3/4 全败) | ✅ **量化机制:** oracle-prior 补回 EN 80% 缺口、labelled k=20 补满 —— 阈值旋钮机制成立,只是 MLLM 供不出准先验(判据 FPR 在时间边界漂移 .372→.238) | `2a69246` · EXP_p1_zerolabel_recal |
| 2 | **P2** 7B 邻居重排 | 边界样本按**可比性**(不出标签)删 INCOMPARABLE 邻居再投票,MLLM 误判只稀释不决定 | B−A **−0.002 EN / −0.020 ZH**;过判 INCOMPARABLE **83% EN / 70% ZH**(均删 ~14–16/20 邻居);selectivity lift **+1.1% EN / −3.2% ZH**(删除与投票正确性无关) | **FAIL**(EN within-noise,ZH 净伤) | ✅ **量化天花板:** oracle 成员编辑(按真标签删)把 gated 子集拉到 100%、整体 **+7.5pt EN / +10.6pt ZH,均跨 0.85** —— 门控+prize 真实 | `bc689e1` · EXP_p2_neighbor_rerank |
| 3 | **P2b/P2c** 强判据 + train 端校准 | 7B/32B/**72B** × 证据(archive/+transcript)× prompt(orig/flip)train 端 selectivity 榜,过 +10pt 才碰 test | 最佳 EN lift **+2.7pt**(1/10 到位);**ZH lift 全 8 配置为负**(−2.8…−6.8);calibration 随 scale 涨(orig drop-rate 7B 72.5%→32B 64.6%→72B 30.9% EN)但 **selectivity 不涨** | **FAIL(train 端即死,不碰 test)** | ⚠️ 无独立免死;贡献 = **机制定论**「comparability ⊥ vote-correctness at every open-source scale」,把重排线彻底关闭 | `cc4ca6e`,`aae1efe` · EXP_p2b_stronger_judge |
| 4 | **P3** 证据密度池化(EN/ZH/HateMM) | MLLM 逐段打 hate-evidence density 0–3 → softmax 重加权池化冻结 CLIP 视频嵌入(无标注输入处理) | EN **probe KILL −0.0055@k20**;ZH val −0.0074 / final +0.0088(<1pt);HateMM **最干净 probe +0.0108(k-consistent)** 却训练 val −0.0041 / final +0.0004(<1pt);floor 复现 published 0.828 | **FAIL(三库均无角色)** | ✅ **定位资产:** 逐窗分是无标注 saliency(hate/benign 段内 var 1.11/0.40 EN、1.28/0.71 HateMM),直接喂给 **P6**;方法论定论「**probe 必要非充分**」 | `c2ba59f`,`15f5f08`,`22fe62a` · EXP_p3_evidence_pooling |
| 5 | **P4** schema 字段蒸馏 | 辅助线性头蒸馏档案字段(explicit/modality/mechanism/target,λ=0.1),eval 丢弃头 | bit-for-bit ✓;**probe PASS**(字段可解码 AUC .62–.93、预测标签 .74–.78);train EN final −0.001 / ZH +0.008(sub-threshold),val-sel 双负 | **within-noise, no claim** | ⚠️ 无独立免死;字段真实但**与已直接监督的仇恨标签冗余** | `6f1f0da`,`00816aa` · EXP_p4_schema_distill |
| 6 | **P5** 反事实孪生负样本 | MLLM 洗白每个 TRAIN 正样本转写 → 同视觉+洗白文本 = 每 anchor 一个额外 hard-neg | 质量门 **CLOSED**:self-verdict flip **0.503 EN / 0.337 ZH**(≪0.80);诊断训练伤 **EN −0.027**、flat ZH;cfrand≈cf(pairing 不胜随机) | **FAIL**(前提不成立) | ⚠️ 无独立免死;机制反证:干净孪生因**共享 anchor 视觉过近**(cos 0.73)反伤正样本簇 | `fc25cac`,`66d3103` · EXP_p5_counterfactual_negs |
| 7 | **P6** MLLM 定位打分器 | 逐窗证据分(帧+ASR)为 HateClipSeg 做**无 span 时序定位**排序(记忆无关) | within-video AUC **0.5435** vs memory 0.5140 / random 0.5088;配对 b>a **Δ+0.0296 CI[+.009,+.050] p=0.007**;对空 p=5.4e-8;seg-AUC 0.635 vs 0.584 | **PASS ✅(唯一正例)** | ✅ **本身即幸存角色:** 可移除的定位角色,幅度 modest、统计稳固;P3 池化失败的同一信号是好的 *localizer* | `c9e3bd8` · EXP_p6_mllm_localization |
| 8 | **P7** 分数级融合 | 把视觉 kNN 投票份额与 MLLM 语义通道(bin=P1 verdict / dens=P3 density)在**分数级**用两条冻结规则融合 | train 端 gate FAIL:corr(channel, vote share) **+0.21…+0.51(正相关,证伪去相关前提)**;8 组 rule×channel net **−0.10…−0.38**(伤>修);通道 AUC 0.54–0.69 < floor 0.81–0.86 | **FAIL(train 端 KILL,前提被证伪)** | ⚠️ 无独立免死;贡献 = 实测证伪「decorrelated error channels」直觉(通道与决策变量冗余,且是更弱分类器) | `8f920e5` · EXP_p7_score_fusion |
| 9 | **P8/P8b/P8c** 语义压缩 speech 通道 | MLLM 写 ≤60 词 evidence-dense 摘要作文本通道输入,单 chunk 编码,端到端重训头 | EN probe 开(B 0.7523 > A 0.7359 > 朴素截断 C 0.7067)却训练 **B −0.023/−0.079(0/3),劣于 C**;ZH/HateMM probe 关(朴素截断 C 最优);P8b vision B_vision 0.7409 > B_text 0.7271 但 < C 0.7910;P8c 中文摘要 0.7168 **最差** | **FAIL(全库)** | ✅ **诊断定论:** ZH 瓶颈 = **冻结 English-centric CLIP text tower 把中文 byte-fragment(97% 截断)**,非摘要内容;campaign **最强 probe 却训练不过**(probe 必要非充分最尖锐实例) | `e63d8fe`,`703f4fd` · EXP_p8_semantic_compression / EXP_p8b_vision_summary |
| 10 | **P9/P9b** 决策级 LMM-SFT | LoRA-SFT 整个 Qwen2.5-VL LMM + 自带 classifier 头(C3);两读出:in-LMM MLP 头 与 **我方 kNN over SFT'd embeddings**;P9b 加 rgcl-ON 臂(D3,用我方检索对比损失训练嵌入空间) | C3-mlp EN 0.7909(**+0.6 noise**)/ ZH 0.8635(**+1.0 vs 协议匹配 LoRA floor 0.8537,noise**);**C3-knn EN −2.7 / ZH −2.2 / HateMM −4.7 BELOW floor**;P9b D3-knn ZH 0.8389(**−1.5**)/ EN 0.7743(**−1.0**);D3−C3′ 机制 = head↔memory **±1.8pt 再分配**,非净增益 | **FAIL(最后架构 locus 关闭)** | ✅ **论文发现:** 首次把 RA-HMD(released rgcl-OFF)LMM-RGCL stage-2 **成功 port 到 video**(5 处 fork 修复 + bs=1 in-batch 退化 + 4-frame/bs4 修复);rgcl-ON 使 knn 读出相对头**修复**(C2 PASS);清晰「gain 不来自 memory、也不超我方现有 LoRA」负结论 | `455e666`,`4d28655` · EXP_p9_lmm_rgcl_video |
| 11 | **P10** 放大定位角色 | 在 HateMM span 上自由标定 scorer、单次测 HateClipSeg;把 P6 的 modest 定位增益做到 substantial | anchor HateMM wv-AUC 0.5387;**A-fuse(K4×K30 coarse×fine)最佳 +0.0305,显著 CI[+.0175,+.0437] p=7e-7 但 < +0.04 bar**;K60/fewshot/A-gate/A-lex 均 ≤anchor 或 ≤+0.006 | **FAIL / no promotion**(HateClipSeg test 未触,P6 as-is 站住 0.5435) | ✅ **A-fuse 杠杆:** 唯一显著移针的放大器(+0.03),是「若换更强 scorer」的自然起点(→ P10-b) | `7194ee2` · EXP_p10_loc_amplify |

---

## 3. 横切机制结论(为什么 7B–72B 开源 MLLM 在这套检索记忆方法里拿不到主表精度)

1. **语义能力 ⊥ 决策变量(最统一的失败形状)。** MLLM 有真实语义能力(会读档案、能定位证据、字段可解码
   AUC .62–.93),但该能力**与决策变量正交或冗余**:comparability ⊥ vote-correctness(P2/P2b 全 8 配置 |lift|≤2.7pt,
   ZH 反号);localized-visual-evidence ⊥ 冻结-CLIP 可分性(P3);verdict-rate 在其要估计的先验上漂移(P1);
   schema 字段 ⊂ 已监督的标签(P4)。语义「关于什么」不等于「在仇恨/冒犯/良性边界的哪一侧」,而后者正是
   检索头**已直接监督**的量 —— 主表提升要移动的正是这条已被监督的边界。

2. **calibration 随 scale 涨、selectivity 不涨(P2b/P2c 规模梯定论)。** 7B→32B→72B,orig-prompt drop-rate
   单调收敛(EN 72.5%→64.6%→30.9%,ZH 58.2%→50.7%→14.9%):更大的判据**更守规矩、不再 trigger-happy**;但
   selectivity(是否偏删会误投的邻居)**全程钉在 ~0**(EN lift 从不超 +2.7,ZH 每一档为负)。**把判据做大只是让它更
   well-behaved,不让它的可比性判断追踪 label-relevance。** 这是「更强闭源模型能否救活重排/判据线」最直接的反证据。
   **唯一例外在定位赛道(与本条恰成对照)。** P10-b(§6)证明 **scale 在 localization 上确实起作用**:A-fuse×scale
   在 HateMM 标定集单调 7B +0.0305 → 32B +0.0437 → 72B +0.0526,72B 迁移到 HateClipSeg wv-AUC 0.5755(对 P6-7B
   +0.0319 配对显著)。区别在于重排里 MLLM 的语义能力(可比性)⊥ 决策变量、scale 只改 calibration 不改 selectivity;
   而定位里 MLLM 的语义能力(段内仇恨 saliency)**就是**被评的目标量,故更大的 scorer 能被 metric 直接兑现。
   localization 因此是这套方法里**唯一** scale 能移动指标的赛道。

3. **过 no-head probe 是必要非充分(方法论定论,两处最尖锐)。** P3-HateMM 是三库最干净 probe(+0.0108,k-consistent,
   证据最密)却训练 within-noise;P8-EN 是**全 campaign 最强 probe**(摘要 probe 同时压过 floor +1.6pt 与朴素截断 +4.6pt)
   却训练劣于朴素截断。机制:**习得的 align-fusion 头(img×text elementwise)吸收输入端重加权 / 更会利用原始(哪怕被稀释的)
   文本**,把 probe 层面的输入端优势冲掉。→ probe 通过不保证训练增益,须双口径(val-selected + final-epoch)双看。

4. **P9b 的 head↔memory 再分配,不是净增益(最后架构 locus)。** 在 [1,1,1] 等权下,rgcl 对比项把精度从 LMM 自带头
   **搬到**我方 kNN 读出(ZH 近乎 ±1.8pt 精确对调:D3-knn−C3′-knn +1.8pt,D3-mlp−C3′-mlp −1.8pt),而非新增。
   rgcl-OFF 的 C3-knn 更是全库 BELOW floor(EN −2.7 / ZH −2.2 / HateMM −4.7)。**MLLM 自带头与 memory 支柱在此
   regime 争同一容量:LMM 头 displaces 而非 enhances memory。** 决策级把整个 LMM 训成分类器,也只是**匹配**我方现有
   LoRA-encoder+RGCL 路线(ZH +1.0 vs 协议匹配 floor,noise),不 substantial。

5. **ZH 的独立瓶颈 = 冻结编码器,不是 MLLM(P8c 归因)。** ZH 上任何 MLLM 摘要(text/vision、EN/CN)都不敌**朴素
   raw-ZH 单 chunk 截断 C 0.7910**;强制中文摘要 0.7168 最差,因 **English-centric CLIP text tokenizer 把中文
   byte-fragment**(≤90 汉字 → ~140 CLIP token,97% 截断)。ZH 的真实杠杆是**换中文能力文本塔**(multilingual/CN CLIP、
   mpnet-zh),那是「换冻结编码器」的另一实验族,不属本 MLLM-方法角色 campaign。

---

## 4. 幸存角色(如实,不夸大)

1. **encoder(已入主表)。** Qwen 特征在 HateMM 上比 CLIP **+4.2 macro-F1 且跨 0.85** —— 这是 MLLM 唯一进主表
   accuracy 的身份,但它是「冻结 encoder」而非本 campaign 追求的「新方法角色」。

2. **定位打分器(唯一挣得的可移除方法角色,P6 modest → P10-b modest-plus,显著)。** 逐窗证据分把 HateClipSeg
   时序定位从「存在性证明」(memory wv-AUC 0.526,4 cell 中仅 1 显著)升级为**显著 localizer**:P6-7B wv-AUC
   **0.5435**,CI[+.533,+.554],p=5.4e-8,配对超 memory **+0.030,p=0.007**;**P10-b 用 72B A-fuse 把它进一步放大到
   wv-AUC 0.5755**(CI[0.5581,0.5933],sign-p 1.4e-9,n=329;对 memory **+0.0615** CI[+.0359,+.0869]、对 P6-7B
   **+0.0319** CI[+.0170,+.0474],两 CI 均排除 0)—— 三档判定 **modest**(0.56≤0.5755<0.60,未达 substantial 0.60)。
   诚实幅度警告:仍是 modest-plus(高出 chance ~7.5 AUC 点),MLLM 在此的主导能力其实是 **video-level density**
   (broadcast AP 0.62),细粒度 within-window 定位是更小但统计稳固、且随 scorer 规模单调增长的增量。

3. **guard-rail / 审计(可控性,非 raw acc)。** auto-repair 的语义票**否决** embedding-only 对真仇恨记忆的过删(C>D);
   可编辑档案记忆支持定向删噪(人审 2-entry 删除改善 EN);标签盲档案审计重找到人审噪声 id —— 移除代价体现在
   **完整性/可控性**,是一条 defensible 的贡献口径,但**不是主表 accuracy**。

4. **A-fuse × scale 杠杆(P10-b 已兑现为 modest-plus)。** P10 round-1 中 coarse×fine(K4×K30)融合是唯一在标定集
   **显著移针**的放大器(7B +0.0305,CI 排除 0,p=7e-7),但低于 +0.04 promotion bar,round-1 未触 test。**P10-b
   把该杠杆沿 scale ladder 爬高兑现**:A-fuse×scale 在 HateMM 标定单调 7B +0.0305 → 32B +0.0437(过线)→ **72B
   +0.0526 CI[+0.0333,+0.0721](最高 Δ,唯一晋级)**;而 raw-K30 规模单独走不过线(7B 0.5387→32B 0.5512→72B
   0.5593)。72B A-fuse 晋级并花掉唯一一次 HateClipSeg test → wv-AUC 0.5755(modest,见上第 2 条)。**localization
   放大线到此关闭**,P6→P10-b 为最终定位数。

---

## 5. 三个决策选项(给用户拍板)

> 用户红线备忘:**NO cross-seed ensembles / NO 重造 codeless baselines / NO 作者邮件(缺失代码自己补)。** 三选项均据此评估相容性。

### 选项 (a) —— 改目标口径:接受 localization + encoder + guard-rail 为 MLLM 的角色故事

- **做法:** 放弃「MLLM 挣得主表 accuracy 角色」的表述,把论文的 MLLM 贡献重定为三件套:**encoder(HateMM 跨 0.85)+
  可移除定位角色(P6 modest → P10-b 72B A-fuse **modest-plus**,wv-AUC 0.5755)+ 可编辑/可审计记忆的 guard-rail**。
  方法学章附「11 路线全负 + 两条定论」作为强负结果与 ruled-out map。
- **代价:** 近零额外计算(全部已 commit);叙事上放弃「substantial main-table improvement」的强 claim,需说服自己/审稿人
  接受「定位 modest-plus + 完整性/可控性」的较弱贡献口径。
- **预期收益:** 立即可定稿;负结果链(尤其 comparability⊥vote-correctness 规模梯、probe 必要非充分两条方法学定论)
  本身有发表价值;定位正例现被 **P10-b 加固**——定位角色升为 72B A-fuse **0.5755**,对 memory **+6.2pt** 配对显著
  (CI[+.0359,+.0869]),比 P6-7B 的 +3.0pt 更强,是干净且幅度更大的正例(仍诚实标注 modest,未达 0.60)。
- **红线相容:** **完全相容**(不需任何新实验/新 baseline/邮件)。**风险最低、可立即执行。**

### 选项 (b) —— 上更大闭源模型(GPT/Gemini/Claude API)重跑关键路线

- **候选路线(按「是否可能翻案」排序):** ①**P2/P2b 重排**(oracle 头空间 +7.5/+10.6 跨 0.85,是唯一有大 prize 的
  route);②P1 先验估计;③定位放大器(A-fuse + 强 scorer)——**此子目标已被开源 72B 在 P10-b 兑现为 modest-plus
  0.5755**,闭源 API 只剩「把 0.5755 推到 0.60+」这一段增量;④P5 反事实洗白(flip rate 是模型能力瓶颈)。
- **代价:** ①API 费用(P2 判据每语言 ~5k–10k pairs × 多配置,是 token 大头);②数据出域(仇恨内容送第三方 API 的
  合规/伦理审查);③闭源不可复现、不可作为「方法组件」写进可开源 pipeline(审稿人会质疑可复现性)。
- **预期收益(据 §3.2 定量外推,偏保守):** P2b 规模梯已证 **selectivity 不随 scale 涨**(comparability⊥vote-correctness
  是**机制**而非**执行力**问题),故换更大模型**最可能仍不选择性** —— 翻案概率低。P1 的瓶颈是 verdict FPR 跨时间边界漂移,
  更强模型可能降低绝对 FPR 但漂移方向未必消失。**唯一相对乐观的仍是定位放大器,且已被 P10-b 部分证实**:开源
  scale ladder(A-fuse×规模 7B +0.0305→32B +0.0437→**72B +0.0526**,单调)已把定位从 modest 抬到 modest-plus
  (HateClipSeg 0.5755)—— 闭源模型能否把 **0.5755→0.60+**(清 substantial bar)属未知,但 7B→32B→72B 的单调梯度
  给了正向外推依据(唯一 scale 起作用的赛道,§3.2)。代价是那放大的仍是**定位**角色、非主表 accuracy,且闭源不可
  复现、不可写进可开源 pipeline。
- **红线相容:** 不触 cross-seed/baseline/邮件三条红线;但**新增「送外部 API」的合规维度**,须用户确认数据可外发。
  **中风险、中成本,主表翻案的期望收益低,定位放大的期望收益中等。**

### 选项 (c) —— 换方法族(离开「冻结检索记忆 + kNN 投票」骨架)

- **诊断依据:** §3.5 与 P8c 指出**冻结编码器**是 ZH 的真瓶颈;§3.4 指出 memory 支柱与 LMM 头**争容量**。若要让 MLLM
  真正带来主表增益,可能需换到 MLLM **本身可训练**且不与检索记忆争容量的架构(如端到端 MLLM 分类器 + 辅助记忆检索作
  in-context,而非 last-token embedding kNN;或换 Chinese-capable 文本塔的检索族)。
- **代价:** 最大 —— 放弃当前四支柱叙事(检索对比+kNN 核心、可更新记忆、共识去噪、可审计记忆)的既有资产;需重做
  实验基座;时间成本以周计。
- **预期收益:** 上限最高(理论上可拿主表增益),但**不确定性最大**,且 P9/P9b 已给出警示:决策级把整个 LMM 训成
  分类器也只匹配、不超我方现有 LoRA。
- **红线相容:** 若换族意味着「重造某个 codeless baseline」则**触红线**;若是自建新方法则相容(缺失代码自己补符合红线)。
  **风险最高、周期最长;仅在用户判定「弱贡献口径不可接受、必须拿主表增益」时才值得。**

**一句话权衡:** (a) 立即可交、零风险、claim 已被 P10-b 加固(定位 modest-plus);(b) 中成本、主表翻案期望低,而
「定位放大」子目标**已被开源 72B 兑现为 modest-plus 0.5755**、闭源只剩 0.60+ 增量、需数据外发许可;(c) 上限最高但
周期/不确定性最大。**当前证据链最支持 (a);(b) 的定位放大期望已部分实现(开源到 modest-plus),闭源续推至
substantial 有单调梯度依据但仍属未知。**

---

## 6. P10-b —— 定位放大器的 stronger-scorer scale ladder(FINAL:MODEST amplification)

> **P10-b = campaign 最后一条在册路径。** 它问:一个**更强的定位 scorer** 能否把 P6 的 modest 定位增益放大到
> 预注册的 substantial bar?round-1(P10,commit 7194ee2)已 FAIL —— A-fuse(K4×K30 coarse×fine)是唯一显著移针的
> 放大器(7B +0.0305,CI[+.0175,+.0437],p=7e-7),但低于 +0.04 promotion bar。P10-b 沿 Qwen2.5-VL scale ladder
> (7B→32B→72B)把 scorer 爬高,配 round-1 的胜出聚合(A-fuse),用**未改动**的 promotion bar 复测。
> 治理链:预注册 **3d641f4**(冻结 5 候选 R2-1..R2-5、+0.04 晋级线、两轮 11 比较记账,bar 不为 round-2 松动);
> 执行 bug 修复 **c5c47ee**(32B bf16 coarse-pass OOM → expandable_segments + 逐视频 empty_cache,score-neutral)、
> **e69065f**(coarse pass 必须 M=16 而非 M=120,匹配 P3-default 配方);校准落地 **24de185**;最终结果 **03880f2**。
> 全部数字见 `research-wiki/EXP_p10_loc_amplify.md`(该文件已 commit 03880f2,本节仅引用、不重测)。

- **状态:FINAL(2026-07-09)。** 校准 round-2 与单次 HateClipSeg test 均已落地并 commit。

### 6.1 HateMM 校准 leaderboard(两轮 11 比较 vs 冻结 7B anchor 0.5387;bar = paired Δ ≥ +0.04 且 CI 排除 0)

| round | variant | HateMM wv-AUC | paired Δ vs anchor | paired Δ 95% CI | 过 bar |
|---|---|---|---|---|---|
| — | **anchor**(7B,raw K30) | 0.5387 | — | — | — |
| 1 | A-gate / K60 / fewshot | 0.5314 / 0.5319 / 0.5359 | −0.0074 / −0.0068 / −0.0028 | 均含 0 | no |
| 1 | A-lex | 0.5450 | +0.0062 | [−0.0000, +0.0123] | no |
| 1 | A-fuse(7B) | 0.5693 | +0.0305 | [+0.0175, +0.0437] | no(Δ<+0.04) |
| 2 | R2-5 · 7B A-fuse×A-lex(CPU) | 0.5752 | +0.0365 | [+0.0223, +0.0506] | no(Δ<+0.04) |
| 2 | R2-1 · 32B anchor-agg | 0.5512 | +0.0125 | [−0.0006, +0.0257] | no |
| 2 | R2-2 · 32B A-fuse | 0.5825 | +0.0437 | [+0.0240, +0.0631] | **yes** |
| 2 | R2-3 · 72B anchor-agg | 0.5593 | +0.0206 | [+0.0065, +0.0347] | no(Δ<+0.04) |
| 2 | **R2-4 · 72B A-fuse** | **0.5913** | **+0.0526** | **[+0.0333, +0.0721]** | **yes — 最高 Δ,晋级** |

**两条干净梯度:**(a) **raw-K30 规模单独走不过线** —— anchor-agg 单调 7B 0.5387 → 32B 0.5512 → 72B 0.5593,但
72B 的 Δ(+0.0206)仍只有 gate 的一半;(b) **A-fuse × 规模是唯一杠杆** —— coarse×fine 融合增益随 scorer 增长
7B +0.0305 → 32B +0.0437(过线)→ **72B +0.0526**;32B/72B A-fuse 均清未改动的 bar,按冻结规则(最高 paired Δ)
**R2-4(72B A-fuse)单独晋级**。R2-5 把 round-1 两个 CPU 胜者叠在 7B 分上只到 +0.0365 —— 缺的是 scorer 强度而非聚合。

### 6.2 HateClipSeg 单次 test(冻结 P6 harness,promoted R2-4;控制组逐位复现 P6)

单次 test pass(job 12585,72B bnb4,395 视频,K30/M120 + K4/M16,5h50;K4 ASR 于 CPU 从存储 chunk 时戳重分箱,
无 Whisper 重跑),fuse 于 CPU,eval = **冻结 P6 harness**(`p6_eval_localization.py`,同 395-video split、同估计量);
harness 完整性经「用 default tag 逐位复现已发表 P6 数字」预验证,控制组(memory 行、random)复现 P6 exactly。

| condition | frame AP / AUC | seg AP / AUC | **within-video AUC** |
|---|---|---|---|
| a — memory `knn_hatemm_subclip` | 0.5329 / 0.5754 | 0.5246 / 0.5839 | 0.5140 |
| **b — R2-4(72B A-fuse,promoted)** | 0.5929 / **0.6488** | 0.5948 / **0.6561** | **0.5755** |
| d — random | 0.4699 / 0.5084 | 0.4507 / 0.5065 | 0.5088 |
| *(P6 参考:b at 7B)* | 0.5421 / 0.6034 | 0.5599 / 0.6353 | 0.5435 |

- **within-video 主指标:** R2-4 wv-AUC **0.5755**,bootstrap 95% CI **[0.5581, 0.5933]**,sign-p **1.4e-9**(n=329)。
- **paired vs memory**(0.5140):Δ **+0.0615**,CI **[+0.0359, +0.0869]**,sign-p 4.9e-5。
- **paired vs P6-7B**(0.5435):Δ **+0.0319**,CI **[+0.0170, +0.0474]**,sign-p 0.0024 —— 校准侧承诺(对 7B anchor
  +0.0526)以 ~60% 强度迁移到 test。
- 支撑指标同向:frame AUC 0.6034→0.6488、seg AUC 0.6353→0.6561;broadcast 控制(video-mean AP 0.62)仍是最高 pooled
  AP,即 video-level density 仍是 MLLM 的主导能力,但 within-window 增量现已更大且与 P6 baseline CI 分离。

### 6.3 三档判定 —— **MODEST amplification**

- substantial(wv-AUC ≥ 0.60):**未达**(0.5755 < 0.60)。
- **modest(0.56 ≤ wv-AUC < 0.60 且 CI 排除 P6 的 0.5435):MET** —— 0.5755 ∈ [0.56, 0.60) 且 CI 下界 **0.5581 > 0.5435**
  (paired-vs-P6 的 CI 亦排除 0)。第二轮 / 11 比较 caveat 如预注册声明。
- 定位角色由 **modest(7B)升为 modest-plus(72B A-fuse)**:earned-roles 判定(encoder + localizer + guard-rail/审计,
  无主表 accuracy 角色)**性质不变、程度加强**。唯一一次 HateClipSeg **test 触碰已花掉**;campaign **最后一条在册路径关闭**。

### 6.4 对本报告其余小节的影响(已同步)

- **§1、§4:** 定位角色数字与措辞已从「P6 modest 0.5435」更新为「P6 → P10-b modest-plus 0.5755」。
- **§3.2:** 已补一条对照定论 —— localization 是这套方法里**唯一** scale 能移动指标的赛道(A-fuse×规模单调,72B 迁移
  wv-AUC 0.5755),与重排线「scale 改 calibration 不改 selectivity」恰成对照。
- **§5:** 选项 (a) 证据加固(定位现为 72B+fuse 0.5755,对 memory +6.2pt 配对显著);选项 (b) 的「定位放大」子目标已被
  **开源 72B 兑现为 modest-plus**,闭源能否把 0.5755→0.60+ 属未知但 7B→32B→72B 单调梯度给了外推依据。
- **主表 accuracy 终局不变:** P10-b 只把定位角色从 modest 升到 modest-plus,**不改**「主表 accuracy 角色被 11 路线
  证伪」的终局;§3 五条横切定论与 §4 幸存角色(encoder + 定位 + guard-rail)结构不变。

---

*(本报告只汇总各前沿 EXP 文档已 commit 的判定与数字,不新增测量。P10-b(commit 03880f2)落地后 §6 占位符已补完,
本文件定稿为 FINAL。)*

---

## EXPLORATORY APPENDIX(不用于晋级 / NOT FOR PROMOTION)

> **性质声明。** 本附录是 **P10 终局之后**追加的一次**纯 CPU 探索性天花板分析**,回答唯一一个
> 问题:在**已落盘的打分文件**里,还有没有任何组合能把 HateMM 校准 wv-AUC 推到 **≈0.616**(该
> 水位按 校准→test 两点映射外推,方可望 test 达 0.60 的 substantial 线)。
> **它不改动上文任何 FINAL 正文,不改预注册,不作任何晋级依据。** 全程**未碰 test、未读任何
> HateClipSeg 文件、未提交 SLURM、未做任何 GPU 打分**;所有行都是对 §6 已落地的 7B/32B/72B
> K30+K4 分数的 **re-aggregation**(fuse/lex 数学复用 `p10_aggregate_b.py` 的注册函数,逐位一致;
> 生成脚本 `scripts/analysis/p10_explore_ceiling.py`,输出以 `p10-xplor-*` 标记且 gitignore)。
> 明确不做:跨模型分数融合(NO ensembles 红线之近亲)、任何新打分。
> 评估仍用 `scripts/analysis/p10_eval_hatemm.py`(n=266,paired bootstrap 10k vs 7B anchor 0.5387)。

**探索行表**(校准 wv-AUC 主指标;"Δ vs anchor" = 对 7B anchor 0.5387 的 paired Δ [95% CI];
"Δ vs 冠军" = wv-AUC − R2-4 冠军 0.5913;"距 0.616" = 0.616 − wv-AUC,正数=尚差):

| 配置 | 来源 | 校准 wv-AUC | Δ vs anchor(paired,CI) | Δ vs 冠军 0.5913 | 距 0.616 目标线 |
|---|---|---|---|---|---|
| *(参照)* anchor 7B raw K30 | §6 | 0.5387 | — | −0.0526 | −0.0773 |
| *(参照)* 7B fuselex(R2-5) | §6 | 0.5752 | +0.0365 [+0.0223,+0.0506] | −0.0161 | −0.0408 |
| *(参照)* **冠军 R2-4 · 72B A-fuse** | §6 | **0.5913** | +0.0526 [+0.0333,+0.0721] | — | −0.0247 |
| **① 72B fuse×lex** | 本附录 | **0.5932** | **+0.0544 [+0.0348,+0.0742]** | **+0.0019** | **−0.0228** |
| ② 32B fuse×lex | 本附录 | 0.5849 | +0.0462 [+0.0267,+0.0657] | −0.0064 | −0.0311 |
| ③a 72B fuse w21(K4:K30=2:1,coarse-heavy) | 本附录 | 0.5909 | +0.0521 [+0.0314,+0.0734] | −0.0004 | −0.0251 |
| ③b 72B fuse w12(K4:K30=1:2,fine-heavy) | 本附录 | 0.5887 | +0.0500 [+0.0316,+0.0686] | −0.0026 | −0.0273 |

- ③ 的两行是对 A-fuse **同一线性 blend** `w·K30 + (1−w)·K4` 的**权重系数敏感性**(默认 0.5/0.5;
  预注册的 fuse 是定值权重、无旋钮,故此二行仅为探索性 coefficient sweep,**非新机制**)。
- 内部一致性核对:本附录 fuselex 配方在 7B 分数上逐位复现 R2-5 的 0.5752 / +0.0365,冠军
  72B A-fuse、32B A-fuse 亦逐位复现 §6 的 0.5913 / 0.5825,harness 可信。

### 天花板结论(EXPLORATORY,不用于晋级)

**剩余杠杆池最强配置 = 72B fuse×lex,校准 wv-AUC 0.5932 —— 仅比现冠军(72B A-fuse 0.5913)高
+0.0019,离 0.616 目标线仍差约 0.023(且这 0.023 与「anchor→冠军」整段爬升 0.0526 同量级的一半,
远超 re-aggregation 能挤出的余量)。** 两条内部梯度显示杠杆池已**饱和**:(i) blend-weight 敏感度
**平坦** —— 把 K4:K30 从 1:2 转到 2:1,wv-AUC 只在 0.5887–0.5913 间浮动(默认 0.5/0.5 已在局部最优,
无权重能超过冠军);(ii) ASR 词典叠加的**边际增益随 scorer 增强而衰减**(7B A-fuse→fuselex +0.0059、
32B +0.0024、72B +0.0019)。据此判断:**substantial 线(校准 ≈0.616 → 外推 test ≥0.60)在现有
scorer 池内 _不可达_** —— 现有 7B/32B/72B 分数的任何合法 re-aggregation 都封顶在 ≈0.593,越过 0.616
需要一个**更强的 scorer 或不同模态**,而非重聚合。此结论**只是天花板判断,不改变 P10-b 的 FINAL
定论(MODEST amplification,test 已花掉、不再触碰)**。

*(EXPLORATORY 附录结束。上文 FINAL 正文不受影响。)*

---

## P10-c:开源代际跳跃(2026-07-09,FAIL)

> **追加小节,不改动上文任何 FINAL 正文 / EXPLORATORY 附录。** P10-c 是 campaign 之外的最后一次开源探针:
> 换**代**(Qwen3-VL)而非换**规模**,能否越过 EXPLORATORY 附录判定的 re-aggregation 天花板(校准 0.5932
> < 0.616)。全部数字见 `research-wiki/EXP_p10_loc_amplify.md` 的 P10-c 节(commit `74f0eac`),本小节仅引用、
> 不重测。

- **预注册(`8810c11`):** Qwen3-VL-32B(dense)+ Qwen3-VL-30B-A3B(MoE,3B active)× {anchor-agg, A-fuse},
  第三轮门槛 = **校准 wv-AUC ≥ 0.616 且 CI(Δ) 不含 0** —— 即两点校准→test 映射外推 test ≥ 0.60 所需的功效分析
  水位(高于已花掉 test 的 72B 冠军 0.5913,是 sequential-testing 控制);累计 **14 比较** vs 7B anchor 0.5387。

- **结果:** 四臂全部低于 0.616。最佳 **C1b(Qwen3-VL-32B A-fuse)校准 wv-AUC 0.5866**,paired Δ vs anchor
  **+0.0479 CI[+0.0287, +0.0677]**(显著但 < gate),且**低于** Qwen2.5-VL-72B A-fuse 冠军 0.5913 → 按预注册
  **HateClipSeg test 未触碰**,**P10-b 的 0.5755 MODEST 定论仍立**。(其余三臂:C2b 30B-A3B A-fuse 0.5821、
  C1a 32B anchor-agg 0.5594、C2a 30B-A3B anchor-agg 0.5469。)

- **机制定论:**(i)**换代 ≠ 换规模** —— Qwen3-VL-32B 在两种聚合上都落在 Qwen2.5-VL-32B 的噪声内(anchor-agg
  0.5594 vs 0.5512;A-fuse 0.5866 vs 0.5825),新一代 32B ≈ 两代前的 32B 档;(ii)**30B-A3B(3B active)最弱**
  (anchor-agg 0.5469,CI 含 0)→ **定位能力由激活参数量主导**,非总参数量、非代际;(iii)**A-fuse 显著性在第 5 个
  scorer 复现**(7B / 32B / 72B / Qwen3-32B / Qwen3-30B-A3B 全部 A-fuse 显著高于同模型 anchor-agg),再证
  coarse×fine 融合是唯一杠杆。此读数**连带排除 Qwen3-VL-235B-A22B**(22B active,按激活参数律预期不过 32B 档;
  且本集群 A100 无 FP8、471 GB bf16 磁盘不可行)—— 记为超范围。

- **结论:** 开源可行域的**三面墙** —— **重聚合**(EXPLORATORY 天花板 0.5932)/ **规模梯**(72B A-fuse 0.5913)/
  **代际同档**(Qwen3-VL-32B A-fuse 0.5866)—— 现已全部闭合,**substantial 线(校准 ≈0.616 → test ≥0.60)在本
  集群开源域正式不可达**。**对 §5 选项 (b) 的读数(不改正文,记于此):开源代际已排除,闭源是 scorer 类别上唯一
  未测的增量。** 主表 accuracy 终局与 §4 幸存角色(encoder + 定位 + guard-rail)不受影响;P10-b 仍是最终定位数
  (0.5755,MODEST)。

*(P10-c 小节结束。上文 FINAL 正文与 EXPLORATORY 附录均不受影响。)*
