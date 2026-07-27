# DECISION MEMO — 悬置用户决策合并单(一页清空版)

> **性质声明(读前必看)。** 本备忘录**只汇总当前所有悬置的用户决策**,**不含任何已批准行动**;
> 每项的「我方推荐」仅是建议,**一切裁决以用户答复为准**。全部理由引用**已 commit** 的证据(文档 + commit)。
> 汇总自:`TERMINUS_mllm_campaign_DRAFT.md`(FINAL)、`OPTION_KITS_terminus.md`、`MORNING_REPORT.md §7/§9`、
> `experiments/exp-archive-knn-seeds.md`、`experiments/exp-consensus-zh-seeds.md`、`PAPER_MASTER_TABLES.md`。
> 生成:2026-07-09 · 基线 HEAD `ea83142`。**六项一次拍完即全部清空。**

---

# ★ 现行裁决单(2026-07-28 更新,round-8 / F88–F98 之后)

> **读法:下面 S1–S4 是当前唯一在等用户的四项裁决**;原 D1–D12 一节整体降为**存档**(仍保留原文与证据链,
> 逐项标注去向,见本节末「D1–D12 去向对照」)。四项都**不是实验**——它们要么是交付物定义,要么是报告口径,
> 要么是资源/语料许可;任何一项在裁决前**都不排 GPU**。证据全部引用已 commit 的记录 + commit 号
> (numeric-provenance:数字在写入时逐条从原始记录重读)。

## S1 — 范式移动:是否把**输出对象**从「全覆盖二分类」换成「三路认证输出 + 策略化 operating point」

- **问题:** round-8 把「用更好的**统一决策规则**去兑现排序优势」这条轴**从六个方向关闭**(vote 算子 F89 /
  邻域深度 F94 / 阈值 F88 / 训练损失 F75 / pair verifier F95 / per-item 门 F97)。LITSWEEP-6 的 paradigm lane
  因此不再推荐第七条规则,而推荐**换输出对象**:**R1** = 三路认证输出 {hate, non-hate, **⊥ refer**},
  以**关系型 nonconformity**(复用 F95 冻结的 verifier)做 split-conformal,给出**分布无关的**「自动判定子集
  错误率」上界;**R2** = 把 operating point 变成 **anytime-valid、drift-adaptive 的策略**(ACI 类更新,O(1)
  标注预算),锚在我方自有的 W4 时间漂移测量上。**要不要走这条路,是交付物定义问题,不是实验问题。**
- **为什么现在问:** 组织性事实已经四次独立测得——**排序质量 ≫ 决策质量**:F95(关系层 +0.13–0.27 pair-AUC,
  18/18 cell,端到端 **0/36**)、W4(EN 时间切分 **ROC 0.8484** > 随机切分 0.7175,macro-F1 却 **−0.084**)、
  F88(正确类比项在**中位 rank ~1.5** 却被压倒,误差 ~90% seed-invariant 的 confident inversion)、
  F50/F48(dev AUC **0.898**「unconvertible」)。R1 正是**唯一把这条事实当前提而不是当障碍**的范式。
- **代价与硬约束(必须与推荐一起读):** (i) **按构造**不满足「+3 acc on ≥2 数据集」的全覆盖目标——在
  100% coverage 下它什么都不改;(ii) conformal 的 exchangeability 要求一个**未被选点消耗的**校准集 ⇒ 与 S2
  **耦合**(retirement 让 dev 变成合法校准集);(iii) 我们**没有**人工审核成本/准确率数据,「referred = gold」
  是 L2D 常规但会被读成 trick,须显式成本核算;(iv) 1 test item = 0.47–0.67% ⇒ risk-coverage 曲线是粗台阶,
  必须打印每 bin 计数并以配对符号检验承载主张。
- **我方推荐:** 若用户接受「交付物 = 可认证的审核流程」而非「更高的全覆盖 accuracy」,则**走 R1**(先跑
  已完全银行化的 **$0 pregate**:关系型审计统计量 vs `|vote margin|` 的 risk-coverage 曲线,预注册
  「B 须在 {0.70, 0.80, 0.90} 三个 coverage 点中 ≥2 个、在 ≥2 个数据集上、全 seed 同号地胜过 A」),R2 作为
  第二篇/第二节;若用户坚持全覆盖 accuracy 目标,则**明确 S1 = 否**,并据此接受当前 box 为空的结论。
- **来源:** `refine-logs/LITSWEEP6_PARADIGM.md` · `49e15ec`(R1 §(a)–(f)、R2 §(a)–(f)、PRE-KILLS §);
  `LITSWEEP6_MEMBANK.md` · `62efd82`;`LITSWEEP6_RELGEN.md` · `f62e777`;`MECHNOV_PAIRVERIFY_PREGATE.md` ·
  `0261b82`;`VGA_PREGATE_RECORD.md` · `db2eae8`;`AGGNET_PREGATE_RECORD.md` · `fa1e3b3`;
  `research-wiki/EVAL_temporal_memory_W4.md`。

## S2 — ZH 主表协议:是否**退休** 78-样本 val-selected 选点(承接原 D2,证据包已大幅加强)

- **问题:** ZH 主表用 val-选点(3-seed **0.8322 / 0.8015**)还是 selection-free final-epoch
  (**0.8456 / 0.8173**)?**退休 = 实测 +0.0134 acc / +0.0159 mF1,3/3 seed**(不是 oracle)。
- **Tier-1 证据包(全部 bit-exact,来自 `ERRPAT_MHC-ZH_2026-07-26.md` §1,commit `ad56a62`;每条都只依赖
  78 项 dev 的性质、与任何 ZH 结果无关):**
  1. **差距恒为 2 个 test 样本**:三个 seed 的 dev argmax 都是 **0.8718 = 68/78**,ep29 只低 1–2 个 dev 样本;
     协议因此丢弃 final epoch,每个 seed 各付 **+0.0134(+2 项 / 149)**,三次一致。
  2. **dev 信号不是弱,是与目标反相关**:25 个合法 epoch 上 dev-acc 与 test-acc 的 Spearman 逐 seed
     **−0.3457 / +0.0419 / −0.1531**,**pooled −0.2402,p = 0.0380(显著为负)**——读它的选择器指向反了。
  3. **信息量近乎为零**:val-选点相对**均匀随机合法 epoch** 只多 **+0.65 个 test 样本**(0.8322 vs 0.8278),
     而 final epoch 多 **+2.7 个**(0.8456),且落在 per-seed test-ORACLE epoch 的 **0.0023(0.34 项)** 内。
  4. **把 dev 信号扩大 3× 反而更差**:pooled-dev argmax(234 个 dev 决策)选中 epoch 19 → 3-seed test
     **0.8210**,在 25 个合法共享 epoch 中排 **19/25**;而 ep29 排 **1/25**(前五名 ep29/27/28/26/25 = 一个宽的
     单调后期平台)⇒ **不是「dev 太小」,是判据在该区间失准**。
  5. **决定性:val-选点读数在换设备后不可复现**——同配方 CPU re-mint 把它移动最多 **−0.0335 acc** 并把 argmax
     从 epoch 20 挪到 epoch 5,而 final-epoch 读数在三个 seed 上**复现 banked test acc 到 4 位小数**
     (ep29 平均曲线差 0.01 项)。**一个输出对浮点归约顺序都不稳定的协议,不是在测量模型。**
- **⚠ rule-shopping 暴露(必须与证据一起披露,不得省略):** 退休恰好把**一个**判决转向我方有利
  (ZH val-sel FAIL → not applicable)。缓解但**不消除**:退休在别处**零代价**——HateMM 双协议皆 PASS(F53)、
  EN 双协议皆 FAIL(F55),**没有第二个判决因此移动**。**可辩护的表述是**「ZH dev 划分过小且与目标反向对齐
  (证据 1–5),不能充当模型选择工具,故单协议 final-epoch 报告是方法学上正确的读法」,**不是**「丢掉
  val-sel 后 ZH 就过线了」。
- **我方推荐:** **退休 val-选点,单协议 final-epoch 报告**,并在方法学附录全文照登证据 1–5 **与本条
  rule-shopping 暴露**;若用户更保守,则维持原 D2 的「两口径并排」。**与 S1 耦合**:退休后 dev 未被选点消耗,
  才是 R1 conformal 校准的合法集合。
- **来源:** `ERRPAT_MHC-ZH_2026-07-26.md` §1.1–§1.7 · `ad56a62`;原始 trainlog
  `slurm/logs/enc3s_MHC_zh_*_13150.trainlog`(ep29 per-seed 0.8456 / 0.8389 / 0.8523,mF1 0.8181 / 0.8113 /
  0.8226,本次重读确认);`PAPER_MASTER_TABLES.md` T1.1 脚注 + T6.5。

## S3 — MNTP S2b 语料裁决:在**我方权重点**自训 MNTP 用什么语料(唯一还活着的文本侧假说)

- **问题:** F92 关掉了 readout 路线,F93 证伪了**零训练移植捷径**却给出**全战役第一个真实 bidir 信号**
  (HateMM text +0.0280 = **+0.6006 crater recovery**,首个越过冻结 50% 门;ZH +0.2941;**两侧同号**)。
  唯一存活形式 = **S2b:在我方权重点自训 MNTP**。它卡在**语料许可**,不卡在证据。
- **三个选项与 veto 分析(逐字承自 `MNTP_FORENSIC_RECON.md` §3,commit `ead9f5d`):**
  - **(a′) 自有 train split,但用部署的多模态格式**(8 帧 + 标题 + 转写 + 指令,**只 mask 文本位置**,视觉
    token 作冻结上下文)——**合法,无需任何裁决**(同数据、同划分、无标签)。优点:**分布对**(82.5% 视觉、
    ~930 token 序列,正是抽取器真实运行的区间,wikitext 永远碰不到);缺点:**预算不变**——HateMM 自有
    转写 **239,382** token = LLM2Vec 参考预算(1000×32×512 = 16,384,000)的 **1.46%**,ZH **52,351** = **0.32%**,
    达到参考步数意味着在几百条转写上跑 ~68(HateMM)/ ~313(ZH)遍 = **背下记忆库自己的文本**,而 bank 就是
    train split ⇒ 对 kNN 投票的污染方向最敏感。
  - **(b) 通用无标注语料(wikitext-103)**——**需要用户放宽 veto**。`banned_constraints` 原文:*"TRAINING DATA
    = single-dataset train split ONLY (user veto 2026-07-14): no cross-dataset split mixing (trivial trick, not
    a contribution); conservatively also bans external unlabeled-pool training (C5)"* ⇒ **保守条款已经够到 (b)**。
    **可供用户权衡的范围细节(recon 的原话,不是我方裁决):** veto 点名的 (C5) 是**外部无标注视频 + MLLM 伪标签**
    的表征训练;**纯文本、零标签、只做架构适配**的语料是另一类对象,veto 的**理由**(「trivial trick」)是否
    够到它**是用户的判断**。优点:**唯一解决预算问题**、唯一有可直接引用的已发表配方;缺点:分布反向缺口
    (纯文本 512 token,永远不进 82.5% 视觉区间)。
  - **(c) 已发表 MNTP 权重移植**——**已花掉**:这就是 F93/S2a,**STOP overdetermined**(collapse belt 触发、
    融合由相加转相消 +0.0467 → −0.0467、每个数低于 causal floor、升级门不可能满足)。veto 分析上 (c) 从来
    不是语料问题(**我方不在任何语料上训练**,它与 base encoder / CLIP 同类),它只需要**下载门**——那道门
    已经开过并已用掉。
- **我方推荐:** **先请用户就 (b) 表态**。若 (b) 获准:按 LLM2Vec 自己的顺序 **(b) → (a′)** 串行(通用语料做
  架构适配,自有多模态划分做分布对齐),这是科学上最强的包;若 (b) 被否:**只跑 (a′)**,并在论文里如实写
  「预算只有参考配方的 1.5% / 0.3%,因此是**分布修正而非预算修正**」;若用户判断两者都不值:**S3 = 关闭
  文本侧**,MNTP 一条线以 F92 + F93 结案(两者都已是可写入论文的机制结果)。**任一分支在裁决前不排 GPU。**
- **来源:** `MNTP_FORENSIC_RECON.md` §3.1–§3.6 · `ead9f5d`;`MNTP_S1_RECORD.md` §6d/§6e · `0663ab7` / `b328dc9`;
  `autoresearch/goal_mllm_plus3/state/directions_tried.json` → `banned_constraints`。

## S4 — 论文框架:是否把故事定为「**可维护的证据记忆**(maintainable evidence memory)」

- **问题:** 现有素材的重心已经不在「更高的 accuracy」上。目前可支撑的四块:**①** 同场决定性胜出的检索-记忆
  检测器(T1/T1.2,数字未变);**②** 一套**机制化的负结果地图**——四条结构律 + round-8 的六方向关闭,
  每条都带机制而不只是 p 值;**③** 记忆的**可维护性**——换库(cross-dataset swap)、时间重校准
  (W4:漂移是**校准漂移不是可分性损失**,ROC 反升 0.8484)、**可审计 + 可外科编辑**(pillar-④,**single-seed
  capability demonstration**,见 F88 更正);**④** span-free 定位(P6 → P10-b 0.5755,modest-plus)。
  **问题是:主线写「detector 涨点」还是写「一个可维护、可审计、可认证的证据记忆系统」?**
- **代价:** 写成 ③ 主线意味着**主动放弃**「substantial 主表增益」这一 claim(它在冻结约束下已被 round 2–8
  反复测死),换取一个**素材齐备、每条都可复现**的系统性贡献;写成 ① 主线则必须解释为什么 EN/ZH 不动,
  而现在这个解释本身(label 语义天花板 + selection-lock 算术)已经比涨点更扎实。
- **我方推荐:** **采用「可维护的证据记忆」主线**,①作为「我们确实是个有竞争力的检测器」的支点而非卖点,
  ②作为方法学贡献单列,③为核心叙事,④为可移除角色;**pillar-④ 的措辞全线固定为**「human-in-the-loop
  capability demonstration, single-seed; not an accuracy claim」(F88 更正已传播到方法章 §5、实验章 §5、
  分析章 §4、intro §1(C4)/§2(e)、T3/T4/T6.5、DEMO/EXP/CAMPAIGN/TERMINUS/OPTION_KITS)。**若 S1 = 是**,
  R1 的三路认证输出正是这条主线的自然终点(记忆 → 证据 → 认证判定),两项应一起裁决。
- **来源:** `PAPER_MASTER_TABLES.md` T1/T3/T6.5;`DRAFT_analysis_chapter.md` §3.6/§3.12/§3.13/§4;
  `DRAFT_experiments_chapter.md` §5/§9;`EVAL_temporal_memory_W4.md`;`ERRPAT_MHC-EN_2026-07-26.md` §6.5 ·
  `ad56a62`;`LITSWEEP6_PARADIGM.md` · `49e15ec`。

### D1–D12 去向对照(存档,不再单独等裁决)

| 原项 | 去向 |
|---|---|
| **D1** MLLM campaign 终局三选项 | **并入 S4**(选项 (c)「换方法族」在 D7 裁定后已 DEAD;(b) 闭源 API 仍只是定位增量,不阻塞) |
| **D2** ZH 主表协议 | **升级为 S2**(证据包由 ERRPAT-ZH §1 的五条 Tier-1 事实大幅加强,推荐随之从「两口径并排」改为「退休 + 全文披露 rule-shopping 暴露」) |
| **D3** EN 近天花板定位 + 摘要措辞 | **并入 S4**;F88 新增机制支撑(EN 残差 = label-semantics 失配,9/22 = **40.9%** 的共识误差是无群体目标的 Offensive)⇒「near-ceiling, label-limited」现在是**测量**而非口径选择 |
| **D4** 三件套 / 四支柱措辞 | **并入 S4**(pillar-④ 措辞已由 F88 固定;三角色框架不变) |
| **D5** 杂项归档 | 存档,运维项,不影响论文 |
| **D6** 投稿 venue + 截稿 | **仍悬置**,但与 S1–S4 正交(素材 venue-agnostic);S4 一旦拍板会影响 venue 偏好(负结果/方法学友好) |
| **D7** encoder-swap 是否计入 novelty | **已裁决(RESOLVED-NEGATIVE,2026-07-14)**;F91(Molmo2)是该裁决下的又一条性能/诊断素材,不改 novelty 边界 |
| **D8 / D9** family-headline / 并比脚注 | 存档:纯性能报告口径,D7 后对 novelty 已 moot |
| **D10** EN-LoRA 形式化闭合跑(~2 min GPU) | 存档:仍是「用户请求项」,不推荐主动跑 |
| **D11** 72B-AWQ scale 点 | 存档:dead-axis grinding,不推荐 |
| **D12** 基建裁决(备份/配额/遗留 flag) | 存档:运维项,按原推荐执行 |

---

# ▽ 存档区(D1–D12,2026-07-09 / 2026-07-14 原文保留)

> **状态(2026-07-28):本节整体为存档**,不再是「当前悬置清单」——当前清单是上面的 **S1–S4**。原文与证据链
> 一字未删(可追溯性),每项去向见上节「D1–D12 去向对照」。**D6(venue)是本节唯一仍独立悬置的项**;
> D7 已裁决;其余或并入 S1–S4,或降为运维/口径存档。

## D1 — MLLM campaign 战略终局(TERMINUS 三选项主裁决)

- **问题:** 13 条预注册路线全结题后,论文的 MLLM 故事走 **(a) 定稿 / (b) 闭源 API 续攻定位(需数据外发批准)/ (c) 换方法族**?
- **选项与代价:** (a) 近零算力、放弃「substantial 主表增益」强 claim;(b) 中成本、须批准把仇恨内容送第三方商业 API 且接受闭源不可复现;(c) 周期以周计、放弃现有四支柱资产。
- **我方推荐:(a) 定稿**(若用户接受数据外发 + 闭源代价,可 **(a)+(b) 并行**,(b) 仅作定位放大的可选增量,不阻塞定稿)。
- **理由:** 13 路线全为诚实 kill / within-noise;开源可行域**三面墙全闭**——重聚合 0.5932 / 规模梯 72B 0.5913 / 代际 Qwen3-VL-32B 0.5866,**均 < 0.616 校准线**→ substantial 定位开源不可达(`TERMINUS §2/§3/§6`,commit `03880f2`/`74f0eac`/`0b3cf40`)。(b) 的两点校准→test 映射(0.5387→0.5435、0.5913→0.5755,transfer 仅 ~60%)诚实外推**闭源翻案期望低**且仅动定位、非主表(`OPTION_KITS Kit-B B.2/B.5`);(c) 上限高但 P9/P9b 已警示决策级 LMM 只匹配现有 LoRA(`EXP_p9_lmm_rgcl_video`,`4d28655`)。
- **影响文档/章节:** 方法章 MLLM 一节整体定位;`TERMINUS §5`;`OPTION_KITS Kit-A/B/C`;`PAPER_MASTER_TABLES` T2.1/T2.2。

## D2 — Headline 协议(ZH 主表口径)

- **问题:** ZH 主表用 **val-选点(0.827,预注册,不过 0.85)** 还是 **final-epoch(0.8537±0.012,过 0.85)**?
- **选项与代价:** 改用 final-epoch 过线好看,但「**因过线才换口径 = rule-shopping**,rebuttal 必死」;且 final-epoch 下 archive 通道 ZH 贡献**恰为零**、EN 为负。
- **我方推荐:两口径并排** —— 主表沿用预注册 val-选点,附录放五规则鲁棒性全表 + 已成文说明段;若改 final-epoch 须以「未来预注册」名义全线统一并自曝时序。
- **理由:** 同一 seed 权重逐位相同、val-选点在 78 样本 dev 上自损 ~2 acc 点(`exp-archive-knn-seeds` Addendum 2,sha1 审计);selection-robustness 段落已成文可直接引用(`MORNING_REPORT §6.1/§7.1`)。
- **影响文档/章节:** `MORNING_REPORT §1` 记分板;主表 T1;方法学附录 selection-robustness 段。

## D3 — EN 近天花板定位 + 摘要措辞

- **问题:** MHC-EN ≈0.79–0.80(所有杠杆穷尽),摘要用「**同场决定性胜出 + 近天花板归因**」如实报数,还是「near-SOTA-at-fraction-of-params」的强措辞?
- **选项与代价:** 强措辞更抢眼但 CRAVE 发表 79.81 F1 在**全量 split**、与我方 clean 子集**不可直接比**,直报 near-SOTA 会被审稿人抓可比性。
- **我方推荐:直报双口径 EN 数 + 同场胜出框架 + 近天花板归因**(不主张绝对 SOTA)。
- **理由:** 同场 MoRE clean 仅 0.69–0.72,我方 +8.7 acc / +22.9 F1(`BASELINE_MoRE_rerun`,`MORNING_REPORT §2`);EN 章主体 = §4③ 归因链 + oracle 复活条件(role-3 门控留 0.857–0.888,`EVAL_role3_selective_reasoning`);0.85 作为该 split 未达公开目标如实报(`HEADTOHEAD_FEASIBILITY §3`,`OPTION_KITS A.4 旧决策②`)。
- **影响文档/章节:** 摘要;`MORNING_REPORT §1/§2`;EN 章 §4③。

## D4 — 三件套 / 四支柱措辞(MLLM 角色定位)

- **问题:** 方法章 MLLM 一节按 campaign 终局的**「三个挣得的角色 + 一条明确的非角色」**重写,并把定位对照改写为 memory 对照?
- **选项与代价:** 不改则停在 campaign 前的视觉-only 记忆键口径(wv-AUC 0.526,仅能力演示),与终局的 P6→P10-b **统计稳固 modest-plus 定位器**(0.5755)不一致。
- **我方推荐:采用 Kit-A A.2 四角色框架** —— encoder(HateMM +4.2 F1 跨 0.85)+ 定位打分器(P6 0.5435 → P10-b 72B A-fuse **0.5755**,对 memory +0.0615 配对显著)+ guard-rail/审计 + **明确非角色**(11+2 路线 ruled-out map);**MORNING_REPORT §3/§4④/§5 三处定位表述同向更新**到 P6→P10-b。
- **理由:** 定位对照写 **memory 而非 MIL**——A-fuse−memory **+0.0996** 显著、A-fuse−MIL n.s.(`TERMINUS P11 §4`,`0b3cf40`);MIL 需目标域视频标签、memory-swap 不需,非同一能力口径。数字与框架见 `TERMINUS §4/§6.2`、`PAPER_MASTER_TABLES`(`d9731e8`)。
- **影响文档/章节:** 方法章 MLLM 节;`MORNING_REPORT §3/§4④/§5`(第 13 撤回行缩范围);第 5 章定位节。

## D5 — 杂项归档(入库 / 保留说明)

- **问题:** ①`disk_guard.log`(现 ~55 万行守护流水)入不入库?②`HateVideoVLM` conda 环境 与 ③`data/gt/HateClipSeg/p11_split.json`(冻结未消费)如何留档?
- **选项与代价:** raw 守护日志入库会污染仓库且非研究产物;环境/split 若无说明,未来会被误删或误当已消费资产。
- **我方推荐:** ①`disk_guard.log` **不入库**(gitignore / 仅留本地)——其 load-bearing 的 sha1 / B2 对账证据已在 `MORNING_REPORT §6.3` 与 `EXP_mm_segment_keys §1` 成文;②`HateVideoVLM` 环境**保留** + 一行溯源(P10-c Qwen3-VL 打分环境,复现 `EXP_p10_loc_amplify` P10-c 节,`74f0eac`);③`p11_split.json` **保持冻结** + 一行说明(为任一前提不同的 HateClipSeg 弱监督训练路线预留,test-touch 从未消费,`TERMINUS P11`,`0b3cf40`)。
- **影响文档/章节:** `.gitignore`;`ITERATION_LOG` 归档节;`EXP_p10_loc_amplify` / `EXP_p11_weaksup_localization` 资产说明。

## D6 — 投稿目标(venue + 截稿)

- **问题:** 目标 venue 与截稿定哪个(候选:WWW / ICWSM / ACL-ARR,尚未定)?
- **选项与代价:** 未定则无法按页数/附录政策裁剪正文-附录分配;素材形态(主表 + 四支柱 + 归因章 + 方法学附录 + 负结果 ruled-out map)本身 **venue-agnostic**,不阻塞其余五项。
- **我方推荐:尽快锁 venue+截稿以解冻 Kit-A A.3 的正文/附录分配**;鉴于方法学章含大量强负结果 + ruled-out map,倾向**欢迎负结果/方法学贡献**的 venue(ICWSM / ACL-ARR 线)。
- **理由:** Kit-A A.3 已给出正文/附录分配草案(同场 MoRE 主表 + 四支柱进正文,11 路线记分板 + 双口径鲁棒性进附录),只待 venue 页数政策定稿(`OPTION_KITS A.3`,`MORNING_REPORT §7.3`)。
- **影响文档/章节:** 全文裁剪;`OPTION_KITS A.3` 表位分配。

---

*(本备忘录随 D1–D6 任一被裁决而逐项失效;裁决后由主会话把结论落地到对应文档,并从本单移除已清项。)*

---

## Round-2 追加裁决项(2026-07-14,B3/B4 之后)

> 本节承接 D1–D6,追加 **round-2 MLLM-integration campaign** 收尾后新增的悬置裁决项:B3(LoRA-Qwen
> 编码器 vs frozen-CLIP on MHC-ZH,job 13150,`final-epoch: PASS (MARGINAL); val-selected: FAIL`)是本轮
> **首个实测(部分)正结果**,B4(EN 侧同一 LoRA 单元)是**第 22 条预注册负结果类条目预-GPU 关闭**。数字与
> 判决语言逐字转录自命名源:`research-wiki/TERMINUS_round2_mllm_plus3.md`(§4/§6/§7)、
> `refine-logs/B3_VERDICT_REVIEW.md`(§4b/§6)、`refine-logs/B4_FORENSIC_RECON.md`、
> `research-wiki/PAPER_MASTER_TABLES.md`(PUR addendum:PUR-1/PUR-2/PUR-banner)。**同样只汇总悬置决策,
> 不含已批准行动;每项「我方推荐」仅是建议。** 编号续 D1–D6。

## D7 — LoRA / RA-HMD-family 编码器杠杆是否计入 goal 的「novel」子句 — ✅ RESOLVED 2026-07-14(RESOLVED-NEGATIVE)

> **裁决(2026-07-14 晚,用户,逐字):**
> - 「哎呀,这个 encoder swap 肯定不算 novelty 啊」
> - 「我不管,反正这个做不出来就一直做,直到做出来为止。」
>
> **编排解读(binding):D7 = RESOLVED-NEGATIVE。** encoder-class 杠杆——frozen swap、LoRA-adapted
> swap,及推而广之的通用决策规则校准(如 B5)——**均不满足 goal 的 novelty 子句**;它们保留为
> **合法的性能 / 消融 / 诊断素材**。TERMINUS 选项 (c)「goal 重议」= **DEAD**;goal 现要求一个
> **NOVEL MECHANISM**(novelty 在 hateful-video 检测范围内判定)× MLLM-integrated × 交付 **≥+3 acc**。
> 据此:D8 的 family-headline 对 novelty 子句已 **moot**(仅剩性能报告口径问题);D9 不变(仍是性能
> 报告口径问题)。以下原始悬置内容留档存证。

- **问题:** B3 把 LoRA-Qwen 编码器 vs frozen-CLIP 在 MHC-ZH 定为 `final-epoch: PASS (MARGINAL)`(mean
  Δacc **+0.0313**)——这是全项目**最接近** goal「+3 acc AND +3 F1」的实测配对结果(其余 21 条搜索轴全为
  负)。但 LoRA / RA-HMD-family 一直被本项目分类为**「MIXED performance lever, not novelty」**
  (`query_pack.md:44`;`B1_PREREG_REVIEW.md:64`)。**一个 LoRA-encoder 的性能 pass 是否计入 goal 的
  「novel」子句?**
- **我方推荐:不主张 novelty。** 把 B3 作为 encoder-adaptation 的**正式消融/性能行**入表,novelty 叙事仍
  挂四支柱(retrieval-contrastive+kNN 核 + 可更新记忆 + 共识去噪 + 可审计档案);LoRA 只提供 ZH 上表征级
  增益的证据。若用户愿按 `TERMINUS §6(b)` 重划边界(把 LoRA 正式化为 encoder 适配消融而非 novelty 主张),
  可据此升级表述——**须用户先划 novelty 边界才可动。**
- **来源:** `TERMINUS §6`(选项 b)· `B3_VERDICT_REVIEW.md §6`(Novelty bullet,显式 PENDING)·
  `PAPER_MASTER_TABLES.md` PUR-banner (i)·性能数 = PUR-1。

## D8 — 「MLLM-encoder family」能否作 ≥2-数据集 headline

- **问题:** 两个编码器级 pass 骑在**不同机制**上——HateMM 是 **frozen-Qwen** swap **双协议 PASS**,ZH 是
  **LoRA-Qwen** 微调 **final-epoch marginal PASS**。可否以「MLLM-encoder family」作为跨 ≥2 数据集的
  headline?**若 goal 要求单一机制跨 ≥2 库过线,则 frozen(仅 HateMM)与 LoRA(仅 ZH)均不单独满足。**
- **我方推荐:不打「单一机制双库」headline。** 如实报「MLLM-encoder family」= 两条不同杠杆各在一库,并在
  正文明确二者机制不同(frozen-swap vs LoRA-adaptation);headline 主张仍走 HateMM 单库最强正效应
  (+5.3–5.6 acc 双协议 3/3)+ 同场 MoRE 胜出。
- **来源:** `B3_VERDICT_REVIEW.md §4b`(No headline upgrade)+ §6(第二 bullet)· `TERMINUS §6`·
  `PAPER_MASTER_TABLES.md` PUR-banner (ii)。
- **D7 后果注(2026-07-14):** D7 裁定 encoder-class 杠杆不入 novelty ⇒「MLLM-encoder family」headline 对
  **novelty 子句已 moot**;本项降为**纯性能报告口径问题**(如何如实报两条不同机制各在一库),不再承载
  goal 达成主张。

## D9 — `PAPER_MASTER_TABLES.md:58`「不可直接同格并比」注是否被 B3 配对覆盖

- **问题:** 主表脚注(`PAPER_MASTER_TABLES.md:58`)记「LoRA-Qwen 主栈与 frozen-CLIP floor 不同编码器,
  **不可直接同格并比**」。B3 用**同 runner、同 `--seed`、同 149 ZH test videos** 的 head-level 配对
  (job 13150 vs 13115),是现存最干净的配对读数并已记于 PUR-1。**这一同 runner 同种子配对是否覆盖 :58
  记账注,以支撑一个论文主张?**
- **我方推荐:覆盖仅限「配对 Δ」这一受控读数**(可在附录以 B3 同 runner 配对表 + 三条敏感度事实呈现),
  **不**升级为主表 headline 并比;主表 T1.1 保留 :58 注。理由:B3 仍是**单一 CLIP 抽样 + 单一 LoRA 编码器
  抽样**(3 种子共享单缓存,只变下游 head),不建立训练种子方差。
- **来源:** `PAPER_MASTER_TABLES.md:58` + PUR-1/PUR-banner (iii)· `B3_VERDICT_REVIEW.md §6`(第三 bullet)·
  `TERMINUS §6`。
- **D7 后果注(2026-07-14):** 不变——本项始终是**性能报告口径问题**(附录 B3 同 runner 配对 Δ 覆盖
  vs 主表并比),与 novelty 子句无关;D7 裁决不改变本项。

## D10 — 可选:EN-LoRA 正式闭合跑(约 2 分钟 GPU,veto-clean)

- **问题:** B4 取证证明 EN 侧同一 LoRA 单元**并非未测**:同 adapter + 同特征缓存 + 同 RGCL+kNN head 已在
  **seed0 双协议**入账为负结果(val-sel **−0.0310 acc** / final-ep +0.0062 acc,`exp-lora-sft-encoder.md:21`)。
  因 adapter 与缓存均在盘,把 seed0 锚定负结果**升级为正式 3-种子配对闭合行仅需约 2 分钟 GPU**,且清过全部
  三条现行 veto(单数据集自有 train split / 无 OCR / 无 gold aux)。**跑不跑?**
- **我方推荐:现在不跑**(遵「不在已关闭轴上烧 GPU」)。它只会把**已知负结果**形式化为论文的一行正式闭合,
  不开新地;诚实先验 = **双协议 FAIL,证伪概率 <5%**。留作**用户请求项**——若用户要一行形式化的 3-种子闭合行
  以补全 LoRA-encoder 三数据集地图(PUR-2),可按 `enc3seed.sbatch` 加三行运行。
- **来源:** `B4_FORENSIC_RECON.md §(iii)/(v)`(成本表 + 诚实先验)· `TERMINUS §7`(用户选项,veto-clean)·
  数字 = PUR-2 MHC-EN 行。

## D11 — 72B-AWQ scale 点(为完整性列出)

- **问题:** 72B-AWQ 编码器是唯一未跑的 scale 点。是否值得跑?
- **我方推荐:不建议(dead-axis grinding)。** B2 已实测 32B 在 HateMM 锚数据集上**介于 CLIP 与 7B 之间**
  (scale **单调退步**,CLIP<32B<7B),对 72B 的先验 ≈0;真要跑需抽取脚本加 AWQ 路径 + autoawq 安装 +
  delta-check + 41G 下载。除非用户另有机制假设,否则不动。
- **来源:** `TERMINUS §4(a)` · scale 单调退步实测 = `B2_VERDICT_REVIEW.md`(第 21 条负结果,`TERMINUS §1`)。

## D12 — 基建裁决(备份 / 配额 / 遗留 flag)

- **问题:** 四项运维待裁:① main 上 **125 个未推送 commit**(实核 `git rev-list --count origin/main..HEAD`
  = 125,备份风险);② **disk_guard quota 解析 bug** —— 当前**设计上全盲**(不自动裁剪),修复可能在**超阈值时
  重新启用自动 pruning**,风险不对称,须用户明确放行才动;③ **lora_p9 83G + Retrieval 41G 未备份**未裁决;
  ④ A 线 **M-A/realbank `is_science` 遗留 flag**。
- **我方推荐:** ① 尽快 push 或另做冷备(纯文档 commit,无 GPU);② disk_guard 修复**须用户明确 go**(在此
  之前维持全盲,避免超阈值自动删);③④ 留待用户逐项裁决,不擅动。
- **来源:** `TERMINUS §4(d)`。

---

*(D7–D12 与 D1–D6 同规:任一被裁决即逐项失效,裁决后由主会话落地并从本单移除已清项。)*
