# F3_DIRECTION_PACKAGE — 立场辅助训练:完整方案包

**日期** 2026-08-17 · **类型** 设计 + 分析,零 API 成本,零训练,零 GPU,零 test 接触
**本文档产生的唯一计算**:从 `idea-stage/r8_blr/results.json` 重算逐数据集 seed 方差(CPU,秒级),
从 `data/gt/*/*.jsonl` 和 `data/_src_Multihateclip/*/annotation(new).json` 数样本数,
用 ffprobe 抽样 120 个视频量时长。没有读任何 test 预测,没有训练任何模型。

**这是方案包,不是预注册。** 用户点头后才写冻结文件。

---

## 0. 一页决策摘要

### 0.1 要买的东西

给 **MHC-EN + MHC-ZH 的 train+val 共 1286 个视频**,每个补一个**人工立场标签**
(说话人是在**认同**还是**切割/引用/批判**其视频里出现的攻击性内容),
然后把它当**辅助任务**加进现有 head 的损失里(不改输入、不改推理、推理时丢弃立场头)。

**不标 HateMM,不标 ImpliHateVid。** 理由在 §0.3。

### 0.2 三种执行方式对照

标注量固定为 1286 项 × 2 个独立标注者 + 分歧仲裁 ≈ **2892 条判断 ≈ 83 标注工时**
(视频中位数 33–35 秒、上限 60 秒;按 1.5 分钟/条 + 15% 培训与质检开销估)。

| | **A 用户自标** | **B 雇学生 RA(推荐)** | **C 众包平台(Prolific)** |
|---|---|---|---|
| 现金支出 | **NZ$0** | **≈ NZ$2,800(US$1,690)** | **≈ US$2,000** |
| 用户自己的时间 | **≈ 37 小时**(单人一遍) + 仲裁 | ≈ 8 小时(写指南、培训、仲裁) | ≈ 12 小时(写指南、审附件、处理退件) |
| 日历时间 | 3–4 周(每天 2 小时) | 3–4 周(含伦理审批 + 培训) | 2–3 周,**但中文半边大概率招不满** |
| 标注质量 | 单人 → **κ 无法计算,质检门结构性失效**;需另找一人标 250 项子集补 κ(约 US$120) | 最高。κ 可算,仲裁由用户做 | 最低。注意力检查退件率按 20% 计已含在报价里 |
| 中文半边可行性 | 取决于用户中文阅读能力 | 招一个中文母语 RA,可行 | **主要风险点**:Prolific 中文母语池小,B 站内容熟悉度无法筛选 |
| 期望增益(见 §3) | 与 B/C 同,但单人标签更噪,转化率打折 | MHC-EN **+1.5 ~ +3.1 分**、MHC-ZH **+1.5 ~ +3.0 分** macro-F1 | 同 B,但标签质量下降会进一步压低 |

价目来源与需要复核的地方:NZ$30/hr 是 UoA casual RA 的估计档(需向系里核实),+13% on-cost;
Prolific 公开的参与者最低 £9.00/hr、建议 £12.00/hr、平台服务费 33%(按 £12/hr 计,汇率 £1≈US$1.27);
MTurk 未列入表中——它的佣金是 20%(单 HIT ≥10 份时 40%),名义时薪可以压低,
但中文母语池实际上不存在,且质量先验明显更差,在本任务上没有价格优势。
**这三行数字都是牌价,承诺前必须自己复核。**

### 0.3 为什么砍掉两个数据集

- **ImpliHateVid:原始视频不在这台机器上**(`data/video/ImpliHateVid/` 只剩一个
  `_id2b2path.tsv`,没有任何视频文件;`IDEA_REPORT` §9.10 第 5 条已记录"ImpliHateVid's raw
  video is gone")。人工标注要求看完整视频,这个数据集**物理上标不了**。与功效无关,是数据可得性。
- **HateMM:标了大概率白标。** 可回收 oracle 只有 **+1.23 分**;判定线是 +0.005(=0.5 分),
  即需要 **41% 以上的转化率**才刚好过线;而 HateMM 视频中位数 76 秒、均值 106 秒,
  标注成本 **≈ 110 工时**,比两个 MHC 加起来还贵。**最贵的数据集买最小的奖品。**

### 0.4 建议的执行顺序(先花 US$250 买一个杀开关)

**先做 200 项试点(100 EN + 100 ZH),约 13 工时,B 方式约 US$230。** 试点后有四道门,
任何一道不过就停,不进全量:

| 门 | 量什么 | 过线 | 不过意味着 |
|---|---|---|---|
| G1 | 两个标注者的 Cohen κ(在"有攻击性内容"的项上) | ≥ 0.60 | 人类之间就标不一致,标签不能当监督 |
| G2 | "是否含攻击性内容"这一步 vs 数据集金标的 κ | ≤ 0.75 | 第一步就等于把金标重新判了一遍(循环),要重设计 |
| G3 | 立场标签能否从冻结特征 `concat(img,txt)` 解码(5 折 logistic AUC) | **0.58 ≤ AUC ≤ 0.85** | 低于 0.58:冻结特征根本承载不了立场,head 级路线死;高于 0.85:与二值标签冗余,重蹈 P4 |
| G4 | 试点里"含攻击性内容"的比例 | ≥ 25% | 燃料太少 |

**G3 是整个方案里最重要的一道门**,理由见 §7.3。

### 0.5 一句话

**MHC-EN 和 MHC-ZH 值得标,期望 +1.5 到 +3.1 分,新测量协议在 30 seed 下完全测得出;
HateMM 标了也测不出(需要 41% 转化率,而它的标注成本是两个 MHC 之和);
ImpliHateVid 没有视频,标不了。总价 NZ$2,800 / 83 工时,但先花 US$230 过四道门。**

---

## 1. 这个方向为什么还活着

### 1.1 奖品(全部采信 `idea-stage/S_PRIZE_DECOMP.md`)

S 桶(立场 / use-vs-mention)49 项,4 票 MLLM 面板判定其中 26 项可回收、21 项争议、2 项分裂。
只修可回收项的 oracle:

| 数据集 | base | 全部 S 项 oracle | **可回收 oracle** |
|---|---|---|---|
| HateMM | 0.8732 | +3.47 | **+1.23** |
| MHC-EN | 0.7776 | +10.48 | **+5.10** |
| MHC-ZH | 0.8183 | +8.90 | **+5.01** |
| ImpliHateVid | 0.9276 | +3.00 | **+1.58** |
| 均值 | — | +6.46 | **+3.23** |

**+3.23 是保守的下界,不是上界。** 面板 4 票全是 MLLM(3 个 Claude + 1 个 qwen),彼此 Fleiss κ=0.90,
输入只有 8 帧 + 转录;`S_PRIZE_DECOMP` §7.2 明确写"人类在这些项上确实拿得到面板拿不到的东西",
§4.3/4.4 两例里人类三票与面板完全反向且人类内部一致。争议 21 项上人类分裂率 0.300,
反而**低于**可回收项的 0.438——争议项不是"人类也说不清"的那批。
所以人类标注(看完整视频)的可回收上限落在 **+3.23 到 +6.46 之间**;本文档一律用 **+3.23** 规划。

### 1.2 立场标签只能人来标(三条独立证据)

1. **`CLAUDE_STANCE_GATE_RESULT.md`**:3 个盲标 Claude Opus 5 标注者(转录 + 8 帧),
   在 32 个立场型错误项上多数票 **0.5625(18/32)**,与 qwen 最好一轮**逐位相同**,
   也与"一律答 DISTANCED"的常数预测器相同。分层后更清楚:检测器答对的 50 项上 Claude 是 **0.820**,
   检测器答错的 49 项上是 **0.551**,而同一批行上常数基线是 **0.612**——
   **在唯一需要它的地方,比乱猜差 6.1 个点。** Fleiss κ=0.90 且一致项准确率 0.519,
   这是自洽性方法必然失效的形状,换标注者 / 加投票都救不了。
2. **`SYNTH_PAIR_PROBE_RESULT.md`**:用规则合成 2600 个最小对训练的归属分类器,
   合成集上 0.981–1.000,真实 ASR 转录上 **AUC 0.441 / 0.467**,6/6 个格子符号反转。
   机制清楚:99 条评测转录里**只有 10 条包含任何归属标记词**,即 **89.9% 的 ASR 转录里
   词汇归属线索根本不存在**。
3. **`PERCEPT_STANCE`**:"感知比判断干净"这个前提在 gate 0 就失败(S_FP 上 2/18 = 0.111,门槛 0.30)。

**结论**:立场标签的唯一来源是人,而且人必须看**完整视频**(面板输入只有 8 帧+转录,
`S_PRIZE_DECOMP` §7.3 明确把这列为面板的天然劣势)。

### 1.3 训练侧辅助任务是唯一没死的接法

四个已死接入点(逐条区别见 §5)覆盖了输入侧、文本侧、决策侧。
**改损失、不改输入、不改推理路径**这一格没有被任何一次实测覆盖。
F75 / F82 的范围核对见 §5.5,结论:**不冲突,但各留一条需要在预注册里正面处理的负担。**

---

## 2. 标注协议

### 2.1 标注对象与数量

**范围:MHC-EN + MHC-ZH 的 train + val,共 1286 项。**

| 数据集 | train | val | 合计 | 三类分布(train+val) |
|---|---|---|---|---|
| MHC-EN | 549 | 80 | **629** | Hateful 52 / Offensive 141 / Normal 436 |
| MHC-ZH | 579 | 78 | **657** | Hateful 77 / Offensive 131 / Normal 449 |

**取舍说明:**

- **为什么不只标金标为正的项(193+208=401 项,能省 69%)。**
  因为 **49 个 S 项里 30 个是 S_FP**(检测器误报,金标 = 非仇恨),
  也就是**目标的 61% 落在金标阴性的项上**。按金标筛选会丢掉大多数要修的错误。这条是决定性的。
- **为什么不用 MLLM 先筛一遍"有没有攻击性内容"再只标筛出来的。**
  `S_PRIZE_DECOMP` §5.4 实测:三个 Claude 标注者判 voice=`none`(即"没看出有攻击性内容")的比例,
  在**可回收项上高达 14/26 = 0.538**。MLLM 的筛选会**优先丢掉我们最想要的那批项**。
  `CLAUDE_STANCE_GATE` §5 同向:37 个可判项里 18 项(48.6%)说视频里根本没有身份指向的攻击内容。
  **不设 MLLM 前筛。**
- **train-only 还是 train+val。** 部署协议(`r4_harness`)只在 train 上训练、在 val 上选 epoch,
  所以严格说只需要 train(549+579=1128 项)。多标 158 项 val(+14% 成本)换来
  5 折 train+val 变体(`r8_decomp` 用的那种)保持可用。**建议标 train+val。**
- **不做跨数据集混标。** `banned_constraints[8]`(用户 2026-07-14 veto):
  训练数据只能是单数据集自己的 train split。EN 的立场标签不进 ZH 的训练,反之亦然。

### 2.2 标注者看什么

**完整视频(带声音)+ ASR 转录 + 标题。** 不是 8 帧。这是本协议相对于已死的 MLLM 标注路线
唯一实质性的输入优势,也是 §1.1 里"人类上限可能高于 +3.23"的全部依据。

视频长度实测(每数据集随机抽 40 个 ffprobe):

| 数据集 | 中位 | 均值 | p90 | 最长 |
|---|---|---|---|---|
| MHC-EN | 35 s | 35 s | 59 s | 60 s |
| MHC-ZH | 32 s | 33 s | 53 s | 59 s |
| (HateMM,供对比) | 76 s | 106 s | 219 s | 449 s |

MHC 两边都被硬性截断在 60 秒以内,这是每条 1.5 分钟工时估计的基础。

### 2.3 标注问题(两步,措辞固定)

**第一步——是否存在可被判为攻击的材料(与"这个视频是不是仇恨"是不同的问题)。**

> Q1. 这段视频里,是否出现了**针对某个身份群体**(种族、族裔、宗教、性别、性取向、性别认同、
> 国籍、残障等)的**攻击、贬低、羞辱、刻板化或侮辱性称呼**的材料?
> 只问材料**在不在**,不问是谁说的、也不问说话人赞不赞成。
> - `YES` / `NO`
> - 若 YES:该材料出现在 → □ 语音 □ 画面 □ 屏幕文字 □ 标题 (可多选)

**第二步——立场(只在 Q1=YES 时问)。**

> Q2. 视频里**主导的说话方/发布方**,对这段材料采取的是什么立场?
> - `ENDORSE`——认同、传播、当作自己的主张说出来,或不加任何批判地呈现
> - `DISTANCED`——引用、报道、批判、反驳、嘲讽该材料,或明确与之切割
> - `UNCERTAIN`——看完整段仍无法判断(**必须在 Q3 写一句理由**)
>
> Q3. 一句话依据(引用视频里的具体一句话、一个画面或一段文字)。必填。

**措辞设计的三条约束:**

1. **复刻各数据集原口径,而不是给一个"更正确"的立场判断。**
   `S_PRIZE_DECOMP` §6 第 4 条把这条定为"决定性变量"。具体做法:
   标注指南的开头**逐字附上 MultiHateClip 论文对 Hateful / Offensive / Normal 的定义**,
   并明确写"Q1 里的'攻击性材料'按这份定义判定;你不需要同意这份定义,
   你的任务是按它执行"。§4.5 里 ImpliHateVid 的物化/贬低口径差异就是这条要预防的形态。
2. **二值 + 不确定选项**,不做五分类。旧预注册(`IDEA_REPORT` §9.9)的五分类
   (endorses/condemns/reports/quotes-mentions/depicts-without-comment)在 749–1608 项的
   训练集上会把每类摊到几十项,且 `CLAUDE_STANCE_GATE` 已证明二值这一层就已经很难。
   **五分类的信息在 Q1 的模态多选和 Q3 的依据句里保留,但不进损失。**
3. **Q3 依据句必填。** 它是仲裁的输入,也是事后核查"标注者是不是在按 Q1 的定义执行"的唯一材料。
   它**不进模型**(进模型就变成 `2604.24179` 已发表失败的自由文本推理路线)。

### 2.4 每项几人标、仲裁规则、质检门

- **每项 2 个独立标注者**,互相看不到对方答案,项目顺序随机化。
- **ZH 半边必须中文母语标注者**;EN 半边英语母语或近母语。
- **仲裁**:Q1 或 Q2 不一致的项,由第三人(建议:用户本人)看完整视频后裁定,裁定即最终标签。
  仲裁量按 **25%** 估(1286 × 0.25 ≈ 320 项)。这个 25% 取自 MHC 自身人类标注的分歧率区间:
  三类标签上非一致率 EN **21.3%** / ZH **29.9%**,跨二值边界分裂率 EN **12.3%** / ZH **16.2%**
  (`IDEA_REPORT:62-66`,实测 2001 个视频)。立场是二值问题但比"是否仇恨"更难,
  取两个区间之间的 25% 作为规划值。
- **UNCERTAIN 的处理**:两人都答 UNCERTAIN → 该项 Q2 标为 `NA`,**在辅助损失里被 mask 掉**,
  不参与训练,也不当作 ENDORSE 或 DISTANCED。一人 UNCERTAIN 一人明确 → 进仲裁。
- **质检门(试点后计算,见 §0.4)**:
  - **G1**:两标注者在 Q1=YES 子集上的 Q2 Cohen κ **≥ 0.60**。低于则 HALT。
    (取 0.60 而不是旧预注册的 Krippendorff α ≥ 0.80:旧门槛是为五分类机器标签的审计设的;
    二值人类立场判断在隐性仇恨上,0.80 不现实。这是一次**明示的放宽**,理由写在这里。)
  - **G2**:Q1 与数据集二值金标的 Cohen κ **≤ 0.75**。这是**循环性检查**——
    如果"有没有攻击性材料"这一步实际上就等于把金标重新判了一遍,那么整个辅助信号
    与二值标签冗余,是 P4 的失败形态。**这道门是用户两次抓到的"证据认定步 = 原判断本身"
    这个缺陷的可测量版本。**
  - **G4**:Q1=YES 的比例 **≥ 25%**。
- **金标不外泄**:标注者不得看到数据集的 Hateful/Offensive/Normal 标签,也不得看到检测器的预测。
  ID 按 `CLAUDE_STANCE_GATE` 的 `manifest.json` 做法匿名化。

### 2.5 工时与成本(逐项算式)

判断条数 = 1286 × 2 + 320(仲裁) = **2892 条**。
每条 1.5 分钟(视频 ≤60 秒,看一遍 35 秒 + 填表与必要回看 55 秒),
加 15% 培训 / 校准 / 质检开销 → **2892 × 1.5 × 1.15 = 4989 分钟 = 83.2 标注工时**。

| 方式 | 算式 | 现金 | 用户时间 |
|---|---|---|---|
| A 自标 | 用户做全部 1286 项一遍 = 1286×1.5×1.15 = 37.0 h;+ 250 项 κ 子集外雇 6.3 h | NZ$0(+ 约 US$120 补 κ) | **37 h** |
| B 学生 RA | 83.2 h × NZ$30/h × 1.13 on-cost | **NZ$2,821 ≈ US$1,693** | 8 h(指南+培训+仲裁) |
| C Prolific | 83.2 h × £12/h × 1.33 平台费 × 1.20 退件缓冲 | **£1,593 ≈ US$2,023** | 12 h |

**试点(200 项)**:450 条判断 × 1.5 × 1.15 = 776 分钟 = **12.9 工时**。
A:5.8 h 用户时间;B:12.9×30×1.13 = **NZ$439 ≈ US$263**;C:12.9×12×1.33×1.20 = **£247 ≈ US$314**。

---

## 3. 功效分析

### 3.1 仪器精度(本文档唯一的新计算)

新测量协议(`R6_CONFIRM_FREEZE_2026-08-17.md`):P1 = 按 val macro-F1 选 epoch(≥warmup 5,并列取最早),
test macro-F1 @0.5;逐 seed 配对;20000 次配对 bootstrap 95% CI;
判定线 **mean ≥ +0.005 且 CI 排零**。

R6 的方差审计只覆盖 HateMM 和 MHC_zh(只有这两个有 `ro_` cache)。
四个数据集齐全的最近一次同协议实测是 **R8-BLR**(`idea-stage/r8_blr/results.json`,
4 数据集 × 5 arm × 30 seeds = 600 次 head 训练,P1,test macro-F1)。
从它的原始逐 seed 数据重算 **(每个非 A0 arm − A0) 的配对差 seed 标准差**:

| 数据集 | 4 个 arm-vs-A0 对比的配对差 std | 取值范围 |
|---|---|---|
| HateMM | **0.01189** | 0.01098 – 0.01281 |
| MHC-EN | **0.01493** | 0.01464 – 0.01522 |
| MHC-ZH | **0.01414** | 0.01352 – 0.01511 |
| ImpliHateVid | **0.00444** | 0.00369 – 0.00503 |

(交叉核对:R6 confirm 60 seeds 上 HateMM P1 CAT−A0 的 std 是 0.01086、MHC_zh 是 0.02566。
HateMM 两者一致;MHC_zh 的 R6 数偏高,因为 CAT 这个 arm 自身方差就大(arm std 0.02109 vs A0 的 0.01406)。
**下面一律用 R8-BLR 的数,它是四数据集齐全且 arm 更接近本方案形态的那一份。**)

**最小可检测效应(MDE)**,配对 t、双侧 α=0.05:

| 数据集 | 30 seed,80% power | 30 seed,50% power | 60 seed,80% power | 60 seed,50% power |
|---|---|---|---|---|
| HateMM | 0.0061 | 0.0043 | 0.0043 | 0.0030 |
| MHC-EN | **0.0076** | 0.0053 | **0.0054** | 0.0038 |
| MHC-ZH | **0.0072** | 0.0051 | **0.0051** | 0.0036 |
| ImpliHateVid | 0.0023 | 0.0016 | 0.0016 | 0.0011 |

head 训练 9–10 秒一次,所以 **60 seed × 4 arm × 2 数据集 = 480 次 ≈ 80 分钟**,建议直接用 60。

### 3.2 期望增益 vs 判定线

期望增益 = 可回收 oracle × 转化率。转化率假设区间 30–60%(任务给定)。

| 数据集 | 可回收 oracle | 30% | 45% | 60% | 60 seed MDE(80%) | 判定线 |
|---|---|---|---|---|---|---|
| **MHC-EN** | +0.0510 | **+0.0153** | **+0.0230** | **+0.0306** | 0.0054 | 0.005 |
| **MHC-ZH** | +0.0501 | **+0.0150** | **+0.0225** | **+0.0301** | 0.0051 | 0.005 |
| HateMM | +0.0123 | +0.0037 | +0.0055 | +0.0074 | 0.0043 | 0.005 |
| ImpliHateVid | +0.0158 | +0.0047 | +0.0071 | +0.0095 | 0.0016 | 0.005 |

**每个数据集需要的最小转化率**(取 判定线 与 MDE 的较大者 ÷ oracle):

| 数据集 | 30 seed | 60 seed | 读法 |
|---|---|---|---|
| **MHC-EN** | **14.9%** | **10.6%** | 假设区间下限的一半就够。**值得标。** |
| **MHC-ZH** | **14.4%** | **10.2%** | 同上。**值得标。** |
| HateMM | 49.6% | **40.7%** | 要求转化率处于假设区间的**上三分之一**。加 seed 帮不上忙:60 seed 之后卡住的是 **+0.005 判定线本身**,不是仪器。**标了大概率白标。** |
| ImpliHateVid | 31.6% | **31.6%** | 仪器精度极高(std 只有别人的 1/3),卡住的完全是判定线。但**没有原始视频,标不了**。 |

**逐数据集结论:**

1. **MHC-EN 和 MHC-ZH:值得标。** 期望增益是 MDE 的 2–6 倍。在 30% 转化率、30 seed 下,
   检验统计量 z = 0.0153×√30/0.01493 = **5.6**,power ≈ 1.0。这两个数据集上
   **仪器完全不是瓶颈,唯一的不确定性是转化率**。
2. **HateMM:如实说——标了也可能白标。** oracle 只有 1.23 分,判定线要吃掉 0.5 分,
   需要 41% 以上转化率才刚过线,而它的标注成本(110 工时)比两个 MHC 加起来还高。
   **不建议标。** 如果最终要在论文里报 HateMM,应当以"MHC-EN/ZH 上的方法,
   在 HateMM 上未做立场标注、按现状报"的形式呈现,而不是花钱标一个测不出来的效应。
3. **ImpliHateVid:标不了。** `data/video/ImpliHateVid/` 里没有视频文件。
   即使有,31.6% 的转化率要求也把它放在与 HateMM 相近的边缘位置。

### 3.3 转化率这个假设本身有多可靠

**支持 30–60% 的证据:**
- 文献里诚实的同类效应密集落在 **+1 到 +3 分**(`STANCE_LIT_RECON` §4.2 汇总:
  `2310.19750` +1.4–2.4 F1;`2206.06423` 立场中间任务预训练 +3 weighted F1;
  `2511.07405` +3 F1;RAMF 整条 typed-record 流水线 +3 macro-F1;ARG 整套 +1.3–1.7)。
  MHC-EN 的 oracle 是 +5.1,+1.5 到 +3.1 对应 **29%–60%**,与假设区间**几乎完全重合**。
- `2206.06423` 还是一个**特异性**结果:在 {更多同域仇恨数据、情感、反讽、立场} 四个中间任务里
  **只有立场有效**,情感 −8、反讽 −7(counter-hate F1 直接掉到 0.00 和 0.08)。
  这直接说明增益来自立场本身,不是"多加一个任务"的正则效应。

**反对的证据(必须并列):**
- **P4(`MLLM_FRONT_RECON` §2)是本项目自己做过的、结构最接近的一次,转化率是 0。**
  辅助线性头预测 MLLM archive 字段(explicitness / modality / mechanism / target_group),
  λ=0.1,eval 时丢弃头。probe **PASS**(字段可解码 AUC .62–.93,字段→标签 AUC .74–.78),
  训练结果**在噪声内**。诊断原文:*"the fields are redundant with the hate label the head is
  already supervised on"*。
- **AGGNET(`TARGET_FINDINGS` gate0 reopen)**:oracle +0.1492/+0.1520/+0.2186,
  96–100% 的部署错误落在它的函数类内,实际交付 **+0.0134/−0.0069/+0.0000**,
  转化率 9% / −5% / 0%。该记录的结论原文:大 oracle 不再是候选的证据,
  它是每个失败候选都已经满足的前提。

**如何调和。** AGGNET 的 oracle 是**重排序 oracle**——用同一批已有信息换个用法;
立场 oracle 是**新信息 oracle**——从外部引进模型手上没有的人类判断。两者不同类。
P4 是更近的类比,而 P4 的失败诊断是**冗余**——它的字段能被冻结特征解码出来,
所以辅助梯度没带进新东西。**§0.4 的 G3 门就是专门用来区分"P4 那种冗余"和"真的带进新信息"的。**
这是本方案里唯一能在花全款之前把这个风险测掉的地方。

**悲观情形必须写明:如果转化率落在本项目自己的历史区间(0–10%),
MHC-EN / MHC-ZH 也不会过线,这笔钱全部沉没。** 30–60% 是文献先验,不是本项目的实测先验。

---

## 4. 集成设计

### 4.1 结构

现有 head(`idea-stage/r4_harness.py::Head`,即部署的 `classifier_hateClipper`):

```
img ──Linear(d,1024)──Dropout(0.2)──l2norm ─┐
                                            ├─ ⊙(逐元素乘) ── Dropout(0.4) ── [Linear(1024,1024)+ReLU+Dropout(0.1)] ×3 ── h ── out: Linear(1024,1)
txt ──Linear(d,1024)──Dropout(0.2)──l2norm ─┘
```

**改动只有一行**:

```python
self.aux = nn.Linear(proj_dim, 1)     # 1025 个参数,占现有 ~4.99M 的 0.02%
# forward 里:return self.out(h), self.aux(h)
```

- **共享的是什么**:`img_proj`、`txt_proj`、3 层 MLP —— 全部可训练参数。
  辅助梯度塑造的正是主任务用的那个表示。
- **不共享的是什么**:`out` 与 `aux` 是两个独立的线性读出。
- **推理时**:`aux` 不被调用。`out` 的计算图**逐位不变**,推理开销**精确为零**,
  不需要任何 MLLM 调用、不需要立场标签、不需要 OCR、不需要额外前向。
- **参数量对比**:DESC_CHANNEL 加了 **+1.84M(+36.8%)** 并输了 −0.0371;
  本方案加 **+1025(+0.02%)** 且这 1025 个参数在推理时不参与。
  `MLLM_FRONT_RECON` 的 **C2 约束("不要给 fusion MLP 加第四个 768 维流")按构造被满足**。

### 4.2 损失

```
L = BCE(out, y) + λ · mean_{i : m_i=1} BCE(aux_i, s_i)
```

- `s_i ∈ {0,1}`:1 = ENDORSE,0 = DISTANCED(仲裁后的最终人工标签)。
- `m_i ∈ {0,1}`:mask。`m_i=1` 当且仅当 Q1=YES 且 Q2 ≠ NA。
  Q1=NO 的项和两人都 UNCERTAIN 的项**不进辅助损失**(不是标 0,是不参与)。
- 辅助项按 mask 内样本数取均值,不按 batch size,避免 mask 比例波动改变有效 λ。
- **λ 的选择**:主协议 = 每个 (数据集, seed) 在 **λ ∈ {0.1, 0.3, 0.5, 1.0}** 里
  按 **val macro-F1(主任务)** 选,与选 epoch 用同一个 val、同一个准则,**不碰 test**。
  副协议 = λ 固定 0.5(继承 `IDEA_REPORT` §9.9 的冻结值),两个协议都报,**主协议定判决**。

### 4.3 四个 arm(预注册草案)

| arm | 辅助目标 | 隔离掉什么 |
|---|---|---|
| **A0** | 无辅助头 | 基线 |
| **AUX** | 人工立场标签 `s_i`,mask `m_i` | 候选 |
| **AUXPERM** | 把 `s_i` 在 mask 内随机打乱(保持边际分布与 mask 完全一致) | "多一个头 + 多一项损失"本身的正则效应 |
| **AUXDUP** | 把二值仇恨标签 `y_i` 当辅助目标,用同一个 mask `m_i` | **立场是否携带二值标签之外的信息** |

**AUXDUP 是用来正面回答 `TARGET_LOOP.md:67`(P4 反重复规则)的**:
*"New supervision must carry information not recoverable from the binary label and prove
conditional information gain."* AUXDUP 就是"完全可从二值标签恢复的辅助监督"的实现,
AUX − AUXDUP 就是条件信息增益的直接测量。

### 4.4 判定规则(预注册草案,看到任何结果前冻结)

协议:P1(val macro-F1 选 epoch)为主,P2(最后一个 epoch)为佐证。
**60 seeds**(seed 区间在冻结时钉死,与 R6/R8 已消费的 0–29 / 30–89 / 100–129 / 200–229 不重叠)。
每个对比:配对逐 seed,20000 次配对 bootstrap 95% CI。

某数据集 **PASS** 当且仅当三条同时成立:

1. `mean(AUX − A0) ≥ +0.005` 且 95% CI 排零
2. `mean(AUX − AUXPERM) ≥ +0.005` 且 95% CI 排零
3. `mean(AUX − AUXDUP) ≥ +0.005` 且 95% CI 排零

- **CONFIRMED-2DS** — MHC-EN 与 MHC-ZH 都 PASS,且 P2 在两边符号一致。
- **CONFIRMED-1DS** — 恰有一个 PASS,P2 在该数据集上符号一致,且另一个数据集
  `mean(AUX − A0) ≥ −0.002`(无实质损害)。
- **NOT CONFIRMED** — 其余一切。

**VOID 条件**:若 `|mean(AUXPERM − A0)| ≥ 0.005`(打乱标签的辅助头自己就产生了效应),
说明对照构造不稳,本次运行作废,不算通过。

在 30% 转化率下三条件各自的 power ≈ 1.0(z ≈ 5.6 @30 seed,≈ 7.9 @60 seed),
所以三条件联合不会显著吃掉 power。

### 4.5 与 `IDEA_REPORT` §9.9 旧预注册的继承与修订

| 项 | 旧(§9.9,冻结未执行) | 新 | 为什么改 |
|---|---|---|---|
| 触发条件 | 375 项人工审计通过 4 个门,审的是**机器标签** | 200 项人工试点通过 4 个门,标的是**人工标签本身** | `CLAUDE_STANCE_GATE` 已证明机器标签在需要它的行上比常数基线还差,机器标签这条路已死,审计对象随之改变 |
| 标注体量 | 750 条人工判断(只审计 375 项机器标签) | 2892 条人工判断(直接标 1286 项) | 机器标签不可用 ⇒ 人必须标全部训练项,不是抽样审计 |
| 分类法 | 五分类 | **二值 + UNCERTAIN** | 五分类在 549–579 项训练集上每类过稀;二值这一层就已经难住所有 MLLM |
| 质检门 | Krippendorff α ≥ 0.80 全局、≥0.67 逐语料 | Cohen κ ≥ 0.60(明示放宽) | 0.80 是为审计机器标签设的;隐性仇恨上人类二值立场判断达不到 |
| 机制 | Qwen2.5-VL-7B **LoRA**,`L = L_verdict + 0.5·L_stance` | **head 级**,冻结特征,`L = BCE + λ·L_stance`,λ 用 val 选 | LoRA 栈不在这台机器上(§9.10 第 5 条),且 head 训练 10 秒/次 vs 54.9 A100-h |
| 判定 | 3 seed,≥ +1.0 绝对分 | 60 seed,≥ +0.005 且 CI 排零 + 两个对照 arm | 新测量协议(`R6_CONFIRM_FREEZE`);3 seed 规则被实测为**在它自己判的效应上分辨不出来** |
| 数据集 | HateMM + MHC-EN + MHC-ZH | **只有 MHC-EN + MHC-ZH** | §3.2 功效分析 + HateMM 标注成本 |
| 预算 | 54.9 A100-h | **≈ 80 分钟本机 head 训练** | 同上 |

**保留不动的**:λ=0.5 作为副协议;"解锁后不得改 taxonomy / prompt / loss-weight / sampling";
"必须有一个 verdict-only 对照 arm,否则结果不可解释"(即 A0,并且加强为三个对照)。

---

## 5. 与已死接入点的逐条区别

### 5.1 对照表

| 已死项 | 接在哪 | 改训练吗 | 推理时还在吗 | 与本方案的区别 |
|---|---|---|---|---|
| **DESC_CHANNEL** | 新的 768 维**第三输入流**进 fusion MLP(+1.84M 参数,+36.8%) | 是 | **是**,每个视频要一次 MLLM 调用 | 本方案**不加任何输入流**,加 1025 个参数(+0.02%)且推理时不参与。DESC_CHANNEL 的失败被自己诊断为容量/优化问题(纯噪声 arm N 是最差的 −0.0539,信息含量与名次不相关);本方案不经过那条 fusion 路径 |
| **TEXT_MERGE** | 把描述并进**转录字符串**,进 encoder 之前(0 新参数) | 是(重编码+重训) | **是**,每个视频要一次 MLLM 调用 | 本方案**不改 encoder 的任何输入**。TEXT_MERGE 的失败落在它从未触碰的 clean 视频上(−2.00/163),机制是训练集里 20.5% 的行长相变了、决策边界跟着变;本方案不改任何一行的输入 |
| **ARBITER** | **决策层**,post-hoc 概率融合,head 冻结 | **否** | **是**,不确定区间的视频要一次 MLLM 调用 | 本方案**不在决策层做任何事**,推理路径逐位不变。ARBITER 的前提(MLLM 在难例上比小 head 强)被直接证伪:24 个格子里 MLLM 严格更差 21 个、打平 3 个、更好 0 个 |
| **R7_OCRPROV** | **决策层**,logistic combiner 吃 head logit + 6 个规则布尔量,在 val(n=107)上拟合 | 否 | **是**,规则要跑 | 本方案**不学任何组合器**,不在 val 上拟合任何决策层参数。R7_OCRPROV 的真规则比同边际率的随机布尔量还差 −0.0249,机制是 107 项 val 拟合的权重不迁移 |
| **R7_SOFTVOTE** | **训练侧,但替换主任务的 BCE 目标** | 是 | 否 | 本方案**不动主损失**,主 BCE 与 A0 逐字相同。SOFTVOTE 的两个失效机制(目标尺度与固定 0.5 阈值的交互;epoch 选择被破坏)都源于**改了主目标**,辅助头不改主目标 |
| **PILOT_C** | **特征拼接**,typed OCR 块 | 是 | 是 | 同 DESC_CHANNEL:输入侧 |
| **CLAUDE_STANCE_GATE** | **标签供给**:MLLM 能不能产立场标签 | n/a | n/a | 这不是接入点被杀,是**标签来源**被杀。本方案的回应是换来源:人。这是唯一的回应,也是全部成本的来源 |
| **SYNTH_PAIR_PROBE** | **标签供给**:合成对能不能训出归属分类器 | n/a | n/a | 同上 |

**一句话概括本方案与前六个的结构差别:它们全都在推理时还需要点什么(一个 MLLM 调用、
一组规则、一个额外的流);本方案在推理时什么都不需要。** 立场标签只在训练的损失里出现一次,
之后永远消失。

### 5.2 最近的邻居不在上面这张表里:P4

`MLLM_FRONT_RECON.md:218`,route **P4 schema-field auxiliary distillation**:
*"aux linear heads predict MLLM archive fields (explicitness/modality/mechanism/target_group)
from the fused embedding, λ=0.1, heads dropped at eval"* → probe PASS(字段可解码 AUC .62–.93,
字段→标签 AUC .74–.78)→ 训练 **within noise**。诊断:*"the fields are redundant with the hate
label the head is already supervised on"*。

**P4 与本方案在结构上是同一个东西,差别只在辅助目标是什么。**
必须正面说清:

| | P4 | 本方案 |
|---|---|---|
| 辅助目标来源 | **7B MLLM 自己生成的 archive 字段** | **人工标注** |
| 该目标能否被机器产出 | 能——它本来就是机器产的 | **不能**。三次独立实测(qwen 三轮、Claude 三盲标、合成对分类器)全灭 |
| 与二值标签的关系 | 冗余(诊断原文) | **未知,由 G3 测量** |
| 与 test 的关系 | 无 | 无 |

**`MLLM_FRONT_RECON` 的 C4 约束是可预注册的**:*"the MLLM channel must not be decodable from the
frozen features… measure decodability of the new channel from `concat(img, text)` before spending
anything downstream, and kill if it is high."* §0.4 的 **G3 就是 C4 的执行**,
而且是**双侧**的:太高(≥0.85)是 P4 冗余,太低(<0.58)是冻结特征承载不了立场。

`MLLM_FRONT_RECON` 同一节还写了一句直接相关的:
*"C4 is satisfiable by construction for a stance channel, and only for a stance channel.
That is the entire technical argument for this round."*
其依据是 §8.13(PCD spec):匹配的 violation/exemption 政策条款在 CLIP 联合空间余弦 **0.920**,
在多语 mpnet 里 0.833/0.869,而带训练读出时条款方向在 4 个数据集中的 3 个**输给维度匹配的随机方向**
(均值 −0.046 ROC),原文结论:*"separating hate from condemnation/quotation/reclaimed use must
come from a model that reasons, not from an embedding direction."*
**这句话同时是本方案最强的论据和最大的风险**,见 §7.3。

### 5.3 也不在表里:C1(旧 §9.7 的 1.5 分候选)

`IDEA_REPORT` §9.7 的 C1 = "Conditional-Mask Stance Auxiliary LoRA",判决
*"DEAD — sparse proxy, marks 1/55 of the errors it targets: no fuel"*。

区别:C1 的 mask 只覆盖 **139 个 Counter-Narrative 投票项**(而且那是 MultiHateClip 里
一个未文档化的既有字段,不是新标注);本方案的 mask 覆盖 **1286 项里 Q1=YES 的那些**,
按 G4 门至少 25%,即 ≥321 项,预期 400–650 项。**燃料量差 3–5 倍,且标签是为这个任务专门标的。**
C1 的"no fuel"判决在本方案上力度大幅下降,但不为零——见 §7.4。

### 5.4 R6/R8 的"辅助头"记录

`R6_PILOT_FREEZE_2026-08-17.md:170-173` 记录过一条:
*"Taxonomy-preserving auxiliary head(fine MHC/ImpliHateVid subclass labels as auxiliary targets).
Blocked, not free: F82 places 'head-side graded auxiliary' under an admissibility gate that is
'only revivable by user ruling WITH a new mechanism argument'"*。
那一条的辅助目标是**数据集已有标注的更细粒度**,本方案的辅助目标是**新采集的人工标签**——
不同对象。见 §5.5。

### 5.5 F75 / F82 范围核对(逐字)

**结论:两条都不覆盖本方案,但各留一条需要在预注册里正面处理的负担。以下是原文与推理,
用户可以自己判是否同意。**

**F75**(`autoresearch/goal_mllm_plus3/state/directions_tried.json:307`,ban_scope 全文):

> `head-loss swaps of the triplet+BCE hybrid toward vote-consistent (NCA/soft-kNN), contrastive
> (SupCon), or mixup-BCE objectives at 7B frozen-encoder feature scale; tau/alpha retunes =
> tactics, banned; sole KS survivor A1a-nca-t0.1 x ZH val-sel +0.0112 3/3 = within-noise
> measured-not-promoted (D7-DEAD limbo, may cite as observation only). First measured negative for
> trained-reshaping-unlocks-oracle-headroom; F66 selection-locked pools untouched.`

- 关键词是 **swaps ... toward**(把现有 hybrid **换成**列举的三个目标之一)。实际跑的四个 arm
  (A1a/A1b/A2/A3)全是**替换**目标函数。
- ban_scope 与 finding body 里**没有出现** auxiliary / aux head / second head / multi-task / lambda
  任何一个词。
- 唯一射程更宽的一句是 `First measured negative for trained-reshaping-unlocks-oracle-headroom`。
  它自称 *first measured negative*,即**证据/先验,不是禁令**,紧接的下一句还保留了未触碰的对象。
- **判定:不覆盖。但它是本方案的一个已定价逆风**——λ 加权的第二个头确实是 trained representation
  reshaping。预注册里必须明写"F75 是逆风,不是屏障",并且用 AUXPERM 这个对照来说明
  本方案的效应(如果有)不是"多一项损失重塑表示"这件事本身。

**F82**(`directions_tried.json:347`,ban_scope 全文):

> `vote-side Offensive reweighting closed both datasets (any monotone weighting, any tau);
> head-side graded auxiliary = F44-capped + admissibility-gated, only revivable by user ruling
> WITH a new mechanism argument; HateMM out of scope (no Offensive class)`

- 该 direction 自己的 `name` 是 `Graded 3-class soft-label (Offensive reweighting, EN+ZH)`,
  对象是 **MultiHateClip 已有的 3 类 `Label` 字段**。
- ban_scope 的收尾从句 `HateMM out of scope (no Offensive class)` **把禁令的射程锚定在
  "这个数据集有没有 Offensive 类"上**,即锚定在既有标注上,而不是锚定在某种 head 架构模式上。
- `admissibility-gated` 指向的可采性裁定(`refine-logs/GRADEDLBL_PREGATE_RECORD.md:196-198`)
  自述:*"the pregate is read-only on the SAME own-split train annotation at finer granularity"*
  ——主语是"重读数据集已有的标注"。
- **本项目自己的控制性先例**(`refine-logs/GATE0_REOPEN_2026-07-31_REVIEW_ROUND2.md:37`):
  > `I-1 · C07's unblock (c) imports F82's head-side clause onto C07's object, the same
  > over-application the Critical was about. F82's head-side clause governs a "head-side graded
  > auxiliary"; C07 is a cone metric over a harm-act partial order, and no text shows those are the
  > same object. Repair: state (c) as conditional.`
  规则是:**除非有文本证明某对象"就是" graded auxiliary,否则该从句是条件而非禁令。**
- **诚实的另一面**:ban_scope 里 `head-side graded auxiliary` 这个短语本身没有任何限定词把它
  限制在 3 类标签上。把 "graded auxiliary" 读成一种**架构模式**的读者会得到一条很宽的禁令。
  反对这种读法的理由是:**二值 endorse/distanced 目标在字面上不是 graded**——
  "graded" 在整份记录里正是用来区分三级序数与二值的那个词。
- **判定:按written scope 不覆盖。残余歧义只在 "graded auxiliary" 两个词上,
  而项目自己的 R2-I-1 先例把这个歧义解向"不覆盖,除非有文本证明是同一对象"。**
  预注册里应当把上面这段逐字写进去,让用户在批准时看到而不是事后发现。
- 附带说明:F82 的 `F44-capped` 指的是 F44 的第 (2) 条腿(MHC-EN 的错误是 label-limited 而非
  representation-limited),那是一个**先验/逆风**,不是对象级封闭;F44 自身在 registry 里
  **没有 ban_scope**(它只出现在 `positives_bank`)。

**真正卡在本方案头上的不是 F75 也不是 F82,是 `TARGET_LOOP.md:67`:**

> `No auxiliary-task reprise using label-subset semantic fields. New supervision must carry
> information not recoverable from the binary label and prove conditional information gain.`

这是**举证责任,不是禁令**,而且新采集的人工立场标签不是 "label-subset semantic field"。
§4.3 的 **AUXDUP arm 和 §0.4 的 G3 门就是这条举证责任的两个执行装置**:
G3 在花全款前测"能不能从冻结特征解码出来",AUXDUP 在最终判决里测"比直接用二值标签当辅助目标好多少"。

**其它约束核对:**
- `banned_constraints[1]`(gold annotations inside method: time-span, target)——
  作用域是时间跨度与 target group 两个维度,不含立场;且本方案的标签是项目**新采集**的,
  不是数据集金标,只进损失、不进推理。F82 记录里引用的可采性裁定原文
  (`LITSWEEP5_HATEMM_EN.md:125-130`)对同类问题的读法是 *"legal — the class label is the
  supervised target (same as the binary), used only in the loss, never at inference"*。
- `banned_constraints[5]`(MLLM-scores-as-training-signal)——不适用,标签是人工的。
- `banned_constraints[8]`(训练数据只能是单数据集自己的 train split)——遵守,EN/ZH 不混。
- `banned_constraints[2]`(cross-seed ensembles)——不适用,本方案不做 ensemble。

---

## 6. novelty 定位

### 6.1 F3 为什么是空的

`idea-stage/STANCE_LIT_RECON.md`(2026-08-11,两路独立宽扫,arXiv API + OpenAlex + S2 + Anthology)
的核心结论:

- **仇恨视频域零占位。** `abs:"hateful video" OR abs:"hate video"` 在 arXiv 上总共返回 **14 篇**,
  全部枚举读过,**没有一篇带立场字段**;`abs:"stance" AND abs:"video"` 总共 **31 篇**,全部枚举,
  没有一篇是关于仇恨视频的。多条短语合取返回**完全空的 feed**(有效 Atom,零 entry):
  `"stance" AND "hateful video"`、`"speaker stance" AND "video"`、`"use-mention" AND "video"`、
  `"counter-speech" AND "video detection"`。
- **没有任何仇恨视频数据集在其文档化 schema 里带立场标签。** 唯一物理存在的立场邻近标注是
  MultiHateClip 的 139 个 `Counter Narrative` 投票,**其论文从未提及也从未使用**,
  并在多数票里被折叠掉。
- **文本域的最干净正例是 `2206.06423`,而且是"中间任务预训练"(顺序的),不是联合多任务。**
  `STANCE_LIT_RECON:361-363` 原话:*"Nobody has replicated it, nobody has done it as joint
  multi-task (only sequential pretraining)"*。
- **`2404.01651`(NAACL 2024)是 prompting-only**,其 Limitations 明确把 fine-tuning 留作未做。
  它已经拿走了推理时提示这条路(FPR 相对下降 82.6%),但**没有拿走训练侧**。
- **"立场作为 MTL 辅助头"在 `STANCE_LIT_RECON:641` 的裁定是 AMBIGUOUS,1 正 2 负,
  明确标为 open slot**:正例 `1806.03713`(COLING 2018,谣言核实,立场辅助确实有效);
  负例 `2307.03377`(IJCNN 2023,sexism/hate/toxicity 之间的负迁移——
  **但它不是立场论文**,`STANCE_LIT_RECON:542` 明确写 *"Not a stance paper, so do not cite it as
  one; but it is the correct citation for 'a bare auxiliary head bolted onto a hate classifier is
  not free'"*);负例 `2602.12818`(identity fusion,null,0.90→0.88 p=0.28;0.67→0.64 p=0.17)。

**因此 novelty 的准确表述**(不能说过头):

> 立场判断作为**联合训练的辅助监督**,在仇恨内容检测里有一个正例(谣言域)、两个负例(都不是立场
> 或不是仇恨),从未被复制;在**仇恨视频域**里完全没有占位。本工作提供第一个视频域的实例,
> 并且是第一个用**人工采集的立场标签**(而不是机器标签或已有标注的细粒度重读)做这件事的。

**不能说的**:不能说"立场在视频里没被研究过"——视频立场检测本身是存在的任务
(MultiClimate、TikStance α≈0.74、Inter-Stance、DIVERSE)。
只能说"**仇恨视频的立场**没被研究过"。`STANCE_LIT_RECON` §4.1 特意标注了这一点,
而且它其实是有利的:它是"从视频判立场这个感知问题可解"的证据。

**已知的检索缺口(必须随结论一起说)**:WebSearch 预算耗尽,Google Scholar 未搜;
CNKI / 万方未搜;ICWSM / CSCW / LREC / ACM MM workshop 等非 arXiv 场地是最可能藏着占位者的地方。
**"没找到"= "通过 §6 列出的渠道没找到",不等于"不存在"。**

### 6.2 与 Just KIDDIN' 的区别

`2411.12174`(Findings ACL 2025):LLaVA-NeXT 当 teacher,Hate-CLIPper 学生跑在冻结 CLIP 上,
**表征空间的朴素 L2 蒸馏**,+10.6% F1 / **+0.5% AUC**。

三条区别:

1. **辅助信号的类型不同。** 它蒸馏的是**机器 teacher 的表征向量**(L2 回归);
   本方案监督的是**离散的人类判断**(交叉熵)。这不是包装差异:
   `CLAUDE_STANCE_GATE` 实测机器在需要这个判断的 49 行上是 0.551,
   低于同一批行上的常数基线 0.612——**机器 teacher 在这个字段上没有可蒸馏的东西。**
2. **它的增益几乎全在阈值上。** +10.6% F1 对 **+0.5% AUC**,说明排序基本没变,
   动的是操作点。`R8_DECOMP_MEMO` §3(a) 在本项目的基座上把操作点头寸量到
   **+0.25 到 +1.2 分**(在 629–1608 项的 train+val 池上),并且所有现实阈值规则
   (dev 拟合、先验匹配)在 4 个数据集里的 3 个上**低于固定 0.5**。
   **所以这条路线在本基座上已被定价,而且很小。** 本方案的判定指标是 macro-F1 @固定 0.5,
   不从操作点取任何东西。
3. **蒸馏目标可从冻结特征恢复。** 学生和 teacher 看的是同一个视频;表征蒸馏的信息瓶颈
   是 teacher 的感知能力。本方案的 G3 门恰恰要求辅助目标**不能**被冻结特征轻易解码。

### 6.3 与 HVGuard 的区别

HVGuard(EMNLP 2025 Main,`10.18653/v1/2025.emnlp-main.456`):
冻结 GPT-4o 分阶段 CoT 产生自由文本 rationale → 用**同一个文本编码器**编码 →
与 XLM/Wav2Vec/ViT 三路模态嵌入拼接 → 8 专家 MoE + softmax gate。
二值 macro-F1:HateMM 0.8597 / MHC-EN 0.7714 / MHC-ZH 0.8219。

四条区别:

1. **它的额外信号是输入特征,训练和推理都在。** 每个视频在**推理时**需要一次 GPT-4o 调用。
   本方案的额外信号只进损失,推理时不存在,零额外调用。
2. **它的信号是自由文本机器推理。** `2604.24179` 已发表:同一个推理器换成自由文本推理而非
   结构化字段,macro-F1 **−10 / −22 / −13**;`STANCE_LIT_RECON` 的裁定是
   *"Never emit a free-text rationale."* 本方案的辅助目标是一个二值离散标签。
3. **它显式丢弃屏幕文字**(prompt 里明写 "ignoring subtitles in the frames")。
   与本方案无冲突,但意味着它和 OCR 通道的证据不构成对立。
4. **它自己的消融不支持它的融合主张。** Table 7 的 `HVGuard(w/o gate)` 在 MHC-EN 二值 M-F1 上
   是 **0.8045**,高于完整模型的 **0.7714**;Table 3 的 cross-attention arm 也是 0.8037 > 0.7714;
   而且 Table 7 的 `w/o gate` 行在 EN 和 ZH 二值格里逐字节相同,几乎肯定是重复行。
   **对比时应当引用它的 0.8597/0.7714/0.8219 作为外部数字,但不应把它的 MoE 主张当作已确立。**
   另外它用的是重新过滤语料上的 **7:2:1 随机划分、单次运行、无 seed 无 CI**,
   与本项目的固定划分 + 60 seed + 配对 bootstrap 不同表,不能混进同一张表。

### 6.4 最近的邻居:RAMF

`2512.02743`(TMLR,HateMM 0.837):冻结 VLM → typed text records → trained fusion,+3 macro-F1。
**流水线形状被占了,typology 没被占。** RAMF 问的是"客观描述 / 假设仇恨的推断 / 假设非仇恨的推断"
——那是**关于分析者假设的、条件化的对抗推理**,不是**对视频里说话人的态度归因**。
同时它是输入侧通道,推理时需要 VLM 调用。

**时效风险,必须写明**:RAMF / MARS(`2601.15115`)/ LELA(`2602.09637`)出自同一个组
(Zeyu Fu 的实验室),沿这条线稳定出产,**说话人立场是他们很自然的下一个 prompt**。
这是整个计划里最大的时间敏感度。

### 6.5 预答:为什么不直接用立场标签当过滤器 / 决策规则

这是评审一定会问的,有三层回答,第三层是决定性的。

1. **已发表的失败。** `2210.00910`(FCS)把"对被引用内容的立场"做成零样本 NLI 决策规则:
   HateCheck F20 从 **0% 提到 100%**,HateCheck 整体 **+4.6 pp**,
   而在自然语料 ETHOS 上是 **+0.0 pp**。把这条规则一般化(FCSp1)后,
   HateCheck 的 +4.6 pp 也退回 **+0.0 pp**。
   **立场决策规则在诊断套件上完美,在自然语料上值零。**
2. **本项目自己的失败。** ARBITER(决策层不确定区间转交)mean **−0.0135**,0/3 seeds;
   R7_OCRPROV(决策层 logistic 组合器)**−0.0463**,0/30 seeds,而且真规则比
   **同边际率的随机布尔量还差 −0.0249**,机制是 107 项 val 拟合的权重不迁移。
   **这个基座上的决策层对新加的规则是敌对的,而且已经量过两次。**
3. **结构上不可实现。** 过滤器需要在**推理时**拿到立场标签。我们在推理时没有立场标签:
   人工标注只覆盖 train+val,而任何机器都产不出这个标签
   (`CLAUDE_STANCE_GATE` 在需要它的 49 行上 0.551 vs 常数基线 0.612;
   `SYNTH_PAIR_PROBE` AUC 0.441/0.467 且 6/6 符号反转)。
   **过滤器不是"更差",是根本没法实现。辅助损失是唯一能消费一个只存在于训练集上的标签的接法。**
   这一条本身就足以回答这个问题;前两条是佐证。

---

## 7. 风险清单

按"会不会让这笔钱白花"排序。

### 7.1 转化率落在本项目自己的历史区间(最高风险)

30–60% 是**文献先验**。本项目自己的实测先验是:P4 辅助蒸馏 **0%**;
AGGNET 9% / −5% / 0%。若真实转化率是 0–10%,MHC-EN 期望增益 +0.000 到 +0.005,
两个数据集都不过线,NZ$2,800 全部沉没。

**缓解**:G3 门(§0.4)是唯一能在花全款前把 P4 那种失败形态测掉的装置,成本 US$263。
**它不能排除全部风险**——G3 只测"辅助目标是否与冻结特征冗余",测不出"梯度会不会真的改善主任务"。
**这个残余风险无法在训练之前消除,必须由用户承担或放弃方向。**

### 7.2 争议 21 项的标签边界会不会污染标注

**会,但方向是压低转化率,不是产生错误判决。**

`S_PRIZE_DECOMP` §4 把 5 个展示项分成两类:4.1/4.2 偏"标签或上下文问题"
(HateMM 的艾伦秀片段;`MHC/8zLoOqXvk64` 上人类三票 `[1,0,0]` 自己就 1 比 2 分裂,
多数票与全部内容证据反向);4.3/4.4/4.5 偏"判断者与数据集口径不合"
(面板按"有没有身份攻击"判,数据集按"有没有物化/隐性贬低"判)。

- 立场标签**不是**金标,所以"标错"不会直接与金标冲突,只会给共享 trunk 一个与错误不对齐的目标。
  **可观测后果是转化率下降,不是判决被污染。**
- **+3.23 这个数已经是"面板口径下的可回收上限"**,即已经把 21 个争议项**全部排除在外**。
  如果人类标注者(看完整视频)在其中一部分上与数据集口径一致,可回收上限**上行**;
  如果人类反而更偏离,**下行**。`S_PRIZE_DECOMP` §7.2 明确说本分析不能当人类上限。
- **缓解**:§2.3 的第 1 条约束(指南开头逐字附上 MultiHateClip 的原始类定义,
  并明写"你不需要同意它,你要按它执行")就是针对 4.3/4.4/4.5 那一类的。
  对 4.1/4.2 那一类(上下文丢失、人类自身分裂)没有缓解手段——那部分损失接受。

### 7.3 冻结特征可能根本承载不了立场(最大的技术风险)

`MLLM_FRONT_RECON` §8.13(PCD spec,规范阶段即关闭):
匹配的 violation/exemption 政策条款在 CLIP 联合空间余弦 **0.920**,多语 mpnet 0.833/0.869;
K 个成对差方向彼此近正交(0.035–0.067);**带训练读出时条款方向在 4 个数据集里的 3 个上
输给维度匹配的随机方向(均值 −0.046 ROC)**。原文结论:
*"separating hate from condemnation/quotation/reclaimed use must come from a model that reasons,
not from an embedding direction."*

**如果这句话在我们的 Qwen2.5-VL-7B LoRA 特征上也成立,辅助头拟合不上去,
梯度就是噪声,方案在 head 级必死。**

两点缓和:(i) §8.13 是在 **CLIP 联合空间和 mpnet** 上量的,不是在本项目部署的
Qwen2.5-VL-7B LoRA 特征上量的——它是**警告,不是对本基座的测量**;
(ii) 这正是 G3 下界(AUC ≥ 0.58)要测的东西,试点后就有答案。

**G3 不过(AUC < 0.58)的后果**:head 级路线死,只剩 encoder 级(LoRA)路线。
而 LoRA 栈**不在这台机器上**(`IDEA_REPORT` §9.10 第 5 条:栈不在、没有 adapter 存活),
重建成本远超本方案的 80 分钟。所以 **G3 不过 = 方向 HALT,不是自动升级到 LoRA。**

### 7.4 燃料量:mask 内真正要修的项可能很少

MHC-EN test 上 S 项 16/161 ≈ 10%。按同比例外推,MHC-EN train+val 629 项里
**约 63 项是 head 目前立场判错的**。辅助监督覆盖的是所有 Q1=YES 项(预期 400–650),
远大于 63,所以不是 C1 那种 139 锚点的稀疏代理;但**真正需要被翻转的项仍然只有几十个**。
`banned_constraints[10]` 给的净项数标尺是 MHC-EN 需要 16.5 净项换 +0.030(train arena, n=549)。
按判定线 +0.005 折算,大致需要 **2–3 个净项**。这个量级看起来可达,
但它同时说明**效应本身很脆**:几个项的翻转方向就能改变判决,这也是为什么必须用 60 seed 而不是 3。

### 7.5 人类标注与金标口径对齐失败的概率

MHC 自身的人类非一致率:**EN 21.3% / ZH 29.9%**(`IDEA_REPORT:64`)——
即在**主任务**上,四分之一到三分之一的项人类之间就有分歧。
立场判断的分歧率应当**不低于**这个数。这直接决定了:

- **κ ≥ 0.60 这道门有实质失败概率。** 若 κ 落在 0.45–0.60,标签仍有信息但噪声大,
  转化率会被进一步压低。**预注册必须明写 κ < 0.60 即 HALT,不得事后放宽。**
- 仲裁量的估计(320 项)如果实际是 400+,工时上浮约 10%,成本影响有限。

### 7.6 伦理审批与标注者福祉

雇人**看完整的仇恨视频**做标注,在 UoA 需要伦理审批(涉及人类参与者 + 有害内容暴露)。
这是**日历时间的主要风险**(可能数周),不是钱的风险。协议里必须包含:
内容预警、可随时退出、单次连续标注时长上限、支持资源指引。
众包平台(方式 C)对仇恨内容有额外的参与者福祉政策,会进一步收窄可用池。

### 7.7 时效(见 §6.4)

RAMF/MARS/LELA 同一组沿这条线稳定出产,说话人立场是自然的下一步。
标注 3–4 周 + 训练 1 天 + 写作,窗口不宽裕。

---

## 8. 降级方案

任务要求给"只标 MHC-EN/ZH"的降级方案——**§0 的主方案已经就是它**。
理由不是省钱,是 §3.2 的功效分析 + ImpliHateVid 无视频。所以这里给的是**更进一步**的三档降级。

| 档 | 范围 | 判断条数 | 工时 | B 方式成本 | 期望结果 |
|---|---|---|---|---|---|
| **D0(主方案)** | MHC-EN + MHC-ZH,train+val,1286 项 | 2892 | 83 h | **NZ$2,821** | 两个数据集,CONFIRMED-2DS 可达 |
| **D1** | 同上,只标 train(1128 项) | 2536 | 73 h | **NZ$2,471** | 同 D0。省 12%,代价是 5 折 train+val 变体不可用 |
| **D2** | **只标 MHC-ZH**(657 项) | 1478 | 43 h | **NZ$1,441** | 只能出 CONFIRMED-1DS。**单数据集结果不构成方法论文**(`R6_CONFIRM_FREEZE` 的原话:单数据集组件级增益"must not be reported as"一篇方法论文) |
| **D3** | **只做 200 项试点,拿 G1–G4 四个数字,不进全量** | 450 | 13 h | **NZ$439 ≈ US$263** | 不出性能数字。产出是四个可发表在方法章节里的测量:人类立场标注的 κ、立场与金标的独立性、立场在冻结特征上的可解码性、含攻击性内容的比例 |

**为什么不推荐 D2(只标 oracle 最大的一个)**:MHC-EN(+5.10)和 MHC-ZH(+5.01)的 oracle
几乎相等,砍掉任何一个都只省一半的钱,却把结果从"两个数据集"降到"一个数据集"。
`R6_CONFIRM_FREEZE` 已经明确写过单数据集组件级增益不能当方法论文报。
**这是本方案里成本-收益最差的一刀。**

**D3 是最有价值的降级**,因为它把 NZ$2,800 的决定拆成 NZ$439 的信息购买 + 一个后续决定。
即使 D3 之后放弃方向,四个测量本身是可发表的分析材料
(尤其 G3:"立场在冻结多模态特征上的可解码性"这个数,配上 §8.13 的 CLIP 结果,
是一个独立的、有人会引用的观察)。

---

## 9. 执行序与冻结点

用户点头后,顺序如下。每一步的产物在下一步开始前提交。

| 步 | 内容 | 成本 | 冻结点 |
|---|---|---|---|
| 1 | 写标注指南(含 MultiHateClip 原始类定义逐字附录)、标注界面、匿名化 manifest | 0 元,约 1 天 | 指南与 Q1/Q2/Q3 措辞冻结并提交,之后不改 |
| 2 | 伦理审批(方式 B/C 必需) | 0 元,2–4 周 | — |
| 3 | **G1–G4 试点预注册冻结**(四道门的阈值、样本抽取规则、G3 的 5 折 logistic 规格) | 0 元 | 提交后才开始试点 |
| 4 | 200 项试点(100 EN + 100 ZH,按金标 50/50 分层) | US$263(B) | 试点数据 + 四个门的计算脚本一次提交 |
| 5 | **门判决**:四门全过 → 继续;任一门不过 → HALT,写 RESULT 文档收摊 | 0 元 | 判决不得事后修改 |
| 6 | 全量标注(剩余 1086 项)+ 仲裁 | US$1,430(B) | 标签文件冻结、哈希 |
| 7 | **正式预注册冻结**(§4.3 的 4 个 arm、§4.4 的判定规则、seed 区间、λ 网格) | 0 元 | 冻结哈希提交后才允许跑 |
| 8 | 单次提交:2 数据集 × 4 arm × 60 seed = 480 次 head 训练 | ≈ 80 分钟本机 | 分析脚本跑一次 |
| 9 | RESULT 文档 | 0 元 | — |

**四条硬红线的遵守**:①零 test 接触——步 8 之前没有任何步骤读 test 标签或预测,
步 8 的 test 标签只用于最终指标;②判决规则在看结果前冻结——步 3 和步 7;
③盲性——标注者不看金标不看预测,λ 与 epoch 都在 val 上选;④正式运行单次提交——步 8。

---

## 10. 本文档不做的事

- **不构成预注册。** 没有冻结哈希,没有 seed 区间,没有提交任何运行。
- **不改任何代码。** `r4_harness.py` 未被修改。
- **不主张 F82 / F75 已被解除。** §5.5 给的是逐字原文 + 项目自己的先例 + 残余歧义,
  结论是"按 written scope 不覆盖",**这是一个需要用户在批准时同时批准的读法**,
  不是一个已经生效的裁定。
- **不主张 30–60% 转化率成立。** 那是任务给定的假设区间;§3.3 同时给了本项目自己的
  0–10% 实测区间,以及在 0–10% 下两个数据集都不过线的结论。
- **不承诺文献检索是穷尽的。** §6.1 列出了已知缺口。

---

## 附:本文档引用的一手数字与出处

| 数字 | 出处 |
|---|---|
| 可回收 oracle +1.23 / +5.10 / +5.01 / +1.58,均值 +3.23 | `idea-stage/S_PRIZE_DECOMP.md` §3 |
| 26 可回收 / 21 争议 / 2 分裂;人类分裂率 0.438 vs 0.300 | 同上 §2、§5.2 |
| Claude 立场标注 0.5625;错误行 0.551 vs 常数基线 0.612;Fleiss κ=0.90 | `idea-stage/CLAUDE_STANCE_GATE_RESULT.md` |
| 合成对 AUC 0.441/0.467;99 条转录里只有 10 条含归属标记 | `idea-stage/SYNTH_PAIR_PROBE_RESULT.md` |
| 配对差 seed std 0.01189 / 0.01493 / 0.01414 / 0.00444 | 本文档从 `idea-stage/r8_blr/results.json` 重算(600 次 head 训练,P1,30 seeds) |
| 判定线 +0.005 + CI 排零;P1/P2 定义;配对 bootstrap 20000 | `idea-stage/R6_CONFIRM_FREEZE_2026-08-17.md` |
| 划分大小 549/80/161、579/78/149、744/107/215、1283/325/401 | `data/gt/*/{train,val,test}.jsonl` 计数 |
| MHC 三类分布(train+val) | `data/_src_Multihateclip/{English,Chinese}/annotation(new).json`,按 gt id join |
| 视频时长中位 35 / 32 / 76 秒 | ffprobe 抽样各 40 个 |
| ImpliHateVid 无原始视频 | `ls data/video/ImpliHateVid/` 只有 `_id2b2path.tsv` |
| 操作点头寸 +0.25 到 +1.2 分;dev 拟合阈值在 3/4 数据集上为负 | `idea-stage/R8_DECOMP_MEMO.md` §3 |
| F75 / F82 ban_scope 全文 | `autoresearch/goal_mllm_plus3/state/directions_tried.json:307, :347` |
| R2-I-1 先例(不得把 F82 的 head-side 从句外推到别的对象) | `refine-logs/GATE0_REOPEN_2026-07-31_REVIEW_ROUND2.md:37` |
| P4 反重复规则 | `TARGET_LOOP.md:67` |
| P4 结果与 C2/C4 约束;§8.13 PCD 条款方向输给随机方向 | `idea-stage/MLLM_FRONT_RECON.md` §2 |
| F3 空置、`2206.06423`、`2210.00910`、`2404.01651`、`2307.03377`、`1806.03713`、`2602.12818` | `idea-stage/STANCE_LIT_RECON.md` §3.4、§3.5、§3.8、§4.2、:641 |
| Just KIDDIN' `2411.12174` +10.6% F1 / +0.5% AUC | `idea-stage/codex_brainstorm_bundle_r6_2026-08-17.md:182` |
| HVGuard 数字与 Table 7 矛盾 | `research-wiki/papers/jing2025_hvguard_utilizing_multimodal.md` |
| 四个已死接入点的机制与数字 | `idea-stage/{DESC_CHANNEL,TEXT_MERGE,ARBITER,R7_OCRPROV,R7_SOFTVOTE,PILOT_C}_RESULT.md` |
| 旧 750 条预注册 | `idea-stage/IDEA_REPORT.md` §9.8–9.9 |
