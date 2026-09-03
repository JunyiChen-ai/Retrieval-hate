# 规则 4 复核：证据链网络（Evidence-Chain Network）提案

日期：2026-09-04。审查人：独立 proposal reviewer（fable agent，一次）。审查对象：`experiments/20260904_evidence_chain_net/README.md`（提案第 0–4 节）。判据：`RESEARCH_ITERATION_RULES.md` 第 4 条（只挡四类）、第 13 条；`research-wiki/STATUS.md` "当前 novelty 裁定"三条（允许 adaptation / 来源未被本任务占用且必须检索记录 / 改造非平凡并对应具体失败模式）。

读过的材料：提案 README；`docs/20260904_evidence_chain_backbone_novelty_precheck.md` 全文（含附录 A–D）；`experiments/20260903_hier_evidence_mil/README.md` 第 0、4.5、7、9 节；`src/verdict_hmm.py`；`scripts/reproduction_baselines/macilsd/CMA_MIL.py`、`InfoNCE.py`；`experiments/20260903_hier_evidence_mil/train.py`（Candidate.forward、CMAL 调用）；`runs/20260903_hier_evidence_mil/verdict_hmm_only/{hatemm,hateclipseg}/hmm_params.json`；`runs/20260903_hier_evidence_mil/analysis/backbone_mechanism/{hatemm,hateclipseg}_seed234.{txt,json}`；uoa-lab3 上尚未回传的 `runs/20260903_hier_evidence_mil/ablations/hatemm/seed{234,2025}/{no_cmal,no_ema,no_text,no_visual,no_audio,no_ps,no_hmm_input}/metrics.json`（只读）。另做两次 WebSearch 抽查（见末节）。没有改任何其它文件，没有跑训练。

---

## 结论

**GO-with-changes，置信度 7/10。**

- 规则 4 四类 STOP：P1–P4、对比项、整体方法均不触发（逐项见第一节）。预检文档的检索与结论经核对可信；我抽查的两次检索也未发现链式结构化头、密度条件化链、粒度可靠性门用于 hateful video 定位的记录。
- 置信度扣分不在 novelty，而在设计：提案第 1 节的训练目标 "1 − Z_0/Z" 在当前链参数下对**正例视频几乎不产生梯度**（第四节第 1 点给出数值），修订 1 中三 seed 两语料都确认的最强训练机制（块级 MIL，−.036/−.024）被删掉且没有替代；切换率 a 在 K=30 窗格上估计却直接用在 0.667 s 片段链上；对比项的替换丢掉了 CMAL 的跨模态配对而 HateMM seed 234 上 CMAL 值 −.085 AP。这些按规则 4 不构成 STOP（"可能退化"不得作为阻断理由），但按第 4 条以外的审查要点属于实现前必须改正的设计错误，否则搜索结果会把设计缺陷与机制主张混在一起。
- 修改清单见末两节。修改后不需要重开 novelty 复核（规则 4 一次）。

---

## 一、规则 4 四类 STOP 逐项

| 部件 | (1) 来源已用于 hateful video | (2) 纯 ensemble | (3) 纯校准/后处理/平滑 | (4) 纯工程 | 判定 |
|---|---|---|---|---|---|
| P1 内容编码器（三路线性投影 + 一层自注意力 + MLP → u_t） | Transformer 时序编码器已在 MultiHateLoc（WWW 2026）用于本任务；作为部件不构成 STOP，但**P1 本身不能作 novelty 主张** | 否 | 否 | 单看是工程部件 | GO（作为骨干部件；不作主张） |
| P2 视频级证据分布编码器（裁定分布 + 池化内容 → d_v → 链的 p0 与 A(d_v)） | 预检附录 B：最近先例 HHMM（Xie 2003）、IOHMM（Bengio 1995）、AMC-WSVAD（CVIU 2023）、基数势 MIL（UAI 2013）；同任务 MultiHateLoc/LELA/TANDEM/HateClipSeg 无视频级密度条件化链。核对成立 | 否 | **边界**：若 d_v 对链参数不敏感，退化为逐视频偏置=校准；提案已把"d_v 只作 logit 偏置"列为对照 | 否（改的是概率结构） | GO |
| P3 粒度可靠性门（内容 + 裁定上下文 + d_v → γ^f_w, γ^c_j） | 预检附录 C：WeaSEL、CoNAL、Li 2022/2024、CrowdAttention、CANE；HVGuard 的 MoE 是特征空间专家选择，非裁定可靠性。核对成立 | 否（同一 VLM 两粒度，规则 3 明文单一 VLM 不算 ensemble） | 否（在训练图内） | 否 | GO；但预检提出的必要条件（两语料 K30 门均值差异显著）提案已收进第 2 节 |
| P4 可微证据链头（三态 (s,h) 链 + 神经一元势 + 两粒度固定发射势，输出 = 后验 logit） | 预检附录 A：MI-DORF（TIP 2018）的 ζ_t 构造与"至少一个"前向后向等价；GLWS（ICML 2024）2 状态 NFA；Dugong/Linked HMM/CHMM 多分辨率噪声源链。均未用于 hateful video。核对成立 | 否（VLM 裁定作为固定发射势进同一联合模型） | 否（链参与训练、梯度经前向后向回传） | 否 | GO；**单独的"链 + 至少一个"不能作主张**（MI-DORF 已占），主张只能是组合 |
| 后验引导的视频内对比（替代 CMAL） | 来源 MACIL-SD CMAL **已在本项目作为 HateMM/HateClipSeg 的最强训练 baseline 之一**（`docs/duplex/OFFICIAL_VAL_RESULTS.md`），因此对比损失本身不是 novelty；新的只有选择方式（链后验）。预检附录 D 的最近先例 LAP（独立阈值伪标签）与之只差时间耦合与用途 | 否 | 否，但预检指出若消融显示对比项无贡献、后验只在推理起作用则被归为后处理 | 否 | GO 作为训练项，不作主 novelty；边际 |
| P5 后验蒸馏（可选） | 提案第 1 节未纳入方法 | — | — | — | 不审；建议不做（见建议清单） |
| 整体方法 | 无同任务先例（预检核对 MultiHateLoc、LELA、TANDEM、HateClipSeg、HVGuard、SafeLens 六篇；我抽查一次） | 否 | 否 | 否 | GO |

STATUS 三条 novelty 裁定：(1) 允许 adaptation——满足；(2) 来源未被本任务占用并已记录检索——预检四份报告各 25–30 条查询加原文核对，满足；(3) 改造对应具体失败模式、有可证伪预期与机制 control——P2/P3/P4 各对应修订 1 的一个量出效应并各有消融，形式上满足；实质上的问题在第二、四节。

---

## 二、第 0 节"依据"是否支持第 1 节设计

| 机制 | 依据核对 | 对应部件 | 支持强度 |
|---|---|---|---|
| A 视频级密度 | 方差份额 92%/79%、ridge 权重 `frac_blocks_fired` +.95 在 `analysis/backbone_mechanism/*_seed234.{txt,json}` 可核对。"每秒分数换成视频均值后 pooled .632/.837、.690/.687"只在修订 1 README 9.4 有转录，json 无对应键。**"触发块比例与 GT 密度相关 .68、单靠它作常数 pooled .551/.848"在 `runs/` 里没有任何输出文件**，属未落盘数字 | P2 | 中。效应是"网络的 z 主要是视频级量"，这说明网络在做密度估计，不说明**显式**密度估计器优于隐式；只能靠 d_v ≡ 常数消融证明。预期幅度"HateMM −.03"是从网络视频均值 .632 对 HMM 视频均值 .550 推的，两者不是同一模型的两个版本，幅度无依据 |
| B 粒度可靠性 | 四格 GT 率 / HMM ℓ / z 排序在 json `cells` 可核对；修订 2 去原始列 −.036 在 README 8.1 | P3 | 强（HateMM）。HateClipSeg 预注册为"可能不成立"，合理 |
| C 视频内排序 | 遮挡全部内容 HateMM AP −.081、ROC 0；HateClipSeg −.004；残差相关 .638/.349 在 json/txt 可核对 | P1、P4 | 中。注意 uoa-lab3 上 HateMM `no_text` seed 234 = .693/.871、seed 2025 = .644/.871，**去掉文本 HateMM 反而上升 +.03 AP / +.03 ROC**（full seed 234 .661/.841、seed 2025 .643/.838）；HateClipSeg no_text −.007/−.016。P1 把 BERT 拼进来在 HateMM 有反向证据，提案写"HateMM 待出"已过时 |
| D 训练项 | HateClipSeg no_cmal −.015/−.026（三 seed，README 9.6）；**HateMM seed 234 no_cmal = .576/.807，对 full −.085/−.034**（uoa-lab3，未回传、未写进 README）；no_ema HateMM seed 234 .648/.837（−.013/−.004，与 HateClipSeg ≈0 不同，但幅度小） | 后验引导对比 | CMAL 是两语料都确认的训练项，且 HateMM 幅度最大。但 CMAL 是**跨模态**配对（audio top-k 均值为 query、visual top-k 为正、visual bottom-k/normal 为负，见 `CMA_MIL.py`），提案换成同一表示 h_t 的视频内对比，同时改了选择方式与配对结构，依据不支持第二处改动 |
| 时间耦合、块 OR | indep_hmm、flat_coarse 三 seed 两语料降（README 7.2） | P4 | 中。这两条消融是把 HMM 后验**作输入/先验**时的效应，新设计把链改成**输出头**，角色不同，只能算间接依据 |
| 块级 MIL（−.036/−.024，修订 1 三 seed 两语料最强机制） | README 7.2 | **无对应部件**。提案第 1 节说"块 OR 观测让 K4 裁定在链上自然给出块级监督"，这是把固定发射势当监督；K4 势对 u_t 的梯度只经 1 − Z_0/Z 传，见第四节第 1 点，实际为零 | 依据薄弱：最强确认机制被删除且未替代，违背提案自己的原则"只做证明在 work 的事" |

不带进来的清单（SniCo、位置、EMA、P(s) 列等）与修订 1 消融一致，没有问题。

---

## 三、预注册消融表是否完整

已有并合格：P2 常数 d_v；P3 γ ≡ 1；P1 u_t ≡ 0；链头对 top-k 头；时间耦合；块 OR 平铺；去对比；对比选择方式三臂；φ ≡ 0；整个骨干换 AVCE；d_v 只作偏置；门只读内容；门权重分布必要条件。预检要求的三处退化对照（noisy-or、偏置、门常数/只读内容）全部收进。

缺失或不严格：
1. **块级 MIL 项**没有"加回"臂（见第二节末行、第四节第 1 点）。
2. "链头 vs top-k 头"臂写的是"修订 1 的头"，须写明是否含块级 MIL（修订 1 的头 = top-k + 加法先验 + 块级 MIL）；不含则不是修订 1 的头。
3. 对比项缺"配对结构"维度：需要"后验选 × 跨模态配对（CMAL 原配对，只换选择）"臂，否则无法区分 CMAL 起作用的是跨模态对齐还是视频内分离。
4. no_text 臂缺（两语料）；HateMM 已有反向证据，必须报。
5. u_t ≡ 0 臂应同时报 within：该臂 ≈ HMM 后验 + d_v + γ，修订 1 的 HMM 后验单独 within HateMM 只有 .570，这一臂直接量出链后验作输出对 within 的下限。
6. P2 输入拆分：d_v 只读裁定分布 vs 读裁定分布 + 池化内容（机制 A 的依据是裁定分布特征 `frac_blocks_fired`，池化内容没有依据）。
7. 预期幅度：只有 P3（.036）、P1（.08）、对比（.015/.085）有对应量；P2 的 −.03/−.02 与"P4 两语料下降"没有量的依据，应改写为方向预期。
8. 提案第 2 节末"目标：不低于修订 1"——规则 8 门是对 baseline 表，对修订 1 是持平即可，但应写明规则 14(b)：领先小于 std 不得写"超过"。

---

## 四、设计漏洞

### 1. 训练目标 "1 − Z_0/Z" 形式正确，但对正例视频几乎无梯度（最重要）
形式：P(y=1|x,b) = 1 − P(s ≡ 0|x,b) = 1 − Z_0/Z，Z_0 = 全 (0,0) 路径在三态链上的未归一化权重。只在 s=1/h=1 上放对数似然比势等价于两态都放似然（s=0 因子对所有路径公共，在比值中约掉），K4 势在块末 h=1 上发射同理。**公式正确**。

梯度：记 ρ = Z_0/Z。正例损失 −log(1−ρ)，∂/∂u_t = −ρ/(1−ρ)·P(s_t=1|x,b)；对 γ、d_v 同样正比于 ρ。负例损失 −log ρ，∂/∂u_t = +P(s_t=1|x,b)。用 `verdict_hmm_only/*/hmm_params.json` 的 EM 值算 ρ 的先验上界：
- HateMM：A00 = .9577，p0(0) = .649。K=30 窗格：.649 × .9577^29 = .19；若按提案在 0.667 s 片段链上直接用同一 a（T=150）：.649 × .9577^149 = 1.0e-3；T=300：1.6e-6。
- HateClipSeg：A00 = .9849，p0(0) = .818。窗格 .53；T=150 时 .084。
- 加上裁定证据：每个触发的 K30 窗给 s=1 路径 φ = log(q_f/r_f) = 2.38（HateMM）/ 2.11（HateClipSeg），ρ 每窗再乘 .09 / .12。正例视频触发 5 个窗即 ρ ≤ 1e-6 量级。

结论：正例视频对 u_t、γ、d_v 的梯度数量级 ≤ 1e-6，训练信号只来自负例（把负例视频里链认为是仇恨的片段压低）。u_t 学成"负例抑制器"，正例内部的排序完全来自固定 VLM 势、d_v 与对比项；而对比项的正负集由后验决定，后验在正例里又由 VLM 势主导，所以 u_t 在正例上被训练成复现 VLM 裁定。这正是 Wang–Li–Metze 2018 的 noisy-or 失败的链版本，链的转移势不改变它（转移只影响 ρ 的先验大小，证据一进来 ρ 就趋零）。提案第 1 节声称 K4 块 OR "自然给出块级监督"不成立，因为块级观测也只经 ρ 传梯度。修订 1 的 top-k / 块级 MIL 之所以有效，就是因为它们给每个正例视频的若干片段一个不消失的正向梯度。

规则 4 不允许以此 STOP；作为设计错误列入必须修改：给正例视频一个片段级正向梯度。可选形式（任选一种预注册）：(a) 保留修订 1 的块级 MIL（top-k 于 u_t，软标签 = 固定 HMM 块后验），权重固定 1；(b) 其精确概率版本：对固定 HMM 后验判正的每个 K30 窗 w（或 K4 块）加窗内"至少一段为 1"的边际似然项 P(∃t∈w: s_t=1|x,b)，窗长 5–10 段，ρ_w 不会趋零；(c) MI-DORF 式计数势。无论哪种，"只用 1 − Z_0/Z"降为消融臂。

### 2. 切换率 a 的时间尺度错误
`src/verdict_hmm.py` 的 A、p0 在 K=30 窗的链上估计（每视频 29 次转移）。提案链在 0.667 s 片段上运行（T 最多 300），却"切换率 a 固定为 EM 值"。同一 a 在片段步上意味着期望切换次数放大 T/K ≈ 5–10 倍，时间耦合被大幅削弱（HateMM 期望仇恨 run 长 1/A10 ≈ 13 步，在窗格是 13 窗 ≈ 视频的 43%，在片段格只有 13 段 ≈ 9 s）。必须做其一：把 A 换算到片段步（A_seg = A^{K/T}，或 a_seg = 1 − (1−a)^{K/T}，按视频 T 逐视频算）；或在片段格上重新 EM（发射按 1/n_w 分摊）。这与第 1 点相关：尺度正确后 ρ 的先验上界回到 .19/.53，但证据一进来仍趋零，第 1 点不因此消失。

### 3. 窗势 1/n_w 分摊是新的近似
q_f、r_f 是"整窗裁定 | 窗状态"的发射率；把 log LR 均分到窗内片段等价于假设窗裁定是片段独立专家的乘积，只有窗内片段全为 1 时才施加完整证据。这不是 EM 拟合的生成模型。预检列的两处理论近似（EM 条件对象、K4 块结构）没有覆盖这一条，须一并写进方法节，或改为在链上引入窗级 OR 变量（与 K4 的 (s,h) 同构）。

### 4. d_v 条件化转移的形式
A(d_v) = [[1 − a·d_v, a·d_v], [a(1 − d_v), 1 − a(1 − d_v)]]，平稳分布 (1 − d_v, d_v) 正确；与 EM 值在 d_v = A01/(A01+A10) 处重合（HateMM .359 ≈ p0(1) .351，HateClipSeg .188 ≈ .182），参数化自洽。有效性要求 a ≤ 1；EM 值 .118/.080 满足，实现须断言。语义上是"密度改占用率、不改突发性"，可辩护。缺陷：p0 = (1−d_v, d_v) 且 d_v ∈ (0,1) 开区间无界时，d_v → 1 使 Z_0 → 0（正例损失恒 0）、d_v → 0 使负例损失恒 0，即**目标函数存在只靠 P2 的精确最优解**（提案风险 2 承认）。加上第 1 点，P2 是唯一从正例得到有效梯度的部件（因为它直接改 ρ 的先验），所以它必然学成视频分类器。若 P2 输入含池化内容，这条捷径更宽。缓解（须择一预注册）：d_v 只读裁定分布、不读内容；或 d_v 限制在 train 视频固定 HMM 后验密度的经验范围内（常数，非搜索量）；并报告 test 上 d_v 与 GT 密度的相关、d_v 饱和比例作为诊断。

### 5. 门 γ 与 u_t 的可识别性
γ ∈ [0,1] 只能缩小势、不能翻转，这与四格证据一致（(1,0) 格 GT .161 高于 (0,0) .070，只需削弱不需翻转）。门读 b^f_w、b^c、邻窗，因此 γ·φ 可表示任意保号的裁定上下文函数，"可靠性"解释只有在报告门权重分布时成立——提案已把两语料门均值差异列为必要条件，够用。γ 与 u_t 互相替代的问题提案已写进风险 3，消融覆盖。另：sigmoid 输出"初始化为 1"不可精确实现（需 +∞ 前激活、零梯度），须写明实际初始值（如偏置使 γ₀ ≈ .95）。

### 6. 链后验作输出分数对 within 的风险
后验 p_t 在证据强处会饱和到 0/1，若用 logit(p + eps) 会产生大量并列，within 计并列为 .5。必须在对数域直接输出 log α_t(1) + log β_t(1) − log α_t(0,·) − …（三态需合并 (0,0),(0,1)），不经 p 再取 logit。修订 1 的 HMM 后验单独 within HateMM .570 < 下限 .632，新方法 within 的来源只有 u_t、γ 和 d_v（d_v 视频级，对 within 无贡献），而 u_t 按第 1 点几乎不从正例学到东西。风险真实；第三节第 5 条要求 u_t ≡ 0 臂报 within 就是为了量它。

### 7. 正例"只需一段为 1"是否让链退化
不会退化为常数，但会退化为"VLM 势 + 负例抑制"，见第 1 点。

### 8. CMAL 替换是否丢失真正起作用的原因
`CMA_MIL.py` 核对：选择器是模型自身 bag 分数（mmil > .5）与单模态 logits 的 top-k / bottom-k（k = T//16 + 1），配对是**跨模态**：audio 异常 top-k 的均值为 query，visual 异常 top-k 为正键，visual 背景（bottom-k）与 visual 正常视频 top-k 为负键，再反向一次；InfoNCE 温度 .1；权重 λ = min(1, lamda_cof·epoch) 线性爬升（`train.py` 第 379 行），不是常数 1。提案的替换同时改了三件事：选择器（后验）、配对（同一 h_t 的视频内三集合，不再跨模态）、权重（常数 1）。依据（no_cmal HateClipSeg −.015/−.026，HateMM seed 234 −.085/−.034）只说明 CMAL 整体有用，没有把跨模态对齐与视频内分离拆开。HateMM 上 −.085 是修订 1 所有单项消融里最大的，替换风险最高。必须：主方法保留 CMAL 的跨模态配对结构、只把选择换成后验（这是与依据最近的最小改动），或至少加"后验选 × 跨模态配对"臂；权重爬升与常数 1 的差别写明。

### 9. 对比正负集用 p_t > .5 / < .5 硬阈值
.5 是隐含常数；且按第 1、4 点正例的后验普遍偏高时背景集可能为空，MACIL-SD 用 top-k/bottom-k 保证非空。须写明空集处理或改为后验排序的 top/bottom 比例（比例也是常数，需声明）。

---

## 五、超参数

"方法级 0"只在"不搜索"意义上成立。以下是固定常数，README 必须逐个列值并标明来源（继承 MACIL-SD / 修订 1 / 本提案新定）：d = 128、4 头、1 层、dropout（搜索）、InfoNCE 温度 .1（继承 CMAL）、对比权重 1（**改自 CMAL 的线性爬升**）、正负集阈值 .5（新）、门初始值（新，且"1"不可实现）、EM 迭代 40 与 q/r/A/p0（继承 `verdict_hmm.py`，按语料 train 估计）、裁定二值化阈值 level ≥ 2（继承）、窗→片段中点映射与 1/n_w 分摊（新）、a 的尺度换算（缺，第四节第 2 点）、d_v 的范围/输入集（缺，第四节第 4 点）、P2 的 7 维特征表与 MLP 宽度、P3 的 MLP 宽度、epoch 50、max_seqlen 采样方式、I3D 5-crop 训练/测试处理（沿用修订 1）、log 域 eps。其中对结果敏感且无依据的：对比权重常数 1（CMAL 在两语料的效应是在爬升下测得的）、阈值 .5、d_v 范围、a 尺度。规则 7 声明的搜索空间（lr、dropout、max_seqlen，20 trial，目标 test (AP+ROC)/2，within 剪枝，validation 选 ckpt）合规。

---

## 六、规则 13

满足：两语料同一架构、同一损失、同一训练与推理流程；语料间差异只有 lr/dropout/max_seqlen（标量，同一搜索空间自动选出）与由同一程序从各语料 train 标签 EM 估出的 q/r/a 常数（数据导出，非人工按语料设定，须在 README 这样表述）。预期"P3 在 HateClipSeg ≈ 0"是预测不是开关，合规。注意 HateMM 的 no_text 反向证据不得转成按语料去文本（那是模块开关，规则 13 禁止）；两语料都带 BERT 或都不带，由同一决定。

---

## 必须修改（实现前）

1. 训练目标增加对正例视频不消失的片段级梯度项（第四节第 1 点三种形式任选其一，预注册），"仅 1 − Z_0/Z"降为消融臂；README 给出 ρ 的数量级论证或直接引用本文数字。
2. 切换率 a（及 p0）换算到片段步或在片段格重估 EM；实现断言 a ≤ 1。
3. 对比项：保留 CMAL 跨模态配对、只换选择器为后验（推荐），或加"后验选 × 跨模态配对"臂；写明权重常数 1 对 CMAL 爬升的改动；写明正负集空集处理。
4. P2 的退化捷径缓解择一预注册：d_v 只读裁定分布，或 d_v 限制在 train 固定 HMM 后验密度的经验范围；加 test 诊断（d_v 对 GT 密度相关、饱和比例）。
5. 输出分数在对数域直接取后验 log-odds，不经 logit(p + eps)。
6. 消融表补：块级 MIL 加回/去掉臂；"top-k 头"臂写明含块级 MIL；no_text 两语料；u_t ≡ 0 臂报 within；P2 输入拆分臂。
7. 第 0 节把 "相关 .68 / pooled .551/.848" 落盘为 `runs/` 文件并引用路径，或标注"未落盘、不作依据"；补 HateMM no_cmal（seed 234 −.085/−.034）、no_text（+.032/+.030）、no_ema（−.013/−.004）数字及来源（uoa-lab3，回传后引用本机路径）。
8. 方法节理论近似增加第三条：窗势 1/n_w 分摊。
9. 超参数节按第五节逐项列固定常数与来源；"方法级 0"改为"方法级不搜索，固定常数如下"。

## 建议修改

1. P4 的 novelty 表述限定为组合（时间转移 + 视频级似然 + 神经一元势 + 两粒度固定发射势 + 可靠性门）；related work 明写 MI-DORF 与 GLWS 的等价部分，AMC-WSVAD 正面引用。
2. 门 γ 初始值改为可实现的 ≈ .95 并写明；考虑 γ 只读裁定上下文 + d_v（不读内容）作主设计，"读内容"作臂，减少与 u_t 的互相替代。
3. P5 蒸馏不做；预检已判其单独 novelty 弱且有自增强回路风险。
4. 预期幅度只对有量的项写数字（P3 .036、P1 .08、对比 .015/.085），其余写方向。
5. 因 HateMM no_text 为正向，预注册时明确：文本保留与否由两语料 no_text 三 seed 均值决定、两语料统一。
6. 先在实现前用固定 HMM 参数离线算一遍两语料 train 正例的 ρ 分布（不训练、不读 test），作为第 1 条修改的依据附在 README。

---

## 检索记录（抽查，2026-09-04）

1. WebSearch `hateful video temporal localization hidden Markov model OR "Markov chain" OR CRF video-level labels forward-backward HateMM 2026`：命中 TANDEM（arXiv 2601.11178）、HateClipSeg（2508.01712）、Temporal Label Noise（2508.04900）、MultiHateLoc（2512.10408）；无任何 HMM/CRF/前向后向用于本任务的记录。与预检附录 A、B 第 3 节一致。
2. WebSearch `weakly supervised video anomaly detection "video-level" density OR ratio conditioned Markov chain transition posterior localization "at least one" marginal likelihood 2025 2026`：命中 STPrompt（MM 2024）、GV-VAD、OrthoVAD 等，无密度条件化链或"至少一个"边际似然的 WSVAD 工作。与预检附录 B 一致。
