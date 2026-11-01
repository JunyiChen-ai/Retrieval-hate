# 规则 4 proposal review（第 6 版）— 20261101_r5_completion 第 3.3 节

**判定：GO。** 规则 4 的四种 STOP 情形均不成立（逐条见第 3 节）。规则 13 检查通过（第 4 节）。

- 审稿人：独立 agent（Fable 5.1），2026-11-01，一次性。未运行实验，未改动本文件以外的任何文件。
- 审阅：`README.md` 第 2、3.2、3.3 节；r5 方法 `experiments/20260925_query_paradigm/README.md` 第 2、17 节与 `research-wiki/method_framework.html`；代码 `src/qtl/qtree.py:fit_anchored_shared`（对照 `fit_anchored`）、`src/qtl/train.py` 的 `answer_model "anchored_shared"` 分支、`cpolicy.run_batch`（copy_pi 入口）、`experiments/20261101_r6_length_table/search.py`；上两次审稿 `REVIEW_RULE4.md`、`REVIEW_RULE4_R4.md`（已查过的树 / noisy-OR / EIG / 标注者模型 / 零膨胀文献不重复）。
- 未以可识别性、退化、shortcut 或预期涨幅作为判定依据（规则 4 明文排除）。

## 1. 代码确认：提案实际改了什么

- `fit_anchored_shared`：状态 0 的表 θ 与每档长度斜率 ω 在负例训练视频全部节点上做多项逻辑回归（标准化对数长度为协变量）；状态 1 的表在 ω 固定下只用正例根节点拟合。与 r5 的 `fit_anchored(length_term=False)` 相比，唯一差别是多了这条共用斜率。输入仍只用视频标签定状态，不用逐秒标注。
- `train.py:239–245`：该表建成 `AnswerModel`（`omega0` 非空），`am.requires_grad_(False)`；`am.loglik(answers, lengths)` 进 `A3` → `ctree.log_evidence`（训练损失，line 373–380），`qam = am` 进 `ev.run`（validation 选 epoch、test 评测）。`cpolicy.run_batch` 用同一张表构造候选节点 P(o | s) 算 EIG 与读到答案后的似然；`copy_pi=None` 时复制混合不出现（line 106–107）。即：训练目标、选节点、后验三处用同一张表，与第 3.3 节表述一致。

## 2. 检索记录（WebSearch 12 次、WebFetch 4 次，2026-11-01）

| # | 查询 | 结果 |
|---|---|---|
| 1 | hateful video localization VLM question answering length-dependent answer model segment length noisy oracle HateMM HateClipSeg | HateClipSeg、MultiHateLoc（2512.10408）、LELA（2602.09637）；无长度相关答案模型 |
| 2 | weakly supervised hateful video localization 2026 HateMM DeHate HateClipSeg frame-level | 只有 MultiHateLoc（MIL + 跨模态融合，无 VLM、无概率观测模型） |
| 3 | hate speech video detection multimodal LLM temporal localization binary search tree queries 2025 2026 | LELA、TANDEM（2601.11178）、MultiHateLoc、2508.04900（标签噪声分析）；无树 / 无查询策略 |
| 4 | hateful video "expected information gain" / "active querying" / "adaptive questioning" VLM 2026 | 无命中 |
| 5 | HateMM Qwen2.5-VL localization segment queries answer noise model length covariate 2026 | 只有 MultiHateLoc |
| 6 | DeHate dataset hateful video localization method 2026 VLM answers probabilistic | DeHate 原文只做检测；CLARA（2608.15905，检测、VLM rationale 经门控 Transformer，无答案模型） |
| 7 | video anomaly detection VLM yes/no window queries answer likelihood depends on window length | Probe-VAD（固定窗 Yes/No 似然，无长度模型）、2603.04727（窗长 1–3 s 只是超参扫描）、VADTree |
| 8 | VLM temporal grounding segment-length bias yes/no verification calibration | FV-Action（2608.08315，窗与事件宽度比的失败分析，非答案模型）、VTune；无长度相关似然 |
| 9 | temporal action localization duration bias length prior post-hoc calibration | ASM-Loc（短段分数偏低→按时长重采样）、VSGN（增强）、Scale Matters（时长先验）；无"长度项进观测模型"的做法 |
| 10 | group testing dilution effect pool size dependent sensitivity Bayesian decoding | Tatsuoka et al. Biostatistics 2023（贝叶斯分组检测，检验响应随池大小变化 + halving 规则）；Wang–McMahan–Gallagher 2015（含稀释效应的分组检测回归） |
| 11 | probabilistic bisection noise depends on interval length | Kaspi–Shayevitz–Javidi IEEE T-IT 2018（1611.08959：噪声随探测区域大小增大的自适应搜索）；Nowak 2009 按查询定噪声 |
| 12 | Dawid-Skene annotator model item covariates GLAD | GLAD（Whitehill 2009）、Carpenter 2008、Han–Adıgüzel–Carpenter 2024（标注者模型加项目协变量） |

全文 / 摘要核对（WebFetch）：LELA 逐帧 LLM 打分、跨模态取 max、固定阈值 τ，无长度项、无概率模型、无查询策略；TANDEM 双模型 tandem RL 给时间戳，无答案似然模型；2603.04727 窗长只是实验因子；CLARA 视频级检测。
结论：hateful video 文献里没有任何"VLM 答案的长度相关观测模型"或其估计方式；最近的任务外先例是 Kaspi 2018（区域大小相关噪声）与 Tatsuoka 2023（池大小相关检验响应），论文须引用并对照。

## 3. 四种 STOP 情形

1. **来源方法已用于 hateful video**：否。第 2 节 1–6、12 条检索无命中；LELA / TANDEM / MultiHateLoc / CLARA 全文或摘要核对均无长度相关答案模型。
2. **纯 training/test ensemble**：否。单一 VLM、单一网络、单一答案表；去掉复制似然后组件更少。
3. **纯 calibration / 后处理 / 平滑**：否。规则 3 禁的是 inference 时对分数的后处理（平滑、calibration、按语料路由）。本提案的长度斜率是观测模型 P(o | s, ℓ) 的参数：(a) 进训练损失 −log P(答案 | G=Y, x)，骨干 π、g、φ_n、链全部按这张表的梯度训练（第 1 节代码确认），训练出的网络与 r5 不同；(b) 进 EIG，改变问哪个节点；(c) 进后验的精确推断。最终分数 P(y_t=1 | x, 答案) 由同一精确推断直接输出，之后没有任何映射作用在分数上。它与 r4/r5 已过审的锚定表性质相同（训练前用训练标签定状态拟合一次、固定），只是多一个协变量；在 r5 上做的只是诊断，提案本身要求重训（第 3.3 节 Q1–Q3）。故不是 calibration / 后处理。
4. **纯工程技巧**：否。斜率是从负例训练视频估计的模型参数，不是搜索的超参；不换特征、不加数据增强、不改训练配置；提案含完整的估计方法、训练 / 推理 / 停止流程与预注册判定。改动小不是 STOP 条件。

## 4. 规则 13

`experiments/20261101_r6_length_table/search.py` 的 `EXTRA_DEFAULTS`（`soft_both`、`node_prior`、`anchored_shared`、`length_term`）对三语料相同；`src/qtl/` 无按语料分支的方法代码（只有目录映射 `data.py:24` 与 `train.py:530` 的诊断子组报表，不影响分数）；答案表参数由同一程序在各语料自己的训练集上估计（与 r5 锚定表同一性质）；搜索空间同 r5 四个标量。按数据集的仇恨定义是 r5 第 1 步已授权的内容，本提案未新增。通过。

## 5. 非阻断备注

- 长度项本身在本项目第 0/1 版就有（`train.py:100` 注释、r5 README 2.2 的 ω·log ℓ，两状态各自斜率、与网络联合学习）；第 6 版新的是估计方式（锚定、共用斜率、只在状态已知的负例上估）。论文的 claim 应落在估计方式上，消融须含"两状态各自斜率（`fit_anchored(length_term=True)`，3.2 节 HCS 崩溃）"对"共用斜率"。
- 共用斜率的依据（有无仇恨节点每加倍 +.35 到 +.52 对数几率）目前只在 README 3.2 一句话；论文需按语料给出该数与出处文件。
- 第 3.2 节表格是旧表下训练的网络的测试期数字，3.3 节正确地要求重训后再判定；汇报时不要把 3.2 的数字当第 6 版结果。
- Q2 的对照 r5 数字含复制似然，第 6 版去掉复制似然——比较对象是部署的 r5 整体，写明即可。
- 相关工作须新增：Kaspi–Shayevitz–Javidi 2018、Tatsuoka et al. 2023、Wang et al. 2015（稀释效应）、GLAD / Han 2024（标注者模型的项目协变量），以及 ASM-Loc 的时长偏差（用重采样而非观测模型处理）作为对照。
