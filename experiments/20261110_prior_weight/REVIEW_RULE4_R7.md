# 规则 4 proposal review（第 7 版候选）— 20261110_prior_weight 第 7、8 节

**判定：GO。** 规则 4 的四种 STOP 情形均不成立（逐条见第 3 节）。四条里第 3 种（纯 calibration / 后处理）最接近成立，判断依据和保留意见写在 3.3。规则 13 检查通过（第 4 节）。第 5.1 节的 5 条开跑前备注已全部写进 README 第 8 节的 "Fixed before the search" 一段（逐条核对见 5.1）。

- 时间线说明：本审稿读的是 README 与 `launch/sbatch_search.sh` 的初始版本。本文件初稿（判定 GO，含 5.1 的 5 条）写出后，主 agent 改了 README 和启动脚本并提交（改动一部分来自规则 6 代码审查，一部分来自本文件初稿的 5.1）（提交说明 "revision 7: rule-4 review GO, rule-6 review PASS; protocol points fixed before the search ..."，2026-11-10 20:48）。本审稿随后逐行看了改动：第 7、8 节的机制、来源、输入、预注册判定未变；变的是搜索协议、报告清单、措辞说明，以及第 2 节一处事实更正（c 在 60 个 trial 里有一个为负：HCS，-.002）。这些不影响第 3、4 节的判断。本文件是定稿，5.1 已改为"提出的备注 + README 现状"。

- 审稿人：独立 agent（Claude Fable），2026-11-10，一次性。未运行实验，未改动本文件以外的任何文件。
- 审阅：`README.md` 第 1–8 节（提案 = 第 7、8 节）；`RESEARCH_ITERATION_RULES.md` 规则 3、4、7、13、14；第 6 版 `experiments/20261101_r6_length_table/README.md`、上次审稿 `experiments/20261101_r5_completion/REVIEW_RULE4_R6.md`（已查过的树 / EIG / 答案模型文献不重复）；`experiments/20261110_answer_likelihood/README.md` 第 2、4、5 节；代码 `src/qtl/prior_weight.py`（`collect`、`_chunks`、`_nll_all`、`fit`、`as_tuple`）、`src/qtl/cpolicy.py:run_batch`、`src/qtl/train.py`（训练循环、每 epoch 的权重估计、`evaluate`）、`src/qtl/ctree.py`（`up`、`log_evidence`、`marginals`）、`src/qtl/search.py`、`experiments/20261110_prior_weight/search.py` 与 `launch/sbatch_search.sh`，以及本轮对 `src/qtl` 的全部改动（`git diff`）。
- 用户 2026-11-10 的裁定（"规则 3 禁止把 calibration 当后处理。估计方式可以不用标签(用 validation 视频的回答的边际似然)"——"我同意这里"）是对规则 3 的裁定，本审稿不重新判断规则 3；规则 4 的四条由本审稿独立判断。
- 未以可识别性、退化、shortcut 或预期涨幅作为判定依据（规则 4 明文排除）。

## 1. 代码确认：候选相对第 6 版实际改了什么

- **配置**：`experiments/20261110_prior_weight/search.py` 的 `EXTRA_DEFAULTS` = 第 6 版的四项（`soft_both`、`node_prior`、`anchored_shared`、`length_term`）加 `prior_weight: "loc"`。`plan`、`eval_table` 取默认 `None`（预算感知计划、validation 拟合的三状态表不启用）；`train.py:288` 只接受 `None / loc / joint / root_all`，`trust` 两种模式进不了训练入口。与 README 第 7 节"只采用位置权重"一致。
- **估计**（`prior_weight.fit(mode="loc")`，`:175–177`）：a = 1、b = 0 固定，对一个无约束标量 c（初值 1，L-BFGS）最小化 `_nll_all`（`:123–131`）= −Σ_视频 log[P(G=1|x) P(o|G=1,x) + P(G=0|x) P(o|G=0)]，其中 `ctree.log_evidence(fo, c·S, G, A3, chain, c·PHI)`。输入只有网络输出（`collect`）、validation 视频全部缓存回答在固定答案表下的对数似然（`_chunks`）、固定的 chain（`_DoubleChain`，detach）。函数签名里没有标签；带标签的 `label_reference`（`:235`）只被 `stage_a_weights.py` 调用，`train.py` 不调用。
- **c 作用在哪**：`ctree.up` 的零膨胀分支返回 `g + v1 − v1_prior`（`ctree.py:142`），所以 P(G=1|x) = sigmoid(g) 不随 c 变；c 只改 G=1 条件下的位置先验 P(y | G=1, x) ∝ exp(c s·y) chain(y) Π exp(c φ_n z_n)。
- **推理**（`cpolicy.run_batch:287–290`）：循环开始前 `S, PHI ← c·S, c·PHI`。之后每一步的 `ctree.marginals`（`:329`）都用缩放后的势：输出分数 `p`（`:341`）、选节点用的节点边缘 `mq` → `w` → `asker.eig` → argmax（`:387–417`）、停止统计的两次钳位推断（`:443–461`）。分数由精确推断直接输出，之后没有任何映射。
- **训练**（`train.py`）：训练循环（`:352–439`）不读权重，`ctree.log_evidence(fo, s_tree, g_tree, A3, chain, phi)`（`:408`）用未缩放的 s、φ，损失与第 6 版相同。每个 epoch 结束后（`:444–451`）在 validation 上估计该 epoch 的 c，validation 的 8 次提问在该 c 下运行，选 checkpoint 的判据（validation AP 与 ROC 均值，`:452–462`）在该 c 下计算；被选 epoch 的 c 存进 `model.pth`（`:472–473`），`evaluate` 的 validation / test 运行都用它（`:479–480`、`:534–537`）。test 视频的回答不参与估计 c。
- 结论：与 README 第 8 节的表述一致。相对第 6 版，改动是 (1) 推理时位置先验乘 c（后验、提问、停止统计同时改），(2) checkpoint 选择在 c 下进行；**网络的梯度更新不变**——同一组超参数下训练轨迹与第 6 版相同，只可能选到不同的 epoch。

## 2. 检索记录（WebSearch 31 次、WebFetch 12 次，2026-11-10）

| # | 查询 | 结果 |
|---|---|---|
| 1 | hateful video detection localization VLM answers Bayesian prior weight temperature estimated without labels marginal likelihood HateMM HateClipSeg | MultiHateLoc（2512.10408）、HateClipSeg（2508.01712）、LELA（2602.09637）；无先验权重、无边际似然估计 |
| 2 | hateful video detection "temperature scaling" OR "calibration" multimodal LLM HateMM MultiHateClip 2025 2026 | LELA、2601.15115、SAGE（ACL 2026）、MultiHateLoc、SafeLens（AAAI demo，只是解码温度 0）；无校准方法 |
| 3 | weakly supervised hateful video temporal localization 2026 MultiHateLoc DeHate HateClipSeg new method prior VLM | 只有 MultiHateLoc（v3，2026-01） |
| 4 | hate speech video detection "empirical Bayes" OR "evidence maximization" OR "type-II maximum likelihood" OR "power prior" LLM | LELA、2601.15115、HVGuard（EMNLP 2025）、2508.04900；无命中 |
| 5 | hateful video detection label-free OR unsupervised fusion weight of classifier prior and LLM judgments, validation unlabeled, "hateful video" 2026 | RAMF（2512.02743，融合由标签监督学习）、SAGE、HVGuard、LELA、2601.15115；无无标签权重 |
| 6 | harmful OR hateful video moderation VLM yes/no answers probabilistic model backbone prior over-confidence tempering "prior" "likelihood" active querying | GuardReasoner-VL（2505.11049）、2505.16643、SafeLens（2605.17610）、HarmVideoBench（2606.27187）、2601.15115；无先验 / 似然模型 |
| 7 | hateful meme OR hate speech detection LLM "unsupervised calibration" OR "label-free calibration" OR "prior adaptation" zero-shot | 2402.12198、2311.06737、2506.08477（零样本 VLM 仇恨 meme）；均无校准或先验修正 |
| 8 | video anomaly detection VLM scores combined with learned detector prior weight estimated unlabeled held-out likelihood tempering | MuST-VAD（2608.06913）、SlowFastVAD（2504.10320，按熵选段）、LAVAD、CoReVAD、VERA；无按似然估计的权重 |
| 9 | power prior discounting parameter estimated by empirical Bayes marginal likelihood "adaptive power priors" Gravestock Held | Gravestock & Held 2017（power prior 的先验权重按最大边际似然估计）、2019（多个历史研究各自权重）；2204.06165、2404.02453（后续讨论） |
| 10 | "unsupervised temperature scaling" calibration deep networks without labels Mozafari | UTS（Mozafari et al. 2019，1905.00174 / 1911.11195）：无标签估计 softmax 温度，标题自称 post-processing |
| 11 | "Unsupervised calibration through prior adaptation" text classification large language models UCPA SUCPA | Estienne et al. 2023（2307.06713）：用无标签样本调整 LLM 输出的类别先验 |
| 12 | Saerens Latinne Decaestecker "adjusting the outputs of a classifier to new a priori probabilities" EM unlabeled; Alexandari "maximum likelihood with bias-corrected calibration" label shift | Saerens et al. 2002（EM 最大化新数据似然，重估类先验并调整分类器输出）；Alexandari et al. 2020（1901.06852） |
| 13 | generalized Bayes learning rate selection tempered posterior SafeBayes Grünwald "power likelihood" misspecification comparison of learning rate selection methods | Grünwald & van Ommen 2017（SafeBayes）、Wu & Martin（2012.11349，四种学习率选择法比较）、Holmes & Walker 2017、Lyddon et al. |
| 14 | Bayesian adaptive group testing OR active search misspecified prior from classifier, prior tempering robust, noisy tests, prior weight estimated from test outcomes | Cuturi et al. 2020（2004.12508，贝叶斯序贯分组检测，可带个体先验）、"Adaptive Group Testing with Mismatched Models"（ICASSP 2022）；无先验权重估计 |
| 15 | Bayesian experimental design expected information gain prior misspecification robust tempered prior empirical Bayes sequential design | Go & Isaac 2022（2205.09914，先验邻域上的稳健 EIG）、Barlas–Sloman–Kaski 2025（2511.07671，Gibbs EIG）、Abdulsamad et al. 2026（2603.14094）、Callahan & Catanach（PMLR v300） |
| 16 | crowdsourcing Dawid-Skene combined with classifier prior weight of classifier versus annotator answers estimated EM without ground truth "learning from crowds" Raykar | Raykar et al. 2010（分类器与标注者参数 EM 联合估计）、Rodrigues & Pereira（1709.01779）；没有"分类器先验的权重"这一参数 |
| 17 | zero-shot vision-language model calibration without labels test-time temperature estimation unlabeled data CLIP "label-free" temperature | FRAP（2606.06335，无标签测试数据上按与基模型预测的散度定温度）、C-TPT（ICLR 2024）、UL-TTA（2511.09101，logit 层温度与类先验）、SoC、TCA |
| 18 | "batch calibration" OR "contextual calibration" LLM in-context learning calibrate before use label-free prior correction Zhao 2021 Zhou 2023 | Zhao et al. 2021（contextual calibration）、Zhou et al. 2023（batch calibration，ICLR 2024） |
| 19 | piecewise training undirected models locally trained factors overconfident scaling weight unary potential CRF "piecewise" Sutton McCallum power on factors | Sutton & McCallum 2005、McCallum & Sutton 2004（因子分片训练后合并）；未见给因子加幂次的做法 |
| 20 | language model weight shallow fusion ASR estimated unsupervised without transcripts OR "internal language model estimation" prior over-counting log-linear fusion weight | ILME、density ratio、HAT（融合权重在开发集上调）；无无监督估计 |
| 21 | hateful video detection "Bayesian" OR "posterior" OR "prior" LLM judgments uncertainty fusion detector 2026 arXiv TANDEM RAMF HVGuard ImpliHateVid | RAMF、HVGuard、2601.15115、LELA、SAGE、SafeLens；无贝叶斯融合 |
| 22 | MacKay evidence framework hyperparameters marginal likelihood; Laplace marginal likelihood prior precision selection neural network Immer 2021 "type-II maximum likelihood" | MacKay 1992（evidence framework）、Immer et al. 2021、Antorán et al.（2206.08900）：用（近似）边际似然选先验精度 |
| 23 | data programming Snorkel generative label model accuracies estimated from unlabeled data marginal likelihood labeling functions Ratner | Ratner et al. 2016 / 2017：无标签数据上用最大似然估计各标注函数的准确率 |
| 24 | information pursuit vision-language model answers sequential queries posterior "20 questions" localization learned prior strength active testing Sznitman Jedynak | V-IP（Chattopadhyay et al. ICLR 2024）、Kolek et al. ICCV 2025、Conformal IP（2507.03279）、Tsiligkaridis et al.（1306.1922）；无先验强度参数 |
| 25 | SAGE ACL 2026 hateful video detection feature dilution expert gating arXiv | SAGE（2026.acl-long.817）、LEAF（2026.findings-acl.604）；SAGE 无 arXiv 版 |
| 26 | hateful video detection overconfident classifier corrected with LLM judgments on unlabeled videos test-time adaptation OR "unlabeled" OR "label-free" weighting 2026 | SCANNER（2602.00132，仇恨视频检测的无标签 test-time adaptation）、RADAR（2601.11981，假新闻视频）、IARE（2606.11953） |
| 27 | informative group testing risk scores prior probabilities misspecified robustness "informative Dorfman" OR "prior misspecification" Bayesian group testing halving algorithm | Aprahamian–Bish–Bish（风险估计有误差时的稳健 Dorfman 方案）、2205.11838（先验失配越大性能越差）；无从检测结果估计先验权重 |
| 28 | "cold posterior" OR "tempered prior" Bayesian neural network prior tempering misspecified prior Wenzel 2020 Aitchison statistical theory of cold posteriors | Wenzel et al. 2020（2002.02405）、"Bayesian Neural Network Priors Revisited"（2102.06571）、"The fine print on tempered posteriors"（2309.05292） |
| 29 | weakly supervised temporal localization VLM OR LLM answers as noisy observations latent variable model EM marginal likelihood unlabeled videos reliability estimated without labels | GEM-TFL（伪造定位的 EM）、MLLM4WTAL、EtC（2312.02483）、PVLR（2408.05955）；无命中 |
| 30 | estimate reliability weight of a pretrained classifier prior against LLM annotations without ground truth "marginal likelihood" fusion classifier and LLM labels unsupervised aggregation | 2504.15432、LARA（2411.06877，用少量人工标注校准）等；无命中 |
| 31 | Gravestock Held 2017 "Adaptive power priors with empirical Bayes for clinical trials" Pharmaceutical Statistics abstract marginal likelihood prior weight | 确认：Pharmaceutical Statistics 2017（DOI 10.1002/pst.1814），摘要写明先验权重按边际似然估计 |

全文 / 摘要核对（WebFetch）：
- LELA（HTML 全文）：逐帧分数跨模态取 max，阈值固定 .5；没有训练出的先验，没有权重、温度、校准，没有概率模型。
- 2601.15115（Yang, Zhang, Fu；HTML v1，PDF 未能解码）：四阶段 VLM 推理，最终标签由 VLM 综合给出，自报置信度只作解释；无校准、无先验、无与训练分类器的加权。
- SAGE、LEAF（ACL Anthology 摘要页；SAGE 的 PDF 未能解码，只核对到摘要级）：监督训练的检测方法，摘要无校准 / 先验权重 / VLM 回答的概率模型。
- SCANNER（HTML 全文）：视频级检测；在无标签目标视频上用熵、聚类对齐、多样性三项损失对归一化层与线性层做梯度更新；不用 LLM / VLM 回答；softmax 温度是手设超参数；没有用似然估计的标量。
- MuST-VAD、FRAP、UTS、Gravestock & Held（arXiv 1708.08239）：摘要级。MuST-VAD 摘要只描述检测器与 LVLM 的交替训练，未提推理期融合权重。
- Barlas–Sloman–Kaski 2025（HTML 前 10 万字符）：似然加幂（学习率 ω），加幂后的后验进入 EIG；正文写明还没有适用于实验设计场景的学习率选择方法。

结论：hateful video 检测 / 定位文献里没有"先验（骨干）权重"这一对象，更没有用留出视频的 VLM / LLM 回答无标签估计它的做法；最接近的 SCANNER 是无标签的梯度适应，对象、目标函数、任务都不同。任务外最近的先例：(1) 估计方式相同——Gravestock & Held 2017（power prior 的先验权重按最大边际似然估计），更一般的是 MacKay 的 evidence framework；(2) 函数形式相同——temperature scaling 及其无标签版本 UTS（作用在最终 softmax 上，自称 post-processing）；(3) 加幂进入 EIG——Barlas–Sloman–Kaski 2025（加幂的是似然，幂次不估计）。未检索到"在序贯提问模型里用留出观测的边际似然估计网络先验的权重，并同时用于后验和提问"的工作。

## 3. 四种 STOP 情形

1. **来源方法已用于 hateful video**：否。第 2 节 1–7、21、25、26 条检索与 LELA / 2601.15115 / SAGE / LEAF / SCANNER 核对均无命中。
2. **纯 training/test ensemble**：否。单一网络、单一 VLM、单一答案表，没有新增模型；c 是同一个联合分布里先验因子的指数，不是两个独立预测的加权平均。网络先验与 VLM 回答在一个概率模型内合并，这一结构自第 2 版起已过审，本提案没有改变它的性质。
3. **纯 calibration / 后处理 / 平滑**：否，但这是最接近成立的一条，理由和保留意见分开写。
   - 不成立的依据（全部来自代码）：
     (a) c 乘在推断的输入（先验势）上，不在输出分数上（`cpolicy.py:287–290`）；分数由 `ctree.marginals` 直接输出（`:341`），之后没有任何映射。
     (b) c 改变提问：同一次推断给出的节点边缘决定 EIG 的 argmax（`:387–417`），从第 1 问起被问的节点就随 c 变，test 上实际读取的回答与第 6 版不同。后处理的含义是上游计算不变、只改分数；去掉 c 会改变问了哪些节点，不满足这个含义。
     (c) c 不是输出的单调变换：P(G=1|x) 不随 c 变（`ctree.py:142`），c 改的是位置先验相对回答似然的权重，视频内、视频间的排序都会变。对最终分数做温度缩放不改变 ROC / AP。
     (d) 估计不用标签，目标函数是模型自己的边际似然（`prior_weight.py:123–131`），与训练时学 chain 参数用的是同一个似然，只是换成网络没见过的视频；不是用标签把分数拟合到频率的映射。
     (e) c 参与 checkpoint 选择（`train.py:444–462`）。
     (f) 不是时间上的平滑：c 逐秒缩放 logit，不在相邻秒之间做平均。
   - 保留意见（必须如实记录）：
     (g) c 不进训练损失。第 6 版审稿第 3 条的第一个理由（"进训练损失，训练出的网络不同"）在本提案不成立；同一组超参数下网络的梯度轨迹与第 6 版相同。本条结论只靠 (a)–(e)，其中 (b) 是决定性的一条。
     (h) 形式上，c 是一个已训练子网络 logit 上的温度，在留出集上估计、test 时固定，与 temperature scaling 同形，最近的是 UTS。0 次调用时第 7 版与第 6 版的差别只有这个温度。论文审稿人很可能用"校准"来描述它，措辞要求见 5.4。
   - 判断：规则 4 要求"纯" calibration / 后处理才 STOP。本候选是模型内一个参数（先验强度）的无标签估计，同时进入后验、提问、停止统计和 checkpoint 选择，不是只作用于分数的校准步骤；用户 11-10 对规则 3 的裁定与此读法一致。不 STOP。
4. **纯工程技巧**：否。c 不是在指标上搜索出来的超参数：没有搜索空间，不看 AP / ROC，由最大边际似然在每个 epoch 自动估计；不换特征、不加增强、不改训练配置。提案包含机制（先验在训练视频的回答上拟合，在未见视频上预测回答明显变差）、估计方法、作用位置和预注册判定。若 c 改为由 Optuna 或网格按 validation / test 指标选，就会落入这一条，也会碰规则 13。改动小不是 STOP 条件（同第 6 版审稿）。

## 4. 规则 13

`search.py` 的 `EXTRA_DEFAULTS` 三语料相同；`src/qtl/prior_weight.py`、`src/qtl/cpolicy.py` 没有任何按语料分支的代码；c 由同一程序在各语料自己的 validation 回答上估计，性质与答案表参数、chain 参数相同（估计出来的参数，不是按语料手设的值），README 第 2 节的中位数 .16 / .20 / .23 也说明三语料落在同一范围。规则 13 列举的"按语料的后处理或校准"指按语料手选或流程不同的做法，这里是一套流程。搜索空间仍是第 5 / 6 版的四个标量。通过。前提：三语料都用 `loc` 模式，不得事后按语料固定 c 或换模式。

## 5. 非阻断备注

### 5.1 开跑前备注（初稿提出 5 条；README 第 8 节 "Fixed before the search" 已全部写入）

1. **消融的定义写死（规则 14(g)）。** "同一 checkpoint、c = 1"只去掉了推理时的权重，checkpoint 仍是在 c 下选的；另一半（选择也在 c = 1 下）就是第 6 版自己的搜索。两行都要报。再加一个 2×2 分解：提问用 c 或 1 × 后验用 c 或 1，现有诊断参数已支持，只需重放：`run_batch(prior_weight=(1, 0, c), policy_prior_scale=1/c)` = 提问按第 6 版先验、后验按 c；`run_batch(prior_weight=None, policy_prior_scale=c)` = 提问按 c、后验按第 6 版先验。3.3 的 (b) 是本审稿放行的决定性依据，论文要有这组数字才能写"权重改变了问什么"。另外，Stage A 的配对差在 HateMM 是 AP +.002 / ROC +.004，低于规则 14(g) 的 .01，README 应事先写明不满足 14(g) 时怎么处理（这一点不是判定依据）。
   README 现状：已写入三行消融 (i)(ii)(iii)；已写明不满足 14(g) 时 c 留在方法里但不作为 novelty 主张。
2. **c 没有约束，且已出现过负值。** `fit` 里 c 初值 1、无上下界、不截断；README 第 2 节更正后记录了一个负的估计值（HCS，-.002）。负的 c 会把网络的位置先验反过来用。处理规则要在搜索前写明，之后不能再补。
   README 现状：已写明"按估计值使用，不设界、不截断（负值或大于 1 也照用）"。
3. **DeHate 单 trial 耗时贴近规则 7 的 1 小时线。** 第 6 版 DeHate seed 234 的 20 个 trial 耗时中位数 3334 秒、最大 3407 秒（`runs/20261101_r6_length_table/dehate/seed234/study_summary.json`，三个 seed 共用一块 GPU）；Stage A 里 559 个 validation 视频的一次 c 估计约 3 秒（`runs/20261110_prior_weight/stage_a_dehate.log`，CPU），50 个 epoch 约多 150 秒。初始启动脚本预写 20 个 trial。
   README 现状：trial 数按每个 seed 首个 trial 的实测耗时由 `src/qtl/search.py` 写入（不超过 3600 秒为 20，否则 5），实测秒数归档，看到耗时后不再改；DeHate 每块 GPU 最多两个 seed。启动脚本不再预写 `budget.json`。
4. **前 10 个 trial 的超参数与第 6 版相同。** TPE 的起始 trial 是按 sampler seed 的随机抽样，与目标值无关；本审稿核对了第 5 版与第 6 版三个语料 seed 234 的 trial 0–9 超参数逐项相同、trial 10 起分开（两次搜索的 `trial<k>/hparams.json`）。第 7 版搜索空间、seed 相同，损失不变，所以 trial 0–9 重训的是第 6 版的同一批网络，只是 epoch 选择和 c 不同。不违反规则 7，但搜索结果与 Stage A 的全 trial 重放高度相关。
   README 现状：已写明这一点，并写明搜索不是独立于第 4 节的证据。
5. **"另外报告"清单补三项**（都只是重放或一次拟合）：c 在训练回答上的估计值（样本内，预期接近 1，是"先验在未见视频上变差"这一机制的直接证据）；手设 c = .5 与用 validation 标签拟合的 c 两个参照（只作诊断，不是方法）；每个 trial 的边际似然改善量（`nll_all_before / after`）。
   README 现状：三项均已列入。

开跑前没有遗留的未处理备注。

### 5.2 论文须引用并划清的工作

- 估计方式：MacKay 1992（evidence framework）；power prior 及其经验贝叶斯权重（Ibrahim & Chen 2000；Gravestock & Held 2017、2019）。Ibrahim & Chen 2000 与 Guo et al. 2017 本次未检索原文，引用前核对。
- 校准：Guo et al. 2017（temperature scaling，需标签、作用在最终输出）；Mozafari et al. 2019（UTS，无标签、作用在最终 softmax）；先验漂移的 EM（Saerens et al. 2002；Alexandari et al. 2020）；LLM 的无标签校准（Zhao et al. 2021；Zhou et al. 2023；Estienne et al. 2023）；无标签 test-time 温度（FRAP、UL-TTA、C-TPT）。区别要写清：这些都作用在最终输出上，不改变观测的获取。
- 加幂 / 模型失配：Grünwald & van Ommen 2017、Holmes & Walker 2017、Wu & Martin；Wenzel et al. 2020。
- 实验设计与分组检测里的先验失配：Barlas–Sloman–Kaski 2025（加幂进 EIG，幂次不估计）、Go & Isaac 2022、Cuturi et al. 2020、2205.11838、"Adaptive Group Testing with Mismatched Models"。
- 无标签估计观测源可靠性：Ratner et al. 2016 / 2017；Raykar et al. 2010。
- 任务内：SCANNER（无标签适应，对象不同）、LELA（固定阈值，无先验）。

### 5.3 消融须包含

- 5.1 第 1 条的三行（同 checkpoint c = 1；第 6 版搜索；2×2 分解）。
- c 的来源：validation 无标签估计（方法）对比手设常数、validation 带标签拟合、训练集样本内估计。
- 固定 0–32 次曲线下的 c = 1 对照（README 已列固定次数；0 次那一列只反映温度本身，见 5.4）。

### 5.4 措辞风险

- 方法名和正文不要写成 calibration / temperature scaling，但必须主动说明它与二者同形并给出区别（3.3 的 (a)–(c)）；不说明会被审稿人直接归为留出集校准。主张应限定为：用未见视频的回答估计训练出的位置先验该占多大权重，这个权重同时决定后验和问什么。估计量本身是标准的经验贝叶斯，不能写成新的估计方法。
- "无标签"只指 c 的估计。checkpoint 选择仍用 validation 逐秒标注，答案表仍用训练视频标签；不要写成"validation 不用标签"。DeHate 的 validation 视频集合本身按标注筛过（去掉没有可用片段的仇恨视频，改动后的 README 已写明），论文要照样写明。
- 训练与推理用的模型不同（训练损失里 c = 1，推理用估计值），论文要明说这是两步估计：网络参数由训练似然定，先验强度由留出视频的边际似然定。
- 视频级 logit g 的过度自信（校准系数 .24–.29）本提案没有处理（(a, b) 无法无标签识别，README 第 2 节）；不要写成"解决了先验过度自信"，只能说位置先验。
- "test 时不增加 VLM 调用"成立，但估计 c 需要 validation 视频整棵树的缓存回答（每视频几十个），这是离线成本，要写明。
- 0 次调用时第 7 版与第 6 版只差温度，这一列的差异不能当作提问策略的证据。
- 按规则 14(e) 汇报时，直接引用用户 2026-11-10 的裁定原文和日期。
