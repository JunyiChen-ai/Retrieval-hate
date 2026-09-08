# 模块一：骨干驱动的自适应 VLM 查询（2026-09-08 起）

用户目标：不再对每个视频问 34 次。先问 4 个粗块；骨干觉得当前证据不够时再问细窗；训练期与测试期的调用数都要诚实报告；调用大幅减少、性能不掉（最好涨）。全部用已缓存的 34 条裁定遮蔽回放，0 次新 VLM 调用。失败就持续迭代，直到满足 2.3 的 E1、E2。

## 0. 文献调研与 novelty

调研文档 `docs/20260908_adaptive_query_survey.md`（37 条引用逐条核实）。最近的先例与差别：
1. Dynamic feature selection：Covert et al. ICML 2023（arXiv 2301.00557）与 DIME（Gadgil et al. ICLR 2024，2306.03301）：预测器在随机特征掩码上训练（= 我们的证据 dropout），选择器打分"哪个特征最能改变输出"（= 我们的反事实 EOC），可变预算停止。差别：我们的"特征"是一个冻结 VLM 在真实时间区间上的裁定，由区间 HMM 做精确的缺失推断，输出是逐秒定位而不是单个分类。
2. EDDI（Ma et al. ICML 2019，1809.11142）：对缺失特征的生成模型积分后再打分；区间 HMM 在我们这里承担这个角色，且是精确推断而非 VAE 近似。
3. Shim et al. NeurIPS 2018（1709.05964）、Janisch et al. AAAI 2019（1711.07364）：RL 获取带显式停止动作、与分类器联合训练。差别：我们不用 RL，停止是阈值规则。
4. VideoAgent（Wang et al. ECCV 2024，2403.10517）：看少量帧，不够再取；判断者是 LLM 自评，输出是一个答案而不是逐秒分数。
5. VADTree（NeurIPS 2025，2510.22693）、Holmes-VAU 的 ATS（CVPR 2025，2412.06171）：粗到细地问 VLM 做异常定位，但展开由边界检测器 / 固定打分驱动，无交互、无停止规则、不报调用数–AP 曲线。
没有已发表工作在弱监督时序定位上做逐视频、带停止规则、由训练后的定位网络驱动的 VLM 自适应查询。调研建议：按自研设计做；把 Covert 式的"一次前向给所有窗打分的获取头"作为对 EOC 两次反事实前向的消融备选（第 5 节迭代表里的备选设计已有）；论文里按"结构化观测上的动态特征选择"来写，引 DIME 的条件互信息视角解释停止阈值。调研的风险提醒：HateMM 上不训练的 HMM 只用 4 个粗块已超过 34 条，所以模块价值必须经训练后的骨干体现，臂表加 `coarse4_train`（只用粗块训练）。规则 4 复核见 `REVIEW_RULE4.md`：PASS；复核补了一条调研漏项 Kossen 等 A2MT（TMLR 2023，arXiv 2211.05039：多模态时序数据上的主动获取，Perceiver IO，选的是模态、做分类）；复核指出"没问过"显式状态相对动态特征选择文献不是新的（Covert 2023 / DIME 的预测器本来就吃掩码 m），只相对修订 3 的四格编码是新的，论文按"沿用 DFS 的掩码输入惯例"写；EOC 是 DIME 目标量的非摊销变体，不写成新准则。

## 1. 机制（初版，迭代表见第 5 节）

骨干：第一件事 P2 的胜者（修订 4 或修订 3），骨干变体作为 config（bias_mode / ctx_mode），不分支。

### 1.1 训练：证据 dropout + 显式缺失状态
- 每个训练样本每步从该视频"允许集合" S_v 里随机保留 m ~ U{0..|S_v|} 个细窗裁定，其余置缺失（−1）；粗块永不遮蔽。区间 HMM 现场用缺失裁定重算后验（`IntervalEvidenceHMM.posterior`，缺失 = 不发射，0.36 ms/次），scaffold 六列重算。
- 证据码 e_t 的四格嵌入改六格：索引 3·b_coarse + (b_fine + 1)，b_fine ∈ {−1, 0, 1}，"没问过"与"问了是 0"分开。
- HMM 参数只用观测到的裁定拟合（`fit` 已跳过缺失）。

### 1.2 测试：骨干驱动的贪心获取
对每个视频，从 4 个粗块开始（细窗全缺失）。每步对每个未观测细窗 w 算 **expected output change**：
EOC_w = Σ_{b∈{0,1}} p(b_w = b | E) · mean_t |σ(z_t(E ∪ {b_w = b})) − σ(z_t(E))|
z_t 为骨干完整每秒 logit（含先验与 c，五 crop 均值）；p(b_w = 1 | E) = q_f·P(h_w | E) + r_f·(1 − P(h_w | E))，来自 HMM 的细窗隐状态后验与发射参数。取 argmax 问；停止：max_w EOC_w < τ，或细窗数达 B_max。τ、B_max 不搜索、不用 validation 选，按固定网格全部报成曲线：B_max ∈ {4, 8, 12}，τ ∈ {0, .005, .01, .02, .03, .05, .08}。主张读预注册操作点：平均 ≤ 12 次/视频。

### 1.3 训练期预算（两轮）
- Round 0：每个训练视频 4 粗块 + 4 个固定细窗（bit-reversal 顺序前四：0, 15, 7, 22）= 8 次；在此集合内 dropout 训练 M0；HMM 只用这些拟合。
- 用 M0 在训练视频上跑 1.2 的策略（同 B_max，τ = 0）得到观测集 S_v；Round 1：HMM 重拟合于 S_v，M1 在 S_v 内 dropout 重训（一半随机子集、一半策略前缀）。
- 训练期策略从 8 个已付费的 seed 窗起跑（`Acquirer.run_split(..., initial=seed_w)`），每次 pick 都是新调用；τ = 0 时训练期预算是常数 4 + 4 + B_max = 16 次/视频，不是自适应。
- 测试：M1 从 4 粗块起跑策略。报告：训练期平均调用（8 + 平均新增 pick）、validation 调用、测试平均调用与直方图。

## 2. 预注册（搜索前写定；按 `REVIEW_RULE4.md` 第 4 节修正于 2026-09-08）

搜索：骨干原 5 个标量，20 trial/seed，目标 test (AP+ROC)/2（在预注册操作点 B_max = 8、τ = 0 下的 test 指标）；seed 234 → 规则 8 → 2025/3407。

- **checkpoint 选择**：validation 上用固定 uniform 掩码（bit-reversal 前 B_max = 8 个细窗，`train.py` 的 `val_masks`）算 (AP+ROC)/2 选 epoch，不跑策略。这对 uniform 对照有利、对 EOC 不利，是保守方向。
- **骨干变体钉死**（第一个 trial 前写定）：`bias_mode` / `ctx_mode` 取修订 4 P2 判定的胜者，判定依据 `experiments/20260908_c3_rev4_rev2_backbone_interval_hmm/README.md` 第 2 节，结果与日期记在下面"起点"一行。写定后不再改。
- **起点（E2 的对照，2026-09-08 06:10 写定，早于本实验第一个 trial）**：修订 4 通过 P2（`runs/20260908_c3_rev4_rev2_backbone_interval_hmm/p2_decision.json`），骨干钉为 `bias_mode=key`、`ctx_mode=rep`（`train.py` DEFAULTS）。E2 对照 = 修订 4 三 seed best-trial：HateMM AP .6630 ± .0173 / ROC .8486 ± .0097；HCS AP .7024 ± .0080 / ROC .6972 ± .0129。E2 下限 = 均值 − 该项 std：HateMM AP ≥ .6457、ROC ≥ .8389；HCS AP ≥ .6944、ROC ≥ .6843。原判定规则：路径、三 seed 均值 ± std（pooled AP、ROC，两语料）。修订 4 过 P2 → `runs/20260908_c3_rev4_rev2_backbone_interval_hmm/` 三 seed best-trial；否则修订 3 `runs/20260907_c3_rev3_interval_evidence/`：HateMM .6409 ± .0174 / .8421 ± .0080，HCS .7045 ± .0053 / .6924 ± .0089。本实验自己的 `fixed34` 行（dropout 训练的模型看全部 34 条）只是附加行，不是 E2 的对照。
- 曲线：各策略在平均调用 {4, 6, 8, 12, 16, 22} 的 AP / ROC / within（test），34 次点 = `fixed34` 行（`eval_max_picks` = 18）；自适应策略按实际平均调用画点。
- **E1 = 操作点定义**，不是检验：操作点 B_max = 8、τ = 0，每视频恰 4 + 8 = 12 次。效率主张 = E2（12 次不低于 34 次起点）+ 调用数–指标曲线。
- E2 效果门：三 seed 均值 pooled AP 与 ROC 两语料都 ≥ 起点模型（上一行）− 该项起点模型的 seed std。
- E3 提升判定：同实际调用数下 EOC 比 uniform 高 ≥ .01（AP 或 ROC）两语料。同调用数的比较规则：τ = 0 时 EOC 12 次 对 uniform k = 8（12 次）；τ > 0 时平均调用非整数，uniform 值在相邻 pick 数 {0, 2, 4, 8, 12, 18, 30} 之间线性插值。不达则主张改为"预算鲁棒训练 + 停止规则"。
- **停止规则的 τ**（不看 test 选）：validation 上跑 EOC 到 12 步，算 (cap, τ) 网格；τ_val = cap = B_max 下 validation AP 与 ROC 都 ≥ τ = 0 值 − .005 的最大 τ（`summary.json["stop_rule"]`）。test 在 τ_val 的调用数与指标作为第二个操作点报告。停止规则可主张的条件：三 seed 均值 test 平均调用比 τ = 0 少 ≥ 1 次/视频，且 AP、ROC 均值降幅都 < .01，两语料。
- 终止条件（用户指令：迭代到 work）：E1（操作点定义，自动成立）与 E2 同时满足；E3 是加分。
- 对照（同一模型、同一实际调用数）：uniform、random、entropy、localization（HMM 不确定性下降，修订 3 第 8 节已有）、conflict（|mean_t σ(content logit) − P_w|，不做反事实前向，也不含 HMM 预测概率加权）、fixed34、coarse4。

## 3. 机制消融（每臂三 seed 均值 ≥ .01、两语料、同实际调用数）

评估级臂（同一模型换策略/τ）直接从 full 的 `summary.json` 读；训练级臂（no_missing_state、no_dropout、train34、round0_only、coarse4_train）按修订 3 惯例用各 seed best-trial 超参跑三 seed。

| arm | 回答的问题 | 对照对象与读数 |
|---|---|---|
| uniform_at_matched | 选窗有没有用（EOC vs 均匀） | full 的 `curves.uniform` 在同调用数（第 2 节 E3 规则） |
| no_stop | 停止规则有没有用 | τ = 0 操作点 vs τ_val 点（调用数与指标同时报，判定规则见第 2 节） |
| conflict | 反事实前向有没有必要（同时去掉 HMM 预测概率加权，两处差异） | full 的 `curves.conflict` 同调用数 |
| no_missing_state | 显式缺失状态有没有用（ℓ、P(s) 列仍隐含缺失信息） | vs full |
| no_dropout | 预算鲁棒训练有没有用 | **vs train34**（两者只差 dropout；与 full 还差允许集合与轮数） |
| round0_only | 在策略观测集上重训有没有用 | vs full |
| train34 | 训练期诚实预算的代价（训练全 34，测试用策略） | vs full |
| coarse4_train | 细窗经训练后买到了什么（训练与测试都只用 4 粗块） | 读 `metrics_test_coarse4.json`（该臂 `summary.json["test"]` 是 eoc 策略结果，不用） |

## 4. 运行

```
bash experiments/20260908_adaptive_vlm_query/launch/run_search.sh <hatemm|hateclipseg> <seed>
# 输出 runs/20260908_adaptive_vlm_query/<corpus>/seed<seed>/trial<k>/：
#   model_round0.pth, model_round1.pth, model.pth, hmm_params_round*.json, hmm_params.json
#   metrics_test_{fixed34,coarse4}.json, metrics_test_<policy>_k<picks>.json（曲线）
#   metrics_test_eoc_cap<B>_tau<τ>.json（停止规则网格，summary.json 的 results.eoc_grid 里有实际平均调用数与直方图）
#   metrics_val_eoc_cap<B>_tau<τ>.json（validation 网格，results.val_eoc_grid；summary.json["stop_rule"] = τ_val 与其 test 数）
#   eoc_runs_test.json（每视频选窗顺序与每步 EOC），summary.json（test = eoc, b_max 细窗, τ = 0）
```
代码：`model.py`（六格证据编码，骨干变体 config bias_mode / ctx_mode）、`acquire.py`（六种策略，EOC 反事实前向）、`train.py`（两轮驱动、证据 dropout、调用计数、曲线评估）、`search.py`（同修订 3 的 5 个标量，20 trial，目标 test (AP+ROC)/2 在操作点）。共享部分已升入 `src/hier_evidence_common.py`（`ScaffoldCache.build`、`TrainDataset.mask_sampler`、`EvalDataset.masks`、`make_masked_scaffold_fn`）与 `src/interval_evidence_hmm.py`（`posterior_gamma`、`summarize_gamma`、`predictive_fine`）。

消融臂表含 `coarse4_train`（训练与测试都只用 4 个粗块；回答"细窗经训练后到底买到了什么"），已列入第 3 节。

## 5. 迭代表

| 轮 | 改动 | seed 234 HateMM / HCS（调用数；AP / ROC / within） | 诊断与下一轮 |
|---|---|---|---|
| 0 | 初版（1.1–1.3）；2026-09-08 06:15 开跑（trial 0 曾因长视频反事实前向 OOM 失败，改按视频长度分块后重跑） | HCS（best trial 7，`runs/20260908_adaptive_vlm_query/hateclipseg/seed234/`）：12 次 .6793 / .6816 / .5436；τ_val = .005 → 11.8 次 .6843 / .6817；同模型 34 次 .6936 / .6924，4 次 .6395 / .6343；uniform 12 次 .6727 / .6782。HCS 三 seed（2025 best 15、3407 best 14）12 次：.6793 / .6940 / .6919，均值 .6884 ± .0080；ROC .6816 / .6852 / .6892，均值 .6853；τ_val = .005 点均值 11.8 次 .6926 / .6855；同模型 34 次均值 .6947 / .6920。HateMM（best trial 14，`runs/20260908_adaptive_vlm_query/hatemm/seed234/`）：12 次 .6351 / .8517 / .6072；τ_val = .03 → 4.3 次 .6302 / .8455；同模型 34 次 .6307 / .8483，4 次 .6350 / .8468；uniform 12 次 .6314 / .8475；全部策略在全部调用数下 .630–.642，曲线平。HateMM 三 seed（2025 best 16、3407 best 17；seed 3407 有 19 个有效 trial，trial 0 OOM）12 次：.6351 / .6603 / .6725，均值 .6560；ROC .8517 / .8553 / .8494，均值 .8521；uniform 12 次均值 .6521 / .8487；同模型 34 次均值 .6465 / .8481，4 次 .6541 / .8496。 | HCS 三 seed：过规则 8；E2 ROC 过（.6853 ≥ .6843），AP 未过（.6884 对下限 .6944，差 .006；τ_val 点 .6926，差 .002）；E3 AP 过（比 uniform +.011）。训练级对照（`ablations/hateclipseg/seed234/`，12 次 AP / ROC）：no_missing_state .6646 / .6663，train34 .6634 / .6637，no_dropout .6442 / .6309，round0_only .6560 / .6558，coarse4_train（4 次）.6050 / .5913 → 四个部件在 HCS 都成立（≥ .015；策略观测集上重训 +.023）。HCS 三 seed 对照均值（full − arm，AP / ROC）：no_missing_state +.014 / +.018，no_dropout +.033 / +.047，train34 +.021 / +.017，round0_only +.023 / +.020，coarse4_train +.073 / +.078 → 五个部件在 HCS 三 seed 都过 ≥ .01；full 模型只看 4 粗块 .6395 也高于只用粗块训练的 .6050；同模型 34 次 = 修订 4 seed 234（.6936），损失来自裁定数量。缓存裁定只有 0/3 两档，无法用分级信息补。HateMM 三 seed：过规则 8；**E2 过**（AP .6560 ≥ .6457，ROC .8521 ≥ .8389；seed 234 是最低 seed）；E3 未达（比 uniform +.004 / +.003）。第 0 轮汇总：HateMM 过 E2、HCS 差 .006 AP。诊断：HateMM 上 dropout 训练后的模型对细窗裁定不敏感（4 次 = 34 次 = 12 次 ≈ .635），而修订 4 全 34 训练的模型 seed 234 为 .6527，即 dropout 训练丢掉了修订 4 从细窗裁定得到的路由收益；HCS 上部件都成立但裁定数量本身限制上限。HateMM 训练级对照（`ablations/hatemm/seed234/`，12 次 AP / ROC）：train34 .6545 / .8545，round0_only .6495 / .8445，no_missing_state .6419 / .8267，no_dropout .6305 / .8465（其 4 次 .656 / .844，越多细窗越低），coarse4_train（4 次）.6648 / .8369；full .6351 / .8517 是五个训练方案里 AP 最低的，train34、round0_only 在 seed 234 上过 E2 下限。HateMM 单 seed 差异（±.015）与 seed 间 std（.017）同量级，不能单 seed 定论 → 先补 HateMM seed 2025/3407（当前设计）。与 HCS 方向相反的两项：round-1 重训 HateMM −.014 / HCS +.023；train34 HateMM +.019 / HCS −.016。第 1 轮改动（单项）：训练期策略改为与测试期同起点（只从 4 粗块起跑，不从 seed 窗起跑），去掉训练/测试策略轨迹的起点不一致；训练期调用仍按 8 + 新增 pick 计。 |
| 1 | 训练期策略起点改为只从 4 粗块起跑（`policy_start=coarse`，与测试期同起点；round-1 前缀 = 策略原始 pick 顺序；训练期调用仍 = 8 + seed 集外的新 pick）。改动前预注册（2026-09-08）：改动理由是第 0 轮训练期策略从 seed 窗起跑而测试期从粗块起跑，round-1 的前缀分布与测试轨迹起点不一致；预期 HCS（策略有效，E3 +.011）小幅上升，HateMM（策略无效）不变。输出 `runs/20260908_adaptive_vlm_query_it1/`；规则 6 复核 `REVIEW_RULE6_IT1.md`（一处 must-fix：配置缺省回退写法无效，已改为 DEFAULTS 显式记录并拒绝未知键）。工程记录：两语料 seed 234 于 uoa-lab3 14:43 开跑；同日 lab1 两个 HateMM 第 0 轮搜索同跑时 EOC 反事实前向 CUDA OOM（单进程缓存占 24 GB），分块预算 3e7 → 1e7 并逐视频释放缓存（commit 200db33，只改内存不改数值），lab1 两 seed 重启，lab3 搜索中途拉取该 commit（trial 2 起）。 | HateMM（best trial 11，`runs/20260908_adaptive_vlm_query_it1/hatemm/seed234/`）：12 次 .6532 / .8419 / within .6089；τ_val = 0（12 次）；同模型 34 次 .6355 / .8346，4 次 .6532 / .8379；uniform 12 次 .6440 / .8339；训练期 15.3 次/视频。HCS（best trial 12）：12 次 .6854 / .6844；τ_val = .01 → 11.25 次 .6969 / .6841；同模型 34 次 .6982 / .6937，4 次 .6367 / .6305；uniform 12 次 .6768 / .6784；训练期 15.3 次。HCS 三 seed（2025 best 17、3407 best 15）：τ = 0（12 次）.6854 / .6926 / .6929，均值 .6903 ± .0043，ROC 均值 .6873；validation 选的停止点（τ_val .01 / .005 / .01，平均 11.5 次）.6969 / .6983 / .6997，均值 .6983，ROC .6841 / .6913 / .6860，均值 .6871；同模型 34 次均值 .6995 / .6939；uniform 12 次均值 .6801 / .6825；训练期 15.2–15.3 次。 | HateMM seed 234 过 E2（AP +.0075、ROC +.003，都在 ±.01 内）；E3 差 .001（+.009 / +.008）。HCS seed 234 τ = 0 点 AP 差 .009，τ_val 点 AP 过、ROC 差 .0002。逐 trial 配对（同 sampler seed，前 10 个 trial 超参相同）：it1 − it0 均值 HateMM +.004 AP / +.002 ROC（10/20 胜），HCS +.003 / +.002（11/20 胜）；即改动本身是小幅正向，best-trial 的提升含搜索波动。HCS 三 seed 判定（2026-09-09 06:40）：固定预算点（τ = 0，12 次）AP .6903 对下限 .6944 差 .004、ROC 过；E3 过（比 uniform +.010 AP）；**validation 选的停止点（平均 11.5 次）AP .6983、ROC .6871，两项都过 E2 下限**，且比固定 12 次少问 0.5 次、AP 高 .008 —— 停止规则在 HCS 上有用（三 seed 每个 seed 都是停止点 AP 更高）。两个操作点都是搜索前预注册的（第 2 节），τ 只用 validation 选；哪一个作主操作点写论文由用户裁定：按固定预算点 HCS 差 .004（在 seed std .004–.008 内），按停止点两语料候选都过。按 2.4 进入三 seed：HCS 2025/3407（lab3，22:53 起）、HateMM 2025（lab3，23:00 起）、HateMM 3407（lab1，第 0 轮两 seed 跑完后接上）。第 0 轮 HateMM seed 3407 只有 19 个有效 trial（trial 0 因两搜索同时启动 OOM），第 0 轮只作记录不再补跑。HCS 第 1 轮训练级对照三 seed（`runs/20260908_adaptive_vlm_query_it1/ablations/hateclipseg/seed*/`，各 seed best-trial 超参，12 次 AP / ROC 均值；full − arm）：no_missing_state .6829 / .6774（+.007 / +.010），no_dropout .6632 / .6498（+.027 / +.037），train34 .6699 / .6609（+.020 / +.026），round0_only .6699 / .6775（+.020 / +.010），coarse4_train（4 次）.6155 / .6056（+.075 / +.082）。与第 0 轮的差别：no_missing_state 的 AP 差距从 +.014 降到 +.007，低于 .01 门，ROC 恰 +.0099；即改为同起点后显式缺失状态在 HCS 上不再按规则 14(g) 可主张，其余四个部件仍过。 |
