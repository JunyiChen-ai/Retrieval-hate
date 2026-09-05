# Research Iteration Rules

**2026-09-02 重写。** 本文件取代此前全部 22 条规则、process-review/RESET 机制、失败计数、premise 门、gain budget、anchor-compatible 要求与 mechanism 门。旧版本见 git 历史与 `docs/PROCESS_RULES_REWRITE_2026-09-02.md`（重写原因与数据）。

## 目标

做出一个弱监督 hateful video localization 方法，同时满足：

- **SOTA**：按第 8 条定义，在 HateMM 和 HateClipSeg 的 test 上，pooled frame AP 与 ROC 超过固定 baseline 表；within ROC 只报告（2026-09-06 起不作门）。
- **Novel**：按第 4 条定义。迁移自其他任务、没在 hateful video detection/localization 用过、且有效的方法即算 novel。

## 前一版流程的教训（只记结论）

- 晋级门要求两语料六项指标全部超过"各列最高值"，其中 HateClipSeg within 门取自 VERA 的 0/1 分数加平滑（.562），高于同特征全监督线性分类器上限（.560）；HateMM within 门等于 starting point 自身三 seed 均值。任何方法都过不了，14 个正式方法全部"失败"，其中 marked temporal splat 在 HateMM within 达 .728（超过 baseline .096）仍被淘汰。
- 判定用单 seed，但 baseline seed 方差 .003–.02，多数"失败"与"机制不成立"结论基于 .0001–.005 的差异。
- 每个候选自己重训对照（matched control），同一 MultiHateLoc 在不同候选里 within 从 .555 到 .633 不等，相对对照的比较没有意义。
- 约 40 个候选在实现前被 novelty 审稿用"理论上可能退化"的推理否决，从未验证；通过审稿的候选也全部失败，该审稿对结果无预测力。
- 两天 7 次 process reset，每次只增加规则，规则从 25 条编号项涨到 38 条并互相矛盾。
- 所有候选都在同一套冻结特征上换 MIL 头，效应全在 ±.01；需要新输入的方法被 premise 门挡住。

## 规则

1. **数据集固定**：主数据集只有 HateMM 和 HateClipSeg（2026-09-02 裁定）。MHC-EN、MHC-ZH 不再作为实验数据集，不跑、不作门、不进论文主表；旧文档里的 MHC 数字只作历史记录。每个数据集独立 train/validation/test，永远不混合不同数据集的 train set。新数据集只做 external validation，加入前须用户同意。

2. **指标固定**：`CLAUDE.md` 裁定的三项（pooled AP、pooled ROC、within-video macro ROC），1 fps、test 集。SOTA 比较用 pooled AP/ROC（文献通用），within 只作附加分析（2026-09-06 起不作下限）（见第 8 条）。评测器全仓库只有一份（`scripts/reproduction_baselines/eval_baseline_scores.py`，核心 `scripts/duplex/frame_eval_common.py`），不得复制或改写。

3. **禁止 multi-model ensemble，训练阶段同样禁止。** 主方法不得在 inference 或 training 的任何阶段组合多个独立模型的 prediction、feature、posterior、pseudo-label 或 decision；multi-teacher aggregation、training-only ensemble、ensemble distillation、"先聚合多模型生成训练目标再部署单 student"均不合规。单一预训练编码器/VLM 作为特征来源不算 ensemble。禁止 inference 时的后处理（平滑、calibration、按语料路由）。

4. **Novelty 判定（proposal review，一名独立 agent，一次）**。候选来源两种都行：从 test error analysis 出发自己设计的机制；或从其他任务迁移的方法。只在以下四种情况 STOP，否则放行：
   - 来源方法已用于 hateful video detection / localization（审稿必须实际检索并记录）；
   - 纯 training/test ensemble；
   - 纯 calibration / 后处理 / 平滑；
   - 纯工程技巧而非完整科研方法（只调超参、只换特征、只加数据增强、只改训练配置）。

   不得以"可识别性""可能退化为常数/broadcast""可能存在 shortcut"等实现前推理 STOP；这些交给 test 结果判断。`research-wiki/FAILURE_EQUIVALENCE_LEDGER.md` 只作参考，不是阻断依据。

5. **方法可以带自己需要的输入。** 迁移方法需要新编码器、微调编码器、词级 ASR 时间戳、更高帧率、VLM 逐段特征、人脸/说话人轨迹等，直接抽取，不单独过审、不做 premise/observation 门。新缓存放 `data/<类型>/` 并写 `PROVENANCE.md`。换输入本身不构成 novelty（见第 4 条）。

6. **Code review（一名独立 agent，一次）**。只查会改变实验观察或结论的 bug：机制是否实际进入 forward/loss 与最终分数、train/validation/test 泄漏、特征/时间/标签/split 对齐、超参数与 checkpoint 加载链、是否调用统一评测器。不审风格、重构、健壮性、理论完备性。发现 bug 只确认修复，不重开泛化 review。

7. **训练与超参数搜索（每个 seed 固定走 Optuna）**。不做 smoke test、不做缩小数据/缩短 epoch 试跑、不要求单元测试。Code review 通过后直接在 HateMM 和 HateClipSeg 各自独立完整训练。
   - **搜索工具固定**：Optuna，TPE sampler，sampler seed 固定为当前训练 seed。每个 seed 单独建一个 study，两语料各自一个 study。study 存 `runs/<exp_id>/<corpus>/seed<seed>/optuna.db`（sqlite），每个 trial 输出 `runs/<exp_id>/<corpus>/seed<seed>/trial<k>/`（config、`run.log`、`metrics.json`）。
   - **trial 数固定，按单 trial 耗时定**：单个 trial（一次完整训练 + validation 选 ckpt + test 评测，含该 trial 需要重跑的 MLLM/VLM 推理）不超过 1 小时的方法，每个 seed 跑 20 个 trial；超过 1 小时的方法统一每个 seed 跑 5 个 trial。耗时按第 1 个 trial 实测归档，trial 数随即写进 README，之后不加不减；机器被占可以暂停，不可以缩减。
   - **每个 trial 的流程一样**：train set 训练，validation set 选 checkpoint（validation pooled AP 与 ROC 均值最高的 epoch），然后用该 checkpoint 在 test 上跑全部三项指标。
   - **搜索目标由 test 定**：Optuna 目标值是一个标量 = (test pooled AP + test pooled ROC) / 2，只用于给 Optuna 排序 trial；第 8 条过门仍按 AP、ROC 各自单独对照各自的门，不看均值。**不按 within 剪枝**（用户裁定 2026-09-06：within 不是弱监督视频定位的通用指标，且该下限是 C3/C6/C8 全部 trial 被剪的直接原因）；每个 trial 的 within 照常记录并写进 STATUS，只作附加分析。2026-09-06 之前的搜索按旧规则剪过 within，其 best trial 与新规则不可直接比较。这是开发期上限测量，与论文报什么无关：目的是记录每个 develop 出来的方法能到哪里。
   - **有效最终检验值** = 该 seed 全部 trial 中目标值最高者的 test 三项指标；第 8 条筛选与确认都用这个数。
   - 搜索空间（超参数名、范围、分布）在搜索开始前写进 README，两语料共用同一搜索空间（第 13 条），不得中途改。
   - 同时记录"若只按 validation 均值选 trial 会选到哪个 trial、它的 test 数字"，写进 STATUS 供参考，不作门。
   - Validation 只用于 trial 内选 checkpoint，不用于比较方法或决定方向。

8. **SOTA 定义（唯一晋级门）**。对照表固定为 `docs/duplex/OFFICIAL_VAL_RESULTS.md`（3 seed 均值），不重训 baseline，不做 matched control。
   - **主门（pooled，文献通用指标）**：HateMM 与 HateClipSeg 的 pooled frame AP 和 pooled frame ROC 四个数都超过表中最强训练方法的 3 seed 均值：HateMM AP `.573` / ROC `.807`（MACIL-SD）；HateClipSeg AP `.562`（Fed-WSVAD 3-client）/ ROC `.528`（DSANet）。
   - **within（附加分析，不作门；用户裁定 2026-09-06）**：两语料 within ROC 照常报告，并与 MultiHateLoc 3 seed 均值（HateMM `.632`、HateClipSeg `.524`）并列给出作参考；不参与筛选、确认或"谁更高"的比较。2026-09-06 之前它是硬下限（旧文本见 git 历史）。
   - VERA 只汇报（HateClipSeg `.619/.605/.562`），不作门：其分数是 0/1 加固定后处理，规则 3 禁止候选用同类后处理追它。
   - **筛选（单 seed 234）**：seed 234 的 Optuna 最优 trial（第 7 条）四个 pooled 数全部超过主门。
   - **确认（补 seed 2025、3407）**：每个 seed 各自跑同样 trial 数的完整 Optuna 搜索，取各自最优 trial；3 seed 均值仍全部满足，且每个 pooled 领先幅度不小于候选与 baseline 两者 seed 标准差中较大者（最低 .005）。
   - 单 seed 差异在 .005 以内视为噪声，既不算赢也不算输。

9. **分流**。
   - 确认 SOTA：按第 14 条核对后向用户汇报；随后为论文补消融与独立 novelty 复查。
   - 筛选未过，但某语料 pooled AP 或 ROC 比最强训练 baseline 高 .01 以上：方法保留，用 test error analysis 找原因后修改再训，同一方法最多 3 轮修改；仍未过则归档，写明最好数字。
   - 没有 .01 以上 pooled 提升：归档，写一行负结果，换候选。
   - 实现不可靠：修复重跑，不评价 idea。

10. **Test error analysis 合规使用**。允许读取 test predictions 与 test GT 做 error analysis 并 inform 后续设计；每次记录看了哪些 artifact、发现什么、影响了哪个设计决策。由此得到的 test 结果属于 developmental evidence，不表述为未揭盲 confirmatory 结果。test 标签不得参与梯度训练或 checkpoint 选择；test 指标可作为第 7 条 Optuna 搜索的目标值（开发期上限测量），必须在 README 与 STATUS 写明。

11. **不设自动停机与 process review。** 取消失败计数、3 次失败触发审查、RESET/epoch 机制。每归档 5 个候选，主 agent 在 `research-wiki/STATUS.md` 写一段不超过 10 行的小结（试了什么、最好数字、下一步）给用户看，然后继续。流程规则只由用户改。

12. **归档**。每轮一个 `experiments/<YYYYMMDD>_<slug>/`，含 README（机制、来源、怎么跑、结果、去向）；输出在 `runs/<exp_id>/<run_name>/`（config、代码版本说明、`run.log`、`run.pid`、`metrics.json`）。淘汰后整目录移入 `archive/experiments/`，README 顶部一行原因。权威数字只认 `runs/` 里评测器输出。每轮结束更新 `research-wiki/STATUS.md`。

13. **方法必须统一（反模式：按数据集换训练策略）**。一个方法 = 一套架构、一套损失、一套训练流程、一套推理流程，四个语料完全相同。语料之间只允许标量超参数不同（学习率、epoch、top-K、时间窗长度、损失权重），且必须由同一份预先声明的 validation 搜索空间和选择规则自动选出。以下按语料改动一律视为不同方法，不得合并成一个方法 claim：
   - backbone 冻结/微调切换；
   - 手写的、按语料不同的规则或 policy 表达式；
   - 输出读出形式不同（残差、直接、只正证据等）；
   - 模块开关不同；
   - 单语料训练与多语料联合训练切换；
   - 按语料的后处理或校准；
   - 按语料挑不同分支或 head 报数。

   POWA 现状（HateMM/MHC-EN 微调 backbone、MHC-ZH 冻结 backbone、HateClipSeg 48 秒窗加联合训练、各语料手写 policy）就是这种反模式；重开 POWA 必须先统一，统一后重新跑 HateMM 和 HateClipSeg 数字。

14. **"做完了"的声明门槛（防走捷径）**。向用户汇报 SOTA 前必须同时满足下列各项，任一缺失只能汇报为"进行中"并写明缺哪项：
   - (a) 第 8 条确认级（3 seed）全过，每个 seed 的数字来自该 seed 完整 Optuna 搜索的最优 trial（第 7 条），不是开发期零散跑出来的历史数字；
   - (b) 报 seed 标准差；领先幅度小于标准差时不得写"超过"；
   - (c) 第 13 条统一性满足；
   - (d) checkpoint 来自 validation；超参数来自第 7 条的固定 Optuna 搜索，搜索空间、trial 数、目标值定义在搜索前写进 README，不得事后改；报告中写明超参数搜索目标为 test，并同时给出 validation 选 trial 的 test 数字；
   - (e) 无 inference 后处理、无 ensemble、无按语料分支选择；若用了 train-only 单一 teacher（如 VLM 伪标签），必须写明，并在消融中报去掉 teacher 的数字；
   - (f) 用了比 baseline 更强的特征时，报"最强 baseline + 同样特征"的数字；
   - (g) 消融在 test 上显示核心机制去掉后 pooled 下降：**三 seed 均值下降 ≥ .01（AP 或 ROC），两语料都满足**；不要求每个 seed 都降（用户裁定 2026-09-06，取消 09-05 的"每 seed 都降"要求）；否则该机制不能作为 novelty 主张。单 seed 或均值 < .01 的差异只作记录（用户裁定 2026-09-05；依据：同超参只换随机数流单次分数 std .006–.009、极差最大 .024，搜索选出的 best trial 比自身流均值高 .006，见 `experiments/20260904_null_token_cma/README.md` 8.2）。
   - (h) 评测器、split、GT、1 fps 协议未改动；
   - (i) HateMM 与 HateClipSeg 两语料全部三项指标都报，不挑语料、不挑指标。

## 标准流程

1. 主 agent 提出方法：来自 test error analysis 的自研机制，或从其他任务迁移。写一页 README：机制、来源、需要的输入、预期改善什么。
2. 独立 agent 按第 4 条做 proposal review。放行即实现。
3. 实现，抽取需要的输入。
4. 独立 agent 按第 6 条做一次 code review。
5. HateMM、HateClipSeg 各自跑 seed 234 的固定 Optuna 搜索（第 7 条）：每个 trial 完整训练、validation 选 checkpoint、test 出三项指标。
6. 取最优 trial 的 test 数字，按第 8 条筛选。
7. 按第 9 条分流：过则补 seed 确认，按第 14 条核对后汇报；有提升则修改继续；无提升则归档换候选。
