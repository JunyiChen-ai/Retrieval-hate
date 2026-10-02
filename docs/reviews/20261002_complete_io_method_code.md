# 最新完整 IO 方法：一次独立 code review

日期：2026-10-02。审稿人：独立 Codex agent。用户任务：**“三个数据集上跑一遍最新完整的给我汇报”**。

**结论：PASS。发现的一项真实输入兼容问题已修复并确认；无待修复的阻断问题。** 本结论仅说明所声明组合的实现可以运行，不说明性能、novelty 或完整组合的 Optuna 确认已经成立。

本次按 `RESEARCH_ITERATION_RULES.md` 第 6 条进行一次 code review，只查影响结果或结论的 bug。阅读新实验 `experiments/20261002_complete_io_method/{README.md,evaluate.py,check_inputs.py,train.py,search.py,launch/}`、共享 `src/qtl/{inside_outside,checkpoint,copy_noise,stop_rules,input_check}.py`，并追踪共享 trainer、search、data、policy、cpolicy、qtree、统一评测调用和现有 monitor。未进行新 proposal review、训练、smoke、缩短实验、模型前向或优化器更新；审稿人仅编辑本记录，业务修复由主 agent 完成。

## 1. 实际发现并修复的问题

新输入检查曾无条件要求每个视频至少有一个 observed 答案。这会误拒绝 HateMM 的合法两秒视频 `hate_video_272`：现有提问树只允许长度至少四秒的节点，该视频没有 queryable 节点，既有训练/推断协议允许零答案并在零调用时使用先验。

主 agent 已改为：只有树中存在 queryable 节点才要求非零 observed；没有 queryable 节点的视频保留在原 cohort，并记录于 `videos_without_queryable_nodes`。已复核修复后的代码、真实缓存和 `qtree.tree` 的长度条件。没有删除视频、补造答案或改变 GT/split。

## 2. 接入与选择链

1. **新骨干和节点势确实参与最终分数。** 新训练入口显式传 `qtl.inside_outside.make_model` 和原 EXTRA_DEFAULTS；重评也显式用同一工厂，核对 inside_outside、outside/gated、soft_both/8、node_prior、anchored/tree/eig 配置。推断沿 `policy.video_prior_nodes → cpolicy.run_batch → ctree.marginals` 传递模型输出的 s、g、phi，未静默替换成旧 PriorNet。
2. **真实 checkpoint 可以严格加载。** 在 CPU 上实际读取 HateMM、HateClipSeg 各三 seed 的 test-selected 和 validation-selected，共 12 个现有 checkpoint。全部 model/answer/chain 的 `strict=True` 加载成功；模型类均为 `InsideOutsidePrior`，checkpoint epoch 与各 trial 的 `summary.best_epoch` 一致。未调用 forward。HateMM 选中 trial 对为 234:10/17、2025:2/11、3407:9/11；HCS 为 234:6/11、2025:2/17、3407:1/10（每对按编号排序，不表示选择类别）。
3. **没有在完整组合 test 上重新选模型。** `selected_trials` 从已经完成的 source study 固定提取两类 winner，检查 budget、全部 COMPLETE 和每 trial 1–50 epoch history；同一 winner 重合时只评一次并保留两个 selection 标签。DeHate 沿用原共享搜索与训练协议：每 seed 独立 TPE、相同搜索空间、完整 50 epoch，validation fixed8 AP/ROC 均值选 checkpoint，开发期 test fixed8 目标选 trial，并单独记录 validation-selected。README 明确本次是预先选定模型的组合重评，不是自动停止目标的重新调参或确认搜索。
4. **复制概率只拟合 train 负例。** `estimate_pi_neg` 的输入限定当前语料 train IDs，内部过滤 video label=0；长度桶与父子答案统计使用这些训练视频及已训练的答案模型。soft 分位数也由共享 loader 的 train 元数据计算。validation/test 的答案、逐秒 GT 不进入复制概率估计。
5. **复制设置实际同时进入提问与似然。** 重评以 `copy_pi` 和 `copy_mode="both"` 调用共享 cpolicy；候选 EIG 使用复制混合概率，读取当前答案后使用对应混合似然更新 A3。参考答案仅从 `asked_o` 中此前已问节点选取，不从未问节点取答案。
6. **停止阈值仅来自 validation，前缀因果。** 固定 qmixSG_rt10 只用 dlsum、stab1、hG。每个位置 k 的统计只依赖 scores[k]、scores[k−1] 和 p_G[k]；完整轨迹中后续答案不会反向改变前缀。分位数参考只取 validation 的有限统计，阈值由 validation 无标签平均调用数目标 8 决定；test 只套用阈值并选相应前缀，最多 32 次。未根据本轮 test 挑规则、松弛率或阈值。初始变化量为 infinity，与原规则一致，没有另加固定调用数下限。完整离线轨迹用于重评，不被报告为部署必须执行 32 次。

## 3. 输入、迁移与完成语义

- **真实 cohort 元数据检查。** 审稿期间实际解析三语料已有缓存和 split，核对每个纳入视频的答案存在性、答案 split 元数据、音频秒数/答案 T、validation/test GT 长度、BERT 存在性及形状。HateMM train/val/test 为 744/109/214；HCS 为 251/63/79；DeHate 为 4679/559/1151。除上述已修复的合法短视频条件外无不一致。DeHate 沿用现有无可用 span 正例的 frame-evaluation 排除清单，以及 train 视频 `8bAfN6vXoIZp` 的已登记无视觉特征排除；未新增排除或混合数据集。
- **远端完整输入检查仍由 owner 执行。** 本次实际审稿解析不替代各执行主机完整 I3D/VGGish/BERT finite、shape、节点覆盖和 GT 检查。新 owner 在训练/重评前调用共享 `check_inputs`，同主机同语料检查受文件锁保护；检查未成功不写可复用结果。训练/重评不向 data 写入。
- **共享迁移未改原算法。** 直接比较 Git HEAD 原初版 backbone 与 `src/qtl/inside_outside.py`，文本完全一致；对 copy_noise 两函数及 stop_rules 六函数逐一比较语法树，全部一致，无内容指纹计算。旧实验用兼容入口复用这些实现；原 data/qtree 兼容层仍指向同一共享 module，软档数 global 不会分叉。共享 checkpoint loader 在原加载逻辑上增加严格状态与 epoch 核对；答案模型和 chain 无 dropout，eval 模式不改变它们的已有数值行为。
- **统一评测与覆盖保持。** 输出经原 `hc.write_scores` 与 `hc.run_evaluator`；没有复制或修改评测逻辑。新评估核对 test IDs、每条秒级分数长度/有限性、统一评测的缺失/多余视频数、总视频数及总帧数；固定 0/8/32 和自动停止均保存原始评测 JSON。AP、ROC、within 同时记录，停止均值及正负例均值由实际消费前缀计数计算。
- **失败不会正常完成。** search 完整预算不足时抛错；owner 的 subprocess 使用 check=True，失败记录 status=failed 并非零退出，只有重评汇总成功后才写 success。已有 identity 阻止重复覆盖运行。现有 monitor 将失败 completion 识别为 FAILED，绑定 host/run/PID/start_ticks/PGID/SID，SSH 失败仅重试。新 owner 的子进程命令使用绝对仓库路径，符合现有子进程筛选。最终完成通知仍不能代替主 agent 的真实进程、训练预算和输出完整性核验。

所有新 Python 文件以及修复后的输入检查通过语法解析，三个 shell launcher 通过 `bash -n`。本 review 未启动实验或 monitor；实际 detached 启动、监控绑定、首轮 RUNNING 与通知接口可达性由主 agent 在开跑时确认。
