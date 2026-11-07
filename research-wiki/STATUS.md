# 当前研究状态

截至 **2026-11-07**（本次核验旧任务完成及关闭监控；当前方法沿用11-03已接受的r6）。2026-10-31 时：上一任务“三个数据集上跑一遍最新完整的给我汇报”已完成；随后按用户“接着去做你觉得有用的实验”补做两项诊断（全trial按设置选模型、预算曲线），**已完成**并回传本机。依据代码 `experiments/20261002_complete_io_method/`（含 `audit_dehate.py`、`select_by_setting.py`、`budget_curve.py`）、共享 `src/qtl/`，以及本机 `runs/20261002_complete_io_method/` 下统一评测与审计。[上一轮backbone收尾状态](../archive/research-wiki/STATUS_20261002_backbone_closed.md)已归档。

## 当前目标与结论

**本次旧heartbeat已关闭。** 它绑定的“三数据集完整组合评估”已完成：9个study各20 trial、每trial50 epoch，72份完整组合统一评测已回传并核验；远端5个owner及同会话进程全部退出。独立monitor均已退出；heartbeat原进程也已不存在，本次写入STOP并记录关闭，避免重新启用。依据 `runs/20261002_complete_io_method/analysis/three_dataset_task_closure.json`、`setup/final_process_closure.json`、`setup/monitor_health.json`。这是旧评估任务收尾，不表示下列历史性能目标全部达标；Pursuing Goal未创建或恢复。

方法示意图（当前方法的数据流、训练、停止规则）：[method_framework.html](method_framework.html)。

**当前方法 = 第 6 版（用户 2026-11-03 裁定取代 r5）。** r5 + 答案表加两状态共用、只在负例节点上估计的长度斜率（训练、EIG、后验、停止同一张表）+ 去掉只在测试用的复制似然。依据与数字：[20261101_r6_length_table/README.md](../experiments/20261101_r6_length_table/README.md)（搜索 3 seed × 20 trial，`runs/20261101_r6_length_table/`）；过程与诊断：[20261101_r5_completion/README.md](../experiments/20261101_r5_completion/README.md)。
- 固定 8 次 AP / ROC / within（test 选 trial，统一评测器）：HateMM .702 / .889 / .783（r5 .691 / .885），HCS .691 / .680 / .563（r5 .687 / .674），DeHate .279 / .770 / .681（r5 .262 / .780）。来源 `runs/20261101_r6_length_table/final_curves/test_selected/<corpus>/summary.json`。
- 单调性：调用越多总体越好，但有局部回落（11-08 外部审稿后更正）。validation 选 trial 时 32 次比 8 次 AP 高 .011 / .012 / .015（HateMM / HCS / DeHate）、ROC −.001 / +.019 / +.022、within 高 .012–.019；相邻点最大回落 HCS 3→4 次 AP / ROC −.009、HateMM 24→32 次 ROC −.006；DeHate 无超过 .005 的回落。按 test 选的 trial 在 8 次后 AP / ROC 略降（≤ .012，trial 与 epoch 都按 8 次选）。
- 代价：DeHate 8 次 ROC 比 r5 低 .010（预注册 Q2 不过，用户接受）；预算 5 次比固定 8 次 AP 低 .012–.019（r5 在 .01 内）。
- 最强 baseline + 同样 8 个回答（14(f)）：仍低于本方法 AP .059 / .093 / .030（HateMM / HCS / DeHate）。
- 消融与对照（14(g)，11-03 三语料完成）：14 个臂（去节点先验、去长度项、两状态各自斜率、文字转录、逐秒相加、按层提问、只用标签训练、独立先验、联合学习表、耦合链、单层 4 / 8 / 16 窗口、无骨干）在 HateMM、HCS、DeHate 上 AP 或 ROC 都降 ≥ .01，全部有用；余量最小 DeHate 按层提问 .013。无骨干时 AP / ROC 降：HateMM .057 / .013，HCS −.003 / .023，DeHate .083 / .013。来源 `runs/20261101_r6_length_table/arms/table.json`（`arms_table.py`）。
- 未采用：每视频回答偏移（AP 随调用下降）、第 7 版"先验加标签项"（代码审查发现该项本来就在损失里，撤回）。

**历史目标（2026-10-31）：用户当时批准两个目标并开启自主迭代，两者都达成后才汇报。后续11-01改为报告用户预算曲线，11-03接受r6及其DeHate ROC代价；不再把当时的待裁定选项当成当前运行任务。** 2026-11-01 用户裁定 DeHate 升为主数据集（`CLAUDE.md`、`RESEARCH_ITERATION_RULES.md` 第 1、7、8、14 条已改），两个目标同样要求在 DeHate 上成立。
- 目标 1（少调用）：HateMM、HCS、DeHate 上最终方法平均约 4–5 次调用；与同一方法固定 8 次相比，AP/ROC 三 seed 均值降幅 ≤ .01，within 不降。
- 目标 2（自研骨干不输 r5）：自研骨干在固定 8 次、规则 7 完整搜索下，三个主数据集三 seed AP 与 ROC 均值不低于 r5（HateMM .6909 / .8849，HCS .6865 / .6737，DeHate .2620 / .7800，`runs/20260929_query_paradigm_r5/summary_dehate.txt`）。

诊断（开发期 test 分析，`experiments/20261031_two_goal_diagnostics/`，输出 `runs/20261031_two_goal_diagnostics/`）：
- 调用分配（`call_allocation.py`）：HCS test 87% 是正例视频，只给负例少问降不到 4–5 次；HateMM 即使用真实标签给正例 8 次、负例 2 次（平均 4.3 次），ROC 也低 .028。只靠重新分配调用不够，需要每次调用更有效，或骨干先验更强。
- 骨干差距（`backbone_gap.py`）：HateMM 的差距在视频内分数形状（换上 r5 的视频内形状后 within 追平、AP 回收 .009）；HCS 的差距主要在视频级区分（视频 ROC .795 对 .865）。
- 停止规则扫描（`rule_scan.py`、`order_anatomy.py`）：4–5 次时两类视频抢调用——看视频级不确定/最后变化的规则（qmixSG 等）保住 AP/ROC，但 HCS within 降 .005–.008；看剩余信息量的规则把调用给正例，within 涨但 AP/ROC 降 .012–.025。现行 qmixSG_rt10 在平均 5 次时 HCS within −.014。
- 目标 1：[qmixE1 停止规则](../experiments/20261031_fewer_calls/README.md)（最后一次回答造成的逐秒 log-odds 总变化 ∨ 下一个最佳问题的 EIG，各取 validation 分位数后取大，validation 平均 5 次定阈值；最多 8 次 = 被比较的固定预算）。proposal GO、code review PASS（review 指出原 32 次上限会改变阈值，改为 8）。**统一评测器、r5 三 seed：HateMM 4.92 次，AP/ROC/within 比固定 8 次 −.0023/−.0024/+.0012；HCS 4.96 次，−.0095/−.0089/+.0046；DeHate 4.95 次，−.0023/−.0022/−.0024——目标 1 在 r5 上达成**，HCS AP 离 .01 只剩 .0005。消融：单用 dlsum HCS within −.006、单用 EIG HCS AP/ROC −.021、均匀 5 次两语料 within −.009/−.007，都不达标；但规则 14(g) 的“去掉部件 pooled 降 ≥.01、两语料”不满足（这是效率结果）。待在目标 2 最终骨干上复核。来源 `runs/20261031_fewer_calls/<corpus>/{r5,r5_ablation}/summary_*.json`。平均 5 次是手设的预算；11-01 检查了让停止规则在 validation 上自己选预算（argmax、1 个标准误规则，`stop_budget_select.py`，诊断第 11 条）：validation 目标的标准误 .025–.053，远大于 .005–.01 的差，选出的预算随 seed 在 1–7 次之间变；argmax 三语料 AP/ROC 都在 .005 内，但 HCS 7.1 次、DeHate 6.5 次。结论：现有 validation 规模下，平均调用次数只能作为使用者设的预算。**11-01 用户裁定：平均调用次数作为使用者按资源设的预算（同固定 k 的 k），论文报 1–7 次的曲线，5 只是其中一点。** 曲线（`stop_budget_curve.py`，`runs/20261031_two_goal_diagnostics/stop_budget_curve/`，fewer_calls README 第 6 节）：HateMM 每个预算都比同次数固定 k 高（AP +.006–.011），预算 ≥5 与固定 8 差 ≤.003；HCS 预算 4–7 AP/ROC 一直比固定 8 低约 .009（规则给负例的调用少，HCS 负例需要调用），within 比固定 k 高 .010–.018；DeHate 预算 ≥4 与固定 8 差 ≤.005。
- 目标 2 候选：[尺度感知跨模态注意力](../experiments/20261031_scale_aware_cma/README.md)，proposal GO、code review PASS；seed 234 搜索 10-31 19:15 起（Slurm job 1301 lab3 HateMM，1302 lab1 HCS）。**HCS seed 234 完成**：最好 trial 7 AP/ROC/within .6853/.6885/.5859，r5 同 seed .6885/.6748/.5561（−.003/+.014，在 .005 内持平）；**HCS 三 seed 完成，目标 2 未达成**：seed 234/2025/3407 最好 trial AP/ROC .6853/.6885、.6921/.6749、.6799/.6495，均值 .6858/.6710，r5 .6865/.6737（−.0007/−.0027）；within .5793 对 .5602。30 对同超参 trial 的差 −.004±.004，偏置大小与差无关，同超参重复训练本身差 ±.02，即该机制在 HCS 无可测效果。HateMM seed 234（job 1301，11-01 完成，作参考）：最好 trial 15 .7083/.8867/.7590，r5 同 seed .6944/.8867/.7887（+.014/.000/−.030）；20 trial 均值低于 r5（.665/.872/.744 对 .676/.878/.751），最好 trial 的 AP 增益在 ±.03 的 trial 噪声内。seed 2025/3407 不跑。r5 剩余误差（`headroom.py`）：HCS 视频内排序完美可到 .806/.796、负例视频全判对 .705/.717；HateMM 分别 .749/.902、.751/.929。来源 `runs/20261031_scale_aware_cma/hateclipseg/seed{234,2025,3407}/`、`runs/20261031_two_goal_diagnostics/headroom/`。网络已移到 `src/qtl/scale_cma.py`（代码不变），`src/qtl/backbones.py` 统一加载各骨干 checkpoint。HCS 同超参逐 trial 对 r5 差 −.040 到 +.012，偏置表只在 lr_answer ≥ .008 时明显学到（HCS 多个 trial 在 epoch 1–3 被选中，偏置≈0，等同 r5，仍差 .04，即同超参不同初始化的波动约 ±.04）。

以下为 10-31 之前的完整组合结论：

[最新完整组合](../experiments/20261002_complete_io_method/README.md)：**初版inside–outside骨干 + 单问软答案8档 + copy_neg复制似然（同时进入EIG与后验）+ qmixSG_rt10自动停止**。当时HateMM、HateClipSeg为主数据，DeHate为external validation（11-01 起 DeHate 也是主数据集）。同一架构/损失/推断流程，不按语料选骨干。

结论（负结果，如实保留）：
1. **自动停止三语料都没有少调用**：平均调用HateMM 7.63→7.86、HCS 8.00→7.98、DeHate 7.26→7.93（固定8次在短视频上问不满8个节点）。
2. **对同复制设置固定8次**：HateMM、DeHate三项变化均在±.003内；HCS三项下降.007–.009。
3. **对旧r5同复制/停止设置**：三语料AP均下降（−.023/−.030/−.035），ROC均下降（−.012/−.054/−.013）；within HateMM/HCS下降，DeHate上升.010。
4. 新骨干接入复制+停止后没有显示省调用或相对r5涨点。
5. **选模型方式不是原因**（10-31）：每个设置在全部20 trial中自己选trial后，自动−fixed8仍是HateMM小正、HCS小负、DeHate≈0；60个trial同trial比较同样。
6. **任何预算都不省调用**（10-31）：停止阈值重校准到平均2/4/6/8/12/16次，与同调用数固定次数之差全部在±.010内、方向不稳定；停止规则没有把调用多分给正例视频。
7. 固定次数曲线：AP/ROC大部分增益在前4次内（fixed4与fixed8 AP差≤.002），within随调用继续上升（fixed4比fixed8低.004–.025）。详见[README第9节](../experiments/20261002_complete_io_method/README.md)。下一方向待用户裁定。

两主数据复用初版各seed234/2025/3407完整20-trial×50 epoch搜索的既定checkpoint；DeHate补同一骨干、同一搜索空间的三seed完整Optuna（各锁20 trial，每trial 50 epoch）。三语料均沿用训练阶段无复制fixed8的validation AP/ROC均值选checkpoint，开发期test(AP+ROC)/2选trial；完整组合结果不参与重新选择。**这是预先选定模型的完整组合验证，不是自动停止目标的重新Optuna调参。**

一次[独立集成code review：PASS](../docs/reviews/20261002_complete_io_method_code.md)。训练/推断不写data，评测器未改，无新增VLM调用或特征抽取；新视频仍需原有输入处理及实际查询成本。

## 三模块实现与缺口

| 模块 | 当前实现 | 证据与限制 |
|---|---|---|
| VLM观测与提问 | 二分时间树、EIG选节点；单问首token软答案，训练答案无标注分位8档 | 使用相同I3D/VGGish/BERT、词级转录和VLM缓存；8档是答案离散精度，与8次调用不同 |
| 弱监督骨干与先验 | r6沿用r5骨干、秒级与节点先验、视频头和零膨胀链；实现入口 `experiments/20261101_r6_length_table/` | 初版inside–outside属于下方历史实验，不是当前采用的骨干；当前无骨干与其它对照见r6 README |
| 答案融合与停止 | 两状态共用、负例节点估计的长度斜率表；训练/EIG/后验一致，去掉复制似然；qmixE1按用户预算在validation定阈值 | 预算曲线已完成；r6预算5的AP比固定8低.012–.019，不能声称原“5次几乎不降”目标已达成 |

## 历史inside–outside完整组合结果与来源

当前r6权威结果见本文首节及r6 README；以下是已完成旧任务的结果。AP / ROC / within，三seed均值±总体标准差，开发期test目标选trial。调用数为实际消费的缓存答案前缀；32次轨迹仅供比较。

| 语料 / 设置 | AP / ROC / within | 平均调用 |
|---|---|---|
| HateMM / 0次 | 0.591739±0.007204 / 0.789032±0.008709 / 0.715851±0.021907 | 0.0000 |
| HateMM / 同复制设置固定8次 | 0.668707±0.018339 / 0.874930±0.004352 / 0.735957±0.019076 | 7.6262 |
| HateMM / 完整组合自动停止 | 0.671441±0.015171 / 0.877612±0.003897 / 0.730922±0.016773 | 7.8583 |
| HateMM / 32次（附加） | 0.684682±0.011156 / 0.881732±0.003157 / 0.749468±0.016806 | 25.0187 |
| HateClipSeg / 0次 | 0.633132±0.032202 / 0.610808±0.037310 / 0.531452±0.001537 | 0.0000 |
| HateClipSeg / 同复制设置固定8次 | 0.660655±0.010263 / 0.630674±0.022582 / 0.552625±0.007539 | 8.0000 |
| HateClipSeg / 完整组合自动停止 | 0.653658±0.010247 / 0.621388±0.018894 / 0.545536±0.004205 | 7.9831 |
| HateClipSeg / 32次（附加） | 0.675262±0.017010 / 0.638592±0.026383 / 0.594112±0.022389 | 32.0000 |
| DeHate / 0次 | 0.188340±0.007451 / 0.702571±0.010153 / 0.668309±0.005206 | 0.0000 |
| DeHate / 同复制设置固定8次 | 0.217502±0.007884 / 0.763608±0.005315 / 0.679874±0.008689 | 7.2641 |
| DeHate / 完整组合自动停止 | 0.219813±0.010473 / 0.761690±0.005629 / 0.682276±0.009662 | 7.9334 |
| DeHate / 32次（附加） | 0.221194±0.013956 / 0.767275±0.000652 / 0.708132±0.004667 | 21.9600 |

自动减同复制固定8次的逐seed配对均值±总体标准差：HateMM **+.002733±.003713 / +.002682±.002496 / −.005035±.005014**；HCS **−.006997±.001360 / −.009286±.003710 / −.007089±.007995**；DeHate **+.002311±.003583 / −.001919±.000723 / +.002402±.004429**。HCS三个seed的AP/ROC均下降；DeHate三个seed的ROC均小幅下降。

仅validation选trial的完整组合自动停止（候选仍来自原test目标驱动搜索，不称独立validation-only调参）：

| 语料 | AP / ROC / within | 平均调用 |
|---|---|---|
| HateMM | 0.643795±0.018646 / 0.875095±0.001461 / 0.677516±0.055519 | 7.8723 |
| HateClipSeg | 0.643983±0.006301 / 0.616759±0.006254 / 0.547546±0.021578 | 7.6160 |
| DeHate | 0.205808±0.019643 / 0.750641±0.010609 / 0.641601±0.035813 | 8.0269 |

对旧r5同样的copy_neg+qmixSG_rt10，最新组合AP/ROC/within均值变化：HateMM **−.023201 / −.012166 / −.038214**；HCS **−.029783 / −.053544 / −.011952**；DeHate **−.034894 / −.012592 / +.010141**。

本机原始权威文件：
- 两主数据 `runs/20261002_complete_io_method/<corpus>/integrated/results/seed<seed>_trial<k>/metrics_test_{fixed0,fixed8,fixed32,qmixSG_rt10}.json`；HateMM test/validation选trial 17/11/11与10/2/9，HCS 11/2/10与6/17/1（seed234/2025/3407）。
- DeHate `runs/20261002_complete_io_method/dehate/seed<seed>/results/seed<seed>_trial<k>/metrics_test_*.json`；test选trial 5/12/14、validation选4/11/8，覆盖1151视频/110839秒。
- 审计与逐seed/配对/r5来源：`runs/20261002_complete_io_method/analysis/{hatemm,hateclipseg,dehate}_complete_method.json`，两主数据汇总 `analysis/main_corpora_complete_method.json`。
- 旧r5对照 `runs/20260929_query_paradigm_r5/stop_check/<corpus>/seed<seed>_trial<k>_r5rt/metrics_test_qmixSG_rt108.json`（HateMM trial9/4/5，HCS 13/11/19，DeHate 8/4/4）。

## 运行任务与monitor

2026-11-07实际检查：本用户Slurm队列为空；三数据旧完整组合5个owner及全部同会话进程均已退出，completion为success。9个源study各20个COMPLETE trial、每trial50 epoch，本机72份完整组合评测与已有全量审计一致。DeHate此前已核对cohort、trial选择、checkpoint、阈值/调用数和分数前缀，本轮没有重复训练或推断。

旧任务5个独立monitor均已退出。对应heartbeat目录 `runs/thread_monitor/01a0f639-b211-75e3-9155-e15e30534b46/20261002_complete_io_method/` 已写STOP，进程不存在；其历史记录最后通知为10-03，不把这次延迟提醒当作仍有运行任务。当前记录 `runs/20261002_complete_io_method/setup/monitor_health.json`，原始结果完成核验 `analysis/three_dataset_task_closure.json`。Pursuing Goal保持用户暂停状态，未调用Goal工具。

r6完整搜索、预算曲线、14臂消融和baseline+同回答对照已完成；11-03接受结果与限制见 `experiments/20261101_r5_completion/README.md` 的P1–P10及r6 README。已知无关tandem.html、lab1 idea-stage/、三机家目录STRAY保留；本轮只更新状态及关闭记录，不改方法、研究规则或GPU运行方式。

## 下一步

本次绑定的三数据评估任务无待运行项，关闭旧monitor，不重复启动旧搜索。当前方法按11-03裁定保留r6；原5次调用性能目标与DeHate相对r5 ROC差距仍作为已知限制报告，不把“实验完成”写成“所有性能门均通过”。后续实验按新的用户任务推进。

## 历史与规则

[本轮运行协议与结果](../experiments/20261002_complete_io_method/README.md)、[此前backbone收尾](../archive/research-wiki/STATUS_20261002_backbone_closed.md)、[初版完整搜索及自身消融](../experiments/20261001_inside_outside_backbone/README.md)、[研究规则](../RESEARCH_ITERATION_RULES.md)。详细过程留各实验README，STATUS只保留当前事实。
