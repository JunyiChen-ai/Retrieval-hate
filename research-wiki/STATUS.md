# 当前研究状态

截至 **2026-10-31**。上一任务“三个数据集上跑一遍最新完整的给我汇报”已完成；随后按用户“接着去做你觉得有用的实验”补做两项诊断（全trial按设置选模型、预算曲线），**已完成**并回传本机。依据代码 `experiments/20261002_complete_io_method/`（含 `audit_dehate.py`、`select_by_setting.py`、`budget_curve.py`）、共享 `src/qtl/`，以及本机 `runs/20261002_complete_io_method/` 下统一评测与审计。[上一轮backbone收尾状态](../archive/research-wiki/STATUS_20261002_backbone_closed.md)已归档。

## 当前目标与结论

[最新完整组合](../experiments/20261002_complete_io_method/README.md)：**初版inside–outside骨干 + 单问软答案8档 + copy_neg复制似然（同时进入EIG与后验）+ qmixSG_rt10自动停止**。HateMM、HateClipSeg为主数据；DeHate为external validation。同一架构/损失/推断流程，不按语料选骨干。

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
| 弱监督骨干与先验 | `src/qtl/inside_outside.py`，初版outside/gated；秒级和节点先验、视频头、零膨胀链、原训练损失 | 初版自己的固定预算outside消融支持有限迁移novelty；相对r5性能有代价。该消融没有在当前复制+停止组合上重新做 |
| 答案融合与停止 | 训练负例估计copy概率，copy同时进EIG/似然；qmixSG_rt10的validation分位参考与平均8次阈值 | 校准仅用validation，test只取因果前缀；三语料调用量都没有下降，HCS三项下降 |

## 最新权威结果与来源

AP / ROC / within，三seed均值±总体标准差，开发期test目标选trial。调用数为实际消费的缓存答案前缀；32次轨迹仅供比较。

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

## 运行任务

无运行中任务。10-31诊断：Slurm job 1295–1299（lab3：HateMM、DeHate seed234/3407；lab1：HCS、DeHate seed2025）07:54–08:50全部COMPLETED，无Traceback；两台CPU预算曲线08:00/08:01输出DONE。输出已回传本机 `runs/20261002_complete_io_method/{all_trials,budget_curve,analysis/select_by_setting.json}`。原选定trial重跑复现：HateMM/HCS完全一致，DeHate seed234 trial4最大差.0012（GPU非确定性）。此前完整组合：HCS（lab1）与HateMM（lab3）10-02 18:35前后结束；DeHate seed2025（lab1）10-03 04:41、seed234/3407（lab3）09:17/09:18结束，`completion.json` 均为success。

DeHate审计需要远端cohort（本机无DeHate split/GT/特征），因此 `audit_dehate.py --seeds` 在lab1（seed2025）、lab3（seed234/3407）运行，部分结果回传后本机 `--merge`；所有核对项通过（完成标记/主机、20 trial×50 epoch、选择重推、checkpoint epoch、复制与分位参考来源、阈值与逐视频调用数重算、分数前缀、24份评测完整覆盖）。

已知无关项保留：本机 `tandem.html`、lab1 `idea-stage/`、三机 `~` 下其它项目的STRAY条目（ctx_pri、MemoryAgen、conversation-pilot等，非本项目产生）。研究规则未改；CLAUDE/AGENTS于10-31补记lab机GPU自10-03起只能经Slurm使用。

## 下一步

1. 两项诊断已完成，结论为当前停止规则在任何预算下都无收益。等待用户对下一方向的裁定（例如：停止统计量改为针对正例分配调用，或接受固定次数并改做骨干相对r5的差距）。

## 历史与规则

[本轮运行协议与结果](../experiments/20261002_complete_io_method/README.md)、[此前backbone收尾](../archive/research-wiki/STATUS_20261002_backbone_closed.md)、[初版完整搜索及自身消融](../experiments/20261001_inside_outside_backbone/README.md)、[研究规则](../RESEARCH_ITERATION_RULES.md)。详细过程留各实验README，STATUS只保留当前事实。
