# 当前研究状态

截至 **2026-10-02 21:37 NZDT**。当前任务为用户“**三个数据集上跑一遍最新完整的给我汇报**”；依据代码 `experiments/20261002_complete_io_method/`、共享 `src/qtl/`，以及本机 `runs/20261002_complete_io_method/` 下统一评测与审计。[上一轮backbone收尾状态](../archive/research-wiki/STATUS_20261002_backbone_closed.md)已归档。

## 当前目标与结论

[最新完整组合](../experiments/20261002_complete_io_method/README.md)：**初版inside–outside骨干 + 单问软答案8档 + copy_neg复制似然（同时进入EIG与后验）+ qmixSG_rt10自动停止**。HateMM、HateClipSeg为主数据；DeHate保持external validation。同一架构/损失/推断流程，不按语料选骨干。

**两主数据各三seed完整组合重评已经结束、回传并核验；DeHate三seed新骨干完整Optuna→完整组合评估链正在运行。任务尚未完成，无硬阻塞。** 新组合自动停止相对同复制设置固定8次：HateMM AP/ROC略升、within降.005035；HCS三项均降（AP−.006997/ROC−.009286/within−.007089）。未显示稳定省调用或跨语料保持性能；相对旧r5同复制/停止设置两主数据三项均值均下降。负结果如实保留，DeHate既定预算继续跑满，不因部分结果换规则/模型。

两主数据复用初版各seed234/2025/3407完整20-trial×50 epoch训练搜索的既定test-selected与validation-selected checkpoint；复制/停止不改变训练损失，不重复相同训练。DeHate补同一骨干与同一搜索空间，每trial50 epoch；三seed首个完整trial分别2627/1821/2651秒（234/2025/3407），均≤1h，已各锁20 trial，随后不增减。三语料均沿用训练阶段无复制fixed8的validation AP/ROC均值选checkpoint，开发期test(AP+ROC)/2选trial；完整组合结果不参与重新选择。**这是预先选定模型的完整组合验证，不是自动停止目标的重新Optuna调参。**

一次[独立集成code review：PASS](../docs/reviews/20261002_complete_io_method_code.md)。原骨干、复制和停止统计升入src，旧入口兼容；实现等价性核对、两主数据12个checkpoint严格加载、三语料split/答案T/GT对齐通过。远端所有使用主机/语料的全输入finite/shape/coverage检查已完成并回传 `setup/inputs_<corpus>_<host>.json`；HateMM合法2秒零queryable视频保留，不删cohort。训练/推断不写data，评测器未改，无新增VLM调用或特征抽取；新视频仍需原有输入处理及实际查询成本。

## 三模块实现与缺口

| 模块 | 当前实现 | 证据与限制 |
|---|---|---|
| VLM观测与提问 | 二分时间树、EIG选节点；单问首token软答案，训练答案无标注分位8档 | 使用相同I3D/VGGish/BERT、词级转录和VLM缓存；8档是答案离散精度，与8次调用不同 |
| 弱监督骨干与先验 | `src/qtl/inside_outside.py`，初版outside/gated；秒级和节点先验、视频头、零膨胀链、原训练损失 | 初版自己的固定预算outside消融支持有限迁移novelty；HCS seed依赖，相对r5性能有代价。该消融没有在当前复制+停止组合上重新做，不能借成当前组合的机制确认 |
| 答案融合与停止 | 训练负例估计copy概率，copy同时进EIG/似然；qmixSG_rt10的validation分位参考与平均8次阈值 | 这次确实接入新骨干，校准仅用validation，test只取因果前缀；HCS停止后三项下降，调用量几乎不变 |

此前backbone固定预算迭代已经收尾：保留初版，后两修订归档；[独立novelty复查](../docs/reviews/20261002_inside_outside_backbone_novelty_recheck.md)支持窄任务迁移主张，不声称新树算法、世界首次、门控优势或相对r5涨点。当前新增任务是完整组合的三数据评估，不自动扩展新候选。

## 最新权威结果与来源

以下为**最新完整组合**，AP / ROC / within，三seed均值±总体标准差。主表使用原完整训练study的test目标选trial；逐trial checkpoint仍来自validation。这些是开发期证据。调用数为实际消费的缓存答案前缀；离线计算32次轨迹供比较，不表示部署必须执行32次。

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

HateMM短视频可提问节点不足8个，因此固定8次实际平均7.6262，自动7.8583反而更多；HCS为8.0000→7.9831，几乎未省。自动减同复制固定8次的逐seed配对均值±总体标准差：HateMM **+.002733±.003713 / +.002682±.002496 / −.005035±.005014**；HCS **−.006997±.001360 / −.009286±.003710 / −.007089±.007995**。HCS三个seed的AP/ROC均下降，不写成停止方案在新骨干上仍保持性能。

仅validation选trial的完整组合test另列（候选仍来自原test目标驱动搜索，不称独立validation-only调参）：

| 语料 | AP / ROC / within | 平均调用 |
|---|---|---|
| HateMM | 0.643795±0.018646 / 0.875095±0.001461 / 0.677516±0.055519 | 7.8723 |
| HateClipSeg | 0.643983±0.006301 / 0.616759±0.006254 / 0.547546±0.021578 | 7.6160 |

对旧r5同样的copy_neg+qmixSG_rt10，最新组合AP/ROC/within均值变化：HateMM **−.023201 / −.012166 / −.038214**；HCS **−.029783 / −.053544 / −.011952**。旧r5参考为同一设置，不能与其不含复制的固定8次表混比。

本机原始权威文件：`runs/20261002_complete_io_method/<corpus>/integrated/results/seed<seed>_trial<k>/metrics_test_{fixed0,fixed8,fixed32,qmixSG_rt10}.json`，字段 `results.score_av`。按seed234/2025/3407，HateMM原test-selected trial17/11/11、validation-selected10/2/9；HCS为11/2/10、6/17/1。每语料6个选中checkpoint共24份统一评测，合计48份，HateMM覆盖214视频/29269秒、HCS覆盖79视频/18839秒。完整审计与逐seed/配对/选择方式 `analysis/{hatemm,hateclipseg}_complete_method.json`；两主数据汇总 `runs/20261002_complete_io_method/analysis/main_corpora_complete_method.json`。本轮核对实际score前缀（沿用原6位小数写出）、重算停止阈值与调用数、无缺失/额外视频，不重写评测指标。

旧r5对照原始来源 `runs/20260929_query_paradigm_r5/stop_check/<corpus>/seed<seed>_trial<k>_r5rt/metrics_test_qmixSG_rt108.json`，HateMM trial9/4/5，HCS trial13/11/19；上述本轮审计逐项引用这些原始文件。DeHate新骨干完整结果尚未出齐，不填三seed均值，不借用旧r5 DeHate。

## 运行任务与monitor

所有任务在输入检查通过后正式运行；脚本与review已提交并同步三机。启动记录 `runs/20261002_complete_io_method/setup/launches.json`，代码同步 `setup/prelaunch_sync.json`，完整组合两主数据真实进程/同会话子进程退出核验 `setup/main_corpora_process_closure.json`。

| 任务 | 主机 | owner / monitor PID | 当前状态 |
|---|---|---|---|
| HCS三seed完整组合重评 | uoa-lab1 / sc474397 | 3470960 / 589210 | 18:34:54结束，已回传并全量核验；monitor通知后退出 |
| HateMM三seed完整组合重评 | uoa-lab3 / sc474398 | 2420804 / 589222 | 18:36:03结束，已回传并全量核验；monitor通知后退出 |
| DeHate seed234完整搜索→完整组合 | uoa-lab3 / sc474398 | 2421016 / 589266 | 已完成4/20 trial，trial4训练中（21:34核验） |
| DeHate seed2025完整搜索→完整组合 | uoa-lab1 / sc474397 | 3472559 / 592974 | 已完成5/20 trial，trial5训练中（21:34核验） |
| DeHate seed3407完整搜索→完整组合 | uoa-lab3 / sc474398 | 2423018 / 593027 | 已完成4/20 trial，trial4训练中（21:34核验） |

三项DeHate各自独立120秒monitor均已核验存活、首次RUNNING、host/identity/当前会话绑定正确；搜索完成后owner自动接该seed两种既定trial选择的完整组合评估，整个链结束才通知。没有待启动seed。本机不训练；新输入/特征抽取为0。21:34实测lab1整卡余2316MiB、利用率43%；lab3余24259MiB、利用率88%，三项就绪搜索全部在跑，不重复建任务。lab-server在开跑前检查时已有他人负载且无本项目环境/缓存；本轮无待迁移任务。

**Pursuing Goal始终paused，没有创建/恢复。** 旧35个run monitor和旧heartbeat继续关闭。新任务heartbeat PID **589267**，3小时间隔，目录 `runs/thread_monitor/01a0f639-b211-75e3-9155-e15e30534b46/20261002_complete_io_method/`，21:33首次提醒已送达，下次00:33 NZDT；状态/日志/进度检查在同目录。当前3个run monitor和新heartbeat的核验 `runs/20261002_complete_io_method/setup/monitor_health.json`。全部完成、原始评测核验回传并汇报后关闭本轮heartbeat。两主数据延迟通知均已处理：再次核对各owner及全部同会话进程已退出、本机各24份原始评测与已完成审计仍齐全；DeHate三条链实际存活，记录 `setup/{hcs,hatemm}_notification_followup.json`。延迟的旧通知不重复启动任务。

本轮仅新增独立lab1启动入口和文档时进行增量同步，未在活动训练中替换模型/训练/推断实现；最终汇报前再运行 `bash scripts/check_layout.sh`，结果在 `setup/current_sync.json`。已知无关tandem.html、lab1 idea-stage/及home STRAY保留。研究规则、AGENTS/CLAUDE未改。

## 下一步

1. 三项DeHate搜索各自已锁20 trial，继续完整预算、每trial50 epoch跑满。21:34已完成13个trial全部回传，history、validation选epoch和169份训练阶段统一评测核验通过，记录 `runs/20261002_complete_io_method/analysis/dehate_interim_training_audit.json`；这不是最终复制+停止结果。实际进程/SQLite/预算在 `setup/dehate_progress.json`。每条链搜索结束自动接完整组合重评，不另起任务。
2. 完成通知后核验真实owner/同会话子进程、Optuna SQLite与全部epoch/checkpoint/统一评测，回传本机；新骨干三语料结果齐全后汇总三主指标、总体标准差、调用量和同设置固定8次差值，同时报告validation-selected。
3. 用户本次请求完成前不宣告整体完成；不把初版novelty收尾当成本次任务完成，不扩展新候选或修改停止配置。

## 历史与规则

[本轮运行协议与结果](../experiments/20261002_complete_io_method/README.md)、[此前backbone收尾](../archive/research-wiki/STATUS_20261002_backbone_closed.md)、[初版完整搜索及自身消融](../experiments/20261001_inside_outside_backbone/README.md)、[研究规则](../RESEARCH_ITERATION_RULES.md)。详细过程留各实验README，STATUS只保留当前事实。
