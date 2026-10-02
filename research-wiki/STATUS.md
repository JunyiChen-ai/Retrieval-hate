# 当前研究状态

截至 **2026-10-02 17:34 NZDT**。依据：本机 `runs/` 统一评测器原始输出、完整study/训练审计、初版独立novelty复查；模型与评测协议未改。完整跨版本统计入口 `runs/20261002_residual_io_backbone/analysis/backbone_completion_readiness.json`。本轮前状态已[归档](../archive/research-wiki/STATUS_20261002_before_backbone_completion.md)。

## 当前目标与结论

**本轮backbone novelty目标按用户“优先novel且涨点；充分尝试仍无提升时优先有证据novelty”的授权完成，自动迭代结束。** 初版、加性统计量版、保留CMA的残差版均已完成HateMM/HateClipSeg各三seed独立20-trial搜索，每trial完整50 epoch，合计360个完整trial；各自结构诊断也已完成并回传。三版均未实现相对r5的主操作点整体涨点，不能写成性能提升。within与pooled AP/ROC并列主指标。

最终保留[初版inside–outside骨干](../experiments/20261001_inside_outside_backbone/README.md)：在固定时间问题树上组合区间内表示和区间外上下文，形成秒级/节点内容先验，通过现有含噪答案边缘似然训练。新增VLM调用与特征抽取为0；新视频原有输入处理和提问成本仍存在，没有已测量的速度优势。[proposal GO](../docs/reviews/20261001_inside_outside_backbone_proposal.md)、[code PASS](../docs/reviews/20261001_inside_outside_backbone_code.md)及[独立novelty复查](../docs/reviews/20261002_inside_outside_backbone_novelty_recheck.md)支持有限的任务迁移贡献；不主张世界首次、递归门控优势或普遍稳定增益。outside排除的是目标区间缓存行，不能声称排除原始编码器感受野重叠。

初版自身outside消融在两主数据达到平均AP贡献要求，但HCS效应有明显seed依赖，且初版相对r5有性能代价。[加性版](../archive/experiments/20261001_associative_io_backbone/README.md)与[残差版](../archive/experiments/20261002_residual_io_backbone/README.md)归档，所有原始runs保留；它们的失败不替换或抹去初版自己的证据。r5保留为性能参照，没有按语料挑版本或ensemble。本轮完成不等于整篇论文和其它模块全部验证完成。

## 三模块实现与缺口

| 模块 | 当前实现 | 证据与限制 |
|---|---|---|
| VLM观测与提问 | 二分时间树、期望信息增益选节点、词级转录、单问首token软答案，8档 | 复用现有缓存；16档仅敏感性分析，不混同于调用次数 |
| 弱监督骨干与先验 | 最终novelty实现 `experiments/20261001_inside_outside_backbone/`；共享基础设施 `src/qtl/` | 同一架构用于两主数据，自身outside有平均贡献；相对r5掉点，HCS有seed依赖，无门控胜过均值的证据 |
| 答案融合与停止 | 锚定答案模型、复制似然、分位数组合停止 `qmixSG_rt10` | 已实现自动停止；8档平均调用约7.8–8.1，尚未证明稳定省调用；validation直接选阈值仍有调用量/性能权衡 |

停止实验的现有结论保留：8档主线不变，DeHate仅外部验证；16档DeHate附加检查within差值−.00503，未通过−.005容差。validation选阈值在HCS/DeHate的改善主要伴随更多调用，HateMM降到2.45次时三项均下降。代码、权威结果与详细比较见[r5 README第17节](../experiments/20260925_query_paradigm/README.md)，本轮不扩展此任务。

## 最新权威结果与来源

以下顺序 **AP / ROC / within**，固定8次为主操作点，三seed均值±总体标准差。Optuna按开发期test(AP+ROC)/2选trial；每trial的checkpoint仍由validation选。这些结果是开发期证据，不是未揭盲的独立确认。固定配置诊断不计入Optuna三seed均值。

| 版本 | HateMM | HateClipSeg |
|---|---|---|
| r5性能参照 | 0.690871±0.013681 / 0.884857±0.002060 / 0.772915±0.014386 | 0.686455±0.003656 / 0.673725±0.005441 / 0.560226±0.013556 |
| 初版（最终保留novelty） | 0.676491±0.010167 / 0.879176±0.001587 / 0.737845±0.014984 | 0.668398±0.003274 / 0.642595±0.005721 / 0.560326±0.004923 |
| 加性版（归档） | 0.670993±0.025027 / 0.875651±0.002628 / 0.761282±0.008523 | 0.681333±0.001454 / 0.653216±0.008294 / 0.526091±0.021952 |
| 残差版（归档） | 0.680315±0.004295 / 0.875683±0.001307 / 0.762926±0.013352 | 0.679954±0.005602 / 0.672929±0.002589 / 0.547479±0.016958 |

初版对r5固定8次均值变化：HateMM **−.014380 / −.005681 / −.035070**；HCS **−.018057 / −.031130 / +.000100**。残差版对应变化：HateMM **−.010556 / −.009174 / −.009989**；HCS **−.006501 / −.000796 / −.012747**。HCS ROC的细小变化不作显著性结论。

最后两项HateMM残差搜索已于17:20:29/30结束，真实主进程及全部同会话非僵尸进程退出；seed2025/3407各20/20 COMPLETE、全部50 epoch、实际checkpoint、SQLite参数/目标与各260份统一评测已核验并回传。残差版合计120 trial、1560份统一test评测；另234份固定配置诊断独立统计。

| 残差版完整三seed | 固定0次 | 固定8次 |
|---|---|---|
| HateMM | 0.548299±0.017447 / 0.774238±0.008364 / 0.734028±0.008749 | 0.680315±0.004295 / 0.875683±0.001307 / 0.762926±0.013352 |
| HateClipSeg | 0.655020±0.013074 / 0.645167±0.017515 / 0.526429±0.022377 | 0.679954±0.005602 / 0.672929±0.002589 / 0.547479±0.016958 |

同时报告仅按validation选trial的固定8次test结果（候选仍来自test目标驱动的搜索，不称独立validation-only调参）：

| 版本 | HateMM | HateClipSeg |
|---|---|---|
| r5 | 0.665761±0.005629 / 0.868486±0.008982 / 0.743114±0.023990 | 0.679647±0.002760 / 0.649995±0.012300 / 0.567823±0.027759 |
| 初版 | 0.640230±0.018754 / 0.868892±0.001408 / 0.681005±0.060963 | 0.652083±0.002717 / 0.620540±0.009350 / 0.561490±0.012623 |
| 加性版 | 0.637346±0.015603 / 0.861325±0.014171 / 0.722427±0.028676 | 0.662610±0.022670 / 0.632999±0.013865 / 0.521973±0.016386 |
| 残差版 | 0.610144±0.034483 / 0.855722±0.007400 / 0.749190±0.007806 | 0.656493±0.024220 / 0.628403±0.034775 / 0.549688±0.015010 |

按seed234/2025/3407，初版HateMM test选trial17/11/11、validation选10/2/9，HCS为11/2/10、6/17/1；残差版HateMM为10/11/15、5/9/16，HCS为19/2/19、16/2/9。全部版本逐seed、选择编号、0/8/32次和标准差均在跨版本统计入口，32次只作附加结果。

原始权威路径：`runs/<exp_id>/<corpus>/seed<seed>/trial<k>/metrics_test_fixed<次数>.json`，字段 `results.score_av`；experiment分别为 `20260929_query_paradigm_r5`、`20261001_inside_outside_backbone`、`20261001_associative_io_backbone`、`20261002_residual_io_backbone`。各study的 `study_summary.json` 记录选择；三版新骨干各自的 `analysis/two_corpus_three_seed_search.json` 与逐seed审计记录完整性，跨版本入口逐项指向原始来源。

**本版本自身机制证据**：初版去outside固定8次配对均值变化，HateMM **−.024767 / −.006117 / −.009780**，HCS **−.012112 / −.009322 / −.016624**，两语料平均AP下降≥.01；HCS的AP主要来自seed234，另外两seed反向，不能声称每seed稳定改善。初版HCS均值组合未输于门控，故不把门控当独立贡献。来源 `runs/20261001_inside_outside_backbone/analysis/two_corpus_outside_diagnosis.json`，原始文件为同版 `diagnostics/<corpus>/<arm>/seed<seed>/metrics_test_fixed8.json`，full seed234分别复用HateMM trial17/HCS trial11。

残差版去outside：HateMM **−.016019 / −.006629 / −.015378**、HCS **+.003000 / −.002068 / +.017037**；去整体残差：HateMM **+.001804 / −.000086 / +.009348**、HCS **+.011532 / +.001434 / −.004776**。outside只在HateMM达到贡献要求，整体残差两语料均未达到；不借用初版证据。锁定配置为HateMM trial10、HCS trial19，来源 `runs/20261002_residual_io_backbone/analysis/two_corpus_{outside,residual}_diagnosis.json`，逐seed配对、标准差及原始评测路径在各语料 `analysis/<corpus>_locked_diagnostics.json`。**所有这些固定配置诊断均不是Optuna确认。**

## 运行任务与monitor

**无活动或待启动的本轮实验，完成通知已全部处理。Pursuing goal保持paused，本轮没有创建或恢复Goal。** 初版/加性版此前已全量回传核验；残差版12项搜索/诊断真实owner及同会话子进程均已结束，最终进程记录 `runs/20261002_residual_io_backbone/setup/final_process_closure.json`。

本轮三版全部35个独立run monitor均已成功通知并退出。原3小时heartbeat（会话 `01a0f639-b211-75e3-9155-e15e30534b46`，PID `3245799`）已按用户“完成则关闭monitor”的指示停止，实际停止时间 **2026-10-02T17:32:49.260670+13:00**；STOP、状态与日志在 `runs/thread_monitor/01a0f639-b211-75e3-9155-e15e30534b46/`。最终monitor核验 `runs/20261002_residual_io_backbone/setup/final_monitor_closure.json`。迟到通知只复核现有结果，不重启旧任务。

最终代码/文档同步与三机工作树检查记录 `runs/20261002_residual_io_backbone/setup/backbone_final_sync.json`；汇报前执行 `bash scripts/check_layout.sh`。归档仅修正源码位置和入口路径，Python解析、shell语法和共享src定位检查通过，不启动训练。已知无关 `tandem.html`、lab1 `idea-stage/`及家目录STRAY不改动、不清理。

## 下一步

当前授权的backbone自动迭代已收尾，无待运行项；保留初版可运行实现、完整搜索/消融、独立复查和性能代价供方法整理。论文主表口径及其余模块验证不在本次完成声明内，不自动扩展实验。

## 历史与规则

[初版结论与收尾](../experiments/20261001_inside_outside_backbone/README.md)、[残差最终完整结果](../archive/experiments/20261002_residual_io_backbone/README.md)、[收尾前状态](../archive/research-wiki/STATUS_20261002_before_backbone_completion.md)、[研究规则](../RESEARCH_ITERATION_RULES.md)、[固定baseline表](../docs/duplex/OFFICIAL_VAL_RESULTS.md)。详细过程保留在各实验README与归档，不在STATUS追加时间线。
