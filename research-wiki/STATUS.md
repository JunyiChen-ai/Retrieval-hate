# 当前研究状态

截至 **2026-09-06 01:14 NZST**。依据：候选6两语料全量回传审计、候选7首版实现及独立提案评审。权威数字均指向本机 runs 原始评测。

## 当前目标与结论

**目标未完成，无硬阻塞。** 两语料确认级SOTA、VLM/骨干/融合三个模块各有有效性支持、整体统一范式、尽少方法超参数，缺一不可。沿现行规则：validation选checkpoint，test(AP+ROC)/2选trial；within仍是下限，不私改研究规则。

候选5两语料通过三seed数值确认，但HCS三个核心模块未通过三seed有效性要求，保留性能参照。候选6完整结束并归档：HateMM 20trial全部within剪枝、没有合格best；HCS虽然过单语料数值门，但新骨干/训练贡献未成立。**不追加这两候选的旧搜索或确认seed。**

当前推进[候选7：上下文条件化的局部证据保留/删除学习](../experiments/20260906_context_witness/README.md)。独立[proposal review GO](../experiments/20260906_context_witness/REVIEW_RULE4.md)，首版代码已实现、一次code review进行中；尚无候选7实验数字。最强baseline+同输入及整体novel paradigm证据仍缺。

## 当前三个模块

| 模块 | 候选7实现 | 必须验证的缺口 |
|---|---|---|
| VLM | 同一冻结Qwen对target/context四种可见性测量六属性；raw logits/entropy形成30维输入 | 相对同语义target-only的贡献；raw_four区分表示作用。改prompt本身不算novelty |
| 骨干 | 双向GRU排除当前位置，重建冻结内容；残差与内容/VLM共同生成局部selector | no_residual及visible_reconstruction；不能把残差+辅助loss合并效应单独归因残差 |
| 融合/训练 | 同一共享分类器对全视频、保留、删除三视图训练，最终仅输出selector q | 删除项的独立增益、避免共同作弊；最终仍需两语料三seed消融 |

代码位于当前实验目录；新输入抽取 `scripts/analysis/extract_context_witness.py`、严格解析 `src/context_witness.py`。共用训练/搜索协议位于 `src/fixed_training_protocol.py`、`src/fixed_optuna_protocol.py`，统一评测器未改。新缓存不复用C5 logits冒充上下文观测。

## 最新权威结果

以下顺序均为 **AP / ROC / within**，全部属于按test选trial的开发期结果。

| 候选/语料 | 结果 | 结论与本机来源 |
|---|---|---|
| C6 HateMM seed234 | 无合格trial；无within约束的诊断trial8为 .601467/.817607/.589623 | 20/20完整训练后PRUNED，within范围 .523138–.606444，低于.632；[审计](../runs/20260905_latent_evidence_sequence/hatemm/seed234/artifact_audit.json)、[trial8原评测](../runs/20260905_latent_evidence_sequence/hatemm/seed234/trial8/metrics.json) |
| C6 HCS seed234 | .690827/.664875/.580322，trial17/epoch2 | 20/20 COMPLETE；[审计](../runs/20260905_latent_evidence_sequence/hateclipseg/seed234/artifact_audit.json)、[原评测](../runs/20260905_latent_evidence_sequence/hateclipseg/seed234/trial17/metrics.json) |
| C6 HCS初始化参照 | .692374/.666562/.584348 | train统计初始化、未经梯度优化，与完整模型相当；[原评测及完整79视频审计](../runs/20260905_latent_evidence_sequence/diagnostics/hcs_seed234_initialization/) |
| C5 HateMM三seed | .631307±.007170 / .845751±.005332 / .659939±.010262 | 数值确认通过；[汇总及各原评测路径](../runs/20260905_interventional_evidence/hatemm/confirmation_summary.json) |
| C5 HCS三seed | .654090±.004189 / .638275±.005749 / .557700±.019617 | 数值确认通过，核心模块证据不通过；[汇总及各原评测路径](../runs/20260905_interventional_evidence/hateclipseg/confirmation_summary.json) |

C6 HCS主替换：对角观测 .6772/.6608/.5717，静态转移 .6936/.6650/.5888，事件训练换top-k .6877/.6585/.5825；后两者不支持有效性，去观测NLL也近似不变。[八臂审计及原评测路径](../runs/20260905_latent_evidence_sequence/ablations/hateclipseg/seed234/artifact_audit.json)。C6两语料40trial、HCS八消融及初始化输出均已回传；[最终汇总](../runs/20260905_latent_evidence_sequence/final_audit.json)、[归档结论](../archive/experiments/20260905_latent_evidence_sequence/README.md)。

C5两语料120trial和已有消融全部回传；HCS三个核心替换在seed3407均略升，融合平均AP增益不足.01，故不能声明三个模块有效。[三seed消融汇总](../runs/20260905_interventional_evidence/ablations/hateclipseg/three_seed_summary.json)。HateMM单seed消融不抵消HCS失败，不再重复补跑。

仅validation排序的零额外训练参考、不作门：C6 HMM trial1 .573789/.795177/.569116（仍PRUNED），HCS trial0 .685846/.663355/.581215；C5三seed均值HMM .6133/.8383/.6634，HCS .6516/.6358/.5566。各study_summary.json/confirmation_summary.json存选择依据。
可靠性能起点C1三seed：HMM .657±.013/.842±.005/.646±.004，HCS .699±.006/.681±.016/.553±.007；仅validation排序 .601/.821/.627 与 .686/.665/.538。[原始study与评测](../runs/20260903_hier_evidence_mil/)。C4负结果见[归档](../archive/experiments/20260904_null_token_cma/README.md)。

## 运行任务与监控

| 任务 | 当前状态 | 位置 |
|---|---|---|
| C7 code review | 独立agent code_review_c7审查中，尚未开训 | experiments/20260906_context_witness/ |
| C7输入抽取 | 待代码检查与多机同步后，HMM→lab1、HCS→lab3并行；尚未启动 | 计划 runs/20260906_context_witness/extract_<corpus>/ |
| C5/C6全部旧任务 | 已结束核验，通知已处理；不重复启动 | 各 runs 下 artifact_audit.json |
| GPU | 01:13实时lab1/lab3空闲，各约31GB可用；本机上次为他人任务；lab-server尚无可用项目环境 | 新输入正式抽取就绪后并行 |
| 长期会话monitor | PID1177638存活；目标未完成、无硬阻塞，保留，不重复创建 | [状态与日志](../runs/thread_monitor/01a06df5-3e92-79b0-be30-820db943e551/) |

每项新长任务配置独立monitor，进程身份/首次检查成功后更新本表；不会声称未启动任务已受监控。等待由事件唤醒，不由模型持续轮询。完成目标或确认无法推进的硬阻塞时关闭长期monitor并报告。

## 下一步

1. 处理C7唯一code review发现的结论级bug；同步代码和工作树检查后，双机正式抽取新输入并自动监控。新缓存回传后检查解析、shape、全部ID及split隔离。
2. 每语料输入就绪后完整seed234搜索；首trial实测冻结20/5预算，不做smoke。仅搜索lr/dropout/max_seqlen，其余固定设计参数照实披露，不称无超参数。
3. 按完整结果分流。先满足两语料筛选，再补确认seed；三个模块、同输入最强baseline及整体范式证据未齐前不声明完成。

## 资料

[研究规则](../RESEARCH_ITERATION_RULES.md)（不改写）、[固定baseline表](../docs/duplex/OFFICIAL_VAL_RESULTS.md)、[评测协议](../docs/duplex/FRAME_EVAL_PROTOCOL.md)、[旧状态索引](../archive/research-wiki/STATUS_20260905_before_cleanup.md)。C5缓存完整：HMM1068视频/2136文件、HCS393视频/786文件，出处在 data/interventional_evidence/<corpus>/PROVENANCE.md。冻结的Hate-follow-up引用不动。
