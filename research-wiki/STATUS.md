# 当前研究状态

截至 **2026-09-06 07:01 NZST**。依据：候选8 HCS20trial完整回传审计与保存预测诊断；HateMM仍运行固定20trial搜索。权威数字均指向本机 runs 原始评测。

## 当前目标与结论

**目标未完成，无硬阻塞。** 两语料确认级SOTA、VLM/骨干/融合三个模块各有有效性支持、整体统一范式、尽少方法超参数，缺一不可。沿现行规则：validation选checkpoint，test(AP+ROC)/2选trial；within仍是下限，不私改研究规则。

候选5两语料通过三seed数值确认，但HCS三个核心模块未通过三seed有效性要求，保留性能参照。候选6完整结束并归档：HateMM 20trial全部within剪枝、没有合格best；HCS虽然过单语料数值门，但新骨干/训练贡献未成立。**不追加这两候选的旧搜索或确认seed。**

**候选7已按用户指令停止并归档，不恢复抽取或训练。** 每窗口四次VLM、每视频120次调用，计算成本过高；收益尚未验证，不能记成性能失败。HateMM302/1068、HCS316/393份完整视频缓存已回传解析，未开训、无候选7性能数字。[停止审计](../runs/20260906_context_witness/cancellation_audit.json)、[归档说明](../archive/experiments/20260906_context_witness/README.md)。当前推进候选8，候选7不恢复。

## 当前方法：候选8的三个模块

[带噪窗口证据监督的局部事件强度学习](../experiments/20260906_censored_evidence_process/README.md)。独立[proposal review GO](../experiments/20260906_censored_evidence_process/REVIEW_RULE4.md)、[code review GO](../experiments/20260906_censored_evidence_process/REVIEW_RULE6.md)。**HCS20trial全部within剪枝，无合格best；不追加HCS确认seed或消融。** HateMM完成固定预算后再按规则9分流，不在活动训练中移动代码。

| 模块 | 实现 | 待验证 |
|---|---|---|
| VLM观测 | 仅用train的原始单窗口Qwen K30/K4裁定，学习尺度相关假阳率/灵敏度 | 对hard_observation和no_vlm；噪声参数不声称真实可辨识 |
| 内容骨干 | 1920维基础内容、轻量时间卷积；总强度与局部分配显式分离 | 对unfactorized；不能把softmax/卷积本身称创新 |
| 融合/监督 | 视频及粗细窗口的至少一次事件概率，带噪观察复合似然；推断只读同一内容模型局部强度 | 对topk_event；是复合似然，不是嵌套观察精确联合似然 |

**当前新增VLM抽取=0；部署新视频VLM调用=0。** 复用最初模块1完整缓存，不用C5/C7四路观察；原训练数据34次/视频的历史VLM成本仍需披露。I3D/VGGish/BERT基础特征预处理仍存在，不能称端到端零成本。只搜索lr/dropout/max_seqlen，其余固定设计参数照实报告。

代码在 experiments/20260906_censored_evidence_process/；时间积分单元在 src/temporal_measure.py，沿现有uniform采样建立覆盖全视频的区间，避免采样密度改变定位分数。VLM仅进入train loss，不进forward或val/test。唯一评测器未改。三个模块和整体有效性均待完整结果，不提前声称novel paradigm成立。

## 最新权威结果

以下顺序均为 **AP / ROC / within**，全部属于按test选trial的开发期结果。

| 候选/语料 | 结果 | 结论与本机来源 |
|---|---|---|
| C8 HateMM 首trial0（非最终） | .581852/.773846/.582817，epoch1 | 完整50epoch、within剪枝；20trial搜索继续；[原评测](../runs/20260906_censored_evidence_process/hatemm/seed234/trial0/metrics.json) |
| C8 HCS seed234 | 无合格trial；无within约束的诊断trial7为 .604427/.589883/.510205，epoch2 | 20/20完整50epoch后PRUNED，within最高.519747<.524；[审计](../runs/20260906_censored_evidence_process/hateclipseg/seed234/artifact_audit.json)、[原评测](../runs/20260906_censored_evidence_process/hateclipseg/seed234/trial7/metrics.json) |
| C6 HateMM seed234 | 无合格trial；无within约束的诊断trial8为 .601467/.817607/.589623 | 20/20完整训练后PRUNED，within范围 .523138–.606444，低于.632；[审计](../runs/20260905_latent_evidence_sequence/hatemm/seed234/artifact_audit.json)、[trial8原评测](../runs/20260905_latent_evidence_sequence/hatemm/seed234/trial8/metrics.json) |
| C6 HCS seed234 | .690827/.664875/.580322，trial17/epoch2 | 20/20 COMPLETE；[审计](../runs/20260905_latent_evidence_sequence/hateclipseg/seed234/artifact_audit.json)、[原评测](../runs/20260905_latent_evidence_sequence/hateclipseg/seed234/trial17/metrics.json) |
| C6 HCS初始化参照 | .692374/.666562/.584348 | train统计初始化、未经梯度优化，与完整模型相当；[原评测及完整79视频审计](../runs/20260905_latent_evidence_sequence/diagnostics/hcs_seed234_initialization/) |
| C5 HateMM三seed | .631307±.007170 / .845751±.005332 / .659939±.010262 | 数值确认通过；[汇总及各原评测路径](../runs/20260905_interventional_evidence/hatemm/confirmation_summary.json) |
| C5 HCS三seed | .654090±.004189 / .638275±.005749 / .557700±.019617 | 数值确认通过，核心模块证据不通过；[汇总及各原评测路径](../runs/20260905_interventional_evidence/hateclipseg/confirmation_summary.json) |

C6 HCS主替换：对角观测 .6772/.6608/.5717，静态转移 .6936/.6650/.5888，事件训练换top-k .6877/.6585/.5825；后两者不支持有效性，去观测NLL也近似不变。[八臂审计及原评测路径](../runs/20260905_latent_evidence_sequence/ablations/hateclipseg/seed234/artifact_audit.json)。C6两语料40trial、HCS八消融及初始化输出均已回传；[最终汇总](../runs/20260905_latent_evidence_sequence/final_audit.json)、[归档结论](../archive/experiments/20260905_latent_evidence_sequence/README.md)。

C5两语料120trial和已有消融全部回传；HCS三个核心替换在seed3407均略升，融合平均AP增益不足.01，故不能声明三个模块有效。[三seed消融汇总](../runs/20260905_interventional_evidence/ablations/hateclipseg/three_seed_summary.json)。HateMM单seed消融不抵消HCS失败，不再重复补跑。

仅validation排序的零额外训练参考、不作门：C8 HCS仍为上述trial7（PRUNED）；C6 HMM trial1 .573789/.795177/.569116（仍PRUNED），HCS trial0 .685846/.663355/.581215；C5三seed均值HMM .6133/.8383/.6634，HCS .6516/.6358/.5566。各study_summary.json/confirmation_summary.json存选择依据。
可靠性能起点C1三seed：HMM .657±.013/.842±.005/.646±.004，HCS .699±.006/.681±.016/.553±.007；仅validation排序 .601/.821/.627 与 .686/.665/.538。[原始study与评测](../runs/20260903_hier_evidence_mil/)。C4负结果见[归档](../archive/experiments/20260904_null_token_cma/README.md)。

## 运行任务与监控

| 任务 | 当前状态 | 位置 |
|---|---|---|
| C8 HateMM seed234，lab1 | PID/PGID2352168；07:01已完成14/20，均PRUNED；仍运行，monitor1713728存活 | [预算](../runs/20260906_censored_evidence_process/hatemm/seed234/budget.json)、[monitor](../runs/20260906_censored_evidence_process/hatemm/seed234/monitor/run.log) |
| C8 HCS seed234，lab3 | 20/20结束、进程退出，结果全部回传；monitor已通知并退出，本次已处理 | [审计](../runs/20260906_censored_evidence_process/hateclipseg/seed234/artifact_audit.json)、[monitor](../runs/20260906_censored_evidence_process/hateclipseg/seed234/monitor/run.log) |
| C7 | 两机抽取和对应monitor已停止；302/316份缓存保留，不恢复或补训 | [停止审计](../runs/20260906_context_witness/cancellation_audit.json) |
| GPU | lab1正式搜索；lab3已空闲；本机有他人任务；lab-server有miniconda但无HateVideo环境/项目 | 没有已满足条件的确认/消融，不为占满GPU重复训练 |
| 长期目标monitor | PID1177638存活，总体目标未完成 | [状态与日志](../runs/thread_monitor/01a06df5-3e92-79b0-be30-820db943e551/) |

HateMM搜索及monitor与SSH解耦，继续等待完成事件，不模型轮询。HCS20trial覆盖val63/test79全部正常，既有缓存无新增抽取。C7旧通知均不授权重启。

多机同步：C8启动时本机/lab1/lab3 commit均为89472f1（仅同步用途），无影响运行的脏代码或未跟踪代码。CLAUDE.md既有修改、tandem.html及lab1 idea-stage/repro_t3al保留，均无关运行；家目录无STRAY。启动后只同步运行文档及不被训练导入的CPU保存预测诊断，不替换活动训练代码。用户计算成本要求已在AGENTS.md，CLAUDE.md和研究规则未改。

## 下一步

1. 接HateMM完成事件后核验进程、回传并审计全部20trial/50epoch/ckpt/评测覆盖，再按规则9分流；异常先诊断，不重复启动。
2. HCS已做20trial保存预测诊断：非恒定输出、噪声通道未塌缩，但局部排序接近随机；trial7在67个混合视频中32个AUC<.5，C1为25个。[诊断及来源](../runs/20260906_censored_evidence_process/error_analysis/hcs_seed234_saved_predictions.json)、[全20trial](../runs/20260906_censored_evidence_process/error_analysis/hcs_seed234_all_trials.json)。不能只归咎within门，也没有单模块因果结论。
3. 下一候选优先保留单次VLM局部证据，研究内容与证据交互；部署VLM=0不是用户硬要求。先完成具体提案与独立评审，不恢复四遍抽取、不仓促追加损失或确认搜索。显著增耗先说明成本及廉价替代；三个模块和同输入最强baseline证据未齐，不宣称完成。

## 资料

[研究规则](../RESEARCH_ITERATION_RULES.md)（不改写）、[固定baseline表](../docs/duplex/OFFICIAL_VAL_RESULTS.md)、[评测协议](../docs/duplex/FRAME_EVAL_PROTOCOL.md)、[旧状态索引](../archive/research-wiki/STATUS_20260905_before_cleanup.md)。C5缓存完整：HMM1068视频/2136文件、HCS393视频/786文件，出处在 data/interventional_evidence/<corpus>/PROVENANCE.md。冻结的Hate-follow-up引用不动。
