# 当前研究状态

截至 **2026-09-06 06:29 NZST**。依据：候选8独立提案评审GO、首版实现、两机完整train裁定/基础cohort核对，唯一code review进行中。既有权威数字均指向本机 runs 原始评测。

## 当前目标与结论

**目标未完成，无硬阻塞。** 两语料确认级SOTA、VLM/骨干/融合三个模块各有有效性支持、整体统一范式、尽少方法超参数，缺一不可。沿现行规则：validation选checkpoint，test(AP+ROC)/2选trial；within仍是下限，不私改研究规则。

候选5两语料通过三seed数值确认，但HCS三个核心模块未通过三seed有效性要求，保留性能参照。候选6完整结束并归档：HateMM 20trial全部within剪枝、没有合格best；HCS虽然过单语料数值门，但新骨干/训练贡献未成立。**不追加这两候选的旧搜索或确认seed。**

**候选7已按用户指令停止并归档，不恢复抽取或训练。** 每窗口四次VLM、每视频120次调用，计算成本过高；收益尚未验证，不能记成性能失败。HateMM302/1068、HCS316/393份完整视频缓存已回传解析，未开训、无候选7性能数字。[停止审计](../runs/20260906_context_witness/cancellation_audit.json)、[归档说明](../archive/experiments/20260906_context_witness/README.md)。当前推进候选8，候选7不恢复。

## 当前方法：候选8的三个模块

[带噪窗口证据监督的局部事件强度学习](../experiments/20260906_censored_evidence_process/README.md)。独立[proposal review GO](../experiments/20260906_censored_evidence_process/REVIEW_RULE4.md)，首版实现完成、一次code review进行中，尚无训练数字。

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
| C8 code review | 独立agent code_review_c8审查中；不做smoke或缩短训练 | experiments/20260906_censored_evidence_process/ |
| C8 HateMM/HCS seed234 | 计划lab1/lab3并行完整搜索；输入已齐，待code review GO及同步 | runs/20260906_censored_evidence_process/<corpus>/seed234/ |
| C7 | 两机抽取和对应monitor已停止；302/316份缓存保留，不恢复或补训 | [停止审计](../runs/20260906_context_witness/cancellation_audit.json) |
| GPU | 06:27 lab1/lab3均空闲、各约31GB可用；已有输入满足准备条件 | 下一正式任务为C8两语料seed234 |
| 长期目标monitor | PID1177638存活，总体目标未完成 | [状态与日志](../runs/thread_monitor/01a06df5-3e92-79b0-be30-820db943e551/) |

新长搜索启动时各配独立monitor，先确认实际进程和首次观察成功再记录PID。不等待模型轮询；任务完成后核验全输出并回传。C7旧通知均不授权重启。

多机同步：06:27本机/lab1/lab3 commit均为2f9a159（仅同步用途）；本机新增C8相关代码待提交同步。CLAUDE.md既有修改、tandem.html及lab1 idea-stage/repro_t3al保留，均无关运行；家目录无STRAY。用户计算成本要求已在AGENTS.md，CLAUDE.md和研究规则未改。

## 下一步

1. 完成候选8唯一代码审查并修复结论级bug；同步后双机完整seed234搜索，不新增VLM抽取。
2. 首trial50epoch+val checkpoint+test实测冻结20/5预算，按test选trial、within约束沿现行规则。两语料都过筛后再补确认seed。
3. 验证三个主替换与no_vlm；同输入最强baseline和整体方法证据未齐前不宣称完成。显著增耗方案先说明必要性及廉价替代，不再默认堆VLM调用。

## 资料

[研究规则](../RESEARCH_ITERATION_RULES.md)（不改写）、[固定baseline表](../docs/duplex/OFFICIAL_VAL_RESULTS.md)、[评测协议](../docs/duplex/FRAME_EVAL_PROTOCOL.md)、[旧状态索引](../archive/research-wiki/STATUS_20260905_before_cleanup.md)。C5缓存完整：HMM1068视频/2136文件、HCS393视频/786文件，出处在 data/interventional_evidence/<corpus>/PROVENANCE.md。冻结的Hate-follow-up引用不动。
