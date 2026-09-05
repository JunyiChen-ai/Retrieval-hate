# 当前研究状态

截至 **2026-09-06 06:12 NZST**。依据：用户叫停候选7、两机进程退出/GPU释放、302/316份缓存回传解析及停止审计。既有权威数字均指向本机 runs 原始评测。

## 当前目标与结论

**目标未完成，无硬阻塞。** 两语料确认级SOTA、VLM/骨干/融合三个模块各有有效性支持、整体统一范式、尽少方法超参数，缺一不可。沿现行规则：validation选checkpoint，test(AP+ROC)/2选trial；within仍是下限，不私改研究规则。

候选5两语料通过三seed数值确认，但HCS三个核心模块未通过三seed有效性要求，保留性能参照。候选6完整结束并归档：HateMM 20trial全部within剪枝、没有合格best；HCS虽然过单语料数值门，但新骨干/训练贡献未成立。**不追加这两候选的旧搜索或确认seed。**

**候选7已按用户指令停止并归档，不恢复抽取或训练。** 每窗口四次VLM、每视频120次调用，计算成本过高；收益尚未验证，不能记成性能失败。HateMM302/1068、HCS316/393份完整视频缓存已回传解析，未开训、无候选7性能数字。[停止审计](../runs/20260906_context_witness/cancellation_audit.json)、[归档说明](../archive/experiments/20260906_context_witness/README.md)。当前没有运行中的研究实验；下一方案尚未确定。

## 最近候选的三个模块（候选7已停止）

| 模块 | 候选7实现 | 必须验证的缺口 |
|---|---|---|
| VLM | 同一冻结Qwen对target/context四种可见性测量六属性；raw logits/entropy形成30维输入 | 相对同语义target-only的贡献；raw_four区分表示作用。改prompt本身不算novelty |
| 骨干 | 双向GRU排除当前位置，重建冻结内容；残差与内容/VLM共同生成局部selector | no_residual及visible_reconstruction；不能把残差+辅助loss合并效应单独归因残差 |
| 融合/训练 | 同一共享分类器对全视频、保留、删除三视图训练，最终仅输出selector q | 删除项的独立增益、避免共同作弊；最终仍需两语料三seed消融 |

候选7代码移至 `archive/experiments/20260906_context_witness/`，已有缓存/日志保留，不补齐、不训练。共享训练/搜索协议和唯一评测器未改。后续优先复用已完整的输入，方法设计需认真考虑成本；不能以冻结/离线为由忽略部署新视频仍需VLM推理。

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
| C7 HateMM，lab1 | 用户取消；PGID1986754及monitor1511203已停止；302份JSON回传并解析 | [停止审计](../runs/20260906_context_witness/cancellation_audit.json)、[日志](../runs/20260906_context_witness/extract_hatemm_serial/run.log) |
| C7 HCS，lab3 | 用户取消；PGID3244050及monitor1511218已停止；316份JSON回传并解析 | [停止审计](../runs/20260906_context_witness/cancellation_audit.json)、[日志](../runs/20260906_context_witness/extract_hateclipseg_serial/run.log) |
| GPU | 06:10两机抽取及相关子进程退出，GPU利用率0%，显存约185/116MiB；没有就绪的新方法实验 | 不为占GPU恢复已取消任务 |
| 长期目标monitor | 保留PID1177638；总体目标未完成，用户本次叫停的是候选7 | [状态与日志](../runs/thread_monitor/01a06df5-3e92-79b0-be30-820db943e551/) |

**候选7的旧完成/异常通知均不授权重启。** 两个对应实验monitor已关闭；不创建替代monitor，不将用户取消标记为成功完成。原batch4 OOM及后续serial运行日志均已保留；未删除任何缓存、模型或实验输出。局部缓存不足全split覆盖，不拿来冒充完整输入训练。

多机同步：停止前本机/lab1/lab3 commit均为ace4f4e（仅同步检查用途），候选7归档后同步更新；无影响运行的脏代码或STRAY。本机CLAUDE.md既有修改/tandem.html及lab1 idea-stage/repro_t3al未动。CLAUDE.md和研究规则未修改。

## 下一步

1. 不再向候选7投入GPU。先基于现有完整缓存重新设计低成本方法，说明三个模块相互作用及相对廉价基线的具体改进假设，而不是再堆VLM调用。
2. 若后续需要新增VLM输入，先明确可复用部分、调用次数和预计GPU时间；不能把更昂贵输入默认视为更好的科研方案。当前尚未选定或启动下一候选。
3. 原目标和科学评测规则保留：validation选checkpoint、test选trial，三个模块有效性与最强baseline+同输入证据未齐前不宣称完成。

## 资料

[研究规则](../RESEARCH_ITERATION_RULES.md)（不改写）、[固定baseline表](../docs/duplex/OFFICIAL_VAL_RESULTS.md)、[评测协议](../docs/duplex/FRAME_EVAL_PROTOCOL.md)、[旧状态索引](../archive/research-wiki/STATUS_20260905_before_cleanup.md)。C5缓存完整：HMM1068视频/2136文件、HCS393视频/786文件，出处在 data/interventional_evidence/<corpus>/PROVENANCE.md。冻结的Hate-follow-up引用不动。
