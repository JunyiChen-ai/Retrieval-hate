# 当前研究状态

截至 **2026-09-06 07:25 NZST**。依据：候选8两语料完整40trial回传审计；候选9双机正式seed234搜索已启动，进程/首epoch/monitor首次RUNNING均已核验。权威数字均引用本机runs原评测。

## 当前目标与结论

**目标未完成，无硬阻塞。** 仍需两语料确认级SOTA、VLM/骨干/融合三个模块有效性与novelty、整体统一方法、尽少方法超参数。validation选checkpoint，test(AP+ROC)/2选trial；within下限沿现行研究规则，不擅改。

C8两语料各20trial全部完整50epoch后within剪枝，没有合格best；按规则9归档，不追加确认/消融。C5数值确认通过但HCS核心模块有效性失败；C1仍是可靠性能起点。C7因每窗口四次VLM成本被用户停止，不恢复，其缓存保留不表示授权补训。

## 当前方法：候选9三个模块

[区间证据的内容条件分配](../experiments/20260906_interval_evidence_transport/README.md)，独立[proposal GO](../experiments/20260906_interval_evidence_transport/REVIEW_RULE4.md)、[code GO](../experiments/20260906_interval_evidence_transport/REVIEW_RULE6.md)。实现位于该实验目录，共享时间输入/归一化在src/interval_observation_data.py；唯一评测器未改。

| 模块 | 当前实现 | 必须补的证据 |
|---|---|---|
| M1 VLM观察 | 原Qwen四等级、内容prior、序数噪声通道Bayes更新，构成区间query | 对hard_observation；原累计码与类别embedding等价，未将其当创新实现 |
| M2 骨干 | 区间内内容条件分配，同一分配做双向跨模态消息交换 | 对uniform_assignment；检验共享分配整体，不称attention首创 |
| M3 融合 | 同一条件分类器，按区间分配边缘化得到最终帧概率 | 对additive_readout；检验联合读出，不拆成两个未经验证的贡献 |

**新增VLM抽取0；新视频34次调用，与原模块1相同。** 不使用C5/C7四路干预，也不恢复已失败的qwenctx抽取。保留局部VLM输入，部署VLM=0不是用户硬要求。分配O(34TH)，内容投影O(TH²)；读出第一层精确因式分解，避免34倍重复大矩阵乘，不预先承诺吞吐。只搜lr/dropout/max_seqlen；固定参数仍在README披露。三模块有效性与整体novelty均未成立，不提前宣称完成。

## 最新权威结果

顺序 **AP / ROC / within**；全部为按test选trial的开发期证据。

| 候选/语料 | 结果 | 结论及本机来源 |
|---|---|---|
| C9两语料 seed234 | 暂无完整trial test结果 | 07:24双机开跑完整50epoch，不以中途validation数值代替test结论 |
| C8 HateMM seed234 | 无合格best；无约束诊断trial11 .593962/.786366/.608225 | 20/20 PRUNED，within最高.608225<.632；[原评测](../runs/20260906_censored_evidence_process/hatemm/seed234/trial11/metrics.json)、[审计](../runs/20260906_censored_evidence_process/hatemm/seed234/artifact_audit.json) |
| C8 HCS seed234 | 无合格best；无约束诊断trial7 .604427/.589883/.510205 | 20/20 PRUNED，within最高.519747<.524；[原评测](../runs/20260906_censored_evidence_process/hateclipseg/seed234/trial7/metrics.json)、[审计](../runs/20260906_censored_evidence_process/hateclipseg/seed234/artifact_audit.json) |
| C5 HateMM三seed | .631307±.007170/.845751±.005332/.659939±.010262 | 数值确认通过；[汇总及原评测路径](../runs/20260905_interventional_evidence/hatemm/confirmation_summary.json) |
| C5 HCS三seed | .654090±.004189/.638275±.005749/.557700±.019617 | 数值确认通过，核心模块不成立；[汇总及原评测路径](../runs/20260905_interventional_evidence/hateclipseg/confirmation_summary.json) |
| C1三seed性能起点 | HMM .657±.013/.842±.005/.646±.004；HCS .699±.006/.681±.016/.553±.007 | [原始搜索及评测](../runs/20260903_hier_evidence_mil/) |

仅validation排序的零额外训练参考、不作门：C8 HMM trial19 .579398/.771111/.585330；HCS仍trial7。C5 HMM .6133/.8383/.6634、HCS .6516/.6358/.5566；C1 HMM .601/.821/.627、HCS .686/.665/.538。各study_summary/confirmation_summary存选择依据。

C8保存预测诊断：HCS不是常数输出、观察通道未塌缩，但局部排序弱，trial7在67个混合视频中32个AUC<.5，C1为25个；pooled也明显较低，不能仅归因within门。[诊断与源路径](../runs/20260906_censored_evidence_process/error_analysis/hcs_seed234_saved_predictions.json)。两语料40trial均已实际解析预测、检查ID/帧长/finite；未做模块消融，不能单独归罪某一模块。[归档结论](../archive/experiments/20260906_censored_evidence_process/README.md)。

## 运行任务与monitor

| 任务 | 当前状态 | 位置 |
|---|---|---|
| C9 HateMM seed234，lab1 | PID/PGID3125429；正式trial0已出epoch1；monitor1734844存活、首次RUNNING成功 | [输出](../runs/20260906_interval_evidence_transport/hatemm/seed234/)、[monitor](../runs/20260906_interval_evidence_transport/hatemm/seed234/monitor/run.log) |
| C9 HCS seed234，lab3 | PID/PGID3530862；正式trial0已出epoch5；monitor1734851存活、首次RUNNING成功 | [输出](../runs/20260906_interval_evidence_transport/hateclipseg/seed234/)、[monitor](../runs/20260906_interval_evidence_transport/hateclipseg/seed234/monitor/run.log) |
| C8两语料 | 全部结束/回传/审计，两个monitor已通知退出；通知已处理，不重启 | [HMM monitor](../runs/20260906_censored_evidence_process/hatemm/seed234/monitor/run.log)、[HCS monitor](../runs/20260906_censored_evidence_process/hateclipseg/seed234/monitor/run.log) |
| 资源 | lab1/lab3两语料并行搜索；本机他人GPU任务约97%；lab-server空闲但无HateVideo环境/项目 | 启动瞬时GPU利用率约33/34%，未声称满载；无额外合格确认任务，不重复训练填GPU |
| 长期目标monitor | PID1177638存活，07:04通知已处理；目标未完成，保留 | [日志](../runs/thread_monitor/01a06df5-3e92-79b0-be30-820db943e551/run.log) |

多机同步：C9启动前本机/lab1/lab3均bf8c201（仅同步用途），运行代码无脏文件或相关未跟踪文件、家目录无STRAY。两机torch2.7.1+cu128/transformers4.49对齐，原VLM两尺度各split全覆盖。CLAUDE.md既有修改、tandem.html、lab1 idea-stage/repro_t3al属于无关既有工作，保留；未改CLAUDE.md和研究规则。归档及共享逻辑迁移在C8所有进程结束后进行；C9开跑后只更新文档，不替换活动训练代码。

## 下一步

1. C9两机及monitor均已与SSH解耦，保留完成事件；收到通知先核验进程和原输出、回传审计，不重启旧任务。
2. 首完整trial实测由search自动冻结20/5预算至各seed234/budget.json，收到首trial完整输出后转录README；每trial完整50epoch，按test选trial，不增加验证排序搜索。不模型轮询等待。
3. 两语料过筛才确认seed；三个主消融、no_vlm及同输入最强baseline证据未齐，不宣称SOTA/目标完成。显著增耗先说明成本和廉价替代，不堆VLM调用。

## 资料与历史

[研究规则](../RESEARCH_ITERATION_RULES.md)、[固定baseline表](../docs/duplex/OFFICIAL_VAL_RESULTS.md)、[评测协议](../docs/duplex/FRAME_EVAL_PROTOCOL.md)。负结果细节：[C6](../archive/experiments/20260905_latent_evidence_sequence/README.md)、[C7用户取消](../archive/experiments/20260906_context_witness/README.md)、[C4](../archive/experiments/20260904_null_token_cma/README.md)。[旧状态索引](../archive/research-wiki/STATUS_20260905_before_cleanup.md)。冻结Hate-follow-up引用不动；缓存出处见各data子目录PROVENANCE.md。
