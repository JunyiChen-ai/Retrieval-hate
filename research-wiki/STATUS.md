# 当前研究状态

截至 **2026-09-06 08:06 NZST**。依据：候选9 HCS四个主/辅助消融完整回传审计及79视频checkpoint内部诊断；两项预声明观察诊断已启动，HateMM仍运行固定20trial。权威数字均引用本机runs原评测。

## 当前目标与结论

**目标未完成，无硬阻塞。** 仍需两语料确认级SOTA、VLM/骨干/融合三个模块有效性与novelty、整体统一方法、尽少方法超参数。validation选checkpoint，test(AP+ROC)/2选trial；within下限沿现行研究规则，不擅改。

C8两语料各20trial全部完整50epoch后within剪枝，没有合格best；按规则9归档，不追加确认/消融。C5数值确认通过但HCS核心模块有效性失败；C1仍是可靠性能起点。C7因每窗口四次VLM成本被用户停止，不恢复，其缓存保留不表示授权补训。

**C9 HCS单语料数值筛选通过，但M1/M2主替换的pooled差小于.005，未显示明确贡献；M3及VLM整体只有单seed正向信号。** HateMM尚未完成固定搜索，不提前补确认seed，不把单seed噪声当永久失败或三模块成立。

## 当前方法：候选9三个模块

[区间证据的内容条件分配](../experiments/20260906_interval_evidence_transport/README.md)，独立[proposal GO](../experiments/20260906_interval_evidence_transport/REVIEW_RULE4.md)、[code GO](../experiments/20260906_interval_evidence_transport/REVIEW_RULE6.md)。实现位于该实验目录，共享时间输入/归一化在src/interval_observation_data.py；唯一评测器未改。

| 模块 | 当前实现 | 必须补的证据 |
|---|---|---|
| M1 VLM观察 | 原Qwen四等级、内容prior、序数噪声通道Bayes更新，构成区间query | HCS full减hard_observation AP/ROC .000029/.003233，尚无明确贡献 |
| M2 骨干 | 区间内内容条件分配，同一分配做双向跨模态消息交换 | full减uniform_assignment .001201/.003406，尚无明确贡献 |
| M3 融合 | 同一条件分类器，按区间分配边缘化得到最终帧概率 | full减additive_readout .012396/.017187，仅单seed初步信号 |

**新增VLM抽取0；新视频34次调用，与原模块1相同。** 不使用C5/C7四路干预，也不恢复已失败的qwenctx抽取。保留局部VLM输入，部署VLM=0不是用户硬要求。分配O(34TH)，内容投影O(TH²)；读出第一层精确因式分解，避免34倍重复大矩阵乘，不预先承诺吞吐。只搜lr/dropout/max_seqlen；固定参数仍在README披露。三模块有效性与整体novelty均未成立，不提前宣称完成。

## 最新权威结果

顺序 **AP / ROC / within**；全部为按test选trial的开发期证据。

| 候选/语料 | 结果 | 结论及本机来源 |
|---|---|---|
| C9 HCS seed234 | .605771/.589308/.547480，trial18/epoch1 | 20/20完整，15 COMPLETE/5 PRUNED；单语料过筛；[原评测](../runs/20260906_interval_evidence_transport/hateclipseg/seed234/trial18/metrics.json)、[审计](../runs/20260906_interval_evidence_transport/hateclipseg/seed234/artifact_audit.json) |
| C9 HateMM 已回传trial10（非最终） | .614210/.815329/.649495，epoch2 | 该trial三项满足筛选数值，固定20trial仍运行，不能代替完整搜索结论；[原评测](../runs/20260906_interval_evidence_transport/hatemm/seed234/trial10/metrics.json) |
| C8 HateMM seed234 | 无合格best；无约束诊断trial11 .593962/.786366/.608225 | 20/20 PRUNED，within最高.608225<.632；[原评测](../runs/20260906_censored_evidence_process/hatemm/seed234/trial11/metrics.json)、[审计](../runs/20260906_censored_evidence_process/hatemm/seed234/artifact_audit.json) |
| C8 HCS seed234 | 无合格best；无约束诊断trial7 .604427/.589883/.510205 | 20/20 PRUNED，within最高.519747<.524；[原评测](../runs/20260906_censored_evidence_process/hateclipseg/seed234/trial7/metrics.json)、[审计](../runs/20260906_censored_evidence_process/hateclipseg/seed234/artifact_audit.json) |
| C5 HateMM三seed | .631307±.007170/.845751±.005332/.659939±.010262 | 数值确认通过；[汇总及原评测路径](../runs/20260905_interventional_evidence/hatemm/confirmation_summary.json) |
| C5 HCS三seed | .654090±.004189/.638275±.005749/.557700±.019617 | 数值确认通过，核心模块不成立；[汇总及原评测路径](../runs/20260905_interventional_evidence/hateclipseg/confirmation_summary.json) |
| C1三seed性能起点 | HMM .657±.013/.842±.005/.646±.004；HCS .699±.006/.681±.016/.553±.007 | [原始搜索及评测](../runs/20260903_hier_evidence_mil/) |

仅validation排序的零额外训练参考、不作门：C9 HCS trial16 .592270/.585958/.549114。C8 HMM trial19 .579398/.771111/.585330；HCS仍trial7。C5 HMM .6133/.8383/.6634、HCS .6516/.6358/.5566；C1 HMM .601/.821/.627、HCS .686/.665/.538。各study_summary/confirmation_summary存选择依据。

C9 HCS四臂全部50epoch、同配置、val63/test79、checkpoint选择及预测ID/秒长/finite检查通过。[消融审计及原metrics路径](../runs/20260906_interval_evidence_transport/ablations/hateclipseg/seed234/artifact_audit.json)。去VLM整体使AP/ROC下降 .010127/.029113，但不能代替M1软观察的贡献证据。全部79个test视频crop0内部诊断：输入grade平均后验概率 .988679，MAP仅 .005585比例窗口改变，解释M1为何接近硬等级；M2分配不是均匀分配，但尚未体现性能收益。[无GT/无训练的内部诊断](../runs/20260906_interval_evidence_transport/error_analysis/hcs_seed234_assignment.json)。C8完整负结果只见[归档](../archive/experiments/20260906_censored_evidence_process/README.md)。

## 运行任务与monitor

| 任务 | 当前状态 | 位置 |
|---|---|---|
| C9 HateMM seed234，lab1 | PID/PGID3125429；08:00已完成12/20，固定20trial继续；monitor1734844存活 | [预算](../runs/20260906_interval_evidence_transport/hatemm/seed234/budget.json)、[monitor](../runs/20260906_interval_evidence_transport/hatemm/seed234/monitor/run.log) |
| C9 HCS seed234，lab3 | 20/20结束，无残留进程，全部结果/checkpoint已回传；对应monitor通知已处理 | [审计](../runs/20260906_interval_evidence_transport/hateclipseg/seed234/artifact_audit.json)、[monitor](../runs/20260906_interval_evidence_transport/hateclipseg/seed234/monitor/run.log) |
| C9 HCS四臂消融，lab3 | 全部结束/回传/审计，无残留进程；原monitor已通知退出，本次已处理 | [审计](../runs/20260906_interval_evidence_transport/ablations/hateclipseg/seed234/artifact_audit.json) |
| C9 HCS预声明观察诊断，lab3 | PGID3583471；no_observation_loss、categorical_noise同trial18配置并行，各50epoch；monitor1750780首次RUNNING成功，两臂epoch输出正常 | [链输出](../runs/20260906_interval_evidence_transport/ablations/hateclipseg/seed234/auxiliary_chain/)、[monitor](../runs/20260906_interval_evidence_transport/ablations/hateclipseg/seed234/auxiliary_chain/monitor/run.log) |
| C8两语料 | 全部结束/回传/审计，两个monitor已通知退出；通知已处理，不重启 | [HMM monitor](../runs/20260906_censored_evidence_process/hatemm/seed234/monitor/run.log)、[HCS monitor](../runs/20260906_censored_evidence_process/hateclipseg/seed234/monitor/run.log) |
| 资源 | lab1搜索、lab3两项观察诊断并行；本机他人GPU任务；lab-server无HateVideo环境/项目 | 不重复旧臂、不加搜索trial，不启动尚未满足条件的确认seed |
| 长期目标monitor | PID1177638存活，07:04通知已处理；目标未完成，保留 | [日志](../runs/thread_monitor/01a06df5-3e92-79b0-be30-820db943e551/run.log) |

多机同步：C9搜索启动前三台均bf8c201；本次辅助链启动前三台均c4e24b6（仅同步用途），运行代码无脏文件或相关未跟踪文件、家目录无STRAY。两机torch2.7.1+cu128/transformers4.49对齐，原VLM两尺度各split全覆盖。CLAUDE.md既有修改、tandem.html、lab1 idea-stage/repro_t3al保留。搜索开始后仅新增诊断工具、可选链输出目录及文档，不替换活动训练模型/损失/采样/评测；CLAUDE.md及研究规则未改。

## 下一步

1. 接两项观察诊断完成事件，核验后回传，与已审计四臂合并；用来分析观察监督/序数约束，不把它们替代三个主消融。
2. 两语料首trial预算均已转录README，每seed固定20trial；HCS全20输出已实际解析ID/秒长/finite，[描述诊断](../runs/20260906_interval_evidence_transport/error_analysis/hcs_seed234_all_trials.json)。HateMM完成后同样回传审计，不模型轮询等待。
3. 两语料过筛才确认seed；三个主消融、no_vlm及同输入最强baseline证据未齐，不宣称SOTA/目标完成。显著增耗先说明成本和廉价替代，不堆VLM调用。

## 资料与历史

[研究规则](../RESEARCH_ITERATION_RULES.md)、[固定baseline表](../docs/duplex/OFFICIAL_VAL_RESULTS.md)、[评测协议](../docs/duplex/FRAME_EVAL_PROTOCOL.md)。负结果细节：[C6](../archive/experiments/20260905_latent_evidence_sequence/README.md)、[C7用户取消](../archive/experiments/20260906_context_witness/README.md)、[C4](../archive/experiments/20260904_null_token_cma/README.md)。[旧状态索引](../archive/research-wiki/STATUS_20260905_before_cleanup.md)。冻结Hate-follow-up引用不动；缓存出处见各data子目录PROVENANCE.md。
