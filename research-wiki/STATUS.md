# 当前研究状态

截至 **2026-10-02 08:12 NZDT**。依据：`experiments/20260925_query_paradigm/` 第 5 版软答案、复制似然及停止规则实现，实验 README 第 17 节，以及下列本机 `runs/` 评测器输出。2026-10-01 核验了 8 / 16 档共 18 个 study、每个 20 个 COMPLETE trial 的固定 8 次评测文件；停止规则汇总也与逐 trial 评测器输出核对。初版inside–outside骨干两主数据各三seed×20 trial已全部完成并回传核验；加性注意力统计量修订版HCS三seed各20/20已完成，HateMM seed234完成20/20、其余两seed继续确认；没有新增抽取。

## 当前目标与结论

当前开发主线是**按需提问定位第 5 版：单问软答案，8 档**。HateMM、HateClipSeg 为主数据集，DeHate 仅为外部验证。旧候选 3 修订 4 及后续模块一实验是历史参照，不再用它们描述最新进展；最终论文主表口径尚未锁定。

**用户 2026-10-01 最新授权：自动迭代 backbone，优先 novel 且涨点；充分尝试仍无提升时优先可验证的新意，完成后汇报。within 与 pooled AP / ROC 并列为主指标。**[第一轮：时间树上的区间内外表示](../experiments/20261001_inside_outside_backbone/README.md)已实现，用区间内组合与区间外上下文学习内容先验，复用现有特征与答案，新增 VLM 调用为 0。[一次独立 proposal review：GO](../docs/reviews/20261001_inside_outside_backbone_proposal.md)，[一次独立 code review：PASS](../docs/reviews/20261001_inside_outside_backbone_code.md)；完整搜索协议已在 README 第 6 节预写，两主数据各三seed完整搜索已完成，且outside满足两语料平均贡献要求；初版已具备机制贡献证据，但相对第5版仍有性能代价，继续修订以争取用户优先的novel且涨点，整体目标暂不宣布完成。创新主张限定为具体问题树上的 inside–outside 表示与答案似然训练。旧版无骨干消融只用于动机，不能代替新骨干验证。共享 QTL 基础设施已升入 `src/qtl/`，新骨干在新实验目录，旧入口仍兼容；评测器未修改。

- 第 5 版三语料各三 seed × 20 trial 已完成。相对第 4 版固定 8 次，AP 分别 +.004 / +.017 / +.037；ROC −.007 / −.008 / +.026；within +.045 / −.011 / +.015。软答案改善 AP，但未实现全部指标不降。
- **此前完成的停止实验是 DeHate 16 档搜索及停止阈值验证**。最终验证链在 2026-10-01 02:25 NZDT 正常结束，日志末尾 ALL_DONE；结果已回传本机，文件可解析且三 seed 齐全。16 档相对 8 档在两主数据 AP 更高，但 DeHate AP −.014；沿用此前决定，8 档保持主线，16 档只作敏感性结果。
- 自动停止候选 `qmixSG_rt10` 不设最少调用次数，在 validation 上按平均 8 次定阈值，配合复制似然，三语料 8 档与同设置固定 8 次大致持平。实际平均调用约 7.8–8.1，**尚未证明稳定省调用**。DeHate 16 档额外检验 within 差值为 −.00503，略低于该检查的 −.005 容差，不能写全部设置通过。
- 用 validation 标注直接选停止阈值也已完成：HCS / DeHate 常选 12–24 次，改善主要来自更多调用；HateMM 常选 1–6 次，16 档按三指标和的容差规则选到 2.45 次，AP / ROC / within 掉 .025 / .017 / .015。仍未得到稳健的统一选阈值方案。

## 三模块实现与缺口

| 模块 | 当前实现 | 证据与缺口 |
|---|---|---|
| VLM 观测与提问 | 二分时间树、按期望信息增益选节点；词级时间戳转录；单问首 token P(Yes) 软答案、8 档 | HateMM 视频内排序改善；五类软答案两主数据均下降，未采用；多视图估计未解决 8 次的 ROC 下降 |
| 弱监督骨干与先验 | 主线保留第 5 版；inside–outside 初版两语料三seed已完成；加性注意力统计量修订版HCS三seed已完成，HateMM其余两seed继续确认 | 初版两语料pooled下降，HateMM within也下降；修订版HCS三seed恢复部分pooled但within下降；HCS去学习注意力反而改善pooled，初版outside已满足两语料平均贡献要求；修订版机制尚未两语料确认 |
| 答案融合与停止 | 锚定答案模型；复制似然处理嵌套回答相关；变化、排序变化、视频后验熵的分位数组合停止 | 复制似然缓解 HateMM 多问变差；放宽停止阈值后约持平，但有手设衰减率 .10、validation 目标 / 容差与实际调用量问题 |

代码与机制、消融及逐项检查见 [实验 README 第 14–17 节](../experiments/20260925_query_paradigm/README.md)。当前优先工作转向 backbone 结构与定位贡献；停止实验结论保留，不能将整套第 5 版称为已完成全部论文验证。

## 最新权威结果与来源

**初版HCS三seed完整Optuna已完成**（每seed20 trial、全部50 epoch）：固定8次AP / ROC / within均值±标准差 **.668398±.003274 / .642595±.005721 / .560326±.004923**；相对第5版同三seed差值 **−.018057 / −.031130 / +.000100**，pooled下降、within基本持平。按seed234/2025/3407顺序，test选trial11/2/10；只按validation选trial6/17/1，test均值.652083/.620540/.561490。原始来源本机 `runs/20261001_inside_outside_backbone/hateclipseg/seed<seed>/trial<编号>/metrics_test_fixed8.json`；派生审计/统计 `runs/20261001_inside_outside_backbone/analysis/hateclipseg_three_seed_search.json`，完整逐seed数字见实验README第10节。HateMM三seed结果也已齐全，见下；仍未实现相对第5版涨点。HCS 固定配置三seed诊断已全部完成并回传；8次均值AP / ROC / within：full **.653391 / .623460 / .573357**，nooutside **.641279 / .614138 / .556733**，mean **.654397 / .626035 / .576118**。原始来源 `runs/20261001_inside_outside_backbone/diagnostics/hateclipseg/<arm>/seed<seed>/metrics_test_fixed8.json`，full seed234复用trial11；标准差与各seed见实验README第8节和 `analysis/hateclipseg_locked_diagnostics.json`。outside有平均效应支持，但AP收益主要由一个seed贡献；递归门控未胜过均值，不能当成已证实贡献。这些固定配置诊断不替代Optuna确认。由此提出[保留加性注意力统计量的inside–outside修订](../experiments/20261001_associative_io_backbone/README.md)，独立proposal review已GO、code review已PASS，两主数据seed234完整搜索于23:05在lab3启动；新增VLM调用0。

**初版HateMM三seed完整Optuna已完成**：每seed20/20 trial、全部50 epoch、实际checkpoint与780份统一评测核验，全部输出已回传。固定8次AP / ROC / within均值±总体标准差 **.676491±.010167 / .879176±.001587 / .737845±.014984**；相对第5版同三seed **−.014380 / −.005681 / −.035070**，三项均下降。test选trial17/11/11，validation选trial10/2/9，其test均值.640230/.868892/.681005。原始来源本机 `runs/20261001_inside_outside_backbone/hatemm/seed<seed>/trial<编号>/metrics_test_fixed8.json`；完整审计 `analysis/hatemm_three_seed_search.json`，两语料汇总 `runs/20261001_inside_outside_backbone/analysis/two_corpus_three_seed_search.json`，逐seed和0/8/32次比较见初版README第12节。至此初版两主数据各三seed完整搜索已齐全；pooled均值超过既定固定baseline，但仍低于当前第5版。

初版自身的两语料outside固定配置诊断也已齐全：HateMM full固定8次均值.674881/.875579/.756881，nooutside .650114/.869462/.747101，配对均值变化 **−.024767/−.006117/−.009780**；HCS变化 **−.012112/−.009322/−.016624**。两语料AP平均下降至少.01，满足规则14(g)平均贡献要求；HCS效应主要来自seed234，不能写每seed都稳定改善。原始来源本机 `runs/20261001_inside_outside_backbone/diagnostics/<corpus>/<arm>/seed<seed>/metrics_test_fixed8.json`，full seed234分别复用HateMM trial17/HCS trial11；配对审计 `runs/20261001_inside_outside_backbone/analysis/two_corpus_outside_diagnosis.json`。这是初版的机制贡献证据，不能替代修订版消融或证明相对第5版涨点；固定配置诊断不替代Optuna确认。

**加性统计量修订版HCS三seed完整Optuna已完成**：每seed20/20、60个trial全部50 epoch、实际checkpoint及780份统一评测核验并回传。固定8次AP / ROC / within均值±总体标准差 **.681333±.001454 / .653216±.008294 / .526091±.021952**；对初版 **+.012936 / +.010621 / −.034235**，对第5版 **−.005122 / −.020509 / −.034135**。pooled较初版恢复，但within下降，相对当前主线三项仍低。seed234/2025/3407按test选trial7/7/0，validation checkpoint epoch1/5/1；只按validation选trial10/11/0，其test均值±标准差 **.662610±.022670 / .632999±.013865 / .521973±.016386**。0次均值.545151/.509617/.521391，内容先验的视频间区分损失仍明显。原始来源本机 `runs/20261001_associative_io_backbone/hateclipseg/seed<seed>/trial<编号>/metrics_test_fixed8.json`；完整审计/三seed统计 `runs/20261001_associative_io_backbone/analysis/hateclipseg_three_seed_search.json`，0/8/32次和逐seed见修订版README第13节。HateMM其余两seed仍在跑，不能称两主数据三seed确认已齐全。

修订版HCS固定配置full三seed已完成，8次均值±标准差 **.648252±.033225 / .623946±.041971 / .548341±.027075**；nooutside、noattention均已完成并回传核验。原始来源 `runs/20261001_associative_io_backbone/diagnostics/hateclipseg/full/seed{2025,3407}/metrics_test_fixed8.json`，seed234复用trial7；派生审计/统计 `runs/20261001_associative_io_backbone/analysis/hateclipseg_locked_full.json`。nooutside三seed均值.640263/.612341/.537365，移除后8次变化−.007989/−.011605/−.010976，HCS有outside平均效应支持，但pooled各seed不一致；不能代替两语料贡献验证。来源对应 `nooutside/seed<seed>/metrics_test_fixed8.json`，审计/配对差值 `runs/20261001_associative_io_backbone/analysis/hateclipseg_locked_nooutside.json`。noattention固定8次均值±标准差 **.675338±.010891 / .640604±.009837 / .535214±.022023**，相对full **+.027086 / +.016658 / −.013126**；学习注意力没有pooled增益证据，不能作为已验证贡献。来源对应 `noattention/seed<seed>/metrics_test_fixed8.json`；三组完整审计与逐seed配对差值 `runs/20261001_associative_io_backbone/analysis/hateclipseg_locked_diagnostics.json`。这些诊断不是Optuna确认，outside在full中的效应不能直接外推到noattention。

**加性统计量修订版HateMM seed234已完成20/20**：test选trial10，固定8次AP / ROC / within **.651747 / .874278 / .765879**，相对初版−.027812/−.007074/+.012480，相对第5版−.042702/−.012428/−.022854；尚无相对主线涨点。validation选trial4为.616726/.868900/.742323。来源本机 `runs/20261001_associative_io_backbone/hatemm/seed234/trial{10,4}/metrics_test_fixed8.json`；完整50 epoch、checkpoint、260份统一评测核验及0/8/32次比较记录在 `runs/20261001_associative_io_backbone/analysis/hatemm_seed234_comparison.json`。两语料四个pooled值超过规则8固定baseline筛选线，按既定协议补seed2025/3407完整搜索，仍需确认和两语料机制贡献，不能称novelty完成。

**修订版HateMM锁定trial10的full诊断已完成**：三seed各50 epoch、checkpoint和39份统一评测完整并回传；固定8次AP / ROC / within均值±总体标准差 **.611048±.040221 / .860227±.009987 / .737149±.024840**，validation选epoch25/23/10。来源本机 `runs/20261001_associative_io_backbone/diagnostics/hatemm/full/seed{2025,3407}/metrics_test_fixed8.json`，seed234复用trial10；审计/统计 `runs/20261001_associative_io_backbone/analysis/hatemm_locked_full.json`。AP存在较大seed差异；noattention正在最后seed，nooutside已接续，配对结论尚待齐全。这些是固定配置诊断，不是Optuna确认。

固定 8 次预算，test、1 fps，三 seed 234 / 2025 / 3407 均值 ± seed 标准差；列顺序 AP / ROC / within。问题树耗尽的视频实际可少于 8 次。**Optuna 按 test AP / ROC 均值选 trial，属于开发期结果；各 trial 的 checkpoint 由 validation 选择。**

| 配置 | HateMM | HateClipSeg | DeHate（外部验证） |
|---|---|---|---|
| 第 5 版 8 档 | .6909 ± .0137 / .8849 ± .0021 / .7729 ± .0144 | .6865 ± .0037 / .6737 ± .0054 / .5602 ± .0136 | .2620 ± .0103 / .7800 ± .0070 / .6792 ± .0157 |
| 第 5 版 16 档 | .7063 ± .0100 / .8858 ± .0020 / .7588 ± .0242 | .6993 ± .0032 / .6764 ± .0074 / .5715 ± .0186 | .2485 ± .0031 / .7763 ± .0075 / .6864 ± .0055 |
| 8 档只按 validation 选 trial | .6658 / .8685 / .7431 | .6796 / .6500 / .5678 | .2337 / .7594 / .6744 |
| 16 档只按 validation 选 trial | .6710 / .8734 / .7223 | .6757 / .6414 / .5408 | .2241 / .7527 / .6741 |

权威文件：`runs/<实验>/<语料>/seed<seed>/trial<编号>/metrics_test_fixed8.json`，评测器字段 `results.score_av`。搜索完成与 trial 选择记录在各 seed 的 `study_summary.json`；以下编号均按 seed 234 / 2025 / 3407 顺序。

| 实验目录 | 语料 | test 选 trial | validation 选 trial |
|---|---|---|---|
| `runs/20260929_query_paradigm_r5/` | hatemm / hateclipseg / dehate | 9,4,5 / 13,11,19 / 8,4,4 | 7,1,15 / 6,3,18 / 12,17,10 |
| `runs/20260929_query_paradigm_r5_lv16/` | hatemm / hateclipseg / dehate | 11,10,19 / 5,16,10 / 16,18,6 | 见各 study_summary.json；DeHate 为 4,18,13 |

停止规则检查的固定线含复制似然，与上表基础固定 8 次配置不同，必须用各自同设置固定线比较：

| 8 档 + 复制似然 + qmixSG_rt10 | 平均调用 | AP / ROC / within | 对同设置固定 8 次的差值 |
|---|---|---|---|
| HateMM | 7.83 | .6946 / .8898 / .7691 | +.0089 / +.0076 / +.0026 |
| HateClipSeg | 8.05 | .6834 / .6749 / .5575 | +.0003 / +.0031 / −.0015 |
| DeHate | 8.03 | .2547 / .7743 / .6721 | −.0035 / +.0040 / +.0049 |

来源：`runs/20260929_query_paradigm_r5/stop_check/<语料>/seed<seed>_trial<编号>_r5rt/metrics_test_qmixSG_rt108.json` 与 `metrics_test_fixed8.json`，对应汇总 `{hatemm,hateclipseg,dehate}_r5rt.json`。DeHate 16 档额外检验为同目录 `dehate/seed<seed>_trial<编号>_lv16e/metrics_test_qmixSG_rt108.json`，汇总 `dehate_lv16e.json`。

最新 validation 选阈值来源：`stop_check/<语料>/seed<seed>_trial<编号>_{r5sel,lv16sel}/metrics_test_sel_qmixSG_rt10_<目标>.json`，汇总 `*_{r5sel,lv16sel}.json`。DeHate 16 档按三指标和取最大：16.58 次，.2570 / .7860 / .6920；取离最大 .005 内最少次数：11.45 次，.2505 / .7808 / .6851。其它规则、目标与成本比较见 README 17.4(11)。本次按原始评测输出更正旧文档的 within 差值 −.0052 为 −.00503，未改变通过与否结论。

## 运行任务与 monitor

- **Pursuing goal 保持 paused，不重新开启。** 已复用原有 `scripts/monitor_thread.py`，当前会话 `01a0f639-b211-75e3-9155-e15e30534b46`，每 3 小时接续；PID `3245799`，日志和状态在 `runs/thread_monitor/01a0f639-b211-75e3-9155-e15e30534b46/`。首条及23:54、02:54、05:54定时 `codex queue` 通知已成功接续。当前目标未完成：初版已有两语料完整搜索和outside贡献证据；修订版三seed确认/两语料机制验证仍在进行，相对当前主线的性能提升仍未实现；没有硬阻塞，保留heartbeat和活动run monitor。
- **初版骨干完整搜索**：两语料于20:57在 uoa-lab3 / sc474398 启动。HCS 于22:14完成20/20 trial，已核验真实主/子进程结束、全部50 epoch历史和完整评测并回传本机。HateMM于10月2日00:15:43完成20/20，已核验真实主/子进程退出并回传全部输出；每trial50 epoch、214视频/29269秒、260份统一评测完整。目录为 `runs/20261001_inside_outside_backbone/{hatemm,hateclipseg}/seed234/`。首 trial HateMM 559.52 秒、HCS 235.49 秒，两套预算各固定20，文件 `budget.json` 已在本机。
- **完成通知**：HCS monitor `3249811` 已正常发出22:16完成通知并退出，有 `notification_sent` 标记；HateMM monitor `3249810` 已于00:16成功通知并退出。状态在本机 `runs/20261001_inside_outside_backbone/<corpus>/seed234/monitor/`。每120秒检查真实进程身份/相关子进程，SSH失败只重试；完成通知自动接续已实际验证，heartbeat仍作兜底。
- **初版其余seed确认**：10月2日00:22在lab3启动。HCS seed2025/3407于03:09:04均已完成20/20，主/子进程退出、全部50 epoch与各260份评测核验并回传，monitor `3478308 / 3478309` 于03:11已通知退出；seed3407若后续通知到达也不重复运行。HateMM seed2025/3407于07:10:19/07:09:59均已完成20/20；主/同会话子进程退出、全部50 epoch、checkpoint及各260份评测已核验回传；monitor `3478306 / 3478307` 于07:10均已通知退出。初版全部确认study已结束，延迟通知不重复启动。输出 `runs/20261001_inside_outside_backbone/<corpus>/seed<seed>/`，本机各run下monitor记录；启动记录 `setup/confirmation_launches.json`。首trial耗时HateMM 1424.30/1424.47秒、HCS 577.19/577.01秒，预算各20。不是固定配置诊断。
- **修订版完整搜索**：`runs/20261001_associative_io_backbone/{hatemm,hateclipseg}/seed234/`，uoa-lab3 / sc474398，10月1日23:05启动。HCS于10月2日01:06:01完成20/20，真实主/子进程退出，全部输出已回传并核验50 epoch和260份评测；monitor `3392350` 已成功通知退出。HateMM于04:45:32完成20/20，真实主/同会话子进程退出，全部50 epoch、checkpoint与260份评测核验并回传；monitor `3392349` 于04:47通知退出。两套预算20，首trial756.27/314.65秒；不是独立吞吐比较。monitor记录在对应本机run目录，启动记录 `runs/20261001_associative_io_backbone/setup/launches.json`。
- **修订版补seed确认已启动**：04:53:50–52在lab3启动两语料seed2025/3407四个完整study；HateMM owner `1908437 / 1908654`、monitor `3778288 / 3778293`；HCS owner `1908910 / 1909155`、monitor `3778306 / 3778313`。四项均已核验独立monitor首轮RUNNING、输入覆盖完整和实际epoch输出。输出 `runs/20261001_associative_io_backbone/<corpus>/seed<seed>/`，本机各run下monitor；启动记录 `runs/20261001_associative_io_backbone/setup/confirmation_launches.json`。按README第10节原空间/目标/50 epoch继续；四项预算均已锁定20，首trial为HateMM 1415.40/1413.64秒、HCS 578.15/577.11秒，对应budget.json已回传。HCS seed2025/3407均于08:01:44完成20/20，真实主/同会话子进程退出，全部50 epoch、checkpoint及各260份评测核验回传；monitor `3778306 / 3778313` 于08:03已成功通知退出，延迟通知不重复启动。08:08 HateMM各完成8/20，真实主/子进程和两个搜索monitor正常，继续原预算；不缩减或重启。这些不是固定配置诊断。
- **修订版HCS结构诊断**：full于03:29:15完成，真实主/子进程均已结束，三seed锁定配置、全部50 epoch、39份统一评测核验并回传；monitor `3668044` 已通知退出。nooutside于03:36:42完成，真实进程退出，三seed各50 epoch、仅io_outside配置差异及39份评测核验并回传；monitor `3668045` 已通知退出。noattention于03:53:19完成，真实主/同会话子进程退出、三seed各50 epoch、仅io_attention差异、checkpoint和39份统一评测核验并回传；monitor `3687104` 于03:55成功通知退出。输出 `runs/20261001_associative_io_backbone/diagnostics/hateclipseg/<arm>/`，monitor在对应本机目录；启动记录 `runs/20261001_associative_io_backbone/setup/diagnostic_launches.json`。全部按锁定配置、完整50 epoch与统一评测，full seed234复用trial7。这些是固定配置诊断，不是Optuna确认，不重复启动已完成或活动arm。
- **修订版HateMM结构诊断**：full于07:59:21结束，真实主/同会话子进程退出、trial10锁定配置、三seed完整50 epoch、checkpoint及39份评测已核验回传；monitor `4025659` 于08:00通知退出。noattention owner `2094003`、monitor `4025672` 仍正常，08:11已完成seed234/2025，seed3407训练至epoch38/50。nooutside已于08:02:29在lab3利用释放位置启动，owner `2126538`、本机monitor `4081282`，已核验首次RUNNING绑定正确身份，08:11 seed234至epoch27/50。三组均按README第11节锁定trial10，各三seed、50 epoch，full seed234复用；输出 `runs/20261001_associative_io_backbone/diagnostics/hatemm/<arm>/`，启动记录 `runs/20261001_associative_io_backbone/setup/diagnostic_hatemm_launches.json`。三个arm均已启动，后续完成通知不得重复启动nooutside；新增VLM调用0，不混入初版，也不是Optuna确认。
- **初版HateMM结构诊断**：full于04:22:06结束，真实主/同会话子进程退出、trial17锁定配置、三seed完整50 epoch、checkpoint及39份评测均已核验，全部输出已回传；monitor `3701013` 于04:23成功通知并退出。输出 `runs/20261001_inside_outside_backbone/diagnostics/hatemm/full/`，monitor在对应本机目录，启动记录 `runs/20261001_inside_outside_backbone/setup/diagnostic_hatemm_launches.json`。full/nooutside协议和配置已锁定（初版README第11节），新增5次完整50 epoch，无新VLM调用；nooutside已于03:57:39利用释放位置启动，owner `1876280`、本机monitor `3716584`，输出为同目录 `nooutside/`；已核验首次RUNNING绑定正确身份、seed234实际epoch输出。nooutside于04:48:33完成，本轮已一并核验真实主/子进程退出、三seed完整50 epoch、仅io_outside配置差异、checkpoint与39份评测完整并回传；monitor `3716584` 已通知退出，后续通知若到达不重复运行，启动记录同上。这是初版自身的两语料配对检验，不与修订版结果混作同一方法贡献，也不是Optuna确认。
- **HCS结构诊断已完成**：full于22:38结束，nooutside / mean于22:42结束，真实主/子进程均已核验退出；三组各三seed、各50 epoch、配置和0/8/32次统一评测完整性均通过，全部输出已回传。对应monitor `3347291 / 3347292 / 3347293` 已通知退出；输出 `runs/20261001_inside_outside_backbone/diagnostics/hateclipseg/<arm>/`，监控记录在本机对应 `monitor/`。不重复启动或把固定配置诊断当成Optuna确认。
- 开跑前两套缓存实际解析、形状、时间轴、完整覆盖和 split isolation 均通过：HateMM 1067 个视频（744 / 109 / 214），HCS 393 个（251 / 63 / 79）。日志已回传本机 `runs/20261001_inside_outside_backbone/setup/inputs_{hatemm,hateclipseg}.log`。本机与 lab3 环境均为 torch 2.7.1+cu128、CUDA 12.8、Optuna 4.9.0。
- 资源检查：本机继续不跑训练。lab1 另有 VLLM 占约 28 GiB，仅余 3.7 GiB；lab-server 仅余 2.6 GiB，不参与本轮；未改动他人任务。lab3 原有 lightmem 任务占约5 GiB；08:11整卡14449 MiB已用/17640 MiB空闲、利用率99%；当前修订版HateMM两项搜索+noattention/nooutside两项诊断并行，HCS三个study及初版全部实验已结束，无已锁定待启动任务。lab1 / lab-server 08:01检查仅余3.7 / 2.6 GiB；实际训练正常，未见OOM。
- `bash scripts/check_layout.sh` 开跑前已执行。多机代码同步检查：本轮回传修订版HCS其余两seed并汇总完整三seed结果，仅更新结果/运行文档；文档提交及三机同步检查记录在本机 `runs/20261001_associative_io_backbone/setup/hcs_confirmation_documentation_sync.json`；活动训练的模型、共享trainer与评测器文件未改动；本机未跟踪的 `tandem.html` 与 lab1 的 `idea-stage/` 为已有无关内容，未改动。本轮仅同步 STATUS / 实验 README，未替换活动实验代码。已查看的家目录存量 `conversation-pilot` / `ctx_pri`、远端 `MemoryAgen` / `ctx_pri_*`、工具数据、旧 `list` / `vllm.pid` 保留，没有擅自清理。

## 下一步

1. 修订版HCS三seed完整Optuna已齐全，仍低于第5版三主指标；HateMM seed2025/3407两项完整Optuna继续固定预算，独立monitor正常；初版两语料三seed搜索和自身outside消融均已齐全，机制有贡献但相对第5版掉点，继续争取novel且涨点。修订版HCS学习注意力没有pooled增益证据，HateMM seed234相对初版pooled下降、within提高，结合本版本两语料结构诊断决定后续修订。
2. 修订版HateMM full已核验回传，noattention最后seed和nooutside三seed链继续，各有独立monitor；三个arm均已启动，不重复运行。剩余两组结束后回传，与本修订版HCS对应消融汇总配对差值，区分outside和注意力效应，不把固定配置诊断当Optuna确认或混入初版。优先提升 AP / ROC / within，再核对现有停止设置。
3. 保留全部授权和研究目标，等待只交给后台 monitor，不通过模型反复轮询；最终核对新骨干贡献、性能代价、成本及论文操作点后汇报。现有训练/搜索规则文件不修改。

## 历史与规则

[实验详细记录](../experiments/20260925_query_paradigm/README.md)、[本次整理前状态](../archive/research-wiki/STATUS_20261001_before_current_summary.md)、[研究规则](../RESEARCH_ITERATION_RULES.md)、[固定 baseline 表](../docs/duplex/OFFICIAL_VAL_RESULTS.md)、[DeHate 评测协议](../docs/duplex/FRAME_EVAL_PROTOCOL_DEHATE.md)。旧结论与其它历史链接保留在归档，不在当前状态追加时间线。
