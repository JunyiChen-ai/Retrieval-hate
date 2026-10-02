# 当前研究状态

截至 **2026-10-02 13:38 NZDT**。依据：`experiments/20260925_query_paradigm/` 第 5 版软答案、复制似然及停止规则实现，实验 README 第 17 节，以及下列本机 `runs/` 评测器输出。2026-10-01 核验了 8 / 16 档共 18 个 study、每个 20 个 COMPLETE trial 的固定 8 次评测文件；停止规则汇总也与逐 trial 评测器输出核对。初版inside–outside骨干两主数据各三seed×20 trial已全部完成并回传核验；加性统计量修订版两语料各三seed×20 trial及三组结构诊断已全部核验回传；固定8次均低于第5版三主指标，且自身outside未满足两语料平均贡献要求，已归档。[残差修订版](../experiments/20261002_residual_io_backbone/README.md)两主数据seed234各20-trial完整搜索均已核验回传，固定8次三项均低于r5同seed，但通过规则8既定pooled筛选，四项2025/3407完整搜索已启动；HCS自身三组诊断未支持8次核心贡献，HateMM锁定full诊断已完整核验，nooutside/noresidual两组运行。没有新增抽取。

## 当前目标与结论

当前开发主线是**按需提问定位第 5 版：单问软答案，8 档**。HateMM、HateClipSeg 为主数据集，DeHate 仅为外部验证。旧候选 3 修订 4 及后续模块一实验是历史参照，不再用它们描述最新进展；最终论文主表口径尚未锁定。

**用户 2026-10-01 最新授权：自动迭代 backbone，优先 novel 且涨点；充分尝试仍无提升时优先可验证的新意，完成后汇报。within 与 pooled AP / ROC 并列为主指标。**[第一轮：时间树上的区间内外表示](../experiments/20261001_inside_outside_backbone/README.md)已实现，用区间内组合与区间外上下文学习内容先验，复用现有特征与答案，新增 VLM 调用为 0。[一次独立 proposal review：GO](../docs/reviews/20261001_inside_outside_backbone_proposal.md)，[一次独立 code review：PASS](../docs/reviews/20261001_inside_outside_backbone_code.md)；完整搜索协议已在 README 第 6 节预写，两主数据各三seed完整搜索已完成，且outside满足两语料平均贡献要求；初版已具备机制贡献证据，但相对第5版仍有性能代价，继续修订以争取用户优先的novel且涨点，整体目标暂不宣布完成。残差修订版保留第5版跨模态编码与视频头，将pre-CMA区间内外内容作为秒级/节点残差；两主数据seed234均已完整，HateMM/HCS固定8次分别.674695/.877526/.747603与.673446/.669614/.541302，三主指标均低于r5同seed；HCS两种消融未满足8次核心贡献要求。现按既定筛选结果补四项独立完整搜索，并做本版本HateMM锁定诊断；不能因过baseline门而宣布novelty或涨点成立。创新主张限定为具体问题树上的 inside–outside 表示与答案似然训练。旧版无骨干消融只用于动机，不能代替新骨干验证。共享 QTL 基础设施已升入 `src/qtl/`，新骨干在新实验目录，旧入口仍兼容；评测器未修改。

- 第 5 版三语料各三 seed × 20 trial 已完成。相对第 4 版固定 8 次，AP 分别 +.004 / +.017 / +.037；ROC −.007 / −.008 / +.026；within +.045 / −.011 / +.015。软答案改善 AP，但未实现全部指标不降。
- **此前完成的停止实验是 DeHate 16 档搜索及停止阈值验证**。最终验证链在 2026-10-01 02:25 NZDT 正常结束，日志末尾 ALL_DONE；结果已回传本机，文件可解析且三 seed 齐全。16 档相对 8 档在两主数据 AP 更高，但 DeHate AP −.014；沿用此前决定，8 档保持主线，16 档只作敏感性结果。
- 自动停止候选 `qmixSG_rt10` 不设最少调用次数，在 validation 上按平均 8 次定阈值，配合复制似然，三语料 8 档与同设置固定 8 次大致持平。实际平均调用约 7.8–8.1，**尚未证明稳定省调用**。DeHate 16 档额外检验 within 差值为 −.00503，略低于该检查的 −.005 容差，不能写全部设置通过。
- 用 validation 标注直接选停止阈值也已完成：HCS / DeHate 常选 12–24 次，改善主要来自更多调用；HateMM 常选 1–6 次，16 档按三指标和的容差规则选到 2.45 次，AP / ROC / within 掉 .025 / .017 / .015。仍未得到稳健的统一选阈值方案。

## 三模块实现与缺口

| 模块 | 当前实现 | 证据与缺口 |
|---|---|---|
| VLM 观测与提问 | 二分时间树、按期望信息增益选节点；词级时间戳转录；单问首 token P(Yes) 软答案、8 档 | HateMM 视频内排序改善；五类软答案两主数据均下降，未采用；多视图估计未解决 8 次的 ROC 下降 |
| 弱监督骨干与先验 | 主线保留第 5 版；inside–outside 初版两语料三seed已完成；加性版两语料三seed及诊断齐全后归档；残差版两语料seed234各20 trial齐全，四项补seed搜索运行；HCS自身三组诊断未支持8次核心贡献，HateMM full诊断完成、两组消融运行 | 初版两语料pooled下降，HateMM within也下降；修订版HCS三seed恢复部分pooled但within下降；两语料去学习注意力均提高8次pooled均值，但有预算/指标代价；初版outside已满足两语料平均贡献要求，修订版outside未满足两语料平均贡献要求 |
| 答案融合与停止 | 锚定答案模型；复制似然处理嵌套回答相关；变化、排序变化、视频后验熵的分位数组合停止 | 复制似然缓解 HateMM 多问变差；放宽停止阈值后约持平，但有手设衰减率 .10、validation 目标 / 容差与实际调用量问题 |

代码与机制、消融及逐项检查见 [实验 README 第 14–17 节](../experiments/20260925_query_paradigm/README.md)。当前优先工作转向 backbone 结构与定位贡献；停止实验结论保留，不能将整套第 5 版称为已完成全部论文验证。

## 最新权威结果与来源

**残差版HateMM seed234完整20-trial搜索已完成并回传**：12:44:09正常结束，主进程及全部同会话子进程退出；每trial50 epoch、实际validation checkpoint、配置/数据库目标及260份统一评测核验通过（214视频/29269秒，无缺失/额外视频）。按开发期test(AP+ROC)/2选trial10、checkpoint epoch3；AP / ROC / within固定0次 **.536213/.766454/.745240**，固定8次 **.674695/.877526/.747603**，附加32次.666814/.868253/.743581。8次对r5同seed变化 **−.019755/−.009180/−.041130**，尚未涨点。仅validation选trial5、epoch42，固定0次.562868/.803026/.745095，固定8次 **.584880/.852691/.760229**。原始本机来源 `runs/20261002_residual_io_backbone/hatemm/seed234/trial{10,5}/metrics_test_fixed{0,8,32}.json`；完整审计 `analysis/hatemm_seed234_search.json`，两主数据40 trial/520评测汇总 `runs/20261002_residual_io_backbone/analysis/two_corpus_seed234_search.json`。两语料四个pooled数过规则8门，按预写协议补seed2025/3407各自20-trial独立搜索；HCS机制贡献失败结论保持。HateMM full/nooutside/noresidual锁定本版trial10配置，各三seed50 epoch，full seed234复用trial10；full三seed已完整核验，两消融均已启动，不把它们当Optuna确认。详见本版README第8节。

**残差版HateMM锁定trial10的full三seed诊断已完整结束并回传**：13:29:27结束，真实主进程及所有同会话非僵尸子进程退出；各seed50 epoch、实际validation checkpoint epoch3/1/2、39份统一test评测核验通过。固定8次AP / ROC / within均值±总体标准差 **.652725±.015862 / .870560±.005199 / .750419±.002098**；0次 **.527624±.011769 / .764420±.002490 / .728768±.019414**，附加32次.660768/.865191/.737213。seed234复用本版trial10，另两seed固定配置完整训练，**不是Optuna确认**；早期checkpoint不等于短训练。原始本机来源 `runs/20261002_residual_io_backbone/hatemm/seed234/trial10/metrics_test_fixed<次数>.json` 及同实验 `diagnostics/hatemm/full/seed<2025或3407>/metrics_test_fixed<次数>.json`，审计 `analysis/hatemm_locked_full.json`，三组汇总入口 `runs/20261002_residual_io_backbone/analysis/hatemm_locked_diagnostics.json`。两消融正在完整训练，当前不填配对差值、不借用旧版贡献；HCS自身主操作点贡献失败结论保持。逐seed及成本见本版README第9节。

**残差修订版HCS seed234完整搜索已完成并回传**：11:12:45结束，真实主进程/同会话子进程退出，20/20 COMPLETE、每trial50 epoch、实际validation checkpoint及260份统一test评测逐项通过（79视频/18839秒，无缺失/额外视频）。开发期按test(AP+ROC)/2选trial19、validation checkpoint epoch29；以下AP / ROC / within，都是单seed：固定0次 **.641423 / .629097 / .526557**，固定8次 **.673446 / .669614 / .541302**，附加32次.690887/.679990/.615161。固定8次对r5同seed变化 **−.015021 / −.005230 / −.014829**，尚未涨点；0次也下降，不能用32次ROC小幅改善替换主结论。仅按validation选trial16、epoch1，固定0次.520974/.498905/.544452，固定8次 **.627898/.588489/.551426**。两语料pooled均通过既定筛选门，四项补seed完整搜索已启动。原始本机来源 `runs/20261002_residual_io_backbone/hateclipseg/seed234/trial{19,16}/metrics_test_fixed{0,8,32}.json`，完整审计和与三版本同seed比较 `runs/20261002_residual_io_backbone/analysis/hateclipseg_seed234_search.json`。本版full/nooutside/noresidual各三seed锁定诊断已全部完整，结论见下；它们不是Optuna确认，不借用旧版机制证据。细节见[残差版README第6节](../experiments/20261002_residual_io_backbone/README.md)。

**残差版HCS锁定trial19的full/nooutside/noresidual三组诊断全部完整并已回传**：三组各三seed×50 epoch，117份统一test评测及实际validation checkpoint核验通过；full seed234复用trial19，固定配置诊断**不是Optuna确认**。固定8次AP / ROC / within均值±总体标准差：full **.667221±.004415 / .651915±.012530 / .560469±.014628**，nooutside **.670221±.004482 / .649848±.007202 / .577506±.007786**，noresidual **.678753±.011413 / .653350±.016401 / .555692±.018537**。去outside逐seed配对均值变化 **+.003000/−.002068/+.017037**；去整体残差 **+.011532/+.001434/−.004776**。两者均未使HCS 8次平均AP或ROC下降至少.01，本版本核心贡献尚不成立，不能借用初版证据。0次去整体残差AP/ROC下降.045464/.046002，32次ROC下降.018557，但有明显seed/预算依赖，不能替换8次主结论。原始本机来源 `runs/20261002_residual_io_backbone/diagnostics/hateclipseg/<arm>/seed<seed>/metrics_test_fixed<次数>.json`，full seed234为同实验 `hateclipseg/seed234/trial19/metrics_test_fixed<次数>.json`。逐组审计 `analysis/hateclipseg_locked_<arm>.json`，全部0/8/32统计与逐seed配对差值 `runs/20261002_residual_io_backbone/analysis/hateclipseg_locked_diagnostics.json`；完整表见本版README第7节。

**初版HCS三seed完整Optuna已完成**（每seed20 trial、全部50 epoch）：固定8次AP / ROC / within均值±标准差 **.668398±.003274 / .642595±.005721 / .560326±.004923**；相对第5版同三seed差值 **−.018057 / −.031130 / +.000100**，pooled下降、within基本持平。按seed234/2025/3407顺序，test选trial11/2/10；只按validation选trial6/17/1，test均值.652083/.620540/.561490。原始来源本机 `runs/20261001_inside_outside_backbone/hateclipseg/seed<seed>/trial<编号>/metrics_test_fixed8.json`；派生审计/统计 `runs/20261001_inside_outside_backbone/analysis/hateclipseg_three_seed_search.json`，完整逐seed数字见实验README第10节。HateMM三seed结果也已齐全，见下；仍未实现相对第5版涨点。HCS 固定配置三seed诊断已全部完成并回传；8次均值AP / ROC / within：full **.653391 / .623460 / .573357**，nooutside **.641279 / .614138 / .556733**，mean **.654397 / .626035 / .576118**。原始来源 `runs/20261001_inside_outside_backbone/diagnostics/hateclipseg/<arm>/seed<seed>/metrics_test_fixed8.json`，full seed234复用trial11；标准差与各seed见实验README第8节和 `analysis/hateclipseg_locked_diagnostics.json`。outside有平均效应支持，但AP收益主要由一个seed贡献；递归门控未胜过均值，不能当成已证实贡献。这些固定配置诊断不替代Optuna确认。由此提出[保留加性注意力统计量的inside–outside修订](../archive/experiments/20261001_associative_io_backbone/README.md)，独立proposal review已GO、code review已PASS，两主数据seed234完整搜索于23:05在lab3启动；新增VLM调用0。

**初版HateMM三seed完整Optuna已完成**：每seed20/20 trial、全部50 epoch、实际checkpoint与780份统一评测核验，全部输出已回传。固定8次AP / ROC / within均值±总体标准差 **.676491±.010167 / .879176±.001587 / .737845±.014984**；相对第5版同三seed **−.014380 / −.005681 / −.035070**，三项均下降。test选trial17/11/11，validation选trial10/2/9，其test均值.640230/.868892/.681005。原始来源本机 `runs/20261001_inside_outside_backbone/hatemm/seed<seed>/trial<编号>/metrics_test_fixed8.json`；完整审计 `analysis/hatemm_three_seed_search.json`，两语料汇总 `runs/20261001_inside_outside_backbone/analysis/two_corpus_three_seed_search.json`，逐seed和0/8/32次比较见初版README第12节。至此初版两主数据各三seed完整搜索已齐全；pooled均值超过既定固定baseline，但仍低于当前第5版。

初版自身的两语料outside固定配置诊断也已齐全：HateMM full固定8次均值.674881/.875579/.756881，nooutside .650114/.869462/.747101，配对均值变化 **−.024767/−.006117/−.009780**；HCS变化 **−.012112/−.009322/−.016624**。两语料AP平均下降至少.01，满足规则14(g)平均贡献要求；HCS效应主要来自seed234，不能写每seed都稳定改善。原始来源本机 `runs/20261001_inside_outside_backbone/diagnostics/<corpus>/<arm>/seed<seed>/metrics_test_fixed8.json`，full seed234分别复用HateMM trial17/HCS trial11；配对审计 `runs/20261001_inside_outside_backbone/analysis/two_corpus_outside_diagnosis.json`。这是初版的机制贡献证据，不能替代修订版消融或证明相对第5版涨点；固定配置诊断不替代Optuna确认。

**加性统计量修订版两语料三seed完整搜索已全部完成，现已归档**：各seed20/20、每trial50 epoch、实际checkpoint及全部1560份统一test评测核验并回传。本机原始来源 `runs/20261001_associative_io_backbone/<corpus>/seed<seed>/trial<编号>/metrics_test_fixed8.json`；完整逐trial审计及0/8/32次比较 `analysis/<corpus>_three_seed_search.json`，两语料汇总 `runs/20261001_associative_io_backbone/analysis/two_corpus_three_seed_search.json`。以下均为三seed均值±总体标准差，固定8次、AP / ROC / within；Optuna按test目标选trial，属于开发期结果，checkpoint仍由validation选。

| 加性版选择方式 | HateMM | HateClipSeg |
|---|---|---|
| test目标选trial | .670993±.025027 / .875651±.002628 / .761282±.008523 | .681333±.001454 / .653216±.008294 / .526091±.021952 |
| 仅validation选trial | .637346±.015603 / .861325±.014171 / .722427±.028676 | .662610±.022670 / .632999±.013865 / .521973±.016386 |
| test选结果对第5版均值变化 | −.019879 / −.009206 / −.011633 | −.005122 / −.020509 / −.034135 |

按seed234/2025/3407顺序，HateMM test选trial10/5/10、checkpoint epoch25/8/3，validation选trial4/10/4；HCS test选trial7/7/0、epoch1/5/1，validation选trial10/11/0。HateMM对初版变化−.005498/−.003525/+.023437，within恢复但pooled未提高；HCS对初版+.012936/+.010621/−.034235。HateMM32次对第5版+.015680/+.001080/+.010261是附加预算改善，HCS32次仍低；不能更换固定8次主操作点或挑单seed声称全面涨点。完整逐seed结果、标准差与去向见[归档实验README第14节](../archive/experiments/20261001_associative_io_backbone/README.md)。

**加性版自身两语料固定配置结构诊断也已齐全**：HateMM锁定trial10、HCS锁定trial7，各full/noattention/nooutside三seed完整50 epoch，每组39份统一评测，均已回传。去注意力8次配对均值变化：HateMM +.028406/+.005344/+.003901，HCS +.027086/+.016658/−.013126；没有学习注意力的两语料pooled收益证据，且去注意力有HCS within和HateMM0次代价。去outside变化：HateMM **+.007412/−.006680/−.001191**，HCS **−.007989/−.011605/−.010976**，未满足规则14(g)两语料平均贡献要求。不能借用初版outside证据，也不能用HateMM32次ROC下降.010930替换主操作点。原始来源本机 `runs/20261001_associative_io_backbone/diagnostics/<corpus>/<arm>/seed<seed>/metrics_test_fixed8.json`，full seed234复用各自source trial；逐seed/标准差 `analysis/<corpus>_locked_diagnostics.json`，两语料配对 `runs/20261001_associative_io_backbone/analysis/two_corpus_{attention,outside}_diagnosis.json`。这些是固定配置诊断，不是Optuna确认。模型原型归档至 `archive/experiments/20261001_associative_io_backbone/`，保留全部原始结果，由残差版继续争取novel且涨点。

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

- **Pursuing goal保持paused，不重新开启。** 保留原heartbeat `scripts/monitor_thread.py`，会话 `01a0f639-b211-75e3-9155-e15e30534b46`，PID `3245799`、间隔3小时；状态/日志 `runs/thread_monitor/01a0f639-b211-75e3-9155-e15e30534b46/`。11:54最近通知成功，下一次14:54 NZDT。本轮确认存活；目标未完成、无硬阻塞。
- **当前四项残差版独立完整Optuna确认搜索**：12:51启动，两语料各seed2025/3407、各20 trial×50 epoch，搜索空间/目标/validation checkpoint/统一评测不变；按test目标选trial，同时报告validation选trial，不依据部分最优下结论。输出 `runs/20261002_residual_io_backbone/<corpus>/seed<seed>/`。HCS seed3407在uoa-lab1 / sc474397，owner/monitor **3369993/209987**；其余在uoa-lab3 / sc474398：HateMM seed2025 **2268717/209995**，HateMM seed3407 **2268962/210038**，HCS seed2025 **2269210/210049**。四个monitor均已首轮RUNNING、绑定正确身份/主机/当前会话；13:30 HCS3407完成14/20、HM2025和HM3407各2/20、HCS2025为5/20，另各1 RUNNING。四个首trial1061.93/1059.48/421.61/170.39秒均自动锁定20，预算已回传，记录 `setup/confirmation_budgets.json`；不使用部分最优作结论。启动/核验记录 `setup/confirmation_launches.json`、`confirmation_startup_check.json`（本版本机runs下）。
- **HateMM锁定trial10的full诊断已完成，nooutside/noresidual两组运行**。full owner/monitor `2270111/212012`已正常结束并成功通知退出，三seed50 epoch及39份评测已回传核验。新启动nooutside owner/monitor **2291136/254919**，noresidual **2292985/257895**，均在uoa-lab3 / sc474398，输出 `runs/20261002_residual_io_backbone/diagnostics/hatemm/<arm>/`。两组各seed234/2025/3407完整50 epoch，配置仅各自io_outside/io_residual设false；13:36实际到seed234 epoch10/3，独立monitor首次RUNNING、身份/会话绑定正确。配置 `setup/hatemm_diagnostic_lock.json`，启动 `hatemm_diagnostic_launches.json`、实训/监控 `hatemm_ablations_startup_check.json`，调度 `hatemm_diagnostic_schedule.json` 已无待启动组。配对差值待各组完整并回传后计算；这些诊断不是Optuna确认，不能修补HCS已失败的两语料共同贡献要求。
- **实时资源与输入**：13:30 lab1 HCS训练中，空闲2018MiB；lab-server空闲2585MiB，已有他人任务不动。full释放后lab3空闲12699MiB；先启动nooutside，实测5376MiB、余7316MiB，再基于noresidual绕过上下文分支的较低需求启动它。13:36两组实际5376/5052MiB，整卡已用29829MiB、空闲2261MiB、利用率99%，所有正式任务继续。lab1 HCS393视频输入/环境对齐记录仍为 `setup/inputs_hateclipseg_lab1.log` 与command.json；torch2.7.1+cu128/CUDA12.8/Optuna4.9.0。本机不训练，没有新增VLM或缓存抽取。
- **残差版已结束部分均已回传核验**：两主数据seed234各20 trial×50 epoch，共520份统一评测；HateMM owner2155073全部同会话子进程退出，monitor4142180于12:44成功通知后退出，进程检查 `setup/hatemm_completion_probe.json`。HCS full/nooutside/noresidual三组各三seed50 epoch、117份评测已完整，三monitor均成功通知退出；见 `setup/hcs_diagnostics_session_check.json`、`hcs_diagnostics_monitor_closure.json`。旧任务不重启，延迟通知不重复训练。
- **前两版均已结束**：初版两语料三seed完整搜索及自身outside诊断保留为机制证据；加性版两语料三seed及诊断全量核验后已归档。原始结果仍在各自本机runs下，不能混为本版本贡献。详情见各实验README和对应analysis汇总。
- `bash scripts/check_layout.sh`开跑前及汇报前检查；三机同步/工作树记录 `runs/20261002_residual_io_backbone/setup/confirmation_prelaunch_sync.json`、`hatemm_full_final_sync.json`。本轮回传核验HateMM full并使用既有入口启动两组已锁定消融，只更新结果文档；模型/共享trainer/统一评测器未改。已知无关 `tandem.html`、lab1 `idea-stage/` 和家目录STRAY不改动、不清理。

## 下一步

1. 四项残差版2025/3407独立20-trial搜索完整跑满，由各自monitor接续；结束核验并回传后汇总两主数据三seed、标准差和validation选trial结果。HateMM nooutside/noresidual各三seed50 epoch已启动并绑定独立monitor，完成后回传核验，与本版full逐seed配对并汇总两语料贡献，不借用旧版证据。
2. 初版有两语料机制贡献但相对第5版掉点；加性版三seed性能与自身机制证据均未满足主线目标，已归档。继续优先争取novel且涨点，同时报告AP/ROC/within与0/8次，32次只附加；不能把任何单seed或单预算改善当成整体目标已完成。
3. 等待交给后台monitor，不反复调用模型轮询。最终核对新骨干贡献、性能代价、成本及论文操作点后汇报；Goal保持paused，研究规则不修改。

## 历史与规则

[实验详细记录](../experiments/20260925_query_paradigm/README.md)、[本次整理前状态](../archive/research-wiki/STATUS_20261001_before_current_summary.md)、[研究规则](../RESEARCH_ITERATION_RULES.md)、[固定 baseline 表](../docs/duplex/OFFICIAL_VAL_RESULTS.md)、[DeHate 评测协议](../docs/duplex/FRAME_EVAL_PROTOCOL_DEHATE.md)。旧结论与其它历史链接保留在归档，不在当前状态追加时间线。
