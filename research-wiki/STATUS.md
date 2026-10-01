# 当前研究状态

截至 **2026-10-02 01:11 NZDT**。依据：`experiments/20260925_query_paradigm/` 第 5 版软答案、复制似然及停止规则实现，实验 README 第 17 节，以及下列本机 `runs/` 评测器输出。2026-10-01 核验了 8 / 16 档共 18 个 study、每个 20 个 COMPLETE trial 的固定 8 次评测文件；停止规则汇总也与逐 trial 评测器输出核对。初版 inside–outside 骨干两主数据 seed234 均已完成20/20 trial并回传核验，达到固定baseline筛选门，其余seed完整搜索已启动；加性注意力统计量修订版HCS已完成20/20，HateMM继续；没有新增抽取。

## 当前目标与结论

当前开发主线是**按需提问定位第 5 版：单问软答案，8 档**。HateMM、HateClipSeg 为主数据集，DeHate 仅为外部验证。旧候选 3 修订 4 及后续模块一实验是历史参照，不再用它们描述最新进展；最终论文主表口径尚未锁定。

**用户 2026-10-01 最新授权：自动迭代 backbone，优先 novel 且涨点；充分尝试仍无提升时优先可验证的新意，完成后汇报。within 与 pooled AP / ROC 并列为主指标。**[第一轮：时间树上的区间内外表示](../experiments/20261001_inside_outside_backbone/README.md)已实现，用区间内组合与区间外上下文学习内容先验，复用现有特征与答案，新增 VLM 调用为 0。[一次独立 proposal review：GO](../docs/reviews/20261001_inside_outside_backbone_proposal.md)，[一次独立 code review：PASS](../docs/reviews/20261001_inside_outside_backbone_code.md)；完整搜索协议已在 README 第 6 节预写，两主数据seed234已跑满并回传，按既有规则补seed确认；相对第5版未实现涨点，不能宣布novelty完成。创新主张限定为具体问题树上的 inside–outside 表示与答案似然训练。旧版无骨干消融只用于动机，不能代替新骨干验证。共享 QTL 基础设施已升入 `src/qtl/`，新骨干在新实验目录，旧入口仍兼容；评测器未修改。

- 第 5 版三语料各三 seed × 20 trial 已完成。相对第 4 版固定 8 次，AP 分别 +.004 / +.017 / +.037；ROC −.007 / −.008 / +.026；within +.045 / −.011 / +.015。软答案改善 AP，但未实现全部指标不降。
- **此前完成的停止实验是 DeHate 16 档搜索及停止阈值验证**。最终验证链在 2026-10-01 02:25 NZDT 正常结束，日志末尾 ALL_DONE；结果已回传本机，文件可解析且三 seed 齐全。16 档相对 8 档在两主数据 AP 更高，但 DeHate AP −.014；沿用此前决定，8 档保持主线，16 档只作敏感性结果。
- 自动停止候选 `qmixSG_rt10` 不设最少调用次数，在 validation 上按平均 8 次定阈值，配合复制似然，三语料 8 档与同设置固定 8 次大致持平。实际平均调用约 7.8–8.1，**尚未证明稳定省调用**。DeHate 16 档额外检验 within 差值为 −.00503，略低于该检查的 −.005 容差，不能写全部设置通过。
- 用 validation 标注直接选停止阈值也已完成：HCS / DeHate 常选 12–24 次，改善主要来自更多调用；HateMM 常选 1–6 次，16 档按三指标和的容差规则选到 2.45 次，AP / ROC / within 掉 .025 / .017 / .015。仍未得到稳健的统一选阈值方案。

## 三模块实现与缺口

| 模块 | 当前实现 | 证据与缺口 |
|---|---|---|
| VLM 观测与提问 | 二分时间树、按期望信息增益选节点；词级时间戳转录；单问首 token P(Yes) 软答案、8 档 | HateMM 视频内排序改善；五类软答案两主数据均下降，未采用；多视图估计未解决 8 次的 ROC 下降 |
| 弱监督骨干与先验 | 主线保留第 5 版；inside–outside 初版两语料seed234已完成，其余seed确认已启动；加性注意力统计量修订版HCS已完成，HateMM仍搜索 | 初版HCS pooled下降；修订版恢复部分pooled但within下降，两个版本尚无两语料机制贡献确认 |
| 答案融合与停止 | 锚定答案模型；复制似然处理嵌套回答相关；变化、排序变化、视频后验熵的分位数组合停止 | 复制似然缓解 HateMM 多问变差；放宽停止阈值后约持平，但有手设衰减率 .10、validation 目标 / 容差与实际调用量问题 |

代码与机制、消融及逐项检查见 [实验 README 第 14–17 节](../experiments/20260925_query_paradigm/README.md)。当前优先工作转向 backbone 结构与定位贡献；停止实验结论保留，不能将整套第 5 版称为已完成全部论文验证。

## 最新权威结果与来源

**新骨干 HCS seed234 已完成完整 20 trial**：test 选 trial11，AP / ROC / within 为 **.668767 / .635762 / .567188**；相对第 5 版同 seed trial13 为 **−.019700 / −.039083 / +.011057**，尚未涨点。只按 validation 选 trial6 为 .648418 / .616496 / .571958。来源：本机 `runs/20261001_inside_outside_backbone/hateclipseg/seed234/trial{11,6}/metrics_test_fixed8.json`，旧线 `runs/20260929_query_paradigm_r5/hateclipseg/seed234/trial13/metrics_test_fixed8.json`。HateMM现已完成，结果如下；不能宣告三seed确认或novelty贡献完成。HCS 固定配置三seed诊断已全部完成并回传；8次均值AP / ROC / within：full **.653391 / .623460 / .573357**，nooutside **.641279 / .614138 / .556733**，mean **.654397 / .626035 / .576118**。原始来源 `runs/20261001_inside_outside_backbone/diagnostics/hateclipseg/<arm>/seed<seed>/metrics_test_fixed8.json`，full seed234复用trial11；标准差与各seed见实验README第8节和 `analysis/hateclipseg_locked_diagnostics.json`。outside有平均效应支持，但AP收益主要由一个seed贡献；递归门控未胜过均值，不能当成已证实贡献。这些固定配置诊断不替代Optuna确认。由此提出[保留加性注意力统计量的inside–outside修订](../experiments/20261001_associative_io_backbone/README.md)，独立proposal review已GO、code review已PASS，两主数据seed234完整搜索于23:05在lab3启动；新增VLM调用0。

**初版HateMM seed234已完成完整20 trial**：test选trial17，AP / ROC / within **.679560 / .881351 / .753400**；相对第5版同seed trial9为 **−.014890 / −.005354 / −.035333**。validation选trial10为.635249/.869817/.753563。来源本机 `runs/20261001_inside_outside_backbone/hatemm/seed234/trial{17,10}/metrics_test_fixed8.json`，旧线 `runs/20260929_query_paradigm_r5/hatemm/seed234/trial9/metrics_test_fixed8.json`；全部50 epoch和260份评测完整性记录在 `runs/20261001_inside_outside_backbone/analysis/hatemm_seed234_comparison.json`。两语料四个pooled数超过既定baseline筛选门，按规则补seed2025/3407完整搜索；它们仍低于当前主线，确认只用于核对seed稳定性，不等于涨点或novelty完成。修订版继续独立搜索。

**加性统计量修订版HCS seed234已完成20/20**：test选trial7，AP / ROC / within **.683294 / .655506 / .511165**；对初版+.014527/+.019744/−.056023，对第5版−.005173/−.019339/−.044966，不能算全面涨点。validation选trial10为.630660/.613407/.510797。trial7的0次仅.502969/.452038/.516390，checkpoint由validation选在epoch1，不能把8次pooled改善当成先验定位贡献。来源本机 `runs/20261001_associative_io_backbone/hateclipseg/seed234/trial{7,10}/metrics_test_fixed8.json`；260份评测和全部50 epoch核验及误差分析在该实验 `analysis/hateclipseg_seed234_comparison.json`、`hateclipseg_prior_diagnosis.json`。修订版HateMM尚未完成；HCS full/nooutside/noattention三seed固定配置诊断已锁定trial7并准备入口，等待计算资源，详情见修订版README第7节。这些诊断不是Optuna确认。

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

- **Pursuing goal 保持 paused，不重新开启。** 已复用原有 `scripts/monitor_thread.py`，当前会话 `01a0f639-b211-75e3-9155-e15e30534b46`，每 3 小时接续；PID `3245799`，日志和状态在 `runs/thread_monitor/01a0f639-b211-75e3-9155-e15e30534b46/`。首条及23:54定时 `codex queue` 通知已成功接续。当前目标未完成：仍缺初版三seed确认、修订版完整搜索和两语料机制贡献验证；没有硬阻塞，保留heartbeat和活动run monitor。
- **初版骨干完整搜索**：两语料于20:57在 uoa-lab3 / sc474398 启动。HCS 于22:14完成20/20 trial，已核验真实主/子进程结束、全部50 epoch历史和完整评测并回传本机。HateMM于10月2日00:15:43完成20/20，已核验真实主/子进程退出并回传全部输出；每trial50 epoch、214视频/29269秒、260份统一评测完整。目录为 `runs/20261001_inside_outside_backbone/{hatemm,hateclipseg}/seed234/`。首 trial HateMM 559.52 秒、HCS 235.49 秒，两套预算各固定20，文件 `budget.json` 已在本机。
- **完成通知**：HCS monitor `3249811` 已正常发出22:16完成通知并退出，有 `notification_sent` 标记；HateMM monitor `3249810` 已于00:16成功通知并退出。状态在本机 `runs/20261001_inside_outside_backbone/<corpus>/seed234/monitor/`。每120秒检查真实进程身份/相关子进程，SSH失败只重试；完成通知自动接续已实际验证，heartbeat仍作兜底。
- **初版其余seed确认运行中**：10月2日00:22在lab3启动四项独立完整Optuna。HateMM seed2025/3407 owner `1742323 / 1742324`、monitor `3478306 / 3478307`；HCS seed2025/3407 owner `1742325 / 1742326`、monitor `3478308 / 3478309`。输出 `runs/20261001_inside_outside_backbone/<corpus>/seed<seed>/`，监控记录在对应本机 `monitor/`，启动记录 `runs/20261001_inside_outside_backbone/setup/confirmation_launches.json`。已确认四个monitor存活且首次RUNNING绑定正确，训练覆盖完整并进入epoch；复用原50 epoch和搜索空间，01:11核验HateMM各2/20、HCS各5/20完成，各有后续trial正常训练。首trial耗时HateMM 1424.30/1424.47秒、HCS 577.19/577.01秒，四项预算各20，budget.json已回传。不是固定配置诊断，不覆盖seed234。
- **修订版完整搜索**：`runs/20261001_associative_io_backbone/{hatemm,hateclipseg}/seed234/`，uoa-lab3 / sc474398，10月1日23:05启动。HCS于10月2日01:06:01完成20/20，真实主/子进程退出，全部输出已回传并核验50 epoch和260份评测；monitor `3392350` 已成功通知退出。HateMM owner `1704438`、monitor `3392349` 仍正常，01:11已完成8/20、trial8正常训练。两套预算20，首trial756.27/314.65秒；不是独立吞吐比较。monitor记录在对应本机run目录，启动记录 `runs/20261001_associative_io_backbone/setup/launches.json`。
- **待调度诊断**：修订版HCS的full/nooutside/noattention已锁定配置和完整50 epoch三seed协议，启动入口 `experiments/20261001_associative_io_backbone/launch/run_diag_hateclipseg_uoa-lab3.sh <arm>`，源trial7；full seed234复用，新增8次完整训练，无新增VLM调用。现有五项搜索GPU约98%使用，不以加并发拖慢所有任务；下一项结束或有可用GPU时调度并绑定独立monitor。尚未启动，不宣称已配置这三项run monitor。
- **HCS结构诊断已完成**：full于22:38结束，nooutside / mean于22:42结束，真实主/子进程均已核验退出；三组各三seed、各50 epoch、配置和0/8/32次统一评测完整性均通过，全部输出已回传。对应monitor `3347291 / 3347292 / 3347293` 已通知退出；输出 `runs/20261001_inside_outside_backbone/diagnostics/hateclipseg/<arm>/`，监控记录在本机对应 `monitor/`。不重复启动或把固定配置诊断当成Optuna确认。
- 开跑前两套缓存实际解析、形状、时间轴、完整覆盖和 split isolation 均通过：HateMM 1067 个视频（744 / 109 / 214），HCS 393 个（251 / 63 / 79）。日志已回传本机 `runs/20261001_inside_outside_backbone/setup/inputs_{hatemm,hateclipseg}.log`。本机与 lab3 环境均为 torch 2.7.1+cu128、CUDA 12.8、Optuna 4.9.0。
- 资源检查：本机继续不跑训练。lab1 另有 VLLM 占约 28 GiB，仅余 3.7 GiB；lab-server 仅余 2.6 GiB，不参与本轮；未改动他人任务。lab3 原有 lightmem 任务占约5 GiB；01:11五项搜索并行整卡约13.5 GiB、利用率98%，已核验实际训练正常，未见OOM。
- `bash scripts/check_layout.sh` 开跑前已执行。多机代码同步检查：本机、lab1、lab3 均为 `24ef51e`，确认study启动时跟踪文件干净；本次仅在初版全部旧任务结束后将其融合/拓扑和owner接入共享定义，并更新文档；修订版活动代码、共享trainer与评测器文件未改动；本机未跟踪的 `tandem.html` 与 lab1 的 `idea-stage/` 为已有无关内容，未改动。运行后仅本机更新 STATUS / 实验 README，未替换活动实验代码。已查看的家目录存量 `conversation-pilot` / `ctx_pri`、远端 `MemoryAgen` / `ctx_pri_*`、工具数据、旧 `list` / `vllm.pid` 保留，没有擅自清理。

## 下一步

1. 初版四项确认搜索与修订版HateMM继续原预算；下一项结束后核验回传，并调度已就绪的修订版HCS诊断（优先检查outside与注意力权重的贡献），每项绑定独立monitor。HCS结果不能代替修订版HateMM完成或三seed Optuna确认。
2. 依完整搜索结果筛选、确认、分析误差并修改；优先提升 AP / ROC / within，候选不能仅凭 proposal GO 算完成。满足确认后补结构消融，分别验证 outside 与注意力权重的贡献；随后检查现有停止设置。
3. 保留全部授权和研究目标，等待只交给后台 monitor，不通过模型反复轮询；最终核对新骨干贡献、性能代价、成本及论文操作点后汇报。现有训练/搜索规则文件不修改。

## 历史与规则

[实验详细记录](../experiments/20260925_query_paradigm/README.md)、[本次整理前状态](../archive/research-wiki/STATUS_20261001_before_current_summary.md)、[研究规则](../RESEARCH_ITERATION_RULES.md)、[固定 baseline 表](../docs/duplex/OFFICIAL_VAL_RESULTS.md)、[DeHate 评测协议](../docs/duplex/FRAME_EVAL_PROTOCOL_DEHATE.md)。旧结论与其它历史链接保留在归档，不在当前状态追加时间线。
