# 当前研究状态

截至 **2026-10-01 20:54 NZDT**。依据：`experiments/20260925_query_paradigm/` 第 5 版软答案、复制似然及停止规则实现，实验 README 第 17 节，以及下列本机 `runs/` 评测器输出。2026-10-01 核验了 8 / 16 档共 18 个 study、每个 20 个 COMPLETE trial 的固定 8 次评测文件；停止规则汇总也与逐 trial 评测器输出核对。无新增训练或抽取。

## 当前目标与结论

当前开发主线是**按需提问定位第 5 版：单问软答案，8 档**。HateMM、HateClipSeg 为主数据集，DeHate 仅为外部验证。旧候选 3 修订 4 及后续模块一实验是历史参照，不再用它们描述最新进展；最终论文主表口径尚未锁定。

**用户 2026-10-01 最新授权：自动迭代 backbone，优先 novel 且涨点；充分尝试仍无提升时优先可验证的新意，完成后汇报。within 与 pooled AP / ROC 并列为主指标。**[第一轮：时间树上的区间内外表示](../experiments/20261001_inside_outside_backbone/README.md)已实现，用区间内组合与区间外上下文学习内容先验，复用现有特征与答案，新增 VLM 调用为 0。[一次独立 proposal review：GO](../docs/reviews/20261001_inside_outside_backbone_proposal.md)，[一次独立 code review：PASS](../docs/reviews/20261001_inside_outside_backbone_code.md)；完整搜索协议已在 README 第 6 节预写。创新主张限定为具体问题树上的 inside–outside 表示与答案似然训练。旧版无骨干消融只用于动机，不能代替新骨干验证。共享 QTL 基础设施已升入 `src/qtl/`，新骨干在新实验目录，旧入口仍兼容；评测器未修改。

- 第 5 版三语料各三 seed × 20 trial 已完成。相对第 4 版固定 8 次，AP 分别 +.004 / +.017 / +.037；ROC −.007 / −.008 / +.026；within +.045 / −.011 / +.015。软答案改善 AP，但未实现全部指标不降。
- **最新完成的是 DeHate 16 档搜索及停止阈值验证**。最终验证链在 2026-10-01 02:25 NZDT 正常结束，日志末尾 ALL_DONE；结果已回传本机，文件可解析且三 seed 齐全。16 档相对 8 档在两主数据 AP 更高，但 DeHate AP −.014；沿用此前决定，8 档保持主线，16 档只作敏感性结果。
- 自动停止候选 `qmixSG_rt10` 不设最少调用次数，在 validation 上按平均 8 次定阈值，配合复制似然，三语料 8 档与同设置固定 8 次大致持平。实际平均调用约 7.8–8.1，**尚未证明稳定省调用**。DeHate 16 档额外检验 within 差值为 −.00503，略低于该检查的 −.005 容差，不能写全部设置通过。
- 用 validation 标注直接选停止阈值也已完成：HCS / DeHate 常选 12–24 次，改善主要来自更多调用；HateMM 常选 1–6 次，16 档按三指标和的容差规则选到 2.45 次，AP / ROC / within 掉 .025 / .017 / .015。仍未得到稳健的统一选阈值方案。

## 三模块实现与缺口

| 模块 | 当前实现 | 证据与缺口 |
|---|---|---|
| VLM 观测与提问 | 二分时间树、按期望信息增益选节点；词级时间戳转录；单问首 token P(Yes) 软答案、8 档 | HateMM 视频内排序改善；五类软答案两主数据均下降，未采用；多视图估计未解决 8 次的 ROC 下降 |
| 弱监督骨干与先验 | 问题树似然训练、零膨胀链、树节点先验势 | 第 4 版已有三 seed 消融；第 5 版不能直接把前版消融当成最终配置验证 |
| 答案融合与停止 | 锚定答案模型；复制似然处理嵌套回答相关；变化、排序变化、视频后验熵的分位数组合停止 | 复制似然缓解 HateMM 多问变差；放宽停止阈值后约持平，但有手设衰减率 .10、validation 目标 / 容差与实际调用量问题 |

代码与机制、消融及逐项检查见 [实验 README 第 14–17 节](../experiments/20260925_query_paradigm/README.md)。当前优先工作转向 backbone 结构与定位贡献；停止实验结论保留，不能将整套第 5 版称为已完成全部论文验证。

## 最新权威结果与来源

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

- **用户最新裁定：关闭 Pursuing goal，改用事先写好的 heartbeat 接续研究。** Goal 状态已设为 paused，不重新开启。复用 `scripts/monitor_thread.py`，绑定当前会话 `01a0f639-b211-75e3-9155-e15e30534b46`，每 3 小时通过 `codex queue` 提醒，已确认后台进程存活、首条接续消息入队成功；状态、日志、PID 在 `runs/thread_monitor/01a0f639-b211-75e3-9155-e15e30534b46/`。新骨干代码已通过独立评审，尚未提交同步或启动训练；下一次接续从此处继续。实验启动后另按规则绑定完成事件 monitor。
- 本次实时检查 uoa-lab1 / uoa-lab3：未发现本轮搜索、停止检查或对应 monitor 在运行。最后输出是 `runs/20260929_query_paradigm_r5/stop_check/run_sc474398_sel_b.log`（02:25，ALL_DONE）及 `dehate_lv16sel.json`；已在本机逐 trial 核对评测输出，不仅依赖 DONE 标记。
- 本机未启动计算任务，沿用此前“本机不跑”的安排。uoa-lab1 GPU 利用率 0%，但另有 VLLM 进程占约 28 GiB，不能按空卡调度；uoa-lab3 GPU 利用率 0%、占约 5 GiB。未改动这些进程；lab-server 本轮未用于实验，未检查。
- `bash scripts/check_layout.sh` 已运行。代码同步检查：本机 `727fb08`，uoa-lab1 `26a4c84`，uoa-lab3 `d726253`。uoa-lab1 落后的是后续停止分析脚本与文档；uoa-lab3 与本机检查前只差文档。检查前跟踪文件均干净；本机未跟踪 `tandem.html`，uoa-lab1 未跟踪 `idea-stage/`，uoa-lab3 无未跟踪文件。已查看具体差异与文件元数据；本次只改状态文档、归档旧状态及修正 README 转录值，未替换远端代码。
- layout 报出的家目录存量包括本机 `conversation-pilot` / `ctx_pri`、远端 `MemoryAgen` / `ctx_pri_*`、工具数据与旧 `list` / `vllm.pid`。未擅自清理；新实验启动前需重新检查进程、GPU、代码同步及相关文件归属。本轮没有新长任务，无需新建 monitor。

## 下一步

1. 本轮唯一一次 code review 已通过，代码同步后在 lab3 并行开两主数据 seed234 完整搜索；按首个完整 trial 耗时锁定试验数，同步启动独立后台 monitor。
2. 依完整搜索结果筛选、确认、分析误差并修改；优先提升 AP / ROC / within，候选不能仅凭 proposal GO 算完成。满足确认后补结构消融，分别验证 outside 与有序组合；随后检查现有停止设置。
3. 保留全部授权和研究目标，等待只交给后台 monitor，不通过模型反复轮询；最终核对新骨干贡献、性能代价、成本及论文操作点后汇报。现有训练/搜索规则文件不修改。

## 历史与规则

[实验详细记录](../experiments/20260925_query_paradigm/README.md)、[本次整理前状态](../archive/research-wiki/STATUS_20261001_before_current_summary.md)、[研究规则](../RESEARCH_ITERATION_RULES.md)、[固定 baseline 表](../docs/duplex/OFFICIAL_VAL_RESULTS.md)、[DeHate 评测协议](../docs/duplex/FRAME_EVAL_PROTOCOL_DEHATE.md)。旧结论与其它历史链接保留在归档，不在当前状态追加时间线。
