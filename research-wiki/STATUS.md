# 当前研究状态

截至 **2026-09-05 11:05 NZST**。依据：候选1源码、候选4已回传并核验的120个trial/18个消融输出、候选5提案。每轮结束替换对应条目，不追加流水账。

## 目标与当前结论

**目标尚未完成。** 等当前实验完成后继续开发：两语料达到研究规则的 SOTA；VLM、骨干、融合三个模块各有消融支持的 novelty；整体有统一机制支撑 novel paradigm；方法超参数尽可能少。

可靠起点是候选1。候选4空token修订1已按预注册淘汰并归档；候选5[干预证据方法提案](../experiments/20260905_interventional_evidence/README.md)正在独立proposal review，尚未实现。候选1达到开发期性能门，但三模块创新证据不足。

协议唯一来源：[研究规则](../RESEARCH_ITERATION_RULES.md)。主数据集为 HateMM、HateClipSeg；within 硬门仍有效，取消硬门的讨论尚未裁定。三个模块均须 novelty 的新要求覆盖[旧计划](../docs/20260903_three_module_program.md)中模块 1 可选的要求。

## 当前方法：三个模块

| 模块 | 实现 | 尚缺证据 |
|---|---|---|
| 1 VLM | 冻结 Qwen2.5-VL-7B，K30/K4 两粒度裁定 | 尚无已确认的模块创新；粗块 ASR 上下文变体已失败 |
| 2 骨干 | MACIL-SD + 裁定块级 MIL；最新尝试空 token 跨模态注意力 | 块级 MIL 有效；新结构尚未在两语料稳定成立 |
| 3 融合 | 分层证据 HMM 后验作为 logit 先验，与骨干分数融合 | 先验路径有效；HMM 相对简单平均先验在 HateMM 的优势未成立 |

代码入口：候选1 [README](../experiments/20260903_hier_evidence_mil/README.md) / [train.py](../experiments/20260903_hier_evidence_mil/train.py)。候选4 [归档README](../archive/experiments/20260904_null_token_cma/README.md)第8.5节为最终结果。共享实现：[输入/损失/评测调用](../src/hier_evidence_common.py)、[HMM](../src/verdict_hmm.py)、[空token](../src/null_token_cma.py)。上表描述已运行方法；候选5计划分别改为VLM干预证据、支持/反证双向读出骨干、冲突转未知的训练内融合，均尚未验证。

## 已核验结果

候选 1 三 seed 均值，顺序为 **pooled AP / pooled ROC / within ROC**。每 seed 20 trial，trial 内 validation 选 checkpoint；开发搜索按 test 选 trial，不是未揭盲确认结果。

| 语料 | 按 test 选 trial（均值 ± std） | 仅按 validation 选 trial（均值） |
|---|---|---|
| HateMM | .657 ± .013 / .842 ± .005 / .646 ± .004 | .601 / .821 / .627 |
| HateClipSeg | .699 ± .006 / .681 ± .016 / .553 ± .007 | .686 / .665 / .538 |

来源：[候选 1 输出](../runs/20260903_hier_evidence_mil/)，`<hatemm|hateclipseg>/seed<234|2025|3407>/study_summary.json` 的 `best`、`validation_selected`；对应 trial 的 `metrics.json` 为评测器原始输出。Baseline 对照：[固定表](../docs/duplex/OFFICIAL_VAL_RESULTS.md)。

候选4修订1最终三seed：HateMM **.644±.025 / .840±.004 / .642±.007**；HateClipSeg **.696±.001 / .681±.012 / .542±.007**（std ddof=1）。validation选trial分别为 **.611/.825/.627**、**.686/.664/.538**。未达到相对候选1的预注册提升；HateClipSeg结构消融不支持空token，归档。来源：[核验汇总](../runs/20260904_null_token_cma/rev1/artifact_audit.json)指向各trial/消融的原始`metrics.json`，120 trial与18消融均齐全。

## 运行与监控

| 任务 | 当前状态 | 输出/日志 |
|---|---|---|
| lab1：候选4修订1，HateMM | 10:54全部完成，链进程已退出；结果已回传并核验 | [输出](../runs/20260904_null_token_cma/rev1/hatemm/) |
| lab3：同修订 HateClipSeg | 三 seed 搜索与消融全部结束 | [已回传输出](../runs/20260904_null_token_cma/rev1/hateclipseg/) |
| 实验完成 monitor | 10:56通知已送达本会话，单次监控完成 | [日志](../runs/20260904_null_token_cma/rev1/monitor_codex/run.log) |
| 候选5 proposal review | 独立agent进行中；通过后才实现和抽取 | [提案](../experiments/20260905_interventional_evidence/README.md) |
| 长期会话 monitor | 本机 PID 1177638 存活，每3小时提醒推进目标；首次今天13:04 NZST | [状态与日志](../runs/thread_monitor/01a06df5-3e92-79b0-be30-820db943e551/) |

等待时不由模型持续轮询、不重复创建 monitor，用户无需手动设置 Goal。目标完成或确认无法推进的硬阻塞时关闭长期 monitor 并报告；正常等待和暂时断连不算硬阻塞。关闭命令：

```bash
python3 scripts/monitor_thread.py --thread 01a06df5-3e92-79b0-be30-820db943e551 --out-dir runs/thread_monitor/01a06df5-3e92-79b0-be30-820db943e551 --stop
```

## 下一步

1. 完成候选5独立proposal review，落实修改；若触发规则4 STOP则更换提案。
2. 放行后实现并做一次code review，准备VLM输入；独立抽取/两语料任务尽量分配所有可用GPU并自动配置monitor。
3. 完整搜索、三seed确认及三个模块替换消融；分别列出方法超参数与通用优化参数，以实验检验范式主张，不凭包装判定完成。

## 历史与资料

- [整理前 STATUS 完整快照](../archive/research-wiki/STATUS_20260905_before_cleanup.md)：旧候选、旧规则、数据与归档路径索引，仅供追溯。
- [评测协议](../docs/duplex/FRAME_EVAL_PROTOCOL.md)、[缓存出处](../data/MLLM_scores/PROVENANCE.md)。原始视频在 `~/data/`；冻结的 Hate-follow-up 溯源路径保持不动。
