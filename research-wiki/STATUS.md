# 当前研究状态

截至 **2026-09-05 14:15 NZST**。依据：HateClipSeg seed234完整20trial原始输出核验、HateMM抽取实时进程。每轮结束替换对应条目，不追加流水账。

## 目标与当前结论

**目标尚未完成。** 等当前实验完成后继续开发：两语料达到研究规则的 SOTA；VLM、骨干、融合三个模块各有消融支持的 novelty；整体有统一机制支撑 novel paradigm；方法超参数尽可能少。

可靠起点是候选1。候选4已淘汰；候选5[干预证据方法](../experiments/20260905_interventional_evidence/README.md)HateClipSeg seed234搜索完成并通过本语料门，但低于候选1；HateMM仍抽取，尚不能做两语料筛选或确认。候选1达到开发期性能门，但三模块创新证据不足。

协议唯一来源：[研究规则](../RESEARCH_ITERATION_RULES.md)。主数据集为 HateMM、HateClipSeg；within 硬门仍有效，取消硬门的讨论尚未裁定。三个模块均须 novelty 的新要求覆盖[旧计划](../docs/20260903_three_module_program.md)中模块 1 可选的要求。

## 当前方法：三个模块

| 模块 | 实现 | 尚缺证据 |
|---|---|---|
| 1 VLM | 冻结 Qwen2.5-VL-7B，K30/K4 两粒度裁定 | 尚无已确认的模块创新；粗块 ASR 上下文变体已失败 |
| 2 骨干 | MACIL-SD + 裁定块级 MIL；最新尝试空 token 跨模态注意力 | 块级 MIL 有效；新结构尚未在两语料稳定成立 |
| 3 融合 | 分层证据 HMM 后验作为 logit 先验，与骨干分数融合 | 先验路径有效；HMM 相对简单平均先验在 HateMM 的优势未成立 |

代码入口：候选1 [README](../experiments/20260903_hier_evidence_mil/README.md) / [train.py](../experiments/20260903_hier_evidence_mil/train.py)。候选4 [归档README](../archive/experiments/20260904_null_token_cma/README.md)第8.5节为最终结果。共享实现：[输入/损失/评测调用](../src/hier_evidence_common.py)、[HMM](../src/verdict_hmm.py)。上表描述已运行方法；候选5为VLM干预证据、正/负关联双向读出骨干、Yager冲突转未知的训练内融合。[训练入口](../experiments/20260905_interventional_evidence/train.py)、[搜索入口](../experiments/20260905_interventional_evidence/search.py)已通过[code review修复确认](../experiments/20260905_interventional_evidence/REVIEW_RULE6.md)，尚无训练结果；负关联不等于语义反证。

## 已核验结果

候选 1 三 seed 均值，顺序为 **pooled AP / pooled ROC / within ROC**。每 seed 20 trial，trial 内 validation 选 checkpoint；开发搜索按 test 选 trial，不是未揭盲确认结果。

| 语料 | 按 test 选 trial（均值 ± std） | 仅按 validation 选 trial（均值） |
|---|---|---|
| HateMM | .657 ± .013 / .842 ± .005 / .646 ± .004 | .601 / .821 / .627 |
| HateClipSeg | .699 ± .006 / .681 ± .016 / .553 ± .007 | .686 / .665 / .538 |

来源：[候选 1 输出](../runs/20260903_hier_evidence_mil/)，`<hatemm|hateclipseg>/seed<234|2025|3407>/study_summary.json` 的 `best`、`validation_selected`；对应 trial 的 `metrics.json` 为评测器原始输出。Baseline 对照：[固定表](../docs/duplex/OFFICIAL_VAL_RESULTS.md)。

候选4修订1最终三seed：HateMM **.644±.025 / .840±.004 / .642±.007**；HateClipSeg **.696±.001 / .681±.012 / .542±.007**（std ddof=1）。validation选trial分别为 **.611/.825/.627**、**.686/.664/.538**。未达到相对候选1的预注册提升；HateClipSeg结构消融不支持空token，归档。来源：[核验汇总](../runs/20260904_null_token_cma/rev1/artifact_audit.json)指向各trial/消融的原始`metrics.json`，120 trial与18消融均齐全。

## 运行与监控

候选5 HateClipSeg seed234：按test选trial18 **.655/.637/.540**，仅按validation选trial4 **.650/.633/.540**。20 trial全部COMPLETE且原始输出齐全；单seed、开发期test搜索，不是确认结果。来源：[完整审计](../runs/20260905_interventional_evidence/hateclipseg/seed234/artifact_audit.json)、[trial18评测](../runs/20260905_interventional_evidence/hateclipseg/seed234/trial18/metrics.json)、[trial4评测](../runs/20260905_interventional_evidence/hateclipseg/seed234/trial4/metrics.json)。

| 任务 | 当前状态 | 输出/日志 |
|---|---|---|
| 候选5 HateMM v2抽取，lab1 | 修复截断的hate_video_95后续跑，PID/PGID 1284916；422/1068完整缓存已回传审计通过，自动跳过已完成项 | [输入审计](../runs/20260905_interventional_evidence/extract_hatemm_v2/input_audit.json)、[恢复monitor](../runs/20260905_interventional_evidence/extract_hatemm_v2/monitor_resume1/run.log)；诊断见候选5README第5节 |
| 候选5 HateClipSeg v2输入 | 抽取进程已退出；393/393视频、786/786文件完整回传并解析通过 | [完整输入审计](../runs/20260905_interventional_evidence/extract_hateclipseg_v2/input_audit.json) |
| 候选5 HateClipSeg seed234，lab3 | 搜索进程已退出；20trial全部完整回传审计，首trial118.659秒 | [完整输出](../runs/20260905_interventional_evidence/hateclipseg/seed234/) |
| 候选5 HateClipSeg单seed消融，lab3 | 锁定trial18配置，准备8个已评审诊断臂、三任务并行；不提前补确认seed | [启动脚本](../experiments/20260905_interventional_evidence/launch/run_hcs_ablations_seed234_lab3.sh) |
| lab3输入准备 | 已完成；393视频头、K30/K4 ASR全覆盖、5模型分片可解析；webm规范别名已修复 | [本机核验输出](../runs/20260905_interventional_evidence/prepare_lab3/coverage.json) |
| 长期会话 monitor | 13:04提醒已送达并处理；目标未完成、无硬阻塞，保留；下次16:04 NZST | [状态与日志](../runs/thread_monitor/01a06df5-3e92-79b0-be30-820db943e551/) |

等待时不由模型持续轮询、不重复创建 monitor，用户无需手动设置 Goal。目标完成或确认无法推进的硬阻塞时关闭长期 monitor 并报告；正常等待和暂时断连不算硬阻塞。关闭命令：

```bash
python3 scripts/monitor_thread.py --thread 01a06df5-3e92-79b0-be30-820db943e551 --out-dir runs/thread_monitor/01a06df5-3e92-79b0-be30-820db943e551 --stop
```

## 下一步

1. 并行推进HateClipSeg锁定配置的模块诊断与HateMM输入抽取；后者完成后审计并独立启动seed234。两语料筛选齐全前不提前补确认seed；单seed消融不用于宣称novelty。
2. 唯一code review三项修复已确认：原始logits/版本隔离、固定评测覆盖、补seed继承234预算。v1进程与monitor已停，缓存/日志保留并回传（详见候选5README第5节），不得混入训练。无需重审。
3. 完整搜索、三seed确认及三个模块替换消融；分别列出方法超参数与通用优化参数，以实验检验范式主张，不凭包装判定完成。

## 历史与资料

- [整理前 STATUS 完整快照](../archive/research-wiki/STATUS_20260905_before_cleanup.md)：旧候选、旧规则、数据与归档路径索引，仅供追溯。
- [评测协议](../docs/duplex/FRAME_EVAL_PROTOCOL.md)、[缓存出处](../data/MLLM_scores/PROVENANCE.md)。原始视频在 `~/data/`；冻结的 Hate-follow-up 溯源路径保持不动。
