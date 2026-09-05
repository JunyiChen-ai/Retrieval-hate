# 当前研究状态

截至 **2026-09-05 23:52 NZST**。依据：候选5两语料三seed全部回传审计、候选6提案/实现及实时多机检查。每轮结束替换对应条目，不追加流水账。

## 目标与当前结论

**目标尚未完成。** 等当前实验完成后继续开发：两语料达到研究规则的 SOTA；VLM、骨干、融合三个模块各有消融支持的 novelty；整体有统一机制支撑 novel paradigm；方法超参数尽可能少。

候选5[干预证据方法](../experiments/20260905_interventional_evidence/README.md)两语料均通过三seed数值确认门，但三个核心模块均未通过HCS三seed有效性要求：seed3407替换后AP/ROC均略升，融合平均AP增益也不足.01。HateMM seed234三个核心替换AP降.161/.025/.028，仅为该seed支持，不能抵消HCS结果。停止为其主张追加旧消融，保留性能参照；最强baseline+同输入及整体范式证据仍缺。

当前进入候选6：**视频标签约束的局部证据状态模型**，独立proposal review已GO，数学与消融对应问题已修正；代码已实现，正在唯一code review，尚未正式训练。目标未完成，无硬阻塞，不能把提案GO称为novelty成立。

协议唯一来源：[研究规则](../RESEARCH_ITERATION_RULES.md)。主数据集为 HateMM、HateClipSeg；within 硬门仍有效，取消硬门的讨论尚未裁定。三个模块均须 novelty 的新要求覆盖[旧计划](../docs/20260903_three_module_program.md)中模块 1 可选的要求。

## 当前方法：三个模块

| 模块 | 实现 | 尚缺证据 |
|---|---|---|
| 1 VLM | 候选6复用同一冻结Qwen四路干预，两粒度联合8维状态条件高斯 | 完整协方差是否优于同输入对角协方差；不声称真实因果效应 |
| 2 骨干 | 内容时间卷积与相邻内容差异决定潜状态转移 | 内容条件转移是否优于静态转移，保留相同内容初始概率 |
| 3 融合/训练 | 精确视频事件概率+归一化观测NLL联合训练，最终输出同模型局部后验 | 相对同模型top-k训练是否有贡献；不是推理平滑 |

当前代码：[候选6提案与实现](../experiments/20260905_latent_evidence_sequence/README.md)、[独立proposal review](../experiments/20260905_latent_evidence_sequence/REVIEW_RULE4.md)。共享训练/评测调用在 `src/fixed_training_protocol.py`，v2缓存解析在 `src/interventional_observations.py`；评测器未改，候选5仅改调用升入共享目录的既有循环/解析，不重跑其结果。可靠起点：[候选1](../experiments/20260903_hier_evidence_mil/README.md)；候选4负结果：[归档结论](../archive/experiments/20260904_null_token_cma/README.md)。

## 已核验结果

候选 1 三 seed 均值，顺序为 **pooled AP / pooled ROC / within ROC**。每 seed 20 trial，trial 内 validation 选 checkpoint；开发搜索按 test 选 trial，不是未揭盲确认结果。

| 语料 | 按 test 选 trial（均值 ± std） | 仅按 validation 选 trial（均值） |
|---|---|---|
| HateMM | .657 ± .013 / .842 ± .005 / .646 ± .004 | .601 / .821 / .627 |
| HateClipSeg | .699 ± .006 / .681 ± .016 / .553 ± .007 | .686 / .665 / .538 |

来源：[候选 1 输出](../runs/20260903_hier_evidence_mil/)，`<hatemm|hateclipseg>/seed<234|2025|3407>/study_summary.json` 的 `best`、`validation_selected`；对应 trial 的 `metrics.json` 为评测器原始输出。Baseline 对照：[固定表](../docs/duplex/OFFICIAL_VAL_RESULTS.md)。

候选4修订1最终三seed：HateMM **.644±.025 / .840±.004 / .642±.007**；HateClipSeg **.696±.001 / .681±.012 / .542±.007**（std ddof=1）。validation选trial分别为 **.611/.825/.627**、**.686/.664/.538**。未达到相对候选1的预注册提升；HateClipSeg结构消融不支持空token，归档。来源：[核验汇总](../runs/20260904_null_token_cma/rev1/artifact_audit.json)指向各trial/消融的原始`metrics.json`，120 trial与18消融均齐全。

## 运行与监控

候选5 HateMM三seed按test选trial16/13/13：**AP .6313±.0072 / ROC .8458±.0053 / within .6599±.0103**（std ddof=1）。60trial（59 COMPLETE/1 within剪枝）全部回传审计，pooled领先固定门.0583/.0388，大于所需std幅度.033/.0194，within不破下限，数值确认通过。仅validation排序的零额外训练参考均值 `.6133/.8383/.6634`，不作门。来源：[三seed汇总及原始评测路径](../runs/20260905_interventional_evidence/hatemm/confirmation_summary.json)。

候选5 HateClipSeg三seed按test选trial：**AP .6541±.0042 / ROC .6383±.0057 / within .5577±.0196**（std ddof=1）。60trial均COMPLETE并回传核验；pooled领先固定门.0921/.1103，大于所需std幅度.0358/.0230，within不破下限，故HCS数值确认通过。仍属开发期test搜索，不代表整体目标完成。来源：[三seed汇总及各原始评测路径](../runs/20260905_interventional_evidence/hateclipseg/confirmation_summary.json)。

HCS仅validation排序的零额外训练参考均值 `.6516/.6358/.5566`，不作门。两语料120trial全量完成；最终声明仍缺模块有效性、最强baseline+同输入及整体范式证据。

HCS消融24/24完整50epoch，配置/validation checkpoint/val及test覆盖率/原始指标一致性审计通过。核心三臂均不满足规则14(g)；`four_logits`与`no_block`在HCS满足同向下降要求，但前者只支持当前训练下的表示差异，后者是已有块监督，不能替代三个新模块的证据。来源：[三seed消融汇总及原始评测路径](../runs/20260905_interventional_evidence/ablations/hateclipseg/three_seed_summary.json)。

HateMM seed234消融8/8完整回传审计。AP/ROC/within：原裁定替换 `.463/.745/.607`，普通注意力 `.599/.820/.671`，加法融合 `.596/.824/.671`；完整模型 `.624/.849/.672`。来源：[八臂审计及原始评测路径](../runs/20260905_interventional_evidence/ablations/hatemm/seed234/artifact_audit.json)。

开发期错误分析：HCS训练正视频219/251，HateMM298/744；HCS完整模型正秒与正视频内背景秒均分差仅.023/.026/.049（三seed），HateMM seed234为.149。HCS最佳checkpoint在epoch1/2/5，此后训练loss下降但validation AP下降。提示局部区分与监督目标值得优先检查，不证明类别比例是原因，也不据此调整checkpoint规则。来源：[已保存预测/GT/训练曲线诊断](../runs/20260905_interventional_evidence/error_analysis/saved_prediction_diagnostics.json)，只分析原输出，未重训或改评测器。

| 任务 | 当前状态 | 输出/日志 |
|---|---|---|
| HateMM v2输入 | 两片各534/534完成，进程均退出；合并1068视频/2136文件已完整回传、解析通过并同步lab1。旧抽取通知均已处理，不重启 | [完整输入审计](../runs/20260905_interventional_evidence/extract_hatemm_v2/input_audit.json)、[合并出处](../data/interventional_evidence/hatemm/PROVENANCE.md) |
| HateMM确认搜索，lab1 | seed2025/3407均退出，各20trial完整回传核验；两个结束状态均已处理，后到通知不重启 | [2025审计](../runs/20260905_interventional_evidence/hatemm/seed2025/artifact_audit.json)、[3407审计](../runs/20260905_interventional_evidence/hatemm/seed3407/artifact_audit.json) |
| HateClipSeg确认搜索 | seed2025/3407均已结束、各20trial完整回传审计；两个结束通知均已处理，不重启 | [2025审计](../runs/20260905_interventional_evidence/hateclipseg/seed2025/artifact_audit.json)、[3407审计](../runs/20260905_interventional_evidence/hateclipseg/seed3407/artifact_audit.json) |
| HateClipSeg确认seed消融，lab3 | 进程组3110708已退出，16/16回传审计通过；22:22结束通知已处理，不重启 | [2025审计](../runs/20260905_interventional_evidence/ablations/hateclipseg/seed2025/artifact_audit.json)、[3407审计](../runs/20260905_interventional_evidence/ablations/hateclipseg/seed3407/artifact_audit.json) |
| HateMM seed234消融，lab3 | 进程退出，8/8完整回传审计，23:04通知已处理，不重启 | [完整审计](../runs/20260905_interventional_evidence/ablations/hatemm/seed234/artifact_audit.json) |
| 候选6 code review | 独立agent `code_review_c6`核对概率递推/梯度、数据与搜索链；尚无正式训练 | [候选6目录](../experiments/20260905_latent_evidence_sequence/) |
| GPU资源 | lab1/lab3空闲且torch2.7.1+cu128一致；本机他人任务97%；lab-server GPU空闲但无项目/环境，当前未就绪 | 就绪后两语料独立并行，不为占满GPU重复实验 |
| 候选5 HateClipSeg v2输入 | 抽取进程已退出；393/393视频、786/786文件完整回传并解析通过 | [完整输入审计](../runs/20260905_interventional_evidence/extract_hateclipseg_v2/input_audit.json) |
| 候选5 HateClipSeg seed234，lab3 | 搜索进程已退出；20trial全部完整回传审计，首trial118.659秒 | [完整输出](../runs/20260905_interventional_evidence/hateclipseg/seed234/) |
| 候选5 HateClipSeg seed234消融 | 8/8完成并完整回传；单seed初步支持已被三seed结果修正，不再单独用来主张模块有效 | [消融审计与原始来源](../runs/20260905_interventional_evidence/ablations/hateclipseg/seed234/artifact_audit.json) |
| lab3输入准备 | 已完成；393视频头、K30/K4 ASR全覆盖、5模型分片可解析；webm规范别名已修复 | [本机核验输出](../runs/20260905_interventional_evidence/prepare_lab3/coverage.json) |
| 长期会话 monitor | 22:04提醒已处理；目标未完成、无硬阻塞，保留 | [状态与日志](../runs/thread_monitor/01a06df5-3e92-79b0-be30-820db943e551/) |

等待时不由模型持续轮询、不重复创建 monitor，用户无需手动设置 Goal。目标完成或确认无法推进的硬阻塞时关闭长期 monitor 并报告；正常等待和暂时断连不算硬阻塞。关闭命令：

```bash
python3 scripts/monitor_thread.py --thread 01a06df5-3e92-79b0-be30-820db943e551 --out-dir runs/thread_monitor/01a06df5-3e92-79b0-be30-820db943e551 --stop
```

## 下一步

1. 完成候选6唯一code review及必要bug修复；同步代码并检查工作树后，lab1/HateMM和lab3/HCS并行seed234完整Optuna，每项自动绑定独立monitor。首trial实测冻结20/5预算，不做smoke/短跑。
2. 按两语料完整test结果分流，再决定确认seed/消融；不重跑候选5搜索或为偶然同向加seed。最强baseline+同输入仍为最终声明的必要缺口。
3. 只搜索lr/dropout/max_seqlen；固定hidden128/kernel3/两状态/两粒度/损失等权等仍是设计参数，不称无超参数。等待由monitor事件唤醒；目标未完成、无硬阻塞，长期monitor保留。

## 历史与资料

- [整理前 STATUS 完整快照](../archive/research-wiki/STATUS_20260905_before_cleanup.md)：旧候选、旧规则、数据与归档路径索引，仅供追溯。
- [评测协议](../docs/duplex/FRAME_EVAL_PROTOCOL.md)、[缓存出处](../data/MLLM_scores/PROVENANCE.md)。原始视频在 `~/data/`；冻结的 Hate-follow-up 溯源路径保持不动。
