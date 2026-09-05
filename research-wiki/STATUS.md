# 当前研究状态

截至 **2026-09-05 22:29 NZST**。依据：HCS全部24项消融原始输出已回传审计、三seed汇总，以及HateMM搜索/消融实时进程。每轮结束替换对应条目，不追加流水账。

## 目标与当前结论

**目标尚未完成。** 等当前实验完成后继续开发：两语料达到研究规则的 SOTA；VLM、骨干、融合三个模块各有消融支持的 novelty；整体有统一机制支撑 novel paradigm；方法超参数尽可能少。

候选5[干预证据方法](../experiments/20260905_interventional_evidence/README.md)HateClipSeg已通过三seed数值确认门，但三个核心模块均未通过HCS三seed有效性要求：seed3407替换后AP/ROC均略升，融合平均AP增益也不足.01。不能继续把单seed正结果表述为三模块novelty支持。HateMM确认搜索及seed234诊断消融继续；最强baseline+同输入及整体范式证据仍缺。候选1是可靠性能起点，候选4已淘汰。

协议唯一来源：[研究规则](../RESEARCH_ITERATION_RULES.md)。主数据集为 HateMM、HateClipSeg；within 硬门仍有效，取消硬门的讨论尚未裁定。三个模块均须 novelty 的新要求覆盖[旧计划](../docs/20260903_three_module_program.md)中模块 1 可选的要求。

## 当前方法：三个模块

| 模块 | 实现 | 尚缺证据 |
|---|---|---|
| 1 VLM | 同一冻结Qwen四路干预，原始Yes/No logits、带符号差分与熵，K30/K4 | HCS平均AP增益.0305，但seed3407为−.0045，不满足每seed同向；交互项平均仅.0044 |
| 2 骨干 | I3D/VGGish/BERT内容与干预证据的正/负关联读出 | HCS平均AP增益.0252，但seed3407为−.0031；负关联不等于语义反证 |
| 3 融合 | 训练内Yager冲突转未知；HMM仅作train块目标，不作推理先验 | HCS平均AP增益.0085，seed3407为−.0024；对Dempster也无稳定优势，不能主张必要性 |

当前代码：[候选5训练](../experiments/20260905_interventional_evidence/train.py)、[搜索](../experiments/20260905_interventional_evidence/search.py)、[已通过code review](../experiments/20260905_interventional_evidence/REVIEW_RULE6.md)。可靠起点：[候选1](../experiments/20260903_hier_evidence_mil/README.md)；候选4负结果：[归档README第8.5节](../archive/experiments/20260904_null_token_cma/README.md)。共享输入/损失/评测调用在 `src/hier_evidence_common.py`。

## 已核验结果

候选 1 三 seed 均值，顺序为 **pooled AP / pooled ROC / within ROC**。每 seed 20 trial，trial 内 validation 选 checkpoint；开发搜索按 test 选 trial，不是未揭盲确认结果。

| 语料 | 按 test 选 trial（均值 ± std） | 仅按 validation 选 trial（均值） |
|---|---|---|
| HateMM | .657 ± .013 / .842 ± .005 / .646 ± .004 | .601 / .821 / .627 |
| HateClipSeg | .699 ± .006 / .681 ± .016 / .553 ± .007 | .686 / .665 / .538 |

来源：[候选 1 输出](../runs/20260903_hier_evidence_mil/)，`<hatemm|hateclipseg>/seed<234|2025|3407>/study_summary.json` 的 `best`、`validation_selected`；对应 trial 的 `metrics.json` 为评测器原始输出。Baseline 对照：[固定表](../docs/duplex/OFFICIAL_VAL_RESULTS.md)。

候选4修订1最终三seed：HateMM **.644±.025 / .840±.004 / .642±.007**；HateClipSeg **.696±.001 / .681±.012 / .542±.007**（std ddof=1）。validation选trial分别为 **.611/.825/.627**、**.686/.664/.538**。未达到相对候选1的预注册提升；HateClipSeg结构消融不支持空token，归档。来源：[核验汇总](../runs/20260904_null_token_cma/rev1/artifact_audit.json)指向各trial/消融的原始`metrics.json`，120 trial与18消融均齐全。

## 运行与监控

候选5 HateMM seed234：按test选trial16 **.624/.849/.672**，20trial完整核验（19 COMPLETE、1 within剪枝）。validation排序仅作零额外训练参考：trial17 `.615/.844/.671`，不用于选择方法。来源：[完整审计](../runs/20260905_interventional_evidence/hatemm/seed234/artifact_audit.json)、[trial16原始评测](../runs/20260905_interventional_evidence/hatemm/seed234/trial16/metrics.json)。

候选5 HateClipSeg三seed按test选trial：**AP .6541±.0042 / ROC .6383±.0057 / within .5577±.0196**（std ddof=1）。60trial均COMPLETE并回传核验；pooled领先固定门.0921/.1103，大于所需std幅度.0358/.0230，within不破下限，故HCS数值确认通过。仍属开发期test搜索，不代表整体目标完成。来源：[三seed汇总及各原始评测路径](../runs/20260905_interventional_evidence/hateclipseg/confirmation_summary.json)。

HCS消融24/24完整50epoch，配置/validation checkpoint/val及test覆盖率/原始指标一致性审计通过。核心三臂均不满足规则14(g)；`four_logits`与`no_block`在HCS满足同向下降要求，但前者只支持当前训练下的表示差异，后者是已有块监督，不能替代三个新模块的证据。来源：[三seed消融汇总及原始评测路径](../runs/20260905_interventional_evidence/ablations/hateclipseg/three_seed_summary.json)。

| 任务 | 当前状态 | 输出/日志 |
|---|---|---|
| HateMM v2输入 | 两片各534/534完成，进程均退出；合并1068视频/2136文件已完整回传、解析通过并同步lab1。旧抽取通知均已处理，不重启 | [完整输入审计](../runs/20260905_interventional_evidence/extract_hatemm_v2/input_audit.json)、[合并出处](../data/interventional_evidence/hatemm/PROVENANCE.md) |
| HateMM确认搜索，lab1 | seed2025 PID1668571 / seed3407 PID1668770；各完成12/20trial，两进程正常 | [2025 monitor](../runs/20260905_interventional_evidence/hatemm/seed2025/monitor/run.log)、[3407 monitor](../runs/20260905_interventional_evidence/hatemm/seed3407/monitor/run.log) |
| HateClipSeg确认搜索 | seed2025/3407均已结束、各20trial完整回传审计；两个结束通知均已处理，不重启 | [2025审计](../runs/20260905_interventional_evidence/hateclipseg/seed2025/artifact_audit.json)、[3407审计](../runs/20260905_interventional_evidence/hateclipseg/seed3407/artifact_audit.json) |
| HateClipSeg确认seed消融，lab3 | 进程组3110708已退出，16/16回传审计通过；22:22结束通知已处理，不重启 | [2025审计](../runs/20260905_interventional_evidence/ablations/hateclipseg/seed2025/artifact_audit.json)、[3407审计](../runs/20260905_interventional_evidence/ablations/hateclipseg/seed3407/artifact_audit.json) |
| HateMM seed234消融，lab3 | 22:27启动PID/PGID3136119，锁定trial16配置跑8臂，每批3任务，完整50epoch；首批正常，GPU100%；monitor PID1441905首次检查RUNNING | [训练日志](../runs/20260905_interventional_evidence/ablations/hatemm/seed234/run.log)、[monitor](../runs/20260905_interventional_evidence/ablations/hatemm/seed234/monitor/run.log) |
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

1. HateMM seed234消融完成后核验回传，检查模块失效是否跨语料存在；确认搜索完成后核验三seed性能。HCS核心机制已不满足novelty要求，不为争取偶然同向重复旧消融或追加seed；结合两语料错误分析决定同方法修订或换候选，再安排必要的后续实验。
2. 唯一code review三项修复已确认：原始logits/版本隔离、固定评测覆盖、补seed继承234预算。v1进程与monitor已停，缓存/日志保留并回传（详见候选5README第5节），不得混入训练。无需重审。
3. 修订优先处理融合缺少稳定贡献、正负关联与交互项必要性不足；现有消融每seed采用不同的搜索最优配置，不能把seed3407反转直接归因为随机数或学习率。最强baseline+同输入仍为最终报告缺口，不能以已有块监督或可逆表示优势替代三个模块novelty。

## 历史与资料

- [整理前 STATUS 完整快照](../archive/research-wiki/STATUS_20260905_before_cleanup.md)：旧候选、旧规则、数据与归档路径索引，仅供追溯。
- [评测协议](../docs/duplex/FRAME_EVAL_PROTOCOL.md)、[缓存出处](../data/MLLM_scores/PROVENANCE.md)。原始视频在 `~/data/`；冻结的 Hate-follow-up 溯源路径保持不动。
