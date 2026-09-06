# 当前研究状态

截至 **2026-09-06 12:50 NZST**。依据：候选 1 精简版搜索空间 v1 两语料 seed 234 各 20 trial 完成并回传；按预注册第 3 条与用户"只限制方法超参"的澄清，v2（加回 CMAL 三个训练权重）两语料 seed 234 已启动。权威数字均引用本机 runs 原评测。

## 当前目标与结论

**目标未完成。** 目标不变：两语料 pooled AP/ROC 过固定 baseline 表（三 seed 确认），VLM/骨干/融合三模块各有可主张的贡献（规则 14(g)：三 seed 均值降 ≥ .01、两语料；09-06 起不要求每 seed 都降），方法统一、超参少。within 从 09-06 起只报告，不剪枝、不作门（`RESEARCH_ITERATION_RULES.md` 第 7/8/9 条）。

09-03 到 09-06 的 C2–C9 九个候选没有一个在任一语料超过候选 1（C1 三 seed HateMM .657/.842、HCS .699/.681），已全部归档；C9 最终数字见其归档 README。用户裁定停止换新架构，回到候选 1 做减法。

## 当前方法：候选 1 精简版

[experiments/20260906_hier_evidence_clean](../experiments/20260906_hier_evidence_clean/README.md)，规则 6 [code review PASS](../experiments/20260906_hier_evidence_clean/REVIEW_RULE6.md)。与候选 1 修订 1 相比：删 K30 调温 w_fine（固定 1）、删 CMAL 权重搜索（MACIL-SD 发表值）、删全部死消融代码；搜索超参 9 → 5（lr、dropout、max_seqlen、α、λ_block），方法级标量只有 α 与 λ_block。保留的每个部件都有候选 1 三 seed 消融 ≥ .01 的依据（README 第 1 节表）。评测器未改。

候选 1 骨干消融结论（`runs/20260903_hier_evidence_mil/ablations/`）：两语料都稳定有用 = VLM 裁定、共享跨模态注意力、CMAL、块级 MIL、HMM 时间耦合；只 HateMM 有用 = 音频、视觉、EMA、P(s) 列、块 OR 层次、裁定拼输入；只 HCS 有用 = HMM 后验替代平均等级先验、先验项。分数方差 92%/79% 在视频之间，骨干主要在估视频级仇恨密度，视频内排序来自 HMM 后验。

## 最新权威结果

顺序 **AP / ROC / within**，test，按 test 选 trial 的开发期证据。

| 候选/语料 | 结果 | 来源 |
|---|---|---|
| 精简版 v1 HateMM seed234 | .647900/.833380/.628441，trial 19 | 过主门，比 C1 seed234 低 .013/.008；[原评测](../runs/20260906_hier_evidence_clean/hatemm/seed234/trial19/metrics.json) |
| 精简版 v1 HCS seed234 | .696499/.688134/.554490，trial 10 | 过主门，与 C1 持平（ROC +.009）；[原评测](../runs/20260906_hier_evidence_clean/hateclipseg/seed234/trial10/metrics.json) |
| C1 三 seed（within 剪枝下搜索） | HMM .657±.013/.842±.005/.646±.004；HCS .699±.006/.681±.016/.553±.007 | [搜索](../runs/20260903_hier_evidence_mil/) |
| C9 HateMM seed234 | .614455/.815451/.649858，trial 11 | [原评测](../runs/20260906_interval_evidence_transport/hatemm/seed234/trial11/metrics.json) |
| C9 HCS seed234 | .605771/.589308/.547480，trial 18 | [原评测](../runs/20260906_interval_evidence_transport/hateclipseg/seed234/trial18/metrics.json) |
| C5 三 seed | HMM .631/.846/.660；HCS .654/.638/.558 | [汇总](../runs/20260905_interventional_evidence/hatemm/confirmation_summary.json) |
| 规则 8 门 | HMM .573/.807；HCS .562/.528；within 参考 .632/.524 | `docs/duplex/OFFICIAL_VAL_RESULTS.md` |

## 运行任务与监控

| 任务 | 状态 | 位置 |
|---|---|---|
| 精简版 v2 HateMM seed234，lab1 | 完成并回传：**.633476/.839300/.618039**（trial 1），过主门；低于 v1 .648/.833 与 C1 .661/.841，差在搜索噪声内，cof 假设未证实（README 第 8 节） | [原评测](../runs/20260906_hier_evidence_clean_v2/hatemm/seed234/trial1/metrics.json) |
| 精简版 v2 HateMM seed2025/3407，lab1 | 12:50 并行启动，各 20 trial | `runs/20260906_hier_evidence_clean_v2/hatemm/seed<seed>/`（远端） |
| 精简版 v2 HCS seed234，lab3 | 完成并回传：**.703024/.683522/.562335**（trial 13），过主门，与 C1 持平 | [原评测](../runs/20260906_hier_evidence_clean_v2/hateclipseg/seed234/trial13/metrics.json) |
| 精简版 v2 HCS seed2025/3407，lab3 | 11:42 并行启动，PID 3732961/3732963，各 20 trial | `runs/20260906_hier_evidence_clean_v2/hateclipseg/seed<seed>/`（远端） |
| 精简版 v1 两语料 | 完成，已回传本机，数字见上表；v1 与 v2 不混算 | `runs/20260906_hier_evidence_clean/` |
| 监控 | 本会话 harness 后台等待 `SEARCH_DONE` 或进程消失，不再有 monitor 脚本/线程 | — |
| C9 | 两语料 seed234 与 HCS 全部诊断已回传；seed2025/3407 被用户中止的部分输出也已回传，不作结果 | [归档](../archive/experiments/20260906_interval_evidence_transport/README.md) |
| 本机 GPU | 他人任务占用（18G/97%），按选机规则用 lab1/lab3 | — |

多机同步：启动前三台均 a2016d8，远程无脏文件（lab1 有既有未跟踪 idea-stage/）。

## 下一步

1. v2 两语料 seed 234 均过主门（HateMM .634/.839、HCS .703/.684）；四个确认 seed 搜索在跑（lab1 HateMM、lab3 HCS 各两 seed）。
2. 确认 seed 结束后回传，算三 seed 均值/标准差对照规则 8 与 C1；三 seed 后按 README 第 4 节九个消融臂跑 `scripts/run_locked_ablations.sh 20260906_hier_evidence_clean_v2 ...`，按 14(g)（三 seed 均值降 ≥ .01、两语料）判定可主张部件。
3. 方法级标量保持 α、λ_block 两个；不加回 w_fine，不换架构。

## 资料与历史

[研究规则](../RESEARCH_ITERATION_RULES.md)、[固定baseline表](../docs/duplex/OFFICIAL_VAL_RESULTS.md)、[评测协议](../docs/duplex/FRAME_EVAL_PROTOCOL.md)。负结果细节：[C6](../archive/experiments/20260905_latent_evidence_sequence/README.md)、[C7用户取消](../archive/experiments/20260906_context_witness/README.md)、[C4](../archive/experiments/20260904_null_token_cma/README.md)。[旧状态索引](../archive/research-wiki/STATUS_20260905_before_cleanup.md)。冻结Hate-follow-up引用不动；缓存出处见各data子目录PROVENANCE.md。
