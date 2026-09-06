# 当前研究状态

截至 **2026-09-07 03:00 NZST**。依据：候选 3（证据引导注意力，修订 2 模型）HateMM 三 seed 在不剪 within 的规则下重跑完成并回传，12 臂消融三 seed 完成并回传；HCS 沿用修订 2 三 seed。权威数字均引用本机 runs 原评测。

## 当前目标与结论

**候选 3 两语料三 seed 过规则 8 确认，12 臂消融中 11 臂两语料成立（唯一不成立：证据进 q/k 编码，HCS ≈ 0）；VLM / 骨干 / 融合三模块各有可主张的部件。候选 3 为当前方法。** 目标不变：两语料 pooled AP/ROC 过固定 baseline 表（三 seed 确认），三模块各有可主张贡献（规则 14(g)：三 seed 均值降 ≥ .01、两语料；不要求每 seed 都降），方法统一、方法级超参少（现在 α、w_fine、λ_block 三个）。within 只报告，不剪枝、不作门。

候选 3 第 7.2 节的"HateMM 不过 within 下限、归档"判定撤销：不剪枝后 HateMM 三 seed 均值 .668 / .850，是所有候选里 pooled 最高的（候选 1 .657 / .842，精简版 v2 .632 / .835），within .623 比候选 1 低 .023（只报告）。

## 当前方法：候选 3（证据引导注意力，修订 2）

[experiments/20260904_evidence_guided_attention](../experiments/20260904_evidence_guided_attention/README.md)（方法第 1、7 节，结果第 8.1 节）。候选 1 的 VLM 裁定 + HMM 后验 + 块级 MIL + 先验不变；骨干改为证据只进跨模态注意力的 query/key（四格嵌入 + 两列线性）、逐头 key 偏置、视频级证据上下文，内容表示保持纯内容；去掉 EMA。搜索 6 标量（lr、max_seqlen、λ_cma、α、w_fine、λ_block）。评测器未改。

## 最新权威结果

顺序 **AP / ROC / within**，test，按 test 选 trial 的开发期证据。

| 候选/语料 | 结果 | 来源 |
|---|---|---|
| **候选 3 HateMM 三 seed（不剪枝）** | **.6678±.0097 / .8504±.0049 / .6233±.0150**（best trial 17/8/13） | [搜索](../runs/20260904_evidence_guided_attention_rev2_noprune/hatemm/)，[两语料汇总](../runs/20260904_evidence_guided_attention_rev2_noprune/ablations/three_seed_summary_both_corpora.json) |
| **候选 3 HCS 三 seed** | **.6976±.0076 / .6843±.0095 / .5488±.0111**（best trial 13/12/16） | [搜索](../runs/20260904_evidence_guided_attention_rev2/hateclipseg/) |
| 候选 3 消融（12 臂 × 3 seed × 2 语料） | 去掉整个证据引导注意力：HMM −.044/−.019，HCS −.009/−.025；去 VLM 裁定：−.159/−.090，−.100/−.118；q/k 编码 HCS −.001 | 同上汇总 |
| 精简版 v2（候选 1 精简）三 seed | HMM .632/.835/.625；HCS .706/.690/.565；外部审稿 3/10 | [目录](../runs/20260906_hier_evidence_clean_v2/) |
| C1 三 seed（within 剪枝下搜索） | HMM .657±.013/.842±.005/.646±.004；HCS .699±.006/.681±.016/.553±.007 | [搜索](../runs/20260903_hier_evidence_mil/) |
| 规则 8 门 | HMM .573/.807；HCS .562/.528；within 参考 .632/.524 | `docs/duplex/OFFICIAL_VAL_RESULTS.md` |

## 运行任务与监控

无运行任务。lab1 / lab3 GPU 空闲；本机 GPU 被他人占用。远程 `runs/20260904_evidence_guided_attention_rev2_noprune/` 已全部 rsync 回本机。

## 下一步

1. 候选 3 送外部审稿（GPT6-Astra，最高 reasoning）找改进方向；审稿记录存实验目录。
2. 待用户裁定：w_fine 去留（三 seed best 取值 .15–.69 不稳定，删则两语料重跑）；30/4 粒度与 VLM 每视频 34 次调用的成本（自适应粒度方案待写）。
3. 搜索目标继续按 test（用户裁定，不再讨论）。

## 资料与历史

[研究规则](../RESEARCH_ITERATION_RULES.md)、[固定baseline表](../docs/duplex/OFFICIAL_VAL_RESULTS.md)、[评测协议](../docs/duplex/FRAME_EVAL_PROTOCOL.md)。负结果细节：[C9](../archive/experiments/20260906_interval_evidence_transport/README.md)、[C6](../archive/experiments/20260905_latent_evidence_sequence/README.md)、[C7用户取消](../archive/experiments/20260906_context_witness/README.md)、[C4](../archive/experiments/20260904_null_token_cma/README.md)。[旧状态索引](../archive/research-wiki/STATUS_20260905_before_cleanup.md)。冻结Hate-follow-up引用不动；缓存出处见各data子目录PROVENANCE.md。
