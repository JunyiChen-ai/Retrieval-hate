# Temporal Expert-Choice MIL

截至 2026-09-01。状态：**双数据集 test 失败，停止并归档。** Novelty `GO 6.9/10`，正式运行前唯一 technical review 为 `PASS`。权威结果：`runs/20260831_temporal_expert_choice/pilot_seed234/summary.json`。

## Result and decision

使用与权威 MultiHateLoc anchor 匹配的语料内训练配置，分别训练 capacity-matched token-choice control 与 temporal expert-choice core；validation 只在各 arm 内选择 checkpoint，随后立即由统一 evaluator 评测 HMM/HCS test 三指标。

| corpus | arm | pooled AP | pooled ROC | within ROC |
|---|---|---:|---:|---:|
| HMM | MultiHateLoc anchor | .492997 | .738259 | .628463 |
| HMM | token-choice | .332487 | .590462 | .550418 |
| HMM | expert-choice | .274928 | .532459 | .550637 |
| HCS | MultiHateLoc anchor | .551339 | .542726 | .520588 |
| HCS | token-choice | .575657 | .554109 | .514930 |
| HCS | expert-choice | .511387 | .475064 | .464062 |

Core 相对 anchor 的 within 差为 HMM `-.077826`、HCS `-.056525`；相对 matched control 为 HMM `+.000219`、HCS `-.050868`。机制门与 performance 门均失败，不扩展 MHC-EN/ZH，不做超参数续命。

Post-test failure analysis：HMM token-choice 几乎把全部 assignment 给 visual（test 每视频均值 visual/audio/text=`51.92/.65/.02`）；expert-choice 的固定 per-modality capacity 强制三路各 `17.53`，audio/text 的平均 signed contribution 为 `-.00649/-.00669`，没有形成正确 temporal ownership。HCS token-choice checkpoint 为 epoch 96，而 expert-choice 由 validation video AP 选到 epoch 2；固定均衡 assignment 把弱 modality 强制注入 final score，expert 的 pooled 与 within 同时下降。结论是 capacity balance 只保证负载，不保证 hate-localization competence；该机制不能从 bag label 学出可靠的 modality ownership。

## Initial failure target

不新增 premise。MultiHateLoc 四语料 test analysis 已证明 fused 相对 best single branch 的 within-video ROC 缺口为 HMM/EN/ZH/HCS `.106/.171/.211/.106`，而 video-global DMS 与最佳模态匹配率仅 `.216/.333/.375/.323`。刚完成的 witness-DGM 又证明只调制 gradient 不足：HMM core-vs-anchor final-score Spearman `.997627`，HCS虽改排序却牺牲 pooled。新机制必须把 time×modality responsibility 直接写入唯一 final score。

## Source

跨任务来源为 Zhou et al., *Mixture-of-Experts with Expert Choice Routing*（NeurIPS 2022）：不是每个 token 选择固定数量expert，而是每个expert按固定capacity选择最相关token，因此expert负载有保证、每个token可由可变数量expert处理。初步检索未发现 Expert Choice routing 用于 hateful-video detection/localization。

## Task adaptation delta

把三个固定 modality branches视为 experts，把一个视频内的1fps seconds视为 tokens。Hate evidence可只属于speech、visual symbol或text/OCR，且跨模态异步；因此每秒强制top-1 modality会丢失协同，video-global router又会整段选择同一模态。Temporal Expert Choice改为：

1. 每个modality expert从本视频选择自己的`ceil(T/K)`个second，capacity沿用该corpus已由validation确定的MultiHateLoc `K`，这里只是routing budget，不声称event duration；
2. 同一second可被0--3个experts选择，允许unimodal、multimodal与无证据秒；
3. 每个expert产生local evidence logit与selection affinity；hard expert-choice index按source top-k形成，选中affinity值和evidence logit保持可微；
4. 唯一frame logit是该second所有被选expert的加权 evidence sum加一个共享background bias，未被任何expert选中的second只取background；唯一video loss对该frame logit做原top-K MIL。

删除原 video-global DMS、原 fused head以及“同一positive label分别监督三个branch”的per-branch MIL。Negative bags通过同一个final MIL压低所有被选expert；positive bags只要求三个expert的联合选择中存在hate evidence。Test只输出单模型raw final score，不读取router之外branch、不做ensemble、calibration或按语料routing。

这不是普通MoE移植：source的token/expert是可交换FFN计算单元，这里把expert identity固定为不可交换的audio/visual/text evidence channel，并把“expert选择token、token接收可变expert数”转成弱监督 temporal ownership。它直接针对已证实的global visual monopoly，同时避免pseudo-owner/deletion teacher。

## Final-score path, falsification, control

Router mask和expert logits共同构成唯一frame score，不存在auxiliary bypass。一般whole-video或position shortcut作为test风险登记，不作实现前identifiability硬门。

唯一 matched control是capacity-matched **token-choice router**：按每个second内部的modality affinity排序分配；先让每秒取得相同的最小expert数，余下assignment给“下一候选affinity”最高的seconds，保证全视频active assignments总数与core完全相同但不约束每个expert的负载。预算小于`T`时表现为只给最高affinity的一部分seconds分配top-1；预算约等于`T`时表现为每秒top-1，ceil余量给少数seconds的top-2。Expert网络、参数量、MIL和输出公式相同。若token-choice等于或优于core，expert-choice机制失败。

可证伪预期：相对 matched MultiHateLoc anchor 与 token-choice control，HMM/HCS core within均提高，至少一边`>=+.020`。固定capacity导致的assignment count均衡是结构必然，不能作为学到ownership的证据；机制诊断必须看相对control的test within以及各modality对final score的实际signed contribution。最终晋级仍要求四主语料全部三项test指标严格SOTA。

Novelty通过后立即实现，一次technical review后用权威per-corpus MultiHateLoc validation-selected超参数独立训练HMM/HCS，选完checkpoint立即test全部三指标；不做新premise。

来源：[Zhou et al., *Mixture-of-Experts with Expert Choice Routing*, NeurIPS 2022](https://papers.nips.cc/paper_files/paper/2022/hash/2f00ecd787b432c1d36f3de9800728eb-Abstract-Conference.html)。
