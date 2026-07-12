# Target-driven Gate 0 · Iteration 3：零片段金标注下的 MLLM→最终分类新路线

**冻结日期：** 2026-07-10（Pacific/Auckland）  
**检索范围：** 2024-01-01 至 2026-07-10；重点补扫最近六个月 `2026-01-10..2026-07-10`  
**本轮操作：** 文献、查新、假设与 fast-fail 设计；未改代码、未提交作业、未调用新的 MLLM teacher  
**目标：** MLLM 必须作为可移除、机制级、train-only privileged teacher，最终在至少 MHC-EN/MHC-ZH 两个数据集、paired seeds 0/1/2 上，相对移动的同协议最强 non-MLLM RGCL，使 **final accuracy 与 macro-F1 各提高至少 `+0.030` absolute**。

> **监督红线：本项目没有片段级金标注。** 唯一 gold supervision 是父视频二分类标签。uniform frames、自动 subclips、ASR/OCR 只是整视频输入；MLLM 的 target、stance、mechanism、counterfactual order、environment、failure mode 或时间描述都是 weak/privileged pseudo-signal，绝不能称为 segment annotation、span gold、oracle localization 或 dense gold supervision。下列三条路线均不做 segment weighting、segment loss 或 segment endpoint。

## 0. 最终 Gate 0 结论

文献候选与代码侧架构合并去重后，本轮只保留 **恰好三个**：

| 顺序 | 最终候选 | MLLM 的不可替代 train-only 角色 | 学生真正改变什么 | 独立查新判断 |
|---|---|---|---|---|
| **A** | **CTE-RGCL：Counterfactual Tangent Evidence Geometry** | 比较同一 train video 的 full / visual-neutralized / language-neutralized 整视频条件，输出 `preserve/weaken/reverse/unknown` 弱序 | 用局部切向顺序约束直接塑造 full-bank true-class retrieval margin；query 与 memory 共同移动 | **5.5/10，PROCEED，唯一首跑** |
| **B** | **SQ-RGCL：Semantic Quotient RGCL** | 为 whole-video presentation/context nuisance 给出软环境分布 | 学习 class-conditional nuisance quotient，最终 kNN 只读 quotient representation | **4.5/10，REVISE，第二储备** |
| **C** | **ECM-RGCL：Executable Constraint Modes** | 从 strict-OOF train residual 诊断受限 whole-video failure-mode posterior | 用 mode-level constrained gradient projection/minimax 改善 worst-mode retrieval margin | **4.0/10，REVISE，第三储备** |

合并边界必须冻结：

- 原 **RSDG** 的 teacher-residual-geometry 意图并入 CTE；diffusion、heat-kernel、GW/OT、residual-kernel PSD 处理全部删除。否则既与强 prior 重叠，又有 residual kernel 不保证 PSD 的可实现性问题。
- 原 **PSD** 并入 SQ；不再声称 causal deconfounding，只声称 class-conditional semantic quotient / nuisance invariance。
- 原 **ESEC** 的 fuzzy-rule、student predicate heads、differentiable energy 与 proximal correction **ABANDON**。仅将 executable-constraint 意图改写为 ECM 的 OOF failure-mode constrained optimization。
- 三条不得并行实现，不得堆叠。唯一批准顺序是 `CTE-0 zero-teacher → CTE-1 teacher pilot → CTE-2 seed-0`；CTE fast-fail 后才可转 SQ，SQ 失败后才可转 ECM。

三条共同排除 simple concat、score fusion、segment weighting、teacher-selected/replaced memory keys、test-time MLLM/rerank、model/data/epoch/ensemble scale、SSR/EDCM 原 universe 调参，以及 BPDMoE-Hate 式 viewpoint gate / MoE / low-rank router。

## 1. 不可放宽的最终门槛

来自 `TARGET_STATE.json` 的历史最强 non-MLLM 下界和最低终局线如下。最终仍取 `max(历史最强点, paired same-seed REMOVE mean)+0.030`，因此目标只会移动上升：

| 数据集 | 历史最强 non-MLLM acc / mF1 | 最低 final target acc / mF1 |
|---|---:|---:|
| MHC-EN | 0.7888 / 0.7262 | **0.8188 / 0.7562** |
| MHC-ZH | 0.8255 / 0.7875 | **0.8555 / 0.8175** |

完整成功还要求两个数据集、seeds 0/1/2、acc 与 mF1 各 `>=+0.030`、3/3 paired delta 同号、hierarchical paired bootstrap 与四项 Holm 校正通过；FULL 显著胜 REMOVE 与 SHUFFLE，并在 calibrated corruption 下保留预注册比例的增益。单 seed、单库、单指标、`+2pt`、native-head、定位、rationale、predicate 或审计增益均不算目标完成。

## 2. SSR / EDCM 新负证据：固定可纠正 universe 不足

### 2.1 SSR 稀疏关系边的乐观全集

`artifacts/ssr/v1/b1/preflight_oracle_upper_bound.json` 假定每条预选 candidate arc 都是正确、可靠的 MLLM relation；实际 accepted set 只能更小。严格五折 train OOF、仅视频标签的结果仍为：

| 数据集 / family | 触及错例 | 乐观 Δacc | 乐观 ΔmF1 |
|---|---:|---:|---:|
| MHC / MI | 2 | +0.0036 | +0.0048 |
| MHC / SC | 7 | +0.0128 | +0.0176 |
| MHC-ZH / MI | 3 | +0.0052 | +0.0065 |
| MHC-ZH / SC | 15 | +0.0259 | +0.0307 |

四格均未过 `+0.050/+0.050` pre-MLLM screen，common family 为空。问题是单邻居 MI/SC event universe 过稀，不是 prompt、teacher、可靠性阈值或 loss 不够好；关系抽取被正确跳过，`segment_gold_used=false`。

### 2.2 EDCM 固定 top-64、至多两次 swap

`artifacts/edcm/v1/a0/{MHC,MHC_zh}/metrics.json` 与 `A0_DECISION.json` 在 **零 MLLM/OCR/teacher 调用**下得到：

| 数据集 | support | reachable errors | 乐观 Δacc | 乐观 ΔmF1 | 决策 |
|---|---:|---:|---:|---:|---|
| MHC | 202/549 = 0.3679 | 15（需 28） | +0.0273 | +0.0394 | STOP |
| MHC-ZH | 364/579 = 0.6287 | 22（需 29） | +0.0380 | +0.0444 | STOP |

两库 support、reachable count、`+0.05` acc 与 `+0.05` mF1 全失败；`A1_unlocked=false`，teacher 调用为零。

### 2.3 正确解释

SSR 只否定单邻居 candidate-arc universe；EDCM 只否定冻结旧表示、原 top-64、每 query 至多两次 swap 的 action space。它们 **不是任何可学习表示方法的理论上界**。共同移动 query 与 memory、允许旧 top-64 外样本成为新邻居的 CTE/SQ/ECM 在结构上超出旧 universe。

但结构上超出不等于能涨三点。每条新路线在调用 MLLM 前仍须过 strict-OOF learned **cost/capacity screen**，证明 action family 的作用 dense 且能真实纠正足够多 OOF 样本。A0 失败只触发成本策略 STOP，不得写成科学理论不可能；A0 通过也不是 MLLM 贡献或终局成功。

## 3. 2024–2026 竞争地图与最近六个月补扫

只采用论文作者页、ACL Anthology、AAAI/ICLR 官方页、OpenReview 或 arXiv 原页；不同数据集/协议的外部增益不能外推成本项目结果。

| 工作 | 状态 | 已占据的机制空间 | 本轮约束 |
|---|---|---|---|
| [RGCL](https://aclanthology.org/2024.acl-long.291/) | ACL 2024 | retrieval-guided contrastive geometry | final train-memory kNN 是真实 endpoint |
| [RA-HMD](https://aclanthology.org/2025.emnlp-main.1215/) | EMNLP 2025 Main Oral | LMM SFT + RGCL | 本地 P9/P9b 显示直接移植只做 head↔memory redistribution |
| [HVGuard](https://aclanthology.org/2025.emnlp-main.456/) | EMNLP 2025 | MLLM CoT + MoE hateful-video classifier | 普通 reasoning feature/MoE 已被直接竞争者占据 |
| [RAMF](https://openreview.net/forum?id=U9KnNiuMu1) | TMLR 2026-06 accepted | objective/hate-assumed/non-hate-assumed reasoning + fusion | 对立 reasoning 文本不是新路线 |
| [IARE](https://arxiv.org/abs/2606.11953) | arXiv 2026-06，文中为 SIGIR 2026 | harmful-element rationale + SFT/DPO | 外部细粒度标注不属于本地可用监督 |
| [DR-HM](https://aclanthology.org/2026.findings-acl.2130/) | Findings ACL 2026-07 | cognition-aware synthesis + SFT + A-GRPO | rationale distillation/RL 已拥挤 |
| [ExPO-HM](https://iclr.cc/virtual/2026/poster/10008633) | ICLR 2026 Poster | explain-then-detect policy optimization | 生成式 reasoning 主线已占据 |
| [BPDMoE-Hate](https://aclanthology.org/2026.acl-long.480/) | ACL 2026 Main | adversarial viewpoints + gating + dual-space MoE | 同类 router/MoE 硬排除 |
| [TextTeacher](https://openreview.net/forum?id=Xwb0aEUwKh) | TMLR 2026 | train-time semantic anchors 塑造表示、推理删除 language teacher | inference-free semantic supervision 不新 |
| [EmbedDistill](https://openreview.net/forum?id=-aEuKX6zQKmr) | OpenReview 原页 | retrieval embedding geometry distillation | retrieval geometry KD 不新 |
| [Geometric KD / Neural Heat Kernel](https://openreview.net/forum?id=7WGNT3MHyBm) | NeurIPS 2022，页面更新至 2026 | heat-kernel teacher/student alignment | 删除 RSDG heat-kernel claim |
| [UCMKD](https://openreview.net/forum?id=z9aetU7Wfl) | ICML 2026 regular page | distribution-level cross-modal KD | distribution alignment 不新 |
| [Geometry-aware alignment](https://openreview.net/forum?id=Yzr27JSBiV) | ICLR 2026 submission | Gram/Procrustes KD | Gram geometry matching 不新 |
| [OptiMAG](https://arxiv.org/abs/2601.22856) | arXiv 2026-01 | Fused GW structure-semantic alignment | OT/GW geometry 不新 |
| [CGO](https://openreview.net/forum?id=Z51RWOPKQQ) | ACL ARR 2026 | harmful-video modality intervention / gradient control | CTE 只能主张 retrieval-margin tangent 的窄 delta |
| [Mitigating Spurious Correlations](https://proceedings.mlr.press/v202/yang23j/yang23j.pdf) | ICML 2023 | 语言属性发现 + multimodal decorrelation | SQ 不能泛称首次 language-guided deconfounding |
| [CDAL](https://openreview.net/forum?id=wtsM3MPn2P) | ICML 2026 regular page | dual subspace + orthogonality + HSIC | SQ 算法骨架 prior 很强 |
| [CARE](https://openreview.net/forum?id=KzH1cLVU1R) | ICLR 2026 workshop | invariant/environment-specific concept directions | invariant factorization 不新 |
| [Dependent-factor disentanglement](https://openreview.net/forum?id=PgwkNC63CS) | TMLR 2026 accepted | 相关因子下 minimal/sufficient disentanglement | SQ 不得假定 target/stance 独立 |
| [DARTVAE](https://arxiv.org/abs/2509.20501) | arXiv 2025 | LLM-generated rules 以 consistency/violation loss塑造 latent clustering | 原 ESEC rule-energy novelty 不成立 |
| [Formal Concept Lattices](https://openreview.net/forum?id=AE8xCfWWL9) | ICML 2026 regular page | concept lattice scaffold | 原 ESEC concept hierarchy 不新 |
| [Energy-Based Constraint Networks](https://openreview.net/forum?id=gl6l8nTXBB) | TMLR under review 2026-05 | multimodal structural-coherence energy | 原 ESEC energy correction 不新 |
| [GRACE](https://openreview.net/forum?id=m276fke38H) | ICLR 2026 Poster | directional coverage / gradient variance 预测 distillation value | gradient rank 单独不足以解锁 full run |

近六个月还补查了 [Privileged Information Distillation](https://arxiv.org/abs/2602.04942) 与负结果 [Rethinking On-Policy Self-Distillation](https://arxiv.org/abs/2607.05184)。它们共同警告：teacher 语义合理、可蒸馏或 gradient-aligned 都不能代替 REMOVE/SHUFFLE/NOISE 和 final-kNN 证据。

## 4. 假设 A：CTE-RGCL

### 4.1 精确定义与可证伪 claim

同一 train video 形成三个确定、whole-video 条件：

- `full`；
- `visual-neutralized`；
- `language-neutralized`。

neutralization 算子在任何 teacher 反馈前预注册，不使用 segment、timestamp 或 test record。冻结 MLLM 只输出受限弱序 `preserve/weaken/reverse/unknown`，不输出 final hate verdict。该弱序约束学生的 **full-bank true-class retrieval margin**：表示沿 deterministic counterfactual 的切向变化，应与 teacher 弱序方向一致。训练共同移动 encoder/fusion 与整库 memory；测试仍是普通 full-video embedding 和项目原 kNN。

**Claim A：** 若 teacher whole-video modality-counterfactual order 在控制视频标签、原 margin、modality energy 与 difficulty 后仍含增量信息，则 ordinal tangent loss 能改变旧 top-64 外的邻域并提升 actual final kNN acc/mF1。

CTE 不做自由文本 embedding、semantic kernel、OT/GW、heat-kernel matching、teacher-selected key、test-time counterfactual 或 score fusion，也不新增 MoE/router 参数。

### 4.2 为什么超越旧 reachability

SSR/EDCM 改旧邻域中的少数 edge/swap；CTE 的 tangent loss 更新学生表示，所有 train keys 和 full-video queries 共同移动，旧 top-64 外样本可以成为新邻居。因此 action space 结构上更宽。该事实只解除旧算术上界，不保证性能。

### 4.3 Closest work、可守 novelty 与风险

- **Closest：** [TextTeacher](https://openreview.net/forum?id=Xwb0aEUwKh)、[EmbedDistill](https://openreview.net/forum?id=-aEuKX6zQKmr)、[CGO](https://openreview.net/forum?id=Z51RWOPKQQ)、[RAMF](https://openreview.net/forum?id=U9KnNiuMu1)。
- **仅可守的 delta：** `train-only MLLM ordinal whole-video modality counterfactual evidence → exact full-bank RGCL retrieval-margin tangent；inference 仍为 unchanged full-video kNN`。
- **不能声称：** 首次 counterfactual supervision、semantic KD、retrieval geometry distillation、modality intervention 或 gradient control。
- **主要风险：** neutralization 只是 OOD artifact；teacher 根据输入缺失程度而非 hate mechanism 排序；多数样本答案 collapse；label-only ordinal loss已解释全部效果；只有 native head 移动；label-only CTE 若强，必须升级为新的 non-MLLM moving baseline。
- **novelty：** 独立 reviewer `5.5/10`，有条件 PROCEED。

### 4.4 Fast-fail

**CTE-0 · strict-OOF zero-teacher capacity screen：**

1. 每个 outer fold 的 neutralized representation 与 label-only ordinal proxy 只由 inner-train 构造；outer-query 不用 teacher、counterfactual loss或 outer label，label 仅用于 endpoint。
2. `>=80%` inner/OOF videos 必须产生非退化 order support；梯度/neighbor churn 必须显著高于 capacity-matched random/order control，不能只检查浮点非零。
3. learned OOF kNN acc 与 mF1 在 MHC、MHC-ZH 均相对 frozen geometry `>=+0.050`，且触及面显著宽于 EDCM 15/22 个 reachable errors。
4. 该门是 learned capacity/cost screen，不是理论上界或 MLLM 成功证据。label-only 版本若成为更强模型，立即提高 moving baseline。

失败则零 teacher 成本 STOP。通过后才允许：

**CTE-1 · teacher-value pilot：** 每库最多 128 个 strict train videos；双 prompt、双输入顺序；`unknown` 和 deterministic fallback 明示。FULL 必须相对 label-only order、simple modality-energy heuristic、strength-matched random orders 与 within-fold order shuffle，在 agreement、held-out conditional information、实质 gradient/correction effect 上有增量；GRACE-style directional coverage 只能作一项证据，不能单独解锁。

**CTE-2 · seed-0：** 两库 dev actual kNN acc/mF1 各比 REMOVE、order shuffle、label-only order、heuristic order、random order 高 `>=+0.010`，且 noise 单调；否则转 SQ。

### 4.5 无片段金标注审计

三个条件均覆盖整视频。neutralization 不寻找关键片段、不读取定位 gold、不输出时间 span；uniform frames 只是整视频输入采样。teacher order 是 weak pseudo-signal。

## 5. 假设 B：SQ-RGCL

### 5.1 精确定义与边界

MLLM 为 train-only whole video 输出 **presentation/context nuisance** 的软环境分布；学生学习 class-conditional quotient：同一 gold class、不同 nuisance environment 拉近，不同 gold class 即使处于同一 environment 仍分离。最终 memory/query 只用 quotient representation，test 无 MLLM/environment。

只允许 presentation style、reportage format、surface topic、speaker/layout/context 等候选 nuisance。speaker endorsement、harm act、target evidence、cross-modal binding 不得自动塞入 nuisance subspace，否则会擦除真正 label signal。

**Claim B：** teacher soft nuisance environments 若不等价于 base embedding cluster 或 cheap caption attribute，则 class-conditional quotient 可全局减少 topic/style confounding，改善 final kNN acc/mF1。

### 5.2 为什么超越旧 reachability

SQ 改变整个 class-conditional 坐标/等价类，所有 covered train videos 与 queries 重新排序，不受原 top-64/two-swap 限制。

### 5.3 Closest、novelty 与风险

- **Closest：** [Yang et al., ICML 2023](https://proceedings.mlr.press/v202/yang23j/yang23j.pdf)、[CDAL](https://openreview.net/forum?id=wtsM3MPn2P)、[CARE](https://openreview.net/forum?id=KzH1cLVU1R)、[dependent-factor disentanglement](https://openreview.net/forum?id=PgwkNC63CS)。
- **仅可守的 delta：** MLLM whole-video soft nuisance environment 对 exact RGCL memory space 做 class-conditional quotient，teacher 在 inference 缺席。
- **主要风险：** environment 是已有 cluster 的自然语言重述；同类拉近损害真实 hateful subclasses；teacher posterior 与 label 同构或 collapse；收益来自 quotient/HSIC 而非 MLLM；算法 novelty 很窄。
- **novelty：** `4.5/10`，REVISE；不得称 causal deconfounding。

### 5.4 Fast-fail

**SQ-0 zero-teacher：** outer-train 内只用现成 full-video visual/text/audio embedding 做 label-blind soft environments，再训练容量一致的 quotient。要求两库 `>=80%` 非退化 support、每类跨多个 environment、held-out error-risk heterogeneity，以及 learned OOF acc/mF1 均 `>=+0.050`。它是 cost screen，不是上界；若 base environment 本身变强，更新 moving baseline。

**SQ-1 pilot：** 仅在 CTE 停止后执行；teacher posterior 必须胜 base cluster、cheap caption/language attributes 与 shuffle。FULL 后续还必须胜 REMOVE、base cluster、field shuffle、posterior noise、Yang-style language-attribute regularization 与 P4-style auxiliary prediction。

**SQ-2 seed-0：** 两库 dev actual kNN acc/mF1 对全部关键 control 各 `>=+0.010`；否则转 ECM。

### 5.5 无片段金标注审计

environment 只属于父视频，不能含 timestamp/span/segment ID；不做 subclip membership 或 weighting。

## 6. 假设 C：ECM-RGCL

### 6.1 原 ESEC 放弃后的精确定义

MLLM 读取 strict-OOF inner-train residual records，为 whole video 输出受限 failure-mode posterior，例如：

- cross-modal binding failure；
- stance inversion；
- context/reportage confusion；
- target attribution failure；
- modality dominance；
- unknown。

MLLM 不生成 fuzzy rules、不预测 final verdict、不创建 predicate head。训练把 failure modes 作为 mode-level constraints，通过 capacity-matched constrained gradient projection 或 minimax curriculum 改善 worst-mode retrieval margin；测试没有 teacher、mode、rule、predicate 或额外 head，仍是 full-video embedding + 原 kNN。

**与 SQ 的硬边界：** SQ 是 label-blind presentation/context nuisance environment，目标是 quotient invariance；ECM 是 strict-OOF residual 定义的 label-conditional failure mechanism，目标是 gradient-conflict control。若 ECM modes 退化成 topic/style environment，立即并入 SQ，不得保留第三候选名额。

### 6.2 为什么超越旧 reachability

mode constraints 改变 encoder 的整体优化轨迹与所有 memory embeddings，可创造旧 top-64 外的新邻域，不是 edge/swap 选择器。

### 6.3 Closest、novelty 与风险

- **Closest：** failure-slice/environment discovery、GroupDRO/minimax、[CGO](https://openreview.net/forum?id=Z51RWOPKQQ)、[GRACE](https://openreview.net/forum?id=m276fke38H)、[CDAL](https://openreview.net/forum?id=wtsM3MPn2P)。
- **仅可守的 delta：** strict-OOF MLLM residual diagnosis 定义 train-only semantic failure modes，并进入 capacity-matched final-RGCL retrieval constrained optimization；inference 无 mode。
- **主要风险：** mode 只是 gold/error 的自然语言复述；等价于 hard-example mining；gradient projection 收益与语义无关；与 SQ 重叠；teacher 看到 residual 后形成 label shortcut；novelty 主要是任务/endpoint 组合。
- **novelty：** `4.0/10`，REVISE，第三储备。

### 6.4 Fast-fail

**ECM-0 zero-teacher：** 用 random modes、loss-only difficulty bins、embedding clusters、label×margin bins 与 capacity-matched vanilla GroupDRO/minimax，测试同 action family 是否在两库有 `>=80%` mode support，并使 learned OOF acc/mF1 均 `>=+0.050`。它不是上界。

**ECM-1 pilot：** 仅在 CTE/SQ 均停止后执行。mode ontology 在 pilot 前冻结；只用 inner-train strict-OOF residual，validation/test errors 不可见；FULL 必须胜 random mode、difficulty bins、embedding cluster、label×margin bins、mode shuffle 与 vanilla GroupDRO/minimax，否则 MLLM 不可替代性失败。

**ECM-2 seed-0：** 两库 dev actual kNN acc/mF1 对所有关键 control 各 `>=+0.010`；否则重新开启下一轮 Gate 0，不扩大 teacher、mode 数或 epoch。

### 6.5 无片段金标注审计

failure modes 只能是 whole-video pseudo-signals，不得含 span、segment ID、关键片段或定位 endpoint。

## 7. 统一终局协议

任一候选只有在自己的 zero-teacher、teacher-value pilot 与 seed-0 全通过后，才进入：

1. MHC-EN、MHC-ZH × paired seeds 0/1/2，同 backbone/data/epoch/checkpoint/vote；无 ensemble。
2. 每 metric 对 `max(历史最强点, paired REMOVE mean)` 至少 `+0.030`。
3. FULL、REMOVE、MLLM-information SHUFFLE、calibrated NOISE 与候选专属 cheap/capacity-matched controls。
4. 3/3 paired deltas positive；每库 acc/mF1 hierarchical paired-bootstrap lower bound `>0`；四项 Holm FWER 0.05。
5. actual kNN acc/mF1 必须提升，native head/teacher agreement/rationale/localization 不可替代。
6. 唯一 gold 是视频二分类标签；teacher cache 只有 train IDs；validation/test 无 teacher records；coverage/confidence/missing/fallback/noise 全量报告。
7. 外部 reviewer 确认 novelty、evidence 与机制归因后才可宣称完成。

## 8. Claim-level query pack 与查新结论

每个 claim 至少使用三种检索表达，并覆盖 arXiv、OpenReview/ICLR/ICML/TMLR 与 ACL Anthology。

### Claim A：whole-video counterfactual weak order → retrieval-margin tangent

1. `site:openreview.net 2025 2026 semantic teacher counterfactual tangent retrieval embedding distillation`
2. `site:arxiv.org 2025 2026 multimodal intervention ordinal supervision retrieval geometry`
3. `site:openreview.net 2026 modality intervention gradient control knowledge distillation retrieval`
4. `site:aclanthology.org 2025 2026 hateful video counterfactual reasoning representation training`

Closest 为 [TextTeacher](https://openreview.net/forum?id=Xwb0aEUwKh)、[EmbedDistill](https://openreview.net/forum?id=-aEuKX6zQKmr)、[CGO](https://openreview.net/forum?id=Z51RWOPKQQ)、[RAMF](https://openreview.net/forum?id=U9KnNiuMu1)。未检到完整的 `train-only MLLM ordinal whole-video modality counterfactual → exact full-bank RGCL margin tangent → unchanged kNN inference` 组合；可守差异很窄，不是首次性的证明。

### Claim B：whole-video soft nuisance environment → class-conditional quotient

1. `site:openreview.net 2026 LLM generated concept groups invariant representation learning semantic environments`
2. `site:arxiv.org 2025 2026 language model privileged concepts nuisance invariant representation distillation`
3. `site:openreview.net 2026 multimodal semantic factor disentanglement target stance invariant representation`
4. `language-guided spurious correlation mitigation multimodal representation learning`

Closest 为 [Yang et al.](https://proceedings.mlr.press/v202/yang23j/yang23j.pdf)、[CDAL](https://openreview.net/forum?id=wtsM3MPn2P)、[CARE](https://openreview.net/forum?id=KzH1cLVU1R)、[dependent-factor disentanglement](https://openreview.net/forum?id=PgwkNC63CS)。SQ 只能守 exact RGCL memory quotient 与 train-only MLLM soft environment 的组合。

### Claim C：strict-OOF semantic failure modes → constrained retrieval optimization

1. `site:openreview.net 2025 2026 LLM failure mode discovery constrained gradient optimization representation`
2. `site:arxiv.org 2025 2026 semantic failure slices GroupDRO multimodal classification`
3. `site:openreview.net 2026 gradient projection language model teacher failure modes retrieval`
4. `site:aclanthology.org 2025 2026 MLLM error taxonomy robust training hate detection`

Closest 为 GroupDRO/minimax、[CGO](https://openreview.net/forum?id=Z51RWOPKQQ)、[GRACE](https://openreview.net/forum?id=m276fke38H)、[CDAL](https://openreview.net/forum?id=wtsM3MPn2P)。未检到直接相同的 full combination，但优化原理本身不新，ECM novelty 主要是 strict-OOF semantic diagnosis 与 final retrieval endpoint。

### 最近六个月直接任务补扫

1. `2026 hateful video detection MLLM reasoning RGCL semantic representation`
2. `2026 harmful meme detection MLLM distillation representation reasoning ACL`
3. `site:aclanthology.org/2026 hateful video detection reasoning multimodal`
4. `site:openreview.net 2026 harmful video detection gradient reasoning MLLM`

已纳入 2026-01-10 后的 DR-HM、BPDMoE-Hate、RAMF、IARE、ExPO-HM、TextTeacher、UCMKD、OptiMAG、CDAL、Formal Concept Lattices、Energy-Based Constraint Networks 等。一手页面中的 accepted/submission/preprint 状态已区分。

## 9. 唯一首跑与自动否决

唯一批准：

1. **CTE-0 strict-OOF zero-teacher capacity screen**；
2. 通过后才做 **CTE-1，最多 128 train videos/库 teacher-value pilot**；
3. 两者通过后才做 **CTE-2 seed-0**；
4. seed-0 两库 actual kNN acc/mF1 均胜全部关键 controls `>=+0.010` 后，才允许 final paired seeds 0/1/2 与 `+0.030/+0.030` 检验。

出现任一项自动否决并将 novelty 降至 `<=3/10`：

- rationale/schema/score/summary embedding concat；
- MLLM score fusion、veto、test-time rerank/arbitration；
- segment weighting、segment pseudo-gold 或 localization 替代 classification；
- teacher-selected/replaced memory keys；
- 回到 SSR 单邻居或 EDCM top-64/two-swap universe 调参；
- 更大 teacher、更多 frames/data/epoch、ensemble 或 BPDMoE 式 MoE/low-rank router；
- 只提升 native head；
- FULL 不胜 REMOVE、SHUFFLE 与 cheap/capacity-matched controls；
- 把 A0 写成理论上界、MLLM 贡献或终局成功。

**当前状态：target 未满足。** CTE 只是最值得先证伪的路线；本报告没有产生任何新性能结果，不能关闭主目标。
