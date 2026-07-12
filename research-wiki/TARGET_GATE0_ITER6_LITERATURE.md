# Target-driven Gate 0 · Iteration 6：MLLM 语义约束到最终全库几何的文献与查新

**冻结日期：** 2026-07-11（Pacific/Auckland）  
**在线检索窗口：** 2024-01-01 至 2026-07-11；最近六个月重点窗口为 `2026-01-11..2026-07-11`  
**本轮范围：** 只做文献、查新、机制定义和 fast-fail 设计；不改实验代码、不提交作业、不生成 teacher cache、不读取 validation/test 内容  
**最终目标：** MLLM 必须是可移除、不可被廉价替代的 train-only 机制，并使 unchanged ordinary full-video train-memory kNN 在至少 MHC-EN/MHC-ZH、paired seeds 0/1/2 上，相对 moving strongest non-MLLM RGCL 的 final accuracy 与 macro-F1 **各提高至少 `+0.030` absolute**。

> **监督红线：本项目没有片段金标注。** 唯一 gold supervision 是父视频二分类标签。uniform frames、完整 ASR/OCR、自动切分或 MLLM 输出都不是片段标注；任何 target、stance、binding、proof clause、policy-lattice assignment 或 discrepancy 都只能称为 train-only weak/privileged pseudo-signal。本轮三个候选都不做 segment weighting、segment loss、timestamp/span supervision 或 localization endpoint。

## 0. 结论先行

前五轮已经排除了 sparse edge/swap、whole-modality tangent、presentation quotient 和 pseudo-group gradient surgery。Iteration 6 只保留 **恰好三个**新机制：

| 候选 | MLLM 的 train-only 作用 | 直接优化对象 | 与已失败路线的关键差异 | 初步查新/风险 |
|---|---|---|---|---|
| **A. LB-SCGP**：Label-Blind Semantic-Certificate Gram Projection | 先对每个 train whole video 生成 label-blind、置信度化 clause graph；cache 冻结后 compiler 才接入 video label | exact-vote-constrained nearest-feasible full-bank Gram target `G*`，factor/Procrustes 后由 shared encoder 拟合 | 无 pseudo-group/权重/raw-QP/teacher key；绝对 Gram row-profile 与 rank-weighted vote magnitude 不能由 ordinal triplet loss表示 | **独立 reviewer 唯一首跑；novelty 7.0 / feasibility 7.2 / +3/+3 likelihood 5.8** |
| **B. LBOP**：Label-Blind Lattice-Barycentric Order Projection | label-blind 地输出固定 moderation lattice 的置信 lower/upper set | PSD nearest-correlation/order-polytope target，含 meet/join affine barycentric identities，再拟合 shared encoder | 不建环境/群组；barycentric identity 在 triplet orders 不变时仍可改变 | **第二储备；novelty 5.3 / feasibility 5.8 / +3/+3 likelihood 4.0** |
| **C. RHT**：Relational-Holonomy Targeting | 对 label-blind frozen candidate whole-video pairs 输出有方向的语义变换类型；teacher 不选 key | 球面 full-bank relation-vector/cycle-holonomy target，Riemannian proximal solver 后拟合 shared encoder | 有向 parallel-transport / loop holonomy 不是 pairwise distance order 或 generic triplet 的函数 | **高新颖高风险储备；novelty 7.3 / feasibility 4.4 / +3/+3 likelihood 5.0** |

**独立 reviewer 唯一批准首跑：A 的 `LB-SCGP-0 zero-teacher full-bank target/fitting capacity screen`。** 这不是因为 A 已经证明会涨三点，而是因为它最直接实现 reviewer 指出的 fresh boundary：

`train-only MLLM semantic certificate -> full-bank proximal target geometry -> shared encoder fitting -> ordinary kNN`。

候选 A 也不是 ECM archival sketch 的改名。ECM sketch 的核心输入仍是 OOF discrepancy-mode posterior，并以 mode-wise worst-margin aggregate 约束 target bank；LB-SCGP 删除 OOF mode/group/error-diagnosis 接口，改为 **label-blind per-video certificate cache + deterministic clause/label compiler + all-bank Gram row-profile identities**。若实现让 teacher 看到 label/prediction/error/margin，或重新出现 mode posterior、sample/group weights、worst-group objective、raw-gradient projection，立即判作 anti-repeat。

独立 novelty reviewer 已将完整去重与评分追加到根目录 `TARGET_REVIEW_RAW.md`。它明确淘汰原 **gold-grounded SCPT**、原 LOP 和原 RUF；本文件后续的 A/B/C 只指 LB-SCGP/LBOP/RHT，三者不得并行或堆叠。

## 1. 当前权威证据：什么真的跑过、什么没有

### 1.1 Final target 与当前 gap

`TARGET_LOOP.md` / `TARGET_STATE.json` 的 binding historical points 为：

| 数据集 | strongest documented non-MLLM acc / mF1 | 最低 `+0.030/+0.030` target |
|---|---:|---:|
| MHC-EN | 0.7888 / 0.7262 | **0.8188 / 0.7562** |
| MHC-ZH | 0.8255 / 0.7875 | **0.8555 / 0.8175** |

最终门槛仍取 `max(historical strongest point, paired same-seed REMOVE mean)+0.030`，因此同 seed baseline 若更强，target 随之上移。成功还要求 3/3 paired deltas positive、hierarchical paired bootstrap lower bound `>0`、四项 Holm FWER 0.05，以及 FULL 显著胜 REMOVE、MLLM-information SHUFFLE 与 calibrated NOISE。

### 1.2 SSR：support/correctable universe 不足

权威记录：`research-wiki/experiments/exp-ssr-b01.md` 与 `artifacts/ssr/v1/B1_DECISION.json`。

- 十个 strict train-OOF heads 与 candidate mining 均完成；没有 segment gold。
- 即使不可能地假定 **每个预选 arc 都是正确 MLLM relation**，MHC 的最好 SC upper bound 只有 `+0.0128 acc / +0.0176 mF1`、触及 7 个错例；MHC-ZH 最好为 `+0.0259 / +0.0307`、触及 15 个错例。
- `B1_DECISION=STOP`，relation extraction 和 teacher calls 没有启动。

结论只适用于 SSR 的 single-neighbour `MI/SC` event universe；不得把它写成所有学习表示的理论上界。Iteration 6 不能再做关系边筛选、单邻居修复或 prompt/teacher/阈值救援。

### 1.3 EDCM：冻结 top-64 / two-swap action space 不足

权威记录：`research-wiki/experiments/exp-edcm-a0.md` 与 `artifacts/edcm/v1/A0_DECISION.json`。

- MHC support 0.3679、reachable errors 15，oracle `+0.0273 acc / +0.0394 mF1`。
- MHC-ZH support 0.6287、reachable errors 22，oracle `+0.0380 / +0.0444`。
- 两库 support、reachable count、`+0.05/+0.05` 均失败；`A1_unlocked=false`，MLLM/OCR/teacher artifacts 为零。

结论只适用于旧表示、原 top-64、每 query 最多两次 swap；不得调 depth、swap count、prompt 或阈值。Iteration 6 必须让整个 query/key bank 共同移动，并允许旧 top-64 外样本成为新邻居。

### 1.4 CTE：不是性能失败，而是 frozen numerics-policy STOP

权威记录：`research-wiki/experiments/exp-cte-c0.md` 与 `artifacts/cte/v1/C0_DECISION.json`。

- 两库的 support、resource、finite、margin、gradient gates 均通过；minimum joint support 为 0.9681/0.9438，peak memory 约 0.096 GiB。
- frozen `2e-5` tangent/cost scalar-vector parity gate 失败：MHC T/cost error `8.053e-5/2.223e-5`；MHC-ZH `1.010e-4/2.191e-5`。
- `C0_DECISION=STOP`；C1/C2--C4、teacher/OCR/MLLM calls 全部为零。

不得事后放宽 tolerance、改 precision、anchor/radius/grid 来救 CTE。它没有证明 whole-bank learning 无效，也没有得到任何 acc/mF1 结果。

### 1.5 SQ：provenance/QC governance STOP，没有性能结果

权威记录：`research-wiki/experiments/exp-sq-s0.md` 与 `artifacts/sq/v1/S0_DECISION.json`。

- 两个 archive 的 ID/hash coverage 完整，reader forbidden-key access 为零；local-CLIP six-way proxy coverage 为 1.0。
- 原 archive 缺少 original prompt、exact model revision、generator code 和 input manifest 的 cryptographic linkage；blind whole-video presentation QC 未完成。
- 因而 `q_signal_status=PROXY_ONLY_CHEAP_FORMAT`，`S0=STOP`，`lambda_Q=null`，S1/S2--S4 locked。
- P0、power、microbenchmark、learned strict-OOF SQ-0 和所有 accuracy/mF1 comparison 都没有运行。

因此不能写“SQ 性能失败”，也不能复用无 provenance archive 作为 promoted MLLM signal。Iteration 6 若生成 teacher records，必须从第一次调用起冻结 prompt/model/input/code hash 与 atomic per-record provenance。

### 1.6 ECM：方法审阅 RETHINK；pseudo-group QP 被排除

权威记录：`refine-logs/ecm/FINAL_PROPOSAL.md`、`round-1-review.md` 和 `round-1-refinement.md`。

- 原 ECM 的 soft modes 形成 `R_m=sum_i w_im ell_i`；其 KKT update 可化为 per-example gradient 的动态重权，且投影本身属于 PCGrad/MGDA/CAGrad/common-descent family。
- raw-gradient constraints 经过 AdamW momentum、adaptive preconditioning 与 weight decay 后不再约束实际更新。
- teacher 即使不直接看到 correctness，也可从视频和 prediction 重构 scalar error propensity；error AUC 不能证明 semantic necessity。
- reviewer `4.98/10 RETHINK`；ECM 在代码、作业、cache、teacher call 之前 ABANDONED。
- `round-1-refinement.md` 的 proximal-bank 内容是 archival/non-canonical sketch，只是未来方向边界，不是可执行 ECM revision。

Iteration 6 的 binding anti-repeat 是：不做 pseudo-groups+GroupDRO/JTT/EIIL，不做 PCGrad/CAGrad/MGDA/raw-gradient QP；必须数值证明 full-bank vector intervention 不能由 scalar sample/relation weighting 重构，并胜 matched ERROR-PROPENSITY control。

## 2. 本轮硬排除区

以下任何实现都自动否决，不进入 teacher pilot：

1. **不存在的片段 gold：** segment labels、timestamp/span、关键片段、stance/target/mechanism gold、segment weighting 或 localization endpoint。
2. **既有 MLLM 接口：** rationale/schema/summary embedding concat、score fusion、veto/boost、final judge、test-time arbitration/rerank。
3. **memory selector：** teacher-selected/replaced keys、pairwise neighbor filtering、single-neighbour repair、top-64/two-swap rescue。
4. **参数路由：** router、MoE、low-rank experts/adapters 作为语义路径；BPDMoE/SSMoE 类设计直接排除。
5. **规模工程：** larger teacher、更多 frame/data/epoch/steps、ensemble 作为主要科学变量。
6. **robust-learning 改名：** semantic pseudo-groups + GroupDRO/PG-DRO/JTT/EIIL、loss/difficulty bin reweighting、PCGrad/CAGrad/MGDA 或任何 raw-gradient QP。
7. **通用 metric-learning 退化：** 若新机制最终只是 weighted pair/triplet/SupCon loss，或 `D*` 能由 ordinary RGCL/metric-loss per-example/per-relation gradients 的非负标量组合以相对残差 `<0.25` 重构，则 novelty 与 mechanism necessity 同时失败。
8. **head/memory redistribution：** 只涨 native head、teacher agreement、rationale quality、localization 或 audit，不算 final target。

## 3. 2024--2026 直接竞争地图与最近六个月补扫

以下只用 ACL Anthology、PMLR、AAAI、ICLR/OpenReview 或 arXiv 原页；外部 split/label/backbone 不同，数值只能说明机制潜力，不能迁移成本项目预期结果。

| 工作 | 状态 | 占据的空间 | 对 Iteration 6 的约束 |
|---|---|---|---|
| [RGCL](https://aclanthology.org/2024.acl-long.291/) | ACL 2024 | retrieval-guided contrastive geometry | final ordinary train-memory kNN 是本项目真实 endpoint |
| [RA-HMD](https://aclanthology.org/2025.emnlp-main.1215/) | EMNLP 2025 Main Oral | LMM adaptation + RGCL | 本地 P9/P9b 已显示直接 SFT port 只做 head/memory redistribution |
| [HVGuard](https://aclanthology.org/2025.emnlp-main.456/) | EMNLP 2025 | MLLM CoT + MoE hateful-video classifier | reasoning feature/MoE 已有直接竞争者；报告的外部大增益不能支持本地普通 kNN |
| [RAMF](https://openreview.net/forum?id=U9KnNiuMu1) | TMLR accepted, 2026-06 | objective/hate-assumed/non-hate-assumed reasoning + fusion | 对立 reasoning 文本/feature 路线已被占据 |
| [LEAF](https://aclanthology.org/2026.findings-acl.604/) | Findings ACL 2026-07 | video-label-grounded Self-Grounding CoT + stage-wise generative SMM distillation | **最近且关键的正证据。** Qwen2.5-VL-3B + LEAF 在其协议上相对 vanilla SMM 报告 substantial acc/mF1 增益；但它训练生成式 SMM 并保留 explain-then-predict inference，不是 full-bank target/ordinary kNN |
| [IARE](https://arxiv.org/abs/2606.11953) | arXiv 2026-06；文中称 SIGIR 2026 | rationale SFT/DPO | fine-grained rationale optimization 已拥挤；不能把 rationale distillation 当本轮新机制 |
| [ExPO-HM](https://iclr.cc/virtual/2026/poster/10008633) | ICLR 2026 Poster | explain-then-detect policy optimization | generative policy optimization 已占据 |
| [BPDMoE-Hate](https://aclanthology.org/2026.acl-long.480/) | ACL 2026 Main | adversarial viewpoints + gating + dual-space MoE | router/MoE 明确排除 |
| [TextTeacher](https://openreview.net/forum?id=Xwb0aEUwKh) | TMLR accepted, 2026-05 | caption/text semantic anchors shape train-time vision representation；inference 删除 teacher | train-only semantic supervision 本身不新；LB-SCGP 必须区别于直接 teacher embedding/anchor matching |
| [Privileged Information Distillation](https://openreview.net/submissions?venue=ICLR.cc%2F2026%2FWorkshop%2FDATA-FM) | ICLR 2026 DATA-FM workshop | privileged teacher information distillation | teacher train-only / inference absent 不是新 claim |
| [EmbedDistill](https://openreview.net/forum?id=-aEuKX6zQKmr) | NeurIPS 2023 poster / OpenReview record | teacher/student retrieval geometry matching | 不能泛称首次 geometric retrieval KD；LB-SCGP 不匹配 teacher embedding geometry |
| [Geometry-aware representational KD](https://openreview.net/forum?id=Yzr27JSBiV) | ICLR 2026 submission | Procrustes/Gram representation alignment | Gram loss/Procrustes 本身不新 |
| [Geometry-Aware Distillation for Biomedical VLMs](https://openreview.net/forum?id=TT7pMLmKPG) | TMLR under review, 2026-05 | class-relation directional targets + global/patch KD | label-guided geometry targets已有近邻；LB-SCGP 必须依赖 label-blind compiler 与 exact full-bank target program |
| [DARTVAE](https://arxiv.org/abs/2509.20501) | arXiv 2025-09 | LLM rules/knowledge graph + consistency/violation losses shape latent clustering | LLM-generated rules进入 latent loss 已不新；不能直接做 rule-energy/violation loss |
| [From Prompts to Proof Obligations](https://openreview.net/forum?id=2Fm081tbH4) | PhilML@ICML 2026 Poster | 倡议给 LLM 输出附 typed、machine-checkable claim/evidence/proof/abstention sidecars | “proof certificate/obligation”措辞和一般接口不是新颖点；LB-SCGP 必须依赖可测的 full-bank target operator |
| [Lattice Representation Hypothesis](https://openreview.net/forum?id=5K1FG92m5s) | ICLR 2026 Poster | concept lattice and meet/join in LLM geometry | B 的 formal-concept novelty 风险很高；只能主张 target-bank/order-polytope interface |
| [Polar Probe](https://openreview.net/forum?id=vv6pZQAc5S) | ICLR 2026 submission | relation type/distance-direction semantic code | 语义关系几何不是新概念 |
| [LLM-Augmented Semantic Steering](https://arxiv.org/abs/2605.01957) | arXiv 2026-05 | LLM externalizes semantic intent and reorganizes projection space | “LLM 语义重排 embedding space”大类已有；其目标是人机可视分析且用 augmentation/blending，不是 classifier target bank |
| [A Fresh Take on Stale Embeddings](https://proceedings.mlr.press/v235/monath24a.html) | ICML 2024 | corrector network updates cached target embeddings | target-bank correction 工程已有 prior；LB-SCGP 不新增 corrector network |
| [Learning the Target Network in Function Space](https://proceedings.mlr.press/v235/asadi24a.html) | ICML 2024 | function-space target-network update | 使原 RUF novelty过窄；因此 RUF 已淘汰，不进入最终三候选 |
| [Fisher-Preserving Guidance](https://openreview.net/forum?id=1fDAyaKHMv) | ICML 2026 regular page | low-rank Jacobian factorization for manifold-preserving update | 原 RUF 的 Jacobian/manifold correction 有强优化近邻 |
| [Gauge-invariant Representation Holonomy](https://openreview.net/forum?id=czJqKToDGq) | ICLR 2026 Poster | parallel transport loop 的 gauge-invariant representation diagnostic | RHT 不能声称首次 representation holonomy；可守点仅是 MLLM relation holonomy 进入 exact-vote full-bank target |
| [Holonomy Grid Codes](https://openreview.net/forum?id=RJBvW7ZdhG) | ICML 2026 regular page | directed actions、projective representations 与 holonomy | directed-action holonomy已有强理论近邻；RHT 必须绑定 hateful-video semantic transformations 与 ordinary kNN |
| [Relational KD using Function Vectors](https://openreview.net/forum?id=yBdD8T5LfB) | ICLR 2026 submission | relation/function-vector distillation与 analogy | relation-vector / analogy distillation不是新颖点 |

最近六个月的最重要更新是 LEAF：其 Table 1 中，Qwen2.5-VL-3B 从 MHClip-Y `73.37/62.82` acc/mF1 提高到 `79.90/76.14`，MHClip-B 从 `75.38/64.54` 到 `81.41/77.14`，HateMM 从 `77.78/76.77` 到 `84.72/83.30`。这些是该论文自己的 split/model/generative-inference protocol，不能与本地 RGCL cell 横比；它们只说明 LMM semantics 在 hateful-video classification 上可能产生 substantial 量级，而不仅是解释质量。LEAF 把 gold label 当 “golden prior”；LB-SCGP **不复制这一点**，teacher 必须 label-blind，标签只能在 cache 冻结后由 compiler 使用。LEAF 的生成式 student/inference 也不同于 ordinary kNN。

## 4. 候选 A：LB-SCGP（Label-Blind Semantic-Certificate Gram Projection）

### 4.1 核心 claim

**Claim A：** label-blind MLLM whole-video clause graph 可在 cache 冻结后与 video label 一起被编译成全库、非样本加权的 Gram row-profile / exact-vote constraints；minimum-displacement target bank 经原 shared encoder 拟合后，可产生 scalar sample/relation weighting 无法实现的几何修复并提高 final kNN acc/mF1。

### 4.2 Teacher record：严格 label-blind certificate

对每个训练视频只提供 whole-video evidence（uniform frames + full ASR/OCR/title）。teacher **绝不看到** label、prediction、margin、error、neighbour 或 segment/span。它只输出受限 certificate：

1. `policy_clause`：仅从冻结 policy grammar 组合 `protected_target`、`derogatory_or_exclusionary_proposition`、`speaker_stance`、`quotation/condemnation/satire/reportage exception`、`cross_modal_binding`；
2. `support_state`：每个 clause 为 `{supported, contradicted, unresolved}`；
3. `proof_links`：哪些原子必须共同成立、哪个 exception 否定哪个表面读取；
4. `confidence` 与四调用一致性；
5. 不允许 output verdict、score、memory ID/key、segment/span/timestamp 或自由 rationale。

这些字段不是 stance/target/mechanism gold，也不训练 auxiliary head。低置信、parse failure、clause contradiction 或四调用 graph 不闭合时，record 进入 exact REMOVE fallback。

[LEAF](https://aclanthology.org/2026.findings-acl.604/) 将 gold label 当作 golden prior；独立 reviewer 判定本候选不能复制该数据流。LB-SCGP 必须先完成 label-blind cache，compiler 才能读取 parent-video binary label。

### 4.3 Deterministic certificate compiler

所有 cache 关闭后，compiler 才读取 video labels。它不请求 MLLM 选择 pair/key，也不筛 memory。对全部可靠 records 构造固定维度 clause-incidence matrix `C` 和 exception operator `E`，并从 **所有训练视频** 计算：

- semantic-equivalent row-profile identities：相同 clause composition、相同 label、不同 surface context 的视频，其对全库 clause bins 的 signed similarity-vote contribution profile应相近；
- exception-reflection identities：表面 proposition 相近但 stance/quotation/condemnation exception 相反的 certificates，其 row profiles 应沿对应 policy dimension 发生已知符号反射；
- unresolved records 不产生 constraint，不被重权或删除；
- 所有 relation 由 frozen compiler 从 certificates 全量生成，不存在 teacher-chosen key 或 test-time graph。

该接口比 P4 更强：P4 让学生逐样本预测 schema field；LB-SCGP 不预测 field，而把 **clause composition 与 exception algebra** 编译成全库 row-profile identities。它也不同于 SSR：没有 MLLM pair judgment 或 single-neighbour event，所有 bank coordinates 都可移动。

### 4.4 Full-bank proximal target program

在每次 registered refresh，以 eval-mode shared encoder 建立 self-excluded、L2-normalized train bank `Z0 in R^(Nxd)` 和 `G0=Z0 Z0^T`。稳定排序为 cosine descending、canonical ID ascending；repository top-20 arithmetic similarity vote 必须逐 ID/rank/cosine/prediction parity。

先在 Gram coordinate 解 nearest-feasible target：

`min_G 0.5 ||G-G0||_F^2 + rho ||P_sem(G;C,E)||_2^2`

subject to：

- `G >= 0`（PSD）、`diag(G)=1`、rank surrogate 不超过可拟合预算；
- 每个视频的 self-excluded exact top-20 signed vote margin不低于 frozen label-only target program 的 registered lower envelope；
- 两个 video classes 的 mean exact-vote margin都不下降；
- semantic row-profile / exception-reflection residual 分别低于预注册阈值；
- per-row displacement 与 class-centroid displacement 有 fixed trust region；
- exact stable top-20 每个 proximal iteration 重新计算，不能固定旧 top-64 universe。

由 `G*` 做 rank-`d` PSD factorization 得到 `Z*`，再用 orthogonal Procrustes 对齐到 `Z0`。`Z*` 是 stopped-gradient target，不是 teacher embedding、text anchor、memory replacement 或 inference key。

### 4.5 Shared-encoder fitting 与 inference

在与 REMOVE 完全相同的参数、总 steps、epochs、optimizer、scheduler 和 bank refresh budget 下，用预注册比例的 uniform full-video minibatches拟合：

`L_fit = mean_i ||row_norm(f_theta(x_i)) - stopgrad(Z_i*)||_2^2`。

其余 steps 执行原 RGCL。REMOVE 把同样 steps 用于普通 RGCL；LABEL-ONLY-TARGET、CERT-SHUFFLE、CERT-NOISE、P4-AUX、TextTeacher-style anchor、pair/triplet metric baseline 和 archival ECM mode-target 获得同等 tuning/fit budget。

每个 fit block 后重建 actual eval-mode bank，只报告 realized rank churn、semantic residual、exact kNN acc/mF1；若任一 class/global margin下降或 target-realized cosine低于门槛，回滚 model+optimizer state 并执行 REMOVE steps。validation/test 完全不加载 teacher/certificate/target bank，只用 full video -> shared encoder -> ordinary kNN。

### 4.6 非重权 / 非通用 metric-learning 的必要证明

LB-SCGP 不能靠文字声明“不是 metric learning”。在 zero-teacher 与 teacher pilot 都要做两个 gradient-cone 审计：

1. `SCALAR-EXAMPLE-FIT`：以 ordinary RGCL per-example embedding gradients 的非负标量组合拟合 `vec(Z*-Z0)`；
2. `SCALAR-RELATION-FIT`：以全部 weighted pair/triplet/SupCon relation gradients 的非负组合拟合同一 displacement。

除 relative residual `>=0.25` 外，还必须输出 Farkas dual separating certificate，dual gap 超出数值容差，且 learned controls 的 OOF acc/mF1 不得匹配 FULL。否则机制退化为 scalar weighting / generic metric learning，LB-SCGP STOP。

### 4.7 Closest work 与可守 delta

- **LEAF：** gold-grounded explanations + generative student；LB-SCGP teacher label-blind、无 rationale prediction。
- **TextTeacher：** text embedding anchors；LB-SCGP 不匹配 teacher embedding。
- **EmbedDistill / geometry-aware KD：** teacher Gram matching；LB-SCGP 的 target由 label-blind certificates、post-cache labels和 exact vote共同求解。
- **DARTVAE：** LLM rules直接进入 latent loss；LB-SCGP 先解 full-bank target并要求 Farkas非重权证书。
- **Formal Sidecars：** proof obligation接口已有；可守点不是 certificate，而是 exact-vote Gram target compiler。
- **ECM sketch：** OOF pseudo-modes + worst-mode target；LB-SCGP 没有 mode/group/error trace。

因此可守的 novelty 只能是完整接口：

`label-blind certificate cache -> post-cache label compiler -> exact-vote-constrained full-bank Gram target -> shared RGCL encoder fitting -> unchanged ordinary kNN`。

不能声称首次 semantic supervision、rule-guided latent learning、geometry KD、target embedding 或 function-space fitting。

### 4.8 Fast-fail

**LB-SCGP-0 · zero-teacher capacity / fitting screen：**

1. 只用 train folds 与 video labels，certificate compiler 退化为 two-clause `label-only` proof；零 teacher/new OCR call。
2. 验证 evaluator parity、PSD/factor scalar-vector parity、self exclusion、stable tie、rank recomputation、target-realized fitting、SLURM cost和两种 scalar-fit residual。
3. strict nested five-fold OOF actual ordinary kNN 相对 frozen geometry在 MHC/MHC-ZH 的 acc 与 mF1 均 `>=+0.050`，每 fold sign positive；否则 operator capacity不足，零 teacher STOP。
4. 若 label-only target成为更强 non-MLLM method，它立即更新 moving baseline；teacher FULL 未来必须在此基础上再过 target。

**LB-SCGP-1 · 128 videos/dataset label-blind teacher governance + semantic necessity pilot：**

1. class×baseline-prediction×confidence 分层抽样在调用前冻结；四调用 schema/graph closure、provenance、confidence/fallback完整。
2. 不以 label/error AUC 解锁。要求 proof-link identity 对 held-out semantic row-profile residual 和 correction direction 提供相对 video label、P4 fields、caption embedding、scalar ERROR-PROPENSITY 的 conditional gain。
3. CERT-SHUFFLE 在 label×prediction×margin×error-propensity fine cells 内破坏 certificate identity；LABEL-ONLY-TARGET、P4-AUX、TextTeacher-anchor、pair/triplet metric、DARTVAE-style rule loss都为 binding controls。
4. teacher FULL target displacement对两种 scalar-fit residual均 `>=0.25`，actual fit 后 semantic residual下降且 exact kNN OOF acc/mF1 在两库相对所有 binding controls各 `>=+0.010`；否则 STOP。

**LB-SCGP-2 · seed-0：** 两库 dev ordinary-kNN acc/mF1 对 REMOVE、LABEL-ONLY、CERT-SHUFFLE、CERT-NOISE与最强 prior-art control均 `>=+0.010`；noise corruption `{0,.25,.50,.75,1}` 对 gain 单调。

### 4.9 Novelty 初评

独立 reviewer：novelty `7.0/10`、feasibility `7.2/10`、达到 `+3/+3` likelihood `5.8/10`、EV `6.6/10`；**唯一首跑**。若 compiler 退化成 pair/triplet margins或缺少 Farkas separation，novelty 降至 `<=3/10`。

## 5. 候选 B：LBOP（Label-Blind Lattice-Barycentric Order Projection）

### 5.1 核心 claim

**Claim B：** label-blind MLLM 为每个 train whole video 返回固定 moderation-policy lattice 的置信 lower/upper set；将 meet/join affine barycentric identities 与 isotonic intervals 投影到全库 PSD Gram/order polytope，再拟合同一个 encoder，能改善 ordinary kNN。

### 5.2 Mechanism

在 teacher 调用前冻结一个很小的 policy lattice：

- atoms：protected target、harmful proposition、endorsement、quotation/condemnation/satire exception、cross-modal binding；
- meet 表示 conjunction，join 表示可替代 evidence path；
- terminal policy state只有 binary video label，但 lattice intermediate state不是 gold。

MLLM 只输出每个视频必然满足的 lower set、可能满足的 upper set、置信度和 unresolved；不输出 label、score、memory pair/key 或 rationale。compiler 形成每个视频的 ordinal interval，不把视频分配给一个 environment/group。

目标变量为全库 correlation/Gram matrix `G` 及固定、非数据依赖的 lattice anchor coordinates `A`。解：

- nearest PSD/unit-diagonal projection to `G0`；
- video-to-lattice compatibility 必须在 lower/upper set 间满足 isotonic inequalities；
- meet/join compatibility 在 Gram row profiles 中近似保持；
- affine barycentric identity 如 `u_(a meet b) ~= norm(alpha u_a + (1-alpha)u_b)`；
- exact ordinary-kNN label margins与两个 class global margins不下降；
- factor `G*`、Procrustes align、shared encoder fit，测试删除所有 lattice/anchors。

与 A 的区别：A 使用真实训练视频之间由 proof certificates 编译的 clause/exception row-profile identities；B 不建立 data-data certificate relation，使用固定 policy lattice 上的 per-video partial order interval。

### 5.3 Closest work 与风险

- [Lattice Representation Hypothesis](https://openreview.net/forum?id=5K1FG92m5s) 已证明/分析 LLM embedding 中 concept-lattice meet/join structure；formal concept lattice 不能作为首创点。
- [Polar Probe](https://openreview.net/forum?id=vv6pZQAc5S) 表明 relation existence/type 可由 distance/direction编码；ordinal semantic geometry不新。
- [LLM semantic steering](https://arxiv.org/abs/2605.01957) 用 LLM externalized intent重排 projection spaces；LLM-guided geometry steering不新。
- [Geometry-aware KD](https://openreview.net/forum?id=Yzr27JSBiV) 占据 Gram/Procrustes alignment。

可守 delta 仅是：`uncertain whole-video moderation lattice interval -> exact-kNN-constrained full-bank order-polytope projection -> shared RGCL fitting`。主要风险是 lattice人为、teacher interval collapse、固定 anchors等价于 TextTeacher/concept bottleneck，以及最终退化为 ordinal metric learning。

### 5.4 Fast-fail

**LBOP-0 zero-teacher：** 仅用 video labels构造 two-element lattice，验证 PSD/order projection、encoder fitting与 ordinary OOF kNN；两库 acc/mF1均需 `>=+0.050`、每 fold同号，并通过共同 Farkas/gradient-cone 审计。失败零 teacher STOP。

**LBOP-1 pilot：** 每库最多128 label-blind train videos；要求 lower/upper interval非退化 coverage `>=80%`、四调用 lattice closure `>=85%`，并胜 label-only lattice、caption anchor、random interval和P4 hierarchy。

**LBOP-2 seed-0：** 两库 dev acc/mF1 对 REMOVE、LABEL-ONLY-LATTICE、INTERVAL-SHUFFLE/NOISE、TextTeacher-anchor、ordinary ordinal/triplet metric controls均 `>=+0.010`。

### 5.5 Novelty 初评

独立 reviewer：novelty `5.3/10`、feasibility `5.8/10`、`+3/+3` likelihood `4.0/10`、EV `4.8/10`；第二储备，不与 A 并行。

## 6. 候选 C：RHT（Relational-Holonomy Targeting）

### 6.1 核心 claim

**Claim C：** label-blind MLLM 对冻结、label-blind 预选的 whole-video pairs 输出有方向的语义变换类型；同类型 pair-of-pairs 的 relation-vector alignment 与 cycle holonomy进入球面 full-bank target，可表达 generic triplet 无法表示的结构并改善 ordinary kNN。

### 6.2 Mechanism

Teacher 不选 key、不看 label/correctness；输入是冻结 base bank 以 label-blind rule 预选的 whole-video pair，输出 quotation->endorsement、reportage->assertion、target-binding change、context inversion 等有方向类型及 confidence。对同类型 quadruple `(i,j,k,l)` 约束：

`L_hol = ||PT_(i->k) log_(u_i)u_j - log_(u_k)u_l||^2`。

同时要求闭环 parallel-transport residual接近零与 exact top-20 margin envelope满足。Riemannian trust-region/SQP 求 minimum-displacement spherical target `U*`，每步 exact重排、vote parity、accept/reject，随后 shared encoder拟合 `U*`；test仍为 ordinary kNN。

### 6.3 Closest work 与风险

- [Gauge-invariant Representation Holonomy](https://openreview.net/forum?id=czJqKToDGq) 已占据 representation holonomy diagnostic。
- [Holonomy Grid Codes](https://openreview.net/forum?id=RJBvW7ZdhG) 已研究 directed actions 与 holonomy。
- relational KD、analogy/translation embedding、Polar Probe 与 geometry KD 占据 relation-vector/geometry 大类。

可守 delta 仅是 `MLLM relation holonomy × exact-vote full-bank proximal target`。主要风险是可靠 quadruple覆盖不足、pair calls/graph closure昂贵和parallel transport数值复杂。

### 6.4 Fast-fail

**RHT-0 zero-teacher：** label-only spherical target/fitting与 holonomy numerical micro；strict OOF 两库 acc/mF1均 `>=+0.050`、每折同号，并过共同 Farkas审计，否则零 teacher STOP。

**RHT-1 pilot：** bounded label-blind pair pilot须证明 transformation/quadruple coverage；FULL 胜 generic triplet、无方向 relation、TRANSFORM-SHUFFLE/NOISE、embedding-cluster transform、ERROR-PROPENSITY、LABEL-ONLY和scalar reweight。

**RHT-2 seed-0：** 两库 dev acc/mF1对所有 binding controls各 `>=+0.010`，corruption单调，并通过共同 Farkas separation。

### 6.5 Novelty 初评

独立 reviewer：novelty `7.3/10`、feasibility `4.4/10`、`+3/+3` likelihood `5.0/10`、EV `5.5/10`；高风险第三储备。原 RUF 已淘汰。

## 7. Claim-level query pack 与可核验 URL

下面记录本轮实际使用的检索表达。查新结论是“未检到完整重合”，不是数学意义上的首次证明。

### Claim A：label-blind certificate -> full-bank Gram target -> encoder fitting -> ordinary kNN

至少使用了以下检索式：

1. `site:openreview.net 2025 2026 optimize target embeddings full dataset proximal representation learning distillation geometry`
2. `site:arxiv.org 2025 2026 full-bank target embedding optimization student encoder fitting knowledge distillation retrieval`
3. `site:openreview.net 2026 executable semantic constraints latent geometry teacher`
4. `2026 hateful video detection MLLM semantic constraints representation geometry kNN`
5. `site:openreview.net 2025 2026 LLM generated semantic constraints representation learning embedding geometry`

核验入口与 closest：

- [LEAF, Findings ACL 2026](https://aclanthology.org/2026.findings-acl.604/)
- [TextTeacher, TMLR 2026](https://openreview.net/forum?id=Xwb0aEUwKh)
- [DARTVAE, arXiv:2509.20501](https://arxiv.org/abs/2509.20501)
- [From Prompts to Proof Obligations, PhilML@ICML 2026](https://openreview.net/forum?id=2Fm081tbH4)
- [EmbedDistill](https://openreview.net/forum?id=-aEuKX6zQKmr)
- [Geometry-aware representational alignment](https://openreview.net/forum?id=Yzr27JSBiV)
- [RedEx, ALT 2024/PMLR](https://proceedings.mlr.press/v237/daniely24b.html)
- [A Fresh Take on Stale Embeddings, ICML 2024](https://proceedings.mlr.press/v235/monath24a.html)

未检到 `label-blind certificate cache + post-cache label compiler + exact-vote constrained Gram target + shared RGCL fitting + ordinary kNN` 的完整组合。风险最高的近邻是 LEAF、DARTVAE、TextTeacher、geometry KD 和 ECM archival target。

### Claim B：label-blind lattice interval + barycentric identities -> Gram/order projection -> kNN

至少使用了以下检索式：

1. `site:openreview.net 2026 semantic constraints Gram matrix projection representation learning`
2. `site:openreview.net 2026 isotonic embedding semantic partial order representation learning`
3. `site:arxiv.org 2025 2026 nearest correlation matrix semantic embedding constraints LLM`
4. `site:openreview.net 2025 2026 LLM generated semantic constraints representation learning embedding geometry`
5. `site:arxiv.org 2025 2026 MLLM teacher symbolic constraints embedding space classification train-only`

核验入口与 closest：

- [Lattice Representation Hypothesis, ICLR 2026](https://openreview.net/forum?id=5K1FG92m5s)
- [Polar Probe](https://openreview.net/forum?id=vv6pZQAc5S)
- [LLM-Augmented Semantic Steering](https://arxiv.org/abs/2605.01957)
- [Geometry-aware representational alignment](https://openreview.net/forum?id=Yzr27JSBiV)
- [Visualizing pairwise similarity via SDP, PMLR](https://proceedings.mlr.press/v2/globerson07b.html)
- [Coding-Theoretic Hyperspherical Prototypes, GRaM 2024](https://proceedings.mlr.press/v251/lindstrom24a.html)

没有发现完全相同的 moderation-lattice full-bank order projection，但 concept lattice、partial-order embeddings、semantic steering、PSD Gram optimization 都是强 prior，故 B 不能声称一般 lattice/ordinal geometry novelty。

### Claim C：label-blind directed semantic relations -> spherical holonomy target -> kNN

至少使用了以下检索式：

1. `site:openreview.net 2024 2025 2026 relational holonomy embedding representation learning parallel transport`
2. `site:arxiv.org 2025 2026 relation vectors cycle consistency holonomy embedding knowledge distillation`
3. `site:proceedings.mlr.press 2024 2025 Riemannian relation embedding parallel transport analogy representation`
4. `site:openreview.net 2026 directed relation vector alignment representation learning`

核验入口与 closest：

- [Gauge-invariant Representation Holonomy, ICLR 2026](https://openreview.net/forum?id=czJqKToDGq)
- [Holonomy Grid Codes, ICML 2026](https://openreview.net/forum?id=RJBvW7ZdhG)
- [Relational KD using Function Vectors](https://openreview.net/forum?id=yBdD8T5LfB)
- [Polar Probe](https://openreview.net/forum?id=vv6pZQAc5S)
- [Relational Knowledge Distillation](https://arxiv.org/abs/1904.05068)

未检到 `MLLM directed whole-video semantic transformations -> relation-vector/cycle holonomy target -> exact-vote hateful-video full bank -> encoder fitting` 的完整组合；但 representation holonomy 与 directed-action holonomy 已有，RHT 不能主张 holonomy primitive 首创。

### 最近六个月 direct-task 补扫

使用了：

1. `2026 hateful video detection MLLM semantic constraints representation geometry kNN`
2. `2026 harmful meme detection MLLM train-only teacher embedding geometry`
3. `site:aclanthology.org/2026 hateful video MLLM representation learning reasoning`
4. `site:openreview.net 2026 harmful meme MLLM semantic geometry retrieval`

已核验 2026-01-11 之后的 [LEAF](https://aclanthology.org/2026.findings-acl.604/)、[RAMF](https://openreview.net/forum?id=U9KnNiuMu1)、[IARE](https://arxiv.org/abs/2606.11953)、[BPDMoE-Hate](https://aclanthology.org/2026.acl-long.480/)、[TextTeacher](https://openreview.net/forum?id=Xwb0aEUwKh)、[Lattice Representation Hypothesis](https://openreview.net/forum?id=5K1FG92m5s)、[Fisher-Preserving Guidance](https://openreview.net/forum?id=1fDAyaKHMv)、[InstEmb](https://openreview.net/forum?id=fwWvqAOXMT) 等一手页面。accepted、submission、workshop 与 preprint 状态均分开表述。

## 8. 统一 final protocol

无论 reviewer 选择哪一个候选，都必须按同一终局标准：

1. zero-teacher screen 只用 train folds 和 parent-video binary labels；它是容量/成本门，不是理论上界或 MLLM evidence。
2. teacher cache 只含 train IDs；从第一次调用起冻结 exact prompt/model revision/generator/input/code hashes；validation/test teacher artifacts 为零。
3. teacher 必须 label-blind：不得收到 video label、prediction、error、margin、loss、neighbour label/ID；label 只在 cache closure 后进入 compiler。
4. MHC-EN 与 MHC-ZH × paired seeds 0/1/2；同 split、preprocessing、backbone、epochs/steps、optimizer、checkpoint rule、top-20 vote和parameter count。
5. 每 metric 对 `max(historical strongest, paired same-seed REMOVE mean)` 至少 `+0.030`；3/3 deltas positive；hierarchical paired-bootstrap lower bound `>0`；四项 Holm FWER 0.05。
6. FULL、REMOVE、teacher-information SHUFFLE、calibrated NOISE 与候选专属 strongest cheap/prior-art controls。
7. 必须胜 LABEL-ONLY target、ERROR-PROPENSITY、scalar-weight/triplet controls；relative cone residual `>=0.25` 且有 Farkas dual separating certificate。
8. actual ordinary full-video kNN acc/mF1 是唯一 endpoint；native head、rationale、localization或target objective都不能替代。
9. 唯一 gold 始终是 video binary label；所有 MLLM semantics 都是 weak/privileged pseudo-signal。

## 9. 唯一首跑建议与自动停止

独立 reviewer 唯一批准：

1. **LB-SCGP-0 zero-teacher full-bank target/fitting capacity screen**；
2. 通过后才做 **LB-SCGP-1，最多 128 label-blind train videos/dataset** 的 certificate pilot；
3. 通过后才做 **LB-SCGP-2 seed-0**；
4. 两库 seed-0 actual kNN acc/mF1 都胜所有 binding controls `>=+0.010` 后，才允许 paired seeds 0/1/2 final `+0.030/+0.030`。

以下任何项自动停止 LB-SCGP 并重新 Gate 0：

- LB-SCGP-0 任一库任一 metric `<+0.050`，或任一 fold sign 非正；
- target displacement residual `<0.25`、Farkas separation失败或 scalar/triplet control匹配；
- teacher 看到 label/prediction/error/margin/neighbour 或 cache closure 后补写 teacher record；
- teacher certificate只复述 label/P4 fields/error propensity，或 proof graph closure/coverage失败；
- FULL 不胜 LABEL-ONLY、CERT-SHUFFLE、CERT-NOISE、ERROR-PROPENSITY、P4/TextTeacher、pair/triplet metric与最强 prior-art control；
- 只有 target objective/native head改善而 actual ordinary kNN 不涨；
- 任何 segment gold/weight、teacher key、test-time MLLM、router/MoE、scale rescue、更多 epoch/frame/teacher size；
- 将 A 改回 ECM mode posterior/worst-group target，或将 B/C 堆叠到 A。

## 10. 独立 novelty reviewer 记录

独立 reviewer `/root/iter6_architect` 已完成合并去重，完整 raw response 已追加根目录 `TARGET_REVIEW_RAW.md`：

- 保留恰好三条：LB-SCGP、LBOP、RHT；
- 淘汰原 gold-grounded SCPT、原 LOP 与原 RUF；
- no-segment-gold 审计 PASS；
- 唯一首跑为 **LB-SCGP-0**，B/C 不并行、不堆叠；
- A/B/C 分别必须以 absolute Gram/row-profile、affine meet/join barycentric identity、directed holonomy 证明结构性非-triplet，并共同提交 gradient-cone residual + Farkas dual certificate。

**当前状态：target 未满足。** 本报告没有产生新 accuracy/macro-F1 结果，没有启动 teacher 或实验，不能关闭主目标。
