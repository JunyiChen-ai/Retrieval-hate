# Independent novelty review: Single-VLM Probabilistic Policy-Factor Localization

截至 2026-09-02。审查对象仅为本目录 `README.md` 中冻结的候选 brief。审查为只读 novelty review：未审代码、未实现、未运行实验、未修改候选定义，也未提出替代 candidate。

## Verdict

**STOP — 4.6/10。**

| Gate | 裁定 | 核心理由 |
|---|---|---|
| Gate 1：允许 adaptation 已有方法 | **PASS（窄口径）** | PVLR 的 probabilistic snippet representation 可以作为跨任务来源；项目不要求从零开发。当前方法也只使用一个 Qwen checkpoint，不构成 multi-model ensemble。 |
| Gate 2：来源方法未被 hateful-video detection/localization 占用 | **PASS（对 PVLR 完整 core）** | 本次检索未发现 Lim et al. 的完整 PVLR core 已用于 hateful-video detection 或 localization。 |
| Gate 3：adaptation non-trivial、非直接套用/简单拼接、非已关闭 family 同构 | **FAIL** | brief 实际只保留 PVLR 的 Gaussian projection 与 text-distribution matching，再拼接目标领域已有的 policy role/counter-evidence 公式和项目已有的 global-prior/centered-local score decomposition。新增 precision 解释没有新的 role supervision 或 load-bearing constraint；atomic control 也不能把 policy semantics 与多因子聚合形式隔离。 |

三门缺一不可，因此本候选不得进入实现或正式训练。

## 检索范围与一手资料

本次联网检索了来源精确标题、简称、核心术语及其与 `hateful video`、`hate detection`、`hate localization` 的组合，并核对以下论文或官方页面：

- Lim et al., [Probabilistic Vision-Language Representation for Weakly Supervised Temporal Action Localization, ACM MM 2024](https://arxiv.org/html/2408.05955v1) 及[官方代码页](https://github.com/sejong-rcv/PVLR)。
- Sun et al., [MultiHateLoc: Towards Temporal Localisation of Multimodal Hate Content in Online Videos](https://arxiv.org/abs/2512.10408)。
- Sun et al., [Towards Training-free Multimodal Hate Localisation with Large Language Models](https://arxiv.org/abs/2602.09637)（LELA）。
- Yang et al., [Training-Free and Interpretable Hateful Video Detection via Multi-stage Adversarial Reasoning](https://arxiv.org/abs/2601.15115) 及[官方代码页](https://github.com/Multimodal-Intelligence-Lab-MIL/MARS)（MARS）。
- Wang et al., [SafeLens: Segment-Level Hate Speech Detection in Online Videos](https://ojs.aaai.org/index.php/AAAI/article/view/42390)。
- Zhang et al., [CLARA: Clip-Level Multimodal Alignment with VLM-Derived Rationales for Hateful Video Detection](https://arxiv.org/abs/2608.15905)。
- Lang et al., [LEAF: Towards Lightweight Explainable Hateful Video Detection via Self-Grounding CoT Guided Stage-Wise Distillation](https://aclanthology.org/2026.findings-acl.604/)。
- 项目内最近邻：`docs/duplex/PREREG_POWA_MACIL.md`、`docs/duplex/FINAL_POWA_REPORT.md`、`docs/V20_V26_FINAL_ITERATION_ARCHIVE.md`，以及 `research-wiki/STATUS.md` 中 policy cluster transport、exception-competitive prompt、inverse-compositional policy grounding、source-scoped proposition graph 与 refusal geometry 的关闭记录。

检索没有发现 PVLR 的精确标题、probabilistic class activation sequence，或其完整 `probabilistic adapter + VLP mean alignment + distribution contrastive learning` core 被上述 hateful-video 方法采用。Gate 2 因而只对这个窄而完整的来源 core 给出 PASS；它不等于“概率 embedding、policy reasoning 或局部 VLM scoring 在目标任务中无人使用”。

## Gate 1：PVLR adaptation 是否可用

**PASS（窄口径）。**

项目规则允许 adaptation 相邻任务方法。PVLR 的来源任务是 video-level label 监督下的 WTAL，公开实验是 THUMOS14 与 ActivityNet v1.3，不是 hateful-video detection/localization。把 snippet representation 从确定向量扩展为对角 Gaussian，并以分布和文本表示的相似度产生 temporal score，是可引用的跨任务来源。

但 brief 必须收窄 source-faithfulness 表述。PVLR 的完整 core 不只是“对角 Gaussian + text overlap”，还包括：

1. action-pretrained RGB/optical-flow base feature与 CO2-Net actionness/CAS；
2. snippet Gaussian 的 mean/covariance及 Monte-Carlo P-CAS；
3. frozen CLIP image feature对 Gaussian mean 的 VLP knowledge alignment；
4. action-category text prompt与 orthogonalization；
5. 基于 mined easy/hard action/background snippets 的 intra-distribution contrast；
6. 以 attention-weighted video GMM 和 video class关系构造的 inter-distribution contrast。

当前 brief 用一个 Qwen state同时承担视觉语言表示和 probabilistic projection，删除了 PVLR 的 CLIP-image mean alignment、action/background mining、intra/inter-distribution contrast、video GMM 与 prompt orthogonalization。这样做符合单模型约束，但它不是“完整 PVLR core 的 hate adaptation”；最多是借用 PVLR/更一般 probabilistic embedding 的局部表示原则。Gate 1 仍可 PASS，因为 adaptation 可以删改来源组件，但不能把删改后的通用 Gaussian head表述成完整 PVLR 迁移。

## Gate 2：PVLR 完整 core 是否已被目标任务占用

**PASS（窄口径）。**

本次检索未发现 hateful-video detection/localization 论文采用 PVLR 的完整 core：snippet Gaussian、VLP mean alignment、probabilistic CAS、intra/inter distribution contrast和 video GMM共同训练。目标领域邻近方法的已公开机制分别是：

- MultiHateLoc：modality-aware temporal encoders、dynamic cross-modal fusion、cross-modal contrast与 modality-aware MIL；
- LELA：多模态 caption、multi-stage prompting、composition matching与 frame-level score；
- MARS：objective description、hate-supporting evidence、counter-evidence及最终综合判断；
- SafeLens：segment-level multimodal evidence与 policy LLM structured decision；
- CLARA：clip MoE、local-global segment contrast和 VLM-derived rationale gating；
- LEAF：self-grounded explanation生成与 stage-wise distillation。

这些工作压缩了宽泛 claim，但没有占用 PVLR 的完整 probabilistic WTAL representation。故 Gate 2 不因来源占用而 STOP。

## Gate 3：task adaptation 是否形成新的 non-trivial 机制

**FAIL。**

### 1. 相对 PVLR 的新增部分是目标领域既有 policy semantics 的概率参数化

brief 将一个 atomic action category拆成 `target + conduct - exception`，HCS 再对 targeted 与 target-free clauses作 differentiable union。这个任务语义合理，但其机制原则已经在目标领域被占用：

- POWA 已使用 typed moderation primitives、predicate-target binding、policy tree以及 contextual-use negation形成 dense hateful score；
- MARS 已显式并行建模 hateful-supporting evidence与 plausible non-hateful counter-evidence；
- 项目 LB-SCGP 已覆盖 target/predicate、speaker/source/stance，以及 quotation、condemnation、reportage exception；
- LELA、SafeLens与 CLARA 已覆盖局部/segment policy-context reasoning或 VLM semantic guidance。

当前候选没有新增 source attribution、stance binding、factor supervision、policy constraint或新的观察量。它把上述 role语义改写成多个 Gaussian text distributions，并用加、减、union进入 logit。概率分布是新的参数化位置，不是新的 hateful-localization learning principle。

### 2. `uncertainty as precision` 没有成为 load-bearing task constraint

PVLR 的 uncertainty通过 probabilistic representation、VLP mean alignment与 intra/inter distribution contrast共同学习；当前 brief 删除了这些来源约束，只说高方差 factor precision较低。只有 video bag label时，没有监督规定某个 Gaussian 必须对应 target、conduct或 exception，也没有约束 covariance 必须表达“该 role 在这个时刻缺失或含混”。

因此 factor可以置换、重分配或共同退化为普通多头 hate evidence；variance也可以只成为通用 confidence/scale参数。最终 loss看到的是组合后的 `ell_t`，并不要求内部 role获得声明的语义。这不是要求实现前证明完整可识别性，而是当前声称的新增机制本身没有增加区别于普通 learnable multi-factor scorer的任务约束。

### 3. `global prior + centered local likelihood` 不是 PVLR delta，也不是新的项目机制

`z_vt = g_v + ell_vt - mean(ell_v)` 确实令常数 global term只影响跨视频尺度、centered local term决定视频内排序；但项目 V20 已明确使用“decomposed global prior + centered local judge”，后续 global/local quotient 与 residual families也反复使用同类结构。它不能作为本候选的新 adaptation delta。

此外，centering只能做代数上的职责分离。若没有归一化生成模型或可核验的 posterior factorization，`g_v` 与 `ell_vt` 仍是联合学习的 logits；把它们命名为 Bayesian prior/likelihood不会额外产生新的概率约束。它至多是已有 global/local architecture separation。

### 4. 与近期关闭 family 的关系

本候选不与 Policy-Constrained Cluster Transport 严格同构：它没有 cluster transport或 policy-feasible assignment。因此不能仅凭该 ledger 行 STOP。

但其声称的新 policy delta与已经关闭的 Exception-Competitive Prompt MIL、POWA contextual negation及 LB-SCGP policy roles在机制原则上重合；其 uncertainty部分又只是 generic probabilistic head，而非新的 hate-specific constraint；其 global/local path则重复项目已有 decomposition。整体是三个已知部件的串接，不满足 Gate 3 的 non-trivial task adaptation要求。

## Matched atomic control 是否能隔离新增机制

**不能充分隔离；只能隔离“当前复合 factorized head”相对“atomic head”的整体效果。**

优点是 control 固定了 Qwen checkpoint、probabilistic dimension、temporal context、global/local公式、MIL、参数规模和训练量，能够排除“只因换 backbone、增加训练或增加 embedding维度”造成的差异。

但它仍同时改变了多个因素：

1. core 使用 signed factor sum与 HCS clause union，control 使用 atomic distributions的 log-sum-exp；比较同时混入了 aggregation algebra、正负分支结构与 semantic role定义；
2. 所有 factor均可学习且只有 bag label监督，core胜出不能说明 learned heads仍分别表示 target、conduct与 exception；
3. control没有判断正确 factor身份/符号是否必要，因此任意多因子分解或额外可塑性也可能产生同样优势；
4. circular time permutation只检验 temporal alignment是否有用，不能检验 policy factor语义是否 load-bearing。

所以即使 core在 HMM/HCS test全面胜 atomic control，也最多证明“这个 multi-factor signed probabilistic head优于 atomic probabilistic head”，不能单凭该 control证明增益来自正确的 policy-event likelihood。更重要的是，性能归因 control本身不能恢复已经被目标领域方法占用的 target/conduct/exception与 counter-evidence机制 novelty。

## 最窄 claim boundary

如果只讨论当前 brief 可安全声称的内容，边界必须收窄为：

> 在单一 Qwen VLM 的 weakly supervised temporal scorer中，测试一个 PVLR-inspired diagonal-Gaussian multi-factor head，并以已有 moderation roles构造 signed local score，同时使用已有 global/centered-local decomposition。

不能 claim：

- 完整 PVLR 首次适配 hateful-video localization；
- 首次 probabilistic hateful-video representation；
- 首次 policy-factor、target/conduct/exception或 counter-evidence reasoning；
- Bayesian posterior或已识别的 role uncertainty；
- atomic control足以证明正确 policy semantics是性能来源。

这个最窄边界描述的是一个可做 baseline 的组合实现，不达到项目 novel method 的 Gate 3。

## Final decision

**STOP before implementation。**

PVLR 完整 core在本次检索范围内未被 hateful-video task占用，Gate 2通过；但当前 brief没有保留该完整 core，实际新增量是 generic Gaussian projection、目标领域已占用的 policy-role/counter-evidence公式，以及项目已有 global/centered-local decomposition的组合。`uncertainty as precision`没有新的 role-level supervision或约束，matched atomic control也不能把正确 policy semantics从多因子聚合形式中隔离。因此 Gate 3失败，候选不得进入正式实现或训练。
