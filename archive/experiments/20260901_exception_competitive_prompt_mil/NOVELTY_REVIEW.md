# Independent novelty review: Exception-Competitive Prompt MIL

截至 2026-09-01。审查对象仅为本目录 `README.md` 中已经固定的候选；未审代码、未实现、未运行实验，也未提出候选修补版。

## Verdict

**STOP — 4.7/10。**

- Gate 1（允许 adaptation 已有方法）：**PASS**。
- Gate 2（被 adaptation 的来源方法不可已用于 hateful-video detection/localization）：**PASS（窄口径）**。
- Gate 3（task adaptation 必须 non-trivial，且不能是已有目标任务机制的实现替换）：**FAIL**。

三门缺一不可，因此本候选不得实现或训练。

## 检索范围与一手资料

本次实际联网检索了精确标题、简称、核心术语及其与 hateful video / hate localization 的组合，并优先核对论文主页、会议论文与官方代码页：

- Chen et al., [Prompt-Enhanced Multiple Instance Learning for Weakly Supervised Video Anomaly Detection, CVPR 2024](https://openaccess.thecvf.com/content/CVPR2024/html/Chen_Prompt-Enhanced_Multiple_Instance_Learning_for_Weakly_Supervised_Video_Anomaly_Detection_CVPR_2024_paper.html)。PE-MIL 用 abnormal-aware semantic prompts 动态增强局部表示，并学习 normal-context prompt 来区分 anomaly 与其耦合 context、改善事件边界。
- Yang et al., [Training-Free and Interpretable Hateful Video Detection via Multi-stage Adversarial Reasoning, ICASSP 2026](https://arxiv.org/abs/2601.15115) 及[官方代码页](https://github.com/Multimodal-Intelligence-Lab-MIL/MARS)。MARS 在 hateful-video detection 中并行形成支持 hateful interpretation 的 evidence 与支持 plausible non-hateful perspective 的 counter-evidence，再综合作决定。
- Sun et al., [Towards Training-free Multimodal Hate Localisation with Large Language Models](https://arxiv.org/abs/2602.09637)。LELA 已在 frame-level hate localization 中使用多模态 caption、video context、multi-stage prompting 与 composition matching输出逐帧分数。
- Zhang et al., [CLARA: Clip-Level Multimodal Alignment with VLM-Derived Rationales for Hateful Video Detection](https://arxiv.org/abs/2608.15905)。CLARA 已在目标任务使用 utterance-aligned clips、local-global contrast、VLM rationale 与 gated Transformer。
- Lang et al., [LEAF: Towards Lightweight Explainable Hateful Video Detection via Self-Grounding CoT Guided Stage-Wise Distillation, Findings of ACL 2026](https://aclanthology.org/2026.findings-acl.604/)。LEAF 已把 self-grounded hateful-video explanation/contextual knowledge通过 stage-wise distillation注入学生模型。
- Wang et al., [SafeLens: Segment-Level Hate Speech Detection in Online Videos, AAAI 2026](https://ojs.aaai.org/index.php/AAAI/article/view/42390)。SafeLens 已做 segment-level multimodal、policy-LLM hate判断，并输出理由和 harm categories。
- 项目内最近邻：`docs/duplex/PREREG_POWA_MACIL.md`、`docs/duplex/FINAL_POWA_REPORT.md`、`archive/refine-logs/lb_scgp/FINAL_PROPOSAL.md`，以及 `research-wiki/STATUS.md` 中 source-scoped proposition graph、ASR rationale transport 与 inverse-compositional grounding 的关闭记录。

## Gate 1：允许 adaptation 已有方法

**PASS。**

候选明确以 PE-MIL 为跨任务来源，并保留其 prompt-enhanced local representation 与 weak MIL 主干。项目规则允许从相邻任务 adaptation 已有方法，不要求从零发明整个网络。因此“用了 PE-MIL”本身不是阻断理由。

## Gate 2：来源方法是否已被 hateful-video task 占用

**PASS（窄口径）。**

精确检索未发现 PE-MIL，或其完整的 `abnormal-aware prompt + learned normal-context prompt + event-relevance reasoning + weak temporal MIL` 来源核心，已经被论文用于 hateful-video detection/localization。PE-MIL 的公开任务与实验仍是 generic weakly supervised video anomaly detection，而不是 hateful video。

这个 PASS 只能支持“PE-MIL 是可用的跨任务来源”，不能支持整个候选 novel。Prompt、context reasoning、逐帧 hate scoring、clip rationale 与 policy moderation 在 hateful-video 文献中都已经拥挤；真正需要单独判断的是候选声称的 task-specific delta。

## Gate 3：task adaptation 是否 non-trivial 且未退化为目标任务已有机制的实现替换

**FAIL。**

候选相对 source-faithful PE-MIL 的唯一实质变化是：把普通 normal-context prompt命名并约束成 quotation、condemnation、reportage、victim narration 等 `exception-use` prompts，然后以

`base + alpha * (harmful-use - exception-use)`

形成最终局部分数，再用 exclusivity loss减少两侧同时激活。这个变化对任务有直观意义，但在当前目标任务版图中不是新的机制原则：

1. **与 MARS 的核心判断原则重合。** MARS 已在 hateful-video detection 中显式并行构造支持 hateful interpretation 的 evidence 与 plausible non-hateful counter-evidence，再综合两侧证据。当前候选把同一“正向 harmful explanation 对抗 non-hateful counter-explanation”原则从 LLM reasoning实现换成可学习 local prompt与一次减法。训练方式、时间粒度和参数化不同，但任务机制故事没有新增独立原则。
2. **与 POWA 的 contextual-use negation 重合。** POWA 已把 benign/reporting context作为 typed primitive；reporting/quotation context subtracts or gates targeted-hate witness，并由 fixed policy中的否定上下文参与最终 dense MIL score。当前 `h-e` 是同一 harmful witness 被 contextual exception抑制的更简单 prompt实现，不能因去掉 policy AST、transport或换成 cross-attention而重新获得 task-adaptation novelty。
3. **与项目内 LB-SCGP 已占用语义重合。** LB-SCGP 已在 hateful-video detection 明确适配 direct-speaker endorsement，以及 quotation/condemnation/reportage exception和 speaker-source/stance binding。`STATUS.md` 已据此关闭只把这些 semantic states 时间化的候选，并明确 whole-video 到 temporal endpoint/granularity 的变化不能恢复 novelty。当前候选恰好再次把同一 exception taxonomy投到每秒 prompt competition，没有新增 source attribution、stance binding或新的可识别监督原则。
4. **LELA 不是精确同构，但不能提供剩余 novelty。** LELA 的 multi-stage prompt、video context和 composition matching已经占用逐帧 context-aware hate prompting；它没有明确的 learnable signed exception branch，因此不是单独的致命碰撞。但在 MARS、POWA和 LB-SCGP 已覆盖正/反证据及 contextual exception 后，当前候选剩余差异主要是 PE-MIL 内的参数化位置。
5. **CLARA、LEAF 与 SafeLens 也不是精确同构。** 它们分别侧重 clip rationale gating、explanation distillation和 segment-level policy moderation，不能单独证明本候选重复；但它们进一步表明“把 contextual/policy semantic guidance送入局部 hateful-video representation”本身不能作为贡献边界。

因此，候选确实针对 hateful localization 的真实困难，但它讲的机制故事已经由目标任务方法讲过。其可识别新增量只剩“用 PE-MIL learnable prompt 和 signed logit实现已有 harmful-vs-exception reasoning”，属于已有目标任务机制的实现替换，违反 Gate 3。

## Matched PE-MIL control 能否隔离 delta

**能隔离工程 delta，但不能建立 novelty delta。**

保持 backbone、prompt数量、参数量、训练量一致，并用普通 learnable normal-context prompt替代 exception prompts，能够检验显式 exception semantics 是否优于 source-faithful PE-MIL。若 core 胜 control，可以说明该语义参数化对性能有用；若不胜，则直接否定其作用。

但这个 control没有比较 MARS式 counter-evidence、POWA contextual-use negation或 LB-SCGP exception/stance semantics。因此，即使它显著胜 matched PE-MIL，也只能证明“已有 hateful-specific counter-evidence语义在 PE-MIL 中有效”，不能证明提出了新的 task mechanism。Matched control解决性能归因，不会消除与目标任务已有机制的 novelty 冲突。

## Final decision

**STOP before implementation。** PE-MIL 作为跨任务来源本身合规，但候选唯一 task adaptation与 MARS 的 harmful/counter-evidence、POWA 的 contextual-use negation及 LB-SCGP 的 quotation/condemnation/reportage exception在机制层面重合。Local learnable prompts、逐秒输出、signed subtraction和 exclusivity loss属于实现选择，不足以构成新的 non-trivial hateful-video adaptation。
