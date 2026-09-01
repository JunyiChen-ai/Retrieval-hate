# Independent novelty review — policy primitive intervention

截至 2026-08-31。Reviewer 未参与实现；本候选在实现前审查。

## Verdict

**STOP，novelty 3.6/10。** 完整“POWA + 六类 dense primitives + moderation AST +
temporal MIL/AWB”组合未发现同构论文；但排除原 POWA、VLM teacher 和 dense KD 后，
新增核心等价于 train-time concept intervention、differentiable logical constraint 与
irrelevant-branch invariance，已是成熟方向。hate temporal localization 的应用组合不足以
构成独立方法贡献。

## Load-bearing mathematical failure

- 在 PEF 输出后固定其他 channels、只把一个 leaf 替换为 0/1 时，AND/OR 正向 leaf 的
  单调性、NOT context 的反单调性及无关 subtree 不变均由 AST 自动满足。zero-margin loss
  近乎恒为零；positive margin 主要推动饱和，并未学到新的 policy reasoning。
- 在 AWB 前干预 hostile/target 时，Sinkhorn 会重新归一化 marginals 并跨 timestamps
  重分配 transport，局部 frame output 没有固定方向。
- HateClipSeg 的 target 同时提高 targeted-hate、降低
  `untargeted_abuse=hostile*(1-target)`，不能指定统一符号。
- top-k bag 只有在所有 frame 同向变化时才继承单调性；AWB 不保证这一点。

因此 post-PEF 版本是 tautology，pre-AWB 版本可能施加错误约束。

## Closest primary work

1. [Laguna et al., Beyond Concept Bottleneck Models: How to Make Black Boxes
   Intervenable?, NeurIPS 2024](https://proceedings.neurips.cc/paper_files/paper/2024/hash/9a439efaa34fe37177eba00737624824-Abstract-Conference.html)：
   formalize intervenability，并用 concept interventions fine-tune；包含 VLM-generated
   concept annotations。
2. [IntCEM, Learning to Receive Help, NeurIPS 2023](https://papers.neurips.cc/paper_files/paper/2023/hash/770cabd044c4eacb6dc5924d9a686dce-Abstract-Conference.html)：
   训练期采样 concept-intervention trajectories。
3. [Counterfactual Concept Bottleneck Models, ICLR 2025](https://arxiv.org/abs/2402.01408)：
   concept counterfactual generator，提高相关 concept 输出效应并压低无关 concept 影响。
4. [Concept Bottleneck Models, ICML 2020](https://proceedings.mlr.press/v119/koh20a.html)：
   建立编辑 predicted concepts 并传播到 task prediction 的范式。
5. [Semantic Loss, ICML 2018](https://proceedings.mlr.press/v80/xu18h.html) 与
   [Logic Tensor Networks, AIJ 2022](https://www.sciencedirect.com/science/article/pii/S0004370221002009)：
   将 Boolean/FOL/fuzzy rules 转成 differentiable constraints。
6. [Information Leakage in CBMs, ICCV-W 2025](https://openaccess.thecvf.com/content/ICCV2025W/BISCUIT/html/Schoen_Measuring_and_Addressing_Information_Leakage_in_Concept_Bottleneck_Models_ICCVW_2025_paper.html)：
   Irrelevant Concept Contribution loss 已覆盖无关 concept 不应影响决策。
7. [Right for the Right Reasons, IJCAI 2017](https://www.ijcai.org/proceedings/2017/371)：
   用梯度约束模型不依赖无关输入维度。
8. [Ju et al., CVPR 2023](https://openaccess.thecvf.com/content/CVPR2023/html/Ju_Distilling_Vision-Language_Pre-Training_To_Collaborate_With_Weakly-Supervised_Temporal_Action_Localization_CVPR_2023_paper.html)：
   VLP dense temporal knowledge/pseudo-label distillation 已用于 WTAL。
9. [Not All Neuro-Symbolic Concepts Are Created Equal, NeurIPS 2023](https://papers.neurips.cc/paper_files/paper/2023/hash/e560202b6e779a82478edb46c6f8f4dd-Abstract-Conference.html)：
   固定逻辑与正确 task output 不保证 intermediate concepts 正确，存在 reasoning shortcut。

## Required controls if retained only as a diagnostic

- random-init/random-primitives/channel-permuted tautology audit；
- post-PEF、pre-AWB、post-AWB intervention 分开；
- dense BCE、generic consistency、AST sign、branch invariance、full 五臂；
- random monotone AST、policy permutation、cyclic channel、hostile-only、untyped max、
  learned monotone head；
- primitive agreement 与 policy consistency 分开报告；
- 0/1 off-manifold toggle 与 conditional replacement 分开；
- HCS target 不施加统一符号，AWB 报告跨时间 redistribution。

这些 controls 只适合证明该 regularizer 是否有工程作用，不能恢复其 headline novelty。
