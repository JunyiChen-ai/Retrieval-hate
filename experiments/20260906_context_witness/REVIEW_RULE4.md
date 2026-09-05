# 候选7独立 proposal review：GO

日期：2026-09-06。审稿者：独立 agent `proposal_review_c7`。对象：本目录 README 的初始提案；依据：`RESEARCH_ITERATION_RULES.md` 第4条。实际检索后结论为 **GO，可以实现；并非三个模块或整体 novelty 已成立**。未实现、未运行实验、未增加流程门。

## 1. 四类 STOP 核对

- **来源已用于 hateful video：未找到足以触发 STOP 的具体同构方案。** 下列已有工作占据“上下文、解释、局部证据、时间定位、VLM+时序融合”等宽泛主张，但查到的方法不是“目标/邻域四输入测量＋排除当前位置的内容残差＋同一分类器保留/删除训练、直接输出 selector”的方案。检索未发现不等于证明不存在；不得写成全球首次。
- **纯 ensemble：否。** 同一冻结 VLM 重复观测、同一个共同训练分类器三次共享参数调用，不构成独立模型集成。此结论以实现保持单模型/共享权重为前提。
- **纯 calibration/后处理：否。** selector 在训练中接受保留/删除目标，最终直接输出 q；并非对成品分数作推理期修正。
- **纯工程技巧：否。** 本提案包含预测残差及弱监督局部选择的完整训练假设，而不只是换 prompt/增加属性/换 GRU。但这些单项工程选择本身不能作为 novelty。

## 2. 实际检索的一手来源及边界

检索词覆盖 `hateful video rationale sufficiency comprehensiveness`、`hateful video erasure/counterfactual/deletion`、五篇指定方法名，以及 `rationalizing neural predictions`、`representation erasure`、`video anomaly masked contextual prediction`。采用论文、作者网页/仓库；未用二手总结建立 STOP。

| 一手来源 | 已有机制与本提案的区别 |
|---|---|
| [Lei et al., Rationalizing Neural Predictions, EMNLP 2016](https://aclanthology.org/D16-1011.pdf) | 无 rationale 标注的 generator/encoder 联训，以简短但充分的片段维持任务预测。C7 的 selector、保留预测及稀疏约束属于这一已有谱系，不能声称首创 rationale 学习。 |
| [DeYoung et al., ERASER, ACL 2020](https://aclanthology.org/2020.acl-main.408/) | 删除/仅保留解释片段衡量 comprehensiveness/sufficiency 已有。C7 把对应思想用于训练而不是只评估；不能把这两个概念作为新定义。 |
| [Li et al., Understanding Neural Networks through Representation Erasure](https://arxiv.org/abs/1612.08220) | 已使用删输入及最小删除集翻转模型判断解释 NLP 模型。C7 的删除翻转假设是迁移，不是原创 counterfactual 原理。 |
| [Balkir et al., NAACL 2022](https://aclanthology.org/2022.naacl-main.192/) | necessity/sufficiency 已用于**文本** hate speech 分类解释；不是 hateful video，因此按当前明确任务边界不触发 STOP，也不能宣称从未用于 hate speech。 |
| [Multi-Contextual Predictions with Vision Transformer for VAD](https://arxiv.org/abs/2206.08568)、[Bi-directional Predictive Network, AAAI 2022](https://ojs.aaai.org/index.php/AAAI/article/view/19898) | 缺失帧上下文预测、双向预测/重建在监控视频异常检测已有。C7 的留一位置残差是迁移假设，不是新建模范式的充分证明；全 train 视频重建也不等同只学正常性。 |
| [RAMF 论文](https://arxiv.org/html/2512.02743v1)、[作者实现](https://github.com/Multimodal-Intelligence-Lab-MIL/RAMF) | 目标描述、hate/non-hate 假定推理，加局部/全局上下文与层次跨模态融合。宽泛“反向证据＋上下文融合”已用于 hateful video；这里并未给出 C7 的真实 target/context 内容删除与 selector 删除损失。 |
| [MARS](https://arxiv.org/html/2601.15115v1) | 训练外的客观描述、支持 hate 的理由、non-hate 反证，再综合裁定。“双面证据推理”不能作为 C7 首创；叙述立场切换不等于四种内容可见性测量。 |
| [MATCH 作者稿](https://jianlang.org/papers/MATCH.pdf) | 双 proposer 生成相反线索，检索时空证据给 verifier，再由 rationale 增强检测。检查了方法及验证消融；未发现本文使用保留/删除 selector 训练。时空证据验证/可解释 hate detection 的宽主张已被占据。 |
| [CLARA](https://arxiv.org/html/2608.15905v1) | 语句对齐 clip、MoE 编码、local/global contrastive、VLM rationale-gated Transformer 与分类目标。局部/上下文建模、VLM rationale 融合本身不是 C7 novelty；方法章节没有 C7 的排除当前位置重建与共享删除分类器。 |
| [TANDEM](https://arxiv.org/html/2601.11178v1) | VL/AL 两模型交替 RL、对方上下文与结构化 timestamps/targets；其奖励明确含 GT timestamp IoU。不能声称 C7 首次提供上下文时间定位；具体双模型 RL 也不是 C7 的视频标签 selector 学习。 |
| [IARE](https://arxiv.org/html/2606.11953v1) | 多模态有害元素、上下文 rationale、CoT 信息增补与正确/错误推理的 DPO。语义属性和上下文解释已有，不能靠扩充六属性构建 novelty。 |

## 3. 三模块消融能支持什么

**M1：** `target_only` 使用同一组六属性并保留同维输入，是必要的公平语义参照：全模型相对它的差异不会仅仅来自把 hateful 改成多属性。但它同时增加邻域内容与多次观测，因此只支持“上下文条件四输入测量整体”，不能独自证明差分必要。已有 `raw_four` 才能在同四次输入条件下分析差分表示；即使胜出，也不证明得到真实因果效应或新信息。改变上下文/target 的缺失提示可能影响输出分布，这是解释边界，不是实现前 STOP。

**M2：** README 的 `no_residual` 同时删除残差输入与 reconstruction loss，**混合了表示和辅助训练两个因素**。它可以作为整个 M2 移除的主消融，但结果只能归因“残差＋重建训练包”，不能写成留一位置结构或残差单独有效。已有 `visible_reconstruction` 在保留重建训练时允许看 x_t，能回答排除当前位置是否必要。若日后要分别主张残差输入/辅助损失各自有效，应拆开单因素；这不是本次 GO 的额外门或要求现在新增实验。GRU、MSE 和一般异常残差均非新机制本体。

**M3：** `no_deletion` 只去删除 BCE，能较清楚测量删除训练在保留/稀疏其余目标之上的增量，但不代表完整 M3 中每项独立有效。`no_sparsity` 可定位稀疏作用。共享参数和归一化池化不证明消除了共同作弊；删除目标0是人为反事实假设，不是真实去害标签。最终应称“可学习证据选择的有效性”，不能仅靠性能/删除损失声称 causal faithfulness。

## 4. 整体声明与执行边界

三模块围绕同一局部选择任务有可执行关系，足以按规则放行完整方法；**把三种已知思想连起来不自动证明 novel paradigm**。只有两语料三 seed 主机制消融满足14(g)，且完整性能和同输入最强 baseline 等既有声明条件齐全后，才有可主张范围。五项等权只是固定权重，并不等于无方法超参数或已最简。

已查看项目旧 coalition/witness 失败账本相邻条目。C7 不应把自模型删除的判断称为独立真值；其新内容干预与新的训练目标可测试，不借旧账本恢复已取消的前置门。上述归因/捷径风险均交给现行完整实验和既有消融，不作为理论 STOP，不新增 smoke、premise 或 matched-control 门。
