# 候选5：独立 proposal review（规则4）

日期：2026-09-05。审阅对象：本目录 README 的提案版，尚无实现或训练结果。审阅者：独立 agent `proposal_review_c5`。只采用现行规则4的四类 STOP；旧 premise、失败同族门不生效。本记录不证明 SOTA、模块有效性或论文 novelty 已成立。

## 裁定：GO

按当前具体提案，四类 STOP 均未成立，允许实现。检索未发现“同一冻结VLM四种内容输入的有符号依赖证据 + 正/负关联时间读出 + 训练内Yager冲突转未知”已用于 hateful video detection/localization。未检出不等于证明不存在；下述已占用主张和项目内重复必须准确披露。

| 规则4检查 | 结论与边界 |
|---|---|
| 来源已用于 hateful video | 未找到完整具体机制的先例。泛化的 VLM 正反证据推理、证据引导融合已存在，不能作为本候选独有贡献。 |
| 纯 ensemble | 提案四次调用同一冻结VLM作为输入测量、两个证据头属于一个共同训练的网络，未使用独立训练模型或teacher集合，不按 ensemble STOP。实现不得换成独立专家checkpoint、独立VLM预测投票或训练期多teacher聚合。 |
| 纯 calibration / 后处理 | Yager融合直接参与训练forward、bag损失及最终分数，是模型内部读出，不是给已有最终预测追加校准。实现必须保持同一训练/推理路径、梯度经过融合、无推理专用变换。 |
| 纯工程技巧 | 全候选改变观测、关系建模和监督内融合，超出只换特征/超参。单独把四路logits当新特征并接旧头，不能借此次GO直接认定三个模块已novel。 |

## 实际检索与一手来源

查询覆盖 `hateful video detection counterfactual modality intervention coalition Shapley`、`hateful video Dempster Shafer Yager evidential fusion`，以及精确短语 `hateful video` 配合 `evidential`、`contrastive decoding`、`counterfactual`、`Yager`、`negative attention`、`Shapley`、`Dempster`。补查近作 MARS、RAMF、MATCH、CLARA。以下结论以论文原文/作者或出版方页面为依据，未使用二手博客作论据。

1. [VCD，CVPR 2024](https://arxiv.org/abs/2311.16922)：对原图/扰动图条件分布作对比解码，任务为VLM幻觉。可作为模型输入依赖测量动机；不是四路二因子分解的唯一数学来源，也不能把本方法称为VCD原算法。
2. [Trusted Multi-View Classification，ICLR 2021](https://arxiv.org/abs/2102.02051)：Dirichlet证据/不确定性与Dempster式多视图融合已有先例。因此 `alpha=e+1` 和证据融合本身不是新数学。
3. [Yager，Information Sciences 1987](https://doi.org/10.1016/0020-0255(87)90007-7)：讨论Dempster冲突归一化并提出替代组合规则，是冲突转未知的原始来源。应明确写“适配Yager规则”，不能说提出全新冲突算子。该来源不涉及 hateful video。
4. [MARS，2026](https://arxiv.org/html/2601.15115v1)，方法2.2：同一内容的 hate/non-hate 假设推理，再综合裁定；已经用于 hateful video。与本提案内容遮蔽测量不同，但已占用“考虑支持与反证”的宽泛主张。
5. [RAMF，2025](https://arxiv.org/html/2512.02743v1)，方法与引言：客观描述、hate/non-hate推理与 Local-Global Context Fusion、Semantic Cross Attention。已经把正反推理接入 hateful video 的可训练融合，不能把“三模块：VLM推理—时序骨干—融合”本身说成首次新范式。未在原文中发现四内容干预/Yager融合。
6. [MATCH，作者论文](https://jianlang.org/papers/MATCH.pdf)：对立视角证据提议、时空证据验证及视频特征整合，也是 hateful video。其多agent生成—验证机制与本方案不同，但禁止声称首次显式证据比较或时空证据整合。
7. [CLARA，2026-08](https://arxiv.org/abs/2608.15905)：clip编码、局部/全局对比和VLM rationale门控整合，已用于 hateful video。仅声称clip级VLM引导与统一三部件不足以区别该工作。
8. [Text or Image?，EACL Findings 2024](https://aclanthology.org/2024.findings-eacl.8/)：Shapley模态贡献已用于 hateful meme 的分析。静态 meme 不触发当前 hateful-video 来源STOP，但需避免泛称首次在“仇恨多模态任务”分解模态贡献。

## 项目旧 coalition / dividend 核对

实际读取 `archive/experiments/20260901_temporal_coalition_dividend/README.md` 与 `archive/experiments/20260831_coalition_witness_candidate/README.md`。旧方案本身就是同一共享网络的 masked forward，并不是多个独立模态模型的投票。新README中“不是聚合多个模态模型预测”可以描述新方案合规性，不能作为区别旧方案的论据。

新 `d_av=L_av-L_v-L_a+L_0` 正是二玩家的有符号二阶 Möbius/Harsanyi 交互形式，数学上不是新分解。实际区别是：旧方案在训练中的三模态特征网络上形成局部唯一分数，旧dividend提案截断负项；新方案在冻结VLM的原始帧/ASR输入上观测二因子条件分布，保留负项和熵，作为后续关系学习的输入，而非直接重构最终得分。该差别允许实验检验；不采用已撤销的“同族必STOP”。

还需披露 `L_0,d_v,d_a,d_av` 与四原始logits是可逆线性变换，全部保留时没有增加信息。差分可以改变归纳偏置，但不能仅凭改坐标宣称新的证据来源。

## 必须明确的实现与主张边界

1. 模块1写清“模型内容依赖性测量”，不声称已识别真实因果效应。空白图/缺失ASR提示可能改变模型分布；这是待检验限制，不是STOP理由。读取Yes/No token须核对分词与相同前缀，不能混用token概率与完整答案概率。
2. 模块2的 `softmax(-qk)` 只是低关联token的读出，不保证语义上反驳当前判断。实现/文档先称正关联、负关联读出；“反证”需由诊断支持。证据key对应的content value必须有明确窗/秒映射，两个长度不同的序列不能直接当成相同索引。
3. 模块3明写二类公式：`C=b1[0]*b2[1]+b1[1]*b2[0]`；`b[k]=b1[k]*b2[k]+b1[k]*u2+u1*b2[k]`；`u=u1*u2+C`；`p[k]=b[k]+u/2`。这是标准Yager适配，不额外加Dempster的 `1/(1-C)`。两个分支相关，不能据此声称融合意见满足独立证据假设或概率已校准。
4. HMM只产生train块目标，不得同时保留推理HMM先验，否则“模块3替换原融合”的归因不成立；train拟合不得读validation/test标签。
5. 删除旧coalition是多模型的暗示；补引Yager；把模块2语义名称收窄。以上是本次评审明确化事项，由主agent落实，不要求第二次proposal review。

## 三模块目标的消融与超参核对

现有 `raw_verdict`、`ordinary_attention`、`additive_fusion` 三个替换臂与用户目标对应；必须按现行14(g)在两语料三seed验证，不能用单seed掉点提前宣布模块成立。`raw_verdict` 同时改变硬/软分数、输入干预及熵，不能单独证明交互分解的贡献。

为使模块1及整体主张可解释，建议在已列消融之外记录：`full_input_only`（保留相同任务的全输入连续logit及熵，其余干预通道置零），区分连续置信度收益与干预收益；`four_logits`（相同四次观测保留原始logits与熵，不做差分），区分新输入与可逆坐标变化。这些是主张归因建议，不新增训练前阻断门。`no_interaction` 仍有用，但要写清究竟去哪个通道且不能从其他保留通道精确还原它。

模块2必须保留相同输入的普通注意力替换。模块3必须保留相同输入/骨干/训练的简单融合替换；标准Dempster归一化替换可解释“冲突转未知”是否必要，不把单纯去掉整个分支当算子创新证据。最强baseline+相同新输入按14(f)补齐。

整体paradigm目前仅为候选主张：需要证明这三个具体改变在共同训练任务中有作用，并与RAMF/MATCH/CLARA明确区别；三个已有操作串联不能仅因命名统一就算新范式。上面的对照与已有规则足够决定后续，不恢复旧mechanism/premise门。

仅搜索lr/dropout/max_seqlen确实减少了搜索维度，但固定K30/K4、每窗帧数、top-k divisor、hidden/head、epoch、HMM设置及等权损失仍是设计超参数。固定不等于没有；报告总设计选择和搜索维度，不claim parameter-free。可学习融合尺度是模型参数，需与人工超参区分。

结论：GO进入实现；本次不运行训练，不修改CLAUDE.md或研究规则。最终三模块novelty和整体paradigm仍未被证实。
