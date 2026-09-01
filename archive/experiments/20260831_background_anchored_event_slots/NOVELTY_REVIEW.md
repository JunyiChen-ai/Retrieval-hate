# Background-Anchored Event-Slot MIL：独立查新与机制审查

截至 2026-08-31。审查对象：本目录 `README.md`。本轮只做查新与数学审查；未实现、未训练、未生成 prediction。

## 裁定

**STOP**  
**novelty：6.0 / 10**

未发现 Slot Attention / event slots 已用于 hateful-video detection/localization，跨任务 source 可以使用；但当前
adaptation 未达到可实现门槛：

1. SlotSPE 已直接占用“用 Slot Attention 把 weakly labeled bag 压成 patient-specific latent events，并选择性
   激活少数 slots”；SloTTAr 已占用 slot-based variable-cardinality temporal abstractions；
2. negative-only background representation 是 normal-prototype / background-memory 路线的直接变体；
3. 从 assignment 读 attention heatmap / frame marginal 是 attention MIL 的常规 localization readout；
4. 更关键的是，当前 positive bag loss配合 activation penalty 的最容易解是 **只激活一个 event slot**。bag label
   没有提供第二、第三个 slot 必须存在或分工的证据，故模型并未解决声称的 multi-event bottleneck；
5. `s_t` 没有被用于定义 bag probability，因而“训练与推理共用同一 marginal”并不成立；
6. proposed temporal permutation control若在 temporal encoder之后打乱 tokens，按 Slot Attention 的输入置换
   不变性应与 core 完全等价，不是 negative control。

所以它目前是 **SlotSPE-style selective event slots + normal anchor + attention-MIL heatmap** 的组件拼接。任务
故事直指多段 hate event，但训练目标没有识别该故事。即使最终 test 提升，也只能说明 slot bottleneck / normal
contrast 有用，不能说明发现多个事件或纠正了 single-witness shortcut。

## 最近邻与占位边界

### Slot Attention

Locatello et al. 提出 exchangeable slots、对 inputs 的竞争式迭代 attention，以及 input-order permutation
invariance / slot-order equivariance。其 object decomposition实验有 reconstruction或 set-property supervision，
不是只靠一个 binary bag label：

- Locatello et al., *Object-Centric Learning with Slot Attention*, NeurIPS 2020，
  [arXiv:2006.15055](https://arxiv.org/abs/2006.15055)。

候选可复用该模块，但竞争 attention 本身不保证 slots 对应 objects/events；是否分解取决于 objective。当前
README 去掉 reconstruction 后，只剩 bag classification、anchor margin和 activation sparsity，监督显著更弱。

### SlotSPE：最直接的新颖性威胁

SlotSPE 把 WSI patches与omics作为 weakly supervised bags，用 Slot Attention 动态实例化 patient-specific
latent prognostic-event slots，并用 MoE-style selective slot activation减少冗余、只激活少数 predictive slots；
同时用 cross-modal reconstruction / biological prior增强 slot structure：

- Zhang et al., *Structural Prognostic Event Modeling for Multimodal Cancer Survival Analysis*, ICLR 2026，
  [arXiv:2512.01116](https://arxiv.org/abs/2512.01116)，
  [作者机构版本](https://repository.kaust.edu.sa/items/bf09a959-46b8-43cf-bee6-3e24d8d4c34e)。

因此下列宽 claim 已被占用：slots as latent events in weakly labeled bags、patient/sample-specific event slots、
selective/emptyable slot activation、sparse predictive event decomposition。hard-concrete 替代 MoE gate不是新机制。

本候选剩余差异只有：binary MIL OR、negative-train-only temporal background anchor，以及 assignment到唯一 frame
score 的 readout。这可以构成任务 adaptation，但必须证明它们共同改变了可识别的训练机制；当前尚未做到。

### SloTTAr：temporal slots 与可变数量已有

SloTTAr 把 Transformer 与 Slot Attention结合，从 state-action trajectories无监督发现 temporal sub-routines，
并用 adaptive computation学习每条 trajectory 的 sub-routine数量：

- Gopalakrishnan et al., *Unsupervised Learning of Temporal Abstractions with Slot-based Transformers*, Neural
  Computation 2022，[arXiv:2203.13573](https://arxiv.org/abs/2203.13573)。

它的 sub-routines 是有边界的时序抽象，候选 slots 允许不连续 seconds且用于 binary localization，两者不等价；
但“首次 temporal event slots”与“首次 unknown number temporal slots”均不可 claim。

### PRSA-Net：slot 已用于 temporal action proposal

PRSA-Net 用 region-based slot attention和多尺度局部 context增强 snippet representations，再预测 action
boundaries与proposal confidence：

- Li et al., *Pyramid Region-based Slot Attention Network for Temporal Action Proposal Generation*, BMVC 2022，
  [arXiv:2206.10095](https://arxiv.org/abs/2206.10095)。

它是 fully supervised proposal-generation context module，不是 exchangeable latent event-set MIL；因此不直接
占用候选公式。但它已占用“把 Slot Attention用于 temporal action localization/proposal”这一宽表述。

### AUMN：弱监督时序中的多 action units 已有

AUMN 在只有 video labels 的 WTAL 中学习 action-unit memory bank，用 attention更新 memory和 unit-specific
classifiers，并加入 diversity、homogeneity、sparsity约束：

- Luo et al., *Action Unit Memory Network for Weakly Supervised Temporal Action Localization*, CVPR 2021，
  [官方论文页](https://openaccess.thecvf.com/content/CVPR2021/html/Luo_Action_Unit_Memory_Network_for_Weakly_Supervised_Temporal_Action_Localization_CVPR_2021_paper.html)。

AUMN 的 units 是跨视频 memory templates，不是每视频动态 slots，且仍阈值/NMS出 proposals；但“弱监督下用
多个 latent action units改善 completeness/background separation”已被占用。

### WSI Slot-MIL / attention MIL

WSI literature 已广泛使用 multiple attention heads / latent tokens把 patches压成 bag representation，并把
attention maps解释为 localization；SlotSPE是其中最直接的 slot-event先例。更基础的 attention MIL已明确用
normalized instance weights产生 bag representation和 instance contribution heatmap：

- Ilse et al., *Attention-based Deep Multiple Instance Learning*, ICML 2018，
  [PMLR](https://proceedings.mlr.press/v80/ilse18a.html)。

因此 direct assignment-to-frame readout是合规设计，但单独不构成 novelty。

### WS-TAL / WSVAD background与normal prototype

- BaS-Net 已在 WTAL 中显式引入 background class和非对称两分支训练以抑制 background frames：
  [AAAI 2020 paper](https://ojs.aaai.org/index.php/AAAI/article/view/6793)。
- Nguyen et al. 已在 video-level weak supervision下显式学习 foreground/background attention models：
  [ICCV 2019 author paper](https://ics.uci.edu/~fowlkes/papers/nguyen_weakaction_iccv2019.pdf)。
- NG-MIL 从 noise-free normal videos学习多 normal prototypes，再以相似度引导 WSVAD MIL：
  [WACV 2023 paper](https://openaccess.thecvf.com/content/WACV2023/html/Park_Normality_Guided_Multiple_Instance_Learning_for_Weakly_Supervised_Video_Anomaly_WACV_2023_paper.html)。
- UR-DMU 用 normal/anomaly memory units与 uncertainty regulation建模 WSVAD：
  [arXiv:2302.05160](https://arxiv.org/abs/2302.05160)。

所以 negative-only EMA background anchor不是新的 normality modeling。它与 Slot Attention竞争结合是具体组合，
但必须证明 background assignment接受有效训练且不仅是一个 ordinary corpus normal centroid。

### Hateful video

检查 MultiHateLoc、LELA、TANDEM、CLARA、MM-HSD 与 HateClipSeg，未发现 slot/event-slot或 negative-only
background slot用于 hateful-video temporal localization。代表性工作：

- MultiHateLoc，[arXiv:2512.10408](https://arxiv.org/abs/2512.10408)；
- LELA，[arXiv:2602.09637](https://arxiv.org/abs/2602.09637)；
- TANDEM，[arXiv:2601.11178](https://arxiv.org/abs/2601.11178)；
- CLARA，[arXiv:2608.15905](https://arxiv.org/abs/2608.15905)；
- MM-HSD，[DOI](https://doi.org/10.1145/3746027.3754558)；
- HateClipSeg，[arXiv:2508.01712](https://arxiv.org/abs/2508.01712)。

目标领域未直接占用，因此 STOP 来自 adaptation 与可识别性，不是“source来自外部”本身。

## 数学与可识别性审查

### 1. Bag objective系统性偏向单 slot

positive bag只要求 `noisy_or_k(a_k h_k)` 为正，同时对 `sum_k a_k` 加 activation penalty。假设一个 slot 已能把
bag score推到正确范围，则再打开任意第二个 slot：

- 对 positive BCE 至多带来很小收益；
- 一定增加 activation penalty；
- 还要在 competitive attention 中与第一个 slot争 tokens。

因此最经济解是一个 active event slot，其余 slots取 null。negative bag又要求全部event slots不激活，进一步
训练 gate倾向关闭。这个 objective 与 `K=1` 在机制上同向，而不是迫使多个 slots解释不同事件。

多 transition视频仍只有一个 video label；相同 hate concept可由一个非连续 slot吸收所有 hate-like tokens，甚至
一个 whole-video/topic slot即可分类。bag labels本身无法识别事件数量、事件边界或 slot-to-event一一对应。slot
exchangeability只允许无意义的 slot permutation，不解决 cardinality/partition identifiability。

`有效event-slot数不全为1或4`是事后使用量统计，不是识别条件。模型可以因随机初始化开 2 个冗余 slots而过门，
也可以正确地让一个语义 slot覆盖多个不连续 hate spans却被误判失败。

### 2. Bag score与最终 frame marginal脱节

README 定义 bag score只用 `a_k h_k`，frame score才用：

`s_t = sum_k assignment(t,k) a_k sigmoid(h_k)`。

bag loss没有由 `s_t` 聚合得到。assignment会间接影响 slot update与 `h_k`，但以下退化仍成立：

- 一个 slot从全视频弱平均/场景topic得到高 `h_k`；
- 其 assignment可接近uniform或覆盖整段；
- bag classification完全正确，但 `s_t`近常数。

因此“assignment直接参与bag loss，保证机制改变 within ranking”并不成立。若要训练/推理一致，bag probability
必须由同一个 `s_t` 确定，例如固定 noisy-OR over time，或有严格等价推导；不能另设 slot-level classifier捷径。

### 3. Background slot并未被迫解释 benign seconds

EMA `b`只由 negative train tokens更新，能得到一个 normal reference，但：

- anchor margin只约束 event-slot vector 与 `b` 的距离，不监督每个negative second分给 background；
- negative bag可通过 `a_k=0`满足 loss，此时 event assignment对最终 score被乘零，assignment梯度可能很弱；
- positive bag中的 benign seconds没有标签，固定 background query可在竞争中输给一个高范数event query；
- 单个 EMA均值无法表示多模态、多场景 normal distribution，这正是 normal-prototype路线的已知容量问题。

所以“background slot必须解释全部 benign seconds”目前只是叙述。需要在 negative train videos上显式强制
`assignment(t,b)`，并用与多 normal prototypes容量匹配的 control；否则 anchor等价于 margin中的普通 centroid。

### 4. hard-concrete null gate不解决 slot semantics

hard-concrete能让 slots为空，也能产生稀疏梯度，但它只识别“是否用计算单元”，不识别不同 event。SlotSPE 已有
selective activation；把其 gate换成 hard-concrete属于实现选择。若没有 reconstruction、set supervision、
independent views或其他 slot-specific约束，unused-slot与all-events-in-one-slot都是合法全局解。

### 5. 不输入时间位置与 multi-event story 的关系

不输入显式 normalized position可减少按片头/片尾固定分工，这是合理的。但 Slot Attention此时把 encoder tokens
当作集合；slot outputs不保留事件先后，noncontiguous assignment是自然结果。它可能按语义/topic聚类，而不是按
event instance聚类。共享 temporal encoder可以先编码local context，但 bag loss没有告诉 slots 哪种聚类对应
真实 hate boundaries。

## K=4 审查

根据 developmental test GT选择 K 不违反本项目规则，前提是明确这是 test-informed method development；但当前
依据不支持 `K=4` 的机制解释：

1. HCS有 4–12 transitions不等于有 4 个 hate events；一次事件通常产生两个 transitions；
2. slots被定义为可非连续的语义 event groups，slot数量本来就不应等于 temporal component数量；
3. HateMM与其余主语料没有给出同样的 capacity依据；
4. emptyable slots使 K 只是上限，但 objective又偏好 K=1，固定 4不能解决 collapse。

因此可以把 K=4称为固定 capacity ceiling，不能称由事件数识别出的合理 K。若不允许扫描 K，至少先做 synthetic
train-only recovery：已知 1/2/4 个可分 event concepts时，模型能否恢复有效 slot数；否则 K control没有解释力。

## Temporal permutation control 无效

README 的“打乱 seconds 后恢复原 index”必须区分打乱位置：

- 若在 temporal encoder **之后**置换 `x_t`，Slot Attention对 input order不变，slot outputs相同，assignments
  只随 tokens置换；恢复 index 后 `s_t`理论上应与 core相同。这应是 equivariance单元测试，不得要求它落后。
- 若在 temporal encoder **之前**打乱 raw seconds，破坏的是 temporal encoder的local neighborhoods；该 control
  检验 encoder是否用时序上下文，不检验 slots是否发现多个 events。
- 若使用每个视频一个固定 permutation，模型还可能学习 corpus-level permutation artifacts；必须明确 train/test
  同一变换规则与随机性，且不得读取 GT。

所以“temporal permutation不得追平core”当前不是有效机制 gate。正确做法是保留 after-encoder permutation作为
必须数值等价的 positive control，另设 before-encoder permutation/no-temporal-encoder作为 temporal-context
ablation，不能把它归因到 slot decomposition。

## 当前 controls/gates 不足

已有 capacity transformer、K=1、no-anchor、no-gate arms是必要但不充分。缺少：

1. **SlotSPE port control**：相同输入与binary bag loss的 vanilla slots + selective activation；否则无法证明
   task adaptation超出 patch-to-second替换。
2. **normal-prototype control**：同一 negative EMA/reference直接接 frame scorer或attention MIL，不使用 slots；
   以及 NG-MIL式多 normal prototypes，排除只是 normal distance有效。
3. **same-marginal control**：bag loss直接由 `s_t`聚合的 K=1/K=4；排除 slot-level classifier捷径。
4. **single active slot forced control**：K=4但每 bag只允许一个 active slot；若追平，multi-slot story失败。
5. **slot duplication/tied-slot control**：检查多个 active slots是否只是重复同一 assignment与logit。
6. **gate-free matched sparsity control**：普通 attention MIL使用相同 activation budget，排除 hard-concrete sparsity。
7. **anchor assignment control**：negative train seconds的 background assignment、event leakage与gradient必须报告；
   只报告 EMA distance不够。
8. **effective-rank/overlap diagnostics**：pairwise assignment overlap、slot contribution deletion、每 slot独有 frame
   mass；`1 < active_count < K`不能证明分工。
9. **equivariance test与temporal ablation分离**，如上一节。

test `>=4 transitions`分层可作为 developmental error analysis，但不能弥补 train objective不可识别；若 core只在
该 test subgroup赢，也不能作为 routing或选择规则。

## 若要重提，必须修改

1. **删除当前 multi-event可识别性表述。** 在只有 binary bag labels时，不能声称 slots对应不同真实 events。
2. **让 bag probability严格来自唯一 frame marginal `s_t`**，并移除绕过 assignment的独立 slot-level shortcut。
3. **negative assignment supervision**：negative train seconds必须直接监督 background assignment / zero event
   marginal，而不只是更新 EMA和关闭 gates。
4. **提供一个真正反对 K=1 的 objective或证据。** 若仍只有 noisy-OR + activation penalty，数学方向就是 single
   witness，应 STOP；仅加 diversity penalty又会回到任意拆分，不能解决 event truth。
5. 用 train-only synthetic recovery明确可恢复的对象是 semantic modes、temporal components还是多个 independent
   evidences；三者不能混称 event slots。
6. 增加 SlotSPE、attention MIL、single/multi normal prototype和强制单-active-slot controls。
7. 修正 temporal permutation：after-encoder permutation必须相等；before-encoder shuffle只能称 temporal encoder
   ablation。
8. K=4只称固定 capacity ceiling并记录来自已查看 test artifact；不得把 transitions直接换算成 slots。
9. 四主语料仍各自独立训练，validation只选固定 arm checkpoint，训练完成后立即 test三指标；test-informed
   结果标 developmental。禁止 ensemble、calibration、routing与跨主数据集训练。

第 4 项是根本 blocker。没有新增能让多个 slots对最终 frame marginal提供不可替代证据的训练约束，不建议只补
controls后实现当前版本。

## 允许与禁止的 claim

当前 STOP 状态不应形成方法 claim。若以后修复训练闭环并通过强 controls，最多允许：

> To our knowledge, we adapt selectively activated latent slots to weakly supervised hateful-video temporal
> localization by anchoring a background assignment with negative-train videos and deriving both bag supervision
> and a single frame output from the same slot-assignment marginal.

这仍只能称 task adaptation。不得 claim 首次 Slot Attention、首次 temporal slots、首次 latent event MIL、首次
unknown-cardinality slots、首次 background modeling、首次 normal prototype或首次 multi-event localization。
在没有额外 instance supervision / identifiability证明时，不得称真实 event discovery、event count recovery、
slot interpretability、slot-to-event correspondence或因果 hate event decomposition。

## 最终理由

按最新标准，外部 SlotSPE source没有用于 hateful-video这一点成立，negative-only anchor与single-frame readout也
具有任务针对性，所以 novelty不是零；但 SlotSPE、SloTTAr、AUMN、normal-prototype与attention-MIL已覆盖几乎
所有组件。当前唯一应当新增科学内容的“多 event slots缓解 single witness”恰好被 objective反向抑制，frame
marginal又没有闭合到 bag loss。故裁定 **STOP，6.0/10**。
