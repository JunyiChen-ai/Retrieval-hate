# Independent narrow novelty review

截至 2026-09-01。依据 RESET2 后修订的 Rule 12，只审查来源占用、是否直接套用/组件拼接、是否与 failure ledger 已关闭机制严格同构且无新增约束、以及来源必要监督是否明确缺失。不审查代码，不要求实现前完整排除一般 shortcut。

## 最终裁定

**GO，novelty 6.9/10。**

| Gate | 裁定 | 理由 |
|---|---|---|
| Gate 1：允许 adaptation 已有来源 | **PASS** | Expert Choice routing 是明确、可迁移的跨任务 routing机制。 |
| Gate 2：来源核心未进入 hateful-video detection/localization | **PASS** | Hateful-video 已有 SAGE、CLARA、MoRE 和 MultiHateLoc 等 expert/fusion方法，但检索未发现 Expert Choice 的“expert沿token维取固定capacity、token接收可变数量experts”被用于 hateful-video detection/localization。 |
| Gate 3：non-trivial、task-specific adaptation | **PASS** | 候选把不可交换的 modality branches定义为experts、seconds定义为tokens，让每个模态选择其时间witness；同时删除 per-branch positive-label broadcast与独立 fused bypass，使selection mask和local evidence直接组成唯一frame score。这不只是把二模态或普通MoE扩成三模态。 |

该 GO 只批准 README 中冻结的最小 HMM/HCS end-to-end pilot，不预判性能，不授权增加新的 routing、loss 或监督机制。

## Gate 1：来源可 adaptation

Primary source：[Zhou et al., *Mixture-of-Experts with Expert Choice Routing*, NeurIPS 2022](https://papers.nips.cc/paper_files/paper/2022/hash/2f00ecd787b432c1d36f3de9800728eb-Abstract-Conference.html)。

来源核心不是一般 MoE，而是 dispatch 方向的反转：

- conventional token-choice让每个token选固定数量experts；
- Expert Choice让每个expert沿token维选择固定capacity的tokens；
- 因而expert负载得到保证，每个token可被0个、1个或多个experts处理；
- routing affinity在被选连接上保持可微。

这些性质可以迁移到固定时长视频token集合与固定模态expert，Gate 1 PASS。

## Gate 2：目标任务占用

### 精确来源检索

检索覆盖：

- `Expert Choice Routing`；
- `Mixture-of-Experts with Expert Choice Routing`；
- `expert selects tokens`、`variable experts per token`；
- 上述术语与 hateful video detection、hate video localization、temporal hate localization 的组合。

未发现 Expert Choice routing 的精确核心用于 hateful-video detection/localization。

### Hateful-video 最近邻

#### SAGE

[SAGE, ACL 2026](https://aclanthology.org/2026.acl-long.817/) 已在 hateful-video detection 中提出 disentangled modality experts、global deliberation与instance-level evidentiary arbitration。它直接占用以下宽 claim：

- modality-specific experts保留稀疏 hate cues；
- adaptive expert gating优于blind fusion；
- dominant benign modality会稀释局部 hateful evidence。

但是 SAGE 的目标是video-level verdict；其gate在instance/video层选择experts，并保留expert prediction supervision。没有证据表明它让每个modality expert从视频时间轴反向选择固定capacity的seconds，也不输出由这些assignments直接形成的单一1fps localizer。

#### CLARA、MoRE

CLARA使用clip-level generic MoE，由每个clip/token选择若干可交换processing experts，并用load-balancing loss防止expert under-use。MoRE使用modality experts与sample-sensitive integration做short-video hate detection。二者进一步占用了普通 `MoE + hateful video` 和 adaptive multimodal integration，但不是当前 temporal Expert Choice dispatch。

#### MultiHateLoc

[MultiHateLoc](https://arxiv.org/abs/2512.10408) 已有 modality-aware temporal encoders、dynamic cross-modal fusion、cross-modal contrast和 modality-aware MIL。它占用动态时间融合的宽 claim。当前候选的差异不是再加一个soft modality weight，而是反转责任分配方向：每个固定modality expert竞争本视频的时间tokens，并允许一个second接收0–3个模态。

因此 Gate 2 PASS，但允许的贡献必须保持窄；不能claim modality experts、adaptive fusion、MoE、sparse evidence arbitration或 temporal MIL本身新。

## Gate 3：为什么不是简单 MoE 移植

### 1. Expert identity 从可交换计算单元变成观测语义

来源中的experts主要是可交换FFN容量；某个expert编号本身没有audio、visual或text语义。当前adaptation把三个expert固定绑定到不可交换的观测通道。某个expert选择某秒，不只是节省计算，而是表示该模态愿意为该时间token提供evidence。

这把source的load-balancing primitive改造成弱监督 temporal responsibility结构。

### 2. Dispatch方向直接对应已证实 failure

项目test evidence显示video-global DMS几乎总偏向visual，并且与test-GT最佳模态匹配很差。Token-choice top-1仍要求每秒强制挑一个winner；video-global routing又可能整段使用同一模态。

Expert Choice改为：

- 每个模态必须在自己的时间轴上挑选有限witness tokens；
- 同一秒可同时获得多个模态，保留协同；
- 无expert选择的秒可保持background，不强迫伪owner。

这是针对 `time × modality` responsibility错配的具体任务改造，不是只把source的expert数改成3。

### 3. 删除 per-branch label broadcast 是 load-bearing delta

SAGE与许多 multimodal expert方法会独立监督各expert，MultiHateLoc也对modality branches传播同一positive video label。当前候选删除三个per-branch MIL：

- positive label只约束三个experts经routing后的联合frame score；
- negative label通过同一个final MIL压低被选evidence；
- 原video-global DMS与独立fused head被删除；
- router mask和expert logits本身就是唯一frame score的组成项。

因此它不是在原网络旁边添加一个可忽略router，也不是把branch outputs在test再做选择。训练语义从“三个branch各自解释整个positive bag”改成“固定容量的多个modality experts共同解释时间witness”。

### 4. 必要监督条件存在

该机制需要的条件为：

- 本语料video-level positive/negative label；
- 同步到1fps的三个modality streams及availability；
- 每个视频的有限temporal token集合；
- 一个预先固定的expert capacity budget。

这些条件均存在。它不需要span label、owner label、其他主数据集train set、teacher或test oracle。来源的专家可交换性也不是Expert Choice算法成立的必要监督条件；固定语义expert仍可沿token维做top-k dispatch。

### 5. 与 ledger 不严格同构

- 不是flat joint witness ownership：没有categorical owner pseudo-label或每秒恰选一个模态；一个token可有0–3个experts。
- 不是single-carrier branch dominance的原样重写：每个expert有显式temporal capacity，且原独立fused bypass被删除。
- 不是ensemble/calibration：只有一个联合训练模型和一个raw frame score。
- 不是auxiliary bypass或direct-head replacement：routing及evidence sum就是final scorer。
- 不是typed RMoE：typed RMoE是在每个模态内部选择temporal-scale experts，再以普通gate/fusion汇合；当前结构是在模态与时间之间建立Expert Choice assignment。

仍可能出现错误selection，但属于端到端test风险，不构成“严格同构且无新增约束”。

## Internal routing 与禁止的 test routing

该方法在inference时确实执行learned Expert Choice router，不能表述为“完全没有routing”。但它不是项目禁止的oracle/per-corpus method routing：

- 不读取test label或GT best branch；
- 不在多个已训练方法间选择；
- 不把branch score作为单独输出后再ensemble；
- router是预先定义的单模型forward的一部分，最终只输出一个frame score。

因此它属于模型内部 sparse routing，不等于test oracle、ensemble或post-hoc calibration。

## 登记为 test/technical 风险，不作为 pre-run STOP

1. 固定per-expert quota会强制弱或缺失模态选择tokens；availability必须在实现中正确mask。
2. 选中expert数量本身可能提高frame logit，形成assignment-count shortcut；是否归一化属于技术审查项。
3. 每个expert固定选择 `ceil(T/K)` 秒隐含occupancy budget，可能不适合长hate或无证据模态。
4. hard top-k可能使未选tokens缺少routing梯度，或在早期训练锁定错误位置。
5. 所有experts可能选择同一批容易的topic/position seconds，仍未必对应hate GT。
6. 无expert选择的秒只得到shared background，可能漏掉capacity外的长事件。

RESET2 明确规定这些一般shortcut交给最小HMM/HCS test及matched control，不要求pre-run theorem。

## Matched control 与机制判据

README 的 token-choice top-1 control能够检验最核心的dispatch方向：token选expert对比expert选token。但正式实现前必须保证两者的总active expert-token assignments与计算量确实匹配；否则routing方向会与activation budget混杂。这是技术审查要求，不改变当前novelty verdict。

另一个需要澄清的归因点是：Expert Choice按定义让每个expert拥有固定capacity，因此“visual占全部selected assignments比例下降”几乎由结构保证，不能单独作为学到正确ownership的证据。机制判断应依赖：

- core相对token-choice和MultiHateLoc的test within提升；
- selected expert evidence对最终score的实际贡献，而不只是assignment计数；
- overlap、unassigned-token与各模态selected evidence的分布。

这些属于通过novelty后的technical review与test attribution，不构成额外pre-run novelty gate。

## 允许的窄 claim

> 将NeurIPS 2022 Expert Choice的反向dispatch改造成弱监督 hateful-video temporal responsibility：固定audio/visual/text experts各自选择时间tokens，每秒接收可变数量模态；删除逐branch positive-label supervision与独立fused bypass，使assignments直接形成唯一frame score。

不允许claim：

- 首次在hateful-video使用modality experts或adaptive gating；
- 首次处理sparse modality-specific hate cues；
- 首次使用MoE、MIL或temporal routing；
- assignments等于真实modality ownership；
- 模型没有inference routing。

## 结论

Expert Choice的精确routing核心未检出进入 hateful-video detection/localization。SAGE、CLARA、MoRE与MultiHateLoc占用了宽泛的expert/fusion叙事，但没有占用“固定modality expert沿时间选择tokens、token接收0–3模态、删除per-branch label broadcast并直接形成唯一frame score”的完整训练语义。

该adaptation具备明确的target-task failure、load-bearing final-score路径、必要监督和可归因token-choice control；它不是简单MoE三模态端口，也不与ledger旧机制严格同构。按RESET2 Rule 12，裁定 **GO，6.9/10**。下一步应停止追加novelty审查，进入最小实现、一次technical review、HMM/HCS独立训练后立即test。
