# Owner-Abstaining ItS2CLR：独立 novelty / mechanism review

截至 2026-08-31。审查对象：本目录 `README.md`。本审查只检查机制与最近先例；未实现、未训练、未运行任何 prediction。

## 裁定

**GO，novelty 6.3/10，但只允许一个很窄的 adaptation claim。**

检索到的 primary literature 中，没有发现以下合取已用于 hateful video detection/localization，也没有发现它作为一个完整训练机制用于相邻的弱监督时序任务：

> 在仅有 video label 的 hateful-video temporal MIL 中，把每秒展开为 modality-specific instances，用该秒逐模态删除造成的 fused-logit 下降选择“deletion-sensitive evidence carrier”，把未获支持的正包模态实例从 supervised-contrastive relation 中排除，而不是当作 background，并在推理时仍输出一个原始 fused frame score。

这不是把 ItS2CLR 的输入 feature 换成视频 feature：它改变了 pseudo-instance 的索引空间、正负关系和哪些实例有资格进入 SupCon，且直接针对“video label 被广播给无证据 modality”这一任务错误。因此按当前允许跨任务 adaptation 的标准，可以做最小 pilot。

但 **`owner` 不是由该方法可识别的语义**。`z_t-z_t^{-m}` 最多是 seed model 在指定遮蔽操作下的 deletion sensitivity，不能证明 modality 真正“拥有”hate evidence，更不能称为 causal ownership。若坚持宽 claim（first modality-aware contrastive hate localization、first evidence arbitration、causal owner discovery），本裁定立即变为 **STOP**。

## 直接回答四个查新问题

### 1. ItS2CLR 或等价核心是否已经用于 hateful video

未找到直接使用。Liu et al. 的 [ItS2CLR, CVPR 2023](https://openaccess.thecvf.com/content/CVPR2023/html/Liu_Multiple_Instance_Learning_via_Iterative_Self-Paced_Supervised_Contrastive_Learning_CVPR_2023_paper.html) 在 Camelyon16、breast ultrasound 和 TCGA-LUAD 上验证，任务是 medical MIL。其核心已经包括：MIL instance probability 二值化、正包内高置信正/负实例的 self-paced 选择、未选实例不进入 SupCon，以及迭代更新 representation 和 pseudo label。因此，本候选不能 claim self-paced MIL、pseudo-instance SupCon、忽略低置信实例或 ItS2CLR 本身的新颖性。

截至检索日，目标领域中已存在 supervised / segment contrastive learning，但没有找到把 ItS2CLR 的完整 iterative self-paced procedure 用于 hateful video localization 的论文。是否采用原论文的工程训练循环不是主要 novelty；真正需要归因的是下面的 modality-conditioned relation。

### 2. 是否已有 per-time, per-modality deletion-owned ternary pseudo label，并让 non-owner 在 SupCon 中 abstain

未发现 exact precedent。最接近的工作分别占据这个机制的不同部分：

- Cheng et al. 的 [JoMoLD, ECCV 2022](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136940424.pdf) 已明确提出：video-level event label 不一定出现在每个 modality，因而产生 modality-specific noisy labels；它按 audio/visual loss inconsistency 动态删除 modality label。JoMoLD 是最强的“问题定义”先例，但它在 video/category level 按 batch loss 排序，把被删正标签改为 0 后用于 BCE；它没有逐秒删除干预、没有三模态局部 pseudo owner，也没有用 abstention 改写 SupCon pair graph。论文还明确说明其附加 contrastive loss不参与 label denoising。
- Yu et al. 的 [MACIL-SD](https://arxiv.org/abs/2207.05500) 已在 weakly supervised audio-visual violence detection 中处理 modality asynchrony：从逐模态 snippet logit 聚类 violent、normal、background semi-bags，并构造 modality-aware contrastive pairs。这比一般 WTAL 更近。它仍使用 top/bottom clustering，并把跨模态 violent semi-bags组成 positive pairs；没有逐模态删除效应，也没有把未证明的 modality 作为 contrastive abstention。
- NeurIPS 2023 的 [Language-guided Segment-level Label Denoising](https://papers.neurips.cc/paper_files/paper/2023/file/7fbae0a0885d3d688840bd34e4a8a698-Paper-Conference.pdf) 已在 weakly supervised AVVP 中产生 per-segment, per-modality event labels，并对 unreliable segments 动态降权；来源是 CLIP/CLAP prompt similarity，不是本模型局部删除，也不是 SupCon 三态关系。
- Li et al. 的 [DCC, CVPR 2022](https://openaccess.thecvf.com/content/CVPR2022/papers/Li_Exploring_Denoised_Cross-Video_Contrast_for_Weakly-Supervised_Temporal_Action_Localization_CVPR_2022_paper.pdf) 已占据 temporal pseudo-label denoising、region-level memory 和 action/background contrast。Zhou et al. 的 [Delta pseudo label, CVPR 2023](https://openaccess.thecvf.com/content/CVPR2023/papers/Zhou_Improving_Weakly_Supervised_Temporal_Action_Localization_by_Bridging_Train-Test_Gap_CVPR_2023_paper.pdf) 用连续轮次 pseudo labels 的差分实现自纠正。两者都不是 modality deletion attribution，也不定义 non-owner abstention。
- Li et al. 的 [Selective-Supervised Contrastive Learning, CVPR 2022](https://openaccess.thecvf.com/content/CVPR2022/papers/Li_Selective-Supervised_Contrastive_Learning_With_Noisy_Labels_CVPR_2022_paper.pdf) 已占据“只让高置信样本/关系进入 SupCon”这一通用思想。因此 `abstain` 或 selective pairs 单独不可主张。

结论是：各部件都有强先例，但“同秒逐模态 deletion sensitivity 决定哪一个 modality-instance 可成为 hate positive，其余不被改写成 negative”的关系合取仍有残余空间。

### 3. 与目标领域最近工作的边界

| 工作 | 已占据的内容 | 本候选只能主张的差异 |
|---|---|---|
| [MultiHateLoc](https://arxiv.org/abs/2512.10408) | 弱监督 hateful temporal localization、modality-aware temporal encoder、dynamic fusion、cross-modal contrast、modality-aware MIL | 训练时 deletion-sensitive modality-instance relation；不能 claim modality-aware fusion、MIL 或 temporal hate localization |
| [ImpliHateVid, ACL 2025](https://aclanthology.org/2025.acl-long.842/) | hateful video 的 modality-specific encoder、两阶段 supervised contrastive 和 cross-modal representation refinement | video label 不直接定义 local pair；按秒、按 modality 选择或 abstain pseudo relation |
| [CLARA](https://arxiv.org/abs/2608.15905) | clip MoE、local-global segment contrast、VLM rationale、hateful video detection | 不使用 rationale/MoE/local-global contrast；只在训练期重写局部 SupCon relation |
| [SAGE, ACL 2026](https://aclanthology.org/2026.acl-long.817/) | “benign modality 稀释 sparse hateful cue”的问题叙事、独立 modality experts、instance-level evidentiary gating | 不能 claim feature dilution、evidence arbitration 或 modality prioritization；仅 claim弱监督时序 pseudo-relation 的构造方式 |
| [MoRE, WWW 2025](https://jianlang.org/assets/papers/WWW-2025-MoRE.pdf) | modality experts、sample-sensitive modality integration、hateful-video detection、retrieved context | 无 retrieval、无 test-time routing claim；差异只在逐秒训练监督语义 |
| JoMoLD | modality-specific weak-label noise 的问题与 label removal | deletion sensitivity、逐秒索引、unknown 与 background 在 SupCon 中的不同作用 |
| MACIL-SD | 逐模态 snippet bags、弱监督 violence、modality-aware instance contrast | 不构造跨模态 positive pair；未获 deletion support 的 modality 不参与 SupCon |
| DCC / Delta pseudo label | 时序 pseudo-label contrast/denoising/self-correction | modality-specific deletion relation，而非一般 pseudo-label refinement |

ImpliHateVid、CLARA、SAGE 和 MoRE 均已把 broad hateful-video contrast / expert / modality contribution 叙事占满。论文贡献不能写成这些宽表述。POWA、LELA 等输出或提示 modality-specific evidence 的方法同样不改变上述边界；“解释哪个 modality 重要”不等于本候选的训练关系，但也意味着本候选不能 claim 首次发现 modality evidence。

### 4. 是否只是组件拼接

**不是纯 feature substitution，但处于“可守的窄 adaptation”和“组件拼接”的边缘。**

支持 GO 的理由是，deletion state不是附加 feature或另一个并列 loss；它直接决定 SupCon 的 adjacency / exclusion mask。`unknown` 与 `background` 的代数作用不同，且这一差异正对应目标任务里“未在某模态证明 hate”不等于“该模态证明 benign”。这是一个统一的监督语义修改。

降低 novelty 的理由是：ItS2CLR 已有 self-paced exclusion，JoMoLD 已有 modality-specific positive-label denoising，MACIL-SD 已有 per-modality temporal contrast，而 leave-one-modality-out 是常见 attribution 操作。若实验只显示“加一个 SupCon loss有提升”，却没有证明提升来自 deletion-based relation而非更多 projection heads、更多训练或 selective sampling，这个工作会退化成已知组件拼接。

## 致命机制问题

### A. 删除差分不能识别 owner

对 observational binary bag labels，真实 modality ownership 一般不可识别：

- 两个模态提供冗余证据时，删掉任一个都可能几乎不改变 logit，真实 carrier 会被标为 abstain；
- 两模态只有联合出现才构成 hate 时，分别删除都可能明显降分，两个都会被称为 owner，但模型无法区分 synergy 与独立证据；
- 某模态同时含 hate cue 和 suppressor/confounder 时，删除后 logit可能上升，`d<=0` 不代表没有 hate evidence；
- 若 `z_t` 来自 temporal encoder，遮蔽 `(t,m)` 会改变上下文与邻秒表示，差分不是严格的“该秒该模态”局部量；
- mask token、零向量或缺失模态分布若与训练分布不一致，差分主要测到 out-of-distribution corruption。

因此正文必须把 `owner` 降格为 **deletion-sensitive carrier pseudo label**。只有在多个合理 deletion baselines 下方向稳定，并通过 insertion/deletion faithfulness 检查后，才能把它解释为 model reliance；仍不能解释为因果 ownership。

### B. 正包 background 规则在冗余模态下会系统性造假负例

“低置信秒且三个 `d_{t,m}` 都不为正”并不能推出 background。冗余、多模态补偿、seed under-confidence 或不合适的 mask 都会使真实 hateful 秒满足该条件。随后将三种 modality 全拉入 background cluster，正好重现候选试图消除的 label poisoning。

这不是小超参数问题。最低限度必须把 **positive-bag low-confidence/all-nonpositive → abstain** 作为主要 control；若它优于当前 background 规则，当前规则应删除，不能继续称为保守负例挖掘。

### C. Cross-fitting 防 leakage，不解决循环确认

三折 out-of-fold prediction能避免同一视频被直接拟合后给自己打 pseudo label，但三个 fold model仍由同一任务、同一 bag labels、同一 architecture biases 学得。视觉 shortcut 或错误 fusion ownership会被复制进所有 folds。Cross-fitting是必要的数据隔离措施，不是 pseudo-label 正确性的证据，也不能作为 novelty claim。

### D. `d>0` 没有可信度含义

任意微小正数都会成为 owner，符号容易受数值噪声、augmentation、mask realization 和 seed影响。self-paced frame confidence并不校准 modality deletion effect。必须预先定义稳定性或 margin 规则，且不能根据 test 指标调它；否则“owner”比例本身成为隐含可调 top-k。

### E. SupCon head 可能与最终 fused score脱耦

候选不设跨模态 positive pair是合理的，但 modality-specific projection head可以吸收全部 SupCon约束，而共享 encoder/fusion几乎不变。这样 owner relation在训练 loss上成立，却不改变最终 frame ranking。必须报告 encoder gradient/coupling或使用去掉 projection-head自由度的 control；仅看最终性能不能证明 ownership机制生效。

## 当前 pilot controls 是否足够

**不够。** README 的四个 arms能检验 core 相对 fused-frame ItS2CLR，以及 abstain 相对强制 negative，但不能排除三类替代解释：更多 modality heads/capacity、任何 per-modality selective SupCon都有效、deletion states只是保持比例的随机正则化。

最低必要 attribution controls如下；这些 controls需与 core一同冻结后训练，并按项目规则直接做 test 三指标：

1. **Capacity-matched per-modality ItS2CLR：** 同样三个 modality projection heads、同样训练预算，用 fused high-confidence frame label广播到三模态，不使用 deletion。它比当前 vanilla fused-frame ItS2CLR 更公平。
2. **Selector control：** 用各 modality 自己的 branch logit/confidence选择 positive，保持 owner rate与 self-paced schedule接近，但不做 deletion。区分“逐模态选择”与“删除归因”。
3. **Rate-preserving shuffled-owner：** 在同 corpus、bag、时间置信层内打乱 modality owner assignment，保持每模态 owner数量。若与 core接近，deletion semantics没有贡献。
4. **Abstention control：** README已有 `abstain→negative`，保留；另加 positive-bag `all d<=0` 仍 abstain，而不是 background。前者检验 unknown≠negative，后者检验危险的 background规则。
5. **Intervention robustness：** 至少两种训练分布内可解释的 replacement（例如原生 modality-dropout mask与matched benign/reference replacement）应给出 owner agreement、符号稳定率和每模态覆盖率。只对一个零遮蔽有效不够。
6. **Projection/coupling control：** 保留同样 pseudo states但停止 SupCon对共享 encoder的梯度，或只训练 projection head。若 final frame score不变，说明 relation未进入 localizer。

Cross-fit vs in-fit可以作为 leakage诊断，但不是最关键的 novelty attribution arm。所有 threshold、schedule与mask定义应在训练前固定；validation只用于各 arm正常选 checkpoint，不用于比较这些机制。

## 最小可证伪 pilot与失败条件

HateMM + HateClipSeg、各自独立训练、seed 234可以作为第一轮，但在 GPU预算有限时，优先保留：capacity-matched per-modality ItS2CLR、core、branch-selector、shuffled-owner、`abstain→negative`、positive-bag-all-nonpositive-abstain。原 MultiHateLoc是性能锚点。

以下任一结果都否定对应机制故事：

- core不同时超过 capacity-matched per-modality ItS2CLR 的两语料 within-video ROC：该 adaptation没有显示跨语料定位价值；
- branch-selector或 shuffled-owner 与 core持平：删除归因没有可归因贡献；
- `abstain→negative` 不差于 core：unknown/background语义不是有效机制；
- 把 positive-bag all-nonpositive 秒留作 abstain反而更好：当前 background规则失败，应删除；
- owner assignment对合理 replacement高度不稳定、长期坍缩到单一 modality，或 deletion faithfulness不高于匹配随机删除：不得使用 evidence-carrier 解释；
- 只改善 pooled指标、不改善两语料 within-video ROC：与提出该机制的 temporal ownership错误不一致。

允许按项目规则使用 test prediction与test GT做上述error analysis，但结果必须写成 iterative/developmental test evidence，不能称为未揭盲确认性结论。

## 最窄可守 claim

建议只写：

> We adapt self-paced supervised-contrastive MIL to weakly supervised hateful-video localization by indexing pseudo instances jointly by time and modality. A modality instance becomes a positive contrastive carrier only when a train-only out-of-fold seed is locally deletion-sensitive to that modality; unsupported modality instances abstain from the contrastive relation rather than being relabeled as background.

明确不主张：真实/因果 modality owner、首次 modality-aware hate model、首次 hateful-video contrastive learning、首次 modality evidence arbitration、首次 temporal pseudo-label denoising、首次弱监督 per-modality localization。

## 最终意见

**GO 做 attribution-complete pilot；6.3/10。** 残余 novelty来自“deletion-sensitive per-time/per-modality selection + non-owner SupCon abstention”这个窄合取，而不是任何单独部件。当前最危险的不是查重，而是把 seed-model deletion response误写成真实 ownership，以及把正包 `all d<=0` 错当可靠 background。若团队不接受收窄 claim、补 capacity-matched / shuffled / selector controls，或仍坚持现有 background规则无需验证，则应在实现前改判 **STOP**。
