# Coalition witness：独立 novelty 与 anti-pattern 评审

**结论：CONDITIONAL GO**  
**Novelty：6.0/10（仅限下述严格版本）；若实现成若干已知 loss 的相加，则为 3.0/10，STOP**  
**评审日期：2026-08-31**  
**范围：**只查论文原文、出版社论文页和作者官方项目页；未参与设计或实现，未使用 validation performance。

## 一句话判断

没有找到已经发表的工作同时完成以下闭环：在弱视频标签下，把正视频的隐变量定义为
`(时间 t, 最小模态 coalition S)`，用同一个共享网络对所有非空模态子集前向，借助
Möbius/Harsanyi 分解把单模态 evidence 与真正依赖联合模态的 interaction atom 分开，
再令 full-modality temporal score 由这些 atom 精确重构并作为唯一 test localizer。

但是，这个闭环的每一块都有很近的先例。尤其是 I2MoE、SynIB 和 INTER 已经分别占据
“masked-modality interaction training”“直接促进 synergy”“用 Harsanyi dividend 分离
多模态交互”。因此可支持的 novelty 不是“首次用 modality coalition / masking / synergy /
minimal subset”，而只能是：**把 irreducible modality-interaction atom 变成弱监督 temporal
MIL 的显式最小 witness，并让训练隐变量和单一 full-modality localizer 是同一个可重构量。**

这个核心只有在数学上写成一个统一 latent-variable objective 时才成立。把 standard MIL、
modality dropout、Möbius regularizer 和 sparsity loss 并列相加，会被合理地判为组件拼接。

## 必须锁定的候选定义

令模态集合 `M={V,A,T}`。共享模型在时间 `t`、可见模态子集 `S⊆M` 上输出 coalition
worth `v_t(S)`；空集由固定、可审计的 null input 定义。对每个非空 `S`，定义

```text
I_t(S) = Σ_{R⊆S} (-1)^(|S|-|R|) v_t(R)
v_t(M) = v_t(∅) + Σ_{∅≠S⊆M} I_t(S)
```

`I_t({m})` 是单模态独有 evidence；`|S|>1` 的 `I_t(S)` 是不能由其严格子集相加解释的
interaction atom。正视频的 bag likelihood 必须边缘化或连续松弛一个**单一**隐变量
`z=(t,S)`；负视频约束所有 `t,S` 不产生 hate evidence。test 只输出 `v_t(M)`，不得读取
coalition posterior 做 branch routing、ensemble、oracle selection 或 score calibration。

这一版与普通 flat `T×M` MMIL 的区别是，实例不再是一个 modality cell，而是一个由
子集格差分定义的 irreducible interaction atom；与 generic synergy loss 的区别是，它允许
某些视频由 singleton witness 解释，另一些由 pair/triple synergy 解释，而不是强迫所有正例
都依赖全部模态。

### “minimal/antichain”目前的必要修正

硬 antichain 不能直接加在任意神经 coalition scores 上。最小 winning coalitions 组成
antichain，前提是 coalition game 对集合包含关系单调；真实多模态 hate 不一定单调：新增
语境可能消除一个单模态看似 hateful 的含义，负的 interaction 也正是 Möbius 分解需要保留的
对象。另外，singleton evidence 与包含它的 pair synergy 可以同时真实存在，它们不是重复项。

因此 pilot 只能采用二者之一：

1. 明确定义并验证一个单调的“evidence availability”函数，再对其 minimal winning sets
   使用 antichain；或
2. 不声称全局 antichain，把 minimality 限定为 posterior 中一次只解释一个 `z=(t,S)`，
   通过 categorical marginalization 避免同一 bag event 的 supersets 被重复累计。

第二种更安全。Möbius atom 本身已经对子集贡献做了 inclusion-exclusion；再对所有嵌套
atom 施加互斥，可能错误删除真实的独有信息加额外 synergy。若实现没有解决这一点，本评审
自动降为 STOP。

## 最近且最接近的工作逐项差异

| 工作 | 已经覆盖 | 与严格候选的关键差异 |
|---|---|---|
| [MultiHateLoc, 2025/WWW 2026](https://arxiv.org/abs/2512.10408) | 同一弱监督 hate temporal localization；三模态 temporal encoder、dynamic fusion、contrastive alignment、modality-aware top-k MIL | 不枚举模态 coalition，不做 Möbius interaction decomposition；DMS 是 video/global 或 branch weighting，不是 `(t,S)` latent atom；其 full score 不是 coalition atom 的精确重构 |
| [HAN / Unified Multisensory Perception, ECCV 2020](https://www.ecva.net/papers/eccv_2020/papers_ECCV/html/513_ECCV_2020_paper.php) | 弱标签下在 time×modality lattice 上做 attentive MMIL，输出 modality-aware temporal parsing | 实例是单个 time-modality cell，不是 modality subset；没有区分 unique evidence 与 irreducible synergy，也没有 minimal coalition |
| [CoLeaF, ECCV 2024](https://www.ecva.net/papers/eccv_2024/papers_ECCV/html/1653_ECCV_2024_paper.php) | 明确区分 audible-only、visible-only 和 audible-visible event，并按对齐程度控制 cross-modal context | 用 reference/anchor 双分支和 contrastive-collaborative learning；没有子集格、Möbius atom、单个 `(t,S)` witness 或 full-score reconstruction |
| [I2MoE, ICML 2025](https://arxiv.org/abs/2505.19190) | masked-modality forward；weak interaction losses；分别建 uniqueness、synergy、redundancy experts；三模态 MOSI 等实验 | 是多个 interaction expert 加 reweighting 的 MoE，test 为 expert 加权和；多于两模态时只保留 `m` 个 uniqueness expert 加一个 global synergy/一个 redundancy，不枚举具体 pair/triple coalition；无 temporal weak MIL 和 minimal coalition |
| [MMoE, EMNLP 2024](https://aclanthology.org/2024.emnlp-main.558/) | 把样本分成 redundancy、uniqueness、synergy，训练相应专家 | interaction type 由单/多模态模型预测关系近似，test 动态融合多个 expert；属于候选明确禁止的 routing/ensemble，且不是 exact interaction decomposition 或 temporal MIL |
| [SynIB, 2026 preprint](https://arxiv.org/abs/2606.09853) | 直接以训练目标促进 synergy；full input 正确、拿掉任一模态后惩罚模型仍然自信；包含 Hateful Memes | 目标会把所有受约束样本推向“缺一模态就不可靠”，不能在同一弱标签下分配 singleton 与 pair/triple witness；没有完整 coalition lattice、Möbius attribution、temporal localization。它是 pilot 必须包含的强 control |
| [INTER, ICCV 2025](https://openaccess.thecvf.com/content/ICCV2025/papers/Dong_INTER_Mitigating_Hallucination_in_Large_Vision-Language_Models_by_Interaction_Guidance_ICCV_2025_paper.pdf) | 直接计算 image-text coalition 的 Harsanyi dividend，并把 interaction logit 用于生成时 guidance | training-free LVLM decoding；两模态、token generation，无弱 temporal MIL、minimal witness 或统一 localizer。它阻止“首次把 Harsanyi interaction 用于多模态预测”的 claim |
| [InterSHAP, AAAI 2025](https://ojs.aaai.org/index.php/AAAI/article/download/35452/37607) | 对任意模态数、单样本级分离单模态 contribution 与 cross-modal Shapley interaction | 训练后解释指标，不是训练目标；不做 temporal weak localization。但它是 coalition attribution 正确性的直接比较对象 |
| [Quantifying & Modeling Multimodal Interactions, NeurIPS 2023](https://proceedings.neurips.cc/paper_files/paper/2023/file/575286a73f238b6516ce0467d67eadb2-Paper-Conference.pdf) | 用 PID 系统刻画 redundancy、uniqueness、synergy，并比较捕获这些结构的模型与 loss | 主要刻画数据/模型交互类型，不把 interaction atom 作为弱 temporal latent witness；也说明“unique/redundant/synergy”分类本身不是 novelty |
| [HarsanyiNet, ICML 2023](https://proceedings.mlr.press/v202/chen23s.html) | 把网络输出分解为稀疏 Harsanyi interactions，并由网络结构支持精确 Shapley 计算 | 通用预测/XAI，不是 modality coalition、视频时序或 MIL；说明 Möbius/Harsanyi 分解和“少量 salient interactions”本身已有 |
| [SimMLM, ICCV 2025](https://openaccess.thecvf.com/content/ICCV2025/papers/Li_SimMLM_A_Simple_Framework_for_Multi-modal_Learning_with_Missing_Modality_ICCV_2025_paper.pdf) | 同一模型在 more/fewer modality subsets 上训练，并用 MoFe ranking 约束较丰富输入的 task loss | 目标是 missing-modality robustness；不估计 pure interaction，不选择 minimal positive witness，也无 temporal weak label。它阻止“训练同一网络遍历 modality subsets”的 claim |
| [Sufficient Input Subsets, AISTATS 2019](https://proceedings.mlr.press/v89/carter19a.html) | 寻找在其他特征缺失时仍足以维持决定的最小输入子集 | post-hoc、feature-level explanation；没有弱监督学习或 multimodal temporal likelihood。它阻止“minimal sufficient subset”概念本身的 claim |
| [Efficient Modality Selection, JMLR 2024](https://www.jmlr.org/papers/v25/23-0439.html) | 在 cardinality constraint 下按 subset utility / approximate submodularity 选择模态，并联系 Shapley contribution | dataset/global resource selection，不是每个时间的 latent explanatory coalition；没有 interaction atom 或 MIL |
| [HateClipSeg, ACM MM 2025](https://arxiv.org/abs/2508.01712) 与 [LELA, 2026](https://arxiv.org/abs/2602.09637) | 前者定义 segment-level hate localization benchmark；后者以多模态 caption 与 composition matching 做 training-free frame scoring | 都没有在 video-label-only training 中学习 minimal modality coalition；LELA 的 composition matching 也不是 game-theoretic coalition decomposition |

## 是否只是组件拼接

### 可以视为一个新核心机制的版本

以下三件事必须是同一个等式链而非三个旁支 loss：

1. `v_t(S)` 的所有 masked forwards 共同定义 `I_t(S)`；
2. positive bag likelihood 只从这些 `I_t(S)` 的 `(t,S)` latent witness 得到；
3. test score `v_t(M)` 由同一批 `I_t(S)` 精确重构，没有另一个未被该 likelihood 识别的 fused head。

这样，移除 Möbius decomposition 会改变 latent instance 的语义；移除 latent coalition 会改变
bag likelihood；移除 reconstruction 会使 test localizer 与训练机制脱节。三者相互依赖，能够
形成一个可检验的核心，而不是附加模块清单。

### 应判为组件拼接的版本

- 原 MultiHateLoc loss 加 modality dropout；
- 再加一个 Shapley/Möbius explanation regularizer；
- 再对 attention weights 加 sparsity 或 antichain penalty；
- test 仍读原 fused head，或融合多个 coalition heads；
- 用 test-GT 选择每个视频/数据集的 coalition。

这类实现的增益无法归因给 latent minimal coalition，并分别被 missing-modality training、
interaction XAI、sparse MIL 和 MoE routing 覆盖。

## 技术风险与 anti-pattern

1. **masked input 不是自然干预。** 零向量、随机向量、learned null token 会产生不同
   `v_t(S)` 和 dividend；interaction 可能只是 null choice artifact。
2. **弱 bag label 不识别 coalition ownership。** 多组 `I_t(S)` 可产生相同 bag likelihood；
   低熵 posterior 不等于正确 ownership。
3. **Möbius dividend 有正有负。** 对它直接 ReLU 会破坏 exact reconstruction；取绝对值会把
   suppressive context 错当 hate evidence。必须说明正、负 interaction 如何进入 likelihood。
4. **硬 antichain 与非单调语义冲突。** 不能因为集合嵌套就认定两个 interaction 重复。
5. **长视频 MIL peak bias 仍在。** 一个 `(t,S)` witness 可能只找到最显著秒，提升视频分类却
   恶化 span extent 和 within-video ranking。
6. **全 coalition negative constraint 主要改善 pooled separation。** 它未必教会正视频内部的
   benign seconds，必须单独检查 within-video ROC。
7. **训练/test 脱节。** 若 test fused head 不是 coalition atom 的 exact reconstruction，候选退化
   为辅助训练技巧。
8. **synergy forcing 会伤害单模态真实 hate。** 不能照搬 SynIB 到所有正视频；latent `S` 必须
   允许 singleton，并证明没有被 easiest modality 全面占据。
9. **计算量混淆。** 三模态每个时间需 8 次含空集前向；收益可能只是更多计算/augmentation。
10. **禁止 oracle 化。** test-GT coalition、best branch、按数据集选择 mask/null、分支路由和
    score ensemble 都只能是诊断，不得成为方法输出。

## 最小可证伪 pilot

### 数据、训练和 test

- HateMM 与 HateClipSeg 分别独立训练；不混合任何主数据集 train set。
- 沿用 MultiHateLoc 的输入特征、temporal encoder、训练预算与固定 checkpoint-selection
  过程；validation 只在每个固定训练内部选 checkpoint，不做方法比较或方向判断。
- checkpoint 选定后立即在 test 跑共享 evaluator 的 pooled AP、pooled ROC-AUC、
  within-video macro ROC-AUC。只以 test 结果决定下一步。
- test 只输出 full-modality `v_t(M)`；不做 routing、ensemble、calibration 或 transport。

### 最小 arms

1. **MultiHateLoc starting point。**
2. **Deletion control。** 保持原 temporal MIL，只删除 unconditional InfoNCE；排除增益仅来自
   去掉已有 harmful alignment。
3. **All-subset masked MIL。** 同一个共享模型、相同前向次数和输入 masks，但每个 coalition
   直接接受普通 bag loss；无 Möbius、无 minimal latent witness。排除 modality dropout /
   augmentation 与额外计算。
4. **SynIB-style control。** full input bag loss加 missing-one confidence penalty；排除“促进
   synergy”这一已知目标即可解释增益。
5. **Möbius non-minimal control。** 计算相同 `I_t(S)`，但把所有正 interaction 直接求和/池化，
   不设单一 `(t,S)` latent witness。
6. **完整候选。** exact Möbius reconstruction 加单一 `(t,S)` latent marginalization；采用上文
   安全的 posterior exclusivity，不先加未经证明的硬 antichain。

### 两层失败标准

**机制失败：**完整候选若不能在两个 test corpus 的 within-video ROC 都超过 arms 3、4、5，
或提升仅来自 arm 2，则 minimal coalition witness 机制被否证，停止该方向。

**项目晋级失败：**即便机制对照成立，只要两个 corpus 任一固定 test 指标未超过当时冻结的
SOTA 阈值，就不晋级为 SOTA 方法。当前 test 门槛以 `research-wiki/STATUS.md` 所列权威结果
为准：HateMM pooled AP / pooled ROC / within ROC 为 `.5938316/.8161838/.6315317`；
HateClipSeg 为 `.6193711/.6050225/.5619079`。最终还必须扩到 MHC-EN、MHC-ZH，且四个主
数据集全部三个指标 SOTA。

### pilot 必须记录的机制证据

- `v_t(M)` 与 `v_t(∅)+Σ_S I_t(S)` 的逐秒 reconstruction residual；
- 每种 coalition 的 posterior mass、被选时间数和正/负 interaction 分布；
- singleton、pair、triple witness 是否跨语料全部塌缩到同一种；
- controlled modality deletion 对同一秒 full score 的影响与 inferred coalition 是否一致；
- 对 test 中“fused 不如最佳单模态”的既有失败组，within-video ranking 是否定向改善；
- positive span coverage，而不只看最高峰或视频分类；
- 正视频内部 benign seconds 与负视频 seconds 分开报告，防止 pooled-only 提升。

test prediction/GT 可用于这些 error analysis，但不得进入训练、checkpoint selection、mask
选择或 inference rule；其后的数字必须标为 iterative/developmental test evidence。

## 后续 attribution controls

若 pilot 通过，完整实验至少需要：

1. temporal-only fused MIL：无 modality lattice；
2. flat `T×M` HAN-style MMIL：无 coalition subsets；
3. random modality dropout 与 all-subset supervised augmentation；
4. SynIB confidence penalty；
5. Möbius decomposition但无 minimal latent variable；
6. minimal subset pooling但不用 Möbius差分；
7. coalition identity 在每个视频内随机置换，保持各 order 的边际分布；
8. singleton-only、pair-only、triple-only；
9. old/new objective × InfoNCE present/absent 的 factorial；
10. 固定 null、learned null、train-distribution replacement 三种 baseline sensitivity；
11. 相同参数量、相同 coalition forward 次数、相同训练时长的 compute control；
12. exact reconstructed full score 对独立 fused head，证明增益确由统一 readout 承载；
13. posterior entropy 与 controlled deletion faithfulness，防止把尖锐权重叫作 ownership；
14. temporal extent、within-video ROC 和三项固定 test 指标，禁止只报视频级或 pooled 增益。

## 可支持与不可支持的 claim

若严格版本通过机制与性能验证，可以尝试的 claim 是：

> 在弱监督 multimodal temporal localization 中，将 Möbius-decomposed modality interaction
> atoms 作为 time-and-minimal-coalition latent MIL witnesses，并由同一分解重构唯一的
> full-modality temporal localizer。

不能声称：

- 首次使用 modality masking、coalition、Shapley、Möbius/Harsanyi interaction；
- 首次区分 multimodal uniqueness/redundancy/synergy；
- 首次学习 minimal sufficient subsets；
- 首次 time×modality MMIL；
- attention/posterior weight 等同真实 causal ownership；
- 仅因换到 hateful-video 数据就构成 novelty。

## 最终裁定

**CONDITIONAL GO，6.0/10。** 精确组合尚未被上述 primary literature 直接覆盖，并且它比已被
STOP 的 flat joint-witness MMIL 多出一个可表达后者不能表达的对象：某个时间点的
irreducible pair/triple modality interaction atom，同时仍允许 singleton witness。这个差异与
MultiHateLoc 暴露的 DMS/branch-supervision 失败有直接对应关系。

授权范围仅是做上述最小 pilot，不是授权论文 novelty claim。实现前必须先写出完整 likelihood、
正负 dividend 处理、null intervention 和 full-score reconstruction；若采用硬 antichain却没有
单调性定义，或 test 仍使用独立 fused head / coalition routing，则立即 STOP。
