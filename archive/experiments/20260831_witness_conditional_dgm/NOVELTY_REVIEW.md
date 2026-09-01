# Independent narrow novelty review

截至 2026-08-31。依据 RESET2 后修订的 Rule 12，只审查来源占用、adaptation 是否直接套用/组件拼接、是否与 failure ledger 严格同构、以及来源必要监督是否缺失。不审查代码，不要求实现前给出完整 identifiability theorem。

## 最终裁定

**GO，novelty 6.7/10。**

| Gate | 裁定 | 理由 |
|---|---|---|
| Gate 1：允许 adaptation 已有来源 | **PASS** | DGM/MSDU 是明确、可迁移的跨任务训练机制；本项目不必从零发明梯度调制。 |
| Gate 2：来源核心未进入 hateful-video detection/localization | **PASS** | 实际检索未发现 Fu、Gao、Xu 的 DGM/MSDU，或等价的 modality-separated dynamic gradient modulation，被用于 hateful-video detection/localization。 |
| Gate 3：non-trivial、task-specific adaptation | **PASS** | 候选没有只把 audio/visual 扩成 visual/audio/text；它把 source 的 video/category optimization competence 改成由 positive-bag temporal witness separation 与 negative-bag worst false alarm 共同定义的 localization competence，并用 source-style DGM 作为 matched control。该改动直接对应已证实的 visual domination 与错误 modality competition。 |

该 GO 只批准 README 所定义的最小 HMM/HCS end-to-end pilot。它不预判方法有效，也不允许在实现前继续扩机制。

## Gate 1：来源机制可 adaptation

Primary source：[Fu, Gao, Xu, *Multimodal Imbalance-Aware Gradient Modulation for Weakly-supervised Audio-Visual Video Parsing*](https://arxiv.org/abs/2307.02041)。

来源已经提出：

- 用 modality-separated decision unit 分别估计各模态的优化进度；
- 根据模态预测的正确类别置信度及正确/错误类别差异衡量 imbalance；
- 对较强模态的 encoder gradient 做 bounded attenuation，让较弱模态获得相对更多优化；
- 不要求改变最终 inference graph。

这与本项目的弱监督 temporal localization相邻，但属于 audio-visual event parsing，不是 hateful-video task。将其作为 source 符合 Gate 1。

## Gate 2：目标任务占用检查

检索覆盖了以下精确术语及其组合：

- `Multimodal Imbalance-Aware Gradient Modulation`；
- `dynamic gradient modulation`；
- `modality-separated decision unit` / `MSDU`；
- 上述术语与 hateful video detection、hate video localization 的组合。

未发现 DGM/MSDU 被用于 hateful-video detection/localization。检索中的 `DGM4` 是多模态媒体篡改检测与定位数据集/任务名称，不是这里的 Dynamic Gradient Modulation，也不构成占用。

目标任务已有 MultiHateLoc 的 modality branch、dynamic fusion、cross-modal contrast与 MIL，也有一般 multimodal fusion、knowledge distillation及 agent reasoning 方法；但未见用 branch optimization competence 动态调制训练梯度来处理 hateful-video modality imbalance。因此 Gate 2 PASS。

允许的 occupation claim 必须保持窄：不能声称 gradient modulation、balanced multimodal learning、MSDU、top-K MIL或弱监督 temporal parsing本身新。

## Gate 3：为什么不是简单三模态移植

### Source statistic 与 task statistic 不同

Source DGM主要回答：某模态是否已经更好地完成 video/category classification。若直接用于 binary hate bag confidence，它只能平衡 positive/negative video classification，无法区分“整段高分”与“局部 witness 被分开”。

候选把 competence 改为：

- positive bag：`topK_mean(p_m) - mean(non_topK(p_m))`；
- negative bag：`1 - topK_mean(p_m)`。

第一项把模态 competence 定义成正视频内部 latent witness 相对其余时间的分离；第二项使用 negative video 的全帧 benign certificate，衡量该模态最危险局部 false alarm。二者共同决定哪个 modality encoder在当前 batch 中应继续得到较多优化。

这不是仅把 source 的两个 modality loop 改成三个 modality loop，也不是在原模型旁边添加普通 top-K loss。它改变的是 DGM 的反馈量：从 classification optimization progress 改成 localization-oriented witness competence。该反馈量随后直接控制三个 encoder及其 fused-input projection 的梯度。

### 与已证实 failure 对齐

项目 test error analysis已经显示：

- DMS 几乎总选择 visual；
- DMS 选择与 test-GT 最佳单模态匹配率很低；
- best-branch oracle 相对 fused within-video ROC在 HMM/HCS 都存在明显缺口。

候选不使用 oracle owner，也不在 test routing；它只在训练期把“哪个分支当前更会区分局部 witness且更能抑制 negative false alarm”用于优化资源分配。这是针对已证实 modality competition failure 的具体 adaptation，而不是仅凭一般 multimodal imbalance 动机移植 DGM。

### 必要监督条件存在

该 adaptation需要：

- video-level positive/negative labels；
- negative video 的全部时间均为 benign；
- 各 modality 的 frame probability；
- 与 starting model一致的 top-K latent-witness assumption。

这些条件在当前任务与 MultiHateLoc training protocol 中都存在。它不需要 span annotation、人工 owner label、其他主数据集 supervision或 test labels。因此不存在“来源不可缺少的监督条件明确缺失”的 STOP 理由。

### 与 ledger 不严格同构

- 它不是 auxiliary-head bypass：modulation直接改变最终 fused score唯一输入的 encoders；MSDU branch score只产生 stop-gradient coefficient，不作为独立 test head。
- 它不是 direct-head replacement：inference graph与 head不变。
- 它不是 teacher-order KD、ensemble、calibration或test routing。
- 它与 single-carrier/branch-dominance风险相关，但增加了一个新的训练约束：competence必须同时考虑正袋内部 topK-vs-rest 与负袋 topK false alarm，并据此改变不同模态的优化速度。因此不是“严格同构且无新增约束”。
- 它也不同于 flat joint witness ownership：不拟合 categorical time×modality owner或 noisy modality label，而是调制 encoder optimization balance。

在 RESET2 口径下，这些差异足以通过 Gate 3。是否真的降低 visual monopoly、是否改善正确 branch或最终 ranking，属于端到端 test问题。

## Novelty 边界

允许的窄 claim：

> 将弱监督 AVVP 的 modality-imbalance gradient modulation 改写为 hateful-video localization 的 witness-conditional competence：正袋使用 topK witness 对其余时间的分离，负袋使用 worst-topK false-alarm suppression，并只在训练期调制进入唯一 fused localizer 的 modality gradients。

不能 claim：

- 首次发现 multimodal optimization imbalance；
- 首次使用 gradient modulation或 modality-separated decisions；
- 首次使用 top-K MIL；
- 已经学到真实 modality owner；
- 已解析排除所有 video-global、随机尖峰或长事件风险。

## 登记为 test 风险，不作为 pre-run STOP

以下风险真实存在，但按 RESET2 Rule 12 应交给最小 test与 matched control，而不是提前要求 theorem：

1. exact broadcast会使 positive competence接近零，DGM可能没有有用差异；
2. 随机单点尖峰可能得到高 topK-vs-rest competence，从而错误地被当成强模态；
3. 长而较均匀的真实 hate span可能比短峰得到更低 contrast；
4. gradient coefficient改变 encoder并不保证 DMS visual monopoly或最终 fused ranking必然改变；
5. 三模态平均可能让两个弱分支共同改变强分支的 attenuation尺度。

这些风险均不会让核心项在代数上无法进入 final score，也不证明 adaptation 与旧机制严格同构。

## Falsification 与 matched control

README 的 control足以支持首轮归因：

- `source-style video-confidence DGM` 保持相同 DGM实现、三模态结构与调制预算；
- 唯一差异是 competence 使用 source-style correct-label bag confidence，还是 witness-conditional positive/negative statistic；
- 若 source-style control等于或优于 core，则 topK-vs-rest 与 negative-topK adaptation不 load-bearing，机制失败。

预期也直接可证伪：core需在 HMM/HCS 的 test within-video ROC都胜 capacity-matched MultiHateLoc，至少一边达到预注册幅度，并在方法 test后检查 visual monopoly、gradient coefficients与 branch concentration。最终 performance gate仍由固定三指标与四主语料裁定。

## 结论

DGM/MSDU来源未被 hateful-video task占用；当前 adaptation不只是三模态端口，而是把分类型 modality competence改造成正袋 temporal witness separation加负袋 worst-false-alarm competence，并通过训练梯度直接作用于唯一 fused localizer。必要监督在任务中存在，且 source-style matched control能够判断新增 statistic是否真正 load-bearing。

按 RESET2 后 Rule 12，裁定 **GO，6.7/10**。下一步应停止追加 novelty/identifiability审查，进入最小实现、一次 technical review与 HMM/HCS 独立训练后立即 test。
