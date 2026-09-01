# Independent novelty review

截至 2026-08-31。审查范围仅为 Rule 12 novelty 三项硬门与机制可识别性；未审查代码，未实现、训练或生成 prediction。

## 最终裁定

**STOP，novelty 5.6/10。**

| Gate | 裁定 | 理由 |
|---|---|---|
| Gate 1：允许 adaptation 已有来源 | **PASS** | WACV 2024 messenger-guided mid-fusion 是可合法迁移的跨任务来源。 |
| Gate 2：来源核心未进入 hateful-video detection/localization | **PASS（窄义）** | 实际检索未发现 messenger-guided mid-fusion 或同名 messenger bottleneck 已用于 hateful-video detection/localization。目标任务已有 cross-attention、dynamic fusion、local/global fusion与 temporal label-noise分析，但不是该 messenger 实现。 |
| Gate 3：non-trivial、task-specific adaptation | **FAIL** | 真正新增部分是对 source messenger 加时间去均值和由 OOF lexical scalar 控制的乘法 gate。这是标准 residualization 加 feature gating；没有新增能识别 hate timing 的局部监督，也没有解析排除已有 broadcast、lexical self-matching 或 ownership shortcut。 |

因此本候选不能进入实现。它可以被描述为一个有动机的 `WACV messenger + lexical gate` baseline，但不能作为满足当前 novelty 标准的新方法。

## 来源占用检查

### 跨任务来源

[Xu, Hu, Lee, *Rethink Cross-Modal Fusion in Weakly-Supervised Audio-Visual Video Parsing*, WACV 2024](https://openaccess.thecvf.com/content/WACV2024/html/Xu_Rethink_Cross-Modal_Fusion_in_Weakly-Supervised_Audio-Visual_Video_Parsing_WACV_2024_paper.html) 已经提出：

- 用低容量 messenger 压缩完整跨模态上下文；
- 将早期强耦合改成 mid-fusion，以减少不相关 audio/visual 信息传播；
- 针对音频到视觉的非对齐噪声加入 prediction-consistency 约束；
- 在只有 video-level union label 的弱监督 AVVP 中做时间和模态解析。

因此，messenger、低容量跨模态通信、以弱 video label 下的 modality noise 为动机，均属于来源已有内容，不能作为本项目 claim。

### Hateful-video 最近邻

- [MultiHateLoc](https://arxiv.org/abs/2512.10408) 已采用 modality-aware temporal encoder、dynamic cross-modal fusion、cross-modal contrast与 modality-aware MIL；它占用了“一般多模态动态融合/对齐用于 hate localization”的宽 claim。
- [Reasoning-Aware Multimodal Fusion](https://arxiv.org/abs/2512.02743) 已在 hateful-video detection 中使用 local-global context fusion与 semantic cross-attention；它进一步压缩了“一般限制或改造融合路径”的可主张空间。
- [Yang et al., *Revealing Temporal Label Noise in Multimodal Hateful Video Classification*](https://arxiv.org/abs/2508.04900) 已建立 coarse video label 与真实局部 hate timing 错配的目标任务动机，但未提出 messenger bottleneck。

未检出精确 source mechanism 进入目标任务，所以 Gate 2 仍窄 PASS；但“未检出逐项相同实现”不能弥补 Gate 3 的组件拼接问题。

## 为什么第三门失败

### 1. 最直接的 broadcast 解完全不需要 messenger

令各 unimodal temporal branch 输出视频常量：

`h_visual,t = a_v`，`h_audio,t = b_v`，`h_text,t = c_v`。

文本 residual 与 lexical innovation 都可为零，因而所有 gated messenger 为零。最终 fused head 仍可直接从 unimodal states 得到视频级常量 `s_t=C_v`，由 MIL 完成 bag classification；positive-video within ROC 严格为 `.5`。

删除 full cross-modal path只阻断了一条 broadcast 路径，没有阻断模型已有的三个 unimodal/global路径。候选也没有目标项要求最终 score 必须依赖 messenger。`gate-off` control可以事后发现 messenger 被忽略，但不能使当前 adaptation 获得新的识别性。

### 2. 时间去均值不消除 video identity

更强的反例为：

`h_text,t = c_v * p_t + eta_t`，其中 `p_t` 是固定、零均值的时间/位置基。

则

`h_text,t - mean_t(h_text) = c_v * p_t + centered(eta_t)`。

视频身份 `c_v` 仍完整保留在 residual 的幅度中。再乘由同一 video labels 学出的 lexical innovation `g_t`，模型即可把全局正视频判断放到 lexical 或固定位置上，而不需要验证同秒 visual/audio 是否含 hate。该解与项目已经否定的 `b_t=c_v, r_t=g_v*a_t` lexical-alignment shortcut同构，只是把乘法从最终 logit 移到 messenger feature。

因此，“constant component被代数消掉”只排除了严格 additive constant；它没有排除 video identity、位置编码、尺度、剪辑模式或 topic 经零均值基传播。

### 3. Lexical gate仍是到输出的直接乘法路径

候选声称 lexical不能直接加到 frame logit，只能调制消息。这不构成信息隔离：下游 fused head可读出

`message_t = phi(h_text,t - mean(h_text)) * tanh(g_t)`。

只要 `phi` 能产生稳定的非零幅度，最终 head即可近似恢复 `g_t` 或其符号。于是模型可能只重现已知 lexical ordering，而不是学习 cross-modal ownership。feature-space乘法路径与 logit concat在参数位置上不同，但没有提供新的监督量。

另外，`g_t` 是由 OOF whole-transcript classifier从同一 video labels 学出的派生 prediction。它程序上可以是 train-only、cross-fitted且不含 span GT，但不能被表述为与 teacher prediction无关的原生观测。

### 4. 与 context-quotient 失败链仅有层级差异

项目旧 context-quotient 方法把 global anchor 与 zero-mean temporal residual分开；正式结果显示 shuffled/position controls可强于声称的 span机制，证明去均值本身没有带来正确局部方向。

当前候选把同一 `global + zero-mean residual` 分解提前到 text representation/messenger层，并保留未受约束的 unimodal global路径。端到端非线性可以重新编码被减掉的均值，或从其他分支恢复同一 nuisance。因此它不是对 context-quotient 识别问题的新解，只是把 quotient 的位置从 score移到 fusion。

### 5. 没有解决 ownership failure

MultiHateLoc 的已知失败是 DMS 几乎总选 visual，且真实最佳 modality 与选择结果严重错配。本候选不学习或监督 time×modality owner，而是预先规定 text→audio/visual通信由 lexical gate控制，其他方向保持 residual messenger。

这是一项固定通信策略，不是 ownership inference。对于 HateClipSeg 的 silent/static/OCR/visual hate seconds，speech mask会令该新增路径消失；最终仍依赖原有 unimodal branch或普通 residual messenger。整体 lexical premise在 HateClipSeg 只提供弱平均方向，不能支持“lexical gate统一修复ownership”的 claim。

## 与近期失败链的关系

- **Lexical posterior regularization**：曾能降低部分 train constraint violation，却没有改善 test ranking。当前方法改变约束位置，但仍没有独立局部 target保证 lexical information进入正确 hate ranking。
- **Counterfactual carrier alignment**：已由 `global video indicator × local lexical/speech pattern` 反例否定。当前 messenger可以实现同一乘法结构，且没有negative invariance，约束更弱。
- **Context quotient / within-between residual**：都表明 zero-mean只移除一种常量表示，不能移除video identity或确定局部方向。
- **Ownership / carrier-energy**：都表明某个分支可被忽略或支配最终 score。当前唯一 fused head并不意味着所有 messenger对最终 ranking均 load-bearing。
- **Auxiliary bypass**：虽然这里没有独立 auxiliary head，但网络可以令 messenger恒零、直接用unimodal branches；这是架构内旁路，而不是旧 projection-head 形式的旁路。

候选已有的 lexical locality premise满足当前 Rule 14 最低证据要求，但 Rule 14 PASS 不等于 Rule 12 Gate 3 PASS。相同信息源不能仅通过移动 gate 所在层级而重新取得 novelty。

## Controls 的判别力

README 的 capacity-matched ungated messenger、time-shuffled lexical、gate-off与 mean-repeated controls是合理的经验归因工具，但不能修复上述结构问题：

- core胜 ungated只说明 lexical conditioning有用，不说明学到了跨模态 hate ownership；
- core败于 gate-off可证明 lexical路径影响prediction，但仍可能只是恢复 lexical score；
- time-shuffle可证明正确时间对应有用，但 lexical-locality premise已经提供这一事实，不能区分 lexical self-matching与真实cross-modal evidence；
- mean-repeated同时破坏大量native temporal information，不能单独归因于 messenger。

若只把本方案保留为 baseline，至少还需要 lexical-only gated branch、禁止 text进入receiver-content的 control、position-basis residual、以及逐视频 messenger-to-final-score responsibility审计。但这些 controls只能限定工程结论，不能把当前组件组合提升为 Rule 12 novelty GO。

## 允许与不允许的表述

允许：

> 在 MultiHateLoc 上测试 lexical-conditioned residual messenger 是否比 full fusion或ungated messenger更适合该数据。

不允许：

- 首次解决 weak-label hateful-video 的 cross-modal broadcast；
- 学到了 temporal modality ownership；
- lexical没有到输出的直接信息路径；
- constant annihilation解析排除了video-global shortcut；
- messenger-guided weak multimodal fusion本身新。

## 结论

精确 WACV messenger来源未被 hateful-video task占用，前两门通过；但候选的有效新增量仍是普通 temporal residualization与 lexical feature gate。它没有新增可识别的局部关系，且与 context-quotient、lexical-alignment和branch-bypass负结果存在明确等价解。按 Rule 12 第三门，裁定 **STOP 5.6/10**，不进入实现。
