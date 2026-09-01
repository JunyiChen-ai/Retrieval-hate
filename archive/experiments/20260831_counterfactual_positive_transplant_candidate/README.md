# 淘汰：Counterfactual positive-window transplant

截至 2026-08-31。独立查新 `STOP`（3/10），未实现、未训练、未读新的 test。
edit mask 只能证明替换位置，不能证明 weakly selected donor 是 hateful；核心与
VideoMix、WS-VAD segment mixing 及 selector pseudo-snippet distillation 直接重合。
见 `NOVELTY_SCOUT.md`。

## 机制假设

在每个主数据集各自的 train split 内，用冻结的多模态局部 evidence 选择正例视频中的高置信窗口；将该窗口的全部对齐模态作为一个整体，移植到随机位置的负例 host。原始负例与合成视频构成 paired counterfactual，已知的移植位置提供 dense change mask。训练一个 POWA residual ordering head 识别这项局部变化；POWA anchor 冻结，推理时不使用 donor selector 或额外 teacher。

这与已淘汰的 benign insertion 相反：不是在正例中插入负片段并直接重写 POWA score mass，而是以原始/合成负例对约束独立的局部 ordering head，尽量保留 frozen POWA 的 pooled 判别尺度。

## 必须先回答的问题

1. 该机制相对 Copy-Paste/CutMix、伪异常生成、snippet mixing、弱监督时序定位 augmentation 和 causal intervention 是否有可 defend 的新贡献。
2. 正例 donor 只由弱监督 evidence 选择，不能称为 certified positive；其噪声是否让机制前提失效。
3. aligned multimodal transplant 是否制造可被模型利用的编辑边界、模态不连续、视频 identity 或位置 shortcut。
4. 是否只是把现有 pseudo-label teacher 包装成数据增强；若是则直接淘汰。

## 若查新允许，最低归因对照

- random positive-video donor；
- negative-video donor；
- shuffled transplant mask；
- 单 view selector 与相同 coverage；
- 不做 paired loss / 不做 transplant；
- transplant location uniform random，并检查 center/edge bias；
- 保持同一 frozen POWA anchor、同一 score readout 和训练预算。

任何 val/test span GT 均不得用于 donor 选择、梯度或 checkpoint selection。test predictions/GT 可按 Rule 10 做 error analysis，但必须记录 exposure。
