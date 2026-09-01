# Semantic-Conductance Diffusion MIL

截至 2026-09-01。**Novelty STOP 3.7/10；未实现、未训练、未生成 prediction。** 当前失败方法计数保持 `2/3`。

Gate 1/2 PASS，但 Gate 3 FAIL：原 GAD 已包含 guide gradient conductance、多 guide minimum 与 edge-preserving propagation；这里只把空间像素换成时间秒、guide image 换成三模态 embedding，本质仍是 content-aware smoothing，并与已关闭 propagation/smoothing 链严格同构。

## Failure

已有 HMM/HCS test 显示 structured localizer 的高分区相对真实 transition 中位膨胀 `4.5×/5.5×`，但固定 7 秒 smoothing 使 HMM within `.63377→.65826`、HCS `.53652→.52213`。因此需要减少内部碎片，同时不能跨过由 speech/text/OCR 而非画面变化定义的真实边界。

## Source and adaptation

来源是 guided anisotropic diffusion（弱监督 change detection）：用观测内容决定局部 conductance，只在同质区域传播预测并在内容边缘停止。检索需确认该方法未用于 hateful-video detection/localization。

适配为 1fps 三模态 temporal reaction-diffusion layer。MultiHateLoc 产生 raw fused logit 与 visual/audio/text embeddings；相邻秒的每个模态分别产生 similarity，最终 edge conductance 采用三者的 soft minimum，因此任一模态出现局部语义变化都能关闭跨边界传播。固定少量显式 diffusion steps 只更新唯一 fused frame logit，同时保留 raw-logit reaction term，避免变成自由 temporal encoder。模型只用同语料 video labels 训练，test 输出一个 raw diffused score。

这不是固定 smoothing：HMM 的内部碎片可在多模态同质区合并；HCS 即使视觉静态，speech/text 变化仍能阻止越界。它也不做 modality routing、配额、teacher、ensemble或 calibration。

## Control and falsification

Matched control 使用相同 backbone、diffusion steps 与 reaction coefficient，但把所有有效边 conductance 固定为 1，代表 isotropic smoothing；另保留 identity/no-diffusion anchor。Core 必须在 HMM/HCS test within 同时胜两者，至少一边 `>=+.020`，且 pooled AP/ROC 不下降到失去 SOTA 可行性。否则记第三次失败方法迭代并立即触发 process review，不调 diffusion step、temperature或reaction coefficient续命。

来源：Saha et al., *Weakly Supervised Change Detection Using Guided Anisotropic Diffusion*。
