# 淘汰：ImageBind feature ceiling diagnostic

淘汰原因：固定统一 concat 在 HMM 抬高 supervised ceiling，但 HCS 三项全部下降，未过
HCS within `+.020` 门。权威输出
`runs/20260831_imagebind_feature_ceiling/analysis.json`（SHA256
`8b728e23e6fcbc6dad7da13a112e346994dae48cadca325fe5357ed6e05582dd`）。不尝试
image-only 或按语料选择 modality，不接入 POWA、不训练弱监督候选。

截至 2026-08-31。不是弱监督方法，不作 novelty/performance claim。只用每个语料自己的
train span rasterization 训练线性 frame probe，再在 test 上按共享 evaluator 评估；test
predictions/GT 属 Rule 10 developmental error analysis，不能冒充 confirmatory 结果。

唯一候选输入在结果产生前固定为：现有 1 fps CLIP-B/16 + VGGish + BERT sentence，拼接
ImageBind-Huge image/video/audio 三个逐行 L2-normalized streams。ImageBind 按物理采样率
image=4 fps、video/audio=.5 fps 映射到 1 fps；缺失 stream 用零向量并记录。`current`
arm 使用原三流，是完全相同训练/evaluation 的 matched control。不尝试 image-only、
per-corpus modality selection、score fusion 或 ensemble。

输入才有资格进入后续弱监督 POWA pilot，必须：HateMM within 不低于 current `.010`，
HCS within 至少提高 `.020`，且两语料 pooled AP/ROC 相对 current 各下降不超过 `.020`。
任一失败即归档；通过只说明统一 backbone 有 capacity，不说明弱监督可实现，也不计方法
创新。最终方法仍不得读取任何 train span。

## 结果

- HateMM current AP/ROC/within `.52275/.79810/.64773`；加入统一 ImageBind 三流后
  `.54823/.80675/.67243`，三项改善。
- HateClipSeg current `.65729/.62984/.56007`；加入后
  `.65167/.62229/.55689`，三项下降。

这说明 ImageBind 含 HMM 的附加容量，但没有修复 HCS feature ceiling。根据 frozen
single-combination 规则，不能事后从 image/video/audio 中挑一个看起来更好的 branch。
