# 淘汰：POWA context-quotient span-marginal pilot

淘汰原因：HateMM 与 HateClipSeg core 均无任何 epoch 保住冻结 pooled
feasibility，正式 supervisor 判定 `KILL_BEFORE_TEST`；且 HateMM 的 shuffled-span
control 明显强于 core，连续 span 归因不成立。

截至 2026-08-31。正式 Rule-9 review PASS，六个 Stage V validation runs 的
integrity 全部 PASS。权威裁定：
`runs/20260831_powa_span_marginal_pilot/stage_v_summary.json`
(SHA256 `c552a06e1b41a18e3f78d3ceee2a1069e0c999fb398f84919e52256c5076a067`)。
两语料均 `selected_epoch=null`，canonical candidate test 未运行。

唯一 starting point 是 corpus-specific POWA。相对 C2 的唯一核心变化是监督：不做
synthetic insertion/order loss，改用原始同语料 train videos 的 video label，对
zero-mean local residual 做 normalized variable-span marginal likelihood。新增 residual
固定工作在与训练一致的 200-bin 全视频相对坐标；验证/测试时映射回 dense snippet
grid 后再与未改动的 POWA anchor 相加。loss 使用 centered POWA logit + residual，
避免 zero-init singleton 对照无梯度。
context quotient 的精确定义只是对 supplied frozen channels 的 additive
video-constant offset 不变，不扩大声称为去除所有 nonlinear video identity。

## 结果与机制结论

- HateMM POWA validation 为 AP/ROC/within `.75766/.87442/.57193`。core 最好
  within `.58821`（`+.01628`），但 AP `.75324` 低于允许下界 `.75566`，改善视频
  比例也只有 `.488`。shuffled-span 最好 within `.64353`，远高于 core，而固定
  center-first 可到 `.75357`：收益由全局位置/分布先验解释，不是连续 hate span。
- HateClipSeg POWA 为 `.50639/.59854/.52707`。core 最好 within `.53008`
  (`+.00301`)，但 pooled AP/ROC 跌到 `.48635/.57115`；singleton 和 shuffled
  也同量级，未发现 span-specific signal。
- `runs/.../dev_smoke_singleton/` 仅是修复零梯度后的非权威开发 smoke；正式数字只认
  六个 corpus/arm run 及 supervisor summary。
