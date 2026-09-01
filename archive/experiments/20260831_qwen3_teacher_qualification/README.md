# REJECTED — Qwen3 pointwise dense-teacher qualification for POWA

截至 2026-08-31。该轮在完整 HateMM validation 触发冻结 kill gate，HCS 在
开始后被提前终止并标为 partial/non-authoritative；未读取 test、未进入 student
训练。该轮只验证 teacher signal，不是新方法。它检查此前因
Qwen2.5-VL teacher 太弱而停在 Stage T 的“dense teacher → order-only POWA
distillation”是否值得恢复。使用 Qwen3-VL-8B 不单独构成 novelty。

## 固定设置

- corpora：HateMM、HateClipSeg，各自完整 validation 正例视频；不读取 test。
- window 16 秒、stride 8 秒，每窗 4 张 1 fps cached frames + 对齐 ASR。
- Qwen3-VL-8B-Instruct，deterministic decoding，输出整数 0–10。
- overlap seconds 取 covering-window mean；共享 evaluator 计算 within ROC。
- 只用 validation GT 做本轮 teacher qualification；不训练、不选 checkpoint。

## 冻结 gate

- parse failure rate 每 corpus `<1%`；coverage 完整；
- HateMM within ROC `>=.60` 且高于 corpus-specific POWA validation `.571931`
  至少 `+.020`；
- HateClipSeg within ROC `>=.56` 且高于 corpus-specific POWA validation
  `.527072` 至少 `+.020`。

两 corpus 全过才允许恢复 order-distillation candidate；任一失败就判定 dense
pointwise teacher family 仍不足。通过只说明 teacher signal 合格，不说明 student
会继承，也不允许直接访问 candidate test。

权威输出：`runs/20260831_qwen3_teacher_qualification/summary.json` 及同目录
teacher/raw artifacts。

## 结果与裁定

HateMM 完整 43 个正例 validation 视频、711 个 windows：Qwen3 teacher within
ROC `.537386`，低于 corpus-specific POWA `.571931`，同时未过绝对 `.60` 与
`POWA +.020` 两个 gate。此前两视频 `.848968` 是明显的小样本假象。

因两 corpus 必须全过，HateMM 已足够判死；HCS 在 4 个完整视频后终止，其 partial
artifact 不报告 performance。结论：从 Qwen2.5 升到 Qwen3 仍不能提供稳定 dense
pointwise ordering signal，原 order-distillation candidate 不恢复。
