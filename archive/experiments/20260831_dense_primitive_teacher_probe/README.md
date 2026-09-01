# 淘汰：Dense primitive teacher qualification（diagnostic）

淘汰原因：完整 HateMM validation 的 compiled within ROC 为 `.56989`，低于 POWA
`.57193`，也未过预注册的 POWA `+.020` 门 `.59193`。虽然 typed policy 相比最好
control 有 `+.05303`，但 teacher ordering 质量不够，故按任一语料失败即停止的规则在
HCS 完成前终止；不生成 train teacher、不训练 student、不读 test。权威 verdict：
`runs/20260831_dense_primitive_teacher_probe/full_validation/verdict.json`。

截至 2026-08-31。仅诊断 POWA PEF 的 teacher coverage 缺口，不是方法创新；不训练、
不读 test。当前 POWA 每个 train video 最多仅有两个 Qwen2-VL primitive chunks。

固定 smoke：HateMM/HateClipSeg 各取按 id 排序前 8 个 eligible positive validation
videos；16 秒 window、8 秒 stride、每窗 4 帧 + aligned ASR；Qwen3-VL-8B 一次输出
六个 0..4 primitive。按 POWA 固定 policy AST 直接 compile window score，再映射到
1 fps，用共享 evaluator 报 subset within ROC。

这只回答“更密的 typed semantic teacher 是否同时有局部 ordering signal”。任一语料
compiled within `<.55`、parse failure `>=1%` 或 coverage 不完整即停止；两者都过才
冻结完整 validation qualification 与 train-teacher generation plan。backbone upgrade、
teacher density、VLM scoring/KD 均不计 novelty。

## Smoke 结果与完整 validation 门（运行前冻结）

权威 smoke：`runs/20260831_dense_primitive_teacher_probe/smoke8/summary.json`。
HateMM/HateClipSeg subset within ROC 分别为 `.58878/.55102`，330/330 calls 可解析，
故只通过“值得完整检查”的门；8-video 数字不作性能结论。

完整 qualification 固定使用全部 eligible positive validation videos，其他配置与 smoke
完全相同。必须同时满足：coverage 完整、parse failure `<1%`，且 compiled teacher
within ROC 相比当前 corpus-specific POWA validation 至少 `+.020`：HateMM
`>=.59193`（POWA `.57193`），HateClipSeg `>=.54707`（POWA `.52707`）。任一失败即
淘汰 dense typed teacher 路线，不生成 train teacher、不训练 student。通过也只说明
teacher 有资格作为 POWA 的监督来源，不构成 novelty 或方法晋级。

在完整输出产生前另冻结 attribution gate，防止把普通 VLM hate score 当成 typed-policy
机制：用同一批 raw primitives 和同一共享 evaluator 比较 `(a)` fixed compiled policy、
`(b)` hostile-only、`(c)` hostile/violence/sexual/self-harm 的 untyped maximum、`(d)`
primitive channels 循环平移一位后再执行原 policy。compiled policy 必须在两个语料分别
严格高于三个 control 中的最好值至少 `.010`；否则 typed policy 不是 load-bearing，
即使上面的 teacher-quality 门通过也停止。control 只做 validation attribution，不据此
改 prompt、policy、window 或阈值。

## 完整结果

HateMM 覆盖 43/43 eligible positive validation videos、711 windows、0 parse failure。
compiled policy within ROC `.569891`；hostile-only `.514438`、untyped harmful maximum
`.516860`、cyclic primitive-policy `.479574`。这说明 typed composition 的确比三个
control 强，但它仍低于未加 dense teacher 的 POWA ordering，并不能作为监督源改善
starting point。HCS 只产生 19 个 partial calls，未完成任何视频，不报告 performance。
