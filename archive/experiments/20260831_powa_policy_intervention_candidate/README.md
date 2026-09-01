# 淘汰：POWA policy-AST primitive intervention consistency

淘汰原因：独立 novelty review 给 `STOP`、`3.6/10`；post-PEF intervention 的方向性
由固定 AST 代数自动保证，loss 近似 tautology，pre-AWB intervention 又因 Sinkhorn
跨时间重分配和 HCS target 的双分支作用而不存在统一单调方向。完整组合未见同构论文，
但去掉既有 POWA、dense VLM teacher/KD 后，剩余核心已被 intervention-aware concept
bottleneck 与 differentiable logic constraint 占位。未实现、未训练、未读 test。

截至 2026-08-31。候选原定义：在 corpus-specific POWA 内使用同语料 dense typed
primitive teacher，并对一个 primitive channel 做训练期 counterfactual replacement，
要求 executable moderation-policy AST 的输出按 leaf polarity 变化，同时保持无关分支
不变。teacher density/backbone/KD 不计 novelty。

独立查新与数学审计见 `NOVELTY_SCOUT.md`。结论是不应把内部 concept overwrite 称作
causal intervention，也不应实现一个固定 compiler 已自动满足的 consistency loss。
