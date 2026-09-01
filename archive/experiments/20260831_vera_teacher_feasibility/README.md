# 淘汰：VERA local-teacher feasibility over POWA anchors

淘汰原因：VERA local ordering 只在 HCS 有明显互补性，在 HMM 无效，不能作为统一
POWA teacher。HMM `official_postprocessed` transport within `.56859` 低于 POWA
`.57193`；HCS 为 `.55800`，高于 POWA `.52707`。

截至 2026-08-31。validation-only upper-bound diagnostic，不是 ensemble、candidate
method 或 performance claim，不读 test。它回答：现有最强 localizer VERA 的局部排序
若承载到 corpus-specific POWA 的绝对 score multiset 上，是否在 HMM/HCS 都有足够的
互补性，值得开发 train-only teacher distillation。

逐视频按 VERA `raw/neighbor/official_postprocessed` 排序重新分配原 POWA 分数；记录
逐视频 multiset exactness、共享 evaluator 三指标和 center-position control。这里直接
使用 VERA validation predictions，因此只能作为 feasibility upper bound；若后续训练，
teacher 必须重新在同语料 train videos 上生成，val 只用于冻结 checkpoint 选择。
EN/ZH 现有 VERA artifact 是 test cohort，不在本 validation probe 偷用；只有 HMM/HCS
存在与当前 val split 完整对齐的 `val_infer/scores.jsonl`。

权威输出：`runs/20260831_vera_teacher_feasibility/analysis.json`。逐视频 multiset
误差为 `0`。HCS VERA neighbor 自身 AP/ROC/within 为
`.62685/.66652/.55698`，确认 visual-semantic evidence 能补 HCS；HMM 对应
`.38337/.60850/.51843`，不能直接蒸馏到所有语料。
