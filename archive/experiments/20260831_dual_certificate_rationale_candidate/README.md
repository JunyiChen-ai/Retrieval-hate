# STOP — Dual-certificate benign-filled temporal rationale

截至 2026-08-31。未实现、未训练、未生成 prediction。

候选拟适配 sufficiency/comprehensiveness rationale learning 与 meaningful/extremal perturbation：positive video 的稀疏连续 mask 在 selected-only composite 上保持 positive，在 remove-selected composite 上变 negative；其余内容由 topic-matched negative-video frames 填充；test 只输出 selector。

两路独立审查 STOP（`2.4/10`、`2.8/10`）。项目内 `distributionally_stable_rationale` 已是 negative replacement + keep/remove dual certificate 的近同构实现，目标任务占用门失败。联合训练还允许 selector/predictor 用 mask 位置、面积、splice/source fingerprint 编码 bag label；任意错误单帧可以同时满足两个 certificate。fixed budget/TV 与 topic matching 不排除该串谋，也不能处理冗余、多段或分布式 hate。无需 premise，不进入训练。

