# 20261102_r7_prior_label：第 7 版（第 6 版 + 先验的视频标签项）

截至 2026-11-02。机制、依据、预注册判定见 `experiments/20261101_r5_completion/README.md` 第 3.4 节；第 6 版结果 `experiments/20261101_r6_length_table/README.md`。

- 改动：在第 6 版（锚定两状态答案表 + 共用对数长度斜率，无复制似然）的损失上加 BCE(g, y) = −log P(G = y | x)，即不看回答时先验的视频标签项，权重 1（`src/qtl/train.py` 的 `prior_label: true`）。答案表、EIG、后验、qmixE1 停止与第 6 版相同。
- 搜索（规则 7）：同 r5 / r6 的四个标量与范围，Optuna TPE，sampler seed = 训练 seed，每 seed 20 trial（预写），三语料 × seed 234 / 2025 / 3407；`search.py`、`train.py`；启动 `launch/sbatch_search.sh <corpus>`。输出 `runs/20261102_r7_prior_label/<corpus>/seed<s>/`。
- 搜索后：`launch/run_final_curves.sh <corpus>`（test 选 / validation 选曲线）、`launch/run_sim.sh <corpus>`（测试期变体与打乱诊断）、`launch/sbatch_arms.sh <corpus>`（消融与对照，只在采用后跑）。
- 结果：待搜索完成。
