# 20261101_r6_length_table：第 6 版（答案表加共用长度斜率）

截至 2026-11-01。机制、依据、预注册判定见 `experiments/20261101_r5_completion/README.md` 第 2、3.2、3.3 节；规则 4 审稿 `experiments/20261101_r5_completion/REVIEW_RULE4_R6.md`。

- 改动：r5 的锚定两状态答案表加一条两状态共用的对数长度斜率（负例训练视频上估计，`qtl.qtree.fit_anchored_shared`），训练、EIG、后验、qmixE1 停止都用它；去掉复制似然。其余同 r5。
- 搜索（规则 7）：同 r5 的四个标量与范围，Optuna TPE，sampler seed = 训练 seed，每 seed 20 trial（r5 同成本、同样预写 20），三语料 × seed 234 / 2025 / 3407；`search.py`、`train.py`；启动 `launch/sbatch_search.sh <corpus>`（一个 GPU 上三 seed 并行）。输出 `runs/20261101_r6_length_table/<corpus>/seed<s>/`。
- 结果：待搜索完成。
