# 20261101_r6_length_table：第 6 版（答案表加共用长度斜率）

截至 2026-11-02。机制、依据、预注册判定见 `experiments/20261101_r5_completion/README.md` 第 2、3.2、3.3 节；规则 4 审稿 `experiments/20261101_r5_completion/REVIEW_RULE4_R6.md`。

- 改动：r5 的锚定两状态答案表加一条两状态共用的对数长度斜率（负例训练视频上估计，`qtl.qtree.fit_anchored_shared`），训练、EIG、后验、qmixE1 停止都用它；去掉复制似然。其余同 r5。
- 搜索（规则 7）：同 r5 的四个标量与范围，Optuna TPE，sampler seed = 训练 seed，每 seed 20 trial（r5 同成本、同样预写 20），三语料 × seed 234 / 2025 / 3407；`search.py`、`train.py`；启动 `launch/sbatch_search.sh <corpus>`（一个 GPU 上三 seed 并行）。输出 `runs/20261101_r6_length_table/<corpus>/seed<s>/`。
- 规则 6 代码审查 PASS（`experiments/20261101_r5_completion/REVIEW_RULE6_R6.md`）。

## 结果（test，固定 8 次 AP / ROC / within，三 seed 均值 ± 标准差；`runs/20261101_r6_length_table/<corpus>/seed<s>/study_summary.json`）

| 语料 | 机器、时间 | test 选 trial（规则 7） | validation 选 trial | r5（部署，含复制似然） | 门（规则 8） |
|---|---|---|---|---|---|
| HateMM | uoa-lab1，Slurm 1383，11-02 03:27–07:38 | .702 ± .006 / .889 ± .001 / .783 ± .015（trial 1 / 4 / 15） | .670 ± .016 / .876 ± .006 / .755 ± .043（trial 7 / 14 / 12） | .691 / .885 | .573 / .807 |
| HCS | uoa-lab1，Slurm 1384，11-02 07:38 起 | 进行中 | | .687 / .674 | .562 / .528 |
| DeHate | uoa-lab3，Slurm 1382，11-02 01:44 起 | 进行中 | | .262 / .780 | .174 / .701 |

HateMM：Q2（不低于 r5）通过，AP +.011、ROC +.004；Q3（门）通过。调用曲线与单调性验收（Q1）在 `launch/run_final_curves.sh` 跑完后补。
