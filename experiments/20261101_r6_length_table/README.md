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

HateMM：Q2（不低于 r5）通过，AP +.011、ROC +.004；Q3（门）通过。

### 调用曲线与单调性（Q1；`launch/run_final_curves.sh`，uoa-lab1 11-02 07:40–07:44；`runs/20261101_r6_length_table/final_curves/<test_selected|val_selected>/<corpus>/summary.json`；验收 `experiments/20261101_r5_completion/accept_check.py <root> <corpus> final`）

HateMM，test，三 seed 均值 AP / ROC / within：

| 固定次数 | 0 | 1 | 2 | 4 | 8 | 16 | 32 |
|---|---|---|---|---|---|---|---|
| test 选 trial | .568 / .801 / .771 | .647 / .851 / .771 | .667 / .865 / .778 | .687 / .881 / .780 | .702 / .889 / .783 | .690 / .885 / .791 | .693 / .879 / .793 |
| validation 选 trial | .574 / .807 / .740 | .620 / .840 / .738 | .641 / .851 / .741 | .653 / .861 / .745 | .670 / .876 / .755 | .674 / .878 / .763 | .681 / .875 / .774 |
| 全部 60 个 trial 平均（不选） | .573 / .800 / .746 | .623 / .838 / .744 | .638 / .851 / .748 | .658 / .868 / .757 | .670 / .878 / .764 | .673 / .879 / .771 | .679 / .876 / .779 |

qmixE1 预算 1–8（test 选）：AP .648 → .679 → .683 → .687 → .689 → .697 → .701 → .702，ROC .854 → .889，within .771 → .787 → .779（预算 7）→ .783。

- within 随调用上升（r5 的 HateMM within 回落已消失）。
- test 选 trial：8 次之后 AP 降 .012（16 次 .690），32 次 ROC 比 8 次低 .010；验收不过。validation 选 trial：只差 32 次 ROC 比 8 次低 .001；全部 trial 平均：AP 一路升，ROC 32 次比 8 次低 .002。
- 原因：规则 7 按 test 固定 8 次的 (AP + ROC) / 2 选 trial，选中的 trial 正好在 8 次这一点偏高；不经 test 选择时曲线不在 8 次后下降。同理，validation 上（epoch 按 validation 8 次选）全部 trial 平均在 8 次后下降（AP .739 → .703）。

### 测试期变体与诊断（`launch/run_sim_r6.sh`，test 选 trial，uoa-lab1 11-02 07:48–07:51；`runs/20261101_r6_length_table/sim_monotone/<corpus>/<trial>/summary.json`；评测核心 `hc.frame_metrics`，within 与统一评测器差 ≤ .005）

HateMM，三 seed 均值，固定 0 / 4 / 8 / 16 / 32 次 AP / ROC / within：

| 答案 | 0 | 4 | 8 | 16 | 32 |
|---|---|---|---|---|---|
| 真实答案（第 6 版本身） | .568 / .801 / .770 | .688 / .882 / .781 | .702 / .889 / .788 | .692 / .888 / .798 | .697 / .882 / .804 |
| 真实答案 + 测试期复制似然 | 同上 | .687 / .876 / .784 | .699 / .884 / .796 | .701 / .887 / .795 | .699 / .879 / .806 |
| 按模型自己的表模拟答案 | 同上 | .746 / .914 / .802 | .808 / .941 / .845 | .862 / .961 / .882 | .906 / .975 / .918 |
| 真实答案跨视频打乱（同真实状态 × 长度档） | 同上 | .729 / .911 / .778 | .770 / .930 / .793 | .809 / .946 / .809 | .834 / .953 / .828 |

- 复制似然在第 6 版上不再需要：8 次 AP −.003、ROC −.005，只让 16 次的 AP 不低于 8 次。
- 打乱后（去掉同一视频内答案的相关）三项指标一路升到 32 次并通过验收；模拟答案同样通过。第 6 版剩下的 8 次之后不升，来自真实答案在同一视频内的相关，加上上一节所说的在 8 次处做的选择。

### 最强 baseline + 同样的 VLM 回答（P7，规则 14(f)；`experiments/20261101_r5_completion/baseline_plus_vlm.py`，用上表"真实答案"那一行实际问到的节点；`runs/20261101_r6_length_table/baseline_plus_vlm/<corpus>_r6_k8.json`）

| 语料 | baseline 本身 | 只用 VLM | baseline + VLM（test 最好 α） | 本方法 8 次 |
|---|---|---|---|---|
| HateMM（MACIL-SD） | .573 / .807 / .595 | .495 / .797 / .619 | .643 / .861 / .647（α .7） | .702 / .889 / .783 |

HateMM：给最强 baseline 同样的 8 个回答，仍比本方法低 AP .059、ROC .028。
