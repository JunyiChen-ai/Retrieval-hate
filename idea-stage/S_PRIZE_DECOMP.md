# S_PRIZE_DECOMP — 立场错误桶奖励的可回收 / 争议分解

**日期**: 2026-08-17
**目的**: 为「是否出钱做 750 条人工立场标注」定价。
**成本**: 0 元。只读已有文件,无训练,无 API 调用。CPU 上重跑一次 oracle 计分脚本。

---

## 0. 规则冻结(写在算任何数字之前)

本节在读取任何面板投票分布、任何重算结果之前写定并提交(commit `PENDING-A`)。
后续小节只填数,不改规则。

### 0.1 面板构成

对 `r5_buckets.json` 里 49 个 S(立场 / use-vs-mention)错误项,构造 4 票的「独立判断者面板」:

| 投票者 | 来源 | 输入 | 口径 |
|---|---|---|---|
| r7k | `claude_stance_gate/annot_r7k/*.jsonl` | 转录 + 8 帧(有帧数据集) | 原生二值 ENDORSE / DISTANCED |
| m3q | `claude_stance_gate/annot_m3q/*.jsonl` | 同上 | 同上 |
| z9x | `claude_stance_gate/annot_z9x/*.jsonl` | 同上 | 同上 |
| qwen-R2 | `mask_stance_pilot/pred_m1.jsonl` | 遮蔽转录 + 8 帧 | 五类二值化:`endorses` → ENDORSE,其余四类 → DISTANCED(沿用 `contrast_stance/score_contrast.py::binarise_5way`) |

不入面板的两轮及理由:
- **qwen R1**(`stance_pilot/pred_strong.jsonl`)二值化 0.500,劣于 R2 的 0.5625;每个模型只取其最好一轮,避免同一模型占 2/5 票。
- **qwen R3**(`contrast_stance/pred_c1.jsonl`)是近乎常数的 ENDORSE 预测器(S_FP 1/18,S_FN 14/14),按任务指令不计入。

金标口径沿用 `claude_stance_gate/score_gate.py`:S_FP 的金标 = DISTANCED,S_FN 的金标 = ENDORSE。

### 0.2 三分类定义

设某项有效票数 n(缺票 / 未解析票不计入 n),其中支持金标方向 k 票:

- **可回收项(RECOVERABLE)**:n = 4 且 k ≥ 3;或 n = 3 且 k ≥ 2。
- **争议项(CONTESTED)**:n = 4 且 k ≤ 1;或 n = 3 且 k ≤ 1。
- **分裂项(SPLIT)**:n = 4 且 k = 2(2-2)。
- n ≤ 2 的项:不存在(检查后若出现,单列 `INSUFFICIENT` 并在奖励重算中按争议项处理,即不计入可回收)。

`MHC_zh/BV1m8411z7mV` 因 DashScope 内容审核拒答而无 qwen 票,n = 3,按上式处理。

### 0.3 奖励重算口径

复用冻结的 `r5_bucket_value.py` 计算方式,不改任何一行逻辑,只改选择集合:
同一份 round-4 最优 ensemble 比较器(HateMM/mlp、MHC/mean_logit、MHC_zh/logistic、
ImpliHateVid/logistic)、同一份 seed 预测重建、同一份 test 标签,把
`z[sel] = y[sel]` 的 `sel` 从「该数据集全部 S 项」换成「该数据集的可回收 S 项」,
重算 macro-F1 增量。base 与 §9.2 的 base 必须逐位相同,否则本节作废。

**为什么争议项不计入**:争议项上,面板 4 个独立判断者里 ≥3 个从内容里读出的立场方向与金标相反。
若未来的立场方法真的做对了立场判断,它在这些项上会输出与金标相反的方向,
监督信号会把模型往错误方向推,而不是修好这一项。因此「完美立场方法」的可达上限只包含可回收项。
分裂项按「一半」不做加权处理——单独列出,给出「含分裂项」与「不含分裂项」两个数,
主表用**不含分裂项**的保守数。

### 0.4 定价判定线(现在冻结)

以 4 数据集**均值**的可回收 oracle 相对 §9.2 原 S 桶 oracle(mean +6.46)之比 R:

- R ≥ 0.60 → **值得**做 750 条人工标注。
- 0.40 ≤ R < 0.60 → **边缘**。
- R < 0.40 → **不值得**。

比值用「不含分裂项」的保守数计算;含分裂项的数作为敏感性一并报告,不改判定。

---
