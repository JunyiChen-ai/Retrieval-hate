# EXPERIMENT TRACKER — 裁定条件化密度估计（2026-09-05 建；全部未开始）

| Block | 项 | 状态 | 输出路径（约定） | 备注 |
|---|---|---|---|---|
| A | A1 posterior-alone 七行 × 两语料 | 未开始 | `runs/<exp_id>/label_model_gate/<corpus>/<model>/metrics.json` | CPU；用统一评测器 |
| A | A2 q_v 比例有效性 | 未开始 | `runs/<exp_id>/label_model_gate/<corpus>/proportion.json` | K2 门 |
| A | A3 合成恢复 | 未开始 | `runs/<exp_id>/label_model_gate/synthetic/` | 报告 |
| A | A4 OOF 后验缓存 | 未开始 | `runs/<exp_id>/label_model_gate/<corpus>/oof/` | 不写 `data/` |
| B | B0–B5 精确对照 | 未开始 | `runs/<exp_id>/controls/<corpus>/seed<seed>/<arm>/` | B0/B1 须 bit-match |
| C | 111 搜索 seed 234 / 2025 / 3407 | 未开始 | `runs/<exp_id>/<corpus>/seed<seed>/` | 规则 7 |
| D | 2^3 八格 + 对照臂 ×3 流 | 未开始 | `runs/<exp_id>/ablations/<corpus>/seed<seed>/<arm>/stream<k>/` | 规则 14(g) |
| E | 追加 seed | 未开始 | 同 C/D | 可选 |

规则 6 code review、规则 4 proposal review：未做（本轮只到 idea 阶段）。
