# Universal teacher simplex diagnostic

> 最终裁定：诊断 gate PASS；knowledge-amalgamation novelty gate STOP。本目录随后归档，不是方法迭代或 SOTA method claim。

截至 2026-08-31。只读 developmental test error analysis，不是方法、ensemble baseline 或可部署 prediction。

## 问题

固定 pair diagnostic 已分别找到 HMM lexical+POWA 与 HCS lexical+VERA 的 all-SOTA test blends，但二者 teacher/weight 不同。普通 blend 和按 corpus 选择 teacher 被禁止作为主方法；在审查 single-student knowledge-amalgamation adaptation 前，先判断是否存在一组 **完全相同** 的共享 teacher weights，使 HMM/HCS 同时超过三项 SOTA。

## 冻结诊断

- 信号固定为 lexical、POWA、VERA、MultiHateLoc seed 234 已有 test predictions。
- 每个信号在各语料完整 test frame pool 上做 empirical-CDF rank normalization，与既有 complementarity diagnostic 相同。
- 扫描 0.05 simplex：四个非负权重和为 1，共 1771 个点；同一 weight tuple 同时应用于 HMM/HCS。
- 每点调用唯一共享 evaluator，报告 pooled AP、pooled ROC、within-video macro ROC。
- Gate：至少一个完全相同的 tuple 在两个语料都同时通过三项 SOTA。失败则关闭统一 teacher knowledge-amalgamation 路线；不得按 corpus 选权重。

输出：`runs/20260831_universal_teacher_simplex_diagnostic/main/metrics.json`。

## 正式结果

正式 test diagnostic 扫完两个语料各 1,771 个 simplex 点：HateMM 有 717 个 all-SOTA 点，HateClipSeg 有 245 个；其中恰有 7 个完全相同的权重 tuple 在两个语料都同时超过 pooled AP、pooled ROC 和 within-video ROC 门槛。一个 margin 较均衡的共同 tuple（权重顺序为 lexical、POWA、VERA、MultiHateLoc）是 `[.10,.25,.40,.25]`：

- HateMM AP/ROC/within = `.600986/.827174/.666756`；
- HateClipSeg AP/ROC/within = `.630876/.619652/.566492`。

独立 post-run reviewer 从声明的 score path/branch 重算全部 3,542 次共享 evaluator 调用，所有 row、passing set、joint tuple、coverage 和指标与 artifact 完全一致，最大数值差为 0。该结果证明统一 teacher 信息存在跨语料 headroom，但它来自 per-corpus test-frame ECDF 和 test-grid 搜索，只能作为 iterative/developmental inference upper bound；没有训练 student，也不能把 blend 称为方法。

## Novelty 裁定

两份独立 review 均为 `STOP`（最强候选评分约 `2.5/10`，另一份严格检索结论为 `NONE`）：

1. 允许 adaptation 既有 knowledge amalgamation，这一门本身可通过；
2. 但 heterogeneous/multi-teacher knowledge distillation 已进入 hateful/harmful video detection，且“多异构视频 teacher 聚合后蒸馏为单一 frame-localization student”也已进入弱监督视频异常定位，来源/core 未满足未占用要求；
3. 把四条同目标 scalar score 用共享 latent、teacher decoder、ECDF 共识或 listwise loss训练成单 student，本质仍是普通 multi-teacher KD 加辅助重构，未形成新的 hateful temporal localization 识别机制，并退回已淘汰的 fixed-blend distillation 路线。

最接近的 primary work：AAAI 2019 [Knowledge Amalgamation](https://ojs.aaai.org/index.php/AAAI/article/download/4165/4043)、WACV 2025 [DAKD](https://openaccess.thecvf.com/content/WACV2025/html/Dalvi_Distilling_Aggregated_Knowledge_for_Weakly-Supervised_Video_Anomaly_Detection_WACV_2025_paper.html)、Information Fusion 2026 [MVKD](https://www.sciencedirect.com/science/article/pii/S1566253526006111)、Findings ACL 2026 [LEAF](https://aclanthology.org/2026.findings-acl.604/)。因此不实现 student、不扩展到 MHC、不把这一 diagnostic 当作方法迭代。
