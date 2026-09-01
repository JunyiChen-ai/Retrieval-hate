# Process Review RESET2 — 2026-08-31

范围：只读审查 `research-wiki/STATUS.md`、`RESEARCH_ITERATION_RULES.md`、failure ledger，以及上一 epoch 的 counterfactual carrier-alignment、ambiguous-point completeness、lexical-gated messenger 三候选。未审查代码，未设计新候选。

## 裁定

`RESET`。

三个候选均未实现、未训练、未生成 prediction，也未进入 HMM/HCS 双 test；三者来源门均通过或窄通过，全部卡在 Gate 3。瓶颈是 pre-implementation gate，而不是算力、代码或 premise。

Rule 14 已允许最低 observation evidence，但 ledger 仍要求实现前完整证明 constant/broadcast、position 和 video identity 解均不成立，实际覆盖了 Rule 12。三候选又连续把同一 lexical evidence 放入 alignment、latent completeness 和 messenger gate，形成 source/placement churn，并偏离 STATUS 已由 test 证实的 MultiHateLoc modality-selection/fusion failure。

## 流程修正

1. Ledger 不再是独立 identifiability 硬门，只拒绝与已关闭机制严格同构且无新增约束，或核心项代数上不能进入 final score 的候选。
2. Rule 14 只判断 observation；Rule 12 只判断来源占用、non-trivial task adaptation、机制故事和可证伪预期。一般 shortcut 风险由端到端 test 和 matched control 判断。
3. Pre-run STOP 限于来源已占用、直接套用/组件拼接、严格失败等价且无新增约束、必要监督条件明确缺失。
4. 候选 brief 最多一页，先锁定 STATUS 已证实的 failure；通过 premise 与 novelty 后立即实现，只做一次 technical review，然后独立训练 HMM/HCS 并立即 test。
5. 停止 raw-statistic、新 lexical producer和新 teacher sweep；不得重开本轮三个失败候选，也不得让一个 premise 连续支撑仅改变 head/loss/proposal/fusion 位置的候选。

## 证据裁定

充分：MultiHateLoc starting point、固定 test 指标与门槛、四语料 modality-selection/fusion failure、HMM/HCS lexical locality、generic smoothing/teacher KD/ensemble/broadcast/branch-bypass失败链。

缺失：通过窄 novelty gate 后真正 load-bearing 的端到端方法双 test，以及相应 final-score intervention evidence。

方向：三个近期候选 `STOP`；继续筛 raw statistic/new producer/new teacher `STOP`；lexical locality仅作为已有诊断 `CONTINUE`；MultiHateLoc modality-selection/fusion failure作为开发目标 `CONTINUE`；整体研究 `RESET` 后继续。
