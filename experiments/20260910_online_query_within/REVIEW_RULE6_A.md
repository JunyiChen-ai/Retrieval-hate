# 规则 6 代码复核：第 4 轮修正版（训练期跨视频分配，小轮分配 + reveal=False）

日期 2026-09-14。复核者：独立 fable 子 agent，只读；对象 = 未提交的工作树改动（`acquire.py` 的 `reveal` 参数，`train.py` 的 `acq_rounds` 小轮分配），对照 commit f95ec23（第 3 轮）与 40569be（第 4 轮首版）。

## 结论

无 must-fix，无 should-fix。修正后 `reveal=False` 路径里对 `bf_true` 的读取只剩已在 `allowed[v]` 里的窗（`acquire.py:157`），选中窗的裁定不读（`:225-226` 在读取前 break）；揭开只发生在 `train.py` 分配之后（`:443-448`），每次揭开都计入。

## 逐项

1. **无未计入的裁定读取**：`ScaffoldCache.build` 每次按传入向量重算（`src/hier_evidence_common.py:173-180`），`make_masked_scaffold_fn` 只读粗块裁定（`:543-562`），HMM 发射跳过 MISSING（`src/interval_evidence_hmm.py:345-354`），HMM 内无按视频存的裁定；EOC 候选循环只用假设值 0/1（`acquire.py:207-212`）；`weight="hmm"` 的 `pred_fine` 来自后验，不来自真值。
2. **小轮重打分**：`initial={v: sorted(allowed[v])}` 在小轮循环内构造（`train.py:434-437`），第 2 小轮看到第 1 小轮揭开的窗；两小轮共用事件前的 HMM，事件后重拟合一次（`:454`）。
3. **`allocate_global` 单步**：每视频最多 1 个；预算 `N // 2` 与 `N // 2 + N % 2` 之和 = N；每次事件恰揭开 N 个窗。
4. **`per_video` 臂**：`n_rounds=1`，单步 EOC argmax 加入 `allowed` / `policy_order`，与第 3 轮代码（f95ec23 train.py 383-391）产生相同集合与顺序；`run_video` 事件后的 `scores` 训练里无人使用。
5. **其它消费者**：`acq_lookahead` 已无引用；`search.py:95` 对未知键报错；`screen.py`、`calls` 字典、`acq_log` 字段与 `say` 引用的变量均存在。
6. **评测路径**：`reveal=True` 下与提交版只差 `picks.append` 位置，数值不变。
7. **随机种子**：rng 只在 `random` 策略用到，`eoc` 不受影响。

## 备注（未改动代码语义）

- `acquire.py` 模块 docstring 的"每步之后骨干打分"只对 `reveal=True` 成立 → 已补一句说明。
- `allocate_global` 增益并列时按 `train_ids` 顺序稳定排序，确定性但依赖列表顺序。
- `runs/20260910_online_query_within_it4/*/trial*/config.json` 仍含 `acq_lookahead`（作废运行的输出快照，不是输入）；磁盘上无 `hparams.json` 含该键。
