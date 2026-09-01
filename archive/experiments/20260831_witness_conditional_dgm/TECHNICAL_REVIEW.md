# Technical pre-run review

日期：2026-08-31  
结论：**FAIL — 修复以下三项后才能启动 formal pilot。**

本评审只覆盖实现与 evaluation result chain，不裁定 novelty 或研究流程。未运行 formal/GPU，未修改方法实现。

## Blockers

### 1. 当前 HateMM prediction cohort 必然使 canonical evaluation 失败

`train.py` 当前直接使用完整 `hatemm_test.txt`，共 215 个视频；canonical test gold 只有 214 个，固定无 localization gold 的 `hate_video_427` 会被写入 `scores.jsonl`。`evaluate.py` 启用了 `--require-full-coverage`，因此会把该视频判为 score-not-in-gold 并终止，HateMM arm 无法产生 `metrics.json`。

必要修正：producer 使用 label/GT-blind 的共享 `scoped_video_protocol.evaluator_test_ids(...)` 从 test manifest 得到固定 214-video cohort；不得重新读取 GT 数值来过滤。

### 2. Test labels 仍在训练前被读取并装入 test dataset

删除训练前 GT 过滤后，`train.py` 仍在训练开始前调用 `hdata.load_labels(args.corpus)`，该映射包含 test label；随后又在训练前以该映射构造 `test_loader`。虽然训练循环没有迭代该 loader，当前实现仍不是 README 声称的 blind post-checkpoint prediction，且 `predict()` 实际从 test loader 取出了 label tensor。

必要修正：训练与 validation 只加载各自 scoped labels；选定并载入 best checkpoint 后才构造 test loader，test dataset 使用 dummy labels或完全不返回 labels。Test split membership可以读取，test label values 不得进入 producer。

### 3. `>= .020` mechanism gate 检查了错误的 gain 集合

README 冻结的是：core 相对 capacity-matched anchor 在两个语料都提高，且至少一个 corpus 的 **core-minus-anchor** within gain `>= .020`。`summarize.py` 却把四个 gain（两语料各自对 anchor 和 source control）混在一起取 `max(gains)`；只要 core 对 source control 的提升达到 `.020`，即使对 anchor 两边都不足 `.020`，该 gate 仍会错误通过并可能输出 `EXPAND`。

必要修正：`at_least_one_core_gain_ge_020` 只能在两个 `core - anchor` gain 上取最大值；core 对 source control 仍由“双语料均严格胜出”的独立 gate 裁定。

## 已核验通过的部分

- `anchor`、`source_dgm`、`witness_dgm` 使用同一模型、loss、optimizer、seed、batch order、capacity 与 checkpoint-selection 路径；差异仅为 competence 定义和由此得到的 gradient coefficient。
- Witness competence 正确实现 positive top-K-minus-rest 与 negative one-minus-top-K，并在两类同时出现时按类别等权；计算全程 stop-gradient。
- Gradient modulation 在 backward 后、optimizer step 前实际缩放对应 modality branch 的全部梯度，并按 `visual/audio/text` 的真实拼接顺序缩放 fused 第一层的对应输入列；现有三项单元测试均通过。
- Validation 只使用 fused video score AP 选择各 arm 自己的 checkpoint；test prediction 在 best state 恢复后生成，且只导出 raw `score_fused` 作为正式评测 branch。
- `evaluate.py` 只调用仓库 canonical evaluator，并启用 exact coverage；没有复制三项 test metric 实现。
- `run_pilot.sh` 的串行失败传播和外部 `nohup`/`setsid` 兼容性正常；任一 train/evaluate 失败都不会进入 summary。
- Smoke artifact 完成了 1 epoch、214-video HateMM prediction 和 canonical evaluation，所有四个 score branches finite；但它来自删除 GT 过滤前的旧路径，因此不能解除 blocker 1/2。
- 未发现禁止的内容校验操作。Python 与 shell 语法检查通过。

## Blocker closure（2026-08-31）

结论：**RESOLVED / PASS — 原三项 blocker 均已关闭，可启动 formal pilot。**

1. **HateMM cohort 已关闭。** `train.py` 现通过共享 `evaluator_test_ids(...)` 从 test manifest 得到固定 evaluator cohort，不读取 GT 数值。`smoke_fix_hatemm` 实际写出 214 个唯一视频，不含 `hate_video_427`；canonical evaluator 报 214 videos、29269 frames、missing 0、extra 0。
2. **Test-label 隔离已关闭。** Train/validation 分别通过 `scoped_video_labels` 加载各自 labels；test 使用全零 dummy labels，且 test loader 只在 best state 选定并载入后构造。当前 producer 路径不读取 test label values 或 temporal GT。
3. **Mechanism gain gate 已关闭。** `summarize.py` 现将“双语料均胜过 anchor 和 source control”与“至少一个 `core-minus-anchor >= .020`”分开计算；后者只在两个 anchor gains 上取最大值。

Closure 仅复核原三项，没有扩展审查范围。`smoke_fix_hatemm` 为 1-epoch witness arm，checkpoint selection、blind prediction、canonical full-coverage evaluation 链均完整；相关 Python 语法与文档 diff 检查通过。
