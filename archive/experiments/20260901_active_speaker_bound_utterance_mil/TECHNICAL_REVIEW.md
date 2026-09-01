# 跑前基础 technical review

截至 2026-09-01。审查范围仅限会改变正式实验观察或结论的 bug；未审代码风格、重构、一般工程健壮性、理论、novelty 或流程，也未运行 smoke、训练或子集实验。静态 Python compile 与 shell syntax 检查通过。

## 裁定

**PASS**

唯一一轮基础审查发现的两个 result-affecting defect均已定向修复并确认。可以启动正式 cache 生成、完整 validation search、checkpoint selection 和 HMM/HCS test evaluation；不需要再开启代码审查。

## 已修复并确认

### 1. source-bound unit 的共同 midpoint 映射：已修复

位置：`dataset.py::load_source`，并与 `powa_macil.dataset::aligned_text` 的重采样共同发生。

`dataset.py::load_source` 现在用同一个 destination midpoint index 同时选择 face、categorical state 和 relation-only raw 1 fps utterance。训练阶段三者随后从相同长度序列用相同的 deterministic uniform index处理；validation/test阶段三者保持同一原始 grid。因此 relation row 不再产生跨秒 `face(i) × utterance(i+1)` 交叉项。

`method.py` 的 relation明确使用新传入的 `source_utterance`；原POWA `f_t` 仍走既有 interval-resampled text path，没有被这项修复改写。`train.py` 与 `infer.py` 已将同一 source utterance tensor送入 core、permuted 和 anchor harness。

### 2. shared-evaluator JSON 层级：已修复

位置：`summarize.py::compact`。

`summarize.py::compact` 现在从 `payload["results"]["score_method"]` 读取 `pr_auc`、`roc_auc` 与 `per_video.macro_auc`，并在branch缺失时显式失败。这与 `run_formal.sh` 传给共享 evaluator 的 `--branch score_method` 一致，可以产生 mechanism gate 与 six-SOTA gate summary。

## 已确认无阻断问题

- producer 不读取项目 label/GT；train、validation、test frozen manifests 两两无重叠。
- TalkNet 的 track frame index 是零基 25 fps；producer 的 `frame // 25` 与 `pyframes/{frame+1}.jpg` 映射一致。多脸可判定秒中，permuted arm 使用同秒另一条可见 track；single-face 和 null/offscreen 秒保持不变。
- relation branch 确实加到 POWA `shared` representation，随后进入 primitive/policy path 和唯一 raw `frame_prob`；`relation_weight=0` 保持同 harness anchor score path。
- validation 完成 2 个 same-LR anchor 与 6 个 core trial，按 validation checkpoint/超参数选择；permuted control 固定使用所选 LR/weight 并独立用 validation 选 checkpoint。test 不参与梯度或 checkpoint selection。
- test inference 调用共享 evaluator并启用 `--require-full-coverage`。当前 frozen cohort 可完整覆盖：HateMM 214 个 gold test 视频（manifest 中额外一个无 gold ID 在 infer 前排除），HateClipSeg 79 个 gold test 视频。

## 正式启动后定向修复确认

首次 formal validation 在 `SourceTestDataset.__getitem__` 触发 `ValueError: too many values to unpack`：wrapper 按 5 项解包当前返回 6 项的 `PowaTestDataset`。该错误会阻断 validation，属于本审查允许处理的结果影响型接口对齐 bug。修复仅将解包改为 `(f_v, _, _, _, n_seconds, vid)`；真实 HMM validation 样本随后确认 wrapper 返回训练代码所需的 9 项，5-crop source tensors 与 `index_map/n_seconds` 对齐。没有重新开启泛化 review；修复后完整 validation 与 test 正常完成。失败日志保留为 `runs/20260901_active_speaker_bound_utterance_mil/formal_seed234/run_failed_dataset_unpack.log`。
