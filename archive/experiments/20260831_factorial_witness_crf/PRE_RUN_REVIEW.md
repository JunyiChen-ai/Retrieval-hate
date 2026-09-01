# Formal pre-run review

截至 2026-08-31。审查对象为本目录的 `README.md`、`model.py`、`protocol.py`、
`train.py`、`predict.py`、`evaluate.py`、`test_model.py`、`run_pilot.sh` 和
`launch_pilot.sh`，以及它们直接调用的共享数据与评测入口。

## 裁定

**PASS，可以启动已登记的 HateMM/HateClipSeg × core/zero_transition/collapsed
首轮 pilot。** 两个会改变训练含义的 blocker 已在本次 review 中修复；修复后的数学测试、梯度
测试、split isolation、test cohort 和脚本静态检查均通过。没有启动正式训练，也没有生成正式
prediction。

这个 PASS 只表示实现忠实且足以执行首轮淘汰 gate。`core > zero_transition` 与
`core > collapsed` 可以分别初筛 learned temporal transition 和 typed subset state 是否值得继续，
但不能构成完整论文归因。若 core 晋级，仍需 README 已登记的 independent-chain、
cardinality-only 和 parameter-matched controls。

## 已修复的 blockers

1. **null partition 曾错误删除 learned coalition prior。** 原实现直接把完整 emission tensor
   清零；这样 typed arms 的 input-independent coalition term 只存在于数据分区、不存在于 null
   分区，模型可单靠状态偏置移动 bag logit。现改为把零 unary 重新送入 `_emissions`，保留与数据
   分区完全相同的 coalition 与 transition prior。任意视频长度、非零初始化 coalition cost 下，
   zero-unary bag logit 现在为零。
2. **coalition 参数原为无约束 signed reward。** README 定义的是抑制多模态无条件共同激活的
   coalition cost；原实现却允许参数变为正 reward，可能重新鼓励 all-modal broadcast。现改为
   `softplus` 非负 cost，并从 typed-state emission 中扣除，初始 cost 为 0.1。
3. **structured attribution 的实现与文档不一致。** README 声称输出各 modality bit posterior，
   原实现只输出 union active posterior。现已按每个 bit 对对应 states 的 forward-backward mass
   求和；`forward`、`predict.py` 和所有 `_one_video` 调用方已同步。collapsed control 输出单列
   active posterior，不伪造三模态 attribution。
4. **运行环境入口不够明确。** `run_pilot.sh` 已固定使用 `HateVideo` 环境的 Python，并在入口
   检查其可执行性，避免 detached shell 因环境未激活而调用错误解释器。

## 数学、shape 与梯度

- `_positive_partition` 的状态语义正确：初始化只允许首帧非空；递推的 `continued` 覆盖已经访问
  过非空 state 的路径，`first_active` 只添加此前一直为空而当前首次非空的路径。空 state 的
  emission 与 `0→0` transition 均为零，因此不缺失此前空前缀能量。
- core、zero-transition 与 collapsed 的 positive partition 均与短序列全路径枚举一致。
- backward recursion 与 full forward mass 的组合，对任一当前非空 state 求和后再除以 positive
  partition，正好是“该时刻 active，且整条路径属于 positive path set”的 posterior；active
  posterior 和 typed bit posterior 均与全路径枚举一致。
- null 分区使用相同 state space、coalition cost、transition 和长度，只把 unary evidence 置零。
  这也使 zero-transition typed arm 的 null 回到其登记的路径计数基线。
- 三个 arms 的 synthetic batched full-model forward、BCE backward 均 finite；core 的 switch 与
  coalition 参数、zero-transition 的 coalition 参数、collapsed 的 switch 参数均获得 finite
  gradient。zero-transition 的 switch 参数和 collapsed 的 typed coalition 参数按 control 定义
  不参与计算。
- 变长 batch 只对每个样本的 `length` 前缀运行 DP，padding 不进入 partition 或 score；输出长度
  与输入有效长度一致。

## Controls 与训练隔离

- `run_pilot.sh` 对两个语料和三个 arms 分别调用一次 `train.py`，每个 arm 新建模型、optimizer、
  checkpoint 与输出目录，不存在同 checkpoint inference ablation。
- core 与 zero_transition 共享 typed emissions 和有效参数结构，仅后者固定 transition 为零；这是
  首轮判断 learned transition 是否 load-bearing 的直接 control。collapsed 是同一 frame-local
  modality encoder 下的二状态 dynamic-MIL control，可初筛 8-state typed subset 是否超过 generic
  temporal MIL。
- train、val、test 四语料实查两两交集均为零。train/val labels 只能通过
  `supervised_split` 读取；该入口拒绝 test。validation video AP 只用于当前独立训练内选择 epoch，
  不进入 loss、方法比较或 test producer。
- `predict.py` 使用 `blind_test_split`：只读取 frozen test IDs，labels 为全零占位；其 import 链不
  加载 temporal GT 或 test video labels。checkpoint 只携带模型参数、arm、语料与所选 epoch。
  temporal GT 只在 prediction 完成后由 evaluator 读取。

## Evaluation 与 cohort

- `evaluate.py` 只转调 `scripts/reproduction_baselines/eval_baseline_scores.py`，与
  `research-wiki/STATUS.md` 登记的全仓库唯一共享 evaluator 一致，没有复制指标实现。
- split 固定为 test，branch 固定为唯一 `score_core`，并强制 `--require-full-coverage`。四个主语料
  的 blind producer cohort 均已与 frozen evaluator GT keys 做集合比对，完全一致；HateMM 214、
  MHC-EN 158、MHC-ZH 153、HateClipSeg 79。
- evaluator 直接输出 pooled AP、pooled ROC-AUC、within-video macro ROC-AUC；shape 不一致、非有限
  score、缺视频或多余视频都会终止。首轮结果必须按 README 同时执行三指标 SOTA gate 与
  core-vs-controls within gate，不能只挑一个指标解释。

## 长任务与输出

- `launch_pilot.sh` 用 `setsid` 与终端会话解耦，在 pilot root 写 `run.log` 和 `run.pid`；
  `run_pilot.sh` 使用 `set -euo pipefail`，任一步失败都会停止，避免继续评测缺失或旧 prediction。
- 每个 corpus/arm 目录写 config snapshot、可读代码版本说明、selected checkpoint、完整 train
  history、blind scores 和 evaluator 原生 `metrics.json`。所有正式产物均位于 `runs/`，不会写入
  `data/`。
- 当前脚本不自动跳过已完成 arm；中断后应先看 root log 与各 arm 是否已有完整
  `metrics.json`，再决定是否以新的 run name 重启。首轮只有六个串行 run，这不是 correctness
  blocker，但不得在同一路径静默混用两次不同代码状态的产物。

## Review checks

以下均在 CPU、小型合成输入或只读元数据上完成：

- positive partition 对全路径枚举：三 arms 通过；
- active 与 bit posterior 对全路径枚举：三 arms 通过；
- nonzero coalition prior 下 zero-unary length normalization：长度 1/2/5/20、三 arms 通过；
- mechanism parameter gradient 与 full model BCE backward：三 arms 通过；
- Python compile、shell syntax、所有 `_one_video` 返回值调用方同步检查：通过；
- 四语料 split disjointness、blind placeholder 和 exact evaluator cohort：通过。

最终裁定：**PASS FOR FORMAL PILOT**。
