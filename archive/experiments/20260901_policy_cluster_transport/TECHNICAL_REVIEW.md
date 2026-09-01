# 基础 technical review

截至 2026-09-01。审查对象为本目录正式运行代码及其直接调用的 POWA 数据、模型与统一评测入口。按项目规则，本文件记录正式运行前唯一一次基础 technical review；未运行 smoke 或训练，也未审代码风格、重构质量、一般工程健壮性、novelty 或理论充分性。

## Verdict

**PASS。** 发现的一个 result-affecting bug 已做最小修复，并已只针对该修复及原定链路完成确认。没有剩余的正式运行阻断项。

## 发现并修复的问题

原 `method.py` 用同一组 masked、temperature-scaled softmax 同时产生 detached cluster target 和预测分布。未触发 harmful-mass projection 时，该项等价于 `CE(p.detach(), p)`，对数据表示的梯度为零；negative bag 又只开放 background，因而 certified-background anchor 也几乎没有对比梯度。HateClipSeg 的 policy arm 开放全部 policy states，这会使其 snippet clustering/prototype 学习尤其容易退化，直接改变对机制和性能的观察。

最小修复位于 `method.py`：

- constrained、temperature-sharpened、harmful-mass-projected assignment 仅作为 detached transport target；
- prediction side 使用未温度化、未屏蔽的全状态 logits；
- 因而 negative background assignment 会压低其他状态，HCS positive assignment 也会训练 shared representation/prototypes，而不再是同分布自蒸馏零梯度。

`test_method.py` 增加了无训练回归检查，分别验证全 negative HMM batch 和全 positive HCS batch 对 `shared` 均产生非零梯度。该检查通过。

## 限定范围检查结果

- **机制进入 forward/loss/final score：PASS。** `shared_rep` 生成 cluster assignment；transport loss 以已选择的 weight 加入未改动 POWA loss；policy/permuted assignment 监督最终 compiler 直接读取的 `primitive_logits`，binary assignment 监督同一最终 `frame_prob`。测试推理只加载所选 POWA model 并输出单一 raw `frame_prob`，没有 transport/prototype inference branch 或结果融合。
- **controls 匹配与语义：PASS。** policy、binary、permuted 使用相同 POWA 参数规模、七个 prototypes、abstain 参数、训练 epochs、学习率、transport weight/temperature、数据与 checkpoint 规则。binary 把 positive harmful states 合并为 generic foreground target；permuted 仅循环移动六个 state-to-primitive targets，background 与 admission budget 不动。二者继承各语料锁定的 policy 超参数，只在 validation 内选择各自 epoch。
- **valid mask 与时间/crop/label 对齐：PASS。** 训练三模态使用同一 snippet bounds 和同一 deterministic length-200 uniform mapping，五个 visual crops 各自与相同 audio/text 时间行对应；loss 和 harmful-mass projection 均只统计 valid rows。validation/test 使用完整五 crop、同一 second-to-snippet `index_map`，并逐视频检查 score 长度等于 gold 长度。实际只读检查确认 HMM/HCS train、validation、test 无交集，validation/test 推理 cohort 完整覆盖各自 gold。
- **validation 超参与 checkpoint：PASS。** 每语料有 2 个 matched learning-rate anchors 和 12 个完整 policy trials；每 trial 只按 validation `within_roc -> pooled_ap -> pooled_roc` 选择 epoch。跨 trial 先要求相同 learning-rate anchor 的 pooled AP/ROC 非劣容差，再按 validation within 为主锁定超参数；没有读取 test prediction 或 test label。
- **正式运行顺序与路径：PASS。** `run_formal.sh` 先完成并锁定 HMM、HCS 两套 policy validation selection，再以各自锁定配置训练 binary/permuted；所有训练结束后才依次生成 test prediction。两语料使用各自 MACIL initialization、split、checkpoint 与输出目录，不共享训练模型。
- **统一评测器：PASS。** test 只调用 `scripts/reproduction_baselines/eval_baseline_scores.py`，使用 `--split test --branch score_method --require-full-coverage` 并直接写各 arm 的 `metrics.json`；未复制或改写评测逻辑。
- **静态确认：PASS。** Python 编译、shell 语法、全部命令行 parser、机制单元测试、split isolation、eval cohort coverage 以及正式输入/初始化文件可读性均通过。未运行任何训练或缩小试跑。

## 裁定

允许直接启动完整 validation hyperparameter search、checkpoint selection 和 HMM/HCS test evaluation。不得再以重复代码审查延迟正式运行。

## 正式运行时针对性修复

首次正式运行在前两个完整 anchor 结束、首个 policy trial 尚未完成时，harmful-mass 下界分支因
`torch.full_like` 的 fill value 为 tensor 而抛出 `TypeError`；没有生成任何 candidate/test结果。
该行已最小改为 tensor `expand_as`，并把原无训练回归测试的下界提高到必定触发projection分支；
测试通过。这里只确认该运行时修复，不重新开启泛化review。
