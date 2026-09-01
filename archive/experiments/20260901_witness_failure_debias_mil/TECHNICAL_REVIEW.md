# 跑前 Technical Review

**日期：2026-09-01**  
**裁定：PASS**

本次只检查会影响正式实验观察或结论的基础 bug；未审查代码风格、文档措辞、泛化防御或无关工程问题。未运行 smoke、缩小数据、缩减 epoch 或任何训练。

实际检查如下：

- 机制确实进入训练：非 anchor arm 同时训练 fused BCE、三个单模态 GCE bias experts，并把 witness-failure loss 加到唯一 fused scorer 的参数更新；test 只输出 raw `score_fused`。
- `anchor`、`uniform`、`relative` 三条路径匹配：`lambda_failure=0` 的 anchor 精确调用原 MultiHateLoc MIL loss；uniform 与 relative 保持相同模型、bias-expert loss、support、参数量和 schedule，唯一机制差异是 support 内权重。
- 梯度和 detach 正确：正例 top-K support、bias probabilities 和 relative weights均 detached；failure loss仍向 fused scorer反传，bias branch logits不从该权重接收梯度。
- support 正确：正视频只使用 detached fused top-K latent witnesses；负视频使用全部有效秒；padding 秒不进入任一 support。
- split isolation 正确：HateMM 与 HateClipSeg 的 train、validation、test 互不重叠；训练和 checkpoint selection 只读取 train/validation，test label不进入梯度或 checkpoint selection。
- validation 运行完整官方 epoch budget，并逐 epoch以 validation within ROC为主、pooled AP/ROC约束为辅保存最佳 checkpoint。
- checkpoint 到 test 的链条正确：`selection.json` 指向所选 trial 的 config、train log和 checkpoint；test inference按该 config重建模型并严格载入对应 checkpoint，没有重新选择。
- 14-trial 设计匹配：每语料为2个 anchor、6个 uniform、6个 relative；每个 uniform对应同 learning rate anchor，每个 relative对应同 learning rate、同 failure strength uniform；最终报告保留这条 matched chain。
- test inference 使用固定 evaluator-test cohort；HateMM为214个视频、HateClipSeg为79个视频。两语料三模态特征均可解析、shape正确、视频内长度一致，test长度与1 fps GT完全一致。
- test evaluation调用共享的 `eval_baseline_scores.py`，其三个固定指标统一落到 `scripts/duplex/frame_eval_common.py`；开启完整覆盖检查。

执行的非训练检查：实验 Python 静态编译、三个 shell 入口语法检查、`test_method.py` 单元测试、共享 frame evaluator self-test、两语料 split/coverage/shape/长度审计；全部通过。

未发现需要修改的 result-relevant blocker，正式完整 validation search 可以启动。
