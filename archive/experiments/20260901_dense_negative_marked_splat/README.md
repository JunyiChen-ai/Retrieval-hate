# Dense-Negative Marked Temporal Splat

截至 2026-09-01。RESET3正式方法 3；不运行premise，不改变novelty、renderer或配置空间。

现有test error artifact显示HMM正帧对负视频帧separation几乎不胜anchor，而标准top-K negative bag BCE只对负视频最高若干秒反传。弱监督下negative video的每一秒都可被确定为benign；本轮唯一result-relevant change是在原MIL loss外，对negative bags全部有效秒施加dense negative BCE，固定权重`1.0`。Positive bags、duration field、noisy-OR、top-K、smoothness和contrastive loss不变。

使用上一正式轮由validation选择的训练配置；本轮validation只选checkpoint。HMM/HCS checkpoint都锁定后才生成test prediction并调用canonical evaluator。若失败，RESET3达到`3/3`，立即停止并新开独立process review。

权威输出：`runs/20260901_dense_negative_marked_splat/formal_seed234/`。

## Formal result and decision

HMM/HCS validation分别选择epoch 15/14；两边checkpoint锁定后才生成test prediction。权威汇总为`runs/20260901_dense_negative_marked_splat/formal_seed234/summary.json`。

- HMM AP/pooled ROC/within为`.516654/.760930/.714220`，相对validation-selected base为`+.000568/+.004705/-.008984`。
- HCS为`.549983/.535196/.533888`，相对base为`-.022302/-.015400/+.001052`。

双数据集performance gate失败。Dense-negative只给HMM pooled带来极小改善并损失within，HCS pooled明显下降且within基本不变；固定机制停止，不调权重。RESET3窗口达到`3/3`，累计连续performance failure=`6`，立即停止新方向并触发新的独立process review。
