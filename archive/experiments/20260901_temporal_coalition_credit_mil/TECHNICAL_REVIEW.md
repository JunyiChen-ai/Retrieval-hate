# Independent technical review

截至 2026-09-01。裁定：**PASS**。

未发现会改变正式实验观察或结论的 blocker，可以启动正式 GPU 运行。

核对范围与结论：

- `alpha=0` 的初始化、forward、base loss 和单步 Adam 更新精确退化原 MultiHateLoc；HateMM 真实 smoke 的 anchor 第一个 epoch 的 loss、MIL、smoothness、contrastive 与官方运行逐值相同。
- 三模态 8-coalition Shapley 系数、positive-part 加 DMS fallback、正视频 base-fused top-K witness、aligned 与循环平移 control 均正确；shift 只改变责任 target 的时间对应。
- 14 trials 确为两个 anchor 加 `2 learning rates × 3 alpha × aligned/shifted`；每个 trial 在 validation 选择 checkpoint，每个 arm 再联合选择配置；pooled `.01` 约束和无可行配置时的 fallback 正确。
- 正式推理加载 selected aligned、它的 same-lr reference anchor、独立选择的 shifted；test 不参与训练或 selection。
- Evaluation 只调用 canonical evaluator并要求完整 coverage；summary gate 正确。
- 候选实现没有任何哈希计算、比较或依赖。
- 单元测试 `PASS`；三个 arm 的真实 1-epoch validation-only smoke 均完成且 responsibility witness 非零。
