# 跑前 Technical Review

**裁定：PASS**  
**日期：2026-09-01**  
**范围：只检查会改变实验观察或结论的基础 bug；未做 smoke、训练、缩数据或缩 epoch。**

检查结果：

- 机制确实进入唯一正式 `score_fused` 路径：逐秒 retain gate、跨模态 projection 和 donor mixture 先生成 substituted embeddings，再进入原 fused MLP/head；单模态分支仍保持原路径。
- `alpha_fusion=0, arm=anchor` 不构造额外参数，并与同初始化 MultiHateLoc 的参数、forward 输出逐项 exact；基础 MIL、smoothness、contrastive loss 与训练 epoch 配置未改变。
- Aligned donor 只取 recipient 同一有效秒；shifted control 仅将每个视频各 donor 在其有效长度内循环平移半个长度。Padding 不参与 shift、pooling、top-K、loss 或最终 score。
- Retain gates 与全部跨模态 projections 均处在 fused MIL 的可微路径；coverage loss 只约束正视频 detached fused top-K witness，outside budget 只读其补集。正 witness 和 outside 计数均受有效 mask 限制。
- Formal search 每语料恰为 2 个 anchor、6 个 shifted、6 个 aligned，共 14 个完整 trial。每组 aligned/shifted/anchor 的 learning rate 一致，aligned 与 shifted 的 fusion strength 一致，reference chain 可追溯且 selection 会拒绝数量或角色异常。
- 每个 trial 均运行该语料完整固定 epoch 数并逐 epoch计算 validation selection key；保存的 checkpoint 对应 validation 联合选出的 epoch。没有 validation gate 阻止随后 test。
- 训练只加载 train/validation scoped labels、validation gold；test 只在 selection 完成后的独立推理脚本读取。Test label 不进入 forward、梯度或 checkpoint selection；它只在预测已经产生后计算机制诊断。
- Test cohort 经固定 evaluator cohort exclusion 生成，最终调用冻结的共享 evaluator `scripts/reproduction_baselines/eval_baseline_scores.py`，开启 full-coverage，并由 evaluator强制逐视频 score/gold shape 一致及 finite score。

执行的非训练检查：

- `test_method.py`：PASS（anchor exact、aligned/shifted 区分、gate/projection 梯度、正 witness、有效长度内 shift、padding 保持）。
- 全部 Python 入口静态编译：PASS。
- 三个 shell 入口语法检查：PASS。
- `train.py`、`infer_selected.py`、`evaluate.py` 实际入口导入检查：PASS。

未发现会影响本轮实验观察或结论的 blocker；可以直接启动完整 validation hyperparameter search、checkpoint selection 与 HMM/HCS test evaluation。
