# PRE-RUN REVIEW — typed REBA adaptation

**日期：2026-08-31**  
**最终裁定：FAIL；不得启动正式训练。**

审查范围包括实验目录全部源码，以及 `src/multimodal_video_data.py`、
`src/scoped_video_protocol.py`。未启动正式 GPU 训练，未生成正式 prediction。

## 根本机制 blocker

1. **class-aware BiAlign 不识别 time×modality ownership。** 当前 alignment 先用同一个 fused
   frame logit 对各模态做视频级加权汇总，再把 `(visual+audio)/2` 与 text 的 bag embedding 按二分类
   label 对齐。它确实把 batch 内所有同类样本设为 positives，因此没有同类 false negatives；但该目标
   不告诉 modality gate 在某个时刻应信 audio、visual 还是 text，也不约束 gate 与真实局部 witness
   对应。模型仍可让 gate 几乎总选 visual，同时通过视频级 embedding alignment 降低 loss，因而没有
   针对 MultiHateLoc 已证实 ownership 错配的可识别修复。
2. **BiAlign 可能退化为类内 bag-level collapse。** negative 视频彼此、positive 视频彼此都被视为
   multi-positives；当一个 batch 只有单一类别时，alignment loss 恒为 0。HateClipSeg train 为
   219 positive / 32 negative，随机 batch-8 下单类 positive batch 并不少见。即使 mixed batch 有梯度，
   它提供的仍是 binary class compactness，不是 temporal localization 或 modality attribution。
3. **scale-1 control 不能补足识别缺口。** control 在同一个由 full residual core 训练的 checkpoint 上
   做 inference ablation。它能测试 residual forward path 是否改变输出，但不能证明 residual experts
   在独立训练下优于 scale-1 模型，更不能把收益归因到 typed ownership。`alpha=0` 时 core 与 control
   精确相同，说明实现隔离正确；问题在归因强度而非 shape。
4. **adaptive pooling 没有固定 top-k，但仍只是普通 bag aggregation。** learned occupancy 只在 mean
   与 temperature-softmax pooling 间插值，所有有效帧都参与；它没有提供新的局部监督变量。因此它能
   改变极端帧依赖，却不能解决“哪个时间、哪个模态拥有 video label”的非识别性。

以上是算法概念层 blocker，不能通过 shape、数值或运行 plumbing 修复；需要新的局部 ownership
约束才可能重新评审。

## 审查中发现并已修正的代码/协议问题

- 原 core 含普通相邻帧平方平滑项。该项与项目已冻结的 generic smoothing 禁令冲突，且已有 test
  evidence 显示 HateMM/HateClipSeg 方向相反；现已从 model loss、训练参数和日志中完全删除。
- 增加 width、temperature、dropout、mask、空视频和 feature shape 的显式检查。
- config 增加可读的 split、scoped label、feature producer、共享 dataset/protocol 路径。
- README 增加 detached long-run、PID/log、resume 与 evaluate 命令；但由于本 review 为 FAIL，
  这些命令不得执行正式 run。

## 已确认无误但不足以改变 FAIL 的部分

- audio/visual/text residual expert 参数独立，shape 为 `[B,T,D]`；padding 输出为 0。
- `residual_alpha=0` 时 core 与 scale-1 control 精确一致；默认值下两者输出不同且有限。
- BiAlign 的 positive mask 为 label equality；同类 batch loss 为 0，没有同类 false negatives。
- alignment 对 projectors/temporal experts 有有限且非零梯度；frame-logit pooling weights被 detach，
  不会仅靠移动 attention 位置投机降低 alignment。
- pooling 源码没有 top-k、排序、分位数或固定 occupancy 选择。
- 四语料 train/validation/test ID 隔离，scoped train/validation video labels coverage 完整；训练 producer
  不调用 temporal gold 或全语料 label API。test prediction 使用 placeholder labels。
- evaluator-test cohort 与 frozen gold 精确一致；`evaluate.py` 唯一调用仓库共享 evaluator，并对三项
  指标使用严格 SOTA gate。
- epoch state、optimizer、best checkpoint、随机状态与 loader state支持原子保存和 resume；输出有
  training/prediction completion marker。

## 快速测试

已运行实验测试与语法检查。9/9 tests passed，覆盖 output/control shape、padding、BiAlign false-negative
边界与梯度、`alpha=0` control、无固定 top-k、短优化、split/scoped labels、exact cohort 和共享 evaluator。

这些测试只说明代码可执行，不推翻上述机制非识别性。最终裁定保持 **FAIL**。
