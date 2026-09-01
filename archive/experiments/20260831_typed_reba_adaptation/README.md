# Typed REBA adaptation for hateful temporal localization

截至 2026-08-31；状态：novelty review FAIL，停止于正式训练前。

## 跨任务来源与 novelty 边界

REBA（CVPR Findings 2026）在 weakly supervised video anomaly detection 中提出 residual
multi-scale mixture-of-experts（RMoE）与 bidirectional video-text alignment（BiAlign），目标是减少
模型只依赖少数极端 snippets 并改善细粒度边界。官方论文与源码只覆盖 UCF-Crime/XD-Violence，
尚未用于 hateful video detection/localization。本候选不主张 RMoE、BiAlign 或 MIL pooling 本身新。

## 非 trivial task adaptation

hateful video 不是纯视觉异常：攻击行为可能只出现在 speech/transcript、画面文字或视觉 target，且
不同模态异步。MultiHateLoc test error analysis 显示当前 modality ownership 严重错位；UOT test
又表明跨模态共享 normal capacity 在高正例占比视频压平排序。因此 adaptation 固定为：

1. audio/visual/text 各自使用共享结构但不共享参数的 residual multi-scale temporal experts，保留各
   模态不同的时间尺度；
2. 每秒由 learned modality gate 融合，但语义约束使用 class-aware multi-positive bidirectional
   alignment，在 speech/text 与 audio-visual witness 之间对齐同类集合，避免官方 instance-pair
   BiAlign 把同一 class 的其他视频当 false negatives；
3. 使用由训练标签学习的 mean/softmax occupancy mixture 形成 bag probability，替代固定 top-K
   事件比例；同一 frame probability 产生最终 1fps score，不做 ensemble、test routing 或后处理。

core 不使用相邻帧平滑、CRF 或 duration penalty；时间结构只来自待归因的 residual temporal experts。

官方训练代码每个 epoch/若干 batch 直接读取 test GT 并按 test AP 保存 checkpoint，该流程禁止使用。
本实现必须只用 frozen validation manifest 在一次训练内选择 checkpoint，选定后立即生成 test
prediction；test GT 只允许共享 evaluator 与后续 error analysis 读取。

## 首轮 gate

固定 seed 234，独立训练 HateMM 与 HateClipSeg。每个 checkpoint 同时输出 core 与将 residual
experts 置零的 scale-1-only inference control。只有两语料 core 三项 test 指标全部严格超过当前
SOTA，且 core within-video ROC 都严格超过 control，才继续做 no-BiAlign retraining ablation 与
MHC-EN/ZH；任一失败即记录原因并进入下一机制，不按 corpus 路由。

## 独立 novelty 结论

`NOVELTY_REVIEW.md` 与 `PRE_RUN_REVIEW.md` verdict 均为 `FAIL`；未启动正式训练、未生成本候选 test prediction。当前实现
只能作为明确标注的 REBA-inspired multimodal baseline，不能作为 novel adaptation：

- 所谓 class-aware BiAlign 只有二元 video label，把所有 hateful videos 当同类 positives，并无
  hate type 或 modality ownership；无条件 AV-average 与 transcript 对齐再次广播了错误模态标签。
- occupancy `o*mean + (1-o)*softmax_pool` 中 softmax pool 通常不小于 mean，bag BCE 会诱导正
  bag 选择 peak、负 bag 选择 mean，退化为 label-dependent selector，而不是可信事件占比。
- `bag_probability * frame_probability` 对同一视频只是正数缩放，within-video ROC 数学上不变；
  不能把它描述为定位机制。
- 三套 REBA-style encoder、普通 late modality gate 与 contrastive regularizer 只是组件串接，没有
  hateful-specific joint constraint。
- HCS train 的 219 positive/32 negative 在 batch 8 下会频繁产生单类 batch，此时当前 multi-positive
  BiAlign 恒为零；scale-1 同 checkpoint inference control 也无法证明 alignment 或 ownership 归因。

去向：`STOP_BEFORE_FORMAL_RUN`。保留源码与 review 作为反模式记录；下一候选必须让监督约束直接
作用于 time×modality ownership，而不是再从二元 bag label 构造全模态同类对齐。

## 冻结运行入口

独立 pre-run review PASS 后，每个任务与 SSH 会话解耦：

```bash
mkdir -p runs/20260831_typed_reba_adaptation/pilot_seed234/hatemm
setsid /home/jehc223/miniconda3/envs/HateVideo/bin/python \
  experiments/20260831_typed_reba_adaptation/train.py \
  --corpus hatemm \
  --output-dir runs/20260831_typed_reba_adaptation/pilot_seed234/hatemm \
  > runs/20260831_typed_reba_adaptation/pilot_seed234/hatemm/run.log 2>&1 < /dev/null &
echo $! > runs/20260831_typed_reba_adaptation/pilot_seed234/hatemm/run.pid
```

若训练中断，使用完全相同参数并追加 `--resume`。训练输出状态为 `prediction_complete` 后才运行：

```bash
/home/jehc223/miniconda3/envs/HateVideo/bin/python \
  experiments/20260831_typed_reba_adaptation/evaluate.py \
  --corpus hatemm \
  --run-dir runs/20260831_typed_reba_adaptation/pilot_seed234/hatemm
```

HateClipSeg 只替换 corpus 与 run 目录。权威结果只认对应 run 的 `metrics.json`。
