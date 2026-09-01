# Single-Qwen temporal mark–erase observation

截至日期：2026-09-02。该运行是 RESET7 Rule 22 对新单模型 observation 的一次性完整 test 检查，不是方法、训练、validation trial或正式 performance iteration。

## 目的与边界

CVTP（Zhao et al., EMNLP 2024）在 weakly supervised spatio-temporal grounding 中比较同一模型对“标记候选”与“擦除候选”的匹配分数。项目已经证明普通 learned-feature replacement effect（V26）与真实 hate 时间不对齐，因此不能仅靠文献故事重开 counterfactual family。本 observation 只回答：同一个 pretrained VLM 在保留前后语境时，其 `marked_score - erased_score` 是否比 marked-only judgment 提供 HMM/HCS 共同、时间对齐且 load-bearing 的局部 necessity ordering。

全程只加载一个 `Qwen/Qwen2.5-VL-7B-Instruct` checkpoint。没有 teacher/student、第二个模型、score blend、训练、超参数搜索或 test GT 参与 producer。Producer覆盖 HMM/HCS 完整 video-positive evaluator-test cohort；完成两个语料后 evaluator 才读取 test GT。

每个16秒 candidate以8秒stride移动。输入上下文为 candidate前16秒、candidate本身、后16秒；每段最多2张既有1fps帧。`marked` arm在candidate帧画固定红框，并把对应ASR放进固定 `[MARKED CANDIDATE]` 区段；`erased` arm只把candidate帧替换为固定灰图、candidate ASR替换为`[ERASED]`，前后上下文完全相同。两arm使用同一模型、prompt、deterministic decoding与0–10整数输出。

## 冻结 gate

按overlap mean还原1fps。对每个同时含两类秒的正例test视频计算within ROC，再macro平均：

1. contrast=`marked-erased` 在 HMM/HCS 均 `>=.52`；
2. contrast相对八个固定relative circular shifts的mean within在两语料均 `>=+.02`；
3. contrast-minus-marked-only within在两语料均非负，且至少一个语料 `>=+.01`。

三项同时成立才允许把CVTP temporal adaptation写成最后一个novelty brief。任一失败即关闭mark/erase、marker格式、context长度、prompt与VLM变体，不扫参数、不换模型续命。输出只作 iterative/developmental test evidence。

## 运行

GPU空闲后顺序运行HMM、HCS producer，再统一evaluation。`launch_formal.sh`要求显存占用低于2GB且GPU利用率低于10%连续三次后才启动，避免争抢其他用户任务。长任务写入`runs/20260902_temporal_mark_erase_observation/formal/`并与SSH解耦：

```bash
mkdir -p runs/20260902_temporal_mark_erase_observation/formal
setsid bash experiments/20260902_temporal_mark_erase_observation/launch_formal.sh \
  > runs/20260902_temporal_mark_erase_observation/formal/orchestrator.log 2>&1 < /dev/null &
echo $! > runs/20260902_temporal_mark_erase_observation/formal/orchestrator.pid
```

launcher只执行完整HMM、完整HCS与统一evaluation。无smoke。
