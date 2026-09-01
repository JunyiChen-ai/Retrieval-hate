# POWA anchored consensus-order distillation pilot

**截至 2026-08-31；状态：TEST GATE FAIL，机制淘汰，未训练。淘汰原因：fixed batch-2 K16
audio+VERA teacher 在 HMM/HCS test 的 pooled AP、pooled ROC、within-video ROC 全部未过 SOTA。**

## 最终 test gate 结果

权威输出：
`runs/20260831_powa_consensus_distillation_pilot/test_teacher_diagnostic/analysis.json`。
这是 Rule-10 iterative/developmental test evidence；test 标签未参与梯度或 checkpoint selection。

| corpus / branch | pooled AP | pooled ROC | within ROC |
|---|---:|---:|---:|
| HMM POWA anchor | .560580 | .809378 | .583421 |
| HMM VERA K16 | .560769 | .810042 | .578200 |
| HMM audio+VERA K16 | .562066 | .807532 | .579847 |
| HMM SOTA gate | .593832 | .816184 | .631532 |
| HCS POWA anchor | .594426 | .551684 | .515965 |
| HCS VERA K16 | .596363 | .555612 | .541567 |
| HCS audio+VERA K16 | .585747 | .549568 | .516027 |
| HCS SOTA gate | .619371 | .605023 | .561908 |

audio-only 也在两语料均降低 within。固定 core 在两语料全部三项门失败，且 audio 会抹掉 HCS
的部分 VERA 增益。因此不生成 HMM train cache、不训练 residual student、不调 teacher 权重，
也不按 corpus 选择不同 teacher。结论是 fixed sparse multi-teacher ordering 不足以提供跨语料、
可蒸馏的 SOTA 定位信号。

## 研究问题

POWA 已经给出较强的 pooled frame score mass，但 HMM/HCS validation 的
within-video ROC 只有 `.57193/.52707`。现有 validation upper bound 表明，使用同一
固定规则把 POWA 的 score multiset 按 audio+VERA 的视频内平均百分位重排，可达到
`.62076/.56402`，同时保住 pooled AP/ROC。直接重排是 ensemble/calibration，不能成为
方法；本 pilot 只检验一个问题：能否把该局部顺序蒸馏进一个测试时只依赖 POWA 输入的
单学生。

## 唯一核心变化

冻结 corpus-specific POWA，在其逐秒 evidence representation 上训练一个 zero-init、
zero-mean temporal residual head。训练 teacher 由同语料 train split 构造：

1. audio linear probe 用 video-level label 做 5-fold out-of-fold 预测；
2. VERA 用同一固定 InternVL2-8B checkpoint、同一五问 prompt 和统一
   `K=16` 时间索引生成 label-free train score；
3. 在完整 student 时间网格计算 audio percentile；把 VERA K16 raw score 线性插值到该
   网格后再计算 percentile；取二者平均形成 pairwise ordinal target；
4. 学生只学习这些 pairwise 顺序，同时用 anchor-preservation 与原 video-level MIL
   loss约束 pooled score。训练和 residual 推理统一使用 POWA 的 fixed 200-bin context grid；
   测试时另跑一次 native dense-grid POWA 得到 anchor，把 residual 线性 lift 回 native grid 并
   对每个 crop 重新减均值后，直接输出 `sigmoid(logit(dense_POWA)+residual)`。因此 student
   是同一个 frozen POWA 的两次 forward，不加载 audio probe、VERA 或 teacher score，不做
   rank transport、ensemble 或 post-hoc calibration。

该蒸馏机制本身不作 novelty claim；若成功，论文方法的 novelty claim 仍限定在已经独立
评审通过的 POWA PEF/AWB/policy compiler。teacher 只作为训练辅助并完整披露。

## 监督与数据边界

- HMM、HCS 分别训练，绝不混合 corpus train set。
- 只使用 train video label；不使用 train span GT。
- validation GT 只用于一次训练内部的 checkpoint selection，不用于方法开发或性能结论。
- 每个训练臂选定 checkpoint 后立即在 test 上评测全部固定指标；test 结果用于方法开发，属于
  Rule 10 下的 iterative/developmental evidence，但 test 标签不参与梯度或 checkpoint selection。
- VERA 固定为 batch 2。HCS 使用已有完整 238/238 K16 cache，13 个无可解码 visual stream
  视频按既有 label-free media audit 排除；HMM 用相同 batch 语义补齐 744/744 train K16
  cache；HMM/HCS test teacher diagnostic 也各自实际生成同 recipe 的 K16 cache，不使用旧
  dense prediction 抽点代替。
- 五问 hate-domain prompt 是代码内唯一固定模板；运行时不读取 validation prompt-selection
  文件。当前候选是否继续只依据 Rule 10 下记录的 test teacher diagnostic。
- 时间支持固定为 `ceil(min(frozen_1fps_length, media_duration))`。只检查 JSON 可解析、完整
  coverage、时间索引和 score 范围；按项目规则不计算或绑定任何哈希。

## 归因与反模式控制

- `core`：真实 audio+VERA ordinal target。
- `residual_no_teacher`：相同 residual、optimizer、epoch、MIL/anchor loss，但关闭 pair loss，
  排除“多训练一个 head”本身解释增益。
- `shuffled_teacher`：先冻结与 core 完全相同的 pair endpoints、coverage 和 pair weights，再仅
  置换 direction，禁止 shuffle 后重新 threshold。
- `audio_only` / `vera_only`：使用与 core 相同的 pair endpoints，各自只在该来源有严格偏好时
  决定 direction；tie pair 不施加 loss，并报告 active coverage。这是 conditional-direction
  attribution，不冒充独立 single-teacher endpoint 方法。
- `anchor_only`：零 residual 的原 POWA，确认 evaluator/input 对齐。
- `direct_additive` 是唯一候选输出；transport 只作为 Rule-10 test teacher diagnostic 的分析
  上限，不进入 checkpoint、student inference 或候选 test 输出。
- Rule-10 test teacher diagnostic 的 transport 只作上限分析，并采用 tie-neutral assignment：
  teacher 打平时保留 POWA anchor 在该 tie group 内的原排序；全 tie 必须严格回到 anchor，禁止
  用时间索引隐式破 tie 产生单调时序信号。
- 禁止根据 corpus 选择 audio-only、VERA-only、不同 prompt、不同 K、不同 loss 权重。
- 如果 core 的增益不超过 shuffled control，机制归因失败，即使某个指标偶然提高也淘汰。
- `audio+VERA` 是看过多个 validation upper-bound arm 后形成的 validation-adaptive 选择，不得
  写成预注册或无选择偏差的 recipe。未来 test 全部按 Rule 10 标为 developmental evidence。

若最小 pilot 的 test 结果支持继续，随后完成 MACIL/base × ordinal KD 的 2×2 matched
control，以及 full POWA / same-time binder / flat-anonymous head / policy-teacher permutation
的当前 within-video ROC 消融。只有加入同一 auxiliary 后 POWA 仍有剩余贡献，才能把最终
localization performance 与 novel core 一起晋级。

具体冻结 gate、超参数与运行顺序见 `PILOT_PLAN.md`。独立 literature/anti-pattern review 为
`NOVELTY_ANTIPATTERN_REVIEW.md`；其中 `GO_AFTER_FIX` 的必改项已纳入本文与 frozen plan。
正式运行前仍需独立 code + evaluation review。
