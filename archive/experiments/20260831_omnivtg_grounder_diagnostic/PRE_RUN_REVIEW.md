# OmniVTG grounder formal teacher diagnosis：独立 pre-run review

日期：2026-08-31  
范围：`protocol.py`、`smoke.py`、`run_teacher.py`、`evaluate.py`、`README.md`  
最终结论：**PASS，授权启动 HateMM 与 HateClipSeg 完整 positive evaluator-test cohort 的
formal teacher premise。** 此授权不等于 student 或方法授权。

## Producer 数据边界与 cohort

- `run_teacher.py` 只 import `hate_common.data` 的 test split membership 与 video-level labels；
  不 import、读取或接收 frame/span GT，也不 import evaluator。
- HateMM 原始 test split 有 86 个 video-level positives，但 `hate_video_427` 没有可用 localization
  gold。初版 producer 会生成 86 条而 evaluator 在该 ID 失败。修订版把这个无时序内容的固定
  membership exclusion 放入纯 `protocol.py`，producer 不打开 GT arrays。正式 evaluator
  positive cohorts 固定为 HateMM 85、HateClipSeg 69；全部 media 在加载模型前必须唯一解析。
- Producer 路径固定为
  `runs/20260831_omnivtg_grounder_diagnostic/formal/<corpus>/predictions.jsonl`，不能通过 CLI
  换目录重跑。固定 checkpoint、query、split、runtime mode 与环境版本写入可读 config；没有
  可替换 model 参数。
- 每行统一记录 contract、corpus、split、model、query、source video、completion、interval 与
  error。Resume 会完整复核 schema/provenance、重新 parse completion、拒绝 duplicate、partial、
  wrong-model/query/source 及成功/失败字段矛盾；结束后重新打开 JSONL 要求 exact cohort。
- 无法解析 completion 写 `ParseFailure`；decode/model exception 也写一行。两类失败都留在固定
  denominator，evaluation 分别计数并赋全零逐帧 score，不静默删除。

## 官方 OmniVTG contract 与兼容环境

- Timestamp interleaving 与官方 demo 相同：2 fps sampled indices，temporal merge 2，奇数末帧
  补齐，每个 temporal row 前写 timestamp，再放独立 vision-start、一个 video placeholder、
  vision-end。Timestamp 数与原 `video_grid_thw` temporal rows 不一致即失败。
- Processor 的 video message、pixel budget、`process_vision_info` metadata/kwargs、空 text、
  `do_resize=False` 与官方一致。Grid 按 temporal dimension repeat 后每行 `t=1`，
  `second_per_grid_ts` 为逐行 0；pixel tensor 不另做 crop、resize 或 reorder。
- Producer 与 smoke 的 LLM kwargs、multimodal prompt dict 和 SamplingParams 静态逐项一致；除
  `enforce_eager=True` 外与官方 demo 保持相同。Eager 只改变 vLLM 执行后端，不改变模型权重、
  prompt、视觉输入、sampling 或 parser。
- 隔离环境为 vLLM 0.9.2、PyTorch 2.7.0/CUDA 12.8、Transformers 4.52.4、qwen-vl-utils
  0.0.14，而官方声明更新的 vLLM/Transformers 组合。该 deviation 已通过当前 eager bytes 的
  真实单视频 smoke：模型完成生成，最后一个 `<answer>` 解析为合法 interval，日志确认 eager
  mode；该 smoke 不读 GT。Processor 对已由 qwen-vl-utils 采样的视频 kwargs 给出兼容 warning，
  但 sampled metadata、timestamp count 和 video grid 硬检查全部通过，不构成错位证据。

## Interval 转换与唯一 evaluation

- `evaluate.py` 只接受两个 canonical prediction files，并在同一次运行中评 HMM/HCS；没有
  single-corpus mode，也不能替换任一 prediction path。
- 合法 interval 通过唯一共享 `frame_eval_common.spans_to_frame_scores` 转到 1 fps，明确使用
  `[start,end)`；超出 grid 自动裁剪，退化/完全越界 interval 保留为全零并计数。合成测试确认
  `[1,3)` 只标记整数秒 1、2。
- 三个项目指标只调用唯一共享 `eval_baseline_scores.evaluate_scores`。Exact row coverage、
  score/GT shape、finite、shared evaluator missing/extra，以及 within cohort count HMM 85/HCS 67
  全部是 fail-closed 门。
- Evaluation 固定读取 test GT。test labels/GT 不参与 producer、梯度、训练或 checkpoint
  selection，只用于 Rule-10 允许的 developmental teacher error analysis。
- Pooled AP/ROC 明确只是 positive-cohort diagnosis，不与 full-test SOTA 比较。唯一晋级门是
  within-video ROC 严格超过同 cohort 的固定 structured controls：HMM `.6337661352`、HCS
  `.5365185533`。

## 双语料 gate、泄漏与方法边界

- Combined output 只有 HMM 与 HCS 都严格过门时才写
  `teacher_premise_pass_both=true` 与 `continue_to_student_design=true`；任一失败统一
  `STOP_BEFORE_STUDENT`。不能挑单 corpus、改 query、换 checkpoint 或做 corpus routing。
- Producer 使用 test video-level labels 只为构造 oracle-positive diagnostic cohort。这意味着它
  不是 deployable full-test method，也不能报告为本项目 prediction、candidate 或 SOTA。
- 本轮 test-informed diagnosis 可以按当前 Rule-10 指导后续开发，但后续 student test 必须标为
  developmental；仍禁止 test GT 进入梯度或 checkpoint selection。
- 即使双语料 teacher gate 通过，也只允许进入 student design；student 实现、novelty、训练和
  test 链必须另行独立 review，不能把外部 grounder 或直接 distillation 宣称为贡献。

## 本次修复与验证

修复了四个 formal blocker：HateMM 86/85 cohort mismatch；resume 只验 ID；单 corpus evaluator
可独立给 PASS；任意 output/model 路径可绕开固定 protocol。另将 method-side rasterization 从
语义上更含混的 GT helper 改为共享 `spans_to_frame_scores`。

完成的非模型检查：

- 两环境 `py_compile` 与 diff check 通过；formal output 目录尚不存在。
- Membership-only cohort test 得到 HMM 85、HCS 69，不读取 frame GT。
- Synthetic resume tests 确认合法 row 可恢复，wrong query、duplicate 与 partial row 全部拒绝。
- Synthetic half-open、parse failure、inference failure、outside-grid tests 全部通过，失败均保留
  为定长全零 score。
- Static check 确认 smoke/producer runtime contract 一致，evaluation 只能 combined 运行。
- 当前 eager technical smoke 已实际通过；本 review 未运行 formal model，也未读取任何 formal
  performance 结果。

**最终 verdict：PASS。** 可以按 README 先后启动两个 canonical producer；两份 exact-complete
JSONL 之后只运行一次 combined evaluator。任何单语料结果都不授权 student。
