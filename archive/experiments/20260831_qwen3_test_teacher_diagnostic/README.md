# Qwen3 dense pointwise teacher — test premise diagnostic

截至 2026-08-31。旧 Qwen3 qualification 只在 validation 上比较 performance，并据此停止
HateClipSeg 与 student；按现行 Rule 11，该结果不能 inform performance-oriented method
development。本轮补做固定 test premise evaluation。Qwen3 是外部 teacher diagnostic，不是
novel 方法，也不进入 SOTA 表。

## 固定协议

- corpus：HateMM 与 HateClipSeg，各自完整 positive evaluator-test cohort；HateMM 中已知没有
  localization gold 的 `hate_video_427` 按固定 protocol 排除。
- model：`Qwen/Qwen3-VL-8B-Instruct`，deterministic decoding。
- 每个 16 秒窗口、stride 8 秒，读取最多 4 张既有 1 fps frame 与对齐 ASR；固定 prompt 只返回
  0–10 整数。
- producer 只读 test membership、预先冻结的 scoped video-label-only JSON、feature length、frames
  与 ASR，不调用 `hdata.load_labels("hateclipseg")`，不打开 segment/span GT 或 evaluator。
  parse/inference failure 保留为零分，不从 cohort 删除。
- 两语料完成后，`evaluate.py` 才读取 test GT，按 overlap-window mean 生成 1 fps score，并只
  调用全仓库共享 evaluator。三个固定指标全部输出；positive-only pooled AP/ROC 只作诊断，
  不与 full-test SOTA 比较。
- Producer 与 evaluator 共用纯 `protocol.py`：固定 cohort 数 HMM 85/HCS 69、HMM no-GT
  exclusion、model、prompt template、16/8 窗口、每窗 4 帧、score 范围和 gate 都只有一个定义。
  每次启动都会用可读 `config.json` 与 `code_version.txt` 绑定这些设置。
- Resume 必须是当前固定 cohort 的精确前缀；每行还必须匹配当前 1 fps feature length 和完整
  `temporal_windows` 序列。成功窗口必须能从 generation 重解析出同一分数；parse/inference
  failure 必须为零分，行级状态必须与窗口状态一致。Evaluator 会重新执行同一契约，而不是只验
  行数或 ID。

正式输出固定为 `runs/20260831_qwen3_test_teacher_diagnostic/formal/`。两语料任一 teacher
within-video ROC 不严格超过当前 test SOTA（HateMM `.631531717970362`、HateClipSeg
`.5619078936355938`），结论即
`STOP_BEFORE_STUDENT`。不得修改 query、按 corpus 路由或只保留通过的一边。受本轮 test
predictions/GT 影响的后续结果属于 iterative/developmental evidence。

正式运行前必须完成独立代码与 evaluation review。

## 正式运行

独立 review PASS 后，两语料按顺序运行；不要同时占用 GPU。Producer 在加载模型前先核完整
cohort、feature lengths、已有 metadata 和 resume rows。每次长任务与 SSH 会话解耦，并把日志与
PID 放在对应正式目录：

```bash
mkdir -p runs/20260831_qwen3_test_teacher_diagnostic/formal/hatemm
nohup /home/jehc223/miniconda3/envs/HateVideo/bin/python \
  experiments/20260831_qwen3_test_teacher_diagnostic/run_teacher.py \
  --corpus hatemm \
  > runs/20260831_qwen3_test_teacher_diagnostic/formal/hatemm/run.log 2>&1 &
echo $! > runs/20260831_qwen3_test_teacher_diagnostic/formal/hatemm/run.pid
```

HateMM 明确 `cohort_complete=85` 后，再以同一命令把 corpus 与输出目录换为
`hateclipseg`；HCS 必须明确 `cohort_complete=69`。两份 canonical predictions 都完整后，只运行
一次 combined evaluation：

```bash
mkdir -p runs/20260831_qwen3_test_teacher_diagnostic/formal
nohup /home/jehc223/miniconda3/envs/HateVideo/bin/python \
  experiments/20260831_qwen3_test_teacher_diagnostic/evaluate.py \
  > runs/20260831_qwen3_test_teacher_diagnostic/formal/evaluate.log 2>&1 &
echo $! > runs/20260831_qwen3_test_teacher_diagnostic/formal/evaluate.pid
```

权威结果只认 combined `formal/metrics.json`。不得在任一 producer 未完整时解释 partial rows，
也不得单独评测或选择某一语料。

## 正式 test 结果与去向

权威输出：`runs/20260831_qwen3_test_teacher_diagnostic/formal/metrics.json`；独立结果审计：
`POST_RUN_REVIEW.md`，完整性 `PASS`。

- HateMM：85 个视频、1480 窗、0 failure；within-video ROC `.5617599321`，低于固定门
  `.6315317180`。
- HateClipSeg：69 个视频、2025 窗、0 failure；within-video ROC `.5396281639`，低于固定门
  `.5619078936`。

双语料 teacher premise 失败，最终 verdict 为 `STOP_BEFORE_STUDENT`。不恢复 Qwen3 student、
不调整 prompt/window、不按 corpus 路由。旧 validation qualification 不再承担任何性能决策；本次
test diagnostic 是该 teacher 路线的最终依据。
