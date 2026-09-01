# OmniVTG grounder diagnostic

截至 2026-08-31。目标是资格测试一个专门做 open-world video temporal grounding 的外部
checkpoint，判断它是否比此前逐窗 Qwen teacher 提供更可靠的 hateful-event ordering。
OmniVTG 是外部 baseline/teacher，不是本项目 novel 方法。

固定 query：`content that attacks, dehumanizes, threatens, or expresses hatred toward a person
or group based on identity`。输入、2 fps、timestamp interleaving、coarse-to-fine prompt 与生成
格式遵循 OmniVTG 官方 demo；checkpoint 已存在于本机模型 cache，不需要下载。Smoke 使用
官方 vLLM multimodal path，而不是把官方的一帧一 placeholder prompt 直接送入 Hugging Face
forward；后者的 token/visual-embedding 契约不同。

先运行不读取 GT 的 technical smoke：

```bash
/home/jehc223/miniconda3/envs/OmniVTG/bin/python \
  experiments/20260831_omnivtg_grounder_diagnostic/smoke.py \
  --video /absolute/path/to/one_video.mp4
```

Smoke 只打印 raw completion 和最后一个 `<answer>` 中的合法 interval，不载入任何 labels、
span GT 或共享 evaluator，也不产生 performance number。通过并经独立 review 后，另行实现的
正式 diagnosis 必须覆盖
HateMM/HateClipSeg 完整 test positive cohort，解析失败也计入，interval 转 1 fps binary score
并只用共享 evaluator 报 within ROC。由于原 grounder 没有 null/negative-query 训练，negative
video 的 pooled AP/ROC 没有可解释性；本轮只判断它能否作为 train-only ordering prior，不能
作为本项目方法、候选 branch 或 SOTA prediction。正式 diagnosis 实现仍需单独 review；本
technical smoke 的 PASS 不授权 teacher、student 或 test evaluation。

实际环境兼容说明：官方仓库同时固定 vLLM 0.11.0 与 PyTorch 2.7.1，但发布包的依赖
会替换为 PyTorch 2.8。为不改变项目 `HateVideo` 环境，本轮建立隔离 `OmniVTG`
环境，使用 vLLM 0.9.2 + PyTorch 2.7.0/CUDA 12.8 + Transformers 4.52.4；该组合
保持官方 `LLM.generate` multimodal 输入 API，并已在 RTX 5090 上通过真实视频 smoke。
4.55 以上 Transformers 会与该 vLLM 的 `aimv2` 注册冲突，因此没有用于正式运行。
引擎固定 `enforce_eager=True`，避免生成或记录编译缓存标识；这只关闭运行时编译优化，
不改变 prompt、视觉输入、sampling 或 interval parser。当前 eager bytes 已重新完成一次真实
单视频 technical smoke，合法 interval 成功解析；该 smoke 不读取 GT。

正式 producer `run_teacher.py` 只读取 test split membership 与 video-level label，覆盖完整
positive evaluator test cohort，模型常驻并支持从 JSONL 断点续跑；不导入 frame GT 或
evaluator。HateMM 的固定 test split 中 `hate_video_427` 是 video-level positive，但没有可用
localization gold；其 membership exclusion 固定在纯 protocol 配置中，producer 不打开 GT
arrays。最终 cohort 为 HateMM 85、HateClipSeg 69。

正式输出路径不可由命令行替换：
`runs/20260831_omnivtg_grounder_diagnostic/formal/<corpus>/predictions.jsonl`。Producer 启动
模型前先解析两语料各自完整 cohort 的 media 路径；resume 时逐行核 corpus、split、固定 model、
query、source video、schema 与 completion/parse/error 一致性，duplicate 或 partial row 直接
失败。解析失败和 inference failure 都写成一行并在 evaluation 中转为全零 score。

两个 corpus 应分别在断开 SSH 后仍存活的后台进程中顺序运行（不要同时占同一 GPU）：

```bash
mkdir -p runs/20260831_omnivtg_grounder_diagnostic/formal/hatemm
nohup /home/jehc223/miniconda3/envs/OmniVTG/bin/python \
  experiments/20260831_omnivtg_grounder_diagnostic/run_teacher.py \
  --corpus hatemm \
  > runs/20260831_omnivtg_grounder_diagnostic/formal/hatemm/run.log 2>&1 &
echo $! > runs/20260831_omnivtg_grounder_diagnostic/formal/hatemm/run.pid
```

HateMM 完成后以同一命令把 corpus 和目录改为 `hateclipseg`。两份 JSONL exact coverage 后，
只运行一次 combined evaluation：

```bash
/home/jehc223/miniconda3/envs/HateVideo/bin/python \
  experiments/20260831_omnivtg_grounder_diagnostic/evaluate.py
```

`evaluate.py` 随后读取固定 test GT，把合法 interval 按共享 evaluator 的 1 fps、半开区间
规则转为 binary score；解析失败保留为全零分数，不从分母删除。它同时报告三个固定指标，
但 pooled AP/ROC 是 positive-cohort diagnosis，不与 full-test SOTA 比较；本 premise 的预注册
gate 只比较同一批正例视频定义不变的 within-video ROC。Evaluation 不接受单 corpus mode 或
可替换 prediction path；它必须同时验证两份 canonical artifacts，并只写一个 combined
`formal/metrics.json`。

如果两个语料任一 within ROC 不超过各自现有 structured control（HMM `.63377`、HCS
`.53652`），则 `STOP_BEFORE_STUDENT`。即使通过，也必须另做 novelty review，禁止直接把
grounder distillation 当贡献。不得选择单独通过的 corpus、切换 query 或按 corpus 路由；后续
student test 也必须标为受本轮 test-informed diagnosis 影响的 developmental evidence。

来源：OmniVTG CVPR 2026 官方实现与论文。

## 最终结果与去向

正式 test diagnosis 覆盖 HateMM 85/85 与 HateClipSeg 69/69 个 positive evaluator-test
视频；推理失败分别为 4、3 个，均按预注册规则保留为全零分数。权威输出为
`runs/20260831_omnivtg_grounder_diagnostic/formal/metrics.json`，独立结果审计见
`POST_RUN_REVIEW.md`，结论为 PASS。

- HateMM：pooled AP `.62571`、pooled ROC `.57810`、within-video ROC `.62638`；固定
  structured control within-video ROC 为 `.63377`。
- HateClipSeg：pooled AP `.62647`、pooled ROC `.54904`、within-video ROC `.53901`；固定
  structured control within-video ROC 为 `.53652`。

HateMM 未超过 control，因此双语料 gate 为 `teacher_premise_pass_both=false`，正式结论为
`STOP_BEFORE_STUDENT`。HateClipSeg 的单语料微小增益不足以授权语料路由、query 调参或
student 训练。OmniVTG 只作为失败的外部 teacher diagnostic 归档，不是本项目方法，也不进入
SOTA 比较。本轮 test predictions/GT 已用于 premise evaluation，因此受其影响的后续方法结果
均属于 iterative/developmental test evidence。
