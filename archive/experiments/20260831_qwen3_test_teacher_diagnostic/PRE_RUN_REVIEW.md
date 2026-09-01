# Qwen3 dense pointwise teacher test diagnostic：独立 pre-run review

日期：2026-08-31  
范围：`README.md`、`protocol.py`、`run_teacher.py`、`evaluate.py`，以及共享
`eval_baseline_scores.evaluate_scores` / `frame_eval_common` 指标链。  
最终结论：**PASS。** 授权按 README 顺序运行 HateMM 与 HateClipSeg 两个 canonical producer，
两者完整后运行一次 combined evaluation。本授权只覆盖 test teacher premise；不授权 student、
query 调整、单语料选择或把 teacher 当作项目方法。

## Test-first 与数据边界

- 本轮直接评测固定 test premise，不设置 validation gate。README 明确旧 validation performance
  不能指导当前性能决策；本轮 test 结果及其影响的后续结果均标为 iterative/developmental。
- Producer 只读取固定 test membership、scoped video-label-only JSON、1 fps feature length、cached
  frames 与 timestamped ASR。尤其 HateClipSeg producer 不再调用会打开 segment annotation 的
  `hdata.load_labels`；它只使用 scoped JSON 的 corpus、split 和 video labels，不 import evaluator，
  不调用或打开 temporal GT。
- Evaluator 是唯一读取 `gt_arrays(..., "test")` 的环节。没有训练、梯度、checkpoint 或 selection；
  test labels 只用于允许的 teacher premise evaluation。
- Producer 与 evaluator 共用纯 `protocol.py`。固定 cohort 是 HateMM 85、HateClipSeg 69；HMM
  `hate_video_427` 必须存在于 test split、必须是 video positive，并因无 localization gold 被固定
  排除。Split duplicate、缺 label、exclusion 漂移或 cohort 数变化全部 fail closed。

## Window、failure 与 resume contract

- 1 fps feature row 同时定义 score 和 GT 网格长度。完整数据预检确认 HMM 85 与 HCS 69 个视频的
  feature length 逐项等于 test GT length；HMM 1480、HCS 2025 个固定 16 秒/stride 8 秒窗口均
  完整覆盖网格，所有窗口都有 cached frame，所有 cohort ID 都有 ASR entry。
- 每行必须严格匹配当前 model、corpus、test-positive split、cohort 顺序、feature length 和完整
  `temporal_windows` 序列。Resume 只接受 exact cohort prefix；blank、extra、duplicate、reorder、
  wrong length、漏窗或改窗均拒绝。
- 成功窗口的 generation 必须由同一 parser 重得所存整数；parse failure 必须保留不可解析
  generation 且分数为 0；inference failure 必须为 0。行级 inference status 必须与窗口状态一致。
  Evaluator 会重新执行相同 row contract，不能靠绕过 producer 提交非零 failure 或不同 window
  recipe。
- Producer 在模型加载前验证全部 feature lengths、可读 config/code-version metadata 和已有
  resume rows。正式路径固定；README 的后台命令把 run log 与 PID 写入对应 run 目录。

## Evaluation 与 gate

- `densify` 只对冻结窗口做 overlap mean；exact spans 和完整覆盖先由共享 contract 验证，随后再
  检查 score finite/range、score/GT shape 与 shared evaluator missing/extra。
- 三项指标只调用一次仓库唯一 `eval_baseline_scores.evaluate_scores`，没有实验内复制指标实现。
  输出字段为 positive-cohort pooled AP、positive-cohort pooled ROC-AUC、within-video ROC-AUC，
  并硬核 within cohort 数 HMM 85/HCS 67。
- Pooled 两项明确命名为 `positive_cohort_diagnostic`，README 明确禁止把它们当 full-test SOTA。
  Within-video 的正例定义与 full test 相同，因此只用它比较固定 test SOTA：HMM
  `0.631531717970362`、HCS `0.5619078936355938`。
- `main` 不接受 corpus 或 prediction path 参数，固定同时读取两个 canonical artifacts；仅当两边
  都严格超过 gate 才允许 `continue_to_student_design=true`。任一失败即
  `STOP_BEFORE_STUDENT`，不能挑单 corpus 或 corpus routing。
- 溯源只使用可读路径、完整固定配置、模型名称、日期和代码版本说明；没有额外内容标识或相应
  前置门。

## Review 中发现并修复

1. 初版 evaluator 从不存在的 `shared["pooled"]` 取 AP/ROC，正式 evaluation 会直接失败；现已
   改为共享返回的顶层 `pr_auc` / `roc_auc`，并以 synthetic shared-evaluator call 验证字段。
2. HCS gate 初版只有四舍五入值 `.56191`；现统一为权威精确值
   `0.5619078936355938`。
3. 初版 resume/evaluator 只核大致 schema 与 ID，未绑定 exact window recipe，failure status 也可
   携带非零分；现由共享纯 protocol 完整阻断，并加入可读 config/code-version binding。
4. 初版 HCS producer 通过共享 `load_labels` 间接打开 segment annotation；现改为固定 scoped
   video-label-only source，producer 不再触碰 temporal GT。
5. README 补齐顺序运行、后台日志/PID、完整 cohort 后唯一 combined evaluation 的正式命令。

## 实际验证

- 三个 Python 文件编译通过，`run_teacher.py --help` 在不加载模型、不写正式输出的情况下通过；
  shell diff check 通过。
- 真实只读 membership/length/frame/ASR 预检得到 HMM 85 与 HCS 69 exact cohorts、零长度错位、
  零 uncovered window、零 empty-frame window、零缺失 ASR ID。
- Synthetic contract tests 覆盖合法 success、parse failure、inference failure，以及 nonzero failure、
  漏窗、换序、generation/score 不一致、wrong length 五类拒绝项；metadata exact-match 与 strict
  prefix resume 测试通过。
- Static check 确认 producer 不调用 temporal GT/evaluator，evaluation 只有一个共享 evaluator
  import 与一个调用点。正式 run 目录在 review 结束时仍不存在；未加载模型、未生成 prediction、
  未读取任何正式 performance。

**最终 verdict：PASS。** 可启动两个正式 producer；必须等两边分别输出
`cohort_complete=85/69` 后才能运行 combined evaluator并解释结果。
