# 规则 6 code review（第 4 版）— 20260925_query_paradigm 第 15 节

**判定：PASS（无 must-fix）。** 第 4 版的两项改动——词级转录答案（`answer_source "words"`）与树节点先验势（`node_prior true`）——都实际进入训练 loss、validation 选 checkpoint、test 后验与 EIG；默认配置（`k30` + `node_prior false`）与第 3 版代码数值逐位一致；没有发现 train/val/test 泄漏、特征/时间/标签/split 错位、超参或 checkpoint 加载链断裂；test 数字仍走 `hc.run_evaluator`。下表列出的全部是 note，不阻断搜索。

- 审稿人：独立 agent（Fable 5.1），2026-09-27，一次性。
- 范围：`RESEARCH_ITERATION_RULES.md` 第 6 条。只查会改变实验观察或结论的 bug；不审风格、重构、健壮性、理论。未改动任何代码；未运行训练或 GPU 任务（本机 GPU 被他人占用），只做了 CPU 数值核对与本地数据文件核对。
- 审阅对象：`git diff 61c2c83 HEAD -- experiments/20260925_query_paradigm/ scripts/asr_words.py`（21 个文件），以及 HEAD 版 `train.py`、`ctree.py`、`model.py`、`policy.py`、`cpolicy.py`、`data.py`、`qtree.py`、`search.py`、`build_tree_manifest.py`、`extract_tree_answers.py`、`summarize_r4.py`、`concern_diagnostics.py`、`pilot_words.py`、`launch/run_search.sh`、`launch/run_ablations_r4.sh`、`launch/run_diag_r4.sh`、`launch/lab_*.sh`。README 第 14.4–14.5、15 节与 `REVIEW_RULE4_R4.md`。

## 1. 做了什么检查

### 1.1 代码阅读
按第 2 节六个问题逐条追踪：答案读取点、phi 的进入点（loss / 选 checkpoint / test 后验 / EIG / record_voi）、Forest 节点编号与五 crop 平均的对齐、checkpoint 保存与加载、默认路径的行为差异、消融臂的超参合成。

### 1.2 CPU 数值核对（`~/miniconda3/envs/HateVideo/bin/python`，`CUDA_VISIBLE_DEVICES=""`）
| 核对 | 结果 |
|---|---|
| 穷举核对 `ctree.up` / `ctree.marginals` 带 phi：T = 1..7，每个 T 三组随机 s、g、phi、随机稀疏 A3、随机闭合零膨胀链；对比枚举 y ∈ {0,1}^T（any(y) = 1）的 log P(o \| G=1)、P(G=1 \| o)、E[z_n]、E[y_t] | 最大绝对误差 2.7e-15 |
| 默认路径与第 3 版一致：`git show 61c2c83:ctree.py` 与 HEAD `ctree.py`，同一批随机输入（4 个视频，T = 5/9/14/3），`up`、`marginals`、`log_evidence`，零膨胀与耦合链各一次 | 误差 0.0（逐位一致） |
| `PriorNet.node_logits` 对每个节点 = 用视频头 `att` 的 logit 在节点区间内做 softmax 池化再过 `node`；叶为 0；根的池化向量过 `vid` 等于 forward 的 g | 内部节点最大误差 9.5e-7（float32），叶 0，根与 g 误差 0.0 |
| `policy.video_prior_nodes` 的 `Forest([T]*5, T)` → `view(5, -1)`：每行等于单树 `Forest([T], T)` 的 phi，N = 2T − 1 | 一致 |
| state_dict 含 `node.weight`、`node.bias`；第 4 版 ckpt 装进 `node_prior false` 模型、第 3 版 ckpt 装进 `node_prior true` 模型 | 两个方向都 `RuntimeError`（响亮失败，不会静默错载） |
| 构造 `node` 层是否消耗全局 RNG | 是（见 note N1） |

### 1.3 本地数据核对（只读）
| 核对 | 结果 |
|---|---|
| `manifest_words.jsonl` 与 `manifest.jsonl`：视频集合、split、T、n_frames、每个节点的 (a, b, 帧索引) | HateMM 1068 / HCS 393 / DeHate 6688 个视频全部一致，0 个结构差异 |
| 节点转录为空的比例（k30 → words）| HateMM train .556→.088、val .533→.103、test .512→.105；HCS train .561→.124、val .555→.096、test .567→.099；DeHate train .630→.074、val .639→.065、test .629→.070。说明 19:19 重建的 `manifest_words.jsonl` 已含 train/val 词级转录，不是早上只有 test 词的那一版 |
| `data/ASR_words/<C>/words.jsonl` 覆盖 | HateMM 1066/1068（缺 2 个 val 视频，见 N2），HCS 393/393，DeHate 6689 ≥ 6688 |
| 音频时长 − 网格 T | 三语料都在 [−1.00, 0.00] s，均值 −0.53；词中点 ≥ T 的词数为 0（没有词落到"最后一个节点兜底"分支） |
| shard 合并 | `words.jsonl` 每行与 shard 行完全相同；shard 间无内容不同的重复 id |
| `data.load_answers` 新 glob `answers_qwen7b_mod5[.]*jsonl` 与旧 glob `answers_qwen7b_mod5*.jsonl` | 三语料匹配到的文件列表完全相同；`answers_words_*` 不被 k30 glob 匹配 |
| 本地 `answers_words_qwen7b_mod5.shard*` | 只有 test（HateMM 215、HCS 79、DeHate 1341）；与 k30 逐视频节点集合与 T 一致，0 个不匹配；解析失败率两者都为 0 |
| `MAX_WORDS = 2000` 截断 | HateMM 超 2000 词的节点 k30 24 个、words 22 个（同一批超长视频）；HCS、DeHate 0。不是新问题 |
| `summarize_r4.py` Q1 参照 | `runs/20260925_query_paradigm_r3/summary.json` 的 `mean.fixed8` = HateMM .6555/.8674、HCS .6766/.6915；减 .005 得 .6505/.8624、.6716/.6865，与 README 15.2 写的门一致 |

## 2. 发现表

| 严重度 | 位置 | 问题 | 后果 |
|---|---|---|---|
| note N1 | `model.py:84-87` | `node_prior true` 时多构造一个 `nn.Linear`，其默认初始化消耗全局 torch RNG（随后置零）。同 seed 下 `node_prior true/false` 两次训练的其他层初值相同，但之后的 dropout mask 等随机流不同 | `no_node` 消融与 `full_rerun` 的差值里含一份随机流噪声（用户 2026-09-05 裁定该噪声 std .006–.009，.01 门已考虑）。不是 bug；解释 Q4 消融时知道即可 |
| note N2 | `data/AV2A_wav/HateMM/` 缺 `non_hate_video_559`、`non_hate_video_585`（val） | 这两个视频没有 wav，词级 ASR 与旧 K30 ASR 都没有记录，两版 manifest 里全部节点转录为空（VLM 得到 `(no speech in this part)`）| 与第 3 版相同，不是回归；只影响 2/109 个 val 视频的答案 |
| note N3 | `launch/run_ablations_r4.sh:23,25,27` | `c_label`、`e_independent`、`g_coupled` 三臂同时置 `node_prior false`（代码上必须：label 目标下 phi 无梯度、`TreeBatch` 与耦合链不支持 phi）。`summarize_r4.py` 只按臂名报差值 | 这三臂的"完整 − 消融"是复合差（臂改动 + 去节点势），不能单独归因于臂本身；README 15.2 写"(a)–(g) 七项重做"时须注明这三项是复合臂，`no_node` 臂给出去节点势的单独贡献 |
| note N4 | `concern_diagnostics.py:250`、`load_trial` | `main` 固定 `load_answers(a.corpus)`（k30），`load_trial` 不读 `summ["cfg"]["answer_source"]` | 只影响诊断脚本：若把它复用到第 4 版 trial，`reliability` 会用 k30 答案而不是训练用的 words 答案；oracle 路径不受影响（答案由 GT 生成）。`load_trial` 用 `summ["cfg"]` 建 `PriorNet`，节点头会正确加载。不进主结果链 |
| note N5 | `scripts/asr_words.py:117-127`（`--merge`） | 合并只读 `words.shard*.jsonl`，不读已有 `words.jsonl` | 本次全部走 shard，合并结果与 shard 逐行一致，未触发。若以后有 `--shard 0/1` 直接写 `words.jsonl` 再合并，会丢那部分 |
| note N6 | `data/vlm_tree/<C>/PROVENANCE.md` | "两次 build 的 test 节点转录相同"只对 HateMM 核过 | 机制上应成立（ASR 断点续跑不重转 test；shard 无冲突重复；已验证合并行 == shard 行），HCS / DeHate 建议同样核一次并写进 PROVENANCE。本地 `answers_words_*` 只有 test，`train.py:176` 在 val/train 答案到齐前会 assert 失败（响亮） |
| note N7 | `ctree._up:165-167` 与 `model.node_logits` | 根节点也算 phi 并进入 `_up`，在零膨胀下 G = 1 时 z_root ≡ 1，`v1 − v1_prior` 中抵消，无梯度 | 已在 `REVIEW_RULE4_R4.md` 5.3(1) 记录；无害 |

## 3. 六个问题的回答

### Q1 `answer_source` 是否到达所有读答案的地方
是。`train.py:173` 是整个训练脚本里唯一的答案加载点：`answers, T_ans = qdata.load_answers(corpus, cfg["answer_source"])`。下游全部由这个 `answers` 派生：训练目标 `train_obs`（:182）、锚定答案模型的拟合样本 `ta`（:192，只用 `ids["train"]`）、`Evaluator(store, answers, ...)`（:246，validation 每 epoch 选 checkpoint与 test 评测都经它）、`objective "posterior"` 分支（:282）。`data.load_answers` 的 glob `ANSWER_SOURCES[source] + "[.]*jsonl"` 对 k30 与旧 glob 匹配同一组文件，对 words 只匹配 `answers_words_qwen7b_mod5[.shard*].jsonl`。`train.py:178` 仍检查答案文件的 T 与 1-fps 网格一致。`evaluate()` 里 `vlm_verdict.load_verdicts(k=30)`（:453）读的是旧的整视频 verdict，只用来划"VLM 沉默组"子集，不是答案。

### Q2 `asr_words.py`
- **窗口偏移**：`windows()` 用 `linspace(0, duration, n+1)` 切等长窗；`Aligner.words` 返回相对窗口起点的秒数（`frame / fps`，`fps = 帧数 / 窗长`）；写出时 `ws + x`、`ws + y`（:170）。`spread()` 同样以 0 起算再加 `ws`。偏移正确。wav2vec2 帧数比 50 fps × 窗长少不到 1 帧，`fps` 相对误差 < 0.1%，30 s 窗末端时间误差 < 0.03 s。
- **中点规则**：`build_tree_manifest.load_words(asr="words")` 用 `0.5 * (s + e)`，与 k30 分支的 `0.5 * (s + e)` 同式；节点归属仍是 `a <= mid < b or (b == T and mid >= T)`（:113），两版同一行代码。
- **时长与 T**：T = VGGish 行数（`hc.video_duration`），wav 时长 − T 在 [−1, 0] s（三语料），没有词中点 ≥ T。词级转录不会落到网格外。
- 2 个 HateMM val 视频没有 wav（N2），两版都无转录。

### Q3 `node_prior`
- **训练 loss**：`train.py:306-307` 在 `objective "tree"` 分支里，`phi = model.node_logits(v_out, a_out, fo)`，与 s、g 同一次前向、同一个 `fo`（同 Tmax，`leaf_row` 编号一致），传入 `ctree.log_evidence(..., phi)`；`log_evidence` 两次 `up` 都带 phi（有答案 `w1a` 与无答案 `w1n`），`up` 内对 `v1`（有答案）与 `v1_prior`（无答案归一化）都传 phi。零膨胀下 `w1n = g`，所以 phi 只进入正例视频的答案项 `lo1`（与 REVIEW_RULE4_R4 5.3(2) 一致），不进标签项——设计如此。phi 未 detach，梯度到 `node` 与 `att`。
- **validation 选 checkpoint 与 test**：都走 `Evaluator.run` → `cpolicy.run_batch` → `policy.video_prior_nodes`（五 crop 前向，`model.eval()`，`no_grad`），phi 五 crop 平均后按 `fo.offs[b]` 填进 `PHI`；每一步 `ctree.marginals(fo, S, G, A3, chain, PHI)`（`cpolicy.py:65`）给后验与 EIG 用的 `m`；`record_voi` 的两次钳住推断也传 `PHI`（:100）。`marginals` 内 `up` 与带 eta 的 `_up` 都含 phi，因此 pG、E[z_n]、E[y_t] 三者都带节点势（穷举核对通过）。
- **节点顺序**：`Forest([T]*5, T)` 五棵同构树，`view(5, -1)` 每行是 `qtree.tree(T)` 顺序，N = 2T − 1；`run_batch` 的 `fo.offs[b]` 是视频 b 的根，其后 2T_b − 1 个节点同一顺序。已数值验证。
- **checkpoint**：`node` 是注册子模块，`best["model"] = deepcopy(model.state_dict())` 含 `node.weight/bias`，`torch.save` 到 `model.pth`。`concern_diagnostics.load_trial` 用 `DEFAULTS` 更新 `summ["cfg"]` 建 `PriorNet`（`summary["cfg"]` 是完整合并配置，含 `node_prior`），`load_state_dict` 严格加载；键不匹配会报错而不是静默。`pilot_words` 经 `cd.load_trial` 同理，且 `cpolicy.run_batch` 会自动带 phi。仅 N4 的答案来源问题。
- **闸门**：`train.py:219-221` 要求 `backbone macil`、`prior chain`、`chain_form zero_inflated`、`objective tree`、`answer_model != refit`；`refit_answer_model`（:392）、`objective label/posterior`（:290）、非链路径 `policy.run_video` 这几个不带 phi 的调用点都被闸门排除。`query_level > 0` 允许（f53aa57），`allowed` 只限制可问节点，phi 照常。

### Q4 默认配置是否严格重现第 3 版
是。逐项：(1) `ctree`：`phi=None` 时 `up`/`_up`/`marginals`/`log_evidence` 与 61c2c83 版逐位一致（1.2 节，误差 0.0）；(2) `cpolicy.run_batch` 改调 `video_prior_nodes`：`node_prior false` 时同一次前向，只是多解包 `v_out, a_out`（`ConstPrior` 也返回 6 元组），phi None → PHI None；`fo` 提前构造、`make_asker`/`answer_ll` 默认 None 都不改行为；(3) `model.py`：`node_prior false` 不建 `node` 层，state_dict 键与第 3 版相同，RNG 消耗相同（1.2 节验证），第 3 版 ckpt 可原样加载；(4) `data.load_answers` 新 glob 与旧 glob 在三个语料匹配同一组文件；(5) `train.py` 只在 `node_prior` 为真时算 phi；(6) `search.py` 的 `extra` 校验 `set(extra) <= set(DEFAULTS)`，新增两键在 `DEFAULTS` 中。

### Q5 消融启动器
`run_ablations_r4.sh` 的 hparams = `study_summary.json["extra"]`（`{"answer_source": "words", "node_prior": true}`，由 `search.py:128` 写入）← 更新 `best.params`（lr、λ_cma、dropout、lr_answer）← 更新臂 JSON；`train.py main` 拒绝未知键。合成顺序正确。`full_rerun={}` 与最优 trial 配置、seed 完全相同，作为消融基线；`summarize_r4` 同时报 `rerun − best_trial`。`no_node`（words，无节点势）与 `k30`（k30 + 节点势）加上 `full_rerun` 与第 3 版构成 2×2，可分开两项改动。置 `node_prior false` 的三臂（`c_label`、`e_independent`、`g_coupled`）在代码上是必要的（否则 `train.py:219-221` 断言失败或 phi 无梯度），但因此是复合臂（N3）。`b_flat` 在 `cpolicy` 里用第 0 步的树后验 `p0`（含 phi）作先验再加 LLR 均值，与第 3 版 (b) 的定义同构。`f_joint` 与节点势可共存（`anchored False`，`am` 进 `small` 参数组）。`grep '^fixed  8'` 与 `run.log` 的 `"fixed %2d calls"` 格式匹配。`run_diag_r4.sh` 的 JSON 由命令行给，`abl_noback`（`backbone const`）必须带 `node_prior false`，否则断言失败（响亮，不会出错数）。

### Q6 其他会改变结果的地方
- `extract_tree_answers.py`：`--manifest`/`--out-prefix` 只改输入输出路径；分片按 manifest 行号 `i % sn == si`，两版 manifest 视频顺序相同；`temperature 0`、`seed 0`；`done` 只读本 shard 文件（同前缀不同 shard 不互相跳过）。词级转录不会因 `MAX_WORDS` 截断产生新的差异（同一批 22–24 个 HateMM 超长节点两版都截）。
- `build_tree_manifest.py --asr words`：`words` 分支在 `REPRO_ASR` 判断之前返回，DeHate 也用 `ASR_words`（PROVENANCE 已写）；视频筛选 `hc.usable` 不变，结构一致已验证。
- `summarize_r4.py`：只读各 run 的 `summary.json`（评测器输出）；Q1 的 r3 参照数与 README 门一致；`claims` 里 `n_seeds == 3` 先于 `max(...)` 短路，缺 seed 时不会对 None 取 max；DeHate 不做 Q1 参照。
- 泄漏：词级 ASR 与 VLM 答案的生成不读标签；锚定答案模型只用 train 答案；validation 只用于选 checkpoint 与阈值校准；test 进搜索目标是用户裁定的既有协议，本次无新增。
- 评测器：test 数字仍全部经 `hc.write_scores` + `hc.run_evaluator`（`train.py:417-422`），validation 用 `hc.frame_metrics`，与第 3 版相同。
- 前置条件（不是 bug）：本地 `answers_words_*` 目前只有 test；val/train 到齐并 rsync 回本机前，`train.py:176` 会以 "videos without tree answers" 断言退出。开跑前应再核一次三个语料 `load_answers(c, "words")` 的视频数 = cohort 视频数、每视频节点集合与 k30 相同（本审稿对 test 已核，val/train 到齐后同一脚本可复用）。
