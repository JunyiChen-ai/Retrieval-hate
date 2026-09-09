# 规则 6 代码复核：机制 T（文本证据，模块一第 2 轮）

复核对象：`git diff --cached`（`src/interval_evidence_hmm.py`、`src/hier_evidence_common.py`、`experiments/20260910_online_query_within/{model,acquire,train,text_eval}.py`、`scripts/build_text_hate_scores.py`、README 第 7 节；`launch/run_diag.sh` 本次无改动，仍是 79be95a 版）。复核日期 2026-09-10，只读，CPU 检查（临时脚本在 scratchpad：`check_hmm_text.py`，`old_ieh.py` = HEAD 版 HMM），未写 `runs/`、`data/`，未启动训练。

## 结论：**必须修一处**（修完即 PASS，不需重开 review）

### 必修 1：ASR 有一半视频是词级时间戳，`build_text_hate_scores.py` 把单个词当"分块"送进分类器，这些视频的 ASR 证据近似常数

- 数据事实（`data/ASR/*/…_asrK30_whisper-large-v3.jsonl` 的 `timestamps` 字段）：HateMM 1066 条 ASR 记录中 **526 条是 `word`**（每条 chunk 平均 1.0 个词、144 条/视频、29% 的 chunk 时长为 0，且同一串词常共用同一时间戳），540 条是 `chunk`（52 词/条、13 条/视频）；HateClipSeg 148 `word` / 246 `chunk`。
- 脚本 `scripts/build_text_hate_scores.py:65,77` 直接遍历 `chunks`，不区分模式，`:88-93` 每个词映射到 1 秒后取平均、权重取 max（= 1）。分类器对孤立单词几乎恒为 0：本机 CPU 实测 " exterminate" .002、" white" .002、" The" .002，而整句 "We have to exterminate white people off of the face of the planet…" .996。
- 后果：(a) 49% 的 HateMM 视频、38% 的 HCS 视频，ASR 族逐秒 p 几乎全落在第 0 档（p < .01），与内容无关；(b) 这批秒同时来自正、负例视频，混进 EM 的 2×10 表，稀释 chunk 模式视频的 LLR（已拟合表 `hmm_text/hatemm/text_w1_params.json` 的 ASR 档 LLR 为 −.42, −.07, .65, 1.97, 2.08, 2.04, .75, .99, 1.37, 2.02，非单调即为混合迹象）；(c) 权重语义在两种模式下不同：chunk 模式一条 20 s 句子总权 1，word 模式每个语音秒权 1（实测 HateMM val 语音秒中仅 20% 权重为 1，HCS 13%）；(d) T1 门（README 第 7 节表）与 `within_oracle` 的 .601/.555 都是在这份缓存上算的。
- 修法：对 `timestamps == "word"` 的记录先把词合并成有真实时间跨度的片段再打分——最省事是所有视频统一用 jsonl 里现成的 `window_text` / `window_bounds`（K = 30 窗，与 OCR 同结构，w = 1/窗秒数；注意 `window_bounds` 是 snippet 网格、首窗较短，按给定边界映射即可），或按停顿（相邻词间隔 > 1 s）合并成句并保留首末时间戳。两种模式必须用同一规则。重建 `data/text_hate/` 与 `PROVENANCE.md`（写明 word/chunk 两种模式的处理），重跑 `text_eval.py` 刷新 T1 表，再开搜索。零 VLM 调用、不训练，分钟级。

## 已确认无问题的项目

### 1. 泄漏
- 文本缓存：`build_text_hate_scores.py` 只读 ASR/OCR jsonl 与 VGGish 行数；ASR 记录里带 `label` 字段，脚本未读取。不碰 GT npz。
- HMM 拟合：`hc.fit_hmm`（`src/hier_evidence_common.py:421-439`）仍只按 train id 用视频标签分正负；`text_obs` 对 val/test 只在推理时作为免费观测进入 `posterior`/`infer`。`text_eval.py` 在 train 拟合、test 只打分，`gt_arrays` 只用作 id 过滤。
- 训练：`window_targets`（`train.py:248-268`）用 `hmm.infer(..., xt=text_obs.get(vid))`，train 视频、目标窗裁定不在输入里，同上一轮。`no_verdict` 分支把 `text_llr` 一并清零（`model.py:204-207`）。

### 2. HMM 文本族（`src/interval_evidence_hmm.py`）
- 发射一次性：`_emissions:355-359` 对每段乘一次 `exp(ll[:, S_OF])`，`ll` 由 `text_counts` 的加权计数得来，每个秒观测只进一个段（`text_counts:152-167` 的 `(t+.5)/T` 段号与 `rows_from_segments` 在秒网格上逐位一致，实测相等）；档号 `searchsorted(side="right")` 与 `hc.text_llr_seconds:442-460` 同式（p = .01 → 档 1，.0099 → 档 0）；NaN、w = 0 的秒被丢弃；T = 0 返回全零。
- 逐段减最大值（`:358`）：合成数据实测 gamma/xi 差 1e-16、`post_w`/`rho` 差 0；正例约束路径 `_log_all_zero` 与 `_fb` 用同一个 e，`lz0 − lz` 不变；R = 3 时 `text_emit` 跨档共享，`logw` 各分量同移，责任度不变；`pred_fine[w] = exp(logmarg(E∪{b_w=1}) − logmarg(E))` 仍成立（.0593 = .0593）。
- M 步（`:506,538,553`）：正例 `ps.T @ C`（P(s) 加权计数），负例计入 s = 0 行，每格 1e-3 平滑后按行归一；无任何文本时（`any_text` 假）表保持均匀 → LLR 全 0。`text_weight` 是全局标量，不改变 M 步极值点。真实似然（把移位加回后另算）合成数据 8 轮单调上升（−7090.9 → −4113.0）；R = 3 四轮无 NaN。
- `text=False`：与 HEAD 版逐位一致（正例约束开/关各 6 轮 EM：参数、`fit_history`、`infer` 差 0.00e+00；传入 `xt` 被忽略）。`from_params`/`params` 往返：`text`、`text_weight`、`text_emit` 一致，`infer` 差 0。旧调用方 `20260908_adaptive_vlm_query/acquire.py:102` 与 `adaptive_query_replay.py:81` 的 `_emissions(bf, bc)`、`interval_hmm_eval.py:55` 的 `IntervalEvidenceHMM(k, j)` 仍可用。

### 3. `hier_evidence_common` 第 7 列
- `scaffold_rows:127`（index HMM 路径）补零列；`scaffold_rows_interval:481-489` 由 `text_llr_rows` 用 `resample_intervals` 把 (T,) 秒级 LLR 重采到 snippet 行（实测 T = 104 = n_seconds，154 行 = snip 行数）；无文本 → 零列。`text_llr_seconds` 的 `hmm.text_weight` 与 `_text_loglik` 同一温度。
- 仓库内其它使用者：本轮及 20260907/20260908 各 model 用 `fc_a = Linear(SCAF_OFFSET)`、`evid` 只取 4 列，不受影响；`interval_observation_data.py:49` 取 `[:SCAF_OFFSET]`；`20260905_interventional_evidence` 与 `score_latent_initialization.py` 用同一 `A_EXT_DIM` 常量自洽。受影响的只有 `20260903_hier_evidence_mil/analysis_backbone_mechanism.py:50-53` 与 `20260906_hier_evidence_clean`：`a_feature_size = A_EXT_DIM` 变宽 1，再加载旧 `model.pth` 会形状不匹配——都是已淘汰/归档分析，不属本轮观察（见备注 3）。

### 4. `model.py`
- `text_llr = clamp(f_a[..., COL_TEXT] / 5, −1, 1)`（`:204`）；已拟合表的 |LLR| ≤ 2.3，缩放后 ≤ .46，裁剪不触发。`no_text_input` 是结构臂（`STRUCT_ARMS`）→ `enc.lin = Linear(2)`。
- `last_content_logit = fc(a_out) + fc(v_out)`（`:228`）：文本只经证据编码器 → CMA/ctx 进入，与 ell、P(s) 同一路径；没有单独的加性文本项，先验项仍只有 α·ell（`:232`）。窗损失与 eoc 权重读到的 content logit 语义与第 1 轮相同。

### 5. `train.py` / `acquire.py` 接线
- `text_obs`（`:189`）按 `all_ids` 建，`grid = ieh.make_grid(K_FINE, J_COARSE)` 与 `hmm.grid` 同构；`text_arrays` 只在 `text_input` 时读（`:190`）。`refit`（`:213-226`）每次用新表重算 `text_llr` 并换 `masked_fn`（默认 scaffold 从未被输入，同上一轮）；`Acquirer` 两处（`:348,376`）都传 `text=text_obs`；`acquire.py` 5 个 `self.state(` 调用点都带 `vid`，`state:127-128` 传 `xt=self.text.get(vid)`，`cache.build` 走含文本的 `masked_fn`。
- `no_text` 臂：`use_text=False` → HMM `text=False`、`text_obs={}`、`text_input=False`（`:166-167`）。
- DataLoader 非 persistent、`fork`，闭包引用的 `text_obs`/`text_llr` 字典随每 epoch 重新 fork 复制，与上一轮已验证的模式相同。

### 6. `build_text_hate_scores.py` 其它
- HF 模型 `id2label = {0: NOT-HATE, 1: HATE}`，`softmax[:, 1]` 正确。OCR 窗 k → 秒 `[int(kn/K), int((k+1)n/K))`，与均匀 D/K 窗（`t_mid` 反推）及 HMM 细窗分数网格差 ≤ 1 s；conf ≥ .5 过滤；w = 1/窗秒数。
- 覆盖：两语料 T 与 `video_duration` 全部一致（各抽 40）；`p` 有限 ⇔ `w > 0`；HateMM val 的 `non_hate_video_559/585` 有 npz 但 ASR 全 NaN（两个视频没有 ASR 记录），`text_observation` 退化为只含 OCR 或 None，不崩。顺序确定，GPU 批内 padding 只带 1e-6 级差异。

## 建议（非必修）

1. **`fit_history` 与 `infer()["logmarg"]` 不是真实对数似然**：`:358` 的移位常数 Σ_g max_s ll_g(s) 依赖当前 `text_emit`，每轮 EM 都变；负例走未移位的 `_neg_loglik_z:453`。已保存的 `hmm_text/{hatemm,hateclipseg}/summary.json` 中 `text_w1.loglik_monotone` 均为 **False**，`fit_loglik_last`（−19014 / −5968）与 `notext`（−9242 / −3899）不可比；README 第 7 节第 115 行"EM 训练集对数似然逐轮单调（已断言）"目前断言的是这个伪量。参数与后验不受影响（第 2 节实测），`logmarg` 无人用于决策（`acquire.py`/`train.py` 无引用）。修法一行：`_emissions` 把每段移位量记下来，`_posterior_video:435` 的 `logmarg` 加回 Σ 移位（或在 `_text_loglik` 之外单独返回常数），`text_eval.py:87-88` 的两个字段即恢复意义。
2. `no_text` 臂的模型仍按 `cfg.text_input=True` 建（`train.py:278` 传的是 `a`，`model.py:167` 读 `cfg.text_input`，`no_text` 不在 `STRUCT_ARMS`）：`enc.lin` 为 `Linear(3)`，第 3 列输入恒 0，功能等价，但初始化随机流与 `Linear(2)` 不同，且 `results["text"]["input"]=False` 与实际结构不符。建议 `train.py:167` 之后 `a["text_input"] = text_input` 再建模。
3. `SCAF_DIM` 变 7 后 `20260903_hier_evidence_mil/analysis_backbone_mechanism.py`、`20260906_hier_evidence_clean` 的旧 checkpoint 不能再加载（输入宽度 +1）；若以后要复现那些分析，需在加载时按旧宽度建模。属存量，不影响本轮。
4. 必修 1 修完后，word 模式若改用 `window_text`，ASR 族与 OCR 族的时间分辨率都变成 K = 30 窗，README 第 7 节"逐秒"措辞与 PROVENANCE 需同步改写；T1 表重算后再决定是否开搜索。

## 判定
- 必修 1（重建文本缓存、统一 word/chunk 两种 ASR 记录的打分粒度，重跑 `text_eval.py`）修复后 PASS。按规则 6 只确认修复，不重开 review；建议 1 顺手一并修，因其让 README 的单调性断言可核。
