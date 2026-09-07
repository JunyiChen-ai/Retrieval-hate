# 规则 6 代码审查：候选 3 修订 4（修订 2 骨干 + 区间证据 HMM）

审查日期 2026-09-08。范围只限规则 6：机制是否真的进入 forward / loss / 最终分数、train/validation/test 泄漏、特征/时间/标签/split 对齐、超参与 checkpoint 加载链、是否调用统一评测器。不审风格、健壮性、理论。

审查对象（未提交、未跟踪的工作树目录）：`experiments/20260908_c3_rev4_rev2_backbone_interval_hmm/{model.py,train.py,search.py,launch/run_search.sh,README.md}`。对照：修订 3 `experiments/20260907_c3_rev3_interval_evidence/`（工作树与 commit 76ef6f0 一致，`git status` 干净）、修订 2 `experiments/20260904_evidence_guided_attention/model.py`。`src/hier_evidence_common.py`、`src/interval_evidence_hmm.py`、`scripts/run_locked_ablations.sh`、评测器自修订 3 审查后无改动（`git status` 无脏文件）。

数值检查脚本（只读，未训练）：`/tmp/claude-135258174/-home-jehc223-Retrieval-hate/85e4ce56-9930-4e7b-937a-61b295613068/scratchpad/check_rev4.py`，`~/miniconda3/envs/HateVideo/bin/python` 运行；下文数字都来自这一次运行。输入：B=3、T=20、seq_len=[20,13,7] 的随机特征（scaffold 六列按真实取值范围构造），所有模型 `eval()`，hid 128 / nhead 4 / ffn 128 / prior_scale 2。

## 结论

**PASS，无必修项。** 修订 4 的 `full` 与修订 2 的 `full` 在参数名一一对应、参数量相同（363,653）的条件下，六个 forward 输出与 `last_content_logit` 全部差 0；除预期的三处改动外，四个文件与修订 3 没有其它差异；训练、选点、评测链与修订 3 逐字相同。

## (a) 与修订 3 的 diff：全部 hunk

| 文件 | hunk | 内容 | 是否预期 |
|---|---|---|---|
| `model.py` | 1–34 行 | 模块 docstring 重写（机制说明、臂表） | 是（文字） |
| `model.py` | `STRUCT_ARMS` | `key_bias`、`ctx_in_rep` 去掉；加 `no_lin`、`gated_bias`、`ctx_on_logit` | 是（改动 2） |
| `model.py` | `EvidenceEncoder.__init__/forward` | 新增 `lin` 开关；`assert cell or lin`；cell 只在 `cell and lin` 时零初始化；`lin=None` 时不加线性项 | 是（改动 2，`no_lin`） |
| `model.py` | `ERCA.__init__` | `enc` 传 `lin=(arm != "no_lin")`；`bias_mode` 默认由 `"gated"` 改 `"key"`，`gated_bias` 映射到 `"gated"`；`ctx_mode` 默认由 `"logit"` 改 `"rep"`，`ctx_on_logit` 映射到 `"logit"` | 是（改动 1、2） |
| `model.py` | forward 两行 | 只改注释文字（`revision-2 placement (full)`、`ctx_on_logit arm only`） | 是（文字） |
| `train.py` | 1–8、10–20 行 | 只改 docstring（训练描述、路径、臂表） | 是（改动 3 + 文字）。`DEFAULTS`、`TRAIN_ARMS`、`FUSION_ARMS`、`ABLATIONS`、`train()`、`main()` 与修订 3 逐字相同（`diff` 无其它 hunk） |
| `search.py` | 3–4 行 | 只改 docstring 中的路径 | 是（改动 3）。`sample()`、objective、best 选取、summary 写法未动 |
| `launch/run_search.sh` | 2、5、10 行 | 路径改为新实验 id / `runs/20260908_...` | 是（改动 3） |

`BiasedMultiHeadAttention`、`EvidenceRoutedCMA`（含 routing、q/k 加码、key padding 屏蔽）、`ERCA.bag`、`ERCA.forward` 的计算语句全部未改。

## (b) 修订 4 `full` 的数值等价性

| 比较 | 权重对齐 | mmil / sig_a / sig_v / av_log / v_out / a_out / last_content_logit 最大绝对差 |
|---|---|---|
| **修订 4 `full` vs 修订 2 `EGCA full`** | 修订 4 权重拷入修订 2，仅 `cma.beta.*`→`cma.bias.*` 改名；两边无剩余未匹配参数；参数量 363,653 = 363,653。拷贝前把修订 4 零初始化的 β 与 cell 嵌入随机化，使比较覆盖偏置与四格嵌入 | 全部 **0** |
| 修订 4 `full` vs 修订 3 `ctx_in_rep`（修订 3 gate 零初始化 ⇒ g≡1） | 同名拷贝；修订 3 多出 `cma.gate.*` 未填 | 全部 **0**；对照：把修订 3 的 gate 随机化后 av_log 差 .113，说明比较不是空的 |
| 修订 4 `full` vs 修订 3 `key_bias`，两边 `ctx=None` | 同名拷贝 | 全部 **0**（偏置部分完全一致） |
| 修订 4 `full` vs 修订 3 `key_bias`，两边开 ctx | `ctx` 形状不同（128×128 vs 1×128），不可拷 | v_out/a_out 差 1.08、av_log 差 .138——这正是 c 放法的差异，预期 |

结论：修订 4 `full` = 修订 2 骨干，逐元素相等；= 修订 3 `key_bias` 的偏置 + 修订 3 `ctx_in_rep` 的 c 放法。

## (c) 修订 4 `full` 中 c 与先验的位置

| 项 | 结果 | 证据 |
|---|---|---|
| `last_calibration` 为 None | 通过 | forward 后 `r4.last_calibration is None == True`（修订 3 `key_bias` 为 (3,1) 张量作对照） |
| c 进入返回的 v_out / a_out（CMAL 读的就是这两个） | 通过 | 同权重的 `full` 与 `no_context` 相比，v_out、a_out 逐行差 = 每视频常数 `ctx(mean_t e_t)`，含 padding 行，最大偏差 2.4e-7；\|c\| 最大 1.08 |
| 块级 MIL 读的 `last_content_logit` 含 c | 通过 | `last_content_logit == fc(a_out) + fc(v_out)`（返回的、已加 c 的表示），差 0；`av_log(full) − av_log(no_context) == 2·fc(c)`，差 2.8e-7，即 c 只经共享头进 logit，没有第二条 logit 通路 |
| 先验只加在 logit | 通过 | `av_log − last_content_logit == prior_scale · ell / ELL_SCALE`，差 6e-8；v_out/a_out 不含先验（见上一行分解） |
| key padding 屏蔽 | 通过 | 改写 padding 行的 f_a/f_v 后，有效行的 av_log/v_out/a_out 与 mmil 差 0 |

`train.py` 侧：`CMAL(mmil, a_log, v_log, seq_len, v_out, a_out)`（第 187 行，`fix_rep_swap=False` 同修订 2/3），`block_bag_loss(model.last_content_logit, ...)`（第 192 行）——两者拿到的都是含 c 的量，与 README 第 1 节和 train.py docstring 所述一致。

## (d) `no_lin` 臂

| 项 | 结果 | 证据 |
|---|---|---|
| 有 cell 嵌入、无线性层 | 通过 | `enc.lin is None`，`enc.cell = Embedding(4, 128)` |
| 嵌入不是零初始化 | 通过 | `model.py:58-59` 只在 `cell and lin` 时清零；实测 `no_lin` 的 cell 权重 std .935、范围 [−3.25, 3.28]（nn.Embedding 默认 N(0,1)）；`full` 臂 cell 仍零初始化（与修订 2/3 相同，从线性映射起步） |
| 列 0–1（ell、p_s）不进编码器 | 通过 | 扰动 ell、p_s：v_out/a_out 差 0；翻转 b_fine：v_out 差 .84（嵌入在用）。av_log 差 2.31 来自先验项 `prior_scale·ell`，那是设计（先验不属于编码器） |

## (e) `train.py`

| 项 | 结果 | 证据 |
|---|---|---|
| `ABLATIONS` 与 `STRUCT_ARMS` 一致 | 通过 | `ABLATIONS = STRUCT_ARMS + TRAIN_ARMS + FUSION_ARMS`，19 项，含 `no_lin/gated_bias/ctx_on_logit`，不含旧名；`--ablation choices=ABLATIONS`；结构臂经 `arm = ablation if ablation in STRUCT_ARMS else "full"` 进 `ERCA`。`make_scaffold_fn` 只对 `mean_prior/mean_prior_all/raw_block_label/indep_hmm/flat_coarse` 这些名字分支，新臂名不会误触 |
| 10 个结构臂开关 | 通过 | 实例化表：avce = fc_a 900 维、无 enc/β/ctx、qk_enc False；no_qk_enc = qk_enc False；no_cell = Linear(4→128)；no_lin 见 (d)；no_bias = β None；gated_bias = β(128→4)+gate(128→4)；shared_bias = β(128→1)；no_context = ctx None；ctx_on_logit = ctx(128→1)、`last_calibration` (3,1)；其余臂 bias_mode=key、ctx_mode=rep |
| `--config` 覆盖 | 通过 | `train.py:262-265` `cfg = dict(DEFAULTS); cfg.update(json)`；`DEFAULTS` 与修订 3 逐字相同（`fusion=interval`、`normalized_time=True`、`positive_constraint=True`、`w_fine=1.0`） |
| test 标签不进训练/选点 | 通过 | `test_gt` 只用于筛 `test_ids`（第 98、101 行）；HMM 只用 `train_ids` 的视频标签拟合（第 122 行）；checkpoint 只按 `crit = (val AP + val ROC)/2` 更新 `best_state`（第 204–215 行）；test 在第 226 行起才打分 |
| 统一评测器 | 通过 | `hc.run_evaluator` → `scripts/reproduction_baselines/eval_baseline_scores.py --corpus --split --scores --json-out`，`metrics.json` 由它写出；`summary.json` 是转录 |
| 输出目录 | 通过 | `train.py` 只写 `--out-dir`；`search.py` 的 `out_dir = <out-root>/<corpus>/seed<seed>/trial<k>`，launcher 传 `--out-root runs/20260908_c3_rev4_rev2_backbone_interval_hmm` |

## (f) `search.py` / `launch/run_search.sh`

| 项 | 结果 | 证据 |
|---|---|---|
| 5 个标量与 README 第 2 节一致 | 通过 | `sample()`：lr log[1e-4,1e-3]、max_seqlen {150,200,300}、lamda_cma [0.5,2]、prior_scale log[0.5,8]、lambda_block log[0.05,2]；不采样 w_fine/dropout/lamda_cof |
| out-root | 通过 | `run_search.sh` 第 10 行 `--out-root runs/20260908_c3_rev4_rev2_backbone_interval_hmm`；`search.TRAIN` 解析为本目录 `train.py` |
| best = COMPLETE 中目标最高 | 通过 | 第 149–154 行；目标 = test (AP+ROC)/2（第 114 行） |
| 不剪 within | 通过 | `--no-within-prune` 默认 True ⇒ `floor=None`，`TrialPruned` 分支不可达 |
| 预算 | 同修订 3 | 首 trial ≤ 1 h 则 20，否则 5；README 写 20。既有流程 |

## (g) `scripts/run_locked_ablations.sh` 对本实验 id 的解析

id `20260908_c3_rev4_rev2_backbone_interval_hmm` 通过第 10 行正则；第 24 行先试完整 id，`experiments/<id>/train.py` 存在，直接命中：`trainer=experiments/20260908_c3_rev4_rev2_backbone_interval_hmm/train.py`。（尾部剥离 `sed -E 's/_(v|rev)[0-9]+(_[a-z0-9]+)?$//'` 对这个 id 不匹配，原样返回，也不会误剥中间的 `_rev4_rev2_`。）消融本轮不跑，但启动链可用。

## 非阻断记录（不改变观察，不要求修）

1. `launch/run_search.sh` 第 3 行注释仍写"candidate 3 revision 3"；`search.py` 第 37 行注释仍写"declared before any search, 2026-09-07"。只是注释。
2. 本目录目前未跟踪（`git status` 显示 `??`）。README 第 3 节计划在 uoa-lab1 / uoa-lab3 跑，按 CLAUDE.md 远程机准备第 1 条，开跑前必须先 commit + push、远程 `git pull`，否则远程拿不到这份代码；`run.log` 首行的 `_git_describe()` 也才有对应的 commit。
3. `full` 臂的 cell 嵌入零初始化（起步等于线性映射）与修订 2/3 相同，不是本轮改动；只有 `no_lin` 臂改用默认随机初始化，见 (d)。

## 必修项

无。
