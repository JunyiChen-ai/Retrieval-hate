# 规则 6 Code review：20260906_hier_evidence_clean

**结论：PASS**（没有发现会改变实验观察或结论的 bug；下面第二部分是几条不阻塞开跑的备注）。

审阅日期 2026-09-06，独立 reviewer。审阅对象：`experiments/20260906_hier_evidence_clean/{train.py,search.py,launch/run_search.sh,README.md}`，对照上游 `experiments/20260903_hier_evidence_mil/{train.py,search.py}`，以及共享代码 `src/hier_evidence_common.py`、`src/verdict_hmm.py`、`src/vlm_verdict.py`、`scripts/reproduction_baselines/macilsd/{avce_network.py,CMA_MIL.py,train.py}`、`scripts/run_locked_ablations.sh`、评测器 `scripts/reproduction_baselines/eval_baseline_scores.py`。没有跑训练；做了 import 检查和一个只读文件存在性/split 覆盖的纯 python 检查（本机与 uoa-lab1、uoa-lab3 各跑一次）。

## 一、逐项核对结果（对应任务的 7 个检查点）

### 1. 与候选 1（w_fine=1，lamda_a2b=lamda_a2n=1，lamda_cof=0.05）的等价性：等价

逐段对照 `train.py` 新 95–129 行 vs 旧 158–216 行（forward）、新 212–271 行 vs 旧 382–451 行（训练循环与选 ckpt）、新 273–299 行 vs 旧 453–483 行（打分与评测）：

| 环节 | 结论 |
|---|---|
| forward | 相同：拷贝 f_a → 清零簿记列（COL_PH、COL_BLOCK）→ ell 列除以 ELL_SCALE → hide_input 时清零四列 → AVCE → 记录内容 logit → 加先验 `prior_scale * ell / ELL_SCALE`（ell 取自原始 f_a）→ top-⌈T/16⌉ bag。 |
| 损失 | 相同：BCE(bag) + lam_a2b·(c1+c3) + lam_a2n·(c2+c4) + λ_block·block_bag_loss；lam 的 `min(lamda, cof·epoch)` 预热相同；伙伴 uni 损失与两次 backward/step 顺序相同。旧版多出的 chain_distill 项在 full 下权重为 0，新版直接删除，不改数值。 |
| EMA | 相同：`distil_step(a, model.av, partner, epoch)`，`m=0.91`、`ema_epochs=50`。 |
| checkpoint 选择 | 相同：每 epoch 用 `hec.frame_metrics` 算 validation pooled AP/ROC 均值，严格大于才更新，deepcopy state_dict。 |
| 打分与评测 | 相同：`hec.score_split`（五 crop sigmoid 均值，index_map 映射到 1 fps）→ `hec.write_scores` → `hec.run_evaluator` 调唯一评测器 `eval_baseline_scores.py --corpus --split --scores --json-out`，读 `results.score_av`。 |
| 随机数流 | 相同：`runtime.setup_seed(seed)` 后依次建 dataset、`Candidate`（AVCE）、`Single_Model`、DataLoader(shuffle)，顺序与旧版一致；旧版多 import 的 `evidence_chain`、`null_token_cma`、`dataset` 模块没有任何 seed/全局状态副作用（grep 无 manual_seed/backends）。 |
| 数据加载 | 相同：旧版的 `dataset.py` 只是 `src/hier_evidence_common.py` 的 re-export，新版直接用 `hec.*`，同一份类。`process_feat(..., is_random=False)`、`crop_repeat=5`、val/test 全长五 crop 相同。 |
| 脚手架列 | 相同：`hec.make_scaffold_fn(hmm, binary, ablation, 1.0)`，六列布局、`N_INPUT_SCAF=4`、`SCAF_OFFSET=896`、`A_EXT_DIM=902` 与旧版 import 检查值一致；`ELL_SCALE` 数值相同（13.8155）。 |
| 裁定缓存 tag | 旧版 `cfg.get("fine_tag","qwen")`，新版固定 `"qwen"`；候选 1 seed234 两语料 best trial 的 `hparams.json` 里没有 `fine_tag` 键（走默认 qwen），所以一致。 |
| 共享代码漂移 | `git diff b2bdca2..HEAD` 显示 `src/hier_evidence_common.py` 自候选 1 以来只新增了未被调用的 `load_fixed_cohort`；`verdict_hmm.py` 无改动；`vlm_verdict.load_verdicts` 只新增默认关闭的 `strict` 参数。评测器、`frame_eval_common`、macilsd 无改动。 |

仅有的行为外差异都在日志/记录字段：`tot` 少一个 `chain_distill` 累加项，`history` 无该键；`study_summary.json` 的 `within_floor` 写 `None`、`ablation` 固定 `"full"`。不影响任何数字。

### 2. 先验与块损失确实进入损失和最终分数：是

- 先验：`train.py:126-128` 在 forward 内加到 `av_log`，bag 与 BCE 用加了先验的 logit；`hec.score_split` 也走同一个 forward，所以 val/test 分数含先验。
- 块损失：`train.py:238-240` `lambda_block>0` 时 `hec.block_bag_loss(model.last_content_logit, ...)` 加入 `total`。
- 内容 logit：`train.py:125` 在 `av_log = av_log + prior` 之前赋值 `self.last_content_logit = av_log`；因为后一行是新建张量而非原地加，`last_content_logit` 保持为加先验前的 z。块损失作用在 z 上，符合 README 与 docstring。

### 3. 10 个消融臂与 README/docstring 一致：是

`ABLATIONS` 元组（`train.py:87-88`）与 docstring 28–38 行、README 第 4 节第 4 条的 9 个非 full 臂一一对应。`mean_prior`/`indep_hmm`/`flat_coarse` 字符串与 `hec.make_scaffold_fn` 内的判断（`hier_evidence_common.py:318,320,329`）完全相同，分别对应 `independent=True`、`flat_coarse=True`、ell 与 P(s) 替换为两粒度均值。`no_block`→λ_block=0（且不再计算块损失）；`no_input`→四列清零但先验仍从原始 f_a 取 ell；`no_prior`→prior_scale=0；`no_verdict`→三者同时；`no_ema`→跳过 `distil_step`；`no_cmal`→`lamda_a2b=lamda_a2n=0`，与旧版逐条相同。旧版的 `raw_block_label` 不在新臂里，`make_scaffold_fn` 中该分支不会触发。

### 4. 泄漏：无

- HMM：`hec.fit_hmm(corpus, train_ids, labels, binary)` 只取 train_ids 的正负例（`hier_evidence_common.py:307-311`）；裁定文件按 video id 加载三 split，但拟合只用 train，val/test 的后验只用其 VLM 裁定，不碰任何标签。
- test 标签：`test_gt` 只用来过滤 `test_ids`（`train.py:155`），此后只在训练结束后交给评测器；训练循环、`crit`、`best_state` 都不读 test。
- validation：只用于每 epoch 的 `crit` 和 ckpt 选择；`hate_ids` 含 test 视频标签但只进 `frame_metrics` 的 within 宏平均，且 within 不参与 `crit`。
- Optuna 目标用 test（规则 7/10 允许，README 第 3 节已写明）。
- 纯 python 检查：两语料 train/val/test id 互不重叠；HateMM 744/109/215、HCS 251/63/79 全部有特征、有裁定、有文本；val GT 覆盖 100%，test 只缺预期的 `hate_video_427`。

### 5. search.py：符合修订后的规则 7

- 无 within 剪枝：objective 直接 `return obj`，没有 `TrialPruned`（对比旧版 122–125 行已删除）。
- 目标 `(test pooled_ap + test pooled_roc)/2`（`search.py:100`）。
- budget：首 trial 后按 `seconds<=3600 → 20 否则 5` 写 `budget.json`，重跑时读回不再改（69–73、110–118 行），与旧版相同。
- `study_summary.json` 含 `n_trials`、`trials[].state`、`best.number`、`validation_selected`（137–141 行），`scripts/run_locked_ablations.sh:16` 需要的字段齐全。
- `SEARCH_DONE` 在 summary 写完之后才写（145–146 行）。

### 6. 超参链：生效

`search.py:sample` 的 5 个键写入 `trial<k>/hparams.json` → `train.py --config` → `cfg.update` → `Args(cfg)`：`lr`→两个 Adam（`train.py:206-207`）；`dropout`→`AVCE_Model(cfg)`/`Single_Model(a)` 读 `args.dropout`；`max_seqlen`→`hec.TrainDataset`；`prior_scale`→`Candidate`；`lambda_block`→块损失权重。`config.json`/`summary.json` 快照的是合并后的 cfg。锁定消融用 `run_locked_ablations.sh` 把 best trial 的同一份 `hparams.json` 传给 `--config`，臂内的覆盖（prior_scale=0 等）在 cfg 之后生效，链正确。

### 7. 启动脚本在 uoa-lab1/uoa-lab3 上：可用，但有一个前置条件

`run_search.sh` 只依赖 `$HOME/Retrieval-hate` 与 `$HOME/miniconda3/envs/HateVideo/bin/python`，子进程都用 `sys.executable`，评测器同解释器。ssh 只读检查（2026-09-06）：两台机器 python 存在，optuna 4.9.0、torch 2.7.1+cu128、CUDA 可用，GPU 空闲；split/GT/特征/裁定/文本覆盖数与本机完全相同，仓库均在 `445ec1f`（与本机 HEAD 和 origin/main 一致）。**前置条件**：`experiments/20260906_hier_evidence_clean/` 目前在本机是未跟踪状态（`git status` 显示 `??`），远程机上没有这个目录；按 CLAUDE.md 必须先 commit + push，远程 `git pull` 后再启动，否则 `run_search.sh` 会找不到 `search.py`。

## 二、不阻塞的备注（不是规则 6 意义上的 bug，只记录）

1. `search.py:105-108,110-121`：崩溃的 trial（`catch=(RuntimeError,)` 后状态 FAIL）计入 `n_done`，会占掉一个预算名额，且 `study_summary.json` 里会出现 `state=FAIL`，`run_locked_ablations.sh:16` 的断言（全部 COMPLETE/PRUNED）会拒绝启动消融链；若恰好是 trial 0 崩溃，`budget` 会按崩溃前的耗时定。与候选 1 的 search.py 行为完全相同，不改变任何已完成 trial 的数字。如果真的遇到，可行的处理是把 FAIL 排除出 `n_done`（只数 COMPLETE/PRUNED）并让 `run_locked_ablations.sh` 容忍 FAIL；这属于流程改动，按规则应先声明再改，本轮不要求。
2. `train.py:114` bag 取 `ceil(T/16)` 个，MACIL-SD 上游 `clas` 取 `T//16+1` 个，T 为 16 的整数倍时相差 1 个。这是候选 1 就有的写法（旧版 180 行相同），README 写的 "top-⌈T/16⌉" 与代码一致，只是 docstring 里 "MACIL-SD verbatim" 略有夸大。不影响新旧等价，不改。
3. README 第 1 节把 `w_fine` 固定为 1 的依据引自候选 1 README 4.9 节的 validation 选择；候选 1 两语料 seed234 的 test 选出的 best trial 实际 `w_fine≈0.73`、`lamda_cof≈0.084–0.094`。这是研究决策，不在规则 6 审查范围；README 预注册第 3 条已经把 `lamda_cof` 列为唯一回查项，这里只提醒后续解读结果时记得这一点。
