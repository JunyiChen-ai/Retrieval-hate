# 规则 6 代码复核：adaptive VLM query 第 1 轮（policy_start=coarse）

复核对象：commit 16149a1、e3436ce（另 35fe0fa 仅 README 预注册行）。复核日期 2026-09-08，只读，CPU 检查；
临时脚本在 scratchpad（`sim.py`），未写 `runs/`、`data/`。机器 bash 5.2.21。

## 结论：**必须修一处**（修完即 PASS，不需重开 review）

### 必修 1：`getattr(a, "policy_start", "seeds")` 在缺 key 时不会回退，直接 KeyError

- `train.py:85` `class Args(dict): __getattr__ = dict.__getitem__`。`dict.__getitem__` 缺 key 抛 **KeyError**，而 `getattr(..., default)` 只吞 AttributeError。
- 实测（用真实 `train.Args` + 第 0 轮 `runs/20260908_adaptive_vlm_query/hatemm/seed234/trial14/hparams.json`，该文件只有 5 个搜索键，无 `policy_start`）：
  `getattr(a, "policy_start", "seeds")` → `KeyError: 'policy_start'`；`a.get("policy_start", "seeds")` → `"seeds"`；带 `policy_start` 的第 1 轮 config → `"coarse"`。
- 后果：任何**不含** `policy_start` 的 config 在 `rounds ≥ 2` 的 arm（`full`、`no_missing_state`）训完 round 0 后在 `train.py:261` 崩溃。
  受影响：(a) 第 0 轮设计的任何重跑——README 第 0 行写明的“先补 HateMM seed 2025/3407（当前设计）”若不带 `EXTRA_CONFIG` 开跑，每个 trial 都在 round 0 训完后失败，`search.py` 用 `catch=(RuntimeError,)` 吞掉后继续，会把 20 个 trial 的预算全部烧成 FAIL；
  (b) 第 0 轮 `ablations` 链（HateMM seed 2025/3407 的 `full`/`no_missing_state` arm）会 `exit 1`；
  (c) 不带 `--config` 直接跑 `train.py` 同样崩。
  第 1 轮本身（hparams.json 由 `--extra-config` 写入 `policy_start`）不受影响，但 commit 说明中“缺 key = 第 0 轮行为”这一意图目前不成立。
- 修法（任选其一，推荐第二种，溯源更完整）：
  1. `train.py:261` 改为 `start = str(a.get("policy_start", "seeds"))`；
  2. `DEFAULTS` 加回 `"policy_start": "seeds"`（第 0 轮行为），并在 `main()` 加载 `--config` 后拒绝 `DEFAULTS` 之外的未知键（见下文“建议”），这样 `config.json`/`summary.json["hparams"]` 会显式记录每个 run 的起点，打错键名也会当场报错。
- 当前无进程在跑（`ps` 无 search.py/train.py），不存在正在被新代码打断的第 0 轮搜索。

## 已确认无问题的项目

### 1. 调用数记账与 round-1 允许集（`train.py:263-271`）
- `coarse` 下 `Acquirer.run_video(initial=None)`（`acquire.py:116-119`）从 4 粗块起跑，`unobserved = 全部 30 窗`，策略每步 `unobserved.discard(w)`、`picks.append(w)`，**pick 互不重复**，可包含 seed 窗。
- `picks = len(set(picks) - set(seed_w))` 排除 seed 窗；`train_total_per_video = 4 + len(seed_w) + mean(picks)` = 8 + seed 集外新 pick。与 README/docstring 一致。
- 仿真（三个假视频：pick 中含 2/0/4 个 seed 窗）：新 pick = [6, 8, 4]，`train_total_per_video` = 14.0，`|allowed|` = {10, 12, 8}；`seeds` 分支同一组数据给出相同记账（第 0 轮下 Acquirer 不会 pick `initial` 内的窗，故两分支新 pick 数一致）。
- round-1 允许集 = `seed_w ∪ picks`（`allowed[v] |= set(picks)`，初值 `set(seed_w)`），仿真验证 seed ⊆ allowed，`policy_order` 每个前缀 `order[:m]` 揭示的窗 ⊆ allowed（前缀分支 `train.py:244` 还再过滤一次 `w in allowed[vid]`，两分支下均无元素被过滤）。
- 副作用（预期内、需知道）：`coarse` 下 pick 命中 seed 窗时，round-1 实际观测窗数 < 8 + b_max（第 0 轮恒 = 8 + b_max）；`run.log` 的 “policy on train: %.1f new picks per video” 与 `calls["train_policy_picks_mean"]` 会记录该数字，可核对。

### 2. 训练/测试轨迹起点一致
- 测试：`train.py:297` `acq.run_split(test_ids, policy, n_steps, seed=seed, log=say)`，`initial` 缺省 None → 只有粗块。
- `coarse` 下 `policy_order[v] = list(picks)`（原始顺序），前缀 `order[:m]` 揭示集 = {粗块} ∪ picks[:m]，与测试第 m 步状态同构；`seeds` 分支的代码与第 0 轮逐行相同（`seed_w + [w not in seed_w]`）。
- 与第 0 轮的唯一差异只在 `r < rounds - 1` 分支内；`DEFAULTS` 净变化为零（16149a1 加入 `policy_start: coarse`，e3436ce 移除）；checkpoint 选择用的 `val_masks`（bit-reversal 前 b_max）、`fixed34`/`coarse4`/eoc 网格、stop rule 均未动。

### 3. 无泄漏
- 训练期策略只对 `train_ids` 跑（`train.py:263`）；HMM 只用 `bin_train` 拟合；`TrainDataset.__getitem__` 只通过 `mask_sampler` 输出重建 scaffold（`src/hier_evidence_common.py:178-182`），`ScaffoldCache.build` 只用传入的 masked 向量（`:143-150`）；验证/测试路径未改。

### 4. 配置链路
- `a = Args(cfg)`（`train.py:191`），`cfg = DEFAULTS` ∪ `--config` JSON（`:394-397`）。
- `search.py:94-100`：`cfg = dict(extra); cfg.update(sample(trial))` 写入 `trial<n>/hparams.json`，再 `--config cfg_path` 传给 `train.py`（`:102-105`，每 trial 一个子进程，用磁盘上当前 train.py）。
- `scripts/run_locked_ablations.sh:13-17`：`study = runs/<experiment>/<corpus>/seed<seed>`，`config = $study/trial<best>/hparams.json`，experiment 传 `20260908_adaptive_vlm_query_it1` 时读的正是第 1 轮 trial 的 hparams（含 `policy_start`）；trainer 解析：`experiments/..._it1/train.py` 不存在 → sed 去掉 `_it1` → `experiments/20260908_adaptive_vlm_query/train.py`。实测 sed：`_it1`、`_it1_x`、`_v2` 均正确剥离，无后缀名不变；id 正则 `^[0-9]{8}_[a-z0-9_]+$` 接受 `_it1`。
- 第 1 轮新根目录 = 新 sqlite study、新 `budget.json`（按 trial 0 时长重新定 20/5），不会续跑第 0 轮 study。

### 5. bash
- bash 5.2：`set -u` 下空数组 `"${extra[@]}"` 展开为零个参数（实测 `argc` 不变）；有 `EXTRA_CONFIG` 时展开为 `[--extra-config]['{"policy_start":"coarse"}']`，JSON 引号完整。`${3:-}` 缺省为空后缀。`bash -n` 通过。
- 小注：后缀不做校验，传 `it1`（无下划线）会得到 `runs/20260908_adaptive_vlm_queryit1`，后续 ablation 脚本剥不掉后缀而报 “no train.py”，不会误跑；按 README 写法 `_it1` 即可。

## 建议（非必修，不改变结果，但关系到 config 记录是否可信）

- `search.py:91` 的 `assert not set(extra) & set(sample.__code__.co_names)` **不起作用**：`co_names` 只有 `('suggest_float', 'suggest_categorical')`，搜索键名 `"lr"` 等是字符串常量（在 `co_consts`）。实测 `{"lr": 1e-3}`、`{"policy_strat": "coarse"}`（错拼）都通过断言；`{"lr": ...}` 随后被 `cfg.update(sample(trial))` 静默覆盖，`hparams.json` 里也看不出冲突。
  正确写法：在 `objective` 内采样后 `assert not set(extra) & set(trial.params), extra`（`trial.params` 就是本 trial 实际采样的键名），或维护一个显式 `SEARCH_KEYS` 常量；再加 `assert set(extra) <= set(train.DEFAULTS)`（配合必修 1 的第二种修法）拦错拼键。
- `run_search.sh` 第 2 行旧 usage 注释未更新（第 11 行已有新 usage），可顺手删。

## 判定
- 必修 1 修复后 PASS。修复只涉及 `train.py` 一处取值方式（或 `DEFAULTS` 一行），不影响第 1 轮已写入 `policy_start` 的 config 的行为；按规则 6 只确认修复，不重开 review。
