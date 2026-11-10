**结论:PASS。没有发现必须修复的 bug,可以按 README 第 10 节开跑 `search_first_level.py` 的完整搜索。** 有 8 条不阻断的备注(第 7 节),其中备注 1(DeHate 必须显式传两个 seed,不能用启动脚本默认的三个 seed)和备注 2(`depths["3"]` 不等于第 5、9 节的"只问一层"策略)需要主 agent 在开跑前和读结果时注意。

# 第 7 版 + validation 选第一层(first level)规则 6 代码审查

- 审查日期:2026-11-10。审查人:独立 agent,一次。
- 审查的代码版本:分支 `r7-first-level`,最新提交 "README: stopping rule against the single level, goal (b) on HateMM (section 9), proposal for the validated first level (section 10); search entry renamed"(2026-11-10 22:30 +1300),工作区除未跟踪的 `tandem.html` 外无修改。
- 对比基线:`main`,最新提交 "all-trials stage A: validation-split replays and the question policy chosen on validation"(2026-11-10)。`main` 在分支点之后没有新提交。分支多 4 个提交,代码改动 7 个文件:`src/qtl/train.py`、`src/qtl/cpolicy.py`、`experiments/20261110_prior_weight/` 下的 `search_first_level.py`(新增)、`launch/sbatch_search_first_level.sh`(新增)、`stage_a_all_trials.py`、`policy_select.py`(只改一行注释)、`launch/run_all_trials_cpu.sh`。`src/qtl/search.py`、`qtree.py`、`policy.py`、`ctree.py`、`prior_weight.py`、`checkpoint.py`、评测器均未改。
- 范围:只查会改变实验观察或结论的 bug(规则 6)。没有跑训练、缩短训练或 smoke(规则 7)。本机不能执行 `hier_evidence_common.load_fixed_cohort`,依赖数据的代码靠阅读;纯函数和选择逻辑用 CPU 上的合成输入核对(第 8 节)。耗时估计读了已有 run 的文件时间(第 5 节)。没有计算或记录哈希。

## 1. `first_level: 0`(默认)时结果是否与 `main` 相同

相同。

- `train.py:88` 新键默认 0。`train.py:304-311`:断言多一个 `first_level in (0, "validated")`,`first_level == "validated"` 的分支和 `or cfg["first_level"]` 在 0 时都不进入。
- 每 epoch 循环(`train.py:363-481`)没有任何改动:validation 仍是 `ev.run(...)`(`train.py:467-470`),不传 `first_depth`。
- `Evaluator.run`(`train.py:151-183`):`first_depth is None` 时 `first = None`(`train.py:173`),`cpolicy.run_batch(..., first=None)`。
- `cpolicy.run_batch`:`first is None` 时 `first_mask = None`(`cpolicy.py:313`),`cpolicy.py:386` 的条件为假,候选集合不变。其余代码未改。
- `evaluate`(`train.py:551-647`):`validated` 为 False 时 `traj` 只有键 0,每个预算把 `(val_runs, test_runs)` 写回 `traj[0]`(`train.py:586-587`,plan 路径每个预算各跑一次,也走这里),`cand = [0]`,`dB = 0`,`rt = test_runs`,`res["fixed"][B]` 的三个字段与 `main` 的算式相同;不写 `first_level` / `depths` / `adaptive_first_level`。用不含 `first_level` 键的旧 config 调 `evaluate` 也可以(`cfg.get("first_level", 0)`,`train.py:565`)。
- 合成核对(第 8 节 A、C):分支的 `run_batch`(不传 `first` 和 `first=None`)与 `main` 的 `run_batch` 在贪心、`record_voi`、`prior_weight`、`plan="disjoint"`、权重加 voi 五种设置下 `asked`、`eig`、`voi`、`p_G`、`n_cand`、`scores` 逐元素相同;分支的 `evaluate` 与 `main` 的 `evaluate` 在 `first_level` 为 0 / 键缺失、贪心 / plan 四种 config 下返回值、评测器输出文件名、策略运行次数全部相同。
- 所以正在跑的第 8 节搜索即使中途换成本分支代码,数字也不变(只有 `run.log` 首行的代码版本说明和 `config.json` 里多一个 `"first_level": 0`)。

## 2. `run_batch(first=)`

正确。

- 节点编号:`qtree.depth_nodes`(`qtree.py:71-78`)返回 `qtree.tree(T)` 的节点 id;`asker.q = np.where(tr["queryable"])[0]`(`cpolicy.py:22`、`policy.py:48`)是同一棵树的节点 id。`first_mask[b] = np.isin(asker.q, first[b])`(`cpolicy.py:313-314`)是 `asker.q` 位置上的布尔数组,与 `rem[b]`、`cand`(`np.where(rem[b])[0]`,`cpolicy.py:381`)同一坐标。空集合时显式 `dtype=np.int64`,得到全 False。
- 视频对齐:`Evaluator.run` 先按 T 排序再切块,`first` 列表和传给 `run_batch` 的视频都取自同一个 `ids[i:i + k]`(`train.py:170-181`);`stage_a_all_trials.py:75-83` 同样。
- 候选限制:`cpolicy.py:386-387`,只要剩余候选里还有第一层节点,候选就只剩这些节点,按 `order` 的目标(EIG)取最大;第一层节点全部问完后条件为假,候选恢复为全部剩余节点。解析失败的回答同样把节点标为已问(`cpolicy.py:471`),不会卡在第一层。
- `rem`:第一层阶段只是 `rem` 的子集,不改 `rem` 的更新方式。`allowed`:`rem` 先按 `allowed` 限制,不在 `allowed` 里的第一层节点不会成为候选;方法里两者不同时出现(`train.py:157` 断言;回放工具的策略表里 `level` 和 `first<d>` 是不同条目)。`budgets`:只决定何时停;预算小于第一层节点数时只问到其中 EIG 最大的几个,`evaluate` 里不会出现(候选 d 满足 2^d <= B),回放里 `first_depth = min(d, floor(log2 budget))`(`stage_a_all_trials.py:69`)。`plan`:第一层过滤在 plan 的 `nest` 过滤之前,代码上可以组合,但方法里被断言禁止(`train.py:158`、`306`)。`cover`:诊断参数,方法里为 None;`cover` 代码块里有一个同名局部变量 `first`(`cpolicy.py:332`),它在 `first_mask` 建好(`cpolicy.py:313`)之后才赋值,不影响结果。
- 太短的视频:深度 d 的节点不全可问时退到更浅的、节点全可问的深度(`qtree.py:76-77`);T 属于 2..7 / 3..15 / 5..31 / 9..63 / 17..127 时 d = 1 / 2 / 3 / 4 / 5 会退。问完这层后余下的调用继续贪心。深度 d 在树里不存在时(T <= 2^(d-1))`depth_nodes` 返回空数组(空选择的 `.all()` 为真),这个视频就是纯贪心;这类视频的可问节点数不超过 2^d(T = 1..3000 逐个核对),而 d 只在 2^d <= B 的预算下被使用,所以它的全部可问节点都会被问到,后验与顺序无关,任何报告的数字都不受影响。T < 4 的视频没有可问节点,`cand` 为空直接跳过。
- 合成核对(第 8 节 B):18 个长度 3 到 257 的合成视频,d = 1..5:前 k 个问题正好是第一层节点,顺序和分数与"只允许这一层"(`allowed=`)的运行逐元素相同;之后每一步的候选数等于全部剩余节点数;把所有可问节点问完后的后验与贪心相同(差 < 1e-9)。
- 已有的阶段 A 结果与此一致(`runs/20261110_prior_weight/all_trials/`):HCS(全部 test 视频 >= 181 秒)上 `r6|loc|first3` 与 `r6|loc|level` 在 60 个 trial 中有 56 个三项指标相差不超过 2e-6,另外 4 个最多相差 .0006 / .0001 / .0014;这些差别来自两次回放各自拟合 c 和做前向(一次 GPU、一次 CPU;4 个 trial 中 2 个的 c 为 2.31 对 2.33、2.26 对 2.20),不是问的节点不同。HateMM、DeHate 上 `first3` 平均 7.63 / 7.26 次调用,`level` 7.13 / 6.56 次,差别只来自短于 32 秒的视频。

## 3. `evaluate`

正确。

- 轨迹:d = 0 是原来的两次贪心运行(带 `record_voi`,`train.py:555-559`);d = 1..floor(log2 max_calls)(= 5)各跑一次 validation 和一次 test,同一个 `prior_weight`(选中 epoch 的 c),`traj[d] = (val, test)`(`train.py:567-571`)。
- 选 d 只用 validation 标注:`vals[d] = hc.frame_metrics(at_budget(traj[d][0], B), gt["val"], hate_val)`(`train.py:589`),`dB = max(cand, key=AP + ROC)`(`train.py:590`);`cand` 升序,Python 的 `max` 在并列时返回第一个,即较小的 d。test 标注只经评测器子进程进入 `test_eval`(`train.py:573-578`),其返回值只被存储,不参与选择。合成核对(第 8 节 C)里 test 的 GT 用一个一读就报错的对象代替,`evaluate` 正常完成;在人为制造并列的设置下选中的都是较小的 d。
- `at_budget` 用在正确的轨迹上:validation 准则用 `traj[d][0]`,test 数字用 `rt = traj[dB][1]`(`train.py:591`、`595`),其它候选用 `traj[d][1]`(`train.py:602`)。候选集合 `d == 0 or 2 ** d <= B`(`train.py:588`):B = 0、1 只有 d = 0,B = 2 / 4 / 8 / 16 / 32 分别到 d = 1 / 2 / 3 / 4 / 5,与 README 第 10 节一致。
- `primary_runs`(`train.py:592-593`)是主预算下选中深度的 test 轨迹;`res["test"]`、`res["test_mean_calls"]`、`metrics.json`(`train.py:642-646`)取 `res["fixed"][pb]`,即选中深度在主预算的评测器输出。`summary["first_level"]`(`train.py:501-503`)= 每个预算选中的 d。`src/qtl/search.py:91-100` 读 `summary["test"]` 和 `summary["fixed"][pb]["val"]`(后者是选中深度的 validation 指标),目标 = (test AP + test ROC) / 2,符合 README 第 10 节的声明。
- `depths`(`train.py:597-604`):选中深度复用 `fixed<B>` 的结果,其它候选各调一次评测器,文件名 `fixed<B>_depth<d>`;`depths["0"]` 在每个预算都有,数值与 `first_level: 0` 时的 `fixed` 相同(合成核对 C)。输出文件名互不覆盖:28 个名字(7 个 `fixed<B>`、15 个 `fixed<B>_depth<d>`、6 个 `adaptive*`)无重复。
- 阈值规则读贪心轨迹:`validated` 时 `planned` 为 False,`val_runs` / `test_runs` 在预算循环里不被改写,`train.py:614-628` 用的仍是 d = 0 的两条轨迹;`test_runs = primary_runs` 的改写在这之后(`train.py:630`)。合成核对:`adaptive`、`adaptive_voi` 与 `first_level: 0` 的结果相同。
- plan 路径不变:`validated and planned` 被断言排除(`train.py:566`、`306`);plan 时每个预算重跑并写回 `traj[0]`,与 `main` 相同(第 1 节)。
- 训练与 checkpoint 选择没有改动,选中 epoch 的 c 照旧进入全部轨迹。

## 4. `search_first_level.py` 与启动脚本

正确。

- `search_first_level.py:12-13`:第 8 节 `search.py` 的五个固定设置加 `first_level: "validated"`,只差这一个键;全部在 `train.DEFAULTS` 内,`src/qtl/search.py:67-68` 和 `train.py:663-664` 的未知键断言都能通过。合并默认值后 `val_budget = primary_budget = 8`、`order = "eig"`、`fusion = "tree"`、`plan = None`、`prior = "chain"`、`answer_offset = False`、`query_level = 0`,满足 `train.py:305-311` 的断言。训练入口是同目录的 `train.py`(调 `qtl.train.main()`)。
- `launch/sbatch_search_first_level.sh` 与第 8 节的 `sbatch_search.sh` 逐行比较只差三处:注释、输出目录 `runs/20261110_prior_weight/search_first_level`(第 20 行)、入口 `experiments/20261110_prior_weight/search_first_level.py`(第 29 行)。`bash -n` 通过。
- 脚本固定 `cd "$HOME/Retrieval-hate"`(第 17 行):目标机器的主仓库必须已经是含本分支代码的版本;如果主仓库还在 `main`,入口文件不存在,三个 seed 立即写 `search.FAILED`,不会用错代码悄悄跑。日志首行的代码版本说明取自同一目录,与实际运行的代码一致。
- import 检查通过:`search_first_level.py`、`search.py`、`stage_a_all_trials.py`、`policy_select.py`、`qtl.train`、`qtl.cpolicy`。

## 5. 单 trial 耗时(规则 7 的 3600 秒线)

DeHate 两个 seed 同卡是安全的,预计 2780-3030 秒;三个 seed 同卡不安全(预计超过 3600 秒)。

实测依据(文件修改时间):

| run | 训练 | 两条贪心轨迹(validation + test,32 次,带 voi) | 13 次评测器 | 整个 trial |
|---|---|---|---|---|
| 第 6 版 DeHate,三 seed 同卡(本机 `runs/20261101_r6_length_table/dehate/seed234/trial0`) | 3221 秒(50 epoch,均值 64.0 秒) | 100.6 秒 | 19.5 秒(每次约 1.5 秒) | 3342 秒 |
| 第 8 节搜索 DeHate,两 seed 同卡(uoa-lab3 `runs/20261110_prior_weight/search/dehate/seed234`、`seed2025` 的 `trial0`,经 ssh 只读查看) | 2580 / 2566 秒 | 54 / 68 秒 | 20 秒 | 2656 秒 |

`first_level: "validated"` 多出来的部分(只在 `evaluate`,训练不变):

- 5 对轨迹(d = 1..5,validation + test,各 32 次,不带 voi;`train.py:569-571`)。每对的逐视频开销(先验网络前向、`Asker`、`Forest`)与贪心那对相同;每一步只调 1 次 `ctree.marginals`(`cpolicy.py:335`),贪心那对带 voi 时是 3 次(另加 `cpolicy.py:451-469`)。所以每对的耗时在贪心那对的约 1/3 到 1 倍之间:5 × (18..68) = 90-340 秒。
- 评测器多 15 次(每个预算的其它候选:1 + 2 + 3 + 4 + 5):约 23 秒。
- validation 指标从 7 次变成 22 次 `hc.frame_metrics`(进程内):几秒。

合计 `evaluate` 从约 75-90 秒变成约 190-460 秒,DeHate 两 seed 同卡的 trial 约 2656 + (120..370) = 2780-3030 秒,离 3600 秒至少还有约 570 秒。三 seed 同卡:第 6 版已是 3342 秒,第 7 版每 epoch 多 c 的拟合(上一份审查估计整个 trial 3430-3590 秒),`evaluate` 再多 5 × (33..100) + 约 28 = 195-530 秒,合计约 3600-4100 秒,会越线。HateMM(第 8 节实测 893 秒,三 seed,uoa-lab1)和 HCS 不受影响。

这是按代码和已有 run 的文件时间做的估计,不是实测;第 1 个 trial 的 `first_trial_seconds` 出来后以它为准。

## 6. 回放工具

没有发现会改变已报告数字的 bug。

- `stage_a_all_trials.py --src`(第 95、102-104 行):trial 列表取自给定搜索目录;`load_trial` 用该 trial 自己的 `summary["cfg"]` 建模型。权重模式 `saved`(第 150-155 行)读 checkpoint 里存的 (a, b, c),并断言表为 `r6`(该 checkpoint 自己的答案表);第 6 版 trial 没有存权重,断言直接失败,不会悄悄退成别的模式。
- `saved_post` / `saved_ask`(第 70-73 行)与 README 第 8 节声明的 2 x 2 拆分一致:`saved_post` = `run_batch(prior_weight=(a, b, c), policy_prior_scale=1 / c)`,后验用 c,选题时 S、PHI 再乘 1 / c 回到权重 1(`cpolicy.py:338-342`);`saved_ask` = `run_batch(prior_weight=None, policy_prior_scale=c)`,后验用权重 1,选题用 c。`loc` 模式 a = 1、b = 0,拆分是准确的。
- `@val`(第 137-140、182-185 行):去掉后缀后解析表 / 权重 / 策略 / 预算,回放 validation 视频,指标用 `gt["val"]` 和 validation 的仇恨视频集合;特征仓库和先验缓存都包含 validation 视频(第 111、133-134 行)。权重在两种 split 的回放里是同一次拟合(validation 回答,不用标签)。
- `first<d>`(第 55-57、69、79 行):`first_depth = min(d, floor(log2 budget))`,与 `evaluate` 的候选集合规则一致;其它策略 `first_depth = 0`,`first = None`。
- `policy_select.py`:只比较同时有 test 和 `@val` 两个回放、且有 `r6|none|eig` 的 trial(第 46 行);选择用 validation 的 AP + ROC(第 54 行),并列取候选列表里靠前的;README 第 10 节那张表的候选顺序是 `eig first1 first2 first3`(`all_trials/policy_select_b8_first.json`),与 `train.py` 的"并列取较小的 d"一致。配对差是同一 trial 上"选中策略 − 对照"的全 trial 均值(第 65-67 行);每 seed 最优 trial 按 test 的 AP + ROC 取(第 70-73 行)。test 指标只用于报告和 `test_best` 参考列,不进入选择。
- `run_all_trials_cpu.sh:12`:改为进入脚本所在 checkout 的根目录,代码、日志首行的版本说明、`stage_a_all_trials.py` 的 `ROOT`(`Path(__file__).resolve().parents[2]`)三者一致;分支 checkout 的 `runs/`、`data/` 需链接到主仓库(脚本注释已写)。

## 7. 备注(不阻断)

1. **DeHate 必须显式传 seed 列表。** `sbatch_search_first_level.sh:22` 默认三个 seed;按第 5 节的估计,三 seed 同卡的第 1 个 trial 可能超过 3600 秒,`src/qtl/search.py:113-114` 会把 5 个 trial 写进 `budget.json` 且之后不改。DeHate 应与第 8 节一样提交成 `"234 2025"` 和 `"3407"` 两个作业;第 1 个 trial 期间同一台机器上不要同时跑 CPU 回放(策略的 Python 循环占 CPU)。第 1 个 trial 结束后把 `first_trial_seconds` 写进 README。
2. **`depths["3"]` 不是第 5、9 节的"只问一层"。** 短于 32 秒的视频(HateMM test 18%,DeHate 29%,HCS 0%)第一层退到 4 个或 2 个窗口,问完后余下的调用继续贪心,而 `level` 策略到此为止(8 次预算下平均调用 7.63 对 7.13、7.26 对 6.56)。阶段 A 里两者单个 trial 最大相差 HateMM .001 / .001 / .006、DeHate .003 / .002 / .004(AP / ROC / within)。R3 和目标 (b) 写"single level (d = 3)"时应说明指的是哪一个;要报只问一层的数字,需对新 checkpoint 做回放(`--src`,`r6|saved|level`)。`stage_a_all_trials.py:18` 的"first3 at 8 calls = level"只对 T >= 32 的视频成立。
3. **`summary["test_runs"]` 和 `silent_group` 在 `validated` 下是主预算选中深度的轨迹**(`train.py:630-641`):d 不为 0 时 `voi` 为空,第一层阶段的 `eig` 是受限候选里的最大值。贪心轨迹本身不再存进 `summary.json`(它的指标在 `depths["0"]` 和 `adaptive*` 里)。以后要对这些 checkpoint 做停止规则分析,需回放,不能读 `test_runs`。`metrics.json` 不含选中的 d(在 `summary.json`);`metrics_test_fixed<B>.json` 是选中深度的评测器输出,`metrics_test_fixed<B>_depth<d>.json` 是其它候选的。
4. **`--src` 不带 `--out-root` 会把两个搜索的 checkpoint 混进同一个 trial 文件。** `stage_a_all_trials.py:124-125` 只按 variant 名判断是否已算过,不核对文件里已有的 `src`;已有的名字被跳过,新名字用另一搜索的 checkpoint 计算,`src` 字段被最后一次覆盖(第 189 行)。按文件头的写法两个参数一起给就没有问题;建议加一行断言(已有文件的 `src` 与 `a.src` 相同)。另外 `len(trials) in (5, 20)`(第 104 行)对一个只完成了 5 个 trial 的 20-trial 搜索也会通过;`run_all_trials_cpu.sh` 不传 `--src` / `--out-root`,只能回放第 6 版搜索。
5. **`policy_select.py` 的"r6"列是 variant `r6|none|eig`。** 对第 6 版 trial 它就是第 6 版;对第 7 版 trial(`--root .../r7_trials`)它是"第 7 版 checkpoint 去掉权重",不是第 6 版搜索的结果,引用时要改叫法。
6. **`qtree.depth_nodes` 的说明与行为有一处不一致**:深度 d 不存在时返回空数组而不是更浅的一层(第 2 节)。当前所有用法(2^d <= 预算)下不影响数字;若以后在预算小于 2^d 时使用 `first=`,这些视频会按纯贪心处理。
7. **"trial 0-9 超参数与第 8 节搜索相同"的前提**(README 第 10 节)与上一份审查第 7 节相同:两个搜索在前 10 个 trial 内都没有中断重启(`src/qtl/search.py:61-63` 重启时按同一 seed 重建 sampler)。读 `depths["0"]` 做第 8 节结论时,先核对两边 `study_summary.json` 的 trial 0-9 参数。第 1 个 trial 结束后还可以零成本核对(读真实输出):`fixed["8"]["depths"]["0"]["val"]` 应与 `history[best_epoch − 1]["val"]` 一致(差 < 1e-4);B = 0、1 的 `first_level` 为 0;同一预算下各深度的 `test_mean_calls` 相同。
8. **本次改动范围之外的一个观察(只记录,未审查原因)。** 阶段 A 的 180 个 trial 里有 2 个(`all_trials/hateclipseg/seed3407/trial8.json`、`trial9.json`)`loc` 拟合出的 c 恰为 1,`nll_all_before` 与 `nll_all_after` 相同(198.05,其它 trial 约 189),即拟合没有移动;GPU 和 CPU 两次回放都是如此。这两个 trial 上"带权重"与"不带权重"是同一个模型。README 第 8 节计划逐 trial 报告 c,建议把 c = 1 且似然无变化的 trial 数一并列出;若新搜索里也出现,查该 trial 上 c = 1 处的梯度。

## 8. 合成核对记录

脚本在会话临时目录,未进仓库;CPU;`main` 版本的 `cpolicy.py`、`train.py` 用 `git show main:<路径>` 导出后与分支版本在同一进程里比较。

| 核对 | 结果 |
|---|---|
| A:分支 `run_batch`(不传 `first` / `first=None`)对 `main` 的 `run_batch`;18 个合成视频(T = 3..257,节点势,5% 解析失败),32 次;贪心、`record_voi`、`prior_weight=(1, 0, .3)`、`plan="disjoint"`、权重加 voi | `asked`、`eig`、`voi`、`p_G`、`n_cand`、`scores` 逐元素相同 |
| B:`first = depth_nodes(T, d)`,d = 1..5:前 k 个问题对"只允许这一层"的运行 | 节点、顺序、分数逐元素相同 |
| B:第一层之后每一步的候选数 | 等于全部剩余可问节点数 |
| B:问完全部可问节点后的后验对贪心 | 差 < 1e-9 |
| B:`depth_nodes` 对 T = 1..3000、d = 1..5:返回的节点同一深度、全部可问(或根)、都在 `asker.q` 内;空集合只在 T <= 2^(d-1) 出现,且该视频可问节点数 <= 2^d | 成立 |
| C:分支 `evaluate` 对 `main` 的 `evaluate`(策略运行、validation 指标、评测器用合成替身),`first_level` 0 / 键缺失 × 贪心 / plan | 返回值、输出文件名、策略运行次数相同 |
| C:`first_level: "validated"`:每个预算选中的 d 对独立重算(validation 准则最大,并列取小;含人为并列的两组设置);`depths` 各候选的 validation / test 数字;`res["test"]`;`adaptive*` 对 `first_level: 0`;`depths["0"]` 对 `first_level: 0` 的 `fixed` | 全部一致;test GT 未被读取;28 个评测器输出名无重复;12 次策略运行 |

## 9. 备注 8 的修复确认

**结论:修复正确,确认。** 只看了这一处,没有重开其它审查。

- 代码版本:分支 `r7-first-level`,提交 "prior weight: the search for c starts from the best grid point (L-BFGS from c = 1 does not move when the weighted prior is saturated: HateClipSeg seed 3407 trials 8, 9, 14, 17, likelihood flat for c >= .2, maximum at c = .003); replay mode locg"(2026-11-10 22:56 +1300)。该提交只改 `src/qtl/prior_weight.py` 和 `stage_a_all_trials.py`;`train.py`、`cpolicy.py`、`checkpoint.py` 未改。
- **修复内容与是否进入训练。** `prior_weight._fit_c`(`prior_weight.py:156-168`):`start="grid"` 时先在不求梯度的情况下对 `C_GRID` 的 13 个点算同一个负对数似然 `_nll_all`,取最小的点(非有限值按无穷大处理;数值完全相同时取离 1 最近的点),把 c 设为该点后再跑原来的 L-BFGS。`fit` 的 `loc` 和 `root_all` 两个模式改为调 `_fit_c`(`prior_weight.py:203`、`208`),默认 `start="grid"`。`train.py:466` 调 `pweight.fit(..., mode=cfg["prior_weight"])` 不传 `start`,所以每个 epoch 的拟合用的是新搜索;之后 c 进入 validation 准则、checkpoint、存盘、最终评测的路径与第 3 节和上一份审查相同,没有改动。`joint`、`root`、`trust`、`trust_len` 的代码路径没有变。
- **无泄漏。** `_fit_c` 只接收 c、似然函数和 `start`;似然函数就是原来的 `_nll_all(chunks, chain_d, a_, b_, c)`,输入仍只有 validation 视频的先验网络输出、缓存回答、答案模型和 chain。`fit` 和 `_fit_c` 的参数里没有 `labels`、`gt`。`C_GRID` 是写死的常数,三个语料共用,只决定起点,最终的 c 仍由似然决定。
- **合成核对**(CPU,脚本在会话临时目录):
  - `start="one"` 与修复前的 `prior_weight.py` 在 `loc`、`root_all`、`joint`、`root`、`trust` 五个模式下返回值逐项相同(6 组合成数据);`stage_a_all_trials.py:164-166` 对 `locg` 以外的模式传 `start="one"`,已缓存的 variant 含义不变。
  - 复现了原问题:逐秒 logit 均值 5、标准差 .8、真实仇恨占 20% 的合成数据上,似然在 c >= .3 处是平的(37.754 / 28.577),从 c = 1 出发 L-BFGS 只算 1 次梯度就停在 c = 1.00000;`start="grid"` 从 c = 0 出发到 c = −.015,似然 19.670 / 17.128,与步长 .001 的细扫描最小值一致(差 < 3e-5)。
  - 普通先验(两种设置各 2 组):两种起点得到同一个 c(五位小数相同)、同一个似然;`start="grid"` 的结果不高于任何网格点的似然值;`nll_all_before` 仍是 c = 1 处的值;`c_start` 等于网格最小点;a = 1、b = 0;拟合后 chain 和答案模型上没有梯度。
- **你没有写到、但会改变结果或耗时的地方**(都不需要改代码):
  1. 每次拟合多 13 次不求梯度的似然计算。合成数据上一次这样的计算约为一次带梯度计算的 0.3 倍;按上一份审查记录的 DeHate 每次拟合 2.4-3.7 秒估算,每 epoch 多约 1.3-2.6 秒,每个 DeHate trial 多约 60-130 秒。第 5 节的估计相应变为两 seed 同卡约 2840-3160 秒(仍低于 3600 秒,余量约 440 秒以上);三 seed 同卡更不可行。这是估计,以第 1 个 trial 的实测为准。
  2. 受影响的不只是那 4 个 trial 的最终 checkpoint:拟合是每个 epoch 做一次,任何 trial 里先验饱和的 epoch 现在都会得到接近 0 的 c,该 epoch 的 validation 准则随之改变,选中的 epoch 可能不同。其余 epoch 的 c 只在 L-BFGS 的收敛精度内变化(合成数据上五位小数相同),数字与旧搜索不是逐位相同。两个搜索都从 trial 0 重跑与此一致。
  3. 饱和的 trial 上 c 会接近 0,也可能略小于 0(合成数据上 −.015;README 第 8 节已规定 c 不设界、不截断)。c 接近 0 即位置先验基本不起作用,报告 c 时这类 trial 会明显增多。若某次 c 恰为 0,回放的 `saved_post`(`stage_a_all_trials.py` 里的 `1.0 / w["c"]`)会报除零错误而不是给出错误数字。
  4. `stage_a_policies.py:121` 和 `stage_a_weights.py:94` 调 `prior_weight.fit` 时不传 `start`,重跑时会用新搜索;它们已有的输出(README 第 4、5 节三个选中 trial 的表)是旧搜索算的。只有重跑这两个脚本时才有差别。
  5. "数值相同时取离 1 最近的点"只在浮点完全相等时生效。似然平台上各点通常在末几位不同,起点可能是平台上任意一点;只有当平台本身就是最优区域时才会走到这一步,此时 c 在平台内取何值对先验没有影响。
