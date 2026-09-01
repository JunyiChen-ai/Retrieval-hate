# Formal pre-run code and evaluation review

截至 2026-08-31。审查范围：`imports.py`、`protocol.py`、`oof.py`、`states.py`、`model.py`、
`train.py`、`predict.py`、`evaluate.py`、`summarize.py`、`run_pilot.sh`、`launch_pilot.sh`、
`test_model.py`，以及直接调用的 MultiHateLoc data/model与共享 evaluator。本文件是修复后 delta
re-review的最终裁定，覆盖此前的 BLOCK。

## 最终裁定

**PASS，可以启动冻结的 HateMM/HateClipSeg formal pilot。**

此前4项 blocker均已闭环：OOF teacher现在进行同步 iterative cross-fit refresh；positive bag同时构造
top carrier候选与bottom pseudo-background；两个遗漏 control已进入机制否决门；neighbor endpoint不再混入
被替换行。没有发现仍会改变正式观察或结论的实现错误。

本 PASS只授权 README与 runner登记的双语料、八 arms、seed-234 pilot。它确认这是合理的
ItS2CLR adaptation，而不是原论文逐项复现；不授权改 schedule、按 corpus选 arm、混分、calibration或扩大
claim。尚未启动 formal training、blind test prediction或 test evaluation。

## 1. Iterative cross-fit 与 held-video exclusion

### Seed与初始 OOF evidence

- 每个非anchor arm独立建立三折 `StratifiedKFold`。某 fold seed的 dataset严格等于该 fold `fit_ids`；
  `held_ids`只进入 batch-size-1、no-gradient evidence inference。
- negative centroid也逐 fold只从同一 `fit_ids`中的negative train frames计算。held video既不进入 seed
  optimizer，也不进入为它生成 evidence所用的 replacement centroid。
- 初始每个 train video只由把它列为 held的 fold model生成 `fused_score`、branch score和两种 deletion
  difference；row中的 fold assignment与 deterministic split一致，rows必须精确覆盖 official train IDs。

### 每5 epoch同步 refresh

- 三个 fold model各自只在自己的 `fit_ids`上做 representation refinement。对任一 fit video `v`，
  `batch_states`读取的是 cache中 `v`的 OOF row；该 row由把 `v`列为 held、因此没有训练过 `v`的另一
  fold model生成。
- 每轮先让所有 fold model在同一旧 cache上完成5 epochs更新，之后才统一进入 `refreshed`阶段。不存在前一
  fold先刷新、后一 fold在同轮读取新 evidence的异步顺序偏差。
- 更新后的每个 fold model只为自己的 `held_ids`重生成 ranking和deletion evidence；三折共同精确重建完整
  train cache，再开始下一轮。formal设置为 seed 40 epochs、refinement 15 epochs、每5 epochs刷新一次。
- `shuffle_key`在最初由固定 seed生成，refresh时逐 video原样保留；它不会随模型性能或validation/test改变。

这实现了 source method的 load-bearing闭环：MIL/representation更新后，instance relation被重新估计，并以
self-paced relation继续优化。它没有逐字复制原 ItS2CLR的医学图像augmentation、独立aggregator工程和
negative-only warm-up；这里用已训练40 epochs的 OOF MIL seed和两侧20%最可信起点进入 refinement。对于
声明为视频MIL adaptation而非官方复现，这一差异已披露且不移除 iterative/self-paced/SupCon核心，因此
不构成 blocker。正文不得称为 exact reproduction。

OOF的准确含义是“每个视频的直接 relation producer及其 fold optimizer都不直接拟合该视频”，不是三个
teacher统计独立；跨fold系统仍共享同一语料其他 train视频。这是 cross-fitting正常边界，不能写成独立
teacher ensemble或 pseudo-label正确性证明。

## 2. Per-arm cache绑定与两语料隔离

- formal runner只运行 HateMM与HateClipSeg，外层逐 corpus、逐 arm串行。每个非anchor arm的 OOF位于本 arm
  自己的 `run_dir/oof/pseudo.pt`，并把 `--arm`同时传给 OOF producer与 final trainer。
- cache payload保存 corpus与 arm；`train.py`对两者精确比较，同时核 modality order、official train ID顺序
  与row集合。不能把 core cache交给broadcast/control，也不能跨 corpus复用。
- anchor不读取 OOF cache，使用原 MultiHateLoc cross-modal contrast；其余七 arms各自完成同样的 seed、
  refresh与final train预算。每个 corpus均重新初始化 fold models、final model与optimizer。
- OOF和final gradient只请求本 corpus official train labels。official validation单独请求且只用于本 arm
  checkpoint selection；producer代码不调用 temporal GT或 test labels。

## 3. 双侧 self-paced states 与 controls

- `_selected_tails`按 OOF fused score stable descending排序。top side数量为
  `ceil(ceil(T/3) * pace)`，bottom side从排除top后的剩余序列末端取得不超过同样数量；中间保持abstain。
- 对长度1–39、多个 pace及全tie synthetic输入穷举，top/bottom集合始终不重叠；短序列也不会把同一秒同时
  标为carrier和background。
- negative train bags所有有效 `(time, modality)`均为background。positive bags的bottom tail在三个
  modality均为pseudo-background；top tail只提供carrier候选。
- broadcast把top tail广播为三个modality carrier，保留bottom pseudo-background，并使用与core相同的三个
  projector heads、训练预算与iterative流程，足以作为 capacity-matched per-modality ItS2CLR adaptation
  control。
- core只把top tail中 centroid与neighbor两种 deletion effect均严格为正的 modality设为carrier；同一top
  秒里unsupported modality仍abstain。projection-only使用完全同样state定义。
- branch-selector与shuffled-carrier在各自当前 OOF relation的top confidence层内，逐modality取与
  deletion-stable core mask相同的carrier数量。因此 control内部严格rate-preserving；由于各 arm独立进行
  iterative refinement，最终绝对carrier count可能随arm发生内生变化，post-run应结合每epoch
  `pseudo_state_counts`披露，不能声称跨arm逐项计数被强制相等。
- `abstain_negative`把top tail中unsupported modality改为background，再恢复stable core carrier；
  `nonpositive_background`在共同bottom tail之外，只把中间且两种 intervention下所有modality均nonpositive
  的时间额外变为background。两者与core语义分开。

## 4. Replacement、SupCon与padding

- centroid replacement使用相应 fold fit-only negative centroid。neighbor replacement的内部时刻使用前后
  邻秒均值；首尾只使用唯一真实邻居，不再包含当前行。singleton显式返回自身，因此neighbor deletion为0，
  不会靠该 intervention成为stable carrier。
- replaced modality重新经过同一 seed branch，其他 modality embed不变；DMS固定为原视频原输出weight，之后
  重算相同 fused head。MultiHateLoc branch/fuse无时序混合，synthetic检查确认一次生成所有replacement rows
  与逐时刻local replacement在logit上等价。
- selective SupCon逐 modality独立建立 carrier/background positives与异类；abstain完全排除，不存在跨
  modality positive pair。self pair被移除，只有存在同类positive的anchor参与loss。
- core等arms的 selective loss对modality encoder有gradient。额外梯度检查确认 projection-only selective
  loss对backbone gradient为0、对projector gradient非零；MIL与smoothness仍共同训练backbone，符合该control
  的冻结语义。
- dataset强制三模态逐视频长度一致；collate只padding不crop。MIL、smoothness、video score与pseudo states
  都读取valid mask；padding state为abstain。predict按真实length切片，输出完整1 fps fused score。

## 5. Validation、blind test与共享 evaluation

- 每个 arm训练60 epochs，每epoch只以 official-validation video AP复制本 arm最佳state。没有validation
  localization指标、跨arm validation比较或validation method gate。
- 恢复best state后，`predict.py`只通过 blind helper获得固定 evaluator-test cohort和零占位label；它不导入
  temporal GT。输出顺序必须与cohort完全一致。
- `evaluate.py`只调用仓库唯一 `eval_baseline_scores.py`，固定 `split=test`、branch `score_core`并启用
  full coverage；score/GT长度不等、non-finite、missing或extra都会失败。
- runner对每个 arm依次完成train、blind predict和共享 evaluation，两个 corpus全部arms完成后才调用
  summarize。没有根据已出现的 test数值改变后续命令的代码分支。

## 6. SOTA与mechanism gates

- SOTA门固定为 README登记值；core必须在HateMM和HateClipSeg各自严格超过 pooled AP、pooled ROC与
  within-video ROC三项门。
- mechanism gate要求core在两语料within都严格超过broadcast，且至少一边提升 `>=.020`；同时要求core在
  两语料都严格超过branch-selector、shuffled-carrier、abstain-negative、nonpositive-background与
  projection-only。
- 新增的最后两项已经进入 `attribution_gates`，最终 `mechanism_gate=all(attribution.values())`；任何相等也
  是失败。合成metrics测试确认：即使core通过SOTA和原有controls，只要它与这两个control持平，
  `mechanism_gate=false`、`advance=false`。
- `advance`只有双语料SOTA和完整mechanism gate同时通过才为true。结果明确标记为test上的
  iterative/developmental evidence。

## 7. 实际执行的检查

- `python -m unittest -v test_model.py`：8/8 PASS，包括neighbor endpoint测试。
- 范围内全部 Python文件 compile PASS；两个 shell入口 syntax PASS。
- 双侧tail长度1–39、多个pace与全tie穷举：disjoint/count invariants PASS。
- runner per-arm OOF path/arm绑定与同步 refine-then-refresh结构检查PASS。
- projection-only gradient与fixed-DMS local replacement检查PASS。
- train-only iterative smoke：
  `runs/20260831_owner_abstaining_its2clr/smoke_iterative_hatemm/`。HateMM official train
  744/744 rows完整，2-fold seed-1、两轮refresh和final train-1均完成；cache corpus/arm/train IDs、fold
  assignment、五组tensor shape与finite检查PASS，两个fold的seed/refine losses均finite。该 smoke未运行
  blind test或读取test GT。
- synthetic summary control-veto测试PASS。
- 未启动 formal training或test。

## 非阻断边界与 post-run要求

- formal artifacts由无resume的串行runner新写；`summarize.py`本身主要读取固定目录，而不是完整artifact
  validator。正式结束后仍需独立核每个arm config、train log、test split、cohort/length/finite与共享
  evaluator复算，不能只引用 `verdict.json`。
- carrier是指定两种replacement与fixed-DMS seed下的 deletion sensitivity，不是真实或因果 owner。
- 本轮只允许双语料最小pilot。任何失败都按冻结规则记录，不得按corpus选择control或围绕test结果调整
  pace、replacement、threshold、output branch。

最终裁定：**PASS FOR FORMAL PILOT**。
