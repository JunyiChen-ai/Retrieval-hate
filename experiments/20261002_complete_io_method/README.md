# 最新完整组合：inside–outside + 复制似然 + 自动停止

截至2026-10-31；用户指令：**“三个数据集上跑一遍最新完整的给我汇报”**。三语料为HateMM、HateClipSeg和DeHate；前两者是主数据，DeHate保持external validation。实现接入完成，[一次独立集成code review：PASS](../../docs/reviews/20261002_complete_io_method_code.md)；已同步并通过远端完整输入核验。**三语料全部完成**：两主数据完整重评10-02结束，DeHate三seed完整搜索→完整组合评估10-03结束，10-31回传本机并审计通过（第7节）。Pursuing Goal保持paused。

## 1. 固定方法与本次范围

同一套已开发部件在三语料组合运行，不提出新的novelty候选、不按语料切换骨干：

- 已保留初版inside–outside backbone，`io_outside=true, io_merge=gated`，秒级/节点先验、原视频头与零膨胀链、原锚定答案模型和训练损失。
- 单问首token软答案，`soft_both`，训练答案无标注分位数8档；同一份I3D/VGGish/BERT与VLM缓存。
- 复制似然：负例训练视频父子答案一致率估计各长度桶复制概率，`copy_mode=both`，同时进入EIG与答案似然。估计不使用validation/test答案或定位GT。
- 自动停止锁定已有 `qmixSG_rt10`：逐秒logit变化总量、秒级排序变化和视频后验熵的validation分位数组合，乘`.9**k`；validation无标注平均8次选择阈值，不重新挑规则、松弛率或预算。没有额外最少调用次数，轨迹最大32次。
- 主结果为该完整组合的自动停止；同一复制设置的固定0/8/32次同时报告，以固定8次直接量化自动停止的代价。AP/ROC/within三项均为主指标，报告三seed均值及总体标准差、逐seed结果、平均调用数及正负例调用数。

这次是已开发方法的完整组合验证。初版的proposal/code review与novelty复查见 `docs/reviews/20261001_inside_outside_backbone_{proposal,code}.md`、`docs/reviews/20261002_inside_outside_backbone_novelty_recheck.md`；复制与停止定义来自原r5 README第17.2/17.4节。没有新的机制提案需要重新泛化proposal review；本次安排一次独立code review核对集成正确性。

## 2. 训练与模型选择（开跑前锁定）

复制似然与停止改变的是推断，不改变现有训练损失；因此不为相同训练重复消耗预算。

- HateMM/HateClipSeg复用初版各seed234/2025/3407已完整20-trial×50 epoch搜索。每trial validation选checkpoint；source `runs/20261001_inside_outside_backbone/<corpus>/seed<seed>/`。
- DeHate还没有该骨干，补三seed各自完整Optuna。与初版同一搜索空间：`lr` log[1e-4,1e-3]、`lamda_cma` uniform[.5,2]、`dropout` uniform[.1,.5]、`lr_answer` log[1e-4,.1]。50 epoch，hidden128、batch32、crop_repeat5、long_T512，其余共享r5训练配置不变。
- DeHate sampler seed等于训练seed；首个完整trial≤1h锁20 trial，>1h锁5 trial，之后不增减。源码使用既有共享search/train，无smoke、短训或固定配置代替搜索。
- 三语料训练阶段均沿用fixed8、无复制版本的validation AP/ROC均值选checkpoint；开发期test(AP+ROC)/2选trial。同时冻结仅validation均值选出的trial，单独报告其完整组合test结果。不会在看到完整组合test后换checkpoint或trial。
- **这是预先选定模型的完整组合重评，不是以自动停止为目标重新执行的Optuna搜索。** 新DeHate也使用同样的训练/选择协议；不把固定配置诊断混入三seed搜索，不把r5旧骨干的DeHate结果当作新骨干结果。
- checkpoint来源与选中epoch实际核验；模型构造使用 `qtl.inside_outside.make_model`，strict加载model/answer/chain，不允许旧PriorNet替代新骨干。

## 3. 推断、阈值与成本

每个被选checkpoint先在完整validation/test上运行已有复制模型的32次轨迹，再按既定因果停止规则取前缀；每一步只访问已选问题的缓存答案。三项统计只读取已发生的预测/回答；分位数参考和阈值仅来自validation，不用test统计定阈值。完整轨迹用于固定预算比较和审计，不代表部署时必须问32次。部署按相同停止判定消费前缀即可。

输出的调用数是策略实际消费的缓存节点答案数；本次不重新调用VLM，处理新视频仍需执行相同的查询及原有输入处理。新增VLM调用/特征抽取0；固定4帧提问观测不变。不把缓存重评CPU时间写成新视频端到端成本。

既有r5 DeHate每study约18.8小时（原三study并发、有资源争用），仅作墙钟参考，不视为新骨干独占GPU耗时。本次先实测新骨干完整trial，再按既有规则锁预算。两主数据只需每seed既定test/validation选出的至多两个checkpoint重评，无新增训练；DeHate为唯一新增训练语料。HateMM/HCS重评与DeHate训练资源允许时并行。

## 4. 实现与运行

初版backbone升入 `src/qtl/inside_outside.py`，原入口保留兼容导入；原复制概率估计、停止统计升入 `src/qtl/{copy_noise,stop_rules}.py`，旧工具复用它们。共享checkpoint loader在 `src/qtl/checkpoint.py`。所有实验入口均仅依赖src，不跨实验import；唯一统一评测器不修改。

- `launch/run_hateclipseg_uoa-lab1.sh`：现有HCS三seed搜索的完整组合重评。
- `launch/run_hatemm_uoa-lab3.sh`：现有HateMM三seed搜索的完整组合重评。
- `launch/run_dehate_uoa-lab{1,3}.sh <seed>`：该seed完整训练搜索，然后自动执行其选中checkpoint的完整组合重评。

所有长任务通过setsid/nohup独立后台启动；owner同时记录host、PID/PGID/SID/start_ticks和原始命令。owner在首次训练/推断前实际解析完整输入、校验shape/finite/coverage/split isolation/1fps长度，验证结果按主机/语料复用。日志首行记录主机；每个owner绑定独立 `scripts/monitor_run.py`，成功与失败均通知本会话。既有已关闭heartbeat不覆盖；新任务使用独立heartbeat目录，Goal不创建/恢复。

输出根为 `runs/20261002_complete_io_method/`。主数据重评在 `<corpus>/integrated/results/`，DeHate搜索在 `dehate/seed<seed>/`、重评在其 `results/`。每个重评trial保存source配置、selection、calibration（负例训练估计、validation参考分布/阈值）、完整轨迹、逐视频调用数、逐秒分数与统一评测器原始JSON。原训练源与旧结果不覆盖。

远端完成后回传本机，核验owner及同会话子进程退出、完整训练预算/50 epoch、全部选中trial、固定0/8/32及自动停止共4份统一评测、完整test覆盖。汇总明确区分开发期test选trial与validation选trial；同设置固定8次作为停止比较，旧r5完整设置作为性能参照。完成标记不代替这些核验。

## 5. 当前进度

独立集成code review已PASS；原共享函数实现等价性已核对，12个既定checkpoint严格加载和三语料split/时间网格检查通过。三机已同步，正式owner全输入finite/shape检查均通过；三语料运行全部结束（第6、7节）。10-31补全trial按设置选模型与预算曲线（第9节）。没有改变研究规则、已归档实验结论或无关tandem.html。


## 6. 两主数据完整组合结果（2026-10-02）

HCS于18:34:54、HateMM于18:36:03结束，真实owner及全部同会话进程已退出；结果完整回传本机。每语料原test-selected/validation-selected共6个checkpoint，合计48份统一评测齐全。源配置/实际checkpoint、validation-only阈值/复制估计输入、因果前缀分数、调用数以及统一评测完整覆盖均核验通过。各阈值为原qmixSG_rt10的validation平均8次校准，不对本轮test重新选规则。

AP / ROC / within，三seed均值±总体标准差；开发期原test目标选trial：

| 语料 / 设置 | AP / ROC / within | 平均调用 |
|---|---|---|
| HateMM / 0次 | 0.591739±0.007204 / 0.789032±0.008709 / 0.715851±0.021907 | 0.0000 |
| HateMM / 同复制设置固定8次 | 0.668707±0.018339 / 0.874930±0.004352 / 0.735957±0.019076 | 7.6262 |
| HateMM / 完整组合自动停止 | 0.671441±0.015171 / 0.877612±0.003897 / 0.730922±0.016773 | 7.8583 |
| HateMM / 32次（附加） | 0.684682±0.011156 / 0.881732±0.003157 / 0.749468±0.016806 | 25.0187 |
| HateClipSeg / 0次 | 0.633132±0.032202 / 0.610808±0.037310 / 0.531452±0.001537 | 0.0000 |
| HateClipSeg / 同复制设置固定8次 | 0.660655±0.010263 / 0.630674±0.022582 / 0.552625±0.007539 | 8.0000 |
| HateClipSeg / 完整组合自动停止 | 0.653658±0.010247 / 0.621388±0.018894 / 0.545536±0.004205 | 7.9831 |
| HateClipSeg / 32次（附加） | 0.675262±0.017010 / 0.638592±0.026383 / 0.594112±0.022389 | 32.0000 |

自动减同复制固定8次的配对均值：HateMM +.002733/+.002682/−.005035；HCS −.006997/−.009286/−.007089。HCS三个seed的AP/ROC都下降；HateMM固定8次受短视频节点数限制实际7.6262次，自动7.8583更多；HCS自动7.9831与固定8.0000几乎相同。当前没有稳定省调用或两语料保持性能的证据。

仅validation选trial的完整组合结果：

| 语料 | AP / ROC / within | 平均调用 |
|---|---|---|
| HateMM | 0.643795±0.018646 / 0.875095±0.001461 / 0.677516±0.055519 | 7.8723 |
| HateClipSeg | 0.643983±0.006301 / 0.616759±0.006254 / 0.547546±0.021578 | 7.6160 |

相对旧r5同copy_neg+qmixSG_rt10，主选择三项均值变化：HateMM −.023201/−.012166/−.038214；HCS −.029783/−.053544/−.011952。该比较是原已选模型的组合重评，不是当前自适应目标的新Optuna搜索，不使用这些结果重新挑trial。

本机原始路径 `runs/20261002_complete_io_method/<corpus>/integrated/results/seed<seed>_trial<k>/metrics_test_{fixed0,fixed8,fixed32,qmixSG_rt10}.json`，停止校准与逐视频调用数在同目录calibration/summary。逐seed、总体与配对标准差、旧r5原始来源在 `analysis/{hatemm,hateclipseg}_complete_method.json`；汇总 `analysis/main_corpora_complete_method.json`。所有analysis位于本轮runs根。HateMM test/validation trial为17/11/11与10/2/9，HCS为11/2/10与6/17/1（seed234/2025/3407）。

DeHate三seed分别在lab3/lab1/lab3运行完整训练→评估链；首trial实测2627/1821/2651秒，均≤1小时，各自已锁定20-trial完整预算（原始 `dehate/seed<seed>/budget.json`）。结果见第7节。

## 7. DeHate完整组合结果（2026-10-03完成，10-31回传审计）

seed2025（lab1）04:41、seed234/3407（lab3）09:17/09:18结束，`completion.json` 均为success；10-31检查两机已无本项目进程。三study各20/20 trial COMPLETE、每trial 50 epoch与13份训练阶段统一评测齐全，无失败trial。全部回传本机 `runs/20261002_complete_io_method/dehate/seed<seed>/`。

审计 `audit_dehate.py`：DeHate cohort（split/GT/特征）只在远端，故按seed在结果所在主机运行（`--seeds`），本机 `--merge` 合并。核对项：主机与完成标记；test/validation选trial由study内各trial属性重新推出，与 `selection.json` 一致；checkpoint epoch；复制概率只用训练负例、分位参考只用validation；按runs.pkl重算qmixSG_rt10阈值、validation平均调用与逐视频调用数，完全一致；四种设置写出分数等于因果前缀（6位小数）；24份统一评测覆盖1151视频/110839秒、无缺失。test-selected trial为5/12/14、validation-selected为4/11/8（seed234/2025/3407），选中epoch分别46/47/45与1/11/15。

AP / ROC / within，三seed均值±总体标准差；开发期test目标选trial：

| 设置 | AP / ROC / within | 平均调用 |
|---|---|---|
| 0次 | 0.188340±0.007451 / 0.702571±0.010153 / 0.668309±0.005206 | 0.0000 |
| 同复制设置固定8次 | 0.217502±0.007884 / 0.763608±0.005315 / 0.679874±0.008689 | 7.2641 |
| 完整组合自动停止 | 0.219813±0.010473 / 0.761690±0.005629 / 0.682276±0.009662 | 7.9334 |
| 32次（附加） | 0.221194±0.013956 / 0.767275±0.000652 / 0.708132±0.004667 | 21.9600 |

自动减同复制固定8次的逐seed配对均值±总体标准差：**+.002311±.003583 / −.001919±.000723 / +.002402±.004429**；ROC三个seed都小幅下降。固定8次因短视频节点不足实际7.2641次，自动停止7.9334次反而更多。

仅validation选trial：自动停止 0.205808±0.019643 / 0.750641±0.010609 / 0.641601±0.035813，7.9334→8.0269次；对同设置固定8次配对 +.011085 / +.002085 / +.001958。

相对旧r5同copy_neg+qmixSG_rt10（seed234/2025/3407 trial8/4/4，`runs/20260929_query_paradigm_r5/stop_check/dehate/seed<seed>_trial<k>_r5rt/metrics_test_qmixSG_rt108.json`，均值0.254707/0.774281/0.672135），主选择三项变化 **−.034894 / −.012592 / +.010141**。

完整审计与逐seed/配对/选择记录 `runs/20261002_complete_io_method/analysis/dehate_complete_method.json`，两机原始审计输出 `analysis/dehate_audit_partial_{2025,234_3407}.json`。

## 8. 三语料结论

- 自动停止在三语料都没有减少调用：HateMM 7.63→7.86、HCS 8.00→7.98、DeHate 7.26→7.93。
- 对同复制设置固定8次：HateMM与DeHate三项变化均在±.003内；HCS三项下降.007–.009。
- 对旧r5同复制/停止设置：三语料AP均下降（−.023/−.030/−.035），ROC均下降（−.012/−.054/−.013）；within HateMM/HCS下降、DeHate上升.010。
- 结论：新inside–outside骨干接入复制+停止后，没有显示省调用或相对r5涨点。本轮按用户指令只做三数据评估，不自动扩展新候选或修改停止配置。

## 9. 全 trial 按设置选模型与预算曲线（2026-10-31）

目的：回答两个问题。(a) 主表各seed的trial是用训练阶段“无复制fixed8”的validation指标选的，可能偏向fixed8；让每个设置各自选trial后结论是否变化。(b) 在相同平均调用次数下，自动停止是否比固定次数好（是否省调用）。不重新训练，复用原20 trial×3 seed的既定checkpoint。

运行：`launch/sbatch_all_trials.sh`（Slurm，lab3跑HateMM与DeHate seed234/3407，lab1跑HCS与DeHate seed2025，job 1295–1299，10-31 07:54–08:50，全部COMPLETED、无Traceback）对每个trial做完整组合评估并另存validation分数；`budget_curve.py`（CPU，同两台机器）用已存32次轨迹的前缀算fixed k∈{0,1,2,3,4,5,6,8,10,12,16,20,24,32}，并把qmixSG_rt10阈值在validation上重新校准到平均1/2/4/6/8/12/16次，test全部走统一评测器。本机汇总 `select_by_setting.py`、`budget_curve.py --merge`。代码审查记录 `docs/reviews/20261031_all_trials_budget_curve_code.md`。

复现核对：原选定trial重跑后，HateMM、HCS全部设置指标与逐视频调用完全一致；DeHate仅seed234 trial4（validation选定）有偏差，三项最大绝对差.0012，自动停止8/1151个视频调用数不同（GPU非确定性）。`select_by_setting.py` 记录偏差并要求≤.005。

### 9.1 各设置自己选trial（AP / ROC / within，三seed均值，test指标）

来源 `runs/20261002_complete_io_method/analysis/select_by_setting.json`。

| 语料 | 选trial依据 | fixed8 | 自动停止 | 自动−fixed8（各自trial） |
|---|---|---|---|---|
| HateMM | validation | .646 / .875 / .663 | .652 / .877 / .743 | +.006 / +.003 / +.080 |
| HateMM | test | .683 / .879 / .752 | .686 / .882 / .730 | +.004 / +.003 / −.022 |
| HCS | validation | .647 / .616 / .565 | .640 / .610 / .558 | −.007 / −.006 / −.008 |
| HCS | test | .668 / .647 / .562 | .669 / .642 / .565 | +.001 / −.005 / +.002 |
| DeHate | validation | .200 / .757 / .666 | .204 / .754 / .653 | +.004 / −.004 / −.013 |
| DeHate | test | .219 / .770 / .673 | .229 / .762 / .690 | +.010 / −.007 / +.017 |

HateMM validation依据的within +.080主要来自seed2025（+.161：fixed8在validation上选中的trial2 test within低）。逐seed差值波动大（如DeHate validation依据AP −.017/−.027/+.058），三seed下无稳定方向。

同一trial上自动−fixed8，全部60个trial的均值（括号内为自动更好的trial比例）：HateMM +.004（72%）/ +.004（90%）/ −.001（48%）；HCS −.004（20%）/ −.003（38%）/ −.003（35%）；DeHate +.001（53%）/ −.001（45%）/ +.001（58%）。

结论(a)：主表的选择方式没有明显压低自动停止。让自动停止自己选trial后，相对fixed8仍是HateMM小幅正、HCS小幅负、DeHate接近0，与主表一致。

### 9.2 相同平均调用下自动停止减固定次数

来源 `runs/20261002_complete_io_method/budget_curve/<corpus>/curve_summary.json`（主表的test选定trial；fixed曲线按平均调用线性插值后逐seed配对）。

| 校准目标 | HateMM | HCS | DeHate |
|---|---|---|---|
| 2次 | −.005 / +.004 / +.000 | −.001 / −.001 / +.000 | +.000 / +.001 / +.001 |
| 4次 | −.005 / −.001 / +.009 | −.004 / −.005 / +.002 | −.001 / +.001 / −.001 |
| 6次 | +.004 / −.001 / +.000 | −.003 / −.004 / +.001 | +.006 / +.001 / +.001 |
| 8次 | +.002 / +.002 / −.006 | −.007 / −.010 / −.007 | +.003 / −.002 / +.000 |
| 12次 | −.001 / −.001 / −.001 | −.003 / −.002 / −.007 | −.002 / −.005 / −.002 |
| 16次 | +.002 / +.002 / +.001 | +.002 / +.000 / +.002 | −.000 / −.004 / −.002 |

validation选定trial的同表差值也全部在±.011内。各目标下自动停止的test平均调用与目标相差<0.6次；正例视频并未多分到调用（HateMM、HCS负例视频调用更多，DeHate正例略多）。

结论(b)：在2–16次的所有预算上，自动停止与同调用数的固定次数之差都在±.010内、方向不稳定，没有省调用。

### 9.3 固定次数曲线本身（test选定trial，AP / ROC / within）

| 固定次数 | HateMM | HCS | DeHate |
|---|---|---|---|
| 0 | .592 / .789 / .716 | .633 / .611 / .531 | .188 / .703 / .668 |
| 1 | .643 / .848 / .717 | .648 / .624 / .531 | .202 / .742 / .668 |
| 2 | .663 / .866 / .728 | .654 / .633 / .540 | .204 / .746 / .668 |
| 4 | .667 / .872 / .711 | .661 / .640 / .544 | .216 / .756 / .676 |
| 8 | .669 / .875 / .736 | .661 / .631 / .553 | .218 / .764 / .680 |
| 32 | .685 / .882 / .749 | .675 / .639 / .594 | .221 / .767 / .708 |

AP/ROC的增益大部分在前4次内取得：fixed4与fixed8的AP差≤.002、ROC差≤.009（HCS fixed4的ROC反而高.009）；within随调用继续上升，fixed4比fixed8低.004–.025（HateMM .711对.736）。fixed32比fixed8 AP再高.003–.016。

### 9.4 对本轮的意义

- 自动停止规则（qmixSG_rt10）在任何预算下都没有带来超过固定次数的收益；“8次调用”问题不能靠当前停止规则解决。
- 若只看AP/ROC，固定4次已接近固定8次；但within下降，按停止规则“within不降”的约束不能直接改为4次。
- 停止统计量没有把调用分给正例视频，这是规则本身没效果的直接表现。下一方向待用户裁定。
