# Backbone 第二次修订：保留跨模态编码的区间内外残差

截至2026-10-02 13:38 NZDT；状态：[一次独立proposal review：GO](../../docs/reviews/20261002_residual_io_backbone_proposal.md)，[一次独立code review：PASS](../../docs/reviews/20261002_residual_io_backbone_code.md)。两主数据seed234各20-trial搜索均已核验回传，均通过既定baseline筛选但三主指标低于r5同seed；四项2025/3407独立完整搜索继续；HateMM锁定full三seed诊断已核验回传，nooutside/noresidual均已启动。HCS三组诊断未支持8次核心贡献，结论保留。Pursuing Goal保持paused，使用已有heartbeat与独立run monitor接续。

## 1. 已观察的问题与修订假设

初版inside–outside的outside在两语料有平均贡献，但完整三seed搜索仍低于第5版。加性修订版HCS三seed已完整结束，相对第5版固定8次AP/ROC/within变化−.005122/−.020509/−.034135；0次内容先验明显变差。其两语料固定配置诊断均不支持学习注意力的8次pooled收益；最新HateMM去outside变化+.007412/−.006680/−.001191，也未满足两语料核心机制贡献要求。原始来源及逐seed审计见本机 `runs/20261001_associative_io_backbone/analysis/hateclipseg_three_seed_search.json`、`two_corpus_attention_diagnosis.json`、`two_corpus_outside_diagnosis.json`；初版单独见 `runs/20261001_inside_outside_backbone/analysis/two_corpus_three_seed_search.json` 和 `two_corpus_outside_diagnosis.json`。这些都是开发期test证据；加性修订版HateMM其余两seed仍按原预算跑满，不提前宣告其搜索结果。

两轮同时移除了第5版跨模态时间编码，并把视频头换成树根读出；结果不能将全部损失因果归于某一个部件。下一步检验：保留跨模态编码和视频头，让互补区间上下文只学习秒级及节点表示的残差，能否保住视频间区分并增加局部定位贡献。残差初始化只用于保留已有表达能力，不把它本身当novelty。

## 2. 单网络结构

1. 沿用第5版同一组模态投影 `P_v/P_a`、CrossModalLayer、共享秒级线性头、视频注意力池化和视频头。所有参数一起从头训练；不加载独立模型、不组合多个模型输出。
2. 用**跨时间attention之前**的投影行作为内容树叶子。每节点携带向量和及有效行数，inside通过子节点相加；根outside为0，子outside通过父outside加兄弟inside得到。读出为按真实行数求均值。没有新学习注意力、递归非线性压缩或新的VLM观测。
3. 每模态、每节点由共享MLP读取 `inside, outside, inside-outside, inside*outside, other_inside, other_outside`。输入分别LayerNorm，MLP hidden128、GELU、原搜索dropout，最后Linear的weight/bias初始化为0，产生残差 `D_n^m`。此处norm只作用于上下文输入，不归一化原跨模态输出。
4. 叶子表示为 `r_t^m = CMA(Px)_t^m + D_t^m`，原共享头输出秒级logit，原CMAL用这些表示。视频logit仍用未加残差的CMA输出和原attention/video head；因此视频头结构与读入方式保留，但共享参数的联合训练仍会改变其数值。
5. 内部节点保留第5版以原attention头对秒级表示做区间池化得到的 `H_n`，并对 `H_n + D_n^v + D_n^a` 使用同一节点线性头得到 `phi_n`。节点头仍从0初始化，叶子phi为0。实现可将此单线性读出写作原节点势加上无bias的残差投影，两项共享同一头、同一encoder，不是独立模型ensemble。
6. 使用原问题树边际似然和视频监督；骨干不读取VLM答案，答案仍只进入既有答案模型。没有输出平滑、校准或后处理。缓存行outside的严格排除只指步骤2；CMA表示本来就可以读全视频，不能宣称整个节点表示与区间内信息隔离，也不能声称原始媒体感受野隔离。

新意仍限于问题树上的区间内外内容条件化与答案似然联合训练；不声称发明inside–outside、均值池化、残差、LayerNorm或多尺度融合。与现有r5区间attention池化的区别是额外使用排除本区间缓存行的outside，经共享非线性条件化同时作用于秒级和节点表示。是否已有hateful-video直接先例由独立reviewer实际检索。

## 3. 开跑前固定协议与检验

- 两主数据HateMM/HateClipSeg，8档soft_both，固定8次主操作点，0次同时报告、32次附加；所有三主指标并列。沿用src/qtl共享trainer/search/统一评测，不改活动实验或研究规则。
- 每seed独立Optuna TPE，sampler seed等于训练seed；50 epoch，validation(AP+ROC)/2选checkpoint后立即test；开发期搜索目标(test AP+ROC)/2，另报validation选trial的test结果，不按within剪枝。
- 四标量空间不变：lr log[1e-4,1e-3]、lamda_cma uniform[.5,2]、dropout uniform[.1,.5]、lr_answer log[1e-4,.1]；hidden128、batch32、crop_repeat5、long_T512，其余沿用r5。首个完整trial≤1小时锁定20个trial，否则5个；不smoke、不缩短训练或预算。
- 两语料先各seed234完整搜索，按既有规则再决定2025/3407；每项独立后台monitor。新目录 `runs/20261002_residual_io_backbone/<corpus>/seed<seed>/`，不覆盖前两轮。
- 预留消融开关 `io_outside=false`：只将outside置0；`io_residual=false`：不加入所有内容树残差、恢复r5结构。后续按完整搜索选定配置、固定三seed诊断，单独验证新版本的outside效应及整体条件化效应，不借用前两轮贡献证据、不冒充Optuna确认。开关不是本轮搜索空间。
- 一名独立agent完成一次proposal review，再由一名独立agent做一次code review；只修影响结论的bug，不新增process review或晋级门。

## 4. 成本与实现边界

新增VLM调用0、缓存抽取0；新视频仍只付出原提问策略的VLM成本。保留r5的跨模态时间attention，另加线性节点数的sum/count内外传递与小MLP，约O(T h²)增量；不使用第二个独立backbone或模型副本。共用一次模态投影，避免重复特征抽取。

r5 seed234的study_summary记录的完整trial平均参考约HateMM739秒、HCS354秒，含当时并发争用。暂以相对该参考1–1.5倍估算，两语料各20 trial累计运行时间约6.1–9.1小时；这是粗略资源预算，不是独占GPU测量或墙钟承诺。新增抽取GPU时间0。以首个完整trial实测修正；没有证据声称更快或保证涨点。保持本机不训练，远端按实际资源调度。

复用 `src/qtl/model.py` 的PriorNet、`src/qtl/content.py` 的拓扑缓存；通用sum/count内外传递放新的src模块。旧加性模型仍运行，不在途中改其导入/模型/trainer；新实验不import旧实验目录。这里只声明机制和协议，GO不代表性能或novelty贡献已成立。

## 5. 实现与开跑准备

原型 `backbone.py` 继承共享PriorNet，仅新增上下文残差；通用加法传递位于新文件 `src/qtl/interval_stats.py`，旧加性模型的活动实现保持原文件。新训练/搜索薄入口调用共享trainer/search；`launch/run_search.py` 调用共享run owner，两语料shell入口在本目录launch。学习率、完整50 epoch、checkpoint选择和统一评测保持第3节协议。

远端uoa-lab3 / sc474398输入已重新全量解析：HateMM1067视频（744/109/214）、HCS393视频（251/63/79），feature shape、时间轴、答案缓存覆盖与split isolation通过；本机记录 `runs/20261002_residual_io_backbone/setup/inputs_<corpus>.log` 和对应command.json。验证复用已提交的旧只读输入检查入口，无跨实验Python import、无训练/smoke。远端环境torch2.7.1+cu128/CUDA12.8/Optuna4.9.0与本机一致。运行主机为uoa-lab3 / sc474398；08:48 GPU9855MiB已用/22235MiB空闲，原加性HateMM两项完整study继续；lab1/lab-server仅余3723/2585MiB，已有他人任务不改动。code review已PASS，已提交推送并同步三机，开跑前layout检查通过；活动加性模型/trainer未改。

独立code review核验1–65长度共4225节点的区间/补集统计（最大误差7.11e-15）、outside自身行/padding/其它视频梯度排除、相同base权重下零残差train/eval的s/g/phi严格等价、非零残差不直接改变g、节点bias仅一次、真实树似然梯度/最终后验和严格checkpoint重载。原始数值 `runs/20261002_residual_io_backbone/code_review/cpu_operator_review.json`。这些是CPU算子核验，未训练、未smoke，不证明性能。


2026-10-02 08:57:02/04 NZDT，HateMM/HCS seed234两项完整搜索已启动，owner与独立monitor分别为2155073/4142180、2155347/4142198。两个monitor均已首轮RUNNING并绑定正确主机、进程身份、输出目录与当前会话，实际输入加载完整。启动记录 `runs/20261002_residual_io_backbone/setup/launches.json`，首轮进程/输出记录 `setup/startup_check.json`；本机各run下 `monitor/` 保存状态，主日志/PID在远端同run目录。首完整trial结束后自动锁定20或5预算，不另作smoke或缩短训练；两语料均已核验实际完整训练进入epoch。新增VLM调用0，旧加性HateMM两项确认仍独立跑满，不覆盖输出。08:57整卡17375MiB已用/14714MiB空闲、利用率98%，无已知OOM。


2026-10-02 10:56接续核验：加性修订版两语料三seed搜索已全部完整结束并回传，因固定8次性能及自身outside贡献未满足主线目标，已归档；本版活动代码不受影响。首完整trial HateMM1085.05秒、HCS434.67秒，两个study均已按预写规则锁定20 trial，本机来源 `runs/20261002_residual_io_backbone/<corpus>/seed234/budget.json`。10:56分别完成6/20、16/20，独立monitor正常，未据部分trial作方法结论，继续完整预算。

## 6. HCS seed234完整搜索与本版本消融锁定

2026-10-02 11:12:45，uoa-lab3 / sc474398的HCS搜索正常结束；11:17核验主进程与同会话子进程均退出，SQLite为20 COMPLETE，无失败/剪枝。全部输出已用rsync回传本机，逐trial核对history与日志各50 epoch、实际model.pth的validation最优epoch、完整配置/四标量参数/数据库目标，以及260份统一test评测（每trial13份，79视频/18839秒，缺失与额外视频均0）。`metrics.json`为主操作点摘要，与原始统一评测一致；审计不重新计算评测指标。完成核验 `runs/20261002_residual_io_backbone/setup/hcs_completion_probe.json`，逐trial审计和比较 `runs/20261002_residual_io_backbone/analysis/hateclipseg_seed234_search.json`。

下表顺序AP / ROC / within，均为test、1fps、单seed；开发期Optuna按固定8次test(AP+ROC)/2选trial19，checkpoint由validation选epoch29。validation单独选trial16、epoch1。原始权威文件为本机 `runs/20261002_residual_io_backbone/hateclipseg/seed234/trial<编号>/metrics_test_fixed<次数>.json`。

| 选择方式 | 固定0次 | 固定8次（主操作点） | 固定32次（附加） |
|---|---|---|---|
| test目标选trial19 | .641423 / .629097 / .526557 | .673446 / .669614 / .541302 | .690887 / .679990 / .615161 |
| 仅validation选trial16 | .520974 / .498905 / .544452 | .627898 / .588489 / .551426 | .664353 / .628651 / .528733 |
| test选结果对r5同seed变化 | −.028162 / −.035291 / −.016291 | −.015021 / −.005230 / −.014829 | −.007369 / +.006743 / −.001908 |

r5来源 `runs/20260929_query_paradigm_r5/hateclipseg/seed234/trial13/metrics_test_fixed<次数>.json`。8次对初版同seed变化+.004679/+.033852/−.025885，对加性版−.009848/+.014108/+.030137，来源分别为各版同路径trial11/trial7。残差结构恢复了加性版部分0次先验与8次ROC/within，但尚未超过r5。HCS pooled通过既定筛选门；HateMM 11:17仅8/20 COMPLETE、另1 RUNNING，不引用部分最优，不据此启动2025/3407确认搜索。

**开跑前锁定本版本HCS诊断**：完整搜索trial19的全部配置直接读取其config.json，lr=.0008342041153904996、lamda_cma=1.1555306370401097、dropout=.2536934471640016、lr_answer=.004638826283259073，其余配置与搜索一致。三组full/nooutside/noresidual各seed234/2025/3407、每次完整50 epoch，validation选checkpoint、统一0/8/32等13份test评测。full seed234复用已审计trial19，其余8次训练完整运行；nooutside仅将io_outside设false，noresidual仅将io_residual设false，使用已通过code review的开关。整体残差消融恢复r5结构，但属于本版锁定配置诊断，不代替r5独立搜索结果。

固定8次报告每seed、均值/总体标准差，以及各消融相对full的逐seed配对差值/均值/标准差；outside与整体残差贡献分别判断。不得借用前两版机制证据，不用32次替换主操作点，这些诊断不是Optuna确认。之后HateMM完整搜索结束再锁定其配置，检验两语料贡献。

11:18实时资源：lab3空闲21620MiB、利用率41%，已有HateMM训练继续；lab1/lab-server仅空闲3723/2585MiB，不动他人任务。在lab3并发三组诊断，新增VLM调用0，按HCS首trial434.67秒粗估8次训练累计约58分钟，受并发影响，不承诺墙钟时间。启动入口 `launch/run_hateclipseg_diagnostics_uoa-lab3.sh <arm>` 只调用共享 `src/qtl/diagnostics.py`；本版模型、活动HateMM训练代码与评测器不变。每组独立detached owner及120秒monitor，输出 `runs/20261002_residual_io_backbone/diagnostics/hateclipseg/<arm>/`。


实际启动时间（full/nooutside/noresidual）2026-10-02T11:24:23.783086+13:00 / 2026-10-02T11:24:24.563051+13:00 / 2026-10-02T11:24:25.354324+13:00；owner/monitor分别2231223/111855、2231458/111858、2231722/111869。三组实际训练已进入epoch5/5/6，独立monitor存活、首轮RUNNING并绑定当前会话，记录 `setup/hcs_diagnostic_launches.json`、`setup/hcs_diagnostic_startup_check.json`；此时HateMM10/20 COMPLETE、另1 RUNNING，整卡已用17038MiB/空闲15052MiB、利用率98%。已有heartbeat与HateMM monitor均保留，HCS已结束搜索的monitor通知后自行退出，不重复启动。


## 7. HCS本版本三组锁定配置诊断与配对结论

full/noresidual/nooutside分别于2026-10-02 11:38:45 / 11:41:32 / 11:43:29正常结束，11:43核验三个主进程及所有同会话非僵尸进程均退出。输出已全部回传本机，三组各seed234/2025/3407、各50 epoch、共117份统一test评测核验通过（每seed13份，79视频/18839秒，无缺失/额外视频）。full seed234复用完整搜索trial19；核对每个seed日志与history的1–50 epoch、实际model.pth的validation最优epoch和锁定配置；nooutside/noresidual仅分别改变io_outside/io_residual。checkpoint epoch按seed234/2025/3407顺序：full29/9/26，nooutside50/50/26，noresidual33/24/2。

原始权威来源：`runs/20261002_residual_io_backbone/diagnostics/hateclipseg/<arm>/seed<seed>/metrics_test_fixed<次数>.json`；full seed234为同实验 `hateclipseg/seed234/trial19/metrics_test_fixed<次数>.json`。逐seed/逐评测审计 `analysis/hateclipseg_locked_<arm>.json`，完整配对统计 `runs/20261002_residual_io_backbone/analysis/hateclipseg_locked_diagnostics.json`，实际同会话进程检查 `setup/hcs_diagnostics_session_check.json`。以下均为本版本固定配置诊断，不是Optuna三seed确认，不混入前两版机制证据；顺序AP / ROC / within，标准差为总体标准差。

| 组别 | 固定0次均值±标准差 | 固定8次均值±标准差（主操作点） | 固定32次均值±标准差（附加） |
|---|---|---|---|
| full | 0.643959±0.008777 / 0.624346±0.011820 / 0.538035±0.010962 | 0.667221±0.004415 / 0.651915±0.012530 / 0.560469±0.014628 | 0.686067±0.003482 / 0.673492±0.005692 / 0.608982±0.012624 |
| nooutside | 0.638233±0.009076 / 0.601705±0.007899 / 0.547337±0.006917 | 0.670221±0.004482 / 0.649848±0.007202 / 0.577506±0.007786 | 0.703319±0.003972 / 0.679409±0.002414 / 0.632004±0.008545 |
| noresidual | 0.598495±0.067173 / 0.578344±0.073539 / 0.546027±0.011561 | 0.678753±0.011413 / 0.653350±0.016401 / 0.555692±0.018537 | 0.678344±0.013257 / 0.654935±0.012044 / 0.579458±0.039373 |

固定8次的原始值及同seed配对差值：

| 组别 | seed | AP / ROC / within | 本组−full |
|---|---|---|---|
| full | 234 | 0.673446 / 0.669614 / 0.541302 | — |
| full | 2025 | 0.664532 / 0.643823 / 0.563311 | — |
| full | 3407 | 0.663686 / 0.642309 / 0.576793 | — |
| nooutside | 234 | 0.674638 / 0.639807 / 0.588414 | +0.001193 / -0.029807 / +0.047112 |
| nooutside | 2025 | 0.664076 / 0.656346 / 0.573355 | -0.000456 / +0.012523 / +0.010044 |
| nooutside | 3407 | 0.671951 / 0.653390 / 0.570750 | +0.008265 / +0.011081 / -0.006044 |
| noresidual | 234 | 0.678894 / 0.664842 / 0.576478 | +0.005448 / -0.004772 / +0.035176 |
| noresidual | 2025 | 0.692660 / 0.665051 / 0.559134 | +0.028128 / +0.021228 / -0.004177 |
| noresidual | 3407 | 0.664706 / 0.630156 / 0.531465 | +0.001020 / -0.012153 / -0.045328 |

| 配对差值 | 固定0次均值±标准差 | 固定8次均值±标准差 | 固定32次均值±标准差 |
|---|---|---|---|
| nooutside−full | -0.005727±0.006143 / -0.022642±0.012132 / +0.009302±0.012186 | +0.003000±0.003783 / -0.002068±0.019624 / +0.017037±0.022257 | +0.017252±0.001084 / +0.005917±0.007883 / +0.023022±0.007317 |
| noresidual−full | -0.045464±0.061582 / -0.046002±0.062505 / +0.007992±0.022417 | +0.011532±0.011874 / +0.001434±0.014317 / -0.004776±0.032868 | -0.007723±0.009790 / -0.018557±0.006570 / -0.029524±0.045435 |

**结论**：固定8次去outside后AP/ROC/within变化+.003000/−.002068/+.017037；去整体残差后+.011532/+.001434/−.004776。两组均未造成HCS平均AP或ROC下降至少.01，因此本版本outside与整体残差都未满足主操作点核心贡献要求，已不能据当前诊断声称两语料贡献达标。去outside的8次ROC效应随seed变号；去整体残差的AP在三个seed均提高，within变化较不稳定（配对标准差.032868）。固定配置消融不等同于为每个消融重新做完整Optuna搜索，不能将它包装成另一版本已获确认的性能胜出。

0次去outside的ROC均值下降.022642，去整体残差的AP/ROC下降.045464/.046002；后者主要受seed3407影响，配对标准差.061582/.062505。32次去整体残差ROC/within下降.018557/.029524。这些说明效果依赖预算/seed，不能用0或32次代替预先锁定的8次主操作点，也不据此重定义novelty成立。保留全部负结果，不借用初版outside证据。

11:43 HateMM完整搜索11/20 COMPLETE、另1 RUNNING，继续原20-trial预算；完成后按两语料完整搜索结果决定补seed与本版本HateMM诊断/后续修订。HCS三组已全部结束，独立monitor成功通知后退出，不重复启动；原heartbeat与HateMM monitor继续。Pursuing Goal保持paused。


## 8. HateMM seed234完整结果与补seed、诊断安排

2026-10-02 12:44:09，uoa-lab3 / sc474398正常结束；12:44核验主进程及全部同会话非僵尸子进程退出，SQLite 20 COMPLETE。全部run已回传本机，逐trial核对50 epoch日志/history、配置/四标量HP/数据库目标、实际validation checkpoint、260份统一test评测（214视频/29269秒，无缺失/额外视频）通过。两语料seed234合计40 trial×50 epoch、520份统一test评测齐全。审计 `runs/20261002_residual_io_backbone/analysis/hatemm_seed234_search.json`，两语料 `analysis/two_corpus_seed234_search.json`；进程检查 `setup/hatemm_completion_probe.json`。

开发期按test固定8次(AP+ROC)/2选trial10，checkpoint由validation选epoch3；仅按validation选trial5、epoch42。以下顺序AP / ROC / within，原始权威路径 `runs/20261002_residual_io_backbone/hatemm/seed234/trial<编号>/metrics_test_fixed<次数>.json`。

| 选择方式 | 固定0次 | 固定8次（主操作点） | 固定32次（附加） |
|---|---|---|---|
| test目标选trial10 | 0.536213 / 0.766454 / 0.745240 | 0.674695 / 0.877526 / 0.747603 | 0.666814 / 0.868253 / 0.743581 |
| 仅validation选trial5 | 0.562868 / 0.803026 / 0.745095 | 0.584880 / 0.852691 / 0.760229 | 0.625901 / 0.871144 / 0.767302 |
| test选结果对r5同seed变化 | -0.098978 / -0.056735 / -0.035454 | -0.019755 / -0.009180 / -0.041130 | -0.016402 / -0.005893 / -0.010571 |

r5来源 `runs/20260929_query_paradigm_r5/hatemm/seed234/trial9/metrics_test_fixed<次数>.json`。8次对初版同seed变化−.004865/−.003825/−.005796，对加性版+.022948/+.003248/−.018276。HateMM相对r5在0/8/32次三个指标均下降；HCS seed234主操作点三项也低于r5。本版尚未实现novel且涨点；HCS outside/整体残差贡献不成立的既有诊断结论保持，不能因过旧baseline门而声称novelty成立或SOTA完成。

**开跑前固定决定**：两主数据seed234最优trial的四个pooled数均超过规则8固定baseline门，按预先协议补seed2025/3407，各语料各seed独立20-trial Optuna TPE、每trial50 epoch，搜索空间/目标/validation选checkpoint/统一评测不变；同时报告validation选trial结果，不按within剪枝或选择性少跑。四项确认搜索独立于固定配置诊断，后者不能替代任何seed完整搜索。

HateMM自身结构诊断锁定trial10完整config.json：lr=.0001023377063766613、lamda_cma=.5918317325823095、dropout=.2764788710115099、lr_answer=.07518483859521102；各full/nooutside/noresidual的seed234/2025/3407都完整50 epoch，full seed234复用已审计trial10，其余共8次完整训练。两消融仅分别改变io_outside/io_residual，使用已经code review通过的开关，运行共享diagnostic owner；不修改模型/trainer/评测器。锁定记录 `setup/hatemm_diagnostic_lock.json`。与本版本HCS对应组分别计算0/8/32次及逐seed配对差值，不借用前两版证据；HateMM证据不能修补HCS主操作点已失败的两语料共同贡献要求。

12:45实时资源：lab3空闲27025MiB，lab1空闲3723MiB，lab-server空闲2585MiB，各已有他人服务不动。lab1环境实测torch2.7.1+cu128/CUDA12.8/Optuna4.9.0，HCS全部393视频（251/63/79）输入重新解析、shape/时间轴/缓存覆盖/split isolation通过，记录 `setup/inputs_hateclipseg_lab1.log` 与command.json。拟将HCS seed3407放lab1，HCS seed2025及HateMM两seed放lab3；根据正式训练实测显存启动lab3的已锁定HateMM诊断，资源不足的组保留待调度。lab-server余量不足以留出HCS训练所需空间，不干扰其服务。本机不训练。

本次无新增VLM调用/抽取。已完成seed234完整trial平均耗时：HateMM 681.23秒、HCS 406.95秒，受先前并发影响；据此四项补seed共80 trial累计约12.09小时，HateMM诊断8次累计约1.51小时，仅为粗略累计运行成本，不是墙钟承诺。每个长任务独立detached owner及120秒monitor，先验证首次RUNNING与实际epoch，不做smoke或缩短预算。

实际启动与监控如下，所有输出均为新的独立目录，未覆盖旧seed234。

| 任务 | 主机 | owner / 本机monitor | 启动时间NZDT |
|---|---|---|---|
| hateclipseg seed3407 完整搜索 | uoa-lab1 / sc474397 | 3369993 / 209987 | 2026-10-02T12:51:45.308402+13:00 |
| hatemm seed2025 完整搜索 | uoa-lab3 / sc474398 | 2268717 / 209995 | 2026-10-02T12:51:46.113947+13:00 |
| hatemm seed3407 完整搜索 | uoa-lab3 / sc474398 | 2268962 / 210038 | 2026-10-02T12:51:46.870513+13:00 |
| hateclipseg seed2025 完整搜索 | uoa-lab3 / sc474398 | 2269210 / 210049 | 2026-10-02T12:51:47.701933+13:00 |
| HateMM full 固定配置诊断 | uoa-lab3 / sc474398 | 2270111 / 212012 | 2026-10-02T12:53:40.241449+13:00 |

四项搜索均已实际进入epoch（首次核验HCS3407/HM2025/HM3407/HCS2025为9/1/1/4），HateMM full已进入seed2025 epoch2；五个独立monitor均存活、首次RUNNING、主机/身份/输出/当前会话绑定正确。记录 `setup/confirmation_launches.json`、`confirmation_startup_check.json`、`hatemm_diagnostic_launches.json`、`hatemm_diagnostic_startup_check.json`。每项结束后由原独立monitor接续，不做模型轮询。

正式训练实测：lab1 HCS3407使用1710MiB，整卡余2008MiB；lab3四项活动任务（两HM搜索、一HCS搜索、HM full）整卡已用25464MiB、余6626MiB、利用率99%，每项HM训练约5984–6072MiB。再放一项HM会使显存余量不足，因此nooutside/noresidual两组已锁定但暂不启动，也不为它们创建空转monitor；full完成通知明确接续检查并调度两组，各次真正启动时立即配独立monitor。待调度记录 `setup/hatemm_diagnostic_schedule.json`，两语料补seed搜索继续完整预算。此前HateMM seed234 monitor已成功通知退出；heartbeat保留，Goal保持paused。


## 9. HateMM锁定full三seed诊断完成与两组消融启动

2026-10-02 13:29:27，uoa-lab3 / sc474398的full诊断正常结束；13:30核验真实主进程及所有同会话非僵尸子进程均退出，monitor已成功通知后退出。全部输出已回传本机。seed234复用本版本完整搜索trial10，seed2025/3407各自使用其锁定配置完整训练；三seed日志/history各1–50 epoch、config完全一致、实际model.pth的validation最优epoch3/1/2、39份统一test评测均核验通过（每seed13份，214视频/29269秒，缺失/额外视频均0）。这些是固定配置诊断，不是Optuna确认；checkpoint较早不意味着缩短了训练。

原始权威路径：`runs/20261002_residual_io_backbone/hatemm/seed234/trial10/metrics_test_fixed<次数>.json` 与同实验 `diagnostics/hatemm/full/seed<2025或3407>/metrics_test_fixed<次数>.json`。逐项审计 `runs/20261002_residual_io_backbone/analysis/hatemm_locked_full.json`，三组汇总入口 `analysis/hatemm_locked_diagnostics.json` 当前只含full，明确把两种配对差值留待完整消融核验。以下AP / ROC / within，标准差为总体标准差。

| seed | 固定0次 | 固定8次（主操作点） | 固定32次（附加） |
|---|---|---|---|
| 234 | 0.536213 / 0.766454 / 0.745240 | 0.674695 / 0.877526 / 0.747603 | 0.666814 / 0.868253 / 0.743581 |
| 2025 | 0.510983 / 0.760913 / 0.701509 | 0.637817 / 0.869113 / 0.751020 | 0.664682 / 0.872653 / 0.743411 |
| 3407 | 0.535675 / 0.765891 / 0.739555 | 0.645661 / 0.865040 / 0.752635 | 0.650808 / 0.854667 / 0.724646 |
| 均值±标准差 | 0.527624±0.011769 / 0.764420±0.002490 / 0.728768±0.019414 | 0.652725±0.015862 / 0.870560±0.005199 / 0.750419±0.002098 | 0.660768±0.007096 / 0.865191±0.007655 / 0.737213±0.008886 |

full当前只构成本版本HateMM消融的配对参照，不能独自证明outside或整体残差贡献；HCS主操作点贡献失败的结论保持，不借用前两版证据，不用0/32次替换固定8次主结论。

释放full位置后，按同一trial10锁定配置启动nooutside（仅io_outside=false），实训实测占5376MiB、lab3余7316MiB，比full约6070MiB少。noresidual会直接绕过内容树/上下文残差分支，故按该较低占用与现有余量继续启动完整三seed诊断（仅io_residual=false），每项均立即绑定独立monitor，未缩短epoch/数据、未smoke。

| 诊断 | owner / monitor | 启动时间NZDT |
|---|---|---|
| nooutside | 2291136 / 254919 | 2026-10-02T13:32:57.050911+13:00 |
| noresidual | 2292985 / 257895 | 2026-10-02T13:35:24.345782+13:00 |

两组入口均为 `launch/run_hatemm_diagnostics_uoa-lab3.sh <arm>`，模型/共享trainer/评测器与配置锁定不变；配置 `setup/hatemm_diagnostic_lock.json`，启动 `setup/hatemm_diagnostic_launches.json`，后续状态以 `setup/hatemm_ablations_startup_check.json` 为准。每组最终都必须三seed各50 epoch并回传核验后再计算配对差值。

13:30实际确认搜索进度：HateMM2025/3407各2/20 COMPLETE，HCS2025为5/20，HCS3407在lab1为14/20，另各1 RUNNING。四个首trial实测均已将预算锁定20：1061.93/1059.48/421.61/170.39秒，budget.json已回传，汇总 `setup/confirmation_budgets.json`；不取部分搜索最优作方法结论。lab1余2018MiB、lab-server余2585MiB，不够运行HateMM；本机不训练，已有他人任务不动。Goal保持paused，heartbeat保留。

两组启动后核验通过：13:36 nooutside/noresidual分别进入seed234 epoch10/3，配置仅各自开关变化，所有主/相关子进程与独立monitor身份绑定正确、首次RUNNING。实际GPU占用分别5376/5052MiB，整卡29829MiB已用、2261MiB空闲、利用率99%。调度表 `setup/hatemm_diagnostic_schedule.json` 已更新为full完成、两组运行、无待启动组；四项独立搜索继续，六个活动run monitor与原heartbeat保留，Goal保持paused。


## 10. HCS seed3407完整搜索核验

2026-10-02 13:47:00，uoa-lab1 / sc474397搜索正常结束；13:48核验主进程及全部同会话非僵尸子进程均退出。输出已完整回传本机，20/20 COMPLETE、每trial日志/history均为1–50 epoch、实际model.pth均对应validation最优checkpoint，配置/SQLite参数和目标一致；260份统一test评测齐全，每份79视频/18839秒且缺失/额外视频为0。首trial170.39秒锁定20个trial，平均trial165.67秒；这是完整独立Optuna搜索，区别于第7节锁定配置诊断。

按开发期test(AP+ROC)/2选trial19，validation checkpoint epoch27；只按validation选trial9，checkpoint epoch20。以下AP / ROC / within，均为seed3407单seed，32次仅附加。

| 选择方式 | 固定0次 | 固定8次（主操作点） | 固定32次（附加） |
|---|---|---|---|
| test目标选trial19 | 0.650968 / 0.636878 / 0.553771 | 0.679294 / 0.675932 / 0.570637 | 0.701952 / 0.672799 / 0.627043 |
| 仅validation选trial9 | 0.657328 / 0.620687 / 0.557645 | 0.654459 / 0.623480 / 0.567141 | 0.682500 / 0.664815 / 0.596521 |
| test选结果对r5同seed变化 | -0.000811 / +0.006853 / +0.007842 | -0.002030 / +0.009360 / -0.007856 | +0.005521 / -0.002082 / +0.014227 |
| test选结果对初版同seed变化 | +0.060228 / +0.078069 / +0.020357 | +0.007084 / +0.033672 / +0.012721 | +0.042739 / +0.053072 / +0.055713 |
| test选结果对加性版同seed变化 | +0.185773 / +0.206178 / +0.020891 | -0.001596 / +0.033823 / +0.060657 | +0.058666 / +0.045136 / +0.114227 |

相对r5同seed，主操作点ROC提高.009360，AP下降.002030、within下降.007856；不能宣布三主指标涨点。HCS seed2025及HateMM两个确认seed尚在运行，三seed均值/标准差留待各自完整搜索全部结束后计算，不混入锁定配置诊断数字。HCS自身outside和整体残差未通过8次平均贡献要求的结论不变。

原始权威来源 `runs/20261002_residual_io_backbone/hateclipseg/seed3407/trial{19,9}/metrics_test_fixed{0,8,32}.json`；全部逐trial审计及三版本同seed比较 `runs/20261002_residual_io_backbone/analysis/hateclipseg_seed3407_search.json`，真实进程/其余任务进度 `setup/hcs_seed3407_completion_probe.json`。HCS seed3407 monitor于13:48:04成功通知后退出；13:54核验其余五个独立monitor存活、最近检查RUNNING、身份/主机/会话绑定正确，原heartbeat存活，记录 `setup/hcs_seed3407_monitor_health.json`。HateMM nooutside/noresidual均已启动，无待启动组；不重复旧任务。lab1现余3723MiB，lab3余865MiB且利用率99%，lab-server余2585MiB；当前已就绪任务均在运行，不为填充显存增加无必要实验或干扰他人任务。模型、训练器和统一评测器未改；Goal保持paused，待完成事件接续。
