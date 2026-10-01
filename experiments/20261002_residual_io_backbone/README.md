# Backbone 第二次修订：保留跨模态编码的区间内外残差

截至2026-10-02 11:47 NZDT；状态：[一次独立proposal review：GO](../../docs/reviews/20261002_residual_io_backbone_proposal.md)，[一次独立code review：PASS](../../docs/reviews/20261002_residual_io_backbone_code.md)。HCS seed234完整20-trial搜索已结束并回传核验，HateMM继续完整预算；HCS本版本full/nooutside/noresidual三组各三seed均已完整核验回传，固定8次未支持outside或整体残差的核心贡献。主仓库代码与本机结果为依据；Pursuing Goal保持paused，已有heartbeat与run monitor接续。用户授权优先novel且涨点，三个主指标并列。

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
