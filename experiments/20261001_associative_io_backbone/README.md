# Backbone 修订候选：保留加性统计量的 inside–outside 编码器

截至2026-10-02 08:47；状态：HCS三seed完整搜索和两语料full/noattention/nooutside诊断已全部核验回传；HateMM seed234完成，其余两seed仍按完整预算搜索；[一次独立proposal review：GO](../../docs/reviews/20261001_associative_io_backbone_proposal.md)，骨干与训练/搜索入口已实现，[一次独立code review：PASS](../../docs/reviews/20261001_associative_io_backbone_code.md)；两主数据seed234完整搜索已于2026-10-01 23:05 NZDT在uoa-lab3 / sc474398启动。inside–outside 初版两主数据seed234现已完整跑满20 trial；本提案提出/启动时未替换其活动代码。用户授权继续自动迭代 backbone，优先 novel 且涨点，within 与 AP / ROC 并列为主指标。Pursuing goal 保持 paused，继续由 heartbeat / 完成通知接续。

## 1. 依据与问题

读取了初版 HCS seed234 完整20-trial搜索，以及预先锁定trial11配置的 full / nooutside / mean 三seed、各50 epoch、统一test评测。它们属于已查看test的开发证据，不是独立确认结果。

完整数字、各seed与误差分析见[初版README第8节](../20261001_inside_outside_backbone/README.md#8-hateclipseg-seed234-完整结果与结构诊断2026-10-01)，原始输出为本机 `runs/20261001_inside_outside_backbone/diagnostics/hateclipseg/<arm>/seed<seed>/metrics_test_fixed<预算>.json`；full seed234复用原搜索trial11。派生汇总 `runs/20261001_inside_outside_backbone/analysis/hateclipseg_locked_diagnostics.json` 保留逐项来源。

去outside使三项均值下降，但AP差异主要来自seed234，另外两seed没有同向改善，不能写成每seed都稳定有益。门控组合没有超过均值：0次提问时，均值组合的三个指标反而都更高；8次差异低于单seed噪声口径。初版相对旧骨干的0次视频间区分也下降。因此优先检验**反复非线性压缩区间表示是否损害内容先验**，保留已有平均效应支持的内外比较；这仍是待证假设，不能把相关诊断直接写成因果结论。

## 2. 拟议计算图

同一套 I3D / VGGish+BERT 缓存，同一问题树、树答案似然、CMAL与提问策略。骨干不读取VLM答案或test标签。

1. 每个缓存行先做点式模态投影，得到视觉与音频文本向量；不提前跨时间混合。
2. 每个模态有固定4个可学习attention head。各head仅从该行向量计算正权重 `w_t`，并形成 `Z_t=w_t`、`S_t=w_t*x_t`。4个head属于一个骨干内部参数，不是多个模型，也不是新增VLM提问。
3. inside沿固定二叉时间树把子节点的 `Z/S` **直接相加**。直到读出节点表示才做 `S/Z` 归一化和head融合，不在每层反复LayerNorm / tanh / 门控压缩。这样任意叶子对祖先的加权统计量有直接路径，组合顺序不改变统计量。
4. outside沿父outside加兄弟inside的 `Z/S` 传播，根outside为空；读出时归一化，空outside输出固定零向量，不设置无梯度的根专用参数。某节点outside严格不依赖自身区间的缓存行。禁止先做含全视频行的softmax再声称outside独立；权重需点式生成，实现采用 `exp(clamp(linear(x),-10,10))`，线性打分不设会被S/Z约掉的head bias；非空区间权重下限为exp(-10)，分母只在空outside时钳至1e-12。原始I3D/BERT感受野仍可能跨行，排除只针对缓存索引。
5. 由inside/outside及其差/乘积、另一模态上下文形成节点表示，叶子输出秒级先验、内部节点输出节点势；视频头读取根的未递归压缩的inside。仍在单一网络内训练，最终分数来自既有答案似然后验，无后处理。

这不是新的attention或加法算子。候选贡献范围是：把问题树内容编码改为**保留可合并注意力统计量的内外表示**，保持目标区间之外上下文的计算排除，并通过同一多尺度答案似然训练。需实际核查 hateful-video 中是否已有此完整来源机制；不能拿一般attention、局部/全局融合或树结构的新名字作为novelty。

## 3. 范围、成本与检验

- 新增VLM调用0，新增缓存抽取0，新视频VLM调用方式不变。加性统计量需要约4倍的单head统计存储，投影与融合仍为单网络；复杂度线性于树节点数，不能未经实测宣称更快。
- 首版完整trial实测 HCS 235秒、HateMM 560秒（有并发），仅作预算参考；新架构以首个完整trial实测锁定每seed20或5 trial。若准备大规模运行时发现显著增耗，按AGENTS.md先说明必要性与替代方案。
- 同一结构用于两主数据，保留原50 epoch、四个标量的同一Optuna空间及checkpoint/test流程，不改搜索标量，不按within剪枝。具体搜索配置已在第5节、开跑之前写明。
- 预期检验：相同输入/8次预算对原骨干及初版；固定配置三seed去outside；attention权重改常数、保留加性统计量；必要时对照递归非线性组合。并列报0/8次（32次仅附加），所有三个主指标和seed方差。移除机制的增益要求仍按研究规则核对，不能只凭提案过审完成目标。
- 一次独立proposal review已GO；实现后做一次独立code review，不做smoke或缩短训练。初版完整搜索及结果不被新版本覆盖。共享实现如需迁移，等待现有训练结束再改其入口；新实验不跨import旧实验。

## 4. 当前决定

提案评审已GO、唯一一次code review已PASS；初版两主数据seed234已完成，完整结果见初版README第9节。当前证据支持检验聚合方式，不支持宣布初版失败归档、已SOTA、已完成backbone novelty，或放弃涨点目标。

## 5. 实现与开跑前固定协议

- 原型在本目录 `backbone.py`；共享trainer、data、QTL推断和统一评测未改。`src/qtl/content.py` 提供原样升入共享的ContextFusion和无模型状态的拓扑缓存。旧原型在HateMM搜索完成前保留冻结定义，已于2026-10-02全部结束后接到共享模块；迁移等价性验证见初版README第9节，未替换本修订版活动代码。新实验不import旧实验。
- 投影hidden128；每模态4个head，key线性权重跨模态共享；每个head保留128维value的加权和与1个mass。head拼接后Linear回128、LayerNorm只发生在节点读出。自底向上inside与自顶向下outside都只有加法，和直接对区间/补集做加权池化数学等价，不能宣称它保留了初版有序非线性递归的表达力。
- 叶子/内部节点融合器与初版相同；视频头读取根inside，秒级与节点势进入原有树似然。attention常数权重和关闭outside是预留消融开关，主配置两者均启用。新增VLM调用和缓存抽取均为0。
- 两主数据各独立训练50 epoch；Optuna TPE每seed独立study，sampler seed=训练seed。空间固定：lr log[1e-4,1e-3]，lamda_cma uniform[.5,2]，dropout uniform[.1,.5]，lr_answer log[1e-4,.1]；hidden128、batch32、crop_repeat5、long_T512、8档soft_both、node_prior=true，其余同第5版。
- validation AP/ROC均值选checkpoint，立即test；搜索目标仍为(test AP+ROC)/2，within与AP/ROC并列纳入方法结论，不改规则文件。每seed预算由首个完整trial ≤1h选20、>1h选5，自动锁定；不做smoke、不缩短训练。另记录validation选trial的test三项。
- 输出 `runs/20261001_associative_io_backbone/<corpus>/seed<seed>/`；全预算结束才判断seed。独立后台monitor、回传、三seed确认/必要消融沿用项目流程；初版结果单独保留。

独立code review已核验127个节点与直接区间/补集S/Z计算一致、outside自身缓存行梯度为0、变长/单叶/padding对齐、23个启用参数进入真实tree loss、s/g/phi进入最终后验、严格checkpoint重载与启动完整性链；未训练、未做smoke。数字记录在本机 `runs/20261001_associative_io_backbone/code_review/numeric_checks.json`，它只验证实现，不证明性能。

运行主机 **uoa-lab3 / sc474398**。新版两主数据seed234已满足完整训练的提案/代码评审条件，于2026-10-01 23:05 NZDT与仍在跑的初版HateMM并行启动；科学判断仍等各study完整预算结束。新版本从已完整结束的HCS搜索与三组诊断提出，不依赖旧HateMM暂时最优结果；不提前宣告初版晋级或淘汰。输出目录不同，新增共享文件不会被旧模型入口导入，在跑旧实现不改动。每个新study独立后台owner与monitor；输入与环境复用本日已全量核验的相同缓存/环境，训练入口继续检查覆盖、时间轴与split。

启动及monitor状态见 `research-wiki/STATUS.md`，本机启动记录 `runs/20261001_associative_io_backbone/setup/launches.json`。两项monitor首轮均已成功绑定并观测RUNNING，训练输入覆盖完整，实际已进入完整50 epoch训练；2026-10-01 23:55核验：HateMM首完整trial756.27秒、HCS314.65秒，两套seed234预算均按预写规则锁定为20 trial，来源为本机 `runs/20261001_associative_io_backbone/<corpus>/seed234/budget.json`；当时分别完成3/20、9/20，另各有一个活动trial，尚不能判候选结果。这是在同卡多任务并发下的耗时，不等同于独立吞吐对比。

## 6. HCS seed234 完整结果与先验诊断（2026-10-02）

运行主机 **uoa-lab3 / sc474398**；01:06:01正常结束，01:06完成通知后核验真实主/同会话子进程均退出并回传全部输出。20/20 COMPLETE；逐trial配置一致、epoch严格1–50、validation最高criterion对应最佳checkpoint；260份统一test评测可解析、三项有限，均覆盖79视频/18839秒，无缺失或额外视频。核验不能只依靠completion标记。

| 固定8次，seed234 | AP | ROC | within |
|---|---:|---:|---:|
| 加性统计量修订版，test选trial7 | .683294 | .655506 | .511165 |
| 递归初版，test选trial11 | .668767 | .635762 | .567188 |
| 第5版原骨干，test选trial13 | .688467 | .674845 | .556131 |
| 修订版仅validation选trial10 | .630660 | .613407 | .510797 |

原始来源为本机 `runs/20261001_associative_io_backbone/hateclipseg/seed234/trial{7,10}/metrics_test_fixed8.json`、`runs/20261001_inside_outside_backbone/hateclipseg/seed234/trial11/metrics_test_fixed8.json`、`runs/20260929_query_paradigm_r5/hateclipseg/seed234/trial13/metrics_test_fixed8.json`；选择记录在各自study_summary。修订版对初版+.014527/+.019744/−.056023，对第5版−.005173/−.019339/−.044966。只能说恢复部分pooled，不能说全面涨点，尤其不能忽略作为主指标的within。

读取固定0/8/32次及全部trial输出后发现：test选trial7的checkpoint来自validation epoch1，0次仅.502969/.452038/.516390，32次.658003/.640937/.516138；8次的视频均分AUC .884058，却没有相应视频内排序改善。全部20个trial的0次指标中位数.617564/.602228/.524718；低质先验并非所有trial同等严重，但不能把trial7的8次pooled优势解释成骨干定位贡献。前10个相同超参相对初版平均+.007944/+.026207/−.006607，相对第5版−.017849/−.017378/−.014971。这支持继续检查outside与注意力权重是否改善逐秒先验，而不是仅凭消除递归压缩就认定机制有效。不是bug已被排除的因果结论，也不据此更改已锁定的checkpoint/search标量。

派生审计/误差分析：`runs/20261001_associative_io_backbone/analysis/hateclipseg_seed234_comparison.json`、`hateclipseg_prior_diagnosis.json`，均保留原始来源；属于读取test后的开发证据。HateMM尚未跑满，不提前判两语料筛选、三seed确认或候选归档。

## 7. 已锁定的HCS结构诊断（训练前预写）

沿用第3节预设开关，固定完整搜索trial7配置，比较full、nooutside（仅io_outside=false）、noattention（仅io_attention=false，其余加性统计量与四head/readout保留）。各seed234/2025/3407、50 epoch、原validation选checkpoint与统一test评测，0/8次为主、32次附加，报告三项均值和seed标准差。full seed234复用trial7，其余8次完整训练，不重新调超参。这是结构诊断，不是Optuna确认，不把三seed固定配置均值冒充三个study最优。

锁定配置 `configs/diagnostic_hateclipseg_seed234.json`；入口 `launch/run_diag_hateclipseg_uoa-lab3.sh <full|nooutside|noattention>`；输出 `runs/20261001_associative_io_backbone/diagnostics/hateclipseg/<arm>/seed<seed>/`。复用的诊断owner已从旧原型升入 `src/qtl/diagnostics.py`，两个实验各自薄入口指定trainer、输出根和开关，避免跨实验import；只改变已完成的旧诊断入口，五项活动搜索的模型/trainer未动。CLI、shell语法和新旧源trial的配置/完整评测解析已核对；没有训练或smoke。

新增VLM调用与缓存抽取均0，按首完整HCS trial314.65秒估算8次训练累计约0.70小时，占卡并发耗时另记；不是独占吞吐承诺。此前五项完整搜索占满GPU，诊断等待资源。2026-10-02 03:13 NZDT，初版HCS两个确认study结束后，在uoa-lab3 / sc474398启动full、nooutside两条独立诊断链，各配monitor并验证首次RUNNING、实际完整训练已进入epoch；full seed234已核验复用。full于03:29:15结束并回传核验后，noattention已于03:31:10在同主机接续，独立monitor首次RUNNING和实际训练均正常；nooutside于03:36:42完成并已回传核验，noattention于03:53:19完成，03:55通知后已回传核验；启动记录 `runs/20261001_associative_io_backbone/setup/diagnostic_launches.json`，PID/monitor路径在STATUS。初版HateMM确认和修订版HateMM仍按原预算继续。

## 8. 固定配置诊断结果（截至2026-10-02 03:59）

full、nooutside、noattention三组均已完成，运行主机uoa-lab3 / sc474398，全部输出已回传本机。full seed234复用trial7，其余8次均为新跑的完整50 epoch；每组各三seed、39份统一test评测，均覆盖79视频/18839秒，无缺失或额外视频。逐seed检查了锁定配置、epoch1–50历史与日志、validation最高criterion所选checkpoint；noattention的model.pth也已实际加载核对epoch。noattention只改变io_attention=false，其它配置不变；03:53:19结束，实际主/同会话子进程为空，monitor成功通知并退出。以下为固定配置均值±总体seed标准差，**不是三个Optuna study最优值**。

| arm | 预算 | AP | ROC | within |
|---|---:|---:|---:|---:|
| full | 0 | 0.569952 ± 0.053677 | 0.544078 ± 0.072351 | 0.535919 ± 0.015260 |
| full | 8 | 0.648252 ± 0.033225 | 0.623946 ± 0.041971 | 0.548341 ± 0.027075 |
| full | 32 | 0.669080 ± 0.016441 | 0.640289 ± 0.021091 | 0.573861 ± 0.042254 |
| nooutside | 0 | 0.570363 ± 0.064341 | 0.535020 ± 0.073226 | 0.526894 ± 0.014329 |
| nooutside | 8 | 0.640263 ± 0.018540 | 0.612341 ± 0.009245 | 0.537365 ± 0.020966 |
| nooutside | 32 | 0.664037 ± 0.028596 | 0.639071 ± 0.016264 | 0.550987 ± 0.043241 |
| noattention | 0 | 0.589454 ± 0.060364 | 0.561298 ± 0.073314 | 0.514029 ± 0.020360 |
| noattention | 8 | 0.675338 ± 0.010891 | 0.640604 ± 0.009837 | 0.535214 ± 0.022023 |
| noattention | 32 | 0.683824 ± 0.020451 | 0.660414 ± 0.016770 | 0.567355 ± 0.041343 |

固定8次逐seed结果及相对同seed full的配对差值：

| arm | seed | validation epoch | AP / ROC / within | ΔAP / ΔROC / Δwithin |
|---|---:|---:|---|---|
| full | 234 | 1 | 0.683294 / 0.655506 / 0.511165 | — |
| full | 2025 | 36 | 0.657839 / 0.651702 / 0.558985 | — |
| full | 3407 | 30 | 0.603623 / 0.564630 / 0.574871 | — |
| nooutside | 234 | 2 | 0.619608 / 0.607107 / 0.508755 | -0.063685 / -0.048399 / -0.002410 |
| nooutside | 2025 | 47 | 0.664577 / 0.625334 / 0.544925 | +0.006737 / -0.026368 / -0.014060 |
| nooutside | 3407 | 6 | 0.636604 / 0.604583 / 0.558414 | +0.032982 / +0.039953 / -0.016458 |
| noattention | 234 | 1 | 0.690654 / 0.649404 / 0.520142 | +0.007360 / -0.006101 / +0.008977 |
| noattention | 2025 | 41 | 0.666275 / 0.626873 / 0.519147 | +0.008436 / -0.024829 / -0.039838 |
| noattention | 3407 | 47 | 0.669084 / 0.645536 / 0.566354 | +0.065462 / +0.080906 / -0.008517 |

移除outside的8次均值变化为−.007989/−.011605/−.010976，配对差值的总体seed标准差为.040815/.037550/.006135。HCS的ROC平均下降超过.01，within三个seed都下降，但pooled各seed不一致，不能称两语料贡献已确认。0次变化+.000410/−.009057/−.009025；32次−.005043/−.001218/−.022874。

移除学习注意力的8次均值变化为**+.027086/+.016658/−.013126**，配对差值的总体seed标准差为.027139/.046068/.020193。AP三个seed都提高，ROC的均值收益主要来自seed3407，within均值下降。0次变化+.019501/+.017221/−.021890；32次+.014745/+.020126/−.006506。**学习注意力没有pooled增益证据，不能作为已验证的novelty贡献**；无注意力版本也不是三主指标全面改进。outside效应是在full注意力配置下得到，不能直接推断无注意力版本仍有同样outside贡献。

原始来源：本机 `runs/20261001_associative_io_backbone/diagnostics/hateclipseg/<arm>/seed<seed>/metrics_test_fixed<预算>.json`；full seed234复用 `runs/20261001_associative_io_backbone/hateclipseg/seed234/trial7/metrics_test_fixed<预算>.json`。完整三组核验、标准差与0/8/32次逐seed配对差值保存在 `runs/20261001_associative_io_backbone/analysis/hateclipseg_locked_diagnostics.json`；先前full/nooutside单组审计仍保留。这些是读取test后的开发诊断，不能替代完整Optuna确认。

当前决定：HCS证据不支持继续将学习注意力当作收益来源；HateMM完整搜索已完成，结果及确认安排见第9–10节；后续修订仍结合两语料自身的结构诊断。已释放的短诊断位置用于初版HateMM nooutside，单独绑定monitor，未覆盖或重启本实验。

## 9. HateMM seed234完整搜索结果（2026-10-02）

运行主机uoa-lab3 / sc474398，04:45:32正常结束；04:47完成通知后核验真实主/同会话子进程为空，全部输出已回传本机。Optuna数据库与study_summary均为20/20 COMPLETE，首trial756.27秒锁定20档。逐trial配置与搜索参数一致，soft_both按既有trainer固定为单问题categories=[0]；全部history与run.log严格epoch1–50，实际加载每个model.pth核对validation最高AP/ROC均值所选epoch。260份统一test评测覆盖214视频/29269秒，三项有限，无缺失或额外视频。

| 固定8次，seed234 | AP | ROC | within |
|---|---:|---:|---:|
| 修订版，test选trial10 | .651747 | .874278 | .765879 |
| 初版，test选trial17 | .679560 | .881351 | .753400 |
| 第5版，test选trial9 | .694450 | .886706 | .788733 |
| 修订版，仅validation选trial4 | .616726 | .868900 | .742323 |

对初版差值−.027812/−.007074/+.012480，对第5版−.042702/−.012428/−.022854。HateMM的within较初版改善，但pooled下降，相对第5版三项均低；与HCS一起看，修订版没有全面涨点。最优trial10的checkpoint来自validation epoch25；validation选trial4的checkpoint为epoch23。

0次修订版为.549902/.764712/.776061，对第5版差−.085289/−.058477/−.004633，主要损失在视频间区分；32次为.682796/.884678/.778866，对第5版−.000420/+.010532/+.024715，仅作附加预算结果。前10个相同超参trial的8次平均差对初版−.013407/−.005263/+.006711，对第5版−.046398/−.013684/−.036521，下降不只发生于单个test最优trial。由这些输出与HCS去注意力诊断，后续需检验内容先验的视频间区分与权重学习，而不能将加性聚合本身等同于有效骨干贡献。上述为读取test后的开发证据，不是未揭盲检验。

原始来源：本机 `runs/20261001_associative_io_backbone/hatemm/seed234/trial{10,4}/metrics_test_fixed<预算>.json`、`runs/20261001_inside_outside_backbone/hatemm/seed234/trial17/metrics_test_fixed<预算>.json`、`runs/20260929_query_paradigm_r5/hatemm/seed234/trial9/metrics_test_fixed<预算>.json`。完整预算、试验选择来自各study_summary与Optuna数据库；逐trial核验、0/8/32次比较和首10个相同超参比较在 `runs/20261001_associative_io_backbone/analysis/hatemm_seed234_comparison.json`。

## 10. 补seed完整确认（启动前固定，2026-10-02）

两主数据seed234已完整跑满20 trial，四个pooled值均超过规则8既定baseline筛选值，因此补seed2025/3407，各自独立完整Optuna。配置、架构、输入、8档软答案、四标量搜索空间、test(AP+ROC)/2目标、validation选checkpoint及50 epoch均沿用第5节，不因HCS消融结果而改变确认中的注意力开关。每个新study按首完整trial记录预算；已有方法耗时对应20-trial档，不缩减。补seed检验随机种子稳定性，不能称已超过第5版或已完成novelty。

复用 `launch/run_<corpus>_uoa-lab3.sh <seed>`，输出 `runs/20261001_associative_io_backbone/<corpus>/seed{2025,3407}/`，各有独立后台owner与monitor。新增VLM调用0、缓存抽取0；按首trial756/315秒估算，两语料两seed共80次完整训练累计运行时间约11.9小时，实际并发墙钟另记，不当成独占GPU时间承诺。资源以实时检查为准，不干扰其他任务，所有活动模型/trainer/评测器保持原文件。

四项确认已于2026-10-02 04:53:50–52在uoa-lab3 / sc474398启动，输入覆盖完整、实际训练均已进入epoch，四个独立monitor均已首轮RUNNING并绑定正确身份；启动记录为 `runs/20261001_associative_io_backbone/setup/confirmation_launches.json`，PID、monitor和当前进度由STATUS维护。初版HateMM两个确认study均已完整结束并回传汇总（初版README第12节）；初版nooutside已结束并独立汇总其两语料机制证据（初版README第11.2节），不与本修订版混作同一方法。


确认study首trial预算已于2026-10-02 05:57核验回传：HateMM seed2025/3407为1415.40/1413.64秒，HCS为578.15/577.11秒，四项均已锁定20 trial；来源为本机 `runs/20261001_associative_io_backbone/<corpus>/seed<seed>/budget.json`。这些是多任务并发耗时，不等同于独占GPU吞吐；完整搜索不缩减。

## 11. HateMM锁定配置结构诊断（开跑前固定，2026-10-02）

沿用第3节预设两项开关，固定已完成20-trial搜索的seed234 trial10配置，比较full、noattention（仅io_attention=false）、nooutside（仅io_outside=false）。各seed234/2025/3407完整50 epoch，validation AP/ROC均值选checkpoint、统一test评测；报告0/8次三个主指标、seed标准差和配对差值，32次仅附加。full seed234复用trial10，其余8次完整训练；不调参、不改变确认study。它对应本修订版HCS第7–8节诊断，检验HCS取消学习注意力的pooled收益能否跨语料成立，以及本版本outside的两语料贡献。初版outside证据不得代替本修订版的检验。这些是固定配置诊断，不是Optuna确认。

锁定配置 `configs/diagnostic_hatemm_seed234.json`，从本机 `runs/20261001_associative_io_backbone/hatemm/seed234/trial10/summary.json` 原样取cfg；已确认50 epoch、full开关和validation checkpoint epoch25。入口 `launch/run_diag_hatemm_uoa-lab3.sh <full|noattention|nooutside>` 使用既有共享diagnostic owner和本版trainer。输出 `runs/20261001_associative_io_backbone/diagnostics/hatemm/<arm>/seed<seed>/`；每条长链独立monitor，复用前检查identity与已有输出。新增配置/启动入口已核对源配置和shell语法，无smoke，无活动模型/trainer/评测器修改。

新增VLM调用0、新缓存抽取0，新视频处理方式不变。按锁定trial10的实测1169.63秒（并发环境）估算，8次完整训练累计运行时间约2.60小时；不是独占GPU时间或总墙钟承诺。05:55 lab3六项搜索占用约16.6 GiB且GPU利用率98%，当前不叠加训练以免延长已运行搜索；本机仍不跑训练，lab1/lab-server仍被其它任务占用。优先利用初版HateMM搜索完成后释放的两个位置启动full、noattention；其中一条诊断结束后接nooutside，各自绑定独立monitor。2026-10-02 07:12:14–15，初版HateMM两个study结束后已在uoa-lab3 / sc474398接续full、noattention；full seed234已核验复用，两个arm实际训练均已进入epoch，各自独立monitor首轮RUNNING绑定正确。full于07:59:21结束并已核验回传，nooutside已于08:02:29利用释放位置接续，独立monitor首次RUNNING绑定正确且seed234已进入epoch；noattention于08:15:37完成并已核验回传；启动记录 `runs/20261001_associative_io_backbone/setup/diagnostic_hatemm_launches.json`，PID和接续维护在STATUS。

## 12. HateMM固定配置诊断结果（截至2026-10-02 08:47）

full于07:59:21在uoa-lab3 / sc474398结束，08:00完成通知后核验真实主/同会话子进程为空，全部输出已回传本机。三seed均使用trial10锁定配置，seed234复用原trial10，其余为新跑完整50 epoch。history与run.log严格epoch1–50，validation最大(AP+ROC)/2所选checkpoint与实际加载model.pth一致；各13份、共39份统一test评测完整，覆盖214视频/29269秒，无缺失或额外视频。这是固定配置结构诊断，不是三seed Optuna确认。

| 预算 | full AP均值±总体seed标准差 | ROC | within |
|---|---:|---:|---:|
| 0 | 0.520258 ± 0.028975 | 0.752572 ± 0.008596 | 0.741415 ± 0.031647 |
| 8 | 0.611048 ± 0.040221 | 0.860227 ± 0.009987 | 0.737149 ± 0.024840 |
| 32 | 0.663253 ± 0.013976 | 0.882091 ± 0.002100 | 0.755020 ± 0.018714 |

| seed | validation epoch | 固定8次 AP / ROC / within |
|---|---:|---|
| 234 | 25 | 0.651747 / 0.874278 / 0.765879 |
| 2025 | 23 | 0.556285 / 0.851954 / 0.740291 |
| 3407 | 10 | 0.625113 / 0.854450 / 0.705277 |

full固定8次AP的seed标准差为.040221，不能据seed234就称性能稳定。noattention于08:15:37在同主机正常结束，08:16通知后核验真实主/同会话子进程为空并回传全部输出；三seed分别完整50 epoch，严格保持trial10配置，仅io_attention=false；实际checkpoint对应validation最高(AP+ROC)/2。39份统一test评测完整，覆盖214视频/29269秒，无缺失或额外视频。

| 预算 | noattention AP / ROC / within，均值±总体seed标准差 | noattention − full 配对均值 |
|---|---|---|
| 0 | 0.432374 ± 0.056303 / 0.732141 ± 0.020458 / 0.718074 ± 0.027467 | -0.087884 / -0.020431 / -0.023340 |
| 8 | 0.639454 ± 0.026836 / 0.865572 ± 0.001456 / 0.741050 ± 0.017690 | +0.028406 / +0.005344 / +0.003901 |
| 32 | 0.666891 ± 0.004986 / 0.878463 ± 0.003070 / 0.764722 ± 0.016267 | +0.003638 / -0.003629 / +0.009702 |

| seed | validation epoch | noattention固定8次 AP / ROC / within | 相对同seed full |
|---|---|---|---|
| 234 | 15 | 0.669696 / 0.867628 / 0.737410 | +0.017948 / -0.006649 / -0.028469 |
| 2025 | 13 | 0.644190 / 0.864629 / 0.721436 | +0.087906 / +0.012675 / -0.018855 |
| 3407 | 42 | 0.604476 / 0.864457 / 0.764305 | -0.020637 / +0.010007 / +0.059028 |

固定8次配对差值的总体seed标准差为 0.044925 / 0.008551 / 0.039178。移除学习注意力后HateMM三项均值提高，但AP收益主要来自seed2025，seed3407的AP下降，不能写每个seed稳定改善；0次三项均下降，32次ROC也下降。主操作点的收益不能外推为所有预算均受益。

本修订版HCS移除注意力的8次配对均值为 +0.027086 / +0.016658 / -0.013126；与HateMM合看，两语料均不支持将学习注意力作为8次pooled增益来源。HCS within下降、HateMM 0次变差，故也不能直接把noattention认定为全面改善的新主线。outside效应仍在full注意力配置下检验，不能自动外推到noattention。初版outside两语料效应单独保留，不移作本修订版证据。

nooutside于08:42:23在uoa-lab3 / sc474398结束，08:42通知后核验真实主/同会话子进程为空，全部输出回传。三seed严格trial10配置、仅io_outside=false，各完整50 epoch；validation最高(AP+ROC)/2对应实际checkpoint，39份统一评测完整，214视频/29269秒无缺失/额外项。

| 预算 | nooutside AP / ROC / within，均值±总体seed标准差 | nooutside − full 配对均值 |
|---|---|---|
| 0 | 0.525464 ± 0.064207 / 0.779407 ± 0.030263 / 0.717649 ± 0.002527 | +0.005206 / +0.026835 / -0.023766 |
| 8 | 0.618460 ± 0.028211 / 0.853547 ± 0.020676 / 0.735958 ± 0.024388 | +0.007412 / -0.006680 / -0.001191 |
| 32 | 0.662751 ± 0.003603 / 0.871161 ± 0.009118 / 0.745743 ± 0.017487 | -0.000502 / -0.010930 / -0.009277 |

| seed | validation epoch | nooutside固定8次 AP / ROC / within | 相对同seed full |
|---|---|---|---|
| 234 | 16 | 0.627231 / 0.862702 / 0.730495 | -0.024517 / -0.011576 / -0.035384 |
| 2025 | 31 | 0.647782 / 0.873019 / 0.709197 | +0.091497 / +0.021065 / -0.031093 |
| 3407 | 37 | 0.580369 / 0.824920 / 0.768181 | -0.044744 / -0.029530 / +0.062904 |

HateMM固定8次移除outside后的配对均值变化为 **+.007412/−.006680/−.001191**，配对差值总体seed标准差.060028/.020944/.045356；AP均值反而提高，ROC下降不足.01。HCS本版本对应变化 **−.007989/−.011605/−.010976**。因此修订版outside **不满足规则14(g)在两语料主操作点的平均贡献要求**。HateMM 32次ROC下降.010930只能作为附加预算观察，不能事后更换主操作点使结论通过。初版已有自己的两语料贡献证据，不能移作本修订版证据。

至此两语料固定配置full/noattention/nooutside诊断均齐全。学习注意力没有8次pooled收益，outside未通过本版两语料贡献检验；这些都是固定配置诊断，不是Optuna确认。HateMM两个确认study仍按20-trial预算继续，不取消或提前宣告其完整结果。下一轮提案改为保留r5跨模态编码与视频头的区间内外残差，检验能否减少内容先验损失，见 `../20261002_residual_io_backbone/README.md`；在独立proposal/code review通过前不训练，不把残差技巧本身称作novelty。

原始来源：本机 `runs/20261001_associative_io_backbone/diagnostics/hatemm/<arm>/seed<seed>/metrics_test_fixed<预算>.json`，full seed234复用 `runs/20261001_associative_io_backbone/hatemm/seed234/trial10/metrics_test_fixed<预算>.json`。三组审计与完整均值/标准差/配对差值 `runs/20261001_associative_io_backbone/analysis/hatemm_locked_diagnostics.json`；两语料对应汇总 `analysis/two_corpus_attention_diagnosis.json`、`analysis/two_corpus_outside_diagnosis.json`；单组审计 `analysis/hatemm_locked_{full,noattention,nooutside}.json`。所有来源均为本修订版，不混入初版。

## 13. HCS三seed完整搜索结果（2026-10-02）

uoa-lab3 / sc474398的seed2025/3407均于08:01:44结束。08:08核验真实主/同会话子进程均为空，全部输出已回传本机；两个monitor于08:03成功通知并退出。连同seed234，各study的预算文件、SQLite与study_summary均为20/20 COMPLETE；60个trial全部history与run.log严格epoch1–50，配置/搜索参数一致，实际加载checkpoint与validation最高(AP+ROC)/2所选epoch一致。780份统一test评测覆盖79视频/18839秒，无缺失或额外视频，三指标有限。

| seed | test选trial / checkpoint epoch | 固定8次 AP / ROC / within | validation选trial / epoch | 对应test AP / ROC / within |
|---|---|---|---|---|
| 234 | 7 / 1 | 0.683294 / 0.655506 / 0.511165 | 10 / 1 | 0.630660 / 0.613407 / 0.510797 |
| 2025 | 7 / 5 | 0.679817 / 0.662033 / 0.557129 | 11 / 16 | 0.676279 / 0.643481 / 0.545142 |
| 3407 | 0 / 1 | 0.680890 / 0.642109 / 0.509980 | 0 / 1 | 0.680890 / 0.642109 / 0.509980 |

| 固定8次三seed均值±总体标准差 | AP / ROC / within |
|---|---|
| 修订版 | 0.681333 ± 0.001454 / 0.653216 ± 0.008294 / 0.526091 ± 0.021952 |
| 初版 | 0.668398 ± 0.003274 / 0.642595 ± 0.005721 / 0.560326 ± 0.004923 |
| 第5版 | 0.686455 ± 0.003656 / 0.673725 ± 0.005441 / 0.560226 ± 0.013556 |
| 修订版只按validation选trial | 0.662610 ± 0.022670 / 0.632999 ± 0.013865 / 0.521973 ± 0.016386 |

修订版对初版固定8次均值变化 **+0.012936 / +0.010621 / -0.034235**：pooled恢复，但within下降。对第5版 **-0.005122 / -0.020509 / -0.034135**，三主指标仍低，不能认定相对当前主线涨点。validation选trial对初版变化 +0.010526 / +0.012459 / -0.039517，对第5版 -0.017037 / -0.016996 / -0.045850，趋势相同。

0次均值±标准差 0.545151 ± 0.087731 / 0.509617 ± 0.096909 / 0.521391 ± 0.008147，对第5版变化 -0.112442 / -0.137046 / -0.018052；32次 0.660977 ± 0.015799 / 0.638784 ± 0.008342 / 0.543288 ± 0.040768，对第5版 -0.034393 / -0.036340 / -0.071644。三seed最优trial中两个checkpoint来自epoch1，虽然实际均训练满50 epoch，其0次pooled也较弱；已有视频间区分损失并未由完整搜索消除。这是读取test后的开发证据，不是未揭盲确认；trial仍按既定test目标选择，checkpoint按validation选择。

结合第8节固定配置诊断，学习注意力没有HCS pooled增益证据；本版outside的HCS平均效应仍须本版HateMM对应消融验证。初版outside已满足两语料平均贡献要求，但不能移作本版证据。HateMM seed2025/3407于08:08各完成8/20，继续完整预算；本版两主数据三seed汇总尚未齐全。HateMM三组诊断已完整结束并与本版HCS对应消融汇总（第12节），不重复启动。HCS三seed完整搜索是Optuna确认，固定配置诊断单独报告。

原始来源：本机 `runs/20261001_associative_io_backbone/hateclipseg/seed<seed>/trial<编号>/metrics_test_fixed<预算>.json`，对照分别为初版与第5版对应路径。逐trial完整性检查、SQLite核对、逐seed及0/8/32次比较、validation选择和标准差：`runs/20261001_associative_io_backbone/analysis/hateclipseg_three_seed_search.json`。过程核验记录 `runs/20261001_associative_io_backbone/setup/hcs_confirmation_completion_probe.json`。
