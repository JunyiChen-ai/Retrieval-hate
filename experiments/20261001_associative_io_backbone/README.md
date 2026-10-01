# Backbone 修订候选：保留加性统计量的 inside–outside 编码器

截至2026-10-01；状态：[一次独立proposal review：GO](../../docs/reviews/20261001_associative_io_backbone_proposal.md)，骨干与训练/搜索入口已实现，[一次独立code review：PASS](../../docs/reviews/20261001_associative_io_backbone_code.md)；准备完整搜索。当前 inside–outside 初版的 HateMM 完整搜索仍在运行，保留其全部20 trial；本提案不替换在跑代码。用户授权继续自动迭代 backbone，优先 novel 且涨点，within 与 AP / ROC 并列为主指标。Pursuing goal 保持 paused，继续由 heartbeat / 完成通知接续。

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

提案评审已GO、唯一一次code review已PASS；初版HateMM完整搜索继续。当前证据支持检验聚合方式，不支持宣布初版失败归档、已SOTA、已完成backbone novelty，或放弃涨点目标。

## 5. 实现与开跑前固定协议

- 原型在本目录 `backbone.py`；共享trainer、data、QTL推断和统一评测未改。`src/qtl/content.py` 提供原样升入共享的ContextFusion和无模型状态的拓扑缓存。旧原型在当前HateMM搜索完成前保留冻结定义；结束后再将旧入口接到共享模块，避免运行中换代码。新实验不import旧实验。
- 投影hidden128；每模态4个head，key线性权重跨模态共享；每个head保留128维value的加权和与1个mass。head拼接后Linear回128、LayerNorm只发生在节点读出。自底向上inside与自顶向下outside都只有加法，和直接对区间/补集做加权池化数学等价，不能宣称它保留了初版有序非线性递归的表达力。
- 叶子/内部节点融合器与初版相同；视频头读取根inside，秒级与节点势进入原有树似然。attention常数权重和关闭outside是预留消融开关，主配置两者均启用。新增VLM调用和缓存抽取均为0。
- 两主数据各独立训练50 epoch；Optuna TPE每seed独立study，sampler seed=训练seed。空间固定：lr log[1e-4,1e-3]，lamda_cma uniform[.5,2]，dropout uniform[.1,.5]，lr_answer log[1e-4,.1]；hidden128、batch32、crop_repeat5、long_T512、8档soft_both、node_prior=true，其余同第5版。
- validation AP/ROC均值选checkpoint，立即test；搜索目标仍为(test AP+ROC)/2，within与AP/ROC并列纳入方法结论，不改规则文件。每seed预算由首个完整trial ≤1h选20、>1h选5，自动锁定；不做smoke、不缩短训练。另记录validation选trial的test三项。
- 输出计划 `runs/20261001_associative_io_backbone/<corpus>/seed<seed>/`；全预算结束才判断seed。独立后台monitor、回传、三seed确认/必要消融沿用项目流程；初版结果单独保留。

独立code review已核验127个节点与直接区间/补集S/Z计算一致、outside自身缓存行梯度为0、变长/单叶/padding对齐、23个启用参数进入真实tree loss、s/g/phi进入最终后验、严格checkpoint重载与启动完整性链；未训练、未做smoke。数字记录在本机 `runs/20261001_associative_io_backbone/code_review/numeric_checks.json`，它只验证实现，不证明性能。

运行主机计划 **uoa-lab3 / sc474398**。新版两主数据seed234已分别满足完整训练的提案/代码评审条件，拟与仍在跑的初版HateMM并行；科学判断仍等各study完整预算结束。新版本从已完整结束的HCS搜索与三组诊断提出，不依赖旧HateMM暂时最优结果；不提前宣告初版晋级或淘汰。输出目录不同，新增共享文件不会被旧模型入口导入，在跑旧实现不改动。每个新study独立后台owner与monitor；输入与环境复用本日已全量核验的相同缓存/环境，训练入口继续检查覆盖、时间轴与split。
