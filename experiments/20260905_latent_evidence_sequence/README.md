# 候选6：视频标签约束的局部证据状态模型

2026-09-05 提案；独立[proposal review](REVIEW_RULE4.md)及唯一[code review](REVIEW_RULE6.md)均GO。HCS seed234完整搜索已回传核验，过本语料单seed门；HateMM仍在完整搜索，尚不能补确认seed。三个模块与整体novelty仍待实证。

## 1. 来源与失败观察

候选5的两语料三seed数值确认已通过，但HCS三个核心替换臂均不满足每seed同向提升。已保存test预测显示HCS正视频内局部区分较弱：67个混合视频中21–32个AUC低于.5，正秒/背景秒均分差仅.023–.049。来源 `runs/20260905_interventional_evidence/error_analysis/saved_prediction_diagnostics.json`、`ablations/hateclipseg/three_seed_summary.json`。这些是developmental error analysis，不把test标签用于训练。

本候选改变监督与推断关系：不用视频top-k BCE再附加冻结HMM块目标，而将视频标签解释为“潜在局部序列是否曾进入正状态”，对所有合法序列精确求和。输入仍为同一个冻结Qwen四路内容干预，直接复用已解析的v2原始logits，不重新抽取。序列模型从同一训练集视频标签联合学习，推断时不提供视频标签。不是把既有预测拿来平滑。

## 2. 三个模块的具体定义

### M1：四路VLM联合观测似然

在K30/K4原窗口记录中读取 `[L_av,L_v,L_a,L_empty]`，按已有时间对齐映射到内容snippet网格，拼为8维联合观测。各通道只用train统计作标准化，保存均值/标准差（数值稳定下限1e-4）；不使用test/val分布估计统计。每个潜在状态s∈{背景,目标}有一个可学习8维高斯观测密度，协方差用Cholesky参数化、对角softplus+1e-4。原提案“两粒度log density平均”是未归一化几何均值，已在实现前删除；只使用归一化联合密度。

完整协方差表示同一VLM的四种内容输入及两粒度之间的依赖，不再把可逆差分称为新增信息；它也不等于真实因果效应。M1不是另一个单独训练teacher，所有均值/协方差参数参与同一目标反传。负视频“所有状态为背景”的约束负责区分状态语义，不能只靠初始化认为前景已识别。消融：`diagonal_emission`（同四路观测但去协方差依赖）、`full_input_emission`（只使用两粒度av观测，密度为2维），另保留旧裁定输入对照。新VLM模块novelty须相对这些实验证据判定，不能凭高斯密度命名成立。

### M2：内容条件化的局部状态转移骨干

I3D/VGGish/BERT内容输入映射为128维h_t，单层kernel3的一维时间卷积（padding不进入有效状态序列），GELU/dropout；相邻 `[h_{t-1},h_t,h_t-h_{t-1}]` 产生两个转移logit，分别定义P(s_t=1|s_{t-1}=0,x)与P(s_t=0|s_{t-1}=1,x)。初始状态概率由首有效token预测。内容不是事后给概率加偏移，而决定何处允许状态变化，模型可学习保持或切换，固定平滑强度/时长先验均不存在。

消融 `static_transition`：所有时间共享可学习的2×2转移矩阵，保留相同VLM似然与训练；`no_temporal_content`：去时间卷积但保留内容条件转移。不把“时间卷积”本身称novel；要检验内容条件转移是否比固定HMM和局部MLP实际有贡献。

### M3：带视频事件约束的联合训练与精确后验融合

对给定内容x和VLM观测o，定义单一模型

`p(s,o|x)=p(s_1|x_1) · ∏_{t>1}p(s_t|s_{t-1},x) · ∏_t p(o_t|s_t)`。

视频标签y=0限定所有s_t=0；y=1限定至少一个s_t=1。用“未出现正状态的背景/正状态/已出现正状态的背景”三状态自动机计算log Z_0和log Z_1，不用近等数相减计算正事件。无标签配分函数log Z=logaddexp(log Z_0,log Z_1)。训练每视频损失为

`L = (log Z − log Z_y) − log Z / (D·T)`，D为该臂观测维数，完整模型D=8。

第一项是精确视频事件负对数似然，第二项为平均每token每观测维度负对数似然，用来学习观测密度而非只用少数top-k位置；两项固定等权，无新的损失权重搜索。连续密度NLL可为负，不把有限负loss视为故障。所有计算log域，padding不参与Z，train标签不进入test后验。正式分数是同一模型forward–backward得到的 `P(s_t=1|o,x)`，沿用既有5crop/1fps映射与统一评测器；不再加HMM先验或Yager融合，不做推理后处理。

上述损失是判别事件项+按维度/长度归一化生成项的混合目标，不是未经加权的联合NLL。标准化空间的密度只用于本模型训练；不同观测维度的NLL绝对值不可直接横比。

主消融 `event_to_topk`：保留完整状态模型、相同生成NLL及最终后验，训练中只把视频事件NLL换成后验top-⌈T/16⌉均值BCE；没有推理top-k或后处理。辅助 `independent_state`：转移不依赖上一状态，使用内容给出的逐token状态概率，保留精确视频事件监督和同一观测似然；`no_observation_likelihood`：去第二项，检查联合观测学习是否有必要。M3的主张是训练中视频约束与局部状态边缘化，不是发明forward–backward。`independent_state`仍保留事件监督，不能单独证明事件约束的贡献。

## 3. 与既有工作的区别和待审查风险

候选1的HMM是先拟合离散裁定、再固定输出后验/块标签给另一MIL网络；本候选无独立冻结标签模型，连续观测密度、内容条件转移与视频事件似然共同反传，是最终模型本身。先前条件HMM比例估计失败，因此不使用q_v比例监督。历史semi-Markov/平滑负结果不能当先验阻断，但本方案必须与它们披露区别：不使用固定平滑或时长先验。

统计/神经HMM、input-output HMM、弱标注序列约束与高斯密度均是已有方法；迁移来源是否已用于hateful video须由独立规则4 review实际检索。不能声称通用数学首创或仅凭三个部件构成novel paradigm。风险包括：长视频事件概率饱和；少量负视频不足以识别前景；重复对齐的观测不独立；高斯假设不合适；生成NLL主导任务；转移学成常量。它们交由完整训练及消融验证，不加CPU/smoke/短跑门。

## 4. 固定训练搜索与去向

两语料使用同一架构、输入处理、损失、推断及搜索空间。仅搜索lr log[1e-4,1e-3]、dropout {.1,.2,.3}、max_seqlen {150,200,300}；固定50epoch、batch32、hidden128、kernel3、两状态、两粒度、两项等权、数值下限1e-4。明确固定设计参数并非无超参数。数据与统一评测覆盖沿用候选5，不修改评测器。

代码评审通过后，HateMM/HCS seed234在lab1/lab3并行完整Optuna。首trial实测≤1小时则各20trial，否则5；不预先缩减预算。validation只在trial内选checkpoint；test(AP+ROC)/2选trial，within沿用现行下限。两语料过筛再补2025/3407。三核心消融主对应M1=`diagonal_emission`、M2=`static_transition`、M3=`event_to_topk`，两语料三seed需各满足14(g)。独立评审如认为对照不能隔离模块，应明确实现边界，而非新增实验前门。

最强baseline+同输入为最终声明缺口；不能以候选5完成过缓存抽取就跳过。未经评审及完整实证，三个模块和范式均仅为待检验提案。

## 5. 实现与启动记录

代码 `model.py/train.py/search.py`，复用 `src/fixed_optuna_protocol.py` 与统一评测器。既有50epoch训练/validation checkpoint/test循环从候选5升入 `src/fixed_training_protocol.py`，原始v2缓存解析升入 `src/interventional_observations.py`；候选5只改调用共享实现，不重跑其已结束实验。没有实验间import。

为避免逐token Python循环占用GPU，forward/backward使用归一化log-semiring并行前缀/后缀矩阵积，数学仍为精确三状态求和；每步矩阵减标量并跟踪累计log scale，后验由归一化alpha/beta得到。不可达状态使用−1e30表示数值零，避免全−inf求和的NaN梯度；padding为单位矩阵。该实现不是额外模型或后处理。

初始高斯均值由train负/正视频全部观测的均值初始化，再联合学习；未读取帧标签。Cholesky初始单位矩阵；初始正概率及背景→正概率初始化为1/max_seqlen，正→背景为.5，转移预测权重零初始化，之后全部可学习。初始化规则固定，不额外搜索。`static_transition`保留相同内容条件初始概率。

训练仍沿用既有均匀抽取的max_seqlen网格，validation/test保留完整snippet网格并按统一映射升至1fps；这是离散观测索引状态模型，不能声称转移是可跨采样率解释的每秒物理发生率。训练/评测采样密度差异、对齐观测时间相关性均为已披露的限制，不因此更改评测协议。

运行主机：HateMM在sc474397/uoa-lab1，HateClipSeg在sc474398/uoa-lab3；正式入口 `launch/run_search.sh <corpus> 234`，输出 `runs/20260905_latent_evidence_sequence/<corpus>/seed234/`。唯一code review已通过，按该入口启动并自动绑定当前会话monitor；首trial完成后记录实测耗时/预算。

23:56正式启动，两搜索与SSH解耦，PID/PGID分别1887909/3170219；自动monitor首次检查均RUNNING，首trial已正常输出epoch日志。同步前后检查了commit一致性和工作树，已有CLAUDE.md脏改动、tandem.html及远端idea-stage未跟踪目录均保留。仅同步本候选/共享实现与本轮文档，未修改CLAUDE.md或研究规则。

## 6. HCS seed234完整结果与接续诊断

2026-09-06 00:20通知后核验lab3进程退出，20/20 trial均COMPLETE并全量回传；50epoch、validation checkpoint、配置、val63/test79覆盖与原始指标一致性通过 `runs/20260905_latent_evidence_sequence/hateclipseg/seed234/artifact_audit.json`。首trial78.196859秒，预算固定20，来源 `budget.json`。

按test选trial17（lr=.00042073384058945477/dropout=.2/max_seqlen=300），validation选epoch2；test AP/ROC/within=`.690827333/.664875187/.580321710`，通过本语料固定单seed门。原始来源 `trial17/metrics.json`。仅validation排序会选trial0，test=`.685845940/.663355041/.581215112`，零额外训练参考、不作门。不能把HCS单seed结果与旧候选三seed均值直接视为已确认提升。

HateMM搜索未结束，不启动2025/3407确认seed。利用空闲lab3，锁定HCS trial17配置跑已评审八臂完整诊断：diagonal_emission、static_transition、event_to_topk、full_input_emission、raw_verdict、no_temporal_content、independent_state、no_observation_likelihood。每臂50epoch/validation checkpoint/test，最多3项并行，不另搜消融超参。主机sc474398，入口 `launch/run_module_ablations.sh hateclipseg 234`，输出 `runs/20260905_latent_evidence_sequence/ablations/hateclipseg/seed234/`，自动配置独立monitor。

已复用两次的锁定配置启动逻辑升入 `scripts/run_locked_ablations.sh`，两候选launcher只提供各自已评审臂列表，不修改模型或训练代码。启动前检查所有目标不存在，并用进程锁防重复；共享消融审计支持显式 `--arms`。完整搜索审计允许全部trial因within被剪枝而best=null，报告“无合格trial”而非把审计脚本异常误认为训练失败。
