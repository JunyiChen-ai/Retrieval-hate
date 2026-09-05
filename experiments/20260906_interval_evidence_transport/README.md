# 候选9：区间证据的内容条件分配

2026-09-06。状态：独立规则4 review及一次code review均GO（见 REVIEW_RULE4.md、REVIEW_RULE6.md）；07:24 NZST已双机启动正式seed234搜索。不是候选7重启，不声称三模块已经有效或已有novel paradigm证据。

## 依据与问题

C8 HCS20trial完整输出均未达到within下限，最高无约束test配置为 .604427/.589883/.510205；保存预测不是常数，但67个混合视频中32个AUC<.5。来源 `runs/20260906_censored_evidence_process/error_analysis/hcs_seed234_saved_predictions.json`。C1 HCS对应配置 .695235/.679345/.545769，C1分析表明直接VLM证据支持定位，而骨干主要学习视频间分离及两尺度可靠性。来源 `runs/20260903_hier_evidence_mil/hateclipseg/seed234/trial8/metrics.json` 和实验README §9。

这里的开发期test分析影响设计，不作为未揭盲证据，也不把test标签加入训练。具体假设：VLM裁定指向一个窗口，不应逐秒广播成相同logit，也不必强求内容模型只通过训练蒸馏记住它。把裁定作为区间观测，用窗口内内容决定证据分配，并让同一个分配算子参与跨模态特征交互与最后读出。

## 三个模块与同一个潜在分配变量

**M1：内容条件的序数观察通道。** 原Qwen K30/K4等级0=无证据、1=含糊、2=隐式/编码仇恨、3=明确仇恨；这是语义明确程度，不声称等级/3是时间占比或概率。窗口音频/文本与视觉按实际交叠求均值，同一网络产生潜在等级z=0..3的prior `pi_j(z|window content)`。每尺度一个可学习精度tau，`E_k(g|z)=softmax_g[-softplus(tau_k)*(g-z)^2]`。读取缓存grade g后得到 `posterior_j(z) ∝ pi_j(z)*E_k(g|z)`，其state embedding期望与两路窗口内容共同形成区间query。不是简单累计码换自由embedding（两者可重参数化等价，已在实现前排除）；这是带序数距离的观察更新，精度不声称可辨识真实噪声。主替换 `hard_observation` 固定posterior=onehot(g)，保留同样内容prior和观察loss，隔离是否需要软观察推断。辅助 `categorical_noise` 换无序4×4发射矩阵；`no_vlm` 不读grade，posterior=prior且去观察loss。

**M2：双向区间—内容分配骨干。** 两路投影到128维（视觉1024；音频+文本896），共享一个区间query/key映射。对每个区间j及模态m，在该区间覆盖的位置t计算 `A[j,t,m] = softmax_t(q_j · k_{t,m}/sqrt(128) + log(overlap[j,t]))`；区间外严格为0。q含M1的两路区间内容及序数观测。用同一A先把两路内容聚合到区间，再把另一模态的区间表示回送到本路token，回送权重由A在覆盖该token的区间中归一化。两路共享更新参数，一轮双向交换，不作全T×T注意力。主替换 `uniform_assignment` 使用相同真实交叠权重而非内容条件A，保留区间token/更新层与损失；因而测试的是内容条件位置分配，不宣称attention首创。

**M3：分配边缘化的联合读出。** 每个区间j与位置t的同一个共享小型MLP输入 `[updated_audio_t, updated_visual_t, interval_j, audio_t*visual_t]`，给出条件概率p(t,j)。最终位置概率为 `p_t = sum_j R[t,j] p(t,j)`，R由M2两模态A的均值按j归一化；即对潜在解释区间j边缘化。不是多个独立预测模型集成：所有项共享单一编码器和条件分类器，区间是输入变量，只有一个最终p_t。主替换 `additive_readout` 用同样内容与区间特征，但改成content_logit+加权interval_logit后sigmoid，取消条件交互与概率边缘化。

所有窗口包含正负裁定，不只选择VLM判阳窗口。窗口交叠包含训练重采样的Voronoi时间单元；部署仍在原I3D网格并用唯一评测器映射1fps。训练video BCE来自同一个p_t的top-ceil(T/16)均值，另对M1的**更新前边缘概率** `P(g|window content)=sum_z pi_j(z)*E_k(g|z)` 求观察NLL，K30/K4各自取均值后两尺度均值，与video BCE等权。不能用已读取g的posterior重构g伪造监督。`no_observation_loss`辅助臂去此项；M2/M3主替换保留观察头及损失。无平滑、EMA/双teacher、额外模型、按语料路由或额外稀疏/一致性正则。

主张边界：这是可证伪的完整方法假设，不声称序数编码、attention或混合分布数学首创。三个模块都必须按当前14(g)两语料三seed有效才作novelty主张；M1若不支持，不能用“编码更理论化”冒充贡献。提案review只按规则4四项STOP，并实际检索hateful video已有采用。

## 成本与协议

复用 `data/MLLM_scores/{HateMM,HateClipSeg}/*_segscoreK{30,4}_qwen.jsonl` 与原I3D/VGGish/BERT。新增VLM调用0；新视频34次VLM调用，与最初模块1相同，不是C7每视频120次。历史及部署VLM成本均如实披露。分配点积O(34T×128)，内容线性投影另需O(T H²)，不新增T²注意力、不抽新视频特征。M3第一线性层按拼接的内容项/区间项分别投影再相加（精确等价），避免对T×34对重复512→128矩阵乘；剩余非线性O(34TH)，推断可按区间循环累加控制显存。实际完整trial耗时在code review通过后按首trial测量，不能预先许诺吞吐。

同结构/损失/推断用于HateMM和HCS。预声明仅搜索lr log[1e-4,1e-3]、dropout {.1,.2,.3}、max_seqlen {150,200,300}；hidden128、单轮双向交换、topk_div16、loss等权固定，固定设计参数仍披露。50epoch、batch32、五crop；val(AP+ROC)/2选checkpoint，test(AP+ROC)/2选trial，within下限不改。首完整trial≤1h则20trial，否则5，冻结后不增减；先seed234两语料，均筛选通过才确认。所有长任务自动monitor；不做smoke或缩短训练。

待review确认：是否完整科研机制而非纯工程组合；来源是否已在hateful video采用。即便GO也不预断性能；后续code review重点检查时间覆盖、分配归一化/梯度、最终p与训练一致、等级语义和split isolation。

## 实现入口

`model.py` 为唯一候选网络；共享输入时间单元与可选观察在 `src/interval_observation_data.py`（train重采样、eval原网格），归一化仅统计train/crop0。`train.py` 调用现有完整50epoch训练协议和唯一评测器，`search.py` 调用现有固定预算TPE。正式启动命令 `bash experiments/20260906_interval_evidence_transport/launch/run_search.sh <corpus> 234`；运行主机和monitor在STATUS记录，首trial耗时及冻结预算完成后补本节。

实现细节：单轮双向消息使用同一个key和更新网络；MLP第一仿射层严格分解内容/区间两部分，推断按34个区间累加，避免T×34重复大矩阵乘及四维输出常驻。初始观察精度1，初始类别噪声臂与同一序数矩阵相同；probability floor1e-6、归一化std下限1e-4、分母下限1e-12均为固定数值参数。骨干没有EMA或独立teacher网络。

实际运行主机：HateMM=uoa-lab1/sc474397，HCS=uoa-lab3/sc474398；两机07:24启动50epoch正式trial，首epoch与validation输出正常，独立monitor首次RUNNING成功。首trial预算尚未实测时不宣称20/5已定，由search完成完整训练+val checkpoint+test后自动写各 `runs/20260906_interval_evidence_transport/<corpus>/seed234/budget.json`，再转录此处。没有新增特征/VLM抽取、预试跑或缩短训练。

首trial预算与输出：HCS trial0完整50epoch+val checkpoint+test耗时88.886395秒，冻结每seed20trial，来源 `runs/20260906_interval_evidence_transport/hateclipseg/seed234/budget.json`。该trial val选epoch1，test .556560/.541946/.510151（AP/ROC/within），within剪枝；不是搜索最终结果，不基于首trial缩减预算。原评测及summary已回传，val63/test79覆盖和50epoch/ckpt选择核对一致。HateMM首trial仍运行，预算以完成后的实测为准。

HateMM首完整trial耗时212.884198秒，同样固定每seed20trial，来源 `runs/20260906_interval_evidence_transport/hatemm/seed234/budget.json`。首trial0 val选epoch4，test .597334/.804517/.646860；完整50epoch、val109/test214覆盖及原评测已回传核对，仍非最终搜索结果。

## HCS seed234完整筛选与消融调度

2026-09-06 07:52搜索结束，无残留进程；20trial全部完整50epoch，15 COMPLETE/5 PRUNED，原输出和checkpoint均回传本机。最佳trial18的test **.605771/.589308/.547480**，val选epoch1，lr .0006240119276172279/dropout .1/max_seqlen300。来源 `runs/20260906_interval_evidence_transport/hateclipseg/seed234/trial18/metrics.json`；全部trial审计 `seed234/artifact_audit.json`，保存预测解析覆盖 `runs/20260906_interval_evidence_transport/error_analysis/hcs_seed234_all_trials.json`。仅validation排序参考trial16：.592270/.585958/.549114，不用于选trial或额外训练。

HCS单语料数值筛选通过，但弱于C1；不能据此说完整方法或三个模块有效。HateMM固定搜索仍进行，确认seed等待两语料均过筛。利用已锁定HCS配置独立运行三个预声明主替换 `hard_observation/uniform_assignment/additive_readout`，并接 `no_vlm`；每臂完整50epoch、独立val选checkpoint/test，都是seed234初步机制诊断，不能替代最终两语料三seed有效性要求。

实际消融主机uoa-lab3/sc474398，入口 `bash scripts/run_locked_ablations.sh 20260906_interval_evidence_transport hateclipseg 234 hard_observation uniform_assignment additive_readout no_vlm`，前三臂同时运行，之后no_vlm；输出 `runs/20260906_interval_evidence_transport/ablations/hateclipseg/seed234/`，独立完成monitor的位置只在STATUS维护。不改活动HateMM模型、不重复HCS旧搜索、无新增VLM或特征抽取。

## HCS seed234四臂结果与内部描述诊断

四臂于07:59通知完成，全部输出/checkpoint已回传；相同trial18超参、50epoch、val checkpoint选择、val63/test79及原预测ID/秒长度/finite均核对通过。来源 `runs/20260906_interval_evidence_transport/ablations/hateclipseg/seed234/artifact_audit.json`（含每臂metrics.json路径）及 `error_analysis/hcs_seed234_ablations.json`。

| 版本 | AP/ROC/within | full减该臂 AP/ROC |
|---|---|---|
| full | .605771/.589308/.547480 | — |
| hard_observation | .605742/.586075/.547270 | .000029/.003233 |
| uniform_assignment | .604569/.585902/.545370 | .001201/.003406 |
| additive_readout | .593375/.572121/.554614 | .012396/.017187 |
| no_vlm | .595643/.560195/.527893 | .010127/.029113 |

M1/M2两项pooled差均小于.005，目前不支持独立贡献；不能从一个seed裁定永久无效。M3及单一VLM整体有初步正向信号，仍不满足两语料三seed要求，no_vlm也不能代替M1软观察更新消融。完整模型及四臂均由val选中epoch1，不事后重选checkpoint或以此为由缩短50epoch。

为区分“算子没变化”与“变化无性能作用”，新增只读 `diagnose_assignment.py`，加载已选checkpoint，在全部79个HCS test视频crop0取原forward hooks；不读GT、不训练、不重算AP/AUC。命令 `python experiments/20260906_interval_evidence_transport/diagnose_assignment.py --run runs/20260906_interval_evidence_transport/hateclipseg/seed234/trial18 --out runs/20260906_interval_evidence_transport/error_analysis/hcs_seed234_assignment.json`。观察到输入grade的平均后验概率 .988679，MAP仅 .005585比例窗口不同于原grade，软观察在此checkpoint接近硬等级；观察更新相对prior的TV为 .379628，不是完全没有读取VLM。音频/文本与视觉的分配相对纯交叠分配平均TV为 .066401/.157809，说明不是数学上相同的均匀分配，但该变化尚未转化为清楚的pooled收益。以上为crop0内部描述，不是五crop新性能指标或真正噪声可辨识证据。

设计/执行影响：不盲目再加损失或更高VLM调用。追加提案已声明的 `no_observation_loss`、`categorical_noise`，同trial18配置、同seed234各完整50epoch，检查观察监督及序数约束是否相关；不是新增trial搜索或确认seed。两臂通过 `launch/run_hcs_aux.sh` 并行；共享launcher的可选链名只把日志/PID/completion分离到 `ablations/hateclipseg/seed234/auxiliary_chain/`，各新臂仍在原消融目录、旧四臂和完成标记均保留，monitor不复用旧完成状态。训练模型与活动HateMM代码不改。
