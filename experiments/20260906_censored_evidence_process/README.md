# 候选8：带噪窗口证据监督的局部事件强度学习

2026-09-06提案，独立proposal/code review均GO；06:31 NZST双机启动完整seed234搜索。HCS20trial已完整结束、全部within剪枝；HateMM仍按固定预算运行。尚未支持有效性/novelty。候选7已被用户因成本叫停，不恢复或换名续跑。

## 1. 出发点与成本

候选1完整三seed性能为HMM .657/.842/.646、HCS .699/.681/.553（AP/ROC/within），但部分模块贡献语料相关；C6 HCS train统计初始化与优化模型相当，提示读取VLM先验不等于新训练机制有效。依据：`runs/20260903_hier_evidence_mil/<corpus>/seed<seed>/study_summary.json`，`runs/20260905_latent_evidence_sequence/diagnostics/hcs_seed234_initialization/metrics.json`及C6八臂消融。它们是开发期test诊断，不进梯度训练或checkpoint选择。

本候选复用最初模块1的完整冻结Qwen K30/K4裁定缓存（每窗口一次、每视频34次已有观测），不使用C5的四次模态干预或C7的四次上下文干预。**当前新增VLM抽取=0；训练后部署新视频的VLM调用=0。** 只有train视频的Qwen裁定参与训练目标，val/test不读取VLM裁定做输入或推断。仍需已有I3D/VGGish/BERT基础特征，不能称端到端零预处理或未测量就声称实时。完整trial GPU时间按首次50epoch实测，不设额外试跑。

统一假设：视频是否有害是局部证据事件的并集；冻结VLM对一个窗口的判断也是有噪声的“窗口里至少有一个证据事件”，不是窗口内每秒都阳性。学习内容特征到局部强度的映射，使这些嵌套窗口观察与视频标签一致。避免推断时依赖VLM分数直接加成；整个方法围绕同一个局部强度过程。

## 2. 三模块

### M1：VLM裁定的带噪窗口事件观测

只读取train的K30/K4原始0–3裁定，沿原缓存约定≥2为1。每尺度一个可学习假阳率r_k和灵敏度q_k，约束0<r_k<q_k<1。r由train负视频各窗口的平均裁定率初始化（固定Beta(1,1)平滑），q初始化为r+(1-r)*.9；后续与同一内容模型共同学习。窗口真实事件概率P_W经单一噪声通道变为 `P(b_W=1)=r_k+(q_k-r_k)*P_W`，对缓存b_W求BCE。不将模型裁定当稠密真值，不读逐段GT。参数受单调约束但不声称学出了可辨识真实混淆率。

M1主替换`hard_observation`固定r=0/q=1，用同样窗口事件与同样缓存；辅助`no_vlm`去两尺度观测loss。前者只测带噪事件测量的作用，后者测train-only VLM整体作用；“用了VLM”本身不算novelty。

### M2：视频总量与局部位置分离的内容骨干

1920维I3D/VGGish/BERT输入，只用train/crop0全snippet统计归一化。线性投影hidden128，两个残差时间卷积块（kernel3、dilation1/2，GELU/dropout），有效token遮挡padding。在归一化视频时间[0,1]上，token代表相对区间宽度d_t；全视频按d_t加权均值经线性层产生总强度 `Lambda=softplus(a)`；逐位置产生分配logit r_t，`w=masked_softmax(r+log(d))`，区间质量 `m_t=Lambda*w_t`，密度 `lambda_t=m_t/d_t`。同一轻量内容骨干同时决定总量和位置，VLM不进forward。

M2主替换`unfactorized`使用同一内容骨干但逐位置独立softplus生成密度lambda，再乘d_t得到质量，取消总量/位置显式分离。只能归因该参数化整体，不称卷积或softmax首创。训练沿固定max_seqlen均匀采样，选中位置中点的Voronoi边界（首尾0/1）构成覆盖整段视频的区间；不是把未选中时间删掉。最终读出密度，不用随采样密度变化的区间质量作分数。

### M3：视频/嵌套窗口的删失事件似然

视频事件概率 `P_video=1-exp(-sum_t m_t)`；每K30/K4窗口事件概率 `P_W=1-exp(-sum_t overlap_length(t,W)*lambda_t)`。窗口与token区间均为归一化相对时间，交叠积分在每尺度质量守恒。窗口表示至少出现一个事件，不以top-k秒或窗口平均分直接代表事件。

`L=BCE(P_video,y)+mean_K30(BCE(P_b,b30))+mean_K4(BCE(P_b,b4))`，固定三项等权。因嵌套事件相关，这是多尺度复合似然(composite likelihood)，不是全部观察的精确联合似然。负视频y=0仍保留真实VLM误报供通道学习。不另加稀疏、平滑、时长先验、teacher ensemble或推断融合。全部目标只作用于同一密度lambda；test返回log(lambda)作为局部logit，经既有sigmoid得到lambda/(1+lambda)，共享五crop/1fps映射不变。该单调强度分数不随训练重采样区间宽度缩放，不声称真实有害概率已校准。

M3主替换`topk_event`将窗口并集概率换成与窗口有交叠位置的局部分数lambda/(1+lambda)的top-ceil(N/16)均值；视频监督、噪声通道、内容骨干均不变。辅助`fine_only`去K4观察，分析粗细约束，不把已有多尺度本身称创新。三模块最终均须两语料三seed满足现行14(g)；single-seed不够。

## 3. 固定协议与执行边界

两语料同结构/损失/推断；50epochs、batch32、5crop；只搜索lr log[1e-4,1e-3]、dropout {.1,.2,.3}、max_seqlen {150,200,300}。隐藏128、卷积核3、dilation1/2、固定等权、数值稳定下限是设计参数，固定不等于无超参数。首trial实测≤1h则20trial，否则5，validation选checkpoint，test(AP+ROC)/2选trial，within仍按现行门剪枝；validation选trial仅零额外训练附记。

Proposal review须实际检索noisy label observation、learning from aggregate labels、Poisson/censored-event MIL、factorized point-process localization及hateful video已有迁移；按规则4四项STOP，不以理论退化预先否决。Code review一次重点查时间重采样质量守恒、数值稳定、标签/裁定隔离、final score和消融。必要的时间对齐基础设施升入src，不跨实验import，不改唯一评测器。两语料seed234独立搜索可并行，长任务自动monitor。最强baseline+同样基础输入及去train-only VLM消融仍需按最终声明规则补齐。

## 4. 运行入口与当前准备

完整搜索入口：`bash experiments/20260906_censored_evidence_process/launch/run_search.sh <corpus> 234`；计划HateMM在uoa-lab1/sc474397、HCS在uoa-lab3/sc474398，输出 `runs/20260906_censored_evidence_process/<corpus>/seed234/`。只有code review GO及多机同步后启动，日志首行记录实际主机；PID和monitor只在STATUS维护。

已在本机核对固定cohort：HateMM train744/val109/test214（保留原test无GT视频427排除），HCS251/63/79。train两尺度裁定完整、等级可解析；观察噪声的负视频原始阳性比例为HateMM K30 .085501/K4 .165359、HCS .171875/.203125，初始化另用Beta(1,1)平滑。只用train标签与裁定，不用test/val统计拟合。

推理输入只含1920维基础内容和时间区间；VLM裁定虽通过训练batch的独立尾列送入loss，forward显式不读取该列。val/test尾列为零，不读取相应裁定。最终checkpoint严格由val pooled指标选择；原缓存初始生成成本仍应在论文披露，不能因当前复用而省略。

实际运行主机：HateMM=uoa-lab1/sc474397；HCS=uoa-lab3/sc474398。06:31启动正式50epoch trial，两机首trial epoch/validation输出均正常，独立monitor首次检查RUNNING。首trial完成前不预先宣称20/5预算，固定预算会由shared search写入budget.json，完成后转录本节。无新增抽取、无smoke、无缩短训练。

首完整trial实测：HCS69.634405秒、HateMM167.251563秒，均固定每seed20trial，来源 `runs/20260906_censored_evidence_process/<corpus>/seed234/budget.json`。不以单epoch时长替代完整trial计时，预算后续不增减。

首trial0完整50epoch且原始评测已回传：HMM .581852/.773846/.582817（AP/ROC/within，val选epoch1），HCS .602207/.588575/.508023（epoch2）。两者within均低于固定下限，按规则PRUNED；这不是搜索最优或完整方法结论。来源各corpus的 `seed234/trial0/metrics.json`、`summary.json`，val/test覆盖与50epoch已核对。搜索继续完整20trial，不基于首trial提前停。

## 5. HCS完整筛选与保存预测诊断

HCS搜索06:51 NZST结束，主进程及子进程均退出；20trial全部回传本机，全部完整50epoch、val63/test79视频覆盖及checkpoint选择通过 `runs/20260906_censored_evidence_process/hateclipseg/seed234/artifact_audit.json` 审计。20个均为PRUNED，没有合格best，不是任务异常。within范围 .496378–.519747，低于 .524。

忽略within、仅供诊断的最高test(AP+ROC)/2为trial7：**AP .604427 / ROC .589883 / within .510205**，val选epoch2。来源 `runs/20260906_censored_evidence_process/hateclipseg/seed234/trial7/metrics.json`。仅validation排序也选trial7，不增加训练；该结果不计作通过筛选。不追加HCS确认seed或消融；HateMM仍完成已冻结20trial，最终按规则9分流，不在活动训练期间归档代码。

已读上述20trial全部test预测与GT，以及C1 HCS seed234/trial8原预测作开发期诊断；没有重新训练、改checkpoint或重写评测。共享描述逻辑升入 `src/saved_prediction_diagnostics.py`，旧C5诊断入口复用；运行命令：

```bash
python scripts/analysis/describe_saved_predictions.py --corpus hateclipseg --run C8_trial7=runs/20260906_censored_evidence_process/hateclipseg/seed234/trial7 --run C1_trial8=runs/20260903_hier_evidence_mil/hateclipseg/seed234/trial8 --out runs/20260906_censored_evidence_process/error_analysis/hcs_seed234_saved_predictions.json
```

诊断产物在同目录 `hcs_seed234_saved_predictions.json`、`hcs_seed234_all_trials.json`，逐一记录输入路径。C8 trial7在67个混合标签视频中32个AUC<.5，C1为25个；原评测器的视频均分ROC分别 .762319/.807246，说明C8有视频级区分但局部排序弱，并非只有within一项的微小门槛落差（C1该trial pooled .695235/.679345）。C8 trial7视频内分数标准差均值 .082567，不是恒定输出；同视频阳性/背景均分差的macro均值仅 .001513。不同模型分数尺度不一致，分数差/方差不能单独作为跨模型性能判据。

全部20trial内分数标准差均值范围 .051638–.277939，噪声通道q-r最小 .713008，不能归因为常数输出或q=r通道塌缩。14/20个checkpoint在epoch1–2，但没有保存每epoch test预测，不能据此声称后期test退化或更换checkpoint规则。**目前只支持C8这套训练监督未取得有效局部排序，不足以证明所有train-only VLM都不可行，也不能把某个模块单独定罪。**

设计影响：下一候选优先保留原单次VLM局部证据作为推断输入，并研究内容与证据的局部交互；不把部署VLM=0作为用户硬要求，不恢复C7四次观察，不通过改within下限掩盖本轮性能落差。具体下一提案仍需独立review，未启动新候选或新抽取。
