# 候选 5：基于干预证据的弱监督定位（提案，尚未实现/验证）

日期：2026-09-05。起点：候选1的裁定输入/块级MIL有效，但硬裁定近乎二值、HMM可靠性在HateMM失配；候选4最终结构收益不能迁移，见 `runs/20260904_null_token_cma/rev1/artifact_audit.json`。目标是 VLM、骨干、融合都对同一个问题作贡献：区分“内容支持的仇恨证据”和“模型/上下文造成的高置信裁定”，不继续添加全局证据到注意力残差。

## 1. 统一机制假设

从同一冻结 VLM 的受控输入干预产生证据分解，骨干学习各秒与这些证据的对应关系，融合显式保留证据冲突与不确定性。整体主张候选是 **interventional evidence learning**，不是三个模块的新名字；只有完整方法和三个替换消融均满足研究规则，才能主张新范式。文献尚待独立规则4审查，以下机制有效性均为待检验假设。

### 模块1：内容干预的VLM证据抽取

保持 Qwen2.5-VL-7B、K30/K4、各窗4帧与窗ASR不变。每窗使用同一模型、同一固定二选一任务，读取 Yes/No 下一token的条件对数概率，分别输入：(a)帧+ASR，(b)真实帧+空ASR，(c)空白帧+真实ASR，(d)空白帧+空ASR。空白帧保留图像尺寸/数量与位置，文本缺失明确标注。

设四个log-odds为 L_av、L_v、L_a、L_0；导出基线 L_0、视觉证据 d_v=L_v−L_0、语音证据 d_a=L_a−L_0、交互证据 d_av=L_av−L_v−L_a+L_0，以及四路二分类熵。**保留全部通道供训练，不把差值直接当最终分数，不截断负交互。** 干预改变内容，只作模型依赖性观测，不声称随机因果识别或保持标签不变。对原始裁定的替换臂 `raw_verdict` 保留相同维度，使用原K30/K4裁定与零交互/不确定性。

来源动机：VCD通过原图/扰动图条件分布对比分析模型依赖（[CVPR2024](https://openaccess.thecvf.com/content/CVPR2024/html/Leng_Mitigating_Object_Hallucinations_in_Large_Vision-Language_Models_through_Visual_Contrastive_CVPR_2024_paper.html)）。这里不进行contrastive decoding，不生成纠正答案；干预证据服务于后续监督学习。与项目旧coalition/dividend方案的区别待review核验：不是聚合多个模态模型预测，而是同一VLM输入依赖性观测及显式交互通道。

### 模块2：证据对应的双向关系骨干

I3D/VGGish/BERT投影为内容token；干预证据投影为局部证据token。每秒 query 对视频内证据token做交叉注意力，用同一可学习打分器产生支持权重 softmax(qk) 与反证权重 softmax(−qk)，分别读出内容value；输出拼接当前内容、支持内容、反证内容及其差，经共享MLP产生二类非负证据。padding全部屏蔽。正反读出共享参数，不用两个独立模型；只在当前视频内匹配，无跨split memory。

目的：保留支持与反证的区别，而非视频级上下文广播。对照 `ordinary_attention` 将双向读出替换成单向普通跨注意力（相同输入、隐藏维度、头与训练）；`no_interaction` 只去模块1交互通道，用于区分干预主效应与交互效应。注意力参数均训练，不另设温度/门阈值。

### 模块3：冲突保留的证据融合

同一个联合网络把VLM证据token与骨干输出分别映射成二类非负 evidence，形成Dirichlet意见：alpha=e+1，b=e/sum(alpha)，u=2/sum(alpha)。使用 **Yager型合取冲突转未知**：同类支持与未知交叉项进入对应 b，冲突量进入 u；再由 p=b+u/2 得到最终逐秒概率。融合在训练forward与bag监督内执行，无测试后处理、无独立teacher模型聚合。与Dempster归一化不同，冲突不放大置信度；小样本固定先验为1，不搜索证据温度或先验缩放。

训练只保留视频级 top-k BCE + 裁定块级MIL（块软标签沿用train拟合HMM，以便单独检验新forward，HMM只产生训练块标签，不作推理先验）。两项分别按有效视频/块归一化、等权相加；无EMA、CMAL或多损失权重搜索。对照 `additive_fusion` 使用两路证据概率的logit加法、可学习一个全局尺度；相同损失/输入/骨干。来源：[TMC ICLR2021](https://openreview.net/pdf?id=OOsR8BzCnl5)，具体冲突规则需查原始文献。

## 2. 搜索与证伪

两语料同一架构/代码/损失/抽取协议。拟搜索仅 lr∈[1e−4,1e−3]（log）、dropout∈{.1,.2,.3}、max_seqlen∈{150,200,300}，属于通用优化参数；不搜索方法损失权重、融合温度、分支开关、证据阈值。隐藏维度128、4头、top-k divisor16、batch32、50epoch固定，必须如实列出，不能称“无超参数”。抽取一次可复用；正式预算按规则7首trial耗时固定20或5，不改现行test目标/within约束。

每语料先seed234完整搜索；过筛后seed2025/3407搜索可跨GPU并行。完成后按各seed最优配置，`raw_verdict`、`ordinary_attention`、`additive_fusion` 三个模块替换臂各三seed两语料；还报 `no_interaction`、`no_block`，以及最强baseline+相同干预输入（规则14f）。各模块都须替换后AP或ROC三seed均值降≥.01且每seed都降，两语料同时满足；不能用“去掉整个VLM”代替模块1相对原输入的novelty证据。

预期：HateMM受益于区分仅ASR与视觉交互导致的假阳性，HateClipSeg不能只提高视频间分数而损害within下限；与候选1的完整搜索数字同时报告。如果只有某个模块有效，不满足用户三模块目标，继续按规则分流。新抽取缓存由专用脚本写 `data/interventional_evidence/`，附PROVENANCE；训练只读。

## 3. 状态

独立[规则4 review](REVIEW_RULE4.md)裁定 GO；进入实现与输入准备，尚未训练。候选4已归档，原输出全部保留。

## 4. 评审落实（覆盖上文提案中的未核验表述）

- 旧coalition也是共享模型的masked forward，不以单模型作为区别。这里改变的是冻结VLM原始输入观测、保留有符号量、学习下游时间对应关系，而非直接由dividend重构最终分数。
- 四路差分是可逆线性变换，不增加信息。新增 `full_input_only`（全输入连续logit/熵，其余置零）、`four_logits`（四原始logits及熵）对照。`no_interaction` 同时去交互通道与全输入熵，避免后者泄露被移除的交互量；其余主效应及对应熵保留。
- 模块2严格称正关联/负关联读出；负关联不自动等于语义反证。先将两粒度证据映射到相同时间网格，再作为key，与同索引content value配对。
- 模块3适配[Yager 1987](https://doi.org/10.1016/0020-0255(87)90007-7)：`C=b1[0]*b2[1]+b1[1]*b2[0]`，`b[k]=b1[k]*b2[k]+b1[k]*u2+u1*b2[k]`，`u=u1*u2+C`，`p[k]=b[k]+u/2`。不除以`1-C`，不声称相关分支独立或概率已校准。补 `dempster_fusion` 对照。
- RAMF/MARS/MATCH/CLARA已占用宽泛的证据推理、VLM引导及融合主张，不能以三段式命名声称首次范式。具体差异与有效性待实验。
- 输入抽取固定max_pixels=151200、每窗4帧、Yes/No单token条件分布、空白RGB=(0,0,0)；所有固定设计参数与三维搜索空间分别报告。
- 启动检查发现HateMM validation的 `non_hate_video_559`、`non_hate_video_585` 无已有ASR（K30/K4），原裁定脚本也是空转录回退。新抽取保留样本，提示为transcript absent并在`input_coverage.json`及各视频输出显式标记，不声称已观测无语音。首次抽取在任何VLM推理前退出；修复后恢复，不评价方法。

## 5. 实现与运行记录

训练入口 `train.py`，搜索入口 `search.py`；共享搜索协议 `src/fixed_optuna_protocol.py`，完整50epoch、Adam及cosine T_max=50、5crop，与提案固定参数一致。核心替换臂及full_input_only/four_logits/no_interaction/no_block/dempster_fusion均已实现；最强baseline+同输入对照尚未实现，不能宣布整体目标完成。冻结VLM输入抽取依规则5先行，训练须等唯一[code review](REVIEW_RULE6.md)修复确认与输入完成。

2026-09-05 code review识别Qwen默认repetition_penalty=1.05，而v1读取generate的processed scores。两机v1抽取已人工停止，已有缓存移到 `data/interventional_evidence_v1/<corpus>/` 保留，旧日志保留；不作为方法失败。v2改读`output_logits=True`的原始`out.logits`，新版本字符串与训练端严格检查防止混用，输出重新写 `data/interventional_evidence/`，运行目录 `runs/20260905_interventional_evidence/extract_<corpus>_v2/`。旧完成monitor随任务停用，新任务重新绑定独立monitor。

固定评测队列：HateMM train744/val109/test214（原test split215中`hate_video_427`无既有GT，沿用baseline排除）；HateClipSeg251/63/79。其它缺GT/缺特征均拒绝，不静默评测子集。补seed继承seed234的冻结trial预算，不按各机耗时改变。

lab3准备覆盖核验：393个视频头可解析、两粒度ASR全覆盖、5个模型分片可解析。原视频混有webm，以 `scripts/prepare_hcs_video_links.py` 创建规范mp4别名，不转码、不删除原视频。依据：`runs/20260905_interventional_evidence/prepare_lab3/coverage.json`。

HateClipSeg于2026-09-05 13:36在lab3启动seed234正式搜索，入口`launch/run_hateclipseg_lab3.sh`；启动前393视频/786文件完整v2审计通过，输出已回传本机。首trial完整耗时118.659秒，预算冻结20 trial，确认seed继承；依据 `runs/20260905_interventional_evidence/hateclipseg/seed234/budget.json`，不根据中途epoch指标作方向决定。

HateMM于13:48因lab1原视频`hate_video_95.mp4`截断退出。已回传422个完整视频的v2缓存并解析通过。逐文件大小对照仅此文件不同（远端37,537,689字节，本机79,748,606字节）；完整本机副本及补传副本均用正式120帧采样器成功解码。远端损坏副本保留在`runs/20260905_interventional_evidence/input_repair/hate_video_95.truncated.mp4`，修复原路径后恢复相同v2任务，自动跳过已验证输出，不重算422个视频。不修改输入协议或抽取代码；新monitor位于`extract_hatemm_v2/monitor_resume1/`。

运行命令：
```bash
python experiments/20260905_interventional_evidence/search.py --corpus hatemm --seed 234 --out-root runs/20260905_interventional_evidence
python experiments/20260905_interventional_evidence/search.py --corpus hateclipseg --seed 234 --out-root runs/20260905_interventional_evidence
```

## 6. HateClipSeg seed234搜索与下一步

20/20 trial均COMPLETE，全部50epoch；checkpoint选择、超参、原始评测和完整覆盖已核验，来源 `runs/20260905_interventional_evidence/hateclipseg/seed234/artifact_audit.json`。

- 开发期按test选trial18，AP/ROC/within=`.654767988/.637348708/.540336361`，validation选epoch1。通过本语料固定单seed门，但低于候选1；不是两语料确认SOTA。
- 仅按validation选trial4，其test=`.650030751/.632801129/.540282000`。
- HateMM尚未完成，暂不启动seed2025/3407。为检验三模块目标，锁定trial18的lr=.0008097806125316698/dropout=.3/max_seqlen=150，先跑已评审的8个seed234诊断臂，三任务一组并行，每臂仍完整50epoch/validation选checkpoint。来源配置固定为`trial18/hparams.json`；不再用消融结果改本轮搜索空间。
- 启动脚本 `launch/run_hcs_ablations_seed234_lab3.sh`，输出 `runs/20260905_interventional_evidence/ablations/hateclipseg/seed234/`。这些单seed诊断不替代规则14(g)的两语料三seed验证。
