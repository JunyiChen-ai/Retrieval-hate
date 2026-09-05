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

两语料各三seed完整20trial已全部结束、回传审计，均通过数值确认；HCS三核心模块未通过三seed有效性要求，不能作为最终三模块方法。保留为性能/诊断参照，不追加旧消融，继续候选6。当前运行与下一步只见 `research-wiki/STATUS.md`；下文保留本轮详细记录。

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

8臂已全部完成/回传/核验，来源该输出目录`artifact_audit.json`及各臂`metrics.json`。AP/ROC/within：raw_verdict `.614/.603/.519`，ordinary_attention `.591/.576/.550`，additive_fusion `.632/.621/.573`，full_input_only `.509/.455/.442`，four_logits `.531/.498/.466`，no_interaction `.650/.637/.531`，dempster_fusion `.641/.628/.582`，no_block `.611/.613/.577`。三个核心替换的pooled下降仅是单seed初步支持；交互项的AP下降仅.00463，不支持必要性。四logits与差分可逆，当前差距体现有限训练下表示/优化差异，不是增加信息的证明。不得据此宣称三模块novelty已确认。

为复用空闲lab3，准备HateMM剩余奇数分片视频（14:33快照256个、约1.38GB）；仅传未完成项，不重算缓存。准备脚本首次因本机缺`non_hate_video_585.mp4`别名退出，原视频存在且与lab1大小一致，已补别名并恢复（未进入GPU任务）。完成准备后才停止旧全列表抽取，回传最新缓存并复制lab3，再按固定排序奇偶分片0/1各534个ID续跑，使用同一v2代码和已有resume校验。两片完成后合并审计1068个视频，再训练；不改变方法、split或输入协议。

HateMM两片均534/534正常结束。2026-09-05 19:05处理通知时已回传合并1068视频/2136文件，严格输入审计通过，合并PROVENANCE保留初始lab1生成与后续两机分片来源。19:06在lab1用`launch/run_hatemm_lab1.sh`启动seed234完整搜索；输出 `runs/20260905_interventional_evidence/hatemm/seed234/`。预算待首trial完整耗时冻结；完成前不提前补确认seed。

HateMM seed234完整搜索已结束并回传：20trial（19 COMPLETE/1 within剪枝），首trial296.590秒，预算20。按test选trial16，test AP/ROC/within=`.623969167/.849230124/.671588356`，validation选epoch4；参考性的validation排序会选trial17，test=`.614934171/.843600194/.671201480`。来源 `runs/20260905_interventional_evidence/hatemm/seed234/artifact_audit.json` 及对应原始metrics。两语料均通过规则8单seed筛选。

2026-09-05 20:37补确认：lab1并行HateMM seed2025/3407，lab3并行HateClipSeg seed2025/3407；每项继承各语料seed234的20trial预算，全部使用`launch/run_search.sh <corpus> <seed>`。每项独立输出/monitor，未重跑seed234。完成后仍须三seed均值/std及核心模块消融，不凭单seed结果宣布novelty。

HateClipSeg三个seed现均完整20trial（60 COMPLETE）并已回传审计。seed2025/3407按test均选trial12，test分别 `.649604234/.633045705/.553784217`、`.657898852/.644432031/.578979123`；三seed均值 `.654090358/.638275481/.557699900`，样本std `.004188622/.005749460/.019616707`。依据 `runs/20260905_interventional_evidence/hateclipseg/confirmation_summary.json`：HCS数值确认门通过，不等同整体目标完成。HateMM确认仍在进行。

21:54前在lab3启动HCS确认seed2025/3407各8臂消融，均锁定其自身trial12配置。共16次完整训练、每批3任务，入口`launch/run_hcs_confirmation_ablations_lab3.sh`，监控输出`runs/20260905_interventional_evidence/ablations/hateclipseg/confirmation_chain/`；不重跑seed234。结束后合并三seed按规则14(g)判断模块贡献。

## 7. HCS三seed模块结论与HateMM接续

2026-09-05 22:22确认消融链结束；实际进程组已退出，新增16臂全部回传，连同seed234共24臂均通过完整50epoch、锁定超参、validation checkpoint和两split评测一致性/覆盖率审计。聚合脚本 `scripts/analysis/summarize_module_ablations.py` 只读取统一评测器输出，不重算帧指标；来源 `runs/20260905_interventional_evidence/ablations/hateclipseg/three_seed_summary.json`。

| 替换/移除臂 | 完整模型减消融AP，seed234 / 2025 / 3407 | 平均AP差 | HCS规则14(g) |
|---|---|---|---|
| raw_verdict | +.040485 / +.055432 / −.004473 | +.030481 | 不满足，每seed同向失败 |
| ordinary_attention | +.063331 / +.015249 / −.003086 | +.025165 | 不满足，每seed同向失败 |
| additive_fusion | +.022521 / +.005442 / −.002368 | +.008532 | 不满足，均值及同向失败 |
| four_logits | +.123367 / +.096071 / +.092987 | +.104142 | 本语料满足，非信息增益证明 |
| no_interaction | +.004629 / −.005267 / +.013816 | +.004393 | 不满足 |
| dempster_fusion | +.014030 / +.009382 / −.000216 | +.007732 | 不满足 |
| no_block | +.044163 / +.057148 / +.033557 | +.044956 | 本语料满足，但为已有监督 |

三个核心臂在seed3407的ROC也全部反向，不能改用ROC补足。`full_input_only`在seed3407为AP .697629/ROC .667300，也优于完整模型；其余两seed较差，所以平均增益不证明逐seed稳定。差分与四logits可逆，`four_logits`差距只反映当前训练下表示/优化差异；不能据此宣称增加信息或交互机制已验证。各seed采用各自搜索最优超参，seed与配置共同变化，不能从三点观察直接诊断为随机性或lr原因。

结论：HCS性能确认仍成立，但三模块novelty主张当前不成立。不得重复旧消融寻找偶然同向。为定位是否跨语料失效，在已就绪且空闲的lab3于22:27启动HateMM seed234八臂完整诊断，锁定其test最优trial16，入口 `launch/run_module_ablations.sh hatemm 234`；主机sc474398，输出 `runs/20260905_interventional_evidence/ablations/hatemm/seed234/`，独立monitor已首次检查RUNNING。lab1两确认搜索不变，等待完整结果后做跨语料错误分析与方法分流。

## 8. HateMM诊断与跨语料错误分析

23:04通知后确认lab3进程退出，8臂全部回传：每臂50epoch、validation选checkpoint、trial16超参、val109/test214覆盖及原始评测一致性均通过 `artifact_audit.json` 核验。完整模型 AP/ROC/within `.623969/.849230/.671588`；原裁定替换 `.463147/.744635/.607294`、普通注意力 `.598809/.819538/.670525`、加法融合 `.595570/.823837/.671072`。完整输入单路 `.607594/.829128/.668801`、四logits `.547277/.802005/.665957`、去交互 `.603688/.825907/.670546`、Dempster `.597849/.823961/.675874`、去块监督 `.545509/.780297/.691070`。来源 `runs/20260905_interventional_evidence/ablations/hatemm/seed234/<arm>/metrics.json`。这只能支持HateMM该seed的pooled贡献，不能修复HCS三seed失败。

随后使用 `scripts/analysis/diagnose_interventional_errors.py` 读取两语料已完成主模型/消融的 `scores_test.jsonl`、`metrics.json`、`summary.json`，固定test GT及split/video标签；输出 `runs/20260905_interventional_evidence/error_analysis/saved_prediction_diagnostics.json`。这是规则10允许的developmental error analysis，不重算AP/AUC、不训练、不用于checkpoint选择。

- 训练正视频比例：HCS219/251，HateMM298/744。两语料的视频级监督分布差异大，但单凭比例不能证明失效原因。
- 完整模型在HCS三seed的正秒/正视频背景秒均分分别 `.7933/.7699`、`.7928/.7664`、`.6641/.6146`；HateMM seed234为 `.4686/.3196`。这些是描述性均分、不是新增评测门。HCS含两类秒的视频内时间标准差均值仅 `.0223/.0294/.0487`，HateMM为 `.1028`；绝对分差受分数尺度影响，不能单独当作排序性能。
- 从统一评测器已有逐视频AUC读取，HCS完整模型有32/30/21个混合视频AUC低于.5（共67个），HateMM为18个。HCS存在大量局部排序反向的视频；不能凭pooled过门称定位机制已解决。
- HCS完整模型选中epoch1/2/5，训练loss从首epoch至末epoch分别 `1.086→.365`、`1.206→.409`、`1.412→.565`；各自validation AP从最佳 `.668/.681/.689` 降到末epoch `.566/.581/.645`。HateMM同样有后期泛化下降。这支持研究训练监督与定位目标的差异，不证明某个模块是原因，更不授权改validation checkpoint协议。
- 设计决策：停止为当前三模块主张追加无必要旧消融；保留正在运行的固定预算确认搜索。下一修订优先检验正视频内部局部正负证据如何进入训练，必须超出已有块MIL/单纯类别重加权/更换融合算子。当前尚无可直接启动的新提案，不因GPU空闲重复实验。旧条件HMM比例目标在 `experiments/20260905_verdict_conditioned_density/README.md` 已失败，不直接复用该比例作新监督。

## 9. 最终数值确认与去向

23:34通知后核验lab1两个确认进程均退出，两seed各20trial完整回传、50epoch/validation checkpoint/原始评测/覆盖审计通过，均20 COMPLETE。seed2025/3407按test均选trial13，AP/ROC/within分别 `.631655699/.839611941/.652235706`、`.638296542/.848411267/.655993148`。HateMM三seed均值 `.631307136/.845751111/.659939070`，样本std `.007170045/.005332418/.010262017`；pooled领先固定门 `.058307/.038751`，超过所需std幅度 `.033/.0194`，within高于.632。来源 `runs/20260905_interventional_evidence/hatemm/confirmation_summary.json` 及其中各trial原始metrics路径。

HCS最终均值/std见第6节。两语料共120trial（119 COMPLETE/1 within剪枝）全量回传核验。仅按validation排序trial的零额外训练参考：HateMM三seed均值 `.613349602/.838257012/.663449771`，HCS `.651619146/.635777955/.556582865`；不用于选方法或方向。

第8条数值确认已全过，不等于第14条最终声明全过：HCS三核心机制消融不满足14(g)，最强baseline+同输入亦未齐。停止为当前三模块主张追加旧实验，保留本轮作为性能/诊断参照；下一候选为 `experiments/20260905_latent_evidence_sequence/`，视频标签约束的局部证据状态模型，独立proposal review已GO，待code review后正式实验。旧输入可直接复用，无需再抽取。
