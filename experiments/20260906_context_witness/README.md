# 候选7：上下文条件化的局部证据保留/删除学习

2026-09-06提案，独立proposal/code review均GO；首版已实现，评分接口bug在正式运行前修复，01:20 NZST已双机正式抽取，不声称novelty已成立。目标仍为两语料性能、三个模块各有有效性支持、统一范式、尽少方法超参数。

## 1. 为什么换方向

C6 HateMM完整20trial全部within剪枝（最高.606444<.632），HCS优化模型 `.6908/.6649/.5803` 与train统计初始化 `.6924/.6666/.5843` 相当。HCS静态转移不差，事件训练替换top-k收益不足.01，不能把现有VLM分数的高信息量算成新骨干/新训练贡献。来源 `runs/20260905_latent_evidence_sequence/hatemm/seed234/artifact_audit.json`、`diagnostics/hcs_seed234_initialization/artifact_audit.json`、`ablations/hateclipseg/seed234/artifact_audit.json`。test仅用于developmental设计诊断，不参与梯度训练。

同时原抽取prompt明确排除“merely offensive but not hateful”，而共享HCS标签定义为五维有害类别并集。这是可确认的任务语义差别，不是数据泄漏或证实全部失败原因。新输入采用同一固定多属性观测向量，不按语料手写路由；下游由各自train视频标签学习目标映射，禁止读取逐段标签训练。

本候选不再只拟合原始裁定密度。范式假设是：**在上下文中测量当前局部窗口的证据，再训练一个既能保留视频判断依据、又能删除该依据的局部选择器**。新观测、时序残差表示与共享分类器的保留/删除目标共同服务局部选择，而非三个互不相干的输出融合。

## 2. M1：目标窗口与上下文的四种观测

同一冻结Qwen2.5-VL-7B-Instruct，固定K=30，每目标窗4帧，前/后邻窗各2帧（视频边缘缺邻窗明确标记，不跨视频）。ASR同样标记before/target/after，来自已有K30转录，无GT时间标签。只问TARGET区域；邻窗仅用作解释上下文，不将邻窗本身有害当成target阳性。

六个固定观测属性，所有语料完全相同：protected-attribute attack、targeted insult/degradation、sexual harmful content、violence/threat/incitement、other harmful conduct、quotation/condemnation/neutral reporting rather than endorsement。它们是预设语义测量，不直接当作某语料真值或六个teacher。每次同一个模型给六项Yes/No条件log-odds，保留raw logits和binary entropy，不硬阈值、不给其声称已校准概率。

四种原始内容输入：(target,context)、(target,empty_context)、(empty_target,context)、(empty_target,empty_context)。视频缺失部分用固定黑帧并在文本明确说明缺失，ASR对应部分同时清空；位置布局/问题保持一致。缓存保存完整4×6观测与缺失标记。原来C5是“视觉/ASR”干预，此处是“目标时间窗/解释上下文”干预，且语义范围不同，不能复用旧logits冒充新观测。

下游输入按六属性分组依次为 `L_target_only`、`L_context_only`、`L_full−L_context_only`（context条件的target依赖）、`L_full−L_target_only−L_context_only+L_empty`（交互）及`H_full`，共30维。前24维是四路raw logits的可逆重参数化，不声称真实因果或新增信息；保留context-only避免原提案24维相对raw_four丢失信息的混杂。M1主对照 `target_only` 仅保留同一新问题的target-only logits/entropy，并使用同维零填充；辅助 `raw_four` 用四路原始logits与同一H_full（30维），同输入不差分。多属性语义修正本身不算novelty，必须隔离上下文依赖测量的作用。

抽取为label-free独立准备，cache `data/context_witness/<corpus>/K30/`，PROVENANCE记录完整问题、模型、输入路径/缺失、采样位置、主机、生成命令。若一次生成六项回答，必须强制可解析格式、逐位置记录实际raw Yes/No logits并明确它们条件于前面答案，不假称独立六次边际推断。不同测量方式不得混cache。

## 3. M2：预测邻域内容后的局部残差表示

输入仍为统一I3D/VGGish/BERT内容向量x_t，按train统计逐维标准化。两条128维单向GRU分别读左侧与右侧序列；对位置t用左t−1状态和右t+1状态，不允许重建器直接看到x_t。拼接两侧状态经线性层重建标准化冻结内容 `xhat_t`，最小化逐有效token/维度MSE。边界缺邻域用零状态，padding不参与。重建目标是冻结输入，不是可一起塌缩的可学习投影。

局部表示包含内容投影、残差投影 `x_t−xhat_t`、新VLM观测投影；MLP输出selector `q_t∈(0,1)`。残差不自动等于有害证据：它只表示不能由邻域预测的内容，必须由视频监督及VLM观测学会区分有害与普通变化。M2主对照 `no_residual` 去selector的残差输入和重建损失，保留相同内容/VLM与保留删除训练；辅助 `visible_reconstruction` 允许重建器见x_t，检查留一位置结构是否必要。不把GRU/MSE本身称novel。

## 4. M3：同一分类器的证据保留/删除训练

一个共同训练的共享视频分类器f，对内容/VLM融合token做加权均值后输出视频概率。三次调用权重分别为1（全视频）、q（保留）、1−q（删除）；三次完全同一参数，不是独立模型ensemble，不输入显式“全/保留/删除”模式标志。加权均值除以有效权重总和（数值下限1e-6），避免直接用未归一化总量识别mask大小；不得声称已排除全部缺失模式捷径。

每视频目标：

`L = BCE(f(x;1),y) + BCE(f(x;q),y) + BCE(f(x;1−q),0) + mean(q) + reconstruction_MSE`。

前两项要求保留原判断，第三项是假设被删除的证据足以移除有害判定，稀疏项选择尽少证据，重建项约束局部残差。全部固定等权，各项按有效元素归一化；不用GT片段或test标签。负视频y=0时三次都应负。删除目标0是方法假设，不是已知真实counterfactual标签；可能存在mask/分类器共同作弊或选择不完整，交完整test消融判断，不在实现前假定成立。

最终局部分数仅为同一forward的q_t，经既有5crop/1fps映射评测；不把分类器概率与q相乘、不加VLM概率、不做推理平滑/阈值校准。M3主对照 `no_deletion` 仅移除删除项；辅助 `no_sparsity`。q必须通过保留/删除路径获得实际梯度，不能只把已拟合的视频分数解释为定位结果。

## 5. 搜索、消融和来源审查

拟仅搜索lr log[1e-4,1e-3]、dropout {.1,.2,.3}、max_seqlen {150,200,300}；固定50epoch、batch32、hidden128、K30、target4/context4帧、六属性、五项等权、数值下限。固定不等于无超参数；比C6更多训练项必须如实披露，不称已达最简。两语料同架构/损失/推断，允许由train学到不同参数，不允许手写语料分支。

规则4独立审查需实际检索“rationale sufficiency/comprehensiveness、counterfactual erasure、contextual anomaly reconstruction”以及hateful video相邻工作。这些是已有通用方法，迁移源若已用于hateful video则按规则STOP；不能把只改prompt、增加属性或GRU当完整创新。也需对照项目旧coalition/witness/erasure失败方向，但不恢复已撤销的旧前置门。

Code review一次，只查影响结论的bug；通过后完整两语料seed234 Optuna，首trial实测决定20/5，validation选checkpoint、test(AP+ROC)/2选trial，within沿现行门。未过筛不提前补确认seed；三模块最终都须两语料三seed满足14(g)。保留三主消融及target-only/初始统计等必要参照，不靠零散seed挑同向。最强baseline+同输入仍是最终声明缺口。

## 6. 实现与运行

抽取：`bash experiments/20260906_context_witness/launch/run_extract.sh <hatemm|hateclipseg>`；计划HateMM在uoa-lab1/sc474397、HCS在uoa-lab3/sc474398，启动前实时核验。每窗口四模式顺序调用同一冻结模型，强制11token（六个Yes/No加五个换行），读取生成器raw logits，不读受grammar掩码的scores。六项条件于此前生成答案。源码 `scripts/analysis/extract_context_witness.py`，严格解析与30维映射 `src/context_witness.py`。

完整搜索：`bash experiments/20260906_context_witness/launch/run_search.sh <corpus> 234`。输入解析/覆盖率通过且code review GO后启动；输出 `runs/20260906_context_witness/<corpus>/seed234/`。首完整trial前未决定20/5，不额外计数试跑。训练归一化用train的完整snippet序列、visual crop0，五crop训练/评测沿既有共享实现。GT、split和统一评测器不改。仅将C5/C6重复cohort校验与抽取ID/ASR读取升入共享代码，不重跑旧结果。

M2主消融只支持“残差输入＋重建训练”整体；不能单独归因留一位置。三分类调用复用同一次dropout后的token特征，分类器完全共享，防止给不同视图额外引入随机差别。

VLM视频processor沿冻结模型默认2fps编码这8张采样帧；这是归一化采样序列，不是原视频真实秒时间。prompt用frame positions、cache用相对窗口，最终定位仍通过共享snippet/1fps映射，不能声称VLM接收了真实秒时间。

正式抽取已启动：HateMM运行主机uoa-lab1/sc474397，HCS运行主机uoa-lab3/sc474398；全部固定split IDs分别1068/393，输入文件存在、环境已核对。各输出 `runs/20260906_context_witness/extract_<corpus>/`，本机同run下monitor独立后台通知当前会话；具体PID和当前状态只在STATUS维护。新方法尚无训练结果，首trial预算尚未产生。

01:21首次正式抽取在两机均于首窗口vision SDPA发生OOM（25.16GiB已占用，需再分配7.91GiB），进程退出、0视频JSON，无可用缓存。已诊断并将四模式batch4改为顺序batch1，问题/帧数/分辨率/模型/六答案条件协议保持不变。失败日志与原config保留；修复后正式输出目录用 `extract_<corpus>_serial`，入口第二参数指定目录名，不将失败当作方法结果或搜索trial。
