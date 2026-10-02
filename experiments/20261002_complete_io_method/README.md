# 最新完整组合：inside–outside + 复制似然 + 自动停止

截至2026-10-02；用户指令：**“三个数据集上跑一遍最新完整的给我汇报”**。三语料为HateMM、HateClipSeg和DeHate；前两者是主数据，DeHate保持external validation。实现接入完成，[一次独立集成code review：PASS](../../docs/reviews/20261002_complete_io_method_code.md)；代码同步和远端完整输入核验后启动。Pursuing Goal保持paused。

## 1. 固定方法与本次范围

同一套已开发部件在三语料组合运行，不提出新的novelty候选、不按语料切换骨干：

- 已保留初版inside–outside backbone，`io_outside=true, io_merge=gated`，秒级/节点先验、原视频头与零膨胀链、原锚定答案模型和训练损失。
- 单问首token软答案，`soft_both`，训练答案无标注分位数8档；同一份I3D/VGGish/BERT与VLM缓存。
- 复制似然：负例训练视频父子答案一致率估计各长度桶复制概率，`copy_mode=both`，同时进入EIG与答案似然。估计不使用validation/test答案或定位GT。
- 自动停止锁定已有 `qmixSG_rt10`：逐秒logit变化总量、秒级排序变化和视频后验熵的validation分位数组合，乘`.9**k`；validation无标注平均8次选择阈值，不重新挑规则、松弛率或预算。没有额外最少调用次数，轨迹最大32次。
- 主结果为该完整组合的自动停止；同一复制设置的固定0/8/32次同时报告，以固定8次直接量化自动停止的代价。AP/ROC/within三项均为主指标，报告三seed均值及总体标准差、逐seed结果、平均调用数及正负例调用数。

这次是已开发方法的完整组合验证。初版的proposal/code review与novelty复查见 `docs/reviews/20261001_inside_outside_backbone_{proposal,code}.md`、`docs/reviews/20261002_inside_outside_backbone_novelty_recheck.md`；复制与停止定义来自原r5 README第17.2/17.4节。没有新的机制提案需要重新泛化proposal review；本次安排一次独立code review核对集成正确性。

## 2. 训练与模型选择（开跑前锁定）

复制似然与停止改变的是推断，不改变现有训练损失；因此不为相同训练重复消耗预算。

- HateMM/HateClipSeg复用初版各seed234/2025/3407已完整20-trial×50 epoch搜索。每trial validation选checkpoint；source `runs/20261001_inside_outside_backbone/<corpus>/seed<seed>/`。
- DeHate还没有该骨干，补三seed各自完整Optuna。与初版同一搜索空间：`lr` log[1e-4,1e-3]、`lamda_cma` uniform[.5,2]、`dropout` uniform[.1,.5]、`lr_answer` log[1e-4,.1]。50 epoch，hidden128、batch32、crop_repeat5、long_T512，其余共享r5训练配置不变。
- DeHate sampler seed等于训练seed；首个完整trial≤1h锁20 trial，>1h锁5 trial，之后不增减。源码使用既有共享search/train，无smoke、短训或固定配置代替搜索。
- 三语料训练阶段均沿用fixed8、无复制版本的validation AP/ROC均值选checkpoint；开发期test(AP+ROC)/2选trial。同时冻结仅validation均值选出的trial，单独报告其完整组合test结果。不会在看到完整组合test后换checkpoint或trial。
- **这是预先选定模型的完整组合重评，不是以自动停止为目标重新执行的Optuna搜索。** 新DeHate也使用同样的训练/选择协议；不把固定配置诊断混入三seed搜索，不把r5旧骨干的DeHate结果当作新骨干结果。
- checkpoint来源与选中epoch实际核验；模型构造使用 `qtl.inside_outside.make_model`，strict加载model/answer/chain，不允许旧PriorNet替代新骨干。

## 3. 推断、阈值与成本

每个被选checkpoint先在完整validation/test上运行已有复制模型的32次轨迹，再按既定因果停止规则取前缀；每一步只访问已选问题的缓存答案。三项统计只读取已发生的预测/回答；分位数参考和阈值仅来自validation，不用test统计定阈值。完整轨迹用于固定预算比较和审计，不代表部署时必须问32次。部署按相同停止判定消费前缀即可。

输出的调用数是策略实际消费的缓存节点答案数；本次不重新调用VLM，处理新视频仍需执行相同的查询及原有输入处理。新增VLM调用/特征抽取0；固定4帧提问观测不变。不把缓存重评CPU时间写成新视频端到端成本。

既有r5 DeHate每study约18.8小时（原三study并发、有资源争用），仅作墙钟参考，不视为新骨干独占GPU耗时。本次先实测新骨干完整trial，再按既有规则锁预算。两主数据只需每seed既定test/validation选出的至多两个checkpoint重评，无新增训练；DeHate为唯一新增训练语料。HateMM/HCS重评与DeHate训练资源允许时并行。

## 4. 实现与运行

初版backbone升入 `src/qtl/inside_outside.py`，原入口保留兼容导入；原复制概率估计、停止统计升入 `src/qtl/{copy_noise,stop_rules}.py`，旧工具复用它们。共享checkpoint loader在 `src/qtl/checkpoint.py`。所有实验入口均仅依赖src，不跨实验import；唯一统一评测器不修改。

- `launch/run_hateclipseg_uoa-lab1.sh`：现有HCS三seed搜索的完整组合重评。
- `launch/run_hatemm_uoa-lab3.sh`：现有HateMM三seed搜索的完整组合重评。
- `launch/run_dehate_uoa-lab{1,3}.sh <seed>`：该seed完整训练搜索，然后自动执行其选中checkpoint的完整组合重评。

所有长任务通过setsid/nohup独立后台启动；owner同时记录host、PID/PGID/SID/start_ticks和原始命令。owner在首次训练/推断前实际解析完整输入、校验shape/finite/coverage/split isolation/1fps长度，验证结果按主机/语料复用。日志首行记录主机；每个owner绑定独立 `scripts/monitor_run.py`，成功与失败均通知本会话。既有已关闭heartbeat不覆盖；新任务使用独立heartbeat目录，Goal不创建/恢复。

输出根为 `runs/20261002_complete_io_method/`。主数据重评在 `<corpus>/integrated/results/`，DeHate搜索在 `dehate/seed<seed>/`、重评在其 `results/`。每个重评trial保存source配置、selection、calibration（负例训练估计、validation参考分布/阈值）、完整轨迹、逐视频调用数、逐秒分数与统一评测器原始JSON。原训练源与旧结果不覆盖。

远端完成后回传本机，核验owner及同会话子进程退出、完整训练预算/50 epoch、全部选中trial、固定0/8/32及自动停止共4份统一评测、完整test覆盖。汇总明确区分开发期test选trial与validation选trial；同设置固定8次作为停止比较，旧r5完整设置作为性能参照。完成标记不代替这些核验。

## 5. 当前进度

独立集成code review已PASS；原共享函数实现等价性已核对，12个既定checkpoint严格加载和三语料split/时间网格检查通过。接下来三机同步，正式owner执行全输入finite/shape检查再进入运行；实际主机/PID/monitor及最新状态只在 `research-wiki/STATUS.md` 维护。没有改变研究规则、已归档实验结论或无关tandem.html。
