# Backbone 第二次修订：保留跨模态编码的区间内外残差

截至2026-10-02；状态：[一次独立proposal review：GO](../../docs/reviews/20261002_residual_io_backbone_proposal.md)，已实现，[一次独立code review：PASS](../../docs/reviews/20261002_residual_io_backbone_code.md)，待提交同步后启动完整搜索。主仓库代码与本机结果为依据；Pursuing Goal保持paused，已有heartbeat与run monitor接续。用户授权优先novel且涨点，三个主指标并列。

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

远端uoa-lab3 / sc474398输入已重新全量解析：HateMM1067视频（744/109/214）、HCS393视频（251/63/79），feature shape、时间轴、答案缓存覆盖与split isolation通过；本机记录 `runs/20261002_residual_io_backbone/setup/inputs_<corpus>.log` 和对应command.json。验证复用已提交的旧只读输入检查入口，无跨实验Python import、无训练/smoke。远端环境torch2.7.1+cu128/CUDA12.8/Optuna4.9.0与本机一致。运行主机拟为uoa-lab3；08:48 GPU9855MiB已用/22235MiB空闲，原加性HateMM两项完整study继续；lab1/lab-server仅余3723/2585MiB，已有他人任务不改动。code review已PASS；提交同步及开跑前layout检查后启动。

独立code review核验1–65长度共4225节点的区间/补集统计（最大误差7.11e-15）、outside自身行/padding/其它视频梯度排除、相同base权重下零残差train/eval的s/g/phi严格等价、非零残差不直接改变g、节点bias仅一次、真实树似然梯度/最终后验和严格checkpoint重载。原始数值 `runs/20261002_residual_io_backbone/code_review/cpu_operator_review.json`。这些是CPU算子核验，未训练、未smoke，不证明性能。
