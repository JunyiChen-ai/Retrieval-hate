# Local-Quotient Adversarial MIL

截至 2026-09-01。RESET4候选；不做premise，复用现有四语料test evidence。状态：等待独立novelty三门裁定。

## Failure

权威test evidence为`runs/20260831_multihateloc_test_error_analysis/main/metrics.json`、`runs/20260901_marked_splat_test_error_analysis/main/metrics.json`。MultiHateLoc fused胜全部单模态比例仅HMM/HCS`.345/.154`，best-branch oracle相对fused within缺口均`.106`。Marked-splat进一步显示position-only在HMM/HCS within已达`.709890/.530324`，而去共同位置轮廓后的内容排序只有`.591156/.512890`。弱video标签允许encoder把video identity和normalized position编码进每秒local score；该shortcut既能支持bag分类，又不提供可迁移的hate timing。

## Cross-task source

来源为Ganin et al., *Domain-Adversarial Training of Neural Networks*（JMLR 2016）的gradient-reversal nuisance removal：主任务feature extractor同时被训练为使domain classifier无法恢复domain identity。检索尚未发现“training-video identity + normalized temporal bin adversarial quotient”用于hateful-video detection/localization。

来源：<https://www.jmlr.org/papers/v17/15-239.html>。

## Task adaptation delta

本任务不存在source/target domain adaptation。候选把每个training video identity与normalized temporal bin定义为两个**已知 nuisance labels**，但只从local localization channel移除它们；video-global carrier明确保留：

1. 使用两套结构相同但参数独立的MultiHateLoc backbone；global backbone的fused representation做masked mean得到`g_v`，local backbone的fused representation去自身masked mean得到`q_vt`；
2. global head从`g_v`输出`c_v`，负责跨视频pooled separation；local head只从`q_vt`输出zero-mean `r_vt`，唯一test frame logit为`c_v+r_vt`；两套backbone只通过最终bag loss共同训练，adversary gradient不能进入global backbone；
3. 两个gradient-reversal adversary只读取`q_vt`：一个预测training-video ID，另一个预测固定的8个normalized temporal bins。它们降低local channel对video identity与absolute relative position的可解码性，而不会直接删除global head保存的video-level hate carrier；
4. 唯一positive/negative bag BCE作用于`c_v+topK(r_vt)`。Adversary在test完全删除；输出是单模型raw frame sigmoid，无ensemble、calibration、router、teacher或post-processing。

这不是把DANN直接用于跨数据集：domain不是corpus，且没有目标域；adaptation利用弱时序MIL特有的监督分解，把video-constant判别合法地隔离到global carrier，同时只对必须承担within ordering的centered local channel做instance-identity与position nuisance suppression。若local pattern只记住“这是某个positive training video的第几个bin”，adversary会直接惩罚；目标是提高跨video复现的local hateful structure进入`r_vt`的相对优势，不声称严格quotient或完整invariance。

## Falsification and matched control

Matched control使用完全相同的global/local decomposition、heads、参数量、MIL和validation预算，但gradient reversal系数为零；adversary自身仍训练，保证唯一差别是nuisance gradient是否进入local representation。Core必须在HMM/HCS test within同时胜control与seed-234 MultiHateLoc anchor，至少一边`>=+.020`。机制control在formal test后检查core相对control是否同时降低video-ID与position-bin probe accuracy，以及core-control within gain是否在position-only高风险视频上为正；video-ID probe对每个training video的frames做不重叠probe-train/probe-eval划分，保证所有video IDs在两侧都出现，不能用unseen IDs测试分类。这些diagnostic不用于checkpoint选择。最终晋级仍要求HMM/HCS三个固定test指标全部SOTA，之后扩MHC-EN/ZH与多seed。

## Novelty verdict

独立裁定：`GO 6.6/10`。三门均通过。最窄claim仅为：在保留video-global bag evidence的同时，adversarially降低最终local scoring channel对training-video identity与absolute position的可解码性。不能claim新DANN、严格数学quotient、完整nuisance invariance、首次video debias或可识别性证明。

超参数为video-adversary权重、position-adversary权重、local residual scale与学习率。Novelty通过后每语料固定12个validation-only trials，以validation within联合选择超参数和checkpoint，锁定后立即test；正式运行前只做一次technical review。

## 正式结果与去向

独立novelty裁定`GO 6.6/10`。唯一technical review在修复global/local未隔离、额外branch BCE、缺mechanism diagnostic与缺可读代码说明后最终`PASS`。HateMM/HateClipSeg各完成12个validation-only trial，以validation within联合选择超参数与checkpoint；selection为`runs/20260901_local_quotient_adversarial_mil/val_search/{hatemm,hateclipseg}/selection.json`。锁定后立即test，权威汇总为`runs/20260901_local_quotient_adversarial_mil/formal_val_selected_seed234/summary.json`。

HateMM control/core AP、pooled ROC、within为`.476703/.741083/.554567`与`.476712/.741098/.554234`；core-control within=`-.000332`，相对MultiHateLoc anchor=`-.074222`。Core没有降低video-ID probe（`.022919→.023121`），position probe仅`.167348→.167045`，high-position-risk组core-control within=`-.000352`。

HateClipSeg control/core为`.526832/.507813/.490752`与`.493789/.448357/.513092`；core-control within=`+.022341`，但相对anchor仍`-.010609`且pooled两项大幅下降。HCS两个probe确实下降：video-ID`.080553→.034736`、position`.182644→.138073`，high-position-risk组within增益`+.057615`，说明GRL在HCS执行了声称的nuisance suppression，但没有形成足够localization或pooled性能。

两语料不存在共同corrective：HMM机制未进入，HCS机制进入但以pooled崩溃和低风险组`-.014003`为代价。按Rule 18不调GRL schedule/weight、local scale、backbone共享方式、bin数或global/local loss；family关闭并归档。本轮为RESET4第三次正式performance failure，触发Rule 13 process review。
