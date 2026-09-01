# Policy-patch excess — 淘汰：冻结patch scalar拼接未通过non-trivial adaptation门

截至 2026-08-31。两份独立审查均为STOP（`3.8/10`、`4.9/10`）；未抽取新特征、未训练、
未生成prediction。窄来源门通过，但标准CLIP patch token不等于source-faithful dense readout，
手工spatial-excess scalar接普通MIL未通过第三道non-trivial adaptation门。

## 研究问题

HCS的现有弱监督localizers和global-frame特征明显不足，而其P-MIL proposal oracle仍有`.63450`
within上限；说明时间候选覆盖并非唯一瓶颈。当前ViT/CLIP visual feature都是整帧CLS/pooled token，
小面积仇恨符号、meme局部图文或手持标识可能在进入temporal MIL前已经被空间平均。这个候选只问：
**固定视觉语言模型的局部patch是否提供一个跨HMM/HCS、相对整帧CLS新增的方向性hate observation？**

## 跨任务来源

来源族是DenseCLIP/MaskCLIP/RegionCLIP的dense或region-level vision-language prediction：把CLIP视觉
token保留到空间位置，并与文本embedding比较，而不是只读整图embedding。来源核心是否已经用于
hateful video detection/localization必须由独立reviewer检索。普通CLIP、prompt、patch token、open-
vocabulary segmentation和MIL本身都不能claim novelty。

## 固定任务改造

使用冻结CLIP ViT-B/16，在1fps帧上同时读取：

- global CLS token；
- 14×14 projected patch tokens；
- 一组从项目固定hate policy定义直接写出的视觉可观察policy clauses，以及一个benign-content
  prototype。clauses在运行前冻结，HMM/HCS完全相同，不根据test例子增删或改写。

对patch `p`和policy clause `c`计算固定cosine logit `a_{t,p,c}`。不直接取196个patch的raw max，
而定义同帧localized excess：

`u_t = logmeanexp_topq_{p,c}(a_{t,p,c}) - trimmed_mean_p(max_c a_{t,p,c})`。

top fraction `q`与trim proportion在任何数据运行前由source-style空间稀疏假设固定。第一项要求至少
一个局部policy witness；第二项是同帧空间null，去掉整幅scene/topic对全部patch的共同抬升。最终正式
模型若获准，只把冻结的`u_t`作为visual branch的一个观测量，与原global visual/audio/text输入同一个
trainable temporal localizer；`u_t`不直接当hate probability，也不在test选择branch、阈值、prompt、
crop或policy clause。

任务机制：hateful visual evidence常是空间稀疏carrier，而video topic/genre通常影响整幅图；patch相对
同帧空间背景的policy excess给positive-video内部提供一个由冻结语言语义定向、而非bag label自举的
候选witness。它不解决audio/text-only hate，故原三模态observations全部保留；也不声称patch excess
等于真实hate。

## 实现前固定 premise

只在HMM/HCS完整test cohort做一次developmental premise，使用相同帧、同一冻结模型和固定clauses：

1. `global_policy`：CLS的policy-vs-benign score；
2. `raw_patch_max`：未经空间null校正的patch最大policy score；
3. `policy_patch_excess`：上述core；
4. `spatially_permuted_clause`：每帧保留patch score multiset，但打乱policy-token对应；
5. `generic_object_prompts`：相同数量、长度的非hate视觉object clauses。

premise只比较这些冻结observations的test pooled AP/ROC与within ROC，不训练、不融合baseline、不做
calibration/smoothing。core必须在HMM/HCS的within相对global同向提高，至少一边`>=.020`；必须优于
raw max与generic clauses，policy correspondence打乱后主要增益消失；两语料pooled AP/ROC不得同时
比global各下降超过`.010`。否则`STOP_BEFORE_FORMAL_METHOD`，不得扫描prompt、q、trim、CLIP版本、
frame sampling或按语料选择readout。

## 若premise通过后的正式边界

HMM/HCS各自独立train，seed234；validation只在固定arm内选checkpoint，随后立即test三个指标。最小
arms为MultiHateLoc起点、raw patch capacity match、global-policy scalar、policy-patch excess、generic-
clause control；正式输出始终一个连续frame posterior。core必须双语料六个SOTA单元全过才扩MHC。

## 必答的novelty/identifiability问题

1. dense/region CLIP或patch-level hate evidence是否已用于hateful video detection/localization？
2. 该方案是否只是zero-shot dense CLIP加一个手工contrast和普通MIL，因而第三门失败？
3. 同帧trimmed spatial null是否真的区分localized carrier与scene topic，还是任意显著物体/字幕都会高？
4. fixed policy clauses是否构成方向性新观测，还是prompt engineering；如何用最小control证伪？
5. CLIP patch token是否在未经dense fine-tuning时与text embedding可靠对齐；若不对齐是否应在抽取前STOP？
