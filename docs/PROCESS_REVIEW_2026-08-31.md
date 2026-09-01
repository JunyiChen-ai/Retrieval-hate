# Process Review — 2026-08-31

只读独立审查；依据 `research-wiki/STATUS.md`、`RESEARCH_ITERATION_RULES.md`、近期候选README/review及`runs/`原始test artifacts。裁定：**RESET**。

## 停滞原因

当前缺少一个合规、同时覆盖HateMM与HateClipSeg、能给positive video内部提供时间方向的新增观测。Binary bag label、video-global feature、position、teacher score、fixed reference都允许正确判断video但输出整段近常数，within-video ROC=`.5`。近期候选主要增加loss、graph、ranking或readout，没有增加可识别局部监督量。

最近一次有效双语料方法迭代是 deletion-carrier ItS2CLR；其core相对broadcast的within只提高HMM`+.003129`、HCS`+.001048`，score Spearman为`.97568/.99723`，说明机制几乎没进入最终ranking。Universal simplex/heldout scale-transfer只证明现有teacher信号组合有headroom，不证明单一novel弱监督方法可学到。已有Qwen3 dense pointwise formal test在HMM/HCS within为`.561760/.539628`，均低于`.631532/.561908`门且零生成故障，pointwise premise已充分失败。

## 流程诊断

- 重复失败链：Graph CTC与两版Hodge都用复杂结构包装没有可靠timestamp/local direction的VLM observation。
- Candidate churn：proposal/review/归档速度高于新增证据积累。
- 无效premise：反复把来源未进入目标任务误当接近novelty；实际第三门因loss/head/graph替换而失败。
- 过早复杂化：局部观测未成立前引入automaton、CTC、Hodge与robust reference graph。
- 失败后局部修补：carrier-energy与benign-anchor Hodge未新增观测。
- 目标偏移：simplex/scale upper bound把注意力推向禁止的blend/KD复现。
- Gate混杂：最小机制pilot不应与最终四语料全指标SOTA gate混成一步。

## 三候选计数

Deletion-carrier ItS2CLR进入有效HMM/HCS test时归零。之后Evidence-Program Graph CTC计1、Privileged Rank Transfer计2、carrier-energy计3，停机最迟在此触发；后两版Hodge属于触发后的越线候选。本RESET落实后新process epoch记为`0`。以后diagnostic、scout或novelty PASS不清零；只有通过novelty并进入有效双数据集test方法迭代才清零。

## 强制流程修正

1. 关闭CTC/Hodge/carrier/KD/teacher-blend修补与Qwen pointwise重跑。
2. 使用`research-wiki/FAILURE_EQUIVALENCE_LEDGER.md`先做失败等价类拦截。
3. Novelty前先做解析identifiability review：新增local observation、broadcast/position/video-identity反例、直接否证control、HMM/HCS统一coverage。
4. 只有同一native/train-only statistic在HMM/HCS均相对time-shuffle改善至少`.020`，且mean-repeated、position-only、carrier-strata controls成立，才允许生成新候选。
5. 最小pilot先检双语料within改善与机制归因；最终晋级再要求pooled AP/ROC、MHC、多seed及四语料全SOTA。
6. 同一信息源失败后不得换head/anchor/graph/loss续命；继续前必须有新独立premise evidence。
7. GPU只用于已批准的最小premise，不做full-test VLM sweep或teacher搜索。

## 方向裁定

彻底关闭：deletion-carrier/carrier-energy；teacher blend/knowledge amalgamation/KD/privileged rank transfer；无timestamp evidence-program/Graph CTC；Hodge/reference rank aggregation；calibration/CDF/routing复现simplex；Qwen pointwise readout改型。

暂停直至reopening gate：native event、ASR/OCR timing、cross-modal synchronization/co-localization、新外部局部观察器或task-specific witness assumption。

继续：MultiHateLoc作为starting architecture；现有test error artifacts作为developmental evidence；simplex/scale-transfer只作为upper bound与信息缺口证据；先做流程和状态治理，不立即生成模型。

## 评测器治理

当前实际唯一canonical evaluator入口仍是`scripts/reproduction_baselines/eval_baseline_scores.py`，核心frame逻辑在`scripts/duplex/frame_eval_common.py`；这与目录新规要求`src/`存在冲突。RESET期间冻结其逻辑与路径，任何实验不得复制或修改。迁移必须作为独立、逐项数值等价审计的基础设施裁定执行，不能夹带在方法实验中；迁移前后不得同时存在两份可编辑评测逻辑。

