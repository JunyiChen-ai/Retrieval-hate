# Bag-Constrained Bayesian Sequence-Crowd Student

**淘汰原因（2026-09-02）：方法以 lexical、POWA、VERA、MultiHateLoc 四个独立模型的输出聚合 train posterior target，本质是 training-stage multi-model ensemble；即使 test 只部署单 student 也不合规。立即停止 HMM producer/训练，不再补双 test，不计有效 formal performance iteration。**

截至日期：2026-09-02。此前独立 novelty 裁定 `GO 6.8/10` 现被项目方法约束覆盖并作废；该 review 只审查跨任务 novelty，没有识别 training-only ensemble 禁令。HCS 已产生的正式结果只保留为失败/流程审计证据，不可用于主方法 claim。

## Failure 与 admission evidence

HMM/HCS 当前 starting-point→SOTA gap（AP/pooled ROC/within ROC）为 `+.100831/+.077925/+.003076` 与 `+.066350/+.060950/+.038207`，依据 `research-wiki/RESET6_GOAL_GAP_AUDIT.md`。RESET6 三个 semantic auxiliary/adapter matched control 均没有双语料共同 load-bearing 增益，依据 `runs/20260901_reset7_cross_candidate_failure_matrix/main/matrix.json`，因此本候选替换 raw final scorer，不再把约束挂到 POWA 上。

实际可用 correction observation 来自 `runs/20260831_universal_teacher_simplex_diagnostic/main/metrics.json` 与 `runs/20260831_teacher_scale_transfer_diagnostic/main/metrics.json`：lexical、POWA、VERA、MultiHateLoc 四个 frozen localizer 的互补 local scores 在 HMM/HCS 都存在；完全相同的七组冻结权重经 fivefold video-heldout、无标签 score-scale mapping 后仍全部在两语料通过六项 SOTA。代表 `[.10,.25,.40,.25]` 的 HMM AP/ROC/within 为 `.597022/.825070/.667356`，HCS 为 `.627108/.620614/.566527`。它证明 independent localizers 包含足够、跨语料共同且非当前 scorer 自确认的纠错信息；但原 artifact 使用 test cohort 的 label-free reference distribution，且 blend 本身只能是 developmental upper bound，不能直接作为方法或 student target。

## 跨任务来源

来源是 noisy sequence annotator aggregation：Nguyen et al., ACL 2017 的 HMM crowd-sequence aggregation，以及 Simpson & Gurevych, EMNLP 2019 的 Bayesian Sequence Combination。来源任务是 NER、information extraction、argument mining 的 crowd sequence tagging，不是 hateful video detection/localization。

## 单一 task-adaptation delta

在每个语料自身 train split 上，四个 frozen localizer 只为 train 视频产生 1fps 连续 score，并只用 train-video reference 建 source-specific ordinal bins。把 localizer 视为 noisy annotator，以 Bayesian sequence-crowd model 推断 latent hate state。真正改变帧排序的核心是 latent transition 与 source-specific boundary-confusion edge emission `P(c_t^j | z_{t-1}, z_t, j)`：某 source 在真实 span 起止附近的提前、延迟或持续错误可以与稳定内部状态分开估计，使每秒 posterior 依赖整段 observation sequence。Negative video 将全部 latent state clamp 为 benign；positive video的 `at least one hateful state` 精确 conditioning只提供 bag consistency与per-video posterior mass scaling，不声称它本身改变within排序。

Latent posterior 只在 train 上生成一次 pseudo target，用来训练单一 multimodal temporal student。Validation 只联合选择该既定方法的超参数与 checkpoint。Test inference 只运行 student 输出一个 raw 1fps score，不读取 test GT，不计算 test-cohort CDF，也不运行 teacher aggregation；因此不是 inference ensemble/calibration。四个 teacher 均冻结，且各语料独立训练，绝不跨数据集混合 train set。

## Gain budget、可证伪预期与 control

上限相对固定 SOTA 的 margin：HMM AP/ROC/within `+.003190/+.008887/+.035824`，HCS `+.007737/+.015592/+.004619`；相对 starting point 则覆盖全部六项原始 gap。正式方法的最低预期是 core 相对 matched control 在 HMM/HCS 三指标同向非负，且每个语料至少一个主要 gap 获得 `>=.02` 绝对增益；最终晋级仍要求六项严格超过固定 SOTA。

Matched controls 使用同一 teacher train scores、同一 ordinal bins、同一 student architecture/training budget：(1) bag-constrained token-wise DS，保留bag conditioning、只删除transition与boundary-confusion，用于隔离sequence mechanism；(2) unconstrained BSC，保留sequence dependency、只删除bag conditioning，用于量化bag consistency。另直接比较 frozen posterior 的core与token-DS within ordering。若 core 不在两语料共同胜 bag-constrained token DS，或 student 的主要 gap 增益均低于 `.02`，则 sequence-crowd mechanism 被否定并关闭，不扫描 transition family、binning或 teacher subset续命。

## 正式运行配置

每个语料独立使用 `lr={3e-5,1e-4,3e-4}` × `bag_weight={.25,1.0}` 的 6 个完整 30-epoch core trial。每个 trial 先在 validation 内联合选择 checkpoint；再在 6 个 trial 间选择 AP 和 pooled ROC 距各自最优不超过 `.01` 时 within ROC 最高的配置。锁定后以相同超参数和训练预算分别训练 bag-constrained token DS 与 unconstrained BSC control，随后三个 arm 都立即在 test 运行统一评测器的 AP、pooled ROC 和 within-video ROC。入口为 `launch_formal.sh`，配置快照为 `formal_config.json`。

## 正式结果进度

HateClipSeg 已完成 6 个 core validation trial并锁定 `lr=3e-4, bag_weight=1.0, epoch=7`，随后完成两个 matched controls 与完整 test。权威 evaluator 文件位于 `runs/20260901_bag_constrained_sequence_crowd_student/formal_seed234/test/hateclipseg/{core,token_ds,unconstrained_bsc}/metrics.json`。AP/pooled ROC/within ROC 分别为：core `.603078/.569414/.533712`，token DS `.605117/.568101/.537693`，unconstrained BSC `.593208/.566153/.525380`。Core 三项均低于固定 HCS SOTA `.619371/.605022/.561908`，且相对 token DS 的 AP/within 为负。由于整个方法已被裁定为 training-stage multi-model ensemble，HMM 正式链永久取消，不再训练或评测；该轮不构成有效双数据集方法迭代。

聚焦 developmental test analysis 为 `runs/20260901_bag_constrained_sequence_crowd_student/formal_seed234/test_error_analysis/hateclipseg.json`。67个eligible正例视频上，core-minus-token-DS within均值为`-.003981`，改善/恶化`34/33`；按正例占比四分位仅最低组为`+.004234`，其余三组全负。Core-minus-unconstrained-BSC均值为`+.008331`，但主要来自最高正例占比组`+.037902`，最低组仅`+.003020`。因此sequence dependency没有形成稳定纠错，bag conditioning也没有针对low-occupancy失败；该证据不支持任何HCS corrective或超参数续命。

## 初步占用检索

- [Nguyen et al. 2017](https://aclanthology.org/P17-1028/)：HMM crowd sequence aggregation，应用于 NER 与 biomedical information extraction。
- [Simpson & Gurevych 2019](https://aclanthology.org/D19-1101/)：Bayesian sequence annotator/ground-truth dependency，应用于 NER、information extraction、argument mining。
- Hateful-video 邻近工作已存在 multi-view KD（MVKD）、多 LMM agent reconciliation（MATCH）、clip-level MoE/rationale guidance（CLARA）与普通 MIL/noisy video labels；初检未见把 frozen temporal localizers 当 noisy sequence annotators、以视频 existential label约束 latent sequence、再训练 single inference student 的方法。最终以独立 novelty reviewer 检索裁定为准。
