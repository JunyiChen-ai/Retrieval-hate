# Independent novelty review

截至日期：2026-09-01。初审遗漏项目内旧 `bag_constrained_dawid_skene` proposal；补审后最终裁定：**GO，6.8/10**。Reviewer 未审代码、未修改实现。

## Rule 12 三门

1. **Gate 1 PASS**：允许 adaptation。Nguyen et al. 的 HMM-crowd 与 Simpson & Gurevych 的 Bayesian Sequence Combination (BSC) 来源于 NER、biomedical information extraction、argument mining 的 crowd sequence tagging，不是 hateful video。
2. **Gate 2 PASS，claim 限于 sequence-crowd adaptation**：以 `Bayesian Sequence Combination`、`HMM-crowd`、crowd/annotator label aggregation、Dawid–Skene 与 hateful/hate video detection/localization 组合检索，未发现 BSC/HMM-crowd 用于 hateful video detection/localization。Generic multi-teacher KD 已被 harmful/hateful video 领域占用，因此 KD 或 single-student 本身不构成 novelty。
3. **Gate 3 PASS**：原 BSC 每个 token 已有多人 noisy labels；本任务只有 video existential label。项目旧 token-independent DS proposal 的 exact OR 只对视频内 posterior 乘共同因子，无法改变 within ordering，已 `STOP 5.8/10`。当前新增 latent true-state transition 与 source-specific boundary-confusion edge emission，使每秒 posterior 依赖整段 observation sequence并能改变帧间排序；四个异构 localizer 还有旧 proposal 不具备的双语料互补证据。这是足以脱离严格同构关闭链的新 load-bearing constraint。Video OR 只可称 bag consistency / posterior mass scaling，不能声称它本身找到更准 witness。

## 最接近工作与边界

- [Nguyen et al., ACL 2017](https://aclanthology.org/P17-1028/)：HMM crowd sequence aggregation，输入是真实 crowd token sequences；无 bag existential supervision、video localizer 或 hateful localization。
- [Simpson & Gurevych, EMNLP 2019](https://aclanthology.org/D19-1101/)：BSC 建模 true-label 与 annotator-label sequence dependencies；无从 bag labels 约束未知 temporal truth。
- [DAKD, WACV 2025](https://openaccess.thecvf.com/content/WACV2025/html/Dalvi_Distilling_Aggregated_Knowledge_for_Weakly-Supervised_Video_Anomaly_Detection_WACV_2025_paper.html)：弱监督视频、多 backbone 与 single student，但机制是 feature aggregation/KD，没有 annotator confusion、latent truth sequence 或 existential bag inference。
- [MVKD, Information Fusion 2026](https://www.sciencedirect.com/science/article/pii/S1566253526006111)：harmful-video modality/segment missingness 下的 multi-view KD；无 temporal sequence-crowd posterior。
- [MATCH](https://jianlang.org/papers/MATCH.html) 与 [CLARA](https://arxiv.org/abs/2608.15905)：分别是多 LMM evidence reconciliation、clip MoE/rationale guidance，不是 probabilistic noisy-localizer truth aggregation。

允许的 novelty claim：将 frozen temporal localizers 视作 noisy sequence annotators，并用 video-level existential labels约束 Bayesian latent hate sequence，在 train split 产生 posterior targets以训练单一 hateful temporal localization student。

不得 claim multi-teacher KD、single-student inference、MIL existential assumption、HMM/BSC或pseudo-label training本身为新，也不得声称首次在 hateful video 使用 KD/多专家。

## 必须 controls

正式运行使用同 teacher、bins、student 与 budget 的两个关键 controls：(1) **bag-constrained token-wise DS**，保留 positive/negative bag conditioning、只删除 latent transition 与 source boundary-confusion，用于隔离真正的新 sequence mechanism；(2) **unconstrained BSC/HMM-crowd**，保留 sequence dependency、只删除 video-label conditioning，用于量化 OR/bag consistency。另报告 frozen posterior 的 core-vs-token-DS within ordering delta，防止 student regularization冒充 sequence aggregation。若实现退化成逐秒 DS/stacking后只加 OR 归一，就重新落入旧关闭链并应停止。
