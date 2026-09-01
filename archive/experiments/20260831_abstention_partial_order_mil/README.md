# 淘汰：abstention-completed partial-order MIL

截至 2026-08-31。本候选经双独立 novelty/identifiability review 淘汰，评分
`2.9/10` 与 `3.7/10`；未实现、未训练、未生成 prediction，也未获准 premise。

## 候选

统一以 MultiHateLoc 为 Stage-1 bag-MIL。拟用严格 OOF + MC-dropout 只保留高置信
frame-pair preferences，不确定 pair abstain；5-fold OOF lexical ordering 仅补全 abstained
edges。Stage-2 同容量 student 从头训练 bag loss + partial-order DAG margin loss，loss 直接
作用唯一 test raw fused score。来源拟适配 partial-abstention label ranking 与 preference
completion。

## 裁定

精确机制未检出进入 hateful-video localization，窄来源门通过；第三门失败：

- Cheng et al. 输出 probabilistic label partial order；Gunasekar et al. 在共享 item universe
  上用 low-rank/nuclear-norm 完成 partial rankings。当前没有共享 time-item matrix 或低秩
  completion，只是 confidence-filtered self-order teacher 加 lexical fallback teacher。
- 严格 5-fold OOF 每个视频只有一个 held-out model；额外 MC-dropout 的稳定性不等于
  pair correctness。稳定的 topic/broadcast shortcut 反而会被认证为 confident edge。
- confident-wrong base edges 永远不能被 lexical 修正；base 大量 abstain 时则退化成此前
  已失败的 lexical order distillation。
- base 与 lexical edges 可形成 cycle；任何删边或优先级规则都会成为新的 teacher routing。
- `s_vt=c_v+r_vt` 中，bag loss 可由 `c_v` 完成，edge loss只约束 `r_i-r_j`；任意
  artifact residual 就能满足伪边，仍不保证真实 hate ordering。
- HCS 的新证据来自 VERA high-tie 视频，不证明 MultiHateLoc uncertainty 能识别该把哪些
  pair交给 lexical。

因此它是 selective pairwise distillation，不是忠实 preference completion，也没有新增局部
识别信息，不进入实现。

## 来源

- Cheng et al., [Label Ranking with Partial Abstention based on Thresholded Probabilistic Models](https://proceedings.neurips.cc/paper/2012/file/fe2d010308a6b3799a3d9c728ee74244-Paper.pdf), NeurIPS 2012.
- Gunasekar et al., [Preference Completion from Partial Rankings](https://arxiv.org/abs/1611.04218), NeurIPS 2016.
- Yang et al., [Uncertainty Guided Collaborative Training for Weakly Supervised Temporal Action Detection](https://openaccess.thecvf.com/content/CVPR2021/papers/Yang_Uncertainty_Guided_Collaborative_Training_for_Weakly_Supervised_Temporal_Action_Detection_CVPR_2021_paper.pdf), CVPR 2021.
