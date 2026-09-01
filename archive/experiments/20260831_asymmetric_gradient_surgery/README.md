# 淘汰：asymmetric conflict-projected local-order MIL

截至 2026-08-31。本候选在 novelty/identifiability review 阶段淘汰；未实现、未训练、
未生成 prediction。

## 候选

以 bag classification 为 protected objective，以 5-fold OOF lexical within-order
为 auxiliary objective。仅当两者梯度冲突时，投影 auxiliary gradient 的冲突分量；
正式输出仍是单个 student 的 raw frame score。原假设是上一轮 posterior
regularization 满足 train constraint 却损害 pooled/raw ranking，非对称投影或可保住
pooled carrier、只学习 local ordering。

## 独立裁定

两份独立 review 均为 `STOP`，评分 `3.3/10` 与 `3.8/10`。窄来源门通过：截至本次
检索，未发现 PCGrad/gradient surgery 明确用于 hateful-video detection/localization。
但第三道 non-trivial mechanism 门失败：

- PCGrad 已占用“冲突时投影 task gradient”的完整原理；Du et al. 的 gradient
  similarity、Bloop 与 wPCGrad 进一步覆盖 primary/auxiliary 和 task-prioritized 版本。
  把 bag 指定为 primary、lexical order 指定为 auxiliary 是直接 specialization。
- 对 `s_vt=c_v+r_vt`，broadcast bag gradient 可沿 `1_T`，pair-order gradient 是
  零和方向，二者天然正交；投影完全不触发，而 within 仍可为 `.5`。
- bag objective 本身偏好 whole-video shortcut；若真正有用的 local-order gradient
  与其冲突，候选反而会删除该分量。
- 一阶 Euclidean 投影不保证 Adam/动量后的实际 step 保持 bag loss，也不解释
  posterior regularization 的 test 泛化失败。

因此该候选至多是 optimizer ablation，不构成任务特定 adaptation，不进入实现。

## 来源与检索边界

- Yu et al., [Gradient Surgery for Multi-Task Learning](https://arxiv.org/abs/2001.06782), NeurIPS 2020.
- Du et al., [Adapting Auxiliary Losses Using Gradient Similarity](https://arxiv.org/abs/1812.02224), arXiv/CoRR 2018 preprint.
- Hsieh et al., [Careful with that Scalpel: Improving Gradient Surgery with an EMA](https://proceedings.mlr.press/v235/hsieh24a.html), ICML 2024.
- Bohn et al., [Task Weighting through Gradient Projection for Multitask Learning](https://arxiv.org/abs/2409.01793), ICONIP 2024.

查询包括 `PCGrad hateful video detection localization gradient surgery hate speech
multimodal`、`"gradient surgery" "hateful video"`、`PCGrad hate speech detection
multimodal`，并核对 MultiHateLoc、ImpliHateVid、CLARA、HVGuard 的方法描述。
“未检出”只覆盖公开可检索论文与上述目标领域论文，不声称全局不存在。
