# 淘汰：SDR-style complementarity blend distillation

截至 2026-08-31。本候选在双独立 novelty/identifiability review 阶段淘汰，评分均
为 `3.4/10`；未实现、未训练、未生成 prediction。

## 候选

尝试适配 Qin et al. 的 Self-Distilled neural Rankers：训练期将 5-fold OOF lexical
local ordering 与 frozen pooled carrier 组合成 per-video teacher ordering，经 score
transformation 和 listwise loss 蒸馏到单个 MultiHateLoc-like student；test 只输出
student raw frame score。诊断给出的构造是 HMM lexical `.35` + POWA `.65`，HCS
lexical `.05` + VERA `.95`。

## 裁定

精确 SDR/listwise temporal-rank distillation 未检出用于 hateful-video localization，
窄来源门通过；但 knowledge distillation 已进入 hateful-video detection，ranking
distillation 也已有直接跨域先例。第三道 non-trivial mechanism 门失败：

- 原 SDR 是同参数化 self-distillation，teacher 由原始 graded ranking labels 训练，
  student 同时使用原 ranking objective 与 transformed teacher listwise loss。当前没有
  frame ranking labels，teacher 是异构 signal blend，机制不再是原 SDR。
- 两个精确权重来自已揭盲 test grid，且 HMM/HCS 选择不同 carrier 和权重；这虽可按
  当前规则作为 developmental test-informed design，却实质是 per-corpus calibration /
  teacher routing，不可作为主方法。
- `s_vt=c_v+d_vt` 中，listwise softmax 会消掉 broadcast offset `c_v`。student 可用
  `c_v` 完成 bag 分类，只以任意小的 `d_vt` 模仿 teacher；没有解决 PR 已暴露的
  train ordering 向 test 泛化问题。
- HCS target 的 `.95` VERA 表明所谓共同机制主要是特定 teacher imitation；`.05`
  lexical 的收益也可能只是 test realization 中的 tie/近平分扰动。

因此它是普通 heterogeneous ensemble ranking distillation，不满足本项目 novelty，也
触及禁止 ensemble/calibration/routing 作为主方法的边界，不进入实现。

## 来源

- Qin et al., [Improving Neural Ranking via Lossless Knowledge Distillation](https://arxiv.org/abs/2109.15285), 2022.
- Reddi et al., [RankDistil: Knowledge Distillation for Ranking](https://proceedings.mlr.press/v130/reddi21a.html), AISTATS 2021.
- Ju et al., [Prompting Visual-Language Models for Efficient Video Understanding](https://arxiv.org/abs/2212.09335), CVPR 2023（弱监督 temporal action localization）。
- [LEAF](https://aclanthology.org/2026.findings-acl.604/), Findings ACL 2026（hateful-video detection distillation）。
