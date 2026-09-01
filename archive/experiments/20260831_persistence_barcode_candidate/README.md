> 淘汰原因（2026-08-31）：独立查新与机制审查判定 1D score barcode 等价普通 peak
> prominence/merge tree，不能识别真实 component 数量，也不能保证抑制高而窄的孤立假峰；
> novelty 与机制证据不足，未实现、未训练、未生成新 test prediction。

# Persistence-barcode MIL candidate

候选曾拟把 frame-score superlevel filtration 的 0D birth/merge-saddle pairs 直接作为 positive/
negative bag likelihood，并在 test 仅输出 raw single frame score。它针对上一轮 test diagnosis
发现的 HMM/HCS transition inflation `4.5×/5.5×`，试图区别于 fixed smoothing。

独立 review 最终 `STOP`（保守 novelty `4/10`；另一 reviewer 仅给 `5.5–6/10 CONDITIONAL
GO`）。决定性问题不是是否能写出可微 barcode，而是：

1. 一维 0D persistence 正是 peak prominence；普通 prominence pooling 是数学等价 control，
   候选无法证明独立 topological mechanism。
2. persistence 不等于 duration。高而窄的 1 秒假峰可以高度 persistent；长但平缓的 HCS
   真事件反而可能低 persistence。
3. 降低 persistence 的梯度既可压峰，也可抬高 merge saddle 填谷；后者仍是 adaptive
   smoothing，而现有 test 已证明 smoothing 在两语料方向相反。
4. video label 不给真实 component 数量；max bar 退化 single-witness MIL，惩罚其他 bars 又会
   伤害多段真事件。
5. 相关占位包括 PISMIL（PH+MIL）、Topology Layer、differentiable barcode optimization、
   differentiable topological local maxima、TopoWalk temporal topology、LEC-VAD event
   completeness 与 LAS-VAD components。窄形式未见完全相同，不足以抵消数学退化与可识别性
   问题。

因此不再做零训练 persistence probe：所谓“persistence 是否在控制 ordinary prominence 后有
增量”在 1D 已被数学等价关系否定。继续实现只会把 ordinary peak-prominence MIL 换成
topology 术语，属于 anti-pattern。
