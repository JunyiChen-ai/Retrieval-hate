# Independent identifiability and mechanism review

截至 2026-08-31。审查对象为：将 ICCV 2023 Inverse Compositional Learning（ICL）从 weakly-supervised
video relation grounding 适配到只含 binary video labels 的 hateful-video temporal localization。未实现、未训练、
未生成 prediction。

## Verdict

**STOP。Novelty：4.2/10。**

[Li et al., *Inverse Compositional Learning for Weakly-supervised Relation Grounding*, ICCV 2023](https://openaccess.thecvf.com/content/ICCV2023/html/Li_Inverse_Compositional_Learning_for_Weakly-supervised_Relation_Grounding_ICCV_2023_paper.html)
未检出用于 hateful-video detection/localization，跨任务来源门可以通过；但当前 adaptation 删除了来源方法赖以识别
relation grounding 的关键监督：**每个视频给定的、具体且会变化的 `<subject,predicate,object>` query 及其三个类别**。
固定 policy-role 类型加 binary bag label 不能替代该信息。inverse reconstruction、partial role classification 与 MIL
存在同时达到最优而 frame ranking任意的解析解，因此第三道机制门失败。

## 1. 先纠正对来源方法的描述

原 ICL 不是“只凭一个 video-level 是否含关系的二元标签，反推出关系”。其训练样本给定具体 relation query

\[
R_i=\langle s_i,p_i,o_i\rangle,
\]

例如 `<person, ride, bicycle>`；弱监督只指**没有 subject/object boxes 和 relation temporal span**。论文的 holistic
目标比较 query relation feature 与 subject/object relevant、partly relevant、irrelevant compositions；partial-level
classification 则直接使用已知的 subject、predicate、object类别。论文式 (13) 甚至假设每帧都含 query relation，
把相同三个类别赋给所有帧。

当前候选只有 `y_i∈{0,1}`，并把 query 换成少数固定 policy clause/role types，例如
`<source, hostile-predicate, protected-target>`。这里的 `source` 与 `protected-target` 是角色类型，不是视频特定的
speaker identity、target identity 或 relation phrase。由此，原 ICL 中跨视频变化的 query supervision 消失；若仍声称
faithful ICL adaptation，监督语义不成立。

## 2. 最小退化反例

### 2.1 Fixed-query reconstruction collapse

对一个固定 clause `c`，设 query tuple 为常数 `q_c`，proposal selector为 `a_{itc}`，聚合表示为

\[
v_{ic}=\sum_t a_{itc}z_{it}.
\]

若 inverse/reconstruction decoder接收 clause embedding，取

\[
D(v_{ic},c)=q_c
\]

即可对所有视频、所有 proposal、所有 selector 得到零 reconstruction error；decoder只需复制 clause或用 bias输出
固定 tuple。即使 decoder不显式接收 `c`，有限 relation bank也可由每个 clause-specific head 的 bias 输出对应 `q_c`。
因此

\[
L_{recon}=0
\]

对 `a_{itc}` 完全无约束。所谓“grounded proposal重构完整 tuple”没有证明 tuple 信息来自 proposal。

partial role classification同样退化：若所有视频的目标只是三个固定 role types，三个 classifier 分别恒输出
`source/predicate/target` 即可。它学到的是 slot name，不是视频内 entity identity。

这是当前 specification 的最小直接 STOP 反例。

### 2.2 Topic-broadcast shortcut

令每个 proposal token为

\[
z_{it}=[g_i,\epsilon_{it}],
\]

其中 `g_i` 是全视频 topic/channel/speaker shortcut，且足以预测 `y_i`；`ε_it` 才包含局部内容。令 relation score

\[
r_{itc}=\sigma(w^Tg_i)
\]

对同一视频所有 `t` 广播。positive bag全部秒高、negative bag全部秒低，max/top-k/noisy-OR MIL均可取得极低 bag
loss；fixed tuple由上一节的 constant decoder重构。此时 positive-video 内 frame score为常数，within-video ROC为 `.5`。

即使 positive bag被规定“只能有一个 proposal负责”，selector也可固定选择第一 proposal：

\[
a_{itc}=\mathbf 1[t=1].
\]

只要 temporal encoder把 `g_i` 注入第一 proposal，bag与 reconstruction目标仍全部满足。把“至少一个”写成完整 relation
MIL不排除 fixed-position witness。

### 2.3 Partial roles不产生有向 relation identity

考虑视频中有以下分散证据：

- 时刻 `t1`：speaker A；
- 时刻 `t2`：hostile predicate，但由 speaker B说出；
- 时刻 `t3`：protected group G 出现，但它是另一段被正面讨论的对象。

三个 marginal role heads分别能检测 `source`、`predicate`、`target`，fixed decoder仍能输出
`<source,hostile,protected-target>`。但视频中不存在 `A hostile-to G` 这一事实。若 role只表示类别而不保留
`A/B/G` identity与共同 proposition ID，composition只是 role co-occurrence，不是有向 relation binding。

把三个 role token放进同一 proposal也不充分：proposal encoder可含全视频 context，仍能把分散的 topic evidence复制到
每个 proposal。需要的是 identity-specific tuple candidates及 one-role-swapped negatives，而 binary bag label没有给出哪个
speaker、predicate和target应该配对。

### 2.4 Relation-bank MIL不可识别

令 `c∈{1,...,C}` 表示 policy relation bank，positive likelihood只依赖

\[
\max_{c,t}r_{itc}
\]

或等价 OR aggregation。对任何 relation permutation `π` 和 positive-video proposal permutation `ρ_i`，变换

\[
r'_{itc}=r_{i,\rho_i(t),\pi(c)}
\]

保持 bag likelihood不变。fixed-query reconstruction heads与 role heads一起按 `π` 重命名即可保持 auxiliary losses不变。
negative bags只要求所有 score低，不能决定 positive bag 中哪个 `c,t` 是真 relation。因此 clause ownership与 temporal
assignment最多识别到任意 permutation，且可被 topic shortcut替代。

## 3. 指定风险逐项裁定

### Fixed query collapse

**成立，且是直接 STOP。** 原 ICL 的 query随 relation实例变化；当前 policy query在全语料固定或只有很小的固定 bank。
inverse decoder可忽略 proposal并输出 query template。

### Topic shortcut

**成立。** Binary bag labels只区分 positive/negative videos，global topic可广播到所有 proposals。inverse/partial losses
没有强制 score使用局部 residual。

### Unary HateClipSeg

**成立。** 若 HCS policy实际只剩 unary hateful predicate，ICL 的 subject/object relevant/inverse四种 composition
`++,+-,-+,--` 不再存在有意义的两端关系。把 `source=current-author`、`target=any` 设成常量只会把 unary prompt
包装成三槽 tuple，数学上仍是普通 prompt-conditioned MIL。若坚持非空 source/target，视觉/文本缺失视频会变成不可行
或迫使模型 hallucinate roles。禁止为 HCS另开 unary branch，否则共享机制已经退化且形成按语料路由。

### Role identity

**未识别。** Role classification只告诉模型“这是 source slot/target slot”，不告诉它是哪一个 source、哪一个 target，
也不证明 predicate在二者之间成立。完整 tuple的字符串可被重构，完整事实仍可不存在。

### Relation-bank MIL identifiability

**未识别。** Positive bag存在性与 negative-bag全负约束不能消除 clause permutation、proposal permutation、first-position
selection或 global broadcast。增加 relation bank容量通常会增加可交换解，而不是提供方向监督。

## 4. 相对普通 prompt-MIL 与 POWA 的真实增量

### 相对 prompt-MIL

在上述退化解上，inverse module是零损失的旁路：最终可训练部分等价于

\[
r_{itc}=f(z_{it},q_c),\qquad \hat y_i=\operatorname{MIL}_{t,c}(r_{itc}),
\]

即普通 clause/prompt-conditioned MIL。只要 constant decoder与constant role heads存在，增加 reconstruction loss不缩小
这一解集。因此它没有可识别的新约束。

### 相对 POWA

POWA 已有 typed moderation primitives、hostile-to-target asynchronous binding及 policy AST MIL。当前候选表面新增的是
“holistic tuple reconstruction + partial roles”；但：

- fixed policy tuple可由 clause embedding自重构；
- partial roles不含 entity/proposition identity；
- inverse relevant/irrelevant ordering可由 query-conditioned attention构造满足，而不证明 relation在proposal中；
- source/predicate/target共现不比 POWA 的 typed primitive binding 更强。

所以当前 ICL adaptation没有证明新增约束超出“换一个 query-conditioned scoring block”。若把 video-specific、
identity-resolved tuples作为额外输入，才可能真正继承 ICL 的 query variation；但这会引入当前 specification没有的外部
pseudo relation supervision，必须作为另一个候选重新审查，不能继续声称只由 binary video label识别。

## 5. 为什么常见 controls 不能挽救当前目标

观察到 reconstruction loss非零、partial accuracy较高或 inverse ablation掉点，都不能排除 decoder bias、query copy或
regularization/capacity效应。最低限度本应要求：

1. decoder不得接收 query/clause embedding，且在 proposal表示上做严格信息瓶颈；
2. 每个视频有会变化的 identity-resolved tuple，而不是固定 role template；
3. 构造只替换一个 role identity 的 hard negative tuples，其他两角色及topic保持不变；
4. query shuffle、proposal shuffle、fixed-first、global-broadcast与同容量 prompt-MIL均为绑定 controls；
5. HCS必须实际存在至少两个非恒定、可观测的 relation endpoints，不允许 unary fallback；
6. 最终 score必须对 one-role swap敏感，同时对无关 proposal permutation稳定。

第2–3项所需监督不由 binary video labels提供。因此这不是“补两个 control后即可运行”的 conditional GO，而是需要改变
问题设定。

## 6. 最终结论

ICL 本身是合理且已有明确监督语义的跨任务来源；问题出在当前 adaptation把“已知且随样本变化的 relation query”替换为
“固定 policy-role template”，却继续假定 inverse reconstruction能定位 relation。它不能。constant decoder、constant
role heads、topic-broadcast MIL构成一个同时满足全部目标但没有 temporal localization的最小解；unary HCS进一步使
composition失去关系含义。

因此裁定 **STOP**：不实现、不训练，也不把它改名为 relation-aware MIL继续。只有获得 train-only、video-specific、
identity-resolved relation tuples并重新定义监督边界后，才可作为不同候选重新查新与审查。

