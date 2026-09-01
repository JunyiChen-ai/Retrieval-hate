# Independent novelty review: inverse-compositional policy grounding

截至 2026-08-31。只做来源占用、机制边界与数学可识别性审查；未实现、未训练、未生成 prediction。

## Verdict

**STOP。Novelty：4.6/10。**

三项硬门：

1. **外部来源可 adaptation：PASS。** Li et al. 的 inverse compositional learning（ICL）确实是一个可迁移的弱监督 video relation grounding 来源。
2. **来源核心尚未用于 hateful-video detection/localization：窄 PASS。** 本次检索没有发现 ICL 的 relevant/inverse constituent competition 被用于 hateful-video detection 或 temporal localization。不能扩大成“首次 relation/evidence grounding”：LELA 已做 composition matching，MATCH 已做 spatiotemporal evidence-grounded verification，LEAF 已做 self-grounded explanation，项目内 POWA 与 LB-SCGP 更已经分别占用 target–predicate transport 和 source/stance/predicate/target policy semantics。
3. **non-trivial task adaptation：FAIL。** 当前 specification 把三个 policy role 名称代入 relation grounding，再增加 proposal MIL 与一个并非来源 ICL 原机制的 role reconstruction decoder。在只有 binary bag label、没有每视频 relation query/role identity supervision时，完整目标存在 fixed-query、clause、topic、whole-video 和 HCS-null collapse；matched role-swap clauses也可由 query syntax 单独识别。因而该 adaptation 尚未产生区别于普通 scalar MIL 的可识别监督量。

当前不应进入 premise 或训练。若只把它实现成 relation-aware baseline，必须准确标成 **ICL-inspired policy-role proposal MIL**，不能称为 source-faithful ICL adaptation，也不能 claim 其 reconstruction 建立了 directed hate relation。

## 1. 来源方法实际核心

[Li et al., ICCV 2023](https://openaccess.thecvf.com/content/ICCV2023/html/Li_Inverse_Compositional_Learning_for_Weakly-supervised_Relation_Grounding_ICCV_2023_paper.html) 的任务输入不是 binary event label，而是与视频配对的已知三元 relation query `<subject, predicate, object>`。它还使用 Faster R-CNN region proposals，在空间和时间上定位 subject/object box sequences。

其 load-bearing 结构是：

- 对 subject 与 object 分别计算相关 attention feature和 inverse/irrelevant attention feature；
- 组合成 `v++`, `v+-`, `v-+`, `v--`；
- 用距离次序约束关系 query 与这些组合的相关性：`v++` 应优于只有一个 constituent relevant 的组合，后者再优于 `v--`；
- partial branch 用 subject、object 与 predicate 类别做 classification；论文明确把同一 query label赋给采样视频的每一帧；
- inference 用 temporal attention生成 candidate segment，再在其中选择 subject/object box pairs。

这几点带来两个直接边界：

1. 来源的 `inverse` 指 **相关/不相关 constituent attention 的组合竞争**，不是从 proposal latent 解码并重构三个 role identity。
2. predicate只进入 partial classification，并没有与 subject/object对称地进入 inverse attention四格；来源也没有 README 所写的 source/predicate/target 三角色 inverse decoder。

因此，README 的“inverse decoder必须从 grounded proposal重构同一 clause 的三个 role identity”不是 ICL 的忠实移植。它更接近 [vRGV, ECCV 2020](https://arxiv.org/abs/2007.08814) 的 relation attending/reconstruction、ICL 的 holistic/partial competition，以及 proposal MIL 的组合。这个新 decoder 可以另行提出，但不能把它当成 ICCV 2023 已经验证的 source core。

## 2. Hateful-video occupation check

### 未发现的精确占用

在检索到的 hateful-video detection/localization 方法中，没有发现以下完整核心：

`policy relation query -> constituent relevant/inverse grounding -> partial/holistic competition -> single temporal relation marginal`。

[MultiHateLoc](https://arxiv.org/abs/2512.10408) 使用 modality-aware temporal encoders、dynamic fusion、cross-modal contrast与 snippet MIL，没有 relation-query decomposition。[SafeLens](https://ojs.aaai.org/index.php/AAAI/article/view/42390) 做 segment-level multimodal moderation，也没有 constituent inverse grounding。

### 已占用的宽泛表述

- [LELA](https://arxiv.org/abs/2602.09637) 已在 hateful-video temporal localization 中使用 modality decomposition、多阶段 prompting和 composition matching。
- [MATCH](https://jianlang.org/papers/MATCH.html) 已把 hateful-video 方法表述为 spatiotemporal evidence-grounded verification，生成/验证 hate与non-hate clues。
- [LEAF](https://aclanthology.org/2026.findings-acl.604/) 已用 self-grounding CoT生成 hateful-video explanation supervision并蒸馏到轻量模型。

所以可守的 occupation claim只能是：

> We did not find inverse relevant/irrelevant constituent competition from weak video relation grounding applied to hateful-video temporal localization.

不能 claim 首次 compositional grounding、relation-aware hateful-video reasoning、policy role decomposition、evidence grounding或 explanation grounding。

## 3. 与项目内最近机制逐项比较

| 最近机制 | 已有 load-bearing 部分 | 当前候选真正新增了什么 | 裁定 |
|---|---|---|---|
| POWA | typed policy primitives；hostile–target 异步 transport；policy AST MIL | proposal内 source/predicate/target partial states和 reconstruction/swap loss | policy relation semantics已占用；若 reconstruction可被删掉而最终排序不变，当前只是 POWA roles 换一个 MIL head |
| Dense primitive teacher | window-level typed hostile/target/context等外部语义分数，并按 policy compile | 不再用 teacher score，尝试由 binary bag label端到端发现 roles | 监督更弱；不能把“typed attribution”当新增信息，也不能假定 bag label能恢复 teacher没有提供好的 temporal ordering |
| Policy-complete P-MIL | proposal内 role coverage与 clause completeness scalar | 声称从 partial states形成 directed relation，而非先算 frozen scalar | 这是唯一潜在增量；但当前 reconstruction不约束 entity/role identity或edge，只用同一 proposal重构固定 clause，仍不能排除 interval内偶然共现 |
| 旧 LB-SCGP | whole-video direct-speaker endorsement、quotation/condemnation/reportage exception、target-predicate与 speaker-source/stance binding | temporal proposal与最终 frame marginal | source/accountability 的 task semantics已被目标领域内部候选占用；仅把相同 atoms 时间展开不构成新 task adaptation |

当前方案与 policy-complete P-MIL 的概念差异比旧 scalar gate更强，但仍没有形成可识别的 **directed edge**。一个 proposal同时含 source-like、hostile-like与target-like特征，不意味着该 source对该 target表达该 predicate；inverse decoder最多证明 proposal latent能预测 clause tokens。

## 4. Binary bag label 下的解析退化

设 clause bank为固定集合 `Q`，proposal为 `I`，完整关系能量为 `R(I,q)`。训练观测只有

`Y=1 => exists (I,q): R(I,q) high`,

`Y=0 => for all (I,q): R(I,q) low`。

这个 likelihood 没有观测：哪个 clause成立、哪个时间属于哪个 role、三个 role是否指向同一实体，或 relation方向。

### 4.1 Fixed-query reconstruction collapse

若 decoder接收 clause embedding或 clause id，令所有 role attention都等于普通 hate feature `h(I)`，decoder直接从 `q` 输出 source/predicate/target tokens。reconstruction loss可很低，而视频相关部分仍只有一个 scalar `h(I)`。

即使 decoder只接收 composite latent，固定且很小的 clause bank也可由 encoder把 clause identity写进 latent；只要 MIL选中任何 positive proposal，decoder记忆 clause code即可。它没有被要求从三个不同的 grounded observations恢复 relation。

### 4.2 Role-sharing collapse

令

`z_source(I)=z_predicate(I)=z_target(I)=phi(h(I))`。

compositional encoder仍能构造一个高分 `R(I,q)`，partial decoder也能用不同输出头产生三个 token。typed heads并不意味着 typed observations；没有互斥、identity conservation或 cross-role edge supervision时，三支可共享同一帧、同一模态和同一 topic shortcut。

### 4.3 Role-swap query-only collapse

对于合法 query `q` 与 role-swapped negative `q_swap`，模型可写成

`R(I,q)=h(I)+b(q)`。

其中 `h(I)`完成 hate bag classification，`b(q)`仅根据 policy grammar给合法role order较高常数、给 swapped模板较低常数。这样所有 matched swap losses都通过，但 video representation完全没有学习谁攻击谁。

要排除它，negative必须保持 query合法性、token multiset、role marginals和视频 topic相同，只破坏 **视频中具体 constituent identity的绑定**；当前数据没有该 identity supervision。

### 4.4 Topic与whole-video collapse

令 `h(I)`只响应新闻主题、政治人物、战争画面或全片字幕风格。positive MIL会选覆盖该shortcut的 proposal；role heads共享 `h(I)`后仍可重构固定 policy clause。扩大到 whole-video proposal还能更容易同时收集三个 marginal cues。necessity、inside/surrounding或proposal长度控制都没有在当前核心中建立 directed relation。

### 4.5 Clause collapse

positive label只要求 clause bank中一个 clause成立，因此所有 positive bags可以路由到最容易的同一 clause。即使其他 clauses有不同 source/target/predicate结构，也没有 per-video clause label迫使它们被使用。加 clause-balance正则只能规定使用频率，不能规定语义正确性。

### 4.6 Co-occurrence without direction

一个长 proposal可以在开头包含speaker/source，中间包含 hostile words，结尾包含 protected-group mention；三者可能来自不同说话人、引述或不同事件。任何对 proposal pooled feature的 reconstruction都可成功，却不证明 `source -> hostile(predicate,target)`。这正是 policy-complete P-MIL 已有的 interval co-occurrence blocker；当前 decoder没有保存 entity track、speaker identity或 proposition edge，因此没有解决它。

## 5. HateClipSeg 的结构性退化

README 已指出 HCS可能没有显式 source或target。两种处理都失败：

- 若要求三角色非空，许多真实 violence/sexual/self-harm等 unary events没有可行完整 relation，机制 coverage不足。
- 若允许固定 null-source/null-target，query成为 `NULL -> harmful predicate -> NULL`。两个 null role可由常数重构，完整关系 posterior代数上只剩 predicate evidence；它与 README要求胜过的 predicate-only control没有不可拆差异。

禁止 corpus-specific unary fallback是正确约束，却不能消除这个两难。相同三角色核心无法同时对 targeted hate与本体为 unary harm的 clauses提供非退化机制。若从 HCS移除这些 clauses，方法与数据集 positive定义不匹配；若保留，第三门不能靠 HateMM 的 relational story通过。

## 6. Controls 是否足够

README 的 controls都是必要的，但不足以支持归因：

- `inverse-loss removed` 能说明辅助loss是否影响结果，不能说明学到directed relation；
- `role identity shuffled` 若只shuffle query token/role label，会被 `b(q)`式syntax shortcut解决；
- `untyped composition` 只能说明不同heads有用，不能说明不同observations被ground；
- `predicate-only/target-only` 能排除单一显式branch，却不能排除三个branch复制同一 `h(I)`；
- final score只来自 relation posterior可以防止显式fallback，但 ordinary hate scalar可藏在该posterior内部。

至少还需要以下 attribution tests；它们能证伪，但不能单独修复当前 identifiability：

1. 三个 role attention 的逐proposal数值相等/高相关审计，以及 branch parameter-sharing capacity match；
2. query-only model和 `h(I)+b(q)`显式control；
3. 同视频、同proposal内保持三个role unary marginals不变，仅把具体 source/target identity跨proposition错配的control；
4. whole-video、length-only、clause-frequency与topic-only controls；
5. role-specific evidence deletion：删除source证据只能降低含该source的有向relation，而不是所有 clauses一起下降；
6. HCS逐clause证明 null-role core与predicate-only在公式和实际score上不等价；若等价，HCS不计mechanism evidence。

## 7. 若未来重提，最小不可拆机制

本版本不是 GO，因此以下不是实现授权。一个可重新送审的最小核心至少必须同时具备：

1. **可追踪的 constituent identity。** source至少对应train input中可重复识别的speaker/face/voice track，target对应mention/entity state；不能只有三个匿名attention vectors。
2. **非可分解的有向 relation energy。** 正式能量必须包含无法写成 `a(source)+b(predicate)+c(target)+h(video)` 的 cross-role interaction，且frame score只由该interaction的proposal posterior边缘化得到。
3. **marginal-preserving binding negatives。** 在同一视频/同一proposal几何内保持source、predicate、target各自证据与query合法性不变，只错配具体identity或direction；core必须胜过这些negatives。普通role-token swap不够。
4. **source-faithful inverse competition。** 同一个 grounded identity的 relevant/relevant组合必须严格优于 relevant/inverse和inverse/inverse组合；不能把“decode固定query”命名为ICL。
5. **HCS非退化定义。** 在不增加per-corpus fallback的前提下，必须说明 unary harmful event为何仍产生非空、非predicate-only的关系变量；若做不到，该机制不能作为四主语料共享方法。

这些条件会把方法变成另一个候选：identity-preserving directed policy-relation MIL，而不是当前 README 的小修。只有先写出其可执行 likelihood并给出 `h(I)+b(q)`、null-role和whole-video解不能同时达到最优的条件，才有资格重新给 CONDITIONAL GO。

## 8. Claim boundary

当前允许写：

> An unimplemented proposal to test whether ICL-inspired constituent competition can be adapted to moderation-policy relation proposals under binary video labels.

当前不允许写：

- 首次 compositional/evidence/relation grounding用于 hateful video；
- faithful reproduction或direct adaptation of ICCV 2023 ICL；
- reconstruction使 source、predicate、target获得可识别grounding；
- role-swap discrimination证明有向关系；
- 相比 POWA/LB-SCGP 新增了source accountability semantics；
- 在 HCS上存在与predicate-only不同的完整关系机制。

## Final decision

**STOP before premise and implementation。** 窄来源占用门通过，但 source fidelity与第三门同时失败。当前最可能的有效部分仍是 `ordinary proposal hate score + fixed valid-clause bias`；binary bag labels与固定query bank不能排除这一解。它没有提供相比 POWA、policy-complete P-MIL或旧 LB-SCGP 更强的 temporal relation identification。
