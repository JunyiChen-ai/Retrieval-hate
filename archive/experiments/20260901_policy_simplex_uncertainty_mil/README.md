# 淘汰：Policy-Simplex Background-Uncertainty MIL

淘汰原因：独立 novelty review 为 `STOP 4.3/10`。来源可adapt且未检出被hateful-video task占用，但当前方案删除了来源load-bearing的feature-norm uncertainty与uncertainty-selected pseudo-background，只把background entropy动机与已有POWA policy primitives、MultiHateLoc residual head拼接，Gate 3失败。Prototype permutation还可被learned residual重参数化抵消，不是有效语义control。未实现、未训练、未生成prediction。

截至 2026-09-01。RESET4候选；不运行 premise，直接复用既有 developmental test evidence。

## Failure

权威 test evidence 为 `runs/20260831_multihateloc_test_error_analysis/main/metrics.json` 与 `runs/20260831_test_signal_complementarity/main/metrics.json`。MultiHateLoc 的 fused score 胜过全部单模态的比例仅 HMM/HCS `.345/.154`，best-branch oracle 相对 fused within 缺口均为 `.106`；但固定 lexical+POWA 与 lexical+VERA 的 test 构造分别在 HMM/HCS 达到 all-SOTA，说明本地 policy-semantic ordering 与强 pooled carrier 存在可利用的互补信息。Inference blend、teacher-order KD、routing与calibration均已关闭；缺少的是让单一学生把 policy-semantic locality直接写进 raw frame score、同时把非政策背景排除的训练约束。

## Cross-task source

来源为 Lee et al., *Weakly-supervised Temporal Action Localization by Uncertainty Modeling*（AAAI 2021）。来源把 action background 视作相对于 action-class simplex 的 OOD 样本，以 uncertainty 与 background entropy 抑制背景干扰；其监督包含多 action class video labels。来源论文与代码检索尚未发现被用于 hateful-video detection/localization。

来源：<https://doi.org/10.1609/aaai.v35i3.16280>；<https://arxiv.org/abs/2006.07006>。

## Task adaptation delta

Hateful-video数据只有 binary video label，不能直接套用来源的多 action-class entropy。候选把可执行moderation policy中的六个固定语义 primitive（hostile、target、violence、sexual、self-harm、quoted/condemned context）改造成每秒 **policy simplex**，使用现有、冻结且双语的 semantic prototype directions；prototype不读取test标签，也不按语料选择新prompt。MultiHateLoc三模态local encoders的fused representation输出六个learned residual logits，并与冻结prototype相似度相加。

Positive bag只有top-K秒被要求在该policy simplex上形成低熵、非context-dominant的集中证据；negative bag的全部有效秒被要求对五个harmful primitive保持高熵/低energy，并允许context primitive集中。最终唯一frame logit不是teacher blend，而是同一学生的base hate logit加上policy concentration residual：harmful log-sum-exp减去context logit和归一化entropy。原MultiHateLoc bag BCE保留以维持跨视频pooled discrimination；policy residual直接进入训练与test唯一raw `score_fused`，没有auxiliary bypass、inference ensemble、CDF、threshold或dataset router。

这不是来源的直接移植：来源拥有真实action类别并把各类别概率均匀作为background；本任务没有hate subtype标签，adaptation用固定moderation primitives构造latent policy simplex，显式区分“对任一harm mechanism都不集中”的background与“对某一policy mechanism集中”的局部evidence，并把quoted/condemned context作为有语义方向的反证，而非额外background class。

## Falsification and matched control

Matched control具有相同encoder、六个residual heads、参数量、原MIL和dense negative预算，但把六个primitive identity固定循环置换后再执行同一policy concentration公式；这保留额外head容量与entropy regularization，只破坏moderation semantics。Core必须在HMM/HCS test within同时胜matched control和seed-234 MultiHateLoc anchor，至少一边 `>=+.020`；机制control要求正确prototype相对permuted prototype的增益在两语料同向。最终晋级仍要求HMM/HCS三个固定test指标全部SOTA，之后才扩MHC-EN/ZH与多seed。

方法超参数包括policy residual权重、background entropy权重、prototype strength与temperature；若novelty通过，每语料预先固定12个validation-only trials，以validation within联合选择超参数和checkpoint，锁定后立即正式test。正式运行前只做一次technical review。

## Novelty verdict

独立裁定：`STOP 4.3/10`。Gate 1 PASS；Gate 2 PASS；Gate 3 FAIL。Lee et al.的完整核心是feature magnitude作为action probability、据此选择pseudo-action/pseudo-background并做magnitude separation，background entropy只施加到该uncertainty选出的pseudo-background，不能独立成立。当前brief没有这些load-bearing变量，而是现有typed primitive head加entropy/energy residual，属于Rule 12允许在实现前阻断的直接移植/组件拼接，也贴近ledger的direct-head replacement。不得实现或训练当前版本。
