# 淘汰：Vacuity-preserving proposal MIL

> 淘汰原因：独立 novelty review `STOP`（4.8/10）。`softplus` evidence不存在声称的严格零证据
> identity；物理 missing context、observed background 与 latent modality non-ownership 被错误合并；
> negative background supervision还与“无 hate modality保持 vacuous”冲突。Masked SCFE与 evidential
> PCE没有统一代数，只是两个补丁。未实现、未训练、未生成新 test prediction。

截至 2026-08-31；状态：查新/机制审查阶段淘汰；完整裁定见 `NOVELTY_REVIEW.md`。

## 跨任务 adaptation 与最窄 claim

来源一是 P-MIL（CVPR 2023）的 proposal-level SCFE/PCE；来源二是 Cascade Evidential Learning for
Open-World WTAL（CVPR 2023）的 evidential known/unknown/background建模与 subjective uncertainty。
当前检索未见后者用于 hateful-video detection/localization。不能 claim evidential learning、Dirichlet
evidence、subjective logic、proposal MIL、SCFE、PCE、uncertainty-aware fusion或 missing-modality学习。

拟审查的最窄 claim：

> A vacuity-preserving adaptation of proposal MIL for weak hateful-video localization, in which missing
> temporal context and modality-local absence of hate evidence are represented as zero-evidence opinions;
> only non-vacuous proposal opinions may teach completeness, while vacuous modalities remain identity
> elements rather than forced consensus teachers.

Primary sources:

- P-MIL, CVPR 2023:
  https://openaccess.thecvf.com/content/CVPR2023/html/Ren_Proposal-Based_Multiple_Instance_Learning_for_Weakly-Supervised_Temporal_Action_Localization_CVPR_2023_paper.html
- Cascade Evidential Learning, CVPR 2023:
  https://openaccess.thecvf.com/content/CVPR2023/html/Chen_Cascade_Evidential_Learning_for_Open-World_Weakly-Supervised_Temporal_Action_Localization_CVPR_2023_paper.html

## 由当前 developmental test evidence 导出的机制问题

权威分析：`runs/20260831_multimodal_pmil_baseline/pilot_seed234/test_error_analysis.json`，独立审计
PASS。P-MIL port 的 proposal oracle within 在 HMM/HCS 为 `.73952/.63450`，说明候选 interval set 有
充分上限；但 full P-MIL仅 `.58990/.47661`。

- HMM 同 checkpoint去掉 completeness / 只看 CAS 的 diagnostic within为 `.64966/.66263`，说明
  PCE把可用 proposal ordering压坏。
- HCS `48/79` 视频把 whole-video proposal排第一，top proposal中位 213 秒；visual-text/audio-text
  frame-rank Spearman在 `53/67` eligible videos因常数序列无定义。无局部证据的 text view仍被 all-pair
  IRC/PCE当作 teacher。
- 当前 SCFE用 zero padding补视频边界。whole-video proposal的左右 context全为零，真实 inside与零向量
  的差被网络误读为最强 surrounding contrast；“没有 context”被错编码成“明确 background context”。

共同错误是：**missing/absent evidence 被当成普通、确定的特征或 teacher，而不是 ignorance。** 本候选
只改变这个机制，不引入 policy primitive、flow、CRF、teacher model、ensemble或 test calibration。

## 固定模型

仍以审计后的 multimodal P-MIL port为直接 control：同一 corpus-specific frozen MultiHateLoc proposal
producer、proposal集合、RoI bins、branch容量、train/official-val/test protocol和dense readout。

### 1. Masked SCFE，不把 missing context写成零背景

每个 proposal 的 RoI仍分 left/inside/right。除特征外显式计算 `a_L,a_R∈[0,1]`，表示扩展 context 中
真实视频秒的占比。SCFE descriptor改为：

`[a_L*(inside-left), inside, a_R*(inside-right), a_L, a_R]`。

缺失一侧时该侧 contrast严格为零，而不是 `inside-0`；whole-video proposal为
`[0, inside, 0, 0, 0]`。availability是几何事实，不由 corpus/test performance选择，也不直接乘到
最终 score。

### 2. 每 modality/proposal 的 subjective opinion

每个 branch输出非负 evidence `e_h,e_b=softplus(g(x))`。令
`S=e_h+e_b+2`，hate/background belief为 `b_h=e_h/S,b_b=e_b/S`，vacuity为 `u=2/S`；
`b_h+b_b+u=1`。没有 learned evidence时 `b_h=b_b=0,u=1`，不会像 softmax logits那样被迫给 hate或
background概率。frame/localization score只读取 hate belief `b_h`，不把 base-rate `.5` 当 evidence。

Negative train bags对所有 sampled proposals监督 background evidence并惩罚 hate evidence。Positive bags
只要求至少一个 proposal/modality产生 hate belief，同时对未被选择的意见保留 evidence regularizer；
video BCE不广播为每 proposal positive label。Validation仍只选 video checkpoint。

### 3. Vacuity-preserving PCE 与 fusion

对 proposal `I`，modality `m` 的 teacher reliability是 stop-gradient `r_mI=1-u_mI`。PCE teacher
attention用 reliability-weighted opinion：`q_I=sum_m r_mI*b_hmI / sum_m r_mI`；若分母低于固定数值
下界，该 proposal没有 pseudo-instance资格，而不是产生均匀/常数 teacher。仍以 official
`q>.8*max(q)` 与 overlap NMS产生 pseudo instances；positive bags才产生。Completeness head目标仍是
proposal-to-pseudo IoU，但其 loss乘 teacher总 reliability。Negative bags completeness回归0。

删除 all-pair IRC。跨模态 proposal opinion用 cumulative evidence fusion：
`e_h=sum_m e_hm,e_b=sum_m e_bm` 后再转为 `b_h,b_b,u`。vacuous opinion的 evidence为零，因此是严格
identity element；有 evidence的单模态 hate不要求无 evidence modalities复制其 rank。为防单个过度自信
错误 view支配，加入 standard evidential incorrect-evidence penalty，权重固定且 core/controls一致。

最终 proposal score是 fused hate belief乘 fused completeness；frame readout仍为覆盖 proposals的最大
score。constant proposal opinions必须产生时间平坦 score。test不使用 video label。

## 单一机制与必要 controls

核心机制是“vacuity-preserving supervision/fusion”；masked context 与 modality abstention都是同一
ignorance语义在 temporal/context 和 view 两个轴上的实现。必须同跑并独立重训：

1. `pmil_control`：已审计 P-MIL port，softmax+zero-padding+all-pair IRC/PCE。
2. `masked_scfe_only`：只改 context availability，仍用 softmax/all-pair teacher。
3. `evidential_only_zero_context`：只改 opinion/PCE/fusion，保留原 zero-padding SCFE。
4. `core`：masked SCFE + vacuity opinion/PCE/fusion。
5. `forced_nonvacuous`：固定每 modality reliability为1，容量/损失相同，恢复所有 view都可 teaching。
6. `vacuity_time_shuffle`：每视频内打乱 `u_mI` 对 proposal的对应，保持每 modality vacuity边际。
7. `probability_average`：相同 evidential branches，但把期望概率直接平均，检验 identity-preserving
   evidence fusion是否 load-bearing。
8. `no_pce`：保留 evidential fusion但删 completeness training/readout。
9. modality删除、availability循环移位、whole-video top比例、opinion evidence/vacuity分布、pseudo-instance
   modality贡献、constant-output、proposal length/center、score multiset与per-positive-rate strata。

不得按 corpus选择保留 IRC、单 modality、删 completeness或不同 availability rule。不得使用同 checkpoint
删组件 diagnostic替代独立重训 controls。

## 冻结首轮与晋级门

首轮 seed 234，HateMM/HateClipSeg，各自独立 train；official validation只在每个固定 arm内部选 checkpoint，
所有 arms选定后立即跑 test 三指标。test predictions+GT可在正式评测后做 developmental error analysis，
不进梯度/selection。

`core` 必须：

1. 两语料 pooled AP、pooled ROC、within ROC全部严格超过当前 SOTA；
2. 两语料 within均严格超过重新训练的 `pmil_control`、`masked_scfe_only`、`evidential_only_zero_context`、
   `forced_nonvacuous`、`vacuity_time_shuffle`；
3. HCS whole-video top比例从 `.60759` 降到 `<.20`，且 HMM不增加超过 `.05`；
4. HCS text相关常数 frame-score eligible videos从 `53/67` 降到 `<10/67`，同时低 evidence时vacuity显著高于
   shuffled control；
5. masked SCFE与 evidential abstention两个单因素均有双语料一致方向，core还需超过两者，避免靠单个
   padding bug fix或普通 uncertainty fusion冒充统一机制。

任一语料三项 SOTA失败或 attribution gate失败即淘汰，不调 evidence prior、PCE threshold、proposal
generator、readout或按语料路由；不扩 MHC-EN/ZH。

## Reviewer 必须优先阻断

- evidential/uncertainty-aware multimodal hate speech或 hateful-video工作是否已占用该 adaptation；
- “vacuity”是否只是 entropy/confidence gate改名，当前公式是否真的有 identity/abstention语义；
- masked SCFE + evidential PCE是否是两个独立补丁而非同一可检验机制；
- `u` 是否会通过整体缩小 evidence作弊，或一条 modality独占全部 positive bags；
- PCE pseudo-instance仍由自身 opinion生成，是否只是 circular self-training；
- HCS whole-video shortcut是否能由简单 reflection/zero-feature control解释，从而无需主方法；
- Cascade Evidential Learning、UCA、missing-modality learning、JoMoLD/CO2-Net/P-MIL与本候选的准确边界。
