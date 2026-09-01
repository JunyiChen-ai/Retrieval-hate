# Coalition witness candidate

> 淘汰原因（2026-08-31）：HateMM 与 HateClipSeg 的 test mechanism gate 和三指标
> SOTA gate 全部失败；不扩展到 MHC-EN/MHC-ZH，也不围绕 test 结果调这个机制。

截至 2026-08-31。独立 novelty / anti-pattern 评审为 `CONDITIONAL GO`；最小 pilot
已经完成并淘汰。权威汇总：
`runs/20260831_coalition_witness_candidate/pilot_seed234/metrics.json`。

候选研究问题：能否把 Möbius-decomposed modality interaction atom 作为弱监督 temporal
MIL 的 `(time, minimal modality coalition)` latent witness，同时让 full-modality temporal
score 成为同一分解的唯一可重构输出。

独立查新结论见 [NOVELTY_REVIEW.md](NOVELTY_REVIEW.md)：`CONDITIONAL GO`。只有先解决
正负 interaction、null intervention、minimality 与非单调语义、以及 exact full-score
reconstruction，才授权进入最小 pilot；当前不构成已验证方法或 novelty claim。

## 冻结机制定义

不对任意神经 score 强加错误的硬 antichain。这里显式定义一个单调的 positive-evidence
availability game。对每个时间 `t` 和非空 modality coalition `R`，同一个共享网络只读取
`R` 中的模态并输出 atom log-intensity `a_t(R)`，令：

```text
h_t(R) = exp(a_t(R) / tau)
v_t(S) = sum_{nonempty R subseteq S} h_t(R)
```

因此 `h_t(R)` 正是 `v_t` 的非负 Möbius atom，且 full-modality test score 唯一定义为：

```text
s_t = tau * log(v_t({V,A,T}) / 7)
    = tau * logmeanexp_R(a_t(R) / tau)
```

`tau` 固定复用该 corpus 的 baseline temperature；它在 atom intensity、bag likelihood、
posterior 与 test reconstruction 全链一致。不存在独立 fused head。正/负 video 的唯一
bag logit为 `tau * logmeanexp_(t,R)(a_t(R)/tau)`；正例条件 posterior 是这些 `(t,R)` 上
与同一 intensity 成比例的 categorical distribution，表示
一次只由一个 latent witness 解释 bag decision。负例同一 likelihood 会压低全部 atoms。
test 只输出 `s_t`，不读取 posterior 做 routing。

这一定义主动放弃 suppressive/negative interaction，以换取可审计的 monotone evidence
availability。它不是对“加入模态后 hate semantics 必然单调”的主张；若该限制造成性能失败，
pilot 应直接淘汰，不得事后改 null、改符号或按语料选择 coalition。

## 固定最小 arms

HateMM 与 HateClipSeg 分别独立训练，validation 只在每个固定 arm 内选 checkpoint；选定后
立即跑 test 全部三个指标，方法比较与后续决策只看 test。
每个 corpus 的所有新 arms 逐项复用其 seed-234 official MultiHateLoc 冻结训练配置
（学习率、训练轮数、top-K、smoothness、hidden/embed、dropout、temperature）；不通过
validation 在 arms 或新超参数间选择。`no_infonce` 的唯一变化是把原配置的
`lambda_contrast` 设为 0。

1. 既有 MultiHateLoc starting point；
2. `no_infonce`：只删除 unconditional InfoNCE；
3. `all_subset_mil`：相同共享网络、全部 coalition forward，但直接对每个 subset 复制 bag
   label；无 Möbius reconstruction、无 latent coalition；
4. `synib`：full coalition MIL + missing-one ranking penalty；
5. `mobius_nonminimal`：同一 nonnegative atoms 和 exact full-score reconstruction，但只在
   重构后的 temporal score 上做普通、已冻结的 corpus-specific top-K MIL：HateMM
   `ceil(T/8)`，HateClipSeg `ceil(T/3)`；
6. `coalition_witness`：完整 `(t,R)` categorical latent likelihood。

机制失败门：完整候选必须在 HMM/HCS test within-video ROC 都超过 arms 3–5，且提升不能只由
arm 2 解释。项目晋级门：两个 corpus 的 pooled AP、pooled ROC、within ROC 必须各自全部
超过 `STATUS.md` 冻结 SOTA；否则不晋级，直接记录并归档。

**运行前修正记录：**上面的 corpus-specific top-K 复用是为了让所有 arms 与各 corpus 的
official MultiHateLoc starting point 保持同一冻结训练配置；它覆盖 `NOVELTY_REVIEW.md`
最小 pilot 中把 arm 5 统一写作 “top-third MIL” 的旧措辞。本修正在任何正式训练或新 test
prediction 之前完成，不由 validation/test performance 触发。

## Test 结果与结论

以下全部是 seed 234、1 fps test evaluation；validation 只在每个固定 arm 内选择
checkpoint，没有用于方法比较或设计。表内顺序为 pooled AP / pooled ROC / within-video ROC。

| corpus | no InfoNCE | all-subset MIL | SynIB | Möbius nonminimal | coalition witness | 冻结 SOTA |
|---|---:|---:|---:|---:|---:|---:|
| HateMM | .5025 / .7418 / .6297 | .5320 / .7510 / .6136 | .5024 / .7371 / .6308 | .5346 / .7516 / .6338 | .5367 / .7570 / .6271 | .5938 / .8162 / .6315 |
| HateClipSeg | .5159 / .5071 / .5291 | .5483 / .5303 / .5303 | .5950 / .5596 / .5363 | .5643 / .5509 / .5365 | .5403 / .5148 / .5235 | .6194 / .6050 / .5619 |

完整候选在两个语料都没有超过 mechanism controls：HateMM within `.6271` 低于
Möbius nonminimal `.6338`，HateClipSeg `.5235` 低于 SynIB `.5363` 和 Möbius
nonminimal `.5365`。两个语料的全部三项 SOTA gate 也都失败。因此
`mechanism_pass_both=false`、`sota_pass_both=false`、`continue_to_four_corpora=false`。

test posterior 诊断只用于解释失败，不用于修改本候选。HateMM 平均 posterior mass 为
singleton `.497`、pair `.154`、triple `.349`；HateClipSeg 为 `.138/.345/.517`，说明
两个语料学到的 coalition ownership 分布差异很大。full-score reconstruction 最大绝对残差
在两者均为 `0`，所以失败不是重构或评测实现错误，而是 categorical single-witness 与
nonnegative monotone evidence 假设没有带来可迁移的 temporal localization 增益。
