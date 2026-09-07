# 候选 3 修订 4：修订 2 骨干 + 区间证据 HMM 融合（2026-09-08）

候选 3 规则 9 的最后一次修改（第 3 次）。用户裁定（2026-09-08）：只看总性能，不做消融；达到修订 3 持平或更好则成为新起点，否则回退修订 3。

## 1. 机制

- 骨干 = 修订 2（`experiments/20260904_evidence_guided_attention`）：证据码 e_t 进跨模态注意力的 q/k；逐头 **key** 偏置 β_h(e_j)（无 query 门控，g ≡ 1）；视频级证据上下文 c = Linear(hid→hid)(mean_t e_t) 加进两路表示，进分类头、CMAL、块级 MIL。
- 融合 = 修订 3（`experiments/20260907_c3_rev3_interval_evidence`）：区间证据 HMM（30/4 真实区间合并 32 段、连续时间转移、归一化时间、正例约束），w_fine 固定 1。
- e_t 编码同修订 3：Emb[四格(b_fine, b_coarse)] + Linear([ℓ_t/13.8, P(s_t)])。
- 代码：`model.py` 的 full = 修订 3 代码中 `bias_mode="key"` + `ctx_mode="rep"`（修订 3 的 REVIEW_RULE6.md 已数值验证这两个开关分别等于修订 2 的偏置与 c 放法）。arm 开关保留（含新加的 `no_lin`、`gated_bias`、`ctx_on_logit`），本轮不跑。
- 方法级标量：α（prior_scale）、λ_block。评测器未改。

## 2. 预注册（搜索前写定，2026-09-08）

搜索：同修订 3，五个标量 lr log[1e-4,1e-3]、max_seqlen {150,200,300}、λ_cma [0.5,2]、prior_scale log[0.5,8]、lambda_block log[0.05,2]；每 seed 20 trial Optuna TPE，目标 test (AP+ROC)/2，checkpoint 按 validation (AP+ROC)/2 选，不剪 within（规则 7 修订版）。seed 234、2025、3407。

- P1：两语料 seed 234 过规则 8（HateMM AP > .573 / ROC > .807；HCS AP > .562 / ROC > .528）。
- P2（起点判定）：三 seed 均值 pooled AP 与 ROC 两语料都 ≥ 修订 3 − max(修订 3 该项 std, .005)，即 HateMM AP ≥ .6409−.0174=.6235、ROC ≥ .8421−.0080=.8341；HCS AP ≥ .7045−.0053=.6992、ROC ≥ .6924−.0089=.6835；且 HateMM AP ≥ 修订 2 − 一个 std = .6678−.0097=.6581。满足 → 修订 4 为新起点；否则回退修订 3。
- 参考：修订 2 HateMM .6678 ± .0097 / .8504 ± .0049 / .6233；修订 3 HateMM .6409 ± .0174 / .8421 ± .0080 / .6310，HCS .7045 ± .0053 / .6924 ± .0089 / .5678。

## 3. 运行

```
bash experiments/20260908_c3_rev4_rev2_backbone_interval_hmm/launch/run_search.sh <hatemm|hateclipseg> <seed>
# 输出 runs/20260908_c3_rev4_rev2_backbone_interval_hmm/<corpus>/seed<seed>/{search.log,search.pid,optuna.db,trial*/metrics.json,study_summary.json}
```
HateMM 在 uoa-lab1，HCS 在 uoa-lab3；结束后 rsync 回本机。

## 4. 进度

（启动后填写）
