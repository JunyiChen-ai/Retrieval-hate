# Coalition witness candidate：独立正式运行前评审

日期：2026-08-31  
阶段：任何正式训练和新 test prediction 之前  
结论：**PASS，可以启动 `launch.sh` 的两语料 seed-234 pilot。**

本评审覆盖 `README.md`、`NOVELTY_REVIEW.md`、`model.py`、`train.py`、
`test_model.py`、`supervise.sh`、`launch.sh`、`summarize.py`，并追到直接使用的
MultiHateLoc data/model/train 实现与唯一共享 frame evaluator。评审期间没有启动正式训练、
checkpoint selection 或 test inference；正式 run 目录尚不存在。

## 运行前发现并修复的问题

1. **`no_infonce` 初版不是单变量 deletion control。** 初版只设置
   `lambda_contrast=0`，其余参数落到脚本默认值，与 summary 中使用的 official
   MultiHateLoc starting point 不同。修订版对 HateMM/HateClipSeg 分别显式复用其 seed-234
   official 配置的学习率、epoch、top-K、smoothness、hidden/embed、dropout 和 temperature，
   `no_infonce` 唯一变化是 contrastive weight 置零；summary 会重新核对 producer train args。
2. **初版 latent likelihood 与 test reconstruction 使用了不同的 temperature 量。** 修订版
   全链统一为 `h_t(R)=exp(a_t(R)/tau)`：worth、full temporal score、bag logit 和
   posterior 都使用同一 `a/tau` intensity。full test score固定为
   `tau * logmeanexp_R(a_t(R)/tau)`，bag logit固定为
   `tau * logmeanexp_(t,R)(a_t(R)/tau)`，posterior 固定为同一 intensity 的 categorical
   distribution。
3. **初版 summary 只看 performance，可能在机制完整性失败时仍给出晋级。** 修订版对每个
   arm 的 corpus、arm、seed、modality order、split policy、test isolation 和全部冻结参数
   做可读字段检查；对两个 Möbius arms 强制 reconstruction residual 为 finite 且不超过
   `1e-5`；对完整候选强制 posterior diagnostics 覆盖完整 test cohort，并检查七项 mass、
   归一化、MAP subset/time 与 atom summaries。
4. **arm 5 的旧文档把所有语料写成 top-third。** 当前冻结实现实际复用 corpus-specific
   starting-point top-K：HateMM 为 `ceil(T/8)`，HateClipSeg 为 `ceil(T/3)`。README 已在
   正式运行前明确记录该修订，并声明覆盖 novelty review 中的旧措辞。

## 严格 novelty 闭环

结论：实现与 `CONDITIONAL GO` 所授权的严格版本一致。

- 三个 modality encoder 后只有一个共享 `coalition_head`；七个非空 subset 使用同一 scorer，
  只改变可见 embedding 和显式 availability bits。
- subset scorer 对未提供模态严格不敏感：未提供模态在 head 输入处固定为零，availability
  bit 同步标识；合成测试对每个 subset 任意扰动所有 absent features，目标 subset logit
  逐元素保持完全一致。
- `h_t(R)=exp(a_t(R)/tau)` 恒为非负。`v_t(S)` 由所有 `R subseteq S` 的 atom intensity
  求和，因此构成单调 evidence-availability game；空集贡献固定为零。
- 对合成 worth lattice 做完整 Möbius inversion 后，七个 recovered atoms 与原 intensity
  一致；full worth 与 reconstructed temporal score 的等式通过测试。
- `coalition_witness` 的唯一正/负 bag likelihood 在全部合法 `(t,R)` 上形成一个 latent
  categorical witness；没有额外 branch loss、routing target 或 test-time selector。
- test prediction 只写一个 `score_full`，它来自七个 atoms 的 full reconstruction。代码中
  不存在独立 fused head；posterior 只写 diagnostics，不进入 test score、routing、ensemble、
  calibration 或 transport。

本实现明确只建模 nonnegative evidence availability，主动放弃 suppressive interaction；因此
posterior 是模型内的 witness allocation，不得解释为真实 causal ownership。该限制若导致 test
失败，应淘汰机制，不能事后改 sign/null 或按语料选择 coalition。

## Arms 与归因公平性

- `multihateloc`：读取既有 seed-234 test evaluator artifact 的 primary `score_fused`。
- `no_infonce`：原 MultiHateLoc 架构和 corpus-specific frozen config 不变，只删除
  unconditional InfoNCE。
- `all_subset_mil`：与候选相同的 encoders、共享 subset scorer、七个 subset forwards 和参数量；
  七个 subset 各自接受普通 top-K bag label，无 Möbius reconstruction 或 latent coalition。
- `synib`：相同七 subset forwards；full subset 使用普通 MIL，三个 missing-one subsets 正确对应
  bitmask `6/5/3`，只在 positive bags 上施加固定 margin ranking penalty。
- `mobius_nonminimal`：与候选使用相同 nonnegative atoms、temperature 和唯一 reconstructed
  full temporal score，但训练目标是 frozen corpus-specific top-K MIL，没有 `(t,R)` latent
  likelihood。
- `coalition_witness`：唯一差异是完整 joint `(t,R)` latent objective。

四个新 arms 使用相同 seed、同一 corpus 的冻结 optimizer/architecture/epoch/top-K/smoothness
配置，并从相同初始化与同一 train ordering 独立训练。不存在按 validation 或 corpus 结果选择
arm、modality、subset、null 或 test branch 的代码。

## Loss、mask、padding 与真实 batch 检查

- `topk_counts=ceil(T/k)`，且不超过合法长度；排序前 padded positions 被固定为极小值，取出的
  value 和 denominator 只覆盖合法行。
- smoothness 只使用相邻两行都合法的 pair；长度不足时返回可反传的零。
- latent log-sum-exp 在 mask 后只覆盖合法 `(t,R)`，normalizer 使用每视频真实
  `length * 7`。
- 所有 test frame arrays 在截断到真实长度后写出；padding score 固定为零。
- modality 顺序在 data、dims、bitmask 和 model 中均为 `visual, audio, text`，构造器会拒绝
  其他顺序。

实际读取 HateMM 与 HateClipSeg train feature 的首个双视频 batch，并按各自正式冻结 hidden、
embed、temperature 与 top-K 配置运行四个新 arms。八次 forward/backward 均得到 finite loss、
finite gradients，24 个参数 tensor 均有非零梯度；frame scores 全 finite，padded outputs 全零。
HateMM 与 HateClipSeg 模型参数量分别在四个新 arms 间完全相同。

仓库自带 `test_model.py` 通过；`py_compile`、两个 shell 的语法检查均通过。

## Train、validation、test 与 evaluator

- HateMM 和 HateClipSeg 分别建立独立 model、optimizer、loader、checkpoint 与输出目录；没有
  跨主语料训练数据。
- official train 只用于梯度；official validation 只在当前 arm 内按 video AP 选择 checkpoint，
  不参与 arm comparison、threshold、超参数或后续设计选择。
- test GT 不参与梯度或 checkpoint selection。checkpoint 选定后立即生成完整 test score，随后
  由 `scripts/reproduction_baselines/eval_baseline_scores.py` 以 full-coverage 模式输出 pooled
  AP、pooled ROC-AUC 和 within-video macro ROC-AUC。
- evaluator branch 对 `no_infonce` 固定为 `score_fused`，对四个新 arms 固定为唯一
  `score_full`；summary 使用相同 branch 和路径。
- `launch.sh` 使用 `setsid nohup` 与 SSH 解耦，顶层及各 arm 均记录 PID/log；`set -euo
  pipefail` 保证任一训练或评测失败时停止，不生成最终 summary。
- 每个新 arm 产生 checkpoint、scores、config、代码版本说明、train record、run log 和共享
  evaluator metrics；`no_infonce` 的 checkpoint/scores/train args 位于其 producer 子目录，
  顶层保留 log、PID 和 metrics。pilot 根目录只在全部 arms 成功并通过 summary integrity 后
  写最终 metrics/config/代码版本说明。

## Summary gates

- 机制门：完整候选在 HateMM 和 HateClipSeg 的 test within-video ROC 上都必须严格超过
  `all_subset_mil`、`synib`、`mobius_nonminimal`，并严格超过 `no_infonce`。
- SOTA 门：候选在两个语料各自的 pooled AP、pooled ROC 和 within ROC 上必须全部严格超过
  `STATUS.md` 当前冻结阈值。
- 只有两语料机制门与 SOTA 门同时通过，`continue_to_four_corpora` 才为真。
- summary 的 baseline、新 arm metrics 路径和 evaluator branches 与 supervisor 的实际输出
  一致；test missing/extra、cohort size、配置、reconstruction 和 posterior diagnostics 都是
  fail-closed 条件。

用临时目录构造完整的两语料、五个新 arm artifact tree 后，`summarize.py` 的读取、完整性门、
branch 路由、严格比较和最终原子写出均正常；该测试未写入项目 `runs/`。

## 内容校验限制

实验目录及本轮 runner、model、tests、summary 不计算、不记录、不比较也不依赖任何文件内容
校验标识。provenance 只使用可读路径、配置字段、日期、代码版本说明与任务级解析/shape/
coverage/invariant 检查。

## 最终授权

**PASS。** 当前未发现会改变 pilot 观察、归因或晋级结论的实现问题。可以运行：

```bash
bash experiments/20260831_coalition_witness_candidate/launch.sh
```

正式输出属于 iterative/developmental test evidence。若两语料任一机制门或任一 SOTA 指标失败，
必须按冻结规则停止并记录负结果，不得按语料选择 arm/subset 或修改 gate 后重解释。

