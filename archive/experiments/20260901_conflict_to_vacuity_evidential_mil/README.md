# 淘汰：Conflict-to-Vacuity Evidential MIL

淘汰原因：独立 novelty review 为 `STOP 5.0/10`。前两门通过，但 core 的 conflict-to-vacuity 等同于已有 Yager combination rule；其余部分主要是 subjective-logic fusion 向 temporal MIL 的直接迁移，尚未形成针对异步 hateful carrier 的非平凡机制改造。未实现、未训练、未生成 prediction。

截至 2026-09-01。RESET4 candidate 1；不运行premise。默认starting architecture为MultiHateLoc。

## Failure

权威四语料test artifact为`runs/20260831_multihateloc_test_error_analysis/main/metrics.json`。DMS最高权重与test-GT最佳单模态匹配率仅HMM/EN/ZH/HCS `.216/.333/.375/.323`，并几乎总选visual；fused胜全部单模态的seed-video比例仅`.345/.159/.042/.154`，best-branch oracle相对fused within缺口`.106/.171/.211/.106`。`score_union`四语料也接近无效，因此问题不是缺少简单max/union，而是现有fusion强制每个模态形成确定分数，却没有表示“该秒该模态没有可用hate判断”的状态。

## Cross-task source

来源为Han et al.的Trusted Multi-View Classification（ICLR 2021；TPAMI 2023 dynamic evidential fusion）。来源对每个sample/view输出Dirichlet evidence，以subjective opinion表示belief与uncertainty，再用Dempster-Shafer evidence-level fusion；检索尚未发现该方法用于hateful-video detection/localization。来源解决fully supervised sample-level multi-view reliability，不包含弱video标签、逐秒定位、异步carrier或temporal MIL。

来源：[ICLR 2021 paper](https://arxiv.org/abs/2102.02051)；[TPAMI dynamic evidential fusion](https://arxiv.org/abs/2204.11423)。

## Non-trivial task adaptation

Visual/audio/text三个MultiHateLoc local encoders每秒分别输出二类Beta/Dirichlet evidence，形成`benign belief / hate belief / vacuity`。本任务不同于来源的co-labelled views：speech slur、visual symbol和text/OCR可以异步出现，无贡献模态不应被迫投票benign或占据softmax权重。

Core使用**conflict-to-vacuity**逐秒融合：两意见的一致belief按subjective-logic合取；互相冲突的hate-vs-benign质量不按标准Dempster rule除以`1-conflict`重新放大，而保留为该秒的额外vacuity。递归融合三模态后，唯一frame score是最终hate belief；没有router、modality quota、branch ensemble、test calibration或后处理。Negative bags对最终opinion提供dense benign supervision；positive bags只对最终hate belief做top-K MIL，单模态head不接独立positive bag loss，防止三个branch各自广播positive标签。

这使“不知道”成为load-bearing final state：一个无本地证据的模态可保持vacuous而不覆盖真实carrier；多个一致carrier会降低vacuity并增强hate belief；互相矛盾时模型不能靠Dempster normalization制造高置信度。

## Control and falsification

Matched control使用完全相同的encoders、evidence heads、参数量、MIL与dense-negative监督，但用来源标准normalized Dempster fusion；core唯一差别是将conflict质量转入vacuity而非除掉。Core必须在HMM/HCS test within同时胜标准Dempster control与MultiHateLoc anchor，并至少一边`>=+.020`；最终晋级仍要求两语料三个固定指标全部SOTA。若失败，不调evidence prior、KL权重、fusion order或dense-negative权重续命。

## Novelty gates

独立 review verdict：`STOP 5.0/10`。

- Gate 1：PASS，来源方法可以 adaptation。
- Gate 2：PASS，未检出来源方法已用于 hateful-video detection/localization；但已有 hateful-meme / multimodal hate 的近邻 evidential precedent。
- Gate 3：FAIL。把冲突质量转入 vacuity 是已有 Yager rule，不是本项目产生的 task adaptation；剩余改动是逐秒 evidence head、temporal MIL 与 dense-negative supervision 的直接组合，没有把异步 carrier 结构写成新的、load-bearing 的 fusion constraint。

因此本候选在 novelty 门停止，不进入实现、technical review、训练或 test；RESET4 正式方法失败计数保持 `0/3`。
