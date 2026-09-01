# 已淘汰：negative-reference unmatched-mass witness

截至 2026-08-31。本候选已完成 HateMM/HateClipSeg formal test pilot 与独立 post-run audit；
两语料 performance/机制 gate 均失败，已淘汰。下方“当前最小实现/冻结 pilot”保留为运行前协议历史，
最终状态与数字以“正式 test 结果与淘汰结论”及对应 `runs/` 文件为准。

## 候选机制

每个主数据集独立训练，只从本语料 negative train videos 建立 normal reference measure。
对视频的 time×modality token 做 one-sided unbalanced optimal transport，以每个 source token
被消除的质量作为逐帧 witness，并由同一 witness 构造 video-label likelihood。原意是用可变
unmatched mass 代替 MultiHateLoc 的固定 top-K 与把正标签复制给所有模态分支的做法。

## 独立查新结论

结论为 `CONDITIONAL GO`，novelty 仅 `4.5–5/10`，不足以进入实现。最接近占位包括：

- POT-OSSL（IJCAI 2024）：reference distribution 与 transported mass 的 instance OOD score；
- Outlier-Robust OT（AISTATS 2021）：由被移除质量定位 outlier；
- partial OT for PU learning（NeurIPS 2020）：只运输匹配子集并丢弃剩余样本；
- NG-MIL（WACV 2023）：从 normal videos 学 normal prototypes 做弱监督视频定位；
- MG-TVMF、ICASSP 2026 OT-WSVAD：OT 已进入弱监督视频异常定位；
- VALOR/MMIL：弱 modality-less video labels 下的 time×modality ownership；
- ASOT（CVPR 2024）：UOT 用于逐帧时间分割。

候选最多只能主张一个很窄的 structured MIL application，不能主张 OT、unmatched-mass
anomaly score、normal reference 或 multimodal ownership 本身新。若模态容量独立，目标严格
分解为多个 one-class detector；若允许跨模态运输，又会产生没有语义依据的错误匹配。固定
运输量使总 unmatched mass 恒定，固定容量则重新引入隐式事件比例；使用 log-sum-exp 或
noisy-OR 后还必须排除普通 MIL pooling 才是真实增益来源。

## 新 novelty 标准下的裁定

此前因“不足够从零原创”作出的 `STOP_BEFORE_IMPLEMENTATION` 已被新 novelty 标准覆盖。UOT、
normal-reference transport 和 unmatched-mass scoring 虽有相邻任务先例，但查新未发现它们已用于
hateful video detection / localization。本候选因此重新开放，但只有下面这一项窄 claim 可进入
实现：以 negative-only normal reference 解释每个 time×modality token，并把不能被解释的运输质量
作为同一个局部 witness，同时驱动时序定位与 video-label likelihood。

这不是直接套用：核心任务机制是处理 hateful video 中“正 video label 被错误广播到全部时间和全部
模态”的监督噪声，并用共享容量与时间约束防止它退化成独立 one-class detector、固定异常比例或普通
MIL pooling。正式实现必须包含 nearest-normal、独立 UOT、无时间约束和普通 pooling controls；若
不能证明 shared witness 的归因或两主数据集 test 定位提升，则淘汰。

## 当前最小实现（尚未获正式 run 授权）

- `model.py`：每个 1 fps audio/visual/text token 都带固定单位 source mass，经各自 projector 后只能
  匹配同一个 latent normal atom bank 或 reject；normal atom 的容量由全部 time×modality token 共享。
  三个 projector 共享 atom identity，避免把三个可独立置换的 atom 编号任意耦合。每秒 witness 是
  三模态 reject fraction 的并集概率。
- normal references 只接收 negative train videos 的梯度；positive bags 可以推动 encoder 暴露无法由
  normal reference 解释的 token，但不能把 reference 本身改造成 positive prototype。
- 同一个 reject witness 经固定 power-mean 形成 bag probability，并用于最终帧分数；没有独立 video
  classifier、test routing、ensemble、固定 top-K 或事件占比。
- loss 只含 video-label BCE、witness temporal penalty、negative-only reference compactness。每个语料
  独立训练；validation video AP 仅在该次固定训练中选择 checkpoint，随后立即生成 test prediction，
  `evaluate.py` 只调用共享 evaluator。

目前通过 syntax、synthetic gradient/data-shape/短优化 CPU smoke、四语料 frozen evaluator cohort
审计与独立 pre-run review。不得把这一状态描述为完成了一轮方法迭代。

## 冻结 pilot 与 gate

首轮固定 seed 234，先独立训练/评测 HateMM 与 HateClipSeg。每个 checkpoint 同时输出 core、
去掉跨模态共享容量的 independent-transport control，以及无容量竞争的 nearest-normal control。
若任一语料 core 未严格超过当前三项 test SOTA，或 core within-video ROC 未同时超过两个 attribution
controls，则本轮淘汰，不运行 MHC-EN/ZH，不围绕 test 数字调参。只有双语料 gate 全过才扩展四语料
与多 seed。

独立 pre-run review PASS 后，每个训练任务与 SSH 解耦：

```bash
mkdir -p runs/20260831_uot_normal_reference_witness/pilot_seed234/hatemm
setsid /home/jehc223/miniconda3/envs/HateVideo/bin/python \
  experiments/20260831_uot_normal_reference_witness/train.py \
  --corpus hatemm \
  --output-dir runs/20260831_uot_normal_reference_witness/pilot_seed234/hatemm \
  > runs/20260831_uot_normal_reference_witness/pilot_seed234/hatemm/run.log 2>&1 < /dev/null &
echo $! > runs/20260831_uot_normal_reference_witness/pilot_seed234/hatemm/run.pid
```

训练明确输出 `complete` 后才运行对应 `evaluate.py`，日志和 PID 写入同一 run 目录；HateClipSeg
使用相同命令，仅替换 corpus 与目录。权威数字只认各目录的 `metrics.json`。

## 正式 test 结果与淘汰结论

权威输出：

- `runs/20260831_uot_normal_reference_witness/pilot_seed234/hatemm/metrics.json`
- `runs/20260831_uot_normal_reference_witness/pilot_seed234/hateclipseg/metrics.json`

独立结果审计 `POST_RUN_REVIEW.md` 结论为完整性 `PASS`、pilot `FAIL`。

- HateMM core AP/ROC/within 为 `.51833/.75187/.58290`，三项均低于 SOTA；independent 与
  nearest controls 的 within 分别为 `.60896/.59578`，均高于 core。
- HateClipSeg core 为 `.58172/.53788/.51963`，三项均低于 SOTA；independent within
  `.53225` 高于 core，nearest 为 `.51417`。

两语料的三指标 SOTA gate 和 shared-capacity attribution gate 全部失败，故本轮淘汰，不扩展
MHC-EN/ZH，也不围绕本轮 test 数字调整 transport 参数。

允许的 test error analysis 进一步显示：shared 相对 independent 的 per-video AUC 损失集中在高
正例占比一半的视频，HMM/HCS 平均分别为 `-.0442/-.0243`；低占比一半仅 `-.0084/-.0013`。
这说明跨模态共享 normal capacity 在长 hate 段制造额外 rejection、压平局部排序。该发现只用于
否定 shared-capacity coupling，并促使下一候选改用不依赖固定正常容量的 multi-scale temporal
adapter；不用于继续调当前 UOT。

若进程在训练完成前中断，使用完全相同参数并追加 `--resume`；每个 epoch 的原子 training state
保存在同一 run 目录。`training_complete.json` 标记训练与 prediction 的状态边界；只有状态为
`prediction_complete` 时 `evaluate.py` 才接受该 run。
