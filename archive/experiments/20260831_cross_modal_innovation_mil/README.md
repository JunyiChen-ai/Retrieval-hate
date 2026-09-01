# Cross-modal innovation MIL — 淘汰：双语料 matched predictor 均未胜 unconditional mean

截至 2026-08-31。候选已在冻结的 train-only OOF premise gate 淘汰；未训练正式 localizer，
未生成 validation/test prediction。

## 最终结论：STOP_BEFORE_FORMAL_LOCALIZER

独立 novelty review 对修订版给 `GO 5.8/10`，独立 pre-run code/protocol review 给 `PASS`。随后
按冻结参数运行 HMM/HateClipSeg 三折 train-only OOF conditional prediction。权威输出：
`runs/20260831_cross_modal_innovation_mil/premise_seed234/analysis.json`。

- HMM aggregate micro Huber：matched `.37571343`，unconditional mean `.37560863`，shuffled
  `.39317686`；matched 没有胜 mean。
- HateClipSeg：matched `.36542560`，mean `.36231313`，shuffled `.37229636`；matched 同样没胜
  mean。
- availability/missing-channel 与 shuffled pair/time alignment contract 均通过，两个语料的
  shuffled error 也都高于 matched；失败项是核心 conditional prediction 连固定均值都没有
  改善，而不是 mask 或 correspondence control 失效。

按预注册 premise kill，不启动 `core/shared/raw/same-modal/availability-only` 等正式 arms，不读取
validation/test，不扫描 PCA width、context radius、predictor capacity、epoch 或 loss。当前证据否定
“固定跨模态 conditional residual 在两语料形成共同可用的候选证据分解”这一前提；不得改成只做
audio、按语料选 modality，或把 shuffled gap 单独包装成机制成功。

## 直接失败证据

MultiHateLoc 的 DMS 在 HCS seed×video 中几乎总把 visual设为最高权重，但 test-GT 最佳
单模态分布为 audio 63、visual 66、text 72；DMS 与最佳模态匹配率只有 `.323`，fused
branch 超过全部单模态 branch 的视频比例只有 `.154`。HMM/HCS best-branch oracle相对
fused 的 within ROC缺口均约 `.106`。这说明统一 cross-modal fusion 经常把 carrier-specific
证据稀释，而不是缺少可候选的模态分数。

现有文本特征又有约 18% 秒没有 ASR覆盖；零向量目前作为普通输入进入融合。候选必须区分
“模态缺失”与“模态存在但其他模态不能预测的 private evidence”。

依据：`runs/20260831_multihateloc_test_error_analysis/main/metrics.json`，以及
`/home/jehc223/Hate-follow-up/results/reproduction/features/bert_sentence_1fps/{hatemm,hateclipseg}/index.json`。

## 跨任务来源方法

主要来源是 Gabeur et al., WACV 2022, *Masking Modalities for Cross-Modal Video Retrieval*：
在 appearance、audio、transcribed speech 中遮掉整个 modality，并用其余 modalities预测它，
从而训练协作的 multimodal video encoder。相关但更近的表示方法包括 CrossMAE（CVPR 2024）
的 cross-conditioned/cross-embedding reconstruction。

当前检索尚未发现 masked-modality prediction/reconstruction 被用于 hateful video detection
或 localization；必须由独立 reviewer 实际查证。普通 cross-modal attention、MoE或
modality dropout不等于该来源方法，但 masked reconstruction、shared/private disentanglement
本身都不能单独 claim novelty。

## Non-trivial task adaptation

### 1. 逐秒 conditional prediction，而非 whole-video retrieval pretraining

每个目标语料独立在其 train split 上先拟合三个冻结的 PCA-64 whitening 变换
`W_m`。PCA 的输入均值、分量与尺度只由该语料 train split 的真实可用秒确定；它不是可学习
projection，也不与 predictor 或 MIL 联合更新。对目标 modality `m`：

`shared_m(t) = F_m({x_n(t-k:t+k), availability_n}_{n != m})`

预测该秒固定的非退化目标 `z_m(t)=W_m x_m(t)`。只在目标 modality真实可用的秒计算
Huber reconstruction loss；ASR uncovered秒不参与 text target loss，也不作为零值负证据。
predictor不得读取 video label、video ID、test/validation输入或目标 modality自身。

训练完成后 predictor冻结。用 train split 的 reconstruction residual逐维均值与标准差定义冻结的
标准化变换，得到可观测 innovation：

`private_m(t) = (z_m(t) - stopgrad(shared_m(t)) - mu_res,m) / sigma_res,m`。

禁止对每一帧的 residual 做 LayerNorm：它会抹掉 residual magnitude，并可能把 predictor error
放大成等幅证据。固定 PCA target 同时排除 learned projection 与 predictor 一起收缩到零的全局
退化解。

这里不声称 `private=hate`。它只表示“给定其他 modalities和局部时间 context仍不能解释的
目标模态信息”；这个 operational definition由 train-only reconstruction确定，不由 latent
frame label定义。

### 2. shared 与 private 均保留，避免把 cross-modal disagreement等同异常

来源 masked-modality方法倾向把可跨模态预测的信息编码进共同表示。本任务不能只用
reconstruction error作 hate/anomaly score：罕见但 benign的音频/视觉也会不可预测。因此最终
localizer同时读取每个 modality的 observed `z_m`、`shared_m`和`private_m`，由 video-label
MIL学习证据方向；private通道不能直接加到 frame score，也不按 residual norm打分。

融合采用一个共享参数的 evidence head分别产生 `observed`、`shared` 与 `private` logits，再以
availability-aware log-mean-exp形成单一 frame posterior：

`frame_logit(t) = logsumexp({ell_c,m(t)} over available channels) - log N_available(t)`。

missing modality的对应 logits 全部被 mask 掉。这里必须减去可用通道数的对数，防止模型仅凭
该秒可用 channel 数量改变分数。所有输入同为 64 维；不得给 visual单独更大容量。

### 3. 为什么针对 hateful temporal localization

仇恨证据可能只存在于一个 carrier：spoken slur只在ASR/audio，视觉符号只在visual，语气只在
audio。普通协同融合把“可被其他模态确认”误当成“更可信”，这在 test oracle中已被否定。
conditional prediction把共同 topic/scene和carrier-specific innovation分成两个可观察通道；
MIL仍决定哪个通道是hate，因而既能保留真正cross-modal关系，也不会因缺少共识而删除单模态
证据。

最终test只做一次原始输入forward，输出一个连续frame posterior；无branch选择、oracle routing、
ensemble、calibration、threshold search或post-hoc smoothing。

## 固定 pilot 与 controls

### 正式 localizer 前的 train-only premise gate

先用目标语料 train videos 做固定 video-level OOF 检查，不读取 validation/test：

1. 固定 PCA target 没有可训练的收缩路径，且各维训练方差非零；
2. HMM/HCS 上，matched cross-modal predictor 的 aggregate held-out Huber error 均低于逐模态
   unconditional-mean predictor；将 conditioning videos 在相同 availability pattern 内打乱后，
   error 在两语料都回升；
3. constant logits 在不同 availability pattern 下经过 masked log-mean-exp 输出完全相同；
4. missing target 的 observed/predicted/private logits均不进入pool。

任一项失败即 `STOP_BEFORE_FORMAL_LOCALIZER`；不得通过扫描 context radius、PCA width、predictor
capacity 或 loss 追过 premise。

Pilot：HateMM、HateClipSeg，各自独立train；seed 234。predictor仅用该语料train split且
label-agnostic；localizer仅用同语料train video labels。Validation只在每个固定arm内部选择
checkpoint；选定后立即test pooled AP、pooled ROC、within-video macro ROC。

Arms使用同一投影维度、localizer容量、训练预算与shared evaluator：

1. `core_shared_private`：冻结cross-modal predictor，observed+shared+private；
2. `standard_masked_shared`：observed+shared，删除private，代表来源式协作表示；
3. `private_without_crossmodal`：用同模态 temporal autoencoder残差替代cross-modal prediction，
   区分cross-modal conditional innovation与普通重建残差；
4. `raw_capacity_matched`：不做prediction，使用相同额外参数处理observed features；
5. `shuffled_condition` diagnostic：冻结predictor输入在train内跨video打乱，保持availability和
   计算量，检验收益是否真的依赖对应cross-modal条件。
6. `availability_only`：只读取 modality availability、ASR coverage 与长度，排除missingness捷径。

另对 `core_shared_private` 选定的同一个 checkpoint 做两个不重新训练的机制诊断：

- `private_drop_same_ckpt`：全部 private channels置零；
- `private_time_permute_same_ckpt`：只在每个视频、每个真实可用 modality内部打乱 private 的
  时间对应，保留其边缘分布与availability。
- `private_matched_noise_same_ckpt`：用各模态 train residual 均值/方差匹配的噪声替代private。

这两项只判断已训练 core 是否真正使用了对齐的 private evidence，不作为可独立选择的模型 arm。

机制通过要求：

1. core 相对 `standard_masked_shared` 与 `raw_capacity_matched` 在HMM/HCS两边within同向提高，
   至少一边 `>= .020`；
2. core优于 `private_without_crossmodal`，且 shuffled condition消除主要增益；
3. 同一 core checkpoint 的 zero/permuted/matched-noise 三种干预中至少两种，必须在两语料各自
   消除至少一半 `core - standard_masked_shared` 的within增益，且至少一边绝对下降 `>= .010`；
   否则 private 不是 load-bearing，机制 claim失败；
4. 改善视频应集中在 baseline fused输给某个单模态branch、且该carrier private通道有效的case；
   若只由residual norm、availability、ASR缺失率或视频长度解释，机制失败；
5. core必须在两语料六个SOTA格全部严格过门才扩EN/ZH；最终四语料所有固定指标必须SOTA。

## Anti-pattern guard

- 不把 reconstruction error直接当hate score。
- 不用 test oracle选择carrier、modality或shared/private权重。
- 不把 missing zero向量当negative evidence；必须显式mask。
- 不允许 predictor与MIL联合训练后通过label shortcut破坏 operational decomposition；pilot中
  predictor预训练后冻结。
- 不扫描context radius、projection width、mask rate、reconstruction loss或fusion rule来追test；
  首版固定同一设置用于HMM/HCS。
- 不声称首次masked modality modeling、cross-modal reconstruction、private/shared features或
  multimodal MIL。唯一可能claim是上述机制作为carrier-specific hateful temporal localization
adaptation，且必须由controls证明load-bearing。

## Novelty/identifiability review 必答

1. 来源 masked-modality prediction或相同核心是否已用于 hateful video detection/localization？
2. `conditional shared prediction + retained private innovation + weak temporal MIL`是否是
   non-trivial task adaptation，还是MISA/MFM/CrossMAE与普通MIL的组件拼接？
3. frozen predictor是否使shared/private operationally identified；LayerNorm/residual、predictor
   underfit或modality尺度是否导致伪innovation？
4. evidence head是否能忽略private通道而退化；如果能，什么最小机制/control才能使claim可证伪，
   而不加入新的不可识别latent owner？
5. 与CLARA、MultiHateLoc、MM-HSD、masked multimodal learning、audio-visual parsing和项目既有
   carrier/ownership失败方向的边界是什么？
