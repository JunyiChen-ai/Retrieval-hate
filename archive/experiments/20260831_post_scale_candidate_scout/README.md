# Post-scale-transfer candidate scout：NONE

截至 2026-08-31。两份独立 scout均返回 `NONE`；未实现、未训练、未生成prediction。

## 输入证据

`runs/20260831_teacher_scale_transfer_diagnostic/main/metrics.json` 证明原7个四信号共同tuple在fivefold video-heldout ECDF下仍7/7双语料all-SOTA，而per-video/raw control均无joint通过。这要求后续机制同时具有observed local direction与train-only跨视频geometry，不能只做视频内rank，也不能再用ordinary teacher blend/KD。

## Scout 1：native timestamp / observed event

结论 `NONE`。最接近来源是MIL-NCE的native narration alignment、audio-visual aligned-vs-shifted correspondence与Temporal Cycle Consistency：

- narration或AV correspondence只证明事件同步，不证明该事件是hate；用binary bag决定方向会退回contrastive MIL；
- audio-visual同步不覆盖speech-only、OCR、visual-only或HCS silent/static meme；
- TCC可由`video identity + normalized position`完成cycle，仍不含hate time，且cross-video recurrence/co-localization已被WTAL占用。

项目已有test证据也否定统一native-event premise：HCS utterance boundary失败、word-level timestamp producer覆盖失败、content-change无统一方向、OCR不抬共同上限、conditional cross-modal prediction双语料失败。HCS `bit_Y4NcS9xwARDO` 几乎整段相同静态图但GT多次切换，是shot/ASR/OCR/AV event无法识别的直接反例。

## Scout 2：privileged rank transfer

ICCV 2013 Learning to Rank Using Privileged Information/Rank Transfer未检出进入hateful-video task，前两门可窄PASS。最强构造可用train-only OOF四信号ECDF定义student的within-video variable-margin pairs和positive-high vs certified-negative跨视频约束；只要pair margin非零，exact broadcast解析不可行。

但第三门失败：pair orientation与margin仍直接来自同一四teacher blend，数学上只是把旧SDR/listwise order distillation换成hinge/variable margin。若改成忠实SVM+、只让privileged signal预测slack而不提供时间方向，whole-video broadcast又完整可行。因此存在无法靠命名绕过的二分：提供teacher时间方向就是KD/pseudo-pair；不提供方向就不能定位。

## 裁定

当前没有同时满足以下条件的候选：来源未进目标任务、nontrivial task mechanism、observed local direction、解析排除broadcast、train-only跨视频geometry、HMM/HCS统一覆盖、test非ensemble/calibration/routing。不得继续包装event loss或privileged rank loss。

重新开放需要新的双语料test premise：同一native statistic在HMM/HCS均比time-shuffle高至少`.020`，mean-repeated与position-only严格不成立，speech/visual/OCR/static strata均有覆盖，且reference只由对应corpus train构建。

Primary sources： [MIL-NCE, CVPR 2020](https://openaccess.thecvf.com/content_CVPR_2020/html/Miech_End-to-End_Learning_of_Visual_Representations_From_Uncurated_Instructional_Videos_CVPR_2020_paper.html)、[Audio-Visual Scene Analysis, ECCV 2018](https://openaccess.thecvf.com/content_ECCV_2018/html/Andrew_Owens_Audio-Visual_Scene_Analysis_ECCV_2018_paper.html)、[TCC, CVPR 2019](https://openaccess.thecvf.com/content_CVPR_2019/html/Dwibedi_Temporal_Cycle-Consistency_Learning_CVPR_2019_paper.html)、[Rank Transfer, ICCV 2013](https://openaccess.thecvf.com/content_iccv_2013/html/Sharmanska_Learning_to_Rank_2013_ICCV_paper.html)。

