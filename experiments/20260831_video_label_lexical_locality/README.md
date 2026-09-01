# Video-label lexical locality premise

截至 2026-08-31。该实验只检验一个 observation premise，不构成 novel method。

## 问题与固定设计

共同 worst-case test inspection 指向局部语言语义，而外部 Twitter hate
classifier 已经失败。这里检验一个不同前提：目标语料自身的 train video labels
能否学到 corpus-specific lexical direction，并通过现有 ASR timestamps 在秒级提供
跨 HateMM / HateClipSeg 的定位信息。

- 两个语料完全独立拟合，不混合 train set。
- producer 只读取 scoped train video labels、train/test ASR 和 label-blind 1 fps
  feature row count；不读取 test GT 数值。
- 固定模型：whole-video transcript 的 `char_wb` TF-IDF 3--5 gram，
  `min_df=2`、`max_features=50000`，加 class-balanced logistic regression。
- 每个测试秒的文本窗口固定为 `[t-2,t+3)`；无 ASR 的窗口使用空文本。
- control 1：同一窗口是否存在 speech；排除只学到 speech coverage。
- control 2：每个 both-class positive test video 内最多 16 个均匀分布的唯一
  非零 circular shifts，先在 video 内平均、再在 video 间平均；所有 AUC 只调用
  共享 evaluator。

这是 Rule 10 允许的 developmental test premise。test GT 只由 evaluator 读取，
不参与训练、checkpoint selection 或参数选择。

## 冻结 gate

HateMM 与 HateClipSeg 必须各自同时满足：

1. lexical within-video ROC-AUC `>= .52`；
2. lexical within 减 speech-presence within `>= .01`；
3. lexical within 减 equal-video/equal-unique shifted within `>= .02`。

任一语料失败即 `STOP_DIRECTION`，不扫描窗口、ngram、正则强度，不做按语料
routing。双语料通过只允许进入一个新的、严格按三道 novelty 门审查的方法设计；
TF-IDF probe 自身不作 novelty claim。

## 运行

```bash
bash experiments/20260831_video_label_lexical_locality/run.sh
```

输出固定在 `runs/20260831_video_label_lexical_locality/premise/`。

## 结果与去向

权威结果：`runs/20260831_video_label_lexical_locality/premise/metrics.json`；
独立 pre-run review 与 post-run audit 均 PASS，详见本目录两份 review 文档。

HateMM lexical 的 AP/pooled ROC/within ROC 为
`.536109/.748662/.632629`，speech/shift within 为 `.546803/.505095`；
HateClipSeg 为 `.502332/.476080/.522700`，controls 为 `.508136/.501199`。
两语料均过冻结 gate，裁定 `PROCEED_TO_NOVELTY`。

HateClipSeg 的 margin 很小且 pooled ROC 低于 `.5`，故这里只确认一个弱的
positive-video 内部 lexical locality signal。下一步必须另行通过三道 novelty
硬门；不得把该 probe、score concat 或 lexical teacher distillation直接称为方法。
