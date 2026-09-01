# Qualitative common-error inspection

截至 2026-08-31。该轮使用 MultiHateLoc test predictions/GT 已确定的 stable worst cases，
只查看既有 1fps frames 与 ASR；不训练、不选择 checkpoint、不生成 prediction。输出 contact
sheets 位于 `runs/20260831_qualitative_common_errors/contact_sheets/`。

## 查看样本与发现

- HateMM `hate_video_282`：GT positive `[49,58]`。画面在前后均为相同 Omegle 布局；ASR
  50–52 秒出现明确 slur。visual change 无法定位，局部 speech semantics 可以区分。
- HateMM `hate_video_408`：GT `[50,52]`；ASR 在该处为短促的针对性词语，画面也从评论者
  切到影视片段，说明单纯 boundary cue 不能决定 hate polarity。
- HateClipSeg `bit_Y4NcS9xwARDO`：整段基本是同一静态 Hitler 图像，但 GT 有多段正负切换；
  visual appearance 不含所需 temporal ordering，必须依赖随时间变化的语言/音频内容。
- HateClipSeg `bit_wXgeo6nAc249`：大量 source cuts、网页截图和字幕与多段 GT 共存；旧 K4 ASR
  中存在 82–206 秒超长 chunk，不能据此假定已有 transcript timestamps 足以提供精确边界。

## 对设计的约束

该证据进一步关闭 generic visual change、固定 smoothing、scene boundary 和 video-global
modality routing。下一候选若使用语言通道，必须证明局部语义与时间搬运分别 load-bearing，
并先通过来源方法未在 hateful-video detection/localization 被占用的硬门；不得把更细 ASR、
prompt/model upgrade 或外部 pseudo-span 本身包装成 novelty。

