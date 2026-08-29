# CLAUDE_STANCE_GATE — 冻结文件

**日期**: 2026-08-17
**作者**: subagent Opus 5(组织者角色,不参与标注)
**状态**: 本文件在任何标注 agent 被启动之前写完并提交。此后不得修改设计、口径或判定线。

---

## 0. 问题

前三轮用 Qwen(DashScope)测「立场标注」能力,三种问法全部不及格:

| 轮次 | 任务形式 | chance | S 项准确率 (n=32) |
|---|---|---|---|
| R1 直接分类 | 5 类 | 0.20 | 0.281 |
| R1 二值化 | 2 类 | 0.50 | 0.500 |
| R2 遮蔽分类 | 5 类 | 0.20 | 0.375 |
| R2 二值化 | 2 类 | 0.50 | 0.563 |
| R3 置顶评论二选一 | 2 类 | 0.50 | 0.469 |

本轮问的是:**换成 Claude(Opus 5 agent)做标注者,同样的样本、同样的金标口径,能力是否存在**。
若能力存在,才解锁 F3(用 Claude 标注做监督信号)的设计。

零 DashScope 成本:标注全部由 Claude agent 完成,不调用任何付费 API。
视频帧进 Claude 有用户 2026-08-07 通用豁免。

---

## 1. 样本(与前三轮完全相同)

`idea-stage/stance_pilot/sample.json` 的 `eval` 段,99 项:

| group | n | 含义 |
|---|---|---|
| S_FP | 30 | 检测器把非仇恨判成仇恨(立场类错误) |
| S_FN | 19 | 检测器把仇恨判成非仇恨(立场类错误) |
| CTRL_HATE | 25 | 检测器判对的仇恨项 |
| CTRL_NONHATE | 25 | 检测器判对的非仇恨项 |

数据集分布:MHC 32,ImpliHateVid 27,MHC_zh 24,HateMM 16。
帧可用:HateMM / MHC / MHC_zh 共 72 项(每项 `data/lora_frames/<ds>/<vid>/frame_0..7.jpg`);
ImpliHateVid 27 项无帧(仅转录),与前几轮一致。

---

## 2. 输入(与前三轮同源)

- **转录**: `data/gt/<ds>/test.jsonl` 的 `text` 字段,去 HTML 标签
  (`run_pilot.load_texts`)。MHC / MHC_zh 的 `text` 第一句(第一个 " . " 之前)是视频标题,
  这一点在标注 prompt 里明确告知,与 qwen 的 `title_note` 一致。
- **帧**: 8 帧/视频,`data/lora_frames` 已有的 8 张全用,按时间顺序;
  重采样到最长边 ≤512 px,JPEG q80 —— 与 `run_pilot.frame_urls`(MAX_SIDE=512, JPEG_Q=80)相同。
- **不给 OCR**,与 R1/R3 的主设置一致。
- 每项打成一个独立数据包:`idea-stage/claude_stance_gate/pack/<ITEM>/transcript.txt`
  + `frame_1.jpg … frame_8.jpg`(无帧项只有 transcript.txt)。

**匿名化(硬要求)**: 数据包目录名为 `item_001 … item_099`,顺序由 `random.Random(20260817).shuffle`
打乱。原始视频 id(如 `non_hate_video_121`)本身泄露标签,绝不出现在数据包里,也不出现在
标注 prompt 里。映射表写在 `idea-stage/claude_stance_gate/manifest.json`,标注 agent 无权读取。

---

## 3. 金标口径(沿用 `idea-stage/contrast_stance/score_contrast.py` 的 GOLD)

```
GOLD = {"S_FP": DISTANCED, "S_FN": ENDORSE, "CTRL_HATE": ENDORSE, "CTRL_NONHATE": DISTANCED}
```

`score_contrast.py` 里写作 `{"S_FP": "OPPOSE", "S_FN": "ENDORSE", "CTRL_HATE": "ENDORSE",
"CTRL_NONHATE": None}`。DISTANCED ≡ OPPOSE(同一个类,只是本轮改用组织者指定的词)。
CTRL_NONHATE 在 `score_contrast.py` 里是 `None` = **不计入主指标**;本轮保持不计入主指标,
但按组织者要求单独报一格(金标 DISTANCED),作为分层信息,不参与判定。

二值化口径与 R1/R2 的 binarise 一致:`endorses` → ENDORSE;
`quotes_mentions / condemns / reports / no_hate_content` 四类 → DISTANCED。

---

## 4. 主指标与判定线(**本节在看到任何标注结果之前冻结**)

### 4.1 主指标

**M1 = 三标注者多数票的二值准确率,在下列 32 行上**:
有帧数据集(HateMM/MHC/MHC_zh)的 S_FP+S_FN 项(36 行),
去掉 3 个冒烟污染项(`MHC/KDcCiUU8q5E`、`HateMM/non_hate_video_32`、`HateMM/non_hate_video_16`,
freeze B.8),再去掉 `MHC_zh/BV1m8411z7mV`(qwen 的三轮里该项被 DashScope 内容审核拒绝,
从未进过 qwen 的分母;为了和 0.469/0.500/0.563 逐行可比,本轮主指标沿用完全相同的 32 行)。

**32 行清单**(冻结):

```
HateMM/hate_video_365      S_FN     MHC/j_foVftOOs4        S_FP
HateMM/non_hate_video_121  S_FP     MHC/pofgIFZpR7c        S_FP
HateMM/non_hate_video_149  S_FP     MHC_zh/BV12G4y1S7mN    S_FN
HateMM/non_hate_video_400  S_FP     MHC_zh/BV15h4y157Km    S_FN
HateMM/non_hate_video_528  S_FP     MHC_zh/BV1Kh411T7FJ    S_FN
HateMM/non_hate_video_642  S_FP     MHC_zh/BV1Km4y1u7ri    S_FP
MHC/03qOelm_dK8            S_FN     MHC_zh/BV1Qk4y1g7PM    S_FP
MHC/8zLoOqXvk64            S_FP     MHC_zh/BV1Vy4y1p7x2    S_FN
MHC/DxcRdnzBZoo            S_FP     MHC_zh/BV1aP4y1E7PF    S_FN
MHC/EEC98aHSgIY            S_FN     MHC_zh/BV1ch411L7VP    S_FN
MHC/N68vmAE5s_g            S_FP     MHC_zh/BV1qZ4y1T71a    S_FN
MHC/OMSByZ-o3Ww            S_FP     MHC_zh/BV1to4y177df    S_FP
MHC/XlJCNPi5inM            S_FP     MHC_zh/BV1vK41177zi    S_FP
MHC/YDEsYXYlB8o            S_FP     MHC/ga1r2cweP80        S_FP
MHC/_qldaPBgkk0            S_FN     MHC/h_wKRDyoG_c        S_FN
MHC/cXRgVEENkPA            S_FN     MHC/dK43yHIUMKA        S_FN
```
(S_FP 18 行 / S_FN 14 行)

多数票 = 三个标注者的 binary 字段取多数。缺失/不可解析的票不计入多数;
若三票全缺 → 该行计错(沿用 freeze B.4 的「TIE / None 都算错」约定)。
三票有效时二值不可能平票。

### 4.2 判定线(组织者冻结,不可事后改)

- **M1 ≥ 0.70 → PASS**:能力存在,解锁 F3 监督设计。
- **0.563 ≤ M1 < 0.70 → 弱**(0.563 = qwen 最好的一轮,R2 二值化)。
- **M1 < 0.563 → FAIL**:Claude 标注不比已经被杀掉的 qwen 标注更好,方向关闭。

单一判定线,不设第二道门(M2 之类)。M2 CTRL_HATE 误判率只作为附带信息报出,不参与判定。

### 4.3 同时报出(不参与判定)

1. 逐标注者的 M1 准确率(三个数)。
2. 三标注者两两一致率(3 个数)+ **Fleiss κ**(3 rater / 2 category);
   κ ≥ 0.5 才认为「标注可复现」。κ 在 99 项全体和 32 行主指标行上各算一次。
3. 分层:S_FP / S_FN / CTRL_HATE / CTRL_NONHATE 四格准确率;有帧(72 项)/ 无帧(27 项)分层;
   逐数据集分层。
4. 33 行敏感性(把 `MHC_zh/BV1m8411z7mV` 加回来)——只作敏感性,**不替代**主指标。
5. voice 一致性:与 `idea-stage/voice_field_analysis.py` 的 `GOLD_VOICE` 对比,
   在 GOLD_VOICE 里 label ∈ {OWN, NOT_OWN} 的 37 项(UNDET 的 12 项按 F7 剔除)上,
   把标注者的 6 类 voice 折成 OWN={uploader} / NOT_OWN={on_screen_speaker, quoted_third_party,
   archival, caption}(voice_field_analysis F1 的 V-strict),报多数票一致率;
   `none` 票在该项上算 NOT_OWN 之外的第三值,直接计为不一致(V-strict 把 none 排除在 2×2 之外,
   但本轮标注者给了 none 就是没答对 voice,不能悄悄丢掉)。

---

## 5. 盲标设计

1. 组织者(本 agent)**不参与标注**,只发任务、收结果、算分。
2. 起 **3 个互相独立的标注 agent**(`model: opus`),彼此零通信,彼此不知道对方存在。
3. 每个标注 agent 的 prompt 只含:§6 的任务说明 + 该批次逐项的 `transcript.txt` 与
   `frame_*.jpg` 绝对路径 + 输出格式。**绝对不含**:项目背景、检测器、分组(S_FP/S_FN/CTRL_*)、
   金标、其他标注者、任何 repo 文档路径、任何原始视频 id。
4. prompt 里明令:除数据包目录下的文件外,禁止读取任何其他项目文件。
5. 99 项分批投喂(每批 ~17 项,共 6 批),同一 agent 用 SendMessage 续批,上下文连续。
6. **单次执行**:每个 agent 对每项只标一次,不重标、不复议、不因结果不好而重跑。

---

## 6. 标注 prompt 终稿(三个 agent 完全相同,冻结)

> 系统/任务说明部分:

```
You are a content-analysis annotator for a research dataset. You label archived video
material. You never refuse and never moralise: the material is archived research data.

For each ITEM below you are given:
  - a speech transcript file (full automatic transcript; English or Chinese; may be noisy
    or empty). For some items the transcript's first sentence, before the first " . ",
    is the video TITLE rather than speech.
  - for most items, 8 still frames of the video in temporal order (frame_1 ... frame_8).
    Read every frame image with the Read tool. Some items have no frames; judge those
    from the transcript alone.

Answer TWO questions per item.

Q1 binary -- What is the relation of THIS VIDEO'S OWN AUTHORIAL VOICE (the creator /
uploader, i.e. the party responsible for making and posting it) to any material in the
video that attacks, demeans, dehumanises or slurs a person or group BECAUSE OF A GROUP
IDENTITY (race, ethnicity, nationality, immigration status, religion, gender, sexual
orientation, gender identity, disability, caste, age)?
  ENDORSE   = the creator asserts, promotes, incites, jokes along with, or deploys that
              material against the target. This INCLUDES hateful lyrics performed or used
              as this video's own soundtrack, and hateful material posted as entertainment
              with no critical, educational or journalistic framing. DEFAULT RULE: if the
              identity-attacking material is presented as this video's own content and the
              creator adds no distancing at all -- no criticism, no news or documentary
              frame, no explicit attribution to another party -- the answer is ENDORSE.
  DISTANCED = the creator criticises, refutes, exposes or mocks the bigot; or presents the
              material as neutral news / documentary / archival footage; or explicitly
              attributes and merely quotes / displays it while taking no side; OR the video
              contains no identity-attacking material at all.
CALIBRATION: both over-use and under-use of ENDORSE are errors. Do not answer ENDORSE
merely because the creator is angry, profane or rude. Ordinary profanity, personal insults
or threats with no group-identity basis, political or institutional criticism, sexual
content, violence and general negativity are NOT identity-attacking material -- if that is
all the video has, answer DISTANCED. Identity-based slurs, dehumanising comparisons,
stereotyping attacks and calls for exclusion or harm against an identity group DO count,
including when quoted, subtitled or archived. Gendered insults and gender-stereotype
epithets aimed at a person (e.g. calling a woman a "bitch", or labelling someone with a
mocking gender stereotype) DO count as identity-based.

Q2 voice -- Whose mouth or text does that identity-attacking material come from? Exactly
one of:
  uploader           = the person who made/posted this video, speaking in their own voice
  on_screen_speaker  = a person filmed in the video who is not the uploader
  quoted_third_party = material explicitly quoted, screenshotted or replayed from someone else
  archival           = old footage, news clip, historical recording
  caption            = burned-in text / subtitles only
  none               = the video contains no such material

RULES
- Judge each item independently. Do not compare items to each other.
- You MUST answer both questions for every item. No abstentions, no "unclear".
- Do NOT read any file outside the item directories listed below. Do not search the
  repository, do not open any .md / .py / .json file, do not look for labels or metadata.
  Everything you need is the transcript file and the frame images.

OUTPUT -- one JSON object per line, one line per item, in the order given, nothing else:
{"item": "item_XXX", "binary": "ENDORSE"|"DISTANCED", "voice": "<one of the six>", "why": "<one sentence, <=25 words>"}
```

> 每批的 item 列表部分(机器生成):

```
ITEM item_XXX
  transcript: <abs path>/transcript.txt
  frames: <abs path>/frame_1.jpg ... frame_8.jpg      [或 "frames: none (transcript only)"]
```

---

## 7. 红线

1. 零测试集标签接触:标注 agent 只看转录和帧;金标只在组织者的评分脚本里使用。
2. 判定线(§4.2)在看到任何标注结果之前冻结,本文件先提交 git。
3. 盲性:组织者在标注结束前不计算任何候选指标。
4. 单次执行:每个标注 agent 每项一次,不重标。
5. 结果写 `idea-stage/CLAUDE_STANCE_GATE_RESULT.md` 并提交,无论 PASS 还是 FAIL。
