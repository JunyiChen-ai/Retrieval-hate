# Temporal Split Feasibility — 四个数据集的"视频上传时间"可恢复性评估

Scope: 只读侦察。未下载任何视频、未提交 SLURM 作业;仅读本地/B2 标注文件 + 轻量元数据查询
(Bilibili web API × 33、yt-dlp 仅元数据 × 33、Zenodo 上 62KB 的 HateMM 官方标注 CSV、GitHub API)。
Date: 2026-07-02. Env: conda `HateVideo`(yt-dlp 2026.03.17)。Host: 登录节点(有外网)。

TL;DR: **MultiHateClip 英文(YouTube)与中文(Bilibili)两个子集可行**——annotation 里的
Video_ID 就是平台原生 ID,抽样验证经 yt-dlp / Bilibili API 可直接拿到精确上传日期,
存活率分别约 **83%** 与 **77%**。**HateMM 与 ImpliHateVid 不可行(LOW)**:两者的 ID 均已
匿名化(`hate_video_N` / `EX_*`),官方发布物(Zenodo / B2 原始包)里没有任何 URL、平台 ID
或日期字段,除联系作者外无恢复路径。另注意:**死链显著偏向 Hateful/Offensive 类**,
temporal split 只能建立在"可定年子集"上,需在实验设计里声明这一选择偏差。

---

## 0. 总表

| 数据集 | 规模(本地标注) | 平台 | 标注中的 ID | 日期字段 | 恢复途径 | 抽样验证 | 预计覆盖率 | 可行性 |
|---|---|---|---|---|---|---|---|---|
| MultiHateClip EN | 891 条(splits 共 1001 ID) | YouTube | 11 位 YouTube ID(891/891 合规) | 无 | `yt-dlp --print upload_date`(需联网,~1–2s/条) | 25/30 成功,日期 2010→2024 | **~83%**(95%CI ≈ 66–93%) | **HIGH** |
| MultiHateClip ZH | 897 条 | Bilibili | BV 号(897/897) | 无 | Bilibili web API `view?bvid=` 的 `pubdate`(精确到秒);BV→aid 可离线解码但 **aid→日期不可离线推算** | 23/30 成功,日期 2018→2024 | **~77%**(95%CI ≈ 59–88%) | **HIGH**(略低于 EN) |
| HateMM | 1066 条 | BitChute | `hate_video_N` / `non_hate_video_N`(匿名) | 无 | 无:官方 Zenodo 标注仅 4 列(file_name/label/hate_snippet/target),无 URL | —(无 ID 可查) | **~0%** | **LOW** |
| ImpliHateVid | 2009 条 | BitChute + Odysee | `EX_*` / `IM_*` / `NH_*`(匿名) | 无 | 无:本地与 B2 的 xlsx 均只有单列 Video_ID | —(无 ID 可查) | **~0%** | **LOW** |

---

## 1. MultiHateClip English — HIGH

**位置**: `/data/jehc223/Multihateclip/English/`

**标注字段**(`annotation(new).json`,list of 891):
`Video_ID, Title, Transcript, Emotion, Frames_path, Audio_path, Frames_description, Text_description, Mix_description, Label`。
无任何日期/时间戳字段。Label 分布:Normal 601 / Offensive 218 / Hateful 72。
splits(无表头):train 701 + valid 100 + test 200 = 1001 ID;clean 版 train 550 + test 161。
本地 `video_mp4/` 有 792 个 mp4,文件名即 YouTube ID。

**平台 ID**: `Video_ID` 全部 891 个都是标准 11 位 YouTube ID(regex `[A-Za-z0-9_-]{11}` 全过)。

**验证**(登录节点,`yt-dlp --skip-download --print upload_date`,随机 30 条,seed=42):

- 25/30 成功返回精确 `upload_date`(如 `0ATva49qP4w → 20240126`,与标注 Title 完全吻合);
- 5/30 "Video unavailable";
- 日期分布:1 条 2010、1 条 2021、6 条 2022、13 条 2023、4 条 2024 —— 主体集中在 2022–2024,
  时间跨度足够做 temporal split(且 2023 内部也可再细分)。

**死链的标签偏差**(样本内):Normal 1/21 死(5%),Offensive 3/7(43%),Hateful 1/2(50%)。
YouTube 对违规内容下架 → 越 hateful 越可能查不到日期。

**结论**:HIGH。全量 891 条约 30–45 分钟可查完(纯元数据,轻量)。对约 17% 死链可选补救:
Wayback Machine 的 youtube watch 页快照(能捞回一部分),或直接把不可定年样本固定进 train。

## 2. MultiHateClip Chinese — HIGH

**位置**: `/data/jehc223/Multihateclip/Chinese/`(结构与 EN 相同)

**标注字段**:同 EN schema,897 条,无日期字段。Label:Normal 605 / Offensive 180 / Hateful 112。
`Video_ID` 全部 897 个都是 BV 号,`video/` 内文件名即 `BV*.mp4`(814 个)。

**离线解码验证**:BV→aid 的公开算法(2023 新版:XOR `23442827791579` + 位重排)本地可跑,
解码结果与 API 返回的 aid 完全一致(如 `BV1Bk4y1U7Fk → av751641214` ✓)。
**但 aid→上传时间不能离线推算**:B 站 2020 年后 aid 分配非顺序,抽样实测非单调,误差可达数年级:

```
aid 223,977,352 → 2023-02-06   而  aid 258,833,288 → 2022-07-26
aid 839,383,798 → 2020-09-03   而  aid 863,571,390 → 2023-02-01
```

所以"BV 号本身编码投稿时间"这条路 **走不通**;必须查 API。

**API 验证**(`api.bilibili.com/x/web-interface/view?bvid=`,随机 30 条,seed=42,0.6s 间隔):

- 23/30 返回 `pubdate`(unix 秒级时间戳,如 `BV1CT411Y78R → 2023-03-03`);
- 7/30 死链(错误码 -404 已删除 / 62002 稿件不可见 / 62012 仅 UP 主可见);
- 日期分布 2018–2024,集中在 2022–2024。

**死链的标签偏差**(样本内):Normal 2/17 死(12%),Offensive 2/8(25%),**Hateful 3/5(60%)**。
比 YouTube 更严重——Hateful 类可定年比例可能只有一半左右。

**结论**:HIGH(覆盖率略低于 EN)。全量 897 条 API 查询约 15 分钟。死链补救:biliplus 等
第三方缓存站可查到部分已删稿件的历史信息(可选,量小)。

## 3. HateMM — LOW

**位置**: `/data/jehc223/HateMM/`(另有 `/data/jehc223/EMNLP2/datasets/HateMM`、
`/data/jehc223/HVGuard/datasets/HateMM` 等副本,schema 全部相同)

**标注字段**:同上 schema,1066 条(Hate 427 / Non Hate 639),`Video_ID` 为匿名的
`hate_video_N` / `non_hate_video_N`。Title 全空。无 URL、无日期。

**官方源核实**:GitHub `hate-alert/HateMM` → Zenodo record 7799469。官方
`HateMM_annotation.csv`(62KB,已拉取核对)只有 4 列:
`video_file_name, label, hate_snippet, target` —— **官方发布本身就不含 BitChute URL/slug**。
本地各副本 grep "bitchute" 仅命中转写文本里口播的网址,非元数据。

**恢复路径**:唯一途径是联系作者索要 file_name→BitChute URL 映射(论文 ICWSM 2023,视频
采集于 ~2022);即使拿到映射,BitChute 大量视频已下架且该平台无正规元数据 API,可回溯比例
预计也很低。用转写/标题去 BitChute 检索反查属于模糊匹配,不可靠且工作量大。

**结论**:LOW,预计覆盖 ~0%(不联系作者);联系作者后也难超过一小部分。

## 4. ImpliHateVid — LOW

**位置**: 本地 `/data/jehc223/ImpliHateVid/`(标注 + splits;video 被裁剪),原始 2009 个视频在
`b2:junyi-data/ImpliHateVid/`(Explicit/Implicit/Non Hate Videos 三个目录)。

**标注字段**:`annotation(new).json` 每条仅 `Video_ID, Label, Title(空), Transcript`,2009 条
(Hateful 1009 / Normal 1000)。`Video_ID` 为匿名的 `EX_*` / `IM_*` / `NH_*`。

**B2 核实**(`rclone ls`,只看清单):bucket 里除 mp4 外只有 `Train/Test/Val_videos.xlsx`
三个文件,与本地相同;xlsx 已用 openpyxl 打开核对——**单 sheet、单列 `Video_ID`**,无 URL、
无日期。B2 上 mp4 文件名也全是匿名 ID(`EX_1.mp4` 等)。

**恢复路径**:来源为 BitChute + Odysee(arXiv 2508.06570,ACL 2025),同 HateMM——只有联系
作者一条路,且两平台可回溯性都差。

**结论**:LOW,预计覆盖 ~0%。

---

## 5. 建议

1. **值得做 temporal split 的:MultiHateClip EN + ZH**。两个子集共 ~1788 条,预计
   ~1430 条(80%)可拿到精确上传日期,时间跨度 2010/2018–2024、主体 2022–2024,
   足够切出有意义的 train(旧)/test(新)边界;EN、ZH 还可各自独立切,做跨语言时间泛化对照。
2. **全量日期采集是登录节点级的轻活**(纯网络、无 GPU):~900 次 Bilibili API + ~900 次
   yt-dlp 元数据查询,合计 <1.5 小时。建议写成**可断点续跑**的脚本(登录节点进程可能被回收),
   结果落成 `Video_ID → upload_date` 的 JSON 存进 `data/gt/` 一类目录,一次采集永久复用。
3. **必须处理的混杂:死链偏向 Hateful/Offensive**(ZH Hateful 样本死亡率 ~60%)。
   可定年子集的标签分布会偏移。建议:(a) 主实验只在可定年子集上做 temporal split 并报告
   该子集的标签分布;(b) 不可定年样本固定放 train(不进 test,避免污染"未来"集);
   (c) 可选用 Wayback/biliplus 补一部分死链日期,缩小偏差。
4. **HateMM / ImpliHateVid 不做 temporal split**,继续按现有角色作为(跨数据集)静态评测集;
   若审稿需要,可注明"官方发布已匿名化、无时间元数据"这一客观限制。顺手之举:给两个数据集
   作者各发一封邮件索要 URL 映射,成本低,万一拿到可作附加实验。

## 附:验证凭证(可复现)

```bash
# YouTube(MHC-EN)
yt-dlp --skip-download --print upload_date https://www.youtube.com/watch?v=0ATva49qP4w
# → 20240126

# Bilibili(MHC-ZH)
curl -s 'https://api.bilibili.com/x/web-interface/view?bvid=BV1CT411Y78R' -H 'User-Agent: Mozilla/5.0'
# → code=0, data.pubdate=1677785555 (2023-03-03), data.aid 与离线 BV→aid 解码一致

# HateMM 官方标注(Zenodo 7799469, 62KB)表头
# video_file_name,label,hate_snippet,target   ← 无 URL/日期列
```

抽样明细(30+30 条逐条结果)存于本次会话 scratchpad `bili_sample.txt` 及上方正文;
随机种子 42,可复现。
