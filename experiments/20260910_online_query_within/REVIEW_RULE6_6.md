# 规则 6 代码复核：模块一第 6 轮统一文本编码器（2026-09-21）

审稿人：独立子 agent（Claude Fable 5.1），只读，不跑训练。对象：`scripts/build_hate_text_1fps.py`（新）、`src/hier_evidence_common.py` 与 `experiments/20260910_online_query_within/train.py` 的 diff（`text_feat` 开关）。结论：无 BLOCKER，两处 MUST-FIX，五条 NOTE。

## MUST-FIX（已改）

1. 文档说法错误：抽取脚本、`data/hate_text_1fps/PROVENANCE.md`、README 第 12 节原写"BERT 行在 HateMM 上按词嵌入、HCS 按 chunk"。审稿人核实：`data/ASR` 两语料都是词级与 chunk 级混合（HateMM 526 词级 / 540 chunk；HCS 147 / 246），而被替换的 BERT 行来自另一份 Whisper 记录 `results/reproduction/asr/<corpus>_all/timestamped_chunks.jsonl`（只有 chunk 级，中位 2.0 s），从未按单词嵌入；两份转录文本相同的视频只有 HateMM 370/1066、HCS 132/393；单位数 17582 对 11541（HateMM）、9196 对 5296（HCS）；覆盖率 .773 对 .817（HateMM，零覆盖视频 2 对 34）、.868 对 .816（HCS，0 对 9）。所以 `bert → hate_roberta` 同时换了转录来源、单位和编码器。**处理**：三处文档改写；加对照特征 `data/bert_utterance_1fps`（bert-base-uncased CLS、同样话语单位、128 token）与运行 `bert_utterance`，把编码器的贡献单独分出（README 第 12 节判定 2）。
2. `train.py`：远程机 `git pull` 后没 rsync 特征目录时，`load_text_rows` 对每个视频返回 None，训练会在全零文本流上照跑，只有日志一行提示。**处理**：构造 cache 后加 `assert n_missing_text < len(all_ids)`。

## PASS（审稿人核对项）

- 网格与归属：T 取 `hc.video_duration`（VGGish 行数），与 BERT 行的 T 在全部 1068 + 393 个视频上一致（0 不符）；`assign_frames` 同一函数；`load_text_rows` 重采样相同。
- 分类头输入：transformers 4.57.6 的 `RobertaClassificationHead.forward` 取 `features[:, 0, :]`；`RobertaForSequenceClassification` 把 `outputs[0]`（= `hidden_states[-1]`，无额外归一化）送入分类头，所以 `hidden_states[-1][:, 0]` 就是分类头的输入；`.eval()` 关闭分类头 dropout；`mod.classifier(h[:, :1])` 的复检有效（最大差 0）。分词与打分脚本相同（max_length 128、截断、右填充带 attention mask）；批组成不同只影响填充长度，浮点级差别。
- 话语处理：`utterances()` 逐条记录应用，词级与 chunk 级在两语料都正确处理；空文本过滤；< 1 s 的跨度按 x_t 同样加宽；`data/text_hate` 的 `p_asr` 掩码与特征的有语音掩码一致（差 0）。
- 写文件：`.tmp.npy` + `os.replace`；`index.json` 每次整体重写（全部 id 都处理，幂等）。不读标签（`hdata.load_split` 只读 id；ASR 记录的 `label` 字段未触碰）。diff 不向 `data/` 写文件。
- 默认行为不变：`text_path` / `load_text_rows` / `ScaffoldCache` 新增关键字参数默认 `"bert"`，`TEXT_ROOTS["bert"] is TEXT_ROOT`；其余 11 处调用者按位置调用，行为逐字节相同；`experiments/20260903_hier_evidence_mil/dataset.py` 仍导入 `TEXT_ROOT`，保留。
- 流向：`cfg → Args(cfg) → a.text_feat → hc.ScaffoldCache(..., text_source=...)`，受 `TEXT_ROOTS` 断言保护；`state["cache"]` 只构造一次，else 分支只换 `masked_fn`；`ScaffoldCache.base[vid]` 在 `__init__` 由所选来源填充，`build()` 复用它，音频+文本块不会被从 BERT 重建；`config.json` 快照含 `text_feat`。
- 配置校验：`DEFAULTS` 含 `text_feat`；未知键在 `train.py` 与 `search.py` 都拒绝。
- 启动脚本：`search.py --extra-config` 与 `run_diag.sh` 的 `tag:abl:{json}` 解析都能把 `{"text_feat": "hate_roberta"}` 送到训练器（审稿人实测含冒号的 JSON 解析正确）。
- 导入副作用：`extract_bert_sentence_features` → `extract_clip_features` 模块级只导入 numpy 与 `frame_eval_common`；`build_text_hate_scores` 模块级只有常量。

## NOTE（记录，不改）

3. 尺度：`model.py` 的 `fc_a = nn.Linear(896, hid)` 直接作用于 [VGGish | 文本]，无输入归一化，LayerNorm 只在 CMA 层内。非零行 L2：新特征 21.3（逐维 std .77，最大逐维均值 8.5，第 588 维 RoBERTa 离群维），BERT 14.9（.54，6.2），VGGish 约 2.7–3.0。比值约 1.4，远小于 10，同一学习率可训；文本块对音频块的主导更强。`bert_utterance` 行 L2 14.9。
4. 逐秒归属与 x_t 不同：22.1%（HateMM）/ 24.1%（HCS）的有语音秒被多条话语覆盖，特征取重叠最大（并列取更早），x_t 取平均。同一模型，但不是逐秒同一输入。
5. 继承自 `utterances()`：词级时间戳非单调（HateMM 205 / HCS 137 条记录）使合并后的话语可跨整段视频，配合"并列取更早"，一条 > 60 s 的话语占 58.6%（HateMM）/ 63.4%（HCS）的有语音帧，且截断到前 128 token。BERT 行有同样性质（55.4% / 63.8%，64 token），不是退化，但"逐秒文本行"多数是长跨度开头的嵌入。
6. `except Exception: pass` 在 `video_duration` 失败时静默丢视频（沿用打分脚本；当前 index 1068 / 393 齐全）。
7. 原 docstring 的 HateMM / HCS 不对称说法已删。
