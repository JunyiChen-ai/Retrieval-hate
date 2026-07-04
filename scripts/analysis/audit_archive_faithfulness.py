#!/usr/bin/env python
"""Experiment 1: archive faithfulness audit (model-assisted, human final review).

Stratified sample of 60 archived videos (EN/ZH x Hateful/Offensive/Normal),
side-by-side export of archive fields vs the original title+transcript, plus a
merge step that folds per-item verdicts into a human-readable markdown audit
table at research-wiki/AUDIT_archive_faithfulness.md.

Read-only w.r.t. src/ and data/. No training, no GPU.

Usage:
  python scripts/analysis/audit_archive_faithfulness.py sample \
      --out_items <scratch>/audit_items.json
  python scripts/analysis/audit_archive_faithfulness.py render \
      --items <scratch>/audit_items.json \
      --judgments <scratch>/audit_judgments.json \
      --out research-wiki/AUDIT_archive_faithfulness.md
"""
import argparse
import html
import json
import os
import random
from collections import Counter, OrderedDict

ROOT = "/data/jehc223/RGCL"
ARCHIVE_TAG = "Qwen2.5-VL-7B-Instruct_archive"
SPLITS = ["train", "dev_seen", "test_seen"]
GT_SPLITS = {"train": "train", "dev_seen": "val", "test_seen": "test"}
DATASETS = {"MHC": "English", "MHC_zh": "Chinese"}
# W2 suspected label-noise samples -- MUST be in the sample (both MHC/train).
FORCE_INCLUDE = {"MHC": ["XScP1AiMkNM", "QvPp8Q7QhWE"], "MHC_zh": []}
PER_STRATUM = 10  # per (language, 3-class label)
SEED = 0
TRANSCRIPT_CHARS = 1500  # excerpt length shown/reviewed (full length reported)


def load_archives(ds):
    """id -> archive record (last occurrence wins across split files)."""
    recs = {}
    for sp in SPLITS:
        p = os.path.join(ROOT, "data/Archive", ds,
                         "{}_{}.jsonl".format(sp, ARCHIVE_TAG))
        for line in open(p):
            r = json.loads(line)
            r["split"] = sp
            recs[r["id"]] = r
    return recs


def load_gt(ds):
    """id -> (binary label, split)."""
    out = {}
    for sp in ["train", "val", "test"]:
        p = os.path.join(ROOT, "data/gt", ds, sp + ".jsonl")
        for line in open(p):
            r = json.loads(line)
            out[r["id"]] = (int(r["label"]), sp)
    return out


def load_source(ds):
    """id -> {title, transcript, label3} from the original MultiHateClip file."""
    lang = DATASETS[ds]
    p = os.path.join(ROOT, "data/_src_Multihateclip", lang, "annotation(new).json")
    out = {}
    for r in json.load(open(p)):
        out[r["Video_ID"]] = {
            "title": r.get("Title") or "",
            "transcript": r.get("Transcript") or "",
            "label3": r.get("Label") or "?",
        }
    return out


def build_sample():
    rng = random.Random(SEED)
    items = []
    for ds in ["MHC", "MHC_zh"]:
        arch = load_archives(ds)
        gt = load_gt(ds)
        src = load_source(ds)
        # audit universe: ids with archive + gt + source annotation
        universe = [i for i in arch if i in gt and i in src]
        strata = {"Hateful": [], "Offensive": [], "Normal": []}
        for i in sorted(universe):
            l3 = src[i]["label3"]
            if l3 in strata:
                strata[l3].append(i)
        chosen = []
        forced = [i for i in FORCE_INCLUDE[ds] if i in universe]
        for l3, pool in strata.items():
            pool_forced = [i for i in forced if src[i]["label3"] == l3]
            rest = [i for i in pool if i not in pool_forced]
            rng.shuffle(rest)
            take = pool_forced + rest[: PER_STRATUM - len(pool_forced)]
            chosen.extend(take)
        for i in chosen:
            a = arch[i].get("archive") or {}
            tr = src[i]["transcript"]
            items.append(OrderedDict(
                id=i, dataset=ds, lang=DATASETS[ds],
                split=gt[i][1], label3=src[i]["label3"],
                label_binary=gt[i][0],
                title=src[i]["title"],
                transcript_excerpt=tr[:TRANSCRIPT_CHARS],
                transcript_len=len(tr),
                archive=OrderedDict(
                    target_groups=a.get("target_groups"),
                    mechanism=a.get("mechanism"),
                    modality_cues=a.get("modality_cues"),
                    explicitness=a.get("explicitness"),
                    neutral_summary=a.get("neutral_summary"),
                ),
                parse_ok=arch[i].get("parse_ok"),
                forced=i in forced,
            ))
    return items


VERDICTS = ["faithful", "hallucination", "whitewash", "insufficient-evidence"]


def render(items, judgments, out_path):
    by_id = {it["id"]: it for it in items}
    missing = [i for i in by_id if i not in judgments]
    if missing:
        raise SystemExit("missing judgments for: {}".format(missing[:10]))

    lines = []
    w = lines.append
    w("# AUDIT: Archive faithfulness (MHC / MHC_zh, Qwen2.5-VL-7B archive)")
    w("")
    w("> **模型辅助审计,终审留给人类。** 本表由 Claude 对照原始 title+transcript "
      "逐条给出初判 (faithful / hallucination / whitewash / insufficient-evidence),"
      "所有判定均待人工抽查确认后方可引用。")
    w("")
    w("## 方法")
    w("")
    w("- 样本: 60 条,EN (MHC) / ZH (MHC_zh) 各 30,按原始 MultiHateClip 三分类 "
      "(Hateful / Offensive / Normal) 各 10 条分层随机抽样 (seed={});"
      "W2 疑似标签噪声样本 `XScP1AiMkNM`(牛油果酱)与 `QvPp8Q7QhWE`(数钱)强制纳入。".format(SEED))
    w("- 对照材料: 原始 `annotation(new).json` 的 Title + Transcript(转写超过 {} 字符时截断展示,"
      "长度已标注)与二值 gt 标注 (harmful=Hateful∪Offensive)。".format(TRANSCRIPT_CHARS))
    w("- 核查项: (1) `target_groups` 是否有文本依据; (2) `mechanism` 是否与转写内容相符; "
      "(3) 幻觉 = 档案声称 speech/on_screen_text 中存在转写里不存在的内容,或凭空给出无任何依据的"
      "目标/机制; (4) 洗白 = 文本毒性明显 (label=Hateful/Offensive 且转写可见毒性) 但档案输出 "
      "explicitness=none / 空 mechanism / benign summary。")
    w("- 重要限制: 档案由 MLLM 看 **视频帧+转写** 生成,而本审计只能看到 **文本**。纯视觉断言"
      "(如 visual cue、画面描述)无法据文本证实或证伪,只要不与文本矛盾就不计为幻觉,"
      "并在备注中标注 `visual-unverifiable`。文本无毒但 label=harmful 的条目"
      "(毒性可能在画面/语音语调里,或本身是标签噪声)判 `insufficient-evidence`。")
    w("")

    # ---- summary ----
    w("## 汇总")
    w("")
    for lang_ds in [("MHC", "EN"), ("MHC_zh", "ZH"), (None, "ALL")]:
        ds, tag = lang_ds
        sel = [i for i in by_id if ds is None or by_id[i]["dataset"] == ds]
        cnt = Counter(judgments[i]["verdict"] for i in sel)
        n = len(sel)
        row = " / ".join("{} {} ({:.0f}%)".format(v, cnt.get(v, 0),
                                                  100.0 * cnt.get(v, 0) / n)
                         for v in VERDICTS)
        w("- **{}** (n={}): {}".format(tag, n, row))
    w("")
    w("| 语言 | 三分类 | n | faithful | hallucination | whitewash | insufficient-evidence |")
    w("|---|---|---|---|---|---|---|")
    for ds in ["MHC", "MHC_zh"]:
        for l3 in ["Hateful", "Offensive", "Normal"]:
            sel = [i for i in by_id
                   if by_id[i]["dataset"] == ds and by_id[i]["label3"] == l3]
            cnt = Counter(judgments[i]["verdict"] for i in sel)
            w("| {} | {} | {} | {} | {} | {} | {} |".format(
                DATASETS[ds], l3, len(sel),
                *[cnt.get(v, 0) for v in VERDICTS]))
    w("")
    # sub-check rates
    n_all = len(by_id)
    t_ok = sum(1 for i in by_id if judgments[i].get("target_ok") == "yes")
    t_na = sum(1 for i in by_id if judgments[i].get("target_ok") == "n.a.")
    m_ok = sum(1 for i in by_id if judgments[i].get("mechanism_ok") == "yes")
    m_na = sum(1 for i in by_id if judgments[i].get("mechanism_ok") == "n.a.")
    w("- target 字段有依据: {}/{} (另 {} 条 target 为空或不可核验,记 n.a.)".format(
        t_ok, n_all - t_na, t_na))
    w("- mechanism 与内容相符: {}/{} (另 {} 条 mechanism 为空或不可核验,记 n.a.)".format(
        m_ok, n_all - m_na, m_na))
    w("")

    # ---- per-item table ----
    w("## 逐条审计")
    w("")
    for ds in ["MHC", "MHC_zh"]:
        w("### {} ({})".format(ds, DATASETS[ds]))
        w("")
        order = sorted((i for i in by_id if by_id[i]["dataset"] == ds),
                       key=lambda i: ({"Hateful": 0, "Offensive": 1,
                                       "Normal": 2}[by_id[i]["label3"]], i))
        for i in order:
            it = by_id[i]
            j = judgments[i]
            a = it["archive"]
            flag = " **[W2 疑似标签噪声,强制纳入]**" if it["forced"] else ""
            w("#### `{}`  — {} / {} / split={}{}".format(
                i, it["label3"], "harmful" if it["label_binary"] else "normal",
                it["split"], flag))
            w("")
            w("- **判定: `{}`** — {}".format(j["verdict"], j["note"]))
            w("- target 有依据: {} | mechanism 相符: {} | 幻觉: {} | 洗白: {}".format(
                j.get("target_ok", "?"), j.get("mechanism_ok", "?"),
                j.get("hallucination", "?"), j.get("whitewash", "?")))
            w("- **Title**: {}".format(md_escape(it["title"])))
            tr = it["transcript_excerpt"]
            suffix = (" …(截断,全长 {} 字符)".format(it["transcript_len"])
                      if it["transcript_len"] > TRANSCRIPT_CHARS else "")
            w("- **Transcript**: {}{}".format(
                md_escape(tr) if tr.strip() else "(空转写)", suffix))
            w("- **档案**: targets={} | mechanism={} | explicitness={}".format(
                json.dumps(a["target_groups"], ensure_ascii=False),
                json.dumps(a["mechanism"], ensure_ascii=False),
                a["explicitness"]))
            mc = a.get("modality_cues") or {}
            nz = {k: v for k, v in mc.items() if v}
            if nz:
                w("- **modality_cues**: {}".format(
                    md_escape(json.dumps(nz, ensure_ascii=False))))
            w("- **neutral_summary**: {}".format(
                md_escape(a.get("neutral_summary") or "")))
            w("")
    with open(out_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print("wrote", out_path)


def md_escape(s):
    s = html.unescape(str(s))
    return s.replace("|", "\\|").replace("\n", " ").strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["sample", "render"])
    ap.add_argument("--out_items", default=None)
    ap.add_argument("--items", default=None)
    ap.add_argument("--judgments", default=None)
    ap.add_argument("--out", default=os.path.join(
        ROOT, "research-wiki/AUDIT_archive_faithfulness.md"))
    args = ap.parse_args()

    if args.mode == "sample":
        items = build_sample()
        cnt = Counter((it["dataset"], it["label3"]) for it in items)
        print("sampled", len(items), dict(cnt))
        with open(args.out_items, "w") as f:
            json.dump(items, f, ensure_ascii=False, indent=1)
        print("wrote", args.out_items)
    else:
        items = json.load(open(args.items))
        judgments = json.load(open(args.judgments))
        render(items, judgments, args.out)


if __name__ == "__main__":
    main()
