#!/usr/bin/env python
"""Memory-editing directionality RE-TEST on archive prompt v2 (ZH focus).

v1 finding (DEMO_memory_editing.md, honest negative): ZH group-takedown had
ZERO flips in the LGBTQ+ test slice because the v1 archives almost never
filled `target_groups` (6/583 non-empty in train) -- editing was capped by
archive quality, not by the mechanism. Prompt v2 (AUDIT fixes) restores
target recall; this script re-runs the ZH half of the demo with v2 archives.

Head: the SAME frozen v1 winning ckpt as the v1 demo (job 12207 ZH / 12210
EN). Per the multi-seed post-mortem + sha1 audit, the archive-kNN key channel
never touches training (pure inference-time key construction), so swapping
v1 -> v2 archive keys under the frozen head is protocol-sound, needs no
retraining, and isolates the archive change from any retraining confound.
The v1-keys baseline row is also reported, so the table doubles as a
retraining-free v1-vs-v2 key-swap comparison.

Differences vs scripts/analysis/memory_editing_demo.py (everything else --
key construction, faiss vote, random controls, ckpt -- is imported from it,
protocol-identical by construction):
  * archives read from data/Archive/<ds>/v2/, kNN keys from
    data/CLIP_Embedding/<ds>/v2/;
  * TWO slice definitions reported:
      - target-field slice: LGBT_RE over `target_groups` ONLY (the claim
        under test: the *field* makes memory addressable);
      - v1-comparable keyword slice: LGBT_RE over the full archive text
        (target_groups + summary + modality cues), as in the v1 demo.

Writes <out_md> + <out_json>; never touches v1 artefacts.
"""
import argparse
import json
import os
import random
import sys
from collections import Counter

import numpy as np
import torch

ROOT = "/data/jehc223/RGCL"
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.join(ROOT, "scripts", "analysis"))

from memory_editing_demo import (  # noqa: E402  (protocol helpers, v1 script)
    ALPHA, TOPK, ARCHIVE_JSONL_TAG, LGBT_RE, CKPT,
    augment, knn_eval, slice_metrics, summarize_random, archive_text)
from eval_cross_dataset import project_split, build_head  # noqa: E402
from data_loader.dataset import (  # noqa: E402
    load_feats_from_CLIP, load_archive_feats_split, resolve_archive_path)
from easydict import EasyDict  # noqa: E402

import re  # noqa: E402

# English-pivot extension of the SAME family v1's LGBT_RE already covered in
# Chinese (娘炮/人妖 = effeminate-men slurs): v2 target_groups are English
# (schema pivot), so the family membership test must recognise the pivot
# phrasing too. This is a translation of the v1 family definition, not a new
# family. Applied ONLY to the target-field slice; the fulltext_v1style slice
# keeps LGBT_RE verbatim for v1 comparability.
LGBT_TARGET_RE = re.compile(
    LGBT_RE.pattern + r"|effeminate|sissy|femboy", re.I)

# ZH-dominant target family (women): the largest family in the ZH v2 archives
# (81 entries; 泼妇/母夜叉/破鞋/老鸨...), reported as a second directionality
# demo at a memory-slice size comparable to the EN v1 demo (91 entries).
WOMEN_TARGET_RE = re.compile(r"\bwomen\b|\bwoman\b|\bfemale|girls?\b|妇女|女性", re.I)


def load_archive_records_v2(ds):
    recs = {}
    for sp in ["train", "dev_seen", "test_seen"]:
        p = os.path.join(ROOT, "data/Archive", ds, "v2",
                         "{}_{}.jsonl".format(sp, ARCHIVE_JSONL_TAG))
        for line in open(p):
            r = json.loads(line)
            recs[r["id"]] = r
    return recs


def target_field_text(rec):
    a = rec.get("archive") or {}
    return " ".join(map(str, a.get("target_groups") or []))


def is_lgbt_target_field(rec):
    return bool(LGBT_TARGET_RE.search(target_field_text(rec)))


def is_lgbt_fulltext(rec):
    return bool(LGBT_RE.search(archive_text(rec)))


def is_women_target_field(rec):
    return bool(WOMEN_TARGET_RE.search(target_field_text(rec)))


def run(ds, args, out):
    cfg = CKPT[ds]  # frozen v1 winning config (head + logged reproduction gate)
    device = "cpu"
    clip_path = os.path.join(ROOT, "data", "CLIP_Embedding")
    train, dev, test = load_feats_from_CLIP(clip_path, ds, cfg["model"])
    model = build_head(train[1].shape[1], train[2].shape[1], EasyDict(
        eval_dataset=ds, num_layers=3, proj_dim=1024, map_dim=1024,
        fusion_mode="align", dropout=[0.2, 0.4, 0.1], batch_norm=False))
    model.load_state_dict(torch.load(cfg["ckpt"], map_location="cpu"))
    model.eval()

    tr_ids, tr_emb, tr_lab = project_split(model, train, device)
    te_ids, te_emb, te_lab = project_split(model, test, device)

    # v1 keys (reproduction gate vs the v1 demo / training log)
    arc_tr_v1 = load_archive_feats_split(
        resolve_archive_path("auto", os.path.join(ROOT, "data"), ds, "train"),
        tr_ids)
    arc_te_v1 = load_archive_feats_split(
        resolve_archive_path("auto", os.path.join(ROOT, "data"), ds,
                             "test_seen"), te_ids)
    # v2 keys (the channel under test)
    v2_dir = os.path.join(ROOT, "data", "CLIP_Embedding", ds, "v2")
    arc_tr = load_archive_feats_split(
        resolve_archive_path(v2_dir, None, ds, "train"), tr_ids)
    arc_te = load_archive_feats_split(
        resolve_archive_path(v2_dir, None, ds, "test_seen"), te_ids)

    mem_keys_v1 = augment(torch.tensor(tr_emb), arc_tr_v1)
    qry_keys_v1 = augment(torch.tensor(te_emb), arc_te_v1)
    mem_keys = augment(torch.tensor(tr_emb), arc_tr)
    qry_keys = augment(torch.tensor(te_emb), arc_te)

    recs = load_archive_records_v2(ds)
    slices = {
        "lgbt_target_field": (
            np.array([is_lgbt_target_field(recs[i]) for i in tr_ids]),
            np.array([is_lgbt_target_field(recs[i]) for i in te_ids])),
        "women_target_field": (
            np.array([is_women_target_field(recs[i]) for i in tr_ids]),
            np.array([is_women_target_field(recs[i]) for i in te_ids])),
        "lgbt_fulltext_v1style": (
            np.array([is_lgbt_fulltext(recs[i]) for i in tr_ids]),
            np.array([is_lgbt_fulltext(recs[i]) for i in te_ids])),
    }

    def evaluate(keep_mask, qry_slice, name, mk=None, qk=None):
        mk = mem_keys if mk is None else mk
        qk = qry_keys if qk is None else qk
        keep = np.where(keep_mask)[0]
        macro, preds, _ = knn_eval(
            mk[keep], tr_lab[keep], [tr_ids[j] for j in keep],
            qk, te_lab)
        return dict(
            edit=name, mem_n=int(keep_mask.sum()),
            removed=int((~keep_mask).sum()),
            overall=dict(acc=float(macro["acc"]),
                         macro_f1=float(macro["macro_f1"]),
                         roc=float(macro["roc"])),
            slice_lgbt=slice_metrics(preds, te_lab, qry_slice),
            slice_rest=slice_metrics(preds, te_lab, ~qry_slice),
        ), preds

    out[ds] = dict(config=dict(model=cfg["model"], ckpt=cfg["ckpt"],
                               v1_job=cfg["job"], sel_epoch=cfg["sel_epoch"],
                               logged=cfg["logged"], topk=TOPK, alpha=ALPHA,
                               archive_version="v2 keys under frozen v1 head"),
                   mem_n=len(tr_ids), test_n=len(te_ids), by_slice={})

    full = np.ones(len(tr_ids), dtype=bool)
    for sl_name, (mem_sl, qry_sl) in slices.items():
        results = []
        # reproduction gate: v1 keys, no edit (must match the logged numbers)
        row_v1, _ = evaluate(full, qry_sl, "baseline v1 keys (reproduction gate)",
                             mk=mem_keys_v1, qk=qry_keys_v1)
        results.append(row_v1)
        # v2-key baseline: the retraining-free key-swap comparison + edit base
        base_row, base_preds = evaluate(full, qry_sl, "baseline v2 keys (no edit)")
        results.append(base_row)

        def add_flips(row, preds):
            row["flips_lgbt"] = int(np.sum((preds != base_preds) & qry_sl))
            row["flips_rest"] = int(np.sum((preds != base_preds) & ~qry_sl))

        fam = {"women_target_field": "women"}.get(sl_name, "LGBTQ+")
        row, preds = evaluate(~mem_sl, qry_sl,
                              "(a) remove %s-targeting memory [%s]" % (fam, sl_name))
        add_flips(row, preds)
        row["removed_label_mix"] = dict(Counter(
            int(l) for l, m in zip(tr_lab, mem_sl) if m))
        row["removed_ids"] = [i for i, m in zip(tr_ids, mem_sl) if m]
        results.append(row)

        rand_rows = []
        for seed in range(5):
            rng = random.Random(seed)
            drop = set(rng.sample(range(len(tr_ids)), int(mem_sl.sum())))
            mask = np.array([j not in drop for j in range(len(tr_ids))])
            r, p = evaluate(mask, qry_sl, "(c) random control, seed %d" % seed)
            add_flips(r, p)
            rand_rows.append(r)
        results.append(summarize_random(rand_rows, "(c) random control, 5 seeds"))

        out[ds]["by_slice"][sl_name] = dict(
            mem_slice_n=int(mem_sl.sum()), qry_slice_n=int(qry_sl.sum()),
            results=results)
        print("[%s/%s] mem_slice=%d qry_slice=%d | v1-keys base acc=%.4f f1=%.4f "
              "(logged %.4f/%.4f) | v2-keys base acc=%.4f f1=%.4f | "
              "edit flips slice/rest = %d/%d" % (
                  ds, sl_name, mem_sl.sum(), qry_sl.sum(),
                  row_v1["overall"]["acc"], row_v1["overall"]["macro_f1"],
                  cfg["logged"]["acc"], cfg["logged"]["macro_f1"],
                  base_row["overall"]["acc"], base_row["overall"]["macro_f1"],
                  results[2]["flips_lgbt"], results[2]["flips_rest"]))


def fmt_metric(v):
    if isinstance(v, dict):
        return "{:.4f} [{:.4f},{:.4f}]".format(v["mean"], v["min"], v["max"])
    return "{:.4f}".format(v)


def fmt_flips(v):
    if isinstance(v, dict):
        return "{:.1f} [{:.0f},{:.0f}]".format(v["mean"], v["min"], v["max"])
    return str(v)


def write_markdown(out, ds, path):
    d = out[ds]
    L = []
    w = L.append
    w("# DEMO v2: ZH 记忆编辑定向性复测(archive prompt v2)")
    w("")
    w("> v1 诚实负结果的复测:ZH 组级删除 0 翻转,归因于 v1 档案 target 字段"
      "召回过低 (train 583 条仅 6 条非空)。本文用 **冻结的 v1 获胜 ckpt + v2 档案键**"
      "重跑同一协议 (topk=20, arithmetic, α=0.25):多 seed post-mortem + sha1 审计"
      "已确认 kNN 键通道不参与训练,换键无需重训,且隔离了重训混淆。"
      "切片样本量仍然很小,只报绝对数,不做显著性声明。")
    w("")
    c = d["config"]
    w("- ckpt: v1 获胜 job {} val-selected epoch {} (`{}`),档案键 = prompt v2。".format(
        c["v1_job"], c["sel_epoch"], os.path.basename(c["ckpt"])))
    w("- 记忆库 N={},test N={};复现门 = v1 键 baseline 应等于训练日志 "
      "{:.4f}/{:.4f} (acc/macro-F1)。".format(
          d["mem_n"], d["test_n"], c["logged"]["acc"], c["logged"]["macro_f1"]))
    w("")
    SLICE_TITLE = {
        "lgbt_target_field": "LGBTQ+ by target_groups 字段(claim 本体;v1 该切片 0 翻转)",
        "women_target_field": "女性 by target_groups 字段(ZH 最大目标族,记忆条目量级≈EN v1 演示)",
        "lgbt_fulltext_v1style": "LGBTQ+ by 全档案文本关键词(v1 同款,可比性)",
    }
    SLICE_FAM = {"lgbt_target_field": "LGBTQ+", "women_target_field": "女性",
                 "lgbt_fulltext_v1style": "LGBTQ+"}
    for sl_name, sl in d["by_slice"].items():
        fam = SLICE_FAM.get(sl_name, sl_name)
        w("## 切片定义: {}".format(SLICE_TITLE.get(sl_name, sl_name)))
        w("")
        w("- {fam} 记忆条目 {m} 条,{fam} test 切片 {q} 条。".format(
            fam=fam, m=sl["mem_slice_n"], q=sl["qry_slice_n"]))
        w("")
        n_l = sl["results"][0]["slice_lgbt"]["n"]
        n_r = sl["results"][0]["slice_rest"]["n"]
        w("| 行 | 删除数 | 整体 acc | 整体 macro-F1 | 切片 acc (n={}) | 其余 acc (n={}) | 翻转(切片/其余, vs v2键基线) |".format(n_l, n_r))
        w("|---|---|---|---|---|---|---|")
        for r in sl["results"]:
            flips = ("{} / {}".format(fmt_flips(r["flips_lgbt"]),
                                      fmt_flips(r["flips_rest"]))
                     if "flips_lgbt" in r else "—")
            w("| {} | {} | {} | {} | {} | {} | {} |".format(
                r["edit"], r["removed"],
                fmt_metric(r["overall"]["acc"]),
                fmt_metric(r["overall"]["macro_f1"]),
                fmt_metric(r["slice_lgbt"]["acc"]) if r["slice_lgbt"].get("n") else "n=0",
                fmt_metric(r["slice_rest"]["acc"]),
                flips))
        w("")
        ra = next(r for r in sl["results"] if str(r["edit"]).startswith("(a)"))
        if ra.get("removed_label_mix") is not None:
            w("- (a) 删除条目的 gt 标签构成: {}".format(ra["removed_label_mix"]))
        w("")
    w("(数字由 scripts/analysis/memory_editing_demo_v2.py 生成;结论正文由报告"
      "撰写时人工核对。)")
    with open(path, "w") as f:
        f.write("\n".join(L) + "\n")
    print("wrote", path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="MHC_zh", choices=["MHC_zh", "MHC"])
    ap.add_argument("--out_md", default=None,
                    help="default: research-wiki/DEMO_memory_editing_v2_<zh|en>.md")
    ap.add_argument("--out_json", required=True)
    args = ap.parse_args()
    if args.out_md is None:
        args.out_md = os.path.join(
            ROOT, "research-wiki/DEMO_memory_editing_v2_{}.md".format(
                "zh" if args.dataset == "MHC_zh" else "en"))
    torch.set_grad_enabled(False)
    out = {}
    run(args.dataset, args, out)
    with open(args.out_json, "w") as f:
        json.dump(out, f, indent=1)
    print("wrote", args.out_json)
    write_markdown(out, args.dataset, args.out_md)


if __name__ == "__main__":
    main()
