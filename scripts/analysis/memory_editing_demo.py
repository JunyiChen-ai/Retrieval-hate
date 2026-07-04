#!/usr/bin/env python
"""Experiment 2: memory-editing capability demo on the winning archive-kNN configs.

Scenario: a platform must (a) remove a whole class of memory targeting one
group, or (b) take down suspected-mislabelled entries. Because the RGCL-video
classifier is a kNN vote over an explicit train-key memory bank, both are
pure CPU index edits -- no retraining.

Protocol (mirrors the winning runs bit-for-bit):
  - ckpt   : val-selected best head of jobs 12210 (MHC) / 12207 (MHC_zh)
  - memory : TRAIN split fused embeds, keys = [l2n(fused) | 0.25*l2n(archive)]
  - query  : TEST split, same key construction
  - vote   : topk=20 faiss cosine, arithmetic weights, similarity-signed
             (use_sim=True), exactly compute_metrics_retrieval as in run_rac.

Edits:
  (a) group takedown  : delete every train key whose ARCHIVE fields mention the
      LGBTQ+ family (top target family in the archives); measure the
      archive-defined LGBTQ+ test slice vs the remaining slice.
  (b) noise takedown  : delete W2-flagged suspected label-noise entries
      (XScP1AiMkNM, QvPp8Q7QhWE) plus same-mechanism low-confidence entries
      (label=1 but archive says: no targets, explicitness=none,
      mechanism subset of {coded_language}).
  (c) random control  : delete the same number of random train keys (5 seeds).

Only reads src/ (project_split, head, loaders, metrics); writes
research-wiki/DEMO_memory_editing.md + a JSON dump in the scratch dir.
"""
import argparse
import json
import os
import random
import re
import sys
from collections import Counter

import numpy as np
import torch
import faiss

ROOT = "/data/jehc223/RGCL"
sys.path.insert(0, os.path.join(ROOT, "src"))

from eval_cross_dataset import project_split, build_head  # noqa: E402  (read-only import)
from data_loader.dataset import (  # noqa: E402
    load_feats_from_CLIP, load_archive_feats_split, resolve_archive_path)
from utils.metrics import compute_metrics_retrieval, sigmoid  # noqa: E402
from easydict import EasyDict  # noqa: E402

TOPK = 20
ALPHA = 0.25
ARCHIVE_JSONL_TAG = "Qwen2.5-VL-7B-Instruct_archive"
CKPT = {
    "MHC": dict(
        model="Qwen2.5-VL-7B-Instruct_HF",
        job=12210, sel_epoch=24,
        logged=dict(acc=0.8075, macro_f1=0.7626),
        ckpt=os.path.join(
            ROOT, "logging/Retrieval/MHC/RAC_video_archive",
            "RAC_lr0.0001_Bz64_Ep30_cosSim_triplet_drop[0.2, 0.4, 0.1]_topK20_"
            "_PseudoGold_positive_1_hard_negative_1_seed0_hybrid_loss_"
            "Qwen2.5-VL-7B-Instruct_HF_arc-knn-a0.25/ckpt/best_model_24_0.7875.pt"),
    ),
    "MHC_zh": dict(
        model="Qwen2.5-VL-7B-Instruct-LoRA_HF",
        job=12207, sel_epoch=18,
        logged=dict(acc=0.8523, macro_f1=0.8270),
        ckpt=os.path.join(
            ROOT, "logging/Retrieval/MHC_zh/RAC_video_archive",
            "RAC_lr0.0001_Bz64_Ep30_cosSim_triplet_drop[0.2, 0.4, 0.1]_topK20_"
            "_PseudoGold_positive_1_hard_negative_1_seed0_hybrid_loss_"
            "Qwen2.5-VL-7B-Instruct-LoRA_HF_arc-knn-a0.25/ckpt/"
            "best_model_18_0.8717948717948718.pt"),
    ),
}
NOISY_IDS = ["XScP1AiMkNM", "QvPp8Q7QhWE"]  # W2 suspected label noise (MHC train)
LGBT_RE = re.compile(
    r"lgbt|gay|lesbian|homosex|queer|trans(gender|\b|sexual)|drag queen|asexual"
    r"|bisex|same[- ]sex|sexual orientation|同性|男同|女同|跨性别|变性|性取向"
    r"|性少数|娘炮|人妖", re.I)


# --------------------------------------------------------------------------- #
def load_archive_records(ds):
    """id -> archive JSONL record (all splits, last occurrence wins)."""
    recs = {}
    for sp in ["train", "dev_seen", "test_seen"]:
        p = os.path.join(ROOT, "data/Archive", ds,
                         "{}_{}.jsonl".format(sp, ARCHIVE_JSONL_TAG))
        for line in open(p):
            r = json.loads(line)
            recs[r["id"]] = r
    return recs


def archive_text(rec):
    a = rec.get("archive") or {}
    mc = a.get("modality_cues") or {}
    parts = [" ".join(map(str, a.get("target_groups") or [])),
             a.get("neutral_summary") or ""]
    parts += [str(v or "") for v in mc.values()]
    return " ".join(parts)


def is_lgbt(rec):
    return bool(LGBT_RE.search(archive_text(rec)))


def is_low_conf_noiselike(rec):
    """Same rule that captures both W2-flagged samples: gt says harmful but the
    archive sees nothing (no targets, explicitness none, mechanism at most
    coded_language)."""
    a = rec.get("archive") or {}
    return (int(rec.get("label", 0)) == 1
            and not (a.get("target_groups") or [])
            and (a.get("explicitness") in ("none", None))
            and set(a.get("mechanism") or []) <= {"coded_language"})


def augment(fused, arc, alpha=ALPHA):
    fused_n = torch.nn.functional.normalize(fused, p=2, dim=1)
    arc_n = torch.nn.functional.normalize(arc.float(), p=2, dim=1)
    return torch.cat((fused_n, alpha * arc_n), dim=1).numpy().astype("float32")


def knn_eval(mem_keys, mem_labels, mem_ids, qry_keys, qry_labels):
    """Winning-config vote: faiss cosine topk=20, arithmetic weights,
    similarity-signed (use_sim=True). Returns (macro dict, preds, votes)."""
    mem = mem_keys.copy()
    qry = qry_keys.copy()
    faiss.normalize_L2(mem)
    faiss.normalize_L2(qry)
    index = faiss.IndexFlatIP(mem.shape[1])
    index.add(mem)
    k = min(TOPK, mem.shape[0])
    D, I = index.search(qry, k)
    logging_dict = EasyDict()
    for i, row in enumerate(D):
        logging_dict["q%d" % i] = {
            "no_retrieved": len(row),
            "retrieved_ids": [mem_ids[j] for j in I[i]],
            "retrieved_scores": [np.float32(v) for v in row],
            "retrieved_label": [int(mem_labels[j]) for j in I[i]],
        }
    acc, roc, pre, recall, f1, votes, _, macro = compute_metrics_retrieval(
        logging_dict, qry_labels, majority_voting="arithmetic", topk=TOPK,
        use_sim=True)
    preds = (sigmoid(np.array(votes)) >= 0.5).astype(int)
    return macro, preds, np.array(votes)


def slice_metrics(preds, labels, mask):
    from sklearn.metrics import f1_score
    idx = np.where(mask)[0]
    if len(idx) == 0:
        return dict(n=0)
    y, p = labels[idx], preds[idx]
    return dict(n=int(len(idx)), acc=float(np.mean(y == p)),
                macro_f1=float(f1_score(y, p, average="macro",
                                        zero_division=0)),
                pos=int(y.sum()))


# --------------------------------------------------------------------------- #
def run_dataset(ds, out):
    cfg = CKPT[ds]
    device = "cpu"
    clip_path = os.path.join(ROOT, "data", "CLIP_Embedding")
    train, dev, test = load_feats_from_CLIP(clip_path, ds, cfg["model"])
    model = build_head(train[1].shape[1], train[2].shape[1], EasyDict(
        eval_dataset=ds, num_layers=3, proj_dim=1024, map_dim=1024,
        fusion_mode="align", dropout=[0.2, 0.4, 0.1], batch_norm=False))
    model.load_state_dict(torch.load(cfg["ckpt"], map_location="cpu"))

    tr_ids, tr_emb, tr_lab = project_split(model, train, device)
    te_ids, te_emb, te_lab = project_split(model, test, device)

    # archive CLIP-text embeddings (the kNN key augmentation channel)
    arc_tr = load_archive_feats_split(
        resolve_archive_path("auto", os.path.join(ROOT, "data"), ds, "train"),
        tr_ids)
    arc_te = load_archive_feats_split(
        resolve_archive_path("auto", os.path.join(ROOT, "data"), ds,
                             "test_seen"), te_ids)
    mem_keys = augment(torch.tensor(tr_emb), arc_tr)
    qry_keys = augment(torch.tensor(te_emb), arc_te)

    recs = load_archive_records(ds)
    lgbt_mem = np.array([is_lgbt(recs[i]) for i in tr_ids])
    lgbt_qry = np.array([is_lgbt(recs[i]) for i in te_ids])
    noise_mem = np.array([recs[i]["id"] in NOISY_IDS or
                          is_low_conf_noiselike(recs[i]) for i in tr_ids])
    noisy_present = [i for i in tr_ids if i in NOISY_IDS]

    def evaluate(keep_mask, name):
        keep = np.where(keep_mask)[0]
        macro, preds, votes = knn_eval(
            mem_keys[keep], tr_lab[keep], [tr_ids[j] for j in keep],
            qry_keys, te_lab)
        row = dict(
            edit=name, mem_n=int(keep_mask.sum()),
            removed=int((~keep_mask).sum()),
            overall=dict(acc=float(macro["acc"]),
                         macro_f1=float(macro["macro_f1"]),
                         roc=float(macro["roc"])),
            slice_lgbt=slice_metrics(preds, te_lab, lgbt_qry),
            slice_rest=slice_metrics(preds, te_lab, ~lgbt_qry),
        )
        return row, preds

    results = []
    full = np.ones(len(tr_ids), dtype=bool)
    base_row, base_preds = evaluate(full, "baseline (no edit)")
    results.append(base_row)

    def add_flips(row, preds):
        row["flips_lgbt"] = int(np.sum((preds != base_preds) & lgbt_qry))
        row["flips_rest"] = int(np.sum((preds != base_preds) & ~lgbt_qry))

    # (a) group takedown
    row, preds = evaluate(~lgbt_mem, "(a) remove LGBTQ+-targeting memory")
    add_flips(row, preds)
    row["removed_label_mix"] = dict(Counter(
        int(l) for l, m in zip(tr_lab, lgbt_mem) if m))
    results.append(row)

    # (a) random control, size-matched, 5 seeds
    rand_rows = []
    for seed in range(5):
        rng = random.Random(seed)
        drop = set(rng.sample(range(len(tr_ids)), int(lgbt_mem.sum())))
        mask = np.array([j not in drop for j in range(len(tr_ids))])
        r, p = evaluate(mask, "(c) random control for (a), seed %d" % seed)
        add_flips(r, p)
        rand_rows.append(r)
    results.append(summarize_random(rand_rows, "(c) random control for (a), 5 seeds"))

    # (b) noise takedown
    name_b = ("(b) remove W2 noisy ids + same-mechanism low-confidence"
              if noisy_present else
              "(b) rule-only low-confidence takedown (no W2-flagged ids in this dataset)")
    row, preds = evaluate(~noise_mem, name_b)
    add_flips(row, preds)
    row["removed_ids"] = [i for i, m in zip(tr_ids, noise_mem) if m]
    row["w2_ids_present"] = noisy_present
    results.append(row)

    if noisy_present:
        just2 = np.array([i not in NOISY_IDS for i in tr_ids])
        row, preds = evaluate(just2, "(b') remove ONLY the 2 W2 noisy ids")
        add_flips(row, preds)
        results.append(row)

    # (b) random control
    rand_rows = []
    for seed in range(5):
        rng = random.Random(100 + seed)
        drop = set(rng.sample(range(len(tr_ids)), int(noise_mem.sum())))
        mask = np.array([j not in drop for j in range(len(tr_ids))])
        r, p = evaluate(mask, "(c) random control for (b), seed %d" % seed)
        add_flips(r, p)
        rand_rows.append(r)
    results.append(summarize_random(rand_rows, "(c) random control for (b), 5 seeds"))

    out[ds] = dict(
        config=dict(model=cfg["model"], job=cfg["job"],
                    sel_epoch=cfg["sel_epoch"], ckpt=cfg["ckpt"],
                    logged=cfg["logged"], topk=TOPK, alpha=ALPHA),
        mem_n=len(tr_ids), test_n=len(te_ids),
        lgbt_mem_n=int(lgbt_mem.sum()), lgbt_qry_n=int(lgbt_qry.sum()),
        noise_mem_n=int(noise_mem.sum()),
        results=results,
    )
    print("[%s] baseline acc=%.4f macroF1=%.4f (logged %.4f/%.4f)" % (
        ds, base_row["overall"]["acc"], base_row["overall"]["macro_f1"],
        cfg["logged"]["acc"], cfg["logged"]["macro_f1"]))


def summarize_random(rows, name):
    def agg(path):
        vals = []
        for r in rows:
            v = r
            for k in path:
                v = v[k]
            vals.append(v)
        return dict(mean=float(np.mean(vals)), min=float(np.min(vals)),
                    max=float(np.max(vals)))
    return dict(
        edit=name, mem_n=rows[0]["mem_n"], removed=rows[0]["removed"],
        overall=dict(acc=agg(["overall", "acc"]),
                     macro_f1=agg(["overall", "macro_f1"])),
        slice_lgbt=dict(n=rows[0]["slice_lgbt"]["n"],
                        acc=agg(["slice_lgbt", "acc"]),
                        macro_f1=agg(["slice_lgbt", "macro_f1"])),
        slice_rest=dict(n=rows[0]["slice_rest"]["n"],
                        acc=agg(["slice_rest", "acc"]),
                        macro_f1=agg(["slice_rest", "macro_f1"])),
        flips_lgbt=agg(["flips_lgbt"]), flips_rest=agg(["flips_rest"]),
        seeds=len(rows), per_seed=rows,
    )


# --------------------------------------------------------------------------- #
def fmt_metric(v):
    if isinstance(v, dict):  # random-control aggregate
        return "{:.4f} [{:.4f},{:.4f}]".format(v["mean"], v["min"], v["max"])
    return "{:.4f}".format(v)


def fmt_flips(v):
    if isinstance(v, dict):
        return "{:.1f} [{:.0f},{:.0f}]".format(v["mean"], v["min"], v["max"])
    return str(v)


def write_markdown(out, path):
    L = []
    w = L.append
    w("# DEMO: kNN 记忆编辑 — 档案字段让记忆库可定向编辑(capability demo)")
    w("")
    w("> **诚实条款**:test 切片样本量很小(LGBTQ+ 切片 EN n={}、ZH n={});"
      "本文只报告绝对数,不做显著性声明。定位是**能力演示**——证明档案字段"
      "使记忆库支持按语义划片的定向编辑——而非性能 claim。".format(
          out["MHC"]["lgbt_qry_n"], out["MHC_zh"]["lgbt_qry_n"]))
    w("")
    w("## 场景与操作")
    w("")
    w("- 场景: 平台需要 (a) \"移除针对某群体的整类记忆\"(演示组: LGBTQ+,档案中最高频目标族)"
      "或 (b) \"下架被误标条目\"(W2 发现的 `XScP1AiMkNM` 牛油果酱、`QvPp8Q7QhWE` 数钱,"
      "以及同 mechanism 的低置信条目)。")
    w("- 记忆库 = 获胜配置(archive-kNN α=0.25)的 train 键;编辑 = 直接从 faiss 索引删除对应"
      "train 条目,**纯 CPU,零训练**。查询协议与训练日志逐位一致"
      "(topk=20、arithmetic 加权、相似度符号投票)。")
    w("- 条目归属完全由**档案字段**决定(target_groups / neutral_summary / modality_cues 的"
      "关键词匹配;低置信 = label=harmful 但档案报 无 target + explicitness=none + "
      "mechanism⊆{coded_language},该规则恰好同时命中两条 W2 噪声样本)。"
      "gt 标签与原始文本不参与编辑决策。")
    w("- 对照: (c) 每次编辑配等量随机删除(5 seeds,报 mean [min,max])。")
    w("")
    for ds in ["MHC", "MHC_zh"]:
        d = out[ds]
        c = d["config"]
        w("## {} ({})".format(ds, "EN" if ds == "MHC" else "ZH"))
        w("")
        w("- ckpt: job {} val-selected epoch {} (`{}`)".format(
            c["job"], c["sel_epoch"], os.path.basename(c["ckpt"])))
        w("- 记忆库 N={},test N={};LGBTQ+ 记忆条目 {} 条,LGBTQ+ test 切片 {} 条,"
          "低置信(噪声样)记忆条目 {} 条。".format(
              d["mem_n"], d["test_n"], d["lgbt_mem_n"], d["lgbt_qry_n"],
              d["noise_mem_n"]))
        base = d["results"][0]
        w("- 复现门:baseline acc {:.4f} / macro-F1 {:.4f},训练日志记录 {:.4f} / {:.4f}。".format(
            base["overall"]["acc"], base["overall"]["macro_f1"],
            c["logged"]["acc"], c["logged"]["macro_f1"]))
        w("")
        w("| 编辑 | 删除数 | 整体 acc | 整体 macro-F1 | LGBTQ+切片 acc (n={}) | 其余切片 acc (n={}) | 翻转(切片/其余) |".format(
            d["results"][0]["slice_lgbt"]["n"], d["results"][0]["slice_rest"]["n"]))
        w("|---|---|---|---|---|---|---|")
        for r in d["results"]:
            flips = ("{} / {}".format(fmt_flips(r["flips_lgbt"]),
                                      fmt_flips(r["flips_rest"]))
                     if "flips_lgbt" in r else "—")
            w("| {} | {} | {} | {} | {} | {} | {} |".format(
                r["edit"], r["removed"],
                fmt_metric(r["overall"]["acc"]),
                fmt_metric(r["overall"]["macro_f1"]),
                fmt_metric(r["slice_lgbt"]["acc"]),
                fmt_metric(r["slice_rest"]["acc"]),
                flips))
        w("")
        rb = next(r for r in d["results"] if r["edit"].startswith("(b)"))
        if rb.get("removed_ids"):
            w("- (b) 删除条目: {}".format(", ".join(
                "`%s`" % i for i in rb["removed_ids"])))
        ra = next(r for r in d["results"] if r["edit"].startswith("(a)"))
        if "removed_label_mix" in ra:
            w("- (a) 删除条目的 gt 标签构成: {}".format(ra["removed_label_mix"]))
        w("")
    w("## 结论(定向性)")
    w("")
    w("(由脚本生成数字,结论正文在报告撰写时人工填写/核对。原始 JSON: 见 scratch 目录 "
      "memory_editing_results.json,本 markdown 由 "
      "scripts/analysis/memory_editing_demo.py 生成。)")
    with open(path, "w") as f:
        f.write("\n".join(L) + "\n")
    print("wrote", path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_md", default=os.path.join(
        ROOT, "research-wiki/DEMO_memory_editing.md"))
    ap.add_argument("--out_json", required=True)
    args = ap.parse_args()
    torch.set_grad_enabled(False)
    out = {}
    for ds in ["MHC", "MHC_zh"]:
        run_dataset(ds, out)
    with open(args.out_json, "w") as f:
        json.dump(out, f, indent=1)
    print("wrote", args.out_json)
    write_markdown(out, args.out_md)


if __name__ == "__main__":
    main()
