#!/usr/bin/env python
"""Role 3 step 3 (CPU): merge gate plans + arbitration outputs into the
selective-reasoning evaluation report.

Protocol discipline:
  - thresholds were chosen on VAL only (gate_margin.py);
  - the WORKING POINT (which deferral rate to deploy) is selected on VAL
    after-arbitration accuracy of the base variant; test numbers at that
    point are the headline. All three working points are still reported.
  - MLLM parse failures fall back to the kNN verdict (counted).

Writes research-wiki/EVAL_role3_selective_reasoning.md (+ raw JSON next to
the gate/arb outputs). Conclusions prose is human-written afterwards.
"""
import argparse
import json
import os
from collections import Counter

import numpy as np
from sklearn.metrics import f1_score

ROOT = "/data/jehc223/RGCL"
OUT_DIR = os.path.join(ROOT, "scripts/role3/out")
RATES = ["0.10", "0.20", "0.30"]
FINE_ANN = {
    "MHC": os.path.join(ROOT, "data/_src_Multihateclip/English/annotation(new).json"),
    "MHC_zh": os.path.join(ROOT, "data/_src_Multihateclip/Chinese/annotation(new).json"),
}


def load_gate(ds, variant):
    p = os.path.join(OUT_DIR, "gate_{}_{}.json".format(ds, variant))
    return json.load(open(p)) if os.path.exists(p) else None


def load_arb(ds, variant, split, mode):
    p = os.path.join(OUT_DIR, "arb_{}_{}_{}_{}.jsonl".format(ds, variant, split, mode))
    if not os.path.exists(p):
        return None
    recs = {}
    for line in open(p):
        line = line.strip()
        if line:
            r = json.loads(line)
            recs[r["id"]] = r  # last wins (resume runs)
    return recs


def load_fine_labels(ds):
    p = FINE_ANN.get(ds)
    if not p or not os.path.exists(p):
        return {}
    return {str(x["Video_ID"]): x.get("Label") for x in json.load(open(p))}


def metrics(preds, labels):
    preds = np.asarray(preds)
    labels = np.asarray(labels)
    return dict(acc=float(np.mean(preds == labels)),
                macro_f1=float(f1_score(labels, preds, average="macro",
                                        zero_division=0)))


def eval_working_points(gate, arb, split):
    """Before/after metrics at each rate for one (gate, arbitration) pair."""
    ss = [s for s in gate["samples"] if s["split"] == split]
    labels = np.array([s["label"] for s in ss])
    knn = np.array([s["pred_knn"] for s in ss])
    out = dict(n=len(ss), before=metrics(knn, labels), rates={})
    for rate in RATES:
        defer = np.array([s["defer"][rate] for s in ss])
        after = knn.copy()
        n_arb = n_fb = n_missing = 0
        det = []  # deferred-subset detail
        for i, s in enumerate(ss):
            if not defer[i]:
                continue
            r = (arb or {}).get(s["id"])
            if r is None:
                n_missing += 1
                det.append(dict(id=s["id"], gt=int(labels[i]),
                                knn=int(knn[i]), mllm=None, used=int(knn[i])))
                continue
            if r.get("verdict_bin") is None:
                n_fb += 1
                used = int(knn[i])
            else:
                n_arb += 1
                used = int(r["verdict_bin"])
                after[i] = used
            det.append(dict(id=s["id"], gt=int(labels[i]), knn=int(knn[i]),
                            mllm=r.get("verdict_bin"), verdict=r.get("verdict"),
                            used=used))
        d_gt = np.array([d["gt"] for d in det]) if det else np.array([])
        d_knn = np.array([d["knn"] for d in det]) if det else np.array([])
        d_used = np.array([d["used"] for d in det]) if det else np.array([])
        d_mllm = [d["mllm"] for d in det]
        have = [i for i, m in enumerate(d_mllm) if m is not None]
        out["rates"][rate] = dict(
            threshold=gate["thresholds"][rate],
            defer_n=int(defer.sum()), defer_rate=float(defer.mean()),
            mllm_calls=int(defer.sum()) - n_missing,
            parse_fallback=n_fb, missing=n_missing,
            after=metrics(after, labels),
            deferred_knn_acc=(float(np.mean(d_knn == d_gt)) if len(det) else None),
            deferred_used_acc=(float(np.mean(d_used == d_gt)) if len(det) else None),
            deferred_mllm_acc=(float(np.mean(
                np.array([d_mllm[i] for i in have]) ==
                np.array([d_gt[i] for i in have]))) if have else None),
            deferred_mllm_n=len(have),
            flips=int(np.sum(d_used != d_knn)) if len(det) else 0,
            flips_good=int(np.sum((d_used != d_knn) & (d_used == d_gt))) if len(det) else 0,
            flips_bad=int(np.sum((d_used != d_knn) & (d_used != d_gt))) if len(det) else 0,
            detail=det,
        )
    return out


def fine_composition(gate, fine, split, rate):
    ss = [s for s in gate["samples"] if s["split"] == split]
    comp = lambda group: Counter(fine.get(s["id"], "?") for s in group)  # noqa: E731
    deferred = [s for s in ss if s["defer"][rate]]
    rest = [s for s in ss if not s["defer"][rate]]
    return dict(deferred=dict(comp(deferred)), non_deferred=dict(comp(rest)))


def flip_examples(gate, arb, split, rate, k=3):
    ss = {s["id"]: s for s in gate["samples"] if s["split"] == split}
    ex = []
    for sid, s in ss.items():
        if not s["defer"][rate]:
            continue
        r = (arb or {}).get(sid)
        if not r or r.get("verdict_bin") is None:
            continue
        if int(r["verdict_bin"]) == int(s["pred_knn"]):
            continue
        ex.append(dict(
            id=sid, gt=s["label"], knn=s["pred_knn"],
            verdict=r["verdict"], verdict_bin=r["verdict_bin"],
            good=bool(int(r["verdict_bin"]) == int(s["label"])),
            margin=round(s["margin"], 4),
            key_evidence=r.get("key_evidence", ""),
            cited_neighbor=r.get("cited_neighbor", ""),
            title=(s["text"] or "")[:160],
        ))
    ex.sort(key=lambda e: (not e["good"], e["margin"]))
    good = [e for e in ex if e["good"]][:2]
    bad = [e for e in ex if not e["good"]][:1]
    return (good + bad)[:k], len(ex)


def pct(x, nd=4):
    return "—" if x is None else "{:.4f}".format(x)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_md", default=os.path.join(
        ROOT, "research-wiki/EVAL_role3_selective_reasoning.md"))
    ap.add_argument("--out_json", default=os.path.join(
        OUT_DIR, "role3_results.json"))
    args = ap.parse_args()

    runs = [
        # ds, variant, mode  (test evaluation)
        ("MHC", "base", "frames"),
        ("MHC", "clean", "frames"),
        ("MHC", "base", "textonly"),
        ("MHC_zh", "base", "frames"),
        ("MHC_zh", "base", "textonly"),
    ]
    R = {}
    for ds, variant, mode in runs:
        gate = load_gate(ds, variant)
        if gate is None:
            continue
        arb_t = load_arb(ds, variant, "test", mode)
        key = "{}|{}|{}".format(ds, variant, mode)
        R[key] = dict(test=eval_working_points(gate, arb_t, "test"))
        if variant == "base" and mode == "frames":
            arb_v = load_arb(ds, variant, "val", mode)
            R[key]["val"] = eval_working_points(gate, arb_v, "val")

    # working-point selection on VAL (base+frames), per dataset
    sel = {}
    for ds in ["MHC", "MHC_zh"]:
        key = "{}|base|frames".format(ds)
        if key not in R or "val" not in R[key]:
            continue
        val = R[key]["val"]
        best = max(RATES, key=lambda r: (
            val["rates"][r]["after"]["acc"],
            val["rates"][r]["after"]["macro_f1"],
            -float(r)))
        sel[ds] = dict(rate=best,
                       val_after=val["rates"][best]["after"],
                       val_before=val["before"])
    # cost accounting from arb wall times
    cost = {}
    for ds, variant, mode in runs:
        for split in ["val", "test"]:
            recs = load_arb(ds, variant, split, mode)
            if not recs:
                continue
            ws = [r["wall_s"] for r in recs.values() if r.get("wall_s")]
            cost["{}|{}|{}|{}".format(ds, variant, split, mode)] = dict(
                calls=len(recs), mean_wall_s=float(np.mean(ws)) if ws else None,
                total_wall_s=float(np.sum(ws)) if ws else None)

    flips = {}
    for ds in ["MHC", "MHC_zh"]:
        gate = load_gate(ds, "base")
        arb = load_arb(ds, "base", "test", "frames")
        if gate and arb and ds in sel:
            flips[ds] = flip_examples(gate, arb, "test", sel[ds]["rate"])

    fine = {}
    for ds in ["MHC", "MHC_zh"]:
        gate = load_gate(ds, "base")
        fl = load_fine_labels(ds)
        if gate and fl:
            fine[ds] = fine_composition(gate, fl, "test", "0.30")

    dump = dict(results=R, selection=sel, cost=cost, flips=flips, fine=fine)
    with open(args.out_json, "w") as f:
        json.dump(dump, f, ensure_ascii=False, indent=1)
    print("wrote", args.out_json)

    write_md(args.out_md, R, sel, cost, flips, fine)
    print("wrote", args.out_md)


def write_md(path, R, sel, cost, flips, fine):
    L = []
    w = L.append
    w("# EVAL: Role 3 — 置信门控的选择性推理(kNN margin gate + Qwen2.5-VL 仲裁)")
    w("")
    w("> **诚实条款**:deferred 子集很小(test 每工作点 15–42 条),只报绝对数,不做显著性声明。"
      "门槛全部在 val 上选(deferral 率 ≈10/20/30% 三个工作点),工作点本身也由 val 仲裁后 acc 选出;"
      "test 不参与任何调参。MLLM 解析失败回退 kNN 判决(计数在表中)。")
    w("")
    w("## 协议")
    w("")
    w("- 基座:获胜 archive-kNN α=0.25 seed0(EN frozen-Qwen job 12210 ckpt epoch24;"
      "ZH LoRA job 12207 ckpt epoch18),零训练;复现门 bit-identical "
      "(EN 0.8075/0.7626, ZH 0.8523/0.8270,见 gate_MHC_base.json / gate_MHC_zh_base.json)。")
    w("- 门控:margin = |similarity-signed arithmetic vote|(与训练日志逐位一致的投票);"
      "val margin 分位数取门槛,defer ⇔ margin < t(三工作点嵌套)。")
    w("- 仲裁:frozen Qwen2.5-VL-7B-Instruct,输入 = 16 帧 + title/transcript + 该视频自己的档案"
      " + top-5 检索邻居的档案+gt 标签(证据卡);输出严格 JSON "
      "{verdict: hateful/offensive/normal, key_evidence, cited_neighbor};"
      "hateful/offensive→1, normal→0 后仅替换 deferred 样本的判决。")
    w("- 变体:base;memory-clean(先删 DEMO_memory_editing 的 2 条 W2 噪声记忆,合法:源于训练侧取证);"
      "text-only 仲裁对照(不给帧)。")
    w("")
    for ds, name in [("MHC", "MHC (EN)"), ("MHC_zh", "MHC_zh (ZH)")]:
        keys = [k for k in R if k.startswith(ds + "|")]
        if not keys:
            continue
        w("## {}".format(name))
        w("")
        if ds in sel:
            s = sel[ds]
            w("**val 选定工作点:deferral ≈ {:.0f}%**(val after-acc {:.4f},"
              "before {:.4f};base+frames 变体)。".format(
                  float(s["rate"]) * 100, s["val_after"]["acc"],
                  s["val_before"]["acc"]))
            w("")
        # val table (base frames)
        key = ds + "|base|frames"
        if key in R and "val" in R[key]:
            v = R[key]["val"]
            w("### val(工作点选择依据;N={})".format(v["n"]))
            w("")
            w("| rate | defer n | before acc/F1 | after acc/F1 | 仲裁正确率(MLLM) | kNN在deferred上 | flips(好/坏) | 回退 |")
            w("|---|---|---|---|---|---|---|---|")
            for r in RATES:
                d = v["rates"][r]
                w("| {} | {} ({:.1%}) | {:.4f} / {:.4f} | {:.4f} / {:.4f} | {} (n={}) | {} | {} ({}/{}) | {} |".format(
                    r, d["defer_n"], d["defer_rate"],
                    v["before"]["acc"], v["before"]["macro_f1"],
                    d["after"]["acc"], d["after"]["macro_f1"],
                    pct(d["deferred_mllm_acc"]), d["deferred_mllm_n"],
                    pct(d["deferred_knn_acc"]),
                    d["flips"], d["flips_good"], d["flips_bad"],
                    d["parse_fallback"]))
            w("")
        w("### test(before/after,每变体)")
        w("")
        w("| 变体 | rate | defer n (率) | MLLM calls | before acc/F1 | after acc/F1 | Δacc | 仲裁正确率 | kNN@deferred | flips(好/坏) | 回退 |")
        w("|---|---|---|---|---|---|---|---|---|---|---|")
        for variant, mode in [("base", "frames"), ("clean", "frames"),
                              ("base", "textonly")]:
            key = "{}|{}|{}".format(ds, variant, mode)
            if key not in R:
                continue
            t = R[key]["test"]
            for r in RATES:
                d = t["rates"][r]
                star = " **⟵ val选定**" if ds in sel and r == sel[ds]["rate"] and variant == "base" and mode == "frames" else ""
                w("| {}+{} | {}{} | {} ({:.1%}) | {} | {:.4f} / {:.4f} | {:.4f} / {:.4f} | {:+.4f} | {} (n={}) | {} | {} ({}/{}) | {} |".format(
                    variant, mode, r, star, d["defer_n"], d["defer_rate"],
                    d["mllm_calls"],
                    t["before"]["acc"], t["before"]["macro_f1"],
                    d["after"]["acc"], d["after"]["macro_f1"],
                    d["after"]["acc"] - t["before"]["acc"],
                    pct(d["deferred_mllm_acc"]), d["deferred_mllm_n"],
                    pct(d["deferred_knn_acc"]),
                    d["flips"], d["flips_good"], d["flips_bad"],
                    d["parse_fallback"]))
        w("")
        if ds in fine:
            w("### deferred 切片的三分类构成(rate 0.30,test;原始 MultiHateClip 标注)")
            w("")
            w("- deferred: {}".format(dict(fine[ds]["deferred"])))
            w("- non-deferred: {}".format(dict(fine[ds]["non_deferred"])))
            w("")
        if ds in flips:
            ex, n_flips = flips[ds]
            w("### 翻转案例(val 选定工作点,test;共 {} 个翻转)".format(n_flips))
            w("")
            for e in ex:
                w("- `{}` gt={} kNN={} → MLLM **{}**({},margin {:.3f});证据:{} "
                  "(cited: {});标题:{}".format(
                      e["id"], e["gt"], e["knn"], e["verdict"],
                      "✔ 纠正" if e["good"] else "✘ 改错", e["margin"],
                      (e["key_evidence"] or "")[:220], e["cited_neighbor"],
                      e["title"].replace("|", "\\|")))
            w("")
    w("## 成本核算素材(MLLM 调用)")
    w("")
    w("| run | calls | 平均 wall s/call | 总 wall s |")
    w("|---|---|---|---|")
    for k, c in cost.items():
        w("| {} | {} | {} | {} |".format(
            k, c["calls"],
            "—" if c["mean_wall_s"] is None else "{:.1f}".format(c["mean_wall_s"]),
            "—" if c["total_wall_s"] is None else "{:.0f}".format(c["total_wall_s"])))
    w("")
    w("## 结论")
    w("")
    w("(数字由 scripts/role3/analyze_role3.py 生成;结论正文人工撰写,见下一次编辑。"
      "原始 JSON:scripts/role3/out/role3_results.json;门控计划 gate_*.json;"
      "仲裁输出 arb_*.jsonl。)")
    with open(path, "w") as f:
        f.write("\n".join(L) + "\n")


if __name__ == "__main__":
    main()
