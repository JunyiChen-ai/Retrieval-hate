#!/usr/bin/env python
"""TARC G2 scorer — predicted target-community vs GT primary (exp-tarc-t0.md §6-G2).

Reads the MLLM predictions (target_pred_<tag>.jsonl) and the GT oracle map
(target_map.json primary). GT is read ONLY here, for scoring — never by the
predictor. Reports, on two populations (all split videos with a GT target, and
hate-only), for two label sets:
  - full 8-class macro-F1 (classes present in GT), and
  - effective 3-class macro-F1 {Blacks, Jews, Other-merged} (the §6 gate metric),
plus a confusion matrix (effective set) and the parse-failure rate.

GATE (pre-registered §6): effective 3-class macro-F1 >= 0.60.
"""
import argparse
import json
import os
from collections import Counter, defaultdict

CODE_NAME = {0: "Blacks", 1: "Jews", 2: "Whites", 3: "Others",
             4: "LGBTQ", 5: "Muslims", 6: "Sexits", 7: "Asian", -1: "None"}
# effective 3-class: Blacks=0, Jews=1, Other=2 (any real code not 0/1); None=99
EFF_OTHER = 2
EFF_NONE = 99
EFF_NAME = {0: "Blacks", 1: "Jews", 2: "Other", 99: "None(pred)"}


def to_eff(code):
    if code == 0:
        return 0
    if code == 1:
        return 1
    if code == -1:
        return EFF_NONE
    return EFF_OTHER  # 2..7 -> Other


def macro_f1(pairs, classes):
    """pairs: list of (y_true, y_pred). Average F1 over `classes` (GT label set).
    Returns (macro_f1, per_class dict of (P,R,F1,support))."""
    tp = Counter(); fp = Counter(); fn = Counter()
    for yt, yp in pairs:
        if yt == yp:
            tp[yt] += 1
        else:
            fp[yp] += 1
            fn[yt] += 1
    per = {}
    f1s = []
    for c in classes:
        p = tp[c] / (tp[c] + fp[c]) if (tp[c] + fp[c]) else 0.0
        r = tp[c] / (tp[c] + fn[c]) if (tp[c] + fn[c]) else 0.0
        f1 = 2 * p * r / (p + r) if (p + r) else 0.0
        sup = tp[c] + fn[c]
        per[c] = (p, r, f1, sup)
        f1s.append(f1)
    return (sum(f1s) / len(f1s) if f1s else 0.0), per


def read_preds(path):
    preds = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            o = json.loads(line)
            preds[str(o["id"])] = {"primary": int(o["primary"]),
                                   "parse_ok": bool(o.get("parse_ok", True)),
                                   "video_ok": bool(o.get("video_ok", True))}
    return preds


def read_gt_primary(target_map_path):
    d = json.load(open(target_map_path))
    return {k: v["primary"] for k, v in d.items() if not k.startswith("_")}


def read_labels(gt_dir, splits):
    lab = {}
    for s in splits:
        p = os.path.join(gt_dir, "{}.jsonl".format(s))
        with open(p) as f:
            for line in f:
                line = line.strip()
                if line:
                    o = json.loads(line)
                    lab[str(o["id"])] = int(o["label"])
    return lab


def report_population(name, pairs8, out):
    out.append("\n=== population: {} (n={}) ===".format(name, len(pairs8)))
    gt_classes8 = sorted({yt for yt, _ in pairs8})
    m8, per8 = macro_f1(pairs8, gt_classes8)
    out.append("  8-class macro-F1 (over GT-present classes {}): {:.4f}".format(
        [CODE_NAME[c] for c in gt_classes8], m8))
    for c in gt_classes8:
        p, r, f1, sup = per8[c]
        out.append("    {:8s} P={:.3f} R={:.3f} F1={:.3f} sup={}".format(
            CODE_NAME[c], p, r, f1, sup))
    # effective 3-class
    eff = [(to_eff(yt), to_eff(yp)) for yt, yp in pairs8]
    eff_classes = [0, 1, 2]  # Blacks, Jews, Other (GT never -1 here)
    m3, per3 = macro_f1(eff, eff_classes)
    out.append("  EFFECTIVE 3-class macro-F1 {{Blacks,Jews,Other}}: {:.4f}   <-- GATE metric".format(m3))
    for c in eff_classes:
        p, r, f1, sup = per3[c]
        out.append("    {:6s} P={:.3f} R={:.3f} F1={:.3f} sup={}".format(
            EFF_NAME[c], p, r, f1, sup))
    # confusion matrix (effective, rows=GT, cols=pred incl None)
    cm = defaultdict(Counter)
    for yt, yp in eff:
        cm[yt][yp] += 1
    cols = [0, 1, 2, EFF_NONE]
    out.append("  confusion (rows=GT, cols=pred): " + " ".join("%8s" % EFF_NAME[c] for c in cols))
    for rc in eff_classes:
        out.append("    {:6s} ".format(EFF_NAME[rc]) + " ".join("%8d" % cm[rc][c] for c in cols))
    return {"macro_f1_8": m8, "macro_f1_eff3": m3, "n": len(pairs8),
            "per_eff": {EFF_NAME[c]: per3[c] for c in eff_classes}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", default="data/gt/HateMM/target_pred_qwen7b.jsonl")
    ap.add_argument("--target_map", default="data/gt/HateMM/target_map.json")
    ap.add_argument("--gt_dir", default="data/gt/HateMM")
    ap.add_argument("--splits", default="train,val,test")
    ap.add_argument("--summary_out", default="")
    args = ap.parse_args()

    splits = [s.strip() for s in args.splits.split(",") if s.strip()]
    preds = read_preds(args.pred)
    gt = read_gt_primary(args.target_map)
    labels = read_labels(args.gt_dir, splits)

    out = []
    out.append("TARC G2 scoring  pred={}  n_pred={}".format(args.pred, len(preds)))

    # parse / video stats over ALL predicted ids
    n_parse_fail = sum(1 for v in preds.values() if not v["parse_ok"])
    n_novideo = sum(1 for v in preds.values() if not v["video_ok"])
    out.append("  parse_ok={} parse_fail={} ({:.3f})  novideo={}".format(
        len(preds) - n_parse_fail, n_parse_fail, n_parse_fail / max(1, len(preds)), n_novideo))

    # predicted-label distribution
    pdist = Counter(CODE_NAME[v["primary"]] for v in preds.values())
    out.append("  predicted distribution: " + ", ".join(
        "{}={}".format(k, pdist[k]) for k in sorted(pdist, key=lambda x: -pdist[x])))
    gdist = Counter(CODE_NAME[gt[i]] for i in preds if i in gt)
    out.append("  GT primary distribution (over predicted ids): " + ", ".join(
        "{}={}".format(k, gdist[k]) for k in sorted(gdist, key=lambda x: -gdist[x])))

    # build (yt, yp) pairs over videos with a GT target (primary >= 0)
    all_pairs = []
    hate_pairs = []
    n_no_gt = 0
    for i, pv in preds.items():
        if i not in gt:
            n_no_gt += 1
            continue
        yt = gt[i]
        if yt < 0:
            continue  # no GT target community; excluded from macro-F1
        yp = pv["primary"]
        all_pairs.append((yt, yp))
        if labels.get(i) == 1:
            hate_pairs.append((yt, yp))
    out.append("  scored (GT target>=0): all={} hate-only={}  (GT primary=-1 excluded, ids-not-in-map={})".format(
        len(all_pairs), len(hate_pairs), n_no_gt))

    s_all = report_population("all split videos w/ GT target", all_pairs, out)
    s_hate = report_population("hate videos only w/ GT target", hate_pairs, out)

    gate_all = s_all["macro_f1_eff3"] >= 0.60
    gate_hate = s_hate["macro_f1_eff3"] >= 0.60
    out.append("\n=== GATE (effective 3-class macro-F1 >= 0.60) ===")
    out.append("  all-videos population : {:.4f}  -> {}".format(
        s_all["macro_f1_eff3"], "PASS" if gate_all else "FAIL"))
    out.append("  hate-only population  : {:.4f}  -> {}".format(
        s_hate["macro_f1_eff3"], "PASS" if gate_hate else "FAIL"))

    print("\n".join(out))
    if args.summary_out:
        json.dump({"all": s_all, "hate": s_hate,
                   "parse_fail": n_parse_fail, "novideo": n_novideo,
                   "n_pred": len(preds), "gate_all": gate_all, "gate_hate": gate_hate,
                   "pred_dist": dict(pdist)},
                  open(args.summary_out, "w"), ensure_ascii=False, indent=2)
        print("\n[summary] -> {}".format(args.summary_out))


if __name__ == "__main__":
    main()
