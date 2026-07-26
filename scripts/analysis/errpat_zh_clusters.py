#!/usr/bin/env python
"""ERRPAT MHC-ZH follow-up: cluster structure, purity, coverage-vs-ranking, convergence, exemplars."""
import json
import pickle
import re
from collections import Counter
from pathlib import Path

import numpy as np
import torch

ROOT = Path("/data/jehc223/RGCL")
DUMPS = ROOT / "scripts/analysis/errpat_remint_dumps"
TAX = json.load(open(ROOT / "scripts/analysis/errpat_zh_taxonomy_OUT.json"))
CURVES = json.load(open(ROOT / "scripts/analysis/errpat_zh_curves_OUT.json"))
OUT = ROOT / "scripts/analysis/errpat_zh_clusters_OUT.json"
SEEDS, FINAL_EP, TOPK = (0, 1, 2), 29, 20
W = np.arange(1, TOPK + 1)[::-1].astype(np.float64)
EM_RE = re.compile(r'<em class="keyword">(.*?)</em>')
PER = TAX["per_item"]


def load_dump(s):
    with open(DUMPS / f"errpat_zh_remint_seed{s}.pkl", "rb") as f:
        d = pickle.load(f)
    return {(r["split"], r["epoch"]): r for r in d["records"]}


def flat(v):
    while len(v) == 1 and isinstance(v[0], list):
        v = v[0]
    return list(v)


def main():
    res = {}
    dumps = {s: load_dump(s) for s in SEEDS}
    ids = dumps[0][("test", FINAL_EP)]["ids"]
    gold = dumps[0][("test", FINAL_EP)]["gold"]
    errc = np.array([PER[v]["n_seeds_wrong"] for v in ids])

    # ---------------- 1. convergence: re-mint vs banked agreement by epoch
    conv = {}
    for s in SEEDS:
        bk = CURVES["seeds"][f"seed{s}"]["test_acc_curve"]
        rm = [dumps[s][("test", e)]["acc"] for e in range(30)]
        conv[f"seed{s}"] = {
            f"ep{e}": {"remint": round(rm[e], 4), "banked": round(bk[e], 4),
                       "diff_items": round((rm[e] - bk[e]) * 149, 2)}
            for e in range(20, 30)}
    # mean |diff| in items, early (5-19) vs late (25-29)
    agg = {}
    for lo, hi, tag in ((5, 20, "ep5-19"), (20, 25, "ep20-24"), (25, 30, "ep25-29"), (29, 30, "ep29")):
        v = []
        for s in SEEDS:
            bk = CURVES["seeds"][f"seed{s}"]["test_acc_curve"]
            rm = [dumps[s][("test", e)]["acc"] for e in range(30)]
            v += [abs(rm[e] - bk[e]) * 149 for e in range(lo, hi)]
        agg[tag] = {"mean_abs_diff_items": round(float(np.mean(v)), 2),
                    "max_abs_diff_items": round(float(np.max(v)), 2), "n": len(v)}
    res["1_remint_vs_banked_convergence"] = {"by_window": agg, "late_epochs": conv}

    # ---------------- 2. purity / margin structure of the 3/3 error core
    core = [i for i in range(len(ids)) if errc[i] == 3]
    prof = []
    for i in core:
        r = PER[ids[i]]
        pur = float(np.mean([r[f"seed{s}"]["top20_purity_vs_gold"] for s in SEEDS]))
        mar = float(np.mean([r[f"seed{s}"]["abs_margin"] for s in SEEDS]))
        col = float(np.mean([r[f"seed{s}"]["collateral_broken"] for s in SEEDS]))
        prof.append({"id": ids[i], "gold": int(gold[i]), "label3": r["cov"]["label3"],
                     "mean_purity": round(pur, 4), "mean_abs_margin": round(mar, 4),
                     "mean_collateral": round(col, 2),
                     "gt_text_chars": r["cov"]["gt_text_chars"],
                     "has_em": r["cov"]["has_em_markup"],
                     "duration_s": r["cov"]["duration_s"],
                     "asr_chars": r["cov"]["asr_chars"]})
    res["2_core_error_profile"] = sorted(prof, key=lambda x: (x["gold"], x["mean_purity"]))
    purs = [p["mean_purity"] for p in prof]
    res["2_core_purity_bands"] = {
        "purity_0.00_0.10": sum(1 for p in purs if p <= 0.10),
        "purity_0.10_0.25": sum(1 for p in purs if 0.10 < p <= 0.25),
        "purity_0.25_0.45": sum(1 for p in purs if 0.25 < p <= 0.45),
        "purity_gt_0.45": sum(1 for p in purs if p > 0.45),
        "median": round(float(np.median(purs)), 4),
    }
    # confidently-wrong: high margin AND inverted neighbourhood
    res["2_confidently_wrong"] = {
        "n_core": len(core),
        "n_margin_ge_0.5_and_purity_le_0.25": sum(
            1 for p in prof if p["mean_abs_margin"] >= 0.5 and p["mean_purity"] <= 0.25),
        "n_margin_ge_0.9_and_purity_le_0.10": sum(
            1 for p in prof if p["mean_abs_margin"] >= 0.9 and p["mean_purity"] <= 0.10),
        "n_low_margin_le_0.3": sum(1 for p in prof if p["mean_abs_margin"] <= 0.3),
    }

    # ---------------- 3. coverage vs ranking (PRE-HEAD raw fused space, full 579 ranking)
    ftr = torch.load(ROOT / "data/CLIP_Embedding/MHC_zh/train_Qwen2.5-VL-7B-Instruct-LoRA_HF.pt",
                     map_location="cpu", weights_only=False)
    fte = torch.load(ROOT / "data/CLIP_Embedding/MHC_zh/test_seen_Qwen2.5-VL-7B-Instruct-LoRA_HF.pt",
                     map_location="cpu", weights_only=False)
    tr_lab = np.asarray(ftr["labels"]).astype(int)
    te_ids = flat(fte["ids"])
    perm = [te_ids.index(v) for v in ids]

    def l2n(x):
        x = np.asarray(x, dtype=np.float64)
        return x / np.linalg.norm(x, axis=1, keepdims=True)

    trk = np.hstack([l2n(ftr["img_feats"]), l2n(ftr["text_feats"])])
    tek = np.hstack([l2n(fte["img_feats"]), l2n(fte["text_feats"])])[perm]
    S = l2n(tek) @ l2n(trk).T
    order = np.argsort(-S, axis=1)
    rank_first_same = []
    for i in range(len(ids)):
        lab_seq = tr_lab[order[i]]
        r = int(np.argmax(lab_seq == gold[i])) + 1
        rank_first_same.append(r)
    rank_first_same = np.array(rank_first_same)
    res["3_coverage_vs_ranking_prehead"] = {
        "NOTE": "PRE-HEAD raw fused (l2n-concat) space over the full 579-row train bank; "
                "rank of the FIRST same-gold-class train neighbour.",
        "core_errors": {
            "median_rank_first_same_class": float(np.median(rank_first_same[core])),
            "n_rank_1": int(np.sum(rank_first_same[core] == 1)),
            "n_rank_le_5": int(np.sum(rank_first_same[core] <= 5)),
            "n_rank_le_20": int(np.sum(rank_first_same[core] <= 20)),
            "n_rank_gt_20": int(np.sum(rank_first_same[core] > 20)),
            "max_rank": int(np.max(rank_first_same[core])),
        },
        "correct_items": {
            "median_rank_first_same_class": float(np.median(
                rank_first_same[[i for i in range(len(ids)) if errc[i] == 0]])),
        },
    }

    # ---------------- 4. text-length x label interaction (control for class)
    L = np.array([PER[v]["cov"]["gt_text_chars"] for v in ids])
    q = np.percentile(L, [25, 50, 75])
    inter = []
    for cls, name in ((1, "positive"), (0, "negative")):
        for lo, hi, tag in ((0, q[0], "Q1"), (q[0], q[1], "Q2"), (q[1], q[2], "Q3"), (q[2], 1e9, "Q4")):
            sel = [i for i in range(len(ids)) if gold[i] == cls and lo <= L[i] < hi]
            if not sel:
                continue
            inter.append({"class": name, "len_band": tag,
                          "chars_range": [round(float(lo), 0), round(float(hi), 0) if hi < 1e8 else None],
                          "n": len(sel),
                          "err_rate_per_seed": round(float(np.sum(errc[sel]) / (3 * len(sel))), 4),
                          "n_3of3": int(np.sum(errc[sel] == 3))})
    res["4_textlen_x_class"] = inter
    res["4_class_composition_by_len_band"] = [
        {"len_band": tag, "n_pos": int(np.sum((gold == 1) & (L >= lo) & (L < hi))),
         "n_neg": int(np.sum((gold == 0) & (L >= lo) & (L < hi)))}
        for lo, hi, tag in ((0, q[0], "Q1"), (q[0], q[1], "Q2"), (q[1], q[2], "Q3"), (q[2], 1e9, "Q4"))]

    # ---------------- 5. protocol-flip pool vs error core
    flip_ids = set(TAX["summary"]["C_protocol_flip_pool"]["flips_by_item"].keys())
    core_ids = {ids[i] for i in core}
    res["5_flip_pool_vs_core"] = {
        "n_flip_pool": len(flip_ids),
        "n_core_errors": len(core_ids),
        "overlap": len(flip_ids & core_ids),
        "flip_only": sorted(flip_ids - core_ids),
        "core_and_flip": sorted(flip_ids & core_ids),
        "flip_pool_profile": [
            {"id": v, "gold": PER[v]["gold"], "label3": PER[v]["cov"]["label3"],
             "n_seeds_wrong_at_final": PER[v]["n_seeds_wrong"],
             "mean_abs_margin": round(float(np.mean(
                 [PER[v][f"seed{s}"]["abs_margin"] for s in SEEDS])), 4),
             "mean_purity": round(float(np.mean(
                 [PER[v][f"seed{s}"]["top20_purity_vs_gold"] for s in SEEDS])), 4)}
            for v in sorted(flip_ids)],
    }

    # ---------------- 6. exemplars with text snippets
    gt = {r["id"]: r for r in (json.loads(l) for l in open(ROOT / "data/gt/MHC_zh/test.jsonl"))}
    asr = {r["id"]: r for r in (json.loads(l) for l in
                                open(ROOT / "data/ASR/MHC_zh/test_seen_asrK4_whisper-large-v3.jsonl"))}

    def snip(v, n=90):
        t = EM_RE.sub(r"[\1]", gt[v]["text"]).replace("\n", " ")
        return t[:n] + ("…" if len(t) > n else "")

    ex = {}
    for p in res["2_core_error_profile"]:
        v = p["id"]
        ex[v] = {"gold": p["gold"], "label3": p["label3"],
                 "mean_purity": p["mean_purity"], "mean_abs_margin": p["mean_abs_margin"],
                 "gt_text_chars": p["gt_text_chars"], "duration_s": p["duration_s"],
                 "text_snippet": snip(v),
                 "asr_snippet": ("".join(c[2] for c in asr[v]["chunks"])[:70]
                                 if v in asr else None)}
    res["6_exemplars_core"] = ex

    with open(OUT, "w") as f:
        json.dump(res, f, indent=1, ensure_ascii=False)
    for k in ("1_remint_vs_banked_convergence", "2_core_purity_bands", "2_confidently_wrong",
              "3_coverage_vs_ranking_prehead", "4_textlen_x_class",
              "4_class_composition_by_len_band", "5_flip_pool_vs_core"):
        print("=" * 30, k)
        v = res[k]
        if k == "1_remint_vs_banked_convergence":
            v = v["by_window"]
        print(json.dumps(v, indent=1, ensure_ascii=False))
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
