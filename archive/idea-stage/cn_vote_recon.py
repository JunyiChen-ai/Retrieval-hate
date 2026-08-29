"""CN_VOTE_RECON -- can MultiHateClip's discarded `Counter Narrative` annotator votes
serve as stance supervision?

Zero API cost, CPU only.  Reads only frozen artefacts:
  data/gt/mhc_votes/mhc_{English,Chinese}_{train,valid,test}.tsv   per-annotator votes
  data/gt/{MHC,MHC_zh}/{train,val,test}.jsonl                      project binary labels
  idea-stage/r5_buckets.json                                       manual error buckets (test)
  idea-stage/r4_pilot1.json                                        round-4 ensemble test scores
  idea-stage/voice_field_analysis.py                               GOLD_VOICE hand codes
  data/CLIP_Embedding/*                                            frozen features (step 4 only)

Steps 1-3 are read-only census / contingency / agreement.  Step 4 (discriminator) runs
only if the frozen decision bar in CN_VOTE_RECON.md is cleared, and touches TRAIN+VAL only.
"""
import ast
import csv
import json
import os
import sys
from collections import Counter

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LANGS = {"MHC": "English", "MHC_zh": "Chinese"}
SPLITS = {"train": "train", "val": "valid", "test": "test"}
OUT = {}


# --------------------------------------------------------------- step 1: census
def load_votes(lang, split):
    p = os.path.join(ROOT, "data", "gt", "mhc_votes", f"mhc_{lang}_{split}.tsv")
    rows = list(csv.DictReader(open(p, newline="", encoding="utf-8"), delimiter="\t"))
    out = {}
    for r in rows:
        v = ast.literal_eval(r["Label"])
        out[r["Video_ID"].strip()] = {"votes": v, "maj": r["Majority_Voting"].strip()}
    return out


def load_bin(ds, split):
    p = os.path.join(ROOT, "data", "gt", ds, f"{split}.jsonl")
    out = {}
    for line in open(p, encoding="utf-8"):
        o = json.loads(line)
        out[o["id"].strip()] = int(o["label"])
    return out


def census():
    res = {"per_dataset": {}, "vote_vocab": {}}
    vocab = Counter()
    allrows = {}
    for ds, lang in LANGS.items():
        per = {}
        for sp, vsp in SPLITS.items():
            votes = load_votes(lang, vsp)
            binlab = load_bin(ds, sp)
            n_cn = n_maj_cn = 0
            cn_ids = []
            maj_dist = Counter()
            bin_dist = Counter()
            for vid, r in votes.items():
                vocab.update(r["votes"])
                k = sum(1 for x in r["votes"] if x == "Counter Narrative")
                allrows[(ds, sp, vid)] = {
                    "votes": r["votes"], "maj": r["maj"], "n_cn": k,
                    "n_ann": len(r["votes"]),
                    "bin": binlab.get(vid),
                }
                if k >= 1:
                    n_cn += 1
                    cn_ids.append(vid)
                    maj_dist[r["maj"]] += 1
                    bin_dist[binlab.get(vid)] += 1
                    if k * 2 > len(r["votes"]):
                        n_maj_cn += 1
            per[sp] = {"n_videos": len(votes), "n_with_cn_vote": n_cn,
                       "n_majority_cn": n_maj_cn,
                       "majority_label_of_cn_videos": dict(maj_dist),
                       "binary_label_of_cn_videos": {str(k): v for k, v in bin_dist.items()},
                       "cn_ids": cn_ids}
        res["per_dataset"][ds] = per
    res["vote_vocab"] = dict(vocab)
    res["_all_rows_n"] = len(allrows)
    # global roll-up
    tot = {"n_videos": 0, "n_with_cn_vote": 0, "n_majority_cn": 0,
           "maj": Counter(), "bin": Counter(), "n_cn_votes": 0}
    for (ds, sp, vid), r in allrows.items():
        tot["n_videos"] += 1
        tot["n_cn_votes"] += r["n_cn"]
        if r["n_cn"] >= 1:
            tot["n_with_cn_vote"] += 1
            tot["maj"][r["maj"]] += 1
            tot["bin"][str(r["bin"])] += 1
            if r["n_cn"] * 2 > r["n_ann"]:
                tot["n_majority_cn"] += 1
    res["total"] = {"n_videos": tot["n_videos"], "n_cn_votes": tot["n_cn_votes"],
                    "n_with_cn_vote": tot["n_with_cn_vote"],
                    "n_majority_cn": tot["n_majority_cn"],
                    "majority_label_of_cn_videos": dict(tot["maj"]),
                    "binary_label_of_cn_videos": dict(tot["bin"])}
    res["annotators_per_video"] = dict(Counter(r["n_ann"] for r in allrows.values()))
    return res, allrows


# ------------------------------------------------- step 2: detector x CN votes
BEST = {"HateMM": "mlp", "MHC": "mean_logit", "MHC_zh": "logistic",
        "ImpliHateVid": "logistic"}
CLIP = "openai_clip-vit-large-patch14-336_HF"


def test_predictions(ds):
    """Recover per-item majority prediction of the round-4 best ensemble on the test
    split, exactly as r5_xbucket_recon.py does (threshold inversion from stored F1)."""
    import torch
    from sklearn.metrics import f1_score
    PIL = json.load(open(os.path.join(ROOT, "idea-stage", "r4_pilot1.json")))
    d = torch.load(os.path.join(ROOT, "data", "CLIP_Embedding", ds,
                                f"test_seen_{CLIP}.pt"), map_location="cpu",
                   weights_only=False)
    ids = list(d["ids"][0]) if isinstance(d["ids"][0], list) else list(d["ids"])
    y = torch.as_tensor(d["labels"]).view(-1).numpy().astype(int)
    key = BEST[ds]
    preds = []
    assert np.array_equal(np.array(PIL["datasets"][ds]["per_seed"][0]["scores"]["y"],
                                   dtype=int), y)
    for s in PIL["datasets"][ds]["per_seed"]:
        sv = np.array(s["scores"][key], dtype=float)
        target = s["methods"][key]["test_macro_f1"]
        cands = np.unique(sv)
        grid = np.concatenate([[cands[0] - 1e-9], (cands[:-1] + cands[1:]) / 2,
                               [cands[-1] + 1e-9]])
        hits = [t for t in grid
                if abs(f1_score(y, (sv >= t).astype(int), average="macro") - target) < 1e-9]
        assert hits, f"threshold recovery failed for {ds}"
        t = float(hits[len(hits) // 2])
        preds.append((sv >= t).astype(int))
    pred = (np.mean(preds, axis=0) >= 0.5).astype(int)
    return ids, y, pred


def contingency(allrows):
    BUCK = json.load(open(os.path.join(ROOT, "idea-stage", "r5_buckets.json")))
    res = {}
    pooled = np.zeros((2, 2), dtype=int)
    for ds in LANGS:
        ids, y, pred = test_predictions(ds)
        buck = BUCK[ds]
        tab = Counter()
        s_cn, s_nocn = [], []
        for i, vid in enumerate(ids):
            r = allrows.get((ds, "test", vid))
            if r is None:
                tab[("MISSING_VOTE", "")] += 1
                continue
            cn = r["n_cn"] >= 1
            correct = bool(pred[i] == y[i])
            tab[("CN" if cn else "noCN", "correct" if correct else "error")] += 1
            b = buck.get(vid)
            if b == "S":
                (s_cn if cn else s_nocn).append(vid)
        # detector error rate among hateful(=1) videos, CN vs non-CN
        hate_cn = [(i, vid) for i, vid in enumerate(ids)
                   if y[i] == 1 and allrows.get((ds, "test", vid), {}).get("n_cn", 0) >= 1]
        hate_no = [(i, vid) for i, vid in enumerate(ids)
                   if y[i] == 1 and allrows.get((ds, "test", vid), {}).get("n_cn", 0) == 0]
        def er(lst):
            if not lst:
                return None
            return float(np.mean([pred[i] != y[i] for i, _ in lst]))
        # S bucket split into false positives (gold 0) and false negatives (gold 1)
        s_fp, s_fn = [], []
        for i, vid in enumerate(ids):
            if buck.get(vid) == "S":
                k = allrows.get((ds, "test", vid), {}).get("n_cn", 0)
                (s_fp if y[i] == 0 else s_fn).append((vid, k))
        res[ds] = {
            "n_test": len(ids),
            "S_FP": {"n": len(s_fp), "n_with_cn": sum(1 for _, k in s_fp if k),
                     "ids_with_cn": [v for v, k in s_fp if k]},
            "S_FN": {"n": len(s_fn), "n_with_cn": sum(1 for _, k in s_fn if k),
                     "ids_with_cn": [v for v, k in s_fn if k]},
            "table": {f"{a}|{b}": c for (a, b), c in tab.items()},
            "S_bucket_total": sum(1 for v in buck.values() if v == "S"),
            "S_bucket_in_test_ids": sum(1 for vid in ids if buck.get(vid) == "S"),
            "S_with_CN_vote": s_cn, "S_without_CN_vote": s_nocn,
            "S_frac_with_CN": (len(s_cn) / (len(s_cn) + len(s_nocn))
                               if (s_cn or s_nocn) else None),
            "hate_CN_n": len(hate_cn), "hate_CN_err_rate": er(hate_cn),
            "hate_nonCN_n": len(hate_no), "hate_nonCN_err_rate": er(hate_no),
        }
        pooled[0, 0] += tab[("noCN", "correct")]
        pooled[0, 1] += tab[("noCN", "error")]
        pooled[1, 0] += tab[("CN", "correct")]
        pooled[1, 1] += tab[("CN", "error")]
    from scipy.stats import fisher_exact
    orr, p = fisher_exact(pooled)
    nfp = sum(res[d]["S_FP"]["n"] for d in res)
    nfp_cn = sum(res[d]["S_FP"]["n_with_cn"] for d in res)
    ns = sum(res[d]["S_bucket_in_test_ids"] for d in res)
    ns_cn = sum(len(res[d]["S_with_CN_vote"]) for d in res)
    res["_pooled"] = {
        "table_rows_noCN_CN_cols_correct_error": pooled.tolist(),
        "fisher_OR": float(orr), "fisher_p": float(p),
        "B1_S_FP_frac_with_cn": nfp_cn / nfp if nfp else None,
        "B1_bar": 0.25, "B1_pass": bool(nfp and nfp_cn / nfp >= 0.25),
        "S_all_frac_with_cn": ns_cn / ns if ns else None,
    }
    return res


# ------------------------------------------- step 3: CN votes vs GOLD_VOICE
def gold_voice_agreement(allrows):
    sys.path.insert(0, os.path.join(ROOT, "idea-stage"))
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "vfa", os.path.join(ROOT, "idea-stage", "voice_field_analysis.py"))
    mod = importlib.util.module_from_spec(spec)
    # voice_field_analysis executes analysis on import guard? load module source only
    src = open(os.path.join(ROOT, "idea-stage", "voice_field_analysis.py"),
               encoding="utf-8").read()
    ns = {}
    start = src.index("GOLD_VOICE = {")
    end = src.index("\n}\n", start) + 3
    exec(src[start:end], ns)
    GOLD = ns["GOLD_VOICE"]
    rows = []
    for (ds, vid), (g, why, blind) in GOLD.items():
        if ds not in LANGS:
            continue
        r = None
        for sp in SPLITS:
            if (ds, sp, vid) in allrows:
                r = allrows[(ds, sp, vid)]
                break
        if r is None:
            rows.append({"ds": ds, "id": vid, "gold": g, "cn": None, "note": "no vote row"})
            continue
        rows.append({"ds": ds, "id": vid, "gold": g, "cn": r["n_cn"] >= 1,
                     "n_cn": r["n_cn"], "maj": r["maj"]})
    det = [r for r in rows if r.get("cn") is not None and r["gold"] in ("OWN", "NOT_OWN")]
    tab = Counter((r["gold"], "CN" if r["cn"] else "noCN") for r in det)
    return {"n_gold_mhc": len(rows), "n_determinate": len(det),
            "table": {f"{a}|{b}": c for (a, b), c in tab.items()},
            "rows": rows}


def main():
    OUT["step1_census"], allrows = census()
    OUT["step2_contingency"] = contingency(allrows)
    OUT["step3_gold_voice"] = gold_voice_agreement(allrows)
    p = os.path.join(ROOT, "idea-stage", "cn_vote_recon.json")
    json.dump(OUT, open(p, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(json.dumps({k: v for k, v in OUT.items() if k != "step3_gold_voice"},
                     ensure_ascii=False, indent=1)[:6000])
    print("--- step3 ---")
    print(json.dumps({k: v for k, v in OUT["step3_gold_voice"].items() if k != "rows"},
                     ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
