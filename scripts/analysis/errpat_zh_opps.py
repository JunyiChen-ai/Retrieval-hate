#!/usr/bin/env python
"""ERRPAT MHC-ZH: opportunity sizing.
  (i)   pre-head vs deployed head-space top-20 purity on the error core
        -> is the trained head DESTROYING recoverable neighbourhood structure?
  (ii)  Whisper-ASR vs deployed gt-text redundancy
        -> is 'ASR as an alternative ZH text channel' a real virgin channel or a duplicate?
  (iii) permutation test on the mid-length (Q2) error enrichment
  (iv)  ceiling arithmetic per candidate fix
"""
import json
import pickle
import re
from collections import Counter
from pathlib import Path

import numpy as np
import torch

ROOT = Path("/data/jehc223/RGCL")
TAX = json.load(open(ROOT / "scripts/analysis/errpat_zh_taxonomy_OUT.json"))
CLU = json.load(open(ROOT / "scripts/analysis/errpat_zh_clusters_OUT.json"))
OUT = ROOT / "scripts/analysis/errpat_zh_opps_OUT.json"
PER = TAX["per_item"]
SEEDS, TOPK = (0, 1, 2), 20
W = np.arange(1, TOPK + 1)[::-1].astype(np.float64)
EM_RE = re.compile(r'<em class="keyword">(.*?)</em>')
rng = np.random.default_rng(20260726)


def flat(v):
    while len(v) == 1 and isinstance(v[0], list):
        v = v[0]
    return list(v)


def l2n(x):
    x = np.asarray(x, dtype=np.float64)
    return x / np.linalg.norm(x, axis=1, keepdims=True)


def main():
    res = {}
    with open(ROOT / "scripts/analysis/errpat_remint_dumps/errpat_zh_remint_seed0.pkl", "rb") as f:
        d0 = pickle.load(f)
    rec = {(r["split"], r["epoch"]): r for r in d0["records"]}[("test", 29)]
    ids, gold = rec["ids"], rec["gold"]
    errc = np.array([PER[v]["n_seeds_wrong"] for v in ids])
    core = [i for i in range(len(ids)) if errc[i] == 3]

    ftr = torch.load(ROOT / "data/CLIP_Embedding/MHC_zh/train_Qwen2.5-VL-7B-Instruct-LoRA_HF.pt",
                     map_location="cpu", weights_only=False)
    fte = torch.load(ROOT / "data/CLIP_Embedding/MHC_zh/test_seen_Qwen2.5-VL-7B-Instruct-LoRA_HF.pt",
                     map_location="cpu", weights_only=False)
    tr_lab = np.asarray(ftr["labels"]).astype(int)
    perm = [flat(fte["ids"]).index(v) for v in ids]

    # ---------------- (i) pre-head purity in three raw key spaces
    spaces = {
        "raw_fused_l2n_concat": (np.hstack([l2n(ftr["img_feats"]), l2n(ftr["text_feats"])]),
                                 np.hstack([l2n(fte["img_feats"]), l2n(fte["text_feats"])])[perm]),
        "raw_text_only": (np.asarray(ftr["text_feats"], np.float64),
                          np.asarray(fte["text_feats"], np.float64)[perm]),
        "raw_image_only": (np.asarray(ftr["img_feats"], np.float64),
                           np.asarray(fte["img_feats"], np.float64)[perm]),
    }
    pur = {}
    for name, (trk, tek) in spaces.items():
        S = l2n(tek) @ l2n(trk).T
        top = np.argsort(-S, axis=1)[:, :TOPK]
        labs = tr_lab[top]
        p = (labs == gold[:, None]).mean(1)
        pur[name] = {
            "core_median_purity": round(float(np.median(p[core])), 4),
            "core_mean_purity": round(float(np.mean(p[core])), 4),
            "core_n_purity_gt_0.5": int(np.sum(p[core] > 0.5)),
            "correct_median_purity": round(float(np.median(p[errc == 0])), 4),
        }
    head_core = [float(np.mean([PER[ids[i]][f"seed{s}"]["top20_purity_vs_gold"] for s in SEEDS]))
                 for i in core]
    head_ok = [float(np.mean([PER[ids[i]][f"seed{s}"]["top20_purity_vs_gold"] for s in SEEDS]))
               for i in range(len(ids)) if errc[i] == 0]
    res["i_purity_prehead_vs_head"] = {
        "deployed_head_space": {
            "core_median_purity": round(float(np.median(head_core)), 4),
            "core_mean_purity": round(float(np.mean(head_core)), 4),
            "core_n_purity_gt_0.5": int(np.sum(np.array(head_core) > 0.5)),
            "correct_median_purity": round(float(np.median(head_ok)), 4),
        },
        "pre_head_raw": pur,
        "READING": "if pre-head core purity is already <=0.5 the neighbourhood inversion is "
                   "INHERITED from the encoder, not created by the trained head.",
    }

    # ---------------- (ii) ASR vs deployed text redundancy
    gt = {r["id"]: r for r in (json.loads(l) for l in open(ROOT / "data/gt/MHC_zh/test.jsonl"))}
    asr = {r["id"]: r for r in (json.loads(l) for l in
                                open(ROOT / "data/ASR/MHC_zh/test_seen_asrK4_whisper-large-v3.jsonl"))}
    PUNCT = set("，。？！,.?!、；：“”‘’…·—《》()（）【】 \n\t🎼😊")

    def norm(t):
        t = EM_RE.sub(r"\1", t)
        return "".join(c for c in t if c not in PUNCT)

    def bigrams(t):
        return Counter(t[i:i + 2] for i in range(len(t) - 1))

    cov_rows = []
    for v in ids:
        dep = norm(gt[v]["text"])
        a = norm("".join(c[2] for c in asr[v]["chunks"]))
        bd, ba = bigrams(dep), bigrams(a)
        inter = sum((bd & ba).values())
        cov_rows.append({
            "id": v, "dep_chars": len(dep), "asr_chars": len(a),
            "asr_bigrams_covered_by_dep": round(inter / max(sum(ba.values()), 1), 4),
            "dep_bigrams_covered_by_asr": round(inter / max(sum(bd.values()), 1), 4),
            "is_core_error": v in {ids[i] for i in core},
        })
    C = np.array([r["asr_bigrams_covered_by_dep"] for r in cov_rows])
    Ccore = np.array([r["asr_bigrams_covered_by_dep"] for r in cov_rows if r["is_core_error"]])
    res["ii_asr_redundancy"] = {
        "NOTE": "fraction of Whisper-ASR character bigrams already present in the DEPLOYED "
                "gt text (Title + ' . ' + MultiHateClip Transcript).",
        "all_test": {"n": len(C), "median": round(float(np.median(C)), 4),
                     "mean": round(float(np.mean(C)), 4),
                     "n_ge_0.80": int(np.sum(C >= 0.80)),
                     "n_ge_0.50": int(np.sum(C >= 0.50)),
                     "n_lt_0.25": int(np.sum(C < 0.25))},
        "core_errors": {"n": len(Ccore), "median": round(float(np.median(Ccore)), 4),
                        "n_ge_0.80": int(np.sum(Ccore >= 0.80)),
                        "n_lt_0.25": int(np.sum(Ccore < 0.25))},
        "asr_adds_new_content_ids": sorted(
            [r["id"] for r in cov_rows if r["asr_bigrams_covered_by_dep"] < 0.25
             and r["asr_chars"] >= 20]),
        "n_asr_empty": int(sum(1 for r in cov_rows if r["asr_chars"] == 0)),
    }

    # ---------------- (iii) permutation test on mid-length enrichment
    L = np.array([PER[v]["cov"]["gt_text_chars"] for v in ids])
    q = np.percentile(L, [25, 50, 75])
    band = (L >= q[0]) & (L < q[1])
    obs = int(np.sum(errc[band] == 3))
    n_core = len(core)
    null = []
    for _ in range(20000):
        pick = rng.choice(len(ids), size=n_core, replace=False)
        null.append(int(np.sum(band[pick])))
    null = np.array(null)
    res["iii_midlength_enrichment"] = {
        "band_chars": [round(float(q[0])), round(float(q[1]))],
        "n_in_band": int(band.sum()),
        "observed_core_errors_in_band": obs,
        "expected_under_null": round(float(np.mean(null)), 2),
        "null_p95": int(np.percentile(null, 95)),
        "p_value_one_sided": round(float(np.mean(null >= obs)), 4),
        "n_perms": 20000,
    }

    # ---------------- (iv) ceiling arithmetic
    xt = TAX["summary"]["D_stream_crosstab_on_3of3_errors"]
    n_test = len(ids)
    flips = TAX["summary"]["C_protocol_flip_pool"]
    res["iv_ceilings"] = {
        "n_test": n_test,
        "acc_per_item": round(1.0 / n_test, 5),
        "current_final_epoch_3seed_acc": 0.8456,
        "current_valsel_3seed_acc": 0.8322,
        "protocol_retirement_gain_measured": 0.0134,
        "core_error_pool_22_items_acc_value": round(22.0 / n_test, 4),
        "stream_crosstab_3of3": xt,
        "no_channel_knows_subset": {
            "n": xt.get("img=N,txt=N", 0),
            "acc_if_all_fixed": round(xt.get("img=N,txt=N", 0) / n_test, 4),
            "interpretation": "irreducible under ANY reweighting of the two banked streams",
        },
        "one_channel_knows_subset": {
            "n": xt.get("img=N,txt=Y", 0) + xt.get("img=Y,txt=N", 0),
            "acc_if_all_fixed": round((xt.get("img=N,txt=Y", 0) + xt.get("img=Y,txt=N", 0)) / n_test, 4),
            "interpretation": "per-item channel selection = F47-closed at all 3 supervision "
                              "sources; F66 arithmetic applies",
        },
        "both_channels_know_subset": {
            "n": xt.get("img=Y,txt=Y", 0),
            "acc_if_all_fixed": round(xt.get("img=Y,txt=Y", 0) / n_test, 4),
            "interpretation": "head/fusion loses information both raw streams have; "
                              "fusion-operator axis closed F85, fixed composition F50",
        },
        "global_threshold_oracle_gain_mean": round(float(np.mean(
            [TAX["summary"]["B_threshold_oracle"][f"seed{s}"]["gain_from_global_recalibration"]
             for s in SEEDS])), 4),
        "protocol_flip_pool_size": flips["n_distinct_items_ever_flipping"],
        "protocol_flip_pool_pct": flips["pct_of_test"],
    }

    with open(OUT, "w") as f:
        json.dump(res, f, indent=1, ensure_ascii=False)
    print(json.dumps(res, indent=1, ensure_ascii=False))
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
