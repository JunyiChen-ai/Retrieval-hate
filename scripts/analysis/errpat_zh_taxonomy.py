#!/usr/bin/env python
"""ERRPAT MHC-ZH: error taxonomy over the CPU re-minted ZH-LoRA floor (proxy for job 13150).

Sections
  A  final-epoch error inventory per seed + consensus (3/3, 2/3, 1/3)
  B  kNN vote structure per error (margin, top-20 purity, collateral cost = F66 framing)
  C  val-selected <-> final-epoch protocol flips (banked val-sel epochs 20/26/19)
  D  stream forensics in the PRE-HEAD raw banked feature space (image-only/text-only/fused)
  E  content covariates: gt-text length, <em> keyword markup, Whisper-ASR length, duration,
     MultiHateClip 3-class label (Offensive vs Hateful)
  F  named clusters
"""
import json
import pickle
import re
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch

ROOT = Path("/data/jehc223/RGCL")
DUMPS = ROOT / "scripts/analysis/errpat_remint_dumps"
OUT = ROOT / "scripts/analysis/errpat_zh_taxonomy_OUT.json"
CURVES = json.load(open(ROOT / "scripts/analysis/errpat_zh_curves_OUT.json"))

SEEDS = (0, 1, 2)
FINAL_EP = 29
N_TEST, N_TRAIN, N_DEV = 149, 579, 78
TOPK = 20
W = np.arange(1, TOPK + 1)[::-1].astype(np.float64)   # [20..1], metrics.py:229-230

# banked (bit-exact) val-selected epochs from the 13150 trainlogs
BANKED_VS = {s: CURVES["seeds"][f"seed{s}"]["val_sel_epoch"] for s in SEEDS}


# ----------------------------------------------------------------- loaders
def load_dump(seed):
    with open(DUMPS / f"errpat_zh_remint_seed{seed}.pkl", "rb") as f:
        d = pickle.load(f)
    return {(r["split"], r["epoch"]): r for r in d["records"]}


def load_gt(split):
    f = {"train": "train", "test": "test", "dev": "val"}[split]
    return {r["id"]: r for r in
            (json.loads(l) for l in open(ROOT / f"data/gt/MHC_zh/{f}.jsonl"))}


def load_3class():
    rows = json.load(open(ROOT / "data/_src_Multihateclip/Chinese/annotation(new).json"))
    return {r["Video_ID"]: r for r in rows}


def load_asr(split):
    f = {"train": "train_asrK4", "test": "test_seen_asrK4", "dev": "dev_seen_asrK4"}[split]
    return {r["id"]: r for r in
            (json.loads(l) for l in open(ROOT / f"data/ASR/MHC_zh/{f}_whisper-large-v3.jsonl"))}


def load_feats(split):
    p = ROOT / f"data/CLIP_Embedding/MHC_zh/{split}_Qwen2.5-VL-7B-Instruct-LoRA_HF.pt"
    d = torch.load(p, map_location="cpu", weights_only=False)
    return d


EM_RE = re.compile(r'<em class="keyword">(.*?)</em>')


def strip_em(t):
    return EM_RE.sub(r"\1", t)


# ----------------------------------------------------------------- vote helpers
def vote_of(nb_lab, nb_sim, n):
    m = (nb_lab[:n] * 2 - 1) * nb_sim[:n]
    return float(np.sum(m * W[:n]) / np.sum(W[:n]))


def collateral(votes, gold, i):
    """F66 framing, per item: to flip item i by a GLOBAL threshold move, how many
    currently-correct items get broken? A global rule is the only law-III-legal operator."""
    v = votes[i]
    if gold[i] == 1:                      # FN: threshold must drop to <= v
        broken = int(np.sum((gold == 0) & (votes >= v) & (votes < 0)))
        also_fixed = int(np.sum((gold == 1) & (votes >= v) & (votes < 0))) - 1
    else:                                 # FP: threshold must rise above v
        broken = int(np.sum((gold == 1) & (votes <= v) & (votes >= 0)))
        also_fixed = int(np.sum((gold == 0) & (votes <= v) & (votes >= 0))) - 1
    return broken, also_fixed


def raw_knn(train_key, train_lab, q_key, topk=TOPK):
    """Rank-weighted signed-cosine top-k vote in a raw (pre-head) key space."""
    tk = train_key / np.linalg.norm(train_key, axis=1, keepdims=True)
    qk = q_key / np.linalg.norm(q_key, axis=1, keepdims=True)
    S = qk @ tk.T
    idx = np.argsort(-S, axis=1)[:, :topk]
    sims = np.take_along_axis(S, idx, axis=1)
    labs = train_lab[idx]
    m = (labs * 2 - 1) * sims
    votes = (m * W).sum(1) / W.sum()
    return votes, idx, sims, labs


def main():
    res = {}
    gt_test, gt_train = load_gt("test"), load_gt("train")
    ann = load_3class()
    asr_test = load_asr("test")

    dumps = {s: load_dump(s) for s in SEEDS}
    # id order is identical across seeds/epochs (dataloader order, shuffle=False for eval)
    ids = dumps[0][("test", FINAL_EP)]["ids"]
    gold = dumps[0][("test", FINAL_EP)]["gold"]
    for s in SEEDS:
        for ep in (FINAL_EP, BANKED_VS[s]):
            assert dumps[s][("test", ep)]["ids"] == ids
            assert np.array_equal(dumps[s][("test", ep)]["gold"], gold)
    pos = {i: ids[i] for i in range(len(ids))}
    idx_of = {v: k for k, v in pos.items()}

    # ---------------------------------------------------------------- A: inventory
    preds, votes = {}, {}
    for s in SEEDS:
        r = dumps[s][("test", FINAL_EP)]
        preds[s] = r["pred"]
        votes[s] = r["vote"]
        # re-derive the vote from the dumped neighbour lists (self-consistency)
        rv = np.array([vote_of(r["nb_lab"][i], r["nb_sim"][i], r["n_retrieved"][i])
                       for i in range(len(ids))])
        assert np.max(np.abs(rv - r["vote"])) < 1e-9

    wrong = {s: (preds[s] != gold) for s in SEEDS}
    nwrong = {s: int(wrong[s].sum()) for s in SEEDS}
    err_count = np.zeros(len(ids), dtype=int)
    for s in SEEDS:
        err_count += wrong[s].astype(int)

    res["A_inventory"] = {
        "n_test": len(ids), "n_pos": int(gold.sum()), "n_neg": int((gold == 0).sum()),
        "per_seed": {f"seed{s}": {
            "acc": round(float(np.mean(preds[s] == gold)), 4),
            "n_errors": nwrong[s],
            "n_FP": int(((preds[s] == 1) & (gold == 0)).sum()),
            "n_FN": int(((preds[s] == 0) & (gold == 1)).sum()),
        } for s in SEEDS},
        "consensus": {
            "wrong_3of3": int((err_count == 3).sum()),
            "wrong_2of3": int((err_count == 2).sum()),
            "wrong_1of3": int((err_count == 1).sum()),
            "wrong_0of3": int((err_count == 0).sum()),
            "union_any_seed": int((err_count > 0).sum()),
        },
        "consensus_by_direction": {
            "FN_3of3": int(((err_count == 3) & (gold == 1)).sum()),
            "FP_3of3": int(((err_count == 3) & (gold == 0)).sum()),
            "FN_any": int(((err_count > 0) & (gold == 1)).sum()),
            "FP_any": int(((err_count > 0) & (gold == 0)).sum()),
        },
    }

    # ---------------------------------------------------------------- B: vote structure
    per_item = {}
    for i, vid in enumerate(ids):
        rec = {"id": vid, "gold": int(gold[i]), "n_seeds_wrong": int(err_count[i])}
        for s in SEEDS:
            r = dumps[s][("test", FINAL_EP)]
            n = int(r["n_retrieved"][i])
            nl = r["nb_lab"][i][:n]
            ns = r["nb_sim"][i][:n]
            purity = float(np.mean(nl == gold[i]))
            br, af = collateral(votes[s], gold, i)
            rec[f"seed{s}"] = {
                "vote": round(float(votes[s][i]), 6),
                "pred": int(preds[s][i]),
                "wrong": bool(wrong[s][i]),
                "abs_margin": round(abs(float(votes[s][i])), 6),
                "top20_purity_vs_gold": round(purity, 4),
                "top20_n_hate": int(nl.sum()),
                "top1_label": int(nl[0]), "top1_sim": round(float(ns[0]), 4),
                "collateral_broken": br, "collateral_also_fixed": af,
            }
        per_item[vid] = rec

    # margin distribution of errors vs correct
    def margins(sel):
        vals = [abs(float(votes[s][i])) for s in SEEDS for i in range(len(ids)) if sel(s, i)]
        return {"n": len(vals), "median": round(float(np.median(vals)), 5),
                "mean": round(float(np.mean(vals)), 5),
                "p90": round(float(np.percentile(vals, 90)), 5)} if vals else None

    allmarg = np.concatenate([np.abs(votes[s]) for s in SEEDS])
    res["B_vote_structure"] = {
        "abs_margin_errors": margins(lambda s, i: wrong[s][i]),
        "abs_margin_correct": margins(lambda s, i: not wrong[s][i]),
        "abs_margin_all_p25": round(float(np.percentile(allmarg, 25)), 5),
        "abs_margin_all_median": round(float(np.median(allmarg)), 5),
        "purity_errors_median": round(float(np.median(
            [per_item[ids[i]][f"seed{s}"]["top20_purity_vs_gold"]
             for s in SEEDS for i in range(len(ids)) if wrong[s][i]])), 4),
        "purity_correct_median": round(float(np.median(
            [per_item[ids[i]][f"seed{s}"]["top20_purity_vs_gold"]
             for s in SEEDS for i in range(len(ids)) if not wrong[s][i]])), 4),
    }
    # threshold-reachability: global-threshold oracle per seed
    tro = {}
    for s in SEEDS:
        v = votes[s]
        cand = np.unique(np.concatenate([v, [v.min() - 1, v.max() + 1]]))
        best = max(cand, key=lambda t: np.mean((v >= t).astype(int) == gold))
        acc_t = float(np.mean((v >= best).astype(int) == gold))
        n_cheap = sum(1 for i in range(len(ids)) if wrong[s][i]
                      and per_item[ids[i]][f"seed{s}"]["collateral_broken"] <= 1)
        tro[f"seed{s}"] = {
            "deployed_threshold_acc": round(float(np.mean((v >= 0).astype(int) == gold)), 4),
            "best_global_threshold": round(float(best), 6),
            "best_global_threshold_acc": round(acc_t, 4),
            "gain_from_global_recalibration": round(acc_t - float(np.mean((v >= 0).astype(int) == gold)), 4),
            "n_errors_with_collateral_le_1": n_cheap,
            "n_errors": nwrong[s],
        }
    res["B_threshold_oracle"] = tro

    # ---------------------------------------------------------------- C: protocol flips
    prot = {}
    flip_counter = Counter()
    flip_dir = defaultdict(Counter)
    for s in SEEDS:
        vs = BANKED_VS[s]
        pv = dumps[s][("test", vs)]["pred"]
        pf = dumps[s][("test", FINAL_EP)]["pred"]
        diff = np.where(pv != pf)[0]
        fixed = [i for i in diff if pf[i] == gold[i]]          # final right, val-sel wrong
        broken = [i for i in diff if pv[i] == gold[i]]         # val-sel right, final wrong
        for i in diff:
            flip_counter[ids[i]] += 1
            flip_dir[ids[i]][f"seed{s}"] = f"valsel={int(pv[i])}->final={int(pf[i])} gold={int(gold[i])}"
        prot[f"seed{s}"] = {
            "banked_valsel_epoch": vs, "final_epoch": FINAL_EP,
            "remint_valsel_acc": round(float(np.mean(pv == gold)), 4),
            "remint_final_acc": round(float(np.mean(pf == gold)), 4),
            "banked_valsel_acc": CURVES["seeds"][f"seed{s}"]["val_sel_test_acc"],
            "banked_final_acc": CURVES["seeds"][f"seed{s}"]["final_test_acc"],
            "n_items_changing_prediction": int(len(diff)),
            "n_fixed_by_final": len(fixed), "n_broken_by_final": len(broken),
            "net_items": len(fixed) - len(broken),
            "fixed_ids": [ids[i] for i in fixed], "broken_ids": [ids[i] for i in broken],
        }
    res["C_protocol_flips"] = prot
    res["C_protocol_flip_pool"] = {
        "n_distinct_items_ever_flipping": len(flip_counter),
        "pct_of_test": round(100.0 * len(flip_counter) / len(ids), 1),
        "flips_by_item": dict(flip_counter.most_common()),
    }
    # protocol-sensitivity of the WHOLE legal epoch window
    sens = {}
    for s in SEEDS:
        P = np.stack([dumps[s][("test", e)]["pred"] for e in range(5, 30)])
        n_ever = int(np.sum(P.min(0) != P.max(0)))
        stable_right = int(np.sum((P == gold).all(0)))
        stable_wrong = int(np.sum((P != gold).all(0)))
        sens[f"seed{s}"] = {
            "n_items_not_constant_over_legal_epochs": n_ever,
            "pct": round(100.0 * n_ever / len(ids), 1),
            "n_always_correct": stable_right, "n_always_wrong": stable_wrong,
        }
    res["C_epoch_sensitivity"] = sens

    # ---------------------------------------------------------------- D: stream forensics
    ftr, fte = load_feats("train"), load_feats("test_seen")

    def flat_ids(d):
        v = d["ids"]
        while len(v) == 1 and isinstance(v[0], list):
            v = v[0]
        return list(v)

    tr_ids, te_ids = flat_ids(ftr), flat_ids(fte)
    assert len(tr_ids) == N_TRAIN and len(te_ids) == N_TEST, (len(tr_ids), len(te_ids))
    tr_lab = np.asarray(ftr["labels"]).astype(int)
    te_lab = np.asarray(fte["labels"]).astype(int)
    tr_img = np.asarray(ftr["img_feats"], dtype=np.float64)
    tr_txt = np.asarray(ftr["text_feats"], dtype=np.float64)
    te_img = np.asarray(fte["img_feats"], dtype=np.float64)
    te_txt = np.asarray(fte["text_feats"], dtype=np.float64)
    # align to the dump order
    perm = [te_ids.index(v) for v in ids]
    te_img, te_txt, te_lab_a = te_img[perm], te_txt[perm], te_lab[perm]
    assert np.array_equal(te_lab_a, gold), "raw-cache labels disagree with dump gold"

    def l2n(x):
        return x / np.linalg.norm(x, axis=1, keepdims=True)

    streams = {}
    stream_pred = {}
    for name, trk, tek in (("image_only", tr_img, te_img),
                           ("text_only", tr_txt, te_txt),
                           ("fused_concat_l2n", np.hstack([l2n(tr_img), l2n(tr_txt)]),
                            np.hstack([l2n(te_img), l2n(te_txt)]))):
        v, _, _, _ = raw_knn(trk, tr_lab, tek)
        p = (v >= 0).astype(int)
        stream_pred[name] = p
        from sklearn.metrics import roc_auc_score
        streams[name] = {
            "acc": round(float(np.mean(p == gold)), 4),
            "auc": round(float(roc_auc_score(gold, v)), 4),
            "n_FP": int(((p == 1) & (gold == 0)).sum()),
            "n_FN": int(((p == 0) & (gold == 1)).sum()),
        }
    res["D_stream_raw_space"] = {
        "NOTE": "PRE-HEAD raw banked LoRA features; NOT the deployed head space. "
                "Deployed fusion is a trained Hadamard align, so no head-space "
                "single-stream vote exists.",
        "streams": streams,
    }
    # cross-tab: which streams get each deployed-error right?
    xt = Counter()
    for s in SEEDS:
        for i in range(len(ids)):
            if not wrong[s][i]:
                continue
            key = (f"img={'Y' if stream_pred['image_only'][i] == gold[i] else 'N'}",
                   f"txt={'Y' if stream_pred['text_only'][i] == gold[i] else 'N'}")
            xt[key] += 1
    res["D_stream_crosstab_on_deployed_errors"] = {
        f"{a},{b}": n for (a, b), n in sorted(xt.items(), key=lambda kv: -kv[1])}
    # same cross-tab on the 3/3 consensus errors only
    xt3 = Counter()
    for i in range(len(ids)):
        if err_count[i] != 3:
            continue
        xt3[(f"img={'Y' if stream_pred['image_only'][i] == gold[i] else 'N'}",
             f"txt={'Y' if stream_pred['text_only'][i] == gold[i] else 'N'}")] += 1
    res["D_stream_crosstab_on_3of3_errors"] = {
        f"{a},{b}": n for (a, b), n in sorted(xt3.items(), key=lambda kv: -kv[1])}

    # ---------------------------------------------------------------- E: covariates
    cov = {}
    for i, vid in enumerate(ids):
        g = gt_test[vid]
        a = ann.get(vid, {})
        s3 = a.get("Label", "MISSING")
        asr = asr_test.get(vid, {})
        asr_txt = "".join(c[2] for c in asr.get("chunks", []) or [])
        raw = g["text"]
        cov[vid] = {
            "gold": int(gold[i]),
            "label3": s3,
            "gt_text_chars": len(strip_em(raw)),
            "has_em_markup": bool(EM_RE.search(raw)),
            "em_terms": EM_RE.findall(raw),
            "asr_chars": len(asr_txt),
            "asr_empty": len(asr_txt) == 0,
            "duration_s": round(float(asr.get("duration", float("nan"))), 2)
            if asr.get("duration") is not None else None,
            "asr_language": asr.get("language"),
            "n_seeds_wrong": int(err_count[i]),
        }
        per_item[vid]["cov"] = cov[vid]

    def grp(pred_fn, label):
        sel = [v for v in ids if pred_fn(cov[v])]
        if not sel:
            return None
        ew = [cov[v]["n_seeds_wrong"] for v in sel]
        return {"label": label, "n": len(sel),
                "err_rate_per_seed": round(float(np.sum(ew) / (3 * len(sel))), 4),
                "n_wrong_3of3": sum(1 for v in sel if cov[v]["n_seeds_wrong"] == 3),
                "n_wrong_any": sum(1 for v in sel if cov[v]["n_seeds_wrong"] > 0)}

    L = [cov[v]["gt_text_chars"] for v in ids]
    q1, q2, q3 = np.percentile(L, [25, 50, 75])
    res["E_covariates"] = {
        "label3_distribution": dict(Counter(cov[v]["label3"] for v in ids)),
        "by_label3": [grp(lambda c, x=x: c["label3"] == x, f"3class={x}")
                      for x in ("Normal", "Offensive", "Hateful")],
        "by_em_markup": [grp(lambda c: c["has_em_markup"], "has <em> keyword markup"),
                         grp(lambda c: not c["has_em_markup"], "no <em> markup")],
        "gt_text_chars_quartiles": [round(float(x), 1) for x in (q1, q2, q3)],
        "by_text_len": [
            grp(lambda c, a=a, b=b: a <= c["gt_text_chars"] < b, f"gt_text_chars [{a:.0f},{b:.0f})")
            for a, b in ((0, q1), (q1, q2), (q2, q3), (q3, 1e9))],
        "by_short_text": [grp(lambda c: c["gt_text_chars"] <= 40, "gt_text <= 40 chars"),
                          grp(lambda c: c["gt_text_chars"] > 40, "gt_text > 40 chars")],
        "by_asr_empty": [grp(lambda c: c["asr_empty"], "Whisper ASR empty"),
                         grp(lambda c: not c["asr_empty"], "Whisper ASR non-empty")],
        "asr_empty_count": sum(1 for v in ids if cov[v]["asr_empty"]),
        "asr_chars_median": float(np.median([cov[v]["asr_chars"] for v in ids])),
        "gt_text_chars_median": float(np.median(L)),
        "duration_median_s": float(np.nanmedian([cov[v]["duration_s"] for v in ids
                                                 if cov[v]["duration_s"] is not None])),
        "by_duration": None,
        "asr_language_dist": dict(Counter(cov[v]["asr_language"] for v in ids)),
    }
    D = [cov[v]["duration_s"] for v in ids if cov[v]["duration_s"] is not None]
    d1, d2, d3 = np.percentile(D, [25, 50, 75])
    res["E_covariates"]["by_duration"] = [
        grp(lambda c, a=a, b=b: c["duration_s"] is not None and a <= c["duration_s"] < b,
            f"duration [{a:.0f},{b:.0f})s") for a, b in ((0, d1), (d1, d2), (d2, d3), (d3, 1e9))]

    # positives only: Offensive vs Hateful error rates (F44/F82 within-positive wall)
    res["E_within_positive"] = {
        x: grp(lambda c, x=x: c["label3"] == x and c["gold"] == 1, f"positive & {x}")
        for x in ("Offensive", "Hateful")}

    with open(OUT, "w") as f:
        json.dump({"summary": res, "per_item": per_item}, f, indent=1, ensure_ascii=False)
    print(json.dumps(res, indent=1, ensure_ascii=False))
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
