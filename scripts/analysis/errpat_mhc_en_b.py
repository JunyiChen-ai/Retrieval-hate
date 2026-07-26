#!/usr/bin/env python
"""ERRPAT MHC-EN part B — solution-mapping arithmetic on top of errpat_mhc_en.py.

READ-ONLY FORENSICS. CPU only. Every "oracle" / "would-be" number below is an UPPER
BOUND computed with test labels for diagnosis; none of it is fed back into any config.

Adds:
  (1) memory-bank deletion replay: exactly re-vote all 4 deployed val-sel seeds with the
      2 human-flagged noisy train entries removed from the bank (top-60 refill makes this
      exact as long as <=40 deletions land inside a top-60, asserted). Reproduces the
      banked seed-0 result (0.8075 -> 0.8199) and names the flipped items. Also the
      14-id rule list.
  (2) decision-threshold family, both directions: dev-SELECTED (deployable, legal) and
      test-oracle (upper bound), on the recomputed final-epoch e29 arms.
  (3) label-definition PROTOCOL arithmetic: Offensive-excluded subset (Hateful-vs-Normal,
      n=125) and Offensive->0 relabel, scored with the SAME deployed predictions.
  (4) transcript-length quartile error rates; image-stream degeneracy check.
  (5) per-cluster max-flippable-item ceilings.
"""
import os, sys, json
from collections import Counter

import numpy as np
import torch
import faiss

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(REPO, "src"))
sys.path.insert(0, os.path.join(REPO, "scripts", "analysis"))
faiss.omp_set_num_threads(4)
torch.set_num_threads(4)
import cross_channel_router_gate as R  # noqa: E402

TOPK = 20
P2 = os.path.join(REPO, "scripts", "analysis", "p2_out")
CACHE = os.path.join(REPO, "data", "CLIP_Embedding", "MHC")
CKPT_DIR = os.path.join(REPO, "refine-logs", "router_ckpt_snapshot")
ANN = "/data/jehc223/Multihateclip/English/annotation(new).json"
IN_A = os.path.join(REPO, "scripts", "analysis", "errpat_mhc_en_out.json")
OUT = os.path.join(REPO, "scripts", "analysis", "errpat_mhc_en_b_out.json")

HUMAN2 = ["XScP1AiMkNM", "QvPp8Q7QhWE"]
RULE14 = ["YNf2tZgh4WM", "TRFp4a4lD0o", "My5PVJLP6Bg", "QvPp8Q7QhWE", "8Pim0TnLQDQ",
          "2ytDPK74q28", "aeOm9oT0_qk", "hKwgFaE7fbQ", "6hFEc1MLZC0", "lNCfDw80YSQ",
          "dcrX2-oto8Y", "EU-dip0ITa4", "XScP1AiMkNM", "Z2Cs5Oqm9iU"]
ANCHOR = {("Qwen", 0): (0.8012, 0.7596), ("Qwen", 1): (0.7702, 0.7203), ("Qwen", 2): (0.7826, 0.7475),
          ("CLIP", 0): (0.7640, 0.7145), ("CLIP", 1): (0.7826, 0.7159), ("CLIP", 2): (0.7888, 0.7303)}
DEV_ANCHOR = {("Qwen", 0): 0.7625, ("Qwen", 1): 0.7875, ("Qwen", 2): 0.7750,
              ("CLIP", 0): 0.7375, ("CLIP", 1): 0.7500, ("CLIP", 2): 0.7000}


def macro_f1(y, p):
    y = np.asarray(y); p = np.asarray(p); fs = []
    for c in (0, 1):
        tp = int(((p == c) & (y == c)).sum()); fp = int(((p == c) & (y != c)).sum())
        fn = int(((p != c) & (y == c)).sum())
        pr = tp / (tp + fp) if tp + fp else 0.0
        rc = tp / (tp + fn) if tp + fn else 0.0
        fs.append(2 * pr * rc / (pr + rc) if pr + rc else 0.0)
    return float(np.mean(fs))


def vote(nb_lab, nb_sim):
    k = len(nb_lab)
    w = np.arange(1, TOPK + 1)[::-1].astype("float64")[:k]
    lm = (np.asarray(nb_lab, dtype="float64") * 2 - 1) * np.asarray(nb_sim, dtype="float64")
    return float(np.sum(lm * w) / np.sum(w))


def load_cache(split, enc):
    m = {"CLIP": "openai_clip-vit-large-patch14-336_HF", "Qwen": "Qwen2.5-VL-7B-Instruct_HF"}[enc]
    d = torch.load(os.path.join(CACHE, f"{split}_{m}.pt"), map_location="cpu")
    ids = d["ids"]
    ids = ids[0] if (isinstance(ids, list) and len(ids) == 1 and isinstance(ids[0], list)) else ids
    return list(ids), d["img_feats"].float(), d["text_feats"].float(), d["labels"].long().numpy()


# ---------------------------------------------------- (1) deletion replay ----------
def deletion_replay(drop_ids, tag):
    res = {"tag": tag, "dropped": list(drop_ids), "per_seed": {}, "flips": {}}
    for s in range(4):
        d = json.load(open(os.path.join(P2, f"cache_MHC_s{s}.json")))
        y, p0, p1 = [], [], []
        flips = []
        for smp in d["samples"]:
            nb = smp["neighbors"]                      # top-60, retrieval order
            base = nb[:TOPK]
            v0 = vote([x[2] for x in base], [x[1] for x in base])
            kept = [x for x in nb if x[0] not in drop_ids]
            assert len(kept) >= TOPK, "top-60 refill exhausted; replay would not be exact"
            keep = kept[:TOPK]
            v1 = vote([x[2] for x in keep], [x[1] for x in keep])
            yy = int(smp["label"])
            a, b = int(v0 >= 0), int(v1 >= 0)
            y.append(yy); p0.append(a); p1.append(b)
            if a != b:
                flips.append(dict(id=smp["id"], y=yy, pred_before=a, pred_after=b,
                                  vote_before=round(v0, 6), vote_after=round(v1, 6),
                                  fixed=int(b == yy and a != yy), broke=int(a == yy and b != yy)))
        y = np.array(y); p0 = np.array(p0); p1 = np.array(p1)
        a0, a1 = float((y == p0).mean()), float((y == p1).mean())
        # sanity: baseline replay must equal the dumped floor
        assert abs(a0 - d["floor"]["acc"]) < 1e-12, (s, a0)
        res["per_seed"][str(s)] = dict(acc_before=a0, acc_after=a1, d_acc=a1 - a0,
                                       mf1_before=macro_f1(y, p0), mf1_after=macro_f1(y, p1),
                                       mf1_d=macro_f1(y, p1) - macro_f1(y, p0),
                                       n_flip=len(flips),
                                       n_fixed=sum(f["fixed"] for f in flips),
                                       n_broke=sum(f["broke"] for f in flips))
        res["flips"][str(s)] = flips
    ks = res["per_seed"]
    res["mean_d_acc"] = float(np.mean([ks[k]["d_acc"] for k in ks]))
    res["mean_d_mf1"] = float(np.mean([ks[k]["mf1_d"] for k in ks]))
    res["n_seeds_positive"] = int(sum(1 for k in ks if ks[k]["d_acc"] > 0))
    return res


# ------------------------------------------- (2) threshold family, both ways -------
def threshold_family():
    out = {}
    for enc in ("Qwen", "CLIP"):
        tr_ids, tr_img, tr_txt, tr_lab = load_cache("train", enc)
        dv_ids, dv_img, dv_txt, dv_lab = load_cache("dev_seen", enc)
        te_ids, te_img, te_txt, te_lab = load_cache("test_seen", enc)
        for s in (0, 1, 2):
            sd = torch.load(os.path.join(CKPT_DIR, f"MHC_{enc}_s{s}_e29.pt"), map_location="cpu")
            m = R.build_head(sd)
            tr_e = R.embed(m, tr_img, tr_txt)
            dv_v = np.array([r["vote"] for r in R.knn_channel(tr_e, tr_lab, R.embed(m, dv_img, dv_txt), False)])
            te_v = np.array([r["vote"] for r in R.knn_channel(tr_e, tr_lab, R.embed(m, te_img, te_txt), False)])
            dev_acc0 = float(((dv_v >= 0).astype(int) == dv_lab).mean())
            assert round(dev_acc0, 4) == DEV_ANCHOR[(enc, s)], (enc, s, dev_acc0)
            te_acc0 = float(((te_v >= 0).astype(int) == te_lab).mean())
            assert round(te_acc0, 4) == ANCHOR[(enc, s)][0], (enc, s, te_acc0)
            # dev-selected threshold (legal, deployable): maximise dev acc, then dev macroF1
            cand = np.unique(np.concatenate([[0.0], dv_v]))
            best = max(((float(((dv_v >= t).astype(int) == dv_lab).mean()),
                         macro_f1(dv_lab, (dv_v >= t).astype(int)), float(t)) for t in cand))
            t_dev = best[2]
            te_acc_dev = float(((te_v >= t_dev).astype(int) == te_lab).mean())
            te_mf1_dev = macro_f1(te_lab, (te_v >= t_dev).astype(int))
            # test-oracle threshold (UPPER BOUND, forensic only)
            candt = np.unique(np.concatenate([[0.0], te_v]))
            bo = max(((float(((te_v >= t).astype(int) == te_lab).mean()), float(t)) for t in candt))
            out[f"{enc}_s{s}"] = dict(
                dev_acc_at0=dev_acc0, test_acc_at0=te_acc0,
                test_mf1_at0=macro_f1(te_lab, (te_v >= 0).astype(int)),
                dev_selected_thr=t_dev, dev_acc_at_thr=best[0],
                test_acc_dev_thr=te_acc_dev, test_mf1_dev_thr=te_mf1_dev,
                d_acc_dev_thr=te_acc_dev - te_acc0, d_mf1_dev_thr=te_mf1_dev - macro_f1(te_lab, (te_v >= 0).astype(int)),
                test_oracle_thr=bo[1], test_oracle_acc=bo[0], d_acc_oracle=bo[0] - te_acc0)
    for enc in ("Qwen", "CLIP"):
        ks = [f"{enc}_s{s}" for s in (0, 1, 2)]
        out[f"mean_{enc}"] = dict(
            d_acc_dev_thr=float(np.mean([out[k]["d_acc_dev_thr"] for k in ks])),
            d_mf1_dev_thr=float(np.mean([out[k]["d_mf1_dev_thr"] for k in ks])),
            d_acc_oracle=float(np.mean([out[k]["d_acc_oracle"] for k in ks])),
            n_seeds_dev_thr_positive=int(sum(1 for k in ks if out[k]["d_acc_dev_thr"] > 0)))
    return out


# ----------------------------------- (3)(4)(5) protocol + covariate arithmetic -----
def protocol_and_covariates(A):
    pi = A["per_item"]
    ids = list(pi.keys())
    ann = {e["Video_ID"]: e for e in json.load(open(ANN))}

    # deployed val-sel per-seed predictions are reconstructable from n_wrong_v only at
    # consensus level, so re-read the caches for exact per-seed protocol arithmetic.
    seeds = {}
    for s in range(4):
        d = json.load(open(os.path.join(P2, f"cache_MHC_s{s}.json")))
        seeds[s] = {smp["id"]: (int(smp["label"]), int(smp["floor_pred"])) for smp in d["samples"]}

    out = {}
    # -- (3a) Offensive-excluded protocol: keep Hateful + Normal only
    keep = [v for v in ids if ann[v]["Label"] in ("Hateful", "Normal")]
    accs, mf1s = [], []
    for s in range(4):
        y = np.array([seeds[s][v][0] for v in keep]); p = np.array([seeds[s][v][1] for v in keep])
        accs.append(float((y == p).mean())); mf1s.append(macro_f1(y, p))
    out["protocol_offensive_excluded"] = dict(
        n=len(keep), n_pos=int(sum(seeds[0][v][0] for v in keep)),
        acc_mean=float(np.mean(accs)), acc_std=float(np.std(accs, ddof=1)),
        mf1_mean=float(np.mean(mf1s)), mf1_std=float(np.std(mf1s, ddof=1)),
        per_seed_acc=[round(a, 6) for a in accs],
        note="same deployed predictions, test subset restricted; no retraining")

    # -- (3b) Offensive->0 relabel (hateful_vs_rest), same predictions
    accs, mf1s = [], []
    for s in range(4):
        y = np.array([1 if ann[v]["Label"] == "Hateful" else 0 for v in ids])
        p = np.array([seeds[s][v][1] for v in ids])
        accs.append(float((y == p).mean())); mf1s.append(macro_f1(y, p))
    out["protocol_offensive_as_negative_samepred"] = dict(
        acc_mean=float(np.mean(accs)), mf1_mean=float(np.mean(mf1s)),
        note="scores the CURRENT binary-trained predictions against a Hateful-only target; "
             "diagnostic only (a hateful_vs_rest model would be retrained)")

    # -- per-class recall of the deployed arm
    rec = {}
    for L in ("Hateful", "Offensive"):
        sub = [v for v in ids if ann[v]["Label"] == L]
        r = [float(np.mean([seeds[s][v][1] for v in sub])) for s in range(4)]
        rec[L] = dict(n=len(sub), recall_mean=float(np.mean(r)), per_seed=[round(x, 4) for x in r])
    sub = [v for v in ids if ann[v]["Label"] == "Normal"]
    r = [float(np.mean([1 - seeds[s][v][1] for v in sub])) for s in range(4)]
    rec["Normal_specificity"] = dict(n=len(sub), mean=float(np.mean(r)), per_seed=[round(x, 4) for x in r])
    out["per_class_recall_deployed"] = rec

    # -- (4) transcript-length quartile error rate (consensus errors)
    nw = np.array([pi[v]["n_tr_word"] for v in ids])
    q = np.percentile(nw, [25, 50, 75])
    bins = {"Q1_shortest": [], "Q2": [], "Q3": [], "Q4_longest": []}
    for v in ids:
        n = pi[v]["n_tr_word"]
        k = "Q1_shortest" if n <= q[0] else "Q2" if n <= q[1] else "Q3" if n <= q[2] else "Q4_longest"
        bins[k].append(v)
    out["transcript_quartiles"] = dict(
        cutpoints_words=[float(x) for x in q],
        bins={k: dict(n=len(vs),
                      n_consensus_err=int(sum(1 for v in vs if pi[v]["n_wrong_v"] == 4)),
                      rate=round(sum(1 for v in vs if pi[v]["n_wrong_v"] == 4) / max(len(vs), 1), 4),
                      mean_words=round(float(np.mean([pi[v]["n_tr_word"] for v in vs])), 1))
              for k, vs in bins.items()})

    # -- (4b) image-stream degeneracy
    vimg = np.array([pi[v]["v_img"] for v in ids]); vtxt = np.array([pi[v]["v_txt"] for v in ids])
    y = np.array([pi[v]["y"] for v in ids])
    out["stream_degeneracy_Qwen_raw"] = dict(
        img_pred_pos_rate=float((vimg >= 0).mean()), img_true_pos_rate=float(y.mean()),
        img_recall_pos=float(((vimg >= 0) & (y == 1)).sum() / max((y == 1).sum(), 1)),
        img_recall_neg=float(((vimg < 0) & (y == 0)).sum() / max((y == 0).sum(), 1)),
        txt_pred_pos_rate=float((vtxt >= 0).mean()),
        txt_recall_pos=float(((vtxt >= 0) & (y == 1)).sum() / max((y == 1).sum(), 1)),
        txt_recall_neg=float(((vtxt < 0) & (y == 0)).sum() / max((y == 0).sum(), 1)),
        img_vote_absmean=float(np.abs(vimg).mean()), txt_vote_absmean=float(np.abs(vtxt).mean()))

    # -- (5) stream-selection ceiling on the consensus-error set
    cons = [v for v in ids if pi[v]["n_wrong_v"] == 4]
    out["stream_selection_ceiling_consensus"] = dict(
        n_consensus=len(cons),
        n_one_raw_stream_right=int(sum(1 for v in cons if pi[v]["img_right"] or pi[v]["txt_right"])),
        n_only_img_right=int(sum(1 for v in cons if pi[v]["img_right"] and not pi[v]["txt_right"])),
        n_only_txt_right=int(sum(1 for v in cons if pi[v]["txt_right"] and not pi[v]["img_right"])),
        n_both_raw_wrong=int(sum(1 for v in cons if not pi[v]["img_right"] and not pi[v]["txt_right"])),
        acc_delta_if_all_flipped=round(
            sum(1 for v in cons if pi[v]["img_right"] or pi[v]["txt_right"]) / len(ids), 4))

    # -- (5b) encoder-selection ceiling (Qwen vs CLIP, final-epoch majority)
    out["encoder_selection_ceiling"] = dict(
        n_qwen_wrong_clip_right=int(sum(1 for v in ids if pi[v]["n_wrong_f"] >= 2 and pi[v]["n_wrong_clip"] <= 1)),
        acc_delta_if_all_flipped=round(
            sum(1 for v in ids if pi[v]["n_wrong_f"] >= 2 and pi[v]["n_wrong_clip"] <= 1) / len(ids), 4))

    # -- (5c) neighbour-purity profile of consensus errors
    prof = [pi[v]["nb_correct_frac_v"] for v in cons]
    out["consensus_neighbour_purity"] = dict(
        n=len(cons), mean_correct_class_frac=float(np.mean(prof)),
        median=float(np.median(prof)),
        n_zero_correct_neighbours=int(sum(1 for x in prof if x == 0.0)),
        n_lt_10pct=int(sum(1 for x in prof if x < 0.10)),
        n_lt_25pct=int(sum(1 for x in prof if x < 0.25)),
        n_ge_50pct=int(sum(1 for x in prof if x >= 0.50)),
        hist=[round(float(x), 4) for x in sorted(prof)])
    return out


def main():
    A = json.load(open(IN_A))
    res = dict(meta=dict(part="B", cpu_only=True, gpu_jobs=0,
                         upstream=IN_A, forensic_test_read=True, tuning_from_test=False))
    res["deletion_human2"] = deletion_replay(set(HUMAN2), "human 2-entry (memory_editing_demo.py:76)")
    res["deletion_rule14"] = deletion_replay(set(RULE14), "14-entry rule-hit list")
    res["threshold_family"] = threshold_family()
    res.update(protocol_and_covariates(A))
    json.dump(res, open(OUT, "w"), indent=1, default=float)
    print("wrote", OUT)
    slim = {k: v for k, v in res.items()}
    for k in ("deletion_human2", "deletion_rule14"):
        slim[k] = {kk: vv for kk, vv in res[k].items() if kk != "flips"}
        slim[k]["flips_seed0"] = res[k]["flips"]["0"]
    print(json.dumps(slim, indent=1, default=float)[:14000])


if __name__ == "__main__":
    main()
