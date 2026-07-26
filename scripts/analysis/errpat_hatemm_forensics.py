#!/usr/bin/env python
"""
errpat_hatemm_forensics.py -- FORENSIC error-pattern analysis of the deployed
HateMM RGCL kNN pipeline. CPU only, read-only diagnostics.

IMPORTANT PROVENANCE NOTE
-------------------------
The deployed HateMM floor (job 13241) head checkpoints were deleted (F78), so
bit-exact floor per-item predictions are UNRECOVERABLE. This script analyses a
CPU-reconstructed PROXY: the byte-identical run_rac.py command re-run with
--device cpu --save_embed True on the banked LoRA-curric feature caches. The
proxy is validated against the floor trainlogs (see errpat_proxy_parity in the
output JSON). It is NEVER presented as the floor.

Everything computed on the test split is descriptive forensics. No selection or
tuning decision is derived from it.
"""
import argparse
import csv
import glob
import json
import os
import pickle
import statistics as st
import sys

import numpy as np
import torch

REPO = "/data/jehc223/RGCL"
PROXY_ROOT_DEFAULT = os.environ.get("ERRPAT_PROXY_ROOT", "")
FLOOR_LOG = os.path.join(
    REPO, "slurm/logs/enc3s_HateMM_Qwen2.5-VL-7B-Instruct-LoRA-curric_HF_seed{}_13241.trainlog")
FEAT = os.path.join(REPO, "data/CLIP_Embedding/HateMM/{}_Qwen2.5-VL-7B-Instruct-LoRA-curric_HF.pt")
GT = os.path.join(REPO, "data/gt/HateMM/{}.jsonl")
SPANS = os.path.join(REPO, "data/gt/HateMM/hate_spans.json")
ANNOT = os.path.join(REPO, "data/gt/HateMM/HateMM_annotation.csv")
TOPK = 20
NUM_FRAMES = 8  # deployed frame budget (F67: 8f saturates)


# ---------------------------------------------------------------- vote replay
def vote_from_entry(entry, topk=TOPK):
    """Bit-faithful replay of src/utils/metrics.py compute_metrics_retrieval
    (use_sim=True, majority_voting='arithmetic'). Decision = sigmoid(v) >= 0.5,
    equivalently v >= 0."""
    labs = np.asarray(entry["retrieved_label"], dtype=np.float64)
    sims = np.asarray([float(s) for s in entry["retrieved_scores"]], dtype=np.float64)
    w = np.arange(1, topk + 1)[::-1].astype(np.float64)  # [20,19,...,1]
    n = len(labs)
    mapped = (labs * 2 - 1) * sims
    return float(np.sum(mapped * w[:n]) / np.sum(w[:n]))


def macro_f1(y, p):
    y = np.asarray(y); p = np.asarray(p)
    f1s = []
    for c in (0, 1):
        tp = np.sum((p == c) & (y == c)); fp = np.sum((p == c) & (y != c))
        fn = np.sum((p != c) & (y == c))
        pr = tp / (tp + fp) if (tp + fp) else 0.0
        rc = tp / (tp + fn) if (tp + fn) else 0.0
        f1s.append(2 * pr * rc / (pr + rc) if (pr + rc) else 0.0)
    return float(np.mean(f1s))


# ---------------------------------------------------------------- trainlog parse
def parse_trainlog(path, warmup=5):
    import re
    log = open(path).read()
    val, test = {}, {}
    vre = re.compile(r"Val_Retrieval Epoch\s+(\d+) macroF1: ([\d.]+) macroP: ([\d.]+) "
                     r"macroR: ([\d.]+) acc: ([\d.]+) roc: ([\d.]+)")
    tre = re.compile(r"Test_Retrieval Epoch\s+(\d+) macroF1: ([\d.]+) macroP: ([\d.]+) "
                     r"macroR: ([\d.]+) acc: ([\d.]+) roc: ([\d.]+)")
    for m in vre.finditer(log):
        val[int(m.group(1))] = tuple(float(x) for x in m.groups()[1:])
    for m in tre.finditer(log):
        test[int(m.group(1))] = tuple(float(x) for x in m.groups()[1:])
    warm = [e for e in val if e >= warmup] or list(val)
    best = max(warm, key=lambda e: (val[e][3], val[e][4]))
    fe = max(test)
    # tuple order = (macroF1, macroP, macroR, acc, roc)
    return {"valsel_epoch": best, "valsel_acc": test[best][3], "valsel_mf1": test[best][0],
            "final_epoch": fe, "final_acc": test[fe][3], "final_mf1": test[fe][0]}


# ---------------------------------------------------------------- kNN vote engine
def knn_vote(query_feats, bank_feats, bank_labels, topk=TOPK, exclude_self=False):
    """Rank-weighted signed-cosine vote, same operator as the deployed head-space
    vote but over an arbitrary (L2-normalised) key space. Returns votes, top-k
    indices, top-k sims."""
    q = torch.nn.functional.normalize(query_feats.double(), p=2, dim=1)
    b = torch.nn.functional.normalize(bank_feats.double(), p=2, dim=1)
    sims = q @ b.T
    if exclude_self:
        n = min(sims.shape)
        sims[torch.arange(n), torch.arange(n)] = -2.0
    D, I = torch.topk(sims, topk, dim=1)
    D = D.numpy(); I = I.numpy()
    lab = bank_labels.numpy()[I]
    w = np.arange(1, topk + 1)[::-1].astype(np.float64)
    mapped = (lab * 2 - 1) * D
    votes = (mapped * w).sum(1) / w.sum()
    return votes, I, D, lab


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--proxy_root", default=PROXY_ROOT_DEFAULT, required=not PROXY_ROOT_DEFAULT)
    ap.add_argument("--out_json", default=os.path.join(REPO, "scripts/analysis/errpat_hatemm_forensics_OUT.json"))
    ap.add_argument("--out_csv", default=os.path.join(REPO, "scripts/analysis/errpat_hatemm_peritem.csv"))
    args = ap.parse_args()

    OUT = {"meta": {"nature": "FORENSIC PROXY (floor head ckpts deleted, F78)",
                    "cpu_only": True, "topk": TOPK,
                    "proxy_root": args.proxy_root,
                    "inputs": {}}}

    # ---------- 0. locate proxy run dirs
    dirs = {}
    for s in (0, 1, 2):
        g = sorted(glob.glob(os.path.join(
            args.proxy_root, "Retrieval/HateMM/RAC_errpat_proxy", "*seed%d*" % s)))
        assert len(g) == 1, (s, g)
        dirs[s] = g[0]

    # ---------- 1. floor vs proxy parity
    floor, proxy = {}, {}
    for s in (0, 1, 2):
        floor[s] = parse_trainlog(FLOOR_LOG.format(s))
        proxy[s] = parse_trainlog(os.path.join(args.proxy_root, "proxy_s%d.trainlog" % s))
    OUT["meta"]["inputs"]["floor_trainlogs"] = [FLOOR_LOG.format(s) for s in (0, 1, 2)]
    OUT["meta"]["inputs"]["proxy_trainlogs"] = [
        os.path.join(args.proxy_root, "proxy_s%d.trainlog" % s) for s in (0, 1, 2)]
    par = {"per_seed": {}, }
    for s in (0, 1, 2):
        par["per_seed"]["seed%d" % s] = {"floor": floor[s], "proxy": proxy[s]}
    for tag, src in (("floor", floor), ("proxy", proxy)):
        par[tag + "_mean"] = {
            "valsel_acc": round(st.mean([src[s]["valsel_acc"] for s in (0, 1, 2)]), 4),
            "valsel_mf1": round(st.mean([src[s]["valsel_mf1"] for s in (0, 1, 2)]), 4),
            "final_acc": round(st.mean([src[s]["final_acc"] for s in (0, 1, 2)]), 4),
            "final_mf1": round(st.mean([src[s]["final_mf1"] for s in (0, 1, 2)]), 4)}
    par["proxy_minus_floor"] = {k: round(par["proxy_mean"][k] - par["floor_mean"][k], 4)
                               for k in par["floor_mean"]}
    OUT["errpat_proxy_parity"] = par

    # ---------- 2. load per-item logging dicts (test), val-sel + final epoch
    per = {}   # per[(seed, protocol)][vid] = dict
    for s in (0, 1, 2):
        for proto, ep in (("valsel", proxy[s]["valsel_epoch"]), ("final", proxy[s]["final_epoch"])):
            p = os.path.join(dirs[s], "testepoch_%d_retrieval_logging_dict.pkl" % ep)
            ld = pickle.load(open(p, "rb"))["logging_dict"]
            d = {}
            for vid, e in ld.items():
                v = vote_from_entry(e)
                d[vid] = {"vote": v, "pred": int(v >= 0.0),
                          "nbr_labels": [int(x) for x in e["retrieved_label"]],
                          "nbr_ids": list(e["retrieved_ids"]),
                          "nbr_sims": [float(x) for x in e["retrieved_scores"]]}
            per[(s, proto)] = {"epoch": ep, "path": p, "items": d}

    # ---------- 3. labels + covariates
    gt = {}
    for sp, tag in (("train", "train"), ("val", "dev"), ("test", "test")):
        for l in open(GT.format(sp)):
            r = json.loads(l)
            gt[r["id"]] = {"split": tag, "label": int(r["label"]), "text": r.get("text", "")}
    spans = json.load(open(SPANS))
    tgt = {}
    with open(ANNOT) as f:
        for row in csv.DictReader(f):
            vid = row["video_file_name"].rsplit(".", 1)[0]
            tgt[vid] = (row.get("target") or "").strip()

    test_ids = [json.loads(l)["id"] for l in open(GT.format("test"))]
    labels = np.array([gt[v]["label"] for v in test_ids])
    OUT["meta"]["inputs"]["gt"] = [GT.format(x) for x in ("train", "val", "test")]
    OUT["meta"]["inputs"]["spans"] = SPANS
    OUT["meta"]["inputs"]["annot"] = ANNOT
    OUT["meta"]["n_test"] = len(test_ids)
    OUT["meta"]["n_test_hate"] = int(labels.sum())
    OUT["meta"]["n_test_nonhate"] = int((1 - labels).sum())

    # ---------- 4. vote-replay parity vs trainlog
    replay = {}
    for (s, proto), blk in per.items():
        p = np.array([blk["items"][v]["pred"] for v in test_ids])
        acc = float(np.mean(p == labels)); mf1 = macro_f1(labels, p)
        ref_acc = proxy[s][proto + "_acc"]; ref_mf1 = proxy[s][proto + "_mf1"]
        replay["seed%d_%s" % (s, proto)] = {
            "replay_acc": round(acc, 4), "trainlog_acc": ref_acc,
            "replay_mf1": round(mf1, 4), "trainlog_mf1": ref_mf1,
            "bit_exact_4dp": bool(round(acc, 4) == ref_acc and round(mf1, 4) == ref_mf1)}
    OUT["vote_replay_parity"] = replay

    # ---------- 5. stream forensics (raw banked encoder space, $0)
    feats = {}
    for split, key in (("train", "train"), ("test_seen", "test")):
        d = torch.load(FEAT.format(split), map_location="cpu", weights_only=False)
        ids = d["ids"][0] if (isinstance(d["ids"], list) and len(d["ids"]) == 1) else d["ids"]
        feats[key] = {"ids": list(ids), "img": d["img_feats"], "txt": d["text_feats"],
                      "lab": d["labels"]}
    OUT["meta"]["inputs"]["feature_caches"] = [FEAT.format(x) for x in ("train", "test_seen")]
    assert feats["test"]["ids"] == test_ids, "test id order mismatch"

    stream = {}
    for name, k in (("image", "img"), ("text", "txt")):
        votes, I, D, lab = knn_vote(feats["test"][k], feats["train"][k], feats["train"]["lab"])
        pred = (votes >= 0).astype(int)
        stream[name] = {"vote": votes, "pred": pred,
                        "acc": round(float(np.mean(pred == labels)), 4),
                        "mf1": round(macro_f1(labels, pred), 4)}
    # fused raw-encoder control: L2-normalised concat of both streams
    cat_tr = torch.cat([torch.nn.functional.normalize(feats["train"]["img"].double(), dim=1),
                        torch.nn.functional.normalize(feats["train"]["txt"].double(), dim=1)], 1)
    cat_te = torch.cat([torch.nn.functional.normalize(feats["test"]["img"].double(), dim=1),
                        torch.nn.functional.normalize(feats["test"]["txt"].double(), dim=1)], 1)
    v, _, _, _ = knn_vote(cat_te, cat_tr, feats["train"]["lab"])
    stream["raw_concat"] = {"vote": v, "pred": (v >= 0).astype(int),
                            "acc": round(float(np.mean((v >= 0).astype(int) == labels)), 4),
                            "mf1": round(macro_f1(labels, (v >= 0).astype(int)), 4)}
    OUT["stream_untrained_knn"] = {
        k: {"acc": stream[k]["acc"], "mf1": stream[k]["mf1"]} for k in stream}

    # ---------- 6. head-space single-stream forensics (proxy head, epoch-wise)
    sys.path.insert(0, os.path.join(REPO, "src"))
    from model.classifier import classifier_hateClipper

    class A:  # minimal args shim
        dataset = "HateMM"; mod_dropout = False; mod_dropout_p = 0.3
    head_stream = {}
    for s in (0, 1, 2):
        for proto in ("valsel", "final"):
            ep = per[(s, proto)]["epoch"]
            cands = glob.glob(os.path.join(glob.escape(dirs[s]), "ckpt",
                                           "epoch_model_%d_*.pt" % ep)) or \
                    glob.glob(os.path.join(glob.escape(dirs[s]), "ckpt",
                                           "best_model_%d_*.pt" % ep))
            assert cands, (s, proto, ep)
            sd = torch.load(sorted(cands)[0], map_location="cpu", weights_only=False)
            if not isinstance(sd, dict) or "img_proj.0.weight" not in sd:
                sd = sd.state_dict() if hasattr(sd, "state_dict") else sd
            m = classifier_hateClipper(3584, 3584, 3, 1024, 1024, "align",
                                       dropout=[0.2, 0.4, 0.1], batch_norm=False, args=A())
            m.load_state_dict(sd); m.eval()
            with torch.no_grad():
                out = {}
                for tag in ("train", "test"):
                    ip = torch.nn.functional.normalize(m.img_proj(feats[tag]["img"]), p=2, dim=1)
                    tp = torch.nn.functional.normalize(m.text_proj(feats[tag]["txt"]), p=2, dim=1)
                    fused = m.mlp[:-2](torch.mul(ip, tp))
                    out[tag] = {"img": ip, "txt": tp, "fused": fused}
            hs = {}
            for nm in ("img", "txt", "fused"):
                vv, _, _, _ = knn_vote(out["test"][nm], out["train"][nm], feats["train"]["lab"])
                pp = (vv >= 0).astype(int)
                hs[nm] = {"vote": vv, "pred": pp,
                          "acc": round(float(np.mean(pp == labels)), 4),
                          "mf1": round(macro_f1(labels, pp), 4)}
            head_stream[(s, proto)] = hs
    OUT["stream_headspace_knn"] = {
        "seed%d_%s" % (s, p): {nm: {"acc": head_stream[(s, p)][nm]["acc"],
                                    "mf1": head_stream[(s, p)][nm]["mf1"]}
                               for nm in ("img", "txt", "fused")}
        for (s, p) in head_stream}

    # ---------- 7. per-item table
    def cov(vid):
        sp = spans.get(vid, {})
        dur = float(sp.get("duration", float("nan")))
        sg = sp.get("spans", []) or []
        span_s = sum(max(0.0, b - a) for a, b in sg)
        txt = gt[vid]["text"]
        nw = len(txt.split())
        # do the 8 uniformly-sampled frames land inside a gold hate span?
        hit = None
        if dur == dur and dur > 0 and sg:
            ts = np.linspace(0.0, dur, NUM_FRAMES)
            hit = int(sum(1 for t in ts if any(a <= t <= b for a, b in sg)))
        return {"duration_s": round(dur, 3) if dur == dur else None,
                "n_spans": len(sg),
                "span_seconds": round(span_s, 3),
                "span_frac": round(span_s / dur, 4) if (dur == dur and dur > 0) else None,
                "frames_in_span_of8": hit,
                "n_words": nw, "empty_text": int(nw <= 1),
                "target": tgt.get(vid, "")}

    rows = []
    for i, vid in enumerate(test_ids):
        r = {"id": vid, "label": int(labels[i])}
        r.update(cov(vid))
        for s in (0, 1, 2):
            for proto in ("valsel", "final"):
                it = per[(s, proto)]["items"][vid]
                pfx = "s%d_%s" % (s, proto)
                r[pfx + "_vote"] = round(it["vote"], 6)
                r[pfx + "_pred"] = it["pred"]
                r[pfx + "_err"] = int(it["pred"] != labels[i])
                nl = np.array(it["nbr_labels"])
                r[pfx + "_purity_true"] = round(float(np.mean(nl == labels[i])), 4)
                r[pfx + "_top1_sim"] = round(it["nbr_sims"][0], 6)
                r[pfx + "_top1_correct"] = int(nl[0] == labels[i])
                fc = np.where(nl == labels[i])[0]
                r[pfx + "_first_correct_rank"] = int(fc[0] + 1) if len(fc) else 0
                # rank-weighted purity toward the true label
                w = np.arange(1, TOPK + 1)[::-1].astype(float)
                r[pfx + "_wpurity_true"] = round(float((w * (nl == labels[i])).sum() / w.sum()), 4)
                r[pfx + "_head_img_pred"] = int(head_stream[(s, proto)]["img"]["pred"][i])
                r[pfx + "_head_txt_pred"] = int(head_stream[(s, proto)]["txt"]["pred"][i])
        r["raw_img_pred"] = int(stream["image"]["pred"][i])
        r["raw_txt_pred"] = int(stream["text"]["pred"][i])
        r["raw_img_vote"] = round(float(stream["image"]["vote"][i]), 6)
        r["raw_txt_vote"] = round(float(stream["text"]["vote"][i]), 6)
        r["n_err_final_3seed"] = sum(r["s%d_final_err" % s] for s in (0, 1, 2))
        r["n_err_valsel_3seed"] = sum(r["s%d_valsel_err" % s] for s in (0, 1, 2))
        rows.append(r)
    with open(args.out_csv, "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        wr.writeheader(); wr.writerows(rows)
    OUT["meta"]["per_item_csv"] = args.out_csv

    # ---------- 8. error counts + FP/FN split
    err = {}
    for (s, proto), blk in per.items():
        p = np.array([blk["items"][v]["pred"] for v in test_ids])
        fp = int(np.sum((p == 1) & (labels == 0))); fn = int(np.sum((p == 0) & (labels == 1)))
        err["seed%d_%s" % (s, proto)] = {"epoch": blk["epoch"], "n_err": fp + fn,
                                         "FP": fp, "FN": fn,
                                         "acc": round(float(np.mean(p == labels)), 4)}
    OUT["error_counts"] = err

    # ---------- 9. threshold reachability (global-threshold sweep on the vote)
    thr = {}
    for (s, proto), blk in per.items():
        v = np.array([blk["items"][x]["vote"] for x in test_ids])
        base = (v >= 0).astype(int)
        acc0 = float(np.mean(base == labels))
        cand = np.unique(np.concatenate([[-1e9, 0.0, 1e9], v - 1e-12, v + 1e-12]))
        accs = np.array([np.mean(((v >= t).astype(int)) == labels) for t in cand])
        ok = cand[accs >= acc0 - 1e-12]
        reach = np.zeros(len(test_ids), dtype=bool)
        for t in ok:
            pr = (v >= t).astype(int)
            reach |= ((pr == labels) & (base != labels))
        bestt = float(cand[int(np.argmax(accs))]); bestacc = float(accs.max())
        errmask = base != labels
        thr["seed%d_%s" % (s, proto)] = {
            "acc_deployed_thr0": round(acc0, 4),
            "best_global_thr_FORENSIC": round(bestt, 6),
            "acc_at_best_thr_FORENSIC": round(bestacc, 4),
            "n_err": int(errmask.sum()),
            "n_threshold_reachable": int((reach & errmask).sum()),
            "n_vote_locked": int((errmask & ~reach).sum()),
            "n_recovered_at_best_thr": int(np.sum(((v >= bestt).astype(int) == labels) & errmask)),
            "reachable_ids": [test_ids[i] for i in np.where(reach & errmask)[0]]}
    OUT["threshold_reachability"] = thr

    # ---------- 10. neighbourhood lock (majority of top-20 carries the wrong label)
    nb = {}
    for (s, proto), blk in per.items():
        cnt = {"err_purity_lt_0.5": 0, "err_purity_ge_0.5": 0, "err_top1_correct": 0}
        for i, vid in enumerate(test_ids):
            it = blk["items"][vid]
            if it["pred"] == labels[i]:
                continue
            nl = np.array(it["nbr_labels"])
            pu = float(np.mean(nl == labels[i]))
            cnt["err_purity_lt_0.5" if pu < 0.5 else "err_purity_ge_0.5"] += 1
            cnt["err_top1_correct"] += int(nl[0] == labels[i])
        nb["seed%d_%s" % (s, proto)] = cnt
    OUT["neighbourhood_lock"] = nb

    # ---------- 11. stream cross-tab on errors (head space, per seed/proto)
    xt = {}
    for (s, proto), blk in per.items():
        hs = head_stream[(s, proto)]
        c = {"both_streams_wrong": 0, "text_right_image_wrong": 0,
             "image_right_text_wrong": 0, "fusion_lost_it": 0}
        cats = {}
        for i, vid in enumerate(test_ids):
            if blk["items"][vid]["pred"] == labels[i]:
                continue
            ir = hs["img"]["pred"][i] == labels[i]
            tr = hs["txt"]["pred"][i] == labels[i]
            k = ("fusion_lost_it" if (ir and tr) else
                 "text_right_image_wrong" if tr else
                 "image_right_text_wrong" if ir else "both_streams_wrong")
            c[k] += 1; cats[vid] = k
        xt["seed%d_%s" % (s, proto)] = {"counts": c, "per_item": cats}
    OUT["stream_crosstab_headspace"] = xt

    # raw-encoder-space cross-tab, consensus errors (seed-agnostic streams)
    OUT["stream_crosstab_raw"] = {}
    for proto in ("valsel", "final"):
        for nmin in (1, 2, 3):
            ids = [v for v in test_ids
                   if sum(int(per[(s, proto)]["items"][v]["pred"] != gt[v]["label"])
                          for s in (0, 1, 2)) >= nmin]
            c = {"both_streams_wrong": 0, "text_right_image_wrong": 0,
                 "image_right_text_wrong": 0, "fusion_lost_it": 0}
            for v in ids:
                i = test_ids.index(v); y = labels[i]
                ir = stream["image"]["pred"][i] == y; tr = stream["text"]["pred"][i] == y
                k = ("fusion_lost_it" if (ir and tr) else
                     "text_right_image_wrong" if tr else
                     "image_right_text_wrong" if ir else "both_streams_wrong")
                c[k] += 1
            OUT["stream_crosstab_raw"]["%s_err_ge%d_seeds" % (proto, nmin)] = {
                "n": len(ids), "counts": c}

    # ---------- 12. seed stability + protocol flips
    stab = {}
    for proto in ("valsel", "final"):
        hist = {0: 0, 1: 0, 2: 0, 3: 0}
        ids3 = []
        for v in test_ids:
            n = sum(int(per[(s, proto)]["items"][v]["pred"] != gt[v]["label"]) for s in (0, 1, 2))
            hist[n] += 1
            if n == 3:
                ids3.append(v)
        stab[proto] = {"n_wrong_in_k_seeds": {str(k): hist[k] for k in hist},
                       "wrong_3of3_ids": ids3,
                       "union_ever_wrong": int(sum(hist[k] for k in (1, 2, 3)))}
    OUT["seed_stability"] = stab

    flips = {}
    for s in (0, 1, 2):
        a = per[(s, "valsel")]["items"]; b = per[(s, "final")]["items"]
        ch = [v for v in test_ids if a[v]["pred"] != b[v]["pred"]]
        flips["seed%d" % s] = {"valsel_epoch": per[(s, "valsel")]["epoch"],
                               "final_epoch": per[(s, "final")]["epoch"],
                               "n_pred_changed": len(ch), "ids": ch}
    OUT["protocol_flips"] = flips

    # ---------- 13. covariate contrasts: errors vs correct
    def grp(mask_ids, key):
        vals = [r[key] for r in rows if r["id"] in mask_ids and r[key] is not None]
        if not vals:
            return None
        return {"n": len(vals), "median": round(float(np.median(vals)), 4),
                "mean": round(float(np.mean(vals)), 4)}
    cova = {}
    for proto in ("valsel", "final"):
        e3 = set(v for v in test_ids
                 if sum(int(per[(s, proto)]["items"][v]["pred"] != gt[v]["label"])
                        for s in (0, 1, 2)) >= 2)
        c0 = set(test_ids) - set(v for v in test_ids
                                 if sum(int(per[(s, proto)]["items"][v]["pred"] != gt[v]["label"])
                                        for s in (0, 1, 2)) >= 1)
        blk = {}
        for key in ("n_words", "duration_s", "span_frac", "frames_in_span_of8"):
            blk[key] = {"err_ge2seeds": grp(e3, key), "always_correct": grp(c0, key)}
        blk["empty_text_rate"] = {
            "err_ge2seeds": round(float(np.mean([r["empty_text"] for r in rows if r["id"] in e3])), 4)
            if e3 else None,
            "always_correct": round(float(np.mean([r["empty_text"] for r in rows if r["id"] in c0])), 4)}
        blk["n_err_ge2seeds"] = len(e3); blk["n_always_correct"] = len(c0)
        cova[proto] = blk
    OUT["covariates"] = cova

    # ---------- 14. target-group breakdown of hard errors
    tb = {}
    for proto in ("valsel", "final"):
        d = {}
        for v in test_ids:
            n = sum(int(per[(s, proto)]["items"][v]["pred"] != gt[v]["label"]) for s in (0, 1, 2))
            t = tgt.get(v, "") or "(none)"
            d.setdefault(t, {"n_total": 0, "n_err_ge2": 0})
            d[t]["n_total"] += 1
            d[t]["n_err_ge2"] += int(n >= 2)
        tb[proto] = d
    OUT["target_breakdown"] = tb

    # ---------- 15. train-side memory: LOO label-noise proxy on the bank
    votes_tr, I_tr, D_tr, lab_tr = knn_vote(feats["train"]["img"], feats["train"]["img"],
                                            feats["train"]["lab"], exclude_self=True)
    # head-space LOO for seed0/final (the deployed key space)
    hs0 = head_stream[(0, "final")]
    sd = torch.load(sorted(glob.glob(os.path.join(
        glob.escape(dirs[0]), "ckpt",
        "epoch_model_%d_*.pt" % per[(0, "final")]["epoch"])))[0],
        map_location="cpu", weights_only=False)
    m = classifier_hateClipper(3584, 3584, 3, 1024, 1024, "align",
                               dropout=[0.2, 0.4, 0.1], batch_norm=False, args=A())
    m.load_state_dict(sd); m.eval()
    with torch.no_grad():
        ip = torch.nn.functional.normalize(m.img_proj(feats["train"]["img"]), p=2, dim=1)
        tp = torch.nn.functional.normalize(m.text_proj(feats["train"]["txt"]), p=2, dim=1)
        fused_tr = m.mlp[:-2](torch.mul(ip, tp))
    v_loo, _, _, _ = knn_vote(fused_tr, fused_tr, feats["train"]["lab"], exclude_self=True)
    p_loo = (v_loo >= 0).astype(int)
    ytr = feats["train"]["lab"].numpy()
    OUT["train_bank_loo"] = {
        "space": "seed0 final-epoch head (fused) space, LOO",
        "n_train": int(len(ytr)),
        "loo_acc": round(float(np.mean(p_loo == ytr)), 4),
        "n_loo_disagree": int(np.sum(p_loo != ytr)),
        "loo_disagree_rate": round(float(np.mean(p_loo != ytr)), 4)}

    # ---------- 16. how much memory-bank purity the errors see
    OUT["error_bank_composition"] = {}
    for proto in ("valsel", "final"):
        agg = []
        for v in test_ids:
            n = sum(int(per[(s, proto)]["items"][v]["pred"] != gt[v]["label"]) for s in (0, 1, 2))
            if n >= 2:
                pu = st.mean([float(np.mean(np.array(per[(s, proto)]["items"][v]["nbr_labels"])
                                            == gt[v]["label"])) for s in (0, 1, 2)])
                agg.append(pu)
        OUT["error_bank_composition"][proto] = {
            "n_err_ge2seeds": len(agg),
            "mean_top20_purity_toward_true": round(float(np.mean(agg)), 4) if agg else None,
            "n_with_purity_lt_0.25": int(sum(1 for x in agg if x < 0.25)),
            "n_with_purity_0.25_0.5": int(sum(1 for x in agg if 0.25 <= x < 0.5)),
            "n_with_purity_ge_0.5": int(sum(1 for x in agg if x >= 0.5))}

    with open(args.out_json, "w") as f:
        json.dump(OUT, f, indent=2, default=str)
    print("wrote", args.out_json)
    print("wrote", args.out_csv)
    print(json.dumps({k: OUT[k] for k in ("errpat_proxy_parity", "vote_replay_parity",
                                          "error_counts", "threshold_reachability",
                                          "stream_untrained_knn", "seed_stability")},
                     indent=2, default=str)[:6000])


if __name__ == "__main__":
    main()
