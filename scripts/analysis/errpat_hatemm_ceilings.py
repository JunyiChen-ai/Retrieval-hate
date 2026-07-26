#!/usr/bin/env python
"""
errpat_hatemm_ceilings.py -- second forensic pass on the HateMM proxy.

Adds, on top of errpat_hatemm_forensics.py:
  (a) FP/FN-split covariates (the first pass pooled classes, which confounds
      span_frac with label);
  (b) error-SET overlap between fused / text-only / image-only head-space kNN;
  (c) quantified ceilings for candidate fixes:
        C1 global-threshold recalibration (test-fitted = FORENSIC ORACLE);
        C2 dev-fitted global threshold (LEGAL: dev only, no test touch);
        C3 memory-bank curation by TRAIN-only LOO disagreement (LEGAL);
        C4 oracle per-item channel selection fused-vs-text (F66-BANNED,
           quoted only as an upper bound);
  (d) exemplar dump per cluster with transcript snippets.

CPU only. Read-only on all banked artifacts. Proxy, not the floor (F78).
"""
import argparse, csv, glob, json, os, pickle, statistics as st, sys
import numpy as np
import torch

REPO = "/data/jehc223/RGCL"
FEAT = os.path.join(REPO, "data/CLIP_Embedding/HateMM/{}_Qwen2.5-VL-7B-Instruct-LoRA-curric_HF.pt")
GT = os.path.join(REPO, "data/gt/HateMM/{}.jsonl")
SPANS = os.path.join(REPO, "data/gt/HateMM/hate_spans.json")
ANNOT = os.path.join(REPO, "data/gt/HateMM/HateMM_annotation.csv")
TOPK = 20
sys.path.insert(0, os.path.join(REPO, "src"))
from model.classifier import classifier_hateClipper  # noqa: E402


class A:
    dataset = "HateMM"; mod_dropout = False; mod_dropout_p = 0.3


def macro_f1(y, p):
    y = np.asarray(y); p = np.asarray(p); f1 = []
    for c in (0, 1):
        tp = np.sum((p == c) & (y == c)); fp = np.sum((p == c) & (y != c))
        fn = np.sum((p != c) & (y == c))
        pr = tp / (tp + fp) if (tp + fp) else 0.0
        rc = tp / (tp + fn) if (tp + fn) else 0.0
        f1.append(2 * pr * rc / (pr + rc) if (pr + rc) else 0.0)
    return float(np.mean(f1))


def knn_vote(q, b, blab, topk=TOPK, drop_bank=None, exclude_self=False):
    qn = torch.nn.functional.normalize(q.double(), p=2, dim=1)
    bn = torch.nn.functional.normalize(b.double(), p=2, dim=1)
    lab = blab.numpy().copy()
    if drop_bank is not None:
        keep = np.ones(bn.shape[0], bool); keep[np.asarray(drop_bank, int)] = False
        bn = bn[torch.from_numpy(keep)]; lab = lab[keep]
    sims = qn @ bn.T
    if exclude_self:
        n = min(sims.shape); sims[torch.arange(n), torch.arange(n)] = -2.0
    D, I = torch.topk(sims, topk, dim=1)
    D = D.numpy(); L = lab[I.numpy()]
    w = np.arange(1, topk + 1)[::-1].astype(float)
    return ((L * 2 - 1) * D * w).sum(1) / w.sum(), I.numpy(), D, L


def best_thr(v, y):
    cand = np.unique(np.concatenate([[-1e9, 0.0, 1e9], v - 1e-12, v + 1e-12]))
    accs = np.array([np.mean(((v >= t).astype(int)) == y) for t in cand])
    i = int(np.argmax(accs))
    return float(cand[i]), float(accs[i])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--proxy_root", required=True)
    ap.add_argument("--out_json", default=os.path.join(REPO, "scripts/analysis/errpat_hatemm_ceilings_OUT.json"))
    args = ap.parse_args()
    O = {"meta": {"nature": "FORENSIC PROXY second pass", "cpu_only": True, "topk": TOPK,
                  "inputs": {"proxy_root": args.proxy_root,
                             "feature_caches": [FEAT.format(x) for x in ("train", "dev_seen", "test_seen")],
                             "gt": [GT.format(x) for x in ("train", "val", "test")],
                             "spans": SPANS, "annot": ANNOT}}}

    dirs = {s: sorted(glob.glob(os.path.join(args.proxy_root,
            "Retrieval/HateMM/RAC_errpat_proxy", "*seed%d*" % s)))[0] for s in (0, 1, 2)}
    EPOCH = {0: {"valsel": 25, "final": 29}, 1: {"valsel": 15, "final": 29},
             2: {"valsel": 29, "final": 29}}
    O["meta"]["selected_epochs"] = EPOCH

    # data
    gt = {}
    for sp in ("train", "val", "test"):
        for l in open(GT.format(sp)):
            r = json.loads(l); gt[r["id"]] = {"label": int(r["label"]), "text": r.get("text", ""), "split": sp}
    test_ids = [json.loads(l)["id"] for l in open(GT.format("test"))]
    y = np.array([gt[v]["label"] for v in test_ids])
    spans = json.load(open(SPANS))
    tgt = {}
    with open(ANNOT) as f:
        for row in csv.DictReader(f):
            tgt[row["video_file_name"].rsplit(".", 1)[0]] = (row.get("target") or "").strip()

    feats = {}
    for split, key in (("train", "train"), ("dev_seen", "dev"), ("test_seen", "test")):
        d = torch.load(FEAT.format(split), map_location="cpu", weights_only=False)
        ids = d["ids"][0] if (isinstance(d["ids"], list) and len(d["ids"]) == 1) else d["ids"]
        feats[key] = {"ids": list(ids), "img": d["img_feats"], "txt": d["text_feats"], "lab": d["labels"]}
    assert feats["test"]["ids"] == test_ids
    ytr = feats["train"]["lab"].numpy()
    ydv = feats["dev"]["lab"].numpy()

    # per-seed/proto head spaces + deployed votes
    heads, votes = {}, {}
    for s in (0, 1, 2):
        for proto in ("valsel", "final"):
            ep = EPOCH[s][proto]
            ck = sorted(glob.glob(os.path.join(glob.escape(dirs[s]), "ckpt", "epoch_model_%d_*.pt" % ep)))[0]
            sd = torch.load(ck, map_location="cpu", weights_only=False)
            m = classifier_hateClipper(3584, 3584, 3, 1024, 1024, "align",
                                       dropout=[0.2, 0.4, 0.1], batch_norm=False, args=A())
            m.load_state_dict(sd); m.eval()
            sp = {}
            with torch.no_grad():
                for tag in ("train", "dev", "test"):
                    ip = torch.nn.functional.normalize(m.img_proj(feats[tag]["img"]), p=2, dim=1)
                    tp = torch.nn.functional.normalize(m.text_proj(feats[tag]["txt"]), p=2, dim=1)
                    sp[tag] = {"img": ip, "txt": tp, "fused": m.mlp[:-2](torch.mul(ip, tp))}
            heads[(s, proto)] = {"ckpt": ck, "space": sp}
            # deployed vote from the saved logging dict (authoritative)
            ld = pickle.load(open(os.path.join(dirs[s], "testepoch_%d_retrieval_logging_dict.pkl" % ep), "rb"))["logging_dict"]
            w = np.arange(1, TOPK + 1)[::-1].astype(float)
            vv = np.array([(np.asarray(ld[v]["retrieved_label"], float) * 2 - 1)
                           * np.asarray([float(x) for x in ld[v]["retrieved_scores"]]) @ w / w.sum()
                           for v in test_ids])
            votes[(s, proto)] = vv

    # ---------------- (a) FP/FN-split covariates
    def covrow(vid):
        sp = spans.get(vid, {}); dur = float(sp.get("duration", float("nan")))
        sg = sp.get("spans", []) or []
        ss = sum(max(0.0, b - a) for a, b in sg)
        nw = len(gt[vid]["text"].split())
        hit = None
        if dur == dur and dur > 0 and sg:
            ts = np.linspace(0.0, dur, 8)
            hit = int(sum(1 for t in ts if any(a <= t <= b for a, b in sg)))
        return {"dur": dur, "span_frac": (ss / dur) if (dur == dur and dur > 0) else None,
                "nw": nw, "frames_in_span": hit}

    def agg(ids, key):
        vals = [covrow(v)[key] for v in ids]
        vals = [x for x in vals if x is not None and x == x]
        if not vals: return None
        return {"n": len(vals), "median": round(float(np.median(vals)), 4),
                "mean": round(float(np.mean(vals)), 4)}

    cons = {}
    for proto in ("valsel", "final"):
        nerr = {v: sum(int((votes[(s, proto)][i] >= 0) != y[i]) for s in (0, 1, 2))
                for i, v in enumerate(test_ids)}
        cons[proto] = nerr
    O["consensus_error_counts"] = {p: {"n_err_3of3": sum(1 for v in cons[p] if cons[p][v] == 3),
                                       "n_err_ge2": sum(1 for v in cons[p] if cons[p][v] >= 2),
                                       "n_err_ge1": sum(1 for v in cons[p] if cons[p][v] >= 1)}
                                   for p in cons}
    cv = {}
    for proto in ("valsel", "final"):
        nerr = cons[proto]
        FN = [v for i, v in enumerate(test_ids) if y[i] == 1 and nerr[v] >= 2]
        FP = [v for i, v in enumerate(test_ids) if y[i] == 0 and nerr[v] >= 2]
        okH = [v for i, v in enumerate(test_ids) if y[i] == 1 and nerr[v] == 0]
        okN = [v for i, v in enumerate(test_ids) if y[i] == 0 and nerr[v] == 0]
        blk = {}
        for nm, ids in (("FN_hate_missed", FN), ("hate_correct", okH),
                        ("FP_nonhate_flagged", FP), ("nonhate_correct", okN)):
            blk[nm] = {"n": len(ids),
                       "nw": agg(ids, "nw"), "dur": agg(ids, "dur"),
                       "span_frac": agg(ids, "span_frac"),
                       "frames_in_span": agg(ids, "frames_in_span"),
                       "empty_text_rate": round(float(np.mean([len(gt[v]["text"].split()) <= 1 for v in ids])), 4) if ids else None}
        cv[proto] = blk
    O["covariates_by_class"] = cv

    # ---------------- (b) error-set overlap fused / txt / img (head space)
    ov = {}
    for s in (0, 1, 2):
        for proto in ("valsel", "final"):
            sp = heads[(s, proto)]["space"]
            pr = {}
            for nm in ("fused", "txt", "img"):
                v, _, _, _ = knn_vote(sp["test"][nm], sp["train"][nm], feats["train"]["lab"])
                pr[nm] = (v >= 0).astype(int)
            pr["deployed"] = (votes[(s, proto)] >= 0).astype(int)
            E = {nm: set(np.where(pr[nm] != y)[0]) for nm in pr}
            ov["seed%d_%s" % (s, proto)] = {
                "acc": {nm: round(float(np.mean(pr[nm] == y)), 4) for nm in pr},
                "mf1": {nm: round(macro_f1(y, pr[nm]), 4) for nm in pr},
                "n_err": {nm: len(E[nm]) for nm in E},
                "fused_err_that_txt_fixes": len(E["deployed"] - E["txt"]),
                "txt_new_err_not_in_fused": len(E["txt"] - E["deployed"]),
                "fused_err_that_img_fixes": len(E["deployed"] - E["img"]),
                "img_new_err_not_in_fused": len(E["img"] - E["deployed"]),
                "err_in_all_three": len(E["deployed"] & E["txt"] & E["img"]),
                "oracle_pick_of_fused_txt_acc_BANNED_F66":
                    round(float(np.mean([(pr["deployed"][i] == y[i]) or (pr["txt"][i] == y[i])
                                         for i in range(len(y))])), 4),
                "oracle_pick_of_fused_txt_img_acc_BANNED_F66":
                    round(float(np.mean([(pr["deployed"][i] == y[i]) or (pr["txt"][i] == y[i])
                                         or (pr["img"][i] == y[i]) for i in range(len(y))])), 4)}
    O["stream_error_set_overlap"] = ov
    O["stream_means"] = {}
    for proto in ("valsel", "final"):
        for nm in ("deployed", "fused", "txt", "img"):
            O["stream_means"]["%s_%s_acc" % (proto, nm)] = round(
                st.mean([ov["seed%d_%s" % (s, proto)]["acc"][nm] for s in (0, 1, 2)]), 4)
            O["stream_means"]["%s_%s_mf1" % (proto, nm)] = round(
                st.mean([ov["seed%d_%s" % (s, proto)]["mf1"][nm] for s in (0, 1, 2)]), 4)

    # ---------------- (c) ceilings
    C = {}
    # C1/C2 threshold recalibration
    c12 = {}
    for s in (0, 1, 2):
        for proto in ("valsel", "final"):
            sp = heads[(s, proto)]["space"]
            vt = votes[(s, proto)]
            acc0 = float(np.mean((vt >= 0).astype(int) == y))
            bt, ba = best_thr(vt, y)                      # test-fitted = FORENSIC ORACLE
            vd, _, _, _ = knn_vote(sp["dev"]["fused"], sp["train"]["fused"], feats["train"]["lab"])
            dt, da = best_thr(vd, ydv)                    # dev-fitted = LEGAL
            pd_ = (vt >= dt).astype(int)
            c12["seed%d_%s" % (s, proto)] = {
                "acc_deployed": round(acc0, 4),
                "C1_test_fitted_thr_ORACLE": round(bt, 6),
                "C1_acc_ORACLE": round(ba, 4),
                "C1_gain_ORACLE": round(ba - acc0, 4),
                "C2_dev_fitted_thr": round(dt, 6),
                "C2_dev_acc": round(da, 4),
                "C2_test_acc": round(float(np.mean(pd_ == y)), 4),
                "C2_test_mf1": round(macro_f1(y, pd_), 4),
                "C2_gain": round(float(np.mean(pd_ == y)) - acc0, 4)}
    C["C1_C2_threshold"] = c12
    C["C1_mean_gain_ORACLE"] = {
        p: round(st.mean([c12["seed%d_%s" % (s, p)]["C1_gain_ORACLE"] for s in (0, 1, 2)]), 4)
        for p in ("valsel", "final")}
    C["C2_mean_gain_dev_fitted_LEGAL"] = {
        p: round(st.mean([c12["seed%d_%s" % (s, p)]["C2_gain"] for s in (0, 1, 2)]), 4)
        for p in ("valsel", "final")}

    # C3 memory-bank curation by TRAIN-only LOO disagreement (no test/dev fitting)
    c3 = {}
    for s in (0, 1, 2):
        for proto in ("valsel", "final"):
            sp = heads[(s, proto)]["space"]
            vloo, _, _, _ = knn_vote(sp["train"]["fused"], sp["train"]["fused"],
                                     feats["train"]["lab"], exclude_self=True)
            bad = np.where((vloo >= 0).astype(int) != ytr)[0]
            acc0 = float(np.mean((votes[(s, proto)] >= 0).astype(int) == y))
            row = {"n_train": int(len(ytr)), "n_loo_disagree": int(len(bad)),
                   "loo_disagree_rate": round(float(len(bad) / len(ytr)), 4),
                   "acc_deployed": round(acc0, 4)}
            for tag, drop in (("drop_loo_disagree", bad),
                              ("drop_random_same_n", np.random.RandomState(0).choice(
                                  len(ytr), len(bad), replace=False))):
                vp, _, _, _ = knn_vote(sp["test"]["fused"], sp["train"]["fused"],
                                       feats["train"]["lab"], drop_bank=drop)
                pp = (vp >= 0).astype(int)
                row[tag + "_acc"] = round(float(np.mean(pp == y)), 4)
                row[tag + "_mf1"] = round(macro_f1(y, pp), 4)
                row[tag + "_gain"] = round(float(np.mean(pp == y)) - acc0, 4)
            # oracle upper bound: drop every train row whose label disagrees with its
            # own kNN AND relabel-free -- i.e. pure deletion ceiling by search over k
            c3["seed%d_%s" % (s, proto)] = row
    C["C3_curation"] = c3
    C["C3_mean_gain"] = {p: round(st.mean([c3["seed%d_%s" % (s, p)]["drop_loo_disagree_gain"]
                                           for s in (0, 1, 2)]), 4) for p in ("valsel", "final")}
    C["C3_mean_gain_random_control"] = {
        p: round(st.mean([c3["seed%d_%s" % (s, p)]["drop_random_same_n_gain"]
                          for s in (0, 1, 2)]), 4) for p in ("valsel", "final")}
    O["ceilings"] = C

    # ---------------- (d) exemplars for the consensus 3/3 error set
    ex = {}
    proto = "final"
    nerr = cons[proto]
    hard = [v for v in test_ids if nerr[v] == 3]
    sp0 = heads[(0, proto)]["space"]
    prs = {}
    for nm in ("txt", "img"):
        v, _, _, _ = knn_vote(sp0["test"][nm], sp0["train"][nm], feats["train"]["lab"])
        prs[nm] = (v >= 0).astype(int)
    idx = {v: i for i, v in enumerate(test_ids)}
    for v in hard:
        i = idx[v]
        cr = covrow(v)
        ex[v] = {"label": int(y[i]),
                 "deployed_vote_seed0": round(float(votes[(0, proto)][i]), 4),
                 "mean_vote_3seed": round(float(np.mean([votes[(s, proto)][i] for s in (0, 1, 2)])), 4),
                 "head_txt_correct": int(prs["txt"][i] == y[i]),
                 "head_img_correct": int(prs["img"][i] == y[i]),
                 "n_words": cr["nw"], "duration_s": round(cr["dur"], 2) if cr["dur"] == cr["dur"] else None,
                 "span_frac": round(cr["span_frac"], 4) if cr["span_frac"] is not None else None,
                 "frames_in_span_of8": cr["frames_in_span"],
                 "target": tgt.get(v, ""),
                 "transcript_snippet": gt[v]["text"][:220].replace("\n", " ")}
    O["exemplars_3of3_final"] = ex

    # vote-margin distributions
    md = {}
    for proto in ("valsel", "final"):
        e = [i for i, v in enumerate(test_ids) if cons[proto][v] >= 2]
        c = [i for i, v in enumerate(test_ids) if cons[proto][v] == 0]
        for nm, ii in (("err_ge2", e), ("always_correct", c)):
            vals = [abs(float(np.mean([votes[(s, proto)][i] for s in (0, 1, 2)]))) for i in ii]
            md["%s_%s_abs_vote" % (proto, nm)] = {
                "n": len(vals), "median": round(float(np.median(vals)), 4),
                "mean": round(float(np.mean(vals)), 4)}
    O["vote_margin"] = md

    with open(args.out_json, "w") as f:
        json.dump(O, f, indent=2, default=str)
    print("wrote", args.out_json)


if __name__ == "__main__":
    main()
