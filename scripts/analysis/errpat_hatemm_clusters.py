#!/usr/bin/env python
"""
errpat_hatemm_clusters.py -- third forensic pass: named error clusters,
transcript-volume stratification, empty-transcript behaviour, and two extra
LEGAL calibration ceilings (dev-fitted by macro-F1; class-prior threshold).

Cluster definitions are DESCRIPTIVE (chosen after inspecting exemplars in
errpat_hatemm_ceilings_OUT.json), not pre-registered. No selection or tuning
decision is taken from them.  CPU only, proxy artifacts (F78).
"""
import argparse, csv, glob, json, os, pickle, re, statistics as st, sys
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

# Minimal forensic slur/hate-lexicon probe. Purpose: separate "the transcript
# literally contains a slur" from "the transcript does not", to test whether the
# text-dominant model's false positives are slur-triggered. Substring match on
# lowercased transcript; documented here so the count is reproducible.
SLUR_RE = re.compile(
    r"\b(nigger|nigga|nicker|niger|coon|jigaboo|spic|wetback|kike|jewburg|"
    r"heeb|yid|raghead|towelhead|paki|chink|gook|tranny|faggot|fag|dyke|"
    r"retard|subhuman|gorilla|monkey|ape)\b", re.I)


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


def knn_vote(q, b, blab, topk=TOPK):
    qn = torch.nn.functional.normalize(q.double(), p=2, dim=1)
    bn = torch.nn.functional.normalize(b.double(), p=2, dim=1)
    D, I = torch.topk(qn @ bn.T, topk, dim=1)
    D = D.numpy(); L = blab.numpy()[I.numpy()]
    w = np.arange(1, topk + 1)[::-1].astype(float)
    return ((L * 2 - 1) * D * w).sum(1) / w.sum()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--proxy_root", required=True)
    ap.add_argument("--out_json", default=os.path.join(REPO, "scripts/analysis/errpat_hatemm_clusters_OUT.json"))
    a = ap.parse_args()
    O = {"meta": {"nature": "FORENSIC PROXY third pass", "cpu_only": True,
                  "slur_lexicon_pattern": SLUR_RE.pattern,
                  "cluster_defs_are": "DESCRIPTIVE, post-hoc (not pre-registered)",
                  "inputs": {"proxy_root": a.proxy_root, "gt": GT.format("test"),
                             "spans": SPANS, "annot": ANNOT,
                             "features": [FEAT.format(x) for x in ("train", "dev_seen", "test_seen")]}}}

    dirs = {s: sorted(glob.glob(os.path.join(a.proxy_root, "Retrieval/HateMM/RAC_errpat_proxy",
                                             "*seed%d*" % s)))[0] for s in (0, 1, 2)}
    EPOCH = {0: {"valsel": 25, "final": 29}, 1: {"valsel": 15, "final": 29},
             2: {"valsel": 29, "final": 29}}

    gt = {}
    for sp in ("train", "val", "test"):
        for l in open(GT.format(sp)):
            r = json.loads(l); gt[r["id"]] = {"label": int(r["label"]), "text": r.get("text", "")}
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
    ydv = feats["dev"]["lab"].numpy()

    # deployed votes from the saved logging dicts (authoritative per-item source)
    votes = {}
    for s in (0, 1, 2):
        for proto in ("valsel", "final"):
            ld = pickle.load(open(os.path.join(
                dirs[s], "testepoch_%d_retrieval_logging_dict.pkl" % EPOCH[s][proto]), "rb"))["logging_dict"]
            w = np.arange(1, TOPK + 1)[::-1].astype(float)
            votes[(s, proto)] = np.array([
                (np.asarray(ld[v]["retrieved_label"], float) * 2 - 1)
                * np.asarray([float(x) for x in ld[v]["retrieved_scores"]]) @ w / w.sum()
                for v in test_ids])

    # single-stream head-space predictions (seed-averaged correctness)
    strm = {"txt": np.zeros((3, len(test_ids))), "img": np.zeros((3, len(test_ids)))}
    dev_fused, test_fused = {}, {}
    for s in (0, 1, 2):
        ck = sorted(glob.glob(os.path.join(glob.escape(dirs[s]), "ckpt",
                                           "epoch_model_%d_*.pt" % EPOCH[s]["final"])))[0]
        m = classifier_hateClipper(3584, 3584, 3, 1024, 1024, "align",
                                   dropout=[0.2, 0.4, 0.1], batch_norm=False, args=A())
        m.load_state_dict(torch.load(ck, map_location="cpu", weights_only=False)); m.eval()
        sp = {}
        with torch.no_grad():
            for tag in ("train", "dev", "test"):
                ip = torch.nn.functional.normalize(m.img_proj(feats[tag]["img"]), p=2, dim=1)
                tp = torch.nn.functional.normalize(m.text_proj(feats[tag]["txt"]), p=2, dim=1)
                sp[tag] = {"img": ip, "txt": tp, "fused": m.mlp[:-2](torch.mul(ip, tp))}
        for nm in ("txt", "img"):
            strm[nm][s] = (knn_vote(sp["test"][nm], sp["train"][nm], feats["train"]["lab"]) >= 0)
        dev_fused[s] = knn_vote(sp["dev"]["fused"], sp["train"]["fused"], feats["train"]["lab"])
        test_fused[s] = knn_vote(sp["test"]["fused"], sp["train"]["fused"], feats["train"]["lab"])

    # ---------- covariates
    def cov(v):
        sp = spans.get(v, {}); dur = float(sp.get("duration", float("nan")))
        sg = sp.get("spans", []) or []
        ss = sum(max(0.0, b - aa) for aa, b in sg)
        nw = len(gt[v]["text"].split())
        fis = None
        if dur == dur and dur > 0 and sg:
            ts = np.linspace(0.0, dur, 8)
            fis = int(sum(1 for t in ts if any(x <= t <= b for x, b in sg)))
        return {"dur": dur, "span_frac": (ss / dur) if (dur == dur and dur > 0) else 0.0,
                "nw": nw, "fis": fis, "slur": bool(SLUR_RE.search(gt[v]["text"] or ""))}

    C = {v: cov(v) for v in test_ids}

    # ---------- empty-transcript behaviour
    emp = [i for i, v in enumerate(test_ids) if C[v]["nw"] <= 1]
    et = {}
    for proto in ("valsel", "final"):
        pr = np.stack([(votes[(s, proto)] >= 0).astype(int) for s in (0, 1, 2)])
        et[proto] = {
            "n_empty_transcript": len(emp),
            "n_empty_hate": int(sum(1 for i in emp if y[i] == 1)),
            "n_empty_nonhate": int(sum(1 for i in emp if y[i] == 0)),
            "pred_hate_count_over_3seeds": int(pr[:, emp].sum()),
            "n_empty_hate_caught_any_seed": int(sum(1 for i in emp if y[i] == 1 and pr[:, i].max() == 1)),
            "n_empty_nonhate_flagged_any_seed": int(sum(1 for i in emp if y[i] == 0 and pr[:, i].max() == 1)),
            "empty_hate_ids": [test_ids[i] for i in emp if y[i] == 1]}
    O["empty_transcript_behaviour"] = et

    # ---------- accuracy stratified by transcript volume x class
    bins = [(0, 1), (2, 50), (51, 150), (151, 400), (401, 10 ** 9)]
    strat = {}
    for proto in ("valsel", "final"):
        pr = np.stack([(votes[(s, proto)] >= 0).astype(int) for s in (0, 1, 2)])
        blk = {}
        for lo, hi in bins:
            for cls in (1, 0):
                ii = [i for i, v in enumerate(test_ids) if lo <= C[v]["nw"] <= hi and y[i] == cls]
                if not ii:
                    continue
                accs = [float(np.mean(pr[s][ii] == y[ii])) for s in range(3)]
                blk["words_%d_%d_%s" % (lo, min(hi, 999999), "hate" if cls else "nonhate")] = {
                    "n": len(ii), "acc_3seed_mean": round(st.mean(accs), 4),
                    "per_seed": [round(x, 4) for x in accs]}
        strat[proto] = blk
    O["stratified_by_transcript_volume"] = strat

    # ---------- named clusters on the consensus (>=2/3 seeds wrong) error set
    def clusters(proto):
        pr = np.stack([(votes[(s, proto)] >= 0).astype(int) for s in (0, 1, 2)])
        nerr = np.array([int(sum(pr[s][i] != y[i] for s in range(3))) for i in range(len(y))])
        E = [i for i in range(len(y)) if nerr[i] >= 2]
        out = {}
        for i in E:
            v = test_ids[i]; c = C[v]
            if y[i] == 1:
                if c["nw"] <= 25:
                    k = "FN1_speech_poor_visual_hate"
                elif (c["span_frac"] < 0.25) or (c["fis"] is not None and c["fis"] <= 1):
                    k = "FN2_needle_hate_diluted_pool"
                else:
                    k = "FN3_talky_hate_text_stream_miss"
            else:
                if c["slur"]:
                    k = "FP1_slur_bearing_nonhate"
                elif c["nw"] >= 100:
                    k = "FP2_longform_talky_nonhate"
                else:
                    k = "FP3_residual_nonhate"
            out.setdefault(k, []).append(v)
        # per-cluster diagnostics
        rep = {}
        for k, ids in out.items():
            ii = [test_ids.index(v) for v in ids]
            rep[k] = {
                "n": len(ids), "pct_of_errors": round(100.0 * len(ids) / len(E), 1),
                "ids": ids,
                "median_words": int(np.median([C[v]["nw"] for v in ids])),
                "median_duration_s": round(float(np.median([C[v]["dur"] for v in ids])), 2),
                "median_span_frac": round(float(np.median([C[v]["span_frac"] for v in ids])), 4),
                "n_wrong_3of3": int(sum(1 for i in ii if nerr[i] == 3)),
                "n_txt_stream_correct": int(sum(1 for i in ii if strm["txt"][:, i].mean() >= 0.5
                                                and (strm["txt"][0, i] == y[i]))),
                "n_img_stream_correct_all3": int(sum(1 for i in ii if all(strm["img"][s, i] == y[i]
                                                                          for s in range(3)))),
                "n_img_stream_correct_any": int(sum(1 for i in ii if any(strm["img"][s, i] == y[i]
                                                                         for s in range(3)))),
                "mean_abs_vote_3seed": round(float(np.mean(
                    [abs(np.mean([votes[(s, proto)][i] for s in (0, 1, 2)])) for i in ii])), 4),
                "mean_top20_purity_toward_true": None}
        return {"n_errors": len(E), "clusters": rep}

    O["clusters"] = {p: clusters(p) for p in ("valsel", "final")}

    # top-20 purity per cluster (from the saved logging dicts)
    for proto in ("valsel", "final"):
        for k, blk in O["clusters"][proto]["clusters"].items():
            pu = []
            for v in blk["ids"]:
                i = test_ids.index(v)
                for s in (0, 1, 2):
                    ld = pickle.load(open(os.path.join(
                        dirs[s], "testepoch_%d_retrieval_logging_dict.pkl" % EPOCH[s][proto]), "rb"))["logging_dict"]
                    pu.append(float(np.mean(np.array(ld[v]["retrieved_label"]) == y[i])))
            blk["mean_top20_purity_toward_true"] = round(float(np.mean(pu)), 4)

    # ---------- extra LEGAL calibration ceilings on dev
    def bt(v, yy, obj):
        cand = np.unique(np.concatenate([[-1e9, 0.0, 1e9], v - 1e-12, v + 1e-12]))
        sc = np.array([(np.mean(((v >= t).astype(int)) == yy) if obj == "acc"
                        else macro_f1(yy, (v >= t).astype(int))) for t in cand])
        i = int(np.argmax(sc)); return float(cand[i]), float(sc[i])
    cal = {}
    for s in (0, 1, 2):
        acc0 = float(np.mean((test_fused[s] >= 0).astype(int) == y))
        mf0 = macro_f1(y, (test_fused[s] >= 0).astype(int))
        row = {"acc_deployed": round(acc0, 4), "mf1_deployed": round(mf0, 4)}
        for obj in ("acc", "mf1"):
            t, dsc = bt(dev_fused[s], ydv, obj)
            p = (test_fused[s] >= t).astype(int)
            row["dev_%s_thr" % obj] = round(t, 6)
            row["dev_%s_devscore" % obj] = round(dsc, 4)
            row["dev_%s_test_acc" % obj] = round(float(np.mean(p == y)), 4)
            row["dev_%s_test_mf1" % obj] = round(macro_f1(y, p), 4)
            row["dev_%s_gain_acc" % obj] = round(float(np.mean(p == y)) - acc0, 4)
            row["dev_%s_gain_mf1" % obj] = round(macro_f1(y, p) - mf0, 4)
        cal["seed%d" % s] = row
    cal["mean_gain"] = {
        "dev_acc_fitted_acc": round(st.mean([cal["seed%d" % s]["dev_acc_gain_acc"] for s in (0, 1, 2)]), 4),
        "dev_acc_fitted_mf1": round(st.mean([cal["seed%d" % s]["dev_acc_gain_mf1"] for s in (0, 1, 2)]), 4),
        "dev_mf1_fitted_acc": round(st.mean([cal["seed%d" % s]["dev_mf1_gain_acc"] for s in (0, 1, 2)]), 4),
        "dev_mf1_fitted_mf1": round(st.mean([cal["seed%d" % s]["dev_mf1_gain_mf1"] for s in (0, 1, 2)]), 4)}
    O["legal_calibration_final_epoch"] = cal

    with open(a.out_json, "w") as f:
        json.dump(O, f, indent=2, default=str)
    print("wrote", a.out_json)


if __name__ == "__main__":
    main()
