"""Error analysis of the text pipeline (module-1 iterations 6 / 6b, README sections 12-12b; user request
2026-09-22): why the hate-tuned text encoder helps HateMM and hurts HateClipSeg, and where the pooled-AP
differences between the text configurations come from. Offline, no training; test split (user ruling:
test may be used for evaluation at any stage). One evaluator (hc.frame_metrics -> frame_eval_common).

A. Text score alone. Per-second classifier score as the localization score, on test: pooled AP / ROC /
   within (all videos), pooled on positive videos only, video-level ROC of the per-video max, and the
   fraction of seconds with p_asr > .5 on negative videos vs hate / non-hate seconds of positives.
   For both ASR sources of x_t ("asr" = data/ASR utterances, "chunks" = sentence-level Whisper chunks).
B. Linear probe on the frozen per-second text rows (bert, hate_chunks, bert_utterance, hate_roberta):
   logistic regression on the test videos, 5-fold grouped by video (out-of-fold predictions), same three
   metrics. Says which rows carry per-second localization information on each corpus.
C. Decomposition of the trained models' pooled AP (default best trial, xt_chunks, unified_chunks,
   text_feat_hate; three seeds; 8-call operating point scores_test_eoc_cap4_tau0.jsonl): AP as is; AP with
   oracle within-video ordering (each video keeps its own score values, re-assigned so that its hate
   seconds rank first); AP with oracle between-video separation (+10 on every second of positive videos,
   within-video order kept); video-level ROC of the per-video max.
D. Geometry of the rows: fraction of the total variance that is within-video, mean cosine between
   consecutive speech seconds, mean cosine of a row to its video mean.

Output: runs/20260910_online_query_within_it5/text_analysis/<corpus>.json and a printed table.
usage: python text_error_analysis.py [--corpus hatemm|hateclipseg] [--skip-probe]
"""
import argparse, json, os, sys
import numpy as np
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "src")); sys.path.insert(0, os.path.join(ROOT, "scripts", "reproduction_baselines"))
import hier_evidence_common as hc                       # noqa: E402
from hate_common import data as hdata                  # noqa: E402

D = os.path.join(ROOT, "runs", "20260910_online_query_within_it5")
OUT = os.path.join(D, "text_analysis")
BEST = {"hatemm": {234: 5, 2025: 10, 3407: 2}, "hateclipseg": {234: 11, 2025: 0, 3407: 12}}
ROW_SETS = ("bert", "hate_chunks", "bert_utterance", "hate_roberta")
ARMS = ("default", "xt_chunks", "unified_chunks", "text_feat_hate")


def r4(x):
    return None if x is None else round(float(x), 4)


def metrics(scores, gt, hate_ids):
    m = hc.frame_metrics(scores, gt, hate_ids)
    return {"ap": r4(m["pooled_ap"]), "roc": r4(m["pooled_roc"]), "within": r4(m["within_roc"]), "n": m["n_videos"]}


def video_roc(scores, labels):
    from sklearn.metrics import roc_auc_score
    ids = sorted(scores)
    y = np.array([int(labels[v]) for v in ids]); s = np.array([float(np.max(scores[v])) for v in ids])
    return r4(roc_auc_score(y, s)) if 0 < y.sum() < len(y) else None


def oracle_within(scores, gt):
    out = {}
    for v, s in scores.items():
        s = np.asarray(s, float); y = np.asarray(gt[v]) > 0
        vals = np.sort(s)[::-1]; o = np.empty_like(s)
        idx = np.concatenate([np.flatnonzero(y), np.flatnonzero(~y)])
        o[idx] = vals
        out[v] = o
    return out


def oracle_between(scores, labels):
    return {v: np.asarray(s, float) + (10.0 if int(labels[v]) else 0.0) for v, s in scores.items()}


def read_scores(path):
    out = {}
    with open(path) as fh:
        for line in fh:
            d = json.loads(line); out[d["video_id"]] = np.asarray(d["score_av"], float)
    return out


def part_a(corpus, test_ids, gt, labels, hate_ids, train_ids):
    res = {}
    for src in ("asr", "chunks"):
        centre = hc.text_centre(corpus, train_ids, src)
        arrs = {v: hc.load_text_hate(corpus, v, src) for v in test_ids}
        xt = {}; pa = {}; po = {}
        for v in test_ids:
            a = arrs[v]; T = len(gt[v])
            x = hc.text_logit_seconds(a, centre) if a is not None else None
            xt[v] = np.zeros(T) if x is None else np.asarray(x, float)[:T]
            pa[v] = np.zeros(T) if a is None else np.nan_to_num(np.asarray(a["p_asr"], float)[:T], nan=0.0)
            po[v] = np.zeros(T) if a is None else np.nan_to_num(np.asarray(a["p_ocr"], float)[:T], nan=0.0)
        pos = {v: xt[v] for v in test_ids if v in hate_ids}
        neg_frac = []; pos_hate = []; pos_nonhate = []; cov = []
        for v in test_ids:
            a = arrs[v]
            if a is None:
                cov.append(0.0); continue
            p = np.asarray(a["p_asr"], float)[:len(gt[v])]; ok = np.isfinite(p); cov.append(ok.mean())
            y = np.asarray(gt[v]) > 0
            if not ok.any():
                continue
            if v in hate_ids:
                if (ok & y).any(): pos_hate.append((p[ok & y] > .5).mean())
                if (ok & ~y).any(): pos_nonhate.append((p[ok & ~y] > .5).mean())
            else:
                neg_frac.append((p[ok] > .5).mean())
        res[src] = {"centre": r4(centre), "coverage_mean": r4(np.mean(cov)),
                    "x_t": metrics(xt, gt, hate_ids), "x_t_positives_only": metrics(pos, gt, hate_ids),
                    "p_asr": metrics(pa, gt, hate_ids), "p_ocr": metrics(po, gt, hate_ids),
                    "video_roc_max_x_t": video_roc(xt, labels),
                    "frac_p_asr_gt_half": {"neg_videos": r4(np.mean(neg_frac)), "pos_hate_seconds": r4(np.mean(pos_hate)),
                                           "pos_nonhate_seconds": r4(np.mean(pos_nonhate))}}
    return res


def part_b(corpus, test_ids, gt, hate_ids, seed=0):
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import GroupKFold
    res = {}
    for src in ROW_SETS:
        X = []; y = []; g = []; lens = {}
        for i, v in enumerate(test_ids):
            r = np.load(hc.text_path(corpus, v, src)).astype(np.float32); T = len(gt[v])
            X.append(r[:T]); y.append(np.asarray(gt[v])[:T] > 0); g += [i] * T; lens[v] = T
        X = np.concatenate(X); y = np.concatenate(y).astype(int); g = np.asarray(g)
        oof = np.zeros(len(y))
        for tr, te in GroupKFold(5).split(X, y, g):
            sc = StandardScaler().fit(X[tr])
            clf = LogisticRegression(C=0.1, max_iter=2000).fit(sc.transform(X[tr]), y[tr])
            oof[te] = clf.predict_proba(sc.transform(X[te]))[:, 1]
        scores = {}; k = 0
        for v in test_ids:
            scores[v] = oof[k:k + lens[v]]; k += lens[v]
        res[src] = metrics(scores, gt, hate_ids)
        res[src]["positives_only"] = metrics({v: scores[v] for v in test_ids if v in hate_ids}, gt, hate_ids)
    return res


def part_c(corpus, gt, labels, hate_ids):
    res = {}
    for arm in ARMS:
        per = []
        for s, b in BEST[corpus].items():
            d = os.path.join(D, corpus, "seed%d" % s, "trial%d" % b) if arm == "default" else os.path.join(D, "diag", corpus, "seed%d" % s, arm)
            p = os.path.join(d, "scores_test_eoc_cap4_tau0.jsonl")
            if not os.path.exists(p):
                continue
            sc = read_scores(p); sc = {v: sc[v][:len(gt[v])] for v in sc if v in gt}
            m = metrics(sc, gt, hate_ids)
            per.append({"seed": s, "ap": m["ap"], "roc": m["roc"], "within": m["within"],
                        "ap_oracle_within": metrics(oracle_within(sc, gt), gt, hate_ids)["ap"],
                        "ap_oracle_between": metrics(oracle_between(sc, labels), gt, hate_ids)["ap"],
                        "video_roc_max": video_roc(sc, labels),
                        "ap_positives_only": metrics({v: sc[v] for v in sc if v in hate_ids}, gt, hate_ids)["ap"]})
        if per:
            keys = [k for k in per[0] if k != "seed"]
            res[arm] = {"n_seeds": len(per), "mean": {k: r4(np.mean([x[k] for x in per if x[k] is not None])) for k in keys}, "per_seed": per}
    return res


def part_d(corpus, test_ids):
    res = {}
    for src in ROW_SETS:
        within = []; total_rows = []; cons = []; tomean = []
        for v in test_ids:
            r = np.load(hc.text_path(corpus, v, src)).astype(np.float64)
            m = np.abs(r).sum(1) > 0
            if m.sum() < 2:
                continue
            x = r[m]; total_rows.append(x)
            within.append(x.var(0).sum() * len(x))
            mu = x.mean(0); n = np.linalg.norm(x, axis=1); nm = np.linalg.norm(mu)
            tomean.append((x @ mu / (n * nm + 1e-9)).mean())
            idx = np.flatnonzero(m); a = r[idx[:-1]]; b = r[idx[1:]]
            keep = (idx[1:] - idx[:-1]) == 1
            if keep.any():
                a = a[keep]; b = b[keep]
                cons.append(((a * b).sum(1) / (np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1) + 1e-9)).mean())
        allx = np.concatenate(total_rows)
        tot = allx.var(0).sum() * len(allx)
        res[src] = {"within_video_variance_fraction": r4(sum(within) / tot), "cos_consecutive_seconds": r4(np.mean(cons)),
                    "cos_to_video_mean": r4(np.mean(tomean)), "row_norm_mean": r4(np.linalg.norm(allx, axis=1).mean())}
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default=None, choices=("hatemm", "hateclipseg"))
    ap.add_argument("--skip-probe", action="store_true")
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    for corpus in ([a.corpus] if a.corpus else ("hatemm", "hateclipseg")):
        gt = hdata.gt_arrays(corpus, "test"); labels = hdata.load_labels(corpus)
        test_ids = sorted(gt); hate_ids = {v for v in test_ids if int(labels[v])}
        train_ids = hc.usable(corpus, hdata.load_split(corpus, "train"))
        out = {"corpus": corpus, "n_test": len(test_ids), "n_pos": len(hate_ids)}
        out["A_text_score_alone"] = part_a(corpus, test_ids, gt, labels, hate_ids, train_ids)
        if not a.skip_probe:
            out["B_linear_probe"] = part_b(corpus, test_ids, gt, hate_ids)
        out["C_model_decomposition"] = part_c(corpus, gt, labels, hate_ids)
        out["D_row_geometry"] = part_d(corpus, test_ids)
        with open(os.path.join(OUT, corpus + ".json"), "w") as fh:
            json.dump(out, fh, indent=1)
        print("==", corpus, "test", len(test_ids), "positives", len(hate_ids))
        for src, r in out["A_text_score_alone"].items():
            print("  A %-6s coverage %.3f | x_t AP/ROC/within %s / %s / %s | positives-only AP %s | p_asr %s / %s / %s | video ROC(max x_t) %s | p_asr>.5: neg %s, pos hate %s, pos non-hate %s"
                  % (src, r["coverage_mean"], r["x_t"]["ap"], r["x_t"]["roc"], r["x_t"]["within"], r["x_t_positives_only"]["ap"],
                     r["p_asr"]["ap"], r["p_asr"]["roc"], r["p_asr"]["within"], r["video_roc_max_x_t"],
                     r["frac_p_asr_gt_half"]["neg_videos"], r["frac_p_asr_gt_half"]["pos_hate_seconds"], r["frac_p_asr_gt_half"]["pos_nonhate_seconds"]))
        for src, r in out.get("B_linear_probe", {}).items():
            print("  B probe %-14s AP/ROC/within %s / %s / %s | positives-only AP %s" % (src, r["ap"], r["roc"], r["within"], r["positives_only"]["ap"]))
        for arm, r in out["C_model_decomposition"].items():
            m = r["mean"]
            print("  C %-14s n=%d AP %s ROC %s within %s | AP oracle-within %s (+%.3f) | AP oracle-between %s (+%.3f) | video ROC(max) %s | positives-only AP %s"
                  % (arm, r["n_seeds"], m["ap"], m["roc"], m["within"], m["ap_oracle_within"], m["ap_oracle_within"] - m["ap"],
                     m["ap_oracle_between"], m["ap_oracle_between"] - m["ap"], m["video_roc_max"], m["ap_positives_only"]))
        for src, r in out["D_row_geometry"].items():
            print("  D %-14s within-video variance fraction %s | cos consecutive %s | cos to video mean %s | norm %s"
                  % (src, r["within_video_variance_fraction"], r["cos_consecutive_seconds"], r["cos_to_video_mean"], r["row_norm_mean"]))


if __name__ == "__main__":
    main()
