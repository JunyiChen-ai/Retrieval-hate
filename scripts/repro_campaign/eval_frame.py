#!/usr/bin/env python
"""REPRO campaign — the one shared frame-level evaluator (freeze §2).

Every method is scored through this file; no method re-implements a metric.

Inputs a method must supply, per dataset:
  * a per-video score curve at the method's native rate, and that rate, and/or
  * a per-video list of intervals (start, end, score) in seconds.
Curves are broadcast piecewise-constant onto the canonical 4 fps grid (freeze §1)
and both gold and score are truncated to T = min(T_gt, T_feat).

Metrics (freeze §2):
  frame ROC-AUC and frame PR-AUC on the pooled frame set of the evaluated split;
  F1@tIoU in {0.3, 0.5, 0.7}, proposal-level, for interval-emitting methods only;
  positive base rate; oracle-normalised AP (AP - AP_rand) / (AP_broadcast - AP_rand),
  with both anchors recomputed on the *same* evaluated pool so the rescaling stays
  exact when a method is missing videos.

Missing / refused / unparseable videos are dropped from the pool, never
interpolated, and their count and frame share are reported.

CLI
  python scripts/repro_campaign/eval_frame.py --method qwen_grounding
  python scripts/repro_campaign/eval_frame.py --method imagebind --channels image,video,audio
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import (auc, average_precision_score,
                             precision_recall_curve, roc_auc_score)

ROOT = Path("/home/jehc223/Retrieval-hate")
sys.path.insert(0, str(ROOT / "scripts/r16_detbase"))
from eval_f1 import match_prf  # noqa: E402  (project's own greedy tIoU matcher)

FPS = 4.0
DATASETS = ["HateMM", "MHC", "MHC_zh", "HateClipSeg"]
TIOUS = (0.3, 0.5, 0.7)


# ------------------------------------------------------------------ gold ---
def load_gt(ds: str) -> dict:
    z = np.load(ROOT / f"data/gt/frame_gt_4fps/{ds}.npz", allow_pickle=True)
    out = {}
    for i, vid in enumerate(z["video_ids"]):
        out[str(vid)] = dict(
            y4=np.asarray(z["y4"][i], dtype=np.int8),
            spans=np.asarray(z["spans"][i], dtype=np.float64).reshape(-1, 2),
            duration=float(z["duration"][i]),
            split=str(z["split"][i]),
            y_video=int(z["y_video"][i]),
            n_spans=int(z["n_spans"][i]),
        )
    return out


HCS_CLASSES = ["normal", "hateful", "insulting", "sexual", "violence", "harm"]


def load_gt_hcs_class(cls: int) -> dict:
    """HateClipSeg gold restricted to one released class (freeze §4 class order).

    The frozen npz stores any-toxic and hateful-only; the other four classes are
    rebuilt here from the same released segments file, on the same 4 fps grid, so
    the six per-class query rows are scored against their own labels.
    """
    base = load_gt("HateClipSeg")
    seg = json.loads((ROOT / "data/gt/HateClipSeg/gold_segments.json").read_text())
    for vid, g in base.items():
        s = seg.get(vid)
        if s is None:
            continue
        T = len(g["y4"])
        y = np.zeros(T, dtype=np.int8)
        spans = []
        for a, b, multi in s["segments"]:
            if not multi[cls]:
                continue
            a = max(0.0, min(float(a), g["duration"]))
            b = max(0.0, min(float(b), g["duration"]))
            if b <= a:
                continue
            spans.append([a, b])
            i0 = int(np.ceil(a * FPS - 1e-9))
            i1 = int(np.ceil(b * FPS - 1e-9))
            y[max(0, i0):max(0, min(T, i1))] = 1
        # merge touching/overlapping spans so the interval metric sees one gold
        # interval per contiguous stretch, as the released annotation intends
        spans.sort()
        merged = []
        for a, b in spans:
            if merged and a <= merged[-1][1] + 1e-9:
                merged[-1][1] = max(merged[-1][1], b)
            else:
                merged.append([a, b])
        g["y4"] = y
        g["spans"] = np.asarray(merged, dtype=np.float64).reshape(-1, 2)
        g["n_spans"] = len(merged)
        g["y_video"] = int(len(merged) > 0)
    return base


def broadcast_to_4fps(curve: np.ndarray, native_rate: float, T: int) -> np.ndarray:
    """Piecewise-constant broadcast of a coarser native rate onto the 4 fps grid."""
    if native_rate == FPS:
        s = curve
    else:
        idx = np.floor(np.arange(T) / FPS * native_rate).astype(int)
        idx = np.clip(idx, 0, len(curve) - 1)
        s = curve[idx]
    if len(s) < T:
        s = np.concatenate([s, np.full(T - len(s), s[-1] if len(s) else 0.0)])
    return s[:T].astype(np.float64)


# --------------------------------------------------------------- metrics ---
def pooled_metrics(gt: dict, curves: dict, native_rate: float, split: str,
                   subset=None) -> dict:
    ys, ss, yv, ns_used, tpairs = [], [], [], 0, []
    for vid, g in gt.items():
        if split != "all" and g["split"] != split:
            continue
        if subset is not None and not subset(g):
            continue
        if vid not in curves:
            continue
        c = curves[vid]
        if c is None or len(c) == 0:
            continue
        T_gt = len(g["y4"])
        T_feat = int(np.ceil(len(c) * FPS / native_rate)) if native_rate != FPS else len(c)
        T = min(T_gt, T_feat)
        if T <= 0:
            continue
        ys.append(g["y4"][:T].astype(np.int8))
        ss.append(broadcast_to_4fps(c, native_rate, T))
        yv.append(np.full(T, g["y_video"], dtype=np.int8))
        tpairs.append((T_gt, T_feat))
        ns_used += 1
    if not ys:
        return dict(n_videos=0)
    y = np.concatenate(ys)
    s = np.concatenate(ss)
    v = np.concatenate(yv)
    # A method may leave individual frames unscored -- an LLM refusal, an
    # unparseable generation -- and marks them NaN.  Those frames are dropped
    # from the pool, never interpolated (freeze §14, MODEL_ASSETS_STATUS §3.11a
    # item 2), and the surviving share is reported as `coverage`.  Curves with
    # no NaN, which is every other front-end, are unaffected.
    n_all = len(y)
    ok = np.isfinite(s)
    cov = float(ok.mean())
    if not ok.all():
        y, s, v = y[ok], s[ok], v[ok]
    if len(y) == 0:
        return dict(n_videos=ns_used, n_frames=0, coverage=0.0)
    base = float(y.mean())
    ap = float(average_precision_score(y, s))
    roc = float(roc_auc_score(y, s)) if 0 < y.sum() < len(y) else float("nan")
    # Freeze §2 fixes PR-AUC = average precision.  LAVAD's own `src/eval.py`
    # (and URF-HVAA's, which is forked from it) instead reports
    # `auc(recall, precision)` -- the trapezoid rule over the PR curve, which is
    # a different and generally larger quantity.  Any third party who ported
    # those repos, LELA included, would have reported the trapezoid number, so
    # it is computed here purely so the §7 alignment can be checked against both
    # conventions.  It is never the headline and never replaces `frame_PR_AUC`.
    if 0 < y.sum() < len(y):
        pr, rc, _ = precision_recall_curve(y, s)
        pr_trapz = float(auc(rc, pr))
    else:
        pr_trapz = float("nan")
    ap_bc = float(average_precision_score(y, v)) if 0 < y.sum() < len(y) else float("nan")
    denom = ap_bc - base
    return dict(
        n_videos=ns_used, n_frames=int(len(y)), n_frames_in_pool=int(n_all),
        coverage=round(cov, 4), base_rate=round(base, 4),
        frame_ROC_AUC=round(roc, 4), frame_PR_AUC=round(ap, 4),
        frame_PR_AUC_trapz=round(pr_trapz, 4),
        AP_broadcast_pool=round(ap_bc, 4), AP_random_pool=round(base, 4),
        AP_norm=round((ap - base) / denom, 4) if denom > 1e-9 else None,
        n_offgrid_gt8=int(sum(1 for a, b in tpairs if abs(a - b) > 8)),
    )


def interval_metrics(gt: dict, intervals: dict, split: str, subset=None) -> dict:
    preds, golds = {}, {}
    for vid, g in gt.items():
        if split != "all" and g["split"] != split:
            continue
        if subset is not None and not subset(g):
            continue
        if vid not in intervals or intervals[vid] is None:
            continue
        golds[vid] = [tuple(x) for x in g["spans"]]
        preds[vid] = [tuple(x) for x in intervals[vid]]
    out = {}
    for t in TIOUS:
        r = match_prf(preds, golds, t)
        out[f"F1@{t}"] = round(r["F1"] / 100.0, 4)
        out[f"P@{t}"] = round(r["P"] / 100.0, 4)
        out[f"R@{t}"] = round(r["R"] / 100.0, 4)
    out["n_pred"] = sum(len(p) for p in preds.values())
    out["n_gold"] = sum(len(g) for g in golds.values())
    out["n_videos"] = len(preds)
    return out


def evaluate(ds: str, gt: dict, curves: dict, native_rate: float,
             intervals: dict | None, split: str, missing: list) -> dict:
    total = sum(1 for g in gt.values() if split == "all" or g["split"] == split)
    miss_in_split = [v for v in missing
                     if v in gt and (split == "all" or gt[v]["split"] == split)]
    res = dict(dataset=ds, split=split, native_rate=native_rate,
               n_videos_in_split=total,
               n_videos_missing=len(miss_in_split),
               missing_frac=round(len(miss_in_split) / max(total, 1), 4),
               missing_ids=sorted(miss_in_split)[:50])
    res["pooled"] = pooled_metrics(gt, curves, native_rate, split)
    if intervals is not None:
        res["intervals"] = interval_metrics(gt, intervals, split)
    if ds != "HateClipSeg":
        # Same convention as the ZS-CLIP section of REPRO_CAMPAIGN_RESULTS.md, so the
        # strata are comparable across methods: zero-span videos supply the negative
        # frames and therefore appear in both strata; without them a stratum is
        # single-class and its ROC/AP are undefined.
        for name, fn in [("single_span", lambda g: g["n_spans"] <= 1),
                         ("multi_span", lambda g: g["n_spans"] >= 2 or g["n_spans"] == 0)]:
            res[f"strat_{name}"] = pooled_metrics(gt, curves, native_rate, split, fn)
            if intervals is not None:
                res[f"strat_{name}_intervals"] = interval_metrics(gt, intervals, split, fn)
    return res


# ------------------------------------------------------- method front-ends ---
def qwen_curves(ds: str, qkey: str, gt: dict):
    """Predicted interval -> binary 0/1 frame curve at 4 fps."""
    path = ROOT / f"idea-stage/repro_qwen_ground/raw/qwen_{ds}_{qkey}.jsonl"
    last = {}
    with open(path) as fh:
        for line in fh:
            try:
                r = json.loads(line)
            except Exception:
                continue
            last[r["video_id"]] = r  # keep the last record per id (retries supersede)
    curves, intervals, missing, reasons = {}, {}, [], {}
    for vid, g in gt.items():
        r = last.get(vid)
        if r is None or r.get("span") is None:
            missing.append(vid)
            reasons[vid] = (r or {}).get("error") or ("unparsed" if r else "not_run")
            continue
        a, b = float(r["span"][0]), float(r["span"][1])
        if b < a:
            a, b = b, a
        D = g["duration"]
        a, b = max(0.0, min(a, D)), max(0.0, min(b, D))
        T = len(g["y4"])
        c = np.zeros(T, dtype=np.float64)
        i0 = int(np.ceil(a * FPS - 1e-9))
        i1 = int(np.ceil(b * FPS - 1e-9))
        c[max(0, i0):max(0, min(T, i1))] = 1.0
        curves[vid] = c
        intervals[vid] = [(a, b, 1.0)]
    return curves, intervals, missing, reasons


def curve_dir_front_end(ds: str, curve_dir: Path, variant: str, gt: dict):
    """Generic front-end for a method that writes one npz per video.

    `<curve_dir>/<DS>/<vid>.npz` must hold the per-variant curve under the key
    `<variant>` and its own native rate under `rate` (a scalar, in samples per
    second).  A per-video rate is allowed: the curve is broadcast onto the 4 fps
    grid here with exactly the function the pooled evaluator would have used, so
    the caller then passes `native_rate=4`.  Videos with no file, an empty curve
    or a non-finite curve are reported missing and dropped, never interpolated.

    Optional `<curve_dir>/<DS>_intervals_<variant>.json` supplies interval
    predictions `{vid: [[start, end, score], ...]}` for interval-emitting methods.
    """
    d = curve_dir / ds
    curves, missing, reasons = {}, [], {}
    for vid, g in gt.items():
        p = d / f"{vid}.npz"
        if not p.exists():
            missing.append(vid)
            reasons[vid] = "not_run"
            continue
        try:
            z = np.load(p, allow_pickle=False)
            c = np.asarray(z[variant], dtype=np.float64).reshape(-1)
            rate = float(z["rate"])
        except Exception as e:  # a truncated or key-less file is a failure, not a zero
            missing.append(vid)
            reasons[vid] = f"load:{type(e).__name__}"
            continue
        # A NaN marks one unscored frame (see `pooled_metrics`); a curve that is
        # NaN everywhere carries no score at all and the video is reported missing.
        if c.size == 0 or not np.isfinite(c).any() or rate <= 0:
            missing.append(vid)
            reasons[vid] = "empty_or_nonfinite"
            continue
        T_feat = int(np.ceil(c.size * FPS / rate))
        curves[vid] = broadcast_to_4fps(c, rate, T_feat)
    ivf = curve_dir / f"{ds}_intervals_{variant}.json"
    intervals = None
    if ivf.exists():
        raw = json.loads(ivf.read_text())
        intervals = {v: [tuple(x) for x in iv] for v, iv in raw.items()
                     if v in gt and iv is not None}
    return curves, intervals, missing, reasons


def imagebind_curves(ds: str, chan: str, gt: dict, text_emb: np.ndarray):
    d = ROOT / f"data/CLIP_Embedding/{ds}/imagebind_{chan}"
    curves, missing = {}, []
    # ImageBind's demo scores `embeddings[VISION] @ embeddings[TEXT].T`; the text
    # postprocessor already carries the learned logit scale (‖t‖ = 100), and the
    # vision postprocessor already unit-normalises.  Unit-normalising each modality
    # and keeping the raw text vector reproduces that product for VISION and puts
    # AUDIO (‖e‖ = 20 from its own scale) on the same temperature.  Softmax over the
    # pair is monotone in sim(hateful) - sim(normal), so the ranking metrics are
    # invariant to this choice either way.
    t = text_emb
    for vid in gt:
        p = d / f"{vid}.npy"
        if not p.exists():
            missing.append(vid)
            continue
        e = np.load(p).astype(np.float32)
        if e.size == 0:
            missing.append(vid)
            continue
        e = e / np.maximum(np.linalg.norm(e, axis=-1, keepdims=True), 1e-8)
        logits = e @ t.T
        m = logits.max(axis=1, keepdims=True)
        p_ = np.exp(logits - m)
        curves[vid] = (p_[:, 1] / p_.sum(axis=1)).astype(np.float64)
    return curves, missing


# ------------------------------------------------------------------- main ---
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", required=True,
                    choices=["qwen_grounding", "imagebind", "curves"])
    ap.add_argument("--curve-dir", default=None,
                    help="curves method: root holding <DS>/<vid>.npz")
    ap.add_argument("--variants", default="main",
                    help="curves method: comma-separated npz keys to score")
    ap.add_argument("--method-name", default="curves",
                    help="curves method: the name written into the result rows")
    ap.add_argument("--wave", type=int, default=1)
    ap.add_argument("--supervision", default="label-free")
    ap.add_argument("--datasets", default=",".join(DATASETS))
    ap.add_argument("--split", default="test")
    ap.add_argument("--channels", default="image,video,audio")
    ap.add_argument("--qkeys", default="main")
    ap.add_argument("--text-emb",
                    default=str(ROOT / "data/CLIP_Embedding/imagebind_text_normal_hateful.npy"))
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    results = []
    for ds in args.datasets.split(","):
        gt = load_gt(ds)
        if args.method == "qwen_grounding":
            for qk in args.qkeys.split(","):
                f = ROOT / f"idea-stage/repro_qwen_ground/raw/qwen_{ds}_{qk}.jsonl"
                if not f.exists():
                    continue
                # a per-class HateClipSeg query is scored against that class's own
                # frame labels; the `main` query is scored against any-toxic (§4).
                g_eval = gt
                if ds == "HateClipSeg" and len(qk) > 1 and qk[0] == "c" and qk[1].isdigit():
                    g_eval = load_gt_hcs_class(int(qk[1]))
                curves, intervals, missing, reasons = qwen_curves(ds, qk, g_eval)
                r = evaluate(ds, g_eval, curves, FPS, intervals, args.split, missing)
                r.update(method="Qwen2.5-VL-7B grounding", variant=f"query={qk}",
                         supervision="label-free", wave=0)
                from collections import Counter
                r["missing_reasons"] = dict(Counter(
                    reasons[v] for v in missing
                    if args.split in ("all", g_eval[v]["split"])))
                results.append(r)
        elif args.method == "curves":
            from collections import Counter
            cdir = Path(args.curve_dir)
            for var in args.variants.split(","):
                g_eval = gt
                if ds == "HateClipSeg" and len(var) > 1 and var[0] == "c" and var[1].isdigit():
                    g_eval = load_gt_hcs_class(int(var[1]))
                curves, intervals, missing, reasons = curve_dir_front_end(
                    ds, cdir, var, g_eval)
                if not curves:
                    continue
                r = evaluate(ds, g_eval, curves, FPS, intervals, args.split, missing)
                r.update(method=args.method_name, variant=var,
                         supervision=args.supervision, wave=args.wave)
                r["missing_reasons"] = dict(Counter(
                    reasons[v] for v in missing
                    if args.split in ("all", g_eval[v]["split"])))
                results.append(r)
        else:
            temb = np.load(args.text_emb)
            for ch in args.channels.split(","):
                curves, missing = imagebind_curves(ds, ch, gt, temb)
                rate = FPS if ch == "image" else 0.5
                r = evaluate(ds, gt, curves, rate, None, args.split, missing)
                r.update(method=f"ZS-ImageBind ({ch})", variant="base",
                         supervision="label-free", wave=0)
                results.append(r)

    tag = args.method if args.method != "curves" else args.method_name
    out = Path(args.out) if args.out else (
        ROOT / f"idea-stage/repro_campaign/eval_{tag}_{args.split}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=1))
    for r in results:
        p = r.get("pooled", {})
        print(f"{r['method']:<28} {r['variant']:<14} {r['dataset']:<12} "
              f"ROC={p.get('frame_ROC_AUC')} AP={p.get('frame_PR_AUC')} "
              f"APn={p.get('AP_norm')} base={p.get('base_rate')} "
              f"cov={p.get('coverage')} "
              f"n={p.get('n_videos')}/{r['n_videos_in_split']} miss={r['missing_frac']}"
              + (f" F1@.5={r['intervals']['F1@0.5']}" if "intervals" in r else ""))
    print(f"[written] {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
