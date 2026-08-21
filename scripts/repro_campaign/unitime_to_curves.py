#!/usr/bin/env python
"""UniTime JSONL -> per-video curves + intervals for the shared evaluator.

UniTime does not emit a per-frame saliency score.  It emits, per (video, query):
  * `pred_relevant_windows` — the answer window(s) in seconds.  Rasterised to a
    binary 0/1 curve on the 4 fps grid (`window` variant) and passed through as
    intervals so F1@tIoU is defined.
  * `pred_relevant_windows_mr_seg` — present only for videos longer than
    `nf_short`, where the pipeline first runs a coarse segment-retrieval pass.
    It is the list of clip timestamps that pass kept, i.e. the closest thing the
    method has to a saliency read-out.  Rasterised (`seg` variant) as the union
    of `[t, t + s)` over the returned timestamps `t`, where `s` is the spacing
    between consecutive returned timestamps (the clip length the pass worked at);
    with fewer than two timestamps the whole span from the first timestamp to the
    end of the answer window is marked.  For a short video, which never runs that
    pass, `seg` is by construction identical to `window`, and the count of such
    videos is reported.

A record with `window = null` (refusal, unparseable generation, decode failure,
missing file) writes no npz at all, so the evaluator reports it missing and drops
it — never interpolated (freeze §14).
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np

ROOT = Path("/home/jehc223/Retrieval-hate")
FPS = 4.0


def raster(spans, D, T):
    c = np.zeros(T, dtype=np.float32)
    for a, b in spans:
        if b is None or a is None:
            continue
        a, b = float(a), float(b)
        if b < a:
            a, b = b, a
        a, b = max(0.0, min(a, D)), max(0.0, min(b, D))
        i0 = int(np.ceil(a * FPS - 1e-9))
        i1 = int(np.ceil(b * FPS - 1e-9))
        c[max(0, i0):max(0, min(T, i1))] = 1.0
    return c


HCS6 = ["c0_normal", "c1_hateful", "c2_insulting", "c3_sexual", "c4_violence",
        "c5_harm"]


def merge_hcs6(out: Path) -> None:
    """Repack the six HateClipSeg per-class query roots into one npz per video.

    The evaluator decides which released class a row is scored against from the
    *variant name* (`load_gt_hcs_class`, triggered by a `c<digit>` prefix), not
    from the directory.  The per-query roots this converter writes hold the
    variant keys `window` / `seg`, which would silently score all six class
    queries against the any-toxic gold.  So the class curves are repacked here
    into `curves/hcs6/HateClipSeg/<vid>.npz` with one key per class, which is
    exactly the layout the LaGoVAD 6-class appendix already used.

    Only the answer window is carried over: `mr_seg` is a property of the video,
    not of the query, so a per-class `seg` row would repeat one curve six times.
    A video is written only if all six class queries answered for it, so no cell
    of the appendix is computed on a different video set than its neighbours.
    """
    dst = out / "hcs6" / "HateClipSeg"
    dst.mkdir(parents=True, exist_ok=True)
    srcs = {c: out / c / "HateClipSeg" for c in HCS6}
    absent = [c for c, d in srcs.items() if not d.exists()]
    if absent:
        print(f"[hcs6] skipped, no curves for {absent}", flush=True)
        return
    ids = set.intersection(*({p.stem for p in d.glob("*.npz")} for d in srcs.values()))
    for vid in sorted(ids):
        arrays, T = {}, None
        for c in HCS6:
            z = np.load(srcs[c] / f"{vid}.npz", allow_pickle=False)
            arrays[c] = np.asarray(z["window"], dtype=np.float32)
            T = len(arrays[c]) if T is None else T
        if any(len(a) != T for a in arrays.values()):
            print(f"[hcs6] {vid}: class curves disagree on length, skipped", flush=True)
            continue
        p = dst / f"{vid}.npz"
        tmp = p.with_name(p.name + ".tmp")
        with open(tmp, "wb") as fh:
            np.savez(fh, rate=np.float32(FPS), **arrays)
        os.replace(tmp, p)
    print(f"[hcs6] videos={len(ids)} -> {dst}", flush=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-dir", default=str(ROOT / "idea-stage/repro_unitime/raw"))
    ap.add_argument("--out-dir", default=str(ROOT / "idea-stage/repro_unitime/curves"))
    ap.add_argument("--datasets", default="HateMM,MHC,MHC_zh,HateClipSeg")
    ap.add_argument("--hcs6-merge", action="store_true",
                    help="also repack the six HateClipSeg class queries into "
                         "curves/hcs6 with one variant key per class")
    args = ap.parse_args()

    raw, out = Path(args.raw_dir), Path(args.out_dir)
    for ds in args.datasets.split(","):
        z = np.load(ROOT / f"data/gt/frame_gt_4fps/{ds}.npz", allow_pickle=True)
        dur = {str(v): float(z["duration"][i]) for i, v in enumerate(z["video_ids"])}
        Tg = {str(v): len(np.asarray(z["y4"][i])) for i, v in enumerate(z["video_ids"])}
        for f in sorted(raw.glob(f"unitime_{ds}_*.jsonl")):
            qk = f.stem[len(f"unitime_{ds}_"):]
            last = {}
            for line in f.read_text().splitlines():
                try:
                    r = json.loads(line)
                except Exception:
                    continue
                last[r["video_id"]] = r
            iv_out, n_ok, n_seg = {}, 0, 0
            qout = out / qk          # one curve root per query, so queries never overwrite
            (qout / ds).mkdir(parents=True, exist_ok=True)
            for vid, r in last.items():
                w = r.get("window")
                if not w or vid not in dur:
                    continue
                D, T = dur[vid], Tg[vid]
                spans = [(a, b) for a, b in w if a is not None and b is not None
                         and not (a == -1 and b == -1)]
                if not spans:
                    continue
                cw = raster(spans, D, T)
                seg = r.get("mr_seg")
                if seg:
                    ts = sorted({float(t) for lvl in seg for t in lvl if t != -1})
                    if len(ts) >= 2:
                        s = min(b - a for a, b in zip(ts, ts[1:]))
                        segspans = [(t, t + s) for t in ts]
                    else:
                        segspans = [(ts[0], max(b for _a, b in spans))] if ts else spans
                    cs = raster(segspans, D, T)
                    n_seg += 1
                else:
                    cs = cw
                p = qout / ds / f"{vid}.npz"
                tmp = p.with_name(p.name + ".tmp")
                with open(tmp, "wb") as fh:
                    np.savez(fh, window=cw, seg=cs, rate=np.float32(FPS))
                os.replace(tmp, p)
                iv_out[vid] = [[float(a), float(b), 1.0] for a, b in spans]
                n_ok += 1
            for var in ("window", "seg"):
                (qout / f"{ds}_intervals_{var}.json").write_text(json.dumps(iv_out))
            print(f"{ds}/{qk}: videos={n_ok} with_mr_seg={n_seg} "
                  f"records={len(last)}", flush=True)
            # Refuse to exit 0 on a silent truncation.  A raw file that holds
            # records but yields (almost) no curves means the run died partway
            # and the conversion is quietly reporting success on nothing -- the
            # failure mode that let the LAVAD chain finish rc=0 with 5 curves
            # for one dataset (campaign ruling 2026-08-19).
            if last and n_ok < 0.5 * len(last):
                raise SystemExit(
                    f"[FAIL] {ds}/{qk}: {len(last)} raw records but only {n_ok} "
                    f"curves. Refusing to report success -- inspect the run "
                    f"before evaluating.")
    if args.hcs6_merge:
        merge_hcs6(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
