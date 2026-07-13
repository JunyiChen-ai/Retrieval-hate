"""SAV (C2) F-G0(b) two-tier reproduction guard (Rev-2 R2, MANDATORY pre-F-G1).

Authority: research-wiki/experiments/exp-sav-f0.md (Rev-2a, APPROVED), §4 F-G0(b).

PRIMARY (feature-level, DECIDING): min per-video cosine between the FRESH-forward pooled
read-out (this campaign's extraction) and the BANKED enc3s cache, for BOTH img_feats and
text_feats, over EVERY train+val video of the gated dataset. Pass iff min cosine >= 0.999.

SECONDARY (probe-level, CONFIRMATORY ONLY): a matched-capacity probe over the fresh pooled
img feature must land within +/-0.010 val acc of the same probe over the cached pooled img
feature. Flip-quantization fact (stated up front): on the 80-sample MHC-EN val one flip =
0.0125 > 0.010, so +/-0.010 permits ZERO flips and CAN trip on benign bf16 nondeterminism.
Therefore: PRIMARY pass + SECONDARY trip => guard PASSES (discrepancy recorded); PRIMARY
fail => guard FAILS regardless of the secondary. No post-hoc tolerance amendment.

Zero-guard videos (a decode-failed video is a cached zero vector, e.g. HateMM/train has 1):
matched zero-in-both = degenerate, excluded from the min statistic and logged; zero-in-one-
only = a decode DRIFT and fails the primary. Fail-closed on any missing cache / id.
"""

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import sav_f0_common as C  # noqa: E402
from sav_f0_probe import fit_logreg_probe  # noqa: E402  (secondary confirmatory probe)


def _cos(a, b):
    na = float(np.linalg.norm(a)); nb = float(np.linalg.norm(b))
    return na, nb, (float(np.dot(a, b) / (na * nb)) if na > C.ZERO_NORM_EPS and nb > C.ZERO_NORM_EPS else None)


def guard_dataset(dataset):
    per_video = []
    min_cos = {"img": 1.0, "text": 1.0}
    min_cos_id = {"img": None, "text": None}
    zero_matched = []
    zero_mismatch = []
    primary_ok = True

    for split in C.SPLITS:
        ext = C.load_extracted_split(dataset, split, with_heads=False)
        cached = C.load_cached_pooled(dataset, split)
        for i, vid in enumerate(ext["ids"]):
            if vid not in cached:
                raise KeyError("id {} in extract but not in cached {}/{}".format(vid, dataset, split))
            c_img, c_txt = cached[vid]
            f_img = ext["img_pooled"][i]; f_txt = ext["text_pooled"][i]
            for stream, cvec, fvec in (("img", c_img, f_img), ("text", c_txt, f_txt)):
                cn, fn, cos = _cos(fvec, cvec)
                c_zero = cn <= C.ZERO_NORM_EPS; f_zero = fn <= C.ZERO_NORM_EPS
                if c_zero and f_zero:
                    zero_matched.append({"id": vid, "split": split, "stream": stream})
                    continue
                if c_zero != f_zero:
                    zero_mismatch.append({"id": vid, "split": split, "stream": stream,
                                          "cached_norm": cn, "fresh_norm": fn})
                    primary_ok = False
                    continue
                per_video.append({"id": vid, "split": split, "stream": stream, "cosine": cos})
                if cos < min_cos[stream]:
                    min_cos[stream] = cos; min_cos_id[stream] = "{}:{}".format(split, vid)

    primary_min = min(min_cos["img"], min_cos["text"])
    primary_pass = bool(primary_ok and primary_min >= C.GUARD_PRIMARY_MIN_COSINE)

    # SECONDARY (confirmatory): fresh-pipeline probe val acc vs the cached-pipeline floor.
    # Fresh  = probe trained on fresh img_pooled(train), evaluated on fresh img_pooled(val).
    # Cached = probe trained on cached img_feats(train), evaluated on cached img_feats(val).
    tr = C.load_extracted_split(dataset, "train", with_heads=False)
    va = C.load_extracted_split(dataset, "val", with_heads=False)
    cached_tr = C.load_cached_pooled(dataset, "train")
    cached_va = C.load_cached_pooled(dataset, "val")
    ytr, yval = tr["labels"], va["labels"]
    Xtr_fresh, Xval_fresh = tr["img_pooled"], va["img_pooled"]
    Xtr_cached = np.stack([cached_tr[v][0] for v in tr["ids"]], axis=0)
    Xval_cached = np.stack([cached_va[v][0] for v in va["ids"]], axis=0)
    proba_fresh, _ = fit_logreg_probe(Xtr_fresh, ytr, Xval_fresh, seed=0)
    proba_cached, _ = fit_logreg_probe(Xtr_cached, ytr, Xval_cached, seed=0)
    acc_fresh = float(((proba_fresh >= 0.5).astype(int) == yval).mean())
    acc_cached = float(((proba_cached >= 0.5).astype(int) == yval).mean())
    sec_delta = abs(acc_fresh - acc_cached)
    secondary_pass = bool(sec_delta <= C.GUARD_SECONDARY_ACC_TOL)

    result = {
        "schema": "sav_f0_guard_v1",
        "dataset": dataset,
        "primary": {
            "min_cosine_img": min_cos["img"], "min_cosine_text": min_cos["text"],
            "min_cosine": primary_min, "min_cosine_img_id": min_cos_id["img"],
            "min_cosine_text_id": min_cos_id["text"], "threshold": C.GUARD_PRIMARY_MIN_COSINE,
            "n_videos_compared": len(per_video), "pass": primary_pass,
        },
        "secondary": {
            "acc_fresh": acc_fresh, "acc_cached": acc_cached, "abs_delta": sec_delta,
            "tolerance": C.GUARD_SECONDARY_ACC_TOL, "pass": secondary_pass,
            "note": "confirmatory only; +/-0.010 on this val permits zero flips; primary decides",
        },
        "zero_matched": zero_matched, "n_zero_matched": len(zero_matched),
        "zero_mismatch": zero_mismatch, "n_zero_mismatch": len(zero_mismatch),
        # PASS iff the PRIMARY feature-level check passes (secondary never blocks a primary pass)
        "pass": primary_pass,
        "guard_verdict": "PASS" if primary_pass else "FAIL",
    }
    C.atomic_write_json(C.guard_path(dataset), result)
    print("[guard] {}: PRIMARY min_cos={:.6f} (>= {}) pass={} | SECONDARY |Δacc|={:.4f} pass={} => {}".format(
        dataset, primary_min, C.GUARD_PRIMARY_MIN_COSINE, primary_pass, sec_delta, secondary_pass,
        result["guard_verdict"]), flush=True)
    return result


def main():
    ap = argparse.ArgumentParser(description="SAV F-G0(b) two-tier reproduction guard.")
    ap.add_argument("--datasets", type=str, default=",".join(C.DATASETS))
    args = ap.parse_args()
    datasets = [d.strip() for d in args.datasets.split(",") if d.strip()]
    all_pass = True
    for ds in datasets:
        r = guard_dataset(ds)
        all_pass = all_pass and r["pass"]
    print("[guard] ALL {} datasets primary-pass={}".format(len(datasets), all_pass), flush=True)
    # non-zero exit if any primary fails (belt-and-braces; the wrapper also jq-gates)
    sys.exit(0 if all_pass else 3)


if __name__ == "__main__":
    main()
