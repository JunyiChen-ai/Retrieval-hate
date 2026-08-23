#!/usr/bin/env python
"""train<->test unlabelled audit.

Three questions, all answered with ZERO labels (neither train nor test):

  1. cross-split near-duplicate census (train<->test, val<->test), reusing the
     P-B decision rules from idea-stage/pilot_b_dup_conflict_census.py
     (c_img thresholds 0.85/0.90/0.95, gate c_img>=0.90 AND token-Jaccard>=0.5).
  2. distribution drift: energy distance + MMD-RBF between train and test in
     feature space, with a split-membership permutation null; plus the
     per-test-sample nearest-train-neighbour distance distribution against a
     size-matched held-out-train control.
  3. test-side degeneracy census: constant (byte-identical) image rows and
     constant text rows on the test/val side.

LABEL GUARD: the ``labels`` tensor of every cache is dropped on load and the
``label`` field of every gt jsonl is dropped on parse; a runtime guard raises if
any code path touches them.  No model is trained, no classification metric is
computed.

Usage:
  python scripts/analysis/traintest_audit.py --out artifacts/traintest_audit/audit.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path("/home/jehc223/Retrieval-hate")
HOME_DATA = Path("/home/jehc223/data")
CLIP = "openai_clip-vit-large-patch14-336_HF"

# ---- frozen P-B constants (idea-stage/PILOT_FREEZE_2026-08-09.md, section P-B) ----
CIMG_THRESHOLDS = (0.85, 0.90, 0.95)
GATE_CIMG = 0.90
GATE_JACCARD = 0.5

# ---- this audit's own constants ----
N_PERM = 200            # permutation null for energy distance / MMD
N_CTRL_REPEATS = 20     # held-out-train control partitions
SEED = 20260809

TOKEN_RE = re.compile(r"[a-z0-9]+|[一-鿿㐀-䶿぀-ヿ가-힯]")

DATASETS = ["HateMM", "MHC", "MHC_zh", "ImpliHateVid"]
SPLITS = {"train": "train", "val": "dev_seen", "test": "test_seen"}
GT_FILE = {"train": "train.jsonl", "val": "val.jsonl", "test": "test.jsonl"}

# raw video roots; None => raw media not on this machine (feature-level only)
RAW_ROOT = {
    "HateMM": [HOME_DATA / "HateMM/video"],
    "MHC": [HOME_DATA / "Multihateclip/English/video_mp4",
            HOME_DATA / "Multihateclip/English/video"],
    "MHC_zh": [HOME_DATA / "Multihateclip/Chinese/video"],
    "ImpliHateVid": [],
}


def log(m):
    print("[%s] %s" % (time.strftime("%H:%M:%S"), m), flush=True)


class LabelGuard:
    """Any attempt to read a label raises."""
    def __getitem__(self, k):
        raise RuntimeError("LABEL_ACCESS_FORBIDDEN")
    def __iter__(self):
        raise RuntimeError("LABEL_ACCESS_FORBIDDEN")
    def __repr__(self):
        return "<labels withheld: zero-label audit>"


def l2np(x):
    return x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), 1e-12)


def tokens(s):
    return set(TOKEN_RE.findall(str(s).lower()))


def jaccard(a, b):
    if not a and not b:
        return 0.0
    u = len(a | b)
    return 0.0 if u == 0 else len(a & b) / u


def md5_file(p):
    h = hashlib.md5()
    with open(p, "rb") as fh:
        for c in iter(lambda: fh.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def rowhash(a):
    return hashlib.sha1(np.ascontiguousarray(a).tobytes()).hexdigest()[:16]


# ------------------------------------------------------------------- loading --
def load_split(ds, split):
    pe = ROOT / "data/CLIP_Embedding" / ds / ("%s_%s.pt" % (SPLITS[split], CLIP))
    pg = ROOT / "data/gt" / ds / GT_FILE[split]
    if not pe.exists() or not pg.exists():
        return None
    c = torch.load(pe, map_location="cpu", weights_only=False)
    c["labels"] = LabelGuard()                      # drop labels immediately
    raw = c["ids"]
    ids = raw[0] if (len(raw) == 1 and isinstance(raw[0], list)) else raw
    ids = [str(v) for v in ids]
    img = c["img_feats"].numpy().astype(np.float64)
    txt = c["text_feats"].numpy().astype(np.float64)
    text = {}
    with open(pg, encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            o = json.loads(line)
            text[str(o["id"])] = o.get("text", "")   # 'label' never read
    miss = [v for v in ids if v not in text]
    if miss:
        raise RuntimeError("JOIN_FAILED:%s:%s:%d" % (ds, split, len(miss)))
    return {"ids": ids, "img": img, "txt": txt,
            "texts": [text[v] for v in ids],
            "cache": str(pe), "n": len(ids)}


# -------------------------------------------------------------- degeneracies --
def degeneracy_census(ds, parts):
    """Byte-identical image rows and byte-identical text rows, over the union of
    all splits of one dataset. Returns per-split flags + group tables."""
    allrows = []
    for sp, P in parts.items():
        for i, vid in enumerate(P["ids"]):
            allrows.append((sp, i, vid))
    ih = {}
    th = {}
    for sp, i, vid in allrows:
        ih.setdefault(rowhash(parts[sp]["img"][i]), []).append((sp, vid))
        th.setdefault(rowhash(parts[sp]["txt"][i]), []).append((sp, vid))

    img_groups = {k: v for k, v in ih.items() if len(v) > 1}
    txt_groups = {k: v for k, v in th.items() if len(v) > 1}

    const_img = {sp: set() for sp in parts}
    for k, v in img_groups.items():
        for sp, vid in v:
            const_img[sp].add(vid)
    const_txt = {sp: set() for sp in parts}
    for k, v in txt_groups.items():
        for sp, vid in v:
            const_txt[sp].add(vid)

    # empty transcript, decided from the gt text itself (independent of hashing)
    empty_txt = {}
    for sp, P in parts.items():
        empty_txt[sp] = {vid for vid, t in zip(P["ids"], P["texts"])
                         if not tokens(t)}

    # black-video signature: the HateMM constant-frame vector, identified by the
    # largest cross-split image group whose members' transcripts are all distinct
    return {
        "img_groups": {k: v for k, v in sorted(img_groups.items(),
                                               key=lambda kv: -len(kv[1]))},
        "txt_groups": {k: v for k, v in sorted(txt_groups.items(),
                                               key=lambda kv: -len(kv[1]))},
        "const_img": const_img,
        "const_txt": const_txt,
        "empty_txt": empty_txt,
    }


# ------------------------------------------------------------------ near-dup --
def cross_split_nearest(A, B, key="img"):
    ka = l2np(A[key])
    kb = l2np(B[key])
    return ka @ kb.T


def near_dup_census(ds, A, B, name_a, name_b, exclude_a, exclude_b):
    """A = query split, B = reference split. Returns pair lists at each threshold."""
    cimg = cross_split_nearest(A, B, "img")
    ctxt = cross_split_nearest(A, B, "txt")
    toks_a = [tokens(t) for t in A["texts"]]
    toks_b = [tokens(t) for t in B["texts"]]
    keep_a = np.array([v not in exclude_a for v in A["ids"]])
    keep_b = np.array([v not in exclude_b for v in B["ids"]])

    out = {"pair": "%s<->%s" % (name_a, name_b), "n_a": A["n"], "n_b": B["n"],
           "thresholds": {}}
    gate_pairs = []
    for thr in CIMG_THRESHOLDS:
        ia, ib = np.where(cimg >= thr)
        clean = keep_a[ia] & keep_b[ib]
        jac = np.array([jaccard(toks_a[x], toks_b[y]) for x, y in zip(ia, ib)])
        jm = jac >= GATE_JACCARD
        out["thresholds"]["%.2f" % thr] = {
            "n_pairs_c_img": int(ia.size),
            "n_pairs_c_img_excl_degenerate": int(clean.sum()),
            "n_conservative_c_img_and_jaccard": int(jm.sum()),
            "n_conservative_excl_degenerate": int((jm & clean).sum()),
            "n_distinct_a_items": int(len(set(ia.tolist()))),
            "n_distinct_b_items": int(len(set(ib.tolist()))),
        }
        if abs(thr - GATE_CIMG) < 1e-9:
            for k in np.argsort(-cimg[ia, ib]):
                gate_pairs.append({
                    "id_a": A["ids"][ia[k]], "id_b": B["ids"][ib[k]],
                    "c_img": float(cimg[ia[k], ib[k]]),
                    "c_txt": float(ctxt[ia[k], ib[k]]),
                    "jaccard": float(jac[k]),
                    "identical_img_row":
                        rowhash(A["img"][ia[k]]) == rowhash(B["img"][ib[k]]),
                    "identical_txt_row":
                        rowhash(A["txt"][ia[k]]) == rowhash(B["txt"][ib[k]]),
                    "a_degenerate": bool(not keep_a[ia[k]]),
                    "b_degenerate": bool(not keep_b[ib[k]]),
                    "conservative": bool(jac[k] >= GATE_JACCARD),
                })
    out["pairs_at_gate_c_img_0.90"] = gate_pairs
    # exact byte-identical image rows across the two splits, regardless of gate
    ha = {rowhash(A["img"][i]): A["ids"][i] for i in range(A["n"])}
    exact = []
    for j in range(B["n"]):
        h = rowhash(B["img"][j])
        if h in ha:
            exact.append({"id_a": ha[h], "id_b": B["ids"][j], "img_row_sha1": h})
    out["byte_identical_img_rows"] = exact
    out["max_c_img"] = float(cimg.max())
    out["max_c_img_excl_degenerate"] = (
        float(cimg[np.ix_(keep_a, keep_b)].max()) if keep_a.any() and keep_b.any()
        else float("nan"))
    return out


# --------------------------------------------------------------------- drift --
def energy_distance(X, Y, rng, n_perm=N_PERM):
    Z = np.vstack([X, Y])
    D = np.sqrt(np.maximum(
        (Z * Z).sum(1)[:, None] + (Z * Z).sum(1)[None, :] - 2 * Z @ Z.T, 0.0))
    n, m = len(X), len(Y)

    def stat(idx_x, idx_y):
        dxy = D[np.ix_(idx_x, idx_y)].mean()
        dxx = D[np.ix_(idx_x, idx_x)].sum() / (len(idx_x) * (len(idx_x) - 1))
        dyy = D[np.ix_(idx_y, idx_y)].sum() / (len(idx_y) * (len(idx_y) - 1))
        return 2 * dxy - dxx - dyy

    obs = stat(np.arange(n), np.arange(n, n + m))
    null = np.empty(n_perm)
    for b in range(n_perm):
        p = rng.permutation(n + m)
        null[b] = stat(p[:n], p[n:])
    return {"energy_distance": float(obs),
            "null_mean": float(null.mean()),
            "null_p95": float(np.percentile(null, 95)),
            "p_value": float((1 + (null >= obs).sum()) / (n_perm + 1)),
            "z_vs_null": float((obs - null.mean()) / max(null.std(), 1e-12))}


def mmd_rbf(X, Y, rng, n_perm=N_PERM):
    Z = np.vstack([X, Y])
    sq = (Z * Z).sum(1)
    D2 = np.maximum(sq[:, None] + sq[None, :] - 2 * Z @ Z.T, 0.0)
    iu = np.triu_indices(len(Z), 1)
    med = np.median(D2[iu])
    gamma = 1.0 / max(med, 1e-12)
    K = np.exp(-gamma * D2)
    n, m = len(X), len(Y)

    def stat(ix, iy):
        kxx = (K[np.ix_(ix, ix)].sum() - len(ix)) / (len(ix) * (len(ix) - 1))
        kyy = (K[np.ix_(iy, iy)].sum() - len(iy)) / (len(iy) * (len(iy) - 1))
        kxy = K[np.ix_(ix, iy)].mean()
        return kxx + kyy - 2 * kxy

    obs = stat(np.arange(n), np.arange(n, n + m))
    null = np.empty(n_perm)
    for b in range(n_perm):
        p = rng.permutation(n + m)
        null[b] = stat(p[:n], p[n:])
    return {"mmd2_unbiased": float(obs),
            "median_heuristic_sigma2": float(med),
            "null_mean": float(null.mean()),
            "null_p95": float(np.percentile(null, 95)),
            "p_value": float((1 + (null >= obs).sum()) / (n_perm + 1)),
            "z_vs_null": float((obs - null.mean()) / max(null.std(), 1e-12))}


def ks_stat(a, b):
    a = np.sort(a); b = np.sort(b)
    allv = np.concatenate([a, b])
    ca = np.searchsorted(a, allv, "right") / len(a)
    cb = np.searchsorted(b, allv, "right") / len(b)
    return float(np.abs(ca - cb).max())


def nn_distance_audit(Xtr, Xte, rng, repeats=N_CTRL_REPEATS):
    """Per-test-sample NN distance into a train reference of size ntr-nte, vs the
    size-matched held-out-train control (queries and reference both from train)."""
    ntr, nte = len(Xtr), len(Xte)
    if ntr - nte < 10:
        return {"error": "train too small for size-matched control"}
    nref = ntr - nte

    def nn(Q, R):
        s = Q @ R.T
        return 1.0 - s.max(1)

    test_stats, ctrl_stats, ks = [], [], []
    for b in range(repeats):
        perm = rng.permutation(ntr)
        ref = Xtr[perm[:nref]]
        held = Xtr[perm[nref:]]
        dte = nn(Xte, ref)
        dct = nn(held, ref)
        test_stats.append(dte)
        ctrl_stats.append(dct)
        ks.append(ks_stat(dte, dct))
    dte = np.concatenate(test_stats)
    dct = np.concatenate(ctrl_stats)
    q = [5, 25, 50, 75, 95]
    return {
        "n_reference": int(nref), "n_test_queries": int(nte),
        "repeats": repeats,
        "test_nn_dist_mean": float(dte.mean()),
        "control_nn_dist_mean": float(dct.mean()),
        "test_nn_dist_quantiles": {str(k): float(v)
                                   for k, v in zip(q, np.percentile(dte, q))},
        "control_nn_dist_quantiles": {str(k): float(v)
                                      for k, v in zip(q, np.percentile(dct, q))},
        "ks_statistic_mean": float(np.mean(ks)),
        "ks_statistic_max": float(np.max(ks)),
        "delta_mean_test_minus_control": float(dte.mean() - dct.mean()),
        "frac_test_nn_dist_below_control_p5":
            float((dte < np.percentile(dct, 5)).mean()),
    }


def keys(P):
    ki, kt = l2np(P["img"]), l2np(P["txt"])
    return {"img": ki, "txt": kt,
            "concat": np.hstack([ki, kt]) / np.sqrt(2.0)}


# ---------------------------------------------------------------------- main --
def find_raw(ds, vid):
    for r in RAW_ROOT.get(ds, []):
        for ext in (".mp4", ".webm", ".mkv", ".flv"):
            p = r / (vid + ext)
            if p.exists():
                return p
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "artifacts/traintest_audit/audit.json"))
    args = ap.parse_args()
    rng = np.random.default_rng(SEED)
    t0 = time.time()
    result = {"audit": "train<->test unlabelled audit", "date": "2026-08-09",
              "labels_used": "none (train or test)",
              "pb_rules": {"c_img_thresholds": list(CIMG_THRESHOLDS),
                           "gate_c_img": GATE_CIMG, "gate_jaccard": GATE_JACCARD,
                           "source": "idea-stage/pilot_b_dup_conflict_census.py"},
              "datasets": {}}

    for ds in DATASETS:
        log("=== %s ===" % ds)
        parts = {}
        for sp in ("train", "val", "test"):
            P = load_split(ds, sp)
            if P is None:
                log("  %s: MISSING cache/gt" % sp)
                continue
            parts[sp] = P
            log("  %s n=%d" % (sp, P["n"]))
        if "train" not in parts or "test" not in parts:
            result["datasets"][ds] = {"error": "train or test split missing"}
            continue

        R = {"n": {sp: parts[sp]["n"] for sp in parts},
             "caches": {sp: parts[sp]["cache"] for sp in parts}}

        # ---- 3. degeneracy census (all splits, test side included) ----
        deg = degeneracy_census(ds, parts)
        R["degeneracy"] = {
            "constant_img_row_counts": {sp: len(deg["const_img"][sp]) for sp in parts},
            "constant_img_row_ids": {sp: sorted(deg["const_img"][sp]) for sp in parts},
            "empty_transcript_counts": {sp: len(deg["empty_txt"][sp]) for sp in parts},
            "empty_transcript_ids": {sp: sorted(deg["empty_txt"][sp]) for sp in parts},
            "constant_txt_row_counts": {sp: len(deg["const_txt"][sp]) for sp in parts},
            "img_groups_top": [
                {"sha1": k, "size": len(v), "members": v}
                for k, v in list(deg["img_groups"].items())[:12]],
            "txt_groups_top": [
                {"sha1": k, "size": len(v),
                 "members": v if len(v) <= 20 else v[:20] + [("...", "...")]}
                for k, v in list(deg["txt_groups"].items())[:6]],
        }
        excl = {sp: set(deg["const_img"][sp]) | set(deg["empty_txt"][sp])
                for sp in parts}
        R["degeneracy"]["excluded_counts"] = {sp: len(excl[sp]) for sp in parts}

        # ---- 1. cross-split near-duplicate census ----
        R["near_duplicates"] = {}
        pairs_to_do = [("train", "test")]
        if "val" in parts:
            pairs_to_do += [("val", "test"), ("train", "val")]
        for a, b in pairs_to_do:
            nd = near_dup_census(ds, parts[a], parts[b], a, b, excl[a], excl[b])
            # md5 the raw media for every gate pair we can reach
            for pr in nd["pairs_at_gate_c_img_0.90"]:
                pa, pb = find_raw(ds, pr["id_a"]), find_raw(ds, pr["id_b"])
                pr["raw_a"] = str(pa) if pa else None
                pr["raw_b"] = str(pb) if pb else None
                pr["md5_a"] = md5_file(pa) if pa else None
                pr["md5_b"] = md5_file(pb) if pb else None
                pr["md5_identical"] = (pr["md5_a"] is not None
                                       and pr["md5_a"] == pr["md5_b"])
            R["near_duplicates"]["%s<->%s" % (a, b)] = nd
            g = nd["thresholds"]["0.90"]
            log("  %s<->%s: c_img>=0.90 pairs=%d (clean %d), conservative=%d "
                "(clean %d), max c_img=%.4f"
                % (a, b, g["n_pairs_c_img"], g["n_pairs_c_img_excl_degenerate"],
                   g["n_conservative_c_img_and_jaccard"],
                   g["n_conservative_excl_degenerate"], nd["max_c_img"]))

        # ---- 2. drift ----
        R["drift"] = {}
        for variant, use_excl in (("with_degenerate", False), ("excl_degenerate", True)):
            block = {}
            for kname in ("img", "txt", "concat"):
                Ktr = keys(parts["train"])[kname]
                Kte = keys(parts["test"])[kname]
                if use_excl:
                    mtr = np.array([v not in excl["train"] for v in parts["train"]["ids"]])
                    mte = np.array([v not in excl["test"] for v in parts["test"]["ids"]])
                    Ktr, Kte = Ktr[mtr], Kte[mte]
                r2 = np.random.default_rng(SEED)
                block[kname] = {
                    "n_train": int(len(Ktr)), "n_test": int(len(Kte)),
                    "energy": energy_distance(Ktr, Kte, r2),
                    "mmd_rbf": mmd_rbf(Ktr, Kte, np.random.default_rng(SEED)),
                    "nn_distance": nn_distance_audit(Ktr, Kte,
                                                     np.random.default_rng(SEED)),
                }
                e = block[kname]["energy"]
                n = block[kname]["nn_distance"]
                log("  drift[%s/%s]: E=%.5f p=%.3f z=%.1f | MMD2=%.5f p=%.3f | "
                    "NN test %.4f vs ctrl %.4f (KS %.3f)"
                    % (variant, kname, e["energy_distance"], e["p_value"],
                       e["z_vs_null"], block[kname]["mmd_rbf"]["mmd2_unbiased"],
                       block[kname]["mmd_rbf"]["p_value"],
                       n.get("test_nn_dist_mean", float("nan")),
                       n.get("control_nn_dist_mean", float("nan")),
                       n.get("ks_statistic_mean", float("nan"))))
            R["drift"][variant] = block

        result["datasets"][ds] = R

    result["elapsed_sec"] = time.time() - t0
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(result, indent=2, ensure_ascii=False,
                                         default=str))
    log("wrote %s (%.1fs)" % (args.out, result["elapsed_sec"]))


if __name__ == "__main__":
    sys.exit(main())
