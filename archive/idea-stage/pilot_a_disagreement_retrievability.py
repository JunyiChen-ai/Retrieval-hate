#!/usr/bin/env python
"""P-A — Disagreement retrievability gate.

Decision rules are frozen in idea-stage/PILOT_FREEZE_2026-08-09.md (section P-A) and are
NOT edited after results are seen.

Zero test-set contact: only `train` / `dev_seen` embedding caches, only
data/gt/{MHC,MHC_zh}/{train,val}.jsonl, only data/gt/mhc_votes/mhc_*_{train,valid}.tsv.
An explicit path guard HALTs on any path whose name contains "test".

Usage:
  python idea-stage/pilot_a_disagreement_retrievability.py --smoke synthetic
  python idea-stage/pilot_a_disagreement_retrievability.py --smoke permuted
  python idea-stage/pilot_a_disagreement_retrievability.py --out idea-stage/pilot_a.json
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from scipy.stats import rankdata

ROOT = Path("/home/jehc223/Retrieval-hate")

# ---- frozen constants (PILOT_FREEZE_2026-08-09.md, P-A) ----
KNN = 20
BOOT = 2000
BOOT_SEED = 20260908          # bootstrap resample seed (shared draws for s and h)
NULL_SEED = 20260909          # frozen in the freeze document
NULL_LO, NULL_HI = 0.45, 0.55
GO_AUROC = 0.60
GO_LB = 0.55
GO_DELTA = 0.03

CLIP = "openai_clip-vit-large-patch14-336_HF"
LANGS = {
    "EN": {"ds": "MHC", "tsv": "English"},
    "ZH": {"ds": "MHC_zh", "tsv": "Chinese"},
}
# project binary protocol, frozen in the freeze document
HARM_VOTES = {"Offensive", "Hateful"}
BENIGN_VOTES = {"Normal", "Counter Narrative"}
VOTE_ALIASES = {"No": "Normal"}   # single stray token in the Chinese release


class Halt(RuntimeError):
    pass


def log(msg):
    print("[%s] %s" % (time.strftime("%H:%M:%S"), msg), flush=True)


# ------------------------------------------------------------------ guards --
_GUARD_ARMED = False
_TOUCHED = []


def arm_guard():
    global _GUARD_ARMED
    _GUARD_ARMED = True
    log("GUARD ARMED: any path whose name contains 'test' HALTs; "
        "allowed split tokens = {train, val, valid, dev_seen}")


def guard_path(p):
    if not _GUARD_ARMED:
        raise Halt("HALT_GUARD_NOT_ARMED")
    p = Path(p)
    low = str(p).lower()
    for part in p.parts:
        if "test" in part.lower():
            raise Halt("HALT_TEST_CONTACT:path=%s" % p)
    if "test_seen" in low:
        raise Halt("HALT_TEST_CONTACT:path=%s" % p)
    _TOUCHED.append(str(p))
    return p


def guard_open(p, **kw):
    return open(guard_path(p), **kw)


def guard_torch_load(p):
    return torch.load(guard_path(p), map_location="cpu", weights_only=False)


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for c in iter(lambda: fh.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


# -------------------------------------------------------------------- math --
def l2np(x):
    return x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), 1e-12)


def auroc(y, s):
    """Mann-Whitney AUROC with tie-corrected ranks. NaN if a class is empty."""
    y = np.asarray(y, dtype=np.int64)
    s = np.asarray(s, dtype=np.float64)
    n1 = int(y.sum())
    n0 = int(y.size - n1)
    if n1 == 0 or n0 == 0:
        return float("nan")
    r = rankdata(s)
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2.0) / (n0 * n1))


# -------------------------------------------------------------------- data --
def parse_votes(raw):
    """raw -> (list of canonical vote strings). Applies the frozen alias map."""
    votes = ast.literal_eval(raw)
    out = []
    aliased = 0
    for v in votes:
        v = str(v).strip()
        if v in VOTE_ALIASES:
            v = VOTE_ALIASES[v]
            aliased += 1
        if v not in HARM_VOTES and v not in BENIGN_VOTES:
            raise Halt("HALT_UNKNOWN_VOTE:%r" % v)
        out.append(v)
    return out, aliased


def load_votes(tsv_lang):
    """Returns {video_id: [votes]} from the train + valid TSVs only."""
    table = {}
    n_alias = 0
    per_file = {}
    for split in ("train", "valid"):
        p = ROOT / "data/gt/mhc_votes" / ("mhc_%s_%s.tsv" % (tsv_lang, split))
        rows = 0
        with guard_open(p, encoding="utf-8") as fh:
            hdr = fh.readline().rstrip("\n").split("\t")
            if hdr[:3] != ["Video_ID", "Majority_Voting", "Label"]:
                raise Halt("HALT_TSV_HEADER:%s:%r" % (p, hdr))
            for line in fh:
                if not line.strip():
                    continue
                parts = line.rstrip("\n").split("\t")
                vid = parts[0].strip()
                votes, a = parse_votes(parts[2])
                n_alias += a
                if vid in table and table[vid] != votes:
                    raise Halt("HALT_DUP_VOTE_ROW:%s" % vid)
                table[vid] = votes
                rows += 1
        per_file[str(p)] = {"rows": rows, "sha256": sha256_file(p)}
    return table, n_alias, per_file


def load_lang(langkey):
    cfg = LANGS[langkey]
    ds = cfg["ds"]
    parts = []
    for split_emb, split_gt in (("train", "train"), ("dev_seen", "val")):
        pe = ROOT / "data/CLIP_Embedding" / ds / ("%s_%s.pt" % (split_emb, CLIP))
        d = guard_torch_load(pe)
        raw = d["ids"]
        ids = raw[0] if (len(raw) == 1 and isinstance(raw[0], list)) else raw
        ids = [str(v) for v in ids]
        img = d["img_feats"].numpy().astype(np.float64)
        txt = d["text_feats"].numpy().astype(np.float64)
        y = d["labels"].numpy().astype(np.int64)
        if not (len(ids) == img.shape[0] == txt.shape[0] == y.shape[0]):
            raise Halt("HALT_CACHE_SHAPE:%s" % pe)
        # cross-check the id list against the local gt jsonl for the same split
        pg = ROOT / "data/gt" / ds / ("%s.jsonl" % split_gt)
        gt_ids, gt_lab = [], {}
        with guard_open(pg, encoding="utf-8") as fh:
            for line in fh:
                if not line.strip():
                    continue
                o = json.loads(line)
                gt_ids.append(str(o["id"]))
                gt_lab[str(o["id"])] = int(o["label"])
        if set(gt_ids) != set(ids):
            raise Halt("HALT_ID_MISMATCH:%s vs %s" % (pe, pg))
        for i, v in enumerate(ids):
            if gt_lab[v] != int(y[i]):
                raise Halt("HALT_LABEL_MISMATCH:%s:%s" % (ds, v))
        parts.append((ids, img, txt, y, split_emb))
    ids = parts[0][0] + parts[1][0]
    img = np.concatenate([parts[0][1], parts[1][1]], 0)
    txt = np.concatenate([parts[0][2], parts[1][2]], 0)
    y = np.concatenate([parts[0][3], parts[1][3]], 0)
    origin = ["train"] * len(parts[0][0]) + ["val"] * len(parts[1][0])
    if len(set(ids)) != len(ids):
        raise Halt("HALT_DUP_ID_ACROSS_SPLITS:%s" % ds)
    return ids, img, txt, y, origin


def targets_from_votes(ids, votes_table):
    """T1 = >=2 distinct raw labels; T2 = votes disagree after binary mapping."""
    missing = [v for v in ids if v not in votes_table]
    if missing:
        raise Halt("HALT_JOIN_FAILED:%d ids, e.g. %r" % (len(missing), missing[:5]))
    t1 = np.zeros(len(ids), dtype=np.int64)
    t2 = np.zeros(len(ids), dtype=np.int64)
    nvotes = np.zeros(len(ids), dtype=np.int64)
    maj_bin = np.zeros(len(ids), dtype=np.int64)
    for i, v in enumerate(ids):
        vs = votes_table[v]
        nvotes[i] = len(vs)
        t1[i] = int(len(set(vs)) >= 2)
        b = [1 if x in HARM_VOTES else 0 for x in vs]
        t2[i] = int(len(set(b)) >= 2)
        maj_bin[i] = int(sum(b) * 2 > len(b))
    if not np.all(t2 <= t1):
        raise Halt("HALT_T2_NOT_SUBSET_T1")
    return t1, t2, nvotes, maj_bin


# --------------------------------------------------------------- predictor --
def neighbour_structure(ids, img, txt):
    """k=20 LOO neighbours by dot product of [l2(img) || l2(txt)]; ties broken
    lexicographically by video_id. Returns (idx[N,k], w[N,k])."""
    key = np.concatenate([l2np(img), l2np(txt)], axis=1)
    sim = key @ key.T
    n = len(ids)
    np.fill_diagonal(sim, -np.inf)
    lexrank = np.empty(n, dtype=np.int64)
    lexrank[np.argsort(np.array(ids, dtype=object), kind="stable")] = np.arange(n)
    # np.lexsort: last key is primary -> primary -sim ascending (= sim desc),
    # tie broken by lexrank ascending (lexicographic video_id).
    idx = np.empty((n, KNN), dtype=np.int64)
    for i in range(n):
        order = np.lexsort((lexrank, -sim[i]))
        idx[i] = order[:KNN]
    if np.any(idx == np.arange(n)[:, None]):
        raise Halt("HALT_SELF_IN_NEIGHBOURS")
    w = np.take_along_axis(sim, idx, axis=1)
    return idx, w


def wmean(idx, w, t):
    """Similarity-weighted mean of t over the neighbours (continuous, non-saturating)."""
    tn = np.asarray(t, dtype=np.float64)[idx]
    den = w.sum(axis=1)
    if np.any(np.abs(den) < 1e-9):
        raise Halt("HALT_ZERO_WEIGHT_DENOM")
    return (w * tn).sum(axis=1) / den


def hardness(idx, w, ybin):
    p = wmean(idx, w, ybin)
    return 1.0 - np.abs(p - 0.5) * 2.0, p


# -------------------------------------------------------------- bootstrap ---
def paired_bootstrap(t, s, h, seed):
    """Same resample draws for s and h => paired Delta."""
    rng = np.random.default_rng(seed)
    n = len(t)
    a_s, a_h, a_d = [], [], []
    n_skip = 0
    for _ in range(BOOT):
        b = rng.integers(0, n, size=n)
        tb = t[b]
        if tb.sum() == 0 or tb.sum() == n:
            n_skip += 1
            continue
        vs = auroc(tb, s[b])
        vh = auroc(tb, h[b])
        a_s.append(vs)
        a_h.append(vh)
        a_d.append(vs - vh)
    return (np.array(a_s), np.array(a_h), np.array(a_d), n_skip)


def ci(a):
    if a.size == 0:
        return [float("nan"), float("nan")]
    return [float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5))]


# ------------------------------------------------------------------- runner --
def run_lang(langkey, ids, img, txt, ybin, t1, t2, meta):
    idx, w = neighbour_structure(ids, img, txt)
    neg_w = int((w < 0).sum())
    res = {
        "n": len(ids),
        "n_neighbour_weights_negative": neg_w,
        "weight_min": float(w.min()),
        "weight_max": float(w.max()),
        "base_rate_T1": float(t1.mean()),
        "base_rate_T2": float(t2.mean()),
        "base_rate_label_pos": float(ybin.mean()),
    }
    s1 = wmean(idx, w, t1)
    s2 = wmean(idx, w, t2)
    h, p = hardness(idx, w, ybin)

    e1 = auroc(t1, s1)
    e2 = auroc(t2, s2)
    e3 = auroc(t1, h)
    res["E1_auroc_s_on_T1"] = e1
    res["E2_auroc_s_on_T2"] = e2
    res["E3_auroc_h_on_T1"] = e3
    res["delta_s_minus_h_on_T1"] = e1 - e3
    # descriptive cross-checks (non-gating)
    res["aux_auroc_sT1_on_T2"] = auroc(t2, s1)
    res["aux_auroc_h_on_T2"] = auroc(t2, h)

    bs, bh, bd, nskip = paired_bootstrap(t1, s1, h, BOOT_SEED)
    res["boot"] = {
        "n_resamples": BOOT,
        "n_skipped_degenerate": nskip,
        "seed": BOOT_SEED,
        "auroc_s_T1_ci95": ci(bs),
        "auroc_h_T1_ci95": ci(bh),
        "delta_ci95": ci(bd),
        "delta_frac_positive": float((bd > 0).mean()) if bd.size else float("nan"),
    }
    # T2 bootstrap (secondary, s2 vs h on T2)
    bs2, bh2, bd2, nskip2 = paired_bootstrap(t2, s2, h, BOOT_SEED)
    res["boot_T2"] = {
        "n_skipped_degenerate": nskip2,
        "auroc_s_T2_ci95": ci(bs2),
        "delta_T2_ci95": ci(bd2),
    }

    # frozen null control: permute the contestedness targets, seed 20260909
    rng = np.random.default_rng(NULL_SEED)
    t1p = t1[rng.permutation(len(t1))]
    s1p = wmean(idx, w, t1p)
    a_null = auroc(t1p, s1p)
    res["null_auroc_T1_permuted"] = a_null
    res["null_seed"] = NULL_SEED
    res["null_in_range"] = bool(NULL_LO <= a_null <= NULL_HI)
    return res


def verdict(per_lang):
    """Transcribed frozen rule:
      GO  -- in both languages: AUROC(s) on T1 >= 0.60 with bootstrap 95% LB > 0.55,
             and Delta >= +0.03.
      AMBIGUOUS -- the AUROC bar is met in both languages but Delta < +0.03; or the full
             GO condition holds in exactly one language.
      NO-GO -- AUROC(s) < 0.60 in either language, or Delta <= 0 in both.
    """
    langs = list(per_lang.keys())
    full = {}
    aurocbar = {}
    for L in langs:
        r = per_lang[L]
        a = r["E1_auroc_s_on_T1"]
        lb = r["boot"]["auroc_s_T1_ci95"][0]
        d = r["delta_s_minus_h_on_T1"]
        aurocbar[L] = bool(a >= GO_AUROC and lb > GO_LB)
        full[L] = bool(aurocbar[L] and d >= GO_DELTA)
    deltas = [per_lang[L]["delta_s_minus_h_on_T1"] for L in langs]
    aurocs = [per_lang[L]["E1_auroc_s_on_T1"] for L in langs]

    nogo = any(a < GO_AUROC for a in aurocs) or all(d <= 0 for d in deltas)
    go = all(full.values())
    amb = (all(aurocbar.values()) and any(
        per_lang[L]["delta_s_minus_h_on_T1"] < GO_DELTA for L in langs)) or (
        sum(full.values()) == 1)

    if go:
        v = "GO"
    elif nogo:
        v = "NO-GO"
    elif amb:
        v = "AMBIGUOUS"
    else:
        v = "AMBIGUOUS"
    return {
        "verdict": v,
        "per_lang_auroc_bar_met": aurocbar,
        "per_lang_full_go": full,
        "nogo_condition_triggered": bool(nogo),
        "go_condition_triggered": bool(go),
        "ambiguous_condition_triggered": bool(amb),
        "precedence_note": ("The three frozen clauses can co-fire (e.g. full GO in exactly "
                            "one language while the other is below 0.60). Precedence applied: "
                            "GO > NO-GO > AMBIGUOUS. All three raw flags are reported so the "
                            "reader can re-adjudicate without re-running."),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", choices=["synthetic", "permuted"], default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    arm_guard()
    t0 = time.time()
    out = {
        "pilot": "P-A disagreement retrievability",
        "freeze": "idea-stage/PILOT_FREEZE_2026-08-09.md",
        "mode": args.smoke or "real",
        "knn": KNN, "boot": BOOT, "boot_seed": BOOT_SEED, "null_seed": NULL_SEED,
        "guard": "armed: any path containing 'test' HALTs",
        "per_lang": {},
    }

    if args.smoke == "synthetic":
        rng = np.random.default_rng(7)
        for L, n in (("EN", 629), ("ZH", 657)):
            ids = ["syn_%s_%04d" % (L, i) for i in range(n)]
            img = rng.normal(size=(n, 1024))
            txt = rng.normal(size=(n, 768))
            ybin = rng.integers(0, 2, size=n)
            t1 = rng.integers(0, 2, size=n)
            t2 = (t1 * rng.integers(0, 2, size=n)).astype(np.int64)
            out["per_lang"][L] = run_lang(L, ids, img, txt, ybin, t1, t2, {})
        out["verdict_block"] = verdict(out["per_lang"])
        log("SMOKE synthetic: %s" % json.dumps(
            {L: {k: out["per_lang"][L][k] for k in
                 ("E1_auroc_s_on_T1", "E3_auroc_h_on_T1", "null_auroc_T1_permuted")}
             for L in out["per_lang"]}, indent=1))
        log("elapsed %.1fs" % (time.time() - t0))
        return

    # ---- real data load (needed for both `permuted` smoke and the real run) ----
    meta = {"files": {}}
    data = {}
    for L, cfg in LANGS.items():
        ids, img, txt, ybin, origin = load_lang(L)
        votes, n_alias, per_file = load_votes(cfg["tsv"])
        meta["files"].update(per_file)
        t1, t2, nvotes, maj_bin = targets_from_votes(ids, votes)
        agree = float((maj_bin == ybin).mean())
        log("%s: n=%d (train=%d val=%d) join=100%%, vote-alias 'No'->'Normal' applied %dx, "
            "majority-vs-cache-label agreement=%.4f, vote-count hist=%s"
            % (L, len(ids), origin.count("train"), origin.count("val"), n_alias, agree,
               dict(zip(*np.unique(nvotes, return_counts=True)))))
        data[L] = dict(ids=ids, img=img, txt=txt, ybin=ybin, t1=t1, t2=t2,
                       nvotes=nvotes, n_alias=n_alias, maj_agree=agree,
                       n_train=origin.count("train"), n_val=origin.count("val"))

    if args.smoke == "permuted":
        # label-permuted null smoke (seed 999, NOT the frozen null seed) -- reveals
        # nothing about the real endpoints because the targets are destroyed.
        rng = np.random.default_rng(999)
        for L in LANGS:
            d = data[L]
            perm = rng.permutation(len(d["ids"]))
            out["per_lang"][L] = run_lang(L, d["ids"], d["img"], d["txt"],
                                          d["ybin"][perm], d["t1"][perm], d["t2"][perm], {})
        log("SMOKE permuted: %s" % json.dumps(
            {L: {k: out["per_lang"][L][k] for k in
                 ("E1_auroc_s_on_T1", "E3_auroc_h_on_T1", "delta_s_minus_h_on_T1")}
             for L in out["per_lang"]}, indent=1))
        log("elapsed %.1fs" % (time.time() - t0))
        return

    # ---------------------------- REAL RUN (single submission) ----------------
    for L in LANGS:
        d = data[L]
        r = run_lang(L, d["ids"], d["img"], d["txt"], d["ybin"], d["t1"], d["t2"], meta)
        r["n_train"] = d["n_train"]
        r["n_val"] = d["n_val"]
        r["vote_count_hist"] = {int(k): int(v) for k, v in
                                zip(*np.unique(d["nvotes"], return_counts=True))}
        r["vote_alias_No_to_Normal"] = d["n_alias"]
        r["majority_vs_cache_label_agreement"] = d["maj_agree"]
        out["per_lang"][L] = r
        log("%s E1=%.4f E2=%.4f E3=%.4f D=%+.4f LB=%.4f null=%.4f"
            % (L, r["E1_auroc_s_on_T1"], r["E2_auroc_s_on_T2"], r["E3_auroc_h_on_T1"],
               r["delta_s_minus_h_on_T1"], r["boot"]["auroc_s_T1_ci95"][0],
               r["null_auroc_T1_permuted"]))

    out["verdict_block"] = verdict(out["per_lang"])
    bad = [L for L in out["per_lang"] if not out["per_lang"][L]["null_in_range"]]
    out["null_control_pass"] = (len(bad) == 0)
    out["null_control_failed_langs"] = bad
    if bad:
        out["verdict_block"]["verdict"] = "VOID (null control out of [0.45,0.55]): " + \
            out["verdict_block"]["verdict"]
        log("!!! NULL CONTROL OUT OF RANGE for %s -- pilot flagged VOID per the freeze" % bad)

    out["meta"] = meta
    out["paths_touched"] = sorted(set(_TOUCHED))
    out["elapsed_sec"] = time.time() - t0
    log("VERDICT: %s" % out["verdict_block"]["verdict"])
    if args.out:
        Path(args.out).write_text(json.dumps(out, indent=2, ensure_ascii=False))
        log("wrote %s" % args.out)


if __name__ == "__main__":
    try:
        main()
    except Halt as e:
        log("HALT %s" % e)
        sys.exit(3)
