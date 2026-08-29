#!/usr/bin/env python
"""P-B — Near-duplicate and label-conflict census.

Decision rules are frozen in idea-stage/PILOT_FREEZE_2026-08-09.md (section P-B) and are
NOT edited after results are seen.

Zero test-set contact: TRAIN splits only. An explicit path guard HALTs on any path whose
name contains "test" (this also excludes `test_seen` embedding caches and `dev_seen` is
simply never referenced).

Usage:
  python idea-stage/pilot_b_dup_conflict_census.py --smoke synthetic
  python idea-stage/pilot_b_dup_conflict_census.py --smoke planted
  python idea-stage/pilot_b_dup_conflict_census.py --out idea-stage/pilot_b.json
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import re
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path("/home/jehc223/Retrieval-hate")

# ---- frozen constants (PILOT_FREEZE_2026-08-09.md, P-B) ----
CIMG_THRESHOLDS = (0.85, 0.90, 0.95)
GATE_CIMG = 0.90
GATE_JACCARD = 0.5
ALIVE_N = 30
DEAD_N = 10
NULL_SEED = 20260909

CLIP = "openai_clip-vit-large-patch14-336_HF"
# every dataset the freeze names; HateClipSeg is checked for a train cache at runtime
CANDIDATE_DS = ["HateMM", "MHC", "MHC_zh", "HateClipSeg", "ImpliHateVid"]
MHC_DS = {"MHC": "English", "MHC_zh": "Chinese"}

TOKEN_RE = re.compile(r"[a-z0-9]+|[一-鿿㐀-䶿぀-ヿ가-힯]")


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
    log("GUARD ARMED: any path whose name contains 'test' HALTs; P-B opens TRAIN splits only "
        "(no dev_seen, no val, no test)")


def guard_path(p, train_only=True):
    if not _GUARD_ARMED:
        raise Halt("HALT_GUARD_NOT_ARMED")
    p = Path(p)
    for part in p.parts:
        pl = part.lower()
        if "test" in pl:
            raise Halt("HALT_TEST_CONTACT:path=%s" % p)
        if train_only and ("dev_seen" in pl or pl in ("val.jsonl", "valid.tsv")):
            raise Halt("HALT_NON_TRAIN_SPLIT:path=%s" % p)
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


def l2np(x):
    return x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), 1e-12)


def tokens(s):
    return set(TOKEN_RE.findall(str(s).lower()))


def jaccard(a, b):
    if not a and not b:
        return 0.0          # both empty -> undefined; frozen to 0 (conservative)
    u = len(a | b)
    return 0.0 if u == 0 else len(a & b) / u


# -------------------------------------------------------------------- data --
def load_pool():
    ds_present, ds_absent = [], []
    ids, img, txt, y, ds, texts = [], [], [], [], [], []
    files = {}
    for d in CANDIDATE_DS:
        pe = ROOT / "data/CLIP_Embedding" / d / ("train_%s.pt" % CLIP)
        pg = ROOT / "data/gt" / d / "train.jsonl"
        if not pe.exists() or not pg.exists():
            ds_absent.append({"dataset": d,
                              "train_embedding": pe.exists(),
                              "train_jsonl": pg.exists()})
            continue
        c = guard_torch_load(pe)
        raw = c["ids"]
        cids = raw[0] if (len(raw) == 1 and isinstance(raw[0], list)) else raw
        cids = [str(v) for v in cids]
        ci = c["img_feats"].numpy().astype(np.float64)
        ct = c["text_feats"].numpy().astype(np.float64)
        cy = c["labels"].numpy().astype(np.int64)
        if not (len(cids) == ci.shape[0] == ct.shape[0] == cy.shape[0]):
            raise Halt("HALT_CACHE_SHAPE:%s" % pe)
        gt = {}
        with guard_open(pg, encoding="utf-8") as fh:
            for line in fh:
                if not line.strip():
                    continue
                o = json.loads(line)
                gt[str(o["id"])] = (o.get("text", ""), int(o["label"]))
        miss = [v for v in cids if v not in gt]
        if miss:
            raise Halt("HALT_JOIN_FAILED:%s:%d e.g.%r" % (d, len(miss), miss[:3]))
        for i, v in enumerate(cids):
            if gt[v][1] != int(cy[i]):
                raise Halt("HALT_LABEL_MISMATCH:%s:%s" % (d, v))
        ids += cids
        img.append(ci)
        txt.append(ct)
        y.append(cy)
        ds += [d] * len(cids)
        texts += [gt[v][0] for v in cids]
        files[str(pe)] = {"n": len(cids), "sha256": sha256_file(pe)}
        ds_present.append({"dataset": d, "n": len(cids), "n_pos": int(cy.sum())})
        log("loaded %s train: n=%d pos=%d" % (d, len(cids), int(cy.sum())))
    if not ds_present:
        raise Halt("HALT_NO_DATA")
    return (ids, np.concatenate(img, 0), np.concatenate(txt, 0),
            np.concatenate(y, 0), np.array(ds), texts, ds_present, ds_absent, files)


def load_mhc_cn_flags(ids, ds):
    """Which items carry >=1 'Counter Narrative' raw vote. TRAIN tsv only."""
    flag = np.zeros(len(ids), dtype=bool)
    known = np.zeros(len(ids), dtype=bool)
    for d, lang in MHC_DS.items():
        p = ROOT / "data/gt/mhc_votes" / ("mhc_%s_train.tsv" % lang)
        table = {}
        with guard_open(p, encoding="utf-8") as fh:
            fh.readline()
            for line in fh:
                if not line.strip():
                    continue
                parts = line.rstrip("\n").split("\t")
                table[parts[0].strip()] = ast.literal_eval(parts[2])
        sel = np.where(ds == d)[0]
        miss = [ids[i] for i in sel if ids[i] not in table]
        if miss:
            raise Halt("HALT_VOTE_JOIN_FAILED:%s:%d e.g.%r" % (d, len(miss), miss[:3]))
        for i in sel:
            known[i] = True
            flag[i] = ("Counter Narrative" in table[ids[i]])
    return flag, known


# ------------------------------------------------------------------ census --
def census(ids, img, txt, y, ds, texts, cn_flag, cn_known, tag=""):
    n = len(ids)
    ki = l2np(img)
    kt = l2np(txt)
    cimg = ki @ ki.T
    ctxt = kt @ kt.T
    iu, ju = np.triu_indices(n, k=1)          # unordered, self-pairs excluded
    ci = cimg[iu, ju]
    ct = ctxt[iu, ju]
    same_ds = (ds[iu] == ds[ju])
    conflict = (y[iu] != y[ju])

    toks = [tokens(t) for t in texts]
    empty_text = int(sum(1 for t in toks if not t))

    res = {"n_items": n, "n_unordered_pairs": int(iu.size),
           "n_items_empty_transcript": empty_text, "tag": tag,
           "thresholds": {}}
    flagged_examples = []
    for thr in CIMG_THRESHOLDS:
        m = ci >= thr
        sel = np.where(m)[0]
        # Jaccard only for flagged pairs (N4)
        jac = np.array([jaccard(toks[iu[k]], toks[ju[k]]) for k in sel])
        cf = conflict[sel]
        sd = same_ds[sel]
        jm = jac >= GATE_JACCARD
        blk = {
            "N1_pairs_total": int(sel.size),
            "N1_pairs_within_dataset": int(sd.sum()),
            "N1_pairs_cross_dataset": int((~sd).sum()),
            "N2_conflicting_total": int(cf.sum()),
            "N2_conflicting_within_dataset": int((cf & sd).sum()),
            "N2_conflicting_cross_dataset": int((cf & ~sd).sum()),
            "N4_jaccard_ge_0.5_total": int(jm.sum()),
            "N4_conservative_conflicting_total": int((cf & jm).sum()),
            "N4_conservative_conflicting_within_dataset": int((cf & jm & sd).sum()),
            "N4_conservative_conflicting_cross_dataset": int((cf & jm & ~sd).sum()),
            "jaccard_mean_flagged": float(jac.mean()) if jac.size else float("nan"),
            "jaccard_median_flagged": float(np.median(jac)) if jac.size else float("nan"),
            "ctxt_mean_flagged": float(ct[sel].mean()) if sel.size else float("nan"),
        }
        # per-dataset within-dataset breakdown
        per_ds = {}
        for d in sorted(set(ds.tolist())):
            dm = sd & (ds[iu[sel]] == d)
            per_ds[d] = {
                "N1": int(dm.sum()),
                "N2_conflicting": int((dm & cf).sum()),
                "N4_conservative_conflicting": int((dm & cf & jm).sum()),
            }
        blk["per_dataset_within"] = per_ds
        # cross-dataset pair-type breakdown
        cross = {}
        for k in sel[~sd]:
            key = "|".join(sorted([ds[iu[k]], ds[ju[k]]]))
            e = cross.setdefault(key, {"N1": 0, "N2_conflicting": 0,
                                       "N4_conservative_conflicting": 0})
            e["N1"] += 1
            c = conflict[k]
            j = jaccard(toks[iu[k]], toks[ju[k]])
            if c:
                e["N2_conflicting"] += 1
                if j >= GATE_JACCARD:
                    e["N4_conservative_conflicting"] += 1
        blk["cross_dataset_pairs"] = cross

        # N3: Counter Narrative on one side, among MHC conflicting pairs
        mhc_pair = cn_known[iu[sel]] | cn_known[ju[sel]]
        mhc_both = cn_known[iu[sel]] & cn_known[ju[sel]]
        cn_any = cn_flag[iu[sel]] | cn_flag[ju[sel]]
        blk["N3"] = {
            "conflicting_pairs_any_side_MHC": int((cf & mhc_pair).sum()),
            "conflicting_pairs_both_sides_MHC": int((cf & mhc_both).sum()),
            "of_those_any_side_MHC_with_CounterNarrative_vote":
                int((cf & mhc_pair & cn_any).sum()),
            "of_those_both_sides_MHC_with_CounterNarrative_vote":
                int((cf & mhc_both & cn_any).sum()),
            "conservative_conflicting_any_side_MHC": int((cf & jm & mhc_pair).sum()),
            "conservative_conflicting_any_side_MHC_with_CN":
                int((cf & jm & mhc_pair & cn_any).sum()),
        }
        res["thresholds"][str(thr)] = blk

        if thr == GATE_CIMG:
            keep = sel[cf]
            jk = jac[cf]
            ordk = np.argsort(-jk)
            for r in ordk[:40]:
                k = keep[r]
                flagged_examples.append({
                    "id_a": ids[iu[k]], "ds_a": ds[iu[k]], "y_a": int(y[iu[k]]),
                    "id_b": ids[ju[k]], "ds_b": ds[ju[k]], "y_b": int(y[ju[k]]),
                    "c_img": float(ci[k]), "c_txt": float(ct[k]),
                    "jaccard": float(jk[r]),
                    "cn_a": bool(cn_flag[iu[k]]), "cn_b": bool(cn_flag[ju[k]]),
                })
    res["top_conflicting_pairs_at_c_img_0.90"] = flagged_examples
    res["_arrays"] = (iu, ju, ci, ct, conflict, same_ds, toks)
    return res


def null_control(res, y, iu, ju, ci, toks, seed=NULL_SEED):
    """Label-permutation control: how many conservative conflicting pairs would appear
    if the binary labels were random w.r.t. the geometry."""
    rng = np.random.default_rng(seed)
    m = ci >= GATE_CIMG
    sel = np.where(m)[0]
    jac = np.array([jaccard(toks[iu[k]], toks[ju[k]]) for k in sel])
    jm = jac >= GATE_JACCARD
    counts = []
    for _ in range(200):
        yp = y[rng.permutation(len(y))]
        cf = yp[iu[sel]] != yp[ju[sel]]
        counts.append(int((cf & jm).sum()))
    counts = np.array(counts)
    return {"seed": seed, "n_permutations": 200,
            "conservative_conflicting_mean": float(counts.mean()),
            "conservative_conflicting_p2.5": float(np.percentile(counts, 2.5)),
            "conservative_conflicting_p97.5": float(np.percentile(counts, 97.5)),
            "n_conservative_pairs_pool": int(jm.sum())}


def verdict(nconserv):
    """Transcribed frozen rule (on the conservative count: c_img >= 0.90 and Jaccard >= 0.5):
      ALIVE      -- >= 30 conflicting pairs across the train pools.
      AMBIGUOUS  -- 10-29 conflicting pairs (would need manual verification of a sample).
      DEAD       -- < 10 conflicting pairs.
    """
    if nconserv >= ALIVE_N:
        return "ALIVE"
    if nconserv >= DEAD_N:
        return "AMBIGUOUS"
    return "DEAD"


def strip(res):
    res.pop("_arrays", None)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", choices=["synthetic", "planted"], default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    arm_guard()
    t0 = time.time()
    out = {"pilot": "P-B near-duplicate and label-conflict census",
           "freeze": "idea-stage/PILOT_FREEZE_2026-08-09.md",
           "mode": args.smoke or "real",
           "c_img_thresholds": list(CIMG_THRESHOLDS),
           "gate": {"c_img": GATE_CIMG, "jaccard": GATE_JACCARD,
                    "ALIVE": ">=%d" % ALIVE_N, "DEAD": "<%d" % DEAD_N},
           "guard": "armed: any path containing 'test' (or dev_seen/val) HALTs"}

    if args.smoke in ("synthetic", "planted"):
        rng = np.random.default_rng(11)
        n = 400
        ids = ["syn_%04d" % i for i in range(n)]
        img = rng.normal(size=(n, 1024))
        txt = rng.normal(size=(n, 768))
        y = rng.integers(0, 2, size=n)
        ds = np.array(["A"] * 200 + ["B"] * 200)
        vocab = ["w%d" % i for i in range(500)]
        texts = [" ".join(rng.choice(vocab, 40)) for _ in range(n)]
        planted = []
        if args.smoke == "planted":
            # 12 planted near-duplicate pairs, 8 of them label-conflicting, all with
            # near-identical transcripts (Jaccard ~1)
            for k in range(12):
                a, b = k, 200 + k
                img[b] = img[a] + rng.normal(scale=0.02, size=1024)
                txt[b] = txt[a] + rng.normal(scale=0.02, size=768)
                texts[b] = texts[a]
                y[a], y[b] = (0, 1) if k < 8 else (1, 1)
                planted.append((ids[a], ids[b], int(k < 8)))
        cn_flag = np.zeros(n, dtype=bool)
        cn_known = np.zeros(n, dtype=bool)
        r = census(ids, img, txt, y, ds, texts, cn_flag, cn_known, tag=args.smoke)
        iu, ju, ci, ct, conflict, same_ds, toks = r["_arrays"]
        nc = r["thresholds"][str(GATE_CIMG)]["N4_conservative_conflicting_total"]
        log("SMOKE %s: N1@0.90=%d conflicting=%d conservative=%d verdict=%s "
            "(planted conflicting=%d)"
            % (args.smoke, r["thresholds"]["0.9"]["N1_pairs_total"],
               r["thresholds"]["0.9"]["N2_conflicting_total"], nc, verdict(nc),
               sum(p[2] for p in planted)))
        log("elapsed %.1fs" % (time.time() - t0))
        return

    # ---------------------------- REAL RUN (single submission) ----------------
    (ids, img, txt, y, ds, texts, ds_present, ds_absent, files) = load_pool()
    out["datasets_used"] = ds_present
    out["datasets_excluded_no_train_cache"] = ds_absent
    out["files"] = files
    cn_flag, cn_known = load_mhc_cn_flags(ids, ds)
    out["mhc_counter_narrative_items"] = int(cn_flag.sum())

    r = census(ids, img, txt, y, ds, texts, cn_flag, cn_known, tag="real")
    iu, ju, ci, ct, conflict, same_ds, toks = r["_arrays"]
    out["null_control"] = null_control(r, y, iu, ju, ci, toks)
    out["census"] = strip(r)

    nc = r["thresholds"][str(GATE_CIMG)]["N4_conservative_conflicting_total"]
    out["gate_conservative_count"] = nc
    out["verdict"] = verdict(nc)
    for thr in CIMG_THRESHOLDS:
        b = r["thresholds"][str(thr)]
        log("c_img>=%.2f: N1=%d (within %d / cross %d)  N2_conflict=%d  "
            "Jacc>=0.5=%d  conservative_conflict=%d"
            % (thr, b["N1_pairs_total"], b["N1_pairs_within_dataset"],
               b["N1_pairs_cross_dataset"], b["N2_conflicting_total"],
               b["N4_jaccard_ge_0.5_total"], b["N4_conservative_conflicting_total"]))
    log("NULL (permuted labels): conservative conflicting mean=%.1f [%.0f, %.0f]"
        % (out["null_control"]["conservative_conflicting_mean"],
           out["null_control"]["conservative_conflicting_p2.5"],
           out["null_control"]["conservative_conflicting_p97.5"]))
    log("VERDICT: %s (conservative count = %d)" % (out["verdict"], nc))

    out["paths_touched"] = sorted(set(_TOUCHED))
    out["elapsed_sec"] = time.time() - t0
    if args.out:
        Path(args.out).write_text(json.dumps(out, indent=2, ensure_ascii=False))
        log("wrote %s" % args.out)


if __name__ == "__main__":
    try:
        main()
    except Halt as e:
        log("HALT %s" % e)
        sys.exit(3)
