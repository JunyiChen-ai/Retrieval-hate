#!/usr/bin/env python
"""Late-interaction segment-level retrieval pilot (arm0 / arm0v / arm1 LI-visual / arm2 LI-visual+OCR).

Decision rules are frozen in idea-stage/LI_RETRIEVAL_PILOT_FREEZE.md and are NOT edited
after results. HateMM-train only (744). dev_seen / test are never opened.

Usage:
  python idea-stage/li_retrieval_pilot.py --out idea-stage/li_retrieval_pilot.json
  python idea-stage/li_retrieval_pilot.py --smoke synthetic
  python idea-stage/li_retrieval_pilot.py --smoke permuted --out /tmp/x.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.model_selection import StratifiedKFold

ROOT = Path("/home/jehc223/Retrieval-hate")
sys.path.insert(0, str(ROOT))
from scripts.tera_gate0.common import select_threshold  # noqa: E402

RUN = ROOT / "artifacts/tera_gate0/tera-gate0-20260807T000625Z-7ba80eaf"
SEG = ROOT / "data/CLIP_Embedding/HateMM/train_subclipK30_openai_clip-vit-large-patch14-336_HF.pt"
WHOLE = ROOT / "data/CLIP_Embedding/HateMM/train_openai_clip-vit-large-patch14-336_HF.pt"
OCRW = ROOT / "data/OCR/HateMM/ocr_windows_K30.jsonl"
SHASUMS = ROOT / "data/OCR/SHA256SUMS.json"
OCR_BLOCKS = ROOT / "data/OCR/HateMM/pilot_ocr_blocks.npz"
OCR_WINVECS = ROOT / "data/OCR/HateMM/pilot_ocr_window_vecs.npz"
CLIP_MODEL = "openai/clip-vit-large-patch14-336"

# ---- frozen constants (LI_RETRIEVAL_PILOT_FREEZE.md) ----
K = 30
KNN = 10
MIN_CONF = 0.5
MIN_TEXT_LEN = 2
ARM1_WINDOWS = (5, 15, 25)          # only used for the OCR-vector self-check
ARM2_WINDOWS = tuple(range(K))
SPLIT_SEEDS = (None, 20260901, 20260902)   # None = the frozen Gate-0 split
BOOT = 2000
BOOT_SEED = 20260903
OCR_SELFCHECK_TOL = 1e-3
GO_F1 = 0.005
GO_PURITY = 0.020
CHUNK_Q = 24
TORCH_THREADS = 8
ARMS = ("0", "0v", "1", "2")


class Halt(RuntimeError):
    pass


def log(msg):
    print("[%s] %s" % (time.strftime("%H:%M:%S"), msg), flush=True)


def l2np(x, axis=-1):
    return x / np.maximum(np.linalg.norm(x, axis=axis, keepdims=True), 1e-8)


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def macro_f1(y, pred):
    y = np.asarray(y, dtype=np.int64)
    pred = np.asarray(pred, dtype=np.int64)
    fs = []
    for c in (0, 1):
        tp = float(((pred == c) & (y == c)).sum())
        fp = float(((pred == c) & (y != c)).sum())
        fn = float(((pred != c) & (y == c)).sum())
        fs.append(0.0 if tp == 0 else 2 * tp / (2 * tp + fp + fn))
    return float(np.mean(fs))


def guard_ids(ids):
    for v in ids:
        lv = str(v).lower()
        if "test" in lv or "dev_seen" in lv:
            raise Halt("HALT_TEST_CONTACT:" + str(v))


# --------------------------------------------------------------------- data --
def load_base():
    who = torch.load(WHOLE, map_location="cpu")
    raw = who["ids"]
    ids = raw[0] if (len(raw) == 1 and isinstance(raw[0], list)) else raw
    ids = [str(v) for v in ids]
    guard_ids(ids)
    img = who["img_feats"].numpy().astype(np.float64)
    txt = who["text_feats"].numpy().astype(np.float64)
    y = who["labels"].numpy().astype(np.int64)
    if not (len(ids) == img.shape[0] == txt.shape[0] == y.shape[0]):
        raise Halt("HALT_CACHE_SHAPE")

    seg = torch.load(SEG, map_location="cpu")
    sids = [str(v) for v in seg["video_ids"]]
    if sids != ids:
        raise Halt("HALT_CACHE_ORDER")
    Sf = seg["subclip_img_feats"].numpy().astype(np.float64)
    if Sf.shape[0] != len(ids) * K:
        raise Halt("HALT_CACHE_SHAPE:seg=%s" % (Sf.shape,))
    S = Sf.reshape(len(ids), K, Sf.shape[1])
    log("loaded: %d videos, seg %s, img %s, txt %s, pos=%d"
        % (len(ids), S.shape, img.shape, txt.shape, int(y.sum())))
    return ids, img, txt, y, S


def make_splits(ids, y, synthetic=False):
    """[(name, [(train_idx, query_idx) x 5]) x 3]; split 0 = frozen Gate-0 folds."""
    idx = {v: i for i, v in enumerate(ids)}
    seeds = (20260900, 20260901, 20260902) if synthetic else SPLIT_SEEDS
    out = []
    for si, seed in enumerate(seeds):
        folds = []
        if seed is None:
            name = "gate0_20260807"
            for f in range(5):
                tr = json.load(open(RUN / ("folds/fold_%d/train_ids.json" % f)))
                qu = json.load(open(RUN / ("folds/fold_%d/query_ids.json" % f)))
                guard_ids(tr)
                guard_ids(qu)
                for v in tr + qu:
                    if v not in idx:
                        raise Halt("HALT_FOLD_ID_NOT_IN_CACHE:" + v)
                folds.append((np.array(sorted(idx[v] for v in tr)),
                              np.array(sorted(idx[v] for v in qu))))
        else:
            name = "skf_%d" % seed
            order = np.argsort(np.array(ids))          # sorted-id order, deterministic
            skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
            for a, b in skf.split(np.zeros(len(order)), y[order]):
                folds.append((np.sort(order[a]), np.sort(order[b])))
        cov = np.sort(np.concatenate([q for _, q in folds]))
        if not np.array_equal(cov, np.arange(len(ids))):
            raise Halt("HALT_FOLD_COVERAGE:split%d" % si)
        for tr, qu in folds:
            if np.intersect1d(tr, qu).size:
                raise Halt("HALT_FOLD_OVERLAP:split%d" % si)
        out.append((name, folds))
    return out


def window_texts(ids):
    """{video_id: [30 strings]} with the frozen conf/len filter applied."""
    want = set(ids)
    got = {}
    with open(OCRW, encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            r = json.loads(line)
            v = str(r["video_id"])
            if v not in want:
                continue
            keep = []
            for d in r.get("texts") or []:
                t = (d.get("text") or "").strip()
                if float(d.get("conf", 0.0)) >= MIN_CONF and len(t) >= MIN_TEXT_LEN:
                    keep.append(t)
            got.setdefault(v, {})[int(r["window_k"])] = " ".join(keep).strip()
    if set(got) != want:
        raise Halt("HALT_OCR_MISSING_VIDEOS:%d" % len(want - set(got)))
    out = {}
    for v, wk in got.items():
        if sorted(wk) != list(range(K)):
            raise Halt("HALT_OCR_WINDOW_COUNT:" + v)
        out[v] = [wk[k] for k in range(K)]
    return out


def encode_texts(uniq, device):
    from transformers import CLIPTextModel, CLIPTokenizer

    tok = CLIPTokenizer.from_pretrained(CLIP_MODEL)
    mdl = CLIPTextModel.from_pretrained(CLIP_MODEL).eval().to(device)
    log("encoding %d unique OCR window texts on %s" % (len(uniq), device))
    vecs = np.zeros((len(uniq), 768), dtype=np.float64)
    B = 64
    with torch.no_grad():
        for i in range(0, len(uniq), B):
            batch = uniq[i:i + B]
            enc = tok(batch, return_tensors="pt", padding=True, truncation=True)
            enc = {k: v.to(device) for k, v in enc.items()}
            vecs[i:i + len(batch)] = mdl(**enc).pooler_output.float().cpu().numpy().astype(np.float64)
            if (i // B) % 20 == 0:
                log("PROGRESS encode %d/%d" % (i, len(uniq)))
    del mdl
    if device == "cuda":
        torch.cuda.empty_cache()
    return vecs


def ocr_segment_block(ids, sha):
    """[V, K, 768] L2-normalized per-window OCR text embedding; all-zero where the window is empty."""
    wtexts = window_texts(ids)
    allt = [wtexts[v][k] for v in ids for k in range(K)]
    uniq = sorted({t for t in allt if t})

    vecs = None
    if OCR_WINVECS.exists():
        z = np.load(OCR_WINVECS, allow_pickle=True)
        if str(z["sha"]) == sha and list(z["texts"]) == uniq:
            vecs = z["vecs"]
            log("reused OCR window-vector cache %s" % OCR_WINVECS)
    if vecs is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        try:
            vecs = encode_texts(uniq, device)
        except RuntimeError as exc:                       # e.g. GPU busy / OOM
            if device != "cuda":
                raise
            log("cuda encode failed (%s); falling back to cpu" % exc)
            vecs = encode_texts(uniq, "cpu")
        np.savez(OCR_WINVECS, vecs=vecs, texts=np.array(uniq, dtype=object), sha=sha)
        log("wrote %s" % OCR_WINVECS)

    pos = {t: i for i, t in enumerate(uniq)}
    vn = l2np(vecs, axis=1)
    V = len(ids)
    O = np.zeros((V, K, 768), dtype=np.float64)
    n_empty_win = 0
    for i, v in enumerate(ids):
        for k in range(K):
            t = wtexts[v][k]
            if t:
                O[i, k] = vn[pos[t]]
            else:
                n_empty_win += 1
    return O, n_empty_win, len(uniq)


def ocr_selfcheck(ids, O, wtext_nonempty):
    """Re-aggregate per-window vectors with the OCR-fusion freeze rule and match pilot_ocr_blocks.npz."""
    if not OCR_BLOCKS.exists():
        log("WARN: %s absent, self-check skipped" % OCR_BLOCKS)
        return None
    z = np.load(OCR_BLOCKS, allow_pickle=True)
    if list(z["ids"]) != list(ids):
        raise Halt("HALT_OCR_VEC_MISMATCH:id_order")
    devs = {}
    for name, wins in (("o3", ARM1_WINDOWS), ("o30", ARM2_WINDOWS)):
        rec = np.zeros((len(ids), 768), dtype=np.float64)
        for i in range(len(ids)):
            rows = [O[i, k] for k in wins if wtext_nonempty[i, k]]
            if not rows:
                continue
            m = np.stack(rows).mean(axis=0)             # rows are already L2-normalized
            rec[i] = m / max(float(np.linalg.norm(m)), 1e-8)
        d = float(np.abs(rec - z[name]).max())
        devs[name] = d
        if d >= OCR_SELFCHECK_TOL:
            raise Halt("HALT_OCR_VEC_MISMATCH:%s max_abs=%.3e" % (name, d))
    log("OCR window-vector self-check passed (max|d| o3=%.2e o30=%.2e)" % (devs["o3"], devs["o30"]))
    return devs


# -------------------------------------------------------------- similarity --
def maxsim_matrix(A):
    """A: [V,K,d] (blocks already L2-normalized). Returns [V,V] mean_k max_j <A[q,k], A[m,j]>."""
    V, Kk, d = A.shape
    F = np.ascontiguousarray(A.reshape(V * Kk, d).astype(np.float32))
    out = np.empty((V, V), dtype=np.float64)
    t0 = time.time()
    for s in range(0, V, CHUNK_Q):
        e = min(s + CHUNK_Q, V)
        sim = F[s * Kk:e * Kk] @ F.T                       # [(e-s)*K, V*K]
        sim = sim.reshape(e - s, Kk, V, Kk)
        out[s:e] = sim.max(axis=3).mean(axis=1).astype(np.float64)
        if (s // CHUNK_Q) % 5 == 0:
            log("PROGRESS maxsim %d/%d dt=%.1fs" % (e, V, time.time() - t0))
    return out


def build_similarities(img, txt, S, O):
    sims = {}
    gi, gt = l2np(img, 1), l2np(txt, 1)
    ci = gi @ gi.T
    sims["0v"] = ci
    sims["0"] = ci + gt @ gt.T
    Sn = l2np(S, 2)
    log("building arm1 MaxSim (visual, d=%d)" % Sn.shape[2])
    sims["1"] = maxsim_matrix(Sn)
    log("building arm2 MaxSim (visual+OCR, d=%d)" % (Sn.shape[2] + O.shape[2]))
    sims["2"] = maxsim_matrix(np.concatenate([Sn, O], axis=2))
    return sims


# ---------------------------------------------------------------- endpoints --
def topk_rows(sim_block, rank_mem, k):
    """sim_block: [nq, nm]. Returns [nq,k] column indices, ties broken by lexicographic id rank."""
    nq, nm = sim_block.shape
    kk = min(k, nm)
    out = np.empty((nq, kk), dtype=np.int64)
    for i in range(nq):
        order = np.lexsort((rank_mem, -sim_block[i]))
        out[i] = order[:kk]
    return out


def knn_scores(sim_block, rank_mem, ymem, k):
    nb = topk_rows(sim_block, rank_mem, k)
    w = np.maximum(np.take_along_axis(sim_block, nb, axis=1), 0.0)
    yb = ymem[nb].astype(np.float64)
    den = w.sum(axis=1)
    sc = np.where(den > 0, (w * yb).sum(axis=1) / np.maximum(den, 1e-12), yb.mean(axis=1))
    return sc, nb, yb


def eval_arm(sim, y, folds, rank_all):
    """Returns per-video purity, per-video chance, OOF predictions, per-fold theta."""
    V = len(y)
    purity = np.full(V, np.nan)
    chance = np.full(V, np.nan)
    pred = np.full(V, -1, dtype=np.int64)
    thetas = []
    for tr, qu in folds:
        rank_mem = rank_all[tr]
        ymem = y[tr]

        # --- purity + query scoring
        blk = sim[np.ix_(qu, tr)]
        sc, nb, yb = knn_scores(blk, rank_mem, ymem, KNN)
        purity[qu] = (yb == y[qu][:, None]).mean(axis=1)
        pfrac = float((ymem == 1).mean())
        chance[qu] = np.where(y[qu] == 1, pfrac, 1.0 - pfrac)

        # --- threshold from memory-side leave-one-out only
        mm = sim[np.ix_(tr, tr)].copy()
        np.fill_diagonal(mm, -np.inf)
        sc_mem, _, _ = knn_scores(mm, rank_mem, ymem, KNN)
        th, _ = select_threshold(sc_mem, ymem)
        thetas.append(float(th))
        pred[qu] = (sc >= th).astype(np.int64)
    if (pred < 0).any() or np.isnan(purity).any():
        raise Halt("HALT_INCOMPLETE_OOF")
    return purity, chance, pred, thetas


# -------------------------------------------------------------------- main --
def build_inputs(smoke):
    if smoke == "synthetic":
        rng = np.random.default_rng(0)
        n = 120
        ids = ["v%03d" % i for i in range(n)]
        y = (rng.random(n) < 0.4).astype(np.int64)
        img = rng.normal(size=(n, 64)) + 0.6 * y[:, None]
        txt = rng.normal(size=(n, 48)) + 0.6 * y[:, None]
        S = rng.normal(size=(n, K, 64)) + 0.5 * y[:, None, None]
        O = l2np(rng.normal(size=(n, K, 32)) + 0.8 * y[:, None, None], 2)
        O[rng.random((n, K)) < 0.3] = 0.0
        meta = {"mode": "synthetic", "n_videos": n}
        return ids, img, txt, y, S, O, meta

    ids, img, txt, y, S = load_base()
    have = json.load(open(SHASUMS))["data/OCR/HateMM/ocr_windows_K30.jsonl"]
    got = sha256_file(OCRW)
    if got != have:
        raise Halt("HALT_OCR_CACHE_SHA:%s" % got)
    log("OCR cache sha256 verified")
    O, n_empty_win, nuniq = ocr_segment_block(ids, got)
    nonempty = np.linalg.norm(O, axis=2) > 0
    devs = ocr_selfcheck(ids, O, nonempty)
    n_zero_video = int((~nonempty.any(axis=1)).sum())
    log("OCR segment blocks: %d/%d empty windows (%.1f%%), %d/%d videos with no OCR at all"
        % (n_empty_win, len(ids) * K, 100.0 * n_empty_win / (len(ids) * K), n_zero_video, len(ids)))
    if smoke == "permuted":
        rng = np.random.default_rng(12345)
        y = y[rng.permutation(len(y))]
        log("SMOKE: labels permuted")
    meta = {"mode": "real" if smoke is None else "permuted",
            "n_videos": len(ids), "n_pos": int(y.sum()), "K": K, "knn_k": KNN,
            "seg_dim": int(S.shape[2]), "ocr_dim": int(O.shape[2]),
            "ocr_empty_windows": n_empty_win, "ocr_videos_all_empty": n_zero_video,
            "ocr_unique_window_texts": nuniq,
            "ocr_windows_sha256": got, "ocr_selfcheck_max_abs_dev": devs}
    return ids, img, txt, y, S, O, meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--smoke", choices=["synthetic", "permuted"], default=None)
    a = ap.parse_args()
    torch.set_num_threads(TORCH_THREADS)

    ids, img, txt, y, S, O, meta = build_inputs(a.smoke)
    rank_all = np.empty(len(ids), dtype=np.int64)
    rank_all[np.argsort(np.array(ids))] = np.arange(len(ids))
    splits = make_splits(ids, y, synthetic=(a.smoke == "synthetic"))
    log("splits ready: %s" % ", ".join(n for n, _ in splits))

    sims = build_similarities(img, txt, S, O)
    log("similarity matrices ready: %s" % {k: str(v.shape) for k, v in sims.items()})

    res = {"meta": meta, "splits": [], "arms_order": list(ARMS)}
    f1 = {arm: [] for arm in ARMS}
    purity_v = {}
    for si, (sname, folds) in enumerate(splits):
        entry = {"split": si, "name": sname, "arms": {}}
        for arm in ARMS:
            t0 = time.time()
            pu, ch, pred, thetas = eval_arm(sims[arm], y, folds, rank_all)
            m = macro_f1(y, pred)
            f1[arm].append(m)
            if si == 0:
                purity_v[arm] = pu
                purity_v["_chance"] = ch
            entry["arms"][arm] = {
                "macro_f1": m,
                "purity_at10": float(pu.mean()),
                "chance_at10": float(ch.mean()),
                "purity_lift": float((pu - ch).mean()),
                "thetas": thetas,
                "pred_pos_rate": float(pred.mean()),
            }
            log("PROGRESS split=%d(%s) arm=%s macroF1=%.4f purity=%.4f lift=%+.4f dt=%.1fs"
                % (si, sname, arm, m, pu.mean(), (pu - ch).mean(), time.time() - t0))
        res["splits"].append(entry)

    # ---- bootstrap on split 0 purity (paired, shared index draws)
    rng = np.random.default_rng(BOOT_SEED)
    V = len(ids)
    draws = rng.integers(0, V, size=(BOOT, V))
    boot = {}
    for a_hi, a_lo in (("2", "0"), ("1", "0v"), ("2", "1"), ("0v", "0"), ("1", "0")):
        d = purity_v[a_hi] - purity_v[a_lo]
        bs = d[draws].mean(axis=1)
        boot["%s_minus_%s" % (a_hi, a_lo)] = {
            "delta": float(d.mean()),
            "ci95": [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))],
        }
    for arm in ARMS:
        d = purity_v[arm] - purity_v["_chance"]
        bs = d[draws].mean(axis=1)
        boot["lift_arm%s" % arm] = {
            "delta": float(d.mean()),
            "ci95": [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))],
        }
    res["purity_bootstrap_split0"] = boot

    # ---- deltas
    f1m = {arm: float(np.mean(f1[arm])) for arm in ARMS}
    paired = {}
    for a_hi, a_lo in (("2", "0"), ("1", "0v"), ("2", "1"), ("0v", "0"), ("1", "0")):
        per = [f1[a_hi][i] - f1[a_lo][i] for i in range(len(splits))]
        paired["%s_minus_%s" % (a_hi, a_lo)] = {"per_split": per, "mean": float(np.mean(per))}
    res["macro_f1_mean"] = f1m
    res["macro_f1_per_split"] = {arm: f1[arm] for arm in ARMS}
    res["macro_f1_deltas"] = paired

    # ---- FROZEN verdict
    d_f1 = paired["2_minus_0"]
    crit_a = (d_f1["mean"] >= GO_F1) and all(v > 0 for v in d_f1["per_split"])
    pu20 = boot["2_minus_0"]
    crit_b = (pu20["delta"] >= GO_PURITY) and (pu20["ci95"][0] > 0)
    if crit_a or crit_b:
        verdict = "GO"
    elif d_f1["mean"] <= 0 and pu20["delta"] <= 0:
        verdict = "NO-GO"
    else:
        verdict = "AMBIGUOUS"
    res["verdict"] = verdict
    res["verdict_detail"] = {
        "criterion_A_performance": {"met": bool(crit_a), "mean_delta_f1": d_f1["mean"],
                                    "per_split": d_f1["per_split"], "bar": GO_F1},
        "criterion_B_mechanism": {"met": bool(crit_b), "purity_delta_split0": pu20["delta"],
                                  "ci95": pu20["ci95"], "bar": GO_PURITY},
    }
    log("VERDICT %s | f1 arm0=%.4f arm0v=%.4f arm1=%.4f arm2=%.4f | d(2-0)=%+.4f per-split=%s"
        % (verdict, f1m["0"], f1m["0v"], f1m["1"], f1m["2"], d_f1["mean"],
           ["%+.4f" % v for v in d_f1["per_split"]]))
    log("VERDICT purity split0: arm0=%.4f arm0v=%.4f arm1=%.4f arm2=%.4f | d(2-0)=%+.4f CI%s"
        % (res["splits"][0]["arms"]["0"]["purity_at10"],
           res["splits"][0]["arms"]["0v"]["purity_at10"],
           res["splits"][0]["arms"]["1"]["purity_at10"],
           res["splits"][0]["arms"]["2"]["purity_at10"],
           pu20["delta"], ["%+.4f" % v for v in pu20["ci95"]]))

    if a.out:
        a.out.parent.mkdir(parents=True, exist_ok=True)
        json.dump(res, open(a.out, "w"), indent=1)
        log("wrote %s" % a.out)


if __name__ == "__main__":
    main()
