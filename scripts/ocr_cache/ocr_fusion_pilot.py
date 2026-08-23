#!/usr/bin/env python
"""OCR three-stream fusion head-level pilot ladder (arm0 / arm1 OCR-3 / arm2 OCR-30).

Decision rules are frozen in idea-stage/OCR_FUSION_PILOT_FREEZE.md and are NOT edited
after results. HateMM-train only (744). dev_seen / test are never opened.

Usage:
  python -m scripts.ocr_cache.ocr_fusion_pilot --out idea-stage/ocr_fusion_pilot.json
  python -m scripts.ocr_cache.ocr_fusion_pilot --smoke synthetic
  python -m scripts.ocr_cache.ocr_fusion_pilot --smoke permuted --out /tmp/x.json
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
import torch.nn as nn
from sklearn.model_selection import StratifiedKFold

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.tera_gate0.common import select_threshold  # noqa: E402

RUN = ROOT / "artifacts/tera_gate0/tera-gate0-20260807T000625Z-7ba80eaf"
WHOLE = ROOT / "data/CLIP_Embedding/HateMM/train_openai_clip-vit-large-patch14-336_HF.pt"
OCRW = ROOT / "data/OCR/HateMM/ocr_windows_K30.jsonl"
SHASUMS = ROOT / "data/OCR/SHA256SUMS.json"
CLIP_MODEL = "openai/clip-vit-large-patch14-336"

# ---- frozen constants (OCR_FUSION_PILOT_FREEZE.md) ----
K = 30
MIN_CONF = 0.5
MIN_TEXT_LEN = 2
ARM1_WINDOWS = (5, 15, 25)          # round((i+0.5)*30/3) for i=0,1,2
ARM2_WINDOWS = tuple(range(K))
SEEDS = (20260810, 20260811, 20260812)
INNER_FOLD_SEED = 20260808
N_INNER = 4
LR = 1e-3
WD = 1e-2
BATCH_SIZE = 64
E_MAX = 200
PATIENCE = 40
MIN_DELTA = 1e-4
GO_T = 0.015
AMBIG_T = 0.005
TORCH_THREADS = 8


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


# --------------------------------------------------------------------- data --
def load_base():
    who = torch.load(WHOLE, map_location="cpu")
    raw = who["ids"]
    ids = raw[0] if (len(raw) == 1 and isinstance(raw[0], list)) else raw
    ids = list(ids)
    img = who["img_feats"].numpy().astype(np.float64)
    txt = who["text_feats"].numpy().astype(np.float64)
    y = who["labels"].numpy().astype(np.int64)
    if not (len(ids) == img.shape[0] == txt.shape[0] == y.shape[0]):
        raise Halt("HALT_CACHE_SHAPE")
    idx = {v: i for i, v in enumerate(ids)}
    folds = []
    for f in range(5):
        tr = json.load(open(RUN / ("folds/fold_%d/train_ids.json" % f)))
        qu = json.load(open(RUN / ("folds/fold_%d/query_ids.json" % f)))
        for v in tr + qu:
            if v not in idx:
                raise Halt("HALT_FOLD_ID_NOT_IN_CACHE:" + v)
            if "test" in v.lower():
                raise Halt("HALT_TEST_CONTACT:" + v)
        folds.append((sorted(tr), sorted(qu)))
    covered = sorted({v for tr, qu in folds for v in qu})
    if covered != sorted(ids):
        raise Halt("HALT_FOLD_COVERAGE")
    return ids, idx, img, txt, y, folds


def window_texts(ids):
    """{video_id: [30 strings]} for the given (train) ids, frozen filter applied."""
    want = set(ids)
    got = {}
    with open(OCRW, encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            r = json.loads(line)
            v = r["video_id"]
            if v not in want:
                continue
            k = int(r["window_k"])
            keep = []
            for d in r.get("texts") or []:
                t = (d.get("text") or "").strip()
                if float(d.get("conf", 0.0)) >= MIN_CONF and len(t) >= MIN_TEXT_LEN:
                    keep.append(t)
            got.setdefault(v, {})[k] = " ".join(keep).strip()
    if set(got) != want:
        raise Halt("HALT_OCR_MISSING_VIDEOS:%d" % len(want - set(got)))
    out = {}
    for v, wk in got.items():
        if sorted(wk) != list(range(K)):
            raise Halt("HALT_OCR_WINDOW_COUNT:" + v)
        out[v] = [wk[k] for k in range(K)]
    return out


def encode_texts(texts, device):
    """CLIP text tower pooler_output, 768-d — same recipe as the project's text_feats."""
    from transformers import CLIPTextModel, CLIPTokenizer

    tok = CLIPTokenizer.from_pretrained(CLIP_MODEL)
    mdl = CLIPTextModel.from_pretrained(CLIP_MODEL).eval().to(device)
    uniq = sorted({t for t in texts if t})
    log("encoding %d unique OCR window texts on %s" % (len(uniq), device))
    vecs = {}
    B = 64
    with torch.no_grad():
        for i in range(0, len(uniq), B):
            batch = uniq[i:i + B]
            enc = tok(batch, return_tensors="pt", padding=True, truncation=True)
            enc = {k: v.to(device) for k, v in enc.items()}
            out = mdl(**enc).pooler_output.float().cpu().numpy().astype(np.float64)
            for t, v in zip(batch, out):
                vecs[t] = v
            if (i // B) % 20 == 0:
                log("PROGRESS encode %d/%d" % (i, len(uniq)))
    del mdl
    if device == "cuda":
        torch.cuda.empty_cache()
    return vecs


def ocr_block(ids, wtexts, vecs, windows):
    """Mean of L2-normalized embeddings over the arm's non-empty windows, then L2."""
    V = len(ids)
    out = np.zeros((V, 768), dtype=np.float64)
    n_empty = 0
    for i, v in enumerate(ids):
        rows = [vecs[wtexts[v][k]] for k in windows if wtexts[v][k]]
        if not rows:
            n_empty += 1
            continue
        m = l2np(np.stack(rows), axis=1).mean(axis=0)
        out[i] = m / max(float(np.linalg.norm(m)), 1e-8)
    return out, n_empty


# ------------------------------------------------------------------- model --
class Head(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.head = nn.Linear(d, 1)
        nn.init.normal_(self.head.weight, 0.0, 0.01)
        nn.init.zeros_(self.head.bias)

    def forward(self, x):
        return self.head(x).squeeze(-1)


def derive_seed(*parts):
    s = "|".join(str(p) for p in parts)
    return int(hashlib.sha256(s.encode()).hexdigest()[:8], 16)


def train_epochs(model, X, y, rows, opt, lossfn, scope, e_from, e_to):
    for epoch in range(e_from, e_to):
        gen = torch.Generator()
        gen.manual_seed((derive_seed(scope) + epoch) % (2 ** 31 - 1))
        perm = torch.randperm(len(rows), generator=gen)
        sh = rows[perm]
        model.train()
        for s in range(0, len(sh), BATCH_SIZE):
            b = sh[s:s + BATCH_SIZE]
            opt.zero_grad()
            loss = lossfn(model(X[b]), y[b])
            loss.backward()
            opt.step()


@torch.no_grad()
def score(model, X, rows):
    model.eval()
    return torch.sigmoid(model(X[rows])).double().numpy()


def run_fold(Xnp, ynp, ids, tr_ids, qu_ids, arm, seed, outer, idx):
    """Inner-4-fold lockstep epoch/threshold selection, then refit on full outer-train."""
    X = torch.as_tensor(Xnp, dtype=torch.float32)
    yt = torch.as_tensor(ynp, dtype=torch.float32)
    d = X.shape[1]
    lossfn = nn.BCEWithLogitsLoss()

    order = sorted(tr_ids)
    yo = np.array([ynp[idx[v]] for v in order], dtype=np.int64)
    skf = StratifiedKFold(n_splits=N_INNER, shuffle=True, random_state=INNER_FOLD_SEED)
    arr = np.array(order)
    inner = []
    for a, b in skf.split(np.zeros(len(order)), yo):
        inner.append((sorted(arr[a].tolist()), sorted(arr[b].tolist())))

    models, opts, itr, iva = [], [], [], []
    for j, (a, b) in enumerate(inner):
        torch.manual_seed(derive_seed(seed, arm, outer, j) % (2 ** 31 - 1))
        m = Head(d)
        models.append(m)
        opts.append(torch.optim.AdamW(m.parameters(), lr=LR, weight_decay=WD,
                                      betas=(0.9, 0.999), eps=1e-8, amsgrad=False))
        itr.append(torch.as_tensor([idx[v] for v in a], dtype=torch.long))
        iva.append(torch.as_tensor([idx[v] for v in b], dtype=torch.long))

    val_rows = torch.cat(iva)
    val_y = ynp[val_rows.numpy()]
    best_f1, best_epoch, best_theta, since = -1.0, 1, 0.5, 0
    for epoch in range(E_MAX):
        for j in range(N_INNER):
            train_epochs(models[j], X, yt, itr[j], opts[j], lossfn,
                         (seed, arm, outer, j), epoch, epoch + 1)
        pooled = np.concatenate([score(models[j], X, iva[j]) for j in range(N_INNER)])
        theta, f1 = select_threshold(pooled, val_y)
        if f1 > best_f1 + MIN_DELTA:
            best_f1, best_epoch, best_theta, since = float(f1), epoch + 1, float(theta), 0
        else:
            since += 1
            if since >= PATIENCE:
                break

    torch.manual_seed(derive_seed(seed, arm, outer, "refit") % (2 ** 31 - 1))
    m = Head(d)
    opt = torch.optim.AdamW(m.parameters(), lr=LR, weight_decay=WD,
                            betas=(0.9, 0.999), eps=1e-8, amsgrad=False)
    rows = torch.as_tensor([idx[v] for v in order], dtype=torch.long)
    train_epochs(m, X, yt, rows, opt, lossfn, (seed, arm, outer, "refit"), 0, best_epoch)
    qrows = torch.as_tensor([idx[v] for v in sorted(qu_ids)], dtype=torch.long)
    s = score(m, X, qrows)
    return qrows.numpy(), (s >= best_theta).astype(np.int64), best_epoch, best_theta, best_f1


def run_arm(Xnp, ynp, ids, folds, arm, seed, idx):
    pred = np.full(len(ids), -1, dtype=np.int64)
    info = []
    for outer, (tr_ids, qu_ids) in enumerate(folds):
        t0 = time.time()
        qr, p, ep, th, inf1 = run_fold(Xnp, ynp, ids, tr_ids, qu_ids, arm, seed, outer, idx)
        pred[qr] = p
        info.append({"outer": outer, "epoch": ep, "theta": th, "inner_macro_f1": inf1,
                     "seconds": round(time.time() - t0, 1)})
        log("PROGRESS arm=%d seed=%d fold=%d epoch=%d theta=%.4f dt=%.1fs"
            % (arm, seed, outer, ep, th, time.time() - t0))
    if (pred < 0).any():
        raise Halt("HALT_INCOMPLETE_OOF")
    return macro_f1(ynp, pred), pred, info


# -------------------------------------------------------------------- main --
def build_features(smoke):
    if smoke == "synthetic":
        rng = np.random.default_rng(0)
        n = 200
        ids = ["v%03d" % i for i in range(n)]
        y = (rng.random(n) < 0.4).astype(np.int64)
        img = rng.normal(size=(n, 1024)) + 0.3 * y[:, None]
        txt = rng.normal(size=(n, 768)) + 0.3 * y[:, None]
        o3 = rng.normal(size=(n, 768))
        o30 = rng.normal(size=(n, 768)) + 0.5 * y[:, None]
        idx = {v: i for i, v in enumerate(ids)}
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=1)
        arr = np.array(ids)
        folds = [(sorted(arr[a].tolist()), sorted(arr[b].tolist()))
                 for a, b in skf.split(np.zeros(n), y)]
        X = {0: np.hstack([l2np(img), l2np(txt)]),
             1: np.hstack([l2np(img), l2np(txt), l2np(o3)]),
             2: np.hstack([l2np(img), l2np(txt), l2np(o30)])}
        return ids, idx, y, folds, X, {"mode": "synthetic"}

    ids, idx, img, txt, y, folds = load_base()
    have = json.load(open(SHASUMS))["data/OCR/HateMM/ocr_windows_K30.jsonl"]
    got = sha256_file(OCRW)
    if got != have:
        raise Halt("HALT_OCR_CACHE_SHA:%s" % got)
    log("OCR cache sha256 verified")
    wtexts = window_texts(ids)
    cache = ROOT / "data/OCR/HateMM/pilot_ocr_blocks.npz"
    device = "cached"
    if cache.exists():
        z = np.load(cache, allow_pickle=True)
        if str(z["sha"]) == got and list(z["ids"]) == list(ids):
            o3, o30 = z["o3"], z["o30"]
            e3, e30 = int(z["e3"]), int(z["e30"])
            nuniq = int(z["nuniq"])
            log("reused OCR block cache %s" % cache)
        else:
            raise Halt("HALT_OCR_BLOCK_CACHE_STALE")
    else:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        allt = [wtexts[v][k] for v in ids for k in range(K)]
        vecs = encode_texts(allt, device)
        o3, e3 = ocr_block(ids, wtexts, vecs, ARM1_WINDOWS)
        o30, e30 = ocr_block(ids, wtexts, vecs, ARM2_WINDOWS)
        nuniq = len(vecs)
        np.savez(cache, o3=o3, o30=o30, e3=e3, e30=e30, nuniq=nuniq,
                 sha=got, ids=np.array(ids, dtype=object))
    log("OCR blocks built: arm1 all-zero videos=%d/%d, arm2 all-zero videos=%d/%d"
        % (e3, len(ids), e30, len(ids)))
    if smoke == "permuted":
        rng = np.random.default_rng(12345)
        y = y[rng.permutation(len(y))]
        log("SMOKE: labels permuted")
    base = np.hstack([l2np(img), l2np(txt)])
    X = {0: base, 1: np.hstack([base, o3]), 2: np.hstack([base, o30])}
    meta = {"mode": "real" if smoke is None else "permuted",
            "n_videos": len(ids), "n_pos": int(y.sum()),
            "arm1_zero_ocr_videos": e3, "arm2_zero_ocr_videos": e30,
            "n_unique_window_texts": nuniq,
            "ocr_windows_sha256": got, "device_text_encoder": device,
            "dims": {str(a): int(X[a].shape[1]) for a in X}}
    return ids, idx, y, folds, X, meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--smoke", choices=["synthetic", "permuted"], default=None)
    ap.add_argument("--seeds", type=int, nargs="*", default=list(SEEDS))
    a = ap.parse_args()
    torch.set_num_threads(TORCH_THREADS)

    ids, idx, y, folds, X, meta = build_features(a.smoke)
    log("features ready: %s" % json.dumps(meta.get("dims", {})))

    res = {"meta": meta, "seeds": a.seeds, "arms": {}}
    per = {0: [], 1: [], 2: []}
    for arm in (0, 1, 2):
        rows = []
        for seed in a.seeds:
            f1, pred, info = run_arm(X[arm], y, ids, folds, arm, seed, idx)
            per[arm].append(f1)
            rows.append({"seed": seed, "oof_macro_f1": f1, "folds": info})
            log("RESULT arm=%d seed=%d oof_macro_f1=%.4f" % (arm, seed, f1))
        res["arms"][str(arm)] = {"per_seed": rows,
                                 "mean": float(np.mean(per[arm])),
                                 "std": float(np.std(per[arm], ddof=1)) if len(per[arm]) > 1 else 0.0}

    d1 = float(np.mean(per[1]) - np.mean(per[0]))
    d2 = float(np.mean(per[2]) - np.mean(per[0]))
    paired1 = [b - c for b, c in zip(per[1], per[0])]
    paired2 = [b - c for b, c in zip(per[2], per[0])]
    verdict = "GO" if d2 >= GO_T else ("AMBIGUOUS" if d2 >= AMBIG_T else "NO-GO")
    res["deltas"] = {"arm1_minus_arm0": d1, "arm2_minus_arm0": d2,
                     "arm2_minus_arm1": float(np.mean(per[2]) - np.mean(per[1])),
                     "paired_arm1_minus_arm0": paired1,
                     "paired_arm2_minus_arm0": paired2,
                     "paired_arm1_mean": float(np.mean(paired1)),
                     "paired_arm1_std": float(np.std(paired1, ddof=1)) if len(paired1) > 1 else 0.0,
                     "paired_arm2_mean": float(np.mean(paired2)),
                     "paired_arm2_std": float(np.std(paired2, ddof=1)) if len(paired2) > 1 else 0.0}
    res["verdict"] = verdict
    res["verdict_rule"] = "arm2-arm0 >= %.3f GO; >= %.3f AMBIGUOUS; else NO-GO" % (GO_T, AMBIG_T)
    log("VERDICT arm0=%.4f arm1=%.4f arm2=%.4f d1=%+.4f d2=%+.4f -> %s"
        % (np.mean(per[0]), np.mean(per[1]), np.mean(per[2]), d1, d2, verdict))
    if a.out:
        a.out.parent.mkdir(parents=True, exist_ok=True)
        json.dump(res, open(a.out, "w"), indent=1)
        log("wrote %s" % a.out)


if __name__ == "__main__":
    main()
