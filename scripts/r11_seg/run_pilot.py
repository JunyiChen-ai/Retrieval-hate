#!/usr/bin/env python
"""R11-SEG pilot runner. Executes exactly the design frozen in idea-stage/R11_SEG_PILOT_FREEZE.md.

Single submission. Writes idea-stage/r11_seg/out/results.json.
"""
from __future__ import annotations

import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path("/home/jehc223/Retrieval-hate")
OUT = ROOT / "idea-stage/r11_seg/out"
K = 30
SEEDS = list(range(2200, 2212))
BOOT_SEED = 2299
SHIFT_SEED = 2298
N_BOOT = 10000
EPOCHS = 40
LR = 1e-3
WD = 1e-2
BATCH = 32
LAMBDA_SMOOTH = 0.15
TAU = 4.0
DEV = "cuda"


# ------------------------------------------------------------------ data
def load_all():
    g = np.load(OUT / "grid_labels.npz", allow_pickle=True)
    vids = [str(v) for v in g["video_ids"]]
    y_win = g["y_win"].astype(np.int64)
    y_change = g["y_change"].astype(np.int64)
    y_video = g["y_video"].astype(np.int64)
    y_ts = list(g["y_ts"])
    win_of_ts = list(g["win_of_ts"])

    ct = torch.load(
        ROOT / "data/CLIP_Embedding/HateClipSeg/test_seen_subclipK30_openai_clip-vit-large-patch14-336_HF.pt",
        map_location="cpu",
    )
    assert list(ct["video_ids"]) == vids
    V = ct["subclip_img_feats"].float().numpy().reshape(len(vids), K, -1)

    t = np.load(OUT / "text_feats.npz")
    assert [str(x) for x in t["video_ids"]] == vids
    T, O = t["asr_feat"], t["ocr_feat"]
    Tm, Om = t["asr_mask"].astype(np.float32), t["ocr_mask"].astype(np.float32)

    a = np.load(OUT / "audio_feats.npz", allow_pickle=True)
    assert [str(x) for x in a["video_ids"]] == vids
    A, E = a["w2v"], a["egemaps"]

    def l2(x):
        n = np.linalg.norm(x, axis=-1, keepdims=True)
        return x / np.maximum(n, 1e-8)

    chans = {"V": l2(V), "T": l2(T), "O": l2(O), "A": l2(A), "E": E.copy()}
    ALL = np.concatenate([chans["V"], chans["T"], chans["O"], chans["A"],
                          Tm[..., None], Om[..., None]], axis=-1).astype(np.float32)

    split = json.loads((ROOT / "data/gt/HateClipSeg/p11_split.json").read_text())
    idx = {v: i for i, v in enumerate(vids)}
    tr = np.array([idx[v] for v in split["train"]])
    va = np.array([idx[v] for v in split["val"]])
    te = np.array([idx[v] for v in split["test"]])
    assert len(set(tr) & set(va)) == 0 and len(set(tr) & set(te)) == 0 and len(set(va) & set(te)) == 0
    assert len(tr) == 237 and len(va) == 39 and len(te) == 119
    print(f"[guard] split disjoint OK  train={len(tr)} val={len(va)} test={len(te)}", flush=True)

    return dict(vids=vids, y_win=y_win, y_change=y_change, y_video=y_video,
                y_ts=y_ts, win_of_ts=win_of_ts, chans=chans, ALL=ALL,
                tr=tr, va=va, te=te)


def zscore(X, tr):
    mu = X[tr].reshape(-1, X.shape[-1]).mean(0)
    sd = X[tr].reshape(-1, X.shape[-1]).std(0)
    sd = np.maximum(sd, 1e-6)
    return ((X - mu) / sd).astype(np.float32)


# ------------------------------------------------------------------ models
class Proj(nn.Module):
    def __init__(self, d_in, d=256):
        super().__init__()
        self.lin = nn.Linear(d_in, d)
        self.drop = nn.Dropout(0.1)

    def forward(self, x):  # [B,T,D]
        return self.drop(F.gelu(self.lin(x)))


class PerWin(nn.Module):
    def __init__(self, d_in):
        super().__init__()
        self.proj = Proj(d_in)
        self.head = nn.Sequential(nn.Linear(256, 256), nn.GELU(), nn.Linear(256, 2))

    def forward(self, x):
        return self.head(self.proj(x))


class BcastCausal(nn.Module):
    def __init__(self, d_in):
        super().__init__()
        self.proj = Proj(d_in)
        self.head = nn.Linear(256, 2)

    def forward(self, x):
        h = self.proj(x)
        c = torch.cumsum(h, dim=1) / torch.arange(1, h.shape[1] + 1, device=h.device).view(1, -1, 1)
        return self.head(c)


class BcastVideo(nn.Module):
    def __init__(self, d_in):
        super().__init__()
        self.proj = Proj(d_in)
        self.head = nn.Linear(256, 2)

    def forward(self, x):
        h = self.proj(x).mean(1, keepdim=True)
        return self.head(h).expand(-1, x.shape[1], -1)


class MILTopK(nn.Module):
    """video-label-only supervision; per-window logits pooled by top-33%."""

    def __init__(self, d_in, frac=1 / 3):
        super().__init__()
        self.inner = PerWin(d_in)
        self.frac = frac

    def forward(self, x):
        return self.inner(x)

    def video_logit(self, win_logits):
        s = win_logits[..., 1] - win_logits[..., 0]
        k = max(1, int(round(self.frac * s.shape[1])))
        top = torch.topk(s, k, dim=1).values.mean(1)
        return torch.stack([-top / 2, top / 2], dim=-1)


class DilatedResidual(nn.Module):
    def __init__(self, c, d):
        super().__init__()
        self.d = d
        self.conv = nn.Conv1d(c, c, 3, dilation=d)
        self.conv1 = nn.Conv1d(c, c, 1)
        self.drop = nn.Dropout(0.1)

    def forward(self, x):  # [B,C,T]
        h = F.pad(x, (2 * self.d, 0))  # causal
        h = F.relu(self.conv(h))
        h = self.drop(self.conv1(h))
        return x + h


class Stage(nn.Module):
    def __init__(self, d_in, c=64, L=5, n_cls=2):
        super().__init__()
        self.inc = nn.Conv1d(d_in, c, 1)
        self.layers = nn.ModuleList([DilatedResidual(c, 2 ** l) for l in range(L)])
        self.out = nn.Conv1d(c, n_cls, 1)

    def forward(self, x):  # [B,Cin,T]
        h = self.inc(x)
        for lay in self.layers:
            h = lay(h)
        return self.out(h)


class CTCN(nn.Module):
    """MS-TCN-style causal multi-stage TCN. No background class, no top-k, no softmax-over-time."""

    def __init__(self, d_in, S=2, c=64, L=5):
        super().__init__()
        self.proj = Proj(d_in)
        self.stages = nn.ModuleList([Stage(256 if i == 0 else 2, c, L) for i in range(S)])

    def forward(self, x):
        h = self.proj(x).transpose(1, 2)
        outs = []
        cur = h
        for i, st in enumerate(self.stages):
            o = st(cur)
            outs.append(o.transpose(1, 2))
            cur = F.softmax(o, dim=1)
        self._all = outs
        return outs[-1]


class CTrans(nn.Module):
    def __init__(self, d_in, nlayer=2, nhead=4):
        super().__init__()
        self.proj = Proj(d_in)
        self.pos = nn.Parameter(torch.zeros(1, K, 256))
        nn.init.normal_(self.pos, std=0.02)
        layer = nn.TransformerEncoderLayer(256, nhead, 512, dropout=0.1,
                                           batch_first=True, norm_first=True)
        self.enc = nn.TransformerEncoder(layer, nlayer)
        self.head = nn.Linear(256, 2)

    def forward(self, x):
        h = self.proj(x) + self.pos[:, : x.shape[1]]
        m = torch.triu(torch.ones(x.shape[1], x.shape[1], device=x.device, dtype=torch.bool), 1)
        return self.head(self.enc(h, mask=m))


ARMS = {
    "A1_BCAST_CAUSAL": BcastCausal,
    "A1b_BCAST_VIDEO": BcastVideo,
    "A2_PERWIN": PerWin,
    "A3_MIL_TOPK": MILTopK,
    "A4_CTCN": CTCN,
    "A5_CTCN_NOSMOOTH": CTCN,
    "A6_CTRANS": CTrans,
}


# ------------------------------------------------------------------ metrics
def ts_counts(y_ts, win_of_ts, pred_win, ids):
    """per-video (tp,fp,fn,tn) on 0.25 s timestamps."""
    out = np.zeros((len(ids), 4), dtype=np.int64)
    for j, i in enumerate(ids):
        p = pred_win[j][win_of_ts[i]]
        y = y_ts[i]
        out[j] = [int(((p == 1) & (y == 1)).sum()), int(((p == 1) & (y == 0)).sum()),
                  int(((p == 0) & (y == 1)).sum()), int(((p == 0) & (y == 0)).sum())]
    return out


def macro_f1_acc(c):
    tp, fp, fn, tn = c
    f1p = 2 * tp / max(2 * tp + fp + fn, 1)
    f1n = 2 * tn / max(2 * tn + fn + fp, 1)
    acc = (tp + tn) / max(tp + fp + fn + tn, 1)
    return 100 * (f1p + f1n) / 2, 100 * acc


def win_macro_f1(y, p):
    tp = int(((p == 1) & (y == 1)).sum()); fp = int(((p == 1) & (y == 0)).sum())
    fn = int(((p == 0) & (y == 1)).sum()); tn = int(((p == 0) & (y == 0)).sum())
    return macro_f1_acc((tp, fp, fn, tn))[0]


def average_precision(y, s):
    o = np.argsort(-s)
    y = y[o]
    tp = np.cumsum(y)
    prec = tp / np.arange(1, len(y) + 1)
    n_pos = y.sum()
    return 100 * float((prec * y).sum() / max(n_pos, 1))


def boot_ci(fn_a, fn_b, n_videos, rng, n=N_BOOT):
    d = np.empty(n)
    for b in range(n):
        idx = rng.integers(0, n_videos, n_videos)
        d[b] = fn_a(idx) - fn_b(idx)
    lo, hi = np.percentile(d, [2.5, 97.5])
    return float(lo), float(hi)


# ------------------------------------------------------------------ training
def train_eval(arm, X, y_target, y_video, tr, va, te, seed, lam, sel_metric,
               y_ts=None, win_of_ts=None):
    torch.manual_seed(seed)
    np.random.seed(seed)
    d_in = X.shape[-1]
    model = ARMS[arm](d_in).to(DEV)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, EPOCHS)
    Xt = torch.from_numpy(X).to(DEV)
    Yt = torch.from_numpy(y_target).to(DEV)
    Vt = torch.from_numpy(y_video).to(DEV)
    is_mil = arm == "A3_MIL_TOPK"
    is_tcn = arm.startswith("A4") or arm.startswith("A5")
    g = np.random.default_rng(seed)
    best = (-1.0, None)

    for ep in range(EPOCHS):
        model.train()
        perm = g.permutation(tr)
        for s in range(0, len(perm), BATCH):
            b = perm[s : s + BATCH]
            xb, yb = Xt[b], Yt[b]
            out = model(xb)
            if is_mil:
                loss = F.cross_entropy(model.video_logit(out), Vt[b])
            else:
                outs = model._all if is_tcn else [out]
                loss = 0.0
                for o in outs:
                    loss = loss + F.cross_entropy(o.reshape(-1, 2), yb.reshape(-1))
                    if lam > 0:
                        lp = F.log_softmax(o, dim=-1)
                        dl = torch.clamp((lp[:, 1:] - lp[:, :-1]) ** 2, max=TAU ** 2)
                        loss = loss + lam * dl.mean()
            opt.zero_grad(); loss.backward(); opt.step()
        sched.step()

        model.eval()
        with torch.no_grad():
            pv = F.softmax(model(Xt[va]), dim=-1)[..., 1].cpu().numpy()
        if sel_metric == "ts_macro_f1":
            c = ts_counts(y_ts, win_of_ts, (pv >= 0.5).astype(int), va).sum(0)
            sc = macro_f1_acc(c)[0]
        elif sel_metric == "win_macro_f1":
            sc = win_macro_f1(y_target[va].ravel(), (pv >= 0.5).astype(int).ravel())
        else:  # ap
            sc = average_precision(y_target[va].ravel(), pv.ravel())
        if sc > best[0]:
            with torch.no_grad():
                pt = F.softmax(model(Xt[te]), dim=-1)[..., 1].cpu().numpy()
            best = (sc, pt)
    return best[0], best[1]


def run_arm_set(D, X, tag, results):
    tr, va, te = D["tr"], D["va"], D["te"]
    for arm in ARMS:
        lam = LAMBDA_SMOOTH if arm.startswith("A4") else 0.0
        probs, vsc = [], []
        t0 = time.time()
        for sd in SEEDS:
            v, p = train_eval(arm, X, D["y_win"], D["y_video"], tr, va, te, sd, lam,
                              "ts_macro_f1", D["y_ts"], D["win_of_ts"])
            probs.append(p); vsc.append(v)
        P = np.stack(probs)  # [S,119,30]
        per_seed = []
        for s in range(P.shape[0]):
            c = ts_counts(D["y_ts"], D["win_of_ts"], (P[s] >= 0.5).astype(int), te).sum(0)
            per_seed.append(macro_f1_acc(c))
        pm = P.mean(0)
        cnt = ts_counts(D["y_ts"], D["win_of_ts"], (pm >= 0.5).astype(int), te)
        f1, acc = macro_f1_acc(cnt.sum(0))
        results[f"{tag}/{arm}"] = dict(
            ts_macro_f1=f1, ts_acc=acc,
            per_seed_macro_f1=[x[0] for x in per_seed],
            per_seed_acc=[x[1] for x in per_seed],
            seed_mean_macro_f1=float(np.mean([x[0] for x in per_seed])),
            seed_sd_macro_f1=float(np.std([x[0] for x in per_seed], ddof=1)),
            val_sel_scores=vsc,
            win_macro_f1=win_macro_f1(D["y_win"][te].ravel(), (pm >= 0.5).astype(int).ravel()),
            secs=time.time() - t0,
        )
        np.save(OUT / f"probs_{tag}_{arm}.npy", pm)
        results[f"{tag}/{arm}"]["_counts"] = cnt.tolist()
        print(f"  [{tag}] {arm:18s} ts-MF1 {f1:6.2f}  acc {acc:6.2f}  "
              f"seedmean {np.mean([x[0] for x in per_seed]):6.2f}±{np.std([x[0] for x in per_seed],ddof=1):.2f}  "
              f"({time.time()-t0:.0f}s)", flush=True)


def causality_guard(d_in=64):
    """Assert every online arm's output at window t is independent of windows > t."""
    torch.manual_seed(0)
    x = torch.randn(4, K, d_in, device=DEV)
    x2 = x.clone(); x2[:, 20:] += 100.0
    for name in ("A1_BCAST_CAUSAL", "A2_PERWIN", "A3_MIL_TOPK", "A4_CTCN",
                 "A5_CTCN_NOSMOOTH", "A6_CTRANS"):
        m = ARMS[name](d_in).to(DEV).eval()
        with torch.no_grad():
            past = (m(x)[:, :20] - m(x2)[:, :20]).abs().max().item()
        assert past < 1e-4, f"{name} leaks future into the past ({past})"
        print(f"[guard] causal OK  {name}  max|delta_past|={past:.2e}", flush=True)
    m = ARMS["A1b_BCAST_VIDEO"](d_in).to(DEV).eval()
    with torch.no_grad():
        leak = (m(x)[:, :20] - m(x2)[:, :20]).abs().max().item()
    assert leak > 1e-4
    print(f"[guard] A1b_BCAST_VIDEO is non-causal as declared (leak={leak:.2e})", flush=True)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    causality_guard()
    D = load_all()
    tr, va, te = D["tr"], D["va"], D["te"]
    results, meta = {}, {}

    # A0 CONST -------------------------------------------------------
    maj = int(D["y_win"][tr].mean() >= 0.5)
    predc = np.full((len(te), K), maj, dtype=int)
    c0 = ts_counts(D["y_ts"], D["win_of_ts"], predc, te)
    f0, a0 = macro_f1_acc(c0.sum(0))
    results["ALL/A0_CONST"] = dict(ts_macro_f1=f0, ts_acc=a0, _counts=c0.tolist(),
                                   majority_class=maj)
    print(f"  [ALL] A0_CONST           ts-MF1 {f0:6.2f}  acc {a0:6.2f}  (majority={maj})", flush=True)

    X_ALL = zscore(D["ALL"], tr)
    X_V = zscore(D["chans"]["V"], tr)
    print(f"[main] ALL dim={X_ALL.shape[-1]}  V dim={X_V.shape[-1]}", flush=True)
    run_arm_set(D, X_ALL, "ALL", results)
    run_arm_set(D, X_V, "V", results)

    # ---- bootstrap contrasts
    rng = np.random.default_rng(BOOT_SEED)
    n = len(te)

    def mk(tag_arm):
        c = np.array(results[tag_arm]["_counts"])
        return lambda idx: macro_f1_acc(c[idx].sum(0))[0]

    contrasts = {}
    for tag in ("ALL", "V"):
        pairs = [("A4_CTCN", "A2_PERWIN"), ("A4_CTCN", "A1_BCAST_CAUSAL"),
                 ("A4_CTCN", "A6_CTRANS"), ("A4_CTCN", "A5_CTCN_NOSMOOTH"),
                 ("A6_CTRANS", "A2_PERWIN"), ("A2_PERWIN", "A1_BCAST_CAUSAL"),
                 ("A3_MIL_TOPK", "A2_PERWIN"), ("A1b_BCAST_VIDEO", "A1_BCAST_CAUSAL")]
        for a, b in pairs:
            ka, kb = f"{tag}/{a}", f"{tag}/{b}"
            if ka not in results or kb not in results:
                continue
            pt = results[ka]["ts_macro_f1"] - results[kb]["ts_macro_f1"]
            lo, hi = boot_ci(mk(ka), mk(kb), n, np.random.default_rng(BOOT_SEED))
            contrasts[f"{tag}/{a}-{b}"] = dict(delta=pt, ci=[lo, hi])
            print(f"  [ci] {tag}/{a}-{b}: {pt:+.3f} [{lo:+.3f},{hi:+.3f}]", flush=True)
    results["_contrasts"] = contrasts

    # ---- B3 2x2 ----------------------------------------------------
    print("[B3] matched 2x2", flush=True)
    b3 = {}
    XV = zscore(D["chans"]["V"], tr)
    XA = zscore(D["chans"]["A"], tr)
    XE = zscore(D["chans"]["E"], tr)
    gsh = np.random.default_rng(SHIFT_SEED)
    off = gsh.integers(1, K, size=len(D["vids"]))

    def shift(X):
        Y = np.empty_like(X)
        for i in range(X.shape[0]):
            Y[i] = np.roll(X[i], int(off[i]), axis=0)
        return Y

    cells = [("VIS", XV, False), ("AUD", XA, False), ("EGE", XE, False),
             ("VIS_shift", shift(XV), True), ("AUD_shift", shift(XA), True)]
    for task, ytgt, metric in [("LABEL", D["y_win"], "win_macro_f1"),
                               ("CHANGE", D["y_change"], "ap")]:
        for name, Xc, is_sh in cells:
            if task == "CHANGE" and is_sh:
                continue
            probs = []
            t0 = time.time()
            for sd in SEEDS:
                _, p = train_eval("A4_CTCN", Xc, ytgt, D["y_video"], tr, va, te, sd,
                                  LAMBDA_SMOOTH, metric)
                probs.append(p)
            pm = np.stack(probs).mean(0)
            yv = ytgt[te]
            if task == "LABEL":
                sc = win_macro_f1(yv.ravel(), (pm >= 0.5).astype(int).ravel())
            else:
                sc = average_precision(yv.ravel(), pm.ravel())
            b3[f"{task}/{name}"] = dict(score=sc, secs=time.time() - t0)
            np.save(OUT / f"b3_{task}_{name}.npy", pm)
            print(f"  [B3] {task:6s} {name:10s} {metric}={sc:6.2f} ({time.time()-t0:.0f}s)", flush=True)

    # B3 CIs
    def mkb3(task, name):
        pm = np.load(OUT / f"b3_{task}_{name}.npy")
        y = (D["y_win"] if task == "LABEL" else D["y_change"])[te]
        if task == "LABEL":
            pr = (pm >= 0.5).astype(int)
            per = np.array([[int(((pr[j] == 1) & (y[j] == 1)).sum()),
                             int(((pr[j] == 1) & (y[j] == 0)).sum()),
                             int(((pr[j] == 0) & (y[j] == 1)).sum()),
                             int(((pr[j] == 0) & (y[j] == 0)).sum())] for j in range(len(te))])
            return lambda idx: macro_f1_acc(per[idx].sum(0))[0]
        return lambda idx: average_precision(y[idx].ravel(), pm[idx].ravel())

    b3c = {}
    for a, b, task in [("AUD", "VIS", "LABEL"), ("VIS", "AUD", "CHANGE"),
                       ("AUD", "AUD_shift", "LABEL"), ("VIS", "VIS_shift", "LABEL"),
                       ("EGE", "VIS", "LABEL"), ("EGE", "VIS", "CHANGE")]:
        d = b3[f"{task}/{a}"]["score"] - b3[f"{task}/{b}"]["score"]
        lo, hi = boot_ci(mkb3(task, a), mkb3(task, b), n, np.random.default_rng(BOOT_SEED))
        b3c[f"{task}/{a}-{b}"] = dict(delta=d, ci=[lo, hi])
        print(f"  [B3 ci] {task}/{a}-{b}: {d:+.3f} [{lo:+.3f},{hi:+.3f}]", flush=True)
    results["_b3"] = b3
    results["_b3_contrasts"] = b3c

    meta = dict(seeds=SEEDS, boot_seed=BOOT_SEED, shift_seed=SHIFT_SEED,
                n_boot=N_BOOT, epochs=EPOCHS, lr=LR, wd=WD, batch=BATCH,
                lambda_smooth=LAMBDA_SMOOTH, tau=TAU,
                n_train=len(tr), n_val=len(va), n_test=len(te),
                train_win_base_rate=float(D["y_win"][tr].mean()),
                test_win_base_rate=float(D["y_win"][te].mean()),
                dim_all=int(X_ALL.shape[-1]))
    results["_meta"] = meta
    (OUT / "results.json").write_text(json.dumps(results, indent=2))
    print(f"[done] wrote {OUT/'results.json'}", flush=True)


if __name__ == "__main__":
    main()
