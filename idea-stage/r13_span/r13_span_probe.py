#!/usr/bin/env python
"""R13-SPAN kill probe -- implements idea-stage/R13_SPAN_FREEZE.md exactly.

Train-split only. No dev/test file is opened (hard-asserted below).
Outputs: idea-stage/r13_span/r13_span.json
"""
import argparse
import ast
import csv
import hashlib
import json
import os
import sys
import time

import numpy as np
import torch

ROOT = "/home/jehc223/Retrieval-hate"
sys.path.insert(0, ROOT)
from src.utils.generate_segment_asr_HF import window_time_bounds  # noqa: E402

OUT_DIR = os.path.join(ROOT, "idea-stage", "r13_span")
TEXT_CACHE = os.path.join(OUT_DIR, "text_feats_cache.pt")
CLIP_MODEL = "openai/clip-vit-large-patch14-336"

SEEDS = list(range(2000, 2020))
FOLD_SEED = 2020
BOOT_SEED = 2021
N_BOOT = 10000
ARMS = ["P0", "P1", "P2", "P3", "P4"]
CHANNELS = ["visual", "text", "concat"]

# --- pre-result amendment D1 (2026-08-18): matched-length coverage sweep ---
SWEEP_RS = [0.10, 0.20, 0.40]
N_ORACLE_CAND = 16


def rcode(r):
    return "%03d" % int(round(r * 100))


def gname(r):
    return "G" + rcode(r)


def rname(r):
    return "R" + rcode(r)


def oname(r):
    return "O" + rcode(r)


SWEEP_ARMS = []
for _r in SWEEP_RS:
    SWEEP_ARMS += [gname(_r), rname(_r), oname(_r)]
ALL_ARMS = ARMS + SWEEP_ARMS

_OPENED = []


def safe_open_path(path):
    """Every data path must go through here: blocks dev/test caches."""
    base = os.path.basename(path)
    for bad in ("_test_seen_", "_dev_seen_", "test_seen", "dev_seen"):
        if bad in base:
            raise RuntimeError("REFUSED to open held-out cache: %s" % path)
    _OPENED.append(path)
    return path


def log(msg):
    print("[%s] %s" % (time.strftime("%H:%M:%S"), msg), flush=True)


# --------------------------------------------------------------------------
# interval helpers
# --------------------------------------------------------------------------
def merge(ivs, D):
    out = []
    for s, e in sorted([[max(0.0, float(s)), min(float(D), float(e))] for s, e in ivs]):
        if e <= s:
            continue
        if out and s <= out[-1][1]:
            out[-1][1] = max(out[-1][1], e)
        else:
            out.append([s, e])
    return out


def total_len(ivs):
    return sum(e - s for s, e in ivs)


def complement(ivs, D):
    out = []
    prev = 0.0
    for s, e in ivs:
        if s > prev:
            out.append([prev, s])
        prev = max(prev, e)
    if prev < D:
        out.append([prev, D])
    return out


def random_interval(cov, D, u):
    """One contiguous interval of length cov*D at position u in [0,1]."""
    cov = float(min(max(cov, 0.0), 1.0))
    L = cov * D
    if L >= D:
        return [[0.0, D]]
    if L <= 0.0:
        return []
    s = u * (D - L)
    return [[s, s + L]]


def overlaps(a, b):
    return max(0.0, min(a[1], b[1]) - max(a[0], b[0])) > 0.0


# --------------------------------------------------------------------------
# text restriction
# --------------------------------------------------------------------------
def restrict_text(chunks, ivs, word_level):
    """Keep a chunk iff its midpoint falls inside the kept interval set."""
    keep = []
    for ch in chunks:
        s, e, txt = ch[0], ch[1], ch[2]
        raw = txt or ""
        if not raw.strip():
            continue
        mid = 0.5 * (float(s) + float(e))
        for a, b in ivs:
            if a <= mid <= b:
                keep.append(raw if word_level else raw.strip())
                break
    joiner = "" if word_level else " "
    return joiner.join(keep).strip()


# --------------------------------------------------------------------------
# CLIP text encoder (exact logic of src/utils/generate_VideoCLIP_embedding_HF.encode_text,
# with equal-length chunk batching -- identical arithmetic, no padding)
# --------------------------------------------------------------------------
class TextEncoder(object):
    def __init__(self, device):
        from transformers import CLIPTokenizer, CLIPTextModel

        self.tok = CLIPTokenizer.from_pretrained(CLIP_MODEL)
        self.model = CLIPTextModel.from_pretrained(CLIP_MODEL).to(device).eval()
        self.device = device
        max_len = getattr(self.tok, "model_max_length", 77)
        if not max_len or max_len > 77:
            max_len = 77
        self.content_window = max_len - 2
        self.bos = self.tok.bos_token_id
        self.eos = self.tok.eos_token_id
        self.dim = self.model.config.hidden_size

    def windows_for(self, text):
        content_ids = self.tok(text if text is not None else "",
                               add_special_tokens=False)["input_ids"]
        cw = self.content_window
        if len(content_ids) <= cw:
            windows = [content_ids] if content_ids else [[]]
        else:
            windows = [content_ids[i:i + cw] for i in range(0, len(content_ids), cw)]
        out = []
        for w in windows:
            ids = ([self.bos] if self.bos is not None else []) + list(w) + \
                  ([self.eos] if self.eos is not None else [])
            out.append(ids)
        return out

    @torch.no_grad()
    def encode_many(self, texts, batch_tokens=8192):
        """texts: list[str] -> [N, D] float32 cpu. Mean of per-chunk pooler_output."""
        all_ids = []          # (text_index, ids)
        for i, t in enumerate(texts):
            for ids in self.windows_for(t):
                all_ids.append((i, ids))
        # group by identical length -> stack without padding (exact same math)
        bylen = {}
        for idx, (ti, ids) in enumerate(all_ids):
            bylen.setdefault(len(ids), []).append(idx)
        pooled = torch.zeros(len(all_ids), self.dim, dtype=torch.float32)
        done = 0
        for L, idxs in sorted(bylen.items()):
            bs = max(1, batch_tokens // max(L, 1))
            for i in range(0, len(idxs), bs):
                sl = idxs[i:i + bs]
                batch = torch.tensor([all_ids[j][1] for j in sl],
                                     dtype=torch.long, device=self.device)
                am = torch.ones_like(batch)
                out = self.model(input_ids=batch, attention_mask=am)
                pooled[sl] = out.pooler_output.detach().cpu().float()
                done += len(sl)
                if done % 20000 < len(sl):
                    log("    text chunks encoded: %d/%d" % (done, len(all_ids)))
        feats = torch.zeros(len(texts), self.dim, dtype=torch.float32)
        cnt = torch.zeros(len(texts))
        for k, (ti, _) in enumerate(all_ids):
            feats[ti] += pooled[k]
            cnt[ti] += 1
        feats = feats / cnt.clamp(min=1).unsqueeze(1)
        return feats


def sha(s):
    return hashlib.sha1(s.encode("utf-8")).hexdigest()


# --------------------------------------------------------------------------
# metrics
# --------------------------------------------------------------------------
def l2norm(X):
    n = np.linalg.norm(X, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return X / n


def auc_2d(y, S):
    """S: [nd, N] scores, y: [N] 0/1 -> [nd] Mann-Whitney AUC (tie-corrected)."""
    S = np.atleast_2d(S)
    nd, n = S.shape
    order = np.argsort(S, axis=1, kind="stable")
    Ss = np.take_along_axis(S, order, axis=1)
    Ys = y[order].astype(np.float64)
    diff = np.empty(Ss.shape, dtype=bool)
    diff[:, 0] = True
    diff[:, 1:] = Ss[:, 1:] != Ss[:, :-1]
    grp = np.cumsum(diff, axis=1) - 1
    off = (grp + (np.arange(nd)[:, None] * n)).ravel()
    r = np.tile(np.arange(1, n + 1, dtype=np.float64), nd)
    csum = np.bincount(off, weights=r, minlength=nd * n)
    cnt = np.bincount(off, minlength=nd * n)
    avg = csum / np.maximum(cnt, 1)
    ranks = avg[off].reshape(nd, n)
    npos = Ys.sum(axis=1)
    nneg = n - npos
    bad = (npos == 0) | (nneg == 0)
    sumpos = (ranks * Ys).sum(axis=1)
    out = (sumpos - npos * (npos + 1) / 2.0) / np.maximum(npos * nneg, 1)
    out[bad] = np.nan
    return out


def fast_auc(y, s):
    return float(auc_2d(y, np.asarray(s)[None, :])[0])


def macro_f1(y, pred):
    f = []
    for c in (0, 1):
        tp = np.sum((pred == c) & (y == c))
        fp = np.sum((pred == c) & (y != c))
        fn = np.sum((pred != c) & (y == c))
        p = tp / (tp + fp) if (tp + fp) else 0.0
        r = tp / (tp + fn) if (tp + fn) else 0.0
        f.append(2 * p * r / (p + r) if (p + r) else 0.0)
    return float(np.mean(f))


def logloss_2d(y, S):
    S = np.atleast_2d(np.clip(S, 1e-12, 1 - 1e-12))
    return -(y[None, :] * np.log(S) + (1 - y)[None, :] * np.log(1 - S)).mean(axis=1)


def oof_lr(X, y):
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold

    X = l2norm(X.astype(np.float64))
    oof = np.zeros(len(y))
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=FOLD_SEED)
    for tr, te in skf.split(X, y):
        clf = LogisticRegression(C=1.0, max_iter=2000)
        clf.fit(X[tr], y[tr])
        oof[te] = clf.predict_proba(X[te])[:, 1]
    return oof


def oof_lr_aug(X_eval, X_aug, y):
    """Amendment D2 arm set C: train on P0 keys + (optionally) a cropped copy of the
    SAME training videos; always evaluate on P0 keys of the held-out fold."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold

    Xe = l2norm(X_eval.astype(np.float64))
    Xa = None if X_aug is None else l2norm(X_aug.astype(np.float64))
    oof = np.zeros(len(y))
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=FOLD_SEED)
    for tr, te in skf.split(Xe, y):
        assert len(set(tr.tolist()) & set(te.tolist())) == 0
        if Xa is None:
            Xtr, ytr = Xe[tr], y[tr]
        else:
            Xtr = np.vstack([Xe[tr], Xa[tr]])
            ytr = np.concatenate([y[tr], y[tr]])
            # the augmented copy of a held-out video must never be in training
            assert Xtr.shape[0] == 2 * len(tr)
        clf = LogisticRegression(C=1.0, max_iter=2000)
        clf.fit(Xtr, ytr)
        oof[te] = clf.predict_proba(Xe[te])[:, 1]
    return oof


def oof_lr_2feat(z_a, z_b, y):
    """Amendment D2 arm set D: cross-fitted LR on the two OOF score columns."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold

    X = np.column_stack([z_a, z_b])
    oof = np.zeros(len(y))
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=FOLD_SEED)
    for tr, te in skf.split(X, y):
        clf = LogisticRegression(C=1.0, max_iter=2000)
        clf.fit(X[tr], y[tr])
        oof[te] = clf.predict_proba(X[te])[:, 1]
    return oof


def knn_loo_f1(X, y, k=20):
    Xn = l2norm(X.astype(np.float64))
    S = Xn @ Xn.T
    np.fill_diagonal(S, -np.inf)
    kk = min(k, len(y) - 1)
    idx = np.argpartition(-S, kk - 1, axis=1)[:, :kk]
    votes = y[idx].sum(axis=1)
    pred = (votes > kk / 2.0).astype(int)
    return macro_f1(y, pred)


# --------------------------------------------------------------------------
# arena builder
# --------------------------------------------------------------------------
class Arena(object):
    def __init__(self, name, vids, labels, durations, gold, chunks, word_level,
                 img_feats, parent_idx, K, M, seg_scores, seg_K, seg_M):
        self.name = name
        self.vids = vids
        self.y = np.array(labels, dtype=int)
        self.D = durations                 # vid -> duration
        self.gold = gold                   # vid -> merged span list (positives only)
        self.chunks = chunks               # vid -> chunk list
        self.word_level = word_level
        self.img = img_feats               # vid -> [K, dim] np array
        self.K, self.M = K, M
        self.wb = {v: window_time_bounds(self.D[v], M, K) for v in vids}
        self.seg = seg_scores              # vid -> list[seg_K] or None
        self.seg_K, self.seg_M = seg_K, seg_M
        self.seg_wb = ({v: window_time_bounds(self.D[v], seg_M, seg_K) for v in vids}
                       if seg_scores is not None else None)
        self.pos_cov = np.array([total_len(self.gold[v]) / self.D[v]
                                 for i, v in enumerate(vids)
                                 if self.y[i] == 1 and v in self.gold])
        assert len(self.pos_cov) > 0, "no positive with a gold span"
        self.counts = {}
        self.cur_arm = ""
        # amendment D1: midpoint + length of each positive's LONGEST gold span
        self.longest_gold = {}
        for v, g in self.gold.items():
            s, e = max(g, key=lambda iv: iv[1] - iv[0])
            self.longest_gold[v] = (0.5 * (s + e), e - s)
        self.oracle_iv = {}
        self._draw_cache = {}

        self.counting = False

    def bump(self, key, n=1):
        if not self.counting:
            return
        k = "%s:%s" % (self.cur_arm, key)
        self.counts[k] = self.counts.get(k, 0) + n

    def sweep_stats(self):
        """Amendment D1 bookkeeping: overlap floor + r-window vs gold-span length."""
        cov = self.pos_cov
        out = {
            "n_positives_with_gold": int(len(cov)),
            "gold_cov_mean": float(np.mean(cov)),
            "gold_cov_median": float(np.median(cov)),
            "gold_cov_p05": float(np.percentile(cov, 5)),
            "gold_cov_p95": float(np.percentile(cov, 95)),
            "frac_cov_ge_0.9": float(np.mean(cov >= 0.9)),
            "frac_cov_eq_1.0": float(np.mean(cov >= 0.999999)),
            "forced_overlap_floor_2c_minus_1_median": float(
                max(0.0, 2 * np.median(cov) - 1)),
        }
        for r in SWEEP_RS:
            L_gt_gold = 0
            for i, v in enumerate(self.vids):
                if self.y[i] != 1 or v not in self.longest_gold:
                    continue
                if r * self.D[v] > self.longest_gold[v][1]:
                    L_gt_gold += 1
            out["r%s_window_longer_than_longest_gold_span" % rcode(r)] = L_gt_gold
        return out

    def sweep_iou(self, arms_present):
        """temporal IoU between G_r and R_r intervals, positives only, all draws."""
        out = {}
        for r in SWEEP_RS:
            if gname(r) not in arms_present:
                continue
            vals = []
            for seed in SEEDS:
                dr_all = self.draws(seed)
                for i, v in enumerate(self.vids):
                    if self.y[i] != 1 or v not in self.gold:
                        continue
                    a = merge(self.interval(gname(r), v, i, dr_all[v]), self.D[v])
                    b = merge(self.interval(rname(r), v, i, dr_all[v]), self.D[v])
                    if not a or not b:
                        continue
                    inter = max(0.0, min(a[-1][1], b[-1][1]) - max(a[0][0], b[0][0]))
                    union = total_len(a) + total_len(b) - inter
                    vals.append(inter / union if union > 0 else 0.0)
            v = np.array(vals)
            out["r" + rcode(r)] = {
                "mean": float(v.mean()), "median": float(np.median(v)),
                "p05": float(np.percentile(v, 5)), "p95": float(np.percentile(v, 95)),
                "frac_zero_overlap": float(np.mean(v <= 1e-12)), "n": int(len(v))}
        return out

    # -- per-seed draws (shared by all arms so P1/P2 negatives are byte-identical)
    def draws(self, seed):
        if seed in self._draw_cache:
            return self._draw_cache[seed]
        d = self._draws(seed)
        self._draw_cache[seed] = d
        return d

    def _draws(self, seed):
        rng = np.random.default_rng(seed)
        d = {}
        for v in self.vids:
            d[v] = {
                "neg_cov": float(rng.choice(self.pos_cov)),
                "neg_pos": float(rng.random()),
                "pos_pos": float(rng.random()),
                "perm_pos": rng.permutation(self.seg_K if self.seg_K else 1),
                "perm_neg": rng.permutation(self.seg_K if self.seg_K else 1),
            }
        # amendment D1: independent derived stream so the P0-P4 draws above are
        # byte-identical to what they were before the sweep was added.
        rs = np.random.default_rng([seed, 13])
        for v in self.vids:
            d[v]["u_pos"] = {rcode(r): float(rs.random()) for r in SWEEP_RS}
            d[v]["u_neg"] = {rcode(r): float(rs.random()) for r in SWEEP_RS}
        return d

    def oracle_candidates(self, v, r, n=N_ORACLE_CAND):
        D = self.D[v]
        L = min(r * D, D)
        hi = max(D - L, 0.0)
        starts = np.linspace(0.0, hi, n) if hi > 1e-9 else np.array([0.0])
        return [[[float(s), float(min(s + L, D))]] for s in starts]

    def topm_interval(self, v, m, perm):
        sc = np.asarray(self.seg[v], dtype=float)
        order = perm[np.argsort(-sc[perm], kind="stable")][:m]
        ivs = [self.seg_wb[v][int(k)] for k in sorted(order)]
        return merge(ivs, self.D[v])

    def interval(self, arm, v, i, dr):
        """Kept interval set for arm/video. Returns (ivs, is_empty)."""
        D = self.D[v]
        pos = self.y[i] == 1 and v in self.gold
        if arm == "P0":
            return [[0.0, D]]
        if arm == "P1":
            if pos:
                return list(self.gold[v])
            return random_interval(dr["neg_cov"], D, dr["neg_pos"])
        if arm == "P2":
            if pos:
                cov = total_len(self.gold[v]) / D
                return random_interval(cov, D, dr["pos_pos"])
            return random_interval(dr["neg_cov"], D, dr["neg_pos"])
        if arm == "P3":
            if pos:
                return complement(self.gold[v], D)
            return complement(merge(random_interval(dr["neg_cov"], D, dr["neg_pos"]), D), D)
        if arm == "P4":
            if self.seg is None or v not in self.seg:
                return None
            if pos:
                cov = total_len(self.gold[v]) / D
                perm = dr["perm_pos"]
            else:
                cov = dr["neg_cov"]
                perm = dr["perm_neg"]
            m = int(round(cov * self.seg_K))
            m = int(min(max(m, 1), self.seg_K))
            return self.topm_interval(v, m, perm)
        # ---- amendment D1 sweep arms ----
        if arm[0] in "GRO" and arm[1:].isdigit():
            code = arm[1:]
            r = int(code) / 100.0
            L = min(r * D, D)
            if arm[0] == "R":
                u = dr["u_pos"][code] if pos else dr["u_neg"][code]
                return random_interval(L / D if D > 0 else 0.0, D, u)
            if not pos:
                return random_interval(L / D if D > 0 else 0.0, D, dr["u_neg"][code])
            if arm[0] == "O":
                return list(self.oracle_iv[(arm, v)])
            mid, glen = self.longest_gold[v]
            s = mid - 0.5 * L
            s = min(max(s, 0.0), max(D - L, 0.0))
            return [[s, min(s + L, D)]]
        raise ValueError(arm)

    def visual_key(self, v, ivs):
        if not ivs:
            self.bump("empty_interval_zero_visual")
            return np.zeros(self.img[v].shape[1], dtype=np.float32)
        sel = [k for k, w in enumerate(self.wb[v]) if any(overlaps(w, iv) for iv in ivs)]
        if not sel:
            self.bump("widened_to_nearest_window")
            c = 0.5 * (ivs[0][0] + ivs[-1][1])
            sel = [int(np.argmin([abs(0.5 * (w[0] + w[1]) - c) for w in self.wb[v]]))]
        return self.img[v][sel].mean(axis=0)


def build_arena_keys(ar, arms, text_lookup, textdim, count=True):
    """Returns keys[arm][channel] -> [n_draws, N, d] and per-arm bookkeeping."""
    out = {}
    covrec = {}
    ar.counting = count
    for arm in arms:
        ar.cur_arm = arm
        nd = 1 if arm == "P0" else len(SEEDS)
        Vs, Ts, covs = [], [], []
        for di in range(nd):
            dr_all = ar.draws(SEEDS[di]) if arm != "P0" else ar.draws(SEEDS[0])
            V = np.zeros((len(ar.vids), ar.img[ar.vids[0]].shape[1]), dtype=np.float32)
            T = np.zeros((len(ar.vids), textdim), dtype=np.float32)
            cv = []
            for i, v in enumerate(ar.vids):
                ivs = ar.interval(arm, v, i, dr_all[v])
                if ivs is None:
                    ivs = [[0.0, ar.D[v]]]
                ivs = merge(ivs, ar.D[v])
                cv.append(total_len(ivs) / ar.D[v])
                V[i] = ar.visual_key(v, ivs)
                if text_lookup is not None:
                    txt = restrict_text(ar.chunks.get(v, []), ivs, ar.word_level)
                    if not txt.strip():
                        ar.bump("empty_text_zero_vector")
                    else:
                        T[i] = text_lookup(txt)
            Vs.append(V)
            Ts.append(T)
            covs.append(cv)
        Vs = np.stack(Vs)
        Ts = np.stack(Ts)
        ch = {"visual": Vs}
        if text_lookup is not None:
            ch["text"] = Ts
            ch["concat"] = np.concatenate([l2norm3(Vs), l2norm3(Ts)], axis=2)
        out[arm] = ch
        covrec[arm] = {"mean": float(np.mean(covs)), "std": float(np.std(np.mean(covs, axis=1)))}
    return out, covrec


def l2norm3(X):
    n = np.linalg.norm(X, axis=2, keepdims=True)
    n[n == 0] = 1.0
    return X / n


def collect_texts(ar, arms):
    """All unique restricted-transcript strings across arms x draws (+ oracle grid)."""
    ar.counting = False
    uniq = set()
    for arm in arms:
        if arm[0] == "O":
            continue  # covered by the candidate grid below
        nd = 1 if arm == "P0" else len(SEEDS)
        for di in range(nd):
            dr_all = ar.draws(SEEDS[di]) if arm != "P0" else ar.draws(SEEDS[0])
            for i, v in enumerate(ar.vids):
                ivs = ar.interval(arm, v, i, dr_all[v])
                if ivs is None:
                    ivs = [[0.0, ar.D[v]]]
                ivs = merge(ivs, ar.D[v])
                t = restrict_text(ar.chunks.get(v, []), ivs, ar.word_level)
                if t.strip():
                    uniq.add(t)
    for r in SWEEP_RS:
        if oname(r) not in arms:
            continue
        for i, v in enumerate(ar.vids):
            if ar.y[i] != 1 or v not in ar.gold:
                continue
            for cand in ar.oracle_candidates(v, r):
                t = restrict_text(ar.chunks.get(v, []), merge(cand, ar.D[v]),
                                  ar.word_level)
                if t.strip():
                    uniq.add(t)
    return sorted(uniq)


def select_oracle(ar, arms, text_lookup, textdim, P0keys, channel):
    """LEAKY positive control: pick, per positive, the length-matched interval that
    maximises its own OUT-OF-FOLD score under a model trained on P0 (full-video) keys.
    Illegal for any claim; exists only to prove the read-out has power."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold

    X0 = l2norm(P0keys["P0"][channel][0].astype(np.float64))
    y = ar.y
    fold_of = np.zeros(len(y), dtype=int)
    models = []
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=FOLD_SEED)
    for f, (tr, te) in enumerate(skf.split(X0, y)):
        clf = LogisticRegression(C=1.0, max_iter=2000)
        clf.fit(X0[tr], y[tr])
        models.append(clf)
        fold_of[te] = f
    ar.counting = False
    for r in SWEEP_RS:
        if oname(r) not in arms:
            continue
        arm = oname(r)
        for i, v in enumerate(ar.vids):
            if y[i] != 1 or v not in ar.gold:
                continue
            cands = ar.oracle_candidates(v, r)
            feats = []
            for cand in cands:
                ivs = merge(cand, ar.D[v])
                vis = ar.visual_key(v, ivs)
                if text_lookup is not None:
                    t = restrict_text(ar.chunks.get(v, []), ivs, ar.word_level)
                    txt = text_lookup(t) if t.strip() else np.zeros(textdim, np.float32)
                    a = vis / (np.linalg.norm(vis) or 1.0)
                    b = txt / (np.linalg.norm(txt) or 1.0)
                    key = np.concatenate([a, b]) if channel == "concat" else (
                        vis if channel == "visual" else txt)
                else:
                    key = vis
                feats.append(key)
            F = l2norm(np.stack(feats).astype(np.float64))
            p = models[fold_of[i]].predict_proba(F)[:, 1]
            ar.oracle_iv[(arm, v)] = merge(cands[int(np.argmax(p))], ar.D[v])
    return


# --------------------------------------------------------------------------
def evaluate(keys, y, arms, channels):
    res = {}
    oof_store = {}
    for arm in arms:
        res[arm] = {}
        oof_store[arm] = {}
        for chn in channels:
            if chn not in keys[arm]:
                continue
            X = keys[arm][chn]
            aucs, f1s, knns, oofs = [], [], [], []
            for di in range(X.shape[0]):
                Xi = X[di]
                oof = oof_lr(Xi, y)
                oofs.append(oof)
                aucs.append(fast_auc(y, oof))
                f1s.append(macro_f1(y, (oof >= 0.5).astype(int)))
                knns.append(knn_loo_f1(Xi, y))
            res[arm][chn] = {
                "auc_mean": float(np.mean(aucs)), "auc_std": float(np.std(aucs)),
                "f1_mean": float(np.mean(f1s)), "f1_std": float(np.std(f1s)),
                "knn_f1_mean": float(np.mean(knns)), "knn_f1_std": float(np.std(knns)),
                "auc_per_draw": [float(a) for a in aucs],
            }
            oof_store[arm][chn] = np.stack(oofs)
    return res, oof_store


def paired_bootstrap(oof_a, oof_b, y, rng, metric=None):
    """oof_*: [nd, N]. Delta of mean-over-draws metric, bootstrapped over videos."""
    metric = metric or auc_2d
    N = len(y)
    obs = float(np.nanmean(metric(y, oof_a)) - np.nanmean(metric(y, oof_b)))
    deltas = np.empty(N_BOOT)
    for b in range(N_BOOT):
        idx = rng.integers(0, N, N)
        yb = y[idx]
        if yb.sum() == 0 or yb.sum() == N:
            deltas[b] = np.nan
            continue
        a = np.nanmean(metric(yb, oof_a[:, idx]))
        c = np.nanmean(metric(yb, oof_b[:, idx]))
        deltas[b] = a - c
    d = deltas[~np.isnan(deltas)]
    return {"point": obs, "ci_lo": float(np.percentile(d, 2.5)),
            "ci_hi": float(np.percentile(d, 97.5)),
            "onesided95_upper": float(np.percentile(d, 95.0)),
            "onesided95_lower": float(np.percentile(d, 5.0)),
            "n_boot": int(len(d)),
            "frac_gt0": float(np.mean(d > 0)), "frac_lt0": float(np.mean(d < 0))}


AUG_ITEMS = [("r020", "G020", "R020"), ("r040", "G040", "R040"),
             ("fullspan", "P1", "P2")]


def run_setC(keys, y, chans):
    """Amendment D2 arm set C: crop-as-training-augmentation, P0-only inference."""
    res, oof = {}, {}
    for chn in chans:
        z = oof_lr_aug(keys["P0"][chn][0], None, y)
        oof.setdefault("AUG_none", {})[chn] = z[None, :]
        res.setdefault("AUG_none", {})[chn] = _summ(y, z[None, :])
    for tag, garm, rarm in AUG_ITEMS:
        for side, arm in (("gold", garm), ("rand", rarm)):
            nm = "AUG_%s_%s" % (side, tag)
            if arm not in keys:
                continue
            for chn in chans:
                A = keys[arm][chn]
                Z = np.stack([oof_lr_aug(keys["P0"][chn][0], A[d], y)
                              for d in range(A.shape[0])])
                oof.setdefault(nm, {})[chn] = Z
                res.setdefault(nm, {})[chn] = _summ(y, Z)
    return res, oof


def _summ(y, Z):
    a = auc_2d(y, Z)
    ll = logloss_2d(y, Z)
    f1 = [macro_f1(y, (Z[d] >= 0.5).astype(int)) for d in range(Z.shape[0])]
    return {"auc_mean": float(np.nanmean(a)), "auc_std": float(np.nanstd(a)),
            "logloss_mean": float(np.mean(ll)), "logloss_std": float(np.std(ll)),
            "f1_mean": float(np.mean(f1)), "f1_std": float(np.std(f1))}


D_ITEMS = [("r020", "G020", "R020"), ("fullspan", "P1", "P2")]


def run_setD(oof_main, y, chans):
    """Amendment D2 arm set D: give a classifier BOTH the full-video OOF score and the
    cropped-view OOF score. Strictly more access than any distilled student."""
    res, oof = {}, {}
    wrong = {}
    for chn in chans:
        z0 = oof_main["P0"][chn][0]
        wrong[chn] = ((z0 >= 0.5).astype(int) != y)
    for tag, garm, rarm in D_ITEMS:
        for side, arm in (("gold", garm), ("rand", rarm)):
            nm = "M_%s_%s" % (side, tag)
            if arm not in oof_main:
                continue
            for chn in chans:
                z0 = oof_main["P0"][chn][0]
                Zc = oof_main[arm][chn]
                Z = np.stack([oof_lr_2feat(z0, Zc[d], y) for d in range(Zc.shape[0])])
                oof.setdefault(nm, {})[chn] = Z
                r = _summ(y, Z)
                w = wrong[chn]
                if w.sum() >= 10 and 0 < y[w].sum() < w.sum():
                    r["auc_on_P0_wrong"] = float(np.nanmean(auc_2d(y[w], Z[:, w])))
                    r["logloss_on_P0_wrong"] = float(np.mean(logloss_2d(y[w], Z[:, w])))
                r["n_P0_wrong"] = int(w.sum())
                res.setdefault(nm, {})[chn] = r
    # reference: P0 score alone
    for chn in chans:
        z0 = oof_main["P0"][chn][0]
        res.setdefault("M_P0_only", {})[chn] = _summ(y, z0[None, :])
        oof.setdefault("M_P0_only", {})[chn] = z0[None, :]
    return res, oof, wrong


def cos_dist(keys, arms=("P0", "P1")):
    out = {}
    for chn in keys[arms[0]]:
        A = keys[arms[0]][chn]      # [1, N, d]
        B = keys[arms[1]][chn]      # [nd, N, d]
        a = l2norm3(A)[0]
        Bn = l2norm3(B)
        v = np.concatenate([(a * Bn[d]).sum(axis=1) for d in range(Bn.shape[0])])
        out[chn] = {"mean": float(np.mean(v)), "median": float(np.median(v)),
                    "p05": float(np.percentile(v, 5)), "p95": float(np.percentile(v, 95))}
    return out


# --------------------------------------------------------------------------
# loaders
# --------------------------------------------------------------------------
def load_jsonl(path):
    with open(safe_open_path(path)) as f:
        return [json.loads(l) for l in f if l.strip()]


def split_ids(path):
    """ids only -- never reads the label field of a held-out split."""
    ids = []
    with open(safe_open_path(path)) as f:
        for l in f:
            if l.strip():
                ids.append(json.loads(l)["id"])
    return set(ids)


def subclip_dict(pt_path):
    d = torch.load(safe_open_path(pt_path), map_location="cpu")
    vids = list(d["video_ids"])
    feats = d["subclip_img_feats"].numpy().astype(np.float32)
    par = d["subclip_parent"].numpy()
    K = int(d["num_subclips"])
    M = int(d["num_frames"])
    per = {}
    for pi, v in enumerate(vids):
        rows = np.where(par == pi)[0]
        per[v] = feats[rows]
        assert per[v].shape[0] == K, (v, per[v].shape)
    return per, K, M


def run_arena(ar, arms, enc, device, do_text=True):
    log("  arena %s: N=%d pos=%d neg=%d" % (ar.name, len(ar.vids),
                                            int(ar.y.sum()), int((ar.y == 0).sum())))
    text_lookup = None
    if do_text:
        log("  collecting unique restricted transcripts ...")
        texts = collect_texts(ar, arms)
        log("  unique strings: %d" % len(texts))
        cache = {}
        if os.path.exists(TEXT_CACHE):
            cache = torch.load(TEXT_CACHE, map_location="cpu")
        need = [t for t in texts if sha(t) not in cache]
        log("  need encoding: %d (cached %d)" % (len(need), len(texts) - len(need)))
        if need:
            B = 4000
            for i in range(0, len(need), B):
                sub = need[i:i + B]
                F = enc.encode_many(sub)
                for j, t in enumerate(sub):
                    cache[sha(t)] = F[j].clone()
                log("  encoded %d/%d strings" % (min(i + B, len(need)), len(need)))
                if (i // B) % 10 == 9 or i + B >= len(need):
                    torch.save(cache, TEXT_CACHE)
        textdim = enc.dim
        text_lookup = lambda t: cache[sha(t)].numpy()  # noqa: E731
    else:
        textdim = 1
    chans = CHANNELS if do_text else ["visual"]
    primary = "concat" if do_text else "visual"

    log("  selecting LEAKY oracle intervals (positive control) ...")
    k0, _ = build_arena_keys(ar, ["P0"], text_lookup, textdim, count=False)
    select_oracle(ar, arms, text_lookup, textdim, k0, primary)

    log("  building keys ...")
    keys, cov = build_arena_keys(ar, arms, text_lookup, textdim, count=True)
    log("  evaluating ...")
    res, oof = evaluate(keys, ar.y, arms, chans)
    boots = {}
    pairs = [("D1_P1_minus_P2", "P1", "P2"), ("D2_P1_minus_P0", "P1", "P0"),
             ("P3_minus_P2", "P3", "P2"), ("P4_minus_P2", "P4", "P2")]
    for r in SWEEP_RS:
        pairs.append(("Dsweep_r%s_G_minus_R" % rcode(r), gname(r), rname(r)))
        pairs.append(("ORACLE_r%s_minus_R" % rcode(r), oname(r), rname(r)))
    for nm, a, b in pairs:
        if a not in arms or b not in arms:
            continue
        boots[nm] = {}
        for chn in chans:
            if chn not in oof.get(a, {}) or chn not in oof.get(b, {}):
                continue
            rr = np.random.default_rng(BOOT_SEED)
            boots[nm][chn] = paired_bootstrap(oof[a][chn], oof[b][chn], ar.y, rr)
        log("    bootstrap %s done" % nm)

    # ---- amendment D2, arm set C: augmentation under unchanged inference ----
    log("  arm set C (augmentation) ...")
    resC, oofC = run_setC(keys, ar.y, chans)
    bootsC = {}
    for tag, _g, _r in AUG_ITEMS:
        for other, lab in (("AUG_rand_%s" % tag, "minus_rand"), ("AUG_none", "minus_none")):
            a = "AUG_gold_%s" % tag
            if a not in oofC or other not in oofC:
                continue
            nm = "Daug_%s_%s" % (tag, lab)
            bootsC[nm] = {}
            for chn in chans:
                rr = np.random.default_rng(BOOT_SEED)
                bootsC[nm][chn] = paired_bootstrap(oofC[a][chn], oofC[other][chn], ar.y, rr)
            log("    bootstrap %s done" % nm)

    # ---- amendment D2, arm set D: privileged-information falsification ----
    log("  arm set D (privileged information) ...")
    resD, oofD, wrongmask = run_setD(oof, ar.y, chans)
    bootsD = {}
    for tag, _g, _r in D_ITEMS:
        a, b = "M_gold_%s" % tag, "M_rand_%s" % tag
        if a not in oofD or b not in oofD:
            continue
        for mname, mfun in (("auc", auc_2d), ("logloss", logloss_2d)):
            nm = "Dpriv_%s_gold_minus_rand_%s" % (tag, mname)
            bootsD[nm] = {}
            for chn in chans:
                rr = np.random.default_rng(BOOT_SEED)
                bootsD[nm][chn] = paired_bootstrap(oofD[a][chn], oofD[b][chn], ar.y,
                                                   rr, metric=mfun)
            log("    bootstrap %s done" % nm)

    cosd = cos_dist(keys) if "P1" in arms else {}
    return {"n": len(ar.vids), "n_pos": int(ar.y.sum()), "metrics": res,
            "coverage": cov, "bootstrap": boots, "cos_P0_P1": cosd,
            "edge_counts": dict(ar.counts),
            "sweep_stats": ar.sweep_stats(),
            "sweep_iou_G_vs_R": ar.sweep_iou(arms),
            "setC_metrics": resC, "setC_bootstrap": bootsC,
            "setD_metrics": resD, "setD_bootstrap": bootsD,
            "n_P0_wrong": {c: int(w.sum()) for c, w in wrongmask.items()}}


# --------------------------------------------------------------------------
def subsample(ar, n):
    pos = [v for i, v in enumerate(ar.vids) if ar.y[i] == 1][:n // 2]
    neg = [v for i, v in enumerate(ar.vids) if ar.y[i] == 0][:n - n // 2]
    keep = set(pos + neg)
    vids = [v for v in ar.vids if v in keep]
    lab = {v: int(ar.y[i]) for i, v in enumerate(ar.vids)}
    return Arena(ar.name + "-smoke", vids, [lab[v] for v in vids],
                 {v: ar.D[v] for v in vids},
                 {v: ar.gold[v] for v in vids if v in ar.gold},
                 {v: ar.chunks[v] for v in vids}, ar.word_level,
                 {v: ar.img[v] for v in vids}, None, ar.K, ar.M,
                 ar.seg, ar.seg_K, ar.seg_M)


def build_hatemm():
    tr = load_jsonl(os.path.join(ROOT, "data/gt/HateMM/train.jsonl"))
    train_ids = [r["id"] for r in tr]
    lab = {r["id"]: int(r["label"]) for r in tr}
    val_ids = split_ids(os.path.join(ROOT, "data/gt/HateMM/val.jsonl"))
    test_ids = split_ids(os.path.join(ROOT, "data/gt/HateMM/test.jsonl"))
    assert not (set(train_ids) & val_ids), "train/val id leak"
    assert not (set(train_ids) & test_ids), "train/test id leak"

    spans_all = json.load(open(safe_open_path(os.path.join(ROOT, "data/gt/HateMM/hate_spans.json"))))
    spans = {k: v for k, v in spans_all.items() if k in set(train_ids)}
    assert not (set(spans) & val_ids) and not (set(spans) & test_ids), "span file leaked held-out ids"
    log("hate_spans.json: %d total -> %d after train filter (val/test intersection = 0)"
        % (len(spans_all), len(spans)))

    asr = {r["id"]: r for r in load_jsonl(
        os.path.join(ROOT, "data/ASR/HateMM/train_asrK30_whisper-large-v3.jsonl"))}
    img, K, M = subclip_dict(os.path.join(
        ROOT, "data/CLIP_Embedding/HateMM/train_subclipK30_openai_clip-vit-large-patch14-336_HF.pt"))
    seg = {r["id"]: r["scores"] for r in load_jsonl(
        os.path.join(ROOT, "data/MLLM_scores/HateMM/train_segscoreK4_qwen.jsonl"))}

    vids = [v for v in train_ids if v in asr and v in img]
    D = {v: float(asr[v]["duration"]) for v in vids}
    gold = {}
    for v in vids:
        if lab[v] == 1:
            gold[v] = merge(spans[v]["spans"], D[v])
    chunks = {v: asr[v].get("chunks") or [] for v in vids}
    ar = Arena("HateMM", vids, [lab[v] for v in vids], D, gold, chunks, True,
               img, None, K, M, seg, 4, 16)
    return ar


def build_mhc(ds, tsv_name, variant):
    tr = load_jsonl(os.path.join(ROOT, "data/gt/%s/train.jsonl" % ds))
    train_ids = [r["id"] for r in tr]
    lab = {r["id"]: int(r["label"]) for r in tr}
    val_ids = split_ids(os.path.join(ROOT, "data/gt/%s/val.jsonl" % ds))
    test_ids = split_ids(os.path.join(ROOT, "data/gt/%s/test.jsonl" % ds))
    assert not (set(train_ids) & val_ids) and not (set(train_ids) & test_ids)

    dur_col = {}
    with open(safe_open_path(os.path.join(ROOT, "data/gt/mhc_votes/%s" % tsv_name))) as f:
        for r in csv.DictReader(f, delimiter="\t"):
            s = (r.get("Duration") or "").strip()
            if s and s != "[]":
                try:
                    tup = ast.literal_eval(s)
                except Exception:
                    continue
                if tup:
                    dur_col[r["Video_ID"]] = [[float(a), float(b)] for a, b in tup]

    asr_path = os.path.join(ROOT, "data/ASR/%s/train_asrK4_whisper-large-v3.jsonl" % ds)
    has_asr = os.path.exists(asr_path)
    asr = {r["id"]: r for r in load_jsonl(asr_path)} if has_asr else {}
    img, K, M = subclip_dict(os.path.join(
        ROOT, "data/CLIP_Embedding/%s/train_subclipK4_openai_clip-vit-large-patch14-336_HF.pt" % ds))
    segp = os.path.join(ROOT, "data/MLLM_scores/%s/train_segscoreK4_qwen.jsonl" % ds)
    seg = ({r["id"]: r["scores"] for r in load_jsonl(segp)}
           if os.path.exists(segp) else None)

    cand = [v for v in train_ids if v in asr and v in img]
    if variant == "restricted":
        vids = [v for v in cand if v in dur_col]
    else:  # neg-augmented
        vids = [v for v in cand if (lab[v] == 1 and v in dur_col) or lab[v] == 0]
    D = {v: float(asr[v]["duration"]) for v in vids}
    gold = {}
    dropped = []
    for v in list(vids):
        if lab[v] == 1:
            g = merge(dur_col.get(v, []), D[v])
            if g:
                gold[v] = g
            else:
                dropped.append(v)
    if dropped:
        vids = [v for v in vids if v not in set(dropped)]
        D = {v: D[v] for v in vids}
    chunks = {v: asr[v].get("chunks") or [] for v in vids}
    word_level = (asr[vids[0]].get("timestamps") == "word") if vids else True
    ar = Arena("%s/%s" % (ds, variant), vids, [lab[v] for v in vids], D, gold,
               chunks, word_level, img, None, K, M, seg, 4 if seg else None, 16)
    ar.counts["_dropped_pos_without_usable_gold_span"] = len(dropped)
    return ar, has_asr, seg is not None


# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip_mhc", action="store_true")
    ap.add_argument("--smoke", type=int, default=0,
                    help="smoke test: keep only N videos, 2 seeds, 200 bootstraps")
    args = ap.parse_args()
    global SEEDS, N_BOOT
    if args.smoke:
        SEEDS = SEEDS[:2]
        N_BOOT = 200
    t0 = time.time()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    log("device=%s" % device)
    enc = TextEncoder(device)

    out = {"freeze": "idea-stage/R13_SPAN_FREEZE.md", "seeds": SEEDS,
           "fold_seed": FOLD_SEED, "boot_seed": BOOT_SEED, "n_boot": N_BOOT,
           "amendment_D1": "matched-length coverage sweep G_r/R_r for r in %s, leaky "
                           "ORACLE_r positive control, one-sided 95%% upper bounds, "
                           "IoU + overlap-floor bookkeeping." % SWEEP_RS,
           "amendment_D2": "arm set C (crop-as-training-augmentation, P0-only "
                           "inference) and arm set D (privileged-information "
                           "falsification on [z_P0, z_crop]).",
           "arenas": {}}

    log("=== HateMM ===")
    ar = build_hatemm()
    if args.smoke:
        ar = subsample(ar, args.smoke)
    out["arenas"]["HateMM"] = run_arena(ar, ALL_ARMS, enc, device, do_text=True)
    out["arenas"]["HateMM"]["note"] = ("P4 uses train_segscoreK4_qwen.jsonl (K=4, M=16); "
                                       "the K=30 score files cover only the 298 hateful train "
                                       "videos so cannot supply negatives.")
    log("HateMM done at t=%.1fs" % (time.time() - t0))
    json.dump(out, open(os.path.join(OUT_DIR, "r13_span.json"), "w"), indent=1)

    if not args.skip_mhc:
        for ds, tsv in [("MHC", "mhc_English_train.tsv"), ("MHC_zh", "mhc_Chinese_train.tsv")]:
            for variant in ["restricted", "neg_augmented"]:
                log("=== %s / %s ===" % (ds, variant))
                a2, has_asr, has_seg = build_mhc(ds, tsv, variant)
                arms = [x for x in ALL_ARMS if x != "P4" or has_seg]
                r = run_arena(a2, arms, enc, device, do_text=has_asr)
                r["has_train_asr"] = has_asr
                r["has_segscores"] = has_seg
                r["limitation"] = "K=4 (M=16) windows: very coarse trimming."
                out["arenas"]["%s/%s" % (ds, variant)] = r
                json.dump(out, open(os.path.join(OUT_DIR, "r13_span.json"), "w"), indent=1)
                log("%s/%s done at t=%.1fs" % (ds, variant, time.time() - t0))

    out["runtime_sec"] = time.time() - t0
    out["files_opened"] = sorted(set(_OPENED))
    json.dump(out, open(os.path.join(OUT_DIR, "r13_span.json"), "w"), indent=1)

    # ---------------- single print, after every arm is computed ----------------
    print("\n" + "=" * 78)
    print("R13-SPAN RESULTS  (runtime %.1f s)" % out["runtime_sec"])
    print("=" * 78)
    for aname, A in out["arenas"].items():
        print("\n### %s  N=%d pos=%d" % (aname, A["n"], A["n_pos"]))
        print("%-4s %-8s %-16s %-16s %-16s %s" %
              ("arm", "channel", "OOF-AUC", "OOF-macroF1", "kNN20-macroF1", "coverage"))
        for arm in ALL_ARMS:
            if arm not in A["metrics"]:
                continue
            for chn in CHANNELS:
                m = A["metrics"][arm].get(chn)
                if not m:
                    continue
                print("%-4s %-8s %.4f +-%.4f  %.4f +-%.4f  %.4f +-%.4f  %.3f" %
                      (arm, chn, m["auc_mean"], m["auc_std"], m["f1_mean"], m["f1_std"],
                       m["knn_f1_mean"], m["knn_f1_std"], A["coverage"][arm]["mean"]))
        print("-- paired bootstrap (10000, seed 2021), OOF ROC-AUC --")
        for nm, B in A["bootstrap"].items():
            for chn, b in B.items():
                print("  %-26s %-8s %+.4f  95%%CI [%+.4f,%+.4f]  1-sided95 upper %+.4f"
                      "  P(>0)=%.3f" % (nm, chn, b["point"], b["ci_lo"], b["ci_hi"],
                                        b["onesided95_upper"], b["frac_gt0"]))
        print("-- cos(P0,P1) --")
        for chn, c in A["cos_P0_P1"].items():
            print("  %-8s mean %.4f med %.4f p05 %.4f p95 %.4f" %
                  (chn, c["mean"], c["median"], c["p05"], c["p95"]))
        print("-- SET C: crop-as-augmentation, inference on P0 keys only --")
        for nm in sorted(A.get("setC_metrics", {})):
            for chn in CHANNELS:
                m = A["setC_metrics"][nm].get(chn)
                if not m:
                    continue
                print("  %-20s %-8s AUC %.4f +-%.4f  logloss %.4f  F1 %.4f" %
                      (nm, chn, m["auc_mean"], m["auc_std"], m["logloss_mean"],
                       m["f1_mean"]))
        for nm, B in A.get("setC_bootstrap", {}).items():
            for chn, b in B.items():
                print("  %-26s %-8s %+.4f  95%%CI [%+.4f,%+.4f]  1-sided95 upper %+.4f"
                      "  P(>0)=%.3f" % (nm, chn, b["point"], b["ci_lo"], b["ci_hi"],
                                        b["onesided95_upper"], b["frac_gt0"]))
        print("-- SET D: privileged information [z_P0, z_crop] --")
        for nm in sorted(A.get("setD_metrics", {})):
            for chn in CHANNELS:
                m = A["setD_metrics"][nm].get(chn)
                if not m:
                    continue
                print("  %-20s %-8s AUC %.4f +-%.4f  logloss %.4f +-%.4f  "
                      "AUC|P0wrong %s  LL|P0wrong %s (n=%s)" %
                      (nm, chn, m["auc_mean"], m["auc_std"], m["logloss_mean"],
                       m["logloss_std"],
                       ("%.4f" % m["auc_on_P0_wrong"]) if "auc_on_P0_wrong" in m else "-",
                       ("%.4f" % m["logloss_on_P0_wrong"]) if "logloss_on_P0_wrong" in m else "-",
                       m.get("n_P0_wrong", "-")))
        for nm, B in A.get("setD_bootstrap", {}).items():
            for chn, b in B.items():
                print("  %-38s %-8s %+.4f  95%%CI [%+.4f,%+.4f]  1-sided95 upper %+.4f"
                      "  P(>0)=%.3f" % (nm, chn, b["point"], b["ci_lo"], b["ci_hi"],
                                        b["onesided95_upper"], b["frac_gt0"]))
        print("-- sweep stats --", json.dumps(A.get("sweep_stats", {})))
        print("-- IoU(G_r, R_r) positives --", json.dumps(A.get("sweep_iou_G_vs_R", {})))
        print("-- edge counts --", json.dumps(A["edge_counts"]))
    print("\nfiles opened:")
    for p in out["files_opened"]:
        print("  " + p)
    print("=" * 78)


if __name__ == "__main__":
    main()
