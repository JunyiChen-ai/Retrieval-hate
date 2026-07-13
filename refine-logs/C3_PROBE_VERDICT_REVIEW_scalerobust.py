#!/usr/bin/env python3
"""Scale-robustness of the corrected (A-free) target/label oracle numbers.
Confirms the corrected target Deltaacc is not an artifact of scaleA=50."""
import json, numpy as np, warnings
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
import torch
warnings.filterwarnings('ignore')
RNG = np.random.default_rng(20260714); EPS = 1e-12
DS = 'HateMM'; N_SPLITS, N_REPEATS = 5, 5
ENC = [('CLIP', 'openai_clip-vit-large-patch14-336_HF'), ('Qwen', 'Qwen2.5-VL-7B-Instruct_HF')]

def load(enc):
    o = torch.load(f'data/CLIP_Embedding/{DS}/train_{enc}.pt', map_location='cpu', weights_only=False)
    Z = np.concatenate([o['img_feats'].numpy(), o['text_feats'].numpy()], 1).astype(np.float64)
    return o['ids'][0], Z, o['labels'].numpy().astype(int)

def target(ids):
    tm = json.load(open(f'data/gt/{DS}/target_map.json')); nt = tm['_meta']['num_targets']
    v = {k: x for k, x in tm.items() if not k.startswith('_')}
    prim = np.array([v[s]['primary'] for s in ids], int)
    A = np.zeros((len(ids), nt + 1)); A[np.arange(len(ids)), np.where(prim == -1, nt, prim)] = 1.0
    return A

def cv(Z, A, y, C, s):
    n = len(y); nb = np.zeros(n); cb = np.zeros(n); na = np.zeros(n); ca = np.zeros(n); cnt = np.zeros(n)
    for rep in range(N_REPEATS):
        for tr, te in StratifiedKFold(N_SPLITS, shuffle=True, random_state=1000 + rep).split(Z, y):
            sc = StandardScaler().fit(Z[tr]); Ztr, Zte = sc.transform(Z[tr]), sc.transform(Z[te])
            lb = LogisticRegression(C=C, max_iter=5000).fit(Ztr, y[tr])
            pb = np.clip(lb.predict_proba(Zte)[:, 1], EPS, 1 - EPS)
            nb[te] += -np.log2(np.where(y[te] == 1, pb, 1 - pb)); cb[te] += (pb >= .5) == y[te]
            la = LogisticRegression(C=C, max_iter=5000).fit(np.concatenate([Ztr, A[tr]*s], 1), y[tr])
            pa = np.clip(la.predict_proba(np.concatenate([Zte, A[te]*s], 1))[:, 1], EPS, 1 - EPS)
            na[te] += -np.log2(np.where(y[te] == 1, pa, 1 - pa)); ca[te] += (pa >= .5) == y[te]; cnt[te] += 1
    return cb/cnt, ca/cnt, nb/cnt, na/cnt

def bci(d, B=5000):
    m = np.array([d[RNG.integers(0, len(d), len(d))].mean() for _ in range(B)])
    return d.mean(), np.percentile(m, 2.5), np.percentile(m, 97.5)

for name, e in ENC:
    ids, Z, y = load(e); A = target(ids)
    Al = np.zeros((len(y), 2)); Al[np.arange(len(y)), y] = 1.0
    print(f'\n[{name}] Z-only acc baseline (C=0.001)')
    for s in [10, 50, 100, 200]:
        cb, ca, _, _ = cv(Z, Al, y, 0.001, s)
        cbt, cat, nbt, nat = cv(Z, A, y, 0.001, s)
        dl = ca.mean() - cb.mean()
        dt = bci(cat - cbt); dbt = bci(nbt - nat)
        print(f'  scale={s:<4} label accZA={ca.mean():.4f} (dacc={dl:+.4f})  '
              f'target accZA={cat.mean():.4f} dacc={dt[0]:+.4f} CI[{dt[1]:+.4f},{dt[2]:+.4f}] '
              f'dbits={dbt[0]:+.4f} CI[{dbt[1]:+.4f},{dbt[2]:+.4f}]')
