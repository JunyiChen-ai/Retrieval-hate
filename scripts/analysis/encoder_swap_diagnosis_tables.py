#!/usr/bin/env python
"""
Encoder-swap diagnosis — supporting tables (imports encoder_swap_geometry).

Reproduces every number in refine-logs/ENCODER_SWAP_DIAGNOSIS.md:
  T1  per-modality train->dev kNN vote (acc/mF1/auc) + delta(Qwen-CLIP)
  T2  per-modality train-LOO AUC (larger sample) + delta, incl. 32B scale column
  T3  per-class recall at the kNN vote (minority=hate), concat, dev
  T4  error-set overlap CLIP vs Qwen on dev (net fixes)
  T5  easy-vs-hard stratification (dev, by CLIP-vote margin)
  T6  CLIP text-embedding degeneracy = empty-transcript proxy (train+dev)

All train/dev only. Zero GPU, zero model inference, zero test-touch.
"""
import sys, json, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from collections import Counter
from encoder_swap_geometry import (load, build_modality, knn_vote, loo_knn,
    knn_homogeneity, macro_f1, bal_acc, auc, DATASETS, K)

ENC = {"CLIP": "openai_clip-vit-large-patch14-336_HF",
       "Qwen7B": "Qwen2.5-VL-7B-Instruct_HF",
       "Qwen32B": "Qwen2.5-VL-32B-Instruct_HF"}
OUT = {}


def sect(t):
    print("\n" + "=" * 78 + "\n" + t + "\n" + "=" * 78)


# ---- T1 per-modality dev kNN ----
sect("T1  per-modality train->dev kNN (acc/mF1/auc) + d(Qwen7B-CLIP)")
OUT["T1"] = {}
dev_cache = {}
for ds, dd in DATASETS.items():
    OUT["T1"][ds] = {}
    for en in ("CLIP", "Qwen7B"):
        ti, tt, ty, _ = load(dd, ENC[en], "train")
        di, dt, dy, _ = load(dd, ENC[en], "dev_seen")
        dev_cache[(ds, en)] = (ti, tt, ty, di, dt, dy)
        for mode in ("img", "text", "concat"):
            X = build_modality(ti, tt, mode); Xd = build_modality(di, dt, mode)
            p, s, _ = knn_vote(X, ty, Xd, K)
            OUT["T1"][ds].setdefault(mode, {})[en] = dict(
                acc=float(np.mean(p == dy)), mf1=macro_f1(dy, p), auc=auc(dy, s))
    for mode in ("img", "text", "concat"):
        c = OUT["T1"][ds][mode]["CLIP"]; q = OUT["T1"][ds][mode]["Qwen7B"]
        print(f"{ds:7s} {mode:6s} CLIP {c['acc']:.3f}/{c['mf1']:.3f}/{c['auc']:.3f}  "
              f"Qwen {q['acc']:.3f}/{q['mf1']:.3f}/{q['auc']:.3f}  "
              f"d {q['acc']-c['acc']:+.3f}/{q['mf1']-c['mf1']:+.3f}/{q['auc']-c['auc']:+.3f}")

# ---- T2 train-LOO AUC + 32B ----
sect("T2  per-modality train-LOO AUC (k=20) incl 32B scale column")
OUT["T2"] = {}
for ds, dd in DATASETS.items():
    OUT["T2"][ds] = {}
    for en, et in ENC.items():
        try:
            ti, tt, ty, _ = load(dd, et, "train")
        except Exception:
            continue
        row = {}
        for mode in ("img", "text", "concat"):
            X = build_modality(ti, tt, mode); p, s = loo_knn(X, ty, K)
            row[mode] = auc(ty, s)
        row["homog20_concat"] = knn_homogeneity(build_modality(ti, tt, "concat"), ty, K)
        OUT["T2"][ds][en] = row
        print(f"{ds:7s} {en:7s} img {row['img']:.3f}  text {row['text']:.3f}  "
              f"concat {row['concat']:.3f}  homog20 {row['homog20_concat']:.3f}")

# ---- T3 per-class recall ----
sect("T3  per-class recall (minority=hate=1) at concat dev kNN")
OUT["T3"] = {}
for ds, dd in DATASETS.items():
    OUT["T3"][ds] = {}
    for en in ("CLIP", "Qwen7B"):
        ti, tt, ty, di, dt, dy = dev_cache[(ds, en)]
        X = build_modality(ti, tt, "concat"); Xd = build_modality(di, dt, "concat")
        p, s, _ = knn_vote(X, ty, Xd, K)
        r0 = float(np.mean(p[dy == 0] == 0)); r1 = float(np.mean(p[dy == 1] == 1))
        OUT["T3"][ds][en] = dict(maj=float(max(dy.mean(), 1 - dy.mean())),
                                 rec_nonhate=r0, rec_hate=r1, mf1=macro_f1(dy, p))
        print(f"{ds:7s} {en:7s} maj={OUT['T3'][ds][en]['maj']:.3f} "
              f"rec_nonhate={r0:.3f} rec_hate={r1:.3f} mF1={macro_f1(dy,p):.3f}")

# ---- T4 error overlap ----
sect("T4  error-set overlap CLIP vs Qwen (concat dev)")
OUT["T4"] = {}
for ds, dd in DATASETS.items():
    tiC, ttC, tyC, diC, dtC, dyC = dev_cache[(ds, "CLIP")]
    tiQ, ttQ, tyQ, diQ, dtQ, dyQ = dev_cache[(ds, "Qwen7B")]
    pc, _, _ = knn_vote(build_modality(tiC, ttC, "concat"), tyC,
                        build_modality(diC, dtC, "concat"), K)
    pq, _, _ = knn_vote(build_modality(tiQ, ttQ, "concat"), tyQ,
                        build_modality(diQ, dtQ, "concat"), K)
    ec = pc != dyC; eq = pq != dyC
    both = int(np.sum(ec & eq)); conly = int(np.sum(ec & ~eq)); qonly = int(np.sum(~ec & eq))
    OUT["T4"][ds] = dict(clip_err=int(ec.sum()), qwen_err=int(eq.sum()), both=both,
                         qwen_fixes=conly, qwen_breaks=qonly, net_fix=conly - qonly)
    print(f"{ds:7s} CLIP_err={ec.sum():2d} Qwen_err={eq.sum():2d} both={both:2d} "
          f"fixes={conly:2d} breaks={qonly:2d} net={conly-qonly:+d}")

# ---- T5 easy vs hard ----
sect("T5  easy-vs-hard (dev, stratified by CLIP-vote |margin| thirds)")
OUT["T5"] = {}
for ds, dd in DATASETS.items():
    tiC, ttC, tyC, diC, dtC, dyC = dev_cache[(ds, "CLIP")]
    tiQ, ttQ, tyQ, diQ, dtQ, dyQ = dev_cache[(ds, "Qwen7B")]
    pc, sc, _ = knn_vote(build_modality(tiC, ttC, "concat"), tyC,
                         build_modality(diC, dtC, "concat"), K)
    pq, sq, _ = knn_vote(build_modality(tiQ, ttQ, "concat"), tyQ,
                         build_modality(diQ, dtQ, "concat"), K)
    order = np.argsort(np.abs(sc)); third = len(dyC) // 3
    OUT["T5"][ds] = {}
    for name, idx in [("hard", order[:third]), ("easy", order[-third:])]:
        cc = float(np.mean(pc[idx] == dyC[idx])); qq = float(np.mean(pq[idx] == dyC[idx]))
        OUT["T5"][ds][name] = dict(n=len(idx), clip=cc, qwen=qq, d=qq - cc)
        print(f"{ds:7s} {name:5s} n={len(idx):2d} CLIP={cc:.3f} Qwen={qq:.3f} d={qq-cc:+.3f}")

# ---- T6 empty-transcript proxy ----
sect("T6  CLIP text-embedding degeneracy (empty-transcript proxy, train+dev)")
OUT["T6"] = {}
for ds, dd in DATASETS.items():
    _, ttc, _, _ = load(dd, ENC["CLIP"], "train")
    _, dtc, _, _ = load(dd, ENC["CLIP"], "dev_seen")
    T = np.concatenate([ttc, dtc]); Tn = T / (np.linalg.norm(T, axis=1, keepdims=True) + 1e-8)
    keys = [tuple(np.round(v, 4)) for v in Tn]
    cnt = Counter(keys); top, topn = cnt.most_common(1)[0]
    modal = Tn[[i for i, k in enumerate(keys) if k == top][0]]
    near = float(np.mean(Tn @ modal > 0.999))
    OUT["T6"][ds] = dict(n=len(T), modal_count=int(topn), modal_frac=topn / len(T),
                         near_modal_frac=near, unique4dp=len(cnt))
    print(f"{ds:7s} N={len(T)} modal={topn} ({topn/len(T)*100:.1f}%) "
          f"near_modal={near*100:.1f}% unique={len(cnt)}")

outp = os.path.join(os.path.dirname(os.path.abspath(__file__)), "encoder_swap_diagnosis_tables_out.json")
with open(outp, "w") as f:
    json.dump(OUT, f, indent=1)
print("\nwrote", outp)
