#!/usr/bin/env python3
"""Step-14 scale-up runs for LOCO-ST (see SCALE_PLAN.md).

Phases (each idempotent, keyed in scale_results.json):
  headline  - valsel + loo_zero, 5 seeds, dense test scores saved
  joint     - joint multitask comparator, 3 seeds
  sources   - single-source / union-minus-hatemm zero-shot, 3 seeds
  sens      - tau/lambda/topk one-at-a-time, hatemm, seed 234

Dense scores: runs/20260830_spantransfer_pilot/scores/<corpus>/<arm>_seed<у>.jsonl
"""
import argparse
import copy
import json
import os
import sys

import numpy as np
import torch
import torch.nn as nn

REPO = "/home/jehc223/Retrieval-hate"
sys.path.insert(0, os.path.join(REPO, "scripts", "reproduction_baselines"))
sys.path.insert(0, os.path.join(REPO, "experiments", "20260830_powa_within_diagnosis"))
sys.path.insert(0, os.path.join(REPO, "experiments", "20260830_spantransfer_pilot"))
from eval_baseline_scores import evaluate_scores  # noqa: E402
from hate_common import data as hdata  # noqa: E402
import skyline as S  # noqa: E402
import spantransfer as ST  # noqa: E402

OUT_DIR = ST.OUT_DIR
SCORE_DIR = os.path.join(OUT_DIR, "scores")
RESULTS = os.path.join(OUT_DIR, "scale_results.json")
SEEDS5 = (234, 2025, 3407, 42, 20260830)
SEEDS3 = (234, 2025, 3407)
CKPT_EPOCHS = (0, 1, 2, 4, 8, 15)


def score_test(model, corpus, arm, seed, save=True):
    gt = hdata.gt_arrays(corpus, "test")
    labels = hdata.load_labels(corpus)
    hate_ids = {v for v in gt if labels.get(v) == 1}
    scores = {}
    model.eval()
    with torch.no_grad():
        for v in gt:
            try:
                x = S.load_feats(corpus, v, ST.DIRS, len(gt[v]))
            except FileNotFoundError:
                continue
            scores[v] = model(torch.from_numpy(x).to(ST.DEV)).cpu().numpy().astype(float)
    if save:
        d = os.path.join(SCORE_DIR, corpus)
        os.makedirs(d, exist_ok=True)
        with open(os.path.join(d, "%s_seed%d.jsonl" % (arm, seed)), "w") as fh:
            for v, s in sorted(scores.items()):
                fh.write(json.dumps({"video_id": v, "score": list(s)}) + "\n")
    m = evaluate_scores(scores, {v: gt[v] for v in scores}, hate_ids)
    hi = [auc for v, auc in m["per_video"]["per_video_auc"].items()
          if np.asarray(gt[v]).mean() > 0.6]
    return {"frame_ap": m["pr_auc"], "frame_roc": m["roc_auc"],
            "within_roc_macro": m["per_video"]["macro_auc"],
            "within_n": m["per_video"]["n_videos_both_classes"],
            "hipos_within_roc": float(np.mean(hi)) if hi else None,
            "video_roc_max": m["video_level"]["max_roc_auc"],
            "per_video_auc": m["per_video"]["per_video_auc"]}


def eval_val_within(model, corpus):
    gt = hdata.gt_arrays(corpus, "val")
    labels = hdata.load_labels(corpus)
    hate_ids = {v for v in gt if labels.get(v) == 1}
    scores = {}
    model.eval()
    with torch.no_grad():
        for v in gt:
            try:
                x = S.load_feats(corpus, v, ST.DIRS, len(gt[v]))
            except FileNotFoundError:
                continue
            scores[v] = model(torch.from_numpy(x).to(ST.DEV)).cpu().numpy().astype(float)
    m = evaluate_scores(scores, {v: gt[v] for v in scores}, hate_ids)
    macro = m["per_video"]["macro_auc"]
    return -0.5 if macro is None else macro


def adapt_with_ckpts(model, weak, seed, tau, lam, kdiv):
    teacher = [None] * len(weak)
    model.eval()
    with torch.no_grad():
        for i, (x, _) in enumerate(weak):
            teacher[i] = model(x.to(ST.DEV)).detach()
    arng = np.random.default_rng(seed + 7)
    opt = torch.optim.Adam(model.parameters(), lr=ST.LR / 3)
    ckpts = {0: copy.deepcopy(model.state_dict())}
    for ep in range(1, max(CKPT_EPOCHS) + 1):
        model.train()
        for i in arng.permutation(len(weak)):
            x, y = weak[i]
            logits = model(x.to(ST.DEV))
            k = max(1, len(logits) // kdiv)
            bag = torch.topk(logits, k).values.mean()
            loss = nn.functional.binary_cross_entropy_with_logits(
                bag, torch.tensor(y, device=ST.DEV))
            if len(logits) > 3:
                t = teacher[i]
                a = torch.from_numpy(arng.integers(len(t), size=ST.N_PAIRS)).to(ST.DEV)
                b = torch.from_numpy(arng.integers(len(t), size=ST.N_PAIRS)).to(ST.DEV)
                keep = (t[a] - t[b]).abs() > tau
                if keep.any():
                    a, b = a[keep], b[keep]
                    sign = torch.sign(t[a] - t[b])
                    loss = loss + lam * nn.functional.margin_ranking_loss(
                        logits[a], logits[b], sign, margin=0.0)
            opt.zero_grad(); loss.backward(); opt.step()
        if ep in CKPT_EPOCHS:
            ckpts[ep] = copy.deepcopy(model.state_dict())
    return ckpts


def run_valsel(target, seed, tau=ST.TAU, lam=ST.LAMBDA_RANK, kdiv=8,
               arm="valsel", sources=None, save=True):
    rng = np.random.default_rng(seed)
    aux = []
    for c in (sources or [c for c in ST.CORPORA if c != target]):
        aux += ST.pack_spans(c, rng=rng, shuffle=False)
    model = ST.pretrain(aux, seed)
    weak = ST.pack_weak(target)
    ckpts = adapt_with_ckpts(model, weak, seed, tau, lam, kdiv)
    best_ep, best_val = 0, -1.0
    for ep in CKPT_EPOCHS:
        model.load_state_dict(ckpts[ep])
        val = eval_val_within(model, target)
        if val > best_val:
            best_val, best_ep = val, ep
    model.load_state_dict(ckpts[best_ep])
    out = score_test(model, target, arm, seed, save=save)
    out["selected_epoch"] = best_ep
    return out


def run_zero(target, seed, sources=None, arm="loo_zero", save=True):
    rng = np.random.default_rng(seed)
    aux = []
    for c in (sources or [c for c in ST.CORPORA if c != target]):
        aux += ST.pack_spans(c, rng=rng, shuffle=False)
    model = ST.pretrain(aux, seed)
    return score_test(model, target, arm, seed, save=save)


def run_joint(target, seed):
    """OSAD-style: aux frame BCE + target MIL jointly, from scratch."""
    rng = np.random.default_rng(seed)
    aux = []
    for c in ST.CORPORA:
        if c != target:
            aux += ST.pack_spans(c, rng=rng, shuffle=False)
    weak = ST.pack_weak(target)
    torch.manual_seed(seed)
    model = S.TemporalConv(aux[0][0].shape[1]).to(ST.DEV)
    opt = torch.optim.Adam(model.parameters(), lr=ST.LR)
    pos = float(np.mean([t[1].mean().item() for t in aux]))
    w = torch.tensor((1 - pos) / max(pos, 1e-6), device=ST.DEV)
    items = [("aux", i) for i in range(len(aux))] + \
            [("weak", i) for i in range(len(weak))]
    for ep in range(ST.PRE_EPOCHS):
        model.train()
        for j in rng.permutation(len(items)):
            kind, i = items[j]
            if kind == "aux":
                x, y = aux[i]
                loss = nn.functional.binary_cross_entropy_with_logits(
                    model(x.to(ST.DEV)), y.to(ST.DEV), pos_weight=w)
            else:
                x, y = weak[i]
                logits = model(x.to(ST.DEV))
                k = max(1, len(logits) // 8)
                bag = torch.topk(logits, k).values.mean()
                loss = nn.functional.binary_cross_entropy_with_logits(
                    bag, torch.tensor(y, device=ST.DEV))
            opt.zero_grad(); loss.backward(); opt.step()
    return score_test(model, target, "joint", seed)


def agg(per_seed):
    out = {}
    for k in per_seed[0]:
        if k == "per_video_auc":
            continue
        if k == "selected_epoch":
            out[k] = [p[k] for p in per_seed]
            continue
        if k in ("within_n",):
            out[k] = per_seed[0][k]
            continue
        vals = [p[k] for p in per_seed if p[k] is not None]
        out[k] = ({"mean": float(np.mean(vals)), "sd": float(np.std(vals))}
                  if vals else None)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phases", default="headline,joint,sources,sens")
    args = ap.parse_args()
    os.makedirs(SCORE_DIR, exist_ok=True)
    blob = json.load(open(RESULTS)) if os.path.exists(RESULTS) else {"results": {}}
    res = blob["results"]

    def save():
        with open(RESULTS, "w") as fh:
            json.dump(blob, fh, indent=1)

    phases = args.phases.split(",")
    if "headline" in phases:
        for target in ST.CORPORA:
            for arm, fn in (("valsel", run_valsel), ("loo_zero", run_zero)):
                key = "%s/%s/5seed" % (target, arm)
                if key in res:
                    continue
                per_seed = [fn(target, s) for s in SEEDS5]
                res[key] = agg(per_seed)
                save()
                print(key, json.dumps(res[key]), flush=True)
    if "headline" in phases:
        # loo_naive dense scores (3 seeds) for the promised bootstrap comparison
        for target in ST.CORPORA:
            key = "%s/loo_naive/3seed" % target
            if key in res:
                continue
            per_seed = []
            for s in SEEDS3:
                rng = np.random.default_rng(s)
                aux = []
                for c in ST.CORPORA:
                    if c != target:
                        aux += ST.pack_spans(c, rng=rng, shuffle=False)
                model = ST.pretrain(aux, s)
                model = ST.adapt(model, ST.pack_weak(target), s, rank_term=False)
                per_seed.append(score_test(model, target, "loo_naive", s))
            res[key] = agg(per_seed)
            save()
            print(key, json.dumps(res[key]), flush=True)
    if "joint" in phases:
        for target in ST.CORPORA:
            key = "%s/joint/3seed" % target
            if key in res:
                continue
            per_seed = [run_joint(target, s) for s in SEEDS3]
            res[key] = agg(per_seed)
            save()
            print(key, json.dumps(res[key]), flush=True)
    if "sources" in phases:
        for target in ST.CORPORA:
            others = [c for c in ST.CORPORA if c != target]
            source_sets = [[c] for c in others]
            if "hatemm" in others and len(others) > 1:
                source_sets.append([c for c in others if c != "hatemm"])
            for srcs in source_sets:
                name = "+".join(srcs)
                key = "%s/zero_src_%s/3seed" % (target, name)
                if key in res:
                    continue
                per_seed = [run_zero(target, s, sources=srcs,
                                     arm="zero_src_" + name, save=False)
                            for s in SEEDS3]
                res[key] = agg(per_seed)
                save()
                print(key, json.dumps(res[key]), flush=True)
    if "sens" in phases:
        grid = ([("tau", t, dict(tau=t)) for t in (0.25, 1.0)] +
                [("lam", l, dict(lam=l)) for l in (0.3, 3.0)] +
                [("kdiv", k, dict(kdiv=k)) for k in (4, 16)])
        for pname, pval, kw in grid:
            key = "hatemm/sens_%s_%s/seed234" % (pname, pval)
            if key in res:
                continue
            out = run_valsel("hatemm", 234, arm="sens_%s_%s" % (pname, pval),
                             save=False, **kw)
            res[key] = {k: v for k, v in out.items() if k != "per_video_auc"}
            save()
            print(key, json.dumps(res[key]), flush=True)
    print("SCALE_DONE", flush=True)


if __name__ == "__main__":
    main()
