#!/usr/bin/env python3
"""Round 4: transfer-seeded self-training (see PILOT_PLAN.md).

Arms: st (seed = A1 valsel model), st_milseed (seed = weak-MIL model).
Targets hatemm/hateclipseg, 5 seeds, dense TEST scores saved, keys appended to
scale_results.json as <target>/<arm>/5seed.
"""
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
import skyline as S  # noqa: E402
import spantransfer as ST  # noqa: E402
import scale_up as SU  # noqa: E402

TARGETS = ("hatemm", "hateclipseg")
SEEDS5 = SU.SEEDS5
Q = 0.2
ST_EPOCHS = 30


def build_seed_model(target, seed, kind):
    """kind='transfer': A1 valsel recipe; kind='mil': weak-MIL control."""
    weak = ST.pack_weak(target)
    if kind == "transfer":
        rng = np.random.default_rng(seed)
        aux = []
        for c in ST.CORPORA:
            if c != target:
                aux += ST.pack_spans(c, rng=rng, shuffle=False)
        model = ST.pretrain(aux, seed)
        ckpts = SU.adapt_with_ckpts(model, weak, seed, ST.TAU, ST.LAMBDA_RANK, 8)
        best_ep, best_val = 0, -1.0
        for ep in SU.CKPT_EPOCHS:
            model.load_state_dict(ckpts[ep])
            v = SU.eval_val_within(model, target)
            if v > best_val:
                best_val, best_ep = v, ep
        model.load_state_dict(ckpts[best_ep])
        return model, weak
    # weak-MIL seed
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model = S.TemporalConv(weak[0][0].shape[1]).to(ST.DEV)
    opt = torch.optim.Adam(model.parameters(), lr=ST.LR)
    for ep in range(30):
        model.train()
        for i in rng.permutation(len(weak)):
            x, y = weak[i]
            logits = model(x.to(ST.DEV))
            k = max(1, len(logits) // 8)
            bag = torch.topk(logits, k).values.mean()
            loss = nn.functional.binary_cross_entropy_with_logits(
                bag, torch.tensor(y, device=ST.DEV))
            opt.zero_grad(); loss.backward(); opt.step()
    return model, weak


def pseudo_labels(model, weak):
    """Per hateful video: top Q pseudo-pos, bottom Q pseudo-neg; benign video:
    every second pseudo-neg. Returns list of (x, mask, target_frame)."""
    out = []
    model.eval()
    with torch.no_grad():
        for x, y in weak:
            T = x.shape[0]
            tgt = np.full(T, -1.0, dtype=np.float32)  # -1 = unlabeled
            if y == 0:
                tgt[:] = 0.0
            else:
                s = model(x.to(ST.DEV)).cpu().numpy()
                k = max(1, int(T * Q))
                order = np.argsort(s)
                tgt[order[-k:]] = 1.0
                tgt[order[:k]] = 0.0
            out.append((x, torch.from_numpy(tgt), float(y)))
    return out


def self_train(pseudo, seed):
    torch.manual_seed(seed + 13)
    rng = np.random.default_rng(seed + 13)
    model = S.TemporalConv(pseudo[0][0].shape[1]).to(ST.DEV)
    opt = torch.optim.Adam(model.parameters(), lr=ST.LR)
    for ep in range(ST_EPOCHS):
        model.train()
        for i in rng.permutation(len(pseudo)):
            x, tgt, y = pseudo[i]
            logits = model(x.to(ST.DEV))
            tgt_d = tgt.to(ST.DEV)
            sel = tgt_d >= 0
            loss = 0.0
            if sel.any():
                loss = nn.functional.binary_cross_entropy_with_logits(
                    logits[sel], tgt_d[sel])
            k = max(1, len(logits) // 8)
            bag = torch.topk(logits, k).values.mean()
            loss = loss + nn.functional.binary_cross_entropy_with_logits(
                bag, torch.tensor(y, device=ST.DEV))
            opt.zero_grad(); loss.backward(); opt.step()
    return model


def run(target, seed, kind):
    seed_model, weak = build_seed_model(target, seed, kind)
    pseudo = pseudo_labels(seed_model, weak)
    st_model = self_train(pseudo, seed)
    v_seed = SU.eval_val_within(seed_model, target)
    v_st = SU.eval_val_within(st_model, target)
    deployed = st_model if v_st > v_seed else seed_model
    arm = "st" if kind == "transfer" else "st_milseed"
    out = SU.score_test(deployed, target, arm, seed, save=(kind == "transfer"))
    out["deployed"] = "self_trained" if v_st > v_seed else "seed"
    out["val_seed"], out["val_st"] = v_seed, v_st
    return out


def main():
    blob = json.load(open(SU.RESULTS))
    res = blob["results"]
    for target in TARGETS:
        for kind, arm in (("transfer", "st"), ("mil", "st_milseed")):
            key = "%s/%s/5seed" % (target, arm)
            if key in res:
                continue
            per_seed = [run(target, s, kind) for s in SEEDS5]
            agg = SU.agg([{k: v for k, v in p.items()
                           if k not in ("deployed", "val_seed", "val_st")}
                          for p in per_seed])
            agg["deployed"] = [p["deployed"] for p in per_seed]
            res[key] = agg
            with open(SU.RESULTS, "w") as fh:
                json.dump(blob, fh, indent=1)
            print(key, json.dumps(agg), flush=True)
    print("R4_DONE", flush=True)


if __name__ == "__main__":
    main()
