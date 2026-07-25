#!/usr/bin/env python
"""GRADED 3-CLASS SOFT-LABEL $0 PRE-GATE (ZERO GPU, ZERO SLURM, ZERO test-touch).

Candidate (litsweep-5 S2 cand-1, refine-logs/LITSWEEP5_HATEMM_EN.md): MultiHateClip ships a
3-class Label {Hateful,Offensive,Normal}; the deployed pipeline merges Offensive+Hateful->1
(harmful_vs_normal). Candidate = give Offensive a softer positive target (tau) in HEAD training,
deployment unchanged (binary kNN vote). This $0 CPU pre-gate is READ-ONLY on own-split train+dev
labels: it bounds the VOTE-side (label-reweighting) mechanism the head could exploit, using the
deployed raw fused-key kNN vote as the F79/readout-style proxy operator (NO head, NO test split).

Machinery reused VERBATIM (no vote reimplemented): rank-weighted signed-cosine top-20 vote and
fused-key construction lifted from scripts/analysis/readout_screen.py (== cross_channel_router_gate
.py:73-79 vote / LP_GATE_RECORD.md fused key / ISR_PREGATE_RECORD.md two arms). Parity: the ZH
LoRA_HF binary baseline reproduces readout_screen ro_L28 (loo 0.8717948718 / devtrain 0.8589743590)
bit-exact.

KEY EXACTNESS FACT: retrieval on the fused key is LABEL-INDEPENDENT, so the top-20 neighbour set is
fixed. The vote is LINEAR in the per-neighbour signed target, hence for any Offensive signed weight
w_off:  vote_q(w_off) = A_q + w_off * B_q, where A_q sums Normal(-1)/Hateful(+1) neighbours and B_q
sums the Offensive neighbours' rank-weighted cosine. The oracle w_off sweep and the tau grid are
therefore EXACT (not sampled).

Run: CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=4 python scripts/analysis/gradedlbl_pregate.py
Writes: refine-logs/GRADEDLBL_PREGATE_OUT.json
"""
import os
import sys
import json
import hashlib
from collections import Counter, OrderedDict

import numpy as np
import torch
import faiss

faiss.omp_set_num_threads(4)
REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
CACHE = os.path.join(REPO, "data", "CLIP_Embedding")

TOPK = 20
KILL_ORACLE = 0.030          # PARK iff BINDING oracle ceiling < +0.030 dev acc on BOTH datasets
N_PERM = 500                 # targeted Offensive-permutation null (>=200)
RNG = 20260725
ALLOWED_SPLITS = {"train", "dev_seen"}

# 3-class -> deployed binary (harmful_vs_normal), matches prep_mhc.py LABEL_SCHEMES.
BIN = {"Normal": 0, "Offensive": 1, "Hateful": 1}
# 3-class -> signed vote target for the BINARY baseline (2*bin-1): Normal -1, Off +1, Hate +1.
# graded: Normal -1, Hateful +1, Offensive = w_off (= 2*tau-1).
TAU_GRID = [0.25, 0.50, 0.75]            # pre-declared proxy tau values -> w_off {-0.5, 0.0, +0.5}
WOFF_SWEEP = np.round(np.arange(-1.0, 1.0001, 0.05), 4).tolist()  # oracle sweep (monotone range)

ANN = {
    "MHC":    "/data/jehc223/Multihateclip/English/annotation(new).json",   # EN
    "MHC_zh": "/data/jehc223/Multihateclip/Chinese/annotation(new).json",   # ZH
}
# (dataset, deployed-encoder tag, role). EN frozen-Qwen is the deployed floor (litsweep5: EN ~.79-.81
# frozen); ZH LoRA_HF is deployed (B3) and is the machinery-parity anchor. Sensitivity encoders added.
ARMS = [
    ("MHC",    "Qwen2.5-VL-7B-Instruct_HF",         "primary"),      # EN deployed (frozen)
    ("MHC_zh", "Qwen2.5-VL-7B-Instruct-LoRA_HF",    "primary"),      # ZH deployed (LoRA) + PARITY
    ("MHC",    "Qwen2.5-VL-7B-Instruct-LoRA_HF",    "sensitivity"),  # EN LoRA (transparency)
]
PARITY = {"dataset": "MHC_zh", "tag": "Qwen2.5-VL-7B-Instruct-LoRA_HF",
          "loo": 0.8717948717948718, "devtrain": 0.8589743589743589,
          "src": "refine-logs/READOUT_SCREEN_OUT.json ro_L28"}


def sha256_full(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def load3(ds):
    a = json.load(open(ANN[ds]))
    return {e["Video_ID"]: e["Label"] for e in a}


def load_cache(ds, split, tag):
    assert split in ALLOWED_SPLITS, "test-touch blocked: split={}".format(split)
    path = os.path.join(CACHE, ds, "{}_{}.pt".format(split, tag))
    d = torch.load(path, map_location="cpu")
    ids = d["ids"]
    ids = ids[0] if (isinstance(ids, list) and len(ids) == 1 and isinstance(ids[0], list)) else ids
    return (list(ids), d["img_feats"].float(), d["text_feats"].float(),
            d["labels"].long().numpy(), path)


def fused_key(img, text):
    img = torch.nn.functional.normalize(img, p=2, dim=1)
    text = torch.nn.functional.normalize(text, p=2, dim=1)
    key = torch.cat([img, text], dim=1)
    key = torch.nn.functional.normalize(key, p=2, dim=1)
    return key.numpy().astype("float32")


def retrieve(mem_feat, q_feat, exclude_self):
    """Return per-query (nb_idx[K], nb_sim[K], nb_w[K], W). Retrieval is LABEL-INDEPENDENT."""
    tr = mem_feat.copy()
    q = q_feat.copy()
    faiss.normalize_L2(tr)
    faiss.normalize_L2(q)
    ix = faiss.IndexFlatIP(tr.shape[1])
    ix.add(tr)
    kk = TOPK + (1 if exclude_self else 0)
    D, I = ix.search(q, kk)
    w_full = np.arange(1, TOPK + 1)[::-1].astype("float64")
    per = []
    for i in range(len(q)):
        idx, sim = I[i], D[i]
        if exclude_self:
            keep = idx != i
            idx, sim = idx[keep][:TOPK], sim[keep][:TOPK]
        else:
            idx, sim = idx[:TOPK], sim[:TOPK]
        w = w_full[:len(idx)]
        per.append((idx.astype(np.int64), sim.astype(np.float64), w, float(w.sum())))
    return per


def votes_from_signed(per, mem_signed):
    """vote_q = sum_k mem_signed[idx_k]*sim_k*w_k / W  (verbatim vote form, generalized to signed)."""
    v = np.empty(len(per))
    for i, (idx, sim, w, W) in enumerate(per):
        v[i] = float(np.sum(mem_signed[idx] * sim * w) / W)
    return v


def decompose_AB(per, mem_cls):
    """A_q = rank-wtd cosine of Normal(-1)+Hateful(+1) neighbours / W ; B_q = Offensive coeff / W.
    Then vote_q(w_off) = A_q + w_off * B_q  (exact)."""
    A = np.empty(len(per))
    B = np.empty(len(per))
    for i, (idx, sim, w, W) in enumerate(per):
        cls = mem_cls[idx]
        sgn = np.where(cls == "Normal", -1.0, np.where(cls == "Hateful", 1.0, 0.0))  # Off -> 0 (goes to B)
        off = (cls == "Offensive").astype(np.float64)
        A[i] = float(np.sum(sgn * sim * w) / W)
        B[i] = float(np.sum(off * sim * w) / W)
    return A, B


def acc_at_cutoff(vote, gold, cutoff):
    return float(((vote >= cutoff).astype(int) == gold).mean())


def best_cutoff_acc(vote, gold):
    """Dev-optimal cutoff (B5-style own oracle threshold). Sweep midpoints of sorted votes + 0."""
    cands = sorted(set(list(vote) + [0.0]))
    cuts = [cands[0] - 1e-9]
    for a, b in zip(cands, cands[1:]):
        cuts.append((a + b) / 2.0)
    cuts.append(cands[-1] + 1e-9)
    cuts.append(0.0)  # ensure deployed operating point is a candidate
    best = -1.0
    bcut = 0.0
    for c in cuts:
        a = acc_at_cutoff(vote, gold, c)
        if a > best:
            best, bcut = a, c
    return best, bcut


def screen_arm(ds, tag, mem3, rng):
    ids_tr, img_tr, txt_tr, lab_tr, p_tr = load_cache(ds, "train", tag)
    ids_dv, img_dv, txt_dv, lab_dv, p_dv = load_cache(ds, "dev_seen", tag)
    cls_tr = np.array([mem3[i] for i in ids_tr], dtype=object)
    cls_dv = np.array([mem3[i] for i in ids_dv], dtype=object)
    # binary-consistency guard (cache binary == harmful_vs_normal(3class))
    bin_tr = np.array([BIN[c] for c in cls_tr])
    bin_dv = np.array([BIN[c] for c in cls_dv])
    consistent = bool((bin_tr == lab_tr).all() and (bin_dv == lab_dv).all())

    kt = fused_key(img_tr, txt_tr)
    kd = fused_key(img_dv, txt_dv)

    out = {
        "dataset": ds, "tag": tag,
        "n_train": int(len(ids_tr)), "n_dev": int(len(ids_dv)),
        "cls_train": dict(Counter(cls_tr.tolist())), "cls_dev": dict(Counter(cls_dv.tolist())),
        "bin_pos_train": int(lab_tr.sum()), "bin_pos_dev": int(lab_dv.sum()),
        "bin_consistent": consistent,
        "sha256_train": sha256_full(p_tr), "sha256_dev": sha256_full(p_dv),
        "arms": {},
    }

    for arm in ("loo", "devtrain"):
        if arm == "loo":
            mem_feat = np.concatenate([kt, kd], axis=0)
            mem_cls = np.concatenate([cls_tr, cls_dv])
            mem_bin = np.concatenate([bin_tr, bin_dv])
            per = retrieve(mem_feat, mem_feat, exclude_self=True)
            per = per[len(kt):]                        # score dev rows only
        else:
            mem_feat, mem_cls, mem_bin = kt, cls_tr, bin_tr
            per = retrieve(mem_feat, kd, exclude_self=False)
        gold = lab_dv

        A, B = decompose_AB(per, mem_cls)
        # binary baseline == w_off=+1
        v_bin = A + 1.0 * B
        acc_bin = acc_at_cutoff(v_bin, gold, 0.0)
        bin_best, bin_cut = best_cutoff_acc(v_bin, gold)

        # --- operator (i): tau grid, deployed operating point (cutoff 0) ---
        tau_rows = OrderedDict()
        for tau in TAU_GRID:
            w_off = 2 * tau - 1.0
            v = A + w_off * B
            pred = (v >= 0.0).astype(int)
            a = float((pred == gold).mean())
            r0 = (v_bin >= 0.0).astype(int)
            fix = int(np.sum((r0 != gold) & (pred == gold)))
            brk = int(np.sum((r0 == gold) & (pred != gold)))
            tau_rows["tau=%.2f" % tau] = {
                "w_off": w_off, "acc": a, "dacc": a - acc_bin,
                "fix": fix, "brk": brk, "net": fix - brk}

        # --- operator (ii): oracle ceiling over full w_off sweep ---
        dop = []          # deployed-operating-point (cutoff 0)
        b5 = []           # per-config dev-optimal cutoff
        for w_off in WOFF_SWEEP:
            v = A + w_off * B
            dop.append(acc_at_cutoff(v, gold, 0.0))
            bb, _ = best_cutoff_acc(v, gold)
            b5.append(bb)
        dop = np.array(dop)
        b5 = np.array(b5)
        i_dop = int(np.argmax(dop))
        i_b5 = int(np.argmax(b5))
        oracle_dop_ceiling = float(dop[i_dop] - acc_bin)            # vs deployed binary (cutoff 0)
        oracle_b5_ceiling = float(b5[i_b5] - bin_best)              # vs binary's OWN best cutoff
        binding = max(oracle_dop_ceiling, oracle_b5_ceiling)

        # --- degenerate-recovery assert: w_off=+1 == binary (dacc exactly 0) ---
        v_deg = A + 1.0 * B
        deg_ok = bool(np.max(np.abs(v_deg - v_bin)) == 0.0 and
                      acc_at_cutoff(v_deg, gold, 0.0) == acc_bin)

        # --- targeted Offensive-permutation null (F63) on the best-dacc tau at cutoff 0 ---
        best_tau_key = max(tau_rows, key=lambda k: tau_rows[k]["dacc"])
        w_best = tau_rows[best_tau_key]["w_off"]
        obs_d = tau_rows[best_tau_key]["dacc"]
        pos_idx = np.where(mem_bin == 1)[0]
        n_off = int((mem_cls == "Offensive").sum())
        null_d = []
        for _ in range(N_PERM):
            sel = rng.choice(pos_idx, size=n_off, replace=False)
            signed = np.where(mem_cls == "Normal", -1.0, 1.0)     # all positives -> +1
            signed[sel] = w_best                                   # random equal-size positive subset
            vv = votes_from_signed(per, signed)
            null_d.append(acc_at_cutoff(vv, gold, 0.0) - acc_bin)
        null_d = np.array(null_d)
        null_p95 = float(np.percentile(null_d, 95))

        out["arms"][arm] = {
            "acc_binary": acc_bin, "binary_best_cutoff_acc": bin_best, "binary_best_cutoff": bin_cut,
            "proxy_tau": tau_rows,
            "oracle": {
                "dop_ceiling": oracle_dop_ceiling, "dop_best_woff": WOFF_SWEEP[i_dop],
                "dop_best_acc": float(dop[i_dop]),
                "b5_ceiling": oracle_b5_ceiling, "b5_best_woff": WOFF_SWEEP[i_b5],
                "b5_best_acc": float(b5[i_b5]),
                "binding_ceiling": binding},
            "degenerate_recovery_ok": deg_ok,
            "perm_null": {"best_tau": best_tau_key, "w_off": w_best, "obs_dacc": obs_d,
                          "n_perm": N_PERM, "null_p95": null_p95, "null_mean": float(null_d.mean()),
                          "null_max": float(null_d.max()), "obs_gt_p95": bool(obs_d > null_p95)},
        }
    return out


def main():
    assert os.environ.get("CUDA_VISIBLE_DEVICES", "") == "", "run CPU-only: export CUDA_VISIBLE_DEVICES=''"
    rng = np.random.default_rng(RNG)
    mem3 = {ds: load3(ds) for ds in ANN}
    out = {"meta": {"topk": TOPK, "kill_oracle": KILL_ORACLE, "n_perm": N_PERM, "rng": RNG,
                    "tau_grid": TAU_GRID, "woff_sweep_step": 0.05, "parity_anchor": PARITY,
                    "note": "READ-ONLY on train+dev own-split 3-class labels; test split never opened. "
                            "Oracle bounds the VOTE-side (label-reweighting) mechanism only; the head's "
                            "representation-reshaping is a residual the raw-key proxy cannot see."},
           "results": OrderedDict()}
    for ds, tag, role in ARMS:
        r = screen_arm(ds, tag, mem3[ds], rng)
        r["role"] = role
        out["results"]["{}::{}".format(ds, tag)] = r

    # parity check
    zh = out["results"]["MHC_zh::Qwen2.5-VL-7B-Instruct-LoRA_HF"]
    p_loo = zh["arms"]["loo"]["acc_binary"]
    p_dt = zh["arms"]["devtrain"]["acc_binary"]
    out["parity_check"] = {
        "loo_match": bool(abs(p_loo - PARITY["loo"]) < 1e-12),
        "devtrain_match": bool(abs(p_dt - PARITY["devtrain"]) < 1e-12),
        "loo": p_loo, "devtrain": p_dt}

    # --- pre-declared verdict logic ---
    prim = {k: v for k, v in out["results"].items() if v["role"] == "primary"}
    # per-dataset binding oracle ceiling = max over arms
    ds_ceiling = {}
    ds_proxy_alive = {}
    for k, v in prim.items():
        ds = v["dataset"]
        cmax = max(v["arms"][a]["oracle"]["binding_ceiling"] for a in ("loo", "devtrain"))
        # proxy "alive" = best-tau obs dacc > 0 AND > perm-null p95 in >=1 arm
        alive = any(v["arms"][a]["perm_null"]["obs_dacc"] > 0 and
                    v["arms"][a]["perm_null"]["obs_gt_p95"] for a in ("loo", "devtrain"))
        ds_ceiling[ds] = max(ds_ceiling.get(ds, -9.9), cmax)
        ds_proxy_alive[ds] = ds_proxy_alive.get(ds, False) or alive
    all_below = all(c < KILL_ORACLE for c in ds_ceiling.values())
    any_proxy_alive = any(ds_proxy_alive.values())
    out["verdict"] = {
        "per_dataset_binding_oracle_ceiling": ds_ceiling,
        "per_dataset_proxy_alive": ds_proxy_alive,
        "oracle_all_below_kill": all_below,
        "any_proxy_alive": any_proxy_alive,
        "recommendation": "PARK" if (all_below and not any_proxy_alive) else
                          ("PARK(oracle-below)" if all_below else
                           ("GO-FOR-CEREMONY" if any_proxy_alive else "PARK(proxy-dead)")),
    }

    dest = os.path.join(REPO, "refine-logs", "GRADEDLBL_PREGATE_OUT.json")
    with open(dest, "w") as f:
        json.dump(out, f, indent=2)
    print(json.dumps({"parity": out["parity_check"],
                      "ds_ceiling": ds_ceiling,
                      "proxy_alive": ds_proxy_alive,
                      "recommendation": out["verdict"]["recommendation"]}, indent=2))
    print("wrote", dest)


if __name__ == "__main__":
    main()
