#!/usr/bin/env python
"""$0 GATE — Label Propagation / graph diffusion over the kNN memory graph vs the deployed one-hop vote.

ZERO GPU / ZERO SLURM / ZERO Modal / ZERO test-touch. CPU only. Train + dev_seen only.

Design/bars are PRE-DECLARED in refine-logs/LP_GATE_RECORD.md (written before any dev number was computed).
This script only computes and dumps refine-logs/LP_GATE_OUT.json.

Vote machinery (_weighted_signed_vote, rank-weighted signed-cosine top-20) is lifted VERBATIM from
scripts/analysis/cross_channel_router_gate.py:73-131. LP is the multi-hop generalisation of the SAME read:
identical query->train edges; the only change is the train-side label field (raw +/-1  ->  LLGC-diffused F).

Run: conda activate HateVideo; OMP_NUM_THREADS=4 python scripts/analysis/lp_gate.py
"""
import os, sys, json, hashlib
import numpy as np
import torch
import faiss

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
faiss.omp_set_num_threads(4)
REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
CACHE = os.path.join(REPO, "data", "CLIP_Embedding")
CKPT_DIR = os.path.join(REPO, "refine-logs", "router_ckpt_snapshot")
TOPK = 20
ALPHAS = [0.5, 0.9]
ALPHA_SANITY = 1e-6
NPERM = 200
BAR = 0.030
RNG = 20260720

# (dataset cache-dir, key-space cache tag, role) — the 6 pre-declared cells
CELLS = [
    ("HateMM", "Qwen2.5-VL-7B-Instruct-LoRA-curric_HF", "deployed-best"),
    ("HateMM", "Qwen2.5-VL-7B-Instruct-LoRA_HF", "adaptation"),
    ("HateMM", "Qwen2.5-VL-7B-Instruct_HF", "frozen"),
    ("MHC_zh", "Qwen2.5-VL-7B-Instruct-LoRA_HF", "deployed"),
    ("MHC_zh", "Qwen2.5-VL-7B-Instruct_HF", "frozen"),
    ("MHC", "Qwen2.5-VL-7B-Instruct_HF", "deployed=frozen"),
]
# banked deployed enc3s Val_Retrieval e29 dev acc anchors (head path, seeds 0/1/2) for the machinery sanity
ANCHOR = {("HateMM", "Qwen"): [.8505, .8224, .8505], ("MHC", "Qwen"): [.7625, .7875, .7750]}


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def load_cache(ds, split, tag):
    # HARD test-touch guard: this loader can only ever open train / dev_seen files.
    assert split in ("train", "dev_seen"), f"illegal split {split}"
    path = os.path.join(CACHE, ds, f"{split}_{tag}.pt")
    assert "test" not in os.path.basename(path).lower(), f"REFUSE test path: {path}"
    d = torch.load(path, map_location="cpu")
    ids = d["ids"]
    ids = ids[0] if (isinstance(ids, list) and len(ids) == 1 and isinstance(ids[0], list)) else ids
    return list(ids), d["img_feats"].float().numpy(), d["text_feats"].float().numpy(), d["labels"].long().numpy(), path


def fused_key(img, txt):
    """L2-normalise each stream, concat, L2-renormalise the concatenation -> [N, 7168] float32."""
    def l2(x):
        n = np.linalg.norm(x, axis=1, keepdims=True)
        return x / np.clip(n, 1e-12, None)
    z = np.concatenate([l2(img.astype("float64")), l2(txt.astype("float64"))], axis=1)
    z = z / np.clip(np.linalg.norm(z, axis=1, keepdims=True), 1e-12, None)
    return z.astype("float32")


def knn(train_key, query_key, exclude_self):
    """Top-20 train neighbours by cosine. Returns idx[Nq,20], sim[Nq,20], rankw[20] (=20..1)."""
    tr = np.ascontiguousarray(train_key.copy())
    q = np.ascontiguousarray(query_key.copy())
    faiss.normalize_L2(tr)
    faiss.normalize_L2(q)
    ix = faiss.IndexFlatIP(tr.shape[1])
    ix.add(tr)
    kk = TOPK + (1 if exclude_self else 0)
    D, I = ix.search(q, kk)
    idx = np.zeros((len(q), TOPK), dtype=np.int64)
    sim = np.zeros((len(q), TOPK), dtype=np.float64)
    for i in range(len(q)):
        ii, ss = I[i], D[i]
        if exclude_self:
            keep = ii != i
            ii, ss = ii[keep][:TOPK], ss[keep][:TOPK]
        else:
            ii, ss = ii[:TOPK], ss[:TOPK]
        idx[i], sim[i] = ii, ss
    rankw = np.arange(1, TOPK + 1)[::-1].astype("float64")  # 20,19,...,1  (neighbours sorted desc cosine)
    return idx, sim, rankw


def read_score(idx, sim, rankw, field):
    """s(q) = sum_j rankw_j * sim_j * field[idx_j]  — the rank-weighted signed-cosine read.
    field = +/-1 train labels (one-hop baseline) OR the LLGC-diffused train field F (LP)."""
    return (rankw[None, :] * sim * field[idx]).sum(axis=1)


def build_S(train_key):
    """Symmetric-normalised non-negative kNN affinity S = D^-1/2 W D^-1/2 over TRAIN nodes (diag(W)=0)."""
    idx, sim, _ = knn(train_key, train_key, exclude_self=True)
    N = train_key.shape[0]
    W = np.zeros((N, N), dtype=np.float64)
    relu = np.clip(sim, 0.0, None)
    for i in range(N):
        W[i, idx[i]] = relu[i]
    W = 0.5 * (W + W.T)
    d = W.sum(axis=1)
    dinv = 1.0 / np.sqrt(np.clip(d, 1e-12, None))
    S = (dinv[:, None] * W) * dinv[None, :]
    return S


def diffusion_operator(S, alpha):
    """M = (1-alpha) (I - alpha S)^-1  ; F = M @ Y  (closed-form LLGC, Zhou et al. 2004)."""
    N = S.shape[0]
    return (1.0 - alpha) * np.linalg.inv(np.eye(N) - alpha * S)


def acc(pred, y):
    return float(np.mean(pred == y))


def flips(base_pred, new_pred, y):
    """counts vs baseline: fixed (base wrong -> new right), broken (base right -> new wrong), net."""
    base_ok = base_pred == y
    new_ok = new_pred == y
    fixed = int(np.sum(~base_ok & new_ok))
    broken = int(np.sum(base_ok & ~new_ok))
    return fixed, broken, fixed - broken


# ---------------- head-path deployed-number sanity (frozen-Qwen only; ckpts exist) ----------------
def head_sanity(ds_cachedir, anchor_key):
    from model.classifier import classifier_hateClipper
    from easydict import EasyDict
    tag = "Qwen2.5-VL-7B-Instruct_HF"
    out = {}
    tids, timg, ttxt, tlab, _ = load_cache(ds_cachedir, "train", tag)
    dids, dimg, dtxt, dlab, _ = load_cache(ds_cachedir, "dev_seen", tag)
    for seed in range(3):
        ck = os.path.join(CKPT_DIR, f"{ds_cachedir}_Qwen_s{seed}_e29.pt")
        sd = torch.load(ck, map_location="cpu")
        m = classifier_hateClipper(sd["img_proj.0.weight"].shape[1], sd["text_proj.0.weight"].shape[1],
                                   num_layers=3, proj_dim=1024, map_dim=1024, fusion_mode="align",
                                   dropout=[0.2, 0.4, 0.1], batch_norm=False, args=EasyDict(dataset="X"))
        m.load_state_dict(sd)
        m.eval()
        with torch.no_grad():
            te = m(torch.tensor(timg), torch.tensor(ttxt), return_embed=True)[1].cpu().numpy().astype("float32")
            de = m(torch.tensor(dimg), torch.tensor(dtxt), return_embed=True)[1].cpu().numpy().astype("float32")
        idx, sim, rankw = knn(te, de, exclude_self=False)
        s = read_score(idx, sim, rankw, (tlab * 2 - 1).astype("float64"))
        a = acc((s >= 0).astype(int), dlab)
        anc = ANCHOR[(anchor_key, "Qwen")][seed]
        out[f"seed{seed}"] = dict(head_dev_acc=round(a, 4), anchor=anc, match=bool(abs(a - anc) < 1.1e-4))
    return out


# ---------------- main per-cell gate ----------------
def run_cell(ds, tag):
    tids, timg, ttxt, tlab, tpath = load_cache(ds, "train", tag)
    dids, dimg, dtxt, dlab, dpath = load_cache(ds, "dev_seen", tag)
    tkey = fused_key(timg, ttxt)
    dkey = fused_key(dimg, dtxt)
    y_tr = tlab.astype("float64")
    ytilde = (y_tr * 2 - 1)  # +/-1 signed train labels
    # dev query edges (fixed, label-independent)
    didx, dsim, rankw = knn(tkey, dkey, exclude_self=False)
    # ---- baseline one-hop ----
    s0 = read_score(didx, dsim, rankw, ytilde)
    base_pred = (s0 >= 0).astype(int)
    base_acc = acc(base_pred, dlab)
    # ---- graph + diffusion operators ----
    S = build_S(tkey)
    res = dict(dataset=ds, key_space=tag, n_train=len(tids), n_dev=len(dids),
               dev_pos=int(dlab.sum()), train_pos=int(tlab.sum()),
               baseline_dev_acc=round(base_acc, 4),
               train_cache_sha=sha(tpath)[:16], dev_cache_sha=sha(dpath)[:16], grid={})
    grid_accs = {}
    grid_preds = {}
    for a in ALPHAS:
        M = diffusion_operator(S, a)
        F = M @ ytilde
        s = read_score(didx, dsim, rankw, F)
        pred = (s >= 0).astype(int)
        ac = acc(pred, dlab)
        fx, bk, net = flips(base_pred, pred, dlab)
        grid_accs[a] = ac
        grid_preds[a] = pred
        res["grid"][f"alpha_{a}"] = dict(dev_acc=round(ac, 4), delta=round(ac - base_acc, 4),
                                         fixed=fx, broken=bk, net=net)
    best_a = max(ALPHAS, key=lambda a: grid_accs[a])
    best_delta = grid_accs[best_a] - base_acc
    res["best_alpha"] = best_a
    res["best_delta"] = round(best_delta, 4)
    # ---- (c) sanity: alpha->0 recovers baseline exactly ----
    Ms = diffusion_operator(S, ALPHA_SANITY)
    Fs = Ms @ ytilde
    pred_s = (read_score(didx, dsim, rankw, Fs) >= 0).astype(int)
    res["sanity_alpha0"] = dict(delta=round(acc(pred_s, dlab) - base_acc, 6),
                                n_disagree=int(np.sum(pred_s != base_pred)),
                                OK=bool(np.array_equal(pred_s, base_pred)))
    # ---- (a) oracle headroom: union-correct over grid ----
    union_ok = (base_pred == dlab)
    for a in ALPHAS:
        union_ok = union_ok | (grid_preds[a] == dlab)
    oracle_acc = float(np.mean(union_ok))
    res["oracle_headroom"] = dict(oracle_acc=round(oracle_acc, 4),
                                  headroom=round(oracle_acc - base_acc, 4))
    # ---- (b) permutation null: shuffle TRAIN labels, best-cell delta ----
    rng = np.random.default_rng(RNG)
    Mcache = {a: diffusion_operator(S, a) for a in ALPHAS}
    null = np.zeros(NPERM)
    for p in range(NPERM):
        yp = ytilde.copy()
        rng.shuffle(yp)
        s0p = read_score(didx, dsim, rankw, yp)
        base_p = (s0p >= 0).astype(int)
        base_acc_p = acc(base_p, dlab)
        best = -1e9
        for a in ALPHAS:
            Fp = Mcache[a] @ yp
            pp = (read_score(didx, dsim, rankw, Fp) >= 0).astype(int)
            best = max(best, acc(pp, dlab) - base_acc_p)
        null[p] = best
    res["perm_null"] = dict(nperm=NPERM, mean=round(float(null.mean()), 4),
                            p95=round(float(np.percentile(null, 95)), 4),
                            p05=round(float(np.percentile(null, 5)), 4),
                            max=round(float(null.max()), 4),
                            real_exceeds_p95=bool(best_delta > np.percentile(null, 95)))
    # ---- (d-i) machinery validity: planted-signal recovery (train-LOO read) ----
    tidx, tsim, trankw = knn(tkey, tkey, exclude_self=True)
    Mplant = diffusion_operator(S, 0.9)
    # (pre-declared) sign(PCA-1) — a max-VARIANCE feature axis (NOT graph-smooth; boundary nodes flip)
    Xc = tkey.astype("float64") - tkey.astype("float64").mean(0, keepdims=True)
    _, _, Vt = np.linalg.svd(Xc, full_matrices=False)
    yplant = np.where(Xc @ Vt[0] >= 0, 1.0, -1.0)
    sp = read_score(tidx, tsim, trankw, Mplant @ yplant)
    recov_pca1 = acc((sp >= 0).astype(int), (yplant > 0).astype(int))
    # (correctly-specified confirm) sign of the Fiedler-like signal = 2nd eigenvector of S (graph-smooth)
    evals, evecs = np.linalg.eigh(S)
    fied = evecs[:, np.argsort(evals)[::-1][1]]  # eigenvector of 2nd-largest eigenvalue
    yfied = np.where(fied >= 0, 1.0, -1.0)
    sf = read_score(tidx, tsim, trankw, Mplant @ yfied)
    recov_fiedler = acc((sf >= 0).astype(int), (yfied > 0).astype(int))
    # context: LP vs one-hop train-LOO on REAL labels
    s_loo_base = read_score(tidx, tsim, trankw, ytilde)
    s_loo_lp = read_score(tidx, tsim, trankw, Mplant @ ytilde)
    res["machinery"] = dict(planted_pca1_recovery=round(recov_pca1, 4),
                            planted_fiedler_recovery=round(recov_fiedler, 4),
                            valid=bool(recov_fiedler >= 0.99),
                            train_loo_onehop=round(acc((s_loo_base >= 0).astype(int), tlab), 4),
                            train_loo_lp_a09=round(acc((s_loo_lp >= 0).astype(int), tlab), 4))
    # ---- per-cell verdict ----
    promote = (best_delta >= BAR and res["perm_null"]["real_exceeds_p95"]
               and res["oracle_headroom"]["headroom"] >= 2 * best_delta and res["machinery"]["valid"])
    res["cell_verdict"] = "PROMOTE" if promote else "KILL"
    return res


def main():
    OUT = dict(config=dict(topk=TOPK, alphas=ALPHAS, nperm=NPERM, bar=BAR, rng=RNG,
                           key="fused L2(img)+L2(txt) concat renorm 7168d",
                           vote="rank-weighted signed-cosine top-20 (router_gate:73-79 verbatim)",
                           lp="LLGC F=(1-a)(I-aS)^-1 Y, relu-cosine kNN affinity, sym-norm, diag0",
                           inductive="train-graph only; query->train edges; NO dev-dev, NO test"),
               script_sha256=sha(os.path.abspath(__file__)), cells=[])
    # machinery sanity: reproduce banked deployed dev acc on frozen-Qwen (head ckpts exist)
    OUT["deployed_number_sanity"] = {"HateMM": head_sanity("HateMM", "HateMM"),
                                     "MHC_EN": head_sanity("MHC", "MHC")}
    for ds, tag, role in CELLS:
        print(f"\n===== {ds} / {tag} ({role}) =====")
        r = run_cell(ds, tag)
        r["role"] = role
        OUT["cells"].append(r)
        g = r["grid"]
        print(f"  baseline dev acc {r['baseline_dev_acc']}  n_dev={r['n_dev']}")
        for a in ALPHAS:
            c = g[f"alpha_{a}"]
            print(f"  LP a={a}: dev {c['dev_acc']} delta {c['delta']:+.4f} (fixed {c['fixed']} broken {c['broken']} net {c['net']})")
        print(f"  best delta {r['best_delta']:+.4f} @a={r['best_alpha']}  |  perm-null p95 {r['perm_null']['p95']:+.4f} "
              f"real>p95={r['perm_null']['real_exceeds_p95']}  |  oracle headroom {r['oracle_headroom']['headroom']:+.4f}")
        print(f"  sanity a->0 OK={r['sanity_alpha0']['OK']} (ndisagree {r['sanity_alpha0']['n_disagree']})  |  "
              f"planted-recovery {r['machinery']['planted_pca1_recovery']} valid={r['machinery']['valid']}  |  VERDICT {r['cell_verdict']}")
    # per-dataset verdict = best key space
    OUT["dataset_verdict"] = {}
    for ds in ["HateMM", "MHC_zh", "MHC"]:
        cs = [c for c in OUT["cells"] if c["dataset"] == ds]
        best = max(cs, key=lambda c: c["best_delta"])
        OUT["dataset_verdict"][ds] = dict(
            verdict="PROMOTE" if any(c["cell_verdict"] == "PROMOTE" for c in cs) else "KILL",
            best_key_space=best["key_space"], best_delta=best["best_delta"],
            baseline_dev_acc=best["baseline_dev_acc"])
    outp = os.path.join(REPO, "refine-logs", "LP_GATE_OUT.json")
    with open(outp, "w") as f:
        json.dump(OUT, f, indent=2)
    print("\n===== DATASET VERDICTS =====")
    for ds, v in OUT["dataset_verdict"].items():
        print(f"  {ds}: {v['verdict']}  best_delta {v['best_delta']:+.4f} @ {v['best_key_space']}")
    print(f"\nwrote {outp}")


if __name__ == "__main__":
    main()
