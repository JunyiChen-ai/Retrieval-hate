#!/usr/bin/env python
"""B5 operating-point conversion probe (G0-cond, CPU-only, zero formal GPU).

Companion prereg:  research-wiki/experiments/exp-conv-zh-b5.md  (r1, APPROVED-WITH-AMENDMENTS)
Executable spec:   refine-logs/B5_PROBE_DESIGN.md  (r1: A1/A2/A3/A6/A7/A9 folded)
Heads snapshot:    refine-logs/b5_ckpt_snapshot/  (B5_HEADS_SAFEKEEP_MANIFEST.md, sha256-verified)

STRICT ORDER (violating it invalidates the probe):
  (a) G-REPRO GATE first  -- deployed (vote>=0) test AND dev acc/macroF1/roc must match the 13115
      trainlog anchors to 4dp for all 12 arm x protocol slots. ANY mismatch -> HALT (exit 2),
      no downstream arms computed.
  (b) FREEZE dev-selected thresholds (tau = argmax dev macro-F1, lower-median plateau tie-break).
  (c) ORACLE kill-switch (each arm its OWN test-optimal threshold; paired Qwen-CLIP).
  (d) VAL-CALIBRATED honest preview (frozen tau applied to test).
  (e) D3 guards (>=1000 paired bootstrap w/ COMMON dev-resample index [A6]; tau stability; tax).

This script applies NO pass/fail interpretation (verdict processing is independent, project rule).
It emits raw numbers + a JSON dump; the executor transcribes into refine-logs/B5_PROBE_RECORD.md.
Deterministic + seeded.  Reuses the real repo modules (does NOT reimplement the vote).
"""
import os
import sys
import json
import types
import random
import numpy as np

REPO = "/data/jehc223/RGCL"
SRC = os.path.join(REPO, "src")
SNAP = os.path.join(REPO, "refine-logs", "b5_ckpt_snapshot")
OUT = os.path.join(REPO, "refine-logs", "b5_probe_out")
os.makedirs(OUT, exist_ok=True)
sys.path.insert(0, SRC)

import torch  # noqa: E402
from sklearn.metrics import f1_score  # noqa: E402

# ---- determinism ------------------------------------------------------------
SEED = 0
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.use_deterministic_algorithms(False)  # CPU faiss/linear are already deterministic

from model.classifier import classifier_hateClipper           # noqa: E402
from model.evaluate_rac import retrieve_evaluate_RAC_          # noqa: E402
from utils.metrics import compute_metrics_retrieval            # noqa: E402
from data_loader.dataset import load_feats_from_CLIP           # noqa: E402
from data_loader.rac_dataloader import CLIP2Dataloader         # noqa: E402

CLIP_MODEL = "openai_clip-vit-large-patch14-336_HF"
QWEN_MODEL = "Qwen2.5-VL-7B-Instruct_HF"

# 12 logical slots: (arm, model, seed, protocol, epoch, ckpt_file)   CLIP s0 e29 serves both slots.
SLOTS = [
    ("CLIP", CLIP_MODEL, 0, "final",  29, "CLIP_s0_e29.pt"),
    ("CLIP", CLIP_MODEL, 0, "valsel", 29, "CLIP_s0_e29.pt"),
    ("CLIP", CLIP_MODEL, 1, "final",  29, "CLIP_s1_e29.pt"),
    ("CLIP", CLIP_MODEL, 1, "valsel", 28, "CLIP_s1_e28.pt"),
    ("CLIP", CLIP_MODEL, 2, "final",  29, "CLIP_s2_e29.pt"),
    ("CLIP", CLIP_MODEL, 2, "valsel", 25, "CLIP_s2_e25.pt"),
    ("Qwen", QWEN_MODEL, 0, "final",  29, "Qwen_s0_e29.pt"),
    ("Qwen", QWEN_MODEL, 0, "valsel", 22, "Qwen_s0_e22.pt"),
    ("Qwen", QWEN_MODEL, 1, "final",  29, "Qwen_s1_e29.pt"),
    ("Qwen", QWEN_MODEL, 1, "valsel", 25, "Qwen_s1_e25.pt"),
    ("Qwen", QWEN_MODEL, 2, "final",  29, "Qwen_s2_e29.pt"),
    ("Qwen", QWEN_MODEL, 2, "valsel", 28, "Qwen_s2_e28.pt"),
]

# G-repro anchors -- re-read from the six enc3s_MHC_zh_*_13115.trainlog primary logs (this session,
# cross-checked to B5_PREREG_REVIEW.md sec2 Item-5 and B5_PROBE_DESIGN.md sec4).  (mf1, acc, roc)
TEST_ANCHOR = {
    ("CLIP", 0, "final"): (0.7706, 0.8054, 0.8382), ("CLIP", 0, "valsel"): (0.7706, 0.8054, 0.8382),
    ("CLIP", 1, "final"): (0.7542, 0.8054, 0.8342), ("CLIP", 1, "valsel"): (0.7579, 0.8054, 0.8346),
    ("CLIP", 2, "final"): (0.7913, 0.8322, 0.8444), ("CLIP", 2, "valsel"): (0.7742, 0.8121, 0.8419),
    ("Qwen", 0, "final"): (0.7864, 0.8188, 0.8906), ("Qwen", 0, "valsel"): (0.7412, 0.7919, 0.8838),
    ("Qwen", 1, "final"): (0.7759, 0.8054, 0.8951), ("Qwen", 1, "valsel"): (0.7871, 0.8121, 0.8874),
    ("Qwen", 2, "final"): (0.7514, 0.7852, 0.8806), ("Qwen", 2, "valsel"): (0.7759, 0.8054, 0.8940),
}
DEV_ANCHOR = {
    ("CLIP", 0, "final"): (0.7857, 0.8077, 0.8329), ("CLIP", 0, "valsel"): (0.7857, 0.8077, 0.8329),
    ("CLIP", 1, "final"): (0.7225, 0.7692, 0.8879), ("CLIP", 1, "valsel"): (0.7471, 0.7821, 0.8836),
    ("CLIP", 2, "final"): (0.7645, 0.7949, 0.8764), ("CLIP", 2, "valsel"): (0.7894, 0.8205, 0.8343),
    ("Qwen", 0, "final"): (0.7650, 0.7821, 0.8579), ("Qwen", 0, "valsel"): (0.7940, 0.8205, 0.8693),
    ("Qwen", 1, "final"): (0.8050, 0.8205, 0.8864), ("Qwen", 1, "valsel"): (0.8628, 0.8718, 0.9307),
    ("Qwen", 2, "final"): (0.7613, 0.7821, 0.8436), ("Qwen", 2, "valsel"): (0.8301, 0.8462, 0.8514),
}

BAR = 0.03  # goal bar (mean paired delta acc AND macro-F1 >= +0.03, 3/3 sign)


def make_args(model, device=None):
    a = types.SimpleNamespace()
    a.dataset = "MHC_zh"
    a.model = model
    a.path = os.path.join(REPO, "data")
    # device: 'cpu' default; the authorized G-repro fallback sets B5_PROBE_DEVICE=cuda so the head
    # forward runs on the same compute path as 13115 (Faiss_GPU stays False -> CPU faiss either way).
    a.device = device or os.environ.get("B5_PROBE_DEVICE", "cpu")
    a.Faiss_GPU = False
    a.topk = 20
    a.similarity_threshold = -1.0
    a.majority_voting = "arithmetic"
    a.num_layers = 3
    a.proj_dim = 1024
    a.map_dim = 1024
    a.fusion_mode = "align"
    a.dropout = [0.2, 0.4, 0.1]
    a.batch_norm = False
    a.batch_size = 64
    a.save_embed = False
    a.output_path = OUT
    a.archive_feats = None
    a.tarc_vote_gamma = 0.0
    return a


# ---- metric helpers ---------------------------------------------------------
def macro_f1_fast(y, p):
    """macro-F1 identical to sklearn f1_score(average='macro', zero_division=0)."""
    y = np.asarray(y).astype(int)
    p = np.asarray(p).astype(int)
    f1s = []
    for c in (0, 1):
        tp = np.sum((p == c) & (y == c))
        fp = np.sum((p == c) & (y != c))
        fn = np.sum((p != c) & (y == c))
        den = 2 * tp + fp + fn
        f1s.append(0.0 if den == 0 else (2.0 * tp) / den)
    return 0.5 * (f1s[0] + f1s[1])


def bal_acc(y, p):
    y = np.asarray(y).astype(int)
    p = np.asarray(p).astype(int)
    tpr = np.mean(p[y == 1] == 1) if np.any(y == 1) else 0.0
    tnr = np.mean(p[y == 0] == 0) if np.any(y == 0) else 0.0
    return 0.5 * (tpr + tnr)


def acc_at(v, y, tau):
    return float(np.mean(((np.asarray(v) >= tau).astype(int)) == np.asarray(y).astype(int)))


def mf1_at(v, y, tau):
    return float(macro_f1_fast(y, (np.asarray(v) >= tau).astype(int)))


def bal_at(v, y, tau):
    return float(bal_acc(y, (np.asarray(v) >= tau).astype(int)))


def grid(v):
    u = np.unique(np.asarray(v, dtype=float))
    if len(u) == 1:
        return np.array([u[0] - 1e-6, u[0] + 1e-6])
    mids = (u[:-1] + u[1:]) / 2.0
    return np.concatenate([[u.min() - 1e-6], mids, [u.max() + 1e-6]])


def lower_median_idx(mask_idx):
    """A3: lower-median of the full argmax index array (flatnonzero); index-median authoritative."""
    n = len(mask_idx)
    return mask_idx[n // 2 - (1 if n % 2 == 0 else 0)]


def select_tau(v_dev, y_dev, scorer):
    G = grid(v_dev)
    s = np.array([scorer(v_dev, y_dev, t) for t in G])
    plateau = np.flatnonzero(np.isclose(s, s.max()))
    return float(G[lower_median_idx(plateau)])


def oracle_max(v_test, y_test, scorer):
    Gt = grid(v_test)
    vals = np.array([scorer(v_test, y_test, t) for t in Gt])
    j = int(np.argmax(vals))
    return float(vals[j]), float(Gt[j])


# ---- (0) load + dump votes per slot ----------------------------------------
_feat_cache = {}


def load_feats(model):
    if model not in _feat_cache:
        train, dev, test = load_feats_from_CLIP(os.path.join(REPO, "data", "CLIP_Embedding"), "MHC_zh", model)
        _feat_cache[model] = (train, dev, test)
    return _feat_cache[model]


def dump_slot(slot):
    arm, model, seed, proto, epoch, ckpt = slot
    args = make_args(model)
    train, dev, test = load_feats(model)
    # ids for cross-encoder pairing check (shuffle=False eval loaders preserve split order).
    ids_dev = list(dev[0])
    ids_test = list(test[0])
    (train_dl, dev_dl, test_dl), _ = CLIP2Dataloader(
        train, dev, test, batch_size=args.batch_size, return_dataset=True, normalize=False)
    img_dim = int(train[1].shape[1])
    txt_dim = int(train[2].shape[1])
    model_obj = classifier_hateClipper(img_dim, txt_dim, 3, 1024, 1024, "align",
                                       dropout=[0.2, 0.4, 0.1], batch_norm=False, args=args)
    state = torch.load(os.path.join(SNAP, ckpt), map_location="cpu")
    model_obj.load_state_dict(state)
    model_obj.eval()
    model_obj.to(args.device)   # cpu default; cuda for the authorized G-repro fallback

    ld_dev, y_dev = retrieve_evaluate_RAC_(train_dl, dev_dl, model_obj, largest_retrieval=20,
                                           threshold=-1.0, args=args, eval_name="dev", epoch=epoch,
                                           archive_bank=None, target_pack=None)
    ld_test, y_test = retrieve_evaluate_RAC_(train_dl, test_dl, model_obj, largest_retrieval=20,
                                             threshold=-1.0, args=args, eval_name="test", epoch=epoch,
                                             archive_bank=None, target_pack=None)
    (_, _, _, _, _, votes_dev, labels_dev, macro_dev) = compute_metrics_retrieval(
        ld_dev, y_dev, majority_voting="arithmetic", topk=20, use_sim=True)
    (_, _, _, _, _, votes_test, labels_test, macro_test) = compute_metrics_retrieval(
        ld_test, y_test, majority_voting="arithmetic", topk=20, use_sim=True)

    votes_dev = np.asarray(votes_dev, dtype=float)
    votes_test = np.asarray(votes_test, dtype=float)
    labels_dev = np.asarray(labels_dev, dtype=int)
    labels_test = np.asarray(labels_test, dtype=int)
    np.savez(os.path.join(OUT, f"{arm}_s{seed}_{proto}.npz"),
             votes_dev=votes_dev, labels_dev=labels_dev,
             votes_test=votes_test, labels_test=labels_test,
             ids_dev=np.array(ids_dev, dtype=object), ids_test=np.array(ids_test, dtype=object))
    return dict(arm=arm, seed=seed, proto=proto, epoch=epoch, ckpt=ckpt,
                votes_dev=votes_dev, labels_dev=labels_dev,
                votes_test=votes_test, labels_test=labels_test,
                ids_dev=ids_dev, ids_test=ids_test,
                macro_dev=dict(macro_dev), macro_test=dict(macro_test))


def r4(x):
    return round(float(x), 4)


def main():
    dev = os.environ.get("B5_PROBE_DEVICE", "cpu")
    print("=" * 100)
    print(f"B5 OPERATING-POINT CONVERSION PROBE  (head-forward device={dev}; Faiss_GPU=False CPU faiss)")
    if dev != "cpu":
        print("  [device=cuda] AUTHORIZED G-REPRO FALLBACK (review sec5 / prereg sec6.3): single "
              "device='cuda', Faiss_GPU=False eval; matches 13115 compute path bit-for-bit.")
    print("=" * 100)

    # sanity: fast macro-F1 == sklearn on a spread of cases
    rng0 = np.random.default_rng(7)
    for _ in range(200):
        yy = rng0.integers(0, 2, 60); pp = rng0.integers(0, 2, 60)
        assert abs(macro_f1_fast(yy, pp) - f1_score(yy, pp, average="macro", zero_division=0)) < 1e-12
    print("[selfcheck] macro_f1_fast == sklearn f1_score(macro, zero_division=0)  OK")

    slots = {}
    for slot in SLOTS:
        d = dump_slot(slot)
        slots[(d["arm"], d["seed"], d["proto"])] = d
        print(f"[dump] {d['arm']} s{d['seed']} {d['proto']:6s} e{d['epoch']}: "
              f"dev n={len(d['labels_dev'])} pos={int(d['labels_dev'].sum())}  "
              f"test n={len(d['labels_test'])} pos={int(d['labels_test'].sum())}")

    # cross-encoder pairing check: dev/test split identical (ids + labels) across CLIP/Qwen per seed/proto
    for seed in (0, 1, 2):
        for proto in ("final", "valsel"):
            c = slots[("CLIP", seed, proto)]; q = slots[("Qwen", seed, proto)]
            assert c["ids_dev"] == q["ids_dev"], f"dev id order mismatch s{seed} {proto}"
            assert c["ids_test"] == q["ids_test"], f"test id order mismatch s{seed} {proto}"
            assert np.array_equal(c["labels_dev"], q["labels_dev"])
            assert np.array_equal(c["labels_test"], q["labels_test"])
    print("[pairing] CLIP vs Qwen dev/test ids+labels identical per seed/protocol  OK")

    # ===================== (a) G-REPRO GATE =====================
    print("\n" + "-" * 100 + "\n(a) G-REPRO GATE  [deployed vote>=0; test AND dev; acc/mF1 exact-4dp, "
          "roc |d|<=1e-3 (A11)]\n" + "-" * 100)
    grepro = []
    all_pass = True
    for slot in SLOTS:
        arm, _m, seed, proto, epoch, _c = slot
        d = slots[(arm, seed, proto)]
        # recompute deployed (vote>=0) test + dev from dumped votes; roc from macro dict (rank-based)
        t_acc = acc_at(d["votes_test"], d["labels_test"], 0.0)
        t_mf1 = mf1_at(d["votes_test"], d["labels_test"], 0.0)
        t_roc = d["macro_test"]["roc"]
        v_acc = acc_at(d["votes_dev"], d["labels_dev"], 0.0)
        v_mf1 = mf1_at(d["votes_dev"], d["labels_dev"], 0.0)
        v_roc = d["macro_dev"]["roc"]
        ta = TEST_ANCHOR[(arm, seed, proto)]; da = DEV_ANCHOR[(arm, seed, proto)]
        checks = {
            "test_mf1": (r4(t_mf1), ta[0]), "test_acc": (r4(t_acc), ta[1]), "test_roc": (r4(t_roc), ta[2]),
            "dev_mf1": (r4(v_mf1), da[0]), "dev_acc": (r4(v_acc), da[1]), "dev_roc": (r4(v_roc), da[2]),
        }
        # A11 gate tolerance (B5_GATE_AMENDMENT_RULING §B/§C): acc AND macroF1 exact-4dp; roc within
        # |Δ| <= 1e-3 (roc is a rank statistic unused downstream + unsatisfiable at 4dp by any replay).
        ok = all(abs(got - exp) <= (1e-3 if k.endswith("_roc") else 5e-5)
                 for k, (got, exp) in checks.items())
        all_pass = all_pass and ok
        grepro.append(dict(arm=arm, seed=seed, proto=proto, epoch=epoch, ok=ok, checks=checks))
        mism = [k for k, (g, e) in checks.items() if abs(g - e) > (1e-3 if k.endswith("_roc") else 5e-5)]
        print(f"  {arm} s{seed} {proto:6s} e{epoch}: {'PASS' if ok else 'FAIL'}  "
              f"test(mf1/acc/roc)={r4(t_mf1)}/{r4(t_acc)}/{r4(t_roc)} "
              f"dev={r4(v_mf1)}/{r4(v_acc)}/{r4(v_roc)}"
              + (f"   MISMATCH: {mism}" if mism else ""))
    print(f"\n  G-REPRO: {'12/12 PASS' if all_pass else 'FAIL -- see mismatches above'}")

    result = dict(grepro=grepro, grepro_all_pass=bool(all_pass))
    with open(os.path.join(OUT, "b5_conv_probe_results.json"), "w") as f:
        json.dump(result, f, indent=2, default=float)

    if not all_pass:
        # cosmetic banner only (device-accurate); no change to logic / thresholds / strict order.
        print(f"\n*** G-REPRO FAILED (head-forward device={dev}) -> HALT (STRICT ORDER). "
              f"No downstream arms computed. ***")
        if dev == "cpu":
            print("*** Authorized fallback: ONE <=1-min device='cuda', Faiss_GPU=False eval via sbatch. ***")
        else:
            print("*** cuda fallback ALSO mismatched -> replay cannot certify at 4dp; report to the "
                  "verdict/prereg process (no further cuda retries authorized). ***")
        sys.exit(2)

    # ===================== (b) FREEZE dev-selected thresholds =====================
    print("\n" + "-" * 100 + "\n(b) FROZEN dev-selected tau  [argmax dev macro-F1; A3 lower-median plateau]"
          "  (computed from DEV ONLY, before any test eval)\n" + "-" * 100)
    for slot in SLOTS:
        arm, _m, seed, proto, _e, _c = slot
        d = slots[(arm, seed, proto)]
        tau = select_tau(d["votes_dev"], d["labels_dev"], mf1_at)
        tau_bal = select_tau(d["votes_dev"], d["labels_dev"], bal_at)
        d["tau"] = tau; d["tau_bal"] = tau_bal
        d["dev_mf1_at_tau"] = mf1_at(d["votes_dev"], d["labels_dev"], tau)
        print(f"  {arm} s{seed} {proto:6s}: tau*(mF1)={tau:+.5f}  (dev macroF1@tau={d['dev_mf1_at_tau']:.4f})   "
              f"tau(balAcc)={tau_bal:+.5f}")
    result["frozen_tau"] = {f"{k[0]}_s{k[1]}_{k[2]}": dict(tau=slots[k]["tau"], tau_bal=slots[k]["tau_bal"])
                            for k in slots}

    # ===================== (c) ORACLE kill-switch =====================
    print("\n" + "-" * 100 + "\n(c) ORACLE kill-switch  [each arm its OWN test-optimal tau; paired Qwen-CLIP]"
          "\n" + "-" * 100)
    for slot in SLOTS:
        arm, _m, seed, proto, _e, _c = slot
        d = slots[(arm, seed, proto)]
        oa, oat = oracle_max(d["votes_test"], d["labels_test"], acc_at)
        of, oft = oracle_max(d["votes_test"], d["labels_test"], mf1_at)
        d["oracle_acc"], d["oracle_acc_tau"] = oa, oat
        d["oracle_mf1"], d["oracle_mf1_tau"] = of, oft

    oracle_summary = {}
    for proto in ("final", "valsel"):
        per_seed_acc, per_seed_mf1 = [], []
        rows = []
        for seed in (0, 1, 2):
            q = slots[("Qwen", seed, proto)]; c = slots[("CLIP", seed, proto)]
            dacc = q["oracle_acc"] - c["oracle_acc"]
            dmf1 = q["oracle_mf1"] - c["oracle_mf1"]
            per_seed_acc.append(dacc); per_seed_mf1.append(dmf1)
            rows.append((seed, q["oracle_acc"], c["oracle_acc"], dacc, q["oracle_mf1"], c["oracle_mf1"], dmf1))
        m_acc = float(np.mean(per_seed_acc)); m_mf1 = float(np.mean(per_seed_mf1))
        sgn_acc = sum(1 for x in per_seed_acc if x > 0); sgn_mf1 = sum(1 for x in per_seed_mf1 if x > 0)
        eligible = (m_acc >= BAR) and (m_mf1 >= BAR)
        oracle_summary[proto] = dict(mean_dAcc=m_acc, mean_dmF1=m_mf1,
                                     per_seed_dAcc=per_seed_acc, per_seed_dmF1=per_seed_mf1,
                                     sign_dAcc=sgn_acc, sign_dmF1=sgn_mf1, eligible=bool(eligible))
        print(f"\n  [{proto}] ORACLE (each own tau)")
        print(f"    {'seed':>4} {'Qacc':>7} {'Cacc':>7} {'dAcc':>8} {'QmF1':>7} {'CmF1':>7} {'dmF1':>8}")
        for (seed, qa, ca, da, qf, cf, df) in rows:
            print(f"    {seed:>4} {qa:7.4f} {ca:7.4f} {da:+8.4f} {qf:7.4f} {cf:7.4f} {df:+8.4f}")
        print(f"    mean paired dAcc_oracle={m_acc:+.4f} ({sgn_acc}/3 +)   "
              f"mean paired dmF1_oracle={m_mf1:+.4f} ({sgn_mf1}/3 +)   "
              f"ELIGIBLE(AND>=+0.03)={eligible}")
    dead = not (oracle_summary["final"]["eligible"] or oracle_summary["valsel"]["eligible"])
    print(f"\n  KILL-SWITCH (A1, per-protocol AND-eligibility): "
          f"B5 DEAD (neither protocol eligible) = {dead}")
    result["oracle"] = oracle_summary
    result["kill_switch_dead"] = bool(dead)

    # ===================== (d) VAL-CALIBRATED honest preview =====================
    print("\n" + "-" * 100 + "\n(d) HONEST preview  [frozen dev-tau applied to test; paired Qwen-CLIP]"
          "  (computed regardless of (c), labeled)\n" + "-" * 100)
    for slot in SLOTS:
        arm, _m, seed, proto, _e, _c = slot
        d = slots[(arm, seed, proto)]
        d["honest_acc"] = acc_at(d["votes_test"], d["labels_test"], d["tau"])
        d["honest_mf1"] = mf1_at(d["votes_test"], d["labels_test"], d["tau"])
        d["balacc_acc"] = acc_at(d["votes_test"], d["labels_test"], d["tau_bal"])
        d["balacc_mf1"] = mf1_at(d["votes_test"], d["labels_test"], d["tau_bal"])
        d["calib_tax_acc"] = d["oracle_acc"] - d["honest_acc"]
        d["calib_tax_mf1"] = d["oracle_mf1"] - d["honest_mf1"]

    honest_summary = {}
    for proto in ("final", "valsel"):
        per_seed_acc, per_seed_mf1 = [], []
        rows = []
        for seed in (0, 1, 2):
            q = slots[("Qwen", seed, proto)]; c = slots[("CLIP", seed, proto)]
            dacc = q["honest_acc"] - c["honest_acc"]
            dmf1 = q["honest_mf1"] - c["honest_mf1"]
            per_seed_acc.append(dacc); per_seed_mf1.append(dmf1)
            rows.append((seed, q["honest_acc"], c["honest_acc"], dacc, q["honest_mf1"], c["honest_mf1"], dmf1))
        m_acc = float(np.mean(per_seed_acc)); m_mf1 = float(np.mean(per_seed_mf1))
        sgn_acc = sum(1 for x in per_seed_acc if x > 0); sgn_mf1 = sum(1 for x in per_seed_mf1 if x > 0)
        clears = (m_acc >= BAR) and (m_mf1 >= BAR) and (sgn_acc == 3) and (sgn_mf1 == 3)
        honest_summary[proto] = dict(mean_dAcc=m_acc, mean_dmF1=m_mf1,
                                     per_seed_dAcc=per_seed_acc, per_seed_dmF1=per_seed_mf1,
                                     sign_dAcc=sgn_acc, sign_dmF1=sgn_mf1, clears_bar_3of3=bool(clears))
        print(f"\n  [{proto}] HONEST (frozen dev-tau)")
        print(f"    {'seed':>4} {'Qacc':>7} {'Cacc':>7} {'dAcc':>8} {'QmF1':>7} {'CmF1':>7} {'dmF1':>8}")
        for (seed, qa, ca, da, qf, cf, df) in rows:
            print(f"    {seed:>4} {qa:7.4f} {ca:7.4f} {da:+8.4f} {qf:7.4f} {cf:7.4f} {df:+8.4f}")
        print(f"    mean paired dAcc={m_acc:+.4f} ({sgn_acc}/3 +)   mean paired dmF1={m_mf1:+.4f} ({sgn_mf1}/3 +)"
              f"   clears +0.03/+0.03 & 3/3 = {clears}")
    result["honest"] = honest_summary

    # calibration tax + secondary balanced-acc arm
    print("\n  calibration tax (oracle - honest) and secondary balanced-acc arm (sensitivity only):")
    print(f"    {'arm/seed/proto':>20} {'honAcc':>7} {'honmF1':>7} {'orcAcc':>7} {'orcmF1':>7} "
          f"{'taxAcc':>7} {'taxmF1':>7} {'balAcc':>7} {'balmF1':>7}")
    tax_tbl = {}
    for slot in SLOTS:
        arm, _m, seed, proto, _e, _c = slot
        d = slots[(arm, seed, proto)]
        key = f"{arm}_s{seed}_{proto}"
        tax_tbl[key] = dict(honest_acc=d["honest_acc"], honest_mf1=d["honest_mf1"],
                            oracle_acc=d["oracle_acc"], oracle_mf1=d["oracle_mf1"],
                            calib_tax_acc=d["calib_tax_acc"], calib_tax_mf1=d["calib_tax_mf1"],
                            balacc_acc=d["balacc_acc"], balacc_mf1=d["balacc_mf1"],
                            tau=d["tau"], tau_bal=d["tau_bal"])
        print(f"    {key:>20} {d['honest_acc']:7.4f} {d['honest_mf1']:7.4f} {d['oracle_acc']:7.4f} "
              f"{d['oracle_mf1']:7.4f} {d['calib_tax_acc']:7.4f} {d['calib_tax_mf1']:7.4f} "
              f"{d['balacc_acc']:7.4f} {d['balacc_mf1']:7.4f}")
    result["tax_and_secondary"] = tax_tbl

    # ===================== (e) D3 GUARDS =====================
    print("\n" + "-" * 100 + "\n(e) D3 GUARDS  [>=1000 paired bootstrap, COMMON dev-resample index (A6); "
          "tau stability; quantiles]\n" + "-" * 100)
    NB = 1000
    n_dev = len(slots[("CLIP", 0, "final")]["labels_dev"])
    rng = np.random.default_rng(1234)
    boot_idx = [rng.integers(0, n_dev, n_dev) for _ in range(NB)]  # A6: precompute once, reuse for all arms

    def boot_metric(d, metric_fn):
        vd, yd, vt, yt = d["votes_dev"], d["labels_dev"], d["votes_test"], d["labels_test"]
        out = np.empty(NB)
        for b in range(NB):
            ix = boot_idx[b]
            tb = select_tau(vd[ix], yd[ix], mf1_at)  # re-select tau on resampled dev macro-F1
            out[b] = metric_fn(vt, yt, tb)           # apply to FIXED test
        return out

    d3 = {}
    for proto in ("final", "valsel"):
        boot_acc = {a: {} for a in ("CLIP", "Qwen")}
        boot_mf1 = {a: {} for a in ("CLIP", "Qwen")}
        for arm in ("CLIP", "Qwen"):
            for seed in (0, 1, 2):
                d = slots[(arm, seed, proto)]
                boot_acc[arm][seed] = boot_metric(d, acc_at)
                boot_mf1[arm][seed] = boot_metric(d, mf1_at)
        # paired Delta per seed at matched b, and 3-seed-mean paired Delta at matched b
        pair_acc_seed, pair_mf1_seed = {}, {}
        for seed in (0, 1, 2):
            pair_acc_seed[seed] = boot_acc["Qwen"][seed] - boot_acc["CLIP"][seed]
            pair_mf1_seed[seed] = boot_mf1["Qwen"][seed] - boot_mf1["CLIP"][seed]
        mean_pair_acc = np.mean([pair_acc_seed[s] for s in (0, 1, 2)], axis=0)
        mean_pair_mf1 = np.mean([pair_mf1_seed[s] for s in (0, 1, 2)], axis=0)

        def q(a):
            return [float(np.percentile(a, 5)), float(np.percentile(a, 50)), float(np.percentile(a, 95))]

        acc_q = q(mean_pair_acc); mf1_q = q(mean_pair_mf1)
        frag_acc = acc_q[0] <= 0.0; frag_mf1 = mf1_q[0] <= 0.0
        tau_by_seed = [slots[("Qwen", s, proto)]["tau"] for s in (0, 1, 2)]
        ctau_by_seed = [slots[("CLIP", s, proto)]["tau"] for s in (0, 1, 2)]
        d3[proto] = dict(
            mean_pair_dAcc_q_5_50_95=acc_q, mean_pair_dmF1_q_5_50_95=mf1_q,
            d3_fragile_acc=bool(frag_acc), d3_fragile_mf1=bool(frag_mf1),
            per_arm_test_acc_q={a: {s: q(boot_acc[a][s]) for s in (0, 1, 2)} for a in ("CLIP", "Qwen")},
            per_arm_test_mf1_q={a: {s: q(boot_mf1[a][s]) for s in (0, 1, 2)} for a in ("CLIP", "Qwen")},
            qwen_tau_by_seed=tau_by_seed, clip_tau_by_seed=ctau_by_seed,
            qwen_tau_spread=float(np.std(tau_by_seed)), clip_tau_spread=float(np.std(ctau_by_seed)),
        )
        print(f"\n  [{proto}] 3-seed-mean paired Delta bootstrap ({NB} resamples, common idx):")
        print(f"    dAcc  5/50/95 pct = {acc_q[0]:+.4f} / {acc_q[1]:+.4f} / {acc_q[2]:+.4f}   "
              f"5th<=0 (D3-fragile)={frag_acc}")
        print(f"    dmF1  5/50/95 pct = {mf1_q[0]:+.4f} / {mf1_q[1]:+.4f} / {mf1_q[2]:+.4f}   "
              f"5th<=0 (D3-fragile)={frag_mf1}")
        print(f"    Qwen tau by seed = {[round(t,4) for t in tau_by_seed]} (std {np.std(tau_by_seed):.4f}); "
              f"CLIP tau by seed = {[round(t,4) for t in ctau_by_seed]} (std {np.std(ctau_by_seed):.4f})")
    result["d3"] = d3

    with open(os.path.join(OUT, "b5_conv_probe_results.json"), "w") as f:
        json.dump(result, f, indent=2, default=float)
    print("\n" + "=" * 100)
    print("PROBE COMPLETE. JSON -> refine-logs/b5_probe_out/b5_conv_probe_results.json")
    print("Executor applies NO pass/fail interpretation; verdict processing is independent.")
    print("=" * 100)


if __name__ == "__main__":
    main()
