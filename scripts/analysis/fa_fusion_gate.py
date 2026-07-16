#!/usr/bin/env python
"""FA — fusion/composition $0 gate (ZERO GPU, ZERO test-touch, ZERO Modal).

Measures whether recovering the F44-documented CANCELLED Qwen-text gain on MHC-EN via a
different modality composition converts to ACCURACY (a Pareto move: hate-recall up, non-hate
~flat) or only re-ranks (a rotation, B5-dead). Design/bars are pre-declared in
refine-logs/WAVE4_CANDIDATES.md §2 (candidate FA) and locked in refine-logs/FA_GATE_RECORD.md §"LOCKED
BEFORE READ"; this script only computes.

Machinery = the F44 concat-kNN proxy (scripts/analysis/encoder_swap_geometry.py): raw frozen
per-video features, per-modality L2-norm, cosine top-20 rank/sim-weighted signed kNN vote
(memory=train, decision score>0). This is the §0.3-validated proxy that reproduces the deployed
align (Hadamard) head's downstream dev SIGN (F44 §1: MHC-EN concat Qwen-CLIP = -0.012).
Raw features => the probe is DETERMINISTIC (no head, no training seed); bootstrap/permutation
over dev items give the CIs. Design is locked BEFORE the judged read (house rule).

Arms (all over the same kNN proxy):
  A0  CLIP-concat            = baseline reference for every delta (CLIP img (+) CLIP text, w=0.5)
  A1  Qwen weighted-concat   = z = [sqrt(w).imghat_Q , sqrt(1-w).texthat_Q], w in W_GRID
                              (w->0 = Qwen-text-only; w->1 = Qwen-image-only; the reweight that
                               the align/Hadamard head structurally CANNOT do, cf. classifier.py:120)
  A2  cross-encoder concat   = z = [sqrt(w).imghat_CLIP , sqrt(1-w).texthat_Q], w in W_GRID
                              (angle-(b): CLIP's strong image (+) Qwen's better text)
  A3a Qwen-align (Hadamard)  = imghat_Q (.) texthat_Q  (the deployed fusion; align<->concat control)
  A3b Qwen-concat (w=0.5)    = A1 at w=0.5  (align<->concat control partner)

Run: conda activate HateVideo; OMP_NUM_THREADS=4 python scripts/analysis/fa_fusion_gate.py
"""
import os, sys, json, hashlib
import numpy as np
import torch

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
CACHE = os.path.join(REPO, "data", "CLIP_Embedding")
MODEL = {"CLIP": "openai_clip-vit-large-patch14-336_HF", "Qwen": "Qwen2.5-VL-7B-Instruct_HF"}
DS_DIR = {"MHC-EN": "MHC", "HateMM": "HateMM"}
K = 20
RNG = 20260717
W_GRID = [round(x, 3) for x in np.linspace(0.0, 1.0, 21)]   # 0.00 .. 1.00 step 0.05
# ---- pre-declared bars (locked before read; verbatim from WAVE4_CANDIDATES.md §2.1(d)) ----
BAR_ACC = 0.02        # K-FA-1: Delta acc >= +0.02
BAR_HATE = 0.03       # K-FA-1: Delta hate-recall >= +0.03
BAR_NONHATE = -0.01   # K-FA-1: Delta non-hate-recall >= -0.01
BAR_ORACLE = 0.03     # K-FA-2: oracle-threshold Delta acc >= +0.03 else easy-example ordering
KFA3_TARGET = -0.012  # K-FA-3: concat-proxy MHC-EN dev (Qwen-CLIP) acc must reproduce F44 -0.012
KFA3_TOL = 0.010      # sign-faithful + within +-0.010 (same machinery => near-exact expected)
NBOOT = 1000
NPERM = 1000


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def load_cache(ds, split, enc):
    d = torch.load(os.path.join(CACHE, DS_DIR[ds], f"{split}_{MODEL[enc]}.pt"),
                   map_location="cpu", weights_only=False)
    ids = d["ids"]
    ids = ids[0] if (isinstance(ids, list) and len(ids) == 1 and isinstance(ids[0], list)) else ids
    return list(ids), d["img_feats"].float().numpy(), d["text_feats"].float().numpy(), d["labels"].long().numpy()


def l2n(x, eps=1e-8):
    return x / (np.linalg.norm(x, axis=1, keepdims=True) + eps)


def knn_scores(mem_X, mem_y, q_X, loo=False):
    """cosine-weighted signed kNN vote (encoder_swap_geometry.knn_vote semantics, exact).
    S normalised to true cosine by row norms; top-K; score = sum_j cos_j * (2 y_j - 1)."""
    S = q_X @ mem_X.T
    qn = np.linalg.norm(q_X, axis=1, keepdims=True)
    mn = np.linalg.norm(mem_X, axis=1, keepdims=True)
    S = S / (qn + 1e-8) / (mn.T + 1e-8)
    if loo:
        np.fill_diagonal(S, -np.inf)
    idx = np.argpartition(-S, kth=K, axis=1)[:, :K]
    sc = np.zeros(len(q_X))
    for i in range(len(q_X)):
        nn = idx[i]
        sc[i] = np.sum(S[i, nn] * (2 * mem_y[nn] - 1))
    return sc


def recalls(y, pred):
    hate = float(np.mean(pred[y == 1] == 1)) if (y == 1).any() else float("nan")   # minority=class1
    nonhate = float(np.mean(pred[y == 0] == 0)) if (y == 0).any() else float("nan")
    return hate, nonhate


def macro_f1(y, p):
    f = []
    for c in (0, 1):
        tp = np.sum((p == c) & (y == c)); fp = np.sum((p == c) & (y != c)); fn = np.sum((p != c) & (y == c))
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        f.append(2 * prec * rec / (prec + rec) if prec + rec else 0.0)
    return float(np.mean(f))


def auc(y, score):
    order = np.argsort(score); ranks = np.empty_like(order, dtype=float); ranks[order] = np.arange(1, len(score) + 1)
    n1 = np.sum(y == 1); n0 = np.sum(y == 0)
    if n1 == 0 or n0 == 0:
        return float("nan")
    return float((np.sum(ranks[y == 1]) - n1 * (n1 + 1) / 2) / (n1 * n0))


def oracle_threshold_acc(y, score):
    """B5-style label-oracle operating point: best decision threshold on the vote score (uses dev
    labels ONLY to pick the threshold => an optimistic per-arm ceiling; test never read)."""
    cand = np.unique(score)
    taus = np.concatenate([[-np.inf], (cand[:-1] + cand[1:]) / 2, [np.inf]]) if len(cand) > 1 else np.array([0.0])
    best_acc, best_tau = -1.0, 0.0
    for t in taus:
        a = float(np.mean((score > t).astype(int) == y))
        if a > best_acc:
            best_acc, best_tau = a, float(t)
    p = (score > best_tau).astype(int)
    return best_acc, macro_f1(y, p), best_tau


def build_weighted(img_unit, txt_unit, w):
    """z = [sqrt(w).imghat , sqrt(1-w).texthat]; each block already L2-unit => ||z||=1,
    cosine(z_a,z_b) = w*img_cos + (1-w)*txt_cos (a convex reweight of the two modality cosines)."""
    return np.concatenate([np.sqrt(w) * img_unit, np.sqrt(1.0 - w) * txt_unit], axis=1)


def align_hadamard(img_unit, txt_unit):
    return img_unit * txt_unit   # deployed fusion_mode='align' (classifier.py:120), row-norm != 1


# ---------------------------------------------------------------------------------------------
def dataset_streams(ds):
    """Return per-split L2-normed image/text streams for CLIP and Qwen, aligned by common id."""
    out = {}
    for split in ("train", "dev_seen"):
        cids, ci, ct, cy = load_cache(ds, split, "CLIP")
        qids, qi, qt, qy = load_cache(ds, split, "Qwen")
        # align by common ids (CLIP order authoritative; defensive even though orders match)
        qmap = {v: k for k, v in enumerate(qids)}
        keep = [i for i, v in enumerate(cids) if v in qmap]
        ci, ct, cy = ci[keep], ct[keep], cy[keep]
        cids = [cids[i] for i in keep]
        qsel = [qmap[v] for v in cids]
        qi, qt, qy = qi[qsel], qt[qsel], qy[qsel]
        assert np.array_equal(cy, qy), f"label mismatch after id-align {ds}/{split}"
        out[split] = dict(ids=cids, y=cy,
                          clip_img=l2n(ci), clip_txt=l2n(ct), qwen_img=l2n(qi), qwen_txt=l2n(qt))
    return out


def eval_config(streams, Xtr_builder):
    """Given a builder(split_dict)->feature matrix, compute train-LOO + train->dev read-outs."""
    tr, dv = streams["train"], streams["dev_seen"]
    Xtr = Xtr_builder(tr); Xdv = Xtr_builder(dv)
    dv_sc = knn_scores(Xtr, tr["y"], Xdv, loo=False)
    tr_sc = knn_scores(Xtr, tr["y"], Xtr, loo=True)
    dv_pred = (dv_sc > 0).astype(int); tr_pred = (tr_sc > 0).astype(int)
    hr, nhr = recalls(dv["y"], dv_pred)
    o_acc, o_mf1, o_tau = oracle_threshold_acc(dv["y"], dv_sc)
    return dict(dev_acc=float(np.mean(dv_pred == dv["y"])), dev_mf1=macro_f1(dv["y"], dv_pred),
                dev_auc=auc(dv["y"], dv_sc), dev_hate_recall=hr, dev_nonhate_recall=nhr,
                train_loo_acc=float(np.mean(tr_pred == tr["y"])),
                oracle_acc=o_acc, oracle_mf1=o_mf1, oracle_tau=o_tau,
                dev_scores=dv_sc, dev_pred=dv_pred)


def arm_configs(streams):
    """All A1/A2 w-configs + A3 controls + A0 baseline; returns dict name->eval + builders."""
    cfgs = {}
    cfgs["A0_CLIP_concat"] = eval_config(streams, lambda s: build_weighted(s["clip_img"], s["clip_txt"], 0.5))
    for w in W_GRID:
        cfgs[f"A1_Qwen_w{w:.2f}"] = eval_config(streams, lambda s, w=w: build_weighted(s["qwen_img"], s["qwen_txt"], w))
        cfgs[f"A2_cross_w{w:.2f}"] = eval_config(streams, lambda s, w=w: build_weighted(s["clip_img"], s["qwen_txt"], w))
    # A3a Qwen-align: raw-feature Hadamard control (both Qwen streams 3584-d). NB CLIP raw streams
    # have mismatched dims (1024 vs 768) so a raw CLIP-Hadamard is undefined; the deployed align head
    # only Hadamards AFTER learned 1024-d projections. This arm is a control, not the F44 proxy.
    cfgs["A3a_Qwen_align"] = eval_config(streams, lambda s: align_hadamard(s["qwen_img"], s["qwen_txt"]))
    return cfgs


def boot_delta_acc(y, pred_cfg, pred_base, n=NBOOT, seed=RNG):
    """bootstrap CI of (acc_cfg - acc_base) over dev items."""
    rng = np.random.default_rng(seed)
    dc = (pred_cfg == y).astype(int); db = (pred_base == y).astype(int)
    diffs = dc - db
    boots = np.array([diffs[rng.integers(0, len(diffs), len(diffs))].mean() for _ in range(n)])
    return float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))


def run_dataset(ds):
    streams = dataset_streams(ds)
    cfgs = arm_configs(streams)
    y = streams["dev_seen"]["y"]
    base = cfgs["A0_CLIP_concat"]
    base_acc = base["dev_acc"]; base_hr = base["dev_hate_recall"]; base_nhr = base["dev_nonhate_recall"]
    base_oracle = base["oracle_acc"]

    def delta_row(name, c):
        return dict(name=name, dev_acc=round(c["dev_acc"], 4), dev_mf1=round(c["dev_mf1"], 4),
                    dev_auc=round(c["dev_auc"], 4), hate_recall=round(c["dev_hate_recall"], 4),
                    nonhate_recall=round(c["dev_nonhate_recall"], 4), train_loo_acc=round(c["train_loo_acc"], 4),
                    oracle_acc=round(c["oracle_acc"], 4), oracle_mf1=round(c["oracle_mf1"], 4),
                    d_acc=round(c["dev_acc"] - base_acc, 4), d_hate=round(c["dev_hate_recall"] - base_hr, 4),
                    d_nonhate=round(c["dev_nonhate_recall"] - base_nhr, 4),
                    d_oracle=round(c["oracle_acc"] - base_oracle, 4))

    rows = {name: delta_row(name, c) for name, c in cfgs.items()}

    # ---- ceiling read: pick, over A1/A2 grids + A3a_Qwen_align, the config maximising dev acc ----
    comp_names = [n for n in cfgs if n.startswith(("A1_Qwen_w", "A2_cross_w")) or n == "A3a_Qwen_align"]
    best_acc_name = max(comp_names, key=lambda n: cfgs[n]["dev_acc"])
    # Pareto-feasible set (K-FA-1 point bars) across the whole grid, then max d_acc among them
    feasible = [n for n in comp_names
                if rows[n]["d_hate"] >= BAR_HATE and rows[n]["d_nonhate"] >= BAR_NONHATE and rows[n]["d_acc"] >= BAR_ACC]
    pareto_exists = len(feasible) > 0
    pareto_name = max(feasible, key=lambda n: rows[n]["d_acc"]) if pareto_exists else best_acc_name

    # the candidate config that K-FA-1 judges = pareto-winner if any, else the max-dev-acc ceiling
    cand = pareto_name
    cand_c = cfgs[cand]
    boot_lo, boot_hi = boot_delta_acc(y, cand_c["dev_pred"], base["dev_pred"])

    # ---- deployable read: train-LOO-selected w per arm ----
    deployable = {}
    for pfx in ("A1_Qwen_w", "A2_cross_w"):
        arm = [n for n in cfgs if n.startswith(pfx)]
        wsel = max(arm, key=lambda n: cfgs[n]["train_loo_acc"])
        deployable[pfx] = dict(selected=wsel, train_loo_acc=round(cfgs[wsel]["train_loo_acc"], 4),
                               dev_d_acc=rows[wsel]["d_acc"], dev_d_hate=rows[wsel]["d_hate"],
                               dev_d_nonhate=rows[wsel]["d_nonhate"])

    # ---- selection permutation null (only meaningful on the primary dataset; run for both) ----
    tr, dv = streams["train"], streams["dev_seen"]
    # candidate arm prefix for the null = the arm the ceiling winner came from
    arm_prefix = "A2" if cand.startswith("A2") else "A1"
    score_mat = []
    for w in W_GRID:
        Xb = ((lambda s, w=w: build_weighted(s["clip_img"], s["qwen_txt"], w)) if arm_prefix == "A2"
              else (lambda s, w=w: build_weighted(s["qwen_img"], s["qwen_txt"], w)))
        score_mat.append(knn_scores(Xb(tr), tr["y"], Xb(dv), loo=False))
    score_mat = np.array(score_mat)
    base_scores = base["dev_scores"]
    rng = np.random.default_rng(RNG + 11)
    null = np.zeros(NPERM)
    for p in range(NPERM):
        yp = y.copy(); rng.shuffle(yp)
        accs = ((score_mat > 0).astype(int) == yp).mean(axis=1)
        base_acc_shuf = float(np.mean((base_scores > 0).astype(int) == yp))
        null[p] = accs.max() - base_acc_shuf
    obs_max_dacc = max(rows[n]["d_acc"] for n in comp_names if n.startswith(arm_prefix + ("_cross_w" if arm_prefix == "A2" else "_Qwen_w")))
    null_p95 = float(np.percentile(null, 95))
    sel_null = dict(arm=arm_prefix, observed_max_d_acc=round(obs_max_dacc, 4), null_p95=round(null_p95, 4),
                    null_mean=round(float(null.mean()), 4), survives=bool(obs_max_dacc > null_p95),
                    p_value=round(float((np.sum(null >= obs_max_dacc) + 1) / (NPERM + 1)), 4))

    # ---- K-FA classification decisions ----
    kfa1_pareto = (pareto_exists and rows[cand]["d_hate"] >= BAR_HATE and rows[cand]["d_nonhate"] >= BAR_NONHATE
                   and rows[cand]["d_acc"] >= BAR_ACC and boot_lo > 0)
    kfa2_pass = rows[cand]["d_oracle"] >= BAR_ORACLE   # oracle-threshold edge converts
    # rotation signature at the max-dev-acc ceiling (for the KILL narrative)
    ceil = rows[best_acc_name]
    rotation_at_ceiling = (ceil["d_acc"] <= 0) and (ceil["d_hate"] > 0) and (ceil["d_nonhate"] < 0)

    result = dict(
        n_train=int(len(tr["y"])), n_dev=int(len(y)), n_dev_hate=int((y == 1).sum()),
        baseline=dict(name="A0_CLIP_concat", dev_acc=round(base_acc, 4), hate_recall=round(base_hr, 4),
                      nonhate_recall=round(base_nhr, 4), oracle_acc=round(base_oracle, 4)),
        anchors=dict(qwen_text_only=rows["A1_Qwen_w0.00"], qwen_img_only=rows["A1_Qwen_w1.00"],
                     qwen_concat_w050=rows["A1_Qwen_w0.50"], qwen_align=rows["A3a_Qwen_align"]),
        ceiling_best_dev_acc=dict(name=best_acc_name, **{k: rows[best_acc_name][k]
                                  for k in ("dev_acc", "d_acc", "d_hate", "d_nonhate", "d_oracle")}),
        pareto_exists=pareto_exists, pareto_feasible_configs=feasible,
        candidate=dict(name=cand, **{k: rows[cand][k] for k in
                       ("dev_acc", "d_acc", "d_hate", "d_nonhate", "oracle_acc", "d_oracle")},
                       boot_d_acc_ci95=[round(boot_lo, 4), round(boot_hi, 4)], boot_ci_low=round(boot_lo, 4)),
        deployable=deployable, selection_null=sel_null,
        rotation_at_ceiling=bool(rotation_at_ceiling),
        K_FA_1_pareto_pass=bool(kfa1_pareto), K_FA_2_oracle_pass=bool(kfa2_pass),
        all_rows=rows)
    return result


def calibration_planted_pareto():
    """Unit calibration of the Pareto/rotation detector: fabricate a prediction set that is a PURE
    hate-recall gain over a baseline (flip some hate FN->TP, touch NO non-hate item) and confirm the
    detector labels it Pareto; then a symmetric-trade set and confirm it labels it rotation."""
    rng = np.random.default_rng(RNG)
    y = np.array([1] * 25 + [0] * 55)                     # MHC-EN-like class balance
    base_pred = y.copy()
    # baseline: miss 10 hate (FN), miss 8 non-hate (FP)
    fn = np.where(y == 1)[0][:10]; fp = np.where(y == 0)[0][:8]
    base_pred[fn] = 0; base_pred[fp] = 1
    b_hr, b_nhr = recalls(y, base_pred); b_acc = np.mean(base_pred == y)
    # planted PURE pareto: fix 6 of the hate FN, leave non-hate untouched
    par = base_pred.copy(); par[fn[:6]] = 1
    p_hr, p_nhr = recalls(y, par); p_acc = np.mean(par == y)
    pareto_ok = (p_hr - b_hr >= BAR_HATE) and (p_nhr - b_nhr >= BAR_NONHATE) and (p_acc - b_acc >= BAR_ACC)
    # planted ROTATION: fix 6 hate FN but break 6 non-hate (symmetric trade)
    rot = base_pred.copy(); rot[fn[:6]] = 1; rot[np.where(y == 0)[0][8:14]] = 1
    r_hr, r_nhr = recalls(y, rot); r_acc = np.mean(rot == y)
    rotation_flagged = (r_hr - b_hr > 0) and (r_nhr - b_nhr < 0) and (r_acc - b_acc <= 0)
    return dict(pareto_detector_fires=bool(pareto_ok), rotation_detector_fires=bool(rotation_flagged),
                planted_pareto=dict(d_hate=round(p_hr - b_hr, 4), d_nonhate=round(p_nhr - b_nhr, 4), d_acc=round(p_acc - b_acc, 4)),
                planted_rotation=dict(d_hate=round(r_hr - b_hr, 4), d_nonhate=round(r_nhr - b_nhr, 4), d_acc=round(r_acc - b_acc, 4)))


def main():
    OUT = {"config": dict(k=K, rng=RNG, w_grid=W_GRID, bar_acc=BAR_ACC, bar_hate=BAR_HATE,
                          bar_nonhate=BAR_NONHATE, bar_oracle=BAR_ORACLE, nboot=NBOOT, nperm=NPERM,
                          proxy="raw-frozen concat/weighted kNN (encoder_swap_geometry semantics)",
                          decision="score>0", deterministic=True),
           "script_sha256": sha(os.path.abspath(__file__))}

    # K-FA-3 machinery validity: concat-proxy MHC-EN dev (Qwen - CLIP) must reproduce F44 -0.012
    mhc = run_dataset("MHC-EN")
    qwen_concat = mhc["all_rows"]["A1_Qwen_w0.50"]["dev_acc"]
    clip_concat = mhc["baseline"]["dev_acc"]
    kfa3_delta = round(qwen_concat - clip_concat, 4)
    kfa3_valid = (kfa3_delta < 0) and (abs(kfa3_delta - KFA3_TARGET) <= KFA3_TOL)
    OUT["K_FA_3_machinery"] = dict(concat_qwen_dev_acc=qwen_concat, concat_clip_dev_acc=clip_concat,
                                   proxy_delta=kfa3_delta, f44_target=KFA3_TARGET, tol=KFA3_TOL, valid=bool(kfa3_valid))

    OUT["calibration"] = calibration_planted_pareto()
    hatemm = run_dataset("HateMM")

    # trim the giant all_rows into the JSON (keep, they are small) but drop numpy arrays inside cfgs
    for r in (mhc, hatemm):
        for row in r["all_rows"].values():
            pass  # rows already plain dicts of floats
    OUT["MHC-EN"] = mhc
    OUT["HateMM"] = hatemm

    # ---- HateMM positive-control / sanity (the Pareto detector MUST fire on the known win) ----
    hm_qwen = hatemm["all_rows"]["A1_Qwen_w0.50"]
    hm_pareto_fires = (hm_qwen["d_hate"] >= BAR_HATE and hm_qwen["d_nonhate"] >= BAR_NONHATE and hm_qwen["d_acc"] >= BAR_ACC)
    # sanity: FA arms must not LOSE on HateMM vs its Qwen-concat known win
    hm_best = hatemm["ceiling_best_dev_acc"]["dev_acc"]
    hm_sanity_ok = hm_best >= hm_qwen["dev_acc"] - 1e-9
    OUT["hatemm_positive_control"] = dict(qwen_concat_vs_clip=hm_qwen, pareto_detector_fires=bool(hm_pareto_fires),
                                          fa_arms_not_worse_than_qwen_concat=bool(hm_sanity_ok))

    # ---- mechanical KILL/PASS (pre-declared, PRIMARY = MHC-EN) ----
    cand = mhc["candidate"]
    kfa1 = mhc["K_FA_1_pareto_pass"]
    kfa2 = mhc["K_FA_2_oracle_pass"]
    seln = mhc["selection_null"]["survives"]
    dep_ok = (mhc["deployable"]["A1_Qwen_w"]["dev_d_acc"] >= BAR_ACC or
              mhc["deployable"]["A2_cross_w"]["dev_d_acc"] >= BAR_ACC)
    mech = dict(
        K_FA_3_valid=bool(kfa3_valid),
        K_FA_1_pareto_exists=bool(mhc["pareto_exists"]),
        K_FA_1_candidate_d_acc=cand["d_acc"], K_FA_1_candidate_d_hate=cand["d_hate"],
        K_FA_1_candidate_d_nonhate=cand["d_nonhate"], K_FA_1_boot_ci_low=cand["boot_ci_low"],
        K_FA_1_pass=bool(kfa1),
        K_FA_2_candidate_d_oracle=cand["d_oracle"], K_FA_2_pass=bool(kfa2),
        selection_null_survives=bool(seln),
        deployable_train_selected_pass=bool(dep_ok),
        rotation_at_ceiling=bool(mhc["rotation_at_ceiling"]),
        hatemm_pareto_control_fires=bool(hm_pareto_fires), hatemm_sanity_ok=bool(hm_sanity_ok),
        calibration_ok=bool(OUT["calibration"]["pareto_detector_fires"] and OUT["calibration"]["rotation_detector_fires"]))

    # Label logic wires the pre-declared kill-switches (K-FA-1 rotation / K-FA-2 oracle / no-Pareto)
    # to the final label. NOTE the K-FA-2 rule is verbatim from the mandate: "If the oracle-threshold
    # acc itself is < +0.03 over CLIP-concat ... => KILL (this is the B5 kill-switch, ported)."
    kill_reason = ""
    if not kfa3_valid:
        label = "MACHINERY_INVALID"
    elif kfa1 and kfa2 and seln and dep_ok and hm_sanity_ok:
        label = "PASS"
    elif cand["d_oracle"] < BAR_ORACLE:
        label = "KILL"                    # pre-declared K-FA-2 B5 kill-switch fires
        kill_reason = (f"K-FA-2 oracle-threshold Delta acc = {cand['d_oracle']:+.4f} < +{BAR_ORACLE} "
                       f"=> AUC edge is easy-example ordering, unconvertible (B5)")
    elif mhc["rotation_at_ceiling"]:
        label = "KILL"                    # pre-declared K-FA-1 rotation signature
        kill_reason = "K-FA-1 rotation (Delta acc<=0 with +hate/-nonhate symmetric trade)"
    elif not mhc["pareto_exists"]:
        label = "KILL"                    # no point-bar config anywhere on the dev-oracle grid
        kill_reason = "no Pareto point-bar config exists on the grid"
    elif kfa1 and kfa2 and (not dep_ok):
        label = "KILL_TRANSFER"           # Pareto+CI+oracle at ceiling but not train-transferable
    else:
        label = "AMBIGUOUS"
    mech["kill_reason"] = kill_reason
    mech["NON_BINDING_LABEL"] = label
    OUT["mechanical"] = mech
    OUT["label"] = label

    # strip dev_scores/dev_pred numpy from serialized all_rows already plain; write
    outp = os.path.join(REPO, "refine-logs", "FA_GATE_OUT.json")
    with open(outp, "w") as f:
        json.dump(OUT, f, indent=2, default=float)

    # ---- console summary ----
    print("=" * 78)
    print(f"K-FA-3 machinery: concat proxy MHC-EN dev Qwen-CLIP = {kfa3_delta:+.4f} "
          f"(F44 target {KFA3_TARGET:+.4f}) valid={kfa3_valid}")
    print(f"calibration: pareto-detector fires={OUT['calibration']['pareto_detector_fires']} "
          f"rotation-detector fires={OUT['calibration']['rotation_detector_fires']}")
    print("-" * 78)
    print("MHC-EN (PRIMARY):")
    b = mhc["baseline"]
    print(f"  baseline CLIP-concat  acc {b['dev_acc']:.4f}  hate-rec {b['hate_recall']:.4f}  "
          f"non-hate-rec {b['nonhate_recall']:.4f}  oracle {b['oracle_acc']:.4f}")
    for k, r in mhc["anchors"].items():
        print(f"  {k:16s} acc {r['dev_acc']:.4f} d_acc {r['d_acc']:+.4f} d_hate {r['d_hate']:+.4f} "
              f"d_nonhate {r['d_nonhate']:+.4f} auc {r['dev_auc']:.4f} oracle {r['oracle_acc']:.4f}")
    c = mhc["ceiling_best_dev_acc"]
    print(f"  CEILING (max dev-acc) = {c['name']}: acc {c['dev_acc']:.4f} d_acc {c['d_acc']:+.4f} "
          f"d_hate {c['d_hate']:+.4f} d_nonhate {c['d_nonhate']:+.4f}  rotation@ceiling={mhc['rotation_at_ceiling']}")
    print(f"  Pareto-feasible configs (K-FA-1 point bars): {mhc['pareto_feasible_configs'] or 'NONE'}")
    cd = mhc["candidate"]
    print(f"  candidate {cd['name']}: d_acc {cd['d_acc']:+.4f} boot-CI {cd['boot_d_acc_ci95']} "
          f"d_oracle {cd['d_oracle']:+.4f}")
    print(f"  deployable(train-w): A1 {mhc['deployable']['A1_Qwen_w']}  A2 {mhc['deployable']['A2_cross_w']}")
    print(f"  selection-null: {mhc['selection_null']}")
    print("-" * 78)
    print("HateMM positive-control:", OUT["hatemm_positive_control"]["pareto_detector_fires"],
          "sanity_ok", OUT["hatemm_positive_control"]["fa_arms_not_worse_than_qwen_concat"])
    print("=" * 78)
    print("MECHANICAL:")
    for k, v in mech.items():
        print(f"  {k} = {v}")
    print(f"\nNON-BINDING LABEL = {label}")
    print(f"wrote {outp}")


if __name__ == "__main__":
    main()
