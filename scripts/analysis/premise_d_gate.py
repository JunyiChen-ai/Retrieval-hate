#!/usr/bin/env python
"""Premise-(d) $0 composition gate (ZERO GPU, ZERO test-touch, ZERO Modal).

Design source: refine-logs/TIE_BRANCH_RECON.md commit 6b9985a, §2-4 (premise-(d) LEAD).
Machinery source: scripts/analysis/fa_fusion_gate.py (FA gate, sha256
9e2fcbf3...) + refine-logs/FA_GATE_RECORD.md §0 locked params. Reused VERBATIM.

QUESTION. F50's FA-A2 arm composed the healthy frozen CLIP-image stream (EN AUC 0.734)
with the *frozen* Qwen-text stream (AUC 0.851) -> composite best-ever EN AUC 0.898, but
d_oracle +0.025 < +0.03 (K-FA-2 kill: AUC edge is easy-example ordering). The F50 ban
carves itself out -- "over banked FROZEN features; conversion requires ADAPTATION (F45)."
Premise-(d) swaps the frozen Qwen-text block for the LoRA-EN-adapted Qwen-text block (the
B4-arm extraction cache, adapter logging/lora/MHC, EN own-train-split only) and re-runs
the FA oracle machinery: does the +0.005 oracle gap close as a PARETO move, or not?

ARMS (all over the raw-frozen concat/weighted kNN proxy, MHC-EN primary):
  A0    CLIP-concat            = [imghat_CLIP , texthat_CLIP] (w=0.5) -- baseline for every delta.
  A2F   FROZEN-text cross      = [sqrt(w).imghat_CLIP , sqrt(1-w).texthat_Qwen(frozen)] -- the
                                 EXACT FA-A2 arm; recomputed here as the machinery-validation
                                 (K-D-0b) reproduction anchor (must match FA_GATE_OUT.json).
  A2L   LoRA-text cross (JUDGED)= [sqrt(w).imghat_CLIP , sqrt(1-w).texthat_Qwen(LoRA-EN)] -- the
                                 premise-(d) composition; the adaptation carve-out the ban names.
  (A1c  Qwen-concat control    = [sqrt(0.5).imghat_Q , sqrt(0.5).texthat_Q] -- frozen Qwen img+txt;
                                 used for the K-FA-3 substrate check + HateMM positive control.)

PRE-DECLARED KILL-SWITCHES (from TIE_BRANCH_RECON.md §2(f), FA-ported):
  K-D-0 (machinery, VOID on fail): planted Pareto/rotation detectors fire (calibration) AND
        the A2F arm reproduces FA-A2 bit-close (AUC 0.898 / ceiling d_oracle +0.025) AND the
        concat proxy reproduces F44 -0.012 (substrate) AND the HateMM positive control would-pass.
  K-D-1 (oracle, BINDING): candidate label-oracle-threshold d_acc vs CLIP-concat < +0.03 => KILL
        (B5 port -- both arms get their OWN dev-oracle tau; dev labels touch threshold only).
  K-D-2 (Pareto-not-rotation): rotation at the ceiling (d_acc<=0 with +hate/-nonhate trade) => KILL.
  K-D-3 (deployable-w sanity): train-LOO-selected w must not regress below CLIP-concat floor.
  + bootstrap CI (1000x, over dev items) and selection-null (shuffle dev y, max-over-w d_acc, 1000x).

Zero test-touch: train + dev_seen features/labels ONLY. Raw-only record; NON-binding label.
Run: conda activate HateVideo; OMP_NUM_THREADS=4 python scripts/analysis/premise_d_gate.py
"""
import os, json, hashlib
import numpy as np
import torch

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
CACHE = os.path.join(REPO, "data", "CLIP_Embedding")
MODEL = {"CLIP": "openai_clip-vit-large-patch14-336_HF",
         "Qwen": "Qwen2.5-VL-7B-Instruct_HF",
         "QwenLoRA": "Qwen2.5-VL-7B-Instruct-LoRA_HF"}
DS_DIR = {"MHC-EN": "MHC", "HateMM": "HateMM"}
K = 20
RNG = 20260717                                          # same seed as FA => reproducible + bit-exact anchor
W_GRID = [round(x, 3) for x in np.linspace(0.0, 1.0, 21)]   # 0.00 .. 1.00 step 0.05
# ---- pre-declared bars (locked before the judged read; FA-ported verbatim) ----
BAR_ACC = 0.02        # K-D point bar: Delta acc >= +0.02
BAR_HATE = 0.03       # K-D point bar: Delta hate-recall >= +0.03
BAR_NONHATE = -0.01   # K-D point bar: Delta non-hate-recall >= -0.01
BAR_ORACLE = 0.03     # K-D-1 (BINDING): oracle-threshold Delta acc >= +0.03 else easy-example ordering
KFA3_TARGET = -0.012  # substrate: concat-proxy MHC-EN dev (Qwen-CLIP) acc reproduces F44 -0.012
KFA3_TOL = 0.010
REPRO_TOL = 1e-4      # K-D-0b: A2F must match FA_GATE_OUT.json A2_cross rows within 4dp rounding
NBOOT = 1000
NPERM = 1000
FA_OUT = os.path.join(REPO, "refine-logs", "FA_GATE_OUT.json")


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
    """cosine-weighted signed kNN vote (encoder_swap_geometry.knn_vote semantics, exact)."""
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
    labels ONLY to pick the threshold => optimistic per-arm ceiling; test never read)."""
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
    """z = [sqrt(w).imghat , sqrt(1-w).texthat]; each block L2-unit => ||z||=1,
    cosine(z_a,z_b) = w*img_cos + (1-w)*txt_cos (convex reweight of the two modality cosines)."""
    return np.concatenate([np.sqrt(w) * img_unit, np.sqrt(1.0 - w) * txt_unit], axis=1)


def eval_config(streams, Xtr_builder):
    """builder(split_dict)->feature matrix; compute train-LOO + train->dev read-outs."""
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


def boot_delta_acc(y, pred_cfg, pred_base, n=NBOOT, seed=RNG):
    rng = np.random.default_rng(seed)
    dc = (pred_cfg == y).astype(int); db = (pred_base == y).astype(int)
    diffs = dc - db
    boots = np.array([diffs[rng.integers(0, len(diffs), len(diffs))].mean() for _ in range(n)])
    return float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))


# ---------------------------------------------------------------------------------------------
def dataset_streams(ds, with_lora):
    """Per-split L2-normed streams aligned by common id. CLIP + frozen Qwen always; LoRA-Qwen text
    only when with_lora (HateMM control never touches the fresh Jul-18 LoRA cache)."""
    out = {}
    shas = {}
    for split in ("train", "dev_seen"):
        cids, ci, ct, cy = load_cache(ds, split, "CLIP")
        qids, qi, qt, qy = load_cache(ds, split, "Qwen")
        shas[f"{split}_CLIP"] = sha(os.path.join(CACHE, DS_DIR[ds], f"{split}_{MODEL['CLIP']}.pt"))
        shas[f"{split}_Qwen"] = sha(os.path.join(CACHE, DS_DIR[ds], f"{split}_{MODEL['Qwen']}.pt"))
        # align CLIP<->Qwen by common id (CLIP order authoritative; identical to FA gate)
        qmap = {v: k for k, v in enumerate(qids)}
        keep = [i for i, v in enumerate(cids) if v in qmap]
        ci, ct, cy = ci[keep], ct[keep], cy[keep]
        cids = [cids[i] for i in keep]
        qsel = [qmap[v] for v in cids]
        qi, qt, qy = qi[qsel], qt[qsel], qy[qsel]
        assert np.array_equal(cy, qy), f"label mismatch after id-align {ds}/{split} (CLIP vs Qwen)"
        rec = dict(ids=cids, y=cy,
                   clip_img=l2n(ci), clip_txt=l2n(ct), qwen_img=l2n(qi), qwen_txt=l2n(qt))
        if with_lora:
            lids, li, lt, ly = load_cache(ds, split, "QwenLoRA")
            shas[f"{split}_QwenLoRA"] = sha(os.path.join(CACHE, DS_DIR[ds], f"{split}_{MODEL['QwenLoRA']}.pt"))
            lmap = {v: k for k, v in enumerate(lids)}
            missing = [v for v in cids if v not in lmap]
            assert not missing, f"{ds}/{split}: {len(missing)} CLIP ids absent from LoRA cache"
            lsel = [lmap[v] for v in cids]
            li, lt, ly = li[lsel], lt[lsel], ly[lsel]
            assert np.array_equal(cy, ly), f"label mismatch after id-align {ds}/{split} (CLIP vs LoRA)"
            rec["lora_img"] = l2n(li); rec["lora_txt"] = l2n(lt)
        out[split] = rec
    out["_shas"] = shas
    return out


def build_arms(streams, with_lora):
    """A0 baseline + A2F(frozen) grid + A2L(LoRA) grid + A1c Qwen-concat control."""
    cfgs = {}
    cfgs["A0_CLIP_concat"] = eval_config(streams, lambda s: build_weighted(s["clip_img"], s["clip_txt"], 0.5))
    cfgs["A1c_Qwen_concat_w0.50"] = eval_config(streams, lambda s: build_weighted(s["qwen_img"], s["qwen_txt"], 0.5))
    for w in W_GRID:
        cfgs[f"A2F_frozen_w{w:.2f}"] = eval_config(streams, lambda s, w=w: build_weighted(s["clip_img"], s["qwen_txt"], w))
        if with_lora:
            cfgs[f"A2L_lora_w{w:.2f}"] = eval_config(streams, lambda s, w=w: build_weighted(s["clip_img"], s["lora_txt"], w))
    return cfgs


def delta_rows(cfgs, base):
    ba, bh, bnh, bo = base["dev_acc"], base["dev_hate_recall"], base["dev_nonhate_recall"], base["oracle_acc"]
    rows = {}
    for name, c in cfgs.items():
        rows[name] = dict(name=name, dev_acc=round(c["dev_acc"], 4), dev_mf1=round(c["dev_mf1"], 4),
                          dev_auc=round(c["dev_auc"], 4), hate_recall=round(c["dev_hate_recall"], 4),
                          nonhate_recall=round(c["dev_nonhate_recall"], 4), train_loo_acc=round(c["train_loo_acc"], 4),
                          oracle_acc=round(c["oracle_acc"], 4), oracle_mf1=round(c["oracle_mf1"], 4),
                          d_acc=round(c["dev_acc"] - ba, 4), d_hate=round(c["dev_hate_recall"] - bh, 4),
                          d_nonhate=round(c["dev_nonhate_recall"] - bnh, 4),
                          d_oracle=round(c["oracle_acc"] - bo, 4))
    return rows


def judge_arm(streams, cfgs, rows, base, arm_prefix, builder):
    """Ceiling / pareto / candidate / boot / deployable / selection-null for a given cross arm."""
    y = streams["dev_seen"]["y"]
    names = [n for n in cfgs if n.startswith(arm_prefix)]
    # ceiling = max dev-acc config on this arm's grid
    best_acc_name = max(names, key=lambda n: cfgs[n]["dev_acc"])
    # pareto-feasible set across the whole grid (K-D point bars), then max d_acc among them
    feasible = [n for n in names
                if rows[n]["d_hate"] >= BAR_HATE and rows[n]["d_nonhate"] >= BAR_NONHATE and rows[n]["d_acc"] >= BAR_ACC]
    pareto_exists = len(feasible) > 0
    cand = max(feasible, key=lambda n: rows[n]["d_acc"]) if pareto_exists else best_acc_name
    cand_c = cfgs[cand]
    boot_lo, boot_hi = boot_delta_acc(y, cand_c["dev_pred"], base["dev_pred"])

    # deployable read: train-LOO-selected w on this arm, evaluated on dev
    wsel = max(names, key=lambda n: cfgs[n]["train_loo_acc"])
    deployable = dict(selected=wsel, train_loo_acc=round(cfgs[wsel]["train_loo_acc"], 4),
                      dev_d_acc=rows[wsel]["d_acc"], dev_d_hate=rows[wsel]["d_hate"],
                      dev_d_nonhate=rows[wsel]["d_nonhate"])

    # selection-null: shuffle dev y, recompute max-over-w d_acc, 1000x
    tr, dv = streams["train"], streams["dev_seen"]
    score_mat = np.array([knn_scores(builder(tr, w), tr["y"], builder(dv, w), loo=False) for w in W_GRID])
    base_scores = base["dev_scores"]
    rng = np.random.default_rng(RNG + 11)
    null = np.zeros(NPERM)
    for p in range(NPERM):
        yp = y.copy(); rng.shuffle(yp)
        accs = ((score_mat > 0).astype(int) == yp).mean(axis=1)
        base_acc_shuf = float(np.mean((base_scores > 0).astype(int) == yp))
        null[p] = accs.max() - base_acc_shuf
    obs_max_dacc = max(rows[n]["d_acc"] for n in names)
    null_p95 = float(np.percentile(null, 95))
    sel_null = dict(observed_max_d_acc=round(obs_max_dacc, 4), null_p95=round(null_p95, 4),
                    null_mean=round(float(null.mean()), 4), survives=bool(obs_max_dacc > null_p95),
                    p_value=round(float((np.sum(null >= obs_max_dacc) + 1) / (NPERM + 1)), 4))

    # rotation signature at the ceiling (for the K-D-2 narrative)
    ceil = rows[best_acc_name]
    rotation_at_ceiling = (ceil["d_acc"] <= 0) and (ceil["d_hate"] > 0) and (ceil["d_nonhate"] < 0)

    return dict(
        ceiling=dict(name=best_acc_name, **{k: rows[best_acc_name][k] for k in
                     ("dev_acc", "dev_auc", "d_acc", "d_hate", "d_nonhate", "oracle_acc", "d_oracle")}),
        pareto_exists=pareto_exists, pareto_feasible_configs=feasible,
        candidate=dict(name=cand, **{k: rows[cand][k] for k in
                       ("dev_acc", "dev_auc", "d_acc", "d_hate", "d_nonhate", "oracle_acc", "d_oracle")},
                       boot_d_acc_ci95=[round(boot_lo, 4), round(boot_hi, 4)], boot_ci_low=round(boot_lo, 4)),
        deployable=deployable, selection_null=sel_null, rotation_at_ceiling=bool(rotation_at_ceiling))


def calibration_planted_pareto():
    """Unit calibration of the Pareto/rotation detector (FA verbatim)."""
    y = np.array([1] * 25 + [0] * 55)                     # MHC-EN-like class balance
    base_pred = y.copy()
    fn = np.where(y == 1)[0][:10]; fp = np.where(y == 0)[0][:8]
    base_pred[fn] = 0; base_pred[fp] = 1
    b_hr, b_nhr = recalls(y, base_pred); b_acc = np.mean(base_pred == y)
    par = base_pred.copy(); par[fn[:6]] = 1
    p_hr, p_nhr = recalls(y, par); p_acc = np.mean(par == y)
    pareto_ok = (p_hr - b_hr >= BAR_HATE) and (p_nhr - b_nhr >= BAR_NONHATE) and (p_acc - b_acc >= BAR_ACC)
    rot = base_pred.copy(); rot[fn[:6]] = 1; rot[np.where(y == 0)[0][8:14]] = 1
    r_hr, r_nhr = recalls(y, rot); r_acc = np.mean(rot == y)
    rotation_flagged = (r_hr - b_hr > 0) and (r_nhr - b_nhr < 0) and (r_acc - b_acc <= 0)
    return dict(pareto_detector_fires=bool(pareto_ok), rotation_detector_fires=bool(rotation_flagged),
                planted_pareto=dict(d_hate=round(p_hr - b_hr, 4), d_nonhate=round(p_nhr - b_nhr, 4), d_acc=round(p_acc - b_acc, 4)),
                planted_rotation=dict(d_hate=round(r_hr - b_hr, 4), d_nonhate=round(r_nhr - b_nhr, 4), d_acc=round(r_acc - b_acc, 4)))


def a2f_reproduction_check(rows):
    """K-D-0b: the A2F(frozen) rows here must reproduce FA_GATE_OUT.json A2_cross rows (bit-close after
    4dp rounding) -- same machinery, same caches. This is the FA-A2 validity anchor (AUC 0.898)."""
    fa = json.load(open(FA_OUT))["MHC-EN"]["all_rows"]
    fields = ("dev_acc", "dev_auc", "d_acc", "d_hate", "d_nonhate", "oracle_acc", "d_oracle")
    max_diff = 0.0; worst = None; per_w = {}
    for w in W_GRID:
        me = rows[f"A2F_frozen_w{w:.2f}"]; fao = fa[f"A2_cross_w{w:.2f}"]
        diffs = {}
        for f in fields:
            fav = fao[f] if f in fao else fao.get(f)
            d = abs(me[f] - fav)
            diffs[f] = d
            if d > max_diff:
                max_diff, worst = d, (f"w{w:.2f}", f, me[f], fav)
        per_w[f"w{w:.2f}"] = round(max(diffs.values()), 6)
    # anchor numbers (verbatim from FA): peak dev AUC and ceiling d_oracle
    peak_auc = max(rows[f"A2F_frozen_w{w:.2f}"]["dev_auc"] for w in W_GRID)
    ceil_name = max((f"A2F_frozen_w{w:.2f}" for w in W_GRID), key=lambda n: rows[n]["dev_acc"])
    return dict(max_abs_diff_vs_FA=round(max_diff, 6), worst=worst, per_w_maxdiff=per_w,
                reproduced=bool(max_diff <= REPRO_TOL),
                peak_dev_auc=round(peak_auc, 4), ceiling_name=ceil_name,
                ceiling_d_oracle=rows[ceil_name]["d_oracle"], ceiling_dev_acc=rows[ceil_name]["dev_acc"])


def main():
    OUT = {"config": dict(k=K, rng=RNG, w_grid=W_GRID, bar_acc=BAR_ACC, bar_hate=BAR_HATE,
                          bar_nonhate=BAR_NONHATE, bar_oracle=BAR_ORACLE, nboot=NBOOT, nperm=NPERM,
                          proxy="raw-frozen concat/weighted kNN (encoder_swap_geometry semantics)",
                          decision="score>0", deterministic=True, repro_tol=REPRO_TOL),
           "script_sha256": sha(os.path.abspath(__file__)),
           "fa_gate_out_sha256": sha(FA_OUT)}

    # ============================ MACHINERY VALIDATION (K-D-0) ============================
    OUT["calibration"] = calibration_planted_pareto()

    mhc = dataset_streams("MHC-EN", with_lora=True)
    OUT["cache_sha256"] = {"MHC-EN": mhc["_shas"]}
    cfgs = build_arms(mhc, with_lora=True)
    base = cfgs["A0_CLIP_concat"]
    rows = delta_rows(cfgs, base)

    # K-D-0b: A2F reproduces FA-A2 (the AUC-0.898 validity anchor)
    repro = a2f_reproduction_check(rows)
    OUT["K_D_0b_A2F_reproduction"] = repro

    # K-FA-3 substrate: concat proxy MHC-EN dev (Qwen - CLIP) reproduces F44 -0.012
    kfa3_delta = round(rows["A1c_Qwen_concat_w0.50"]["dev_acc"] - base["dev_acc"], 4)
    kfa3_valid = (kfa3_delta < 0) and (abs(kfa3_delta - KFA3_TARGET) <= KFA3_TOL)
    OUT["K_FA_3_substrate"] = dict(concat_qwen_dev_acc=round(cfgs["A1c_Qwen_concat_w0.50"]["dev_acc"], 4),
                                   concat_clip_dev_acc=round(base["dev_acc"], 4),
                                   proxy_delta=kfa3_delta, f44_target=KFA3_TARGET, tol=KFA3_TOL, valid=bool(kfa3_valid))

    # K-D-0c: HateMM positive control (frozen Qwen-concat vs CLIP-concat; detector MUST fire, d_oracle>=+0.03)
    hm = dataset_streams("HateMM", with_lora=False)
    OUT["cache_sha256"]["HateMM"] = hm["_shas"]
    hm_cfgs = build_arms(hm, with_lora=False)
    hm_base = hm_cfgs["A0_CLIP_concat"]
    hm_rows = delta_rows(hm_cfgs, hm_base)
    hm_qwen = hm_rows["A1c_Qwen_concat_w0.50"]
    hm_pareto_fires = (hm_qwen["d_hate"] >= BAR_HATE and hm_qwen["d_nonhate"] >= BAR_NONHATE and hm_qwen["d_acc"] >= BAR_ACC)
    hm_oracle_converts = hm_qwen["d_oracle"] >= BAR_ORACLE
    # sanity: A2F/A2 arms must not lose the HateMM Qwen-concat win
    hm_best_a2f = max(hm_rows[f"A2F_frozen_w{w:.2f}"]["dev_acc"] for w in W_GRID)
    hm_sanity_ok = hm_best_a2f >= hm_qwen["dev_acc"] - 1e-9
    OUT["hatemm_positive_control"] = dict(qwen_concat_vs_clip=hm_qwen, pareto_detector_fires=bool(hm_pareto_fires),
                                          oracle_converts=bool(hm_oracle_converts),
                                          a2f_not_worse_than_qwen_concat=bool(hm_sanity_ok),
                                          baseline=dict(dev_acc=round(hm_base["dev_acc"], 4),
                                                        oracle_acc=round(hm_base["oracle_acc"], 4)))

    calib_ok = OUT["calibration"]["pareto_detector_fires"] and OUT["calibration"]["rotation_detector_fires"]
    machinery_valid = bool(calib_ok and repro["reproduced"] and kfa3_valid
                           and hm_pareto_fires and hm_oracle_converts and hm_sanity_ok)
    OUT["K_D_0_machinery_valid"] = machinery_valid

    # ============================ JUDGED READ (premise-(d), A2L) ============================
    builder_A2L = lambda s, w: build_weighted(s["clip_img"], s["lora_txt"], w)
    jr = judge_arm(mhc, cfgs, rows, base, "A2L_lora_w", builder_A2L)
    # frozen anchor's own ceiling judged read (for side-by-side; NOT the binding object)
    builder_A2F = lambda s, w: build_weighted(s["clip_img"], s["qwen_txt"], w)
    jr_frozen = judge_arm(mhc, cfgs, rows, base, "A2F_frozen_w", builder_A2F)

    y = mhc["dev_seen"]["y"]
    cand = jr["candidate"]
    A2L_rows = {n: rows[n] for n in rows if n.startswith("A2L_lora_w")}
    A2F_rows = {n: rows[n] for n in rows if n.startswith("A2F_frozen_w")}

    OUT["MHC-EN"] = dict(
        n_train=int(len(mhc["train"]["y"])), n_dev=int(len(y)), n_dev_hate=int((y == 1).sum()),
        baseline=dict(name="A0_CLIP_concat", dev_acc=round(base["dev_acc"], 4),
                      hate_recall=round(base["dev_hate_recall"], 4), nonhate_recall=round(base["dev_nonhate_recall"], 4),
                      oracle_acc=round(base["oracle_acc"], 4)),
        anchors=dict(lora_text_only=A2L_rows["A2L_lora_w0.00"], lora_w050=A2L_rows["A2L_lora_w0.50"],
                     clip_img_only=A2L_rows["A2L_lora_w1.00"],
                     frozen_text_only=A2F_rows["A2F_frozen_w0.00"]),
        judged_A2L=jr, frozen_anchor_A2F=jr_frozen,
        A2L_all_rows=A2L_rows, A2F_all_rows=A2F_rows)

    # ============================ PARETO-vs-ROTATION DECOMPOSITION (mandatory) ============================
    decomp = dict(
        candidate=dict(name=cand["name"], d_acc=cand["d_acc"], d_hate=cand["d_hate"], d_nonhate=cand["d_nonhate"],
                       shape=("pareto" if (cand["d_hate"] >= BAR_HATE and cand["d_nonhate"] >= BAR_NONHATE and cand["d_acc"] >= BAR_ACC)
                              else ("rotation" if (cand["d_acc"] <= 0 and cand["d_hate"] > 0 and cand["d_nonhate"] < 0)
                                    else "neither"))),
        ceiling=dict(name=jr["ceiling"]["name"], d_acc=jr["ceiling"]["d_acc"], d_hate=jr["ceiling"]["d_hate"],
                     d_nonhate=jr["ceiling"]["d_nonhate"], rotation=jr["rotation_at_ceiling"]))
    OUT["MHC-EN"]["pareto_rotation_decomposition"] = decomp

    # ============================ MECHANICAL KILL/PASS (pre-declared, verbatim) ============================
    kd1_pass = cand["d_oracle"] >= BAR_ORACLE                                    # K-D-1 (BINDING)
    kd_point = (cand["d_hate"] >= BAR_HATE and cand["d_nonhate"] >= BAR_NONHATE and cand["d_acc"] >= BAR_ACC
                and cand["boot_ci_low"] > 0)                                     # K-D point bars + boot CI
    seln = jr["selection_null"]["survives"]
    dep_ok = jr["deployable"]["dev_d_acc"] >= BAR_ACC
    kd3_floor = jr["deployable"]["dev_d_acc"] >= 0.0                             # K-D-3 (not below CLIP floor)

    mech = dict(
        K_D_0_machinery_valid=machinery_valid,
        K_D_0b_A2F_reproduced=bool(repro["reproduced"]), K_D_0b_max_abs_diff_vs_FA=repro["max_abs_diff_vs_FA"],
        K_D_0b_anchor_peak_auc=repro["peak_dev_auc"], K_D_0b_anchor_ceiling_d_oracle=repro["ceiling_d_oracle"],
        K_FA_3_substrate_valid=bool(kfa3_valid), K_FA_3_proxy_delta=kfa3_delta,
        calibration_ok=bool(calib_ok),
        hatemm_pareto_fires=bool(hm_pareto_fires), hatemm_oracle_converts=bool(hm_oracle_converts),
        hatemm_sanity_ok=bool(hm_sanity_ok),
        candidate_name=cand["name"],
        K_D_point_pareto=bool(cand["d_hate"] >= BAR_HATE and cand["d_nonhate"] >= BAR_NONHATE and cand["d_acc"] >= BAR_ACC),
        K_D_candidate_d_acc=cand["d_acc"], K_D_candidate_d_hate=cand["d_hate"],
        K_D_candidate_d_nonhate=cand["d_nonhate"], K_D_boot_ci_low=cand["boot_ci_low"],
        K_D_point_and_boot_pass=bool(kd_point),
        K_D_1_candidate_d_oracle=cand["d_oracle"], K_D_1_pass=bool(kd1_pass),
        selection_null_survives=bool(seln), selection_null_p=jr["selection_null"]["p_value"],
        K_D_2_rotation_at_ceiling=bool(jr["rotation_at_ceiling"]),
        K_D_3_deployable_floor_ok=bool(kd3_floor), deployable_dev_d_acc=jr["deployable"]["dev_d_acc"],
        deployable_train_selected_pass=bool(dep_ok),
        pareto_exists=bool(jr["pareto_exists"]))

    # Label logic (FA decision tree, ported; K-D-1 is the binding switch = FA K-FA-2)
    kill_reason = ""
    if not machinery_valid:
        label = "MACHINERY_INVALID"
        kill_reason = "K-D-0 machinery gate failed (calibration / A2F reproduction / substrate / HateMM control)"
    elif kd1_pass and kd_point and seln and dep_ok and hm_sanity_ok:
        label = "PASS"
    elif cand["d_oracle"] < BAR_ORACLE:
        label = "KILL"                          # K-D-1 (BINDING) B5 kill-switch fires
        kill_reason = (f"K-D-1 oracle-threshold Delta acc = {cand['d_oracle']:+.4f} < +{BAR_ORACLE} "
                       f"=> AUC edge is easy-example ordering, unconvertible (B5 port)")
    elif jr["rotation_at_ceiling"]:
        label = "KILL"                          # K-D-2 rotation signature
        kill_reason = "K-D-2 rotation (Delta acc<=0 with +hate/-nonhate symmetric trade at ceiling)"
    elif not jr["pareto_exists"]:
        label = "KILL"                          # no point-bar config anywhere on the dev-oracle grid
        kill_reason = "no Pareto point-bar config exists on the A2L grid"
    elif kd_point and kd1_pass and (not dep_ok):
        label = "KILL_TRANSFER"                 # Pareto+CI+oracle at ceiling but not train-transferable
        kill_reason = "Pareto+oracle at ceiling but train-LOO-selected w does not transfer (>=+0.02)"
    else:
        label = "AMBIGUOUS"
    mech["kill_reason"] = kill_reason
    mech["NON_BINDING_LABEL"] = label
    OUT["mechanical"] = mech
    OUT["label"] = label

    outp = os.path.join(REPO, "refine-logs", "PREMISE_D_GATE_OUT.json")
    with open(outp, "w") as f:
        json.dump(OUT, f, indent=2, default=float)

    # ---- console summary ----
    print("=" * 82)
    print("PREMISE-(d) $0 GATE  --  CLIP-img (x) LoRA-EN-Qwen-text composition on MHC-EN")
    print("=" * 82)
    print("MACHINERY VALIDATION (K-D-0):")
    print(f"  calibration: pareto-detector fires={OUT['calibration']['pareto_detector_fires']} "
          f"rotation-detector fires={OUT['calibration']['rotation_detector_fires']}")
    print(f"  K-D-0b A2F reproduction of FA-A2: max|diff vs FA| = {repro['max_abs_diff_vs_FA']} "
          f"(tol {REPRO_TOL}) reproduced={repro['reproduced']}")
    print(f"           anchor: peak dev AUC={repro['peak_dev_auc']} (FA 0.8982); "
          f"ceiling {repro['ceiling_name']} d_oracle={repro['ceiling_d_oracle']:+.4f} (FA +0.0250)")
    print(f"  K-FA-3 substrate: Qwen-concat - CLIP-concat = {kfa3_delta:+.4f} (F44 {KFA3_TARGET:+.4f}) valid={kfa3_valid}")
    print(f"  HateMM positive control: pareto fires={hm_pareto_fires} d_oracle={hm_qwen['d_oracle']:+.4f} "
          f"converts={hm_oracle_converts} sanity_ok={hm_sanity_ok}")
    print(f"  >>> K-D-0 machinery_valid = {machinery_valid}")
    print("-" * 82)
    b = OUT["MHC-EN"]["baseline"]
    print(f"MHC-EN JUDGED READ (A2L = CLIP-img (x) LoRA-Qwen-text)  n_train={OUT['MHC-EN']['n_train']} "
          f"n_dev={OUT['MHC-EN']['n_dev']} ({OUT['MHC-EN']['n_dev_hate']} hate)")
    print(f"  baseline A0 CLIP-concat: dev_acc {b['dev_acc']:.4f} hate-rec {b['hate_recall']:.4f} "
          f"non-hate-rec {b['nonhate_recall']:.4f} oracle {b['oracle_acc']:.4f}")
    for k, r in OUT["MHC-EN"]["anchors"].items():
        print(f"  {k:16s} acc {r['dev_acc']:.4f} d_acc {r['d_acc']:+.4f} d_hate {r['d_hate']:+.4f} "
              f"d_nonhate {r['d_nonhate']:+.4f} auc {r['dev_auc']:.4f} d_oracle {r['d_oracle']:+.4f}")
    c = jr["ceiling"]
    print(f"  CEILING (max dev-acc) {c['name']}: acc {c['dev_acc']:.4f} d_acc {c['d_acc']:+.4f} "
          f"d_hate {c['d_hate']:+.4f} d_nonhate {c['d_nonhate']:+.4f} auc {c['dev_auc']:.4f} "
          f"d_oracle {c['d_oracle']:+.4f} rotation@ceiling={jr['rotation_at_ceiling']}")
    print(f"  Pareto-feasible (K-D point bars): {jr['pareto_feasible_configs'] or 'NONE'}")
    print(f"  CANDIDATE {cand['name']}: d_acc {cand['d_acc']:+.4f} boot-CI {cand['boot_d_acc_ci95']} "
          f"d_oracle {cand['d_oracle']:+.4f}")
    print(f"  deployable(train-w): {jr['deployable']}")
    print(f"  selection-null: {jr['selection_null']}")
    print(f"  [frozen anchor A2F ceiling for reference: {jr_frozen['ceiling']['name']} "
          f"d_acc {jr_frozen['ceiling']['d_acc']:+.4f} d_oracle {jr_frozen['ceiling']['d_oracle']:+.4f}]")
    print("-" * 82)
    print("PARETO-vs-ROTATION decomposition:")
    print(f"  candidate shape = {decomp['candidate']['shape']}  "
          f"(d_hate {decomp['candidate']['d_hate']:+.4f} / d_nonhate {decomp['candidate']['d_nonhate']:+.4f})")
    print(f"  ceiling rotation = {decomp['ceiling']['rotation']}")
    print("=" * 82)
    print("MECHANICAL:")
    for k, v in mech.items():
        print(f"  {k} = {v}")
    print(f"\nNON-BINDING LABEL = {label}")
    print(f"wrote {outp}")


if __name__ == "__main__":
    main()
