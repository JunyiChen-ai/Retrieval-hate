#!/usr/bin/env python
"""mech_probes_a.py -- the ONE driver for the MECH-PROBES-A $0 CPU kill probes.

Frozen design: refine-logs/MECH_PROBES_A_PREREG.md (rules frozen before any probe
metric was computed).

    Probe 1  NCA softmax concentration on the deployed head key space   -> RVS
    Probe 2  fold-head vs full-head deployed-vote margin gap (ZH)       -> XFM
    Probe 3  epoch-snapshot asymmetric bank in the F113 fold arena      -> AQM (zero-param)

Probes 1 and 2 read the banked C06 mints read-only and train nothing.  Probe 3 re-mints
30 fold heads by CALLING the frozen scripts/analysis/headspace_mint.py main() unmodified
(the c06_falsifier_mint.py pattern), adding one per-epoch snapshot hook.

TEST CONTACT: NONE.  Three layers, all inherited: headspace_mint's torch.load guard, the
frozen load_split (train_*.pt / dev_seen_*.pt only), and c09_guard on PYTHONPATH.  Every
query in every probe is a train-split item; K_dev is never read.

STAGES (one process each, driven by scripts/slurm/mech_probes_a_cpu.sbatch):
    probe12   read banked mints -> probe1.json, probe2.json
    mint      one (dataset, seed, fold) head re-mint with epoch snapshots
    probe3    aggregate the 30 snapshot files -> probe3.json
    report    fold the three probe files into MECH_PROBES_A_RESULT.json
    selftest  synthetic end-to-end drive of every numeric path (no cluster data)

COST: CPU only, <= 8 threads.  Zero GPU, zero cloud.
"""
import argparse
import hashlib
import json
import os
import shutil
import sys
import time
from datetime import datetime, timezone

import numpy as np

_T_START = time.time()

REPO = "/data/jehc223/RGCL"
sys.path.insert(0, os.path.join(REPO, "src"))
sys.path.insert(0, os.path.join(REPO, "scripts/analysis"))

DATASETS = ("hatemm", "zh")
SEEDS = (0, 1, 2)
FOLDS = (0, 1, 2, 3, 4)
SNAP_EPOCHS = (10, 15, 20, 25, 29)
NCA_TAU = 0.1                      # src/model/loss.py:647, --nca_tau default, arm-pinned
TOPK = 20                          # deployed
PARITY_TOL = 1e-6                  # prereg §1 Probe 3 frozen tie-break

BANKED_MINTS = os.path.join(REPO, "artifacts/c06_falsifier/mints")

# Banked fold-head floors, prereg §1 Probe 3 (scripts/analysis/headspace_arena_*_OUT.json).
BANKED_FLOOR_ACC = {"hatemm": [0.8884, 0.8858, 0.8858],
                    "zh": [0.8929, 0.8895, 0.8946]}
BANKED_FLOOR_MF1 = {"hatemm": [0.8838, 0.8811, 0.8812],
                    "zh": [0.8747, 0.8710, 0.8765]}

PROJECTED_SECONDS = float(os.environ.get("MPA_PROJECTED_SECONDS", 1210.0))

FROZEN_SHA = {
    "scripts/analysis/headspace_mint.py":
        "cefdf8dc2f4a9aefa042ef7bec9b1d06c9721ae5b4a70ec117e9929ff0916612",
    "scripts/analysis/mechnov_pairverify.py":
        "77b0defd8eaa3688e58b6d5d17202bd55d16cf1f4a5aaafbe4b2b98598b7240d",
    "scripts/analysis/mechfix_ops.py": None,     # recorded, not pinned (not frozen upstream)
}


def sha256_of(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for blk in iter(lambda: f.read(1 << 20), b""):
            h.update(blk)
    return h.hexdigest()


def assert_frozen():
    """Refuse to run if a module this driver reuses verbatim has changed."""
    got = {}
    for rel, want in FROZEN_SHA.items():
        g = sha256_of(os.path.join(REPO, rel))
        got[rel] = g
        if want is not None and g != want:
            raise AssertionError("FROZEN MODULE CHANGED: {} {}".format(rel, g))
    return got


def heartbeat(progress_path, phase, done=None, total=None, extra=""):
    """Line-buffered, append-only, one handle per call (no descriptor held across a
    40 s head train).  Mandatory per the C09 process rule."""
    stamp = datetime.now(timezone.utc).isoformat(timespec="seconds")
    units = "{}/{}".format(done, total) if total is not None else "-"
    elapsed = time.time() - _T_START
    line = "{} | {} | {} | {:.1f}s | {:.3f}x{}".format(
        stamp, phase, units, elapsed, elapsed / PROJECTED_SECONDS,
        (" | " + extra) if extra else "")
    if progress_path:
        try:
            os.makedirs(os.path.dirname(progress_path), exist_ok=True)
            with open(progress_path, "a", buffering=1) as fh:
                fh.write(line + "\n")
        except Exception:
            pass
    print(line, flush=True)


def f64(X):
    """Independent float64 copy.  mechfix_ops._norm32 L2-normalises IN PLACE on whatever
    np.asarray(X, 'float32') hands back, so every array crossing into it is materialised
    here first and never reused by the caller afterwards."""
    return np.ascontiguousarray(np.asarray(X, dtype="float64"))


def load_banked(ds, seed, fold):
    tag = "full" if fold < 0 else str(fold)
    p = os.path.join(BANKED_MINTS, "mint_{}_N_s{}_f{}.npz".format(ds, seed, tag))
    if not os.path.exists(p):
        raise AssertionError("banked mint absent: {}".format(p))
    return np.load(p, allow_pickle=True)


# ===================================================================== PROBE 1
def nca_concentration(keys, tau=NCA_TAU, topk=TOPK):
    """Per-anchor NCA softmax over all OTHER train items, exactly the geometry
    src/model/loss.py:649-657 builds: L2-normalised keys, cosine logits / tau, the
    anchor's own row masked to -inf.

    Returns (mean exp(entropy)/N, mean top-`topk` probability mass, N)."""
    K = f64(keys)
    K = K / np.maximum(np.linalg.norm(K, axis=1, keepdims=True), 1e-12)
    n = K.shape[0]
    logits = (K @ K.T) / float(tau)
    np.fill_diagonal(logits, -np.inf)
    logits = logits - logits.max(axis=1, keepdims=True)
    P = np.exp(logits)
    P /= P.sum(axis=1, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        lp = np.where(P > 0, np.log(P), 0.0)
    H = -(P * lp).sum(axis=1)
    ess = np.exp(H) / float(n)
    part = np.partition(P, n - topk, axis=1)[:, n - topk:]
    top_mass = part.sum(axis=1)
    return float(ess.mean()), float(top_mass.mean()), int(n)


def run_probe1(outdir, progress=None):
    cells, per_ds = [], {}
    for ds in DATASETS:
        vals = []
        for seed in SEEDS:
            z = load_banked(ds, seed, -1)
            ess, mass, n = nca_concentration(z["K_train"])
            cells.append({"dataset": ds, "seed": seed, "n": n,
                          "mean_ess_frac": round(ess, 6),
                          "mean_top20_mass": round(mass, 6),
                          "uniform_top20_mass": round(TOPK / float(n), 6)})
            vals.append(mass)
            heartbeat(progress, "PROBE1", len(cells), 6,
                      "{} s{} top20_mass={:.6f} ess_frac={:.6f}".format(ds, seed, mass, ess))
        per_ds[ds] = float(np.mean(vals))

    kill = all(per_ds[d] > 0.5 for d in DATASETS)
    alive = all(per_ds[d] < 0.1 for d in DATASETS)
    verdict = "KILLED" if kill else ("ALIVE" if alive else "INDETERMINATE")
    out = {"probe": "P1_nca_softmax_concentration", "candidate": "RVS",
           "tau": NCA_TAU, "topk": TOPK, "cells": cells,
           "mean_top20_mass_by_dataset": {k: round(v, 6) for k, v in per_ds.items()},
           "rule": "top20_mass > 0.5 on both -> RVS KILLED; < 0.1 on both -> RVS ALIVE; "
                   "else INDETERMINATE",
           "verdict": verdict}
    write_json(os.path.join(outdir, "probe1.json"), out)
    return out


# ===================================================================== PROBE 2
def fold_and_full_margins(ds, seed, MECH):
    """(a) full head, LOO row-exclusion; (b) fold head, item out of fold.
    (b) is gate_floor's protocol verbatim (c06_falsifier_arena.py:1167-1178)."""
    zf = load_banked(ds, seed, -1)
    lab = np.asarray(zf["lab"]).astype(int)
    fold_of = np.asarray(zf["fold_of"]).astype(int)
    Kfull = f64(zf["K_train"])
    v_full, _, _, _ = MECH.deployed_vote(Kfull, lab, f64(Kfull), topk=TOPK,
                                         exclude_self=True)
    v_fold = np.full(len(lab), np.nan, dtype="float64")
    acc_fold_pred = np.full(len(lab), -1, dtype=int)
    for f in FOLDS:
        K = f64(load_banked(ds, seed, f)["K_train"])
        ho = np.flatnonzero(fold_of == f)
        fit = np.flatnonzero(fold_of != f)
        v, p, _, _ = MECH.deployed_vote(K[fit], lab[fit], f64(K[ho]), topk=TOPK)
        v_fold[ho] = v
        acc_fold_pred[ho] = p
    assert np.isfinite(v_fold).all(), "fold margins incomplete"
    return lab, v_full, v_fold, (v_full >= 0).astype(int), acc_fold_pred


def run_probe2(outdir, progress=None):
    import mechfix_ops as MECH
    from scipy import stats
    per_ds, done = {}, 0
    for ds in DATASETS:
        vf, vd, accs_loo, accs_fold, shifts = [], [], [], [], []
        for seed in SEEDS:
            lab, v_full, v_fold, p_loo, p_fold = fold_and_full_margins(ds, seed, MECH)
            vf.append(v_full)
            vd.append(v_fold)
            shifts.append(v_fold - v_full)
            accs_loo.append(float((p_loo == lab).mean()))
            accs_fold.append(float((p_fold == lab).mean()))
            done += 1
            heartbeat(progress, "PROBE2", done, 6,
                      "{} s{} loo_acc={:.4f} fold_acc={:.4f}".format(
                          ds, seed, accs_loo[-1], accs_fold[-1]))
        a = np.concatenate(vf)
        b = np.concatenate(vd)
        d = np.concatenate(shifts)
        ks = stats.ks_2samp(a, b)
        per_ds[ds] = {
            "n_pooled": int(len(a)),
            "ks_stat": round(float(ks.statistic), 6),
            "ks_p": float(ks.pvalue),
            "mean_abs_margin_shift": round(float(np.abs(d).mean()), 6),
            "mean_margin_shift": round(float(d.mean()), 6),
            "mean_margin_full_loo": round(float(a.mean()), 6),
            "mean_margin_fold": round(float(b.mean()), 6),
            "acc_full_loo_by_seed": [round(x, 4) for x in accs_loo],
            "acc_fold_by_seed": [round(x, 4) for x in accs_fold],
        }
    z = per_ds["zh"]
    killed = (z["ks_p"] > 0.05) and (z["mean_abs_margin_shift"] < 0.02)
    out = {"probe": "P2_fold_vs_full_margin_gap", "candidate": "XFM",
           "decided_on": "zh", "per_dataset": per_ds,
           "rule": "ZH: KS p > 0.05 AND mean|margin shift| < 0.02 -> XFM KILLED; "
                   "otherwise ALIVE. HateMM reported descriptively only.",
           "verdict": "KILLED" if killed else "ALIVE"}
    write_json(os.path.join(outdir, "probe2.json"), out)
    return out


# ============================================================ PROBE 3 -- mint
def snapshot_mint(ds, seed, fold, out, scratch, threads, progress=None):
    """Train ONE fold head under the frozen recipe, snapshotting the full-train key
    matrix at the end of each epoch in SNAP_EPOCHS.

    The frozen headspace_mint.main() is CALLED, never re-implemented.  The only addition
    is a wrapper around run_rac.eval_and_save_epoch_end (src/run_rac.py:875 -- the last
    call of the epoch body; no optimiser step follows it inside the epoch).  The wrapper
    runs the original first, then forwards the FULL train split through the head under
    torch.no_grad() in whatever mode the original left it (eval; metrics.py:546), and
    restores that mode exactly.  Dropout is inert in eval mode and this recipe has no
    BatchNorm (--batch_norm False), so the extra forwards draw no RNG and cannot perturb
    the trajectory -- which the cross-run parity check against the banked mint tests
    directly rather than assuming."""
    assert_frozen()
    import headspace_mint as HM
    import mechnov_pairverify as P
    import torch
    import run_rac

    if os.path.exists(out):
        heartbeat(progress, "MINT-SKIP", extra="{} s{} f{} (resume)".format(ds, seed, fold))
        return

    cfg_ds = P.DATASETS[ds]
    tr = HM.load_split(cfg_ds["cache_dir"], "train", cfg_ds["model"])
    tr_img, tr_txt = tr[1], tr[2]

    snaps = {}

    _orig_epoch_end = run_rac.eval_and_save_epoch_end

    def _epoch_end_with_snapshot(*ar, **kw):
        res = _orig_epoch_end(*ar, **kw)
        assert len(ar) >= 7, ("eval_and_save_epoch_end call shape changed "
                              "(src/run_rac.py:875 passes 7 positional args)")
        model, epoch = ar[5], ar[6]
        if int(epoch) in SNAP_EPOCHS:
            was_training = model.training
            model.eval()
            with torch.no_grad():
                _, emb = model(tr_img, tr_txt, return_embed=True)
            snaps[int(epoch)] = emb.detach().cpu().numpy().astype("float64")
            if was_training:
                model.train()
        return res

    run_rac.eval_and_save_epoch_end = _epoch_end_with_snapshot

    stage_dir = os.path.join(scratch, "mpa_stage", os.path.basename(out))
    if os.path.isdir(stage_dir):
        shutil.rmtree(stage_dir)
    os.makedirs(stage_dir, exist_ok=True)
    stage_npz = os.path.join(stage_dir, "frozen_mint.npz")

    argv_saved = sys.argv
    sys.argv = ["headspace_mint.py", "--dataset", ds, "--seed", str(seed),
                "--fold", str(fold), "--out", stage_npz, "--scratch", stage_dir,
                "--threads", str(threads)]
    t0 = time.time()
    try:
        HM.main()
    finally:
        sys.argv = argv_saved
        run_rac.eval_and_save_epoch_end = _orig_epoch_end
    secs = time.time() - t0

    assert os.path.exists(stage_npz), "frozen mint produced no output"
    missing = [e for e in SNAP_EPOCHS if e not in snaps]
    assert not missing, "snapshot hook never fired for epochs {}".format(missing)

    z = np.load(stage_npz, allow_pickle=True)
    K_final = np.asarray(z["K_train"], dtype="float64")
    # In-process HALT check: the epoch-29 snapshot and the frozen module's own post-run
    # K_train are the same forward of the same weights.  Any difference means the hook
    # is not at the end of the last epoch.
    inproc = float(np.abs(snaps[29] - K_final).max())
    assert inproc == 0.0, ("HALT: epoch-29 snapshot != frozen K_train (max abs {:g}) -- "
                           "hook is not at end-of-epoch".format(inproc))
    # Cross-run parity against the banked C06 mint: this is the trajectory test.
    banked = float(np.abs(K_final - np.asarray(load_banked(ds, seed, fold)["K_train"],
                                               dtype="float64")).max())

    meta = json.loads(str(z["meta"]))
    meta["mech_probes_a"] = {
        "driver_sha256": sha256_of(os.path.abspath(__file__)),
        "snap_epochs": list(SNAP_EPOCHS),
        "secs": round(secs, 1),
        "inprocess_epoch29_vs_frozen_K_train_maxabs": inproc,
        "banked_parity_maxabs": banked,
        "banked_parity_pass": bool(banked <= PARITY_TOL),
        "parity_tol": PARITY_TOL,
    }
    arrays = {"K_e{}".format(e): snaps[e] for e in SNAP_EPOCHS}
    tmp = out + ".tmp.npz"
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    np.savez(tmp, lab=z["lab"], fold_of=z["fold_of"], fit_idx=z["fit_idx"],
             meta=json.dumps(meta), **arrays)
    os.replace(tmp, out)
    z.close()
    shutil.rmtree(stage_dir, ignore_errors=True)
    heartbeat(progress, "MINT-DONE",
              extra="{} s{} f{} {:.1f}s banked_parity_maxabs={:.3g}".format(
                  ds, seed, fold, secs, banked))


# ======================================================= PROBE 3 -- aggregate
def fold_arena_accuracy(snapdir, ds, seed, t_star, MECH):
    """Bank = fitting-pool keys at epoch t*, query = held-out fifth keys at epoch 29,
    both from the SAME fold head's trajectory.  Deployed top-20 vote."""
    lab = fold_of = None
    pred = None
    for f in FOLDS:
        z = np.load(os.path.join(snapdir, "snap_{}_s{}_f{}.npz".format(ds, seed, f)),
                    allow_pickle=True)
        if lab is None:
            lab = np.asarray(z["lab"]).astype(int)
            fold_of = np.asarray(z["fold_of"]).astype(int)
            pred = np.full(len(lab), -1, dtype=int)
        Kb = f64(z["K_e{}".format(t_star)])
        Kq = f64(z["K_e29"])
        ho = np.flatnonzero(fold_of == f)
        fit = np.flatnonzero(fold_of != f)
        _, p, _, _ = MECH.deployed_vote(Kb[fit], lab[fit], Kq[ho], topk=TOPK)
        pred[ho] = p
    assert (pred >= 0).all(), "arena predictions incomplete"
    return MECH.acc(lab, pred), MECH.macro_f1(lab, pred)


def run_probe3(outdir, snapdir, progress=None):
    import mechfix_ops as MECH
    parity, per = [], {}
    done = 0
    for ds in DATASETS:
        per[ds] = {}
        for t in SNAP_EPOCHS:
            accs, mf1s = [], []
            for seed in SEEDS:
                a, m = fold_arena_accuracy(snapdir, ds, seed, t, MECH)
                accs.append(a)
                mf1s.append(m)
                done += 1
                heartbeat(progress, "PROBE3", done, len(DATASETS) * len(SNAP_EPOCHS) * 3,
                          "{} s{} t*={} acc={:.4f} mF1={:.4f}".format(ds, seed, t, a, m))
            per[ds][t] = {"acc_by_seed": [round(x, 4) for x in accs],
                          "mF1_by_seed": [round(x, 4) for x in mf1s],
                          "acc_mean": round(float(np.mean(accs)), 6),
                          "mF1_mean": round(float(np.mean(mf1s)), 6)}
        for seed in SEEDS:
            for f in FOLDS:
                z = np.load(os.path.join(snapdir, "snap_{}_s{}_f{}.npz".format(ds, seed, f)),
                            allow_pickle=True)
                mm = json.loads(str(z["meta"]))["mech_probes_a"]
                parity.append({"dataset": ds, "seed": seed, "fold": f,
                               "banked_parity_maxabs": mm["banked_parity_maxabs"],
                               "pass": mm["banked_parity_pass"],
                               "secs": mm["secs"]})

    parity_all_pass = all(p["pass"] for p in parity)
    floor_src = "banked" if parity_all_pass else "own_remint_epoch29"
    floors = {}
    for ds in DATASETS:
        floors[ds] = {
            "banked_acc_mean": round(float(np.mean(BANKED_FLOOR_ACC[ds])), 6),
            "banked_mF1_mean": round(float(np.mean(BANKED_FLOOR_MF1[ds])), 6),
            "remint_e29_acc_mean": per[ds][29]["acc_mean"],
            "remint_e29_mF1_mean": per[ds][29]["mF1_mean"],
        }
        floors[ds]["primary_acc_mean"] = (floors[ds]["banked_acc_mean"] if parity_all_pass
                                          else floors[ds]["remint_e29_acc_mean"])

    deltas = {}
    for ds in DATASETS:
        base = floors[ds]["primary_acc_mean"]
        deltas[ds] = {t: round(per[ds][t]["acc_mean"] - base, 6) for t in SNAP_EPOCHS}
    clearing = [t for t in SNAP_EPOCHS
                if all(deltas[ds][t] >= 0.020 for ds in DATASETS)]
    if clearing:
        best_t = max(clearing, key=lambda t: min(deltas[ds][t] for ds in DATASETS))
        verdict = "ALIVE"
    else:
        best_t = max(SNAP_EPOCHS, key=lambda t: min(deltas[ds][t] for ds in DATASETS))
        verdict = "KILLED"

    out = {"probe": "P3_epoch_snapshot_asymmetric_bank",
           "candidate": "AQM_zero_parameter_realization",
           "snap_epochs": list(SNAP_EPOCHS), "per_dataset": per,
           "floors": floors, "floor_source": floor_src,
           "acc_delta_vs_floor": {ds: {str(k): v for k, v in deltas[ds].items()}
                                  for ds in DATASETS},
           "parity": parity, "parity_all_pass": parity_all_pass,
           "best_t_star": best_t,
           "best_t_star_deltas": {ds: deltas[ds][best_t] for ds in DATASETS},
           "best_t_star_mF1_delta": {
               ds: round(per[ds][best_t]["mF1_mean"]
                         - (floors[ds]["banked_mF1_mean"] if parity_all_pass
                            else floors[ds]["remint_e29_mF1_mean"]), 6)
               for ds in DATASETS},
           "rule": "no t* with >= +0.020 acc (3-seed mean) over the epoch-29 floor on BOTH "
                   "datasets -> AQM zero-parameter realization KILLED; the trained-g_phi "
                   "version is a separate later question either way.",
           "verdict": verdict}
    write_json(os.path.join(outdir, "probe3.json"), out)
    return out


# ===================================================================== report
def write_json(path, obj):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w") as fh:
        json.dump(obj, fh, indent=2, sort_keys=True)
    os.replace(tmp, path)
    print("[mpa] wrote {}".format(path), flush=True)


def run_report(outdir, progress=None):
    parts = {}
    for name in ("probe1", "probe2", "probe3"):
        p = os.path.join(outdir, name + ".json")
        parts[name] = json.load(open(p)) if os.path.exists(p) else None
    out = {"prereg": "refine-logs/MECH_PROBES_A_PREREG.md",
           "driver_sha256": sha256_of(os.path.abspath(__file__)),
           "frozen_sha256": assert_frozen(),
           "verdicts": {
               "RVS": parts["probe1"]["verdict"] if parts["probe1"] else "MISSING",
               "XFM": parts["probe2"]["verdict"] if parts["probe2"] else "MISSING",
               "AQM_zero_parameter": parts["probe3"]["verdict"] if parts["probe3"] else "MISSING",
           },
           "probes": parts,
           "elapsed_seconds": round(time.time() - _T_START, 1)}
    write_json(os.path.join(outdir, "MECH_PROBES_A_RESULT.json"), out)
    heartbeat(progress, "REPORT", extra=json.dumps(out["verdicts"]))
    return out


# =================================================================== selftest
def _synth_keys(n, d, rng, tightness):
    """n keys on the unit sphere with a controllable pairwise-cosine floor:
    tightness -> 1 reproduces the near-degenerate deployed head geometry."""
    base = rng.normal(size=(1, d))
    base /= np.linalg.norm(base)
    jitter = rng.normal(size=(n, d))
    jitter /= np.linalg.norm(jitter, axis=1, keepdims=True)
    X = tightness * base + (1.0 - tightness) * jitter
    return X / np.linalg.norm(X, axis=1, keepdims=True)


def run_selftest(outdir):
    """Drives every numeric path on synthetic data.  No cluster artifact is read and no
    candidate metric is computed."""
    import mechfix_ops as MECH
    from scipy import stats
    rng = np.random.default_rng(0)
    ok = []

    # --- Probe 1 path: both regimes of the frozen rule must be reachable ---------
    ess_t, mass_t, n_t = nca_concentration(_synth_keys(200, 64, rng, 0.9999))
    ess_l, mass_l, _ = nca_concentration(_synth_keys(200, 64, rng, 0.0))
    assert n_t == 200
    assert mass_t < 0.5, "degenerate-geometry synthetic should be near-uniform"
    assert mass_l > mass_t, "spread geometry must concentrate more than the tight one"
    assert 0.0 < ess_t <= 1.0 and 0.0 < ess_l <= 1.0
    # analytic check: a perfectly uniform softmax over n-1 others
    P = np.full(199, 1.0 / 199)
    assert abs(np.exp(-(P * np.log(P)).sum()) / 200 - 199.0 / 200) < 1e-9
    ok.append("probe1_paths mass_tight={:.4f} mass_loose={:.4f} ess_tight={:.4f}".format(
        mass_t, mass_l, ess_t))

    # --- Probe 2 path: margins, LOO exclusion, KS, shift ------------------------
    n, d = 120, 32
    lab = np.array([0, 1] * (n // 2))
    K = _synth_keys(n, d, rng, 0.98) + 0.02 * lab.reshape(-1, 1)
    K = f64(K / np.linalg.norm(K, axis=1, keepdims=True))
    v_a, p_a, I_a, _ = MECH.deployed_vote(K, lab, f64(K), topk=TOPK, exclude_self=True)
    assert (I_a != np.arange(n)[:, None]).all(), "LOO exclusion failed"
    v_self, _, I_self, _ = MECH.deployed_vote(K, lab, f64(K), topk=TOPK)
    assert (I_self[:, 0] == np.arange(n)).all(), "self should rank 0 without exclusion"
    assert np.abs(v_a - v_self).mean() > 0, "LOO must move the margin"
    ks = stats.ks_2samp(v_a, v_self)
    assert 0.0 <= ks.pvalue <= 1.0
    assert float(np.abs(v_a - v_self).mean()) > 0
    ok.append("probe2_paths mean|shift|={:.3g} ks_p={:.3g} acc={:.3f}".format(
        float(np.abs(v_a - v_self).mean()), float(ks.pvalue), float((p_a == lab).mean())))

    # --- Probe 3 path: synthetic snapshot files through the real aggregator ------
    snapdir = os.path.join(outdir, "selftest_snaps")
    if os.path.isdir(snapdir):
        shutil.rmtree(snapdir)
    os.makedirs(snapdir)
    fold_of = np.tile(np.arange(5), n // 5)
    for ds in DATASETS:
        for seed in SEEDS:
            for f in FOLDS:
                arrays = {}
                for e in SNAP_EPOCHS:
                    drift = (29 - e) / 100.0
                    arrays["K_e{}".format(e)] = f64(
                        K + drift * rng.normal(size=K.shape))
                meta = {"mech_probes_a": {"banked_parity_maxabs": 0.0,
                                          "banked_parity_pass": True, "secs": 1.0}}
                np.savez(os.path.join(snapdir, "snap_{}_s{}_f{}.npz".format(ds, seed, f)),
                         lab=lab, fold_of=fold_of, fit_idx=np.arange(n),
                         meta=json.dumps(meta), **arrays)
    a29, m29 = fold_arena_accuracy(snapdir, "zh", 0, 29, MECH)
    a10, _ = fold_arena_accuracy(snapdir, "zh", 0, 10, MECH)
    assert 0.0 <= a29 <= 1.0 and 0.0 <= m29 <= 1.0 and 0.0 <= a10 <= 1.0
    sub = os.path.join(outdir, "selftest_out")
    res3 = run_probe3(sub, snapdir)
    assert res3["verdict"] in ("ALIVE", "KILLED")
    assert set(res3["acc_delta_vs_floor"]["zh"]) == {str(e) for e in SNAP_EPOCHS}
    assert res3["floor_source"] == "banked"
    ok.append("probe3_paths verdict={} e29_acc={:.4f} e10_acc={:.4f}".format(
        res3["verdict"], a29, a10))

    # --- report path -------------------------------------------------------------
    write_json(os.path.join(sub, "probe1.json"),
               {"verdict": "ALIVE", "probe": "synthetic"})
    write_json(os.path.join(sub, "probe2.json"),
               {"verdict": "KILLED", "probe": "synthetic"})
    rep = run_report(sub)
    assert rep["verdicts"]["RVS"] == "ALIVE" and rep["verdicts"]["XFM"] == "KILLED"
    ok.append("report_path verdicts={}".format(rep["verdicts"]))

    print("\n[mpa selftest] ALL PATHS OK")
    for line in ok:
        print("  - " + line)
    shutil.rmtree(snapdir, ignore_errors=True)
    shutil.rmtree(sub, ignore_errors=True)
    return 0


# ======================================================================= main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["probe12", "mint", "probe3", "report", "selftest"])
    ap.add_argument("--outdir", default=os.path.join(REPO, "artifacts/mech_probes_a"))
    ap.add_argument("--snapdir", default=os.path.join(REPO, "artifacts/mech_probes_a/snaps"))
    ap.add_argument("--scratch", default=os.path.join(REPO, "artifacts/mech_probes_a/scratch"))
    ap.add_argument("--progress", default=None)
    ap.add_argument("--dataset", choices=list(DATASETS))
    ap.add_argument("--seed", type=int)
    ap.add_argument("--fold", type=int)
    ap.add_argument("--threads", type=int, default=8)
    a = ap.parse_args()

    if a.stage == "selftest":
        return run_selftest(a.outdir)

    assert_frozen()
    if a.stage == "probe12":
        r1 = run_probe1(a.outdir, a.progress)
        r2 = run_probe2(a.outdir, a.progress)
        print("[mpa] P1 {} | P2 {}".format(r1["verdict"], r2["verdict"]), flush=True)
    elif a.stage == "mint":
        assert a.dataset and a.seed is not None and a.fold is not None
        out = os.path.join(a.snapdir, "snap_{}_s{}_f{}.npz".format(
            a.dataset, a.seed, a.fold))
        snapshot_mint(a.dataset, a.seed, a.fold, out, a.scratch, a.threads, a.progress)
    elif a.stage == "probe3":
        r = run_probe3(a.outdir, a.snapdir, a.progress)
        print("[mpa] P3 {}".format(r["verdict"]), flush=True)
    elif a.stage == "report":
        run_report(a.outdir, a.progress)
    return 0


if __name__ == "__main__":
    sys.exit(main())
