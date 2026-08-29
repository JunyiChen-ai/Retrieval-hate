#!/usr/bin/env python3
import argparse, hashlib, json, sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path[:0] = [str(HERE.parent), str(ROOT / "scripts/duplex")]
from relation_v4.io import sha256
from relation_v8.run import load_split_exact, atomic_json
from relation_v11.score_stream_benchmark import metrics
from relation_v12.diagnostic import frozen_v10_identity
from relation_v17.select_eval import read_frozen, shuffled, paired_ci

GRID = [0, .01, .025, .05, .1, .2]
KEYS = ("frame_ap", "frame_roc", "within_macro_ap", "within_macro_roc")


def ecdf_fit(x):
    x = np.asarray(x, dtype=np.float64)
    if not len(x) or not np.isfinite(x).all():
        raise RuntimeError("invalid ECDF reference")
    return np.sort(x)


def ecdf_apply(ref, x):
    # Mid-distribution convention is frozen and deterministic.
    x = np.asarray(x, dtype=np.float64)
    lo = np.searchsorted(ref, x, side="left")
    hi = np.searchsorted(ref, x, side="right")
    return (lo + hi) / (2.0 * len(ref))


def raw_components(rows, ids, lengths, base, state=None, uncovered_zero=False):
    by = {v: [] for v in ids}
    for r in rows:
        if r["video_id"] in by and r.get("temporal_span_valid", True) and r["start"] is not None and r["end"] is not None:
            by[r["video_id"]].append(r)
    graw, lraw, available = {}, {}, {}
    for v in ids:
        q = by[v]
        available[v] = bool(q)
        if not q:
            graw[v] = np.nan
            lraw[v] = np.zeros(lengths[v], dtype=np.float64)
            continue
        graw[v] = float(np.mean([x["scores"]["causal_continuous"] for x in q]))
        centers = np.asarray([(x["start"] + x["end"]) / 2 for x in q])
        vals = np.asarray([x["scores"]["masked_branch_reset"] for x in q], dtype=np.float64)
        frame_centers=np.arange(lengths[v])+.5
        nearest = np.abs(frame_centers[:, None] - centers[None]).argmin(1)
        z = vals[nearest]
        if uncovered_zero:
            spans=np.asarray([(x['start'],x['end']) for x in q],dtype=np.float64)
            covered=((frame_centers[:,None]>=spans[None,:,0])&(frame_centers[:,None]<spans[None,:,1])).any(1)
            zz=np.zeros(lengths[v],dtype=np.float64)
            if covered.any():zz[covered]=z[covered]-z[covered].mean()
            lraw[v]=zz
        else:lraw[v] = z - z.mean()
    if state is None:
        valid = [v for v in ids if available[v]]
        state = {
            "global_causal_mean_ecdf": ecdf_fit([graw[v] for v in valid]).tolist(),
            "base_video_mean_ecdf": ecdf_fit([np.mean(base[v]) for v in valid]).tolist(),
            "local_rms": float(np.sqrt(np.mean(np.concatenate([lraw[v] for v in valid]) ** 2)) + 1e-12),
            "missing_policy": "zero global and local correction",
            "global_formula": "ECDF_val(causal video mean)-ECDF_val(base video mean)",
            "local_formula": "nearest-center masked-reset score minus per-video frame mean, divided by val RMS",
        }
    gr = np.asarray(state["global_causal_mean_ecdf"])
    br = np.asarray(state["base_video_mean_ecdf"])
    global_c, local_c = {}, {}
    for v in ids:
        if not available[v]:
            global_c[v] = np.zeros(lengths[v]); local_c[v] = np.zeros(lengths[v]); continue
        delta = float(ecdf_apply(gr, [graw[v]])[0] - ecdf_apply(br, [np.mean(base[v])])[0])
        global_c[v] = np.full(lengths[v], delta)
        local_c[v] = lraw[v] / state["local_rms"]
        if abs(local_c[v].mean()) > 1e-10:
            raise RuntimeError("local correction is not zero mean")
    return global_c, local_c, state


def fuse(base, glob, loc, alpha, beta):
    return {v: base[v] + alpha * glob[v] + beta * loc[v] for v in base}


def point_pareto(m, identity, eps=1e-12):
    return all(m[k] >= identity[k] - eps for k in KEYS)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", required=True); p.add_argument("--val-dir", required=True)
    p.add_argument("--test-raw-dir", required=True); p.add_argument("--out-dir", required=True)
    a = p.parse_args(); out = Path(a.out_dir); out.mkdir(parents=True, exist_ok=False)
    manifest = json.load(open(a.manifest))
    vr, vg, _ = load_split_exact(manifest, "val")
    bv, _ = frozen_v10_identity(manifest, vr); bv = {v: x[:, 0] for v, x in bv.items()}
    vrows, _ = read_frozen(a.val_dir); ids = sorted(vg)
    gc, lc, state = raw_components(vrows, ids, {v: len(vg[v]) for v in ids}, bv)
    identity = metrics(bv, vg); surface = []
    for alpha in GRID:
        for beta in GRID:
            pred = fuse(bv, gc, lc, alpha, beta); mm = metrics(pred, vg)
            surface.append({"alpha": alpha, "beta": beta, "metrics": mm,
                            "point_pareto_noninferior": point_pareto(mm, identity)})
    # Expensive inference is confined to point-Pareto candidates; point noninferiority
    # itself is an explicit necessary gate, so this cannot discard an eligible cell.
    for cell in surface:
        if not cell["point_pareto_noninferior"]:
            cell.update({"bootstrap_run": False, "eligible": False}); continue
        pred = fuse(bv, gc, lc, cell["alpha"], cell["beta"])
        ci = paired_ci(bv, pred, vg, 2000, seed=1818)
        ci_gate = all(ci[k]["lower95"] >= -1e-12 for k in KEYS)
        # A local correction must beat deterministic temporal shuffles on both within metrics.
        if cell["beta"] > 0:
            sh = [metrics(fuse(bv, gc, shuffled(lc, j), cell["alpha"], cell["beta"]), vg) for j in range(200)]
            q = {k: float(np.quantile([x[k] for x in sh], .95)) for k in ("within_macro_ap", "within_macro_roc")}
            shuffle_gate = all(cell["metrics"][k] > q[k] for k in q)
        else:
            q = None; shuffle_gate = True
        cell.update({"bootstrap_run": True, "paired_video_ci_B2000": ci,
                     "ci_pareto_gate": bool(ci_gate), "local_shuffle_B200_q95": q,
                     "local_shuffle_gate": bool(shuffle_gate),
                     "eligible": bool((cell["alpha"] == 0 and cell["beta"] == 0) or (ci_gate and shuffle_gate))})
    eligible = [x for x in surface if x["eligible"]]
    selected = max(eligible, key=lambda x: tuple(x["metrics"][k] for k in KEYS) + (-x["alpha"], -x["beta"]))
    frozen = {"method": "relation_v18_low_cost_dual_pareto_fusion", "test_informed_design_from_v16": True,
              "corpus": manifest["corpus"], "formula_state": state, "grid": GRID,
              "selection_rule": "point Pareto noninferior on pooled AP/ROC and within macro AP/ROC; paired video-bootstrap B=2000 lower95 nonnegative for all four; beta>0 additionally exceeds B=200 deterministic time-shuffle q95 on both within metrics; lexicographic AP,ROC,within AP,within ROC; (0,0) exact fallback",
              "validation_identity": identity, "validation_surface": surface,
              "selected": {k: selected[k] for k in ("alpha", "beta", "metrics")},
              "val_raw_manifest_sha256": sha256(Path(a.val_dir) / "raw_manifest.json"),
              "manifest": str(Path(a.manifest).resolve()), "manifest_sha256": sha256(a.manifest),
              "test_opened": False}
    atomic_json(out / "frozen_config.json", frozen)
    # Test is opened only after the complete validation decision is frozen.
    tr, tg, _ = load_split_exact(manifest, "test")
    bt, _ = frozen_v10_identity(manifest, tr); bt = {v: x[:, 0] for v, x in bt.items()}
    trows, _ = read_frozen(a.test_raw_dir); tids = sorted(tg)
    tgc, tlc, _ = raw_components(trows, tids, {v: len(tg[v]) for v in tids}, bt, state)
    aa, bb = selected["alpha"], selected["beta"]
    formal = metrics(fuse(bt, tgc, tlc, aa, bb), tg)
    identity_t = metrics(bt, tg)
    # Explicitly test-informed diagnostic surface; never used by the formal selector.
    oracle = []
    for alpha in GRID:
        for beta in GRID:
            mm = metrics(fuse(bt, tgc, tlc, alpha, beta), tg)
            oracle.append({"alpha": alpha, "beta": beta, "metrics": mm,
                           "four_metric_double_positive": all(mm[k] > identity_t[k] for k in KEYS)})
    best_oracle = max(oracle, key=lambda x: tuple(x["metrics"][k] for k in KEYS))
    payload = {"method": frozen["method"], "corpus": manifest["corpus"],
               "formal_selected": {"alpha": aa, "beta": bb, "metrics": formal},
               "identity": identity_t, "test_labels_used_for_formal_selection": False,
               "TEST_INFORMED_ORACLE_DIAGNOSTIC_NOT_CLAIMABLE": {"surface": oracle, "best_lexicographic": best_oracle,
                   "n_four_metric_double_positive": sum(x["four_metric_double_positive"] for x in oracle)},
               "test_raw_manifest_sha256": sha256(Path(a.test_raw_dir) / "raw_manifest.json"),
               "frozen_config_sha256": sha256(out / "frozen_config.json")}
    atomic_json(out / "test_eval_and_oracle.json", payload); print(json.dumps(payload, indent=2))


if __name__ == "__main__": main()
