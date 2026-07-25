#!/usr/bin/env python
"""
AMD-4 follow-up for the LSMI $0 gate: disentangle SAMPLE SIZE from OPTIMISATION BUDGET in the
G1 XOR power gate, and (only if the budget fix restores power) re-read the real datasets under
the matched budget.

Why: the released recipe trains the discriminators for a FIXED 30 EPOCHS, so the number of
gradient steps is proportional to n. G1b (n=8000) and G1 (n~600) therefore differ in BOTH n and
step count (~7500 vs ~570 steps) -- a null at n~600 could be a sample-size wall or merely an
under-trained head. This script adds equal-STEP arms at our n.

Imports lsmi_gate.py as a module; touches nothing else. CPU-only, read-only on caches,
train+dev only, no test split.
"""
import os, sys, json, time, math, argparse
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import lsmi_gate as LG                                             # noqa: E402

ROOT = LG.ROOT
OUTP = os.path.join(ROOT, "refine-logs", ".lsmi_out_power.json")


def patch_epochs(ep):
    """Override the released 30-epoch budget for BOTH sub-estimators."""
    def mk(d1, d2, n_classes=2, ep_=ep, bs=32):
        c = LG.make_cfg.__wrapped__(d1, d2, n_classes, 30, bs) if hasattr(LG.make_cfg, "__wrapped__") else None
        import types as _t
        c = _t.SimpleNamespace()
        c.device, c.batch_size = LG.DEV, bs
        c.input_size_1, c.input_size_2 = int(d1), int(d2)
        c.embed_size, c.n_classes = 64, n_classes
        c.num_epochs_discriminator = c.num_epochs_entropy_estimator = ep_
        return c
    LG.make_cfg = mk


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=400)     # ~= G1b's 7500 steps at n~600
    ap.add_argument("--nperm", type=int, default=50)
    ap.add_argument("--stage", default="power", choices=["power", "data"])
    a = ap.parse_args()
    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "16")))
    R = json.load(open(OUTP)) if os.path.exists(OUTP) else {}
    save = lambda: json.dump(R, open(OUTP, "w"), indent=1)
    patch_epochs(a.epochs)
    tag = f"ep{a.epochs}"

    if a.stage == "power":
        R.setdefault("controls", {})
        LG.log(f"=== AMD-4 G1f: XOR at OUR n with MATCHED optimisation budget ({a.epochs} epochs) ===")
        for ds in LG.DATASETS:
            d, _ = LG.load_ds(ds)
            ntr, ndv = d["train"][0].shape[0], d["dev"][0].shape[0]
            (x1, x2, y), (v1, v2, vy) = LG.xor_control(ntr, ndv, dim=64)
            P, Q = LG.project((x1, x2), (v1, v2), "pca", 64, True)
            k = f"G1f_xor_{ds}_n{ntr}_d64_{tag}"
            R["controls"][k] = LG.run_cell("v2_" + k, P, Q, y, vy); save()
        LG.log("=== AMD-4 G1g: XOR n=8000 d=64 at 30 epochs is the reference; add n=8000 matched-n control ===")
        (x1, x2, y), (v1, v2, vy) = LG.xor_control(600, 150, dim=64)
        P, Q = LG.project((x1, x2), (v1, v2), "pca", 64, True)
        R["controls"][f"G1h_xor_n600_d64_{tag}"] = LG.run_cell(f"v2_G1h_xor_n600_{tag}", P, Q, y, vy); save()
        LG.log("WROTE " + OUTP); return

    # stage == data : re-read the three real datasets on the PRIMARY arm at the matched budget
    R.setdefault("datasets", {})
    for ds in LG.DATASETS:
        d, shas = LG.load_ds(ds)
        e = R["datasets"].setdefault(ds, {})
        tr, dv = (d["train"][0], d["train"][1]), (d["dev"][0], d["dev"][1])
        ytr, ydv = d["train"][2], d["dev"][2]
        LG.log(f"=== {ds} / A6 = A1 at {a.epochs} epochs (matched-budget primary) ===")
        P, Q = LG.project(tr, dv, "pca", 64, True)
        e[f"A6_{tag}"] = LG.run_cell(f"v2_{ds}_A6_{tag}", P, Q, ytr, ydv, nperm=a.nperm)
        e[f"A6_{tag}"]["arm_label"] = f"d'=64 PCA-whitened, {a.epochs} epochs (AMD-4 matched budget)"
        save()
        LG.log(f"=== {ds} / C1 duplicate-stream control at {a.epochs} epochs ===")
        P, Q = LG.project((tr[0], tr[0]), (dv[0], dv[0]), "pca", 64, True)
        e[f"C1_dup_img_{tag}"] = LG.run_cell(f"v2_{ds}_C1dup_{tag}", P, Q, ytr, ydv); save()
    LG.log("WROTE " + OUTP)


if __name__ == "__main__":
    main()
