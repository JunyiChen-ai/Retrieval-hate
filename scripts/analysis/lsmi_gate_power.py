#!/usr/bin/env python
"""
AMD-4 / AMD-5 follow-up for the LSMI $0 gate.

AMD-2's G1 ladder (job 13522, gates stage) showed:
  d'=64, n=579/744/549 : joint out-of-fold acc 0.513 / 0.530 / 0.508  -> NO detection
  d'=64, n=8000        : joint OOF acc 0.995                          -> detection
  d'= 8, n=579         : joint OOF acc 0.998, S = 0.7077 (truth 0.6931), share 1.097 -> EXACT
  d'= 8, n=8000        : joint OOF acc 0.995, S = 0.7179, share 1.060
i.e. the LSMI machinery is accurate at OUR sample size; what fails at d'=64 is the *discriminator*
learning a joint function from 128 input dims and ~570 gradient steps.

This script therefore (a) walks the projection dimension d' in {8,16,32,64} at our own n to find
the largest CERTIFIED dimension (AMD-5), (b) walks the optimisation budget at d'=64 to separate
sample size from step count (AMD-4), and (c) re-reads the three real datasets at the certified
dimension with the full control set.

Imports lsmi_gate.py as a module and edits nothing in it. CPU-only, read-only on banked caches,
train+dev only, no test split.
"""
import os, sys, json, argparse, types
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import lsmi_gate as LG                                             # noqa: E402
from sklearn.decomposition import PCA                              # noqa: E402

OUTP = os.path.join(LG.ROOT, "refine-logs", ".lsmi_out_power.json")


def patch_epochs(ep):
    def mk(d1, d2, n_classes=2, bs=32):
        c = types.SimpleNamespace()
        c.device, c.batch_size = LG.DEV, bs
        c.input_size_1, c.input_size_2 = int(d1), int(d2)
        c.embed_size, c.n_classes = 64, n_classes
        c.num_epochs_discriminator = c.num_epochs_entropy_estimator = ep
        return c
    LG.make_cfg = mk


def evr(X, dim):
    return float(PCA(n_components=dim, random_state=0, svd_solver="full")
                 .fit(X.numpy().astype(np.float64)).explained_variance_ratio_.sum())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=["dimladder", "budget", "data"])
    ap.add_argument("--dims", default="8,16,32")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--nperm", type=int, default=50)
    a = ap.parse_args()
    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "16")))
    R = json.load(open(OUTP)) if os.path.exists(OUTP) else {}
    save = lambda: json.dump(R, open(OUTP, "w"), indent=1)
    dims = [int(x) for x in a.dims.split(",")]
    patch_epochs(a.epochs)

    if a.stage == "dimladder":       # AMD-5: XOR at OUR n, across projection dimension
        R.setdefault("controls", {})
        for ds in LG.DATASETS:
            d, _ = LG.load_ds(ds)
            ntr, ndv = d["train"][0].shape[0], d["dev"][0].shape[0]
            for dim in dims:
                (x1, x2, y), (v1, v2, vy) = LG.xor_control(ntr, ndv, dim=dim)
                P, Q = LG.project((x1, x2), (v1, v2), "pca", dim, True)
                k = f"G1dim_xor_{ds}_n{ntr}_d{dim}_ep{a.epochs}"
                R["controls"][k] = LG.run_cell("v3_" + k, P, Q, y, vy); save()
        LG.log("WROTE " + OUTP); return

    if a.stage == "budget":          # AMD-4: XOR at OUR n, d'=64, matched gradient-step budget
        R.setdefault("controls", {})
        for ds in LG.DATASETS:
            d, _ = LG.load_ds(ds)
            ntr, ndv = d["train"][0].shape[0], d["dev"][0].shape[0]
            (x1, x2, y), (v1, v2, vy) = LG.xor_control(ntr, ndv, dim=64)
            P, Q = LG.project((x1, x2), (v1, v2), "pca", 64, True)
            k = f"G1f_xor_{ds}_n{ntr}_d64_ep{a.epochs}"
            R["controls"][k] = LG.run_cell("v3_" + k, P, Q, y, vy); save()
        LG.log("WROTE " + OUTP); return

    # stage == data : real datasets at the certified dimension(s), full control set
    R.setdefault("datasets", {})
    for ds in LG.DATASETS:
        d, shas = LG.load_ds(ds)
        e = R["datasets"].setdefault(ds, {})
        e["sha256"], e["lineage"] = shas, LG.DATASETS[ds]["lineage"]
        tr, dv = (d["train"][0], d["train"][1]), (d["dev"][0], d["dev"][1])
        ytr, ydv = d["train"][2], d["dev"][2]
        for dim in dims:
            tag = f"d{dim}_ep{a.epochs}"
            e[f"evr_{tag}"] = {"img": evr(tr[0], dim), "text": evr(tr[1], dim)}
            LG.log(f"=== {ds} / A7_{tag} (certified-dimension primary) ===")
            P, Q = LG.project(tr, dv, "pca", dim, True)
            e[f"A7_{tag}"] = LG.run_cell(f"v3_{ds}_A7_{tag}", P, Q, ytr, ydv, nperm=a.nperm,
                                         fidelity=True)
            e[f"A7_{tag}"]["arm_label"] = (f"d'={dim} PCA-whitened, {a.epochs} ep "
                                           f"(AMD-5 certified-dimension PRIMARY)")
            save()
            LG.log(f"=== {ds} / C1 duplicate-stream at {tag} ===")
            P, Q = LG.project((tr[0], tr[0]), (dv[0], dv[0]), "pca", dim, True)
            e[f"C1_dup_img_{tag}"] = LG.run_cell(f"v3_{ds}_C1dup_{tag}", P, Q, ytr, ydv); save()
            LG.log(f"=== {ds} / C2 split-half at {tag} ===")
            h = tr[0].shape[1] // 2
            P, Q = LG.project((tr[0][:, :h], tr[0][:, h:]), (dv[0][:, :h], dv[0][:, h:]),
                              "pca", dim, True)
            e[f"C2_splithalf_img_{tag}"] = LG.run_cell(f"v3_{ds}_C2half_{tag}", P, Q, ytr, ydv); save()
    LG.log("WROTE " + OUTP)


if __name__ == "__main__":
    main()
