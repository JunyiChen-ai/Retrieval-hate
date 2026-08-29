"""R6-1 -- build the four arm caches for the multi-layer readout pilot.

Frozen design: idea-stage/R6_PILOT_FREEZE_2026-08-17.md, pilot R6-1.

Arms (per dataset, per split), written next to the source caches so that
src/data_loader/dataset.py's `{path}/{dataset}/{split}_{model}.pt` resolution
picks them up via `--model R6RO-<ARM>`:

  R6RO-A0      : verbatim copy of the ro_L28 cache
  R6RO-L24     : verbatim copy of the ro_L24 cache
  R6RO-CAT     : [l2norm(L28) || l2norm(L24)]              (7168-d)
  R6RO-RANDCAT : [l2norm(L28) || l2norm(L28 @ R)]          (7168-d)

R is ONE fixed Gaussian matrix (3584, 3584), entries ~ N(0, 1/sqrt(3584)),
drawn once with numpy.random.default_rng(20260817) and reused across every
split, dataset and both streams.

`ids` and `labels` are carried through unchanged from the source files.
No model, no GPU, no labels are read for any decision here.
"""
import hashlib
import json
import os

import numpy as np
import torch

ROOT = "/home/jehc223/Retrieval-hate"
EMB = os.path.join(ROOT, "data", "CLIP_Embedding")
HERE = os.path.dirname(os.path.abspath(__file__))

DATASETS = {
    "HateMM": "Qwen2.5-VL-7B-Instruct-LoRA-curric_HF",
    "MHC_zh": "Qwen2.5-VL-7B-Instruct-LoRA_HF",
}
SPLITS = ["train", "dev_seen", "test_seen"]
DIM = 3584
RNG_SEED = 20260817
ARM_PREFIX = "R6RO"


def src_path(ds, split, base, layer):
    return os.path.join(EMB, ds, "{}_{}-ro_{}.pt".format(split, base, layer))


def out_path(ds, split, arm):
    return os.path.join(EMB, ds, "{}_{}-{}.pt".format(split, ARM_PREFIX, arm))


def l2norm(x):
    return x / x.norm(dim=1, keepdim=True).clamp_min(1e-12)


def rownorm_dev(x):
    """max |‖row‖ - 1| over rows."""
    return float((x.norm(dim=1) - 1.0).abs().max())


def sha_tensor(t):
    return hashlib.sha256(
        np.ascontiguousarray(t.detach().cpu().numpy().astype(np.float32)).tobytes()
    ).hexdigest()


def main():
    rng = np.random.default_rng(RNG_SEED)
    R_np = rng.normal(0.0, 1.0 / np.sqrt(DIM), size=(DIM, DIM)).astype(np.float32)
    R = torch.from_numpy(R_np)
    r_sha = hashlib.sha256(np.ascontiguousarray(R_np).tobytes()).hexdigest()
    print("R sha256 = %s  shape=%s  std=%.6f (target %.6f)"
          % (r_sha, tuple(R_np.shape), float(R_np.std()), 1.0 / np.sqrt(DIM)))

    meta = {
        "freeze": "idea-stage/R6_PILOT_FREEZE_2026-08-17.md (pilot R6-1)",
        "rng_seed": RNG_SEED,
        "R_shape": list(R_np.shape),
        "R_sha256_float32": r_sha,
        "R_std": float(R_np.std()),
        "datasets": DATASETS,
        "files": {},
        "row_norm_check": {},
        "id_label_identity_check": {},
    }

    for ds, base in DATASETS.items():
        for split in SPLITS:
            p28, p24 = src_path(ds, split, base, "L28"), src_path(ds, split, base, "L24")
            d28 = torch.load(p28, map_location="cpu")
            d24 = torch.load(p24, map_location="cpu")

            # ---- hard assertion: ids and labels must be identical across layers
            ok_ids = d28["ids"] == d24["ids"]
            ok_lab = bool(torch.equal(d28["labels"], d24["labels"]))
            meta["id_label_identity_check"]["%s/%s" % (ds, split)] = {
                "ids_identical": bool(ok_ids), "labels_identical": ok_lab}
            if not (ok_ids and ok_lab):
                raise SystemExit(
                    "ABORT: ids/labels differ between L24 and L28 for %s/%s "
                    "(ids_identical=%s labels_identical=%s)" % (ds, split, ok_ids, ok_lab))

            i28, t28 = d28["img_feats"].float(), d28["text_feats"].float()
            i24, t24 = d24["img_feats"].float(), d24["text_feats"].float()
            assert i28.shape == i24.shape == t28.shape == t24.shape, "shape mismatch"
            assert i28.shape[1] == DIM, "unexpected dim %s" % (i28.shape,)

            devs = {"L28_img": rownorm_dev(i28), "L28_text": rownorm_dev(t28),
                    "L24_img": rownorm_dev(i24), "L24_text": rownorm_dev(t24)}
            # all-zero rows are pre-existing extraction failures in the source
            # caches; they are identical across L24/L28 and across all arms.
            zr = {k: int((v.norm(dim=1) == 0).sum()) for k, v in
                  [("L28_img", i28), ("L28_text", t28),
                   ("L24_img", i24), ("L24_text", t24)]}
            devs_nz = {}
            for k, v in [("L28_img", i28), ("L28_text", t28),
                         ("L24_img", i24), ("L24_text", t24)]:
                nz = v[v.norm(dim=1) > 0]
                devs_nz[k] = rownorm_dev(nz) if nz.shape[0] else 0.0
            meta["row_norm_check"]["%s/%s" % (ds, split)] = {
                "max_abs_dev_from_1": devs,
                "max_abs_dev_from_1_excluding_zero_rows": devs_nz,
                "n_all_zero_rows": zr,
            }
            print("%-8s %-10s n=%4d  row-norm max|.-1|: %s"
                  % (ds, split, i28.shape[0],
                     " ".join("%s=%.2e" % (k, v) for k, v in devs.items())))

            n28i, n28t = l2norm(i28), l2norm(t28)
            n24i, n24t = l2norm(i24), l2norm(t24)
            r28i, r28t = l2norm(i28 @ R), l2norm(t28 @ R)

            arms = {
                "A0": (d28["img_feats"], d28["text_feats"]),
                "L24": (d24["img_feats"], d24["text_feats"]),
                "CAT": (torch.cat([n28i, n24i], dim=1), torch.cat([n28t, n24t], dim=1)),
                "RANDCAT": (torch.cat([n28i, r28i], dim=1),
                            torch.cat([n28t, r28t], dim=1)),
            }
            for arm, (im, tx) in arms.items():
                obj = {"ids": d28["ids"], "img_feats": im.contiguous(),
                       "text_feats": tx.contiguous(), "labels": d28["labels"]}
                op = out_path(ds, split, arm)
                torch.save(obj, op)
                meta["files"]["%s/%s/%s" % (ds, split, arm)] = {
                    "path": os.path.relpath(op, ROOT),
                    "rows": int(im.shape[0]),
                    "img_dim": int(im.shape[1]),
                    "text_dim": int(tx.shape[1]),
                    "n_ids": int(sum(len(s) for s in obj["ids"])),
                    "n_labels": int(obj["labels"].shape[0]),
                    "img_sha256": sha_tensor(im),
                    "text_sha256": sha_tensor(tx),
                }
                print("   wrote %-44s img=%s text=%s"
                      % (os.path.basename(op), tuple(im.shape), tuple(tx.shape)))

    mp = os.path.join(HERE, "build_meta.json")
    with open(mp, "w") as f:
        json.dump(meta, f, indent=1)
    print("wrote", mp)

    allrn = [v for d in meta["row_norm_check"].values()
             for v in d["max_abs_dev_from_1_excluding_zero_rows"].values()]
    nz = sum(v for d in meta["row_norm_check"].values()
             for v in d["n_all_zero_rows"].values())
    meta["row_norm_summary"] = {
        "max_abs_dev_from_1_over_nonzero_rows": max(allrn),
        "renormalisation_is_noop": bool(max(allrn) < 1e-4),
        "total_all_zero_row_slots": int(nz),
    }
    with open(mp, "w") as f:
        json.dump(meta, f, indent=1)
    print("GLOBAL max row-norm deviation from 1.0 (non-zero rows) = %.3e "
          "-> re-normalisation was %s ; all-zero row slots = %d"
          % (max(allrn), "a no-op" if max(allrn) < 1e-4 else "NOT a no-op", nz))


if __name__ == "__main__":
    main()
