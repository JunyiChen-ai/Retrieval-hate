"""R6-1C -- build the two independent random-control arms.

Frozen design: idea-stage/R6_CONFIRM_FREEZE_2026-08-17.md.

A0 and CAT caches already exist (built by idea-stage/r6_readout/build_arms.py)
and are REUSED verbatim -- they are not rebuilt here.

New arms (per dataset, per split):
  R6RO-RANDA : [l2norm(L28) || l2norm(L28 @ R_A)]   (7168-d)
  R6RO-RANDB : [l2norm(L28) || l2norm(L28 @ R_B)]   (7168-d)

R_A and R_B are TWO INDEPENDENT Gaussian (3584, 3584) draws, entries
~ N(0, 1/sqrt(3584)), from numpy.random.default_rng(20260817001) and
numpy.random.default_rng(20260817002).  Each is drawn once and reused across
every split, dataset and both streams (img/text), exactly as r6_readout did
for its single R.

`ids` and `labels` are carried through unchanged from the L28 source files.
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
RNG_SEEDS = {"RANDA": 20260817001, "RANDB": 20260817002}
ARM_PREFIX = "R6RO"


def src_path(ds, split, base, layer):
    return os.path.join(EMB, ds, "{}_{}-ro_{}.pt".format(split, base, layer))


def out_path(ds, split, arm):
    return os.path.join(EMB, ds, "{}_{}-{}.pt".format(split, ARM_PREFIX, arm))


def l2norm(x):
    return x / x.norm(dim=1, keepdim=True).clamp_min(1e-12)


def sha_tensor(t):
    return hashlib.sha256(
        np.ascontiguousarray(t.detach().cpu().numpy().astype(np.float32)).tobytes()
    ).hexdigest()


def main():
    Rs, shas = {}, {}
    for arm, sd in RNG_SEEDS.items():
        rng = np.random.default_rng(sd)
        R_np = rng.normal(0.0, 1.0 / np.sqrt(DIM), size=(DIM, DIM)).astype(np.float32)
        shas[arm] = hashlib.sha256(np.ascontiguousarray(R_np).tobytes()).hexdigest()
        Rs[arm] = torch.from_numpy(R_np)
        print("R_%s sha256 = %s  shape=%s  std=%.6f (target %.6f)  rng_seed=%d"
              % (arm[-1], shas[arm], tuple(R_np.shape), float(R_np.std()),
                 1.0 / np.sqrt(DIM), sd))

    # independence sanity: the two draws must not be the same matrix
    assert shas["RANDA"] != shas["RANDB"], "R_A and R_B are identical -- ABORT"
    corr = float(np.corrcoef(Rs["RANDA"].numpy().ravel()[:1000000],
                             Rs["RANDB"].numpy().ravel()[:1000000])[0, 1])
    print("corr(R_A, R_B) over first 1e6 entries = %.6f (expect ~0)" % corr)

    meta = {
        "freeze": "idea-stage/R6_CONFIRM_FREEZE_2026-08-17.md (R6-1C)",
        "reused_arms": ["R6RO-A0", "R6RO-CAT"],
        "reused_arms_note": "built by idea-stage/r6_readout/build_arms.py; not rebuilt",
        "rng_seeds": RNG_SEEDS,
        "R_shape": [DIM, DIM],
        "R_sha256_float32": shas,
        "R_corr_first_1e6": corr,
        "datasets": DATASETS,
        "files": {},
        "id_label_identity_check": {},
    }

    for ds, base in DATASETS.items():
        for split in SPLITS:
            p28, p24 = src_path(ds, split, base, "L28"), src_path(ds, split, base, "L24")
            d28 = torch.load(p28, map_location="cpu")
            d24 = torch.load(p24, map_location="cpu")

            ok_ids = d28["ids"] == d24["ids"]
            ok_lab = bool(torch.equal(d28["labels"], d24["labels"]))
            chk = {"ids_identical_L24_L28": bool(ok_ids),
                   "labels_identical_L24_L28": ok_lab}
            if not (ok_ids and ok_lab):
                raise SystemExit(
                    "ABORT: ids/labels differ between L24 and L28 for %s/%s (%s)"
                    % (ds, split, chk))

            # also assert the reused A0/CAT caches carry the same ids/labels
            for reuse in ("A0", "CAT"):
                rp = out_path(ds, split, reuse)
                if not os.path.exists(rp):
                    raise SystemExit("ABORT: expected existing cache missing: %s" % rp)
                dr = torch.load(rp, map_location="cpu")
                same_ids = dr["ids"] == d28["ids"]
                same_lab = bool(torch.equal(dr["labels"], d28["labels"]))
                chk["ids_identical_%s" % reuse] = bool(same_ids)
                chk["labels_identical_%s" % reuse] = same_lab
                if not (same_ids and same_lab):
                    raise SystemExit(
                        "ABORT: ids/labels of existing %s cache differ from L28 "
                        "for %s/%s" % (reuse, ds, split))
                del dr
            meta["id_label_identity_check"]["%s/%s" % (ds, split)] = chk

            i28, t28 = d28["img_feats"].float(), d28["text_feats"].float()
            assert i28.shape[1] == DIM and t28.shape[1] == DIM, \
                "unexpected dim %s / %s" % (i28.shape, t28.shape)

            n28i, n28t = l2norm(i28), l2norm(t28)
            for arm in ("RANDA", "RANDB"):
                R = Rs[arm]
                im = torch.cat([n28i, l2norm(i28 @ R)], dim=1).contiguous()
                tx = torch.cat([n28t, l2norm(t28 @ R)], dim=1).contiguous()
                obj = {"ids": d28["ids"], "img_feats": im,
                       "text_feats": tx, "labels": d28["labels"]}
                op = out_path(ds, split, arm)
                torch.save(obj, op)
                meta["files"]["%s/%s/%s" % (ds, split, arm)] = {
                    "path": os.path.relpath(op, ROOT),
                    "rows": int(im.shape[0]),
                    "img_dim": int(im.shape[1]),
                    "text_dim": int(tx.shape[1]),
                    "n_labels": int(obj["labels"].shape[0]),
                    "img_sha256": sha_tensor(im),
                    "text_sha256": sha_tensor(tx),
                }
                print("   %-8s %-10s wrote %-26s img=%s text=%s"
                      % (ds, split, os.path.basename(op), tuple(im.shape),
                         tuple(tx.shape)))

    # RANDA and RANDB caches must differ (they share the first 3584 dims only)
    for k in list(meta["files"]):
        pass
    for ds in DATASETS:
        for split in SPLITS:
            a = meta["files"]["%s/%s/RANDA" % (ds, split)]["img_sha256"]
            b = meta["files"]["%s/%s/RANDB" % (ds, split)]["img_sha256"]
            assert a != b, "RANDA/RANDB caches identical for %s/%s" % (ds, split)

    mp = os.path.join(HERE, "build_meta.json")
    with open(mp, "w") as f:
        json.dump(meta, f, indent=1)
    print("wrote", mp)


if __name__ == "__main__":
    main()
