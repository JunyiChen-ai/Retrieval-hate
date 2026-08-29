"""RE-AUDIT R1b -- build the three RAC-level OCR arms for HateMM.

Frozen design: idea-stage/REAUDIT_FREEZE.md, candidate R1b.

Arms (HateMM only -- it is the only dataset with train+dev+test OCR coverage;
data/OCR/MHC_test and data/OCR/MHC_zh_test hold test splits only):

  RAOC-A0    : verbatim copy of the deployed A0 cache
               (Qwen2.5-VL-7B-Instruct-LoRA-curric_HF)
  RAOC-OCR   : text stream <- [l2norm(text_3584) || l2norm(ocr_768)]   (4352-d)
  RAOC-RAND  : text stream <- [l2norm(text_3584) || l2norm(text_3584 @ R)] (4352-d)

R is ONE fixed Gaussian (3584, 768), entries ~ N(0, 1/sqrt(3584)), drawn once with
numpy.random.default_rng(20260817) and reused across every split.  RAOC-RAND is the
dimension-matched, content-free control: it carries exactly the added capacity of
RAOC-OCR and none of the on-screen text, so OCR-RAND isolates content from capacity.
This is the RANDCAT construction of idea-stage/R6_CONFIRM_FREEZE_2026-08-17.md.

The image stream, ids and labels are carried through unchanged in all three arms.
No labels are read for any decision here; no test metric is computed.
"""
import hashlib
import json
import os

import numpy as np
import torch

ROOT = "/home/jehc223/Retrieval-hate"
EMB = os.path.join(ROOT, "data", "CLIP_Embedding", "HateMM")
OCR = os.path.join(ROOT, "data", "OCR", "HateMM")
HERE = os.path.dirname(os.path.abspath(__file__))

BASE = "Qwen2.5-VL-7B-Instruct-LoRA-curric_HF"
SPLITS = ["train", "dev_seen", "test_seen"]
RNG_SEED = 20260817
D_OCR = 768
ARM_PREFIX = "RAOC"


def l2norm(x):
    return x / x.norm(dim=1, keepdim=True).clamp_min(1e-12)


def sha_tensor(t):
    return hashlib.sha256(
        np.ascontiguousarray(t.detach().cpu().numpy().astype(np.float32)).tobytes()
    ).hexdigest()


def main():
    rng = np.random.default_rng(RNG_SEED)
    R = torch.from_numpy(
        rng.normal(0.0, 1.0 / np.sqrt(3584), size=(3584, D_OCR)).astype(np.float32))
    r_sha = hashlib.sha256(
        np.ascontiguousarray(R.numpy()).tobytes()).hexdigest()
    print("R sha256 = %s shape=%s" % (r_sha, tuple(R.shape)))

    meta = {"freeze": "idea-stage/REAUDIT_FREEZE.md (candidate R1b)",
            "rng_seed": RNG_SEED, "R_shape": [3584, D_OCR],
            "R_sha256_float32": r_sha, "base": BASE, "files": {}, "checks": {}}

    for split in SPLITS:
        src = torch.load(os.path.join(EMB, "%s_%s.pt" % (split, BASE)),
                         map_location="cpu", weights_only=False)
        ocr = torch.load(os.path.join(OCR, "rac_ocrmean30_%s.pt" % split),
                         map_location="cpu", weights_only=False)

        raw_ids = src["ids"]
        # the RAC caches store ids as a list holding one list of ids
        while len(raw_ids) == 1 and isinstance(raw_ids[0], list):
            raw_ids = raw_ids[0]
        ids = list(raw_ids)
        oid = {v: i for i, v in enumerate(ocr["ids"])}
        missing = [v for v in ids if v not in oid]
        if missing:
            raise SystemExit("HALT: %d ids missing from OCR cache (%s)"
                             % (len(missing), split))
        order = torch.tensor([oid[v] for v in ids], dtype=torch.long)
        o = ocr["text_feats"].float()[order]
        n_zero = int((o.norm(dim=1) == 0).sum())

        img = src["img_feats"].float()
        txt = src["text_feats"].float()
        t_n = l2norm(txt)
        o_n = l2norm(o)
        r_n = l2norm(txt @ R)

        arms = {
            "A0": (img, txt),
            "OCR": (img, torch.cat([t_n, o_n], dim=1)),
            "RAND": (img, torch.cat([t_n, r_n], dim=1)),
        }
        for arm, (i_f, t_f) in arms.items():
            out = {"ids": src["ids"], "img_feats": i_f.contiguous(),
                   "text_feats": t_f.contiguous(),
                   "labels": src["labels"]}
            p = os.path.join(EMB, "%s_%s-%s.pt" % (split, ARM_PREFIX, arm))
            torch.save(out, p)
            meta["files"]["%s/%s" % (split, arm)] = {
                "path": os.path.relpath(p, ROOT), "rows": len(ids),
                "img_dim": int(i_f.shape[1]), "text_dim": int(t_f.shape[1]),
                "img_sha256": sha_tensor(i_f), "text_sha256": sha_tensor(t_f)}
            print("wrote %s  rows=%d img=%d txt=%d"
                  % (os.path.basename(p), len(ids), i_f.shape[1], t_f.shape[1]))
        meta["checks"][split] = {
            "n_rows": len(ids), "n_all_zero_ocr_rows": n_zero,
            "ocr_ids_in_same_order_as_rac": bool(list(ocr["ids"]) == ids),
            "ocr_block_dim": int(o.shape[1]),
            "rand_block_dim": int(r_n.shape[1]),
            "labels_carried_unchanged": True}

    json.dump(meta, open(os.path.join(HERE, "build_ocr_meta.json"), "w"), indent=1)
    print("meta written")


if __name__ == "__main__":
    main()
