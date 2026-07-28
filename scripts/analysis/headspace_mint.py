#!/usr/bin/env python
"""headspace_mint.py -- CPU re-mint of DEPLOYED-RECIPE RGCL heads and extraction of the
DEPLOYED HEAD KEY SPACE, for the HEADSPACE-TRANSFER pregate
(record: refine-logs/HEADSPACE_TRANSFER_PREGATE.md).

WHY THIS EXISTS
    Every recent pregate verdict (F89, F94, F95, F96, F97, F98, F105) was rendered on the
    banked RAW fused key space (7168-d) under 5-fold item-disjoint train-LOO.  The deployed
    system retrieves in the TRAINED HEAD's 1024-d space.  This script mints the instrument
    that lets the identical operator battery be re-run in that space.

    The deployed head checkpoints are gone (F78, extended to the whole inventory by
    HEADCOV_PREGATE_RECORD.md §1.1: 228 surviving .pt files all belong to the F92-dead
    bidir heads; 97 empty ckpt/ dirs; 0 of 9 P2-era deployed ckpts exist).  So the head
    space must be RE-MINTED.  ERRPAT_HateMM_2026-07-26.md §0.1/§8 prices this at ~52 s of
    CPU wall per seed and measures the proxy against the banked floor.

WHAT IS MINTED
    fold >= 0 : a head trained on the FITTING POOL of frozen F95/VSW fold `fold`
                (StratifiedKFold(5, shuffle=True, random_state=0) over the train split --
                the SAME assignment mechnov_pairverify.py uses, asserted against the banked
                vsw_ckpt/<ds>/f<fold>.npz ho_idx).  The held-out fifth is NEVER seen by the
                head -- not as training data, not as a dev set, not as an eval set.
    fold == -1: the DEPLOYED-CONFIGURATION head, trained on the FULL train split with the
                REAL dev split as its dev set.  This is the FIDELITY instrument: its dev
                retrieval curve is comparable to the banked GPU floor's Val_Retrieval curve.

TEST CONTACT: NONE, ENFORCED IN CODE.
    `load_feats_from_CLIP` is replaced wholesale so that only `train_<model>.pt` and
    `dev_seen_<model>.pt` are ever opened, and a global `torch.load` guard raises on any
    path containing "test_seen" / "test_".  The harness's own "test" dataloader is a DUMMY
    stratified slice of data the head already sees.  No test-split file is opened, no test
    label is read, no test metric is computed.

RECIPE FIDELITY
    The CLI below is byte-identical to scripts/slurm/enc3seed_lora_curric.sbatch (HateMM)
    / scripts/slurm/enc3seed_zh_b3.sbatch as replayed by errpat_zh_remint.py (MHC-ZH),
    except --device cpu, --num_workers 0, --group_name, --output_path.  In particular
    --eval_retrieval stays True: retrieve_evaluate_RAC_ is the ONLY caller of model.eval()
    in the whole training loop (src/model/evaluate_rac.py:330; there is no matching
    model.train()), so switching it off would leave dropout active for all 30 epochs and
    would NOT be the deployed recipe.  Evaluation itself draws no RNG (dropout is inert in
    eval mode), so the CONTENTS and SIZE of the dev/test dataloaders cannot perturb the
    training trajectory -- verified by construction and reported in the record.

DETERMINISM: clause DET-1/DET-2 (refine-logs/PREGATE_DETERMINISM_CLAUSE.md).  The thread
    environment must be exported BEFORE the process starts (the driver does this); this
    script asserts it and records the full runtime block in every output.

COST: CPU only, <= 8 threads.  Zero GPU, zero SLURM, zero Modal.
"""
import argparse
import hashlib
import json
import os
import socket
import sys
import time

import numpy as np

REPO = "/data/jehc223/RGCL"
sys.path.insert(0, os.path.join(REPO, "src"))
sys.path.insert(0, os.path.join(REPO, "scripts/analysis"))
os.chdir(REPO)

import torch  # noqa: E402
from sklearn.model_selection import StratifiedKFold  # noqa: E402

# --------------------------------------------------------------- DET-1 (hard assert)
DET1_KEYS = ("OMP_NUM_THREADS", "MKL_NUM_THREADS",
             "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS")


def det1_assert(expect="8"):
    miss = [k for k in DET1_KEYS if os.environ.get(k) != expect]
    assert not miss, ("DET-1 violated: {} not exported as {} BEFORE process start "
                      "(see refine-logs/PREGATE_DETERMINISM_CLAUSE.md §2)".format(
                          miss, expect))


def runtime_block():
    import threadpoolctl
    import scipy
    import sklearn
    return {
        "env": {k: os.environ.get(k) for k in DET1_KEYS},
        "threadpools": threadpoolctl.threadpool_info(),
        "versions": {"python": sys.version.split()[0], "numpy": np.__version__,
                     "scipy": scipy.__version__, "sklearn": sklearn.__version__,
                     "torch": torch.__version__},
        "torch_num_threads": int(torch.get_num_threads()),
        "node": socket.gethostname(),
    }


def sha256_of(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for blk in iter(lambda: f.read(1 << 20), b""):
            h.update(blk)
    return h.hexdigest()


# --------------------------------------------------------------------- test-split guard
_ORIG_TORCH_LOAD = torch.load


def _guarded_torch_load(f, *a, **kw):
    s = str(f)
    assert "test_seen" not in s and "/test" not in s, \
        "TEST-SPLIT GUARD: refusing to open {}".format(s)
    return _ORIG_TORCH_LOAD(f, *a, **kw)


torch.load = _guarded_torch_load

# --------------------------------------------------------------- dataset configurations
# Cache dir / encoder model are taken from the FROZEN F95 DATASETS table so the head is
# minted over exactly the feature cache the raw arena used.
import mechnov_pairverify as P  # noqa: E402

FROZEN_PAIRVERIFY_SHA = "77b0defd8eaa3688e58b6d5d17202bd55d16cf1f4a5aaafbe4b2b98598b7240d"

# Deployed CLI, per dataset, verbatim from the banked sbatch / re-mint scripts.
CLI = {
    # scripts/slurm/enc3seed_lora_curric.sbatch (job 13241)
    "hatemm": ["--batch_size", "64", "--lr", "0.0001", "--epochs", "30", "--topk", "20",
               "--dataset", "HateMM", "--model", "Qwen2.5-VL-7B-Instruct-LoRA-curric_HF",
               "--proj_dim", "1024", "--map_dim", "1024",
               "--dropout", "0.2", "0.4", "0.1", "--fusion_mode", "align",
               "--hard_negatives_loss", "True", "--no_hard_negatives", "1",
               "--final_eval", "False", "--metric", "cos", "--loss", "triplet",
               "--batch_norm", "False", "--hybrid_loss", "True", "--warmup", "5",
               "--majority_voting", "arithmetic", "--no_pseudo_gold_positives", "1",
               "--lambda_seg", "0", "--seg_mode", "full", "--num_subclips", "4",
               "--em_rounds", "2", "--consensus_topk", "10", "--consensus_margin", "0.2",
               "--exp_comment", "_Qwen2.5-VL-7B-Instruct-LoRA-curric_HF",
               "--Faiss_GPU", "False"],
    # scripts/slurm/enc3seed_zh_b3.sbatch (job 13150), as replayed by errpat_zh_remint.py
    "zh": ["--batch_size", "64", "--lr", "0.0001", "--epochs", "30", "--topk", "20",
           "--dataset", "MHC_zh", "--model", "Qwen2.5-VL-7B-Instruct-LoRA_HF",
           "--proj_dim", "1024", "--map_dim", "1024",
           "--dropout", "0.2", "0.4", "0.1", "--fusion_mode", "align",
           "--hard_negatives_loss", "True", "--no_hard_negatives", "1",
           "--final_eval", "False", "--metric", "cos", "--loss", "triplet",
           "--batch_norm", "False", "--hybrid_loss", "True", "--warmup", "5",
           "--majority_voting", "arithmetic", "--no_pseudo_gold_positives", "1",
           "--lambda_seg", "0", "--seg_mode", "full", "--num_subclips", "4",
           "--em_rounds", "2", "--consensus_topk", "10", "--consensus_margin", "0.2",
           "--exp_comment", "_Qwen2.5-VL-7B-Instruct-LoRA_HF",
           "--Faiss_GPU", "False"],
}

DUMMY_N_PER_CLASS = 20     # size of the dummy dev/test dataloaders for fold heads


def load_split(cache_dir, split, model):
    """(ids, img, text, labels) as torch tensors -- the shape CLIP2Dataloader wants."""
    d = _ORIG_TORCH_LOAD(os.path.join(cache_dir, "{}_{}.pt".format(split, model)),
                         map_location="cpu", weights_only=False)
    ids = d["ids"]
    if isinstance(ids, list) and len(ids) == 1 and isinstance(ids[0], list):
        ids = ids[0]
    return (list(ids), d["img_feats"].float(), d["text_feats"].float(),
            torch.as_tensor(np.asarray(d["labels"]).astype("int64")))


def subset(sp, idx):
    ids, img, txt, lab = sp
    idx = np.asarray(idx)
    return ([ids[i] for i in idx], img[idx], txt[idx], lab[idx])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=sorted(CLI))
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--fold", type=int, required=True,
                    help="0..4 = fitting-pool head for that frozen F95 fold; "
                         "-1 = deployed-configuration full-train head (fidelity instrument)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--scratch", required=True)
    ap.add_argument("--threads", type=int, default=8)
    a = ap.parse_args()

    det1_assert(str(a.threads))
    assert sha256_of(os.path.join(REPO, "scripts/analysis/mechnov_pairverify.py")) \
        == FROZEN_PAIRVERIFY_SHA, "FROZEN F95 ARMS MODULE CHANGED -- refusing to run"
    torch.set_num_threads(a.threads)

    if os.path.exists(a.out):
        print("[mint] exists, skipping: {}".format(a.out))
        return

    cfg = P.DATASETS[a.dataset]
    cache_dir, model_name = cfg["cache_dir"], cfg["model"]
    tr = load_split(cache_dir, "train", model_name)
    dv = load_split(cache_dir, "dev_seen", model_name)
    lab = tr[3].numpy().astype(int)
    n = len(lab)

    # ---- frozen fold assignment, asserted against the banked RAW arena checkpoint
    skf = StratifiedKFold(n_splits=P.K_FOLDS, shuffle=True, random_state=P.FOLD_SEED)
    splits = list(skf.split(np.zeros((n, 1)), lab))
    fold_of = np.full(n, -1, dtype=int)
    for f, (_, ho) in enumerate(splits):
        fold_of[ho] = f
    ck = os.path.join(REPO, "scripts/analysis/vsw_ckpt", a.dataset)
    fold_parity = []
    for f in range(P.K_FOLDS):
        z = np.load(os.path.join(ck, "f{}.npz".format(f)))
        fold_parity.append(bool(np.array_equal(np.sort(z["ho_idx"]),
                                               np.sort(splits[f][1]))))
    assert all(fold_parity), ("FOLD PARITY FAILED vs banked vsw_ckpt: {}"
                              .format(fold_parity))

    if a.fold >= 0:
        fit_idx = np.asarray(splits[a.fold][0])
        train_sp = subset(tr, fit_idx)
        # dummy dev/test: a stratified slice of the FITTING POOL, i.e. data the head
        # already trains on.  Never the held-out fifth, never dev, never test.
        dummy = np.concatenate([fit_idx[lab[fit_idx] == c][:DUMMY_N_PER_CLASS]
                                for c in (0, 1)])
        dev_sp = subset(tr, dummy)
        tst_sp = subset(tr, dummy)
    else:
        fit_idx = np.arange(n)
        train_sp = tr
        dev_sp = dv
        dummy = np.concatenate([np.flatnonzero(lab == c)[:DUMMY_N_PER_CLASS]
                                for c in (0, 1)])
        tst_sp = subset(tr, dummy)

    # ------------------------------------------------------------------ monkeypatches
    import run_rac  # noqa: E402

    def _patched_loader(path, dataset, model, all=False):
        return train_sp, dev_sp, tst_sp

    run_rac.load_feats_from_CLIP = _patched_loader

    _ORIG_MODEL_PASS = run_rac.model_pass
    _HOLD = {}

    def _model_pass_spy(train_dl, evaluate_dl, test_seen_dl, model, **kw):
        _HOLD["model"] = model
        return _ORIG_MODEL_PASS(train_dl, evaluate_dl, test_seen_dl, model, **kw)

    run_rac.model_pass = _model_pass_spy

    _ORIG_RETRIEVE = run_rac.retrieve_evaluate_RAC_
    _ORIG_METRICS = run_rac.compute_metrics_retrieval
    _ST = {"eval_name": None, "epoch": None}
    _CURVE = []

    def _retrieve_spy(*ar, **kw):
        _ST["eval_name"] = kw.get("eval_name")
        _ST["epoch"] = kw.get("epoch")
        return _ORIG_RETRIEVE(*ar, **kw)

    def _metrics_spy(logging_dict, labels, **kw):
        out = _ORIG_METRICS(logging_dict, labels, **kw)
        acc_, roc_, pre_, rec_, f1_, _, _, macro = out
        _CURVE.append({"split": _ST["eval_name"], "epoch": int(_ST["epoch"]),
                       "acc": round(float(acc_), 4),
                       "macroF1": round(float(macro["macro_f1"]), 4),
                       "roc": round(float(roc_), 4)})
        return out

    run_rac.retrieve_evaluate_RAC_ = _retrieve_spy
    run_rac.compute_metrics_retrieval = _metrics_spy

    # every-epoch state_dict dumps are ~34 MB each and nothing downstream reads them
    # (best_epoch_path is only re-loaded on the EM branch, which this recipe never takes)
    nsave = {"n": 0}

    def _no_save(obj, path, *ar, **kw):
        nsave["n"] += 1

    torch.save = _no_save

    # --------------------------------------------------------------------------- run
    tag = "{}_s{}_f{}".format(a.dataset, a.seed, a.fold)
    outp = os.path.join(a.scratch, "mint", tag)
    os.makedirs(outp, exist_ok=True)
    sys.argv = (["run_rac.py"] + CLI[a.dataset]
                + ["--seed", str(a.seed), "--device", "cpu", "--num_workers", "0",
                   "--group_name", "HEADSPACE_" + tag,
                   "--output_path", outp, "--force", "True"])
    args = run_rac.parse_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    t0 = time.time()
    run_rac.main(args)
    secs = time.time() - t0

    model = _HOLD["model"]
    model.eval()

    def keys_of(sp):
        with torch.no_grad():
            _, emb = model(sp[1], sp[2], return_embed=True)
        return emb.detach().cpu().numpy().astype("float64")

    K_train = keys_of(tr)
    K_dev = keys_of(dv) if a.fold < 0 else np.zeros((0, K_train.shape[1]))

    meta = {"script_sha256": sha256_of(os.path.abspath(__file__)),
            "frozen_pairverify_sha256": FROZEN_PAIRVERIFY_SHA,
            "dataset": a.dataset, "ds": cfg["ds"], "encoder_model": model_name,
            "seed": a.seed, "fold": a.fold, "n_train": int(n), "n_dev": int(len(dv[0])),
            "n_fit": int(len(fit_idx)), "head_dim": int(K_train.shape[1]),
            "argv": sys.argv, "secs": round(secs, 1),
            "fold_parity_vs_banked_vsw_ckpt": fold_parity,
            "n_state_dict_saves_suppressed": nsave["n"],
            "eval_curve": _CURVE,
            "test_contact": "NONE -- torch.load guard + patched loader; only "
                            "train_*.pt and dev_seen_*.pt opened",
            "runtime": runtime_block()}
    tmp = a.out + ".tmp.npz"
    np.savez(tmp, K_train=K_train, K_dev=K_dev, lab=lab,
             lab_dev=dv[3].numpy().astype(int), fold_of=fold_of,
             fit_idx=fit_idx, meta=json.dumps(meta))
    os.replace(tmp, a.out)
    print("[mint] {} done in {:.1f}s -> {}".format(tag, secs, a.out))
    dev_rows = [r for r in _CURVE if r["split"] == "dev"]
    if dev_rows:
        print("[mint] {} dev acc final-epoch {:.4f}".format(
            tag, dev_rows[-1]["acc"]))


if __name__ == "__main__":
    main()
