#!/usr/bin/env python
"""c02_a0_mint.py -- CPU re-mint of the DEPLOYED-RECIPE RGCL head for the C02 A0,
plus extraction of the DEPLOYED HEAD KEYS of every C02 density view.

Record: refine-logs/C02_A0_V9_RECORD.md.

WHY A MINT AT ALL
    F113 (refine-logs/HEADSPACE_TRANSFER_PREGATE.md) and the registry's
    unified_pilot_gate both say a raw-key arena may KILL but may not PROMOTE. A
    Stage-0 PASS therefore has to be rendered in the trained head's key space. The
    deployed head checkpoints no longer exist (F78 / HEADCOV §1.1), so the head is
    re-minted on CPU exactly as the head-space instrument does.

REUSE, NOT REWRITE
    scripts/analysis/headspace_mint.py is imported and its sha256 asserted. Its
    dataset table, deployed CLI, fold assignment, fold-parity assertion against the
    banked scripts/analysis/vsw_ckpt/<ds>/f<fold>.npz, dummy-dataloader construction,
    monkeypatches, seeding and DET-1 contract are used unchanged. The ONLY addition
    is: after training, forward the SAME head over (banked native img_feats, view
    text_feats) for every C02 view and store those keys.

    img_feats are byte-identical across views by construction -- the deployed image
    stream never sees the transcript -- so the view axis is the only thing that moves.

TEST CONTACT: NONE.
    headspace_mint installs a torch.load guard that raises on any path containing
    "test_seen" or "/test"; this file adds the same guard on its own view loads and
    refuses any view path that is not train_*.
"""
import argparse
import json
import os
import sys
import time

import numpy as np

REPO = "/data/jehc223/RGCL"
sys.path.insert(0, os.path.join(REPO, "src"))
sys.path.insert(0, os.path.join(REPO, "scripts/analysis"))
os.chdir(REPO)

import torch  # noqa: E402
from sklearn.model_selection import StratifiedKFold  # noqa: E402

if not __debug__:
    raise SystemExit("REFUSING TO RUN: python -O strips the assert-based guards")

import headspace_mint as HM  # noqa: E402  (installs the torch.load test guard)
import mechnov_pairverify as P  # noqa: E402

sys.path.insert(0, os.path.join(REPO, "src/utils"))
import c02_density_views as V  # noqa: E402

FROZEN_HEADSPACE_MINT_SHA = (
    "cefdf8dc2f4a9aefa042ef7bec9b1d06c9721ae5b4a70ec117e9929ff0916612")


def load_view_text(cache_dir, split, base_tag, view, n_expect, ids_expect, lab_expect):
    """Load one C02 view file and assert exact ID/label/order parity with the bank."""
    path = os.path.join(cache_dir, "{}_{}-c02den-{}.pt".format(split, base_tag, view))
    low = path.lower()
    assert "test_seen" not in low and "/test" not in low, "TEST GUARD: {}".format(path)
    assert split == "train", "C02 A0 opens the train split only, got '{}'".format(split)
    d = torch.load(path, map_location="cpu", weights_only=False)
    assert d.get("c02_view") == view, "view tag mismatch in {}".format(path)
    assert "img_feats" not in d, "view file must not carry img_feats"
    ids = d["ids"]
    if isinstance(ids, list) and len(ids) == 1 and isinstance(ids[0], list):
        ids = ids[0]
    ids = list(ids)
    assert len(ids) == n_expect, "row count mismatch in {}".format(path)
    assert ids == list(ids_expect), "ID ORDER MISMATCH vs native bank in {}".format(path)
    lab = np.asarray(d["labels"]).astype("int64")
    assert np.array_equal(lab, np.asarray(lab_expect).astype("int64")), \
        "LABEL MISMATCH vs native bank in {}".format(path)
    return d["text_feats"].float()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=sorted(HM.CLI))
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--fold", type=int, required=True,
                    help="0..4 fitting-pool head; -1 deployed-configuration head")
    ap.add_argument("--out", required=True)
    ap.add_argument("--scratch", required=True)
    ap.add_argument("--threads", type=int, default=8)
    a = ap.parse_args()

    HM.det1_assert(str(a.threads))
    assert HM.sha256_of(os.path.join(REPO, "scripts/analysis/headspace_mint.py")) \
        == FROZEN_HEADSPACE_MINT_SHA, "FROZEN HEAD-SPACE MINT CHANGED -- refusing to run"
    assert HM.sha256_of(os.path.join(REPO, "scripts/analysis/mechnov_pairverify.py")) \
        == HM.FROZEN_PAIRVERIFY_SHA, "FROZEN F95 ARMS MODULE CHANGED -- refusing to run"
    torch.set_num_threads(a.threads)

    if os.path.exists(a.out):
        print("[c02mint] exists, skipping: {}".format(a.out))
        return

    cfg = P.DATASETS[a.dataset]
    cache_dir, model_name = cfg["cache_dir"], cfg["model"]
    tr = HM.load_split(cache_dir, "train", model_name)
    dv = HM.load_split(cache_dir, "dev_seen", model_name)
    lab = tr[3].numpy().astype(int)
    n = len(lab)

    view_text = {}
    for view in V.VIEW_NAMES:
        view_text[view] = load_view_text(cache_dir, "train", model_name, view,
                                         n, tr[0], lab)

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
    assert all(fold_parity), "FOLD PARITY FAILED vs banked vsw_ckpt: {}".format(fold_parity)

    if a.fold >= 0:
        fit_idx = np.asarray(splits[a.fold][0])
        train_sp = HM.subset(tr, fit_idx)
        dummy = np.concatenate([fit_idx[lab[fit_idx] == c][:HM.DUMMY_N_PER_CLASS]
                                for c in (0, 1)])
        dev_sp = HM.subset(tr, dummy)
        tst_sp = HM.subset(tr, dummy)
    else:
        fit_idx = np.arange(n)
        train_sp = tr
        dev_sp = dv
        dummy = np.concatenate([np.flatnonzero(lab == c)[:HM.DUMMY_N_PER_CLASS]
                                for c in (0, 1)])
        tst_sp = HM.subset(tr, dummy)

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

    nsave = {"n": 0}

    def _no_save(obj, path, *ar, **kw):
        nsave["n"] += 1

    torch.save = _no_save

    # --------------------------------------------------------------------------- run
    tag = "{}_s{}_f{}".format(a.dataset, a.seed, a.fold)
    outp = os.path.join(a.scratch, "mint", tag)
    os.makedirs(outp, exist_ok=True)
    sys.argv = (["run_rac.py"] + HM.CLI[a.dataset]
                + ["--seed", str(a.seed), "--device", "cpu", "--num_workers", "0",
                   "--group_name", "C02A0_" + tag,
                   "--output_path", outp, "--force", "True"])
    args = run_rac.parse_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    t0 = time.time()
    run_rac.main(args)
    secs = time.time() - t0

    model = _HOLD["model"]
    model.eval()

    def keys_of(img, text):
        with torch.no_grad():
            _, emb = model(img, text, return_embed=True)
        return emb.detach().cpu().numpy().astype("float32")

    keys = {}
    for view in V.VIEW_NAMES:
        keys[view] = keys_of(tr[1], view_text[view])
    K_dev = keys_of(dv[1], dv[2]) if a.fold < 0 else np.zeros((0, keys["NAT"].shape[1]),
                                                              dtype="float32")

    meta = {"script_sha256": HM.sha256_of(os.path.abspath(__file__)),
            "frozen_headspace_mint_sha256": FROZEN_HEADSPACE_MINT_SHA,
            "frozen_pairverify_sha256": HM.FROZEN_PAIRVERIFY_SHA,
            "view_module_sha256": HM.sha256_of(
                os.path.join(REPO, "src/utils/c02_density_views.py")),
            "dataset": a.dataset, "ds": cfg["ds"], "encoder_model": model_name,
            "views": list(V.VIEW_NAMES),
            "seed": a.seed, "fold": a.fold, "n_train": int(n), "n_dev": int(len(dv[0])),
            "n_fit": int(len(fit_idx)), "head_dim": int(keys["NAT"].shape[1]),
            "argv": sys.argv, "secs": round(secs, 1),
            "fold_parity_vs_banked_vsw_ckpt": fold_parity,
            "n_state_dict_saves_suppressed": nsave["n"],
            "eval_curve": _CURVE,
            "test_contact": "NONE -- torch.load guard + patched loader + train-only "
                            "view loader; only train_*.pt and dev_seen_*.pt opened",
            "runtime": HM.runtime_block()}

    payload = {"lab": lab, "lab_dev": dv[3].numpy().astype(int),
               "fold_of": fold_of, "fit_idx": fit_idx, "K_dev": K_dev,
               "meta": json.dumps(meta)}
    for view in V.VIEW_NAMES:
        payload["K_" + view] = keys[view]
    tmp = a.out + ".tmp.npz"
    np.savez(tmp, **payload)
    os.replace(tmp, a.out)
    print("[c02mint] {} done in {:.1f}s -> {}".format(tag, secs, a.out))
    dev_rows = [r for r in _CURVE if r["split"] == "dev"]
    if dev_rows:
        print("[c02mint] {} dev acc final-epoch {:.4f}".format(tag, dev_rows[-1]["acc"]))


if __name__ == "__main__":
    main()
