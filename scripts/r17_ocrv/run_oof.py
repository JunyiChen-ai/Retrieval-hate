#!/usr/bin/env python
"""R17-OCRV P1: cross-fitted ActionFormer inside the 237 train videos, three feature arms.

Frozen in `idea-stage/R17_OCRV_FREEZE.md` (commit 1e268c6), written after it.

Structurally identical to `scripts/r16_detbase/run_af.py` -- same config, same dataset objects,
same optimiser / scheduler / EMA / `train_one_epoch`, same epoch-and-threshold selection on the
39-video val split, same metric.  The only differences are frozen in §2 of the freeze:

  * the annotation JSON is one of the three fold files, so `train` is 158 videos,
  * the evaluation subset is `oof` (79 held-out TRAIN videos), never `test`,
  * the 119 test videos carry subset `unused` and are asserted absent from every loaded split.

Per (arm, seed) the three folds' out-of-fold predictions are unioned into one 237-video
prediction set and the frozen endpoint -- corpus-level F1@tIoU0.5 -- is computed on it.
"""
from __future__ import annotations

import argparse
import copy
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

ROOT = Path("/home/jehc223/Retrieval-hate")
AF = ROOT / "third_party/actionformer"
sys.path.insert(0, str(AF))
sys.path.insert(0, str(ROOT / "scripts/r16_detbase"))

from libs.core import load_config                                    # noqa: E402
from libs.datasets import make_dataset, make_data_loader             # noqa: E402
from libs.modeling import make_meta_arch                             # noqa: E402
from libs.utils import (train_one_epoch, fix_random_seed,            # noqa: E402
                        make_optimizer, make_scheduler, ModelEma)
from eval_f1 import match_prf, sweep_threshold                       # noqa: E402
from run_af import thr_grid, gold_intervals, infer                   # noqa: E402

EMB = ROOT / "data/CLIP_Embedding/HateClipSeg"
ARMS = {
    "VAT":       (str(EMB / "dense4fps_vat"), 2816),
    "VATO":      (str(EMB / "dense4fps_vato"), 3584),
    "VATO_SHUF": (str(EMB / "dense4fps_vato_shuf"), 3584),
}
NFOLD = 3
TIOUS = (0.3, 0.5, 0.7)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", nargs="+", default=["VAT", "VATO", "VATO_SHUF"])
    ap.add_argument("--seeds", type=int, nargs="+", default=[6200, 6201, 6202])
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--out", default=str(ROOT / "idea-stage/r17_ocrv/out"))
    args = ap.parse_args()
    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)

    split = json.loads((ROOT / "data/gt/HateClipSeg/p11_split.json").read_text())
    test_ids = set(split["test"])
    print(f"[guard] split disjoint OK  train={len(split['train'])} val={len(split['val'])} "
          f"test={len(split['test'])}", flush=True)

    base = load_config(str(AF / "configs/hateclipseg_clip.yaml"))
    all_res = {}
    t_start = time.time()

    for arm in args.arms:
        feat, dim = ARMS[arm]
        for seed in args.seeds:
            oof_preds, oof_gold, per_fold = {}, {}, []
            for f in range(NFOLD):
                cfg = copy.deepcopy(base)
                cfg["dataset"]["json_file"] = str(
                    AF / f"data/hateclipseg/hateclipseg_rawseg_fold{f}.json")
                cfg["dataset"]["feat_folder"] = feat
                cfg["dataset"]["input_dim"] = dim
                cfg["model"]["input_dim"] = dim
                if args.epochs:
                    cfg["opt"]["epochs"] = args.epochs
                cfg["init_rand_seed"] = seed
                jf = cfg["dataset"]["json_file"]
                g_val = gold_intervals(jf, "val")
                g_oof = gold_intervals(jf, "oof")

                rng = fix_random_seed(seed, include_cuda=True)
                train_ds = make_dataset(cfg["dataset_name"], True, ["train"], **cfg["dataset"])
                val_ds = make_dataset(cfg["dataset_name"], False, ["val"], **cfg["dataset"])
                oof_ds = make_dataset(cfg["dataset_name"], False, ["oof"], **cfg["dataset"])
                tr_ids = set(train_ds.data_list[i]["id"] for i in range(len(train_ds)))
                va_ids = set(val_ds.data_list[i]["id"] for i in range(len(val_ds)))
                oo_ids = set(oof_ds.data_list[i]["id"] for i in range(len(oof_ds)))
                assert not (tr_ids & oo_ids) and not (tr_ids & va_ids) and not (va_ids & oo_ids)
                assert not ((tr_ids | va_ids | oo_ids) & test_ids), "TEST ID LOADED"
                cfg["model"]["train_cfg"]["head_empty_cls"] = \
                    train_ds.get_attributes()["empty_label_ids"]
                train_loader = make_data_loader(train_ds, True, rng, **cfg["loader"])
                val_loader = make_data_loader(val_ds, False, None, 1, cfg["loader"]["num_workers"])
                oof_loader = make_data_loader(oof_ds, False, None, 1, cfg["loader"]["num_workers"])

                model = nn.DataParallel(make_meta_arch(cfg["model_name"], **cfg["model"]),
                                        device_ids=cfg["devices"])
                opt = make_optimizer(model, cfg["opt"])
                sch = make_scheduler(opt, cfg["opt"], len(train_loader))
                ema = ModelEma(model)

                best = dict(f1=-1.0, epoch=-1, thr=None)
                best_state = None
                for ep in range(cfg["opt"]["epochs"] + cfg["opt"]["warmup_epochs"]):
                    train_one_epoch(train_loader, model, opt, sch, ep, model_ema=ema,
                                    clip_grad_l2norm=cfg["train_cfg"]["clip_grad_l2norm"],
                                    tb_writer=None, print_freq=10 ** 9)
                    vp = infer(ema.module, val_loader)
                    thr, f1, _ = sweep_threshold(vp, g_val, thr_grid(vp), tiou=0.5)
                    if f1 > best["f1"]:
                        best = dict(f1=f1, epoch=ep, thr=thr)
                        best_state = copy.deepcopy(ema.module.state_dict())
                ema.module.load_state_dict(best_state)
                op = infer(ema.module, oof_loader)
                (outdir / f"pool_{arm}_s{seed}_f{f}.json").write_text(json.dumps(op))
                kept = {v: [p for p in ps if p[2] >= best["thr"]] for v, ps in op.items()}
                oof_preds.update(kept)
                oof_gold.update(g_oof)
                per_fold.append(dict(fold=f, best_epoch=best["epoch"], thr=best["thr"],
                                     val_f1_50=best["f1"],
                                     oof_f1_50=match_prf(kept, g_oof, 0.5)["F1"]))
                print(f"  [{arm} s{seed} f{f}] ep={best['epoch']} thr={best['thr']:.3f} "
                      f"val={best['f1']:.2f} oof={per_fold[-1]['oof_f1_50']:.2f} "
                      f"t={time.time()-t_start:.0f}s", flush=True)
                del model, ema, best_state
                torch.cuda.empty_cache()

            assert len(oof_gold) == 237, len(oof_gold)
            m = {str(t): match_prf(oof_preds, oof_gold, t) for t in TIOUS}
            key = f"{arm}|{seed}"
            all_res[key] = dict(arm=arm, seed=seed, folds=per_fold, oof=m,
                                n_pred=sum(len(v) for v in oof_preds.values()))
            (outdir / f"preds_oof_{arm}_s{seed}.json").write_text(json.dumps(oof_preds))
            (outdir / "res_p1.json").write_text(json.dumps(all_res, indent=1))
            print(f"[OOF] {arm} seed={seed} " + "  ".join(
                f"t{t}: F1={m[str(t)]['F1']:.2f} P={m[str(t)]['P']:.2f} R={m[str(t)]['R']:.2f}"
                for t in TIOUS), flush=True)

    print("\n=== P1 SUMMARY (237 out-of-fold train videos) ===", flush=True)
    for arm in args.arms:
        for t in TIOUS:
            xs = [all_res[k]["oof"][str(t)]["F1"] for k in all_res if all_res[k]["arm"] == arm]
            print(f"  {arm:10s} tIoU {t}: F1 {np.mean(xs):.2f} +- {np.std(xs):.2f}  "
                  f"({', '.join(f'{x:.2f}' for x in xs)})", flush=True)
    print(f"[wall] {time.time()-t_start:.0f}s", flush=True)


if __name__ == "__main__":
    main()
