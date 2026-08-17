#!/usr/bin/env python
"""R16-DETBASE post-hoc arm: does the paper's 80% training share explain the residual gap?

Our frozen split gives ActionFormer 237 training videos (60%); the paper gives it 80%.  This
retrains on train+val (276 videos, 70%) with NO new selection surface: for each seed the epoch
count and the proposal-score threshold are taken from that seed's train-only run, which selected
them on val.  Descriptive, post-hoc, no gate.
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

from libs.core import load_config                                  # noqa: E402
from libs.datasets import make_dataset, make_data_loader           # noqa: E402
from libs.modeling import make_meta_arch                           # noqa: E402
from libs.utils import (train_one_epoch, fix_random_seed,          # noqa: E402
                        make_optimizer, make_scheduler, ModelEma)
from eval_f1 import prf_at                                         # noqa: E402
from run_af import gold_intervals, infer                           # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(AF / "configs/hateclipseg_clip.yaml"))
    ap.add_argument("--gt", default="rawseg")
    ap.add_argument("--from-res", default="res_v_rawseg.json")
    ap.add_argument("--tag", default="v_rawseg_trainval")
    ap.add_argument("--out", default=str(ROOT / "idea-stage/r16_detbase/out"))
    args = ap.parse_args()
    outdir = Path(args.out)

    src = json.loads((outdir / args.from_res).read_text())
    cfg = load_config(args.config)
    cfg["dataset"]["json_file"] = str(AF / f"data/hateclipseg/hateclipseg_{args.gt}.json")
    g_test = gold_intervals(cfg["dataset"]["json_file"], "test")

    all_res = {}
    for seed_s, ref in src.items():
        seed = int(seed_s)
        t0 = time.time()
        n_ep = int(ref["best_epoch"]) + 1
        thr = float(ref["thr"])
        cfg_s = copy.deepcopy(cfg)
        rng = fix_random_seed(seed, include_cuda=True)
        tr = make_dataset(cfg_s["dataset_name"], True, ["train", "val"], **cfg_s["dataset"])
        te = make_dataset(cfg_s["dataset_name"], False, ["test"], **cfg_s["dataset"])
        loader = make_data_loader(tr, True, rng, **cfg_s["loader"])
        te_loader = make_data_loader(te, False, None, 1, cfg_s["loader"]["num_workers"])
        cfg_s["model"]["train_cfg"]["head_empty_cls"] = tr.get_attributes()["empty_label_ids"]
        model = nn.DataParallel(make_meta_arch(cfg_s["model_name"], **cfg_s["model"]),
                                device_ids=cfg_s["devices"])
        opt = make_optimizer(model, cfg_s["opt"])
        sch = make_scheduler(opt, cfg_s["opt"], len(loader))
        ema = ModelEma(model)
        print(f"[seed {seed}] train+val={len(tr)} epochs={n_ep} thr={thr:.3f}", flush=True)
        for ep in range(n_ep):
            train_one_epoch(loader, model, opt, sch, ep, model_ema=ema,
                            clip_grad_l2norm=cfg_s["train_cfg"]["clip_grad_l2norm"],
                            tb_writer=None, print_freq=10 ** 9)
        preds = infer(ema.module, te_loader)
        m = prf_at(preds, g_test, thr)
        all_res[seed_s] = {str(k): v for k, v in m.items()}
        print(f"[TEST] seed={seed} " + " ".join(
            f"t{t}: F1={m[t]['F1']:.2f} P={m[t]['P']:.2f} R={m[t]['R']:.2f}"
            for t in (0.3, 0.5, 0.7)) + f"  ({time.time()-t0:.0f}s)", flush=True)
        del model, ema
        torch.cuda.empty_cache()

    for t in (0.3, 0.5, 0.7):
        f = [all_res[s][str(t)]["F1"] for s in all_res]
        print(f"[SUMMARY {args.tag}] test tIoU {t}: F1 {np.mean(f):.2f}+-{np.std(f):.2f}",
              flush=True)
    (outdir / f"res_{args.tag}.json").write_text(json.dumps(all_res, indent=1))


if __name__ == "__main__":
    main()
