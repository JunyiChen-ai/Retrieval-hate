#!/usr/bin/env python
"""R16-DETBASE: train the official ActionFormer on HateClipSeg and report F1@tIoU.

Mirrors `third_party/actionformer/train.py` step for step (same dataset objects, same model
factory, same optimizer/scheduler, same EMA, same `train_one_epoch`/`valid_one_epoch`), and
adds only the two things the official scripts do not provide:
  1. per-epoch validation on the 39-video val split with the paper's F1@tIoU metric,
  2. selection of (epoch, score threshold) on val, then a single evaluation pass on test.

Test is opened once, at the end, and only when --touch-test is given.
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
from eval_f1 import prf_at, sweep_threshold, average_precision       # noqa: E402

def thr_grid(preds):
    """Data-driven score-threshold grid: 200 quantiles of the observed proposal scores.
    A fixed absolute grid is unusable because ActionFormer's post-SoftNMS score scale moves
    with the epoch; quantiles cover the whole operating curve at any scale."""
    s = np.array([p[2] for ps in preds.values() for p in ps], dtype=np.float64)
    if s.size == 0:
        return [0.0]
    q = np.unique(np.quantile(s, np.linspace(0.0, 0.999, 200)))
    return [float(x) for x in q]


def gold_intervals(json_file, subset):
    db = json.loads(Path(json_file).read_text())["database"]
    return {v: [tuple(a["segment"]) for a in d["annotations"]]
            for v, d in db.items() if d["subset"] == subset}


@torch.no_grad()
def infer(model, loader):
    model.eval()
    preds = {}
    for video_list in loader:
        out = model(video_list)
        for o in out:
            segs = o["segments"].cpu().numpy()
            scr = o["scores"].cpu().numpy()
            preds[o["video_id"]] = [(float(a), float(b), float(c))
                                    for (a, b), c in zip(segs, scr)]
    return preds


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(AF / "configs/hateclipseg_clip.yaml"))
    ap.add_argument("--gt", default="blocks", choices=["blocks", "rawseg"])
    ap.add_argument("--feat", default=None, help="override feat_folder")
    ap.add_argument("--input-dim", type=int, default=None)
    ap.add_argument("--seeds", type=int, nargs="+", default=[5100, 5101, 5102])
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--touch-test", action="store_true")
    ap.add_argument("--tag", default="v_blocks")
    ap.add_argument("--out", default=str(ROOT / "idea-stage/r16_detbase/out"))
    ap.add_argument("--dump-preds", action="store_true")
    args = ap.parse_args()

    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)

    cfg = load_config(args.config)
    cfg["dataset"]["json_file"] = str(AF / f"data/hateclipseg/hateclipseg_{args.gt}.json")
    if args.feat:
        cfg["dataset"]["feat_folder"] = args.feat
    if args.input_dim:
        cfg["dataset"]["input_dim"] = args.input_dim
        cfg["model"]["input_dim"] = args.input_dim
    if args.epochs:
        cfg["opt"]["epochs"] = args.epochs
    jf = cfg["dataset"]["json_file"]

    g_val = gold_intervals(jf, "val")
    g_test = gold_intervals(jf, "test") if args.touch_test else None
    print(f"[cfg] gt={args.gt} feat={cfg['dataset']['feat_folder']} "
          f"dim={cfg['dataset']['input_dim']} epochs={cfg['opt']['epochs']} "
          f"seeds={args.seeds} touch_test={args.touch_test}", flush=True)
    print(f"[gold] val videos={len(g_val)} instances={sum(len(v) for v in g_val.values())}",
          flush=True)

    all_res = {}
    for seed in args.seeds:
        t0 = time.time()
        cfg_s = copy.deepcopy(cfg)
        cfg_s["init_rand_seed"] = seed
        rng = fix_random_seed(seed, include_cuda=True)

        train_ds = make_dataset(cfg_s["dataset_name"], True, ["train"], **cfg_s["dataset"])
        val_ds = make_dataset(cfg_s["dataset_name"], False, ["val"], **cfg_s["dataset"])
        cfg_s["model"]["train_cfg"]["head_empty_cls"] = \
            train_ds.get_attributes()["empty_label_ids"]
        train_loader = make_data_loader(train_ds, True, rng, **cfg_s["loader"])
        val_loader = make_data_loader(val_ds, False, None, 1, cfg_s["loader"]["num_workers"])
        print(f"[data] seed={seed} train={len(train_ds)} val={len(val_ds)}", flush=True)

        model = make_meta_arch(cfg_s["model_name"], **cfg_s["model"])
        model = nn.DataParallel(model, device_ids=cfg_s["devices"])
        opt = make_optimizer(model, cfg_s["opt"])
        sch = make_scheduler(opt, cfg_s["opt"], len(train_loader))
        ema = ModelEma(model)

        max_ep = cfg_s["opt"]["epochs"] + cfg_s["opt"]["warmup_epochs"]
        best = dict(f1=-1.0, epoch=-1, thr=None)
        best_state = None
        curve = []
        for ep in range(max_ep):
            train_one_epoch(train_loader, model, opt, sch, ep, model_ema=ema,
                            clip_grad_l2norm=cfg_s["train_cfg"]["clip_grad_l2norm"],
                            tb_writer=None, print_freq=10 ** 9)
            vp = infer(ema.module, val_loader)
            thr, f1, _ = sweep_threshold(vp, g_val, thr_grid(vp), tiou=0.5)
            curve.append(dict(epoch=ep, val_f1_50=f1, thr=thr))
            print(f"  [ep {ep:02d}] val F1@0.5={f1:.2f} (thr={thr:.2f}) "
                  f"t={time.time()-t0:.0f}s", flush=True)
            if f1 > best["f1"]:
                best = dict(f1=f1, epoch=ep, thr=thr)
                best_state = copy.deepcopy(ema.module.state_dict())

        print(f"[select] seed={seed} best epoch={best['epoch']} thr={best['thr']:.2f} "
              f"val F1@0.5={best['f1']:.2f}", flush=True)
        ema.module.load_state_dict(best_state)
        vp = infer(ema.module, val_loader)
        val_m = prf_at(vp, g_val, best["thr"])
        res = dict(seed=seed, best_epoch=best["epoch"], thr=best["thr"], curve=curve,
                   val={str(k): v for k, v in val_m.items()},
                   val_ap={str(t): average_precision(vp, g_val, t)
                           for t in (0.3, 0.5, 0.7)})

        if args.touch_test:
            test_ds = make_dataset(cfg_s["dataset_name"], False, ["test"], **cfg_s["dataset"])
            test_loader = make_data_loader(test_ds, False, None, 1,
                                           cfg_s["loader"]["num_workers"])
            assert not (set(test_ds.data_list[i]["id"] for i in range(len(test_ds))) &
                        set(train_ds.data_list[i]["id"] for i in range(len(train_ds))))
            tp_ = infer(ema.module, test_loader)
            test_m = prf_at(tp_, g_test, best["thr"])
            res["test"] = {str(k): v for k, v in test_m.items()}
            res["test_ap"] = {str(t): average_precision(tp_, g_test, t)
                              for t in (0.3, 0.5, 0.7)}
            print(f"[TEST] seed={seed} " + " ".join(
                f"tIoU{t}: F1={test_m[t]['F1']:.2f} P={test_m[t]['P']:.2f} "
                f"R={test_m[t]['R']:.2f}" for t in (0.3, 0.5, 0.7)), flush=True)
            if args.dump_preds:
                (outdir / f"preds_test_{args.tag}_s{seed}.json").write_text(json.dumps(tp_))
        if args.dump_preds:
            (outdir / f"preds_val_{args.tag}_s{seed}.json").write_text(json.dumps(vp))
        res["wall_s"] = time.time() - t0
        all_res[str(seed)] = res
        (outdir / f"res_{args.tag}.json").write_text(json.dumps(all_res, indent=1))
        del model, ema, best_state
        torch.cuda.empty_cache()

    # summary
    def agg(key, t, field):
        xs = [all_res[s][key][str(t)][field] for s in all_res if key in all_res[s]]
        return float(np.mean(xs)), float(np.std(xs))

    print("\n=== SUMMARY tag=%s (%d seeds) ===" % (args.tag, len(all_res)), flush=True)
    for split in ("val", "test"):
        if not any(split in all_res[s] for s in all_res):
            continue
        for t in (0.3, 0.5, 0.7):
            m, sd = agg(split, t, "F1")
            p, ps = agg(split, t, "P")
            r, rs = agg(split, t, "R")
            print(f"{split:5s} tIoU {t}: F1 {m:.2f}+-{sd:.2f}  P {p:.2f}+-{ps:.2f}  "
                  f"R {r:.2f}+-{rs:.2f}", flush=True)
    (outdir / f"res_{args.tag}.json").write_text(json.dumps(all_res, indent=1))


if __name__ == "__main__":
    main()
