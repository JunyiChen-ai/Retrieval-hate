"""
W4 wrapper: train a frozen-CLIP RGCL head on the temporal splits
(MHC_temporal / MHC_zh_temporal) WITHOUT modifying any existing src/ file.

run_rac.main() is reused verbatim. The only change is injected at runtime:
run_rac's module-level reference to load_feats_from_CLIP is replaced with a
temporal-aware loader that assembles [ids, img_feats, text_feats, labels]
for the temporal train/val/test splits by re-indexing the BASE dataset's
cached random-split CLIP embeddings by video id (the temporal split is a
re-partition of the same universe, sharing the same underlying .pt caches).

Every other dataset name falls through to the original loader, and the
original dataset.py / run_rac.py files on disk are untouched.

Usage: identical CLI to src/run_rac.py, e.g.
  python src/train_temporal_head.py --dataset MHC_temporal \
      --group_name RAC_video_temporal ... (same flags as run_rac.py)
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import torch

import run_rac
from eval_temporal_memory import TEMPORAL_BASE, load_temporal_feats

_orig_loader = run_rac.load_feats_from_CLIP


def _temporal_aware_loader(path, dataset, model, all=False):
    if dataset in TEMPORAL_BASE:
        return load_temporal_feats(path, dataset, model)
    return _orig_loader(path, dataset, model, all=all)


if __name__ == "__main__":
    args = run_rac.parse_args()
    if args.dataset not in TEMPORAL_BASE:
        raise SystemExit(
            "train_temporal_head.py only accepts --dataset in {}; use "
            "src/run_rac.py directly for other datasets.".format(
                sorted(TEMPORAL_BASE)))

    # inject the temporal-aware loader into run_rac's namespace (runtime only)
    run_rac.load_feats_from_CLIP = _temporal_aware_loader

    # mirror run_rac's __main__ seeding exactly
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    run_rac.main(args)
