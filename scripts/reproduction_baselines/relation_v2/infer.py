#!/usr/bin/env python3
"""Corpus-bound dense inference for Relation-V2."""

from __future__ import annotations

import argparse
import json
import os
import sys
from types import SimpleNamespace

import numpy as np
import torch
from torch.utils.data import DataLoader

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

from hate_common import data as hdata  # noqa: E402
from powa_macil.dataset import PowaTestDataset, usable_text_ids  # noqa: E402
from relation_v2.model import RelationV2  # noqa: E402
from relation_v2.protocol import (checkpoint_corpus, frozen_splits,
                                  sha256_file)  # noqa: E402


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint-dir", required=True)
    ap.add_argument("--corpus", required=True, choices=hdata.CORPORA)
    ap.add_argument("--split", default="test", choices=("val", "test"))
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)
    meta_path = os.path.join(args.checkpoint_dir, "train_meta.json")
    model_path = os.path.join(args.checkpoint_dir, "model.pth")
    complete_path = os.path.join(args.checkpoint_dir, "COMPLETE.json")
    with open(meta_path) as fh:
        meta = json.load(fh)
    with open(complete_path) as fh:
        complete = json.load(fh)
    checkpoint_corpus(meta, args.corpus)
    checkpoint_corpus(complete, args.corpus)
    if (complete.get("model_sha256") != sha256_file(model_path) or
            complete.get("meta_sha256") != sha256_file(meta_path) or
            meta.get("model_sha256") != complete.get("model_sha256")):
        raise RuntimeError("checkpoint/meta completion hash mismatch")
    cfg = SimpleNamespace(**meta["args"])
    manifest_ids = frozen_splits(args.corpus)[args.split]
    ids = usable_text_ids(args.corpus, manifest_ids)
    if ids != manifest_ids:
        raise RuntimeError("inference features do not exactly cover manifest")
    gt = hdata.gt_arrays(args.corpus, args.split)
    eval_ids = [v for v in ids if v in gt]
    ds = PowaTestDataset(args.corpus, eval_ids, cfg.max_seqlen, cfg.grid, "av")
    loader = DataLoader(ds, batch_size=1, shuffle=False,
                        num_workers=cfg.num_workers)
    model = RelationV2(cfg).to(args.device)
    model.load_state_dict(torch.load(model_path,
                                     map_location=args.device))
    model.eval()
    path = args.out or os.path.join(args.checkpoint_dir, args.corpus,
                                    "%s_scores.jsonl" % args.split)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    seen = set()
    with torch.no_grad(), open(path, "w") as fh:
        for f_v, f_a, f_t, index_map, _, vid in loader:
            vid = vid[0]
            if vid in seen:
                raise RuntimeError("duplicate inference ID %s" % vid)
            seen.add(vid)
            f_v, f_a, f_t = f_v[0].to(args.device), f_a[0].to(args.device), f_t[0].to(args.device)
            lengths = torch.full((f_v.shape[0],), f_v.shape[1], dtype=torch.long)
            out = model(f_a, f_v, f_t, lengths)
            score = out["frame_prob"].mean(0).cpu().numpy()[index_map[0].numpy()]
            if len(score) != len(gt[vid]) or not np.isfinite(score).all():
                raise RuntimeError("inference alignment/nonfinite %s" % vid)
            fh.write(json.dumps({"video_id": vid,
                                 "score_relation_v2": score.tolist()}) + "\n")
    if seen != set(gt):
        raise RuntimeError("inference GT coverage mismatch: missing=%s extra=%s" %
                           (sorted(set(gt) - seen)[:5],
                            sorted(seen - set(gt))[:5]))
    print("wrote %d videos to %s" % (len(seen), path))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
