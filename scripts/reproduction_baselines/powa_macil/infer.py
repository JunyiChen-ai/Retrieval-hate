#!/usr/bin/env python3
"""Dense frame-score export for a frozen POWA-MACIL checkpoint."""

from __future__ import annotations

import argparse
import json
import os
import sys
from types import SimpleNamespace

import torch
from torch.utils.data import DataLoader

HERE = os.path.dirname(os.path.abspath(__file__))
PARENT = os.path.dirname(HERE)
sys.path.insert(0, PARENT)

from hate_common import data as hdata  # noqa: E402
from powa_macil.dataset import PowaTestDataset, usable_text_ids  # noqa: E402
from powa_macil.model import POWAMACIL  # noqa: E402


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint-dir", required=True)
    ap.add_argument("--corpus", required=True, choices=list(hdata.CORPORA))
    ap.add_argument("--split", default="test")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)
    with open(os.path.join(args.checkpoint_dir, "train_meta.json")) as fh:
        meta = json.load(fh)
    cfg = SimpleNamespace(**meta["args"])
    ids = usable_text_ids(args.corpus, hdata.load_split(args.corpus, args.split))
    gt = hdata.gt_arrays(args.corpus, args.split)
    ids = [v for v in ids if v in gt]
    ds = PowaTestDataset(args.corpus, ids, cfg.max_seqlen, cfg.grid, "av")
    loader = DataLoader(ds, batch_size=1, shuffle=False,
                        num_workers=getattr(cfg, "num_workers", 0))
    model = POWAMACIL(cfg).to(args.device)
    state = torch.load(os.path.join(args.checkpoint_dir, "model.pth"),
                       map_location=args.device)
    legacy_typed_only = "policy_residual_gate" not in state
    model.load_state_dict(state, strict=not legacy_typed_only)
    model.use_policy_residual = (not legacy_typed_only and
                                 not getattr(cfg, "typed_only", False))
    model.eval()
    out_path = args.out or os.path.join(args.checkpoint_dir, args.corpus,
                                        "scores.jsonl")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with torch.no_grad(), open(out_path, "w") as fh:
        for f_v, f_a, f_t, index_map, n_seconds, vid in loader:
            vid = vid[0]
            f_v, f_a, f_t = f_v[0].to(args.device), f_a[0].to(args.device), f_t[0].to(args.device)
            lengths = torch.full((f_v.shape[0],), f_v.shape[1], dtype=torch.long)
            pred = model(f_a, f_v, f_t, lengths, policy=args.corpus)
            score = pred["frame_prob"].mean(0).cpu().numpy()
            score = score[index_map[0].numpy()]
            base = torch.sigmoid(pred["base_frame_logits"].squeeze(-1)).mean(0)
            base = base.cpu().numpy()[index_map[0].numpy()]
            audio = torch.sigmoid(pred["audio_logits"].squeeze(-1)).mean(0)
            audio = audio.cpu().numpy()[index_map[0].numpy()]
            visual = torch.sigmoid(pred["visual_logits"].squeeze(-1)).mean(0)
            visual = visual.cpu().numpy()[index_map[0].numpy()]
            if len(score) != int(n_seconds) or len(score) != len(gt[vid]):
                raise RuntimeError("alignment mismatch %s" % vid)
            row = {"video_id": vid, "n_frames": len(score),
                   "score_powa": [round(float(x), 6) for x in score],
                   "score_base": [round(float(x), 6) for x in base],
                   "score_audio": [round(float(x), 6) for x in audio],
                   "score_visual": [round(float(x), 6) for x in visual]}
            fh.write(json.dumps(row) + "\n")
    print("wrote %d videos to %s" % (len(ids), out_path))
    return 0


if __name__ == "__main__":
    sys.exit(main())
