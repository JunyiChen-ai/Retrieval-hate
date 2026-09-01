#!/usr/bin/env python
"""Export the selected model's single raw frame_prob score."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import torch
from torch.utils.data import DataLoader

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
BASE = ROOT / "scripts" / "reproduction_baselines"
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(BASE))

from dataset import SourceTestDataset  # noqa: E402
from hate_common import data as hdata  # noqa: E402
from method import ActiveSpeakerBoundPOWA  # noqa: E402
from powa_macil.dataset import usable_text_ids  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint-dir", required=True)
    ap.add_argument("--corpus", required=True,
                    choices=("hatemm", "hateclipseg"))
    ap.add_argument("--split", default="test", choices=("val", "test"))
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    checkpoint_dir = Path(args.checkpoint_dir)
    meta = json.loads((checkpoint_dir / "train_meta.json").read_text())
    if meta["corpus"] != args.corpus:
        raise ValueError("checkpoint corpus mismatch")
    cfg = SimpleNamespace(**meta["config"])
    data_arm = "permuted" if meta["arm"] == "permuted" else "core"
    ids = usable_text_ids(args.corpus, hdata.load_split(args.corpus, args.split))
    gt = hdata.gt_arrays(args.corpus, args.split)
    ids = [vid for vid in ids if vid in gt]
    loader = DataLoader(SourceTestDataset(
        args.corpus, ids, cfg.max_seqlen, cfg.grid, "av", arm=data_arm),
        batch_size=1, shuffle=False, num_workers=cfg.num_workers)
    model = ActiveSpeakerBoundPOWA(cfg).to(args.device)
    model.load_state_dict(torch.load(checkpoint_dir / "model.pt",
                                     map_location=args.device))
    model.use_policy_residual = not cfg.typed_only
    model.eval()
    destination = Path(args.out)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with torch.no_grad(), destination.open("w") as handle:
        for batch in loader:
            (f_v, f_a, f_t, index_map, n_seconds, vid, face, state,
             source_utterance) = batch
            name = vid[0]
            f_v, f_a, f_t = (f_v[0].float().to(args.device),
                              f_a[0].float().to(args.device),
                              f_t[0].float().to(args.device))
            face = face[0].float().to(args.device)
            state = state[0].long().to(args.device)
            source_utterance = source_utterance[0].float().to(args.device)
            lengths = torch.full((f_v.shape[0],), f_v.shape[1], dtype=torch.long)
            pred = model(f_a, f_v, f_t, face, state, source_utterance, lengths,
                         policy=args.corpus)
            score = pred["frame_prob"].mean(0).cpu().numpy()
            score = score[index_map[0].numpy()]
            if len(score) != int(n_seconds) or len(score) != len(gt[name]):
                raise RuntimeError(f"test alignment mismatch {args.corpus}/{name}")
            handle.write(json.dumps({
                "video_id": name,
                "score_method": [float(value) for value in score]}) + "\n")
    print(f"wrote {len(ids)} videos to {destination}")


if __name__ == "__main__":
    main()
