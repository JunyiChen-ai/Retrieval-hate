#!/usr/bin/env python3
"""Audit whether declared primitive channels track their fixed semantic anchors."""

import argparse
import json
import os
import sys
from types import SimpleNamespace

import numpy as np
import torch
from scipy.stats import spearmanr
from torch.utils.data import DataLoader

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from hate_common import data as hdata  # noqa: E402
from powa_macil.dataset import PowaTestDataset, usable_text_ids  # noqa: E402
from powa_macil.model import POWAMACIL, PRIMITIVES  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint-dir", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    with open(os.path.join(args.checkpoint_dir, "train_meta.json")) as fh:
        cfg = SimpleNamespace(**json.load(fh)["args"])
    model = POWAMACIL(cfg).to(args.device)
    model.load_state_dict(torch.load(os.path.join(args.checkpoint_dir, "model.pth"),
                                     map_location=args.device))
    model.eval()
    predicted, anchored = [], []
    with torch.no_grad():
        for corpus in hdata.CORPORA:
            ids = usable_text_ids(corpus, hdata.load_split(corpus, "val"))
            ds = PowaTestDataset(corpus, ids, cfg.max_seqlen, cfg.grid, "av")
            for f_v, f_a, f_t, _, _, _ in DataLoader(ds, batch_size=1):
                f_v, f_a, f_t = f_v[0].to(args.device), f_a[0].to(args.device), f_t[0].to(args.device)
                lengths = torch.full((f_v.shape[0],), f_v.shape[1], dtype=torch.long)
                out = model(f_a, f_v, f_t, lengths, policy=corpus)
                if out["semantic_logits"] is None:
                    raise RuntimeError("checkpoint has no semantic grounding")
                mask = out["semantic_text_mask"][0].cpu().numpy().astype(bool)
                predicted.append(out["primitive_prob"].mean(0).cpu().numpy()[mask])
                anchored.append(torch.sigmoid(out["semantic_logits"]).mean(0).cpu().numpy()[mask])
    predicted, anchored = np.concatenate(predicted), np.concatenate(anchored)
    matrix = np.asarray([[spearmanr(predicted[:, i], anchored[:, j]).statistic
                          for j in range(len(PRIMITIVES))]
                         for i in range(len(PRIMITIVES))])
    eye = np.eye(len(PRIMITIVES), dtype=bool)
    result = {"channels": list(PRIMITIVES), "n_text_frames": len(predicted),
              "spearman_matrix": matrix.tolist(),
              "mean_declared_channel": float(np.diag(matrix).mean()),
              "mean_wrong_channel": float(matrix[~eye].mean()),
              "declared_minus_wrong": float(np.diag(matrix).mean() - matrix[~eye].mean())}
    out = args.out or os.path.join(args.checkpoint_dir, "semantic_grounding_audit.json")
    with open(out, "w") as fh:
        json.dump(result, fh, indent=2)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
