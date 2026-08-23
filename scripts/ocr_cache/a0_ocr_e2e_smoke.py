#!/usr/bin/env python
"""Synthetic smoke for the A0 +- OCR end-to-end run.

Builds a scratch data root whose HateMM caches are RANDOM tensors with fake ids
and fake labels (same shapes/split sizes as the real cache), plus matching random
OCR archive files, then prints the two arm command lines that the real runner
will use, pointed at the scratch root. Nothing real is read, so no real val
metric can be seen while the code is being validated.

Usage:
  python scripts/ocr_cache/a0_ocr_e2e_smoke.py --out /path/to/scratch
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import torch

MODEL = "openai_clip-vit-large-patch14-336_HF"
SIZES = {"train": 120, "dev_seen": 40, "test_seen": 40}
IMG_D, TXT_D, OCR_D = 1024, 768, 768


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    root = Path(a.out)
    ce = root / "CLIP_Embedding/HateMM"
    ocr = root / "OCR/HateMM"
    ce.mkdir(parents=True, exist_ok=True)
    ocr.mkdir(parents=True, exist_ok=True)

    g = torch.Generator().manual_seed(1234)
    dev_arc = None
    for split, n in SIZES.items():
        ids = ["%s_fake_%d" % (split, i) for i in range(n)]
        torch.save({
            "ids": [ids],
            "img_feats": torch.randn(n, IMG_D, generator=g),
            "text_feats": torch.randn(n, TXT_D, generator=g),
            "labels": (torch.rand(n, generator=g) < 0.4).long(),
        }, ce / ("%s_%s.pt" % (split, MODEL)))
        # Mirror the real layout: the test_seen OCR file is a PLACEHOLDER holding
        # the dev_seen rows, because --val_only_eval swaps dev into the test slot.
        arc = ({"ids": ids, "text_feats": torch.randn(n, OCR_D, generator=g)}
               if split != "test_seen" else dev_arc)
        if split == "dev_seen":
            dev_arc = arc
        torch.save(arc, ocr / ("rac_ocrmean30_%s.pt" % split))
        print("wrote synthetic %s (%d rows)" % (split, n))

    print("\nSCRATCH_ROOT=%s" % root)


if __name__ == "__main__":
    main()
