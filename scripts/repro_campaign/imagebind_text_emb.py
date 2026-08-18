#!/usr/bin/env python
"""Dump the two ZS-ImageBind text prompt embeddings (freeze §9, Wave 0).

Prompt pair, fixed a priori and never tuned: ["normal", "hateful"].  Column 1 of
the resulting (2, 1024) array is the hateful prompt; the score used by
`eval_frame.py` is softmax over the pair, i.e. a monotone function of
sim(hateful) - sim(normal), which is what a two-prompt zero-shot baseline reports.

Runs on CPU: it is two forward passes, and the GPU is busy with the extractors.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path("/home/jehc223/Retrieval-hate")
IB = ROOT / "third_party/lavad/libs/ImageBind"
PROMPTS = ["normal", "hateful"]
OUT = ROOT / "data/CLIP_Embedding/imagebind_text_normal_hateful.npy"


def main() -> int:
    sys.path.insert(0, str(IB))
    os.chdir(ROOT / "third_party/lavad")  # imagebind.data.BPE_PATH is relative
    from imagebind import data as ibdata
    from imagebind.models import imagebind_model
    from imagebind.models.imagebind_model import ModalityType

    m = imagebind_model.imagebind_huge(pretrained=False)
    m.load_state_dict(torch.load(ROOT / "third_party/_ckpt/imagebind_huge.pth",
                                 map_location="cpu"))
    m = m.eval()
    with torch.no_grad():
        tok = ibdata.load_and_transform_text(PROMPTS, "cpu")
        e = m({ModalityType.TEXT: tok})[ModalityType.TEXT].float().numpy()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    np.save(OUT, e)
    print(f"[ok] {OUT} shape={e.shape} prompts={PROMPTS} "
          f"norms={np.linalg.norm(e, axis=-1).round(3).tolist()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
