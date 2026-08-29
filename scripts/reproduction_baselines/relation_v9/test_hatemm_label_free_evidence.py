#!/usr/bin/env python3
"""Static/runtime guards for the label-free HateMM train evidence path."""
import ast
from pathlib import Path

import numpy as np

from macilsd.dataset import usable_ids
from relation_v9.train_timeline import hatemm_train_timeline

ROOT = Path(__file__).resolve().parents[3]
AUDITED = (
    "scripts/reproduction_baselines/relation_v9/train_timeline.py",
    "scripts/reproduction_baselines/relation_v9/produce_hatemm_macil_train.py",
    "scripts/reproduction_baselines/macilsd/infer_train_label_free.py",
    "scripts/reproduction_baselines/relation_v9/vera_hatemm_train_label_free.py",
    "scripts/reproduction_baselines/relation_v9/finalize_hatemm_vera_train.py",
    "scripts/reproduction_baselines/relation_v9/hatemm_preflight.py",
)
FORBIDDEN_NAMES = {"gt_arrays", "build_gt_array", "scoped_labels"}


def main():
    for relative in AUDITED:
        source = (ROOT / relative).read_text(); tree = ast.parse(source, relative)
        calls = {node.func.attr if isinstance(node.func, ast.Attribute) else node.func.id
                 for node in ast.walk(tree) if isinstance(node, ast.Call)
                 and isinstance(node.func, (ast.Attribute, ast.Name))}
        imports = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imports.update(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                imports.add(node.module or "")
                imports.update(f"{node.module}.{alias.name}" for alias in node.names)
        assert not (calls & FORBIDDEN_NAMES), (relative, calls & FORBIDDEN_NAMES)
        assert not any(name == "hate_common.data" or name.startswith("hate_common.data.")
                       for name in imports), (relative, "localization data import")
    ids, lengths, _ = hatemm_train_timeline()
    assert len(ids) == len(set(ids)) == 744
    assert set(usable_ids("hatemm", ids)) == set(ids)
    for root in (ROOT / "results/reproduction/features/vggish_1s/hatemm",
                 ROOT / "results/reproduction/features/clip_b16_1fps/hatemm"):
        for vid in ids:
            value = np.load(root / f"{vid}.npy", mmap_mode="r")
            assert len(value) == lengths[vid], (root.name, vid, len(value), lengths[vid])
    print("V9 HateMM label-free evidence static/744 coverage: PASS")


if __name__ == "__main__": main()
