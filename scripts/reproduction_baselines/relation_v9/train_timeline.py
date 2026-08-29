"""Label-free frozen train cohort and 1 fps support for V9 evidence producers."""
import json
from pathlib import Path

from relation_v2.protocol import frozen_splits
from relation_v4.io import sha256

INDEX = Path("results/reproduction/features/vggish_1s/hatemm/index.json")


def hatemm_train_timeline():
    ids = list(frozen_splits("hatemm")["train"])
    index = json.loads(INDEX.read_text())
    missing = sorted(set(ids) - set(index))
    if missing:
        raise RuntimeError(f"label-free HateMM timeline missing IDs: {missing[:3]}")
    lengths = {vid: int(index[vid]["n_frames"]) for vid in ids}
    if any(value <= 0 for value in lengths.values()):
        raise RuntimeError("nonpositive label-free timeline length")
    return ids, lengths, {"path": str(INDEX.resolve()), "sha256": sha256(INDEX),
                          "kind": "label-free frozen VGGish 1fps support"}
