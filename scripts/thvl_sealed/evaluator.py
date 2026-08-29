from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
from cryptography.fernet import Fernet

from scripts.dehate_sealed.core import SealedEvaluator, rasterize_1hz


def steward_decryptor(key_path: Path, duration_manifest: Path):
    """Build a decryptor without exposing labels to the development process."""
    key = key_path.read_bytes()
    durations = json.loads(duration_manifest.read_text())
    if durations.get("schema_version") != 1 or set(durations) != {"schema_version", "duration_seconds_by_hashed_id"}:
        raise RuntimeError("invalid steward duration manifest")
    duration_map = durations["duration_seconds_by_hashed_id"]

    def decrypt(bundle_path: Path) -> dict:
        payload = json.loads(Fernet(key).decrypt(bundle_path.read_bytes()))
        spans = payload["temporal_gt"]
        if set(spans) != set(duration_map):
            raise RuntimeError("sealed GT/media-duration cohort mismatch")
        frame_gt = {}
        segments = {}
        for hashed_id in sorted(spans):
            duration = float(duration_map[hashed_id])
            if not np.isfinite(duration) or duration <= 0:
                raise RuntimeError("invalid frozen media duration")
            pairs = [(float(x["start"]), float(x["end"])) for x in spans[hashed_id]]
            if any(not (np.isfinite(s) and np.isfinite(e) and 0 <= s < e <= duration + 1.0) for s, e in pairs):
                raise RuntimeError("temporal GT outside frozen media support")
            clipped = [(max(0.0, s), min(duration, e)) for s, e in pairs]
            frame_gt[hashed_id] = rasterize_1hz(duration, clipped).tolist()
            segments[hashed_id] = [list(x) for x in clipped]
        return {"frame_gt": frame_gt, "segments": segments}

    return decrypt


def make_test_evaluator(key_path: Path, duration_manifest: Path, signing_key: bytes) -> SealedEvaluator:
    if not signing_key:
        raise ValueError("steward signing key required")
    return SealedEvaluator(steward_decryptor(key_path, duration_manifest), signing_key, steward_mode=True)

