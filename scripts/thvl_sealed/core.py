from __future__ import annotations

import hashlib
import json

from scripts.dehate_sealed.core import (
    SealedEvaluator,
    canonical_json,
    frame_metrics,
    paired_video_cluster_bootstrap,
    rasterize_1hz,
    sign_freeze_manifest,
    temporal_ap,
)

FIXED_REMOTE_IDENTITY = {
    "repository": "THVL/THVL-Bench",
    "revision": "5ea20ec4074dea9d3419e88fea944313ab25818d",
    "annotation_path": "THVL-Bench.csv",
    "annotation_bytes": 71604,
    "annotation_sha256": "2ba5127eb05bee6e614ff4e6da511422eb4d2bac830281f55edb860daeedf5a7",
    "annotation_etag": "ffb9cf062d86a3eaf001fd0f70d0e742f4bcce38",
    "readme_sha256": "858b271afbf95abf99d214102cf7c87a1409eac40665bbfc0fb7682290faee19",
}

SPLIT_SALT = "THVL-Bench-selfsealed-v1-2026-08-29"


def verify_remote_identity(observed: dict) -> dict:
    if set(observed) != set(FIXED_REMOTE_IDENTITY):
        raise ValueError("THVL remote identity keys must be exact")
    mismatches = {
        key: (FIXED_REMOTE_IDENTITY[key], observed[key])
        for key in FIXED_REMOTE_IDENTITY
        if observed[key] != FIXED_REMOTE_IDENTITY[key]
    }
    if mismatches:
        raise RuntimeError(f"THVL remote identity mismatch: {mismatches}")
    return {
        "verified": True,
        "identity_sha256": hashlib.sha256(canonical_json(observed)).hexdigest(),
    }


def group_key(canonical_ids: list[str]) -> str:
    if not canonical_ids or len(canonical_ids) != len(set(canonical_ids)):
        raise ValueError("group canonical IDs must be nonempty and unique")
    return "\n".join(sorted(canonical_ids))


def assign_group(canonical_ids: list[str]) -> tuple[str, float]:
    key = group_key(canonical_ids)
    digest = hashlib.sha256((SPLIT_SALT + "\n" + key).encode("utf-8")).digest()
    u = int.from_bytes(digest[:8], "big") / 2**64
    split = "train" if u < 0.70 else ("validation" if u < 0.80 else "test")
    return split, u


def identity_sha256() -> str:
    return hashlib.sha256(json.dumps(FIXED_REMOTE_IDENTITY, sort_keys=True).encode()).hexdigest()

