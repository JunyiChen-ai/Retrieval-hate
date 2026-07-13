#!/usr/bin/env python
"""M1 Stage 1 - deterministic label-blind evidence-pack builder.

For each TRAIN video, deterministically assemble the evidence pack the restricted
MLLM will read: 16 uniform full-video frames (spec only here; pixels are decoded by
the producer), the video title, and the Whisper ASR transcript (deterministically
truncated).  Each pack is content-addressed by evidence_pack_sha256 over the pack
SPEC, and packs are deduplicated on that hash (U_D = unique packs; base calls =
4 * U_D).

Gold isolation (the reason this is a separate, auditable stage): the pack contains
and reads NO label/split/seed/neighbor/prediction/margin/correctness signal.  The
title loader reads only id + text; the ASR loader reads only id + transcript chunks;
neither touches the per-line label field that also lives in those files.  The
train-ID list is split membership (allowlist), not gold.  A zero-gold grep
self-attestation over this file is part of the freeze doc.

This module is imported by the cache producer and also runs standalone to emit an
auditable evidence_packs.jsonl (no MLLM, no GPU, no label).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path("/data/jehc223/RGCL")
sys.path.insert(0, str(ROOT / "scripts/analysis"))

from lb_scgp_global_r2_m1_cache_v2_common import (  # noqa: E402
    EXPECTED_TRAIN_N,
    FRAME_RULE,
    MAX_ASR_CHARS,
    MAX_TITLE_CHARS,
    NUM_FRAMES,
    TrainEvidenceAccessLedger,
    canonical_json,
    canonical_root_path,
    canonical_video_path,
    sha256_bytes,
    sha256_file,
    sha256_obj,
)

GT_TRAIN = "data/gt/{dataset}/train.jsonl"
ASR_TRAIN = "data/ASR/{dataset}/train_asrK4_whisper-large-v3.jsonl"
VIDEO_DIR = "data/video/{dataset}/All"


def evidence_allowlist(dataset: str) -> set[str]:
    """The only train-evidence files the builder/producer may open (title + ASR)."""
    return {GT_TRAIN.format(dataset=dataset), ASR_TRAIN.format(dataset=dataset)}


def _truncate(text: str, limit: int) -> str:
    t = (text or "").strip()
    if len(t) > limit:
        t = t[:limit] + " ...[truncated]"
    return t


def load_titles(fs_path: Path) -> dict[str, str]:
    """Return {video_id: title}. Reads ONLY the id and free-text title fields.

    The gt line also carries a per-video annotation field; it is never referenced
    here, so no gold is read.
    """
    titles: dict[str, str] = {}
    with open(fs_path, encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            vid = str(obj["id"])
            titles[vid] = "" if obj.get("text") is None else str(obj["text"])
    return titles


def load_train_ids(fs_path: Path) -> list[str]:
    """Sorted train-ID allowlist (split membership). Reads ONLY the id field."""
    ids: list[str] = []
    with open(fs_path, encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            ids.append(str(obj["id"]))
    if len(ids) != len(set(ids)):
        raise RuntimeError("duplicate train IDs in gt train file")
    return sorted(ids)


def load_asr(fs_path: Path) -> dict[str, str]:
    """Return {video_id: transcript}. Reads ONLY id + the transcript chunk text.

    The ASR line also carries a per-video annotation field; it is never referenced
    here.  Transcript = chunk texts concatenated in file order (already timestamp
    ordered), falling back to window_text, then to empty.
    """
    asr: dict[str, str] = {}
    with open(fs_path, encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            vid = str(obj["id"])
            transcript = ""
            chunks = obj.get("chunks")
            if isinstance(chunks, list) and chunks:
                parts = []
                for chunk in chunks:
                    if isinstance(chunk, (list, tuple)) and len(chunk) >= 3 and chunk[2] is not None:
                        parts.append(str(chunk[2]).strip())
                    elif isinstance(chunk, dict) and chunk.get("text") is not None:
                        parts.append(str(chunk["text"]).strip())
                transcript = " ".join(p for p in parts if p)
            if not transcript:
                wt = obj.get("window_text")
                if isinstance(wt, list):
                    transcript = " ".join(str(s).strip() for s in wt if s)
                elif isinstance(wt, str):
                    transcript = wt.strip()
            asr[vid] = transcript
    return asr


def build_pack(video_id: str, dataset: str, title: str, transcript: str,
               video_relpath: str, video_sha256: str) -> dict[str, Any]:
    """Deterministic evidence pack for one video (NO label/split/seed/neighbor).

    evidence_pack_sha256 content-addresses the pack over its deterministic spec:
    id + truncated title + truncated transcript + frame-sampling spec + the video
    file identity (sha256).  Two videos collide (dedup) only if all of these match.
    """
    title_t = _truncate(title, MAX_TITLE_CHARS)
    asr_t = _truncate(transcript, MAX_ASR_CHARS)
    spec = {
        "video_id": video_id,
        "dataset": dataset,
        "title": title_t,
        "asr_transcript": asr_t,
        "num_frames": NUM_FRAMES,
        "frame_rule": FRAME_RULE,
        "video_relpath": video_relpath,
        "video_sha256": video_sha256,
    }
    pack = dict(spec)
    pack["evidence_pack_sha256"] = sha256_obj(spec)
    return pack


def build_dataset_packs(dataset: str, ledger: TrainEvidenceAccessLedger,
                        hash_videos: bool = True) -> dict[str, Any]:
    """Build the deterministic pack for every train video of `dataset`.

    Returns {packs: {video_id: pack}, order: [sorted video_ids],
    dedup: {evidence_pack_sha256: [video_ids]}, train_id_allowlist_sha256, counts}.
    Opens only allowlisted title/ASR files and the dataset's train-video dir.
    """
    gt_rel = GT_TRAIN.format(dataset=dataset)
    asr_rel = ASR_TRAIN.format(dataset=dataset)
    gt_fs = ledger.open_evidence(gt_rel, "train_title_source", dataset)
    asr_fs = ledger.open_evidence(asr_rel, "train_asr_source", dataset)

    order = load_train_ids(gt_fs)
    if len(order) != EXPECTED_TRAIN_N[dataset]:
        raise RuntimeError(f"{dataset} train_n drift: {len(order)} != {EXPECTED_TRAIN_N[dataset]}")
    titles = load_titles(gt_fs)
    asr = load_asr(asr_fs)

    train_id_allowlist_sha256 = sha256_bytes(("\n".join(order) + "\n").encode("utf-8"))
    video_root = VIDEO_DIR.format(dataset=dataset)

    packs: dict[str, Any] = {}
    dedup: dict[str, list[str]] = {}
    missing_video = 0
    for vid in order:
        video_rel = f"{video_root}/{vid}.mp4"
        # v2 fix: the mp4 is an in-repo symlink escaping the repo; canonical_video_path
        # contains on the symlink LOCATION and returns the in-repo path for OS-follow decode/
        # hash (the v1 canonical_root_path resolve() burned here). A containment violation
        # (not the legitimate external mp4 target) still raises fail-closed.
        video_fs = canonical_video_path(video_rel, dataset)
        if video_fs.exists() and hash_videos:
            ledger.note_video_read(video_rel, dataset)               # records the followed target
            video_sha = sha256_file(video_fs)                        # bytes via followed symlink
        elif video_fs.exists():
            video_sha = "not_hashed"
        else:
            missing_video += 1
            video_sha = "missing"
        pack = build_pack(vid, dataset, titles.get(vid, ""), asr.get(vid, ""), video_rel, video_sha)
        packs[vid] = pack
        dedup.setdefault(pack["evidence_pack_sha256"], []).append(vid)

    return {
        "dataset": dataset,
        "packs": packs,
        "order": order,
        "dedup": dedup,
        "unique_pack_count": len(dedup),
        "video_count": len(order),
        "missing_video_count": missing_video,
        "train_id_allowlist_sha256": train_id_allowlist_sha256,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="M1 evidence-pack builder (standalone audit emit; no MLLM/label).")
    parser.add_argument("--dataset", required=True, choices=list(EXPECTED_TRAIN_N.keys()))
    parser.add_argument("--out", required=True, help="in-repo evidence_packs.jsonl path")
    parser.add_argument("--no-hash-videos", action="store_true",
                        help="skip mp4 byte hashing (spec uses video_sha256='not_hashed')")
    args = parser.parse_args()

    ledger = TrainEvidenceAccessLedger(evidence_allowlist(args.dataset))
    result = build_dataset_packs(args.dataset, ledger, hash_videos=not args.no_hash_videos)
    out_fs, _ = canonical_root_path(args.out)
    out_fs.parent.mkdir(parents=True, exist_ok=True)
    with open(out_fs, "w", encoding="utf-8") as handle:
        for vid in result["order"]:
            handle.write(canonical_json(result["packs"][vid]) + "\n")
    summary = {
        "dataset": result["dataset"],
        "video_count": result["video_count"],
        "unique_pack_count": result["unique_pack_count"],
        "missing_video_count": result["missing_video_count"],
        "train_id_allowlist_sha256": result["train_id_allowlist_sha256"],
    }
    print(canonical_json(summary))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
