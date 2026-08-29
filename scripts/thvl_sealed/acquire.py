#!/usr/bin/env python3
"""Steward-only THVL acquisition. Never prints IDs or per-video labels."""
from __future__ import annotations

import argparse
import ast
import base64
import csv
import hashlib
import hmac
import json
import os
import re
from datetime import datetime, timezone
from collections import Counter, defaultdict
from pathlib import Path

from cryptography.fernet import Fernet

from scripts.dehate_sealed.core import canonical_json
from scripts.thvl_sealed.core import FIXED_REMOTE_IDENTITY, SPLIT_SALT, assign_group, verify_remote_identity

CATEGORIES = (
    "Information Harm", "Verbal Abuse", "Hate", "Bias", "Addiction Harm",
    "Sexual Harm", "Physical Harm", "Violence", "Blood/Gore",
    "Criminal Activity", "Danger",
)
TARGET_INDICES = (1, 2, 3)
MEDIA_SUFFIXES = {".mp4", ".mov"}


def canonical(raw: str) -> str:
    raw = raw.strip()
    if raw.startswith("yt_"):
        return "youtube:" + raw[3:]
    if raw.startswith("BV"):
        return "bilibili:" + raw
    raise ValueError("unsupported platform ID")


def source_group(canonical_id: str) -> str:
    platform, source = canonical_id.split(":", 1)
    # Released media contains numbered excerpts from the same source. Group them
    # before splitting, using only the platform ID syntax and never annotations.
    if platform == "youtube" and re.fullmatch(r".{11}_[0-9]+", source):
        source = source[:11]
    elif platform == "bilibili" and re.search(r"_[0-9]+$", source):
        source = re.sub(r"_[0-9]+$", "", source)
    return platform + ":" + source


def parse_time(value: str) -> float:
    parts = value.strip().split(":")
    if len(parts) not in (2, 3):
        raise ValueError("invalid timestamp")
    nums = [float(x) for x in parts]
    return nums[-1] + 60 * nums[-2] + (3600 * nums[-3] if len(nums) == 3 else 0)


def load_rows(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            anns = ast.literal_eval(row["segment-level annotation"])
            times = ast.literal_eval(row["segment-level timestamp"])
            mods = ast.literal_eval(row["contributing modalities"])
            if not (len(anns) == len(times) == len(mods)):
                raise RuntimeError("annotation/timestamp/modality length mismatch")
            segments = []
            for ann, span, mod in zip(anns, times, mods):
                if len(ann) != 11 or len(mod) != 3 or len(span) != 2:
                    raise RuntimeError("unexpected THVL annotation schema")
                if any(x not in (0, 1) for x in ann + mod):
                    raise RuntimeError("non-binary THVL annotation")
                start, end = map(parse_time, span)
                if end <= start:
                    raise RuntimeError("invalid THVL span")
                relevant = int(any(ann[i] for i in TARGET_INDICES))
                segments.append({"start": start, "end": end, "relevant": relevant, "labels": ann, "modalities": mod})
            cid = canonical(row["videoID"])
            rows.append({"raw_id": row["videoID"], "canonical_id": cid, "source_group": source_group(cid), "segments": segments})
    if len(rows) != 450 or len({r["canonical_id"] for r in rows}) != len(rows):
        raise RuntimeError("THVL row/ID invariant failed")
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--private-dir", type=Path, required=True)
    parser.add_argument("--public-dir", type=Path, required=True)
    args = parser.parse_args()
    private = args.private_dir.resolve()
    public = args.public_dir.resolve()
    private.mkdir(parents=True, exist_ok=True, mode=0o700)
    public.mkdir(parents=True, exist_ok=True)
    csv_path = private / "THVL-Bench.csv"
    api_path = private / "hf_api_metadata.json"
    if hashlib.sha256(csv_path.read_bytes()).hexdigest() != FIXED_REMOTE_IDENTITY["annotation_sha256"]:
        raise RuntimeError("annotation SHA-256 mismatch")
    api = json.loads(api_path.read_text())
    observed = dict(FIXED_REMOTE_IDENTITY)
    observed["revision"] = api["sha"]
    verify_remote_identity(observed)

    ledger_path = private / "access_ledger.jsonl"
    ledger_event = {
        "utc": datetime.now(timezone.utc).isoformat(),
        "actor_role": "independent_data_steward",
        "event": "ANNOTATION_ACQUIRE_PARSE_AND_RESEAL",
        "revision": FIXED_REMOTE_IDENTITY["revision"],
        "annotation_sha256": FIXED_REMOTE_IDENTITY["annotation_sha256"],
        "outputs": "opaque manifests plus encrypted validation/test temporal GT",
        "per_video_labels_disclosed": False,
    }
    with ledger_path.open("a", encoding="utf-8") as ledger:
        ledger.write(json.dumps(ledger_event, sort_keys=True) + "\n")
        ledger.flush()
        os.fsync(ledger.fileno())
    ledger_path.chmod(0o600)

    media = [x["rfilename"] for x in api["siblings"] if Path(x["rfilename"]).suffix.lower() in MEDIA_SUFFIXES]
    media_by_stem = defaultdict(list)
    for name in media:
        media_by_stem[Path(name).stem].append(name)
    rows = load_rows(csv_path)
    groups = defaultdict(list)
    for row in rows:
        groups[row["source_group"]].append(row["canonical_id"])
    group_assignments = {group: assign_group(ids)[0] for group, ids in groups.items()}

    hmac_path = private / "manifest_hmac.key"
    if not hmac_path.exists():
        hmac_path.write_bytes(os.urandom(32))
        hmac_path.chmod(0o600)
    hkey = hmac_path.read_bytes()
    fernet_path = private / "test_labels.fernet.key"
    if not fernet_path.exists():
        fernet_path.write_bytes(Fernet.generate_key())
        fernet_path.chmod(0o600)
    fernet = Fernet(fernet_path.read_bytes())

    def opaque(value: str) -> str:
        return hmac.new(hkey, value.encode(), hashlib.sha256).hexdigest()

    manifests = {name: [] for name in ("train", "validation", "test")}
    private_rows = []
    sealed_labels = {"validation": {}, "test": {}}
    split_counts = Counter()
    related_video_counts = Counter()
    related_segment_counts = Counter()
    platform_counts = Counter()
    repo_matches = Counter()
    for row in rows:
        split = group_assignments[row["source_group"]]
        split_counts[split] += 1
        platform = row["canonical_id"].split(":", 1)[0]
        platform_counts[platform] += 1
        matches = media_by_stem.get(row["raw_id"], [])
        repo_status = "unique_path" if len(matches) == 1 else ("missing_path" if not matches else "ambiguous_path")
        repo_matches[repo_status] += 1
        hid = opaque(row["canonical_id"])
        weak = int(any(s["relevant"] for s in row["segments"]))
        manifests[split].append({
            "hashed_id": hid,
            "duplicate_group_hash": opaque("group:" + row["source_group"]),
            "media_repository_status": repo_status,
            **({"weak_video_label": weak} if split == "train" else {}),
        })
        private_rows.append({
            "raw_id": row["raw_id"], "canonical_id": row["canonical_id"], "hashed_id": hid,
            "source_group": row["source_group"], "split": split, "repository_paths": matches,
        })
        if weak:
            related_video_counts[split] += 1
        related_segment_counts[split] += sum(s["relevant"] for s in row["segments"])
        if split in sealed_labels:
            sealed_labels[split][hid] = [{"start": s["start"], "end": s["end"]} for s in row["segments"] if s["relevant"]]

    for split, records in manifests.items():
        records.sort(key=lambda x: x["hashed_id"])
        payload = {
            "schema_version": 1, "dataset": "THVL-Bench", "revision": FIXED_REMOTE_IDENTITY["revision"],
            "split_protocol": "label-free source-group SHA256 70/10/20", "split_salt": SPLIT_SALT,
            "split": split, "records": records,
        }
        (public / f"{split}_opaque_manifest.json").write_bytes(canonical_json(payload) + b"\n")

    private_payload = {"remote_identity": FIXED_REMOTE_IDENTITY, "records": private_rows}
    (private / "steward_id_media_map.json").write_bytes(canonical_json(private_payload) + b"\n")
    (private / "steward_id_media_map.json").chmod(0o600)
    for split, labels in sealed_labels.items():
        encrypted = fernet.encrypt(canonical_json({"dataset": "THVL-Bench", "split": split, "temporal_gt": labels}))
        target = private / f"{split}_temporal_gt.fernet"
        target.write_bytes(encrypted)
        target.chmod(0o600)

    summary = {
        "dataset": "THVL-Bench", "revision": FIXED_REMOTE_IDENTITY["revision"],
        "license_status": "AMBIGUOUS: YAML/API say CC-BY-4.0; README prose says CC-BY-NC-4.0; use stricter non-commercial interpretation pending author clarification",
        "n_videos": len(rows), "n_source_groups": len(groups), "n_segments_all_harms": sum(len(r["segments"]) for r in rows),
        "split_video_counts": dict(split_counts),
        "target_scope": "Verbal Abuse OR Hate OR Bias",
        "target_related_video_counts": dict(related_video_counts),
        "target_related_segment_counts": dict(related_segment_counts),
        "platform_video_counts": dict(platform_counts),
        "hf_media_file_count": len(media), "annotation_to_hf_media_path_status": dict(repo_matches),
        "bulk_media_downloaded": False,
        "test_labels_encrypted": True,
        "validation_labels_encrypted": True,
        "public_manifest_sha256": {split: hashlib.sha256((public / f"{split}_opaque_manifest.json").read_bytes()).hexdigest() for split in manifests},
    }
    (public / "aggregate_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
