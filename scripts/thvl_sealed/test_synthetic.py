#!/usr/bin/env python3
import json
import tempfile
from pathlib import Path

from cryptography.fernet import Fernet

from scripts.thvl_sealed.core import FIXED_REMOTE_IDENTITY, assign_group, verify_remote_identity
from scripts.thvl_sealed.evaluator import steward_decryptor


def main():
    assert verify_remote_identity(dict(FIXED_REMOTE_IDENTITY))["verified"]
    tampered = dict(FIXED_REMOTE_IDENTITY)
    tampered["revision"] = "0" * 40
    try:
        verify_remote_identity(tampered)
    except RuntimeError:
        pass
    else:
        raise AssertionError("remote tamper accepted")
    split, u = assign_group(["youtube:abcdefghijk"])
    assert split in {"train", "validation", "test"} and 0 <= u < 1
    assert assign_group(["youtube:abcdefghijk", "bilibili:BV1example"])[1] == assign_group(["bilibili:BV1example", "youtube:abcdefghijk"])[1]
    with tempfile.TemporaryDirectory() as raw:
        root = Path(raw)
        key = Fernet.generate_key()
        (root / "key").write_bytes(key)
        labels = {"dataset": "THVL-Bench", "split": "test", "temporal_gt": {"opaque": [{"start": 1.2, "end": 2.1}]}}
        (root / "labels.fernet").write_bytes(Fernet(key).encrypt(json.dumps(labels).encode()))
        (root / "duration.json").write_text(json.dumps({"schema_version": 1, "duration_seconds_by_hashed_id": {"opaque": 3.0}}))
        decoded = steward_decryptor(root / "key", root / "duration.json")(root / "labels.fernet")
        assert decoded["frame_gt"]["opaque"] == [0, 1, 1]
    print("PASS")


if __name__ == "__main__":
    main()
