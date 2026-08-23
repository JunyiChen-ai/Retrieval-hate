from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Iterable

ROOT = Path(__file__).resolve().parents[2]
TEST_TOKENS = ("/test/", "test.jsonl", "test_seen", "/test_")


def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        while True:
            block = fh.read(chunk)
            if not block:
                return h.hexdigest()
            h.update(block)


def canonical_bytes(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":"),
                       ensure_ascii=False, allow_nan=False) + "\n").encode("utf-8")


def atomic_write(path: Path, payload: bytes, refuse_overwrite: bool = True) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if refuse_overwrite and path.exists():
        raise FileExistsError(f"refusing overwrite: {path}")
    fd, tmp = tempfile.mkstemp(prefix=f".{path.name}.", dir=str(path.parent))
    try:
        with os.fdopen(fd, "wb") as fh:
            fh.write(payload)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, path)
        dfd = os.open(path.parent, os.O_DIRECTORY)
        try:
            os.fsync(dfd)
        finally:
            os.close(dfd)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def atomic_json(path: Path, value: Any, refuse_overwrite: bool = True) -> None:
    atomic_write(path, canonical_bytes(value), refuse_overwrite)


class ContactLedger:
    def __init__(self) -> None:
        self.opened: list[dict[str, str]] = []
        self.test_contact_count = 0

    def register(self, path: Path, role: str) -> Path:
        resolved = path.resolve()
        normalized = "/" + str(resolved).lower().strip("/")
        if any(token in normalized for token in TEST_TOKENS):
            self.test_contact_count += 1
            raise RuntimeError(f"HALT_TEST_CONTACT: {resolved}")
        self.opened.append({"path": str(resolved), "role": role})
        return resolved

    def snapshot(self) -> dict[str, Any]:
        return {"test_contact_count": self.test_contact_count,
                "opened_paths": list(self.opened)}


def load_jsonl(path: Path, ledger: ContactLedger, role: str) -> list[dict[str, Any]]:
    ledger.register(path, role)
    with path.open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def sorted_id_bytes(ids: Iterable[str]) -> bytes:
    return ("\n".join(sorted(set(ids))) + "\n").encode("utf-8")

