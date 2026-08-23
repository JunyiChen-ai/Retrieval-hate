from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .common import ContactLedger, atomic_json, canonical_bytes, sha256_file
import hashlib

COMPLETENESS = tuple(f"C{i}" for i in range(1, 16))
CANDIDATE_KEYS = ("macro_f1", "macroF1", "auroc", "delta_star", "prediction",
                  "candidate_metric", "log_loss_reduction")


def load_ledger(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"schema": "cvoi-completeness/1",
                "gates": {key: {"status": "PENDING", "evidence": []}
                          for key in COMPLETENESS}}
    obj = json.loads(path.read_text())
    if set(obj.get("gates", {})) != set(COMPLETENESS):
        raise RuntimeError("HALT_COMPLETENESS_SCHEMA")
    return obj


def assert_metric_locked(completeness_path: Path, frozen_config: Path | None = None) -> None:
    ledger = load_ledger(completeness_path)
    pending = [key for key in COMPLETENESS
               if ledger["gates"][key].get("status") != "PASS"]
    if pending:
        raise RuntimeError("HALT_METRIC_LOCKED:" + ",".join(pending))
    if frozen_config is None or not frozen_config.exists():
        raise RuntimeError("HALT_METRIC_LOCKED:NO_FROZEN_CONFIG")
    manifest=json.loads(frozen_config.read_text())
    claimed=manifest.get("payload_sha256")
    if not claimed:raise RuntimeError("HALT_METRIC_LOCKED:NO_PAYLOAD_HASH")
    payload={k:v for k,v in manifest.items() if k!="payload_sha256"}
    actual=hashlib.sha256(canonical_bytes(payload)).hexdigest()
    if actual!=claimed:raise RuntimeError("HALT_CONFIG_HASH_MISMATCH")
    for path,expected in manifest.get("source_file_sha256",{}).items():
        p=Path(path)
        if not p.exists() or sha256_file(p)!=expected:raise RuntimeError("HALT_INPUT_HASH_MISMATCH:"+path)
    comp_expected=manifest.get("completeness_sha256")
    if comp_expected!=sha256_file(completeness_path):raise RuntimeError("HALT_COMPLETENESS_HASH_MISMATCH")


def publish_ledger(path: Path, ledger: dict[str, Any]) -> None:
    for key, row in ledger["gates"].items():
        if row.get("status") == "PASS":
            for evidence in row.get("evidence", []):
                p = Path(evidence["path"])
                if not p.exists() or sha256_file(p) != evidence["sha256"]:
                    raise RuntimeError(f"HALT_BAD_COMPLETENESS_EVIDENCE:{key}:{p}")
    atomic_json(path, ledger)


def assert_no_candidate_payload(value: Any) -> None:
    text = json.dumps(value, sort_keys=True)
    if any(key in text for key in CANDIDATE_KEYS):
        raise RuntimeError("HALT_PREMETRIC_CANDIDATE_PAYLOAD")
