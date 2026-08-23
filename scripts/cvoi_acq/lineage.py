from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from .common import ContactLedger, ROOT, atomic_write, sorted_id_bytes

AUDIT = ROOT / "artifacts/tera_gate0/tera-gate0-20260807T000625Z-7ba80eaf/gate_c_audit.jsonl"
SAMPLE = ROOT / "artifacts/tera_gate0/tera-gate0-20260807T000625Z-7ba80eaf/gate_c_sample.json"
STRICT_HASH = "058aac5fa3bc8360429fe99331fdbfbb4dc3c025de14740827c86aaeddf1f317"
BROAD_HASH = "9d268f469bb0e8731d2681b9ceaab5fa3db6db14cc03ba4773f608d690cafe74"


def materialize(out_dir: Path) -> dict[str, object]:
    ledger = ContactLedger()
    ledger.register(AUDIT, "diagnostic_gate_c_audit")
    rows = [json.loads(line) for line in AUDIT.open() if line.strip()]
    ledger.register(SAMPLE, "diagnostic_gate_c_sample")
    sample = json.loads(SAMPLE.read_text())
    final = {r["video_id"]: r for r in rows if r["coder_id"].endswith("c1")}
    final.update({r["video_id"]: r for r in rows if r["coder_id"].endswith("adj")})
    fn_ids = set(sample["audit_fn"])
    strict, broad = [], []
    for vid in sorted(fn_ids):
        req = set(final[vid]["required_modalities"])
        if "on_screen_text" in req and "speech" not in req:
            broad.append(vid)
            if "transcript" not in req:
                strict.append(vid)
    payloads = {
        "strict_ocr_no_speech.ids": (strict, STRICT_HASH),
        "ocr_no_speech_flag.ids": (broad, BROAD_HASH),
    }
    result = {"schema": "cvoi-lineage/1", "source_audit": str(AUDIT.relative_to(ROOT)),
              "source_sample": str(SAMPLE.relative_to(ROOT)), "contact": ledger.snapshot(),
              "cohorts": {}}
    for name, (ids, expected) in payloads.items():
        data = sorted_id_bytes(ids)
        actual = hashlib.sha256(data).hexdigest()
        if actual != expected:
            raise RuntimeError(f"HALT_LINEAGE_HASH:{name}:{actual}")
        atomic_write(out_dir / name, data)
        result["cohorts"][name] = {"count": len(ids), "sha256": actual,
                                    "definition": ("on_screen_text required; speech and transcript absent"
                                                   if len(ids) == 22 else
                                                   "on_screen_text required; speech absent; transcript allowed")}
    return result


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", type=Path, required=True)
    args = ap.parse_args()
    print(json.dumps(materialize(args.out_dir), sort_keys=True))


if __name__ == "__main__":
    main()
