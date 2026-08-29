#!/usr/bin/env python3
"""Fail-closed readiness report for V9 HateMM train-only evidence."""
import argparse
import json
from pathlib import Path

from relation_v4.io import sha256
from relation_v2.protocol import frozen_splits
from relation_v9.train_timeline import hatemm_train_timeline


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", default="results/reproduction/relation_v9/manifests/hatemm_macil_vera.json")
    p.add_argument("--require-complete", action="store_true")
    a = p.parse_args()
    manifest = json.loads(Path(a.manifest).read_text())
    splits = frozen_splits("hatemm"); ids, lengths, timeline = hatemm_train_timeline()
    train, forbidden = set(ids), set(splits["val"]) | set(splits["test"])
    if train & forbidden:
        raise RuntimeError("frozen HateMM split overlap")
    report = {"corpus": "hatemm", "expected_train_videos": len(train),
              "train_val_test_disjoint": True, "experts": [], "ready": True,
              "validation_substitution_allowed": False, "timeline": timeline}
    for expert in manifest["experts"]:
        paths = expert["train_scores"]
        paths = [paths] if isinstance(paths, str) else paths
        item = {"name": expert["name"], "score_key": expert["score_key"], "paths": []}
        for value in paths:
            path = Path(value)
            row = {"path": str(path.resolve()), "exists": path.is_file()}
            if not path.is_file():
                row["gap"] = "missing legal same-corpus train dense output"
                report["ready"] = False
            else:
                records = {}
                for line in path.read_text().splitlines():
                    record = json.loads(line)
                    if record.get("video_id") in records: raise RuntimeError("duplicate train score ID")
                    records[record["video_id"]] = record
                observed = set(records)
                row.update(exact_train_ids=observed == train,
                           forbidden_ids_present=bool(observed & forbidden),
                           sha256=sha256(path))
                valid = observed == train and not (observed & forbidden)
                for vid in train & observed:
                    values = records[vid].get(expert["score_key"], [])
                    valid &= (len(values) == lengths[vid]
                              and all(__import__("math").isfinite(float(x)) for x in values))
                row["length_and_finite"] = bool(valid)
                evidence_path = path.parent / "evidence_manifest.json"
                row["evidence_manifest"] = str(evidence_path.resolve())
                row["evidence_manifest_exists"] = evidence_path.is_file()
                if evidence_path.is_file():
                    evidence = json.loads(evidence_path.read_text())
                    provenance_valid = (evidence.get("corpus") == "hatemm"
                                        and evidence.get("split") == "train"
                                        and evidence.get("scores_sha256") == sha256(path)
                                        and evidence.get("val_or_test_scores_used_as_train") is False)
                else:
                    provenance_valid = False
                row["same_corpus_train_provenance_valid"] = provenance_valid
                valid &= provenance_valid
                if not valid:
                    report["ready"] = False
            item["paths"].append(row)
        report["experts"].append(item)
    print(json.dumps(report, indent=2))
    if a.require_complete and not report["ready"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
