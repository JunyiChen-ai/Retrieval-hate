#!/usr/bin/env python
"""Smallest SLURM-only SSR B0 implementation sanity (no model training)."""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path("/data/jehc223/RGCL")
sys.path.insert(0, str(ROOT / "scripts/analysis"))
from ssr_common import (  # noqa: E402
    RELATION_SCHEMA, atomic_write_json, build_prompt, calls_to_record,
    canonicalize_order, forbidden_payload_keys, load_config, relation_family,
    resolve, sha256_file, strict_parse_relation,
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--run-id", required=True)
    args = ap.parse_args()
    if not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("sanity must run under SLURM")
    cfg = load_config(args.config, require_frozen=False)
    sample = {k: v[0] for k, v in RELATION_SCHEMA.items()}
    sample.update({
        "target_relation": "different", "proposition_relation": "unrelated",
        "stance_a": "endorse", "stance_b": "condemn",
        "stance_relation": "opposed", "mechanism_a": "slur",
        "mechanism_b": "slur", "mechanism_relation": "same",
        "topic_surface_relation": "different",
    })
    parsed, err = strict_parse_relation(json.dumps(sample))
    repaired = dict(sample)
    repaired["target_relation"] = " Same "
    repaired_parsed, _ = strict_parse_relation(json.dumps(repaired))
    calls = []
    for pv in ("P0", "P1"):
        for order in ("AB", "BA"):
            raw = dict(sample)
            if order == "BA":
                raw["stance_a"], raw["stance_b"] = raw["stance_b"], raw["stance_a"]
                raw["mechanism_a"], raw["mechanism_b"] = raw["mechanism_b"], raw["mechanism_a"]
            calls.append({"prompt_version": pv, "order": order, "parsed": raw,
                          "parse_error": None})
    record = calls_to_record("synthetic_pair", calls)
    family, predicate, rho = relation_family(record, 1, 1)
    payload = {
        "dataset": "MHC", "canonical_pair_id": "synthetic_pair",
        "prompt_version": "P0", "order": "AB", "video_a_id": "a",
        "video_b_id": "b", "evidence_a": "automatic evidence",
        "evidence_b": "automatic evidence",
        "frame_count_a": 4, "frame_count_b": 4,
        "user_prompt": build_prompt("P0", "automatic evidence", "automatic evidence"),
    }
    scripts = [
        ROOT / "scripts/analysis/ssr_common.py",
        ROOT / "scripts/analysis/ssr_make_folds.py",
        ROOT / "scripts/analysis/ssr_oof.py",
        ROOT / "scripts/analysis/ssr_mine_pairs.py",
        ROOT / "scripts/analysis/ssr_extract_relations.py",
    ]
    checks = {
        "slurm": True,
        "conda_hatevideo": os.environ.get("CONDA_DEFAULT_ENV") == "HateVideo",
        "config_hash_computed": len(cfg["computed_config_sha256"]) == 64,
        "config_frozen": cfg.get("config_sha256") == cfg["computed_config_sha256"],
        "strict_parse": err is None and parsed == sample,
        "strict_parse_rejects_value_repair": repaired_parsed is None,
        "ba_canonicalization": canonicalize_order(
            canonicalize_order(sample, "BA"), "BA") == sample,
        "four_call_record": record["status"] == "relation" and
                            record["complete_four_calls"],
        "label_applied_after_record": family == "MI" and rho == 1.0 and
                                      predicate is not None,
        "payload_forbidden_keys": forbidden_payload_keys(payload),
        "segment_schema_fields": [k for k in RELATION_SCHEMA
                                  if "segment" in k or "span" in k or "local" in k],
        "source_scripts": {str(p.relative_to(ROOT)): sha256_file(p) for p in scripts},
    }
    status = all([
        checks["slurm"], checks["conda_hatevideo"], checks["config_hash_computed"],
        checks["config_frozen"],
        checks["strict_parse"], checks["ba_canonicalization"],
        checks["strict_parse_rejects_value_repair"],
        checks["four_call_record"], checks["label_applied_after_record"],
        not checks["payload_forbidden_keys"], not checks["segment_schema_fields"],
    ])
    out = (resolve(cfg, "artifacts") / "sanity" /
           "{}_{}.json".format(args.run_id, os.environ["SLURM_JOB_ID"]))
    report = {
        "run_id": args.run_id, "status": "GO" if status else "FAIL",
        "required_config_sha256": cfg["computed_config_sha256"],
        "checks": checks, "only_gold": "video_level_binary_label",
        "segment_gold_exists": False, "slurm_job_id": os.environ["SLURM_JOB_ID"],
    }
    atomic_write_json(out, report)
    print(json.dumps(report, ensure_ascii=False, sort_keys=True))
    if not status:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
