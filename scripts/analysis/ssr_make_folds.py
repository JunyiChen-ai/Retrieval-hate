#!/usr/bin/env python
"""Freeze SSR config/schema/prompts and strict train-only 5-fold maps."""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path

import torch
from sklearn.model_selection import StratifiedKFold

ROOT = Path("/data/jehc223/RGCL")
sys.path.insert(0, str(ROOT / "scripts/analysis"))
from ssr_common import (  # noqa: E402
    PROMPT_P0, PROMPT_P1, RELATION_SCHEMA, SYSTEM_PROMPT, atomic_write_json,
    canonical_json, load_config, read_jsonl, resolve, sha256_file, sha256_obj,
)


def train_gt_path(gt_root, dataset):
    path = gt_root / dataset / "train.jsonl"
    if not path.exists():
        raise FileNotFoundError(path)
    return path


def id_only_split_path(gt_root, dataset, split):
    name = {"dev": "val.jsonl", "test": "test.jsonl"}[split]
    path = gt_root / dataset / name
    if not path.exists():
        raise FileNotFoundError(path)
    return path


def read_id_only_split(path):
    # Deliberately extract only the JSON string assigned to `id`.  This B0
    # path never json.loads the dev/test record and therefore never reads or
    # materializes its label/text fields.
    pattern = re.compile(r'^\s*\{\s*"id"\s*:\s*("(?:[^"\\]|\\.)*")')
    ids = []
    with open(path, encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            if not line.strip():
                continue
            match = pattern.match(line)
            if match is None:
                raise ValueError("cannot extract ID-only field {}:{}".format(path, lineno))
            ids.append(str(json.loads(match.group(1))))
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate IDs in {}".format(path))
    return ids


def flatten_ids(cache):
    ids = cache["ids"]
    if ids and isinstance(ids[0], (list, tuple)):
        return [str(x) for batch in ids for x in batch]
    return [str(x) for x in ids]


def load_split_records(path):
    rows = read_jsonl(path)
    out = []
    for row in rows:
        if "id" not in row or "label" not in row:
            raise ValueError("missing id/label in {}".format(path))
        label = int(row["label"])
        if label not in (0, 1):
            raise ValueError("non-binary video label in {}: {}".format(path, label))
        out.append({"id": str(row["id"]), "label": label})
    ids = [x["id"] for x in out]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate IDs in {}".format(path))
    return out


def make_one(cfg, dataset):
    gt_root = resolve(cfg, "gt")
    clip_root = resolve(cfg, "clip")
    model = cfg["comparator"]["model"]
    gt_train_path = train_gt_path(gt_root, dataset)
    train_rows = load_split_records(gt_train_path)
    id_paths = {s: id_only_split_path(gt_root, dataset, s) for s in ("dev", "test")}
    split_id_lists = {"train": [x["id"] for x in train_rows]}
    split_id_lists.update({s: read_id_only_split(p) for s, p in id_paths.items()})
    ids = {s: set(v) for s, v in split_id_lists.items()}
    if {x["id"] for x in train_rows} != ids["train"]:
        raise RuntimeError("train GT and ID-only split disagree for {}".format(dataset))
    overlaps = {
        "train_dev": sorted(ids["train"] & ids["dev"]),
        "train_test": sorted(ids["train"] & ids["test"]),
        "dev_test": sorted(ids["dev"] & ids["test"]),
    }
    if any(overlaps.values()):
        raise RuntimeError("split leakage for {}: {}".format(dataset, overlaps))

    # B0 consumes only train labels/cache. Dev/test are represented exclusively
    # by label-free source ID lists for disjointness assertions.
    cache_paths = {"train": clip_root / dataset / "train_{}.pt".format(model)}
    cache_checks = {}
    for split, path in cache_paths.items():
        cache = torch.load(path, map_location="cpu")
        cache_ids = flatten_ids(cache)
        cache_set = set(cache_ids)
        if cache_set != ids[split] or len(cache_ids) != len(cache_set):
            raise RuntimeError(
                "GT/cache ID mismatch {} {}: gt={} cache={} dup_cache={}".format(
                    dataset, split, len(ids[split]), len(cache_set),
                    len(cache_ids) - len(cache_set)))
        labels = torch.as_tensor(cache["labels"]).reshape(-1).tolist()
        by_gt = {x["id"]: x["label"] for x in train_rows}
        if len(labels) != len(cache_ids):
            raise RuntimeError("cache label count mismatch {} {}".format(dataset, split))
        bad = [vid for vid, lab in zip(cache_ids, labels)
               if int(lab) != int(by_gt[vid])]
        if bad:
            raise RuntimeError("GT/cache video-label mismatch: {}".format(bad[:10]))
        cache_checks[split] = {
            "path": str(path.relative_to(ROOT)),
            "sha256": sha256_file(path),
            "n": len(cache_ids),
            "id_sha256": sha256_obj(sorted(cache_ids)),
            "label_sha256": sha256_obj([[x, by_gt[x]] for x in sorted(cache_ids)]),
        }

    subclip_path = clip_root / dataset / "train_subclipK{}_{}.pt".format(
        cfg["comparator"]["num_subclips"], model)
    if not subclip_path.exists():
        raise FileNotFoundError(subclip_path)

    ordered = sorted(train_rows, key=lambda x: x["id"])
    ordered_ids = [x["id"] for x in ordered]
    labels = [x["label"] for x in ordered]
    fold_cfg = cfg["folds"]
    skf = StratifiedKFold(
        n_splits=int(fold_cfg["n_splits"]),
        shuffle=bool(fold_cfg["shuffle"]),
        random_state=int(fold_cfg["random_state"]),
    )
    assignment = {}
    for fold, (_, query_idx) in enumerate(skf.split(ordered_ids, labels)):
        for idx in query_idx:
            assignment[ordered_ids[int(idx)]] = int(fold)
    if set(assignment) != ids["train"]:
        raise AssertionError("fold assignment is incomplete")

    records = [{"id": x["id"], "label": x["label"],
                "fold": assignment[x["id"]]} for x in ordered]
    fold_counts = {}
    for fold in range(int(fold_cfg["n_splits"])):
        rr = [x for x in records if x["fold"] == fold]
        fold_counts[str(fold)] = {
            "n": len(rr), "label_0": sum(x["label"] == 0 for x in rr),
            "label_1": sum(x["label"] == 1 for x in rr),
            "ids_sha256": sha256_obj([x["id"] for x in rr]),
        }

    artifact = {
        "schema_version": 1,
        "dataset": dataset,
        "only_gold_supervision": "video_level_binary_label",
        "segment_gold_exists": False,
        "config_sha256": cfg["computed_config_sha256"],
        "fold_spec": fold_cfg,
        "records": records,
        "id_to_fold_sha256": sha256_obj(
            [[x["id"], x["fold"]] for x in records]),
        "fold_counts": fold_counts,
        "split_assertions": {
            "pairwise_disjoint": True,
            "overlaps": overlaps,
            "split_id_sha256": {s: sha256_obj(sorted(ids[s])) for s in ids},
            "train_gt": {"path": str(gt_train_path.relative_to(ROOT)),
                         "sha256": sha256_file(gt_train_path), "n": len(train_rows)},
            "id_only_splits": {s: {"path": str(id_paths[s].relative_to(ROOT)),
                                    "sha256": sha256_file(id_paths[s]),
                                    "n": len(split_id_lists[s])}
                               for s in id_paths},
            "clip_cache": cache_checks,
            "subclip_cache": {"path": str(subclip_path.relative_to(ROOT)),
                              "sha256": sha256_file(subclip_path)},
        },
    }
    out = resolve(cfg, "artifacts") / "folds" / "{}.json".format(dataset)
    atomic_write_json(out, artifact)
    artifact["artifact_path"] = str(out.relative_to(ROOT))
    artifact["artifact_sha256"] = sha256_file(out)
    return artifact


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--dataset", default="all", choices=["all", "MHC", "MHC_zh"])
    ap.add_argument("--allow-unfrozen", action="store_true")
    args = ap.parse_args()
    cfg = load_config(args.config, require_frozen=not args.allow_unfrozen)
    if cfg["supervision"] != {
            "only_gold": "video_level_binary_label",
            "segment_gold_exists": False,
            "segment_gold_forbidden": True,
            "mllm_relations": "weak_privileged_train_only_pseudo_signals"}:
        raise RuntimeError("supervision contract changed")

    datasets = list(cfg["datasets"]) if args.dataset == "all" else [args.dataset]
    results = [make_one(cfg, ds) for ds in datasets]
    prompt_manifest = {
        "system_sha256": sha256_obj(SYSTEM_PROMPT),
        "P0_template_sha256": sha256_obj(PROMPT_P0),
        "P1_template_sha256": sha256_obj(PROMPT_P1),
        "schema": RELATION_SCHEMA,
        "schema_sha256": sha256_obj(RELATION_SCHEMA),
        "input_builder_sha256": sha256_file(ROOT / "scripts/analysis/ssr_common.py"),
    }
    out_root = resolve(cfg, "artifacts")
    atomic_write_json(out_root / "freeze_manifest.json", {
        "run_id": os.environ.get("RUN_ID", "SSR-B0-FREEZE-v1"),
        "status": "GO" if cfg.get("config_sha256") == cfg["computed_config_sha256"] else "NEEDS_CONFIG_HASH",
        "stored_config_sha256": cfg.get("config_sha256"),
        "required_config_sha256": cfg["computed_config_sha256"],
        "config_canonical_sha256_excluding_hash_field": cfg["computed_config_sha256"],
        "supervision_contract": cfg["supervision"],
        "prompts_and_schema": prompt_manifest,
        "datasets": [{k: r[k] for k in (
            "dataset", "artifact_path", "artifact_sha256", "id_to_fold_sha256",
            "fold_counts", "split_assertions")} for r in results],
        "environment": {
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
            "conda_default_env": os.environ.get("CONDA_DEFAULT_ENV"),
        },
    })
    print(canonical_json({"status": "ok", "datasets": datasets,
                          "required_config_sha256": cfg["computed_config_sha256"]}))


if __name__ == "__main__":
    main()
