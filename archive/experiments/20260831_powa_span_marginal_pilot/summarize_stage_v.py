#!/usr/bin/env python3
"""Verify all six artifacts and apply frozen span-marginal Stage-V gates."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from train import sha256


HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
CORPORA = ("hatemm", "hateclipseg")
ARMS = ("span_marginal", "singleton", "shuffled_span")
EXPECTED_ANCHORS = {
    "hatemm": REPO / (
        "results/reproduction/powa_macil/"
        "final_maskfix_finetune_hatemm_seed234_e5"
    ),
    "hateclipseg": REPO / (
        "runs/20260831_powa_starting_point/hcs_maskfix_seed234"
    ),
}
EXPECTED_ARGS = {
    "epochs": 5, "batch_size": 24, "lr": 0.0002,
    "weight_decay": 0.0001, "temperature": 0.5,
    "negative_dense_weight": 1.0, "residual_l2_weight": 0.01,
    "pooled_tolerance": 0.002, "seed": 234,
    "device": "cuda", "num_workers": 4,
    "limit_train_videos": 0, "limit_val_videos": 0,
}


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-root", required=True, type=Path)
    p.add_argument("--out", required=True, type=Path)
    return p


def review_verdict(path):
    match = re.search(r"^Verdict:\s*\*\*(PASS|BLOCK)\*\*", path.read_text(), re.M)
    return match.group(1) if match else None


def verify_snapshot(run):
    manifest = run / "source_snapshot.sha256"
    if not manifest.is_file():
        return False, None
    inventory = {}
    root = run / "source_snapshot"
    try:
        for line in manifest.read_text().splitlines():
            expected, raw = line.split(None, 1)
            path = Path(raw.strip())
            if not path.is_file() or sha256(path) != expected:
                return False, None
            inventory[str(path.relative_to(root))] = expected
    except (ValueError, OSError):
        return False, None
    return True, inventory


def current_source_path(name):
    group, relative = name.split("/", 1)
    roots = {
        "experiment": HERE,
        "shared": REPO / "src",
        "powa_macil": REPO / "scripts/reproduction_baselines/powa_macil",
        "macilsd": REPO / "scripts/reproduction_baselines/macilsd",
        "hate_common": REPO / "scripts/reproduction_baselines/hate_common",
    }
    if group in roots:
        return roots[group] / relative
    if name == "evaluator/eval_baseline_scores.py":
        return REPO / "scripts/reproduction_baselines/eval_baseline_scores.py"
    if name == "evaluator/frame_eval_common.py":
        return REPO / "scripts/duplex/frame_eval_common.py"
    raise KeyError(name)


def snapshot_matches_current(inventory):
    if not inventory:
        return False
    try:
        return all(current_source_path(name).is_file()
                   and sha256(current_source_path(name)) == expected
                   for name, expected in inventory.items())
    except (KeyError, OSError):
        return False


def evaluator_matches(meta, evaluator):
    validation = meta.get("selected_validation")
    if validation is None:
        return evaluator is None
    if evaluator is None:
        return False
    for branch, internal in validation["metrics"].items():
        official = evaluator["results"].get(branch)
        if official is None:
            return False
        values = (
            (internal["pooled_ap"], official["pr_auc"]),
            (internal["pooled_roc"], official["roc_auc"]),
            (internal["within_roc"], official["per_video"]["macro_auc"]),
        )
        if any(abs(float(a) - float(b)) > 1e-12 for a, b in values):
            return False
    return True


def load_record(root, corpus, arm, plan_hash, review_hash):
    run = root / f"{corpus}_{arm}_seed234"
    meta_path = run / "train_meta.json"
    completion_path = run / "completion.json"
    meta = json.loads(meta_path.read_text())
    completion = json.loads(completion_path.read_text())
    selected = meta.get("selected_epoch") is not None
    metrics_path = run / "metrics.json"
    scores_path = run / "val_scores.jsonl"
    evaluator = json.loads(metrics_path.read_text()) if metrics_path.is_file() else None
    config = json.loads((run / "config.json").read_text())
    snapshot_ok, inventory = verify_snapshot(run)
    frozen_args = all(meta.get("args", {}).get(key) == value
                      for key, value in EXPECTED_ARGS.items())
    checks = {
        "identity": meta.get("corpus") == corpus and meta.get("arm") == arm
        and meta.get("seed") == 234
        and meta.get("method") == "powa_context_quotient_span_marginal"
        and Path(meta.get("args", {}).get("out_dir", "")).resolve()
        == run.resolve(),
        "frozen_args": frozen_args and config == meta.get("args"),
        "anchor": Path(meta.get("anchor_checkpoint", "")).resolve()
        == EXPECTED_ANCHORS[corpus].resolve()
        and meta.get("anchor_model_sha256")
        == sha256(EXPECTED_ANCHORS[corpus] / "model.pth")
        and meta.get("anchor_train_meta_sha256")
        == sha256(EXPECTED_ANCHORS[corpus] / "train_meta.json"),
        "plan": meta.get("pilot_plan_sha256") == plan_hash,
        "corpus_only": meta.get("cross_corpus_training") is False,
        "no_test": meta.get("test_labels_used") is False,
        "powa_frozen": meta.get("powa_parameters_trainable") == 0,
        "snapshot": snapshot_ok,
        "current_source_matches_snapshot": snapshot_matches_current(inventory),
        "selected_artifacts": (
            selected and all((run / name).is_file() for name in (
                "residual_head.pth", "val_scores.jsonl", "metrics.json"
            ))
        ) or (
            not selected and all(not (run / name).exists() for name in (
                "residual_head.pth", "val_scores.jsonl", "metrics.json"
            ))
        ),
        "evaluator": evaluator_matches(meta, evaluator),
        "evaluator_score_hash": not selected or (
            evaluator.get("scores_sha256") == sha256(scores_path)
        ),
        "completion": (
            completion.get("corpus") == corpus
            and completion.get("arm") == arm
            and completion.get("seed") == 234
            and completion.get("selected_epoch") == meta.get("selected_epoch")
            and completion.get("train_meta_sha256") == sha256(meta_path)
            and completion.get("config_sha256") == sha256(run / "config.json")
            and completion.get("pilot_plan_sha256") == plan_hash
            and completion.get("pre_run_review_sha256") == review_hash
            and completion.get("source_snapshot_manifest_sha256")
            == sha256(run / "source_snapshot.sha256")
            and completion.get("residual_head_sha256")
            == (sha256(run / "residual_head.pth") if selected else None)
            and completion.get("val_scores_sha256")
            == (sha256(scores_path) if selected else None)
            and completion.get("metrics_sha256")
            == (sha256(metrics_path) if selected else None)
        ),
    }
    return {
        "run_dir": str(run.resolve()),
        "selected_epoch": meta.get("selected_epoch"),
        "validation": meta.get("selected_validation"),
        "checks": checks,
        "integrity_pass": all(checks.values()),
        "snapshot_inventory": inventory,
    }


def main(argv=None):
    args = parser().parse_args(argv)
    root = args.run_root.resolve()
    plan_hash = sha256(HERE / "PILOT_PLAN.md")
    review_hash = sha256(HERE / "PRE_RUN_REVIEW.md")
    verdict = review_verdict(HERE / "PRE_RUN_REVIEW.md")
    payload = {
        "stage": "V", "split": "val", "corpora": {},
        "pilot_plan_sha256": plan_hash,
        "pre_run_review_sha256": review_hash,
        "pre_run_review_verdict": verdict,
    }
    all_pass = verdict == "PASS"
    all_inventories = []
    for corpus in CORPORA:
        records = {
            arm: load_record(root, corpus, arm, plan_hash, review_hash)
            for arm in ARMS
        }
        all_inventories.extend(record["snapshot_inventory"]
                               for record in records.values())
        core = records["span_marginal"]["validation"]
        gates = {"has_feasible_core": core is not None}
        if core is not None:
            metrics = core["metrics"]
            anchor = metrics["score_powa"]
            candidate = metrics["score_candidate"]
            core_gain = candidate["within_roc"] - anchor["within_roc"]
            gates.update({
                "pooled_ap_feasible": candidate["pooled_ap"]
                >= anchor["pooled_ap"] - .002,
                "pooled_roc_feasible": candidate["pooled_roc"]
                >= anchor["pooled_roc"] - .002,
                "within_gain_at_least_0.020": core_gain >= .020,
                "improvement_ratio_at_least_0.55":
                core["paired_within"]["improvement_ratio"] >= .55,
                "zero_mean_residual":
                core["max_abs_grid_residual_mean"] <= 1e-6,
            })
            for arm in ("singleton", "shuffled_span"):
                control = records[arm]["validation"]
                gates[f"beats_{arm}_by_0.010"] = control is not None and (
                    candidate["within_roc"]
                    >= control["metrics"]["score_candidate"]["within_roc"] + .010
                )
            position_gain = max(
                metrics[f"score_{name}"]["within_roc"] - anchor["within_roc"]
                for name in ("chronological", "reverse_chronological",
                             "edge_first", "center_first")
            )
            gates["position_control_below_half_core_gain"] = (
                position_gain <= .5 * core_gain
            )
            if corpus == "hateclipseg":
                high = core["paired_within"]["high_pos"]
                gates["high_pos_gain_at_least_0.015"] = (
                    high["candidate"] >= high["anchor"] + .015
                )
                gates["high_pos_above_0.50"] = high["candidate"] > .50
        integrity = all(record["integrity_pass"] for record in records.values())
        corpus_pass = integrity and all(gates.values())
        payload["corpora"][corpus] = {
            "records": records, "integrity_pass": integrity,
            "gates": gates, "pass": corpus_pass,
        }
        all_pass = all_pass and corpus_pass
    snapshot_serialized = [
        json.dumps(inventory, sort_keys=True) if inventory is not None else None
        for inventory in all_inventories
    ]
    payload["all_six_snapshots_identical"] = (
        len(snapshot_serialized) == 6
        and None not in snapshot_serialized
        and len(set(snapshot_serialized)) == 1
    )
    payload["pass"] = all_pass and payload["all_six_snapshots_identical"]
    payload["verdict"] = (
        "ADVANCE_TO_STAGE_P" if payload["pass"] else "KILL_BEFORE_TEST"
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))
    return 0 if payload["pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
