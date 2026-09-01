#!/usr/bin/env python3
"""Validate run provenance and apply the frozen Stage-V gates."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path


HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
CORPORA = ("hatemm", "hateclipseg")
ARMS = ("negative_donor", "positive_donor", "shifted_mask")
EXPECTED_ANCHORS = {
    "hatemm": REPO / "results/reproduction/powa_macil/"
              "final_maskfix_finetune_hatemm_seed234_e5",
    "hateclipseg": REPO / "runs/20260831_powa_starting_point/"
                   "hcs_maskfix_seed234",
}
CURRENT_SOURCE_MAP = {
    **{
        f"experiment/{name}": HERE / name
        for name in (
            "README.md", "PILOT_PLAN.md", "NOVELTY_SCOUT.md",
            "PRE_RUN_REVIEW.md", "model.py", "train.py", "infer.py",
            "complete_run.py", "audit_interventions.py",
            "summarize_stage_v.py", "run_stage_v.sh",
            "test_rank_transport.py",
        )
    },
    "shared/__init__.py": REPO / "src/weak_supervision/__init__.py",
    "shared/same_corpus_insertion.py": (
        REPO / "src/weak_supervision/same_corpus_insertion.py"
    ),
    "powa_macil/model.py": (
        REPO / "scripts/reproduction_baselines/powa_macil/model.py"
    ),
    "powa_macil/dataset.py": (
        REPO / "scripts/reproduction_baselines/powa_macil/dataset.py"
    ),
    "macilsd/Transformer.py": (
        REPO / "scripts/reproduction_baselines/macilsd/Transformer.py"
    ),
    "macilsd/avce_network.py": (
        REPO / "scripts/reproduction_baselines/macilsd/avce_network.py"
    ),
    "evaluator/eval_baseline_scores.py": (
        REPO / "scripts/reproduction_baselines/eval_baseline_scores.py"
    ),
    "evaluator/frame_eval_common.py": (
        REPO / "scripts/duplex/frame_eval_common.py"
    ),
}


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-root", required=True, type=Path)
    p.add_argument("--out", required=True, type=Path)
    return p


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_if_file(path: Path):
    return sha256(path) if path.is_file() else None


def review_status(review_path: Path):
    text = review_path.read_text()
    match = re.search(r"^Verdict:\s*\*\*(PASS|BLOCK)\*\*\s*$", text, re.M)
    return match.group(1) if match else None


def verify_snapshot(manifest: Path) -> bool:
    if not manifest.is_file():
        return False
    for line in manifest.read_text().splitlines():
        if not line.strip():
            continue
        expected, raw_path = line.split(None, 1)
        path = Path(raw_path.strip())
        if not path.is_file() or sha256(path) != expected:
            return False
    return True


def snapshot_inventory(manifest: Path):
    if not manifest.is_file():
        return None
    root = manifest.parent / "source_snapshot"
    inventory = {}
    try:
        for line in manifest.read_text().splitlines():
            if not line.strip():
                continue
            expected, raw_path = line.split(None, 1)
            relative = str(Path(raw_path.strip()).relative_to(root))
            inventory[relative] = expected
    except (ValueError, OSError):
        return None
    return inventory


def inventory_digest(inventory):
    if inventory is None:
        return None
    payload = json.dumps(inventory, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def current_source_matches(inventory):
    if inventory is None or set(inventory) != set(CURRENT_SOURCE_MAP):
        return False
    return all(
        path.is_file() and sha256(path) == inventory[relative]
        for relative, path in CURRENT_SOURCE_MAP.items()
    )


def evaluator_matches(validation, evaluator):
    if validation is None:
        return False
    for branch, internal in validation["metrics"].items():
        official = evaluator["results"].get(branch)
        if official is None:
            return False
        pairs = (
            (internal["pooled_ap"], official["pr_auc"]),
            (internal["pooled_roc"], official["roc_auc"]),
            (internal["within_roc"], official["per_video"]["macro_auc"]),
        )
        if any(abs(float(left) - float(right)) > 1e-12
               for left, right in pairs):
            return False
    return True


def load_selected(run_root, corpus, arm, plan_hash, review_hash):
    run_dir = run_root / f"{corpus}_{arm}_seed234"
    path = run_dir / "train_meta.json"
    meta = json.loads(path.read_text())
    metrics_path = run_dir / "metrics.json"
    scores_path = run_dir / "val_scores.jsonl"
    evaluator = json.loads(metrics_path.read_text()) if metrics_path.is_file() else None
    completion_path = run_dir / "completion.json"
    completion = (
        json.loads(completion_path.read_text())
        if completion_path.is_file() else None
    )
    selected = meta.get("selected_epoch") is not None
    inventory = snapshot_inventory(run_dir / "source_snapshot.sha256")
    recorded_args = meta.get("args", {})
    frozen_args = {
        "epochs": 5,
        "batch_size": 24,
        "lr": 2e-4,
        "weight_decay": 1e-4,
        "margin": 1.0,
        "stability_weight": 0.5,
        "topk_divisor": 16,
        "min_donor_rows": 12,
        "max_donor_rows": 36,
        "boundary_buffer": 3,
        "pooled_tolerance": 0.002,
    }
    integrity = {
        "meta_corpus": meta.get("corpus") == corpus,
        "meta_arm": meta.get("arm") == arm,
        "meta_seed": meta.get("seed") == 234,
        "meta_out_dir": Path(recorded_args.get("out_dir", "")).resolve() == run_dir,
        "frozen_hyperparameters": all(
            recorded_args.get(name) == value for name, value in frozen_args.items()
        ),
        "plan_hash": meta.get("pilot_plan_sha256") == plan_hash,
        "anchor_path": (
            Path(meta.get("anchor_checkpoint", "")).resolve()
            == EXPECTED_ANCHORS[corpus].resolve()
        ),
        "anchor_model_hash": (
            meta.get("anchor_model_sha256")
            == sha256_if_file(EXPECTED_ANCHORS[corpus] / "model.pth")
        ),
        "corpus_specific": meta.get("anchor_corpus_specific") is True,
        "no_cross_corpus": meta.get("cross_corpus_training") is False,
        "no_test_selection": (
            meta.get("test_labels_used_for_training_or_selection") is False
        ),
        "powa_frozen": meta.get("powa_parameters_trainable") == 0,
        "selected_files_consistent": (
            selected
            and all((run_dir / name).is_file() for name in (
                "rank_head.pth", "val_scores.jsonl", "val_metrics.json",
                "metrics.json",
            ))
            or not selected
            and all(not (run_dir / name).exists() for name in (
                "rank_head.pth", "val_scores.jsonl", "val_metrics.json",
                "metrics.json",
            ))
        ),
        "evaluator_metrics": (evaluator is not None) == selected,
        "evaluator_split": (
            not selected or evaluator.get("split") == "val"
        ),
        "evaluator_corpus": (
            not selected or evaluator.get("corpus") == corpus
        ),
        "evaluator_score_hash": (
            not selected or (
                evaluator is not None
                and scores_path.is_file()
                and evaluator.get("scores_sha256") == sha256(scores_path)
            )
        ),
        "evaluator_matches_internal": (
            not selected or (
                evaluator is not None
                and evaluator_matches(meta.get("selected_validation"), evaluator)
            )
        ),
        "source_snapshot": verify_snapshot(run_dir / "source_snapshot.sha256"),
        "snapshot_covers_current_source": current_source_matches(inventory),
        "source_records": all(
            (run_dir / name).is_file()
            for name in ("code_commit.txt", "tracked_code.patch", "run.pid")
        ),
        "completion_record": completion is not None,
        "completion_identity": (
            completion is not None
            and completion.get("corpus") == corpus
            and completion.get("arm") == arm
            and completion.get("seed") == 234
            and completion.get("selected_epoch") == meta.get("selected_epoch")
            and completion.get("train_meta_sha256") == sha256(path)
            and completion.get("pilot_plan_sha256") == plan_hash
            and completion.get("pre_run_review_sha256") == review_hash
            and completion.get("source_snapshot_manifest_sha256")
            == sha256_if_file(run_dir / "source_snapshot.sha256")
            and completion.get("metrics_sha256")
            == (sha256_if_file(metrics_path) if selected else None)
            and completion.get("val_scores_sha256")
            == (sha256_if_file(scores_path) if selected else None)
            and completion.get("rank_head_sha256")
            == (sha256_if_file(run_dir / "rank_head.pth") if selected else None)
        ),
    }
    return {
        "run_dir": str(run_dir.resolve()),
        "train_meta": str(path.resolve()),
        "selected_epoch": meta.get("selected_epoch"),
        "anchor_model_sha256": meta.get("anchor_model_sha256"),
        "validation": meta.get("selected_validation"),
        "integrity": integrity,
        "integrity_pass": all(integrity.values()),
        "snapshot_content_digest": inventory_digest(inventory),
    }, meta


def build_stage_v_summary(run_root: Path):
    """Recompute Stage-V authority from the current six run artifacts."""
    run_root = run_root.resolve()
    plan_path = HERE / "PILOT_PLAN.md"
    review_path = HERE / "PRE_RUN_REVIEW.md"
    plan_hash = sha256(plan_path)
    review_hash = sha256(review_path)
    review_verdict = review_status(review_path)
    payload = {
        "stage": "V",
        "split": "val",
        "frozen_plan": str(plan_path),
        "frozen_plan_sha256": plan_hash,
        "pre_run_review": str(review_path),
        "pre_run_review_sha256": review_hash,
        "pre_run_review_verdict": review_verdict,
        "corpora": {},
    }
    all_pass = review_verdict == "PASS"
    all_snapshot_digests = []
    for corpus in CORPORA:
        records = {}
        metas = {}
        for arm in ARMS:
            records[arm], metas[arm] = load_selected(
                run_root, corpus, arm, plan_hash, review_hash
            )
        anchor_hashes = {
            record["anchor_model_sha256"] for record in records.values()
        }
        snapshot_digests = {
            record["snapshot_content_digest"] for record in records.values()
        }
        all_snapshot_digests.extend(
            record["snapshot_content_digest"] for record in records.values()
        )
        integrity = {
            "all_runs_valid": all(
                record["integrity_pass"] for record in records.values()
            ),
            "matched_anchor_hash": len(anchor_hashes) == 1,
            "matched_snapshot_content": (
                len(snapshot_digests) == 1 and None not in snapshot_digests
            ),
            "review_pass": review_verdict == "PASS",
        }
        core = records["negative_donor"]["validation"]
        gates = {}
        if core is None:
            gates["has_feasible_checkpoint"] = False
        else:
            metric = core["metrics"]
            anchor = metric["score_powa"]
            candidate = metric["score_rank_transport"]
            epoch0 = metas["negative_donor"]["history"][0]["validation"]
            gates.update({
                "has_feasible_checkpoint": True,
                "pooled_ap_feasible": (
                    candidate["pooled_ap"] >= anchor["pooled_ap"] - 0.002
                ),
                "pooled_roc_feasible": (
                    candidate["pooled_roc"] >= anchor["pooled_roc"] - 0.002
                ),
                "beats_no_insertion_identity_by_0.020": (
                    candidate["within_roc"] >= anchor["within_roc"] + 0.020
                ),
                "epoch0_exact_framewise_identity": (
                    epoch0["invariants"]["zero_residual_videos"]
                    == len(metas["negative_donor"]["splits"]["val_ids"])
                    and epoch0["invariants"][
                        "zero_residual_pointwise_identity_max_abs_error"
                    ] == 0.0
                    and epoch0["metrics"]["score_rank_transport"]
                    == epoch0["metrics"]["score_powa"]
                ),
                "improvement_ratio_at_least_0.55": (
                    core["paired_within"]["improvement_ratio"] >= 0.55
                ),
                "unique_ratio_at_least_0.95": (
                    core["invariants"]["mean_order_unique_ratio"] >= 0.95
                ),
                "exact_score_multiset": core["invariants"]["exact_float64"],
                "raw_order_matches_transport_below_0.002": abs(
                    metric["score_order_raw"]["within_roc"]
                    - candidate["within_roc"]
                ) < 0.002,
                "tie_reverse_delta_below_0.002": abs(
                    metric["score_tie_reverse"]["within_roc"]
                    - candidate["within_roc"]
                ) < 0.002,
                "tie_random_delta_below_0.002": abs(
                    metric["score_tie_random"]["within_roc"]
                    - candidate["within_roc"]
                ) < 0.002,
            })
            if corpus == "hateclipseg":
                stratum = core["paired_within"]["strata"]["gt_0.6"]
                gates["high_pos_gain_at_least_0.015"] = (
                    stratum["candidate_within"]
                    >= stratum["anchor_within"] + 0.015
                )
                gates["high_pos_ends_above_0.50"] = (
                    stratum["candidate_within"] > 0.50
                )
            for arm in ("positive_donor", "shifted_mask"):
                control = records[arm]["validation"]
                gates[f"beats_{arm}_by_0.010"] = (
                    control is not None
                    and candidate["within_roc"]
                    >= control["metrics"]["score_rank_transport"]["within_roc"]
                    + 0.010
                )
            position_gain = max(
                metric[name]["within_roc"] - anchor["within_roc"]
                for name in (
                    "score_chronological", "score_reverse_chronological",
                    "score_edge_first", "score_center_first",
                )
            )
            core_gain = candidate["within_roc"] - anchor["within_roc"]
            gates["position_control_below_half_core_gain"] = (
                position_gain <= 0.5 * core_gain
            )
            direct = metric["score_direct_additive"]
            direct_feasible = (
                direct["pooled_ap"] >= anchor["pooled_ap"] - 0.002
                and direct["pooled_roc"] >= anchor["pooled_roc"] - 0.002
            )
            transport_cross = core["ordering_diagnostics"][
                "score_rank_transport"
            ]["hate_vs_negative_video_frames_auc"]
            direct_cross = core["ordering_diagnostics"][
                "score_direct_additive"
            ]["hate_vs_negative_video_frames_auc"]
            gates["constraint_attribution_vs_direct"] = (
                (not direct_feasible) or direct_cross < transport_cross - 0.002
            )
        corpus_pass = all(integrity.values()) and all(gates.values())
        all_pass = all_pass and corpus_pass
        payload["corpora"][corpus] = {
            "records": records,
            "integrity": integrity,
            "gates": gates,
            "pass": corpus_pass,
        }
    payload["all_six_runs_same_snapshot"] = (
        len(set(all_snapshot_digests)) == 1
        and len(all_snapshot_digests) == len(CORPORA) * len(ARMS)
        and None not in all_snapshot_digests
    )
    all_pass = all_pass and payload["all_six_runs_same_snapshot"]
    payload["pass"] = all_pass
    payload["verdict"] = (
        "ADVANCE_TO_STAGE_P" if all_pass else "KILL_BEFORE_TEST"
    )
    return payload


def main(argv=None):
    args = parser().parse_args(argv)
    payload = build_stage_v_summary(args.run_root)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))
    return 0 if payload["pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
