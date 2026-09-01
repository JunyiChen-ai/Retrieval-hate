#!/usr/bin/env python
"""Scale-transfer falsification for the seven frozen shared teacher tuples."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import rankdata

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts/reproduction_baselines"))
sys.path.insert(0, str(ROOT / "src"))

from eval_baseline_scores import evaluate_scores  # noqa: E402
from hate_common import data as hdata  # noqa: E402
from score_diagnostics import compact_frame_metrics, load_score_branch, passes_all  # noqa: E402

NAMES = ("lexical", "powa", "vera", "multihateloc")
OLD = Path("/home/jehc223/Hate-follow-up/results/reproduction")
SIGNALS = {
    "hatemm": {
        "lexical": (ROOT / "runs/20260831_video_label_lexical_locality/premise/hatemm/scores.jsonl", "score_lexical"),
        "powa": (OLD / "powa_macil/final_maskfix_finetune_hatemm_seed234_e5/hatemm/scores.jsonl", "score_powa"),
        "vera": (OLD / "official_val/final/vera/hatemm/seed_234/scores.jsonl", "score_official_postprocessed"),
        "multihateloc": (OLD / "official_val/final/multihateloc/hatemm/seed_234/hatemm/scores.jsonl", "score_fused"),
    },
    "hateclipseg": {
        "lexical": (ROOT / "runs/20260831_video_label_lexical_locality/premise/hateclipseg/scores.jsonl", "score_lexical"),
        "powa": (ROOT / "runs/20260831_powa_starting_point/hcs_maskfix_seed234/scores.jsonl", "score_powa"),
        "vera": (OLD / "official_val/final/vera/hateclipseg/seed_234/scores.jsonl", "score_official_postprocessed"),
        "multihateloc": (OLD / "official_val/final/multihateloc/hateclipseg/seed_234/hateclipseg/scores.jsonl", "score_fused"),
    },
}
SOTA = {
    "hatemm": {"pr_auc": 0.5938315566, "roc_auc": 0.8161837922, "within": 0.6315317180},
    "hateclipseg": {"pr_auc": 0.6193710950, "roc_auc": 0.6050224699, "within": 0.5619078936},
}
FROZEN_WEIGHTS = (
    (0.05, 0.30, 0.30, 0.35),
    (0.05, 0.30, 0.35, 0.30),
    (0.10, 0.25, 0.35, 0.30),
    (0.10, 0.25, 0.40, 0.25),
    (0.10, 0.30, 0.45, 0.15),
    (0.15, 0.20, 0.40, 0.25),
    (0.15, 0.25, 0.45, 0.15),
)


def per_video_ecdf(scores: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    return {
        video_id: ((rankdata(values, method="average") - 1.0) /
                   max(len(values) - 1, 1))
        for video_id, values in scores.items()
    }


def reference_mid_cdf(values: np.ndarray, reference: np.ndarray) -> np.ndarray:
    ordered = np.sort(reference)
    left = np.searchsorted(ordered, values, side="left")
    right = np.searchsorted(ordered, values, side="right")
    return (left + right) / (2.0 * len(ordered))


def heldout_ecdf(scores: dict[str, np.ndarray], folds: int = 5) -> dict[str, np.ndarray]:
    video_ids = sorted(scores)
    assignment = {video_id: index % folds for index, video_id in enumerate(video_ids)}
    output = {}
    for fold in range(folds):
        reference_ids = [video_id for video_id in video_ids if assignment[video_id] != fold]
        target_ids = [video_id for video_id in video_ids if assignment[video_id] == fold]
        reference = np.concatenate([scores[video_id] for video_id in reference_ids])
        if not len(reference):
            raise ValueError("empty held-out ECDF reference")
        for video_id in target_ids:
            output[video_id] = reference_mid_cdf(scores[video_id], reference)
    return output


def evaluate_grid(mapped, gt, hate_ids, threshold):
    rows = []
    for weights in FROZEN_WEIGHTS:
        scores = {
            video_id: sum(weights[index] * mapped[name][video_id]
                          for index, name in enumerate(NAMES))
            for video_id in gt
        }
        metrics = compact_frame_metrics(evaluate_scores(scores, gt, hate_ids))
        rows.append({"weights": list(weights), **metrics,
                     "all_sota": passes_all(metrics, threshold)})
    return rows


def main():
    source = ROOT / "runs/20260831_universal_teacher_simplex_diagnostic/main/metrics.json"
    prior = json.loads(source.read_text())
    observed = tuple(tuple(row["weights"]) for row in prior["joint_all_sota"])
    if observed != FROZEN_WEIGHTS:
        raise RuntimeError("frozen shared tuple set no longer matches source artifact")
    frozen_config = json.loads((Path(__file__).with_name("config.json")).read_text())
    expected_signals = {
        corpus: {name: {"path": str(path), "branch": branch, "seed": 234}
                 for name, (path, branch) in specs.items()}
        for corpus, specs in SIGNALS.items()
    }
    if frozen_config["implementation_version"] != "teacher-scale-transfer-v1":
        raise RuntimeError("config implementation version mismatch")
    if frozen_config["split"] != "test" or not frozen_config["developmental_test_evidence"]:
        raise RuntimeError("config split/developmental status mismatch")
    if frozen_config["source_artifact"] != str(source):
        raise RuntimeError("config source artifact mismatch")
    if tuple(frozen_config["weight_order"]) != NAMES:
        raise RuntimeError("config weight order mismatch")
    if tuple(tuple(row) for row in frozen_config["frozen_weights"]) != FROZEN_WEIGHTS:
        raise RuntimeError("config frozen weights mismatch")
    if frozen_config["input_signals"] != expected_signals or frozen_config["sota_thresholds"] != SOTA:
        raise RuntimeError("config signal or SOTA mismatch")
    output = {
        "status": "developmental_test_diagnostic",
        "split": "test",
        "implementation_version": "teacher-scale-transfer-v1",
        "source_artifact": str(source),
        "weight_order": list(NAMES),
        "frozen_weights": [list(weights) for weights in FROZEN_WEIGHTS],
        "input_signals": expected_signals,
        "sota_thresholds": SOTA,
        "normalizations": {
            "fivefold_video_heldout_ecdf": "sorted video IDs round-robin; empirical mid-CDF from other four folds only",
            "per_video_ecdf": "average ranks within each target video",
            "raw_identity": "stored branch values without mapping",
        },
        "corpora": {},
    }
    for corpus in ("hatemm", "hateclipseg"):
        gt = hdata.gt_arrays(corpus, "test")
        labels = hdata.load_labels(corpus)
        hate_ids = {video_id for video_id in gt if labels.get(video_id) == 1}
        raw = {name: load_score_branch(*SIGNALS[corpus][name]) for name in NAMES}
        for name, rows in raw.items():
            if set(rows) != set(gt):
                raise ValueError(f"{corpus}/{name}: non-exact cohort")
            for video_id, values in rows.items():
                if values.shape != gt[video_id].shape:
                    raise ValueError(f"{corpus}/{name}/{video_id}: length mismatch")
        variants = {
            "fivefold_video_heldout_ecdf": {name: heldout_ecdf(rows) for name, rows in raw.items()},
            "per_video_ecdf": {name: per_video_ecdf(rows) for name, rows in raw.items()},
            "raw_identity": raw,
        }
        output["corpora"][corpus] = {}
        for variant, mapped in variants.items():
            rows = evaluate_grid(mapped, gt, hate_ids, SOTA[corpus])
            output["corpora"][corpus][variant] = {
                "rows": rows,
                "n_all_sota": sum(row["all_sota"] for row in rows),
            }
            print(corpus, variant, "all-SOTA", sum(row["all_sota"] for row in rows), flush=True)
    joint = {}
    for variant in output["normalizations"]:
        passing = []
        for weights in FROZEN_WEIGHTS:
            ok = all(next(row for row in output["corpora"][corpus][variant]["rows"]
                          if tuple(row["weights"]) == weights)["all_sota"]
                     for corpus in ("hatemm", "hateclipseg"))
            if ok:
                passing.append(list(weights))
        joint[variant] = passing
    output["joint_all_sota_weights"] = joint
    output["gate_pass"] = bool(joint["fivefold_video_heldout_ecdf"])
    output["decision"] = ("SCALE_TRANSFER_PREMISE_PASS" if output["gate_pass"]
                          else "STOP_SHARED_TARGET_TRANSFER")
    target = ROOT / "runs/20260831_teacher_scale_transfer_diagnostic/main/metrics.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps({"gate_pass": output["gate_pass"], "decision": output["decision"],
                      "joint_counts": {key: len(value) for key, value in joint.items()}}, indent=2))


if __name__ == "__main__":
    main()
