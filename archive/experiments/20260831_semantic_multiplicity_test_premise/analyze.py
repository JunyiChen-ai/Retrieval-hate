#!/usr/bin/env python3
"""Test diagnostic for semantic-recurrence multiplicity bias in MIL evidence."""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score


HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
BASE = REPO / "scripts/reproduction_baselines"
sys.path.insert(0, str(BASE))

from eval_baseline_scores import evaluate_scores  # noqa: E402
from hate_common import data as hdata  # noqa: E402
from multihateloc import data as mdata  # noqa: E402


CORPORA = ("hatemm", "hateclipseg")
SCORE_PATHS = {
    corpus: REPO / (
        "runs/20260831_coalition_witness_candidate/pilot_seed234/"
        f"{corpus}/mobius_nonminimal/scores.jsonl"
    )
    for corpus in CORPORA
}
KERNEL_TEMPERATURE = 0.05
NONLOCAL_GAP_SECONDS = 5
HIGH_SCORE_FRACTION = 0.25
DIAGNOSTIC_DENSITY_PENALTY = 0.25


def load_scores(path: Path):
    output = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            video_id = row["video_id"]
            if video_id in output:
                raise RuntimeError(f"duplicate score row: {path}/{video_id}")
            output[video_id] = np.asarray(row["score_full"], dtype=float)
    return output


def validate_test_source(corpus, raw, gt):
    test_ids = hdata.load_split(corpus, "test")
    if len(test_ids) != len(set(test_ids)) or not set(gt).issubset(test_ids):
        raise RuntimeError(f"{corpus}: test manifest/GT cohort mismatch")
    if set(raw) != set(gt):
        raise RuntimeError(f"{corpus}: score/test coverage mismatch")

    source_dir = SCORE_PATHS[corpus].parent
    config = json.loads((source_dir / "config.json").read_text())
    metrics = json.loads((source_dir / "metrics.json").read_text())
    if (
        config.get("corpus") != corpus
        or config.get("arm") != "mobius_nonminimal"
        or config.get("test_labels_used_for_gradient_or_checkpoint_selection") is not False
        or "immediate test" not in config.get("split_policy", "")
    ):
        raise RuntimeError(f"{corpus}: invalid source config provenance")
    result = metrics.get("results", {}).get("score_full", {})
    if (
        metrics.get("corpus") != corpus
        or metrics.get("split") != "test"
        or Path(metrics.get("scores_file", "")).resolve() != SCORE_PATHS[corpus].resolve()
        or int(result.get("n_videos", -1)) != len(gt)
        or int(result.get("n_frames", -1)) != sum(len(value) for value in gt.values())
    ):
        raise RuntimeError(f"{corpus}: invalid source test metrics provenance")
    return sorted(set(test_ids) - set(gt))


def metric_triplet(scores, gt, positives):
    result = evaluate_scores(scores, gt, positives)
    return {
        "pooled_ap": float(result["pr_auc"]),
        "pooled_roc": float(result["roc_auc"]),
        "within_roc": float(result["per_video"]["macro_auc"]),
        "within_n": int(result["per_video"]["n_videos_both_classes"]),
    }


def normalized_rank(values):
    values = np.asarray(values, dtype=float)
    if len(values) <= 1:
        return np.zeros(len(values), dtype=float)
    return (rankdata(values, method="average") - 1.0) / (len(values) - 1.0)


def top_fraction_superlevel(values, fraction):
    values = np.asarray(values, dtype=float)
    if not len(values) or not 0.0 < fraction <= 1.0:
        raise ValueError("top-fraction superlevel needs nonempty values and fraction in (0, 1]")
    intended_count = max(1, int(math.ceil(len(values) * fraction)))
    threshold = float(np.partition(values, len(values) - intended_count)[
        len(values) - intended_count
    ])
    # Include the complete cutoff plateau instead of introducing time index as
    # an unregistered secondary key.
    return values >= threshold, intended_count, threshold


def semantic_nonlocal_density(corpus, video_id, length):
    path = Path(mdata.feature_path("visual", corpus, video_id))
    # mmap_mode="r" is deliberately read-only; normalization needs a private
    # working array and must never mutate the shared frozen feature cache.
    feature = np.array(np.load(path, mmap_mode="r"), dtype=np.float32, copy=True)
    if feature.shape != (length, mdata.FEATURE_DIMS["visual"]):
        raise RuntimeError(f"unaligned visual feature: {path} {feature.shape} vs {length}")
    if not np.isfinite(feature).all():
        raise RuntimeError(f"non-finite visual feature: {path}")
    feature /= np.maximum(np.linalg.norm(feature, axis=1, keepdims=True), 1e-12)
    similarity = feature @ feature.T
    index = np.arange(length)
    nonlocal_mask = np.abs(index[:, None] - index[None, :]) >= NONLOCAL_GAP_SECONDS
    kernel = np.exp(np.clip((similarity - 1.0) / KERNEL_TEMPERATURE, -50.0, 0.0))
    # The leading one is the instance's own quotient mass.  Local temporal
    # persistence is deliberately excluded because the preceding test
    # diagnosis showed opposite smoothing effects across the two corpora.
    return 1.0 + np.sum(kernel * nonlocal_mask, axis=1)


def safe_auc(label, score):
    label = np.asarray(label, dtype=np.int8)
    if len(label) < 2 or len(np.unique(label)) != 2:
        return None
    return float(roc_auc_score(label, score))


def finite_mean(values):
    values = np.asarray([value for value in values if value is not None], dtype=float)
    return float(values.mean()) if len(values) else None


def strictly_greater(left, right):
    return left is not None and right is not None and left > right


def analyze_corpus(corpus):
    gt = hdata.gt_arrays(corpus, "test")
    labels = hdata.load_labels(corpus)
    positives = {video_id for video_id in gt if labels[video_id] == 1}
    eligible = sorted(
        video_id for video_id in positives if len(np.unique(gt[video_id])) == 2
    )
    raw = load_scores(SCORE_PATHS[corpus])
    test_manifest_excluded = validate_test_source(corpus, raw, gt)

    corrected = {}
    reversed_density_control = {}
    fp_labels_pooled = []
    density_pooled = []
    reversed_density_pooled = []
    per_video_fp_density_auc = []
    per_video_reversed_auc = []
    fp_minus_tp_log_density = []
    raw_fp_evidence_share = []
    corrected_fp_evidence_share = []
    high_evidence_counts = []
    intended_high_evidence_counts = []
    high_evidence_fractions = []
    cutoff_tie_expansions = []

    for video_id, target_value in gt.items():
        target = np.asarray(target_value, dtype=np.int8)
        score = raw[video_id]
        if (
            score.shape != target.shape
            or not np.isfinite(score).all()
            or not np.isin(target, (0, 1)).all()
        ):
            raise RuntimeError(f"{corpus}/{video_id}: invalid aligned score")
        density = semantic_nonlocal_density(corpus, video_id, len(target))
        log_density = np.log(density)
        score_rank = normalized_rank(score)
        density_rank = normalized_rank(log_density)
        corrected[video_id] = score_rank - DIAGNOSTIC_DENSITY_PENALTY * density_rank
        reversed_density_control[video_id] = (
            score_rank - DIAGNOSTIC_DENSITY_PENALTY * density_rank[::-1]
        )

        if video_id not in eligible:
            continue
        high, intended_count, _ = top_fraction_superlevel(
            score, HIGH_SCORE_FRACTION
        )
        high_evidence_counts.append(int(high.sum()))
        intended_high_evidence_counts.append(int(intended_count))
        high_evidence_fractions.append(float(high.mean()))
        cutoff_tie_expansions.append(int(high.sum()) - int(intended_count))
        fp_label = 1 - target[high]
        high_density = log_density[high]
        high_reversed = log_density[::-1][high]
        fp_labels_pooled.extend(fp_label.tolist())
        density_pooled.extend(high_density.tolist())
        reversed_density_pooled.extend(high_reversed.tolist())
        per_video_fp_density_auc.append(safe_auc(fp_label, high_density))
        per_video_reversed_auc.append(safe_auc(fp_label, high_reversed))
        if np.any(fp_label == 1) and np.any(fp_label == 0):
            fp_minus_tp_log_density.append(float(
                high_density[fp_label == 1].mean() - high_density[fp_label == 0].mean()
            ))

        # This is a diagnostic share of model score mass inside the same
        # high-score support used above, not a reconstruction of the producer's
        # training-time bag likelihood.
        evidence = score[high]
        inverse_density_evidence = evidence / density[high]
        if evidence.sum() > 0 and inverse_density_evidence.sum() > 0:
            raw_fp_evidence_share.append(float(
                evidence[fp_label == 1].sum() / evidence.sum()
            ))
            corrected_fp_evidence_share.append(float(
                inverse_density_evidence[fp_label == 1].sum() /
                inverse_density_evidence.sum()
            ))

    return {
        "inputs": {
            "scores": str(SCORE_PATHS[corpus]),
            "visual_feature_family": mdata.FEATURE_DIRS["visual"],
            "n_test_videos": len(gt),
            "n_test_frames": int(sum(len(value) for value in gt.values())),
            "n_test_manifest_ids": len(hdata.load_split(corpus, "test")),
            "test_manifest_ids_excluded_from_gold_cohort": test_manifest_excluded,
            "n_eligible_positive_videos": len(eligible),
            "coverage_alignment_finiteness_pass": True,
        },
        "raw_metrics": metric_triplet(raw, gt, positives),
        "diagnostic_readout_metrics": {
            "inverse_semantic_density_rank_penalty": metric_triplet(
                corrected, gt, positives
            ),
            "time_reversed_density_control": metric_triplet(
                reversed_density_control, gt, positives
            ),
            "warning": (
                "These readouts use a fixed diagnostic score transform and are not a "
                "candidate method, deployable branch, or SOTA claim."
            ),
        },
        "premise_statistics": {
            "high_score_fraction": HIGH_SCORE_FRACTION,
            "top_fraction_rule": "ceil(T*fraction), cutoff-tie-inclusive superlevel",
            "mean_intended_high_evidence_seconds": float(
                np.mean(intended_high_evidence_counts)
            ),
            "mean_high_evidence_seconds": float(np.mean(high_evidence_counts)),
            "mean_actual_high_evidence_fraction": float(
                np.mean(high_evidence_fractions)
            ),
            "n_videos_with_cutoff_tie_expansion": int(sum(
                value > 0 for value in cutoff_tie_expansions
            )),
            "mean_cutoff_tie_expansion_seconds": float(
                np.mean(cutoff_tie_expansions)
            ),
            "pooled_auc_density_identifies_false_vs_true_high_evidence": safe_auc(
                fp_labels_pooled, density_pooled
            ),
            "pooled_auc_reversed_density_control": safe_auc(
                fp_labels_pooled, reversed_density_pooled
            ),
            "macro_auc_density_identifies_false_vs_true_high_evidence": finite_mean(
                per_video_fp_density_auc
            ),
            "macro_auc_reversed_density_control": finite_mean(per_video_reversed_auc),
            "mean_fp_minus_tp_log_density_within_high_evidence": finite_mean(
                fp_minus_tp_log_density
            ),
            "mean_raw_false_positive_high_score_mass_share": finite_mean(
                raw_fp_evidence_share
            ),
            "mean_inverse_density_false_positive_high_score_mass_share": finite_mean(
                corrected_fp_evidence_share
            ),
            "n_videos_with_defined_within_high_evidence_auc": int(sum(
                value is not None for value in per_video_fp_density_auc
            )),
        },
    }


def main():
    corpora = {corpus: analyze_corpus(corpus) for corpus in CORPORA}
    gates = {
        corpus: {
            "pooled_density_fp_auc_above_half": strictly_greater(
                row["premise_statistics"]
                ["pooled_auc_density_identifies_false_vs_true_high_evidence"], .5
            ),
            "pooled_density_beats_reversed_control": strictly_greater(
                row["premise_statistics"]
                ["pooled_auc_density_identifies_false_vs_true_high_evidence"],
                row["premise_statistics"]["pooled_auc_reversed_density_control"]
            ),
            "macro_density_fp_auc_above_half": strictly_greater(
                row["premise_statistics"]
                ["macro_auc_density_identifies_false_vs_true_high_evidence"], .5
            ),
            "macro_density_beats_reversed_control": strictly_greater(
                row["premise_statistics"]
                ["macro_auc_density_identifies_false_vs_true_high_evidence"],
                row["premise_statistics"]["macro_auc_reversed_density_control"]
            ),
            "inverse_density_reduces_fp_high_score_mass_share": (
                row["premise_statistics"]
                ["mean_inverse_density_false_positive_high_score_mass_share"] is not None
                and row["premise_statistics"]
                ["mean_raw_false_positive_high_score_mass_share"] is not None
                and row["premise_statistics"]
                ["mean_inverse_density_false_positive_high_score_mass_share"] <
                row["premise_statistics"]
                ["mean_raw_false_positive_high_score_mass_share"]
            ),
        }
        for corpus, row in corpora.items()
    }
    payload = {
        "date": "2026-08-31",
        "split": "test",
        "evidence_status": "iterative/developmental",
        "test_predictions_and_gt_used_for_error_analysis": True,
        "test_labels_used_for_gradient_or_checkpoint_selection": False,
        "fixed_density_definition": {
            "feature": "MultiHateLoc frozen visual ViT-B/16 feature",
            "kernel": "exp((cosine_similarity - 1) / temperature)",
            "kernel_temperature": KERNEL_TEMPERATURE,
            "exclude_temporal_distance_below_seconds": NONLOCAL_GAP_SECONDS,
            "self_mass": 1.0,
            "high_score_support": (
                "per-video ceil(T*0.25) cutoff, complete cutoff plateau included"
            ),
            "fp_auc_positive_class": "false-positive high-score second",
            "reverse_control": "reverse density order within the same video",
            "evidence_share": (
                "model score mass inside high-score support; diagnostic only, not "
                "producer bag-likelihood reconstruction"
            ),
            "diagnostic_rank_penalty": DIAGNOSTIC_DENSITY_PENALTY,
        },
        "corpora": corpora,
        "premise_gates": gates,
        "premise_pass_both": all(all(value.values()) for value in gates.values()),
        "verdict_if_fail": "STOP_BEFORE_METHOD_IMPLEMENTATION",
    }
    output = REPO / "runs/20260831_semantic_multiplicity_test_premise/main/metrics.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n")
    temporary.replace(output)
    print(json.dumps({
        "status": "complete",
        "output": str(output),
        "premise_pass_both": payload["premise_pass_both"],
    }, indent=2))


if __name__ == "__main__":
    main()
