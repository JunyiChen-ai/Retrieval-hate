#!/usr/bin/env python3
"""Read-only MultiHateLoc test error analysis under the test-first protocol."""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr


HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
BASE = REPO / "scripts/reproduction_baselines"
sys.path.insert(0, str(BASE))

from hate_common import data as hdata  # noqa: E402


CORPORA = ("hatemm", "mhclip_en", "mhclip_zh", "hateclipseg")
SEEDS = (234, 2025, 3407)
BRANCHES = (
    "score_audio",
    "score_visual",
    "score_text",
    "score_fused",
    "score_dms",
    "score_union",
)
# Must match data.MODALITIES and therefore the DMS weight slots exactly.
MODALITY_BRANCHES = ("score_visual", "score_audio", "score_text")
SOURCE_ROOT = Path(
    "/home/jehc223/Hate-follow-up/results/reproduction/official_val/final/"
    "multihateloc"
)
RUN_DIR = REPO / "runs/20260831_multihateloc_test_error_analysis/main"


def load_json(path: Path):
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def load_scores(path: Path):
    rows = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            video_id = row.pop("video_id")
            if video_id in rows:
                raise RuntimeError(f"duplicate prediction row: {video_id}")
            rows[video_id] = {
                key: np.asarray(value, dtype=float) for key, value in row.items()
            }
    return rows


def finite_mean(values):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    return float(values.mean()) if len(values) else None


def finite_sd(values):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    return float(values.std(ddof=1)) if len(values) > 1 else None


def safe_spearman(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    keep = np.isfinite(x) & np.isfinite(y)
    if keep.sum() < 3 or len(np.unique(x[keep])) < 2 or len(np.unique(y[keep])) < 2:
        return {"rho": None, "pvalue": None, "n": int(keep.sum())}
    result = spearmanr(x[keep], y[keep])
    return {
        "rho": float(result.statistic),
        "pvalue": float(result.pvalue),
        "n": int(keep.sum()),
    }


def ranges(binary):
    binary = np.asarray(binary, dtype=bool)
    padded = np.r_[False, binary, False].astype(np.int8)
    change = np.diff(padded)
    return [[int(a), int(b - 1)] for a, b in zip(
        np.flatnonzero(change == 1), np.flatnonzero(change == -1)
    )]


def metric_triplet(result):
    return {
        "pooled_ap": float(result["pr_auc"]),
        "pooled_roc": float(result["roc_auc"]),
        "within_roc": float(result["per_video"]["macro_auc"]),
        "within_n": int(result["per_video"]["n_videos_both_classes"]),
    }


def summarize_metric_values(values):
    keys = ("pooled_ap", "pooled_roc", "within_roc")
    return {
        key: {
            "mean": finite_mean([row[key] for row in values]),
            "sample_sd": finite_sd([row[key] for row in values]),
            "values_by_seed": {
                str(seed): float(row[key]) for seed, row in zip(SEEDS, values)
            },
        }
        for key in keys
    }


def source_paths(corpus, seed):
    root = SOURCE_ROOT / corpus / f"seed_{seed}"
    return {
        "frame_eval": root / "frame_eval.json",
        "scores": root / corpus / "scores.jsonl",
        "train_log": root / corpus / "train_log.json",
    }


def analyze_corpus(corpus):
    gt = hdata.gt_arrays(corpus, "test")
    labels = hdata.load_labels(corpus)
    eligible = sorted(
        video_id for video_id, gold in gt.items()
        if labels.get(video_id) == 1 and len(np.unique(gold)) == 2
    )
    occupancy = {video_id: float(np.mean(gt[video_id])) for video_id in eligible}
    top_k_count = {
        video_id: int(math.ceil(len(gt[video_id]) / 3.0)) for video_id in eligible
    }
    top_k_fraction = {
        video_id: float(top_k_count[video_id] / len(gt[video_id]))
        for video_id in eligible
    }
    transitions = {
        video_id: int(np.sum(np.asarray(gt[video_id])[1:] != np.asarray(gt[video_id])[:-1]))
        for video_id in eligible
    }

    source_records = []
    branch_metrics = {branch: [] for branch in BRANCHES}
    fused_auc_by_video = {video_id: [] for video_id in eligible}
    best_modality_auc_by_video = {video_id: [] for video_id in eligible}
    oracle_auc_by_video = {video_id: [] for video_id in eligible}
    fused_beats_modalities = []
    dms_selector_correct = []
    dms_weight_entropy = []
    dms_weight_max = []
    best_modality_tie_pairs = 0
    selected_modality_counts = {name: 0 for name in ("audio", "visual", "text")}
    best_modality_counts = {name: 0 for name in ("audio", "visual", "text")}

    for seed in SEEDS:
        paths = source_paths(corpus, seed)
        for path in paths.values():
            if not path.is_file():
                raise FileNotFoundError(path)
        evaluated = load_json(paths["frame_eval"])
        if evaluated.get("split") != "test" or evaluated.get("corpus") != corpus:
            raise RuntimeError(f"non-test or wrong-corpus evaluator artifact: {paths['frame_eval']}")
        if set(BRANCHES) - set(evaluated["results"]):
            raise RuntimeError(f"missing evaluated branch in {paths['frame_eval']}")
        scores = load_scores(paths["scores"])
        if set(scores) != set(gt):
            missing = sorted(set(gt) - set(scores))
            extra = sorted(set(scores) - set(gt))
            raise RuntimeError(
                f"coverage mismatch {corpus}/{seed}: missing={missing[:3]} extra={extra[:3]}"
            )
        for video_id, gold in gt.items():
            if set(BRANCHES) - set(scores[video_id]):
                raise RuntimeError(f"missing score branch {corpus}/{seed}/{video_id}")
            for branch in BRANCHES:
                values = scores[video_id][branch]
                if values.shape != np.asarray(gold).shape or not np.isfinite(values).all():
                    raise RuntimeError(f"invalid score grid {corpus}/{seed}/{video_id}/{branch}")

        train_log = load_json(paths["train_log"])
        video_state = train_log.get("test_video_scores", {})
        if set(video_state) != set(gt):
            raise RuntimeError(f"DMS state coverage mismatch {corpus}/{seed}")

        source_records.append({
            "seed": seed,
            "frame_eval": str(paths["frame_eval"]),
            "scores": str(paths["scores"]),
            "train_log": str(paths["train_log"]),
            "n_test_videos": len(scores),
            "n_test_frames": int(sum(len(value) for value in gt.values())),
            "coverage_exact": True,
            "alignment_and_finiteness_pass": True,
        })
        for branch in BRANCHES:
            branch_metrics[branch].append(metric_triplet(evaluated["results"][branch]))

        per_auc = {
            branch: evaluated["results"][branch]["per_video"]["per_video_auc"]
            for branch in BRANCHES
        }
        for video_id in eligible:
            fused_auc = float(per_auc["score_fused"][video_id])
            modality_auc = np.asarray([
                float(per_auc[branch][video_id]) for branch in MODALITY_BRANCHES
            ])
            all_auc = np.r_[modality_auc, fused_auc]
            fused_auc_by_video[video_id].append(fused_auc)
            best_modality_auc_by_video[video_id].append(float(modality_auc.max()))
            oracle_auc_by_video[video_id].append(float(all_auc.max()))
            fused_beats_modalities.append(float(fused_auc > modality_auc.max()))

            weights = np.asarray(video_state[video_id]["weights"], dtype=float)
            if weights.shape != (3,) or not np.isfinite(weights).all() or np.any(weights < 0):
                raise RuntimeError(f"invalid DMS weights {corpus}/{seed}/{video_id}")
            total = float(weights.sum())
            if not math.isclose(total, 1.0, rel_tol=1e-5, abs_tol=1e-5):
                raise RuntimeError(f"DMS weights do not sum to one {corpus}/{seed}/{video_id}")
            selected = int(np.argmax(weights))
            best = int(np.argmax(modality_auc))
            tied_best = np.isclose(modality_auc, modality_auc.max(), rtol=1e-9, atol=1e-12)
            best_modality_tie_pairs += int(tied_best.sum() > 1)
            dms_selector_correct.append(float(tied_best[selected]))
            names = ("visual", "audio", "text")
            selected_modality_counts[names[selected]] += 1
            best_modality_counts[names[best]] += 1
            clipped = np.clip(weights, 1e-12, 1.0)
            dms_weight_entropy.append(float(-(clipped * np.log(clipped)).sum() / np.log(3.0)))
            dms_weight_max.append(float(weights.max()))

    mean_fused = {video_id: finite_mean(values) for video_id, values in fused_auc_by_video.items()}
    mean_best_modality = {
        video_id: finite_mean(values) for video_id, values in best_modality_auc_by_video.items()
    }
    mean_oracle = {video_id: finite_mean(values) for video_id, values in oracle_auc_by_video.items()}

    bins = {
        "positive_fraction_le_1_3": [v for v in eligible if occupancy[v] <= 1.0 / 3.0],
        "positive_fraction_1_3_to_2_3": [
            v for v in eligible if 1.0 / 3.0 < occupancy[v] <= 2.0 / 3.0
        ],
        "positive_fraction_gt_2_3": [v for v in eligible if occupancy[v] > 2.0 / 3.0],
    }
    occupancy_strata = {
        name: {
            "n_videos": len(video_ids),
            "mean_fused_within_roc_across_seed_video_pairs": finite_mean([
                auc for video_id in video_ids for auc in fused_auc_by_video[video_id]
            ]),
            "mean_positive_fraction": finite_mean([occupancy[v] for v in video_ids]),
        }
        for name, video_ids in bins.items()
    }

    mismatch = [abs(occupancy[v] - top_k_fraction[v]) for v in eligible]
    fused_values = [mean_fused[v] for v in eligible]
    oracle_values = [mean_oracle[v] for v in eligible]
    best_modality_values = [mean_best_modality[v] for v in eligible]
    stable_rows = []
    for video_id in eligible:
        values = fused_auc_by_video[video_id]
        stable_rows.append({
            "video_id": video_id,
            "mean_fused_auc": finite_mean(values),
            "sample_sd_fused_auc": finite_sd(values),
            "fused_auc_by_seed": {str(seed): float(v) for seed, v in zip(SEEDS, values)},
            "positive_fraction": occupancy[video_id],
            "top_k_count": top_k_count[video_id],
            "top_k_fraction": top_k_fraction[video_id],
            "top_k_occupancy_mismatch": abs(
                occupancy[video_id] - top_k_fraction[video_id]
            ),
            "n_label_transitions": transitions[video_id],
            "positive_ranges": ranges(gt[video_id]),
            "mean_best_modality_auc": mean_best_modality[video_id],
            "mean_best_branch_oracle_auc": mean_oracle[video_id],
        })
    stable_rows.sort(key=lambda row: row["mean_fused_auc"])

    return {
        "inputs_and_integrity": source_records,
        "n_eligible_positive_videos": len(eligible),
        "branch_test_metrics": {
            branch: summarize_metric_values(values)
            for branch, values in branch_metrics.items()
        },
        "fixed_top_third_diagnostic": {
            "median_absolute_positive_fraction_mismatch": float(np.median(mismatch)),
            "fused_auc_vs_absolute_mismatch_spearman": safe_spearman(mismatch, fused_values),
            "fused_auc_vs_positive_fraction_spearman": safe_spearman(
                [occupancy[v] for v in eligible], fused_values
            ),
            "fused_auc_vs_label_transition_count_spearman": safe_spearman(
                [transitions[v] for v in eligible], fused_values
            ),
            "occupancy_strata": occupancy_strata,
        },
        "fusion_and_dms_diagnostic": {
            "fraction_seed_video_pairs_fused_beats_every_modality": finite_mean(
                fused_beats_modalities
            ),
            "mean_fused_auc": finite_mean(fused_values),
            "mean_best_single_modality_auc_test_oracle": finite_mean(best_modality_values),
            "mean_best_branch_auc_test_oracle": finite_mean(oracle_values),
            "mean_oracle_minus_fused_auc": finite_mean(
                np.asarray(oracle_values) - np.asarray(fused_values)
            ),
            "dms_highest_weight_matches_best_modality_auc_fraction": finite_mean(
                dms_selector_correct
            ),
            "dms_normalized_weight_entropy_mean": finite_mean(dms_weight_entropy),
            "dms_max_weight_mean": finite_mean(dms_weight_max),
            "dms_selected_modality_counts": selected_modality_counts,
            "test_oracle_best_modality_counts": best_modality_counts,
            "test_oracle_best_modality_tie_pairs": best_modality_tie_pairs,
            "best_modality_count_tie_break": (
                "deterministic first slot in model order visual, audio, text; "
                "selector agreement counts any tied-best slot as correct"
            ),
            "warning": (
                "Oracle and selector-agreement values use test GT and are diagnostic only; "
                "they are not candidate predictions or deployable selection rules."
            ),
        },
        "worst_10_by_three_seed_mean": stable_rows[:10],
        "best_10_by_three_seed_mean": stable_rows[-10:][::-1],
    }


def main():
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    config = {
        "date": "2026-08-31",
        "split": "test",
        "corpora": list(CORPORA),
        "seeds": list(SEEDS),
        "primary_branch": "score_fused",
        "top_k_proportion_parameter": 3,
        "top_k_count_formula": "ceil(video_length / 3)",
        "test_predictions_and_gt_used_for_error_analysis": True,
        "test_labels_used_for_gradient_or_checkpoint_selection": False,
        "future_test_evidence_status": "iterative/developmental",
    }
    (RUN_DIR / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    (RUN_DIR / "code_version.txt").write_text(
        "Working-tree snapshot dated 2026-08-31; entrypoint "
        "experiments/20260831_multihateloc_test_error_analysis/analyze.py\n"
    )
    payload = dict(config)
    payload["corpora_analysis"] = {
        corpus: analyze_corpus(corpus) for corpus in CORPORA
    }
    target = RUN_DIR / "metrics.json"
    temporary = target.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n")
    temporary.replace(target)
    print(json.dumps({
        "status": "complete",
        "output": str(target),
        "corpora": list(CORPORA),
    }, indent=2))


if __name__ == "__main__":
    main()
