"""Developmental test analysis of the frozen P-MIL baseline predictions."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import torch
from scipy.stats import spearmanr


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
BASELINES = REPO / "scripts/reproduction_baselines"
for path in (REPO, BASELINES):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from hate_common import data as hdata  # noqa: E402
from eval_baseline_scores import evaluate_scores  # noqa: E402

RUN_SPEC = importlib.util.spec_from_file_location("pmil_frozen_run", HERE / "run.py")
PORT = importlib.util.module_from_spec(RUN_SPEC)
assert RUN_SPEC.loader is not None
RUN_SPEC.loader.exec_module(PORT)


RUN_ROOT = REPO / "runs/20260831_multimodal_pmil_baseline/pilot_seed234"
CORPORA = ("hatemm", "hateclipseg")


def intervals(binary):
    return PORT.contiguous_components(np.asarray(binary) > 0)


def interval_iou(first, second):
    left = max(first[0], second[0])
    right = min(first[1], second[1])
    intersection = max(0, right - left)
    union = first[1] - first[0] + second[1] - second[0] - intersection
    return intersection / max(union, 1)


def proposal_oracle_scores(proposals, gold_intervals):
    proposals = np.asarray(proposals)
    scores = np.zeros(len(proposals), dtype=np.float64)
    if not gold_intervals:
        return scores, []
    for index, proposal in enumerate(proposals):
        scores[index] = max(
            interval_iou(proposal, gold) for gold in gold_intervals
        )
    best_per_gold = [
        max(interval_iou(proposal, gold) for proposal in proposals)
        for gold in gold_intervals
    ]
    return scores, best_per_gold


def top_proposal_summary(proposals, scores, whole_index):
    proposals = np.asarray(proposals)
    scores = np.asarray(scores)
    tied = np.isclose(scores, scores.max(), rtol=0.0, atol=1e-12)
    lengths = proposals[tied, 1] - proposals[tied, 0]
    return {
        "whole_is_top": bool(tied[int(whole_index)]),
        "top_tie_count": int(tied.sum()),
        "top_tied_length_median": float(np.median(lengths)),
    }


def compact_metrics(result):
    return {
        "pooled_ap": result["pr_auc"],
        "pooled_roc": result["roc_auc"],
        "within_roc": result["per_video"]["macro_auc"],
        "within_n": result["per_video"]["n_videos_both_classes"],
    }


def analyze_corpus(corpus, device):
    run_dir = RUN_ROOT / corpus
    required = ("config.json", "train_log.json", "model.pt", "scores.jsonl", "metrics.json")
    missing = [name for name in required if not (run_dir / name).is_file()]
    if missing:
        raise RuntimeError(f"incomplete frozen run for {corpus}: {missing}")
    config = json.loads((run_dir / "config.json").read_text())
    train_log = json.loads((run_dir / "train_log.json").read_text())
    evaluator_metrics = json.loads((run_dir / "metrics.json").read_text())
    if config.get("corpus") != corpus or train_log.get("selected_epoch") is None:
        raise RuntimeError(f"invalid selected-run metadata for {corpus}")
    if evaluator_metrics.get("corpus") != corpus or evaluator_metrics.get("split") != "test":
        raise RuntimeError(f"invalid frozen evaluator scope for {corpus}")
    test_ids = PORT.evaluator_test_ids(corpus, hdata.load_split(corpus, "test"))
    gt = hdata.gt_arrays(corpus, "test")
    if set(test_ids) != set(gt):
        raise RuntimeError(f"analysis cohort mismatch for {corpus}")
    frozen_predictions = hdata.load_scores_jsonl(run_dir / "scores.jsonl")
    if set(frozen_predictions) != set(test_ids):
        raise RuntimeError(f"frozen prediction cohort mismatch for {corpus}")

    source_path = Path(config["source_checkpoint_path"])
    if source_path.resolve() != PORT.source_checkpoint(corpus, config["seed"]).resolve():
        raise RuntimeError(f"unexpected frozen proposal checkpoint for {corpus}")
    source_state = torch.load(
        source_path,
        map_location="cpu", weights_only=True,
    )
    source = PORT.MultiHateLoc(
        {name: PORT.mdata.FEATURE_DIMS[name] for name in PORT.MODALITIES},
        hidden=config["source_model_hidden"], embed=config["source_model_embed"],
        dropout=0.05, k_proportion=3, temperature=0.07,
    ).to(device)
    source.load_state_dict(source_state)
    source.eval()

    model = PORT.MultimodalPMIL(
        {name: PORT.mdata.FEATURE_DIMS[name] for name in PORT.MODALITIES},
        hidden=config["hidden"], roi_size=config["roi_size"], dropout=0.1,
        max_train_proposals=config["maximum_sampled_train_proposals"],
    ).to(device)
    model.load_state_dict(torch.load(run_dir / "model.pt", map_location=device, weights_only=True))
    model.eval()

    arm_scores = {
        "source_smil": {},
        "pmil_full": {},
        "pmil_without_completeness": {},
        "pmil_hate_cas_only": {},
        "proposal_oracle": {},
    }
    for modality in PORT.MODALITIES:
        arm_scores[f"pmil_{modality}_full"] = {}
    whole_top = []
    top_lengths = []
    top_tie_counts = []
    proposal_gt_iou = []
    pairwise_rank = {f"{a}__{b}": [] for i, a in enumerate(PORT.MODALITIES)
                     for b in PORT.MODALITIES[i + 1:]}

    with torch.no_grad():
        for index, video_id in enumerate(test_ids, 1):
            features, length = PORT.load_features(corpus, video_id, device)
            if len(gt[video_id]) != length:
                raise RuntimeError(f"feature/GT length mismatch for {corpus}/{video_id}")
            source_score = PORT.base_frame_scores(source, features)
            bounds_np = PORT.generate_proposals(source_score)
            bounds = torch.from_numpy(bounds_np).to(device)
            outputs, used = model(features, bounds, training_sample=False)
            used_np = used.cpu().numpy()

            per_modality = {}
            hate_terms, no_comp_terms, full_terms = [], [], []
            for modality in PORT.MODALITIES:
                out = outputs[modality]
                hate = torch.softmax(out["cas"], 1)[:, 0]
                attention = torch.sigmoid(out["attention"])
                completeness = torch.sigmoid(out["completeness"])
                hate_terms.append(hate)
                no_comp_terms.append(hate * attention)
                full_terms.append(hate * attention * completeness)
                per_modality[modality] = PORT.proposal_to_frames(
                    length, used_np, (hate * attention * completeness).cpu().numpy()
                )
                arm_scores[f"pmil_{modality}_full"][video_id] = per_modality[modality]

            full = torch.stack(full_terms).mean(0).cpu().numpy()
            no_comp = torch.stack(no_comp_terms).mean(0).cpu().numpy()
            hate_only = torch.stack(hate_terms).mean(0).cpu().numpy()
            arm_scores["source_smil"][video_id] = source_score
            arm_scores["pmil_full"][video_id] = PORT.proposal_to_frames(
                length, used_np, full
            )
            frozen_row = frozen_predictions[video_id]
            if "score_pmil" not in frozen_row or not np.allclose(
                arm_scores["pmil_full"][video_id],
                np.asarray(frozen_row["score_pmil"], dtype=np.float64),
                rtol=0.0, atol=1e-6,
            ):
                raise RuntimeError(
                    f"recomputed full score differs from frozen prediction for "
                    f"{corpus}/{video_id}"
                )
            arm_scores["pmil_without_completeness"][video_id] = PORT.proposal_to_frames(
                length, used_np, no_comp
            )
            arm_scores["pmil_hate_cas_only"][video_id] = PORT.proposal_to_frames(
                length, used_np, hate_only
            )

            whole_index = np.where(
                (used_np[:, 0] == 0) & (used_np[:, 1] == length)
            )[0]
            if len(whole_index) != 1:
                raise RuntimeError(f"whole proposal multiplicity for {corpus}/{video_id}")
            top = top_proposal_summary(used_np, full, whole_index[0])
            whole_top.append(top["whole_is_top"])
            top_lengths.append(top["top_tied_length_median"])
            top_tie_counts.append(top["top_tie_count"])

            gold_intervals = intervals(gt[video_id])
            oracle, best_per_gold = proposal_oracle_scores(used_np, gold_intervals)
            proposal_gt_iou.extend(best_per_gold)
            arm_scores["proposal_oracle"][video_id] = PORT.proposal_to_frames(
                length, used_np, oracle
            )

            if len(np.unique(gt[video_id])) == 2:
                for key in pairwise_rank:
                    first, second = key.split("__")
                    value = spearmanr(per_modality[first], per_modality[second]).statistic
                    if np.isfinite(value):
                        pairwise_rank[key].append(float(value))
            if index % 50 == 0:
                print(f"{corpus} analysis {index}/{len(test_ids)}", flush=True)

    hate_ids = {video_id for video_id, rows in gt.items() if np.asarray(rows).max() > 0}
    if not proposal_gt_iou:
        raise RuntimeError(f"no positive GT intervals in analysis cohort for {corpus}")
    metrics = {
        arm: compact_metrics(evaluate_scores(scores, gt, hate_ids))
        for arm, scores in arm_scores.items()
    }
    return {
        "corpus": corpus,
        "developmental_test_evidence": True,
        "frozen_model_file": str((run_dir / "model.pt").resolve()),
        "test_prediction_file": str((run_dir / "scores.jsonl").resolve()),
        "test_gt_use": (
            "post-training error analysis only: evaluator metrics, proposal boundary "
            "coverage, and modality rank agreement; no gradient or checkpoint selection"
        ),
        "arm_policy": {
            "reported_baseline_arm": "pmil_full (must reproduce frozen score_pmil)",
            "source_smil": "frozen proposal producer baseline diagnostic",
            "same_checkpoint_diagnostic_arms": [
                "pmil_without_completeness", "pmil_hate_cas_only",
                *[f"pmil_{name}_full" for name in PORT.MODALITIES],
            ],
            "proposal_oracle": "GT-informed proposal-set upper-bound diagnostic",
            "diagnostic_arms_are_not_retrained_methods_or_sota_entries": True,
        },
        "metrics": metrics,
        "diagnostics": {
            "whole_video_is_top_proposal_fraction": float(np.mean(whole_top)),
            "top_score_tied_proposal_length_seconds_median": float(np.median(top_lengths)),
            "top_score_tied_proposal_length_seconds_mean": float(np.mean(top_lengths)),
            "top_score_tie_video_fraction": float(np.mean(np.asarray(top_tie_counts) > 1)),
            "gt_interval_max_proposal_iou_event_macro_mean": float(
                np.mean(proposal_gt_iou)
            ),
            "gt_interval_recall_iou_010_event_macro": float(
                np.mean(np.asarray(proposal_gt_iou) >= 0.10)
            ),
            "gt_interval_recall_iou_030_event_macro": float(
                np.mean(np.asarray(proposal_gt_iou) >= 0.30)
            ),
            "gt_interval_recall_iou_050_event_macro": float(
                np.mean(np.asarray(proposal_gt_iou) >= 0.50)
            ),
            "proposal_recall_definition": (
                "for each contiguous positive GT interval, maximum temporal IoU "
                "over the frozen proposals; averages weight GT intervals equally"
            ),
            "eligible_video_pairwise_modality_spearman_mean": {
                key: float(np.mean(values)) if values else None
                for key, values in pairwise_rank.items()
            },
            "eligible_video_pairwise_modality_spearman_n_finite": {
                key: len(values) for key, values in pairwise_rank.items()
            },
            "eligible_video_pairwise_modality_spearman_n_undefined": {
                key: sum(len(np.unique(gt[video_id])) == 2 for video_id in test_ids)
                - len(values)
                for key, values in pairwise_rank.items()
            },
            "pairwise_spearman_definition": (
                "Spearman correlation between per-modality full frame score vectors, "
                "restricted to test videos with both GT frame classes; constant vectors "
                "are undefined and counted separately"
            ),
        },
    }


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable")
    payload = {
        "analysis_type": "iterative/developmental test error analysis",
        "method_development_may_use_this_analysis": True,
        "test_labels_used_for_gradient_or_checkpoint_selection": False,
        "corpora": {},
    }
    for corpus in CORPORA:
        payload["corpora"][corpus] = analyze_corpus(corpus, torch.device("cuda"))
    target = RUN_ROOT / "test_error_analysis.json"
    target.write_text(json.dumps(payload, indent=2) + "\n")
    print(target)


if __name__ == "__main__":
    main()
