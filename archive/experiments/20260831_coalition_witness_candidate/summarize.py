#!/usr/bin/env python3
"""Summarize the fixed two-corpus test pilot and apply frozen gates."""

from __future__ import annotations

import json
import math
from pathlib import Path


REPO = Path(__file__).resolve().parents[2]
RUN = REPO / "runs/20260831_coalition_witness_candidate/pilot_seed234"
CORPORA = ("hatemm", "hateclipseg")
ARMS = (
    "multihateloc",
    "no_infonce",
    "all_subset_mil",
    "synib",
    "mobius_nonminimal",
    "coalition_witness",
)
THRESHOLDS = {
    "hatemm": {"pooled_ap": 0.5938316, "pooled_roc": 0.8161838, "within_roc": 0.6315317},
    "hateclipseg": {"pooled_ap": 0.6193711, "pooled_roc": 0.6050225, "within_roc": 0.5619079},
}
BASELINE_ROOT = Path(
    "/home/jehc223/Hate-follow-up/results/reproduction/official_val/final/"
    "multihateloc"
)
EXPECTED = {
    "hatemm": {
        "lr": 0.00001849152228476098,
        "batch_size": 32,
        "max_epoch": 50,
        "k_proportion": 8,
        "lambda_smooth": 0.01420807210603241,
        "hidden": 512,
        "embed": 64,
        "dropout": 0.05,
        "temperature": 0.07,
    },
    "hateclipseg": {
        "lr": 0.00018190822304650636,
        "batch_size": 32,
        "max_epoch": 100,
        "k_proportion": 3,
        "lambda_smooth": 0.10337306075094418,
        "hidden": 512,
        "embed": 256,
        "dropout": 0.05,
        "temperature": 0.03,
    },
}
RECONSTRUCTION_TOLERANCE = 1e-5


def read_json(path):
    with Path(path).open(encoding="utf-8") as handle:
        return json.load(handle)


def triplet(result):
    return {
        "pooled_ap": float(result["pr_auc"]),
        "pooled_roc": float(result["roc_auc"]),
        "within_roc": float(result["per_video"]["macro_auc"]),
        "within_n": int(result["per_video"]["n_videos_both_classes"]),
        "n_videos": int(result["n_videos"]),
        "n_frames": int(result["n_frames"]),
    }


def equal_config_value(observed, expected):
    if isinstance(expected, float):
        return math.isclose(float(observed), expected, rel_tol=1e-12, abs_tol=1e-12)
    return observed == expected


def require_new_arm_integrity(corpus, arm, out_dir, expected_test_count):
    config = read_json(out_dir / "config.json")
    if config.get("corpus") != corpus or config.get("arm") != arm:
        raise RuntimeError(f"config arm/corpus mismatch: {out_dir}")
    if config.get("seed") != 234 or config.get("modalities") != ["visual", "audio", "text"]:
        raise RuntimeError(f"config seed/modality mismatch: {out_dir}")
    if config.get("test_labels_used_for_gradient_or_checkpoint_selection") is not False:
        raise RuntimeError(f"test isolation not asserted: {out_dir}")
    if config.get("split_policy") != (
        "official train; official validation only selects checkpoint; immediate test"
    ):
        raise RuntimeError(f"split policy mismatch: {out_dir}")
    for key, expected in EXPECTED[corpus].items():
        if key not in config or not equal_config_value(config[key], expected):
            raise RuntimeError(f"frozen config mismatch {corpus}/{arm}/{key}")

    record = read_json(out_dir / "train_record.json")
    if record.get("corpus") != corpus or record.get("arm") != arm:
        raise RuntimeError(f"train record arm/corpus mismatch: {out_dir}")
    if record.get("n_test") != expected_test_count:
        raise RuntimeError(f"train record test count mismatch: {out_dir}")
    if arm in ("mobius_nonminimal", "coalition_witness"):
        residual = record.get("full_score_reconstruction_max_abs_residual")
        if residual is None or not math.isfinite(float(residual)):
            raise RuntimeError(f"missing/nonfinite reconstruction residual: {out_dir}")
        if float(residual) > RECONSTRUCTION_TOLERANCE:
            raise RuntimeError(f"reconstruction residual exceeds tolerance: {out_dir}")
    if arm == "coalition_witness":
        diagnostics = record.get("test_coalition_diagnostics")
        if not isinstance(diagnostics, dict) or len(diagnostics) != expected_test_count:
            raise RuntimeError(f"candidate posterior coverage mismatch: {out_dir}")
        for video_id, diagnostic in diagnostics.items():
            mass = diagnostic.get("coalition_posterior_mass")
            if not isinstance(mass, list) or len(mass) != 7:
                raise RuntimeError(f"invalid posterior shape: {out_dir}/{video_id}")
            if not all(math.isfinite(float(value)) and float(value) >= 0 for value in mass):
                raise RuntimeError(f"invalid posterior values: {out_dir}/{video_id}")
            if not math.isclose(sum(float(value) for value in mass), 1.0,
                                rel_tol=1e-5, abs_tol=1e-5):
                raise RuntimeError(f"posterior mass does not sum to one: {out_dir}/{video_id}")
            if diagnostic.get("map_subset") not in range(1, 8):
                raise RuntimeError(f"invalid MAP coalition: {out_dir}/{video_id}")
            if not isinstance(diagnostic.get("map_second"), int) or diagnostic["map_second"] < 0:
                raise RuntimeError(f"invalid MAP time: {out_dir}/{video_id}")
            for key in ("atom_logit_mean", "atom_logit_max"):
                values = diagnostic.get(key)
                if (not isinstance(values, list) or len(values) != 7 or
                        not all(math.isfinite(float(value)) for value in values)):
                    raise RuntimeError(f"invalid {key}: {out_dir}/{video_id}")
    return True


def require_no_infonce_integrity(corpus, out_dir, expected_test_count):
    record = read_json(out_dir / "producer" / corpus / "train_log.json")
    args = record.get("args", {})
    if record.get("corpus") != corpus or record.get("n_test") != expected_test_count:
        raise RuntimeError(f"no_infonce record scope mismatch: {out_dir}")
    if args.get("seed") != 234 or float(args.get("lambda_contrast", -1)) != 0.0:
        raise RuntimeError(f"no_infonce did not only disable contrastive weight: {out_dir}")
    for key, expected in EXPECTED[corpus].items():
        if key not in args or not equal_config_value(args[key], expected):
            raise RuntimeError(f"no_infonce frozen config mismatch {corpus}/{key}")
    return True


def main():
    results = {}
    sources = {}
    for corpus in CORPORA:
        results[corpus] = {}
        sources[corpus] = {}
        baseline = BASELINE_ROOT / corpus / "seed_234/frame_eval.json"
        baseline_payload = read_json(baseline)
        if baseline_payload.get("split") != "test":
            raise RuntimeError(f"baseline is not test: {baseline}")
        results[corpus]["multihateloc"] = triplet(
            baseline_payload["results"]["score_fused"]
        )
        sources[corpus]["multihateloc"] = str(baseline)
        expected_test_count = results[corpus]["multihateloc"]["n_videos"]

        for arm in ARMS[1:]:
            out_dir = RUN / corpus / arm
            path = out_dir / "metrics.json"
            payload = read_json(path)
            if payload.get("split") != "test" or payload.get("corpus") != corpus:
                raise RuntimeError(f"wrong evaluator scope: {path}")
            branch = "score_fused" if arm == "no_infonce" else "score_full"
            result = payload["results"][branch]
            if result["n_videos_missing_from_scores"] or result["n_videos_not_in_gold"]:
                raise RuntimeError(f"incomplete test coverage: {path}")
            if int(result["n_videos"]) != expected_test_count:
                raise RuntimeError(f"test cohort size mismatch: {path}")
            if arm == "no_infonce":
                require_no_infonce_integrity(corpus, out_dir, expected_test_count)
            else:
                require_new_arm_integrity(corpus, arm, out_dir, expected_test_count)
            results[corpus][arm] = triplet(result)
            sources[corpus][arm] = str(path)

    candidate = "coalition_witness"
    mechanism_controls = ("all_subset_mil", "synib", "mobius_nonminimal")
    mechanism_pass_by_corpus = {
        corpus: all(
            results[corpus][candidate]["within_roc"] > results[corpus][control]["within_roc"]
            for control in mechanism_controls
        ) and (
            results[corpus][candidate]["within_roc"] > results[corpus]["no_infonce"]["within_roc"]
        )
        for corpus in CORPORA
    }
    sota_pass_by_corpus = {
        corpus: all(
            results[corpus][candidate][metric] > threshold
            for metric, threshold in THRESHOLDS[corpus].items()
        )
        for corpus in CORPORA
    }
    payload = {
        "date": "2026-08-31",
        "split": "test",
        "evidence_status": "iterative/developmental",
        "test_labels_used_for_gradient_or_checkpoint_selection": False,
        "corpora_trained_independently": True,
        "seed": 234,
        "integrity_pass": True,
        "full_score_reconstruction_tolerance": RECONSTRUCTION_TOLERANCE,
        "results": results,
        "sources": sources,
        "frozen_sota_thresholds": THRESHOLDS,
        "mechanism_controls": list(mechanism_controls),
        "mechanism_pass_by_corpus": mechanism_pass_by_corpus,
        "mechanism_pass_both": all(mechanism_pass_by_corpus.values()),
        "sota_pass_by_corpus": sota_pass_by_corpus,
        "sota_pass_both": all(sota_pass_by_corpus.values()),
        "continue_to_four_corpora": (
            all(mechanism_pass_by_corpus.values()) and all(sota_pass_by_corpus.values())
        ),
    }
    target = RUN / "metrics.json"
    temporary = target.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n")
    temporary.replace(target)
    (RUN / "config.json").write_text(json.dumps({
        "corpora": list(CORPORA),
        "arms": list(ARMS),
        "seed": 234,
        "training": "independent per corpus; validation only checkpoint selection; immediate test",
        "metrics": ["pooled_ap", "pooled_roc", "within_roc"],
    }, indent=2) + "\n")
    (RUN / "code_version.txt").write_text(
        "Working-tree snapshot dated 2026-08-31; strict coalition witness pilot\n"
    )
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
