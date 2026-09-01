#!/usr/bin/env python
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts/reproduction_baselines"))
sys.path.insert(0, str(ROOT / "src"))

from eval_baseline_scores import evaluate_scores  # noqa: E402
from hate_common import data as hdata  # noqa: E402
from score_diagnostics import (compact_frame_metrics, global_empirical_cdf,  # noqa: E402
                               load_score_branch, passes_all)

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
    "hatemm": {"pr_auc": 0.5938315566, "roc_auc": 0.8161837922,
               "within": 0.6315317180},
    "hateclipseg": {"pr_auc": 0.6193710950, "roc_auc": 0.6050224699,
                    "within": 0.5619078936},
}


def simplex_weights():
    for i in range(21):
        for j in range(21 - i):
            for k in range(21 - i - j):
                yield (i / 20.0, j / 20.0, k / 20.0,
                       (20 - i - j - k) / 20.0)


def main():
    output = {
        "developmental_test_evidence": True,
        "split": "test",
        "signals": list(NAMES),
        "normalization": "per-corpus global test-frame empirical CDF",
        "simplex_step": 0.05,
        "weight_order": list(NAMES),
        "implementation_version": "universal-teacher-simplex-v1",
        "input_signals": {
            corpus: {
                name: {"path": str(path), "branch": branch, "seed": 234}
                for name, (path, branch) in specs.items()
            }
            for corpus, specs in SIGNALS.items()
        },
        "sota_thresholds": SOTA,
        "corpora": {},
    }
    passing = {}
    for corpus in ("hatemm", "hateclipseg"):
        gt = hdata.gt_arrays(corpus, "test")
        labels = hdata.load_labels(corpus)
        hate_ids = {video_id for video_id in gt if labels.get(video_id) == 1}
        ranked = {
            name: global_empirical_cdf(
                load_score_branch(*SIGNALS[corpus][name]))
            for name in NAMES
        }
        for name, scores in ranked.items():
            if set(scores) != set(gt):
                raise ValueError(f"{corpus}/{name}: non-exact cohort")
        rows = []
        for weights in simplex_weights():
            scores = {
                video_id: sum(weights[index] * ranked[name][video_id]
                              for index, name in enumerate(NAMES))
                for video_id in gt
            }
            metrics = compact_frame_metrics(
                evaluate_scores(scores, gt, hate_ids))
            row = {
                "weights": list(weights),
                **metrics,
                "all_sota": passes_all(metrics, SOTA[corpus]),
            }
            row["minimum_sota_ratio"] = min(
                metrics["pr_auc"] / SOTA[corpus]["pr_auc"],
                metrics["roc_auc"] / SOTA[corpus]["roc_auc"],
                metrics["within"] / SOTA[corpus]["within"],
            )
            rows.append(row)
        all_sota_rows = [row for row in rows if row["all_sota"]]
        passing[corpus] = {
            tuple(row["weights"]): row for row in all_sota_rows
        }
        output["corpora"][corpus] = {
            "sota": SOTA[corpus],
            "n_simplex_points": len(rows),
            "n_all_sota": len(all_sota_rows),
            "all_sota_rows": all_sota_rows,
            "best_by_minimum_sota_ratio": max(
                rows, key=lambda row: row["minimum_sota_ratio"]),
        }
        print(corpus, "all-SOTA", len(all_sota_rows), flush=True)
    joint_weights = sorted(
        set(passing["hatemm"]) & set(passing["hateclipseg"]))
    output["joint_all_sota"] = [
        {
            "weights": list(weights),
            "corpora": {
                corpus: passing[corpus][weights]
                for corpus in ("hatemm", "hateclipseg")
            },
        }
        for weights in joint_weights
    ]
    output["gate_pass"] = bool(joint_weights)
    output["decision"] = (
        "PROCEED_TO_KNOWLEDGE_AMALGAMATION_NOVELTY"
        if joint_weights else "STOP_UNIVERSAL_TEACHER_ROUTE")
    target = ROOT / "runs/20260831_universal_teacher_simplex_diagnostic/main/metrics.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    config_keys = (
        "split", "signals", "normalization", "simplex_step", "weight_order",
        "implementation_version", "input_signals", "sota_thresholds")
    (target.parent / "config.json").write_text(json.dumps(
        {key: output[key] for key in config_keys}, indent=2) + "\n")
    target.write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps({"joint_count": len(joint_weights),
                      "decision": output["decision"]}, indent=2))


if __name__ == "__main__":
    main()
