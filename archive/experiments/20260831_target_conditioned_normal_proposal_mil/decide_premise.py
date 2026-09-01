"""Apply the frozen two-corpus premise gate without inspecting predictions."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


CORPORA = ("hatemm", "hateclipseg")


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", required=True)
    args = parser.parse_args(argv)
    root = Path(args.run_root)

    corpora = {}
    for corpus in CORPORA:
        metrics_path = root / corpus / "metrics.json"
        support_path = root / corpus / "support.json"
        metrics = json.loads(metrics_path.read_text())
        support = json.loads(support_path.read_text())
        if metrics.get("corpus") != corpus or metrics.get("split") != "test":
            raise RuntimeError(f"unexpected evaluator scope in {metrics_path}")
        results = metrics["results"]
        conditional = float(results["score_conditional"]["per_video"]["macro_auc"])
        unconditional = float(results["score_unconditional"]["per_video"]["macro_auc"])
        support_fraction = float(support["test_proposal_support_fraction"])
        if not all(map(math.isfinite, (conditional, unconditional, support_fraction))):
            raise RuntimeError(f"non-finite premise input for {corpus}")
        support_pass = support_fraction >= 0.80
        if bool(support["support_pass"]) != support_pass:
            raise RuntimeError(f"inconsistent support decision in {support_path}")
        delta = conditional - unconditional
        corpora[corpus] = {
            "metrics_file": str(metrics_path.resolve()),
            "support_file": str(support_path.resolve()),
            "conditional_within_roc": conditional,
            "unconditional_within_roc": unconditional,
            "conditional_minus_unconditional": delta,
            "strict_direction_pass": delta > 0.0,
            "test_proposal_support_fraction": support_fraction,
            "support_pass": support_pass,
        }

    support_pass_both = all(row["support_pass"] for row in corpora.values())
    strict_direction_pass_both = all(
        row["strict_direction_pass"] for row in corpora.values()
    )
    one_corpus_gain_at_least_020 = max(
        row["conditional_minus_unconditional"] for row in corpora.values()
    ) >= 0.020 - 1e-12
    premise_pass_both = (
        support_pass_both
        and strict_direction_pass_both
        and one_corpus_gain_at_least_020
    )
    payload = {
        "developmental_test_evidence": True,
        "producer_used_test_labels_or_temporal_gt": False,
        "frozen_gate": {
            "support_fraction_each_corpus_at_least": 0.80,
            "conditional_minus_unconditional_each_corpus_strictly_greater_than": 0.0,
            "conditional_minus_unconditional_at_least_one_corpus_at_least": 0.020,
        },
        "corpora": corpora,
        "support_pass_both": support_pass_both,
        "strict_direction_pass_both": strict_direction_pass_both,
        "one_corpus_gain_at_least_020": one_corpus_gain_at_least_020,
        "premise_pass_both": premise_pass_both,
        "continue_to_formal_method": premise_pass_both,
        "decision": "IMPLEMENT_FORMAL_METHOD" if premise_pass_both else "STOP_BEFORE_FORMAL_METHOD",
    }
    target = root / "verdict.json"
    target.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps({
        "verdict": str(target),
        "premise_pass_both": premise_pass_both,
        "decision": payload["decision"],
    }))


if __name__ == "__main__":
    main()
