#!/usr/bin/env python3
"""CPU-only unit tests for the V8 matched-seed statistical evaluator."""
import numpy as np
import tempfile
from pathlib import Path

from relation_v8.fair_eval import (SEEDS, cluster_bootstrap, extended_metrics,
                                   matched_manifests, seed_map,
                                   select_shared, summarize_seed_metrics)
from relation_v4.io import sha256


def main():
    paths = [f"root/seed_{s}/scores.jsonl" for s in SEEDS]
    mapping, kind = seed_map(paths, "trainable", "val")
    assert kind == "trainable_matched" and list(mapping) == list(SEEDS)
    shared, kind = seed_map("root/frozen/scores.jsonl", "fixed", "val", True)
    assert kind == "fixed_shared" and len(set(shared.values())) == 1
    try:
        seed_map("root/unmarked/scores.jsonl", "unmarked", "val")
        raise AssertionError("unmarked single path did not fail")
    except RuntimeError:
        pass
    try:
        seed_map(paths, "false-fixed", "val", True)
        raise AssertionError("multi-path fixed expert did not fail")
    except RuntimeError:
        pass
    for bad in (paths[:2], [paths[0], paths[0], paths[2]]):
        try:
            seed_map(bad, "broken", "val")
            raise AssertionError("missing/duplicate seed did not fail")
        except RuntimeError:
            pass
    with tempfile.TemporaryDirectory() as directory:
        identity = Path(directory) / "frozen_config.json"
        identity.write_text('{"checkpoint":"same"}\n')
        digest = sha256(identity)
        fixed_provenance = {"val_checkpoint_identity": "checkpoint-A",
                            "test_checkpoint_identity": "checkpoint-A",
                            "identity_source": str(identity),
                            "identity_source_sha256": digest}
        manifest = {"experts": [
            {"name": "fixed", "val_scores": "v", "test_scores": "t",
             "fixed_across_seeds": True, "fixed_provenance": fixed_provenance},
            {"name": "learned", "val_scores": paths,
             "test_scores": [x.replace("root/", "test/") for x in paths]}]}
        matched, audit = matched_manifests(manifest)
        assert matched[2025]["experts"][1]["val_scores"] == paths[1]
        assert [x["kind"] for x in audit] == ["fixed_shared", "trainable_matched"]
        assert audit[0]["fixed_provenance"]["identity_source_hash_verified"]
        broken = dict(fixed_provenance, test_checkpoint_identity="checkpoint-B")
        manifest["experts"][0]["fixed_provenance"] = broken
        try:
            matched_manifests(manifest)
            raise AssertionError("val/test checkpoint identity mismatch did not fail")
        except RuntimeError:
            pass

    gt = {"mixed": np.array([0, 1, 0, 1]),
          "positive": np.array([1, 1]), "negative": np.array([0, 0, 0])}
    baseline = {v: np.zeros(len(y)) for v, y in gt.items()}
    method = {"mixed": np.array([0., 1., 0., 1.]),
              "positive": np.ones(2), "negative": np.zeros(3)}
    metric = extended_metrics(method, gt)
    assert metric["within_video_macro_roc"] == 1.0
    assert metric["within_video_macro_roc_eligible_videos"] == 1
    assert metric["hateful_video_count"] == 2
    method_runs = {s: method for s in SEEDS}
    baseline_runs = {s: baseline for s in SEEDS}
    ci1 = cluster_bootstrap(method_runs, baseline_runs, gt, 100, 17)
    ci2 = cluster_bootstrap(method_runs, baseline_runs, gt, 100, 17)
    assert ci1 == ci2
    assert ci1["frame_ap"]["point_difference"] > 0
    summary = summarize_seed_metrics([metric, metric, metric])
    assert summary["frame_ap"]["std"] == 0.0
    grids = {
        234: [{"beta": 0., "gamma": 0., "frame_ap": .5, "frame_roc": .5},
              {"beta": 1., "gamma": 0., "frame_ap": .9, "frame_roc": .6}],
        2025: [{"beta": 0., "gamma": 0., "frame_ap": .5, "frame_roc": .5},
               {"beta": 1., "gamma": 0., "frame_ap": .4, "frame_roc": .6}],
        3407: [{"beta": 0., "gamma": 0., "frame_ap": .5, "frame_roc": .5},
               {"beta": 1., "gamma": 0., "frame_ap": .8, "frame_roc": .6}],
    }
    selected, fallback, eligible, averaged = select_shared(grids)
    assert (selected["beta"], selected["gamma"]) == (1., 0.)
    assert fallback["frame_ap"] == .5 and len(eligible) == 2 and len(averaged) == 2
    print("Relation-V8 fair evaluator synthetic tests: PASS")


if __name__ == "__main__":
    main()
