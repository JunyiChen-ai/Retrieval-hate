#!/usr/bin/env python3
"""Export dense scores from a selected frozen-score assignment checkpoint."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader


HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
BASELINES = REPO / "scripts" / "reproduction_baselines"
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(BASELINES))

from hate_common import data as hdata  # noqa: E402
from powa_macil.dataset import PowaTestDataset, usable_text_ids  # noqa: E402

from model import FrozenPowaTemporalAssignment, safe_logit  # noqa: E402
from train import load_anchor, sha256, stable_transport  # noqa: E402
from summarize_stage_v import (  # noqa: E402
    build_stage_v_summary,
    current_source_matches,
    review_status,
    snapshot_inventory,
    verify_snapshot,
)


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint-dir", required=True, type=Path)
    p.add_argument("--split", default="val", choices=("val", "test"))
    p.add_argument("--device", default="cuda")
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--num-workers", type=int, default=4)
    return p


def require_test_authorization(checkpoint: Path, meta):
    run_root = checkpoint.parent
    summary_path = run_root / "stage_v_summary.json"
    completion_path = checkpoint / "completion.json"
    plan_path = HERE / "PILOT_PLAN.md"
    review_path = HERE / "PRE_RUN_REVIEW.md"
    if not summary_path.is_file() or not completion_path.is_file():
        raise RuntimeError("test inference requires Stage-V summary and completion")
    summary = json.loads(summary_path.read_text())
    recomputed_summary = build_stage_v_summary(run_root)
    completion = json.loads(completion_path.read_text())
    corpus = meta.get("corpus")
    record = (
        summary.get("corpora", {})
        .get(corpus, {})
        .get("records", {})
        .get("negative_donor", {})
    )
    checks = {
        "stage_v_pass": (
            summary.get("pass") is True
            and summary.get("verdict") == "ADVANCE_TO_STAGE_P"
            and summary.get("all_six_runs_same_snapshot") is True
        ),
        "summary_recomputed_exactly": summary == recomputed_summary,
        "review_pass": review_status(review_path) == "PASS",
        "review_hash": summary.get("pre_run_review_sha256") == sha256(review_path),
        "plan_hash": (
            summary.get("frozen_plan_sha256") == sha256(plan_path)
            == meta.get("pilot_plan_sha256")
            == completion.get("pilot_plan_sha256")
        ),
        "core_identity": (
            meta.get("arm") == "negative_donor"
            and meta.get("seed") == 234
            and Path(record.get("run_dir", "")).resolve() == checkpoint
            and Path(record.get("train_meta", "")).resolve()
            == (checkpoint / "train_meta.json").resolve()
            and record.get("selected_epoch") == meta.get("selected_epoch")
        ),
        "corpus_pass": summary.get("corpora", {}).get(corpus, {}).get("pass") is True,
        "summary_record_integrity": (
            record.get("integrity_pass") is True
            and all(record.get("integrity", {}).values())
        ),
        "completion_meta": (
            completion.get("corpus") == corpus
            and completion.get("arm") == "negative_donor"
            and completion.get("seed") == 234
            and completion.get("selected_epoch") == meta.get("selected_epoch")
            and completion.get("train_meta_sha256")
            == sha256(checkpoint / "train_meta.json")
            and completion.get("pre_run_review_sha256") == sha256(review_path)
            and completion.get("rank_head_sha256")
            == sha256(checkpoint / "rank_head.pth")
        ),
        "source_snapshot": (
            completion.get("source_snapshot_manifest_sha256")
            == sha256(checkpoint / "source_snapshot.sha256")
            and verify_snapshot(checkpoint / "source_snapshot.sha256")
            and current_source_matches(
                snapshot_inventory(checkpoint / "source_snapshot.sha256")
            )
        ),
    }
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise RuntimeError(f"test inference authorization failed: {failed}")
    return {
        "stage_v_summary": str(summary_path.resolve()),
        "stage_v_summary_sha256": sha256(summary_path),
        "completion": str(completion_path.resolve()),
        "completion_sha256": sha256(completion_path),
        "checks": checks,
    }


def claim_one_shot_test(checkpoint: Path, output: Path):
    canonical = checkpoint / "test_scores.jsonl"
    if output.resolve() != canonical.resolve():
        raise RuntimeError(f"test output must be canonical: {canonical}")
    claim = checkpoint / "test_inference.claim.json"
    payload = {
        "checkpoint": str(checkpoint),
        "canonical_output": str(canonical),
        "pid": os.getpid(),
        "claimed_unix_time": time.time(),
    }
    try:
        descriptor = os.open(claim, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    except FileExistsError as error:
        raise RuntimeError("one-shot test inference already claimed") from error
    with os.fdopen(descriptor, "w") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    return claim


@torch.no_grad()
def main(argv=None):
    args = parser().parse_args(argv)
    checkpoint = args.checkpoint_dir.resolve()
    meta = json.loads((checkpoint / "train_meta.json").read_text())
    if meta.get("selected_epoch") is None:
        raise SystemExit("candidate has no feasible validation checkpoint")
    corpus = meta["corpus"]
    authorization = (
        require_test_authorization(checkpoint, meta)
        if args.split == "test" else None
    )
    claim_path = (
        claim_one_shot_test(checkpoint, args.out)
        if args.split == "test" else None
    )
    anchor_path = Path(meta["anchor_checkpoint"])
    powa, cfg, _, anchor_sha = load_anchor(anchor_path, corpus, args.device)
    if anchor_sha != meta["anchor_model_sha256"]:
        raise RuntimeError("anchor checkpoint changed after training")
    model = FrozenPowaTemporalAssignment(
        powa, text_dim=cfg.text_feature_size, hidden=cfg.hid_dim
    ).to(args.device)
    rank_head_path = checkpoint / "rank_head.pth"
    model.order_head.load_state_dict(
        torch.load(rank_head_path, map_location=args.device)
    )
    model.eval()

    ids = usable_text_ids(corpus, hdata.load_split(corpus, args.split))
    dataset = PowaTestDataset(
        corpus, ids, cfg.max_seqlen, cfg.grid, "av"
    )
    loader = DataLoader(
        dataset, batch_size=1, shuffle=False, num_workers=args.num_workers
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.out.with_suffix(args.out.suffix + ".tmp")
    invariant_error = 0.0
    rows = 0
    with temporary.open("w") as handle:
        for f_v, f_a, f_t, index_map, n_seconds, video_id in loader:
            video_id = video_id[0]
            f_v = f_v[0].float().to(args.device)
            f_a = f_a[0].float().to(args.device)
            f_t = f_t[0].float().to(args.device)
            lengths = torch.full(
                (f_v.shape[0],), f_v.shape[1], dtype=torch.long,
                device=args.device,
            )
            valid = torch.ones(
                (f_v.shape[0], f_v.shape[1]), dtype=torch.bool,
                device=args.device,
            )
            output = model(f_a, f_v, f_t, lengths, valid, corpus)
            anchor_tensor = output["anchor_prob"].mean(0)
            order_tensor = (
                safe_logit(anchor_tensor) + output["order_residual"].mean(0)
            )
            index = index_map[0].numpy()
            anchor = anchor_tensor.cpu().numpy()[index]
            order = order_tensor.cpu().numpy()[index]
            direct = torch.sigmoid(order_tensor).cpu().numpy()[index]
            transported = stable_transport(anchor, order)
            if len(anchor) != int(n_seconds):
                raise RuntimeError(f"inference alignment mismatch {video_id}")
            error = float(
                np.max(np.abs(np.sort(anchor) - np.sort(transported)))
            )
            invariant_error = max(invariant_error, error)
            handle.write(
                json.dumps(
                    {
                        "video_id": video_id,
                        "n_frames": len(anchor),
                        "score_powa": anchor.tolist(),
                        "score_rank_transport": transported.tolist(),
                        "score_direct_additive": direct.tolist(),
                        "score_order_raw": order.tolist(),
                    }
                )
                + "\n"
            )
            rows += 1
    temporary.replace(args.out)
    inference_meta = {
        "corpus": corpus,
        "split": args.split,
        "candidate_checkpoint": str(checkpoint),
        "candidate_rank_head_sha256": sha256(rank_head_path),
        "anchor_checkpoint": str(anchor_path),
        "anchor_model_sha256": anchor_sha,
        "selected_epoch": meta["selected_epoch"],
        "scores_file": str(args.out.resolve()),
        "scores_sha256": sha256(args.out),
        "videos": rows,
        "per_video_score_multiset_max_abs_error": invariant_error,
        "exact_float64": invariant_error == 0.0,
        "test_labels_used_for_training_or_selection": False,
        "test_authorization": authorization,
        "test_claim": str(claim_path) if claim_path is not None else None,
        "test_claim_sha256": (
            sha256(claim_path) if claim_path is not None else None
        ),
    }
    meta_path = args.out.parent / f"{args.split}_inference_meta.json"
    meta_path.write_text(json.dumps(inference_meta, indent=2) + "\n")
    print(json.dumps(inference_meta, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
