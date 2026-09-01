#!/usr/bin/env python3
"""Authorized inference for a selected corpus-specific residual head."""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import tempfile
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from train import (FrozenPowaContextResidual, PowaTestDataset, hdata,
                   load_corpus_powa, runtime, save_json, sha256,
                   usable_text_ids, validate, write_scores)
from summarize_stage_v import main as recompute_stage_v


HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--corpus", required=True,
                   choices=("hatemm", "hateclipseg"))
    p.add_argument("--anchor-checkpoint", required=True, type=Path)
    p.add_argument("--train-run", required=True, type=Path)
    p.add_argument("--authorization", required=True, type=Path)
    p.add_argument("--out-dir", required=True, type=Path)
    p.add_argument("--split", default="test", choices=("val", "test"))
    p.add_argument("--device", default="cuda")
    p.add_argument("--num-workers", type=int, default=4)
    return p


def authorize(path, corpus, train_run):
    canonical = train_run.parent / "stage_v_summary.json"
    if path.resolve() != canonical.resolve():
        raise PermissionError("authorization must be the canonical Stage-V summary")
    with tempfile.NamedTemporaryFile(
        dir=canonical.parent, prefix=".stage_v_recheck_", suffix=".json",
        delete=False,
    ) as handle:
        recomputed = Path(handle.name)
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            code = recompute_stage_v([
                "--run-root", str(train_run.parent),
                "--out", str(recomputed),
            ])
        if code != 0 or recomputed.read_bytes() != canonical.read_bytes():
            raise PermissionError("canonical Stage-V summary failed live recomputation")
    finally:
        recomputed.unlink(missing_ok=True)
    decision = json.loads(path.read_text())
    record = decision.get("corpora", {}).get(corpus, {}).get(
        "records", {}
    ).get("span_marginal", {})
    if not (
        decision.get("pass") is True
        and decision.get("verdict") == "ADVANCE_TO_STAGE_P"
        and decision.get("pre_run_review_verdict") == "PASS"
        and record.get("integrity_pass") is True
        and Path(record.get("run_dir", "")).resolve() == train_run.resolve()
    ):
        raise PermissionError("Stage-V summary does not authorize this run")
    return decision, record


def main(argv=None):
    args = parser().parse_args(argv)
    if args.split != "test":
        raise PermissionError("authorized inference is test-only")
    args.device = runtime.resolve_device(args.device)
    args.out_dir = args.out_dir.resolve()
    train_run = args.train_run.resolve()
    expected_out = train_run.parent / f"test_{args.corpus}_seed234"
    if args.out_dir != expected_out.resolve():
        raise PermissionError("test output must use the canonical one-shot directory")
    authorization_path = args.authorization.resolve()
    authorization, authorization_record = authorize(
        authorization_path, args.corpus, train_run
    )
    expected_source = authorization_record["snapshot_inventory"]
    current_source = {
        "experiment/train.py": HERE / "train.py",
        "experiment/infer.py": HERE / "infer.py",
        "shared/powa_residual.py": REPO / "src/powa_residual.py",
    }
    mismatched = [
        name for name, path in current_source.items()
        if expected_source.get(name) != sha256(path)
    ]
    if mismatched:
        raise RuntimeError(f"inference source differs from Stage-V: {mismatched}")
    train_meta = json.loads((train_run / "train_meta.json").read_text())
    if (train_meta.get("arm") != "span_marginal"
            or train_meta.get("selected_epoch") is None):
        raise RuntimeError("training run has no selected core checkpoint")
    if args.anchor_checkpoint.resolve() != Path(
        train_meta["anchor_checkpoint"]
    ).resolve():
        raise RuntimeError("anchor path differs from selected training run")
    powa, cfg, _, anchor_hash = load_corpus_powa(
        args.anchor_checkpoint, args.corpus, args.device
    )
    if anchor_hash != train_meta.get("anchor_model_sha256"):
        raise RuntimeError("anchor checkpoint differs from selected training run")
    if sha256(args.anchor_checkpoint.resolve() / "train_meta.json") != (
        train_meta.get("anchor_train_meta_sha256")
    ):
        raise RuntimeError("anchor configuration differs from selected training run")
    model = FrozenPowaContextResidual(
        powa, cfg.text_feature_size, cfg.hid_dim, cfg.max_seqlen
    ).to(args.device)
    state_path = train_run / "residual_head.pth"
    completion = json.loads((train_run / "completion.json").read_text())
    if sha256(state_path) != completion["residual_head_sha256"]:
        raise RuntimeError("selected residual hash differs from completion")
    model.residual_head.load_state_dict(
        torch.load(state_path, map_location=args.device)
    )
    args.out_dir.mkdir(parents=True, exist_ok=True)
    claim = args.out_dir / "test_claim.json"
    with claim.open("x") as handle:
        json.dump({
            "corpus": args.corpus,
            "authorization_sha256": sha256(authorization_path),
        }, handle, indent=2)
        handle.write("\n")
    save_json(args.out_dir / "config.json", {
        key: str(value) if isinstance(value, Path) else value
        for key, value in vars(args).items()
    })
    ids = usable_text_ids(
        args.corpus, hdata.load_split(args.corpus, args.split)
    )
    dataset = PowaTestDataset(
        args.corpus, ids, cfg.max_seqlen, cfg.grid, "av"
    )
    loader = DataLoader(
        dataset, batch_size=1, shuffle=False, num_workers=args.num_workers
    )
    result = validate(
        model, loader, args.corpus, args.device, split=args.split
    )
    rows = result.pop("score_rows")
    score_path = args.out_dir / f"{args.split}_scores.jsonl"
    write_scores(score_path, rows)
    save_json(args.out_dir / f"{args.split}_infer_meta.json", {
        "method": "powa_context_quotient_span_marginal",
        "corpus": args.corpus,
        "split": args.split,
        "train_run": str(train_run),
        "selected_epoch": train_meta["selected_epoch"],
        "anchor_model_sha256": anchor_hash,
        "residual_head_sha256": sha256(state_path),
        "authorization": str(authorization_path),
        "authorization_sha256": sha256(authorization_path),
        "authorization_verdict": authorization["verdict"],
        "n_videos": len(ids),
        "scores_sha256": sha256(score_path),
        "internal_metrics": result,
    })


if __name__ == "__main__":
    main()
