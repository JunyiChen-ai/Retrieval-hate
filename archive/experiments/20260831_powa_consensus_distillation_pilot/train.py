#!/usr/bin/env python3
"""Train and immediately test a frozen-POWA ordinal residual student."""

from __future__ import annotations

import argparse
import copy
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import rankdata
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader


HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
BASE = REPO / "scripts/reproduction_baselines"
sys.path.insert(0, str(BASE))
sys.path.insert(0, str(REPO))

from eval_baseline_scores import evaluate_scores  # noqa: E402
from hate_common import data as hdata  # noqa: E402
from hate_common import runtime  # noqa: E402
from macilsd import align  # noqa: E402
from macilsd.train import _seq_len_of  # noqa: E402
from macilsd.utils import process_feat  # noqa: E402
from powa_macil.dataset import (PowaTestDataset, PowaTrainDataset,
                                usable_text_ids)  # noqa: E402
from src.powa_residual import (FrozenPowaContextResidual,
                               load_corpus_powa, safe_logit)  # noqa: E402


ARMS = ("no_teacher", "shuffled", "audio", "vera", "audio_vera")


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--corpus", required=True,
                   choices=("hatemm", "hateclipseg"))
    p.add_argument("--arm", required=True, choices=ARMS)
    p.add_argument("--checkpoint-dir", required=True, type=Path)
    p.add_argument("--vera-root", required=True, type=Path)
    p.add_argument("--out-dir", required=True, type=Path)
    p.add_argument("--seed", type=int, default=234)
    p.add_argument("--device", default="cuda")
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--batch-size", type=int, default=24)
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--pair-gap", type=float, default=.20)
    p.add_argument("--pair-margin", type=float, default=.25)
    p.add_argument("--anchor-weight", type=float, default=.5)
    p.add_argument("--mil-weight", type=float, default=.5)
    p.add_argument("--max-audio-rows", type=int, default=200)
    p.add_argument("--audio-epochs", type=int, default=5)
    return p


def percentile(values):
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 1 or not len(values):
        raise ValueError("percentile expects a nonempty vector")
    if len(values) == 1:
        return np.zeros(1, dtype=np.float64)
    return (rankdata(values, method="average") - 1.0) / (len(values) - 1.0)


def normalize(rows):
    rows = np.asarray(rows, dtype=np.float32)
    return rows / np.maximum(np.linalg.norm(rows, axis=1, keepdims=True), 1e-6)


def load_vera_order(raw_root, corpus, video_id):
    path = raw_root / f"{video_id}.json"
    if not path.is_file():
        raise FileNotFoundError(f"missing VERA teacher: {corpus}/{video_id}")
    row = json.loads(path.read_text())
    if row.get("video_id") != video_id:
        raise ValueError(f"VERA video identity mismatch: {video_id}")
    segments = row.get("segments", [])
    starts = np.asarray([float(item["start"]) for item in segments])
    scores = np.asarray([float(item["score"]) for item in segments])
    if (not len(starts) or np.any(np.diff(starts) <= 0)
            or not np.isin(scores, [0.0, 1.0]).all()):
        raise ValueError(f"invalid VERA teacher rows: {corpus}/{video_id}")
    snippet = align.snippet_bounds(corpus, video_id)
    midpoint = snippet.mean(1)
    return percentile(np.interp(midpoint, starts, scores))


def audio_rows(corpus, video_id, max_rows=None):
    rows, _, _ = align.aligned_audio(corpus, video_id, "snippet")
    rows = normalize(rows)
    if max_rows and len(rows) > max_rows:
        index = np.linspace(0, len(rows) - 1, max_rows, dtype=np.uint16)
        rows = rows[index.astype(np.int64)]
    return rows


def fit_audio_model(corpus, train_ids, labels, args):
    model = SGDClassifier(
        loss="log_loss", penalty="l2", alpha=1e-4, average=True,
        random_state=args.seed,
    )
    counts = {value: sum(labels[v] == value for v in train_ids)
              for value in (0, 1)}
    rng = np.random.default_rng(args.seed)
    cache = {video_id: audio_rows(corpus, video_id, args.max_audio_rows)
             for video_id in train_ids}
    for _ in range(args.audio_epochs):
        order = np.asarray(train_ids, dtype=object)
        rng.shuffle(order)
        for video_id in order:
            value = int(labels[str(video_id)])
            rows = cache[str(video_id)]
            target = np.full(len(rows), value, dtype=np.int64)
            weight = np.full(len(rows), len(train_ids) / (2.0 * counts[value]))
            model.partial_fit(rows, target, classes=np.asarray([0, 1]),
                              sample_weight=weight)
    return model


def oof_audio_orders(corpus, video_ids, labels, args):
    y = np.asarray([labels[v] for v in video_ids], dtype=np.int64)
    folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=args.seed)
    output = {}
    for fold, (fit_index, held_index) in enumerate(folds.split(video_ids, y), 1):
        fit_ids = [video_ids[i] for i in fit_index]
        model = fit_audio_model(corpus, fit_ids, labels, args)
        for i in held_index:
            video_id = video_ids[i]
            score = model.decision_function(audio_rows(corpus, video_id))
            output[video_id] = percentile(score)
        print(json.dumps({"stage": "audio_oof", "fold": fold,
                          "held_videos": len(held_index)}), flush=True)
    if set(output) != set(video_ids):
        raise RuntimeError("OOF audio coverage mismatch")
    return output


def nearest_gap_pairs(order, gap):
    """At most one higher and one lower local pair per anchor position."""
    order = np.asarray(order, dtype=np.float64)
    pairs = set()
    position = np.arange(len(order))
    for i in range(len(order)):
        higher = position[order >= order[i] + gap]
        lower = position[order <= order[i] - gap]
        if len(higher):
            j = int(higher[np.argmin(np.abs(higher - i))])
            pairs.add((j, i))
        if len(lower):
            j = int(lower[np.argmin(np.abs(lower - i))])
            pairs.add((i, j))
    return sorted(pairs)


def orient_pairs(endpoints, order):
    output = []
    for a, b in endpoints:
        if order[a] > order[b]:
            output.append((a, b))
        elif order[b] > order[a]:
            output.append((b, a))
        # A source tie has no ordinal preference. Dropping it avoids turning
        # ties into an accidental earlier/later positional teacher.
    return output


def process_order(values, max_seqlen, valid_length):
    values = process_feat(np.asarray(values)[:, None], max_seqlen,
                          is_random=False)[:, 0]
    return np.asarray(values[:valid_length], dtype=np.float32)


def fixed_context_tensors(f_v, f_a, f_t, max_seqlen):
    """Match PowaTrainDataset's uniform-200/pad grid at inference."""
    length = f_v.shape[1]
    if length > max_seqlen:
        index = np.linspace(0, length - 1, max_seqlen,
                            dtype=np.uint16).astype(np.int64)
        tensor_index = torch.as_tensor(index, device=f_v.device)
        return (f_v.index_select(1, tensor_index),
                f_a.index_select(1, tensor_index),
                f_t.index_select(1, tensor_index), index, max_seqlen)
    pad = max_seqlen - length
    return (F.pad(f_v, (0, 0, 0, pad)),
            F.pad(f_a, (0, 0, 0, pad)),
            F.pad(f_t, (0, 0, 0, pad)),
            np.arange(length, dtype=np.int64), length)


def lift_residual(residual, native_length, index):
    """Lift to the native grid and restore the per-crop zero-mean constraint."""
    residual = np.asarray(residual, dtype=np.float64)
    if native_length <= residual.shape[1]:
        lifted = residual[:, :native_length]
    else:
        target = np.arange(native_length)
        lifted = np.stack([
            np.interp(target, index, row[:len(index)]) for row in residual
        ])
    return lifted - lifted.mean(axis=1, keepdims=True)


def build_teacher(corpus, video_ids, labels, raw_root, max_seqlen, args):
    audio = oof_audio_orders(corpus, video_ids, labels, args)
    vera = {v: load_vera_order(raw_root, corpus, v) for v in video_ids}
    rng = np.random.default_rng(args.seed)
    records, pair_counts = {}, []
    for video_id in video_ids:
        n_rows = len(audio[video_id])
        valid_length = min(n_rows, max_seqlen)
        audio_order = process_order(audio[video_id], max_seqlen, valid_length)
        vera_order = process_order(vera[video_id], max_seqlen, valid_length)
        core_order = .5 * audio_order + .5 * vera_order
        endpoints = nearest_gap_pairs(core_order, args.pair_gap)
        if args.arm == "audio":
            direction = audio_order
        elif args.arm == "vera":
            direction = vera_order
        elif args.arm == "shuffled":
            direction = core_order[rng.permutation(len(core_order))]
        else:
            direction = core_order
        pairs = orient_pairs(endpoints, direction)
        records[video_id] = pairs
        pair_counts.append(len(pairs))
    return records, {
        "videos": len(video_ids), "pairs_total": int(sum(pair_counts)),
        "pairs_median": float(np.median(pair_counts)),
        "videos_without_pairs": int(sum(value == 0 for value in pair_counts)),
    }


class OrdinalTrainDataset(PowaTrainDataset):
    def __init__(self, *args, pair_records, **kwargs):
        super().__init__(*args, **kwargs)
        self.pair_records = pair_records
        self.max_pairs = 2 * self.max_seqlen

    def __getitem__(self, index):
        base = super().__getitem__(index)
        video_id = self.video_ids[index // self.crop_repeat]
        pairs = self.pair_records[video_id]
        if len(pairs) > self.max_pairs:
            raise RuntimeError(f"too many pairs: {video_id}")
        high = np.full(self.max_pairs, -1, dtype=np.int64)
        low = np.full(self.max_pairs, -1, dtype=np.int64)
        if pairs:
            high[:len(pairs)] = [pair[0] for pair in pairs]
            low[:len(pairs)] = [pair[1] for pair in pairs]
        return base + (torch.from_numpy(high), torch.from_numpy(low))


def pairwise_loss(logits, high, low, margin):
    active = (high >= 0) & (low >= 0)
    if not active.any():
        return logits.sum() * 0.0
    hi = logits.gather(1, high.clamp_min(0))
    lo = logits.gather(1, low.clamp_min(0))
    loss = F.softplus(float(margin) - (hi - lo))
    return (loss * active).sum() / active.sum()


def mil_probability(frame_prob, lengths):
    bags = []
    for row, length in zip(frame_prob, lengths):
        n = int(length)
        k = max(1, n // 16 + 1)
        bags.append(row[:n].topk(k).values.mean())
    return torch.stack(bags)


def metric_summary(report):
    return {
        "pooled_ap": report["pr_auc"], "pooled_roc": report["roc_auc"],
        "within_roc": report["per_video"]["macro_auc"],
        "within_n": report["per_video"]["n_videos_both_classes"],
    }


@torch.no_grad()
def score_split(model, corpus, split, cfg, device):
    ids = usable_text_ids(corpus, hdata.load_split(corpus, split))
    gt = hdata.gt_arrays(corpus, split)
    ids = [video_id for video_id in ids if video_id in gt]
    if set(ids) != set(gt):
        missing = sorted(set(gt) - set(ids))
        raise RuntimeError(
            f"incomplete {corpus}/{split} evaluation coverage: {missing[:5]}"
        )
    dataset = PowaTestDataset(corpus, ids, cfg.max_seqlen, cfg.grid, "av")
    scores, anchors = {}, {}
    model.eval()
    for i in range(len(dataset)):
        f_v, f_a, f_t, index_map, n_seconds, video_id = dataset[i]
        f_v = f_v.float().to(device)
        f_a = f_a.float().to(device)
        f_t = f_t.float().to(device)
        native_length = f_v.shape[1]
        native_lengths = torch.full(
            (f_v.shape[0],), native_length, dtype=torch.long
        )
        native_valid = torch.ones(
            (f_v.shape[0], native_length), dtype=torch.bool, device=device
        )
        dense = model.powa(
            f_a, f_v, f_t, native_lengths, native_valid, policy=corpus
        )
        ctx_v, ctx_a, ctx_t, ctx_index, ctx_length = fixed_context_tensors(
            f_v, f_a, f_t, cfg.max_seqlen
        )
        ctx_lengths = torch.full(
            (ctx_v.shape[0],), ctx_length, dtype=torch.long
        )
        ctx_valid = torch.arange(cfg.max_seqlen, device=device)[None] < ctx_lengths.to(device)[:, None]
        context = model(ctx_a, ctx_v, ctx_t, ctx_lengths, ctx_valid, corpus)
        lifted = lift_residual(
            context["residual"].cpu().numpy(), native_length, ctx_index
        )
        lifted = torch.from_numpy(lifted).to(
            device=device, dtype=dense["frame_prob"].dtype
        )
        candidate_native = torch.sigmoid(
            safe_logit(dense["frame_prob"]) + lifted
        )
        candidate = candidate_native.mean(0).cpu().numpy()[index_map]
        anchor = dense["frame_prob"].mean(0).cpu().numpy()[index_map]
        if len(candidate) != n_seconds or len(candidate) != len(gt[video_id]):
            raise RuntimeError(f"score/GT alignment mismatch: {video_id}")
        scores[video_id] = candidate
        anchors[video_id] = anchor
    labels = hdata.load_labels(corpus)
    positives = {video_id for video_id in gt if labels[video_id] == 1}
    reports = {
        "score_candidate": evaluate_scores(scores, gt, positives),
        "score_anchor": evaluate_scores(anchors, gt, positives),
    }
    return scores, anchors, reports


def write_scores(path, scores, anchors):
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w") as handle:
        for video_id in sorted(scores):
            handle.write(json.dumps({
                "video_id": video_id,
                "score_candidate": np.asarray(scores[video_id]).tolist(),
                "score_anchor": np.asarray(anchors[video_id]).tolist(),
            }) + "\n")
    os.replace(temporary, path)


def train_epoch(model, loader, optimizer, args, device):
    model.train()
    totals = {"loss": 0.0, "pair": 0.0, "anchor": 0.0, "mil": 0.0,
              "batches": 0}
    for f_v, f_a, f_t, label, high, low in loader:
        lengths = _seq_len_of(f_v)
        keep = int(lengths.max())
        f_v = f_v[:, :keep].float().to(device)
        f_a = f_a[:, :keep].float().to(device)
        f_t = f_t[:, :keep].float().to(device)
        label = label.float().to(device)
        valid = torch.arange(keep, device=device)[None] < lengths.to(device)[:, None]
        output = model(f_a, f_v, f_t, lengths, valid, args.corpus)
        candidate_logits = output["anchor_logit"] + output["residual"]
        pair = (candidate_logits.sum() * 0.0 if args.arm == "no_teacher" else
                pairwise_loss(candidate_logits, high.to(device), low.to(device),
                              args.pair_margin))
        anchor = F.smooth_l1_loss(
            output["residual"][valid],
            torch.zeros_like(output["residual"][valid]),
        )
        bag = mil_probability(output["candidate_prob"], lengths)
        mil = F.binary_cross_entropy(bag.clamp(1e-5, 1 - 1e-5), label)
        loss = pair + args.anchor_weight * anchor + args.mil_weight * mil
        if not torch.isfinite(loss):
            raise FloatingPointError("non-finite training loss")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.residual_head.parameters(), 5.0)
        optimizer.step()
        for key, value in (("loss", loss), ("pair", pair),
                           ("anchor", anchor), ("mil", mil)):
            totals[key] += float(value.detach())
        totals["batches"] += 1
    return {key: (value if key == "batches" else value / totals["batches"])
            for key, value in totals.items()}


def main(argv=None):
    args = parser().parse_args(argv)
    args.out_dir = args.out_dir.resolve()
    args.checkpoint_dir = args.checkpoint_dir.resolve()
    args.vera_root = args.vera_root.resolve()
    args.device = runtime.resolve_device(args.device)
    runtime.setup_seed(args.seed)
    allowed_existing = {"run.log", "run.pid"}
    if args.out_dir.exists():
        unexpected = {path.name for path in args.out_dir.iterdir()} - allowed_existing
        if unexpected:
            raise RuntimeError(
                f"output directory contains prior artifacts: {sorted(unexpected)}"
            )
    args.out_dir.mkdir(parents=True, exist_ok=True)
    initial_config = {
        "date": "2026-08-31", "code_version": "current working tree",
        "args": {key: str(value) if isinstance(value, Path) else value
                 for key, value in vars(args).items()},
        "status": "running",
    }
    (args.out_dir / "config.json").write_text(
        json.dumps(initial_config, indent=2) + "\n"
    )

    labels = hdata.load_labels(args.corpus)
    train_ids = usable_text_ids(args.corpus, hdata.load_split(args.corpus, "train"))
    raw_ids = {path.stem for path in args.vera_root.glob("*.json")}
    train_ids = [video_id for video_id in train_ids if video_id in raw_ids]
    if args.corpus == "hatemm" and len(train_ids) != 744:
        raise RuntimeError(f"HMM teacher coverage must be 744, got {len(train_ids)}")
    if args.corpus == "hateclipseg" and len(train_ids) != 238:
        raise RuntimeError(f"HCS teacher coverage must be 238, got {len(train_ids)}")

    powa, cfg, anchor_meta = load_corpus_powa(
        args.checkpoint_dir, args.corpus, args.device
    )
    teacher, teacher_summary = build_teacher(
        args.corpus, train_ids, labels, args.vera_root, cfg.max_seqlen, args
    )
    dataset = OrdinalTrainDataset(
        args.corpus, train_ids, labels, cfg.max_seqlen, cfg.grid, "av", 5,
        pair_records=teacher,
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True,
                        num_workers=args.num_workers, drop_last=False)
    model = FrozenPowaContextResidual(
        powa, text_dim=cfg.text_feature_size, hidden=cfg.hid_dim,
        max_seqlen=cfg.max_seqlen,
    ).to(args.device)
    optimizer = torch.optim.AdamW(
        model.residual_head.parameters(), lr=args.lr,
        weight_decay=args.weight_decay,
    )

    epoch0_scores, epoch0_anchors, epoch0 = score_split(
        model, args.corpus, "val", cfg, args.device
    )
    epoch0_candidate = metric_summary(epoch0["score_candidate"])
    epoch0_anchor = metric_summary(epoch0["score_anchor"])
    identity_error = max(
        float(np.max(np.abs(epoch0_scores[video_id] - epoch0_anchors[video_id])))
        for video_id in epoch0_scores
    )
    if identity_error > 1e-6:
        raise RuntimeError(f"zero-init identity failed: {identity_error}")

    history, checkpoints = [], []
    for epoch in range(1, args.epochs + 1):
        started = time.time()
        losses = train_epoch(model, loader, optimizer, args, args.device)
        val_scores, val_anchor, reports = score_split(
            model, args.corpus, "val", cfg, args.device
        )
        metrics = metric_summary(reports["score_candidate"])
        feasible = (
            metrics["pooled_ap"] >= epoch0_anchor["pooled_ap"] - .010
            and metrics["pooled_roc"] >= epoch0_anchor["pooled_roc"] - .010
        )
        record = {"epoch": epoch, "train": losses, "validation": metrics,
                  "checkpoint_feasible": feasible,
                  "seconds": round(time.time() - started, 2)}
        history.append(record)
        print(json.dumps(record), flush=True)
        checkpoints.append((feasible, metrics["within_roc"], epoch,
                            copy.deepcopy(model.residual_head.state_dict())))
    selected_feasible, _, selected_epoch, state = max(
        checkpoints, key=lambda item: (item[0], item[1], -item[2])
    )
    model.residual_head.load_state_dict(state)
    torch.save(state, args.out_dir / "residual_head.pth")

    test_scores, test_anchor, test_reports = score_split(
        model, args.corpus, "test", cfg, args.device
    )
    write_scores(args.out_dir / "scores.jsonl", test_scores, test_anchor)
    metrics_payload = {
        "corpus": args.corpus, "split": "test", "arm": args.arm,
        "status": "iterative_developmental_evidence",
        "test_labels_used_for_gradient_or_checkpoint_selection": False,
        "selected_epoch_from_validation": selected_epoch,
        "results": test_reports,
    }
    (args.out_dir / "metrics.json").write_text(
        json.dumps(metrics_payload, indent=2) + "\n"
    )
    config = {
        "date": "2026-08-31", "code_version": "current working tree",
        "args": {key: str(value) if isinstance(value, Path) else value
                 for key, value in vars(args).items()},
        "anchor_checkpoint": str(args.checkpoint_dir),
        "anchor_training_args": anchor_meta["args"],
        "train_ids": train_ids, "teacher_summary": teacher_summary,
        "validation_epoch0": epoch0_anchor, "history": history,
        "selected_epoch": selected_epoch,
        "selected_checkpoint_pooled_feasible": selected_feasible,
        "test_evaluation_started_immediately_after_selection": True,
    }
    (args.out_dir / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    print(json.dumps({
        "selected_epoch": selected_epoch,
        "test_candidate": metric_summary(test_reports["score_candidate"]),
        "test_anchor": metric_summary(test_reports["score_anchor"]),
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
