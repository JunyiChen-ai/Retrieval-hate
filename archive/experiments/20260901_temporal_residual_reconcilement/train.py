#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.utils.data as tdata

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "scripts" / "reproduction_baselines" / "multihateloc"
COMMON = ROOT / "scripts" / "reproduction_baselines"
DUPLEX = ROOT / "scripts" / "duplex"
sys.path.insert(0, str(BASE))
sys.path.insert(0, str(COMMON))
sys.path.insert(0, str(DUPLEX))
sys.path.insert(0, str(ROOT))

import data as mdata  # noqa: E402
from hate_common import data as hdata  # noqa: E402
from frame_eval_common import evaluate as evaluate_frames  # noqa: E402
from src.scoped_video_protocol import evaluator_test_ids, scoped_video_labels  # noqa: E402
from method_model import ARMS, TemporalResidualModel, configure_training  # noqa: E402


def average_precision(y_true, y_score):
    y_true = np.asarray(y_true, dtype=float)
    order = np.argsort(-np.asarray(y_score, dtype=float), kind="mergesort")
    y = y_true[order]
    tp = np.cumsum(y)
    fp = np.cumsum(1.0 - y)
    if y.sum() == 0:
        return float("nan")
    precision = tp / np.maximum(tp + fp, 1e-12)
    recall = tp / y.sum()
    return float(np.sum(np.diff(np.concatenate([[0.0], recall])) * precision))


def make_loader(corpus, ids, labels, batch_size, shuffle, workers, generator=None):
    return tdata.DataLoader(
        mdata.MultiModalDataset(corpus, ids, labels), batch_size=batch_size,
        shuffle=shuffle, drop_last=False, collate_fn=mdata.collate,
        num_workers=workers, generator=generator)


def run_epoch(model, loader, optimizer, args, active):
    model.train()
    configure_training(model, active)
    totals, seen = {}, 0
    for feats, labels, lengths, mask, _ in loader:
        feats = {name: value.to(args.device, non_blocking=True)
                 for name, value in feats.items()}
        labels = labels.to(args.device)
        lengths = lengths.to(args.device)
        mask = mask.to(args.device)
        output = model(feats, mask)
        bag, per_branch = model.bag_losses(output["probs"], mask, lengths, labels)
        smooth = model.smoothness_loss(output["probs"], mask)
        contrast = model.contrastive_loss(output["embeds"], mask)
        residual = (model.temporal_residual_loss(
            output, mask, lengths, labels, active)
                    if args.arm == "temporal_residual" else bag * 0.0)
        loss = (bag + args.lambda_residual * residual +
                args.lambda_smooth * smooth + args.lambda_contrast * contrast)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        terms = {"loss": loss, "bag": bag, "residual": residual,
                 "smooth": smooth, "contrast": contrast}
        terms.update({"bag_" + name: value for name, value in per_branch.items()})
        batch = len(labels)
        for name, value in terms.items():
            totals[name] = totals.get(name, 0.0) + float(value.detach()) * batch
        seen += batch
    return {name: value / max(seen, 1) for name, value in totals.items()}


@torch.no_grad()
def predict(model, loader, device):
    model.eval()
    frame_records, video_records, labels_out = {}, {}, {}
    for feats, labels, lengths, mask, video_ids in loader:
        feats = {name: value.to(device) for name, value in feats.items()}
        lengths_device = lengths.to(device)
        mask = mask.to(device)
        output = model(feats, mask)
        video_scores = model.video_scores(output["probs"], mask, lengths_device)
        for index, video_id in enumerate(video_ids):
            length = int(lengths[index])
            frame_records[video_id] = {
                "score_" + name: prob[index, :length].cpu().numpy()
                for name, prob in output["probs"].items()}
            video_records[video_id] = {
                name: float(value[index]) for name, value in video_scores.items()}
            labels_out[video_id] = int(labels[index])
    return frame_records, video_records, labels_out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", required=True, choices=("hatemm", "hateclipseg"))
    parser.add_argument("--arm", required=True, choices=ARMS)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seed", type=int, default=234)
    parser.add_argument("--lr", type=float, required=True)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-epoch", type=int, required=True)
    parser.add_argument("--k-proportion", type=int, required=True)
    parser.add_argument("--lambda-smooth", type=float, required=True)
    parser.add_argument("--lambda-contrast", type=float, required=True)
    parser.add_argument("--lambda-residual", type=float, default=1.0)
    parser.add_argument("--hidden", type=int, required=True)
    parser.add_argument("--embed", type=int, required=True)
    parser.add_argument("--dropout", type=float, required=True)
    parser.add_argument("--temperature", type=float, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--select-metric", default="within_roc",
                        choices=("within_roc", "video_ap"))
    parser.add_argument("--run-test", action="store_true")
    args = parser.parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("ABORT: CUDA unavailable")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "config.json").write_text(json.dumps({
        "date": "2026-09-01", "method": "temporal_residual_reconcilement",
        "evaluation_split": "test", "args": vars(args)}, indent=2) + "\n")

    train_ids = hdata.load_split(args.corpus, "train")
    val_ids = hdata.load_split(args.corpus, "val")
    train_labels = scoped_video_labels(args.corpus, "train", train_ids)
    val_labels = scoped_video_labels(args.corpus, "val", val_ids)
    val_gold = hdata.gt_arrays(args.corpus, "val")
    if set(train_ids) & set(val_ids):
        raise RuntimeError("train and validation manifests overlap")
    generator = torch.Generator().manual_seed(args.seed)
    train_loader = make_loader(args.corpus, train_ids, train_labels,
                               args.batch_size, True, 4, generator)
    val_loader = make_loader(args.corpus, val_ids, val_labels,
                             args.batch_size, False, 2)
    model = TemporalResidualModel(
        {name: mdata.FEATURE_DIMS[name] for name in mdata.MODALITIES},
        args.hidden, args.embed, args.dropout, args.k_proportion,
        args.temperature).to(args.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    best_ap, best_epoch, best_state = -1.0, None, None
    history, started = [], time.time()
    order = tuple(model.modalities)
    for epoch in range(1, args.max_epoch + 1):
        active = order[(epoch - 1) % len(order)]
        stats = run_epoch(model, train_loader, optimizer, args, active)
        val_frames, video, video_labels = predict(model, val_loader, args.device)
        ids = sorted(video)
        val_ap = average_precision([video_labels[v] for v in ids],
                                   [video[v]["fused"] for v in ids])
        val_localization = evaluate_frames(
            {video_id: (val_frames[video_id]["score_fused"], val_gold[video_id])
             for video_id in ids},
            macro_over={video_id for video_id in ids if video_labels[video_id] == 1})
        val_within = val_localization["per_video"]["macro_auc"]
        selection_value = val_within if args.select_metric == "within_roc" else val_ap
        stats.update({"epoch": epoch, "active_modality": active,
                      "val_video_ap": val_ap,
                      "val_within_roc": val_within,
                      "elapsed_seconds": round(time.time() - started, 1)})
        history.append(stats)
        cycle_complete = epoch % len(order) == 0
        if (cycle_complete and selection_value is not None and
                selection_value == selection_value and selection_value > best_ap):
            best_ap, best_epoch = selection_value, epoch
            best_state = {name: value.detach().cpu().clone()
                          for name, value in model.state_dict().items()}
        if epoch == 1 or epoch % 10 == 0:
            print(f"{args.corpus}/{args.arm} epoch {epoch:03d} active={active} "
                  f"loss={stats['loss']:.4f} val_ap={val_ap:.4f} "
                  f"val_within={val_within:.4f}", flush=True)
    if best_state is None:
        raise RuntimeError("validation checkpoint selection produced no state")
    model.load_state_dict(best_state)
    test_video = {}
    if args.run_test:
        test_ids = evaluator_test_ids(
            args.corpus, hdata.load_split(args.corpus, "test"))
        test_dummy_labels = {video_id: 0 for video_id in test_ids}
        test_loader = make_loader(args.corpus, test_ids, test_dummy_labels,
                                  args.batch_size, False, 2)
        frames, test_video, _ = predict(model, test_loader, args.device)
        with (output_dir / "scores.jsonl").open("w", encoding="utf-8") as handle:
            for video_id in test_ids:
                record = {"video_id": video_id}
                record.update({name: [float(x) for x in value]
                               for name, value in frames[video_id].items()})
                handle.write(json.dumps(record) + "\n")
    torch.save(best_state, output_dir / "checkpoint.pt")
    (output_dir / "train_log.json").write_text(json.dumps({
        "corpus": args.corpus, "arm": args.arm,
        "selected_epoch": best_epoch,
        "selection_metric": args.select_metric,
        "selected_validation_metric": best_ap,
        "test_prediction_generated_immediately_after_selection": args.run_test,
        "test_labels_used_for_gradient_or_checkpoint_selection": False,
        "test_video_diagnostics": test_video, "history": history,
        "elapsed_seconds": round(time.time() - started, 1)}, indent=2) + "\n")
    print(f"selected epoch {best_epoch}; run_test={args.run_test}", flush=True)


if __name__ == "__main__":
    main()
