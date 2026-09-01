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
BASE = ROOT / "scripts/reproduction_baselines/multihateloc"
COMMON = ROOT / "scripts/reproduction_baselines"
DUPLEX = ROOT / "scripts/duplex"
sys.path[:0] = [str(BASE), str(COMMON), str(DUPLEX), str(ROOT)]

import data as mdata  # noqa: E402
from hate_common import data as hdata  # noqa: E402
from frame_eval_common import average_precision, evaluate as evaluate_frames  # noqa: E402
from src.scoped_video_protocol import scoped_video_labels  # noqa: E402
from method import ARMS, WitnessPreservingTokenFusion  # noqa: E402


OFFICIAL = {
    "hatemm": {
        "lr": 1.849152228476098e-5, "max_epoch": 50, "k_proportion": 8,
        "lambda_smooth": .01420807210603241,
        "lambda_contrast": .18733857665415116,
        "hidden": 512, "embed": 64, "dropout": .05, "temperature": .07,
    },
    "hateclipseg": {
        "lr": .00018190822304650636, "max_epoch": 100, "k_proportion": 3,
        "lambda_smooth": .10337306075094418,
        "lambda_contrast": .03728675834293724,
        "hidden": 512, "embed": 256, "dropout": .05, "temperature": .03,
    },
}


def loader(corpus, ids, labels, batch_size, shuffle, workers, generator=None):
    return tdata.DataLoader(
        mdata.MultiModalDataset(corpus, ids, labels), batch_size=batch_size,
        shuffle=shuffle, drop_last=False, collate_fn=mdata.collate,
        num_workers=workers, generator=generator)


def run_epoch(model, data_loader, optimizer, args):
    model.train()
    totals, seen, witnesses = {}, 0, 0
    for feats, labels, lengths, mask, _ in data_loader:
        feats = {name: value.to(args.device, non_blocking=True)
                 for name, value in feats.items()}
        labels = labels.to(args.device)
        lengths = lengths.to(args.device)
        mask = mask.to(args.device)
        output = model(feats, mask)
        mil, per_branch = model.mil_loss(
            output["probs"], mask, lengths, labels)
        smooth = model.smoothness_loss(output["probs"], mask)
        contrast = model.contrastive_loss(output["embeds"], mask)
        gate, gate_diag = model.gate_loss(output, mask, lengths, labels)
        loss = (mil + args.lambda_smooth * smooth +
                args.lambda_contrast * contrast + args.lambda_gate * gate)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        terms = {"loss": loss, "mil": mil, "smooth": smooth,
                 "contrast": contrast, "gate": gate,
                 "gate_coverage": gate_diag["coverage"],
                 "gate_budget": gate_diag["budget"]}
        terms.update({"mil_" + name: value
                      for name, value in per_branch.items()})
        batch_size = len(labels)
        for name, value in terms.items():
            totals[name] = totals.get(name, 0.0) + float(value.detach()) * batch_size
        witnesses += gate_diag["witness_count"]
        seen += batch_size
    result = {name: value / max(seen, 1) for name, value in totals.items()}
    result["gate_witness_count"] = witnesses
    return result


@torch.no_grad()
def predict(model, data_loader, device, include_diagnostics=False):
    model.eval()
    frames, videos, labels_out = {}, {}, {}
    gate_sum = gate_count = coverage_sum = witness_count = 0.0
    for feats, labels, lengths, mask, video_ids in data_loader:
        feats = {name: value.to(device) for name, value in feats.items()}
        labels_device = labels.to(device)
        lengths_device = lengths.to(device)
        mask_device = mask.to(device)
        output = model(feats, mask_device)
        video_scores = model.video_scores(
            output["probs"], mask_device, lengths_device)
        if include_diagnostics and model.alpha_fusion > 0:
            gates = output["retain_gates"]
            gate_sum += float((gates * mask_device[..., None]).sum())
            gate_count += int(mask_device.sum()) * gates.shape[-1]
            _, diag = model.gate_loss(
                output, mask_device, lengths_device, labels_device)
            coverage_sum += float(diag["coverage"]) * diag["witness_count"]
            witness_count += diag["witness_count"]
        for index, video_id in enumerate(video_ids):
            length = int(lengths[index])
            frames[video_id] = {
                "score_" + name: probability[index, :length].cpu().numpy()
                for name, probability in output["probs"].items()}
            videos[video_id] = {
                name: float(value[index]) for name, value in video_scores.items()}
            labels_out[video_id] = int(labels[index])
    diagnostics = {
        "mean_retain_gate": gate_sum / gate_count if gate_count else None,
        "mean_witness_coverage_penalty": (
            coverage_sum / witness_count if witness_count else None),
        "witness_count": int(witness_count),
    }
    return frames, videos, labels_out, diagnostics


def validation_metrics(frame_records, video_records, labels, gold):
    ids = sorted(video_records)
    video_ap = average_precision(
        [video_records[video_id]["fused"] for video_id in ids],
        [labels[video_id] for video_id in ids])
    frame_eval = evaluate_frames(
        {video_id: (frame_records[video_id]["score_fused"], gold[video_id])
         for video_id in ids},
        macro_over={video_id for video_id in ids if labels[video_id] == 1})
    return {"video_ap": video_ap,
            "within_roc": frame_eval["per_video"]["macro_auc"],
            "pooled_ap": frame_eval["pr_auc"],
            "pooled_roc": frame_eval["roc_auc"]}


def reference_metrics(path):
    return (None if path is None else
            json.loads(Path(path).read_text())["selected_validation_metrics"])


def selection_key(metrics, reference):
    within = metrics["within_roc"]
    if within is None or not np.isfinite(within):
        return (-1, float("-inf"), float("-inf"))
    if reference is None:
        return (2, float(within),
                float(metrics["pooled_ap"] + metrics["pooled_roc"]))
    ap_drop = float(reference["pooled_ap"] - metrics["pooled_ap"])
    roc_drop = float(reference["pooled_roc"] - metrics["pooled_roc"])
    violation = max(ap_drop - .01, roc_drop - .01, 0.0)
    return ((2, float(within), -max(ap_drop, roc_drop)) if violation <= 0.0
            else (1, -violation, float(within)))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", required=True, choices=tuple(OFFICIAL))
    parser.add_argument("--arm", required=True, choices=ARMS)
    parser.add_argument("--alpha-fusion", type=float, required=True)
    parser.add_argument("--lr", type=float, required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--reference-log")
    parser.add_argument("--lambda-gate", type=float, default=.1)
    parser.add_argument("--seed", type=int, default=234)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("ABORT: CUDA unavailable")
    fixed = OFFICIAL[args.corpus]
    args.lambda_smooth = fixed["lambda_smooth"]
    args.lambda_contrast = fixed["lambda_contrast"]
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    config = {
        "date": "2026-09-01",
        "method": "witness_preserving_temporal_token_fusion",
        "code_version_description": (
            "MultiHateLoc with per-second aligned donor substitution and "
            "positive latent-witness native-carrier preservation."),
        "evaluation_split": "validation_only", "args": vars(args),
        "official_fixed_config": fixed,
    }
    (output_dir / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    train_ids = hdata.load_split(args.corpus, "train")
    val_ids = hdata.load_split(args.corpus, "val")
    if set(train_ids) & set(val_ids):
        raise RuntimeError("train and validation manifests overlap")
    train_labels = scoped_video_labels(args.corpus, "train", train_ids)
    val_labels = scoped_video_labels(args.corpus, "val", val_ids)
    val_gold = hdata.gt_arrays(args.corpus, "val")
    generator = torch.Generator().manual_seed(args.seed)
    train_data = loader(args.corpus, train_ids, train_labels,
                        args.batch_size, True, 4, generator)
    val_data = loader(args.corpus, val_ids, val_labels,
                      args.batch_size, False, 2)
    model = WitnessPreservingTokenFusion(
        {name: mdata.FEATURE_DIMS[name] for name in mdata.MODALITIES},
        alpha_fusion=args.alpha_fusion, arm=args.arm,
        hidden=fixed["hidden"], embed=fixed["embed"],
        dropout=fixed["dropout"], k_proportion=fixed["k_proportion"],
        temperature=fixed["temperature"]).to(args.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    reference = reference_metrics(args.reference_log)
    if args.arm == "anchor" and reference is not None:
        raise ValueError("anchor cannot have reference")
    if args.arm != "anchor" and reference is None:
        raise ValueError("result-relevant arm requires matched reference")
    best_key = best_epoch = best_state = best_metrics = None
    history, started = [], time.time()
    for epoch in range(1, fixed["max_epoch"] + 1):
        stats = run_epoch(model, train_data, optimizer, args)
        frames, videos, observed, diagnostics = predict(
            model, val_data, args.device, include_diagnostics=True)
        metrics = validation_metrics(frames, videos, observed, val_gold)
        key = selection_key(metrics, reference)
        stats.update({"epoch": epoch, "validation": metrics,
                      "validation_selection_key": list(key),
                      "validation_diagnostics": diagnostics,
                      "elapsed_seconds": round(time.time() - started, 1)})
        history.append(stats)
        if best_key is None or key > best_key:
            best_key, best_epoch, best_metrics = key, epoch, metrics
            best_state = {name: value.detach().cpu().clone()
                          for name, value in model.state_dict().items()}
        if epoch == 1 or epoch % 10 == 0:
            print(f"{args.corpus}/{args.arm}/a={args.alpha_fusion:g} "
                  f"epoch={epoch:03d} loss={stats['loss']:.4f} "
                  f"within={metrics['within_roc']:.4f} "
                  f"AP={metrics['pooled_ap']:.4f} ROC={metrics['pooled_roc']:.4f}",
                  flush=True)
    torch.save(best_state, output_dir / "checkpoint.pt")
    payload = {
        "corpus": args.corpus, "arm": args.arm,
        "alpha_fusion": args.alpha_fusion, "lr": args.lr,
        "selection_rule": (
            "within primary; pooled AP/ROC max drop .01 against matched "
            "reference; if infeasible minimize maximum violation then maximize within"),
        "reference_log": args.reference_log,
        "selected_epoch": best_epoch,
        "selected_validation_key": list(best_key),
        "selected_validation_metrics": best_metrics,
        "test_prediction_generated": False,
        "test_labels_used_for_training_or_selection": False,
        "history": history,
        "elapsed_seconds": round(time.time() - started, 1),
    }
    (output_dir / "train_log.json").write_text(json.dumps(payload, indent=2) + "\n")
    print(f"selected epoch {best_epoch}; validation only", flush=True)


if __name__ == "__main__":
    main()
