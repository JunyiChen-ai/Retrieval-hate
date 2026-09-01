#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import math
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
from model import MultiHateLoc  # noqa: E402
from hate_common import data as hdata  # noqa: E402
from frame_eval_common import average_precision, evaluate as evaluate_frames  # noqa: E402
from src.scoped_video_protocol import evaluator_test_ids, scoped_video_labels  # noqa: E402
from method import ARMS, LocalQuotientModel  # noqa: E402


def make_loader(corpus, ids, labels, batch_size, shuffle, workers, generator=None):
    return tdata.DataLoader(
        mdata.MultiModalDataset(corpus, ids, labels), batch_size=batch_size,
        shuffle=shuffle, drop_last=False, collate_fn=mdata.collate,
        num_workers=workers, generator=generator)


def dann_schedule(epoch, max_epoch):
    progress = float(epoch - 1) / float(max(max_epoch - 1, 1))
    return 2.0 / (1.0 + math.exp(-10.0 * progress)) - 1.0


def run_epoch(model, loader, optimizer, args, video_index, epoch):
    model.train()
    ramp = dann_schedule(epoch, args.max_epoch)
    if args.arm == "local_adversarial":
        model.set_grl(args.lambda_video * ramp, args.lambda_position * ramp)
    else:
        model.set_grl(0.0, 0.0)
    totals, seen = {}, 0
    for feats, labels, lengths, mask, video_ids in loader:
        feats = {name: value.to(args.device, non_blocking=True)
                 for name, value in feats.items()}
        labels = labels.to(args.device)
        lengths = lengths.to(args.device)
        mask = mask.to(args.device)
        indices = torch.tensor([video_index[v] for v in video_ids],
                               dtype=torch.long, device=args.device)
        output = model(feats, mask)
        mil, per_branch = model.mil_loss(output["probs"], mask, lengths, labels)
        smooth = model.smoothness_loss(output["probs"], mask)
        contrast = model.contrastive_loss(output["local_embeds"], mask)
        video_adv, position_adv = model.nuisance_loss(
            output, mask, lengths, indices)
        loss = (mil + args.lambda_smooth * smooth +
                args.lambda_contrast * contrast + video_adv + position_adv)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        terms = {"loss": loss, "mil": mil, "smooth": smooth,
                 "contrast": contrast, "video_adversary": video_adv,
                 "position_adversary": position_adv}
        terms.update({"mil_" + name: value for name, value in per_branch.items()})
        batch = len(labels)
        for name, value in terms.items():
            totals[name] = totals.get(name, 0.0) + float(value.detach()) * batch
        seen += batch
    totals = {name: value / max(seen, 1) for name, value in totals.items()}
    totals.update({"grl_ramp": ramp, "video_grl": model.video_grl,
                   "position_grl": model.position_grl})
    return totals


@torch.no_grad()
def predict(model, loader, device):
    model.eval()
    model.set_grl(0.0, 0.0)
    frame_records, video_records, labels_out = {}, {}, {}
    for feats, labels, lengths, mask, video_ids in loader:
        feats = {name: value.to(device) for name, value in feats.items()}
        lengths_device = lengths.to(device)
        mask = mask.to(device)
        output = model(feats, mask)
        video_scores = model.video_scores(output["probs"], mask, lengths_device)
        for index, video_id in enumerate(video_ids):
            length = int(lengths[index])
            record = {"score_" + name: probability[index, :length].cpu().numpy()
                      for name, probability in output["probs"].items()}
            record["score_local"] = torch.sigmoid(
                output["local_logit"][index, :length]).cpu().numpy()
            record["score_global"] = torch.sigmoid(
                output["global_logit"][index]).repeat(length).cpu().numpy()
            frame_records[video_id] = record
            video_records[video_id] = {
                name: float(value[index]) for name, value in video_scores.items()}
            labels_out[video_id] = int(labels[index])
    return frame_records, video_records, labels_out


def build_model(args, n_video_ids):
    dims = {name: mdata.FEATURE_DIMS[name] for name in mdata.MODALITIES}
    global_base = MultiHateLoc(
        dims, hidden=args.hidden, embed=args.embed, dropout=args.dropout,
        k_proportion=args.k_proportion, temperature=args.temperature)
    local_base = MultiHateLoc(
        dims,
        hidden=args.hidden, embed=args.embed, dropout=args.dropout,
        k_proportion=args.k_proportion, temperature=args.temperature)
    return LocalQuotientModel(
        global_base, local_base, embed=args.embed, n_video_ids=n_video_ids,
        n_position_bins=args.position_bins,
        local_scale=args.local_scale).to(args.device)


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
    parser.add_argument("--lambda-video", type=float, required=True)
    parser.add_argument("--lambda-position", type=float, required=True)
    parser.add_argument("--local-scale", type=float, required=True)
    parser.add_argument("--position-bins", type=int, default=8)
    parser.add_argument("--hidden", type=int, required=True)
    parser.add_argument("--embed", type=int, required=True)
    parser.add_argument("--dropout", type=float, required=True)
    parser.add_argument("--temperature", type=float, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--run-test", action="store_true")
    args = parser.parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("ABORT: CUDA unavailable")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "config.json").write_text(json.dumps({
        "date": "2026-09-01", "method": "local_quotient_adversarial_mil",
        "code_version_description": "independent global/local MultiHateLoc backbones; centered local raw-score channel; dual GRL nuisance suppression",
        "evaluation_split": "test" if args.run_test else "validation_only",
        "args": vars(args)}, indent=2) + "\n")

    train_ids = hdata.load_split(args.corpus, "train")
    val_ids = hdata.load_split(args.corpus, "val")
    train_labels = scoped_video_labels(args.corpus, "train", train_ids)
    val_labels = scoped_video_labels(args.corpus, "val", val_ids)
    val_gold = hdata.gt_arrays(args.corpus, "val")
    if set(train_ids) & set(val_ids):
        raise RuntimeError("train and validation manifests overlap")
    video_index = {video_id: index for index, video_id in enumerate(sorted(train_ids))}
    generator = torch.Generator().manual_seed(args.seed)
    train_loader = make_loader(args.corpus, train_ids, train_labels,
                               args.batch_size, True, 4, generator)
    val_loader = make_loader(args.corpus, val_ids, val_labels,
                             args.batch_size, False, 2)
    model = build_model(args, len(video_index))
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    best_value, best_epoch, best_state = -1.0, None, None
    history, started = [], time.time()
    for epoch in range(1, args.max_epoch + 1):
        stats = run_epoch(model, train_loader, optimizer, args,
                          video_index, epoch)
        val_frames, val_video, observed_labels = predict(
            model, val_loader, args.device)
        ids = sorted(val_video)
        val_video_ap = average_precision(
            [val_video[video_id]["fused"] for video_id in ids],
            [observed_labels[video_id] for video_id in ids])
        val_eval = evaluate_frames(
            {video_id: (val_frames[video_id]["score_fused"], val_gold[video_id])
             for video_id in ids},
            macro_over={video_id for video_id in ids
                        if observed_labels[video_id] == 1})
        val_within = val_eval["per_video"]["macro_auc"]
        stats.update({"epoch": epoch, "val_video_ap": val_video_ap,
                      "val_within_roc": val_within,
                      "val_pooled_ap": val_eval["pr_auc"],
                      "val_pooled_roc": val_eval["roc_auc"],
                      "elapsed_seconds": round(time.time() - started, 1)})
        history.append(stats)
        if val_within is not None and val_within > best_value:
            best_value, best_epoch = val_within, epoch
            best_state = {name: value.detach().cpu().clone()
                          for name, value in model.state_dict().items()}
        if epoch == 1 or epoch % 10 == 0:
            print(f"{args.corpus}/{args.arm} epoch {epoch:03d} "
                  f"loss={stats['loss']:.4f} video_adv={stats['video_adversary']:.4f} "
                  f"position_adv={stats['position_adversary']:.4f} "
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
                record.update({name: [float(x) for x in values]
                               for name, values in frames[video_id].items()})
                handle.write(json.dumps(record) + "\n")
    torch.save(best_state, output_dir / "checkpoint.pt")
    (output_dir / "train_log.json").write_text(json.dumps({
        "corpus": args.corpus, "arm": args.arm,
        "selection_metric": "validation_within_roc",
        "selected_epoch": best_epoch,
        "selected_validation_metric": best_value,
        "test_prediction_generated_immediately_after_selection": args.run_test,
        "test_labels_used_for_training_or_checkpoint_selection": False,
        "test_video_diagnostics": test_video, "history": history,
        "elapsed_seconds": round(time.time() - started, 1)}, indent=2) + "\n")
    print(f"selected epoch {best_epoch}; run_test={args.run_test}", flush=True)


if __name__ == "__main__":
    main()
