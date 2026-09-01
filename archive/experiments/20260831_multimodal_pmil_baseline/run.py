"""Train and infer a corpus-specific multimodal P-MIL baseline port."""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import average_precision_score


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
BASELINES = REPO / "scripts/reproduction_baselines"
MULTIHATELOC = BASELINES / "multihateloc"
for path in (REPO, BASELINES):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from hate_common import data as hdata  # noqa: E402
from scripts.reproduction_baselines.multihateloc import data as mdata  # noqa: E402
from scripts.reproduction_baselines.multihateloc.model import MultiHateLoc  # noqa: E402
from src.scoped_video_protocol import evaluator_test_ids, scoped_video_labels  # noqa: E402

MODEL_SPEC = importlib.util.spec_from_file_location("pmil_port_model", HERE / "model.py")
PMIL_MODEL = importlib.util.module_from_spec(MODEL_SPEC)
assert MODEL_SPEC.loader is not None
MODEL_SPEC.loader.exec_module(PMIL_MODEL)
MultimodalPMIL = PMIL_MODEL.MultimodalPMIL


SOURCE_ROOT = Path("/home/jehc223/Hate-follow-up/results/reproduction/official_val/final")
SOURCE_URL = (
    "https://openaccess.thecvf.com/content/CVPR2023/html/"
    "Ren_Proposal-Based_Multiple_Instance_Learning_for_Weakly-Supervised_"
    "Temporal_Action_Localization_CVPR_2023_paper.html"
)
MODALITIES = tuple(mdata.MODALITIES)


def source_checkpoint(corpus, seed):
    return SOURCE_ROOT / "multihateloc" / corpus / f"seed_{seed}" / corpus / "model.pt"


def load_features(corpus, video_id, device):
    features = {}
    length = None
    for modality in MODALITIES:
        rows = np.load(mdata.feature_path(modality, corpus, video_id)).astype(np.float32)
        expected_dim = mdata.FEATURE_DIMS[modality]
        if rows.ndim != 2 or rows.shape[1] != expected_dim or len(rows) == 0:
            raise RuntimeError(
                f"invalid {modality} feature shape for {corpus}/{video_id}: "
                f"{rows.shape}, expected non-empty (T, {expected_dim})"
            )
        if not np.isfinite(rows).all():
            raise RuntimeError(f"non-finite {modality} features for {corpus}/{video_id}")
        if length is None:
            length = len(rows)
        elif len(rows) != length:
            raise RuntimeError(f"unaligned feature length for {corpus}/{video_id}")
        features[modality] = torch.from_numpy(rows).to(device)
    return features, int(length)


@torch.no_grad()
def base_frame_scores(model, features):
    length = len(next(iter(features.values())))
    batch = {name: rows[None] for name, rows in features.items()}
    mask = torch.ones((1, length), dtype=torch.bool, device=next(model.parameters()).device)
    return model(batch, mask)["probs"]["fused"][0].detach().cpu().numpy()


def contiguous_components(mask):
    padded = np.pad(np.asarray(mask, dtype=np.int8), (1, 1))
    changes = np.diff(padded)
    return list(zip(np.where(changes == 1)[0], np.where(changes == -1)[0]))


def generate_proposals(score, maximum=256):
    score = np.asarray(score, dtype=np.float64)
    if score.ndim != 1 or not len(score) or not np.isfinite(score).all():
        raise RuntimeError("invalid S-MIL score sequence")
    low, high = float(score.min()), float(score.max())
    proposals = {(0, len(score))}
    for level in np.linspace(0.1, 0.9, 9):
        threshold = low + level * (high - low)
        proposals.update(contiguous_components(score >= threshold))
    peak_count = min(16, len(score))
    peaks = np.argsort(-score, kind="stable")[:peak_count]
    for peak in peaks:
        for width in (1, 2, 4, 8, 16, 32, 64):
            width = min(width, len(score))
            start = min(max(0, int(peak) - width // 2), len(score) - width)
            proposals.add((start, start + width))
    proposals = [bound for bound in proposals if bound[1] > bound[0]]

    def priority(bound):
        start, end = bound
        values = score[start:end]
        return (float(values.max()), float(values.mean()), -(end - start), -start)

    proposals.sort(key=priority, reverse=True)
    if (0, len(score)) not in proposals[:maximum]:
        proposals = proposals[:maximum - 1] + [(0, len(score))]
    else:
        proposals = proposals[:maximum]
    return np.asarray(sorted(set(proposals)), dtype=np.float32)


def proposal_to_frames(length, proposals, scores):
    proposals = np.asarray(proposals)
    scores = np.asarray(scores, dtype=np.float64)
    if length <= 0 or proposals.ndim != 2 or proposals.shape[1] != 2:
        raise RuntimeError("invalid proposal readout shape")
    if len(proposals) != len(scores) or scores.ndim != 1:
        raise RuntimeError("proposal/score count mismatch")
    if not np.isfinite(proposals).all() or not np.isfinite(scores).all():
        raise RuntimeError("non-finite proposal readout input")
    if not np.equal(proposals, np.floor(proposals)).all():
        raise RuntimeError("proposal bounds must lie on the 1 fps integer grid")
    integer_bounds = proposals.astype(np.int64)
    if np.any(integer_bounds[:, 0] < 0) or np.any(integer_bounds[:, 1] > length):
        raise RuntimeError("proposal bounds lie outside the video")
    if np.any(integer_bounds[:, 1] <= integer_bounds[:, 0]):
        raise RuntimeError("proposal bounds must be non-empty")
    frames = np.full(length, -np.inf, dtype=np.float64)
    for (start, end), score in zip(integer_bounds, scores):
        frames[start:end] = np.maximum(frames[start:end], float(score))
    if not np.isfinite(frames).all():
        raise RuntimeError("proposal readout did not cover every second")
    return frames


@torch.no_grad()
def predict_one(model, features, proposals):
    model.eval()
    outputs, used = model(features, proposals, training_sample=False)
    proposal_score, video_score = model.scores(outputs)
    length = len(next(iter(features.values())))
    frames = proposal_to_frames(
        length, used.detach().cpu().numpy(), proposal_score.detach().cpu().numpy()
    )
    return frames, float(video_score), proposal_score.detach().cpu().numpy()


def validation_ap(model, corpus, video_ids, labels, proposals, device):
    truth, score = [], []
    for video_id in video_ids:
        features, _ = load_features(corpus, video_id, device)
        _, video_score, _ = predict_one(
            model, features, torch.from_numpy(proposals[video_id]).to(device)
        )
        truth.append(labels[video_id])
        score.append(video_score)
    return float(average_precision_score(truth, score))


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", required=True, choices=("hatemm", "hateclipseg"))
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--seed", type=int, default=234)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--batch-size", type=int, default=10)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args(argv)

    if args.batch_size <= 0:
        raise ValueError("batch size must be positive")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is required by the frozen pilot config")
    device = torch.device(args.device)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    train_ids = hdata.load_split(args.corpus, "train")
    val_ids = hdata.load_split(args.corpus, "val")
    train_labels = scoped_video_labels(args.corpus, "train", train_ids)
    val_labels = scoped_video_labels(args.corpus, "val", val_ids)
    labels = {**train_labels, **val_labels}
    if len(labels) != len(train_labels) + len(val_labels):
        raise RuntimeError("official train and validation manifests overlap")
    fit_ids = list(train_ids)
    test_ids = evaluator_test_ids(args.corpus, hdata.load_split(args.corpus, "test"))

    checkpoint = source_checkpoint(args.corpus, args.seed)
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    base_state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    source_hidden = int(base_state["branches.visual.proj.0.weight"].shape[0])
    source_embed = int(base_state["branches.visual.proj.3.weight"].shape[0])
    base = MultiHateLoc(
        {name: mdata.FEATURE_DIMS[name] for name in MODALITIES},
        hidden=source_hidden, embed=source_embed, dropout=0.05,
        k_proportion=3, temperature=0.07,
    ).to(device)
    base.load_state_dict(base_state)
    base.eval()

    producer_ids = fit_ids + val_ids
    proposals = {}
    proposal_counts = {}
    started = time.time()
    for index, video_id in enumerate(producer_ids, 1):
        features, _ = load_features(args.corpus, video_id, device)
        proposals[video_id] = generate_proposals(base_frame_scores(base, features))
        proposal_counts[video_id] = len(proposals[video_id])
        if index % 100 == 0:
            print(f"train/validation proposal generation {index}/{len(producer_ids)}", flush=True)
    base.to("cpu")
    torch.cuda.empty_cache()

    config = {
        "corpus": args.corpus,
        "seed": args.seed,
        "epochs": args.epochs,
        "validation_manifest": "official frozen validation split",
        "learning_rate": args.lr,
        "bag_batch_size": args.batch_size,
        "device": args.device,
        "source_method": "P-MIL, CVPR 2023",
        "source_url": SOURCE_URL,
        "source_proposal_model": "corpus-specific MultiHateLoc seed 234 checkpoint",
        "source_checkpoint_path": str(checkpoint),
        "source_model_hidden": source_hidden,
        "source_model_embed": source_embed,
        "proposal_thresholds": [round(float(x), 1) for x in np.linspace(0.1, 0.9, 9)],
        "proposal_peak_count": 16,
        "proposal_peak_widths_seconds": [1, 2, 4, 8, 16, 32, 64],
        "maximum_generated_proposals": 256,
        "maximum_sampled_train_proposals": 128,
        "roi_size": 12,
        "hidden": 128,
        "topk_divisor": 8,
        "completeness_gamma": 0.8,
        "completeness_loss_weight": 20.0,
        "rank_consistency_loss_weight": 2.0,
        "rampup_epochs": 10,
        "frame_readout": "maximum score over covering proposals",
        "test_labels_or_temporal_gt_loaded_by_producer": False,
        "code_version": "2026-08-31 tri-modal binary P-MIL baseline port",
    }
    (out_dir / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    model = MultimodalPMIL(
        {name: mdata.FEATURE_DIMS[name] for name in MODALITIES},
        hidden=128, roi_size=12, dropout=0.1, max_train_proposals=128,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    rng = np.random.default_rng(args.seed)
    best_ap = -math.inf
    best_epoch = None
    best_state = None
    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        order = np.asarray(fit_ids, dtype=object)
        rng.shuffle(order)
        sums = {}
        for offset in range(0, len(order), args.batch_size):
            batch_ids = order[offset:offset + args.batch_size]
            optimizer.zero_grad(set_to_none=True)
            for video_id in batch_ids:
                features, _ = load_features(args.corpus, str(video_id), device)
                bounds = torch.from_numpy(proposals[str(video_id)]).to(device)
                outputs, used = model(features, bounds, training_sample=True)
                terms = model.loss(outputs, used, labels[str(video_id)], epoch)
                (terms["total"] / len(batch_ids)).backward()
                for name, value in terms.items():
                    sums[name] = sums.get(name, 0.0) + float(value.detach())
            optimizer.step()
        val_ap = validation_ap(model, args.corpus, val_ids, labels, proposals, device)
        row = {name: value / len(fit_ids) for name, value in sums.items()}
        row.update({"epoch": epoch, "validation_video_ap": val_ap})
        history.append(row)
        print(json.dumps(row), flush=True)
        if val_ap > best_ap:
            best_ap = val_ap
            best_epoch = epoch
            best_state = {
                name: value.detach().cpu().clone()
                for name, value in model.state_dict().items()
            }
    if best_state is None:
        raise RuntimeError("validation checkpoint selection failed")
    model.load_state_dict(best_state)
    torch.save(best_state, out_dir / "model.pt")

    base.to(device)
    base.eval()
    score_path = out_dir / "scores.jsonl"
    inference = {}
    with score_path.open("w") as handle:
        for index, video_id in enumerate(test_ids, 1):
            features, _ = load_features(args.corpus, video_id, device)
            test_proposals = generate_proposals(base_frame_scores(base, features))
            proposal_counts[video_id] = len(test_proposals)
            frames, video_score, proposal_score = predict_one(
                model, features, torch.from_numpy(test_proposals).to(device)
            )
            inference[video_id] = {
                "video_score": video_score,
                "proposal_score_min": float(proposal_score.min()),
                "proposal_score_mean": float(proposal_score.mean()),
                "proposal_score_max": float(proposal_score.max()),
            }
            handle.write(json.dumps({
                "video_id": video_id,
                "score_pmil": frames.tolist(),
            }) + "\n")
            if index % 50 == 0:
                print(f"test blind inference {index}/{len(test_ids)}", flush=True)
    del base
    (out_dir / "proposal_diagnostics.json").write_text(json.dumps({
        "n_fit": len(fit_ids),
        "n_validation": len(val_ids),
        "n_test": len(test_ids),
        "proposal_count_min": min(proposal_counts.values()),
        "proposal_count_median": float(np.median(list(proposal_counts.values()))),
        "proposal_count_max": max(proposal_counts.values()),
        "per_video": proposal_counts,
    }, indent=2) + "\n")
    (out_dir / "train_log.json").write_text(json.dumps({
        "selected_epoch": best_epoch,
        "selected_validation_video_ap": best_ap,
        "history": history,
        "test_blind_summary": inference,
        "wall_seconds": time.time() - started,
    }, indent=2) + "\n")
    print(json.dumps({
        "corpus": args.corpus,
        "selected_epoch": best_epoch,
        "scores": str(score_path),
        "wall_seconds": time.time() - started,
    }), flush=True)


if __name__ == "__main__":
    main()
