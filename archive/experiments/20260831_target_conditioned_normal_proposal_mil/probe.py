"""Fixed conditional-Gaussian proposal premise producer.

This producer fits only target-corpus negative-train data and never imports
temporal gold or test video labels.  It is a diagnostic, not the proposed flow
method.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge

REPO = Path(__file__).resolve().parents[2]
BASELINES = REPO / "scripts/reproduction_baselines"
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
if str(BASELINES) not in sys.path:
    sys.path.insert(0, str(BASELINES))

from hate_common import data as hdata  # noqa: E402
from src.hate_local_features import aligned_local_features  # noqa: E402
from src.scoped_video_protocol import evaluator_test_ids, scoped_video_labels  # noqa: E402


LENGTHS = (1, 2, 4, 8, 16, 32, 64, 128)
PRIMITIVE_FILE = REPO / "results/reproduction/powa_macil/semantic_prototypes.npz"
HARM_ROWS = (0, 2, 3, 4, 5)


def topic_projection(rows, basis):
    return rows - (rows @ basis) @ basis.T


def proposal_bounds(length):
    bounds = []
    for width in LENGTHS:
        if width > length:
            continue
        bounds.extend((start, start + width) for start in range(length - width + 1))
    if (0, length) not in bounds:
        bounds.append((0, length))
    return np.asarray(bounds, dtype=np.int64)


def range_means(rows, starts, ends):
    prefix = np.concatenate(
        [np.zeros((1, rows.shape[1]), dtype=np.float64), np.cumsum(rows, axis=0)],
        axis=0,
    )
    return (prefix[ends] - prefix[starts]) / (ends - starts)[:, None]


def proposal_features(topic_frames, residual_frames):
    length = len(topic_frames)
    bounds = proposal_bounds(length)
    starts, ends = bounds[:, 0], bounds[:, 1]
    widths = ends - starts
    topic = range_means(topic_frames, starts, ends)
    inside = range_means(residual_frames, starts, ends)

    prefix = np.concatenate([
        np.zeros((1, residual_frames.shape[1]), dtype=np.float64),
        np.cumsum(residual_frames, axis=0),
    ], axis=0)
    left_starts = np.maximum(0, starts - widths)
    right_ends = np.minimum(length, ends + widths)
    outside_sum = (prefix[starts] - prefix[left_starts]) + (prefix[right_ends] - prefix[ends])
    outside_count = (starts - left_starts) + (right_ends - ends)
    outside = np.zeros_like(inside)
    valid = outside_count > 0
    outside[valid] = outside_sum[valid] / outside_count[valid, None]
    residual = inside - outside
    return bounds, topic, residual


def gaussian_energy(residual, mean, variance):
    return 0.5 * np.mean(
        (residual - mean) ** 2 / variance + np.log(variance), axis=1
    )


def frame_readout(length, bounds, energy):
    total_diff = np.zeros(length + 1, dtype=np.float64)
    count_diff = np.zeros(length + 1, dtype=np.float64)
    np.add.at(total_diff, bounds[:, 0], energy)
    np.add.at(total_diff, bounds[:, 1], -energy)
    np.add.at(count_diff, bounds[:, 0], 1.0)
    np.add.at(count_diff, bounds[:, 1], -1.0)
    total = np.cumsum(total_diff[:-1])
    count = np.cumsum(count_diff[:-1])
    if np.any(count <= 0):
        raise RuntimeError("proposal grid failed to cover every second")
    return total / count


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", required=True, choices=("hatemm", "hateclipseg"))
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--seed", type=int, default=234)
    args = parser.parse_args(argv)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "config.json").write_text(json.dumps({
        "corpus": args.corpus,
        "seed": args.seed,
        "proposal_lengths_seconds": list(LENGTHS),
        "include_whole_video": True,
        "topic_pca_dim": 16,
        "residual_pca_dim": 32,
        "ridge_alpha": 1.0,
        "topic_clusters": 64,
        "support_quantile": 0.95,
        "support_gate": 0.80,
        "code_version": "2026-08-31 fixed target-conditioned Gaussian premise probe",
    }, indent=2) + "\n")

    train_ids = hdata.load_split(args.corpus, "train")
    train_labels = scoped_video_labels(args.corpus, "train", train_ids)
    negative_ids = [video_id for video_id in train_ids if train_labels[video_id] == 0]
    if len(negative_ids) < 2:
        raise RuntimeError("conditional premise requires at least two negative train videos")

    language = "en"
    with np.load(PRIMITIVE_FILE) as data:
        harm = np.asarray(data[language][list(HARM_ROWS)], dtype=np.float64)
    harm /= np.maximum(np.linalg.norm(harm, axis=1, keepdims=True), 1e-12)
    basis, _ = np.linalg.qr(harm.T, mode="reduced")

    cached = {}
    topic_frame_rows, residual_frame_rows = [], []
    for video_id in negative_ids:
        parts = aligned_local_features(args.corpus, video_id)
        topic_rows = topic_projection(np.asarray(parts["text"], dtype=np.float64), basis)
        residual_rows = np.asarray(parts["concat"], dtype=np.float64)
        cached[video_id] = (topic_rows, residual_rows)
        topic_frame_rows.append(topic_rows)
        residual_frame_rows.append(residual_rows)
    topic_frame_rows = np.concatenate(topic_frame_rows, axis=0)
    residual_frame_rows = np.concatenate(residual_frame_rows, axis=0)
    topic_pca = PCA(n_components=16, svd_solver="randomized", random_state=args.seed)
    residual_pca = PCA(n_components=32, svd_solver="randomized", random_state=args.seed)
    topic_pca.fit(topic_frame_rows)
    residual_pca.fit(residual_frame_rows)

    negative_topic, negative_residual = [], []
    for video_id in negative_ids:
        topic_rows, residual_rows = cached[video_id]
        _, topic, residual = proposal_features(
            topic_pca.transform(topic_rows), residual_pca.transform(residual_rows)
        )
        negative_topic.append(topic)
        negative_residual.append(residual)
    negative_topic = np.concatenate(negative_topic, axis=0)
    negative_residual = np.concatenate(negative_residual, axis=0)
    topic_mean = negative_topic.mean(0)
    topic_scale = np.maximum(negative_topic.std(0), 1e-6)
    negative_topic_std = (negative_topic - topic_mean) / topic_scale

    conditional = Ridge(alpha=1.0, fit_intercept=True).fit(
        negative_topic_std, negative_residual
    )
    conditional_error = negative_residual - conditional.predict(negative_topic_std)
    conditional_variance = np.maximum(conditional_error.var(0), 1e-6)
    unconditional_mean = negative_residual.mean(0)
    unconditional_variance = np.maximum(negative_residual.var(0), 1e-6)
    train_conditional = gaussian_energy(
        conditional_error, np.zeros(negative_residual.shape[1]), conditional_variance
    )
    train_unconditional = gaussian_energy(
        negative_residual, unconditional_mean, unconditional_variance
    )
    conditional_center, conditional_scale = train_conditional.mean(), max(train_conditional.std(), 1e-6)
    unconditional_center, unconditional_scale = train_unconditional.mean(), max(train_unconditional.std(), 1e-6)

    clusters = min(64, len(negative_topic_std))
    kmeans = MiniBatchKMeans(
        n_clusters=clusters, random_state=args.seed, batch_size=4096, n_init=3
    ).fit(negative_topic_std)
    negative_distance = kmeans.transform(negative_topic_std).min(1)
    support_threshold = float(np.quantile(negative_distance, 0.95))

    test_ids = evaluator_test_ids(args.corpus, hdata.load_split(args.corpus, "test"))
    score_path = out_dir / "scores.jsonl"
    supported, total_proposals = 0, 0
    per_video_support = {}
    with score_path.open("w") as handle:
        for video_id in test_ids:
            parts = aligned_local_features(args.corpus, video_id)
            topic_rows = topic_projection(np.asarray(parts["text"], dtype=np.float64), basis)
            residual_rows = np.asarray(parts["concat"], dtype=np.float64)
            bounds, topic, residual = proposal_features(
                topic_pca.transform(topic_rows), residual_pca.transform(residual_rows)
            )
            proposal_topic_std = (topic - topic_mean) / topic_scale
            conditional_error = residual - conditional.predict(proposal_topic_std)
            energy_conditional = (
                gaussian_energy(conditional_error, np.zeros(residual.shape[1]), conditional_variance)
                - conditional_center
            ) / conditional_scale
            energy_unconditional = (
                gaussian_energy(residual, unconditional_mean, unconditional_variance)
                - unconditional_center
            ) / unconditional_scale
            distance = kmeans.transform(proposal_topic_std).min(1)
            is_supported = distance <= support_threshold
            supported += int(is_supported.sum())
            total_proposals += len(is_supported)
            per_video_support[video_id] = float(is_supported.mean())
            record = {
                "video_id": video_id,
                "score_conditional": frame_readout(len(topic_rows), bounds, energy_conditional).tolist(),
                "score_unconditional": frame_readout(len(topic_rows), bounds, energy_unconditional).tolist(),
            }
            handle.write(json.dumps(record) + "\n")

    support_fraction = supported / total_proposals
    (out_dir / "support.json").write_text(json.dumps({
        "corpus": args.corpus,
        "n_negative_train_videos": len(negative_ids),
        "n_negative_train_frames": len(topic_frame_rows),
        "n_negative_train_proposals": len(negative_topic),
        "n_test_videos": len(test_ids),
        "n_test_proposals": total_proposals,
        "support_threshold_from_negative_train_p95": support_threshold,
        "test_proposal_support_fraction": support_fraction,
        "support_gate": 0.80,
        "support_pass": support_fraction >= 0.80,
        "per_video_support": per_video_support,
        "test_labels_or_temporal_gt_loaded_by_producer": False,
    }, indent=2) + "\n")
    print(json.dumps({"corpus": args.corpus, "scores": str(score_path), "support": support_fraction}))


if __name__ == "__main__":
    main()
