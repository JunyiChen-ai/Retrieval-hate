"""Train-only premise test for frozen cross-modal conditional prediction.

This script never opens validation or test manifests.  It fits one fixed PCA
space per modality on the target corpus' train videos, then uses three
video-level folds to measure held-out conditional prediction.  The decision
is frozen: both pilot corpora must beat an unconditional target mean and must
be worse after cross-video conditioning is substituted while retaining the
target video's availability pattern.
"""

from __future__ import annotations

import argparse
from collections import Counter
import json
import os
import random
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.decomposition import PCA

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
BASELINES = os.path.join(REPO, "scripts", "reproduction_baselines")
MULTIHATELOC = os.path.join(BASELINES, "multihateloc")
sys.path.insert(0, BASELINES)
sys.path.insert(0, MULTIHATELOC)

from hate_common import data as hdata  # noqa: E402
import data as mdata  # noqa: E402
from components import masked_logmeanexp  # noqa: E402

MODALITIES = tuple(mdata.MODALITIES)


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def available(x: np.ndarray) -> np.ndarray:
    """True only when an observed feature row is present and finite."""
    return np.isfinite(x).all(axis=1) & (np.linalg.norm(x, axis=1) > 1e-8)


def load_train(corpus: str) -> dict[str, dict[str, np.ndarray]]:
    ids = hdata.load_split(corpus, "train")
    videos = {}
    for vid in ids:
        row = {}
        length = None
        for mod in MODALITIES:
            x = np.load(mdata.feature_path(mod, corpus, vid)).astype(np.float32)
            if length is None:
                length = len(x)
            if len(x) != length:
                raise ValueError(f"unaligned modalities for {corpus}/{vid}")
            row[mod] = x
        videos[vid] = row
    return videos


def fit_fixed_spaces(videos, width: int, sample_frames: int, seed: int):
    rng = np.random.default_rng(seed)
    spaces = {}
    for mod in MODALITIES:
        chunks = []
        for vid in sorted(videos):
            x = videos[vid][mod]
            mask = available(x)
            if mask.any():
                chunks.append(x[mask])
        all_x = np.concatenate(chunks, axis=0)
        if len(all_x) > sample_frames:
            take = rng.choice(len(all_x), size=sample_frames, replace=False)
            fit_x = all_x[take]
        else:
            fit_x = all_x
        if width > min(fit_x.shape):
            raise ValueError(f"PCA width {width} invalid for {mod} {fit_x.shape}")
        pca = PCA(n_components=width, whiten=True, svd_solver="randomized",
                  random_state=seed)
        pca.fit(fit_x)
        if not np.all(np.isfinite(pca.explained_variance_)):
            raise ValueError(f"non-finite PCA variance for {mod}")
        if float(np.min(pca.explained_variance_)) <= 1e-10:
            raise ValueError(f"collapsed PCA component for {mod}")
        spaces[mod] = pca
    return spaces


def project(videos, spaces):
    out = {}
    for vid, raw in videos.items():
        item = {}
        for mod in MODALITIES:
            x = raw[mod]
            mask = available(x)
            z = np.zeros((len(x), spaces[mod].n_components_), np.float32)
            if mask.any():
                z[mask] = spaces[mod].transform(x[mask]).astype(np.float32)
            item[mod] = z
            item[mod + "_available"] = mask
        out[vid] = item
    return out


def video_folds(ids, n_folds: int, seed: int):
    order = np.asarray(sorted(ids), dtype=object)
    rng = np.random.default_rng(seed)
    order = order[rng.permutation(len(order))]
    return [list(x) for x in np.array_split(order, n_folds)]


class Predictor(nn.Module):
    """Other modalities and availability -> target in a fixed PCA space."""

    def __init__(self, width: int, hidden: int, radius: int):
        super().__init__()
        in_ch = 2 * width + 2
        self.net = nn.Sequential(
            nn.Conv1d(in_ch, hidden, 2 * radius + 1, padding=radius),
            nn.GELU(),
            nn.Conv1d(hidden, width, 1),
        )

    def forward(self, x):
        return self.net(x.transpose(1, 2)).transpose(1, 2)


def fully_observed(projected, vid, modalities):
    return all(projected[vid][mod + "_available"].all()
               for mod in modalities)


def balanced_donor_assignment(projected, recipients, target):
    """Evenly assign eligible held-out donor videos, always excluding self."""
    others = [m for m in MODALITIES if m != target]
    eligible = [v for v in sorted(recipients)
                if fully_observed(projected, v, others)]
    if len(eligible) < 2:
        raise ValueError(f"fewer than two eligible donors for {target}")
    assignment = {}
    for i, vid in enumerate(sorted(recipients)):
        donor = eligible[i % len(eligible)]
        if donor == vid:
            donor = eligible[(i + 1) % len(eligible)]
        assignment[vid] = donor
    loads = Counter(assignment.values())
    return assignment, {
        "n_recipients": len(recipients),
        "n_eligible_donors": len(eligible),
        "n_used_donors": len(loads),
        "max_donor_load": max(loads.values()),
        "max_donor_fraction": max(loads.values()) / len(recipients),
        "self_assignments": sum(v == d for v, d in assignment.items()),
    }


def batch(projected, ids, target, device, substitute=None):
    """Pad videos; optional substituted conditions retain target availability."""
    others = [m for m in MODALITIES if m != target]
    lengths = [len(projected[v][target]) for v in ids]
    tmax = max(lengths)
    width = projected[ids[0]][target].shape[1]
    cond = np.zeros((len(ids), tmax, 2 * width + 2), np.float32)
    y = np.zeros((len(ids), tmax, width), np.float32)
    target_mask = np.zeros((len(ids), tmax), bool)
    for i, vid in enumerate(ids):
        length = lengths[i]
        y[i, :length] = projected[vid][target]
        target_mask[i, :length] = projected[vid][target + "_available"]
        source_vid = substitute[i] if substitute is not None else vid
        keeps = [projected[vid][mod + "_available"] for mod in others]
        if substitute is None:
            sources = [projected[vid][mod] for mod in others]
        else:
            # Choose one donor with both conditioning modalities observed for
            # its full duration.  A single monotone uniform time map is shared
            # by both modalities; recipient availability is applied only
            # afterwards.  Thus shuffling introduces neither cross-modal
            # desynchrony nor artificial local time reversals/discontinuities.
            ordered = sorted(projected)
            start = ordered.index(source_vid)
            candidates = [ordered[(start + offset) % len(ordered)]
                          for offset in range(len(ordered))]
            donor = None
            for candidate in candidates:
                if candidate == vid:
                    continue
                if fully_observed(projected, candidate, others):
                    donor = candidate
                    break
            if donor is None:
                raise ValueError(f"no synchronized donor for {vid}/{target}")
            donor_length = len(projected[donor][others[0]])
            positions = np.linspace(0, donor_length - 1, length)
            donor_index = np.rint(positions).astype(int)
            sources = [projected[donor][mod][donor_index] for mod in others]
        for j, (mod, keep, source) in enumerate(zip(others, keeps, sources)):
            cond[i, :length, j * width:(j + 1) * width] = source * keep[:, None]
            cond[i, :length, 2 * width + j] = keep.astype(np.float32)
    return (torch.from_numpy(cond).to(device), torch.from_numpy(y).to(device),
            torch.from_numpy(target_mask).to(device), lengths)


def iter_minibatches(ids, batch_size, rng):
    ids = list(ids)
    rng.shuffle(ids)
    for start in range(0, len(ids), batch_size):
        yield ids[start:start + batch_size]


def train_predictor(projected, train_ids, target, args, device, fold_seed):
    model = Predictor(args.width, args.hidden, args.radius).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    rng = random.Random(fold_seed)
    model.train()
    for _ in range(args.epochs):
        for ids in iter_minibatches(train_ids, args.batch_size, rng):
            cond, y, mask, _ = batch(projected, ids, target, device)
            pred = model(cond)
            loss = F.smooth_l1_loss(pred[mask], y[mask])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
    return model


@torch.no_grad()
def evaluate(model, projected, eval_ids, train_ids, target, args, device,
             seed):
    model.eval()
    train_targets = np.concatenate([
        projected[v][target][projected[v][target + "_available"]]
        for v in train_ids
    ], axis=0)
    mean = torch.from_numpy(train_targets.mean(axis=0).astype(np.float32)).to(device)
    donor_by_vid, donor_balance = balanced_donor_assignment(
        projected, eval_ids, target)
    totals = {"matched": 0.0, "mean": 0.0, "shuffled": 0.0, "n": 0}
    for start in range(0, len(eval_ids), args.batch_size):
        ids = eval_ids[start:start + args.batch_size]
        subs = [donor_by_vid[v] for v in ids]
        cond, y, mask, _ = batch(projected, ids, target, device)
        cond_shuf, _, _, _ = batch(projected, ids, target, device, subs)
        pred = model(cond)
        pred_shuf = model(cond_shuf)
        n = int(mask.sum()) * y.shape[-1]
        totals["matched"] += float(F.smooth_l1_loss(pred[mask], y[mask],
                                                    reduction="sum"))
        totals["mean"] += float(F.smooth_l1_loss(
            mean.expand_as(y[mask]), y[mask], reduction="sum"))
        totals["shuffled"] += float(F.smooth_l1_loss(
            pred_shuf[mask], y[mask], reduction="sum"))
        totals["n"] += n
    n = max(totals.pop("n"), 1)
    return {"errors": {k: v / n for k, v in totals.items()},
            "loss_sums": totals, "n_elements": n,
            "donor_balance": donor_balance}


def availability_invariance() -> dict:
    # Channel order is modality-major: observed, predicted, residual for each
    # of visual/audio/text.  A missing target must mask all three channels.
    c = 0.37
    outputs = []
    masks = []
    for modality_available in (
            (True, True, True), (True, True, False), (True, False, True),
            (True, False, False)):
        valid = torch.tensor([
            present for present in modality_available for _ in range(3)
        ], dtype=torch.bool)
        logits = torch.full((9,), c)
        # Extreme absent logits prove masking, not their stored value, controls
        # the result.
        logits[~valid] = 1000.0
        outputs.append(float(masked_logmeanexp(logits, valid)))
        masks.append(valid.tolist())
    spread = max(outputs) - min(outputs)
    missing_target_all_channels_masked = all(
        not any(mask[3 * m:3 * m + 3])
        for mask, pattern in zip(masks, (
            (True, True, True), (True, True, False), (True, False, True),
            (True, False, False)))
        for m, present in enumerate(pattern) if not present)
    return {"outputs": outputs, "channel_masks": masks,
            "max_abs_spread": spread,
            "missing_target_all_channels_masked": missing_target_all_channels_masked,
            "pass": spread < 1e-6 and missing_target_all_channels_masked}


def shuffle_pair_alignment_test() -> dict:
    """Use encoded time pairs to prove both conditions keep one donor index."""
    target_avail = np.ones(4, dtype=bool)
    donor_avail = np.ones(6, dtype=bool)
    text_avail = np.asarray([True, False, True, True])
    projected = {
        "target": {
            "visual": np.zeros((4, 1), np.float32),
            "visual_available": target_avail,
            "audio": np.arange(10, 14, dtype=np.float32)[:, None],
            "audio_available": target_avail,
            "text": np.asarray([100, 0, 102, 103], np.float32)[:, None],
            "text_available": text_avail,
        },
        "donor": {
            "visual": np.ones((6, 1), np.float32),
            "visual_available": donor_avail,
            "audio": np.arange(20, 26, dtype=np.float32)[:, None],
            "audio_available": donor_avail,
            "text": np.arange(200, 206, dtype=np.float32)[:, None],
            "text_available": donor_avail,
        },
    }
    cond, _, _, _ = batch(projected, ["target"], "visual", "cpu",
                          substitute=["donor"])
    cond = cond[0].numpy()
    paired = text_avail
    # At every jointly available time, donor text encodes the same time index
    # as donor audio by a fixed +180 offset.
    max_pair_error = float(np.max(np.abs(
        (cond[paired, 1] - cond[paired, 0]) - 180.0)))
    min_time_step = float(np.min(np.diff(cond[:, 0])))
    flags_exact = bool(np.array_equal(cond[:, 2].astype(bool), target_avail)
                       and np.array_equal(cond[:, 3].astype(bool), text_avail))
    return {"max_pair_error": max_pair_error,
            "minimum_encoded_donor_time_step": min_time_step,
            "flags_exact": flags_exact,
            "pass": max_pair_error < 1e-7 and min_time_step >= 0 and flags_exact}


def run_corpus(corpus, args, device):
    started = time.time()
    videos = load_train(corpus)
    folds = video_folds(list(videos), args.folds, args.seed)
    fold_rows = []
    fixed_space_rows = []
    for fold_idx, eval_ids in enumerate(folds):
        train_ids = [v for i, fold in enumerate(folds) if i != fold_idx
                     for v in fold]
        # Fit every preprocessing statistic without the held-out videos.  The
        # premise is genuinely OOF even though all folds remain inside train.
        fit_videos = {v: videos[v] for v in train_ids}
        spaces = fit_fixed_spaces(fit_videos, args.width,
                                  args.pca_sample_frames,
                                  args.seed + 1000 * fold_idx)
        projected = project(videos, spaces)
        fixed_space_rows.append({
            "fold": fold_idx,
            "modalities": {
                m: {
                    "width": args.width,
                    "min_explained_variance": float(
                        spaces[m].explained_variance_.min()),
                    "max_explained_variance": float(
                        spaces[m].explained_variance_.max()),
                } for m in MODALITIES
            },
        })
        for mod_idx, target in enumerate(MODALITIES):
            fold_seed = args.seed + 100 * fold_idx + mod_idx
            seed_all(fold_seed)
            model = train_predictor(projected, train_ids, target, args, device,
                                    fold_seed)
            measured = evaluate(model, projected, eval_ids, train_ids, target,
                                args, device, fold_seed)
            fold_rows.append({"fold": fold_idx, "target": target,
                              "n_eval_videos": len(eval_ids), **measured})
            del model
    aggregate_micro = {}
    aggregate_macro = {}
    for key in ("matched", "mean", "shuffled"):
        total_loss = sum(r["loss_sums"][key] for r in fold_rows)
        total_n = sum(r["n_elements"] for r in fold_rows)
        aggregate_micro[key] = float(total_loss / max(total_n, 1))
        aggregate_macro[key] = float(np.mean([
            r["errors"][key] for r in fold_rows]))
    decision = {
        "matched_beats_mean": aggregate_micro["matched"] < aggregate_micro["mean"],
        "shuffle_worse_than_matched": aggregate_micro["shuffled"] > aggregate_micro["matched"],
    }
    decision["pass"] = all(decision.values())
    return {
        "corpus": corpus,
        "n_train_videos": len(videos),
        "fixed_space_by_fold": fixed_space_rows,
        "folds": fold_rows,
        "aggregate_micro_huber_per_element": aggregate_micro,
        "diagnostic_macro_fold_modality_huber": aggregate_macro,
        "decision": decision,
        "wall_seconds": round(time.time() - started, 2),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpora", nargs="+", default=["hatemm", "hateclipseg"])
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--seed", type=int, default=234)
    ap.add_argument("--width", type=int, default=64)
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--radius", type=int, default=2)
    ap.add_argument("--folds", type=int, default=3)
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--pca-sample-frames", type=int, default=30000)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--smoke", action="store_true",
                    help="allow non-frozen settings but never emit a proceed verdict")
    args = ap.parse_args()
    frozen = {
        "corpora": ["hatemm", "hateclipseg"], "seed": 234, "width": 64,
        "hidden": 128, "radius": 2, "folds": 3, "epochs": 6,
        "batch_size": 16, "lr": 3e-4, "pca_sample_frames": 30000,
        "device": "cuda",
    }
    actual = {key: getattr(args, key) for key in frozen}
    if not args.smoke and actual != frozen:
        raise SystemExit("formal premise settings are frozen; use exact defaults")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA unavailable")
    os.makedirs(args.out_dir, exist_ok=True)
    seed_all(args.seed)
    result = {
        "protocol": "train-only three-fold video OOF; no validation/test input",
        "args": vars(args),
        "availability_invariance": availability_invariance(),
        "shuffle_pair_alignment": shuffle_pair_alignment_test(),
        "corpora": {},
    }
    for corpus in args.corpora:
        result["corpora"][corpus] = run_corpus(corpus, args, args.device)
        with open(os.path.join(args.out_dir, "analysis.partial.json"), "w") as fh:
            json.dump(result, fh, indent=2)
    corpus_pass = all(x["decision"]["pass"] for x in result["corpora"].values())
    formal_pass = (corpus_pass and result["availability_invariance"]["pass"]
                   and result["shuffle_pair_alignment"]["pass"])
    result["verdict"] = {
        "all_corpora_pass": corpus_pass,
        "availability_invariance_pass": result["availability_invariance"]["pass"],
        "shuffle_pair_alignment_pass": result["shuffle_pair_alignment"]["pass"],
        "decision": ("SMOKE_ONLY_NO_DECISION" if args.smoke else
                     "PROCEED_TO_FORMAL_LOCALIZER" if formal_pass else
                     "STOP_BEFORE_FORMAL_LOCALIZER"),
    }
    with open(os.path.join(args.out_dir, "analysis.json"), "w") as fh:
        json.dump(result, fh, indent=2)
    print(json.dumps(result["verdict"], indent=2), flush=True)


if __name__ == "__main__":
    main()
