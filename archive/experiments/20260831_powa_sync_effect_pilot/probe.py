#!/usr/bin/env python3
"""Validation-only probe for POWA within-video modality sync effects."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from torch.utils.data import DataLoader


HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
BASELINES = REPO / "scripts/reproduction_baselines"
sys.path.insert(0, str(BASELINES))

from eval_baseline_scores import evaluate_scores  # noqa: E402
from hate_common import data as hdata  # noqa: E402
from powa_macil.dataset import PowaTestDataset, usable_text_ids  # noqa: E402
from powa_macil.model import POWAMACIL  # noqa: E402


ANCHORS = {
    "hatemm": REPO / (
        "results/reproduction/powa_macil/"
        "final_maskfix_finetune_hatemm_seed234_e5"
    ),
    "hateclipseg": REPO / (
        "runs/20260831_powa_starting_point/hcs_maskfix_seed234"
    ),
}
MODES = ("text", "audio", "visual", "speech")


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def safe_logit(probability):
    probability = probability.clamp(1e-5, 1.0 - 1e-5)
    return torch.log(probability) - torch.log1p(-probability)


def load_anchor(corpus, device):
    checkpoint = ANCHORS[corpus]
    meta = json.loads((checkpoint / "train_meta.json").read_text())
    cfg = SimpleNamespace(**meta["args"])
    if list(getattr(cfg, "corpora", [corpus])) != [corpus]:
        raise RuntimeError(f"non-corpus-specific anchor: {checkpoint}")
    model = POWAMACIL(cfg, policy=corpus).to(device)
    state_path = checkpoint / "model.pth"
    state = torch.load(state_path, map_location=device)
    legacy = "policy_residual_gate" not in state
    model.load_state_dict(state, strict=not legacy)
    model.use_policy_residual = (
        not legacy and not getattr(cfg, "typed_only", False)
    )
    model.eval()
    return model, cfg, checkpoint, sha256(state_path)


def shifted_inputs(f_a, f_v, f_t, shift, mode):
    a, v, t = f_a, f_v, f_t
    if mode in ("audio", "speech"):
        a = torch.roll(a, shift, dims=1)
    if mode == "visual":
        v = torch.roll(v, shift, dims=1)
    if mode in ("text", "speech"):
        t = torch.roll(t, shift, dims=1)
    return a, v, t


def metric_summary(report):
    return {
        "pooled_ap": report["pr_auc"],
        "pooled_roc": report["roc_auc"],
        "within_roc": report["per_video"]["macro_auc"],
        "within_n": report["per_video"]["n_videos_both_classes"],
    }


@torch.no_grad()
def probe_corpus(corpus, device):
    model, cfg, checkpoint, anchor_hash = load_anchor(corpus, device)
    labels = hdata.load_labels(corpus)
    _, val_ids = hdata.load_train_val(corpus, labels)
    val_ids = usable_text_ids(corpus, val_ids)
    dataset = PowaTestDataset(corpus, val_ids, cfg.max_seqlen, cfg.grid, "av")
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=4)
    branches = {"score_powa": {}}
    for mode in MODES:
        branches[f"effect_{mode}"] = {}
        branches[f"powa_plus_effect_{mode}"] = {}
    for f_v, f_a, f_t, index_map, n_seconds, video_id in loader:
        video_id = video_id[0]
        f_v = f_v[0].float().to(device)
        f_a = f_a[0].float().to(device)
        f_t = f_t[0].float().to(device)
        lengths = torch.full(
            (f_v.shape[0],), f_v.shape[1], dtype=torch.long, device=device
        )
        valid = torch.ones(
            (f_v.shape[0], f_v.shape[1]), dtype=torch.bool, device=device
        )
        original = model(f_a, f_v, f_t, lengths, valid, policy=corpus)
        original_logit = safe_logit(original["frame_prob"]).mean(0)
        index = index_map[0].numpy()
        original_second = torch.sigmoid(original_logit).cpu().numpy()[index]
        branches["score_powa"][video_id] = original_second
        width = f_v.shape[1]
        magnitudes = sorted({max(1, width // 4), max(1, width // 2)})
        shifts = sorted({direction * magnitude
                         for magnitude in magnitudes
                         for direction in (-1, 1)})
        for mode in MODES:
            counterfactual_logits = []
            for shift in shifts:
                shifted = shifted_inputs(f_a, f_v, f_t, shift, mode)
                output = model(
                    shifted[0], shifted[1], shifted[2], lengths, valid,
                    policy=corpus,
                )
                counterfactual_logits.append(
                    safe_logit(output["frame_prob"]).mean(0)
                )
            mean_counterfactual = torch.stack(counterfactual_logits).mean(0)
            effect = original_logit - mean_counterfactual
            effect_second = effect.cpu().numpy()[index]
            combined_second = torch.sigmoid(
                original_logit + effect
            ).cpu().numpy()[index]
            if len(effect_second) != int(n_seconds):
                raise RuntimeError(f"alignment mismatch: {video_id}")
            branches[f"effect_{mode}"][video_id] = effect_second
            branches[f"powa_plus_effect_{mode}"][video_id] = combined_second
    gt = hdata.gt_arrays(corpus, "val")
    positive_ids = {video_id for video_id in gt if labels[video_id] == 1}
    reports = {
        name: evaluate_scores(scores, gt, positive_ids)
        for name, scores in branches.items()
    }
    return {
        "corpus": corpus,
        "split": "val",
        "anchor_checkpoint": str(checkpoint.resolve()),
        "anchor_model_sha256": anchor_hash,
        "train_or_test_labels_read": False,
        "val_labels_used_for_mechanism_probe": True,
        "interventions": {
            "modes": list(MODES),
            "circular_shift_fractions": [0.25, 0.5],
            "directions": [-1, 1],
            "modality_content_multisets_preserved": True,
        },
        "metrics": {
            name: metric_summary(report) for name, report in reports.items()
        },
    }


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    payload = {
        "date": "2026-08-31",
        "stage": "validation_only_zero_training_mechanism_probe",
        "device": device,
        "corpora": {
            corpus: probe_corpus(corpus, device)
            for corpus in ("hatemm", "hateclipseg")
        },
    }
    out = REPO / "runs/20260831_powa_sync_effect_pilot/val_probe.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    temporary = out.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n")
    temporary.replace(out)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
