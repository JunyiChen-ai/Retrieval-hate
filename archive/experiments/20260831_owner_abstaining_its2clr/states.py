"""Frozen pseudo-state construction for all attribution arms."""

from __future__ import annotations

import math

import torch

from model import ABSTAIN, BACKGROUND, CARRIER


def pace_for_epoch(epoch: int, epochs: int) -> float:
    if epochs <= 1:
        return 1.0
    return 0.2 + 0.8 * (epoch - 1) / (epochs - 1)


def _selected_tails(score, pace, k_proportion):
    length = len(score)
    candidate_count = max(1, math.ceil(length / k_proportion))
    active_count = max(1, math.ceil(candidate_count * pace))
    order = torch.argsort(score, descending=True, stable=True)
    positive = torch.zeros(length, dtype=torch.bool)
    negative = torch.zeros(length, dtype=torch.bool)
    positive[order[:active_count]] = True
    remaining = order[active_count:]
    negative[remaining[-min(active_count, len(remaining)):]] = True
    return positive, negative


def positive_states(row, arm, pace, k_proportion=3):
    score = row["fused_score"]
    branch = row["branch_score"]
    first = row["deletion_centroid"]
    second = row["deletion_neighbor"]
    shuffle_key = row["shuffle_key"]
    if not (branch.shape == first.shape == second.shape == shuffle_key.shape):
        raise RuntimeError("OOF pseudo evidence shape mismatch")
    length, modalities = branch.shape
    selected, negative_selected = _selected_tails(score, pace, k_proportion)
    stable = (first > 0) & (second > 0)
    core = stable & selected[:, None]
    states = torch.full((length, modalities), ABSTAIN, dtype=torch.long)
    states[negative_selected] = BACKGROUND
    if arm in ("core", "projection_only"):
        states[core] = CARRIER
    elif arm == "broadcast":
        states[selected] = CARRIER
    elif arm in ("branch_selector", "shuffled_carrier"):
        for modality in range(modalities):
            count = int(core[:, modality].sum())
            if not count:
                continue
            candidates = torch.where(selected)[0]
            ranking = (branch[candidates, modality] if arm == "branch_selector"
                       else -shuffle_key[candidates, modality])
            chosen = candidates[torch.argsort(
                ranking, descending=True, stable=True
            )[:count]]
            states[chosen, modality] = CARRIER
    elif arm == "shuffled_carrier":
        raise AssertionError("unreachable")
    elif arm == "abstain_negative":
        states[selected] = BACKGROUND
        states[core] = CARRIER
    elif arm == "nonpositive_background":
        states[core] = CARRIER
        all_nonpositive = (first <= 0).all(1) & (second <= 0).all(1)
        background_time = ~selected & ~negative_selected & all_nonpositive
        states[background_time] = BACKGROUND
    elif arm == "anchor":
        pass
    else:
        raise ValueError(f"unknown arm {arm}")
    return states


def batch_states(video_ids, labels, lengths, mask, cache, arm, epoch, epochs,
                 k_proportion=3):
    batch, longest = mask.shape
    modalities = len(cache["modalities"])
    states = torch.full((batch, longest, modalities), ABSTAIN, dtype=torch.long)
    pace = pace_for_epoch(epoch, epochs)
    for index, video_id in enumerate(video_ids):
        length = int(lengths[index])
        if video_id not in cache["rows"]:
            raise RuntimeError(f"OOF pseudo evidence missing {video_id}")
        if labels[index].item() == 0:
            states[index, :length] = BACKGROUND
        else:
            row_states = positive_states(
                cache["rows"][video_id], arm, pace, k_proportion
            )
            if len(row_states) != length:
                raise RuntimeError(f"OOF length mismatch for {video_id}")
            states[index, :length] = row_states
    if not torch.equal(states != ABSTAIN, (states != ABSTAIN) & mask[:, :, None]):
        raise RuntimeError("pseudo state escaped the valid temporal mask")
    return states
