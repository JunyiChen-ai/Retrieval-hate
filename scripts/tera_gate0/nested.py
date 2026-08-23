#!/usr/bin/env python
"""TERA Gate-0 — nested OOF protocol (appendix sec 7.1, 7.2, 7.3).

5 outer video-stratified folds (seed 20260807), 4 inner folds inside every outer
training partition (seed 20260808), the four inner models advanced in LOCKSTEP
one epoch at a time, pooled inner-OOF macro-F1 with min_delta 1e-4 and patience
40, then one refit on the full outer-train at the selected epoch.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
import torch
from sklearn.model_selection import StratifiedKFold

from .arms import A3, make_model
from .common import (BATCH_SIZE, E_MAX, INNER_FOLD_SEED, MIN_DELTA, OUTER_FOLD_SEED,
                     PATIENCE, TeraHalt, derive_seed, progress, select_threshold)


# ------------------------------------------------------------------- folds --
def stratified_folds(ids, label_map, n_splits, seed):
    """StratifiedKFold over the SORTED video-id list (appendix sec 7.1)."""
    order = sorted(ids)
    y = np.array([int(label_map[v]) for v in order], dtype=np.int64)
    for cls in (0, 1):
        if int((y == cls).sum()) < n_splits:
            raise TeraHalt("HALT_FOLD_INFEASIBLE",
                           "class %d has %d members for %d folds"
                           % (cls, int((y == cls).sum()), n_splits))
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    folds = []
    arr = np.array(order)
    for tr, te in skf.split(np.zeros(len(order)), y):
        folds.append((sorted(arr[tr].tolist()), sorted(arr[te].tolist())))
    return folds


def outer_folds(ids, label_map, n_splits=5):
    return stratified_folds(ids, label_map, n_splits, OUTER_FOLD_SEED)


def inner_folds(ids, label_map, n_splits=4):
    return stratified_folds(ids, label_map, n_splits, INNER_FOLD_SEED)


# -------------------------------------------------------------- arm inputs --
@dataclass
class ArmData:
    """Per-video input tensors for one arm, aligned to `ids`."""

    ids: list
    y: torch.Tensor
    inputs: tuple
    d: int
    seg_input: torch.Tensor = None       # [V, K, d] when the arm scores segments

    def __post_init__(self):
        self.index = {v: i for i, v in enumerate(self.ids)}

    def rows(self, id_list):
        return torch.as_tensor([self.index[v] for v in id_list], dtype=torch.long)


# ---------------------------------------------------------------- training --
def _train_one_epoch(model, data, rows, opt, lossfn, scope, epoch):
    gen = torch.Generator()
    gen.manual_seed(derive_seed(scope) + epoch)
    perm = torch.randperm(rows.numel(), generator=gen)
    shuffled = rows[perm]
    model.train()
    for start in range(0, shuffled.numel(), BATCH_SIZE):
        batch = shuffled[start:start + BATCH_SIZE]
        opt.zero_grad()
        logits = model(*[t[batch] for t in data.inputs])
        loss = lossfn(logits, data.y[batch])
        loss.backward()
        opt.step()


@torch.no_grad()
def score_rows(model, data, rows):
    """Video scores plus registered per-segment scores (appendix sec 3.1).

    Returns score, seg_score (sigmoid of the segment logit), seg_logit (the raw
    logit the oracles pool) and A3's attention weights.
    """
    model.eval()
    logits = model(*[t[rows] for t in data.inputs])
    scores = torch.sigmoid(logits).double().numpy()
    seg = seg_logit = att = None
    if data.seg_input is not None and hasattr(model, "segment_logits"):
        x = data.seg_input[rows]
        zl = model.segment_logits(x)
        seg_logit = zl.double().numpy()
        seg = torch.sigmoid(zl).double().numpy()
        if isinstance(model, A3):
            att = model.attention(x).double().numpy()
    return scores, seg, seg_logit, att


def _make_optimizer(model, cfg):
    return torch.optim.AdamW(model.parameters(), lr=cfg["lr"],
                             weight_decay=cfg["weight_decay"], betas=(0.9, 0.999),
                             eps=1e-8, amsgrad=False)


@dataclass
class ArmFoldResult:
    arm: str
    outer: int
    config: dict
    epoch: int
    theta: float
    inner_macro_f1: float
    seed: int
    inner_scores: dict = field(default_factory=dict)
    inner_seg: dict = field(default_factory=dict)
    inner_seg_logit: dict = field(default_factory=dict)
    inner_att: dict = field(default_factory=dict)
    query_scores: dict = field(default_factory=dict)
    query_seg: dict = field(default_factory=dict)
    query_seg_logit: dict = field(default_factory=dict)
    query_att: dict = field(default_factory=dict)
    n_trainings: int = 0
    candidates: list = field(default_factory=list)


def run_arm_fold(stage, arm, dataset, data, outer_idx, train_ids, query_ids,
                 configs, label_map, h3=None, t0=None, log_every_cfg=True):
    """One outer fold of the single registered selection loop (appendix sec 7.2)."""
    lossfn = torch.nn.BCEWithLogitsLoss()
    inner = inner_folds(train_ids, label_map)
    inner_rows = [(data.rows(tr), data.rows(va), va) for tr, va in inner]

    best = None            # (macro_f1, epoch, cfg_index, cfg, theta, snapshot)
    candidates = []
    n_trainings = 0
    for ci, cfg in enumerate(configs):
        models, opts, scopes = [], [], []
        for j, (tr_rows, _, _) in enumerate(inner_rows):
            scope = {"stage": stage, "arm": arm, "dataset": dataset,
                     "outer": outer_idx, "inner": j, "config": cfg["config_id"]}
            model = make_model(arm, cfg, data.d, derive_seed(scope), h3=h3)
            models.append(model)
            opts.append(_make_optimizer(model, cfg))
            scopes.append(scope)
            n_trainings += 1

        cfg_best = None
        stall = 0
        last_epoch = 0
        for epoch in range(1, E_MAX + 1):
            last_epoch = epoch
            pooled_scores, pooled_labels = [], []
            snap_scores, snap_seg, snap_logit, snap_att = {}, {}, {}, {}
            for j, (tr_rows, va_rows, va_ids) in enumerate(inner_rows):
                _train_one_epoch(models[j], data, tr_rows, opts[j], lossfn,
                                 scopes[j], epoch)
                s, seg, seg_logit, att = score_rows(models[j], data, va_rows)
                pooled_scores.append(s)
                pooled_labels.append(np.array([label_map[v] for v in va_ids],
                                              dtype=np.int64))
                for pos, vid in enumerate(va_ids):
                    snap_scores[vid] = float(s[pos])
                    if seg is not None:
                        snap_seg[vid] = [float(x) for x in seg[pos]]
                        snap_logit[vid] = [float(x) for x in seg_logit[pos]]
                    if att is not None:
                        snap_att[vid] = [float(x) for x in att[pos]]
            scores = np.concatenate(pooled_scores)
            labels = np.concatenate(pooled_labels)
            theta, m = select_threshold(scores, labels)
            if cfg_best is None or m > cfg_best[0] + MIN_DELTA:
                cfg_best = (m, epoch, theta, snap_scores, snap_seg, snap_logit, snap_att)
                stall = 0
            else:
                stall += 1
            if stall >= PATIENCE:
                break

        candidates.append({"config_id": cfg["config_id"], "macro_f1": cfg_best[0],
                           "epoch": cfg_best[1], "theta": cfg_best[2],
                           "epochs_run": last_epoch})
        cand = (cfg_best[0], cfg_best[1], ci, cfg, cfg_best[2],
                (cfg_best[3], cfg_best[4], cfg_best[5], cfg_best[6]))
        if best is None:
            best = cand
        else:
            # argmax macro-F1; ties -> smaller epoch; then smaller config index
            if (cand[0] > best[0]) or (cand[0] == best[0] and cand[1] < best[1]):
                best = cand
        if log_every_cfg:
            progress(stage, arm, outer_idx, ci + 1, len(configs), cfg_best[1], t0)

    macro, epoch_star, _, cfg_star, theta_star, snapshot = best
    scope = {"stage": stage, "arm": arm, "dataset": dataset, "outer": outer_idx,
             "inner": -1, "config": cfg_star["config_id"]}
    seed = derive_seed(scope)
    model = make_model(arm, cfg_star, data.d, seed, h3=h3)
    opt = _make_optimizer(model, cfg_star)
    lossfn = torch.nn.BCEWithLogitsLoss()
    tr_rows = data.rows(train_ids)
    for epoch in range(1, epoch_star + 1):
        _train_one_epoch(model, data, tr_rows, opt, lossfn, scope, epoch)
    n_trainings += 1

    result = ArmFoldResult(arm=arm, outer=outer_idx, config=cfg_star, epoch=epoch_star,
                           theta=theta_star, inner_macro_f1=macro, seed=seed,
                           n_trainings=n_trainings, candidates=candidates)
    (result.inner_scores, result.inner_seg, result.inner_seg_logit,
     result.inner_att) = snapshot
    if query_ids:
        q_rows = data.rows(query_ids)
        q_scores, q_seg, q_seg_logit, q_att = score_rows(model, data, q_rows)
        for pos, vid in enumerate(query_ids):
            result.query_scores[vid] = float(q_scores[pos])
            if q_seg is not None:
                result.query_seg[vid] = [float(x) for x in q_seg[pos]]
                result.query_seg_logit[vid] = [float(x) for x in q_seg_logit[pos]]
            if q_att is not None:
                result.query_att[vid] = [float(x) for x in q_att[pos]]
    result.model = model
    return result


def refit_full(stage, arm, dataset, data, train_ids, cfg, epochs, h3=None, outer=-1):
    """One refit on a whole partition (confirmation protocol, appendix sec 7.10)."""
    scope = {"stage": stage, "arm": arm, "dataset": dataset, "outer": outer,
             "inner": -1, "config": cfg["config_id"]}
    seed = derive_seed(scope)
    model = make_model(arm, cfg, data.d, seed, h3=h3)
    opt = _make_optimizer(model, cfg)
    lossfn = torch.nn.BCEWithLogitsLoss()
    rows = data.rows(train_ids)
    for epoch in range(1, int(epochs) + 1):
        _train_one_epoch(model, data, rows, opt, lossfn, scope, epoch)
    return model, seed


def select_on_partition(stage, arm, dataset, data, train_ids, configs, label_map,
                        h3=None, t0=None):
    """Inner-OOF selection + single refit, no outer loop (sec 7.10.2).

    Returns the ArmFoldResult whose `.model` is the registered single refit on the
    whole partition at (cfg*, epoch*).
    """
    return run_arm_fold(stage, arm, dataset, data, -1, train_ids, [], configs,
                        label_map, h3=h3, t0=t0)
