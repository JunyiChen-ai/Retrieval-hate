#!/usr/bin/env python
"""Train a matched MultiHateLoc anchor or lexical-PR core."""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.data as tdata

ROOT = Path(__file__).resolve().parents[2]
MULTI_DIR = ROOT / "scripts/reproduction_baselines/multihateloc"
sys.path.insert(0, str(MULTI_DIR))
sys.path.insert(1, str(ROOT / "scripts/reproduction_baselines"))
sys.path.insert(2, str(ROOT / "src"))
import data as mdata  # noqa: E402
from hate_common import data as hdata  # noqa: E402
from model import MultiHateLoc  # noqa: E402
from scoped_video_protocol import (evaluator_test_ids,
                                   scoped_video_labels)  # noqa: E402

PROJECTION_EPS = 1e-12
POS_QUANTILE = 0.20
POS_GAP = 0.20
NEG_MAX = 0.10
MIN_SPEECH = 10
MIN_SET = 2


def load_arrays(path: Path) -> dict[str, np.ndarray]:
    with np.load(path) as z:
        return {key: np.asarray(z[key], dtype=np.float64) for key in z.files}


def support_masks(evidence: np.ndarray, speech: np.ndarray, label: int
                  ) -> tuple[np.ndarray, np.ndarray] | None:
    spoken = np.asarray(speech).astype(bool)
    if int(spoken.sum()) < MIN_SPEECH:
        return None
    values = np.asarray(evidence, dtype=float)[spoken]
    if label == 1:
        low_cut = float(np.quantile(values, POS_QUANTILE))
        high_cut = float(np.quantile(values, 1.0 - POS_QUANTILE))
        if not high_cut > low_cut:
            return None
        high = spoken & (evidence >= high_cut)
        low = spoken & (evidence <= low_cut)
        high &= ~low
        low &= ~high
        if int(high.sum()) < MIN_SET or int(low.sum()) < MIN_SET:
            return None
        return high, low
    high_cut = float(np.quantile(values, 1.0 - POS_QUANTILE))
    high = spoken & (evidence >= high_cut) & (evidence > 0.0)
    if int(high.sum()) < MIN_SET:
        return None
    return high, np.zeros_like(high)


def _logit(p: torch.Tensor) -> torch.Tensor:
    p = p.clamp(PROJECTION_EPS, 1.0 - PROJECTION_EPS)
    return torch.log(p) - torch.log1p(-p)


def projected_target(prob: torch.Tensor, high: torch.Tensor,
                     low: torch.Tensor, label: int
                     ) -> tuple[torch.Tensor, bool, float]:
    """Exact one-constraint I-projection of detached Bernoulli posteriors."""
    # Solve the dual in float64.  Float32 logit round-trips can create KKT
    # residuals just above the frozen tolerance even when bisection converged.
    base = prob.detach().to(torch.float64).clamp(
        PROJECTION_EPS, 1.0 - PROJECTION_EPS)
    target = base.clone()
    if label == 1:
        current = base[high].mean() - base[low].mean()
        if float(current) >= POS_GAP:
            return target, False, 0.0
        hi_logit, lo_logit = _logit(base[high]), _logit(base[low])
        n_high = int(high.sum())
        n_low = int(low.sum())
        left, right = 0.0, 1.0
        while float(torch.sigmoid(hi_logit + right / n_high).mean() -
                    torch.sigmoid(lo_logit - right / n_low).mean()) < POS_GAP:
            right *= 2.0
            if right > 1e7:
                raise RuntimeError("failed to bracket positive PR dual")
        for _ in range(60):
            mid = (left + right) / 2.0
            gap = torch.sigmoid(hi_logit + mid / n_high).mean() \
                - torch.sigmoid(lo_logit - mid / n_low).mean()
            if float(gap) >= POS_GAP:
                right = mid
            else:
                left = mid
        hi_shifted = hi_logit + right / n_high
        lo_shifted = lo_logit - right / n_low
        target[high] = torch.sigmoid(hi_shifted)
        target[low] = torch.sigmoid(lo_shifted)
        primal = abs(float(target[high].mean() - target[low].mean()) - POS_GAP)
        # Check stationarity in the dual logit domain.  Reconstructing logits
        # from q is numerically invalid when sigmoid rounds q to 0 or 1.
        dual_hi = (hi_shifted - hi_logit) * n_high
        dual_lo = (lo_logit - lo_shifted) * n_low
        dual_all = torch.cat((dual_hi, dual_lo))
        kkt = float(torch.max(dual_all) - torch.min(dual_all))
        residual = max(primal, kkt)
        if residual > 2e-4:
            raise RuntimeError(f"positive PR projection residual {residual}")
        return target, True, residual
    current = base[high].mean()
    if float(current) <= NEG_MAX:
        return target, False, 0.0
    hi_logit = _logit(base[high])
    n_high = int(high.sum())
    left, right = 0.0, 1.0
    while float(torch.sigmoid(hi_logit - right / n_high).mean()) > NEG_MAX:
        right *= 2.0
        if right > 1e7:
            raise RuntimeError("failed to bracket negative PR dual")
    for _ in range(60):
        mid = (left + right) / 2.0
        if float(torch.sigmoid(hi_logit - mid / n_high).mean()) <= NEG_MAX:
            right = mid
        else:
            left = mid
    hi_shifted = hi_logit - right / n_high
    target[high] = torch.sigmoid(hi_shifted)
    primal = abs(float(target[high].mean()) - NEG_MAX)
    dual = (hi_logit - hi_shifted) * n_high
    kkt = float(torch.max(dual) - torch.min(dual))
    residual = max(primal, kkt)
    if residual > 2e-4:
        raise RuntimeError(f"negative PR projection residual {residual}")
    return target, True, residual


def bernoulli_kl(target: torch.Tensor, prob: torch.Tensor,
                 selected: torch.Tensor) -> torch.Tensor:
    q = target[selected]
    p = prob[selected].to(torch.float64).clamp(
        PROJECTION_EPS, 1.0 - PROJECTION_EPS)
    return (torch.special.xlogy(q, q) - torch.special.xlogy(q, p) +
            torch.special.xlogy(1.0 - q, 1.0 - q) -
            torch.special.xlogy(1.0 - q, 1.0 - p)).mean()


def pr_loss(prob: torch.Tensor, labels: torch.Tensor, vids: list[str],
            evidence: dict[str, np.ndarray], speech: dict[str, np.ndarray]
            ) -> tuple[torch.Tensor, dict[str, float]]:
    losses = []
    eligible_pos = eligible_neg = active = 0
    max_residual = 0.0
    for i, vid in enumerate(vids):
        length = len(evidence[vid])
        if length > prob.shape[1]:
            raise ValueError(f"evidence longer than batch tensor for {vid}")
        y = int(labels[i].item())
        support = support_masks(evidence[vid], speech[vid], y)
        if support is None:
            continue
        high_np, low_np = support
        high = torch.as_tensor(high_np, dtype=torch.bool, device=prob.device)
        low = torch.as_tensor(low_np, dtype=torch.bool, device=prob.device)
        p = prob[i, :length]
        target, changed, residual = projected_target(p, high, low, y)
        selected = high | low
        losses.append(bernoulli_kl(target, p, selected))
        eligible_pos += int(y == 1)
        eligible_neg += int(y == 0)
        active += int(changed)
        max_residual = max(max_residual, residual)
    zero = prob.sum() * 0.0
    loss = torch.stack(losses).mean() if losses else zero
    return loss, {"pr_eligible_pos": float(eligible_pos),
                  "pr_eligible_neg": float(eligible_neg),
                  "pr_active": float(active),
                  "pr_projection_residual_max": max_residual}


def average_precision(y_true, y_score) -> float:
    y_true = np.asarray(y_true, dtype=float)
    score = np.asarray(y_score, dtype=float)
    order = np.argsort(-score, kind="mergesort")
    y = y_true[order]
    npos = float(y.sum())
    if npos == 0:
        return float("nan")
    tp = np.cumsum(y)
    precision = tp / np.arange(1, len(y) + 1)
    recall = tp / npos
    return float(np.sum(np.diff(np.r_[0.0, recall]) * precision))


def run_epoch(model, loader, device, optimizer, arm, evidence, speech,
              lambda_pr):
    model.train()
    sums: dict[str, float] = {}
    n = 0
    for feats, labels, lengths, mask, vids in loader:
        feats = {k: v.to(device, non_blocking=True) for k, v in feats.items()}
        labels = labels.to(device)
        lengths = lengths.to(device)
        mask = mask.to(device)
        out = model(feats, mask)
        mil, _ = model.mil_loss(out["probs"], mask, lengths, labels)
        smooth = model.smoothness_loss(out["probs"], mask)
        contrast = model.contrastive_loss(out["embeds"], mask)
        lexical = out["probs"]["fused"].sum() * 0.0
        counts = {"pr_eligible_pos": 0.0, "pr_eligible_neg": 0.0,
                  "pr_active": 0.0, "pr_projection_residual_max": 0.0}
        if arm == "core":
            lexical, counts = pr_loss(
                out["probs"]["fused"], labels, vids, evidence, speech)
        loss = mil + 0.1 * smooth + 0.2 * contrast + lambda_pr * lexical
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        batch = len(vids)
        for key, value in {"loss": float(loss.detach()),
                           "mil": float(mil.detach()),
                           "smooth": float(smooth.detach()),
                           "contrast": float(contrast.detach()),
                           "pr": float(lexical.detach())}.items():
            sums[key] = sums.get(key, 0.0) + value * batch
        for key, value in counts.items():
            if key == "pr_projection_residual_max":
                sums[key] = max(sums.get(key, 0.0), value)
            else:
                sums[key] = sums.get(key, 0.0) + value
        n += batch
    out = {key: value / max(n, 1) for key, value in sums.items()
           if not key.startswith("pr_eligible") and
           key not in ("pr_active", "pr_projection_residual_max")}
    for key in ("pr_eligible_pos", "pr_eligible_neg", "pr_active"):
        out[key] = sums.get(key, 0.0)
    out["pr_projection_residual_max"] = sums.get(
        "pr_projection_residual_max", 0.0)
    return out


@torch.no_grad()
def predict(model, loader, device):
    model.eval()
    frames, video = {}, {}
    for feats, _, lengths, mask, vids in loader:
        feats = {k: v.to(device) for k, v in feats.items()}
        mask = mask.to(device)
        out = model(feats, mask)
        vid_scores = model.video_scores(out["probs"], mask, lengths.to(device))
        for i, vid in enumerate(vids):
            length = int(lengths[i])
            frames[vid] = out["probs"]["fused"][i, :length].cpu().numpy()
            video[vid] = float(vid_scores["fused"][i])
    return frames, video


@torch.no_grad()
def constraint_diagnostics(model, loader, device, evidence, speech):
    model.eval()
    pos_gaps, pos_violations, neg_means, neg_violations = [], [], [], []
    for feats, labels, lengths, mask, vids in loader:
        feats = {k: v.to(device) for k, v in feats.items()}
        out = model(feats, mask.to(device))
        prob = out["probs"]["fused"]
        for i, vid in enumerate(vids):
            y = int(labels[i].item())
            support = support_masks(evidence[vid], speech[vid], y)
            if support is None:
                continue
            high_np, low_np = support
            p = prob[i, :int(lengths[i])]
            high = torch.as_tensor(high_np, dtype=torch.bool, device=device)
            if y == 1:
                low = torch.as_tensor(low_np, dtype=torch.bool, device=device)
                gap = float(p[high].mean() - p[low].mean())
                pos_gaps.append(gap)
                pos_violations.append(max(0.0, POS_GAP - gap))
            else:
                mean = float(p[high].mean())
                neg_means.append(mean)
                neg_violations.append(max(0.0, mean - NEG_MAX))
    def mean(values):
        return float(np.mean(values)) if values else None
    return {"n_positive_support": len(pos_gaps),
            "positive_gap_mean": mean(pos_gaps),
            "positive_violation_mean": mean(pos_violations),
            "n_negative_support": len(neg_means),
            "negative_high_mean": mean(neg_means),
            "negative_violation_mean": mean(neg_violations)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", required=True, choices=("hatemm", "hateclipseg"))
    ap.add_argument("--arm", required=True, choices=("anchor", "core"))
    ap.add_argument("--evidence-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--seed", type=int, default=234)
    ap.add_argument("--max-epoch", type=int, default=100)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lambda-pr", type=float, default=1.0)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA unavailable")
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    evidence_dir = Path(args.evidence_dir).resolve() / args.corpus

    train_ids = hdata.load_split(args.corpus, "train")
    val_ids = hdata.load_split(args.corpus, "val")
    test_ids = evaluator_test_ids(args.corpus, hdata.load_split(args.corpus, "test"))
    train_labels = scoped_video_labels(args.corpus, "train", train_ids)
    val_labels = scoped_video_labels(args.corpus, "val", val_ids)
    labels = {**train_labels, **val_labels, **{v: 0 for v in test_ids}}
    evidence = load_arrays(evidence_dir / "train_evidence.npz")
    speech = load_arrays(evidence_dir / "train_speech.npz")
    if set(evidence) != set(train_ids) or set(speech) != set(train_ids):
        raise RuntimeError("train evidence coverage mismatch")

    generator = torch.Generator().manual_seed(args.seed)
    train_loader = tdata.DataLoader(
        mdata.MultiModalDataset(args.corpus, train_ids, labels),
        batch_size=args.batch_size, shuffle=True, collate_fn=mdata.collate,
        num_workers=4, generator=generator)
    val_loader = tdata.DataLoader(
        mdata.MultiModalDataset(args.corpus, val_ids, labels),
        batch_size=args.batch_size, shuffle=False, collate_fn=mdata.collate,
        num_workers=2)
    test_loader = tdata.DataLoader(
        mdata.MultiModalDataset(args.corpus, test_ids, labels),
        batch_size=args.batch_size, shuffle=False, collate_fn=mdata.collate,
        num_workers=2)
    model = MultiHateLoc(
        {m: mdata.FEATURE_DIMS[m] for m in mdata.MODALITIES},
        hidden=256, embed=128, dropout=0.1, k_proportion=3,
        temperature=0.07).to(args.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    initial_constraints = constraint_diagnostics(
        model, train_loader, args.device, evidence, speech)
    best_ap = -1.0
    best_epoch = None
    best_state = None
    history = []
    start = time.time()
    for epoch in range(1, args.max_epoch + 1):
        stats = run_epoch(model, train_loader, args.device, optimizer,
                          args.arm, evidence, speech, args.lambda_pr)
        _, val_video = predict(model, val_loader, args.device)
        ids = sorted(val_video)
        val_ap = average_precision([val_labels[v] for v in ids],
                                   [val_video[v] for v in ids])
        stats.update({"epoch": epoch, "val_video_ap": val_ap,
                      "elapsed_seconds": round(time.time() - start, 1)})
        history.append(stats)
        if val_ap > best_ap:
            best_ap, best_epoch = val_ap, epoch
            best_state = {k: v.detach().cpu().clone()
                          for k, v in model.state_dict().items()}
        if epoch == 1 or epoch % 10 == 0:
            print(args.corpus, args.arm, epoch, stats, flush=True)
    if best_state is None:
        raise RuntimeError("no checkpoint selected")
    model.load_state_dict(best_state)
    selected_constraints = constraint_diagnostics(
        model, train_loader, args.device, evidence, speech)
    frames, _ = predict(model, test_loader, args.device)
    with (out_dir / "scores.jsonl").open("w", encoding="utf-8") as fh:
        for vid in test_ids:
            fh.write(json.dumps({"video_id": vid,
                                 "score_fused": frames[vid].astype(float).tolist()}) + "\n")
    torch.save(model.state_dict(), out_dir / "model.pt")
    log = {"corpus": args.corpus, "arm": args.arm, "args": vars(args),
           "n_train": len(train_ids), "n_val": len(val_ids),
           "n_test": len(test_ids), "selected_epoch": best_epoch,
           "selected_val_video_ap": best_ap,
           "initial_train_constraints": initial_constraints,
           "selected_train_constraints": selected_constraints,
           "wall_seconds": round(time.time() - start, 1), "history": history}
    (out_dir / "train_log.json").write_text(json.dumps(log, indent=2) + "\n")
    print("selected", args.corpus, args.arm, best_epoch, best_ap, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
