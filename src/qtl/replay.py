"""Replay the batched EIG policy (cpolicy.run_batch) of a trained checkpoint on cached answers, per split.

Shared by experiments/20261101_r5_completion (sim_monotone.py, final_curves.py) and
experiments/20261108_astra_review (policy_replay.py)."""
import torch

from . import cpolicy, qtree


def run_source(model, store, ids, am, chain, answers, cfg, max_calls, device, copy_fn, splits=("val", "test"),
               level=0):
    """Per split, video id -> the run of cpolicy.run_batch (scores after 0..n calls, eig, asked, ...).
    level L > 0 (review control E3): only the nodes of one tree depth (qtree.level_nodes) may be asked."""
    runs = {}
    for split in splits:
        vids = sorted(ids[split], key=lambda v: store.T[v])
        runs[split] = {}
        for i in range(0, len(vids), int(cfg["eval_chunk"])):
            chunk = vids[i:i + int(cfg["eval_chunk"])]
            allowed = None if not level else [set(qtree.level_nodes(store.T[v], level).tolist()) for v in chunk]
            with torch.no_grad():
                runs[split].update(cpolicy.run_batch(
                    model, store, chunk, am, chain, answers, cfg["categories"],
                    max_calls, device, order="eig", allowed=allowed, copy_pi=copy_fn, copy_mode="both"))
    for split in runs:                                   # rule_values reads "voi"; qmixE1 does not use it
        for r in runs[split].values():
            r["voi"] = [0.0] * len(r["eig"])
    return runs


def truncate(run, cap):
    n = min(cap, len(run["eig"]))
    return {k: (list(v[:n + 1]) if k in ("scores", "p_G") else list(v[:n])) for k, v in run.items()}
