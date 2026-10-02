"""Load QTL state dictionaries with the explicitly selected content architecture."""
import json
from pathlib import Path

import numpy as np
import torch

from . import ctree, qtree
from .train import DEFAULTS


def load_trial(trial, device, answers, ids, model_factory):
    trial = Path(trial)
    summary = json.loads((trial / "summary.json").read_text())
    cfg = dict(DEFAULTS)
    cfg.update(summary["cfg"])
    assert cfg["prior"] == "chain" and cfg["chain"] == "learned"
    checkpoint = torch.load(trial / "model.pth", map_location=device, weights_only=False)
    assert int(checkpoint["epoch"]) == int(summary["best_epoch"])
    model = model_factory(cfg).to(device)
    model.load_state_dict(checkpoint["model"], strict=True)
    model.eval()
    loglen = np.log([b - a for v in ids["train"] for (a, b) in answers[v]])
    answer_model = qtree.AnswerModel(
        checkpoint["am"]["theta"].cpu().numpy(), loglen.mean(), loglen.std(),
        bool(cfg["length_term"]), int(cfg["n_state"]), cfg["categories"]).to(device)
    answer_model.load_state_dict(checkpoint["am"], strict=True)
    answer_model.len_mu, answer_model.len_sd = float(loglen.mean()), float(loglen.std())
    answer_model.eval()
    chain = ctree.Chain(cfg["boundary"] == "closed", cfg["chain_form"] == "zero_inflated",
                        cfg["chain_form"] == "normalized").to(device)
    chain.load_state_dict(checkpoint["chain"], strict=True)
    chain.eval()
    return summary, cfg, model, answer_model, chain
