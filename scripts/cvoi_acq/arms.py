"""Registered arm roster and per-arm architectures (appendix section 8, prereg section 7).

This module is the executable expression of the frozen arm contract.  It holds no
data, computes no candidate metric and never reads a label outside the explicitly
marked ``oracle`` namespace.  ``models.py`` owns the shared state classifier and
action tokenizer; this module owns the arm-specific heads and the deterministic
selection mechanics that ``selectors.py`` executes.
"""
from __future__ import annotations

import hashlib
import itertools
import math

import numpy as np
import torch
from torch import nn

from .protocol import exact_knapsack

# ---------------------------------------------------------------------------
# Roster
# ---------------------------------------------------------------------------

#: Prereg section 7 mandatory arms, in registered table order.
REGISTERED_ARMS = ("B0", "B1-O", "B1-D", "B1-J", "B2", "B3", "B4", "B5", "B6",
                   "B7", "B8", "B9", "B10", "B11", "B12", "O1", "O2")

#: Arms whose prediction rows may bind a gate (prereg section 8.1 admissible set).
NON_CVOI_COMPARATORS = ("B2", "B3", "B4", "B5", "B6", "B7", "B8", "B11", "B12")
CEILINGS = ("B0", "B1-O", "B1-D", "B1-J")
ORACLES = ("O1", "O2")
CANDIDATE = "B10"
CONTROLS = ("B10-width-matched",)

#: Action universes each deployable selector is instantiated and frozen for.
ACTION_UNIVERSES = ("ocr", "dense", "joint")

ARM_ROLES = {
    "B0": "zero-cost floor",
    "B1-O": "OCR full-information ceiling",
    "B1-D": "dense full-information ceiling",
    "B1-J": "joint full-information ceiling",
    "B2": "random-budget control",
    "B3": "coverage control",
    "B4": "cheap heuristic selector",
    "B5": "uncertainty router",
    "B6": "matched-cost coarse routing",
    "B7": "in-domain selection control",
    "B8": "relevance-versus-utility control",
    "B9": "non-set-conditioned CVoI control",
    "B10": "proposed method",
    "B11": "optimization control",
    "B12": "cost-aware learned top-k/knapsack",
    "O1": "evaluation-only oracle ceiling",
    "O2": "evaluation-only subset ceiling",
}

WIDTH_MATCHED_SHUFFLE_SEED = 20260816
B11_ALPHA_GRID = (0.1, 1.0, 10.0)
B2_DRAWS = 20
B3_INITIAL_WINDOW = 15
ACTION_TYPE_ORDER = ("ocr", "dense4")


def _window(action_id: str) -> int:
    return int(action_id.rsplit(":", 1)[1])


def _kind(action_id: str) -> str:
    return action_id.split(":")[1]


def _type_rank(action_id: str) -> int:
    return ACTION_TYPE_ORDER.index(_kind(action_id))


# ---------------------------------------------------------------------------
# B0 / B1 fixed-set arms
# ---------------------------------------------------------------------------

def fixed_action_set(arm: str, actions) -> tuple:
    """B0 and the three B1 always-acquire ceilings are label- and score-free."""
    ordered = tuple(sorted(actions, key=lambda a: (_type_rank(a), _window(a))))
    if arm == "B0":
        return ()
    if arm == "B1-O":
        return tuple(a for a in ordered if _kind(a) == "ocr")
    if arm == "B1-D":
        return tuple(a for a in ordered if _kind(a) == "dense4")
    if arm == "B1-J":
        return ordered
    raise KeyError("not a fixed-set arm: " + arm)


class SpecialistHead(nn.Module):
    """B0/B1 strong controls: identical architecture to the shared classifier head."""

    def __init__(self, dropout: float = 0.1):
        super().__init__()
        self.head = nn.Sequential(nn.Linear(512, 256), nn.GELU(), nn.Dropout(dropout),
                                  nn.Linear(256, 1))

    def forward(self, joint):
        return self.head(joint).squeeze(-1)


# ---------------------------------------------------------------------------
# B2 randomness: registered Philox stream
# ---------------------------------------------------------------------------

def philox_key(split_seed, refit_seed, video_id, budget, draw_id) -> int:
    """Counter-based key over exactly the registered tuple."""
    material = "||".join(str(x) for x in (split_seed, refit_seed, video_id, budget, draw_id))
    return int.from_bytes(hashlib.sha256(material.encode("utf-8")).digest()[:16], "big")


def philox_generator(split_seed, refit_seed, video_id, budget, draw_id):
    return np.random.Generator(np.random.Philox(key=philox_key(
        split_seed, refit_seed, video_id, budget, draw_id)))


# ---------------------------------------------------------------------------
# B4 salience (cheap midpoint visual only)
# ---------------------------------------------------------------------------

def b4_window_distance(cheap_windows) -> np.ndarray:
    """Cosine distance of each legal cheap midpoint feature from their mean."""
    x = np.asarray(cheap_windows, dtype=np.float64)
    if x.ndim != 2:
        raise ValueError("cheap_windows must be [n_windows, dim]")
    centre = x.mean(0)
    cn = np.linalg.norm(centre)
    rn = np.linalg.norm(x, axis=1)
    denom = np.where((rn * cn) > 0, rn * cn, np.nan)
    cos = (x @ centre) / denom
    return 1.0 - np.nan_to_num(cos, nan=0.0)


def b4_scores(cheap_windows, actions, train_mean: float, train_std: float) -> dict:
    """Outer-train standardized salience, shared by both action types of a window."""
    d = b4_window_distance(cheap_windows)
    s = float(train_std) if float(train_std) > 0 else 1.0
    z = (d - float(train_mean)) / s
    return {a: float(z[_window(a)]) for a in actions}


def b4_order(actions, scores) -> tuple:
    """Descending salience; ties by lower window index, then ocr before dense4."""
    return tuple(sorted(actions, key=lambda a: (-float(scores[a]), _window(a), _type_rank(a))))


# ---------------------------------------------------------------------------
# B5 uncertainty proxy (no action output is consulted)
# ---------------------------------------------------------------------------

def b5_empty_token_gradient_norms(classifier, z, empty_tokens, windows, kinds) -> dict:
    """Expected absolute change proxy = grad norm w.r.t. the type-specific empty token.

    ``empty_tokens`` is a [n_actions, d] leaf tensor produced from the frozen
    per-type EMPTY embedding; no acquired outcome is read.
    """
    tokens = empty_tokens.detach().clone().requires_grad_(True)
    logits = classifier(z.expand(tokens.shape[0], -1), tokens[:, None, :])
    logits.abs().sum().backward()
    grad = tokens.grad.detach()
    norms = torch.linalg.vector_norm(grad, dim=-1)
    return {f"{k}:{w:02d}": float(v) for k, w, v in zip(kinds, windows, norms.tolist())}


# ---------------------------------------------------------------------------
# B6 per-video router
# ---------------------------------------------------------------------------

class CoarseRouter(nn.Module):
    """Logistic gate on the base representation z; all-or-none B3 package."""

    def __init__(self, z_dim: int):
        super().__init__()
        self.gate = nn.Linear(z_dim, 1)

    def forward(self, z):
        return self.gate(z).squeeze(-1)


# ---------------------------------------------------------------------------
# B7 / B8 MultiHateLoc-style hard temporal selector
# ---------------------------------------------------------------------------

class HardTemporalSelector(nn.Module):
    """Registered architecture: Linear(cheap_window_dim,128) - tanh - Linear(128,2)."""

    def __init__(self, cheap_window_dim: int):
        super().__init__()
        self.body = nn.Sequential(nn.Linear(cheap_window_dim, 128), nn.Tanh(),
                                  nn.Linear(128, 2))

    def forward(self, cheap_windows):
        return self.body(cheap_windows)

    def action_scores(self, cheap_windows, actions) -> dict:
        out = self.forward(cheap_windows)
        return {a: float(out[_window(a), _type_rank(a)]) for a in actions}


B8_TARGETS = ("privileged_gatec_required_type", "cheap_ocr_presence")


def b8_target_namespace(variant: str) -> str:
    if variant not in B8_TARGETS:
        raise KeyError("unregistered B8 target: " + variant)
    return "diagnostics" if variant == B8_TARGETS[0] else "deployable"


# ---------------------------------------------------------------------------
# B9 / B10 / B12 utility policies
# ---------------------------------------------------------------------------

B10_WIDTHS = (1024, 512, 128, 1)


class SetConditionedPolicy(nn.Module):
    """B10: MLP([stateCLS; cheap_action; type; position; c_hat; remaining])."""

    def __init__(self, input_dim: int, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 1024), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(1024, 512), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(512, 128), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(128, 1))

    def forward(self, x):
        return self.net(x).squeeze(-1)


class SingletonPolicy(nn.Module):
    """B9: identical architecture to B10, but the state slot is frozen to empty.

    The empty state CLS is registered at construction; passing a non-empty state
    slot is a contract violation, so B9 provably sums no set interaction.
    """

    def __init__(self, input_dim: int, state_dim: int, dropout: float = 0.1):
        super().__init__()
        self.net = SetConditionedPolicy(input_dim, dropout).net
        self.state_dim = int(state_dim)
        self.register_buffer("empty_state", torch.zeros(1, int(state_dim)))

    def set_empty_state(self, empty_cls):
        with torch.no_grad():
            self.empty_state.copy_(empty_cls.reshape(1, self.state_dim))

    def forward(self, action_block, state_block=None):
        if state_block is not None:
            raise RuntimeError("HALT_B9_SET_INTERACTION")
        n = action_block.shape[0]
        x = torch.cat([self.empty_state.expand(n, -1), action_block], dim=-1)
        return self.net(x).squeeze(-1)


def width_matched_targets(utilities, fit_fold_index, seed: int = WIDTH_MATCHED_SHUFFLE_SEED):
    """Shuffle utility targets within each training fold; parameter count unchanged."""
    y = np.asarray(utilities, dtype=np.float64).copy()
    folds = np.asarray(fit_fold_index)
    rng = np.random.default_rng(seed)
    for f in sorted(set(folds.tolist())):
        ix = np.flatnonzero(folds == f)
        y[ix] = y[rng.permutation(ix)]
    return y


# ---------------------------------------------------------------------------
# B11 ridge on singleton targets
# ---------------------------------------------------------------------------

class FrozenSingletonRidge:
    """Inner-selected alpha in {0.1,1,10}; one empty-state prediction per trajectory."""

    def __init__(self, alpha_grid=B11_ALPHA_GRID):
        self.alpha_grid = tuple(float(a) for a in alpha_grid)
        self.alpha = None
        self.coef = None

    @staticmethod
    def _solve(x, y, alpha):
        a = np.c_[np.ones(len(x)), np.asarray(x, float)]
        p = a.shape[1]
        reg = float(alpha) * np.eye(p)
        reg[0, 0] = 0.0
        return np.linalg.solve(a.T @ a + reg, a.T @ np.asarray(y, float))

    def fit(self, x, y, inner_folds):
        """``inner_folds`` selects alpha by inner-OOF squared error only."""
        x = np.asarray(x, float)
        y = np.asarray(y, float)
        folds = np.asarray(inner_folds)
        scored = []
        for alpha in self.alpha_grid:
            err = []
            for f in sorted(set(folds.tolist())):
                tr = folds != f
                va = ~tr
                coef = self._solve(x[tr], y[tr], alpha)
                pred = np.c_[np.ones(va.sum()), x[va]] @ coef
                err.append(float(np.mean((pred - y[va]) ** 2)))
            scored.append((float(np.mean(err)), alpha))
        self.alpha = min(scored)[1]
        self.coef = self._solve(x, y, self.alpha)
        return self

    def predict(self, x):
        if self.coef is None:
            raise RuntimeError("HALT_B11_UNFIT")
        return np.c_[np.ones(len(x)), np.asarray(x, float)] @ self.coef


def b11_frozen_order(actions, empty_state_predictions, estimated_costs) -> tuple:
    """Order by predicted singleton utility / estimated cost, frozen at the empty state."""
    return tuple(sorted(actions, key=lambda a: (
        -(float(empty_state_predictions[a]) / float(estimated_costs[a])),
        float(estimated_costs[a]), a)))


# ---------------------------------------------------------------------------
# B12 exact cost-aware solver (appendix section 10)
# ---------------------------------------------------------------------------

def cost_ticks(cost_ms) -> int:
    return int(math.ceil(float(cost_ms) * 10.0))


def budget_ticks(budget_ms) -> int:
    """Budget is floored so a rounded tick can never authorize an infeasible buy."""
    return int(math.floor(float(budget_ms) * 10.0))


def b12_solve(actions, scores, costs_ms, budget_ms) -> tuple:
    ids = list(actions)
    chosen = exact_knapsack([float(scores[a]) for a in ids],
                            [cost_ticks(costs_ms[a]) for a in ids],
                            budget_ticks(budget_ms))
    return tuple(ids[i] for i in chosen)


def greedy_ratio(actions, scores, costs_ms, budget_ms) -> tuple:
    out, spent = [], 0.0
    for a in sorted(actions, key=lambda q: (-(float(scores[q]) / float(costs_ms[q])),
                                            float(costs_ms[q]), q)):
        if spent + float(costs_ms[a]) <= float(budget_ms):
            out.append(a)
            spent += float(costs_ms[a])
    return tuple(out)


def cost_blind_topk(actions, scores, k) -> tuple:
    return tuple(sorted(actions, key=lambda q: (-float(scores[q]), q))[:int(k)])


# ---------------------------------------------------------------------------
# O1 / O2 oracles (explicit namespace; never write a deployable field)
# ---------------------------------------------------------------------------

def o1_greedy(actions, costs_ms, budget_ms, realized_marginal_utility):
    """Label-aware: recompute realized marginal log-loss reduction after every buy."""
    chosen, spent = [], 0.0
    remaining = list(actions)
    while remaining:
        feasible = [a for a in remaining if spent + float(costs_ms[a]) <= float(budget_ms)]
        if not feasible:
            break
        ratios = {a: realized_marginal_utility(tuple(chosen), a) / float(costs_ms[a])
                  for a in feasible}
        best = max(feasible, key=lambda a: (ratios[a], -float(costs_ms[a]), a))
        if ratios[best] <= 0:
            break
        chosen.append(best)
        spent += float(costs_ms[best])
        remaining.remove(best)
    return {"oracle_actions": tuple(chosen), "oracle_realized_cost_ms": spent,
            "namespace": "oracle"}


def o2_exhaustive(reduced_universe, costs_ms, budget_ms, evaluate, max_actions: int = 18):
    """Exhaustive feasible subsets on the pre-frozen reduced universe (diagnostic)."""
    ids = list(reduced_universe)
    if len(ids) > max_actions:
        raise RuntimeError("HALT_O2_UNIVERSE_TOO_LARGE")
    best = ((), float("-inf"))
    for r in range(len(ids) + 1):
        for combo in itertools.combinations(ids, r):
            if sum(float(costs_ms[a]) for a in combo) > float(budget_ms):
                continue
            value = float(evaluate(combo))
            if value > best[1] or (value == best[1] and combo < best[0]):
                best = (combo, value)
    return {"oracle_actions": best[0], "namespace": "oracle"}


# ---------------------------------------------------------------------------
# Contract description used by the C8 audit
# ---------------------------------------------------------------------------

def arm_contract() -> dict:
    return {
        "registered_arms": list(REGISTERED_ARMS),
        "controls": list(CONTROLS),
        "action_universes": list(ACTION_UNIVERSES),
        "candidate": CANDIDATE,
        "non_cvoi_comparators": list(NON_CVOI_COMPARATORS),
        "ceilings": list(CEILINGS),
        "oracles": list(ORACLES),
        "b2_draws": B2_DRAWS,
        "b3_initial_window": B3_INITIAL_WINDOW,
        "b11_alpha_grid": list(B11_ALPHA_GRID),
        "b10_widths": list(B10_WIDTHS),
        "width_matched_shuffle_seed": WIDTH_MATCHED_SHUFFLE_SEED,
        "b8_targets": list(B8_TARGETS),
        "action_type_order": list(ACTION_TYPE_ORDER),
    }
