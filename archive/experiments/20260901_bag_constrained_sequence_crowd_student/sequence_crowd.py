"""EM label models for heterogeneous noisy temporal localizers."""
from __future__ import annotations

import numpy as np
from scipy.special import logsumexp


EPS = 1e-12


def _normalize(counts, axis=-1):
    counts = np.asarray(counts, dtype=np.float64)
    return counts / np.maximum(counts.sum(axis=axis, keepdims=True), EPS)


def _emission_mean(probability):
    bins = np.arange(probability.shape[-1], dtype=np.float64)
    return (probability * bins).sum(-1)


class SequenceCrowdEM:
    """Binary latent sequence with source-specific transition-edge emissions.

    Observations are ordinal bins ``[T, J]``; ``-1`` denotes a missing source.
    ``sequential=False`` is the token-wise Dawid--Skene control.
    ``bag_conditioned=True`` clamps negative bags and conditions positive bags
    on at least one latent positive state.
    """

    def __init__(self, n_sources, n_bins=5, sequential=True,
                 bag_conditioned=True, n_iter=20, smoothing=1.0):
        self.n_sources = int(n_sources)
        self.n_bins = int(n_bins)
        self.sequential = bool(sequential)
        self.bag_conditioned = bool(bag_conditioned)
        self.n_iter = int(n_iter)
        self.smoothing = float(smoothing)
        self.params = None
        self.history = []

    def _initialize_posteriors(self, observations, labels):
        out = {}
        for video_id, obs in observations.items():
            valid = obs >= 0
            scaled = np.where(valid, obs / max(self.n_bins - 1, 1), np.nan)
            with np.errstate(invalid="ignore"):
                score = np.nanmean(scaled, axis=1)
            score = np.nan_to_num(score, nan=.25)
            q1 = np.clip(.05 + .90 * score, .02, .98)
            if self.bag_conditioned and labels[video_id] == 0:
                q1[:] = 0.0
            q = np.stack([1.0 - q1, q1], axis=1)
            xi = np.einsum("ti,tj->tij", q[:-1], q[1:])
            out[video_id] = (q, xi)
        return out

    def _m_step(self, observations, posteriors):
        smooth = self.smoothing
        init = np.full(2, smooth)
        transition = np.full((2, 2), smooth)
        token_emit = np.full((self.n_sources, 2, self.n_bins), smooth)
        initial_emit = np.full_like(token_emit, smooth)
        edge_emit = np.full((self.n_sources, 2, 2, self.n_bins), smooth)
        state_mass = np.full(2, smooth)
        for video_id, obs in observations.items():
            q, xi = posteriors[video_id]
            init += q[0]
            state_mass += q.sum(0)
            if len(obs) > 1:
                transition += xi.sum(0)
            for source in range(self.n_sources):
                for t, value in enumerate(obs[:, source]):
                    if value < 0:
                        continue
                    token_emit[source, :, value] += q[t]
                    if t == 0:
                        initial_emit[source, :, value] += q[t]
                    else:
                        edge_emit[source, :, :, value] += xi[t - 1]
        params = {
            "prior": _normalize(state_mass),
            "init": _normalize(init),
            "transition": _normalize(transition, axis=1),
            "token_emit": _normalize(token_emit),
            "initial_emit": _normalize(initial_emit),
            "edge_emit": _normalize(edge_emit),
        }
        # Fix the latent class identity by ordinal direction. This is not a
        # fitted threshold: state 1 is definitionally the higher-score state.
        means = _emission_mean(params["token_emit"])
        if float(np.mean(means[:, 1] - means[:, 0])) < 0:
            params["prior"] = params["prior"][::-1].copy()
            params["init"] = params["init"][::-1].copy()
            params["transition"] = params["transition"][::-1, ::-1].copy()
            params["token_emit"] = params["token_emit"][:, ::-1].copy()
            params["initial_emit"] = params["initial_emit"][:, ::-1].copy()
            params["edge_emit"] = params["edge_emit"][:, ::-1, ::-1].copy()
        return params

    def _log_terms(self, obs):
        p = self.params
        if self.sequential:
            log_init = np.log(np.maximum(p["init"], EPS))
            for source, value in enumerate(obs[0]):
                if value >= 0:
                    log_init += np.log(np.maximum(
                        p["initial_emit"][source, :, value], EPS))
            edges = np.empty((max(len(obs) - 1, 0), 2, 2), dtype=np.float64)
            for t in range(1, len(obs)):
                edge = np.log(np.maximum(p["transition"], EPS)).copy()
                for source, value in enumerate(obs[t]):
                    if value >= 0:
                        edge += np.log(np.maximum(
                            p["edge_emit"][source, :, :, value], EPS))
                edges[t - 1] = edge
        else:
            log_init = np.log(np.maximum(p["prior"], EPS))
            for source, value in enumerate(obs[0]):
                if value >= 0:
                    log_init += np.log(np.maximum(
                        p["token_emit"][source, :, value], EPS))
            edges = np.empty((max(len(obs) - 1, 0), 2, 2), dtype=np.float64)
            for t in range(1, len(obs)):
                state = np.log(np.maximum(p["prior"], EPS))
                for source, value in enumerate(obs[t]):
                    if value >= 0:
                        state += np.log(np.maximum(
                            p["token_emit"][source, :, value], EPS))
                edges[t - 1] = np.broadcast_to(state[None, :], (2, 2))
        return log_init, edges

    def _forward_backward(self, obs):
        log_init, edges = self._log_terms(obs)
        length = len(obs)
        alpha = np.empty((length, 2), dtype=np.float64)
        beta = np.zeros((length, 2), dtype=np.float64)
        alpha[0] = log_init
        for t in range(1, length):
            alpha[t] = logsumexp(alpha[t - 1, :, None] + edges[t - 1], axis=0)
        log_z = float(logsumexp(alpha[-1]))
        for t in range(length - 2, -1, -1):
            beta[t] = logsumexp(edges[t] + beta[t + 1][None, :], axis=1)
        q = np.exp(alpha + beta - log_z)
        xi = np.empty((max(length - 1, 0), 2, 2), dtype=np.float64)
        for t in range(length - 1):
            xi[t] = np.exp(alpha[t, :, None] + edges[t]
                           + beta[t + 1][None, :] - log_z)
        log_all_zero = log_init[0]
        if length > 1:
            log_all_zero += float(edges[:, 0, 0].sum())
        p_all_zero = float(np.clip(np.exp(log_all_zero - log_z), 0.0, 1.0))
        return q, xi, log_z, p_all_zero

    @staticmethod
    def _condition_positive(q, xi, p_all_zero):
        normalizer = max(1.0 - p_all_zero, EPS)
        conditioned_q = q.copy()
        conditioned_q[:, 1] = q[:, 1] / normalizer
        conditioned_q[:, 0] = (q[:, 0] - p_all_zero) / normalizer
        conditioned_q = np.clip(conditioned_q, 0.0, 1.0)
        conditioned_q = _normalize(conditioned_q)
        conditioned_xi = xi.copy()
        if len(xi):
            conditioned_xi /= normalizer
            conditioned_xi[:, 0, 0] = (xi[:, 0, 0] - p_all_zero) / normalizer
            conditioned_xi = np.clip(conditioned_xi, 0.0, 1.0)
            conditioned_xi /= np.maximum(
                conditioned_xi.sum(axis=(1, 2), keepdims=True), EPS)
        return conditioned_q, conditioned_xi

    def fit(self, observations, labels):
        if not observations:
            raise ValueError("empty observations")
        for video_id, obs in observations.items():
            if obs.ndim != 2 or obs.shape[1] != self.n_sources or len(obs) == 0:
                raise ValueError(f"bad observation shape for {video_id}: {obs.shape}")
            if video_id not in labels or labels[video_id] not in (0, 1):
                raise ValueError(f"missing/bad bag label for {video_id}")
        posteriors = self._initialize_posteriors(observations, labels)
        for iteration in range(self.n_iter):
            self.params = self._m_step(observations, posteriors)
            updated = {}
            likelihood = 0.0
            positive_mass = []
            for video_id, obs in observations.items():
                q, xi, log_z, p0 = self._forward_backward(obs)
                likelihood += log_z
                if self.bag_conditioned:
                    if labels[video_id] == 0:
                        q = np.zeros_like(q); q[:, 0] = 1.0
                        xi = np.zeros_like(xi)
                        if len(xi): xi[:, 0, 0] = 1.0
                    else:
                        q, xi = self._condition_positive(q, xi, p0)
                        positive_mass.append(float(q[:, 1].sum()))
                updated[video_id] = (q, xi)
            posteriors = updated
            self.history.append({
                "iteration": iteration + 1,
                "observation_log_likelihood": float(likelihood),
                "mean_positive_bag_mass": (float(np.mean(positive_mass))
                                           if positive_mass else None),
            })
        self.params = self._m_step(observations, posteriors)
        return {video_id: q[:, 1].astype(np.float32)
                for video_id, (q, _) in posteriors.items()}

    def serializable(self):
        if self.params is None:
            raise RuntimeError("model is not fitted")
        return {
            "n_sources": self.n_sources, "n_bins": self.n_bins,
            "sequential": self.sequential,
            "bag_conditioned": self.bag_conditioned,
            "n_iter": self.n_iter, "smoothing": self.smoothing,
            "history": self.history,
            "parameters": {k: v.tolist() for k, v in self.params.items()},
        }
