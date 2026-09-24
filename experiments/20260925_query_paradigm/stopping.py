"""Decision-relevant stopping for the question policy (README section 2.4, revision of 2026-09-25 03:30).

The EIG of the best question stays near a constant (about 0.48 bits on HCS trial 0) because the full labelling y
always has many uncertain seconds, so an EIG threshold does not say when to stop. The rule here keeps the EIG
choice of the question and stops when that question is not expected to change the output: the expected reduction
of the squared-error risk of the per-second posterior,
    VOI(n) = sum_t Var_o( P(y_t = 1 | o_n = o, past) ) = sum_t E_o[(p_t^o - p_t)^2],
computed exactly. Given the past, o_n depends on (G, y) only through s_n, so
p_t^o = sum_s P(s_n = s | o, past) q_{t,s} with q_{t,s} = P(y_t = 1 | s_n = s, past): q_{.,0} = 0 (G = 0), and
q_{.,1}, q_{.,2} are two inference passes with s_n clamped. Stop rule: ask while VOI of the chosen question >= c.
"""
from __future__ import annotations

import numpy as np

import qtree

NEG = qtree.NEG


def clamped(vt, s, g, logA, node, state):
    """Second marginals with s_node clamped to state 1 (G=1, z=0) or 2 (G=1, z=1)."""
    la = logA.copy()
    keep = np.full(3, NEG)
    keep[state] = 0.0
    la[node] = la[node] + keep
    _pG, _m, p = vt.infer(s, g, la)
    return p


def voi(vt, s, g, logA, asker, pos, node, w):
    """Exact expected squared-error risk reduction of asking `node` (queryable position `pos`), state weights w (3,)."""
    q1 = clamped(vt, s, g, logA, node, 1) if w[1] > 1e-12 else np.zeros(vt.T)
    q2 = clamped(vt, s, g, logA, node, 2) if w[2] > 1e-12 else np.zeros(vt.T)
    joint = w[:, None] * asker.po_s[pos]                     # 3, O
    po = joint.sum(0)
    keep = po > 1e-12
    W = joint[:, keep] / po[keep]
    p_o = W[1][:, None] * q1[None] + W[2][:, None] * q2[None]   # O', T
    p_now = po[keep] @ p_o / po[keep].sum()
    return float(np.sum(po[keep][:, None] * (p_o - p_now[None]) ** 2))


def run_video(vt, s, g, asker, answers_v, max_calls):
    """policy.run_video (EIG choice, tree fusion) that also records the VOI of each chosen question before asking."""
    logA = np.zeros((vt.N, 3))
    pG, m, p = vt.infer(s, g, logA)
    scores, eigs, vois, asked = [p], [], [], []
    rem = np.ones(len(asker.q), dtype=bool)
    tr = vt.tr
    for _ in range(min(max_calls, len(asker.q))):
        cand = np.where(rem)[0]
        nodes = asker.q[cand]
        w = np.stack([np.full(len(cand), 1.0 - pG), pG * (1.0 - m[nodes]), pG * m[nodes]], axis=1)
        e = asker.eig(cand, w)
        j = int(np.argmax(e))
        pos, node = int(cand[j]), int(nodes[j])
        vois.append(voi(vt, s, g, logA, asker, pos, node, w[j]))
        eigs.append(float(e[j]))
        rem[pos] = False
        o = answers_v.get((int(tr["a"][node]), int(tr["b"][node])))
        if o is not None:
            logA[node] = asker.loglik(node, o)
        pG, m, p = vt.infer(s, g, logA)
        scores.append(p)
        asked.append(node)
    return {"scores": scores, "eig": eigs, "voi": vois, "asked": asked}
