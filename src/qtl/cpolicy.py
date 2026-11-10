"""Question policy for the chain prior (revision 1, README section 7), run for many videos at once: each step one
batched exact inference (ctree.marginals) gives every video's posterior; each video asks its own EIG-best
unasked node; with record_voi the chosen questions' expected squared-error risk reduction (stopping rule,
stopping.py) is computed from two clamped batched passes before the answers are read.
Returns per video the same dict as policy.run_video: scores (after 0..n calls), eig, voi, asked, p_G."""
from __future__ import annotations

import types

import numpy as np
import torch

from . import ctree
from . import policy
from . import qtree

BIG = ctree.BIG


def _vt(T):
    tr = qtree.tree(T)
    return types.SimpleNamespace(T=T, tr=tr, q_ids=np.where(tr["queryable"])[0],
                                 length=(tr["b"] - tr["a"]).astype(np.float64))


def _flat(p0, llr, cnt):
    lp = np.log(np.clip(p0, 1e-12, 1 - 1e-12)) - np.log1p(-np.clip(p0, 1e-12, 1 - 1e-12))
    return 1.0 / (1.0 + np.exp(-(lp + llr / np.maximum(cnt, 1))))


def _ancestors(tr):
    """(n_nodes, depth + 1) ancestor at each depth (-1 below the node's own depth), for tree distances."""
    D = int(tr["depth"].max())
    anc = np.full((len(tr["a"]), D + 1), -1, dtype=np.int64)
    for n in range(len(tr["a"])):
        m = n
        while m >= 0:
            anc[n, tr["depth"][m]] = m
            m = tr["parent"][m]
    return anc


def _min_distance(tr, anc, cand, asked):
    """Tree distance from each candidate node to its nearest asked node (cand (n,), asked list)."""
    dep = tr["depth"]
    best = np.full(len(cand), np.iinfo(np.int64).max)
    for m in asked:
        same = (anc[cand] == anc[m][None, :]) & (anc[cand] >= 0)
        lca_depth = same.sum(1) - 1                                  # common ancestors form a prefix from the root
        best = np.minimum(best, dep[cand] + dep[m] - 2 * lca_depth)
    return best


def _nested_source(tr, node, asked_b, o_len):
    """The earlier-asked node nested with `node` (containing it or inside it) that has an answer, nearest in scale
    (revision 5 step 2, README section 17.2); None if there is none."""
    a_, b_ = tr["a"][node], tr["b"][node]
    best, best_d = None, None
    for m, om in asked_b:
        if om is None:
            continue
        ma, mb = tr["a"][m], tr["b"][m]
        if (ma <= a_ and b_ <= mb) or (a_ <= ma and mb <= b_):
            d = abs(np.log(b_ - a_) - np.log(mb - ma))
            if best is None or d < best_d:
                best, best_d = (m, om), d
    return best


def _answer_tables(asker, cand, nodes, tr, copy_pi, asked_b):
    """(n, 3, O) P(o | s) of the candidates, with the copy mixture for nodes nested with an earlier answer."""
    po = asker.po_s[cand].copy()
    if copy_pi is not None and asked_b:
        for i, n in enumerate(nodes):
            src = _nested_source(tr, int(n), asked_b, None)
            if src is not None:                          # pi bucketed by the shorter of the two (the child)
                pi = float(copy_pi(float(min(tr["b"][n] - tr["a"][n], tr["b"][src[0]] - tr["a"][src[0]]))))
                po[i] *= (1.0 - pi)
                po[i][:, src[1]] += pi
    return po


def _expected_error_reduction(asker, cand, nodes, vt, p, pG, mm, copy_pi, asked_b):
    """Expected reduction of the per-second squared error sum_t (p_t - y_t)^2 from asking each candidate node, i.e.
    sum_o P(o) sum_t (E[p_t | o] - p_t)^2 (the score is a probability, so this is the expected Brier gain).

    Uses the exact current posteriors p_t = pG r_t (r_t = P(y_t = 1 | G = 1, answers)) and node marginals
    m_n = P(z_n = 1 | G = 1, answers), and one approximation: a node's answer moves the seconds outside the node only
    through P(G = 1 | answers) (chain correlation across the node boundary is ignored). Inside the node the update is
    exact: y_t = 1 implies z_n = 1, so E[y_t | G = 1, z_n = 1] = r_t / m_n and E[y_t | z_n = 0] = 0. Closed form:
        (1 - pG)^2 R_out sum_o P(o) (1 - P(o | s=0) / P(o))^2 + pG^2 R_in sum_o P(o) (P(o | s=2) / P(o) - 1)^2,
    with R_in / R_out the sums of r_t^2 inside / outside the node."""
    tr = vt.tr
    po = _answer_tables(asker, cand, nodes, tr, copy_pi, asked_b)                     # n, 3, O
    w = np.stack([np.full(len(cand), 1.0 - pG), pG * (1.0 - mm), pG * mm], axis=1)
    P = np.einsum("ns,nso->no", w, po)                                                # n, O
    P = np.clip(P, 1e-300, None)
    r = np.asarray(p, dtype=np.float64) / max(float(pG), 1e-12)
    c2 = np.concatenate([[0.0], np.cumsum(r * r)])
    R_in = c2[tr["b"][nodes]] - c2[tr["a"][nodes]]
    R_out = c2[-1] - R_in
    out_term = (P * (1.0 - po[:, 0] / P) ** 2).sum(1)
    in_term = (P * (po[:, 2] / P - 1.0) ** 2).sum(1)
    return (1.0 - pG) ** 2 * R_out * out_term + pG ** 2 * R_in * in_term


def _abs_sum(sorted_ref, prefix, x):
    """sum_u |x_i - ref_u| for every x_i, given ref sorted ascending and its prefix sums (prefix[0] = 0)."""
    k = np.searchsorted(sorted_ref, x)
    n = len(sorted_ref)
    return x * k - prefix[k] + (prefix[-1] - prefix[k]) - x * (n - k)


def _expected_spread_gain(po, tr, nodes, p, pG, mm):
    """Diagnostic (experiments/20261109_tree_within README section 7): expected gain in ranking accuracy from asking
    each candidate node. Ranking accuracy of a set of probabilities = mean over pairs of |q_i - q_j| (for any pair,
    ordering by the posterior is right with probability 1/2 + |q_i - q_j| / 2 under independent marginals). Two
    pairs sets: videos (P(G = 1) of this video against a uniform reference, E|pG - U| = (pG^2 + (1 - pG)^2) / 2) and
    seconds within the video given G = 1 (r_t = P(y_t = 1 | G = 1)). After answer o of node n: P(G = 1 | o) by Bayes;
    r_t for t in n scaled by alpha_o = P(o | s = 2) / P(o | G = 1) (exact: y_t = 1 implies z_n = 1), r_t outside n
    unchanged (chain correlation across the node boundary ignored, as in _expected_error_reduction). Value = expected
    video term gain + P(G = 1) x expected within term gain (outcomes under G = 1), both per pair."""
    T = len(p)
    r = np.clip(np.asarray(p, dtype=np.float64) / max(float(pG), 1e-12), 0.0, 1.0)
    P1 = (1.0 - mm)[:, None] * po[:, 1] + mm[:, None] * po[:, 2]                     # n, O: P(o | G = 1)
    Po = (1.0 - pG) * po[:, 0] + pG * P1
    P1c, Poc = np.clip(P1, 1e-300, None), np.clip(Po, 1e-300, None)
    pGo = pG * P1 / Poc
    f = lambda x: 0.5 * (x * x + (1.0 - x) ** 2)
    video = (Po * f(pGo)).sum(1) - f(pG)
    alpha = po[:, 2] / P1c                                                            # n, O
    rs = np.sort(r)
    pre = np.concatenate([[0.0], np.cumsum(rs)])
    npairs = max(T * (T - 1) / 2.0, 1.0)
    within = np.zeros(len(nodes))
    for i, n in enumerate(nodes):
        a_, b_ = int(tr["a"][n]), int(tr["b"][n])
        rin = np.sort(r[a_:b_])
        k = len(rin)
        if k == T:                                                                    # root: no contrast in-video
            continue
        prin = np.concatenate([[0.0], np.cumsum(rin)])
        ii = float((rin * (2.0 * np.arange(k) - (k - 1))).sum())                      # sum over in-node pairs
        cross1 = float((_abs_sum(rs, pre, rin) - _abs_sum(rin, prin, rin)).sum())
        g = 0.0
        for o in range(po.shape[2]):
            x = alpha[i, o] * rin
            cross = float((_abs_sum(rs, pre, x) - _abs_sum(rin, prin, x)).sum())
            g += P1[i, o] * ((alpha[i, o] - 1.0) * ii + cross - cross1)
        within[i] = g / npairs
    return video + pG * within


def _cover_feasible(tr, cov_b, q, cand):
    """Diagnostic cover (run_batch cover=): the candidates (positions into q) that keep an exact cover with the
    remaining cover calls possible, and per kept candidate (frontier ancestor, siblings on the path below it)."""
    fine, front, left = cov_b
    need = left - 1
    tot = sum(int(fine[f]) for f in front)
    keep, path = [], []
    for c in cand:
        n = int(q[c])
        x, sibs = n, []
        while x not in front and x > 0:
            p_ = int(tr["parent"][x])
            sibs.append(int(tr["left"][p_] + tr["right"][p_] - x))
            x = p_
        if x not in front or any(fine[s_] == 0 for s_ in sibs):
            continue
        minc = len(front) - 1 + len(sibs)
        maxc = tot - int(fine[x]) + sum(int(fine[s_]) for s_ in sibs)
        if minc <= need <= maxc:
            keep.append(c)
            path.append((x, sibs))
    if not keep:                                 # cannot happen when the cover size is feasible; stay unconstrained
        cov_b[2] = 0
        return cand, None
    return np.asarray(keep, dtype=cand.dtype), path


def _plan_disjoint(tr, q, cand, val, K):
    """Budget-aware plan (revision 7): among the candidate nodes (cand: positions into q, ascending; val their values
    under the current posterior) the at most K pairwise disjoint ones with the largest total value. Exact max-plus
    dynamic programme on the tree: best(n, k) = the largest total of k disjoint candidates inside node n =
    max(value of n itself if k = 1, max_j best(left, j) + best(right, k - j)). Returns positions into cand."""
    K = int(min(K, len(cand)))
    if K <= 0:
        return np.zeros(0, dtype=np.int64)
    nq = len(q)
    qpos = np.full(len(tr["a"]), -1, dtype=np.int64)
    qpos[q] = np.arange(nq)
    V = np.full(nq, -1e18)
    V[cand] = np.maximum(np.asarray(val, dtype=np.float64), 0.0) + 1e-12   # a value of exactly 0 still fills the plan
    best = np.full((nq + 1, K + 1), -1e18)       # last row: a child that cannot be asked (index -1)
    best[:, 0] = 0.0
    own = np.zeros(nq, dtype=bool)
    split = np.zeros((nq, K + 1), dtype=np.int64)
    left, right = tr["left"], tr["right"]
    ks = np.arange(K + 1)
    for i in range(nq - 1, -1, -1):              # q is breadth-first: children after their parent
        n = q[i]
        M = best[qpos[left[n]]][:, None] + best[qpos[right[n]]][None, :]
        comb = np.full(K + 1, -1e18)
        arg = np.zeros(K + 1, dtype=np.int64)
        for k in range(K + 1):
            vals = M[ks[:k + 1], k - ks[:k + 1]]
            arg[k] = int(np.argmax(vals))
            comb[k] = vals[arg[k]]
        if V[i] > comb[1]:
            comb[1] = V[i]
            own[i] = True
        best[i], split[i] = comb, arg
    out, stack = [], [(0, int(np.argmax(best[0])))]
    while stack:
        i, k = stack.pop()
        if k == 0 or i < 0:
            continue
        if k == 1 and own[i]:
            out.append(i)
            continue
        n, j = q[i], int(split[i, k])
        stack.append((int(qpos[left[n]]), j))
        stack.append((int(qpos[right[n]]), k - j))
    return np.searchsorted(cand, np.sort(np.asarray(out, dtype=np.int64)))


def run_batch(model, store, vids, am, chain, answers, cats, max_calls, device, order="eig", fusion="tree",
              record_voi=False, allowed=None, answer_ll=None, make_asker=None, no_nested=False, copy_pi=None,
              copy_mode="both", record_erm=0, no_adjacent=False, dist_weight=None, prior_scale=1.0,
              policy_prior_scale=None, budgets=None, cover=None, prior_weight=None, plan=None, priors=None,
              first=None):
    """allowed (diagnostic arm, README section 12): per video, the node ids that may be asked (None = every
    queryable node). answer_ll (diagnostic, concern_diagnostics.py): function (video index, node id) -> the (3,)
    log-likelihood of the observation to use instead of the cached VLM answer, or None for no observation
    (None = the cached answers, the method). make_asker (diagnostic, answer_model_ceiling.py): function (video index,
    video tree) -> an Asker with per-node outcome tables (None = policy.Asker under `am`, the method). no_nested
    (diagnostic, asking_check.py): after each question, the nodes that contain or lie inside it can no longer be
    asked. copy_pi (revision 5 step 2, README section 17.2): copy-type persistent noise for nested questions,
    None = the method; else a function (child length in seconds) -> pi in [0, 1): the answer of a node nested with
    an earlier-asked answered node m (nearest in scale) is, with probability pi, a copy of m's answer and otherwise
    drawn from the answer model: P(o | s, o_m) = pi 1[o = o_m] + (1 - pi) P(o | s). The mixture enters both the EIG
    of the candidates and the likelihood of the answer read; the tree inference is unchanged (per-node factor).
    copy_mode (README section 17.2 variants): "both" = the mixture enters the EIG and the likelihood; "eig" = the
    EIG only (the answer read is scored by the plain answer model); "lik" = the likelihood only.
    record_erm (README 17.4, the nonmyopic stopping statistic): K > 0 samples, before every call, K complete answer
    sets for ALL still-askable nodes from the model's posterior predictive (video state ~ P(G), node states from their
    marginals independently, outcomes from the answer model), runs the tree inference on each and records the mean
    over samples of the per-second log-odds change (mean over seconds: erm; sum over seconds: ermsum) between the
    current posterior and the posterior with everything answered: the expected remaining movement of the scores.
    budgets (diagnostic, experiments/20261109_tree_within): per video, the number of calls (None = max_calls).
    cover (diagnostic, experiments/20261109_tree_within): None = the method; "plain" = budget-aware cover: the first
    C = min(budget, most disjoint queryable nodes) questions are pairwise disjoint tree nodes that together cover the
    whole video (each candidate must leave an uncovered region that C - 1 - asked disjoint queryable nodes can still
    cover exactly), chosen by the order's objective; later calls (if any) unconstrained. "root" = the root first,
    then the same cover with C = min(budget - 1, ...).
    prior_weight (revision 7, prior_weight.py): None = the prior as the network gives it; (a, b, c) = the label-free
    weights estimated on the validation answers: video logit a g + b, per-second logits c s, node potentials c phi,
    used for the posterior, the EIG and every recorded statistic alike.
    plan (revision 7): None = greedy, one question at a time. "disjoint" = budget-aware: before every call the
    remaining calls of the video's budget are planned as the pairwise disjoint unasked nodes with the largest total
    value under the current posterior (_plan_disjoint, value = the order's objective); the planned node with the
    largest value is asked; a node nested with an asked node is not asked while a node disjoint from all asked
    nodes is left (nested answers repeat each other, which the answer model does not describe). The plan is rebuilt
    after every answer. When no disjoint node is left and calls remain, the policy falls back to the greedy choice
    among the nested nodes (refining what was asked).
    priors: None = run the prior network here; {video: (s, g, phi)} = its outputs computed before
    (prior_weight.collect), so that the weight fit and the policy share one forward pass.
    first (revision 7, first_level): None = the method of revisions 2-6; else per video the node ids that are asked
    before any other node (in the order's objective among them); once they are all asked, the policy continues
    over all remaining nodes. With the nodes of one tree depth this is the breadth-first start."""
    assert copy_mode in ("both", "eig", "lik")
    assert plan in (None, "disjoint") and (plan is None or order in ("eig", "eer", "spread", "eig_loc"))
    copy_eig = copy_pi is not None and copy_mode in ("both", "eig")
    copy_lik = copy_pi is not None and copy_mode in ("both", "lik")
    Ts = [store.T[v] for v in vids]
    Tm = max(Ts)
    S = torch.zeros(len(vids), Tm, dtype=torch.float64)
    G = torch.zeros(len(vids), dtype=torch.float64)
    fo = ctree.Forest(Ts, Tm)
    PHI = None                                   # revision 4 node potentials (README section 15)
    for b, v in enumerate(vids):
        s, g, phi = policy.video_prior_nodes(model, store, v, device) if priors is None else priors[v]
        S[b, :len(s)] = torch.from_numpy(s)
        G[b] = g
        if phi is not None:
            if PHI is None:
                PHI = torch.zeros(fo.N, dtype=torch.float64)
            PHI[int(fo.offs[b]):int(fo.offs[b]) + len(phi)] = torch.from_numpy(phi)
    if prior_scale != 1.0:                       # diagnostic (experiments/20261109_tree_within): softer prior shape
        S, PHI = S * prior_scale, None if PHI is None else PHI * prior_scale
    if prior_weight is not None:                 # revision 7: label-free prior weights (prior_weight.py)
        w_a, w_b, w_c = (float(x) for x in prior_weight)
        G = w_a * G + w_b
        S, PHI = S * w_c, None if PHI is None else PHI * w_c
    S, G = S.to(device), G.to(device)
    PHI = None if PHI is None else PHI.to(device)
    vts = [_vt(T) for T in Ts]
    askers = [policy.Asker(vt, am, cats) if make_asker is None else make_asker(b, vt)
              for b, vt in enumerate(vts)]
    A3 = torch.zeros(fo.N, 3, dtype=torch.float64, device=device)
    out = [{"scores": [], "eig": [], "voi": [], "asked": [], "p_G": [], "eig_sum": [], "eig_top3": [], "n_cand": [],
            "erm": [], "ermsum": []}
           for _ in vids]                      # eig_sum / eig_top3: over all candidate questions at that step (README 17.4)
    rng = np.random.default_rng(0)

    def _logit(x):
        x = np.clip(np.asarray(x, dtype=np.float64), 1e-6, 1.0 - 1e-6)
        return np.log(x) - np.log1p(-x)
    rem = [np.ones(len(a.q), dtype=bool) if allowed is None else np.isin(a.q, np.asarray(sorted(allowed[b])))
           for b, a in enumerate(askers)]
    llr = [np.zeros(T) for T in Ts]
    cnt = [np.zeros(T) for T in Ts]
    first_mask = None if first is None else [np.isin(a.q, np.asarray(sorted(first[b]), dtype=np.int64))
                                             for b, a in enumerate(askers)]
    nest = [np.zeros(len(a.q), dtype=bool) for a in askers]   # plan: nodes nested with an asked node
    asked_o = [[] for _ in vids]                 # per video: (node, outcome index or None) of the asked nodes
    anc = [None for _ in vids]
    p0 = None
    budgets = [max_calls] * len(vids) if budgets is None else [int(x) for x in budgets]
    cov = [None for _ in vids]                   # diagnostic cover state: [fine, frontier set, cover calls left]
    if cover is not None:
        assert cover in ("plain", "root")
        for b, vt in enumerate(vts):
            tr = vt.tr
            fine = np.zeros(len(tr["a"]), dtype=np.int64)    # most disjoint queryable nodes covering the node exactly
            for n in range(len(tr["a"]) - 1, -1, -1):
                l_, r_ = tr["left"][n], tr["right"][n]
                if l_ >= 0 and fine[l_] > 0 and fine[r_] > 0:
                    fine[n] = fine[l_] + fine[r_]
                elif tr["queryable"][n]:
                    fine[n] = 1
            first = 1 if cover == "root" and budgets[b] >= 1 else 0
            cov[b] = [fine, {0}, min(budgets[b] - first, int(fine[0]))]
    for step in range(max_calls + 1):
        pG, m, p = ctree.marginals(fo, S, G, A3, chain, PHI)
        pG, m, p = pG.cpu().numpy(), m.cpu().numpy(), p.cpu().numpy()
        pGq, mq, pq = pG, m, p
        if policy_prior_scale is not None:       # diagnostic (experiments/20261109_tree_within): the questions are
            pGq, mq, pq = ctree.marginals(       # chosen under the prior shape scaled by policy_prior_scale; the
                fo, S * policy_prior_scale, G, A3, chain,       # scores keep the full prior
                None if PHI is None else PHI * policy_prior_scale)
            pGq, mq, pq = pGq.cpu().numpy(), mq.cpu().numpy(), pq.cpu().numpy()
        if p0 is None:
            p0 = [p[b, :T].copy() for b, T in enumerate(Ts)]
        for b, T in enumerate(Ts):
            if step <= len(askers[b].q):
                out[b]["scores"].append(p[b, :T].copy() if fusion == "tree" else _flat(p0[b], llr[b], cnt[b]))
                out[b]["p_G"].append(float(pG[b]))
        if record_erm > 0 and step < max_calls:
            acc = [np.zeros(2) for _ in vids]
            for _k in range(int(record_erm)):
                A3s = A3.clone()
                for b, (asker, off) in enumerate(zip(askers, fo.offs)):
                    cand = np.where(rem[b])[0]
                    if len(cand) == 0:
                        continue
                    nodes = asker.q[cand]
                    mm = m[off + nodes]
                    gs = rng.random() < pG[b]
                    st = np.where(rng.random(len(cand)) < mm, 2, 1) if gs else np.zeros(len(cand), dtype=int)
                    po = asker.po_s[cand]                                    # n, 3, O
                    cum = np.cumsum(po[np.arange(len(cand)), st], axis=1)
                    oi = np.minimum((rng.random(len(cand))[:, None] > cum).sum(1), po.shape[2] - 1)
                    ll = np.log(np.clip(po[np.arange(len(cand)), :, oi], 1e-300, None))   # n, 3
                    A3s[off + torch.as_tensor(nodes, device=device)] = torch.as_tensor(ll, dtype=torch.float64, device=device)
                _, _, ps = ctree.marginals(fo, S, G, A3s, chain, PHI)
                ps = ps.cpu().numpy()
                for b, T in enumerate(Ts):
                    d = np.abs(_logit(ps[b, :T]) - _logit(p[b, :T]))
                    acc[b] += np.array([d.mean(), d.sum()])
            for b in range(len(vids)):
                if rem[b].any():
                    out[b]["erm"].append(float(acc[b][0] / record_erm))
                    out[b]["ermsum"].append(float(acc[b][1] / record_erm))
        if step == max_calls:
            break
        chosen = []
        for b, (asker, off) in enumerate(zip(askers, fo.offs)):
            if len(asked_o[b]) >= budgets[b]:
                continue
            cand = np.where(rem[b])[0]
            if len(cand) == 0:
                continue
            cov_path = None
            planned = False
            if first_mask is not None and first_mask[b][cand].any():     # revision 7: the first-level nodes first
                cand = cand[first_mask[b][cand]]
            if plan == "disjoint" and not nest[b][cand].all():
                cand, planned = cand[~nest[b][cand]], True
            if cov[b] is not None and cover == "root" and not asked_o[b] and asker.q[0] == 0 and rem[b][0]:
                cand = cand[asker.q[cand] == 0]
            elif cov[b] is not None and cov[b][2] > 0:
                cand, cov_path = _cover_feasible(vts[b].tr, cov[b], asker.q, cand)
            nodes = asker.q[cand]
            mm = mq[off + nodes]
            w = np.stack([np.full(len(cand), 1.0 - pGq[b]), pGq[b] * (1.0 - mm), pGq[b] * mm], axis=1)
            if order == "eer":
                e = _expected_error_reduction(asker, cand, nodes, vts[b], p[b, :Ts[b]], pG[b], mm,
                                              copy_pi if copy_eig else None, asked_o[b])
                j = int(np.argmax(e))
            elif order == "eig" and copy_eig and asked_o[b]:
                po = asker.po_s[cand].copy()                              # n, 3, O
                tr_b = vts[b].tr
                for i, n in enumerate(nodes):
                    src = _nested_source(tr_b, int(n), asked_o[b], None)
                    if src is not None:                  # pi bucketed by the shorter of the two (the child)
                        pi = float(copy_pi(float(min(tr_b["b"][n] - tr_b["a"][n],
                                                     tr_b["b"][src[0]] - tr_b["a"][src[0]]))))
                        po[i] *= (1.0 - pi)
                        po[i][:, src[1]] += pi
                e = qtree.eig(np.log(np.clip(po, 1e-300, None)), w)
                j = int(np.argmax(e))
            elif order == "eig" and dist_weight is not None:
                # diagnostic (experiments/20261109_tree_within): an answer whose nearest asked node is at tree
                # distance d carries the fraction w(d) = 1 - r(d)^2 of its information (r: answer correlation)
                e = asker.eig(cand, w)
                if asked_o[b]:
                    if anc[b] is None:
                        anc[b] = _ancestors(vts[b].tr)
                    dd = _min_distance(vts[b].tr, anc[b], nodes, [m for m, _ in asked_o[b]])
                    e = e * dist_weight[np.minimum(dd, len(dist_weight) - 1)]
                j = int(np.argmax(e))
            elif order == "eig":
                e = asker.eig(cand, w)
                j = int(np.argmax(e))
            elif order == "spread":              # diagnostic (experiments/20261109_tree_within): ranking gain
                e = _expected_spread_gain(asker.po_s[cand], vts[b].tr, nodes, pq[b, :Ts[b]], pGq[b], mm)
                j = int(np.argmax(e))
            elif order == "eig_loc":             # diagnostic (experiments/20261109_tree_within): I(o_n; y | G = 1)
                e = asker.eig(cand, np.stack([np.zeros(len(cand)), 1.0 - mm, mm], axis=1))
                j = int(np.argmax(e))
            else:
                e = asker.eig(cand[:1], w[:1])
                j = 0
            if planned:
                e = np.asarray(e, dtype=np.float64)
                sel = _plan_disjoint(vts[b].tr, asker.q, cand, e, budgets[b] - len(asked_o[b]))
                j = int(sel[int(np.argmax(e[sel]))])
            chosen.append((b, int(cand[j]), int(nodes[j]), w[j], float(e[j])))
            if cov_path is not None:             # diagnostic cover: the frontier loses the chosen node's frontier
                f_, sibs = cov_path[j]           # ancestor and gains the siblings along the path below it
                cov[b][1].discard(f_)
                cov[b][1].update(sibs)
                cov[b][2] -= 1
            e_all = np.asarray(e, dtype=np.float64)
            out[b]["eig_sum"].append(float(e_all.sum()))
            out[b]["eig_top3"].append(float(np.sort(e_all)[-3:].sum()))
            out[b]["n_cand"].append(int(len(cand)))
        if not chosen:
            break
        if record_voi:
            q = {}
            for state in (1, 2):
                Ac = A3.clone()
                for b, pos, node, w, e in chosen:
                    keep = torch.full((3,), -BIG, dtype=torch.float64, device=device)
                    keep[state] = 0.0
                    Ac[fo.offs[b] + node] += keep
                _, _, pc = ctree.marginals(fo, S, G, Ac, chain, PHI)
                q[state] = pc.cpu().numpy()
            for b, pos, node, w, e in chosen:
                T = Ts[b]
                joint = w[:, None] * askers[b].po_s[pos]
                po = joint.sum(0)
                k = po > 1e-12
                W = joint[:, k] / po[k]
                p_o = W[1][:, None] * q[1][b, :T][None] + W[2][:, None] * q[2][b, :T][None]
                p_now = po[k] @ p_o / po[k].sum()
                out[b]["voi"].append(float(np.sum(po[k][:, None] * (p_o - p_now[None]) ** 2)))
        for b, pos, node, w, e in chosen:
            rem[b][pos] = False
            tr = vts[b].tr
            if no_nested or plan == "disjoint":
                qa, qb = tr["a"][askers[b].q], tr["b"][askers[b].q]
                a_, b_ = tr["a"][node], tr["b"][node]
                nested = ((a_ <= qa) & (qb <= b_)) | ((qa <= a_) & (b_ <= qb))
                if no_nested:
                    rem[b] &= ~nested
                else:
                    nest[b] |= nested
            if no_adjacent:                      # diagnostic (experiments/20261109_tree_within): nodes within tree
                if anc[b] is None:               # distance no_adjacent (1: parent and children) of the asked node
                    anc[b] = _ancestors(tr)
                dd = _min_distance(tr, anc[b], askers[b].q, [node])
                rem[b] &= ~(dd <= int(no_adjacent))
            o_idx = None
            if answer_ll is None:
                o = answers[vids[b]].get((int(tr["a"][node]), int(tr["b"][node])))
                ll = None if o is None else askers[b].loglik(node, o)
                if o is not None:
                    o_idx = int(qtree.answer_index(np.asarray(o)[askers[b].cats]))
                    if copy_lik:
                        src = _nested_source(tr, node, asked_o[b], None)
                        if src is not None:              # pi bucketed by the shorter of the two (the child)
                            pi = float(copy_pi(float(min(tr["b"][node] - tr["a"][node],
                                                         tr["b"][src[0]] - tr["a"][src[0]]))))
                            if pi > 0.0:
                                ll = np.logaddexp(np.log1p(-pi) + ll,
                                                  np.log(pi) + (0.0 if o_idx == src[1] else -np.inf))
            else:
                ll = answer_ll(b, node)
            if dist_weight is not None and ll is not None and asked_o[b]:
                if anc[b] is None:
                    anc[b] = _ancestors(tr)
                dd = int(_min_distance(tr, anc[b], np.array([node]), [m for m, _ in asked_o[b]])[0])
                ll = ll * float(dist_weight[min(dd, len(dist_weight) - 1)])
            asked_o[b].append((node, o_idx))
            if ll is not None:
                A3[fo.offs[b] + node] = torch.as_tensor(ll, dtype=torch.float64, device=device)
                a_, b_ = int(tr["a"][node]), int(tr["b"][node])
                llr[b][a_:b_] += ll[2] - ll[0]
                cnt[b][a_:b_] += 1
            out[b]["eig"].append(e)
            out[b]["asked"].append(node)
    return {v: r for v, r in zip(vids, out)}
