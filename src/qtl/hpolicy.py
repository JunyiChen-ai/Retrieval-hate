"""Question policy under the hidden Markov tree answer model (hmt.py; experiments/20261109_tree_within/README.md
section 3), many videos at once. Same loop as cpolicy.run_batch with order "eig": each step one batched exact
inference (hmt.marginals_hmt) gives every video's posterior over seconds, node states and node biases; each video asks
its unasked node with the largest I(s_n; o_n), where the answer table of a candidate is mixed over its posterior
bias given each state, P(o | s) = sum_k P(b_n = k | s, answers so far) P(o | s, b_k). Returns per video the same
dict as cpolicy.run_batch (scores after 0..n calls, eig, asked, p_G, eig_sum, eig_top3, n_cand; voi zeros)."""
from __future__ import annotations

import numpy as np
import torch

from . import cpolicy, ctree, hmt, policy, qtree


def run_batch_hmt(model, store, vids, ham, chain, answers, max_calls, device, allowed=None):
    """ham: hmt.HMTAnswerModel; allowed: per video the node ids that may be asked (None = every queryable node)."""
    Ts = [store.T[v] for v in vids]
    Tm = max(Ts)
    S = torch.zeros(len(vids), Tm, dtype=torch.float64)
    G = torch.zeros(len(vids), dtype=torch.float64)
    fo = ctree.Forest(Ts, Tm)
    PHI = None
    for b, v in enumerate(vids):
        s, g, phi = policy.video_prior_nodes(model, store, v, device)
        S[b, :len(s)] = torch.from_numpy(s)
        G[b] = g
        if phi is not None:
            if PHI is None:
                PHI = torch.zeros(fo.N, dtype=torch.float64)
            PHI[int(fo.offs[b]):int(fo.offs[b]) + len(phi)] = torch.from_numpy(phi)
    S, G = S.to(device), G.to(device)
    PHI = None if PHI is None else PHI.to(device)
    params = ham.params(device)
    K = ham.K
    vts = [cpolicy._vt(T) for T in Ts]
    po = [np.exp(ham.outcome_tables(vt.length[vt.q_ids])) for vt in vts]          # per video (nq, K, 3, O)
    rem = [np.ones(len(vt.q_ids), dtype=bool) if allowed is None else np.isin(vt.q_ids, np.asarray(sorted(allowed[b])))
           for b, vt in enumerate(vts)]
    A3K = torch.zeros(fo.N, K, 3, dtype=torch.float64, device=device)
    out = [{"scores": [], "eig": [], "voi": [], "asked": [], "p_G": [], "eig_sum": [], "eig_top3": [], "n_cand": []}
           for _ in vids]
    for step in range(max_calls + 1):
        pG, J1, Q0, p = hmt.marginals_hmt(fo, S, G, A3K, chain, params, PHI)
        pG, J1, Q0, p = pG.cpu().numpy(), J1.cpu().numpy(), Q0.cpu().numpy(), p.cpu().numpy()
        for b, T in enumerate(Ts):
            if step <= len(vts[b].q_ids):
                out[b]["scores"].append(p[b, :T].copy())
                out[b]["p_G"].append(float(pG[b]))
        if step == max_calls:
            break
        chosen = []
        for b, (vt, off) in enumerate(zip(vts, fo.offs)):
            cand = np.where(rem[b])[0]
            if len(cand) == 0:
                continue
            nodes = vt.q_ids[cand]
            j1 = J1[off + nodes]                                                     # n, K, 2
            q0 = Q0[off + nodes]                                                     # n, K
            m0, m1 = j1[:, :, 0].sum(1), j1[:, :, 1].sum(1)
            qs = np.stack([q0 / np.clip(q0.sum(1, keepdims=True), 1e-300, None),
                           j1[:, :, 0] / np.clip(m0[:, None], 1e-300, None),
                           j1[:, :, 1] / np.clip(m1[:, None], 1e-300, None)], axis=1)  # n, 3, K
            pos = po[b][cand]                                                        # n, K, 3, O
            pos_mix = np.einsum("nsk,nkso->nso", qs, pos)                            # n, 3, O
            mm = m1 / np.clip(m0 + m1, 1e-300, None)
            w = np.stack([np.full(len(cand), 1.0 - pG[b]), pG[b] * (1.0 - mm), pG[b] * mm], axis=1)
            e = qtree.eig(np.log(np.clip(pos_mix, 1e-300, None)), w)
            j = int(np.argmax(e))
            chosen.append((b, int(cand[j]), int(nodes[j]), float(e[j])))
            out[b]["eig_sum"].append(float(e.sum()))
            out[b]["eig_top3"].append(float(np.sort(e)[-3:].sum()))
            out[b]["n_cand"].append(int(len(cand)))
        if not chosen:
            break
        for b, pos_i, node, e in chosen:
            rem[b][pos_i] = False
            tr = vts[b].tr
            o = answers[vids[b]].get((int(tr["a"][node]), int(tr["b"][node])))
            out[b]["asked"].append(node)
            out[b]["eig"].append(e)
            out[b]["voi"].append(0.0)
            if o is not None:
                ll = ham.loglik(np.asarray(o)[None], [float(tr["b"][node] - tr["a"][node])])[0]   # K, 3
                A3K[fo.offs[b] + node] = ll.to(device)
    return out
