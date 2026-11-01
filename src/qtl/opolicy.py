"""Question policy with the per-video answer offset (revision 6, answer_offset.py), many videos at once.

Same loop as cpolicy.run_batch with the EIG order: each step one exact inference per offset value
(ctree.marginals_mix) gives every video's posterior, each video asks its own EIG-best unasked node, the cached answer
of that node enters the per-offset answer factors. EIG is the information the answer gives about the node state
s in {0: G = 0, 1: G = 1 and z_n = 0, 2: G = 1 and z_n = 1} with the offset summed out:
    P(o | s) = sum_k P(b_k | s, answers) P(o | s, b_k),   EIG = H(o) - sum_s P(s) H(o | s).
copy_pi (optional, cpolicy): the copy mixture for nested questions, applied under every offset value, in the EIG
and in the likelihood of the answer read.
Returns per video: scores (after 0..n calls), eig, asked, p_G, voi (zeros; stop_rules.rule_values reads the key).
"""
from __future__ import annotations

import numpy as np
import torch

from . import cpolicy, ctree, policy, qtree


def _copy_mix(po, nodes, tr, copy_pi, asked_o):
    """po (K, n, 3, O) -> the copy mixture for the nodes nested with an earlier answer (cpolicy._answer_tables)."""
    if copy_pi is None or not asked_o:
        return po
    po = po.copy()
    for i, n in enumerate(nodes):
        src = cpolicy._nested_source(tr, int(n), asked_o, None)
        if src is not None:
            pi = float(copy_pi(float(min(tr["b"][n] - tr["a"][n], tr["b"][src[0]] - tr["a"][src[0]]))))
            po[:, i] *= (1.0 - pi)
            po[:, i, :, src[1]] += pi
    return po


def run_batch(model, store, vids, oam, chain, answers, max_calls, device, copy_pi=None):
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
    vts = [cpolicy._vt(T) for T in Ts]
    lpo = [oam.outcome_tables(len(vt.q_ids)) for vt in vts]                  # per video (K, nq, 3, O)
    po_all = [np.exp(x) for x in lpo]
    qpos = [{int(n): i for i, n in enumerate(vt.q_ids)} for vt in vts]
    K = oam.K
    A3K = torch.zeros(K, fo.N, 3, dtype=torch.float64, device=device)
    out = [{"scores": [], "eig": [], "voi": [], "asked": [], "p_G": []} for _ in vids]
    rem = [np.ones(len(vt.q_ids), dtype=bool) for vt in vts]
    asked_o = [[] for _ in vids]
    for step in range(max_calls + 1):
        pG, m, p, post, mk = ctree.marginals_mix(fo, S, G, A3K, oam.logw, chain, PHI)
        pG, p, post, mk = pG.cpu().numpy(), p.cpu().numpy(), post.cpu().numpy(), mk.cpu().numpy()
        for b, T in enumerate(Ts):
            if step <= len(vts[b].q_ids):
                out[b]["scores"].append(p[b, :T].copy())
                out[b]["p_G"].append(float(pG[b]))
        if step == max_calls:
            break
        for b, (vt, off) in enumerate(zip(vts, fo.offs)):
            cand = np.where(rem[b])[0]
            if len(cand) == 0:
                continue
            nodes = vt.q_ids[cand]
            mm = mk[:, off + nodes]                                           # K, n
            w = np.stack([np.repeat(post[:, b, 0][:, None], len(cand), 1), post[:, b, 1][:, None] * (1.0 - mm),
                          post[:, b, 1][:, None] * mm], axis=-1)               # K, n, 3
            po = _copy_mix(po_all[b][:, cand], nodes, vt.tr, copy_pi, asked_o[b])   # K, n, 3, O
            joint = w[..., None] * po                                         # K, n, 3, O
            p_o = joint.sum((0, 2))                                           # n, O
            ps = w.sum(0)                                                     # n, 3
            p_os = joint.sum(0) / np.clip(ps[..., None], 1e-300, None)        # n, 3, O
            h = -np.sum(p_o * np.log(np.clip(p_o, 1e-300, None)), axis=1)
            h_s = -np.sum(p_os * np.log(np.clip(p_os, 1e-300, None)), axis=2)
            e = (h - np.sum(ps * h_s, axis=1)) / np.log(2.0)
            j = int(np.argmax(e))
            node = int(nodes[j])
            rem[b][cand[j]] = False
            tr = vt.tr
            o = answers[vids[b]].get((int(tr["a"][node]), int(tr["b"][node])))
            o_idx = None
            if o is not None:
                o_idx = int(qtree.answer_index(np.asarray(o)[oam.cats]))
                pk = po_all[b][:, qpos[b][node], :, o_idx]                    # K, 3
                src = cpolicy._nested_source(tr, node, asked_o[b], None) if copy_pi is not None else None
                if src is not None:
                    pi = float(copy_pi(float(min(tr["b"][node] - tr["a"][node], tr["b"][src[0]] - tr["a"][src[0]]))))
                    pk = (1.0 - pi) * pk + pi * (o_idx == src[1])
                A3K[:, off + node] = torch.as_tensor(np.log(np.clip(pk, 1e-300, None)), dtype=torch.float64,
                                                     device=device)
            asked_o[b].append((node, o_idx))
            out[b]["eig"].append(float(e[j]))
            out[b]["voi"].append(0.0)
            out[b]["asked"].append(node)
    return {v: r for v, r in zip(vids, out)}
