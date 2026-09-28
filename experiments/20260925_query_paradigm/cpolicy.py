"""Question policy for the chain prior (revision 1, README section 7), run for many videos at once: each step one
batched exact inference (ctree.marginals) gives every video's posterior; each video asks its own EIG-best
unasked node; with record_voi the chosen questions' expected squared-error risk reduction (stopping rule,
stopping.py) is computed from two clamped batched passes before the answers are read.
Returns per video the same dict as policy.run_video: scores (after 0..n calls), eig, voi, asked, p_G."""
from __future__ import annotations

import types

import numpy as np
import torch

import ctree
import policy
import qtree

BIG = ctree.BIG


def _vt(T):
    tr = qtree.tree(T)
    return types.SimpleNamespace(T=T, tr=tr, q_ids=np.where(tr["queryable"])[0],
                                 length=(tr["b"] - tr["a"]).astype(np.float64))


def _flat(p0, llr, cnt):
    lp = np.log(np.clip(p0, 1e-12, 1 - 1e-12)) - np.log1p(-np.clip(p0, 1e-12, 1 - 1e-12))
    return 1.0 / (1.0 + np.exp(-(lp + llr / np.maximum(cnt, 1))))


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


def run_batch(model, store, vids, am, chain, answers, cats, max_calls, device, order="eig", fusion="tree",
              record_voi=False, allowed=None, answer_ll=None, make_asker=None, no_nested=False, copy_pi=None):
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
    of the candidates and the likelihood of the answer read; the tree inference is unchanged (per-node factor)."""
    Ts = [store.T[v] for v in vids]
    Tm = max(Ts)
    S = torch.zeros(len(vids), Tm, dtype=torch.float64)
    G = torch.zeros(len(vids), dtype=torch.float64)
    fo = ctree.Forest(Ts, Tm)
    PHI = None                                   # revision 4 node potentials (README section 15)
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
    vts = [_vt(T) for T in Ts]
    askers = [policy.Asker(vt, am, cats) if make_asker is None else make_asker(b, vt)
              for b, vt in enumerate(vts)]
    A3 = torch.zeros(fo.N, 3, dtype=torch.float64, device=device)
    out = [{"scores": [], "eig": [], "voi": [], "asked": [], "p_G": []} for _ in vids]
    rem = [np.ones(len(a.q), dtype=bool) if allowed is None else np.isin(a.q, np.asarray(sorted(allowed[b])))
           for b, a in enumerate(askers)]
    llr = [np.zeros(T) for T in Ts]
    cnt = [np.zeros(T) for T in Ts]
    asked_o = [[] for _ in vids]                 # per video: (node, outcome index or None) of the asked nodes
    p0 = None
    for step in range(max_calls + 1):
        pG, m, p = ctree.marginals(fo, S, G, A3, chain, PHI)
        pG, m, p = pG.cpu().numpy(), m.cpu().numpy(), p.cpu().numpy()
        if p0 is None:
            p0 = [p[b, :T].copy() for b, T in enumerate(Ts)]
        for b, T in enumerate(Ts):
            if step <= len(askers[b].q):
                out[b]["scores"].append(p[b, :T].copy() if fusion == "tree" else _flat(p0[b], llr[b], cnt[b]))
                out[b]["p_G"].append(float(pG[b]))
        if step == max_calls:
            break
        chosen = []
        for b, (asker, off) in enumerate(zip(askers, fo.offs)):
            cand = np.where(rem[b])[0]
            if len(cand) == 0:
                continue
            nodes = asker.q[cand]
            mm = m[off + nodes]
            w = np.stack([np.full(len(cand), 1.0 - pG[b]), pG[b] * (1.0 - mm), pG[b] * mm], axis=1)
            if order == "eig" and copy_pi is not None and asked_o[b]:
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
            elif order == "eig":
                e = asker.eig(cand, w)
                j = int(np.argmax(e))
            else:
                e = asker.eig(cand[:1], w[:1])
                j = 0
            chosen.append((b, int(cand[j]), int(nodes[j]), w[j], float(e[j])))
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
            if no_nested:
                qa, qb = tr["a"][askers[b].q], tr["b"][askers[b].q]
                a_, b_ = tr["a"][node], tr["b"][node]
                rem[b] &= ~(((a_ <= qa) & (qb <= b_)) | ((qa <= a_) & (b_ <= qb)))
            o_idx = None
            if answer_ll is None:
                o = answers[vids[b]].get((int(tr["a"][node]), int(tr["b"][node])))
                ll = None if o is None else askers[b].loglik(node, o)
                if o is not None:
                    o_idx = int(qtree.answer_index(np.asarray(o)[askers[b].cats]))
                    if copy_pi is not None:
                        src = _nested_source(tr, node, asked_o[b], None)
                        if src is not None:              # pi bucketed by the shorter of the two (the child)
                            pi = float(copy_pi(float(min(tr["b"][node] - tr["a"][node],
                                                         tr["b"][src[0]] - tr["a"][src[0]]))))
                            if pi > 0.0:
                                ll = np.logaddexp(np.log1p(-pi) + ll,
                                                  np.log(pi) + (0.0 if o_idx == src[1] else -np.inf))
            else:
                ll = answer_ll(b, node)
            asked_o[b].append((node, o_idx))
            if ll is not None:
                A3[fo.offs[b] + node] = torch.as_tensor(ll, dtype=torch.float64, device=device)
                a_, b_ = int(tr["a"][node]), int(tr["b"][node])
                llr[b][a_:b_] += ll[2] - ll[0]
                cnt[b][a_:b_] += 1
            out[b]["eig"].append(e)
            out[b]["asked"].append(node)
    return {v: r for v, r in zip(vids, out)}
