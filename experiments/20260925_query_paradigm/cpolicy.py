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


def run_batch(model, store, vids, am, chain, answers, cats, max_calls, device, order="eig", fusion="tree",
              record_voi=False):
    Ts = [store.T[v] for v in vids]
    Tm = max(Ts)
    S = torch.zeros(len(vids), Tm, dtype=torch.float64)
    G = torch.zeros(len(vids), dtype=torch.float64)
    for b, v in enumerate(vids):
        s, g = policy.video_prior(model, store, v, device)
        S[b, :len(s)] = torch.from_numpy(s)
        G[b] = g
    S, G = S.to(device), G.to(device)
    fo = ctree.Forest(Ts, Tm)
    vts = [_vt(T) for T in Ts]
    askers = [policy.Asker(vt, am, cats) for vt in vts]
    A3 = torch.zeros(fo.N, 3, dtype=torch.float64, device=device)
    out = [{"scores": [], "eig": [], "voi": [], "asked": [], "p_G": []} for _ in vids]
    rem = [np.ones(len(a.q), dtype=bool) for a in askers]
    llr = [np.zeros(T) for T in Ts]
    cnt = [np.zeros(T) for T in Ts]
    p0 = None
    for step in range(max_calls + 1):
        pG, m, p = ctree.marginals(fo, S, G, A3, chain)
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
            if order == "eig":
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
                _, _, pc = ctree.marginals(fo, S, G, Ac, chain)
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
            o = answers[vids[b]].get((int(tr["a"][node]), int(tr["b"][node])))
            if o is not None:
                ll = askers[b].loglik(node, o)
                A3[fo.offs[b] + node] = torch.as_tensor(ll, dtype=torch.float64, device=device)
                a_, b_ = int(tr["a"][node]), int(tr["b"][node])
                llr[b][a_:b_] += ll[2] - ll[0]
                cnt[b][a_:b_] += 1
            out[b]["eig"].append(e)
            out[b]["asked"].append(node)
    return {v: r for v, r in zip(vids, out)}
