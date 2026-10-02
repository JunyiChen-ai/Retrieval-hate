"""Existing negative-training-video estimate for nested-answer copying."""
import numpy as np
from . import cpolicy, policy, qtree

BUCKETS = [(4, 8), (8, 16), (16, 32), (32, 64), (64, 128), (128, 1e9)]

def bucket_of(length):
    for i, (lo, hi) in enumerate(BUCKETS):
        if lo <= length < hi:
            return i
    return len(BUCKETS) - 1


def estimate_pi_neg(answers, labels, train_ids, T, am, cats):
    """Label-free copy probability per child-length bucket from parent-child pairs of negative training videos."""
    r_num, r_den, q_sum = np.zeros(len(BUCKETS)), np.zeros(len(BUCKETS)), np.zeros(len(BUCKETS))
    for v in train_ids:
        if labels[v] != 0 or v not in answers:
            continue
        vt = cpolicy._vt(T[v])
        asker = policy.Asker(vt, am, cats)
        tr = vt.tr
        for n in asker.q:
            par = int(tr["parent"][n]) if "parent" in tr else -1
            if par < 0:
                continue
            oc = answers[v].get((int(tr["a"][n]), int(tr["b"][n])))
            op = answers[v].get((int(tr["a"][par]), int(tr["b"][par])))
            if oc is None or op is None:
                continue
            k = bucket_of(tr["b"][n] - tr["a"][n])
            ic = qtree.answer_index(np.asarray(oc)[cats]); ip = qtree.answer_index(np.asarray(op)[cats])
            r_num[k] += ic == ip
            r_den[k] += 1
            q_sum[k] += float(np.sum(asker.po_s[asker.qpos[int(n)], 0] ** 2))
    r = r_num / np.maximum(r_den, 1)
    q = q_sum / np.maximum(r_den, 1)
    pi = np.clip((r - q) / np.maximum(1 - q, 1e-9), 0.0, 0.95)
    pi[r_den == 0] = 0.0
    return pi, r, q, r_den
