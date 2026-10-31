"""Inference-time answer model fitted on validation span labels (no gradient to the network; user ruling 2026-09-29).

The anchored training model only knows the states the video label fixes, so it over-trusts short-node answers
(experiments/20260925_query_paradigm/README.md, section 16). Here every answered queryable node of the validation
videos gets its state from the validation GT: 0 = negative video, 1 = positive video and no GT harm in the node,
2 = GT harm in the node. Per state, a multinomial logistic regression of the answer level on the standardised log
node length (qtree.fit_weighted, one Laplace pseudo-count per level). The result is a three-state AnswerModel with a
length term, usable wherever the anchored model is (policy.Asker for EIG/EER and the answer likelihood).
"""
import numpy as np

from . import qtree


def fit_validation_answer_model(answers, labels, gt, val_ids, T, len_mu, len_sd, categories):
    rows = {0: ([], []), 1: ([], []), 2: ([], [])}
    for v in val_ids:
        tr = qtree.tree(T[v])
        y = np.asarray(gt[v])
        for n in np.where(tr["queryable"])[0]:
            a, b = int(tr["a"][n]), int(tr["b"][n])
            o = answers[v].get((a, b))
            if o is None:
                continue
            s = 0 if labels[v] == 0 else (2 if y[a:b].any() else 1)
            rows[s][0].append(np.asarray(o, dtype=np.int64))
            rows[s][1].append((np.log(b - a) - len_mu) / len_sd)
    theta = np.zeros((qtree.N_CAT, 3, qtree.N_LEV))
    omega = np.zeros((qtree.N_CAT, 3, qtree.N_LEV))
    counts = []
    for s in range(3):
        O, U = np.stack(rows[s][0]), np.asarray(rows[s][1])
        theta[:, s], omega[:, s] = qtree.fit_weighted(O, U, np.ones(len(U)), length_term=True)
        counts.append(len(U))
    am = qtree.AnswerModel(theta, len_mu, len_sd, length_term=True, n_state=3, categories=categories, omega0=omega)
    am.requires_grad_(False)
    return am, counts
