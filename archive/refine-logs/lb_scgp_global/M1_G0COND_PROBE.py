#!/usr/bin/env python3
"""G0-cond conditional-information probe for lb_scgp_global M1 sealed cache.

Fail-closed conditional codelength/accuracy probe:
  g(Z) vs g'([Z, A])   with 4 A-arms:
    (1) real-A full-set        A = encoded certificate (9 observables one-hot + conf); const for uncovered
    (2) real-A covered-only    same, restricted to covered (parse-ok/non-constant-consensus) videos
    (3) oracle-A @ measured c  A = gold label revealed ONLY on covered videos (3-way one-hot)
    (4) oracle-A @ coverage 1  A = gold label for ALL videos (2-way one-hot)  [prices v3 repair]

Z = z-scored concat(img_feats, text_feats) = raw frozen features feeding the trainable
    projection (DPI => most generous to A => a KILL here is maximally fail-closed).
Probe = L2 logistic regression (RGCL final linear classifier proxy); same C for g and g'.
CV = RepeatedStratifiedKFold(5x5). Codelength in bits = sum -log2 p(y_true) on held-out.
"""
import json, sys
import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold

RNG = np.random.default_rng(20260713)
EPS = 1e-12
N_SPLITS, N_REPEATS = 5, 5
C_GRID = [0.001, 0.01, 0.1, 1.0]
B_BOOT = 5000

TRI_STATES = ["contradicted", "supported", "unresolved"]
MOD_STATES = ["multi_modal", "single_modal", "text_audio", "unresolved", "visual_audio", "visual_text"]
TRI_OBS = ["context_shift_observable","counter_context_observable","cross_modal_binding_observable",
           "dehumanizing_or_threat_surface_observable","harmful_surface_observable",
           "source_alignment_observable","text_audio_reference_observable","visual_reference_observable"]
MOD_OBS = "modality_binding_observable"


def load_features(ds, enc):
    o = torch.load(f'data/CLIP_Embedding/{ds}/train_{enc}.pt', map_location='cpu', weights_only=False)
    ids = o['ids'][0]
    Z = np.concatenate([o['img_feats'].numpy(), o['text_feats'].numpy()], axis=1).astype(np.float64)
    y = o['labels'].numpy().astype(int)
    return ids, Z, y


def load_cert_A(ds, ids):
    """Encode certificate A per video from cache.jsonl, aggregating R=4 replicas.
    State = consensus (mode); confidence = mean over 4 replicas. One-hot(state)+conf per observable."""
    # gather replicas per video
    rows = {}
    with open(f'artifacts/lb_scgp_global/v1/m1/cache/{ds}/cache.jsonl') as f:
        for line in f:
            r = json.loads(line)
            rows.setdefault(r['video_id'], []).append(r)
    # consensus states from manifest
    m = json.load(open(f'artifacts/lb_scgp_global/v1/m1/cache/{ds}/cache_manifest.json'))
    cons = {c['video_id']: c['consensus'] for c in m['consensus']}

    A = []
    covered = []
    for vid in ids:
        reps = rows[vid]
        cvec = cons[vid]
        feat = []
        # tri-state observables
        for ob in TRI_OBS:
            st = cvec[ob]  # consensus state (0 means unresolved-coded? check) -- consensus uses ints/strings
            # consensus stores tri as int in {-1,0,1}? inspect: earlier we saw 0/'unresolved'. Map:
            st_str = tri_int_to_str(st)
            oh = [1.0 if st_str == s else 0.0 for s in TRI_STATES]
            conf = np.mean([rep['observables'][ob]['confidence'] for rep in reps])
            feat.extend(oh + [conf])
        # modality
        stm = cvec[MOD_OBS]
        oh = [1.0 if stm == s else 0.0 for s in MOD_STATES]
        confm = np.mean([rep['observables'][MOD_OBS]['confidence'] for rep in reps])
        feat.extend(oh + [confm])
        A.append(feat)
        # covered = consensus not all-unresolved/0
        is_cov = any((cvec[k] != 0 and cvec[k] != 'unresolved') for k in cvec)
        covered.append(is_cov)
    return np.array(A, dtype=np.float64), np.array(covered, dtype=bool)


def tri_int_to_str(v):
    # consensus tri stored as int: 1->supported, -1->contradicted, 0->unresolved; or already string
    if isinstance(v, str):
        return v
    if v == 1:
        return "supported"
    if v == -1:
        return "contradicted"
    return "unresolved"


def pick_C(Z, y):
    best_c, best_acc = C_GRID[0], -1
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=0)
    for c in C_GRID:
        accs = []
        for tr, te in skf.split(Z, y):
            pipe = Pipeline([('sc', StandardScaler()),
                             ('lr', LogisticRegression(C=c, max_iter=2000))])
            pipe.fit(Z[tr], y[tr])
            accs.append((pipe.predict(Z[te]) == y[te]).mean())
        if np.mean(accs) > best_acc:
            best_acc, best_c = np.mean(accs), c
    return best_c, best_acc


def cv_eval(X, y, C, seeds=range(N_REPEATS)):
    """Return per-video mean held-out NLL (bits) and mean held-out correct (over repeats)."""
    n = len(y)
    nll = np.zeros(n); cor = np.zeros(n); cnt = np.zeros(n)
    for rep in seeds:
        skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=1000 + rep)
        for tr, te in skf.split(X, y):
            pipe = Pipeline([('sc', StandardScaler()),
                             ('lr', LogisticRegression(C=C, max_iter=2000))])
            pipe.fit(X[tr], y[tr])
            p = pipe.predict_proba(X[te])[:, 1]
            p = np.clip(p, EPS, 1 - EPS)
            pt = np.where(y[te] == 1, p, 1 - p)
            nll[te] += -np.log2(pt)
            cor[te] += (p >= 0.5).astype(int) == y[te]
            cnt[te] += 1
    return nll / cnt, cor / cnt  # per-video mean bits, per-video mean accuracy


def boot_ci(delta, B=B_BOOT):
    n = len(delta)
    means = np.array([delta[RNG.integers(0, n, n)].mean() for _ in range(B)])
    return delta.mean(), np.percentile(means, 2.5), np.percentile(means, 97.5)


def run_arm(name, Z, A, y, C, mask=None):
    """Compare g(Z) vs g([Z,A]) under CV; optional mask restricts to a subset (covered-only)."""
    if mask is not None:
        Zx, Ax, yx = Z[mask], A[mask], y[mask]
    else:
        Zx, Ax, yx = Z, A, y
    XZ = Zx
    XZA = np.concatenate([Zx, Ax], axis=1)
    nll_z, acc_z = cv_eval(XZ, yx, C)
    nll_za, acc_za = cv_eval(XZA, yx, C)
    dnll = nll_z - nll_za            # bits saved per video by A (positive = A helps)
    dacc = acc_za - acc_z            # accuracy gain per video by A
    m_bits, lo_bits, hi_bits = boot_ci(dnll)
    m_acc, lo_acc, hi_acc = boot_ci(dacc)
    return {
        'arm': name, 'n': len(yx),
        'acc_Z': float(acc_z.mean()), 'acc_ZA': float(acc_za.mean()),
        'dacc_mean': float(m_acc), 'dacc_ci': [float(lo_acc), float(hi_acc)],
        'bits_Z_total': float(nll_z.sum()), 'bits_ZA_total': float(nll_za.sum()),
        'dbits_per_video_mean': float(m_bits), 'dbits_ci': [float(lo_bits), float(hi_bits)],
        'dbits_total': float(dnll.sum()),
    }


def main():
    out = {}
    for ds in ['MHC', 'MHC_zh']:
        ids, Zc, y = load_features(ds, 'openai_clip-vit-large-patch14-336_HF')
        idsq, Zq, yq = load_features(ds, 'Qwen2.5-VL-7B-Instruct_HF')
        assert ids == idsq and (y == yq).all()
        A_real, covered = load_cert_A(ds, ids)
        n = len(y); c = covered.mean()
        # oracle A @ measured coverage: 3-way one-hot {uncovered, cov_neg, cov_pos}
        A_or_c = np.zeros((n, 3))
        for i in range(n):
            if not covered[i]:
                A_or_c[i, 0] = 1.0
            elif y[i] == 0:
                A_or_c[i, 1] = 1.0
            else:
                A_or_c[i, 2] = 1.0
        # oracle A @ coverage 1.0: 2-way one-hot(gold)
        A_or_1 = np.zeros((n, 2))
        A_or_1[np.arange(n), y] = 1.0

        out[ds] = {'n': n, 'n_covered': int(covered.sum()), 'coverage': float(c),
                   'base_rate': float(y.mean()), 'A_real_dim': A_real.shape[1], 'encoders': {}}
        for enc, Z in [('CLIP', Zc), ('Qwen', Zq)]:
            C, cacc = pick_C(Z, y)
            # a_cov,Z : Z-only per-video CV accuracy restricted to covered
            _, acc_z_full = cv_eval(Z, y, C)
            a_cov = float(acc_z_full[covered].mean())
            a_all = float(acc_z_full.mean())
            arms = {}
            arms['real_full'] = run_arm('real-A full-set', Z, A_real, y, C)
            arms['real_covered'] = run_arm('real-A covered-only', Z, A_real, y, C, mask=covered)
            arms['oracle_measured'] = run_arm('oracle-A @ measured coverage', Z, A_or_c, y, C)
            arms['oracle_full'] = run_arm('oracle-A @ coverage 1.0', Z, A_or_1, y, C)
            out[ds]['encoders'][enc] = {
                'Z_dim': int(Z.shape[1]), 'C': C, 'C_cv_acc': float(cacc),
                'a_all_Zonly': a_all, 'a_covered_Zonly': a_cov, 'arms': arms}
            print(f'[{ds}/{enc}] C={C} Zdim={Z.shape[1]} a_all={a_all:.4f} a_cov={a_cov:.4f}', flush=True)
            for k, r in arms.items():
                print(f'   {r["arm"]:32s} n={r["n"]:3d} accZ={r["acc_Z"]:.4f} accZA={r["acc_ZA"]:.4f} '
                      f'dacc={r["dacc_mean"]:+.4f} CI[{r["dacc_ci"][0]:+.4f},{r["dacc_ci"][1]:+.4f}] '
                      f'dbits/vid={r["dbits_per_video_mean"]:+.4f} CI[{r["dbits_ci"][0]:+.4f},{r["dbits_ci"][1]:+.4f}] '
                      f'dbits_tot={r["dbits_total"]:+.1f}', flush=True)
    json.dump(out, open(sys.argv[1] if len(sys.argv) > 1 else 'g0cond_out.json', 'w'), indent=1)


if __name__ == '__main__':
    main()
