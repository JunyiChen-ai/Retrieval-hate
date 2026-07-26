#!/usr/bin/env python3
"""CLAP lane — K-CLAP-3 comparator / confound PRE-CHECK.

Run BEFORE any CLAP feature existed (extraction job 13647 was still in its disk_guard preamble),
using ONLY already-banked caches (the killed Whisper-large-v3 block, the deployed LoRA-curric Z)
and the gt transcripts. **No CLAP data is touched here and no bar can be moved by it** — the
K-CLAP-* bars were frozen and committed at 6c8929d before this ran.

Purpose (two things, both of which fix the comparator before the treatment is measured):
  1. Validate the K-CLAP-3 AUC helpers (oof_scores / auc_boot) on real data.
  2. Pin the head-to-head comparator (Whisper stratum AUC) and the trivial-covariate baselines,
     so the CLAP number is interpretable the moment it lands.

What it found (see refine-logs/CLAP_STRATUM_PRECHECK_OUT.json) is load-bearing for reading the
gate: within the FN1 <=25-word stratum the ALREADY-KILLED Whisper block scores AUC ~0.85 and the
deployed Z alone ~0.89, i.e. the spec's absolute C1 thresholds (kill <=0.60 / go >=0.65) are SLACK
— 0.65 is below even the n_words-alone covariate baseline. The bars are NOT changed (that would be
a relaxation after the fact); the record documents that the operative NARROW-GO conditions are
therefore C2 (beat Whisper by >=+0.05) and C3 (conditional dAUC over Z > 0 with CI-low > 0), both
of which remain strict.

Zero test-touch (train U val only). CPU, ~30 s.
"""
import importlib.util
import json
import time

import numpy as np
import torch
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

REPO = '/data/jehc223/RGCL'
OUT = f'{REPO}/refine-logs/CLAP_STRATUM_PRECHECK_OUT.json'


def main():
    t0 = time.time()
    spec = importlib.util.spec_from_file_location('g', f'{REPO}/scripts/analysis/clap_g0cond_gate.py')
    g = importlib.util.module_from_spec(spec); spec.loader.exec_module(g)

    wh = torch.load(f'{REPO}/data/audio/HateMM/whisper_whisper-large-v3_trainval.pt',
                    map_location='cpu', weights_only=False)
    ids = list(wh['ids'][0])
    y_all = wh['labels'].numpy().astype(int)
    W_all = wh['emb'].numpy().astype(np.float64)
    words = g.load_words()
    nw = np.array([words[i] for i in ids], dtype=float)
    spans = json.load(open(f'{REPO}/data/gt/HateMM/hate_spans.json'))
    dur = np.array([float(spans[i]['duration']) for i in ids])
    dep = g._pooled_id2row(g.DEPLOY)
    Z_all = np.stack([dep[i][0] for i in ids])

    m = nw <= g.STRATUM_WORDS
    y, W, Z, nws, ds = y_all[m], W_all[m], Z_all[m], nw[m], dur[m]

    Cw, _ = g.pick_C(W, y); sw = g.oof_scores(W, y, Cw)
    auc_w, ci_w, nb = g.auc_boot(sw, y, g.BOOT_SEED + 5002)
    Cz, _ = g.pick_C(Z, y); sz = g.oof_scores(Z, y, Cz)
    auc_z, ci_z, _ = g.auc_boot(sz, y, g.BOOT_SEED + 5004)

    r_wn = spearmanr(sw, nws); r_wd = spearmanr(sw, ds)
    r_yn = spearmanr(y.astype(float), nws); r_yd = spearmanr(y.astype(float), ds)

    out = {
        'purpose': 'K-CLAP-3 comparator + confound pre-check. NO CLAP data touched. Moves NO bar.',
        'bars_frozen_at_commit': '6c8929d (spec 6c8929d / gate script 687ea82)',
        'scope': {'N_trainval': len(ids), 'n_pos': int(y_all.sum()),
                  'stratum_rule': f'n_words <= {g.STRATUM_WORDS} (FN1 rule)',
                  'stratum_n': int(m.sum()), 'stratum_pos': int(y.sum()),
                  'stratum_base_rate': float(y.mean())},
        'comparators': {
            'whisper_alone_stratum_auc': auc_w, 'whisper_ci': ci_w, 'whisper_C': Cw,
            'Z_deployed_alone_stratum_auc': auc_z, 'Z_ci': ci_z, 'Z_C': Cz,
            'n_boot_valid': nb},
        'trivial_covariate_baselines_stratum': {
            'n_words_alone_auc': float(roc_auc_score(y, nws)),
            'duration_alone_auc': float(roc_auc_score(y, ds))},
        'confound_correlations_stratum': {
            'rho_whisperscore_nwords': [float(r_wn.statistic), float(r_wn.pvalue)],
            'rho_whisperscore_duration': [float(r_wd.statistic), float(r_wd.pvalue)],
            'rho_label_nwords': [float(r_yn.statistic), float(r_yn.pvalue)],
            'rho_label_duration': [float(r_yd.statistic), float(r_yd.pvalue)]},
        'implied_narrow_go_thresholds': {
            'C1_frozen_auc_ge': g.AUC_GO, 'C1_frozen_cilow_gt': g.AUC_GO_CILOW,
            'C1_is_slack_because': ('the already-KILLED Whisper block scores %.4f and even the '
                                    'n_words-alone covariate scores %.4f, both above the frozen '
                                    '0.65 go-threshold' % (auc_w, float(roc_auc_score(y, nws)))),
            'C2_effective_clap_auc_required': float(auc_w + g.WHISPER_MARGIN),
            'C3_baseline_Z_auc_to_beat': auc_z,
            'ruling': ('bars NOT changed (changing them after measuring the comparator would be a '
                       'relaxation); the operative NARROW-GO conditions are C2 and C3, both strict.')},
        'elapsed_s': round(time.time() - t0, 1)}
    json.dump(out, open(OUT, 'w'), indent=1)

    print(f"stratum n={int(m.sum())} pos={int(y.sum())} base={y.mean():.4f}")
    print(f"  Whisper-alone (KILLED block)  AUC = {auc_w:.4f} CI[{ci_w[0]:.4f},{ci_w[1]:.4f}] C={Cw}")
    print(f"  Z_deployed-alone              AUC = {auc_z:.4f} CI[{ci_z[0]:.4f},{ci_z[1]:.4f}] C={Cz}")
    print(f"  n_words alone                 AUC = {roc_auc_score(y, nws):.4f}")
    print(f"  duration alone                AUC = {roc_auc_score(y, ds):.4f}")
    print(f"  rho(whisper_score, n_words)   = {r_wn.statistic:+.4f} (p={r_wn.pvalue:.3g})")
    print(f"  rho(whisper_score, duration)  = {r_wd.statistic:+.4f} (p={r_wd.pvalue:.3g})")
    print(f"  => C2 requires CLAP stratum AUC >= {auc_w + g.WHISPER_MARGIN:.4f}")
    print(f"  => C3 must add over Z at {auc_z:.4f}")
    print(f"wrote {OUT}")


if __name__ == '__main__':
    main()
