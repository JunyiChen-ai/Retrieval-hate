#!/usr/bin/env python
"""P2 forensic pass 2 — tie-break-free re-reads of H1/H2/H3/H4. Analysis only."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

exec(open("/home/jehc223/Retrieval-hate/idea-stage/p2_forensic.py").read().split("def main()")[0])

CACHE = ROOT / "idea-stage/p2_forensic_cache.npz"


def build():
    vids, idx, S, W_img, W_txt, y, folds, spans = load()
    V = len(vids)
    Mg, haveg = interval_seg_mask(
        vids, lambda v: (float(spans[v]["duration"]), spans[v].get("spans") or []) if v in spans else None)
    Sn = l2(S, axis=2)
    P = np.full((V, K), np.nan)
    for tr, qu in folds:
        mem = Sn[tr].reshape(-1, Sn.shape[2])
        mem_lab = np.repeat(y[tr], K)
        for i in qu:
            sim = Sn[i] @ mem.T
            nb = np.argpartition(-sim, 20, axis=1)[:, :20]
            P[i] = mem_lab[nb].mean(axis=1)
        log("  fold done")
    np.savez(CACHE, P=P, Mg=Mg, haveg=haveg, y=y, vids=np.array(vids))
    return vids, idx, S, W_img, W_txt, y, folds, spans, Mg, haveg, P


def main():
    out = {}
    vids, idx, S, W_img, W_txt, y, folds, spans, Mg, haveg, P = build()
    V = len(vids)
    ev = np.where((y == 1) & haveg & Mg.any(axis=1))[0]
    cov = Mg[ev].sum(axis=1) / K
    pmax = P.max(axis=1)
    tie_n = (P == pmax[:, None]).sum(axis=1)
    jfirst = P.argmax(axis=1)
    rng = np.random.default_rng(SEED)

    def rand_tie(ii, R=400):
        h = []
        for _ in range(R):
            j = np.array([rng.choice(np.flatnonzero(P[i] == pmax[i])) for i in ii])
            h.append(Mg[ii, j].mean())
        return float(np.mean(h)), float(np.std(h))

    # ---------- counterfactual selector audit -------------------------------
    unif = np.array([Mg[ev, rng.integers(0, K, len(ev))].mean() for _ in range(400)])
    rt_m, rt_s = rand_tie(ev)
    uniq = ev[tie_n[ev] == 1]
    nonsat = ev[pmax[ev] < 1.0 - 1e-9]
    out["selector_audit"] = {
        "chance_mean_coverage": float(cov.mean()),
        "uniform_random_segment": {"mean": float(unif.mean()), "sd": float(unif.std())},
        "frozen_np.argmax_first_index": float(Mg[ev, jfirst[ev]].mean()),
        "random_tiebreak": {"mean": rt_m, "sd": rt_s},
        "last_index_tiebreak": float(Mg[ev, np.array([np.flatnonzero(P[i] == pmax[i])[-1] for i in ev])].mean()),
        "subset_unique_argmax": {
            "n": int(len(uniq)),
            "hit": float(Mg[uniq, jfirst[uniq]].mean()),
            "chance": float((Mg[uniq].sum(axis=1) / K).mean()),
            "frac_selected_at_k0": float((jfirst[uniq] == 0).mean())},
        "subset_pmax_below_1": {
            "n": int(len(nonsat)),
            "hit_frozen": float(Mg[nonsat, jfirst[nonsat]].mean()),
            "hit_random_tiebreak": rand_tie(nonsat, 200)[0],
            "chance": float((Mg[nonsat].sum(axis=1) / K).mean()),
            "mean_tie_multiplicity": float(tie_n[nonsat].mean())},
        "distinct_p_values_per_video_hateful": {
            "mean": float(np.mean([len(np.unique(P[i])) for i in ev])),
            "median": float(np.median([len(np.unique(P[i])) for i in ev]))},
    }
    log("selector_audit", json.dumps(out["selector_audit"], indent=1))

    # ---------- H1 extra: positional profile of p vs gold --------------------
    out["H1_positional"] = {
        "mean_p_by_position_hateful": [float(v) for v in P[ev].mean(axis=0)],
        "gold_membership_by_position": [float(v) for v in Mg[ev].mean(axis=0)],
        "corr_meanP_vs_goldmembership_over_positions": float(
            np.corrcoef(P[ev].mean(axis=0), Mg[ev].mean(axis=0))[0, 1]),
        "mean_p_by_position_nonhateful": [float(v) for v in P[y == 0].mean(axis=0)],
    }
    log("H1_positional", json.dumps(out["H1_positional"], indent=1))

    # ---------- H2 tie-break-free, with CI on the stratum difference ---------
    fin = gate_c_final()
    rows = [(i, set(fin[vids[i]].get("required_modalities") or [])) for i in ev if vids[i] in fin]
    ii = np.array([r[0] for r in rows])
    auc = np.array([roc_auc_score(Mg[i].astype(int), P[i]) if Mg[i].sum() not in (0, K) else np.nan
                    for i in ii])
    # tie-break-free hit: expected hit under random tie-break, computed exactly
    exp_hit = np.array([Mg[i][P[i] == pmax[i]].mean() for i in ii])
    covi = Mg[ii].sum(axis=1) / K
    need_txt = np.array(["on_screen_text" in r[1] for r in rows])
    need_sp = np.array([bool({"transcript", "audio"} & r[1]) for r in rows])

    def diff_ci(mask, vals):
        a, b = vals[mask], vals[~mask]
        a, b = a[~np.isnan(a)], b[~np.isnan(b)]
        r2 = np.random.default_rng(SEED)
        d = np.array([a[r2.integers(0, len(a), len(a))].mean() - b[r2.integers(0, len(b), len(b))].mean()
                      for _ in range(BOOT)])
        return float(a.mean()), float(b.mean()), float(a.mean() - b.mean()), \
            [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))]

    lift = exp_hit - covi
    h2 = {"n": len(rows), "n_on_screen_text_required": int(need_txt.sum()),
          "n_speech_required": int(need_sp.sum())}
    for nm, msk in [("on_screen_text_required", need_txt), ("speech_required", need_sp)]:
        a, b, d, ci = diff_ci(msk, auc)
        h2[nm + "__within_video_AUROC"] = {"group": a, "complement": b, "diff": d, "diff_ci95": ci}
        a, b, d, ci = diff_ci(msk, lift)
        h2[nm + "__tiebreakfree_hit_minus_chance"] = {"group": a, "complement": b, "diff": d, "diff_ci95": ci}
    out["H2_tiebreak_free"] = h2
    log("H2", json.dumps(h2, indent=1))

    # ---------- H4 extra: class-prior baseline for purity --------------------
    Wn = l2(W_img)
    prior = float(y.mean())
    base_purity = prior * prior + (1 - prior) * (1 - prior)
    pur_w, pur_s_mean = [], []
    for tr, qu in folds:
        p_tr = y[tr].mean()
        for i in qu:
            sim = Wn[i] @ Wn[tr].T
            nb = tr[np.argpartition(-sim, 20)[:20]]
            pur_w.append((y[nb] == y[i]).mean())
            pv = P[i] if y[i] == 1 else 1 - P[i]
            pur_s_mean.append(pv.mean())
    out["H4_purity_baseline"] = {
        "expected_purity_under_random_neighbours": base_purity,
        "whole_video_top20_purity": float(np.mean(pur_w)),
        "segment_mean_top20_purity": float(np.mean(pur_s_mean)),
        "lift_whole": float(np.mean(pur_w) - base_purity),
        "lift_segment_mean": float(np.mean(pur_s_mean) - base_purity),
    }
    log("H4", json.dumps(out["H4_purity_baseline"], indent=1))

    o = ROOT / "idea-stage/p2_forensic2.json"
    json.dump(out, open(o, "w"), indent=1)
    log("wrote", o)


if __name__ == "__main__":
    main()
