#!/usr/bin/env python
"""P2 forensic dissection — analysis only, no method development.

Reproduces the frozen P2 selector, then instruments it. HateMM-train only.
Writes idea-stage/p2_forensic.json.
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

ROOT = Path("/home/jehc223/Retrieval-hate")
RUN = ROOT / "artifacts/tera_gate0/tera-gate0-20260807T000625Z-7ba80eaf"
SEG = ROOT / "data/CLIP_Embedding/HateMM/train_subclipK30_openai_clip-vit-large-patch14-336_HF.pt"
WHOLE = ROOT / "data/CLIP_Embedding/HateMM/train_openai_clip-vit-large-patch14-336_HF.pt"
SPANS = ROOT / "data/gt/HateMM/hate_spans.json"
GATEC = ROOT / "logging/runs/gate_c_annotation"
K = 30
BOOT = 2000
SEED = 20260809  # forensic seed, distinct from the pilot seed


def log(*a):
    print(*a, flush=True)


def l2(x, axis=-1):
    n = np.linalg.norm(x, axis=axis, keepdims=True)
    return x / np.maximum(n, 1e-8)


def load():
    seg = torch.load(SEG, map_location="cpu")
    who = torch.load(WHOLE, map_location="cpu")
    vids = list(seg["video_ids"])
    idx = {v: i for i, v in enumerate(vids)}
    S = seg["subclip_img_feats"].numpy().astype(np.float64).reshape(len(vids), K, -1)
    W_img = who["img_feats"].numpy().astype(np.float64)
    W_txt = who["text_feats"].numpy().astype(np.float64)
    y = who["labels"].numpy().astype(int)
    folds = []
    for f in range(5):
        tr = json.load(open(RUN / f"folds/fold_{f}/train_ids.json"))
        qu = json.load(open(RUN / f"folds/fold_{f}/query_ids.json"))
        folds.append((np.array([idx[v] for v in tr]), np.array([idx[v] for v in qu])))
    spans = json.load(open(SPANS))
    return vids, idx, S, W_img, W_txt, y, folds, spans


def interval_seg_mask(vids, get_intervals):
    """[V,K] bool from a per-video (duration, [[a,b],...]) provider; same discretisation as pilots.py."""
    V = len(vids)
    M = np.zeros((V, K), dtype=bool)
    have = np.zeros(V, dtype=bool)
    for i, v in enumerate(vids):
        e = get_intervals(v)
        if e is None:
            continue
        D, sp = e
        if D is None or D <= 0 or not sp:
            continue
        have[i] = True
        for a, b in sp:
            k0 = int(np.floor(max(0.0, a) / D * K))
            k1 = int(np.ceil(min(D, b) / D * K))
            M[i, max(0, k0):min(K, max(k1, k0 + 1))] = True
    return M, have


def gate_c_final():
    """Adjudicated-final census rows: c1 then overwritten by adj (same rule as pilots.py)."""
    final = {}
    for fn in ["claude_c1_rows.jsonl", "claude_adj_rows.jsonl"]:
        p = GATEC / fn
        if not p.exists():
            continue
        for line in open(p):
            r = json.loads(line)
            final[r["video_id"]] = r
    return final


def boot_mean_ci(x, rng, n=BOOT):
    x = np.asarray(x, dtype=float)
    if len(x) == 0:
        return (float("nan"), float("nan"))
    b = np.array([x[rng.integers(0, len(x), len(x))].mean() for _ in range(n)])
    return float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))


def main():
    out = {}
    vids, idx, S, W_img, W_txt, y, folds, spans = load()
    V = len(vids)
    log(f"loaded V={V} pos={int(y.sum())}")

    Mg, haveg = interval_seg_mask(
        vids, lambda v: (float(spans[v]["duration"]), spans[v].get("spans") or []) if v in spans else None)

    # ---- reproduce the frozen selector, keeping the full p_j matrix -------------
    Sn = l2(S, axis=2)
    P = np.full((V, K), np.nan)          # neighbourhood hateful fraction per segment
    TOPSIM = np.full((V, K), np.nan)     # similarity of the single nearest memory segment
    NBSIM20 = np.full((V, K), np.nan)    # mean similarity of the top-20
    NB_INGOLD = np.full((V, K), np.nan)  # frac of *hateful-parent* neighbours that sit in their own gold span
    NB_HATEPAR = np.full((V, K), np.nan)
    MEANSIM = np.full((V, K), np.nan)    # mean sim to whole memory (hubness proxy)
    for fi, (tr, qu) in enumerate(folds):
        mem = Sn[tr].reshape(-1, Sn.shape[2])
        mem_lab = np.repeat(y[tr], K)
        mem_gold = Mg[tr].reshape(-1)
        mem_have = np.repeat(haveg[tr], K)
        for i in qu:
            sim = Sn[i] @ mem.T
            nb = np.argpartition(-sim, 20, axis=1)[:, :20]
            P[i] = mem_lab[nb].mean(axis=1)
            NBSIM20[i] = np.take_along_axis(sim, nb, axis=1).mean(axis=1)
            TOPSIM[i] = sim.max(axis=1)
            MEANSIM[i] = sim.mean(axis=1)
            lab = mem_lab[nb]
            gold = mem_gold[nb] & mem_have[nb]
            hp = lab == 1
            NB_HATEPAR[i] = hp.mean(axis=1)
            with np.errstate(invalid="ignore"):
                NB_INGOLD[i] = np.where(hp.sum(axis=1) > 0,
                                        (gold & hp).sum(axis=1) / np.maximum(hp.sum(axis=1), 1),
                                        np.nan)
        log(f"  fold {fi} done")
    assert not np.isnan(P).any()

    jstar_first = P.argmax(axis=1)                       # frozen behaviour: np.argmax -> lowest index
    pmax = P.max(axis=1)
    tie_n = (P == pmax[:, None]).sum(axis=1)

    ev = np.where((y == 1) & haveg & Mg.any(axis=1))[0]
    cov = Mg[ev].sum(axis=1) / K
    rng = np.random.default_rng(SEED)

    hit_first = Mg[ev, jstar_first[ev]].astype(float)
    out["repro"] = {
        "n_eval": int(len(ev)),
        "hit_rate_frozen_argmax": float(hit_first.mean()),
        "chance_rate": float(cov.mean()),
        "matches_pilot_json": bool(abs(float(hit_first.mean()) - 0.5436241610738255) < 1e-9),
    }
    log("repro", out["repro"])

    # =========================================================== H5: tie-breaking / position
    # random tie-break
    hits_rt = []
    for _ in range(200):
        jr = np.array([rng.choice(np.flatnonzero(P[i] == pmax[i])) for i in ev])
        hits_rt.append(Mg[ev, jr].mean())
    # last-index tie-break
    jlast = np.array([np.flatnonzero(P[i] == pmax[i])[-1] for i in ev])
    # positional prior of gold membership
    gold_by_k = Mg[ev].mean(axis=0)
    pos_counts = Counter(jstar_first[ev].tolist())
    out["H5_tiebreak_position"] = {
        "tie_multiplicity": {"mean": float(tie_n[ev].mean()), "median": float(np.median(tie_n[ev])),
                             "frac_unique_argmax": float((tie_n[ev] == 1).mean()),
                             "frac_ties_ge_5": float((tie_n[ev] >= 5).mean()),
                             "max": int(tie_n[ev].max())},
        "pmax_dist": {"mean": float(pmax[ev].mean()), "median": float(np.median(pmax[ev])),
                      "frac_eq_1.0": float((pmax[ev] >= 1.0 - 1e-9).mean())},
        "hit_random_tiebreak_mean": float(np.mean(hits_rt)),
        "hit_random_tiebreak_sd": float(np.std(hits_rt)),
        "hit_last_index_tiebreak": float(Mg[ev, jlast].mean()),
        "jstar_first_position_hist": {str(k): int(pos_counts.get(k, 0)) for k in range(K)},
        "jstar_frac_k0": float((jstar_first[ev] == 0).mean()),
        "jstar_frac_k_le2": float((jstar_first[ev] <= 2).mean()),
        "jstar_frac_k_ge27": float((jstar_first[ev] >= 27).mean()),
        "gold_membership_by_position": [float(v) for v in gold_by_k],
        "gold_membership_k0": float(gold_by_k[0]),
        "gold_membership_k29": float(gold_by_k[-1]),
        "gold_membership_middle_k10_19": float(gold_by_k[10:20].mean()),
    }
    log("H5", json.dumps(out["H5_tiebreak_position"], indent=1)[:800])

    # =========================================================== H1: benign-segment dominance
    # Does p_j carry ANY within-video signal about gold membership?
    aucs, in_p, out_p = [], [], []
    for i in ev:
        m = Mg[i]
        if m.all() or (~m).any() is False:
            continue
        if m.sum() == 0 or m.sum() == K:
            continue
        aucs.append(roc_auc_score(m.astype(int), P[i]))
        in_p.append(P[i][m].mean())
        out_p.append(P[i][~m].mean())
    aucs = np.array(aucs)
    lo, hi = boot_mean_ci(aucs, np.random.default_rng(SEED))
    dpm = np.array(in_p) - np.array(out_p)
    dlo, dhi = boot_mean_ci(dpm, np.random.default_rng(SEED))
    # neighbour composition of the selected segment
    sel_nb_ingold = NB_INGOLD[ev, jstar_first[ev]]
    base_nb_ingold = np.nanmean(NB_INGOLD[ev], axis=1)
    # base rate of "hateful memory segment lies in its own gold span"
    mem_gold_all = (Mg & haveg[:, None])[y == 1].mean()
    out["H1_benign_dominance"] = {
        "n_videos_with_mixed_gold_mask": int(len(aucs)),
        "within_video_AUROC_p_predicts_gold_mean": float(aucs.mean()),
        "within_video_AUROC_ci95": [lo, hi],
        "mean_p_inside_gold": float(np.mean(in_p)),
        "mean_p_outside_gold": float(np.mean(out_p)),
        "delta_p_in_minus_out": float(dpm.mean()),
        "delta_p_ci95": [dlo, dhi],
        "selected_seg_frac_hateful_parent_neighbours": float(NB_HATEPAR[ev, jstar_first[ev]].mean()),
        "selected_seg_frac_of_hateful_neighbours_that_are_in_their_own_gold_span": float(np.nanmean(sel_nb_ingold)),
        "all_segments_same_quantity": float(np.nanmean(base_nb_ingold)),
        "base_rate_hateful_memory_segment_in_own_gold_span": float(mem_gold_all),
    }
    log("H1", json.dumps(out["H1_benign_dominance"], indent=1))

    # =========================================================== H5b: near-duplicate / boilerplate
    sel = (np.arange(V), jstar_first)
    topsim_sel = TOPSIM[ev, jstar_first[ev]]
    topsim_all = TOPSIM[ev].mean(axis=1)
    ms_sel = MEANSIM[ev, jstar_first[ev]]
    ms_all = MEANSIM[ev].mean(axis=1)
    # rank of selected segment's topsim within its own video
    rank_topsim = np.array([(TOPSIM[i] < TOPSIM[i, jstar_first[i]]).mean() for i in ev])
    # near-duplicate flag
    nd = topsim_sel > 0.95
    out["H5b_near_duplicate"] = {
        "selected_top1_neighbour_sim_mean": float(topsim_sel.mean()),
        "video_mean_top1_neighbour_sim": float(topsim_all.mean()),
        "selected_within_video_percentile_of_top1_sim": float(rank_topsim.mean()),
        "frac_selected_with_top1_sim_gt_0.95": float(nd.mean()),
        "frac_all_segments_top1_sim_gt_0.95": float((TOPSIM[ev] > 0.95).mean()),
        "hit_rate_when_selected_is_near_dup": float(hit_first[nd].mean()) if nd.any() else None,
        "hit_rate_when_not_near_dup": float(hit_first[~nd].mean()) if (~nd).any() else None,
        "selected_mean_sim_to_memory": float(ms_sel.mean()),
        "video_mean_sim_to_memory": float(ms_all.mean()),
    }
    log("H5b", json.dumps(out["H5b_near_duplicate"], indent=1))

    # =========================================================== H3: span-noise / metric artifact
    fin = gate_c_final()
    msi = {}
    for v, r in fin.items():
        iv = r.get("minimal_sufficient_intervals") or []
        if v in spans and iv:
            msi[v] = (float(spans[v]["duration"]), [[float(a), float(b)] for a, b in iv])
    Mm, havem = interval_seg_mask(vids, lambda v: msi.get(v))
    evm = np.array([i for i in ev if havem[i] and Mm[i].any()])
    res_h3 = {"n_videos_with_coder_intervals_and_hateful_with_gold": int(len(evm))}
    if len(evm):
        hitm = Mm[evm, jstar_first[evm]].astype(float)
        covm = Mm[evm].sum(axis=1) / K
        rr = np.random.default_rng(SEED)
        # coverage-matched random-selector control on the same videos
        rand_hits = np.array([Mm[evm, rr.integers(0, K, len(evm))].mean() for _ in range(500)])
        lo3, hi3 = boot_mean_ci(hitm - covm, np.random.default_rng(SEED))
        aucm = []
        for i in evm:
            m = Mm[i]
            if m.sum() in (0, K):
                continue
            aucm.append(roc_auc_score(m.astype(int), P[i]))
        alo, ahi = boot_mean_ci(aucm, np.random.default_rng(SEED))
        # same-video comparison against the official spans
        hitg_sub = Mg[evm, jstar_first[evm]].astype(float)
        covg_sub = Mg[evm].sum(axis=1) / K
        aucg_sub = []
        for i in evm:
            m = Mg[i]
            if m.sum() in (0, K):
                continue
            aucg_sub.append(roc_auc_score(m.astype(int), P[i]))
        res_h3.update({
            "coder_minimal_hit": float(hitm.mean()),
            "coder_minimal_chance": float(covm.mean()),
            "coder_minimal_ratio": float(hitm.mean() / covm.mean()),
            "coder_minimal_random_selector_control": float(rand_hits.mean()),
            "paired_hit_minus_chance_ci95": [lo3, hi3],
            "within_video_AUROC_vs_coder_minimal_mean": float(np.mean(aucm)) if aucm else None,
            "within_video_AUROC_vs_coder_minimal_ci95": [alo, ahi],
            "same_videos_official_hit": float(hitg_sub.mean()),
            "same_videos_official_chance": float(covg_sub.mean()),
            "same_videos_official_AUROC": float(np.mean(aucg_sub)) if aucg_sub else None,
        })
    out["H3_span_noise"] = res_h3
    log("H3", json.dumps(res_h3, indent=1))

    # =========================================================== H2: modality gap
    rows = []
    for i in ev:
        v = vids[i]
        r = fin.get(v)
        if r is None:
            continue
        req = set(r.get("required_modalities") or [])
        rows.append((i, req))
    res_h2 = {"n_eval_videos_in_census": len(rows)}
    if rows:
        ii = np.array([r[0] for r in rows])
        need_txt = np.array(["on_screen_text" in r[1] for r in rows])
        need_speech = np.array([bool({"transcript", "audio"} & r[1]) for r in rows])
        vis_only = np.array([r[1] == {"visual"} for r in rows])
        h = Mg[ii, jstar_first[ii]].astype(float)
        c = Mg[ii].sum(axis=1) / K
        aucv = []
        for i in ii:
            m = Mg[i]
            aucv.append(roc_auc_score(m.astype(int), P[i]) if m.sum() not in (0, K) else np.nan)
        aucv = np.array(aucv, dtype=float)

        def grp(mask, name):
            if mask.sum() == 0:
                return {"name": name, "n": 0}
            return {"name": name, "n": int(mask.sum()),
                    "hit": float(h[mask].mean()), "chance": float(c[mask].mean()),
                    "hit_minus_chance": float((h[mask] - c[mask]).mean()),
                    "within_video_AUROC": float(np.nanmean(aucv[mask]))}
        res_h2["groups"] = [
            grp(vis_only, "visual_only_required"),
            grp(~need_txt, "on_screen_text_NOT_required"),
            grp(need_txt, "on_screen_text_required"),
            grp(~need_speech, "speech_NOT_required"),
            grp(need_speech, "speech_required"),
        ]
    out["H2_modality_gap"] = res_h2
    log("H2", json.dumps(res_h2, indent=1))

    # =========================================================== H4: pooling dilution, reversed
    Wn = l2(W_img)
    def sep(X, lab):
        """mean within-class cosine minus mean between-class cosine, on L2-normalised rows."""
        G = X @ X.T
        n = len(lab)
        same = lab[:, None] == lab[None, :]
        eye = np.eye(n, dtype=bool)
        w = G[same & ~eye].mean()
        b = G[~same].mean()
        return float(w), float(b), float(w - b)
    w1, b1, d1 = sep(Wn, y)
    # segment level: subsample to keep the Gram matrix cheap
    rr = np.random.default_rng(SEED)
    pick_v = rr.choice(V, size=V, replace=False)
    seg_rows = np.stack([Sn[i, rr.integers(0, K)] for i in pick_v])
    w2, b2, d2 = sep(seg_rows, y[pick_v])
    # per-video segment mean of the normalised segments (renormalised) — pooling of the same units
    pooln = l2(Sn.mean(axis=1))
    w3, b3, d3 = sep(pooln, y)
    # retrieval-purity comparison: whole-video kNN vs segment kNN aggregated
    pur_whole, pur_seg_mean, pur_seg_max = [], [], []
    for tr, qu in folds:
        for i in qu:
            sim = Wn[i] @ Wn[tr].T
            nb = tr[np.argpartition(-sim, 20)[:20]]
            pur_whole.append((y[nb] == y[i]).mean())
            pv = P[i]  # hateful fraction per segment
            phat = pv if y[i] == 1 else 1 - pv
            pur_seg_mean.append(phat.mean())
            pur_seg_max.append(phat.max())
    out["H4_pooling"] = {
        "whole_video_visual_within_cos": w1, "between_cos": b1, "separation": d1,
        "single_segment_within_cos": w2, "single_segment_between_cos": b2, "single_segment_separation": d2,
        "segment_mean_pooled_within_cos": w3, "segment_mean_pooled_between_cos": b3,
        "segment_mean_pooled_separation": d3,
        "top20_label_purity_whole_video_retrieval": float(np.mean(pur_whole)),
        "mean_over_segments_of_top20_purity": float(np.mean(pur_seg_mean)),
        "max_over_segments_of_top20_purity": float(np.mean(pur_seg_max)),
    }
    log("H4", json.dumps(out["H4_pooling"], indent=1))

    # =========================================================== selector sanity: does p even track the label?
    pmean = P.mean(axis=1)
    out["selector_sanity"] = {
        "AUROC_video_label_from_mean_p": float(roc_auc_score(y, pmean)),
        "AUROC_video_label_from_max_p": float(roc_auc_score(y, P.max(axis=1))),
        "mean_p_hateful_videos": float(pmean[y == 1].mean()),
        "mean_p_nonhateful_videos": float(pmean[y == 0].mean()),
        "memory_hateful_segment_fraction": float(y.mean()),
        "pmax_hateful": float(P.max(axis=1)[y == 1].mean()),
        "pmax_nonhateful": float(P.max(axis=1)[y == 0].mean()),
    }
    log("sanity", json.dumps(out["selector_sanity"], indent=1))

    o = ROOT / "idea-stage/p2_forensic.json"
    json.dump(out, open(o, "w"), indent=1)
    log("wrote", o)


if __name__ == "__main__":
    main()
