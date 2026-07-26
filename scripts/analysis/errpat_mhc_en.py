#!/usr/bin/env python
"""ERRPAT MHC-EN — forensic per-item error-pattern analysis of the deployed EN method.

READ-ONLY DIAGNOSTICS. CPU only. No training, no tuning, nothing written back into
state/. Test-set is read for forensics only; no threshold/hyper-parameter is carried
back into any deployed config (every oracle number below is explicitly labelled as an
upper bound, not a deployable operator).

Two prediction arms:
  ARM-V (BANKED, EXACT)  = deployed EN best-stack, val-selected protocol, 4 seeds.
      source scripts/analysis/p2_out/cache_MHC_s{0,1,2,3}.json (produced by
      p2_rerank_eval.py --mode collect; its header carries a bit-identical repro gate
      against the logged val-selected floor of exp-archive-knn-seeds).
      config: frozen Qwen2.5-VL-7B encoder -> RGCL align head -> archive-kNN a=0.25
              augmented key -> top-20 rank-weighted signed-cosine vote over train bank.
  ARM-F (RECOMPUTED, machinery-validated) = frozen-encoder no-key floor, FINAL EPOCH
      (e29), 3 seeds, both encoders (Qwen + CLIP). Heads reloaded from
      refine-logs/router_ckpt_snapshot/MHC_{CLIP,Qwen}_s{0,1,2}_e29.pt, features from
      the banked test cache. Validated by asserting the recomputed test acc/macroF1
      equals the primary trainlog Test_Retrieval e29 line to 4 dp.

Outputs a single JSON: scripts/analysis/errpat_mhc_en_out.json
"""
import os, sys, json, math
from collections import Counter, defaultdict

import numpy as np
import torch
import faiss

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(REPO, "src"))
sys.path.insert(0, os.path.join(REPO, "scripts", "analysis"))
faiss.omp_set_num_threads(4)
torch.set_num_threads(4)

import cross_channel_router_gate as R  # noqa: E402  (build_head/embed/knn_channel/raw_modality_vote/TOPK)

TOPK = 20
CACHE = os.path.join(REPO, "data", "CLIP_Embedding", "MHC")
CKPT_DIR = os.path.join(REPO, "refine-logs", "router_ckpt_snapshot")
P2 = os.path.join(REPO, "scripts", "analysis", "p2_out")
ANN = "/data/jehc223/Multihateclip/English/annotation(new).json"
OUT = os.path.join(REPO, "scripts", "analysis", "errpat_mhc_en_out.json")

# primary-log anchors (Test_Retrieval / Val_Retrieval epoch 29), re-read at report time
ANCHOR_TEST = {
    ("Qwen", 0): (0.8012, 0.7596),   # slurm/logs/enc3s_MHC_Qwen2.5-VL-7B-Instruct_HF_seed0_12850.trainlog:272
    ("Qwen", 1): (0.7702, 0.7203),   # slurm/logs/arcbase_MHC_Qwen2.5-VL-7B-Instruct_HF_seed1_12275.trainlog:273
    ("Qwen", 2): (0.7826, 0.7475),   # slurm/logs/arcbase_MHC_Qwen2.5-VL-7B-Instruct_HF_seed2_12276.trainlog:274
    ("CLIP", 0): (0.7640, 0.7145),   # enc3s_MHC_openai_clip-...-336_HF_seed0_12850.trainlog
    ("CLIP", 1): (0.7826, 0.7159),
    ("CLIP", 2): (0.7888, 0.7303),
}

# consensus-denoising id lists (train-side memory-bank entries)
HUMAN2 = ["XScP1AiMkNM", "QvPp8Q7QhWE"]          # scripts/analysis/memory_editing_demo.py:76
RULE14 = ["YNf2tZgh4WM", "TRFp4a4lD0o", "My5PVJLP6Bg", "QvPp8Q7QhWE", "8Pim0TnLQDQ",
          "2ytDPK74q28", "aeOm9oT0_qk", "hKwgFaE7fbQ", "6hFEc1MLZC0", "lNCfDw80YSQ",
          "dcrX2-oto8Y", "EU-dip0ITa4", "XScP1AiMkNM", "Z2Cs5Oqm9iU"]


def macro_f1(y, p):
    y = np.asarray(y); p = np.asarray(p)
    fs = []
    for c in (0, 1):
        tp = int(((p == c) & (y == c)).sum()); fp = int(((p == c) & (y != c)).sum())
        fn = int(((p != c) & (y == c)).sum())
        pr = tp / (tp + fp) if tp + fp else 0.0
        rc = tp / (tp + fn) if tp + fn else 0.0
        fs.append(2 * pr * rc / (pr + rc) if pr + rc else 0.0)
    return float(np.mean(fs))


def load_cache(split, enc):
    m = {"CLIP": "openai_clip-vit-large-patch14-336_HF", "Qwen": "Qwen2.5-VL-7B-Instruct_HF"}[enc]
    d = torch.load(os.path.join(CACHE, f"{split}_{m}.pt"), map_location="cpu")
    ids = d["ids"]
    ids = ids[0] if (isinstance(ids, list) and len(ids) == 1 and isinstance(ids[0], list)) else ids
    return list(ids), d["img_feats"].float(), d["text_feats"].float(), d["labels"].long().numpy()


# ---------------------------------------------------------------- ARM-V (banked) ---
def arm_v():
    """Per-item deployed val-selected predictions, 4 seeds, straight off the banked dump."""
    per_seed = {}
    for s in range(4):
        d = json.load(open(os.path.join(P2, f"cache_MHC_s{s}.json")))
        rows = {}
        for smp in d["samples"]:
            nb = smp["neighbors"][:TOPK]
            nlab = np.array([x[2] for x in nb], dtype="float64")
            nsim = np.array([x[1] for x in nb], dtype="float64")
            rows[smp["id"]] = dict(
                y=int(smp["label"]), vote=float(smp["floor_vote"]), pred=int(smp["floor_pred"]),
                margin=float(abs(smp["floor_vote"])),
                nb_ids=[x[0] for x in nb], nb_lab=nlab.astype(int).tolist(),
                nb_sim=nsim.tolist(),
                purity_pos=float(nlab.mean()),
                nb60_ids=[x[0] for x in smp["neighbors"]],
                nb60_lab=[int(x[2]) for x in smp["neighbors"]],
            )
        y = np.array([rows[i]["y"] for i in rows]); p = np.array([rows[i]["pred"] for i in rows])
        per_seed[s] = dict(rows=rows, acc=float((y == p).mean()), mf1=macro_f1(y, p),
                           ckpt=d["ckpt"], logged=d["logged"], floor=d["floor"])
        # repro gate: our recompute of acc/mf1 from the dumped per-item preds == dumped header
        assert abs(per_seed[s]["acc"] - d["floor"]["acc"]) < 1e-12, s
        assert abs(per_seed[s]["mf1"] - d["floor"]["macro_f1"]) < 1e-12, s
        assert round(per_seed[s]["acc"], 4) == d["logged"][0], (s, per_seed[s]["acc"], d["logged"])
    return per_seed


# ------------------------------------------------------------- ARM-F (recomputed) --
def arm_f():
    """Final-epoch e29 no-key floor per-item test predictions + raw single-stream votes."""
    out = {}
    tr_cache, te_cache = {}, {}
    for enc in ("Qwen", "CLIP"):
        tr_cache[enc] = load_cache("train", enc)
        te_cache[enc] = load_cache("test_seen", enc)
    for enc in ("Qwen", "CLIP"):
        tr_ids, tr_img, tr_txt, tr_lab = tr_cache[enc]
        te_ids, te_img, te_txt, te_lab = te_cache[enc]
        # raw single-modality votes (no head) — identical construction to router gate
        vimg = R.raw_modality_vote(tr_img, tr_lab, te_img, False)
        vtxt = R.raw_modality_vote(tr_txt, tr_lab, te_txt, False)
        for s in (0, 1, 2):
            sd = torch.load(os.path.join(CKPT_DIR, f"MHC_{enc}_s{s}_e29.pt"), map_location="cpu")
            m = R.build_head(sd)
            tr_e = R.embed(m, tr_img, tr_txt)
            te_e = R.embed(m, te_img, te_txt)
            st = R.knn_channel(tr_e, tr_lab, te_e, exclude_self=False)
            # full neighbour detail for the fused channel
            trn = tr_e.copy(); qn = te_e.copy()
            faiss.normalize_L2(trn); faiss.normalize_L2(qn)
            ix = faiss.IndexFlatIP(trn.shape[1]); ix.add(trn)
            D, I = ix.search(qn, TOPK)
            rows = {}
            for i, vid in enumerate(te_ids):
                v = st[i]["vote"]
                rows[vid] = dict(
                    y=int(te_lab[i]), vote=float(v), pred=int(v >= 0), margin=float(abs(v)),
                    purity_pos=float(st[i]["phate"]), topsim=st[i]["topsim"],
                    meansim=st[i]["meansim"],
                    nb_ids=[tr_ids[j] for j in I[i]], nb_lab=[int(tr_lab[j]) for j in I[i]],
                    nb_sim=[float(x) for x in D[i]],
                    v_img=float(vimg[i]), v_txt=float(vtxt[i]),
                    pred_img=int(vimg[i] >= 0), pred_txt=int(vtxt[i] >= 0),
                )
            y = np.array([rows[v]["y"] for v in te_ids]); p = np.array([rows[v]["pred"] for v in te_ids])
            acc, mf1 = float((y == p).mean()), macro_f1(y, p)
            a_acc, a_mf1 = ANCHOR_TEST[(enc, s)]
            assert round(acc, 4) == a_acc, (enc, s, acc, a_acc)
            assert round(mf1, 4) == a_mf1, (enc, s, mf1, a_mf1)
            out[f"{enc}_s{s}"] = dict(rows=rows, acc=acc, mf1=mf1,
                                      anchor=[a_acc, a_mf1], validated=True)
        # single-stream arm accuracies (seed-independent: raw features, no head)
        y = te_lab
        out[f"{enc}_streams"] = dict(
            img_acc=float(((vimg >= 0).astype(int) == y).mean()),
            img_mf1=macro_f1(y, (vimg >= 0).astype(int)),
            txt_acc=float(((vtxt >= 0).astype(int) == y).mean()),
            txt_mf1=macro_f1(y, (vtxt >= 0).astype(int)),
        )
    return out, te_cache["Qwen"][0], te_cache["Qwen"][3]


# ------------------------------------------------------------------- covariates ----
def covariates(test_ids):
    ann = {e["Video_ID"]: e for e in json.load(open(ANN))}
    cov = {}
    for vid in test_ids:
        e = ann.get(vid, {})
        tr = (e.get("Transcript") or "").strip()
        ti = (e.get("Title") or "").strip()
        cov[vid] = dict(label3=e.get("Label", "MISSING"),
                        title=ti, transcript=tr,
                        n_tr_char=len(tr), n_tr_word=len(tr.split()),
                        n_ti_char=len(ti), n_ti_word=len(ti.split()),
                        empty_transcript=int(len(tr) == 0))
    return cov


# ------------------------------------------------------ oracle / reachability ------
def global_threshold_oracle(votes, y):
    """Label-oracle best SINGLE GLOBAL threshold on the deployed vote (upper bound of the
    only law-III-legal symmetric operator on the vote scale). Forensic; not deployable."""
    v = np.asarray(votes, dtype="float64"); y = np.asarray(y)
    cand = np.unique(np.concatenate([[-1e9, 0.0, 1e9], v]))
    best = (-1, None)
    for t in cand:
        a = float(((v >= t).astype(int) == y).mean())
        if a > best[0]:
            best = (a, float(t))
    return best


def main():
    res = dict(meta=dict(
        purpose="forensic per-item error-pattern analysis, MHC-EN (n_test=161)",
        cpu_only=True, gpu_jobs=0, training=0,
        arm_v="banked exact (p2_out/cache_MHC_s0..3.json), deployed best-stack val-selected, 4 seeds",
        arm_f="recomputed from router_ckpt_snapshot e29 heads + banked test caches, validated 4dp vs trainlogs",
    ))

    V = arm_v()
    F, test_ids, test_y = arm_f()
    cov = covariates(test_ids)

    res["arm_v_seeds"] = {str(s): dict(acc=round(V[s]["acc"], 6), mf1=round(V[s]["mf1"], 6),
                                       ckpt=V[s]["ckpt"], logged=V[s]["logged"]) for s in V}
    res["arm_v_mean"] = dict(acc=float(np.mean([V[s]["acc"] for s in V])),
                             acc_std=float(np.std([V[s]["acc"] for s in V], ddof=1)),
                             mf1=float(np.mean([V[s]["mf1"] for s in V])),
                             mf1_std=float(np.std([V[s]["mf1"] for s in V], ddof=1)))
    res["arm_f_seeds"] = {k: dict(acc=round(F[k]["acc"], 6), mf1=round(F[k]["mf1"], 6),
                                  anchor=F[k]["anchor"]) for k in F if k.endswith(("_s0", "_s1", "_s2"))}
    for enc in ("Qwen", "CLIP"):
        ks = [f"{enc}_s{s}" for s in (0, 1, 2)]
        res[f"arm_f_mean_{enc}"] = dict(
            acc=float(np.mean([F[k]["acc"] for k in ks])),
            acc_std=float(np.std([F[k]["acc"] for k in ks], ddof=1)),
            mf1=float(np.mean([F[k]["mf1"] for k in ks])),
            mf1_std=float(np.std([F[k]["mf1"] for k in ks], ddof=1)))
        res[f"streams_{enc}"] = F[f"{enc}_streams"]

    # ---------------- label / class composition of the test split ----------------
    lab3 = Counter(cov[v]["label3"] for v in test_ids)
    res["test_composition"] = dict(
        n=len(test_ids), n_pos=int(sum(test_y)), n_neg=int(len(test_y) - sum(test_y)),
        label3=dict(lab3),
        label3_by_binary={str(b): dict(Counter(cov[v]["label3"] for v, yy in zip(test_ids, test_y) if yy == b))
                          for b in (0, 1)},
        missing_ann=[v for v in test_ids if cov[v]["label3"] == "MISSING"],
    )

    # ---------------- per-item consensus across the 4 deployed val-sel seeds ------
    ids_v = list(V[0]["rows"].keys())
    assert set(ids_v) == set(test_ids), "id set mismatch between arms"
    per_item = {}
    for vid in test_ids:
        y = V[0]["rows"][vid]["y"]
        vs = [V[s]["rows"][vid] for s in range(4)]
        wrong_v = [int(r["pred"] != y) for r in vs]
        fs = [F[f"Qwen_s{s}"]["rows"][vid] for s in (0, 1, 2)]
        wrong_f = [int(r["pred"] != y) for r in fs]
        cs = [F[f"CLIP_s{s}"]["rows"][vid] for s in (0, 1, 2)]
        wrong_c = [int(r["pred"] != y) for r in cs]
        q0 = fs[0]
        per_item[vid] = dict(
            y=y, label3=cov[vid]["label3"],
            n_wrong_v=sum(wrong_v), n_wrong_f=sum(wrong_f), n_wrong_clip=sum(wrong_c),
            err_type=("FN" if y == 1 else "FP"),
            mean_margin_v=float(np.mean([r["margin"] for r in vs])),
            mean_vote_v=float(np.mean([r["vote"] for r in vs])),
            mean_purity_pos_v=float(np.mean([r["purity_pos"] for r in vs])),
            mean_vote_f=float(np.mean([r["vote"] for r in fs])),
            mean_purity_pos_f=float(np.mean([r["purity_pos"] for r in fs])),
            v_img=q0["v_img"], v_txt=q0["v_txt"],
            pred_img=q0["pred_img"], pred_txt=q0["pred_txt"],
            img_right=int(q0["pred_img"] == y), txt_right=int(q0["pred_txt"] == y),
            n_tr_char=cov[vid]["n_tr_char"], n_tr_word=cov[vid]["n_tr_word"],
            empty_transcript=cov[vid]["empty_transcript"],
            n_ti_word=cov[vid]["n_ti_word"],
            title=cov[vid]["title"][:160],
            transcript_head=cov[vid]["transcript"][:220],
            # neighbour-level: does top-20 contain any correct-class neighbour (selection ceiling)
            nb_correct_frac_v=float(np.mean([np.mean(np.array(r["nb_lab"]) == y) for r in vs])),
            nb_topsim_wrongclass_v=float(np.mean([
                max([sm for lb, sm in zip(r["nb_lab"], r["nb_sim"]) if lb != y], default=0.0) for r in vs])),
            noisy2_in_top20=int(any(any(n in HUMAN2 for n in r["nb_ids"]) for r in vs)),
            rule14_in_top20=int(any(any(n in RULE14 for n in r["nb_ids"]) for r in vs)),
            noisy2_in_top60=int(any(any(n in HUMAN2 for n in r["nb60_ids"]) for r in vs)),
        )
    res["per_item"] = per_item

    # ---------------- error inventory -------------------------------------------
    errs_v = {vid: d for vid, d in per_item.items() if d["n_wrong_v"] >= 1}
    consensus_v = {vid: d for vid, d in per_item.items() if d["n_wrong_v"] == 4}
    res["error_counts"] = dict(
        arm_v_per_seed={str(s): dict(
            n_err=int(sum(1 for vid in test_ids if V[s]["rows"][vid]["pred"] != V[s]["rows"][vid]["y"])),
            n_FP=int(sum(1 for vid in test_ids
                         if V[s]["rows"][vid]["y"] == 0 and V[s]["rows"][vid]["pred"] == 1)),
            n_FN=int(sum(1 for vid in test_ids
                         if V[s]["rows"][vid]["y"] == 1 and V[s]["rows"][vid]["pred"] == 0)),
        ) for s in range(4)},
        arm_f_qwen_per_seed={str(s): dict(
            n_err=int(sum(1 for vid in test_ids if F[f"Qwen_s{s}"]["rows"][vid]["pred"] != test_y[test_ids.index(vid)])),
        ) for s in (0, 1, 2)},
        arm_v_union_any_seed=len(errs_v),
        arm_v_consensus_all4=len(consensus_v),
        arm_v_seedflipped=len(errs_v) - len(consensus_v),
        arm_v_consensus_FP=int(sum(1 for d in consensus_v.values() if d["err_type"] == "FP")),
        arm_v_consensus_FN=int(sum(1 for d in consensus_v.values() if d["err_type"] == "FN")),
    )

    # error mass by 3-way label
    def by3(sel):
        return dict(Counter(d["label3"] for d in sel.values()))
    res["error_by_label3"] = dict(
        consensus_all4=by3(consensus_v),
        any_seed=by3(errs_v),
        consensus_FN=by3({k: v for k, v in consensus_v.items() if v["err_type"] == "FN"}),
        consensus_FP=by3({k: v for k, v in consensus_v.items() if v["err_type"] == "FP"}),
        # base rates for comparison
        test_base=dict(lab3),
        per_class_error_rate={
            L: dict(n=int(lab3[L]),
                    n_consensus_err=int(sum(1 for d in consensus_v.values() if d["label3"] == L)),
                    n_any_err=int(sum(1 for d in errs_v.values() if d["label3"] == L)),
                    rate_consensus=round(sum(1 for d in consensus_v.values() if d["label3"] == L) / lab3[L], 4))
            for L in lab3},
    )

    # ---------------- stream cross-tab (F86 tie-in) ------------------------------
    ct = Counter()
    for vid, d in per_item.items():
        fused_right = int(d["n_wrong_f"] <= 1)  # majority of 3 Qwen seeds correct
        ct[(d["img_right"], d["txt_right"], fused_right)] += 1
    res["stream_crosstab_all"] = {f"img{a}_txt{b}_fused{c}": n for (a, b, c), n in sorted(ct.items())}
    cte = Counter()
    for vid, d in consensus_v.items():
        cte[(d["img_right"], d["txt_right"])] += 1
    res["stream_crosstab_consensus_errors"] = {f"img{a}_txt{b}": n for (a, b), n in sorted(cte.items())}

    # ---------------- reachability: global-threshold oracle (forensic UPPER bound)
    thr = {}
    for s in range(4):
        v = [V[s]["rows"][vid]["vote"] for vid in test_ids]
        y = [V[s]["rows"][vid]["y"] for vid in test_ids]
        a, t = global_threshold_oracle(v, y)
        thr[str(s)] = dict(deployed_acc=V[s]["acc"], oracle_thr_acc=a, thr=t, d_acc=a - V[s]["acc"])
    res["global_threshold_oracle_arm_v"] = thr
    res["global_threshold_oracle_arm_v_mean_d"] = float(np.mean([thr[k]["d_acc"] for k in thr]))

    # per-item: fixable by the seed's own oracle global threshold?
    for s in range(4):
        t = thr[str(s)]["thr"]
        for vid in test_ids:
            r = V[s]["rows"][vid]
            per_item[vid].setdefault("thr_fixed", []).append(
                int((r["pred"] != r["y"]) and (int(r["vote"] >= t) == r["y"])))
    for vid in test_ids:
        per_item[vid]["thr_fixed_n"] = int(sum(per_item[vid].pop("thr_fixed")))

    # ---------------- vote-locked vs neighbour-reachable ------------------------
    lock = dict(vote_locked=0, neighbour_reachable=0, thr_reachable=0)
    for vid, d in consensus_v.items():
        if d["nb_correct_frac_v"] < 0.10:
            lock["vote_locked"] += 1
        else:
            lock["neighbour_reachable"] += 1
        if d["thr_fixed_n"] >= 2:
            lock["thr_reachable"] += 1
    res["consensus_error_reachability"] = lock

    # ---------------- val-sel vs final-epoch flips ------------------------------
    flips = dict(v_wrong_f_right=0, v_right_f_wrong=0, both_wrong=0, both_right=0)
    for vid, d in per_item.items():
        vw = d["n_wrong_v"] >= 3   # majority of 4 val-sel seeds wrong
        fw = d["n_wrong_f"] >= 2   # majority of 3 final-ep seeds wrong
        flips["both_wrong" if (vw and fw) else
              "v_wrong_f_right" if vw else
              "v_right_f_wrong" if fw else "both_right"] += 1
    res["protocol_flips"] = flips

    # ---------------- encoder flips (Qwen vs CLIP, final-epoch) ------------------
    ef = dict(qwen_wrong_clip_right=0, clip_wrong_qwen_right=0, both_wrong=0, both_right=0)
    for vid, d in per_item.items():
        qw = d["n_wrong_f"] >= 2; cw = d["n_wrong_clip"] >= 2
        ef["both_wrong" if (qw and cw) else
           "qwen_wrong_clip_right" if qw else
           "clip_wrong_qwen_right" if cw else "both_right"] += 1
    res["encoder_flips_finalep"] = ef

    # ---------------- transcript-length forensics -------------------------------
    def stat(sel, key):
        a = [d[key] for d in sel.values()]
        return dict(n=len(a), mean=float(np.mean(a)), median=float(np.median(a)),
                    p10=float(np.percentile(a, 10)), p90=float(np.percentile(a, 90)))
    correct = {vid: d for vid, d in per_item.items() if d["n_wrong_v"] == 0}
    res["transcript_forensics"] = dict(
        consensus_err=stat(consensus_v, "n_tr_word"),
        always_correct=stat(correct, "n_tr_word"),
        all_items=stat(per_item, "n_tr_word"),
        empty_transcript_total=int(sum(d["empty_transcript"] for d in per_item.values())),
        empty_transcript_in_consensus_err=int(sum(d["empty_transcript"] for d in consensus_v.values())),
    )

    # ---------------- noisy-memory contamination --------------------------------
    res["noisy_memory_overlap"] = dict(
        human2_ids=HUMAN2,
        n_items_with_human2_in_top20=int(sum(d["noisy2_in_top20"] for d in per_item.values())),
        n_items_with_human2_in_top60=int(sum(d["noisy2_in_top60"] for d in per_item.values())),
        n_consensus_err_with_human2_top20=int(sum(d["noisy2_in_top20"] for d in consensus_v.values())),
        n_rule14_in_top20_all=int(sum(d["rule14_in_top20"] for d in per_item.values())),
        n_rule14_in_top20_consensus_err=int(sum(d["rule14_in_top20"] for d in consensus_v.values())),
    )

    res["consensus_error_ids"] = sorted(consensus_v.keys())
    res["any_error_ids"] = sorted(errs_v.keys())
    json.dump(res, open(OUT, "w"), indent=1, default=float)
    print("wrote", OUT)
    print(json.dumps({k: v for k, v in res.items() if k != "per_item"}, indent=1, default=float)[:9000])


if __name__ == "__main__":
    main()
