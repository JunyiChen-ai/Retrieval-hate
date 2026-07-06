#!/usr/bin/env python
"""P2 — margin-gated MLLM reranking of retrieved neighbors (CPU).

Pre-registration: research-wiki/EXP_p2_neighbor_rerank.md. Two modes:

  collect : per (ds, seed) load the val-selected archive-kNN head, project the
            splits, build the v1-keyed augmented memory/query keys, retrieve
            top-60, reproduce the floor top-20 vote (bit-identical repro gate),
            pick the 25%-deferral gate threshold on VAL, and cache every test
            sample's top-60 (nid, sim, nlabel) + gate flag + floor pred. Emits
            the deduplicated union of (query_id, neighbor_id) pairs over all
            seeds' GATED queries -> the MLLM comparability judge.

  revote  : read the cache + the MLLM verdicts and apply the four conditions
            (A floor / B MLLM rerank / C random-drop / D oracle) with the
            frozen revote rule, writing per-seed + mean±std metrics (overall and
            on the gated subset), paired deltas B-A / B-C, B's fraction of D,
            drop/extension/fallback counts, verdict distribution, and a
            kept/dropped-neighbor audit.

All decision code (vote) replicates src/utils/metrics.compute_metrics_retrieval
(use_sim=True, arithmetic) bit-for-bit. No src/ file is modified. CPU only.
"""
import argparse
import json
import os
import random
import sys
from collections import Counter

import numpy as np
import torch
import faiss
from sklearn.metrics import f1_score

ROOT = "/data/jehc223/RGCL"
sys.path.insert(0, os.path.join(ROOT, "src"))

from eval_cross_dataset import project_split, build_head          # noqa: E402
from data_loader.dataset import (                                 # noqa: E402
    load_feats_from_CLIP, load_archive_feats_split, resolve_archive_path)
from utils.metrics import sigmoid                                 # noqa: E402
from easydict import EasyDict                                     # noqa: E402

TOPK = 20            # the vote's K
N_JUDGE = 60         # 3*K reranking depth (extension cap)
ALPHA = 0.25
DEFER_RATE = 0.25    # fixed a priori
MIN_SURVIVE = 3      # extension target
RANDOM_SEED = 0      # condition C

MODEL = {"MHC": "Qwen2.5-VL-7B-Instruct_HF",
         "MHC_zh": "Qwen2.5-VL-7B-Instruct-LoRA_HF"}
SEEDS = {"MHC": [0, 1, 2, 3], "MHC_zh": [0, 1, 2, 3, 4]}
CKPT_FILE = {
    "MHC": {0: "best_model_24_0.7875.pt", 1: "best_model_29_0.7875.pt",
            2: "best_model_21_0.8125.pt", 3: "best_model_27_0.7875.pt"},
    "MHC_zh": {0: "best_model_18_0.8717948717948718.pt",
               1: "best_model_23_0.8846153846153846.pt",
               2: "best_model_14_0.8846153846153846.pt",
               3: "best_model_17_0.8717948717948718.pt",
               4: "best_model_12_0.8717948717948718.pt"},
}
# logged val-selected floor (acc/macroF1), exp-archive-knn-seeds.md, repro gate.
LOGGED_A = {
    "MHC": {0: (0.8075, 0.7626), 1: (0.7640, 0.7145),
            2: (0.7950, 0.7505), 3: (0.8075, 0.7713)},
    "MHC_zh": {0: (0.8523, 0.8270), 1: (0.8456, 0.8158), 2: (0.8322, 0.8046),
               3: (0.8188, 0.7837), 4: (0.7852, 0.7266)},
}


def ckpt_path(ds, seed):
    group = "RAC_video_archive" if seed == 0 else "RAC_video_archive_seeds"
    run = ("RAC_lr0.0001_Bz64_Ep30_cosSim_triplet_drop[0.2, 0.4, 0.1]_topK20_"
           "_PseudoGold_positive_1_hard_negative_1_seed{s}_hybrid_loss_{m}_arc-knn-a0.25"
           ).format(s=seed, m=MODEL[ds])
    return os.path.join(ROOT, "logging/Retrieval", ds, group, run, "ckpt",
                        CKPT_FILE[ds][seed])


# ----- vote (bit-identical to compute_metrics_retrieval use_sim arithmetic) --- #
def sim_vote(labels, sims, topk=TOPK):
    """Similarity-signed, rank-weighted arithmetic vote over the given neighbor
    list (retrieval order, top first). Weights topk..1 by position; uses at most
    topk neighbors. pred = sigmoid(vote) >= 0.5."""
    labels = np.asarray(labels[:topk], dtype=np.float64)
    sims = np.asarray(sims[:topk], dtype=np.float64)
    L = len(labels)
    if L == 0:
        return 0.0
    weight = np.arange(1, topk + 1)[::-1]           # [topk, ..., 1]
    lab_map = (labels * 2.0 - 1.0) * sims
    return float(np.sum(lab_map * weight[:L]) / np.sum(weight[:L]))


def vote_pred(vote):
    return int(sigmoid(np.array([vote]))[0] >= 0.5)


def augment(fused, arc, alpha=ALPHA):
    fused_n = torch.nn.functional.normalize(fused, p=2, dim=1)
    arc_n = torch.nn.functional.normalize(arc.float(), p=2, dim=1)
    return torch.cat((fused_n, alpha * arc_n), dim=1).numpy().astype("float32")


def retrieve(mem_keys, mem_labels, mem_ids, qry_keys, k):
    """faiss cosine top-k. Returns (D, I) with rows in query order."""
    mem = mem_keys.copy()
    qry = qry_keys.copy()
    faiss.normalize_L2(mem)
    faiss.normalize_L2(qry)
    index = faiss.IndexFlatIP(mem.shape[1])
    index.add(mem)
    return index.search(qry, min(k, mem.shape[0]))


def macro(preds, labels):
    preds = np.asarray(preds)
    labels = np.asarray(labels)
    if len(preds) == 0:
        return dict(acc=None, macro_f1=None, n=0)
    return dict(acc=float(np.mean(preds == labels)),
                macro_f1=float(f1_score(labels, preds, average="macro",
                                        zero_division=0)),
                n=int(len(preds)))


# --------------------------------------------------------------------------- #
def build_seed_cache(ds, seed):
    device = "cpu"
    clip_path = os.path.join(ROOT, "data", "CLIP_Embedding")
    model_name = MODEL[ds]
    train, dev, test = load_feats_from_CLIP(clip_path, ds, model_name)
    model = build_head(train[1].shape[1], train[2].shape[1], EasyDict(
        eval_dataset=ds, num_layers=3, proj_dim=1024, map_dim=1024,
        fusion_mode="align", dropout=[0.2, 0.4, 0.1], batch_norm=False))
    model.load_state_dict(torch.load(ckpt_path(ds, seed), map_location="cpu"))
    model.eval()

    tr_ids, tr_emb, tr_lab = project_split(model, train, device)
    dv_ids, dv_emb, dv_lab = project_split(model, dev, device)
    te_ids, te_emb, te_lab = project_split(model, test, device)

    arc = {}
    for name, ids in [("train", tr_ids), ("dev_seen", dv_ids),
                      ("test_seen", te_ids)]:
        arc[name] = load_archive_feats_split(
            resolve_archive_path("auto", os.path.join(ROOT, "data"), ds, name),
            ids)
    mem = augment(torch.tensor(tr_emb), arc["train"])
    val = augment(torch.tensor(dv_emb), arc["dev_seen"])
    tst = augment(torch.tensor(te_emb), arc["test_seen"])
    tr_lab = np.asarray(tr_lab, dtype=int)
    dv_lab = np.asarray(dv_lab, dtype=int)
    te_lab = np.asarray(te_lab, dtype=int)

    # --- val margins -> 25% threshold ---
    vD, vI = retrieve(mem, tr_lab, tr_ids, val, TOPK)
    v_margins = []
    for i in range(len(dv_ids)):
        labs = [tr_lab[j] for j in vI[i]]
        sims = [float(vD[i, r]) for r in range(vI.shape[1])]
        v_margins.append(abs(sim_vote(labs, sims)))
    v_margins = np.sort(np.asarray(v_margins))
    n = len(v_margins)
    kk = int(round(DEFER_RATE * n))
    kk = max(1, min(kk, n - 1))
    threshold = float((v_margins[kk - 1] + v_margins[kk]) / 2.0)

    # --- test top-N_JUDGE retrieval + floor vote + gate ---
    tD, tI = retrieve(mem, tr_lab, tr_ids, tst, N_JUDGE)
    samples = []
    floor_preds = []
    for i in range(len(te_ids)):
        neigh = []
        for r in range(tI.shape[1]):
            j = int(tI[i, r])
            neigh.append([tr_ids[j], float(tD[i, r]), int(tr_lab[j])])
        top_labels = [x[2] for x in neigh[:TOPK]]
        top_sims = [x[1] for x in neigh[:TOPK]]
        vote = sim_vote(top_labels, top_sims)
        pred = vote_pred(vote)
        margin = abs(vote)
        floor_preds.append(pred)
        samples.append(dict(
            id=te_ids[i], label=int(te_lab[i]), margin=margin,
            gated=bool(margin < threshold), floor_vote=vote, floor_pred=pred,
            neighbors=neigh))
    fl = macro(floor_preds, te_lab)
    return dict(
        dataset=ds, seed=seed, ckpt=os.path.basename(ckpt_path(ds, seed)),
        model=model_name, n_val=int(n), val_threshold=threshold,
        n_test=len(te_ids), n_gated=int(sum(s["gated"] for s in samples)),
        floor=dict(acc=fl["acc"], macro_f1=fl["macro_f1"]),
        logged=list(LOGGED_A[ds][seed]), samples=samples)


def cmd_collect(args):
    torch.set_grad_enabled(False)
    os.makedirs(args.cache_dir, exist_ok=True)
    dss = [d.strip() for d in args.datasets.split(",") if d.strip()]
    for ds in dss:
        pairs = set()
        for seed in SEEDS[ds]:
            cache = build_seed_cache(ds, seed)
            path = os.path.join(args.cache_dir, "cache_{}_s{}.json".format(ds, seed))
            with open(path, "w") as f:
                json.dump(cache, f, ensure_ascii=False)
            a_ok = abs(cache["floor"]["acc"] - cache["logged"][0]) < 1e-3
            f_ok = abs(cache["floor"]["macro_f1"] - cache["logged"][1]) < 1e-3
            print("[{} s{}] floor acc/F1 {:.4f}/{:.4f} logged {:.4f}/{:.4f} "
                  "REPRO={} | thr={:.4f} gated {}/{}".format(
                      ds, seed, cache["floor"]["acc"], cache["floor"]["macro_f1"],
                      cache["logged"][0], cache["logged"][1],
                      "OK" if (a_ok and f_ok) else "FAIL",
                      cache["val_threshold"], cache["n_gated"], cache["n_test"]),
                  flush=True)
            for s in cache["samples"]:
                if s["gated"]:
                    for nb in s["neighbors"][:N_JUDGE]:
                        pairs.add((s["id"], nb[0]))
        pp = os.path.join(args.cache_dir, "pairs_{}.jsonl".format(ds))
        with open(pp, "w") as f:
            for qid, nid in sorted(pairs):
                f.write(json.dumps({"query_id": qid, "neighbor_id": nid}) + "\n")
        print("[{}] {} unique gated (query,neighbor) pairs -> {}".format(
            ds, len(pairs), pp), flush=True)


# ----- revote conditions ---------------------------------------------------- #
def revote_from_survivors(neighbors, keep_idx):
    """neighbors: list of [nid, sim, nlabel] (rank order). keep_idx: sorted list
    of indices (into `neighbors`) that survive. Revote over them in rank order."""
    labs = [neighbors[j][2] for j in keep_idx]
    sims = [neighbors[j][1] for j in keep_idx]
    v = sim_vote(labs, sims)
    return vote_pred(v), v


def apply_drop_rule(neighbors, drop_top20, addable_beyond):
    """Frozen revote machinery shared by B / C / D.

    drop_top20      : set of indices in [0,20) to drop from the top-20.
    addable_beyond  : callable(idx)->bool for ranks [20, N_JUDGE): may this deeper
                      neighbor be appended during extension?
    Returns (pred, meta) where meta records survivors, dropped count, extension
    and fallback flags. Falls back to the ORIGINAL top-20 vote if <3 survive even
    after extension to N_JUDGE.
    """
    n = len(neighbors)
    top = min(TOPK, n)
    survivors = [j for j in range(top) if j not in drop_top20]
    dropped = top - len(survivors)
    extended = 0
    fallback = False
    if len(survivors) < MIN_SURVIVE:
        for j in range(TOPK, min(N_JUDGE, n)):
            if addable_beyond(j):
                survivors.append(j)
                extended += 1
                if len(survivors) >= MIN_SURVIVE:
                    break
    if len(survivors) < MIN_SURVIVE:
        fallback = True
        survivors = list(range(top))       # original top-20 vote
        extended = 0
    survivors = sorted(survivors)
    pred, vote = revote_from_survivors(neighbors, survivors)
    return pred, dict(dropped=dropped, n_survive=len(survivors),
                      extended=extended, fallback=fallback, vote=vote)


def cmd_revote(args):
    dss = [d.strip() for d in args.datasets.split(",") if d.strip()]
    out = {}
    for ds in dss:
        verdicts = {}
        tag = getattr(args, "verdicts_tag", "") or ""
        fname = "verdicts_{}{}.jsonl".format(ds, ("_" + tag) if tag else "")
        vp = os.path.join(args.cache_dir, fname)
        n_fb = 0
        vcount = Counter()
        if os.path.exists(vp):
            for line in open(vp):
                line = line.strip()
                if not line:
                    continue
                r = json.loads(line)
                verdicts[(r["query_id"], r["neighbor_id"])] = r["verdict"]
                if r.get("fallback"):
                    n_fb += 1
        per_seed = []
        for seed in SEEDS[ds]:
            cache = json.load(open(os.path.join(
                args.cache_dir, "cache_{}_s{}.json".format(ds, seed))))
            rng = random.Random(RANDOM_SEED)
            rows = sorted(cache["samples"], key=lambda s: s["id"])
            labels = np.array([s["label"] for s in rows])
            gated = np.array([s["gated"] for s in rows])
            predA, predB, predC, predD = [], [], [], []
            drop_hist, ext_ct, fb_ct = [], dict(B=0, C=0, D=0), dict(B=0, C=0, D=0)
            for s in rows:
                nb = s["neighbors"]
                fp = s["floor_pred"]
                if not s["gated"]:
                    predA.append(fp); predB.append(fp)
                    predC.append(fp); predD.append(fp)
                    continue
                top = min(TOPK, len(nb))
                vlist = [verdicts.get((s["id"], nb[j][0]), "UNSURE")
                         for j in range(top)]
                for v in vlist:
                    vcount[v] += 1
                # --- A ---
                predA.append(fp)
                # --- B: drop INCOMPARABLE; extend over non-INCOMPARABLE ---
                dropB = {j for j in range(top) if vlist[j] == "INCOMPARABLE"}
                addB = lambda j: (verdicts.get((s["id"], nb[j][0]), "UNSURE")
                                  != "INCOMPARABLE")
                pB, mB = apply_drop_rule(nb, dropB, addB)
                predB.append(pB); drop_hist.append(mB["dropped"])
                ext_ct["B"] += mB["extended"]; fb_ct["B"] += int(mB["fallback"])
                # --- C: same #drops, random, over the top-20 ---
                d = len(dropB)
                idx = list(range(top))
                rng.shuffle(idx)
                dropC = set(idx[:d])
                addC = lambda j: True     # random control has no comparability
                pC, mC = apply_drop_rule(nb, dropC, addC)
                predC.append(pC)
                ext_ct["C"] += mC["extended"]; fb_ct["C"] += int(mC["fallback"])
                # --- D: oracle, keep only gold-label neighbors ---
                gold = s["label"]
                dropD = {j for j in range(top) if nb[j][2] != gold}
                addD = lambda j: nb[j][2] == gold
                pD, mD = apply_drop_rule(nb, dropD, addD)
                predD.append(pD)
                ext_ct["D"] += mD["extended"]; fb_ct["D"] += int(mD["fallback"])

            g = gated
            row = dict(
                seed=seed, n_test=len(rows), n_gated=int(g.sum()),
                overall=dict(A=macro(predA, labels), B=macro(predB, labels),
                             C=macro(predC, labels), D=macro(predD, labels)),
                gated=dict(
                    A=macro(np.array(predA)[g], labels[g]),
                    B=macro(np.array(predB)[g], labels[g]),
                    C=macro(np.array(predC)[g], labels[g]),
                    D=macro(np.array(predD)[g], labels[g])),
                drop_mean=float(np.mean(drop_hist)) if drop_hist else 0.0,
                drop_hist=drop_hist, ext=ext_ct, fb=fb_ct)
            per_seed.append(row)
            print("[{} s{}] gated {}/{} | A={:.4f} B={:.4f} C={:.4f} D={:.4f} "
                  "(gated: A={:.4f} B={:.4f} C={:.4f} D={:.4f}) drop~{:.1f}".format(
                      ds, seed, int(g.sum()), len(rows),
                      row["overall"]["A"]["acc"], row["overall"]["B"]["acc"],
                      row["overall"]["C"]["acc"], row["overall"]["D"]["acc"],
                      row["gated"]["A"]["acc"], row["gated"]["B"]["acc"],
                      row["gated"]["C"]["acc"], row["gated"]["D"]["acc"],
                      row["drop_mean"]), flush=True)
        out[ds] = dict(model=MODEL[ds], seeds=SEEDS[ds], per_seed=per_seed,
                       verdict_used=dict(vcount), parse_fallback=n_fb,
                       n_verdicts=len(verdicts))
    summarize(out)
    with open(args.out_json, "w") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    print("wrote", args.out_json)
    write_block(out, args.out_block)
    print("wrote", args.out_block)


# ----- aggregation ---------------------------------------------------------- #
def _vals(ps, scope, cond, field):
    return [r[scope][cond][field] for r in ps]


def agg(vals):
    vals = [v for v in vals if v is not None]
    if not vals:
        return None
    a = np.array(vals, float)
    return dict(mean=float(a.mean()), std=float(a.std(ddof=0)),
                vals=[float(x) for x in a])


def paired(ps, scope, x, y, field):
    ds = []
    for r in ps:
        rx, ry = r[scope][x][field], r[scope][y][field]
        ds.append(None if (rx is None or ry is None) else rx - ry)
    good = [d for d in ds if d is not None]
    npos = sum(1 for d in good if d > 1e-9)
    s = dict(per_seed=ds, npos=npos, n=len(good))
    if good:
        a = np.array(good)
        s.update(mean=float(a.mean()), std=float(a.std(ddof=0)))
    return s


def summarize(out):
    for ds, d in out.items():
        ps = d["per_seed"]
        d["mean_std"] = {}
        for scope in ["overall", "gated"]:
            d["mean_std"][scope] = {
                c: dict(acc=agg(_vals(ps, scope, c, "acc")),
                        macro_f1=agg(_vals(ps, scope, c, "macro_f1")))
                for c in ["A", "B", "C", "D"]}
        d["paired"] = {}
        for scope in ["overall", "gated"]:
            d["paired"][scope] = {
                "B_minus_A": dict(acc=paired(ps, scope, "B", "A", "acc"),
                                  macro_f1=paired(ps, scope, "B", "A", "macro_f1")),
                "B_minus_C": dict(acc=paired(ps, scope, "B", "C", "acc"),
                                  macro_f1=paired(ps, scope, "B", "C", "macro_f1")),
                "D_minus_A": dict(acc=paired(ps, scope, "D", "A", "acc"),
                                  macro_f1=paired(ps, scope, "D", "A", "macro_f1")),
            }
        # B fraction of D (overall acc)
        mo = d["mean_std"]["overall"]
        num = mo["B"]["acc"]["mean"] - mo["A"]["acc"]["mean"]
        den = mo["D"]["acc"]["mean"] - mo["A"]["acc"]["mean"]
        d["B_frac_of_D"] = (None if abs(den) < 1e-9 else float(num / den))


def fmt_ms(m):
    return "n/a" if not m else "{:.4f} ± {:.4f}".format(m["mean"], m["std"])


def fmt_pair(p):
    return "n/a" if "mean" not in p else "{:+.4f} ± {:.4f} (+{}/{})".format(
        p["mean"], p["std"], p["npos"], p["n"])


def write_block(out, path):
    L = []
    w = L.append
    w("### Machine-generated results block "
      "(scripts/analysis/p2_rerank_eval.py --mode revote)\n")
    for ds in out:
        d = out[ds]
        tag = "EN" if ds == "MHC" else "ZH"
        ps = d["per_seed"]
        w("#### {} ({}) — {} seeds\n".format(ds, tag, len(d["seeds"])))
        w("Verdict distribution over USED top-20 gated neighbors: {} "
          "| parse-fallback rows: {} | total verdicts cached: {}\n".format(
              dict(d["verdict_used"]), d["parse_fallback"], d["n_verdicts"]))
        # overall
        w("Per-seed OVERALL accuracy / macro-F1:\n")
        w("| seed | gated/n | A floor | B MLLM | C random | D oracle | drop~/q | ext B/C/D | fb B/C/D |")
        w("|---|---|---|---|---|---|---|---|---|")
        for r in ps:
            def cell(c):
                m = r["overall"][c]
                return "{:.4f}/{:.4f}".format(m["acc"], m["macro_f1"])
            w("| {} | {}/{} | {} | {} | {} | {} | {:.2f} | {}/{}/{} | {}/{}/{} |".format(
                r["seed"], r["n_gated"], r["n_test"], cell("A"), cell("B"),
                cell("C"), cell("D"), r["drop_mean"],
                r["ext"]["B"], r["ext"]["C"], r["ext"]["D"],
                r["fb"]["B"], r["fb"]["C"], r["fb"]["D"]))
        mo = d["mean_std"]["overall"]
        w("\nOverall mean ± std (acc):")
        w("| A | B | C | D |")
        w("|---|---|---|---|")
        w("| {} | {} | {} | {} |".format(
            fmt_ms(mo["A"]["acc"]), fmt_ms(mo["B"]["acc"]),
            fmt_ms(mo["C"]["acc"]), fmt_ms(mo["D"]["acc"])))
        w("\nOverall mean ± std (macro-F1):")
        w("| A | B | C | D |")
        w("|---|---|---|---|")
        w("| {} | {} | {} | {} |".format(
            fmt_ms(mo["A"]["macro_f1"]), fmt_ms(mo["B"]["macro_f1"]),
            fmt_ms(mo["C"]["macro_f1"]), fmt_ms(mo["D"]["macro_f1"])))
        # gated
        w("\nPer-seed GATED-SUBSET accuracy / macro-F1:\n")
        w("| seed | n_gated | A floor | B MLLM | C random | D oracle |")
        w("|---|---|---|---|---|---|")
        for r in ps:
            def cellg(c):
                m = r["gated"][c]
                if m["acc"] is None:
                    return "—"
                return "{:.4f}/{:.4f}".format(m["acc"], m["macro_f1"])
            w("| {} | {} | {} | {} | {} | {} |".format(
                r["seed"], r["n_gated"], cellg("A"), cellg("B"),
                cellg("C"), cellg("D")))
        mg = d["mean_std"]["gated"]
        w("\nGated-subset mean ± std (acc): A {} | B {} | C {} | D {}\n".format(
            fmt_ms(mg["A"]["acc"]), fmt_ms(mg["B"]["acc"]),
            fmt_ms(mg["C"]["acc"]), fmt_ms(mg["D"]["acc"])))
        # paired
        po, pg = d["paired"]["overall"], d["paired"]["gated"]
        w("Paired per-seed deltas (acc / macro-F1):\n")
        w("| delta | overall | gated subset |")
        w("|---|---|---|")
        w("| B − A (ours vs floor) | {} | {} |".format(
            fmt_pair(po["B_minus_A"]["acc"]) + " / " + fmt_pair(po["B_minus_A"]["macro_f1"]),
            fmt_pair(pg["B_minus_A"]["acc"]) + " / " + fmt_pair(pg["B_minus_A"]["macro_f1"])))
        w("| B − C (rent test: ours vs random) | {} | {} |".format(
            fmt_pair(po["B_minus_C"]["acc"]) + " / " + fmt_pair(po["B_minus_C"]["macro_f1"]),
            fmt_pair(pg["B_minus_C"]["acc"]) + " / " + fmt_pair(pg["B_minus_C"]["macro_f1"])))
        w("| D − A (oracle headroom) | {} | {} |".format(
            fmt_pair(po["D_minus_A"]["acc"]) + " / " + fmt_pair(po["D_minus_A"]["macro_f1"]),
            fmt_pair(pg["D_minus_A"]["acc"]) + " / " + fmt_pair(pg["D_minus_A"]["macro_f1"])))
        w("\nB's fraction of the oracle headroom (overall acc): {}\n".format(
            "n/a" if d["B_frac_of_D"] is None else "{:.2f}".format(d["B_frac_of_D"])))
    with open(path, "w") as f:
        f.write("\n".join(L) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", required=True, choices=["collect", "revote"])
    ap.add_argument("--datasets", default="MHC,MHC_zh")
    ap.add_argument("--verdicts_tag", default="",
                    help="revote: read verdicts_<ds>_<tag>.jsonl (P2b promoted config)")
    ap.add_argument("--cache_dir",
                    default=os.path.join(ROOT, "scripts/analysis/p2_out"))
    ap.add_argument("--out_json",
                    default=os.path.join(ROOT, "scripts/analysis/p2_out/p2_results.json"))
    ap.add_argument("--out_block",
                    default=os.path.join(ROOT, "scripts/analysis/p2_out/p2_results_block.md"))
    args = ap.parse_args()
    os.makedirs(args.cache_dir, exist_ok=True)
    if args.mode == "collect":
        cmd_collect(args)
    else:
        cmd_revote(args)


if __name__ == "__main__":
    main()
