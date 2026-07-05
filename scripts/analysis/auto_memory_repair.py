#!/usr/bin/env python
"""Automatic memory repair — CPU evaluation (research-wiki/EXP_auto_memory_repair.md).

Upgrades the MANUAL memory-editing demo into a fully AUTOMATIC two-vote deletion rule and
measures whether it recovers the manual gain with no human in the loop. Design is
pre-registered in the wiki doc; thresholds here are FIXED (never tuned on val/test).

Two votes over TRAIN memory entries only:
  1. Embedding vote (per seed): leave-one-out kNN over the memory bank in the SAME augmented
     -key cosine space the decision uses (k=10, self excluded). Flag entry i iff >= 80% of its
     10 neighbours carry the OPPOSITE label. Continuous disagreement fraction also recorded.
  2. Semantic vote (MLLM, seed-independent): Qwen verdict {SUPPORT, CONTRADICT, UNSURE} from
     scripts/analysis/judge_memory_archive.py, keyed by (dataset, version, id).
Deletion (condition C): embedding-flag AND verdict == CONTRADICT.

Conditions per seed (each = ONE test measurement, no repeats, no selection):
  A no deletion | B manual 2-id (EN only) | C auto two-vote | D embedding-only (Cleanlab-style)
  | E random |C|-matched (seed 0).

Reuses memory_editing_demo.knn_eval/augment (which reproduce the training-log floor
bit-for-bit) and eval_cross_dataset.project_split/build_head, all READ-ONLY. CPU-only.
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

ROOT = "/data/jehc223/RGCL"
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.join(ROOT, "scripts", "analysis"))

from memory_editing_demo import (  # noqa: E402  (protocol helpers; reproduces the floor)
    ALPHA, TOPK, CKPT, NOISY_IDS,
    augment, knn_eval, load_archive_records, archive_text)
from eval_cross_dataset import project_split, build_head  # noqa: E402
from data_loader.dataset import (  # noqa: E402
    load_feats_from_CLIP, load_archive_feats_split, resolve_archive_path)
from easydict import EasyDict  # noqa: E402

# --- pre-registered thresholds (FIXED) ---
EMB_K = 10
EMB_THRESH = 0.80         # flag if >= 80% of k neighbours carry the opposite label
RANDOM_SEED = 0           # condition E

MODEL = {"MHC": "Qwen2.5-VL-7B-Instruct_HF", "MHC_zh": "Qwen2.5-VL-7B-Instruct-LoRA_HF"}
# val-selected best checkpoints per seed (epochs verified against exp-archive-knn-seeds.md;
# seed 0 == the manual-demo ckpt in CKPT[...]).
CKPT_FILE = {
    "MHC": {0: "best_model_24_0.7875.pt", 1: "best_model_29_0.7875.pt",
            2: "best_model_21_0.8125.pt", 3: "best_model_27_0.7875.pt"},
    "MHC_zh": {0: "best_model_18_0.8717948717948718.pt",
               1: "best_model_23_0.8846153846153846.pt",
               2: "best_model_14_0.8846153846153846.pt",
               3: "best_model_17_0.8717948717948718.pt",
               4: "best_model_12_0.8717948717948718.pt"},
}
SEEDS = {"MHC": [0, 1, 2, 3], "MHC_zh": [0, 1, 2, 3, 4]}
# logged val-selected floor (acc/macroF1) per seed, from exp-archive-knn-seeds.md, for a
# reproduction sanity print of condition A.
LOGGED_A = {
    "MHC": {0: (0.8075, 0.7626), 1: (0.7640, 0.7145), 2: (0.7950, 0.7505), 3: (0.8075, 0.7713)},
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


def embedding_vote(mem_keys, mem_lab, k=EMB_K, thresh=EMB_THRESH):
    """Leave-one-out kNN label-disagreement over the augmented-key cosine space."""
    mem = mem_keys.copy()
    faiss.normalize_L2(mem)
    index = faiss.IndexFlatIP(mem.shape[1])
    index.add(mem)
    _, I = index.search(mem, k + 1)  # rank 0 is self (IP with itself = 1)
    n = len(mem_lab)
    flags = np.zeros(n, dtype=bool)
    disagree = np.zeros(n, dtype=float)
    for i in range(n):
        neigh = [j for j in I[i] if j != i][:k]
        opp = np.mean([1.0 if mem_lab[j] != mem_lab[i] else 0.0 for j in neigh])
        disagree[i] = opp
        flags[i] = opp >= thresh
    return flags, disagree


def one_liner(rec, disagree_frac=None, verdict=None):
    a = rec.get("archive") or {}
    tg = a.get("target_groups") or []
    summ = (a.get("neutral_summary") or "").strip().replace("\n", " ")
    s = "label={} targets={} expl={} :: {}".format(
        int(rec.get("label", 0)), tg, a.get("explicitness"), summ[:110])
    extra = []
    if disagree_frac is not None:
        extra.append("emb_disagree={:.2f}".format(disagree_frac))
    if verdict is not None:
        extra.append("mllm={}".format(verdict))
    if extra:
        s = "[" + " ".join(extra) + "] " + s
    return s


def run_dataset(ds, verdicts, version, out):
    device = "cpu"
    clip_path = os.path.join(ROOT, "data", "CLIP_Embedding")
    model_name = MODEL[ds]
    train, dev, test = load_feats_from_CLIP(clip_path, ds, model_name)
    recs = load_archive_records(ds)               # v1 archive records (for audit one-liners)
    vmap = (verdicts.get(ds, {}) or {}).get(version, {})   # id -> {verdict,...}

    per_seed = []
    for seed in SEEDS[ds]:
        cp = ckpt_path(ds, seed)
        model = build_head(train[1].shape[1], train[2].shape[1], EasyDict(
            eval_dataset=ds, num_layers=3, proj_dim=1024, map_dim=1024,
            fusion_mode="align", dropout=[0.2, 0.4, 0.1], batch_norm=False))
        model.load_state_dict(torch.load(cp, map_location="cpu"))
        model.eval()

        tr_ids, tr_emb, tr_lab = project_split(model, train, device)
        te_ids, te_emb, te_lab = project_split(model, test, device)
        arc_tr = load_archive_feats_split(
            resolve_archive_path("auto", os.path.join(ROOT, "data"), ds, "train"), tr_ids)
        arc_te = load_archive_feats_split(
            resolve_archive_path("auto", os.path.join(ROOT, "data"), ds, "test_seen"), te_ids)
        mem_keys = augment(torch.tensor(tr_emb), arc_tr)
        qry_keys = augment(torch.tensor(te_emb), arc_te)
        N = len(tr_ids)

        # --- votes (train-side only) ---
        flags, disagree = embedding_vote(mem_keys, tr_lab)
        contradict = np.array(
            [vmap.get(vid, {}).get("verdict", "UNSURE") == "CONTRADICT" for vid in tr_ids])
        missing_verdict = int(sum(1 for vid in tr_ids if vid not in vmap))
        c_mask = flags & contradict           # condition C deletion set
        d_mask = flags                        # condition D deletion set

        def evaluate(keep_mask):
            keep = np.where(keep_mask)[0]
            macro, preds, _ = knn_eval(
                mem_keys[keep], tr_lab[keep], [tr_ids[j] for j in keep], qry_keys, te_lab)
            return float(macro["acc"]), float(macro["macro_f1"]), preds

        # A: no deletion
        full = np.ones(N, dtype=bool)
        accA, f1A, _ = evaluate(full)
        # B: manual 2-id (EN only; ids present)
        manual_present = [i for i in tr_ids if i in NOISY_IDS]
        if manual_present:
            keepB = np.array([i not in NOISY_IDS for i in tr_ids])
            accB, f1B, _ = evaluate(keepB)
        else:
            accB, f1B = None, None
        # C: auto two-vote
        keepC = ~c_mask
        accC, f1C, _ = evaluate(keepC) if c_mask.any() else (accA, f1A, None)
        # D: embedding-only
        keepD = ~d_mask
        accD, f1D, _ = evaluate(keepD) if d_mask.any() else (accA, f1A, None)
        # E: random |C|-matched, fixed seed 0
        nC = int(c_mask.sum())
        if nC > 0:
            rng = random.Random(RANDOM_SEED)
            drop = set(rng.sample(range(N), nC))
            keepE = np.array([j not in drop for j in range(N)])
            accE, f1E, _ = evaluate(keepE)
        else:
            accE, f1E = accA, f1A
        # F: semantic-vote-only (EXPLORATORY, post-hoc; not pre-registered). Delete every
        # CONTRADICT entry regardless of the embedding vote — isolates the MLLM signal and
        # explains whether the AND in C is what suppresses the manual recovery.
        f_mask = contradict
        accF, f1F, _ = evaluate(~f_mask) if f_mask.any() else (accA, f1A, None)

        del_C = [(vid, one_liner(recs[vid], disagree[j], vmap.get(vid, {}).get("verdict"))
                  ) for j, vid in enumerate(tr_ids) if c_mask[j]]
        del_D = [(vid, one_liner(recs[vid], disagree[j], vmap.get(vid, {}).get("verdict"))
                  ) for j, vid in enumerate(tr_ids) if d_mask[j]]
        del_F = [(vid, one_liner(recs[vid], disagree[j], vmap.get(vid, {}).get("verdict"))
                  ) for j, vid in enumerate(tr_ids) if f_mask[j]]

        row = dict(
            seed=seed, ckpt=os.path.basename(cp), mem_n=N, test_n=len(te_ids),
            n_flag=int(flags.sum()), n_contradict=int(contradict.sum()),
            n_C=nC, n_D=int(d_mask.sum()), n_F=int(f_mask.sum()),
            missing_verdict=missing_verdict,
            A=dict(acc=accA, f1=f1A),
            B=(dict(acc=accB, f1=f1B) if accB is not None else None),
            C=dict(acc=accC, f1=f1C),
            D=dict(acc=accD, f1=f1D),
            E=dict(acc=accE, f1=f1E),
            F=dict(acc=accF, f1=f1F),
            logged_A=LOGGED_A[ds][seed],
            deleted_C=del_C, deleted_D=del_D, deleted_F=del_F,
        )
        per_seed.append(row)
        gate = ""
        if seed == 0 and ds == "MHC":
            gate = " | GATE A~0.8075:{} B~0.8199:{}".format(
                "OK" if abs(accA - 0.8075) < 1e-3 else "FAIL(%.4f)" % accA,
                "OK" if (accB is not None and abs(accB - 0.8199) < 1e-3)
                else "FAIL(%s)" % accB)
        print("[{} s{}] A={:.4f}/{:.4f} C={:.4f}/{:.4f} D={:.4f}/{:.4f} E={:.4f}/{:.4f} "
              "| flag={} contra={} |C|={} |D|={}{}".format(
                  ds, seed, accA, f1A, accC, f1C, accD, f1D, accE, f1E,
                  int(flags.sum()), int(contradict.sum()), nC, int(d_mask.sum()), gate),
              flush=True)

    out[ds] = dict(model=model_name, version=version, seeds=SEEDS[ds],
                   emb_k=EMB_K, emb_thresh=EMB_THRESH, per_seed=per_seed)


def agg(vals):
    vals = [v for v in vals if v is not None]
    if not vals:
        return None
    a = np.array(vals, dtype=float)
    return dict(mean=float(a.mean()), std=float(a.std(ddof=0)),
                vals=[float(x) for x in a])


def paired(per_seed, x, y, field):
    """per-seed deltas x-y for a metric field ('acc'/'f1')."""
    ds = []
    for r in per_seed:
        rx, ry = r[x], r[y]
        if rx is None or ry is None:
            ds.append(None)
        else:
            ds.append(rx[field] - ry[field])
    good = [d for d in ds if d is not None]
    npos = sum(1 for d in good if d > 1e-9)
    summ = dict(per_seed=ds, npos=npos, n=len(good))
    if good:
        a = np.array(good)
        summ.update(mean=float(a.mean()), std=float(a.std(ddof=0)))
    return summ


def summarize(out):
    for ds, d in out.items():
        ps = d["per_seed"]
        d["mean_std"] = {cond: dict(
            acc=agg([r[cond]["acc"] if r[cond] else None for r in ps]),
            f1=agg([r[cond]["f1"] if r[cond] else None for r in ps]))
            for cond in ["A", "B", "C", "D", "E", "F"]}
        d["paired"] = {
            "C_minus_A": dict(acc=paired(ps, "C", "A", "acc"), f1=paired(ps, "C", "A", "f1")),
            "C_minus_D": dict(acc=paired(ps, "C", "D", "acc"), f1=paired(ps, "C", "D", "f1")),
            "D_minus_A": dict(acc=paired(ps, "D", "A", "acc"), f1=paired(ps, "D", "A", "f1")),
            "F_minus_A": dict(acc=paired(ps, "F", "A", "acc"), f1=paired(ps, "F", "A", "f1")),
        }


def fmt_ms(m):
    if not m:
        return "n/a"
    return "{:.4f} ± {:.4f}".format(m["mean"], m["std"])


def fmt_pair(p):
    if "mean" not in p:
        return "n/a"
    return "{:+.4f} ± {:.4f} (+{}/{})".format(p["mean"], p["std"], p["npos"], p["n"])


def write_block(out, path):
    L = []
    w = L.append
    w("### Machine-generated results block "
      "(scripts/analysis/auto_memory_repair.py)\n")
    for ds in out:
        d = out[ds]
        tag = "EN" if ds == "MHC" else "ZH"
        w("#### {} ({}) — v{} verdicts, {} seeds\n".format(
            ds, tag, d["version"].lstrip("v"), len(d["seeds"])))
        w("Per-seed accuracy / macro-F1 (F = exploratory semantic-only):\n")
        w("| seed | A floor | B manual | C auto | D emb-only | E random | F sem-only | flag | contra | |C| | |D| | |F| |")
        w("|---|---|---|---|---|---|---|---|---|---|---|---|")
        for r in d["per_seed"]:
            def cell(c):
                return "{:.4f}/{:.4f}".format(r[c]["acc"], r[c]["f1"]) if r[c] else "—"
            w("| {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} |".format(
                r["seed"], cell("A"), cell("B"), cell("C"), cell("D"), cell("E"), cell("F"),
                r["n_flag"], r["n_contradict"], r["n_C"], r["n_D"], r["n_F"]))
        ms = d["mean_std"]
        w("\nMean ± std (acc):")
        w("| A | B | C | D | E | F |")
        w("|---|---|---|---|---|---|")
        w("| {} | {} | {} | {} | {} | {} |".format(
            fmt_ms(ms["A"]["acc"]), fmt_ms(ms["B"]["acc"]), fmt_ms(ms["C"]["acc"]),
            fmt_ms(ms["D"]["acc"]), fmt_ms(ms["E"]["acc"]), fmt_ms(ms["F"]["acc"])))
        w("\nMean ± std (macro-F1):")
        w("| A | B | C | D | E | F |")
        w("|---|---|---|---|---|---|")
        w("| {} | {} | {} | {} | {} | {} |".format(
            fmt_ms(ms["A"]["f1"]), fmt_ms(ms["B"]["f1"]), fmt_ms(ms["C"]["f1"]),
            fmt_ms(ms["D"]["f1"]), fmt_ms(ms["E"]["f1"]), fmt_ms(ms["F"]["f1"])))
        pr = d["paired"]
        w("\nPaired per-seed deltas (acc / macro-F1):\n")
        w("| delta | acc | macro-F1 |")
        w("|---|---|---|")
        w("| C − A (method vs floor) | {} | {} |".format(fmt_pair(pr["C_minus_A"]["acc"]),
                                                          fmt_pair(pr["C_minus_A"]["f1"])))
        w("| C − D (method vs emb-only) | {} | {} |".format(fmt_pair(pr["C_minus_D"]["acc"]),
                                                            fmt_pair(pr["C_minus_D"]["f1"])))
        w("| D − A (emb-only vs floor) | {} | {} |".format(fmt_pair(pr["D_minus_A"]["acc"]),
                                                           fmt_pair(pr["D_minus_A"]["f1"])))
        w("| F − A (sem-only vs floor, exploratory) | {} | {} |".format(
            fmt_pair(pr["F_minus_A"]["acc"]), fmt_pair(pr["F_minus_A"]["f1"])))
        # deleted-entry audit (dedup across seeds; report which seeds deleted each id)
        w("\nEntries deleted by **C** (auto two-vote), by id (seeds that deleted it):\n")
        cdel = {}
        for r in d["per_seed"]:
            for vid, ol in r["deleted_C"]:
                cdel.setdefault(vid, (ol, []))[1].append(r["seed"])
        if cdel:
            for vid, (ol, seeds) in sorted(cdel.items()):
                w("- `{}` (seeds {}): {}".format(vid, seeds, ol))
        else:
            w("- (none — C deleted nothing on any seed)")
        w("\nEntries deleted by **D** (embedding-only), union across seeds "
          "(count {}):\n".format(len({v for r in d["per_seed"] for v, _ in r["deleted_D"]})))
        ddel = {}
        for r in d["per_seed"]:
            for vid, ol in r["deleted_D"]:
                ddel.setdefault(vid, (ol, []))[1].append(r["seed"])
        for vid, (ol, seeds) in sorted(ddel.items()):
            w("- `{}` (seeds {}): {}".format(vid, seeds, ol))
        fdel = {}
        for r in d["per_seed"]:
            for vid, ol in r["deleted_F"]:
                fdel.setdefault(vid, (ol, []))[1].append(r["seed"])
        w("\nEntries deleted by **F** (exploratory semantic-only), union across seeds "
          "(count {}) — note this is seed-independent (verdicts do not depend on the head), "
          "so all entries show every seed:\n".format(len(fdel)))
        for vid, (ol, seeds) in sorted(fdel.items()):
            w("- `{}`: {}".format(vid, ol))
        w("")
    with open(path, "w") as f:
        f.write("\n".join(L) + "\n")
    print("wrote", path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--verdicts", required=True, help="verdicts.json from the MLLM judge")
    ap.add_argument("--version", default="v1", help="archive version for the semantic vote")
    ap.add_argument("--datasets", default="MHC,MHC_zh")
    ap.add_argument("--out_json", required=True)
    ap.add_argument("--out_block", required=True, help="markdown results block to append")
    args = ap.parse_args()
    torch.set_grad_enabled(False)

    with open(args.verdicts) as f:
        verdicts = json.load(f)

    out = {}
    for ds in [d.strip() for d in args.datasets.split(",") if d.strip()]:
        run_dataset(ds, verdicts, args.version, out)
    summarize(out)
    with open(args.out_json, "w") as f:
        json.dump(out, f, indent=1, ensure_ascii=False)
    print("wrote", args.out_json)
    write_block(out, args.out_block)


if __name__ == "__main__":
    main()
