#!/usr/bin/env python
"""
ksweep_vote.py -- FORENSIC top-k sweep of the deployed kNN vote.

QUESTION
--------
The deployed retrieval vote uses top-20 neighbours. Has anyone tried REDUCING k
to cut neighbourhood noise? No explicit sweep exists in the records. This script
runs it at $0 from ALREADY-BANKED per-item neighbour lists -- no model, no
retraining, no GPU, no new test inference.

WHY THIS IS EXACTLY EQUIVALENT TO RE-RUNNING WITH --topk k
----------------------------------------------------------
`--topk` is consumed in only two places in src/run_rac.py: the retrieval depth
`largest_retrieval=args.topk` (line 818/836) and `topk=args.topk` in
compute_metrics_retrieval (line 828/846), plus the experiment-name string
(line 1016). It never enters the loss, the hard-negative miner, or the optimiser.
So the trained head at a given (seed, epoch) is IDENTICAL under any k, and the
top-k neighbour list is the length-k prefix of the banked top-20 list. Truncating
the banked lists therefore reproduces a `--topk k` re-run exactly (the similarity
threshold is not binding: every banked item has n_retrieved == 20).

VOTE FORMULA (src/utils/metrics.py:262-301, use_sim=True + majority_voting='arithmetic')
    w    = [k, k-1, ..., 1]                      (metrics.py:229-231 with topk=k)
    v    = sum_i (2*lab_i - 1) * cos_i * w_i / sum_i w_i
    pred = 1  iff  sigmoid(v) >= 0.5  <=>  v >= 0
Note w is RE-DERIVED at each k (as the deployed code does), not the 20-vector
truncated -- [k..1] is not proportional to [20..21-k].

DISCIPLINE
----------
Read-only forensics over test predictions that were already banked and already
consumed (same basis as the ERRPAT reports). No new test inference. Any k chosen
by test accuracy is labelled ORACLE / FORENSIC and is NOT deployable. The one
deployment-legal read -- "what k would DEV have picked, and what does that k
score on test" -- is computed separately and reported honestly, including when
it loses.

Outputs: scripts/analysis/ksweep_OUT.json
"""
import glob
import json
import os
import pickle
import re
import sys

import numpy as np

REPO = "/data/jehc223/RGCL"
sys.path.insert(0, os.path.join(REPO, "src"))
sys.path.insert(0, os.path.join(REPO, "scripts", "analysis"))

K_GRID = [1, 2, 3, 5, 7, 10, 15, 20]
K_GRID_EXT = [1, 2, 3, 5, 7, 10, 15, 20, 30, 40, 60]
DEPLOYED_K = 20

OUT_PATH = os.path.join(REPO, "scripts", "analysis", "ksweep_OUT.json")

HATEMM_PROXY_ROOT = ("/data/jehc223/home/tmp/claude-135258174/-data-jehc223-RGCL/"
                     "e8f03e41-3e21-4cea-b12c-29207373bfca/scratchpad/errpat")
HATEMM_GT = os.path.join(REPO, "data/gt/HateMM/{}.jsonl")
ZH_DUMP = os.path.join(REPO, "scripts/analysis/errpat_remint_dumps/errpat_zh_remint_seed{}.pkl")
EN_P2 = os.path.join(REPO, "scripts/analysis/p2_out")
EN_CKPT = os.path.join(REPO, "refine-logs", "router_ckpt_snapshot")

# ---- recorded k=20 values, transcribed from the ERRPAT reports / banked headers.
# Every one of these is re-read from its primary source below where possible; the
# hardcoded copy is the independent second gate.
ANCHOR_HATEMM = {  # (seed, protocol) -> (acc, mf1)  refine-logs/ERRPAT_HateMM_2026-07-26.md sec 1
    (0, "valsel"): (0.8791, 0.8730), (1, "valsel"): (0.8744, 0.8684),
    (2, "valsel"): (0.8791, 0.8730),
    (0, "final"): (0.8698, 0.8632), (1, "final"): (0.8791, 0.8735),
    (2, "final"): (0.8791, 0.8730),
}
ANCHOR_ZH_FINAL = {  # refine-logs/ERRPAT_MHC-ZH_2026-07-26.md sec 0.  re-mint column
    0: (0.8456, 0.8158), 1: (0.8389, 0.8090), 2: (0.8523, 0.8226),
}
ANCHOR_EN_ARMF = {  # scripts/analysis/errpat_mhc_en.py:47-54 (primary trainlogs)
    0: (0.8012, 0.7596), 1: (0.7702, 0.7203), 2: (0.7826, 0.7475),
}


# --------------------------------------------------------------------- helpers
def vote_at_k(nb_lab, nb_sim, k):
    """Deployed vote truncated at k. nb_lab/nb_sim: (n_items, >=k) arrays."""
    lab = np.asarray(nb_lab, dtype=np.float64)[:, :k]
    sim = np.asarray(nb_sim, dtype=np.float64)[:, :k]
    w = np.arange(1, k + 1)[::-1].astype(np.float64)
    return ((lab * 2.0 - 1.0) * sim * w).sum(1) / w.sum()


def macro_f1(y, p):
    y = np.asarray(y)
    p = np.asarray(p)
    f1s = []
    for c in (0, 1):
        tp = np.sum((p == c) & (y == c))
        fp = np.sum((p == c) & (y != c))
        fn = np.sum((p != c) & (y == c))
        pr = tp / (tp + fp) if (tp + fp) else 0.0
        rc = tp / (tp + fn) if (tp + fn) else 0.0
        f1s.append(2 * pr * rc / (pr + rc) if (pr + rc) else 0.0)
    return float(np.mean(f1s))


def score(y, votes):
    p = (votes >= 0.0).astype(int)
    return float((p == y).mean()), macro_f1(y, p), p


def parse_trainlog(path, warmup=5):
    """Deployed epoch-selection rule, verbatim from scripts/slurm/*.sbatch:
    warm = epochs >= 5; best = max(warm, key=(val_acc, val_roc)) -> earliest on tie."""
    log = open(path).read()
    val, test = {}, {}
    vre = re.compile(r"Val_Retrieval Epoch\s+(\d+) macroF1: ([\d.]+) macroP: ([\d.]+) "
                     r"macroR: ([\d.]+) acc: ([\d.]+) roc: ([\d.]+)")
    tre = re.compile(r"Test_Retrieval Epoch\s+(\d+) macroF1: ([\d.]+) macroP: ([\d.]+) "
                     r"macroR: ([\d.]+) acc: ([\d.]+) roc: ([\d.]+)")
    for m in vre.finditer(log):
        val[int(m.group(1))] = tuple(float(x) for x in m.groups()[1:])
    for m in tre.finditer(log):
        test[int(m.group(1))] = tuple(float(x) for x in m.groups()[1:])
    warm = [e for e in val if e >= warmup] or list(val)
    best = max(warm, key=lambda e: (val[e][3], val[e][4]))
    fe = max(test)
    return {"valsel_epoch": best, "valsel_acc": test[best][3], "valsel_mf1": test[best][0],
            "final_epoch": fe, "final_acc": test[fe][3], "final_mf1": test[fe][0]}


# ------------------------------------------------------------------ Cell object
class Cell(object):
    """One (dataset, seed, protocol) unit: gold labels + banked neighbour lists for
    test, and optionally for dev."""

    def __init__(self, dataset, seed, protocol, epoch, ids, y, nb_lab, nb_sim,
                 dev=None, kgrid=None, anchor=None, nature=""):
        self.dataset, self.seed, self.protocol, self.epoch = dataset, seed, protocol, epoch
        self.ids, self.y = ids, np.asarray(y)
        self.nb_lab, self.nb_sim = np.asarray(nb_lab), np.asarray(nb_sim)
        self.dev = dev          # dict(y=, nb_lab=, nb_sim=) or None
        self.kgrid = kgrid or K_GRID
        self.anchor = anchor
        self.nature = nature

    def curve(self, split="test"):
        out = {}
        if split == "test":
            y, lab, sim = self.y, self.nb_lab, self.nb_sim
        else:
            y = self.dev["y"]
            lab, sim = self.dev["nb_lab"], self.dev["nb_sim"]
        for k in self.kgrid:
            acc, mf1, p = score(y, vote_at_k(lab, sim, k))
            out[k] = {"acc": acc, "mf1": mf1, "pred": p}
        return out


# ------------------------------------------------------------------ loaders
def load_hatemm():
    """CPU-reconstructed PROXY (job 13241 head ckpts deleted, F78). Per-item top-20
    neighbour lists from the ERRPAT proxy run; test AND dev."""
    gt = {}
    for sp in ("train", "val", "test"):
        for line in open(HATEMM_GT.format(sp)):
            r = json.loads(line)
            gt[r["id"]] = int(r["label"])
    dirs = {}
    for s in (0, 1, 2):
        g = sorted(glob.glob(os.path.join(
            HATEMM_PROXY_ROOT, "Retrieval/HateMM/RAC_errpat_proxy", "*seed%d*" % s)))
        assert len(g) == 1, (s, g)
        dirs[s] = g[0]

    def read(path):
        ld = pickle.load(open(path, "rb"))["logging_dict"]
        ids, lab, sim = [], [], []
        for vid, e in ld.items():
            assert len(e["retrieved_label"]) == DEPLOYED_K, (path, vid)
            ids.append(vid)
            lab.append([int(x) for x in e["retrieved_label"]])
            sim.append([float(x) for x in e["retrieved_scores"]])
        return ids, np.array([gt[v] for v in ids]), np.array(lab), np.array(sim)

    cells = []
    for s in (0, 1, 2):
        tl = parse_trainlog(os.path.join(HATEMM_PROXY_ROOT, "proxy_s%d.trainlog" % s))
        for proto, ep in (("valsel", tl["valsel_epoch"]), ("final", tl["final_epoch"])):
            ids, y, lab, sim = read(os.path.join(
                dirs[s], "testepoch_%d_retrieval_logging_dict.pkl" % ep))
            d_ids, d_y, d_lab, d_sim = read(os.path.join(
                dirs[s], "devepoch_%d_retrieval_logging_dict.pkl" % ep))
            # gate the hardcoded anchor against the proxy trainlog itself
            a = ANCHOR_HATEMM[(s, proto)]
            assert (round(tl[proto + "_acc"], 4), round(tl[proto + "_mf1"], 4)) == a, (s, proto, tl, a)
            cells.append(Cell("HateMM", s, proto, ep, ids, y, lab, sim,
                              dev=dict(y=d_y, nb_lab=d_lab, nb_sim=d_sim),
                              kgrid=K_GRID, anchor=a,
                              nature="CPU proxy of job 13241 (floor ckpts deleted, F78)"))
    return cells


def load_zh():
    """CPU re-mint PROXY of job 13150 (floor ckpts deleted). Per-epoch dev+test dumps."""
    cells = []
    for s in (0, 1, 2):
        d = pickle.load(open(ZH_DUMP.format(s), "rb"))
        by = {(r["split"], r["epoch"]): r for r in d["records"]}
        dev_eps = sorted(e for (sp, e) in by if sp == "dev")
        warm = [e for e in dev_eps if e >= 5] or dev_eps
        # deployed rule: max by (dev acc, dev roc), earliest on tie
        valsel_ep = max(warm, key=lambda e: (by[("dev", e)]["acc"], by[("dev", e)]["roc"]))
        final_ep = max(e for (sp, e) in by if sp == "test")
        for proto, ep in (("valsel", valsel_ep), ("final", final_ep)):
            rt, rd = by[("test", ep)], by[("dev", ep)]
            assert int(rt["n_retrieved"].min()) == DEPLOYED_K, (s, ep)
            anchor = None
            if proto == "final":
                anchor = ANCHOR_ZH_FINAL[s]
                assert (round(rt["acc"], 4), round(rt["macroF1"], 4)) == anchor, (s, rt["acc"])
            else:
                anchor = (round(rt["acc"], 4), round(rt["macroF1"], 4))
            cells.append(Cell("MHC_zh", s, proto, ep, rt["ids"], rt["gold"],
                              rt["nb_lab"], rt["nb_sim"],
                              dev=dict(y=rd["gold"], nb_lab=rd["nb_lab"], nb_sim=rd["nb_sim"]),
                              kgrid=K_GRID, anchor=anchor,
                              nature="CPU re-mint proxy of job 13150 (floor ckpts deleted)"))
    return cells


def load_en_armv():
    """ARM-V: the deployed EN headline stack (frozen-Qwen -> align head -> archive-kNN
    alpha=0.25 key -> top-20 vote), val-selected, 4 seeds. Banked EXACT (not a proxy)
    with top-60 neighbour lists. NO dev neighbours were banked and the val-selected
    checkpoints are deleted, so ARM-V supports no dev read."""
    cells = []
    for s in range(4):
        d = json.load(open(os.path.join(EN_P2, "cache_MHC_s%d.json" % s)))
        ids = [x["id"] for x in d["samples"]]
        y = np.array([int(x["label"]) for x in d["samples"]])
        lab = np.array([[int(t[2]) for t in x["neighbors"]] for x in d["samples"]])
        sim = np.array([[float(t[1]) for t in x["neighbors"]] for x in d["samples"]])
        assert lab.shape[1] == 60, lab.shape
        # gate: our k=20 vote must reproduce the banked per-item floor_vote bit-exactly
        v20 = vote_at_k(lab, sim, DEPLOYED_K)
        banked = np.array([float(x["floor_vote"]) for x in d["samples"]])
        assert np.abs(v20 - banked).max() == 0.0, (s, np.abs(v20 - banked).max())
        cells.append(Cell("MHC_EN_ARM-V", s, "valsel", d["ckpt"], ids, y, lab, sim,
                          dev=None, kgrid=K_GRID_EXT, anchor=tuple(d["logged"]),
                          nature="banked EXACT (deployed headline stack, 4 seeds)"))
    return cells


def load_en_armf():
    """ARM-F: EN final-epoch (e29) no-archive-key floor, frozen-Qwen, 3 seeds.
    Recomputed from snapshotted heads -- validated bit-exact to 4 dp against the
    primary trainlogs (ERRPAT MHC-EN sec 0). Gives a dev read EN ARM-V cannot."""
    import faiss
    import torch
    import cross_channel_router_gate as R
    faiss.omp_set_num_threads(4)
    torch.set_num_threads(4)
    CACHE = os.path.join(REPO, "data", "CLIP_Embedding", "MHC")
    M = "Qwen2.5-VL-7B-Instruct_HF"

    def cache(split):
        d = torch.load(os.path.join(CACHE, "%s_%s.pt" % (split, M)), map_location="cpu")
        ids = d["ids"]
        ids = ids[0] if (isinstance(ids, list) and len(ids) == 1 and isinstance(ids[0], list)) else ids
        return list(ids), d["img_feats"].float(), d["text_feats"].float(), d["labels"].long().numpy()

    tr_ids, tr_img, tr_txt, tr_lab = cache("train")
    te_ids, te_img, te_txt, te_lab = cache("test_seen")
    dv_ids, dv_img, dv_txt, dv_lab = cache("dev_seen")
    depth = min(60, len(tr_ids))

    cells = []
    for s in (0, 1, 2):
        sd = torch.load(os.path.join(EN_CKPT, "MHC_Qwen_s%d_e29.pt" % s), map_location="cpu")
        m = R.build_head(sd)
        tr_e = R.embed(m, tr_img, tr_txt)
        ix_src = tr_e.copy()
        faiss.normalize_L2(ix_src)
        ix = faiss.IndexFlatIP(ix_src.shape[1])
        ix.add(ix_src)

        def nb(img, txt):
            q = R.embed(m, img, txt).copy()
            faiss.normalize_L2(q)
            D, I = ix.search(q, depth)
            return tr_lab[I].astype(np.int64), D.astype(np.float64)

        te_l, te_s = nb(te_img, te_txt)
        dv_l, dv_s = nb(dv_img, dv_txt)
        acc20, mf120, _ = score(te_lab, vote_at_k(te_l, te_s, DEPLOYED_K))
        a = ANCHOR_EN_ARMF[s]
        assert (round(acc20, 4), round(mf120, 4)) == a, (s, acc20, mf120, a)
        cells.append(Cell("MHC_EN_ARM-F", s, "final", 29, te_ids, te_lab, te_l, te_s,
                          dev=dict(y=dv_lab, nb_lab=dv_l, nb_sim=dv_s),
                          kgrid=K_GRID_EXT, anchor=a,
                          nature="recomputed from snapshotted e29 head, 4dp-validated vs trainlog"))
    return cells


# ------------------------------------------------------------------ analysis
def dev_pick(devcurve, kgrid):
    """Deployment-legal k choice: argmax dev accuracy. Tie rule declared UP FRONT and
    applied uniformly: keep the incumbent k=20 whenever it is among the dev-argmax
    set (a tie is not evidence to move); otherwise take the LARGEST tied k (closest
    to the incumbent). Returns (k_star, tied_ks, dev_acc)."""
    best = max(devcurve[k]["acc"] for k in kgrid)
    tied = [k for k in kgrid if devcurve[k]["acc"] == best]
    kstar = DEPLOYED_K if DEPLOYED_K in tied else max(tied)
    return kstar, tied, best


def main():
    OUT = {"meta": {
        "question": "does REDUCING the deployed top-20 kNN vote k help?",
        "status": "FORENSIC -- read-only replay of already-banked, already-consumed "
                  "test predictions. No GPU, no SLURM, no Modal, no retraining, "
                  "no new test inference.",
        "vote_formula": "v = sum_i (2*lab_i-1)*cos_i*w_i / sum_i w_i, w=[k..1]; pred = v>=0",
        "formula_source": "src/utils/metrics.py:228-231,262-301 (use_sim=True, arithmetic)",
        "equivalence_argument": "--topk enters src/run_rac.py only at lines 818/828/836/846 "
                                "(retrieval depth + vote) and the exp-name string at 1016; it "
                                "never touches the loss or the optimiser, so truncating banked "
                                "top-20 lists == re-running with --topk k at the same checkpoint.",
        "k_grid": K_GRID, "k_grid_extended": K_GRID_EXT, "deployed_k": DEPLOYED_K,
        "cpu_only": True,
    }, "parity_gate": {}, "curves": {}, "dev_deployment_read": {},
        "oracle_read_FORENSIC": {}, "flip_accounting_FORENSIC": {}}

    cells = []
    cells += load_hatemm()
    cells += load_zh()
    cells += load_en_armv()
    cells += load_en_armf()

    # ---------------- parity gate at k=20 (abort on any mismatch) ----------------
    store = {}
    for c in cells:
        cur = c.curve("test")
        key = "%s/seed%d/%s" % (c.dataset, c.seed, c.protocol)
        got = (round(cur[DEPLOYED_K]["acc"], 4), round(cur[DEPLOYED_K]["mf1"], 4))
        ok = got == tuple(round(x, 4) for x in c.anchor)
        OUT["parity_gate"][key] = {
            "epoch_or_ckpt": c.epoch, "nature": c.nature,
            "recorded_acc_mf1": [round(x, 4) for x in c.anchor],
            "replay_k20_acc_mf1": list(got), "bit_exact_4dp": bool(ok),
        }
        assert ok, ("PARITY GATE FAILED", key, c.anchor, got)
        store[key] = {"cell": c, "test": cur,
                      "dev": c.curve("dev") if c.dev is not None else None}

    # ---------------- curves ----------------
    for key, blk in store.items():
        c = blk["cell"]
        OUT["curves"][key] = {
            "kgrid": c.kgrid, "n_test": int(len(c.y)),
            "test": {str(k): {"acc": round(blk["test"][k]["acc"], 4),
                              "mf1": round(blk["test"][k]["mf1"], 4),
                              "d_acc_vs_k20": round(blk["test"][k]["acc"]
                                                    - blk["test"][DEPLOYED_K]["acc"], 4),
                              "d_mf1_vs_k20": round(blk["test"][k]["mf1"]
                                                    - blk["test"][DEPLOYED_K]["mf1"], 4)}
                     for k in c.kgrid},
            "dev": None if blk["dev"] is None else
                   {str(k): {"acc": round(blk["dev"][k]["acc"], 4),
                             "mf1": round(blk["dev"][k]["mf1"], 4)} for k in c.kgrid},
        }

    # ---------------- group means over seeds ----------------
    groups = {}
    for key, blk in store.items():
        c = blk["cell"]
        groups.setdefault("%s/%s" % (c.dataset, c.protocol), []).append(blk)
    OUT["group_means"] = {}
    for g, blks in groups.items():
        kg = blks[0]["cell"].kgrid
        n = len(blks)
        rows = {}
        for k in kg:
            accs = [b["test"][k]["acc"] for b in blks]
            mf1s = [b["test"][k]["mf1"] for b in blks]
            d = [b["test"][k]["acc"] - b["test"][DEPLOYED_K]["acc"] for b in blks]
            dm = [b["test"][k]["mf1"] - b["test"][DEPLOYED_K]["mf1"] for b in blks]
            rows[str(k)] = {
                "mean_acc": round(float(np.mean(accs)), 4),
                "mean_mf1": round(float(np.mean(mf1s)), 4),
                "d_acc_vs_k20": round(float(np.mean(d)), 4),
                "d_mf1_vs_k20": round(float(np.mean(dm)), 4),
                "per_seed_acc": [round(a, 4) for a in accs],
                "per_seed_d_acc": [round(x, 4) for x in d],
                "n_seeds_acc_positive": int(sum(1 for x in d if x > 0)),
                "n_seeds_acc_negative": int(sum(1 for x in d if x < 0)),
            }
        OUT["group_means"][g] = {"n_seeds": n, "kgrid": kg, "rows": rows}

    # ---------------- deployment-legal read: dev picks k ----------------
    for g, blks in groups.items():
        if blks[0]["dev"] is None:
            OUT["dev_deployment_read"][g] = {
                "available": False,
                "reason": "no dev neighbour lists banked for this arm and the val-selected "
                          "checkpoints are deleted -- a dev read is not reconstructible at $0",
            }
            continue
        kg = blks[0]["cell"].kgrid
        per = []
        for b in blks:
            kstar, tied, dacc = dev_pick(b["dev"], kg)
            per.append({
                "seed": b["cell"].seed, "dev_pick_k": kstar, "dev_tied_ks": tied,
                "dev_acc_at_pick": round(dacc, 4),
                "dev_acc_at_k20": round(b["dev"][DEPLOYED_K]["acc"], 4),
                "test_acc_at_pick": round(b["test"][kstar]["acc"], 4),
                "test_mf1_at_pick": round(b["test"][kstar]["mf1"], 4),
                "test_acc_at_k20": round(b["test"][DEPLOYED_K]["acc"], 4),
                "test_mf1_at_k20": round(b["test"][DEPLOYED_K]["mf1"], 4),
                "d_test_acc": round(b["test"][kstar]["acc"] - b["test"][DEPLOYED_K]["acc"], 4),
                "d_test_mf1": round(b["test"][kstar]["mf1"] - b["test"][DEPLOYED_K]["mf1"], 4),
            })
        # POOLED variant: k is a config-level hyper-parameter, normally set ONCE and
        # shared across seeds. Pool the dev items of all seeds, take one argmax.
        pooled = {k: float(np.mean(np.concatenate(
            [(b["dev"][k]["pred"] == b["cell"].dev["y"]) for b in blks]))) for k in kg}
        bestp = max(pooled.values())
        tiedp = [k for k in kg if pooled[k] == bestp]
        kpool = DEPLOYED_K if DEPLOYED_K in tiedp else max(tiedp)
        pool_test_acc = [b["test"][kpool]["acc"] for b in blks]
        pool_test_mf1 = [b["test"][kpool]["mf1"] for b in blks]
        base_acc = [b["test"][DEPLOYED_K]["acc"] for b in blks]
        base_mf1 = [b["test"][DEPLOYED_K]["mf1"] for b in blks]
        OUT["dev_deployment_read"][g] = {
            "available": True,
            "tie_rule": "argmax dev acc; keep incumbent k=20 if tied at the max, else largest tied k",
            "per_seed": per,
            "mean_d_test_acc": round(float(np.mean([p["d_test_acc"] for p in per])), 4),
            "mean_d_test_mf1": round(float(np.mean([p["d_test_mf1"] for p in per])), 4),
            "n_seeds_dev_moved_off_k20": int(sum(1 for p in per if p["dev_pick_k"] != DEPLOYED_K)),
            "pooled_dev_pick": {
                "k": kpool, "tied_ks": tiedp,
                "pooled_dev_acc_at_pick": round(bestp, 4),
                "pooled_dev_acc_at_k20": round(pooled[DEPLOYED_K], 4),
                "mean_test_acc": round(float(np.mean(pool_test_acc)), 4),
                "mean_test_mf1": round(float(np.mean(pool_test_mf1)), 4),
                "d_test_acc": round(float(np.mean(pool_test_acc)) - float(np.mean(base_acc)), 4),
                "d_test_mf1": round(float(np.mean(pool_test_mf1)) - float(np.mean(base_mf1)), 4),
                "per_seed_d_acc": [round(a - b, 4) for a, b in zip(pool_test_acc, base_acc)],
            },
        }

    # ---------------- oracle read (FORENSIC, not deployable) ----------------
    for g, blks in groups.items():
        kg = blks[0]["cell"].kgrid
        per = []
        for b in blks:
            best = max(b["test"][k]["acc"] for k in kg)
            tied = [k for k in kg if b["test"][k]["acc"] == best]
            kstar = DEPLOYED_K if DEPLOYED_K in tied else max(tied)
            per.append({"seed": b["cell"].seed, "oracle_k": kstar, "oracle_tied_ks": tied,
                        "test_acc": round(best, 4),
                        "test_mf1": round(b["test"][kstar]["mf1"], 4),
                        "d_acc_vs_k20": round(best - b["test"][DEPLOYED_K]["acc"], 4)})
        # shared-k oracle: one k for all seeds, chosen on 3-seed-mean test acc
        means = {k: float(np.mean([b["test"][k]["acc"] for b in blks])) for k in kg}
        bestm = max(means.values())
        tiedm = [k for k in kg if means[k] == bestm]
        kshared = DEPLOYED_K if DEPLOYED_K in tiedm else max(tiedm)
        OUT["oracle_read_FORENSIC"][g] = {
            "per_seed": per,
            "mean_d_acc_per_seed_oracle": round(
                float(np.mean([p["d_acc_vs_k20"] for p in per])), 4),
            "shared_oracle_k": kshared,
            "shared_oracle_mean_acc": round(bestm, 4),
            "shared_oracle_d_acc": round(bestm - means[DEPLOYED_K], 4),
        }

    # ---------------- flip accounting at the shared oracle k ----------------
    for g, blks in groups.items():
        ko = OUT["oracle_read_FORENSIC"][g]["shared_oracle_k"]
        n_seeds = len(blks)
        ids = list(blks[0]["cell"].ids)
        y = blks[0]["cell"].y
        # stable core = wrong at k=20 in EVERY seed
        wrong20 = np.zeros(len(ids), dtype=int)
        wrongko = np.zeros(len(ids), dtype=int)
        for b in blks:
            assert list(b["cell"].ids) == ids, g
            wrong20 += (b["test"][DEPLOYED_K]["pred"] != y).astype(int)
            wrongko += (b["test"][ko]["pred"] != y).astype(int)
        core = [ids[i] for i in range(len(ids)) if wrong20[i] == n_seeds]
        core_idx = [i for i in range(len(ids)) if wrong20[i] == n_seeds]
        core_fixed_all = [ids[i] for i in core_idx if wrongko[i] == 0]
        core_fixed_any = [ids[i] for i in core_idx if wrongko[i] < n_seeds]
        # per-seed fixed/broken tallies
        fixed, broken = [], []
        for b in blks:
            p20 = b["test"][DEPLOYED_K]["pred"]
            pko = b["test"][ko]["pred"]
            fixed.append(int(np.sum((p20 != y) & (pko == y))))
            broken.append(int(np.sum((p20 == y) & (pko != y))))
        OUT["flip_accounting_FORENSIC"][g] = {
            "oracle_k": ko, "n_test": len(ids), "n_seeds": n_seeds,
            "stable_core_wrong_all_seeds_at_k20": len(core),
            "stable_core_ids": core,
            "core_fixed_in_all_seeds_at_oracle_k": core_fixed_all,
            "core_fixed_in_at_least_one_seed": core_fixed_any,
            "per_seed_items_fixed": fixed,
            "per_seed_items_broken": broken,
            "net_per_seed": [f - br for f, br in zip(fixed, broken)],
        }

    # ---------------- mechanism: why the low-k end is flat ----------------
    # With descending cosines s_0 >= s_1 >= ... and weights [k..1], the rank-1 term
    # k*s_0 dominates the rest for k <= 3 (3*s_0 >= 2*s_1 + 1*s_2 always holds), so
    # k in {1,2,3} is algebraically a plain 1-NN classifier. Verified per cell.
    OUT["mechanism"] = {"k_le_3_is_1nn": {}, "items_changed_vs_k20": {},
                        "one_item_in_acc": {}}
    for key, blk in store.items():
        c = blk["cell"]
        top1 = (c.nb_lab[:, 0] == 1).astype(int)
        same = {str(k): bool(np.array_equal(blk["test"][k]["pred"], top1))
                for k in (1, 2, 3) if k in c.kgrid}
        OUT["mechanism"]["k_le_3_is_1nn"][key] = same
        OUT["mechanism"]["items_changed_vs_k20"][key] = {
            str(k): int(np.sum(blk["test"][k]["pred"] != blk["test"][DEPLOYED_K]["pred"]))
            for k in c.kgrid}
        OUT["mechanism"]["one_item_in_acc"][key] = round(1.0 / len(c.y), 4)

    json.dump(OUT, open(OUT_PATH, "w"), indent=1)
    print("wrote", OUT_PATH)

    # ------- console summary -------
    print("\n=== PARITY GATE (k=20 vs recorded) ===")
    for k, v in OUT["parity_gate"].items():
        print("  %-30s recorded %s  replay %s  %s"
              % (k, v["recorded_acc_mf1"], v["replay_k20_acc_mf1"],
                 "OK" if v["bit_exact_4dp"] else "FAIL"))
    print("\n=== 3/4-SEED MEAN TEST acc / mF1 BY k ===")
    for g, blk in OUT["group_means"].items():
        print("\n%s (n_seeds=%d)" % (g, blk["n_seeds"]))
        print("   k |    acc |    mF1 |   dAcc |   dmF1 | seed signs (+/-)")
        for k in blk["kgrid"]:
            r = blk["rows"][str(k)]
            print("  %2d | %.4f | %.4f | %+.4f | %+.4f | %d+/%d-"
                  % (k, r["mean_acc"], r["mean_mf1"], r["d_acc_vs_k20"],
                     r["d_mf1_vs_k20"], r["n_seeds_acc_positive"], r["n_seeds_acc_negative"]))
    print("\n=== DEV-PICKED-k (deployment-legal) ===")
    for g, v in OUT["dev_deployment_read"].items():
        if not v["available"]:
            print("  %-24s UNAVAILABLE: %s" % (g, v["reason"]))
            continue
        print("  %-24s mean dTest acc %+.4f / mF1 %+.4f ; picks %s"
              % (g, v["mean_d_test_acc"], v["mean_d_test_mf1"],
                 [p["dev_pick_k"] for p in v["per_seed"]]))
    print("\n=== ORACLE-k (FORENSIC, not deployable) ===")
    for g, v in OUT["oracle_read_FORENSIC"].items():
        print("  %-24s shared k=%-2d mean acc %.4f (%+.4f) ; per-seed k %s mean %+.4f"
              % (g, v["shared_oracle_k"], v["shared_oracle_mean_acc"], v["shared_oracle_d_acc"],
                 [p["oracle_k"] for p in v["per_seed"]], v["mean_d_acc_per_seed_oracle"]))
    print("\n=== FLIP ACCOUNTING at shared oracle k ===")
    for g, v in OUT["flip_accounting_FORENSIC"].items():
        print("  %-24s k=%-2d core=%d fixed_all_seeds=%d fixed_any=%d | per-seed fixed %s broken %s net %s"
              % (g, v["oracle_k"], v["stable_core_wrong_all_seeds_at_k20"],
                 len(v["core_fixed_in_all_seeds_at_oracle_k"]),
                 len(v["core_fixed_in_at_least_one_seed"]),
                 v["per_seed_items_fixed"], v["per_seed_items_broken"], v["net_per_seed"]))


if __name__ == "__main__":
    main()
