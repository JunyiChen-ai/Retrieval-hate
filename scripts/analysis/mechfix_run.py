#!/usr/bin/env python
"""
mechfix_run.py -- harness for the MECHFIX $0 pregate.

Paired same-head design: for each dataset x seed x protocol we load THE SAME head the
errpat analysis used, recompute train-bank + query embeddings from the banked feature
caches, and evaluate the deployed vote and every treatment arm from those same
embeddings. The claim object is  D(arm) = metric(arm) - metric(deployed)  per seed.

Gate order (enforced by construction):
  1. floor parity -- deployed-vote reproduction must equal the recorded primary value
     at 4 dp; hard assert, aborts the run.
  2. train-side sanity per arm (train items only, LOO; recorded, never used to tune).
  3. test reads for the 5 frozen arms; dev reads as free same-head corroboration.
  4. flip/break accounting against the errpat stable-core id lists.

CPU ONLY. No GPU, no SLURM, no Modal, no training.
"""
import argparse
import csv
import glob
import json
import os
import pickle
import re
import sys

import numpy as np
import torch

REPO = "/data/jehc223/RGCL"
sys.path.insert(0, os.path.join(REPO, "src"))
sys.path.insert(0, os.path.join(REPO, "scripts", "analysis"))

import faiss  # noqa: E402
import mechfix_ops as M  # noqa: E402

SCRATCH = ("/data/jehc223/home/tmp/claude-135258174/-data-jehc223-RGCL/"
           "e8f03e41-3e21-4cea-b12c-29207373bfca/scratchpad/errpat")

ARMS = ["T1", "T2a", "T2b", "T3", "T4"]


# ------------------------------------------------------------------ trainlog parse
_TRE = re.compile(r"Test_Retrieval Epoch\s+(\d+) macroF1: ([\d.]+) macroP: ([\d.]+) "
                  r"macroR: ([\d.]+) acc: ([\d.]+) roc: ([\d.]+)")
_VRE = re.compile(r"Val_Retrieval Epoch\s+(\d+) macroF1: ([\d.]+) macroP: ([\d.]+) "
                  r"macroR: ([\d.]+) acc: ([\d.]+) roc: ([\d.]+)")


def parse_curves(path):
    """-> dict split -> epoch -> (acc, mf1, roc), straight off a primary/proxy trainlog."""
    log = open(path).read()
    out = {"test": {}, "dev": {}}
    for rex, key in ((_TRE, "test"), (_VRE, "dev")):
        for m in rex.finditer(log):
            e = int(m.group(1))
            out[key][e] = (float(m.group(5)), float(m.group(2)), float(m.group(6)))
    return out


def valsel_epoch(curves, warmup=5):
    dev = curves["dev"]
    warm = [e for e in dev if e >= warmup] or list(dev)
    return max(warm, key=lambda e: (dev[e][0], dev[e][2]))


# ------------------------------------------------------------------- feature caches
def load_cache(cache_dir, split, model):
    d = torch.load(os.path.join(cache_dir, f"{split}_{model}.pt"),
                   map_location="cpu", weights_only=False)
    ids = d["ids"]
    if isinstance(ids, list) and len(ids) == 1 and isinstance(ids[0], list):
        ids = ids[0]
    return (list(ids), d["img_feats"].float(), d["text_feats"].float(),
            np.asarray(d["labels"]).astype(int))


def load_gt_text(gt_dir):
    out = {}
    for sp in ("train", "val", "test"):
        for line in open(os.path.join(gt_dir, f"{sp}.jsonl")):
            r = json.loads(line)
            out[r["id"]] = r["text"]
    return out


def volume_scalar(ids, gt_text, mode):
    """transcript volume per id: whitespace tokens (EN/HateMM) or characters (ZH)."""
    vals = []
    for i in ids:
        t = gt_text[i]
        vals.append(len(t.split()) if mode == "words" else len(t))
    return np.asarray(vals, dtype="float64")


# ------------------------------------------------------------------------- the head
def build_head(sd):
    from model.classifier import classifier_hateClipper

    class A:
        dataset = "X"
        mod_dropout = False
        mod_dropout_p = 0.3

    m = classifier_hateClipper(sd["img_proj.0.weight"].shape[1],
                               sd["text_proj.0.weight"].shape[1],
                               num_layers=3, proj_dim=1024, map_dim=1024,
                               fusion_mode="align", dropout=[0.2, 0.4, 0.1],
                               batch_norm=False, args=A())
    m.load_state_dict(sd)
    m.eval()
    return m


@torch.no_grad()
def embed(m, img, txt):
    _, e = m(img, txt, return_embed=True)
    return e.numpy().astype("float32")


def load_sd(path):
    sd = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(sd, dict) or "img_proj.0.weight" not in sd:
        sd = sd.state_dict() if hasattr(sd, "state_dict") else sd
    return sd


# ------------------------------------------------------------------ dataset configs
def cfg_hatemm():
    model = "Qwen2.5-VL-7B-Instruct-LoRA-curric_HF"
    dirs = {}
    for s in (0, 1, 2):
        g = glob.glob(os.path.join(SCRATCH, "Retrieval/HateMM/RAC_errpat_proxy", f"*seed{s}*"))
        assert len(g) == 1, (s, g)
        dirs[s] = g[0]
    curves = {s: parse_curves(os.path.join(SCRATCH, f"proxy_s{s}.trainlog")) for s in (0, 1, 2)}
    vs = {s: valsel_epoch(curves[s]) for s in (0, 1, 2)}
    assert vs == {0: 25, 1: 15, 2: 29}, ("HateMM proxy val-sel epochs moved", vs)
    return dict(
        key="hatemm", ds="HateMM", model=model,
        cache_dir=os.path.join(REPO, "data/CLIP_Embedding/HateMM"),
        gt_dir=os.path.join(REPO, "data/gt/HateMM"),
        splits=("train", "dev_seen", "test_seen"), vol="words",
        protocols={"valsel": vs, "final": {s: 29 for s in (0, 1, 2)}},
        curves=curves, ckpt_dirs=dirs,
        parity_note="errpat CPU proxy heads; anchors = proxy trainlogs (ERRPAT-HateMM 0.2)",
    )


def cfg_zh():
    model = "Qwen2.5-VL-7B-Instruct-LoRA_HF"
    dirs, dumps = {}, {}
    for s in (0, 1, 2):
        g = glob.glob(os.path.join(REPO, "logging/Retrieval/MHC_zh/errpat_zh_remint_v2",
                                   f"*seed{s}*"))
        assert len(g) == 1, (s, g)
        dirs[s] = g[0]
        d = pickle.load(open(os.path.join(
            REPO, f"scripts/analysis/errpat_remint_dumps/errpat_zh_remint_seed{s}.pkl"), "rb"))
        dumps[s] = d["records"]
    curves = {}
    for s in (0, 1, 2):
        c = {"test": {}, "dev": {}}
        for r in dumps[s]:
            c[r["split"]][int(r["epoch"])] = (round(float(r["acc"]), 4),
                                              round(float(r["macroF1"]), 4),
                                              round(float(r["roc"]), 4))
        curves[s] = c
    vs = {s: valsel_epoch(curves[s]) for s in (0, 1, 2)}
    return dict(
        key="zh", ds="MHC_zh", model=model,
        cache_dir=os.path.join(REPO, "data/CLIP_Embedding/MHC_zh"),
        gt_dir=os.path.join(REPO, "data/gt/MHC_zh"),
        splits=("train", "dev_seen", "test_seen"), vol="chars",
        protocols={"final": {s: 29 for s in (0, 1, 2)}, "valsel": vs},
        curves=curves, ckpt_dirs=dirs,
        parity_note="errpat CPU re-mint heads; anchors = re-mint dumps (ERRPAT-ZH 0.2)",
    )


def cfg_en():
    model = "Qwen2.5-VL-7B-Instruct_HF"
    logs = {
        0: os.path.join(REPO, "slurm/logs/enc3s_MHC_Qwen2.5-VL-7B-Instruct_HF_seed0_12850.trainlog"),
        1: os.path.join(REPO, "slurm/logs/arcbase_MHC_Qwen2.5-VL-7B-Instruct_HF_seed1_12275.trainlog"),
        2: os.path.join(REPO, "slurm/logs/arcbase_MHC_Qwen2.5-VL-7B-Instruct_HF_seed2_12276.trainlog"),
    }
    curves = {s: parse_curves(p) for s, p in logs.items()}
    ck = {s: os.path.join(REPO, f"refine-logs/router_ckpt_snapshot/MHC_Qwen_s{s}_e29.pt")
          for s in (0, 1, 2)}
    for s in ck:
        assert os.path.exists(ck[s]), ck[s]
    return dict(
        key="en", ds="MHC", model=model,
        cache_dir=os.path.join(REPO, "data/CLIP_Embedding/MHC"),
        gt_dir=os.path.join(REPO, "data/gt/MHC"),
        splits=("train", "dev_seen", "test_seen"), vol="words",
        protocols={"final": {s: 29 for s in (0, 1, 2)}},
        curves=curves, ckpt_files=ck, trainlogs=logs,
        parity_note="ARM-F snapshot heads; anchors = primary trainlogs (ERRPAT-EN 2.2)",
    )


CFG = {"hatemm": cfg_hatemm, "zh": cfg_zh, "en": cfg_en}


# ------------------------------------------------------------------ core error lists
def core_ids(key):
    """Documented stable-core error id lists from the errpat machine outputs."""
    out = {}
    if key == "hatemm":
        j = json.load(open(os.path.join(REPO, "scripts/analysis/errpat_hatemm_forensics_OUT.json")))
        for proto in ("valsel", "final"):
            out[f"{proto}_3of3"] = list(j["seed_stability"][proto]["wrong_3of3_ids"])
        rows = list(csv.DictReader(open(os.path.join(
            REPO, "scripts/analysis/errpat_hatemm_peritem.csv"))))
        for proto in ("valsel", "final"):
            ge2 = [r["id"] for r in rows
                   if sum(int(r[f"s{s}_{proto}_err"]) for s in (0, 1, 2)) >= 2]
            out[f"{proto}_ge2of3"] = ge2
    elif key == "zh":
        j = json.load(open(os.path.join(REPO, "scripts/analysis/errpat_zh_taxonomy_OUT.json")))
        out["final_3of3"] = [k for k, v in j["per_item"].items() if int(v["n_seeds_wrong"]) == 3]
        out["valsel_3of3"] = out["final_3of3"]  # only the final-epoch taxonomy exists (ERRPAT-ZH 0.2)
    elif key == "en":
        j = json.load(open(os.path.join(REPO, "scripts/analysis/errpat_mhc_en_out.json")))
        out["final_armv_4of4"] = list(j["consensus_error_ids"])
    return out


# ------------------------------------------------------------------------ evaluation
def eval_all_arms(bank_keys, bank_lab, q_keys, q_lab, r_hub, whit, vhat):
    """deployed + 5 treatments on one (bank, query) pair. Returns per-arm dict."""
    res = {}
    v, p, I, S = M.deployed_vote(bank_keys, bank_lab, q_keys)
    res["deployed"] = dict(pred=p, acc=M.acc(q_lab, p), mf1=M.macro_f1(q_lab, p),
                           top1sim=float(S[:, 0].mean()),
                           pos_rate=float(p.mean()))
    mg, p1, _, _ = M.t1_class_balanced(bank_keys, bank_lab, q_keys)
    res["T1"] = dict(pred=p1, acc=M.acc(q_lab, p1), mf1=M.macro_f1(q_lab, p1),
                     pos_rate=float(p1.mean()))
    v2, p2, I2, S2 = M.t2a_csls(bank_keys, bank_lab, q_keys, r_hub)
    res["T2a"] = dict(pred=p2, acc=M.acc(q_lab, p2), mf1=M.macro_f1(q_lab, p2),
                      pos_rate=float(p2.mean()),
                      nb_overlap=float(np.mean([len(set(I[i]) & set(I2[i])) / M.TOPK
                                                for i in range(len(q_lab))])))
    mu, W = whit
    bw = M.apply_whitener(bank_keys, mu, W)
    qw = M.apply_whitener(q_keys, mu, W)
    v3, p3, I3, S3 = M.deployed_vote(bw, bank_lab, qw)
    res["T2b"] = dict(pred=p3, acc=M.acc(q_lab, p3), mf1=M.macro_f1(q_lab, p3),
                      pos_rate=float(p3.mean()), top1sim=float(S3[:, 0].mean()),
                      nb_overlap=float(np.mean([len(set(I[i]) & set(I3[i])) / M.TOPK
                                                for i in range(len(q_lab))])))
    bp = M.remove_direction(bank_keys, vhat)
    qp = M.remove_direction(q_keys, vhat)
    v4, p4, I4, S4 = M.deployed_vote(bp, bank_lab, qp)
    res["T3"] = dict(pred=p4, acc=M.acc(q_lab, p4), mf1=M.macro_f1(q_lab, p4),
                     pos_rate=float(p4.mean()), top1sim=float(S4[:, 0].mean()),
                     nb_overlap=float(np.mean([len(set(I[i]) & set(I4[i])) / M.TOPK
                                               for i in range(len(q_lab))])))
    mg5, p5, _, _ = M.t1_class_balanced(bw, bank_lab, qw)
    res["T4"] = dict(pred=p5, acc=M.acc(q_lab, p5), mf1=M.macro_f1(q_lab, p5),
                     pos_rate=float(p5.mean()))
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=list(CFG))
    ap.add_argument("--threads", type=int, default=8)
    args = ap.parse_args()

    torch.set_num_threads(args.threads)
    faiss.omp_set_num_threads(args.threads)
    assert not torch.cuda.is_available() or os.environ.get("CUDA_VISIBLE_DEVICES", "") == "", \
        "CPU-only pregate: set CUDA_VISIBLE_DEVICES=''"

    c = CFG[args.dataset]()
    OUT = {"meta": {"dataset": c["ds"], "key": c["key"], "model": c["model"],
                    "cpu_only": True, "gpu_jobs": 0, "training": 0,
                    "ops_file": os.path.join(REPO, "scripts/analysis/mechfix_ops.py"),
                    "ops_sha256": None, "topk": M.TOPK,
                    "t1_k_per_class": M.T1_K_PER_CLASS, "t2a_hub_k": M.T2A_HUB_K,
                    "parity_note": c["parity_note"], "protocols": {}}}
    import hashlib
    OUT["meta"]["ops_sha256"] = hashlib.sha256(
        open(OUT["meta"]["ops_file"], "rb").read()).hexdigest()
    for p, m in c["protocols"].items():
        OUT["meta"]["protocols"][p] = {f"seed{s}": int(e) for s, e in m.items()}

    # ---------------- features
    tr_ids, tr_img, tr_txt, tr_lab = load_cache(c["cache_dir"], c["splits"][0], c["model"])
    dv_ids, dv_img, dv_txt, dv_lab = load_cache(c["cache_dir"], c["splits"][1], c["model"])
    te_ids, te_img, te_txt, te_lab = load_cache(c["cache_dir"], c["splits"][2], c["model"])
    gt_text = load_gt_text(c["gt_dir"])
    for nm, ids in (("train", tr_ids), ("dev", dv_ids), ("test", te_ids)):
        miss = [i for i in ids if i not in gt_text]
        assert not miss, (nm, miss[:5])
    tr_vol = volume_scalar(tr_ids, gt_text, c["vol"])
    OUT["meta"]["n"] = {"train": len(tr_ids), "dev": len(dv_ids), "test": len(te_ids)}
    OUT["meta"]["test_pos"] = int(te_lab.sum())
    OUT["meta"]["train_pos_rate"] = round(float(tr_lab.mean()), 4)
    OUT["meta"]["vol_mode"] = c["vol"]

    cores = core_ids(c["key"])
    OUT["core_error_lists"] = {k: len(v) for k, v in cores.items()}

    # ---------------- per seed x protocol
    per = {}
    parity = {}
    sanity = {}
    for proto, epmap in c["protocols"].items():
        for s in (0, 1, 2):
            ep = epmap[s]
            if "ckpt_dirs" in c:
                cands = (glob.glob(os.path.join(glob.escape(c["ckpt_dirs"][s]), "ckpt",
                                                f"epoch_model_{ep}_*.pt")) or
                         glob.glob(os.path.join(glob.escape(c["ckpt_dirs"][s]), "ckpt",
                                                f"best_model_{ep}_*.pt")))
                assert cands, (proto, s, ep)
                ckpt = sorted(cands)[0]
            else:
                ckpt = c["ckpt_files"][s]
            m = build_head(load_sd(ckpt))
            E = {"train": embed(m, tr_img, tr_txt),
                 "dev": embed(m, dv_img, dv_txt),
                 "test": embed(m, te_img, te_txt)}

            # ---- one-time per-head fits, all TRAIN-ONLY
            r_hub = M.bank_hubness(E["train"])
            mu, W, shrink, ev = M.fit_whitener(E["train"])
            vhat = M.fit_length_direction(E["train"], np.log1p(tr_vol))

            # ---- GATE 2: floor parity on the deployed vote
            res_te = eval_all_arms(E["train"], tr_lab, E["test"], te_lab,
                                   r_hub, (mu, W), vhat)
            a_acc, a_mf1, _ = c["curves"][s]["test"][ep]
            got = (round(res_te["deployed"]["acc"], 4), round(res_te["deployed"]["mf1"], 4))
            parity[f"{proto}_s{s}"] = {"epoch": int(ep), "ckpt": ckpt,
                                       "anchor_acc": a_acc, "anchor_mf1": a_mf1,
                                       "recomputed_acc": got[0], "recomputed_mf1": got[1],
                                       "PASS": bool(got == (a_acc, a_mf1))}
            assert got == (a_acc, a_mf1), \
                f"FLOOR PARITY FAIL {c['ds']} {proto} s{s} ep{ep}: got {got} want {(a_acc, a_mf1)}"

            res_dv = eval_all_arms(E["train"], tr_lab, E["dev"], dv_lab,
                                   r_hub, (mu, W), vhat)
            if ep in c["curves"][s]["dev"]:
                d_acc, d_mf1, _ = c["curves"][s]["dev"][ep]
                parity[f"{proto}_s{s}"]["dev_anchor_acc"] = d_acc
                parity[f"{proto}_s{s}"]["dev_recomputed_acc"] = round(res_dv["deployed"]["acc"], 4)
                parity[f"{proto}_s{s}"]["dev_PASS"] = bool(
                    round(res_dv["deployed"]["acc"], 4) == d_acc
                    and round(res_dv["deployed"]["mf1"], 4) == d_mf1)

            # ---- GATE 3: TRAIN-side sanity (train items only, LOO, no test)
            sn = {}
            vl, pl, Il, Sl = M.deployed_vote(E["train"], tr_lab, E["train"], exclude_self=True)
            sn["deployed_loo_train_acc"] = round(M.acc(tr_lab, pl), 4)
            sn["deployed_loo_pos_rate"] = round(float(pl.mean()), 4)
            sn["deployed_train_top1sim_mean"] = round(float(Sl[:, 0].mean()), 6)
            # T1: does the class-balanced rule collapse to one class on train (LOO)?
            b32 = M._norm32(E["train"])
            w10 = M._rank_weights(M.T1_K_PER_CLASS)
            sc = {}
            for cl in (0, 1):
                idx = np.flatnonzero(tr_lab == cl)
                D, I = M._flat_ip(np.ascontiguousarray(b32[idx]), b32, M.T1_K_PER_CLASS + 1)
                keep = np.empty((len(b32), M.T1_K_PER_CLASS))
                for i in range(len(b32)):
                    mm = idx[I[i]] != i
                    keep[i] = D[i][mm][:M.T1_K_PER_CLASS]
                sc[cl] = (keep * w10).sum(1) / w10.sum()
            p1l = ((sc[1] - sc[0]) >= 0).astype(int)
            sn["T1_loo_train_acc"] = round(M.acc(tr_lab, p1l), 4)
            sn["T1_loo_pos_rate"] = round(float(p1l.mean()), 4)
            sn["T1_collapsed"] = bool(p1l.mean() in (0.0, 1.0))
            sn["T1_margin_abs_median"] = round(float(np.median(np.abs(sc[1] - sc[0]))), 8)
            # T2a: r(x) spread
            sn["T2a_r_min"] = round(float(r_hub.min()), 6)
            sn["T2a_r_median"] = round(float(np.median(r_hub)), 6)
            sn["T2a_r_max"] = round(float(r_hub.max()), 6)
            sn["T2a_r_std"] = round(float(r_hub.std()), 6)
            sn["T2a_r_iqr"] = round(float(np.percentile(r_hub, 75) - np.percentile(r_hub, 25)), 6)
            # T2b: shrinkage + de-collapse check on train keys
            sn["T2b_lw_shrinkage"] = round(float(shrink), 6)
            sn["T2b_eig_min"] = float(f"{ev.min():.6e}")
            sn["T2b_eig_max"] = float(f"{ev.max():.6e}")
            bw = M.apply_whitener(E["train"], mu, W)
            _, plw, _, Slw = M.deployed_vote(bw, tr_lab, bw, exclude_self=True)
            sn["T2b_train_top1sim_mean"] = round(float(Slw[:, 0].mean()), 6)
            sn["T2b_loo_train_acc"] = round(M.acc(tr_lab, plw), 4)
            # T3: does the direction encode length on train?
            proj = M._norm32(E["train"]) @ vhat.astype("float32")
            ll = np.log1p(tr_vol)
            from scipy.stats import pearsonr, spearmanr
            pr = pearsonr(proj, ll); sr = spearmanr(proj, ll)
            sn["T3_pearson_proj_loglen"] = round(float(pr[0]), 4)
            sn["T3_pearson_p"] = float(f"{pr[1]:.3e}")
            sn["T3_spearman_proj_loglen"] = round(float(sr[0]), 4)
            sn["T3_spearman_p"] = float(f"{sr[1]:.3e}")
            bp = M.remove_direction(E["train"], vhat)
            projr = M._norm32(bp) @ vhat.astype("float32")
            sn["T3_residual_abs_proj_max"] = float(f"{np.abs(projr).max():.3e}")
            sn["T3_pearson_after_removal"] = round(float(pearsonr(projr, ll)[0]), 4) \
                if np.std(projr) > 0 else 0.0
            sanity[f"{proto}_s{s}"] = sn

            # ---- results + flip accounting
            cell = {"epoch": int(ep),
                    "test": {}, "dev": {}}
            dep_pred = res_te["deployed"]["pred"]
            dep_err = {te_ids[i] for i in range(len(te_ids)) if dep_pred[i] != te_lab[i]}
            cell["n_deployed_err_test"] = len(dep_err)
            for arm in ["deployed"] + ARMS:
                rr = res_te[arm]
                d = {"acc": round(rr["acc"], 4), "mf1": round(rr["mf1"], 4),
                     "pos_rate": round(rr["pos_rate"], 4)}
                if arm != "deployed":
                    d["d_acc"] = round(rr["acc"] - res_te["deployed"]["acc"], 4)
                    d["d_mf1"] = round(rr["mf1"] - res_te["deployed"]["mf1"], 4)
                    ap_ = rr["pred"]
                    arm_err = {te_ids[i] for i in range(len(te_ids)) if ap_[i] != te_lab[i]}
                    fixed = sorted(dep_err - arm_err)
                    broken = sorted(arm_err - dep_err)
                    d["n_fixed"] = len(fixed)
                    d["n_broken"] = len(broken)
                    d["fixed_ids"] = fixed
                    d["broken_ids"] = broken
                    for cname, cids in cores.items():
                        if not cname.startswith(proto) and c["key"] != "en":
                            continue
                        S_ = set(cids)
                        d[f"core[{cname}]_n"] = len(S_)
                        d[f"core[{cname}]_fixed"] = len(S_ & set(fixed))
                        d[f"core[{cname}]_still_wrong"] = len(S_ & arm_err)
                for k in ("top1sim", "nb_overlap"):
                    if k in rr:
                        d[k] = round(rr[k], 6)
                cell["test"][arm] = d
                rd = res_dv[arm]
                dd = {"acc": round(rd["acc"], 4), "mf1": round(rd["mf1"], 4)}
                if arm != "deployed":
                    dd["d_acc"] = round(rd["acc"] - res_dv["deployed"]["acc"], 4)
                    dd["d_mf1"] = round(rd["mf1"] - res_dv["deployed"]["mf1"], 4)
                cell["dev"][arm] = dd
            per[f"{proto}_s{s}"] = cell
            print(f"[{c['ds']}] {proto} s{s} ep{ep}  floor {got[0]:.4f}/{got[1]:.4f} PARITY OK  "
                  + "  ".join(f"{a}:{cell['test'][a]['d_acc']:+.4f}/"
                              f"{cell['test'][a]['d_mf1']:+.4f}" for a in ARMS), flush=True)

    OUT["floor_parity"] = parity
    OUT["train_side_sanity"] = sanity
    OUT["cells"] = per

    # ---------------- 3-seed means
    means = {}
    for proto in c["protocols"]:
        mm = {}
        for arm in ARMS:
            for split in ("test", "dev"):
                da = [per[f"{proto}_s{s}"][split][arm]["d_acc"] for s in (0, 1, 2)]
                df = [per[f"{proto}_s{s}"][split][arm]["d_mf1"] for s in (0, 1, 2)]
                mm[f"{arm}_{split}"] = {
                    "mean_d_acc": round(float(np.mean(da)), 4),
                    "mean_d_mf1": round(float(np.mean(df)), 4),
                    "per_seed_d_acc": da, "per_seed_d_mf1": df,
                    "sign_acc": "".join("+" if x > 0 else ("-" if x < 0 else "0") for x in da),
                    "sign_mf1": "".join("+" if x > 0 else ("-" if x < 0 else "0") for x in df),
                    "n_pos_acc": int(sum(1 for x in da if x > 0)),
                    "n_pos_mf1": int(sum(1 for x in df if x > 0)),
                    "bar_pass": bool(all(x >= 0.030 for x in da) and all(x >= 0.030 for x in df)),
                }
            mm[f"{arm}_flips"] = {
                "fixed_per_seed": [per[f"{proto}_s{s}"]["test"][arm]["n_fixed"] for s in (0, 1, 2)],
                "broken_per_seed": [per[f"{proto}_s{s}"]["test"][arm]["n_broken"] for s in (0, 1, 2)],
            }
            for cname in cores:
                kk = f"core[{cname}]_fixed"
                if kk in per[f"{proto}_s0"]["test"][arm]:
                    mm[f"{arm}_flips"][f"{cname}_fixed_per_seed"] = [
                        per[f"{proto}_s{s}"]["test"][arm][kk] for s in (0, 1, 2)]
        mm["deployed_test_acc_3seed"] = round(float(np.mean(
            [per[f"{proto}_s{s}"]["test"]["deployed"]["acc"] for s in (0, 1, 2)])), 4)
        mm["deployed_test_mf1_3seed"] = round(float(np.mean(
            [per[f"{proto}_s{s}"]["test"]["deployed"]["mf1"] for s in (0, 1, 2)])), 4)
        means[proto] = mm
    OUT["means_3seed"] = means

    out_path = os.path.join(REPO, f"scripts/analysis/mechfix_{c['key']}_OUT.json")
    with open(out_path, "w") as f:
        json.dump(OUT, f, indent=1, default=str)
    print("wrote", out_path)


if __name__ == "__main__":
    main()
