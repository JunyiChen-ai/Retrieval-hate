#!/usr/bin/env python
"""
$0 probe (Stage-A, DEV-SIDE ONLY): validation-free checkpoint selection of the
RGCL head by the head-GRADIENT NORM, as a candidate fix for the F45 val-selection
tax. F68 ledger P2; source: refine-logs/LITSURVEY_NOVEL_MECHANISMS.md C2 (arXiv
2601.16874, "No Validation, No Problem: Predicting Model Performance from a Single
Gradient").

CONTRAST WITH THE DEAD SWA PROBE (F62/F62b, scripts/analysis/swa_probe.py):
this operator SELECTS one existing epoch checkpoint by a validation-free score; it
NEVER averages weights. Different operator class -> the F62 SWA weight-averaging
kill does not subsume it (see the record's governance note). This script reuses the
SWA probe's loader / head-forward / dev-eval machinery verbatim, only swapping the
operator: argmin(head-gradient-norm) instead of uniform weight-average.

MECHANISM (paper, as characterised in the litsurvey C2 + arXiv 2601.16874v1 abstract):
  For each banked per-epoch head checkpoint theta_e, run ONE forward-backward pass of
  the CLASSIFICATION loss on a fixed batch of detached TRAIN features through the head,
  record ||g||_F = ||dL/dW||_F (Frobenius norm of the gradient w.r.t. ALL head weights),
  scale-normalise, and SELECT the checkpoint that MINIMISES the score within a short
  TAIL window. Lower head-gradient -> flatter minimum -> better generalisation. The
  paper's rule (v1 abstract, verbatim): "Selecting the checkpoint with the minimum head
  gradient in a short tail window closes most of the gap to the oracle." The public
  arXiv version DEFERS the exact algorithm ("full algorithmic details ... will appear in
  a forthcoming paper"), so the rule below is pre-declared from that characterisation.

PRE-DECLARED DESIGN (locked before computing selection results; see the record):
  * Statistic  S(e) = ||grad_W L_BCE(B; theta_e)||_F / (||W_e||_F + 1e-12)
      - L_BCE  = plain nn.BCEWithLogitsLoss (pos_weight=None) on the head's
                 classification logit vs the binary label, i.e. the CLASSIFICATION
                 (BCE) term of the training hybrid loss -- NOT the full hybrid.
                 Reason (pre-declared): (a) the paper computes the *classification*
                 loss gradient through the head; (b) all three arms trained the BCE
                 term with pos_weight=None (default), so plain BCE is faithful; (c) the
                 hybrid's triplet term needs memory-bank retrieval + pseudo-gold /
                 hard-negative MINING, which is bank- and batch-composition-dependent
                 (not "a single gradient through the head") and would inject non-head
                 structure; (d) F45's tax is a classification-accuracy phenomenon that
                 BCE directly targets. The triplet term is excluded BY DESIGN.
      - W_e    = concatenation of ALL trainable head params (12 tensors: img_proj.0,
                 text_proj.0, mlp.1/4/7, output_layer -- weight+bias each). ||.||_F over
                 the concatenation. Head-scale (relative-gradient) normalisation is the
                 ONE scale-norm the paper offers to mitigate small-head instability;
                 it is scale-invariant across epochs (weights grow during training).
      - model.eval() so dropout is OFF -> S(e) is a DETERMINISTIC function of
                 (weights, batch); no RNG in the gradient (needed for the stability test).
  * Selection rule (ONE rule, no menu): argmin S(e) over the TAIL window ep20-29
      (the paper's "short tail window"; 30-epoch run -> last-10 = short tail). This
      DEVIATES from the launch brief's tentative "argmin over epochs >= warmup 5" and
      instead follows the paper's tail-window rule; the deviation is documented in the
      record with the v1 quote. Warmup=5 is subsumed (the tail starts at 20).
  * Two fixed probe batches (stability, bar i): a single seed-0 permutation of the
      train indices; batch A = perm[0:64] (the PRIMARY, used for the reported
      selection), batch B = perm[64:128] (disjoint; the pre-declared SWAP). Both fixed
      across all epochs and seeds. STABLE iff |sel_A - sel_B| <= 2.
  * Diagnostics (reported, NOT the rule): argmin over full post-warmup ep5-29, and
      Spearman(S(e), dev_acc(e)) over all 30 epochs and over the tail. Paper claims
      r ~ -0.85..-0.98 (high grad-norm <-> low acc). Reported honestly.

STAGE-A PROMOTE BAR (pre-declared; DEV-only -- a probe cannot prove a test gain):
  the rule is PROMOTABLE on a dataset iff
    (i)   STABLE: selected epoch shifts <= 2 when the probe batch is swapped (A vs B), AND
    (ii)  NON-VACUOUS: selected epoch differs from val-sel on >= 2/3 seeds, AND
    (iii) NOT-BROKEN: selected-epoch dev acc >= val-sel dev acc - 0.02,
  holding on >= 1 dataset. PROMOTE would authorise a prereg whose SINGLE test-touch
  compares {grad-norm-selected} vs {val-sel} vs {final} per seed -- that prereg is NOT
  written or run here.

HARD RULES honoured: CPU-only, no GPU/SLURM/Modal. NO TEST READS -- only train +
dev_seen caches are torch.load'ed (load_feats_from_CLIP is bypassed precisely because
it also loads test_seen; here test_seen*.pt is never opened, and no Test_* trainlog
line is read). autoresearch/goal_mllm_plus3/state/ is NOT modified.

GOVERNANCE: this is a SELECTION rule (one checkpoint at inference), not an
ensemble/average -> no collision with the cross-seed-ensemble veto and NOT subsumed by
the F62 SWA kill (different operator class). No number here enters any claims table.
"""
import os
import re
import sys
import json
import glob
import copy
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn as nn

REPO = "/data/jehc223/RGCL"
SRC = os.path.join(REPO, "src")
sys.path.insert(0, SRC)

# force CPU everywhere (no GPU, no SLURM, no Modal)
os.environ["CUDA_VISIBLE_DEVICES"] = ""

from data_loader.rac_dataloader import CLIP2Dataloader          # noqa: E402
from model.classifier import classifier_hateClipper            # noqa: E402
from model.evaluate_rac import retrieve_evaluate_RAC_          # noqa: E402
from utils.metrics import compute_metrics_retrieval            # noqa: E402

# ----------------------------------------------------------------------------
# Legs. Two are the promotable F45 targets (BLOCKED if their per-epoch ckpts have
# been pruned); "fb16" is the ONLY group with live per-epoch head checkpoints as of
# the census below and is a MACHINERY / mechanism-transfer leg (NOT promotable: it is
# the killed frame-16 HateMM arm, not an F45-measured arm; its test-touch status is
# irrelevant because this probe is dev-only and never promotes fb16).
# ----------------------------------------------------------------------------
CONFIGS = {
    "fb16": {
        "dataset": "HateMM",
        "model": "Qwen2.5-VL-7B-Instruct_HF-16f",
        "group": "RAC_video_fb16",
        "dev_n": 107,
        "promotable": False,
        "out": "refine-logs/GRADNORM_SELECT_PROBE_OUT.json",
        "note": ("frame-16 HateMM arm (F67, KILLED) -- the ONLY group with LIVE "
                 "per-epoch head ckpts; MACHINERY / mechanism-transfer leg, not a promote"),
    },
    "hatemm_curric": {
        "dataset": "HateMM",
        "model": "Qwen2.5-VL-7B-Instruct-LoRA-curric-rep2_HF",
        "group": "RAC_video_lora_curric_rep2",
        "dev_n": 107,
        "promotable": True,
        "out": "refine-logs/GRADNORM_SELECT_PROBE_HATEMM_OUT.json",
        "note": "HateMM curriculum-LoRA rep2 (job 13246) -- promotable F45 secondary target",
    },
    "zh": {
        "dataset": "MHC_zh",
        "model": "Qwen2.5-VL-7B-Instruct-LoRA_HF",
        "group": "RAC_video_lora_swaregen",
        "dev_n": 78,
        "promotable": True,
        "out": "refine-logs/GRADNORM_SELECT_PROBE_ZH_OUT.json",
        "note": "MHC_zh generic-LoRA regen (job 13294) -- promotable F45 PRIMARY target",
    },
}
LEG = sys.argv[1] if len(sys.argv) > 1 else "fb16"
assert LEG in CONFIGS, f"unknown leg {LEG!r} (use {'|'.join(CONFIGS)})"
CFG = CONFIGS[LEG]
DATASET = CFG["dataset"]
MODEL = CFG["model"]
GROUP = CFG["group"]
DATA_PATH = os.path.join(REPO, "data", "CLIP_Embedding")
CKPT_ROOT = os.path.join(REPO, "logging", "Retrieval", DATASET, GROUP)
SEEDS = [0, 1, 2]
WARMUP = 5
EPOCHS = 30
TOPK = 20
TAIL = (20, 29)          # the pre-declared "short tail window" (inclusive)
PROBE_BS = 64
NOT_BROKEN_TOL = 0.02    # bar (iii)
STABLE_TOL = 2           # bar (i): |sel_A - sel_B| <= 2

ARGS = SimpleNamespace(
    device="cpu", Faiss_GPU=False, metric="cos", save_embed=False,
    output_path=os.path.join(REPO, "scratchpad"),
    dataset=DATASET,           # HateMM & MHC_zh hit the else-branch -> 1-logit head
    batch_norm=False, num_layers=3, proj_dim=1024, map_dim=1024,
    fusion_mode="align", dropout=[0.2, 0.4, 0.1], topk=TOPK,
    similarity_threshold=-1.0, majority_voting="arithmetic", tarc_vote_gamma=0.0,
)


# ---- data (train + dev_seen ONLY; test_seen*.pt is NEVER opened) ----
def load_split(split):
    """Replicate data_loader.dataset.load_feats_split for ONE split, but only for
    train / dev_seen -- test_seen is never referenced here."""
    assert split in ("train", "dev_seen"), split
    path = os.path.join(DATA_PATH, DATASET, f"{split}_{MODEL}.pt")
    d = torch.load(path, map_location="cpu")
    ids = [item for sub in d["ids"] for item in sub]
    return [ids, d["img_feats"].float(), d["text_feats"].float(), d["labels"]]


def seed_ckpt_dir(seed):
    exp = ("RAC_lr0.0001_Bz64_Ep30_cosSim_triplet_drop[0.2, 0.4, 0.1]_topK20"
           "__PseudoGold_positive_1_hard_negative_1_seed{}_hybrid_loss_{}"
           .format(seed, MODEL))
    return os.path.join(CKPT_ROOT, exp, "ckpt")


def list_epoch_ckpts(seed):
    d = seed_ckpt_dir(seed)
    out = {}
    for p in glob.glob(os.path.join(glob.escape(d), "epoch_model_*.pt")):
        m = re.match(r"epoch_model_(\d+)_([0-9.]+)\.pt$", os.path.basename(p))
        if m:
            out[int(m.group(1))] = (p, float(m.group(2)))
    return out


def census():
    """Per-seed count of live per-epoch ckpts. BLOCKED if any seed lacks all 30."""
    counts = {s: len(list_epoch_ckpts(s)) for s in SEEDS}
    blocked = any(counts[s] < EPOCHS for s in SEEDS)
    return counts, blocked


def build_model(image_dim, text_dim):
    return classifier_hateClipper(
        image_dim, text_dim, ARGS.num_layers, ARGS.proj_dim, ARGS.map_dim,
        ARGS.fusion_mode, dropout=ARGS.dropout, batch_norm=ARGS.batch_norm, args=ARGS)


BCE = nn.BCEWithLogitsLoss()          # plain, pos_weight=None (mirrors training)


def grad_norm_stat(model, state_dict, img, text, label):
    """S(e) = ||grad_W L_BCE(batch)||_F / (||W||_F + 1e-12), deterministic (eval,
    dropout OFF). One forward-backward over the fixed probe batch."""
    model.load_state_dict(state_dict)
    model.eval()                       # dropout OFF -> deterministic gradient
    model.zero_grad(set_to_none=True)
    logit = model(img, text)           # [B, 1] classification logit
    loss = BCE(logit, label.float().reshape(-1, 1))
    loss.backward()
    gsq, wsq = 0.0, 0.0
    for p in model.parameters():
        if p.grad is not None:
            gsq += float(torch.sum(p.grad.double() ** 2))
        wsq += float(torch.sum(p.detach().double() ** 2))
    model.zero_grad(set_to_none=True)
    gnorm = float(np.sqrt(gsq))
    wnorm = float(np.sqrt(wsq))
    return {"loss": float(loss.detach()), "grad_fro": gnorm, "weight_fro": wnorm,
            "S": gnorm / (wnorm + 1e-12)}


def eval_state_dict(model, state_dict, train_dl, dev_dl):
    """DEV acc / macro-F1 / roc via the project's own retrieval-vote path (top-20
    signed-similarity arithmetic vote, use_sim). Train+dev only; test never read."""
    model.load_state_dict(state_dict)
    model.eval()
    with torch.no_grad():
        logging_dict, dev_labels = retrieve_evaluate_RAC_(
            train_dl, dev_dl, model, largest_retrieval=ARGS.topk,
            threshold=ARGS.similarity_threshold, args=ARGS, eval_name="dev",
            epoch=None, archive_bank=None, target_pack=None)
        acc, roc, pre, recall, f1, _, _, macro = compute_metrics_retrieval(
            logging_dict, dev_labels, majority_voting=ARGS.majority_voting,
            topk=ARGS.topk, use_sim=True, tarc_vote_gamma=0.0)
    return float(acc), float(macro["macro_f1"]), float(roc)


def spearman(xs, ys):
    """Spearman rho with average ranks for ties."""
    def rank(v):
        order = sorted(range(len(v)), key=lambda i: v[i])
        r = [0.0] * len(v)
        i = 0
        while i < len(v):
            j = i
            while j + 1 < len(v) and v[order[j + 1]] == v[order[i]]:
                j += 1
            avg = (i + j) / 2.0 + 1.0
            for k in range(i, j + 1):
                r[order[k]] = avg
            i = j + 1
        return r
    rx, ry = rank(xs), rank(ys)
    mx, my = sum(rx) / len(rx), sum(ry) / len(ry)
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    den = (sum((a - mx) ** 2 for a in rx) * sum((b - my) ** 2 for b in ry)) ** 0.5
    return num / den if den > 0 else float("nan")


def run_seed(seed, train, dev, probeA_idx, probeB_idx):
    torch.manual_seed(0)
    np.random.seed(0)

    (train_dl_shuf, dev_dl), _ = CLIP2Dataloader(
        train, dev, batch_size=64, return_dataset=True, normalize=False)
    from torch.utils.data import DataLoader
    train_dl = DataLoader(train_dl_shuf.dataset, batch_size=256, shuffle=False,
                          num_workers=0)

    img_all, text_all, lab_all = train[1], train[2], train[3]
    imgA, textA, labA = img_all[probeA_idx], text_all[probeA_idx], lab_all[probeA_idx]
    imgB, textB, labB = img_all[probeB_idx], text_all[probeB_idx], lab_all[probeB_idx]

    image_dim, text_dim = train[1].shape[1], train[2].shape[1]
    model = build_model(image_dim, text_dim)

    ckpts = list_epoch_ckpts(seed)
    assert len(ckpts) == EPOCHS, f"seed{seed}: {len(ckpts)} ckpts (want {EPOCHS})"

    per_epoch = {}
    for e in range(EPOCHS):
        path, facc = ckpts[e]
        sd = torch.load(path, map_location="cpu")
        acc, mf1, roc = eval_state_dict(model, sd, train_dl, dev_dl)
        sA = grad_norm_stat(model, sd, imgA, textA, labA)
        sB = grad_norm_stat(model, sd, imgB, textB, labB)
        per_epoch[e] = {"dev_acc": acc, "dev_mf1": mf1, "dev_roc": roc,
                        "file_acc": facc, "S_A": sA["S"], "S_B": sB["S"],
                        "grad_fro_A": sA["grad_fro"], "weight_fro_A": sA["weight_fro"],
                        "bce_loss_A": sA["loss"]}
    repro_maxdiff = max(abs(per_epoch[e]["dev_acc"] - per_epoch[e]["file_acc"])
                        for e in range(EPOCHS))

    warm = [e for e in range(EPOCHS) if e >= WARMUP]
    tail = [e for e in range(TAIL[0], TAIL[1] + 1)]

    # val-selected epoch (post-warmup argmax dev acc, tie-break dev roc)
    valsel = max(warm, key=lambda e: (per_epoch[e]["dev_acc"], per_epoch[e]["dev_roc"]))

    # grad-norm selection: argmin S over the TAIL window (A = primary, B = swap)
    selA = min(tail, key=lambda e: per_epoch[e]["S_A"])
    selB = min(tail, key=lambda e: per_epoch[e]["S_B"])
    # diagnostic-only: argmin over full post-warmup (NOT the rule) -- exposes whether
    # the tail-window restriction is load-bearing (an inverted mechanism sends the
    # full-range argmin to an early, low-acc epoch).
    sel_full = min(warm, key=lambda e: per_epoch[e]["S_A"])

    final = EPOCHS - 1

    Svec = [per_epoch[e]["S_A"] for e in range(EPOCHS)]
    Avec = [per_epoch[e]["dev_acc"] for e in range(EPOCHS)]
    sp_all = spearman(Svec, Avec)
    sp_tail = spearman([per_epoch[e]["S_A"] for e in tail],
                       [per_epoch[e]["dev_acc"] for e in tail])

    return {
        "seed": seed,
        "repro_maxdiff_vs_filename": repro_maxdiff,
        "valsel_epoch": valsel,
        "valsel_dev_acc": per_epoch[valsel]["dev_acc"],
        "valsel_dev_mf1": per_epoch[valsel]["dev_mf1"],
        "gradsel_epoch_A": selA,
        "gradsel_dev_acc_A": per_epoch[selA]["dev_acc"],
        "gradsel_dev_mf1_A": per_epoch[selA]["dev_mf1"],
        "gradsel_epoch_B": selB,
        "gradsel_dev_acc_B": per_epoch[selB]["dev_acc"],
        "gradsel_epoch_fullrange_diag": sel_full,
        "gradsel_dev_acc_fullrange_diag": per_epoch[sel_full]["dev_acc"],
        "final_epoch": final,
        "final_dev_acc": per_epoch[final]["dev_acc"],
        "final_dev_mf1": per_epoch[final]["dev_mf1"],
        "spearman_S_vs_devacc_all30": sp_all,
        "spearman_S_vs_devacc_tail": sp_tail,
        "stable_i": abs(selA - selB) <= STABLE_TOL,
        "nonvacuous_ii_seed": selA != valsel,
        "notbroken_iii": per_epoch[selA]["dev_acc"] >= per_epoch[valsel]["dev_acc"] - NOT_BROKEN_TOL,
        "per_epoch": {e: {k: per_epoch[e][k] for k in
                          ("dev_acc", "dev_mf1", "dev_roc", "file_acc",
                           "S_A", "S_B", "grad_fro_A", "weight_fro_A", "bce_loss_A")}
                      for e in range(EPOCHS)},
    }


def main():
    counts, blocked = census()
    print("=" * 78)
    print(f"GRAD-NORM SELECT PROBE [{LEG}] -- {DATASET} / {GROUP}, dev n={CFG['dev_n']}")
    print(f"leg note: {CFG['note']}")
    print(f"promotable target: {CFG['promotable']}")
    print("=" * 78)
    print("CENSUS (live per-epoch ckpts, want 30/seed):",
          {f"seed{s}": counts[s] for s in SEEDS})

    if blocked:
        status = "BLOCKED"
        out = {"probe": "grad-norm validation-free checkpoint selection ($0, CPU, dev-only)",
               "leg": LEG, "dataset": DATASET, "group": GROUP, "model": MODEL,
               "promotable_target": CFG["promotable"], "census": counts,
               "status": status,
               "reason": ("per-epoch head checkpoints missing/incomplete (pruned to B2 "
                          "by disk_guard); STOP per the STEP-0 rule")}
        outpath = os.path.join(REPO, CFG["out"])
        with open(outpath, "w") as f:
            json.dump(out, f, indent=2)
        print(f"\nSTATUS: {status} -- {out['reason']}")
        print(f"OUT.json -> {outpath}")
        return

    # fixed probe batches: one seed-0 permutation of train indices; A=[0:64] B=[64:128]
    train0 = load_split("train")
    dev0 = load_split("dev_seen")
    n_train = train0[1].shape[0]
    g = torch.Generator().manual_seed(0)
    perm = torch.randperm(n_train, generator=g)
    probeA_idx = perm[:PROBE_BS]
    probeB_idx = perm[PROBE_BS:2 * PROBE_BS]
    assert len(probeA_idx) == PROBE_BS and len(probeB_idx) == PROBE_BS

    results = []
    for s in SEEDS:
        # each seed shares the SAME feature cache (features are seed-independent;
        # only the trained head differs), so the probe batch is identical across seeds.
        results.append(run_seed(s, train0, dev0, probeA_idx, probeB_idx))

    # Stage-A conditions (evaluated for BOTH promotable and machinery legs; the VERDICT
    # label differs -- machinery legs report MACHINERY, promotable legs PROMOTE/KILL)
    cond_i_stable = all(r["stable_i"] for r in results)
    n_nonvac = sum(r["nonvacuous_ii_seed"] for r in results)
    cond_ii_nonvac = n_nonvac >= 2
    cond_iii_notbroken = all(r["notbroken_iii"] for r in results)
    stageA_pass = cond_i_stable and cond_ii_nonvac and cond_iii_notbroken

    # MECHANISM SANITY (the paper's whole premise): ||g|| must be NEGATIVELY correlated
    # with accuracy (paper: rho ~ -0.85..-0.98), so that argmin(||g||) picks HIGH acc.
    # If our Spearman is POSITIVE, the argmin rule anti-selects and any Stage-A "pass"
    # is a boundary artifact of the flat tail window, not the mechanism working.
    sp_all = sorted(r["spearman_S_vs_devacc_all30"] for r in results)
    median_sp = sp_all[len(sp_all) // 2]
    mechanism_holds = median_sp < 0

    if CFG["promotable"]:
        # a promote needs BOTH the Stage-A conditions AND the mechanism holding.
        verdict = "PROMOTE" if (stageA_pass and mechanism_holds) else "KILL"
    else:
        if stageA_pass and mechanism_holds:
            verdict = "MACHINERY_WELL_BEHAVED"
        elif stageA_pass and not mechanism_holds:
            verdict = "MACHINERY_DEGENERATE_PASS_MECHANISM_REFUTED"
        else:
            verdict = "MACHINERY_NOT_WELL_BEHAVED"

    out = {
        "probe": "grad-norm validation-free checkpoint selection ($0, CPU, dev-only)",
        "paper": "arXiv 2601.16874 (litsurvey C2); tail-window argmin of head-gradient norm",
        "leg": LEG, "leg_note": CFG["note"], "dataset": DATASET, "group": GROUP,
        "model": MODEL, "dev_n": CFG["dev_n"], "promotable_target": CFG["promotable"],
        "census": counts, "status": "RUN",
        "design": {
            "statistic": "S(e)=||grad_W L_BCE(B)||_F / (||W||_F+1e-12), model.eval (dropout off)",
            "loss": "classification (BCE) term only, pos_weight=None; triplet term excluded by design",
            "rule": f"argmin S over TAIL window ep{TAIL[0]}-{TAIL[1]} (paper's 'short tail window')",
            "probe_batches": "seed-0 perm; A=idx[0:64] (primary), B=idx[64:128] (swap)",
            "warmup": WARMUP,
        },
        "seeds": SEEDS, "per_seed": results,
        "stageA": {
            "cond_i_stable_batchswap": cond_i_stable,
            "cond_ii_nonvacuous_ge2of3": cond_ii_nonvac,
            "n_seeds_gradsel_ne_valsel": n_nonvac,
            "cond_iii_notbroken_dev": cond_iii_notbroken,
            "all_pass": stageA_pass,
        },
        "mechanism_check": {
            "paper_claim_spearman": "-0.85..-0.98 (NEGATIVE: high ||g|| <-> low acc)",
            "our_spearman_all30_per_seed": [r["spearman_S_vs_devacc_all30"] for r in results],
            "our_median_spearman_all30": median_sp,
            "mechanism_holds_negative": mechanism_holds,
            "note": ("if positive, argmin(||g||) anti-selects; the full-range argmin "
                     "diag lands at an EARLY low-acc epoch and only the tail-window "
                     "boundary keeps the rule 'not broken' on the flat tail"),
            "fullrange_argmin_epoch_per_seed": [r["gradsel_epoch_fullrange_diag"] for r in results],
            "fullrange_argmin_devacc_per_seed": [r["gradsel_dev_acc_fullrange_diag"] for r in results],
        },
        "verdict": verdict,
        "governance": ("SELECTION rule (one checkpoint at inference), not an "
                       "ensemble/average -> no cross-seed-ensemble-veto collision and "
                       "NOT subsumed by the F62 SWA weight-averaging kill (different "
                       "operator class). No number here enters any claims table."),
    }
    outpath = os.path.join(REPO, CFG["out"])
    with open(outpath, "w") as f:
        json.dump(out, f, indent=2)

    # console table
    for r in results:
        print(f"\n--- seed {r['seed']} ---")
        print(f"  reproduction max|recomputed-filename| dev acc = "
              f"{r['repro_maxdiff_vs_filename']:.6f}")
        print(f"  val-sel   ep{r['valsel_epoch']:>2}: dev acc {r['valsel_dev_acc']:.4f} "
              f"mF1 {r['valsel_dev_mf1']:.4f}")
        print(f"  grad-sel  ep{r['gradsel_epoch_A']:>2} (batch A): dev acc "
              f"{r['gradsel_dev_acc_A']:.4f} mF1 {r['gradsel_dev_mf1_A']:.4f}")
        print(f"  grad-sel  ep{r['gradsel_epoch_B']:>2} (batch B, swap)   "
              f"[full-range diag argmin ep{r['gradsel_epoch_fullrange_diag']}]")
        print(f"  final     ep{r['final_epoch']:>2}: dev acc {r['final_dev_acc']:.4f} "
              f"mF1 {r['final_dev_mf1']:.4f}")
        print(f"  Spearman(S, dev_acc): all30 {r['spearman_S_vs_devacc_all30']:+.3f}  "
              f"tail {r['spearman_S_vs_devacc_tail']:+.3f}")
        print(f"  (i) stable={r['stable_i']}  (ii) grad-sel!=val-sel={r['nonvacuous_ii_seed']}  "
              f"(iii) not-broken={r['notbroken_iii']}")
    print("\n" + "=" * 78)
    print(f"STAGE-A: (i) stable={cond_i_stable}  (ii) non-vacuous>=2/3={cond_ii_nonvac} "
          f"({n_nonvac}/3)  (iii) not-broken={cond_iii_notbroken}  -> pass={stageA_pass}")
    print(f"MECHANISM: median Spearman(S,dev_acc) all30 = {median_sp:+.3f}  "
          f"(paper: NEGATIVE ~-0.85..-0.98)  -> holds={mechanism_holds}")
    print(f"  full-range argmin (no tail window) lands at ep"
          f"{[r['gradsel_epoch_fullrange_diag'] for r in results]} "
          f"dev acc {[round(r['gradsel_dev_acc_fullrange_diag'],4) for r in results]}")
    print(f"VERDICT ({DATASET}, promotable={CFG['promotable']}): {verdict}")
    print(f"OUT.json -> {outpath}")
    print("=" * 78)


if __name__ == "__main__":
    main()
