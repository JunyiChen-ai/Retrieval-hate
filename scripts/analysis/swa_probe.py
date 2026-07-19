#!/usr/bin/env python
"""
$0 probe: single-trajectory Stochastic Weight Averaging (SWA) of the RGCL head's
per-epoch checkpoints, as a candidate fix for the F45 val-selection tax.

Family C of refine-logs/REDTEAM_EXTERNAL_FAMILIES.md (§3). ZERO GPU / CPU only.
No test-set reads: train + dev_seen features ONLY. The per-epoch checkpoints are
the banked artifacts saved by src/run_rac.py:764 (one head state_dict per epoch).

What it does, per seed:
  * reproduce the banked per-epoch DEV retrieval acc (the `select_acc` embedded in
    each ckpt filename) via the project's own retrieval-eval path, to prove the
    inference is faithful, then
  * uniform-average the head weights over 3 pre-declared windows
    {post-warmup ep5-29, last-10 ep20-29, last-5 ep25-29} and evaluate each SWA
    arm's DEV acc / macro-F1 against (a) the val-selected single-epoch max and
    (b) the final-epoch checkpoint, on the SAME cached dev features.
  * free train-side diagnostics: L2(val-sel, final) and last-10 dev-acc jitter.

The head is pure Linear/ReLU/Dropout + parameter-free L2-normalize (batch_norm
False) -> NO BatchNorm running stats, so uniform weight averaging is clean and
dropout is inactive under model.eval().

PRE-DECLARED PROMOTE CRITERION (dev-only; a probe cannot prove a test gain):
  PROMOTE iff, on >=1 dataset, EVERY SWA window's dev acc >= (max post-warmup
  single-epoch dev acc - 0.005) AND the SWA dev spread (max-min across the 3
  windows) is < the single-epoch post-warmup dev spread (max-min) -- i.e. SWA does
  not pay the selection tax on dev. Otherwise KILL. Promotion authorizes drafting a
  prereg whose SINGLE test-touch compares {SWA-last10} vs {val-sel} vs {final}.

GOVERNANCE: the standing veto bans CROSS-SEED ensembles. Single-trajectory weight
averaging is ONE model at inference from ONE seed -> plain-text NOT covered, but a
user micro-ruling is required before any SWA number enters a claims table.
"""
import os
import sys
import re
import json
import glob
import copy
from types import SimpleNamespace

import numpy as np
import torch

REPO = "/data/jehc223/RGCL"
SRC = os.path.join(REPO, "src")
sys.path.insert(0, SRC)

# force CPU everywhere (no GPU, no SLURM)
os.environ["CUDA_VISIBLE_DEVICES"] = ""

from data_loader.dataset import load_feats_from_CLIP          # noqa: E402
from data_loader.rac_dataloader import CLIP2Dataloader         # noqa: E402
from model.classifier import classifier_hateClipper           # noqa: E402
from model.evaluate_rac import retrieve_evaluate_RAC_         # noqa: E402
from utils.metrics import compute_metrics_retrieval           # noqa: E402

# ----------------------------------------------------------------------------
# Run configuration (mirrors scripts/slurm/enc3seed_lora_curric_rep2.sbatch).
# ONLY the HateMM curriculum-LoRA rep2 group has live per-epoch checkpoints; every
# other banked group's ckpt dir was cleaned (empty on disk) -> ZH is BLOCKED.
# ----------------------------------------------------------------------------
# Two legs share the IDENTICAL §2 SWA design; only this config block differs.
# Select with argv[1] in {"hatemm","zh"} (default "hatemm").
CONFIGS = {
    "hatemm": {
        "dataset": "HateMM",
        "model": "Qwen2.5-VL-7B-Instruct-LoRA-curric-rep2_HF",
        "group": "RAC_video_lora_curric_rep2",
        "dev_n": 107,
        "grepro_ref": None,   # HateMM leg: filename cross-check only
        "out": "refine-logs/SWA_PROBE_OUT.json",
        "note": "curriculum-LoRA rep2 (job 13246) -- the only group with live ckpts",
    },
    "zh": {
        "dataset": "MHC_zh",
        "model": "Qwen2.5-VL-7B-Instruct-LoRA_HF",
        "group": "RAC_video_lora_swaregen",
        "dev_n": 78,
        # G-repro reference: the banked B3/13150 Val_Retrieval DEV curve (dev acc
        # per epoch), parsed from the trainlog Val_ lines ONLY (never a Test_ line).
        "grepro_ref": {
            "job": 13150,
            "trainlog": ("slurm/logs/enc3s_MHC_zh_Qwen2.5-VL-7B-Instruct-"
                         "LoRA_HF_seed{seed}_13150.trainlog"),
        },
        "out": "refine-logs/SWA_PROBE_ZH_OUT.json",
        "note": ("generic-LoRA regen (job 13294) of the B3/13150 arm where F45's "
                 "val-selection tax was measured"),
    },
}
LEG = sys.argv[1] if len(sys.argv) > 1 else "hatemm"
assert LEG in CONFIGS, f"unknown leg {LEG!r} (use hatemm|zh)"
CFG = CONFIGS[LEG]
DATASET = CFG["dataset"]
MODEL = CFG["model"]
GROUP = CFG["group"]
DATA_PATH = os.path.join(REPO, "data")
CKPT_ROOT = os.path.join(REPO, "logging", "Retrieval", DATASET, GROUP)
SEEDS = [0, 1, 2]
WARMUP = 5
EPOCHS = 30
TOPK = 20

# pre-declared SWA windows: (name, first_epoch, last_epoch) inclusive
WINDOWS = [
    ("swa_postwarmup_5_29", 5, 29),
    ("swa_last10_20_29", 20, 29),
    ("swa_last5_25_29", 25, 29),
]
PROMOTE_TOL = 0.005

# args object consumed by retrieve_evaluate_RAC_ / classifier_hateClipper.
ARGS = SimpleNamespace(
    device="cpu",
    Faiss_GPU=False,
    metric="cos",
    save_embed=False,
    output_path=os.path.join(REPO, "scratchpad"),  # unused (save_embed=False)
    dataset=DATASET,  # HateMM & MHC_zh both hit the else-branch -> binary 1-logit head

    batch_norm=False,
    num_layers=3,
    proj_dim=1024,
    map_dim=1024,
    fusion_mode="align",
    dropout=[0.2, 0.4, 0.1],
    topk=TOPK,
    similarity_threshold=-1.0,
    majority_voting="arithmetic",
    tarc_vote_gamma=0.0,
)


def seed_ckpt_dir(seed):
    # exp_name = ..._hybrid_loss + exp_comment("_{MODEL}"); MODEL already ends _HF.
    exp = ("RAC_lr0.0001_Bz64_Ep30_cosSim_triplet_drop[0.2, 0.4, 0.1]_topK20"
           "__PseudoGold_positive_1_hard_negative_1_seed{}_hybrid_loss_{}"
           .format(seed, MODEL))
    return os.path.join(CKPT_ROOT, exp, "ckpt")


def list_epoch_ckpts(seed):
    """epoch -> (path, filename_select_acc). Parses epoch_model_{e}_{acc}.pt."""
    d = seed_ckpt_dir(seed)
    out = {}
    # NB: the ckpt dir path contains "[0.2, 0.4, 0.1]" -> glob.escape so the
    # brackets are not parsed as a character class (otherwise 0 matches).
    for p in glob.glob(os.path.join(glob.escape(d), "epoch_model_*.pt")):
        m = re.match(r"epoch_model_(\d+)_([0-9.]+)\.pt$", os.path.basename(p))
        if m:
            out[int(m.group(1))] = (p, float(m.group(2)))
    return out


def build_model(image_dim, text_dim):
    return classifier_hateClipper(
        image_dim, text_dim, ARGS.num_layers, ARGS.proj_dim, ARGS.map_dim,
        ARGS.fusion_mode, dropout=ARGS.dropout, batch_norm=ARGS.batch_norm,
        args=ARGS)


def average_state_dicts(state_dicts):
    avg = copy.deepcopy(state_dicts[0])
    for k in avg:
        acc = torch.zeros_like(state_dicts[0][k], dtype=torch.float32)
        for sd in state_dicts:
            acc += sd[k].float()
        avg[k] = acc / float(len(state_dicts))
    return avg


def eval_state_dict(model, state_dict, train_dl, dev_dl):
    """Load a state_dict and return (dev_acc, dev_macro_f1) via the project's
    own retrieval-vote path (top-20 signed-similarity arithmetic vote, use_sim)."""
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


def flat_params(state_dict):
    return torch.cat([v.float().reshape(-1) for v in state_dict.values()])


def run_seed(seed):
    torch.manual_seed(0)
    np.random.seed(0)

    # data: train + dev ONLY (never touch test_seen for selection or eval)
    train, dev, _test = load_feats_from_CLIP(
        os.path.join(DATA_PATH, "CLIP_Embedding"), DATASET, MODEL)
    # deterministic memory bank: build both dataloaders with shuffle off by
    # feeding dev as the "non-first" position -> shuffle=False; for train we
    # rebuild a shuffle-free loader so the kNN memory is identical across every
    # checkpoint (kNN vote is order-invariant anyway; this makes it explicit).
    (train_dl_shuf, dev_dl, _), _ = CLIP2Dataloader(
        train, dev, _test, batch_size=64, return_dataset=True, normalize=False)
    from torch.utils.data import DataLoader
    train_dl = DataLoader(train_dl_shuf.dataset, batch_size=256,
                          shuffle=False, num_workers=0)

    image_dim = train[1].shape[1]
    text_dim = train[2].shape[1]
    model = build_model(image_dim, text_dim)

    ckpts = list_epoch_ckpts(seed)
    assert len(ckpts) == EPOCHS, f"seed{seed}: found {len(ckpts)} ckpts (want {EPOCHS})"

    # -- single-epoch dev curve (recomputed) + filename cross-check --
    per_epoch = {}
    file_acc = {}
    for e in range(EPOCHS):
        path, facc = ckpts[e]
        sd = torch.load(path, map_location="cpu")
        acc, mf1, roc = eval_state_dict(model, sd, train_dl, dev_dl)
        per_epoch[e] = {"dev_acc": acc, "dev_mf1": mf1, "dev_roc": roc,
                        "file_acc": facc}
        file_acc[e] = facc
    repro_maxdiff = max(abs(per_epoch[e]["dev_acc"] - file_acc[e])
                        for e in range(EPOCHS))

    # -- val-selected epoch (post-warmup argmax dev acc, tie-break dev roc) --
    warm = [e for e in range(EPOCHS) if e >= WARMUP]
    valsel_epoch = max(warm, key=lambda e: (per_epoch[e]["dev_acc"],
                                            per_epoch[e]["dev_roc"]))
    valsel_acc = per_epoch[valsel_epoch]["dev_acc"]
    valsel_mf1 = per_epoch[valsel_epoch]["dev_mf1"]

    postwarm_accs = [per_epoch[e]["dev_acc"] for e in warm]
    max_single_acc = max(postwarm_accs)
    single_spread = max(postwarm_accs) - min(postwarm_accs)

    final_epoch = EPOCHS - 1
    final_acc = per_epoch[final_epoch]["dev_acc"]
    final_mf1 = per_epoch[final_epoch]["dev_mf1"]

    # -- SWA windows --
    swa = {}
    for name, e0, e1 in WINDOWS:
        sds = [torch.load(ckpts[e][0], map_location="cpu")
               for e in range(e0, e1 + 1)]
        avg = average_state_dicts(sds)
        acc, mf1, roc = eval_state_dict(model, avg, train_dl, dev_dl)
        swa[name] = {"window": [e0, e1], "n": e1 - e0 + 1,
                     "dev_acc": acc, "dev_mf1": mf1, "dev_roc": roc}

    swa_accs = [swa[n]["dev_acc"] for n, _, _ in WINDOWS]
    swa_spread = max(swa_accs) - min(swa_accs)

    # -- free train-side diagnostics --
    valsel_sd = torch.load(ckpts[valsel_epoch][0], map_location="cpu")
    final_sd = torch.load(ckpts[final_epoch][0], map_location="cpu")
    l2_valsel_final = float(torch.norm(
        flat_params(valsel_sd) - flat_params(final_sd), p=2))
    last10 = [per_epoch[e]["dev_acc"] for e in range(20, 30)]
    last10_jitter = max(last10) - min(last10)

    # -- pre-declared decision (per seed) --
    cond_a = all(a >= (max_single_acc - PROMOTE_TOL) for a in swa_accs)
    cond_b = swa_spread < single_spread
    promote = cond_a and cond_b

    return {
        "seed": seed,
        "repro_maxdiff_vs_filename": repro_maxdiff,
        "valsel_epoch": valsel_epoch,
        "valsel_dev_acc": valsel_acc,
        "valsel_dev_mf1": valsel_mf1,
        "max_single_postwarmup_dev_acc": max_single_acc,
        "single_postwarmup_dev_spread": single_spread,
        "final_epoch": final_epoch,
        "final_dev_acc": final_acc,
        "final_dev_mf1": final_mf1,
        "swa": swa,
        "swa_dev_spread": swa_spread,
        "l2_valsel_final": l2_valsel_final,
        "last10_dev_acc_jitter": last10_jitter,
        "cond_a_all_swa_ge_maxsingle_minus_tol": cond_a,
        "cond_b_swa_spread_lt_single_spread": cond_b,
        "seed_verdict": "PROMOTE" if promote else "KILL",
        "per_epoch_dev_acc": {e: per_epoch[e]["dev_acc"] for e in range(EPOCHS)},
    }


def grepro_ref(seed):
    """G-repro sanity for the ZH leg: parse the banked reference run's
    Val_Retrieval DEV acc per epoch (legacy line, acc==select_acc) from the
    trainlog Val_ lines ONLY, and compare to the regenerated ckpt filenames'
    select_acc (both GPU-computed 4dp; bit-identical run => exact match)."""
    ref_path = os.path.join(REPO, CFG["grepro_ref"]["trainlog"].format(seed=seed))
    ref = {}
    with open(ref_path) as f:
        for line in f:
            # legacy line "Val_Retrieval Epoch  {e} acc: {:.4f} roc: ..." only;
            # the macroF1 line has "Epoch {e} macroF1:" so it does NOT match here,
            # and NO Test_ line is ever read.
            m = re.search(r"Val_Retrieval Epoch\s+(\d+) acc: ([0-9.]+)", line)
            if m:
                ref[int(m.group(1))] = float(m.group(2))
    ckpts = list_epoch_ckpts(seed)
    per_epoch = {}
    for e in range(EPOCHS):
        new4 = round(ckpts[e][1], 4)
        per_epoch[e] = {"regen": new4, "banked": ref.get(e)}
    diffs = [abs(per_epoch[e]["regen"] - per_epoch[e]["banked"])
             for e in range(EPOCHS) if per_epoch[e]["banked"] is not None]
    return {
        "seed": seed,
        "ref_job": CFG["grepro_ref"]["job"],
        "max_abs_diff_4dp": max(diffs) if diffs else None,
        "n_epoch_mismatch": sum(1 for d in diffs if d > 1e-9),
        "per_epoch": per_epoch,
    }


def main():
    # ZH leg: run the G-repro sanity gate BEFORE the SWA design.
    grepro = None
    grepro_pass = None
    if CFG["grepro_ref"] is not None:
        grepro = [grepro_ref(s) for s in SEEDS]
        gmax = max(g["max_abs_diff_4dp"] for g in grepro)
        # bit-exact expected (same seed/code/features, deterministic config).
        # Tolerance: accept isolated <=1-item/dev_n drift; STOP on anything larger.
        one_item = 1.0 / CFG["dev_n"] + 1e-9
        grepro_pass = gmax <= one_item
        print("=" * 78)
        print(f"G-REPRO SANITY (regen job 13294 vs banked job "
              f"{CFG['grepro_ref']['job']}) -- DEV curve, 4dp")
        print("=" * 78)
        for g in grepro:
            print(f"  seed {g['seed']}: max|regen-banked| dev acc (4dp) = "
                  f"{g['max_abs_diff_4dp']:.4f}   epochs differing = "
                  f"{g['n_epoch_mismatch']}/{EPOCHS}")
        print(f"  overall max abs diff = {gmax:.4f}   "
              f"(<= 1 item = {one_item:.4f} ?)  -> "
              f"{'PASS' if grepro_pass else 'FAIL/STOP'}")
        if not grepro_pass:
            print("\nG-REPRO FAILED: regenerated dev curve does not match the "
                  "banked run beyond 1-item tolerance. STOPPING before SWA "
                  "(do not trust these checkpoints).")
            outpath = os.path.join(REPO, CFG["out"])
            with open(outpath, "w") as f:
                json.dump({"leg": LEG, "grepro": grepro,
                           "grepro_pass": False,
                           "status": "STOP_GREPRO_FAIL"}, f, indent=2)
            print(f"OUT.json -> {outpath}")
            return

    results = [run_seed(s) for s in SEEDS]

    # aggregate verdict: PROMOTE only if the criterion holds (mission says
    # "on >=1 dataset ... every SWA window ...", evaluated per seed then read
    # across seeds -- we require the criterion to hold on all 3 seeds to call
    # the DATASET a promote, since a single-seed pass is seed-noise).
    n_promote = sum(r["seed_verdict"] == "PROMOTE" for r in results)
    dataset_verdict = "PROMOTE" if n_promote == len(SEEDS) else "KILL"

    out = {
        "probe": "single-trajectory SWA of RGCL head checkpoints ($0, CPU)",
        "leg": LEG,
        "leg_note": CFG["note"],
        "dataset": DATASET,
        "dev_n": CFG["dev_n"],
        "group": GROUP,
        "model": MODEL,
        "seeds": SEEDS,
        "warmup": WARMUP,
        "windows": [{"name": n, "e0": e0, "e1": e1} for n, e0, e1 in WINDOWS],
        "promote_tol": PROMOTE_TOL,
        "grepro": grepro,
        "grepro_pass": grepro_pass,
        "per_seed": results,
        "n_seeds_promote": n_promote,
        "dataset_verdict": dataset_verdict,
        "governance_flag": ("single-trajectory weight averaging is ONE model "
                            "from ONE seed; plain-text NOT the cross-seed-ensemble "
                            "ban, but requires a user micro-ruling before any SWA "
                            "number enters a claims table"),
    }

    outpath = os.path.join(REPO, CFG["out"])
    with open(outpath, "w") as f:
        json.dump(out, f, indent=2)

    # human-readable console table
    print("=" * 78)
    print(f"SWA PROBE [{LEG}] -- {DATASET} / {GROUP}, dev n={CFG['dev_n']}, CPU $0")
    print("=" * 78)
    for r in results:
        print(f"\n--- seed {r['seed']} ---")
        print(f"  reproduction max|recomputed-filename| dev acc = "
              f"{r['repro_maxdiff_vs_filename']:.6f}  (0 => faithful)")
        print(f"  val-selected epoch {r['valsel_epoch']:>2}: dev acc "
              f"{r['valsel_dev_acc']:.4f}  mF1 {r['valsel_dev_mf1']:.4f}")
        print(f"  final epoch     {r['final_epoch']:>2}: dev acc "
              f"{r['final_dev_acc']:.4f}  mF1 {r['final_dev_mf1']:.4f}")
        print(f"  max post-warmup single-epoch dev acc = "
              f"{r['max_single_postwarmup_dev_acc']:.4f}  "
              f"(spread {r['single_postwarmup_dev_spread']:.4f})")
        for name, _, _ in WINDOWS:
            s = r["swa"][name]
            print(f"  {name:22s} ep{s['window'][0]}-{s['window'][1]} "
                  f"(n={s['n']:>2}): dev acc {s['dev_acc']:.4f}  "
                  f"mF1 {s['dev_mf1']:.4f}")
        print(f"  SWA dev-acc spread across windows = {r['swa_dev_spread']:.4f}")
        print(f"  L2(val-sel, final) = {r['l2_valsel_final']:.4f}   "
              f"last-10 dev-acc jitter = {r['last10_dev_acc_jitter']:.4f}")
        print(f"  cond_A (all SWA >= maxsingle-{PROMOTE_TOL}): "
              f"{r['cond_a_all_swa_ge_maxsingle_minus_tol']}   "
              f"cond_B (SWA spread < single spread): "
              f"{r['cond_b_swa_spread_lt_single_spread']}")
        print(f"  SEED VERDICT: {r['seed_verdict']}")
    print("\n" + "=" * 78)
    print(f"DATASET VERDICT ({DATASET}): {dataset_verdict}  "
          f"({n_promote}/{len(SEEDS)} seeds promote)")
    if grepro_pass is not None:
        print(f"G-repro: {'PASS' if grepro_pass else 'FAIL'}")
    print(f"OUT.json -> {outpath}")
    print("=" * 78)


if __name__ == "__main__":
    main()
