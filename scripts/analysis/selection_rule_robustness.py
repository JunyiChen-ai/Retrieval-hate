#!/usr/bin/env python3
"""Selection-rule robustness re-analysis (zero GPU; reparses existing trainlogs).

Applies four epoch-selection rules to every arm of the multi-seed archive-kNN
experiment (see research-wiki/experiments/exp-archive-knn-seeds.md):

  (a) val-acc   : argmax Val acc, roc tie-break, epoch >= WARMUP   [pre-registered]
  (b) val-roc   : argmax Val roc, acc tie-break, epoch >= WARMUP
  (c) top3-mean : mean Test over the 3 best epochs by (Val acc, roc), epoch >= WARMUP
  (d) last5-mean: mean Test over the last 5 epochs (no selection reference)

Arms:
  ZH archive-kNN a0.25 (LoRA-Qwen)   seeds 0-4   jobs 12207,12215-12218
  ZH LoRA-only baseline              seeds 0-4   jobs 12223-12227
  EN archive-kNN a0.25 (frozen Qwen) seeds 0-3   jobs 12210,12219-12221
  EN frozen-Qwen floor (no archive)  seed 0      job 12113  [reference only]

Output: per-rule per-arm mean+-std (Test acc / macroF1) and ZH paired deltas.
"""
import math
import re
import statistics as st

LOGDIR = "/data/jehc223/RGCL/slurm/logs"
WARMUP = 5

VRE = re.compile(r"Val_Retrieval Epoch\s+(\d+) macroF1: ([\d.]+) macroP: ([\d.]+) macroR: ([\d.]+) acc: ([\d.]+) roc: ([\d.]+)")
TRE = re.compile(r"Test_Retrieval Epoch\s+(\d+) macroF1: ([\d.]+) macroP: ([\d.]+) macroR: ([\d.]+) acc: ([\d.]+) roc: ([\d.]+)")

ARMS = {
    "ZH_archive": {s: f"{LOGDIR}/arc_MHC_zh_Qwen2.5-VL-7B-Instruct-LoRA_HF_knn_a0.25_seg0full_{j}.trainlog"
                   for s, j in zip(range(5), [12207, 12215, 12216, 12217, 12218])},
    "ZH_baseline": {s: f"{LOGDIR}/arcbase_MHC_zh_Qwen2.5-VL-7B-Instruct-LoRA_HF_seed{s}_{j}.trainlog"
                    for s, j in zip(range(5), [12223, 12224, 12225, 12226, 12227])},
    "EN_archive": {s: f"{LOGDIR}/arc_MHC_Qwen2.5-VL-7B-Instruct_HF_knn_a0.25_seg0full_{j}.trainlog"
                   for s, j in zip(range(4), [12210, 12219, 12220, 12221])},
    "EN_floor": {0: f"{LOGDIR}/mllm_train_12113.out"},
}


def parse(path):
    log = open(path, errors="replace").read()
    val, test = {}, {}
    for m in VRE.finditer(log):
        val[int(m.group(1))] = tuple(float(x) for x in m.groups()[1:])
    for m in TRE.finditer(log):
        test[int(m.group(1))] = tuple(float(x) for x in m.groups()[1:])
    assert val and test, path
    return val, test  # (F1, P, R, acc, roc) per epoch; last occurrence wins


def apply_rule(rule, val, test):
    """Return (testF1, testAcc) under the given selection rule."""
    eligible = sorted(e for e in val if e >= WARMUP and e in test)
    if rule == "a_val_acc":
        e = max(eligible, key=lambda e: (val[e][3], val[e][4]))
        return test[e][0], test[e][3]
    if rule == "b_val_roc":
        e = max(eligible, key=lambda e: (val[e][4], val[e][3]))
        return test[e][0], test[e][3]
    if rule == "c_top3_mean":
        top = sorted(eligible, key=lambda e: (val[e][3], val[e][4]), reverse=True)[:3]
        return st.mean(test[e][0] for e in top), st.mean(test[e][3] for e in top)
    if rule == "d_last5_mean":
        last = sorted(test)[-5:]
        return st.mean(test[e][0] for e in last), st.mean(test[e][3] for e in last)
    raise ValueError(rule)


def fmt(xs):
    if len(xs) == 1:
        return f"{xs[0]:.4f} (n=1)"
    return f"{st.mean(xs):.4f}±{st.stdev(xs):.4f}"


def main():
    data = {arm: {s: parse(p) for s, p in seeds.items()} for arm, seeds in ARMS.items()}
    rules = ["a_val_acc", "b_val_roc", "c_top3_mean", "d_last5_mean"]
    res = {}  # (arm, rule) -> {seed: (F1, acc)}
    for arm in data:
        for rule in rules:
            res[(arm, rule)] = {s: apply_rule(rule, *vt) for s, vt in data[arm].items()}

    for rule in rules:
        print(f"\n===== rule {rule} =====")
        for arm in ARMS:
            r = res[(arm, rule)]
            accs = [r[s][1] for s in sorted(r)]
            f1s = [r[s][0] for s in sorted(r)]
            per = " ".join(f"s{s}:{r[s][1]:.4f}" for s in sorted(r))
            print(f"  {arm:12s} acc {fmt(accs):18s} F1 {fmt(f1s):18s} | {per}")
        # ZH paired deltas
        da = [res[('ZH_archive', rule)][s][1] - res[('ZH_baseline', rule)][s][1] for s in range(5)]
        df = [res[('ZH_archive', rule)][s][0] - res[('ZH_baseline', rule)][s][0] for s in range(5)]
        ta = st.mean(da) / (st.stdev(da) / math.sqrt(5)) if st.stdev(da) else float("inf")
        tf = st.mean(df) / (st.stdev(df) / math.sqrt(5)) if st.stdev(df) else float("inf")
        print(f"  ZH paired dAcc {st.mean(da):+.4f}±{st.stdev(da):.4f} (t={ta:+.2f}, +{sum(x>0 for x in da)}/5) "
              f"dF1 {st.mean(df):+.4f}±{st.stdev(df):.4f} (t={tf:+.2f}, +{sum(x>0 for x in df)}/5)")
        print(f"  ZH per-seed dAcc: " + " ".join(f"s{s}:{da[s]:+.4f}" for s in range(5)))


if __name__ == "__main__":
    main()
