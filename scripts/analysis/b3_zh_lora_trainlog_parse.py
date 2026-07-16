#!/usr/bin/env python
"""
B3 ZH LoRA decomposition — parse banked trainlogs (NO new test-touch; reads
already-logged per-epoch Val/Test retrieval metrics, exactly as B3_VERDICT_REVIEW
was written). Answers Q2 (Pareto vs rotation on TEST) and Q3 (val-selection).

Arms (all current-code enc3seed runner, MHC-ZH):
  CLIP        job 13115  enc3s_MHC_zh_openai_clip-vit-large-patch14-336_HF_seed{s}_13115.trainlog
  frozenQwen  job 13115  enc3s_MHC_zh_Qwen2.5-VL-7B-Instruct_HF_seed{s}_13115.trainlog
  LoRAQwen    job 13150  enc3s_MHC_zh_Qwen2.5-VL-7B-Instruct-LoRA_HF_seed{s}_13150.trainlog
"""
import re, os, json
import numpy as np

LOG = "/data/jehc223/RGCL/slurm/logs"
ARMS = {
    "CLIP":       ("openai_clip-vit-large-patch14-336_HF", "13115"),
    "frozenQwen": ("Qwen2.5-VL-7B-Instruct_HF", "13115"),
    "LoRAQwen":   ("Qwen2.5-VL-7B-Instruct-LoRA_HF", "13150"),
}
SEEDS = (0, 1, 2)
WARMUP = 5

# line A: "Val_Retrieval Epoch  N acc: .. roc: .. pre: .. recall: .. f1: .."  (pos-class)
reA = re.compile(r"(Val|Test)_Retrieval Epoch\s+(\d+) acc: ([\d.]+) roc: ([\d.]+) pre: ([\d.]+) recall: ([\d.]+) f1: ([\d.]+)\s*$")
# line B: "Val_Retrieval Epoch N macroF1: .. macroP: .. macroR: .. acc: .. roc: .."
reB = re.compile(r"(Val|Test)_Retrieval Epoch\s+(\d+) macroF1: ([\d.]+) macroP: ([\d.]+) macroR: ([\d.]+) acc: ([\d.]+) roc: ([\d.]+)")


def parse(path):
    ep = {}  # ep -> dict
    with open(path) as f:
        for ln in f:
            m = reA.match(ln.strip())
            if m:
                split, e, acc, roc, pre, rec, f1 = m.group(1), int(m.group(2)), *map(float, m.groups()[2:])
                d = ep.setdefault(e, {})
                d[f"{split}_acc"] = acc; d[f"{split}_roc"] = roc
                d[f"{split}_hate_recall"] = rec; d[f"{split}_hate_pre"] = pre; d[f"{split}_hate_f1"] = f1
                continue
            m = reB.match(ln.strip())
            if m:
                split, e, mf1, mp, mr, acc, roc = m.group(1), int(m.group(2)), *map(float, m.groups()[2:])
                d = ep.setdefault(e, {})
                d[f"{split}_mF1"] = mf1; d[f"{split}_macroR"] = mr; d[f"{split}_acc"] = acc
    # derive non-hate recall = 2*macroR - hate_recall
    for e, d in ep.items():
        for sp in ("Val", "Test"):
            if f"{sp}_macroR" in d and f"{sp}_hate_recall" in d:
                d[f"{sp}_nonhate_recall"] = 2 * d[f"{sp}_macroR"] - d[f"{sp}_hate_recall"]
    return ep


def valselect(ep):
    cand = [e for e in ep if e >= WARMUP and "Val_acc" in ep[e]]
    # max Val acc, roc tie-break
    best = max(cand, key=lambda e: (ep[e]["Val_acc"], ep[e].get("Val_roc", 0)))
    return best


OUT = {}
for arm, (tag, job) in ARMS.items():
    OUT[arm] = {"seeds": {}}
    for s in SEEDS:
        p = f"{LOG}/enc3s_MHC_zh_{tag}_seed{s}_{job}.trainlog"
        ep = parse(p)
        fin = max(ep)
        vs = valselect(ep)
        # true best test epoch (oracle, for regret)
        best_test = max((e for e in ep if e >= WARMUP), key=lambda e: ep[e]["Test_acc"])
        OUT[arm]["seeds"][s] = dict(
            final_ep=fin, valsel_ep=vs, best_test_ep=best_test,
            final=ep[fin], valsel=ep[vs], best_test_acc=ep[best_test]["Test_acc"])
    # aggregate
    for proto, key in [("final", "final"), ("valsel", "valsel")]:
        for met in ("Test_acc", "Test_mF1", "Test_hate_recall", "Test_nonhate_recall"):
            vals = [OUT[arm]["seeds"][s][key].get(met, float("nan")) for s in SEEDS]
            OUT[arm].setdefault(proto + "_mean", {})[met] = float(np.mean(vals))

print("=== per-arm final-epoch (ep29) TEST, mean over seeds ===")
print(f"{'arm':11s} | acc     mF1     hate_rec  nonhate_rec")
for arm in ARMS:
    m = OUT[arm]["final_mean"]
    print(f"{arm:11s} | {m['Test_acc']:.4f}  {m['Test_mF1']:.4f}  {m['Test_hate_recall']:.4f}    {m['Test_nonhate_recall']:.4f}")

print("\n=== deltas vs CLIP (final-epoch, TEST) : Pareto vs rotation ===")
c = OUT["CLIP"]["final_mean"]
for arm in ("frozenQwen", "LoRAQwen"):
    m = OUT[arm]["final_mean"]
    print(f"{arm:11s} dAcc={m['Test_acc']-c['Test_acc']:+.4f}  dmF1={m['Test_mF1']-c['Test_mF1']:+.4f}  "
          f"d_hate_rec={m['Test_hate_recall']-c['Test_hate_recall']:+.4f}  "
          f"d_nonhate_rec={m['Test_nonhate_recall']-c['Test_nonhate_recall']:+.4f}")

print("\n=== Q3 val-selection: per-seed val-sel epoch, its TEST acc, vs final & oracle-best ===")
for arm in ARMS:
    print(f"-- {arm} --")
    for s in SEEDS:
        d = OUT[arm]["seeds"][s]
        print(f"  seed{s}: valsel_ep={d['valsel_ep']:2d} (Val_acc={d['valsel'].get('Val_acc'):.4f}) "
              f"-> Test_acc={d['valsel']['Test_acc']:.4f} | final_ep{d['final_ep']} Test_acc={d['final']['Test_acc']:.4f} "
              f"| oracle_best_ep={d['best_test_ep']:2d} Test_acc={d['best_test_acc']:.4f} "
              f"| valsel_regret={d['best_test_acc']-d['valsel']['Test_acc']:+.4f}")

print("\n=== val-selected TEST means + delta vs CLIP ===")
cv = OUT["CLIP"]["valsel_mean"]
for arm in ARMS:
    m = OUT[arm]["valsel_mean"]
    d = f" dAcc={m['Test_acc']-cv['Test_acc']:+.4f} dmF1={m['Test_mF1']-cv['Test_mF1']:+.4f}" if arm != "CLIP" else ""
    print(f"{arm:11s} valsel Test acc={m['Test_acc']:.4f} mF1={m['Test_mF1']:.4f}{d}")

outp = "/data/jehc223/RGCL/scripts/analysis/b3_zh_lora_trainlog_parse_out.json"
with open(outp, "w") as f:
    json.dump(OUT, f, indent=1, default=float)
print("\nwrote", outp)
