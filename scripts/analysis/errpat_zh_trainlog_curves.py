#!/usr/bin/env python
"""ERRPAT MHC-ZH: bit-exact per-epoch curve extraction from job 13150 trainlogs.

Primary source (NOT a proxy): slurm/logs/enc3s_MHC_zh_Qwen2.5-VL-7B-Instruct-LoRA_HF_seed{0,1,2}_13150.trainlog

Quantifies the val-selected <-> final-epoch protocol question:
  - deployed val-sel rule: warmup>=5, argmax Val_Retrieval acc, roc tie-break (earliest on full tie)
  - final-epoch: epoch 29
  - dev plateau structure: how many epochs sit within k dev-items of the argmax, and the
    spread of their TEST accs = the selection lottery.
"""
import json
import re
import sys
from pathlib import Path

ROOT = Path("/data/jehc223/RGCL")
LOGS = ROOT / "slurm/logs"
OUT = ROOT / "scripts/analysis/errpat_zh_curves_OUT.json"

TPL = "enc3s_MHC_zh_Qwen2.5-VL-7B-Instruct-LoRA_HF_seed{s}_13150.trainlog"

# Val_Retrieval Epoch  0 acc: 0.8462 roc: 0.9064 pre: 0.7667 recall: 0.8214 f1: 0.7931
RE_VAL = re.compile(
    r"^Val_Retrieval Epoch\s+(\d+) acc: ([\d.]+) roc: ([\d.]+) pre: ([\d.]+) recall: ([\d.]+) f1: ([\d.]+)\s*$")
RE_TEST = re.compile(
    r"^Test_Retrieval Epoch\s+(\d+) acc: ([\d.]+) roc: ([\d.]+) pre: ([\d.]+) recall: ([\d.]+) f1: ([\d.]+)\s*$")
# Val_Retrieval Epoch  0 macroF1: 0.8353 macroP: 0.8313 macroR: 0.8407 acc: 0.8462 roc: 0.9064
RE_VAL_M = re.compile(
    r"^Val_Retrieval Epoch\s+(\d+) macroF1: ([\d.]+) macroP: ([\d.]+) macroR: ([\d.]+) acc: ([\d.]+) roc: ([\d.]+)\s*$")
RE_TEST_M = re.compile(
    r"^Test_Retrieval Epoch\s+(\d+) macroF1: ([\d.]+) macroP: ([\d.]+) macroR: ([\d.]+) acc: ([\d.]+) roc: ([\d.]+)\s*$")

N_DEV, N_TEST = 78, 149
WARMUP = 5


def parse(seed):
    p = LOGS / TPL.format(s=seed)
    rows = {}
    with open(p) as f:
        for raw in f:
            # tqdm writes carriage-return chunks onto the same physical line
            for line in raw.replace("\r", "\n").split("\n"):
                line = line.strip()
                for rx, tag in ((RE_VAL, "val"), (RE_TEST, "test")):
                    m = rx.match(line)
                    if m:
                        e = int(m.group(1))
                        rows.setdefault(e, {})[tag] = dict(
                            acc=float(m.group(2)), roc=float(m.group(3)),
                            pre=float(m.group(4)), recall=float(m.group(5)),
                            f1=float(m.group(6)))
                for rx, tag in ((RE_VAL_M, "val"), (RE_TEST_M, "test")):
                    m = rx.match(line)
                    if m:
                        e = int(m.group(1))
                        d = rows.setdefault(e, {}).setdefault(tag, {})
                        d["macroF1"] = float(m.group(2))
                        d["macroP"] = float(m.group(3))
                        d["macroR"] = float(m.group(4))
                        # cross-check the acc/roc reported on both lines
                        assert abs(d["acc"] - float(m.group(5))) < 1e-9, (seed, e, tag)
                        assert abs(d["roc"] - float(m.group(6))) < 1e-9, (seed, e, tag)
    assert sorted(rows) == list(range(30)), (seed, sorted(rows))
    for e in rows:
        assert set(rows[e]) == {"val", "test"}, (seed, e)
        for t in ("val", "test"):
            assert "macroF1" in rows[e][t], (seed, e, t)
    return [rows[e] for e in range(30)]


def val_sel_epoch(cur, warmup=WARMUP):
    """Deployed rule: among epochs >= warmup, argmax val acc, roc tie-break, earliest wins."""
    cand = list(range(warmup, len(cur)))
    best = max(cand, key=lambda e: (cur[e]["val"]["acc"], cur[e]["val"]["roc"], -e))
    return best


def main():
    res = {"provenance": {}, "seeds": {}}
    curves = {}
    for s in (0, 1, 2):
        p = LOGS / TPL.format(s=s)
        import hashlib
        res["provenance"][f"seed{s}"] = {
            "path": str(p),
            "sha256": hashlib.sha256(p.read_bytes()).hexdigest(),
        }
        curves[s] = parse(s)

    for s in (0, 1, 2):
        cur = curves[s]
        vs = val_sel_epoch(cur)
        fe = 29
        vacc = [cur[e]["val"]["acc"] for e in range(30)]
        tacc = [cur[e]["test"]["acc"] for e in range(30)]
        tmf1 = [cur[e]["test"]["macroF1"] for e in range(30)]
        vmax = max(vacc[WARMUP:])
        # dev-item tolerance bands: k dev items = k/78 acc
        bands = {}
        for k in (0, 1, 2, 3):
            tol = k / N_DEV + 1e-9
            eps = [e for e in range(WARMUP, 30) if vacc[e] >= vmax - tol]
            bands[f"within_{k}_dev_items"] = {
                "n_epochs": len(eps),
                "epochs": eps,
                "test_acc_min": round(min(tacc[e] for e in eps), 4),
                "test_acc_max": round(max(tacc[e] for e in eps), 4),
                "test_acc_mean": round(sum(tacc[e] for e in eps) / len(eps), 4),
                "test_acc_span_items": round((max(tacc[e] for e in eps) - min(tacc[e] for e in eps)) * N_TEST, 2),
                "test_mf1_min": round(min(tmf1[e] for e in eps), 4),
                "test_mf1_max": round(max(tmf1[e] for e in eps), 4),
            }
        res["seeds"][f"seed{s}"] = {
            "val_sel_epoch": vs,
            "val_sel_val_acc": cur[vs]["val"]["acc"],
            "val_sel_val_roc": cur[vs]["val"]["roc"],
            "val_sel_test_acc": cur[vs]["test"]["acc"],
            "val_sel_test_mF1": cur[vs]["test"]["macroF1"],
            "final_epoch": fe,
            "final_val_acc": cur[fe]["val"]["acc"],
            "final_test_acc": cur[fe]["test"]["acc"],
            "final_test_mF1": cur[fe]["test"]["macroF1"],
            "gap_test_acc_final_minus_valsel": round(cur[fe]["test"]["acc"] - cur[vs]["test"]["acc"], 4),
            "gap_test_acc_items": round((cur[fe]["test"]["acc"] - cur[vs]["test"]["acc"]) * N_TEST, 2),
            "gap_test_mF1_final_minus_valsel": round(cur[fe]["test"]["macroF1"] - cur[vs]["test"]["macroF1"], 4),
            "dev_argmax_val_acc": vmax,
            "best_possible_test_acc_after_warmup": max(tacc[WARMUP:]),
            "best_possible_test_epoch": WARMUP + tacc[WARMUP:].index(max(tacc[WARMUP:])),
            "worst_test_acc_after_warmup": min(tacc[WARMUP:]),
            "dev_plateau_bands": bands,
            "val_acc_curve": vacc,
            "val_roc_curve": [cur[e]["val"]["roc"] for e in range(30)],
            "test_acc_curve": tacc,
            "test_mF1_curve": tmf1,
            "test_recall_curve": [cur[e]["test"]["recall"] for e in range(30)],
            "test_precision_curve": [cur[e]["test"]["pre"] for e in range(30)],
        }

    # 3-seed means
    def mean(key):
        return round(sum(res["seeds"][f"seed{s}"][key] for s in (0, 1, 2)) / 3, 4)

    res["floor_3seed"] = {
        "val_sel_test_acc_mean": mean("val_sel_test_acc"),
        "val_sel_test_mF1_mean": mean("val_sel_test_mF1"),
        "final_test_acc_mean": mean("final_test_acc"),
        "final_test_mF1_mean": mean("final_test_mF1"),
        "val_sel_epochs": [res["seeds"][f"seed{s}"]["val_sel_epoch"] for s in (0, 1, 2)],
        "gap_acc_mean": mean("gap_test_acc_final_minus_valsel"),
        "gap_mF1_mean": mean("gap_test_mF1_final_minus_valsel"),
    }

    # "protocol lottery": for every epoch that is a legal argmax under a +-k dev-item
    # perturbation of the 78-item dev set, what is the resulting 3-seed mean test acc?
    OUT.write_text(json.dumps(res, indent=1))
    print(json.dumps(res["floor_3seed"], indent=1))
    for s in (0, 1, 2):
        d = res["seeds"][f"seed{s}"]
        print(f"\n--- seed{s} ---")
        print(f"  val-sel epoch {d['val_sel_epoch']:2d} (dev acc {d['val_sel_val_acc']:.4f}) "
              f"-> test {d['val_sel_test_acc']:.4f} / mF1 {d['val_sel_test_mF1']:.4f}")
        print(f"  final  epoch 29 (dev acc {d['final_val_acc']:.4f}) "
              f"-> test {d['final_test_acc']:.4f} / mF1 {d['final_test_mF1']:.4f}")
        print(f"  gap final-valsel = {d['gap_test_acc_final_minus_valsel']:+.4f} acc "
              f"({d['gap_test_acc_items']:+.2f} items) / {d['gap_test_mF1_final_minus_valsel']:+.4f} mF1")
        for k in (0, 1, 2, 3):
            b = d["dev_plateau_bands"][f"within_{k}_dev_items"]
            print(f"  dev-tol {k} item(s): {b['n_epochs']:2d} epochs {b['epochs']} "
                  f"test acc [{b['test_acc_min']:.4f},{b['test_acc_max']:.4f}] "
                  f"span {b['test_acc_span_items']:.0f} items")
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
