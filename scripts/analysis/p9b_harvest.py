#!/usr/bin/env python
"""P9b harvest: collect D3 / C3' x {ZH,EN} x seeds{0,1,2}, dev+test, mlp+knn read-outs.

Sources
  - mlp DEV : logging/lora_p9/<train_dir>/eval_results.json  (final-epoch eval_accuracy/eval_f1)
  - mlp TEST: logging/lora_p9/predict/<train_dir>_test/eval_results.json
  - knn dev+test: scripts/analysis/p9b_knn_out/<tag>.json  (dev/test acc+macro_f1)

Prints a markdown table + per-cell mean+/-std + the pre-registered verdict.
Floors (protocol-matched): EN test 0.7847, ZH test 0.8537.
"""
import json, os, math

ROOT = "/data/jehc223/RGCL"
EN_FLOOR, ZH_FLOOR = 0.7847, 0.8537

# cell -> (train_dir_basename, knn_tag), seeds appended
CELLS = {
    "D3_ZH":  ("qwen25vl_mhc_zh_d3f4_s{}",       "p9d3_zh_s{}"),
    "C3p_ZH": ("qwen25vl_mhc_zh_c3primef4_s{}",  "p9c3p_zh_s{}"),
    "D3_EN":  ("qwen25vl_mhc_d3f4_s{}",          "p9d3_en_s{}"),
    "C3p_EN": ("qwen25vl_mhc_c3primef4_s{}",     "p9c3p_en_s{}"),
}
SEEDS = [0, 1, 2]


def rj(p):
    try:
        with open(p) as f:
            return json.load(f)
    except Exception:
        return None


def get(cell, seed):
    tdir, tag = CELLS[cell]
    tdir = tdir.format(seed); tag = tag.format(seed)
    mlp_dev = rj(os.path.join(ROOT, "logging/lora_p9", tdir, "eval_results.json"))
    mlp_test = rj(os.path.join(ROOT, "logging/lora_p9/predict", tdir + "_test", "eval_results.json"))
    knn = rj(os.path.join(ROOT, "scripts/analysis/p9b_knn_out", tag + ".json"))
    def acc(d, k="eval_accuracy"): return None if not d else d.get(k)
    def f1(d, k="eval_f1"): return None if not d else d.get(k)
    return {
        "mlp_dev":  acc(mlp_dev),  "mlp_dev_f1":  f1(mlp_dev),
        "mlp_test": acc(mlp_test), "mlp_test_f1": f1(mlp_test),
        "knn_dev":  (knn or {}).get("dev", {}).get("acc"),
        "knn_dev_f1": (knn or {}).get("dev", {}).get("macro_f1"),
        "knn_test": (knn or {}).get("test", {}).get("acc"),
        "knn_test_f1": (knn or {}).get("test", {}).get("macro_f1"),
    }


def fmt(x):
    return "  NA " if x is None else "{:.4f}".format(x)


def mean_std(xs):
    xs = [x for x in xs if x is not None]
    if not xs: return (None, None, 0)
    m = sum(xs) / len(xs)
    sd = math.sqrt(sum((x - m) ** 2 for x in xs) / len(xs)) if len(xs) > 1 else 0.0
    return (m, sd, len(xs))


def main():
    data = {c: {s: get(c, s) for s in SEEDS} for c in CELLS}

    print("\n### P9b raw per-seed (acc)\n")
    hdr = "| cell | seed | mlp_dev | mlp_test | knn_dev | knn_test |"
    print(hdr); print("|" + "---|" * 6)
    for c in CELLS:
        for s in SEEDS:
            d = data[c][s]
            print("| {} | s{} | {} | {} | {} | {} |".format(
                c, s, fmt(d["mlp_dev"]), fmt(d["mlp_test"]), fmt(d["knn_dev"]), fmt(d["knn_test"])))

    print("\n### P9b cell means (mean+/-std, n)\n")
    print("| cell | mlp_dev | mlp_test | knn_dev | knn_test |")
    print("|" + "---|" * 5)
    agg = {}
    for c in CELLS:
        row = {}
        for k in ("mlp_dev", "mlp_test", "knn_dev", "knn_test"):
            row[k] = mean_std([data[c][s][k] for s in SEEDS])
        agg[c] = row
        def cell(ms):
            m, sd, n = ms
            return "NA" if m is None else "{:.4f}+/-{:.3f} (n={})".format(m, sd, n)
        print("| {} | {} | {} | {} | {} |".format(
            c, cell(row["mlp_dev"]), cell(row["mlp_test"]), cell(row["knn_dev"]), cell(row["knn_test"])))

    print("\n### VERDICT (pre-registered bar)\n")
    for ds, floor in (("ZH", ZH_FLOOR), ("EN", EN_FLOOR)):
        d3 = "D3_" + ds; c3 = "C3p_" + ds
        d3_knn = [data[d3][s]["knn_test"] for s in SEEDS]
        d3_mlp = [data[d3][s]["mlp_test"] for s in SEEDS]
        c3_knn = [data[c3][s]["knn_test"] for s in SEEDS]
        m_d3knn, sd, n = mean_std(d3_knn)
        m_d3mlp, _, _ = mean_std(d3_mlp)
        m_c3knn, _, _ = mean_std(c3_knn)
        print("## {} (floor {:.4f})".format(ds, floor))
        if m_d3knn is None:
            print("  D3-knn: incomplete\n"); continue
        # criterion 1: D3-knn beats floor by >1.5pt mean AND >=2/3 seeds
        delta = m_d3knn - floor
        seeds_pass = sum(1 for x in d3_knn if x is not None and x - floor >= 0.015)
        c1 = (delta >= 0.015) and (seeds_pass >= 2)
        # criterion 2: D3-knn >= D3-mlp - 1pt
        c2 = (m_d3mlp is not None) and (m_d3knn >= m_d3mlp - 0.01)
        print("  D3-knn test  = {:.4f}+/-{:.3f} (n={})  delta_vs_floor = {:+.4f} ({:+.1f}pt)".format(
            m_d3knn, sd, n, delta, delta * 100))
        print("  seeds >= floor+1.5pt: {}/3   per-seed: {}".format(
            seeds_pass, [None if x is None else round(x, 4) for x in d3_knn]))
        print("  D3-mlp test  = {}   (D3-knn >= D3-mlp-1pt ? {})".format(
            "NA" if m_d3mlp is None else "{:.4f}".format(m_d3mlp), c2))
        print("  C1 (>+1.5pt mean & >=2/3): {}   C2 (knn>=mlp-1): {}   => {}".format(
            c1, c2, "PASS" if (c1 and c2) else "FAIL"))
        if m_c3knn is not None:
            mech = m_d3knn - m_c3knn
            print("  MECHANISM D3-knn - C3'-knn = {:+.4f} ({:+.1f}pt)  [rgcl-term effect on memory]".format(
                mech, mech * 100))
        print()


if __name__ == "__main__":
    main()
