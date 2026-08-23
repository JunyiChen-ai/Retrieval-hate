#!/usr/bin/env python
"""Render logging/runs/rgcl_ablation/analysis.json into markdown tables."""
import json
import os

A = json.load(open("/home/jehc223/Retrieval-hate/logging/runs/rgcl_ablation/analysis.json"))
G = A["grid"]

ENCS = ["CLIP", "QWEN", "LORA"]
DSS = ["HateMM", "MHC", "MHC_zh", "ImpliHateVid"]


def fmt(a):
    if a is None:
        return "n/a"
    return "{:.4f}±{:.4f}".format(a["mean"], a["std"])


print("### 全网格数表(主列 = test macro-F1,mean±std over 3 seeds;括号内 = val macro-F1)\n")
print("| encoder | dataset | loss | I1 head test | I1 head val | I2 kNN test | I2 kNN val |")
print("|---|---|---|---|---|---|---|")
for enc in ENCS:
    for ds in DSS:
        for loss in ["L1", "L2", "L3"]:
            k1 = "|".join([enc, ds, loss, "I1"])
            k2 = "|".join([enc, ds, loss, "I2"])
            if k1 not in G:
                continue
            print("| {} | {} | {} | {} | {} | {} | {} |".format(
                enc, ds, loss,
                fmt(G[k1]["test"]), fmt(G[k1]["val"]),
                fmt(G[k2]["test"]), fmt(G[k2]["val"])))

print()
for name, title in [("knn_readout", "kNN 读出贡献 (I2 − I1)"),
                    ("retrieval_guidance", "检索引导贡献 (L3 − L2)"),
                    ("contrastive_reg", "对比正则贡献 (L2 − L1)")]:
    for split in ["test", "val"]:
        c = A["components"]["{}|{}".format(name, split)]
        print("#### {} — **{}** [{}]  支持格 {}/{} (需 ≥{}), 跨格均值 {:+.4f}".format(
            title, c["verdict"], split.upper(), c["support"], c["cells"], c["need"],
            c["mean_over_cells"]))
        print()
        print("| 格 | 均值差 | seed 差 | 支持? |")
        print("|---|---|---|---|")
        for r in c["rows"]:
            print("| {} | {:+.4f} | {} | {} |".format(
                r["cell"], r["mean"],
                " ".join("{:+.4f}".format(d) for d in r["diffs"]),
                "YES" if r["supports"] else "no"))
        print()

for split in ["test", "val"]:
    it = A["interaction"][split]
    print("#### 交互项 [(L3,I2)−(L3,I1)] − [(L1,I2)−(L1,I1)] — **{}** [{}] 支持格 {}/{} (需 ≥{})".format(
        it["verdict"], split.upper(), it["support"], it["cells"], it["need"]))
    print()
    print("| 格 | kNN 增益 @L3 | kNN 增益 @L1 | 差之差 | seed 差之差 | 支持? |")
    print("|---|---|---|---|---|---|")
    for r in it["rows"]:
        print("| {} | {:+.4f} | {:+.4f} | {:+.4f} | {} | {} |".format(
            r["cell"], r["knn_gain_at_L3"], r["knn_gain_at_L1"], r["dd_mean"],
            " ".join("{:+.4f}".format(d) for d in r["dd"]),
            "YES" if r["supports"] else "no"))
    print()

if A["missing"]:
    print("MISSING/FAILED runs: {}".format(A["missing"]))
