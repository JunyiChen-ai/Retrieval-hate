"""CAD reporting helper: gate accounting, API spend, and three rewrite examples.

Read-only; touches no split but train, and no metric.
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, "idea-stage", "desc_channel"))
from defect import load_gt  # noqa: E402
from cadgen import load_done  # noqa: E402
from gates import norm, run as run_gates  # noqa: E402


def excerpt(s, n=340):
    s = norm(s)
    return s if len(s) <= n else s[:n] + " ..."


def main():
    gt = load_gt(ROOT)
    g = run_gates(ROOT, verbose=False)
    acc, counts = g["accepted"], g["counts"]
    print("== gate accounting ==")
    print(json.dumps(counts, indent=1))
    print("\ndrops by gate:")
    byg = {}
    for d in g["drops"]:
        byg.setdefault(d["gate"].split(":")[0], []).append(d)
    for k, v in sorted(byg.items()):
        print("  %-6s %3d   e.g. %s" % (k, len(v),
                                        ", ".join("%s(%s)" % (x["id"], x["detail"])
                                                  for x in v[:3])))

    rows = load_done()
    ti = sum((r.get("usage") or {}).get("prompt_tokens", 0) for r in rows.values())
    to = sum((r.get("usage") or {}).get("completion_tokens", 0) for r in rows.values())
    print("\n== API spend ==")
    print("rows requested=%d  tokens in=%d out=%d  approx CNY %.3f (qwen-plus list)"
          % (len(rows), ti, to, ti / 1000 * 0.0008 + to / 1000 * 0.002))

    if acc:
        rat = sorted((v["ratio"], k) for k, v in acc.items())
        ned = sorted((v["n_edits"], k) for k, v in acc.items())
        print("\n== accepted rows ==")
        print("N=%d  ratio min %.3f median %.3f max %.3f"
              % (len(acc), rat[0][0], rat[len(rat) // 2][0], rat[-1][0]))
        print("n_edits min %d median %d max %d"
              % (ned[0][0], ned[len(ned) // 2][0], ned[-1][0]))
        # three examples spread over the edit-count distribution
        picks = [ned[len(ned) // 6][1], ned[len(ned) // 2][1], ned[-1][1]]
        for vid in picks:
            print("\n---- %s  (n_edits=%d, ratio=%.2f) ----"
                  % (vid, acc[vid]["n_edits"], acc[vid]["ratio"]))
            print("ORIGINAL : %s" % excerpt(gt[vid]["text"]))
            print("REWRITTEN: %s" % excerpt(acc[vid]["rewritten"]))


if __name__ == "__main__":
    main()
