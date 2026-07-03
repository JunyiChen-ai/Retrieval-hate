"""Parse train_consensus job logs -> final val-selected test metrics.

Protocol (matches src/run_rac.py selection): within the FINAL EM round
(seg_mode=full has a single round), pick the epoch with the best
Val_Retrieval acc (ties broken by roc), only epochs >= warmup (5); report
Test_Retrieval macroF1 / acc / roc at that epoch.

Usage: python scripts/analysis/parse_consensus_results.py slurm/logs/foo_123.out [...]
"""

import re
import sys

RE_ROUND = re.compile(r"\[EM\] ===== round (\d+)/(\d+)")
RE_VAL = re.compile(
    r"Val_Retrieval Epoch\s+(\d+) macroF1: ([\d.]+).* acc: ([\d.]+) roc: ([\d.]+)")
RE_TEST = re.compile(
    r"Test_Retrieval Epoch\s+(\d+) macroF1: ([\d.]+).* acc: ([\d.]+) roc: ([\d.]+)")
RE_CFG = re.compile(r"\[train_cons\] (DATASET=\S+.*)")

WARMUP = 5


def parse(path):
    cfg = ""
    rounds = [{"val": {}, "test": {}}]
    with open(path, errors="replace") as f:
        for line in f:
            m = RE_CFG.search(line)
            if m:
                cfg = m.group(1)
            m = RE_ROUND.search(line)
            if m and int(m.group(1)) > 1:
                rounds.append({"val": {}, "test": {}})
            m = RE_VAL.search(line)
            if m:
                ep = int(m.group(1))
                rounds[-1]["val"][ep] = tuple(float(x) for x in m.group(2, 3, 4))
            m = RE_TEST.search(line)
            if m:
                ep = int(m.group(1))
                rounds[-1]["test"][ep] = tuple(float(x) for x in m.group(2, 3, 4))
    final = rounds[-1]
    if not final["val"]:
        return cfg, None, None, len(rounds)
    eligible = [e for e in final["val"] if e >= WARMUP] or list(final["val"])
    # best val acc, tie -> best roc, then earliest epoch (matches '>' update rule)
    best_ep = min(eligible, key=lambda e: (-final["val"][e][1], -final["val"][e][2], e))
    return cfg, best_ep, final["test"].get(best_ep), len(rounds)


def main():
    print("{:<44} {:>5} {:>8} {:>8} {:>8}".format(
        "log", "epoch", "T-maF1", "T-acc", "T-roc"))
    for path in sys.argv[1:]:
        cfg, ep, test, nr = parse(path)
        name = path.split("/")[-1]
        if test is None:
            print("{:<44} PARSE FAILURE ({} rounds) cfg: {}".format(name, nr, cfg))
            continue
        print("{:<44} {:>5} {:>8.4f} {:>8.4f} {:>8.4f}   rounds={} {}".format(
            name, ep, test[0], test[1], test[2], nr, cfg[:90]))


if __name__ == "__main__":
    main()
