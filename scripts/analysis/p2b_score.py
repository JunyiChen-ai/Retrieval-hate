#!/usr/bin/env python
"""P2b — score the TRAIN-side selectivity benchmark into a leaderboard.

For each config's train verdicts (tb_verdicts_<ds>_<config>.jsonl, carrying the
correct_vote pass-through), compute drop_rate(wrong-vote), drop_rate(correct-vote),
overall drop-rate, selectivity lift = wrong-drop - correct-drop, and the verdict
distribution. Evaluate the pre-registered promotion bar (EN lift >= +10pt AND EN
drop-rate in [15%,50%] AND ZH lift > 0). Read-only; prints + writes p2b_trainbench.json.
"""
import argparse
import glob
import json
import os
import re
from collections import Counter

ROOT = "/data/jehc223/RGCL"
OUT = os.path.join(ROOT, "scripts/analysis/p2_out")
CONFIG_DESC = {
    "C0": "7B · archive · orig (P2 ref)", "C1": "7B · archive · flip",
    "C2": "7B · archive+transcript · orig", "C3": "7B · archive+transcript · flip",
    "C4": "32B · archive+transcript · flip", "C5": "32B · archive+transcript · orig",
    "C6": "72B(bnb4) · archive+transcript · flip", "C7": "72B(bnb4) · archive+transcript · orig"}


def score_file(path):
    n_c = n_c_drop = n_w = n_w_drop = 0
    vc = Counter()
    fb = 0
    for line in open(path):
        line = line.strip()
        if not line:
            continue
        r = json.loads(line)
        drop = int(r["verdict"] == "INCOMPARABLE")
        vc[r["verdict"]] += 1
        fb += int(r.get("fallback", False))
        if r.get("correct_vote") == 1:
            n_c += 1; n_c_drop += drop
        else:
            n_w += 1; n_w_drop += drop
    cdr = n_c_drop / max(1, n_c)
    wdr = n_w_drop / max(1, n_w)
    n = n_c + n_w
    return dict(n=n, n_correct=n_c, n_wrong=n_w,
                drop_rate=(n_c_drop + n_w_drop) / max(1, n),
                correct_drop=cdr, wrong_drop=wdr, lift=wdr - cdr,
                verdicts=dict(vc), fallback=fb)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_json", default=os.path.join(OUT, "p2b_trainbench.json"))
    args = ap.parse_args()
    files = sorted(glob.glob(os.path.join(OUT, "tb_verdicts_*.jsonl")))
    board = {}
    for f in files:
        m = re.match(r"tb_verdicts_(MHC(?:_zh)?)_(C\d)\.jsonl", os.path.basename(f))
        if not m:
            continue
        ds, cfg = m.group(1), m.group(2)
        board.setdefault(cfg, {})[ds] = score_file(f)

    print("\n===== P2b TRAIN-side selectivity leaderboard =====")
    print("bar: EN lift >= +10.0pt AND EN drop-rate in [15%,50%] AND ZH lift > 0\n")
    hdr = "{:4s} {:34s} | {:>26s} | {:>26s} | {}".format(
        "cfg", "description", "EN drop% (corr/wrong)  lift", "ZH drop% (corr/wrong)  lift",
        "PROMOTE?")
    print(hdr); print("-" * len(hdr))
    promo = {}
    for cfg in sorted(board):
        en = board[cfg].get("MHC"); zh = board[cfg].get("MHC_zh")
        def cell(s):
            if not s:
                return "{:>26s}".format("(missing)")
            return "{:5.1f}% ({:4.1f}/{:4.1f})  {:+5.1f}pt".format(
                100 * s["drop_rate"], 100 * s["correct_drop"], 100 * s["wrong_drop"],
                100 * s["lift"])
        ok = (en and zh and 100 * en["lift"] >= 10.0
              and 0.15 <= en["drop_rate"] <= 0.50 and zh["lift"] > 0)
        promo[cfg] = bool(ok)
        print("{:4s} {:34s} | {} | {} | {}".format(
            cfg, CONFIG_DESC.get(cfg, "?"), cell(en), cell(zh),
            "YES" if ok else "no"))
    # winner = highest EN lift among promoted (tie -> ZH lift -> cheaper/lower Cn)
    winners = [c for c in board if promo.get(c)]
    winner = None
    if winners:
        winner = max(winners, key=lambda c: (board[c]["MHC"]["lift"],
                                             board[c]["MHC_zh"]["lift"], -int(c[1:])))
    print("\nPROMOTED:", winner if winner else "NONE (P2b dies train-side)")
    with open(args.out_json, "w") as f:
        json.dump(dict(board=board, promote=promo, winner=winner), f, indent=1)
    print("wrote", args.out_json)


if __name__ == "__main__":
    main()
