#!/usr/bin/env python
"""P2 failure audit: is the MLLM's INCOMPARABLE drop discriminative (drops the
neighbours that would MISVOTE, i.e. opposite gold label) or indiscriminate
(= random dropping)? Plus a 10-query kept/dropped-neighbour example dump.

Reads the caches + verdicts under scripts/analysis/p2_out. CPU only, read-only.
"""
import json
import os
import sys
from collections import Counter

ROOT = "/data/jehc223/RGCL"
OUT = os.path.join(ROOT, "scripts/analysis/p2_out")
TOPK = 20
SEEDS = {"MHC": [0, 1, 2, 3], "MHC_zh": [0, 1, 2, 3, 4]}


def load_verdicts(ds):
    v = {}
    for line in open(os.path.join(OUT, "verdicts_{}.jsonl".format(ds))):
        line = line.strip()
        if line:
            r = json.loads(line)
            v[(r["query_id"], r["neighbor_id"])] = (r["verdict"], r.get("reason", ""))
    return v


def load_v2_cards(ds):
    cards = {}
    for sp in ["train", "dev_seen", "test_seen"]:
        p = os.path.join(ROOT, "data/Archive", ds, "v2",
                         "{}_Qwen2.5-VL-7B-Instruct_archive.jsonl".format(sp))
        for line in open(p):
            line = line.strip()
            if line:
                r = json.loads(line)
                a = r.get("archive") or {}
                cards[r["id"]] = "tg={} mech={} expl={} :: {}".format(
                    a.get("target_groups") or [], a.get("mechanism") or [],
                    a.get("explicitness"),
                    (a.get("neutral_summary") or "").strip()[:90])
    return cards


def audit(ds):
    verd = load_verdicts(ds)
    # discriminativeness: among gated queries' top-20 neighbours, drop rate split
    # by whether the neighbour's label == query gold (a "correct-vote" neighbour)
    # vs != gold (a "wrong-vote" neighbour). A useful judge drops wrong-vote more.
    n_same = n_same_drop = n_opp = n_opp_drop = 0
    for seed in SEEDS[ds]:
        cache = json.load(open(os.path.join(OUT, "cache_{}_s{}.json".format(ds, seed))))
        for s in cache["samples"]:
            if not s["gated"]:
                continue
            gold = s["label"]
            for nb in s["neighbors"][:TOPK]:
                nid, _, nlab = nb
                dropped = verd.get((s["id"], nid), ("UNSURE", ""))[0] == "INCOMPARABLE"
                if nlab == gold:
                    n_same += 1; n_same_drop += int(dropped)
                else:
                    n_opp += 1; n_opp_drop += int(dropped)
    return dict(
        n_same=n_same, same_drop_rate=n_same_drop / max(1, n_same),
        n_opp=n_opp, opp_drop_rate=n_opp_drop / max(1, n_opp))


def examples(ds, k=10):
    verd = load_verdicts(ds)
    cards = load_v2_cards(ds)
    cache = json.load(open(os.path.join(OUT, "cache_{}_s0.json".format(ds))))
    gated = [s for s in cache["samples"] if s["gated"]]
    gated.sort(key=lambda s: s["id"])
    out = []
    for s in gated[:k]:
        gold = s["label"]
        rows = []
        for nb in s["neighbors"][:TOPK]:
            nid, sim, nlab = nb
            v, reason = verd.get((s["id"], nid), ("UNSURE", ""))
            rows.append(dict(nid=nid, sim=round(sim, 3), nlab=nlab, verdict=v,
                             correct_vote=int(nlab == gold), reason=reason[:120]))
        kept = [r for r in rows if r["verdict"] != "INCOMPARABLE"]
        out.append(dict(qid=s["id"], gold=gold, n_kept=len(kept),
                        card=cards.get(s["id"], ""), rows=rows))
    return out


if __name__ == "__main__":
    for ds in ["MHC", "MHC_zh"]:
        a = audit(ds)
        print("\n===== {} discriminativeness (gated top-20 neighbours) =====".format(ds))
        print("  correct-vote neighbours (label==gold): n={} dropped {:.1%}".format(
            a["n_same"], a["same_drop_rate"]))
        print("  wrong-vote   neighbours (label!=gold): n={} dropped {:.1%}".format(
            a["n_opp"], a["opp_drop_rate"]))
        print("  -> lift (wrong-drop - correct-drop) = {:+.1%} "
              "(positive = useful; ~0 = indiscriminate/random)".format(
                  a["opp_drop_rate"] - a["same_drop_rate"]))
    if len(sys.argv) > 1 and sys.argv[1] == "--examples":
        for ds in ["MHC", "MHC_zh"]:
            print("\n########## {} — 10 gated-query examples (seed 0) ##########".format(ds))
            for e in examples(ds):
                print("\nQUERY {} gold={} kept={}/20\n  card: {}".format(
                    e["qid"], e["gold"], e["n_kept"], e["card"]))
                for r in e["rows"][:8]:
                    tag = "KEPT " if r["verdict"] != "INCOMPARABLE" else "DROP "
                    cv = "correct-vote" if r["correct_vote"] else "WRONG-vote"
                    print("    {}{:12s} sim={:.3f} nlab={} {} | {} | {}".format(
                        tag, r["nid"], r["sim"], r["nlab"], cv, r["verdict"], r["reason"]))
