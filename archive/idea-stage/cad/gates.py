"""CAD step 2 -- frozen quality gates (idea-stage/CAD_FREEZE.md section 3).

Label-free and metric-free: every gate reads only the original transcript and the
rewritten transcript. No model, no GPU, no test data, no downstream metric.

  G0  eligibility (applied in cadgen.targets, before any API call):
      original transcript has >= 40 non-whitespace characters.
  G1  the API returned a parseable rewrite (parse == "ok").
  G2  length ratio  len(rewritten.strip()) / len(original.strip())  in [0.40, 1.20].
  G3  the rewritten text contains no term from the identity-attack word list below
      (case-insensitive, word-boundary match).
  G4  the rewritten text is not byte-identical to the original after whitespace
      normalisation (an unchanged rewrite relabelled 0 would be a wrong gold label).

  python gates.py            # report + write accepted.json
"""
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, "idea-stage", "desc_channel"))
from defect import load_gt  # noqa: E402

REW = os.path.join(HERE, "rewrites_train_hate.jsonl")
ACC = os.path.join(HERE, "accepted.json")

RATIO_LO = 0.40
RATIO_HI = 1.20

# Identity-attack word list for G3. Purely a residual-attack detector applied to the
# MODEL OUTPUT: any hit means the rewrite failed to remove the attack, so the row is
# dropped. It is never used as a feature, a label, or a training signal.
ATTACK_TERMS = [
    # ethnic / racial slurs
    "nigger", "niggers", "nigga", "niggas", "coon", "coons", "spic", "spics",
    "wetback", "wetbacks", "chink", "chinks", "gook", "gooks", "paki", "pakis",
    "kike", "kikes", "yid", "yids", "raghead", "ragheads", "towelhead", "towelheads",
    "sandnigger", "beaner", "beaners", "wop", "wops", "kaffir", "kaffirs",
    "abo", "abos", "gypo", "gyppo", "zipperhead", "jungle bunny", "porch monkey",
    "half-breed", "mudblood", "mud people", "subhuman", "subhumans", "untermensch",
    # religious / national hate
    "christ-killer", "christ killers", "mudslime", "muzzie", "muzzies",
    "jewish vermin", "islamic vermin",
    # gender / sexuality / disability slurs
    "faggot", "faggots", "fag", "fags", "dyke", "dykes", "tranny", "trannies",
    "shemale", "shemales", "retard", "retards", "retarded", "spastic", "mongoloid",
    "cripple", "cripples",
    # dehumanising / exterminationist framing
    "vermin", "cockroaches", "parasites", "infestation", "rats and", "gas the",
    "gas them", "gas chamber for", "exterminate them", "exterminate the",
    "kill all the", "kill all ", "hang them all", "back to the trees",
    "white power", "heil hitler", "sieg heil", "gas the jews", "jews will not replace",
    "final solution for", "racial holy war", "rahowa",
]
_RE = [(t, re.compile(r"(?<![a-z0-9])" + re.escape(t) + r"(?![a-z0-9])"))
       for t in ATTACK_TERMS]


def norm(s):
    return re.sub(r"\s+", " ", (s or "")).strip()


def attack_hits(text):
    low = (text or "").lower()
    return [t for t, r in _RE if r.search(low)]


def load_rewrites():
    d = {}
    if not os.path.exists(REW):
        return d
    with open(REW, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                r = json.loads(line)
                d[r["id"]] = r
    return d


def run(root=ROOT, verbose=True):
    sys.path.insert(0, HERE)
    from cadgen import targets
    elig, short, gt = targets(root)
    rows = load_rewrites()

    counts = {"train_hate_total": len(elig) + len(short),
              "G0_skipped_short": len(short),
              "eligible": len(elig),
              "no_row": 0,
              "G1_moderation_refused": 0, "G1_api_error": 0, "G1_parse_fail": 0,
              "G2_length_ratio": 0, "G3_attack_terms": 0, "G4_unchanged": 0,
              "accepted": 0}
    accepted, drops = {}, []
    for vid in elig:
        r = rows.get(vid)
        if r is None:
            counts["no_row"] += 1
            drops.append((vid, "no_row", ""))
            continue
        p = r.get("parse")
        if p != "ok":
            if p == "moderation_refused":
                counts["G1_moderation_refused"] += 1
            elif str(p).startswith("error:"):
                counts["G1_api_error"] += 1
            else:
                counts["G1_parse_fail"] += 1
            drops.append((vid, "G1:" + str(p)[:40], ""))
            continue
        o = norm(gt[vid]["text"])
        n = norm(r["rewritten"])
        ratio = len(n) / max(len(o), 1)
        if not (RATIO_LO <= ratio <= RATIO_HI):
            counts["G2_length_ratio"] += 1
            drops.append((vid, "G2", "ratio %.3f" % ratio))
            continue
        hits = attack_hits(n)
        if hits:
            counts["G3_attack_terms"] += 1
            drops.append((vid, "G3", ",".join(hits[:4])))
            continue
        if n.lower() == o.lower():
            counts["G4_unchanged"] += 1
            drops.append((vid, "G4", "identical"))
            continue
        accepted[vid] = {"rewritten": r["rewritten"], "n_edits": r.get("n_edits", -1),
                         "orig_chars": len(o), "new_chars": len(n),
                         "ratio": round(ratio, 4)}
        counts["accepted"] += 1

    out = {"counts": counts, "accepted": accepted,
           "drops": [{"id": a, "gate": b, "detail": c} for a, b, c in drops],
           "params": {"MIN_CHARS": 40, "RATIO_LO": RATIO_LO, "RATIO_HI": RATIO_HI,
                      "n_attack_terms": len(ATTACK_TERMS)}}
    if verbose:
        print(json.dumps(counts, indent=1))
        with open(ACC, "w", encoding="utf-8") as f:
            json.dump(out, f, indent=1, ensure_ascii=False)
        print("wrote", ACC)
    return out


if __name__ == "__main__":
    run()
