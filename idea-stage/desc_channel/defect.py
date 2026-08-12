"""Frozen input-defect detector (idea-stage/DESC_CHANNEL_FREEZE.md section 3).

Label-free, input-only:  DEFECT(i)  <=>  U_i < 10  OR  nwr_i >= 0.30
  U   = number of alphabetic tokens present in /usr/share/dict/american-english
  nwr = 1 - U/T (T = total alphabetic tokens); nwr := 1.0 when T == 0
"""
import json
import os
import re

LEXICON = "/usr/share/dict/american-english"
U_MIN = 10
NWR_MAX = 0.30
TOK = re.compile(r"[a-zA-Z']+")

_LEX = None


def lexicon():
    global _LEX
    if _LEX is None:
        with open(LEXICON, errors="replace") as f:
            _LEX = {w.strip().lower() for w in f}
    return _LEX


def stats(text):
    """-> (T, U, nwr)"""
    lex = lexicon()
    toks = [t.lower().strip("'") for t in TOK.findall(text or "")]
    toks = [t for t in toks if t]
    if not toks:
        return 0, 0, 1.0
    u = sum(1 for t in toks if t in lex)
    return len(toks), u, 1.0 - u / len(toks)


def is_defect(text):
    _, u, nwr = stats(text)
    return bool(u < U_MIN or nwr >= NWR_MAX)


def load_gt(root, dataset="HateMM"):
    """-> {id: {"text":.., "label":.., "split": train|val|test}} over all three splits."""
    out = {}
    for sp in ("train", "val", "test"):
        p = os.path.join(root, "data", "gt", dataset, sp + ".jsonl")
        with open(p, encoding="utf-8") as f:
            for line in f:
                r = json.loads(line)
                out[r["id"]] = {"text": r.get("text") or "",
                                "label": int(r["label"]), "split": sp}
    return out


if __name__ == "__main__":
    import collections
    import sys
    root = sys.argv[1] if len(sys.argv) > 1 else "/home/jehc223/Retrieval-hate"
    gt = load_gt(root)
    c = collections.Counter()
    for vid, r in gt.items():
        c[(r["split"], is_defect(r["text"]))] += 1
        c[(r["split"], "empty")] += int(not (r["text"] or "").strip())
    print(json.dumps({str(k): v for k, v in sorted(c.items(), key=str)}, indent=1))
