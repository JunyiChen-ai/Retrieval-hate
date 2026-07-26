#!/usr/bin/env python
"""ERRPAT MHC-ZH: what is the mid-length (49-96 char) error cluster actually made of?"""
import json
import re
from pathlib import Path

import numpy as np

ROOT = Path("/data/jehc223/RGCL")
TAX = json.load(open(ROOT / "scripts/analysis/errpat_zh_taxonomy_OUT.json"))
PER = TAX["per_item"]
EM = re.compile(r'<em class="keyword">(.*?)</em>')

ann = {r["Video_ID"]: r for r in
       json.load(open(ROOT / "data/_src_Multihateclip/Chinese/annotation(new).json"))}
gt = [json.loads(l) for l in open(ROOT / "data/gt/MHC_zh/test.jsonl")]

rows = []
for r in gt:
    a = ann.get(r["id"], {})
    tr = (a.get("Transcript") or "").strip()
    ti = (a.get("Title") or "").strip()
    rows.append(dict(id=r["id"], gold=r["label"],
                     chars=len(EM.sub(r"\1", r["text"])),
                     title_chars=len(EM.sub(r"\1", ti)), tr_chars=len(tr),
                     transcript_empty=(tr == ""),
                     errc=PER[r["id"]]["n_seeds_wrong"]))

L = np.array([x["chars"] for x in rows])
q = np.percentile(L, [25, 50, 75])
band = (L >= q[0]) & (L < q[1])
te = np.array([x["transcript_empty"] for x in rows])
errc = np.array([x["errc"] for x in rows])


def med(key, sel):
    v = [x[key] for x, b in zip(rows, sel) if b]
    return float(np.median(v)) if v else float("nan")


print(f"deployed-text quartile cuts (chars): {[round(float(x)) for x in q]}")
print(f"transcript EMPTY (deployed text = Title only): {int(te.sum())}/149")
print(f"  err rate/seed transcript-empty   {errc[te].sum() / (3 * max(int(te.sum()), 1)):.4f} "
      f"(n={int(te.sum())}, 3of3={int((errc[te] == 3).sum())})")
print(f"  err rate/seed transcript-present {errc[~te].sum() / (3 * int((~te).sum())):.4f} "
      f"(n={int((~te).sum())}, 3of3={int((errc[~te] == 3).sum())})")
print()
print("per deployed-text-length quartile:")
for tag, sel in (("Q1 <49", L < q[0]), ("Q2 49-96", band),
                 ("Q3 96-183", (L >= q[1]) & (L < q[2])), ("Q4 >=183", L >= q[2])):
    print(f"  {tag:10s} n={int(sel.sum()):3d} tr_empty={int((te & sel).sum()):2d} "
          f"med_transcript_chars={med('tr_chars', sel):6.0f} med_title_chars={med('title_chars', sel):5.0f} "
          f"3of3_errors={int((errc[sel] == 3).sum()):2d} "
          f"err_rate={errc[sel].sum() / (3 * int(sel.sum())):.4f}")
print()
# the 12 core errors inside the band
print("the 12 core (3/3) errors inside the mid-length band:")
for x in rows:
    i = rows.index(x)
    if band[i] and x["errc"] == 3:
        c = PER[x["id"]]["cov"]
        print(f"  {x['id']} gold={x['gold']} {c['label3']:<9s} chars={x['chars']:3d} "
              f"title={x['title_chars']:3d} transcript={x['tr_chars']:3d} "
              f"dur={c['duration_s']}s asr={c['asr_chars']:3d}")
print()
# is the band effect explained by transcript length rather than total length?
print("core-error rate by TRANSCRIPT length band:")
T = np.array([x["tr_chars"] for x in rows])
tq = np.percentile(T, [25, 50, 75])
for tag, sel in ((f"<{tq[0]:.0f}", T < tq[0]), (f"{tq[0]:.0f}-{tq[1]:.0f}", (T >= tq[0]) & (T < tq[1])),
                 (f"{tq[1]:.0f}-{tq[2]:.0f}", (T >= tq[1]) & (T < tq[2])), (f">={tq[2]:.0f}", T >= tq[2])):
    print(f"  transcript {tag:9s} n={int(sel.sum()):3d} 3of3={int((errc[sel] == 3).sum()):2d} "
          f"err_rate={errc[sel].sum() / (3 * max(int(sel.sum()), 1)):.4f}")
