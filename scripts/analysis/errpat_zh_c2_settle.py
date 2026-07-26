#!/usr/bin/env python
"""ERRPAT MHC-ZH: settle ONE definition of the thin-transcript cluster C2 and test it."""
import json, re, numpy as np
from pathlib import Path
ROOT = Path("/data/jehc223/RGCL")
PER = json.load(open(ROOT / "scripts/analysis/errpat_zh_taxonomy_OUT.json"))["per_item"]
ann = {r["Video_ID"]: r for r in json.load(open(ROOT / "data/_src_Multihateclip/Chinese/annotation(new).json"))}
gt = [json.loads(l) for l in open(ROOT / "data/gt/MHC_zh/test.jsonl")]
EM = re.compile(r'<em class="keyword">(.*?)</em>')
rng = np.random.default_rng(20260726)

T = np.array([len((ann.get(r["id"], {}).get("Transcript") or "").strip()) for r in gt])
C = np.array([len(EM.sub(r"\1", r["text"])) for r in gt])
errc = np.array([PER[r["id"]]["n_seeds_wrong"] for r in gt])
gold = np.array([r["label"] for r in gt])
core = errc == 3
q = np.percentile(T, [25, 50, 75])
print(f"transcript-char quartiles: {[round(float(x),2) for x in q]}")
print(f"composed-text chars: min={C.min()} max={C.max()} (n with 0 chars = {int((C==0).sum())})")
print(f"transcript chars: n with 0 = {int((T==0).sum())}")

def perm(mask, label, n=50000):
    obs = int(np.sum(core & mask)); k = int(core.sum())
    null = np.array([int(mask[rng.choice(len(gt), size=k, replace=False)].sum()) for _ in range(n)])
    print(f"{label}: n_band={int(mask.sum())} core_in_band={obs} expected={null.mean():.2f} "
          f"p95={np.percentile(null,95):.0f} p={np.mean(null>=obs):.4f} "
          f"err_rate/seed={errc[mask].sum()/(3*int(mask.sum())):.4f}")
    return dict(n_band=int(mask.sum()), observed=obs, expected=round(float(null.mean()),2),
                null_p95=float(np.percentile(null,95)), p=round(float(np.mean(null>=obs)),4),
                err_rate=round(float(errc[mask].sum()/(3*int(mask.sum()))),4))

out = {"transcript_quartiles": [round(float(x),2) for x in q]}
out["C2_definition_A_quartile_band"] = perm((T >= q[0]) & (T < q[1]),
    f"A: transcript in [{q[0]:.2f},{q[1]:.2f}) (Q2 of transcript length)")
out["C2_definition_B_round_3176"] = perm((T >= 31) & (T < 76), "B: transcript in [31,76) integers")
print()
for lo,hi,tag in ((0,q[0],'Q1'),(q[0],q[1],'Q2'),(q[1],q[2],'Q3'),(q[2],1e9,'Q4')):
    sel=(T>=lo)&(T<hi)
    print(f"  {tag} transcript [{lo:.0f},{hi if hi<1e8 else 999:.0f}): n={int(sel.sum()):3d} core={int((core&sel).sum()):2d} err/seed={errc[sel].sum()/(3*max(int(sel.sum()),1)):.4f}")
json.dump(out, open(ROOT/'scripts/analysis/errpat_zh_c2_settle_OUT.json','w'), indent=1)
