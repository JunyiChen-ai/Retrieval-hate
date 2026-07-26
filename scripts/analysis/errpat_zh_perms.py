#!/usr/bin/env python
"""ERRPAT MHC-ZH: permutation tests for the two surviving cluster hypotheses."""
import json, re, numpy as np
from pathlib import Path
ROOT = Path("/data/jehc223/RGCL")
PER = json.load(open(ROOT / "scripts/analysis/errpat_zh_taxonomy_OUT.json"))["per_item"]
MECH = json.load(open(ROOT / "scripts/analysis/errpat_zh_mech_OUT.json"))["per_item"]
rng = np.random.default_rng(20260726)
gold = np.array([x["gold"] for x in MECH]); errc = np.array([x["errc"] for x in MECH])
core = errc == 3
TR = np.array([x["tr_chars"] for x in MECH]); HEM = np.array([x["has_em"] for x in MECH])

def perm(mask, pool, label, n=50000):
    """core errors inside `mask`, permuting core-error identity WITHIN `pool`."""
    obs = int(np.sum(core & mask & pool)); k = int(np.sum(core & pool)); idx = np.where(pool)[0]
    sub = mask[idx]
    null = np.array([int(sub[rng.choice(len(idx), size=k, replace=False)].sum()) for _ in range(n)])
    print(f"{label}\n  pool n={int(pool.sum())} (core in pool={k}); mask n={int((mask&pool).sum())}")
    print(f"  observed={obs} expected={null.mean():.2f} null_p95={np.percentile(null,95):.0f} "
          f"p_one_sided={np.mean(null>=obs):.4f}")
    return {"observed": obs, "expected": round(float(null.mean()), 2),
            "null_p95": float(np.percentile(null, 95)),
            "p_one_sided": round(float(np.mean(null >= obs)), 4),
            "pool_n": int(pool.sum()), "mask_n": int((mask & pool).sum()), "n_perms": n}

allp = np.ones(len(MECH), bool)
out = {}
q = np.percentile(TR, [25, 50, 75])
thin = (TR >= q[0]) & (TR < q[1])
out["H1_thin_transcript_band"] = {
    "band_chars": [round(float(q[0])), round(float(q[1]))],
    "test_all": perm(thin, allp, "H1 thin transcript (whole test set)")}
out["H1_thin_transcript_band"]["test_neg"] = perm(thin, gold == 0, "H1 thin transcript | negatives only")
out["H1_thin_transcript_band"]["test_pos"] = perm(thin, gold == 1, "H1 thin transcript | positives only")
out["H2_keyword_absent_positives"] = {
    "n_pos_with_em": int(np.sum((gold == 1) & HEM)), "n_pos_without_em": int(np.sum((gold == 1) & ~HEM)),
    "FN_core_with_em": int(np.sum(core & (gold == 1) & HEM)),
    "FN_core_without_em": int(np.sum(core & (gold == 1) & ~HEM)),
    "test": perm(~HEM, gold == 1, "H2 keyword-absent | positives only (FN pool)")}
json.dump(out, open(ROOT / "scripts/analysis/errpat_zh_perms_OUT.json", "w"), indent=1)
print("\nH2 raw rates: FN/seed with-em "
      f"{errc[(gold==1)&HEM].sum()/(3*np.sum((gold==1)&HEM)):.4f} (n={int(np.sum((gold==1)&HEM))}) vs "
      f"without-em {errc[(gold==1)&~HEM].sum()/(3*np.sum((gold==1)&~HEM)):.4f} (n={int(np.sum((gold==1)&~HEM))})")
