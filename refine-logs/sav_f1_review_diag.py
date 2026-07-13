"""SAV F-G1 INDEPENDENT VERDICT REVIEW — machinery calibration + corrected re-run.

CPU-only. Reads the cached extract artifacts under artifacts/sav_f0/extract/.
Imports the ACTUAL probe machinery (sav_f0_probe, sav_f0_common) so any reproduction
is byte-faithful to what produced verdict.json. Adds:
  (1) reproduction check vs verdict.json (pooled, SAV@10) — machinery is what it claims;
  (2) MANDATED calibration: label-oracle arms (append gold one-hot to pooled AND SAV),
      plus a STANDALONE gold one-hot arm; must reach ~full Fano headroom or machinery invalid;
  (3) lambda sensitivity sweep (fixed C, both directions past the grid edge);
  (4) corrected-machinery re-run of the key cells if needed.

No GPU, no SLURM, no network, no commits. Gold labels read for probe/calibration only.
"""
import os, sys, json, time
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "scripts", "analysis"))
import sav_f0_common as C
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

REPO = C.REPO_ROOT
OUT = {}

def load(ds):
    tr = C.load_extracted_split(ds, "train", with_heads=True)
    va = C.load_extracted_split(ds, "val", with_heads=True)
    return tr, va

def base_rate(y):
    y = np.asarray(y); return float(y.mean()), int((y==1).sum()), int((y==0).sum())

# ---- faithful arm score at a FIXED lambda (no CV) for the sweep ----
def score_fixed(Xtr, ytr, Xval, yval, lam):
    Cinv = 1.0/lam
    clf = LogisticRegression(C=Cinv, penalty="l2", solver="lbfgs", max_iter=C.PROBE_MAX_ITER)
    pipe = make_pipeline(StandardScaler(), clf)
    pipe.fit(Xtr, ytr)
    proba = pipe.predict_proba(Xval)[:,1]
    bits = C.per_example_bits(proba, yval)
    pred = (proba>=0.5).astype(np.int64)
    return {"acc": float((pred==yval).mean()), "L_bits": float(bits.sum()),
            "ell": float(bits.mean()), "bits": bits, "correct": (pred==yval).astype(np.int64),
            "proba": proba}

# ---- faithful CV arm (exactly the deployed probe) ----
def score_cv(Xtr, ytr, Xval, yval, seed):
    proba, lam = None, None
    from sav_f0_probe import fit_logreg_probe
    proba, lam = fit_logreg_probe(Xtr, ytr, Xval, seed)
    bits = C.per_example_bits(proba, yval)
    pred = (proba>=0.5).astype(np.int64)
    return {"acc": float((pred==yval).mean()), "L_bits": float(bits.sum()),
            "bits": bits, "correct": (pred==yval).astype(np.int64), "lambda": lam}

def sav_select(head_tr, ytr, per_class, seed, k):
    rng = np.random.default_rng(1000+seed)
    idx=[]
    for c in (0,1):
        ci=np.where(ytr==c)[0]; rng.shuffle(ci); idx.extend(ci[:per_class].tolist())
    sel=np.asarray(sorted(idx))
    acc = C.head_nearest_centroid_accuracy(head_tr[sel], ytr[sel])
    order = C.rank_heads(acc)
    return order[:k], order

def strat(labels, frac, seed):
    rng=np.random.default_rng(2000+seed); idx=[]
    for c in (0,1):
        ci=np.where(labels==c)[0]; rng.shuffle(ci); k=max(1,int(round(frac*len(ci)))); idx.extend(ci[:k].tolist())
    return np.asarray(sorted(idx))

# ================================================================= #
print("### loading MHC + HateMM ...", flush=True)
data = {}
for ds in ["MHC","HateMM"]:
    t0=time.time(); data[ds]=load(ds); print(f"  {ds} loaded {time.time()-t0:.1f}s", flush=True)

for ds in ["MHC","HateMM"]:
    tr,va=data[ds]
    br=base_rate(va["labels"]); brtr=base_rate(tr["labels"])
    print(f"{ds}: val base_rate pos={br[0]:.4f} ({br[1]}/{br[1]+br[2]}); train pos={brtr[0]:.4f}", flush=True)
    OUT.setdefault(ds,{})["base_rate_val"]=br[0]; OUT[ds]["base_rate_train"]=brtr[0]

# ---------- (1) REPRODUCTION: pooled + SAV@10 on MHC, 5 seeds, exactly as deployed ----------
print("\n### (1) reproduction of deployed CV probe (MHC pooled & SAV@10) ...", flush=True)
ds="MHC"; tr,va=data[ds]; yval=va["labels"]; ytr_all=tr["labels"]; n_val=len(yval)
store={}
for arm in ["pooled","SAV@10"]:
    store[arm]={"bits":np.zeros((n_val,5)),"correct":np.zeros((n_val,5)),"lam":[]}
for si,seed in enumerate(C.SEEDS):
    ptr=strat(ytr_all, C.PROBE_TRAIN_FRAC, seed); ytr=ytr_all[ptr]
    top,order=sav_select(tr["head_final"], ytr_all, C.SELECTION_PER_CLASS, seed, 10)
    r=score_cv(tr["img_pooled"][ptr], ytr, va["img_pooled"], yval, seed)
    store["pooled"]["bits"][:,si]=r["bits"]; store["pooled"]["correct"][:,si]=r["correct"]; store["pooled"]["lam"].append(r["lambda"])
    Xtr=tr["head_final"][ptr][:,top,:].reshape(len(ptr),-1); Xva=va["head_final"][:,top,:].reshape(n_val,-1)
    r=score_cv(Xtr,ytr,Xva,yval,seed)
    store["SAV@10"]["bits"][:,si]=r["bits"]; store["SAV@10"]["correct"][:,si]=r["correct"]; store["SAV@10"]["lam"].append(r["lambda"])
def cmp_arm(arm):
    bp=store["pooled"]["bits"]; ba=store[arm]["bits"]; cp=store["pooled"]["correct"]; ca=store[arm]["correct"]
    dL=(bp-ba).mean(axis=1); da=(ca-cp).mean(axis=1)
    ellp=bp.mean(axis=1); ella=ba.mean(axis=1)
    return (float(ba.sum(axis=0).mean()), float(ca.mean()),
            C.clustered_bootstrap_mean(dL), C.clustered_bootstrap_mean(da),
            C.clustered_bootstrap_projection(ellp,ella))
print("pooled: L_bits_mean=%.4f acc=%.4f lam_mean=%.1f"%(float(store['pooled']['bits'].sum(axis=0).mean()), float(store['pooled']['correct'].mean()), np.mean(store['pooled']['lam'])))
Lsav,accsav,dL,da,pg=cmp_arm("SAV@10")
print("SAV@10: L_bits_mean=%.4f acc=%.4f lam_mean=%.1f"%(Lsav,accsav,np.mean(store['SAV@10']['lam'])))
print("  dL   mean=%.6f CI[%.6f,%.6f] x0=%s"%(dL['mean'],dL['ci_low'],dL['ci_high'],dL['excludes_zero']))
print("  dacc mean=%.6f CI[%.6f,%.6f] x0=%s"%(da['mean'],da['ci_low'],da['ci_high'],da['excludes_zero']))
print("  projG mean=%.6f CI[%.6f,%.6f] x0=%s (proj_pooled=%.4f proj_sav=%.4f)"%(pg['mean'],pg['ci_low'],pg['ci_high'],pg['excludes_zero'],pg['acc_proj_pooled'],pg['acc_proj_sav']))
OUT["repro_MHC_SAV10"]={"dL":dL,"dacc":da,"projgain":pg,"pooled_ell":float(store['pooled']['bits'].sum(axis=0).mean())/n_val}
