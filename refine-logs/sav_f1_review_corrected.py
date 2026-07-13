"""SAV F-G1 review — corrected-machinery re-run + C3-style un-crush calibration.

Correction = widen the L2 lambda grid upward so per-arm CV is not pinned at the grid edge
(the deployed grid maxed at lambda=100 = strongest reg; the pooled baseline's true optimum
is lambda~1e3-1e4). Everything else (StandardScaler, LogisticRegressionCV inner 5-fold,
holdout-log-loss bits, Fano projection, example-level clustered bootstrap, the pre-declared
decision rules) is byte-identical to the deployed machinery.

Also: C3-style un-crush calibration (standardize base features only; append raw gold*s so the
gold column is effectively unpenalized) to confirm the probe+Fano pipeline reaches full Fano
headroom when the signal is NOT crushed -> pins the pathology on regularization, not the pipeline.
"""
import os, sys, time, json
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "scripts", "analysis"))
import sav_f0_common as C
from sklearn.linear_model import LogisticRegressionCV, LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

LAMBDAS_WIDE = np.logspace(-4, 5, 10)   # widen upward: 1e-4 .. 1e5 (deployed was 1e-4..1e2)

def load(ds): return C.load_extracted_split(ds,"train",True), C.load_extracted_split(ds,"val",True)
def strat(labels,frac,seed):
    rng=np.random.default_rng(2000+seed); idx=[]
    for c in (0,1):
        ci=np.where(labels==c)[0]; rng.shuffle(ci); k=max(1,int(round(frac*len(ci)))); idx.extend(ci[:k].tolist())
    return np.asarray(sorted(idx))
def sav_top(head_tr,ytr,per_class,seed,k):
    rng=np.random.default_rng(1000+seed); idx=[]
    for c in (0,1):
        ci=np.where(ytr==c)[0]; rng.shuffle(ci); idx.extend(ci[:per_class].tolist())
    sel=np.asarray(sorted(idx)); acc=C.head_nearest_centroid_accuracy(head_tr[sel],ytr[sel])
    order=C.rank_heads(acc); return order[:k], order

def probe_cv_wide(Xtr,ytr,Xva,yval,seed):
    Cs=(1.0/LAMBDAS_WIDE).tolist()
    cv=StratifiedKFold(n_splits=C.CV_FOLDS,shuffle=True,random_state=int(seed))
    clf=LogisticRegressionCV(Cs=Cs,cv=cv,penalty="l2",solver="lbfgs",scoring="neg_log_loss",
                             max_iter=C.PROBE_MAX_ITER,n_jobs=-1,refit=True)
    pipe=make_pipeline(StandardScaler(),clf); pipe.fit(Xtr,ytr)
    proba=pipe.predict_proba(Xva)[:,1]; bits=C.per_example_bits(proba,yval)
    pred=(proba>=0.5).astype(np.int64)
    return {"bits":bits,"correct":(pred==yval).astype(np.int64),"lam":float(1.0/clf.C_[0])}

def onehot(y):
    z=np.zeros((len(y),2)); z[np.arange(len(y)),y.astype(int)]=1.0; return z

print("### loading ...",flush=True)
D={ds:load(ds) for ds in ["MHC","HateMM"]}
RESULT={}

# ---------- C3-style un-crush calibration (proves pipeline reaches headroom when un-crushed) ----------
print("\n### C3-STYLE UN-CRUSH CALIBRATION (standardize base only, append raw gold*s) ###",flush=True)
for ds in ["MHC","HateMM"]:
    tr,va=D[ds]; ytr_all=tr["labels"]; yval=va["labels"]; n_val=len(yval)
    seed=0; ptr=strat(ytr_all,C.PROBE_TRAIN_FRAC,seed); ytr=ytr_all[ptr]
    ptr_pool=tr["img_pooled"][ptr]; va_pool=va["img_pooled"]
    sc=StandardScaler().fit(ptr_pool)  # standardize Z (pooled) on train only
    Ztr=sc.transform(ptr_pool); Zva=sc.transform(va_pool)
    ohtr=onehot(ytr); ohva=onehot(yval)
    for s in [50.0]:
        Xtr=np.concatenate([Ztr, ohtr*s],1); Xva=np.concatenate([Zva, ohva*s],1)
        # keep Z at the deployed crush lambda=100 (C=0.01); gold*s effectively unpenalized
        clf=LogisticRegression(C=0.01,penalty="l2",solver="lbfgs",max_iter=C.PROBE_MAX_ITER)
        clf.fit(Xtr,ytr); proba=clf.predict_proba(Xva)[:,1]
        bits=C.per_example_bits(proba,yval); acc=float(((proba>=0.5).astype(int)==yval).mean())
        print(f"[{ds}] pooled(Z@lam100) + raw gold*{s:.0f} (uncrushed): acc={acc:.4f} ell={bits.mean():.4f}b fano={C.fano_acc(bits.mean()):.4f}  headroom_reached={acc>=0.99}",flush=True)

# ---------- CORRECTED RE-RUN (wide-grid CV, 5 seeds) ----------
def run_arms(ds, arms):
    tr,va=D[ds]; ytr_all=tr["labels"]; yval=va["labels"]; n_val=len(yval)
    store={a:{"bits":np.zeros((n_val,5)),"correct":np.zeros((n_val,5)),"lam":[]} for a in (["pooled"]+arms)}
    for si,seed in enumerate(C.SEEDS):
        ptr=strat(ytr_all,C.PROBE_TRAIN_FRAC,seed); ytr=ytr_all[ptr]
        feats={}
        feats["pooled"]=(tr["img_pooled"][ptr], va["img_pooled"])
        for a in arms:
            if a=="U-1":
                feats["U-1"]=(tr["head_final"][ptr].reshape(len(ptr),-1), va["head_final"].reshape(n_val,-1))
            elif a=="C-pos":
                feats["C-pos"]=(tr["img_hidden_final"][ptr], va["img_hidden_final"])
            elif a.startswith("SAV@"):
                k=int(a.split("@")[1]); top,_=sav_top(tr["head_final"],ytr_all,C.SELECTION_PER_CLASS,seed,k)
                feats[a]=(tr["head_final"][ptr][:,top,:].reshape(len(ptr),-1), va["head_final"][:,top,:].reshape(n_val,-1))
        for a,(Xtr,Xva) in feats.items():
            r=probe_cv_wide(Xtr,ytr,Xva,yval,seed)
            store[a]["bits"][:,si]=r["bits"]; store[a]["correct"][:,si]=r["correct"]; store[a]["lam"].append(r["lam"])
        print(f"  [{ds}] seed {seed} done",flush=True)
    out={}
    bp=store["pooled"]["bits"]; cp=store["pooled"]["correct"]
    out["pooled"]={"acc":float(cp.mean()),"ell":float(bp.mean(axis=1).mean()),"lam_mean":float(np.mean(store["pooled"]["lam"]))}
    for a in arms:
        ba=store[a]["bits"]; ca=store[a]["correct"]
        dL=(bp-ba).mean(axis=1); da=(ca-cp).mean(axis=1); ellp=bp.mean(axis=1); ella=ba.mean(axis=1)
        out[a]={"acc":float(ca.mean()),"ell":float(ella.mean()),"lam_mean":float(np.mean(store[a]["lam"])),
                "dL":C.clustered_bootstrap_mean(dL),"dacc":C.clustered_bootstrap_mean(da),
                "projgain":C.clustered_bootstrap_projection(ellp,ella)}
    return out

print("\n### CORRECTED RE-RUN: MHC (carrying) — wide grid 1e-4..1e5, 5 seeds ###",flush=True)
RESULT["MHC"]=run_arms("MHC", ["SAV@10","SAV@20","U-1","C-pos"])
print("\n### CORRECTED RE-RUN: HateMM (no-harm) ###",flush=True)
RESULT["HateMM"]=run_arms("HateMM", ["SAV@10","SAV@20","U-1"])

def fmt(a,r):
    d=r["dL"]; da=r["dacc"]; pg=r["projgain"]
    return (f"{a:>10}: lam_mean={r['lam_mean']:>8.1f} acc={r['acc']:.4f} ell={r['ell']:.4f} | "
            f"dL={d['mean']:+.4f}[{d['ci_low']:+.4f},{d['ci_high']:+.4f}]x0={int(d['excludes_zero'])} | "
            f"dacc={da['mean']:+.4f}[{da['ci_low']:+.4f},{da['ci_high']:+.4f}]x0={int(da['excludes_zero'])} | "
            f"projG={pg['mean']:+.4f}[{pg['ci_low']:+.4f},{pg['ci_high']:+.4f}]x0={int(pg['excludes_zero'])} "
            f"(pp={pg['acc_proj_pooled']:.3f},ps={pg['acc_proj_sav']:.3f})")

print("\n================ CORRECTED SUMMARY ================",flush=True)
for ds in ["MHC","HateMM"]:
    r=RESULT[ds]
    print(f"\n--- {ds} --- pooled: lam_mean={r['pooled']['lam_mean']:.1f} acc={r['pooled']['acc']:.4f} ell={r['pooled']['ell']:.4f} (val base-rate maj = {max(D[ds][1]['labels'].mean(),1-D[ds][1]['labels'].mean()):.4f})",flush=True)
    for a in r:
        if a=="pooled": continue
        print(fmt(a,r[a]),flush=True)

# decision rule application (pre-declared)
print("\n================ PRE-DECLARED RULE APPLICATION (corrected) ================",flush=True)
BAR=C.PROJECTED_GAIN_BAR; NH=C.HATEMM_NOHARM_DACC
for k in ["SAV@10","SAV@20"]:
    r=RESULT["MHC"][k]; dL=r["dL"]; pg=r["projgain"]
    pass_dL=bool(dL["mean"]>0 and dL["ci_low"]>0); pass_pg=bool(pg["mean"]>BAR and pg["ci_low"]>0)
    print(f"MHC {k}: pass_deltaL={pass_dL} pass_projgain={pass_pg} -> mhc_pass={pass_dL and pass_pg}",flush=True)
for k in ["SAV@10","SAV@20"]:
    r=RESULT["HateMM"][k]; dL=r["dL"]; da=r["dacc"]
    ok_dL=bool(dL["ci_high"]>=0); ok_acc=bool(da["ci_low"]>=NH)
    print(f"HateMM {k} no-harm: ok_deltaL={ok_dL} ok_dacc(ci_low>={NH})={ok_acc} -> noharm={ok_dL and ok_acc}",flush=True)

json.dump({ds:{a:({k:v for k,v in r.items()}) for a,r in RESULT[ds].items()} for ds in RESULT},
          open(os.path.join(os.path.dirname(os.path.abspath(__file__)),"sav_f1_review_corrected_out.json"),"w"),
          default=lambda o: o.tolist() if hasattr(o,"tolist") else str(o), indent=0)
print("\nwrote sav_f1_review_corrected_out.json",flush=True)
