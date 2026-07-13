"""SAV F-G1 review — (2) MANDATED label-oracle calibration + (3) lambda sweep.

Tests whether the deployed machinery can credit a KNOWN-PERFECT signal (Fano headroom),
and how the pooled/SAV cells move as regularization is swept past the grid edge.
"""
import os, sys, time
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "scripts", "analysis"))
import sav_f0_common as C
from sav_f0_probe import fit_logreg_probe
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

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
    return C.rank_heads(acc)[:k]

def onehot(y):
    z=np.zeros((len(y),2)); z[np.arange(len(y)),y.astype(int)]=1.0; return z

def probe_cv(Xtr,ytr,Xva,yval,seed):
    proba,lam=fit_logreg_probe(Xtr,ytr,Xva,seed)
    bits=C.per_example_bits(proba,yval); pred=(proba>=0.5).astype(int)
    return float((pred==yval).mean()), float(bits.mean()), lam
def probe_fixed(Xtr,ytr,Xva,yval,lam):
    clf=LogisticRegression(C=1.0/lam,penalty="l2",solver="lbfgs",max_iter=C.PROBE_MAX_ITER)
    pipe=make_pipeline(StandardScaler(),clf); pipe.fit(Xtr,ytr)
    proba=pipe.predict_proba(Xva)[:,1]; bits=C.per_example_bits(proba,yval); pred=(proba>=0.5).astype(int)
    return float((pred==yval).mean()), float(bits.mean())

print("### loading ...",flush=True)
D={ds:load(ds) for ds in ["MHC","HateMM"]}

for ds in ["MHC","HateMM"]:
    tr,va=D[ds]; ytr_all=tr["labels"]; yval=va["labels"]; n_val=len(yval)
    seed=0; ptr=strat(ytr_all,C.PROBE_TRAIN_FRAC,seed); ytr=ytr_all[ptr]
    top=sav_top(tr["head_final"],ytr_all,C.SELECTION_PER_CLASS,seed,10)
    pooled_tr,pooled_va=tr["img_pooled"][ptr],va["img_pooled"]
    sav_tr=tr["head_final"][ptr][:,top,:].reshape(len(ptr),-1); sav_va=va["head_final"][:,top,:].reshape(n_val,-1)
    oh_tr=onehot(ytr); oh_va=onehot(yval)
    fano=lambda ell: C.fano_acc(ell)
    print(f"\n========== {ds} (n_val={n_val}, val base rate={yval.mean():.4f}, seed 0) ==========",flush=True)

    # ---- (2) CALIBRATION ----
    print("--- (2) label-oracle calibration (deployed CV probe picks lambda) ---",flush=True)
    # standalone gold one-hot
    a,e,l=probe_cv(oh_tr,ytr,oh_va,yval,seed)
    print(f"[oracle standalone 2-d]           acc={a:.4f} ell={e:.4f}b fano={fano(e):.4f} lam={l:.1f}",flush=True)
    # pooled baseline (reference)
    ap,ep,lp=probe_cv(pooled_tr,ytr,pooled_va,yval,seed)
    print(f"[pooled baseline 3584-d]          acc={ap:.4f} ell={ep:.4f}b fano={fano(ep):.4f} lam={lp:.1f}",flush=True)
    # pooled + gold one-hot appended (C3-style: perfect signal embedded in big Z, shared L2)
    a,e,l=probe_cv(np.concatenate([pooled_tr,oh_tr],1),ytr,np.concatenate([pooled_va,oh_va],1),yval,seed)
    print(f"[pooled(+)gold  CV]               acc={a:.4f} ell={e:.4f}b fano={fano(e):.4f} lam={l:.1f}  headroom_reached={a>=0.99}",flush=True)
    # SAV + gold one-hot appended
    a,e,l=probe_cv(np.concatenate([sav_tr,oh_tr],1),ytr,np.concatenate([sav_va,oh_va],1),yval,seed)
    print(f"[SAV@10(+)gold  CV]               acc={a:.4f} ell={e:.4f}b fano={fano(e):.4f} lam={l:.1f}  headroom_reached={a>=0.99}",flush=True)
    # un-crushed: scale gold one-hot by s (post-StandardScaler it dominates; effectively unpenalized)
    for s in [1.0, 20.0, 100.0]:
        Xtr=np.concatenate([pooled_tr, oh_tr*s],1); Xva=np.concatenate([pooled_va, oh_va*s],1)
        a,e=probe_fixed(Xtr,ytr,Xva,yval,100.0)  # fixed at the crush lambda, but gold scaled up
        print(f"[pooled(+)gold*{s:<5.0f} lam=100]     acc={a:.4f} ell={e:.4f}b fano={fano(e):.4f}",flush=True)

    # ---- (3) LAMBDA SWEEP (fixed lambda, pooled vs SAV@10) ----
    print("--- (3) lambda sweep, fixed lambda, seed 0 (grid edge is lam=100) ---",flush=True)
    print(f"{'lambda':>10} | {'pooled_acc':>10} {'pooled_ell':>10} {'pool_fano':>9} | {'sav_acc':>8} {'sav_ell':>8} {'sav_fano':>8} | {'projgain':>8}",flush=True)
    for lam in [1e-4,1e-3,1e-2,1e-1,1e0,1e1,1e2,1e3,1e4,1e5]:
        ap,ep=probe_fixed(pooled_tr,ytr,pooled_va,yval,lam)
        asv,esv=probe_fixed(sav_tr,ytr,sav_va,yval,lam)
        pg=fano(esv)-fano(ep)
        print(f"{lam:>10.0e} | {ap:>10.4f} {ep:>10.4f} {fano(ep):>9.4f} | {asv:>8.4f} {esv:>8.4f} {fano(esv):>8.4f} | {pg:>+8.4f}",flush=True)
    # constant base-rate predictor reference codelength
    p0=ytr.mean(); ell_const=float(np.mean(np.where(yval==1,-np.log2(p0),-np.log2(1-p0))))
    print(f"[reference] constant train-base-rate predictor p={p0:.4f}: val ell={ell_const:.4f}b fano={fano(ell_const):.4f}; majority-class acc={max(yval.mean(),1-yval.mean()):.4f}",flush=True)
