#!/usr/bin/env python3
"""C3-NONTARGET verdict review — null-anomaly mechanical diagnosis (CPU, checkpointed).

Question: shuffled_text_k8 on MHC/CLIP reports Δacc +0.0227 [+0.0020,+0.0447] (CI excludes 0),
but a video-permuted text embedding should carry ZERO conditional info. The probe used ONE
shuffle seed (12345) and a per-VIDEO bootstrap that holds that permutation FIXED, so the
reported CI reflects video resampling, NOT the permutation-null spread.

This script:
  (1) reproduces the trigger cell (MHC/CLIP) baseline / text_pca_k8 / shuffled_k8(seed 12345);
  (2) builds the PERMUTATION-NULL DISTRIBUTION over many shuffle seeds -> true null mean/spread,
      systematic-bias estimate, and a permutation p-value for the real arm;
  (3) a GAUSSIAN-noise-block control (append 8 iid N(0,1) x s columns) over many seeds -> does
      appending unpenalized columns bias Δacc regardless of content?
  (4) a C_Z sweep of the shuffled/gauss null -> does the positive null appear only at weak reg?
Faithful to scripts/analysis/c3_nontarget_probe.py machinery (Z std alone @ C_Z; aux PCA block
fit on train fold, std, x s=50, appended, same C_Z; 5x5 RepeatedStratifiedKFold rs=1000+rep).
CPU only, no GPU/SLURM/net. Not committed.
"""
import os, json, sys, time
for _v in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):
    os.environ.setdefault(_v, '4')
import numpy as np, torch, warnings
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
warnings.filterwarnings('ignore')

REPO='/data/jehc223/RGCL'; ART=f'{REPO}/artifacts/c3_nontarget'
N_SPLITS,N_REPEATS=5,5; C_GRID=[0.001,0.01,0.1,1.0]; EPS=1e-12; SCALE_A=50.0; MAX_ITER=2000; KS=[8,16,32,64]
CLIP='openai_clip-vit-large-patch14-336_HF'; QWEN='Qwen2.5-VL-7B-Instruct_HF'
OUT=f'{REPO}/refine-logs/c3nt_verdict_review_diag_OUT.json'

def load_sample(ds):
    m=json.load(open(f'{ART}/{ds}_sample300.json')); ids=list(m['ids'])
    y=np.array([int(m['labels'][i]) for i in ids],dtype=int); return ids,y
def load_Z(ds,enc,ids):
    o=torch.load(f'{REPO}/data/CLIP_Embedding/{ds}/train_{enc}.pt',map_location='cpu',weights_only=False)
    cache_ids=o['ids'][0]; pos={s:i for i,s in enumerate(cache_ids)}; idx=[pos[s] for s in ids]
    Z=np.concatenate([o['img_feats'].numpy()[idx],o['text_feats'].numpy()[idx]],axis=1).astype(np.float64)
    return Z, o['labels'].numpy().astype(int)[idx]
def load_Atext(ds,ids):
    return np.stack([np.load(f'{ART}/{ds}/emb/{s}.npy').astype(np.float64) for s in ids],axis=0)

def pick_C(Z,y):
    best_c,best=C_GRID[0],-1.0; skf=StratifiedKFold(N_SPLITS,shuffle=True,random_state=0)
    for c in C_GRID:
        a=[]
        for tr,te in skf.split(Z,y):
            sc=StandardScaler().fit(Z[tr]); lr=LogisticRegression(C=c,max_iter=MAX_ITER).fit(sc.transform(Z[tr]),y[tr])
            a.append((lr.predict(sc.transform(Z[te]))==y[te]).mean())
        if np.mean(a)>best: best,best_c=float(np.mean(a)),c
    return best_c
def _fit_cor(Xtr,ytr,Xte,yte,C):
    lr=LogisticRegression(C=C,max_iter=MAX_ITER).fit(Xtr,ytr)
    return ((lr.predict_proba(Xte)[:,1]>=0.5).astype(int)==yte).astype(float)

def baseline_cor(Z,y,C_Z):
    n=len(y); cor=np.zeros(n); cnt=np.zeros(n)
    for rep in range(N_REPEATS):
        for tr,te in StratifiedKFold(N_SPLITS,shuffle=True,random_state=1000+rep).split(Z,y):
            scZ=StandardScaler().fit(Z[tr]); cnt[te]+=1
            cor[te]+=_fit_cor(scZ.transform(Z[tr]),y[tr],scZ.transform(Z[te]),y[te],C_Z)
    return cor/cnt

def arm_cor(Z,y,C_Z,block_fn,k=8):
    """block_fn(tr, te, rng) -> (Btr, Bte) already standardized*s, appended to Z_std. Refit at C_Z."""
    n=len(y); cor=np.zeros(n); cnt=np.zeros(n)
    for rep in range(N_REPEATS):
        for tr,te in StratifiedKFold(N_SPLITS,shuffle=True,random_state=1000+rep).split(Z,y):
            scZ=StandardScaler().fit(Z[tr]); Ztr,Zte=scZ.transform(Z[tr]),scZ.transform(Z[te]); cnt[te]+=1
            Btr,Bte=block_fn(tr,te)
            cor[te]+=_fit_cor(np.concatenate([Ztr,Btr],1),y[tr],np.concatenate([Zte,Bte],1),y[te],C_Z)
    return cor/cnt

def pca_block(src,k):
    def f(tr,te):
        scS=StandardScaler().fit(src[tr]); Str,Ste=scS.transform(src[tr]),scS.transform(src[te])
        kk=min(k,len(tr)-1,src.shape[1]); pca=PCA(n_components=kk,random_state=0).fit(Str)
        Ptr,Pte=pca.transform(Str),pca.transform(Ste)
        scP=StandardScaler().fit(Ptr); return scP.transform(Ptr)*SCALE_A, scP.transform(Pte)*SCALE_A
    return f

def arm_cor_allk(Z,y,C_Z,src,ks=KS):
    """Per-video cor for every k in ks in ONE CV pass (PCA fit to max(ks) once, sliced), mirroring
    the original probe's kmax-then-slice. Returns {k: cor[n]}."""
    n=len(y); cor={k:np.zeros(n) for k in ks}; cnt=np.zeros(n); kmax=max(ks)
    for rep in range(N_REPEATS):
        for tr,te in StratifiedKFold(N_SPLITS,shuffle=True,random_state=1000+rep).split(Z,y):
            scZ=StandardScaler().fit(Z[tr]); Ztr,Zte=scZ.transform(Z[tr]),scZ.transform(Z[te]); cnt[te]+=1
            scS=StandardScaler().fit(src[tr]); Str,Ste=scS.transform(src[tr]),scS.transform(src[te])
            kk=min(kmax,len(tr)-1,src.shape[1]); pca=PCA(n_components=kk,random_state=0).fit(Str)
            Ptr,Pte=pca.transform(Str),pca.transform(Ste)
            for k in ks:
                j=min(k,kk); scP=StandardScaler().fit(Ptr[:,:j])
                Btr=scP.transform(Ptr[:,:j])*SCALE_A; Bte=scP.transform(Pte[:,:j])*SCALE_A
                cor[k][te]+=_fit_cor(np.concatenate([Ztr,Btr],1),y[tr],np.concatenate([Zte,Bte],1),y[te],C_Z)
    return {k:cor[k]/cnt for k in ks}
def gauss_block(G,k):
    def f(tr,te):
        scP=StandardScaler().fit(G[tr]); return scP.transform(G[tr][:, :k])*SCALE_A, scP.transform(G[te][:, :k])*SCALE_A
    return f

def dacc_mean(cor_arm,cor_base): return float((cor_arm-cor_base).mean())

def run():
    out={} if not os.path.exists(OUT) else json.load(open(OUT))
    ids_c,y_c=load_sample('MHC'); Zc,yk=load_Z('MHC',CLIP,ids_c); assert np.array_equal(yk,y_c)
    A_c=load_Atext('MHC',ids_c); n=len(y_c)
    C_Z=pick_C(Zc,y_c)  # expect 1.0
    base=baseline_cor(Zc,y_c,C_Z); accZ=float(base.mean())
    # (1) reproduce (real all-k + shuffled seed 12345 all-k)
    if 'repro' not in out:
        rk=arm_cor_allk(Zc,y_c,C_Z,A_c); real_k={k:dacc_mean(rk[k],base) for k in KS}
        A_shuf=A_c[np.random.default_rng(12345).permutation(n)]
        sk=arm_cor_allk(Zc,y_c,C_Z,A_shuf); shuf_k={k:dacc_mean(sk[k],base) for k in KS}
        out['repro']={'C_Z':C_Z,'accZ':accZ,
                      'real_dacc':{str(k):real_k[k] for k in KS},'real_max_over_k':float(max(real_k.values())),
                      'shuf_dacc_seed12345':{str(k):shuf_k[k] for k in KS}}
        json.dump(out,open(OUT,'w'),indent=1); print('[repro]',out['repro'],flush=True)
    real_k={int(k):v for k,v in out['repro']['real_dacc'].items()}
    d_real=real_k[8]; real_max=out['repro']['real_max_over_k']
    # (2) permutation-null DISTRIBUTION over many seeds, all k + max-over-k (selection-corrected)
    NSEED=int(os.environ.get('NSEED','150'))
    if 'perm_null' not in out or len(out.get('perm_null',{}).get('maxk',[]))<NSEED:
        pk={str(k):out.get('perm_null',{}).get('perk',{}).get(str(k),[]) for k in KS}
        maxk=out.get('perm_null',{}).get('maxk',[])
        for si in range(len(maxk),NSEED):
            perm=np.random.default_rng(70000+si).permutation(n)
            c=arm_cor_allk(Zc,y_c,C_Z,A_c[perm]); dk={k:dacc_mean(c[k],base) for k in KS}
            for k in KS: pk[str(k)].append(dk[k])
            maxk.append(float(max(dk.values())))
            if si%10==9 or si==NSEED-1:
                a8=np.array(pk['8']); am=np.array(maxk)
                out['perm_null']={'n_seed':len(maxk),'perk':pk,'maxk':maxk,
                    'k8_mean':float(a8.mean()),'k8_sd':float(a8.std(ddof=1)),
                    'k8_q':[float(np.percentile(a8,q)) for q in (2.5,50,97.5)],
                    'p_realk8_ge':float((a8>=d_real).mean()),
                    'maxk_mean':float(am.mean()),'maxk_sd':float(am.std(ddof=1)),
                    'maxk_q':[float(np.percentile(am,q)) for q in (2.5,50,97.5)],
                    'p_realmax_ge_permmax':float((am>=real_max).mean()),
                    'p_permmax_ge_bar040':float((am>=0.040).mean())}
                json.dump(out,open(OUT,'w'),indent=1)
                print(f'[perm {len(maxk)}] k8_mean={a8.mean():+.4f} sd={a8.std(ddof=1):.4f} '
                      f'p(k8null>=real{d_real:+.4f})={(a8>=d_real).mean():.3f} | '
                      f'maxk_mean={am.mean():+.4f} p(permmax>=realmax{real_max:+.4f})={(am>=real_max).mean():.3f}',flush=True)
    # (3) gaussian-noise-block null over many seeds (k=8)
    if 'gauss_null' not in out or len(out.get('gauss_null',{}).get('daccs',[]))<NSEED:
        daccs=out.get('gauss_null',{}).get('daccs',[])
        for si in range(len(daccs),NSEED):
            G=np.random.default_rng(90000+si).standard_normal((n,8))
            c=arm_cor(Zc,y_c,C_Z,gauss_block(G,8)); daccs.append(dacc_mean(c,base))
            if si%10==9 or si==NSEED-1:
                arr=np.array(daccs)
                out['gauss_null']={'k':8,'n_seed':len(daccs),'daccs':daccs,'mean':float(arr.mean()),
                    'std':float(arr.std(ddof=1)),'q':[float(np.percentile(arr,q)) for q in (2.5,50,97.5)]}
                json.dump(out,open(OUT,'w'),indent=1); print(f'[gauss {len(daccs)}] mean={arr.mean():+.4f} sd={arr.std(ddof=1):.4f}',flush=True)
    # (4) C_Z sweep of the null (perm+gauss), 40 seeds each, to localize the artifact
    if 'cz_sweep' not in out:
        sweep={}
        for cz in C_GRID:
            b=baseline_cor(Zc,y_c,cz); pl=[]; gl=[]
            for si in range(40):
                perm=np.random.default_rng(70000+si).permutation(n)
                pl.append(dacc_mean(arm_cor(Zc,y_c,cz,pca_block(A_c[perm],8)),b))
                G=np.random.default_rng(90000+si).standard_normal((n,8))
                gl.append(dacc_mean(arm_cor(Zc,y_c,cz,gauss_block(G,8)),b))
            sweep[str(cz)]={'accZ':float(b.mean()),'perm_mean':float(np.mean(pl)),'perm_sd':float(np.std(pl,ddof=1)),
                            'gauss_mean':float(np.mean(gl)),'gauss_sd':float(np.std(gl,ddof=1))}
            out['cz_sweep']=sweep; json.dump(out,open(OUT,'w'),indent=1)
            print(f'[cz={cz}] accZ={b.mean():.4f} perm_mean={np.mean(pl):+.4f} gauss_mean={np.mean(gl):+.4f}',flush=True)
    print('DONE',flush=True)

if __name__=='__main__':
    t0=time.time(); run(); print(f'elapsed {time.time()-t0:.0f}s',flush=True)
