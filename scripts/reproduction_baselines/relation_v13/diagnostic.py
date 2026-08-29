#!/usr/bin/env python3
"""V13: label-free repeated-seed reliability for prior/locator roles."""
import argparse,hashlib,json,sys
from pathlib import Path
import numpy as np
from scipy.stats import spearmanr

HERE=Path(__file__).resolve().parent
sys.path[:0]=[str(HERE.parent),str(HERE.parent.parent/'duplex')]
from hate_common import data as hdata
from relation_v4.io import fit_ecdf,apply_ecdf,sha256
from relation_v8.run import load_split_exact,atomic_json
from relation_v11.score_stream_benchmark import metrics
from relation_v12.diagnostic import frozen_v10_identity
from relation_v9.train import load as load_v9

EPS=1e-8

def paths(x): return [x] if isinstance(x,str) else list(x)

def load_repeats(manifest,expert,split,ids,refs=None):
    rows=[]
    key=expert['score_key']
    for path in paths(expert[f'{split}_scores']):
        rec=hdata.load_scores_jsonl(path);missing=sorted(set(ids)-set(rec))
        if missing: raise RuntimeError(f'{expert["name"]} {split} missing {missing[:3]}')
        rows.append({v:np.asarray(rec[v][key],np.float64) for v in ids})
    if refs is None:
        refs=[]
        for row in rows:
            ref=np.sort(np.concatenate([row[v] for v in ids]));refs.append(ref)
    out=[]
    for row,ref in zip(rows,refs):
        out.append({v:np.searchsorted(ref,row[v],side='right')/max(len(ref),1) for v in ids})
    return out,refs

def reliability(items):
    """One-way random effects reliability of the K-repeat mean."""
    x=np.asarray(items,np.float64);n,k=x.shape
    if k<2:return None
    means=x.mean(1);grand=means.mean()
    msb=k*np.sum((means-grand)**2)/max(n-1,1)
    msw=np.sum((x-means[:,None])**2)/max(n*(k-1),1)
    between=max((msb-msw)/k,0.);noise=max(msw,0.)
    rel=between/(between+noise/k+EPS);snr=between/(noise/k+EPS)
    return {'n_items':int(n),'n_repeats':int(k),'between':float(between),'noise':float(noise),'icc_mean':float(rel),'snr_mean':float(snr),'precision':float(1/(noise/k+1e-4))}

def role_stats(repeats,ids):
    prior=np.asarray([[r[v].mean() for r in repeats] for v in ids])
    centered=[]
    for v in ids:
        centered.extend(np.stack([r[v]-r[v].mean() for r in repeats],1))
    return reliability(prior),reliability(np.asarray(centered))

def collapse_exact(expert_mean,ids):
    names=list(expert_mean);groups=[]
    for name in names:
        placed=False
        for g in groups:
            if all(np.array_equal(expert_mean[name][v],expert_mean[g[0]][v]) for v in ids):g.append(name);placed=True;break
        if not placed:groups.append([name])
    return groups

def cluster_weights(groups,stats,role):
    # Reliability odds (signal / repeat-noise of the mean) is the relevant
    # precision for estimating a latent role. Pure inverse noise would
    # incorrectly reward an almost-constant but repeatable expert.
    measured=[stats[n][role]['snr_mean'] for n in stats if stats[n][role] is not None]
    neutral=float(np.median(measured)) if measured else 1.
    raw=[];explain=[]
    for group in groups:
        vals=[stats[n][role]['snr_mean'] for n in group if stats[n][role] is not None]
        value=float(np.median(vals)) if vals else neutral
        raw.append(value);explain.append({'members':group,'precision':value,'single_repeat_neutral_availability':not bool(vals)})
    raw=np.asarray(raw);raw=raw/raw.sum()
    return raw,explain

def aggregate(values,groups,wp,wl,prior=True,locator=True):
    out={}
    for v in values[next(iter(values))]:
        z=np.stack([np.mean(np.stack([values[n][v] for n in g]),0) for g in groups],1)
        p=float(z.mean(0)@wp) if prior else 0.
        l=(z-z.mean(0,keepdims=True))@wl if locator else np.zeros(len(z))
        out[v]=p+l
    return out

def shuffled(pred):
    out={}
    for v,x in pred.items():
        seed=int.from_bytes(hashlib.sha256(v.encode()).digest()[:8],'little');rng=np.random.default_rng(seed)
        out[v]=x.mean()+rng.permutation(x-x.mean())
    return out

def main():
    p=argparse.ArgumentParser();p.add_argument('--manifest',required=True);p.add_argument('--out',required=True);a=p.parse_args()
    m=json.load(open(a.manifest));val_raw,vg,_=load_split_exact(m,'val');test_raw,tg,_=load_split_exact(m,'test');vids=sorted(vg);tids=sorted(tg)
    use_train=all('train_scores' in e for e in m['experts']);state_ids=load_v9(m,'train')[0] if use_train else vids;state_split='train' if use_train else 'val';values_state={};values_val={};values_test={};stats={};source=[]
    for e in m['experts']:
        sr,refs=load_repeats(m,e,state_split,state_ids);vr,_=load_repeats(m,e,'val',vids,refs);tr,_=load_repeats(m,e,'test',tids,refs)
        name=e['name'];values_state[name]={v:np.mean([r[v] for r in sr],0) for v in state_ids};values_val[name]={v:np.mean([r[v] for r in vr],0) for v in vids};values_test[name]={v:np.mean([r[v] for r in tr],0) for v in tids}
        pr,lo=role_stats(sr,state_ids);stats[name]={'prior':pr,'locator':lo,'repeat_count':len(sr),'single_repeat_policy':'neutral availability precision; no ICC/SNR claim' if len(sr)==1 else None}
        source.append({'name':name,'state_split':state_split,'repeat_count':len(sr),'paths':[str(Path(x).resolve()) for x in paths(e[f'{state_split}_scores'])],'sha256':[sha256(x) for x in paths(e[f'{state_split}_scores'])]})
    groups=collapse_exact(values_state,state_ids);wp,prior_explain=cluster_weights(groups,stats,'prior');wl,locator_explain=cluster_weights(groups,stats,'locator')
    # Freeze all label-free state before any role-quality audit or test metric.
    frozen={'method':'relation_v13_repeated_measure_reliability','corpus':m['corpus'],'state_split':state_split,'formula':'one-way random-effects ICC of repeat mean; cluster role precision=ICC/(1-ICC)=between/(repeat-noise/K); score=precision prior + precision centered locator','clusters':groups,'prior_weights':wp.tolist(),'locator_weights':wl.tolist(),'prior_weight_detail':prior_explain,'locator_weight_detail':locator_explain,'expert_reliability':stats,'sources':source,'frame_labels_used_for_state':False,'single_repeat_experts_do_not_receive_fabricated_repeats':True,'manifest':str(Path(a.manifest).resolve()),'manifest_sha256':sha256(a.manifest),'test_opened':False}
    freeze=Path(a.out).with_suffix('.frozen.json');atomic_json(freeze,frozen)
    # Post-freeze validation audit: does label-free locator ranking agree with
    # actual within-video localization quality? This never changes weights.
    audit=[]
    for n in values_val:
        q=metrics(values_val[n],vg);audit.append({'expert':n,'reliability':None if stats[n]['locator'] is None else stats[n]['locator']['icc_mean'],'within_centered_ap':q['within_centered_ap'],'within_centered_roc':q['within_centered_roc']})
    comparable=[x for x in audit if x['reliability'] is not None];rank={'n_comparable':len(comparable),'spearman_icc_vs_within_ap':float(spearmanr([x['reliability'] for x in comparable],[x['within_centered_ap'] for x in comparable]).statistic) if len(comparable)>1 else None,'matches_best_repeated_expert':(max(comparable,key=lambda x:x['reliability'])['expert']==max(comparable,key=lambda x:x['within_centered_ap'])['expert']) if comparable else None}
    pred=aggregate(values_test,groups,wp,wl);prior=aggregate(values_test,groups,wp,wl,locator=False);locator_equal=aggregate(values_test,groups,np.full(len(groups),1/len(groups)),np.full(len(groups),1/len(groups)),prior=False)
    identity,_=frozen_v10_identity(m,test_raw);identity={v:x[:,0] for v,x in identity.items()}
    rows={'v13':metrics(pred,tg),'prior_only':metrics(prior,tg),'equal_locator_only':metrics(locator_equal,tg),'locator_shuffle':metrics(shuffled(pred),tg),'identity_fallback':metrics(identity,tg)}
    payload={'method':frozen['method'],'corpus':m['corpus'],'validation_role_audit_after_freeze':audit,'automatic_role_ranking_check':rank,'test':rows,'delta_v13_minus_identity':{k:rows['v13'][k]-rows['identity_fallback'][k] for k in ('frame_ap','frame_roc','within_centered_ap','within_centered_roc','within_macro_ap','within_macro_roc')},'frozen_config':str(freeze.resolve()),'frozen_config_sha256':sha256(freeze),'test_labels_used_for_state_or_selection':False};atomic_json(a.out,payload);print(json.dumps(payload,indent=2))
if __name__=='__main__':main()
