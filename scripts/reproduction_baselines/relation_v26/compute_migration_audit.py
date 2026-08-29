#!/usr/bin/env python3
"""Label/GT-blind workload census and finite-RF numerical/runtime audit."""
import argparse,copy,json,os,time,hashlib
from pathlib import Path
import torch
from artifacts import atomic,sha
from core import CTW,MIGRATION_SHA,ARCH,ch
from feature_manifest import verify as verify_features

def deterministic():
 os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8');torch.use_deterministic_algorithms(True);torch.backends.cudnn.deterministic=True;torch.backends.cudnn.benchmark=False
def workload(feature_manifest,out):
 m=verify_features(feature_manifest);rows=[]
 for v in m['ids']:
  p=m['records'][v];r=json.load(open(p));te=sum(any(s['availability']) for s in r['seconds']);rows.append({'opaque_id':v,'T_eff':te,'feature_record_sha256':sha(p)})
 first=rows[:20]
 x={'schema':'v26_finite_rf_workload_v1','migration_sha256':MIGRATION_SHA,'architecture':ARCH,'feature_manifest':{'path':str(Path(feature_manifest).resolve()),'sha256':sha(feature_manifest),'root_sha256':m['root_sha256']},'ordered_rows':rows,'first20_ids':[r['opaque_id'] for r in first],'sum_Teff_first20':sum(r['T_eff'] for r in first),'sum_Teff_all':sum(r['T_eff'] for r in rows),'rows_root_sha256':ch(rows),'labels_or_gt_read':False,'test_read':False,'producer_sha256':sha(__file__)};atomic(out,x);return x
def sample(T,device):
 gen=torch.Generator().manual_seed(26026);xs=[torch.randn(T,d,generator=gen).to(device) for d in (512,128,768)];ms=[(torch.rand(T,generator=gen)>.15).to(device) for _ in range(3)];bs=[torch.randn(x.shape,generator=gen).to(device) for x in xs];return xs,ms,bs,torch.tensor(.37,device=device),torch.tensor(1.,device=device)
def compare(T,device):
 base=CTW(model_seed=26026).to(device);gen=torch.Generator().manual_seed(9)
 with torch.no_grad():base.contribution_head.weight.copy_(.01*torch.randn(base.contribution_head.weight.shape,generator=gen).to(device));base.contribution_head.bias.fill_(.002)
 xs,ms,bs,g,y=sample(T,device);fast=copy.deepcopy(base);slow=copy.deepcopy(base)
 t=time.perf_counter();ef=fast.effects(xs,ms,bs,g);lf=torch.nn.functional.binary_cross_entropy_with_logits(fast(xs,ms,g),y)+ef.square().mean();lf.backward();
 if device.type=='cuda':torch.cuda.synchronize()
 ft=time.perf_counter()-t;t=time.perf_counter();es=slow.effects_slow(xs,ms,bs,g);ls=torch.nn.functional.binary_cross_entropy_with_logits(slow(xs,ms,g),y)+es.square().mean();ls.backward();
 if device.type=='cuda':torch.cuda.synchronize()
 st=time.perf_counter()-t;ve=(ef-es).abs();grad=[];ok=True
 for (n,p),(n2,q) in zip(fast.named_parameters(),slow.named_parameters()):
  e=(p.grad-q.grad).abs();tol=1e-6+1e-5*q.grad.abs();gate=e<=tol;ok=ok and bool(gate.all());grad.append({'name':n,'max_abs':float(e.max()),'max_gate_ratio':float((e/tol).max()),'gate':bool(gate.all())})
 again=fast.effects(xs,ms,bs,g);etol=1e-6+1e-5*es.abs();return {'T':T,'device':str(device),'fast_seconds':ft,'slow_seconds':st,'effect_max_abs':float(ve.max()),'effect_max_gate_ratio':float((ve/etol).max()),'effect_gate':bool(torch.all(ve<=etol)),'loss_abs':abs(float(lf-ls)),'gradient_gate':ok,'gradient':grad,'fast_repeat_bitexact':bool(torch.equal(ef,again))}
def scaling(T,device):
 m=CTW(model_seed=26026).to(device);xs,ms,bs,g,_=sample(T,device);torch.cuda.reset_peak_memory_stats();t=time.perf_counter()
 with torch.no_grad():m.effects(xs,ms,bs,g)
 if device.type=='cuda':torch.cuda.synchronize()
 return {'wall_seconds':time.perf_counter()-t,'peak_memory_bytes':torch.cuda.max_memory_allocated()}
def main():
 p=argparse.ArgumentParser();p.add_argument('--features',required=True);p.add_argument('--workload-out',required=True);p.add_argument('--audit-out',required=True);p.add_argument('--skip-cpu',action='store_true');a=p.parse_args();deterministic();w=workload(a.features,a.workload_out);devs=[] if a.skip_cpu else [torch.device('cpu')]
 if torch.cuda.is_available():devs.append(torch.device('cuda'))
 res=[]
 for d in devs:
  for T in (17,61,301):res.append(compare(T,d))
 scale={}
 if torch.cuda.is_available():scale={str(T):scaling(T,torch.device('cuda')) for T in (900,3600)}
 atomic(a.audit_out,{'schema':'v26_finite_rf_compute_audit_v1','migration_sha256':MIGRATION_SHA,'architecture':ARCH,'workload_sha256':sha(a.workload_out),'results':res,'scaling_seconds':scale,'runtime':{'torch':torch.__version__,'gpu':torch.cuda.get_device_name(0) if torch.cuda.is_available() else None},'source_sha256':sha(__file__),'labels_or_gt_read':False,'test_read':False})
if __name__=='__main__':main()
