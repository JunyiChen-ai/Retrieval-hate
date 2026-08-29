#!/usr/bin/env python3
"""Frozen first-20 one-arm/one-epoch compute projection; no val labels/GT/test."""
import argparse,json,time,torch
from pathlib import Path
from artifacts import atomic,sha
from core import CTW,MIGRATION_SHA,ARCH,ctw_loss
from train import load_rows,attach_oof,preload
from reference import verify_reference
def main():
 p=argparse.ArgumentParser();p.add_argument('--features',required=True);p.add_argument('--labels',required=True);p.add_argument('--reference',required=True);p.add_argument('--workload',required=True);p.add_argument('--out',required=True);a=p.parse_args();torch.use_deterministic_algorithms(True);torch.backends.cudnn.deterministic=True;torch.backends.cudnn.benchmark=False
 if not torch.cuda.is_available() or 'RTX 5090' not in torch.cuda.get_device_name(0):raise RuntimeError('pinned RTX 5090 required')
 w=json.load(open(a.workload));t=time.perf_counter();raw=json.load(open(a.reference));ref=verify_reference(a.reference,a.features,a.labels,raw['inputs']['val_features']['path']);reference_seconds=time.perf_counter()-t
 rows,_=load_rows(a.features,a.labels);attach_oof(rows,ref);ids=w['first20_ids'];dev=torch.device('cuda');torch.cuda.reset_peak_memory_stats();t=time.perf_counter();preload(rows,dev);torch.cuda.synchronize();preload_seconds=time.perf_counter()-t;preload_peak=torch.cuda.max_memory_allocated();rr=[next(r for r in rows if r['id']==v) for v in ids]
 m=CTW(model_seed=234).to(dev);opt=torch.optim.AdamW(m.parameters(),lr=3e-4,weight_decay=1e-4);torch.cuda.synchronize();t=time.perf_counter()
 for s in range(0,len(rr),4):
  losses=[ctw_loss(m,r,r['oof_b'])[0] for r in rr[s:s+4]];opt.zero_grad();torch.stack(losses).mean().backward();torch.nn.utils.clip_grad_norm_(m.parameters(),1.);opt.step()
 torch.cuda.synchronize();epoch_seconds=time.perf_counter()-t;hours=(epoch_seconds*(w['sum_Teff_all']/w['sum_Teff_first20'])*3*8+reference_seconds+preload_seconds)/3600
 atomic(a.out,{'schema':'v26_finite_rf_first20_benchmark_v2','migration_sha256':MIGRATION_SHA,'architecture':ARCH,'device':torch.cuda.get_device_name(0),'float':'float32','seed':234,'cone_chunk':64,'preload_strategy':'all314_once','preload_seconds':preload_seconds,'preload_peak_bytes':preload_peak,'gpu_total_bytes':torch.cuda.get_device_properties(0).total_memory,'preload_fits':preload_peak<torch.cuda.get_device_properties(0).total_memory,'first20_epoch_one_arm_seconds':epoch_seconds,'reference_load_seconds':reference_seconds,'sum_Teff_first20':w['sum_Teff_first20'],'sum_Teff_all314':w['sum_Teff_all'],'projected_three_arms_eight_epochs_hours':hours,'gate_le_12h':hours<=12,'inputs':{k:{'path':str(Path(v).resolve()),'sha256':sha(v)} for k,v in {'features':a.features,'labels':a.labels,'reference':a.reference,'workload':a.workload}.items()},'source_sha256':sha(__file__),'val_labels_or_gt_read':False,'test_read':False})
if __name__=='__main__':main()
