#!/usr/bin/env python3
"""Only authoritative V6 selector: fixed two regularization candidates."""
import argparse,json,os,shutil,subprocess,sys
HERE=os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0,os.path.dirname(HERE))
from relation_v4.io import sha256
def main():
 p=argparse.ArgumentParser(); p.add_argument("--manifest",required=True); p.add_argument("--out-dir",required=True); p.add_argument("--device",default="cuda"); p.add_argument("--seed",type=int,default=234); a=p.parse_args()
 if os.path.exists(a.out_dir) and os.listdir(a.out_dir): raise RuntimeError("outer out-dir must be absent or empty")
 os.makedirs(a.out_dir,exist_ok=True); rows=[]
 # All structure and optimization values are preregistered here; callers may
 # choose only manifest, output, device, and seed.
 for reg in (0.,.01):
  out=os.path.join(a.out_dir,"candidate_reg_%s"%("0" if reg==0 else "001")); cmd=[sys.executable,os.path.join(HERE,"train.py"),"--manifest",a.manifest,"--out-dir",out,"--device",a.device,"--seed",str(a.seed),"--epochs","20","--batch-size","16","--lr","2e-4","--hidden","32","--heads","4","--window","12","--temperature",".2","--dropout",".1","--regularization",str(reg),"--teacher-weight",".1","--lambda-grid","0,.25,.5,1,1.5,2"]; subprocess.run(cmd,check=True); meta=json.load(open(os.path.join(out,"train_meta.json"))); rows.append({"regularization":reg,"candidate_dir":os.path.abspath(out),"selected_epoch":meta["selected_epoch"],"selected_locator_scale":meta["selected_locator_scale"],"validation_frame_ap":meta["selected_validation_frame_ap"],"validation_frame_roc":meta["selected_validation_frame_roc"],"model_sha256":meta["model_sha256"]})
 best=max(rows,key=lambda r:(r["validation_frame_ap"],r["validation_frame_roc"])); selected=os.path.join(a.out_dir,"selected"); os.makedirs(selected)
 for name in ("model.pth","train_meta.json","COMPLETE.json"): shutil.copy2(os.path.join(best["candidate_dir"],name),os.path.join(selected,name))
 payload={"method":"relation_v6_outer_selection","fixed_candidates":{"regularization":[0.,.01],"teacher_weight":.1,"epochs":20,"lambda_grid":[0,.25,.5,1,1.5,2]},"selection_rule":"max validation pooled Frame AP; ROC tie-break","candidates":rows,"selected":best,"test_opened":False}; path=os.path.join(selected,"OUTER_SELECTION.json"); json.dump(payload,open(path+".tmp","w"),indent=2); os.replace(path+".tmp",path); print(json.dumps(payload,indent=2))
if __name__=="__main__": main()
