#!/usr/bin/env python3
"""Adversarial checks for the V6 pre-pilot validity gates."""
import hashlib,json,os,tempfile
from relation_v2.protocol import frozen_splits
from relation_v4.io import sha256
from relation_v6.audited_data import verify_sparse_manifest
def write(path,value):
 with open(path,"w") as f: json.dump(value,f)
def manifest(root,vid,file_hash,audit_path):
 train=frozen_splits("hateclipseg")["train"]
 excluded={v:{"reason":"no_decodable_visual_stream","error_type":"Synthetic"} for v in train if v!=vid}
 aggregate=hashlib.sha256(f"{vid}\t{file_hash}\n".encode()).hexdigest(); return {"corpus":"hateclipseg","split":"train","k":16,"selection_cohort_ids":[vid],"excluded_train_ids":excluded,"media_audit":audit_path,"media_audit_sha256":sha256(audit_path),"root":root,"files":{vid:{"sha256":file_hash}},"prompt":"fixed test prompt","backend":"test backend","index_rule":"uniform K<=16 over aligned support","root_set_sha256":aggregate}
def reject(fn):
 try: fn()
 except RuntimeError: return
 raise AssertionError("invalid provenance was accepted")
def main():
 vid=frozen_splits("hateclipseg")["train"][0]
 with tempfile.TemporaryDirectory() as temporary:
  from relation_v6.audited_data import VGG_INDEX
  train=frozen_splits("hateclipseg")["train"]; audit_path=os.path.join(temporary,"media_audit.json"); audit={"corpus":"hateclipseg","split":"train","label_access":"none","frozen_train_ids":train,"records":{v:{"status":("decodable_visual_stream" if v==vid else "no_decodable_visual_stream")} for v in train}}; write(audit_path,audit)
  n=int(json.load(open(VGG_INDEX))[vid]["n_frames"]); duration=n-.2; length=int(__import__('math').ceil(min(n,duration))); starts=__import__('numpy').unique(__import__('numpy').rint(__import__('numpy').linspace(0,length-1,min(16,length))).astype(int)); segments=[{"start":float(s),"end":min(duration,float(n),float(s)+1),"score":.7,"response":"x"} for s in starts]; root=os.path.join(temporary,"raw"); os.makedirs(root); fp=os.path.join(root,vid+".json"); row={"video_id":vid,"duration":duration,"segments":segments}; write(fp,row); mp=os.path.join(temporary,"manifest.json"); payload=manifest(root,vid,sha256(fp),audit_path); write(mp,payload); info=verify_sparse_manifest(mp,train); assert info["cohort"]==[vid]
  bad=dict(payload); bad["files"]={vid:{"sha256":"0"*64}}; write(mp,bad); reject(lambda:verify_sparse_manifest(mp,train))
  write(fp,{**row,"segments":[{"start":n-1.,"end":n+1.,"score":.7}]}); payload=manifest(root,vid,sha256(fp),audit_path); write(mp,payload); reject(lambda:verify_sparse_manifest(mp,train))
  write(fp,row); payload=manifest(root,vid,sha256(fp),audit_path); payload["selection_cohort_ids"]=[]; payload["excluded_train_ids"][vid]={"reason":"no_decodable_visual_stream"}; payload["files"]={}; write(mp,payload); reject(lambda:verify_sparse_manifest(mp,train))
  payload=manifest(root,vid,sha256(fp),audit_path); payload["selection_cohort_ids"]=[frozen_splits("hateclipseg")["val"][0]]; write(mp,payload); reject(lambda:verify_sparse_manifest(mp,train))
 print("Relation-V6 audit smoke: PASS")
if __name__=="__main__": main()
