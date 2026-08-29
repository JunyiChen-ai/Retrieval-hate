#!/usr/bin/env python3
import json,sys
from pathlib import Path
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[2];sys.path[:0]=[str(HERE.parent)]
from relation_v4.io import sha256
from relation_v8.run import atomic_json
def main():
 formal=ROOT/'results/reproduction/relation_v20/mhclip_zh_formal';raw=ROOT/'results/reproduction/relation_v20/mhclip_zh_test_raw';sp=formal/'stable_centered_controls.json';s=json.load(open(sp));s['raw_manifest_sha256']=sha256(raw/'raw_manifest.json');atomic_json(sp,s)
 pp=formal/'artifact_provenance.json';p=json.load(open(pp));p['stable_centered_metrics']={'path':str(sp.resolve()),'sha256':sha256(sp)};atomic_json(pp,p)
 fp=formal/'frozen_config.json';f=json.load(open(fp));f['metadata_provenance_sha256']=sha256(pp);atomic_json(fp,f)
 ep=formal/'test_eval.json';e=json.load(open(ep));e['frozen_config_sha256']=sha256(fp);e['stable_centered_metrics_artifact']={'path':str(sp.resolve()),'sha256':sha256(sp)};e['stable_centered_identity']=s['stable_identity'];e['stable_centered_selected']=s['stable_selected'];e['deprecated_fields']=['test_identity.within_centered_ap/roc','test_selected.within_centered_ap/roc'];e['artifact_provenance_sha256']=sha256(pp);atomic_json(ep,e)
 mr=formal/'METADATA_MIGRATION_RECORD.json';m=json.load(open(mr));m['final_frozen_sha256']=sha256(fp);m['final_eval_sha256']=sha256(ep);m['final_provenance_sha256']=sha256(pp);atomic_json(mr,m)
 # Fail closed on the complete provenance chain.
 for split in ('val','test'):
  d=ROOT/f'results/reproduction/relation_v20/mhclip_zh_{split}_raw';cfg=d/'preregistered_config.json';rm=json.load(open(d/'raw_manifest.json'))
  if rm['config_sha256']!=sha256(cfg):raise RuntimeError('config/raw-manifest hash mismatch')
 if json.load(open(fp))['val_raw_manifest_sha256']!=sha256(ROOT/'results/reproduction/relation_v20/mhclip_zh_val_raw/raw_manifest.json'):raise RuntimeError('val chain mismatch')
 e=json.load(open(ep))
 if e['test_raw_manifest_sha256']!=sha256(raw/'raw_manifest.json') or e['frozen_config_sha256']!=sha256(fp):raise RuntimeError('test chain mismatch')
 enf=ROOT/'results/reproduction/relation_v19/mhclip_en_formal';esp=enf/'stable_centered_controls.json';eep=enf/'test_eval.json';es=json.load(open(esp));ee=json.load(open(eep));ee['stable_centered_identity']=es['stable_identity'];ee['stable_centered_selected']=es['stable_selected'];ee['deprecated_fields']=['test_identity.within_centered_ap/roc','test_selected.within_centered_ap/roc'];atomic_json(eep,ee)
 print(json.dumps({'provenance':sha256(pp),'frozen':sha256(fp),'eval':sha256(ep),'migration':sha256(mr)},indent=2))
if __name__=='__main__':main()
