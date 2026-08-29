import hashlib,json,math,tempfile,unittest,sys
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).parent))
from artifacts import atomic,sha
from core import DESIGN_SHA,fold
from feature_manifest import MODELS,verify
import reference,train,val_predict

def cohort_ids(prefix,n=6):
 out=[];seen=set();i=0
 while len(out)<n:
  v=f'{prefix}{i}';k=fold(v)
  if k not in seen or len(seen)==5:out.append(v);seen.add(k)
  i+=1
 return out
def features(root,split,ids,T=3):
 root=Path(root);root.mkdir();rd=root/'records';rd.mkdir();mom=root/'mom.npz';np.savez(mom,x=np.array([0.]));records={}
 for ii,v in enumerate(ids):
  secs=[]
  for t in range(T):secs.append({'second':t,'visual':([float(ii+t)/10]*512),'audio':([float(ii-t)/10]*128),'text':([float(t)/10]*768),'availability':[1,1,1]})
  r={'corpus':'thvl','split':split,'opaque_id':v,'duration':float(T),'G':float(ii-2)/4,'G_domain':'signed_logit','seconds':secs,'source_hashes':{'synthetic':'true'}};p=rd/f'{v}.json';atomic(p,r);records[v]=str(p.resolve())
 gb={'domain':'signed_logit','source_sha256':'1'*64,'raw_root_sha256':'2'*64,'raw_seal_sha256':'3'*64,'split_manifest_sha256':'4'*64,'finalizer_sha256':'5'*64};norm={'path':str(mom.resolve()),'sha256':sha(mom),'fit_split':'train'};rh=hashlib.sha256((gb['raw_root_sha256']+gb['raw_seal_sha256']+gb['finalizer_sha256']+gb['split_manifest_sha256']+norm['sha256']+''.join(v+'\t'+sha(records[v])+'\n' for v in sorted(ids))).encode()).hexdigest();m={'schema':'v26_features_v1','design_sha256':DESIGN_SHA,'corpus':'thvl','split':split,'ids':ids,'records':records,'models':MODELS,'normalization':norm,'G_binding':gb,'root_sha256':rh,'labels_or_gt_read':False};atomic(root/'manifest.json',m);verify(root/'manifest.json');return root/'manifest.json'
class E2E(unittest.TestCase):
 def test_six_video_reference_train_predict(self):
  with tempfile.TemporaryDirectory() as z:
   d=Path(z);ids=cohort_ids('tr');self.assertEqual(len(set(map(fold,ids))),5);fm=features(d/'f','train',ids);labels={v:int(i==5) for i,v in enumerate(ids)};lm={'schema':'v26_train_video_labels_v1','design_sha256':DESIGN_SHA,'split':'train','ids':ids,'labels':labels,'feature_manifest_sha256':sha(fm),'temporal_labels_read':False};atomic(d/'labels.json',lm);rows,_=train.load_rows(fm,d/'labels.json');vids=cohort_ids('va');vf=features(d/'vf','val',vids);vr,_=train.load_rows(vf);ref=reference.build(rows,vr,d/'ref',fm,d/'labels.json',vf);run=train.run(fm,d/'labels.json',d/'ref/manifest.json',d/'run',seed=234,epochs=8);self.assertTrue(run['matched_steps']);self.assertEqual(run['epochs'],list(range(9)));self.assertEqual(set(ref['oof']),set(ids));self.assertEqual(set(ref['val_backgrounds']),set(vids))
   pm=val_predict.run(vf,d/'ref/manifest.json',d/'run/manifest.json',d/'pred');self.assertEqual(set(pm['files']),{'real','permuted','negative_mean'});p=json.load(open(pm['files']['real']['0']['path']));self.assertTrue(all(r['epoch0_G_exact'] and all(x==.5 for x in r['local']) for r in p['records'].values()))
   mutations=[];bad=json.loads((d/'ref/manifest.json').read_text());bad['sources']['core_sha256']='0'*64;mutations.append(bad);bad=json.loads((d/'ref/manifest.json').read_text());bad['folds']['0']['negative_ids']=bad['folds']['0']['negative_ids'][1:];mutations.append(bad);bad=json.loads((d/'ref/manifest.json').read_text());bad['folds']['0']['modality_target_counts'][0]+=1;mutations.append(bad);bad=json.loads((d/'ref/manifest.json').read_text());bad['folds']['0']['zero_influence_gates'][0]['gradient_zero']=False;mutations.append(bad);bad=json.loads((d/'ref/manifest.json').read_text());next(iter(bad['oof'].values()))['source_state_sha256']='0'*64;mutations.append(bad)
   for i,bad in enumerate(mutations):atomic(d/f'badref{i}.json',bad);self.assertRaises(RuntimeError,reference.verify_reference,d/f'badref{i}.json',fm,d/'labels.json',vf)
   oe=next(iter(ref['oof'].values()));Path(oe['path']).write_bytes(Path(oe['path']).read_bytes()+b'x');self.assertRaises(RuntimeError,reference.verify_reference,d/'ref/manifest.json',fm,d/'labels.json',vf)
if __name__=='__main__':unittest.main()
