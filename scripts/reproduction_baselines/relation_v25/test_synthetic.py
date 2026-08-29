#!/usr/bin/env python3
import hashlib,hmac,json,tempfile,unittest,sys
from pathlib import Path
import numpy as np,torch
sys.path.insert(0,str(Path(__file__).resolve().parent))
from core import *
from reference_builder import build,verify,load_bags
from train import permute,verify_permutation_manifest,verify_declaration
from selector import removal_ratio
from inference import exact_global_fallback_1hz
from val_predict import verify_outputs
class T(unittest.TestCase):
 def test_fractional_replication_exact(self):
  x=torch.tensor([-2.,0.,1.,3.,4.],dtype=torch.float64);self.assertTrue(torch.equal(fractional_lme(x),fractional_lme(x.repeat(7))))
 def test_fractional_short_bag_exact_max(self):self.assertTrue(torch.equal(fractional_lme([2.]),torch.tensor(2.,dtype=torch.float64)))
 def test_ecdf_ties_finite(self):
  z=ecdf_logit(np.array([0,1,2]),[-1,0,0,3]);self.assertTrue(np.isfinite(z).all());self.assertLess(z[0],z[2])
 def test_reducer_boundary_overlap_mask(self):
  s,m=reduce_1hz([{'start':0,'end':1,'logit':0},{'start':1,'end':2,'logit':2}],3);self.assertEqual(m,[1,1,0]);self.assertAlmostEqual(s[0],.5);self.assertTrue(np.isnan(s[2]))
 def test_global_fallback_constant(self):
  s,m=reduce_1hz([{'start':0,'end':2,'logit':-1.25}],2);self.assertEqual(m,[1,1]);self.assertEqual(s[0],s[1])
  raw=-3.930223519881542;z,m=exact_global_fallback_1hz(raw,2.2);self.assertTrue(all(x.hex()==raw.hex() for x in z));self.assertEqual(m,[1,1,1])
 def test_negative_loss_and_epoch0(self):
  m=V25();z=torch.tensor([[0.,1.],[0.,1.]],dtype=torch.float64);L,e=m(-2,z);self.assertTrue(torch.equal(L,torch.tensor(-2.,dtype=torch.float64)));q=loss_one(m,-2,z,0);q.backward();self.assertIsNotNone(m.b.grad)
 def test_real_permuted_grad_update_equivalence(self):
  a,b=V25(True),V25(True);b.load_state_dict(a.state_dict());z=torch.tensor([[0.,1.],[1.,0.]],dtype=torch.float64)
  la=loss_one(a,.2,z,1);lb=loss_one(b,.2,z,1);la.backward();lb.backward();self.assertEqual([p.requires_grad for p in a.parameters()],[p.requires_grad for p in b.parameters()]);self.assertTrue(all(torch.equal(x.grad,y.grad) for x,y in zip(a.parameters(),b.parameters())))
 def test_permutation_counts(self):
  rows=[{'id':f'v{i}','y':i%2,'g':i,'z':np.full((2,3),i)} for i in range(10)];ms={};
  for s in SEEDS:p,m=permute(rows,s);ms[str(s)]=m
  self.assertTrue(verify_permutation_manifest(rows,ms));bad=json.loads(json.dumps(ms));bad['234']['donor_folds']['v0']=99;self.assertRaises(RuntimeError,verify_permutation_manifest,rows,bad)
 def test_final_declaration(self):
  # The declaration is intentionally historical after the post-access identity
  # correction; current-source verification belongs to the correction verifier.
  d=json.load(open(Path(__file__).with_name('FINAL_PROTOCOL_DECLARATION.json')));self.assertEqual(d['status'],'FINAL_AUTHORITATIVE_PRETRAINING_DECLARATION')
 def test_val_prediction_stale_and_raw_tamper(self):
  with tempfile.TemporaryDirectory() as z:
   d=Path(z);files={}
   for s in SEEDS:
    for k in ('raw','shuffle'):
     p=d/f'seed{s}_{k}.jsonl';p.write_text('{}\n');files[p.name]=sha(p)
   ck={str(s):str(s)*64 for s in SEEDS};st={str(s):('a'+str(s))*32 for s in SEEDS};di=json.load(open(Path(__file__).with_name('FINAL_PROTOCOL_DECLARATION.json')))['identities'];base={'schema':'v25_val_predictions_v1','status':'VAL_LABEL_GT_BLIND','epoch':2,'ids_sha256':'i'*64,'evidence_manifest_sha256':di['authoritative_val_evidence_manifest']['sha256'],'evidence_config_sha256':di['authoritative_val_evidence_config']['sha256'],'reference_manifest_sha256':'r'*64,'checkpoint_sha256_by_seed':ck,'state_sha256_by_seed':st,'files':files,'shuffle_rule':'x','producer_sha256':sha(Path(__file__).with_name('val_predict.py')),'reducer_sha256':sha(Path(__file__).with_name('inference.py')),'labels_read':False,'test_read':False};(d/'manifest.json').write_text(json.dumps(base));self.assertRaises(RuntimeError,verify_outputs,d,'i'*64,2,ck,st,'r'*64)
   for field in ('epoch','ids_sha256','checkpoint_sha256_by_seed'):
    x=dict(base);x[field]=3 if field=='epoch' else ('x'*64 if field=='ids_sha256' else {});(d/'manifest.json').write_text(json.dumps(x));self.assertRaises(RuntimeError,verify_outputs,d,'i'*64,2,ck,st,'r'*64)
   (d/'manifest.json').write_text(json.dumps(base));(d/'seed234_raw.jsonl').write_text('tamper');self.assertRaises(RuntimeError,verify_outputs,d,'i'*64,2,ck,st,'r'*64)
 def _bags(self,p):
  rows=[];ids=[]
  for k in range(5):
   i=0
   while fold(f'n{k}_{i}')!=k:i+=1
   ids.append(f'n{k}_{i}')
  ids += [f'p{i}' for i in range(5)]
  for i,v in enumerate(ids):rows.append({'corpus':'toy','split':'train','video_id':v,'video_label':int(i>=5),'global_causal_score':0.,'families':{'text':[[i,i+1]],'multimodal':[[i-.5,i+.5]]},'source_hashes':{}})
  p.write_text(''.join(json.dumps(x)+'\n' for x in rows))
 def test_reference_hash_tamper_and_crossfit(self):
  with tempfile.TemporaryDirectory() as z:
   d=Path(z);b=d/'b';self._bags(b);build(b,d/'r');self.assertEqual(verify(d/'r',b)['full_reference_materializations'],1);p=d/'r/text_full.json';p.write_text('[99]\n');self.assertRaises(RuntimeError,verify,d/'r',b)
 def test_source_tamper(self):
  with tempfile.TemporaryDirectory() as z:
   d=Path(z);b=d/'b';self._bags(b);build(b,d/'r');b.write_text(b.read_text()+'\n');self.assertRaises(RuntimeError,verify,d/'r',b)
 def test_bag_zero_length_mismatch_nonfinite_g(self):
  with tempfile.TemporaryDirectory() as z:
   p=Path(z)/'b';self._bags(p)
   for mode in ('zero','mismatch','g'):
    rows=[json.loads(x) for x in p.read_text().splitlines()]
    if mode=='zero':rows[0]['families']['text']=[[]]
    elif mode=='mismatch':rows[0]['families']['text'][0].append(1)
    else:rows[0]['global_causal_score']=float('nan')
    q=Path(z)/mode;q.write_text(''.join(json.dumps(x)+'\n' for x in rows));self.assertRaises(RuntimeError,load_bags,q)
 def test_reference_producer_tamper(self):
  with tempfile.TemporaryDirectory() as z:
   d=Path(z);b=d/'b';self._bags(b);build(b,d/'r');p=d/'r/manifest.json';x=json.load(open(p));x['producer_sha256']='0'*64;p.write_text(json.dumps(x));self.assertRaises(RuntimeError,verify,d/'r')
 def test_shuffle_ratio(self):
  self.assertAlmostEqual(removal_ratio([.6,.7,.8],[.52,.54,.56]),.8);self.assertRaises(RuntimeError,removal_ratio,[.5,.5,.5],[.5,.5,.5])
 def test_seal_fail_closed(self):
  with tempfile.TemporaryDirectory() as z:
   d=Path(z);sel=d/'s';sel.write_text(json.dumps({'status':'VIDEO_VAL_PASS_PENDING_TEST_SEAL'}));key=d/'k';key.write_bytes(b'k');chain={}
   for n in ('test_records','evidence_manifest','evidence_config'):
    p=d/n;p.write_text('opaque label-blind bytes');chain[n]={'path':str(p.resolve()),'sha256':sha(p)}
   ap=d/'a';payload={'status':'TEST_SEAL_APPROVED','selected_sha256':sha(sel),'test_input_chain':chain};msg=json.dumps(payload,sort_keys=True,separators=(',',':')).encode();payload['signature_hmac_sha256']=hmac.new(b'k',msg,hashlib.sha256).hexdigest();ap.write_text(json.dumps(payload));import subprocess;subprocess.run([sys.executable,str(Path(__file__).with_name('seal.py')),'--selected',str(sel),'--approval',str(ap),'--key',str(key),'--out',str(d/'o')],check=True);o=json.load(open(d/'o'));self.assertTrue(o['test_seal_signed']);self.assertEqual(o['approved_test_input_chain'],chain)
if __name__=='__main__':unittest.main()
