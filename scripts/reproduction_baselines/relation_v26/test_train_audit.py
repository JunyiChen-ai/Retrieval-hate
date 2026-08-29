import json,tempfile,unittest,sys
import shutil
from pathlib import Path
import torch
sys.path.insert(0,str(Path(__file__).parent))
from core import CTW,DESIGN_SHA,ctw_loss,permutation
from inference import load_authorized
from artifacts import atomic,sha
from reference import source_identities
from train import save_ckpt
from steward import verify_signed_report,canon,V25_MANIFEST,V25_MANIFEST_SHA
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
class T(unittest.TestCase):
 def test_finite_rf_checkpoint_schema_rejects_transformer_identity(self):
  with tempfile.TemporaryDirectory() as z:
   p=Path(z)/'c.pt';save_ckpt(p,CTW(),234,'real',0,0,{});x=torch.load(p,map_location='cpu',weights_only=False);self.assertEqual(x['schema'],'v26_finite_rf_checkpoint_v2');self.assertEqual(x['architecture'],'v26_finite_rf_dilated_v1');self.assertNotIn('enc.layers.0.self_attn.in_proj_weight',x['state']);x['schema']='v26_checkpoint_v1';torch.save(x,Path(z)/'old.pt');self.assertNotEqual(torch.load(Path(z)/'old.pt',weights_only=False)['schema'],'v26_finite_rf_checkpoint_v2')
 def test_reference_train_loader_source_tamper_identity(self):
  with tempfile.TemporaryDirectory() as z:
   authoritative=Path(__file__).with_name('train.py');copy=Path(z)/'train.py';shutil.copyfile(authoritative,copy);self.assertEqual(source_identities(copy)['train_loader_sha256'],source_identities()['train_loader_sha256']);copy.write_bytes(copy.read_bytes()+b'\n# tamper\n');self.assertNotEqual(source_identities(copy)['train_loader_sha256'],source_identities()['train_loader_sha256'])
 def row(self,v='a'):
  return {'id':v,'T':4,'X':[torch.randn(4,d) for d in (512,128,768)],'masks':[torch.ones(4,dtype=torch.bool) for _ in range(3)],'oof_b':[torch.zeros(4,d) for d in (512,128,768)],'G':torch.tensor(.2),'y':torch.tensor(1.)}
 def test_epoch0_exact_and_loss_finite(self):
  r=self.row();m=CTW();self.assertEqual(float(m(r['X'],r['masks'],r['G'])),float(r['G']));self.assertTrue(torch.equal(m.effects(r['X'],r['masks'],r['oof_b'],r['G']),torch.zeros(4)));loss,_,_=ctw_loss(m,r,r['oof_b']);self.assertTrue(torch.isfinite(loss))
 def test_permutation_moves_whole_tuple_and_grad_matched(self):
  rows=[self.row(str(i)) for i in range(5)];p,man=permutation(rows,234);self.assertTrue(all(x['nonself'] for x in man));self.assertEqual(sorted(x['T'] for x in p),[4]*5);self.assertEqual({x['recipient'] for x in man},{r['id'] for r in rows})
  a=CTW();b=CTW();b.load_state_dict(a.state_dict());la=ctw_loss(a,rows[0],rows[0]['oof_b'])[0];lb=ctw_loss(b,rows[0],rows[0]['oof_b'])[0];la.backward();lb.backward();self.assertTrue(all(torch.equal(x.grad,y.grad) for x,y in zip(a.parameters(),b.parameters())))
 def test_test_seal_failclosed(self):
  with tempfile.TemporaryDirectory() as z:
   d=Path(z);f=d/'test.features';f.write_bytes(b'opaque');sel=d/'sel';atomic(sel,{'design_sha256':DESIGN_SHA,'status':'FINAL_PASS','test_authorized':True});seal=d/'seal';atomic(seal,{'schema':'v26_test_seal_v1','design_sha256':DESIGN_SHA,'selection_sha256':sha(sel),'test_feature_path':str(f.resolve()),'test_feature_sha256':sha(f),'status':'FINAL_PASS'});load_authorized(sel,seal,f);f.write_bytes(b'tamper');self.assertRaises(RuntimeError,load_authorized,sel,seal,f)
 def test_steward_signed_report_no_caller_stats_and_tamper(self):
  with tempfile.TemporaryDirectory() as z:
   d=Path(z);pf=d/'epoch2';pf.write_bytes(b'p');chain={};
   for n in ('features','reference','train_run'):p=d/n;p.write_bytes(n.encode());chain[n]={'path':str(p),'sha256':sha(p)}
   pm=d/'pred';atomic(pm,{'files':{'real':{'2':{'path':str(pf),'sha256':sha(pf)}}},**chain});dummy=d/'bound';dummy.write_bytes(b'x');b={'path':str(dummy.resolve()),'sha256':sha(dummy)};sel=d/'sel';sel.write_bytes(b's');inputs={k:dict(b) for k in ('encrypted_cipher','encrypted_manifest','ledger','public_val_manifest','qc','taxonomy','private_source','raw_id_map','gt_reducer','calculator','v25_verifier','v25_reducer')};inputs['selection']={'path':str(sel.resolve()),'sha256':sha(sel)};inputs['v25_manifest']={'path':str(V25_MANIFEST.resolve()),'sha256':V25_MANIFEST_SHA};inputs['predictions']=[{'path':str(pm.resolve()),'sha256':sha(pm),'seed':234,'prediction_file':{'path':str(pf.resolve()),'sha256':sha(pf)},**chain}];inputs['bootstraps']=[dict(b)]*3
   inputs['v25_provenance']=dict(b);inputs['v25_reference']=dict(b);inputs['v25_checkpoints']=[dict(b)]*3
   x={'schema':'v26_signed_temporal_report_v2','design_sha256':DESIGN_SHA,'selected_epoch':2,'calculator_sha256':sha(Path(__file__).with_name('steward_private_calc.py')),'inputs':inputs,'all_ids':[f'v{i}' for i in range(32)],'positive_ids':['v0'],'mixed_ids':['v0'],'stats':{'computed':1},'gates':{'x':False},'all_gates_pass':False,'test_opened':False};key=Path('/home/jehc223/Hate-follow-up/results/steward_private/thvl_bench/v26_steward_key_frozen/ed25519_private.key');priv=Ed25519PrivateKey.from_private_bytes(key.read_bytes());x['signature_hex']=priv.sign(canon(x)).hex();atomic(d/'report',x);verify_signed_report(d/'report',pm,sel)
   attacker=Ed25519PrivateKey.generate();y=dict(x);y['stats']={'caller_fake_all_pass':True};y['signature_hex']=attacker.sign(canon({k:v for k,v in y.items() if k!='signature_hex'})).hex();atomic(d/'forged',y);self.assertRaises(Exception,verify_signed_report,d/'forged',pm,sel);dummy.write_bytes(b'tamper');self.assertRaises(Exception,verify_signed_report,d/'report',pm,sel)
if __name__=='__main__':unittest.main()
