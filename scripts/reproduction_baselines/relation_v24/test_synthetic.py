#!/usr/bin/env python3
import ast,json,sys,tempfile,unittest
from pathlib import Path
import torch
sys.path.insert(0,str(Path(__file__).resolve().parent))
from model import V24,cardinality_stable_lme
from train import load_bags,negative_control,train
H='a'*64
def row(vid='a',split='train',label=1,n=3):
 r={'corpus':'toy','split':split,'video_id':vid,'video_label':label,'global_causal_score':.2,'families':{'text':[[1.,2.,3.][:n]],'multimodal':[[3.,1.,2.][:n]]},'source_hashes':{'text_scores_sha256':H,'multimodal_scores_sha256':H,'v23_global_source_sha256':H}}
 return r
class TestV24(unittest.TestCase):
 def test_long_bag_replication_exact(self):
  x=torch.tensor([-2.,0.,1.],dtype=torch.float64);self.assertAlmostEqual(float(cardinality_stable_lme(x)),float(cardinality_stable_lme(x.repeat(100))),places=12)
 def test_exact_global_fallback(self):
  m=V24();g=torch.tensor(.123,dtype=torch.float64);fam={'text':[[1.,2.,4.]],'multimodal':[[-4.,3.,8.]]};z,f=m(g,fam)
  self.assertEqual(float(z),float(g));self.assertTrue(torch.equal(f,torch.full((3,),g,dtype=torch.float64)))
 def test_centered_local(self):
  m=V24();x=m.local({'text':[[1.,2.,9.]],'multimodal':[[4.,5.,7.]]});self.assertLess(abs(float(x.sum())),1e-12)
 def test_duplicate_expert_no_reweight(self):
  m=V24();a={'text':[[1.,2.,3.]],'multimodal':[[3.,1.,8.]]};b={'text':[[1.,2.,3.],[1.,2.,3.]],'multimodal':[[3.,1.,8.]]};self.assertTrue(torch.equal(m.local(a),m.local(b)))
 def test_nonleakage_rejects_temporal_key_and_val(self):
  with tempfile.TemporaryDirectory() as d:
   p=Path(d)/'x';r=row();r['annotations']=[];p.write_text(json.dumps(r)+'\n')
   with self.assertRaises(RuntimeError):load_bags(p,['a'],'toy','train',H)
   r=row();r['timestamps']=[0,1];p.write_text(json.dumps(r)+'\n')
   with self.assertRaises(RuntimeError):load_bags(p,['a'],'toy','train',H)
   p.write_text(json.dumps(row(split='val'))+'\n')
   with self.assertRaises(RuntimeError):load_bags(p,['a'],'toy','train',H)
 def test_finite_alignment_and_full_coverage(self):
  with tempfile.TemporaryDirectory() as d:
   p=Path(d)/'x';r=row();r['families']['text'][0]=[1.,2.];p.write_text(json.dumps(r)+'\n')
   with self.assertRaises(RuntimeError):load_bags(p,['a'],'toy','train',H)
   p.write_text(json.dumps(row())+'\n')
   with self.assertRaises(RuntimeError):load_bags(p,['a','b'],'toy','train',H)
 def test_negative_control_preserves_global_label(self):
  b={'a':{'global':1.,'label':1,'families':{'text':[[1.,2.]],'multimodal':[[2.,3.]]}},'b':{'global':-1.,'label':0,'families':{'text':[[8.,9.]],'multimodal':[[7.,6.]]}}}
  q=negative_control(b);self.assertEqual((q['a']['global'],q['a']['label']),(1.,1));self.assertEqual(set(q),set(b))
 def test_seed234_reproducible(self):
  bags={str(i):{'global':float(i%2)-.5,'label':i%2,'families':{'text':[[i,1.,2.]],'multimodal':[[2.,i,0.]]}} for i in range(6)}
  with tempfile.TemporaryDirectory() as d:
   a=Path(d)/'a.pt';b=Path(d)/'b.pt';train(bags,a);train(bags,b);x=torch.load(a,weights_only=False);y=torch.load(b,weights_only=False)
   for ex,ey in zip(x['history'],y['history']):
    for k in ex['state']:self.assertTrue(torch.equal(ex['state'][k],ey['state'][k]))
 def test_selector_never_calls_negative_control(self):
  tree=ast.parse((Path(__file__).resolve().parent/'selector.py').read_text())
  calls=[n.func.id for n in ast.walk(tree) if isinstance(n,ast.Call) and isinstance(n.func,ast.Name)]
  self.assertNotIn('negative_control',calls)
if __name__=='__main__':unittest.main()
