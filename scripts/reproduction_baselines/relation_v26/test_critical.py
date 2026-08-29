import copy,json,tempfile,unittest,sys
from pathlib import Path
import numpy as np,torch
sys.path.insert(0,str(Path(__file__).parent));from core import *
from inference import fallback
from extract_thvl_1hz import overlap
class T(unittest.TestCase):
 def setUp(self):torch.manual_seed(1);torch.use_deterministic_algorithms(True)
 def test_decoder_zero_influence_bits_and_grad(self):
  d=Decoder(5).double();x=torch.randn(30,5,dtype=torch.double,requires_grad=True);m=torch.ones(30,dtype=torch.bool);a=d(x,m,15);x2=x.detach().clone();x2[14:17]=999;b=d(x2,m,15);self.assertTrue(torch.equal(a,b));a.sum().backward();self.assertTrue(torch.equal(x.grad[14:17],torch.zeros_like(x.grad[14:17])))
 def test_availability_and_effect_epoch0(self):
  m=CTW((2,2,2)).double();xs=[torch.randn(4,2,dtype=torch.double) for _ in range(3)];ms=[torch.tensor([1,1,1,1],dtype=torch.bool),torch.tensor([1,0,1,1],dtype=torch.bool),torch.tensor([0,0,0,0],dtype=torch.bool)];bs=[torch.zeros_like(x) for x in xs];g=torch.tensor(-3.25,dtype=torch.double);self.assertTrue(torch.equal(m(xs,ms,g),g));self.assertTrue(torch.equal(m.effects(xs,ms,bs,g),torch.zeros(4,dtype=torch.double)));self.assertEqual(fallback(float(g),4),[-3.25]*4)
 def test_deterministic(self):
  m=CTW((2,2,2)).eval();xs=[torch.randn(5,2) for _ in range(3)];ms=[torch.ones(5,dtype=torch.bool) for _ in range(3)];bs=[torch.zeros_like(x) for x in xs];a=m.effects(xs,ms,bs,torch.tensor(1.));b=m.effects(xs,ms,bs,torch.tensor(1.));self.assertTrue(torch.equal(a,b))
 def test_migrated_rf_mask_boundary_and_oracle_gradient(self):
  T=61;m=CTW((2,2,2),26026);self.assertFalse(any(isinstance(z,torch.nn.TransformerEncoder) for z in m.modules()));
  with torch.no_grad():m.contribution_head.weight.fill_(.01)
  xs=[torch.randn(T,2) for _ in range(3)];ms=[torch.ones(T,dtype=torch.bool) for _ in range(3)];ms[0][30]=False;ms[1][30]=False;ms[2][30]=False;bs=[torch.randn_like(x) for x in xs];g=torch.tensor(.2);fast=m.effects(xs,ms,bs,g);slow=m.effects_slow(xs,ms,bs,g);self.assertEqual(len(fast),T);self.assertTrue(torch.allclose(fast,slow,atol=1e-4,rtol=1e-4));self.assertEqual(float(fast[30]),0.)
  loss=fast.square().mean()+m(xs,ms,g).square();loss.backward();gr=[p.grad for p in m.parameters() if p.requires_grad];self.assertTrue(all(q is not None and torch.isfinite(q).all() for q in gr));self.assertTrue(any(float(q.abs().sum())>0 for q in gr))
  # A target cannot affect contribution tokens outside the frozen radius 15.
  a,_=m.contributions_batch(xs,ms);cf=[x.clone() for x in xs]
  for f in range(3):cf[f][5]=bs[f][5]
  b,_=m.contributions_batch(cf,ms);changed=(a-b).abs()>1e-6;self.assertFalse(bool(changed[21:].any()))
  xg=[x.detach().clone().requires_grad_(True) for x in xs];m.zero_grad();m.effects(xg,ms,bs,g)[5].backward();self.assertTrue(all(float(x.grad[36:].abs().max())==0. for x in xg))
 def test_permutation_moves_x_mask_b_together(self):
  rows=[]
  for i in range(10):rows.append({'id':f'v{i}','T':3,'X':[[i]],'masks':[[i%2]],'oof_b':[[i+.5]],'G':i,'y':i%2})
  out,man=permutation(rows,234);self.assertTrue(all(r['G']==i for i,r in enumerate(out)));self.assertTrue(all(z['T']==3 and z['raw_sha']==ch(out[i]['X']) and z['mask_sha']==ch(out[i]['masks']) and z['b_sha']==ch(out[i]['oof_b']) for i,z in enumerate(man)))
 def test_three_bootstrap_cohorts(self):
  a=three_bootstraps([0,1]*16,5,7,B=5);self.assertEqual([len(x) for x in a['all32']],[32]*5);self.assertEqual([len(x) for x in a['positive']],[5]*5);self.assertEqual([len(x) for x in a['mixed']],[7]*5);self.assertTrue(all(len(set(np.array([0,1]*16)[x]))==2 for x in a['all32']))
 def test_design_sha(self):self.assertEqual(DESIGN_SHA,'203960db2e45cd9f4e25eb61f9864ca6ba204fd184ffed1fd415b94cacccc859')
 def test_golden_overlap_boundary(self):
  self.assertEqual(overlap(1,2,0,3),0);self.assertEqual(overlap(1,2,1,3),1);self.assertAlmostEqual(overlap(.5,1.5,1,3),.5);self.assertAlmostEqual(overlap(2.8,4,2,3),.2)
if __name__=='__main__':unittest.main()
