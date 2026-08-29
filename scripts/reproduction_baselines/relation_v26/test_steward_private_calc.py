import json,unittest,sys
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).parent))
from steward_private_calc import metrics,load_v25,constant_ap_paired,constant_ap_gate
from steward import V25_MANIFEST

class T(unittest.TestCase):
 def test_constant_g_formula_is_recomputed(self):
  gt={'a':{'target_1hz':[0,1,1,0],'valid_1hz':[1,1,1,1]},'b':{'target_1hz':[0,1],'valid_1hz':[1,1]}}
  local={'a':[0.,.8,.7,.1],'b':[0.,1.]};constant={v:[.3]*len(gt[v]['target_1hz']) for v in gt}
  self.assertGreater(metrics(local,gt)['within_ap']-metrics(constant,gt)['within_ap'],.01)
 def test_authoritative_v25_live_files(self):
  root=V25_MANIFEST.parent;ids=sorted(json.loads(x)['video_id'] for x in open(root/'seed234_raw.jsonl'));p,m=load_v25(ids);self.assertEqual(sorted(p),ids);self.assertEqual(m['epoch'],2)
 def test_constant_point_pass_but_paired_ci_crosses_zero_fails(self):
  seed=[{'per_ap':[.60,.45]},{'per_ap':[.60,.45]}];constant={'per_ap':[.50,.50]};bm=[[0] for _ in range(1000)]+[[1] for _ in range(1000)];point,ci=constant_ap_paired(seed,constant,bm);self.assertGreaterEqual(point,.01);self.assertLessEqual(ci[0],0);self.assertFalse(constant_ap_gate(point,ci))
if __name__=='__main__':unittest.main()
