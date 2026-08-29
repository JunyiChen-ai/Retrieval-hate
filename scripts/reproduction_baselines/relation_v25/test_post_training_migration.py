#!/usr/bin/env python3
import json,tempfile,unittest,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).parent));from post_training_migration import verify
M=Path(__file__).with_name('POST_TRAINING_PRE_TEMPORAL_MIGRATION.json')
class T(unittest.TestCase):
 def test_old_to_new_chain(self):
  m=verify(allow_post_access=True);self.assertEqual(m['old_training_identities']['selector_sha256'],'0a6ce65fac40c1362c2e64208201c8028b2139a1c5bd3842798f8e8a1b484c91');self.assertIn('selector',m['new_evaluation_identities'])
 def test_old_identity_replacement_fails(self):
  with tempfile.TemporaryDirectory() as z:
   x=json.load(open(M));x['old_training_identities']['selector_sha256']=x['new_evaluation_identities']['selector']['sha256'];p=Path(z)/'m';p.write_text(json.dumps(x));self.assertRaises(RuntimeError,verify,p)
 def test_artifact_and_new_source_tamper_fail(self):
  for sec in ('immutable_artifacts','new_evaluation_identities'):
   with self.subTest(sec=sec),tempfile.TemporaryDirectory() as z:
    x=json.load(open(M));next(iter(x[sec].values()))['sha256']='0'*64;p=Path(z)/'m';p.write_text(json.dumps(x));self.assertRaises(RuntimeError,verify,p)
 def test_scope_and_seen_fail(self):
  for key in ('scope','seen'):
   with tempfile.TemporaryDirectory() as z:
    x=json.load(open(M));x['change_scope'].append('model_change') if key=='scope' else x.update(temporal_metrics_seen=True);p=Path(z)/'m';p.write_text(json.dumps(x));self.assertRaises(RuntimeError,verify,p)
if __name__=='__main__':unittest.main()
