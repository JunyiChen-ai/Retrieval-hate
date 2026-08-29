#!/usr/bin/env python3
import json,tempfile,unittest,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).parent));from post_access_correction import verify,correction_verify_declaration
C=Path(__file__).with_name('POST_ACCESS_PROVENANCE_CORRECTION.json')
class T(unittest.TestCase):
 def mutate(self,fn):
  z=tempfile.TemporaryDirectory();p=Path(z.name)/'c.json';x=json.load(open(C));fn(x);p.write_text(json.dumps(x));return z,p
 def test_valid_label_blind_provenance(self):self.assertEqual(verify()['status'],'POST_ACCESS_IDENTITY_REPAIR_ONLY')
 def test_missing_correction_fails(self):
  with tempfile.TemporaryDirectory() as z:self.assertRaises(FileNotFoundError,verify,Path(z)/'missing')
 def test_wrong_old_new_and_artifact_fail(self):
  changes=[lambda x:x['old_identities'].update(selector_sha256='0'*64),lambda x:x['new_identities']['selector'].update(sha256='0'*64),lambda x:x['frozen_artifacts']['encrypted_artifact'].update(sha256='0'*64)]
  for f in changes:
   z,p=self.mutate(f)
   with z:self.assertRaises(RuntimeError,verify,p)
 def test_semantic_or_binding_tamper_fails(self):
  changes=[lambda x:x.update(metrics_or_gates_changed=True),lambda x:x['unchanged_evaluation_bindings'].update(taxonomy_sha256='0'*64),lambda x:x['observed_aggregate'].update(paired_ci_gate_pass=True)]
  for f in changes:
   z,p=self.mutate(f)
   with z:self.assertRaises(RuntimeError,verify,p)
 def test_unauthorized_third_declaration_identity_fails(self):
  from core import canon_hash
  with tempfile.TemporaryDirectory() as z:
   d=Path(z);src=Path(__file__).with_name('FINAL_PROTOCOL_DECLARATION.json');x=json.load(open(src));fake=d/'core.py';fake.write_text('third identity drift\n');x['identities']['core']={'path':str(fake),'sha256':'0'*64};x['identity_set_sha256']=canon_hash(x['identities']);p=d/'decl.json';p.write_text(json.dumps(x));self.assertRaises(RuntimeError,correction_verify_declaration,None,p)
if __name__=='__main__':unittest.main()
