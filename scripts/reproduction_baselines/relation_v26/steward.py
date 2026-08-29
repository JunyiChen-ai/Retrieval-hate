#!/usr/bin/env python3
import argparse,json,os
from pathlib import Path
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey
from artifacts import sha
from core import DESIGN_SHA
KEY_MANIFEST=Path('/home/jehc223/Hate-follow-up/results/steward_private/thvl_bench/v26_steward_key_frozen/public_manifest.json');KEY_MANIFEST_SHA='38a510ab4dca9bdfacb3551385d01ad8053eb157176f6ce20c7b8ba3730d0f35';PUBLIC_KEY_SHA='61c7d93dd0da16e8ffb44843201e9a926ecd0699f5e5ba0e0aa76e9cab18e960'
V25_MANIFEST=Path('/home/jehc223/Hate-follow-up/results/steward_private/thvl_bench/v25_val_predictions_epoch2_frozen/manifest.json');V25_MANIFEST_SHA='b428d200ecfe7d073a96b95d711874d49b37a5569917672d447669779764b5c9'
def canon(x):return json.dumps(x,sort_keys=True,separators=(',',':')).encode()
def bound_file(x):
 if type(x)is not dict or set(x)!={'path','sha256'} or not Path(x['path']).is_absolute() or sha(x['path'])!=x['sha256']:raise RuntimeError('steward binding tamper')
def pinned_public_key():
 if sha(KEY_MANIFEST)!=KEY_MANIFEST_SHA:raise RuntimeError('steward key manifest identity')
 m=json.load(open(KEY_MANIFEST));req={'schema','algorithm','status','private_key_path','private_key_mode','public_key_path','public_key_sha256','producer_path','producer_sha256'}
 if set(m)!=req or m['schema']!='v26_steward_ed25519_identity_v1' or m['algorithm']!='Ed25519' or m['status']!='FROZEN' or m['private_key_mode']!='0600' or m['public_key_sha256']!=PUBLIC_KEY_SHA or sha(m['public_key_path'])!=PUBLIC_KEY_SHA or sha(m['producer_path'])!=m['producer_sha256'] or (os.stat(m['private_key_path']).st_mode&0o777)!=0o600:raise RuntimeError('steward key identity')
 return Ed25519PublicKey.from_public_bytes(Path(m['public_key_path']).read_bytes())
def bootstrap_file(x,cohort,seed,ids):
 bound_file(x);z=json.load(open(x['path']))
 if set(z)!={'schema','cohort','seed','B','ids','arrays'} or z['schema']!='v26_bootstrap_indices_v1' or z['cohort']!=cohort or z['seed']!=seed or z['B']!=2000 or z['ids']!=ids or len(z['arrays'])!=2000 or any(len(a)!=len(ids) or any(type(i)is not int or i<0 or i>=len(ids) for i in a) for a in z['arrays']):raise RuntimeError('bootstrap canonical arrays')
def verify_signed_report(path,prediction_manifest,selection=None):
 x=json.load(open(path));keys={'schema','design_sha256','selected_epoch','calculator_sha256','inputs','all_ids','positive_ids','mixed_ids','stats','gates','all_gates_pass','test_opened','signature_hex'}
 if set(x)!=keys or x['schema']!='v26_signed_temporal_report_v2' or x['design_sha256']!=DESIGN_SHA or x['calculator_sha256']!=sha(Path(__file__).with_name('steward_private_calc.py')) or x['test_opened'] is not False or len(x['all_ids'])!=32 or len(set(x['all_ids']))!=32 or not set(x['positive_ids'])<=set(x['all_ids']) or not set(x['mixed_ids'])<=set(x['all_ids']):raise RuntimeError('signed report schema')
 sig=bytes.fromhex(x['signature_hex']);unsigned=dict(x);del unsigned['signature_hex'];pinned_public_key().verify(sig,canon(unsigned));inp=x['inputs']
 req={'encrypted_cipher','encrypted_manifest','ledger','public_val_manifest','qc','taxonomy','private_source','raw_id_map','gt_reducer','calculator','selection','predictions','v25_manifest','v25_provenance','v25_reference','v25_checkpoints','v25_verifier','v25_reducer','bootstraps'}
 if set(inp)!=req or type(inp['predictions'])is not list or not inp['predictions'] or len(inp['bootstraps'])!=3:raise RuntimeError('signed inputs')
 for k in req-{'predictions','bootstraps','v25_checkpoints'}:bound_file(inp[k])
 if type(inp['v25_checkpoints'])is not list or len(inp['v25_checkpoints'])!=3:raise RuntimeError('V25 checkpoints')
 [bound_file(z) for z in inp['v25_checkpoints']]
 if Path(inp['v25_manifest']['path']).resolve()!=V25_MANIFEST.resolve() or inp['v25_manifest']['sha256']!=V25_MANIFEST_SHA:raise RuntimeError('V25 authority substitution')
 [bound_file(z) for z in inp['bootstraps']]
 for z in inp['predictions']:
  if set(z)!={'path','sha256','seed','prediction_file','features','reference','train_run'}:raise RuntimeError('prediction binding')
  bound_file({'path':z['path'],'sha256':z['sha256']});[bound_file(z[k]) for k in ('prediction_file','features','reference','train_run')]
 if not any(Path(z['path']).resolve()==Path(prediction_manifest).resolve() and z['sha256']==sha(prediction_manifest) for z in inp['predictions']):raise RuntimeError('selected prediction absent')
 if selection and (Path(inp['selection']['path']).resolve()!=Path(selection).resolve() or inp['selection']['sha256']!=sha(selection)):raise RuntimeError('selection substitution')
 if x['all_gates_pass']!=all(x['gates'].values()):raise RuntimeError('gate aggregate')
 return x
def main():
 p=argparse.ArgumentParser();p.add_argument('--report',required=True);p.add_argument('--predictions',required=True);p.add_argument('--selection');a=p.parse_args();verify_signed_report(a.report,a.predictions,a.selection)
if __name__=='__main__':main()
