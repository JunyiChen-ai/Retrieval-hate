#!/usr/bin/env python3
"""Explicit test seal; never reads test inputs or labels."""
import argparse,json,hmac,hashlib
from pathlib import Path
from core import sha
def main():
 p=argparse.ArgumentParser();p.add_argument('--selected',required=True);p.add_argument('--approval',required=True);p.add_argument('--key',required=True);p.add_argument('--out',required=True);a=p.parse_args();s=json.load(open(a.selected));q=json.load(open(a.approval));allowed={'status','selected_sha256','test_input_chain','signature_hmac_sha256'}
 if set(q)!=allowed or q['status']!='TEST_SEAL_APPROVED' or q['selected_sha256']!=sha(a.selected):raise RuntimeError('invalid seal')
 chain=q['test_input_chain'];ck={'test_records','evidence_manifest','evidence_config'}
 if set(chain)!=ck or any(set(chain[k])!={'path','sha256'} or str(Path(chain[k]['path']).resolve())!=chain[k]['path'] or sha(chain[k]['path'])!=chain[k]['sha256'] for k in ck):raise RuntimeError('test input chain')
 msg=json.dumps({k:q[k] for k in ('selected_sha256','status','test_input_chain')},sort_keys=True,separators=(',',':')).encode();sig=hmac.new(Path(a.key).read_bytes(),msg,hashlib.sha256).hexdigest()
 if not hmac.compare_digest(sig,q['signature_hmac_sha256']) or s['status']!='VIDEO_VAL_PASS_PENDING_TEST_SEAL':raise RuntimeError('seal denied')
 s['status']='FINAL_PASS';s['test_seal_signed']=True;s['approval_sha256']=sha(a.approval);s['approved_test_input_chain']=chain;Path(a.out).write_text(json.dumps(s,indent=2,sort_keys=True)+'\n')
if __name__=='__main__':main()
