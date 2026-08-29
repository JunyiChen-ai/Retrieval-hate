#!/usr/bin/env python3
"""Verify steward HMAC for within/shuffle gates and emit the only testable config."""
import argparse,hashlib,hmac,json
from pathlib import Path
from train import sha
KEYS={'status','frozen_config_sha256','within_macro_ap_gain','within_macro_roc_gain','shuffle_pass','steward_id','signature_hmac_sha256'}
def canonical(g):return json.dumps({k:g[k] for k in sorted(KEYS-{'signature_hmac_sha256'})},sort_keys=True,separators=(',',':')).encode()
def main():
 p=argparse.ArgumentParser();p.add_argument('--video-frozen',required=True);p.add_argument('--steward-gate',required=True);p.add_argument('--steward-key',required=True);p.add_argument('--out',required=True);a=p.parse_args();f=json.load(open(a.video_frozen));g=json.load(open(a.steward_gate))
 if f.get('status')!='VIDEO_VAL_PASS_PENDING_TEMPORAL' or not f.get('all_video_gates_pass'):raise RuntimeError('video validation did not pass')
 if set(g)!=KEYS or g['status']!='WITHIN_SHUFFLE_PASS' or g['frozen_config_sha256']!=sha(a.video_frozen):raise RuntimeError('invalid/unbound steward gate')
 expected=hmac.new(Path(a.steward_key).read_bytes(),canonical(g),hashlib.sha256).hexdigest()
 if not hmac.compare_digest(expected,g['signature_hmac_sha256']):raise RuntimeError('invalid steward signature')
 if not g['shuffle_pass'] or g['within_macro_ap_gain']<.01 or g['within_macro_roc_gain']<.02:raise RuntimeError('temporal kill gate failed')
 f.update({'status':'FINAL_PASS','temporal_steward_gate_signed':True,'video_frozen_sha256':sha(a.video_frozen),'steward_gate_sha256':sha(a.steward_gate),'steward_id':g['steward_id'],'temporal_diagnostics':{k:g[k] for k in ('within_macro_ap_gain','within_macro_roc_gain','shuffle_pass')}});Path(a.out).write_text(json.dumps(f,indent=2,sort_keys=True)+'\n')
if __name__=='__main__':main()
