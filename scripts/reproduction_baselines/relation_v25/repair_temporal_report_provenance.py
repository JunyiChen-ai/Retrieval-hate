#!/usr/bin/env python3
"""Byte-preserving provenance-field insertion; does not parse metric values."""
import hashlib,os,tempfile
from pathlib import Path
from post_access_correction import verify,sha
ROOT=Path('/home/jehc223/Hate-follow-up')
SRC=ROOT/'results/steward_private/thvl_bench/v25_temporal_steward_report.INVALID_STALE_ADDENDUM.json'
OUT=ROOT/'results/steward_private/thvl_bench/v25_temporal_steward_report.POST_ACCESS_CORRECTED_V2.json'
def main():
 c=verify();b=SRC.read_bytes()
 if hashlib.sha256(b).hexdigest()!=c['unchanged_evaluation_bindings']['temporal_report_sha256'] or not b.startswith(b'{\n'):raise RuntimeError('stale report identity/format')
 field=('  "post_access_provenance_correction_sha256": "'+sha(Path(__file__).with_name('POST_ACCESS_PROVENANCE_CORRECTION_V2.json'))+'",\n').encode();new=b'{\n'+field+b[2:]
 if OUT.exists():
  if OUT.read_bytes()!=new:raise RuntimeError('corrected report exists with different bytes')
  return
 fd,tmp=tempfile.mkstemp(prefix=OUT.name+'.',dir=OUT.parent)
 with os.fdopen(fd,'wb') as f:f.write(new);f.flush();os.fsync(f.fileno())
 os.replace(tmp,OUT)
if __name__=='__main__':main()
