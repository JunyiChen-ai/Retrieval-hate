#!/usr/bin/env python3
"""One-time creation of the V26 validation steward signing identity."""
import argparse,json,os,tempfile
from pathlib import Path
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from artifacts import atomic,sha

def write_bytes(path,data,mode):
 path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
 fd,tmp=tempfile.mkstemp(dir=path.parent,prefix=path.name+'.')
 try:
  os.fchmod(fd,mode);os.write(fd,data);os.fsync(fd);os.close(fd);os.replace(tmp,path)
 finally:
  if os.path.exists(tmp):os.unlink(tmp)

def main():
 p=argparse.ArgumentParser();p.add_argument('--out',required=True);a=p.parse_args();out=Path(a.out).resolve()
 if out.exists():raise RuntimeError('refuse to replace frozen steward identity')
 out.mkdir(parents=True);priv=Ed25519PrivateKey.generate();raw=priv.private_bytes(serialization.Encoding.Raw,serialization.PrivateFormat.Raw,serialization.NoEncryption());pub=priv.public_key().public_bytes(serialization.Encoding.Raw,serialization.PublicFormat.Raw)
 private=out/'ed25519_private.key';public=out/'ed25519_public.key';write_bytes(private,raw,0o600);write_bytes(public,pub,0o644)
 manifest={'schema':'v26_steward_ed25519_identity_v1','algorithm':'Ed25519','status':'FROZEN','private_key_path':str(private),'private_key_mode':'0600','public_key_path':str(public),'public_key_sha256':sha(public),'producer_path':str(Path(__file__).resolve()),'producer_sha256':sha(__file__)}
 atomic(out/'public_manifest.json',manifest);os.chmod(private,0o600)
 print(json.dumps({'manifest':str(out/'public_manifest.json'),'manifest_sha256':sha(out/'public_manifest.json'),'public_key_sha256':sha(public)},sort_keys=True))
if __name__=='__main__':main()
