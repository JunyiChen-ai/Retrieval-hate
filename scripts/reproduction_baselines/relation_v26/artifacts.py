import hashlib,json,os,tempfile
from pathlib import Path
from core import DESIGN_SHA,ch
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def atomic(p,x):
 p=Path(p);fd,t=tempfile.mkstemp(dir=p.parent,prefix=p.name+'.')
 with os.fdopen(fd,'w') as f:json.dump(x,f,sort_keys=True);f.write('\n');f.flush();os.fsync(f.fileno())
 os.replace(t,p)
def verify_manifest(p,schema):
 x=json.load(open(p));
 if x.get('schema')!=schema or x.get('design_sha256')!=DESIGN_SHA:raise RuntimeError('manifest design/schema')
 for _,v in x.get('files',{}).items():
  if set(v)!={'path','sha256'} or sha(v['path'])!=v['sha256']:raise RuntimeError('artifact tamper')
 return x
def seal(out,inputs,status):atomic(out,{'schema':'v26_seal_v1','design_sha256':DESIGN_SHA,'status':status,'inputs':{str(Path(p).resolve()):sha(p) for p in inputs},'test_read':False})
