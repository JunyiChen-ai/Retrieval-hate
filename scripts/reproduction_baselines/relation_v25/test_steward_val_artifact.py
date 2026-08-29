#!/usr/bin/env python3
import csv,hashlib,json,tempfile,unittest,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).parent));from steward_val_artifact import *
def put(p,x):p.write_text(json.dumps(x,sort_keys=True)+'\n')
class T(unittest.TestCase):
 def fixture(self,d):
  val=[hashlib.sha256(f'r{i}'.encode()).hexdigest() for i in range(32)];vp=d/'v';qp=d/'q';tp=d/'t';mp=d/'map';src=d/'x.csv';key=d/'key';put(vp,{'schema':'thvl_public_val_ids_v1','ids':val});put(qp,{'schema':'thvl_qc_durations_v1','durations':{x:2.2 for x in val}});put(tp,{'schema':'thvl_taxonomy_indices_v1','category_count':11,'target_indices':[1,2,3],'other_harm_indices':[0,4,5,6,7,8,9,10]});key.write_bytes(bytes(32));rows=[];recs=[]
  for i in range(450):
   raw=f'r{i}';oid=hashlib.sha256(raw.encode()).hexdigest();split='validation' if i<32 else ('train' if i<346 else 'test');recs.append({'raw_id':raw,'hashed_id':oid,'canonical_id':'youtube:'+raw,'repository_paths':[],'source_group':'youtube:'+raw,'split':split});lab=str([[0,1,0,0,0,0,0,0,0,0,0]] if i==0 else []);tim=str([['00:00','00:01.1']] if i==0 else []);rows.append([raw,lab,tim,'[]'])
  remote={'annotation_sha256':'pending'};put(mp,{'remote_identity':remote,'records':recs})
  with src.open('w',newline='') as f:w=csv.writer(f,lineterminator='\n');w.writerow(HEADER);w.writerows(rows)
  m=json.load(open(mp));m['remote_identity']['annotation_sha256']=sha(src);put(mp,m);return vp,qp,tp,src,mp,key,rows
 def test_roundtrip_overlap(self):
  with tempfile.TemporaryDirectory() as z:
   d=Path(z);a=self.fixture(d);generate(*a[:6],d/'o');x=decrypt_and_verify(d/'o',a[5],*a[:5]);r=x['records'][hashlib.sha256(b'r0').hexdigest()];self.assertEqual(r['target_1hz'],[1,1,0])
 def test_nonval_all_three_malicious_never_extracted(self):
  with tempfile.TemporaryDirectory() as z:
   d=Path(z);vp,qp,tp,src,mp,key,rows=self.fixture(d);rows[40]=['r40','"unterminated-python','{not timestamp','malicious\nquoted,"value']
   with src.open('w',newline='') as f:w=csv.writer(f,lineterminator='\n');w.writerow(HEADER);w.writerows(rows)
   m=json.load(open(mp));m['remote_identity']['annotation_sha256']=sha(src);put(mp,m);generate(vp,qp,tp,src,mp,key,d/'o')
 def test_val_invalid_and_length_mismatch(self):
  for kind in ('invalid','mismatch'):
   with self.subTest(kind=kind),tempfile.TemporaryDirectory() as z:
    d=Path(z);vp,qp,tp,src,mp,key,rows=self.fixture(d);rows[0][1]='bad' if kind=='invalid' else str([[0]*11,[0]*11])
    with src.open('w',newline='') as f:w=csv.writer(f,lineterminator='\n');w.writerow(HEADER);w.writerows(rows)
    m=json.load(open(mp));m['remote_identity']['annotation_sha256']=sha(src);put(mp,m);self.assertRaises(Exception,generate,vp,qp,tp,src,mp,key,d/'o')
 def test_header_map_swap_tamper_duplicate(self):
  for kind in ('header','swap','tamper','duplicate'):
   with self.subTest(kind=kind),tempfile.TemporaryDirectory() as z:
    d=Path(z);vp,qp,tp,src,mp,key,rows=self.fixture(d);m=json.load(open(mp))
    if kind=='header':
     b=src.read_bytes();src.write_bytes(b.replace(HB,b'videoID,segment-level timestamp,segment-level annotation,contributing modalities',1));m['remote_identity']['annotation_sha256']=sha(src)
    elif kind=='swap':m['records'][0]['hashed_id'],m['records'][32]['hashed_id']=m['records'][32]['hashed_id'],m['records'][0]['hashed_id']
    elif kind=='tamper':m['remote_identity']['annotation_sha256']='0'*64
    else:m['records'][1]['raw_id']=m['records'][0]['raw_id']
    put(mp,m);self.assertRaises(Exception,generate,vp,qp,tp,src,mp,key,d/'o')
if __name__=='__main__':unittest.main()
