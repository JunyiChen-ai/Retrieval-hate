import json,sys
from pathlib import Path
import tempfile,unittest
sys.path.insert(0,str(Path(__file__).resolve().parent))
import steward_val_join as vj
from evidence_producer import sha

def dump(p,x): p.write_text(json.dumps(x,sort_keys=True)+"\n")
def fixture(tmp_path):
 e=tmp_path/"e";(e/"records").mkdir(parents=True)
 iw={"window_index":0,"start":0.,"end":1.,"center":.5,"speech":"x","speech_sha256":"a"*64};item={"video_id":"v1","media_path":"/opaque","media_sha256":"b"*64,"duration":1.,"asr_record_sha256":"c"*64,"windows":[iw]};(e/"frozen_inputs.jsonl").write_text(json.dumps(item,sort_keys=True)+"\n")
 cfg={"split":"val","labels_read":False,"producer_sha256":vj.EVIDENCE_PRODUCER,"local_forward_sha256":vj.EVIDENCE_PRODUCER,"model_revision":"rev","prompt_spec_sha256":"d"*64,"frozen_inputs_sha256":sha(e/"frozen_inputs.jsonl"),"n_videos":1,"n_windows":1};dump(e/"preregistered_config.json",cfg);ch=sha(e/"preregistered_config.json")
 w={"window_index":0,"start":0.,"end":1.,"center":.5,"frame_time":.5,"frame_index":0,"fps":1.,"frame_fallback_offset":0.,"speech_sha256":"a"*64,"prompt_sha256":{"text":"e"*64,"multimodal":"f"*64},"text_isolated_score":.1,"multimodal_isolated_score":.3};r={"video_id":"v1","duration":1.,"media_sha256":"b"*64,"asr_record_sha256":"c"*64,"input_item_sha256":__import__('hashlib').sha256(json.dumps(item,sort_keys=True,separators=(',',':')).encode()).hexdigest(),"config_sha256":ch,"producer_source_sha256":vj.EVIDENCE_PRODUCER,"model_revision":"rev","prompt_spec_sha256":"d"*64,"windows":[w],"global_causal_score":.2,"labels_read":False};dump(e/"records/v1.json",r)
 dump(e/"evidence_manifest.json",{"status":"COMPLETE_LABEL_BLIND","config_sha256":ch,"frozen_inputs_sha256":sha(e/"frozen_inputs.jsonl"),"n_videos":1,"n_windows":1,"records":{"v1":sha(e/"records/v1.json")}})
 lp=tmp_path/"labels.json";dump(lp,{"schema_version":"v24_video_labels_v1","corpus":"thvl","split":"val","label_semantics":"any_target_video_level","records":[{"video_id":"v1","any_target_label":1}]})
 return e,lp

class TestJoinV2(unittest.TestCase):
 def test_join_and_old_schema_rejected(self):
  with tempfile.TemporaryDirectory() as z:
   d=Path(z);e,l=fixture(d);m=vj.join(e,l,d/"out");self.assertEqual(m["evidence_producer_sha256"],vj.EVIDENCE_PRODUCER);self.assertEqual(vj.load_val_manifest(d/"out/val_id_manifest.json")["split"],"val")
   old=d/"old.json";dump(old,{"corpus":"thvl","split":"val","ids":["v1"],"v23_global_source_sha256":"0"*64,"producer_sha256":"0"*64});self.assertRaises(RuntimeError,vj.load_val_manifest,old)
 def test_fail_closed_mutations(self):
  for mutation in ("wrong_split","temporal","swapped_producer","hash_tamper"):
   with self.subTest(mutation=mutation),tempfile.TemporaryDirectory() as z:
    d=Path(z);e,l=fixture(d)
    if mutation in ("wrong_split","temporal"):
     x=json.load(open(l));x["split"]="train" if mutation=="wrong_split" else x["split"]
     if mutation=="temporal":x["records"][0]["timestamps"]=[0,1]
     dump(l,x)
    elif mutation=="swapped_producer":
     p=e/"preregistered_config.json";x=json.load(open(p));x["producer_sha256"]="0"*64;dump(p,x);em=json.load(open(e/"evidence_manifest.json"));em["config_sha256"]=sha(p);dump(e/"evidence_manifest.json",em)
    else:
     p=e/"preregistered_config.json";x=json.load(open(p));x["extra"]="tamper";dump(p,x)
    self.assertRaises(RuntimeError,vj.join,e,l,d/"out")
 def test_manifest_hash_tamper(self):
  with tempfile.TemporaryDirectory() as z:
   d=Path(z);e,l=fixture(d);vj.join(e,l,d/"out");p=d/"out/val_id_manifest.json";x=json.load(open(p));x["bags_sha256"]="x"*64;dump(p,x);self.assertRaises(RuntimeError,vj.load_val_manifest,p)
 def test_score_tamper_and_same_count_id_swap(self):
  for mode in ('score','id_swap'):
   with self.subTest(mode=mode),tempfile.TemporaryDirectory() as z:
    d=Path(z);e,l=fixture(d)
    if mode=='score':
     p=e/'records/v1.json';x=json.load(open(p));x['windows'][0]['text_isolated_score']=.9;dump(p,x) # manifest deliberately unchanged
    else:
     p=e/'records/v1.json';q=e/'records/v2.json';p.rename(q);em=json.load(open(e/'evidence_manifest.json'));em['records']={'v2':next(iter(em['records'].values()))};dump(e/'evidence_manifest.json',em)
    self.assertRaises(RuntimeError,vj.join,e,l,d/'out')
if __name__=="__main__":unittest.main()
