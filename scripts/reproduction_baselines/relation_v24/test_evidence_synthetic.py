#!/usr/bin/env python3
import json,subprocess,sys,tempfile,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
from evidence_producer import windows,window_speech,load_atomic,preregister,assert_runtime_config,sha,V16_FORWARD,V16_PROMPT,MODEL_REVISION,MM_POLICY,V23_OFFSETS,ROOT
from train import load_bags,load_id_manifest
class TestEvidence(unittest.TestCase):
 def atomic(self,v,i=0):
  q={'opaque_id':v,'hf_path':'x','hf_revision':'r','source_manifest_sha256':'a'*64,'status':'ok','cache_path':f'/media/{v}','bytes':1,'media_sha256':str(i)*64,'duration_seconds':31,'full_decode_ok':True,'audio_available':True,'video_available':True,'wav_path':'x.wav','wav_sha256':'b'*64}
  a={'opaque_id':v,'wav_sha256':'b'*64,'model':'whisper','text':'speech','chunks':[{'start':1,'end':2,'text':'speech'}],'n_chunks':1,'language':None,'labels_or_temporal_gt_opened':False};return q,a
 def fake_v16(self,d,ids):
  d.mkdir();cfg={'model_revision':MODEL_REVISION,'forward_implementation_sha256':sha(V16_FORWARD),'prompt_implementation_sha256':sha(V16_PROMPT),'arms':['causal_continuous']};(d/'preregistered_config.json').write_text(json.dumps(cfg));raw=d/'per_chunk_raw.jsonl';raw.write_text(''.join(json.dumps({'video_id':v,'scores':{'causal_continuous':float(i)}})+'\n' for i,v in enumerate(ids)));man={'raw_sha256':sha(raw),'config_sha256':sha(d/'preregistered_config.json'),'model_revision':MODEL_REVISION};(d/'raw_manifest.json').write_text(json.dumps(man))
 def test_windows_full_nonoverlap_last(self):
  w=windows(61.2);self.assertEqual([(x['start'],x['end']) for x in w],[(0,30),(30,60),(60,61.2)])
 def test_missing_speech_and_cap(self):
  self.assertEqual(window_speech([],0,30,False),'[NO SPEECH]');self.assertEqual(len(window_speech([{'start':1,'end':2,'text':'x'*4000}],0,30,False)),3000)
 def test_v23_mm_semantic_equality_binding(self):
  ref=json.load(open(ROOT/'results/steward_private/thvl_bench/v23_val_multimodal_raw/preregistered_config.json'));self.assertEqual(MM_POLICY,ref['policy']);self.assertEqual(V23_OFFSETS,ref['decode']['center_fallback_offsets_seconds'])
 def test_atomic_incomplete_and_label_rejected(self):
  with tempfile.TemporaryDirectory() as z:
   q=Path(z)/'q';a=Path(z)/'a';q.mkdir();a.mkdir();(q/'v.json').write_text(json.dumps({'opaque_id':'v','status':'ok'}))
   with self.assertRaises(RuntimeError):load_atomic(q,a)
   _,ar=self.atomic('v');ar['weak_video_label']=1;(a/'v.json').write_text(json.dumps(ar))
   with self.assertRaises(RuntimeError):load_atomic(q,a)
 def test_dry_prepare_and_steward_exact_bags(self):
  with tempfile.TemporaryDirectory() as z:
   d=Path(z);q=d/'q';a=d/'a';q.mkdir();a.mkdir()
   for i in range(2):
    v=f'v{i}';qr,ar=self.atomic(v,i);(q/f'{v}.json').write_text(json.dumps(qr));(a/f'{v}.json').write_text(json.dumps(ar))
   v16=d/'v16';self.fake_v16(v16,['v0','v1']);ev=d/'ev';cfg=preregister(q,a,v16,ev);self.assertEqual((cfg['n_videos'],cfg['n_windows']),(2,4));self.assertEqual([json.loads(x)['global_causal_score'] for x in open(ev/'v16_global_means.jsonl')],[0.,1.]);self.assertIn('exact V16 packed causal_continuous',cfg['global_definition']);self.assertEqual(assert_runtime_config(cfg).name,MODEL_REVISION);bad=dict(cfg);bad['prompt_spec_sha256']='0'*64
   with self.assertRaises(RuntimeError):assert_runtime_config(bad)
   rd=ev/'records';rd.mkdir()
   hashes={}
   for i in range(2):
    v=f'v{i}';r={'video_id':v,'global_causal_score':float(i),'windows':[{'text_isolated_score':i+.1,'multimodal_isolated_score':i+.2},{'text_isolated_score':i+.3,'multimodal_isolated_score':i+.4}]};p=rd/f'{v}.json';p.write_text(json.dumps(r));hashes[v]=sha(p)
   (ev/'evidence_manifest.json').write_text(json.dumps({'records':hashes}));weak=d/'weak.json';weak.write_text(json.dumps({'records':[{'opaque_id':'v0','weak_video_label':0},{'opaque_id':'v1','weak_video_label':1}]}));out=d/'joined';subprocess.run([sys.executable,str(Path(__file__).resolve().parent/'steward_join.py'),'--evidence-dir',str(ev),'--weak-manifest',str(weak),'--out-dir',str(out)],check=True)
   m=load_id_manifest(out/'train_id_manifest.json','thvl','train');bags=load_bags(out/'bags.jsonl',m['ids'],'thvl','train',m['v23_global_source_sha256']);self.assertEqual(set(bags),{'v0','v1'})
if __name__=='__main__':unittest.main()
