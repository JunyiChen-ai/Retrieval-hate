#!/usr/bin/env python3
import argparse,hashlib,json,math,os,tempfile
from pathlib import Path
import numpy as np
from extract_thvl_1hz import sha,load_inputs,composite,canonical_hash,identity,video_probe,frame_index,DESIGN_SHA,MODELS,EXPECTED,BERT_FILES,CREV,BREV,EXPECTED_VGG_TREE_SHA,RUNTIME
MANIFEST_KEYS={'schema','design_sha256','split','split_manifest_sha256','g_source_sha256','models','revisions','expected_hashes','runtime','vggish_frontend','n','root_sha256','elapsed','labels_or_gt_read'}
RECORD_KEYS={'schema','id','T','duration','file','sha256','composite_input_sha256','shapes','finite','video_probe','frame_backends','frame_indices','fps','effective_timestamps','source','labels_or_gt_read'}
def ids_hash(ids):return hashlib.sha256(json.dumps(sorted(ids),separators=(',',':')).encode()).hexdigest()
def atomic_json(p,x):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);fd,t=tempfile.mkstemp(dir=p.parent,prefix=p.name+'.')
 try:
  with os.fdopen(fd,'w') as f:json.dump(x,f,sort_keys=True);f.write('\n');f.flush();os.fsync(f.fileno())
  os.replace(t,p)
 finally:
  if os.path.exists(t):os.unlink(t)
def verify(raw,split,split_manifest,qc,asr,gfile,allow_partial=False):
 raw=Path(raw);m=json.load(open(raw/'manifest.json'));sm=json.load(open(split_manifest));q,a=load_inputs(qc,asr);gg={};gh={}
 for b in open(gfile,'rb'):
  r=json.loads(b)
  if set(r)!={'video_id','global_causal_score'} or r['video_id'] in gg:raise RuntimeError('G')
  gg[r['video_id']]=float(r['global_causal_score']);gh[r['video_id']]=hashlib.sha256(b).hexdigest()
 ids=sorted(sm['ids']);seen=sorted(p.stem for p in raw.glob('*.json') if p.name!='manifest.json')
 if set(q)!=set(ids) or set(gg)!=set(ids) or (not allow_partial and seen!=ids):raise RuntimeError('raw coverage')
 front={'package':'torchvggish','source_tree_sha256':EXPECTED_VGG_TREE_SHA,'weights_sha256':EXPECTED['vggish_weights'],'sample_rate':16000,'mel_bins':64,'mel_min':125,'mel_max':7500,'stft_window_ms':25,'stft_hop_ms':10,'example_window_s':.96,'example_hop_s':1.0,'postprocess':False}
 if set(m)!=MANIFEST_KEYS or m['schema']!='v26_thvl_features_v1' or m['design_sha256']!=DESIGN_SHA or m['split']!=split or m['split_manifest_sha256']!=sha(split_manifest) or m['g_source_sha256']!=sha(gfile) or m['models']!=MODELS or m['revisions']!={'clip':CREV,'bert':BREV} or m['expected_hashes']!=EXPECTED|BERT_FILES or m['runtime']!=RUNTIME or m['vggish_frontend']!=front or m['labels_or_gt_read'] is not False or m['n']!=len(seen):raise RuntimeError('raw manifest')
 rows=[]
 for v in seen:
  r=json.load(open(raw/(v+'.json')));p=raw/(v+'.npz')
  src={'qc_sha256':canonical_hash(q[v]),'media_sha256':sha(q[v]['cache_path']),'wav_sha256':sha(q[v]['wav_path']),'asr_sha256':canonical_hash(a[v]),'identity_sha256':canonical_hash(identity()),'producer_sha256':sha(Path(__file__).with_name('extract_thvl_1hz.py'))}
  dec={'probe':r.get('video_probe'),'backends':r.get('frame_backends'),'indices':r.get('frame_indices'),'effective_times':r.get('effective_timestamps')}
  if set(r)!=RECORD_KEYS or r['schema']!='v26_thvl_video_features_v1' or r['id']!=v or Path(r['file']).resolve()!=p.resolve() or r['composite_input_sha256']!=composite(q[v],a[v],gg[v],dec) or sha(p)!=r['sha256'] or r['labels_or_gt_read'] is not False or r['source']!=src:raise RuntimeError('raw record')
  with np.load(p,allow_pickle=False) as x:
   if set(x.files)!={'visual','audio','text','availability','duration','G'}:raise RuntimeError('raw npz keys')
   T=math.ceil(float(x['duration']));av=x['availability']
   expected_av=np.stack([np.full(T,int(q[v]['video_available']),'uint8'),np.full(T,int(q[v]['audio_available']),'uint8'),av[:,2]],1)
   if r['T']!=T or T!=math.ceil(float(q[v]['duration_seconds'])) or abs(float(r['duration'])-float(x['duration']))>1e-9 or abs(float(x['duration'])-float(q[v]['duration_seconds']))>1e-9 or x['visual'].shape!=(T,512) or x['audio'].shape!=(T,128) or x['text'].shape!=(T,768) or av.shape!=(T,3) or av.dtype!=np.uint8 or not np.array_equal(av,expected_av) or not np.isin(av,[0,1]).all() or not all(np.isfinite(x[k]).all() for k in ('visual','audio','text','duration','G')) or not np.all(av.any(1)) or float(x['G'])!=gg[v]:raise RuntimeError('raw arrays')
   if any(np.any(x[k][~av[:,j].astype(bool)]!=0) for j,k in enumerate(('visual','audio','text'))):raise RuntimeError('missing nonzero')
  if q[v]['video_available']:
   pr=video_probe(q[v]['cache_path']);ix=[frame_index(j,float(q[v]['duration_seconds']),pr['fps'],pr['nframes']) for j in range(T)];tm=[i/pr['fps'] for i in ix]
   if r['video_probe']!=pr or r['fps']!=pr['fps'] or r['frame_indices']!=ix or r['effective_timestamps']!=tm or len(r['frame_backends'])!=T or any(b not in ('decord','ffmpeg') for b in r['frame_backends']):raise RuntimeError('decode identity')
  elif r['frame_backends']!=['unavailable']*T:raise RuntimeError('unavailable backend')
  if r['shapes']!=[[T,512],[T,128],[T,768],[T,3]] or r['finite'] is not True or len(r['frame_indices'])!=T or len(r['effective_timestamps'])!=T:raise RuntimeError('timeline')
  rows.append(r)
 root=hashlib.sha256((sha(split_manifest)+sha(gfile)+json.dumps(m['vggish_frontend'],sort_keys=True)+''.join(r['id']+'\t'+r['sha256']+'\t'+r['composite_input_sha256']+'\t'+gh[r['id']]+'\n' for r in sorted(rows,key=lambda x:x['id']))).encode()).hexdigest()
 if root!=m['root_sha256']:raise RuntimeError('raw root')
 return m,rows
def seal(path,raw,split,split_manifest,qc,asr,gfile):
 m,rows=verify(raw,split,split_manifest,qc,asr,gfile);ids=[r['id'] for r in rows];x={'schema':'v26_raw_seal_v2','split':split,'raw_path':str(Path(raw).resolve()),'raw_manifest_sha256':sha(Path(raw)/'manifest.json'),'raw_root_sha256':m['root_sha256'],'split_manifest_path':str(Path(split_manifest).resolve()),'split_manifest_sha256':sha(split_manifest),'g_source_sha256':sha(gfile),'ids_sha256':ids_hash(ids),'n':len(ids),'producer_sha256':sha(Path(__file__).with_name('extract_thvl_1hz.py')),'verifier_sha256':sha(__file__)};atomic_json(path,x);return x
def main():
 p=argparse.ArgumentParser();p.add_argument('--raw',required=True);p.add_argument('--split',choices=('train','val'),required=True);p.add_argument('--split-manifest',required=True);p.add_argument('--qc-dir',required=True);p.add_argument('--asr-dir',required=True);p.add_argument('--g-jsonl',required=True);p.add_argument('--seal',required=True);a=p.parse_args();seal(a.seal,a.raw,a.split,a.split_manifest,a.qc_dir,a.asr_dir,a.g_jsonl)
if __name__=='__main__':main()
