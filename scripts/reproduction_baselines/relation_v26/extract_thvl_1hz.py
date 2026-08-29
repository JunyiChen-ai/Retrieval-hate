#!/usr/bin/env python3
"""Complete label-blind atomic THVL 1 Hz CTW feature producer."""
import argparse,hashlib,json,math,os,subprocess,tempfile,time,wave
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
from pathlib import Path
import numpy as np,torch
from feature_manifest import MODELS
from core import DESIGN_SHA
CLIP='openai/clip-vit-base-patch16';CREV='57c216476eefef5ab752ec549e440a49ae4ae5f3';BERT='bert-base-uncased';BREV='86b5e0934494bd15c9632b12f734a8a67f723594'
EXPECTED={'clip_weights':'846bf12e84b91c2a65b44833a915241443b39356f1000ac1345c61d7bf8a209e','clip_config':'eaf1c9089a8553c913d27ea66407f8bfc2be9989c80c9f331ddb3d63d4c5e8ad','clip_preprocessor':'910e70b3956ac9879ebc90b22fb3bc8a75b6a0677814500101a4c072bd7857bd','vggish_weights':'10086976245803799d9194e9a73d9b6c1549c71d1b80106f5cade5608a561f4b'}
EXPECTED_VGG_TREE_SHA='fe88878a1de0059088fdd42497548cdc1df163e9dec84364c70149fd75ae1653'
RUNTIME={'torch':'2.7.1+cu128','transformers':'4.57.6','numpy':'1.26.4','decord':'0.6.0'}
BERT_FILES={'config.json':'7160e1553ad2ca51d8c1cb066be533db31826e12d173824c1bb0cb1a4f187d20','model.safetensors':'68d45e234eb4a928074dfd868cead0219ab85354cc53d20e772753c6bb9169d3','tokenizer.json':'ce64fce797c24f68df90b40a3f74f579b336a493db14bd583fd520ea0d8c9a98','tokenizer_config.json':'a025160ef0431f1a392f6f050c1310f4c5d9fb6f275932dbccba73c4d214bf10','vocab.txt':'07eced375cec144d27c900241f3e339478dec958f92fddbc551f295c992038a3'}
AUTH={'train':('/home/jehc223/Hate-follow-up/results/steward_private/thvl_bench/train314/v24_train_bags_frozen/train_id_manifest.json','3d4820624093d48d45db76e344e4fa812c6f641c10f8e05403bdd68cf4861b51',314),'val':('/home/jehc223/Hate-follow-up/results/steward_private/thvl_bench/v24_val_bags_frozen/val_id_manifest.json','c5d74ac98df84753e19df0312ce4507b4b6a3e1e18b5e85b593dcb0360dbab3f',32)}
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def atomic_text(p,s):
 p=Path(p);fd,t=tempfile.mkstemp(dir=p.parent,prefix=p.name+'.');
 try:
  with os.fdopen(fd,'w') as f:f.write(s);f.flush();os.fsync(f.fileno())
  os.replace(t,p)
 finally:
  if os.path.exists(t):os.unlink(t)
def canonical_hash(x):return hashlib.sha256(json.dumps(x,sort_keys=True,separators=(',',':')).encode()).hexdigest()
def identity():return {'models':MODELS,'revisions':{'clip':CREV,'bert':BREV},'expected':EXPECTED|BERT_FILES,'vggish_tree':EXPECTED_VGG_TREE_SHA,'runtime':RUNTIME,'producer_sha256':sha(__file__)}
def composite(q,a,g,decode=None):return hashlib.sha256((canonical_hash(q)+sha(q['cache_path'])+sha(q['wav_path'])+canonical_hash(a)+repr(float(g))+canonical_hash(identity())+canonical_hash(decode or {})).encode()).hexdigest()
def frame_index(second,duration,fps,nframes):return min(nframes-1,max(0,round(min(second+.5,duration-1e-6)*fps)))
def video_probe(path):
 p=subprocess.run(['ffprobe','-v','error','-count_frames','-select_streams','v:0','-show_entries','stream=avg_frame_rate,codec_name,width,height,nb_frames,nb_read_frames','-of','json',str(path)],capture_output=True,text=True,check=True);s=(json.loads(p.stdout).get('streams') or [None])[0]
 if not s:raise RuntimeError('ffprobe no video')
 a,b=s['avg_frame_rate'].split('/');fps=float(a)/float(b);raw_n=s.get('nb_frames') or s.get('nb_read_frames');n=int(raw_n) if str(raw_n).isdigit() else 0
 if not math.isfinite(fps) or fps<=0 or n<=0:raise RuntimeError('ffprobe fps/count')
 return {'fps':fps,'codec':s['codec_name'],'width':int(s['width']),'height':int(s['height']),'nframes':n,'ffprobe_sha256':hashlib.sha256(p.stdout.encode()).hexdigest()}
def ffmpeg_rgb(path,t,w,h):
 p=subprocess.run(['ffmpeg','-nostdin','-v','error','-ss',f'{t:.9f}','-i',str(path),'-frames:v','1','-f','rawvideo','-pix_fmt','rgb24','-'],capture_output=True)
 if p.returncode or len(p.stdout)!=w*h*3:raise RuntimeError('ffmpeg frame decode')
 return np.frombuffer(p.stdout,dtype=np.uint8).reshape(h,w,3).copy()
def verify_resume(r,q,a,g):
 fp=Path(r.get('file',''))
 dec={'probe':r.get('video_probe'),'backends':r.get('frame_backends'),'indices':r.get('frame_indices'),'effective_times':r.get('effective_timestamps')}
 if all(x is None for x in dec.values()):dec=None
 if r.get('schema')!='v26_thvl_video_features_v1' or r.get('source',{}).get('producer_sha256')!=sha(__file__) or r.get('composite_input_sha256')!=composite(q,a,g,dec) or not fp.is_file() or sha(fp)!=r.get('sha256') or r.get('labels_or_gt_read') is not False:raise RuntimeError('stale/tampered resume record')
 return r
def overlap(a,b,j,d):return max(0.,min(b,j+1,d)-max(a,j))
def load_inputs(qd,ad):
 q={p.stem:json.load(open(p)) for p in Path(qd).glob('*.json')};a={p.stem:json.load(open(p)) for p in Path(ad).glob('*.json')}
 if set(q)!=set(a) or not q:raise RuntimeError('QC/ASR coverage')
 qk={'audio_available','bytes','cache_path','duration_seconds','full_decode_ok','hf_path','hf_revision','media_sha256','opaque_id','source_manifest_sha256','status','video_available','wav_path','wav_sha256'};ak={'chunks','labels_or_temporal_gt_opened','language','model','n_chunks','opaque_id','text','wav_sha256'}
 for v in q:
  if set(q[v])!=qk or set(a[v])!=ak or any(set(c)!={'start','end','text'} for c in a[v]['chunks']) or q[v]['opaque_id']!=v or a[v]['opaque_id']!=v or q[v]['status']!='ok' or a[v]['labels_or_temporal_gt_opened'] is not False or q[v]['wav_sha256']!=a[v]['wav_sha256'] or sha(q[v]['cache_path'])!=q[v]['media_sha256'] or sha(q[v]['wav_path'])!=q[v]['wav_sha256']:raise RuntimeError('input provenance/allowlist/live hash')
 return q,a
def atomic_npz(p,**x):
 p=Path(p);fd,t=tempfile.mkstemp(dir=p.parent,prefix=p.name+'.',suffix='.npz');os.close(fd)
 try:
  np.savez(t,**x)
  with open(t,'rb') as f:os.fsync(f.fileno())
  os.replace(t,p)
 finally:
  if os.path.exists(t):os.unlink(t)
def models():
 from transformers import CLIPModel,CLIPImageProcessor,AutoModel,AutoTokenizer
 cs=Path.home()/'.cache/huggingface/hub/models--openai--clip-vit-base-patch16/snapshots'/CREV
 if sha(cs/'model.safetensors')!=EXPECTED['clip_weights'] or sha(cs/'config.json')!=EXPECTED['clip_config'] or sha(cs/'preprocessor_config.json')!=EXPECTED['clip_preprocessor']:raise RuntimeError('CLIP identity')
 vw=Path.home()/'.cache/torch/hub/checkpoints/vggish-10086976.pth'
 if sha(vw)!=EXPECTED['vggish_weights']:raise RuntimeError('VGGish identity')
 bs=Path.home()/'.cache/huggingface/hub/models--bert-base-uncased/snapshots'/BREV
 if any(sha(bs/n)!=h for n,h in BERT_FILES.items()):raise RuntimeError('BERT identity')
 import inspect,transformers,decord,torchvggish
 td=Path(inspect.getfile(torchvggish)).parent;tree=hashlib.sha256(b''.join(f.relative_to(td).as_posix().encode()+b'\0'+f.read_bytes() for f in sorted(td.rglob('*.py')))).hexdigest()
 if tree!=EXPECTED_VGG_TREE_SHA or {'torch':torch.__version__,'transformers':transformers.__version__,'numpy':np.__version__,'decord':decord.__version__}!=RUNTIME:raise RuntimeError('runtime/source identity')
 torch.manual_seed(26026);np.random.seed(26026);torch.use_deterministic_algorithms(True);dev='cuda' if torch.cuda.is_available() else 'cpu';clip=CLIPModel.from_pretrained(CLIP,revision=CREV,local_files_only=True).to(dev).eval();proc=CLIPImageProcessor.from_pretrained(CLIP,revision=CREV,local_files_only=True);bert=AutoModel.from_pretrained(BERT,revision=BREV,local_files_only=True).to(dev).eval();tok=AutoTokenizer.from_pretrained(BERT,revision=BREV,local_files_only=True);vg=torchvggish.vggish(postprocess=False).to(dev).eval();return dev,clip,proc,bert,tok,vg
def wav16(p):
 with wave.open(str(p)) as w:
  if w.getnchannels()!=1 or w.getframerate()!=16000 or w.getsampwidth()!=2:raise RuntimeError('wav contract')
  return np.frombuffer(w.readframes(w.getnframes()),dtype='<i2').astype(np.float32)/32768
def one(v,q,a,out,M,g):
 dev,clip,proc,bert,tok,vg=M;d=float(q['duration_seconds']);T=math.ceil(d)
 if q['video_available']:
  from decord import VideoReader,cpu
  probe=video_probe(q['cache_path']);fps=probe['fps'];ix=[frame_index(j,d,fps,probe['nframes']) for j in range(T)];times=[i/fps for i in ix];imgs=[];backs=[]
  try:vr=VideoReader(q['cache_path'],ctx=cpu(0),num_threads=1)
  except Exception:vr=None
  for i,t in zip(ix,times):
   try:
    if vr is None:raise RuntimeError('decord init')
    imgs.append(vr[i].asnumpy());backs.append('decord')
   except Exception:
    imgs.append(ffmpeg_rgb(q['cache_path'],t,probe['width'],probe['height']));backs.append('ffmpeg')
  vis=[]
  for s in range(0,T,32):
   enc=proc(images=imgs[s:s+32],return_tensors='pt')['pixel_values'].to(dev)
   with torch.inference_mode():vis.append(clip.get_image_features(pixel_values=enc).float().cpu().numpy())
  visual=np.concatenate(vis).astype('float32')
 else:visual=np.zeros((T,512),'float32');fps=0.;ix=[-1]*T;times=[-1.]*T;backs=['unavailable']*T;probe={'fps':0.,'codec':'unavailable','width':0,'height':0,'nframes':0,'ffprobe_sha256':'0'*64}
 audio=[];wav=wav16(q['wav_path']) if q['audio_available'] else np.zeros(0,'float32')
 if q['audio_available']:
  from torchvggish import vggish_input,vggish_params
  vggish_params.EXAMPLE_HOP_SECONDS=1.0;need=int(math.ceil(((T-1)+vggish_params.EXAMPLE_WINDOW_SECONDS+.05)*16000));wav=np.pad(wav,(0,max(0,need-len(wav))));patch=vggish_input.waveform_to_examples(wav,16000)
  if len(patch)<T:raise RuntimeError('VGGish coverage')
  for s in range(0,T,128):
   with torch.inference_mode():audio.extend(vg(patch[s:min(T,s+128)].to(dev)).float().cpu().numpy())
  audio=np.asarray(audio,dtype='float32')
 else:audio=np.zeros((T,128),'float32')
 chunks=[c for c in a['chunks'] if isinstance(c.get('text'),str) and c['text'].strip() and isinstance(c.get('start'),(int,float)) and isinstance(c.get('end'),(int,float)) and c['end']>c['start']]
 emb=[]
 for s in range(0,len(chunks),64):
  e=tok([c['text'] for c in chunks[s:s+64]],padding=True,truncation=True,max_length=64,return_tensors='pt').to(dev)
  with torch.inference_mode():emb.extend(bert(**e).last_hidden_state[:,0].float().cpu().numpy())
 text=np.zeros((T,768),'float32');tm=np.zeros(T,'uint8')
 for j in range(T):
  ww=np.array([overlap(c['start'],c['end'],j,d) for c in chunks]);sel=np.where(ww>0)[0]
  if len(sel):text[j]=np.average(np.asarray(emb)[sel],axis=0,weights=ww[sel]);tm[j]=1
 vm=np.full(T,int(bool(q['video_available'])),'uint8');am=np.full(T,int(bool(q['audio_available'])),'uint8')
 if not vm.any():visual.fill(0)
 if not am.any():audio.fill(0)
 masks=np.stack([vm,am,tm],1);dst=Path(out)/f'{v}.npz';atomic_npz(dst,visual=visual,audio=audio,text=text,availability=masks,duration=np.array(d),G=np.array(g))
 dec={'probe':probe,'backends':backs,'indices':ix,'effective_times':times};comp=composite(q,a,g,dec);return {'schema':'v26_thvl_video_features_v1','id':v,'T':T,'duration':d,'file':str(dst.resolve()),'sha256':sha(dst),'composite_input_sha256':comp,'shapes':[list(visual.shape),list(audio.shape),list(text.shape),list(masks.shape)],'finite':bool(np.isfinite(visual).all() and np.isfinite(audio).all() and np.isfinite(text).all()),'video_probe':probe,'frame_backends':backs,'frame_indices':ix,'fps':fps,'effective_timestamps':times,'source':{'qc_sha256':canonical_hash(q),'media_sha256':sha(q['cache_path']),'wav_sha256':sha(q['wav_path']),'asr_sha256':canonical_hash(a),'identity_sha256':canonical_hash(identity()),'producer_sha256':sha(__file__)},'labels_or_gt_read':False}
def main():
 p=argparse.ArgumentParser();p.add_argument('--split',choices=('train','val'),required=True);p.add_argument('--split-manifest',required=True);p.add_argument('--qc-dir',required=True);p.add_argument('--asr-dir',required=True);p.add_argument('--g-jsonl',required=True);p.add_argument('--out',required=True);p.add_argument('--limit',type=int);p.add_argument('--id');a=p.parse_args();sm=json.load(open(a.split_manifest));expected=314 if a.split=='train' else 32
 ap,ah,expected=AUTH[a.split]
 if str(Path(a.split_manifest).resolve())!=ap or sha(ap)!=ah or set(sm)!=( {'corpus','ids','producer_sha256','split','v23_global_source_sha256'} if a.split=='train' else {'bags_sha256','corpus','evidence_config_sha256','evidence_manifest_sha256','evidence_producer_sha256','ids','join_producer_sha256','labels_manifest_sha256','schema_version','split','v23_global_source_sha256'}) or sm.get('corpus')!='thvl' or sm.get('split')!=a.split or len(sm.get('ids',[]))!=expected or len(set(sm['ids']))!=expected:raise RuntimeError('authoritative split')
 q,z=load_inputs(a.qc_dir,a.asr_dir);gg={};grows={}
 for raw in open(a.g_jsonl,'rb'):
  r=json.loads(raw)
  if set(r)!={'video_id','global_causal_score'} or r['video_id'] in gg:raise RuntimeError('G exact schema/duplicate')
  gg[r['video_id']]=float(r['global_causal_score']);grows[r['video_id']]=hashlib.sha256(raw).hexdigest()
 if set(q)!=set(sm['ids']) or set(gg)!=set(sm['ids']):raise RuntimeError('exact split/QC/ASR/G coverage')
 out=Path(a.out);out.mkdir(parents=True,exist_ok=True);ids=[a.id] if a.id else sorted(q)[:a.limit] if a.limit else sorted(q);M=models();st=time.time();rows=[]
 for v in ids:
  rp=out/(v+'.json')
  if rp.exists():
   old=json.load(open(rp));rows.append(verify_resume(old,q[v],z[v],gg[v]));continue
  if v not in gg or not math.isfinite(gg[v]):raise RuntimeError('G binding')
  r=one(v,q[v],z[v],out,M,gg[v]);atomic_text(rp,json.dumps(r,sort_keys=True)+'\n');rows.append(r)
 import inspect,torchvggish
 td=Path(inspect.getfile(torchvggish)).parent;tree=hashlib.sha256(b''.join(f.relative_to(td).as_posix().encode()+b'\0'+f.read_bytes() for f in sorted(td.rglob('*.py')))).hexdigest();
 if tree!=EXPECTED_VGG_TREE_SHA:raise RuntimeError('VGGish tree identity')
 front={'package':'torchvggish','source_tree_sha256':tree,'weights_sha256':EXPECTED['vggish_weights'],'sample_rate':16000,'mel_bins':64,'mel_min':125,'mel_max':7500,'stft_window_ms':25,'stft_hop_ms':10,'example_window_s':.96,'example_hop_s':1.0,'postprocess':False};root=hashlib.sha256((sha(a.split_manifest)+sha(a.g_jsonl)+json.dumps(front,sort_keys=True)+''.join(r['id']+'\t'+r['sha256']+'\t'+r['composite_input_sha256']+'\t'+grows[r['id']]+'\n' for r in sorted(rows,key=lambda x:x['id']))).encode()).hexdigest();atomic_text(out/'manifest.json',json.dumps({'schema':'v26_thvl_features_v1','design_sha256':DESIGN_SHA,'split':a.split,'split_manifest_sha256':sha(a.split_manifest),'g_source_sha256':sha(a.g_jsonl),'models':MODELS,'revisions':{'clip':CREV,'bert':BREV},'expected_hashes':EXPECTED|BERT_FILES,'runtime':RUNTIME,'vggish_frontend':front,'n':len(rows),'root_sha256':root,'elapsed':time.time()-st,'labels_or_gt_read':False},indent=2,sort_keys=True)+'\n')
if __name__=='__main__':main()
