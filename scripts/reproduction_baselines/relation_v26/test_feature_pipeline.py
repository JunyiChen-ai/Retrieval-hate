import hashlib,json,subprocess,sys,tempfile,unittest,wave
from pathlib import Path
import numpy as np,torch
sys.path.insert(0,str(Path(__file__).parent));import extract_thvl_1hz as ex,raw_verifier as rv
from feature_manifest import verify
def put(p,x):p.write_text(json.dumps(x,sort_keys=True)+'\n')
class T(unittest.TestCase):
 def test_resume_all_input_and_output_tamper(self):
  with tempfile.TemporaryDirectory() as z:
   d=Path(z);media=d/'m';wav=d/'w';media.write_bytes(b'm');wav.write_bytes(b'w');q={'cache_path':str(media),'wav_path':str(wav)};a={'chunks':[]};g=1.;npz=d/'x';npz.write_bytes(b'x');r={'schema':'v26_thvl_video_features_v1','file':str(npz),'sha256':ex.sha(npz),'composite_input_sha256':ex.composite(q,a,g),'source':{'producer_sha256':ex.sha(ex.__file__)},'labels_or_gt_read':False};self.assertIs(ex.verify_resume(r,q,a,g),r)
   for f in (lambda:media.write_bytes(b'M'),lambda:wav.write_bytes(b'W'),lambda:a.update(chunks=[1]),lambda:npz.write_bytes(b'X')):
    f();self.assertRaises(RuntimeError,ex.verify_resume,r,q,a,g);r['composite_input_sha256']=ex.composite(q,a,g);r['sha256']=ex.sha(npz)
   self.assertRaises(RuntimeError,ex.verify_resume,r,q,a,2.)
 def test_qc_semantic_identity_and_frame_rounding(self):
  with tempfile.TemporaryDirectory() as z:
   d=Path(z);media=d/'m';wav=d/'w';media.write_bytes(b'm');wav.write_bytes(b'w');q={'cache_path':str(media),'wav_path':str(wav),'duration_seconds':2.,'video_available':True,'audio_available':True,'hf_revision':'r'};a={'chunks':[]};g=0.;p=d/'x';p.write_bytes(b'x');r={'schema':'v26_thvl_video_features_v1','file':str(p),'sha256':ex.sha(p),'composite_input_sha256':ex.composite(q,a,g),'source':{'producer_sha256':ex.sha(ex.__file__)},'labels_or_gt_read':False}
   for k,val in [('duration_seconds',3.),('video_available',False),('audio_available',False),('hf_revision','other')]:
    qq=dict(q);qq[k]=val;self.assertRaises(RuntimeError,ex.verify_resume,r,qq,a,g)
  self.assertEqual(ex.frame_index(0,3.,1.,3),0);self.assertEqual(ex.frame_index(0,3.,2.,6),1);self.assertEqual(ex.frame_index(2,2.1,2.,5),4);self.assertEqual(ex.frame_index(99,2.1,2.,5),4)
 def raw(self,d,split,vals,masks):
  raw=d/(split+'raw');raw.mkdir();ids=[];qd=d/(split+'q');ad=d/(split+'a');qd.mkdir();ad.mkdir();gf=d/(split+'g');gl=[];rows=[]
  for i,(v,m) in enumerate(zip(vals,masks)):
   vid=f'{split}{i}';ids.append(vid);T=len(v);media=d/(vid+'.media');wav=d/(vid+'.wav');media.write_bytes(b'm'+bytes([i]));wav.write_bytes(b'w'+bytes([i]));q={'audio_available':True,'bytes':2,'cache_path':str(media),'duration_seconds':float(T),'full_decode_ok':True,'hf_path':'x','hf_revision':'x','media_sha256':ex.sha(media),'opaque_id':vid,'source_manifest_sha256':'s','status':'ok','video_available':True,'wav_path':str(wav),'wav_sha256':ex.sha(wav)};a={'chunks':[],'labels_or_temporal_gt_opened':False,'language':'en','model':'x','n_chunks':0,'opaque_id':vid,'text':'','wav_sha256':ex.sha(wav)};put(qd/(vid+'.json'),q);put(ad/(vid+'.json'),a);gl.append(json.dumps({'video_id':vid,'global_causal_score':float(i)},separators=(',',':'))+'\n');npz=raw/(vid+'.npz');av=np.asarray(m,dtype='uint8');visual=np.tile(np.asarray(v)[:,None],(1,512))*av[:,0,None];audio=np.tile(np.asarray(v)[:,None],(1,128))*av[:,1,None];text=np.tile(np.asarray(v)[:,None],(1,768))*av[:,2,None];np.savez(npz,visual=visual,audio=audio,text=text,availability=av,duration=np.array(float(T)),G=np.array(float(i)));r={'schema':'v26_thvl_video_features_v1','id':vid,'T':T,'duration':float(T),'file':str(npz.resolve()),'sha256':ex.sha(npz),'composite_input_sha256':ex.composite(q,a,float(i)),'shapes':[[T,512],[T,128],[T,768],[T,3]],'finite':True,'frame_indices':list(range(T)),'fps':1.,'effective_timestamps':[j+.5 for j in range(T)],'source':{'media_sha256':ex.sha(media),'wav_sha256':ex.sha(wav),'asr_sha256':hashlib.sha256(json.dumps(a,sort_keys=True,separators=(',',':')).encode()).hexdigest(),'producer_sha256':ex.sha(ex.__file__)},'labels_or_gt_read':False};put(raw/(vid+'.json'),r);rows.append(r)
  for r in rows:
   q=json.load(open(qd/(r['id']+'.json')));a=json.load(open(ad/(r['id']+'.json')));av=np.load(raw/(r['id']+'.npz'))['availability'];q['video_available']=bool(av[:,0].all());q['audio_available']=bool(av[:,1].all());
   if q['video_available']:
    subprocess.run(['ffmpeg','-y','-v','error','-f','lavfi','-i','color=black:s=16x16:r=1','-t',str(r['T']),'-pix_fmt','yuv420p','-f','mp4',q['cache_path']],check=True);q['media_sha256']=ex.sha(q['cache_path']);pr=ex.video_probe(q['cache_path']);r['video_probe']=pr;r['fps']=pr['fps'];r['frame_indices']=[ex.frame_index(j,r['duration'],pr['fps'],pr['nframes']) for j in range(r['T'])];r['effective_timestamps']=[i/pr['fps'] for i in r['frame_indices']];r['frame_backends']=['decord']*r['T']
   else:r['video_probe']={'fps':0.,'codec':'unavailable','width':0,'height':0,'nframes':0,'ffprobe_sha256':'0'*64};r['fps']=0.;r['frame_indices']=[-1]*r['T'];r['effective_timestamps']=[-1.]*r['T'];r['frame_backends']=['unavailable']*r['T']
   put(qd/(r['id']+'.json'),q);dec={'probe':r['video_probe'],'backends':r['frame_backends'],'indices':r['frame_indices'],'effective_times':r['effective_timestamps']};r['composite_input_sha256']=ex.composite(q,a,float(np.load(raw/(r['id']+'.npz'))['G']),dec);r['source']={'qc_sha256':ex.canonical_hash(q),'media_sha256':ex.sha(q['cache_path']),'wav_sha256':ex.sha(q['wav_path']),'asr_sha256':ex.canonical_hash(a),'identity_sha256':ex.canonical_hash(ex.identity()),'producer_sha256':ex.sha(ex.__file__)};put(raw/(r['id']+'.json'),r)
  gf.write_text(''.join(gl));sm=d/(split+'ids');put(sm,{'ids':ids,'split':split});front={'package':'torchvggish','source_tree_sha256':ex.EXPECTED_VGG_TREE_SHA,'weights_sha256':ex.EXPECTED['vggish_weights'],'sample_rate':16000,'mel_bins':64,'mel_min':125,'mel_max':7500,'stft_window_ms':25,'stft_hop_ms':10,'example_window_s':.96,'example_hop_s':1.0,'postprocess':False};gline={json.loads(b)['video_id']:hashlib.sha256(b).hexdigest() for b in gf.read_bytes().splitlines(keepends=True)};root=hashlib.sha256((ex.sha(sm)+ex.sha(gf)+json.dumps(front,sort_keys=True)+''.join(r['id']+'\t'+r['sha256']+'\t'+r['composite_input_sha256']+'\t'+gline[r['id']]+'\n' for r in rows)).encode()).hexdigest();rm={'schema':'v26_thvl_features_v1','design_sha256':ex.DESIGN_SHA,'split':split,'split_manifest_sha256':ex.sha(sm),'g_source_sha256':ex.sha(gf),'models':ex.MODELS,'revisions':{'clip':ex.CREV,'bert':ex.BREV},'expected_hashes':ex.EXPECTED|ex.BERT_FILES,'runtime':ex.RUNTIME,'vggish_frontend':front,'n':len(ids),'root_sha256':root,'elapsed':0.,'labels_or_gt_read':False};put(raw/'manifest.json',rm);seal=d/(split+'seal');rv.seal(seal,raw,split,sm,qd,ad,gf);return raw,sm,ids,seal,qd,ad,gf
 def test_missing_and_tiny_train_val_finalizer(self):
  with tempfile.TemporaryDirectory() as z:
   d=Path(z);tr,ti,tids,ts,tq,ta,tg=self.raw(d,'train',[[1,3],[5,7]],[[[1,1,0],[1,1,0]],[[1,1,1],[1,1,1]]]);va,vi,vids,vs,vq,vaa,vg=self.raw(d,'val',[[3,5]],[[[1,0,0],[1,0,1]]]);script=Path(__file__).with_name('finalize_thvl_features.py');common=lambda raw,seal,q,a,g,sm:['--raw',str(raw),'--raw-seal',str(seal),'--qc-dir',str(q),'--asr-dir',str(a),'--g-jsonl',str(g),'--split-manifest',str(sm)];subprocess.run([sys.executable,str(script),'--split','train',*common(tr,ts,tq,ta,tg,ti),'--out',str(d/'to')],check=True);mp=d/'to/train_moments.npz';mh=hashlib.sha256(mp.read_bytes()).hexdigest();subprocess.run([sys.executable,str(script),'--split','val',*common(va,vs,vq,vaa,vg,vi),'--train-moments',str(mp),'--out',str(d/'vo')],check=True);self.assertEqual(hashlib.sha256(mp.read_bytes()).hexdigest(),mh);m=verify(d/'vo/manifest.json');r=json.load(open(m['records'][vids[0]]));self.assertEqual(r['seconds'][0]['audio'],[]);self.assertEqual(r['seconds'][0]['text'],[]);mom=np.load(mp);gold=(3-mom['visual_mean'][0])/mom['visual_std'][0];self.assertAlmostEqual(r['seconds'][0]['visual'][0],gold);self.assertTrue(all(any(x['availability']) for x in r['seconds']))
   mm=json.load(open(d/'vo/manifest.json'));mm['root_sha256']='0'*64;put(d/'vo/manifest.bad.json',mm);self.assertRaises(RuntimeError,verify,d/'vo/manifest.bad.json')
 def test_audio_frontend_096_hop1_golden(self):
  from torchvggish import vggish_input,vggish_params
  vggish_params.EXAMPLE_HOP_SECONDS=1.;T=3;sr=16000;need=int(np.ceil(((T-1)+vggish_params.EXAMPLE_WINDOW_SECONDS+.05)*sr));x=np.zeros(need,np.float32);a=vggish_input.waveform_to_examples(x,sr);inside=x.copy();inside[int(1.10*sr):int(1.80*sr)]=np.linspace(-.8,.8,int(.70*sr),endpoint=False);ai=vggish_input.waveform_to_examples(inside,sr);outside=x.copy();outside[int(2.10*sr):int(2.80*sr)]=np.linspace(-.8,.8,int(.70*sr),endpoint=False);ao=vggish_input.waveform_to_examples(outside,sr);self.assertGreaterEqual(len(a),T);self.assertFalse(torch.equal(a[1],ai[1]));self.assertTrue(torch.equal(a[1],ao[1]));self.assertFalse(torch.equal(a[2],ao[2]));self.assertEqual(vggish_params.EXAMPLE_WINDOW_SECONDS,.96);self.assertEqual(vggish_params.EXAMPLE_HOP_SECONDS,1.)
 def test_raw_npz_sidecar_seal_and_live_input_tamper(self):
  with tempfile.TemporaryDirectory() as z:
   d=Path(z);raw,sm,ids,seal,q,a,g=self.raw(d,'train',[[1,2]],[[[1,1,1],[1,1,1]]]);v=ids[0]
   # NPZ bytes, sidecar bytes, seal bytes, and a currently-bound input each fail.
   (raw/(v+'.npz')).write_bytes(b'bad');self.assertRaises(Exception,rv.verify,raw,'train',sm,q,a,g)
  with tempfile.TemporaryDirectory() as z:
   d=Path(z);raw,sm,ids,seal,q,a,g=self.raw(d,'train',[[1,2]],[[[1,1,1],[1,1,1]]]);v=ids[0];r=json.load(open(raw/(v+'.json')));r['duration']=9.;put(raw/(v+'.json'),r);self.assertRaises(Exception,rv.verify,raw,'train',sm,q,a,g)
  with tempfile.TemporaryDirectory() as z:
   d=Path(z);raw,sm,ids,seal,q,a,g=self.raw(d,'train',[[1,2]],[[[1,1,1],[1,1,1]]]);s=json.load(open(seal));s['raw_root_sha256']='0'*64;put(seal,s);script=Path(__file__).with_name('finalize_thvl_features.py');cmd=[sys.executable,str(script),'--split','train','--raw',str(raw),'--raw-seal',str(seal),'--qc-dir',str(q),'--asr-dir',str(a),'--g-jsonl',str(g),'--split-manifest',str(sm),'--out',str(d/'o')];self.assertNotEqual(subprocess.run(cmd,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL).returncode,0)
  with tempfile.TemporaryDirectory() as z:
   d=Path(z);raw,sm,ids,seal,q,a,g=self.raw(d,'train',[[1,2]],[[[1,1,1],[1,1,1]]]);v=ids[0];qq=json.load(open(q/(v+'.json')));Path(qq['cache_path']).write_bytes(b'tamper');self.assertRaises(Exception,rv.verify,raw,'train',sm,q,a,g)
 def test_raw_nan_and_wrong_dimension(self):
  for bad in ('nan','dim'):
   with tempfile.TemporaryDirectory() as z:
    d=Path(z);raw,sm,ids,seal,q,a,g=self.raw(d,'train',[[1,2]],[[[1,1,1],[1,1,1]]]);v=ids[0];p=raw/(v+'.npz');x=dict(np.load(p));x['visual']=x['visual'].astype('float32',copy=True);x['visual']=(np.zeros((2,1),'float32') if bad=='dim' else x['visual']);
    if bad=='nan':x['visual'][0,0]=np.nan
    np.savez(p,**x);r=json.load(open(raw/(v+'.json')));r['sha256']=ex.sha(p);put(raw/(v+'.json'),r);self.assertRaises(Exception,rv.verify,raw,'train',sm,q,a,g)
 def test_qc_fields_vgg_tree_and_partial_mix_fail(self):
  for field,value in [('duration_seconds',3.),('video_available',False),('audio_available',False)]:
   with tempfile.TemporaryDirectory() as z:
    d=Path(z);raw,sm,ids,seal,q,a,g=self.raw(d,'train',[[1,2]],[[[1,1,1],[1,1,1]]]);v=ids[0];qq=json.load(open(q/(v+'.json')));qq[field]=value;put(q/(v+'.json'),qq);self.assertRaises(Exception,rv.verify,raw,'train',sm,q,a,g)
  with tempfile.TemporaryDirectory() as z:
   d=Path(z);raw,sm,ids,seal,q,a,g=self.raw(d,'train',[[1,2]],[[[1,1,1],[1,1,1]]]);m=json.load(open(raw/'manifest.json'));m['vggish_frontend']['source_tree_sha256']='0'*64;put(raw/'manifest.json',m);self.assertRaises(Exception,rv.verify,raw,'train',sm,q,a,g)
  with tempfile.TemporaryDirectory() as z:
   d=Path(z);raw,sm,ids,seal,q,a,g=self.raw(d,'train',[[1,2],[3,4]],[[[1,1,1],[1,1,1]],[[1,1,1],[1,1,1]]]);(raw/(ids[1]+'.json')).unlink();self.assertRaises(Exception,rv.verify,raw,'train',sm,q,a,g)
 def test_backend_tamper_and_ffmpeg_repeat_pixel_exact(self):
  with tempfile.TemporaryDirectory() as z:
   d=Path(z);raw,sm,ids,seal,q,a,g=self.raw(d,'train',[[1,2]],[[[1,1,1],[1,1,1]]]);v=ids[0];r=json.load(open(raw/(v+'.json')));r['frame_backends'][0]='ffmpeg';put(raw/(v+'.json'),r);self.assertRaises(Exception,rv.verify,raw,'train',sm,q,a,g)
  with tempfile.TemporaryDirectory() as z:
   p=Path(z)/'v.mp4';subprocess.run(['ffmpeg','-y','-v','error','-f','lavfi','-i','testsrc2=s=64x48:r=5','-t','2','-pix_fmt','yuv420p',str(p)],check=True);pr=ex.video_probe(p);t=ex.frame_index(1,2.,pr['fps'],pr['nframes'])/pr['fps'];a=ex.ffmpeg_rgb(p,t,pr['width'],pr['height']);b=ex.ffmpeg_rgb(p,t,pr['width'],pr['height']);self.assertTrue(np.array_equal(a,b));from decord import VideoReader,cpu;vr=VideoReader(str(p),ctx=cpu(0),num_threads=1);self.assertTrue(np.array_equal(a,vr[ex.frame_index(1,2.,pr['fps'],pr['nframes'])].asnumpy()))
if __name__=='__main__':unittest.main()
