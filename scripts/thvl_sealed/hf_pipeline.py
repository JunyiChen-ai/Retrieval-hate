#!/usr/bin/env python3
"""Label-blind THVL validation media acquisition/QC/ASR at a pinned HF revision."""
import argparse,hashlib,json,os,subprocess,sys,time
from pathlib import Path
REPO='THVL/THVL-Bench';REV='5ea20ec4074dea9d3419e88fea944313ab25818d';ROOT=Path(__file__).resolve().parents[2]
PRIVATE=ROOT/'results/steward_private/thvl_bench';MAP=PRIVATE/'steward_id_media_map.json';PUBLIC=ROOT/'results/reproduction/thvl_sealed/validation_opaque_manifest.json';MEDIA=ROOT/'results/reproduction/thvl_sealed/validation_media_manifest.json'
def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
def cohort():
 pub=json.load(open(PUBLIC));media=json.load(open(MEDIA));want={r['hashed_id'] for r in pub['records']};q=[]
 for r in media['records']:
  if r['split']=='validation':q.append({'hashed_id':r['opaque_id'],'canonical_id':r['platform']+':'+r['id'],'repository_paths':[r['hf_path']]})
 if pub['revision']!=REV or len(q)!=32 or {r['hashed_id'] for r in q}!=want or any(len(r['repository_paths'])!=1 for r in q):raise RuntimeError('reconciled pinned revision/cohort mismatch')
 return sorted(q,key=lambda r:r['hashed_id'])
def atomic_json(path,x):
 t=path.with_suffix(path.suffix+'.tmp');t.parent.mkdir(parents=True,exist_ok=True);t.write_text(json.dumps(x,indent=2)+'\n');os.replace(t,path)
def metadata():
 from huggingface_hub import HfApi
 q=cohort();paths=sorted({p for r in q for p in r['repository_paths']});api=HfApi();info=api.get_paths_info(REPO,paths,revision=REV,repo_type='dataset',expand=True);by={x.path:x for x in info};rows=[]
 for r in q:
  ps=[]
  for p in r['repository_paths']:
   x=by.get(p);ps.append({'path':p,'exists':x is not None,'size_bytes':None if x is None else x.size,'blob_id':None if x is None else x.blob_id,'lfs_sha256':None if x is None or x.lfs is None else x.lfs.get('sha256')})
  rows.append({'hashed_id':r['hashed_id'],'canonical_platform':r['canonical_id'].split(':',1)[0],'paths':ps,'repository_media_available':bool(ps and all(x['exists'] for x in ps)),'repository_subtitle_sidecars':[]})
 payload={'repo_id':REPO,'revision':REV,'cohort':'opaque self-sealed validation, no labels','n_videos':len(q),'n_repository_paths':len(paths),'n_available_videos':sum(x['repository_media_available'] for x in rows),'total_bytes':sum(x['size_bytes'] or 0 for r in rows for x in r['paths']),'subtitle_availability':'no subtitle/caption sidecar files exist in the pinned repository tree','rows':rows};atomic_json(PRIVATE/'val32_hf_metadata.json',payload);print(json.dumps({k:payload[k] for k in ('n_videos','n_repository_paths','n_available_videos','total_bytes','subtitle_availability')},indent=2))
def download_qc():
 from huggingface_hub import hf_hub_download
 q=cohort();meta=json.load(open(PRIVATE/'val32_hf_metadata.json'));total=meta['total_bytes']
 if total>20*1024**3:raise RuntimeError('predeclared 20GiB low-cost cap exceeded')
 cache=PRIVATE/'hf_media_cache';wav=PRIVATE/'val32_wav16k';cache.mkdir(parents=True,exist_ok=True);wav.mkdir(parents=True,exist_ok=True);rows=[];t0=time.time()
 for i,r in enumerate(q):
  rec={'hashed_id':r['hashed_id'],'paths':[],'status':'missing_repository_path' if not r['repository_paths'] else 'pending'}
  for j,name in enumerate(r['repository_paths']):
   try:
    p=Path(hf_hub_download(REPO,name,repo_type='dataset',revision=REV,cache_dir=cache,resume_download=True));probe=subprocess.run(['ffprobe','-v','error','-show_streams','-show_format','-of','json',str(p)],capture_output=True,text=True);pj=json.loads(probe.stdout) if probe.returncode==0 else {};decode=subprocess.run(['ffmpeg','-v','error','-i',str(p),'-f','null','-'],capture_output=True);audio=any(x.get('codec_type')=='audio' for x in pj.get('streams',[]));video=any(x.get('codec_type')=='video' for x in pj.get('streams',[]));wpath=wav/f'{r["hashed_id"]}_{j}.wav'
    if audio:subprocess.run(['ffmpeg','-y','-v','error','-i',str(p),'-vn','-ac','1','-ar','16000',str(wpath)],check=True)
    rec['paths'].append({'repository_path':name,'cache_path':str(p.resolve()),'bytes':p.stat().st_size,'sha256':sha(p),'ffprobe_ok':probe.returncode==0,'decode_ok':decode.returncode==0,'video_available':video,'audio_available':audio,'duration_seconds':float((pj.get('format') or {}).get('duration') or 0),'wav_path':str(wpath.resolve()) if audio else None,'wav_sha256':sha(wpath) if audio else None});rec['status']='ok' if probe.returncode==0 and decode.returncode==0 and video else 'qc_failed'
   except Exception as e:rec['paths'].append({'repository_path':name,'error_type':type(e).__name__});rec['status']='download_failed'
  rows.append(rec);print(f'{i+1}/{len(q)} {r["hashed_id"][:10]} {rec["status"]}',flush=True)
 payload={'repo_id':REPO,'revision':REV,'label_or_gt_access':False,'download_cap_bytes':20*1024**3,'metadata_total_bytes':total,'elapsed_seconds':time.time()-t0,'coverage':{'videos':len(q),'ok':sum(x['status']=='ok' for x in rows),'missing_path':sum(x['status']=='missing_repository_path' for x in rows),'failed':sum(x['status'] not in ('ok','missing_repository_path') for x in rows),'audio_available':sum(any(p.get('audio_available') for p in x['paths']) for x in rows)},'rows':rows};atomic_json(PRIVATE/'val32_download_qc.json',payload);print(json.dumps(payload['coverage'],indent=2))
def main():
 p=argparse.ArgumentParser();p.add_argument('--stage',choices=['metadata','download-qc'],required=True);a=p.parse_args();metadata() if a.stage=='metadata' else download_qc()
if __name__=='__main__':main()
