#!/usr/bin/env python3
from __future__ import annotations
import argparse,hashlib,json,subprocess,sys
from pathlib import Path
SPLITS=('train','validation','test');EXACT_KEYS={'platform','id','url','split'};SALT='THVL-media-availability-v1-2026-08-29'
def sha256_file(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
def load_opaque_manifest(path):
 p=Path(path)
 if p.suffix.lower()=='.csv':raise PermissionError('CSV annotations/labels are forbidden; provide steward opaque JSONL')
 rows=[];seen=set()
 for n,line in enumerate(open(p,encoding='utf-8'),1):
  if not line.strip():continue
  r=json.loads(line)
  if set(r)!=EXACT_KEYS:raise RuntimeError(f'line {n}: exact keys required; labels/extras forbidden')
  if r['split'] not in SPLITS or not all(isinstance(r[k],str) and r[k].strip() for k in EXACT_KEYS):raise RuntimeError(f'line {n}: invalid value')
  key=(r['platform'],r['id'])
  if key in seen:raise RuntimeError(f'duplicate opaque media identity: {key}')
  seen.add(key);rows.append(r)
 return rows
def mechanical_first10(rows,n=10,salt=SALT):
 out=[]
 for split in SPLITS:
  q=[r for r in rows if r['split']==split]
  q.sort(key=lambda r:(hashlib.sha256((salt+'\n'+r['platform']+'\n'+r['id']).encode()).hexdigest(),r['platform'],r['id']))
  out.extend(q[:n])
 return out
def command(url):
 return ['yt-dlp','--skip-download','--dump-single-json','--no-write-info-json','--no-write-playlist-metafiles','--no-write-subs','--no-write-auto-subs','--no-write-thumbnail','--no-call-home','--',url]
def default_runner(cmd):return subprocess.run(cmd,capture_output=True,text=True,check=False)
def summarize(row,payload):
 formats=payload.get('formats') or [];sizes=[x.get('filesize') or x.get('filesize_approx') for x in formats];sizes=[int(x) for x in sizes if isinstance(x,(int,float)) and x>=0]
 return {'platform':row['platform'],'opaque_id':row['id'],'split':row['split'],'url_sha256':hashlib.sha256(row['url'].encode()).hexdigest(),'available':True,'duration_seconds':payload.get('duration'),'estimated_max_format_bytes':max(sizes) if sizes else None,'subtitles_languages':sorted((payload.get('subtitles') or {}).keys()),'automatic_caption_languages':sorted((payload.get('automatic_captions') or {}).keys()),'audio_available':any(x.get('acodec') not in (None,'none') for x in formats),'extractor':payload.get('extractor_key') or payload.get('extractor'),'webpage_url_sha256':hashlib.sha256(str(payload.get('webpage_url',row['url'])).encode()).hexdigest()}
def preflight(rows,runner=default_runner):
 out=[]
 for r in mechanical_first10(rows):
  cmd=command(r['url'])
  if '--skip-download' not in cmd or any(x in cmd for x in ('--write-subs','--write-auto-subs','--write-thumbnail')):raise AssertionError('media-write guard failure')
  p=runner(cmd)
  if p.returncode!=0:out.append({'platform':r['platform'],'opaque_id':r['id'],'split':r['split'],'url_sha256':hashlib.sha256(r['url'].encode()).hexdigest(),'available':False,'error_type':'yt_dlp_metadata_unavailable','returncode':int(p.returncode)});continue
  try:payload=json.loads(p.stdout)
  except Exception:out.append({'platform':r['platform'],'opaque_id':r['id'],'split':r['split'],'url_sha256':hashlib.sha256(r['url'].encode()).hexdigest(),'available':False,'error_type':'invalid_metadata_json','returncode':int(p.returncode)});continue
  out.append(summarize(r,payload))
 return out
def main():
 p=argparse.ArgumentParser();p.add_argument('--opaque-media-manifest',required=True);p.add_argument('--out-dir',required=True);a=p.parse_args();out=Path(a.out_dir);out.mkdir(parents=True,exist_ok=False);rows=load_opaque_manifest(a.opaque_media_manifest);selected=mechanical_first10(rows);result=preflight(rows);rp=out/'availability.jsonl'
 with open(rp,'x') as f:
  for r in result:f.write(json.dumps(r,sort_keys=True)+'\n')
 prov={'method':'THVL_label_blind_metadata_availability_preflight','license_constraint':'CC BY-NC 4.0; non-commercial research only; no media redistribution','annotation_csv_opened':False,'labels_opened':False,'media_body_downloaded':False,'yt_dlp_mode':'--skip-download --dump-single-json; subtitles/captions availability metadata only','manifest_path':str(Path(a.opaque_media_manifest).resolve()),'manifest_sha256':sha256_file(a.opaque_media_manifest),'manifest_exact_schema':sorted(EXACT_KEYS),'selection':'SHA256(salt\\nplatform\\nid), first 10 per split','salt':SALT,'selected_counts':{s:sum(r['split']==s for r in selected) for s in SPLITS},'result_sha256':sha256_file(rp)};(out/'provenance.json').write_text(json.dumps(prov,indent=2)+'\n');print(json.dumps(prov,indent=2))
if __name__=='__main__':main()
