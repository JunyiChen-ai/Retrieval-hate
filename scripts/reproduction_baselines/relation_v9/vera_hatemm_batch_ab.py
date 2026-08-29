#!/usr/bin/env python3
"""Mechanical multi-video HateMM VERA batch1/batch2 semantic A/B."""
import hashlib,json,re
from pathlib import Path
import vera_adapter as vera
from vera_fast_infer import ReusableVideoReader,predict_batch
def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def normalized(text):return re.sub(r'\s+',' ',text).strip()
def main():
 roots=[Path('results/reproduction/official_val/final/vera/hatemm/seed_234/raw'),Path('results/reproduction/official_val/final/vera/hatemm/seed_234/val_infer/raw'),Path('results/reproduction/official_val/final/vera/hatemm/seed_234/train_infer/raw')];candidates=[]
 for path in sorted(p for root in roots for p in root.glob('*.json')):
  row=json.loads(path.read_text());scores={x['score'] for x in row['segments']};n=len(row['segments'])
  if len(scores)>1 and n%2==1:candidates.append((n,str(path),row))
 selected=sorted(candidates,key=lambda x:(x[0],x[1]))[:3]
 if len(selected)<3:raise RuntimeError('need three mechanically selected odd/mixed raw videos')
 prompt_path=Path('results/reproduction/official_val/tuning/vera/hatemm/selected_prompt.json');selection=json.loads(prompt_path.read_text());model,tok,backend=vera.load_model(selection['attention_backend']);root=Path('results/reproduction/relation_v9/hatemm_vera_batch_ab');root.mkdir(parents=True,exist_ok=True);reports=[]
 for _,reference_path,reference in selected:
  vid=reference['video_id'];starts=[float(x['start']) for x in reference['segments']];reader=ReusableVideoReader(vera.video_path('hatemm',vid));outputs={}
  for batch_size in (1,2):
   records=[]
   for offset in range(0,len(starts),batch_size):
    batch=starts[offset:offset+batch_size];images=[reader.frames(start,10.,8) for start in batch]
    for start,(score,response) in zip(batch,predict_batch(model,tok,images,selection['prompts'],batch_size)):
     records.append({'start':start,'end':min(reader.duration,start+10.),'score':score,'response':response})
   outputs[batch_size]={'video_id':vid,'duration':reader.duration,'segments':records}
  paths={}
  for batch,payload in outputs.items():
   path=root/f'{vid}.batch{batch}.json';path.write_text(json.dumps(payload,ensure_ascii=False));paths[batch]=path
  pairs=list(zip(outputs[1]['segments'],outputs[2]['segments']))
  reports.append({'video_id':vid,'n_windows':len(starts),'reference_raw':str(Path(reference_path).resolve()),'reference_sha256':digest(reference_path),'batch1_sha256':digest(paths[1]),'batch2_sha256':digest(paths[2]),'start_exact':all(a['start']==b['start'] for a,b in pairs),'end_exact':all(a['end']==b['end'] for a,b in pairs),'order_exact':[x['start'] for x in outputs[1]['segments']]==[x['start'] for x in outputs[2]['segments']],'parsed_score_exact':all(a['score']==b['score'] for a,b in pairs),'normalized_response_exact':all(normalized(a['response'])==normalized(b['response']) for a,b in pairs),'raw_payload_exact':outputs[1]==outputs[2]})
 keys=('start_exact','end_exact','order_exact','parsed_score_exact','normalized_response_exact');report={'corpus':'hatemm','selection_rule':'existing raw only: mixed scores and odd windows; first three by (window count,path), no GT','prompt':str(prompt_path.resolve()),'prompt_sha256':digest(prompt_path),'backend':backend,'videos':reports,'semantic_exact':all(all(x[k] for k in keys) for x in reports)}
 (root/'report.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
 if not report['semantic_exact']:raise SystemExit(2)
if __name__=='__main__':main()
