#!/usr/bin/env python3
from __future__ import annotations
import argparse,json,sys
from pathlib import Path
import torch
import torch.utils.data as tdata
ROOT=Path(__file__).resolve().parents[2]; BASE=ROOT/'scripts/reproduction_baselines'; MM=BASE/'multihateloc'; sys.path[:0]=[str(ROOT),str(BASE),str(MM)]
import data as mdata
from hate_common import data as hdata
from src.marked_temporal_splat import MarkedTemporalSplatMIL
from src.scoped_video_protocol import evaluator_test_ids
p=argparse.ArgumentParser(); p.add_argument('--corpus',required=True,choices=('hatemm','hateclipseg')); p.add_argument('--corpus-dir',required=True); p.add_argument('--device',default='cuda'); args=p.parse_args(); root=Path(args.corpus_dir).resolve(); selected=json.loads((root/'selection.json').read_text())['selected']; trial=Path(selected['trial_dir']); cfg=json.loads((trial/'config.json').read_text())['args']
model=MarkedTemporalSplatMIL({n:mdata.FEATURE_DIMS[n] for n in mdata.MODALITIES},cfg['hidden'],cfg['embed'],cfg['dropout'],cfg['k_proportion'],cfg['temperature']).to(args.device); model.load_state_dict(torch.load(trial/'checkpoint.pt',map_location=args.device,weights_only=True)); model.eval()
ids=evaluator_test_ids(args.corpus,hdata.load_split(args.corpus,'test')); batches=tdata.DataLoader(mdata.MultiModalDataset(args.corpus,ids,{v:0 for v in ids}),batch_size=cfg['batch_size'],shuffle=False,collate_fn=mdata.collate,num_workers=2); records={}
with torch.no_grad():
    for feats,_,lengths,mask,video_ids in batches:
        feats={k:v.to(args.device) for k,v in feats.items()}; prob=model(feats,mask.to(args.device))['prob']
        for i,vid in enumerate(video_ids): records[vid]=prob[i,:int(lengths[i])].cpu().tolist()
with (root/'scores.jsonl').open('w') as handle:
    for vid in ids: handle.write(json.dumps({'video_id':vid,'score_final':records[vid]})+'\n')
(root/'prediction_record.json').write_text(json.dumps({'split':'test','selected_trial':str(trial),'selected_by_validation_before_test_prediction':True,'test_labels_used_for_gradient_or_checkpoint_selection':False,'n_videos':len(records)},indent=2)+'\n'); print(f'wrote {len(records)} test videos')
