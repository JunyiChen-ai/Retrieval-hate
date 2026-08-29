#!/usr/bin/env python3
"""Steward join of THVL train video labels only; temporal fields are forbidden."""
import argparse,json
from pathlib import Path
from artifacts import atomic,sha
from core import DESIGN_SHA
from feature_manifest import verify
def run(features,bags,id_manifest,out):
 fm=verify(features);im=json.load(open(id_manifest));ids=sorted(im['ids'])
 if fm['split']!='train' or sorted(fm['ids'])!=ids:raise RuntimeError('train ID binding')
 labels={};forbidden={'spans','timestamps','intervals','frame_gt','annotations','segments'}
 for line in open(bags):
  r=json.loads(line)
  if forbidden&set(r) or set(r)!={'corpus','families','global_causal_score','source_hashes','split','video_id','video_label'} or r['corpus']!='thvl' or r['split']!='train' or r['video_id'] in labels or r['video_label'] not in (0,1):raise RuntimeError('weak bag schema')
  labels[r['video_id']]=r['video_label']
 if sorted(labels)!=ids:raise RuntimeError('weak label coverage')
 x={'schema':'v26_train_video_labels_v1','design_sha256':DESIGN_SHA,'split':'train','ids':ids,'labels':labels,'feature_manifest_sha256':sha(features),'temporal_labels_read':False};atomic(out,x);return x
def main():
 p=argparse.ArgumentParser();p.add_argument('--features',required=True);p.add_argument('--bags',required=True);p.add_argument('--id-manifest',required=True);p.add_argument('--out',required=True);a=p.parse_args();run(a.features,a.bags,a.id_manifest,a.out)
if __name__=='__main__':main()
