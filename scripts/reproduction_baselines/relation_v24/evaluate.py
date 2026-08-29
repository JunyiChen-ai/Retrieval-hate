#!/usr/bin/env python3
"""Separate post-freeze video-label evaluator; never used by train/infer."""
import argparse,json,sys
from pathlib import Path
from sklearn.metrics import average_precision_score,roc_auc_score
sys.path.insert(0,str(Path(__file__).resolve().parent))
from train import sha
def main():
 p=argparse.ArgumentParser();p.add_argument('--pred-dir',required=True);p.add_argument('--labels',required=True);p.add_argument('--out',required=True);a=p.parse_args();d=Path(a.pred_dir);m=json.load(open(d/'raw_manifest.json'));rp=d/'predictions.jsonl'
 if not m['raw_frozen_before_labels'] or sha(rp)!=m['predictions_sha256']:raise RuntimeError('unfrozen predictions')
 rows=list(map(json.loads,open(rp)));lab=json.load(open(a.labels))
 if set(lab)!=set(r['video_id'] for r in rows) or any(x not in (0,1) for x in lab.values()):raise RuntimeError('label coverage/schema mismatch')
 y=[lab[r['video_id']] for r in rows];s=[r['video_score'] for r in rows];Path(a.out).write_text(json.dumps({'video_ap':float(average_precision_score(y,s)),'video_roc':float(roc_auc_score(y,s)),'predictions_sha256':sha(rp)},indent=2)+'\n')
if __name__=='__main__':main()
