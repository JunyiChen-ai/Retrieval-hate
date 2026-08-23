from __future__ import annotations
import argparse
from pathlib import Path
from .common import atomic_json
from .dense_asset_policy import assert_new_dense_asset,DENIED_OLD_K30
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,required=True);a=ap.parse_args();rejected=[]
 for p in (Path('data/CLIP_Embedding/HateMM/train_subclipK30_openai_clip-vit-large-patch14-336_HF.pt'),Path('data/CLIP_Embedding/HateMM/dev_seen_subclipK30_openai_clip-vit-large-patch14-336_HF.pt')):
  try:assert_new_dense_asset(p)
  except RuntimeError:rejected.append(str(p))
 new=assert_new_dense_asset(Path('artifacts/cvoi_acq/premetric-v2/visual-v10/train_dense4.f32'))
 if len(rejected)!=2:raise RuntimeError('HALT_OLD_K30_DENY_FIXTURE')
 atomic_json(a.out,{'schema':'cvoi-c5-consumer-deny-fixture/1','passed':True,'rejected':rejected,'denied_sha256':sorted(DENIED_OLD_K30),'accepted_new_asset':str(new),'candidate_metric_computed':False})
if __name__=='__main__':main()
