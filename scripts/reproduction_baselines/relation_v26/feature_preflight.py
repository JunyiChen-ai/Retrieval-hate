#!/usr/bin/env python3
"""Label-blind dependency/snapshot census; never performs full extraction."""
import argparse,hashlib,json,importlib.util,shutil
from pathlib import Path
from feature_manifest import MODELS
def main():
 p=argparse.ArgumentParser();p.add_argument('--out',required=True);a=p.parse_args();cache=Path.home()/'.cache/huggingface/hub';snaps={}
 for model in ('models--openai--clip-vit-base-patch16','models--bert-base-uncased','models--bert-base-chinese'):
  d=cache/model/'snapshots';snaps[model]=sorted(x.name for x in d.glob('*')) if d.exists() else []
 x={'schema':'v26_dependency_census_v1','models':MODELS,'snapshots':snaps,'modules':{n:bool(importlib.util.find_spec(n)) for n in ('torch','transformers','torchvggish','decord')},'ffmpeg':shutil.which('ffmpeg'),'full_extraction_started':False,'labels_or_gt_read':False};Path(a.out).write_text(json.dumps(x,indent=2,sort_keys=True)+'\n');print(json.dumps(x,indent=2))
if __name__=='__main__':main()
