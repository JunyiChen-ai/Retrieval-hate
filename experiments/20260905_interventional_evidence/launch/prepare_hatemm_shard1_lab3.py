"""Transfer only still-needed odd-shard raw videos; never touch active extraction."""
import json
import os
from pathlib import Path
import socket
import subprocess

root = Path(__file__).resolve().parents[3]
os.chdir(root)
run = root/'runs/20260905_interventional_evidence/prepare_hatemm_shard1_lab3'
run.mkdir(parents=True, exist_ok=True)
(run/'run.pid').write_text(str(os.getpid()))
print('host='+socket.gethostname(), flush=True)
ids = sorted(sum([(root/'results/reproduction/splits'/f'hatemm_{s}.txt').read_text().split()
                  for s in ['train','val','test']], []))[1::2]
cache = root/'data/interventional_evidence/hatemm'
needed = [v for v in ids if not all((cache/f'K{k}'/f'{v}.json').is_file() for k in [30,4])]
sources = [root/'data/video/HateMM/All'/f'{v}.mp4' for v in needed]
assert all(p.is_file() and p.stat().st_size > 0 for p in sources)
manifest = dict(host=socket.gethostname(), shard=1, shards=2, expected_shard_ids=ids,
                transferred_ids=needed, source='data/video/HateMM/All',
                destination='uoa-lab3:~/data/HateMM/video', bytes=sum(p.stat().st_size for p in sources))
(run/'config.json').write_text(json.dumps(manifest, indent=2))
subprocess.run(['ssh','uoa-lab3','mkdir -p ~/data/HateMM/video ~/Retrieval-hate/data/video/HateMM'],check=True)
for i in range(0,len(sources),20):
    subprocess.run(['scp','-q',*[str(p) for p in sources[i:i+20]],'uoa-lab3:data/HateMM/video/'],check=True)
    print(f'transferred {min(i+20,len(sources))}/{len(sources)}',flush=True)
subprocess.run(['ssh','uoa-lab3','test -e ~/Retrieval-hate/data/video/HateMM/All || ln -s ~/data/HateMM/video ~/Retrieval-hate/data/video/HateMM/All'],check=True)
subprocess.run(['scp','-rq',str(root/'data/ASR/HateMM'),'uoa-lab3:Retrieval-hate/data/ASR/'],check=True)
(run/'completion.json').write_text(json.dumps(dict(state='TRANSFER_FINISHED', videos=len(needed))))
print('Transfer finished. Recheck current cache and stop the old full-list process before splitting; do not launch here.',flush=True)
