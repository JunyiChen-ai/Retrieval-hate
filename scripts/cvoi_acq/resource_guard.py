from __future__ import annotations

import argparse,json,os,resource,shutil,time,signal,subprocess
from pathlib import Path

from .common import ROOT,atomic_write,canonical_bytes


def snapshot(event,pid=None):
    disk=shutil.disk_usage(ROOT);row={"schema":"cvoi-resource-event/1","event":event,"monotonic_ns":time.monotonic_ns(),
      "pid":pid,"disk_free_bytes":disk.free,"self_maxrss_kib":resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}
    if pid is not None and Path(f"/proc/{pid}/status").exists():
        fields={}
        for line in Path(f"/proc/{pid}/status").read_text().splitlines():
            if line.startswith(("VmRSS:","VmHWM:","Threads:")):fields[line.split(":")[0]]=line.split(":",1)[1].strip()
        row["process_status"]=fields
    try:
        import torch
        row["gpu"]={"available":torch.cuda.is_available(),"name":torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
                    "allocated":torch.cuda.memory_allocated() if torch.cuda.is_available() else 0,
                    "reserved":torch.cuda.memory_reserved() if torch.cuda.is_available() else 0}
    except Exception as exc:row["gpu_error"]=type(exc).__name__
    return row

def process_tree(root):
    seen=set();stack=[root]
    while stack:
        pid=stack.pop()
        if pid in seen or not Path(f"/proc/{pid}").exists():continue
        seen.add(pid)
        try:stack.extend(int(x) for x in Path(f"/proc/{pid}/task/{pid}/children").read_text().split())
        except Exception:pass
    return sorted(seen)
def target_gpu_bytes(pids):
    total=0
    try:
        out=subprocess.check_output(["nvidia-smi","--query-compute-apps=pid,used_gpu_memory","--format=csv,noheader,nounits"],text=True)
        for line in out.splitlines():
            q=line.split(",");
            if int(q[0].strip()) in set(pids):total+=int(q[1].strip())*(1<<20)
    except Exception:return 0
    return total

def main():
    ap=argparse.ArgumentParser();ap.add_argument("--pid",type=int,required=True);ap.add_argument("--out",type=Path,required=True);ap.add_argument("--interval",type=float,default=30)
    ap.add_argument("--min-disk-bytes",type=int,required=True);ap.add_argument("--max-rss-bytes",type=int,required=True);ap.add_argument("--max-gpu-bytes",type=int,required=True);a=ap.parse_args()
    a.out.parent.mkdir(parents=True,exist_ok=True)
    aborted=False
    with a.out.open("xb",buffering=0) as stream:
      while True:
        alive=Path(f"/proc/{a.pid}").exists();row=snapshot("heartbeat" if alive else "exit",a.pid);row["limits"]={"min_disk_bytes":a.min_disk_bytes,"max_rss_bytes":a.max_rss_bytes,"max_gpu_bytes":a.max_gpu_bytes}
        if alive:
            pids=process_tree(a.pid);rss=0
            for pid in pids:
                try:
                    for line in Path(f"/proc/{pid}/status").read_text().splitlines():
                        if line.startswith("VmRSS:"):rss+=int(line.split()[1])*1024;break
                except Exception:pass
            gpu=target_gpu_bytes(pids);row.update({"process_tree_pids":pids,"target_tree_rss_bytes":rss,"target_tree_gpu_bytes":gpu})
            violations=[]
            if row["disk_free_bytes"]<a.min_disk_bytes:violations.append("disk")
            if rss>a.max_rss_bytes:violations.append("rss")
            if gpu>a.max_gpu_bytes:violations.append("gpu")
            if violations:
                row["event"]="abort";row["violations"]=violations;os.kill(a.pid,signal.SIGTERM);alive=False;aborted=True
        stream.write(canonical_bytes(row));os.fsync(stream.fileno())
        if not alive:break
        time.sleep(min(60,max(1,a.interval)))
    if aborted:raise SystemExit(42)

if __name__=="__main__":main()
