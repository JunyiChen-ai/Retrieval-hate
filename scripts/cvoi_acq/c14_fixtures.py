from __future__ import annotations
import argparse,json,subprocess,sys,time
from pathlib import Path
from .common import atomic_json,sha256_file

def main():
    ap=argparse.ArgumentParser();ap.add_argument("--out-dir",type=Path,required=True);ap.add_argument("--completeness",type=Path,required=True);a=ap.parse_args();a.out_dir.mkdir(parents=True,exist_ok=True);py=sys.executable;rows=[]
    def guard(pid,out,*limits):return subprocess.Popen([py,"-m","scripts.cvoi_acq.resource_guard","--pid",str(pid),"--out",str(out),"--interval","1","--min-disk-bytes",str(limits[0]),"--max-rss-bytes",str(limits[1]),"--max-gpu-bytes",str(limits[2])])
    p=subprocess.Popen(["sleep","1"]);out=a.out_dir/"normal.jsonl";r=guard(p.pid,out,1,1<<30,1<<30);p.wait();rr=r.wait();events=[json.loads(x) for x in out.open()];rows.append({"case":"normal","pass":rr==0 and events[-1]["event"]=="exit","rc":rr,"events":len(events)})
    pf=a.out_dir/"preflight_fail.json";r=subprocess.run([py,"-m","scripts.cvoi_acq.preflight","--completeness",str(a.completeness),"--projected-bytes",str(10**18),"--out",str(pf),"--enforce"]);rows.append({"case":"preflight_fail","pass":r.returncode==41,"rc":r.returncode})
    p=subprocess.Popen(["sleep","30"]);out=a.out_dir/"threshold_abort.jsonl";r=guard(p.pid,out,2**63,1<<30,1<<30);p.wait();rr=r.wait();events=[json.loads(x) for x in out.open()];rows.append({"case":"threshold_abort","pass":rr==42 and events[-1]["event"]=="abort" and p.returncode<0,"guard_rc":rr,"formal_rc":p.returncode})
    formal=subprocess.Popen(["sleep","30"]);guard_crash=subprocess.Popen(["bash","-c","exit 9"]);t=time.monotonic();grc=guard_crash.wait();formal.terminate();frc=formal.wait();rows.append({"case":"guard_crash_fail_closed","pass":grc==9 and frc<0 and time.monotonic()-t<5,"guard_rc":grc,"formal_rc":frc})
    if not all(x["pass"] for x in rows):raise RuntimeError("HALT_C14_FIXTURE")
    atomic_json(a.out_dir/"report.json",{"schema":"cvoi-c14-fixtures/1","cases":rows,"passed":len(rows),"failed":0,
      "source_sha256":{p:sha256_file(Path(p)) for p in ("scripts/cvoi_acq/c14_fixtures.py","scripts/cvoi_acq/resource_guard.py","scripts/cvoi_acq/preflight.py","scripts/slurm/cvoi_formal.sbatch")}})
if __name__=="__main__":main()
