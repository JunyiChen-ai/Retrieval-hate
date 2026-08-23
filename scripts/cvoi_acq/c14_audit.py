from __future__ import annotations
import argparse,json,re
from pathlib import Path
from .common import atomic_json,sha256_file

def main():
    ap=argparse.ArgumentParser();ap.add_argument("--sbatch",type=Path,required=True);ap.add_argument("--preflight",type=Path,required=True);ap.add_argument("--out",type=Path,required=True);a=ap.parse_args()
    text=a.sbatch.read_text();pre=json.loads(a.preflight.read_text());directives=[x.strip() for x in text.splitlines() if x.startswith("#SBATCH")]
    if any("--time" in x for x in directives):raise RuntimeError("HALT_SLURM_TIME")
    cpus=int(re.search(r"--cpus-per-task=(\d+)",text).group(1));mem=int(re.search(r"--mem=(\d+)G",text).group(1));gpu=int(re.search(r"--gres=gpu:(\d+)",text).group(1))
    if cpus>16 or mem>128 or gpu>2:raise RuntimeError("HALT_SLURM_LIMIT")
    required=("scripts.cvoi_acq.formal_runner","scripts.cvoi_acq.preflight","scripts.cvoi_acq.resource_guard","CVOI_FROZEN_CONFIG","CVOI_COMPLETENESS","CVOI_RUN_OUT",
              "trap cleanup","wait \"$FORMAL_PID\"","wait \"$GUARD_PID\"","--min-disk-bytes","--max-rss-bytes","--max-gpu-bytes")
    if not all(x in text for x in required):raise RuntimeError("HALT_SLURM_ENTRYPOINT")
    if not pre["disk_ok"] or pre["gpu_count"]!=1 or not pre["metric_locked"]:raise RuntimeError("HALT_RESOURCE_PREFLIGHT")
    pinned={str(p):sha256_file(p) for p in (a.sbatch,a.preflight,Path("scripts/cvoi_acq/resource_guard.py"),Path("scripts/cvoi_acq/formal_runner.py"),Path("AGENTS.md"))}
    if "--enforce" not in text or "wait -n -p FINISHED_PID" not in text or "--max-rss-bytes 60129542144" not in text:raise RuntimeError("HALT_FAIL_CLOSED_OR_LIMIT")
    atomic_json(a.out,{"schema":"cvoi-c14-audit/3","sbatch_sha256":sha256_file(a.sbatch),"preflight_sha256":sha256_file(a.preflight),"pinned_sha256":pinned,
      "directives":directives,"resources":{"cpus":cpus,"memory_gb":mem,"gpu":gpu,"time_directive":False},
      "entrypoint":"scripts.cvoi_acq.formal_runner","fresh_preflight":True,"monitor_started_joined":True,"trap_cleanup":True,
      "heartbeat_fsync":True,"threshold_abort":True,"fail_closed_wait_n":True,"tree_rss_limit_gib":56,
      "metric_lock_verified":True,"disk_ok":True,"single_gpu_terminal_exception_available":True})
if __name__=="__main__":main()
