from __future__ import annotations

import gc, statistics, time, random, os, platform, subprocess, sys
from typing import Callable, Any

def benchmark_action(action_id: str, fn: Callable[[], Any], repetitions: int = 5,
                     use_cuda: bool = False) -> dict:
    if repetitions != 5: raise ValueError("registered protocol requires five repetitions")
    raw=[]
    for rep in range(repetitions):
        if use_cuda:
            import torch
            torch.cuda.synchronize(); start_ev=torch.cuda.Event(True); end_ev=torch.cuda.Event(True); start_ev.record()
        gc.collect(); t0=time.perf_counter_ns(); status="ok"; error=None
        try: fn()
        except Exception as exc: status="failed"; error=type(exc).__name__+":"+str(exc)
        if use_cuda:
            end_ev.record(); torch.cuda.synchronize(); cuda_ms=float(start_ev.elapsed_time(end_ev))
        else: cuda_ms=None
        raw.append({"repetition":rep+1,"wall_ns":time.perf_counter_ns()-t0,"cuda_ms":cuda_ms,
                    "status":status,"error":error})
    binding=[r["wall_ns"] for r in raw[1:]]
    return {"schema":"cvoi-action-cost/1","action_id":action_id,"raw_repetitions":raw,
            "binding_wall_ns":int(statistics.median(binding)),
            "binding_cuda_ms":(statistics.median(r["cuda_ms"] for r in raw[1:]) if use_cuda else None)}

def benchmark_phased(action_id, phases, repetitions=5, use_cuda=False):
    """Time decode/preprocess/inference/postprocess without hiding failed/retry cost."""
    import tracemalloc
    if tuple(phases)!=("decode","preprocess","inference","postprocess"):
        raise ValueError("registered phase order required")
    raw=[]
    for rep in range(1,repetitions+1):
        if use_cuda:
            import torch;torch.cuda.synchronize();gpu_start=int(torch.cuda.memory_allocated());torch.cuda.reset_peak_memory_stats();start=torch.cuda.Event(True);end=torch.cuda.Event(True);start.record()
        phase_ns={};status="ok";error=None;retries=0;value=None
        tracemalloc.start();wall0=time.perf_counter_ns()
        try:
            for name,fn in phases.items():
                t=time.perf_counter_ns();value=fn(value);phase_ns[name]=time.perf_counter_ns()-t
                if isinstance(value,dict):retries=max(retries,int(value.get("retries",0)))
        except Exception as exc:status="failed";error=type(exc).__name__+":"+str(exc)
        current,peak=tracemalloc.get_traced_memory();tracemalloc.stop()
        if use_cuda:
            end.record();torch.cuda.synchronize();cuda_ms=float(start.elapsed_time(end));gpu_peak=int(torch.cuda.max_memory_allocated());gpu_end=int(torch.cuda.memory_allocated())
        else:cuda_ms=None;gpu_peak=None;gpu_start=None;gpu_end=None
        raw.append({"repetition":rep,"wall_ns":time.perf_counter_ns()-wall0,"cuda_ms":cuda_ms,
                    "phase_ns":phase_ns,"allocated_bytes_peak":peak,"gpu_allocated_bytes_start":gpu_start,
                    "gpu_allocated_bytes_end":gpu_end,"gpu_allocated_bytes_peak":gpu_peak,
                    "gpu_allocated_bytes_incremental_peak":(gpu_peak-gpu_start if use_cuda else None),
                    "retries":retries,"status":status,"error":error})
    binding=raw[1:] if len(raw)>1 else raw
    return {"schema":"cvoi-action-cost/1","action_id":action_id,"raw_repetitions":raw,
            "binding_wall_ns":int(statistics.median(x["wall_ns"] for x in binding)),
            "binding_cuda_ms":(statistics.median(x["cuda_ms"] for x in binding) if use_cuda else None)}

def randomized_benchmark(action_ids, factory, use_cuda=False, seed=20260814, warmup_ids=None):
    """Run exactly 100 unrecorded train warmups, then five trials/action in seeded order."""
    ids=list(action_ids)
    if not ids:raise ValueError("no actions")
    rng=random.Random(seed)
    warm=list(warmup_ids) if warmup_ids is not None else ids
    if not warm:raise ValueError("no train warmup actions")
    for _ in range(100):
        aid=warm[rng.randrange(len(warm))];phases=factory(aid)
        value=None
        for fn in phases.values():value=fn(value)
    order=[aid for aid in ids for _ in range(5)];rng.shuffle(order);by={aid:[] for aid in ids}
    # Each randomized invocation is individually synchronized and recorded; aggregate 2--5 later.
    for aid in order:
        row=benchmark_phased(aid,factory(aid),repetitions=1,use_cuda=use_cuda)["raw_repetitions"][0]
        row["repetition"]=len(by[aid])+1;by[aid].append(row)
    out=[]
    for aid in ids:
        rows=by[aid];out.append({"schema":"cvoi-action-cost/1","action_id":aid,"raw_repetitions":rows,
            "binding_wall_ns":int(statistics.median(x["wall_ns"] for x in rows[1:])),
            "binding_cuda_ms":(statistics.median(x["cuda_ms"] for x in rows[1:]) if use_cuda else None)})
    return out

def benchmark_overheads(policy_fn,encode_fn,retrieval_fn,solver_fn,repetitions=5,use_cuda=False):
    phases={"policy":policy_fn,"encoding":encode_fn,"retrieval":retrieval_fn,"solver":solver_fn}
    rows=[]
    for name,fn in phases.items(): rows.append(benchmark_action("overhead:"+name,fn,repetitions,use_cuda))
    return rows

def hardware_software_lock():
    lock={"python":sys.version,"platform":platform.platform(),"hostname":platform.node(),
          "processor":platform.processor(),"pid":os.getpid()}
    try:
        import torch
        lock.update({"torch":torch.__version__,"cuda":torch.version.cuda,
                     "cudnn":torch.backends.cudnn.version(),
                     "gpu":torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
                     "gpu_properties":(str(torch.cuda.get_device_properties(0)) if torch.cuda.is_available() else None)})
    except Exception as exc: lock["torch_error"]=type(exc).__name__
    for command,key in ((["nvidia-smi","--query-gpu=name,uuid,driver_version,pstate,power.limit,clocks.current.graphics,clocks.current.memory","--format=csv,noheader"],"nvidia_smi"),
                        (["ffmpeg","-version"],"ffmpeg")):
        try: lock[key]=subprocess.run(command,check=True,text=True,capture_output=True).stdout.splitlines()[0]
        except Exception as exc: lock[key+"_error"]=type(exc).__name__+":"+str(exc)
    try:
        import decord;lock["decord"]=decord.__version__
    except Exception as exc:lock["decord_error"]=type(exc).__name__
    return lock

def summarize_costs(rows):
    if not rows: raise ValueError("no cost rows")
    retry=sum(sum(int(x.get("retries",0)) for x in r["raw_repetitions"]) for r in rows)
    failed=sum(sum(x["status"]!="ok" for x in r["raw_repetitions"]) for r in rows)
    vals=[int(r["binding_wall_ns"]) for r in rows]
    return {"schema":"cvoi-cost-summary/1","n_actions":len(rows),"median_wall_ns":int(statistics.median(vals)),
            "p95_wall_ns":int(sorted(vals)[min(len(vals)-1,int(.95*len(vals)))]),
            "retry_count":retry,"failed_repetitions":failed,"hardware_software":hardware_software_lock()}

class FoldCostRegressor:
    """Small deterministic ridge regressor; fit-group provenance is mandatory."""
    def __init__(self,l2=1e-3): self.l2=float(l2);self.coef=None;self.fit_groups=None
    def fit(self,x,y,groups):
        import numpy as np
        X=np.c_[np.ones(len(x)),np.asarray(x,float)];Y=np.asarray(y,float)
        self.coef=np.linalg.solve(X.T@X+self.l2*np.eye(X.shape[1]),X.T@Y)
        self.fit_groups=tuple(sorted(set(groups)));return self
    def predict(self,x,eval_groups=None):
        import numpy as np
        if self.coef is None:raise RuntimeError("HALT_COST_REGRESSOR_UNFIT")
        if eval_groups is not None and set(eval_groups)&set(self.fit_groups):raise RuntimeError("HALT_COST_FOLD_LEAKAGE")
        return np.c_[np.ones(len(x)),np.asarray(x,float)]@self.coef

class FrozenGBDCostRegressor:
    """The preregistered fold-internal cost estimator (appendix section 6)."""
    def __init__(self):
        from sklearn.ensemble import GradientBoostingRegressor
        self.model=GradientBoostingRegressor(loss="squared_error",n_estimators=100,
            max_depth=3,learning_rate=.05,min_samples_leaf=10,random_state=20260815)
        self.fit_groups=None;self.type_medians={}
    def fit(self,x,y,groups,action_types):
        import numpy as np
        if not (len(x)==len(y)==len(groups)==len(action_types)):raise ValueError("cost fit length mismatch")
        self.model.fit(np.asarray(x,float),np.asarray(y,float));self.fit_groups=tuple(sorted(set(groups)))
        for typ in sorted(set(action_types)):
            z=[float(v) for v,t in zip(y,action_types) if t==typ]
            self.type_medians[typ]=float(np.median(z))
        return self
    def predict(self,x,eval_groups=None,action_types=None):
        import numpy as np
        if self.fit_groups is None:raise RuntimeError("HALT_COST_REGRESSOR_UNFIT")
        if eval_groups is not None and set(eval_groups)&set(self.fit_groups):raise RuntimeError("HALT_COST_FOLD_LEAKAGE")
        out=self.model.predict(np.asarray(x,float))
        if action_types is not None:
            out=np.asarray([v if np.isfinite(v) and v>=0 else self.type_medians[t] for v,t in zip(out,action_types)])
        return out

def quantize_cost_ns(value_ns):
    """Feasibility unit is ceil(cost / 0.1 ms), represented in integer ticks."""
    import math
    return int(math.ceil(max(0.0,float(value_ns))/100000.0))

def enforce_estimated_budget(order,estimated_costs,budget,max_actions=12):
    chosen=[];spent=0.0
    for a in order:
        c=max(0.0,float(estimated_costs[a]))
        if len(chosen)<max_actions and spent+c<=budget:chosen.append(a);spent+=c
    return {"actions":tuple(chosen),"estimated_spent":spent,"budget":float(budget)}

def charge_execution(action_rows,overhead_rows,estimated_budget_ns):
    """Join acquisition and all mandatory overhead charges for one execution."""
    required={"policy","encoding","retrieval","solver"};seen={}
    for r in overhead_rows:
        name=r["action_id"].split(":",1)[-1];seen[name]=seen.get(name,0)+int(r["binding_wall_ns"])
    missing=required-set(seen)
    if missing:raise RuntimeError("HALT_UNCHARGED_OVERHEAD:"+",".join(sorted(missing)))
    action_ns=sum(int(r["binding_wall_ns"]) for r in action_rows);overhead_ns=sum(seen.values());total=action_ns+overhead_ns
    return {"schema":"cvoi-execution-cost/1","action_ns":action_ns,"overhead_ns":seen,"total_ns":total,
            "estimated_budget_ns":int(estimated_budget_ns),"realized_overshoot_ns":max(0,total-int(estimated_budget_ns)),
            "failed_repetitions":sum(sum(x["status"]!="ok" for x in r["raw_repetitions"]) for r in action_rows+list(overhead_rows)),
            "retry_count":sum(sum(int(x.get("retries",0)) for x in r["raw_repetitions"]) for r in action_rows+list(overhead_rows))}
