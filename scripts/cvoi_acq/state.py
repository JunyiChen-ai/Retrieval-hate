from __future__ import annotations
import hashlib,random

LENGTHS=(0,1,2,4,8)
def philox_like_seed(*parts):return int.from_bytes(hashlib.sha256("||".join(map(str,parts)).encode()).digest()[:8],"big")
def random_prefixes(video_id,actions,n=8,seed=20260817):
    out=[]
    for j in range(n):
        order=list(actions);random.Random(philox_like_seed(video_id,seed,j)).shuffle(order)
        out.append([tuple(order[:m]) for m in LENGTHS if m<=len(order)]+[tuple(order)])
    return out
def target_rows(eval_group,fit_groups,states,actions):
    if eval_group in set(fit_groups):raise RuntimeError("HALT_TARGET_GROUP_LEAKAGE")
    return [{"eval_group":eval_group,"fit_groups":list(fit_groups),"state":list(s),"action":a}
            for s in states for a in actions if a not in s]

def uniform_order(actions):
    return tuple(sorted(actions,key=lambda a:(int(a.rsplit(":",1)[1]),a.split(":")[1])))

def salience_order(actions,salience):
    return tuple(sorted(actions,key=lambda a:(-float(salience[a]),int(a.rsplit(":",1)[1]),a)))

def rollout(order,budget,costs,max_actions=12):
    picked=[];spent=0.0
    for a in order:
        c=float(costs[a])
        if len(picked)>=max_actions or spent+c>float(budget): continue
        picked.append(a);spent+=c
    return {"actions":tuple(picked),"spent":spent,"terminal":True,"remaining":float(budget)-spent}

def deduplicate_states(rows):
    """Deduplicate without losing which generators supplied a state."""
    out={}
    for r in rows:
        key=(str(r["video_id"]),tuple(r["state"]))
        if key not in out: out[key]={**r,"weight":0.0,"provenance":[]}
        out[key]["weight"]+=float(r.get("weight",1.0))
        out[key]["terminal"]=bool(out[key].get("terminal",False) or r.get("terminal",False))
        p=str(r["generator"])
        if p not in out[key]["provenance"]: out[key]["provenance"].append(p)
    result=[out[k] for k in sorted(out)]
    totals={}
    for r in result:totals[r["video_id"]]=totals.get(r["video_id"],0.0)+r["weight"]
    for r in result:r["weight"]/=totals[r["video_id"]]
    return result

#: Appendix section 9 generator 3: one trajectory per fixed policy.
POLICY_GENERATORS=("uniform","salience","b5","b7")

def frozen_state_mixture(video_id,actions,salience,costs,budget,on_policy=None,seed=20260817,
                         uncertainty=None,b7_score=None):
    """Frozen equal-weight generator mixture (appendix section 9 generators 1--4).

    Generator 1 (empty state) is the length-0 prefix retained for every
    trajectory. Generator 2 is the eight seed-20260817 random orders. Generator
    3 is one trajectory each for uniform, salience, B5 and B7. Generator 4 is
    the three DAgger rounds. ``uncertainty``/``b7_score`` are the B5 and B7
    pre-action scores; omitting them yields a reduced mixture that the C9 audit
    rejects for the formal path.
    """
    generators={"uniform":uniform_order(actions),"salience":salience_order(actions,salience)}
    if uncertainty is not None:
        generators["b5"]=salience_order(actions,uncertainty)
    if b7_score is not None:
        generators["b7"]=salience_order(actions,b7_score)
    for j,order in enumerate(random_prefixes(video_id,actions,n=8,seed=seed)):
        generators["random_%02d"%j]=order[-1]
    if on_policy is not None:
        for r in range(3): generators["dagger_%d"%r]=tuple(on_policy(r))
    rows=[]
    for name,order in generators.items():
        rr=rollout(order,budget,costs,12)
        for m in LENGTHS:
            state=rr["actions"][:min(m,len(rr["actions"]))]
            rows.append({"video_id":video_id,"state":state,"generator":name,"weight":1.0/len(generators)})
        rows.append({"video_id":video_id,"state":rr["actions"],"generator":name,"weight":1.0/len(generators),"terminal":True})
    return deduplicate_states(rows)

def candidate_targets(video_id,state,proposed,all_actions,budget,costs,seed=20260818,limit=12):
    feasible=[a for a in proposed if a not in state and float(costs[a])<=budget]
    rest=[a for a in all_actions if a not in state and a not in feasible and float(costs[a])<=budget]
    rng=random.Random(philox_like_seed(video_id,tuple(state),seed));rng.shuffle(rest)
    return tuple((feasible+rest)[:limit])

def dagger_schedule(fit_policy,generate,base_rows):
    """Frozen four fits: round0=10, then generated rounds fit fresh 10/20/30."""
    rows=list(base_rows);trace=[]
    policy=fit_policy(tuple(rows),10,0)
    trace.append({"round":0,"epochs":10,"n_rows":len(rows)})
    for round_id,epochs in ((1,10),(2,20),(3,30)):
        new=list(generate(policy,round_id));rows.extend(new)
        policy=fit_policy(tuple(rows),epochs,round_id)
        trace.append({"round":round_id,"epochs":epochs,"n_added":len(new),"n_rows":len(rows)})
    return policy,trace

def utility_target_rows(video_id,eval_group,fit_groups,states,all_actions,costs,budget,predict_before_after,action_features,proposals):
    if eval_group in set(fit_groups):raise RuntimeError("HALT_TARGET_GROUP_LEAKAGE")
    merged={}
    for row in states:
        state=tuple(row["state"]);remaining=float(budget)-sum(float(costs[a]) for a in state)
        candidates=candidate_targets(video_id,state,proposals(row),all_actions,remaining,costs)
        for action in candidates:
            before,after,label_loss_delta=predict_before_after(state,action)
            key=(video_id,state,action)
            if key not in merged:
                merged[key]={"video_id":video_id,"eval_group":eval_group,"fit_groups":tuple(sorted(fit_groups)),
                  "state":state,"action_id":action,"action_features":list(action_features(action)),
                  "before_probability":float(before),"after_probability":float(after),"utility":float(label_loss_delta),
                  "estimated_cost":float(costs[action]),"remaining_budget":remaining,"weight":0.0,
                  "generator_provenance":[],"terminal":bool(row.get("terminal",False))}
            q=merged[key];q["weight"]+=float(row["weight"]);q["terminal"]|=bool(row.get("terminal",False))
            for p in row["provenance"]:
                if p not in q["generator_provenance"]:q["generator_provenance"].append(p)
    rows=[merged[k] for k in sorted(merged)]
    total=sum(x["weight"] for x in rows)
    if total<=0:raise RuntimeError("HALT_TARGET_WEIGHT")
    for x in rows:x["weight"]/=total
    return rows
