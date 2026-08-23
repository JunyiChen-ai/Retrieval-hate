from __future__ import annotations

import hashlib, random

from .protocol import exact_knapsack


def _seed(*parts):return int.from_bytes(hashlib.sha256("||".join(map(str,parts)).encode()).digest()[:8],"big")

def b2_executions(execute,split_seed,refit_seed,video_id,budget):
    """Twenty complete executions; callers average run metrics, never scores.

    The draw stream is the registered Philox counter-based generator keyed by
    exactly ``(split_seed, refit_seed, video_id, budget, draw_id)``.
    """
    from .arms import philox_generator
    rows=[]
    for draw_id in range(20):
        rng=philox_generator(split_seed,refit_seed,video_id,budget,draw_id)
        row=dict(execute(rng,draw_id));row["draw_id"]=draw_id;rows.append(row)
    if len(rows)!=20 or len({r["draw_id"] for r in rows})!=20:raise RuntimeError("HALT_B2_DRAWS")
    return rows

def b3_order(actions):
    by={(a.split(":")[1],int(a.rsplit(":",1)[1])):a for a in actions};selected=[];windows=[]
    while len(selected)<len(actions):
        remaining=sorted({w for _,w in by if by[(_,w)] not in selected})
        if not remaining:break
        w=15 if not windows and 15 in remaining else max(remaining,key=lambda q:(min(abs(q-x) for x in windows),-q))
        windows.append(w)
        for kind in ("ocr","dense4"):
            a=by.get((kind,w))
            if a is not None:selected.append(a)
    return tuple(selected)

def b6_route(router_positive,b3_feasible_package):return tuple(b3_feasible_package) if bool(router_positive) else ()

def b12_first(scores,cost_ms,budget_ms):
    ticks=[int(__import__("math").ceil(float(c)*10)) for c in cost_ms];budget=int(__import__("math").floor(float(budget_ms)*10))
    chosen=exact_knapsack(scores,ticks,budget);return None if not chosen else chosen[0]

def rank_feasible(actions,scores,costs,budget,max_actions=12,ratio=False):
    ranked=sorted(actions,key=lambda a:(-(float(scores[a])/float(costs[a]) if ratio else float(scores[a])),float(costs[a]),a))
    out=[];spent=0.0
    for a in ranked:
        c=float(costs[a])
        if len(out)<max_actions and spent+c<=budget:out.append(a);spent+=c
    return tuple(out)

def execute_registered_arm(arm,actions,costs,budget,signals,router_positive=True):
    """Frozen inference mechanics for B3--B12; training is outside this helper."""
    if arm=="B3":return rank_feasible(b3_order(actions),{a:-i for i,a in enumerate(b3_order(actions))},costs,budget)
    if arm=="B4":return rank_feasible(actions,signals["salience"],costs,budget)
    if arm=="B5":return rank_feasible(actions,signals["uncertainty"],costs,budget)
    if arm=="B6":return b6_route(router_positive,execute_registered_arm("B3",actions,costs,budget,signals))
    if arm in ("B7","B8"):return rank_feasible(actions,signals[arm],costs,budget)
    if arm=="B9":return rank_feasible(actions,signals["singleton"],costs,budget)
    if arm=="B10":return rank_feasible(actions,signals["set_utility"],costs,budget)
    if arm=="B11":return rank_feasible(actions,signals["singleton_ridge"],costs,budget,ratio=True)
    if arm=="B12":
        remaining=list(actions);out=[];spent=0.0
        while remaining and len(out)<12:
            feasible=[a for a in remaining if spent+float(costs[a])<=budget]
            if not feasible:break
            scores=[signals["set_utility"][a] for a in feasible];cs=[costs[a] for a in feasible]
            ix=b12_first(scores,cs,budget-spent)
            if ix is None:break
            chosen=feasible[ix];out.append(chosen);spent+=float(costs[chosen]);remaining.remove(chosen)
        return tuple(out)
    raise KeyError("unregistered arm: "+arm)

def execute_stateful_arm(arm,actions,costs,budget,score_callback,
                         router_positive=True,decision_overhead=0.0,max_actions=12):
    """Discrete acquisition loop with post-purchase state recomputation.

    ``score_callback(chosen)`` is invoked at every decision for the stateful
    arms.  The returned record is directly serializable as an acquisition
    trace; STOP and charged decision overhead are never implicit.
    """
    stateful=arm in ("B7","B8","B10","B12")
    if arm=="B6" and not router_positive:
        charge=float(decision_overhead)
        if charge<=0 or charge>budget:raise RuntimeError("HALT_ROUTER_DECISION_OVERHEAD")
        return {"actions":(),"estimated_cost":charge,"realized_cost":charge,"overhead_cost":charge,
                "decisions":[{"step":0,"state":[],"status":"STOP","reason":"ROUTER_NULL","decision_overhead_ms":charge}]}
    chosen=[];action_spent=0.;overhead_spent=0.;decisions=[]
    while len(chosen)<max_actions:
        remaining=[a for a in actions if a not in chosen]
        if not remaining:break
        # Every policy decision consumes the registered overhead.  Stateful
        # arms additionally recompute their scores, but budget semantics do not
        # change by arm.
        charge=float(decision_overhead)
        if action_spent+overhead_spent+charge>budget:
            decisions.append({"step":len(decisions),"state":list(chosen),"status":"STOP","reason":"OVERHEAD_BUDGET","decision_overhead_ms":0.})
            break
        overhead_spent+=charge
        signals=score_callback(tuple(chosen))
        expected_remaining=float(budget)-action_spent-overhead_spent
        declared=signals.get("remaining_budget")
        if declared is not None and abs(float(declared)-expected_remaining)>1e-8:
            raise RuntimeError("HALT_REMAINING_BUDGET_SEMANTICS")
        feasible=[a for a in remaining if action_spent+overhead_spent+float(costs[a])<=budget]
        if not feasible:
            decisions.append({"step":len(decisions),"state":list(chosen),"status":"STOP","reason":"NO_FEASIBLE_ACTION","decision_overhead_ms":charge})
            break
        if arm=="B3":order=b3_order(actions);pick=min(feasible,key=lambda a:order.index(a))
        elif arm=="B4":pick=max(feasible,key=lambda a:(signals["salience"][a],-float(costs[a]),a))
        elif arm=="B5":pick=max(feasible,key=lambda a:(signals["uncertainty"][a],-float(costs[a]),a))
        elif arm=="B6":
            order=b3_order(actions);pick=min(feasible,key=lambda a:order.index(a))
        elif arm in ("B7","B8"):pick=max(feasible,key=lambda a:(signals[arm][a],-float(costs[a]),a))
        elif arm=="B9":pick=max(feasible,key=lambda a:(signals["singleton"][a],-float(costs[a]),a))
        elif arm=="B10":pick=max(feasible,key=lambda a:(signals["set_utility"][a],-float(costs[a]),a))
        elif arm=="B11":pick=max(feasible,key=lambda a:(signals["singleton_ridge"][a]/float(costs[a]),-float(costs[a]),a))
        elif arm=="B12":
            ix=b12_first([signals["set_utility"][a] for a in feasible],[costs[a] for a in feasible],budget-action_spent-overhead_spent)
            if ix is None:
                decisions.append({"step":len(decisions),"state":list(chosen),"status":"STOP","reason":"NONPOSITIVE_UTILITY","decision_overhead_ms":charge})
                break
            pick=feasible[ix]
        else:raise KeyError("unregistered arm: "+arm)
        decisions.append({"step":len(decisions),"state":list(chosen),"status":"ACQUIRE","action_id":pick,
                          "estimated_cost_ms":float(costs[pick]),"realized_cost_ms":float(costs[pick]),"decision_overhead_ms":charge})
        chosen.append(pick);action_spent+=float(costs[pick])
        if not stateful and arm not in ("B3","B4","B5","B6","B9","B11"):break
    return {"actions":tuple(chosen),"estimated_cost":action_spent+overhead_spent,
            "realized_cost":action_spent+overhead_spent,"overhead_cost":overhead_spent,"decisions":decisions}

def execute_random_arm(actions,costs,budget,rng,decision_overhead=0.25):
    """A complete, costed B2 policy execution including its terminal decision."""
    order=list(actions);rng.shuffle(order);chosen=[];action_spent=0.;overhead_spent=0.;decisions=[]
    while True:
        charge=float(decision_overhead)
        if charge<=0:raise RuntimeError("HALT_RANDOM_DECISION_OVERHEAD")
        if action_spent+overhead_spent+charge>budget:
            raise RuntimeError("HALT_RANDOM_UNCHARGED_STOP")
        overhead_spent+=charge
        feasible=[a for a in order if a not in chosen and action_spent+overhead_spent+float(costs[a])<=budget]
        if not feasible:
            decisions.append({"step":len(decisions),"state":list(chosen),"status":"STOP","reason":"NO_FEASIBLE_ACTION","decision_overhead_ms":charge})
            break
        pick=feasible[0];chosen.append(pick);action_spent+=float(costs[pick])
        decisions.append({"step":len(decisions),"state":list(chosen[:-1]),"status":"ACQUIRE","action_id":pick,
                          "estimated_cost_ms":float(costs[pick]),"realized_cost_ms":float(costs[pick]),"decision_overhead_ms":charge})
    total=action_spent+overhead_spent
    return {"actions":tuple(chosen),"estimated_cost":total,"realized_cost":total,
            "overhead_cost":overhead_spent,"decisions":decisions}
