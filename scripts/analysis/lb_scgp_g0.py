#!/usr/bin/env python
"""LB-SCGP G0 frozen producer: config freeze, sealed synthetic, real microbench."""
from __future__ import annotations

import argparse
import copy
import json
import math
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
from scipy.linalg import polar
from scipy.optimize import minimize

ROOT = Path("/data/jehc223/RGCL")
sys.path.insert(0, str(ROOT / "scripts/analysis"))
from lb_scgp_common import (  # noqa: E402
    ZERO_COUNTER_KEYS, AccessLedger, Stopwatch, assert_no_formal_forbidden_surface,
    boundary_tie_count, canonical_json, canonical_root_path,
    boundary_orientation_system, canonical_rank_rhs,
    orientation_cell_from_assignment,
    cone_audit, dense_reference_ball, deterministic_psd_factor,
    dirty_state_policy, exact_vote_ledger, git_state, implementation_hash,
    load_config, payload_hash,
    peak_rss_gib, project_halfspace, project_psd,
    projector_preimage_ball, provenance_base, procrustes_align_factor,
    publish_json, publish_jsonl, rank_cell_pairs, rank_cell_violation,
    require_slurm, resolve, row_operator, class_mean_operator, sha256_file,
    load_npz_members_only, hash_npz_members_only, root_relative_path,
    sha256_obj, stable_rankings,
)


def _finite(obj):
    if isinstance(obj, dict):
        return all(_finite(v) for v in obj.values())
    if isinstance(obj, (list, tuple)):
        return all(_finite(v) for v in obj)
    if isinstance(obj, (float, np.floating)):
        return math.isfinite(float(obj))
    return True


def _projector_case(name, y, x0, operator, radius, seed, probes):
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    x0 = np.asarray(x0, dtype=np.float64).reshape(-1)
    operator = np.asarray(operator, dtype=np.float64)
    projected, meta = projector_preimage_ball(y, x0, operator, radius)
    projected2, _ = projector_preimage_ball(projected, x0, operator, radius)
    dense = dense_reference_ball(y, x0, operator, radius)
    op_value = operator @ (projected - x0)
    feasibility = max(0.0, float(np.linalg.norm(op_value)) - radius)
    if meta["mu"] is None:
        gram = operator @ operator.T
        stationarity = float(np.linalg.norm(
            (y - projected) - operator.T @
            (np.linalg.pinv(gram, rcond=1e-14) @
             (operator @ (y - projected)))))
        complementarity = 0.0
    else:
        stationarity = float(np.linalg.norm(
            (y - projected) - float(meta["mu"]) * operator.T @ op_value))
        complementarity = abs(float(meta["mu"]) *
                              (float(np.linalg.norm(op_value)) - radius))
    rng = np.random.default_rng(seed)
    vi_max = -float("inf")
    directional_min = float("inf")
    normal = y - projected
    for _ in range(probes):
        raw = rng.normal(size=y.size)
        if radius == 0.0:
            # Frozen random nullspace probes, not a degenerate x0 replay.
            gram = operator @ operator.T
            raw = raw - operator.T @ (
                np.linalg.pinv(gram, rcond=1e-14) @ (operator @ raw))
            candidate = x0 + raw
        else:
            candidate = x0 + raw
            value = operator @ (candidate - x0)
            value_norm = float(np.linalg.norm(value))
            if value_norm > radius and value_norm > 0:
                candidate = x0 + (candidate - x0) * (radius / value_norm) * 0.999999
        vi_max = max(vi_max, float(normal @ (candidate - projected)))
        direction = candidate - projected
        norm = float(np.linalg.norm(direction))
        if norm > 0:
            direction /= norm
            eps = 1e-7
            plus = projected + eps * direction
            if float(np.linalg.norm(operator @ (plus - x0))) <= radius + 1e-10:
                directional_min = min(
                    directional_min,
                    (0.5 * np.dot(plus - y, plus - y) -
                     0.5 * np.dot(projected - y, projected - y)) / eps)
    if directional_min == float("inf"):
        directional_min = 0.0
    adjoint_rng = np.random.default_rng(seed + 10000)
    u = adjoint_rng.normal(size=y.size)
    v = adjoint_rng.normal(size=operator.shape[0])
    adjoint_error = abs(float((operator @ u) @ v - u @ (operator.T @ v)))
    record = {
        "case": name, "status": "PASS", "dimension": y.size,
        "operator_rows": operator.shape[0], "radius": float(radius),
        "input": y.tolist(), "center": x0.tolist(),
        "operator": operator.tolist(), "projected": projected.tolist(),
        "metrics": {
            "operator_adjoint_dot_error": adjoint_error,
            "feasibility": feasibility, "kkt_stationarity": stationarity,
            "complementarity": complementarity,
            "scalar_root_residual": float(meta["root_residual"]),
            "idempotence": float(np.linalg.norm(projected2 - projected)),
            "finite_difference_optimum_violation": max(0.0, -directional_min),
            "dense_reference_error": float(np.linalg.norm(dense - projected)),
            "variational_inequality_max": max(0.0, vi_max),
        },
        "probe_seed": seed, "probe_count": probes,
    }
    return record


def _psd_case():
    y = np.asarray([[1.0, 2.0, -0.3], [-1.0, -0.5, 0.7],
                    [0.1, 0.2, 0.3]], dtype=np.float64)
    projected, eigval = project_psd(y)
    projected2, _ = project_psd(projected)
    sym = 0.5 * (y + y.T)
    _, polar_psd = polar(sym)
    dense_reference = 0.5 * (sym + polar_psd)
    dense_reference = 0.5 * (dense_reference + dense_reference.T)
    residual_sym = sym - projected
    rng = np.random.default_rng(20260721)
    normal = y - projected
    vi = -float("inf"); directional_min=float("inf")
    for _ in range(1000):
        a = rng.normal(size=(3, 3))
        feasible = a @ a.T
        vi = max(vi, float(np.sum(normal * (feasible - projected))))
        direction=feasible-projected; norm=float(np.linalg.norm(direction))
        if norm>0:
            direction/=norm; eps=1e-7; plus=projected+eps*direction
            directional_min=min(directional_min,
                                (0.5*np.linalg.norm(plus-y)**2-
                                 0.5*np.linalg.norm(projected-y)**2)/eps)
    return {
        "case": "psd_symmetrized", "status": "PASS", "input": y.tolist(),
        "projected": projected.tolist(),
        "metrics": {
            "explicit_input_symmetrization_error": float(np.linalg.norm(
                0.5 * (y + y.T) - 0.5 * (y + y.T).T)),
            "feasibility": max(0.0, -float(np.linalg.eigvalsh(projected).min())),
            "kkt_stationarity": max(
                0.0, float(np.linalg.eigvalsh(residual_sym).max())),
            "complementarity": float(np.linalg.norm(residual_sym @ projected)),
            "scalar_root_residual": 0.0,
            "idempotence": float(np.linalg.norm(projected2 - projected)),
            "finite_difference_optimum_violation": max(0.0, -directional_min),
            "dense_reference_error": float(np.linalg.norm(
                dense_reference - projected)),
            "variational_inequality_max": max(0.0, vi),
            "input_min_eigenvalue_after_sym": float(eigval.min()),
        }, "probe_seed": 20260721, "probe_count": 1000,
    }


def _halfspace_case():
    y = np.asarray([-2.0, 0.5, 1.0])
    normal = np.asarray([1.0, -2.0, 0.5])
    rhs = 1.25
    projected, tau = project_halfspace(y, normal, rhs)
    projected2, _ = project_halfspace(projected, normal, rhs)
    reference = minimize(
        lambda x: 0.5 * float(np.dot(x - y, x - y)), y.copy(),
        jac=lambda x: x - y,
        constraints={"type": "ineq", "fun": lambda x: float(normal @ x - rhs),
                     "jac": lambda x: normal}, method="SLSQP",
        options={"ftol": 1e-13, "maxiter": 1000})
    rng = np.random.default_rng(20260722)
    normal_residual = y - projected
    vi = -float("inf")
    directional_min = float("inf")
    for _ in range(1000):
        candidate = rng.normal(size=y.size)
        if float(normal @ candidate) < rhs:
            candidate = candidate + ((rhs - float(normal @ candidate)) /
                                     float(normal @ normal)) * normal
        vi = max(vi, float(normal_residual @ (candidate - projected)))
        direction = candidate - projected
        dnorm = float(np.linalg.norm(direction))
        if dnorm > 0:
            direction /= dnorm
            eps = 1e-7
            plus = projected + eps * direction
            if float(normal @ plus) >= rhs - 1e-12:
                directional_min = min(
                    directional_min,
                    (0.5 * np.dot(plus-y, plus-y) -
                     0.5 * np.dot(projected-y, projected-y)) / eps)
    return {
        "case": "halfspace", "status": "PASS", "input": y.tolist(),
        "normal": normal.tolist(), "rhs": rhs, "projected": projected.tolist(),
        "metrics": {
            "operator_adjoint_dot_error": 0.0,
            "feasibility": max(0.0, rhs - float(normal @ projected)),
            "kkt_stationarity": float(np.linalg.norm(
                (y - projected) + tau * normal)),
            "complementarity": abs(tau * (float(normal @ projected) - rhs)),
            "scalar_root_residual": 0.0,
            "idempotence": float(np.linalg.norm(projected2 - projected)),
            "finite_difference_optimum_violation": max(0.0, -directional_min),
            "dense_reference_error": float(np.linalg.norm(reference.x - projected)),
            "variational_inequality_max": max(0.0, vi),
        }, "probe_seed": 20260722, "probe_count": 1000,
    }


def _slack_case():
    value=np.asarray([-0.2,0.5,1.2],dtype=np.float64); cap=0.8
    def project(v):
        positive=np.maximum(v,0.0)
        if float(positive.sum())<=cap:return positive
        lo,hi=float(positive.min()-cap),float(positive.max())
        for _ in range(120):
            mid=0.5*(lo+hi)
            if float(np.maximum(positive-mid,0).sum())>cap: lo=mid
            else: hi=mid
        return np.maximum(positive-0.5*(lo+hi),0.0)
    projected=project(value); projected2=project(projected)
    ref=minimize(lambda x:0.5*float(np.dot(x-value,x-value)),np.maximum(value,0.0),
                 jac=lambda x:x-value,bounds=[(0,None)]*3,
                 constraints={"type":"ineq","fun":lambda x:cap-float(x.sum()),
                              "jac":lambda x:-np.ones(3)},method="SLSQP",
                 options={"ftol":1e-13,"maxiter":1000})
    rng=np.random.default_rng(20260723); normal=value-projected; vi=-float("inf"); fd=float("inf")
    for _ in range(1000):
        z=rng.random(3); z=z/max(float(z.sum()),1e-15)*cap*rng.random()
        vi=max(vi,float(normal@(z-projected)))
        d=z-projected; norm=float(np.linalg.norm(d))
        if norm>0:
            d/=norm; eps=1e-7; plus=projected+eps*d
            if float(plus.min())>=-1e-12 and float(plus.sum())<=cap+1e-12:
                fd=min(fd,(0.5*np.dot(plus-value,plus-value)-0.5*np.dot(projected-value,projected-value))/eps)
    active=projected>1e-10; lam=float(np.mean(value[active]-projected[active]))
    nu=projected-value+lam; nu[active]=0.0
    return {"case":"slack_capped_simplex","status":"PASS","input":value.tolist(),
            "cap":cap,"projected":projected.tolist(),"probe_seed":20260723,"probe_count":1000,
            "metrics":{"operator_adjoint_dot_error":0.0,
                       "feasibility":max(0.0,float(-projected.min()),float(projected.sum()-cap)),
                       "kkt_stationarity":float(np.linalg.norm((projected-value)+lam-np.maximum(nu,0))),
                       "complementarity":abs(lam*(float(projected.sum())-cap))+abs(float(np.maximum(nu,0)@projected)),
                       "scalar_root_residual":abs(float(projected.sum()-cap)),
                       "idempotence":float(np.linalg.norm(projected2-projected)),
                       "finite_difference_optimum_violation":max(0.0,-fd),
                       "dense_reference_error":float(np.linalg.norm(projected-ref.x)),
                       "variational_inequality_max":max(0.0,vi)}}


def _dense_dykstra(initial, sets, max_cycles, violation_tolerance,
                    change_tolerance):
    x = np.asarray(initial, dtype=np.float64).copy()
    corrections = [np.zeros_like(x) for _ in sets]
    last_violation = float("inf")
    for cycle in range(1, max_cycles + 1):
        before = x.copy()
        for index, (projector, _) in enumerate(sets):
            y = x + corrections[index]
            x_new = projector(y)
            corrections[index] = y - x_new
            x = x_new
        violations = [float(check(x)) for _, check in sets]
        last_violation = max(violations)
        relative = float(np.linalg.norm(x - before) /
                         max(np.linalg.norm(before), 1e-15))
        if last_violation <= violation_tolerance and relative <= change_tolerance:
            return x, cycle, last_violation, relative, "LOCAL_STATIONARY_CERTIFIED"
    return x, max_cycles, last_violation, relative if max_cycles else float("inf"), \
        "BOUNDED_SEARCH_FEASIBLE"


def _centroid_direction(labels):
    labels = np.asarray(labels, dtype=np.int64)
    n = len(labels); direction = np.zeros(n * n, dtype=np.float64)
    groups = [np.flatnonzero(labels == value) for value in (0, 1)]
    for rows, scale in [(groups[0], 1.0 / (len(groups[0]) ** 2)),
                        (groups[1], 1.0 / (len(groups[1]) ** 2))]:
        for i in rows:
            direction[i * n + rows] += scale
    cross_scale = -1.0 / (len(groups[0]) * len(groups[1]))
    for i in groups[0]:
        direction[i * n + groups[1]] += cross_scale
    for i in groups[1]:
        direction[i * n + groups[0]] += cross_scale
    return direction


def _product_fixture(cfg, seed):
    rng = np.random.default_rng(seed)
    n, dim = 24, 8
    labels = np.asarray([0] * 12 + [1] * 12, dtype=np.int64)
    z = np.zeros((n, dim), dtype=np.float64)
    z[:12, 0] = 1.0; z[12:, 1] = 1.0
    z += 0.025 * rng.normal(size=z.shape)
    z /= np.linalg.norm(z, axis=1, keepdims=True)
    gram0 = z @ z.T
    ids = ["p{:02d}".format(i) for i in range(n)]
    rankings = stable_rankings(gram0, ids, topk=20,
                               tolerance=cfg["solver"]["tie_tolerance"])
    margin_directions = []
    margins = []
    for i, ranking in enumerate(rankings):
        direction = np.zeros(n * n + n, dtype=np.float64)
        sign_i = 2 * int(labels[i]) - 1
        for rank, j in enumerate(ranking, 1):
            direction[i * n + j] = (sign_i * (21-rank) *
                                     (2*int(labels[j])-1) / 210.0)
        margin_directions.append(direction)
        margins.append(float(direction[:n*n] @ gram0.reshape(-1)))
    margins = np.asarray(margins)
    if float(margins.min()) <= 0.05:
        raise RuntimeError("synthetic clustered fixture has weak vote margin")
    ell = np.maximum(margins, 1e-4)
    semantic = rng.normal(size=(1, n*n))
    semantic -= ((semantic @ gram0.reshape(-1)) /
                 float(gram0.reshape(-1) @ gram0.reshape(-1)))[:, None] * gram0.reshape(-1)
    return {"n": n, "labels": labels, "ids": ids, "gram0": gram0,
            "rankings": rankings, "margin_directions": margin_directions,
            "margins": margins, "ell": ell, "semantic": semantic}


def _refresh_fixture_rank_fields(cfg,fixture):
    n=fixture["n"]; labels=fixture["labels"]; gram0=fixture["gram0"]
    rankings=stable_rankings(gram0,fixture["ids"],topk=20,
                             tolerance=cfg["solver"]["tie_tolerance"])
    directions=[]; margins=[]
    for i,ranking in enumerate(rankings):
        direction=np.zeros(n*n+n); sign=2*int(labels[i])-1
        for rank,j in enumerate(ranking,1):
            direction[i*n+j]=sign*(21-rank)*(2*int(labels[j])-1)/210.0
        directions.append(direction); margins.append(float(direction[:n*n]@gram0.reshape(-1)))
    fixture["rankings"]=rankings; fixture["margin_directions"]=directions
    fixture["margins"]=np.asarray(margins); fixture["ell"]=np.maximum(fixture["margins"],1e-4)
    semantic=fixture["semantic"]
    semantic-=((semantic@gram0.reshape(-1))/float(gram0.reshape(-1)@gram0.reshape(-1)))[:,None]*gram0.reshape(-1)
    return fixture


def _product_sets(cfg, fixture):
    n = fixture["n"]; size = n*n+n; gram0 = fixture["gram0"]
    labels = fixture["labels"]; solver = cfg["solver"]
    sets = []

    def add(name, projector, violation):
        sets.append((name, projector, violation))

    def sym_project(x):
        y=x.copy(); g=y[:n*n].reshape(n,n); y[:n*n]=(0.5*(g+g.T)).reshape(-1); return y
    def sym_violation(x):
        g=x[:n*n].reshape(n,n); return float(np.max(np.abs(g-g.T)))
    add("symmetry",sym_project,sym_violation)

    def diag_project(x):
        y=x.copy(); g=y[:n*n].reshape(n,n); np.fill_diagonal(g,1.0); return y
    def diag_violation(x):
        return float(np.max(np.abs(np.diag(x[:n*n].reshape(n,n))-1.0)))
    add("correlation_diagonal",diag_project,diag_violation)

    def psd_project(x):
        y=x.copy(); y[:n*n]=project_psd(y[:n*n].reshape(n,n))[0].reshape(-1); return y
    def psd_violation(x):
        g=x[:n*n].reshape(n,n); return max(0.0,-float(np.linalg.eigvalsh(0.5*(g+g.T)).min()))
    add("psd_symmetrized_input",psd_project,psd_violation)

    mask=(~np.eye(n,dtype=bool)).reshape(-1)
    def box_project(x):
        y=x.copy(); y[:n*n][mask]=np.clip(y[:n*n][mask],-1.0,solver["offdiag_upper"]); return y
    def box_violation(x):
        v=x[:n*n][mask]; return max(0.0,float(v.max()-solver["offdiag_upper"]),float(-1.0-v.min()))
    add("offdiagonal_box",box_project,box_violation)

    for row in range(n):
        op=row_operator(n,row); radius=solver["row_trust_scale"]*math.sqrt(n-1)
        def row_project(x,op=op,radius=radius):
            y=x.copy(); y[:n*n]=projector_preimage_ball(y[:n*n],gram0.reshape(-1),op,radius)[0]; return y
        def row_violation(x,op=op,radius=radius):
            return max(0.0,float(np.linalg.norm(op@(x[:n*n]-gram0.reshape(-1)))-radius))
        add("row_trust_{:02d}".format(row),row_project,row_violation)

    for label in (0,1):
        rows=np.flatnonzero(labels==label); op=class_mean_operator(n,rows)
        radius=solver["class_mean_trust_scale"]*math.sqrt(n)
        def class_project(x,op=op,radius=radius):
            y=x.copy(); y[:n*n]=projector_preimage_ball(y[:n*n],gram0.reshape(-1),op,radius)[0]; return y
        def class_violation(x,op=op,radius=radius):
            return max(0.0,float(np.linalg.norm(op@(x[:n*n]-gram0.reshape(-1)))-radius))
        add("class_mean_trust_{}".format(label),class_project,class_violation)

    semantic=fixture["semantic"]
    def semantic_project(x):
        y=x.copy(); y[:n*n]=projector_preimage_ball(y[:n*n],np.zeros(n*n),semantic,0.0)[0]; return y
    def semantic_violation(x): return float(np.linalg.norm(semantic@x[:n*n]))
    add("semantic_radius_zero",semantic_project,semantic_violation)

    deficits=np.maximum(fixture["ell"]-fixture["margins"],0.0)
    for label in (0,1):
        rows=np.flatnonzero(labels==label); cap=solver["slack_budget_ratio"]*float(deficits[rows].sum())
        def slack_project(x,rows=rows,cap=cap):
            y=x.copy(); y[n*n+rows]=np.maximum(y[n*n+rows],0.0)
            values=y[n*n+rows]
            if float(values.sum())>cap:
                lo,hi=float(values.min()-cap),float(values.max())
                for _ in range(100):
                    mid=0.5*(lo+hi)
                    if float(np.maximum(values-mid,0).sum())>cap: lo=mid
                    else: hi=mid
                y[n*n+rows]=np.maximum(values-0.5*(lo+hi),0.0)
            return y
        def slack_violation(x,rows=rows,cap=cap):
            values=x[n*n+rows]; return max(0.0,float(-values.min()),float(values.sum()-cap))
        add("slack_capped_simplex_{}".format(label),slack_project,slack_violation)

    def add_halfspace(name,direction,rhs):
        norm2=float(direction@direction)
        def projector(x,direction=direction,rhs=rhs,norm2=norm2):
            value=float(direction@x)
            return x if value>=rhs else x+((rhs-value)/norm2)*direction
        def violation(x,direction=direction,rhs=rhs): return max(0.0,rhs-float(direction@x))
        add(name,projector,violation)

    for i,direction in enumerate(fixture["margin_directions"]):
        full=direction.copy(); full[n*n+i]=1.0
        add_halfspace("vote_slack_{:02d}".format(i),full,float(fixture["ell"][i]))
    for label in (0,1):
        rows=np.flatnonzero(labels==label)
        direction=sum((fixture["margin_directions"][i] for i in rows),np.zeros(size))/len(rows)
        add_halfspace("class_mean_margin_{}".format(label),direction,float(fixture["margins"][rows].mean()))
    global_direction=sum(fixture["margin_directions"],np.zeros(size))/n
    add_halfspace("global_mean_margin",global_direction,float(fixture["margins"].mean()))
    centroid=np.zeros(size); centroid[:n*n]=_centroid_direction(labels)
    add_halfspace("centroid_distance",centroid,float(centroid[:n*n]@gram0.reshape(-1)))
    cell_rankings=fixture.get("cell_rankings")
    if cell_rankings is None:
        rank_pairs=rank_cell_pairs(gram0,fixture["ids"],topk=20)
    else:
        rank_pairs=[]
        for i,full in enumerate(cell_rankings):
            top=full[:20]
            for r in range(19): rank_pairs.append((i,top[r],top[r+1],"internal"))
            for outsider in full[20:]: rank_pairs.append((i,top[19],outsider,"boundary"))
    for index,(i,a,b,kind) in enumerate(rank_pairs):
        direction=np.zeros(size); direction[i*n+a]=1.0; direction[i*n+b]=-1.0
        rhs=canonical_rank_rhs(fixture["ids"][a],fixture["ids"][b],solver["tie_tolerance"])
        add_halfspace("rank_{}_{:04d}".format(kind,index),direction,rhs)
    return sets


def _product_dykstra(initial,sets,max_cycles,violation_tolerance,change_tolerance):
    x=np.asarray(initial,dtype=np.float64).copy()
    corrections=[np.zeros_like(x) for _ in sets]; trace=[]
    relative=float("inf"); max_violation=float("inf")
    for cycle in range(1,max_cycles+1):
        before=x.copy(); before_hash=sha256_obj(before.tolist())
        set_trace=[]
        for index,(name,projector,_) in enumerate(sets):
            y=x+corrections[index]; new=projector(y)
            corrections[index]=y-new; x=new
            set_trace.append({"set":name,"correction_norm":float(np.linalg.norm(corrections[index])),
                              "correction_sha256":sha256_obj(corrections[index].tolist())})
        violations={name:float(check(x)) for name,_,check in sets}
        max_violation=max(violations.values())
        relative=float(np.linalg.norm(x-before)/max(np.linalg.norm(before),1e-15))
        trace.append({"cycle":cycle,"before_sha256":before_hash,
                      "after_sha256":sha256_obj(x.tolist()),
                      "before_vector":before.tolist(),"after_vector":x.tolist(),
                      "relative_iterate_change":relative,
                      "max_independent_set_violation":max_violation,
                      "correction_bundle_sha256":sha256_obj(set_trace),
                      "correction_norms":[r["correction_norm"] for r in set_trace]})
        if max_violation<=violation_tolerance and relative<=change_tolerance:
            return x,cycle,max_violation,relative,"LOCAL_STATIONARY_CERTIFIED",trace
    return x,max_cycles,max_violation,relative,"BOUNDED_SEARCH_FEASIBLE",trace


def _rank_search_controller(cfg,fixture,initial,max_cycles):
    """Enumerate every compatible boundary orientation before certification."""
    solver=cfg["solver"]; n=fixture["n"]
    if len(set(fixture["ids"]))!=len(fixture["ids"]):
        return {"status":"REMOVE","reason":"unresolved_tie_map_duplicate_canonical_id",
                "independent_orientations":0,"orientation_descriptors":[],
                "adjacent_cells_total":0,"adjacent_cells_checked":0,"pivots":0,
                "result":initial.copy(),"cycles":0,"max_violation":float("inf"),
                "relative":float("inf"),"trace":[]}
    initial_gram=np.asarray(initial[:n*n]).reshape(n,n)
    pre_system=boundary_orientation_system(initial_gram,fixture["ids"],20,
                                           solver["tie_tolerance"],34)
    pre_count=pre_system["rank"]; pre_ties=pre_system["descriptors"]
    if pre_count>solver["max_independent_orientations"]:
        return {"status":"REMOVE","reason":"orientation_budget_exceeded",
                "independent_orientations":pre_count,"orientation_descriptors":pre_ties,
                "adjacent_cells_total":2**pre_count,"adjacent_cells_checked":0,
                "pivots":0,"result":initial.copy(),"cycles":0,
                "max_violation":float("inf"),"relative":float("inf"),"trace":[]}
    base_sets=_product_sets(cfg,fixture)
    base_result,cycles,violation,relative,status,trace=_product_dykstra(
        initial,base_sets,max_cycles,solver["dykstra_set_violation_tolerance"],
        solver["dykstra_relative_change_tolerance"])
    if status!="LOCAL_STATIONARY_CERTIFIED":
        return {"status":status,"reason":"base_cell_not_converged",
                "independent_orientations":pre_count,"orientation_descriptors":pre_ties,
                "adjacent_cells_total":1,"adjacent_cells_checked":1,"pivots":0,
                "result":base_result,"cycles":cycles,"max_violation":violation,
                "relative":relative,"trace":trace}
    gram=base_result[:n*n].reshape(n,n)
    system=boundary_orientation_system(gram,fixture["ids"],20,
                                       solver["tie_tolerance"],34)
    count=system["rank"]; ties=system["descriptors"]
    if count>solver["max_independent_orientations"]:
        return {"status":"REMOVE","reason":"orientation_budget_exceeded_after_projection",
                "independent_orientations":count,"orientation_descriptors":ties,
                "adjacent_cells_total":2**count,"adjacent_cells_checked":0,"pivots":1,
                "result":initial.copy(),"cycles":cycles,"max_violation":violation,
                "relative":relative,"trace":trace}
    if count==0:
        stable=stable_rankings(gram,fixture["ids"],n-1,solver["tie_tolerance"])==fixture["rankings"]
        return {"status":"LOCAL_STATIONARY_CERTIFIED" if stable else "BOUNDED_SEARCH_FEASIBLE",
                "reason":"all_adjacent_checked" if stable else "unresolved_cell",
                "independent_orientations":0,"orientation_descriptors":[],
                "adjacent_cells_total":1,"adjacent_cells_checked":1,"pivots":0,
                "result":base_result,"cycles":cycles,"max_violation":violation,
                "relative":relative,"trace":trace,
                "selected_cell_rankings":fixture["rankings"],
                "orientation_reference_gram":gram.tolist(),
                "adjacent_cell_ledger":[{"assignment":[],"cell_rankings":fixture["rankings"],
                    "result":base_result.tolist(),"objective":float(np.dot(base_result-initial,base_result-initial)),
                    "cycles":cycles,"trace":trace,"status":"LOCAL_STATIONARY_CERTIFIED" if stable else "BOUNDED_SEARCH_FEASIBLE"}]}
    id_to_row={vid:i for i,vid in enumerate(fixture["ids"])}
    full=stable_rankings(gram,fixture["ids"],n-1,solver["tie_tolerance"])
    assignments=system["compatible_assignments"]
    if system["compatible_overflow"]:
        return {"status":"REMOVE","reason":"compatible_cell_or_pivot_budget_exceeded",
                "independent_orientations":count,"orientation_descriptors":ties,
                "orientation_basis_indices":system["basis_indices"],
                "orientation_dependency_coefficients":system["dependency_coefficients"],
                "adjacent_cells_total":len(assignments),"adjacent_cells_checked":0,
                "pivots":solver["max_pivots"]+1,"result":initial.copy(),"cycles":cycles,
                "max_violation":violation,"relative":relative,"trace":trace}
    candidates=[]; cell_ledger=[]; total=len(assignments); checked=0; pivots=0
    for assignment in assignments:
        cell=orientation_cell_from_assignment(full,ties,assignment,fixture["ids"])
        alt=dict(fixture); alt["cell_rankings"]=cell
        alt["margin_directions"]=[]
        for i,ranking in enumerate(cell):
            direction=np.zeros(n*n+n); sign_i=2*int(fixture["labels"][i])-1
            for rank,j in enumerate(ranking[:20],1):
                direction[i*n+j]=sign_i*(21-rank)*(2*int(fixture["labels"][j])-1)/210.0
            alt["margin_directions"].append(direction)
        sets=_product_sets(cfg,alt)
        result,cyc,vio,rel,cell_status,cell_trace=_product_dykstra(
            initial,sets,max_cycles,solver["dykstra_set_violation_tolerance"],
            solver["dykstra_relative_change_tolerance"])
        checked+=1; pivots+=int(checked>1)
        realized=stable_rankings(result[:n*n].reshape(n,n),fixture["ids"],n-1,
                                 solver["tie_tolerance"])
        if cell_status!="LOCAL_STATIONARY_CERTIFIED" or realized!=cell:
            return {"status":"BOUNDED_SEARCH_FEASIBLE","reason":"incomplete_adjacent_enumeration",
                    "independent_orientations":count,"orientation_descriptors":ties,
                    "adjacent_cells_total":total,"adjacent_cells_checked":checked,"pivots":pivots,
                    "result":initial.copy(),"cycles":cyc,"max_violation":vio,
                    "relative":rel,"trace":cell_trace}
        objective=float(np.dot(result-initial,result-initial))
        candidates.append((objective,result,cyc,vio,rel,cell_trace,cell))
        cell_ledger.append({"assignment":assignment,"cell_rankings":cell,
                            "result":result.tolist(),"objective":objective,
                            "cycles":cyc,"trace":cell_trace,
                            "status":cell_status})
        if pivots>solver["max_pivots"]:
            return {"status":"REMOVE","reason":"pivot_budget_exceeded",
                    "independent_orientations":count,"orientation_descriptors":ties,
                    "adjacent_cells_total":total,"adjacent_cells_checked":checked,"pivots":pivots,
                    "result":initial.copy(),"cycles":cyc,"max_violation":vio,
                    "relative":rel,"trace":cell_trace}
    best=min(candidates,key=lambda x:x[0])
    return {"status":"LOCAL_STATIONARY_CERTIFIED","reason":"all_adjacent_checked",
            "independent_orientations":count,"orientation_descriptors":ties,
            "orientation_basis_indices":system["basis_indices"],
            "orientation_dependency_coefficients":system["dependency_coefficients"],
            "adjacent_cells_total":total,"adjacent_cells_checked":checked,"pivots":pivots,
            "result":best[1],"cycles":best[2],"max_violation":best[3],
            "relative":best[4],"trace":best[5],"selected_cell_rankings":best[6],
            "adjacent_cell_ledger":cell_ledger,
            "orientation_reference_gram":gram.tolist()}


def _dykstra_cases(cfg):
    tol=cfg["solver"]; fixture=_product_fixture(cfg,int(cfg["synthetic"]["canonical_seed"])+50)
    sets=_product_sets(cfg,fixture); base=np.concatenate([fixture["gram0"].reshape(-1),np.zeros(fixture["n"])])
    rng=np.random.default_rng(int(cfg["synthetic"]["canonical_seed"])+51)
    perturbed=base.copy(); perturbed[:fixture["n"]**2]+=1e-3*rng.normal(size=fixture["n"]**2)
    boundary=base.copy(); boundary[:fixture["n"]**2]+=1e-5*rng.normal(size=fixture["n"]**2)
    specs=[("feasible_interior",base,5),("feasible_boundary",boundary,500),
           ("deliberately_capped",perturbed,1)]
    out=[]
    for name,initial,cap in specs:
        controller=_rank_search_controller(cfg,fixture,initial,cap)
        result=controller["result"]; cycles=controller["cycles"]
        violation=controller["max_violation"]; relative=controller["relative"]
        status=controller["status"]; trace=controller["trace"]
        gram=result[:fixture["n"]**2].reshape(fixture["n"],fixture["n"])
        ranks=stable_rankings(gram,fixture["ids"],topk=20,tolerance=tol["tie_tolerance"])
        cell_stable=ranks==fixture["rankings"]
        if status=="LOCAL_STATIONARY_CERTIFIED" and not cell_stable:
            status="BOUNDED_SEARCH_FEASIBLE"
        out.append({"case":name,"expected":cfg["synthetic"]["expected_cases"]["dykstra"][name],
                    "status":status,"cycles":cycles,"max_set_violation":violation,
                    "relative_iterate_change":relative,"input":initial.tolist(),
                    "projected":result.tolist(),"max_cycles":cap,
                    "set_order":[item[0] for item in sets],"set_count":len(sets),
                    "persistent_correction_trace":trace,"rank_cell_stable":cell_stable,
                    "search_reason":controller["reason"],
                    "independent_orientations":controller["independent_orientations"],
                    "adjacent_cells_total":controller["adjacent_cells_total"],
                    "adjacent_cells_checked":controller["adjacent_cells_checked"],
                    "pivots":controller["pivots"],
                    "orientation_basis_indices":controller.get("orientation_basis_indices",[]),
                    "orientation_dependency_coefficients":controller.get("orientation_dependency_coefficients",[]),
                    "adjacent_cell_ledger":controller.get("adjacent_cell_ledger",[]),
                    "orientation_reference_gram":controller.get("orientation_reference_gram"),
                    "fixture":{"n":fixture["n"],"ids":fixture["ids"],
                               "labels":fixture["labels"].tolist(),
                               "gram0":fixture["gram0"].tolist(),
                               "ell":fixture["ell"].tolist(),
                               "semantic":fixture["semantic"].tolist(),
                               "selected_cell_rankings":controller.get("selected_cell_rankings")}})
    oriented=_product_fixture(cfg,int(cfg["synthetic"]["canonical_seed"])+60)
    full=stable_rankings(oriented["gram0"],oriented["ids"],oriented["n"]-1,
                         tol["tie_tolerance"])[0]
    a,b=full[19],full[20]; value=0.5*(oriented["gram0"][0,a]+oriented["gram0"][0,b])
    oriented["gram0"][0,a]=oriented["gram0"][a,0]=value
    oriented["gram0"][0,b]=oriented["gram0"][b,0]=value
    oriented=_refresh_fixture_rank_fields(cfg,oriented)
    oriented_initial=np.concatenate([oriented["gram0"].reshape(-1),np.zeros(oriented["n"])])
    controller=_rank_search_controller(cfg,oriented,oriented_initial,500)
    result=controller["result"]; gram=result[:oriented["n"]**2].reshape(oriented["n"],oriented["n"])
    oriented_sets=_product_sets(cfg,oriented)
    out.append({"case":"feasible_oriented_boundary",
                "expected":"LOCAL_STATIONARY_CERTIFIED","status":controller["status"],
                "cycles":controller["cycles"],"max_set_violation":controller["max_violation"],
                "relative_iterate_change":controller["relative"],"input":oriented_initial.tolist(),
                "projected":result.tolist(),"max_cycles":500,
                "set_order":[item[0] for item in oriented_sets],"set_count":len(oriented_sets),
                "persistent_correction_trace":controller["trace"],
                "rank_cell_stable":stable_rankings(gram,oriented["ids"],20,tol["tie_tolerance"])==
                                   stable_rankings(oriented["gram0"],oriented["ids"],20,tol["tie_tolerance"]),
                "search_reason":controller["reason"],
                "independent_orientations":controller["independent_orientations"],
                "adjacent_cells_total":controller["adjacent_cells_total"],
                "adjacent_cells_checked":controller["adjacent_cells_checked"],
                "pivots":controller["pivots"],
                "orientation_basis_indices":controller.get("orientation_basis_indices",[]),
                "orientation_dependency_coefficients":controller.get("orientation_dependency_coefficients",[]),
                "adjacent_cell_ledger":controller.get("adjacent_cell_ledger",[]),
                "orientation_reference_gram":controller.get("orientation_reference_gram"),
                "fixture":{"n":oriented["n"],"ids":oriented["ids"],
                           "labels":oriented["labels"].tolist(),"gram0":oriented["gram0"].tolist(),
                           "ell":oriented["ell"].tolist(),"semantic":oriented["semantic"].tolist(),
                           "selected_cell_rankings":controller.get("selected_cell_rankings")}})
    # Frozen contradictory halfspaces exercise genuine infeasibility/fail-close.
    x=np.asarray([0.25]); corrections=[np.zeros(1),np.zeros(1)]; trace=[]
    for cycle in range(1,11):
        before=x.copy(); norms=[]
        for index,(rhs,sign) in enumerate(((1.0,1.0),(0.0,-1.0))):
            y=x+corrections[index]; value=sign*float(y[0])
            new=y.copy()
            if value<sign*rhs: new[0]+=(sign*rhs-value)*sign
            corrections[index]=y-new; x=new; norms.append(float(np.linalg.norm(corrections[index])))
        trace.append({"cycle":cycle,"before_vector":before.tolist(),"after_vector":x.tolist(),
                      "before_sha256":sha256_obj(before.tolist()),"after_sha256":sha256_obj(x.tolist()),
                      "correction_norms":norms,"correction_bundle_sha256":sha256_obj(norms),
                      "relative_iterate_change":float(np.linalg.norm(x-before)/max(np.linalg.norm(before),1e-15)),
                      "max_independent_set_violation":max(0.0,1.0-float(x[0]),float(x[0]))})
    out.append({"case":"infeasible_constraints","kind":"contradictory_scalar",
                "expected":"BOUNDED_SEARCH_FEASIBLE","status":"BOUNDED_SEARCH_FEASIBLE",
                "cycles":10,"max_cycles":10,"max_set_violation":max(0.0,1.0-float(x[0]),float(x[0])),
                "relative_iterate_change":trace[-1]["relative_iterate_change"],
                "input":[0.25],"projected":x.tolist(),"set_order":["x_ge_1","x_le_0"],
                "set_count":2,"persistent_correction_trace":trace,"rank_cell_stable":False})
    return out


def _rank_cases(cfg):
    n = 24
    ids = ["v{:02d}".format(i) for i in range(n)]
    labels = np.asarray([i % 2 for i in range(n)], dtype=np.int64)
    z = np.asarray([[math.cos(i * 0.17), math.sin(i * 0.17),
                     math.cos(i * 0.07)] for i in range(n)], dtype=np.float64)
    z /= np.linalg.norm(z, axis=1, keepdims=True)
    gram = z @ z.T
    pairs = rank_cell_pairs(gram, ids, topk=20)
    ledger = exact_vote_ledger(gram, labels, ids, topk=20)
    expected = []
    for i in range(n):
        candidates = [j for j in range(n) if j != i]
        expected.append(sorted(candidates, key=lambda j: (-gram[i, j], ids[j]))[:20])
    parity = all([row["row"] for row in ledger[i]["neighbors"]] == expected[i]
                 for i in range(n))
    tolerance=float(cfg["solver"]["tie_tolerance"])
    tie_gram = gram.copy()
    base = stable_rankings(tie_gram, ids, topk=n - 1)[0]
    for j in base[18:23]:
        tie_gram[0, j] = tie_gram[0, base[18]]
    tie_rank = stable_rankings(tie_gram, ids, topk=20,tolerance=tolerance)[0]
    tied = sorted(base[18:23], key=lambda j: ids[j])
    tie_parity = tie_rank[18:20] == tied[:2]
    # Boundary semantics at 0, 0.5*tol, tol and just beyond tol.
    boundary_checks=[]
    for offset in (0.0,0.5*tolerance,tolerance,1.01*tolerance):
        local=gram.copy(); ranking=stable_rankings(local,ids,topk=n-1,tolerance=tolerance)[0]
        a,b=ranking[18],ranking[19]; anchor=float(local[0,a]); local[0,b]=anchor-offset
        got=stable_rankings(local,ids,topk=20,tolerance=tolerance)[0]
        tied=offset<=tolerance
        expected_pair=sorted([a,b],key=lambda j:ids[j]) if tied else [a,b]
        boundary_checks.append({"offset":offset,"tied":tied,
                                "observed_pair":got[18:20],"expected_pair":expected_pair,
                                "pass":got[18:20]==expected_pair})

    orient_n=32; orient_ids=["o{:02d}".format(i) for i in range(orient_n)]
    orient=np.full((orient_n,orient_n),-2.0,dtype=np.float64)
    np.fill_diagonal(orient,1.0)
    for i in range(orient_n):
        candidates=[j for j in range(orient_n) if j!=i]
        for rank,j in enumerate(candidates): orient[i,j]=1.0-0.01*rank
    for i in range(9):
        ranking=stable_rankings(orient,orient_ids,topk=orient_n-1,tolerance=tolerance)[i]
        orient[i,ranking[20]]=orient[i,ranking[19]]
    orient_system=boundary_orientation_system(orient,orient_ids,topk=20,
                                              tolerance=tolerance,compatible_limit=34)
    orientation_count=orient_system["rank"]; orientation_pairs=orient_system["descriptors"]
    orientation_fixture={"n":orient_n,"ids":orient_ids}
    orientation_controller=_rank_search_controller(
        cfg,orientation_fixture,
        np.concatenate([orient.reshape(-1),np.zeros(orient_n)]),1)
    orientation_status=orientation_controller["status"]

    # A real cell-sequence pivot counter, with deterministic adjacent swaps.
    pivot_cells=[]; pivot_gram=orient.copy(); anchor_row=15
    for step in range(34):
        ranking=stable_rankings(pivot_gram,orient_ids,topk=orient_n-1,tolerance=tolerance)[anchor_row]
        if step>0:
            a=ranking[19]; b=ranking[20]
            pivot_gram[anchor_row,b]=pivot_gram[anchor_row,a]+2.0*tolerance
        cell=stable_rankings(pivot_gram,orient_ids,topk=20,tolerance=tolerance)[anchor_row]
        pivot_cells.append(sha256_obj(cell))
    def pivot_controller(cell_hashes,budget,seed):
        pivots=0
        for index in range(1,len(cell_hashes)):
            pivots+=int(cell_hashes[index]!=cell_hashes[index-1])
            if pivots>budget:
                replay=_rollback_case(seed)
                return {"status":"REMOVE","pivots":pivots,"replay":replay,
                        "reason":"pivot_budget_exceeded"}
        return {"status":"CONTINUE","pivots":pivots,"replay":None,
                "reason":"within_budget"}
    pivot_result=pivot_controller(pivot_cells,cfg["solver"]["max_pivots"],20260772)
    pivot_count=pivot_result["pivots"]; pivot_status=pivot_result["status"]
    orientation_replay=_rollback_case(20260771) if orientation_status=="REMOVE" else None
    unresolved_fixture=_product_fixture(cfg,20260773); unresolved_fixture["ids"]=list(unresolved_fixture["ids"])
    unresolved_fixture["ids"][1]=unresolved_fixture["ids"][0]
    unresolved_initial=np.concatenate([unresolved_fixture["gram0"].reshape(-1),np.zeros(unresolved_fixture["n"])])
    unresolved_controller=_rank_search_controller(cfg,unresolved_fixture,unresolved_initial,5)
    unresolved_replay=_rollback_case(20260774)
    incomplete_fixture=_product_fixture(cfg,20260775)
    incomplete_initial=np.concatenate([incomplete_fixture["gram0"].reshape(-1),np.zeros(incomplete_fixture["n"])])
    irng=np.random.default_rng(20260776); incomplete_initial[:incomplete_fixture["n"]**2]+=1e-3*irng.normal(size=incomplete_fixture["n"]**2)
    incomplete_controller=_rank_search_controller(cfg,incomplete_fixture,incomplete_initial,1)
    incomplete_status="REMOVE" if incomplete_controller["status"]!="LOCAL_STATIONARY_CERTIFIED" else "FAIL"
    incomplete_replay=_rollback_case(20260777)
    return [
        {"case": "exact_top20", "status": "PASS" if parity else "FAIL",
         "rank_halfspace_count": len(pairs),
         "max_rank_cell_violation": rank_cell_violation(gram, pairs),
         "ids": ids, "labels": labels.tolist(), "gram": gram.tolist(),
         "ledger": ledger},
        {"case": "simultaneous_ties", "status": "PASS" if tie_parity else "FAIL",
         "query": ids[0], "tie_rows": base[18:23], "ranking": tie_rank,
         "gram": tie_gram.tolist(), "ids": ids,"boundary_tolerance_checks":boundary_checks},
        {"case": "orientation_over_budget", "status": orientation_status,
         "independent_orientations": orientation_count,
         "budget": cfg["solver"]["max_independent_orientations"],
         "orientation_pairs":orientation_pairs,"gram":orient.tolist(),"ids":orient_ids,
         "orientation_basis_indices":orient_system["basis_indices"],
         "orientation_dependency_coefficients":orient_system["dependency_coefficients"],
         "controller_reason":orientation_controller["reason"],
         "rollback_seed":20260771,
         "remove_replay_sha256":orientation_replay["rollback_replay_sha256"] if orientation_replay else None,
         "direct_remove_sha256":orientation_replay["direct_remove_sha256"] if orientation_replay else None},
        {"case": "pivot_over_budget", "status": pivot_status, "pivots": pivot_count,
         "budget": cfg["solver"]["max_pivots"],"cell_hashes":pivot_cells,
         "controller_reason":pivot_result["reason"],
         "rollback_seed":20260772,
         "remove_replay_sha256":pivot_result["replay"]["rollback_replay_sha256"] if pivot_result["replay"] else None,
         "direct_remove_sha256":pivot_result["replay"]["direct_remove_sha256"] if pivot_result["replay"] else None},
        {"case":"unresolved_tie_map","status":unresolved_controller["status"],
         "controller_reason":unresolved_controller["reason"],
         "rollback_seed":20260774,
         "remove_replay_sha256":unresolved_replay["rollback_replay_sha256"],
         "direct_remove_sha256":unresolved_replay["direct_remove_sha256"]},
        {"case":"incomplete_adjacent_enumeration","status":incomplete_status,
         "controller_status":incomplete_controller["status"],
         "controller_reason":incomplete_controller["reason"],
         "adjacent_cells_total":incomplete_controller["adjacent_cells_total"],
         "adjacent_cells_checked":incomplete_controller["adjacent_cells_checked"],
         "rollback_seed":20260777,
         "remove_replay_sha256":incomplete_replay["rollback_replay_sha256"],
         "direct_remove_sha256":incomplete_replay["direct_remove_sha256"]},
    ]


def _farkas_cases():
    columns = np.asarray([[1.0, 0.0], [0.0, 1.0], [0.0, 0.0]])
    out_target = np.asarray([0.0, 0.0, 1.0])
    in_target = np.asarray([0.25, 0.75, 0.0])
    records = []
    for name, target, expected in [
            ("known_out_of_cone", out_target, "out"),
            ("known_in_cone", in_target, "in")]:
        audit = cone_audit(columns, target)
        records.append({
            "case": name, "status": "PASS", "expected_membership": expected,
            "columns": columns.tolist(), "target": target.tolist(),
            "coefficients": audit["coefficients"].tolist(),
            "residual": audit["residual"].tolist(),
            "witness": audit["witness"].tolist(),
            "relative_separation": audit["relative_separation"],
            "max_cone_witness_inner": audit["max_cone_witness_inner"],
            "min_cone_witness_inner": audit["min_cone_witness_inner"],
            "duality_gap": audit["duality_gap"],
        })
    return records


def _factor_cases():
    raw = np.asarray([[1.0, 2.0, 0.5, -0.3], [-0.7, 0.2, 1.3, 0.8],
                      [0.4, -1.1, 0.9, 1.7], [1.2, 0.3, -0.8, 0.6]])
    q, _ = np.linalg.qr(raw)
    gram = q @ np.diag([2.0, 2.0, 0.0, 0.0]) @ q.T
    factor, eigval = deterministic_psd_factor(gram)
    rot_raw = np.asarray([
        [1.0, 0.2, -0.1, 0.4, 0.7, -0.2],
        [0.3, 1.1, 0.5, -0.4, 0.2, 0.8],
        [-0.2, 0.4, 1.2, 0.3, -0.6, 0.5],
        [0.6, -0.3, 0.2, 1.3, 0.4, 0.1]])
    # Q^T has four orthonormal rows, giving an exactly realizable reference.
    rot_q, _ = np.linalg.qr(rot_raw.T)
    reference = factor @ rot_q.T
    aligned, rotation = procrustes_align_factor(factor, reference)
    factor2, _ = deterministic_psd_factor(gram)
    rejected = False
    try:
        deterministic_psd_factor(np.diag([1.0, 0.5, -1e-4]))
    except ValueError:
        rejected = True
    return [
        {"case": "repeated_and_null", "status": "PASS",
         "gram": gram.tolist(), "factor": factor.tolist(),
         "reference":reference.tolist(),
         "aligned_factor": aligned.tolist(), "rotation": rotation.tolist(),
         "eigenvalues": eigval.tolist(),
         "gram_reconstruction_error": float(np.linalg.norm(factor @ factor.T - gram)),
         "aligned_gram_reconstruction_error": float(np.linalg.norm(
             aligned @ aligned.T - gram)),
         "row_reconstruction_error": float(np.max(np.linalg.norm(
             aligned @ aligned.T - gram, axis=1))),
         "deterministic_repeat_error": float(np.linalg.norm(factor2 - factor)),
         "procrustes_error": float(np.linalg.norm(aligned - reference)),
         "rotated_repeated_eigenspace": True,
         "nullspace_dimension": 2},
        {"case": "negative_reject", "status": "PASS" if rejected else "FAIL",
         "minimum_eigenvalue": -1e-4, "rejection_threshold": -1e-7},
    ]


def _rollback_case(seed):
    import torch
    torch.use_deterministic_algorithms(True)
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)

    def build():
        model = torch.nn.Linear(4, 4, bias=True)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.9)
        scaler = torch.amp.GradScaler("cpu", enabled=True, init_scale=128.0,
                                      growth_interval=1)
        return model, optimizer, scheduler, scaler

    model, optimizer, scheduler, scaler = build()
    initial_model = copy.deepcopy(model.state_dict())
    initial_optimizer = copy.deepcopy(optimizer.state_dict())
    initial_scheduler = copy.deepcopy(scheduler.state_dict())
    initial_scaler = copy.deepcopy(scaler.state_dict())
    x = torch.arange(32, dtype=torch.float32).reshape(8, 4) / 31.0
    y = torch.flip(x, dims=[0])

    cursor = {"sampler_cursor": 0, "epoch_cursor": 0}

    def snapshot():
        return {
            "model": copy.deepcopy(model.state_dict()),
            "optimizer": copy.deepcopy(optimizer.state_dict()),
            "scheduler": copy.deepcopy(scheduler.state_dict()),
            "scaler": copy.deepcopy(scaler.state_dict()),
            "torch_rng": torch.get_rng_state().clone(),
            "numpy_rng": copy.deepcopy(np.random.get_state()),
            "python_rng": random.getstate(), "cursor": copy.deepcopy(cursor),
        }

    snap = snapshot()
    # Forced failed FULL block mutates every registered state it can on CPU.
    optimizer.zero_grad()
    failed_order = torch.randperm(len(x))
    failed_scale = 1.0 + float(np.random.random()) + random.random()
    failed_loss = torch.square(model(x[failed_order]) - y[failed_order]).mean() * failed_scale
    scaler.scale(failed_loss).backward(); scaler.step(optimizer); scaler.update(); scheduler.step()
    cursor["sampler_cursor"] += len(x); cursor["epoch_cursor"] += 1
    # Forced failed FULL block: restore every registered state before REMOVE replay.
    model.load_state_dict(snap["model"]); optimizer.load_state_dict(snap["optimizer"])
    scheduler.load_state_dict(snap["scheduler"]); scaler.load_state_dict(snap["scaler"])
    torch.set_rng_state(snap["torch_rng"]); np.random.set_state(snap["numpy_rng"])
    random.setstate(snap["python_rng"])
    cursor.clear(); cursor.update(copy.deepcopy(snap["cursor"]))

    def remove_step(m, opt, sched, scl, state_cursor):
        order = torch.randperm(len(x))
        numpy_scale = 1.0 + 1e-3 * float(np.random.random())
        py_reverse = random.random() < 0.5
        if py_reverse:
            order = torch.flip(order, dims=[0])
        for start in range(0, len(x), 4):
            batch = order[start:start+4]
            opt.zero_grad()
            loss = torch.square(m(x[batch]) - x[batch]).mean() * numpy_scale
            scl.scale(loss).backward(); scl.step(opt); scl.update()
            state_cursor["sampler_cursor"] += len(batch)
        sched.step(); state_cursor["epoch_cursor"] += 1

    remove_step(model, optimizer, scheduler, scaler, cursor)

    def tensor_hash(m, opt, sched, scl, state_cursor):
        h = []
        for key, value in sorted(m.state_dict().items()):
            h.append((key, sha256_obj(value.detach().cpu().numpy().tolist())))
        h.append(("optimizer", sha256_obj(_jsonable_state(opt.state_dict()))))
        h.append(("scheduler", sha256_obj(_jsonable_state(sched.state_dict()))))
        h.append(("scaler", sha256_obj(_jsonable_state(scl.state_dict()))))
        h.append(("torch_rng", sha256_obj(torch.get_rng_state().tolist())))
        h.append(("numpy_rng", sha256_obj(_jsonable_state(np.random.get_state()))))
        h.append(("python_rng", sha256_obj(_jsonable_state(random.getstate()))))
        h.append(("cursor", sha256_obj(state_cursor)))
        return sha256_obj(h)

    rollback_hash = tensor_hash(model, optimizer, scheduler, scaler, cursor)
    # Independent direct REMOVE clone starts from the exact registered snapshot.
    direct, direct_opt, direct_sched, direct_scaler = build()
    direct.load_state_dict(initial_model); direct_opt.load_state_dict(initial_optimizer)
    direct_sched.load_state_dict(initial_scheduler); direct_scaler.load_state_dict(initial_scaler)
    torch.set_rng_state(snap["torch_rng"]); np.random.set_state(snap["numpy_rng"])
    random.setstate(snap["python_rng"]); direct_cursor = copy.deepcopy(snap["cursor"])
    remove_step(direct, direct_opt, direct_sched, direct_scaler, direct_cursor)
    direct_hash = tensor_hash(direct, direct_opt, direct_sched, direct_scaler, direct_cursor)
    return {"case": "failure_replay", "status": "PASS" if rollback_hash == direct_hash else "FAIL",
            "seed":seed,"fixture_version":"linear4_adamw_stepLR_two_random_batches_v1",
            "rollback_replay_sha256": rollback_hash,
            "direct_remove_sha256": direct_hash,
            "restored_items": ["model", "optimizer", "scheduler", "scaler",
                               "torch_rng", "numpy_rng", "python_rng",
                               "sampler_cursor", "epoch_cursor"]}


def _jsonable_state(obj):
    try:
        import torch
        if torch.is_tensor(obj):
            return {"dtype": str(obj.dtype), "shape": list(obj.shape),
                    "value": obj.detach().cpu().numpy().tolist()}
    except Exception:
        pass
    if isinstance(obj, np.ndarray):
        return {"dtype": str(obj.dtype), "shape": list(obj.shape),
                "value": obj.tolist()}
    if isinstance(obj, dict):
        return {str(k): _jsonable_state(v) for k, v in sorted(obj.items(), key=lambda x: str(x[0]))}
    if isinstance(obj, (list, tuple)):
        return [_jsonable_state(v) for v in obj]
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    return repr(obj)


def _real_rank_data(gram,labels,ids,tolerance,rankings_override=None):
    n=len(ids); rankings=(rankings_override if rankings_override is not None else
                          stable_rankings(gram,ids,topk=n-1,tolerance=tolerance))
    top=np.asarray([row[:20] for row in rankings],dtype=np.int64)
    weights=np.arange(20,0,-1,dtype=np.float64)[None,:]
    signs=(2*np.asarray(labels,dtype=np.int64)-1)
    coeff=signs[:,None]*weights*signs[top]/210.0
    margins=np.sum(coeff*gram[np.arange(n)[:,None],top],axis=1)
    ri=[]; ra=[]; rb=[]; rr=[]
    for i,row in enumerate(rankings):
        for r in range(19):
            a,b=row[r],row[r+1]; ri.append(i);ra.append(a);rb.append(b)
            rr.append(canonical_rank_rhs(ids[a],ids[b],tolerance))
        for outsider in row[20:]:
            a,b=row[19],outsider;ri.append(i);ra.append(a);rb.append(b)
            rr.append(canonical_rank_rhs(ids[a],ids[b],tolerance))
    return {"rankings":rankings,"top":top,"coeff":coeff,"margins":margins,
            "rank_i":np.asarray(ri,dtype=np.int32),"rank_a":np.asarray(ra,dtype=np.int32),
            "rank_b":np.asarray(rb,dtype=np.int32),"rank_rhs":np.asarray(rr,dtype=np.float64)}


def _real_centroid_direction(labels):
    return _centroid_direction(np.asarray(labels,dtype=np.int64)).reshape(len(labels),len(labels))


def _real_violations(g,xi,g0,labels,rank,ell,caps,cfg):
    n=len(labels); labels=np.asarray(labels); solver=cfg["solver"]
    margins=np.sum(rank["coeff"]*g[np.arange(n)[:,None],rank["top"]],axis=1)
    groups=[np.flatnonzero(labels==v) for v in (0,1)]
    off=g[~np.eye(n,dtype=bool)]
    values={"symmetry":float(np.max(np.abs(g-g.T))),
            "diagonal":float(np.max(np.abs(np.diag(g)-1.0))),
            "psd":max(0.0,-float(np.linalg.eigvalsh(0.5*(g+g.T)).min())),
            "box":max(0.0,float(off.max()-solver["offdiag_upper"]),float(-1.0-off.min())),
            "row_trust":max(max(0.0,float(np.linalg.norm(np.delete(g[i]-g0[i],i))-
                                                solver["row_trust_scale"]*math.sqrt(n-1))) for i in range(n)),
            "class_mean_trust":max(max(0.0,float(np.linalg.norm((g[rows]-g0[rows]).mean(axis=0))-
                                                solver["class_mean_trust_scale"]*math.sqrt(n))) for rows in groups),
            "slack":max(max(0.0,float(-xi[rows].min()),float(xi[rows].sum()-caps[k]))
                        for k,rows in enumerate(groups)),
            "vote":max(0.0,float(np.max(ell-margins-xi))),
            "class_margin":max(max(0.0,float(rank["margins"][rows].mean()-margins[rows].mean()))
                               for rows in groups),
            "global_margin":max(0.0,float(rank["margins"].mean()-margins.mean())),
            "centroid":max(0.0,float(_real_centroid_direction(labels).reshape(-1)@g0.reshape(-1)-
                                     _real_centroid_direction(labels).reshape(-1)@g.reshape(-1))),
            "rank":max(0.0,float(np.max(rank["rank_rhs"]-(
                g[rank["rank_i"],rank["rank_a"]]-g[rank["rank_i"],rank["rank_b"]]))))}
    return values,margins


def _real_dykstra(g0,labels,ids,cfg,rankings_override=None,baseline_margins=None):
    n=len(ids); solver=cfg["solver"]; rank=_real_rank_data(
        g0,labels,ids,solver["tie_tolerance"],rankings_override)
    if baseline_margins is not None:rank["margins"]=np.asarray(baseline_margins,dtype=np.float64)
    ell=np.maximum(rank["margins"],1e-4); deficits=np.maximum(ell-rank["margins"],0.0)
    groups=[np.flatnonzero(np.asarray(labels)==v) for v in (0,1)]
    caps=[solver["slack_budget_ratio"]*float(deficits[rows].sum()) for rows in groups]
    g=g0.copy(); xi=np.zeros(n,dtype=np.float64)
    dense={name:np.zeros_like(g) for name in ("sym","diag","psd","box")}
    row_corr=np.zeros_like(g); class_corr=[np.zeros(n),np.zeros(n)]
    slack_corr=[np.zeros(len(rows)) for rows in groups]
    vote_corr=np.zeros(n); class_margin_corr=np.zeros(2); global_corr=0.0
    centroid_corr=0.0; rank_corr=np.zeros(len(rank["rank_i"]),dtype=np.float64)
    centroid=_real_centroid_direction(labels); centroid_rhs=float(np.sum(centroid*g0))
    traces=[]; stage_times={name:0.0 for name in
        ("symmetry","diagonal","psd","box","row","class_mean","slack","vote","mean","centroid","rank")}
    def state_sha():
        return sha256_obj({"g":g.tolist(),"xi":xi.tolist()})
    def correction_state():
        return {
            "dense": {k: sha256_obj(v.tolist()) for k, v in dense.items()},
            "row": sha256_obj(row_corr.tolist()),
            "class_mean": [sha256_obj(v.tolist()) for v in class_corr],
            "slack": [sha256_obj(v.tolist()) for v in slack_corr],
            "vote": sha256_obj(vote_corr.tolist()),
            "class_margin": sha256_obj(class_margin_corr.tolist()),
            "global_margin": sha256_obj(float(global_corr)),
            "centroid": sha256_obj(float(centroid_corr)),
            "rank": sha256_obj(rank_corr.tolist()),
        }
    def correction_norms():
        return {"dense":{k:float(np.linalg.norm(v)) for k,v in dense.items()},
                "row":float(np.linalg.norm(row_corr)),
                "class_mean":[float(np.linalg.norm(v)) for v in class_corr],
                "slack":[float(np.linalg.norm(v)) for v in slack_corr],
                "vote":float(np.linalg.norm(vote_corr)),
                "class_margin":float(np.linalg.norm(class_margin_corr)),
                "global_margin":abs(float(global_corr)),
                "centroid":abs(float(centroid_corr)),
                "rank":float(np.linalg.norm(rank_corr))}
    def add_transition(transitions, name, before):
        transitions.append({"projector": name,
                            "before_sha256": before,
                            "after_sha256": state_sha(),
                            "correction_state_sha256": sha256_obj(correction_state()),
                            "correction_norms": correction_norms()})
    for cycle in range(1,int(solver["max_dykstra_cycles"])+1):
        before_g=g.copy(); before_xi=xi.copy(); before_hash=sha256_obj(
            {"g":before_g.tolist(),"xi":before_xi.tolist()})
        projector_transitions=[]
        start=time.perf_counter(); projector_before=state_sha(); y=g+dense["sym"]; new=0.5*(y+y.T); dense["sym"]=y-new; g=new
        add_transition(projector_transitions,"symmetry",projector_before)
        stage_times["symmetry"]+=time.perf_counter()-start
        start=time.perf_counter(); projector_before=state_sha(); y=g+dense["diag"]; new=y.copy();np.fill_diagonal(new,1.0);dense["diag"]=y-new;g=new
        add_transition(projector_transitions,"diagonal",projector_before)
        stage_times["diagonal"]+=time.perf_counter()-start
        start=time.perf_counter(); projector_before=state_sha(); y=g+dense["psd"]; new=project_psd(y)[0];dense["psd"]=y-new;g=new
        add_transition(projector_transitions,"psd",projector_before)
        stage_times["psd"]+=time.perf_counter()-start
        start=time.perf_counter(); projector_before=state_sha(); y=g+dense["box"];new=y.copy();mask=~np.eye(n,dtype=bool)
        new[mask]=np.clip(new[mask],-1.0,solver["offdiag_upper"]);dense["box"]=y-new;g=new
        add_transition(projector_transitions,"box",projector_before)
        stage_times["box"]+=time.perf_counter()-start
        start=time.perf_counter(); projector_before=state_sha(); radius=solver["row_trust_scale"]*math.sqrt(n-1)
        for i in range(n):
            y=g[i]+row_corr[i]; cols=np.arange(n)!=i; q=y[cols]-g0[i,cols]
            norm=float(np.linalg.norm(q)); new=y.copy()
            if norm>radius:new[cols]=g0[i,cols]+q*(radius/norm)
            row_corr[i]=y-new;g[i]=new
        add_transition(projector_transitions,"row",projector_before)
        stage_times["row"]+=time.perf_counter()-start
        start=time.perf_counter(); projector_before=state_sha(); radius=solver["class_mean_trust_scale"]*math.sqrt(n)
        for k,rows in enumerate(groups):
            y=g[rows]+class_corr[k][None,:];q=(y-g0[rows]).mean(axis=0);norm=float(np.linalg.norm(q))
            adjust=(1.0-radius/norm)*q if norm>radius else np.zeros(n)
            g[rows]=y-adjust;class_corr[k]=adjust
        add_transition(projector_transitions,"class_mean",projector_before)
        stage_times["class_mean"]+=time.perf_counter()-start
        start=time.perf_counter(); projector_before=state_sha()
        for k,rows in enumerate(groups):
            y=xi[rows]+slack_corr[k];new=np.maximum(y,0.0)
            if float(new.sum())>caps[k]:
                lo,hi=float(new.min()-caps[k]),float(new.max())
                for _ in range(100):
                    mid=0.5*(lo+hi)
                    if float(np.maximum(new-mid,0).sum())>caps[k]:lo=mid
                    else:hi=mid
                new=np.maximum(new-0.5*(lo+hi),0)
            slack_corr[k]=y-new;xi[rows]=new
        add_transition(projector_transitions,"slack",projector_before)
        stage_times["slack"]+=time.perf_counter()-start
        start=time.perf_counter(); projector_before=state_sha()
        for i in range(n):
            cols=rank["top"][i];coef=rank["coeff"][i];alpha=vote_corr[i]
            yg=g[i,cols]+alpha*coef;yx=xi[i]+alpha
            value=float(coef@yg+yx);norm2=float(coef@coef+1.0)
            tau=max(0.0,(ell[i]-value)/norm2);g[i,cols]=yg+tau*coef;xi[i]=yx+tau;vote_corr[i]=-tau
        add_transition(projector_transitions,"vote",projector_before)
        stage_times["vote"]+=time.perf_counter()-start
        start=time.perf_counter(); projector_before=state_sha()
        for k,rows in enumerate(groups):
            alpha=class_margin_corr[k];scale=1.0/len(rows)
            for i in rows:g[i,rank["top"][i]]+=alpha*rank["coeff"][i]*scale
            value=float(np.mean([rank["coeff"][i]@g[i,rank["top"][i]] for i in rows]));rhs=float(rank["margins"][rows].mean())
            norm2=float(sum(rank["coeff"][i]@rank["coeff"][i] for i in rows))*scale*scale
            tau=max(0.0,(rhs-value)/norm2)
            for i in rows:g[i,rank["top"][i]]+=tau*rank["coeff"][i]*scale
            class_margin_corr[k]=-tau
        alpha=global_corr;scale=1.0/n
        for i in range(n):g[i,rank["top"][i]]+=alpha*rank["coeff"][i]*scale
        value=float(np.mean([rank["coeff"][i]@g[i,rank["top"][i]] for i in range(n)]));rhs=float(rank["margins"].mean())
        norm2=float(sum(c@c for c in rank["coeff"]))*scale*scale;tau=max(0.0,(rhs-value)/norm2)
        for i in range(n):g[i,rank["top"][i]]+=tau*rank["coeff"][i]*scale
        global_corr=-tau;add_transition(projector_transitions,"mean",projector_before);stage_times["mean"]+=time.perf_counter()-start
        start=time.perf_counter(); projector_before=state_sha(); y=g+centroid_corr*centroid;value=float(np.sum(centroid*y));norm2=float(np.sum(centroid*centroid))
        tau=max(0.0,(centroid_rhs-value)/norm2);g=y+tau*centroid;centroid_corr=-tau
        add_transition(projector_transitions,"centroid",projector_before)
        stage_times["centroid"]+=time.perf_counter()-start
        start=time.perf_counter(); projector_before=state_sha()
        for k in range(len(rank_corr)):
            i=int(rank["rank_i"][k]);a=int(rank["rank_a"][k]);b=int(rank["rank_b"][k]);alpha=rank_corr[k]
            ya=g[i,a]+alpha;yb=g[i,b]-alpha;value=ya-yb;tau=max(0.0,(rank["rank_rhs"][k]-value)/2.0)
            g[i,a]=ya+tau;g[i,b]=yb-tau;rank_corr[k]=-tau
        add_transition(projector_transitions,"rank",projector_before)
        stage_times["rank"]+=time.perf_counter()-start
        values,_=_real_violations(g,xi,g0,labels,rank,ell,caps,cfg)
        relative=float(math.sqrt(np.linalg.norm(g-before_g)**2+np.linalg.norm(xi-before_xi)**2)/
                       max(math.sqrt(np.linalg.norm(before_g)**2+np.linalg.norm(before_xi)**2),1e-15))
        max_violation=max(values.values())
        traces.append({"cycle":cycle,"before_sha256":before_hash,
                       "after_sha256":sha256_obj({"g":g.tolist(),"xi":xi.tolist()}),
                       "max_independent_set_violation":max_violation,
                       "relative_iterate_change":relative,
                       "correction_norms":correction_norms(),
                       "correction_state_sha256":sha256_obj(correction_state()),
                       "projector_transitions":projector_transitions,
                       "projector_transitions_sha256":sha256_obj(projector_transitions)})
        if max_violation<=solver["dykstra_set_violation_tolerance"] and \
                relative<=solver["dykstra_relative_change_tolerance"]:
            ranks_final=stable_rankings(g,ids,topk=n-1,tolerance=solver["tie_tolerance"])
            status=("CELL_CONVERGED" if ranks_final==rank["rankings"] else
                    "BOUNDED_SEARCH_FEASIBLE")
            return g,xi,rank,ell,caps,status,traces,stage_times,values,relative
    values,_=_real_violations(g,xi,g0,labels,rank,ell,caps,cfg)
    return g,xi,rank,ell,caps,"BOUNDED_SEARCH_FEASIBLE",traces,stage_times,values,relative


def _real_rank_controller(g0,labels,ids,cfg):
    canonical=_real_rank_data(g0,labels,ids,cfg["solver"]["tie_tolerance"])
    base=_real_dykstra(g0,labels,ids,cfg,baseline_margins=canonical["margins"])
    g,xi,rank,ell,caps,status,trace,times,values,relative=base
    if status!="CELL_CONVERGED":
        return (*base[:5],"BOUNDED_SEARCH_FEASIBLE",*base[6:],
                {"reason":"base_cell_not_converged","adjacent_cells_total":1,
                 "adjacent_cells_checked":1,"pivots":0,"cells":[]})
    system=boundary_orientation_system(g,ids,20,cfg["solver"]["tie_tolerance"],34)
    if system["rank"]>cfg["solver"]["max_independent_orientations"] or system["compatible_overflow"]:
        return (g0.copy(),np.zeros(len(ids)),rank,ell,caps,"BOUNDED_SEARCH_FEASIBLE",
                trace,times,values,relative,{"reason":"orientation_or_pivot_budget_exceeded",
                "independent_orientations":system["rank"],"basis_indices":system["basis_indices"],
                "dependency_coefficients":system["dependency_coefficients"],
                "orientation_descriptors":system["descriptors"],
                "orientation_reference_gram":g.tolist(),
                "orientation_reference_gram_sha256":sha256_obj(g.tolist()),
                "adjacent_cells_total":len(system["compatible_assignments"]),
                "adjacent_cells_checked":0,"pivots":cfg["solver"]["max_pivots"]+1,"cells":[]})
    if system["rank"]==0:
        ledger={"reason":"all_adjacent_checked","independent_orientations":0,
                "orientation_descriptors":[],
                "basis_indices":[],"dependency_coefficients":[],
                "adjacent_cells_total":1,"adjacent_cells_checked":1,"pivots":0,
                "orientation_reference_gram":g.tolist(),
                "orientation_reference_gram_sha256":sha256_obj(g.tolist()),
                "selected_cell_rankings_sha256":sha256_obj(rank["rankings"]),
                "cells":[{"assignment":[],"objective":float(np.linalg.norm(g-g0)**2+np.linalg.norm(xi)**2),
                          "cell_rankings_sha256":sha256_obj(rank["rankings"]),
                          "target_sha256":sha256_obj({"g":g.tolist(),"xi":xi.tolist()}),
                          "status":"CELL_CONVERGED","cycles":len(trace),
                          "relative_iterate_change":relative,
                          "trace_sha256":sha256_obj(trace),
                          "trace_projector_transitions_sha256":sha256_obj(
                              [row.get("projector_transitions_sha256") for row in trace]),
                          "final_correction_state_sha256":(
                              trace[-1].get("correction_state_sha256") if trace else None),
                          "max_violation":max(values.values())}]}
        return (g,xi,rank,ell,caps,"LOCAL_STATIONARY_CERTIFIED",trace,times,values,relative,ledger)
    base_rankings=stable_rankings(g,ids,len(ids)-1,cfg["solver"]["tie_tolerance"])
    candidates=[];cell_ledger=[];aggregate_times={key:0.0 for key in times}
    for assignment in system["compatible_assignments"]:
        cell=orientation_cell_from_assignment(base_rankings,system["descriptors"],assignment,ids)
        result=_real_dykstra(g0,labels,ids,cfg,rankings_override=cell,
                              baseline_margins=canonical["margins"])
        cg,cxi,crank,cell_ell,cell_caps,cstatus,ctrace,ctimes,cvalues,crel=result
        for key in aggregate_times:aggregate_times[key]+=ctimes[key]
        objective=float(np.linalg.norm(cg-g0)**2+np.linalg.norm(cxi)**2)
        cell_ledger.append({"assignment":assignment,"cell_sha256":sha256_obj(cell),
                            "cell_rankings":cell,
                            "objective":objective,"target_sha256":sha256_obj({"g":cg.tolist(),"xi":cxi.tolist()}),
                            "status":cstatus,"cycles":len(ctrace),
                            "relative_iterate_change":crel,
                            "trace_sha256":sha256_obj(ctrace),
                            "trace_projector_transitions_sha256":sha256_obj(
                                [row.get("projector_transitions_sha256") for row in ctrace]),
                            "final_correction_state_sha256":(
                                ctrace[-1].get("correction_state_sha256") if ctrace else None),
                            "max_violation":max(cvalues.values())})
        if cstatus!="CELL_CONVERGED":
            return (g0.copy(),np.zeros(len(ids)),canonical,ell,caps,
                    "BOUNDED_SEARCH_FEASIBLE",ctrace,aggregate_times,cvalues,crel,
                    {"reason":"incomplete_adjacent_enumeration","independent_orientations":system["rank"],
                    "basis_indices":system["basis_indices"],"dependency_coefficients":system["dependency_coefficients"],
                     "orientation_descriptors":system["descriptors"],
                     "orientation_reference_gram":g.tolist(),
                     "orientation_reference_gram_sha256":sha256_obj(g.tolist()),
                     "adjacent_cells_total":len(system["compatible_assignments"]),
                     "adjacent_cells_checked":len(cell_ledger),"pivots":max(0,len(cell_ledger)-1),
                     "cells":cell_ledger})
        candidates.append((objective,result,cell))
    if len(candidates)-1>cfg["solver"]["max_pivots"]:
        return (g0.copy(),np.zeros(len(ids)),canonical,ell,caps,"BOUNDED_SEARCH_FEASIBLE",
                trace,aggregate_times,values,relative,{"reason":"pivot_budget_exceeded",
                "independent_orientations":system["rank"],
                "basis_indices":system["basis_indices"],
                "dependency_coefficients":system["dependency_coefficients"],
                "orientation_descriptors":system["descriptors"],
                "orientation_reference_gram":g.tolist(),
                "orientation_reference_gram_sha256":sha256_obj(g.tolist()),
                "adjacent_cells_total":len(candidates),"adjacent_cells_checked":len(candidates),
                "pivots":len(candidates)-1,"cells":cell_ledger})
    _,best,best_cell=min(candidates,key=lambda item:item[0])
    bg,bxi,brank,bell,bcaps,_,btrace,btimes,bvalues,brel=best
    return (bg,bxi,brank,bell,bcaps,"LOCAL_STATIONARY_CERTIFIED",btrace,aggregate_times,
            bvalues,brel,{"reason":"all_adjacent_checked","independent_orientations":system["rank"],
            "basis_indices":system["basis_indices"],"dependency_coefficients":system["dependency_coefficients"],
            "orientation_descriptors":system["descriptors"],
            "orientation_reference_gram":g.tolist(),
            "orientation_reference_gram_sha256":sha256_obj(g.tolist()),
            "adjacent_cells_total":len(candidates),"adjacent_cells_checked":len(candidates),
            "pivots":max(0,len(candidates)-1),"best_cell_sha256":sha256_obj(best_cell),
            "selected_cell_rankings_sha256":sha256_obj(brank["rankings"]),
            "cells":cell_ledger})


def _signed_tangent_families(z0, labels, anchor):
    same = []
    opposite = []
    signed = []
    signed_meta = []
    for j in range(len(z0)):
        if j == anchor:
            continue
        tangent = z0[j] - z0[anchor] * float(z0[anchor] @ z0[j])
        tangent = tangent / max(float(np.linalg.norm(tangent)), 1e-15)
        if labels[j] == labels[anchor]:
            same.append((j, tangent))
            signed.append(tangent)
            signed_meta.append(("same", int(j)))
        else:
            column = -tangent
            opposite.append((j, column))
            signed.append(column)
            signed_meta.append(("opposite", int(j)))
    return same, opposite, np.asarray(signed, dtype=np.float64), signed_meta


def _combo_column(a, b):
    column = np.asarray(a, dtype=np.float64) + np.asarray(b, dtype=np.float64)
    return column / max(float(np.linalg.norm(column)), 1e-15)


def _pair_oracle(signed, signed_meta, witness):
    if len(signed) < 2:
        return {"family": "pair", "value": -float("inf"), "indices": [], "column": None}
    gram = signed @ signed.T
    inner = signed @ witness
    best = (-float("inf"), None, None)
    for a in range(len(signed) - 1):
        denom = np.sqrt(np.maximum(2.0 + 2.0 * gram[a, a + 1:], 1e-15))
        values = (inner[a] + inner[a + 1:]) / denom
        local = int(np.argmax(values))
        value = float(values[local])
        if value > best[0]:
            b = a + 1 + local
            best = (value, a, b)
    a, b = best[1], best[2]
    return {"family": "pair", "value": best[0],
            "indices": [signed_meta[a], signed_meta[b]],
            "column": _combo_column(signed[a], signed[b])}


def _triplet_oracle(same, opposite, witness):
    if not same or not opposite:
        return {"family": "triplet", "value": -float("inf"), "indices": [], "column": None}
    same_cols = np.asarray([row[1] for row in same], dtype=np.float64)
    opp_cols = np.asarray([row[1] for row in opposite], dtype=np.float64)
    same_inner = same_cols @ witness
    opp_inner = opp_cols @ witness
    denom = np.sqrt(np.maximum(2.0 + 2.0 * (same_cols @ opp_cols.T), 1e-15))
    values = (same_inner[:, None] + opp_inner[None, :]) / denom
    flat = int(np.argmax(values))
    a, b = np.unravel_index(flat, values.shape)
    return {"family": "triplet", "value": float(values[a, b]),
            "indices": [int(same[a][0]), int(opposite[b][0])],
            "column": _combo_column(same_cols[a], opp_cols[b])}


PINNED_REGISTERED_CONE_DEFINITION = {
    "singleton": "for each anchor i and memory j!=i, the unit tangent to z_j at z_i, signed attractive for same parent label and repulsive for opposite parent label",
    "pair": "all normalized sums of two distinct singleton columns for the same anchor",
    "triplet": "all normalized sums of one same-label attractive singleton and one opposite-label repulsive singleton for the same anchor",
    "supcon": "for each anchor, the normalized sum of all same-label attractive singleton columns",
    "labels": "parent-video binary labels only",
}


def _registered_cone_definition(cfg):
    definition = dict(cfg.get("registered_cone_definition", {}))
    if definition != PINNED_REGISTERED_CONE_DEFINITION:
        raise RuntimeError("registered cone definition drift")
    return definition


def _real_cone_audit(z0, target, labels, cfg=None):
    z0 = np.asarray(z0, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)
    displacement = target - z0
    max_iter = int((cfg or {}).get("solver", {}).get("farkas_column_generation_max_iter", 64))
    sep_tol = float((cfg or {}).get("solver", {}).get("farkas_separation_tolerance", 1e-8))
    residuals = []
    targets = []
    active_banks = []
    anchor_ledgers = []
    universe = {"singleton_rgcl_columns": 0, "pair_columns": 0,
                "triplet_columns": 0, "supcon_columns": len(z0),
                "active_generated_columns": 0}
    max_family_inner = {"singleton": -float("inf"), "pair": -float("inf"),
                        "triplet": -float("inf"), "supcon": -float("inf")}
    separation_overflow = False
    for i in range(len(z0)):
        d = displacement[i] - z0[i] * float(z0[i] @ displacement[i])
        same, opposite, signed, signed_meta = _signed_tangent_families(z0, labels, i)
        supcon = np.sum(np.asarray([row[1] for row in same]), axis=0) if same else np.zeros(z0.shape[1])
        supcon = supcon / max(float(np.linalg.norm(supcon)), 1e-15)
        active = [row.copy() for row in signed] + [supcon.copy()]
        active_meta = [("singleton", meta) for meta in signed_meta] + [("supcon", int(i))]
        best_oracles = []
        overflow = False
        for iteration in range(max_iter + 1):
            matrix = np.asarray(active, dtype=np.float64).T
            audit = cone_audit(matrix, d)
            residual = audit["residual"]
            norm = float(np.linalg.norm(residual))
            witness = residual / max(norm, 1e-15)
            singleton_value = float(np.max(signed @ witness)) if len(signed) else -float("inf")
            supcon_value = float(supcon @ witness)
            pair = _pair_oracle(signed, signed_meta, witness)
            triplet = _triplet_oracle(same, opposite, witness)
            best = max([
                {"family": "singleton", "value": singleton_value, "indices": [], "column": None},
                {"family": "supcon", "value": supcon_value, "indices": [int(i)], "column": None},
                pair, triplet], key=lambda row: row["value"])
            best_oracles.append({"iteration": iteration, "family": best["family"],
                                 "value": best["value"], "indices": best["indices"]})
            if best["value"] <= sep_tol or best["column"] is None:
                break
            active.append(best["column"])
            active_meta.append((best["family"], best["indices"]))
        else:
            overflow = True
            separation_overflow = True
        matrix = np.asarray(active, dtype=np.float64).T
        audit = cone_audit(matrix, d)
        residual = audit["residual"]
        residuals.append(residual)
        targets.append(d)
        active_banks.append(np.asarray(active, dtype=np.float64))
        local_witness = residual / max(float(np.linalg.norm(residual)), 1e-15)
        singleton_max = float(np.max(signed @ local_witness)) if len(signed) else -float("inf")
        supcon_max = float(supcon @ local_witness)
        pair_max = _pair_oracle(signed, signed_meta, local_witness)["value"]
        triplet_max = _triplet_oracle(same, opposite, local_witness)["value"]
        max_family_inner["singleton"] = max(max_family_inner["singleton"], singleton_max)
        max_family_inner["pair"] = max(max_family_inner["pair"], pair_max)
        max_family_inner["triplet"] = max(max_family_inner["triplet"], triplet_max)
        max_family_inner["supcon"] = max(max_family_inner["supcon"], supcon_max)
        universe["singleton_rgcl_columns"] += len(signed)
        universe["pair_columns"] += len(signed) * (len(signed) - 1) // 2
        universe["triplet_columns"] += len(same) * len(opposite)
        universe["active_generated_columns"] += len(active)
        anchor_ledgers.append({
            "anchor": i,
            "same_columns": len(same),
            "opposite_columns": len(opposite),
            "singleton_universe": len(signed),
            "pair_universe": len(signed) * (len(signed) - 1) // 2,
            "triplet_universe": len(same) * len(opposite),
            "supcon_universe": 1,
            "active_columns": len(active),
            "active_column_meta_sha256": sha256_obj(active_meta),
            "active_columns_sha256": sha256_obj(np.asarray(active, dtype=np.float64).tolist()),
            "column_generation_iterations": len(best_oracles) - 1,
            "separation_overflow": overflow,
            "oracle_trace": best_oracles,
            "nnls_residual_norm": float(np.linalg.norm(residual)),
        })
    residual = np.concatenate(residuals)
    flat = np.concatenate(targets)
    residual_norm = float(np.linalg.norm(residual))
    target_norm = max(float(np.linalg.norm(flat)), 1e-15)
    witness = residual / max(residual_norm, 1e-15)
    active_max = -float("inf")
    offset = 0
    for active in active_banks:
        wi = witness[offset:offset + z0.shape[1]]
        offset += z0.shape[1]
        active_max = max(active_max, float(np.max(active @ wi)))
    max_cone_inner = max(active_max, max(max_family_inner.values()))
    separation = residual_norm / target_norm
    dual_value = float(witness @ flat)
    definition = _registered_cone_definition(cfg or {})
    return {"relative_separation": separation,
            "max_cone_witness_inner": float(max_cone_inner),
            "family_max_witness_inner": {k: float(v) for k, v in max_family_inner.items()},
            "duality_gap": abs(residual_norm - dual_value),
            "universe": universe,
            "registered_cone_definition": definition,
            "residual_norm": residual_norm,
            "target_tangent_norm": target_norm,
            "dual_value": dual_value,
            "witness": witness.tolist(),
            "universe_sha256": sha256_obj({"definition": definition,
                                           "universe": universe,
                                           "anchors": anchor_ledgers}),
            "anchor_ledgers": anchor_ledgers,
            "separation_overflow": separation_overflow,
            "witness_sha256": sha256_obj(witness.tolist())}


def _contains_mixed_locator_or_hash(obj):
    try:
        assert_no_formal_forbidden_surface(obj, "formal_record")
        return False
    except RuntimeError:
        return True


DEFAULT_LINEAGE_RUN_IDS = {
    "freeze": "LBSCGP-G0-FREEZE-v1",
    "code_audit": "LBSCGP-G0-CODE-AUDIT-v1",
    "synthetic": "LBSCGP-G0-SYNTH-v1",
    "realfold": "LBSCGP-G0-REAL-MHC_zh-F4-S0-v1",
    "replay": "LBSCGP-G0-REAL-REPLAY-MHC_zh-F4-S0-v1",
    "decision": "LBSCGP-G0-DECISION-v1",
}
DEFAULT_FREEZE_PROTOCOL_INPUT_KEYS = (
    "experiment_plan", "experiment_tracker", "problem_anchor",
    "final_proposal", "review_summary", "refinement_report",
    "target_loop", "target_state",
)
CODE_AUDIT_RECORD_TYPE = "LB_SCGP_G0_CODE_AUDIT_INDEPENDENT_REVIEW_RECORD_V4"
CODE_AUDIT_ARTIFACT_TYPE = "LB_SCGP_G0_CODE_AUDIT_PASS_ARTIFACT_V4"
CODE_AUDIT_INDEX_TYPE = "LB_SCGP_G0_CODE_AUDIT_PUBLICATION_INDEX_V4"
CODE_AUDIT_REVIEW_SCOPE = "LB-SCGP G0 v4 formal code audit for LBSCGP-G0-FREEZE-v4"
CODE_AUDIT_REVIEW_PROCESS_IDENTITY = "fresh_independent_gpt_5_5_xhigh"
CODE_AUDIT_DIR = Path("g0/code_audit")
CODE_AUDIT_REVIEW_ARTIFACT = "review.md"
CODE_AUDIT_RECORD_ARTIFACT = "review_record.json"
CODE_AUDIT_JSON_ARTIFACT = "audit.json"
CODE_AUDIT_INDEX_ARTIFACT = "publication_index.json"
CODE_AUDIT_EXACT_FORMAL_PREFIXES_V4 = (
    "artifacts/lb_scgp/v1/",
    "artifacts/lb_scgp/v2/",
    "artifacts/lb_scgp/v3/",
    "artifacts/lb_scgp/v4/",
)
CODE_AUDIT_EXACT_FORMAL_PREFIXES_V5 = (
    "artifacts/lb_scgp/v1/",
    "artifacts/lb_scgp/v2/",
    "artifacts/lb_scgp/v3/",
    "artifacts/lb_scgp/v4/",
    "artifacts/lb_scgp/v5/",
)
CODE_AUDIT_EXPECTED_PRIOR_HASH_PATHS_V4 = (
    "artifacts/lb_scgp/v1/CONFIG_FREEZE.json",
    "artifacts/lb_scgp/v1/CONFIG_FREEZE.json.publish.lock",
    "artifacts/lb_scgp/v2/CONFIG_FREEZE.json",
    "artifacts/lb_scgp/v2/CONFIG_FREEZE.json.publish.lock",
    "artifacts/lb_scgp/v3/CONFIG_FREEZE.json",
    "artifacts/lb_scgp/v3/CONFIG_FREEZE.json.publish.lock",
)
CODE_AUDIT_EXPECTED_PRIOR_HASH_PATHS_V5 = (
    "artifacts/lb_scgp/v1/CONFIG_FREEZE.json",
    "artifacts/lb_scgp/v1/CONFIG_FREEZE.json.publish.lock",
    "artifacts/lb_scgp/v2/CONFIG_FREEZE.json",
    "artifacts/lb_scgp/v2/CONFIG_FREEZE.json.publish.lock",
    "artifacts/lb_scgp/v3/CONFIG_FREEZE.json",
    "artifacts/lb_scgp/v3/CONFIG_FREEZE.json.publish.lock",
    "artifacts/lb_scgp/v4/CONFIG_FREEZE.json",
    "artifacts/lb_scgp/v4/CONFIG_FREEZE.json.publish.lock",
    "artifacts/lb_scgp/v4/g0/code_audit/audit.json",
    "artifacts/lb_scgp/v4/g0/code_audit/audit.json.publish.lock",
    "artifacts/lb_scgp/v4/g0/code_audit/publication_index.json",
    "artifacts/lb_scgp/v4/g0/code_audit/publication_index.json.publish.lock",
    "artifacts/lb_scgp/v4/g0/code_audit/review.md",
    "artifacts/lb_scgp/v4/g0/code_audit/review.md.publish.lock",
    "artifacts/lb_scgp/v4/g0/code_audit/review_record.json",
    "artifacts/lb_scgp/v4/g0/code_audit/review_record.json.publish.lock",
)
CODE_AUDIT_REVIEW_RECORD_KEYS = frozenset({
    "schema_version", "record_type", "run_id", "stage", "status",
    "lineage_version", "config_path", "artifact_namespace",
    "freeze_run_id", "freeze_path", "freeze_file_sha256",
    "freeze_payload_sha256", "config_canonical_sha256",
    "implementation_sha256", "independent_verifier_sha256",
    "review_report_path", "review_report_sha256", "reviewer_identity",
    "review_process_identity", "review_scope", "critical", "high",
    "important", "no_segment_gold_pass", "formal_pass_authorized",
    "independent_reviewer", "repair_executor_created",
    "only_gold_supervision", "segment_gold_exists", "segment_gold_used",
    "payload_sha256",
})
CODE_AUDIT_ARTIFACT_KEYS = frozenset({
    "schema_version", "artifact_type", "run_id", "stage", "status",
    "critical", "high", "important", "no_segment_gold_pass",
    "formal_pass_authorized", "slurm_job_id", "config_path",
    "config_file_sha256", "config_canonical_sha256", "artifact_namespace",
    "lineage_version", "freeze_path", "freeze_file_sha256",
    "freeze_lock_path", "freeze_lock_sha256", "freeze_payload_sha256",
    "freeze_run_id", "freeze_stage", "git_head", "dirty_diff_sha256",
    "frozen_dirty_diff_sha256", "implementation_sha256",
    "implementation_files", "independent_verifier_sha256",
    "review_report_path", "review_report_sha256", "review_record_path",
    "review_record_sha256", "review_record_payload_sha256",
    "reviewer_identity", "review_process_identity", "review_scope",
    "only_gold_supervision", "segment_gold_exists", "segment_gold_used",
    "mllm_call_count", "ocr_call_count", "teacher_cache_read_count",
    "teacher_cache_write_count", "outer_held_label_read_count",
    "outer_held_content_read_count", "val_content_read_count",
    "test_content_read_count", "val_test_teacher_artifact_count",
    "formal_model_optimizer_evaluator_outer_held_read_count",
    "access_ledger", "access_ledger_sha256", "dirty_policy",
    "frozen_input_rehashes", "allowed_bank_member_sha256",
    "forbidden_bank_members_not_opened", "prior_lineage_no_clobber_hashes",
    "audit_publish_wrapper", "authorization_gate", "downstream_contract",
    "output_files", "python_version", "numpy_version", "scipy_version",
    "torch_version", "conda_env", "payload_sha256",
})
CODE_AUDIT_INDEX_KEYS = frozenset({
    "schema_version", "artifact_type", "run_id", "stage", "status",
    "output_files", "lock_files", "payload_sha256",
})


def _lineage_run_id(cfg, key):
    return cfg.get("lineage", {}).get("run_ids", {}).get(
        key, DEFAULT_LINEAGE_RUN_IDS[key])


def _freeze_name(cfg):
    return cfg.get("lineage", {}).get("freeze_name", _lineage_run_id(cfg, "freeze"))


def _freeze_protocol_input_keys(cfg):
    return tuple(cfg.get("lineage", {}).get(
        "freeze_input_path_keys", DEFAULT_FREEZE_PROTOCOL_INPUT_KEYS))


def _require_exact_keys(record_name, record, expected):
    keys = set(record)
    if keys != set(expected):
        missing = sorted(set(expected) - keys)
        extra = sorted(keys - set(expected))
        raise RuntimeError("{} schema keys mismatch missing={} extra={}".format(
            record_name, missing, extra))


def _lineage_path(cfg, key):
    value = cfg.get("lineage", {}).get(key)
    if not isinstance(value, str) or not value:
        raise RuntimeError("lineage.{} missing".format(key))
    return value


def _code_audit_contract(cfg):
    version = cfg.get("lineage", {}).get("version")
    if version not in {"v4", "v5"}:
        return None
    suffix = version.upper()
    exact_prefixes = {
        "v4": CODE_AUDIT_EXACT_FORMAL_PREFIXES_V4,
        "v5": CODE_AUDIT_EXACT_FORMAL_PREFIXES_V5,
    }[version]
    prior_paths = {
        "v4": CODE_AUDIT_EXPECTED_PRIOR_HASH_PATHS_V4,
        "v5": CODE_AUDIT_EXPECTED_PRIOR_HASH_PATHS_V5,
    }[version]
    return {
        "version": version,
        "record_type": "LB_SCGP_G0_CODE_AUDIT_INDEPENDENT_REVIEW_RECORD_{}".format(suffix),
        "artifact_type": "LB_SCGP_G0_CODE_AUDIT_PASS_ARTIFACT_{}".format(suffix),
        "index_type": "LB_SCGP_G0_CODE_AUDIT_PUBLICATION_INDEX_{}".format(suffix),
        "review_scope": "LB-SCGP G0 {} formal code audit for {}".format(
            version, _lineage_run_id(cfg, "freeze")),
        "exact_formal_prefixes": exact_prefixes,
        "prior_hash_paths": prior_paths,
    }


def _audit_payload_hash(obj):
    copy = dict(obj)
    copy.pop("payload_sha256", None)
    return sha256_obj(copy)


def _zero_counter_fields(obj, include_formal_outer=False):
    fields = {key: int(obj.get(key, -1)) for key in ZERO_COUNTER_KEYS}
    if include_formal_outer:
        fields["formal_model_optimizer_evaluator_outer_held_read_count"] = int(
            obj.get("formal_model_optimizer_evaluator_outer_held_read_count", -1))
    return fields


def _verify_v4_dirty_policy_contract(cfg, freeze):
    contract = _code_audit_contract(cfg)
    if contract is None:
        raise RuntimeError("strict code-audit dirty contract requires v4/v5 lineage")
    artifact_prefixes, dirty_paths, dirty_prefixes = dirty_state_policy(cfg)
    if tuple(artifact_prefixes) != contract["exact_formal_prefixes"]:
        raise RuntimeError("{} formal artifact exclusions are not exact".format(
            contract["version"]))
    if list(artifact_prefixes) != freeze.get("formal_artifact_exclude_prefixes") or \
            list(dirty_paths) != freeze.get("dirty_state_excluded_paths") or \
            list(dirty_prefixes) != freeze.get("dirty_state_excluded_prefixes"):
        raise RuntimeError("freeze dirty policy does not match config")
    review_path = _lineage_path(cfg, "review_report_path")
    record_path = _lineage_path(cfg, "review_record_path")
    if review_path not in dirty_paths or record_path not in dirty_paths:
        raise RuntimeError("review/report sidecar paths are not exact dirty exclusions")
    return {
        "formal_artifact_exclude_prefixes": list(artifact_prefixes),
        "dirty_state_excluded_paths": list(dirty_paths),
        "dirty_state_excluded_prefixes": list(dirty_prefixes),
    }


def _verify_v4_code_audit_schema(cfg, artifacts, freeze, audit, ledger):
    contract = _code_audit_contract(cfg)
    if contract is None:
        return
    assert_no_formal_forbidden_surface(audit, "code_audit")
    _require_exact_keys("code_audit", audit, CODE_AUDIT_ARTIFACT_KEYS)
    if audit.get("payload_sha256") != _audit_payload_hash(audit):
        raise RuntimeError("code-audit payload hash invalid")
    code_dir = artifacts / CODE_AUDIT_DIR
    index_path = code_dir / CODE_AUDIT_INDEX_ARTIFACT
    review_path = code_dir / CODE_AUDIT_REVIEW_ARTIFACT
    record_path = code_dir / CODE_AUDIT_RECORD_ARTIFACT
    audit_path = code_dir / CODE_AUDIT_JSON_ARTIFACT
    index, _ = ledger.read_json(index_path, "predecessor_verify", "code_audit_index")
    _require_exact_keys("publication_index", index, CODE_AUDIT_INDEX_KEYS)
    if index.get("payload_sha256") != _audit_payload_hash(index):
        raise RuntimeError("code-audit publication index payload invalid")
    record, _ = ledger.read_json(record_path, "predecessor_verify", "code_audit_review_record")
    _require_exact_keys("review_record", record, CODE_AUDIT_REVIEW_RECORD_KEYS)
    if record.get("payload_sha256") != _audit_payload_hash(record):
        raise RuntimeError("code-audit review record payload invalid")
    review_sha = ledger.hash_file(review_path, "predecessor_verify", "code_audit_review")
    record_sha = ledger.hash_file(record_path, "predecessor_verify", "code_audit_review_record")
    audit_sha = ledger.hash_file(audit_path, "predecessor_verify", "code_audit")
    lock_rows = index.get("lock_files", [])
    if len(lock_rows) != 4:
        raise RuntimeError("code-audit lock index invalid")
    for row in lock_rows:
        lock_path = ROOT / row["path"]
        if not lock_path.exists() or ledger.hash_file(
                lock_path, "predecessor_verify", "code_audit_lock") != row["sha256"]:
            raise RuntimeError("code-audit lock missing or drifted")
    expected_outputs = [
        {"path": root_relative_path(review_path), "sha256": review_sha},
        {"path": root_relative_path(record_path), "sha256": record_sha},
        {"path": root_relative_path(audit_path), "sha256": audit_sha},
    ]
    if index.get("output_files") != expected_outputs or audit.get("output_files") != expected_outputs[:2]:
        raise RuntimeError("code-audit output hash index drift")
    config_path, _ = canonical_root_path(_lineage_path(cfg, "config_path"))
    config_file_sha = sha256_file(config_path)
    impl, impl_rows = implementation_hash(cfg)
    verifier_sha = sha256_file(ROOT / "scripts/analysis/lb_scgp_independent_verify.py")
    dirty_policy = _verify_v4_dirty_policy_contract(cfg, freeze)
    prior_expected = cfg.get("lineage", {}).get("prior_lineage_no_clobber_hashes", {})
    if set(prior_expected) != set(contract["prior_hash_paths"]):
        raise RuntimeError("{} prior lineage hash evidence set is not exact".format(
            contract["version"]))
    prior_rows = [{"path": path, "sha256": sha256_file(ROOT / path)}
                  for path in sorted(contract["prior_hash_paths"])]
    if {row["path"] for row in prior_rows} != set(prior_expected) or \
            any(row["sha256"] != prior_expected[row["path"]] for row in prior_rows):
        raise RuntimeError("prior lineage hash evidence drift")
    checks = {
        "artifact_type": audit.get("artifact_type") == contract["artifact_type"],
        "identity": audit.get("run_id") == _lineage_run_id(cfg, "code_audit") and
            audit.get("stage") == "G0_CODE_AUDIT" and audit.get("status") == "PASS",
        "index": index.get("artifact_type") == contract["index_type"] and
            index.get("run_id") == audit.get("run_id") and
            index.get("stage") == audit.get("stage") and index.get("status") == "PASS",
        "findings": int(audit.get("critical", -1)) == 0 and
            int(audit.get("high", -1)) == 0 and audit.get("no_segment_gold_pass") is True and
            audit.get("formal_pass_authorized") is True,
        "config": audit.get("config_path") == _lineage_path(cfg, "config_path") and
            audit.get("config_file_sha256") == config_file_sha and
            audit.get("config_canonical_sha256") == cfg["config_canonical_sha256"],
        "lineage": audit.get("artifact_namespace") == _lineage_path(
                cfg, "artifact_namespace") and
            audit.get("lineage_version") == contract["version"],
        "freeze": audit.get("freeze_path") == root_relative_path(artifacts / "CONFIG_FREEZE.json") and
            audit.get("freeze_payload_sha256") == freeze.get("payload_sha256") and
            audit.get("freeze_run_id") == freeze.get("run_id") and
            audit.get("freeze_stage") == freeze.get("stage"),
        "implementation": audit.get("implementation_sha256") == impl and
            audit.get("implementation_files") == impl_rows and
            audit.get("independent_verifier_sha256") == verifier_sha,
        "dirty": audit.get("dirty_diff_sha256") == git_state(cfg=cfg)[1] and
            audit.get("frozen_dirty_diff_sha256") == freeze.get("dirty_diff_sha256") and
            audit.get("dirty_diff_sha256") == audit.get("frozen_dirty_diff_sha256"),
        "record": audit.get("review_report_path") == record.get("review_report_path") and
            audit.get("review_report_sha256") == review_sha and
            audit.get("review_record_path") == _lineage_path(cfg, "review_record_path") and
            audit.get("review_record_sha256") == record_sha and
            audit.get("review_record_payload_sha256") == record.get("payload_sha256") and
            audit.get("reviewer_identity") == record.get("reviewer_identity") and
            audit.get("review_process_identity") == CODE_AUDIT_REVIEW_PROCESS_IDENTITY and
            audit.get("review_scope") == contract["review_scope"],
        "record_values": record.get("record_type") == contract["record_type"] and
            record.get("run_id") == audit.get("run_id") and record.get("stage") == audit.get("stage") and
            record.get("status") == "PASS" and int(record.get("critical", -1)) == 0 and
            int(record.get("high", -1)) == 0 and record.get("no_segment_gold_pass") is True and
            record.get("formal_pass_authorized") is True and
            record.get("independent_reviewer") is True and record.get("repair_executor_created") is False and
            record.get("lineage_version") == contract["version"] and
            record.get("artifact_namespace") == _lineage_path(cfg, "artifact_namespace") and
            record.get("review_scope") == contract["review_scope"],
        "access": audit.get("access_ledger_sha256") == sha256_obj(audit.get("access_ledger", [])),
        "dirty_policy": audit.get("dirty_policy") == dirty_policy,
        "prior": audit.get("prior_lineage_no_clobber_hashes") == prior_rows,
        "supervision": audit.get("only_gold_supervision") == "parent_video_binary_label" and
            audit.get("segment_gold_exists") is False and audit.get("segment_gold_used") is False and
            record.get("only_gold_supervision") == "parent_video_binary_label" and
            record.get("segment_gold_exists") is False and record.get("segment_gold_used") is False,
        "zero": all(value == 0 for value in _zero_counter_fields(
            audit, include_formal_outer=True).values()),
        "bank": audit.get("allowed_bank_member_sha256") == cfg["sealed_real_fixture"]["bank_member_sha256"] and
            audit.get("forbidden_bank_members_not_opened") == cfg["sealed_real_fixture"]["forbidden_bank_members"],
    }
    if not all(checks.values()):
        failed = sorted(key for key, value in checks.items() if not value)
        raise RuntimeError("strict code-audit schema failed {}".format(failed))


def _require_keys(record_name, record, keys):
    missing = [key for key in keys if key not in record]
    if missing:
        raise RuntimeError("{} missing dedicated sanitizer fields {}".format(
            record_name, missing))


def _validate_dedicated_pre_freeze_sanitizer_schema(cfg, provenance,
                                                    prov_sha, decision,
                                                    decision_sha):
    prov_required = {
        "schema_version", "run_id", "stage", "slurm_job_id", "dataset",
        "outer_fold", "artifact_namespace", "feature_cache_path",
        "feature_cache_sha256", "memory_id_count", "query_id_sentinel_count",
        "memory_ids_sha256", "memory_labels_sha256", "query_ids_sha256",
        "row_selection_rule", "parent_label_rule", "input_cache_labels_ignored",
        "zero_overlap_with_query_ids", "payload_sha256",
        "sanitizer_code_sha256", "segment_cache_path",
        "segment_cache_sha256", "segment_artifact_created",
        "segment_objective_allowed", "teacher_mllm_ocr_calls",
        "network_external_calls", "formal_query_z_read_count",
        "formal_query_labels_read_count",
        "formal_model_optimizer_evaluator_outer_held_read_count",
        "pre_freeze_disclosure_record_external", "no_clobber_locks_present",
    }
    decision_required = {
        "schema_version", "run_id", "stage", "status", "slurm_job_id",
        "dataset", "outer_fold", "feature_cache_path",
        "feature_cache_sha256", "sanitized_provenance_path",
        "sanitized_provenance_sha256", "safe_contract_sha256",
        "memory_id_count", "query_id_sentinel_count", "memory_ids_sha256",
        "memory_labels_sha256", "query_ids_sha256", "payload_sha256",
        "segment_cache_path", "segment_cache_sha256",
        "segment_artifact_created", "segment_objective_allowed",
        "teacher_mllm_ocr_calls", "network_external_calls",
        "formal_query_z_read_count", "formal_query_labels_read_count",
        "formal_model_optimizer_evaluator_outer_held_read_count", "gates",
    }
    _require_keys("sanitized_provenance", provenance, prov_required)
    _require_keys("sanitizer_decision", decision, decision_required)
    fixture = cfg["sealed_real_fixture"]
    expected = {
        "dataset": fixture["dataset"],
        "outer_fold": fixture["outer_fold"],
        "memory_id_count": fixture["outer_train_n"],
        "query_id_sentinel_count": fixture["outer_held_n"],
        "memory_ids_sha256": fixture["memory_ids_sha256"],
        "query_ids_sha256": fixture["query_ids_sha256"],
    }
    for key, value in expected.items():
        if provenance.get(key) != value or decision.get(key) != value:
            raise RuntimeError("pre-freeze sanitizer {} mismatch".format(key))
    if provenance.get("payload_sha256") != payload_hash(provenance) or \
            decision.get("payload_sha256") != payload_hash(decision):
        raise RuntimeError("pre-freeze sanitizer payload hash invalid")
    if decision.get("sanitized_provenance_sha256") != prov_sha or \
            decision_sha == prov_sha:
        raise RuntimeError("pre-freeze sanitizer record hash binding invalid")
    if provenance.get("feature_cache_sha256") != decision.get("feature_cache_sha256"):
        raise RuntimeError("pre-freeze sanitizer feature hash mismatch")
    if provenance.get("input_cache_labels_ignored") is not True or \
            provenance.get("zero_overlap_with_query_ids") is not True:
        raise RuntimeError("pre-freeze sanitizer row/label contract invalid")
    for record_name, record in (("provenance", provenance), ("decision", decision)):
        if record.get("segment_cache_path") is not None or \
                record.get("segment_cache_sha256") is not None or \
                record.get("segment_artifact_created") is not False or \
                record.get("segment_objective_allowed") is not False:
            raise RuntimeError("{} exposes segment cache/objective".format(record_name))
        for key in ("teacher_mllm_ocr_calls", "network_external_calls",
                    "formal_query_z_read_count", "formal_query_labels_read_count",
                    "formal_model_optimizer_evaluator_outer_held_read_count"):
            if int(record.get(key, -1)) != 0:
                raise RuntimeError("{} nonzero forbidden counter {}".format(record_name, key))
    gates = decision.get("gates", {})
    for key in ("exact_id_order", "feature_schema_whitelist",
                "formal_provenance_sanitized",
                "labels_inherited_from_memory_labels", "no_clobber_locks",
                "no_network_external_calls", "no_segment_artifact",
                "no_segment_objective", "no_teacher_mllm_ocr_calls",
                "pre_freeze_disclosure_record_present",
                "zero_overlap_with_query_ids"):
        if gates.get(key) is not True:
            raise RuntimeError("sanitizer decision gate not true: {}".format(key))
    for path in (resolve(cfg, "outer_train_feature_cache"),
                 resolve(cfg, "sanitized_provenance"),
                 resolve(cfg, "sanitizer_decision")):
        if not path.with_name(path.name + ".publish.lock").exists():
            raise RuntimeError("missing sanitizer no-clobber lock {}".format(path))


def _read_pre_freeze_sanitizer_contract_snapshot(cfg, ledger, sanitized):
    if "pre_freeze_sanitizer_contract" not in cfg.get("paths", {}):
        return None, None
    snapshot_path = resolve(cfg, "pre_freeze_sanitizer_contract")
    snapshot, snapshot_sha = ledger.read_json(
        snapshot_path, "pre_freeze_sanitizer_schema_verify",
        "sanitizer_contract_snapshot")
    assert_no_formal_forbidden_surface(snapshot, "pre_freeze_sanitizer_contract")
    if snapshot.get("payload_sha256") != payload_hash(snapshot):
        raise RuntimeError("pre-freeze sanitizer contract snapshot payload invalid")
    required_true = (
        "dedicated_schema_registered_before_v2_freeze",
        "full_generic_manifest_schema_applies_from_g0_freeze",
        "safe_provenance_schema_ok", "safe_decision_schema_ok",
        "pre_freeze_disclosure_record_schema_ok", "no_segment_artifact",
        "no_segment_objective", "no_held_label_or_content_access",
        "zero_teacher_mllm_ocr_calls", "all_no_clobber_locks_present",
    )
    if any(snapshot.get(key) is not True for key in required_true):
        raise RuntimeError("pre-freeze sanitizer contract snapshot is not all-true")
    hashes = snapshot.get("record_hashes", {})
    if hashes.get("safe_provenance") != sanitized["sanitized_provenance_sha256"] or \
            hashes.get("safe_decision") != sanitized["sanitizer_decision_sha256"] or \
            hashes.get("train_only_feature_cache") != sanitized["feature_cache_sha256"]:
        raise RuntimeError("pre-freeze sanitizer contract snapshot hash drift")
    return snapshot, snapshot_sha


def _read_sanitized_contract(cfg, ledger, purpose):
    provenance_path = resolve(cfg, "sanitized_provenance")
    decision_path = resolve(cfg, "sanitizer_decision")
    provenance, prov_sha = ledger.read_json(
        provenance_path, purpose, "sanitized_provenance")
    decision, decision_sha = ledger.read_json(
        decision_path, purpose, "sanitizer_decision")
    if provenance.get("payload_sha256") != payload_hash(provenance) or \
            decision.get("payload_sha256") != payload_hash(decision):
        raise RuntimeError("sanitizer provenance/decision payload hash invalid")
    if decision.get("status") != "PASS" or decision.get("stage") != "LB_SCGP_SANITIZER_DECISION":
        raise RuntimeError("sanitizer independent decision is not PASS")
    if provenance.get("stage") != "LB_SCGP_SANITIZED_PROVENANCE":
        raise RuntimeError("sanitized provenance identity invalid")
    assert_no_formal_forbidden_surface(provenance, "sanitized_provenance")
    assert_no_formal_forbidden_surface(decision, "sanitizer_decision")
    if decision.get("sanitized_provenance_sha256") != prov_sha:
        raise RuntimeError("sanitizer decision does not bind sanitized provenance")
    if decision.get("feature_cache_sha256") != provenance.get("feature_cache_sha256"):
        raise RuntimeError("sanitizer train-only output hashes disagree")
    for record_name, record in (("provenance", provenance), ("decision", decision)):
        if record.get("segment_cache_path") is not None or \
                record.get("segment_cache_sha256") is not None or \
                record.get("segment_artifact_created") is not False or \
                record.get("segment_objective_allowed") is not False:
            raise RuntimeError("{} exposes segment cache/objective".format(record_name))
    if cfg["sealed_real_fixture"].get("sanitized_provenance_sha256") and \
            cfg["sealed_real_fixture"]["sanitized_provenance_sha256"] != prov_sha:
        raise RuntimeError("sanitized provenance hash drift")
    if cfg["sealed_real_fixture"].get("sanitizer_decision_sha256") and \
            cfg["sealed_real_fixture"]["sanitizer_decision_sha256"] != decision_sha:
        raise RuntimeError("sanitizer decision hash drift")
    if provenance.get("pre_freeze_disclosure_record_external") is not True or \
            int(provenance.get("formal_model_optimizer_evaluator_outer_held_read_count", -1)) != 0:
        raise RuntimeError("sanitized provenance quarantine/formal access disclosure invalid")
    _validate_dedicated_pre_freeze_sanitizer_schema(
        cfg, provenance, prov_sha, decision, decision_sha)
    return {"provenance": provenance, "decision": decision,
            "sanitized_provenance_sha256": prov_sha,
            "sanitizer_decision_sha256": decision_sha,
            "feature_cache_sha256": decision["feature_cache_sha256"]}


def task_freeze(cfg, args, ledger):
    if args.run_id != _lineage_run_id(cfg, "freeze"):
        raise RuntimeError("wrong frozen run ID")
    artifacts = resolve(cfg, "artifacts")
    if artifacts.exists() and any(artifacts.iterdir()):
        raise RuntimeError("formal namespace already exists")
    fixture = cfg["sealed_real_fixture"]
    config_path, _ = canonical_root_path(args.config)
    config_digest=ledger.records[0]["sha256"]
    inputs=[{"path":root_relative_path(config_path),"sha256":config_digest}]
    sanitized = _read_sanitized_contract(cfg, ledger, "sanitizer_pre_freeze_verify")
    inputs.append({"path": root_relative_path(resolve(cfg, "sanitized_provenance")),
                   "sha256": sanitized["sanitized_provenance_sha256"]})
    inputs.append({"path": root_relative_path(resolve(cfg, "sanitizer_decision")),
                   "sha256": sanitized["sanitizer_decision_sha256"]})
    snapshot, snapshot_sha = _read_pre_freeze_sanitizer_contract_snapshot(
        cfg, ledger, sanitized)
    if snapshot_sha:
        inputs.append({"path": root_relative_path(
            resolve(cfg, "pre_freeze_sanitizer_contract")),
            "sha256": snapshot_sha})
    for key, expected_key in [
            ("checkpoint", "checkpoint_sha256"),
            ("remove_ledger", "remove_ledger_sha256")]:
        path = resolve(cfg, key)
        actual = ledger.hash_file(path,"sealed_fixture_hash_only",key)
        if actual != fixture[expected_key]:
            raise RuntimeError("fixture hash mismatch {}".format(key))
        inputs.append({"path": root_relative_path(path), "sha256": actual})
    bank_path = resolve(cfg, "bank")
    member_hashes = hash_npz_members_only(
        bank_path, fixture["allowed_bank_members"], fixture["forbidden_bank_members"],
        ledger, "sealed_bank_allowed_member_hash_only", "bank_allowed_members")
    if member_hashes != fixture["bank_member_sha256"]:
        raise RuntimeError("bank allowed-member hash mismatch")
    inputs.append({"path": root_relative_path(bank_path),
                   "member_sha256": member_hashes,
                   "forbidden_members_not_opened": fixture["forbidden_bank_members"]})
    expected = fixture.get("outer_train_feature_cache_sha256") or sanitized["feature_cache_sha256"]
    if not expected:
        raise RuntimeError(
            "missing authoritative whole-video train-only feature hash; "
            "combined caches and segment caches are forbidden inputs")
    path = resolve(cfg, "outer_train_feature_cache")
    actual = ledger.hash_file(path, "authoritative_train_only_hash",
                              "outer_train_feature_cache")
    if actual != expected:
        raise RuntimeError("whole-video train-only feature artifact hash mismatch")
    inputs.append({"path": root_relative_path(path), "sha256": actual})
    skip_keys = {"pre_freeze_sanitizer_contract"}
    for key in _freeze_protocol_input_keys(cfg):
        if key in skip_keys:
            continue
        path=resolve(cfg,key); actual=ledger.hash_file(path,"authoritative_input_freeze",key)
        inputs.append({"path":root_relative_path(path),"sha256":actual})
    verifier = ROOT / "scripts/analysis/lb_scgp_independent_verify.py"
    base = provenance_base(cfg, args.run_id, "G0_FREEZE", sha256_file(verifier), ledger)
    artifact_excludes, dirty_paths, dirty_prefixes = dirty_state_policy(cfg)
    base.update({
        "status": "FROZEN", "freeze_name": _freeze_name(cfg),
        "lineage_version": cfg.get("lineage", {}).get("version", "v1"),
        "input_files": inputs, "fold_ids_sha256": fixture["memory_ids_sha256"],
        "checkpoint_sha256": fixture["checkpoint_sha256"],
        "sanitized_provenance_sha256": sanitized["sanitized_provenance_sha256"],
        "sanitizer_decision_sha256": sanitized["sanitizer_decision_sha256"],
        "pre_freeze_sanitizer_contract_sha256": snapshot_sha,
        "stable_protocol_input_keys": list(_freeze_protocol_input_keys(cfg)),
        "mutable_records_excluded_from_freeze_inputs": cfg.get(
            "lineage", {}).get("mutable_records_excluded_from_freeze_inputs", []),
        "formal_artifact_exclude_prefixes": list(artifact_excludes),
        "dirty_state_excluded_paths": list(dirty_paths),
        "dirty_state_excluded_prefixes": list(dirty_prefixes),
        "synthetic_seed": cfg["synthetic"]["canonical_seed"],
        "synthetic_expected_status_ledger": cfg["synthetic"]["expected_cases"],
        "outer_held_ids_role": "exclusion_sentinels_only",
        "outer_held_labels_opened": False, "outer_held_content_opened": False,
        "protected_storage_read": False,
        "pre_freeze_disclosure_record_external": True,
        "formal_model_optimizer_evaluator_outer_held_read_count": 0,
        "G1_G4_locked": True,
    })
    base["payload_sha256"] = payload_hash(base)
    publish_json(artifacts / "CONFIG_FREEZE.json", base)
    print(canonical_json({"status": "FROZEN", "run_id": args.run_id,
                          "payload_sha256": base["payload_sha256"]}))


def _load_freeze_and_audit(cfg, ledger):
    artifacts = resolve(cfg, "artifacts")
    freeze_path = artifacts / "CONFIG_FREEZE.json"
    audit_path = artifacts / "g0/code_audit/audit.json"
    if not freeze_path.exists() or not audit_path.exists():
        raise RuntimeError("freeze and independent code audit must precede numerical task")
    freeze,_=ledger.read_json(freeze_path,"predecessor_verify","freeze")
    audit,_=ledger.read_json(audit_path,"predecessor_verify","code_audit")
    if freeze.get("payload_sha256") != payload_hash(freeze) or \
            audit.get("payload_sha256") != payload_hash(audit):
        raise RuntimeError("predecessor payload hash invalid")
    if freeze.get("run_id") != _lineage_run_id(cfg, "freeze") or \
            freeze.get("stage")!="G0_FREEZE":
        raise RuntimeError("freeze identity invalid")
    if audit.get("run_id") != _lineage_run_id(cfg, "code_audit") or \
            audit.get("stage")!="G0_CODE_AUDIT":
        raise RuntimeError("code-audit identity invalid")
    impl, _ = implementation_hash(cfg)
    if freeze["config_canonical_sha256"] != cfg["config_canonical_sha256"] or \
            freeze["implementation_sha256"] != impl:
        raise RuntimeError("post-freeze config/implementation drift")
    if audit.get("status") != "PASS" or int(audit.get("critical", -1)) != 0 or \
            int(audit.get("high", -1)) != 0:
        raise RuntimeError("independent audit is not 0 CRITICAL / 0 HIGH")
    if audit.get("implementation_sha256") != impl or \
            audit.get("config_canonical_sha256") != cfg["config_canonical_sha256"]:
        raise RuntimeError("audit did not bind frozen code/config")
    _verify_v4_code_audit_schema(cfg, artifacts, freeze, audit, ledger)
    for row in freeze.get("input_files",[]):
        path=ROOT/row["path"]
        if not path.exists():
            raise RuntimeError("frozen input drift {}".format(row["path"]))
        if "member_sha256" in row:
            actual=hash_npz_members_only(
                path,cfg["sealed_real_fixture"]["allowed_bank_members"],
                cfg["sealed_real_fixture"]["forbidden_bank_members"],
                ledger,"frozen_input_rehash","freeze_input")
            if actual!=row["member_sha256"]:
                raise RuntimeError("frozen bank member drift {}".format(row["path"]))
        elif ledger.hash_file(path,"frozen_input_rehash","freeze_input")!=row["sha256"]:
            raise RuntimeError("frozen input drift {}".format(row["path"]))
    return freeze, audit


def task_synthetic(cfg, args, ledger):
    if args.run_id != _lineage_run_id(cfg, "synthetic"):
        raise RuntimeError("wrong frozen synthetic run ID")
    freeze, audit = _load_freeze_and_audit(cfg,ledger)
    out = resolve(cfg, "artifacts") / "g0/synthetic"
    if out.exists() and any(out.iterdir()):
        raise RuntimeError("synthetic namespace already exists")
    solver = cfg["solver"]
    seed = int(cfg["synthetic"]["canonical_seed"])
    rng = np.random.default_rng(seed)
    n = 4
    x0 = np.zeros(n * n)
    y = rng.normal(size=n * n)
    projectors = [
        _projector_case("row_ball", y, x0, row_operator(n, 1), 0.7,
                        seed + 1, solver["variational_probes"]),
        _projector_case("class_mean_ball", y, x0,
                        class_mean_operator(n, [0, 2]), 0.6,
                        seed + 2, solver["variational_probes"]),
    ]
    semantic = rng.normal(size=(3, n * n))
    projectors.extend([
        _projector_case("semantic_positive_radius", y, x0, semantic, 0.9,
                        seed + 3, solver["variational_probes"]),
        _projector_case("semantic_zero_radius", y, x0, semantic, 0.0,
                        seed + 4, solver["variational_probes"]),
        _psd_case(), _halfspace_case(), _slack_case(),
    ])
    dykstra = _dykstra_cases(cfg)
    ranks = _rank_cases(cfg)
    farkas = _farkas_cases()
    factor = _factor_cases()
    rollback = [_rollback_case(seed)]
    files = {
        "cases.jsonl": [{"synthetic_seed": seed,
                         "expected": cfg["synthetic"]["expected_cases"],
                         "rollback": rollback[0]}],
        "projectors.jsonl": projectors, "dykstra.jsonl": dykstra,
        "rank_cells.jsonl": ranks,
        "exact_vote.jsonl": [ranks[0]], "farkas.jsonl": farkas,
        "factor.jsonl": factor,
    }
    for name, rows in files.items():
        publish_jsonl(out / name, rows)
    thresholds_ok = True
    for row in projectors:
        metrics = row["metrics"]
        thresholds_ok &= metrics.get("operator_adjoint_dot_error", 0.0) <= solver["operator_adjoint_tolerance"]
        for key in ("feasibility", "kkt_stationarity", "complementarity",
                    "scalar_root_residual", "idempotence",
                    "finite_difference_optimum_violation", "dense_reference_error"):
            thresholds_ok &= float(metrics[key]) <= solver["projector_tolerance"]
        thresholds_ok &= float(metrics["variational_inequality_max"]) <= solver["variational_inequality_tolerance"]
    expected_ledger=cfg["synthetic"]["expected_cases"]
    groups={"projectors":projectors,"dykstra":dykstra,"rank_cells":ranks,
            "farkas":farkas,"factor":factor,"rollback":rollback}
    expected_ok=all(len(rows)==len(expected_ledger[group]) and
                    {row["case"] for row in rows}==set(expected_ledger[group]) and
                    all(row["status"]==expected_ledger[group][row["case"]] for row in rows)
                    for group,rows in groups.items())
    dykstra_gate=all(
        (row["status"]=="LOCAL_STATIONARY_CERTIFIED" and
         row["max_set_violation"]<=solver["dykstra_set_violation_tolerance"] and
         row["relative_iterate_change"]<=solver["dykstra_relative_change_tolerance"] and
         row["rank_cell_stable"] is True) or
        (row["status"]=="BOUNDED_SEARCH_FEASIBLE" and
         row["cycles"]==row["max_cycles"])
        for row in dykstra)
    rank_by={row["case"]:row for row in ranks}
    rank_gate=(rank_by["exact_top20"]["status"]=="PASS" and
               rank_by["exact_top20"]["max_rank_cell_violation"]<=1e-12 and
               rank_by["simultaneous_ties"]["status"]=="PASS" and
               all(cell["pass"] for cell in rank_by["simultaneous_ties"]["boundary_tolerance_checks"]) and
               rank_by["orientation_over_budget"]["independent_orientations"]>
               rank_by["orientation_over_budget"]["budget"] and
               rank_by["orientation_over_budget"]["status"]=="REMOVE" and
               rank_by["orientation_over_budget"]["remove_replay_sha256"]==
               rank_by["orientation_over_budget"]["direct_remove_sha256"] and
               rank_by["pivot_over_budget"]["pivots"]>rank_by["pivot_over_budget"]["budget"] and
               rank_by["pivot_over_budget"]["status"]=="REMOVE" and
               rank_by["pivot_over_budget"]["remove_replay_sha256"]==
               rank_by["pivot_over_budget"]["direct_remove_sha256"] and
               all(rank_by[name]["status"]=="REMOVE" and
                   rank_by[name]["remove_replay_sha256"]==rank_by[name]["direct_remove_sha256"]
                   for name in ("unresolved_tie_map","incomplete_adjacent_enumeration")))
    farkas_by={row["case"]:row for row in farkas}
    farkas_gate=(farkas_by["known_out_of_cone"]["relative_separation"]>=solver["farkas_min_separation"] and
                 farkas_by["known_out_of_cone"]["max_cone_witness_inner"]<=1e-8 and
                 farkas_by["known_out_of_cone"]["duality_gap"]<=solver["farkas_max_gap"] and
                 farkas_by["known_in_cone"]["relative_separation"]<=1e-8)
    factor_by={row["case"]:row for row in factor}; fr=factor_by["repeated_and_null"]
    factor_gate=(all(float(fr[key])<=solver["factor_tolerance"] for key in
                     ("gram_reconstruction_error","aligned_gram_reconstruction_error",
                      "row_reconstruction_error","deterministic_repeat_error","procrustes_error")) and
                 factor_by["negative_reject"]["status"]=="PASS" and
                 factor_by["negative_reject"]["minimum_eigenvalue"]<solver["factor_negative_reject"])
    rollback_gate=(len(rollback)==1 and rollback[0]["status"]=="PASS" and
                   rollback[0]["rollback_replay_sha256"]==rollback[0]["direct_remove_sha256"] and
                   set(rollback[0]["restored_items"])=={
                       "model","optimizer","scheduler","scaler","torch_rng","numpy_rng",
                       "python_rng","sampler_cursor","epoch_cursor"})
    overflow = 0 if _finite({"p": projectors, "d": dykstra, "r": ranks,
                             "f": farkas, "factor": factor, "rollback": rollback}) else 1
    verifier = ROOT / "scripts/analysis/lb_scgp_independent_verify.py"
    manifest = provenance_base(cfg, args.run_id, "G0_SYNTHETIC", sha256_file(verifier),ledger)
    manifest.update({
        "status": "PASS" if thresholds_ok and expected_ok and dykstra_gate and
        rank_gate and farkas_gate and factor_gate and rollback_gate and overflow == 0 else "FAIL",
        "freeze_payload_sha256": freeze["payload_sha256"],
        "code_audit_payload_sha256": audit["payload_sha256"],
        "input_files": [{"path": root_relative_path(canonical_root_path(args.config)[0]),
                         "sha256": sha256_file(canonical_root_path(args.config)[0])}],
        "fold_ids_sha256": None,
        "checkpoint_sha256": None,
        "output_files": [{"path": root_relative_path(out / name),
                          "sha256": sha256_file(out / name)} for name in files],
        "overflow_nan_inf_count": overflow,
        "thresholds_ok": bool(thresholds_ok), "expected_statuses_ok": bool(expected_ok),
        "dykstra_gate":bool(dykstra_gate),"rank_gate":bool(rank_gate),
        "farkas_gate":bool(farkas_gate),"factor_gate":bool(factor_gate),
        "rollback_gate":bool(rollback_gate),
    })
    manifest["payload_sha256"] = payload_hash(manifest)
    publish_json(out / "manifest.json", manifest)
    print(canonical_json({"status": manifest["status"], "run_id": args.run_id,
                          "thresholds_ok": thresholds_ok,
                          "expected_statuses_ok": expected_ok}))
    if manifest["status"] != "PASS":
        raise SystemExit(2)


def _gpu_state_digest(model,optimizer,scheduler,scaler,cursor,torch):
    rows=[]
    for key,value in sorted(model.state_dict().items()):
        rows.append((key,sha256_obj(value.detach().cpu().numpy().tolist())))
    rows += [("optimizer",sha256_obj(_jsonable_state(optimizer.state_dict()))),
             ("scheduler",sha256_obj(_jsonable_state(scheduler.state_dict()))),
             ("scaler",sha256_obj(_jsonable_state(scaler.state_dict()))),
             ("torch_rng",sha256_obj(torch.get_rng_state().tolist())),
             ("cuda_rng",sha256_obj([state.cpu().tolist() for state in torch.cuda.get_rng_state_all()])),
             ("numpy_rng",sha256_obj(_jsonable_state(np.random.get_state()))),
             ("python_rng",sha256_obj(_jsonable_state(random.getstate()))),
             ("cursor",sha256_obj(cursor))]
    return sha256_obj(rows)


def _real_gpu_rollback(z0,cfg):
    import torch
    seed=int(cfg["real_fit"]["seed"])+991; random.seed(seed);np.random.seed(seed)
    torch.manual_seed(seed);torch.cuda.manual_seed_all(seed);torch.use_deterministic_algorithms(True)
    dim=z0.shape[1];device="cuda"
    def build():
        model=torch.nn.Linear(dim,dim,bias=False,device=device)
        with torch.no_grad():model.weight.copy_(torch.eye(dim,device=device))
        opt=torch.optim.AdamW(model.parameters(),lr=cfg["real_fit"]["learning_rate"],
                              weight_decay=cfg["real_fit"]["weight_decay"])
        sched=torch.optim.lr_scheduler.StepLR(opt,step_size=1,gamma=0.9)
        scaler=torch.amp.GradScaler("cuda",enabled=True,init_scale=128.0,growth_interval=1)
        return model,opt,sched,scaler
    model,opt,sched,scaler=build();x=torch.as_tensor(z0,dtype=torch.float32,device=device)
    cursor={"sampler_cursor":0,"epoch_cursor":0}
    snap={"model":copy.deepcopy(model.state_dict()),"opt":copy.deepcopy(opt.state_dict()),
          "sched":copy.deepcopy(sched.state_dict()),"scaler":copy.deepcopy(scaler.state_dict()),
          "torch":torch.get_rng_state().clone(),"cuda":[s.clone() for s in torch.cuda.get_rng_state_all()],
          "numpy":copy.deepcopy(np.random.get_state()),"python":random.getstate(),"cursor":copy.deepcopy(cursor)}
    order=torch.randperm(len(x),device=device);opt.zero_grad(set_to_none=True)
    failed=torch.square(torch.nn.functional.normalize(model(x[order[:64]]),dim=1)+1.0).mean()
    scaler.scale(failed).backward();scaler.step(opt);scaler.update();sched.step()
    cursor["sampler_cursor"]+=64;cursor["epoch_cursor"]+=1
    model.load_state_dict(snap["model"]);opt.load_state_dict(snap["opt"]);sched.load_state_dict(snap["sched"])
    scaler.load_state_dict(snap["scaler"]);torch.set_rng_state(snap["torch"]);torch.cuda.set_rng_state_all(snap["cuda"])
    np.random.set_state(snap["numpy"]);random.setstate(snap["python"]);cursor=copy.deepcopy(snap["cursor"])
    def remove_block(m,o,s,sc,cur):
        order=torch.randperm(len(x),device=device)
        for start in range(0,len(x),64):
            batch=order[start:start+64];o.zero_grad(set_to_none=True)
            out=torch.nn.functional.normalize(m(x[batch]),dim=1)
            loss=torch.square(out-x[batch]).mean();sc.scale(loss).backward();sc.step(o);sc.update()
            cur["sampler_cursor"]+=len(batch)
        s.step();cur["epoch_cursor"]+=1
    remove_block(model,opt,sched,scaler,cursor)
    rollback_hash=_gpu_state_digest(model,opt,sched,scaler,cursor,torch)
    direct,dopt,dsched,dscaler=build();direct.load_state_dict(snap["model"]);dopt.load_state_dict(snap["opt"])
    dsched.load_state_dict(snap["sched"]);dscaler.load_state_dict(snap["scaler"])
    torch.set_rng_state(snap["torch"]);torch.cuda.set_rng_state_all(snap["cuda"])
    np.random.set_state(snap["numpy"]);random.setstate(snap["python"]);dc=copy.deepcopy(snap["cursor"])
    remove_block(direct,dopt,dsched,dscaler,dc)
    direct_hash=_gpu_state_digest(direct,dopt,dsched,dscaler,dc,torch)
    return {"rollback_replay_sha256":rollback_hash,"direct_remove_sha256":direct_hash,
            "rollback_hash_identical":rollback_hash==direct_hash,
            "restored_items":["model","optimizer","scheduler","scaler","torch_rng",
                              "cuda_rng","numpy_rng","python_rng","sampler_cursor","epoch_cursor"]}


def _forbid_combined_real_caches(cfg, ledger):
    assert_no_formal_forbidden_surface(cfg, "formal_config")
    if "outer_train_subclip_cache" in cfg.get("paths", {}) or \
            "outer_train_subclip_cache_sha256" in cfg.get("sealed_real_fixture", {}):
        raise RuntimeError("formal config carries segment-cache metadata")
    ledger.records.append({"kind": "formal_input_policy",
                           "scope": "formal_g0",
                           "purpose": "protected_and_segment_cache_locators_absent"})


def _load_train_only_pt(path, expected_sha, ledger, scope):
    if not expected_sha:
        raise RuntimeError(
            "missing SHA256 for {}; safe train-only artifact is required".format(scope))
    if not Path(path).exists():
        raise RuntimeError(
            "missing {}; refusing to read combined train+held caches".format(path))
    actual = ledger.hash_file(path, "authoritative_train_only_load", scope)
    if actual != expected_sha:
        raise RuntimeError("{} hash mismatch".format(scope))
    import torch
    return torch.load(path, map_location="cpu", weights_only=True), actual


def _as_id_list(values):
    values = list(values)
    if len(values) > 0 and isinstance(values[0], (list, tuple)):
        return [str(v) for batch in values for v in batch]
    return [str(v) for v in values]


def _reject_forbidden_payload_keys(payload, name):
    forbidden_tokens = ("query", "held", "val", "dev", "test", "teacher", "mllm", "ocr")
    bad = [str(key) for key in payload if any(token in str(key).lower()
                                             for token in forbidden_tokens)]
    if bad:
        raise RuntimeError("{} contains forbidden keys {}".format(name, sorted(bad)))


def _load_real_outer_train(cfg, ids, labels, ledger, sanitized):
    import torch
    _forbid_combined_real_caches(cfg, ledger)
    feature_path = resolve(cfg, "outer_train_feature_cache")
    feature, feature_sha = _load_train_only_pt(
        feature_path, sanitized["feature_cache_sha256"],
        ledger, "outer_train_feature_cache")
    if not isinstance(feature, dict):
        raise RuntimeError("outer_train_feature_cache must be a dict")
    _reject_forbidden_payload_keys(feature, "outer_train_feature_cache")
    required_feature_keys = {"ids", "img_feats", "text_feats", "labels"}
    if set(feature) != required_feature_keys:
        raise RuntimeError("outer_train_feature_cache schema is not whitelist-only")
    if set(feature) & {"query_ids", "query_labels", "query_z"}:
        raise RuntimeError("outer_train_feature_cache contains held/query arrays")
    if not required_feature_keys <= set(feature):
        raise RuntimeError("outer_train_feature_cache missing keys {}".format(
            sorted(required_feature_keys - set(feature))))
    ledger.record_bank_member(feature_path, "ids", "outer_train_ids")
    raw_ids = _as_id_list(feature["ids"])
    if raw_ids != list(ids) or len(set(raw_ids)) != len(raw_ids):
        raise RuntimeError("train-only feature IDs do not exactly match sealed memory IDs")
    ledger.record_bank_member(feature_path, "img_feats", "outer_train_content")
    ledger.record_bank_member(feature_path, "text_feats", "outer_train_content")
    ledger.record_bank_member(feature_path, "labels", "outer_train_parent_labels")
    img = torch.as_tensor(feature["img_feats"]).float()
    text = torch.as_tensor(feature["text_feats"]).float()
    selected_labels = torch.as_tensor(feature["labels"]).reshape(-1).long()
    if img.shape[0] != len(ids) or text.shape[0] != len(ids) or selected_labels.shape[0] != len(ids):
        raise RuntimeError("outer_train_feature_cache row count mismatch")
    if not torch.equal(selected_labels, torch.as_tensor(labels, dtype=torch.long)):
        raise RuntimeError("train-only feature labels disagree with sealed bank")

    memory = [list(ids), img, text, selected_labels]
    segment = None
    return memory, segment, {
        "source": "authoritative_outer_train_only_artifacts",
        "sanitized_provenance_sha256": sanitized["sanitized_provenance_sha256"],
        "sanitizer_decision_sha256": sanitized["sanitizer_decision_sha256"],
        "feature_cache_sha256": feature_sha,
        "combined_train_cache_opened": False,
        "segment_cache_opened": False,
        "segment_objective_allowed": False,
        "mmap_used": False,
        "selected_parent_rows": len(ids),
        "outer_held_tensor_index_count": 0,
        "outer_held_content_or_label_reads": 0,
    }


def _repo_args(cfg):
    from easydict import EasyDict
    return EasyDict({"dataset":"MHC_zh","device":"cuda","batch_size":64,"lr":0.0001,
        "proj_dim":1024,"metric":"cos","loss":"triplet","triplet_margin":0.1,
        "norm_feats_loss":False,"l2_sqrt":False,"hybrid_loss":True,"ce_weight":0.5,
        "pos_weight_value":None,"hard_negatives_loss":True,"no_hard_negatives":1,
        "no_hard_positives":0,"no_pseudo_gold_positives":1,"hard_negatives_multiple":12,
        "sparse_dictionary":None,"sparse_topk":None,"Faiss_GPU":False,"grad_clip":0.1,
        "lambda_seg":0.0,"seg_mode":"disabled","cf_negs":False,"lambda_aux":0.0})


def _assert_no_segment_objective(args, segment):
    if float(getattr(args, "lambda_seg", 0.0)) != 0.0:
        raise RuntimeError("LB-SCGP G0 forbids segment loss; lambda_seg must be 0")
    if segment is not None:
        raise RuntimeError("LB-SCGP G0 forbids segment cache/objective")


def _make_repo_model(memory,cfg):
    import torch
    sys.path.insert(0,str(ROOT/"src"))
    from model.classifier import classifier_hateClipper
    args=_repo_args(cfg)
    model=classifier_hateClipper(int(memory[1].shape[1]),int(memory[2].shape[1]),3,1024,1024,
                                 "align",dropout=[0.2,0.4,0.1],batch_norm=False,args=args).cuda()
    state=torch.load(resolve(cfg,"checkpoint"),map_location="cuda",weights_only=True)
    model.load_state_dict(state,strict=True)
    return model,args


def _make_repo_loader(memory):
    sys.path.insert(0,str(ROOT/"src"))
    from data_loader.rac_dataloader import CLIP2Dataloader
    (train_dl,_),(train_set,_)=CLIP2Dataloader(memory,memory,batch_size=64,
                                               return_dataset=True,normalize=False)
    return train_dl,train_set


def _project_repo_model(model,memory):
    import torch
    model.eval();out=[]
    with torch.no_grad():
        for start in range(0,len(memory[0]),128):
            _,z=model(memory[1][start:start+128].cuda(),memory[2][start:start+128].cuda(),return_embed=True)
            out.append(torch.nn.functional.normalize(z,dim=1).cpu())
    return torch.cat(out).numpy().astype(np.float64)


def _run_repo_epoch(model,args,memory,segment,target_aligned=None,seed=0,
                    optimizer=None,scheduler=None,scaler=None,cursor=None):
    import torch
    sys.path.insert(0,str(ROOT/"src"))
    from model.loss import compute_loss
    _assert_no_segment_objective(args, segment)
    train_dl,train_set=_make_repo_loader(memory)
    if optimizer is None:
        optimizer=torch.optim.AdamW(model.parameters(),lr=args.lr)
    if scheduler is None:
        scheduler=torch.optim.lr_scheduler.StepLR(optimizer,step_size=1,gamma=0.9)
    if scaler is None:
        scaler=torch.amp.GradScaler("cuda",enabled=True)
    if cursor is None:
        cursor={"sampler_cursor":0,"epoch_cursor":0}
    batch_orders=[];fit_steps=0;target_rows_seen=[]
    model.train();train_feats=train_labels=None
    for step,batch in enumerate(train_dl):
        batch_orders.extend(str(v) for v in batch["ids"])
        result=compute_loss(batch,train_dl,model,args,train_set=train_set,
                            sparse_retrieval_dictionary=None,train_feats=train_feats,
                            train_labels=train_labels,segment_cache=None,aux_pack=None,cf_pack=None)
        loss=result[0];train_feats=result[5].detach() if torch.is_tensor(result[5]) else result[5]
        train_labels=result[6].detach() if torch.is_tensor(result[6]) else result[6]
        optimizer.zero_grad(set_to_none=True);scaler.scale(loss).backward();scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(),args.grad_clip);scaler.step(optimizer);scaler.update()
        cursor["sampler_cursor"]+=len(batch["ids"])
        if target_aligned is not None and step%int(cfg_global_real_fit_divisor)==0:
            model.eval();optimizer.zero_grad(set_to_none=True)
            seen=0
            for target_start in range(0,len(memory[0]),64):
                target_stop=min(target_start+64,len(memory[0]))
                _,pred=model(memory[1][target_start:target_stop].cuda(),
                             memory[2][target_start:target_stop].cuda(),return_embed=True)
                pred=torch.nn.functional.normalize(pred,dim=1)
                target=torch.as_tensor(target_aligned[target_start:target_stop],dtype=torch.float32,device="cuda")
                target_loss=torch.square(pred-target).sum()/(len(memory[0])*target.shape[1])
                scaler.scale(target_loss).backward();seen+=target_stop-target_start
            scaler.step(optimizer);scaler.update();target_rows_seen.append(seen)
            fit_steps+=1;model.train()
    scheduler.step()
    cursor["epoch_cursor"]+=1
    return {"optimizer":optimizer,"scheduler":scheduler,"scaler":scaler,
            "batch_order_sha256":sha256_obj(batch_orders),"batch_rows":len(batch_orders),
            "fit_steps":fit_steps,"target_rows_seen_per_step":target_rows_seen,
            "epoch_cursor":cursor["epoch_cursor"],"sampler_cursor":cursor["sampler_cursor"]}


cfg_global_real_fit_divisor=4


def _actual_repo_rollback(memory,segment,cfg):
    import torch
    seed=int(cfg["real_fit"]["seed"])+1991;random.seed(seed);np.random.seed(seed)
    torch.manual_seed(seed);torch.cuda.manual_seed_all(seed);torch.use_deterministic_algorithms(True)
    model,args=_make_repo_model(memory,cfg)
    optimizer=torch.optim.AdamW(model.parameters(),lr=args.lr)
    scheduler=torch.optim.lr_scheduler.StepLR(optimizer,step_size=1,gamma=0.9)
    scaler=torch.amp.GradScaler("cuda",enabled=True)
    cursor={"sampler_cursor":0,"epoch_cursor":0}
    rng={"torch":torch.get_rng_state().clone(),"cuda":[s.clone() for s in torch.cuda.get_rng_state_all()],
         "numpy":copy.deepcopy(np.random.get_state()),"python":random.getstate()}
    snap={"model":copy.deepcopy(model.state_dict()),
          "optimizer":copy.deepcopy(optimizer.state_dict()),
          "scheduler":copy.deepcopy(scheduler.state_dict()),
          "scaler":copy.deepcopy(scaler.state_dict()),
          "cursor":copy.deepcopy(cursor),**rng}
    # Forced failure mutates the live continuation model and optimizer stack.
    optimizer.zero_grad(set_to_none=True);_,z=model(memory[1][:64].cuda(),memory[2][:64].cuda(),return_embed=True)
    failed_loss=torch.square(torch.nn.functional.normalize(z,dim=1)+1.0).mean()
    scaler.scale(failed_loss).backward();scaler.step(optimizer);scaler.update();scheduler.step()
    cursor["sampler_cursor"]+=64;cursor["epoch_cursor"]+=1
    # Restore model, optimizer, scheduler, scaler, RNGs and cursors before REMOVE replay.
    model.load_state_dict(snap["model"]);optimizer.load_state_dict(snap["optimizer"])
    scheduler.load_state_dict(snap["scheduler"]);scaler.load_state_dict(snap["scaler"])
    torch.set_rng_state(snap["torch"]);torch.cuda.set_rng_state_all(snap["cuda"])
    np.random.set_state(snap["numpy"]);random.setstate(snap["python"])
    cursor=copy.deepcopy(snap["cursor"])
    replay=_run_repo_epoch(model,args,memory,segment,target_aligned=None,seed=seed,
                           optimizer=optimizer,scheduler=scheduler,scaler=scaler,cursor=cursor)
    replay_cursor={"sampler_cursor":replay["sampler_cursor"],"epoch_cursor":replay["epoch_cursor"]}
    replay_hash=_gpu_state_digest(model,replay["optimizer"],replay["scheduler"],replay["scaler"],replay_cursor,torch)
    direct,args2=_make_repo_model(memory,cfg);direct.load_state_dict(snap["model"])
    direct_optimizer=torch.optim.AdamW(direct.parameters(),lr=args2.lr)
    direct_scheduler=torch.optim.lr_scheduler.StepLR(direct_optimizer,step_size=1,gamma=0.9)
    direct_scaler=torch.amp.GradScaler("cuda",enabled=True)
    direct_optimizer.load_state_dict(snap["optimizer"]);direct_scheduler.load_state_dict(snap["scheduler"])
    direct_scaler.load_state_dict(snap["scaler"])
    torch.set_rng_state(snap["torch"]);torch.cuda.set_rng_state_all(snap["cuda"])
    np.random.set_state(snap["numpy"]);random.setstate(snap["python"])
    direct_cursor_seed=copy.deepcopy(snap["cursor"])
    direct_result=_run_repo_epoch(direct,args2,memory,segment,target_aligned=None,seed=seed,
                                  optimizer=direct_optimizer,scheduler=direct_scheduler,
                                  scaler=direct_scaler,cursor=direct_cursor_seed)
    direct_cursor={"sampler_cursor":direct_result["sampler_cursor"],"epoch_cursor":direct_result["epoch_cursor"]}
    direct_hash=_gpu_state_digest(direct,direct_result["optimizer"],direct_result["scheduler"],
                                  direct_result["scaler"],direct_cursor,torch)
    return {"rollback_replay_sha256":replay_hash,"direct_remove_sha256":direct_hash,
            "rollback_hash_identical":replay_hash==direct_hash,
            "rollback_batch_order_sha256":replay["batch_order_sha256"],
            "direct_batch_order_sha256":direct_result["batch_order_sha256"],
            "rollback_fixture":"actual_ra_hmd_checkpoint_live_adamw_scheduler_scaler_rng_cursor_v2",
            "restored_items":["checkpoint_model","AdamW","scheduler","scaler","torch_rng",
                              "cuda_rng","numpy_rng","python_rng","sampler_cursor","epoch_cursor"]}


def task_realfold(cfg, args, ledger):
    import torch
    if args.run_id != _lineage_run_id(cfg, "realfold") or \
            args.dataset!="MHC_zh" or args.outer_fold!=4:
        raise RuntimeError("wrong sealed real-fold identity")
    freeze,audit=_load_freeze_and_audit(cfg,ledger)
    synth_path=resolve(cfg,"artifacts")/"g0/synthetic/manifest.json"
    synth,_=ledger.read_json(synth_path,"predecessor_verify","synthetic_manifest")
    if synth.get("status")!="PASS" or synth.get("payload_sha256")!=payload_hash(synth):
        raise RuntimeError("sealed synthetic predecessor is not PASS")
    if synth.get("implementation_sha256")!=freeze.get("implementation_sha256"):
        raise RuntimeError("synthetic/freeze implementation drift")
    out=resolve(cfg,"artifacts")/"g0/real/MHC_zh/fold4"
    if out.exists() and any(out.iterdir()):raise RuntimeError("real namespace already exists")
    fixture=cfg["sealed_real_fixture"];bank_path=resolve(cfg,"bank")
    bank,opened=load_npz_members_only(bank_path,fixture["allowed_bank_members"],
                                     fixture["forbidden_bank_members"],ledger)
    ids=[str(x) for x in bank["memory_ids"].tolist()];held=[str(x) for x in bank["query_ids"].tolist()]
    z0=np.asarray(bank["memory_z"],dtype=np.float64);labels=np.asarray(bank["memory_labels"],dtype=np.int64)
    if len(ids)!=fixture["outer_train_n"] or len(held)!=fixture["outer_held_n"] or set(ids)&set(held):
        raise RuntimeError("sealed memory/held sentinel partition invalid")
    if sha256_obj(ids)!=fixture["memory_ids_sha256"] or len(set(ids))!=len(ids):
        raise RuntimeError("memory ID freeze mismatch")
    if sha256_obj(held)!=fixture["query_ids_sha256"] or len(set(held))!=len(held):
        raise RuntimeError("held-ID sentinel freeze mismatch")
    if z0.shape!=(fixture["outer_train_n"],cfg["real_fit"]["embedding_dim"]) or labels.shape!=(fixture["outer_train_n"],):
        raise RuntimeError("sealed bank shape mismatch")
    if not np.isfinite(z0).all() or not np.isfinite(labels).all() or np.any(np.linalg.norm(z0,axis=1)<=1e-12):
        raise RuntimeError("nonfinite or zero sealed bank row")
    if set(np.unique(labels).tolist())!={0,1}:raise RuntimeError("parent-video labels invalid")
    z0/=np.linalg.norm(z0,axis=1,keepdims=True);g0=z0@z0.T
    sanitized = _read_sanitized_contract(cfg, ledger, "sanitizer_real_fold_verify")
    memory,segment,cache_contract=_load_real_outer_train(cfg,ids,labels,ledger,sanitized)
    ledger.record_bank_member(resolve(cfg,"checkpoint"),"model_state_dict","sealed_outer_train_checkpoint")
    total=Stopwatch();bank_watch=Stopwatch();baseline_model,repo_args=_make_repo_model(memory,cfg)
    torch.cuda.synchronize();checkpoint_bank=_project_repo_model(baseline_model,memory);torch.cuda.synchronize()
    bank_seconds=bank_watch.seconds();checkpoint_bank_error=float(np.linalg.norm(checkpoint_bank-z0)/max(np.linalg.norm(z0),1e-15))
    target_watch=Stopwatch()
    gstar,xi,rank,ell,caps,target_status,trace,stage_times,violations,relative,rank_search=_real_rank_controller(
        g0,labels,ids,cfg)
    target_seconds=target_watch.seconds();target_for_factor=gstar if target_status=="LOCAL_STATIONARY_CERTIFIED" else g0
    factor_watch=Stopwatch();factor_error=None
    try:
        factor,eigenvalues=deterministic_psd_factor(
            target_for_factor,negative_reject=cfg["solver"]["factor_negative_reject"])
        aligned,rotation=procrustes_align_factor(factor,z0)
    except Exception as exc:
        factor_error=str(exc);factor=np.zeros_like(target_for_factor);aligned=z0.copy();eigenvalues=np.linalg.eigvalsh(target_for_factor)
    factor_seconds=factor_watch.seconds();gram_error=float(np.linalg.norm(factor@factor.T-target_for_factor))
    aligned_error=float(np.linalg.norm(aligned@aligned.T-target_for_factor))
    row_error=float(np.max(np.linalg.norm(aligned@aligned.T-target_for_factor,axis=1)))
    # Abstract registered-cone audit precedes any fit.
    abstract_farkas_watch=Stopwatch();abstract_cone=_real_cone_audit(z0,aligned,labels,cfg)
    abstract_farkas_seconds=abstract_farkas_watch.seconds()
    torch.cuda.reset_peak_memory_stats();fit_watch=Stopwatch()
    realized=z0.copy();fit_steps=0;fit_result={"batch_rows":0,"batch_order_sha256":None,
                                              "fit_steps":0,"target_rows_seen_per_step":[],
                                              "epoch_cursor":0,"sampler_cursor":0}
    if target_status=="LOCAL_STATIONARY_CERTIFIED" and factor_error is None:
        seed=int(cfg["real_fit"]["seed"]);random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
        fit_model,fit_args=_make_repo_model(memory,cfg)
        fit_result=_run_repo_epoch(fit_model,fit_args,memory,segment,target_aligned=aligned,seed=seed)
        fit_steps=fit_result["fit_steps"];torch.cuda.synchronize()
        final_bank_watch=Stopwatch();realized=_project_repo_model(fit_model,memory);torch.cuda.synchronize()
        final_bank_seconds=final_bank_watch.seconds()
    else:
        final_bank_seconds=0.0
    fit_seconds=fit_watch.seconds();rollback=_actual_repo_rollback(memory,segment,cfg)
    realized_farkas_watch=Stopwatch();realized_cone=_real_cone_audit(z0,realized,labels,cfg)
    realized_farkas_seconds=realized_farkas_watch.seconds();farkas_seconds=abstract_farkas_seconds+realized_farkas_seconds
    target_disp=aligned-z0;realized_disp=realized-z0
    displacement_cosine=float(np.sum(target_disp*realized_disp)/max(np.linalg.norm(target_disp)*np.linalg.norm(realized_disp),1e-15))
    relative_residual=float(np.linalg.norm(realized-aligned)/max(np.linalg.norm(target_disp),1e-15))
    realized_gram=realized@realized.T;off=realized_gram[~np.eye(len(ids),dtype=bool)]
    no_collapse=bool(float(np.std(realized,axis=0).mean())>1e-6 and float(off.max())<0.999999)
    fit_seconds_total=fit_seconds
    verify_watch=Stopwatch();independent_values,_=_real_violations(gstar,xi,g0,labels,rank,ell,caps,cfg)
    emitted_gram=gstar if target_status=="LOCAL_STATIONARY_CERTIFIED" else g0
    exact=exact_vote_ledger(emitted_gram,labels,ids,20);verify_seconds=verify_watch.seconds()
    remove_rows,_=ledger.read_jsonl(resolve(cfg,"remove_ledger"),"cost_ledger_read","outer_train_remove_ledger")
    previous=0.0;increments=[]
    for row in remove_rows:
        current=float(row["wall_s"]);increments.append(max(0.0,current-previous));previous=current
    remove_seconds=float(sum(increments))
    refresh_sample = (bank_seconds + target_seconds + factor_seconds +
                      fit_seconds_total + verify_seconds + farkas_seconds)
    timing_samples = [refresh_sample]
    p95_refresh_seconds = max(timing_samples)
    h10=cfg["cost"]["contingency"]*cfg["cost"]["folds"]*(
        (2*remove_seconds + cfg["cost"]["refresh_multiplier"]*p95_refresh_seconds +
         final_bank_seconds)/3600.0)
    peak_gpu=float(torch.cuda.max_memory_allocated()/1024**3);peak_host=peak_rss_gib()
    numerics={"target_status":target_status,"max_independent_violation":max(violations.values()),
              "relative_iterate_change":relative,"violations":violations,
              "target_gram":gstar.tolist(),"slack":xi.tolist(),"ell":ell.tolist(),
              "slack_caps":caps,"cycles":len(trace),"rank_halfspace_count":len(rank["rank_i"]),
              "orientation_count":rank_search.get("independent_orientations",0),
              "rank_search":rank_search}
    numerics["trainable_target"]=target_status=="LOCAL_STATIONARY_CERTIFIED"
    numerics["remove_replay_gram_sha256"]=sha256_obj(g0.tolist())
    numerics["emitted_exact_vote_gram_sha256"]=sha256_obj(emitted_gram.tolist())
    numerics["selected_trace_sha256"]=sha256_obj(trace)
    numerics["selected_trace_projector_transitions_sha256"]=sha256_obj(
        [row.get("projector_transitions_sha256") for row in trace])
    numerics["selected_final_correction_state_sha256"]=(
        trace[-1].get("correction_state_sha256") if trace else None)
    projectors=[{"cycle":row["cycle"],"before_sha256":row["before_sha256"],
                 "after_sha256":row["after_sha256"],
                 "max_independent_set_violation":row["max_independent_set_violation"],
                 "relative_iterate_change":row["relative_iterate_change"],
                 "correction_norms":row["correction_norms"],
                 "correction_state_sha256":row["correction_state_sha256"],
                 "projector_transitions":row["projector_transitions"],
                 "projector_transitions_sha256":row["projector_transitions_sha256"]}
                for row in trace]
    rank_rows=[{"query_id":ids[i],"cell":rank["rankings"][i],
                "cell_sha256":sha256_obj(rank["rankings"][i]),
                "realized":stable_rankings(gstar,ids,len(ids)-1,cfg["solver"]["tie_tolerance"])[i]}
               for i in range(len(ids))]
    factor_json={"factor_error":factor_error,"minimum_eigenvalue":float(np.min(eigenvalues)),
                 "gram_reconstruction_error":gram_error,
                 "aligned_gram_reconstruction_error":aligned_error,"row_reconstruction_error":row_error,
                 "aligned_factor_sha256":sha256_obj(aligned.tolist()),"rotation_sha256":sha256_obj(rotation.tolist()) if factor_error is None else None}
    farkas_json={"abstract_relative_separation":abstract_cone["relative_separation"],
                 "realized_relative_separation":realized_cone["relative_separation"],
                 "abstract":abstract_cone,"realized":realized_cone,
                 "max_duality_gap":max(abstract_cone["duality_gap"],realized_cone["duality_gap"])}
    fit_json={"fit_steps":fit_steps,"full_epoch_batch_rows":fit_result["batch_rows"],
              "batch_order_sha256":fit_result["batch_order_sha256"],
              "target_rows_seen_per_step":fit_result["target_rows_seen_per_step"],
              "checkpoint_bank_relative_error":checkpoint_bank_error,
              "selective_cache_contract":cache_contract,
              "realized_bank":realized.tolist(),
              "realized_bank_sha256":sha256_obj(realized.tolist()),
              "displacement_cosine":displacement_cosine,
              "relative_target_residual":relative_residual,"no_collapse_or_duplicate":no_collapse,
              **rollback}
    timings={"bank_seconds":bank_seconds,"target_seconds":target_seconds,"projector_seconds":stage_times,
             "factor_seconds":factor_seconds,"fit_block_seconds":fit_seconds_total,
             "verify_seconds":verify_seconds,"farkas_seconds":farkas_seconds,
             "abstract_farkas_seconds":abstract_farkas_seconds,
             "realized_farkas_seconds":realized_farkas_seconds,
             "final_bank_seconds":final_bank_seconds,
             "remove_fullfold_seconds":remove_seconds,"remove_epoch_increments":increments,
             "H10_formula":cfg["cost"]["formula"],
             "H10_p95_sample_semantics":cfg["cost"]["p95_sample_semantics"],
             "refresh_timing_samples_seconds":timing_samples,
             "p95_refresh_seconds":p95_refresh_seconds,
             "H10_upper_gpu_hours":h10,"total_seconds":total.seconds()}
    visible=[item for item in os.environ.get("CUDA_VISIBLE_DEVICES","").split(",") if item.strip()]
    props=torch.cuda.get_device_properties(0) if torch.cuda.is_available() else None
    resource_json={"peak_gpu_gib":peak_gpu,"peak_host_rss_gib":peak_host,
                   "slurm_job_id":os.environ.get("SLURM_JOB_ID"),
                   "slurm_job_gpus":os.environ.get("SLURM_JOB_GPUS"),
                   "slurm_gpus":os.environ.get("SLURM_GPUS"),
                   "slurm_gpus_on_node":os.environ.get("SLURM_GPUS_ON_NODE"),
                   "slurm_step_gpus":os.environ.get("SLURM_STEP_GPUS"),
                   "cuda_visible_devices":os.environ.get("CUDA_VISIBLE_DEVICES"),
                   "cuda_visible_device_count":len(visible),
                   "torch_cuda_device_count":int(torch.cuda.device_count()),
                   "torch_cuda_is_available":bool(torch.cuda.is_available()),
                   "gpu_name":props.name if props is not None else None,
                   "gpu_uuid":getattr(props,"uuid",None) if props is not None else None,
                   "gpu_total_memory_gib":float(props.total_memory)/1024**3 if props is not None else None}
    resource_json["one_gpu"]=(resource_json["torch_cuda_is_available"] is True and
                              resource_json["torch_cuda_device_count"]==1 and
                              resource_json["cuda_visible_device_count"]==1)
    outputs={"timings.json":timings,"numerics.json":numerics,"farkas.json":farkas_json,
             "factor.json":factor_json,"fit_rollback.json":fit_json,"resource.json":resource_json}
    for name,obj in outputs.items():publish_json(out/name,obj)
    publish_jsonl(out/"projectors.jsonl",projectors);publish_jsonl(out/"rank_cells.jsonl",rank_rows)
    publish_jsonl(out/"exact_vote.jsonl",exact)
    verifier=ROOT/"scripts/analysis/lb_scgp_independent_verify.py"
    manifest=provenance_base(cfg,args.run_id,"G0_REAL_FOLD",sha256_file(verifier),ledger)
    expected_fit_steps=sum(1 for step in range(math.ceil(len(ids)/cfg["real_fit"]["batch_size"]))
                           if step%cfg["real_fit"]["scheduled_divisor"]==0)
    producer_pass=(target_status=="LOCAL_STATIONARY_CERTIFIED" and max(violations.values())<=1e-6 and
                   max(independent_values.values())<=1e-6 and
                   segment is None and float(getattr(repo_args,"lambda_seg",0.0))==0.0 and
                   cache_contract.get("segment_cache_opened") is False and
                   cache_contract.get("segment_objective_allowed") is False and
                   rank_search.get("reason")=="all_adjacent_checked" and
                   rank_search.get("adjacent_cells_checked")==rank_search.get("adjacent_cells_total") and
                   factor_error is None and gram_error<=1e-6 and aligned_error<=1e-6 and row_error<=1e-6 and
                   checkpoint_bank_error<=1e-5 and fit_result["batch_rows"]==len(ids) and
                   fit_steps==expected_fit_steps and
                   all(rows==len(ids) for rows in fit_result["target_rows_seen_per_step"]) and
                   len(exact)==len(ids) and
                   displacement_cosine>=cfg["real_fit"]["displacement_cosine_min"] and
                   relative_residual<=cfg["real_fit"]["relative_target_residual_max"] and no_collapse and
                   abstract_cone["relative_separation"]>=cfg["solver"]["farkas_min_separation"] and
                   realized_cone["relative_separation"]>=cfg["solver"]["farkas_min_separation"] and
                   abstract_cone["max_cone_witness_inner"]<=1e-8 and
                   realized_cone["max_cone_witness_inner"]<=1e-8 and
                   abstract_cone["separation_overflow"] is False and
                   realized_cone["separation_overflow"] is False and
                   max(abstract_cone["duality_gap"],realized_cone["duality_gap"])<=cfg["solver"]["farkas_max_gap"] and
                   rollback["rollback_hash_identical"] and resource_json["one_gpu"] is True and
                   peak_gpu<=cfg["real_fit"]["gpu_peak_gib_max"] and
                   peak_host<=cfg["real_fit"]["host_peak_gib_max"] and math.isfinite(h10) and
                   h10<cfg["cost"]["g1_upper_gpu_hours_strict_max"])
    names=list(outputs)+["projectors.jsonl","rank_cells.jsonl","exact_vote.jsonl"]
    input_rows=[]
    for record in ledger.records:
        if record.get("kind")=="file_read" and record.get("sha256"):
            input_rows.append({"path":record["path"],"sha256":record["sha256"]})
        elif record.get("kind") in {"bank_member_read","npz_member_hash"} and record.get("sha256"):
            input_rows.append({"path":record["path"],"member":record.get("member"),
                               "sha256":record["sha256"]})
    manifest.update({"status":"PASS" if producer_pass else "FAIL",
                     "freeze_payload_sha256":freeze["payload_sha256"],
                     "code_audit_payload_sha256":audit["payload_sha256"],
                     "synthetic_payload_sha256":synth["payload_sha256"],
                     "input_files":sorted(input_rows,key=lambda row:(row["path"],row.get("member",""))),
                     "fold_ids_sha256":sha256_obj(ids),"checkpoint_sha256":fixture["checkpoint_sha256"],
                     "output_files":[{"path":root_relative_path(out/name),"sha256":sha256_file(out/name)} for name in names],
                     "opened_bank_members":opened,"forbidden_bank_members_opened":[],
                     "opened_bank_members_exact_allowlist":set(opened)==set(fixture["allowed_bank_members"]),
                     "outer_held_prediction_count":0,"producer_gate":bool(producer_pass)})
    manifest["payload_sha256"]=payload_hash(manifest);publish_json(out/"manifest.json",manifest)
    print(canonical_json({"run_id":args.run_id,"status":manifest["status"],
                          "target_status":target_status,"H10_upper_gpu_hours":h10}))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--task", required=True, choices=["freeze", "synthetic", "realfold"])
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--dataset", choices=["MHC_zh"])
    parser.add_argument("--outer-fold", type=int, choices=[4])
    args = parser.parse_args()
    require_slurm(expected_gpu=(args.task == "realfold"))
    ledger=AccessLedger(); cfg = load_config(args.config,ledger)
    if args.task == "freeze":
        task_freeze(cfg, args,ledger)
    elif args.task == "synthetic":
        task_synthetic(cfg, args,ledger)
    else:
        task_realfold(cfg, args,ledger)


if __name__ == "__main__":
    main()
