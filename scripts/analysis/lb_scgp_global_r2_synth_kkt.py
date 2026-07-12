#!/usr/bin/env python
"""Run2 synthetic KKT producer and strict semantic verifier."""
from __future__ import annotations

import argparse
import copy
import json
import math
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path("/data/jehc223/RGCL")
sys.path.insert(0, str(ROOT / "scripts/analysis"))

from lb_scgp_global_r2_common import (  # noqa: E402
    CERT_SCHEMA_ID,
    TRI_OBSERVABLES,
    AccessLedger,
    canonical_json,
    canonical_root_path,
    consensus_replicas,
    encode_certificate,
    exclusive_publish_json,
    factor_from_psd_gram,
    git_dirty_hash,
    implementation_hashes,
    old_protected_hash_manifest,
    payload_hash,
    procrustes_align,
    read_json,
    require_slurm_cpu,
    row_normalize,
    sha256_file,
    sha256_obj,
    structural_moment,
    structural_operator_summary,
    vech,
    orth_cap,
)


RUN1 = "LBSCGP-GLOBAL-G0-M0-CONTRACT-FREEZE-v1"
RUN2 = "LBSCGP-GLOBAL-G0-M0-SYNTH-KKT-v1"
SCHEMA_ID = "scgp_global_synth_kkt_payload_v1"
CASE_IDS = ("FULL", "REMOVE_NULL", "SHUFFLE", "NOISE")
INJECTION_IDS = (
    "stationarity_tamper",
    "psd_sign_flip",
    "rank_tail_corrupt",
    "segment_counter_nonzero",
    "finite_vi_acceptance_tamper",
    "payload_hash_tamper",
)


def assert_equal(actual: Any, expected: Any, label: str) -> None:
    if actual != expected:
        raise RuntimeError(f"{label} drift: expected {expected!r}, got {actual!r}")


def arr(values: Any) -> np.ndarray:
    return np.asarray(values, dtype=np.float64)


def to_list(values: np.ndarray) -> list[Any]:
    return np.asarray(values, dtype=np.float64).tolist()


def zeros(n: int, m: int | None = None) -> np.ndarray:
    if m is None:
        return np.zeros(n, dtype=np.float64)
    return np.zeros((n, m), dtype=np.float64)


def max_abs(values: np.ndarray) -> float:
    if values.size == 0:
        return 0.0
    return float(np.max(np.abs(values)))


def offdiag_values(matrix: np.ndarray) -> np.ndarray:
    n = matrix.shape[0]
    mask = ~np.eye(n, dtype=bool)
    return matrix[mask]


def normalize_rows(matrix: np.ndarray) -> np.ndarray:
    return row_normalize(np.asarray(matrix, dtype=np.float64))


def certificate_record(pattern: int, unresolved: bool = False) -> dict[str, Any]:
    tri_states = ("supported", "contradicted", "unresolved")
    modality_states = (
        "multi_modal",
        "visual_text",
        "visual_audio",
        "text_audio",
        "single_modal",
        "unresolved",
    )
    record: dict[str, Any] = {"schema_version": CERT_SCHEMA_ID}
    for idx, field in enumerate(TRI_OBSERVABLES):
        state = "unresolved" if unresolved else tri_states[(idx + pattern) % len(tri_states)]
        record[field] = {"state": state, "confidence": 0}
    modality = "unresolved" if unresolved else modality_states[pattern % len(modality_states)]
    record["modality_binding_observable"] = {"state": modality, "confidence": 0}
    record["parse_flags"] = []
    return record


def synthetic_phi(case_id: str) -> np.ndarray:
    if case_id == "REMOVE_NULL":
        records = [certificate_record(0, unresolved=True) for _ in range(6)]
    else:
        patterns = [0, 1, 2, 3, 4, 5]
        if case_id == "SHUFFLE":
            patterns = [2, 4, 1, 5, 0, 3]
        records = [certificate_record(pattern) for pattern in patterns]
        if case_id == "NOISE":
            # Keep the certificate route discrete; the geometric noise is in Y.
            records = [consensus_replicas([record, record, record]) for record in records]
    return np.stack([encode_certificate(record) for record in records], axis=0)


def unit_cone_embeddings(case_id: str) -> np.ndarray:
    if case_id == "REMOVE_NULL":
        angles = np.asarray([0.0, 0.6, 1.2, 1.8, 2.4, 3.0], dtype=np.float64)
        y = np.stack([np.cos(angles), np.sin(angles), np.zeros_like(angles)], axis=1)
    else:
        angles = np.asarray([0.0, 0.55, 1.15, 1.85, 2.55, 3.25], dtype=np.float64)
        z = 0.65
        r = math.sqrt(1.0 - z * z)
        y = np.stack(
            [r * np.cos(angles), r * np.sin(angles), np.full_like(angles, z)],
            axis=1,
        )
        if case_id == "SHUFFLE":
            y = y[[2, 4, 1, 5, 0, 3], :]
        elif case_id == "NOISE":
            mix = np.asarray(
                [
                    [0.00, 0.03, -0.01],
                    [0.02, -0.01, 0.01],
                    [-0.02, 0.01, 0.02],
                    [0.01, 0.02, -0.02],
                    [-0.01, -0.02, 0.01],
                    [0.03, 0.00, -0.01],
                ],
                dtype=np.float64,
            )
            y = normalize_rows(y + mix)
    return normalize_rows(y)


def robust_coverage(gram: np.ndarray) -> dict[str, Any]:
    eps = 1e-4
    values = offdiag_values(gram)
    stable = np.abs(values) > eps
    total = int(values.size)
    covered = int(np.sum(stable))
    return {
        "interval_radius": eps,
        "covered_offdiag_entries": covered,
        "total_offdiag_entries": total,
        "coverage": float(covered / max(total, 1)),
        "status": "PASS" if covered == total else "LOW_COVERAGE_DIAGNOSTIC_ONLY",
        "safety_claim": "disabled_if_low_coverage",
    }


def finite_vi_diagnostic(gram: np.ndarray, objective_center: np.ndarray, s_psd: np.ndarray) -> dict[str, Any]:
    probes = [
        np.eye(gram.shape[0], dtype=np.float64),
        0.5 * gram + 0.5 * np.eye(gram.shape[0], dtype=np.float64),
        gram,
    ]
    normal = -s_psd
    values = [float(np.sum(normal * (probe - gram))) for probe in probes]
    center_delta = float(np.linalg.norm(gram - objective_center))
    return {
        "used_for_acceptance": False,
        "probe_count": len(probes),
        "max_normal_cone_vi_value": max(values),
        "values": values,
        "objective_center_delta_norm": center_delta,
        "role": "diagnostic_only",
    }


def case_hash(payload: dict[str, Any]) -> str:
    clone = copy.deepcopy(payload)
    clone.setdefault("hashes", {}).pop("case_payload_sha256", None)
    return sha256_obj(clone)


def attach_case_hash(payload: dict[str, Any]) -> dict[str, Any]:
    clone = copy.deepcopy(payload)
    arrays = {
        "G_star": clone["primal"]["G_star"],
        "objective_center_G": clone["primal"]["objective_center_G"],
        "S_psd": clone["psd_normal"]["S_psd"],
        "v_psd": clone["psd_normal"]["v_psd"],
        "Q": clone["common_basis_operator"]["Q"],
    }
    clone["hashes"]["array_bundle_sha256"] = sha256_obj(arrays)
    clone["hashes"]["case_payload_sha256"] = case_hash(clone)
    return clone


def build_case_payload(case_id: str, cfg: dict[str, Any]) -> dict[str, Any]:
    thresholds = cfg["semantic_thresholds"]
    ids = [f"synth_{idx:02d}" for idx in range(6)]
    labels = [0, 1, 0, 1, 0, 1]
    d = int(thresholds["rank_d"])
    phi = synthetic_phi(case_id)
    q = orth_cap(phi, ids, rank_cap=8)
    cert_rows = normalize_rows(phi)
    k_consensus = cert_rows @ cert_rows.T
    y_ref = unit_cone_embeddings(case_id)
    gram = y_ref @ y_ref.T
    gram = 0.5 * (gram + gram.T)
    np.fill_diagonal(gram, 1.0)

    eigval, eigvec = np.linalg.eigh(gram)
    null_mask = eigval < 1e-10
    nullspace = eigvec[:, null_mask]
    if nullspace.size:
        s_psd = 0.125 * (nullspace @ nullspace.T)
    else:
        s_psd = zeros(gram.shape[0], gram.shape[0])
    s_psd = 0.5 * (s_psd + s_psd.T)
    v_psd = -s_psd
    objective_center = gram + v_psd

    m_star = structural_moment(q, gram)
    m_target = structural_moment(q, k_consensus)
    r_struct = vech(m_star - m_target)
    r_center = r_struct.copy()
    zero_g = zeros(gram.shape[0], gram.shape[0])
    zero_r = zeros(r_struct.size)
    stationarity_g = gram - objective_center + v_psd
    stationarity_r = r_struct - r_center + zero_r

    y_factor, rank_tail = factor_from_psd_gram(gram, d=d)
    if y_factor is None:
        y_factor = zeros(gram.shape[0], d)
    z_star, rotation, orth_resid = procrustes_align(y_factor, y_ref)
    operator_summary = structural_operator_summary(q, gram, k_consensus)
    box_delta = float(thresholds["box_delta"])
    offdiag = offdiag_values(gram)
    psd_eigs = np.linalg.eigvalsh(gram)
    s_eigs = np.linalg.eigvalsh(s_psd)
    robust = robust_coverage(gram)
    finite_vi = finite_vi_diagnostic(gram, objective_center, s_psd)
    q_rank = int(q.shape[1])
    expected_null = case_id == "REMOVE_NULL"
    offdiag_std = float(np.std(offdiag))
    nondegenerate = {
        "q_rank": q_rank,
        "gram_rank_eps": int(rank_tail["rank_eps"]),
        "offdiag_std": offdiag_std,
        "expected_null_control": expected_null,
        "status": (
            "EXPECTED_NULL_CONTROL"
            if expected_null and q_rank == 0
            else "PASS"
            if q_rank >= 2 and int(rank_tail["rank_eps"]) >= 2 and offdiag_std > 1e-3
            else "DEGENERATION_FAIL"
        ),
    }
    payload: dict[str, Any] = {
        "artifact_schema_id": SCHEMA_ID,
        "run_id": RUN2,
        "case_id": case_id,
        "case_role": {
            "FULL": "primary_acceptance",
            "REMOVE_NULL": "remove_null_control",
            "SHUFFLE": "shuffle_control",
            "NOISE": "noise_control",
        }[case_id],
        "primal": {
            "N": int(gram.shape[0]),
            "d": d,
            "ids": ids,
            "labels": labels,
            "G_star": to_list(gram),
            "objective_center_G": to_list(objective_center),
            "r_struct_star": to_list(r_struct),
            "objective_center_r": to_list(r_center),
            "Phi": to_list(phi),
            "K_consensus": to_list(k_consensus),
            "coordinate_trust_center_G": to_list(gram),
            "row_trust_center_G": to_list(gram),
            "class_trust_center_G": to_list(gram),
            "reference_Z0": to_list(y_ref),
        },
        "metric": {
            "G_frobenius_weight": 1.0,
            "r_struct_weight": float(thresholds["lambda_struct"]),
            "H_metric": "block_diag_identity_G_lambda_struct_r",
            "strong_convexity_min_weight": min(1.0, float(thresholds["lambda_struct"])),
            "closed_convex_set": True,
        },
        "affine_normals": {
            "normal_G": to_list(zero_g),
            "normal_r": to_list(zero_r),
            "diag_equal_one_dual": [0.0 for _ in ids],
            "symmetry_dual": "implicit_symmetric_variable_zero_dual",
            "structural_equality_dual": to_list(zero_r),
        },
        "box_coordinate_normals": {
            "normal_G": to_list(zero_g),
            "normal_r": to_list(zero_r),
            "box_delta": box_delta,
            "box_dual_min": 0.0,
            "coordinate_trust_dual_min": 0.0,
            "coordinate_trust_radius": 0.25,
            "active_count": 0,
        },
        "soc_normals": {
            "normal_G": to_list(zero_g),
            "normal_r": to_list(zero_r),
            "row_trust_dual_min": 0.0,
            "class_trust_dual_min": 0.0,
            "row_trust_radius": 0.5,
            "class_trust_radius": 0.5,
            "active_count": 0,
        },
        "psd_normal": {
            "sign_convention": "v_psd=-S_psd",
            "S_psd": to_list(s_psd),
            "v_psd": to_list(v_psd),
            "S_psd_min_eigenvalue": float(np.min(s_eigs)) if s_eigs.size else 0.0,
            "trace_SG": float(np.sum(s_psd * gram)),
        },
        "halfspace_normals": {
            "normal_G": to_list(zero_g),
            "normal_r": to_list(zero_r),
            "robust_halfspace_dual_min": 0.0,
            "active_count": 0,
        },
        "stationarity": {
            "G_residual_inf": max_abs(stationarity_g),
            "r_residual_inf": max_abs(stationarity_r),
            "max_abs": max(max_abs(stationarity_g), max_abs(stationarity_r)),
            "tolerance": float(thresholds["kkt_stationarity_inf"]),
            "accepted": max(max_abs(stationarity_g), max_abs(stationarity_r))
            <= float(thresholds["kkt_stationarity_inf"]),
        },
        "dual_feasibility": {
            "psd_S_min_eigenvalue": float(np.min(s_eigs)) if s_eigs.size else 0.0,
            "nonnegative_inequality_dual_min": 0.0,
            "tolerance": float(thresholds["dual_feasibility"]),
            "accepted": (float(np.min(s_eigs)) if s_eigs.size else 0.0)
            >= -float(thresholds["dual_feasibility"]),
        },
        "complementarity": {
            "psd_trace_SG": float(np.sum(s_psd * gram)),
            "inactive_inequality_product_max_abs": 0.0,
            "tolerance": float(thresholds["complementarity_abs"]),
            "accepted": abs(float(np.sum(s_psd * gram)))
            <= float(thresholds["complementarity_abs"]),
        },
        "duality_gap": {
            "value": 0.0,
            "accepted": True,
            "source": "analytic_normal_cone_certificate",
        },
        "primal_feasibility": {
            "symmetry_inf": max_abs(gram - gram.T),
            "unit_diag_inf": max_abs(np.diag(gram) - 1.0),
            "psd_min_eigenvalue": float(np.min(psd_eigs)),
            "offdiag_box_violation": float(
                max(0.0, np.max(np.abs(offdiag)) - (1.0 - box_delta))
            ),
            "coordinate_trust_violation": 0.0,
            "row_trust_violation": 0.0,
            "class_trust_violation": 0.0,
            "structural_residual_inf": max_abs(r_struct - vech(m_star - m_target)),
            "accepted": True,
        },
        "common_basis_operator": {
            "orth_cap_exercised": True,
            "rank_cap": 8,
            "Q": to_list(q),
            "Q_shape": [int(q.shape[0]), int(q.shape[1])],
            "M_Q_star": to_list(m_star),
            "M_Q_target": to_list(m_target),
            "M_Q_residual_vech": to_list(r_struct),
            "summary": operator_summary,
            "M_Q_star_sha256": sha256_obj(to_list(m_star)),
            "M_Q_target_sha256": sha256_obj(to_list(m_target)),
        },
        "rank_tail": rank_tail,
        "factor": {
            "status": "PASS" if rank_tail["status"] == "PASS" else "FAIL",
            "Y_shape": [int(y_factor.shape[0]), int(y_factor.shape[1])],
            "Y_sha256": sha256_obj(to_list(y_factor)),
        },
        "procrustes": {
            "Zstar_shape": [int(z_star.shape[0]), int(z_star.shape[1])],
            "rotation_shape": [int(rotation.shape[0]), int(rotation.shape[1])],
            "orthogonality_residual": orth_resid,
            "Zstar_sha256": sha256_obj(to_list(z_star)),
        },
        "nondegeneration": nondegenerate,
        "robust_interval_coverage": robust,
        "finite_vi_diagnostic": finite_vi,
        "isolation_counters": {
            "mllm_call_count": 0,
            "ocr_call_count": 0,
            "segment_gold_exists": False,
            "segment_gold_used": False,
            "segment_gold_read_count": 0,
            "validation_content_read_count": 0,
            "test_content_read_count": 0,
            "held_content_read_count": 0,
            "query_z_read_count": 0,
            "query_labels_read_count": 0,
            "teacher_cache_read_count": 0,
            "teacher_cache_write_count": 0,
        },
        "acceptance": {
            "accepted_for_run2_go": case_id == "FULL",
            "kkt_is_only_acceptance": True,
            "finite_vi_used_for_acceptance": False,
            "solver_trace_used_for_acceptance": False,
            "failure_policy": "fail_closed_no_parameter_tuning_rerun",
        },
        "hashes": {
            "hash_excludes": ["hashes.case_payload_sha256"],
        },
    }
    return attach_case_hash(payload)


def verify_case_payload(payload: dict[str, Any], cfg: dict[str, Any]) -> dict[str, Any]:
    required = {
        "primal",
        "metric",
        "affine_normals",
        "box_coordinate_normals",
        "soc_normals",
        "psd_normal",
        "halfspace_normals",
        "stationarity",
        "dual_feasibility",
        "complementarity",
        "duality_gap",
        "hashes",
    }
    missing = sorted(required - set(payload))
    if missing:
        raise RuntimeError(f"{payload.get('case_id')} missing KKT fields: {missing}")
    if payload["artifact_schema_id"] != SCHEMA_ID or payload["run_id"] != RUN2:
        raise RuntimeError("case run/schema mismatch")
    if payload["finite_vi_diagnostic"]["used_for_acceptance"]:
        raise RuntimeError("finite VI was marked as acceptance")
    if not payload["acceptance"]["kkt_is_only_acceptance"]:
        raise RuntimeError("case does not bind KKT-only acceptance")
    counters = payload["isolation_counters"]
    for key, value in counters.items():
        if key in {"segment_gold_exists", "segment_gold_used"}:
            if value is not False:
                raise RuntimeError(f"{payload['case_id']} segment flag nonzero: {key}")
        elif value != 0:
            raise RuntimeError(f"{payload['case_id']} counter nonzero: {key}={value}")
    expected_hash = case_hash(payload)
    if payload["hashes"].get("case_payload_sha256") != expected_hash:
        raise RuntimeError(f"{payload['case_id']} case payload hash mismatch")

    thresholds = cfg["semantic_thresholds"]
    primal = payload["primal"]
    gram = arr(primal["G_star"])
    objective_center = arr(primal["objective_center_G"])
    r_struct = arr(primal["r_struct_star"])
    r_center = arr(primal["objective_center_r"])
    phi = arr(primal["Phi"])
    k_consensus = arr(primal["K_consensus"])
    q = arr(payload["common_basis_operator"]["Q"])
    n = int(primal["N"])
    d = int(primal["d"])
    if gram.shape != (n, n) or objective_center.shape != (n, n):
        raise RuntimeError("G shape mismatch")
    if d != int(thresholds["rank_d"]):
        raise RuntimeError("rank d drift")
    if max_abs(gram - gram.T) > 1e-10:
        raise RuntimeError("G is not symmetric")
    if max_abs(np.diag(gram) - 1.0) > 1e-10:
        raise RuntimeError("unit diagonal failed")
    eigs = np.linalg.eigvalsh(gram)
    if float(np.min(eigs)) < float(thresholds["psd_min_eigenvalue"]):
        raise RuntimeError("PSD primal failed")
    offdiag = offdiag_values(gram)
    if np.max(np.abs(offdiag)) > 1.0 - float(thresholds["box_delta"]) + 1e-10:
        raise RuntimeError("offdiag box failed")

    recomputed_q = orth_cap(phi, primal["ids"], rank_cap=payload["common_basis_operator"]["rank_cap"])
    if recomputed_q.shape != q.shape or max_abs(recomputed_q - q) > 1e-10:
        raise RuntimeError("orth_cap replay mismatch")
    m_star = structural_moment(q, gram)
    m_target = structural_moment(q, k_consensus)
    if max_abs(m_star - arr(payload["common_basis_operator"]["M_Q_star"])) > 1e-10:
        raise RuntimeError("M_Q star replay mismatch")
    if max_abs(m_target - arr(payload["common_basis_operator"]["M_Q_target"])) > 1e-10:
        raise RuntimeError("M_Q target replay mismatch")
    if max_abs(vech(m_star - m_target) - r_struct) > 1e-10:
        raise RuntimeError("r_struct structural equality mismatch")

    s_psd = arr(payload["psd_normal"]["S_psd"])
    v_psd = arr(payload["psd_normal"]["v_psd"])
    if payload["psd_normal"]["sign_convention"] != "v_psd=-S_psd":
        raise RuntimeError("PSD sign convention mismatch")
    if max_abs(v_psd + s_psd) > 1e-10:
        raise RuntimeError("v_psd is not -S_psd")
    s_min = float(np.min(np.linalg.eigvalsh(s_psd)))
    if s_min < -float(thresholds["dual_feasibility"]):
        raise RuntimeError("PSD dual infeasible")
    psd_comp = float(np.sum(s_psd * gram))
    if abs(psd_comp) > float(thresholds["complementarity_abs"]):
        raise RuntimeError("PSD complementarity failed")

    normal_g = (
        arr(payload["affine_normals"]["normal_G"])
        + arr(payload["box_coordinate_normals"]["normal_G"])
        + arr(payload["soc_normals"]["normal_G"])
        + v_psd
        + arr(payload["halfspace_normals"]["normal_G"])
    )
    normal_r = (
        arr(payload["affine_normals"]["normal_r"])
        + arr(payload["box_coordinate_normals"]["normal_r"])
        + arr(payload["soc_normals"]["normal_r"])
        + arr(payload["halfspace_normals"]["normal_r"])
    )
    residual_g = gram - objective_center + normal_g
    residual_r = float(payload["metric"]["r_struct_weight"]) * (r_struct - r_center) + normal_r
    residual = max(max_abs(residual_g), max_abs(residual_r))
    if residual > float(thresholds["kkt_stationarity_inf"]):
        raise RuntimeError(f"stationarity failed: {residual}")
    if not payload["stationarity"]["accepted"]:
        raise RuntimeError("serialized stationarity did not accept")
    if abs(float(payload["stationarity"]["max_abs"]) - residual) > 1e-12:
        raise RuntimeError("serialized stationarity residual mismatch")

    y_factor, rank_tail = factor_from_psd_gram(gram, d=d)
    if y_factor is None or rank_tail["status"] != "PASS":
        raise RuntimeError("rank/factor gate failed")
    if payload["rank_tail"]["status"] != "PASS":
        raise RuntimeError("serialized rank gate failed")
    if int(payload["rank_tail"]["rank_eps"]) != int(rank_tail["rank_eps"]):
        raise RuntimeError("rank_eps mismatch")
    if float(payload["rank_tail"]["tail_ratio"]) > float(thresholds["rank_tail_ratio"]):
        raise RuntimeError("rank tail ratio failed")
    _, rotation, orth_resid = procrustes_align(y_factor, arr(primal["reference_Z0"]))
    if rotation.shape != tuple(payload["procrustes"]["rotation_shape"]):
        raise RuntimeError("Procrustes rotation shape mismatch")
    if orth_resid > 1e-10:
        raise RuntimeError("Procrustes orthogonality failed")

    coverage = payload["robust_interval_coverage"]["coverage"]
    if coverage < float(thresholds["robust_coverage_min_for_synthetic"]):
        raise RuntimeError("robust interval coverage failed")
    nondeg = payload["nondegeneration"]
    if payload["case_id"] == "REMOVE_NULL":
        if nondeg["status"] not in {"EXPECTED_NULL_CONTROL", "PASS"}:
            raise RuntimeError("REMOVE/null parity failed")
    elif nondeg["status"] != "PASS":
        raise RuntimeError("nondegeneration failed")
    return {
        "case_id": payload["case_id"],
        "status": "PASS",
        "stationarity_max_abs": residual,
        "psd_min_eigenvalue": float(np.min(eigs)),
        "psd_dual_min_eigenvalue": s_min,
        "psd_complementarity": psd_comp,
        "rank_eps": int(rank_tail["rank_eps"]),
        "rank_tail_ratio": float(rank_tail["tail_ratio"]),
        "q_rank": int(q.shape[1]),
        "coverage": float(coverage),
        "case_payload_sha256": payload["hashes"]["case_payload_sha256"],
    }


def mutate_payload(payload: dict[str, Any], injection_id: str) -> dict[str, Any]:
    bad = copy.deepcopy(payload)
    if injection_id == "stationarity_tamper":
        bad["psd_normal"]["v_psd"][0][0] = float(bad["psd_normal"]["v_psd"][0][0]) + 0.05
    elif injection_id == "psd_sign_flip":
        bad["psd_normal"]["v_psd"] = bad["psd_normal"]["S_psd"]
    elif injection_id == "rank_tail_corrupt":
        bad["primal"]["d"] = 2
    elif injection_id == "segment_counter_nonzero":
        bad["isolation_counters"]["segment_gold_read_count"] = 1
    elif injection_id == "finite_vi_acceptance_tamper":
        bad["finite_vi_diagnostic"]["used_for_acceptance"] = True
    elif injection_id == "payload_hash_tamper":
        bad["hashes"]["case_payload_sha256"] = "0" * 64
    else:
        raise RuntimeError(f"unknown injection {injection_id}")
    return bad


def run_injection_checks(full_payload: dict[str, Any], cfg: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for injection_id in INJECTION_IDS:
        bad = mutate_payload(full_payload, injection_id)
        try:
            verify_case_payload(bad, cfg)
        except Exception as exc:  # noqa: BLE001 - store fail-closed reason.
            rows.append(
                {
                    "injection_id": injection_id,
                    "expected": "REJECT",
                    "observed": "REJECT",
                    "fail_closed": True,
                    "reason": str(exc)[:300],
                }
            )
        else:
            rows.append(
                {
                    "injection_id": injection_id,
                    "expected": "REJECT",
                    "observed": "ACCEPT",
                    "fail_closed": False,
                    "reason": "semantic verifier accepted corrupted payload",
                }
            )
    return rows


def verify_manifest(manifest: dict[str, Any], cfg: dict[str, Any]) -> dict[str, Any]:
    if manifest["artifact_schema_id"] != SCHEMA_ID or manifest["run_id"] != RUN2:
        raise RuntimeError("manifest run/schema mismatch")
    if manifest["finite_vi_diagnostic"]["used_for_acceptance"]:
        raise RuntimeError("top-level finite VI acceptance drift")
    if manifest.get("payload_sha256") and payload_hash(manifest) != manifest["payload_sha256"]:
        raise RuntimeError("manifest payload_sha256 mismatch")
    case_results = []
    for case_id in CASE_IDS:
        case_results.append(verify_case_payload(manifest["case_payloads"][case_id], cfg))
    full = manifest["case_payloads"]["FULL"]
    injections = run_injection_checks(full, cfg)
    if not all(row["fail_closed"] for row in injections):
        raise RuntimeError("one or more injection failures did not fail closed")
    full_result = next(row for row in case_results if row["case_id"] == "FULL")
    if full_result["stationarity_max_abs"] > float(cfg["semantic_thresholds"]["kkt_stationarity_inf"]):
        raise RuntimeError("FULL KKT stationarity not accepted")
    return {
        "schema_version": "lb_scgp_global_synth_kkt_semantic_verifier_v1",
        "status": "PASS",
        "strict_semantic_verifier": True,
        "case_results": case_results,
        "injection_results": injections,
        "acceptance_rule": "serialized_H_metric_normal_cone_KKT_only",
        "finite_vi_role": "diagnostic_only_not_acceptance",
    }


def verify_run1_bindings(cfg: dict[str, Any], ledger: AccessLedger) -> dict[str, Any]:
    deps = cfg["dependencies"]
    artifact_hash = ledger.hash_file(
        deps["run1_artifact_path"], "run1_frozen_artifact_hash", "authoritative_input"
    )
    lock_hash = ledger.hash_file(
        deps["run1_publish_lock_path"], "run1_frozen_lock_hash", "authoritative_input"
    )
    assert_equal(artifact_hash, deps["run1_artifact_sha256"], "Run1 artifact hash")
    assert_equal(lock_hash, deps["run1_publish_lock_sha256"], "Run1 publish lock hash")
    run1 = read_json(deps["run1_artifact_path"])
    assert_equal(run1["run_id"], RUN1, "Run1 artifact run id")
    assert_equal(run1["terminal_state"], "FROZEN", "Run1 terminal state")
    assert_equal(run1["payload_sha256"], deps["run1_payload_sha256"], "Run1 payload hash")
    assert_equal(run1["access_ledger_sha256"], deps["run1_access_ledger_sha256"], "Run1 access ledger")
    assert_equal(run1["dirty_tree_sha256"], deps["run1_dirty_tree_sha256"], "Run1 dirty tree")
    assert_equal(run1["git_head"], deps["run1_git_head"], "Run1 git head")
    return {
        "run1_artifact_sha256": artifact_hash,
        "run1_payload_sha256": run1["payload_sha256"],
        "run1_publish_lock_sha256": lock_hash,
        "run1_access_ledger_sha256": run1["access_ledger_sha256"],
        "run1_dirty_tree_sha256": run1["dirty_tree_sha256"],
        "run1_git_head": run1["git_head"],
        "run1_terminal_state": run1["terminal_state"],
    }


def verify_machine_run(cfg: dict[str, Any], ledger: AccessLedger) -> dict[str, Any]:
    machine_hash = ledger.hash_file(
        cfg["paths"]["experiment_machine"], "machine_plan_hash", "authoritative_input"
    )
    machine = read_json(cfg["paths"]["experiment_machine"])
    run2 = next(row for row in machine["runs"] if row["run_id"] == RUN2)
    assert_equal(run2["artifact_paths"], [cfg["run"]["artifact_path"]], "machine artifact")
    assert_equal(run2["artifact_schema_ids"], [SCHEMA_ID], "machine schema")
    assert_equal(run2["slurm"], cfg["run"]["slurm"], "machine slurm")
    assert_equal(run2["dependencies"], [RUN1], "machine dependency")
    return {"machine_sha256": machine_hash, "machine_run_record": run2}


def tracker_binding(cfg: dict[str, Any], ledger: AccessLedger) -> dict[str, Any]:
    tracker_path = cfg["paths"]["experiment_tracker"]
    tracker_hash = ledger.hash_file(tracker_path, "current_tracker_hash", "authoritative_input")
    fs_path, _ = canonical_root_path(tracker_path)
    text = fs_path.read_text(encoding="utf-8")
    run1_line = next(line for line in text.splitlines() if f"| `{RUN1}` |" in line)
    run2_line = next(line for line in text.splitlines() if f"| `{RUN2}` |" in line)
    if "FROZEN" not in run1_line:
        raise RuntimeError("tracker does not bind Run1 FROZEN")
    allowed_markers = ("MUST TODO", "RUN2_IMPLEMENTED", "RUN2_SUBMITTED", "RUN2_SYNTH_KKT_GO")
    if not any(marker in run2_line for marker in allowed_markers):
        raise RuntimeError("tracker Run2 status is not an allowed takeover state")
    return {
        "tracker_sha256_at_execution": tracker_hash,
        "run1_line_sha256": sha256_obj(run1_line),
        "run2_line_sha256": sha256_obj(run2_line),
        "run2_status_line": run2_line,
    }


def build_manifest(cfg: dict[str, Any], validation: dict[str, Any], ledger: AccessLedger) -> dict[str, Any]:
    assert_equal(validation["run_id"], RUN2, "validation run id")
    assert_equal(validation["status"], "PASS", "validation status")
    run1_binding = verify_run1_bindings(cfg, ledger)
    machine = verify_machine_run(cfg, ledger)
    tracker = tracker_binding(cfg, ledger)
    schema_hash = ledger.hash_file(cfg["paths"]["schema"], "synth_schema_hash", "schema")
    config_hash = ledger.hash_file("configs/lb_scgp_global_r2/m0_synth_kkt_v1.json", "config_hash", "authoritative_input")
    old_hash, old_count = old_protected_hash_manifest()
    if old_hash != "243e89b69b169b222dd97f9df092d511f823fb26201e91fd89cb581710940462":
        raise RuntimeError("old protected LB-SCGP hash drift")
    cases = {case_id: build_case_payload(case_id, cfg) for case_id in CASE_IDS}
    semantic = {
        "schema_version": "lb_scgp_global_synth_kkt_semantic_verifier_v1",
        "status": "PENDING",
    }
    full = cases["FULL"]
    manifest: dict[str, Any] = {
        "artifact_schema_id": SCHEMA_ID,
        "schema_version": SCHEMA_ID,
        "run_id": RUN2,
        "terminal_state": "RUN2_SYNTH_KKT_GO",
        "no_m0_full_pass_claim": True,
        "rerun_policy": "failure_no_parameter_tuning_rerun",
        "slurm_policy": {
            "required": True,
            "conda_env": "HateVideo",
            "job_id": os.environ.get("SLURM_JOB_ID"),
            "cpu": cfg["run"]["slurm"]["cpu"],
            "ram_gb": cfg["run"]["slurm"]["ram_gb"],
            "gpu": cfg["run"]["slurm"]["gpu"],
            "no_time_flag": cfg["run"]["slurm"]["no_time_flag"],
        },
        "primal": full["primal"],
        "metric": full["metric"],
        "affine_normals": full["affine_normals"],
        "box_coordinate_normals": full["box_coordinate_normals"],
        "soc_normals": full["soc_normals"],
        "psd_normal": full["psd_normal"],
        "halfspace_normals": full["halfspace_normals"],
        "stationarity": full["stationarity"],
        "dual_feasibility": full["dual_feasibility"],
        "complementarity": full["complementarity"],
        "duality_gap": full["duality_gap"],
        "hashes": full["hashes"],
        "case_payloads": cases,
        "case_order": list(CASE_IDS),
        "injection_failures": [],
        "strict_semantic_verifier": semantic,
        "decision": {
            "run2_gate": "GO",
            "go_basis": "FULL serialized H-metric normal-cone/KKT verifies and all corruptions fail closed",
            "finite_vi_used_for_acceptance": False,
            "solver_trace_used_for_acceptance": False,
            "not_m0_full_pass": True,
        },
        "rank_factor_procrustes": {
            case_id: {
                "rank_tail": cases[case_id]["rank_tail"],
                "factor": cases[case_id]["factor"],
                "procrustes": cases[case_id]["procrustes"],
                "nondegeneration": cases[case_id]["nondegeneration"],
                "coverage": cases[case_id]["robust_interval_coverage"],
            }
            for case_id in CASE_IDS
        },
        "run1_frozen_binding": run1_binding,
        "machine_plan": machine,
        "tracker_binding": tracker,
        "schema_hashes": {cfg["paths"]["schema"]: schema_hash},
        "config_sha256": config_hash,
        "validator": validation,
        "old_protected_hash_comparison": {
            "current_manifest_sha256": old_hash,
            "current_path_count": old_count,
            "matched_run1_protected_snapshot": True,
        },
        "supervision_boundary": {
            "only_gold_supervision": "parent_video_binary_label",
            "segment_gold_exists": False,
            "segment_gold_used": False,
            "mllm_call_count": 0,
            "ocr_call_count": 0,
            "validation_or_test_content_opened": False,
        },
        **ledger.fields(),
        **git_dirty_hash(),
    }
    impl_hash, impl_files = implementation_hashes(cfg["implementation_files"])
    manifest["implementation_sha256"] = impl_hash
    manifest["implementation_files"] = impl_files
    semantic = verify_manifest(manifest, cfg)
    manifest["strict_semantic_verifier"] = semantic
    manifest["injection_failures"] = semantic["injection_results"]
    manifest["payload_sha256"] = payload_hash(manifest)
    verify_manifest(manifest, cfg)
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--validation-json", required=True)
    args = parser.parse_args()

    require_slurm_cpu()
    assert_equal(args.run_id, RUN2, "authorized run id")
    cfg = read_json(args.config)
    assert_equal(cfg["run"]["run_id"], RUN2, "config run id")
    assert_equal(cfg["run"]["schema_id"], SCHEMA_ID, "config schema id")
    assert_equal(cfg["authorization"]["authorized_run_ids"], [RUN2], "authorized run ids")
    if any(
        cfg["authorization"][key]
        for key in [
            "mllm_calls_allowed",
            "ocr_calls_allowed",
            "performance_evaluation_allowed",
            "query_labels_allowed",
            "query_z_allowed",
            "run3_or_later_allowed",
            "training_allowed",
            "validation_or_test_allowed",
        ]
    ):
        raise RuntimeError("Run2 authorization must remain fail-closed")
    artifact_path, _ = canonical_root_path(cfg["run"]["artifact_path"])
    if artifact_path.exists() or artifact_path.with_name(artifact_path.name + ".publish.lock").exists():
        raise FileExistsError("Run2 artifact or lock already exists")
    with open(args.validation_json, encoding="utf-8") as handle:
        validation = json.load(handle)
    ledger = AccessLedger()
    manifest = build_manifest(cfg, validation, ledger)
    exclusive_publish_json(cfg["run"]["artifact_path"], manifest)
    print(canonical_json({"run_id": RUN2, "status": manifest["terminal_state"], "payload_sha256": manifest["payload_sha256"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
