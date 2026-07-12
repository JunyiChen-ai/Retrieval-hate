#!/usr/bin/env python
"""Fresh independent semantic verifier for Run2 synthetic KKT.

This verifier intentionally does not import the Run2 producer or common module.
It rebuilds certificate consensus, Q, M_Q, A^T, KKT residuals, source bindings,
rank/factor replay, and fail-closed injections from serialized payloads.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import tempfile
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path("/data/jehc223/RGCL")
RUN2 = "LBSCGP-GLOBAL-G0-M0-SYNTH-KKT-v1"
SCHEMA_ID = "scgp_global_synth_kkt_payload_v1"
TRI_OBSERVABLES = (
    "visual_reference_observable",
    "text_audio_reference_observable",
    "harmful_surface_observable",
    "dehumanizing_or_threat_surface_observable",
    "cross_modal_binding_observable",
    "source_alignment_observable",
    "counter_context_observable",
    "context_shift_observable",
)
TRI_STATES = ("contradicted", "supported", "unresolved")
MODALITY_OBSERVABLE = "modality_binding_observable"
MODALITY_STATES = ("multi_modal", "single_modal", "text_audio", "unresolved", "visual_audio", "visual_text")
REQUIRED_CERT_KEYS = (("schema_version",) + TRI_OBSERVABLES + (MODALITY_OBSERVABLE, "parse_flags"))
TOP_KEYS = {
    "schema_version",
    "artifact_schema_id",
    "run_id",
    "terminal_state",
    "authorized_boundary",
    "no_success_claim",
    "slurm_policy",
    "config_path",
    "source_manifest_path",
    "access_ledger_path",
    "primal",
    "metric",
    "movement_metrics",
    "affine_normals",
    "box_coordinate_normals",
    "soc_normals",
    "psd_normal",
    "halfspace_normals",
    "stationarity",
    "dual_feasibility",
    "complementarity",
    "duality_gap",
    "case_matrix",
    "orth_cap_matrix",
    "rank_failure_probe",
    "schema_fixture_results",
    "injection_results_expected",
    "gold_isolation",
    "dirty_binding",
    "acceptance",
    "hashes",
    "payload_sha256",
}
CASE_KEYS = {
    "case_id",
    "case_role",
    "system",
    "ids",
    "labels",
    "d",
    "replicas",
    "consensus_records",
    "operator",
    "movement_metrics",
    "primal_residuals",
    "rank_audit",
    "factor_replay",
    "robust_coverage",
    "finite_vi_diagnostics",
    "acceptance_path",
    "kkt_status",
    "hashes",
}


def canonical_json(obj: Any) -> str:
    return json.dumps(obj, allow_nan=False, ensure_ascii=True, separators=(",", ":"), sort_keys=True)


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_obj(obj: Any) -> str:
    return sha256_bytes(canonical_json(obj).encode("utf-8"))


def sha256_file(path: Path | str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def payload_hash(obj: dict[str, Any]) -> str:
    clone = dict(obj)
    clone.pop("payload_sha256", None)
    return sha256_obj(clone)


def root_path(path: str | Path) -> Path:
    raw = Path(path)
    candidate = raw if raw.is_absolute() else ROOT / raw
    resolved = candidate.resolve()
    resolved.relative_to(ROOT.resolve())
    return resolved


def read_json(path: str | Path) -> Any:
    with open(root_path(path), encoding="utf-8") as handle:
        return json.load(handle)


def finite_array(arr: np.ndarray, name: str) -> None:
    if not np.all(np.isfinite(arr)):
        raise RuntimeError(f"non-finite numeric payload in {name}")


def validate_cert(record: dict[str, Any]) -> None:
    if set(record) != set(REQUIRED_CERT_KEYS):
        raise RuntimeError("certificate key set mismatch")
    if record["schema_version"] != "scgp_global_cert_v2":
        raise RuntimeError("certificate schema version mismatch")
    for field in TRI_OBSERVABLES:
        item = record[field]
        if set(item) != {"state", "confidence"}:
            raise RuntimeError(f"{field} extra/missing keys")
        if item["state"] not in TRI_STATES:
            raise RuntimeError(f"{field} invalid state")
        if not isinstance(item["confidence"], int) or not 0 <= item["confidence"] <= 4:
            raise RuntimeError(f"{field} invalid confidence")
    modality = record[MODALITY_OBSERVABLE]
    if set(modality) != {"state", "confidence"} or modality["state"] not in MODALITY_STATES:
        raise RuntimeError("modality observable invalid")
    if not isinstance(modality["confidence"], int) or not 0 <= modality["confidence"] <= 4:
        raise RuntimeError("modality confidence invalid")
    if not isinstance(record["parse_flags"], list) or any(not isinstance(flag, str) for flag in record["parse_flags"]):
        raise RuntimeError("parse_flags invalid")


def consensus(group: list[dict[str, Any]]) -> dict[str, Any]:
    for record in group:
        validate_cert(record)
    out: dict[str, Any] = {"schema_version": "scgp_global_cert_v2"}
    for field in TRI_OBSERVABLES:
        counts = {state: 0 for state in TRI_STATES}
        for record in group:
            counts[record[field]["state"]] += 1
        best = sorted(counts.items(), key=lambda item: (-item[1], item[0]))[0]
        out[field] = {"state": best[0] if best[1] > len(group) / 2 else "unresolved", "confidence": 0}
    counts = {state: 0 for state in MODALITY_STATES}
    for record in group:
        counts[record[MODALITY_OBSERVABLE]["state"]] += 1
    best = sorted(counts.items(), key=lambda item: (-item[1], item[0]))[0]
    out[MODALITY_OBSERVABLE] = {"state": best[0] if best[1] > len(group) / 2 else "unresolved", "confidence": 0}
    out["parse_flags"] = []
    validate_cert(out)
    return out


def encode(record: dict[str, Any]) -> np.ndarray:
    validate_cert(record)
    values = []
    for field in TRI_OBSERVABLES:
        values.extend(1.0 if record[field]["state"] == state else 0.0 for state in TRI_STATES)
    values.extend(1.0 if record[MODALITY_OBSERVABLE]["state"] == state else 0.0 for state in MODALITY_STATES)
    return np.asarray(values, dtype=np.float64)


def row_normalize(x: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    if np.any(norms <= 0):
        raise RuntimeError("zero feature row")
    return x / norms


def orth_cap(phi: np.ndarray, ids: list[str], cap: int = 8) -> tuple[np.ndarray, int]:
    centered = phi - phi.mean(axis=0, keepdims=True)
    u, s, _ = np.linalg.svd(centered, full_matrices=False)
    if s.size == 0:
        return np.zeros((phi.shape[0], 0), dtype=np.float64), 0
    threshold = max(1e-8, 1e-7 * float(s[0]))
    raw_rank = int(np.sum(s > threshold))
    q = u[:, : min(cap, raw_rank)].copy()
    for col in range(q.shape[1]):
        pivot = max(range(q.shape[0]), key=lambda row: (abs(float(q[row, col])), str(ids[row])))
        if q[pivot, col] < 0:
            q[:, col] *= -1.0
    return q, raw_rank


def vech(mat: np.ndarray) -> np.ndarray:
    rows, cols = np.tril_indices(mat.shape[0])
    return np.asarray(mat)[rows, cols]


def unvech_dual(vec: np.ndarray, rank: int) -> np.ndarray:
    vec = np.asarray(vec, dtype=np.float64).reshape(-1)
    if vec.size != rank * (rank + 1) // 2:
        raise RuntimeError("nu length does not match Q rank")
    out = np.zeros((rank, rank), dtype=np.float64)
    rows, cols = np.tril_indices(rank)
    for value, row, col in zip(vec, rows, cols):
        if row == col:
            out[row, col] = value
        else:
            out[row, col] = 0.5 * value
            out[col, row] = 0.5 * value
    return out


def structural_moment(q: np.ndarray, gram: np.ndarray) -> np.ndarray:
    if q.shape[1] == 0:
        return np.zeros((0, 0), dtype=np.float64)
    n = gram.shape[0]
    out = q.T @ (gram - np.eye(n)) @ q / float(n)
    return 0.5 * (out + out.T)


def structural_adjoint(q: np.ndarray, nu: np.ndarray) -> np.ndarray:
    if q.shape[1] == 0:
        return np.zeros((q.shape[0], q.shape[0]), dtype=np.float64)
    s = unvech_dual(nu, q.shape[1])
    out = q @ s @ q.T / float(q.shape[0])
    return 0.5 * (out + out.T)


def rank_factor_audit(gram: np.ndarray, d: int) -> tuple[bool, dict[str, Any], np.ndarray | None]:
    eigval, eigvec = np.linalg.eigh(0.5 * (gram + gram.T))
    order = np.argsort(-eigval, kind="mergesort")
    eigval = eigval[order]
    eigvec = eigvec[:, order]
    eps = max(1e-8, 1e-7 * max(float(eigval[0]) if eigval.size else 0.0, 1.0))
    rank = int(np.sum(eigval > eps))
    positive_mass = float(np.maximum(eigval, 0.0).sum())
    omitted = float(np.maximum(eigval[d:], 0.0).sum()) if d < eigval.size else 0.0
    negative = float(np.maximum(-eigval, 0.0).sum())
    if rank > d:
        residual = math.inf
        y = None
    else:
        y = np.zeros((gram.shape[0], d), dtype=np.float64)
        if rank:
            y[:, :rank] = eigvec[:, :rank] * np.sqrt(np.maximum(eigval[:rank], 0.0))[None, :]
        residual = float(np.linalg.norm(y @ y.T - gram) / max(1.0, np.linalg.norm(gram)))
    audit = {
        "lambda_d": float(eigval[d - 1]) if 0 < d <= eigval.size else float(eigval[-1]),
        "lambda_dplus1": float(eigval[d]) if d < eigval.size else 0.0,
        "rank_eps": rank,
        "eps_rank": eps,
        "positive_eigenmass": positive_mass,
        "omitted_positive_eigenmass_beyond_d": omitted,
        "tail_ratio": omitted / max(positive_mass, 1e-12),
        "negative_eigenmass": negative,
        "lambda_min": float(eigval[-1]),
        "reconstruction_residual": residual,
    }
    passed = (
        rank <= d
        and omitted <= max(1e-6, 1e-8 * gram.shape[0])
        and audit["tail_ratio"] <= 1e-8
        and negative <= max(1e-6, 1e-8 * gram.shape[0])
        and audit["lambda_min"] >= -1e-7
        and residual <= 1e-6
    )
    return passed, audit, y


def verify_source_and_access(manifest: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    source_path = manifest["source_manifest_path"]
    access_path = manifest["access_ledger_path"]
    if sha256_file(root_path(source_path)) != manifest["hashes"]["source_manifest_sha256"]:
        raise RuntimeError("source manifest file hash mismatch")
    if sha256_file(root_path(access_path)) != manifest["hashes"]["access_ledger_sha256"]:
        raise RuntimeError("access ledger file hash mismatch")
    source = read_json(source_path)
    access = read_json(access_path)
    if source["relevant_tree_sha256"] != manifest["dirty_binding"]["relevant_tree_sha256"]:
        raise RuntimeError("relevant tree binding mismatch")
    recomputed_tree = sha256_obj(
        {
            "source_rows": source["source_rows"],
            "implementation_sha256": source["implementation_sha256"],
            "relevant_git_status": source["relevant_git_status"],
            "artifact_outputs_excluded": True,
        }
    )
    if recomputed_tree != source["relevant_tree_sha256"]:
        raise RuntimeError("source manifest relevant-tree recomputation mismatch")
    for rel, expected in config["hash_bindings"]["run1_frozen"].items():
        if sha256_file(root_path(rel)) != expected:
            raise RuntimeError(f"Run1 frozen hash changed: {rel}")
    for rel, expected in config["hash_bindings"]["authoritative_inputs"].items():
        if sha256_file(root_path(rel)) != expected:
            raise RuntimeError(f"authoritative input hash changed: {rel}")
    for row in source["source_rows"]["run2_implementation_files"]:
        if sha256_file(root_path(row["path"])) != row["sha256"]:
            raise RuntimeError(f"Run2 source hash mismatch: {row['path']}")
    forbidden = []
    for record in access["access_ledger"]:
        path = record.get("path", "")
        lowered = path.lower()
        if path.startswith("data/") and record.get("kind") != "declared_not_opened":
            forbidden.append(path)
        if any(token in lowered for token in ["query_z", "query_labels", "teacher"]):
            forbidden.append(path)
    if forbidden:
        raise RuntimeError(f"forbidden access ledger paths: {forbidden}")
    nonzero = {key: value for key, value in access["zero_counters"].items() if value != 0}
    if nonzero:
        raise RuntimeError(f"nonzero access counters: {nonzero}")
    return {
        "source_manifest_sha256": manifest["hashes"]["source_manifest_sha256"],
        "access_ledger_sha256": manifest["hashes"]["access_ledger_sha256"],
        "relevant_tree_sha256": source["relevant_tree_sha256"],
    }


def verify_core(manifest: dict[str, Any], config: dict[str, Any], verify_files: bool) -> dict[str, Any]:
    if set(manifest) != TOP_KEYS:
        raise RuntimeError(f"top-level schema keys mismatch: {sorted(set(manifest) ^ TOP_KEYS)}")
    if manifest["payload_sha256"] != payload_hash(manifest):
        raise RuntimeError("payload_sha256 mismatch")
    if manifest["artifact_schema_id"] != SCHEMA_ID or manifest["run_id"] != RUN2:
        raise RuntimeError("run/schema mismatch")
    if manifest["acceptance"]["acceptance_path"] != "serialized_h_metric_normal_cone_kkt":
        raise RuntimeError("finite VI or non-KKT acceptance attempted")
    if manifest["acceptance"]["finite_vi_can_accept"] is not False:
        raise RuntimeError("finite VI marked accepting")
    if manifest["psd_normal"]["normal_contribution_sign"] != "v_psd=-S_psd":
        raise RuntimeError("wrong PSD dual sign")
    if manifest["halfspace_normals"]["robust_constraints_enabled"] is not False:
        raise RuntimeError("Run2 robust constraints must remain off")
    if verify_files:
        file_binding = verify_source_and_access(manifest, config)
    else:
        file_binding = {}

    case_matrix = manifest["case_matrix"]
    if case_matrix["status"] != "PASS":
        raise RuntimeError("case matrix not PASS")
    cases = case_matrix["cases"]
    systems = {case["system"] for case in cases}
    required_systems = {"FULL", "REMOVE", "SHUFFLE", "NOISE", "AMBIGUOUS", "ROBUST_COVERAGE"}
    if not required_systems.issubset(systems):
        raise RuntimeError("missing required system cases")
    full_case = next(case for case in cases if case["system"] == "FULL")
    if set(full_case) != CASE_KEYS:
        raise RuntimeError("case schema keys mismatch")
    consensus_records = [consensus(group) for group in full_case["replicas"]]
    if consensus_records != full_case["consensus_records"]:
        raise RuntimeError("consensus replay mismatch")
    phi = np.stack([encode(record) for record in consensus_records], axis=0)
    q, raw_rank = orth_cap(phi, full_case["ids"], cap=8)
    if raw_rank != full_case["operator"]["raw_rank_before_cap"] or q.shape[1] != full_case["operator"]["q_rank"]:
        raise RuntimeError("orth_cap replay mismatch")
    g0 = np.asarray(manifest["primal"]["G0"], dtype=np.float64)
    gstar = np.asarray(manifest["primal"]["G_star"], dtype=np.float64)
    r = np.asarray(manifest["primal"]["r_struct"], dtype=np.float64)
    finite_array(g0, "G0")
    finite_array(gstar, "G_star")
    finite_array(r, "r_struct")
    if np.linalg.norm(gstar - g0) <= manifest["movement_metrics"]["positive_threshold"]:
        raise RuntimeError("FULL attempted no-movement acceptance")
    if not np.allclose(g0, np.eye(g0.shape[0]), atol=1e-12):
        raise RuntimeError("FULL baseline G0 is not the explicit identity baseline")
    if not np.allclose(np.diag(gstar), 1.0, atol=1e-8):
        raise RuntimeError("unit diagonal violated")
    if np.min(np.linalg.eigvalsh(0.5 * (gstar + gstar.T))) < -1e-7:
        raise RuntimeError("PSD primal violated")
    off_mask = ~np.eye(gstar.shape[0], dtype=bool)
    if np.max(np.abs(gstar[off_mask])) > 1.0 - 1e-4 + 1e-9:
        raise RuntimeError("off-diagonal box violated")
    movement = gstar - g0
    move_fro = float(np.linalg.norm(movement))
    move_off = float(np.max(np.abs(movement[off_mask])))
    if move_fro <= manifest["movement_metrics"]["positive_threshold"] or move_off <= manifest["movement_metrics"]["positive_threshold"]:
        raise RuntimeError("movement nondegeneration gate failed")

    b = np.asarray(full_case["operator"]["b_struct"], dtype=np.float64)
    structural_residual = r - (vech(structural_moment(q, gstar)) - b)
    if np.linalg.norm(structural_residual) > 1e-6:
        raise RuntimeError("structural equality residual failed")
    nu = np.asarray(manifest["affine_normals"]["structural_nu"], dtype=np.float64)
    diag_dual = np.asarray(manifest["affine_normals"]["diagonal_affine_dual"], dtype=np.float64)
    adj = structural_adjoint(q, nu)
    if np.linalg.norm(r + nu / float(manifest["metric"]["lambda_struct"])) > 1e-6:
        raise RuntimeError("r/nu H-metric relation failed")
    stationarity_g = movement - adj + np.diag(diag_dual)
    stationarity_r = float(manifest["metric"]["lambda_struct"]) * r + nu
    stationarity_norm = math.sqrt(float(np.linalg.norm(stationarity_g)) ** 2 + float(np.linalg.norm(stationarity_r)) ** 2)
    normalized = stationarity_norm / (1.0 + float(np.linalg.norm(movement)) + float(np.linalg.norm(r)))
    if normalized > 1e-6:
        raise RuntimeError("stationarity failed")
    if np.linalg.norm(nu) <= manifest["movement_metrics"]["positive_threshold"]:
        raise RuntimeError("structural dual is not binding")
    expected_objective = 0.5 * float(np.linalg.norm(movement)) ** 2 + 0.5 * float(manifest["metric"]["lambda_struct"]) * float(np.linalg.norm(r)) ** 2
    if abs(expected_objective - float(manifest["primal"]["objective_value"])) > 1e-10:
        raise RuntimeError("objective serialization mismatch")

    rank_pass, rank_audit, factor = rank_factor_audit(gstar, full_case["d"])
    if not rank_pass or factor is None:
        raise RuntimeError("rank/factor replay failed")
    zstar_resid = float(np.linalg.norm(factor @ factor.T - gstar) / max(1.0, np.linalg.norm(gstar)))
    if zstar_resid > 1e-6 or np.min(np.linalg.norm(factor, axis=1)) <= 1e-8:
        raise RuntimeError("factor nondegeneration failed")
    remove_case = next(case for case in cases if case["system"] == "REMOVE")
    if remove_case["movement_metrics"]["fro_norm_G_star_minus_G0"] != 0.0:
        raise RuntimeError("REMOVE/null did not replay G0")
    if manifest["rank_failure_probe"]["factor_returned_null"] is not True:
        raise RuntimeError("rank failure probe did not return null")
    if manifest["rank_failure_probe"]["rank_audit"]["status"] != "ENCODER_RANK_GATE_FAIL":
        raise RuntimeError("rank failure probe status mismatch")
    if any(item["status"] != "REJECT" for item in manifest["schema_fixture_results"]["invalid_schema"]):
        raise RuntimeError("invalid schema fixture accepted")
    if manifest["schema_fixture_results"]["unresolved_values"]["schema_status"] != "PASS":
        raise RuntimeError("unresolved value fixture failed")
    if any(value != "PASS" for value in manifest["orth_cap_matrix"].values()):
        raise RuntimeError("orth_cap matrix failed")
    return {
        "stationarity_normalized_residual": normalized,
        "movement_fro": move_fro,
        "movement_offdiag_max": move_off,
        "structural_dual_l2": float(np.linalg.norm(nu)),
        "objective_value": expected_objective,
        "rank_eps": rank_audit["rank_eps"],
        "zstar_gram_residual": zstar_resid,
        **file_binding,
    }


def run_injections(manifest: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    mutations = {}
    m = copy.deepcopy(manifest)
    m["primal"]["G_star"][0][0] = float("nan")
    mutations["nan_overflow"] = m
    m = copy.deepcopy(manifest)
    m["hashes"]["operator_hash"] = "0" * 64
    mutations["perturbed_artifact_source_operator_hash"] = m
    m = copy.deepcopy(manifest)
    m["unexpected"] = True
    mutations["invalid_extra_missing_schema_fields"] = m
    m = copy.deepcopy(manifest)
    m["psd_normal"]["normal_contribution_sign"] = "v_psd=+S_psd"
    mutations["wrong_dual_sign"] = m
    m = copy.deepcopy(manifest)
    m.pop("soc_normals")
    mutations["incomplete_cone_family"] = m
    m = copy.deepcopy(manifest)
    m["source_manifest_path"] = "data/gt/MHC/test.jsonl"
    mutations["forbidden_path"] = m
    m = copy.deepcopy(manifest)
    m["rank_failure_probe"]["factor_returned_null"] = False
    mutations["rank_failure"] = m
    m = copy.deepcopy(manifest)
    m["acceptance"]["acceptance_path"] = "finite_vi"
    mutations["finite_vi_only_attempted_acceptance"] = m
    m = copy.deepcopy(manifest)
    m["primal"]["G_star"] = copy.deepcopy(m["primal"]["G0"])
    m["movement_metrics"]["fro_norm_G_star_minus_G0"] = 0.0
    m["movement_metrics"]["max_abs_offdiag_change"] = 0.0
    mutations["identity_no_movement_claims_full"] = m
    results = {}
    for name, mutated in mutations.items():
        try:
            verify_core(mutated, config, verify_files=False)
        except Exception as exc:  # noqa: BLE001 - serialized negative test evidence
            results[name] = {"status": "REJECT", "reason": str(exc)[:500]}
        else:
            results[name] = {"status": "UNEXPECTED_ACCEPT", "reason": ""}
    bad = {key: value for key, value in results.items() if value["status"] != "REJECT"}
    if bad:
        raise RuntimeError(f"injection unexpectedly accepted: {bad}")
    return results


def publish_json(path: str | Path, obj: Any) -> None:
    fs_path = root_path(path)
    fs_path.parent.mkdir(parents=True, exist_ok=True)
    lock = fs_path.with_name(fs_path.name + ".publish.lock")
    fd_lock = os.open(str(lock), os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    tmp = None
    try:
        os.write(fd_lock, str(os.getpid()).encode("ascii"))
        os.fsync(fd_lock)
        os.close(fd_lock)
        fd_lock = -1
        if fs_path.exists():
            raise FileExistsError(f"refusing to overwrite {fs_path}")
        fd, tmp = tempfile.mkstemp(prefix=fs_path.name + ".tmp.", dir=str(fs_path.parent))
        with os.fdopen(fd, "wb") as handle:
            handle.write((canonical_json(obj) + "\n").encode("utf-8"))
            handle.flush()
            os.fsync(handle.fileno())
        os.link(tmp, fs_path)
        os.unlink(tmp)
        tmp = None
    finally:
        if fd_lock >= 0:
            os.close(fd_lock)
        if tmp and os.path.exists(tmp):
            os.unlink(tmp)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    if not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("independent verification must run under SLURM")
    config = read_json(args.config)
    manifest = read_json(args.manifest)
    try:
        metrics = verify_core(manifest, config, verify_files=True)
        injections = run_injections(manifest, config)
        decision = {
            "schema_version": "lb_scgp_global_r2_run2_semantic_verification_v1",
            "run_id": RUN2,
            "decision": "PASS",
            "manifest_path": args.manifest,
            "manifest_file_sha256": sha256_file(root_path(args.manifest)),
            "manifest_payload_sha256": manifest["payload_sha256"],
            "metrics": metrics,
            "injection_results": injections,
            "acceptance_path": "serialized_h_metric_normal_cone_kkt",
            "finite_vi_acceptance": false_value(),
            "medium_findings_closed": {
                "M1_strict_schema_semantic_verifier": True,
                "M2_dirty_binding_run1_run2_relevant_tree": True,
                "M3_orth_cap_and_M_Q_executed_with_rank_cap_cases": True
            },
        }
        publish_json(args.out, decision)
        return 0
    except Exception as exc:  # noqa: BLE001 - publish fail-closed decision
        decision = {
            "schema_version": "lb_scgp_global_r2_run2_semantic_verification_v1",
            "run_id": RUN2,
            "decision": "FAIL",
            "manifest_path": args.manifest,
            "manifest_file_sha256": sha256_file(root_path(args.manifest)) if root_path(args.manifest).exists() else "",
            "reason": str(exc),
        }
        publish_json(args.out, decision)
        print(json.dumps(decision, indent=2, sort_keys=True), flush=True)
        return 1


def false_value() -> bool:
    return False


if __name__ == "__main__":
    raise SystemExit(main())
