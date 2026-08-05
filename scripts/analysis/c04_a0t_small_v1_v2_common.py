#!/usr/bin/env python
"""Shared, fail-closed contract for C04-A0T-SMALL-v1.

This module contains only deterministic contract logic.  It deliberately has no
dataset-label reader, no dev/test path, no network client, no OCR path, and no
SLURM submission code.  The prospective preflight and producer import it; the
frozen V2/V3/V4 design files remain untouched.
"""
from __future__ import annotations

import base64
import hashlib
import json
import math
import os
import stat
import struct
import tempfile
import unicodedata
from pathlib import Path
from typing import Any, Iterable

ROOT = Path("/data/jehc223/RGCL")
RUN_ID = "C04-A0T-SMALL-v1"
SCHEMA_VERSION = "c04_a0t_small_v1_impl_v2"
DATASETS = ("HateMM", "MHC_zh")
PROMPT_FORMS = ("A", "B")
SLOTS = ("S", "P", "T", "H")
SELECT_TAG = "C04-A0T-SMALL-v1"
SELECT_SUFFIX = "20260729"
SELECT_N = 200
NUM_FRAMES = 8
TRANSCRIPT_CAP = 2048
TRANSCRIPT_HEAD = 1024
TRANSCRIPT_TAIL = 1024
TRANSCRIPT_SEPARATOR = "\n<TRUNCATED>\n"
MAX_NEW_TOKENS = 256
CONFIDENCE_MIN = 0
CONFIDENCE_MAX = 4
RELIABLE_CONFIDENCE_MIN = 3
PROPOSITION_COSINE_MIN = 0.80
EPS = 1e-12
ROLE_DIM = 256
TEACHER_DIM = 3584
Q_DIM = 257
LE3_INPUT_DIM = 14 * Q_DIM
ADDITIVE_INPUT_DIM = 4 * ROLE_DIM
ROLE_MAP_TAG = "C04-SPASH-ROLEMAP-v2"
LE3_TAG = "C04-LE3-DENSEJL-v3"
ADDITIVE_TAG = "C04-ADDITIVE-DENSEJL-v3"
ARTIFACT_ROOT = Path("artifacts/c04/a0t_small_v1_impl_v2")
CONFIG_RELATIVE = "configs/c04/c04_a0t_small_v1_v2.json"
TRAIN_ASR_RELATIVE = {
    "HateMM": "data/ASR/HateMM/train_asrK4_whisper-large-v3.jsonl",
    "MHC_zh": "data/ASR/MHC_zh/train_asrK4_whisper-large-v3.jsonl",
}
LEXICAL_VIDEO_ROOTS = {
    "HateMM": ROOT / "data/video/HateMM/All",
    "MHC_zh": ROOT / "data/video/MHC_zh/All",
}
PHYSICAL_TRAIN_VIDEO_ROOTS = {
    "HateMM": Path("/data/jehc223/HateMM/video"),
    "MHC_zh": Path("/data/jehc223/Multihateclip/Chinese/video"),
}
REVIEW_PIN_FIELDS = (
    "code_resource_authorization_sha256",
    "payload_review_sha256",
    "gpu_execution_authorization_sha256",
)
REVIEW_STATUS_FIELDS = (
    "code_resource_verdict",
    "payload_hash_verdict",
    "gpu_execution_verdict",
)

SOURCE_RELATIONS = (
    "current_presenter",
    "quoted_or_embedded",
    "performed_or_lyric",
    "mixed",
    "uncertain",
)
PRESENTER_STANCES = (
    "endorse_or_promote",
    "reject_or_counter",
    "report_or_describe",
    "perform_without_clear_commitment",
    "uncertain",
)
PROTECTED_TARGETS = (
    "race",
    "ethnicity",
    "religion",
    "nationality",
    "gender",
    "sexual_orientation",
    "disability",
    "other_protected",
    "no_protected_target",
    "uncertain",
)
HARM_ACTS = (
    "attack",
    "dehumanize",
    "threaten",
    "exclude",
    "harass",
    "other",
    "none",
    "uncertain",
)
RELIABILITY_STATES = ("stable", "single_valid", "conflict", "missing")

SYSTEM_PROMPT = (
    "You are a deterministic semantic-role extractor for an academic video-research "
    "pipeline. Analyze only the supplied eight sampled frames and native transcript. "
    "Describe source relation, one bounded proposition, the current presenter's stance, "
    "and a protected-target harm relation. Do not decide whether the video is hateful, "
    "do not recommend a moderation action, and do not output a rationale, timestamp, "
    "evidence span, or any key outside the requested JSON schema. Return exactly one "
    "JSON object and nothing else."
)

_SCHEMA_TEXT = (
    'Return exactly this JSON shape: {"source_relation":"current_presenter|'
    'quoted_or_embedded|performed_or_lyric|mixed|uncertain","proposition":'
    '"one neutral bounded clause","presenter_stance":"endorse_or_promote|'
    'reject_or_counter|report_or_describe|perform_without_clear_commitment|uncertain",'
    '"protected_target":"race|ethnicity|religion|nationality|gender|sexual_orientation|'
    'disability|other_protected|no_protected_target|uncertain","harm_act":"attack|'
    'dehumanize|threaten|exclude|harass|other|none|uncertain","confidence":'
    '{"S":0,"P":0,"T":0,"H":0}}. Confidence values must be integers from 0 to 4.'
)

PROMPT_A = (
    "Extract the origin relation, the single neutral proposition currently presented, "
    "the current presenter's stance toward it, and any protected-target harm relation. "
    "Describe roles, not whether the video is hateful. Use only the supplied frames and "
    "transcript. " + _SCHEMA_TEXT + "\n\nNative transcript:\n{transcript}"
)

PROMPT_B = (
    "Independently restate one bounded literal proposition from the supplied evidence, "
    "then identify who voices it, whether the present speaker supports, counters, reports, "
    "or performs it, and its protected-target plus harm act if any. Do not judge the "
    "video. " + _SCHEMA_TEXT + "\n\nNative transcript:\n{transcript}"
)

PROMPTS = {"A": PROMPT_A, "B": PROMPT_B}


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_obj(value: Any) -> str:
    return sha256_bytes(canonical_json_bytes(value))


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _forbidden_path_component(path: Path) -> str | None:
    for part in path.parts:
        lowered = part.casefold()
        if lowered in {"dev", "development", "test", "tests", "testing", "validation"}:
            return part
        if lowered.startswith(("test_", "dev_")) or lowered.endswith(("_test", "_dev")):
            return part
    return None


def root_path(relative: str | Path) -> Path:
    raw = Path(relative)
    if raw.is_absolute():
        raise RuntimeError(f"absolute project path forbidden: {relative}")
    candidate = (ROOT / raw).resolve(strict=False)
    if candidate != ROOT and ROOT not in candidate.parents:
        raise RuntimeError(f"path escapes project root: {relative}")
    forbidden = _forbidden_path_component(candidate)
    if forbidden is not None:
        raise RuntimeError(f"dev/test-like path component forbidden ({forbidden}): {relative}")
    return candidate


def _strict_regular_identity(lexical: Path, resolved: Path, label: str) -> None:
    followed = lexical.stat()
    target = resolved.stat()
    if not stat.S_ISREG(followed.st_mode) or not stat.S_ISREG(target.st_mode):
        raise RuntimeError(f"{label} is not a regular file")
    if (followed.st_dev, followed.st_ino) != (target.st_dev, target.st_ino):
        raise RuntimeError(f"{label} lexical/resolved regular-file identity mismatch")


def train_asr_path(cfg: dict[str, Any], dataset: str) -> Path:
    if dataset not in DATASETS:
        raise RuntimeError(f"unsupported dataset: {dataset}")
    expected_relative = TRAIN_ASR_RELATIVE[dataset]
    if cfg["datasets"][dataset]["train_asr"] != expected_relative:
        raise RuntimeError(f"HALT_PATH_POLICY: {dataset} train ASR path is not hard-bound")
    lexical = ROOT / expected_relative
    resolved = lexical.resolve(strict=True)
    expected_root = (ROOT / "data/ASR" / dataset).resolve(strict=True)
    if resolved.parent != expected_root:
        raise RuntimeError(f"HALT_PATH_POLICY: {dataset} ASR escapes hard-bound train root")
    if lexical.is_symlink():
        raise RuntimeError(f"HALT_PATH_POLICY: {dataset} ASR symlink forbidden")
    _strict_regular_identity(lexical, resolved, f"{dataset} train ASR")
    forbidden = _forbidden_path_component(resolved)
    if forbidden is not None:
        raise RuntimeError(f"HALT_PATH_POLICY: forbidden ASR component {forbidden}")
    return resolved


def video_path(cfg: dict[str, Any], dataset: str, video_id: str) -> Path:
    if (
        dataset not in DATASETS
        or not video_id
        or "/" in video_id
        or "\\" in video_id
        or video_id in {".", ".."}
        or "\x00" in video_id
    ):
        raise RuntimeError("invalid video locator")
    lexical_root = LEXICAL_VIDEO_ROOTS[dataset]
    configured_lexical = cfg["datasets"][dataset]["video_root"]
    expected_lexical = lexical_root.relative_to(ROOT).as_posix()
    if configured_lexical != expected_lexical:
        raise RuntimeError(f"HALT_PATH_POLICY: {dataset} lexical video root is not hard-bound")
    physical_root = PHYSICAL_TRAIN_VIDEO_ROOTS[dataset].resolve(strict=True)
    if cfg["datasets"][dataset]["physical_train_video_root"] != physical_root.as_posix():
        raise RuntimeError(f"HALT_PATH_POLICY: {dataset} physical train root is not hard-bound")
    lexical = lexical_root / f"{video_id}.mp4"
    if lexical.parent != lexical_root:
        raise RuntimeError("video locator escapes lexical dataset root")
    if not lexical.is_symlink() or not lexical.exists():
        raise RuntimeError(f"selected video missing: {lexical}")
    resolved = lexical.resolve(strict=True)
    if resolved.parent != physical_root:
        raise RuntimeError(
            f"HALT_PATH_POLICY: {dataset} video escapes hard-bound physical train root"
        )
    _strict_regular_identity(lexical, resolved, f"{dataset}/{video_id} video")
    for candidate in (lexical, resolved):
        forbidden = _forbidden_path_component(candidate)
        if forbidden is not None:
            raise RuntimeError(f"HALT_PATH_POLICY: forbidden video component {forbidden}")
    return resolved


def load_json(relative: str | Path) -> dict[str, Any]:
    path = root_path(relative)
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise RuntimeError(f"expected JSON object: {relative}")
    return value


def require_exact_keys(value: Any, expected: set[str], label: str) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != expected:
        actual = sorted(value) if isinstance(value, dict) else type(value).__name__
        raise RuntimeError(f"{label} exact-key failure: {actual!r}")
    return value


def config_contract_sha256(cfg: dict[str, Any]) -> str:
    """Hash the complete static config while normalizing staged review authority.

    Review pins and stage authorization are separately bound inside each strict
    authorization manifest.  Normalizing only those fields avoids an impossible
    config<->manifest hash cycle without leaving either field unaudited.
    """
    view = json.loads(json.dumps(cfg, ensure_ascii=False))
    view["authorization"] = "<BOUND_BY_STRICT_STAGE_AUTHORIZATION_MANIFEST>"
    review = view["review"]
    for key in REVIEW_PIN_FIELDS:
        review[key] = "<BOUND_BY_EXACT_FILE_SHA256>"
    for key in REVIEW_STATUS_FIELDS:
        review[key] = "<BOUND_BY_STRICT_STAGE_AUTHORIZATION_MANIFEST>"
    return sha256_obj(view)


def verify_bound_file_map(mapping: dict[str, str], label: str) -> None:
    if not isinstance(mapping, dict) or not mapping:
        raise RuntimeError(f"{label} must be a nonempty hash mapping")
    for relative, expected in sorted(mapping.items()):
        if not isinstance(expected, str) or len(expected) != 64:
            raise RuntimeError(f"{label} invalid SHA-256 for {relative}")
        actual = sha256_file(root_path(relative))
        if actual != expected:
            raise RuntimeError(f"{label} hash mismatch for {relative}: {actual} != {expected}")


def source_hash_closure(cfg: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {
        dataset: {
            "train_asr": cfg["datasets"][dataset]["train_asr"],
            "train_asr_sha256": cfg["datasets"][dataset]["train_asr_sha256"],
            "train_asr_size": cfg["datasets"][dataset]["train_asr_size"],
            "lexical_video_root": cfg["datasets"][dataset]["video_root"],
            "physical_train_video_root": cfg["datasets"][dataset][
                "physical_train_video_root"
            ],
        }
        for dataset in DATASETS
    }


def model_hash_closure(cfg: dict[str, Any]) -> dict[str, Any]:
    return {
        "model_id": cfg["model"]["model_id"],
        "snapshot_revision": cfg["model"]["snapshot_revision"],
        "snapshot_path": cfg["model"]["snapshot_path"],
        "model_tree_sha256": cfg["model"]["model_tree_sha256"],
        "processor_tree_sha256": cfg["model"]["processor_tree_sha256"],
        "files": cfg["model"]["files"],
    }


def verify_closure_hash(value: dict[str, Any], label: str) -> dict[str, Any]:
    body = dict(value)
    claimed = body.pop("closure_sha256", None)
    if not isinstance(claimed, str) or sha256_obj(body) != claimed:
        raise RuntimeError(f"{label} closure SHA-256 mismatch")
    return body


def _verified_review_file(
    cfg: dict[str, Any],
    path_field: str,
    pin_field: str,
    schema_field: str,
    label: str,
) -> tuple[dict[str, Any], str]:
    relative = cfg["review"][path_field]
    pin = cfg["review"][pin_field]
    if (
        not isinstance(pin, str)
        or len(pin) != 64
        or any(char not in "0123456789abcdef" for char in pin)
    ):
        raise RuntimeError(f"HALT_REVIEW_LINEAGE: {label} SHA-256 is unpinned")
    path = root_path(relative)
    if sha256_file(path) != pin:
        raise RuntimeError(f"HALT_REVIEW_LINEAGE: {label} exact file SHA-256 mismatch")
    value = load_json(relative)
    validate_schema(value, cfg["schemas"][schema_field], label)
    return value, pin


def verify_preflight_manifest(
    cfg: dict[str, Any],
    allow_claimed_gpu_ledger: bool = False,
) -> tuple[dict[str, Any], str]:
    relative = cfg["paths"]["preflight_manifest"]
    path = root_path(relative)
    manifest = load_json(relative)
    body = dict(manifest)
    claimed = body.pop("payload_sha256", None)
    if not isinstance(claimed, str) or sha256_obj(body) != claimed:
        raise RuntimeError("HALT_REVIEW_LINEAGE: preflight manifest payload mismatch")
    if manifest["run_id"] != RUN_ID or manifest["implementation_version"] != "v2_prospective":
        raise RuntimeError("HALT_REVIEW_LINEAGE: foreign preflight manifest")
    for staged_relative, expected in manifest["staged_output_hashes"].items():
        if (
            allow_claimed_gpu_ledger
            and staged_relative == cfg["paths"]["gpu_ledger"]
        ):
            continue
        if sha256_file(root_path(staged_relative)) != expected:
            raise RuntimeError(
                f"HALT_REVIEW_LINEAGE: staged preflight hash mismatch {staged_relative}"
            )
    return manifest, sha256_file(path)


def verify_historical_code_resource_authorization(
    cfg: dict[str, Any],
    preflight: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], str]:
    manifest, pin = _verified_review_file(
        cfg,
        "code_resource_authorization_manifest",
        "code_resource_authorization_sha256",
        "stage_authorization",
        "code/resource authorization",
    )
    body = verify_closure_hash(manifest, "code/resource authorization")
    expected_static = {
        "run_id": RUN_ID,
        "implementation_version": "v2_prospective",
        "stage": "CPU_PREFLIGHT",
        "verdict": "GO",
        "config_contract_sha256": config_contract_sha256(cfg),
        "implementation_hashes": cfg["implementation_hashes"],
        "frozen_design_hashes": cfg["frozen_design_hashes"],
        "design_go_review_sha256": cfg["frozen_design_hashes"][
            "refine-logs/C04_V4_DESIGN_REVIEW.md"
        ],
        "source_hash_closure": source_hash_closure(cfg),
        "model_hash_closure": model_hash_closure(cfg),
        "payload_binding": "NO_PREFLIGHT_PAYLOAD_YET",
    }
    for key, value in expected_static.items():
        if body[key] != value:
            raise RuntimeError(f"HALT_REVIEW_LINEAGE: code authorization mismatch {key}")
    auth = body["authorization_snapshot"]
    required_true = ("implementation_authorized", "preflight_materialization_authorized")
    required_false = tuple(key for key in auth if key not in required_true)
    if any(auth[key] is not True for key in required_true) or any(
        auth[key] is not False for key in required_false
    ):
        raise RuntimeError("HALT_REVIEW_LINEAGE: invalid historical preflight authorization")
    if preflight is not None and preflight["code_resource_authorization_sha256"] != pin:
        raise RuntimeError("HALT_REVIEW_LINEAGE: preflight/code authorization pin mismatch")
    verify_bound_file_map(cfg["implementation_hashes"], "code-authorized implementation")
    verify_bound_file_map(cfg["frozen_design_hashes"], "code-authorized frozen design")
    return manifest, pin


def verify_payload_review(
    cfg: dict[str, Any],
    preflight: dict[str, Any],
    preflight_sha256: str,
) -> tuple[dict[str, Any], str]:
    if cfg["review"]["payload_hash_verdict"] != "GO":
        raise RuntimeError("HALT_REVIEW_LINEAGE: payload hash verdict is not GO")
    review, pin = _verified_review_file(
        cfg,
        "payload_review",
        "payload_review_sha256",
        "payload_review",
        "payload review",
    )
    body = verify_closure_hash(review, "payload review")
    attestation_core = dict(body)
    attested = attestation_core.pop("attested_closure_sha256", None)
    reviewed_payload_sha256 = attestation_core.pop("reviewed_payload_sha256", None)
    if sha256_obj(attestation_core) != reviewed_payload_sha256:
        raise RuntimeError("HALT_REVIEW_LINEAGE: reviewed payload SHA-256 mismatch")
    expected_attestation = sha256_bytes(
        ("C04-PAYLOAD-REVIEW-GO-v2\n" + reviewed_payload_sha256).encode("utf-8")
    )
    if attested != expected_attestation:
        raise RuntimeError("HALT_REVIEW_LINEAGE: payload review attestation mismatch")
    if body["run_id"] != RUN_ID or body["implementation_version"] != "v2_prospective":
        raise RuntimeError("HALT_REVIEW_LINEAGE: foreign payload review")
    if body["verdict"] != "GO":
        raise RuntimeError("HALT_REVIEW_LINEAGE: payload review is not GO")
    if body["config_contract_sha256"] != config_contract_sha256(cfg):
        raise RuntimeError("HALT_REVIEW_LINEAGE: payload config contract mismatch")
    if body["preflight_manifest_sha256"] != preflight_sha256:
        raise RuntimeError("HALT_REVIEW_LINEAGE: reviewed preflight hash mismatch")
    if body["code_resource_authorization_sha256"] != preflight[
        "code_resource_authorization_sha256"
    ]:
        raise RuntimeError("HALT_REVIEW_LINEAGE: payload/code authorization mismatch")
    if body["prompt_hashes"] != preflight["prompt_hashes"]:
        raise RuntimeError("HALT_REVIEW_LINEAGE: reviewed prompt hashes mismatch")
    if body["map_hashes"] != preflight["map_hashes"]:
        raise RuntimeError("HALT_REVIEW_LINEAGE: reviewed map hashes mismatch")
    if body["staged_output_hashes"] != preflight["staged_output_hashes"]:
        raise RuntimeError("HALT_REVIEW_LINEAGE: reviewed staged output hashes mismatch")
    return review, pin


def verify_gpu_execution_authorization(
    cfg: dict[str, Any],
    preflight: dict[str, Any],
    preflight_sha256: str,
    payload_review_sha256: str,
) -> tuple[dict[str, Any], str]:
    if cfg["review"]["gpu_execution_verdict"] != "GO":
        raise RuntimeError("HALT_REVIEW_LINEAGE: GPU execution verdict is not GO")
    manifest, pin = _verified_review_file(
        cfg,
        "gpu_execution_authorization_manifest",
        "gpu_execution_authorization_sha256",
        "stage_authorization",
        "GPU execution authorization",
    )
    body = verify_closure_hash(manifest, "GPU execution authorization")
    expected = {
        "run_id": RUN_ID,
        "implementation_version": "v2_prospective",
        "stage": "GPU_TEACHER_PRELABEL_SEAL",
        "verdict": "GO",
        "authorization_snapshot": cfg["authorization"],
        "config_contract_sha256": config_contract_sha256(cfg),
        "implementation_hashes": cfg["implementation_hashes"],
        "frozen_design_hashes": cfg["frozen_design_hashes"],
        "design_go_review_sha256": cfg["frozen_design_hashes"][
            "refine-logs/C04_V4_DESIGN_REVIEW.md"
        ],
        "source_hash_closure": source_hash_closure(cfg),
        "model_hash_closure": model_hash_closure(cfg),
        "payload_binding": {
            "payload_review_sha256": payload_review_sha256,
            "preflight_manifest_sha256": preflight_sha256,
            "prompt_hashes": preflight["prompt_hashes"],
            "map_hashes": preflight["map_hashes"],
        },
    }
    for key, value in expected.items():
        if body[key] != value:
            raise RuntimeError(f"HALT_REVIEW_LINEAGE: GPU authorization mismatch {key}")
    verify_bound_file_map(cfg["implementation_hashes"], "GPU implementation")
    verify_bound_file_map(cfg["frozen_design_hashes"], "GPU frozen design")
    return manifest, pin


def exclusive_publish_bytes(relative: str | Path, payload: bytes) -> None:
    path = root_path(relative)
    path.parent.mkdir(parents=True, exist_ok=True)
    lock = path.with_name(path.name + ".publish.lock")
    if path.exists() or lock.exists():
        raise FileExistsError(f"no-clobber refusal: {relative}")
    fd = os.open(str(lock), os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o640)
    try:
        os.write(fd, sha256_bytes(payload).encode("ascii") + b"\n")
        os.fsync(fd)
    finally:
        os.close(fd)
    temp_name = ""
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb", dir=str(path.parent), prefix=path.name + ".tmp.",
            delete=False,
        ) as handle:
            temp_name = handle.name
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_name, path)
    except Exception:
        if temp_name:
            try:
                Path(temp_name).unlink()
            except FileNotFoundError:
                pass
        try:
            lock.unlink()
        except FileNotFoundError:
            pass
        raise


def exclusive_publish_json(relative: str | Path, value: Any) -> None:
    exclusive_publish_bytes(relative, canonical_json_bytes(value) + b"\n")


def exclusive_publish_jsonl(relative: str | Path, rows: Iterable[Any]) -> None:
    payload = b"".join(canonical_json_bytes(row) + b"\n" for row in rows)
    exclusive_publish_bytes(relative, payload)


def validate_schema(value: Any, schema_relative: str, label: str) -> None:
    from jsonschema import Draft7Validator

    schema = load_json(schema_relative)
    errors = sorted(
        Draft7Validator(schema).iter_errors(value),
        key=lambda error: list(error.absolute_path),
    )
    if errors:
        detail = "; ".join(
            f"{list(error.absolute_path)}: {error.message}" for error in errors[:5]
        )
        raise RuntimeError(f"{label} schema failure: {detail}")


def normalize_transcript(value: str) -> str:
    text = unicodedata.normalize("NFKC", value or "")
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    scalars = list(text)
    if len(scalars) <= TRANSCRIPT_CAP:
        return text
    return (
        "".join(scalars[:TRANSCRIPT_HEAD])
        + TRANSCRIPT_SEPARATOR
        + "".join(scalars[-TRANSCRIPT_TAIL:])
    )


def normalize_proposition(value: str) -> str:
    text = unicodedata.normalize("NFKC", value or "")
    text = text.casefold()
    text = " ".join(text.split())
    while text and text[-1] in ".!?。！？；;，,":
        text = text[:-1].rstrip()
    return text


def bounded_proposition(value: Any, dataset: str) -> bool:
    if not isinstance(value, str) or not value.strip():
        return False
    if dataset == "HateMM":
        return len(value.split()) <= 32
    if dataset == "MHC_zh":
        return len(list(value)) <= 64
    return False


def selection_digest(dataset: str, video_id: str) -> str:
    if dataset not in DATASETS:
        raise RuntimeError(f"unsupported dataset: {dataset}")
    payload = (SELECT_TAG + dataset + video_id + SELECT_SUFFIX).encode("utf-8")
    return sha256_bytes(payload)


def _skip_json_value(text: str, start: int) -> int:
    """Skip one JSON value without materializing it.

    This is used for the label-bearing train-ASR container: only allowlisted
    evidence fields are decoded.  The `label` token is syntactically skipped and
    never converted to a Python value.
    """
    n = len(text)
    i = start
    while i < n and text[i].isspace():
        i += 1
    if i >= n:
        raise ValueError("missing JSON value")
    if text[i] == '"':
        i += 1
        escaped = False
        while i < n:
            char = text[i]
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                return i + 1
            i += 1
        raise ValueError("unterminated JSON string")
    if text[i] in "[{":
        stack = ["]" if text[i] == "[" else "}"]
        i += 1
        in_string = False
        escaped = False
        while i < n:
            char = text[i]
            if in_string:
                if escaped:
                    escaped = False
                elif char == "\\":
                    escaped = True
                elif char == '"':
                    in_string = False
            elif char == '"':
                in_string = True
            elif char == "[":
                stack.append("]")
            elif char == "{":
                stack.append("}")
            elif char in "]}":
                if not stack or char != stack[-1]:
                    raise ValueError("mismatched JSON composite")
                stack.pop()
                if not stack:
                    return i + 1
            i += 1
        raise ValueError("unterminated JSON composite")
    i += 1
    while i < n and text[i] not in ",}":
        i += 1
    return i


def project_train_asr_line(line: str) -> tuple[dict[str, Any], dict[str, int]]:
    """Decode only id/window_text/language from a top-level JSON object."""
    decoder = json.JSONDecoder()
    text = line.strip()
    if not text.startswith("{"):
        raise ValueError("ASR row is not an object")
    i = 1
    projected: dict[str, Any] = {}
    seen: set[str] = set()
    skipped_label = 0
    while True:
        while i < len(text) and text[i].isspace():
            i += 1
        if i < len(text) and text[i] == "}":
            i += 1
            break
        key, end = decoder.raw_decode(text, i)
        if not isinstance(key, str) or key in seen:
            raise ValueError("non-string or duplicate JSON key")
        seen.add(key)
        i = end
        while i < len(text) and text[i].isspace():
            i += 1
        if i >= len(text) or text[i] != ":":
            raise ValueError("missing JSON colon")
        i += 1
        while i < len(text) and text[i].isspace():
            i += 1
        if key in {"id", "window_text", "language"}:
            value, i = decoder.raw_decode(text, i)
            projected[key] = value
        else:
            if key == "label":
                skipped_label += 1
            i = _skip_json_value(text, i)
        while i < len(text) and text[i].isspace():
            i += 1
        if i < len(text) and text[i] == ",":
            i += 1
            continue
        if i < len(text) and text[i] == "}":
            i += 1
            break
        raise ValueError("bad JSON object delimiter")
    if text[i:].strip():
        raise ValueError("extra text after ASR object")
    if set(projected) != {"id", "window_text", "language"}:
        raise ValueError(f"required projected fields missing: {sorted(projected)}")
    if not isinstance(projected["id"], str) or not projected["id"]:
        raise ValueError("invalid id")
    windows = projected["window_text"]
    if not isinstance(windows, list) or any(not isinstance(x, str) for x in windows):
        raise ValueError("window_text must be list[str]")
    if not isinstance(projected["language"], str):
        raise ValueError("language must be string")
    transcript = "\n".join(x for x in windows if x)
    return {
        "id": projected["id"],
        "language": projected["language"],
        "transcript": normalize_transcript(transcript),
    }, {
        "label_field_syntactically_skipped": skipped_label,
        "label_value_materialized": 0,
    }


def parse_teacher_response(raw: str, dataset: str) -> dict[str, Any]:
    result: dict[str, Any] = {
        "form_valid": False,
        "object_recoverable": False,
        "parse_error": "",
        "slots": {
            slot: {
                "valid": False,
                "content": "",
                "confidence": None,
                "error": "unparsed",
            }
            for slot in SLOTS
        },
    }
    try:
        value = json.loads((raw or "").strip())
    except Exception:
        result["parse_error"] = "json_decode_error"
        for slot in SLOTS:
            result["slots"][slot]["error"] = "global_json_decode_error"
        return result
    expected = {
        "source_relation", "proposition", "presenter_stance",
        "protected_target", "harm_act", "confidence",
    }
    if not isinstance(value, dict):
        result["parse_error"] = "top_level_not_object"
        for slot in SLOTS:
            result["slots"][slot]["error"] = "global_top_level_not_object"
        return result
    result["object_recoverable"] = True
    form_errors: list[str] = []
    if set(value) != expected:
        form_errors.append("top_level_schema_error")
    confidence = value.get("confidence")
    if not isinstance(confidence, dict):
        form_errors.append("confidence_not_object")
        confidence = {}
    elif set(confidence) != set(SLOTS):
        form_errors.append("confidence_schema_error")

    slot_specs: dict[str, tuple[bool, str, str]] = {
        "S": (
            value.get("source_relation") in SOURCE_RELATIONS,
            str(value.get("source_relation", "")),
            "source_relation_enum_error",
        ),
        "P": (
            bounded_proposition(value.get("proposition"), dataset),
            value.get("proposition", "").strip()
            if isinstance(value.get("proposition"), str)
            else "",
            "proposition_bounds_error",
        ),
        "T": (
            value.get("presenter_stance") in PRESENTER_STANCES,
            str(value.get("presenter_stance", "")),
            "presenter_stance_enum_error",
        ),
        "H": (
            value.get("protected_target") in PROTECTED_TARGETS
            and value.get("harm_act") in HARM_ACTS,
            (
                f"protected_target={value.get('protected_target', '')};"
                f"harm_act={value.get('harm_act', '')}"
            ),
            "protected_target_or_harm_act_enum_error",
        ),
    }
    for slot in SLOTS:
        content_valid, content, content_error = slot_specs[slot]
        confidence_value = confidence.get(slot)
        confidence_valid = (
            isinstance(confidence_value, int)
            and not isinstance(confidence_value, bool)
            and CONFIDENCE_MIN <= confidence_value <= CONFIDENCE_MAX
        )
        slot_errors = []
        if not content_valid:
            slot_errors.append(content_error)
        if not confidence_valid:
            slot_errors.append("confidence_value_error")
        result["slots"][slot] = {
            "valid": not slot_errors,
            "content": content if content_valid else "",
            "confidence": confidence_value if confidence_valid else None,
            "error": ";".join(slot_errors),
        }
        form_errors.extend(f"{slot}:{error}" for error in slot_errors)
    result["form_valid"] = not form_errors
    result["parse_error"] = ";".join(form_errors)
    return result


def enum_agreement(slot: str, a: str, b: str) -> bool:
    if slot == "P":
        raise RuntimeError("P agreement requires frozen embedding cosine")
    return a == b


def build_slot_reliability(
    slot: str,
    parsed_a: dict[str, Any],
    parsed_b: dict[str, Any],
    proposition_cosine: float | None,
) -> dict[str, Any]:
    a = parsed_a["slots"][slot]
    b = parsed_b["slots"][slot]
    valid_a = bool(a["valid"])
    valid_b = bool(b["valid"])
    high_a = valid_a and int(a["confidence"]) >= RELIABLE_CONFIDENCE_MIN
    high_b = valid_b and int(b["confidence"]) >= RELIABLE_CONFIDENCE_MIN
    if valid_a and valid_b:
        agrees = (
            proposition_cosine is not None and proposition_cosine >= PROPOSITION_COSINE_MIN
            if slot == "P"
            else enum_agreement(slot, str(a["content"]), str(b["content"]))
        )
    else:
        agrees = False
    if valid_a and valid_b and agrees and high_a and high_b:
        state = "stable"
        content = str(a["content"])
        selected_form = "A"
    elif high_a and not valid_b:
        state = "single_valid"
        content = str(a["content"])
        selected_form = "A"
    elif high_b and not valid_a:
        state = "single_valid"
        content = str(b["content"])
        selected_form = "B"
    elif not valid_a and not valid_b:
        state = "missing"
        content = f"NO_CONTENT_{slot}"
        selected_form = "none"
    else:
        state = "conflict"
        content = f"NO_CONTENT_{slot}"
        selected_form = "none"
    return {
        "slot": slot,
        "state": state,
        "content": content,
        "selected_form": selected_form,
        "a_valid": valid_a,
        "b_valid": valid_b,
        "a_confidence": a["confidence"],
        "b_confidence": b["confidence"],
        "agreement": agrees,
        "proposition_cosine": proposition_cosine if slot == "P" else None,
    }


def render_slot(
    content: str,
    state: str,
    control: str = "FULL",
    slot: str | None = None,
) -> str:
    if control == "FULL":
        return f"{content}<fallback={state}>"
    if control == "STATE_ONLY":
        if slot not in SLOTS:
            raise RuntimeError("STATE_ONLY render requires an explicit slot")
        return f"NO_CONTENT_{slot}<fallback={state}>"
    if control == "STATE_BLIND":
        return content
    if control == "FALLBACK_COLLAPSE":
        return f"{content}<fallback=unavailable>"
    raise RuntimeError(f"unsupported render control: {control}")


class HashStream:
    def __init__(self, prefix: bytes) -> None:
        self.prefix = prefix
        self.counter = 0
        self.buffer = b""

    def _fill(self) -> None:
        block = hashlib.sha256(self.prefix + struct.pack(">Q", self.counter)).digest()
        self.counter += 1
        self.buffer += block

    def take(self, n: int) -> bytes:
        while len(self.buffer) < n:
            self._fill()
        value, self.buffer = self.buffer[:n], self.buffer[n:]
        return value

    def u64(self) -> int:
        return struct.unpack(">Q", self.take(8))[0]

    def uniform(self, upper_exclusive: int) -> int:
        if upper_exclusive <= 0:
            raise ValueError("upper_exclusive must be positive")
        limit = (1 << 64) - ((1 << 64) % upper_exclusive)
        while True:
            value = self.u64()
            if value < limit:
                return value % upper_exclusive


def materialize_role_map(role: str) -> dict[str, Any]:
    if role not in SLOTS:
        raise RuntimeError(f"bad role: {role}")
    stream = HashStream((ROLE_MAP_TAG + role).encode("utf-8"))
    indices = list(range(TEACHER_DIM))
    for i in range(TEACHER_DIM - 1, 0, -1):
        j = stream.uniform(i + 1)
        indices[i], indices[j] = indices[j], indices[i]
    selected = indices[:ROLE_DIM]
    sign_bytes = stream.take(math.ceil(ROLE_DIM / 8))
    signs = [
        1 if ((sign_bytes[i // 8] >> (7 - (i % 8))) & 1) == 0 else -1
        for i in range(ROLE_DIM)
    ]
    payload = {
        "schema_version": "c04_role_map_v1",
        "tag": ROLE_MAP_TAG,
        "role": role,
        "teacher_dim": TEACHER_DIM,
        "output_dim": ROLE_DIM,
        "indices": selected,
        "signs": signs,
    }
    payload["payload_sha256"] = sha256_obj(payload)
    return payload


def dense_rademacher_payload(tag: str, rows: int, cols: int, scale: float) -> bytes:
    payload = bytearray(rows * cols * 4)
    offset = 0
    prefix = tag.encode("utf-8")
    for row in range(rows):
        for col in range(cols):
            digest = hashlib.sha256(
                prefix + struct.pack(">H", row) + struct.pack(">H", col)
            ).digest()
            value = scale if (digest[-1] & 1) == 0 else -scale
            struct.pack_into("<f", payload, offset, value)
            offset += 4
    return bytes(payload)


def f32le_b64(values: Iterable[float]) -> dict[str, Any]:
    data = b"".join(struct.pack("<f", float(value)) for value in values)
    return {
        "encoding": "float32_le_base64",
        "count": len(data) // 4,
        "sha256": sha256_bytes(data),
        "data": base64.b64encode(data).decode("ascii"),
    }


def safe_vector(values: list[float]) -> list[float]:
    norm = math.sqrt(sum(value * value for value in values))
    if norm <= EPS:
        return [0.0 for _ in values]
    return [value / max(norm, EPS) for value in values]


def q_product(vectors: list[list[float]]) -> list[float]:
    if not vectors:
        raise ValueError("at least one vector required")
    product = [1.0] * len(vectors[0])
    for vector in vectors:
        if len(vector) != len(product):
            raise ValueError("vector dimension mismatch")
        product = [left * right for left, right in zip(product, vector)]
    norm = math.sqrt(sum(value * value for value in product))
    normalized = safe_vector(product)
    return normalized + [1.0 if norm <= EPS else 0.0]


def merkle_root(rows: Iterable[Any]) -> str:
    leaves = [bytes.fromhex(sha256_obj(row)) for row in rows]
    if not leaves:
        return sha256_bytes(b"")
    level = leaves
    while len(level) > 1:
        if len(level) % 2:
            level.append(level[-1])
        level = [
            hashlib.sha256(level[i] + level[i + 1]).digest()
            for i in range(0, len(level), 2)
        ]
    return level[0].hex()


def prompt_hashes() -> dict[str, str]:
    return {
        "system": sha256_bytes(SYSTEM_PROMPT.encode("utf-8")),
        "A": sha256_bytes(PROMPT_A.encode("utf-8")),
        "B": sha256_bytes(PROMPT_B.encode("utf-8")),
        "combined": sha256_obj({
            "system": SYSTEM_PROMPT,
            "A": PROMPT_A,
            "B": PROMPT_B,
            "enum_order": {
                "source_relation": SOURCE_RELATIONS,
                "presenter_stance": PRESENTER_STANCES,
                "protected_target": PROTECTED_TARGETS,
                "harm_act": HARM_ACTS,
            },
        }),
    }


def self_test_fixtures() -> list[tuple[str, bool]]:
    valid = {
        "source_relation": "current_presenter",
        "proposition": "A speaker presents a bounded proposition",
        "presenter_stance": "report_or_describe",
        "protected_target": "no_protected_target",
        "harm_act": "none",
        "confidence": {"S": 4, "P": 4, "T": 4, "H": 4},
    }
    valid_raw = json.dumps(valid, ensure_ascii=False)
    parsed = parse_teacher_response(valid_raw, "HateMM")
    malformed = parse_teacher_response("{", "HateMM")
    partial = dict(valid)
    partial["presenter_stance"] = "not_an_enum"
    partial_parsed = parse_teacher_response(
        json.dumps(partial, ensure_ascii=False), "HateMM"
    )
    return [
        ("valid_form", parsed["form_valid"] is True),
        ("malformed_form", malformed["form_valid"] is False),
        (
            "slot_local_invalidity",
            partial_parsed["object_recoverable"] is True
            and partial_parsed["slots"]["T"]["valid"] is False
            and all(
                partial_parsed["slots"][slot]["valid"] is True
                for slot in ("S", "P", "H")
            ),
        ),
        (
            "transcript_cap",
            len(normalize_transcript("x" * 3000))
            == TRANSCRIPT_HEAD + len(TRANSCRIPT_SEPARATOR) + TRANSCRIPT_TAIL,
        ),
        (
            "selection_deterministic",
            selection_digest("HateMM", "x") == selection_digest("HateMM", "x"),
        ),
        (
            "missing_state",
            build_slot_reliability("S", malformed, malformed, None)["state"] == "missing",
        ),
        (
            "single_valid_state",
            build_slot_reliability("S", parsed, malformed, None)["state"] == "single_valid",
        ),
    ]
