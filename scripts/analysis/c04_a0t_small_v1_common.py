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
import struct
import tempfile
import unicodedata
from pathlib import Path
from typing import Any, Iterable

ROOT = Path("/data/jehc223/RGCL")
RUN_ID = "C04-A0T-SMALL-v1"
SCHEMA_VERSION = "c04_a0t_small_v1"
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
ARTIFACT_ROOT = Path("artifacts/c04/a0t_small_v1")

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


def root_path(relative: str | Path) -> Path:
    raw = Path(relative)
    if raw.is_absolute():
        raise RuntimeError(f"absolute project path forbidden: {relative}")
    candidate = (ROOT / raw).resolve(strict=False)
    if candidate != ROOT and ROOT not in candidate.parents:
        raise RuntimeError(f"path escapes project root: {relative}")
    lowered = candidate.as_posix().lower()
    if "/test" in lowered or "test_seen" in lowered:
        raise RuntimeError(f"test-like path forbidden: {relative}")
    return candidate


def video_path(dataset: str, video_id: str) -> Path:
    if dataset not in DATASETS or not video_id or "/" in video_id or "\x00" in video_id:
        raise RuntimeError("invalid video locator")
    lexical_root = ROOT / "data" / "video" / dataset / "All"
    lexical = lexical_root / f"{video_id}.mp4"
    if lexical.parent != lexical_root:
        raise RuntimeError("video locator escapes lexical dataset root")
    if not lexical.exists():
        raise RuntimeError(f"selected video missing: {lexical}")
    resolved = lexical.resolve(strict=True)
    if not resolved.is_file():
        raise RuntimeError(f"selected video target is not a file: {lexical}")
    if "test" in lexical.as_posix().lower():
        raise RuntimeError("test-like video path forbidden")
    return lexical


def load_json(relative: str | Path) -> dict[str, Any]:
    path = root_path(relative)
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise RuntimeError(f"expected JSON object: {relative}")
    return value


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
        "parse_error": "",
        "slots": {slot: {"valid": False, "content": "", "confidence": None} for slot in SLOTS},
    }
    try:
        value = json.loads((raw or "").strip())
    except Exception:
        result["parse_error"] = "json_decode_error"
        return result
    expected = {
        "source_relation", "proposition", "presenter_stance",
        "protected_target", "harm_act", "confidence",
    }
    if not isinstance(value, dict) or set(value) != expected:
        result["parse_error"] = "top_level_schema_error"
        return result
    confidence = value.get("confidence")
    if not isinstance(confidence, dict) or set(confidence) != set(SLOTS):
        result["parse_error"] = "confidence_schema_error"
        return result
    if any(
        not isinstance(confidence[slot], int)
        or isinstance(confidence[slot], bool)
        or not CONFIDENCE_MIN <= confidence[slot] <= CONFIDENCE_MAX
        for slot in SLOTS
    ):
        result["parse_error"] = "confidence_value_error"
        return result
    if value["source_relation"] not in SOURCE_RELATIONS:
        result["parse_error"] = "source_relation_enum_error"
        return result
    if value["presenter_stance"] not in PRESENTER_STANCES:
        result["parse_error"] = "presenter_stance_enum_error"
        return result
    if value["protected_target"] not in PROTECTED_TARGETS:
        result["parse_error"] = "protected_target_enum_error"
        return result
    if value["harm_act"] not in HARM_ACTS:
        result["parse_error"] = "harm_act_enum_error"
        return result
    if not bounded_proposition(value["proposition"], dataset):
        result["parse_error"] = "proposition_bounds_error"
        return result
    result["form_valid"] = True
    result["slots"] = {
        "S": {
            "valid": True,
            "content": value["source_relation"],
            "confidence": confidence["S"],
        },
        "P": {
            "valid": True,
            "content": value["proposition"].strip(),
            "confidence": confidence["P"],
        },
        "T": {
            "valid": True,
            "content": value["presenter_stance"],
            "confidence": confidence["T"],
        },
        "H": {
            "valid": True,
            "content": (
                f"protected_target={value['protected_target']};"
                f"harm_act={value['harm_act']}"
            ),
            "confidence": confidence["H"],
        },
    }
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
    return [
        ("valid_form", parsed["form_valid"] is True),
        ("malformed_form", malformed["form_valid"] is False),
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
