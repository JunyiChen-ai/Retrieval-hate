#!/usr/bin/env python
"""M1 train-only label-blind MLLM cache utilities for LB-SCGP Global-R2.

Runs LBSCGP-GLOBAL-M1-CACHE-{MHC,MHC_zh}-v1 (GPU producers) and
LBSCGP-GLOBAL-M1-CACHE-SEAL-v1 (CPU seal).  This module is deliberately
self-contained: the pure serialization / hashing / path-guard / schema helpers
are byte-faithful copies of the frozen-and-ACCEPTED realbank (Run2-v4-derived)
code, so the M1 pipeline reuses exactly the verified plumbing without a
cross-lineage import.  The M1-specific orchestration (SLURM guards, runs[4..6]
machine verifier, train-evidence access ledger, the restricted scgp_global_cert_v2
prompt/schema, R=4 replica consensus, Merkle root, seal decision) is new.

Discipline: the cache producer reads only train-video evidence — 16 uniform
frames, the video title (gt 'text' field), and the Whisper ASR transcript — and
NO label/split/seed/neighbor/prediction/margin/correctness signal.  Train labels
are NOT opened before seal.  It runs a LOCAL Qwen2.5-VL-7B under HF_HUB_OFFLINE=1
(no external/network/provider API; ocr_calls=0, no live OCR).  It never opens
validation/test/held content or labels, caches (other than its own output),
teacher artifacts, query_z, or query_labels.  It makes no accuracy/macro-F1 claim
and does no training or kNN.  The only project gold is parent_video_binary_label
and it is not read here.
"""
from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any

ROOT = Path("/data/jehc223/RGCL")

RUN_MHC = "LBSCGP-GLOBAL-M1-CACHE-MHC-v1"
RUN_MHC_ZH = "LBSCGP-GLOBAL-M1-CACHE-MHC_zh-v1"
RUN_SEAL = "LBSCGP-GLOBAL-M1-CACHE-SEAL-v1"
REALBANK_DEP = "LBSCGP-GLOBAL-G0-M0-REALBANK-RESOURCE-v2"

# runs[] indices in the authoritative machine plan (v3 index-drift lesson: pin
# both the index literal and the run_id, and let the machine verifier assert both).
RUN_INDEX = {RUN_MHC: 4, RUN_MHC_ZH: 5, RUN_SEAL: 6}
DATASET_RUN = {"MHC": RUN_MHC, "MHC_zh": RUN_MHC_ZH}
RUN_DATASET = {RUN_MHC: "MHC", RUN_MHC_ZH: "MHC_zh"}

DATASETS = ("MHC", "MHC_zh")
EXPECTED_TRAIN_N = {"MHC": 549, "MHC_zh": 579}

MODEL_ID = "Qwen/Qwen2.5-VL-7B-Instruct"
NUM_FRAMES = 16          # project M=16 uniform full-video discipline
FRAME_RULE = "uniform_full_video_M16"
REPLICAS = 4             # R=4 deterministic calls per unique evidence pack
MAX_TITLE_CHARS = 2000
MAX_ASR_CHARS = 4000
MAX_NEW_TOKENS = 320     # generous cap for the strict-JSON certificate
CONF_MIN, CONF_MAX = 0, 4

REPLICA_SCHEMA_ID = "scgp_global_cache_replica_v2"
SEAL_SCHEMA_ID = "scgp_global_cache_seal_v1"
CERT_SCHEMA_VERSION = "scgp_global_cert_v2"
CACHE_MANIFEST_SCHEMA_VERSION = "lb_scgp_global_r2_m1_cache_manifest_v1"
SEAL_MANIFEST_SCHEMA_VERSION = "lb_scgp_global_r2_m1_cache_seal_manifest_v1"

# The eight restricted structural observables (FINAL_PROPOSAL.md scgp_global_cert_v2).
OBSERVABLE_KEYS = (
    "visual_reference_observable",
    "text_audio_reference_observable",
    "harmful_surface_observable",
    "dehumanizing_or_threat_surface_observable",
    "cross_modal_binding_observable",
    "source_alignment_observable",
    "counter_context_observable",
    "context_shift_observable",
)
MODALITY_KEY = "modality_binding_observable"
STATE_VALUES = ("supported", "contradicted", "unresolved")
MODALITY_STATES = ("visual_text", "visual_audio", "text_audio", "multi_modal", "single_modal", "unresolved")
# consensus map for the eight ternary observables: supported=+1, contradicted=-1, unresolved=0
CONSENSUS_MAP = {"supported": 1, "contradicted": -1, "unresolved": 0}

# Forbidden-read counters that must all remain exactly 0 (three-way aligned with the
# strict seal schema's zero_counters required set and the access ledger).
ZERO_COUNTER_KEYS = (
    "cache_outer_held_content_read_count",
    "cache_outer_held_label_read_count",
    "cache_test_content_read_count",
    "cache_test_label_read_count",
    "cache_validation_content_read_count",
    "cache_validation_label_read_count",
    "certificate_read_count",
    "compiler_target_read_count",
    "forbidden_path_read_count",
    "held_content_read_count",
    "held_label_read_count",
    "margin_read_count",
    "mllm_calls_outside_train_cache",
    "neighbor_read_count",
    "network_model_api_call_count",
    "non_allowlisted_train_content_read_count",
    "ocr_call_count",
    "prediction_read_count",
    "query_labels_read_count",
    "query_z_read_count",
    "seed_read_count",
    "segment_gold_read_count",
    "split_statistic_read_count",
    "teacher_artifact_read_count",
    "teacher_cache_read_count",
    "test_content_read_count",
    "test_label_read_count",
    "train_label_read_count",
    "validation_content_read_count",
    "validation_label_read_count",
)


# --------------------------------------------------------------------------- #
# Serialization / hashing helpers (byte-faithful to the frozen realbank module) #
# --------------------------------------------------------------------------- #
def canonical_json(obj: Any) -> str:
    return json.dumps(obj, allow_nan=False, ensure_ascii=True, separators=(",", ":"), sort_keys=True)


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_obj(obj: Any) -> str:
    return sha256_bytes(canonical_json(obj).encode("utf-8"))


def sha256_text(text: str) -> str:
    return sha256_bytes(text.encode("utf-8"))


def sha256_file(path: Path | str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def canonical_root_path(path: Path | str) -> tuple[Path, Path]:
    root = ROOT.resolve()
    raw = Path(path)
    candidate = raw if raw.is_absolute() else root / raw
    resolved = candidate.resolve()
    try:
        rel = resolved.relative_to(root)
    except ValueError as exc:
        raise RuntimeError(f"path escapes repository root: {path}") from exc
    return resolved, rel


def read_json(path: Path | str) -> Any:
    fs_path, _ = canonical_root_path(path)
    with open(fs_path, encoding="utf-8") as handle:
        return json.load(handle)


def payload_hash(obj: dict[str, Any], field: str = "payload_sha256") -> str:
    clone = dict(obj)
    clone.pop(field, None)
    return sha256_obj(clone)


def assert_equal(actual: Any, expected: Any, label_text: str) -> None:
    if actual != expected:
        raise RuntimeError(f"{label_text} drift: expected {expected!r}, got {actual!r}")


def _fsync_dir(path: Path) -> None:
    fd = os.open(str(path), os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def exclusive_publish_json(path: Path | str, obj: Any) -> None:
    fs_path, _ = canonical_root_path(path)
    fs_path.parent.mkdir(parents=True, exist_ok=True)
    lock = fs_path.with_name(fs_path.name + ".publish.lock")
    lock_fd = os.open(str(lock), os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    tmp = None
    try:
        os.write(lock_fd, str(os.getpid()).encode("ascii"))
        os.fsync(lock_fd)
        os.close(lock_fd)
        lock_fd = -1
        if fs_path.exists():
            raise FileExistsError(f"refusing to overwrite {fs_path}")
        # explicit in-repo dir=; the tempfile default (gettempdir -> $TMPDIR) is never
        # consulted (realbank-v1 $TMPDIR landmine lesson).
        fd, tmp = tempfile.mkstemp(prefix=fs_path.name + ".tmp.", dir=str(fs_path.parent))
        with os.fdopen(fd, "wb") as handle:
            handle.write((canonical_json(obj) + "\n").encode("utf-8"))
            handle.flush()
            os.fsync(handle.fileno())
        os.link(tmp, fs_path)
        os.unlink(tmp)
        tmp = None
        _fsync_dir(fs_path.parent)
    finally:
        if lock_fd >= 0:
            os.close(lock_fd)
        if tmp and os.path.exists(tmp):
            os.unlink(tmp)


def exclusive_publish_jsonl(path: Path | str, records: list[dict[str, Any]]) -> None:
    """Atomic publish of a JSONL bank (one canonical record per line)."""
    fs_path, _ = canonical_root_path(path)
    fs_path.parent.mkdir(parents=True, exist_ok=True)
    lock = fs_path.with_name(fs_path.name + ".publish.lock")
    lock_fd = os.open(str(lock), os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    tmp = None
    try:
        os.write(lock_fd, str(os.getpid()).encode("ascii"))
        os.fsync(lock_fd)
        os.close(lock_fd)
        lock_fd = -1
        if fs_path.exists():
            raise FileExistsError(f"refusing to overwrite {fs_path}")
        fd, tmp = tempfile.mkstemp(prefix=fs_path.name + ".tmp.", dir=str(fs_path.parent))
        with os.fdopen(fd, "wb") as handle:
            for record in records:
                handle.write((canonical_json(record) + "\n").encode("utf-8"))
            handle.flush()
            os.fsync(handle.fileno())
        os.link(tmp, fs_path)
        os.unlink(tmp)
        tmp = None
        _fsync_dir(fs_path.parent)
    finally:
        if lock_fd >= 0:
            os.close(lock_fd)
        if tmp and os.path.exists(tmp):
            os.unlink(tmp)


# --------------------------------------------------------------------------- #
# Strict JSON-Schema helpers (byte-faithful to the frozen realbank module)     #
# --------------------------------------------------------------------------- #
def schema_requires_no_additional_properties(schema: Any, path: str = "$") -> list[str]:
    errors: list[str] = []
    if isinstance(schema, dict):
        if (schema.get("type") == "object" or "properties" in schema) and schema.get("additionalProperties") is not False:
            errors.append(path)
        for key, value in schema.items():
            if key in {"properties", "definitions"} and isinstance(value, dict):
                for name, child in value.items():
                    errors.extend(schema_requires_no_additional_properties(child, f"{path}.{key}.{name}"))
            elif key in {"items"}:
                errors.extend(schema_requires_no_additional_properties(value, f"{path}.{key}"))
            elif isinstance(value, dict) and "$ref" not in value:
                errors.extend(schema_requires_no_additional_properties(value, f"{path}.{key}"))
            elif isinstance(value, list):
                for idx, item in enumerate(value):
                    errors.extend(schema_requires_no_additional_properties(item, f"{path}.{key}[{idx}]"))
    return errors


def validate_against_schema(instance: Any, schema_path: Path | str, label_text: str) -> None:
    try:
        from jsonschema import Draft7Validator
        from jsonschema.exceptions import SchemaError
    except Exception as exc:  # noqa: BLE001 - fail closed on missing validator dependency
        raise RuntimeError("jsonschema dependency unavailable; refusing to validate M1 payload") from exc
    schema = read_json(schema_path)
    try:
        Draft7Validator.check_schema(schema)
    except SchemaError as exc:
        raise RuntimeError(f"M1 JSON Schema is invalid ({label_text}): {exc.message}") from exc
    errors = sorted(Draft7Validator(schema).iter_errors(instance), key=lambda e: list(e.absolute_path))
    if errors:
        rendered = []
        for error in errors[:20]:
            location = "$"
            if getattr(error, "absolute_path", None):
                location = "$." + ".".join(str(item) for item in error.absolute_path)
            rendered.append(f"{location}: {error.message}")
        raise RuntimeError(f"M1 {label_text} schema validation failed: " + "; ".join(rendered))


# --------------------------------------------------------------------------- #
# SLURM / machine-plan guards                                                  #
# --------------------------------------------------------------------------- #
def _forbid_ambient_gpu_when_cpu_only() -> None:
    for key in ("SLURM_GPUS", "SLURM_GPUS_ON_NODE", "SLURM_STEP_GPUS", "SLURM_JOB_GPUS"):
        value = os.environ.get(key)
        if value and value not in {"0", "(null)", "NoDevFiles"}:
            raise RuntimeError(f"M1 seal is CPU-only but {key}={value}")


def _require_common_slurm() -> None:
    if not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("M1 run must run under SLURM")
    if os.environ.get("CONDA_DEFAULT_ENV") != "HateVideo":
        raise RuntimeError("expected conda environment HateVideo")
    cpus = os.environ.get("SLURM_CPUS_PER_TASK")
    if cpus and int(cpus) != 4:
        raise RuntimeError(f"M1 requires exactly 4 CPU, got {cpus}")
    mem = os.environ.get("SLURM_MEM_PER_NODE") or os.environ.get("SLURM_MEM_PER_CPU")
    if mem and int(mem) not in {32768, 32000, 32}:
        raise RuntimeError(f"M1 requires 32GB memory allocation, got {mem}")


def require_slurm_cache() -> None:
    """GPU cache producer: 4 CPU / 32 GB / GPU under HateVideo.

    GPU presence is asserted via CUDA_VISIBLE_DEVICES being non-empty -- the project's
    accepted, proven idiom (sq_common.require_runtime(gpu=True),
    lb_scgp_common.require_slurm(expected_gpu=True)). We do NOT string-compare the
    last-seen SLURM_*_GPUS var to "1": SLURM_JOB_GPUS/SLURM_STEP_GPUS hold the global
    device ID(s) (e.g. "3"), not the count, so an exactly-1 compare false-fails on a
    correctly-provisioned 1-GPU allocation whose device id != "1". The exactly-one-GPU
    budget is enforced upstream by the sbatch (--gres=gpu:a100:1) and the
    m1_cache_parallel_max2 concurrency cap (<= 2 GPU total).
    """
    _require_common_slurm()
    if not os.environ.get("CUDA_VISIBLE_DEVICES"):
        raise RuntimeError("M1 cache producer (GPU) has no CUDA_VISIBLE_DEVICES")


def require_slurm_seal() -> None:
    """CPU seal: 4 CPU / 32 GB / 0 GPU under HateVideo."""
    _require_common_slurm()
    _forbid_ambient_gpu_when_cpu_only()


def expected_cache_slurm_block() -> dict[str, Any]:
    return {"cpu": 4, "ram_gb": 32, "gpu": 1, "env": "HateVideo", "no_time_flag": True}


def expected_seal_slurm_block() -> dict[str, Any]:
    return {"cpu": 4, "ram_gb": 32, "gpu": 0, "env": "HateVideo", "no_time_flag": True}


def verify_machine_cache(cfg: dict[str, Any], dataset: str) -> dict[str, Any]:
    """Assert the authoritative machine plan runs[4|5] matches the config, in lock-step."""
    run_id = DATASET_RUN[dataset]
    index = RUN_INDEX[run_id]
    machine_path = cfg["paths"]["experiment_machine"]
    machine_hash = sha256_file(canonical_root_path(machine_path)[0])
    machine = read_json(machine_path)
    run = machine["runs"][index]
    assert_equal(machine["run_order"][index], run_id, f"machine run_order[{index}]")
    assert_equal(run["run_id"], run_id, "machine cache run_id")
    assert_equal(run["milestone"], "M1", "machine cache milestone")
    assert_equal(run["dataset"], dataset, "machine cache dataset")
    assert_equal(run["artifact_paths"], [cfg["run"]["artifact_path"]], "machine cache artifact path")
    assert_equal(run["artifact_schema_ids"], [REPLICA_SCHEMA_ID], "machine cache schema id")
    assert_equal(run["slurm"], expected_cache_slurm_block(), "machine cache slurm block (post-amendment)")
    assert_equal(run["slurm"], cfg["run"]["slurm"], "machine cache slurm==config")
    assert_equal(run["dependencies"], [REALBANK_DEP], "machine cache dependency")
    assert_equal(run["model_pin"]["model_id"], MODEL_ID, "machine cache model pin")
    assert_equal(int(run["evidence_pack_protocol"]["frames"]["num_frames"]), NUM_FRAMES, "machine cache frames")
    assert_equal(int(run["replica_protocol"]["replicas"]), REPLICAS, "machine cache replicas")
    assert_equal(bool(run["ocr_policy"]["live_ocr"]), False, "machine cache ocr policy")
    return {"machine_sha256": machine_hash, "machine_run_record_run_id": run["run_id"]}


def verify_machine_seal(cfg: dict[str, Any]) -> dict[str, Any]:
    index = RUN_INDEX[RUN_SEAL]
    machine_path = cfg["paths"]["experiment_machine"]
    machine_hash = sha256_file(canonical_root_path(machine_path)[0])
    machine = read_json(machine_path)
    run = machine["runs"][index]
    assert_equal(machine["run_order"][index], RUN_SEAL, f"machine run_order[{index}]")
    assert_equal(run["run_id"], RUN_SEAL, "machine seal run_id")
    assert_equal(run["milestone"], "M1", "machine seal milestone")
    assert_equal(run["artifact_paths"], [cfg["run"]["artifact_path"]], "machine seal artifact path")
    assert_equal(run["artifact_schema_ids"], [SEAL_SCHEMA_ID], "machine seal schema id")
    assert_equal(run["slurm"], expected_seal_slurm_block(), "machine seal slurm block")
    assert_equal(run["slurm"], cfg["run"]["slurm"], "machine seal slurm==config")
    assert_equal(sorted(run["dependencies"]), sorted([RUN_MHC, RUN_MHC_ZH]), "machine seal dependencies")
    return {"machine_sha256": machine_hash, "machine_run_record_run_id": run["run_id"]}


# --------------------------------------------------------------------------- #
# Train-evidence access ledger (every val/test/held/label/query/... read forbidden) #
# --------------------------------------------------------------------------- #
FORBIDDEN_TOKENS = (
    "val", "test", "dev_seen", "test_seen", "held",
    "query_z", "query_labels", "teacher", "certificate",
    "margin", "prediction", "neighbor", "compiler_target",
)


def forbidden_reason(rel: str, allowlist: set[str]) -> str | None:
    """Return a rejection reason if `rel` is a forbidden read, else None.

    Only the preregistered train-evidence paths (train gt title source, train ASR,
    and the dataset's train video directory) may be opened.  Any validation/test/
    held path or cache/teacher/query path is forbidden.  Pure classifier: performs
    no read, mutates no counter.
    """
    lowered = rel.lower()
    for token in FORBIDDEN_TOKENS:
        if token in lowered:
            # allow the literal allowlisted train paths even if a token substring
            # coincidentally appears (none of the allowlisted train paths contain a
            # forbidden token by construction; this guard is defense-in-depth).
            if rel in allowlist:
                return None
            return f"forbidden token {token!r} in path"
    if rel in allowlist:
        return None
    if rel.startswith("data/gt/") or rel.startswith("data/ASR/") or rel.startswith("data/video/"):
        return "non-allowlisted dataset path"
    return None


class TrainEvidenceAccessLedger:
    """Records the train-evidence reads and holds the forbidden zero-counters.

    Train **labels** are never read: the title/ASR loaders extract only id + text /
    id + transcript, so train_label_read_count stays 0 (enforced at the code level
    and attested by the zero-gold grep self-test in the freeze doc).
    """

    def __init__(self, allowlist: set[str]) -> None:
        self.records: list[dict[str, Any]] = []
        self.counters = {key: 0 for key in ZERO_COUNTER_KEYS}
        self.allowlist = set(allowlist)
        self.authorized_train_evidence_read_count = 0

    def open_evidence(self, path: Path | str, kind: str, dataset: str) -> Path:
        fs_path, rel_path = canonical_root_path(path)
        rel = rel_path.as_posix()
        reason = forbidden_reason(rel, self.allowlist)
        if reason is not None:
            self.counters["forbidden_path_read_count"] += 1
            raise RuntimeError(f"access ledger refuses {rel}: {reason}")
        if rel not in self.allowlist:
            self.counters["non_allowlisted_train_content_read_count"] += 1
            raise RuntimeError(f"train-evidence path not on allowlist: {rel}")
        self.authorized_train_evidence_read_count += 1
        self.records.append({"kind": kind, "dataset": dataset, "path": rel})
        return fs_path

    def note_video_read(self, path: Path | str, dataset: str) -> Path:
        """A single train-video read (frame decode or byte hash); allowlisted by dir."""
        fs_path, rel_path = canonical_root_path(path)
        rel = rel_path.as_posix()
        video_dir = f"data/video/{dataset}/All/"
        if not rel.startswith(video_dir):
            self.counters["non_allowlisted_train_content_read_count"] += 1
            raise RuntimeError(f"train-video path outside allowlisted dir: {rel}")
        reason = forbidden_reason(rel, self.allowlist | {rel})
        if reason is not None:
            self.counters["forbidden_path_read_count"] += 1
            raise RuntimeError(f"access ledger refuses video {rel}: {reason}")
        self.authorized_train_evidence_read_count += 1
        return fs_path

    def fields(self, dataset: str) -> dict[str, Any]:
        return {
            "schema_version": "lb_scgp_global_r2_m1_access_ledger_v1",
            "dataset": dataset,
            "access_ledger": self.records,
            "access_ledger_sha256": sha256_obj(self.records),
            "zero_counters": dict(self.counters),
            "authorized_train_evidence_read_count": self.authorized_train_evidence_read_count,
        }


# --------------------------------------------------------------------------- #
# Merkle root + consensus over the R=4 replica bank                            #
# --------------------------------------------------------------------------- #
def merkle_root(leaf_hashes: list[str]) -> str:
    """Deterministic binary Merkle root over sorted leaf hashes (duplicate-last)."""
    if not leaf_hashes:
        return sha256_bytes(b"")
    level = sorted(leaf_hashes)
    while len(level) > 1:
        if len(level) % 2 == 1:
            level = level + [level[-1]]
        nxt = []
        for i in range(0, len(level), 2):
            nxt.append(sha256_bytes((level[i] + level[i + 1]).encode("ascii")))
        level = nxt
    return level[0]


def record_leaf_hash(record: dict[str, Any]) -> str:
    return sha256_obj(record)


def consensus_for_video(replica_records: list[dict[str, Any]]) -> dict[str, Any]:
    """Majority-vote consensus over the R replicas of one video.

    Each of the eight observables maps supported=+1 / contradicted=-1 /
    unresolved=0; the per-observable consensus is the sign of the vote sum
    (ties -> 0).  Confidence and modality are reported as diagnostics only; they
    are NOT weights or selectors.  No label is used.
    """
    consensus: dict[str, Any] = {}
    for key in OBSERVABLE_KEYS:
        vote = 0
        for record in replica_records:
            state = record["observables"][key]["state"]
            vote += CONSENSUS_MAP.get(state, 0)
        consensus[key] = 1 if vote > 0 else (-1 if vote < 0 else 0)
    # modality consensus: plurality of states, deterministic tie-break by MODALITY_STATES order
    modality_counts = {s: 0 for s in MODALITY_STATES}
    for record in replica_records:
        modality_counts[record["observables"][MODALITY_KEY]["state"]] += 1
    best = max(MODALITY_STATES, key=lambda s: (modality_counts[s], -MODALITY_STATES.index(s)))
    consensus[MODALITY_KEY] = best
    return consensus


# --------------------------------------------------------------------------- #
# Restricted scgp_global_cert_v2 prompt + parser                              #
# --------------------------------------------------------------------------- #
SYSTEM_PROMPT = (
    "You are a careful, deterministic forensic annotator for a hateful-video "
    "detection research pipeline. You are shown uniformly sampled frames of ONE "
    "short video together with its title and spoken transcript. You emit ONLY a "
    "single strict JSON object of structural observables. You never output free "
    "text, explanations, verdicts, target names, propositions, mechanisms, "
    "timestamps, spans, localization, rationales, or stance labels."
)

# The user prompt enumerates the exact scgp_global_cert_v2 fields and states. Each
# observable is a NOISY structural cue, not gold. confidence is 0..4 for parse/
# stability diagnostics only, never a weight or selector.
USER_PROMPT = (
    "Look at the sampled frames, the Title, and the Transcript below. Output ONE "
    "JSON object with EXACTLY these keys and no others.\n"
    "For each of the eight observables, \"state\" is one of "
    "\"supported\" | \"contradicted\" | \"unresolved\" and \"confidence\" is an "
    "integer 0..4:\n"
    "- visual_reference_observable: the frames plausibly reference a group, person, "
    "symbol, or topic that could be relevant to hateful content.\n"
    "- text_audio_reference_observable: the title/transcript plausibly reference "
    "such a group, person, or topic.\n"
    "- harmful_surface_observable: any surface cue of offensive/harmful content is "
    "present in any channel.\n"
    "- dehumanizing_or_threat_surface_observable: any surface cue of dehumanizing, "
    "demeaning, or threatening content is present.\n"
    "- cross_modal_binding_observable: the harmful/reference cues bind across "
    "modalities (e.g. on-screen text + speech + imagery point the same way).\n"
    "- source_alignment_observable: the title/transcript source aligns with what the "
    "frames show.\n"
    "- counter_context_observable: there is a counter-speech / reclaiming / educational "
    "/ news framing that recontextualizes any harmful surface.\n"
    "- context_shift_observable: the video's framing shifts across its span in a way "
    "that changes how a surface cue should be read.\n"
    "Then add modality_binding_observable with \"state\" one of "
    "\"visual_text\" | \"visual_audio\" | \"text_audio\" | \"multi_modal\" | "
    "\"single_modal\" | \"unresolved\" and \"confidence\" 0..4 (which channels carry "
    "the main evidence).\n"
    "Finally add \"parse_flags\": [] (leave empty; the harness fills it).\n"
    "Output ONLY the JSON object.\n\n"
    "Title: {title}\n\nTranscript: {transcript}"
)


def build_user_prompt(title: str, transcript: str) -> str:
    t = (title or "").strip() or "(none)"
    x = (transcript or "").strip() or "(none)"
    return USER_PROMPT.format(title=t, transcript=x)


def canonical_unresolved_observables() -> dict[str, Any]:
    obs: dict[str, Any] = {}
    for key in OBSERVABLE_KEYS:
        obs[key] = {"state": "unresolved", "confidence": 0}
    obs[MODALITY_KEY] = {"state": "unresolved", "confidence": 0}
    return obs


def parse_certificate(raw_text: str) -> tuple[dict[str, Any], list[str]]:
    """Parse a raw MLLM string into (observables, parse_flags).

    Strict: extra keys, missing keys, bad states, or out-of-range confidence are
    parse failures.  On ANY failure the observables fall back to canonical
    all-unresolved (no prompt/schema rescue) and the reason is recorded in
    parse_flags.  Returns validated observables + flags.
    """
    flags: list[str] = []
    text = (raw_text or "").strip()
    # tolerate a leading/trailing code fence but nothing else structural
    if text.startswith("```"):
        text = text.strip("`")
        if text.lower().startswith("json"):
            text = text[4:]
        text = text.strip()
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end < 0 or end < start:
        return canonical_unresolved_observables(), ["no_json_object"]
    if start != 0 or end != len(text) - 1:
        flags.append("extra_text_around_json")
    blob = text[start:end + 1]
    try:
        parsed = json.loads(blob)
    except Exception:  # noqa: BLE001
        return canonical_unresolved_observables(), ["json_decode_error"]
    if not isinstance(parsed, dict):
        return canonical_unresolved_observables(), ["json_not_object"]

    expected_keys = set(OBSERVABLE_KEYS) | {MODALITY_KEY, "parse_flags"}
    extra = sorted(set(parsed.keys()) - expected_keys)
    if extra:
        flags.append("extra_keys:" + ",".join(extra))
        return canonical_unresolved_observables(), flags

    observables: dict[str, Any] = {}
    for key in OBSERVABLE_KEYS:
        node = parsed.get(key)
        if not isinstance(node, dict) or "state" not in node or "confidence" not in node:
            flags.append(f"missing_or_malformed:{key}")
            return canonical_unresolved_observables(), flags
        state = node["state"]
        conf = node["confidence"]
        if state not in STATE_VALUES:
            flags.append(f"bad_state:{key}")
            return canonical_unresolved_observables(), flags
        if not isinstance(conf, int) or isinstance(conf, bool) or not (CONF_MIN <= conf <= CONF_MAX):
            flags.append(f"bad_confidence:{key}")
            return canonical_unresolved_observables(), flags
        observables[key] = {"state": state, "confidence": conf}

    mnode = parsed.get(MODALITY_KEY)
    if not isinstance(mnode, dict) or "state" not in mnode or "confidence" not in mnode:
        flags.append(f"missing_or_malformed:{MODALITY_KEY}")
        return canonical_unresolved_observables(), flags
    mstate, mconf = mnode["state"], mnode["confidence"]
    if mstate not in MODALITY_STATES:
        flags.append(f"bad_state:{MODALITY_KEY}")
        return canonical_unresolved_observables(), flags
    if not isinstance(mconf, int) or isinstance(mconf, bool) or not (CONF_MIN <= mconf <= CONF_MAX):
        flags.append(f"bad_confidence:{MODALITY_KEY}")
        return canonical_unresolved_observables(), flags
    observables[MODALITY_KEY] = {"state": mstate, "confidence": mconf}
    return observables, flags


def cert_v2_object(observables: dict[str, Any], parse_flags: list[str]) -> dict[str, Any]:
    """Reconstruct a full scgp_global_cert_v2 object from the record's observables.

    Used to cross-validate the observables against the Run1-FROZEN
    scgp_global_cert_v2.schema.json (making that frozen contract a live dependency),
    while the cache record itself hoists schema_version + parse_flags out of the cert
    to conform to the cache_replica_v2 artifact-schema field list.
    """
    cert = {"schema_version": CERT_SCHEMA_VERSION, "parse_flags": list(parse_flags)}
    cert.update(observables)
    return cert


def make_replica_record(video_id: str, evidence_pack_sha256: str, replica_index: int,
                        observables: dict[str, Any], parse_flags: list[str]) -> dict[str, Any]:
    """Assemble a strict scgp_global_cache_replica_v2 record (exactly six keys)."""
    return {
        "video_id": video_id,
        "evidence_pack_sha256": evidence_pack_sha256,
        "replica_index": int(replica_index),
        "schema_version": CERT_SCHEMA_VERSION,
        "observables": observables,
        "parse_flags": list(parse_flags),
    }


# --------------------------------------------------------------------------- #
# Provenance hashes for the seal (prompt / input builder / model+processor)     #
# --------------------------------------------------------------------------- #
def prompt_hash() -> str:
    return sha256_obj({
        "system_prompt": SYSTEM_PROMPT,
        "user_prompt_template": USER_PROMPT,
        "cert_schema_version": CERT_SCHEMA_VERSION,
        "observable_keys": list(OBSERVABLE_KEYS),
        "modality_key": MODALITY_KEY,
        "state_values": list(STATE_VALUES),
        "modality_states": list(MODALITY_STATES),
    })


def input_builder_hash(builder_source_sha256: str, common_source_sha256: str) -> str:
    return sha256_obj({
        "evidence_pack_builder_sha256": builder_source_sha256,
        "common_sha256": common_source_sha256,
        "num_frames": NUM_FRAMES,
        "frame_rule": FRAME_RULE,
        "max_title_chars": MAX_TITLE_CHARS,
        "max_asr_chars": MAX_ASR_CHARS,
    })


def model_processor_hash() -> str:
    return sha256_obj({
        "model_id": MODEL_ID,
        "decoding": {"do_sample": False, "temperature": 0.0, "num_beams": 1,
                     "max_new_tokens": MAX_NEW_TOKENS},
        "processor": {"images": None, "videos_are_frame_lists": True, "num_frames": NUM_FRAMES},
        "replicas": REPLICAS,
        "offline": "HF_HUB_OFFLINE=1",
    })
