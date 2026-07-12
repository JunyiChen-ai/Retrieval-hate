#!/usr/bin/env python
"""Shared frozen utilities for SSR B0/B1.

The only gold target used anywhere in this package is the parent video's
binary label.  No segment boundary, segment label, stance, target, mechanism,
rationale, or localization field is read as gold supervision.
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import tempfile
from collections import Counter
from pathlib import Path


ROOT = Path("/data/jehc223/RGCL")

RELATION_SCHEMA = {
    "target_relation": ("same", "different", "unclear"),
    "proposition_relation": ("same", "opposed", "unrelated", "unclear"),
    "stance_a": ("endorse", "quote", "report", "condemn", "satire", "unclear"),
    "stance_b": ("endorse", "quote", "report", "condemn", "satire", "unclear"),
    "stance_relation": ("same", "opposed", "unclear"),
    "mechanism_a": (
        "slur", "dehumanization", "threat", "exclusion", "violence_praise",
        "stereotype", "harassment", "none", "unclear",
    ),
    "mechanism_b": (
        "slur", "dehumanization", "threat", "exclusion", "violence_praise",
        "stereotype", "harassment", "none", "unclear",
    ),
    "mechanism_relation": ("same", "different", "unclear"),
    "topic_surface_relation": ("same", "different", "unclear"),
}

PROMPT_P0 = """Compare Video A and Video B only as content-relation evidence.
Do not decide whether either video is hateful, harmful, offensive, or benign.
Do not infer dataset labels or a training role. The frames and automatic text
may be noisy. Describe only the requested categorical relations. Output one
JSON object with exactly these keys and allowed values:
{schema}

[VIDEO A AUTOMATIC FULL-VIDEO EVIDENCE]
{evidence_a}

[VIDEO B AUTOMATIC FULL-VIDEO EVIDENCE]
{evidence_b}

Return JSON only. No rationale, scores, spans, timestamps, localization, or
segment fields."""

PROMPT_P1 = """Given two complete-video evidence bundles, encode their semantic
relationship without judging either video's moderation class. Treat every
frame/transcript/OCR item as noisy model input, never as annotation. Produce
exactly one JSON object and nothing else, using exactly this schema:
{schema}

Evidence bundle A:
{evidence_a}

Evidence bundle B:
{evidence_b}

Do not add a verdict, label, confidence, explanation, rationale, time span,
localization, or segment output."""

SYSTEM_PROMPT = (
    "You are a relation encoder for a train-only academic diagnostic. "
    "You never classify either video and always emit strict JSON."
)


def canonical_json(obj) -> str:
    return json.dumps(obj, ensure_ascii=False, sort_keys=True,
                      separators=(",", ":"))


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_obj(obj) -> str:
    return sha256_text(canonical_json(obj))


def sha256_file(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(1024 * 1024)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def load_config(path, require_frozen=True):
    with open(path, encoding="utf-8") as f:
        cfg = json.load(f)
    payload = dict(cfg)
    stored = payload.pop("config_sha256", None)
    computed = sha256_obj(payload)
    if require_frozen and stored != computed:
        raise RuntimeError(
            "config is not frozen: stored={} computed={}".format(stored, computed))
    cfg["computed_config_sha256"] = computed
    return cfg


def resolve(cfg, key):
    p = Path(cfg["paths"][key])
    return p if p.is_absolute() else Path(cfg["paths"]["root"]) / p


def read_jsonl(path):
    out = []
    with open(path, encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except Exception as exc:
                raise ValueError("{}:{}: {}".format(path, lineno, exc))
    return out


def atomic_write_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=path.name + ".", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(obj, f, ensure_ascii=False, sort_keys=True, indent=2)
            f.write("\n")
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def atomic_write_jsonl(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=path.name + ".", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            for row in rows:
                f.write(canonical_json(row) + "\n")
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def id_hash(salt, *parts):
    return sha256_text("|".join([str(salt)] + [str(x) for x in parts]))


def head_tail(text, max_chars):
    text = str(text or "").strip()
    if len(text) <= max_chars:
        return text
    marker = "\n...[HEAD/TAIL TRUNCATED]...\n"
    left = (max_chars - len(marker)) // 2
    right = max_chars - len(marker) - left
    return text[:left] + marker + text[-right:]


def schema_for_prompt():
    return canonical_json({k: "|".join(v) for k, v in RELATION_SCHEMA.items()})


def build_prompt(version, evidence_a, evidence_b):
    template = {"P0": PROMPT_P0, "P1": PROMPT_P1}[version]
    return template.format(
        schema=schema_for_prompt(), evidence_a=evidence_a,
        evidence_b=evidence_b)


def extract_json_candidate(raw):
    """Return the complete trimmed response only when it is one JSON object.

    Code fences, leading/trailing prose, brace extraction, and repair are
    deliberately forbidden by the frozen B0 protocol.
    """
    text = str(raw or "").strip()
    if not text.startswith("{") or not text.endswith("}"):
        return None
    return text


def strict_parse_relation(raw):
    candidate = extract_json_candidate(raw)
    if candidate is None:
        return None, "no_json_object"
    try:
        obj = json.loads(candidate)
    except Exception as exc:
        return None, "json_error:{}".format(exc)
    if not isinstance(obj, dict):
        return None, "not_object"
    if set(obj) != set(RELATION_SCHEMA):
        return None, "keys_mismatch"
    clean = {}
    for key, allowed in RELATION_SCHEMA.items():
        value = obj[key]
        if not isinstance(value, str):
            return None, "non_string:{}".format(key)
        if value not in allowed:
            return None, "bad_value:{}:{}".format(key, value)
        clean[key] = value
    return clean, None


def canonicalize_order(obj, order):
    obj = dict(obj)
    if order == "BA":
        obj["stance_a"], obj["stance_b"] = obj["stance_b"], obj["stance_a"]
        obj["mechanism_a"], obj["mechanism_b"] = (
            obj["mechanism_b"], obj["mechanism_a"])
    return obj


def modal_field(values, unclear="unclear"):
    counts = Counter(values)
    best = max(counts.values())
    winners = sorted(k for k, v in counts.items() if v == best)
    value = winners[0] if len(winners) == 1 else unclear
    return value, best / float(len(values))


def calls_to_record(pair_id, calls):
    expected = [(p, o) for p in ("P0", "P1") for o in ("AB", "BA")]
    by_key = {(c["prompt_version"], c["order"]): c for c in calls}
    complete = all(k in by_key for k in expected) and len(by_key) == 4
    if not complete or any(by_key[k].get("parsed") is None for k in expected):
        return {
            "canonical_pair_id": pair_id,
            "status": "missing/no_edge",
            "complete_four_calls": complete,
            "fields": None,
            "failure_reasons": [by_key[k].get("parse_error") if k in by_key
                                else "missing_call" for k in expected],
        }
    canon = [canonicalize_order(by_key[k]["parsed"], k[1]) for k in expected]
    fields = {}
    for key in RELATION_SCHEMA:
        value, rho = modal_field([x[key] for x in canon])
        fields[key] = {"value": value, "rho": rho}
    return {
        "canonical_pair_id": pair_id,
        "status": "relation",
        "complete_four_calls": True,
        "fields": fields,
        "failure_reasons": [],
    }


def relation_family(record, label_a, label_b, min_rho=0.75):
    """Apply labels only after the label-blind record is frozen.

    Returns (family|None, predicate|None, rho|None).  A low-agreement or
    unclear required field yields no edge; it never receives a semantic type
    from the video label.
    """
    if record.get("status") != "relation" or not record.get("fields"):
        return None, None, None
    fields = record["fields"]

    def usable(key, wanted):
        x = fields[key]
        return x["value"] in wanted and float(x["rho"]) >= min_rho

    equal = int(label_a) == int(label_b)
    if equal:
        if not usable("mechanism_relation", {"same"}):
            return None, None, None
        for key, wanted in [
                ("topic_surface_relation", {"different"}),
                ("target_relation", {"different"})]:
            if usable(key, wanted):
                rho = min(float(fields["mechanism_relation"]["rho"]),
                          float(fields[key]["rho"]))
                return "MI", "mechanism_relation=same&{}={}".format(
                    key, fields[key]["value"]), rho
        return None, None, None
    if not usable("topic_surface_relation", {"same"}):
        return None, None, None
    for key, wanted in [
            ("stance_relation", {"opposed"}),
            ("target_relation", {"different"}),
            ("proposition_relation", {"opposed", "unrelated"})]:
        if usable(key, wanted):
            rho = min(float(fields["topic_surface_relation"]["rho"]),
                      float(fields[key]["rho"]))
            return "SC", "topic_surface_relation=same&{}={}".format(
                key, fields[key]["value"]), rho
    return None, None, None


FORBIDDEN_PAYLOAD_KEYS = {
    "label", "labels", "gold", "prediction", "pred", "rank", "margin",
    "fold", "seed", "loss", "event", "y_sc", "y_mi", "family",
}


def forbidden_payload_keys(obj, path=""):
    hits = []
    if isinstance(obj, dict):
        for key, value in obj.items():
            here = "{}.{}".format(path, key) if path else str(key)
            if str(key).strip().lower() in FORBIDDEN_PAYLOAD_KEYS:
                hits.append(here)
            hits.extend(forbidden_payload_keys(value, here))
    elif isinstance(obj, list):
        for i, value in enumerate(obj):
            hits.extend(forbidden_payload_keys(value, "{}[{}]".format(path, i)))
    return hits


def exact_vote(neighbors, topk=20):
    top = list(neighbors)[:topk]
    if len(top) != topk:
        raise ValueError("need exactly {} neighbors, got {}".format(topk, len(top)))
    vote = 0.0
    denom = 0.0
    for rank, n in enumerate(top, 1):
        weight = topk + 1 - rank
        sim = float(n["cosine"])
        vote += weight * sim * (2 * int(n["label"]) - 1)
        denom += weight * abs(sim)
    return vote, int(vote >= 0.0), denom


def rank_bin(rank):
    rank = int(rank)
    if rank <= 5:
        return "1-5"
    if rank <= 10:
        return "6-10"
    if rank <= 15:
        return "11-15"
    if rank <= 20:
        return "16-20"
    return ">20"


def wilson(valid, n, z=1.959963984540054):
    if n <= 0:
        return {"point": None, "lower": None, "upper": None}
    p = valid / float(n)
    den = 1.0 + z * z / n
    centre = (p + z * z / (2.0 * n)) / den
    half = z * math.sqrt((p * (1.0 - p) + z * z / (4.0 * n)) / n) / den
    return {"point": p, "lower": centre - half, "upper": centre + half}


def read_audit_csv(path):
    rows = []
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            rows.append({k: (v or "").strip() for k, v in row.items()})
    return rows
