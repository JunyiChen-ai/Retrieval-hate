#!/usr/bin/env python
"""TERA Gate-0 — test-seal guard and the sealed-id restricted reader.

Implements appendix sec 2.8 (BLOCKING-FIX B-4) and sec 10.4:

* `SealGuard` wraps `builtins.open` process-wide.  A resolved path matching a
  forbidden path increments `test_contact_count`, is appended to
  `opened_test_paths`, and raises `HALT_TEST_CONTACT`.  A registered
  corpus-spanning artifact opened outside `load_corpus_spanning` raises
  `HALT_UNRESTRICTED_GOLD_HANDLE`.
* `Authorization` holds the currently authorized id set per dataset and the
  single `unlock_confirmation()` phase switch (a second call is a HALT).
* `load_corpus_spanning` is the only admissible reader for the registered
  corpus-spanning artifacts.  It keeps `authorized_ids & file_ids` (whitelist
  semantics) and discards every other entry BEFORE returning.
"""
from __future__ import annotations

import builtins
import datetime as _dt
import json
import os
from pathlib import Path

from .common import TeraHalt, sha256_ids

CORPUS_SPANNING_RELPATHS = (
    "gt/HateMM/hate_spans.json",
    "gt/HateClipSeg/gold_segments.json",
    "gt/HateClipSeg/video_durations.jsonl",
    "gt/HateClipSeg/test.jsonl",
    "CLIP_Embedding/HateClipSeg/test_seen_subclipK30_openai_clip-vit-large-patch14-336_HF.pt",
    "CLIP_Embedding/HateClipSeg/test_seen_openai_clip-vit-large-patch14-336_HF.pt",
)

FORBIDDEN_EXACT_RELPATHS = ("gt/HateMM/test.jsonl",)
FORBIDDEN_PREFIX_RELPATHS = ("CLIP_Embedding/HateMM/test_seen_",)

WHITELIST_EXACT_RELPATHS = (
    "gt/HateClipSeg/test.jsonl",
    "CLIP_Embedding/HateClipSeg/test_seen_subclipK30_openai_clip-vit-large-patch14-336_HF.pt",
    "CLIP_Embedding/HateClipSeg/test_seen_openai_clip-vit-large-patch14-336_HF.pt",
)


class SealGuard(object):
    """Process-wide file-open wrapper (appendix sec 10.4)."""

    _installed = None

    def __init__(self, data_root):
        self.data_root = Path(data_root).resolve()
        self.forbidden_exact = {str(self.data_root / r) for r in FORBIDDEN_EXACT_RELPATHS}
        self.forbidden_prefix = tuple(str(self.data_root / r) for r in FORBIDDEN_PREFIX_RELPATHS)
        self.whitelist_exact = {str(self.data_root / r) for r in WHITELIST_EXACT_RELPATHS}
        self.corpus_spanning = {str(self.data_root / r) for r in CORPUS_SPANNING_RELPATHS}
        self.test_contact_count = 0
        self.opened_test_paths = []
        self._reader_depth = 0
        self._real_open = None

    # -- installation -------------------------------------------------------
    def install(self):
        if SealGuard._installed is not None:
            SealGuard._installed.uninstall()
        self._real_open = builtins.open
        guard = self

        def _guarded_open(file, *args, **kwargs):
            try:
                path = os.fspath(file)
            except TypeError:
                path = None
            if isinstance(path, (str, bytes)):
                if isinstance(path, bytes):
                    path = path.decode("utf-8", "replace")
                guard.check(os.path.abspath(path))
            return guard._real_open(file, *args, **kwargs)

        builtins.open = _guarded_open
        SealGuard._installed = self
        return self

    def uninstall(self):
        if self._real_open is not None:
            builtins.open = self._real_open
            self._real_open = None
        if SealGuard._installed is self:
            SealGuard._installed = None

    # -- checks -------------------------------------------------------------
    def check(self, abspath):
        if abspath in self.whitelist_exact:
            pass
        elif abspath in self.forbidden_exact or abspath.startswith(self.forbidden_prefix):
            self.test_contact_count += 1
            self.opened_test_paths.append(abspath)
            raise TeraHalt("HALT_TEST_CONTACT", abspath)
        if abspath in self.corpus_spanning and self._reader_depth == 0:
            raise TeraHalt("HALT_UNRESTRICTED_GOLD_HANDLE", abspath)

    class _ReaderScope(object):
        def __init__(self, guard):
            self.guard = guard

        def __enter__(self):
            self.guard._reader_depth += 1
            return self.guard

        def __exit__(self, *exc):
            self.guard._reader_depth -= 1
            return False

    def reader_scope(self):
        return SealGuard._ReaderScope(self)

    def report(self):
        return {
            "test_contact_count": self.test_contact_count,
            "opened_test_paths": list(self.opened_test_paths),
            "forbidden_paths": sorted(self.forbidden_exact) +
                               [p + "*" for p in self.forbidden_prefix],
            "whitelist_exact": sorted(self.whitelist_exact),
        }


def active_guard():
    return SealGuard._installed


# ------------------------------------------------------------ authorization --
class Authorization(object):
    """Authorized id sets and the single confirmation phase switch (sec 2.8)."""

    def __init__(self, development, confirmation):
        """`confirmation` may be a dict or a zero-argument factory.

        The factory form keeps `val.jsonl` unopened until `unlock_confirmation()`
        (appendix sec 2.9: val is loaded exactly once, at confirmation time).
        """
        self.development = {k: set(v) for k, v in development.items()}
        self._confirmation_factory = confirmation if callable(confirmation) else None
        self.confirmation = ({k: set(v) for k, v in confirmation.items()}
                             if not callable(confirmation) else {})
        self.phase = "development"
        self.unlock_calls = 0
        self.confirmation_unlock_utc = None
        self.hash_history = [{"phase": "development",
                              "hash": self.id_hash_all()}]
        self.sealed_ids_dropped = {}
        self.hateclipseg_test_ids = set()

    def authorized(self, dataset):
        table = self.development if self.phase == "development" else self.confirmation
        if dataset not in table:
            raise TeraHalt("HALT_UNAUTHORIZED_DATASET", dataset)
        return table[dataset]

    def id_hash(self, dataset):
        return sha256_ids(self.authorized(dataset))

    def id_hash_all(self):
        table = self.development if self.phase == "development" else self.confirmation
        return {ds: sha256_ids(ids) for ds, ids in sorted(table.items())}

    def unlock_confirmation(self):
        self.unlock_calls += 1
        if self.unlock_calls > 1:
            raise TeraHalt("HALT_SECOND_UNLOCK",
                           "unlock_confirmation() called %d times" % self.unlock_calls)
        if self._confirmation_factory is not None:
            self.confirmation = {k: set(v) for k, v in self._confirmation_factory().items()}
        self.phase = "confirmation"
        self.confirmation_unlock_utc = _dt.datetime.now(_dt.timezone.utc).strftime(
            "%Y-%m-%dT%H:%M:%SZ")
        self.hash_history.append({"phase": "confirmation", "hash": self.id_hash_all()})
        return self.confirmation_unlock_utc

    def record_drop(self, dataset, path, n_dropped):
        self.sealed_ids_dropped.setdefault(dataset, {})[str(path)] = int(n_dropped)

    def dropped_totals(self):
        return {ds: int(sum(v.values())) for ds, v in sorted(self.sealed_ids_dropped.items())}


# ---------------------------------------------------------------- reader ----
def _restrict_mapping(obj, allowed):
    return {k: v for k, v in obj.items() if k in allowed}


def _restrict_pt(obj, allowed):
    """Restrict a segment or whole-video cache dict to the authorized ids."""
    import torch

    if "video_ids" in obj:                      # segment cache (flat ids)
        vids = list(obj["video_ids"])
        keep_idx = [i for i, v in enumerate(vids) if v in allowed]
        keep_set = set(keep_idx)
        parent = obj["subclip_parent"].tolist()
        rows = [r for r, p in enumerate(parent) if p in keep_set]
        remap = {old: new for new, old in enumerate(keep_idx)}
        out = dict(obj)
        out["video_ids"] = [vids[i] for i in keep_idx]
        out["subclip_img_feats"] = obj["subclip_img_feats"][torch.as_tensor(rows, dtype=torch.long)]
        out["subclip_parent"] = torch.as_tensor([remap[parent[r]] for r in rows],
                                                dtype=obj["subclip_parent"].dtype)
        out["labels"] = obj["labels"][torch.as_tensor(rows, dtype=torch.long)]
        return out, vids, out["video_ids"]
    if "ids" in obj:                            # whole-video cache (nested ids)
        vids = list(obj["ids"][0])              # appendix sec 2.2 / review F-1
        keep_idx = [i for i, v in enumerate(vids) if v in allowed]
        sel = torch.as_tensor(keep_idx, dtype=torch.long)
        out = dict(obj)
        out["ids"] = [[vids[i] for i in keep_idx]]
        for key in ("img_feats", "text_feats", "labels"):
            if key in obj:
                out[key] = obj[key][sel]
        return out, vids, out["ids"][0]
    raise TeraHalt("HALT_UNKNOWN_CACHE_SCHEMA", "neither video_ids nor ids present")


def load_corpus_spanning(path, dataset, auth, guard=None):
    """The single admissible reader for corpus-spanning gold artifacts (sec 2.8).

    Restricts to `authorized_ids & file_ids` and discards everything else before
    returning.  Assertions 1-4 of sec 2.8 are enforced; any failure is a HALT.
    """
    guard = guard if guard is not None else active_guard()
    path = Path(path)
    allowed = auth.authorized(dataset)

    scope = guard.reader_scope() if guard is not None else _NullScope()
    with scope:
        suffix = path.suffix.lower()
        if suffix == ".json":
            with open(path, encoding="utf-8") as handle:
                raw = json.load(handle)
            file_ids = list(raw.keys())
            restricted = _restrict_mapping(raw, allowed)
            kept_ids = list(restricted.keys())
        elif suffix == ".jsonl":
            raw = {}
            order = []
            with open(path, encoding="utf-8") as handle:
                for line in handle:
                    line = line.strip()
                    if not line:
                        continue
                    rec = json.loads(line)
                    key = rec.get("id", rec.get("video_id"))
                    raw[key] = rec
                    order.append(key)
            file_ids = order
            restricted = _restrict_mapping(raw, allowed)
            kept_ids = list(restricted.keys())
        elif suffix == ".pt":
            import torch

            obj = torch.load(path, map_location="cpu", weights_only=False)
            restricted, file_ids, kept_ids = _restrict_pt(obj, allowed)
        else:
            raise TeraHalt("HALT_UNKNOWN_GOLD_ARTIFACT", str(path))

    file_id_set = set(file_ids)
    kept_set = set(kept_ids)
    # assertion 1 -- only authorized ids survive (whitelist semantics)
    if not kept_set <= allowed:
        raise TeraHalt("HALT_UNAUTHORIZED_ID_SURVIVED", str(path))
    # assertion 2 -- exact intersection size
    if len(kept_set) != len(allowed & file_id_set):
        raise TeraHalt("HALT_RESTRICTION_SIZE", str(path))
    # assertion 3 -- the restriction actually ran
    dropped = len(file_id_set) - len(kept_set)
    if dropped <= 0:
        raise TeraHalt("HALT_ZERO_SEALED_DROP", str(path))
    # assertion 4 -- no HateClipSeg p11 test id survives
    if auth.hateclipseg_test_ids and kept_set & auth.hateclipseg_test_ids:
        raise TeraHalt("HALT_SEALED_TEST_ID_SURVIVED", str(path))

    auth.record_drop(dataset, path, dropped)
    return restricted


class _NullScope(object):
    def __enter__(self):
        return None

    def __exit__(self, *exc):
        return False
