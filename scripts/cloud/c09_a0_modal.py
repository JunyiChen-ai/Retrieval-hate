"""C09 A0 on Modal -- SUBSTRATE PORT of scripts/slurm/c09_a0_cpu.sbatch.

Authority: USER DIRECTIVE 2026-08-02 (the local SLURM approval gate has released
nothing since 2026-08-01 15:49; the user ordered this A0 moved to cloud).
Port record: refine-logs/C09_A0_CLOUD_PORT_RECORD.md.
Design of record: refine-logs/C09_A0_V17_RECORD.md (GO 0C/0H/0I, round 17).
Freeze record:    refine-logs/C09_A0_RECORD.md (the nine frozen sha256).

ZERO SCIENTIFIC CHANGE.  No frozen file is edited.  Same arms, thresholds, decision
rule, Feldman discriminator, label-use discipline and HALT-only validity gates.  What
changes is the scheduler and the machine.

THREE INVARIANTS THIS FILE EXISTS TO HOLD
  1. REPO PATH.  headspace_mint.py:62, mechnov_pairverify.py:51, c09_a0_arena.py:42 and
     c09guard.py:38 all hard-code "/data/jehc223/RGCL" as an absolute literal, and
     c09guard scopes its test-split PREDICATE to it.  The container therefore mounts
     everything at exactly that path.  Mounting anywhere else would silently DISABLE
     the guard (every path would fail startswith(REPO) and be judged non-test-like).
     PORT-CHECK-2 makes that a hard stop rather than a silent pass.
  2. SAME-TABLE-SAME-HARDWARE (user ruling 2026-07-31).  The entire A0 -- 36 mints,
     both GATE-DEVFID runs, the arena, every arm, floor and control, both tau, both
     datasets, all three spaces -- runs in ONE @app.function invocation, one container,
     one host.  Every internal comparison is same-host by construction.
  3. DATA BOUNDARY.  Only the explicitly enumerated frozen input closure is uploaded:
     derived float feature caches (.pt), label JSONLs, the banked fold .npz, the banked
     arena JSONs, the banked encoder .trainlogs, and the code subset.  NO raw video.  NO
     test-split artifact of any kind.  Every uploaded file passes the upstream
     modal_probe_runner media guard, unweakened (see c09_assert_uploadable).

The sbatch is TRANSLITERATED, not invoked: `sbatch` is a SLURM verb and there is no
SLURM in a container.  a0() reproduces its ordered steps 1-5 exactly and fails closed on
any non-zero return code, which is what `set -euo pipefail` buys.  The sbatch's own
sha256 is carried as a provenance anchor (PORT-CHECK-5): the port asserts it is
transliterating the reviewed file, unchanged.

Usage:
    modal run scripts/cloud/c09_a0_modal.py::preflight    # numerics probe, no A0 quantity
    modal run scripts/cloud/c09_a0_modal.py::a0           # the run
"""

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import modal

REPO = "/data/jehc223/RGCL"
REPO_LOCAL = Path(REPO)
RUN_ID = "C09-A0-v1-MODAL"
APP_NAME = "rgcl-c09-a0"
OUT_VOLUME = "rgcl-c09-a0-out"

# ---------------------------------------------------------------------------
# The nine frozen sha256 (refine-logs/C09_A0_RECORD.md section 3), verified locally
# before upload and again INSIDE the container (PORT-CHECK-5).
# ---------------------------------------------------------------------------
FROZEN_SHA256 = {
    "scripts/analysis/c09_a0_arena.py":
        "7562e43477ed5d9705ea357d4815aaea5cddd3bc0c1db8741ea5a25a04b52844",
    "configs/c09/c09_a0.json":
        "21ffdc3ff59913cd91f9d001ca66664f56b3d7f54bc62a607a63583820c626da",
    "scripts/slurm/c09_a0_cpu.sbatch":
        "3f9f181cb635afc1eb15647aaeeee2ae963290651ac64de3763acdbd66f139c7",
    "scripts/analysis/c09_guard/c09guard.py":
        "aed50842c232105f1b06182aa89512ee89dd050bdcaedec2706062c9d745f062",
    "scripts/analysis/c09_guard/sitecustomize.py":
        "b238789fd80076b0b890c4894fd8b69255792af51c80cd9fe2d6db6c53383850",
    "scripts/analysis/headspace_mint.py":
        "cefdf8dc2f4a9aefa042ef7bec9b1d06c9721ae5b4a70ec117e9929ff0916612",
    "scripts/analysis/mechfix_ops.py":
        "635c13124e79ba1a299bc13fc1175a03aa11e09924f5413ce51061793c83fc8d",
    "scripts/analysis/mechnov_pairverify.py":
        "77b0defd8eaa3688e58b6d5d17202bd55d16cf1f4a5aaafbe4b2b98598b7240d",
    "scripts/analysis/headspace_fidelity.py":
        "72fd8e0aab61b635b4421b87bdbccc8ef6c58bf28fe1ff64cab0671e08bf6598",
}

# ---------------------------------------------------------------------------
# THE FROZEN INPUT CLOSURE -- enumerated, never globbed over a data directory.
# Derived from the frozen record's input manifest and re-verified by reading the
# code: headspace_mint.load_split (train + dev_seen only), the vsw_ckpt fold parity
# assert, the arena's banked-arena / gt / train-cache reads, and
# headspace_fidelity.FLOOR's six banked trainlogs.
# ---------------------------------------------------------------------------
DATA_CLOSURE = [
    # operative feature caches -- TRAIN + DEV_SEEN ONLY.  No test_seen, no _shards.
    "data/CLIP_Embedding/HateMM/train_Qwen2.5-VL-7B-Instruct-LoRA-curric_HF.pt",
    "data/CLIP_Embedding/HateMM/dev_seen_Qwen2.5-VL-7B-Instruct-LoRA-curric_HF.pt",
    "data/CLIP_Embedding/MHC_zh/train_Qwen2.5-VL-7B-Instruct-LoRA_HF.pt",
    "data/CLIP_Embedding/MHC_zh/dev_seen_Qwen2.5-VL-7B-Instruct-LoRA_HF.pt",
    # ground-truth text, TRAIN split only (DATA_DEFECT flags + gt-order parity)
    "data/gt/HateMM/train.jsonl",
    "data/gt/MHC_zh/train.jsonl",
]
DATA_CLOSURE += [
    # banked frozen fold assignment -- headspace_mint.py:209-216 fold parity assert
    "scripts/analysis/vsw_ckpt/{}/f{}.npz".format(ds, f)
    for ds in ("hatemm", "zh") for f in range(5)
]
DATA_CLOSURE += [
    # banked fold-head arena -- GATE-FLOOR / GATE-PARITY-FOLD / raw parity anchors
    "scripts/analysis/headspace_arena_{}_s{}_OUT.json".format(ds, s)
    for ds in ("hatemm", "zh") for s in (0, 1, 2)
]
DATA_CLOSURE += [
    # banked encoder trainlogs -- GATE-DEVFID (reporting only).  Val_Retrieval-only
    # hard filter in headspace_fidelity.floor_dev_curve; no Test_Retrieval line is
    # ever parsed or stored.
    "slurm/logs/enc3s_HateMM_Qwen2.5-VL-7B-Instruct-LoRA-curric_HF_seed{}_13241.trainlog".format(s)
    for s in (0, 1, 2)
] + [
    "slurm/logs/enc3s_MHC_zh_Qwen2.5-VL-7B-Instruct-LoRA_HF_seed{}_13150.trainlog".format(s)
    for s in (0, 1, 2)
]

CODE_CLOSURE = [
    "scripts/analysis/c09_a0_arena.py",
    "scripts/analysis/headspace_mint.py",
    "scripts/analysis/mechfix_ops.py",
    "scripts/analysis/mechnov_pairverify.py",
    "scripts/analysis/headspace_fidelity.py",
    "scripts/analysis/c09_guard/c09guard.py",
    "scripts/analysis/c09_guard/sitecustomize.py",
    "configs/c09/c09_a0.json",
    # provenance anchor for the transliteration (PORT-CHECK-5); never executed here
    "scripts/slurm/c09_a0_cpu.sbatch",
]

# ---------------------------------------------------------------------------
# Upload guard.  The upstream media guard is imported and applied UNWEAKENED.
# ---------------------------------------------------------------------------
# NOTE (port review C-2): Modal 1.x re-imports THIS MODULE inside the container, and
# scripts/cloud/ is not part of the uploaded closure.  Everything that reads the local
# filesystem -- the upstream guard import, the manifest, the staging tree -- is therefore
# local-only.  The container receives the authoritative manifest as a function argument,
# so nothing is lost by not recomputing it there.
if modal.is_local():
    sys.path.insert(0, str(REPO_LOCAL / "scripts" / "cloud"))
    from modal_probe_runner import guard_reason  # noqa: E402
else:                                            # pragma: no cover - container side
    def guard_reason(path):
        raise RuntimeError("guard_reason is a LOCAL-ONLY upload guard; nothing is "
                           "uploaded from inside the container")

# The upstream allowlist is {.pt,.jsonl,.json,.csv,.npy,.txt} -- it was written for the
# feature-cache sync path.  This closure additionally carries three DERIVED, non-media
# artefact types.  They are added to the ALLOWLIST only; the media-extension blocklist
# and the forbidden-media-directory check are untouched and are evaluated FIRST inside
# guard_reason, so widening the allowlist cannot let a video or an audio file through.
#   .npz      -- banked StratifiedKFold hold-out index arrays (vsw_ckpt/*/f*.npz)
#   .trainlog -- banked encoder training text logs (GATE-DEVFID, reporting only)
#   .py       -- the code subset (the upstream runner ships code via the image layer,
#                which bypasses the data guard entirely; routing it THROUGH the guard
#                here is strictly more checking than upstream does, not less)
#   .sbatch   -- scripts/slurm/c09_a0_cpu.sbatch, carried as the PROVENANCE ANCHOR for
#                the transliteration (PORT-CHECK-5) and never executed in the container
_C09_EXTRA_EXTS = {".npz", ".trainlog", ".py", ".sbatch"}


def c09_assert_uploadable(local_path):
    """Fail-loud unless `local_path` is in the C09 closure's allowlist.

    Never bypasses a media rejection: guard_reason checks the media extension and the
    forbidden media directory FIRST and returns those reasons before it ever reaches
    the allowlist clause, so the only reason this function may tolerate is the
    allowlist one, and only for the three declared derived types.

    Adds a check upstream does not have: a path c09guard would judge TEST-LIKE is
    refused outright, so no test-split artefact can enter the container at all.

    The four tolerated extras are declared in `_C09_EXTRA_EXTS` above, each with its
    reason; there is no undeclared widening.
    """
    real = Path(local_path).resolve()
    for candidate in (Path(local_path), real):
        reason = guard_reason(candidate)
        if reason is None:
            continue
        tolerable = (reason.startswith("extension ")
                     and Path(candidate).suffix.lower() in _C09_EXTRA_EXTS)
        if not tolerable:
            raise RuntimeError(
                "[VIDEO-GUARD] REFUSING to upload {}: {} (resolved={})".format(
                    local_path, reason, real))
    # strictly-stronger C09 check: nothing test-like may be uploaded
    ap = str(real)
    for q in ap.split(os.sep):
        q = q.lower()
        if not q or "site-packages" in ap:
            continue
        if ("test_seen" in q or q.startswith("test.") or q.startswith("test_")
                or q == "test" or "_test." in q or "_test_" in q or q.endswith("_test")):
            raise RuntimeError(
                "[C09-TEST-GUARD] REFUSING to upload {}: path component {!r} is "
                "test-like; no test-split artefact may enter the container".format(
                    local_path, q))


def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for blk in iter(lambda: f.read(1 << 20), b""):
            h.update(blk)
    return h.hexdigest()


def build_manifest():
    """Guard + hash every closure member locally, BEFORE any byte is uploaded."""
    man = {}
    for rel in DATA_CLOSURE + CODE_CLOSURE:
        p = REPO_LOCAL / rel
        if not p.is_file():
            raise SystemExit("[closure] MISSING: {}".format(p))
        c09_assert_uploadable(p)
        man[rel] = {"sha256": _sha256(p), "bytes": p.stat().st_size}
    for rel, want in FROZEN_SHA256.items():
        got = man[rel]["sha256"]
        if got != want:
            raise SystemExit(
                "[closure] FROZEN SET CHANGED: {}\n  want {}\n  got  {}".format(
                    rel, want, got))
    return man


MANIFEST = build_manifest() if modal.is_local() else {}
MANIFEST_JSON = json.dumps(MANIFEST, indent=1, sort_keys=True)

# ---------------------------------------------------------------------------
# Staging tree.  Every uploaded byte is copied into one directory whose layout IS the
# container layout, so (a) the upload set is auditable by listing one tree, (b) nothing
# outside the guarded closure can be swept in by a directory glob, and (c) the image is
# one layer rather than 60-odd.  Rebuilt from scratch each invocation.
# ---------------------------------------------------------------------------
STAGE = Path(os.environ.get(
    "C09_STAGE_DIR",
    "/data/jehc223/home/tmp/claude-135258174/-data-jehc223-RGCL/"
    "3ab1f506-990f-485a-8326-331bed01a558/scratchpad/c09_stage"))

_SRC_SKIP_DIRS = {"__pycache__", "logging", "moka"}


def build_stage():
    import shutil
    if STAGE.exists():
        shutil.rmtree(STAGE)
    for rel in DATA_CLOSURE + CODE_CLOSURE:
        dst = STAGE / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(REPO_LOCAL / rel, dst)
    # run_rac's import chain.  __pycache__ is excluded so no stale bytecode can shadow
    # a source file; `logging` (an empty artefact tree) and `moka` (unused) are excluded
    # -- neither omission changes an import, because a directory with no __init__.py is
    # only a namespace PORTION and the real stdlib module later on sys.path wins.
    n_src = 0
    for p in sorted((REPO_LOCAL / "src").rglob("*")):
        if not p.is_file() or p.suffix == ".pyc":
            continue
        rel = p.relative_to(REPO_LOCAL)
        if _SRC_SKIP_DIRS & set(rel.parts):
            continue
        c09_assert_uploadable(p)
        dst = STAGE / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(p, dst)
        n_src += 1
    staged = sorted(str(p.relative_to(STAGE)) for p in STAGE.rglob("*") if p.is_file())
    assert len(staged) == len(DATA_CLOSURE) + len(CODE_CLOSURE) + n_src, \
        "staging tree size mismatch"
    return staged


STAGED = build_stage() if modal.is_local() else []

# ---------------------------------------------------------------------------
# The pinned image.  Dependency versions are the HateVideo env's, so the five members
# DET-3 / GATE-FLOOR's RUNTIME_DRIFT tracks (python numpy scipy sklearn torch) match
# the banked runtime block exactly.  CPU only: no `gpu=` anywhere in this file.
# ---------------------------------------------------------------------------
_IGNORE = ["**/__pycache__", "**/__pycache__/**", "*.pyc", "**/*.pyc",
           "logging", "logging/**", "moka", "moka/**"]

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch==2.6.0",
        "faiss-cpu==1.13.2",
        "scikit-learn==1.5.2",
        "numpy==1.26.4",
        "scipy==1.17.1",
        "pandas==2.3.3",
        "pillow==11.1.0",
        "tqdm==4.67.3",
        "easydict==1.13",
        "rank-bm25==0.2.2",
        "torchmetrics==1.9.0",
        "wandb==0.28.0",
        "threadpoolctl==3.6.0",
    )
    # DET-1 (PREGATE_DETERMINISM_CLAUSE): exported into the image env, so the four
    # thread variables are set BEFORE any interpreter in this container starts --
    # which is what the clause requires and what headspace_mint.det1_assert checks.
    .env({
        "OMP_NUM_THREADS": "8", "MKL_NUM_THREADS": "8",
        "OPENBLAS_NUM_THREADS": "8", "NUMEXPR_NUM_THREADS": "8",
        "CUDA_VISIBLE_DEVICES": "",
        "PYTHONUNBUFFERED": "1",
        "WANDB_MODE": "disabled", "WANDB_DISABLED": "true",
        "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
        # PYTHONPATH is DELIBERATELY NOT SET HERE (port review C-1, measured).  An
        # image-level PYTHONPATH replaces the one Modal's own container runtime uses
        # to reach its vendored client deps, and forces `site` to import the C09
        # sitecustomize into Modal's runner process before Modal itself is loaded:
        # every container then died with `ModuleNotFoundError: No module named
        # 'grpclib'` and crash-looped.  This costs the port NOTHING, because the
        # guard is needed in the JOB'S python processes, not in the driver -- exactly
        # as on SLURM, where the sbatch's own shell carries no guard either.  _env()
        # sets PYTHONPATH for every child, with the sbatch's ${PYTHONPATH:+:$PYTHONPATH}
        # prepend semantics, and PORT-CHECK-1 verifies the guard IS installed in a
        # child before any mint runs.
    })
)
if modal.is_local():
    # ONE layer, from the guarded staging tree, whose layout IS the container layout.
    # copy=True bakes it into the image, so the image digest covers the input closure
    # byte-for-byte and the pin in the port record is a pin on the data as well.
    # Local-only (port review C-2): STAGE does not exist inside the container, and the
    # container never needs to re-derive the layer it is already running.
    image = image.add_local_dir(str(STAGE), REPO, ignore=_IGNORE, copy=True)

app = modal.App(APP_NAME, image=image)
outvol = modal.Volume.from_name(OUT_VOLUME, create_if_missing=True)

# *** BLOCKER, MEASURED IN THIS REPO -- READ BEFORE INVOKING a0 (port review r2 C-1) ***
# Modal clamps the effective function timeout to ~3600 s SERVER-SIDE on this account's
# plan.  refine-logs/W2A_PROBE_RECORD.md:32-38 and W2A_CHUNK_LOG.md:1-6 verify it twice:
# MODAL_PROBE_TIMEOUT=43200 reached the child and computed 43200, yet BOTH single-
# container attempts were killed at ~62 min.  So a large value here proves nothing.
#
# C09 A0's MEASURED wall on the faster local substrate (job 13885, AMD EPYC 7742,
# 8 threads): ~28 min for the 36 mints + a ~20-30 min arena = ~50-58 min.  A Modal
# 8-vCPU container is unlikely to be faster.  That is AT OR BEYOND the ~62 min wall.
#
# The repo's existing workaround for this cap (modal_probe_runner._execute's soft-budget
# chunk loop) resumes in a NEW CONTAINER, which violates this port's invariant 2 (one
# container, one host) and therefore the same-table-same-hardware ruling.  It is NOT
# available here.  Until the cap is lifted or measured otherwise, a0 MUST NOT be invoked:
# a kill at ~62 min lands after the first mint, which §1 of the port record counts as a
# SPENT invocation.
TIMEOUT_S = int(os.environ.get("C09_MODAL_TIMEOUT", "28800"))  # requested; expect a clamp


# ---------------------------------------------------------------------------
# shared container-side helpers
# ---------------------------------------------------------------------------
def _hostinfo():
    info = {}
    try:
        with open("/proc/cpuinfo") as f:
            for line in f:
                if line.lower().startswith("model name"):
                    info["cpu_model"] = line.split(":", 1)[1].strip()
                    break
    except Exception as exc:                                   # noqa: BLE001
        info["cpu_model"] = "unreadable: {!r}".format(exc)
    info["n_cpu_online"] = os.cpu_count()
    try:
        import platform
        info["uname"] = platform.uname()._asdict()
    except Exception:                                          # noqa: BLE001
        pass
    return info


def _envpin():
    """Capture what the port record's section 7 must carry before a0 is authorised
    (port review r2 H-2): the container's resolved dependency set, its Python patch
    version, and the Modal task/image identifiers.  A post-hoc freeze is a RECORD of
    one build, not a constraint on the next -- the record says so."""
    pin = {"python": sys.version, "python_patch": ".".join(
        map(str, sys.version_info[:3]))}
    try:
        pin["pip_freeze"] = subprocess.run(
            [sys.executable, "-m", "pip", "freeze"], capture_output=True, text=True,
            timeout=120).stdout.splitlines()
    except Exception as exc:                                   # noqa: BLE001
        pin["pip_freeze"] = "unavailable: {!r}".format(exc)
    for k in ("MODAL_TASK_ID", "MODAL_IMAGE_ID", "MODAL_ENVIRONMENT", "MODAL_REGION"):
        pin[k] = os.environ.get(k)
    try:
        import threadpoolctl
        pin["threadpools"] = threadpoolctl.threadpool_info()
    except Exception as exc:                                   # noqa: BLE001
        pin["threadpools"] = "unavailable: {!r}".format(exc)
    return pin


def _verify_upload(manifest):
    """PORT-CHECK-3: post-upload sha256 == pre-upload sha256, every closure member."""
    bad = []
    for rel, rec in manifest.items():
        p = os.path.join(REPO, rel)
        if not os.path.isfile(p):
            bad.append({"path": rel, "problem": "MISSING IN CONTAINER"})
            continue
        h = hashlib.sha256()
        with open(p, "rb") as f:
            for blk in iter(lambda: f.read(1 << 20), b""):
                h.update(blk)
        got = h.hexdigest()
        if got != rec["sha256"]:
            bad.append({"path": rel, "pre_upload": rec["sha256"], "in_container": got})
    return bad


def _port_checks(manifest):
    """PORT-CHECK-2/3/5 in the driver process, before any mint runs."""
    report = {}
    bad = _verify_upload(manifest)
    report["PORT_CHECK_3_upload_sha256"] = {"mismatches": bad, "pass": not bad}
    if bad:
        raise SystemExit("[PORT-CHECK-3] upload integrity FAILED: {}".format(bad))

    sys.path.insert(0, os.path.join(REPO, "scripts/analysis/c09_guard"))
    import c09guard
    report["PORT_CHECK_2_guard_scope"] = {
        "c09guard_REPO": c09guard.REPO,
        "repo_root_exists": os.path.isdir(c09guard.REPO),
        "arena_present_under_repo": os.path.isfile(
            os.path.join(c09guard.REPO, "scripts/analysis/c09_a0_arena.py")),
        "predicate_live_on_a_probe_path": c09guard.is_test_like(
            os.path.join(c09guard.REPO, "data/CLIP_Embedding/HateMM/test_seen_x.pt")),
        "predicate_rejects_an_operative_path": not c09guard.is_test_like(
            os.path.join(c09guard.REPO,
                         "data/CLIP_Embedding/HateMM/"
                         "train_Qwen2.5-VL-7B-Instruct-LoRA-curric_HF.pt")),
    }
    pc2 = report["PORT_CHECK_2_guard_scope"]
    ok2 = (pc2["c09guard_REPO"] == REPO and pc2["repo_root_exists"]
           and pc2["arena_present_under_repo"]
           and pc2["predicate_live_on_a_probe_path"]
           and pc2["predicate_rejects_an_operative_path"])
    pc2["pass"] = bool(ok2)
    if not ok2:
        raise SystemExit(
            "[PORT-CHECK-2] the split guard is NOT correctly scoped in this container "
            "-- a guard mounted off {} silently passes every path. {}".format(REPO, pc2))

    # PORT-CHECK-5: hash the CONTAINER's own bytes directly against the freeze
    # record's table.  (Port review I-5: comparing the passed-in manifest against
    # FROZEN_SHA256 would only be transitively sound via PORT-CHECK-3; hashing the
    # container files here makes the check say what the record claims it says.)
    in_container = {}
    for rel in FROZEN_SHA256:
        h = hashlib.sha256()
        with open(os.path.join(REPO, rel), "rb") as f:
            for blk in iter(lambda: f.read(1 << 20), b""):
                h.update(blk)
        in_container[rel] = h.hexdigest()
    report["PORT_CHECK_5_frozen_sha256"] = {
        "checked": len(FROZEN_SHA256),
        "hashed_where": "the container's own files",
        "all_match": all(in_container[r] == w for r, w in FROZEN_SHA256.items()),
        "table": in_container}
    if not report["PORT_CHECK_5_frozen_sha256"]["all_match"]:
        raise SystemExit("[PORT-CHECK-5] frozen sha256 mismatch in container: {}".format(
            {r: (in_container[r], w) for r, w in FROZEN_SHA256.items()
             if in_container[r] != w}))
    return report


def _env(job_token=None):
    """The sbatch's exported environment, carried verbatim into every subprocess.

    PYTHONPATH reproduces the sbatch's `${PYTHONPATH:+:$PYTHONPATH}` prepend, so the
    guard directory wins without discarding anything already on the path.

    `job_token` (port review I-1): c09guard names its ledger files
    `led_{SLURM_JOB_ID or "nojob"}_{pid}_{t0}.json` and `aggregate()` sums only the
    files whose prefix matches the CURRENT SLURM_JOB_ID, routing the rest to `stale`.
    With no SLURM in the container both writer and reader would see the literal
    "nojob", collapsing that partition so a previous attempt's processes would count
    toward GATE-LEDGER's `n_processes_reporting >= 1` conjunct -- the very conjunct
    the design added so that a ledger reading zero because nothing reported cannot
    pass.  Setting a per-invocation token restores the partition exactly.
    """
    e = os.environ.copy()
    guard_dir = "{}/scripts/analysis/c09_guard".format(REPO)
    prev = e.get("PYTHONPATH", "")
    e.update({
        "OMP_NUM_THREADS": "8", "MKL_NUM_THREADS": "8",
        "OPENBLAS_NUM_THREADS": "8", "NUMEXPR_NUM_THREADS": "8",
        "CUDA_VISIBLE_DEVICES": "", "PYTHONUNBUFFERED": "1",
        "WANDB_MODE": "disabled", "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
        "PYTHONPATH": (guard_dir + ":" + prev) if prev else guard_dir,
    })
    if job_token:
        e["SLURM_JOB_ID"] = str(job_token)
    # Port review r2 I-5: every guard in headspace_mint.py is an `assert` -- DET-1, the
    # fold-parity check against the banked vsw_ckpt, the frozen-pairverify sha, and the
    # torch.load test guard -- and `-O` / PYTHONOPTIMIZE strips them all.  The arena
    # refuses to run under -O (c09_a0_arena.py:47), but only AFTER 36 unprotected mints.
    # On SLURM the variable is unset; a container inherits whatever the image carries,
    # so it is scrubbed explicitly rather than assumed absent.
    e.pop("PYTHONOPTIMIZE", None)
    return e


def _run(cmd, env, allow_fail=False, label=""):
    print("### {} :: {}".format(label, " ".join(cmd)), flush=True)
    rc = subprocess.run(cmd, cwd=REPO, env=env).returncode
    if rc != 0 and not allow_fail:
        raise SystemExit("[c09_a0] FAIL rc={} on {}".format(rc, label))
    return rc


# ---------------------------------------------------------------------------
# THE RUN -- one container, one host, the sbatch's five steps in order
# ---------------------------------------------------------------------------
@app.function(cpu=8, memory=32768, timeout=TIMEOUT_S, retries=0,
              volumes={"/c09out": outvol})
def a0(manifest: dict, manifest_json: str) -> dict:
    # retries=0 is PINNED, not left to the platform default (port review r2 H-1.3): a
    # retry would silently re-run the science on a FRESH INDEPENDENT HOST DRAW, which is
    # exactly what the one-invocation preregistration exists to forbid.
    import shutil
    import time as _time

    t_start = _time.time()
    job_token = "{}-{}".format(RUN_ID, int(t_start))
    env = _env(job_token)
    base = "{}/artifacts/c09_topo/v1/a0/{}".format(REPO, RUN_ID)
    sc = base + "/scratch"
    outdir = base
    ledger = base + "/ledger"
    for d in (sc, outdir, ledger):
        os.makedirs(d, exist_ok=True)
    env["C09_LEDGER_DIR"] = ledger
    cfg = "{}/configs/c09/c09_a0.json".format(REPO)

    host = _hostinfo()
    print("[c09_a0] MODAL PORT run_id={} host={}".format(RUN_ID, host), flush=True)
    port = _port_checks(manifest)
    port["host"] = host
    port["environment_pin"] = _envpin()
    print("[c09_a0] port checks PASSED", flush=True)

    # SPENT-INVOCATION SENTINEL (port review r2 H-1.2).  On SLURM, sacct is a permanent
    # third-party record of every submission and the approval gate is a second party;
    # `modal run` has neither, so the one-invocation preregistration needs a mechanism
    # rather than prose.  Written BEFORE the first mint -- which is exactly where §1 of
    # the port record draws the "invocation is spent" boundary -- and committed, so it
    # survives container death.
    try:
        outvol.reload()
    except Exception as exc:                                   # noqa: BLE001
        print("[c09_a0] volume reload skipped: {!r}".format(exc), flush=True)
    prior = sorted(f for f in os.listdir("/c09out")
                   if f.startswith("INVOCATION_") and f.endswith(".json")) \
        if os.path.isdir("/c09out") else []
    if prior:
        raise SystemExit(
            "[ONE-INVOCATION] refusing: a prior cloud A0 invocation is already recorded "
            "({}). The preregistration allows exactly ONE; a run that reached the "
            "science is SPENT whatever its outcome.".format(prior))
    with open("/c09out/INVOCATION_{}.json".format(job_token), "w") as f:
        json.dump({"run_id": RUN_ID, "job_token": job_token, "t_start": t_start,
                   "host": host, "environment_pin": port["environment_pin"],
                   "note": "written before the first mint; presence of this file means "
                           "the single pre-registered cloud invocation is SPENT"}, f,
                  indent=1)
    outvol.commit()

    # ---- sbatch step 1: startup guard + zero-GPU assert, in a python process
    # The sbatch's step-1 heredoc, plus (port review r2 I-4) an import probe over the
    # WHOLE mint chain.  The sbatch imports only 5 packages; a missing one -- say
    # threadpoolctl, which headspace_mint reaches only in runtime_block() -- would
    # otherwise surface after ~52 s of training inside mint #1, i.e. on the "spent"
    # side of the one-invocation boundary.  HALT-direction only; no science is touched.
    _run([sys.executable, "-c",
          "import builtins, torch, numpy, faiss, sys, sklearn, scipy\n"
          "assert builtins.open.__name__ == '_guarded_open', "
          "'C09 startup guard NOT installed'\n"
          "assert __debug__, 'PYTHONOPTIMIZE strips the assert-based guards'\n"
          "print('[c09_a0] python', sys.version.split()[0], 'numpy', numpy.__version__,"
          " 'scipy', scipy.__version__, 'sklearn', sklearn.__version__,"
          " 'torch', torch.__version__, 'faiss', faiss.__version__)\n"
          "assert torch.cuda.device_count() == 0, "
          "'ZERO-GPU VIOLATION: a device is visible'\n"
          "import threadpoolctl, wandb, pandas, PIL, easydict, rank_bm25, torchmetrics,"
          " tqdm\n"
          "sys.path.insert(0, '{r}/src'); sys.path.insert(0, '{r}/scripts/analysis')\n"
          "import run_rac, mechfix_ops, mechnov_pairverify\n"
          "print('[c09_a0] full mint import chain OK')\n".format(r=REPO)],
         env, label="PORT-CHECK-1/4 startup guard + zero-GPU + import probe")

    # ---- sbatch step 2: frozen-module sha256 BEFORE the mints (already done in
    #      _port_checks via PORT-CHECK-5; the arena re-asserts at import as well)

    # ---- sbatch step 3: 36 CPU head mints
    n_minted = 0
    for ds in ("hatemm", "zh"):
        for seed in (0, 1, 2):
            for fold in (0, 1, 2, 3, 4, -1):
                tag = "full" if fold < 0 else str(fold)
                out = "{}/mint_{}_s{}_f{}.npz".format(sc, ds, seed, tag)
                if os.path.isfile(out):
                    print("[c09_a0] have {}".format(out), flush=True)
                    continue
                _run([sys.executable,
                      "{}/scripts/analysis/headspace_mint.py".format(REPO),
                      "--dataset", ds, "--seed", str(seed), "--fold", str(fold),
                      "--out", out, "--scratch", sc],
                     env, label="mint ds={} seed={} fold={}".format(ds, seed, fold))
                n_minted += 1
    print("[c09_a0] all 36 mints present ({} minted this run)".format(n_minted),
          flush=True)

    # Persist every artifact to the output volume.  The 36 mint .npz and the ledger
    # files go too (port review I-2): on SLURM they survive on the cluster filesystem
    # indefinitely, and if this run takes the tie-break its instrument must remain
    # auditable rather than dying with the container.
    # Keyed on job_token, NOT on the constant RUN_ID (port review r2 H-1.1): a constant
    # destination would let a second invocation overwrite the first's decision, manifest
    # and mints -- destroying the one artifact that proves the one-invocation rule was
    # broken, by breaking it.
    # Called INCREMENTALLY and from a `finally` (port review r2 I-3): the only commit
    # used to sit on the success path, so any crash, OOM, preemption or timeout-kill
    # lost all 36 mints and the whole ledger -- precisely the state §1 calls SPENT.
    dest = "/c09out/{}".format(job_token)
    state = {"stage": "mints", "devfid_rc": {}, "verdict": None}

    def _snapshot(stage):
        state["stage"] = stage
        try:
            os.makedirs(dest, exist_ok=True)
            for fn in sorted(os.listdir(outdir)):
                p = os.path.join(outdir, fn)
                if os.path.isfile(p):
                    shutil.copy2(p, os.path.join(dest, fn))
            for sub in ("scratch", "ledger"):
                src_dir = os.path.join(base, sub)
                if not os.path.isdir(src_dir):
                    continue
                dst_dir = os.path.join(dest, sub)
                os.makedirs(dst_dir, exist_ok=True)
                for fn in sorted(os.listdir(src_dir)):
                    p = os.path.join(src_dir, fn)
                    if os.path.isfile(p):
                        shutil.copy2(p, os.path.join(dst_dir, fn))
            with open(os.path.join(dest, "C09_PORT_MANIFEST.json"), "w") as f:
                json.dump({"run_id": RUN_ID, "stage_reached": stage,
                           "port_checks": port, "host": host,
                           "elapsed_s": round(_time.time() - t_start, 1),
                           "devfid_returncodes": state["devfid_rc"],
                           "n_minted_this_run": n_minted,
                           "ledger_job_token": job_token,
                           "verdict": state["verdict"],
                           "input_closure_manifest": json.loads(manifest_json)},
                          f, indent=1)
            outvol.commit()
            print("[c09_a0] snapshot committed at stage={}".format(stage), flush=True)
        except Exception as exc:                               # noqa: BLE001
            print("[c09_a0] snapshot at stage={} failed: {!r}".format(stage, exc),
                  flush=True)

    try:
        _snapshot("mints_complete")

        # ---- sbatch step 4: GATE-DEVFID, REPORTING ONLY -- must not abort the job
        for ds in ("hatemm", "zh"):
            state["devfid_rc"][ds] = _run(
                [sys.executable,
                 "{}/scripts/analysis/headspace_fidelity.py".format(REPO),
                 "--dataset", ds, "--mintdir", sc,
                 "--out", "{}/C09_FIDELITY_{}.json".format(outdir, ds)],
                env, allow_fail=True, label="GATE-DEVFID {}".format(ds))
        print("[c09_a0] GATE-DEVFID written (reporting instrument, does NOT gate)",
              flush=True)
        _snapshot("devfid_complete")

        # ---- sbatch step 5: the arena
        _run([sys.executable, "{}/scripts/analysis/c09_a0_arena.py".format(REPO),
              "--config", cfg, "--mintdir", sc, "--outdir", outdir, "--threads", "8"],
             env, label="arena")

        elapsed = _time.time() - t_start
        dec = json.load(open(os.path.join(outdir, "C09_A0_DECISION.json")))
        state["verdict"] = dec["DECISION"]["verdict"]
        print("[c09_a0] VERDICT: {}  ({:.1f} s wall)".format(state["verdict"], elapsed),
              flush=True)
    finally:
        _snapshot(state["stage"] + "_final")

    return {"run_id": RUN_ID, "verdict": state["verdict"],
            "elapsed_s": round(elapsed, 1),
            "host": host, "port_checks": port, "job_token": job_token,
            "decision": dec["DECISION"], "gate_ledger": dec.get("GATE_LEDGER"),
            "gate_devfid": dec.get("GATE_DEVFID")}


# ---------------------------------------------------------------------------
# NUMERICS PREFLIGHT -- instrument portability only.  Computes NO A0 quantity,
# renders NO verdict, writes to a SEPARATE namespace, and its mints are DISCARDED.
# It tests exactly the two class-(b) gates: does a head minted in THIS image
# reproduce the banked local fold-head arena at 4 dp?
# ---------------------------------------------------------------------------
@app.function(cpu=8, memory=32768, timeout=3600)
def preflight(manifest: dict, dataset: str = "hatemm", seed: int = 0) -> dict:
    import time as _time
    t0 = _time.time()
    env = _env("PREFLIGHT-{}".format(int(t0)))
    sc = "{}/artifacts/c09_preflight/scratch".format(REPO)
    ledger = "{}/artifacts/c09_preflight/ledger".format(REPO)
    os.makedirs(sc, exist_ok=True)
    os.makedirs(ledger, exist_ok=True)
    env["C09_LEDGER_DIR"] = ledger

    host = _hostinfo()
    print("[preflight] host={}".format(host), flush=True)
    _port_checks(manifest)

    for fold in range(5):
        _run([sys.executable, "{}/scripts/analysis/headspace_mint.py".format(REPO),
              "--dataset", dataset, "--seed", str(seed), "--fold", str(fold),
              "--out", "{}/mint_{}_s{}_f{}.npz".format(sc, dataset, seed, fold),
              "--scratch", sc],
             env, label="preflight mint {} s{} f{}".format(dataset, seed, fold))
    mint_s = _time.time() - t0

    # Compare using the FROZEN arena's own code -- no reimplementation, so the
    # preflight cannot diverge from what GATE-FLOOR / GATE-PARITY-FOLD will compute.
    os.environ.update({k: env[k] for k in
                       ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                        "NUMEXPR_NUM_THREADS", "PYTHONPATH")})
    sys.path.insert(0, os.path.join(REPO, "scripts/analysis/c09_guard"))
    sys.path.insert(0, os.path.join(REPO, "scripts/analysis"))
    import numpy as np
    import c09_a0_arena as A
    from sklearn.model_selection import StratifiedKFold

    cfg = json.load(open("{}/configs/c09/c09_a0.json".format(REPO)))
    z0 = np.load("{}/mint_{}_s{}_f0.npz".format(sc, dataset, seed), allow_pickle=True)
    lab = z0["lab"].astype(int)
    n = len(lab)
    splits = list(StratifiedKFold(n_splits=cfg["k_folds"], shuffle=True,
                                  random_state=cfg["fold_seed"]).split(
                                      np.zeros((n, 1)), lab))
    X, NRM = {seed: []}, {seed: []}
    for f in range(cfg["k_folds"]):
        z = np.load("{}/mint_{}_s{}_f{}.npz".format(sc, dataset, seed, f),
                    allow_pickle=True)
        K = z["K_train"]
        NRM[seed].append(np.linalg.norm(np.asarray(K, dtype="float64"), axis=1))
        X[seed].append(A.P.l2n(K))
    banklab = A.BankLabels(lab, splits)
    cells = A.build_features(banklab, X, NRM, splits, [seed], cfg["topk"],
                             cfg["fixk_grid"], None)
    c = cells[seed]
    fold_of = c["fold"]
    m = c["pred"] >= 0
    got_acc = float(A.acc(lab[m], c["pred"][m]))
    got_mf1 = float(A.mf1(lab[m], c["pred"][m]))
    got_fold = [round(A.acc(lab[m & (fold_of == f)], c["pred"][m & (fold_of == f)]), 4)
                for f in range(cfg["k_folds"])]

    bk = json.load(open("{}/scripts/analysis/headspace_arena_{}_s{}_OUT.json".format(
        REPO, dataset, seed)))["result"]
    res = {
        "dataset": dataset, "seed": seed, "host": host,
        "runtime": A.runtime_block(),
        "GATE_FLOOR_probe": {
            "banked_acc": bk["acc_deployed"], "got_acc": round(got_acc, 4),
            "banked_mF1": bk["mF1_deployed"], "got_mF1": round(got_mf1, 4),
            "acc_ok": round(got_acc, 4) == bk["acc_deployed"],
            "mF1_ok": round(got_mf1, 4) == bk["mF1_deployed"]},
        "GATE_PARITY_FOLD_probe": {
            "banked": bk["fold_acc_deployed"], "got": got_fold,
            "ok": [round(x, 4) == round(y, 4)
                   for x, y in zip(got_fold, bk["fold_acc_deployed"])]},
        "mint_wall_s": round(mint_s, 1),
        "total_wall_s": round(_time.time() - t0, 1),
        "SCOPE": "INSTRUMENT PORTABILITY ONLY. No A0 quantity is computed here, no "
                 "verdict is rendered, no threshold is read or written, and these "
                 "mints are DISCARDED -- the real run mints all 36 fresh in one "
                 "container.",
        "WHAT_A_PROCEED_DOES_NOT_MEAN": "Modal exposes no CPU-SKU selector, so this is "
                 "ONE host draw and the real run gets an INDEPENDENT one. PROCEED is "
                 "COST-AVOIDANCE ONLY -- it is not a portability guarantee for the "
                 "run, and it licenses no relaxation of GATE-FLOOR or "
                 "GATE-PARITY-FOLD, which the run carries verbatim and which fail "
                 "closed to a HALT on any 4-dp mismatch.",
    }
    res["PROCEED"] = bool(res["GATE_FLOOR_probe"]["acc_ok"]
                          and res["GATE_FLOOR_probe"]["mF1_ok"]
                          and all(res["GATE_PARITY_FOLD_probe"]["ok"]))
    print(json.dumps({k: v for k, v in res.items() if k != "runtime"}, indent=1),
          flush=True)
    return res


# ---------------------------------------------------------------------------
# local entrypoints
# ---------------------------------------------------------------------------
@app.local_entrypoint()
def run_preflight(dataset: str = "hatemm", seed: int = 0):
    print("[preflight] closure: {} files, {:.1f} MB".format(
        len(MANIFEST), sum(v["bytes"] for v in MANIFEST.values()) / 1e6))
    r = preflight.remote(MANIFEST, dataset, seed)
    Path("/data/jehc223/RGCL/artifacts/c09_port/preflight_{}_s{}.json".format(
        dataset, seed)).parent.mkdir(parents=True, exist_ok=True)
    with open("/data/jehc223/RGCL/artifacts/c09_port/preflight_{}_s{}.json".format(
            dataset, seed), "w") as f:
        json.dump(r, f, indent=1)
    print("PROCEED =", r["PROCEED"])


@app.local_entrypoint()
def run_a0():
    print("[a0] closure: {} files, {:.1f} MB; timeout {}s".format(
        len(MANIFEST), sum(v["bytes"] for v in MANIFEST.values()) / 1e6, TIMEOUT_S))
    r = a0.remote(MANIFEST, MANIFEST_JSON)
    d = Path("/data/jehc223/RGCL/artifacts/c09_port")
    d.mkdir(parents=True, exist_ok=True)
    with open(d / "modal_a0_return.json", "w") as f:
        json.dump(r, f, indent=1, default=str)
    print("VERDICT =", r["verdict"], " wall =", r["elapsed_s"], "s")
