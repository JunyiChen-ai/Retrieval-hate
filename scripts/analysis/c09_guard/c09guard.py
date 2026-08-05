"""C09 A0 -- job-wide split guard and cross-process GATE-LEDGER.

Installed at interpreter startup in EVERY python process of the C09 A0 job (see
the sibling sitecustomize.py, put on PYTHONPATH by scripts/slurm/c09_a0_cpu.sbatch),
so the perimeter covers the 36 mints and the 2 fidelity runs as well as the arena.
The frozen modules headspace_mint.py / headspace_fidelity.py are NOT edited: the
guard reaches them through builtins.open, which torch.load, np.load and json.load
all bottom out in.

Predicate (deliberately repo-scoped -- a bare "/test" substring would fire on
site-packages test directories and break `import numpy`):

    a path is TEST-LIKE iff it is under /data/jehc223/RGCL, is not inside
    site-packages, and ANY of its path components (directories included, not just
    the basename) contains "test_seen", begins with "test." or "test_", is
    exactly "test", or carries "_test" as a token ("_test.", "_test_", or a
    trailing "_test").

The component-wise form matters: 754 real test artifacts live under directories
like data/CLIP_Embedding/HateMM/grounded_qwen7b_8f/_shards/test_seen/<id>.pt, where
"test_seen" is a DIRECTORY, and a basename-only predicate let every one of them
through (that was round-2 finding H-1).  The "_test" token then adds the remaining
test-split artifacts whose split sits in the middle of the name -- the Role-3
arbitration outputs arb_*_test_*.jsonl and data/audio/*/clap_*_test.pt among them.
The coverage claim is NOT a static assertion: verify_predicate() re-derives it
against the live tree, and the freeze record quotes the number it returned.

Each process appends its own counts to $C09_LEDGER_DIR at exit, and the arena
aggregates them into GATE-LEDGER, so the ledger reports MEASURED opens rather than
literals.
"""
import atexit
import builtins
import json
import os
import sys

REPO = "/data/jehc223/RGCL"

LEDGER = {"test_path_opens": 0,
          "test_label_materialisations": 0,
          "dev_path_opens": 0,
          "dev_label_materialisations_outside_decisions": 0,
          "dev_or_test_labels_into_decision_quantities": 0,
          "banked_trainlog_opens": 0}

TEST_PATHS_SEEN = []
DEV_PATHS_SEEN = []
_INSTALLED = False
_ORIG_OPEN = builtins.open
_T0 = __import__("time").time()


def is_test_like(path):
    try:
        ap = os.path.abspath(str(path))
    except Exception:
        return False
    if not ap.startswith(REPO + os.sep):
        return False
    if "site-packages" in ap or "/.git/" in ap:
        return False
    for q in ap.split(os.sep):
        q = q.lower()
        if not q:
            continue
        if ("test_seen" in q or q.startswith("test.") or q.startswith("test_")
                or q == "test" or "_test." in q or "_test_" in q
                or q.endswith("_test")):
            return True
    return False


def is_banked_trainlog(path):
    """headspace_fidelity.py reads the six banked encoder trainlogs; GATE-LEDGER's
    declared dev-side expectation counts them separately from the 36 mint loads."""
    try:
        ap = os.path.abspath(str(path))
    except Exception:
        return False
    return ap.startswith(REPO + os.sep) and ap.endswith(".trainlog")


def is_dev_like(path):
    try:
        ap = os.path.abspath(str(path))
    except Exception:
        return False
    if not ap.startswith(REPO + os.sep):
        return False
    base = os.path.basename(ap).lower()
    return "dev_seen" in base or base.startswith("dev.") or base.startswith("dev_")


def _guarded_open(file, *a, **kw):
    if is_test_like(file):
        LEDGER["test_path_opens"] += 1
        TEST_PATHS_SEEN.append(str(file))
        _flush()
        raise AssertionError("C09 TEST-SPLIT GUARD: refusing to open {}".format(file))
    if is_dev_like(file):
        LEDGER["dev_path_opens"] += 1
        if len(DEV_PATHS_SEEN) < 200:
            DEV_PATHS_SEEN.append(str(file))
    elif is_banked_trainlog(file):
        LEDGER["banked_trainlog_opens"] += 1
    return _ORIG_OPEN(file, *a, **kw)


def _ledger_path():
    d = os.environ.get("C09_LEDGER_DIR", "")
    if not d:
        return None
    try:
        os.makedirs(d, exist_ok=True)
    except Exception:
        return None
    job = os.environ.get("SLURM_JOB_ID", "nojob")
    return os.path.join(d, "led_{}_{}_{}.json".format(
        job, os.getpid(), int(_T0 * 1000) % 10 ** 9))


def _flush():
    p = _ledger_path()
    if p is None:
        return
    try:
        with _ORIG_OPEN(p, "w") as fh:
            json.dump({"pid": os.getpid(), "t0": _T0, "argv": sys.argv,
                       "counts": dict(LEDGER),
                       "dev_paths": DEV_PATHS_SEEN[:200],
                       "test_paths": TEST_PATHS_SEEN}, fh, indent=1)
    except Exception:
        pass


def install():
    """Idempotent: safe whether reached via sitecustomize or an explicit import."""
    global _INSTALLED
    if _INSTALLED:
        return LEDGER
    builtins.open = _guarded_open
    atexit.register(_flush)
    _INSTALLED = True
    return LEDGER


def aggregate(ledger_dir):
    """Sum every process's ledger file (the arena's own counts are added by the
    caller, since this process has not run its atexit hook yet)."""
    tot = {k: 0 for k in LEDGER}
    procs, stale = [], []
    if ledger_dir and os.path.isdir(ledger_dir):
        for fn in sorted(os.listdir(ledger_dir)):
            if not fn.startswith("led_") or not fn.endswith(".json"):
                continue
            job = os.environ.get("SLURM_JOB_ID", "nojob")
            if not fn.startswith("led_{}_".format(job)):
                # a previous attempt: reported WITH its counts (so a resume can show
                # whether the aborted attempt tripped the guard), never summed in
                try:
                    with _ORIG_OPEN(os.path.join(ledger_dir, fn)) as fh:
                        sd = json.load(fh)
                    stale.append({"file": fn, "counts": sd.get("counts", {}),
                                  "test_paths": sd.get("test_paths", [])})
                except Exception:
                    stale.append({"file": fn, "counts": "UNREADABLE"})
                continue
            try:
                with _ORIG_OPEN(os.path.join(ledger_dir, fn)) as fh:
                    d = json.load(fh)
            except Exception:
                continue
            if d.get("pid") == os.getpid() and d.get("t0") == _T0:
                continue
            for k in tot:
                tot[k] += int(d.get("counts", {}).get(k, 0))
            procs.append({"pid": d.get("pid"),
                          "argv": " ".join(map(str, d.get("argv", [])))[:200],
                          "counts": d.get("counts", {})})
    return tot, procs, stale


def verify_predicate(root=None):
    """Re-derive the docstring's coverage claim on the live tree (read-only)."""
    root = root or REPO
    hit, miss_testish = [], []
    for dp, dn, fn in os.walk(root):
        if "site-packages" in dp or "/.git" in dp:
            dn[:] = [d for d in dn if d != ".git"]
            continue
        for f in fn:
            ap = os.path.join(dp, f)
            if is_test_like(ap):
                hit.append(ap)
            elif "test" in ap.lower():
                miss_testish.append(ap)
    return {"n_matched": len(hit), "n_unmatched_containing_test": len(miss_testish),
            "sample_matched": hit[:5], "sample_unmatched": miss_testish[:10]}
