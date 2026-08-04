#!/usr/bin/env python
"""c06_falsifier_mint.py -- THE ONE SHARED MINT DRIVER for the C06 $0 CPU falsifier.

Frozen design: refine-logs/C06_FALSIFIER_PREREG_DRAFT_V15E2.md (GO at round 15,
+ ERRATUM 1 + CODE-R1 §8 correction + ERRATUM 2).
Implements §3.3 ("Two head lineages, one driver") and §13.1 items 1-4, 22.

WHAT THIS IS
    ONE driver serves BOTH head lineages.  It imports headspace_mint with its sha256
    asserted and reuses its dataset table, deployed CLI, fold assignment, fold-parity
    assertion, dummy-dataloader construction, monkeypatches, seeding and DET-1 contract
    UNCHANGED -- headspace_mint.main() is CALLED, never re-implemented (§13.1 items 1, 4).

    --train-cache is the ONLY lineage-varying argument (§13.1 item 2).  It redirects the
    TRAIN split load and nothing else: model_name, the dev load and the dataset table come
    from the frozen table inside headspace_mint, which this driver does not touch.

      Head-N : --train-cache absent  -> the native deployed cache; anchors GATE-FLOOR
      Head-R : --train-cache <path>  -> train_<model>-ro_L24.pt; in-domain

    There is NO branch conditional on the cache filename or suffix (§13.1 item 3).  The
    redirect is a single unconditional substitution of the path handed to the frozen
    loader when split == "train".

WHAT IT ADDS, AND WHY IT MUST
    §8 Phase 1b prices THREE key forwards per Head-N fold mint {native, ro_std, ro_ow} and
    TWO per Head-R fold mint {ro_std, ro_ow} -- Head-R's ro_std forward IS its K_train,
    because its training cache is the ro_L24 cache.  §13.1 item 22 requires all key
    forwards to happen INSIDE the mint process and each mint to write its key matrices
    into its own .npz.

    The frozen headspace_mint.np.savez at :321-325 writes K_train / K_dev / lab / lab_dev /
    fold_of / fit_idx / meta and nothing else, and headspace_mint.main() neither returns
    the trained model nor exposes a hook for extra arrays.  This driver therefore:
      (1) captures the trained model by pre-patching run_rac.model_pass BEFORE calling
          main() -- main() then wraps THIS driver's spy as its own _ORIG_MODEL_PASS, so
          both capture and nothing frozen changes;
      (2) calls the frozen main() with --out pointing at a per-mint staging path;
      (3) forwards the two ro caches through the captured head, in the same process;
      (4) writes ONE final .npz carrying the frozen arrays plus h_std / h_ow.
    See IMPLEMENTATION NOTE (a) at the bottom of this docstring.

TEST CONTACT: NONE.  Layer 1 is headspace_mint's torch.load guard (:106-116), inherited
    unchanged.  Layer 2 is this driver's `split == "train"` assertion on every ro-cache
    load (§12).  Layer 3 is the frozen c09_guard sitecustomize, which the sbatch puts on
    PYTHONPATH (§13.1 item 28).

DETERMINISM: DET-1 asserted by the frozen det1_assert inside headspace_mint.main().
COST: CPU only, <= 8 threads.  Zero GPU, zero SLURM inside this process.

IMPLEMENTATION NOTE (a), for the separate code/resource review lineage:
    §13.1 item 22 says each mint "writes all of its key matrices into its own .npz
    (headspace_mint.py:321-325's np.savez pattern)".  The frozen savez does not write
    h_std/h_ow and cannot be given them without editing a frozen module.  This driver
    keeps the frozen call untouched and performs a SECOND savez of its own, producing one
    final .npz per mint that carries every array the arena reads.  One process, one final
    file, the pattern preserved; the write is the driver's, not headspace_mint's.  This is
    recorded as an implementation decision, not a design change.
"""
import argparse
import hashlib
import json
import os
import shutil
import sys
import time

import numpy as np

_T_START = time.time()

REPO = "/data/jehc223/RGCL"
sys.path.insert(0, os.path.join(REPO, "src"))
sys.path.insert(0, os.path.join(REPO, "scripts/analysis"))

# --- frozen imports, sha256 asserted before any behaviour depends on them (§11) --------
FROZEN_SHA = {
    "scripts/analysis/headspace_mint.py":
        "cefdf8dc2f4a9aefa042ef7bec9b1d06c9721ae5b4a70ec117e9929ff0916612",
    "scripts/analysis/mechnov_pairverify.py":
        "77b0defd8eaa3688e58b6d5d17202bd55d16cf1f4a5aaafbe4b2b98598b7240d",
}

RO_SUFFIX = {"std": "ro_L24", "ow": "ro_ow_L24"}


def sha256_of(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for blk in iter(lambda: f.read(1 << 20), b""):
            h.update(blk)
    return h.hexdigest()


def assert_frozen():
    for rel, want in FROZEN_SHA.items():
        got = sha256_of(os.path.join(REPO, rel))
        assert got == want, "FROZEN MODULE CHANGED: {} {}".format(rel, got)


assert_frozen()

import headspace_mint as HM              # noqa: E402  frozen, sha asserted above
import mechnov_pairverify as P           # noqa: E402  frozen, sha asserted above
import torch                             # noqa: E402

# CODE-R1 C-1: the FROZEN loader, bound at import BEFORE any override can be installed.
# Every ro-cache load in this driver goes through this object, never through the module
# attribute, so a train-split override cannot reach them.
_FROZEN_LOAD_SPLIT = HM.load_split


# ERRATUM 2 §7: configs/c06/c06_falsifier.json:"projected_seconds" is the SINGLE
# SOURCE; the sbatch exports it.  This literal is the hand-run fallback only.
PROJECTED_SECONDS = float(os.environ.get("C06_PROJECTED_SECONDS", 3674.0))


def heartbeat(progress_path, phase, done=None, total=None, extra=""):
    """CODE-R1 H-3: §9 requires every python process THIS LINEAGE AUTHORS to append
    through a handle opened `buffering=1`; the six sha-frozen headspace_fidelity.py
    processes do not, and the bash driver writes their line (sbatch:128-129).  72 of 74
    processes previously wrote nothing -- the two that did were the arena and the
    --gate-sha-only driver leg -- leaving the mint phase, 68.3 % of §8's budget, dark for
    its whole span.  One handle, opened and closed per call so no descriptor is held
    across a 40 s train, line-buffered, append-only."""
    if not progress_path:
        return
    from datetime import datetime, timezone
    stamp = datetime.now(timezone.utc).isoformat(timespec="seconds")
    units = "{}/{}".format(done, total) if total is not None else "-"
    elapsed = time.time() - _T_START
    line = "{} | {} | {} | {:.1f}s | {:.3f}x{}".format(
        stamp, phase, units, elapsed, elapsed / PROJECTED_SECONDS,
        (" | " + extra) if extra else "")
    try:
        os.makedirs(os.path.dirname(progress_path), exist_ok=True)
        with open(progress_path, "a", buffering=1) as fh:
            fh.write(line + "\n")
    except Exception:
        pass
    print(line, flush=True)


def assert_guard_active():
    """CODE-R1 H-2 / §13.1 item 28: layer 3 must be ACTIVE, not merely importable.
    c09_guard's sitecustomize swallows every failure by design, so a silent guard failure
    is otherwise indistinguishable from a clean run."""
    try:
        import c09guard
    except Exception as exc:
        raise AssertionError("TEST GUARD layer 3 not importable: {}".format(exc))
    assert getattr(c09guard, "_INSTALLED", False), \
        "TEST GUARD layer 3 imported but install() did not take effect"
    return c09guard


def ro_cache_path(dataset, which):
    """The ro cache path for `dataset`, built from the FROZEN dataset table.

    §3.1: the four L24 files are byte-identical to the ones C01 measured.  L28 is not
    used and is not constructible from this function.
    """
    cfg = P.DATASETS[dataset]
    return os.path.join(cfg["cache_dir"], "train_{}-{}.pt".format(
        cfg["model"], RO_SUFFIX[which]))


def load_ro_split(dataset, which, split="train"):
    """Load a ro cache through the FROZEN headspace_mint.load_split.

    §12 layer 2: the driver asserts `split == "train"` on every ro-cache load.  No ro
    dev_seen or test_seen file is reachable from here at all -- the split is a literal.

    CODE-R1 C-1.  This calls the FROZEN loader captured at import, never the module
    attribute.  The earlier version called `HM.load_split`, which on Head-R was still the
    driver's train-split override; the override discarded the `model_with_suffix` argument
    and returned the STANDARD ro cache for both which="std" and which="ow", so every
    Head-R mint carried h_std == h_ow and the battery HALTed in `l2_rows` on
    `displacement`.  Binding the frozen callable here makes the ro loads unreachable from
    the override by construction, and introduces NO branch on filename or suffix, so
    §13.1 item 3 is still satisfied literally.

    Returns (split_tuple, resolved_path) so the caller can assert the file it got is the
    file it asked for.
    """
    assert split == "train", \
        "SPLIT GUARD: this battery reads the train split only, got {!r}".format(split)
    cfg = P.DATASETS[dataset]
    model_with_suffix = "{}-{}".format(cfg["model"], RO_SUFFIX[which])
    resolved = os.path.join(cfg["cache_dir"],
                            "{}_{}.pt".format(split, model_with_suffix))
    assert os.path.realpath(resolved) == os.path.realpath(ro_cache_path(dataset, which)), \
        "RO PATH GUARD: resolved {} != expected {}".format(
            resolved, ro_cache_path(dataset, which))
    return _FROZEN_LOAD_SPLIT(cfg["cache_dir"], split, model_with_suffix), resolved


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=sorted(HM.CLI))
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--fold", type=int, required=True,
                    help="0..4 = fitting-pool head for that frozen fold; "
                         "-1 = full-train head (fidelity instrument)")
    ap.add_argument("--lineage", required=True, choices=["N", "R"],
                    help="record-only label written into meta; it selects NOTHING. "
                         "The lineage is determined by --train-cache alone.")
    ap.add_argument("--train-cache", default=None,
                    help="absolute path overriding ONLY the train split load. "
                         "Absent => the frozen native cache (Head-N).")
    ap.add_argument("--out", required=True)
    ap.add_argument("--scratch", required=True)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--progress", default=None,
                    help="§9 progress file; every python process this lineage authors "
                         "appends to it (H-3)")
    a = ap.parse_args()

    guard = assert_guard_active()
    tag = "{} {} s{} f{}".format(a.dataset, a.lineage, a.seed, a.fold)
    heartbeat(a.progress, "MINT-START", extra=tag)

    # ---- resume: the final .npz is written atomically and always carries every array,
    #      so its presence is a complete record (§12 "Why mints_executed and not 66").
    if os.path.exists(a.out):
        heartbeat(a.progress, "MINT-SKIP", extra="{} (resume; .npz already complete)".format(tag))
        return

    # ---- the ONLY lineage-varying behaviour: redirect the train split load ------------
    # No branch on filename or suffix (§13.1 item 3): a single unconditional substitution
    # of the path, applied when and only when the frozen loader asks for "train".
    override = a.train_cache
    if override is not None:
        assert os.path.exists(override), "train-cache not found: {}".format(override)
    _frozen_load_split = HM.load_split

    def _load_split_with_override(cache_dir, split, model):
        if split == "train" and override is not None:
            directory, filename = os.path.split(override)
            assert filename.startswith("train_") and filename.endswith(".pt"), \
                "SPLIT GUARD: --train-cache must name a train split, got {!r}".format(
                    filename)
            stem = filename[len("train_"):-len(".pt")]
            return _frozen_load_split(directory, "train", stem)
        return _frozen_load_split(cache_dir, split, model)

    HM.load_split = _load_split_with_override

    # ---- capture the trained head without touching the frozen module ------------------
    import run_rac                                          # noqa: E402
    _outer_model_pass = run_rac.model_pass
    hold = {}

    def _capture(train_dl, evaluate_dl, test_seen_dl, model, **kw):
        hold["model"] = model
        return _outer_model_pass(train_dl, evaluate_dl, test_seen_dl, model, **kw)

    run_rac.model_pass = _capture

    # ---- call the FROZEN mint, unmodified, into a staging path ------------------------
    stage_dir = os.path.join(a.scratch, "c06_stage", os.path.basename(a.out))
    if os.path.isdir(stage_dir):
        shutil.rmtree(stage_dir)
    os.makedirs(stage_dir, exist_ok=True)
    stage_npz = os.path.join(stage_dir, "frozen_mint.npz")

    argv_saved = sys.argv
    sys.argv = ["headspace_mint.py",
                "--dataset", a.dataset, "--seed", str(a.seed), "--fold", str(a.fold),
                "--out", stage_npz, "--scratch", stage_dir,
                "--threads", str(a.threads)]
    try:
        HM.main()
    finally:
        sys.argv = argv_saved
        # CODE-R1 C-1: the override is scoped to the frozen main() call ONLY.  It is
        # removed here so that nothing after this point -- in particular the driver's own
        # ro forwards -- can see it.  Belt and braces with _FROZEN_LOAD_SPLIT.
        HM.load_split = _frozen_load_split

    assert os.path.exists(stage_npz), "frozen mint produced no output"
    assert "model" in hold, "trained head was not captured"
    model = hold["model"]
    model.eval()

    def keys_of(split_tuple):
        with torch.no_grad():
            _, emb = model(split_tuple[1], split_tuple[2], return_embed=True)
        return emb.detach().cpu().numpy().astype("float64")

    # ---- the ro forwards, in this same process (§13.1 item 22) ------------------------
    # Head-R's K_train IS its h_std, because --train-cache pointed at the ro_L24 cache;
    # recomputing it would be a second forward of the same object.  §8 Phase 1b prices
    # exactly this: 3 forwards per Head-N fold mint, 2 per Head-R fold mint.
    z = np.load(stage_npz, allow_pickle=True)
    frozen = {k: z[k] for k in z.files}
    meta = json.loads(str(z["meta"]))

    ro_std_is_k_train = (override is not None
                         and os.path.realpath(override)
                         == os.path.realpath(ro_cache_path(a.dataset, "std")))
    if ro_std_is_k_train:
        h_std = frozen["K_train"]
        resolved_std = os.path.realpath(override)
        n_extra_forwards = 1
    else:
        split_std, resolved_std = load_ro_split(a.dataset, "std")
        h_std = keys_of(split_std)
        n_extra_forwards = 2
    split_ow, resolved_ow = load_ro_split(a.dataset, "ow")
    h_ow = keys_of(split_ow)

    assert h_std.shape == h_ow.shape == frozen["K_train"].shape, \
        "ro key matrices disagree in shape with the frozen K_train"
    # CODE-R1 C-1: a one-line falsifier for this whole defect class.  It cannot fire on a
    # correct run -- §7.8 measures min_i d_i at 0.018-0.038 across four trained cells -- and
    # it fires immediately if the two ro forwards ever read the same file again.
    assert not np.array_equal(h_std, h_ow), (
        "RO PROVENANCE: h_std and h_ow are bit-identical, so the one-word forward read the "
        "standard cache. std={} ow={}".format(resolved_std, resolved_ow))
    assert os.path.realpath(resolved_ow) != os.path.realpath(resolved_std), \
        "RO PROVENANCE: the two ro forwards resolved to the same file"

    meta["c06"] = {
        "driver_sha256": sha256_of(os.path.abspath(__file__)),
        "frozen_sha256": FROZEN_SHA,
        "lineage": a.lineage,
        "train_cache": override,
        "ro_std_path_resolved": resolved_std,
        "ro_ow_path_resolved": resolved_ow,
        "ro_std_path_expected": ro_cache_path(a.dataset, "std"),
        "ro_ow_path_expected": ro_cache_path(a.dataset, "ow"),
        "h_std_equals_h_ow": False,
        "ro_std_is_k_train": bool(ro_std_is_k_train),
        "n_extra_key_forwards": int(n_extra_forwards),
        "split_guard": "train only; asserted on every ro-cache load",
    }

    tmp = a.out + ".tmp.npz"
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    np.savez(tmp,
             K_train=frozen["K_train"], K_dev=frozen["K_dev"],
             lab=frozen["lab"], lab_dev=frozen["lab_dev"],
             fold_of=frozen["fold_of"], fit_idx=frozen["fit_idx"],
             h_std=h_std, h_ow=h_ow,
             meta=json.dumps(meta))
    os.replace(tmp, a.out)
    z.close()
    shutil.rmtree(stage_dir, ignore_errors=True)
    heartbeat(a.progress, "MINT-DONE",
              extra="{} -> {} (+{} ro forwards; dev_opens={})".format(
                  tag, os.path.basename(a.out), n_extra_forwards,
                  guard.LEDGER.get("dev_path_opens", "?")))


if __name__ == "__main__":
    main()
