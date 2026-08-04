#!/usr/bin/env python
"""c06_falsifier_arena.py -- the C06 $0 CPU falsifier battery.

Frozen design: refine-logs/C06_FALSIFIER_PREREG_DRAFT_V15E2.md (GO at round 15,
    + ERRATUM 1 + CODE-R1 §8 correction + ERRATUM 2).
Implemented exactly as written; every departure is flagged IMPLEMENTATION NOTE and was
reported to the design lineage rather than improvised.

WHAT THIS RUNS
    §3.4 arm builder  -> 13 key-space arms + avg_score + 2 guard arms, in head space,
                         built by ONE construction parameterised by a block list.  The
                         two-block instantiation is compared BIT-EXACTLY against C01's own
                         prepare_views (GATE-C01PARITY, predicate max|diff| == 0.0).
    §6  twenty gates  -> 12 global (HALT the battery) / 6 per-lineage / 2 reporting.
    §5  decision rule -> S1-S7 on each lineage that passed its per-lineage gates, on BOTH
                         datasets, with the Holm family frozen at 92 per dataset.
    §9  heartbeat     -> line-buffered per-phase appends to the progress file.
    §12 ledger        -> GATE-LEDGER's declared-count predicate set, process count binding.

REUSE, NOT REWRITE (§13.1 items 19, 23).  The algebra is C01's own, imported unmodified
    with its sha256 asserted: l2_rows, prepare_views, holm_adjust, id_hash_permutation,
    select_strongest_ordinary_control.  The arm->formula map is pinned by GATE-C01PARITY
    against prepare_views and by NOTHING ELSE; this file never re-derives it from prose.

TEST CONTACT: NONE.  This process opens the banked mint .npz, the four train-split ro
    caches, the banked vsw_ckpt npz, the banked arena OUT JSONs and the sha-gated configs.
    No dev_seen_*-ro_* file and no test_seen file is reachable from any path here.

DETERMINISM: DET-1 asserted before any compute module is imported.
COST: CPU only, <= 8 threads.  Zero GPU.  §8 projects 3674.0 s (4592.5 s conservative)
    under v15 + ERRATUM 1 + CODE-R1 H-4 + ERRATUM 2
    (refine-logs/C06_FALSIFIER_PREREG_DRAFT_V15E2.md).
"""
import argparse
import hashlib
import json
import math
import os
import sys
import time
from datetime import datetime, timezone

import numpy as np

REPO = "/data/jehc223/RGCL"
sys.path.insert(0, os.path.join(REPO, "scripts/analysis"))

# ERRATUM 2 §7: configs/c06/c06_falsifier.json:"projected_seconds" is the SINGLE SOURCE.
# The sbatch exports it; this literal is the hand-run fallback only, and main() asserts
# environment == this constant == cfg["projected_seconds"] before any battery compute.
PROJECTED_SECONDS = float(os.environ.get("C06_PROJECTED_SECONDS", 3674.0))
TOPK = 20                           # §3.2 deployed budget
TIE_RANK_WINDOW = 21                # §6.5 "the union of the two arms' top-21 sets"
GATE_ALGEBRA_BAR = 2e-6             # §6
DESIGN_SHA_NOTE = ("declared digest is not pinned inside the job; the external anchor is "
                   "the freeze record in "
                   "refine-logs/C06_FALSIFIER_IMPLEMENTATION_RECORD.md")


def _derived_design_sha(bat):
    """ERRATUM 2 round-4 I-1: every artifact publishes the DERIVED digest beside the
    declared one.  emit_halt is reachable from GATE-DET1, before gate_sha has run."""
    try:
        return bat.reports.get("design_sha256_derived", "NOT_DERIVED")
    except Exception:
        return "NOT_DERIVED"
UPPER_ARENA_BAR = 0.98              # §6.3


# ============================================================ heartbeat (§9, item 12)
class Heartbeat(object):
    """One append-only, line-buffered handle, opened once and never re-wrapped."""

    def __init__(self, path, projected=PROJECTED_SECONDS):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.fh = open(path, "a", buffering=1)
        self.t0 = time.time()
        self.projected = float(projected)

    def __call__(self, phase, done=None, total=None, extra=""):
        elapsed = time.time() - self.t0
        stamp = datetime.now(timezone.utc).isoformat(timespec="seconds")
        units = "{}/{}".format(done, total) if total is not None else "-"
        line = "{} | {} | {} | {:.1f}s | {:.3f}x{}".format(
            stamp, phase, units, elapsed, elapsed / self.projected,
            (" | " + extra) if extra else "")
        self.fh.write(line + "\n")
        print(line, flush=True)

    def close(self):
        try:
            self.fh.close()
        except Exception:
            pass


# ============================================================ small utilities
def sha256_of(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for blk in iter(lambda: f.read(1 << 20), b""):
            h.update(blk)
    return h.hexdigest()


class GateFailure(Exception):
    """A gate that did not pass.  Carries the gate name for the final heartbeat line."""

    def __init__(self, gate, detail):
        Exception.__init__(self, "{}: {}".format(gate, detail))
        self.gate = gate
        self.detail = detail


def finite(name, *arrays):
    """§5.6 / §13.1 item 16: every gate AND DECISION quantity is asserted finite and
    present.  CODE-R1 M-7: called on S4's lower/one_sided_raw_p, S5's observed/p95, net_s
    and S7's fixed_fraction/threshold as well as on acc_s/mf1_s."""
    for a in arrays:
        arr = np.asarray(a, dtype="float64")
        if arr.size == 0 or not np.isfinite(arr).all():
            raise GateFailure("FINITENESS", "{} is absent or non-finite".format(name))


# ============================================================ §3.4 the arm builder
class ArmBuilder(object):
    """ONE construction, parameterised by an ordered list of blocks (§3.4).

        fuse(blocks) = l2(concat[ l2(b) for b in blocks ])
        paired(A,B)  = fuse([ l2(concat[ l2(A_m), l2(B_m) ]) for m in blocks ])

    Two blocks [img, text] must reproduce C01's prepare_views bit-exactly; one block
    (the fused head key) gives the head-space arms.  Every l2 is C01's own l2_rows,
    called through the imported module, with the epsilon read from the sha-gated config
    and an EXPLICIT boolean zero_mask -- never None (§3.7, §13.1 item 8).
    """

    def __init__(self, c01, epsilon, angles, arm_prefix):
        self.c01 = c01
        self.epsilon = float(epsilon)
        self.angles = list(angles)
        self.arm_prefix = arm_prefix

    def l2(self, array, context, zero_mask):
        assert zero_mask is not None and isinstance(zero_mask, np.ndarray) \
            and zero_mask.dtype == np.bool_, \
            "mask convention: zero_mask must be an explicit boolean array (§3.7)"
        assert zero_mask.shape == (len(array),), \
            "mask/population mismatch at {} (§13.1 item 8)".format(context)
        out, _ = self.c01.l2_rows(array, self.epsilon, context, zero_mask)
        return out

    def fuse(self, blocks, context, zero_mask):
        parts = [self.l2(b, context + "/b{}".format(i), zero_mask)
                 for i, b in enumerate(blocks)]
        return self.l2(np.concatenate(parts, axis=1), context + "/fused", zero_mask)

    def paired(self, blocks_a, blocks_b, context, zero_mask):
        assert len(blocks_a) == len(blocks_b)
        joined = []
        for i, (a, b) in enumerate(zip(blocks_a, blocks_b)):
            pa = self.l2(a, context + "/a{}".format(i), zero_mask)
            pb = self.l2(b, context + "/b{}".format(i), zero_mask)
            joined.append(self.l2(np.concatenate([pa, pb], axis=1),
                                  context + "/pair{}".format(i), zero_mask))
        return self.fuse(joined, context + "/pair_fused", zero_mask)

    def build_views(self, standard_blocks, oneword_blocks, context, zero_mask,
                    only=None):
        """The 13 key-space arms plus the two guard arms.

        CODE-R1 H-4: `only` is an optional allow-list of arm names.  It changes WHICH arms
        are emitted, never HOW any of them is built -- every emitted arm runs the identical
        code path it runs when `only is None`, so this remains ONE construction and
        §13.1 item 19's parity warrant (GATE-C01PARITY against prepare_views) transfers
        unchanged.  GATE-C01PARITY always calls with `only=None`, so the anchor is fired on
        the full set on every run.  S5 reads two arms and previously built fifteen: measured
        10.8x, and §8's U4 = 0.0891 s prices a two-arm build, so building fifteen made the
        realised Phase 3 ~6001 s against a priced 273.7 s.

        The arm->formula map below is the one GATE-C01PARITY pins against prepare_views.
        The ENDPOINT PRE-NORMALISATION is explicit and load-bearing: std[m] and ow[m] are
        the l2_rows-NORMALISED endpoint blocks, matching prepare_views:1296-1304.  Omitting
        it is a wrong-verdict path (round-7 C-1) that a 2e-6 tolerance would have admitted.
        """
        std = [self.l2(b, context + "/standard/{}".format(i), zero_mask)
               for i, b in enumerate(standard_blocks)]
        ow = [self.l2(b, context + "/oneword/{}".format(i), zero_mask)
              for i, b in enumerate(oneword_blocks)]

        want = None if only is None else set(only)

        def need(name):
            return want is None or name in want

        common, displacement, interaction = None, None, None
        if any(need(k) for k in ("common", "displacement", "common_displacement",
                                 "common_interaction")):
            common, displacement, interaction = [], [], []
            need_inter = need("common_interaction")
            for i, (s, o) in enumerate(zip(std, ow)):
                c = self.l2(s + o, context + "/{}/common".format(i), zero_mask)
                d = self.l2(o - s, context + "/{}/displacement".format(i), zero_mask)
                common.append(c)
                displacement.append(d)
                if need_inter:
                    interaction.append(self.l2(
                        c * d, context + "/{}/common_interaction".format(i), zero_mask))

        views = {}
        if need("endpoint_std"):
            views["endpoint_std"] = self.fuse(std, context + "/endpoint_std", zero_mask)
        if need("endpoint_ow"):
            views["endpoint_ow"] = self.fuse(ow, context + "/endpoint_ow", zero_mask)
        if need("endpoint_concat"):
            views["endpoint_concat"] = self.paired(std, ow, context + "/endpoint_concat",
                                                   zero_mask)
        if need("common"):
            views["common"] = self.fuse(common, context + "/common", zero_mask)
        if need("displacement"):
            views["displacement"] = self.fuse(displacement, context + "/displacement",
                                              zero_mask)
        if need("common_displacement"):
            views["common_displacement"] = self.paired(
                common, displacement, context + "/common_displacement", zero_mask)
        if need("common_interaction"):
            views["common_interaction"] = self.paired(
                common, interaction, context + "/common_interaction", zero_mask)
        for angle in self.angles:
            name = self.c01.rotation_arm_name(self.arm_prefix, angle)
            if not need(name):
                continue
            first, second = self._rotation(std, ow, angle, context, zero_mask)
            views[name] = self.paired(first, second, context + "/" + name, zero_mask)

        # §3.5 guard arms: built by the ROTATION route, never aliased to their
        # counterparts, and voted (§13.1 item 14).
        for angle, gname in ((0.0, "guard_orthrot_0"), (45.0, "guard_orthrot_45")):
            if not need(gname):
                continue
            first, second = self._rotation(std, ow, angle, context, zero_mask)
            views[gname] = self.paired(first, second, context + "/" + gname, zero_mask)
        if want is not None and set(views) != want:
            raise GateFailure("ARM-BUILDER",
                              "allow-list {} produced {}".format(sorted(want),
                                                                 sorted(views)))
        return views

    def _rotation(self, std, ow, angle, context, zero_mask):
        theta = math.radians(float(angle))
        cosine, sine = math.cos(theta), math.sin(theta)
        first, second = [], []
        for i, (s, o) in enumerate(zip(std, ow)):
            first.append(self.l2(cosine * s + sine * o,
                                 context + "/rot{}/{}/first".format(angle, i), zero_mask))
            second.append(self.l2(-sine * s + cosine * o,
                                  context + "/rot{}/{}/second".format(angle, i),
                                  zero_mask))
        return first, second


# ============================================================ voting and metrics
def vote(mech, bank_keys, bank_lab, query_keys):
    """The deployed top-20 rank-weighted signed-cosine vote (mechfix_ops.py:94)."""
    votes, pred, idx, sim = mech.deployed_vote(bank_keys, bank_lab, query_keys, topk=TOPK)
    return votes, pred, idx, sim


def accuracy(y, p):
    return float((np.asarray(y) == np.asarray(p)).mean())


def macro_f1(mech, y, p):
    return float(mech.macro_f1(y, p))


def rho_of(keys):
    """§6.1: rho = ||mean_i k_i|| over UNIT keys, mean accumulated in float64 over the
    float32 keys (reduction order frozen by round-7 I-1)."""
    K = np.asarray(keys, dtype="float32")
    n = np.linalg.norm(K, axis=1)
    if not np.isfinite(n).all() or (n <= 0).any():
        # CODE-R1 H-1: GATE-ORBITDISP is a PER-LINEAGE gate (§5.6).  Raising here HALTed the
        # whole battery, reintroducing round-4 H-3 inside §5.6's own repair.  Return NaN;
        # the caller turns it into a drop of this lineage on BOTH datasets.
        return float("nan")
    U = (K / n[:, None]).astype("float32")
    return float(np.linalg.norm(U.astype("float64").mean(axis=0)))


# ============================================================ §6.5 the tie diagnostic
def vote_bounds_over_orderings(sim, lab, window):
    """Analytic worst/best case of the rank-weighted vote over all orderings of every
    near-tie group (§6.5; §13.1 item 10 -- ANALYTIC, never g! enumeration).

    `sim` and `lab` are the top-21 similarities (descending) and their bank labels.
    Neighbours whose similarities lie within `window` of each other are order-ambiguous;
    within such a group the rank weights may be permuted arbitrarily.  By the rearrangement
    inequality the extreme sums pair sorted signed values with sorted weights, so the bound
    is O(k log k) rather than O(g!).
    """
    k = len(sim)
    w_full = np.arange(1, TOPK + 1)[::-1].astype("float64")   # 20..1 for ranks 0..19
    w = np.zeros(k, dtype="float64")
    w[:min(k, TOPK)] = w_full[:min(k, TOPK)]                  # rank 20 (the 21st) weighs 0
    signed = (np.asarray(lab, dtype="float64") * 2.0 - 1.0) * np.asarray(sim,
                                                                        dtype="float64")

    groups, start = [], 0
    for j in range(1, k + 1):
        if j == k or (sim[j - 1] - sim[j]) > window:
            groups.append((start, j))
            start = j
    lo = hi = 0.0
    for a, b in groups:
        gw = np.sort(w[a:b])[::-1]
        gs = np.sort(signed[a:b])[::-1]
        hi += float(np.dot(gs, gw))                 # largest values on largest weights
        lo += float(np.dot(gs[::-1], gw))           # smallest values on largest weights
    denom = w_full.sum()
    return lo / denom, hi / denom


def _union_ranked(sim, lab, idx, keep_ids):
    """This arm's (sim, lab) restricted to `keep_ids`, still in this arm's descending
    order.  Union members this arm did not retrieve are outranked by the whole retrieved
    depth and carry weight 0 in every admissible ordering, so omitting them is sound."""
    m = np.isin(idx, keep_ids) & (idx >= 0)
    return sim[m], lab[m]


def tie_casualties(sim_a, lab_a, idx_a, sim_b, lab_b, idx_b, mismatch, window,
                   rank_window):
    """Which mismatching items COULD agree under some admissible ordering.

    CODE-R1 I-4: SS6.5's ranking is the UNION of the two arms' top-`rank_window` sets, not
    each arm's own.  Scoring each arm over its own window alone hid neighbours the other
    arm ranks inside the window and this one ranks just outside -- exactly the boundary
    case the window exists to catch -- which UNDER-counted casualties and could convert a
    warranted CLOSE into a HALT.
    """
    out = np.zeros(len(mismatch), dtype=bool)
    for i in np.flatnonzero(mismatch):
        ia = idx_a[i][:rank_window]
        ib = idx_b[i][:rank_window]
        keep_ids = np.union1d(ia[ia >= 0], ib[ib >= 0])
        sa, la = _union_ranked(sim_a[i], lab_a[i], idx_a[i], keep_ids)
        sb, lb = _union_ranked(sim_b[i], lab_b[i], idx_b[i], keep_ids)
        lo_a, hi_a = vote_bounds_over_orderings(sa, la, window)
        lo_b, hi_b = vote_bounds_over_orderings(sb, lb, window)
        both_pos = (hi_a >= 0.0) and (hi_b >= 0.0)
        both_neg = (lo_a < 0.0) and (lo_b < 0.0)
        out[i] = bool(both_pos or both_neg)
    return out


# ============================================================ statistics (§5.4, §5.4.1)
def bootstrap_deltas(correct_a, correct_c, draws):
    """§5.4 ACCURACY leg -- RETAINED VERBATIM under ERRATUM 1 obligation 1.

    Delta_b = mean_{i in draw_b}[cbar_A(i)] - mean_{i in draw_b}[cbar_c(i)], where cbar_X(i)
    is the mean over the three seeds of item i's 0/1 correctness.  The seed axis is INSIDE
    the statistic, not a hidden multiplicity.

    This expression is NOT re-associated into C01's per-seed form.  The two are the same real
    number (means commute) but not bit-identical (max|diff| = 5.55e-16 measured), and S4's two
    predicates -- a strict `lower > 0` and a zero-adverse-count at p = 1/2001 -- are the only
    ulp-sensitive predicates in the design: 38.8% of near-identical arm pairs get a different
    one_sided_raw_p under the two forms and 1.2% flip `lower > 0`.  Retaining this expression
    departs from C01 by nothing and from fifteen reviewed rounds by nothing.
    """
    a = np.asarray(correct_a, dtype="float64")
    c = np.asarray(correct_c, dtype="float64")
    return a[draws].mean(axis=1) - c[draws].mean(axis=1)


def resampled_macro_f1(pred_by_seed, lab, draws):
    """§5.4 MACRO-F1 leg -- ADDED by ERRATUM 1.  C01's recompute-per-resample form.

    Returns the (B,) vector of seed-mean macro-F1 on each resample, for ONE arm.  The metric is
    recomputed from the resampled predictions and the resampled gold, exactly as C01's
    paired_bootstrap does (c01_policy_contrast_a0.py:1742-1772), with the seed mean inside the
    statistic as the accuracy bullet requires.

    The function is `mechfix_ops.macro_f1` (mechfix_ops.py:56-66, sha-frozen at §11 as
    635c1312...), NAMED IN THE DESIGN rather than only here: it and C01's metric_bundle differ on
    39.84% of all 68,915,480 confusion triples with tp+fp+fn <= 743, and the tie direction is
    disclosed at §5.9 item 10.  The loop below is a vectorised replica of that function -- same
    operations in the same order -- verified bit-identical over 300 draws (max|diff| = 0.000e+00).

    Degenerate draws: a class absent from the draw, or never predicted in it, contributes
    per-class F1 = 0.0.  Both candidate functions agree exactly there and neither returns None,
    so C01's class-degenerate `die` guard has no object here (holm_metrics excludes roc_auc).

    Predictions are resampled directly, which is identical to C01's resample-scores-then-threshold
    because retrieval.prediction_cutoff = 0.0 matches deployed_vote's `votes >= 0` convention and
    item-wise thresholding commutes with resampling.
    """
    lab = np.asarray(lab)
    ld = lab[draws]
    per_seed = []
    for pred in pred_by_seed:
        pd = np.asarray(pred)[draws]
        fs = []
        for cls in (0, 1):
            tp = ((pd == cls) & (ld == cls)).sum(1).astype("float64")
            fp = ((pd == cls) & (ld != cls)).sum(1).astype("float64")
            fn = ((pd != cls) & (ld == cls)).sum(1).astype("float64")
            pr = np.where(tp + fp > 0, tp / np.where(tp + fp > 0, tp + fp, 1.0), 0.0)
            rc = np.where(tp + fn > 0, tp / np.where(tp + fn > 0, tp + fn, 1.0), 0.0)
            fs.append(np.where(pr + rc > 0,
                               2 * pr * rc / np.where(pr + rc > 0, pr + rc, 1.0), 0.0))
        per_seed.append((fs[0] + fs[1]) / 2.0)
    return np.mean(per_seed, axis=0)


def one_sided_p(deltas):
    """C01's own form at :1769."""
    d = np.asarray(deltas, dtype="float64")
    return float((1 + np.sum(d <= 0.0)) / (len(d) + 1))


def null_p(observed, nulls):
    """§5.4.1: p = (1 + #{b : null_b >= observed}) / (256 + 1)."""
    nn = np.asarray(nulls, dtype="float64")
    return float((1 + np.sum(nn >= observed)) / (len(nn) + 1))


# ============================================================ the battery
class Battery(object):
    DATASET_KEYS = ("hatemm", "zh")
    LINEAGES = ("N", "R")
    SEEDS = (0, 1, 2)

    def __init__(self, cfg, args, hb):
        self.cfg = cfg
        self.args = args
        self.hb = hb
        self.ledger = {}          # CODE-R1 H-2: filled from c09guard's MEASURED counts,
        self.guard = None         # never from literals.  See gate_ledger().
        self.gates = {}
        self.reports = {}
        self.dropped = {}          # lineage -> reason
        self.mint_cache = {}
        self.ro_cache = {}           # CODE-R1 M-12

    # -------------------------------------------------------- test guard (H-2)
    def assert_guard_active(self):
        """§13.1 item 28: layer 3 must be ACTIVE in every process, not merely importable.
        c09_guard's sitecustomize swallows every failure by design, so without this a
        silent guard failure is indistinguishable from a clean run and GATE-LEDGER would
        still publish test_path_opens: 0."""
        try:
            import c09guard
        except Exception as exc:
            raise GateFailure("GATE-LEDGER",
                              "test guard layer 3 not importable: {}".format(exc))
        if not getattr(c09guard, "_INSTALLED", False):
            raise GateFailure("GATE-LEDGER",
                              "test guard layer 3 imported but install() did not take effect")
        self.guard = c09guard
        return c09guard

    # -------------------------------------------------------- GATE-LEDGER (H-2)
    def gate_ledger(self, mints_executed):
        """§12's declared-count predicate set, MEASURED via c09guard rather than asserted.

        CODE-R1 H-2: this gate was previously hardcoded PASS with five never-incremented
        counters, and the verdict face published those literals as counts.  It now
        aggregates every process's ledger file and evaluates each predicate as a
        pass-condition.

        ERRATUM 2 (LANDED).  §12 previously bound `dev_path_opens == mints_executed + 0`,
        warranted on headspace_fidelity.py opening no dev_seen file -- which is true, and it
        reads only meta out of the banked mint .npz, referencing no label array at all.  But
        round-8 H-1 widened GATE-SHA to the input caches, and two of those are
        `dev_seen_*.pt`; hashing them with builtins.open is counted by c09guard.is_dev_like,
        and GATE-SHA runs in TWO processes (the driver's --gate-sha-only call and the arena),
        so a clean run measures mints_executed + 4.  The predicate is now the TWO-TERM form
        `dev_path_opens == mints_executed + expected_sha_dev_opens`, and the second term is
        DERIVED here from the digest tables and ASSERTED against the config's declared value
        -- never merely read from it.
        """
        guard = self.guard or self.assert_guard_active()
        ledger_dir = os.environ.get("C09_LEDGER_DIR", "")
        tot, procs, stale = guard.aggregate(ledger_dir)
        # ERRATUM 2: the GATE-SHA pass count is AUDITED, never declared -- this process's
        # own pass plus every ledger-writing process whose recorded argv carries
        # --gate-sha-only.  A mismatch against the config HALTs rather than being trusted.
        # c09guard.aggregate returns argv as a joined STRING, not a list.
        self.GATE_SHA_PASSES = int("GATE-SHA" in self.gates) + sum(
            1 for pr in procs if "--gate-sha-only" in (pr.get("argv") or ""))
        for k, v in guard.LEDGER.items():          # this process has not run atexit yet
            tot[k] = tot.get(k, 0) + int(v)
        # ERRATUM 2 §3 criterion: only counters some code path INCREMENTS are published as
        # measured integers.  c09guard increments exactly three (:97, :102, :106); the other
        # three are published as by-construction warranted strings in a separate block and
        # are never binding.  The vacuous runtime assertions below are RETAINED VERBATIM as
        # defence-in-depth -- publication moves, checking does not.
        _MEASURED = ("test_path_opens", "dev_path_opens", "banked_trainlog_opens")
        self.ledger = {k: int(v) for k, v in tot.items() if k in _MEASURED}
        for k in _MEASURED:
            self.ledger.setdefault(k, 0)
        self.ledger_by_construction = dict(self.cfg["ledger_by_construction"])
        self.ledger["processes_reporting"] = len(procs) + 1
        self.ledger["mints_present_before_arena"] = self.reports[
            "mints_present_before_arena"]
        self.ledger["mints_executed"] = int(mints_executed)
        self.ledger["stale_attempt_files"] = len(stale)

        fails = []
        if tot.get("test_path_opens", 0) != 0:
            fails.append("test_path_opens = {} != 0".format(tot["test_path_opens"]))
        if tot.get("test_label_materialisations", 0) != 0:
            fails.append("test_label_materialisations = {} != 0".format(
                tot["test_label_materialisations"]))
        if tot.get("dev_or_test_labels_into_decision_quantities", 0) != 0:
            fails.append("dev_or_test_labels_into_decision_quantities != 0")
        if self.reports["mints_present_before_arena"] != 66:
            fails.append("mints_present_before_arena != 66")
        want_procs = int(self.cfg["ledger"]["processes_reporting"]["expected"])
        if self.ledger["processes_reporting"] != want_procs:
            fails.append("processes_reporting = {} != {} ({})".format(
                self.ledger["processes_reporting"], want_procs,
                self.cfg["ledger"]["processes_reporting"]["decomposition"]))
        # ERRATUM 2: the second term is DERIVED from the digest tables, then asserted
        # against the config's declared value (round-3 M-3: never merely read).
        derived_dev_files = sum(
            1 for rel in list(self.cfg["frozen_sha256"])
            + list(self.cfg["frozen_sha256_input_caches"])
            if guard.is_dev_like(os.path.join(REPO, rel)))
        derived_sha_dev_opens = derived_dev_files * self.GATE_SHA_PASSES
        declared = int(self.cfg["ledger"]["dev_path_opens"]["expected_sha_dev_opens"])
        if derived_sha_dev_opens != declared:
            fails.append(
                "expected_sha_dev_opens derived {} ({} dev-like x {} GATE-SHA passes) "
                "!= declared {}".format(derived_sha_dev_opens, derived_dev_files,
                                        self.GATE_SHA_PASSES, declared))
        self.ledger["expected_sha_dev_opens"] = derived_sha_dev_opens
        dev = tot.get("dev_path_opens", 0)
        if dev != mints_executed + derived_sha_dev_opens:
            fails.append(
                "dev_path_opens = {} != mints_executed ({}) + expected_sha_dev_opens ({}) "
                "-- measured excess {}. The second term is GATE-SHA hashing the two "
                "dev_seen input caches with builtins.open in each of its two passes.".format(
                    dev, mints_executed, derived_sha_dev_opens,
                    dev - mints_executed - derived_sha_dev_opens))
        # the guard's own predicate check, called rather than reimplemented
        vp = guard.verify_predicate()
        self.ledger["guard_verify_predicate"] = vp
        self.ledger["predicate_failures"] = fails
        if fails:
            raise GateFailure("GATE-LEDGER", "; ".join(fails))
        self.gates["GATE-LEDGER"] = "PASS (measured)"

    # -------------------------------------------------------- frozen inputs
    def load_frozen(self):
        self.c01cfg = json.load(open(os.path.join(REPO, "configs/c01/c01_a0_v2.json")))
        import c01_policy_contrast_a0 as C01
        # §3.4 deferred-import note: c01_policy_contrast_a0.py:387 sets np = torch =
        # faiss = None; the algebra is unusable until import_compute_modules runs.
        C01.import_compute_modules(self.c01cfg)
        import mechfix_ops as MECH
        import mechnov_pairverify as PV
        # CODE-R1 M-10: SS13.1 item 27 makes the arena's import set the deliverable that
        # U11's arena class was measured over.  headspace_mint and vsw_pregate were absent
        # and runtime_block() was never called, so the verdict carried no runtime
        # provenance.  Both are imported here and the block is recorded.
        import headspace_mint as HM
        try:
            import vsw_pregate as VSW           # noqa: F401
        except Exception as exc:                # pragma: no cover
            VSW = None
            self.reports["vsw_pregate_import"] = "ABSENT: {}".format(exc)
        self.reports["runtime"] = HM.runtime_block()
        self.C01, self.MECH, self.PV = C01, MECH, PV

        tr = self.c01cfg["transforms"]
        # §3.7 second block, §13.1 item 5b: READ from the sha-gated config and asserted
        # equal to the design's frozen values.  Never recomputed.
        frozen = self.cfg["frozen_c01_constants_read_and_asserted"]
        pairs = [("normalization_epsilon", tr["normalization_epsilon"]),
                 ("tiny_displacement_epsilon", tr["tiny_displacement_epsilon"]),
                 ("max_tiny_displacement_fraction", tr["max_tiny_displacement_fraction"]),
                 ("max_small_displacement_fix_fraction",
                  tr["max_small_displacement_fix_fraction"]),
                 ("small_displacement_train_quantile",
                  tr["small_displacement_train_quantile"])]
        d = self.c01cfg["decision"]
        st = self.c01cfg["statistics"]
        rot = self.c01cfg["orthogonal_rotation_control"]
        # M-1: the config key is `frozen_c01_constants_read_and_asserted` and §13.1 item 5b's
        # verb is "asserted equal to it"; five of eighteen were asserted.  All eighteen are.
        pairs += [("minimum_gain_over_strongest_control",
                   d["minimum_gain_over_strongest_control"]),
                  ("statistics_seed", st["seed"]),
                  ("n_bootstrap", st["n_bootstrap"]),
                  ("bootstrap_lower_quantile", st["bootstrap_lower_quantile"]),
                  ("bootstrap_upper_quantile", st["bootstrap_upper_quantile"]),
                  ("n_id_hash_permutations", st["n_id_hash_permutations"]),
                  ("holm_alpha", st["holm_alpha"]),
                  ("topk_deployed", self.c01cfg["retrieval"]["topk"])]
        for name, got in pairs:
            if float(frozen[name]) != float(got):
                raise GateFailure("GATE-SHA", "C01 constant {} drifted: {} vs {}".format(
                    name, got, frozen[name]))
        for name, got in [("minimum_net_fixes", d["minimum_net_fixes"]),
                          ("gain_controls", d["gain_controls"]),
                          ("angles_degrees", rot["angles_degrees"]),
                          ("rotation_arm_prefix", rot["arm_prefix"]),
                          ("permutation_hash", st["permutation_hash"]),
                          ("holm_metrics", st["holm_metrics"])]:
            if frozen[name] != got:
                raise GateFailure("GATE-SHA", "C01 constant {} drifted: {!r} vs {!r}".format(
                    name, got, frozen[name]))
        if frozen["small_set_comparison_operator"] != "<=":
            raise GateFailure("GATE-SHA", "S7's small-set operator is not '<='")
        # M-11: decision-relevant literals come from the sha-gated config, not the module.
        global TOPK, GATE_ALGEBRA_BAR, TIE_RANK_WINDOW, UPPER_ARENA_BAR
        TOPK = int(self.c01cfg["retrieval"]["topk"])
        GATE_ALGEBRA_BAR = float(self.cfg["gates"]["GATE-ALGEBRA"]["bar"])
        TIE_RANK_WINDOW = int(self.cfg["gates"]["GATE-ZEROOP"].get("rank_window", 21))
        UPPER_ARENA_BAR = float(self.cfg["gates"]["GATE-ARENA"]["upper_bound"])
        self.epsilon = float(tr["normalization_epsilon"])
        self.gain_controls = list(self.c01cfg["decision"]["gain_controls"])
        self.angles = list(self.c01cfg["orthogonal_rotation_control"]["angles_degrees"])
        self.arm_prefix = self.c01cfg["orthogonal_rotation_control"]["arm_prefix"]
        self.builder = ArmBuilder(C01, self.epsilon, self.angles, self.arm_prefix)
        self.rot_names = [C01.rotation_arm_name(self.arm_prefix, a) for a in self.angles]
        self.key_arms = ["endpoint_std", "endpoint_ow", "endpoint_concat", "common",
                         "displacement", "common_displacement",
                         "common_interaction"] + self.rot_names
        self.all_arms = self.key_arms + ["avg_score"]          # the fourteen (§3.5)

    # -------------------------------------------------------- GATE-SHA / GATE-DET1
    def gate_det1(self):
        want = self.cfg["execution"]["required_environment"]
        bad = [k for k, v in want.items() if os.environ.get(k) != v]
        if bad:
            raise GateFailure("GATE-DET1", "thread env not exported before start: "
                                           "{}".format(bad))
        self.gates["GATE-DET1"] = "PASS"

    def gate_sha(self):
        """§6: every frozen import, input cache, the sixteen banked artifacts AND the
        design document itself (ERRATUM 2 §3).
        The FIRST OF TWO passes; the arena repeats it at the point of use (TOCTOU)."""
        n = 0
        # ERRATUM 2 §3: the design document becomes the 38th GATE-SHA artifact, hashed
        # through the same sha256_of and compared against cfg["design_sha256"].  It is a
        # .md in refine-logs/ -- neither dev-like nor test-like under c09guard -- so the
        # dev-like count over this iterable stays 2 and expected_sha_dev_opens stays 4.
        design_rel = self.cfg["design_document"]
        design_want = self.cfg["design_sha256"]
        design_path = os.path.join(REPO, design_rel)
        if not os.path.exists(design_path):
            raise GateFailure("GATE-SHA", "design document missing: {}".format(design_rel))
        design_got = sha256_of(design_path)
        self.reports["design_sha256_derived"] = design_got
        if design_got != design_want:
            raise GateFailure("GATE-SHA",
                              "design document {} on-disk digest {} != declared {}"
                              .format(design_rel, design_got, design_want))
        n += 1
        for rel, want in list(self.cfg["frozen_sha256"].items()) \
                + list(self.cfg["frozen_sha256_input_caches"].items()):
            path = os.path.join(REPO, rel)
            if not os.path.exists(path):
                raise GateFailure("GATE-SHA", "missing frozen artifact {}".format(rel))
            got = sha256_of(path)
            if got != want:
                raise GateFailure("GATE-SHA", "{} digest {}".format(rel, got))
            n += 1
        for ds in self.DATASET_KEYS:
            for s in self.SEEDS:
                p = os.path.join(REPO, "scripts/analysis",
                                 "headspace_arena_{}_s{}_OUT.json".format(ds, s))
                if not os.path.exists(p):
                    raise GateFailure("GATE-SHA", "missing banked anchor {}".format(p))
                n += 1
            for f in range(5):
                p = os.path.join(REPO, "scripts/analysis/vsw_ckpt", ds,
                                 "f{}.npz".format(f))
                if not os.path.exists(p):
                    raise GateFailure("GATE-SHA", "missing vsw_ckpt {}".format(p))
                n += 1
        self.gates["GATE-SHA"] = "PASS"
        self.reports["gate_sha_artifacts"] = n
        self.hb("GATE-SHA", n, n, "all frozen digests match")

    # -------------------------------------------------------- mints, folds, populations
    def mint_path(self, ds, lineage, seed, fold):
        return os.path.join(self.args.mintdir,
                            "mint_{}_{}_s{}_f{}.npz".format(
                                ds, lineage, seed, "full" if fold < 0 else fold))

    def load_mint(self, ds, lineage, seed, fold):
        key = (ds, lineage, seed, fold)
        if key not in self.mint_cache:
            p = self.mint_path(ds, lineage, seed, fold)
            if not os.path.exists(p):
                raise GateFailure("GATE-LEDGER", "mint absent: {}".format(p))
            z = np.load(p, allow_pickle=True)
            self.mint_cache[key] = z
        return self.mint_cache[key]

    def gate_fold_and_ledger_presence(self):
        """§3.2 / §6 GATE-FOLD: re-read meta['fold_parity_vs_banked_vsw_ckpt'] and fold_of
        from ALL 66 banked .npz -- resume-safe, and free.  Also the binding
        mints_present_before_arena == 66 (§12)."""
        present = 0
        for ds in self.DATASET_KEYS:
            for seed in self.SEEDS:
                for lineage in self.LINEAGES:
                    folds = list(range(5)) + ([-1] if lineage == "N" else [])
                    for fold in folds:
                        z = self.load_mint(ds, lineage, seed, fold)
                        meta = json.loads(str(z["meta"]))
                        par = meta.get("fold_parity_vs_banked_vsw_ckpt")
                        if not (isinstance(par, list) and len(par) == 5 and all(par)):
                            raise GateFailure("GATE-FOLD",
                                              "parity flag bad in {}".format(
                                                  self.mint_path(ds, lineage, seed, fold)))
                        if (np.asarray(z["fold_of"]) < 0).any():
                            raise GateFailure("GATE-FOLD", "fold_of incomplete")
                        present += 1
        if present != 66:
            raise GateFailure("GATE-LEDGER",
                              "mints_present_before_arena = {} != 66".format(present))
        self.gates["GATE-FOLD"] = "PASS"
        self.reports["mints_present_before_arena"] = present
        self.hb("GATE-FOLD", present, 66, "banked parity flags re-read")

    def load_ro(self, ds):
        """The two train-split ro caches.  Layer-2 split guard: the split is a literal
        'train' and no other value is reachable here (§12).

        CODE-R2 M-1: this calls `torch.load` DIRECTLY -- it does not route through
        headspace_mint.load_split.  Layer 1 still covers it, because load_frozen() imports
        headspace_mint, which installs its module-level guarded torch.load
        (headspace_mint.py:106-116) process-wide; load_frozen() runs before any load_ro call
        on both the full path and --dry-parity-only.  That import ordering is what makes the
        direct call safe and is stated here rather than assumed.

        CODE-R1 M-12: memoised per dataset.  Uncached, the three call sites executed six
        load events against §8 Phase 1c's one for the arena process; memoising makes the
        repeats free, so the code matches the design's count instead of drifting from it.
        """
        if ds in self.ro_cache:
            return self.ro_cache[ds]
        import torch
        cfg = self.PV.DATASETS[ds]
        out = {}
        for policy, suffix in (("standard", "ro_L24"), ("oneword", "ro_ow_L24")):
            path = os.path.join(cfg["cache_dir"], "train_{}-{}.pt".format(
                cfg["model"], suffix))
            assert os.path.basename(path).startswith("train_"), "SPLIT GUARD"
            d = torch.load(path, map_location="cpu", weights_only=False)
            ids = d["ids"]
            if isinstance(ids, list) and len(ids) == 1 and isinstance(ids[0], list):
                ids = ids[0]
            out[policy] = {
                "ids": list(ids),
                "img": d["img_feats"].float().numpy().astype("float64"),
                "text": d["text_feats"].float().numpy().astype("float64"),
                "labels": np.asarray(d["labels"]).astype(int),
            }
        self.ro_cache[ds] = out
        return out

    def gate_idparity_zeromask_nullremoved(self, ds, ro, native_lab):
        """GATE-IDPARITY, GATE-ZEROMASK (feature space only) and GATE-NULLREMOVED."""
        s, o = ro["standard"], ro["oneword"]
        if s["ids"] != o["ids"]:
            raise GateFailure("GATE-IDPARITY", "{} ro ids order differs".format(ds))
        if not np.array_equal(s["labels"], o["labels"]):
            raise GateFailure("GATE-IDPARITY", "{} ro labels differ".format(ds))
        if not np.array_equal(s["labels"], np.asarray(native_lab).astype(int)):
            raise GateFailure("GATE-IDPARITY",
                              "{} ro labels differ from the native bank".format(ds))
        zero = np.flatnonzero(np.all(s["img"] == 0, axis=1)
                              & np.all(s["text"] == 0, axis=1)
                              & np.all(o["img"] == 0, axis=1)
                              & np.all(o["text"] == 0, axis=1))
        want = list(self.cfg["population_derived_constants"][ds]["removed_zero_rows"])
        if sorted(zero.tolist()) != sorted(want):
            raise GateFailure("GATE-ZEROMASK",
                              "{} exact-zero rows {} != {}".format(ds, zero.tolist(), want))
        return zero

    # -------------------------------------------------------- raw leg (§3.6)
    def raw_leg(self, ds, ro, keep):
        """GATE-C01PARITY, GATE-ROWSUBSET and rho_raw, on the same rows as the head leg."""
        full_n = len(ro["standard"]["img"])
        zero = np.flatnonzero(np.all(ro["standard"]["img"] == 0, axis=1)
                              & np.all(ro["standard"]["text"] == 0, axis=1))
        mask_full = np.zeros(full_n, dtype=bool)
        mask_full[zero] = True

        std_full = [ro["standard"]["img"], ro["standard"]["text"]]
        ow_full = [ro["oneword"]["img"], ro["oneword"]["text"]]

        # --- GATE-C01PARITY: ONE predicate, max|diff| == 0.0 (§6) --------------------
        t0 = time.time()
        mine = self.builder.build_views(std_full, ow_full,
                                        "c06/parity/{}".format(ds), mask_full)
        theirs, _, _, _, _, _ = self.C01.prepare_views(
            {"img": ro["standard"]["img"], "text": ro["standard"]["text"]},
            {"img": ro["oneword"]["img"], "text": ro["oneword"]["text"]},
            self.c01cfg, "c01/parity/{}".format(ds), mask_full)
        worst = 0.0
        for arm in self.key_arms:
            if arm not in theirs:
                raise GateFailure("GATE-C01PARITY", "arm {} absent from C01".format(arm))
            worst = max(worst, float(np.max(np.abs(mine[arm] - theirs[arm]))))
        if worst != 0.0:
            raise GateFailure("GATE-C01PARITY",
                              "{} max|diff| = {:.3e} (bit-exactness required)".format(
                                  ds, worst))
        # CODE-R2 C-1: GATE-COMPLETENESS requires all twenty declared names on the verdict
        # face.  This gate reported only through the heartbeat, so a clean run HALTed at the
        # finish line.  Reporting wiring only -- the predicate above is unchanged.  The gate
        # runs once per dataset, so the face carries a per-dataset value.
        self.gates.setdefault("GATE-C01PARITY", {})[ds] = "PASS max|diff|=0.0"
        self.hb("GATE-C01PARITY", 1, 1, "{} max|diff|=0.0 in {:.1f}s".format(
            ds, time.time() - t0))

        # --- GATE-ROWSUBSET: HateMM only (§3.7) --------------------------------------
        if len(zero):
            sub_mask = np.zeros(int(keep.sum()), dtype=bool)
            arms_sub = self.builder.build_views(
                [b[keep] for b in std_full], [b[keep] for b in ow_full],
                "c06/rowsubset/{}".format(ds), sub_mask)
            worst_sub = 0.0
            for arm in self.key_arms:
                worst_sub = max(worst_sub,
                                float(np.max(np.abs(arms_sub[arm] - mine[arm][keep]))))
            if worst_sub != 0.0:
                raise GateFailure("GATE-ROWSUBSET",
                                  "{} bridge max|diff| = {:.3e}".format(ds, worst_sub))
            self.gates["GATE-ROWSUBSET"] = "PASS"
            self.hb("GATE-ROWSUBSET", 1, 1, "{} bit-exact bridge".format(ds))
            raw_arena = arms_sub
        else:
            raw_arena = {k: v for k, v in mine.items()}
            self.gates.setdefault("GATE-ROWSUBSET", "N/A (no null row on this dataset)")

        # --- GATE-RHORAW (global): 26 frozen values at 4 dp (§6.1) --------------------
        rho_raw = {}
        for arm in self.key_arms:
            rho_raw[arm] = rho_of(raw_arena[arm])
            want = self.cfg["rho_raw_frozen_6dp"][arm][ds]
            if round(rho_raw[arm], 4) != round(float(want), 4):
                raise GateFailure("GATE-RHORAW",
                                  "{} {} rho_raw {:.6f} != frozen {:.6f} at 4 dp".format(
                                      ds, arm, rho_raw[arm], want))
        # CODE-R2 C-1: same -- reported only through the heartbeat until now.
        self.gates.setdefault("GATE-RHORAW", {})[ds] = "PASS 13 arms at 4 dp"
        self.hb("GATE-RHORAW", 13, 13, "{} 13 arms at 4 dp".format(ds))
        return raw_arena, rho_raw

    # -------------------------------------------------------- head leg
    def head_cell_arms(self, ds, lineage, seed, fold, keep):
        """The 13 head-space arms + 2 guard arms for one (dataset, lineage, seed, fold),
        built from that fold's OWN head (§13.1 item 13: no arm built under head f is ever
        voted for a query outside fold f's held-out fifth)."""
        z = self.load_mint(ds, lineage, seed, fold)
        h_std = np.asarray(z["h_std"], dtype="float64")[keep]
        h_ow = np.asarray(z["h_ow"], dtype="float64")[keep]
        mask = np.zeros(len(h_std), dtype=bool)
        views = self.builder.build_views(
            [h_std], [h_ow], "c06/head/{}/{}/s{}/f{}".format(ds, lineage, seed, fold),
            mask)
        d_i = np.linalg.norm(
            self.builder.l2(h_ow, "c06/s7/ow", mask).astype("float64")
            - self.builder.l2(h_std, "c06/s7/std", mask).astype("float64"), axis=1)
        return views, d_i


    PER_LINEAGE_GATES = ("GATE-ARENA", "GATE-ORBITDISP", "GATE-NESTED",
                         "GATE-SELFTEST", "GATE-ZEROOP", "GATE-ALGEBRA")

    def _record_per_lineage_gates(self, ds, lineage, drop_reasons):
        """CODE-R2 C-1: put every per-lineage gate on the verdict face.

        Five of the six recorded their result only by appending to `drop_reasons`, so
        GATE-COMPLETENESS -- which requires all twenty declared names in `self.gates` --
        HALTed on EVERY clean run, after S1-S7 and both Holm families had already been
        computed.  This is reporting wiring only: no gate's predicate, threshold or drop
        semantics is touched, and a reason that was appended is still appended.

        A reason is attributed to a gate by its own "GATE-NAME" prefix, which is how every
        append in this file is already formatted.  The value is keyed by (dataset, lineage)
        because these gates are evaluated once per lineage per dataset.
        """
        key = "{}/{}".format(ds, lineage)
        status = {}
        for name in self.PER_LINEAGE_GATES:
            hits = [r for r in drop_reasons if r.split(":")[0].split(" ")[0] == name]
            status[name] = ("PASS" if not hits
                            else "FAILED (lineage dropped): " + "; ".join(hits))
            self.gates.setdefault(name, {})[key] = status[name]
        return status

    # -------------------------------------------------------- GATE-POP (I-5)
    def gate_pop(self, ds, ro, keep, raw_keep=None):
        """§13: GATE-POP runs BEFORE any population-consuming gate (§13.1 item 11), and
        asserts row identity by INDEX SET between the head leg and the raw leg.

        CODE-R1 I-5: this was previously inside run_lineage, i.e. after GATE-C01PARITY,
        GATE-ROWSUBSET, GATE-RHORAW, GATE-NULLREMOVED and GATE-FLOOR had already consumed
        the realised population, and the index-set limb was never asserted at all.
        """
        pdc = self.cfg["population_derived_constants"][ds]
        lab = np.asarray(ro["standard"]["labels"]).astype(int)[keep]
        n_arena = int(keep.sum())
        pos, neg = int((lab == 1).sum()), int((lab == 0).sum())
        maj = max(pos, neg) / float(n_arena)
        if n_arena != int(pdc["arena_n"]) or pos != int(pdc["class_counts"]["pos"]) \
                or neg != int(pdc["class_counts"]["neg"]):
            raise GateFailure("GATE-POP", "{} realised population ({}, {}, {}) != frozen"
                              .format(ds, n_arena, pos, neg))
        if round(maj, 4) != float(pdc["arena_majority"]):
            raise GateFailure("GATE-POP", "{} majority {:.6f} != frozen {}".format(
                ds, maj, pdc["arena_majority"]))
        band_lo = round(maj + float(self.c01cfg["decision"][
            "minimum_gain_over_strongest_control"]), 4)
        if band_lo != float(pdc["gate_arena_band"][0]):
            raise GateFailure("GATE-POP", "{} arena band {} != frozen {}".format(
                ds, band_lo, pdc["gate_arena_band"][0]))
        tie_cap = int(math.floor(0.01 * n_arena))
        if tie_cap != int(pdc["tie_cap"]):
            raise GateFailure("GATE-POP", "{} tie cap {} != frozen {}".format(
                ds, tie_cap, pdc["tie_cap"]))
        full_maj = max(int((np.asarray(ro["standard"]["labels"]).astype(int) == 1).sum()),
                       int((np.asarray(ro["standard"]["labels"]).astype(int) == 0).sum())
                       ) / float(len(keep))
        if round(full_maj, 4) != float(pdc["full_majority"]):
            raise GateFailure("GATE-POP", "{} full-population majority {:.6f} != frozen {}"
                              .format(ds, full_maj, pdc["full_majority"]))
        # index-set identity between the two legs (§6's gate row, §13.1 item 11)
        if raw_keep is not None:
            if set(np.flatnonzero(keep).tolist()) != set(np.flatnonzero(raw_keep).tolist()):
                raise GateFailure("GATE-POP",
                                  "{} head-leg and raw-leg row index sets differ".format(ds))
        self.gates["GATE-POP"] = "PASS"
        return {"n_arena": n_arena, "pos": pos, "neg": neg, "majority": maj,
                "full_majority": full_maj, "tie_cap": tie_cap,
                "band": [band_lo, UPPER_ARENA_BAR]}

    # -------------------------------------------------------- assemble one lineage
    def run_lineage(self, ds, lineage, ro, keep, rho_raw):
        """Build, vote and assemble every OOF quantity for one (dataset, lineage) cell."""
        pdc = self.cfg["population_derived_constants"][ds]
        n_arena = int(keep.sum())
        lab_full = np.asarray(ro["standard"]["labels"]).astype(int)
        lab = lab_full[keep]
        arena_ids = [i for i, k in enumerate(keep) if k]

        # GATE-POP already ran, standalone and before every consumer (I-5).  Its computed
        # values are reused here; nothing re-reads them from the config.
        popv = self.pop[ds]
        maj, tie_cap = popv["majority"], popv["tie_cap"]

        skf_fold_of = np.asarray(self.load_mint(ds, lineage, 0, 0)["fold_of"]).astype(int)
        fold_arena = skf_fold_of[keep]

        drop_early = []          # CODE-R1 H-1: per-lineage failures found before §5.6's
                                 # drop list exists; merged into `drop` below.
        tail = {}                # I-3 / §13.1 item 25: per-cell displacement tail
        preds, votes, dis = {}, {}, {}
        guard_sim, guard_lab, guard_idx = {}, {}, {}
        rho_head_max = {}
        algebra_res = 0.0
        for seed in self.SEEDS:
            for arm in self.key_arms + ["guard_orthrot_0", "guard_orthrot_45"]:
                preds.setdefault(arm, {})[seed] = np.full(n_arena, -1, dtype=int)
                votes.setdefault(arm, {})[seed] = np.full(n_arena, np.nan)
            dis[seed] = np.full(n_arena, np.nan)
            for arm in ("guard_orthrot_0", "endpoint_concat",
                        "guard_orthrot_45", "common_displacement"):
                # CODE-R1 I-4: SS6.5 ranks on the UNION of the two arms' top-21 sets, so
                # each arm is retrieved to depth 2*window.  A union member outside an arm's
                # top-2*window is outranked by >= 2*window - 1 items in that arm and carries
                # weight 0 in every admissible ordering, so it cannot move the bound.
                guard_sim.setdefault(arm, {})[seed] = np.full(
                    (n_arena, 2 * TIE_RANK_WINDOW), np.nan)
                guard_lab.setdefault(arm, {})[seed] = np.full(
                    (n_arena, 2 * TIE_RANK_WINDOW), np.nan)
                guard_idx.setdefault(arm, {})[seed] = np.full(
                    (n_arena, 2 * TIE_RANK_WINDOW), -1, dtype=int)

            for fold in range(5):
                views, d_i = self.head_cell_arms(ds, lineage, seed, fold, keep)
                ho = np.flatnonzero(fold_arena == fold)
                fit = np.flatnonzero(fold_arena != fold)
                if len(ho) == 0 or len(fit) == 0:
                    # CODE-R1 H-1: GATE-NESTED is PER-LINEAGE (§5.6); a failure drops the
                    # lineage on both datasets, it does not HALT the battery.
                    drop_early.append("GATE-NESTED: empty fold {}".format(fold))
                    continue
                dis[seed][ho] = d_i[ho]
                # I-3 / §13.1 item 25: every cell records its own min_i d_i and
                # frac(d_i <= 0.001) at run time, so tiny_ok's non-carriage (§5.2.3) rests
                # on measurement in all 60 cells rather than the four §7.8 reports.
                dv = d_i[ho]
                tail["s{}f{}".format(seed, fold)] = {
                    "min_d_i": float(dv.min()), "median_d_i": float(np.median(dv)),
                    "max_d_i": float(dv.max()),
                    "frac_le_tiny_eps": float((dv <= float(
                        self.c01cfg["transforms"]["tiny_displacement_epsilon"])).mean())}
                for arm, X in views.items():
                    v, p, _, _ = vote(self.MECH, X[fit], lab[fit], X[ho])
                    preds[arm][seed][ho] = p
                    votes[arm][seed][ho] = v
                    if arm in guard_sim:
                        depth = min(int(fit.sum()) if fit.dtype == bool else len(fit),
                                    2 * TIE_RANK_WINDOW)
                        _, _, gi, gs = self.MECH.deployed_vote(
                            X[fit], lab[fit], X[ho], topk=depth)
                        guard_sim[arm][seed][ho, :depth] = gs
                        guard_lab[arm][seed][ho, :depth] = lab[fit][gi]
                        guard_idx[arm][seed][ho, :depth] = gi
                    # GATE-ORBITDISP is per fold, all 60 head cells, all 13 arms (§6.1)
                    if arm in self.key_arms:
                        r = rho_of(X)
                        rho_head_max[arm] = max(rho_head_max.get(arm, 0.0), r)
                # GATE-ALGEBRA: key-level residual on both identities (§6.5)
                for a, b in (("guard_orthrot_0", "endpoint_concat"),
                             ("guard_orthrot_45", "common_displacement")):
                    algebra_res = max(algebra_res, float(np.max(np.abs(
                        views[a].astype("float64") - views[b].astype("float64")))))
                self.hb("ARMS", fold + 1, 5,
                        "{} {} s{} f{}".format(ds, lineage, seed, fold))

            # avg_score: mean of the two endpoint vote SCORES (§3.5)
            avg = 0.5 * (votes["endpoint_std"][seed] + votes["endpoint_ow"][seed])
            votes.setdefault("avg_score", {})[seed] = avg
            preds.setdefault("avg_score", {})[seed] = (avg >= 0.0).astype(int)

        # ---- GATE-NESTED: per item, the head that scored it excluded its fold -------
        checked = 0
        for seed in self.SEEDS:
            for arm in self.all_arms:
                if (preds[arm][seed] < 0).any():
                    drop_early.append("GATE-NESTED: {} {} unscored items".format(
                        arm, lineage))
                # M-5: an INDEPENDENT count -- items actually assigned a prediction by a
                # fold's head -- not the tautological accumulator the review found.
                checked += int((preds[arm][seed] >= 0).sum())
        if checked != n_arena * len(self.all_arms) * len(self.SEEDS):
            drop_early.append("GATE-NESTED: scored {} != item count {}".format(
                checked, n_arena * len(self.all_arms) * len(self.SEEDS)))

        # ---- metrics ---------------------------------------------------------------
        acc_s, mf1_s, correct = {}, {}, {}
        for arm in self.all_arms:
            acc_s[arm] = {s: accuracy(lab, preds[arm][s]) for s in self.SEEDS}
            mf1_s[arm] = {s: macro_f1(self.MECH, lab, preds[arm][s]) for s in self.SEEDS}
            correct[arm] = np.mean([(preds[arm][s] == lab).astype("float64")
                                    for s in self.SEEDS], axis=0)
            finite(arm, list(acc_s[arm].values()), list(mf1_s[arm].values()))
        acc = {a: float(np.mean(list(acc_s[a].values()))) for a in self.all_arms}
        mf1 = {a: float(np.mean(list(mf1_s[a].values()))) for a in self.all_arms}

        # ---- reference arm, D-1 (§5.2.1) -------------------------------------------
        evaluations = {a: {"metrics": {"accuracy": acc[a], "macro_f1": mf1[a]}}
                       for a in self.all_arms}
        reference = self.C01.select_strongest_ordinary_control(evaluations,
                                                              self.gain_controls)

        # ---- GATE-SELFTEST: net_s(A) = n_D*(acc_s(A) - acc_s(reference)), all 14 arms
        net_s = {}
        for arm in self.all_arms:
            net_s[arm] = {}
            for s in self.SEEDS:
                a_ok = (preds[arm][s] == lab)
                r_ok = (preds[reference][s] == lab)
                net = int(np.sum(a_ok & ~r_ok) - np.sum(~a_ok & r_ok))
                identity = n_arena * (acc_s[arm][s] - acc_s[reference][s])
                if abs(identity - net) > 1e-6:
                    # CODE-R1 H-1: GATE-SELFTEST is PER-LINEAGE (§5.6).
                    drop_early.append(
                        "GATE-SELFTEST: {} {} s{} net {} != n_D*(dacc) {:.6f}".format(
                            ds, arm, s, net, identity))
                finite("net_s", net)                                  # CODE-R1 M-7
                net_s[arm][s] = net
        # CODE-R2 C-1: the six per-lineage gates are evaluated ONCE PER LINEAGE PER DATASET.
        # The face therefore carries a per-(dataset, lineage) value rather than a single
        # string -- that is the aggregation rule the seven blank names had concealed.
        # GATE-SELFTEST is recorded by _record_per_lineage_gates with the other five.

        # ---- per-lineage gates ------------------------------------------------------
        drop = list(drop_early)
        lo, hi = float(pdc["gate_arena_band"][0]), UPPER_ARENA_BAR
        if acc["endpoint_std"] < lo:
            drop.append("GATE-ARENA lower: endpoint_std {:.4f} < {}".format(
                acc["endpoint_std"], lo))
        for arm in ("endpoint_std", "displacement", "common_displacement"):
            if acc[arm] > hi:
                drop.append("GATE-ARENA upper: {} {:.4f} > {}".format(arm, acc[arm], hi))
        rho_star = float(self.cfg["gates"]["GATE-ORBITDISP"]["rho_star"][ds])
        for arm in self.key_arms:
            if rho_head_max[arm] > rho_star and rho_raw[arm] <= rho_star:
                drop.append("GATE-ORBITDISP: {} rho_head {:.6f} > {:.6f}".format(
                    arm, rho_head_max[arm], rho_star))
        if algebra_res > GATE_ALGEBRA_BAR:
            drop.append("GATE-ALGEBRA: residual {:.3e} > {:.1e}".format(
                algebra_res, GATE_ALGEBRA_BAR))

        # ---- GATE-ZEROOP with the §6.5 tie diagnostic -------------------------------
        # §6.5's unit correction: a key perturbation with max|dk| = eps moves an inner
        # product against a unit query by up to ||dk||_2 <= sqrt(d)*eps, sqrt(2048) = 45.25.
        window = float(algebra_res) * math.sqrt(
            float(self.cfg["arena"]["head_space_arm_dims"]["paired_block"]))
        zeroop = {"per_seed_mismatch": {}, "per_seed_casualties": {},
                  "residual": algebra_res, "similarity_window": window,
                  "cap": tie_cap}
        for seed in self.SEEDS:
            mismatch_total, casualty_total = 0, 0
            for a, b in (("guard_orthrot_0", "endpoint_concat"),
                         ("guard_orthrot_45", "common_displacement")):
                mm = preds[a][seed] != preds[b][seed]
                if mm.any():
                    cas = tie_casualties(guard_sim[a][seed], guard_lab[a][seed],
                                         guard_idx[a][seed],
                                         guard_sim[b][seed], guard_lab[b][seed],
                                         guard_idx[b][seed],
                                         mm, window, TIE_RANK_WINDOW)
                    outside = int(np.sum(mm & ~cas))
                    if outside:
                        drop.append("GATE-ZEROOP: {} vs {} {} mismatch(es) outside the "
                                    "tie set".format(a, b, outside))
                    casualty_total += int(cas.sum())
                mismatch_total += int(mm.sum())
            zeroop["per_seed_mismatch"][seed] = mismatch_total
            zeroop["per_seed_casualties"][seed] = casualty_total
            if mismatch_total > tie_cap:
                drop.append("GATE-ZEROOP: seed {} mismatches {} > cap {}".format(
                    seed, mismatch_total, tie_cap))

        return {"n": n_arena, "lab": lab, "arena_ids": arena_ids,
                "fold": fold_arena, "preds": preds, "votes": votes,
                "acc": acc, "mf1": mf1, "acc_s": acc_s, "mf1_s": mf1_s,
                "correct": correct, "net_s": net_s, "reference": reference,
                # CODE-R1 M-6: recorded from the SELECTOR's own return so SS13.1 item 24's
                # assertion compares two independently-carried values rather than a value
                # with itself.
                "selected_control": self.C01.select_strongest_ordinary_control(
                    evaluations, self.gain_controls),
                "displacement_norm": dis, "rho_head_max": rho_head_max,
                "displacement_tail": tail,
                "algebra_residual": algebra_res, "zeroop": zeroop,
                "drop_reasons": drop, "tie_cap": tie_cap,
                "per_lineage_gate_status": self._record_per_lineage_gates(
                    ds, lineage, drop),
                "majority": maj, "band": [lo, hi]}

    # -------------------------------------------------------- GATE-FLOOR (global)
    def gate_floor(self, ds, lab_full):
        want_acc = self.cfg["gates"]["GATE-FLOOR"]["acc_deployed"][ds]
        want_mf1 = self.cfg["gates"]["GATE-FLOOR"]["mf1_deployed"][ds]
        for si, seed in enumerate(self.SEEDS):
            banked = json.load(open(os.path.join(
                REPO, "scripts/analysis",
                "headspace_arena_{}_s{}_OUT.json".format(ds, seed))))["result"]
            fold_of = np.asarray(self.load_mint(ds, "N", seed, 0)["fold_of"]).astype(int)
            n = len(lab_full)
            pred = np.full(n, -1, dtype=int)
            per_fold = []
            for fold in range(5):
                X = np.asarray(self.load_mint(ds, "N", seed, fold)["K_train"],
                               dtype="float64")
                ho = np.flatnonzero(fold_of == fold)
                fit = np.flatnonzero(fold_of != fold)
                _, p, _, _ = vote(self.MECH, X[fit], lab_full[fit], X[ho])
                pred[ho] = p
                per_fold.append(round(accuracy(lab_full[ho], p), 4))
            got_acc = round(accuracy(lab_full, pred), 4)
            got_mf1 = round(macro_f1(self.MECH, lab_full, pred), 4)
            if got_acc != round(float(want_acc[si]), 4):
                raise GateFailure("GATE-FLOOR", "{} s{} acc {:.4f} != {}".format(
                    ds, seed, got_acc, want_acc[si]))
            if got_mf1 != round(float(want_mf1[si]), 4):
                raise GateFailure("GATE-FLOOR", "{} s{} mF1 {:.4f} != {}".format(
                    ds, seed, got_mf1, want_mf1[si]))
            if per_fold != [round(float(x), 4) for x in banked["fold_acc_deployed"]]:
                raise GateFailure("GATE-FLOOR",
                                  "{} s{} fold_acc_deployed {} != banked {}".format(
                                      ds, seed, per_fold, banked["fold_acc_deployed"]))
            self.hb("GATE-FLOOR", si + 1, 3, "{} s{} acc {:.4f} mF1 {:.4f}".format(
                ds, seed, got_acc, got_mf1))
        self.gates["GATE-FLOOR"] = "PASS"

    # -------------------------------------------------------- S4 / S5 statistics
    def s4_family(self, ds, cells):
        """The frozen 92-hypothesis Holm family per dataset (§5.5).  A dropped lineage's
        hypotheses are recorded NOT_TESTED with p = 1; the family is NEVER shrunk."""
        rng = np.random.default_rng(int(self.c01cfg["statistics"]["seed"]))
        B = int(self.c01cfg["statistics"]["n_bootstrap"])
        lower_q = float(self.c01cfg["statistics"]["bootstrap_lower_quantile"])
        entries, detail = [], {}
        draws = None
        mf1_cache = {}          # (lineage, arm) -> (B,) seed-mean macro-F1 per resample

        def mf1_vector(lineage, cell, arm):
            """ERRATUM 1: the macro-F1 leg's per-arm precompute.

            §5.4 shares the draw indices across all comparators AND both lineages within a
            dataset, so an arm's resampled-metric vector is the SAME object in every comparison
            it appears in.  Computing it once per (arm, seed) and differencing runs the same
            operations in the same order as computing it per comparison -- bit-identical, and
            it is what §8 Phase 4 prices at 168 x U_mF1.
            """
            key = (lineage, arm)
            if key not in mf1_cache:
                pv = [cell["preds"][arm][s] for s in self.SEEDS]
                vec = resampled_macro_f1(pv, cell["lab"], draws)
                # M-9: turn resampled_macro_f1's docstring claim into a run-time fact.
                # 32 sampled draws against direct mechfix_ops.macro_f1 calls; any non-zero
                # difference HALTs.  Costs milliseconds and closes the
                # re-implementation-without-anchor class.
                spot = np.linspace(0, len(draws) - 1, 32).astype(int)
                for b in spot:
                    ref = float(np.mean([self.MECH.macro_f1(cell["lab"][draws[b]],
                                                           p[draws[b]]) for p in pv]))
                    if vec[b] != ref:
                        raise GateFailure(
                            "S4-MF1-FIDELITY",
                            "vectorised macro-F1 diverged from mechfix_ops.macro_f1 at "
                            "draw {}: {!r} vs {!r}".format(int(b), vec[b], ref))
                mf1_cache[key] = vec
            return mf1_cache[key]

        for lineage in self.LINEAGES:
            cell = cells.get(lineage)
            for real, comparators in self.cfg["arms"]["comparators"].items():
                family = list(comparators) + self.cfg["arms"]["rotation_family"]
                for comp in family:
                    for metric in self.c01cfg["statistics"]["holm_metrics"]:
                        hid = (lineage, real, comp, metric)
                        if cell is None:
                            summary = {"one_sided_raw_p": 1.0, "lower": None,
                                       "status": "NOT_TESTED"}
                        else:
                            if draws is None:
                                draws = rng.integers(0, cell["n"], size=(B, cell["n"]))
                            if metric == "accuracy":
                                # §5.4's frozen expression, RETAINED VERBATIM (obligation 1)
                                d = bootstrap_deltas(cell["correct"][real],
                                                     cell["correct"][comp], draws)
                            else:
                                # §5.4's ADDED macro-F1 bullet: C01's recompute-per-resample
                                # form via mechfix_ops.macro_f1, seed mean inside (ERRATUM 1)
                                d = (mf1_vector(lineage, cell, real)
                                     - mf1_vector(lineage, cell, comp))
                            summary = {"one_sided_raw_p": one_sided_p(d),
                                       "lower": float(np.quantile(d, lower_q)),
                                       "status": "TESTED"}
                            # CODE-R1 M-7: DECISION quantities, not only gate quantities
                            finite("S4 lower/one_sided_raw_p",
                                   summary["lower"], summary["one_sided_raw_p"])
                        entries.append((hid, summary))
                        detail[str(hid)] = summary
        if len(entries) != 92:
            raise GateFailure("S4-FAMILY", "family size {} != 92".format(len(entries)))
        self.C01.holm_adjust(entries, float(self.c01cfg["statistics"]["holm_alpha"]))
        return {str(h): s for h, s in entries}

    def s5_null(self, ds, lineage, ro, keep, cell):
        """§5.4.1: 256 id-hash draws; permute the one-word endpoint rows, rebuild the two
        real arms in head space, seed-mean OOF accuracy and macro-F1, p95 and p."""
        n_draw = int(self.c01cfg["statistics"]["n_id_hash_permutations"])
        ids = [ro["standard"]["ids"][i] for i in cell["arena_ids"]]
        lab, fold_arena = cell["lab"], cell["fold"]
        real = list(self.cfg["arms"]["real"])
        nulls = {a: {"accuracy": [], "macro_f1": []} for a in real}
        cache = {}
        for seed in self.SEEDS:
            for fold in range(5):
                z = self.load_mint(ds, lineage, seed, fold)
                cache[(seed, fold)] = (
                    np.asarray(z["h_std"], dtype="float64")[keep],
                    np.asarray(z["h_ow"], dtype="float64")[keep])
        mask = np.zeros(cell["n"], dtype=bool)
        for d in range(n_draw):
            order = self.C01.id_hash_permutation(
                ids, ds, "train", d, int(self.c01cfg["statistics"]["seed"]),
                fixed_indices=())
            per_seed = {a: {"accuracy": [], "macro_f1": []} for a in real}
            for seed in self.SEEDS:
                pred = {a: np.full(cell["n"], -1, dtype=int) for a in real}
                for fold in range(5):
                    h_std, h_ow = cache[(seed, fold)]
                    # CODE-R1 H-4: build ONLY the two arms S5 reads.  §8's U4 prices a
                    # two-arm build; building all fifteen cost 10.8x and would have made
                    # the realised total ~2.4x the conservative bound, tripping §8's own
                    # overrun clause.  Same construction, fewer emissions (see build_views).
                    views = self.builder.build_views(
                        [h_std], [h_ow[order]],
                        "c06/null/{}/{}/s{}/f{}/d{}".format(ds, lineage, seed, fold, d),
                        mask, only=real)
                    ho = np.flatnonzero(fold_arena == fold)
                    fit = np.flatnonzero(fold_arena != fold)
                    for a in real:
                        _, p, _, _ = vote(self.MECH, views[a][fit], lab[fit],
                                          views[a][ho])
                        pred[a][ho] = p
                for a in real:
                    per_seed[a]["accuracy"].append(accuracy(lab, pred[a]))
                    per_seed[a]["macro_f1"].append(macro_f1(self.MECH, lab, pred[a]))
            for a in real:
                nulls[a]["accuracy"].append(float(np.mean(per_seed[a]["accuracy"])))
                nulls[a]["macro_f1"].append(float(np.mean(per_seed[a]["macro_f1"])))
            if (d + 1) % 32 == 0:
                self.hb("S5-NULL", d + 1, n_draw, "{} {}".format(ds, lineage))

        upper_q = float(self.c01cfg["statistics"]["bootstrap_upper_quantile"])
        entries, out = [], {}
        for a in real:
            for metric in self.c01cfg["statistics"]["holm_metrics"]:
                obs = cell["acc"][a] if metric == "accuracy" else cell["mf1"][a]
                arr = np.asarray(nulls[a][metric], dtype="float64")
                summary = {"observed": obs, "p95": float(np.quantile(arr, upper_q)),
                           "above_p95": bool(obs > float(np.quantile(arr, upper_q))),
                           "one_sided_raw_p": null_p(obs, arr)}
                # CODE-R1 M-7
                finite("S5 observed/p95/one_sided_raw_p", summary["observed"],
                       summary["p95"], summary["one_sided_raw_p"])
                entries.append(((a, metric), summary))
                out[a + "|" + metric] = summary
        if len(entries) != 4:
            raise GateFailure("S5-FAMILY", "family size {} != 4".format(len(entries)))
        self.C01.holm_adjust(entries, float(self.c01cfg["statistics"]["holm_alpha"]))
        return out

    # -------------------------------------------------------- S1-S7
    def evaluate_conditions(self, ds, lineage, cell, s4, s5):
        # CODE-R1 M-6 / SS13.1 item 24: C01's :2724-equivalent consistency assertion --
        # S7's reference must BE the selected control.  The property held by construction
        # (run_lineage passes the same value to S6, S7 and GATE-SELFTEST); item 24 requires
        # the assertion to exist, so it is asserted rather than relied upon.
        if cell.get("reference") != cell.get("selected_control", cell.get("reference")):
            raise GateFailure("S7-CONSISTENCY",
                              "{} {} S7 reference {} != selected control {}".format(
                                  ds, lineage, cell.get("reference"),
                                  cell.get("selected_control")))
        pdc = self.cfg["population_derived_constants"][ds]
        gain = float(self.c01cfg["decision"]["minimum_gain_over_strongest_control"])
        min_net = int(self.c01cfg["decision"]["minimum_net_fixes"][
            {"hatemm": "HateMM", "zh": "MHC_zh"}[ds]])
        rot = self.cfg["arms"]["rotation_family"]
        q = float(self.c01cfg["transforms"]["small_displacement_train_quantile"])
        dom = float(self.c01cfg["transforms"]["max_small_displacement_fix_fraction"])
        out = {}
        for A in self.cfg["arms"]["real"]:
            C = self.cfg["arms"]["comparators"][A]
            s1 = (cell["acc"][A] > max(cell["acc"][r] for r in rot)
                  and cell["mf1"][A] > max(cell["mf1"][r] for r in rot))
            s2 = all(cell["acc_s"][A][s] > max(cell["acc_s"][r][s] for r in rot)
                     for s in self.SEEDS)
            s3 = (cell["acc"][A] - max(cell["acc"][c] for c in C) >= gain
                  and cell["mf1"][A] - max(cell["mf1"][c] for c in C) >= gain)
            s4_ok = True
            for comp in list(C) + rot:
                for metric in self.c01cfg["statistics"]["holm_metrics"]:
                    key = str((lineage, A, comp, metric))
                    rec = s4.get(key)
                    if rec is None or rec.get("status") != "TESTED":
                        s4_ok = False
                    elif not (rec.get("lower") is not None and rec["lower"] > 0.0
                              and rec.get("holm_reject")):
                        s4_ok = False
            s5_ok = all(s5[a + "|" + m]["above_p95"] and s5[a + "|" + m]["holm_reject"]
                        for a in self.cfg["arms"]["real"]
                        for m in self.c01cfg["statistics"]["holm_metrics"])
            s6 = all(cell["net_s"][A][s] >= min_net for s in self.SEEDS)
            # S7: common_displacement only (§5.2.2)
            if A == "common_displacement":
                s7 = True
                s7_detail = {}
                for s in self.SEEDS:
                    d = cell["displacement_norm"][s]
                    thr = float(np.quantile(d, q))
                    small = d <= thr                     # frozen operator '<=' (§5.2.2)
                    a_ok = cell["preds"][A][s] == cell["lab"]
                    r_ok = cell["preds"][cell["reference"]][s] == cell["lab"]
                    fixed = a_ok & ~r_ok
                    nfix = int(fixed.sum())
                    frac = 0.0 if nfix == 0 else float((fixed & small).sum()) / nfix
                    dominated = bool(nfix > 0 and frac > dom)
                    s7_detail[s] = {"n_fixed": nfix, "fixed_fraction": frac,
                                    "dominated": dominated, "threshold": thr}
                    finite("S7 fixed_fraction/threshold", frac, thr)   # CODE-R1 M-7
                    if dominated:
                        s7 = False
            else:
                s7, s7_detail = True, "not applicable (arm scope is common_displacement)"
            out[A] = {"S1": bool(s1), "S2": bool(s2), "S3": bool(s3), "S4": bool(s4_ok),
                      "S5": bool(s5_ok), "S6": bool(s6), "S7": bool(s7),
                      "S7_detail": s7_detail,
                      "clears": bool(s1 and s2 and s3 and s4_ok and s5_ok and s6 and s7)}
        return out


# ============================================================ CLI
def build_argparser():
    ap = argparse.ArgumentParser(description="C06 $0 CPU falsifier arena")
    ap.add_argument("--config", default=os.path.join(REPO,
                                                     "configs/c06/c06_falsifier.json"))
    ap.add_argument("--mintdir", default=os.path.join(REPO,
                                                      "artifacts/c06_falsifier/mints"))
    ap.add_argument("--out", default=os.path.join(REPO,
                                                  "artifacts/c06_falsifier/C06_VERDICT.json"))
    ap.add_argument("--progress", default=os.path.join(
        REPO, "artifacts/c06_falsifier/progress/C06_PROGRESS.txt"))
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--gate-sha-only", action="store_true",
                    help="run GATE-DET1 and GATE-SHA and exit; the sbatch driver calls "
                         "this once, as the FIRST OF TWO GATE-SHA passes (§6, §13)")
    ap.add_argument("--dry-parity-only", action="store_true",
                    help="$0 dry check: GATE-C01PARITY / GATE-ROWSUBSET / GATE-RHORAW on "
                         "the raw leg only. Opens no mint, computes no arm accuracy.")
    return ap


def emit_halt(args, cfg, bat, gate, context):
    """CODE-R1 I-1: §5.6 requires the RuntimeError context in BOTH the final heartbeat line
    and the decision JSON.  Previously no verdict artifact was written on any HALT path, so
    a HALT left a line in a text file and nothing else -- no gates dict, no dropped
    lineages, no ledger, no scope block."""
    try:
        out = {"verdict": "HALT",
               "status": "INSTRUMENT_INCONCLUSIVE",
               "failing_gate": gate,
               "context": context,
               "design": {"document": cfg.get("design_document"),
                          "sha256_declared": cfg.get("design_sha256"),
                          "sha256_derived": _derived_design_sha(bat),
                          "note": DESIGN_SHA_NOTE},
               "gates": getattr(bat, "gates", {}),
               "dropped_lineages": getattr(bat, "dropped", {}),
               "ledger": getattr(bat, "ledger", {}),
               "scope": cfg.get("scope")}
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        tmp = args.out + ".tmp"
        json.dump(out, open(tmp, "w"), indent=1, default=str)
        os.replace(tmp, args.out)
    except Exception:
        pass


def main():
    args = build_argparser().parse_args()
    cfg = json.load(open(args.config))
    hb = Heartbeat(args.progress)
    bat = Battery(cfg, args, hb)
    try:
        bat.gate_det1()
        hb("GATE-DET1", 1, 1, "thread env verified")
        bat.assert_guard_active()            # H-2 / §13.1 item 28: layer 3 ACTIVE
        hb("GUARD", 1, 1, "c09_guard layer 3 active in this process")
        # ERRATUM 2 §7 (round-5 M-3): the heartbeat denominator has ONE source,
        # cfg["projected_seconds"].  Assert environment == module constant == config here,
        # in the pre-gate_sha_only block, so a drift HALTs in process 1 of 74 rather than
        # in process 74 after 66 mints have published ratios against a stale denominator.
        _cfg_proj = float(cfg["execution"]["projected_seconds"])
        _env_proj = float(os.environ.get("C06_PROJECTED_SECONDS", PROJECTED_SECONDS))
        if not (_cfg_proj == PROJECTED_SECONDS == _env_proj):
            raise GateFailure(
                "GATE-DET1",
                "projected_seconds disagree: config {} / module {} / env {}".format(
                    _cfg_proj, PROJECTED_SECONDS, _env_proj))
        hb("PROJECTION", 1, 1, "single source {} s agrees across config/module/env".format(
            _cfg_proj))
        # M-2: GATE-SHA verifies the C01 module and config digests BEFORE they are imported
        # and read.  gate_sha() needs only the config's digest tables, which load_frozen
        # does not provide, so the ordering is now sha-then-import.
        bat.gate_sha()
        bat.load_frozen()
        if args.gate_sha_only:
            hb("GATE-SHA-ONLY", 1, 1, "driver precondition satisfied")
            return 0

        if args.dry_parity_only:
            for ds in Battery.DATASET_KEYS:
                ro = bat.load_ro(ds)
                zero = bat.gate_idparity_zeromask_nullremoved(
                    ds, ro, ro["standard"]["labels"])
                keep = np.ones(len(ro["standard"]["img"]), dtype=bool)
                keep[zero] = False
                bat.raw_leg(ds, ro, keep)
                hb("DRY-PARITY", 1, 1, "{} raw leg gates pass".format(ds))
            hb("DRY-PARITY-COMPLETE", 2, 2, "no arm accuracy computed")
            return 0

        # ---- global gates that must precede every population-consuming gate --------
        bat.gate_fold_and_ledger_presence()

        cells, per_ds = {}, {}
        bat.pop, bat.keep = {}, {}
        for ds in Battery.DATASET_KEYS:
            ro = bat.load_ro(ds)
            lab_full = np.asarray(ro["standard"]["labels"]).astype(int)
            zero = bat.gate_idparity_zeromask_nullremoved(ds, ro, lab_full)
            keep = np.ones(len(lab_full), dtype=bool)
            keep[zero] = False
            bat.gates["GATE-IDPARITY"] = "PASS"
            bat.gates["GATE-ZEROMASK"] = "PASS"
            # I-5: GATE-POP runs HERE -- before raw_leg (GATE-C01PARITY / ROWSUBSET /
            # RHORAW), GATE-NULLREMOVED and GATE-FLOOR, all of which consume the
            # realised population -- and asserts index-set identity between the legs.
            bat.pop[ds] = bat.gate_pop(ds, ro, keep, raw_keep=keep)
            bat.keep[ds] = keep      # M-8: the MEASURED keep, reused by S5, never re-read
            hb("GATE-POP", 1, 1, "{} n={} maj={:.6f}".format(
                ds, bat.pop[ds]["n_arena"], bat.pop[ds]["majority"]))

            raw_arena, rho_raw = bat.raw_leg(ds, ro, keep)
            # GATE-NULLREMOVED: no arena population contains an exact-zero row
            for arm, X in raw_arena.items():
                if np.any(np.all(np.asarray(X) == 0, axis=1)):
                    raise GateFailure("GATE-NULLREMOVED",
                                      "{} {} has an exact-zero arena row".format(ds, arm))
            bat.gates["GATE-NULLREMOVED"] = "PASS"
            bat.gate_floor(ds, lab_full)

            cells[ds] = {}
            for lineage in Battery.LINEAGES:
                cell = bat.run_lineage(ds, lineage, ro, keep, rho_raw)
                cells[ds][lineage] = cell
                hb("LINEAGE", 1, 1, "{} {} reference={} drops={}".format(
                    ds, lineage, cell["reference"], len(cell["drop_reasons"])))
            per_ds[ds] = {"rho_raw": rho_raw, "majority": cells[ds]["N"]["majority"]}

        # ---- §5.6 dataset axis: a lineage failing on ANY dataset drops on BOTH -------
        passed = {}
        for lineage in Battery.LINEAGES:
            reasons = []
            for ds in Battery.DATASET_KEYS:
                reasons += ["{}: {}".format(ds, r)
                            for r in cells[ds][lineage]["drop_reasons"]]
            passed[lineage] = (len(reasons) == 0)
            if reasons:
                bat.dropped[lineage] = reasons
        hb("PER-LINEAGE-GATES", 2, 2, "passed={}".format(
            [k for k, v in passed.items() if v]))

        # ---- S4 (Holm family frozen at 92 per dataset) and S5 ------------------------
        verdicts = {}
        for ds in Battery.DATASET_KEYS:
            live = {L: cells[ds][L] for L in Battery.LINEAGES if passed[L]}
            s4 = bat.s4_family(ds, live)
            hb("S4", 92, 92, "{} Holm family frozen at 92".format(ds))
            verdicts[ds] = {"s4": s4, "lineages": {}}
            for lineage in Battery.LINEAGES:
                if not passed[lineage]:
                    verdicts[ds]["lineages"][lineage] = {"status": "INSTRUMENT_FAILED"}
                    continue
                # M-8: the keep vector is the one GATE-ZEROMASK measured, not one rebuilt
                # from the config -- §13.1 item 5a's verb is "computed from the arena".
                s5 = bat.s5_null(ds, lineage, bat.load_ro(ds), bat.keep[ds],
                                 cells[ds][lineage])
                cond = bat.evaluate_conditions(ds, lineage, cells[ds][lineage], s4, s5)
                verdicts[ds]["lineages"][lineage] = {"status": "TESTED", "s5": s5,
                                                     "conditions": cond}
                hb("S1-S7", 1, 1, "{} {} clears={}".format(
                    ds, lineage, [a for a in cond if cond[a]["clears"]]))

        # ---- §5.6 combination -------------------------------------------------------
        survive = False
        for lineage in Battery.LINEAGES:
            if not passed[lineage]:
                continue
            for A in bat.cfg["arms"]["real"]:
                if all(verdicts[ds]["lineages"][lineage]["conditions"][A]["clears"]
                       for ds in Battery.DATASET_KEYS):
                    survive = True
        if survive:
            verdict = "SURVIVE"
        elif all(passed[L] for L in Battery.LINEAGES):
            verdict = "CLOSE"
        else:
            verdict = "HALT"

        # ---- I-2: the two REPORTING gates, which carry no bar but are mandatory on the
        #      verdict face (§6.4, §10.2).
        domain = {}
        for ds in Battery.DATASET_KEYS:
            if not passed["N"]:
                domain[ds] = "INSTRUMENT_FAILED"
                continue
            cell = cells[ds]["N"]
            banked = [json.load(open(os.path.join(
                REPO, "scripts/analysis",
                "headspace_arena_{}_s{}_OUT.json".format(ds, s))))["result"]["acc_deployed"]
                for s in Battery.SEEDS]
            acc_native = float(np.mean(banked))
            maj_arena = bat.pop[ds]["majority"]
            maj_full = bat.pop[ds]["full_majority"]
            denom = acc_native - maj_full
            domain[ds] = {
                "recovery_fraction": (float((cell["acc"]["endpoint_std"] - maj_arena)
                                            / denom) if denom else None),
                "acc_ro_endpoint_std_arena": cell["acc"]["endpoint_std"],
                "acc_native_deployed_full": acc_native,
                "maj_arena": maj_arena, "maj_full": maj_full,
                "raw_vs_head_endpoint_std": {
                    "rho_raw_endpoint_std": per_ds[ds]["rho_raw"]["endpoint_std"]},
            }
        bat.gates["GATE-DOMAIN"] = "REPORTED"
        devfid = {}
        for ds in Battery.DATASET_KEYS:
            for s in Battery.SEEDS:
                p = os.path.join(os.path.dirname(os.path.abspath(args.out)),
                                 "fidelity", "devfid_{}_s{}.json".format(ds, s))
                devfid["{}_s{}".format(ds, s)] = (json.load(open(p)).get("gate")
                                                  if os.path.exists(p) else "ABSENT")
        bat.gates["GATE-DEVFID"] = "REPORTED"

        # ---- GATE-LEDGER: MEASURED, never literals (H-2) -----------------------------
        mints_executed = int(os.environ.get("C06_MINTS_EXECUTED",
                                            bat.reports["mints_present_before_arena"]))
        bat.gate_ledger(mints_executed)

        # completeness: all twenty gate names must appear on the face (I-2)
        want = set(bat.cfg["gates"]["global"] + bat.cfg["gates"]["per_lineage"]
                   + bat.cfg["gates"]["reporting"])
        missing = sorted(want - set(bat.gates))
        if missing:
            raise GateFailure("GATE-COMPLETENESS",
                              "verdict face missing gates: {}".format(missing))

        out = {
            "verdict": verdict,
            "design": {"document": cfg["design_document"],
                       "sha256_declared": cfg["design_sha256"],
                       "sha256_derived": _derived_design_sha(bat),
                       "note": DESIGN_SHA_NOTE},
            "gates": bat.gates,
            "dropped_lineages": bat.dropped,
            "lineages_passed": passed,
            "per_dataset": {ds: {
                "arena_n": cells[ds]["N"]["n"],
                "majority": cells[ds]["N"]["majority"],
                "reference_arm": {L: cells[ds][L]["reference"]
                                  for L in Battery.LINEAGES},
                "algebra_residual": {L: cells[ds][L]["algebra_residual"]
                                     for L in Battery.LINEAGES},
                "zeroop": {L: cells[ds][L]["zeroop"] for L in Battery.LINEAGES},
                "displacement_tail": {L: cells[ds][L]["displacement_tail"]
                                      for L in Battery.LINEAGES},
                "rho_raw": per_ds[ds]["rho_raw"],
                "rho_head_max": {L: cells[ds][L]["rho_head_max"]
                                 for L in Battery.LINEAGES},
            } for ds in Battery.DATASET_KEYS},
            "decision": verdicts,
            "gate_domain": domain,
            "gate_devfid": devfid,
            "ledger": bat.ledger,
            "scope": cfg["scope"],
        }
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        tmp = args.out + ".tmp"
        json.dump(out, open(tmp, "w"), indent=1, default=str)
        os.replace(tmp, args.out)
        hb("VERDICT", 1, 1, "{} -> {}".format(verdict, args.out))
        return 0
    except GateFailure as e:
        hb("HALT", extra="INSTRUMENT_INCONCLUSIVE | gate={} | {}".format(e.gate, e.detail))
        emit_halt(args, cfg, bat, gate=e.gate, context=e.detail)
        return 2
    except RuntimeError as e:
        # §5.6: l2_rows / prepare_views signal by die() -> RuntimeError; a crash, not a
        # gate result.  Recorded INSTRUMENT_INCONCLUSIVE with the context string, written
        # to BOTH the final heartbeat line AND the decision JSON (I-1).
        hb("HALT", extra="INSTRUMENT_INCONCLUSIVE | c01_algebra RuntimeError | {}".format(e))
        emit_halt(args, cfg, bat, gate="C01_ALGEBRA_RUNTIMEERROR", context=str(e))
        return 2
    finally:
        hb.close()


if __name__ == "__main__":
    sys.exit(main())
