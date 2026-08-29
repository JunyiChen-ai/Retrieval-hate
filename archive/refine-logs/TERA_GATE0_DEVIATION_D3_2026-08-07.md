# TERA Gate-0 — REGISTERED DEVIATION D-3

- **Study**: TERA-GATE0
- **Prereg**: `research-wiki/EXP_tera_gate0_prereg.md` (sha256 `f6c1ce6c652bcedd18451d4ee3a490ca2c72c603489e89c6a161855537ed6e98`, unchanged by this record)
- **Appendix at time of discovery**: `research-wiki/EXP_tera_gate0_impl_appendix.md` v3 (sha256 `ea158b2c23bd0a9ed8cecdbaccdecd21e97621f9a88b3db8a7c2dcbba2c42ffc`)
- **Frozen config at time of discovery**: `research-wiki/tera_gate0_frozen_config.json`, `payload_sha256 = 7ba80eaf697ac46bb90b30161b1726aba7ee238e73001dd832ce30dba8a1dabe`
- **Frozen harness at time of discovery**: `scripts/tera_gate0/*.py`, `package_aggregate_sha256 = 7e20884b6272bc98a94a367dc2823ac06c772c16d54a5f1bd415993c11f8e9f2` — re-verified byte-identical to the frozen payload's `fixtures.package_sha256` map immediately before this record was written, i.e. the defect is in the frozen bytes and not in a drifted working copy.
- **Registered (UTC date)**: 2026-08-07
- **Affected stage**: Gate-B rescue criterion / FP side condition (appendix §6.7), via the `msc_subset.json` artefact written by stage C. Run 2 only.
- **Authority**: prereg §12, second bullet — a defect in a registered decision input, found **before** the affected candidate metric is computed. Fix, re-freeze a new payload hash, run the affected stage under the new hash.

---

## 1. Discovery point and blinding status

Discovered **after** Gate-C annotation was completed and `gate_c_audit.jsonl` was assembled
(`artifacts/tera_gate0/tera-gate0-20260807T000625Z-7ba80eaf/gate_c_audit.jsonl`), and **before**
Run 2 — the registered decision run — was submitted.

At the time of writing:

- `msc_subset.json` **does not exist** in any run directory. The only completed run is Run 1
  (`--stages A,C --confirmation none`, D-1), which was submitted **without** `--gate-c-audit`, so
  `run_stage_c()` returned at `run_gate0.py:783-786` and the `msc_subset` code path was never
  reached. `msc_subset_sha256` in Run 1's `manifest.json` is therefore `null`.
- No msc subset, no rescue rate, no FP side-condition count, no Gate-B decision, and no Gate-C
  coverage/kappa quantity has ever been computed by the harness or by hand.
- The defect was found by code reading during a Gate-C hand-off review, not by inspecting any
  result.

This is the prereg §12 "stop before computing the affected candidate metric" path. It is **not** a
post-unblinding correction: nothing that this fix changes has ever been observed, so the third
§12 bullet (retain-and-label-invalid) does not apply and no verdict is invalidated.

## 2. The defect

`scripts/tera_gate0/run_gate0.py:817-819` (frozen bytes):

```python
        msc_ids = gc.msc_subset([r for r in audit_rows
                                 if r.get("adjudicated") or
                                 len(by_video[r["video_id"]]) == 1])
```

`gc.msc_subset` (`gate_c.py:188-195`) expects a list of **already-resolved** rows, one per audited
video, and tests each for `multi_segment_complementary` under the primary-or-secondary presence
rule. The row filter above is meant to perform that resolution but does not: it admits a row only
if the row itself is an adjudication row, or if its video carries exactly one row. A video that was
double-coded and whose two coders **agreed** has two rows and no adjudication row (adjudication is
triggered only by a `primary_cause` disagreement — D-2 §3), so **neither** of its rows passes the
filter and the video is dropped from the msc subset entirely, regardless of its cause.

Structural counts of the submitted audit file (row/`video_id`/`adjudicated` structure only; no
cause field was read to produce them):

| quantity | value |
|---|---|
| rows in `gate_c_audit.jsonl` (none `superseded`) | 165 |
| distinct audited videos | 133 |
| videos with exactly one row | 106 |
| double-coded videos **with** an adjudication row | 5 |
| double-coded videos **without** an adjudication row (coders agreed) | **22** |
| videos reaching `gc.msc_subset` under the frozen filter | 111 |
| videos that should reach it under the registered definition | 133 |

For the 106 single-row videos and the 5 adjudicated videos the frozen filter already yields exactly
the registered resolution, so the defect is confined to the 22 agreeing double-coded videos.

No downstream second filter compensates. `run_gate0.py:703-707` consumes `self.msc_ids` directly
and `gate_c.rescue_metrics` only splits it by gold label; the truncated subset propagates unchanged
into both Gate-B inputs and into `msc_subset.json` / `msc_subset_sha256`.

## 3. The registered definition (verbatim)

`research-wiki/EXP_tera_gate0_impl_appendix.md` §6.7 (BLOCKING-FIX B-5), quoted byte-for-byte:

> The msc subset is the set of **audited videos of any category** — audited false negatives **and**
> the 30 TP + 30 FP controls — carrying `multi_segment_complementary` as **primary or secondary**
> cause, per prereg §4.3's presence rule. It is frozen when Gate-C's adjudicated audit is written
> and is stored as `msc_subset.json` with a SHA256 in `manifest.json`.

The per-video cause used for that test is the one the frozen harness already computes for coverage
at `run_gate0.py:794-796` — the adjudicated row if the video has one, else the video's first row in
file order:

```python
        for vid, rws in by_video.items():
            final = [r for r in rws if r.get("adjudicated")] or rws[:1]
            adjudicated[vid] = gc.mechanisms_of(final[0])
```

D-2 §3 registers that this ordering is load-bearing and that rows are append-only, so "first row"
is the c1 row for every double-coded video. The registered subset is therefore: *every* audited
video whose adjudicated-else-first row carries `multi_segment_complementary` as primary or
secondary. The frozen filter implements a strictly smaller set.

Nothing about the definition is being changed here. The appendix text is the registration; the code
is being brought into agreement with it.

## 4. Minimal fix (predeclared, before any edit)

Two files, one behavioural change: the adjudicated-else-first resolution is lifted into a single
shared helper in `gate_c.py` so that the coverage path and the msc path cannot diverge, and
`msc_subset` is given raw audit rows and performs the resolution itself.

`scripts/tera_gate0/gate_c.py` — new helper after `mechanisms_of`:

```python
def resolve_audit_rows(audit_rows):
    """One row per audited video: the adjudicated row if present, else the first
    row in file order (appendix sec 6.7 resolution; deviation D-3)."""
    by_video = {}
    for row in audit_rows:
        by_video.setdefault(row["video_id"], []).append(row)
    resolved = {vid: ([r for r in rws if r.get("adjudicated")] or rws[:1])[0]
                for vid, rws in by_video.items()}
    return by_video, resolved
```

`scripts/tera_gate0/gate_c.py` — `msc_subset` takes raw rows:

```python
def msc_subset(audit_rows):
    """Frozen msc subset (sec 6.7): EVERY audited video of any category whose
    resolved cause carries multi_segment_complementary as primary or secondary.
    Resolution is adjudicated-else-first, so a double-coded video on which the
    coders agreed (two rows, no adjudication row) is included (deviation D-3)."""
    _, resolved = resolve_audit_rows(audit_rows)
    return sorted(vid for vid, rec in resolved.items()
                  if "multi_segment_complementary" in mechanisms_of(rec))
```

`scripts/tera_gate0/run_gate0.py:789-796` — the existing loop is rebased on the helper, with its
behaviour (the `adjudicated` mechanism map and the kappa `pairs` list, including their construction
order) unchanged:

```python
        by_video, resolved = gc.resolve_audit_rows(audit_rows)
        adjudicated = {}
        pairs = []
        for vid, rws in by_video.items():
            adjudicated[vid] = gc.mechanisms_of(resolved[vid])
            if len(rws) >= 2:
                pairs.append((rws[0]["primary_cause"], rws[1]["primary_cause"]))
```

`scripts/tera_gate0/run_gate0.py:817-819` — the defective filter is deleted:

```python
        msc_ids = gc.msc_subset(audit_rows)
```

Plus `scripts/tera_gate0/fixtures.py`: the `from .gate_c import (...)` list gains `msc_subset`, and
`fixture_f11` gains a fourth block of assertions on synthetic audit rows covering the defect case
(see §6). No other line of the harness is touched.

**Explicitly unchanged by this fix:** the taxonomy, the presence rule, the union set, the tercile
weights, the sampling seed, the bootstrap protocol, the kappa pair construction and its order, the
Gate-C thresholds (`union >= 0.30`, `ci_lower >= 0.20`, `msc >= 0.15`, `noise <= 0.25`,
`kappa >= 0.60`), the Gate-B thresholds, the rescue criterion, the FP side condition and both
`not_evaluable` conventions, every seed, and every HALT condition. Gate-C's own coverage numbers are
computed from `audit_fn` and the `adjudicated` map, which the fix leaves bit-identical, so **no
Gate-C quantity changes at all**; the fix is confined to `msc_subset.json` and the two Gate-B
criteria that read it.

## 5. Directional effect on the registered endpoints

Stated before any affected number exists. Both directions are written out because the sign is not
determined by the design.

The fix **enlarges** the msc subset from "audited videos that are single-coded or adjudicated, and
carry msc" to "all audited videos that carry msc" — the candidate pool grows from 111 videos to the
full 133, and the subset can only gain members, never lose them (the 111 resolutions are unchanged).
Consequences:

1. **Rescue rate** `= |{v in msc, label 1: B0 predicts 0 and B2 predicts 1}| / |{v in msc, label 1: B0 predicts 0}|`.
   Both numerator and denominator can only grow. The ratio can move **either way**: added videos
   that B2 rescues push it up, added videos that B2 fails to rescue push it down. What is
   directional is the *precision*: the frozen behaviour would have evaluated a `>= 0.20` criterion —
   and its Wilson interval — on a denominator drawn from a strictly smaller, non-registered
   sample, i.e. a noisier estimate of a quantity the appendix already flags as small (§6.7,
   review N-12: "the msc label-1 subset could be ~18 videos"). The fix restores the registered
   sample. It also makes the "denominator is 0 => not_evaluable => NOT satisfied" branch strictly
   less likely to fire.
2. **FP side condition** `FP_B2 <= FP_B0 + max(1, ceil(0.10 * FP_B0))`, in counts. Both `FP_B0` and
   `FP_B2` can only grow. The guard can therefore flip **either way**: a larger `FP_B0` widens the
   allowance, while added label-0 videos that B2 alone fires on tighten the margin. The
   "no label-0 member => not_evaluable => SATISFIED" branch becomes strictly less likely to fire,
   which makes the do-no-harm guard **more** likely to be genuinely evaluated — a tightening.
3. **Gate-B verdict.** Since the rescue criterion is a positive requirement and the FP condition a
   do-no-harm guard, the fix can turn a would-be Gate-B pass into a fail or a would-be fail into a
   pass. No claim of neutrality is made. The point is that the post-fix evaluation is the
   registered one and the pre-fix evaluation was not.
4. **Gate-A, Gate-C, the temporal metrics and the confirmation protocol**: unaffected — no code
   they read is touched, and the Gate-C coverage/kappa path is bit-identical (§4).

**Direction of the defect itself, for the record:** the frozen behaviour silently narrowed a
registered denominator by ~17% of audited videos on the basis of a coding-process artefact
(whether a video happened to be drawn into the 20% double-coding sample *and* whether its two
coders happened to agree) that is orthogonal to the scientific quantity. It is not conservative in
either direction; it is simply a different, unregistered, and partly luck-determined subset. That
is why it is fixed rather than accepted.

## 6. Self-test evidence

Per the project's proportional-ceremony rule, a fix of this class is released by author self-test
evidence rather than a new review round. The evidence is:

1. A new assertion block in `fixture_f11` (`scripts/tera_gate0/fixtures.py`) over synthetic audit
   rows, asserting the full membership set of `msc_subset` on six constructed videos: msc as
   primary (in), msc as secondary (in), no msc (out), **double-coded with two agreeing msc rows and
   no adjudication row (in — the exact case the defect dropped)**, disagreement adjudicated *to* msc
   (in), and a first row carrying msc that adjudication overrides *away* from msc (out). The frozen
   code fails the fourth case; the fixed code passes all six.
2. A full re-run of the §9 fixture battery (F1–F15, 16 cases) on the fixed package, logged to
   `logging/runs/tera_gate0_fixtures_v2/run.log`, required to be 16/16 PASS.

## 7. Consequences for the freeze

Editing harness bytes changes `fixtures.package_sha256` / `package_aggregate_sha256`, which live
inside the hashed payload; editing the appendix changes `study.appendix_sha256`, also inside the
payload. Both therefore change `payload_sha256` and hence the `run_id` prefix (appendix §10.3), by
design and per appendix §10.2 ("any post-freeze payload change produces a new hash and a new
`run_id`; prereg §12 governs").

Accordingly, and unlike D-1 §3 / D-2 §5 — which deliberately avoided editing the frozen documents
because no code change was in scope — this deviation **does** re-freeze:

- `scripts/tera_gate0/gate_c.py`, `run_gate0.py`, `fixtures.py` are edited exactly as §4 predeclares
  and re-hashed;
- `research-wiki/EXP_tera_gate0_impl_appendix.md` gains one §14 change-log entry (v4) citing this
  file, and its digest is re-embedded;
- `research-wiki/tera_gate0_frozen_config.json` gets the new per-file map, aggregate, fixture report
  reference and a recomputed `payload_sha256`;
- the whole chain is recorded in `refine-logs/TERA_GATE0_REFREEZE_2026-08-07.md`.

Run 1's artefacts stay under the old `…-7ba80eaf` namespace and are untouched; they remain valid as
the Gate-C prediction source and sample, because the fix changes no quantity Run 1 produced (Run 1
never reached the msc path at all). Run 2 executes under the new `run_id` prefix against the new
config, exactly once, as prereg §12 requires for the affected stage.

The documentary back-fill into the prereg's `REGISTERED DEVIATIONS / ERRATA` subsection is deferred
to campaign close-out for the reason in D-1 §3 (the prereg digest is embedded in the payload and is
not being changed by this record). **Until then, this file is the authoritative timestamp for D-3.**
