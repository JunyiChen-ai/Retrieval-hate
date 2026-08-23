# TERA Gate-0 — RE-FREEZE RECORD (2026-08-07, appendix v4)

Closing record for the re-freeze performed under registered deviation **D-3**
(`refine-logs/TERA_GATE0_DEVIATION_D3_2026-08-07.md`), which was written **before** any code was
edited. Nothing below is a candidate result. No Gate-A, Gate-B, Gate-C, temporal, or msc quantity
was computed at any point in this work; `msc_subset.json` still does not exist in any run
directory.

Supersedes `refine-logs/TERA_GATE0_FREEZE_2026-08-07.md` for the digests it restates; everything
that record says about the asset audit, the extraction record and the environment is unchanged and
is not repeated here.

---

## 1. Hash chain, old → new

| item | v3 (frozen 2026-08-07) | v4 (re-frozen 2026-08-07) |
|---|---|---|
| **canonical payload hash** (`cfg["payload"]`) | `7ba80eaf697ac46bb90b30161b1726aba7ee238e73001dd832ce30dba8a1dabe` | **`f2caade97712f8421232dee0a9c6b02545e3ac9ce95357e82e664802316a81e0`** |
| `run_id` prefix that follows | `tera-gate0-<UTC>-7ba80eaf` | `tera-gate0-<UTC>-f2caade9` |
| implementation appendix | `ea158b2c23bd0a9ed8cecdbaccdecd21e97621f9a88b3db8a7c2dcbba2c42ffc` (v3) | `06808e12d737bd5b43cb8b0cd4779428c443adfa064e3c7cf750216aa356231e` (v4) |
| frozen config, whole file | `fdebff8bd72b704f0a5da8e007145bdb06a1f365c6ed2ab4e38507bf92541bdc` | `e45abb3749130e43b2135016f92047e068605aa18616d69dffbdc887267fcb82` |
| harness `package_aggregate_sha256` | `7e20884b6272bc98a94a367dc2823ac06c772c16d54a5f1bd415993c11f8e9f2` | `cb619464b0223ed551f6078d31a67a4a9f832bb42f59d540136fe8d7dd7463aa` |
| `fixtures.py` | `1cd6c48226345c91c7423c7c61e805f94c8d05c36492af814a1bc266491dfd36` | `d967f78e87fe31e4275ca163834bc304f6314a36f4e031b0de90825f0d282f7c` |
| `gate_c.py` | `27dc026d1f0fea882ee71e007660a6efe5d06cac4531dbe39d07c5e37c05bd6b` | `811b292dd0aa5831c1b7b2ffd8ea5eeda5b1d755f10c29637cbdad779a42e6f9` |
| `run_gate0.py` | `a947af0cf05548c802b4f30cccdc3eb7e4d32c533606a9dd08fee162a122d81d` | `12a5f10d513983f843e95aefe13bc06f370043b4cd4ccbabe59cc736e5799e9a` |
| fixtures report | `f21b465e69ac11dc620dfdf9bc66e676cd6749bde65344f1e33762ed979a1fb5` (`fix-20260806T231531Z`) | `b9161be50cd33227eb1c158e378f32ff0f3624e5f903ad1267a83fca137021e0` (`fix-20260807T083546Z`) |
| pre-registration | `f6c1ce6c652bcedd18451d4ee3a490ca2c72c603489e89c6a161855537ed6e98` | **unchanged** — not edited |
| independent review record | `9147ad4c1adacf1160566f0503937d45e0f5205b43549359d3eefe68263637c5` | **unchanged** — not edited |

The other 11 `*.py` files of `scripts/tera_gate0/` are byte-identical to v3; their digests were
re-verified against the frozen v3 map before the first edit and again after the re-freeze.

Verification performed after writing: `load_frozen_config()` re-derives
`f2caade9…` from the file on disk (no `HALT_CONFIG_HASH_MISMATCH`); the payload's
`study.appendix_sha256` equals the appendix file byte-for-byte; the payload's
`fixtures.package_sha256` map equals the map the v2 battery itself wrote into its report;
`status` is still `FROZEN`.

## 2. What changed in the payload, and proof that nothing else did

Exactly these payload fields were rewritten:

- `study.appendix_version` `v3` → `v4`, `study.appendix_sha256` re-embedded;
- `fixtures.script_sha256`, `fixtures.report_path`, `fixtures.report_sha256`,
  `fixtures.run_mode`, `fixtures.fixture_run_id`, `fixtures.package_sha256`,
  `fixtures.package_aggregate_sha256`, `fixtures.battery_result`;
- one appended `change_log` entry (`v4`).

**Proof of completeness.** Reverting exactly and only those fields to their v3 values and
recomputing the canonical hash reproduces `7ba80eaf697ac46bb90b30161b1726aba7ee238e73001dd832ce30dba8a1dabe`
bit-for-bit. No threshold, seed, arm, split, cache digest, taxonomy entry, decision rule, HALT
condition, whitelist or asset-audit value was touched. `asset_audit.documents.appendix_v3_sha256` is
deliberately left at the v3 digest: it is a historical record of the document as read during the
read-only asset audit, and its key name says so.

The `_hash_note` (a sibling of `payload`, outside the hashed object by construction) was extended to
name the superseded payload hash and point at this record.

## 3. Harness diff — the complete change

The three edited files were reconstructed back to their v3 form by reversing the edits below; all
three reconstructions hash **exactly** to the v3 frozen digests in §1, which proves this diff is
the entire change to the harness package.

```diff
--- a/scripts/tera_gate0/gate_c.py
+++ b/scripts/tera_gate0/gate_c.py
@@ -128,6 +128,17 @@
     return mech


+def resolve_audit_rows(audit_rows):
+    """One row per audited video: the adjudicated row if present, else the first
+    row in file order (appendix sec 6.7 resolution; deviation D-3)."""
+    by_video = {}
+    for row in audit_rows:
+        by_video.setdefault(row["video_id"], []).append(row)
+    resolved = {vid: ([r for r in rws if r.get("adjudicated")] or rws[:1])[0]
+                for vid, rws in by_video.items()}
+    return by_video, resolved
+
+
 def weighted_coverage(audit_fn, mech_by_video, weights, mech_set):
     num = sum(weights[v] for v in audit_fn
               if mech_by_video[v] & set(mech_set))
@@ -185,13 +196,14 @@
     return {"checks": checks, "pass": all(checks.values())}


-def msc_subset(audit_records):
-    """Frozen msc subset: audited videos of ANY category carrying msc (sec 6.7)."""
-    out = []
-    for rec in audit_records:
-        if "multi_segment_complementary" in mechanisms_of(rec):
-            out.append(rec["video_id"])
-    return sorted(set(out))
+def msc_subset(audit_rows):
+    """Frozen msc subset (sec 6.7): EVERY audited video of any category whose
+    resolved cause carries multi_segment_complementary as primary or secondary.
+    Resolution is adjudicated-else-first, so a double-coded video on which the
+    coders agreed (two rows, no adjudication row) is included (deviation D-3)."""
+    _, resolved = resolve_audit_rows(audit_rows)
+    return sorted(vid for vid, rec in resolved.items()
+                  if "multi_segment_complementary" in mechanisms_of(rec))


 def rescue_metrics(msc_ids, labels, pred_b0, pred_b2):
--- a/scripts/tera_gate0/run_gate0.py
+++ b/scripts/tera_gate0/run_gate0.py
@@ -786,14 +786,11 @@
             return
         audit_rows = [r for r in read_jsonl(self.args.gate_c_audit)
                       if not r.get("superseded")]
+        by_video, resolved = gc.resolve_audit_rows(audit_rows)
         adjudicated = {}
         pairs = []
-        by_video = {}
-        for row in audit_rows:
-            by_video.setdefault(row["video_id"], []).append(row)
         for vid, rws in by_video.items():
-            final = [r for r in rws if r.get("adjudicated")] or rws[:1]
-            adjudicated[vid] = gc.mechanisms_of(final[0])
+            adjudicated[vid] = gc.mechanisms_of(resolved[vid])
             if len(rws) >= 2:
                 pairs.append((rws[0]["primary_cause"], rws[1]["primary_cause"]))
         audit_fn = [v for v in sample["audit_fn"] if v in adjudicated]
@@ -814,9 +811,7 @@
                                  "n_audited_fn": len(audit_fn),
                                  "unweighted_union": gc.unweighted_coverage(
                                      audit_fn, adjudicated, gc.UNION_SET)}
-        msc_ids = gc.msc_subset([r for r in audit_rows
-                                 if r.get("adjudicated") or
-                                 len(by_video[r["video_id"]]) == 1])
+        msc_ids = gc.msc_subset(audit_rows)
         path = write_json_new(self.run_dir / "msc_subset.json",
                               {"video_ids": msc_ids, "n": len(msc_ids)})
         self.msc_subset_sha256 = sha256_file(path)
--- a/scripts/tera_gate0/fixtures.py
+++ b/scripts/tera_gate0/fixtures.py
@@ -35,8 +35,8 @@
 from .arms import head_capacity_check, params_b2, params_b3, solve_h3
 from .common import (FIXTURE_SEED_BASE, K_WINDOWS, TeraHalt, canonical_json, note,
                      read_jsonl, repo_root, select_threshold, sha256_file, sha256_obj)
-from .gate_c import (coverage_bootstrap, redistribute, select_audit_sample,
-                     unweighted_coverage, weighted_coverage)
+from .gate_c import (coverage_bootstrap, msc_subset, redistribute,
+                     select_audit_sample, unweighted_coverage, weighted_coverage)
 from .guards import Authorization, SealGuard, load_corpus_spanning
 from .nested import inner_folds
 from .synthetic import DURATION, K, build_dataset, pattern_score_override
@@ -403,6 +403,42 @@
     # (3) deterministic deficit redistribution on an undersized tercile.
     got = redistribute({0: 40, 1: 40, 2: 40}, {0: 150, 1: 130, 2: 20})
     out.append(check("F11.deficit_redistribution", got == {0: 60, 1: 40, 2: 20}, got))
+
+    # (4) msc-subset membership on synthetic audit rows (sec 6.7; deviation D-3).
+    #     Every audited video of any category whose adjudicated-else-first row
+    #     carries msc as primary or secondary, including a double-coded video on
+    #     which the two coders agreed and which therefore has no adjudication row.
+    msc = "multi_segment_complementary"
+    audit_rows = [
+        {"video_id": "v_single_msc", "primary_cause": msc, "secondary_causes": []},
+        {"video_id": "v_single_sec", "primary_cause": "short_localized",
+         "secondary_causes": [msc]},
+        {"video_id": "v_single_out", "primary_cause": "global_evidence",
+         "secondary_causes": []},
+        # double-coded, coders AGREE on msc, no adjudication row -> must be in
+        {"video_id": "v_double_agree", "primary_cause": msc, "secondary_causes": []},
+        {"video_id": "v_double_agree", "primary_cause": msc, "secondary_causes": []},
+        # double-coded, coders disagree, adjudicated TO msc -> in
+        {"video_id": "v_adj_in", "primary_cause": "global_evidence",
+         "secondary_causes": []},
+        {"video_id": "v_adj_in", "primary_cause": "cross_modal", "secondary_causes": []},
+        {"video_id": "v_adj_in", "primary_cause": msc, "secondary_causes": [],
+         "adjudicated": True},
+        # first row carries msc but adjudication overrides it AWAY from msc -> out
+        {"video_id": "v_adj_out", "primary_cause": msc, "secondary_causes": []},
+        {"video_id": "v_adj_out", "primary_cause": "global_evidence",
+         "secondary_causes": []},
+        {"video_id": "v_adj_out", "primary_cause": "global_evidence",
+         "secondary_causes": [], "adjudicated": True},
+    ]
+    got = msc_subset(audit_rows)
+    out.append(check("F11.msc_subset_membership",
+                     got == ["v_adj_in", "v_double_agree", "v_single_msc",
+                             "v_single_sec"], got))
+    out.append(check("F11.msc_subset_includes_agreeing_double_coded_pair",
+                     "v_double_agree" in got, got))
+    out.append(check("F11.msc_subset_respects_adjudication_both_ways",
+                     "v_adj_in" in got and "v_adj_out" not in got, got))
     return out
```

### 3.1 Behavioural scope of the diff

- `msc_subset` is the only routine whose output changes. It now returns every audited video whose
  adjudicated-else-first row carries `multi_segment_complementary`, instead of only those videos
  that are single-coded or adjudicated.
- The Gate-C coverage/kappa path is **bit-identical**. The `adjudicated` mechanism map and the
  `pairs` list (including their construction order, which D-2 §3 registers as load-bearing) were
  checked against the v3 loop on 500 randomized synthetic row structures — dict ordering, mechanism
  sets and pair tuples matched on 500/500. Consequently no Gate-C check
  (`union >= 0.30`, `ci_lower >= 0.20`, `msc >= 0.15`, `noise <= 0.25`, `kappa >= 0.60`) can move.
- Nothing else in the package is touched, so Gate-A, arm `D`, the temporal metrics, the bootstrap,
  the confirmation protocol, the sealed-id reader and every HALT condition are unaffected by
  construction.

## 4. Fixture battery v2 (self-test evidence for the fix)

- Report: `artifacts/tera_gate0/_fixtures/fix-20260807T083546Z/fixtures_report.json`
  (sha256 `b9161be50cd33227eb1c158e378f32ff0f3624e5f903ad1267a83fca137021e0`).
- **16 requested / 16 PASS / 0 FAIL** — F1, F2, F3, F4, F5, F6, F7, F7b, F8, F9, F10, F11, F12,
  F13, F14, F15. Wall clock 965.2 s, `fixture_bootstrap_n = 1000`, `seed_base = 424242`,
  75 assertions total (72 at v3 + the 3 new F11 assertions).
- Log: `logging/runs/tera_gate0_fixtures_v2/run.log`.
- The report's own `package_sha256` block equals the map written into the re-frozen payload, so the
  battery and the frozen config describe the same bytes.
- **Directed regression evidence.** The three new F11 assertions were run against the v3 bytes in a
  scratch copy: `F11.msc_subset_membership` and
  `F11.msc_subset_includes_agreeing_double_coded_pair` **FAIL** there (the v3 code returns
  `['v_adj_in', 'v_single_msc', 'v_single_sec']`, dropping the agreeing double-coded video), and all
  three PASS on v4. The new assertions therefore test the defect, not merely the code.
- Per the project's proportional-ceremony rule and the appendix status block ("author self-test
  evidence (a full §9 fixture-battery pass) releases the freeze"), this is the release evidence; no
  new review round was opened.

### 4.1 Procedural note — the draft-path config

`run_gate0.py`'s `--config` argparse default still names the pre-freeze
`research-wiki/tera_gate0_frozen_config.draft.json` path, deliberately unpatched (appendix §10.1),
and `fixtures.py`'s subprocess launcher does not pass `--config`. The v3 release battery therefore
ran before the `.draft.json` → `.json` rename. To reproduce that condition without touching a
frozen byte, a **byte-identical copy** of the frozen config (sha256
`fdebff8bd72b704f0a5da8e007145bdb06a1f365c6ed2ab4e38507bf92541bdc`, i.e. the v3 file, since the
re-freeze had not yet been written) was placed at the draft path for the duration of the battery
and **deleted immediately afterwards**. `research-wiki/` again contains only
`tera_gate0_frozen_config.json`, so the §10.1 guarantee — a default launch points at a path that
does not exist and fails immediately — is restored.

A first launch attempt made before this copy existed died in 2 s with
`FileNotFoundError` on that path; its artefact directory
`artifacts/tera_gate0/_fixtures/fix-20260807T083405Z/` is retained as the record of that failure and
is **not** the release report.

## 5. Effect on existing artefacts

- **Run 1** (`artifacts/tera_gate0/tera-gate0-20260807T000625Z-7ba80eaf/`) is **not modified** and
  not re-run. It was submitted without `--gate-c-audit`, so it never entered the defective code
  path; `msc_subset_sha256` in its `manifest.json` is `null` and no `msc_subset.json` exists. Its
  registered products (the A0 OOF prediction source, `gate_c_sample.json`,
  `annotation_protocol.json`, `gate_c_items_blinded.jsonl`) are unaffected by the fix and remain
  the Gate-C sampling frame. D-1 §2's isolation clause continues to apply to its `metrics.json` and
  `verdict.json`.
- **`gate_c_audit.jsonl`** is **not modified**. It was read for row/`video_id`/`adjudicated`
  structure only, to confirm the defect's blast radius (133 audited videos, 22 of them double-coded
  with agreeing coders and hence dropped by the v3 code). No cause field was read and no coverage,
  kappa or msc quantity was computed.
- **Run 2** executes under the new `run_id` prefix `tera-gate0-<UTC>-f2caade9`, against
  `--config research-wiki/tera_gate0_frozen_config.json`, exactly once, as prereg §12 requires for
  the affected stage. The launch line is otherwise identical to D-1 §4's with
  `--stages A,C,B --gate-c-audit …` and the confirmation protocol, and must export the **new**
  fixture digests:

  ```bash
  export TERA_FIXTURES_REPORT_SHA256=b9161be50cd33227eb1c158e378f32ff0f3624e5f903ad1267a83fca137021e0
  export TERA_FIXTURE_CODE_SHA256=d967f78e87fe31e4275ca163834bc304f6314a36f4e031b0de90825f0d282f7c
  ```

## 6. Standing discipline (unchanged)

1. Zero test-set contact — `data/gt/HateMM/test.jsonl` was not opened during this work;
   `test_contact_count` must still end at 0 in Run 2.
2. Decision rules frozen before any result is seen — no rule was changed here, and D-3 was
   registered before the first byte was edited.
3. Blinding — no candidate metric was computed during the verification, the fix, the battery, or
   this record.
4. Single-submission execution — Run 2 remains one registered submission per stage.
5. Launch with an explicit `--config research-wiki/tera_gate0_frozen_config.json`.
