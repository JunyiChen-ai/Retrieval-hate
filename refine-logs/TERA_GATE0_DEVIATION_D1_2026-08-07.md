# TERA Gate-0 — REGISTERED DEVIATION D-1

- **Study**: TERA-GATE0
- **Prereg**: `research-wiki/EXP_tera_gate0_prereg.md` (sha256 `f6c1ce6c652bcedd18451d4ee3a490ca2c72c603489e89c6a161855537ed6e98`)
- **Appendix**: `research-wiki/EXP_tera_gate0_impl_appendix.md` v3 (sha256 `ea158b2c23bd0a9ed8cecdbaccdecd21e97621f9a88b3db8a7c2dcbba2c42ffc`)
- **Frozen config**: `research-wiki/tera_gate0_frozen_config.json`, `payload_sha256 = 7ba80eaf697ac46bb90b30161b1726aba7ee238e73001dd832ce30dba8a1dabe`
- **Frozen harness**: `scripts/tera_gate0/*.py`, `package_aggregate_sha256 = 7e20884b6272bc98a94a367dc2823ac06c772c16d54a5f1bd415993c11f8e9f2` (verified byte-identical to the frozen payload on 2026-08-07, immediately before this record was written)
- **Registered (UTC date)**: 2026-08-07
- **Registered BEFORE**: any affected metric was computed. No Gate-A, Gate-B, Gate-C or temporal number exists at the time of writing; `artifacts/tera_gate0/` contains only `_fixtures/`.
- **Authority**: main-conversation adjudication, 2026-08-07 (option C of the pre-submission escalation).

---

## 1. D-1 — stage A is necessarily executed twice

### 1.1 The structural cause

The hash-frozen harness couples the three gates through **in-process state**, not through
on-disk artifacts. Three dependencies, all in `scripts/tera_gate0/run_gate0.py`:

1. **C needs A, in-process.** `run()` raises
   `TeraHalt("HALT_STAGE_ORDER", "C needs the A0 OOF run (stage A)")` unless `self.gate_a`
   is not `None` (L1195-1198). `self.gate_a` is assigned only inside `compute_metrics()`
   (L498), which runs only under the `"A" in self.stages` branch (L1191-1194). There is no
   code path, CLI flag, or frozen-config field that loads a previous run's
   `oof_predictions.jsonl`, and none that restricts `run_stage_a()` to the A0 arm alone —
   it iterates the full `A_ARMS = ("A0","A1","A2","A3","A4")` (L246-249) and
   `compute_metrics()` additionally computes O1/O2, the arm-`D` selection, the temporal
   metrics, the bootstrap, and the Gate-A verdict.

2. **Confirmation needs A, in-process.** `run_confirmation()` (L878) reaches
   `confirm_hatemm_val()` → `d_outer_oof_segment_scores()` (L872), which reads
   `self.a_results` and `self.d_arm`. Confirmation therefore cannot be a separate
   submission from stage A.

3. **B needs C's human audit.** Stage B's Gate-B rescue criterion reads `self.msc_ids`
   (L703). `self.msc_ids` is produced only by `run_stage_c()` and only when
   `--gate-c-audit` is supplied (L817-823), i.e. only after human annotation has been
   collected.

### 1.2 The consequence

Gate-C's annotation package requires a stage-A execution. Gate-B and the Gate-C decision
require the human audit, which requires that annotation package, and they also require a
live in-process stage A. **There is therefore no execution schedule under which stage A
runs exactly once.** The frozen design admits a minimum of two stage-A executions:

- **Run 1 (this run) — the prediction-source run.** `--stages A,C --confirmation none`.
  Purpose: produce the A0 whole-video OOF predictions on HateMM-train, the Gate-C
  stratified sample, the frozen tercile weights, and the blank annotation package.
- **Run 2 (later) — the registered decision run.** After the human audit is complete:
  `--stages A,C,B --gate-c-audit <path> --confirmation …`. This is the run that produces
  the registered Gate-A, Gate-B and Gate-C verdicts.

This is a property of the frozen artefact, not a choice made after seeing a number.

### 1.3 Guarantee — no result-selection freedom

Appendix §7.9 registers the whole A/B/O head path as CPU-only precisely because "CPU gives
bitwise reproducibility under a pinned thread count and removes GPU kernel nondeterminism"
(`torch.set_num_threads = 8`, pinned in the frozen environment block).

The frozen fixture battery tests exactly this property. **F13 (determinism)**: "rerun F1 end
to end in two separate processes" with assertions "byte-identical `oof_predictions.jsonl`;
identical `metrics.json` numbers" (appendix §9.2, table row F13). The battery report
`artifacts/tera_gate0/_fixtures/fix-20260806T231531Z/fixtures_report.json`
(sha256 `f21b465e69ac11dc620dfdf9bc66e676cd6749bde65344f1e33762ed979a1fb5`) records
`summary = {"n": 16, "passed": 16, "failed": 0}`.

Run 2 consumes the same frozen config, the same frozen harness bytes, the same seed
register (§7.6) and the same input caches as Run 1. Its stage-A output is therefore
bit-for-bit determined before Run 1 is launched. **No quantity in Run 2 can be selected,
tuned, or shopped for by the existence of Run 1.**

**Expected directional effect on any registered endpoint: none.** The deviation is an
execution-count artefact with a proof of invariance, not a change to the design, the data,
the arms, the seeds, the thresholds, or the decision rules.

---

## 2. Isolation clause — Run 1's Gate-A output is void ab initio

Run 1 is submitted with `--confirmation none`. Gate-A check 6
(`6_confirmations_positive`, `scripts/tera_gate0/verdict.py:22`) then evaluates to
`"not_evaluated"`, `all_pass` is `False`, and `gate_a_decision` necessarily returns the
verdict string `NO-GO-A-SELECTOR` (or `NO-GO-A-NO-HEADROOM` if the oracle checks fail
first). **This string is a mechanical consequence of running without confirmation. It is
not a scientific finding.**

Accordingly, by main-conversation adjudication of 2026-08-07:

1. **Run 1's scope is the Gate-C prediction source and sample, and nothing else.** Its
   registered products are: `oof_predictions.jsonl` (A0 subset only), the A0 confusion
   matrix and FN population, `gate_c_sample.json`, `annotation_protocol.json`,
   `gate_c_items_blinded.jsonl`, `split_manifest.json`, `feature_manifest.json`,
   `manifest.json`, and the per-fold id lists.

2. **Run 1's `verdict.json` is void ab initio and constitutes no registered decision of any
   kind** — not a Gate-A verdict, not a provisional verdict, not evidence. Its path is
   recorded for audit completeness; its contents are not a result.

3. **Reading restriction.** No one may read, quote, summarise, or act on the A1, A2, A3,
   A4, O1, O2, arm-`D`, temporal, or bootstrap numbers written by Run 1, whether in
   `metrics.json` or `verdict.json`. Confirming that these files exist and hashing them is
   permitted; opening those sections is not. The A0 confusion matrix **is** readable — it
   is the Gate-C sampling frame, not a TERA candidate metric.

4. **If Gate-C returns NO-GO**, Run 1's `metrics.json` and `verdict.json` remain sealed and
   this record stands as the note that they were void from the moment they were written.

5. **The registered Gate-A decision is whatever Run 2 produces**, after Gate-C passes,
   under its own single frozen submission with the confirmation protocol of appendix §7.10.

---

## 3. Document-integrity note

`research-wiki/EXP_tera_gate0_prereg.md` and
`research-wiki/EXP_tera_gate0_impl_appendix.md` are **not edited**. Their sha256 digests are
embedded in the frozen payload and are re-verified at run start; changing a byte of either
would alter `payload_sha256`, alter the `run_id`, and trigger
`HALT_CONFIG_HASH_MISMATCH`.

The documentary back-fill into the prereg's `REGISTERED DEVIATIONS / ERRATA` subsection is
therefore deferred to campaign close-out. **Until then, this file is the authoritative
timestamp for D-1.** Per prereg §12, a documentary correction of this kind does not require
a rerun.

---

## 4. Run 1 — the exact submission this record precedes

```bash
source /home/jehc223/miniconda3/etc/profile.d/conda.sh && conda activate HateVideo
cd /home/jehc223/Retrieval-hate
export PYTHONPATH=/home/jehc223/Retrieval-hate/scripts
export TERA_FIXTURES_REPORT_SHA256=f21b465e69ac11dc620dfdf9bc66e676cd6749bde65344f1e33762ed979a1fb5
export TERA_FIXTURE_CODE_SHA256=1cd6c48226345c91c7423c7c61e805f94c8d05c36492af814a1bc266491dfd36
scripts/tera_gate0/run_detached.sh tera_gate0_stage_c \
  python -m tera_gate0.run_gate0 \
    --config research-wiki/tera_gate0_frozen_config.json \
    --stages A,C \
    --confirmation none
```

Explicit `--config` is mandatory: the argparse default still names the pre-freeze
`…_frozen_config.draft.json` path, deliberately left uncorrected because the harness is
hash-frozen (appendix §10.1). The two exported digests populate the `fixtures_report_sha256`
and `fixture_code_sha256` manifest fields, which `run_detached.sh` does not set on its own.

`git_dirty` will be recorded as `true` (the harness and the wiki documents are untracked at
submission time, and this file is itself new); `git_diff_sha256` is hard-coded to `null` at
`run_gate0.py:1088`. Harness provenance rests on the verified
`package_aggregate_sha256` recorded in §0 of this file and on the two exported digests.
