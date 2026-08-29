# SAV (C2) F-G0/F-G1 — CONDITIONAL EXECUTION AUTHORIZATION

**Authorizer:** fresh, independent EXECUTION AUTHORIZER (ceremony role separate from prep,
reviewer, and smoke-executor). Read-only except this doc. **No SLURM job submitted, nothing
committed.**
**Date:** 2026-07-14 (NZST).
**Object of authorization:** ONE submission of `sbatch scripts/slurm/sav_f0.sbatch` (the full
F-G0 extraction → two-tier reproduction guard → F-G1 statistics-engine chain, single serial
lineage).

---

## VERDICT: **CONDITIONALLY AUTHORIZED** (effective on SMOKE_PASS + hash re-match; see §4).

The prereg → code-review → smoke chain is intact and internally consistent. This authorization
becomes effective the moment `SAV_F0_SMOKE_RECORD.md` records **SMOKE_PASS** (all four criteria,
incl. guard-preview cosines ≥ 0.999 per dataset) **and** the seven frozen hashes below still
match at submit time. No further agent round-trip is required between SMOKE_PASS and the single
submission.

One **non-blocking documentation note** (does not withhold authorization): the exp-sav-f0.md
front-matter `verdict:` field and §7 status still read `DRAFT-REV2-AWAITING-DELTA-CHECK`, i.e.
they were not bumped after the independent prereg reviewer's third-pass **APPROVED** delta-check.
The authoritative approvals are the two independent external artifacts (PREREG_REVIEW Rev-2
delta-check = APPROVED; CODE_REVIEW = APPROVED, 0 blocking); the code reviewer's authority line
already treats the design as "(Rev-2b, APPROVED)". This is a stale text field inside the frozen
doc — no design, gate, number, or code is affected. Flagged for a future non-authorization edit.

---

## 1. Chain verification (read + cited)

### 1.1 Prereg is Rev-2b and independently APPROVED
- `research-wiki/experiments/exp-sav-f0.md` §8 Revision history runs through
  **"### Rev-2b 2026-07-13 main-loop rulings: noise band 0.010 (A-line precedent),
  img-stream-only scope pinned"** (exp-sav-f0.md:541–560). Rev-2b pins the two load-bearing
  constants used downstream: `NOISE_BAND_ACC = 0.010 ⇒ PROJECTED_GAIN_BAR = 0.040`
  (exp-sav-f0.md:278–282) and `PROBE_STREAM = img` (exp-sav-f0.md:287–294, 553–557). ✅
- `refine-logs/SAV_F0_PREREG_REVIEW.md` ends with the **third-pass "Rev-2 DELTA-CHECK … VERDICT:
  APPROVED"** (PREREG_REVIEW.md:368–419): all residuals R1–R3 + Rec-1..3 verified landed as
  written (R2 stronger than asked); "No blocking items remain." The prior "Rev-1 RE-REVIEW"
  had pre-authorized APPROVED-on-delta-check (PREREG_REVIEW.md:259–266), and the delta-check
  discharged that pre-authorization. ✅

### 1.2 Code review APPROVED, 0 blocking; smoke prescription == smoke record's execution
- `refine-logs/SAV_F0_CODE_REVIEW.md` **VERDICT: APPROVED** (CODE_REVIEW.md:12), "No blocking
  items. Three non-blocking notes" (CODE_REVIEW.md:20, 206–219). Nine checklists re-verified
  against primary sources incl. the real transformers 4.49.0 modeling source (hook layout,
  Checklist 6), the local model config (geometry 28×28×128, Checklist 1), and a live end-to-end
  run of the statistics engine (Checklist 7). ✅
- Smoke prescription (CODE_REVIEW.md:228–231):
  `python scripts/analysis/sav_f0_extract.py --datasets HateMM,MHC,MHC_zh --splits train,val --limit 2`
  — **byte-for-byte identical** to what `SAV_F0_SMOKE_RECORD.md` is executing (SMOKE_RECORD.md:84,
  inside the `scripts/slurm/sav_f0_smoke.sbatch` body; and SMOKE_RECORD.md:57). Job **13058**
  (`sav_f0_smoke`), submitted once at 2026-07-13T23:00:51 NZST, status **WAITING**
  (`PENDING (JobHeldUser)`), no artifacts yet (SMOKE_RECORD.md:8–18). The four smoke pass criteria
  in the record (SMOKE_RECORD.md:89–99) reproduce the prescription's four criteria
  (CODE_REVIEW.md:241–249), incl. criterion 3 = PRIMARY-guard-preview cosines ≥ 0.999 per dataset
  for both `img_feats` and `text_feats`. ✅

### 1.3 Extraction-only smoke; guard/probe intentionally not smoked
The smoke exercises the real GPU entry point (extraction) only; the guard/probe assert full
`EXPECTED_COUNTS` by design and were exercised offline on synthetic caches running the real
functions (CODE_REVIEW.md:232–233, Checklist 7). The full `sbatch scripts/slurm/sav_f0.sbatch`
runs the complete extract → guard → F-G1 chain serially with fail-closed `jq` gates between
stages (sav_f0.sh:32–65). ✅

---

## 2. FREEZE — sha256 of the seven frozen entities (computed 2026-07-14, this authorization)

Any hash change after this authorization **VOIDS** it. The executor MUST re-hash all seven at
submit time and record the values; a mismatch on any one aborts the submission.

```
0a580a5db752e908d02d35ada72ae5b0a156f04b6115348d56221be92ed34d5e  scripts/analysis/sav_f0_common.py
c92ae952bfa73ebbc236599659921616fe8f9dff7cfe943aec5bf324e57fa776  scripts/analysis/sav_f0_extract.py
23ad8d41606dc2ceec4e25376d03bde1610919a4eadbdf89744f6c7ec81f88ff  scripts/analysis/sav_f0_guard.py
597101ef9d82a93e670520373ca04132b3b3e0d62e7caa564c6f901e826b98f9  scripts/analysis/sav_f0_probe.py
26c5cdedf614e244731fb94f1e9727d5055c233485d707cd3fa96eab9ee97b2c  scripts/wrappers/sav_f0.sh
197e7db77b11107ef66d95e3d6f7fe8e305db3d9a801517374426fb1368e74a2  scripts/slurm/sav_f0.sbatch
c65e40bf2a1a1789fe296321736e7dba76d04598609ebf15af5ffb4f2fedd11a  research-wiki/experiments/exp-sav-f0.md
```

Re-hash command (executor, at submit time):
```
cd /data/jehc223/RGCL && sha256sum \
  scripts/analysis/sav_f0_common.py scripts/analysis/sav_f0_extract.py \
  scripts/analysis/sav_f0_guard.py scripts/analysis/sav_f0_probe.py \
  scripts/wrappers/sav_f0.sh scripts/slurm/sav_f0.sbatch \
  research-wiki/experiments/exp-sav-f0.md
```

---

## 3. Sanity checks (all pass)

- **SLURM resources within caps.** `scripts/slurm/sav_f0.sbatch`: `--gres=gpu:a100:1` (1 GPU ≤ 2),
  `--cpus-per-task=16` (= 16 CPU cap), `--mem=96G` (≤ 128 GB). **No `--time`** (sav_f0.sbatch:3–9;
  the header comment states the omission is intentional per project policy). Same partition
  (`slurmpartition`) + `gpu:a100:1` as the known-good banked `gen_embed_mllm.sbatch`
  (CODE_REVIEW.md:189–190). ✅
- **Single-submit discipline stated.** sav_f0.sbatch:14 "SINGLE-SUBMIT-PER-LINEAGE ceremony: this
  sbatch runs the whole F-G0/F-G1 chain serially"; wrapper refuses any `RUN_ID != EXPECTED`
  (sav_f0.sh:22–25, exit 2; sbatch passes `RUN_ID=EXPECTED=SAV-F0-FG0-FG1`, sav_f0.sbatch:29–30). ✅
- **Warm-start-compatible with the smoke's artifacts.** The smoke writes its 12 per-video caches
  into the SAME dir the full run uses (`artifacts/sav_f0/extract/<ds>/<split>/<id>.pt`) — the
  extractor CLI has no RUN_ID/output-redirect flag (SMOKE_RECORD.md:32–41). Resumability evidence
  (verified live in code this pass):
  - **Per-video skip-if-exists resume:** `sav_f0_extract.py:305–309` —
    `op = C.extract_video_path(...); if op.exists(): skipped += 1; done += 1; continue`. The full
    run skips each pre-existing smoke cache and still counts it toward `done`.
  - **Atomic same-dir writes:** payloads via `C.atomic_torch_save` (sav_f0_extract.py:325, 355),
    which uses `tempfile.mkstemp(dir=<target parent>) + os.replace` — in-repo, never `$TMPDIR`
    (CODE_REVIEW.md Checklist 3). No force-delete on failure (sav_f0.sh:11–12).
  - **Manifest always rewritten at full count:** `sav_f0_extract.py:362–382` writes the split
    manifest unconditionally at the end of every `(dataset,split)` pass with `n = len(items)`,
    `n_expected = exp` (full EXPECTED_COUNTS when `--limit 0`), `complete = bool(done == len(items))`.
    So the smoke's transient `n=2` manifests are overwritten at full count, and **gate1**
    (`.complete == true and (.n == .n_expected)`, sav_f0.sh:40) passes on the warm start. Code
    review independently verified `EXPECTED_COUNTS` == real gt counts (744/107, 549/80, 579/78;
    CODE_REVIEW.md Checklist 1), so `len(items) == exp`. ✅

---

## 4. THE AUTHORIZATION (conditional; self-effecting on SMOKE_PASS)

**AUTHORIZED:** exactly **ONE** submission of

```
sbatch scripts/slurm/sav_f0.sbatch
```

by the smoke-executor agent (or a successor executor), **EFFECTIVE ONLY WHEN ALL of the
following hold at submit time:**

1. `refine-logs/SAV_F0_SMOKE_RECORD.md` records **verdict SMOKE_PASS** with **all four** pass
   criteria met — specifically incl. criterion 3, the **PRIMARY-guard-preview cosines ≥ 0.999
   per dataset** for both `img_pooled↔img_feats` and `text_pooled↔text_feats`
   (CODE_REVIEW.md:241–249 / SMOKE_RECORD.md:95–98).
2. All **seven frozen sha256 hashes in §2 still match** (executor re-hashes via the §2 command
   and records the values). Any mismatch → this authorization is VOID; do NOT submit.

No further authorizer round-trip is needed between SMOKE_PASS and this single submission.

### 4.1 Executor duties
- **Record** the returned job id + submit timestamp (and the seven re-verified hashes) in a **new**
  `refine-logs/SAV_F0_EXECUTION_RECORD.md`. Do not overwrite this authorization doc.
- **JobHeldUser = wait.** `PENDING (JobHeldUser)` auto-releases; **never force-release, never
  resubmit** (single-submission ceremony; holds can stall hours).
- **On terminal state, verify artifacts per the prereg F-G0/F-G1 outputs and report:**
  - extraction manifests complete & full-count (`artifacts/sav_f0/extract/<ds>/<split>/_manifest.json`,
    `.complete==true and .n==.n_expected`, gate1);
  - **guard JSON pass** for every dataset (`artifacts/sav_f0/guard/<ds>/guard.json`, `.pass==true`,
    gate2);
  - **probe verdict JSON present and parseable** (`artifacts/sav_f0/probe/verdict.json`,
    `.status=="COMPLETE"`, gate3; read `.verdict`).
- **Any FAILED / non-zero exit** (wrapper gate exits: 2 RUN_ID, 3 extraction-manifest, 4 guard
  PRIMARY, 5 verdict): **collect evidence (sacct terminal line, the `.out`/`.err` logs, whichever
  JSON was written), do NOT resubmit.** A fresh result-to-claim review decides the next step.

### 4.2 Decision-rule pointer (F-G1 verdict processing)
The F-G1 verdict is processed per the **exp-sav-f0.md pinned rules** (do NOT re-derive):
projected-gain **bar +0.040** (`+0.030 acc + 0.010 noise band`, exp-sav-f0.md:278–282), the
**example-level clustered bootstrap** CI-excludes-0 (exp-sav-f0.md:255–264, R1b), and **MDL /
holdout-log-loss codelength as the primary read** with the Fano bits→acc projection
(exp-sav-f0.md:228–242, R1c). PROCEED requires, at some swept k, MHC-EN ΔL>0 & CI-low>0 **and**
projected-gain>0.040 & CI-low>0, **and** HateMM no-harm (ΔL CI-high≥0 & Δacc CI-low≥−0.010)
(exp-sav-f0.md:268–270, 324–326; realised in `_k_pass_mhc`/`_noharm_hatemm`/`decide`,
CODE_REVIEW.md Checklist 7). **The F-G1 verdict is a conclusion-bearing result** and MUST go to an
**independent verdict reviewer** (project rule for conclusion-bearing tables) **before any F-G2
step.** This authorization covers the F-G1 submission only; it does NOT pre-authorize F-G2.

### 4.3 SMOKE_FAIL path
If the smoke records **SMOKE_FAIL** (e.g. criterion 3 cosine < 0.999 = pipeline drift): this
authorization is **VOID**. Do NOT submit the full chain. Route the evidence to a **fresh
result-to-claim review**, which decides re-scope/fix. No resubmission of the smoke either
(single-submission ceremony).

---

## 5. Provenance index (this authorization)
- Prereg (frozen): `research-wiki/experiments/exp-sav-f0.md` (Rev-2b; §8:432–560; F-G1 bar:278–282;
  probe stream:287–294; MDL/bootstrap:228–264; F-G1 decision:268–270,324–326).
- Prereg review (APPROVED): `refine-logs/SAV_F0_PREREG_REVIEW.md:368–419` (Rev-2 delta-check).
- Code review (APPROVED, 0 blocking): `refine-logs/SAV_F0_CODE_REVIEW.md:12,20,189–190,206–249`.
- Smoke record (WAITING, job 13058): `refine-logs/SAV_F0_SMOKE_RECORD.md:8–41,84,89–99`.
- Full-run sbatch/wrapper: `scripts/slurm/sav_f0.sbatch:3–36`, `scripts/wrappers/sav_f0.sh:22–65`.
- Resumability (live-read): `scripts/analysis/sav_f0_extract.py:305–309,325,355,362–382`.
- Caps: 2 GPU / 16 CPU / 128 GB, no `--time` (project policy, CLAUDE.md).
</content>
</invoke>
