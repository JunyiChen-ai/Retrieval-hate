# W2-A Stage-P′ Probe — INDEPENDENT VERDICT REVIEW (BINDING)

**Reviewer:** fresh, zero-prior-context, zero-stake independent verdict reviewer. **Date:** 2026-07-17.
**Scope:** READ-ONLY except this deliverable. NO GPU / NO SLURM / NO Modal. Frozen artifacts + probe record
NOT modified. NO push.
**Subject:** W2-A (transcript-first grounded retrieval key) Stage-P′ probe, raw record commit
`688ef874e0596058f9516ba0fe79481768c85350`.
**Posture:** adversarial toward wishful reading in BOTH directions (survive-bias AND kill-bias). Judged ONLY
against the pre-registered rules as written.

---

## BINDING OVERALL VERDICT

- **HateMM:** **DEAD** — the sole binding performance gate **K9 (conditional-info vs `Z_best` 8960-d)** returns
  `GROUNDED_DEAD_AT_ZBEST` (Δacc −0.0000, all three sub-conditions FAIL) on **valid machinery** (Fano 1.0,
  calib accZA 1.0, K2/K3 LIVE).
- **MHC-EN:** **DEAD** — K9 returns `GROUNDED_DEAD_AT_ZBEST` (Δacc −0.0038, all three sub-conditions FAIL) on
  valid machinery.
- **ROUTE (W2-A cross-modal grounded key):** **DEAD — outcome (d) "neither clears" (§6.6). The
  cross-modal-grounding cell is CLOSED.**
- **Branch classification:** **binding-K9-FAIL on BOTH datasets + K5-oracle-SURVIVE + advisory-raw-kNN-FAIL.**
  This is the "raw-FAIL + oracle-SURVIVE" family the team lead named, but the decisive fact is stronger: the
  route dies at the **sole binding performance adjudicator (K9)**, not merely at the demoted advisory raw bar,
  and the oracle-SURVIVE is necessary-not-sufficient and rescues nothing.
- **Lineage:** **ADMISSIBLE** as equivalent to one uninterrupted judged run (frozen probe + grounded-cache
  hashes pinned and matched; deterministic per-seed rng; 150/150/150 perms; point-arms cached-once;
  independently reproduced by the SLURM chunk-3 stdout).
- **Authorized next:** **nothing proceeds.** §11 head-training does NOT trigger (it was never authorized here
  and requires a K9 PASS that did not occur). Route closed; extracted-but-unscored test grounded keys are
  stranded at 0 test-touch.

---

## RULE-BY-RULE TABLE (prereg language verbatim · raw value · verdict)

Governing docs (frozen): `research-wiki/experiments/exp-w2a-grounded.md` (r1, hash `076bfa5e…`, §4/§5/§6/§12/§16),
`refine-logs/W2A_HASH_FREEZE_r2.md` (§16 constants), `refine-logs/W2A_PREREG_REVIEW.md` (r1 amendments source).
Raw values: `W2A_PROBE_RECORD.md` §3/§4/§5, cross-checked against `slurm/logs/w2a_probe_chunk_13212_3.log`
and `W2A_EXTRACTION_RECORD.md` (commit `c013884`).

### A. Stage-E′ extraction preconditions (must be satisfied for the probe to be judgeable)

| gate | prereg language (verbatim) | raw value | verdict |
|---|---|---|---|
| **K0 grid + vision-pad assert** | §12 K0: "grid gate: `n_vis` ≠ grid count, or vision-pad positions not a contiguous grid-count block, in either forward … → HALT" | extraction record: all 6 splits PASS gate 0; self_test PASS both datasets | **PASS** (no HALT) |
| **K1 G-recon-IMG** | §4 gate 1: "`img_recon` vs banked `img_feats[v]`: **cos ≥ 0.9999 AND max-abs ≤ 1e-3** for every non-guard video" | `grecon_cos_min` ∈ [0.9999995231628418, 0.9999997019767761]; `grecon_maxabs_max = 0.0` — all 6 splits | **PASS** (harness IS the banked forward) |
| **K2 GroundingLive** | §12 K2 / §16: "grounding-live: **present-set median** `cos(grd, ungrd_vis) ≥ 0.999` (transcript is a silent no-op) → §4 gate 2 → **VOID**" | present-median 0.9368 / 0.9475 (HateMM tr/dev), 0.9605 / 0.9609 (MHC tr/dev) — **all < 0.999**, `grounding_VOID=False` | **PASS — LIVE** (both datasets; NOT a silent no-op) |
| **K3 Placebo** | §12 K3: "placebo: **cross-video mismatched** transcript does not move `grd` (`cos ≥ 0.999`) → key reflects position/length, not content → §4 gate 3 → **VOID**" | placebo-median 0.9804 / 0.9812 (HateMM), 0.9711 / 0.9709 (MHC) — **all < 0.999**, `placebo_VOID=False` | **PASS — LIVE** (both datasets; content-sensitive) |

**Extraction-record cross-check (MANDATORY, satisfied):** the probe's own Stage-E′ gate read-back
(`W2A_PROBE_RECORD.md` §4) is **bit-consistent** with the extraction record `c013884` saved-lines (HateMM
grounding 0.9368/0.9475, placebo 0.9804/0.9812; MHC grounding 0.9605/0.9609, placebo 0.9711/0.9709). K2 and
K3 are **LIVE from the extraction gatelogs themselves**, on both datasets, on the memory splits that feed the
probe. The precondition for honoring any K9 conclusion is met: the grounding channel is real and
content-sensitive, so the K9 verdict is a genuine "adds no convertible info," not an un-interpretable no-op.

### B. Machine-validity + kill-switch

| gate | prereg language (verbatim) | raw value | verdict |
|---|---|---|---|
| **K4 Fano** | §6.3: "Vote acc **must reach ≥ 0.99** on both datasets, else the vote machine is VOID and **no negative verdict is acceptable**." | 1.0000 (HateMM), 1.0000 (MHC) | **PASS** — machine VALID ⇒ **a negative verdict IS acceptable** |
| **K5 oracle-ceiling kill-switch** | §6.4 / §12 K5: "If the oracle-ceiling paired **Δacc < +0.04** on **every** dataset … → **W2-A DEAD**, zero head GPU … The oracle ceiling is an upper bound and can **NEVER** be claimed as a result." | Δ(oracle−CONCAT) acc **+0.0635** (HateMM), **+0.0970** (MHC); chose-grd fraction 0.395 / 0.353 | **SURVIVES** (kill-switch does NOT fire) — but **necessary-not-sufficient**; authorizes nothing (see Ruling a) |
| K5 ordering-consistency | §6.4: "oracle Δ ≥ raw Δ; a raw Δ materially exceeding the oracle ⇒ construction bug" | oracle Δ +0.0635/+0.0970 ≥ raw Δ −0.0259/−0.0509 | **CONSISTENT** — proper upper bound, no construction-bug flag |

### C. K9 — the SOLE BINDING performance adjudicator (§5, §6.1 gate 7, §12 K9)

Verbatim rule: *"**C3-template conditional-info probe vs `Z_best`(8960-d): `grd` Δacc < +0.040 OR CI-lower ≤ 0
OR not > all ≥150 perm maxima — a grounded key CLIP-redundant against `Z_best` is DEAD (Amdt 1/2b, the SOLE
binding performance gate)**."* Triple rule (§5): (C1) Δacc ≥ **+0.040**; (C2) per-video-clustered bootstrap
CI-lower **> 0**; (C3) real **> all** ≥150 permutation maxima; label-oracle calibration `accZA ≈ 1.0` else
`MACHINERY_INVALID`.

| dataset | calib accZA | C1: Δacc ≥ +0.040 | C2: CI-lower > 0 | C3: real > all perm max | probe VERDICT | gate |
|---|---|---|---|---|---|---|
| **HateMM** (Z_best 8960) | 1.0000 (VALID) | −0.0000 → **False** | CI[−0.0052,+0.0049], lower ≤ 0 → **False** | perm max +0.0085; −0.0000 ≯ +0.0085 → **False** | `GROUNDED_DEAD_AT_ZBEST` | **FAIL** |
| **MHC** (Z_best 8960) | 1.0000 (VALID) | −0.0038 → **False** | CI[−0.0099,+0.0019], lower ≤ 0 → **False** | perm max +0.0076; −0.0038 ≯ +0.0076 → **False** | `GROUNDED_DEAD_AT_ZBEST` | **FAIL** |
| HateMM covered-rows-only (Amdt 5, n=802) | — | −0.0032 → **False** | CI[−0.0075,+0.0012], lower ≤ 0 → **False** | (secondary) | `GROUNDED_DEAD_AT_ZBEST` | **FAIL (secondary confirms)** |

**All three sub-conditions FAIL on BOTH datasets on VALID machinery (accZA=1.0, MACHINERY_INVALID not
triggered).** The pre-declared Amdt-5 covered-rows-only HateMM view (which removes the ~8% empty-transcript
dilution) also returns DEAD — the negative is not a dilution artifact. The Qwen-only-7168 secondary (HateMM
Δacc −0.0038 CI[−0.0092,+0.0014]; MHC +0.0032 CI[−0.0067,+0.0137], `SECONDARY_NO_PERM`) is explicitly
non-binding and is likewise null/CI-straddling; it cannot lift the verdict.

### D. Advisory gates (§6.5/§6.6/§7 — reported, NON-gating per Amdt 2b)

| gate | prereg language (verbatim) | raw value (HateMM) | verdict |
|---|---|---|---|
| **K6** raw kNN bar | §12 K6 (advisory): "raw HateMM kNN Δ(GROUNDED − CONCAT) acc < +0.05 OR mF1 < +0.05, or not beating CONCAT-PCA/CONCAT-α in sign — **corroborative, non-gating**" | Δacc −0.0259, ΔmF1 −0.0295; beat-PCA/α sign False | **BELOW (advisory FAIL)** — corroborates DEAD |
| **K7** perm null | §12 K7 (advisory): "observed kNN Δ ≤ 95th pct of the key-shuffle permutation null" | obs −0.0259 ≤ null-95th +0.0330 | **BELOW (advisory FAIL)** |
| **K7b** near-dup exclusion | §12 K7b: "GROUNDED advantage does not survive near-dup-excluded retrieval" | excluded Δacc −0.0259 (still negative) | **BELOW (advisory FAIL)** |
| **K8** bootstrap | §12 K8 (advisory): "bootstrap 5th-pct of paired kNN Δ crosses 0 (D3-fragile)" | 5th-pct −0.0470 (< 0) | **BELOW (advisory FAIL / D3-dead)** |
| **K10** partial | §12 K10: "MHC-EN fails while HateMM passes → mechanism real but binding gap not closed (honest partial)" | HateMM ALSO fails K9 | **NOT-APPLICABLE** — not outcome (b); both fail ⇒ outcome (d) |

MHC advisory arms are uniformly more negative (Δacc −0.0509, ΔmF1 −0.1125; bootstrap-5th −0.0843). Every
advisory arm on both datasets points the same direction as the binding K9: DEAD.

### E. Dataset rule (§6.6) → outcome assignment

Verbatim: *"**(a)** both clear → strongest … **(b)** HateMM clears, MHC-EN fails → … binding gap NOT closed …
**(c)** MHC-EN clears, HateMM fails → … **(d)** neither → **DEAD, cross-modal-grounding cell closed.** No
post-hoc dataset shopping."* Binding K9 FAILS on **both** ⇒ **outcome (d): DEAD, cross-modal-grounding cell
closed.**

---

## THE THREE EXPLICIT RULINGS REQUESTED

**(a) Does oracle-SURVIVE authorize anything? — NO. Necessary-not-sufficient only.** §6.4 is a **kill-switch**:
its only pre-declared action is to DECLARE DEAD when it *fails* ("Δacc < +0.04 on every dataset → W2-A DEAD").
Surviving it means the route was **not auto-killed at the oracle stage**; it does **not** license any progression.
The prereg forecloses reading the oracle as evidence-for-life in the strongest possible terms: *"The oracle
ceiling is an upper bound and can **NEVER** be claimed as a result."* Adversarial note (survive-bias guard):
the oracle's +0.0635/+0.0970 headroom is produced by **gold labels choosing, per query, which key to trust** —
an unrealizable ceiling. The binding K9 asks the realizable question (can a trained linear head, given
`Z_best`, extract anything from `grd`?) and answers Δacc ≈ 0 on valid machinery. The oracle gap is exactly the
label-leakage the "never a result" clause exists to quarantine. Oracle-SURVIVE rescues nothing.

**(b) Does any escalation / §11 head-training clause trigger? — NO.** §11 is gated verbatim: *"Only if Stage P′
survives:"* and §6.1/status-block: head-training is *"NOT authorized here (separate prereg behind the oracle
kill-switch)."* "Stage P′ survives" requires a K9 PASS (the sole binding performance gate). K9 FAILED on both
datasets ⇒ Stage P′ does **not** survive ⇒ §11 is **not** triggered. (It was independently un-authorized in this
prereg regardless of outcome.) No downstream GPU is authorized.

**(c) Branch classification (exact terms).** **K9-BINDING-FAIL on both datasets + K5-oracle-SURVIVE +
advisory-raw-kNN-FAIL → outcome (d) → route DEAD.** In the "raw-FAIL vs oracle-SURVIVE" framing: the advisory
raw kNN indeed FAILS and the oracle indeed SURVIVES, so this is *not* the "kill-switch fires" branch. But under
the amended prereg the raw kNN is only ADVISORY (Amdt 2b); the operative fact is that the **sole binding
adjudicator K9 also FAILS**, independently and on valid machinery, on **both** datasets — including the
empty-transcript-corrected covered-rows-only HateMM view. The route therefore dies at the binding performance
gate, and the oracle-SURVIVE (necessary-not-sufficient) does not lift it. This is the clean "C3-style
CLIP-redundancy null" the §13 honest-prior pre-declared as a most-likely informative outcome.

---

## PROVENANCE + LINEAGE-ADMISSIBILITY CHECKLIST

| check | requirement | finding | admissible? |
|---|---|---|---|
| Frozen-probe hash | probe = r2b freeze `af4a2f9f…` | echoed in-container **every chunk**; loop log line 8 + chunk-3 log line 12 both show `af4a2f9f5b35461173fd82c176bd52c6fc84bf8fc0d09736f938d38d8f6fe06d`; adapter aborts on mismatch | ✅ |
| Grounded-cache pinning (`grd_sha`) | checkpoint `_meta.grd_sha` = c013884 extraction `.pt` sha256s | checkpoint pins HateMM `1cae1f83…+41bda7de…`, MHC `9f8da7a1…+7c1a1a4f…` — I re-matched these against `W2A_EXTRACTION_RECORD.md` §4 manifest: **EXACT MATCH** (train+dev = memory splits). Test caches `23634bcb…`/`372640a3…` are **NOT** referenced (test never opened) | ✅ (+ zero test-touch confirmed by hash) |
| Per-seed rng determinism | container-independent | each perm seed `np.random.default_rng(CI_PERM_BASE+si)`; resume continues from `len(maxk)`; each perm's maxk is a pure function of (seed, cached point-arms) ⇒ accumulation across chunk boundaries = one run | ✅ |
| Point-arms cached-once | binding Δacc computed once, not per chunk | record + chunk log: point-arms computed once, cached; resume only extends the perm loop; the decision-point Δacc is a single deterministic computation | ✅ |
| Final perm counts | ≥150 permutation null (§16) | final checkpoint: `HateMM|Z_best_8960`=150, `HateMM|Z_best_covered`=150, `MHC|Z_best_8960`=150; Qwen-only cells `run_perm=False` = 0 by design (secondary/non-binding). Chunk-3 banner: `CI_NSEED=150` | ✅ (150 = pre-declared floor, on all binding cells) |
| Memory sizing / test-touch | train∪val only, N=851/629, test never opened (§5, §9) | chunk-3 log: "N4 fail-closed: train+dev_seen ONLY; expected memory {HateMM:851, MHC:629}"; probe memory N=851 / 629 | ✅ 0 test-touch |
| 2 timeout cancellations | no artifact mutation | both died at Modal ~3600s server cap; frozen probe + grd caches unchanged (hashes constant); each committed partial perms to the shared hash-pinned checkpoint | ✅ (interruptions cosmetic to the judged numbers) |
| `rm`-race resume | not clean-from-scratch | `modal volume rm` raced eventual-consistency; checkpoint persisted and was RESUMED. Because `_meta` pins probe_sha + grd_sha (both matched), a genuinely drifted checkpoint would have been rejected; the resumed lineage is the SAME frozen-artifact lineage. Record corrected the provenance honestly | ✅ (deterministic resume of the same pinned lineage) |
| Reaped driver | login-node reaping | client chain-driver reaped ~07:31Z (the recurring login=compute reaping mode); progress was committed pre-reap; re-run inside CPU-only SLURM 13212 (reap-proof) completed 3 chunks. No numeric consequence | ✅ |
| Independent stdout reproduction | record ↔ primary output | `slurm/logs/w2a_probe_chunk_13212_3.log` (the probe's own stdout) reproduces **every** mechanical-gate number in `W2A_PROBE_RECORD.md` §5 **verbatim** (Fano 1.0/1.0; OracleDacc 0.06345…/0.09697…; both K9 `GROUNDED_DEAD_AT_ZBEST`; RawDacc −0.02585…; null-95th 0.03301…; bootstrap-5th −0.04700…) | ✅ two-source agreement |

**Lineage ruling: ADMISSIBLE.** The chunked, twice-cancelled, once-`rm`-raced, once-reaped run is equivalent
to one uninterrupted judged run: the binding numbers are functions only of (frozen probe `af4a2f9f`, pinned
grounded caches matching c013884, deterministic seeds, cached-once point-arms), all of which are invariant
across the interruptions and hash-verified. The 150-perm null is complete on every binding cell.

**Provenance caveats (do not alter the ruling):**
1. **Primary results JSON not in-repo.** `/w2a_probe_results.json` and `/w2a_ci_ckpt.json` live on the Modal
   `rgcl-features` volume and were NOT committed. I could not open the primary JSON directly. This is mitigated
   to a non-issue by two independent in-repo sources that agree bit-for-bit: (i) the SLURM chunk-3 **stdout**
   (probe's own mechanical-gate print), and (ii) the extraction record's K2/K3 medians. Numeric-provenance
   discipline is satisfied by the stdout cross-check; I recommend committing the raw JSON to the repo for the
   permanent record, but its absence does not impugn the verdict.
2. **Cloud / CPU triage-tier (~1.4pt drift).** Per CLAUDE.md, Modal numbers are triage-only and never mixed
   into local tables. **This is prereg-sanctioned for THIS stage:** §10 + the hardware line declare Stage P′
   "CPU only / cloud-eligible on derived float caches," and it is a zero-test-touch features-only screen — not
   a paper-table number. See the drift ruling below.

---

## CLOUD-DRIFT CAVEAT RULING (per instruction: flag any decisive margin < ~1.4pt drift)

- **K9 HateMM:** Δacc −0.0000 vs bar +0.040 → **~0.040 below the bar**, ≫ ~0.014 drift. **Drift-robust FAIL.**
  C2 (CI-lower −0.0052 ≤ 0) and C3 (−0.0000 ≯ perm-max +0.0085) fail structurally, independent of any drift.
- **K9 MHC:** Δacc −0.0038 vs +0.040 → **~0.044 below the bar**, ≫ drift. **Drift-robust FAIL.** C2/C3 also
  fail structurally.
- **K5 oracle (necessary-not-sufficient, immaterial to the route):** HateMM +0.0635 clears the +0.04 bar by
  +0.0235 (> drift but same order); MHC +0.0970 clears by +0.0570 (≫ drift). Even if drift erased the HateMM
  oracle margin, the route outcome is unchanged (K5-kill → DEAD, or K5-survive → K9-fail → DEAD).
- **Advisory K6 HateMM:** Δacc −0.0259 is ~0.076 below the +0.05 advisory bar — drift-robust, and non-gating.

**No margin that decides the route verdict is within cloud drift.** The two datasets' binding K9 fails by ~1σ
of the *bar itself*, an order of magnitude beyond triage drift, and are additionally locked by the structural
C2/C3 failures (a permutation-null and a bootstrap CI, which drift cannot rescue). The DEAD ruling is
drift-robust.

---

## ADVERSARIAL BALANCE STATEMENT (both directions)

- **Survive-bias rejected:** the only positive signals are (i) the oracle ceiling (gold-leakage upper bound,
  prereg says "NEVER a result") and (ii) the GROUNDED_TEXT sensitivity arm (0.7873/0.7838), which §6.1/S2S-N3
  pre-declare "cannot rescue a failed binding adjudicator" and which is not the binding key. Neither is
  admissible as a pass. K9 fails on both datasets, including the dilution-corrected covered-rows view.
- **Kill-bias rejected:** the DEAD verdict is rendered on **valid, live machinery** — Fano 1.0 (so a negative
  verdict is prereg-acceptable, §6.3), calib accZA 1.0 (not MACHINERY_INVALID), and K2/K3 LIVE on both datasets
  (so this is a genuine "grounding works but adds nothing convertible over `Z_best`," not an un-interpretable
  no-op VOID). The route is not being killed on a broken instrument; it is killed on a working one that found no
  conditional information.

---

## WHAT IS AUTHORIZED NEXT

1. **Nothing proceeds.** Route W2-A is **DEAD**; the **cross-modal-grounding cell is CLOSED** (outcome (d)).
2. **§11 head-training: NOT authorized** (Stage P′ did not survive; K9 failed both).
3. **No further GPU / SLURM / Modal** on this route.
4. Bookkeeping (non-GPU, discretionary): record W2-A as a pre-registered negative in the graveyard
   (`directions_tried.json`) with epitaph "grounded transcript-first Qwen vision key: conditional-info Δacc ≈ 0
   vs `Z_best`(8960-d) on both datasets, oracle-survives (necessary-not-sufficient) — CLIP/marginal-redundant,
   C3-style null, outcome (d)"; and (recommended) commit the raw `w2a_probe_results.json` + final
   `w2a_ci_ckpt.json` from the Modal volume into the repo for the permanent numeric record. Extracted-but-
   unscored **test** grounded keys remain at **0 test-touch** and are now stranded (the formal stage they were
   cached for will not occur).

---

## PROVENANCE (verified this review)
- Prereg rules: `research-wiki/experiments/exp-w2a-grounded.md` §4/§5/§6.1–§6.6/§12/§16; `refine-logs/W2A_HASH_FREEZE_r2.md`
  (§16 constants, probe `af4a2f9f…`); `refine-logs/W2A_PREREG_REVIEW.md` (K5 kill-switch + Amdt 1/2b, r1).
- Raw numbers: `refine-logs/W2A_PROBE_RECORD.md` §3/§4/§5, commit `688ef874…`.
- Independent stdout reproduction: `slurm/logs/w2a_probe_chunk_13212_3.log` (lines 18–20 config; 44–62
  mechanical gates), `slurm/logs/w2a_probe_loop_13212.log` (frozen-probe hash + chunk perms-after 230→310→COMPLETE).
- K2/K3 LIVE + grounded-cache hashes: `refine-logs/W2A_EXTRACTION_RECORD.md` §2/§4/§5, commit `c013884`.
- Lineage: `refine-logs/W2A_CHUNK_LOG.md` (cancellations, rm-race, reap, SLURM 13212).
