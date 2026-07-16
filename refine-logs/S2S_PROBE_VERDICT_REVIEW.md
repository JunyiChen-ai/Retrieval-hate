# S2S Stage-P G0-cond probe — INDEPENDENT BINDING VERDICT REVIEW

**Reviewer:** independent, zero-context verdict reviewer (no stake in outcome; adversarial toward
wishful interpretation). **Date:** 2026-07-16. **Repo HEAD at review:** `c939c13`.

**Subject.** S2S (segment-set frameset) Stage-P G0-cond probe, executed on Modal cloud CPU
(app `ap-iE3OEkUra7f37UuzDk2uz3`), raw record committed at `c939c13`
(`refine-logs/S2S_PROBE_RECORD.md`, `refine-logs/s2s_probe_results_cloud.json`).

**Frozen decision authority.** `research-wiki/experiments/exp-s2s-r3.md` (prereg, r4;
sha256 `64a489f2…765276` per design §10 r4 table) + `refine-logs/S2S_PROBE_DESIGN.md` §5/§6/§7.
I judge **only** against the pre-registered rules as written. Where the prereg is silent or ambiguous
on a point the verdict depends on, I rule on the conservative (kill-leaning) reading and say so.

---

## 0. Bottom line (binding)

| scope | verdict |
|---|---|
| **HateMM (primary dataset)** | **KILL** — does NOT clear the §6.5 raw survival bar (fails all five sub-conditions). |
| **MHC-EN (binding-gap co-primary)** | **KILL** — paired Δ is *negative* (SET worse than POOLED); does NOT clear under any reading. |
| **Dataset rule §6.6 outcome** | **(d) neither clears → DEAD, retrieval-object family closed.** |
| **S2S route (symmetric SET / MeanMaxSim)** | **KILL.** |
| **Folded C2 / ASYM cell** | **KILL** — dies with the family; the "escalate-to-§11-asym" tag is REFUTED (see §4). |

**No head GPU is authorized. No §11 downstream stage is authorized. The retrieval-object ("don't-pool")
family — S2S symmetric SET *and* its C2/ASYM multi-view cousin — is closed by this screen.**

The oracle-ceiling kill-switch (§6.4) did *not* fire (headroom exists), but that is a **necessary, not
sufficient** condition: §6.4 states the oracle ceiling "can **NEVER** be claimed as a result," and the
route is killed one gate later at the raw survival bar (§6.5, gate 5). Passing gate 4 while failing
gate 5 = the exact "ceiling exists, raw effect does not convert" failure §6.5 was written to catch
(the graveyard's "probe passes, training goes flat," repeated ≥4×).

---

## 1. Pre-declared decision rules (quoted verbatim from the frozen prereg)

**RAW BAR — §6.5 (`exp-s2s-r3.md:324`), the survival test:**
> **RAW BAR (binding).** On the **primary dataset (HateMM)**, mean paired **Δacc ≥ +0.05 AND mean
> paired Δmacro-F1 ≥ +0.05** (MeanMaxSim vs POOLED), with the bootstrap 5th-pct of the paired Δ **> 0**
> (D3), the observed Δ **above the 95th pct of the permutation null** (§6.6), **and (r1: A2) the
> rank-only sim-neutralized arm corroborating the sign + significance** (an uncorroborated Δ is a
> sim-scaling artifact, not a mechanism effect).

**ORACLE KILL-SWITCH — §6.4 (`exp-s2s-r3.md:314`):**
> **KILL-SWITCH (binding).** If the **oracle-ceiling** paired Δacc **< +0.04** on **every** dataset,
> then pooling was **not** discarding convertible alignment structure — the whole "don't-pool" family
> (S2S + its C2 multi-view cousin) is **DEAD** … The oracle ceiling is an upper bound and can **NEVER**
> be claimed as a result.

**DATASET RULE — §6.6 (`exp-s2s-r3.md:347-358`):**
> **(a) HateMM clears + MHC-EN clears** → strongest … **(b) HateMM clears, MHC-EN fails** → mechanism
> is **real** … **(c) MHC-EN clears, HateMM fails** → advances the goal … **(d) neither clears** →
> DEAD, retrieval-object family closed. No post-hoc dataset shopping: these four rows are fixed now.

**C2 / ASYM FOLD — §5 (`exp-s2s-r3.md:254-260`), the escalation clause the record invokes:**
> **Pre-declared C2 kill logic:** (a) if S2S's oracle Δacc < +0.04 on **every** dataset (the §6.4
> kill-switch fires), the whole "don't-pool" family — S2S **and** ASYM — is **DEAD together**, no
> separate ASYM adjudication; (b) if S2S's symmetric SET survives (oracle did not fire), ASYM is **dead
> unless it beats symmetric SET on acc AND macro-F1 (paired) on ≥1 dataset** — otherwise asymmetric
> multi-view memory adds nothing over S2S's symmetric operator, and a beating ASYM would **escalate only
> as the asymmetric arm of the §11 downstream stage (never a standalone route)**.
> (Identical language in `S2S_PROBE_DESIGN.md:289-294`.)

**§11 AUTHORIZATION GATE — §7 (`exp-s2s-r3.md:383-385`):**
> Only outcome (a)/(b)/(c) that also clears gates 0–6 authorizes drafting the **downstream head-training
> formal pre-registration (§11)** … NOT authorized here.

**SENSITIVITY-CANNOT-RESCUE — §7 N3 (`exp-s2s-r3.md:378-381`):**
> The sensitivity arms (Chamfer, WITH-TEXT, 16-frame, near-dup-excluded) **cannot rescue a failed
> primary** … no OR-ing beyond the four fixed dataset-rule rows.

---

## 2. Rule-by-rule table (prereg threshold + raw value + verdict)

Gate order per §7. Raw values verbatim from `s2s_probe_results_cloud.json` (`c939c13`).

### Gate chain (Stage-E → Stage-P), both datasets

| # | gate (prereg) | threshold | HateMM raw | MHC raw | verdict |
|---|---|---|---|---|---|
| 0a′ | causal-prefix onset-invariance (§7 gate 0, K0) | prefix cos ≥0.999, changed diverge | 1.0000/1.0000; changed max 0.9273 | idem | **PASS** (Stage-E job 13189/13182) |
| 0b | grid-consistency (§4, K0) | `n_vis`==grid, `T`==grid_t | silent-pass, T=4 all splits | idem | **PASS** |
| 1 | G-decomp residual (§4, K1) | ≤ 1e-5 | 5.96e-08 | 5.96e-08 | **PASS** |
| 2 | G-recon vs banked (§4, K2) | cos ≥0.9999 AND maxabs ≤1e-3 | cos ≥0.9999995, maxabs 0.0 | idem | **PASS** (bit-exact) |
| 3 | Fano ±1 gold-key acc (§6.3, K3) | ≥ 0.99 | 1.0000 | 1.0000 | **PASS** (vote machine valid) |
| 4 | Oracle kill-switch (§6.4, K4) | DEAD iff oracle Δacc <+0.04 on **every** ds | oracle Δacc **+0.0917** | oracle Δacc **+0.1399** | **DOES NOT FIRE** → survives gate 4 (necessary, not sufficient; ceiling never claimable) |

### Gate 5 — RAW survival bar (§6.5), primary dataset HateMM — the decisive gate

| sub-condition (prereg §6.5) | threshold | HateMM raw | verdict |
|---|---|---|---|
| mean paired Δacc (SET−POOLED) | ≥ +0.05 | **+0.0035** | **FAIL** (~14× under bar) |
| mean paired Δmacro-F1 | ≥ +0.05 | **+0.0003** | **FAIL** (~170× under bar) |
| bootstrap 5th-pct of paired Δacc | > 0 | **−0.0106** | **FAIL** (D3-fragile, crosses 0) |
| observed Δacc vs 95th-pct permutation null | obs > null-95 | +0.0035 vs **+0.0189** | **FAIL** (inside null) |
| observed ΔmF1 vs 95th-pct permutation null | obs > null-95 | +0.0003 vs **+0.0298** | **FAIL** (inside null) |
| rank-only (A2) corroborates sign + significance | True | **False** (sign T, null F, boot F) | **FAIL** |

**Gate 5 = FAIL on HateMM (all six sub-conditions).** Kill-switches K5, K5b, K6, K7 all fire on the
primary dataset. This alone kills the route.

### Gate 6 / dataset assignment

| # | gate | threshold | HateMM | MHC | verdict |
|---|---|---|---|---|---|
| 6 | near-dup-excluded SET advantage survives (§5, A3, K6b) | excluded-Δ > 0 | +0.0082 (>0) | −0.0397 (<0) | **N/A** — moot: gate 5 already failed; §7 N3 forbids a sensitivity arm rescuing a failed primary |
| — | MHC "clears" test (§6.6) | same survival test, Δ>0 minimum | — | Δacc **−0.0397**, ΔmF1 **−0.0231** | **FAIL** (wrong sign; SET *worse* than POOLED) |
| 7 | dataset rule (§6.6) | assign (a)/(b)/(c)/(d) | does not clear | does not clear | **outcome (d): DEAD, family closed** |

Note the near-dup HateMM excluded-Δ is +0.0082 (>0) — the JSON's `NearDupExclSurvives=ABOVE`. This is
**not** a survival: A3/K6b guards a *passing* SET against duplicate-rediscovery; per §7 (N3) it cannot
convert a failed primary (+0.0082 is itself ~6× under the +0.05 bar and within the permutation null).

---

## 3. The oracle ceiling did not rescue the route — why gate 4 ≠ pass

The record's mechanical block shows `OracleKillSwitch(all-datasets) = SURVIVES` and both
`OracleDacc … = ABOVE`. **This is correctly computed and correctly is NOT a pass.** §6.4 makes the
oracle a one-way *kill* switch: oracle < +0.04 everywhere ⇒ DEAD; oracle ≥ +0.04 ⇒ *not killed at gate
4*, nothing more. §6.4 is explicit the ceiling "can **NEVER** be claimed as a result," and §6.5 is a
**separate, additional** survival requirement ("Survival **additionally** requires the **raw** …
effect to clear a bar …"). The observed pattern — large oracle headroom (+0.09 / +0.14) but a
flat/negative raw Δ (+0.0035 / −0.0397) — is exactly the scenario §6.5's ~1.7× pessimism factor was
pre-registered to reject: the discriminative frame *exists* (oracle can find it with gold), but the
data-driven MeanMaxSim operator *cannot* recover it without the gold selector, so pooling was
effectively lossless for the achievable retrieval. **Route KILL stands.**

---

## 4. C2 / ASYM escalation — the record's "SURVIVES(escalate-to-§11-asym)" is REFUTED

The raw record's mechanical arithmetic ends with
`C2 route adjudication (ASYM beats SET on >=1 dataset) → SURVIVES(escalate-to-§11-asym)`. The executor
correctly labels this "**NOT a verdict**." Judged against the frozen clause, it does **not** authorize
any escalation. Three independent grounds, any one sufficient:

**(i) The escalation clause's precondition is FALSE.** §5 clause (b) triggers only "if S2S's symmetric
SET **survives**." Symmetric SET did **not** survive: it failed the §6.5 raw bar on HateMM (all six
sub-conditions) and is *negative* on MHC → outcome (d). "Survive/survives" is the prereg's **defined
term** for passing the raw survival test (§6.5 is titled "Raw-effect **survival** bar"; §7 gate 5 is
"survival test"; §6.6 "clears" is its synonym). The probe implemented the parenthetical "(oracle did
not fire)" **as if it were the definition** of "survives" — i.e. it gated ASYM escalation on the oracle
kill-switch alone and never on symmetric-SET survival. That is a misreading: the parenthetical is a
**necessary co-condition**, not a redefinition; reading it as the definition contradicts the clause's
own consequent and §7/§11. Under the mandated conservative reading, clause (b)'s precondition is unmet
and its escalation branch is never entered.

**(ii) Even granting the MHC beat, the consequent has nowhere to escalate.** The clause restricts a
beating ASYM to "escalate **only as the asymmetric arm of the §11 downstream stage (never a standalone
route)**." §7 authorizes the §11 downstream stage **only** for "outcome (a)/(b)/(c) that also clears
gates 0–6." The outcome is **(d)** → **there is no §11 downstream stage** for ASYM to be the asymmetric
arm of. The escalation target does not exist.

**(iii) The MHC "beat" is not a real effect anyway.** `asym_beats_set=True` on MHC rests on
Δ(ASYM−SET) acc **+0.0143**, mF1 **+0.0145**. But: (a) ASYM (0.6773 acc) is itself **below** POOLED
(0.7027) on MHC — ASYM "beats" SET only because SET (0.6630) sank *further* below the pooled incumbent;
the entire don't-pool family loses to POOLED on MHC. (b) The beat is **not significant**:
`asym_vs_set_obs_gt_p95=False` (+0.0143 < null-95th **+0.0159**) and `asym_vs_set_boot_dacc_p5 = −0.0079`
(crosses 0). (c) The +0.0143 / +0.0145 margin is **smaller than the documented ~1.4pt (≈0.014)
cross-hardware cloud drift** — within-drift (see §5). The prereg is silent on drift/borderline handling
for the ASYM beat; conservative reading ⇒ not a credited beat.

**Structural note (prereg gap, ruled conservatively).** §5 clauses (a)/(b) partition on
oracle-fired / oracle-not-fired and *implicitly assume* "oracle did not fire ⇒ SET survives." The
actual cell — oracle headroom present **but** symmetric SET flat/negative (outcome d) — is not cleanly
covered by either clause under the defined meaning of "survives." Per the instruction to rule
conservatively where the prereg is silent, **C2/ASYM dies with the family** (clause (a)'s spirit:
"the whole don't-pool family … DEAD together"). **C2/ASYM: KILL. No standalone route, no §11 escalation.**

---

## 5. Cloud-drift caveat (CLAUDE.md triage policy)

All numbers are **Modal cloud CPU triage** (~1.4pt documented cross-hardware drift); per CLAUDE.md they
are triage-only and **never** mixed into a local/paper table, and any formal stage must re-run locally
under G-repro. Impact on this verdict:

- The **primary failure is not marginal** and is drift-immune: HateMM Δacc +0.0035 is ~14× under the
  +0.05 bar; MHC Δacc −0.0397 is the wrong sign. No plausible ±1.4pt drift converts either into a
  clear. The kill is robust to drift.
- The **only within-drift number** in the whole record is the ASYM-beats-SET-on-MHC margin
  (+0.0143 acc / +0.0145 mF1 < ≈0.014 drift). I flag it explicitly as within-drift; the prereg says
  nothing about drift handling for this comparison, so conservatively it is not a credited beat — and
  it is blocked regardless by §4(i)/(ii).
- Separately: because these are cloud triage numbers, they could not license a formal escalation *even
  if* a gate had passed (formal validation is local-SLURM-only per CLAUDE.md). Moot here — verdict is KILL.

---

## 6. Provenance checklist

| item | status | evidence |
|---|---|---|
| Frozen probe hash matches freeze | ✅ | `s2s_probe.py` on-disk **`141a0441…2301b3`** == r4 design §10 table (unchanged since r3 ASYM fold) == in-container adapter hash-gate (record §1) |
| Adapter shim = plumbing-only | ✅ (per record) | `s2s_probe_cloud_adapter.py` `48e5b375…` = hash-gate + symlink `{data,src}` + `runpy` frozen probe UNCHANGED; mirrors approved W2-A adapter `fb609d4b…` (record §1). Frozen probe verified byte-identical in-container, so the shim cannot alter probe logic. |
| 0 test-touch (N=851/629 train+val only) | ✅ | N4 fail-closed: memory N=851 (HateMM) / 629 (MHC) = train∪val exactly; `test_seen` never constructed/opened (record §1, JSON `expected_mem`) |
| Prereg config parity | ✅ | in-container `topk=20`, `null_seeds=100` (0..99), `n_boot=1000`, `n_perframe_null=100` == frozen design (record §1, JSON `meta`) |
| Gate chain feeding probe was GREEN | ✅ | Stage-E job **13189** (`S2S_EXTRACTION_RECORD.md`, HEAD `cc3d90e`/`77ed845`): all 4 HARD gates GREEN over full N (0a′/0b/G-decomp 5.96e-08/G-recon maxabs 0.0 bit-exact); N counts == prereg (744/107/215, 549/80/161). Smoke-3 job **13182** (`S2S_SMOKE3_RECORD.md`) GREEN on ≥1 real video/dataset. |
| Fresh forward == banked cache | ✅ | G-recon maxabs_max = 0.0 on all 1855 non-guard videos (extraction record §3) — representations the probe consumed are the certified banked forward |
| Commit reconciliation | ✅ | Task cited `c939c13` = the probe RAW-record commit (verified HEAD at review). Record's internal "Repo HEAD at run: `cc3d90e`" = the Stage-E extraction-record commit (HEAD when the cloud probe executed); `c939c13` is its child that lands the probe record. Consistent, no discrepancy. |

**One honest limit:** I verified the adapter is plumbing-only from the record's description and from the
fact that the frozen probe's own sha256 was re-verified inside the container (so the scored code is the
reviewed code); I did not line-by-line re-audit the 3.5 KB adapter source. Given the in-container
hash-gate on the frozen probe, this does not affect the verdict.

---

## 7. Kill-switches triggered (pre-declared, §13 "what-would-kill-this")

| killer | fires? | evidence |
|---|---|---|
| K4 oracle Δacc <+0.04 every dataset | NO | oracle +0.0917 / +0.1399 (route not killed *at gate 4*) |
| **K5** raw HateMM Δacc <+0.05 OR ΔmF1 <+0.05 | **YES** | +0.0035 / +0.0003 |
| **K5b** rank-only does not corroborate | **YES** | corroborates=False (null F, boot F) |
| **K6** obs Δ ≤ 95th-pct permutation null | **YES** | +0.0035 ≤ +0.0189 (acc); +0.0003 ≤ +0.0298 (mF1) |
| **K7** bootstrap 5th-pct crosses 0 | **YES** | −0.0106 (D3-fragile) |
| K8 MHC fails while HateMM passes (honest partial) | N/A | HateMM also fails ⇒ not (b), it is (d) |

Four pre-declared killers fire on the primary dataset; MHC is negative outright. Convergent KILL.

---

## 8. What is authorized now (per the prereg)

- **Nothing further on this route.** Outcome (d) → no §11 downstream head-training pre-registration, no
  head GPU, for either S2S symmetric SET or the C2/ASYM cell.
- **The retrieval-object ("don't-pool") family is closed** by this G0-cond screen: set-to-set /
  late-interaction retrieval over frozen Qwen frame-group tokens does **not** beat pooled-cosine
  retrieval on HateMM (flat, within-null) or MHC-EN (negative), and the gold-oracle ceiling that exists
  is **not convertible** by the data-driven operator. Record as a pre-registered negative in the
  graveyard (per §6.4 "exhaustion re-confirmed for the retrieval-object cell").
- The frozen artifacts (frame-set caches incl. the untouched `test_seen` sets, extraction gatelogs) may
  be retained for audit; **no test scoring is authorized.**

---

**Binding verdict:** **HateMM = KILL · MHC-EN = KILL · dataset-rule outcome (d) · S2S route = KILL ·
C2/ASYM = KILL (no §11 escalation).** Governing escalation clause: `exp-s2s-r3.md` §5
(`:254-260`) — a beating ASYM "would escalate only as the asymmetric arm of the §11 downstream stage
(never a standalone route)," and §7 (`:383-385`) authorizes §11 only for outcome (a)/(b)/(c) clearing
gates 0–6; outcome is (d), so no §11 stage exists and the escalation is void. Reviewed at `c939c13`.
