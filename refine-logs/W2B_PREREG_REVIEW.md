# W2-B Probe-Design Pre-Registration Review — sub-clip set-matching on banked CLIP caches (cloud, features-only)

**Reviewer:** fresh zero-prior-context independent probe-design reviewer (read-only; CPU verification only;
NO GPU / NO SLURM / NO Modal submission; NO commits except this deliverable).
**Date:** 2026-07-15.
**Under review (commit `1f8265a`):** `refine-logs/W2B_FORENSIC_RECON.md` (sha256 `17f18bf5…e43b7e5`,
matches on-disk) + `refine-logs/W2B_PROBE_DESIGN.md` (sha256 `04785113…00a8bc8e`, matches the STEP-3 commit
message record).
**Context read to judge (not obey):** `refine-logs/S2S_PROBE_DESIGN.md`, `research-wiki/experiments/exp-s2s-r3.md`,
`refine-logs/S2S_PREREG_REVIEW.md` + `S2S_CODE_REVIEW.md` (the house bars the S2S versions cleared),
`research-wiki/ROUND3_CANDIDATES_WAVE2_2026-07-15.md §W2-B`, `autoresearch/goal_mllm_plus3/state/directions_tried.json`
(graveyard), `research-wiki/experiments/exp-seg-mode-ablation.md` (the Delta-1 close precedent),
`scripts/cloud/modal_probe_runner.py` (cloud runner), `REFLECTION §4` (G0-cond mandate).
**Method:** every cache claim re-derived directly by CPU `torch.load` (map_location=cpu, `CUDA_VISIBLE_DEVICES=""`);
the Modal volume/guard/entrypoints re-read from source; the Delta-1 non-isomorphism and P3/P11/P6
distinctions checked against the graveyard and the seg-mode ablation record; the S2S machinery template
diffed clause-by-clause against the W2-B adaptation for dropped guards.

---

## VERDICT: APPROVED-WITH-AMENDMENTS

Three blocking amendments (all cheap, mechanical, **none touches a number** — they are doc/pre-registration
clarifications, not method changes): **B1** make the video-level LOO / parent-exclusion structure explicit
and gate it on the future code review; **B2** reconcile the §5 KILL conjunction with the §4 K-budget rule so
the K4 primary is the sole survival-determining arm (K30 can never rescue); **B3** state that the oracle
kill-switch is computed on the K4 primary arm. Five non-blocking. The design is machinery-faithful to the
S2S template, the caches are exactly as the recon claims (I re-derived all of them), the cloud spec is
accurate against the live runner, the hash-freeze is clean, and the scope is honestly self-deprecating
(W2-B's own novelty = low; value = a ~$0 CLIP-encoder second witness that de-risks the S2S/don't-pool
family). No number is in dispute; the blocking items close pre-registration ambiguities, exactly the
APPROVED-WITH-AMENDMENTS pattern the S2S prereg followed.

**Parent-exclusion ruling (the flagged biggest false-PASS risk), stated up front:** the flat cache layout
`[V·K, D]` would be a catastrophic trivial-leakage surface IF the probe built a flat sub-clip bank (each
sub-clip a separate memory entry) and voted at sub-clip level — a query sub-clip would then retrieve its
K−1 sibling sub-clips (same parent, adjacent windows, near-identical, same inherited label). **The design
does NOT do this.** Following the S2S template it reshapes `g = subclip_img_feats.view(V, K, D)`
(`W2B_PROBE_DESIGN.md:51`) and scores **video-to-video** MeanMaxSim/POOLED (Q's K-set vs M's K-set) under
**video-level LOO** — a query video's own K sub-clips live only in its own set, which LOO holds out
entirely, so siblings are structurally un-retrievable. Parent-exclusion is therefore **subsumed by
video-level LOO, by construction**, and the pre-declared memory-size assert `len(memory) == 851/629`
(`:54-55`) is a **video-count** guard that would itself trip (→ 3404/2516) if a flat sub-clip bank were
built. This is correct AS DESIGNED. It is a blocking amendment only because it is currently *implicit*
(inherited from S2S) and the executable probe does not yet exist — B1 requires the design to state it in one
line and the future code review to verify it.

---

## 1. Independent cache verification (CPU `torch.load`, re-derived 2026-07-15) — recon §1 is EXACT

Every load-bearing claim in `W2B_FORENSIC_RECON.md §1` reproduces exactly. My numbers:

| cache | V | K | rows (=V·K) | V·K==rows | parent==arange(V)·K | count min/max | label const/parent | zero-norm rows | ids==pooled[0] same-order | per-vid lbl==pooled |
|---|---|---|---|---|---|---|---|---|---|---|
| HateMM train K4 | 744 | 4 | 2976 | ✅ | ✅ | 4/4 | ✅ | **4** | ✅ | ✅ |
| HateMM dev K4 | 107 | 4 | 428 | ✅ | ✅ | 4/4 | ✅ | 0 | ✅ | ✅ |
| HateMM test K4 | 215 | 4 | 860 | ✅ | ✅ | 4/4 | ✅ | 0 | ✅ | ✅ |
| HateMM train K30 | 744 | 30 | 22320 | ✅ | ✅ | 30/30 | ✅ | **30** | ✅ | ✅ |
| MHC-EN train K4 | 549 | 4 | 2196 | ✅ | ✅ | 4/4 | ✅ | 0 | ✅ | ✅ |
| MHC-EN dev K4 | 80 | 4 | 320 | ✅ | ✅ | 4/4 | ✅ | 0 | ✅ | ✅ |
| MHC-EN test K4 | 161 | 4 | 644 | ✅ | ✅ | 4/4 | ✅ | 0 | ✅ | ✅ |
| MHC-EN train K4_mm | 549 | 4 | 2196 | ✅ | ✅ | 4/4 | ✅ | 0 | ✅ | ✅ |

- **Feature norms ≈ 21–35 (mean ≈ 30), NOT L2-normalized at storage** — confirmed on every cache. The probe
  MUST normalize at score time with an eps floor (recon §1.1(d), `:46-47`; loader `:52`). Verified.
- **Zero-guard rows:** HateMM **train K4 = 4** and **train K30 = 30** zero-norm rows (= 1 undecodable video ×
  K); **all other splits = 0**. Exactly recon §1.4 (`:107-110`). The 1e-6 eps-floor + per-video
  `zero_guard = (g.norm(dim=-1)<1e-6).all(dim=1)` (`:53`) is necessary and specified.
- **`_mm` (MHC-EN train):** `subclip_txt_feats [2196,768]`; `has_text` **1562/2196 = 71.1%**; **all 549
  videos have ≥1 ASR sub-clip**; zero-text rows carry a **zero-norm** text vector (min txt norm = 0.000) —
  so the `has_text=False` mask (`:96-98`) is **necessary** (un-masked, cos(0,·) is a 0/0 NaN). Exactly recon
  §1.3/§1.4 (`:111-113`).
- **num_frames:** K4=16, K30=120 → 4 frames/sub-clip in both. Consistent with recon §1.1.
- **Availability partition:** K30 = HateMM-train-only; `_mm` = MHC/MHC_zh-train-only; HateClipSeg = test-only
  (no train memory, correctly declared unusable). Confirmed by directory listing.
- **Memory sizes:** train∪dev = HateMM **851** (744+107) / MHC-EN **629** (549+80) — **identical to S2S
  `EXPECTED_MEM`** because they are the same videos; the S2S N4 size-guard transfers verbatim as claimed.
  Train-only sensitivity arms: HateMM K30 = 744, MHC-EN `_mm` = 549. All verified.

**No train/test leakage in the sub-clip→parent mapping:** splits are separate files; labels are MIL-inherited
from the parent (constant within a video, verified); ids are index-identical to the pooled caches in the
same order. Parent indexing is sound (contiguous K-blocks). **Item-1 ruling: PASS, fully re-derived.**

---

## 2. Machinery faithfulness to the S2S template + parent-exclusion — item 2

I diffed every S2S guard against the W2-B adaptation. **No guard is silently dropped.**

| S2S guard (cleared in `S2S_CODE_REVIEW.md`) | W2-B location | ruling |
|---|---|---|
| N4 fail-closed no-test-touch (never open `test_seen`; assert memory size) | `:54-55`, §6 `:161-163` | ✅ present (851/629 primary; 744/549 train-only) |
| Zero-guard eps-floor + exclusion from near-dup | `:53`, `:125` | ✅ present; zero rows independently verified to exist |
| Rank-only sim-neutralized co-diagnostic + **B3 credit rule** (sign AND null-95th AND boot-5th>0) | `:83-86` | ✅ present, "verbatim from `s2s_probe.py` B3 fix" |
| Permutation null seeds 0..99, **same permutation both arms**, paired Δ preserved | `:115-118` | ✅ present ("shuffle sub-clip **sets** across videos" = correct video-level-set analog) |
| Bootstrap ≥1000, 5th-pct>0 else D3-FRAGILE | `:119` | ✅ present |
| Near-dup audit `pooled_cos≥0.995 OR MeanMaxSim≥0.995`, report 0.98/0.99/0.995 + single-unit max-cos, excluded-retrieval must survive | `:120-125` | ✅ present; 1/K-floor argument holds for K4 (0.25) and K30 (0.033) |
| Oracle A4 exact per-query selection, video-level gold only, tie-break, ordering expectation | `:88-92` | ✅ present, `t*=argmax_t (2y−1)·v_t`, memory keeps full sets (no double-dip) |
| Fano ±1 gold-label key, vote acc ≥0.99 both datasets else VOID | `:93-94` | ✅ present |
| Synthetic set-matching positive control (planted shared unit ⇒ MMS>POOLED else HALT) | §6 `:166-168` | ✅ present ("transfers verbatim T→K") |
| Determinism (CPU, fixed seeds, deterministic tie-break) | §6 `:164-165` | ✅ present (seed 20260714) |
| Vote reused not reimplemented (`compute_metrics_retrieval`, `use_sim=True`, topk=20) | `:61-62` | ✅ present |
| Extraction-correctness anchors (grid gate / temporal control / G-decomp / G-recon) | **N/A** | ✅ correctly absent — W2-B has **no fresh forward** (features-only on banked caches); correctness rests on the §1 integrity checks (re-derived above) + the PIPELINE-ANCHOR arm (`:78`) |

- **Fano/oracle calibration reaching headroom:** the label-oracle (`:88-92`) upper-bounds convertible
  structure and the +0.04 kill-switch (`:107-109`) gates on it; Fano (`:93-94`) is the machine-validity
  headroom check (≥0.99). Both present, mirroring the REFLECTION §4 null-as-distribution + calibration
  mandate.
- **Rank-only co-diagnostic present, credit rule pre-declared** (`:86`). Faithful.
- **No-test-read guard present** and, crucially, the pre-declared memory-size assert is a *video-count*
  (851/629), which doubles as a structural guard against a flat sub-clip bank (see B1).

**Parent-exclusion — the explicit ruling.** See the up-front box. In one line: **video-level LOO subsumes
parent-exclusion; there is no flat sub-clip retrieval bank; the query's own K sub-clips are confined to its
LOO-held-out set; the `len(memory)==851/629` video-count assert would itself catch a flat-bank
implementation.** Correct as designed. **Blocking B1** only because it is currently implicit and the code
does not yet exist. **Item-2 ruling: PASS with blocking B1.**

**Residual (non-blocking, N2):** sub-clips are *coarse* (4-frame means). A query and a **distinct** memory
video that share one viral/re-uploaded segment produce **one** near-identical sub-clip pair, contributing
≈1/K to MeanMaxSim — **below** the 0.995 global-near-dup flag (which needs ~all K to match). So a
*partial-overlap* re-upload is not flagged. This is the **same** surface S2S carries (approved there),
mitigated identically by reporting the single-sub-clip max-cos distribution (`:121`). At K=4 the effect is
larger than S2S's T=4 frame-groups only in that a sub-clip is a semantically coarser unit; still not
blocking, but the design should name it.

---

## 3. Kill bars — item 3

- **Raw anchor** (`:110-112`): HateMM K4 primary paired **Δacc ≥ +0.05 AND ΔmF1 ≥ +0.05**, rank-only
  corroborated — the same P3-priced (~1.7×) shrinkage bar S2S uses, AND-rule (not shoppable). ✅
- **MHC-EN survival** (`:113-114`): **+0.03/+0.03** — the goal's cross-dataset increment, weakest defensible
  "could matter." ✅
- **Oracle headroom** (`:107-109`): DEAD-family if oracle Δ(oracle−POOLED) **< +0.04 on every dataset**. ✅
- **Single-vs-both dataset rule** (`:126-136`): four mutually-exclusive pre-declared rows
  (a DEAD-family / b BOTH / c SINGLE / d NEGATIVE), no OR-ing beyond them. Internally consistent with the
  N5 ordering (oracle Δ ≥ raw Δ, so "oracle dead everywhere" ⇒ raw dead everywhere ⇒ (a)). ✅
- **K-budget** (`:138-141`): K4 primary decides; K30/`_mm` are matched-memory sensitivity contrasts that
  "cannot rescue a failed K4 primary" (mirror S2S N3). ✅ **as stated in §4** — but see **B2**: the §5 KILL
  restatement contradicts it.

**B2 (BLOCKING) — the §5 KILL conjunction re-imports K30 into a K4-primary decision, contradicting the
§4 K-budget rule.** §5 (`:146-149`) declares KILL = `HateMM K4 <+0.05/+0.05 AND MHC-EN <+0.03/+0.03 AND the
K30-train-only contrast on HateMM also <+0.05/+0.05`. Because it is a conjunction *including* the K30
sensitivity arm, a scenario where **K4 primary fails on both datasets but K30 clears +0.05** makes the KILL
literally FALSE — yet §4 gate 8 (`:126-136`) still assigns outcome (a)/(d) NEGATIVE (neither K4 raw bar
cleared), and the §4 K-budget rule says K30 "cannot rescue." The two sections give different-looking
decision rules for the same state — precisely the ambiguous-gating-rule class the house blocked as B5-A1.
**Required (cheap, doc-only, no number):** state explicitly that the **K4 primary is the sole
survival-determining arm** (outcomes a/b/c/d), and the K30 term in §5 may only **modulate the breadth** of a
negative (K4-specific vs family-wide-across-granularity) — it can **never** convert a failed K4 primary into
a survival. Simplest fix: demote the K30 clause in §5 from a KILL *conjunct* to a reported breadth-modifier.

**Gameability elsewhere:** none found. The AND-rules, the fixed four-row dataset rule, and the "sensitivity
cannot rescue primary" statement (§4) close the arm/metric-shopping surfaces. **Item-3 ruling: PASS with
blocking B2.**

---

## 4. Cloud execution — item 4 (verified against the live runner)

`scripts/cloud/modal_probe_runner.py` re-read:
- **Volume** `VOLUME_NAME = "rgcl-features"` (`:31`) — matches the design (`:173`). The recon's correction of
  the stale `rgcl-feats` is right.
- **Sync guard (features-only):** `sync(dataset)` (`:124`) walks `base.rglob("*")` (`:134`) and calls
  `assert_uploadable(f)` **fail-loud before any upload** (`:137`); `_ALLOWED_EXTS = {.pt,.jsonl,.json,.csv,
  .npy,.txt}` (`:85`) + a media/`video/`-dir refusal (`:77,:93`). The sub-clip caches are `.pt` files not
  under a `video/` dir → they pass the guard and are **included automatically** by the whole-dir rglob, as
  the design claims (`:174-182`). ✅
- **Mount + run:** volume at `/root/data` (`:170`); `scripts/analysis/` and `src/` auto-mounted (`:68-70`);
  `run_probe_cpu` is CPU-default (`:171`). The design's `--data_root` portability requirement (`:190`) and
  the "probe must live in `scripts/analysis/`" constraint are correct against the mount.
- **Triage-only stamping:** §7 G-repro note (`:195-197`) states cloud results are **exploratory triage
  only**; a paper number would be re-run locally on the table's hardware; a *negative family-de-risk* cloud
  number is sufficient to decline GPU. ✅ Stated, and correct (a CPU CLIP kNN negative is low-stakes and does
  not enter a formal table).

**Item-4 ruling: PASS.** (The probe script `scripts/analysis/w2b_probe.py` does **not** yet exist —
correctly deferred to a separate code-review + hash-freeze gate, §11 `:237-239`, mirroring S2S A5.)

---

## 5. Scope honesty — item 5

The design is honestly, even self-deprecatingly, scoped: the one-paragraph framing (`:11-21`) and the recon
bottom-line (`:9-20`) both state W2-B's standalone novelty is **low** (mechanism ≈ S2S), its value is a
**~$0 family de-risker + CLIP-encoder second witness**, and — the sharper honesty the scout under-stated —
that the **trained** version of essentially this idea (Delta-1 / seg-mode) was already **killed
high-confidence**, so the prior on a *positive* is **weak** and the most likely outcome is a **clean kill
that corroborates Delta-1 at ~$0**. I verified the Delta-1 precedent (`exp-seg-mode-ablation.md`: verdict
`no`, confidence `high`, MHClip-only, "noisy MIL pseudo-positives without gold spans", anti-repeat note) and
the non-isomorphism claim (W2-B trains nothing, mines no pseudo-positives, isolates the training-signal
confound Delta-1's failure was blamed on — the same non-isomorphism logic that makes G0-cond probes legal).
The graveyard distinctions (**vs P3** = P3 re-weighted one pooled vector by an MLLM *score*, W2-B keeps the
*set* and changes the *metric*, no score; **vs P11** = weak-sup *training* labels on the same K30 cache, W2-B
does frozen kNN; **vs P6** = P6 *scores* segments for localization, W2-B *matches* segment sets for
classification) all hold against `directions_tried.json`. **No overclaim.**

The interpretation is pre-declared in §4.8 (`:126-136`): a W2-B negative revises the S2S/C2/W2-C priors
**down** (bounded by the pre-declared CLIP<Qwen asymmetry, threat #1 `:214`), a W2-B positive revises S2S
**up** and escalates to the novelty-carrying Qwen-token version. That is the load-bearing forward-direction
matrix (W2-B result → S2S GPU decision) W2-B exists to produce. **Non-blocking N5:** consolidate the
"a CLIP-negative does NOT *veto* S2S, only lowers its prior" point (currently split across §1 and threat #1)
into the §4.8 interpretation rows so the joint reading is in one place. **Item-5 ruling: PASS.**

---

## 6. Other false-PASS / false-KILL surfaces — item 6

- **B3 (BLOCKING) — oracle kill-switch arm not pinned to K4.** §4 gate 2 (`:107-109`) / §3.8 (`:88-92`) do
  not state that the oracle Δ is computed on the **K4 primary** arm. A K30 oracle (30-way max) inflates the
  ceiling, making the "< +0.04 everywhere ⇒ DEAD" switch *less* likely to fire — a potential **false-survival
  of the early kill** (the route would then have to be caught by the downstream K4 raw bar, which it would
  be, so the final verdict is safe, but a pre-registered gating statistic should not be ambiguous — B5-A1
  class). **Required (cheap, one line):** pin the oracle kill-switch to the **K4 primary** arm (the K30
  oracle, if reported at all, is a sensitivity number only).
- **N3 (non-blocking) — code-level guards to carry into the future probe.** The design reuses `s2s_probe.py`
  machinery but the executable does not exist yet. The code-review gate MUST re-verify: (i) the **NEG_INF
  filter** (S2S NB-a: `row[topk_idx] > NEG_INF/2` before the vote) survives into the near-dup-excluded vote;
  (ii) the **`_mm` text normalization eps-guards** the verified zero-norm empty-text rows (else 0/0 NaN);
  (iii) the vote's diagonal exclusion is **video-level** (LOO holds out the whole query set), tying back to
  B1.
- **N4 (non-blocking) — honesty refinement on MHC-EN.** The recon/design say the trained precedent "already
  sign-flipped on the same data" (`:16-20`, `:20`). Precisely: on the **binding MHC-EN** dataset the trained
  full-seg was weakly **positive** (+0.015 F1); it was **MHC_zh** that sign-flipped (−0.066). So the
  pessimism the framing imports onto the binding dataset is slightly overstated — MHC-EN was a marginal
  positive-below-bar, not a negative. Refine the one-liner so the binding-dataset prior is not over-pessimized.
- **N1 (non-blocking) — MHC_zh optional arm not independently re-derived here.** I verified HateMM + MHC-EN
  (the primary + binding-gap) caches exhaustively; the OPTIONAL MHC_zh K4 arm (train∪dev = 657) I did not
  torch.load (it is non-primary, out-of-goal, and cannot gate the verdict). If MHC_zh is ever promoted past
  "language-robustness color," re-derive it first.

**Item-6 ruling: PASS with blocking B3.**

---

## 7. Required amendments

### BLOCKING (fold into the design + re-hash before cloud execution is authorized; none touches a number)
- **B1 — make video-level LOO / parent-exclusion explicit + gate it on the code review.** Add one line to
  §1/§2 stating the memory is **V video-level sets** (`view(V,K,D)`), retrieval is **video-to-video**, LOO
  holds out the **whole query set**, there is **no flat sub-clip bank**, and the `len(memory)==851/629`
  assert is a **video-count** guard against exactly that mistake. Add to §10/§11 that the future
  `w2b_probe.py` code review MUST verify (a) the contiguous-parent reshape, (b) video-level LOO diagonal
  exclusion, (c) no sub-clip-level retrieval index anywhere.
- **B2 — reconcile §5 KILL with the §4 K-budget rule.** Make the **K4 primary the sole survival-determining
  arm**; the K30 term in the §5 KILL may only modulate a negative's *breadth*, never rescue a failed K4
  primary. (Simplest: demote the K30 clause from a KILL conjunct to a reported breadth-modifier.)
- **B3 — pin the oracle kill-switch to the K4 primary arm** (§3.8/§4 gate 2), so a K30-inflated ceiling
  cannot bypass the early DEAD-family kill.

### NON-BLOCKING (fold at execution / code-review; do not re-open the design authorization)
- **N1** — independently re-derive the MHC_zh optional arm only if it is ever promoted past robustness color.
- **N2** — name the coarse-sub-clip partial-overlap re-upload surface (one shared segment ≈ 1/K, below the
  0.995 flag) as an explicit residual, discharged by the reported single-sub-clip max-cos distribution.
- **N3** — code-review gate must carry the NEG_INF filter, the `_mm` zero-text eps-guard, and video-level
  diagonal exclusion into `w2b_probe.py`.
- **N4** — refine the "already sign-flipped on the same data" framing: on the binding MHC-EN dataset the
  trained precedent was a marginal positive-below-bar (+0.015 F1); the sign-flip was MHC_zh.
- **N5** — consolidate "a CLIP-negative lowers but does not veto the S2S prior (CLIP<Qwen)" into the §4.8
  interpretation rows.

---

## 8. Conditional authorization — CLOUD PROBE EXECUTION of W2-B (Stage P only)

**Granted, conditional on B1–B3 folded + both docs re-hashed AND a separate code-review + hash-freeze of the
authored `scripts/analysis/w2b_probe.py` before any Modal dispatch.** Scope: the zero-training,
zero-test-touch, CPU, features-only sub-clip set-matching probe on banked CLIP caches. Terms:

1. **Amend first.** Fold B1–B3 into `W2B_PROBE_DESIGN.md` (N1–N5 SHOULD be folded now; N2/N3/N5 are
   probe-correctness-relevant), re-compute and re-record both doc sha256 (design self-hash in the commit
   message, recon in §11), per the existing ceremony.
2. **Author + code-review the probe script — a SEPARATE gate.** `w2b_probe.py` does not exist. This
   authorization does **not** cover its submission until it is written to the amended spec, independently
   code-reviewed (the S2S machinery reuse + the B1 video-level-LOO / no-flat-bank check + the N3 guards +
   real-vote reuse via `compute_metrics_retrieval`), and sha256-pinned. Quick code check + a synthetic-data
   dry run (the planted-shared-unit self-test must show MMS>POOLED) precede any cloud run.
3. **Sync first, features-only.** `modal run …::sync --dataset HateMM` then `--dataset MHC` (optional
   `--dataset MHC_zh`); the `assert_uploadable` guard is fail-loud — verify the log shows only allowlisted
   `.pt`/label files uploaded, nothing under `video/`.
4. **Run on Modal CPU** via `…::run --script scripts/analysis/w2b_probe.py --args "--data_root /root/data
   --datasets HateMM,MHC"`. No GPU. Deterministic (`CUDA_VISIBLE_DEVICES=""`, fixed seeds).
5. **Raw-only transcription.** The executor writes `W2B_PROBE_RESULTS.md` (raw per-arm AUC/acc/mF1, paired Δ
   sim-weighted AND rank-only, Fano, oracle ceiling, near-dup audit at 0.98/0.99/0.995 + single-sub-clip
   max-cos, null percentiles, bootstrap percentiles, K4/K30/`_mm` contrasts, per dataset) with **NO
   pass/fail interpretation**; the machine JSON carries the `mechanical_gate_check` block stamped
   "pre-registered arithmetic, NOT the binding verdict."
6. **Triage-only.** Cloud numbers **never enter a formal table**; a negative family-de-risk update is the
   sanctioned use; any number destined for a paper is re-run locally on the table's hardware.
7. **Independent verdict review after.** A fresh reviewer renders the binding verdict against the four
   pre-declared dataset-rule rows (post-B2, K4-primary-determined).

**Out of scope:** any downstream head-training stage; any GPU; any test-label use beyond the pre-declared
Fano/oracle ceiling (train∪val, video-level); treating the oracle or any sensitivity arm (K30, `_mm`,
Chamfer, WITH-TEXT, MHC_zh) as the reported result; any config/state/CLAUDE.md mutation.

---

## 9. Bottom line

The W2-B design is a faithful, honestly-scoped, ~$0 features-only adaptation of the house-approved S2S
machinery to banked CLIP sub-clip caches whose reality I fully re-derived. It de-risks the don't-pool family
with a CLIP second witness and correctly foregrounds the Delta-1 anti-repeat flag with a legitimate
zero-training non-isomorphism. The three blocking amendments are pre-registration clarifications (video-level
LOO explicitness, §5-vs-§4 kill reconciliation, oracle-on-K4) — cheap, mechanical, none touching a number.
**APPROVED-WITH-AMENDMENTS**; cloud execution conditionally authorized per §8, gated behind the amendments
and the separate probe-script code-review + hash-freeze.

---

## 10. r1 CODE RE-CHECK (commit `bc1810b`, 2026-07-15) — VERDICT: CLOUD EXECUTION CLEARED

The §8 conditions are met: B1–B3 + N1–N5 are folded into the design, and `scripts/analysis/w2b_probe.py`
(794 lines) is authored. This section is the independent code review that §8 condition 2 required. **Method:**
full static gate walk with concrete values; deferred-import audit against the HateVideo env; real-vote
reuse verified line-for-line against `src/utils/metrics.py`; and the entire probe pipeline
(build_matrices → run_vote → rank-only → ASYM → fano → oracle_ceiling → near_dup_audit → permutation_null →
bootstrap → mechanical_gate_check → JSON) **driven end-to-end on synthetic data** so it exercised the REAL
`compute_metrics_retrieval`, plus a targeted B2 stress test. `py_compile` clean.

### 10.1 Hash-freeze — ALL MATCH on-disk
| artifact | frozen sha256 (design §11 r1 / commit `bc1810b`) | on-disk | match |
|---|---|---|---|
| `refine-logs/W2B_FORENSIC_RECON.md` | `17f18bf5…e43b7e5` (UNCHANGED vs r0) | `17f18bf5…e43b7e5` | ✅ |
| `refine-logs/W2B_PROBE_DESIGN.md` (r1) | `d8421dfe…5b3ffd` (commit msg) | `d8421dfe…5b3ffd` | ✅ |
| `scripts/analysis/w2b_probe.py` | `d22aac02…d3a15776` | `d22aac02…d3a15776` | ✅ |

### 10.2 The three blocking amendments — HONORED IN CODE
- **B1 (video-level LOO / parent-exclusion) — DONE.** Loader reshapes `g = subclip_img_feats.view(V,K,D)`
  (`w2b_probe.py:115`) behind a **contiguous-parent HALT** (`:111-114`, `subclip_parent ==
  repeat_interleave(arange(V),K)` AND `V*K==rows`) and a **video-count assert** (`:137-139`,
  `V != expected_mem → raise`, 851/629 primary / 744/549 train-only). The vote runs on **`[V,V]` score
  matrices** with **`np.fill_diagonal(St, NEG_INF)`** (`:218`) = video-level LOO (whole query set held out).
  The flat `[V·K,D]` tensor `G` (`:169`) is used **only** to form the `[V,K,V,K]` pairwise cos tensor, which
  is **immediately reduced to `[V,V]`** by `max(axis=3).mean(axis=1)` (`:171`) — there is **no retrieval or
  vote over the `V·K` axis anywhere** (grep-confirmed: every arm's score matrix is `[V,V]`). The header
  docstring (`:10-15`) states this. A flat-bank mistake would trip the video-count assert (→ 3404/2516).
- **B2 (K4 = sole survival arm; K30 breadth-only) — DONE, and stress-tested.** `mechanical_gate_check`
  derives `hate_pass`/`mhc_pass` **only** from `by_ds[...]["primary"]` (the K4 arm, `:610-614`); the a/b/c/d
  outcome uses **only** `oracle_all_below + hate_pass + mhc_pass` (`:615-623`); K30 (`:626-631`) and `_mm`
  (`:632-637`) are appended as `"REPORTED"` breadth/modality rows carrying the note *"cannot rescue K4."* I
  **stress-tested** it: forcing K4 to fail on both datasets (Δ=0, rank-uncorroborated), oracle set to
  survive (so outcome (a) is not trivially taken), and a K30 report that **clears +0.20** → the dataset rule
  returned **`(d) NEGATIVE`, not survival.** The §5-vs-§4 ambiguity I flagged in the prereg review is
  resolved in code: K30 is structurally outside the survival logic.
- **B3 (oracle kill-switch on K4 primary) — DONE.** `oracle_ceiling(M["C"], …)` is called with the **K4
  primary** Sff tensor (`:470`); `oracle_all_below = all(r["oracle"]["d_acc"] < 0.04 for r in primaries)`
  ranges over the K4 primaries only (`:584`); the gate is labeled `OracleKillSwitch(K4-primary,…)` (`:588`).
  K30/`_mm` have no oracle arm, so a K30-inflated ceiling cannot bypass the early kill.

### 10.3 House code-review lessons — PASS
- **Gate walk (fail-loud).** N4 path guards refuse any non-`{train,dev_seen}` split at the path builder
  (`:83,:90`); video-count assert (`:137`); zero-guard `(g.norm<1e-6).all(dim=1)` (`:140`) — the 1
  HateMM-train guard video reduces to a ~0 vector in **every** arm (POOLED/SET/ASYM rows all ≈0 → identical
  arbitrary top-20 → **cancels in the paired Δ**) and is excluded from the near-dup audit (`:296`); `_mm`
  `has_text` mask **and** eps-L2norm double-guard the verified zero-norm empty-text rows (`:195,:199-200`);
  NEG_INF post-topk filter + degenerate-retrieval raise (`:227-229`) inherited from S2S NB-a. No bare/broad
  excepts anywhere (the only guard is the fail-loud missing-cache raise, `:103-104`).
- **Deferred-import audit.** `from utils.metrics import compute_metrics_retrieval` (`:56`, after the
  `sys.path` insert of `<repo>/src`) and the deferred `from sklearn.metrics import f1_score` (`:392`) both
  resolve in the HateVideo env (metrics.py's top-level wandb/torchmetrics/etc. present, per the S2S audit).
- **Real-vote reuse — verified line-for-line.** `run_vote` (`:236-238`) calls
  `compute_metrics_retrieval(logging_dict, torch.tensor(labels), majority_voting="arithmetic", topk=k,
  use_sim=True)` and unpacks `acc, roc, pre, rec, f1, votes, _lab, macro` — matching metrics.py's
  `return acc, roc, pre, recall, f1, list_majority_voted, labels, macro` (`metrics.py:320`) in order;
  `macro["macro_f1"]` exists (`metrics.py:312`); the dict keys `retrieved_label`/`retrieved_scores`
  (`:233-234`) match `metrics.py:264-265`; each score is a `np.float64` (has `.item()`, required by
  `metrics.py:266`). The oracle margin inline (`:251` `weight = arange(1,k+1)[::-1]`, `wsum`) reproduces
  metrics.py's `weight = np.arange(1,topk+1)[::-1]` (`metrics.py:229-231`) exactly. Vote never reimplemented.
- **Same-perm-both-arms null (N1).** `permutation_null` (`:323-326`) applies `ix = np.ix_(perm, perm)` (one
  `default_rng(s).permutation(V)` per seed 0..99) to `mms`, `spool`, AND `asym` identically; the rank-only
  variants reuse the same `mms_s`/`spool_s` (`:330-331`), labels fixed. ASYM is evaluated under the same
  per-seed permutation (`:329`). N1-compliant.
- **Deterministic seeds.** Global `20260714` (`:737-738`), null seeds `0..99` via `default_rng(s)`, bootstrap
  `seed=20260714` (`:391`), per-frame null `default_rng(10000+s)`; hard CPU via `CUDA_VISIBLE_DEVICES=""`
  (`:736`) and deterministic index tie-break (`_tiebreak`, `:160-162`).
- **Near-dup 0.995 + 1/K floor.** `NEAR_DUP_THRESH=0.995`; audit reports pooled/mms/**maxframe** (single
  sub-clip) at 0.98/0.99/0.995 (`:298-301`); flag = `(pooled≥0.995)|(mms≥0.995)`, symmetrized (`:302-303`),
  excluded via `run_vote(exclude=flag)` (`:475-476`).
- **Dry-run result.** Drove the full pipeline on synthetic data: Fano `1.0000` (machine-validity converts),
  the `_mm` path produced **no NaN** (eps+mask work), JSON serialized, the oracle≥raw ordering held, and the
  B2 stress returned `(d) NEGATIVE`. `py_compile` clean.

### 10.4 Modal execution — CLEARED with two non-blocking execution terms
- **Sync is correct.** `sync(dataset)` rglob + `assert_uploadable` allowlist auto-includes the subclip `.pt`
  (verified earlier). Whether they are *already* on `rgcl-features` is immaterial — **execution step 0**
  (idempotent, ~$0) re-runs sync.
- **Runner reuse is correct.** The probe lives in `scripts/analysis/` (mounted at `/root/scripts/analysis`),
  takes `--data_root` (→ `/root/data` on Modal), imports the vote from `/root/src`. `_execute` does
  `features.reload()` and streams the subprocess stdout to the Modal logs (so the config echo + the
  mechanical-gate lines are visible live).
- **TERM-1 (output retrieval — mandatory before dispatch).** The image mounts only `/root/src`,
  `/root/scripts/analysis`, and the volume `/root/data`; there is **no `/root/refine-logs`**, so the probe's
  **default** `--out_md/--out_json` (`_REPO_ROOT/refine-logs/…` → `/root/refine-logs/…`) would
  **FileNotFoundError at the final write**, and the runner does **no `features.commit()`**. Stdout carries
  the mechanical-gate arithmetic but **not** the full raw per-arm table. So the run MUST redirect outputs to
  the mounted volume and retrieve them, e.g.
  `--args "--data_root /root/data --datasets HateMM,MHC --out_md /root/data/W2B_PROBE_RESULTS.md
  --out_json /root/data/w2b_probe_results.json"`, then `modal volume get rgcl-features W2B_PROBE_RESULTS.md
  …` (ensuring a volume commit; if the runner does not auto-commit, add `features.commit()` after
  `_execute`, or capture stdout with the full raw dump). This is **plumbing, not a probe-code defect** — the
  code writes to correct local paths and runs fine locally.
- **TERM-2 (K30 memory — advisory).** The K30 sensitivity arm forms a `22320×22320` all-pairs tensor
  (~2 GB) → ~2–4 GB transient RAM. The **K4 primary + `_mm`** arms are tiny. Confirm the Modal CPU container
  memory, or run cloud with `--k30_sensitivity 0` and do K30 locally. K30 is non-survival-determining (B2),
  so dropping it on cloud costs nothing for the verdict.

### 10.5 Verdict — CLOUD EXECUTION CLEARED
The probe code is machinery-faithful (real vote, same-perm null, rank-only B3 credit rule, NB-a filter,
oracle A4, Fano, near-dup, synthetic self-test), the three blocking amendments are honored **and
stress-verified in code**, the hash-freeze matches, and py_compile + the synthetic end-to-end dry run pass.
No defect blocks execution. **Execution terms:** step-0 `sync HateMM` + `sync MHC` (features-only guard) →
single `modal run …::run --script scripts/analysis/w2b_probe.py` on **CPU** with outputs redirected to the
`/root/data` volume (TERM-1) and K30 memory handled (TERM-2) → **raw-only** transcription (the executor
writes `W2B_PROBE_RESULTS.md`/`.json` with NO pass/fail interpretation; the `mechanical_gate_check` block is
stamped "NOT the binding verdict") → **independent verdict review** renders the ruling against the four
pre-declared K4-primary dataset-rule rows. Triage-only: cloud numbers never enter a formal table.
