# B5 Pre-Registration Review — operating-point conversion of the frozen-Qwen MHC-ZH ranking advantage

**Reviewer:** fresh zero-prior-context independent pre-registration reviewer (read-only; CPU
verification only; NO GPU / NO SLURM / NO commits except this deliverable).
**Date:** 2026-07-14.
**Under review:** `research-wiki/experiments/exp-conv-zh-b5.md` (status `DRAFT-UNREVIEWED`) +
`refine-logs/B5_PROBE_DESIGN.md` (executable probe spec, DRAFT-UNREVIEWED).
**Context read (to judge, not to obey):** `refine-logs/EXHAUSTION_AUDIT_2026-07-14.md` §1/§7,
`refine-logs/B1_VERDICT_REVIEW.md`, `refine-logs/B3_PREREG_REVIEW.md`,
`research-wiki/REFLECTION_mllm_integration_failures.md` §4.
**Method:** every load-bearing number re-parsed independently from the six primary raw trainlogs
`slurm/logs/enc3s_MHC_zh_*_13115.trainlog` (fresh `\r`/`\n`-split parser, val-selection = epoch ≥
warmup 5 max Val_Retrieval acc, roc tie-break); code facts re-read from `src/`; checkpoints, cached
features, and split balance verified by direct filesystem / `torch.load` inspection.

**VERDICT: APPROVED-WITH-AMENDMENTS.** Two BLOCKING amendments (A1 kill-switch wording, A2 dev-side
G-repro anchor); eight non-blocking. Conditional authorization for the **PROBE STAGE ONLY** granted in
§5. The design is numerically exact, code-faithful, fair-paired, and honestly scoped; the two blocking
items close a governing-rule ambiguity and a one-sided hard gate — both cheap to fix.

---

## 0. Checklist result summary

| # | Checklist item | Ruling |
|---|---|---|
| 1 | Fair pairing (calibrated-vs-calibrated everywhere, incl. each-arm-own-oracle kill-switch) | **PASS** |
| 2 | Circularity / metric-shopping (single pre-justified primary; deterministic tie-break; grid) | **PASS** (non-blk A3) |
| 3 | Kill-switch soundness (oracle gate, AND-eligibility, protocol handling, oracle-never-claimed, ledger) | **PASS w/ BLOCKING A1** |
| 4 | Honest-preview gate (is it a test-touch? peeking impossible? frozen-before-test?) | **PASS** (non-blk A8) |
| 5 | G-repro hard gate (4dp reproduce deployed numbers; HALT logic; fallback) | **PASS w/ BLOCKING A2** |
| 6 | D3 guards (≥1000 bootstrap → distribution; τ stability; marginal-language ruling present) | **PASS** (non-blk A10) |
| 7 | Stat/code reality (vote = continuous rank-weighted signed cosine sum; determinism) | **PASS** (non-blk A9) |
| 8 | Veto/scope (no MLLM aux, no aux gold, single-dataset, no OCR, no ensembles; novelty honest; ledger) | **PASS** |
| 9 | Missed false-PASS / false-KILL surfaces | 5 findings → A4/A5/A6/A7 (non-blk) + A1 (blk) |

---

## 1. Numeric & code verification (independent re-derivation — all load-bearing facts EXACT)

**1.1 G-repro anchors, both protocols — all 12 match to 4 dp.** My fresh re-parse of the six 13115
trainlogs reproduces every cell of `B5_PROBE_DESIGN.md` §4 (Protocol B `:189-194`, Protocol A
`:200-205`) exactly on macroF1 / acc / roc:

| arm·seed | final e29 (mF1/acc/roc) | val-sel (epoch) (mF1/acc/roc) |
|---|---|---|
| CLIP 0 | 0.7706/0.8054/0.8382 | e29 0.7706/0.8054/0.8382 |
| CLIP 1 | 0.7542/0.8054/0.8342 | e28 0.7579/0.8054/0.8346 |
| CLIP 2 | 0.7913/0.8322/0.8444 | e25 0.7742/0.8121/0.8419 |
| Qwen 0 | 0.7864/0.8188/0.8906 | e22 0.7412/0.7919/0.8838 |
| Qwen 1 | 0.7759/0.8054/0.8951 | e25 0.7871/0.8121/0.8874 |
| Qwen 2 | 0.7514/0.7852/0.8806 | e28 0.7759/0.8054/0.8940 |

All 6 val-selection epochs (Qwen 22/25/28, CLIP 29/28/25) independently reproduced under the
warmup-5 / val-acc / roc-tie-break rule — matches `B5_PROBE_DESIGN.md` §2.1 `:57-62` and
`B1_VERDICT_REVIEW.md:35-40,43-44`.

**1.2 Anomaly means — the exact numbers the review request flagged, CONFIRMED.** Final-epoch 3-seed
means: **CLIP acc 0.8143, roc 0.8389; Qwen acc 0.8031, roc 0.8888** — identical to the designer's
claim and to prereg §2 `:55-59`. Derived Δ: Δacc −0.0112, Δroc +0.0499 (prereg `:57`). roc−acc
calibration gap: CLIP 0.0246→0.025 (prereg `:55`), Qwen 0.0857→**0.086** (prereg `:56`; the audit's
"0.085" at `EXHAUSTION_AUDIT:50` is a looser round — the prereg's 0.086 is the more precise value).
The 3.4× asymmetry is real. The load-bearing anomaly is genuine and correctly transcribed.

**1.3 Splits & class balance — EXACT.** `torch.load` on
`data/CLIP_Embedding/MHC_zh/{dev_seen,test_seen}_Qwen2.5-VL-7B-Instruct_HF.pt`: **dev = 78 (28 pos,
0.3590), test = 149 (45 pos, 0.3020)** — matches prereg §2 `:75` and audit `:83,:305`. The 30/70
imbalance that motivates the class-blind-cut suboptimality is confirmed; the 78-dev size that bounds
the D3 tax is confirmed.

**1.4 Checkpoints — all 12 recoverable, ZERO-GPU claim holds.** All six run dirs present under
`logging/Retrieval/MHC_zh/RAC_video_archive_seeds/`; each `ckpt/` holds 30 epoch checkpoints
(e0..e29). Spot-checked required heads exist with the exact float suffixes in §2.1:
Qwen s0 `epoch_model_22_0.8205128205128205.pt` + `epoch_model_29_0.782051282051282.pt`;
CLIP s2 `epoch_model_25_0.8205128205128205.pt` + `epoch_model_29_0.7948717948717948.pt`. The
zero-GPU CPU-replay premise is sound.

**1.5 Code facts — all TRUE (`src/utils/metrics.py`, `src/model/evaluate_rac.py`, `src/run_rac.py`).**
- Vote: `metrics.py:284` `np.sum(retrieved_labels_map*weight[:length])/np.sum(weight[:length])` with
  `retrieved_labels_map=(2·label−1)·sim` (`:268-270`) and `weight=np.arange(1,21)[::-1]=[20..1]`
  (`:229-231`). = **continuous rank-weighted signed cosine sum**. The prereg's "vote-granularity
  correction" (§2 `:78-84`: continuous, NOT 21-level) is CORRECT and supersedes the brief's 21-level
  assumption — a genuine catch, since `arithmetic`+`use_sim` produces a real-valued vote.
- Deployed cut: `metrics.py:300` `(sigmoid(vote)>=0.5)` ⇔ `vote>=0` (use_sim=True). ✓
- roc on the continuous vote: `metrics.py:294`. ✓ (so AUC is rank-based — load-bearing for A2/§2.5.)
- macro-F1: `metrics.py:309` `f1_score(...,average='macro',zero_division=0)` — identical to the probe's
  `metrics_at` (`B5_PROBE_DESIGN.md:144`). ✓
- Return position: `metrics.py:320` `return acc,roc,pre,recall,f1,list_majority_voted,labels,macro` —
  6th = per-video continuous vote, 8th = macro dict; the probe's unpacking (`:123-126`) is correct.
- faiss: `evaluate_rac.py:412` `IndexFlatIP` (exact brute-force), CPU branch `:423-430`
  (`normalize_L2`+`add`+`search`) — deterministic; both 13115 and the probe use **CPU faiss**, so faiss
  build is NOT a divergence source. faiss searches over `all_feats` = the **head-projected embeddings**
  (`evaluate_rac.py:353,388` `model(...,return_embed=True)`), so the ONE device-dependent step is the
  head forward (CPU-probe vs CUDA-13115), correctly identified and gated (see §2.7).

Nothing in the prereg's numeric or code claims is wrong. The two blocking amendments are about gate
completeness and rule wording, not about any incorrect number.

---

## 2. Per-checklist rulings

### Item 1 — FAIR PAIRING · **PASS**
Calibrated-vs-calibrated is enforced in **every** comparison path, and the deployed arm is quarantined
to G-repro sanity only:
- Formal Δ = (Qwen_calibrated − CLIP_calibrated), both arms the identical §5 procedure (prereg §4
  `:121-128`; §7 "Application" `:270-271`).
- Kill-switch: each arm uses **its own** test-optimal threshold (prereg §6.4 `:229-231`;
  `B5_PROBE_DESIGN.md:246-247`) — the fair ceiling (each-own-oracle is the correct FAIR upper bound;
  handicapping CLIP to a bad τ would inflate Δ but is forbidden).
- Honest preview (§6.5) and bootstrap (§8, paired Δ both calibrated) — both fair.
- Deployed arm = "= the 13115 banked number, G-repro sanity" (§6.2 arm 1 `:205`); a Qwen-calibrated
  vs CLIP-deployed headline is explicitly **FORBIDDEN** (§4 `:126-128`).
- **K4** (`:349`) kills the "calibration helps CLIP as much as Qwen (Δ≈0)" case — the fair-pairing
  premise is itself a declared killer.
No residual path where Qwen-calibrated is scored against CLIP-as-deployed and claimed.

### Item 2 — CIRCULARITY / METRIC-SHOPPING · **PASS** (non-blocking A3)
- **Single, pre-justified primary:** τ = argmax **dev macro-F1**, with a 4-point rationale declared
  before results (prereg §5.1 `:138-158`). It is self-consistent (macro-F1 is a *goal* metric, not a
  proxy), the harder/more threshold-sensitive clause, and class-aware. Because τ is selected on **dev**
  and the AND-rule's *acc* clause is **not** the selection target, acc must come along un-optimized —
  a conservative (if anything acc-binding) construction, not a favourable one.
- **Balanced-acc demoted** to sensitivity-only and "cannot be swapped in after results" (§5.1
  `:160-162`; `B5_PROBE_DESIGN.md:225-227`). The garden-of-forking-paths across {macroF1-τ,
  balacc-τ, oracle} is pruned to ONE pre-registered decision path per protocol.
- **Tie-break deterministic & complete.** §5.3 "lower-median of the maximal plateau, then
  nearest-to-deployed." Probe code `:157` `G[plateau[len(plateau)//2 - (1 if len%2==0 else 0)]]`
  correctly returns the lower-median for even L and the middle for odd L (verified L=1,2,3,4). Grid
  (`:147-150`): `np.unique` collapses duplicate votes; midpoints between adjacent uniques; sentinels
  `u.min()−1e-6` / `u.max()+1e-6` make both "predict-all-one-class" endpoints reachable (±1e-6 is
  cleaner than ±inf and identical in effect since votes are bounded).
- **A3 (non-blocking):** "plateau" is only strictly well-defined if the argmax-macroF1 set is
  contiguous; `np.flatnonzero` returns *all* argmax indices, which a multi-modal macro-F1(τ) could make
  non-contiguous. The **code is deterministic regardless** (index-median of the flatnonzero array), so
  reproducibility is not at risk, but the PROSE ("median of the plateau") could be read by an
  implementer as "median of the longest contiguous run," diverging from the code. Fix: state in §5.3 /
  `B5_PROBE_DESIGN.md`§5 that "plateau = the full set of argmax grid indices (`np.flatnonzero`);
  the tie-break is the lower-median of that index array; contiguity is not assumed and the code in
  §3.2 is authoritative."

### Item 3 — KILL-SWITCH SOUNDNESS · **PASS w/ BLOCKING A1**
Sound in substance: oracle = upper bound; AND-consistent eligibility (both metrics ≥ +0.03 per
protocol, §6.4 `:236-242`); "oracle NEVER claimable as a result" stated **three times** (§6.2 `:210`,
§6.4 `:234`, §10 `:322`); oracle test-touch bounded and ledgered (§10 `:319-329`, "1 bounded,
probe-only"). Kill logic is conservative-correct: the each-own-oracle Δ is the loosest FAIR ceiling, so
an oracle-kill is never a false-kill, and an inflated oracle only causes cheap false-*continues* (honest
remains the arbiter). K2 (`:347`) correctly kills "acc-oracle passes but mF1-oracle fails every
protocol."

**BLOCKING A1 — the binding kill-switch blockquote is internally ambiguous and could false-KILL or be
gamed.** The blockquote (§6.4 `:229-234`) states an **unconditional, acc-only, protocol-unqualified**
kill: *"If the 3-seed mean paired ΔAcc_oracle < +0.03 on ZH, B5 is DEAD."* The following paragraph
(`:236-242`) makes the operative rule **per-protocol AND-eligibility, "DEAD if neither protocol
eligible,"** with final-epoch "primary" and val-selected "judged in parallel." These conflict: if
final-epoch ΔAcc_oracle = +0.02 (fails the blockquote) but val-selected has ΔAcc_oracle ≥ +0.03 AND
ΔmF1_oracle ≥ +0.03 (eligible under the paragraph), the blockquote says DEAD while the paragraph says
alive. Which governs is undefined — a false-KILL of a val-selected-eligible route, or a
protocol-shopping lever. Since this is THE core kill-switch, the ambiguity must be removed. Exact
replacement text in §3.

### Item 4 — HONEST-PREVIEW GATE · **PASS** (non-blocking A8)
The honest preview **is** a test-touch (it applies dev-τ to test and reads test acc/F1), and peeking is
structurally impossible:
- τ is a **deterministic function of dev only** (pre-registered statistic + grid + tie-break), frozen
  before test evaluation: §6.2 arm 2 "computed on dev, applied to test" (`:207-209`); probe `:155-158`
  computes `tau_star` from `votes_dev/labels_dev` then evaluates test. No free knob a peeker could nudge
  after seeing test/oracle numbers; the balanced-acc and oracle arms cannot be swapped in (§5.1).
- The preview = the parent's FULL criterion incl. 3/3 sign (§6.5 `:246-248`), so it IS the formal
  decision; the later GPU submit is "a clean-room re-derivation, not a new draw" (§6.5 `:250-251`,
  §11 `:338-340`).
- **A8 (non-blocking):** the §10 ledger lists "oracle ceiling" (row 2) and "formal single submit"
  (row 3) but does not explicitly say the honest-arm test evaluation **computed in the probe** IS row
  3's single touch (spent once; the optional GPU submit re-derives identical numbers and consumes no
  additional test information). Add one sentence to §10 to make the accounting explicit and preclude a
  reading in which the probe's honest arm plus a GPU submit look like two test draws.

### Item 5 — G-REPRO · **PASS w/ BLOCKING A2**
The test-side gate is strong: deployed acc **AND** macro-F1 **AND** roc must match the §4 anchors to 4
dp for all 12 (6 arms × 2 protocols); roc pins the vote **ordering**, acc/F1 pin the **signs**; a single
neighbor-flip moves a metric by ≥1/149 = 0.0067 ≫ 1e-4, so the gate is sensitive enough to catch the
exact CPU-vs-GPU failure mode. HALT logic sound: CPU mismatch → 1-min GPU eval fallback
(device='cuda', Faiss_GPU=False = bit-exact to 13115's compute path); GPU mismatch → HALT, probe does
not proceed (K6 `:351`).

**BLOCKING A2 — the hard gate validates the TEST votes but not the DEV votes, and DEV drives τ
selection.** The entire calibration selects τ on the 78-dev; the honest number is a function of the
**dev** vote ordering. The gate currently anchors only the test deployed metrics (§6.3 `:219-221`,
`B5_PROBE_DESIGN.md:207-214`). A dev-specific replay error with a clean test reproduction is *unlikely*
(dev and test share one code path — `retrieve_evaluate_RAC_` on a different eval loader — with no
dev-specific branch), but "unlikely" is not "gated," and the dev anchors are **free**: the trainlogs
print `Val_Retrieval Epoch NN macroF1/acc/roc` at every epoch. Require the recomputed **dev** deployed
acc/macroF1/roc at each loaded checkpoint to match the trainlog `Val_Retrieval` line to 4 dp, alongside
the existing test anchors. Verified-available dev anchors (this review, for the executor):

| arm·seed | dev final e29 (mF1/acc/roc) | dev val-sel (epoch) |
|---|---|---|
| CLIP 0 | 0.7857/0.8077/0.8329 | e29 0.7857/0.8077/0.8329 |
| CLIP 1 | 0.7225/0.7692/0.8879 | e28 0.7471/0.7821/0.8836 |
| CLIP 2 | 0.7645/0.7949/0.8764 | e25 0.7894/0.8205/0.8343 |
| Qwen 0 | 0.7650/0.7821/0.8579 | e22 0.7940/0.8205/0.8693 |
| Qwen 1 | 0.8050/0.8205/0.8864 | e25 0.8628/0.8718/0.9307 |
| Qwen 2 | 0.7613/0.7821/0.8436 | e28 0.8301/0.8462/0.8514 |

(Qwen s0 val-sel e22 dev acc 0.8205 cross-checks the ckpt suffix `0.8205128…` and roc 0.8693 the B1
tie-break at `B1_VERDICT_REVIEW.md:44`.) This closes the gate's one gap at zero cost.

### Item 6 — D3 GUARDS · **PASS** (non-blocking A10)
- Bootstrap: ≥1000 dev resamples, re-select τ per resample, apply to FIXED test, report 5/50/95 pct of
  test-acc/mF1 and of paired Δ; K7 (`:352`) labels a pass **D3-fragile** if the paired-Δ 5th pct
  crosses 0 (prereg §8.1 `:293-299`; probe §8 `:265-283`). Correctly targets the NEW risk B5 introduces
  (dev→test threshold-selection tax) rather than test-sampling noise (out of scope of the parent rule,
  consistent with B1/B3).
- τ cross-seed stability (§8.2) and calibration tax oracle→honest (§8.3) both reported.
- **Marginal-language ruling present and faithful:** §7 `:277-283` mandates the fixed format
  `final-epoch: PASS (MARGINAL); val-selected: …` with the **three** B3 facts (proximity to bar;
  uneven per-seed carry; margin < between-seed spread), explicitly citing `B3_PREREG_REVIEW §2.2`
  (which I verified requires exactly those three). Paired-t is effect-size-only, n=3 (§7 rule 3 `:264`).
- **A10 (non-blocking):** state that the §7 MARGINAL label and the §8 D3-FRAGILE label **compose** —
  both applied if both trigger (a marginal pass whose bootstrap 5th-pct also crosses 0 is
  "PASS (MARGINAL, D3-FRAGILE)").

### Item 7 — STAT/CODE REALITY · **PASS** (non-blocking A9)
Vote description verified TRUE (§1.5); the probe calls the real `compute_metrics_retrieval` rather than
reimplementing the vote (`B5_PROBE_DESIGN.md:96-98,122-126`), and the only reimplemented piece
(threshold arithmetic) is itself gated by the τ=0 positive-F1 sanity (`:166`). Determinism correctly
characterized: exact given identical head-forward embeddings; the CPU-vs-GPU head forward is the sole
device-dependent step, gated by G-repro + GPU fallback. dtype (`.astype("float32")`, `:425-426`) and
faiss build (CPU IndexFlatIP both sides) are not divergence sources.
- **A9 (non-blocking):** tighten §6.1 `:196` "the retrieval is bit-reproducible on CPU" — precisely,
  faiss retrieval is deterministic *conditional on* identical head-forward embeddings; end-to-end
  CPU-vs-13115(GPU) reproduction is not guaranteed and is exactly what the G-repro gate checks. The
  design already adds the caveat in the next clause; just align the standalone phrase.

### Item 8 — VETO / SCOPE · **PASS**
- **No MLLM aux signal** (§3 `:90-93`, §4 `:132-133`) — the D1-escape; **no aux gold annotations**:
  only own-**dev class labels** select the scalar τ (§4 `:133-134`), which is (i) within the checklist's
  own ratified carve-out ("class labels on own dev = fine") and (ii) the *same* supervision the head is
  already trained on (BCE) and epoch-selected on (max Val_Retrieval acc); it introduces no new
  annotation type. Test labels are used **only** for the bounded oracle probe ("gold 仅用于 probing",
  REFLECTION §4), never in-method.
- **Single-dataset** (MHC-ZH only; the ≥2-dataset claim combines B5-ZH with the separately-established
  HateMM pass — the goal's structure, not cross-dataset training), **no OCR**, **no ensembles** (seeds
  reported paired, not averaged into an ensemble). ✓
- **Novelty scope honest:** §9 `:305-315` — "threshold calibration per se is generic," performance
  clause only, novelty is a **D7-class user ruling** (same as B3 LoRA), no overclaim. ✓
- **Test-touch ledger complete** (§10) and **disk-guard step-0 head-copy mitigation present** (§6.1
  `:197-199`; `B5_PROBE_DESIGN.md:75-91` DEST outside `logging/`; executor step 0 `:289`). ✓

### Item 9 — MISSED false-PASS / false-KILL surfaces
- **(blocking) A1** — kill-switch wording (above).
- **A4 (non-blocking) — "3/3 seeds" is not 3 independent draws.** The 3 seeds vary only the
  **head-training seed**; the 78-dev and 149-test splits are FIXED across seeds. So "3/3 sign
  consistency" reflects head-seed variance under a **single data-split draw** — the same B3 single-draw
  caveat (`B3_PREREG_REVIEW.md:110-119`). B5 adds a threshold-selection step on that same fixed 78-dev,
  a further correlated-across-seeds noise source. State this caveat and require it to travel with any
  B5 claim (do not oversell "3/3 seeds" as 3 dataset draws).
- **A5 (non-blocking) — val-selected double-dips the 78-dev.** On the val-selected protocol the 78-dev
  drives BOTH epoch selection AND threshold selection; the §8 bootstrap resamples the threshold step but
  holds the checkpoint FIXED, so it **understates** the total val-selected dev-selection tax. The
  final-epoch protocol (fixed e29, single dev use = threshold only) is cleaner and is correctly the
  primary — note this asymmetry explicitly so a val-selected-only marginal pass is not oversold.
- **A6 (non-blocking) — pin the paired-bootstrap RNG construction.** The paired Δ_b requires a **common
  dev-resample index** across the seed-matched Qwen and CLIP runs (`B5_PROBE_DESIGN.md:279-281`). The
  §8 code seeds `default_rng(1234)` once; the executor must ensure the *same* 1000 index arrays are used
  for both encoders at a given seed. Cleanest spec: **precompute the 1000 resample-index arrays once**
  (rng(1234), size 78) and reuse identically across all (encoder, seed, protocol). State this so the
  pairing cannot silently desynchronize.
- **A7 (non-blocking, verdict-processing) — G-repro validates votes, not the calibration arithmetic.**
  A probe-script bug in the grid / tie-break / macro-F1-at-τ path would NOT be caught by the deployed-τ
  anchors (which only exercise τ=0) and would produce plausible-looking wrong numbers. The CPU probe
  script is authored by the executor and not independently code-reviewed before it runs. Require, at
  verdict processing, an **independent hand-recomputation** of one (arm, seed, protocol) honest cell
  (τ_star, honest_acc, honest_mF1) directly from the dumped `votes_dev/labels_dev`,`votes_test` —
  closing the "votes-gated but calibration-ungated" gap. (The A2 dev anchor + this hand-check together
  fully validate the machine.)

---

## 3. Required amendments

**Blocking (must be applied and re-hashed before any authorization is exercised on the FORMAL stage;
the PROBE stage authorization in §5 is conditioned on A1+A2 being folded in first):**

- **A1 (BLOCKING) — reconcile the kill-switch blockquote (prereg §6.4 `:229-234`).** Replace the
  blockquote with a single per-protocol rule:

  > **KILL-SWITCH (binding, per-protocol).** Per seed *s* and protocol *P* ∈ {final-epoch,
  > val-selected}, compute paired oracle-vs-oracle deltas ΔAcc_oracle(s,P) = acc(Qwen_oracle,s,P) −
  > acc(CLIP_oracle,s,P) and likewise ΔmF1_oracle(s,P), each arm using **its own** test-optimal
  > threshold (fair pairing). Protocol *P* is **ELIGIBLE** for the formal stage iff, under *P*, the
  > 3-seed mean paired ΔAcc_oracle(P) ≥ +0.03 **AND** the 3-seed mean paired ΔmF1_oracle(P) ≥ +0.03.
  > **B5 is DEAD iff NEITHER protocol is eligible** — the ranking advantage is not convertible even
  > with a perfect threshold; no formal run is submitted; exhaustion is re-confirmed for this cell.
  > **final-epoch** is the reporting-emphasis reference but holds **no veto** over an independently
  > eligible val-selected protocol; conversely an eligible protocol authorizes the formal stage **only
  > under that same protocol** (no cross-protocol claim). Oracle numbers are an upper bound and can
  > **NEVER** be claimed as a result.

  Mirror the same wording in `B5_PROBE_DESIGN.md` §7 `:249-254`.

- **A2 (BLOCKING) — add a DEV-side G-repro anchor (prereg §6.3 `:217-225`; `B5_PROBE_DESIGN.md` §4
  `:207-211`).** Require: the probe's recomputed **dev** deployed acc, macroF1 AND roc at each loaded
  checkpoint MUST match the corresponding trainlog `Val_Retrieval Epoch NN` line to 4 dp, for all 6
  arms × both protocols, as a co-equal HARD gate with the existing test anchors (same HALT-and-fallback
  logic). Anchor values are tabulated in §2 Item-5 above (freely available in the six 13115 trainlogs).

**Non-blocking (fold in at verdict processing / before the FORMAL stage; do NOT re-open the probe
authorization):** A3 (plateau non-contiguity wording), A4 (single-data-split-draw caveat), A5
(val-selected double-dip caveat), A6 (paired-bootstrap common-index precompute), A7 (independent
calibration-arithmetic hand-check at verdict), A8 (ledger: honest preview = row-3 touch), A9 (§6.1
"bit-reproducible" phrasing), A10 (MARGINAL ⊕ D3-FRAGILE labels compose).

---

## 4. Final verdict — **APPROVED-WITH-AMENDMENTS**

The B5 pre-registration is numerically exact (all 12 anchors, both protocol means, splits, checkpoints,
and every code fact independently reproduced), fair-paired in every comparison path, escapes D1/D2/D3
and every epitaph on a genuine untouched sub-cell of B1, and is honestly scoped (performance-clause
only; novelty deferred to a D7-class user ruling). The vote-granularity correction (continuous, not
21-level) is a correct improvement on the brief. **Two blocking amendments** (A1 kill-switch wording,
A2 dev-side anchor) close a governing-rule ambiguity and a one-sided hard gate — both cheap and
mechanical, neither touching a number. Eight non-blocking amendments strengthen reporting honesty and
replay validity.

---

## 5. Conditional authorization — **PROBE STAGE ONLY**

**Granted 2026-07-14 by the B5 pre-registration reviewer**, scoped to the **zero-GPU G0-cond probe
only** (CPU checkpoint-reload replay off cached features; or the ≤1-min single `device='cuda'`,
`Faiss_GPU=False` eval-only fallback if and only if CPU G-repro fails). The formal single-submit stage
is **NOT** authorized here — it requires a separate authorization after the probe results and the
oracle/honest gates are adjudicated by independent verdict processing.

Conditions:
1. **A1 and A2 folded into both files first** (kill-switch wording + dev-side anchor), then the probe
   runs against the amended spec. A3–A10 SHOULD be folded now but are not blocking for the probe.
2. **Step 0: snapshot the 12 heads** (`B5_PROBE_DESIGN.md` §2.3) to a guard-excluded path before the
   disk guard can prune.
3. **G-repro HARD gate FIRST**, now including the **dev** anchors (A2): any 4-dp mismatch on CPU →
   the §6 GPU fallback; mismatch on GPU too → **HALT**, probe does not proceed.
4. **Oracle kill-switch per the amended A1 rule**; if DEAD → write the negative verdict, **no GPU**,
   exhaustion re-confirmed for the cell.
5. **Oracle test labels are used only for the bounded, ledgered upper-bound probe**; the oracle number
   is NEVER reported as a result. The honest τ is selected on dev only and frozen before test
   evaluation.
6. **Executor writes raw numbers + τ values + grid sizes with line-numbered provenance and applies NO
   pass/fail interpretation** (verdict processing is independent). Include the A7 hand-check inputs so
   the calibration arithmetic can be independently validated.
7. **No SLURM/GPU beyond the single ≤1-min fallback**; no config/state/CLAUDE.md mutation; the only
   writes are the probe script, the head snapshot, and the probe output table.

**Out of scope:** the formal single-submit; any second submission; any change to encoder / seed /
dataset / model; any test-label use beyond the pre-declared oracle ceiling; treating the oracle or the
balanced-acc sensitivity arm as the reported result.
