---
type: experiment
node_id: exp:exp-conv-zh-b5
title: "B5 — operating-point conversion of the frozen-Qwen MHC-ZH ranking advantage: calibrated-vs-calibrated threshold test with a zero-GPU oracle kill-switch probe (PRE-REGISTRATION, DRAFT-UNREVIEWED)"
idea_id: ""
status: PREREG APPROVED-WITH-AMENDMENTS (applied r1); PROBE AUTHORIZED
verdict: approved-with-amendments
confidence: n/a
date: "2026-07-14"
hardware: "ZERO GPU for the G0-cond probe (CPU checkpoint-reload replay off cached features + 12 existing head checkpoints). Formal stage (only if probe passes): 1 serial sbatch reusing the 13115 heads, seconds/run, 1x A100."
duration: "Probe: minutes on CPU. Formal stage: <20 min wall, 1x A100, single submit."
provenance: "PRE-REGISTRATION ONLY — NO runs executed. Reuses job 13115 artifacts: 6 trainlogs slurm/logs/enc3s_MHC_zh_{openai_clip-vit-large-patch14-336_HF,Qwen2.5-VL-7B-Instruct_HF}_seed{0,1,2}_13115.trainlog and their per-epoch head checkpoints under logging/Retrieval/MHC_zh/RAC_video_archive_seeds/.../ckpt/epoch_model_*.pt (verified present 2026-07-14). Cached features data/CLIP_Embedding/MHC_zh/{train,dev_seen,test_seen}_{model}.pt. Audit lineage: refine-logs/EXHAUSTION_AUDIT_2026-07-14.md §1,§7 (lead open cell + G0-cond probe spec); refine-logs/B1_VERDICT_REVIEW.md:29-40,160-163 (roc>acc anomaly); src/utils/metrics.py:294-301 (fixed-threshold code); research-wiki/REFLECTION_mllm_integration_failures.md §4 (G0-cond gate + calibration mandate). Executable probe spec + G-repro anchors: refine-logs/B5_PROBE_DESIGN.md."
added: 2026-07-14T00:00:00Z
tags: ["hateful-video", "MLLM-encoder", "frozen-Qwen", "MHC_zh", "MHC-ZH", "operating-point", "threshold-calibration", "decision-rule", "G0-cond", "oracle-kill-switch", "pre-registered", "DRAFT-UNREVIEWED", "B5"]
---

# B5 — operating-point conversion of the frozen-Qwen MHC-ZH ranking advantage (PRE-REGISTRATION)

> **STATUS: `PREREG APPROVED-WITH-AMENDMENTS (applied r1); PROBE AUTHORIZED`. Fresh
> pre-registration review complete (`refine-logs/B5_PREREG_REVIEW.md`, 2026-07-14: verdict
> APPROVED-WITH-AMENDMENTS; blocking A1 kill-switch wording + A2 dev-side G-repro anchor applied
> in r1, non-blocking A3–A10 folded). The zero-GPU G0-cond PROBE STAGE is authorized (review §5).
> The formal single-submit stage is NOT authorized — it needs a separate authorization after the
> probe results are adjudicated by independent verdict processing. No SLURM job submitted for the
> formal stage; no GPU used; no ZH test touch spent beyond the pre-declared oracle-ceiling probe
> (a bounded, probe-only touch, §10); the formal single-submit stage is gated behind the probe's
> oracle kill-switch (§6.4).**

**verdict:** `approved-with-amendments` — probe authorized (review §5); this file establishes the
probe + the gated formal test. · **confidence:** n/a

## 1. Purpose (one line) + audit lineage

Test whether **calibrating the decision threshold** converts the frozen-Qwen encoder's measured
**ranking advantage** on MHC-ZH (Test roc/AUC 0.88–0.90 vs frozen-CLIP 0.83–0.84, 3/3 seeds) into
the goal's second-dataset **accuracy + macro-F1** pass — under a **fair, calibrated-vs-calibrated**
comparison, with a zero-GPU **oracle-threshold kill-switch** run first.

**Audit lineage.** The round-2 exhaustion audit (`refine-logs/EXHAUSTION_AUDIT_2026-07-14.md`)
refuted "the search space is exhausted" by identifying exactly one open, never-measured cell sitting
on the binding gap (a second passing dataset): B1 (the 20th negative) disposed of "frozen-Qwen
encoder on MHC-ZH" as a clean FAIL, but B1 only measured the encoder swap **under the pipeline's
unchanged, uncalibrated decision rule** (§7 of the audit). It never measured the quantity that
decides whether the representation advantage is real-but-blocked vs absent: the calibrated-threshold
accuracy of the frozen-Qwen ZH representation. B5 measures it.

This experiment is the **B1 sub-cell that B1 left untouched** — same features, same heads (literally
the job-13115 checkpoints), a different *lever* (the decision threshold), added on top of an
already-trained head. It is **not** a new encoder (B1), **not** LoRA (B3), **not** an added MLLM
signal (P1–P5, TARC, A-line). See §3 for the non-isomorphism argument.

## 2. The load-bearing anomaly (cell definition)

On MHC-ZH the frozen-Qwen encoder ranks **far** better than frozen-CLIP but scores **worse** on the
pipeline's thresholded accuracy — a large, 3/3-seed-consistent, **unconverted** AUC advantage.
Numbers re-read from the primary 13115 trainlogs (final-epoch, provenance in
`refine-logs/B5_PROBE_DESIGN.md` §G-repro; matches `B1_VERDICT_REVIEW.md:35-40`):

| ZH final-epoch (job 13115) | acc | roc/AUC | roc − acc (calibration gap) |
|---|---|---|---|
| frozen-CLIP mean | 0.8143 | 0.8389 | 0.025 (near-optimal operating point) |
| **frozen-Qwen mean** | **0.8031** | **0.8888** | **0.086 (badly mis-placed operating point)** |
| Δ (Qwen − CLIP) | **−0.0112** | **+0.0499** | — |

(CLIP roc mean = (0.8382+0.8342+0.8444)/3 = 0.8389; Qwen roc mean = (0.8906+0.8951+0.8806)/3 = 0.8888.)

**The signature.** Frozen-Qwen's roc−acc gap is **0.086 — ~3.4× CLIP's 0.025.** Qwen ranks ZH videos
excellently and the fixed decision rule throws that ranking away; CLIP is already near its own
operating-point ceiling. That asymmetry is the signature of a **convertible** representation
advantage blocked at the decision layer — the hypothesis B5 tests. It is a hypothesis, not a
foregone conclusion: an AUC of 0.889 does **not** guarantee a large best-threshold accuracy (under an
equal-variance score model it could already be near-optimal), so the convertible headroom is
**empirically undetermined and must be probed** (§6). This is why the oracle kill-switch exists.

**The code fact (why the operating point is fixed and class-blind).** Per-epoch Val/Test lines call
`compute_metrics_retrieval(..., use_sim=True)` (`src/run_rac.py:673-674,691-692`); inside
(`src/utils/metrics.py:297-301`) accuracy thresholds the similarity-weighted signed kNN vote at a
**fixed** cut: `list_majority_voted_round = (sigmoid(vote) >= 0.5)` ⇔ `vote >= 0`, and macro-F1
(`:307-309`) uses the same fixed cut. The threshold is **never calibrated on validation**; model
selection (`run_rac.py:757,761,771-773`) only picks the epoch with max Val_Retrieval acc *at that
fixed cut*. **MHC-ZH is ~30% positive** (test 45/149 = 0.302; dev 28/78 = 0.359; audit §1.2), so a
symmetric class-blind cut is *a priori* suboptimal for both metrics — and macro-F1 especially.

**IMPORTANT vote-granularity correction (design-relevant; supersedes the brief's "21 levels").** The
13115 runs use `majority_voting='arithmetic'` with `use_sim=True` (Namespace-verified,
`B5_PROBE_DESIGN.md` §1). The realized per-video vote is therefore
`sum_k [ (2·label_k − 1)·sim_k · w_k ] / sum_k w_k` with rank weights `w = [20,19,…,1]` — a
**continuous** real number, NOT a 21-level integer count. The threshold search grid is consequently
the **empirical unique-vote grid** (midpoints between adjacent sorted vote values), not a fixed
21-level grid (§5.2).

## 3. Why this cell is open — escapes D1/D2/D3 and every epitaph

- **D1 (redundancy law) does not bite.** D1 kills *adding a low-bandwidth decision-side MLLM signal*
  that is conditionally redundant given the frozen representation. B5 adds **no signal at all** — no
  MLLM scalar, no channel, nothing concatenated to Z. It re-places the operating point of an
  *already-better* representation using only **class labels on the 78-video val split** (labels are
  explicitly allowed). There is no "conditional information beyond Z" question because nothing is
  added to Z. **This is exactly why B5 escapes the D1 redundancy law** that closed P1–P5/TARC/A-line.
- **D2 (representation law) actively PREDICTS this.** D2: only representation-level levers ever
  cleared +3; gains come from representation, never decoration. Frozen-Qwen-ZH **is** a representation
  with a large ranking advantage (AUC +0.050, 3/3 seeds); it failed to clear +3 *only because the
  decision rule discards the advantage*. Unlocking it is not decoration (adds no signal) and not a new
  representation. D2 is the reason to run this, not to skip it.
- **D3 (measurement law) is the real risk, and it is bounded and quantified here.** 78-dev selection
  is noisy (±~2 acc pts). A **single scalar threshold** is far less overfit-prone than epoch- or
  feature-subset selection, and the AUC advantage (+0.050) is well above the noise floor. D3 is why
  the *honest* (val-calibrated) number may land marginal rather than clean — not a reason the cell is
  closed. The probe (§6) puts an **oracle-threshold arm** as a kill-switch (D3 as a hard gate) and a
  **≥1000-resample bootstrap of the val-threshold choice** as a quantified distribution (§8), so D3 is
  measured, not assumed.

**Epitaph / isomorphism check (non-isomorphic to all priors).**
- **B1 (20th negative, `exp-encoder-zh-b1.md`)** measured the encoder swap under the **unchanged
  fixed-threshold rule**. B5's injection point is the **decision threshold on the same frozen head** —
  the untouched sub-cell. B1's own verdict flags the unconverted AUC advantage
  (`B1_VERDICT_REVIEW.md:160-163`) and discards it as rule-neutral; B5 is that discard, measured.
- **B3 (`exp-lora-zh-b3.md`)** converts ZH via **LoRA** (re-training the encoder). B5 converts via the
  **decision rule on the frozen features** — a cheaper, LoRA-free lever that sidesteps the
  "LoRA = performance lever, not novelty" pending ruling. Different mechanism.
- **P1 (`CAMPAIGN_mllm_method_role.md:51`)** adjusted a vote threshold, but via an **MLLM-generated
  prior** (a decision-side MLLM signal, bandwidth = bits) to correct temporal drift. B5 uses **no MLLM
  signal** and **val labels only**; non-isomorphic (different signal source, bandwidth class, failure
  mode). encoder-swap / SAV / P9–P11 / TARC / A-line all touch representation-mining or decision-side
  *signals*; none touches operating-point calibration of the frozen-Qwen ZH vote.

## 4. Arms — FAIR, calibrated-vs-calibrated (binding design rule 1)

The formal comparison is **calibrated-frozen-Qwen vs calibrated-frozen-CLIP**, paired within seed,
MHC-ZH, seeds 0/1/2, anchored to job 13115's cached features and head checkpoints. **Both arms get
the IDENTICAL calibration procedure** (§5). A Qwen-calibrated vs CLIP-uncalibrated comparison is a
confound and is **FORBIDDEN** as a headline — the whole claim is that calibration helps Qwen *more*
than CLIP (predicted by the 0.086-vs-0.025 roc−acc asymmetry; if it helps both equally, the cell
dies). The paired quantity reported is Δ = (Qwen_calibrated − CLIP_calibrated), never Qwen alone.

- **Control arm:** frozen-CLIP head (13115 CLIP checkpoints), threshold calibrated by §5.
- **Treatment arm:** frozen-Qwen head (13115 Qwen checkpoints), threshold calibrated by §5.
- **No MLLM auxiliary signal anywhere.** B5 adds no signal; this is what makes it escape the D1
  redundancy law (§3). Gold labels on the 78-video val split (= class labels) are used only to select
  the scalar threshold — explicitly allowed.

## 5. Primary calibration statistic + grid + tie-break (binding design rule 2)

### 5.1 PRIMARY selection statistic (declared before results): **maximize dev macro-F1**

**Chosen primary = the threshold τ that maximizes macro-F1 on the 78-video ZH dev split**, applied to
test. Rationale (declared before seeing any calibrated number, so no metric-shopping):

1. **Self-consistent, not a proxy.** macro-F1 is itself **one of the two binding goal metrics**
   (the AND-rule needs Δacc ≥ +0.03 AND ΔmacroF1 ≥ +0.03). Selecting the operating point to optimize a
   *goal* metric is the honest, pre-registrable choice; selecting on a non-goal proxy (e.g. balanced
   accuracy) and then reporting acc + macro-F1 invites the "optimized X, reported Y" criticism.
2. **The harder, more threshold-sensitive clause.** On a 30/70 set macro-F1 is the more
   threshold-sensitive goal metric (audit §7, `EXHAUSTION_AUDIT:291`) and the harder clause to
   satisfy. Selecting on the harder clause is the conservative choice; raw accuracy tends to come
   along because the macro-F1-optimal cut on a 30/70 set sits near the balanced operating point that
   also protects majority-class accuracy — but this is exactly the empirical question B5 answers.
3. **Class-aware without over-rebalancing.** macro-F1 weights both classes' precision and recall, so
   it counters the imbalance pathology that a raw-dev-acc threshold would exploit (collapse toward the
   majority/normal class), while not fully rebalancing toward the minority the way balanced accuracy
   can — which would risk *tanking* raw test-acc, the exact clause Qwen currently fails.
4. **Granularity supports it.** Because the vote is continuous (§2 correction; ~≤78 distinct dev
   candidate thresholds), the selection statistic is well-resolved — the choice should track the
   *goal*, not smoothness, and macro-F1 is half the goal.

**Balanced accuracy is retained as a SECONDARY sensitivity arm** (computed in the same probe at zero
extra cost, per audit §7's suggestion) and reported for transparency — but it is **not** the decision
statistic and cannot be swapped in after results (anti-metric-shopping).

### 5.2 Candidate-threshold grid (declared): empirical unique-vote midpoints

The vote is continuous (§2). The candidate threshold set on dev = the sorted **midpoints between
adjacent unique dev vote values**, plus two sentinels (below the min vote and above the max vote) so
the "predict-all-one-class" endpoints are reachable. Search maximizes dev macro-F1 over this grid.
(This replaces the brief's assumed 21-level grid, which is only correct for an unweighted majority
vote; the `arithmetic`+`use_sim` pipeline produces continuous votes.)

### 5.3 Tie-breaking rule (declared): mid-plateau, then nearest-to-deployed

If several candidate thresholds tie on max dev macro-F1 (a plateau — common on 78 samples), pick the
**median threshold of the maximal plateau** (most robust to dev noise). If the plateau has an even
count, take the lower-median. Secondary tie-break (exact numeric tie of medians, measure-zero):
choose the threshold **closest to the deployed cut (vote = 0)**, i.e. the minimal operating-point
move. Both rules are fixed now, before any run.

**Implementation note (amendment A3).** "Plateau" = the **full set of argmax grid indices**
(`np.flatnonzero` over the max-dev-macroF1 mask), NOT the longest contiguous run; the tie-break is the
**lower-median of that index array**; contiguity is not assumed, and the `B5_PROBE_DESIGN.md` §3.2
probe code (index-median of the flatnonzero array) is authoritative for reproducibility.

## 6. G0-cond probe — exact procedure + oracle kill-switch (binding design rule 3 & 4)

**Object.** On the dumped per-video votes (from the 13115 heads), compute — per encoder × seed ×
protocol — three threshold arms, and decide with the oracle kill-switch whether the formal
single-submit stage is warranted. **Zero GPU** (CPU checkpoint-reload replay; §6.1). Full executable
spec, exact paths, and the G-repro anchor table live in `refine-logs/B5_PROBE_DESIGN.md`.

### 6.1 Mechanics (zero-GPU preferred; verified recoverable)

**All 12 required head checkpoints exist on disk (verified 2026-07-14)** — 6 final-epoch (`epoch_model_29_*.pt`)
+ 6 val-selected-epoch (`epoch_model_{22,25,28,29,28,25}_*.pt` for Qwen s0/s1/s2 and CLIP s0/s1/s2),
under `logging/Retrieval/MHC_zh/RAC_video_archive_seeds/.../ckpt/` (exact paths in `B5_PROBE_DESIGN.md`
§2). The probe reloads each head (`classifier_hateClipper`, dims 3584/3584 Qwen · 1024/768 CLIP),
runs `retrieve_evaluate_RAC_` on dev+test off the cached features, and dumps the per-video signed
vote `list_majority_voted` + labels. 13115 ran `--Faiss_GPU False` (CPU faiss `IndexFlatIP`, exact
search), so the faiss retrieval is **deterministic on CPU conditional on identical head-forward
embeddings** (amendment A9: end-to-end CPU-vs-13115(GPU) reproduction is *not* guaranteed and is
exactly what the G-repro gate checks); the only residual float risk is the head forward on CPU vs GPU,
caught by the G-repro gate (§6.3). **Disk-guard alert:** these checkpoints are
today's (newest); `scripts/disk_guard.sh` prunes *oldest* logging checkpoints and mirrors to B2 before
pruning, so immediate risk is LOW but non-zero — mitigation (copy the 12 files, ~372 MB, to a
guard-excluded path as step 0) is in `B5_PROBE_DESIGN.md` §2.3. Fallback if checkpoints are pruned
before the probe runs: a G-repro-anchored head re-run of the 13115 `enc3seed_zh_b1` configs
(seconds/run, 1x A100), instrumented to dump votes.

### 6.2 Three threshold arms (per encoder e, seed s, protocol P ∈ {final-epoch, val-selected})

1. **deployed** — fixed cut (vote ≥ 0). = the 13115 banked number. **G-repro sanity (§6.3).**
2. **honest / method** — τ = argmax dev macro-F1 (§5), computed on dev, applied to test. **The
   deployable number.** (Secondary: balanced-acc-selected τ, sensitivity only.)
3. **oracle** — τ = argmax **test** macro-F1, and separately argmax **test** acc. **Upper bound,
   label-oracle arm; gold labels used only to compute the ceiling, NEVER in-method, never claimed as
   a result.** The gap oracle→honest = the val→test **calibration tax** on a single scalar.

For each protocol P the checkpoint is fixed: final-epoch loads `epoch_model_29`; val-selected loads
the run's val-sel-epoch checkpoint (Qwen s0/s1/s2 = e22/e25/e28; CLIP s0/s1/s2 = e29/e28/e25). The
threshold is then calibrated **on val at that checkpoint** — i.e. threshold selection and epoch
selection are two distinct steps, both pre-declared.

### 6.3 G-repro hard gate (kill rule, runs FIRST)

The deployed-arm test acc AND macro-F1 AND roc recomputed from the dumped votes MUST reproduce the
13115 banked readings **to 4 decimal places** for all 6 arms × both protocols (anchor table in
`B5_PROBE_DESIGN.md` §4, e.g. Qwen s0 final = macroF1 0.7864 / acc 0.8188 / roc 0.8906). Any
mismatch ⇒ **HALT** the CPU replay, escalate to the 1-min GPU eval fallback (§6.1); if that still
mismatches, the replay machinery is invalid and the probe does not proceed. (This is the D3-mandated
"probe machine validity" check in the spirit of the REFLECTION §4 calibration mandate: if the probe
cannot reproduce the known operating point, no calibrated number it produces is trustworthy.)

**DEV-side anchor (amendment A2, BLOCKING; `B5_PREREG_REVIEW.md` §3).** Because the calibration selects
τ on the **78-dev** vote ordering, a gate that validates only the test votes is one-sided. As a
**co-equal HARD gate** with identical HALT-and-fallback logic, the probe's recomputed **dev** deployed
acc AND macroF1 AND roc at each loaded checkpoint MUST also match the corresponding trainlog
`Val_Retrieval Epoch NN` line **to 4 dp**, for all 6 arms × both protocols. The dev anchor table lives
in `B5_PROBE_DESIGN.md` §4 (freely available in the six 13115 trainlogs; e.g. Qwen s0 final =
macroF1 0.7650 / acc 0.7821 / roc 0.8579; Qwen s0 val-sel e22 = 0.7940 / 0.8205 / 0.8693). **Both the
test AND dev anchors must pass** for the probe to proceed.

### 6.4 ORACLE KILL-SWITCH (pre-declared, binding) — decides whether ANY formal GPU is spent

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

The blockquote above is the **single binding kill-switch** (amendment A1, `B5_PREREG_REVIEW.md` §3);
its per-protocol AND-eligibility **supersedes** any earlier unconditional or acc-only phrasing.
Rationale: the formal stage is itself an AND-rule on both metrics, so spending the single-submit is
warranted only where the oracle ceiling clears **both** clauses under a single protocol. final-epoch is
the primary reporting-emphasis reference (consistent with B1's designation and the anomaly's
final-epoch framing) but holds no veto over an independently eligible val-selected protocol, which is
judged in parallel and never swapped in post hoc.

### 6.5 Val-calibrated preview gate (before the formal single-submit)

Even if the oracle gate passes, the formal single-submit runs only if the **honest** (val-calibrated)
preview on the dumped votes already clears the goal bar under an eligible protocol:
mean paired Δacc ≥ +0.030 AND ΔmacroF1 ≥ +0.030, 3/3 sign. Because the probe computes the honest arm
directly from the same votes the formal stage would produce, this preview is (up to the G-repro
tolerance) the formal result itself — the formal single-submit is a clean-room re-derivation for the
record, not a new draw. If the honest preview is a near-miss (oracle passes, honest < +0.03), B5 is
banked as a **quantified near-miss that dies on D3** (threshold-selection noise on 78-dev), with the
oracle headroom and the §8 bootstrap distribution recorded — no formal GPU spent.

## 7. Formal-stage decision rule (verbatim goal bar; binding design rule 5)

Transcribed **verbatim** from the parent rule (`exp-encoder-3seed.md:73-85`; identical wording the
B1/B3 reviewers verified):

> For each dataset × protocol:
> 1. **Per-seed paired difference** Δ = (Qwen − CLIP) for acc and macro-F1 at seeds 0/1/2.
> 2. **3-seed mean ± std** of the paired Δ; **sign consistency** (how many of 3 seeds positive).
> 3. n=3 is too small for a formal bootstrap; report the paired-t statistic **as an effect-size
>    descriptor only** — no significance claim from n=3.
> 4. **Pass criterion (per dataset × protocol):** mean paired Δacc ≥ +0.030 AND mean paired ΔmF1 ≥
>    +0.030 AND sign consistency 3/3 positive.
> 5. **Headline claim requires the pass criterion met on ≥ 2 datasets under a stated protocol.** Each
>    protocol judged separately; write-up format fixed as "final-epoch: pass/fail; val-selected: pass/fail".

**Application to B5.** The two arms being differenced are **calibrated-Qwen − calibrated-CLIP** (fair
pairing, §4), threshold selected on val at the respective checkpoint (§6.2), judged **independently
under BOTH protocols**. A PASS under either protocol supplies the **second dataset** the parent rule
(5) needs (HateMM already PASSes both), completing "MLLM-as-encoder helps on ≥2 datasets" under that
protocol — with the mechanism amended to "…once the operating point is calibrated."

**Marginal-pass language (pre-declared, B3-style, binding on the verdict).** The honest realistic
outcome is a **marginal** pass or near-miss (§9). If the calibrated pass is marginal (mean Δ within
the between-seed spread of the +0.030 bar, or carried by <3 uniformly-clearing seeds), it MUST be
reported in the fixed format `final-epoch: PASS (MARGINAL); val-selected: <…>` with the three
mandatory sensitivity facts (proximity to bar; uneven per-seed carry; margin < between-seed spread),
exactly as B3_PREREG_REVIEW §2.2 mandates. No headline upgrade; with HateMM it yields at most a
**family** claim (frozen-encoder-on-HateMM + calibrated-frozen-encoder-on-ZH) pending the user
novelty ruling (§9). **Label composition (amendment A10):** the MARGINAL label (this section) and the
D3-FRAGILE label (§8 guard 1) **compose** — both are applied whenever both trigger (e.g. a marginal
pass whose bootstrap 5th-percentile paired Δ also crosses 0 is reported `PASS (MARGINAL, D3-FRAGILE)`).

**F1-risk pre-declaration (binding design rule 5).** Threshold moves that gain accuracy on a 30%-
positive set can **lose** macro-F1, and vice versa. The AND-rule stays: a threshold that lifts test
acc but drops test macro-F1 below the +0.03 bar is a **FAIL**, reported as FAIL-with-direction. The
primary statistic (max dev macro-F1) is chosen partly to protect the harder clause, but the AND-rule
is the arbiter regardless.

## 8. D3 guards (binding design rule 4)

1. **Bootstrap of the val-threshold choice (≥1000 resamples).** Resample the 78 dev videos with
   replacement; on each resample re-select τ = argmax (resampled-dev) macro-F1 (§5); apply τ to the
   **fixed** test set; record test-acc and test-macroF1. Report, per encoder × seed × protocol, the
   induced **distribution** (5th/50th/95th percentiles) of test-acc, test-macroF1, and of the paired
   Δ = (Qwen − CLIP) — the val-selection noise as a distribution, not a single point (the
   permutation/bootstrap-as-distribution rule, REFLECTION §4). If the bootstrap 5th-percentile paired
   Δ crosses 0, the honest pass is D3-fragile and must be labelled so. **Paired-index construction
   (amendment A6):** precompute the 1000 dev-resample index arrays **once** (`np.random.default_rng(1234)`,
   each size 78) and reuse them **identically** across every (encoder, seed, protocol), so the paired
   Δ_b pairs the seed-matched Qwen and CLIP runs on the *same* resampled dev videos and cannot silently
   desynchronize.
2. **Threshold stability across seeds.** Report the 3 selected τ per encoder and their spread; a
   claim that rests on 3 wildly different operating points is weaker than one with a stable τ.
3. **Calibration tax.** Report oracle→honest gap per arm (the D3 selection tax on a single scalar);
   this is expected to be small relative to epoch/feature selection, and quantifying it is the point.
4. **Single data-split draw (amendment A4).** The 3 seeds vary only the **head-training seed**; the
   78-dev and 149-test splits are **FIXED** across seeds. "3/3 sign consistency" therefore reflects
   head-seed variance under a **single data-split draw** (the same caveat as `B3_PREREG_REVIEW.md`),
   not three independent dataset draws; B5's dev-threshold-selection step adds a further
   across-seed-correlated noise source. This caveat MUST travel with any B5 claim (do not oversell
   "3/3 seeds" as three dataset draws).
5. **Val-selected double-dips the 78-dev (amendment A5).** Under the val-selected protocol the 78-dev
   drives **both** epoch selection AND threshold selection; the guard-1 bootstrap resamples the
   threshold step but holds the checkpoint **FIXED**, so it **understates** the total val-selected
   dev-selection tax. The final-epoch protocol (fixed e29, single dev use = threshold only) is cleaner
   and is correctly the primary; a val-selected-only marginal pass must not be oversold relative to a
   final-epoch pass.

## 9. Novelty scope statement (honest; a user ruling, not decided here)

Threshold calibration *per se* is generic and not "novel MLLM integration." The defensible claim
under test is only the goal's **performance clause** on the binding gap: *MLLM encoders produce
better-ranked but operating-point-miscalibrated representations for a retrieval-vote head; a
calibration-aware decision rule (val-selected threshold, no added signal) is what lets the
MLLM-as-encoder role clear the ≥2-dataset bar (HateMM + ZH).* That is a small, real, honest
methodological finding targeting the binding gap directly. Whether it satisfies the goal's **novelty
clause** is a **user ruling (D7-class)** — the same class of ruling already pending for B3 (LoRA). The
prereg does not decide novelty; it establishes (or refutes) the performance clause. No overclaim: a
pass here is "calibrated frozen-Qwen reaches the ZH goal bar," not "MLLM reasoning solves ZH."

## 10. Test-touch ledger (binding design rule 7)

| stage | ZH test touch | nature | pre-declared? |
|---|---|---|---|
| B1 (20th negative) | 1 | encoder swap under fixed cut (job 13115) | already spent |
| **B5 probe — oracle ceiling** | **1 (bounded, probe-only)** | test labels used only to compute the oracle upper bound + G-repro; NEVER in-method, never a result | **yes, this doc §6** |
| **B5 formal — single submit** | **1** | one calibrated-vs-calibrated evaluation, single sbatch | **yes, this doc §7** |

The oracle probe touches the ZH test set (it reads test labels to compute the ceiling). This is a
**bounded, pre-declared, probe-only** touch sanctioned by the project's gold-for-probing rule
(REFLECTION §4: "gold 仅用于 probing"). The honest/method arm and the formal stage never use test
labels to choose anything. No adaptive re-running against ZH test under this pre-registration; if the
formal touch is spent, no knob-tweaked re-runs.

**Accounting note (amendment A8).** The **honest-arm test evaluation computed inside the probe** (dev-τ
applied to test votes, §6.5) **IS** the row-3 "formal — single submit" touch — the same test
information is read exactly once. An optional later GPU single-submit merely **re-derives the identical
numbers** (up to the G-repro tolerance) and consumes **no additional test information**; it is a
clean-room re-derivation for the record, not a second test draw. The probe's honest arm plus a GPU
submit therefore count as **one** row-3 touch, not two.

## 11. Cost estimate

- **Probe:** ZERO GPU. CPU reload of 12 heads + `retrieve_evaluate_RAC_` on ~579 train / 78 dev /
  149 test off cached features + threshold arithmetic + a 1000-resample bootstrap = **minutes on CPU**.
  Fallback GPU eval (only if G-repro fails on CPU): ~1 min, 1x A100.
- **Formal stage (only if probe passes §6.4 + §6.5):** reuse the 13115 heads; one serial sbatch,
  seconds/run, **< 20 min wall, 1x A100, single submit** (no `--time`; expect `PENDING (JobHeldUser)`,
  wait for auto-release). The formal stage may equivalently be discharged by the probe's own honest
  arm if the reviewer accepts the CPU replay as the record (the calibrated numbers are identical up to
  the G-repro tolerance) — to be decided at verdict processing.

## 12. What-would-kill-this table

| # | Killer | Mechanism | Outcome |
|---|---|---|---|
| K1 | Neither protocol eligible at the oracle ceiling — per-protocol AND-eligibility (3-seed mean paired ΔAcc_oracle ≥ +0.03 AND ΔmF1_oracle ≥ +0.03) fails under **both** protocols (§6.4, A1) | ranking advantage non-convertible even with a perfect cut | **route DEAD** (§6.4); exhaustion re-confirmed for the cell |
| K2 | Oracle acc-clause passes on a protocol but its oracle ΔmF1 < +0.03 there (and likewise the other protocol) ⇒ no protocol clears the AND | AND-rule unreachable even at the ceiling | **route DEAD** (§6.4 AND-eligibility) |
| K3 | Honest (val-calibrated) preview < +0.03 while oracle ≥ +0.03 | D3: 78-dev threshold-selection noise eats the headroom | banked **quantified near-miss** (dies on D3, §6.5); no formal GPU |
| K4 | Calibration helps CLIP as much as Qwen (paired Δ ≈ 0) | the asymmetry premise is false; not a Qwen-specific effect | **route DEAD** (§4 fair-pairing premise refuted) |
| K5 | Acc gains but macro-F1 falls below bar (or vice versa) | 30/70 threshold trade-off breaks the AND-rule | **FAIL-with-direction** (§7 F1-risk) |
| K6 | G-repro mismatch on CPU *and* GPU | replay machinery invalid | **HALT**, probe does not proceed (§6.3) |
| K7 | Bootstrap 5th-percentile paired Δ crosses 0 | honest pass is D3-fragile | pass labelled **D3-fragile / marginal** (§8) |

**Interpretation ladder (all three outcomes decision-useful; none yet produced).** Oracle < +0.03 →
route dead (advantage genuinely non-convertible; exhaustion confirmed for this cell). Oracle ≥ +0.03
but honest < +0.03 → dies on D3, banked as a quantified near-miss with oracle headroom recorded.
Honest ≥ +0.03 both metrics, 3/3, under an eligible protocol → **first frozen-encoder second dataset
via calibration**, pending the user novelty ruling.

## 13. Honest prior / expected outcome (declared before running)

**MODEST prior — realistic outcome is a marginal pass or a D3 near-miss, not a clean +3** (audit §1.4,
§5). The roc−acc gap of 0.086 shows real blocked headroom and the 30/70 imbalance guarantees the fixed
symmetric cut is suboptimal (helps macro-F1 in particular), but AUC 0.889 does not guarantee a large
best-threshold accuracy, and 78-dev threshold selection carries a documented ~2-pt tax. Realistic
band after honest val-calibration: paired Δacc ≈ +0.02 to +0.05 on ZH — a B3-class marginal pass or
near-miss. **That is still a refutation of "exhausted":** a binding-gap cell with a live, cheap,
never-computed path to a second passing dataset is not an exhausted search — which is the point of
running the probe regardless of the modest prior.

## 14. Single-submit ceremony (formal stage; pre-registered)

1. Freeze this pre-registration (fresh review sign-off + delta-check).
2. Run the zero-GPU probe (`B5_PROBE_DESIGN.md`); apply G-repro (§6.3) → oracle kill-switch (§6.4) →
   honest preview (§6.5). Only if an eligible protocol survives both gates: proceed.
3. Author the formal runner (if a GPU record is required) as a CONFIGS-only copy of the 13115
   `enc3seed_zh_b1` runner with vote-dump + threshold-calibration instrumentation; diff-verified,
   hash-pinned, `FORCE=False`, fresh group (no collision with the 13115 anchors).
4. One `sbatch`; no `--time`; `PENDING (JobHeldUser)` = WAIT; no resubmission after any terminal state.
5. Read back every number from raw logs (line-numbered provenance); apply the G-repro gate FIRST,
   then the calibrated-vs-calibrated decision rule verbatim under both protocols; apply the §7
   marginal-pass language.

**Verdict-stage check (amendment A7 — for the independent verdict reviewer, NOT the probe executor).**
The G-repro gate validates the deployed votes (τ=0) but not the calibration arithmetic (grid /
tie-break / macro-F1-at-τ), and the CPU probe script is not independently code-reviewed before it runs.
At verdict processing, an **independent hand-recomputation** of **one** (arm, seed, protocol) honest
cell — τ_star, honest_acc, honest_mF1 — must be performed directly from the dumped
`votes_dev/labels_dev` and `votes_test/labels_test`, to close the "votes-gated but calibration-ungated"
gap. Together with the A2 dev anchor, this fully validates the probe machine. The probe executor
**records the raw inputs** for this check but does **not** perform it (verdict processing is
independent).

## 15. Connections

- extends → `exp:exp-encoder-zh-b1` (the frozen-Qwen ZH FAIL under the fixed cut; B5 = the untouched calibrated-threshold sub-cell)
- contrasts-with → `exp:exp-lora-zh-b3` (ZH via LoRA re-training; B5 = ZH via the decision rule on frozen features)
- controls-against → `exp:exp-consensus-zh-seeds` (frozen-CLIP ZH λ=0 floor)
- motivated-by → `refine-logs/EXHAUSTION_AUDIT_2026-07-14.md` §1,§7 (lead open cell + G0-cond probe spec)
- gated-by → `research-wiki/REFLECTION_mllm_integration_failures.md` §4 (G0-cond gate + label-oracle calibration mandate)
- implemented-by → `refine-logs/B5_PROBE_DESIGN.md` (executable probe spec + G-repro anchors)

## 16. Revision history

| rev | date | status | change | authority |
|---|---|---|---|---|
| r0 | 2026-07-14 | DRAFT-UNREVIEWED | Initial pre-registration authored from the exhaustion-audit lead cell; primary statistic = max dev macro-F1; oracle kill-switch on paired Δacc < +0.03; vote-granularity correction (continuous `arithmetic`+`use_sim` vote, not 21 levels); checkpoint-recoverability verified (12 heads on disk); zero-GPU CPU-replay probe. NO runs. | B5 prereg designer |
| r1 | 2026-07-14 | PREREG APPROVED-WITH-AMENDMENTS (applied r1); PROBE AUTHORIZED | Folded the fresh review `refine-logs/B5_PREREG_REVIEW.md` (verdict APPROVED-WITH-AMENDMENTS): **blocking** A1 (kill-switch rewritten to a single per-protocol AND-eligibility rule) + A2 (co-equal DEV-side G-repro anchor added); **non-blocking** A3–A10 folded. Status banner + verdict updated; PROBE STAGE authorized per review §5 (formal single-submit still NOT authorized). NO runs. See amendment log below. | B5 amendment-applier (r1) |

### Amendment log (r1 — applied 2026-07-14; review pointer `refine-logs/B5_PREREG_REVIEW.md`)

REPLACE-in-place house rule observed: contradictory passages were replaced where they stood; no
contradicting text was appended after them. Companion spec `refine-logs/B5_PROBE_DESIGN.md` carries the
mirrored A1/A2/A3/A6/A9 edits.

| amendment | blocking? | review ref | applied where (this file) | applied where (`B5_PROBE_DESIGN.md`) |
|---|---|---|---|---|
| A1 kill-switch = single per-protocol AND-eligibility rule (removes the acc-only / protocol-unqualified blockquote vs per-protocol-paragraph contradiction) | **BLOCKING** | §3, Item 3 | §6.4 blockquote replaced verbatim + following paragraph rewritten to rationale-only (no independent rule); K1/K2 (§12) realigned | §7 KILL-SWITCH bullets rewritten to the per-protocol AND rule |
| A2 co-equal DEV-side G-repro anchor (4dp match to trainlog `Val_Retrieval` for all 6 arms × 2 protocols) | **BLOCKING** | §3, Item 5 | §6.3 dev-anchor paragraph added; test gate widened to acc+macroF1+roc | §4 two DEV anchor tables added + gate paragraph widened to test AND dev |
| A3 plateau = full `np.flatnonzero` argmax set (non-contiguity); index-median authoritative | no | §2 Item 2 | §5.3 implementation note | §5 tie-break bullet note |
| A4 single-data-split-draw caveat (3 seeds = head-seed only; splits fixed) travels with any claim | no | §2 Item 9 (A4) | §8 guard 4 | — (reporting caveat) |
| A5 val-selected double-dips the 78-dev; final-epoch cleaner/primary | no | §2 Item 9 (A5) | §8 guard 5 | — (reporting caveat) |
| A6 precompute 1000 common dev-resample index arrays (rng 1234, size 78), reuse across all arms | no | §2 Item 9 (A6) | §8 guard 1 addendum | §8 bootstrap common-index note |
| A7 independent calibration-arithmetic hand-check at **verdict** (not executor) | no | §2 Item 9 (A7) | §14 verdict-stage note | §9 verdict note |
| A8 ledger: probe honest arm = the row-3 single test touch (GPU re-derive = 0 extra info) | no | §2 Item 4 | §10 accounting note | — |
| A9 tighten "bit-reproducible on CPU" → deterministic conditional on identical head embeddings | no | §2 Item 7 | §6.1 phrasing | §1 phrasing |
| A10 MARGINAL ⊕ D3-FRAGILE labels compose | no | §2 Item 6 | §7 marginal-pass note | — |
