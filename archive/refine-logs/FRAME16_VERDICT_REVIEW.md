# FRAME16 stage-1 (frozen-Qwen-16f vs banked frozen-Qwen-8f, HateMM) — INDEPENDENT 0-CONTEXT VERDICT REVIEW

**Reviewer role:** independent 0-context verdict reviewer. No prior project context; trusts ONLY primary
artifacts. Renders the binding verdict strictly against the frozen pre-registration
`refine-logs/FRAME16_PREREG.md` VERBATIM. Zero user interaction. CPU-only (no GPU/SLURM/Modal). Modified nothing
except this file; `autoresearch/goal_mllm_plus3/state/` untouched; nothing pushed.
**Out of scope (F0.3 / §8):** the D7 novelty boundary (density is D7-DEAD by construction) and goal-level
satisfaction — this review decides the **PERFORMANCE clause only**.
**Date:** 2026-07-21 NZST.

---

## 0. Hash-freeze verification (done FIRST, before any metric was read)

```
on-disk sha256(refine-logs/FRAME16_PREREG.md)
  = 5c240518217e1ab69cbd52de34c8849450d8b37ad163d68b4247c5f2c791c725
expected (task + refine-logs/FRAME16_FREEZE.md frozen block, commit 0b5cbb5)
  = 5c240518217e1ab69cbd52de34c8849450d8b37ad163d68b4247c5f2c791c725
```
**MATCH.** The prereg on disk is the frozen binding text. NOT VOID. Proceeding.

**Frozen artifacts + reused machinery re-verified on disk at verdict time (no drift since submit):**
```
B a600e74c0a6483095329f9ce15a3df19c842554362f7a3ef1f6e76e26fe3c750  scripts/slurm/gen_embed_mllm_16f.sbatch  [MATCH]
C 99e7e8b10286e22d7913e85c14141c8fa02c90ae27adc0da6facaceeb703864a  scripts/slurm/enc3seed_fb16.sbatch       [MATCH]
extractor d89a912602d763aa055a54f50b0188e302e554b70ff6c0eb872f250bd454b67c  src/utils/generate_VideoMLLM_embedding_HF.py [MATCH]
fork src  9357fa1087e775d059779e6c5f86e19e71b78b2d166f904fa3c71a1a1cbb3268  scripts/slurm/gen_embed_mllm.sbatch          [MATCH]
head anch dbe3fb81800897cb7bac56d71f5d881d54d46421fdbda214df00d4deb0815c3d  scripts/slurm/enc3seed.sbatch (same-code)    [MATCH]
```
**Same-code head block:** `run_one`…`PY` of `enc3seed_fb16.sbatch` vs `enc3seed.sbatch` = `diff` **empty**;
extracted-block sha256 `286a9e44953ff2b2f17af3821f3ed3e254569cb68893fefe6b451b04d6ab9101` for both (matches the
freeze summary and the banked-control block hash). The ONLY manipulated head variables vs the 8f control are
`--model` (`…_HF` → `…_HF-16f`) and `--group_name` (`RAC_video_fb16`), plus derived `--exp_comment`.

**Measurement provenance (raw logs only, job IDs per `FRAME16_SUBMIT_RECORD.md`):** 16f arm = job **13353**,
trainlogs `slurm/logs/enc3s_HateMM_Qwen2.5-VL-7B-Instruct_HF-16f_seed{0,1,2}_13353.trainlog` (chain: extract
**13352** → head **13353**, `afterok`; smoke **13349**). Comparison floor = frozen-Qwen-8f job **12850** raw
trainlogs `slurm/logs/enc3s_HateMM_Qwen2.5-VL-7B-Instruct_HF_seed{0,1,2}_12850.trainlog` — **re-parsed, not
re-run** (§4.1d). Every number below re-derived with the **byte-identical `enc3seed.sbatch` embedded parser**
(val-sel = epoch ≥ warmup 5 maximising `(Val_Retrieval acc, roc)`, report that epoch's TEST metrics; final =
max-epoch TEST), independently re-implemented and hand-verified against raw lines.

---

## 1. Comparison floor — re-derived vs the prereg's §2.1 pinned table (numeric-provenance discipline)

Independent re-parse of the raw 12850 trainlogs. **Every per-seed value, every selected epoch, and both 3-seed
means reproduce the prereg §2.1 EXACTLY to 4dp** — no discrepancy, no blocking flag.

| protocol | s0 acc/mF1 | s1 acc/mF1 | s2 acc/mF1 | mean (mine) | prereg §2.1 |
|---|---|---|---|---|---|
| val-sel (sel ep 28/22/29) | 0.8698/0.8606 | 0.8651/0.8586 | 0.8837/0.8753 | **0.8729/0.8648** | ✔ |
| final-ep (ep 29) | 0.8605/0.8507 | 0.8605/0.8514 | 0.8837/0.8753 | **0.8682/0.8591** | ✔ |

Line-numbered provenance (raw `…_HF_seed{s}_12850.trainlog`, `Test_Retrieval … macroF1` line): s0 val e28
`:293` / final e29 `:303`; s1 val e22 `:235` / final e29 `:299`; s2 val=final e29 `:302`. Seed2's val-sel selects
ep29 (= final), so its two rows coincide — exactly as the prereg states. The pinned floor (val-sel 0.8729/0.8648;
final 0.8682/0.8591) reproduces to 4dp.

---

## 2. 16f arm — raw measured numbers (job 13353), re-parsed + line-verified

Val-selection argmax hand-verified from the raw `Val_Retrieval` epochs (warmup ≥ 5, max acc, roc tie-break):
**s0 → ep23** (val acc 0.8411, the unique max ≥ ep5); **s1 → ep28** (val acc 0.8318, unique max); **s2 → ep20**
(val acc 0.8224, tie with ep26 broken by roc 0.9084 > 0.8946 → ep20). All three match the parser.

| arm | protocol | s0 acc/F1 (sel ep) | s1 acc/F1 (sel ep) | s2 acc/F1 (sel ep) | mean acc/F1 |
|---|---|---|---|---|---|
| **HateMM-16f** (13353) | val-sel | 0.8698/0.8606 (ep23) | 0.8651/0.8567 (ep28) | 0.8605/0.8514 (ep20) | **0.8651/0.8562** |
| **HateMM-16f** (13353) | final-ep | 0.8605/0.8514 (ep29) | 0.8744/0.8666 (ep29) | 0.8744/0.8653 (ep29) | **0.8698/0.8611** |

Line-numbered TEST provenance (raw `…_HF-16f_seed{s}_13353.trainlog`, `Test_Retrieval … macroF1` line): s0
val-sel ep23 `:250` / final ep29 `:305`; s1 val-sel ep28 `:292` / final ep29 `:302`; s2 val-sel ep20 `:221` /
final ep29 `:303`. (All acc values sit on the discrete 215-sample HateMM test grid: 0.8605=185/215,
0.8651=186/215, 0.8698=187/215, 0.8744=188/215, 0.8837=190/215 — every delta below is an integer sample-count
step.)

---

## 3. Outcome table — HateMM 16f vs 8f floor (prereg §7.1), paired within head-seed (Δ = 16f − 8f)

| seed | protocol | 16f acc/F1 | 8f floor acc/F1 (§2.1) | Δ(16f−8f) acc/F1 |
|---|---|---|---|---|
| 0 | val-sel | 0.8698/0.8606 | 0.8698/0.8606 | **+0.0000/+0.0000** |
| 1 | val-sel | 0.8651/0.8567 | 0.8651/0.8586 | **+0.0000/−0.0019** |
| 2 | val-sel | 0.8605/0.8514 | 0.8837/0.8753 | **−0.0232/−0.0239** |
| **mean** | **val-sel** | **0.8651/0.8562** | **0.8729/0.8648** | **−0.0077/−0.0086** |
| 0 | final-ep | 0.8605/0.8514 | 0.8605/0.8507 | **+0.0000/+0.0007** |
| 1 | final-ep | 0.8744/0.8666 | 0.8605/0.8514 | **+0.0139/+0.0152** |
| 2 | final-ep | 0.8744/0.8653 | 0.8837/0.8753 | **−0.0093/−0.0100** |
| **mean** | **final-ep** | **0.8698/0.8611** | **0.8682/0.8591** | **+0.0015/+0.0020** |

**Sign vectors — Δacc(16f−8f):** val-sel `[+0.0000, +0.0000, −0.0232]` = **0/3 strictly positive**; final-ep
`[+0.0000, +0.0139, −0.0093]` = **1/3 strictly positive** (a zero delta is a tie, not positive). Δmacro-F1:
val-sel 0/3 positive; final-ep 2/3 positive.

*Note:* the two `+0.0000` val-sel acc ties are coincidences of the discrete 215-sample grid (16f seed0 val-sel
selects ep23 with test acc 187/215 = the identical value the 8f seed0 arm reaches at its own ep28; likewise 16f
seed1 acc 186/215 equals the 8f floor). Different epochs, identical discrete acc → exact-zero paired delta. Not
a copy artifact — the F1 legs differ (seed1 val-sel F1 −0.0019).

---

## 4. Per-switch rulings (frozen text VERBATIM; each ruled exactly as worded)

### KS-16f-dead — the KILL bar (§3.2, DEV-1 sign formalism) → **KILLED (LoRA-16f AUTO-DEAD)**

> KILL iff, on BOTH protocols, the 16f arm ties-or-regresses the 8f floor — i.e. under each protocol
> `mean paired Δacc ≤ 0` OR the acc sign is not 3/3 positive (so neither protocol produces a clean
> positive-mean-and-3/3-sign result). Then: the frozen-16f cell is KILLED, AND the expensive LoRA-16f stage-2 is
> AUTO-DEAD (banked, never run).

- **val-sel:** mean Δacc = **−0.0077 ≤ 0** → ties-or-regresses (also acc sign 0/3, not 3/3). **TIE/REGRESS.**
- **final-ep:** mean Δacc = +0.0015 > 0, **BUT acc sign = 1/3 (not 3/3 positive)** → the OR-clause fires →
  ties-or-regresses. **TIE/REGRESS.**

BOTH protocols tie-or-regress ⇒ **KS-16f-dead = KILLED.** Neither protocol produced a clean
positive-mean-and-3/3-sign result. **The frozen-16f cell is CLOSED, and LoRA-16f stage-2 is AUTO-DEAD (banked,
never run).** Per carried Review Note 2, this LoRA-16f auto-death is recorded explicitly as a **pre-declared
SPEND verdict** (defund the ≥2-variable contaminated follow-up), **not** a scientific proof that a LoRA-adapted
forward is inert at 16 frames — the gate only ever defunds (conservative direction) and cannot manufacture a pass.

### CONTINUE-to-stage-2 gate (§3.3, internal spend gate) → **NOT CLEARED (LoRA-16f banked, not funded)**

> Continue iff frozen-16f `mean paired Δacc ≥ +0.010` AND acc sign 3/3 positive on ≥ 1 protocol.

- **val-sel:** mean Δacc −0.0077 (< +0.010), sign 0/3. **Fails.**
- **final-ep:** mean Δacc +0.0015 (< +0.010), sign 1/3 (not 3/3). **Fails.**

Neither protocol clears ⇒ **CONTINUE gate NOT cleared → LoRA-16f is NOT funded (banked).** Consistent with the
KS kill above.

### FORMAL verdict bar (§3.4, goal-facing) → **NEGATIVE on both protocols (no PASS)**

> +0.030 acc AND +0.030 mF1, 3/3 seeds positive, under BOTH protocols vs the banked 8f floor, judged
> independently per protocol.

- **val-sel:** mean Δacc −0.0077 / ΔmF1 −0.0086 (both well below +0.030), sign 0/3. **FAIL (NEGATIVE).**
- **final-ep:** mean Δacc +0.0015 / ΔmF1 +0.0020 (both far below +0.030), sign 1/3 acc. **FAIL (NEGATIVE).**

No formal pass under either protocol. (D7-DEAD regardless — even a PASS would have been an engineering/ablation
row, never a novelty win, per F0.3.)

---

## 5. Fixed write-up line (prereg §7.2)

```
HateMM (16f vs 8f):  final-epoch: FAIL; val-selected: FAIL   [FORMAL bar §3.4].
KS-16f-dead: KILLED (LoRA-16f auto-dead).  CONTINUE gate (§3.3): not cleared → LoRA-16f banked.
```

Task-format line (one line, all four bars):

`HateMM-16f: final-epoch: FAIL (Δacc +0.0015 / ΔmF1 +0.0020, acc sign 1/3, < +0.030 bar); val-selected: FAIL
(Δacc −0.0077 / ΔmF1 −0.0086, acc sign 0/3); KS-16f-dead: KILLED — both protocols tie-or-regress (val-sel mean
Δacc ≤ 0; final-ep acc sign not 3/3) → LoRA-16f AUTO-DEAD; CONTINUE gate: NOT cleared (neither protocol ≥ +0.010
acc with 3/3 sign) → LoRA-16f banked, not funded.`

---

## 6. Compliance clauses (prereg binds; checked)

- **Same-code head (§4.1b/§4.2) — COMPLIANT.** `run_one`…`PY` block byte-identical to `enc3seed.sbatch` (`diff`
  empty; block sha `286a9e44…` == freeze). **Runtime confirmation:** the job-13353 seed0 trainlog `:1` Namespace
  shows every config pin landed — `fusion_mode='align'`, `topk=20`, `metric='cos'`, `loss='triplet'`,
  `hybrid_loss=True`, `proj_dim=1024`, `map_dim=1024`, `dropout=[0.2,0.4,0.1]`, `batch_norm=False`, `epochs=30`,
  `batch_size=64`, `lr=0.0001`, `no_hard_negatives=1`, `hard_negatives_loss=True`, `warmup=5`, `lambda_seg=0.0`,
  `archive_feats=None` (archive OFF), and the inert TARC/oracle defaults (`tarc_target_source='off'`,
  `oracle_probe=False`, `lambda_tarc=0.0`) — differing from the 8f floor run ONLY in `model`
  (`…_HF-16f`), `group_name` (`RAC_video_fb16`), and derived `exp_comment`/`output_path`. The Namespace diff is
  exactly what §4.1b pins.
- **Single test-touch (F0.1) — COMPLIANT.** The 3 job-13353 head reads are the ONLY budgeted frozen-16f-encoder
  test evaluations = exactly ONE new single-test-touch; zero test-touch before this verdict (the submit executor
  transcribed no gates/deltas, and §6 of the submit record was left unfilled). Prior HateMM-test exposures under
  the identical `enc3s` protocol (pre-declared, NOT this cell): frozen-CLIP + frozen-Qwen-8f (job 12850),
  generic-LoRA (job 13235 / F53), the LoRA-HateMM verdict, and cand-2 curriculum. This cell's reads are
  re-measurements under that protocol, not first exposures.
- **Collision safety / banked 8f floor untouched (§4.3) — COMPLIANT.** Banked 8f caches present with bytes AND
  mtimes **bit-identical to the submit-record §2 pre-run table**: `train_…_HF.pt` 21358780 B @ 2026-07-02
  00:11:19.293608963, `dev_seen_…_HF.pt` 3073233 B @ 00:13:44.933286504, `test_seen_…_HF.pt` 6173272 B @
  00:18:33.187669051 — **untouched** (distinct out-tag `…_HF-16f` cannot clobber `…_HF`). The fresh 16f caches
  (`…_HF-16f.pt`, dim 3584, dated 2026-07-21, job 13352) exist under the distinct tag. `RAC_video_fb16` head
  group present (the real run); the smoke throwaway `logging/_smoke_fb16` is ABSENT (cleaned per submit record
  §3); smoke log `slurm/logs/smoke_fb16_13349.out` retained as evidence.
- **G-repro items the prereg pins — COMPLIANT.** (a) Extractor sha `d89a9126…` unchanged at verdict time (§0),
  no code edit; `--num_frames`/`--out_model_tag` are pre-existing args (single manipulated variable). (b)
  **Extraction-determinism argument (F0.2):** the frozen forward is deterministic given (weights, sampled indices,
  max_pixels) — `np.linspace` frame sampling (no RNG), `attn=sdpa`, `bf16`, `no_grad`, single forward — so the
  ±band is purely head-seed variance, symmetric with the 8f floor; there is no single-encoder-draw confound, and
  the <16-decodable-frame degeneracy biases 16f *toward* 8f (conservative, against finding an effect). (c)
  **Cache sanity:** the pre-head smoke (job 13349, submit record §3) confirmed `img_feats`/`text_feats` shape
  `(N, 3584)`, finite (NaN=0/inf=0), per-row L2 norm 1.0, zero-vector videos=0, no OOM, and the L283
  masked-scatter assert held at 16 frames; the real head Namespace confirms Dv=Dt=3584 routing via
  `…_HF-16f`.
- **Freeze integrity (carried) — COMPLIANT.** Prereg self-sha and B/C + reused-machinery shas all MATCH on disk
  at verdict time (§0); no drift since submit.

**Carried Review Notes (from `FRAME16_PREREG_REVIEW.md`, APPROVED-WITH-NOTES; all non-blocking):**
1. §4.1a/§4.2 diff-enumeration undercounts the `gen_embed_mllm_16f.sbatch` diff by two cosmetic hunks (a comment
   + a log-tag echo); the manipulated python args are exactly the two claimed. **Non-material** — decision-inert.
2. §3.2 "LoRA-16f auto-dead" is a strong pre-declared inference: a *frozen* forward tying/regressing does not
   strictly prove a *LoRA-adapted* forward cannot extract more from 16 frames. Recorded here **explicitly as a
   SPEND verdict** (defund direction only; cannot manufacture a pass), not a scientific proof of 16f inertness.
3. §3.5 "three nested bars" is a threshold ordering, not literal set-nesting (FORMAL ⟹ CONTINUE holds).
   **Cosmetic.**

**No compliance violations found.** (One documentation observation, decision-inert: the submit record §6 raw
transcription table was left unfilled — the raw per-seed numbers are nonetheless fully recoverable from the
banked trainlogs, and this review re-derived them from the raw logs directly.)

---

## 7. FINAL VERDICT BLOCK (performance clause only)

**Prereg §7.2 fixed write-up:**
```
HateMM (16f vs 8f):  final-epoch: FAIL; val-selected: FAIL   [FORMAL bar §3.4].
KS-16f-dead: KILLED (LoRA-16f auto-dead).  CONTINUE gate (§3.3): not cleared → LoRA-16f banked.
```

**Per-switch (verbatim rulings):**
- **KS-16f-dead (§3.2):** **KILLED.** Both protocols tie-or-regress the 8f floor — val-sel mean Δacc −0.0077 ≤ 0
  (sign 0/3); final-ep mean Δacc +0.0015 > 0 but acc sign 1/3 (not 3/3). Neither protocol yields a clean
  positive-mean-and-3/3-sign result ⇒ **frozen-16f cell CLOSED and LoRA-16f stage-2 AUTO-DEAD (banked)** — a
  pre-declared SPEND verdict (Note 2).
- **CONTINUE gate (§3.3):** **NOT CLEARED.** Neither protocol reaches mean Δacc ≥ +0.010 with 3/3 acc sign
  (val-sel −0.0077/0-3; final-ep +0.0015/1-3) ⇒ LoRA-16f **not funded**, banked.
- **FORMAL bar (§3.4):** **FAIL on both protocols** (val-sel −0.0077/−0.0086, 0/3; final-ep +0.0015/+0.0020,
  1/3) — both far below the +0.030/+0.030 3/3 conjunct. D7-DEAD regardless (F0.3).

**Composite (performance clause only):** Doubling visual sampling density from 8 to 16 frames through the
**frozen** Qwen2.5-VL-7B encoder + mean-pool, paired within head-seed on HateMM, produces **no head-level gain**:
val-selected is a net regression (−0.0077 acc, sign 0/3), and final-epoch is a within-noise +0.0015 acc carried
by a single seed (1/3 sign, one seed regressing). This is the prereg's pre-declared honest most-likely outcome
(F0.5 dilution/redundancy) — the frame-budget door is **CLOSED**: a prose-argued gap ("denser frames untested")
is now a measured-and-closed negative at ~0.6 GPU-h, and the expensive contaminated LoRA-16f stage-2 is
**auto-dead** on the pre-declared spend rule. No formal pass, no CONTINUE, no novelty (density is D7-DEAD by
construction). No compliance violation; the banked 8f floor is untouched; the head is byte-identical same-code.

**Out of scope for this reviewer (F0.3 / §8):** density carries no novelty weight regardless of sign (D7-DEAD by
construction), and goal-level satisfaction is a USER ruling. This review renders the **performance clause only**,
as the frozen prereg mandates.

---

*Reviewer statements: hash verified before any metric was read; the 8f floor re-derived from raw 12850 trainlogs
with the byte-identical enc3seed parser and matches §2.1 to 4dp; the 16f arm (job 13353) re-parsed from raw and
line-verified (per-seed both protocols, selected-epoch argmax, TEST-line line numbers); the head Namespace
confirmed runtime same-code vs the 8f floor; banked 8f caches confirmed untouched by bytes + mtime vs the submit
record; no GPU/SLURM/Modal spent; no state/ mutated; nothing pushed; no goal/novelty claim made.*
