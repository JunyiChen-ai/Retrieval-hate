# MOKA-ZH — INDEPENDENT 0-CONTEXT VERDICT REVIEW

**Reviewer:** independent 0-context verdict reviewer (no role in the recon, prereg, freeze, re-freeze
or execution of this family).
**Date:** 2026-07-26 (`date -u` `Sun Jul 26 05:25:16 UTC 2026`).
**Repo state at review:** `HEAD = ed609eb` ("moka submit round 2: S6' CLOSEOUT …"), branch `main`.
**Cost of this review:** CPU only — `sha256sum`, `git`, an independently written trainlog re-parser,
`grep`/`sed` over completed logs. **ZERO GPU / SLURM / Modal. No job submitted. No new test metric
produced** (the 6 head runs were already complete; this review re-parses their banked trainlogs).
**No `state/` mutation. No `research-wiki/` mutation. No frozen artifact edited. Not pushed.**

---

## VERDICT (binding language, §7 fixed write-up format)

```
KS-MOKA-0  (machinery, incl. identity control + KS-parity bit-exact): PASS
KS-MOKA-0b (merge drift): worst mean per-item cosine = 0.99954879  -> same-path floor MANDATORY (FIRED)
KS-MOKA-2  (non-degeneracy floor ONLY, per note N1): median ||A_v-A_t||_F/||A_t||_F = 1.4170
           -> the two down-projections did NOT collapse. NOT evidence that routing is real.
MokA-ZH:   final-epoch: fail; val-selected: fail   [FORMAL §2.3, vs the §3.4-binding UNMERGED floor 13573]
           (also fail; fail vs the secondary merged floor 13150)
KS-MOKA-1: parse (A) applied (note N4) -> primary clause does NOT fire on the binding pairing;
           secondary within-noise clause does NOT fire. Under parse (B) the primary clause DOES fire.
           Against the secondary merged floor the kill fires under BOTH parses and the secondary
           clause fires too. ZH arm is NOT PROMOTED on any reading; stage-2 HateMM remains
           UNAUTHORISED (§3.10 new bite + §8 amendment), and D7 below removes its scientific premise.
KS-MOKA-3: neither moved -> NULL-OP at the stream level (reading 3 of §3.7).
           Text FLAT under both floors; image AMBIGUOUS under both floors (not MOVED).
           Neither the "text-side mechanism" reading nor the "9th law-I instance" reading is earned.
           Visual-modality-protection narration is BARRED (image did not move AND the head did not follow).
KS-regression: NOT triggered (no protocol reaches -0.030 mean Dacc against either floor).
F0.2 restated: ONE MokA SFT draw; --seed varies the head only; encoder-seed noise is CONFOUNDED with
  the routing effect and is NOT separable within this budget.
F0.6 restated: our SFT is 94.6 % vision tokens (median 2,688 vision + 153 text); MokA's own shipped
  regime is 98.4 % TEXT (16,128 vs 256). A_t is undiluted but data-starved (~5.4 % of positions);
  transfer from MokA's reported gains is weak. The measured outcome is consistent with the low end
  of the frozen 5-8 % prior.
```

**One-line verdict:** **MEASURED — NOT PROMOTED. Modality-routed LoRA (MokA `A`-split) FAILS the
FORMAL bar on ZH under both protocols against the binding same-path floor and against the merged
floor, and `KS-MOKA-3` records a stream-level NULL-OP; the only above-noise number in the family —
the +0.0268 val-selected accuracy gain over the unmerged floor — is adjudicated in D7 as NOT
attributable to routing.**

---

## 0. Governing clauses, quoted VERBATIM before application

Re-hashed at review time: `sha256(refine-logs/MOKA_PREREG.md) =`
`dc3f1078a89fc2e1de30c870103c2b7f2986fd419698d6c49b5b9ec0966c53f8` — **MATCHES** the value the task
pins and the value frozen at `MOKA_FREEZE.md §1` / re-affirmed at `§7.1`.

**§3.4 `KS-MOKA-0b`** — verbatim:

> **Bar:** **mean per-item cosine(unmerged, merged) ≥ 0.9999** on the banked ZH cache, on **all** 6
> (split × stream) cells. **If ANY cell < 0.9999**, a same-path **unmerged** floor head run (3 seeds,
> +0.05 GPU-h, **+3 test evaluations**) becomes **MANDATORY** before any verdict, and the arm is then
> paired against **that** floor instead of 13150.

**§3.5 `KS-MOKA-1`** — verbatim:

> **If on BOTH protocols the 3-seed mean paired Δacc ≤ 0, OR the acc sign is not 3/3 positive, the ZH
> arm is DEAD and the HateMM stage-2 leg is AUTO-DEFUNDED** (saves ≈ 4.2 GPU-h). Bank as Law-I / FLAT.
> **Secondary read (within-noise kill):** mean paired Δacc `< +0.015` on **both** protocols (inside the
> ±0.014 band, §2.3) ⇒ also **KILL**. State the kill explicitly at verdict time.

**§3.1 decision rule** — verbatim (from `exp-encoder-3seed.md:73-85`):

> (4) **pass = mean paired Δacc ≥ +0.030 AND mean paired Δmacro-F1 ≥ +0.030 AND sign 3/3 positive**;
> (5) headline claim requires pass on ≥ 2 datasets under a stated protocol; both protocols judged
> separately; verdict written exactly "final-epoch: pass/fail; val-selected: pass/fail".

**§2.3 FORMAL** — verbatim:

> **FORMAL (vs 13150):** val-sel mean acc ≥ **0.8622** AND mF1 ≥ **0.8315**; final mean acc ≥ **0.8756**
> AND mF1 ≥ **0.8473**; **3/3 per-seed positive on both metrics, both protocols.**
> - **Head-seed noise band:** ±**0.014**

**§3.6 `KS-MOKA-2`** — verbatim: *"If the median layer is < 0.05 (5 % relative), the two
down-projections have converged: the arm is a NULL-OP and any observed delta is head-seed noise, NOT
routing"* — **as bounded by reviewer note N1**, verbatim from `MOKA_PREREG_REVIEW.md`:

> **Binding on the verdict write-up:** `KS-MOKA-2` must be reported as a **non-degeneracy floor**
> (the two `A`s did not collapse onto each other), **not** as evidence that routing changed the
> learned function. … **The `median ≥ 0.05 ⇒ routing real` direction is unsupported.**

**§3.7 `KS-MOKA-3`** — verbatim:

> movement rule … **MOVED** iff `dAUC ≥ +0.010` on train-LOO **and** `≥ +0.005` on dev with the same
> sign; **FLAT** iff `|dAUC| < 0.010` on train-LOO … Three **pre-declared** readings …
> - **text moved** ⇒ the §F0.5 bet is confirmed and the result **must be reported as a *text-side*
>   mechanism — NEVER as "MokA protected the visual modality."**
> - **image moved, head flat** ⇒ the **9th law-I instance**; report as such.
> - **neither moved** ⇒ null-op; cross-check `KS-MOKA-2`.
>
> **This clause is binding on the write-up regardless of the performance outcome.** … a PASS may not
> be narrated as visual-modality protection unless `KS-MOKA-3` independently shows the image stream
> moved *and* the head followed.

**F0.2** — verbatim: *"**ONE** MokA SFT run. `--seed` varies the **head** only … here the arm's
encoder is a **different SFT draw** from the floor's, so **encoder-seed noise is confounded with the
routing effect and is NOT separable within this budget.** This is a material limitation and must be
restated at verdict time"*.

**F0.3 / D7 + credit clause** — verbatim: *"transplant novelty is claimable **ONLY** as (a) *first
application* of modality-routed PEFT to hateful-video encoders, **and** (b) with **MokA explicitly
credited** … Any phrasing that implies we invented modality-routed LoRA is **banned**."*

**F0.6** — verbatim (abridged to the load-bearing sentences): *"median vision share 94.6 % … MokA's
own shipped setting is the OPPOSITE … **98.4 % TEXT** … the routing makes `A_t` *undiluted* (the bet)
but also *data-starved* … ~18× fewer token-gradients … it is a reason to sit at **5 %**, not 8 % …
**This paragraph must be restated at verdict time whatever the outcome.**"*

**§3.8 `KS-regression`** — verbatim: *"MokA below floor by **≥ 0.030** mean Δacc on either protocol ⇒
report as a measured **REGRESSION** finding"*.

---

## D1. PROVENANCE — **PASS (independent re-derivation reproduces §S5' bit-for-bit)**

### D1.1 Prereg + REFREEZE-1 shas on disk

| # | path | REFREEZE-1 pin (`MOKA_FREEZE.md §7.1`) | on disk now | match |
|---|---|---|---|---|
| P | `refine-logs/MOKA_PREREG.md` | `dc3f1078…0966c53f8` | `dc3f1078…0966c53f8` | ✔ |
| A | `src/moka/routed_lora.py` | `6b7bdb6c…e37c85be` | `6b7bdb6c…e37c85be` | ✔ |
| B | `src/moka/train_moka.py` | `fae40487…891c9749` | `fae40487…891c9749` | ✔ |
| C | `src/utils/generate_VideoMLLM_embedding_lora_HF.py` | `75bb8156…c48612399` | `75bb8156…c48612399` | ✔ |
| D | `scripts/analysis/moka_smoke.py` | `bd258553…bf46c4ef` | `bd258553…bf46c4ef` | ✔ |
| E | `scripts/slurm/lora_sft_moka.sbatch` | `020dd10b…482745e6` | `020dd10b…482745e6` | ✔ |
| F | `scripts/slurm/moka_extract_head.sbatch` | `fd1b7f29…2f48b31bde` | `fd1b7f29…2f48b31bde` | ✔ |
| G | `RA-HMD/…/mhc_zh_qwen25vl_lora_moka_sft.yaml` | `51b883e9…c8f6e49764` | `51b883e9…c8f6e49764` | ✔ |

Reused-unchanged machinery: `src/run_rac.py = b85eb72a690bc8fccc2ff5d5358fd6523359bf6596d2b2a0d6d0701bec9e53e3` ✔;
`scripts/slurm/enc3seed_zh_b3.sbatch = 4379224671defe7dafb638c4f0c8b69295a27d11646b685912a249e2385e29ad` ✔;
`git status --porcelain` on `run_rac.py`, `loss.py`, `classifier.py`, `retrieval.py`, `src/moka/`,
both frozen sbatch, the smoke and the extractor = **EMPTY (CLEAN)** ✔;
gitlink `RA-HMD/LLAMA-FACTORY-Ver202512` = `160000 a912747c408b3c661b4029ecf1d88b9d91c7f1a8` ✔.
`run_one()` block: `enc3seed_zh_b3.sbatch:42-83` and `moka_extract_head.sbatch:112-153` both hash to
`286a9e44953ff2b2f17af3821f3ed3e254569cb68893fefe6b451b04d6ab9101` ✔; the throwaway umfloor runner's
`run_one()` (`:61-102`) hashes to the **same** value and `diff` against the anchor is **empty** ✔ —
so the contingency floor was produced by byte-identical head code to the arm and to floor 13150.

### D1.2 Independent re-parse of all 6 head trainlogs (+ the 3 merged-floor references)

Parser written fresh for this review (line scan + token split; deliberately **not** the frozen regex),
applying the pinned protocol: **val-sel = epoch ≥ warmup 5, max `Val_Retrieval` acc, `roc` tie-break;
final = max epoch**; metric read from the macro-variant `Test_Retrieval` line. All 9 logs carry 30 Val
and 30 Test epochs.

**Arm** — `slurm/logs/enc3s_MHC_zh_Qwen2.5-VL-7B-Instruct-LoRA-moka_HF_seed{s}_13566.trainlog`

| seed | val-sel ep | val-sel acc / mF1 | line | final ep | final acc / mF1 | line | §S5'.3 |
|---|---|---|---|---|---|---|---|
| 0 | 23 | 0.8121 / 0.7679 | 244 | 29 | 0.8456 / 0.8107 | 299 | ✔ |
| 1 | 27 | 0.8389 / 0.8039 | 286 | 29 | 0.8456 / 0.8080 | 305 | ✔ |
| 2 | 28 | 0.8456 / 0.8107 | 289 | 29 | 0.8456 / 0.8107 | 299 | ✔ |
| **mean** | | **0.8322 / 0.7942** | | | **0.8456 / 0.8098** | | ✔ |

**UNMERGED floor (binding)** — `…-LoRA_HF-um_seed{s}_13573.trainlog`

| seed | val-sel ep | val-sel acc / mF1 | line | final ep | final acc / mF1 | line | §S5'.3 |
|---|---|---|---|---|---|---|---|
| 0 | 5 | 0.7718 / 0.7259 | 80 | 29 | 0.8456 / 0.8181 | 297 | ✔ |
| 1 | 25 | 0.8121 / 0.7742 | 267 | 29 | 0.8255 / 0.7956 | 304 | ✔ |
| 2 | 26 | 0.8322 / 0.8023 | 271 | 29 | 0.8456 / 0.8181 | 299 | ✔ |
| **mean** | | **0.8054 / 0.7675** | | | **0.8389 / 0.8106** | | ✔ |

**MERGED floor 13150 (secondary)** — `…-LoRA_HF_seed{s}_13150.trainlog`

| seed | val-sel ep | val-sel acc / mF1 | line | final ep | final acc / mF1 | line | prereg §2.1 |
|---|---|---|---|---|---|---|---|
| 0 | 20 | 0.8322 / 0.8023 | 220 | 29 | 0.8456 / 0.8181 | 302 | ✔ |
| 1 | 26 | 0.8255 / 0.7956 | 275 | 29 | 0.8389 / 0.8113 | 303 | ✔ |
| 2 | 19 | 0.8389 / 0.8065 | 207 | 29 | 0.8523 / 0.8226 | 298 | ✔ |
| **mean** | | **0.8322 / 0.8015** | | | **0.8456 / 0.8173** | | ✔ |

**Every one of the 36 arm+floor values, every selected epoch and every source line number reproduces
`MOKA_SUBMIT_RECORD.md §S5'.3` exactly**, and the merged-floor block re-derives prereg §2.1 exactly
(the third independent re-derivation of 13150 on record, after the prereg's and the executor's).

**Selection audit of the one anomalous cell.** UMFLOOR seed 0 selected **epoch 5** — the earliest
epoch the warmup rule admits. I verified this is protocol-correct, not a parser artifact: the maximum
`Val_Retrieval` acc over epochs ≥ 5 is **0.8718**, attained at exactly two epochs — **ep 5 (roc
0.9207, l.79)** and ep 24 (roc 0.9129, l.251) — and the pinned `roc` tie-break selects ep 5. That cell
carries the family's single largest paired delta (−0.0604 acc vs the merged floor, i.e. **−9 test
items of 149**) and is load-bearing for D7 below.

### D1.3 `KS-MOKA-0b` 6-cell table — verified against the job-2 log verbatim

`slurm/logs/moka_eh_13566.out:155-162`, transcribed here from the log, not from the record:

| split | stream | mean per-item cos | min per-item cos | ≥ 0.9999 ? |
|---|---|---|---|---|
| train (579) | img | 0.99984443 | 0.99894041 | **NO** |
| train | text | 0.99957055 | 0.99807644 | **NO** |
| dev_seen (78) | img | 0.99987048 | 0.99896044 | **NO** |
| dev_seen | text | **0.99954879** | 0.99750400 | **NO** |
| test_seen (149) | img | 0.99983090 | 0.99933851 | **NO** |
| test_seen | text | 0.99955094 | 0.99884427 | **NO** |

`[KS-MOKA-0b] WORST mean per-item cosine over all 6 (split x stream) = 0.99954879` /
`[KS-MOKA-0b] >= 0.9999 ? False`. **All 6 cells < 0.9999 ⇒ §3.4 FIRES.** This probe reads no labels;
**test-touch 0** ✔.

### D1.4 Other primary readouts verified at source

- `KS-MOKA-2`, `logging/slurm/lora_sft_moka_13552.out:1994-1995` verbatim:
  `[moka_sft] KS-MOKA-2 rel ||A_v-A_t||_F/||A_t||_F : min 1.4039 median 1.4170 max 1.4292`, preceded by
  `[moka_sft] adapter keys: lora_A 196 | lora_A_v 196 | lora_B 196 | total tensors 588 | params 58490880`
  — both frozen job-1 post-run asserts pass (196/196/196 and the exact 58,490,880 of §F0.7).
- `KS-MOKA-3` JSONs re-read from `refine-logs/MOKA_KS3_stream_decomp_vs_{unmerged,merged}.json`;
  thresholds recorded in-file as `train_loo 0.01 / dev 0.005`, `K=20`, `n_train 579`, `n_dev 78`.
- Extraction comparability: both the Stage-A0 `-um` extraction and the `--moka` extraction ran with
  **identical** `Namespace(... num_frames=8, max_pixels=151200, splits='train,val,test', limit=0)`
  (`moka_eh_13566.out:65` and `:164`), differing only in `no_merge=True, moka=False` vs
  `no_merge=False, moka=True` ✔. `KS-parity` (GPU smoke job 13551 LEG 3) reproduced the banked generic
  cache at `max|delta| = 0.000000e+00` on both streams ⇒ **no stack drift** between the banked merged
  floor and this cell's extractor.
- Test-touch ledger: **6 spent / 6 budgeted** (3 arm + 3 contingent, the latter reserved in F0.1 and
  released by §3.4). **No unbudgeted test evaluation.** ✔

**D1 result: PASS.** Provenance, shas, protocol application and every transcribed number are
independently confirmed. I found **no numeric discrepancy** anywhere in §S5'.

---

## D2. §3.4 — binding pairing is the UNMERGED floor. **Correctly applied.**

`KS-MOKA-0b` fired on **all 6** cells (D1.3), so §3.4's contingency is unconditional: the same-path
unmerged floor run became **MANDATORY** and *"the arm is then paired against **that** floor instead of
13150."* The executor ran it (job 13573, 3 seeds, byte-identical `run_one`, fresh group
`RAC_video_moka_umfloor`, `--force False`) and designated the pairing accordingly. **The RAW pairing
used the binding floor.** ✔ The merged-floor comparison is correctly labelled secondary/non-binding —
though see D7, where it becomes the decisive *interpretive* evidence without ever becoming the
*binding* bar.

**Paired deltas, arm − UNMERGED floor (BINDING)** — re-derived by this reviewer:

| protocol | metric | seed 0 | seed 1 | seed 2 | mean | sd | sign + | t(2)† |
|---|---|---|---|---|---|---|---|---|
| val-sel | Δacc | +0.0403 | +0.0268 | +0.0134 | **+0.0268** | 0.0135 | **3/3** | +3.456 |
| val-sel | ΔmF1 | +0.0420 | +0.0297 | +0.0084 | **+0.0267** | 0.0170 | **3/3** | +2.720 |
| final-ep | Δacc | +0.0000 | +0.0201 | +0.0000 | **+0.0067** | 0.0116 | **1/3** | +1.000 |
| final-ep | ΔmF1 | −0.0074 | +0.0124 | −0.0074 | **−0.0008** | 0.0114 | **1/3** | −0.121 |

† paired-t reported **as an effect-size descriptor only**, per §3.1(3) — *no significance claim*.

**Paired deltas, arm − MERGED floor 13150 (secondary, non-binding)**

| protocol | metric | seed 0 | seed 1 | seed 2 | mean | sd | sign + |
|---|---|---|---|---|---|---|---|
| val-sel | Δacc | −0.0201 | +0.0134 | +0.0067 | **+0.0000** | 0.0177 | 2/3 |
| val-sel | ΔmF1 | −0.0344 | +0.0083 | +0.0042 | **−0.0073** | 0.0236 | 2/3 |
| final-ep | Δacc | +0.0000 | +0.0067 | −0.0067 | **+0.0000** | 0.0067 | 1/3 |
| final-ep | ΔmF1 | −0.0074 | −0.0033 | −0.0119 | **−0.0075** | 0.0043 | 0/3 |

**FLOOR-vs-FLOOR, unmerged 13573 − merged 13150 (routing ENTIRELY ABSENT)**

| protocol | metric | seed 0 | seed 1 | seed 2 | mean | sign + |
|---|---|---|---|---|---|---|
| val-sel | Δacc | −0.0604 | −0.0134 | −0.0067 | **−0.0268** | 0/3 |
| val-sel | ΔmF1 | −0.0764 | −0.0214 | −0.0042 | **−0.0340** | 0/3 |
| final-ep | Δacc | +0.0000 | −0.0134 | −0.0067 | **−0.0067** | 0/3 |
| final-ep | ΔmF1 | +0.0000 | −0.0157 | −0.0045 | **−0.0067** | 0/3 |

All three tables reproduce §S5'.4 / §S5'.5 / §S5'.6 exactly.

---

## D3. `KS-MOKA-1` applied exactly as frozen — **parse (A) pinned (note N4)**

The frozen sentence is quoted verbatim in §0. Note **N4** flagged its quantifier as ambiguous and
required the verdict to state which parse was applied. **I pin parse (A)** — *"on both protocols:
(Δacc ≤ 0 **or** sign ≠ 3/3)"* — because the fronted adverbial **"on BOTH protocols"** scopes the
entire following predicate, and because only under (A) is the clause the "**ZH decisive**" switch its
own §3.5 heading announces. Both parses are evaluated below; per N4 neither can manufacture a false
PASS, and here neither does.

**Primary clause, on the §3.4-binding pairing (arm − unmerged floor):**

| protocol | mean Δacc | ≤ 0 ? | acc sign | ≠ 3/3 ? | per-protocol failure? |
|---|---|---|---|---|---|
| val-selected | +0.0268 | NO | 3/3 | NO | **no** |
| final-epoch | +0.0067 | NO | 1/3 | **YES** | **yes** |

- **Parse (A):** the disjunction must hold on **both** protocols. It holds on final-epoch and **fails
  on val-selected** ⇒ the primary clause does **NOT** fire. The ZH arm formally "**survives**".
- **Parse (B)** (`(∀protocol Δacc ≤ 0) ∨ (sign ≠ 3/3)`, the sign clause unscoped): the second
  disjunct is satisfied by final-epoch's 1/3 ⇒ the primary clause **DOES** fire, ZH arm **DEAD**,
  HateMM stage-2 **AUTO-DEFUNDED**.

**Secondary read (within-noise kill), binding pairing:** *"mean paired Δacc `< +0.015` on **both**
protocols … ⇒ also **KILL**."* val-selected is **+0.0268 ≥ +0.015** ⇒ the condition is not met on both
protocols ⇒ the within-noise kill does **NOT** fire on the binding pairing. **Stated explicitly, as
§3.5 requires.**

**Same clause on the secondary merged floor 13150 (recorded, not binding):** val-sel Δacc = **+0.0000**
(≤ 0, and sign 2/3), final Δacc = **+0.0000** (≤ 0, sign 1/3) ⇒ the primary clause fires under **both**
parses, and the secondary within-noise clause fires as well (+0.0000 < +0.015 on both protocols).

**Ruling.** `KS-MOKA-1`'s mechanical auto-defund switch does **not** fire on the binding pairing under
the parse I pin, and does fire under the alternative parse and under the merged-floor comparison. This
divergence is exactly the contained case N4 pre-adjudicated ("the only reachable divergence is a
one-protocol-positive arm that 'survives' … That outcome is contained anyway"). **It changes nothing
operationally:** stage 2 is separately barred by §3.10 ("a HateMM arm … is a **new** pre-declared arm
and re-costs a bite"), by §8 (it "would … unlock the stage-2 HateMM leg under a **new prereg
amendment**" whose own case recon §4.2 already calls "arithmetically implausible"), and by job 1's
hard `exit 2` refusal of `HateMM`. **And D7 removes the scientific premise on which any such
amendment would rest.** No stage-2 GPU is authorised by this verdict.

---

## D4. FORMAL applied exactly as frozen — **final-epoch: fail; val-selected: fail**

The frozen bar is conjunctive: **mean paired Δacc ≥ +0.030 AND mean paired ΔmF1 ≥ +0.030 AND sign 3/3
positive**, judged per protocol, both protocols required for "clears FORMAL" (§2.3's bundled phrasing;
note N5(ii) confirms the bundled reading is the stricter one and that reporting is per protocol).

**Against the §3.4-BINDING unmerged floor 13573:**

| protocol | Δacc mean | ≥ +0.030 ? | ΔmF1 mean | ≥ +0.030 ? | sign 3/3 ? | clause |
|---|---|---|---|---|---|---|
| **val-selected** | +0.0268 | **NO** (−0.0032) | +0.0267 | **NO** (−0.0033) | yes (3/3, 3/3) | **FAIL** |
| **final-epoch** | +0.0067 | **NO** | −0.0008 | **NO** | **no** (1/3, 1/3) | **FAIL** |

Equivalent absolute-level check (means, identical arithmetic because the pairing is complete): val-sel
required acc ≥ 0.8354 / mF1 ≥ 0.7975, measured **0.8322 / 0.7942** — both short; final-epoch required
acc ≥ 0.8689 / mF1 ≥ 0.8406, measured **0.8456 / 0.8098** — both short.

**Against the secondary merged floor 13150** (the §2.3 literal thresholds): required val-sel acc ≥
0.8622 / mF1 ≥ 0.8315, measured **0.8322 / 0.7942**; required final acc ≥ 0.8756 / mF1 ≥ 0.8473,
measured **0.8456 / 0.8098**. **FAIL on all four**, with mean Δ of +0.0000/−0.0073 (val-sel) and
+0.0000/−0.0075 (final).

**FORMAL verdict, in the exact §3.1(5) wording: `final-epoch: fail; val-selected: fail`.** The
val-selected leg misses the accuracy bar by **0.0032** (less than one test item, 1/149 = 0.0067) and
the macro-F1 bar by 0.0033 — but it misses, and D7 shows that even the +0.0268 it does carry is not
attributable to the manipulated variable. §3.1(5)'s ≥2-dataset headline is structurally unreachable in
stage 1 in any case (single dataset, §3.1).

**`KS-regression` (§3.8):** no protocol reaches −0.030 mean Δacc against either floor (worst is
+0.0000 vs merged). **Not triggered.**

---

## D5. `KS-MOKA-2` per note N1 — **non-degeneracy floor ONLY**

Measured: `min 1.4039 | median 1.4170 | max 1.4292` over 196 layers (true even-sample median; the
round-1 P1-B off-by-one is fixed and the fix was rehearsed on a real 196-layer adapter in GPU-smoke
LEG 1b, `statistics.median == (v97+v98)/2 -> True`).

N1's calibration, re-quoted: two **independent** Kaiming draws at the deployed `A` shape (16 × 3584)
sit at **1.4136** (min 1.4089 / max 1.4191, 20 trials), while a **trained** `lora_A`'s *total*
displacement over the real 3-epoch deployed ZH SFT is median **0.0506** / max **0.1267**.

**What MAY be claimed:** the frozen `< 0.05 ⇒ NULL-OP` trap did not spring — `A_v` and `A_t` did not
collapse onto each other. Nothing more.

**What MAY NOT be claimed:** that routing changed the learned function. The measured **1.4170** is
within the spread of the independent-draws reference **1.4136** (which itself ranges 1.4089–1.4191);
it is arithmetically indistinguishable from "the two matrices are still two independent random draws
that each moved by ≲ 0.13 relative-Frobenius." Reaching the 0.05 bar would require ≈ 27× the entire
measured training displacement of an `A` in this recipe, so the bar is **structurally unable to fire**
and the `median ≥ 0.05 ⇒ routing real` direction is **unsupported**. **This number must never be
reported, in the paper or elsewhere, as evidence that routing is real.**

**Where the routing-is-active evidence actually lives (N1's own designation):** GPU-smoke LEG 1 on the
real `PeftModelForCausalLM` under `MOKA_STRICT=1` recorded `hook_calls 314`, `routed_calls 77,224`,
**`fallback_calls 0`** across 10 optimizer steps *and* 3 eval loops, and job 13552 completed all 3
epochs under `MOKA_STRICT=1` with no strict raise. **Routing was mechanically live on every routed
call.** That is a machinery fact, not a function-change fact — and `KS-MOKA-3` (D6) is where the
function-change question is answered, negatively.

---

## D6. `KS-MOKA-3` §3.7 — reading assignment: **"neither moved" ⇒ NULL-OP**

Mechanical labels, re-read from the two JSONs (thresholds and `K=20` recorded in-file; train + dev
only, **zero test-touch**):

| floor | stream | Δ train-LOO AUC | ≥ +0.010 ? | Δ dev AUC | ≥ +0.005 same sign ? | label |
|---|---|---|---|---|---|---|
| unmerged (binding) | img | **+0.0137** | yes | **−0.0121** | **no (opposite sign)** | **AMBIGUOUS** |
| unmerged | text | −0.0007 | no | +0.0107 | — | **FLAT** |
| unmerged | concat | +0.0000 | no | +0.0143 | — | FLAT |
| merged (§3.7 literal) | img | **+0.0120** | yes | **+0.0043** | **no (0.0043 < 0.005)** | **AMBIGUOUS** |
| merged | text | +0.0018 | no | +0.0071 | — | **FLAT** |
| merged | concat | +0.0006 | no | +0.0143 | — | FLAT |

**Assignment.** The frozen rule certifies **MOVED** only on the conjunction. **Neither stream is
MOVED under either floor.** Therefore §3.7's **third** pre-declared reading applies:
**"neither moved ⇒ null-op; cross-check `KS-MOKA-2`."** The cross-check is performed in D5 and, per
N1, can only report a non-degeneracy floor — so it neither confirms nor rebuts the null-op reading;
`fallback_calls == 0` establishes that the null-op is **functional, not mechanical** (routing ran on
77,224 calls and still moved no stream past the bar).

**Mandatory reporting discipline, applied:**

1. **The "text-side mechanism" reading is NOT earned.** The §F0.5 bet — *"the dominant TEXT stream
   gets its own `A_t`, undiluted"* — predicted the text stream would move. It did not: Δ train-LOO
   **−0.0007** (binding floor) / **+0.0018** (merged floor), i.e. **FLAT** under both, against F45's
   reference where a *shared* `A` moved ZH text train-LOO 0.847 → 0.925. (Provenance cross-check: the
   merged floor's text train-LOO **0.9254** reproduces F45's published **0.925**.) **The prereg's own
   bet is refuted at the mechanism level.**
2. **The "9th law-I instance" reading is NOT earned either.** It requires *image **moved**, head
   flat*. The head is indeed flat (+0.0000 mean Δacc vs the merged floor on **both** protocols), but
   the image stream is **AMBIGUOUS**, not MOVED, under both floors — the dev leg fails by sign under
   the binding floor and by **0.0007 AUC** under the merged floor. **This result must NOT be banked as
   the 9th law-I instance.** The honest description is *law-I-**shaped** but not law-I-**certified***:
   the only stream showing any movement pressure is the image stream, and the head did not follow it —
   which is directionally consistent with the eight banked law-I instances without meeting the frozen
   bar that would let us count a ninth.
3. **Visual-modality-protection narration is BARRED, doubly.** §3.7: *"a PASS may not be narrated as
   visual-modality protection unless `KS-MOKA-3` independently shows the image stream moved **and**
   the head followed."* There is no PASS; the image stream did not move by the frozen rule; and the
   head did not follow. **Any sentence in the paper or elsewhere claiming MokA protected the visual
   modality on our data is prohibited by this clause and unsupported by this measurement.**
4. **Note N5(i) — the "both moved" case** did not arise (no stream moved), so no unenumerated-case
   statement is owed.

---

## D7. THE LOAD-BEARING ADJUDICATION — is the val-selected +0.0268 attributable to routing?

### D7.1 The arithmetic, stated exactly

Three complete, seed-paired comparisons over the same 149-item ZH test split (granularity 1/149 =
0.0067 accuracy per item):

| val-selected comparison | mean Δacc | in test items | mean ΔmF1 | sign + |
|---|---|---|---|---|
| **arm − unmerged floor** (binding) | **+0.0268** | +6 / +4 / +2 → mean **+4** | +0.0267 | 3/3 |
| **unmerged floor − merged floor** (routing ABSENT) | **−0.0268** | −9 / −2 / −1 → mean **−4** | −0.0340 | 0/3 |
| **arm − merged floor** (secondary) | **+0.0000** | −3 / +2 / +1 → mean **0** | −0.0073 | 2/3 |

The three rows are one identity: `(arm − merged) = (arm − unmerged) + (unmerged − merged)`, i.e.
`+0.0000 = +0.0268 + (−0.0268)`. **The arm's entire binding val-selected gain is exactly the size of
the unmerged path's own deficit, and the arm lands on the merged floor's level, not above it.**

### D7.2 What produced the −0.0268 that the arm "recovers"

**Nothing that MokA does.** Both floors are the **same banked generic adapter** `logging/lora/MHC_zh`,
the same 8 frames, the same `max_pixels=151200`, the same prompts, the same head code, the same three
head seeds. The **only** difference between them is `merge_and_unload()`'s folded `W+BA` single matmul
versus the unmerged `Wx + B(Ax)` — a **bf16 accumulation-order** difference whose feature-space size
is `KS-MOKA-0b`: mean per-item cosine **0.99955–0.99987** on all six cells. In method space this
manipulation is **null by construction**: routing is entirely absent from it, and it is not even a
different model — it is the same weights evaluated in a different arithmetic order.

Yet that null manipulation moved the val-selected readout by **−0.0268 acc / −0.0340 mF1**, 0/3 sign,
driven overwhelmingly by one seed (**−0.0604 = −9 items**) whose val-selection collapsed from a
mid-training epoch to **epoch 5**, the earliest the warmup rule admits, on a 78-item dev split where
two epochs tied at Val acc 0.8718 and the `roc` tie-break picked the earlier one (D1.2). The
final-epoch protocol — which performs no selection — shows the same floor-vs-floor manipulation at
only **−0.0067** (one item). **The gap is a selection artifact of the 78-item dev wall amplified by a
numerically-null perturbation, not an encoder-quality difference.**

### D7.3 Ruling

> **RULING (D7). The +0.0268 val-selected accuracy gain of the MokA arm over its same-path unmerged
> floor is NOT attributable to modality routing. It is arithmetically indistinguishable from — and
> exactly cancelled by — the −0.0268 that the unmerged path loses to the merged path with routing
> entirely absent; the arm's absolute position against the banked merged floor 13150 is +0.0000 acc /
> −0.0073 mF1 on val-selected and +0.0000 acc / −0.0075 mF1 on final-epoch, i.e. the routed encoder
> lands on the generic-LoRA floor and not above it. The measurement channel that produced the +0.0268
> has a demonstrated sensitivity of the same magnitude (−0.0268 acc, −0.0340 mF1) to a manipulation
> known to be method-null — larger than the ±0.014 house head-seed band and larger than the effect it
> is being asked to certify — so under the pre-registered dual-protocol readout this quantity carries
> no discriminating power about the manipulated variable. The correct reading is that routing merely
> recovers the unmerged path's own selection-driven degradation; the honest summary of the cell is
> "modality-routed LoRA lands the ZH encoder on the shared-`A` floor," not "modality-routed LoRA gains
> +0.027 val-selected."**

### D7.4 The three independent supports for that ruling

1. **F0.2 (mandatory restatement, and it bites here).** *"the arm's encoder is a **different SFT
   draw** from the floor's, so **encoder-seed noise is confounded with the routing effect and is NOT
   separable within this budget.**"* There is **one** MokA SFT draw; `--seed` varied the head only. So
   even if the +0.0268 had survived the cancellation above, it could not have been separated from
   encoder-draw noise without an SFT seed sweep (~9 GPU-h) that the family budget never held. **No
   causal attribution to routing is available from this design at any effect size below the FORMAL
   bar, and the measured effect is below it.**
2. **The ±0.014 band (§2.3) does not rescue the number.** The binding val-sel per-seed deltas are
   +0.0403 / +0.0268 / +0.0134 (sd 0.0135) — the band is as wide as the mean, and the smallest seed
   sits **at** the band edge. The paired-t descriptor (+3.456, §3.1(3) — descriptor only, no
   significance claim) is driven by the same seed-0 cell that D7.2 identifies as a selection artifact
   of the floor, not of the arm.
3. **`KS-MOKA-3` finds no mechanism to attach the number to (D6).** The text stream — the one the
   prereg explicitly bet on — is **FLAT** under both floors. A performance delta with no stream-level
   movement, no separable encoder draw, and an exactly-cancelling null-manipulation control is not a
   result about the manipulated variable.

### D7.5 What this does *not* say

The ruling does **not** claim MokA harmed the ZH encoder — `KS-regression` is not triggered and the
arm is flat, not below, against the merged floor. It does **not** invalidate §3.4's pairing choice:
same-path pairing was and remains the methodologically correct **bar** (the merged floor is not
bit-reproducible from an unmerged forward), and the arm fails FORMAL against it regardless. What D7
establishes is narrower and decisive: **the residual sub-bar quantity that the same-path pairing
produces is not evidence for routing**, so no "measured-not-promoted weak positive" may be banked
from this cell, and no stage-2 amendment may cite the +0.0268 as its premise.

---

## D8. PER-CLAUSE TABLE + NON-BINDING OBSERVATIONS

### D8.1 Per-clause ruling table

| clause | frozen bar | measured | ruling |
|---|---|---|---|
| §4.1 G-repro / freeze integrity | P + A–G at REFREEZE-1 shas; run_rac `b85eb72a…`; gitlink `a912747…`; `run_one` `286a9e44…` | all re-hashed by this reviewer, **12/12 MATCH**, working tree clean | **PASS** |
| `KS-MOKA-0` (§3.2) machinery incl. S2 identity control + KS-parity | max\|Δ\|=0.0 identity, bit-exact KS-parity, grads to `A_t`/`A_v`/`B`, `fallback_calls==0` | S1–S9 all PASS; KS-parity `0.000000e+00` both streams; `fallback_calls 0` over 77,224 routed calls | **PASS** |
| `KS-MOKA-0b` (§3.4) | mean per-item cos ≥ 0.9999 on all 6 cells | worst **0.99954879**, **0/6** cells pass | **FIRED** ⇒ unmerged floor MANDATORY and BINDING |
| §3.4 pairing | arm paired vs the unmerged floor "instead of 13150" | job 13573, byte-identical `run_one`, fresh group | **COMPLIED** |
| `KS-MOKA-2` (§3.6) as bounded by **N1** | median ≥ 0.05 ⇒ not degenerate (and nothing more) | **1.4170** vs independent-draws **1.4136** | **NON-DEGENERACY FLOOR ONLY**; "routing is real" **NOT CLAIMABLE** |
| **FORMAL** (§2.3 + §3.1(4)) val-selected | Δacc ≥ +0.030 AND ΔmF1 ≥ +0.030 AND 3/3 | +0.0268 / +0.0267 / 3/3 (binding); +0.0000 / −0.0073 / 2/3 (merged) | **FAIL** |
| **FORMAL** final-epoch | Δacc ≥ +0.030 AND ΔmF1 ≥ +0.030 AND 3/3 | +0.0067 / −0.0008 / 1/3 (binding); +0.0000 / −0.0075 / 1/3, 0/3 (merged) | **FAIL** |
| §3.1(5) ≥2-dataset headline | pass on ≥ 2 datasets | single dataset by design | **STRUCTURALLY UNREACHABLE** |
| `KS-MOKA-1` primary (§3.5) | see verbatim; **parse (A)** pinned per **N4** | binding: fails on val-sel ⇒ no fire under (A), fires under (B); merged: fires under both | **DOES NOT FIRE (parse A, binding)** — disclosed both ways; operationally moot (§3.10/§8/D7) |
| `KS-MOKA-1` secondary within-noise | Δacc < +0.015 on **both** protocols | val-sel +0.0268 ≥ +0.015 (binding) | **DOES NOT FIRE** (binding); **FIRES** vs merged floor |
| `KS-MOKA-3` (§3.7) | 3 pre-declared readings | text FLAT ×2 floors; image AMBIGUOUS ×2 floors | **"neither moved" ⇒ NULL-OP**; text-side reading refuted; 9th law-I **not certified**; visual-protection **BARRED** |
| `KS-regression` (§3.8) | ≤ −0.030 mean Δacc on either protocol | worst +0.0000 | **NOT TRIGGERED** |
| §3.10 scope | one family, one bite; no cross-attn / `r_v≠r_t` / routed-`B` / 2nd mask / EN / HateMM / 2nd SFT seed | none attempted | **CLEAN** |
| F0.1 test-touch | 3 budgeted + 3 contingent | **6 spent / 6 budgeted** | **WITHIN BUDGET** |
| F0.4 standing vetoes | own-train-split only, no OCR, no gold, no ensembles, no MLLM-scores-as-signal, no raw video off-machine, GPU via SLURM | all held | **COMPLIANT** |
| F0.2 / F0.6 restatement (**N2**) | mandatory at verdict | restated in the verdict block and in D7.4 / D8.3 | **DISCHARGED** |
| **N7** hook-site supersession | must state the hook sits on the outer `PeftModelForCausalLM` | stated in D8.3 | **DISCHARGED** |
| F0.3 / D7 novelty bound | first-application only, MokA credited, "we invented it" banned | see D8.2 | **BINDING ON WRITE-UP** |
| §6 budget | **CAP = 4.7 GPU-h** (+0.05 contingent) | **5.573** | **OVERRUN +0.873 (+18.6 %), DISCLOSED** — see D8.4 |

### D8.2 Novelty / credit disposition (F0.3 + the D7 clause)

The cell **failed**, so no novelty claim of any kind is unlocked. Recorded for completeness so the
boundary survives into the paper: had it passed, the claim would have been bounded to *first
application of modality-routed PEFT to hateful-video encoders*, **with MokA (GeWu-Lab, NeurIPS 2025,
`b28e834`) explicitly credited** in an acknowledgement and a citation, and any phrasing implying we
invented modality-routed LoRA is **banned**. The code credit headers in `src/moka/routed_lora.py`
remain correct and must stay. As it stands the reportable content is a **measured door-closer plus a
mechanism decomposition** — the PEFT-adapter-structure axis, never previously varied in this campaign,
is now measured and closed on ZH at 5.573 GPU-h.

### D8.3 Mandatory restatements (N2, N7)

- **F0.2 (verbatim obligation).** ONE MokA SFT run; `--seed` varied the **head** only (head-init +
  data-shuffle), pairing per head-seed. The encoder is a **single draw** and its seed variance is
  **not estimated**; the arm's encoder is a **different SFT draw** from the floor's, so **encoder-seed
  noise is confounded with the routing effect and is NOT separable within this budget.** This is the
  same limitation B3 / F53 / curric / bidir carry, accepted for the same reason. **It is load-bearing
  in D7.**
- **F0.6 (verbatim obligation).** At the deployed `image_max_pixels: 262144` one ZH SFT record
  tokenizes to **2,688 vision-pad tokens + a median 153 text tokens ⇒ median vision share 94.6 %**
  (record #0: 2,688 + 135 = 2,823, 95.2 %; text tokens over 579 rows min 81 / p25 112 / median 153 /
  p75 210 / max 393). **MokA's own shipped regime is the mirror image: `my_text_mask 16128` vs
  `my_image_mask 256` = 98.4 % TEXT.** Consequence: routing makes `A_t` undiluted (the bet) but also
  **data-starved** — from 100 % of positions to ~5.4 %, ~18× fewer token-gradients over the same 3
  epochs — and the transfer from MokA's reported gains is **weak**, because their text-dominant regime
  is not ours. **The measured outcome sits at the low end of the frozen 5–8 % prior, exactly as F0.6
  argued it would.** A corroborating raw observation from GPU-smoke LEG 1: `A_v`'s gradient norm ran
  **~25–40× below `A_t`'s** throughout the 10 steps *despite* vision tokens being 94.6 % of positions —
  i.e. the token-share asymmetry did not translate into gradient-share, which is consistent with the
  stream decomposition finding no text-side sharpening.
- **N7 (hook-site supersession).** The modality-mask forward-pre-hook is registered on the **outer
  `PeftModelForCausalLM` wrapper**, **not** on the base `Qwen2_5_VLForConditionalGeneration` as prereg
  §4.5 item 2 describes. The prereg text is superseded on this point by `MOKA_FREEZE.md §7 REFREEZE-1`
  and `MOKA_REFREEZE_FIX.md §2` (round-1 finding **P1-A**: the hook on `get_base_model()` never fires,
  `hook_calls=0`, `fallback_calls=1`). **Never quote §4.5 item 2 bare.**

### D8.4 Non-binding observations (no pre-registered threshold; none affects the verdict)

1. **`KS-MOKA-0b` text-stream 3× drift asymmetry.** The **text** stream drifts ≈3× further from the
   merged reference than the **image** stream (text means ≈0.99955 vs image ≈0.99985; worst per-item
   minima 0.99750 text on dev_seen vs 0.99894 image), and the 8-item GPU-smoke rehearsal predicted
   both the magnitude and the ordering (img 0.99977 > text 0.99946). Measured on the **banked generic
   adapter with routing entirely absent**, so it is a property of merged-vs-unmerged **bf16
   accumulation order**, not of MokA. **Carries no pre-registered threshold and no bar; recorded
   only.** Two forward-looking notes, both non-binding: (i) it is a plausible partial mechanism for
   why the unmerged floor's *val-selected* readout degraded most — the more drifted stream is the one
   both measured passes ride on; (ii) any future cell that must compare a merged-path floor with an
   unmerged-path arm should expect a same-path floor to be **mandatory**, i.e. §3.4's contingency
   should be treated as the default cost, not a contingency, in the next prereg of this shape.
2. **The `KS-MOKA-2` bar is dead weight and should not be re-used as drafted.** N1 predicted it would
   report ≥ 0.05 with probability ≈ 1; it did (1.4170). N1's $0 constructive alternative —
   `‖A_v^final − A_v^init‖_F / ‖A_v^init‖_F` per layer, computable offline because DEV-3 pins `A_v`'s
   init deterministically (`MOKA_INIT_SEED = 20260726 + 8·layer_index` inside `torch.random.fork_rng`)
   — was correctly **not** implemented post-freeze (it would have fired §4.6). It remains available as
   a **post-hoc, $0, non-pre-registered** analysis on the saved adapter and would be the right
   discriminator if this bar is ever re-drafted. **Nothing in this verdict depends on it, and it must
   not be run and then narrated as if it had been pre-registered.**
3. **Budget overrun, disclosed: 5.573 GPU-h against the §6 `CAP = 4.7`, +0.873 (+18.6 %).**
   Attribution from the executor's ledger, which I re-checked against the `sacct` elapsed times in
   §S5'.1: +0.33 GPU smoke (the executor's own throwaway `eval_strategy: steps` deviation → 3 full
   78-item eval passes; it bought the eval-surface `fallback_calls == 0` evidence), +0.31 SFT (the
   frozen recipe's own per-epoch evals plus build/load — **not** routing compute, measured at
   **+1.72 %**), +0.61 job 2, +0.36 the contingent floor above its 0.05 estimate. **Assessment:**
   every GPU-hour maps to a pre-registered item (plus one 11-second throwaway resubmit); **no
   unauthorised experiment, no extra arm, no extra test evaluation** — the overrun is wall-clock
   estimation error, not scope creep, and it was disclosed by the executor unprompted. It is a
   **discipline note, not a breach that alters the verdict**. Recommendation for the next prereg of
   this shape: cost caps from `sacct`-measured wall for the *exact* recipe including its eval passes,
   and budget §3.4-style contingencies at their measured (not estimated) cost.
4. **F0.6 vision-token regime — forward-looking, non-binding.** The 94.6 %-vision / 98.4 %-text
   inversion is now not merely a prior-lowering argument but the most economical *explanation* of the
   null: MokA's mechanism is defined by giving the **dominant** modality its own down-projection, and
   in our SFT the dominant modality by token count (vision) is the one F45/F58 already price at ≈ 0
   for the vote, while the modality that carries both measured passes (text) is the one the routing
   **starves**. Any future proposal to re-open modality-routed PEFT here must first move the token
   balance (e.g. a text-dominant SFT record format), and would be a **new pre-declared arm costing a
   new bite** (§3.10). Recorded as an observation, **not** as a recommendation to spend GPU.
5. **Positive machinery outcome worth banking.** Independently of the scientific null, this family
   produced a working, identity-controlled, strict-mode modality-routed LoRA on the deployed stack
   with **zero vendored-tree edits**, a bit-exact default==identity guarantee (`KS-parity`
   `0.000000e+00`), and a runtime-verified hook on the production PEFT wrapper. The round-1 §4.6
   fire-and-re-review cycle worked exactly as designed: a `GATE: BLOCK` caught a defect
   (`fallback_calls=1` on the real class) that the CPU smoke was blind to, the fix added **S9** which
   fails on the pre-fix code, and the re-review restored authorization. **The prereg machinery, not
   the hypothesis, is the durable output of this cell.**

---

## What I did NOT do

No GPU, no SLURM, no Modal, no job submitted, **no new test metric produced** (the only test numbers
in this document are re-parses of the 9 already-produced trainlogs, which cost nothing and touch no
new labels), no `state/` mutation, no `research-wiki/` mutation, no edit to `MOKA_PREREG.md`,
`MOKA_FREEZE.md`, `MOKA_PREREG_REVIEW.md`, `MOKA_REFREEZE_REVIEW.md`, `MOKA_SUBMIT_RECORD.md` or to
any of the 7 frozen artifacts, and no push. I did not re-open, re-weight or amend any frozen bar,
threshold, gate order or budget; where the frozen text was ambiguous (`KS-MOKA-1`, note N4) I pinned a
parse and disclosed the alternative's outcome rather than choosing an outcome and back-fitting a parse.

---

## RULING

**MEASURED — NOT PROMOTED. `final-epoch: fail; val-selected: fail`.**

Modality-routed LoRA (MokA `A`-split, `r_v = r_t = 16`, shared `B`, no cross-attention) transplanted
into the deployed ZH LoRA-SFT recipe **fails the pre-registered FORMAL bar on both protocols against
the §3.4-binding same-path unmerged floor and against the banked merged floor 13150**, lands at
**+0.0000 acc** on both protocols against that merged floor, and records a **stream-level null-op**
under `KS-MOKA-3` — with the prereg's own text-side bet (F0.5) **refuted** (text FLAT under both
floors) and MokA's advertised visual-protection premise **unmeasurable and un-narratable** here (image
AMBIGUOUS, head flat). The single above-noise number in the family, the +0.0268 val-selected gain over
the unmerged floor, is adjudicated in **D7** as **not attributable to routing**: it is exactly
cancelled by the −0.0268 the unmerged path loses with routing entirely absent, it is confounded with
an unestimated single encoder draw (F0.2), and it has no mechanism behind it (`KS-MOKA-3`).

**Bank as: the PEFT-adapter-structure axis — shared vs modality-routed LoRA down-projection, the one
axis no banked adapter in this campaign ever varied — is now MEASURED and CLOSED on ZH at 5.573
GPU-h, as a null-op, not as a law-I instance.** HateMM stage 2 is **not authorised** (§3.10 new bite +
§8 amendment, and D7 removes its premise). `KS-MOKA-2` is banked as a **non-degeneracy floor only**
(N1). **F0.2 and F0.6 are restated above as conditions of this verdict (N2); N7's hook-site
supersession is restated (D8.3); F0.3's novelty bound is moot (no pass) but recorded.** Any write-up
of this cell must carry the D7 ruling — reporting the +0.0268 as a routing gain would be a numeric-
provenance violation of the same class the project's `0.8732` discipline exists to prevent.
