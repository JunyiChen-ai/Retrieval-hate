# FUSIONCAT — INDEPENDENT 0-CONTEXT VERDICT REVIEW (job 13514)

**Reviewer:** independent 0-context verdict reviewer. **Date:** 2026-07-25 NZST. **CPU only** — zero GPU/SLURM/Modal
spent by this review; no `state/` mutation; no `research-wiki/` mutation; no push.
**Judged against:** `refine-logs/FUSIONCAT_PREREG.md` **VERBATIM**, frozen sha
`c88332b8972e3270081600d0a8cb892a8d24afefbc73e378a5a3104a433c0830` — **re-hashed this review: MATCH ✓**.
**Authorization chain:** `FUSIONCAT_PREREG_REVIEW.md` (APPROVED-WITH-NOTES) → `FUSIONCAT_FREEZE.md` →
`FUSIONCAT_SUBMIT_RECORD.md` (S0–S5). **Every number below was re-parsed from the raw trainlogs by this reviewer**
(numeric-provenance discipline); the executor's §S4 table was treated as a claim to be checked, not as input.

---

## 0. Binding clauses quoted VERBATIM before application

**KS-arm-dead — the KILL bar (prereg §3.3):**

> - **KS-arm-dead (per-dataset screen kill).** A dataset's **3-seed mean paired Δacc ≤ 0 vs its own align floor on
>   EITHER protocol ⇒ that dataset cell KILLED** (banked as the concat-fusion null for that dataset). **Secondary
>   read:** mean paired Δacc `< +0.015` on **BOTH** protocols (inside the ±0.014 head-seed band, §2.3) ⇒ also KILL.
>   **Per-dataset, datasets INDEPENDENT:** a KS-arm-dead ZH cell does NOT auto-kill the HateMM cell, and vice versa;
>   each is judged only vs its own floor. State each killed cell explicitly at verdict time. (Rationale for the
>   "either-protocol" gate: the GOAL bar is dual-protocol, so a cell ≤0 on even one protocol can never clear FORMAL.)
> - **Family one bite (§3.6):** the two dataset cells share the single "trained concat fusion" multiplicity bite.

**FORMAL promote bar (prereg §3.2):**

> House **+0.030 acc AND +0.030 mF1** conjunct, **3/3 seeds positive**, under **BOTH** protocols vs the dataset's
> banked floor (§2.3). Below the conjunct under a protocol → **NEGATIVE** on that protocol. A dataset cell clears
> FORMAL only if it passes BOTH protocols. **D7-DEAD (F0.3): even a formal PASS is a performance/robustness row
> ("trained concat fusion vs Hadamard on <dataset>"), NEVER a novelty win** — and honestly attributed to the 2× first-
> Linear capacity + concat operator BUNDLED (F0.6), not the operator alone.

**Decision rule (prereg §3.1, verbatim-ported from `exp-encoder-3seed.md:73-85`):**

> For each dataset × protocol: (1) per-seed paired difference δ = (treatment − control) for acc and macro-F1 at
> seeds 0/1/2; (2) 3-seed mean ± std + sign consistency (how many of 3 positive); (3) n=3 too small for a
> bootstrap — report the paired-t **as an effect-size descriptor only**, no significance claim; (4) **pass =
> mean paired Δacc ≥ +0.030 AND mean paired Δmacro-F1 ≥ +0.030 AND sign 3/3 positive**; (5) headline claim
> requires pass on ≥ 2 datasets under a stated protocol; both protocols judged separately; verdict written
> exactly "final-epoch: pass/fail; val-selected: pass/fail".

**One-bite / scope-frozen (prereg §3.6):**

> - **ONE sbatch = ONE pre-registered family = ONE multiplicity bite** across both dataset cells. `concat` is the ONLY
>   arm; the two datasets **share** the single "trained concat fusion" bite.
> - **Scope FROZEN.** **NO** post-hoc arm additions: a **`cross`/gated/cross-attention** arm, a **param-matched
>   control** (a narrower first Linear to isolate the operator from capacity), a **different loss**, a **third
>   dataset/encoder**, or any knob edit is a **new** pre-declared family and re-costs a bite.

**D7-DEAD closure (prereg §3.5):**

> Whatever the numbers, the fusion operator is a **generic architecture/capacity knob** — a formal PASS is a
> robustness/ablation row, a KILL is a door-closer for the fusion axis; **neither yields a novelty contribution**
> (F0.3). The paper role is: "we measured trained concat fusion against the deployed Hadamard fusion — here is the
> number." That is the entire deliverable.

**KS-regression note (prereg §3.4):**

> If concat − floor **mean Δacc ≤ −0.014** on a leg (below the full head-seed spread, §2.3), concat fusion
> **degraded** the head → bank "concat fusion hurts on <dataset>." A note within the KS-arm-dead frame, not a separate
> multiplicity bite.

---

## D1 — PROVENANCE: all 6 primary trainlogs independently re-parsed — §S4 CONFIRMED BIT-FOR-BIT

### D1.1 Parser (this reviewer's own, written from the prereg text, not from the executor's code)

Protocol pinned by prereg §2 / §3.1 / §S4.1: **val-sel = among epochs ≥ warmup 5, max `Val_Retrieval` acc, `roc`
tie-break, read `Test_Retrieval` at that epoch; final = last epoch (29)**. Selection reads **Val only** (DEV-A
no-peek). Line numbers are `grep -n` (`\n`-based) of the `macroF1`-bearing `Test_Retrieval` line, matching the
executor's stated convention (tqdm `\r` makes Python universal-newline numbering differ — verified, see D1.5).

### D1.2 Six concat runs (job 13514) — reviewer's re-parse

| run | val-sel ep | val-sel test acc / mF1 / roc | line | final ep | final test acc / mF1 / roc | line |
|---|---|---|---|---|---|---|
| ZH seed0 | 25 | 0.8389 / 0.8135 / 0.9325 | 241 | 29 | 0.8389 / 0.8135 / 0.8944 | 274 |
| ZH seed1 | 8 | 0.8456 / 0.8133 / 0.9348 | 101 | 29 | 0.8456 / 0.8181 / 0.9124 | 270 |
| ZH seed2 | 26 | 0.8322 / 0.8068 / 0.9158 | 248 | 29 | 0.8389 / 0.8135 / 0.8989 | 273 |
| HateMM seed0 | 15 | 0.8791 / 0.8730 / 0.9401 | 173 | 29 | 0.8698 / 0.8626 / 0.9246 | 300 |
| HateMM seed1 | 5 | 0.8698 / 0.8632 / 0.9162 | 80 | 29 | 0.8791 / 0.8724 / 0.9239 | 297 |
| HateMM seed2 | 18 | 0.8744 / 0.8672 / 0.9242 | 202 | 29 | 0.8791 / 0.8724 / 0.9239 | 302 |

**Every cell — selected epoch, acc, mF1, roc, and the `grep -n` line number — matches submit-record §S4.2/§S4.3
exactly (4dp).** All 6 runs have 30 epochs (0..29); nan/inf count = **0** in each.

**Val-only selection audit re-derived independently (ties and tie-breaks):**

| run | ep≥5 rows tied at max Val acc | winning ep (roc tie-break) | winning Val acc / roc |
|---|---|---|---|
| ZH s0 | {16,17,18,19,20,25} (6 rows @ 0.8590) | **25** (unique roc-max) | 0.8590 / 0.9207 |
| ZH s1 | {8} (unique max) | **8** | 0.8590 / 0.9200 |
| ZH s2 | {26} (unique max) | **26** | 0.8718 / 0.9171 |
| HateMM s0 | {15,20} (2 rows @ 0.8598) | **15** (unique roc-max) | 0.8598 / 0.9135 |
| HateMM s1 | {5,8,24,28,29} (5 rows @ 0.8411) | **5** (unique roc-max) | 0.8411 / 0.9193 |
| HateMM s2 | {18} (unique max) | **18** | 0.8598 / 0.8943 |

Every tie-break resolved **uniquely on roc** — no residual tie needed a further rule. This matches the executor's
§S4.1 audit exactly (six/two/five tied rows respectively).

### D1.3 Third-party cross-check — the sbatch's own embedded parser

`slurm/logs/fuscat_13514.out` `RESULT_ROW` lines **309, 588, 870, 1179, 1485, 1796** were read directly and agree
with D1.2 on **all 6 runs × 2 protocols × {ep, mF1, acc, roc} = 48 values, bit-exactly**. Three independent parsers
(embedded sbatch, executor's scratchpad cross-parse, this reviewer's) therefore concur. Job `.out` shows exactly **6**
`########## RUN:` banners (lines 28, 311, 590, 872, 1181, 1487), each stamped `FUSION=concat`, and
`======== fuscat ALL DONE (13514) ========` at line 1798 ⇒ **exactly 6 test reads consumed, the budgeted number**.

### D1.4 Floors re-derived from the banked raw trainlogs — prereg §2 CONFIRMED

**ZH floor, job 13150** (`enc3s_MHC_zh_Qwen2.5-VL-7B-Instruct-LoRA_HF_seed{0,1,2}_13150.trainlog`):

| seed | val-sel ep | val-sel acc/mF1 | final ep | final acc/mF1 | vs prereg §2.1 |
|---|---|---|---|---|---|
| 0 | 20 | 0.8322 / 0.8023 | 29 | 0.8456 / 0.8181 | ✓ (ep 20) |
| 1 | 26 | 0.8255 / 0.7956 | 29 | 0.8389 / 0.8113 | ✓ (ep 26) |
| 2 | 19 | 0.8389 / 0.8065 | 29 | 0.8523 / 0.8226 | ✓ (ep 19) |
| **mean** | | **0.8322 / 0.8015** | | **0.8456 / 0.8173** | ✓ **bit-match** |

Mean arithmetic: val-sel acc (0.8322+0.8255+0.8389)/3 = 2.4966/3 = **0.8322**; mF1 (0.8023+0.7956+0.8065)/3 =
2.4044/3 = 0.80147 → **0.8015**. Final acc (0.8456+0.8389+0.8523)/3 = 2.5368/3 = **0.8456**; mF1
(0.8181+0.8113+0.8226)/3 = 2.4520/3 = 0.81733 → **0.8173**.

**HateMM floor, job 13241** (`enc3s_HateMM_…-LoRA-curric_HF_seed{0,1,2}_13241.trainlog`):

| seed | val-sel ep | val-sel acc/mF1 | final ep | final acc/mF1 | vs prereg §2.2 |
|---|---|---|---|---|---|
| 0 | 29 | 0.8791 / 0.8730 | 29 | 0.8791 / 0.8730 | ✓ (ep 29) |
| 1 | 14 | 0.8744 / 0.8678 | 29 | 0.8791 / 0.8724 | ✓ (ep 14) |
| 2 | 10 | 0.8791 / 0.8724 | 29 | 0.8791 / 0.8724 | ✓ (ep 10) |
| **mean** | | **0.8775 / 0.8711** | | **0.8791 / 0.8726** | ✓ **bit-match** |

Mean arithmetic: val-sel acc (0.8791+0.8744+0.8791)/3 = 2.6326/3 = 0.87753 → **0.8775**; mF1
(0.8730+0.8678+0.8724)/3 = 2.6132/3 = 0.87107 → **0.8711**. Final acc 3×0.8791/3 = **0.8791**; mF1
(0.8730+0.8724+0.8724)/3 = 2.6178/3 = **0.8726**.

**Both floors reproduce the prereg §2 tables to 4dp on every seed, epoch, and mean.** The floor trainlogs are the
primary evidence (the ZH `logging/` group dir absence noted in submit-record O-2 is immaterial — the floors were
re-derived here from the raw trainlogs, which are present and parse cleanly).

### D1.5 Branch-assert (prereg §4.4.2 / §S4.0) — re-verified per run by this reviewer

| trainlog (`…_13514.trainlog`) | `grep -c "fusion_mode='concat'"` | `grep -c "fusion_mode='align'"` | nan/inf |
|---|---|---|---|
| `fuscat_MHC_zh_…-LoRA_HF_seed0` | **1** | **0** | 0 |
| `fuscat_MHC_zh_…-LoRA_HF_seed1` | **1** | **0** | 0 |
| `fuscat_MHC_zh_…-LoRA_HF_seed2` | **1** | **0** | 0 |
| `fuscat_HateMM_…-LoRA-curric_HF_seed0` | **1** | **0** | 0 |
| `fuscat_HateMM_…-LoRA-curric_HF_seed1` | **1** | **0** | 0 |
| `fuscat_HateMM_…-LoRA-curric_HF_seed2` | **1** | **0** | 0 |

**Branch-assert: PASS 6/6** (concat matched, align empty, in every run), from the unmodified `run_rac.py:1065`
`print(args)` echo at line 1 of each trainlog.

**Single-manipulated-variable check (F0.2), re-read from the same echo.** All 6 concat runs:
`fusion_mode='concat', epochs=30, warmup=5, topk=20, proj_dim=1024, map_dim=1024, metric='cos', loss='triplet',
hybrid_loss=True, ce_weight=0.5, majority_voting='arithmetic', force=False, group_name='RAC_video_fuscat'`, inert
keys off (`sam=False, mod_dropout=False, head_loss='triplet', mixup=False`), `seed` = 0/1/2 as labelled.
Floor echoes (13150 s0, 13241 s0) carry the **identical** knob set with `fusion_mode='align'`. ⇒ the only manipulated
variable between each treatment run and its paired floor is `--fusion_mode`, as pre-declared.

**Post-hoc integrity (re-run this review):** all 4 reused-machinery shas + the sbatch sha still match the freeze
block byte-exact (`e7b61df4…`, `b85eb72a…`, `2ae7a73f…`, `d43e3bc4…`, `62bfb773…`); `git status --porcelain src/`
**empty**. Prereg §4.6 (code-fix ⇒ re-freeze) was never triggered; the §4.5 codex-gate exemption stands.

**D1 RESULT: PASS.** §S4 is confirmed bit-for-bit; floors confirmed vs prereg §2; branch-assert confirmed 6/6;
zero-code premise held across the whole chain. No provenance defect found.

---

## D5(a) — DELTA TABLES WITH ARITHMETIC SHOWN

### ZH (`MHC_zh`) — concat 13514 − align floor 13150

| seed | protocol | concat acc/mF1 | floor acc/mF1 | Δacc (arithmetic) | ΔmF1 (arithmetic) |
|---|---|---|---|---|---|
| 0 | val-sel | 0.8389 / 0.8135 | 0.8322 / 0.8023 | 0.8389−0.8322 = **+0.0067** | 0.8135−0.8023 = **+0.0112** |
| 1 | val-sel | 0.8456 / 0.8133 | 0.8255 / 0.7956 | 0.8456−0.8255 = **+0.0201** | 0.8133−0.7956 = **+0.0177** |
| 2 | val-sel | 0.8322 / 0.8068 | 0.8389 / 0.8065 | 0.8322−0.8389 = **−0.0067** | 0.8068−0.8065 = **+0.0003** |
| **mean** | **val-sel** | **0.8389 / 0.8112** | **0.8322 / 0.8015** | (0.0067+0.0201−0.0067)/3 = 0.0201/3 = **+0.0067** | (0.0112+0.0177+0.0003)/3 = 0.0292/3 = **+0.0097** |
| | | | **sign** | **acc 2/3 positive** (s2 negative) | **mF1 3/3 positive** |
| | | | **std (n−1)** | **0.0134** | **0.0088** |
| 0 | final-ep | 0.8389 / 0.8135 | 0.8456 / 0.8181 | **−0.0067** | **−0.0046** |
| 1 | final-ep | 0.8456 / 0.8181 | 0.8389 / 0.8113 | **+0.0067** | **+0.0068** |
| 2 | final-ep | 0.8389 / 0.8135 | 0.8523 / 0.8226 | **−0.0134** | **−0.0091** |
| **mean** | **final-ep** | **0.8411 / 0.8150** | **0.8456 / 0.8173** | (−0.0067+0.0067−0.0134)/3 = −0.0134/3 = **−0.0045** | (−0.0046+0.0068−0.0091)/3 = −0.0069/3 = **−0.0023** |
| | | | **sign** | **acc 1/3 positive** | **mF1 1/3 positive** |
| | | | **std (n−1)** | **0.0102** | **0.0082** |

Concat means: val-sel acc (0.8389+0.8456+0.8322)/3 = 2.5167/3 = **0.8389**; mF1 (0.8135+0.8133+0.8068)/3 = 2.4336/3 =
**0.8112**. Final acc (0.8389+0.8456+0.8389)/3 = 2.5234/3 = 0.84113 → **0.8411**; mF1 (0.8135+0.8181+0.8135)/3 =
2.4451/3 = 0.81503 → **0.8150**.

### HateMM — concat 13514 − align floor 13241

| seed | protocol | concat acc/mF1 | floor acc/mF1 | Δacc (arithmetic) | ΔmF1 (arithmetic) |
|---|---|---|---|---|---|
| 0 | val-sel | 0.8791 / 0.8730 | 0.8791 / 0.8730 | **+0.0000** | **+0.0000** |
| 1 | val-sel | 0.8698 / 0.8632 | 0.8744 / 0.8678 | 0.8698−0.8744 = **−0.0046** | 0.8632−0.8678 = **−0.0046** |
| 2 | val-sel | 0.8744 / 0.8672 | 0.8791 / 0.8724 | 0.8744−0.8791 = **−0.0047** | 0.8672−0.8724 = **−0.0052** |
| **mean** | **val-sel** | **0.8744 / 0.8678** | **0.8775 / 0.8711** | (0.0000−0.0046−0.0047)/3 = −0.0093/3 = **−0.0031** | (0.0000−0.0046−0.0052)/3 = −0.0098/3 = **−0.0033** |
| | | | **sign** | **acc 0/3 positive** | **mF1 0/3 positive** |
| | | | **std (n−1)** | **0.0027** | **0.0028** |
| 0 | final-ep | 0.8698 / 0.8626 | 0.8791 / 0.8730 | **−0.0093** | **−0.0104** |
| 1 | final-ep | 0.8791 / 0.8724 | 0.8791 / 0.8724 | **+0.0000** | **+0.0000** |
| 2 | final-ep | 0.8791 / 0.8724 | 0.8791 / 0.8724 | **+0.0000** | **+0.0000** |
| **mean** | **final-ep** | **0.8760 / 0.8691** | **0.8791 / 0.8726** | (−0.0093+0+0)/3 = **−0.0031** | (−0.0104+0+0)/3 = **−0.0035** |
| | | | **sign** | **acc 0/3 positive** | **mF1 0/3 positive** |
| | | | **std (n−1)** | **0.0054** | **0.0060** |

Concat means: val-sel acc (0.8791+0.8698+0.8744)/3 = 2.6233/3 = 0.87443 → **0.8744**; mF1 (0.8730+0.8632+0.8672)/3 =
2.6034/3 = **0.8678**. Final acc (0.8698+0.8791+0.8791)/3 = 2.6280/3 = **0.8760**; mF1 (0.8626+0.8724+0.8724)/3 =
2.6074/3 = 0.86913 → **0.8691**.

Per §3.1 rule (3) — n=3 is too small for a bootstrap; the std columns above are **effect-size descriptors only**.
**No significance claim is made anywhere in this verdict**, and no paired-t p-value is asserted.

---

## D2 — KS-arm-dead applied per dataset, exactly as frozen (§3.3)

The bar is stated on **Δacc** only (mF1 does not enter the kill screen). Datasets judged **INDEPENDENTLY**.

| dataset | mean Δacc val-sel | mean Δacc final-ep | primary: ≤ 0 on EITHER protocol? | secondary: < +0.015 on BOTH? | **KS-arm-dead** |
|---|---|---|---|---|---|
| **ZH** | **+0.0067** | **−0.0045** | **YES** — final-ep −0.0045 ≤ 0 | **YES** — +0.0067 < +0.015 AND −0.0045 < +0.015 | **KILLED** |
| **HateMM** | **−0.0031** | **−0.0031** | **YES** — both protocols ≤ 0 | **YES** — −0.0031 < +0.015 on both | **KILLED** |

- **ZH cell — KILLED.** Triggered on the **primary** clause: the frozen bar reads "mean paired Δacc ≤ 0 … on **EITHER**
  protocol ⇒ that dataset cell KILLED", and the final-epoch leg is **−0.0045 ≤ 0**. The val-selected leg being
  positive (+0.0067) does **not** rescue the cell — the clause is disjunctive by design ("the GOAL bar is
  dual-protocol, so a cell ≤0 on even one protocol can never clear FORMAL"). The **secondary** clause fires
  independently as well (both protocols < +0.015, inside the ±0.014 head-seed band).
- **HateMM cell — KILLED.** Triggered on the **primary** clause on **both** protocols (−0.0031 and −0.0031, each ≤ 0);
  the secondary clause also fires.
- **Datasets independent:** each was judged only against its own banked floor (ZH vs 13150, HateMM vs 13241). Neither
  kill was inherited from the other; both were reached on their own numbers.

**KS-regression note (§3.4) — NOT triggered on either dataset.** The clause requires **mean** Δacc ≤ −0.014 on a leg.
Observed worst means: ZH final-ep **−0.0045**, HateMM **−0.0031** (both protocols) — all above −0.014. (For the record,
the most negative *per-seed* value anywhere is ZH seed2 final-ep **−0.0134**, which is also above the −0.014 line and
is in any case not what §3.4 measures.) ⇒ **no "concat fusion hurts on <dataset>" note is banked.** Both cells are
nulls, not regressions.

---

## D3 — FORMAL promote bar applied per dataset, exactly as frozen (§3.2 / §2.3)

Bar: **mean Δacc ≥ +0.030 AND mean ΔmF1 ≥ +0.030 AND sign 3/3 positive**, under **BOTH** protocols.

| dataset | protocol | mean Δacc (bar +0.030) | mean ΔmF1 (bar +0.030) | sign 3/3? | conjunct |
|---|---|---|---|---|---|
| ZH | val-sel | +0.0067 ✗ | +0.0097 ✗ | acc 2/3 ✗ | **FAIL** |
| ZH | final-ep | −0.0045 ✗ | −0.0023 ✗ | acc 1/3 ✗ | **FAIL** |
| HateMM | val-sel | −0.0031 ✗ | −0.0033 ✗ | 0/3 ✗ | **FAIL** |
| HateMM | final-ep | −0.0031 ✗ | −0.0035 ✗ | 0/3 ✗ | **FAIL** |

Cross-checked against the prereg's **absolute** §2.3 thresholds (same conclusion, independent route):

| dataset | protocol | required mean acc / mF1 (§2.3) | observed mean acc / mF1 | shortfall acc / mF1 |
|---|---|---|---|---|
| ZH | val-sel | ≥ 0.8622 / ≥ 0.8315 | 0.8389 / 0.8112 | −0.0233 / −0.0203 |
| ZH | final-ep | ≥ 0.8756 / ≥ 0.8473 | 0.8411 / 0.8150 | −0.0345 / −0.0323 |
| HateMM | val-sel | ≥ 0.9075 / ≥ 0.9011 | 0.8744 / 0.8678 | −0.0331 / −0.0333 |
| HateMM | final-ep | ≥ 0.9091 / ≥ 0.9026 | 0.8760 / 0.8691 | −0.0331 / −0.0335 |

**No cell clears FORMAL on any protocol.** Both cells are **NEGATIVE** under both protocols. (Consistent with prereg
DEV-G / F0.5(d): the HateMM +0.030 bar at 0.909 was pre-declared arithmetically implausible from a 0.879 floor.)

---

## D4 — FAMILY ONE-BITE OUTCOME + D7-DEAD RESTATEMENT

**One bite, spent, closed.** Per §3.6, **one sbatch = one pre-registered family = one multiplicity bite** shared by
both dataset cells. The record shows exactly **one** `sbatch` (job **13514**, COMPLETED 0:0, 07:18) and exactly
**6** budgeted test reads (2 datasets × 3 seeds), confirmed by the 6 `RUN:` banners and 6 `RESULT_ROW` lines. The
throwaway GPU smoke (job 13496) ran to a deleted `RAC_video_smoke_fuscat` group at 3 epochs and consumed no budgeted
evaluation. **No second submission, no re-run, no post-hoc arm.** The bite is now **spent**.

**Family outcome: BOTH cells KS-arm-dead ⇒ the fusion-operator axis is CLOSED** — the prereg's own §8 bullet 1
outcome ("trained concat fusion carries no net vote signal beyond Hadamard on either dataset ⇒ the fusion-operator
axis is **CLOSED** at ~0.1 GPU-h, and the live reviewer question ('why Hadamard, not concat?') is answered with a
**measured null**"). This is the pre-declared modal outcome under the honest LOW prior (P(goal) 3–6 %, F0.5).

**Scope remains FROZEN (§3.6).** A `cross`/gated/cross-attention arm, a **param-matched control**, a different loss,
or a third dataset/encoder are each a **new** pre-registered family costing a **new** bite — none is authorized by
this verdict, and none may be run as a "follow-up" to it. Since no cell survived, the §8 bullet-3 obligation (a
surviving cell owes a param-matched control before any operator-level claim) is moot: **there is no positive claim to
qualify.**

**D7-DEAD restatement (§3.5 / F0.3), unconditional.** The fusion operator is a **generic architecture/capacity knob**.
This outcome is a **door-closer for the fusion axis** and yields **NO novelty contribution** — exactly as it would
have yielded none had it passed. The entire deliverable is the sentence: *we measured trained concat fusion against
the deployed Hadamard fusion — here is the number (null on both datasets, both protocols).* The **F0.6 bundling
caveat** is carried into the null as well: the arm that failed was `concat` **with 2.0× first-Linear params**
(2,098,176 vs 1,049,600), so the honest reading is "the concat-fusion arm, capacity bump included, does not beat
Hadamard here" — the measured null is **not** attributed to the operator in isolation, and it does **not** license a
claim that "extra head capacity cannot help" in general.

---

## D5(b) — PER-DATASET RULINGS + VERDICT LINE (binding language)

**ZH (`MHC_zh`, `Qwen2.5-VL-7B-Instruct-LoRA_HF`) vs floor 13150.** Mean paired Δacc **+0.0067** val-selected /
**−0.0045** final-epoch; ΔmF1 **+0.0097** / **−0.0023**; sign acc 2/3 / 1/3. Fails the FORMAL conjunct on both
protocols; hits the KS-arm-dead primary clause on the final-epoch leg (≤ 0) and the secondary clause on both legs
(< +0.015). **Ruling: FORMAL NEGATIVE both protocols; KS-arm-dead KILLED.** No KS-regression note (mean > −0.014).

**HateMM (`Qwen2.5-VL-7B-Instruct-LoRA-curric_HF`) vs floor 13241.** Mean paired Δacc **−0.0031** on **both**
protocols; ΔmF1 **−0.0033** / **−0.0035**; sign 0/3 positive on every leg. Fails the FORMAL conjunct on both
protocols; hits the KS-arm-dead primary clause on **both** protocols and the secondary clause as well.
**Ruling: FORMAL NEGATIVE both protocols; KS-arm-dead KILLED.** No KS-regression note (mean > −0.014).

### VERDICT — in the frozen §7.2 write-up format, verbatim

```
Smoke: fusion_mode='concat' branch-assert PASS.
ZH concat:     final-epoch: fail; val-selected: fail  [FORMAL §3.2]. KS-arm-dead: KILLED.
HateMM concat: final-epoch: fail; val-selected: fail  [FORMAL §3.2]. KS-arm-dead: KILLED.
(no KS-regression note: no mean Δacc ≤ −0.014 on any leg; no PASS ⇒ the F0.6 capacity+operator-bundled caveat
 applies to the null instead of to a claim; D7-DEAD regardless of outcome.)
```

**FAMILY VERDICT: BOTH CELLS KS-ARM-DEAD ⇒ FUSIONCAT FAMILY NEGATIVE; the fusion-operator axis is CLOSED as a
measured null; one bite spent; D7-DEAD — no novelty claim, no promoted number, no follow-on arm authorized.**

---

## D6 — NON-BINDING OBSERVATIONS (clearly labelled; NOT part of any ruling)

These are **descriptive only**. None of them changes, softens, or qualifies the rulings in D2/D3/D5(b); the frozen
bars were applied first and stand on their own.

- **NB-1 (the ZH val-sel mF1 3/3-positive pattern).** ZH's val-selected leg is the single sub-cell where every seed
  moved the same way: ΔmF1 **+0.0112 / +0.0177 / +0.0003**, 3/3 positive, mean **+0.0097**. **What it does not mean:**
  (i) mF1 is **not** the KS-arm-dead metric — the frozen kill screen is Δacc-only, and the paired Δacc on the same
  sub-cell is 2/3 positive with a mean of +0.0067; (ii) the same-seed **final-epoch** leg is negative on both metrics
  (−0.0045 / −0.0023, 1/3 positive), so the pattern does not survive the protocol swap and cannot be protocol-shopped
  into a result; (iii) the whole move sits **inside the ±0.014 head-seed band** declared in §2.3 — mean +0.0097 with
  a per-seed std of 0.0088 and a seed range of 0.0003→0.0177 is not distinguishable from seed noise at n=3, and §3.1
  rule (3) forbids any significance claim here; (iv) the third seed's +0.0003 is a rounding-scale move. Scale check:
  the ZH test split holds **149** items (`…/MHC_zh/test_seen_…-LoRA_HF.pt`, `labels` length 149, read this review) ⇒
  one flipped prediction = **1/149 = 0.00671** acc, so the whole ZH val-selected Δacc spread is **+1 / +3 / −1 items**
  and its mean is **+1 item per seed**. The honest descriptor is "one protocol × one metric drifted slightly positive
  on ZH, within band, at a magnitude of about one test item" — a **noise-consistent pattern**, not a weak positive to
  bank.
- **NB-2 (the arm was a 2× bundle).** Because concat doubles the first-Linear parameters (F0.6: 2,098,176 vs
  1,049,600), this null is evidence about the **bundle**, not about the operator alone. It is equally consistent with
  "the concat operator is neutral", "extra first-layer width is neutral here", or "a positive from one is cancelled
  by a negative from the other". Disentangling would need the param-matched control that §3.6 explicitly places
  **outside** this family. Any future write-up must not upgrade this null into "capacity does not help".
- **NB-3 (HateMM saturation texture; test-grid quantum verified).** The HateMM test split holds **215** items
  (`data/CLIP_Embedding/HateMM/test_seen_…-LoRA-curric_HF.pt`, `labels` tensor length 215, read this review) ⇒ one
  flipped prediction = **1/215 = 0.00465** acc, which is exactly the observed spacing of the HateMM acc values
  (0.8791 → 0.8744 → 0.8698). Four of the six HateMM Δacc values are exactly **0.0000**; the only non-zero moves are
  −0.0046 (1 item), −0.0047 (1 item) and −0.0093 (**2 items**). **The entire HateMM effect is ≤ 2 flipped predictions
  on any seed.** This is the pre-declared near-ceiling regime (DEV-G / F0.5(d)) and is why the prereg called HateMM a
  "hold-the-pass leg". Descriptively, concat **held the pass** — but "held the pass" is **not** a pass, and the frozen
  bar reads ≤ 0 ⇒ KILLED.
- **NB-4 (val-sel selection instability, ZH and HateMM).** Selected epochs ranged 5→26 and several selections were
  decided by a **roc tie-break over up to six Val-tied epochs** (ZH s0: 6 tied rows; HateMM s1: 5 tied rows). This is
  the F45/F63 78-dev selection wall showing in the raw data. Non-binding, but it is the mechanistic reason the
  val-selected leg is the noisier of the two protocols, and it is a further reason not to read NB-1 as signal.
- **NB-5 (execution hygiene, no defect found).** The chain is clean on every point this reviewer could check
  independently: freeze shas hold at freeze → submit-instant → post-run → **now**; `src/` git-clean throughout; the
  branch-assert passes 6/6 from an unmodified args echo; the smoke throwaways are absent from `logging/` and
  `slurm/logs/`; the floors' trainlogs are intact and re-parse to the prereg's numbers; the test-touch is exactly the
  6 budgeted reads. The executor's §S4 contains no pass/fail language, as §3.7 required. The four disclosed
  deviations (D-1 scratchpad smoke sbatch, D-2 smoke breadth, D-3 `grep -n` line convention, D-4 watcher restart) are
  **immaterial to the verdict** — none touches the frozen artifacts, the protocol, or the numbers, and D-3 was
  verified by reproducing the executor's line numbers under the stated convention.

---

## Reviewer statements

- **Independent 0-context review** against the frozen prereg VERBATIM. Every metric re-parsed by this reviewer from
  the raw trainlogs; §S4 was checked, not trusted. Prereg re-hashed to
  `c88332b8972e3270081600d0a8cb892a8d24afefbc73e378a5a3104a433c0830` (freeze match) before any gate was applied.
- **CPU only.** No GPU, no SLURM, no Modal, no job submitted, no re-run, no additional test read. No `state/`
  mutation, no `research-wiki/` mutation, no source edit, **no push** — this file is committed locally on `main`.
- **No number in this document was transcribed without re-reading its source log** (numeric-provenance discipline);
  every mean is shown with its arithmetic.
