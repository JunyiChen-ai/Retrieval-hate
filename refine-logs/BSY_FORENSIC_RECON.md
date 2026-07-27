# BSY — BANK SYNTHESIS (LITSWEEP-6 membank C2) — FORENSIC RECON (zero-GPU)

**Agent:** forensic-recon (2026-07-28 adversarial wave, candidate 3 of 4) · **Date:** 2026-07-28 NZST.
**Discipline honoured.** CPU-only reading + $0 forensic arithmetic on **banked TRAIN-split caches**, ≤4
threads. **ZERO GPU / SLURM / Modal / training.** **TEST-SPLIT CONTACT: NONE.** No prereg written, no job
submitted, no frozen artifact mutated. The frozen operator module `scripts/analysis/mechfix_ops.py` was
**imported unmodified** for the replay in §5.

**Status of this document.** **Recon-level PRE-CLOSURE**, not a measured KILL. C2 has never been run under a
frozen bar. What follows pre-closes it on (a) an arithmetic bound on its own mechanism, (b) three ban/premise
failures, and (c) a cost correction — and explicitly flags one open user ruling.

---

## 0. THE CELL, PRECISELY

**Mechanism statement (C2 as specified, `LITSWEEP6_MEMBANK.md:209-359`).** Stratify the train bank by
(gold class × transcript-length quartile); identify donor-rich and recipient-poor cells; synthesise new bank
rows for the poor cells by FeTrIL rigid translation (arm S1) or DC/FeCAM shrinkage-Gaussian sampling (arm S2)
in a 128-d PCA space; project back and re-L2-normalise. **The bank gains rows; nothing else changes** —
"encoder, head, retrieval, k=20, weights, threshold" all stay (`:283-285`). It is "the **only** candidate
that changes which items are retrievable" (`:212`, restated `:30-31`).

**Target.** CP1 — "the bank's class prior is a function of transcript length: `P(hate | 0-1 words) = 0.1096`
rising monotonically to `0.5538` at 401+ words" (`LITSWEEP6_MEMBANK.md:97-98`); "the (short × hate) cell is
nearly empty, so no local estimate of the class prior in that region can be right" (`:255-256`).

**C2's own frozen bars** (`LITSWEEP6_MEMBANK.md:308-320`), quoted because §2-§5 fire against them:

1. "Primary Δacc ≥ **+0.010**, 5/5 fold signs ≥ 0, ≥3/5 strictly positive, ≥1 dataset."
2. "**Exchange rate ≥ 1.2** on the pathology population."
3. "median top-20 true-label purity on the pathology population, **before vs after**, currently 0.12-0.22."
4. Occupancy control over ρ ∈ {0.1, 0.25, 0.5}.
5. "**Near-duplicate control:** max cosine between each synthetic row and its parent; a distribution piled
   at **>0.99** means the interpolants are copies and the arm is **void**."

---

## 1. THE REACH ORACLE IS LARGE — AND IT DOES **NOT** KILL C2

Stated first so the record is not read as an oracle argument. The family that contains "change what sits in
the top-20" has the largest oracle ever measured on this object:

| dataset | F98 family oracle | F98 realised (best of 45 cells) |
|---|---|---|
| HateMM | **+0.1492** | +0.0134 |
| MHC-ZH | **+0.1520** | −0.0069 |
| MHC-EN | **+0.2186** | +0.0000 |

Source `AGGNET_PREGATE_RECORD.md:368-370` (oracle) and `:682` (realised); "96-100 % of every deployed error
is inside its function class" (`:635`).

**Independently reproduced in this recon** (§5 harness; gold-class × length-quartile in-sample cell oracle
over the train-side LOO vote): **+0.1505 / +0.1520 / +0.2313**. ZH matches F98 **exactly to 4dp**; HateMM is
within 0.0013 and EN within 0.0127, the residual explained by the bank difference (this recon uses the
full-bank LOO, F98 uses the 5-fold fitting-fold bank). **The oracle is real and it is not the reason C2
dies.**

---

## 2. WHAT KILLS IT (i) — THE EXCHANGE RATE OF QUERY-AGNOSTIC INJECTED MASS IS THE LOCAL CLASS ODDS

This is the decisive argument and it is **derivable, not merely measured**.

**Derivation.** Inject synthetic *hate*-labelled mass into a length region `R`. The synthesis is
**query-agnostic** by construction (C2 places rows by *cell*, not by query — the anchor-local variant is
explicitly the *alternative*, `LITSWEEP6_MEMBANK.md:296-299`). So within `R` a synthetic row enters top-20
lists without regard to the query's gold class. Each synthetic hate row that displaces an existing neighbour
shifts that query's vote toward hate by the same amount regardless of the query. Among the boundary-adjacent
queries in `R` — the only ones a bounded shift can move — a fraction `p_R` are gold hate (the shift **fixes**)
and `1 − p_R` are gold non-hate (the shift **breaks**). Hence

```
expected exchange rate  =  fixed / broken  ≈  p_R / (1 − p_R)  =  the LOCAL CLASS ODDS of region R.
```

**Crossed with CP1's own measured priors.** Re-derived in this recon on the HateMM train split under the
frozen band definition (`RESTRANS_PREGATE_RECORD.md:386-387`), and **reproducing that record's band table
exactly**:

| length band (words) | n | hate | P(hate) | **local class odds = exchange-rate ceiling** | vs C2 bar 2 (≥1.2) |
|---|---|---|---|---|---|
| **0-1 (the cell CP1 names)** | 73 | 8 | **0.1096** | **0.1231** | **10× under** |
| 2-50 | 188 | 55 | 0.2926 | 0.4135 | 2.9× under |
| 51-150 | 136 | 52 | 0.3824 | 0.6190 | 1.9× under |
| 151-400 | 217 | 111 | 0.5115 | 1.0472 | under |
| 401+ | 130 | 72 | 0.5538 | **1.2414** | clears — but this cell is **not** under-populated |

*(n and P(hate) re-derived this session and identical to `RESTRANS_PREGATE_RECORD.md:386-387`, which is
therefore a parity gate on this recon's harness, not an independent claim.)*

**The trap is structural.** The only band whose odds clear the 1.2 bar is the one that is **already
hate-rich** (55.4 % hate, n=130) — i.e. the band with nothing to repair. The band C2 exists to repair has
odds **0.1231**, an order of magnitude under its own bar. **Synthesis pays for the pathology at the local
base rate, and the local base rate is the pathology.**

This is the exact-form instance of LITSWEEP-6's own law (iii): "Every mechanism that surfaces the pathology
*symmetrically* pays for it at par or worse … exchange rate 0.53-0.95, never above **1.17** anywhere in 36
cells. … **Reaching the pathology is not the hard part and no candidate should be sold on reaching it.**"
(`LITSWEEP6_MEMBANK.md:39-44`).

**And the cell is already reached.** Re-derived here: HateMM queries in the 0-1-word cell retrieve top-20
neighbours of **median 1.0 word**; the global median top-20 neighbour volume is 189.2 words. Retrieval is
strongly length-organised, exactly as CP1 says — so the short cell is **not** a coverage failure that
synthesis fixes; it is correctly reached and ~89 % non-hate.

---

## 3. WHAT KILLS IT (ii) — CP1 IS A **HateMM-ONLY** FACT

C2's stratification premise does not exist on two of three datasets:

> "So the diagnosis C1 was built from is real — **on one of three datasets.** On MHC-ZH the association runs
> the *other way* (longer transcript → *less* hate) and on MHC-EN there is none at all."
> — `RESTRANS_PREGATE_RECORD.md:387-389`

Corroborated by F96's own two statistics: `AUC(p̂, gold)` = 0.6495-0.6703 (HateMM) / 0.5268-0.5752 (ZH) /
**0.0000, 0.3373, 0.4240, 0.4314, 0.0000** (EN) — "MHC-EN's `p̂` is near-constant *and* its AUC is below
chance in all five folds, **exactly 0 in two**" (`RESTRANS_PREGATE_RECORD.md:335-341`).

**A (class × length-quartile) synthesis plan on a dataset where length carries no class information is
synthesis into arbitrary cells.** Against a **≥2-dataset** goal bar, C2's premise survives on one dataset.

---

## 4. WHAT KILLS IT (iii) — TWO PREMISES THAT HAVE ALREADY FAILED

### 4.1 C2's designated placement-criterion supplier is gone (F96)

LITSWEEP-6 sequenced C2 after C1 so that C1's `p̂` could target the synthesis. F96 revoked that:

> "§6 of LITSWEEP6 recommends writing C2 *after* C1 'because if C1 shows the label field is correctable then
> C2's placement criterion can use `p̂` and becomes much better targeted.' **It cannot.** `p̂` is a usable
> targeting signal on HateMM only; on MHC-ZH it would target the wrong direction and on MHC-EN it would
> target noise. **Any C2 prereg must choose a placement criterion that does not rest on the length
> covariate.**" — `RESTRANS_PREGATE_RECORD.md:393-397`, routed at `:445-446` ("**C2's prereg must be
> rewritten**").

C2 has, at time of writing, **no legal placement criterion**. The remaining candidate named in the sweep is
the F95 pair verifier as an accept/reject filter (`LITSWEEP6_MEMBANK.md:350-353`) — but F97 subsequently
settled that asset as "**ANALYSIS-GRADE ONLY**" with the F47-family features beating it as gating features
on 3/3 datasets (`findings.jsonl` F97). Using it to gate *which bank rows exist* is a new use that needs its
own ruling.

### 4.2 THE COST CORRECTION — "0 GPU-h" is wrong for the full version

The sweep prices C2's full version as "**0 GPU-h** for the bank-side change itself (the head is not
retrained); budget ~0.3 GPU-h **only if** the ceremony requires a same-path floor re-mint"
(`LITSWEEP6_MEMBANK.md:305-306`). The conditional is **not** conditional:

> "F78 parked curation because 'head embeddings never persisted, floor head ckpts **all 6 deleted**' …
> `p2_out/cache_MHC_s{0..3}.json` banks the top-60 neighbour lists … which supports **exact, $0, multi-seed,
> deletion-only** bank replay … **it applies to bank *additions*, key-space changes, and re-training, not to
> pruning.**" — `ERRPAT_MHC-EN_2026-07-26.md:570-576`

The banked-neighbour-list replay that makes $0 bank surgery possible is scoped to **deletions only**, and
BSY is by definition an **addition**. A new row can appear in a query's top-20 only if the neighbour lists
are recomputed in the head space — which needs the head, and F78 records **6/6 deployed floor head ckpts
missing** (`findings.jsonl` F78: "head embeddings never persisted, floor head ckpts all 6 deleted;
multi-seed pregate needs ~0.3 GPU-h head re-mint"). **BSY's true full-version cost is a head re-mint per
dataset per seed**, not zero. Even the *$0 pregate* is $0 only in the raw banked encoder space (the F95/F96/
F98 arena), never in the deployed head space.

---

## 5. NEWLY MEASURED ARITHMETIC (this recon, $0 CPU, TRAIN SPLIT ONLY)

**Protocol.** `scripts/analysis/mechfix_ops.py` imported **unmodified**; `deployed_vote(..., topk=20,
exclude_self=True)` replayed over the **full train bank** (LOO) in the F95/F96/F98 **fused raw space**
(`l2n(concat[l2n(img), l2n(txt)])`, `mechnov_pairverify.py:115-125`) built from the frozen per-dataset
caches of `restrans_pregate.py:83-97`: HateMM `Qwen2.5-VL-7B-Instruct-LoRA-curric_HF`, MHC-ZH
`Qwen2.5-VL-7B-Instruct-LoRA_HF`, MHC-EN `Qwen2.5-VL-7B-Instruct_HF`. Transcript volume per the frozen F89-T3
definition (`restrans_pregate.py:117-128`): whitespace tokens for HateMM/MHC-EN, **characters** for MHC-ZH.
Strata = equal-count quantile bins of volume. "Ceiling" = the **gold-fitted, in-sample** best constant
threshold per stratum, minus the deployed accuracy. Train split only; no test file opened.

**Parity read (free, and it is the honest calibration of everything below):** deployed train-LOO accuracy
**0.8495 / 0.8480 / 0.7687** against F95's fitting-fold-bank figures **0.8441 / 0.8480 / 0.7796**
(`MECHNOV_PAIRVERIFY_PREGATE.md:298-300`) — MHC-ZH exact, HateMM +0.0054, MHC-EN −0.0109. The offsets are
the expected consequence of a full bank vs a 4/5 bank and are **within-cell-cancelling** for the Δ read below.

### The stratum-shift ceiling ladder

| strata (query-observable length bins) | HateMM (n=744) | MHC-ZH (n=579) | MHC-EN (n=549) |
|---|---|---|---|
| **1** (= the global threshold, a measured-dead lever) | +0.0121 | +0.0035 | +0.0128 |
| **2** | +0.0188 | +0.0121 | +0.0128 |
| **4 — C2's declared quartile granularity** | **+0.0202** | **+0.0207** | **+0.0255** |
| 8 (finer than C2 declares) | +0.0282 | +0.0311 | +0.0437 |

*(Newly computed this session; not previously persisted to any repo artifact.)*

**Reading.** At **C2's own declared granularity**, a **gold-fitted, in-sample, zero-generalisation** shift of
the decision per length stratum buys **+0.0202 / +0.0207 / +0.0255** — **0 of 3 datasets over the +0.030
bar**, before any of it has to survive a fold split, and before C2 has to *realise* the shift through
synthetic mass rather than by fiat. Only at 8 strata (twice C2's granularity) do 2 of 3 clear +0.030, by
1.7 and 8 items respectively, at a stratification that would put 68-71 items per bin and overfit outright.

**Sensitivity, stated because it is the one thing that could look like a rescue.** If the strata are taken to
be **gold class × length quartile** — i.e. the query's own gold label is allowed into the partition — the
"ceiling" degenerates into the label oracle: **+0.1505 / +0.1520 / +0.2313** (§1). That is not a stratum
shift; it is the answer key. Any C2 arithmetic quoting a class-conditioned in-sample ceiling in the +0.02-0.03
band is quoting a **different, query-observable partition** and must say so.

---

## 6. CORRECTIONS TO THE TASKING (claims that did NOT reproduce)

Recorded rather than propagated.

1. **"At C2's own declared (class × length-quartile) granularity the gold-fitted in-sample stratum-shift
   ceiling is +0.0309 / +0.0173 / +0.0200."** **DID NOT REPRODUCE.** Under the literal reading (partition by
   gold class × quartile) the ceiling is the label oracle, **+0.1505 / +0.1520 / +0.2313**. Under the
   query-observable reading (length quartiles) it is **+0.0202 / +0.0207 / +0.0255**. Neither is the quoted
   triple, and the quoted triple's dataset ordering (HateMM highest) is inverted relative to both. **The
   +0.0309/+0.0173/+0.0200 triple is NOT carried into this record.**
2. **"the synthetic mass lands at median top-20 neighbour volume 164.5 words against a 7.5-word target
   cell (22× undershoot)."** **NOT VERIFIABLE AND PARTLY REFUTED.** The cell CP1 names is **0-1 words**
   (`LITSWEEP6_MEMBANK.md:97`, `RESTRANS_PREGATE_RECORD.md:386`), not 7.5 words. The measurable adjacent
   quantity — what those queries actually retrieve — is **median 1.0 word**, not 164.5 (§2); 189.2 is the
   *global* median top-20 neighbour volume. The claim as posed requires constructing the FeTrIL synthetics,
   which this recon did not do and which is not persisted anywhere. **Not carried.**
3. **"frac(cos > 0.99 to parent) = 1.000 on MHC-EN in raw 7168-d space, so bar 5 fires outright."**
   **NOT VERIFIED.** The 7168-d fused space is confirmed (3584 img ⊕ 3584 text, MHC-EN train n=549), and the
   mechanism is documented (`LITSWEEP6_MEMBANK.md:290-294`: an unnormalised or near-duplicate synthetic row
   "is either never retrieved or always retrieved"; the space "is cone-collapsed (deployed top-1 cosine
   0.9439-0.9686, F91)"). But the number itself requires building the synthetics. **Recorded as a
   recon-reported prediction, not as a measurement.**
4. **"exchange rate 0.10-0.80 measured across 24 of 24 cells, Δacc negative in all."** **NOT VERIFIED as a
   measurement** — not persisted anywhere. It is, however, **exactly what §2's derivation predicts** (CP1's
   band odds run 0.1231-1.2414 and the four repairable bands run 0.1231-1.0472), so the derivation is
   carried and the measurement is not.

---

## 7. THE OPEN BAN QUESTION — SURFACED, NOT ANSWERED

`autoresearch/goal_mllm_plus3/state/directions_tried.json:458` (`banned_constraints[3]`, 0-indexed):

> "kNN-vote-pool expansion via pseudo-labels (refuted-as-posed by 3 lit scouts; **representation-training
> expansion only**)"

C2 argues the ban's object is the *pseudo-label*: "C2's synthetic points are **within-class translations of
labelled parents along the nuisance axis** … **keeps its own hate label**. No unlabelled item is ever scored"
(`LITSWEEP6_MEMBANK.md:267-272`). But the constraint's own trailing clause — "representation-training
expansion only" — reads as licensing expansion **for representation training** and withholding it for the
**vote pool**, which would ban C2 regardless of label provenance.

**This is a genuine ambiguity in the ledger and it is a user ruling, not an agent call.** Two readings:

- **(A) the banned object is the pseudo-label.** C2 is legal; §2-§5 still pre-close it on arithmetic.
- **(B) the banned object is vote-pool expansion.** C2 is banned outright by name, and so is any future
  bank-addition candidate.

C2 itself already concedes the neighbouring case is ruling-gated: "**M2m-style majority→minority translation
is EXCLUDED** … it needs a user ruling before anyone spends on it" (`LITSWEEP6_MEMBANK.md:273-277`).
**Recommendation: obtain reading (A)/(B) before any BSY prereg is written**, because under (B) no amount of
arithmetic matters.

---

## 8. VERDICT

> **PRE-CLOSED as a goal-bearing bet, AND blocked behind a ban ruling.**

- **Not** killed by the reach oracle — that is large and independently reproduced (§1).
- **Killed as a bet** by the exchange rate: query-agnostic injected mass is worth the **local class odds**,
  which in the cell C2 exists to repair is **0.1231** against C2's own bar 2 of **≥1.2** — a factor of ten
  (§2, derived, with CP1's band table reproduced exactly).
- **Killed as a ≥2-dataset bet** by CP1 being HateMM-only (§3).
- **Blocked** on placement criterion (F96 revoked `p̂`, §4.1) and **mis-priced** on cost (§4.2: additions
  need a head re-mint; F78's 6/6 missing ckpts).
- **Ceilinged** at C2's own granularity: gold-fitted in-sample **+0.0202 / +0.0207 / +0.0255**, 0 of 3 over
  bar (§5).
- **Ruling-gated** on `banned_constraints[3]` (§7).

**No GPU. No prereg until the §7 ruling.** If the ruling comes back (A) and someone still wants it, the
*only* version worth writing is the **anchor-local** variant (`LITSWEEP6_MEMBANK.md:296-299`) — translate
onto the retrieved rank-1.5 analogue's location rather than a cell centroid — because that is the only
construction that is **not** query-agnostic and therefore the only one §2's derivation does not bound. It
would, however, then need to explain why it is not per-item selection (Law III).

### P(pass) estimates

| bar | estimate | reasoning |
|---|---|---|
| P(≥ +0.030 acc on ≥2 datasets, both protocols) | **< 1 %** | §5's gold-fitted in-sample ceiling is under bar on 3/3 at C2's granularity; §3 leaves one dataset with a premise |
| P(C2 bar 1: Δacc ≥ +0.010 on ≥1 dataset, 5/5 fold signs) | **10–15 %** | HateMM only, and only if the synthesis lands; §2 says the fixes are paid for at 0.12 |
| P(C2 bar 2: exchange rate ≥ 1.2 on the pathology population) | **≤ 3 %** | §2 is a derivation, not an estimate; 36/36 F95 cells and 21/21 F96 cells never reached 1.2 |
| P(C2 bar 5 fires — synthetics pile at cos > 0.99 to parent) | **60–80 %** | the space is cone-collapsed (top-1 cosine 0.9439-0.9686 raw, ~0.9999 head) and Blagus-Lusa predicts exactly this at d≫n |
| P(the direction is legal at all) | **ruling-dependent** | §7 |

---

## PROVENANCE

- Specification and bars: `LITSWEEP6_MEMBANK.md` §0 (`:19-64`), §1(c) CP1 (`:95-112`), §2 C2 in full
  (`:209-359`).
- Ban ledger: `autoresearch/goal_mllm_plus3/state/directions_tried.json:454-463` (`banned_constraints`),
  `state/findings.jsonl` F78, F95, F96, F97, F98.
- Records read directly: `RESTRANS_PREGATE_RECORD.md` (F96 — CP1 band table, `p̂` degeneracy, the C2
  placement ruling), `AGGNET_PREGATE_RECORD.md` (F98 — family oracle), `MECHNOV_PAIRVERIFY_PREGATE.md`
  (F95 — arena parity, exchange-rate law), `ERRPAT_MHC-EN_2026-07-26.md:570-576` (the $0-replay scoping),
  `VGA_PREGATE_RECORD.md` (F97 — the verifier's analysis-grade settlement).
- **Code imported unmodified (read-only):** `scripts/analysis/mechfix_ops.py` (`deployed_vote`);
  space construction copied from `scripts/analysis/mechnov_pairverify.py:115-125`; dataset/volume contract
  from `scripts/analysis/restrans_pregate.py:83-97,117-128`.
- **Caches read (train split only):**
  `data/CLIP_Embedding/HateMM/train_Qwen2.5-VL-7B-Instruct-LoRA-curric_HF.pt`,
  `data/CLIP_Embedding/MHC_zh/train_Qwen2.5-VL-7B-Instruct-LoRA_HF.pt`,
  `data/CLIP_Embedding/MHC/train_Qwen2.5-VL-7B-Instruct_HF.pt`,
  `data/gt/{HateMM,MHC,MHC_zh}/train.jsonl`.
- **Reproducibility note, stated rather than hidden:** §1's reproduction, §2's band table and §5's ladder
  were produced by an inline `python3` replay, **not** by a hash-frozen script, and are therefore
  **recon-grade, not gate-grade**. Two free parity gates are reported (the ZH deployed accuracy 0.8480 exact
  vs F95, and the CP1 band table exact vs F96) so the arena is verifiable; the protocol is stated in enough
  detail to re-emit under the standard ceremony if any number is ever cited in the paper.
- **Required statements:** ZERO GPU / SLURM / Modal / training spent by this recon; **no test-split file was
  opened**; no held-out test metric read or produced; no `state/` mutated by this file; no prereg, config,
  or frozen artifact touched.
