# C05–C14 Zero-GPU Forensic Reconnaissance

**Date:** 2026-07-31 (Pacific/Auckland)
**Scope:** `C05`–`C08` deep, `C09`–`C14` brief
**Status:** **ADVISORY RECON ONLY.** This is **not** a Gate-0 decision, not a
pre-registration, not a kill record and not a scientific verdict. It creates no
`CONTINUE`/`KILL` label for any candidate and moves no registry status. It is the
evidence package the `fast_fail` clause asks for before the post-C04 Gate-0 reopen
(`registry_update_2026_07_28.serial_execution.fast_fail`: *"After two consecutive
active-candidate failures, reopen Gate 0 before continuing the ordered backlog"* —
C01 and C02 are those two failures).

**Cost:** `$0`. **ZERO** GPU, SLURM, Modal, teacher call, model load, training, cache
write, and **ZERO test-split contact**. Nothing in the active C04 lineage was read,
touched or referenced beyond its registry status string.

---

## 0. What was actually opened, and what was inferred

Evidence discipline for this record: every number below is tagged.

**`[R]` read from an existing frozen artifact or record** (no recomputation):

- `TARGET_STATE.json` → `registry_update_2026_07_28`,
  `iteration_8_stage0_bounded_extraction_amendment`
- `autoresearch/goal_mllm_plus3/state/{findings.jsonl,directions_tried.json}`
  (F88–F114; `banned_constraints[0..10]`; 76 dead entries)
- `artifacts/c01_policy_contrastive/v4/a0/C01-A0-v4/C01_A0_{OUT,DECISION}.json`
- `artifacts/c02_edq/v1/a0/C02-A0-v9/C02_A0_{OUT,DECISION}.json`
- `refine-logs/{C02_DESIGN_REVIEW,C03_ASSET_AUDIT,C02_A0_V9_RECORD}.md`,
  `ERRPAT_{HateMM,MHC-EN,MHC-ZH}_2026-07-26.md`,
  `LITSWEEP{2_INPUT_FIDELITY,3_ZH_SPECIFIC,5_HATEMM_EN}.md`, `TARGET_{FINDINGS,LOOP}.md`
- `sacct -j 13843,13846,13847` (read-only accounting query)

**`[M]` measured in this recon**, on **label files only** — `data/gt/<DS>/{train,val}.jsonl`
for `HateMM`, `MHC_zh`, `MHC`. No `.pt` cache was opened, no `test.jsonl` was opened,
nothing was written. This follows the precedent set by `C02_A0_V9_RECORD.md` §5, which
counts `json`-parse statistics over the same train/val files as preparation-time work.

**`[I]` inference** — reasoning over `[R]`/`[M]`, flagged in place.

**Directory census `[R]`** (filenames only, `ls`; no cache opened):
`data/CLIP_Embedding/` holds 100 entries (HateMM), 130 (`MHC_zh`), 71 (`MHC`),
6 (`ImpliHateVid`), 2 (`HateClipSeg`). This is consistent with the amendment's
"99 top-level `train_*.pt` caches" census and adds the 12 new `*-c02den-*` view caches
that C02's extraction job `13843` produced on 2026-07-31.

---

## 1. Calibration — the three instruments that bound everything below

Any candidate acting on the retrieval key or the retrieved neighbourhood now runs into
three *independent* measured ceilings. They are stated once here and referenced by name.

### 1.1 The label-tuple bottleneck (F106 + F107) `[R]`

The deployed decision is a rank-weighted **label** majority over the top-20. Discarding
every cosine reproduces it on `99.60 / 100.00 / 100.00 %` of items in the raw arena
(F106 §2 Result A) and on `100.00 %` — `0/78 × 3 seeds`, K-HC-3 identity `1.0000` — in
the deployed head space (F107). The metric therefore enters the decision **only** through
which twenty items are retrieved and in what order.

Coverage (`≥1` correct-label neighbour already inside the deployed top-20) is
`0.9933 / 1.0000 / 0.9982` raw (HateMM / MHC-ZH / MHC-EN) and `0.9829` in head space
(worst seed `0.9744`). Because the free-weight oracle at pool `M` is exactly
`coverage(M) − acc` and `coverage ≤ 1`, the **marginal** oracle of expanding or
reordering the pool beyond rank 20 is `≤ 1 − coverage(20)` = **`+0.0171`** head-space,
`+0.0067 / +0.0000 / +0.0018` raw at `20 → 400`. The re-ranking / candidate-set family is
dead on its own gold-cheating oracle in both arenas.

### 1.2 The pure-permutation oracle (F99) `[R]`

For the **set-preserving** channel (re-metrication that changes the order but not the
membership of the top-20), with `w = [20…1]`, `Σw = 210` and decision `[s ≥ 0]`, a hate
query is flippable iff purity `≥ 6/20` and a non-hate query iff purity `≥ 7/20`. Crossed
with the measured error purities — HateMM `purity < 0.25` for `21/27` errors; MHC-ZH's
22-item stable core is `8 ≤ 0.10`, `7 ∈ (0.10, 0.25]`, `7 ∈ (0.25, 0.45]`, **zero above
0.45** — the zero-break upper bounds are **HateMM `≤ +0.0279`** (`6/215`) and
**MHC-ZH `≤ +0.0470`** (`7/149`). Both assume zero breaks, which has never happened:
the measured exchange rate is `0.53–0.95` and never above `1.17` in 36 cells.

HateMM's cap is **below the `+0.030` final bar outright**, against a `≥2`-dataset
requirement. `[I]` The `+0.050` Stage-0 bar is therefore unreachable on HateMM through
this channel by arithmetic, independent of mechanism quality.

### 1.3 The purity → accuracy conversion (F107 Q3) `[R]`

Over 30 epochs × 3 seeds of MHC-ZH dev, held-out top-20 purity rises `+0.1043`
(`0.7207 → 0.8250`, correlation with epoch `+0.9237` — the metric channel genuinely
transfers) while dev accuracy rises only `+0.0043` and coverage *falls* `−0.0171`.
Observed conversion `d(acc)/d(purity) = +0.0410`; pooled regression slope `+0.0661`,
95 % bootstrap CI `[−0.0221, +0.1637]`, `R² 0.027`. Driving held-out purity to a
**perfect 1.000** implies `+0.0072` (observed ratio) / `+0.0116` (point slope) /
**`+0.0286`** (upper 95 % CI) — under the `+0.030` bar even at the conservative end.

**Honest limitation, stated by F107 itself and reaffirmed here:** weak association,
MHC-ZH dev only (`n = 78`), observational within-trajectory, linearity assumed. F114
retracted F107's neural-collapse leg (the `0.998` train-LOO premise was a CLIP number;
deployed heads are `0.9406 / 0.8915 / 0.8154`) and re-scoped the metric-channel ruling
from analytic to **empirical** — the verdict stands, the argument is weaker than it
reads. This bound should be used as a **strong prior, not a proof**.

### 1.4 The currency, and the arena `[R]`

`banned_constraints[10]`: the binding screen is **NET ITEMS** against
`22.3 / 17.4 / 16.5` (HateMM / MHC-ZH / MHC-EN) for `+0.030`; exchange rate `≥ 1.2` is
explicitly **not** a screening criterion (F105 reached ER 6.0 and failed; F112 reached
ER 2.8889 with net `−20`). `[I]` Scaling to Stage-0's `+0.050`: `37.2 / 29.0 / 27.5`
net items.

F113 stands: the raw arena may **kill** but may not **promote**; a Stage-0 PASS must be
rendered on the fold-head / deployed-head path. The instrument exists and is banked —
`scripts/analysis/headspace_{mint,arena,fidelity,report}.py` with
`headspace_arena_{hatemm,zh}_s{0,1,2}_OUT.json` — at **~35 s CPU per fold-head, zero GPU**.
C02's A0 used exactly this path (`acc 0.8875 / 0.8912`, matching F113's fold-head
`0.8867 / 0.8923`).

### 1.5 Today's C02 verdict, and the test it hands the backlog `[R]`

C02-A0-v9, job `13847`, exit `0:0`, `KILL_C02_DENSITY_ORBIT_UNREACHABLE`:

| dataset | `delta_acc` | `delta_mf1` | bar | acc 95 % CI | precision on changed |
|---|---|---|---|---|---|
| HateMM | **`+0.000896`** | `+0.000579` | `+0.050 / +0.050` | `[−0.008961, +0.010753]` | `0.5040` |
| MHC-ZH | **`−0.001151`** | `−0.002746` | `+0.050 / +0.050` | `[−0.013241, +0.011514]` | `0.4881` |

`0 of 4` Holm nulls rejected; all five validity gates passed on both datasets. The
mechanism reading: duplicating transcript content barely moves the deployed head key,
and the items the orbit *does* change flip at **coin-flip precision** (`0.504 / 0.488`).

**The C02 test, applied to every candidate below:** *a mechanism that perturbs a channel's
quantity or shape without introducing new semantic content is expected to measure ≈ 0.*
Each candidate is screened against it explicitly.

---

## 2. C05 — Full-Bank Signed Discourse Manifold

> **Claim.** "Dense signed discourse relations over the entire train bank can reshape
> membership globally rather than re-score a frozen top-20 tuple."
> **Registry dedup boundary.** "Requires a fresh non-isomorphism gate against SSR/EDCM,
> RDK and LB-SCGP Global-R2; fixed-top20 reordering and raw-arena promotion remain
> forbidden."

The registry demands a written non-isomorphism gate before C05 starts. The parent task
asks for that gate to be attempted **now, on paper**, with the rule that failure to write
it is itself a pre-kill. Here is the attempt.

### 2.1 The gate, comparator by comparator

C05's object is a three-stage pipeline: *(i) a dense signed relation matrix over the
whole train bank → (ii) an encoder fit that realises it → (iii) the ordinary deployed
kNN vote, untouched.* A non-isomorphism gate must name a property of the **mechanism**,
not of the vocabulary, that no comparator has.

**(a) vs LB-SCGP Global-R2** `[R]`. Global-R2 is, verbatim: *"a closed train-only
label-blind structural-certificate cache compiled into one replayable full-bank
PSD/unit-diagonal Gram target fitted uniformly by the shared encoder, with test as
ordinary full-video train-memory top-20 kNN"* (TARGET_FINDINGS, Iteration 7). Stages
(i)–(iii) are identical. **The only free variable is the source of the relation matrix.**
Enumerate the legal sources:

| relation source | status |
|---|---|
| MLLM-emitted discourse relations | `banned_constraints[5]` (MLLM-scores-as-training-signal) and `[6]` (P1–P5 re-proposals) |
| gold labels (`1[lab_i == lab_j]` or signed variants) | this is NCA's own objective; F75/NCA-side by RDK's own ruling (§2.1b) |
| label-blind structural certificates | **this *is* Global-R2** — dead as the 15th negative, killed pre-GPU by G0-cond |

Global-R2's epitaph `[R]`: *"killed pre-GPU by G0-cond: cache 91-93 % constant,
oracle@coverage 10× under bar, v3 rejected (parsed certs noise-quality)"*; coverage was
`8.7 % / 6.9 %` and real-A conditional information `≤ 0`.

`[I]` **The gate cannot be written unless C05 names a fourth relation source that is
neither MLLM output, nor gold labels, nor the Global-R2 certificate family. The registry
entry names none.** "Signed discourse relations" is a description of the relations'
*semantics*, not of their *provenance*, and provenance is the only axis that separates
C05 from Global-R2.

**(b) vs RDK (F99)** `[R]`. RDK trains one shared map on banked keys so that cosine
geometry reproduces a relation geometry, leaving the deployed vote untouched — the same
shape as C05. F99's own ruling: RDK *"applies ONE SHARED MAP TO BOTH SIDES, i.e. it falls
on NCA's side"*, and the verifier it distils was fitted on `1[lab_i == lab_j]`, *"the SAME
label-agreement matrix NCA optimises"* — RDK is *"NCA with a two-stage estimator."* A
signed discourse relation matrix over the whole bank is the same object with a different
label on the arrows. Additionally F113 measured that **any fitted relation score over head
keys memorises the bank** (in-sample pair AUC `0.9999`) and is **worse than the plain
cosine on held-out pairs** — `ΔAUC +0.1572 / +0.2302` raw collapsing to `−0.0643 / −0.1294`
head, on **30/30 fold cells**.

**(c) vs EDCM** `[R]`. EDCM was *"label-blind train-only `V/S/O` coalition
pseudo-signatures control the listwise gradient of ordinary full-video keys used by final
kNN"* — that is, verbatim, dense relations over the train bank reshaping which items get
retrieved. Its A0 was the closest existing measurement of C05's oracle: an exact
video-label-only top-64/two-swap screen, which reached `+0.0273 acc / +0.0394 mF1` on
MHC-EN (15 errors, 36.79 % support) and `+0.0380 / +0.0444` on MHC-ZH (22 errors,
62.87 % support) — **both below the `+0.050 / +0.050` gate that C05 must also clear.**
EDCM's anti-repeat clause is explicit: *"The next route must change the video-level
correctable unit."* C05 does not change the correctable unit; it is still whole videos.

**(d) vs SSR** `[R]`. SSR used reliability-filtered MLLM stance–target–mechanism relations
to select paired-seed directed memory constraints. C05's "signed discourse relations" is
SSR's relation vocabulary made *dense* rather than *sparse*. SSR's terminal preflight:
even under the impossible best case that every candidate is an accepted reliable relation,
oracles were `+0.0036 / +0.0128 / +0.0052 / +0.0259` acc — all below `+0.050`. Its
anti-repeat: *"future methods must prove sufficient video-label-only OOF correctable
coverage before MLLM calls."* `[I]` Density changes how many constraints are asserted per
item; it does not change how many errors are membership- or permutation-reachable, which
is what the SSR/EDCM oracles were measuring.

### 2.2 Reachability prior

`[I]` C05's realisable channel is *raising top-20 purity*. Three independent bounds
(§1.1–1.3) put that channel below the `+0.050` Stage-0 bar on at least HateMM, and the
error forensics say membership is not where the failure lives: on MHC-ZH, in the pre-head
raw fused space over the full 579-row bank, **the first same-gold-class train neighbour
sits at median rank 1.5** for the 22 core errors (11 of 22 at rank 1; all 22 within rank
14) — *"The right analogues are present and top-ranked; they are simply out-voted"* `[R]`.
HateMM's median rank of the first true-label neighbour is `3.0`, with `6/27` errors having
**no** true-label neighbour in the top-20 at all.

The family's own delivery record is the sharpest warning `[R]`: F98/AGGNET held *"the
LARGEST ORACLE CEILING EVER MEASURED ON THIS OBJECT"* — `+0.1492 / +0.1520 / +0.2186`,
96–100 % of every deployed error inside its function class — and delivered
`+0.0134 / −0.0069 / +0.0000`. *"What binds is neither reach nor capacity but that the
local configuration carries no learnable signal about which neighbours to trust at
n = 549–744."*

### 2.3 Assets and cost

`[R]` **No existing bank adjudicates C05's Stage-0.** Every banked family varies image
pooling (`p3pool_*`), attention/readout (`bidir*`, `*-ro_L{24,28}*`), LoRA weight point,
PEFT merge path (`nullop2merge`) or transcript density (`*-c02den-*`). None expresses a
re-fitted encoder under a full-bank relation target — and such a bank cannot be produced
by extraction at all: it requires **training** the encoder, which is the same order of
commitment the amendment explicitly declined to price for C03.

`[I]` C05's Stage-0 oracle is nonetheless **CPU-reachable without any extraction**: the
legal screen is a full-bank membership-reshaping oracle in the existing fold-head arena
(§1.4), which is what EDCM's A0 already did in restricted form (top-64, two swaps). Cost
estimate: `0 GPU-h`, a few CPU-hours of analysis, plus review ceremony.

### 2.4 Verdict

**PRE-KILL (gate unwritable as posed).** The registry's own rule fires: the
non-isomorphism gate against Global-R2 cannot be written without naming a relation source
outside the three that are individually banned or dead.

> **Strongest single kill-risk.** C05's only realisable channel is raising top-20 purity,
> and the one measured purity→accuracy conversion in this project caps a *perfect* purity
> at `+0.0286` (95 % upper bound) — below the `+0.030` final bar, let alone the `+0.050`
> Stage-0 bar.

**Reopening condition, stated so it is usable.** C05 becomes writable iff a proponent
names a relation source that is (a) not MLLM output, (b) not the gold label agreement
matrix, (c) not the Global-R2 certificate family, and (d) carries measured conditional
information over the deployed key — the G0-cond gate that killed the A-line. If such a
source exists, the *first* legal step is the `$0` full-bank membership oracle in the
fold-head arena, not an encoder fit.

---

## 3. C06 — Prompt-Orbit Tangent/Curvature

> **Claim.** "The tangent and curvature of a video's representation across a fixed prompt
> orbit may encode policy-bound semantic instability missed by any single prompt."
> **Dedup boundary.** "One trained representation uses orbit geometry; it may not become
> prompt selection or a multi-prompt prediction ensemble."

### 3.1 The two-point case is already measured — and it failed against random rotations

C01's A0 (job `13738`, `KILL_CURRENT_ENDPOINT_ROUTE_ONLY`, `MHC_zh = false`,
`HateMM = false`) measured exactly the **first-order tangent** of a prompt orbit: the
displacement between the standard-prompt L24 endpoint and the one-word-prompt L24
endpoint. Its frozen control set included **fixed orthogonal endpoint rotations with
identical block L2** — i.e. random directions with the real displacement's norm. Read
directly from `C01_A0_OUT.json` `[R]`:

| arm | HateMM acc / net | MHC-ZH acc / net |
|---|---|---|
| `endpoint_std` (reference) | `0.8411` / `0` | `0.8590` / `0` |
| `displacement` (real) | `0.8505` / `+1` | `0.8846` / `+2` |
| `common_displacement` (**primary**) | `0.8598` / `+2` | `0.8590` / `0` |
| `common_interaction` (secondary) | `0.8224` / `−2` | `0.8333` / `−2` |
| best **random** rotation (`orthrot_83p8`) | `0.8692` / `+3` | `0.8974` / `+3` |
| `orthrot_72p7` | `0.8505` / `+1` | `0.8974` / `+3` |

`[I]` **A random orthogonal direction with matched block norm matches or beats the real
prompt displacement on both datasets** — on MHC-ZH by `+0.0384` accuracy over the primary
arm. That is the strongest available evidence against C06's premise: at the two-point
case, the prompt-orbit direction carries no more usable structure than a matched-norm
random direction.

**Honest scoping, all three caveats stated.** (i) C01's own decision text limits it: *"A
KILL decision retires only the current standard-L24 versus one-word-L24 endpoint-contrast
route. It does not falsify policy contrast under same-pooling caches."* (ii) The standard
and one-word endpoints confound the prompt with the readout/pooling token, which is why
C01's claim was narrowed in review. (iii) F66/NCA recon states that raw-arena selection
results do **not** bind *trained* reshaping. Under F113, however, a raw-arena **kill** is
one-sidedly secure, so the negative transfers well even if a trained version is not
literally refuted.

### 3.2 The C02 test, and a structural objection

`[R]` Adjacent prompt-axis results: F80 (ZH extraction-instruction language, `*-zhp`
banks) is `0/3` on both arms and both protocols with an explicit *"do NOT re-propose
prompt-language matching elsewhere"*; F70 (readout grid: intermediate-layer /
one-word-prompt / last-token) caps all cells at `≤ +0.020`.

`[I]` **Structural objection, and it is the decisive one.** The prompt is a *constant*:
`TEXT_INSTRUCTION` and `IMG_INSTRUCTION` are fixed English strings identical for every
video. A prompt orbit therefore injects **no per-item information**. The only per-item
quantity is a given video's *response* to a fixed perturbation — which is precisely what
C01 measured and found rotation-indistinguishable. Under the C02 test this is the same
family: perturbing the input's shape without new semantic content.

C14 is the ensemble form of the same idea and is registry-forbidden as a final method;
TVB ranked multi-prompt ensembling **7 of 7 at ~0** `[R]`.

### 3.3 Assets and cost

`[R]` Existing prompt/readout banks on **both** required datasets:
`{train,dev_seen}_Qwen2.5-VL-7B-Instruct-LoRA{,-curric}_HF-ro_{L24,L28,ow_L24,ow_L28}.pt`
— 2 prompt points × 2 layers. `-zhp` exists on MHC-ZH only. **Two points give a chord,
not a curvature**; a tangent/curvature orbit needs `≥ 3` prompt points ⇒ extraction
required.

`[I]` **Cost, calibrated on C02's actual extraction** (job `13843`, `[R]`: `02:28:15` =
`2.4708 GPU-h`, `8727` forwards ⇒ `≈ 1.02 s/forward`). Train + dev over both datasets is
`744 + 107 + 579 + 78 = 1508` items, so one additional **text-leg** prompt point costs
`≈ 0.42 GPU-h`; four to six new points is `1.7–2.5 GPU-h`, inside the `4.0` cap. Varying
the image-leg instruction roughly doubles it.

### 3.4 Verdict

**PRE-KILL (premise measured rotation-indistinguishable at the two-point case), with a
`$0` falsifier that should be run before any strike is final.**

> **Strongest single kill-risk.** The prompt is identical for every video, so a prompt
> orbit adds no per-item information; the only per-item quantity is the response to a
> fixed perturbation, and C01 measured that response to be indistinguishable from a
> matched-norm random rotation on both datasets.

**The `$0` falsifier, priced not spent.** Re-run the C01 v4 arm battery — real
displacement vs. the matched-norm orthogonal rotations — in the **fold-head arena** on the
four already-banked `ro_*` caches. Zero GPU, zero extraction, minutes of CPU on an
instrument that already exists. If the rotations again match the real displacement in the
deployed head space, C06 closes for `$0` and the `1.7–2.5 GPU-h` is never spent. If they
do not, C06 has earned its extraction.

---

## 4. C07 — Harm-Lattice Cone Metric

> **Claim.** "A cone metric over harm-act partial order may preserve distinctions
> collapsed by binary supervision."
> **Dedup boundary.** "High collision risk with prior LBOP/lattice work; promotion
> requires a written mathematical delta and a reachability screen before implementation."

### 4.1 The supervision the mechanism needs does not exist legally

`[R]` `hard_constraints`: *"only the parent-video binary label is gold; no segment/span/
target gold."* `banned_constraints[1]`: *"gold annotations inside method (time-span,
target)."* `[4]`: *"target-as-structure at 7B."* `[5]`: MLLM-scores-as-training-signal.

`[I]` A harm-act partial order must come from somewhere. From gold — forbidden, and none
exists. From an MLLM — `[5]`. From the binary label — a binary label induces a two-element
chain, whose "cone metric" is the ordinary two-class geometry the head already fits.
**The legal source set is empty**, exactly as it was for EUM's unit definitions.

### 4.2 The one realisation that was measured came in at oracle `+0.0256`

`[R]` The closest existing instantiation of "distinctions collapsed by binary supervision"
is the graded 3-class route (Offensive / Hateful / neither), which is in the dead list
("Graded 3-class soft-label (Offensive reweighting, EN+ZH; litsweep-5 S2 #1)"). F82's
ZH graded oracle is **`+0.0256`** and down-weighting Offensive is *monotonically harmful*.
ERRPAT-ZH §5.3 adds the mechanism: on ZH the errors are **not** concentrated in Offensive
(Offensive `n=28` err `0.2500`; Hateful `n=17` err `0.2941`) — *"there is no
Offensive-specific error mass to reallocate."*

`[I]` An oracle of `+0.0256` is below the `+0.030` final bar and roughly half the
`+0.050` Stage-0 bar, on the dataset where the graded structure is richest.

### 4.3 A dedup reference that cannot be resolved

`[M]` "LBOP" returns **zero hits** across `refine-logs/*.md` and
`autoresearch/goal_mllm_plus3/state/directions_tried.json`. `[I]` The registry's own
dedup boundary points at a body of work that does not exist in the repository under that
name; the most likely intended referent is **LB-SCGP** (label-blind structural
certificates → PSD Gram target), whose local rank-cell v1–v7 and Global-R2 lineages are
both in `standing_eliminated_families`. If that is the referent, C07 inherits the same
comparator problem as C05 §2.1(a). This should be resolved by whoever wrote the entry
before C07 is scheduled.

### 4.4 Verdict

**PRE-KILL.** It is also a **metric-channel** candidate and therefore inherits §1.3's
`+0.0286` ceiling on top of everything above.

> **Strongest single kill-risk.** The partial order C07 needs is not gold, cannot be
> legally derived, and its one measured realisation — the graded 3-class label — has an
> *oracle* of `+0.0256`, below the final bar before any conversion loss.

---

## 5. C08 — Provenance-Antisymmetric / Title-Source Encoder

> **Claim.** "Antisymmetric encoding of content source versus author stance can reduce
> title/source leakage and quoted-hate false positives."
> **Dedup boundary.** "Changes source/provenance representation content; not OCR, archive
> key addition, stream weighting, or a decision router."

C08 rests on three premises. All three are checkable at `$0`, and all three fail.

### 5.1 Premise 1 — "title leakage": measured FALSE as posed

`[M]` The gt rows on all three datasets carry exactly `['id', 'label', 'text']` — there is
no `title` field to leak. `[R]` LITSWEEP2: *"`title_present = 0` in every split of
HateMM/MHC/MHC_zh"*, so the extractor's `Title:` line is always literally `(none)`.
LITSWEEP5 adds that MHC-EN's title *content* is already folded into `text` by
`prep_mhc.py:72` (`Title + " . " + Transcript`) and is not separable at the gt level.

`[I]` There is no title channel to de-leak, and no way to isolate EN's folded title
without re-deriving the source metadata (a data-collection act, veto-adjacent, and
declined LOW/~0 in litsweep2).

### 5.2 Premise 2 — "source leakage": real, but single-dataset

This is the one genuine provenance artifact in the corpus, and I measured its extent
directly.

`[M]` Census over `data/gt/<DS>/{train,val}.jsonl` (label files only; no cache, no test):

| dataset | split | `n` | pos | whitespace-only `text` | rows with an HTML tag | rows with an HTML entity | median chars |
|---|---|---|---|---|---|---|---|
| HateMM | train | 744 | 298 | **39** | 0 | 0 | 696 |
| HateMM | val | 107 | 43 | **9** | 0 | 0 | 429 |
| MHC-ZH | train | 579 | 180 | 0 | **243** | 1 | 106 |
| MHC-ZH | val | 78 | 28 | 0 | **34** | 0 | 111 |
| MHC-EN | train | 549 | 168 | 0 | 0 | 64 | 369 |
| MHC-EN | val | 80 | 25 | 0 | 0 | 9 | 443 |

The ZH tags are **exclusively** `<em class="keyword">…</em>` (histogram: `em` 254,
`/em` 254, nothing else) — the Bilibili search-result highlight around the term the clip
was harvested by. `[R]` This matches LITSWEEP3's independently measured `243/579` and
`34/78` and its identification of the deployed ZH "transcript" as the Bilibili
**description**, not the Whisper ASR.

`[I]` The provenance artifact exists on **MHC-ZH only**. HateMM and MHC-EN carry zero
tags. A `≥2`-dataset route cannot be built on it.

### 5.3 Premise 3 — "quoted-hate false positives": measured at chance

`[R]` ERRPAT-ZH §5.4 tested exactly this population: negatives carrying the markup
false-positive at `0.172` vs `0.0776` without — a `2.2×` raw ratio — but at the core-error
level **`5` observed vs `4.57` expected, `p = 0.5022`**. The five stable FPs that do carry
it are all clips *about* the slur (`公主病`, `妈宝男`, `花痴`, `流氓`, `绿帽`), the expected
counter-speech confusion, *"but the count is exactly chance."*

`[I]` And the representation-level version of "separate endorsement from quotation" is
C01's claim verbatim, already killed at its endpoint route.

### 5.4 Verdict

**PRE-KILL (all three premises fail).**

> **Strongest single kill-risk.** Both named leakage sources are measured away: the title
> field does not exist in any split of any dataset, and the source markup exists on
> MHC-ZH alone (`243/579` train, `0` on HateMM and MHC-EN) — so no `≥2`-dataset route
> exists even if the mechanism worked perfectly.

---

## 6. C09–C14 — brief screens

### C09 — Stable-Inversion Topology Surgery · **VIABLE (Stage-0 is zero-GPU)**

`[I]` The **only** candidate in the backlog aimed squarely at the population the error
forensics actually found: `~90 %` seed-invariant, confident neighbourhood inversions
(F88; ZH `22/25` fail in all three seeds and nothing fails in exactly two; all 12 ZH false
negatives are 3/3 stable). And it is the **only** candidate whose Stage-0 needs **no
extraction at all**: OOF-stable inversions can be identified and their oracle priced in
the already-banked fold-head arena (`headspace_{mint,arena}.py`, HateMM + MHC-ZH × 3
seeds, `~35 s` CPU per fold-head). `0 GPU-h` to a Stage-0 verdict.

Kill risks, in order: (i) any encoder-level pull of an inversion toward its right analogue
is a label-using metric move ⇒ F75/NCA and §1.3's `+0.0286`; (ii) Feldman /
Feldman & Zhang predict this exact seed-invariant confident error set **and** that no
operator fixes it, because memorisation is necessary and unavailable for held-out items;
(iii) F78 curation already failed its own random control on this population; (iv) the
"constraining break exposure" clause has to beat an exchange rate never observed above
`1.17` in 36 cells. Counterweight: `NCA_FORENSIC_RECON.md:110` explicitly rules that
**F66 does not bind trained-space reshaping**, so this is not foreclosed the way the
eval-time families are.

> **Kill-risk in one sentence.** The error set it targets is the one the literature it
> would have to cite predicts is unfixable by any operator, because the fix requires
> memorising items the model has never seen.

### C10 — Gold-free Reasoning-Boundary Structured Memory · **PRE-KILL**

`[R]` Its dedup boundary requires a strict gate against EUM, bank synthesis, segment set
matching and archive memory. EUM's ban_scope closes *"Sub-video retrieval UNITS of any
kind — 'evidence units', spans, segments, clips, shots — as the object stored in and
retrieved from the memory bank … closed three times over … Do NOT re-propose under a new
name"*, and states that without gold spans (`[1]`), without MLLM boundaries (`[5]`/`[6]`)
and without per-item selection (Law III), **the legal unit-definition space is EMPTY**.
"Reasoning-boundary unit" is a sub-video unit under a new name. Separately, BSY records an
**open user ruling** on `banned_constraints[3]` (pseudo-label vs. vote-pool expansion)
that *"BLOCKS ANY PREREG"* for every bank-addition candidate until resolved — C10 is one.

> **Kill-risk.** The unit definition is the whole mechanism, and the legal definition space
> for sub-video units was already enumerated and found empty.

### C11 — Null-aware Evidence Representation · **PRE-KILL on arithmetic**

`[M]` Whitespace-only `text`: HateMM `39/744` train and `9/107` val; **MHC-ZH `0/579` and
`0/78`; MHC-EN `0/549` and `0/80`.** `[R]` C02's extraction independently reported the same
HateMM degeneracy census (`EMPTY_TEXT 39 + LENGTH_GUARD 9` on train, `9 + 1` on val).

`[I]` The null-evidence state **exists on exactly one dataset**. The ceiling on MHC-ZH and
MHC-EN is identically zero, so the `≥2`-dataset conjunct is unreachable regardless of how
well the mechanism works — a structural, not statistical, block. A softer "thin evidence"
reading would have to redefine the null state on a continuous length variable, which is
the quantity channel C02 measured at `+0.0009 / −0.0012` today.

> **Kill-risk.** Two of the three datasets contain zero instances of the phenomenon.

### C12 — Archive-version Stability Curriculum · **PRE-KILL**

`[M]` `data/Archive/` contains exactly `MHC` and `MHC_zh` (each with v1 and `v2/`) — the
archive **does not exist for HateMM**, confirming the OCR recon's statement `[R]`. `[R]`
The archive is Qwen-generated text, so using its cross-version (in)stability as a training
signal is `banned_constraints[5]`. Archive-as-retrieval-key measured `ΔAcc −0.0014 ± 0.0313`
over 5 seeds with **zero vote flips**; archive-as-key accuracy claims were **withdrawn** as
selection artifacts; and the AUTO two-vote archive repair returned `C − A = 0`. MHC-EN is
additionally closed at all three levels (F55).

> **Kill-risk.** Its signal is MLLM-generated text used as a training signal — the ban that
> closed the P4–P11 family — and its own channel already measured a zero-vote-flip null.

### C13 — ZH HTML Markup Invariance · **STRIKE as a route; KEEP as a limitation**

Single-dataset by its own dedup boundary. Beyond that, `[M]` the markup is **not neutral
nuisance** — it is the strongest lexical shortcut in the ZH text channel:

| split | hate rate, `<em>` present | hate rate, `<em>` absent | base rate |
|---|---|---|---|
| train | **0.580** (141/243) | 0.116 (39/336) | 0.311 |
| val | **0.588** (20/34) | 0.182 (8/44) | 0.359 |

49 distinct keywords, 254 occurrences; the top terms are slurs (`傻逼` 42, `阴茎` 20,
`娘炮` 16, `傻屌` 11, `公主病` 11). Markup as a fraction of characters: median `0.0000`,
p90 `0.5155`, max `0.8621`, with `203/579` train rows above 10 % — substantial against a
median text length of only 106 characters. `[R]` LITSWEEP3 already noted the highlight *"is
baked into the current 0.8537 floor and inadvertently surfaces the slur."*

`[I]` Making the encoder invariant to the markup **removes a strongly predictive marker**
and is at least as likely to cost ZH accuracy as to gain it. The genuinely valuable
reading is the inverse one: part of the ZH floor rests on a collection artifact. That is
paper-grade limitations / pillar-4 material, and worth writing down — it is **not** a
performance candidate.

> **Kill-risk.** The "nuisance" it proposes to remove predicts the label at `0.580` vs
> `0.116`; invariance to it is a plausible *regression*.

### C14 — Multi-prompt Representation Ensemble · **STRIKE (already registry-ineligible)**

`[R]` `eligible_for_primary_target: false`; ensemble predictions are forbidden as a final
method and a positive diagnostic cannot satisfy novelty. TVB ranked multi-prompt
ensembling **7 of 7 at ~0**. `[I]` C02's verdict further reduces its diagnostic value:
the orbit family's own best arm added only a retrieval-length artifact.

---

## 7. Cost sketch to a Stage-0 verdict

Calibration `[R]`: C02's bounded extraction (job `13843`) ran `02:28:15` = `2.4708 GPU-h`
for **24 view caches / 8727 forwards** across 6 views × 2 datasets × 2 splits — `61.8 %`
of the `4.0` cap, `≈ 1.02 s` per forward. Its A0 (job `13847`) was **8 CPU / 0 GPU / 32 G,
`00:29:49` wall**, of which the arena itself was `53.9 s` and the rest was fold-head mints.

**Two routing facts that the 2026-07-31 cloud ruling does not relieve** `[R]`+`[I]`:

1. **A bounded Stage-0 extraction is local-only.** `generate_c02_density_view_text_embedding_HF.py`
   takes `--video_dir`/`--num_frames` and calls `BASE.load_video_frames(...mp4)`; job
   `13843` failed to decode `hate_video_95.mp4` (decord `av_read_frame failed with
   1094995529`, then PyAV `Error splitting the input into NAL units`) and took the frozen
   zero-vector guard path. Extraction reads **raw video**, which is the one category the
   data boundary keeps off Modal. Every extraction row below is therefore **local SLURM**.
2. **The A0 is CPU-only once the banks exist.** Confirmed by job `13847`.

| candidate | extraction to Stage-0 | routing | A0 | total GPU-h |
|---|---|---|---|---|
| C05 | **none possible** — needs encoder *training*, not extraction; but the legal oracle needs no bank | n/a | CPU, fold-head arena | **0** |
| C06 | 4–6 new text-leg prompt points × 1508 items ≈ `1.7–2.5 GPU-h` | local only (reads mp4) | CPU | **1.7–2.5** |
| C07 | n/a (no legal supervision to extract) | — | — | — |
| C08 | n/a (premises fail at `$0`) | — | — | — |
| C09 | **none** — fold-head arena already banked | n/a | CPU, `~35 s`/fold-head | **0** |
| C10–C14 | n/a | — | — | — |

---

## 8. Recommended order for the post-C04 Gate-0 reopen

**Order: C09 → (C06 `$0` falsifier) → C05 only if its gate becomes writable. Strike C07,
C08, C10, C11, C12, C13, C14.**

**Justification.** The reopen should be told plainly that the ordered backlog does not
hold ten live candidates. Seven of the ten fail a `$0` screen on premises that are already
measured — C07 has no legal supervision and a `+0.0256` oracle, C08's title channel does
not exist and its source channel is single-dataset, C10 re-proposes the sub-video unit
whose legal definition space was enumerated and found empty, C11's phenomenon has **zero**
instances on two of three datasets, C12's signal is MLLM-generated text under
`banned_constraints[5]` on a channel that measured zero vote flips, C13 would delete a
feature that predicts the label at `0.580` vs `0.116`, and C14 is registry-ineligible by
construction. Of the three that survive, **C09 should go first and alone**: it is the only
candidate aimed at the population the error forensics actually found, and the only one
whose Stage-0 costs **zero GPU-hours** because the fold-head arena it needs is already
banked — so it can be adjudicated while C04's teacher tranche holds the serial-execution
lock, and a kill costs nothing but review time. **C06 should not be scheduled as a
candidate at all until its `$0` falsifier runs**: re-running C01's real-displacement-vs-
matched-norm-rotation battery in the fold-head arena costs minutes of CPU and, on the
two-point evidence already in hand, is more likely to close the direction than to open it —
spending `1.7–2.5 GPU-h` of local queue on new prompt points before that check would be
the sequencing error the amendment was written to prevent. **C05 should be held, not
scheduled**, until someone can name a relation source that is not MLLM output, not the
gold label-agreement matrix, and not the Global-R2 certificate family; the registry made
the non-isomorphism gate a precondition, and on today's evidence it cannot be written.

**The reopen's real agenda item, therefore, is not ordering — it is replenishment.** After
C04 resolves, the backlog contains at most one candidate with a live mechanism, one `$0`
falsifier, and one blocked gate.

---

## 9. Strongest overall observation

**Every remaining candidate acts on the retrieval-key / representation channel or on the
input-text channel, and both are now bounded below the Stage-0 bar by measurements the
campaign already owns.**

On the retrieval side there are three *independent* ceilings: `coverage(20) ≥ 0.9829`
means there is nothing outside the top-20 worth retrieving (marginal oracle `≤ +0.0171`);
the pure-permutation oracle caps set-preserving re-metrication at `+0.0279` on HateMM and
`+0.0470` on MHC-ZH under a zero-break assumption never once observed; and the measured
purity→accuracy conversion caps *perfect* purity at `+0.0286`. Each of those sits under the
`+0.050` Stage-0 bar on at least one of the two datasets a PASS must clear. On the input
side, C02 measured the quantity/shape channel at `+0.0009 / −0.0012` today, with changed
items flipping at coin-flip precision.

`[I]` The consequence for the reopen is a reframing, not a ranking: **the backlog's problem
is not that C05–C14 are weak candidates, it is that they all live inside the one channel
this campaign has already bounded.** The two candidates that are *not* fully inside it —
C09, which targets the error population rather than the retrieval geometry, and C04, which
is out of scope here — are the only places the current registry has left to look. A
Gate-0 reopen that only re-orders C05–C14 will re-derive this document in three candidates'
time.

---

*No Python experiment on GPU, no model load, no teacher call, no cache write, no SLURM
submission, no test-split access, and no edit to `TARGET_STATE.json`, `TARGET_LOOP.md`,
`TARGET_FINDINGS.md` or `TARGET_REVIEW_RAW.md` occurred in producing this record. The only
files opened for measurement were `data/gt/{HateMM,MHC_zh,MHC}/{train,val}.jsonl`. The only
file written is this one.*
