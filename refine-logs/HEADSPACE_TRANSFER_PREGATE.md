# HEADSPACE-TRANSFER — does the campaign's best RAW-arena result survive the trip to the DEPLOYED HEAD SPACE?

**Date:** 2026-07-28 NZST · **Agent:** headspace-transfer · **Cost: `$0`** — CPU only, ≤ 8 threads,
**zero GPU, zero SLURM, zero Modal, zero test-split contact.**
Repo sha at freeze time `e841ff3` (working tree dirty; several agents writing `main` concurrently).

**Test contact: NONE, enforced in code.** The only feature files any script in this record opens are
`data/CLIP_Embedding/HateMM/{train,dev_seen}_Qwen2.5-VL-7B-Instruct-LoRA-curric_HF.pt`. A `torch.load`
guard installed at import time (`scripts/analysis/headspace_mint.py`, `_guarded_torch_load`) raises on
any path containing `test_seen` or `/test`, and the harness's own "test" dataloader is a **dummy
stratified slice of data the head already trains on**. No test-split file is opened, no test label is
read, no test metric is computed or reported. The one banked artefact read from a mixed file is the
**dev-only** `Val_Retrieval` line set of the floor trainlogs, extracted with a `grep -o` filter that
cannot emit a `Test_Retrieval` line (§4.1).

**Determinism:** adopts `refine-logs/PREGATE_DETERMINISM_CLAUSE.md` (commit `d431782`) clauses
**DET-1** (thread env exported by the driver *before* any python process starts —
`scripts/analysis/headspace_drive.sh:10`; asserted in-process by `headspace_mint.det1_assert`),
**DET-2** (full `runtime` block in every output JSON/NPZ), **DET-3 Tier B**, **DET-4** (noted where a
non-convex estimator is used), and **GIT-1** (pathspec commits only).

**Companion records, cross-referenced and NOT duplicated:**
`refine-logs/INSTRUMENT_VALIDATION_RECON.md` (sibling agent — owns the *broad* instrument-validation
question), `refine-logs/STREAMCOMP_FORENSIC_RECON.md` §5.3 (the two matched raw→held-out pairs, F108),
`refine-logs/HEADCOV_PREGATE_RECORD.md` (the head-space instrument inventory and the ZH dev head-space
read). Not opened this pass: `PROVENANCE_AUDIT_2026-07-28.md`, `MEMBANK_C4_PREGATE_RECORD.md`.

---

## §0. FRAMING — WHY THIS EXISTS, AND WHY EITHER OUTCOME IS A RESULT

Nearly every verdict of the last three days was rendered in the **banked RAW fused key space**
(`l2n(concat(l2n(img), l2n(text)))`, 7168-d) under 5-fold item-disjoint train-LOO, via the F89-frozen
`scripts/analysis/mechfix_ops.py:75-96` (`deployed_vote`): F94 KSWEEP, F95 MECHNOV, F96 RESTRANS,
F97 VGA, F98 AGGNET, F105 VSW, F112 MEMBANK-C4. **The deployed system retrieves in the trained head's
1024-d space, not in that one.**

The reason raw was used is real and is stated in the frozen module itself
(`scripts/analysis/mechnov_pairverify.py:21-25`):

> *"Banked RAW encoder key spaces (seed-independent), NOT trained head spaces. The trained RGCL head
> memorises its own train split (LOO train acc 0.998, F47), so a verifier fitted in head space would be
> measuring memorisation. The raw space is the honest pregate arena."*

That argument is correct about **train-LOO over the full train split**, and it has a fix the campaign
never used: **train the head on 4/5 of the train split and query it with the held-out fifth.** The
held-out fifth is then genuinely unseen, the arena is not saturated, and the item set, the fold
assignment, the bank, the operator and every constant can be held **bit-identical to the raw arena**.
The key space becomes the only variable.

The cost of the raw shortcut is that we judge in a space the product does not use, and two matched
pairs already show the raw arena failing to predict held-out
(`refine-logs/STREAMCOMP_FORENSIC_RECON.md` §5.3(a)-(b), F108): MHC-ZH accuracy **+0.0156 raw →
+0.0067 held-out** (2.3× shrink), HateMM AUC **+0.011 train-LOO (n=744) → −0.011 dev (n=107)**, a
genuine sign inversion. Two pairs make the instrument **unvalidated**; they do not characterise it.

**Both outcomes of this record are first-class.**
* If the raw-space effect **survives** in head space, the raw arena is a usable screen and F105 is a
  near-miss in the space that matters.
* If it **evaporates**, the raw arena is **optimistic**, and every raw-space *closure* of the last
  three days is **more** secure than it was, not less — a screen that over-reports effects cannot have
  produced a false kill. Only the raw-space *positives* would be at risk, and F105 is the only one.

**This is a pregate and a diagnostic. It promotes nothing, and it is not a prereg for a GPU cell.**

---

## §1. WHAT IS UNDER TEST

### 1.1 The operator — F105/VSW, verbatim

`refine-logs/VSW_PREGATE_RECORD.md` §1 (design source `refine-logs/LITSWEEP6_RELGEN.md` §2 C4):

```
v(λ) = Σ_i (2·lab_i − 1)·cos_i·w_i·m_i(λ) / Σ_i w_i·m_i(λ) ,   m_i(0) ≡ 1
predict 1 iff v(λ) ≥ 0 ,   w = [20, 19, …, 1],  top-20 own-train neighbours
```

`m_i(λ) ≥ 0` monotone non-decreasing in `p_i`, the F95 pair verifier's `P(same-class)` for the pair
(query, i-th deployed neighbour). Retrieval, `k = 20`, the neighbour labels, the cosines and the
threshold are untouched. Three multiplier families, **identical grids to F105**
(`scripts/analysis/vsw_pregate.py:94-99`): `pow` (**PRIMARY**), `exp`, `lin`. λ selected on **inner**
folds inside the fitting pool (`StratifiedKFold(5, shuffle=True, random_state=17)`), ties toward
λ = 0. λ = 0 must reproduce the deployed vote **bit-exactly** (hard assert, §3.4).

**The raw-space result this record is testing:** `VSW_PREGATE_RECORD.md` §5 — HateMM × `pow` ×
λ-selected = **Δacc +0.0255**, ΔmF1 +0.0242, fold signs `+++++`, 36 fixed / 17 broken,
ER 2.1176, 53 changed, pos-rate 0.4368 vs bank 0.4005, λ\* per fold 3/2/3/2/3, on a deployed floor of
**0.8441** (n = 744). Permutation-validated at p = 0.0050 with an informative null. **1.2× under the
+0.030 K-VSW-1 bar.** It is the largest honest raw-arena effect the campaign has produced.

### 1.2 The arena — matched to F105 in every respect except the key space

| | **RAW arena (F105, banked)** | **HEAD arena (this record, new)** |
|---|---|---|
| key | `l2n(concat(l2n(img), l2n(text)))`, **7168-d** | `l2n( mlp[:-2]( l2n(img_proj(img)) ⊙ l2n(text_proj(text)) ) )`, **1024-d** (`src/model/classifier.py:114-149`; the deployed key contract is quoted at `scripts/analysis/mechfix_ops.py:16-17`) |
| items | HateMM train, n = 744, pos-rate 0.4005 | **identical** |
| folds | `StratifiedKFold(5, shuffle=True, random_state=0)` (`mechnov_pairverify.py:56-57`) | **identical, asserted item-for-item against the banked `scripts/analysis/vsw_ckpt/hatemm/f{0..4}.npz` `ho_idx`** |
| bank | the 4/5 fitting pool, raw keys | the 4/5 fitting pool, **head keys of a head trained on that same 4/5** |
| queries | the held-out 1/5, raw keys | the held-out 1/5, head keys — **never seen by that head in any role** |
| vote | `mechfix_ops.deployed_vote` | **the same frozen function, same sha** |
| verifier | F95 PRIMARY: PCA(256, full) on fitting-fold items → `[|z−z'|, z⊙z']` → MLP(128, 30 ep, Adam) | **the same frozen functions, same constants, same seeds** |
| seeds | none — raw features are seed-independent | **3 head seeds (0, 1, 2)**; the head is a trained object, so the arena inherits its seed |

### 1.3 Why the head must be re-minted, and what that costs

`HEADCOV_PREGATE_RECORD.md` §1.1 established the inventory this record inherits: `logging/Retrieval`
holds **228 `.pt` files, all `mntp_s1_cpuhead`** (the F92-dead bidir heads), **97 empty `ckpt/`
directories**, and **0 of the 9** P2-era deployed checkpoints named at
`scripts/analysis/p2_rerank_eval.py:55-63` exist — F78 extended to the whole inventory. There is no
deployed head to load.

`ERRPAT_HateMM_2026-07-26.md:526-532` prices the replacement: *"The align head trains and evaluates
end-to-end in **52 s of wall time on 8 CPUs** … the marginal cost of … every head-space diagnostic,
ablation and control the paper might want — is now a CPU minute, not a queue slot"*, with the caveat at
`:534-536` that *"CPU-trained heads are not bit-exact to the CUDA floor (−0.0031 final-epoch acc here),
so a CPU-trained arm must be paired against a **CPU-trained floor**, never against the banked GPU
floor"*. **This record honours that rule: every Δ in it is paired against a floor computed by the same
CPU-minted head in the same session.**

### 1.4 One recipe subtlety, found by reading the code and load-bearing for fidelity

`src/model/evaluate_rac.py:330` (`retrieve_evaluate_RAC_`) is the **only** caller of `model.eval()` in
the whole training loop, and there is **no matching `model.train()`** anywhere in `model_pass`
(the single `.train()` in `src/run_rac.py:650` belongs to the NCA bank builder, which this recipe never
enters). Consequences, both verified by reading:

1. Dropout is active only during **epoch 0**'s training steps and is inert for epochs 1-29. This is a
   property of the *deployed* recipe and is reproduced, not corrected.
2. Because evaluation runs with dropout inert, **the dev/test dataloaders draw no RNG at all**, so
   their size and contents **cannot perturb the training trajectory**. This is what licenses replacing
   the harness's test dataloader with a dummy: it changes logging only.
3. Switching `--eval_retrieval` off would leave dropout **active for all 30 epochs** and would *not* be
   the deployed recipe. It is therefore left on.

---

## §2. FROZEN BARS — all of §2 and §3 written before any head-space treatment number existed

### 2.1 GATE-FID — instrument fidelity. **Declared before the instrument was measured.**

A proxy head is usable only if it reproduces the banked deployed floor. The floor's *test* numbers are
out of bounds here, so the anchor is the floor's **dev** retrieval curve, which is a dev-split artefact
and costs no test touch.

> **FID-1.** Mint the **deployed-configuration** head (full train split, real dev split) for seeds
> 0/1/2 and compare its `Val_Retrieval` accuracy at the **final epoch (29)** against job **13241**'s
> banked `Val_Retrieval` accuracy at epoch 29, seed for seed. Report the per-seed Δ, the 3-seed-mean Δ,
> and the mean |Δ| over all 30 epochs. Call the 3-seed-mean |Δ| the **fidelity band `B_fid`**.
>
> **STOP RULE:** if `B_fid ≥ +0.0255` — the raw-space effect under test — **halt and report**; the
> measurement would be meaningless because the instrument's own error exceeds the signal.
>
> **Conclusions are allowed to live only outside the band.** Any head-space Δ whose magnitude is
> smaller than `B_fid` is reported as *inside the instrument band* and cannot carry a verdict, in the
> DET-3 Tier-C sense.

*Disclosed in advance:* the ERRPAT test-side band for the identical proxy recipe is **+0.0000
(val-selected) / −0.0031 (final-epoch)** on 3 seeds (`ERRPAT_HateMM_2026-07-26.md:42-44`). It is
external corroboration, not re-measured here, and it is a **test-side** number that this record does
not reproduce and does not need.

### 2.2 GATE-ARENA — the fold heads are a different object and are gated differently

A head trained on 4/5 of the train split **cannot** reproduce the deployed floor and is not asked to.
Two things are gated instead, both declared now:

> **ARENA-1 (fold identity, hard assert).** The 5-fold assignment used by the mint and by the arena
> must be **item-for-item identical** to the banked raw arena's, asserted against
> `scripts/analysis/vsw_ckpt/hatemm/f{0..4}.npz::ho_idx`. A mismatch aborts.
>
> **ARENA-2 (non-degeneracy).** The pooled head-space deployed accuracy over the 744 held-out items
> must lie in **[majority-rate + 0.02, 0.98]** = [0.6195, 0.98]. Outside ⇒ the arena is saturated (the
> F47 0.998 problem) or collapsed, no operator can show an effect in it, and the record **halts and
> reports that instead**.

### 2.3 PARITY-λ0 — hard assert, aborts the run

Imported from `VSW_PREGATE_RECORD.md` §3.6.1 unchanged and executed by F105's own frozen
`vsw_pregate.parity_lambda0` (`scripts/analysis/vsw_pregate.py:357-381`): for every family × fold the
VSW vote engine at **λ = 0** must return a vote vector and a prediction vector `np.array_equal` to
`mechfix_ops.deployed_vote`, and the pooled accuracy must match at 4 dp. **18 gates per seed, 54 over
3 seeds.** A single mismatch aborts before any treatment number is written.

### 2.4 K-HST-1 — **the transfer bar. THE PRIMARY READ.**

Head-space `VSW_pow` Δacc against the same-session head-space deployed floor, λ\* selected on inner
folds only, pooled over all 744 held-out train items, **3-seed mean**, HateMM.

| outcome | condition |
|---|---|
| **TRANSFERS** | `Δ_head ≥ +0.0128` (**half** the raw +0.0255) **and** ≥ 3/5 fold signs ≥ 0 on the seed mean **and** ≥ 2/3 seeds strictly positive |
| **PARTIAL** | `+0.0000 < Δ_head < +0.0128` |
| **DOES NOT TRANSFER** | `Δ_head ≤ +0.0000` |

Any of the three verdicts is reported as measured. **No promotion is made under any outcome**, and a
TRANSFERS verdict would require escalation to a formal prereg with independent review, not a promotion
inside this record. The **transfer ratio** `Δ_head / +0.0255` is reported alongside.

### 2.5 K-HST-2 — the membership diagnostic (declared reading rule, no pass/fail)

For the same query, same fold, same bank, measure the overlap of the **deployed top-20 membership**
between the raw arena and the head arena, and the change in the retrieved **label tuple**.

> **Reading rule, frozen now.** The campaign's central finding is that the vote reads only the
> retrieved label tuple: `LITSWEEP8_PATHOLOGY_MATCH.md` §2 Result A measures 99.6-100 % decision
> identity between the deployed vote and a label-only vote in the raw arena, and
> `HEADCOV_PREGATE_RECORD.md` §4.2 (K-HC-3) measures **1.0000** in the deployed head space.
> **Therefore: if the top-20 membership differs substantially between the two arenas, non-transfer is
> mechanically explained by the change of retrieved set, and is not evidence about the operator.**
> Threshold for "substantially", declared now: **mean overlap < 10 of 20**.

### 2.6 K-HST-3 — verifier informativeness in head space (declared reading rule)

The F95 control-1 quantity, recomputed in head space on the **full** held-out × in-fold pair matrix:
pair-AUC of the verifier vs pair-AUC of the cosine, plus the **in-sample fitting-pair AUC** (the
memorisation read the frozen module's own docstring predicts).

> **Reading rule, frozen now.** If `d_AUC = AUC_verifier − AUC_cosine ≤ 0` on the held-out pairs, the
> verifier carries **no** ordering information the cosine does not have in head space, VSW has nothing
> to spend, and a null is a property of the space rather than of the operator. F95 measured
> `d_AUC = +0.1572 / +0.2302 / +0.1785` in the raw fused space (`VSW_PREGATE_RECORD.md` §1). If in
> addition the **in-sample** AUC is far above the held-out AUC, the mechanism is the memorisation the
> frozen module warned about, now measured instead of assumed.

### 2.7 Degeneracy controls — imported from F105 §3.6 / F98 unchanged, same thresholds

`DEG-A` threshold twin ≥ 0.95 ⇒ FIRES; `DEG-B` fixed-k twin (the eight F94 profiles) ≥ 0.95 ⇒ FIRES;
`CLASS BALANCE` pos-rate within 0.10 of the bank rate else the nulls for that cell are VOID;
`DEG-D` cosine twin reported. All are computed by F105's own frozen `run_arms`
(`scripts/analysis/vsw_pregate.py:576-664`) and none is re-implemented here.

### 2.8 Permutation null — budget declared in advance, with an escalation rule

The raw F105 null cost ~10 s per (fold × draw) (`scripts/analysis/vsw_perm_hatemm_OUT.log`), i.e.
**~2.8 h of wall time per seed at 200 draws**; the head-space fit is the same size, so 3 seeds × 200
draws is ~8.5 h and is not affordable inside this record's budget.

> **Declared budget:** `N_PERM_HEAD = 30` on **seed 0**, PRIMARY family, F105's own `PERM_SEED = 12345`
> and F105's own shuffling scheme (fitting-fold **item** labels permuted; bank labels, retrieval,
> cosines, deployed floor and every gold label untouched). p-resolution 1/31 = **0.0323**.
>
> **Escalation rule, declared now:** the null is escalated to the full 200 draws **only if** the
> primary head-space Δ clears the F105 K-VSW-0 interest threshold of **+0.010**. If the primary Δ is
> ≤ 0, a null is not run at all and the record says so — a null tests whether an observed *positive*
> is beyond chance, and there is nothing to test otherwise.

### 2.9 Anti-shopping rule

**PRIMARY READ = HateMM × `pow` × λ\* × 3-seed mean × final epoch (29), pooled over 744 items.**
Every other cell in this record — the `exp`/`lin` families, the λ curve, the free transfer ladder of
§2.10, the per-seed reads, any other dataset — is **secondary and cannot be promoted into the primary
read after the fact**. No epoch is selected: the head is read at epoch 29, the last one, always.

### 2.10 The free transfer ladder — declared before the run, and why it replaces the tasked secondaries

The tasking named two secondary targets: the best AGGNET/F98 cell (+0.0134 HateMM) and the best
MECHFIX/F89 cell (T4 on MHC-ZH, +0.0067). Both are **priced but not spent**, for reasons recorded here
before any result exists:

* **AGGNET/C3** is a 1316-parameter trained conditional aggregator with its own harness
  (`AGGNET_PREGATE_RECORD.md`); porting it is a second implementation, not a re-run.
* **MECHFIX/T4 is already a head-space measurement.** `mechfix_ops.py:16-21` names the deployed head's
  fused embedding as its key space, and `MECHFIX_PREGATE_2026-07-27.md:310` reports T4 on MHC-ZH as
  **+0.0067 / +0.0052 (3-seed, val-selected / final)** on **proxy heads on the test split** — i.e. it
  is a head-space, held-out number already, and re-deriving it here would be a third arena rather than
  a transfer pair. **This is a correction to the tasking's premise and is reported as such.**

What *is* run, at **zero extra cost**, because the identical emitter produces them, is a **four-rung
ladder of raw-space effect sizes put through the same head space**:

| rung | operator | raw-arena datum being transferred |
|---|---|---|
| **1** | **VSW `pow`** (F105) | **+0.0255** — PRIMARY |
| 2 | F95 nominate-and-verify, `mlp_max` / `mlp_mean3` | −0.0040 / +0.0054 recorded, +0.0027 / +0.0107 same-session anchor (`VSW_PREGATE_RECORD.md` §4.3) |
| 3 | F94 fixed-k profiles `k ∈ {1,2,3,5,7,10,15,20}` | the k-sweep F94 closed |
| 4 | F89 MECHFIX eval-time operators T1 / T2a / T2b / T4, run through `mechfix_ops` unmodified | the calibration/geometry family |

Rungs 2-4 are **secondary**, cannot carry a bar, and exist so the transfer question is answered with a
monotone ladder rather than a single point.

---

## §3. FROZEN DESIGN

### 3.1 Reuse, not rewrite

Three frozen modules are imported **unmodified** with their sha256 asserted at run time
(`scripts/analysis/headspace_arena.py:47-51`):

| module | sha256 | what is used |
|---|---|---|
| `scripts/analysis/mechfix_ops.py` (F89) | `635c13124e79ba1a299bc13fc1175a03aa11e09924f5413ce51061793c83fc8d` | `deployed_vote`, `macro_f1`, `t1_class_balanced`, `bank_hubness`, `t2a_csls`, `fit_whitener`, `apply_whitener` |
| `scripts/analysis/mechnov_pairverify.py` (F95) | `77b0defd8eaa3688e58b6d5d17202bd55d16cf1f4a5aaafbe4b2b98598b7240d` | `l2n`, `pair_features`, `all_unordered_pairs`, `fit_mlp`, `predict_mlp`, `DATASETS`, **every frozen constant** |
| `scripts/analysis/vsw_pregate.py` (F105) | `ba9982dba98fb14dd53297ac6c087f2e1a4aa068490879d2121c50cf1f932eea` | `_fold_setup`, `multiplier`, `decide`, `vote_with_weights`, `select_lambda`, `selected_arm`, `curve`, `summarise`, `run_arms`, `parity_lambda0`, `best_threshold`, `load_perm_table`, **every λ grid and constant** |

**Every treatment quantity in this record is produced by F105's own `run_arms` / `parity_lambda0`.**
The only new numerical code is the **emitter**, which must be new because the key matrix is per-fold
rather than global; it is a line-for-line mirror of `vsw_pregate.emit_arena`
(`scripts/analysis/vsw_pregate.py:203-341`) with `X` replaced by `X_f` and with the extra
diagnostics of §2.5/§2.6/§2.10 appended.

### 3.2 The mint — `scripts/analysis/headspace_mint.py`

CLI byte-identical to `scripts/slurm/enc3seed_lora_curric.sbatch:56-70` (job 13241) except
`--device cpu`, `--num_workers 0`, `--group_name`, `--output_path`, `--force True`. Per-epoch
`state_dict` dumps are suppressed (~34 MB each, 1 361 of them over the 36 mint units ≈ 46 GB, and
nothing downstream reads them: `best_epoch_path` is re-loaded only on the EM branch, `src/run_rac.py:1521-1523`, which this
recipe never enters). The head read out is the **final-epoch** model object returned by `model_pass`,
never a re-loaded best checkpoint. Keys are extracted as `model.eval()` + `no_grad` forward with
`return_embed=True` (`src/model/classifier.py:147-149`) over the banked features exactly as
`CLIP2Dataloader` feeds them (`src/data_loader/rac_dataloader.py:95`, `.float()` only, `normalize=False`).

18 mint units: 3 seeds × (5 fitting-pool heads + 1 deployed-configuration head). Each is a separate
process driven by a retry loop (`scripts/analysis/headspace_drive.sh`) so a login-node reap costs at
most one unit — the `VSW_PREGATE_RECORD.md` §3.10 / LSMI precedent.

### 3.3 Held-out purity of the fold heads

For `fold ≥ 0` the patched loader returns `train = fitting pool`, `dev = a stratified 40-item slice of
the fitting pool`, `test = the same dummy`. **The held-out fifth is never given to the head in any
role** — not as training data, not as a dev set, not as an eval set, not for best-epoch selection
(which is not used anyway: epoch 29 always). Its head keys are computed only *after* training, by
forwarding the frozen final-epoch model.

### 3.4 Gate order

§0-§3 written and frozen → mint 18 heads → **GATE-FID** → **ARENA-1** (hard assert, inside the mint) →
**ARENA-2** → **PARITY-λ0** (hard assert) → arms, λ curves and degeneracy controls → K-HST-1/2/3 →
permutation null per §2.8 → verdict. Machine outputs:
`scripts/analysis/headspace_arena_hatemm_s{0,1,2}_OUT.json`,
`scripts/analysis/headspace_fidelity_OUT.json`, `scripts/analysis/headspace_report_OUT.json`.
**Every number in §4 onward is re-read from those JSONs at report time at 4 dp, never transcribed from
a run log.**

### 3.5 Frozen script sha256

Frozen here, before any head-space treatment number was computed (`sha256sum`, re-asserted in every
output JSON's `meta.script_sha256`):

| path | sha256 |
|---|---|
| `scripts/analysis/headspace_mint.py` | `cefdf8dc2f4a9aefa042ef7bec9b1d06c9721ae5b4a70ec117e9929ff0916612` |
| `scripts/analysis/headspace_arena.py` | `761b256bb0f27ca75ef7eb49e93a9c1edc9590d7de6c7544c1354d232ff95243` |
| `scripts/analysis/headspace_drive.sh` | `b5acd085324f65e7b652f28557ebb2bad215e8c6d77417d4ef455f0ea47c66b0` (orchestration + DET-1 export only; changes no arm) |

`scripts/analysis/headspace_report.py` is **reporting only** and is not part of the freeze.
`scripts/analysis/headspace_fidelity.py` is a **reader** for GATE-FID; its as-run sha256 is recorded in
§4.1 rather than frozen here, because §3.7 extends its floor-log table by one entry and the honest
record of that is the two shas plus a bit-identical re-run of the first dataset (§4.1).

### 3.7 PRE-RUN AMENDMENT — a SECOND dataset, declared before any MHC-ZH number existed

*Written after the HateMM primary completed and **before any MHC-ZH head was minted**, so that the
scope extension is visible rather than shopped.*

The HateMM primary answers *"does F105's +0.0255 survive the trip"* on **one** cell. The question the
tasking ultimately asks — *"is the raw pregate arena trustworthy for screening"* — is a question about
the **arena**, and the arena-level quantities (K-HST-2 membership, K-HST-3 verifier informativeness,
the cosine spread, the borderline population) are the ones that would have to generalise. One dataset
cannot establish that. **MHC-ZH is therefore added as a SECONDARY cell**, at the same `$0`:

* Encoder `Qwen2.5-VL-7B-Instruct-LoRA_HF`, n = 579 train, bank pos-rate 0.3109, floor job **13150**,
  CLI as replayed by `scripts/analysis/errpat_zh_remint.py:109-128`; identical mint, arena, gates.
* **It is NOT a transfer test of a positive and is not allowed to become one.** F105's raw MHC-ZH
  `VSW_pow` is **−0.0017** (`VSW_PREGATE_RECORD.md` §5), i.e. there is no raw-space effect to transfer.
  ZH carries **only** GATE-FID, ARENA-2, PARITY-λ0, K-HST-2 and K-HST-3, plus the free ladder as
  descriptive context.
* **The PRIMARY READ of §2.9 is unchanged**: HateMM × `pow` × λ\* × 3-seed mean. No ZH number may be
  promoted into it, and the K-HST-1 verdict is computed on HateMM alone.
* No bar, arm, constant, seed, family, λ grid, nesting rule, null spec or degeneracy control changes.

### 3.6 Disclosed pre-freeze smoke test

One mint unit (HateMM, seed 0, deployed configuration) was run **before** this document was frozen, to
verify that the harness runs at all and to time it: it completed in **40.3 s** and reported a
final-epoch dev retrieval accuracy. That number is an **instrument** number, it is re-derived and
reported in §4.1 with the other two seeds, and **no head-space treatment number existed at any point
before this freeze**.

<!-- ============ EVERYTHING BELOW THIS LINE WAS WRITTEN AFTER THE RUN ============ -->

---

## §4. RESULTS — HateMM (PRIMARY)

Every number below is **newly computed this session** and re-read at 4 dp from
`scripts/analysis/headspace_report_OUT.json` (produced by `headspace_report.py` from
`headspace_arena_hatemm_s{0,1,2}_OUT.json` + `headspace_fidelity_OUT.json`), never transcribed from a
run log. Raw-arena comparators are re-read from the banked `scripts/analysis/vsw_main_hatemm_OUT.json`
(F105) and are labelled as such.

### 4.0 What ran

18 mint units (3 seeds × {5 fitting-pool heads + 1 deployed-configuration head}), **30.6-40.4 s wall
each on 8 CPUs** (607.0 s for all 18), comfortably inside `ERRPAT_HateMM_2026-07-26.md:526-529`'s
~52 s/seed price. 3 arena runs, ~87 s each. Total for the HateMM primary: **~15 CPU-minutes. Zero
GPU, zero SLURM, zero Modal.**
Two arena processes were SIGTERM-reaped mid-run by the login node and resumed from their per-fold
checkpoints with no recomputation, exactly as designed (§3.2).

**`headspace_fidelity.py` sha256 as run:** `3e0a35cd93c7d1bd89f0acabd7288594b6a6b2f79dd49052a45c207aad90d2a2`
for the HateMM-only version; `72fd8e0aab61b635b4421b87bdbccc8ef6c58bf28fe1ff64cab0671e08bf6598` after
§3.7 added one entry to its floor-log table. **The HateMM output is bit-identical under both**
(verified by dict comparison excluding the sha and runtime blocks); the diff is two lines of data, no
logic.

### 4.1 GATE-FID — **PASS on the declared statistic, with a caveat that is reported, not buried**

Deployed-configuration CPU proxy head vs the banked GPU floor **job 13241**, on the **dev** split
(n = 107), final epoch (29). Dev-only reader, `Test_Retrieval` lines discarded at the point of read.

| seed | floor dev acc @29 | proxy dev acc @29 | Δ | mean \|Δ\| over 30 epochs | max \|Δ\| | corr over 30 epochs |
|---|---|---|---|---|---|---|
| 0 | 0.8505 | 0.8318 | **−0.0187** | 0.0125 | 0.0467 | 0.2502 |
| 1 | 0.8224 | 0.8411 | **+0.0187** | 0.0087 | 0.0281 | 0.5643 |
| 2 | 0.8411 | 0.8131 | **−0.0280** | 0.0122 | 0.0374 | 0.4839 |
| **3-seed mean** | **0.8380** | **0.8287** | **−0.0093** | 0.0111 | — | — |

> **`B_fid` = 0.0093 = exactly one dev item** (1/107 = 0.00935). The raw effect under test is
> **+0.0255**, i.e. **2.7× the band. The §2.1 STOP RULE is NOT triggered and the measurement is
> admissible.**

**The caveat, stated plainly.** The *per-seed* deviation reaches **0.0280**, which **exceeds** the
+0.0255 effect. So a single-seed head-space read could not settle this question; the 3-seed mean can,
which is why §2.9 froze the primary read as a 3-seed mean before any of these numbers existed. **No
per-seed head-space number in this record carries a verdict.** The per-seed deviations are also of
mixed sign (−, +, −), i.e. the proxy is unbiased-looking at this n rather than systematically low; the
3-seed mean −0.0093 is one item and is consistent with `ERRPAT`'s independently measured test-side
band of +0.0000 (val-selected) / **−0.0031** (final-epoch) at `ERRPAT_HateMM_2026-07-26.md:42-44`.

### 4.2 ARENA-1 / ARENA-2 — **both PASS; and the head arena is emphatically NOT saturated**

**ARENA-1:** the fold assignment was asserted item-for-item against
`scripts/analysis/vsw_ckpt/hatemm/f{0..4}.npz::ho_idx` inside **every one of the 18 mint units**
(5 assertions each, **90/90**). Fitting pools 595/595/595/595/596, held-out 149/149/149/149/148.

**ARENA-2:** pooled head-space deployed accuracy over the 744 held-out train items:

| | head seed 0 | head seed 1 | head seed 2 | **3-seed mean** | **RAW arena (F105, banked)** |
|---|---|---|---|---|---|
| pooled acc | 0.8884 | 0.8858 | 0.8858 | **0.8867** | **0.8441** |
| pooled mF1 | 0.8838 | 0.8811 | 0.8812 | 0.8820 | 0.8419 |
| per-fold acc | 0.8725 0.8993 0.8926 0.8725 0.9054 | 0.8591 0.8859 0.9060 0.8658 0.9122 | 0.8591 0.8859 0.9060 0.8658 0.9122 | | 0.7987 0.8322 0.8926 0.8255 0.8716 |

Bar [0.6195, 0.98]: **PASS on 3/3 seeds.** This is the load-bearing methodological result of the
record: **a head trained on 4/5 of the train split and queried with the held-out fifth gives an
unsaturated head-space arena** — 0.8867, not the 0.998 of full-train LOO that
`mechnov_pairverify.py:21-25` and F47 correctly ruled unusable. **The head-space arena the campaign
believed did not exist, exists, and costs ~40 s of CPU per fold-head.**

It is also **+0.0426 better than the raw arena on the identical items** — the trained head is doing
real work, which is precisely why an operator tuned to repair the raw arena's mistakes has less to do
there.

*(Seeds 1 and 2 land on identical per-fold accuracies. This is a coincidence, not a duplicate run: the
two mints' head keys differ (max abs 0.0264-0.0397) and their deployed predictions disagree on 1-4
items per fold, agreement 0.9732-1.0000; the errors swap rather than cancel. Checked before reporting.)*

### 4.3 PARITY-λ0 — **54/54 PASS, bit-exact**

18 gates per seed (3 families × 5 folds `np.array_equal` on both the vote and the prediction vector,
+ 3 pooled-accuracy-at-4dp), **18/18 on each of seeds 0, 1, 2**, executed by F105's own frozen
`vsw_pregate.parity_lambda0`. Every Δ below is therefore a paired Δ against a floor the harness
reproduces exactly, in the same session, on the same head.

### 4.4 K-HST-1 — **THE PRIMARY READ. The +0.0255 does not survive the trip.**

| | RAW arena (F105, banked) | **HEAD arena, seed 0** | **seed 1** | **seed 2** | **3-seed mean** |
|---|---|---|---|---|---|
| deployed floor | 0.8441 | 0.8884 | 0.8858 | 0.8858 | 0.8867 |
| **`VSW_pow` Δacc** | **+0.0255** | **+0.0027** | **−0.0027** | **+0.0027** | **+0.0009** |
| ΔmF1 | +0.0242 | +0.0028 | −0.0026 | +0.0026 | +0.0009 |
| fold signs | `+++++` | `00+00` | `0−000` | `000+0` | 5/5 fold means ≥ 0 |
| fixed / broken | **36 / 17** | 4 / 2 | 1 / 3 | 3 / 1 | 2.7 / 2.0 |
| **changed decisions** | **53** | **6** | **4** | **4** | **4.7** |
| pos-rate (bank 0.4005) | 0.4368 | 0.4099 | 0.4045 | 0.4018 | — |
| λ\* per fold | 3, 2, 3, 2, 3 | 1, 8, 8, 8, 8 | 96, **0**, 192, 192, 6 | 12, 12, 12, 12, 0.25 |  |

> **Transfer ratio = +0.0009 / +0.0255 = 0.035.** A **28-fold** shrink. The point estimate is
> **+0.7 items on 744**, against **+19 items** in the raw arena. Per-seed the arm moves **+2 / −2 / +2**
> items — the sign is not stable across head seeds.
>
> **Verdict by the frozen §2.4 rule: PARTIAL** (because +0.0009 > 0). **But +0.0009 is one tenth of the
> instrument band `B_fid` = 0.0093, so by the §2.1 rule it CANNOT CARRY A VERDICT.** The bankable
> statement is the bound: **the head-space effect is smaller than one dev item of instrument error, and
> the largest transfer consistent with this measurement is `+0.0009 + B_fid` ≈ **+0.0102**, i.e. a
> transfer ratio of at most ~0.40 and a point estimate of 0.035.** The honest one-line reading is
> **"no measurable transfer"**, and the record says that rather than "PARTIAL".

**The ceiling closes it independently of the instrument band.** `ORACLE_lambda_pow` — λ chosen on the
**held-out fold itself**, a cheating selector reported only as a ceiling — gives **+0.0072** (3-seed
mean; +0.0094 / +0.0054 / +0.0067) in head space against **+0.0349** in the raw arena. **Even a
hindsight-optimal λ cannot reach the raw arena's honest +0.0255 in head space**, and the ceiling of
the entire λ family is 4.2× under the +0.030 bar. That statement does not depend on `B_fid` at all,
because the oracle and the arm are computed on the same head in the same session.

### 4.5 K-HST-2 — **the membership diagnostic FIRES. This is the mechanism.**

Same query, same fold, same bank, same item indices; only the key space differs.

| quantity | seed 0 | seed 1 | seed 2 | **3-seed mean** |
|---|---|---|---|---|
| **mean size of raw ∩ head top-20** | 2.03 | 2.10 | 1.98 | **2.04 of 20 (10.2 %)** |
| median overlap | 2 | 2 | 2 | 2 |
| **queries sharing ZERO neighbours** | 19.6 % | 19.1 % | 21.1 % | **19.9 %** |
| queries sharing ≥ 10 neighbours | 0.40 % | 0.27 % | 0.27 % | **0.31 %** |
| mean \|Δ(number of positive-label neighbours)\| | 1.57 | 1.63 | 1.63 | 1.61 |
| raw-vs-head deployed decision agreement | 0.8965 | 0.8965 | 0.8965 | 0.8965 |

Bar (§2.5): mean overlap **2.04 < 10** ⇒ **SUBSTANTIAL MEMBERSHIP CHANGE.**

> **The two arenas are not two views of the same neighbourhood — they are different neighbourhoods.**
> One query in five shares **no** neighbour at all between raw and head, and the typical query shares
> two. Since the deployed vote reads **only the retrieved label tuple** — 99.6-100 % decision identity
> with a label-only vote in the raw arena (`LITSWEEP8_PATHOLOGY_MATCH.md` §2 Result A) and **1.0000**
> in the deployed head space (`HEADCOV_PREGATE_RECORD.md` §4.2, K-HC-3) — **an operator that re-weights
> the retrieved list is, in the two arenas, operating on two different label tuples.** Non-transfer is
> therefore mechanically explained by the change of retrieved set and is **not** evidence about VSW as
> an operator. **The frozen §2.5 reading rule applies as written.**

### 4.6 K-HST-3 — **the verifier's advantage over the cosine INVERTS SIGN in head space**

F95 control-1, recomputed in head space on the **full** held-out × in-fold pair matrix
(88 208-88 655 pairs per fold, 5 folds × 3 seeds).

| | RAW fused (F95, banked) | **HEAD space (this record, 3-seed × 5-fold mean)** |
|---|---|---|
| pair-AUC, verifier, held-out pairs | — | **0.8317** |
| pair-AUC, cosine, held-out pairs | — | **0.8960** |
| **`d_AUC` = verifier − cosine, held-out** | **+0.1572** (5/5 fold signs, 18/18 cells) | **−0.0643** (per-seed −0.0657 / −0.0663 / −0.0609; **15/15 fold cells negative**) |
| pair-AUC, verifier, **in-sample fitting pairs** | — | **0.9999** |
| pair-AUC, cosine, in-sample fitting pairs | — | 0.9264 |

> **In the deployed head space the trained pair verifier is WORSE than the plain cosine at telling
> same-class pairs from cross-class pairs — by −0.064 AUC, on 15 of 15 fold cells, having achieved
> 0.9999 on the pairs it was fitted on.** The entire premise of VSW — *"the verifier supplies
> second-order information the cosine does not"* (`VSW_PREGATE_RECORD.md` §1, quoting F95 control 1's
> +0.1572 / +0.2302 / +0.1785) — **is true in raw space and false in head space.** VSW has nothing to
> spend there, and the §2.6 reading rule fires.
>
> **This is exactly the failure the frozen module predicted and never measured.**
> `mechnov_pairverify.py:22-24` justified the raw arena on the grounds that *"a verifier fitted in head
> space would be measuring memorisation"*. **Measured: in-sample pair AUC 0.9999, held-out 0.8317.**
> The prediction was right — and the consequence, which the module did not draw, is that the raw arena
> is not a conservative stand-in for head space but a **systematically more favourable** one for any
> operator built on a fitted relation score.

### 4.7 WHY — four collapse diagnostics, all newly computed, all pointing the same way

| quantity | **RAW arena** | **HEAD arena** | ratio |
|---|---|---|---|
| median within-query cosine spread, rank 1 − rank 20 | **0.02532** | **1.59e-04** (s0) / 1.23e-04 (s1) | **~160-206×** collapse |
| min / max top-20 cosine | 0.000000 / 1.000000 | 0.935091 / 1.000000 (s0) | — |
| PCA(256) explained variance on the fitting pool | 0.9459 (5-fold mean 0.9450) | **1.0000 (5/5 folds)** | head keys are effectively ≤ 256-rank |
| mean top-20 neighbourhood purity toward gold | 0.7474 | **0.8553 / 0.8519 / 0.8555** | +0.108 |
| median \|deployed vote\| | 0.6587 | **0.9523 / 0.9760 / 0.9904** | — |
| **fraction of items with \|vote\| < 0.10 (the re-weightable population)** | **5.78 % (43 items)** | **0.27 % / 1.08 % / 1.34 %** (2 / 8 / 10 items) | **4-21× smaller** |
| coverage(20) — ∃ a correct-label neighbour | 0.9933 | 0.9798 / 0.9772 / 0.9758 | — |

The head-space spread **1.59e-04** independently reproduces `HEADCOV_PREGATE_RECORD.md` §4.4's
**1.95e-04**, measured on a different dataset (MHC-ZH), a different split (dev) and a different
re-mint. **Two independent instruments, same order of magnitude.**

> **The arithmetic of the non-transfer, in one line.** A monotone re-weighting of the top-20 can only
> flip a decision whose vote is near zero. In the raw arena **43 of 744** items are within 0.10 of the
> boundary; in head space **2-10 of 744** are. Multiply that by a verifier that is now *worse* than the
> cosine (§4.6), and an operator that fixed 36 and broke 17 in raw fixes 3 and breaks 2 in head.
> **Nothing about VSW changed. The arena did.**

### 4.8 Degeneracy controls — **DEG-A and DEG-B FIRE in head space and did NOT fire in raw**

| control | **RAW arena (F105, banked)** | **HEAD, s0** | **s1** | **s2** | fires? |
|---|---|---|---|---|---|
| DEG-A, agreement with the fitted threshold twin | 0.9220 — no fire | **0.9758** | **0.9839** | **0.9691** | **FIRES 3/3** |
| DEG-B, max agreement with a fixed-k profile | 0.9328 at **k=15** — no fire | **0.9919** at k=20 | **0.9946** at k=20 | **0.9946** at k=20 | **FIRES 3/3** |
| DEG-D, agreement with the cosine twin | 0.9382 | 0.9879 | 0.9879 | 0.9946 | reported |
| agreement with the deployed rule itself | 0.9288 | 0.9919 | 0.9946 | 0.9946 | — |
| class balance, deviation from bank pos-rate 0.4005 | 0.0363 | 0.0094 | 0.0040 | 0.0013 | **PASS 3/3** |

`FIXK_20` **is** the deployed rule, so DEG-B firing at k = 20 is F105 §3.6.4's pre-declared verdict
form: *"an arm that falls back to the deployed rule IS a member of the closed family"*. In head space
`VSW_pow` agrees with the deployed rule on **99.2-99.5 %** of items. **The arm has collapsed into the
floor.** The class-balance control passes, so no head-space number here is a collapse artefact.

### 4.9 K-VSW-2 in head space — the λ curve is flat at zero across a 16 384× range

3-seed means, fixed λ, no selection (a property of the operator, not of a selector). RAW comparator
re-read from `vsw_main_hatemm_OUT.json`.

| λ | **HEAD Δacc** | HEAD changed | HEAD net | RAW Δacc | RAW changed | RAW net |
|---|---|---|---|---|---|---|
| 0 | 0.0000 | 0 | 0 | 0.0000 | 0 | 0 |
| 0.25 | 0.0000 | 2.7 | 0 | **+0.0202** | 21 | +15 |
| 1 | +0.0014 | 3.7 | +1.0 | +0.0215 | 42 | +16 |
| 3 | +0.0014 | 3.7 | +1.0 | **+0.0282** (raw max) | 55 | +21 |
| 8 | **+0.0022** (head max) | 4.3 | +1.7 | +0.0255 | 63 | +19 |
| 16 | +0.0022 | 5.7 | +1.7 | +0.0255 | 67 | +19 |
| 64 | +0.0009 | 7.3 | +0.7 | +0.0215 | 68 | +16 |
| 256 | +0.0009 | 7.3 | +0.7 | +0.0175 | 67 | +13 |
| 1024 | 0.0000 | 8.0 | 0 | +0.0148 | 71 | +11 |
| 4096 | −0.0018 | 9.3 | −1.3 | +0.0148 | 77 | +11 |
| **∞** (single best-verified neighbour) | **−0.0067** | 15.7 | −5.0 | +0.0148 | 79 | +11 |

> **Over the entire aggregation-sharpness continuum the head-space operator never changes more than
> 15.7 of 744 decisions and never nets more than +1.7 items**, against 21-79 changed and +11 to +21 net
> in raw. The `λ = ∞` endpoint — *"emit the label of the single best-verified neighbour"* — is
> **negative** in head space (−0.0067) and positive in raw (+0.0148). **F105's K-VSW-2 outcome (b) —
> the aggregation axis is closed across the sharpness continuum — holds a fortiori in head space, and
> holds there by a much wider margin.** F105's law `net = changed × (2·precision − 1)` is confirmed:
> head-space precision on changed items sits at ~0.57 at λ = 8 (3.0 fixed of 4.3+1.3 changed) and falls
> to ~0.34 at λ = ∞ (5.3 of 15.7), so net goes negative while `changed` still grows.

### 4.10 The transfer ladder (§2.10) — four operator families, one arena swap

3-seed means, head arena. RAW column re-read from `vsw_main_hatemm_OUT.json` (F105/F94/F95 all share
that emitter). **Secondary — cannot carry a bar.**

| rung | arm | **RAW Δacc** | **HEAD Δacc (3-seed)** | HEAD per-seed | HEAD changed |
|---|---|---|---|---|---|
| **1** | **`VSW_pow`** (F105 PRIMARY) | **+0.0255** | **+0.0009** | +0.0027 / −0.0027 / +0.0027 | 6 / 4 / 4 |
| 1 | `VSW_exp` | +0.0255 | +0.0013 | +0.0040 / −0.0027 / +0.0027 | 7 / 4 / 4 |
| 1 | `VSW_lin` | +0.0188 | **+0.0000** | 0 / 0 / 0 | **0 / 0 / 0** (λ\* = 0 every fold) |
| 1 | `ORACLE_lambda_pow` (hindsight ceiling) | +0.0349 | **+0.0072** | +0.0094 / +0.0054 / +0.0067 | 11 / 4 / 9 |
| 1 | `CTRL_cos_pow` (DEG-D, no verifier) | +0.0067 | −0.0009 | −0.0040 / +0.0013 / 0.0000 | 3 / 7 / 0 |
| **2** | `F95_mlp_max` (MECHNOV PRIMARY) | −0.0040 | **−0.0784** | −0.0753 / −0.0645 / −0.0954 | 124 / 114 / 117 |
| 2 | `F95_mlp_mean3` | +0.0054 | **−0.0641** | −0.0605 / −0.0618 / −0.0699 | 111 / 106 / 102 |
| 2 | `F95_cos_shape` (control 2b) | −0.0417 | −0.0322 | −0.0403 / −0.0255 / −0.0309 | 56 / 53 / 51 |
| **3** | `FIXK_1` = `FIXK_2` = `FIXK_3` | −0.0430 | −0.0332 | −0.0417 / −0.0255 / −0.0323 | 57 / 53 / 52 |
| 3 | `FIXK_5` | −0.0054 | −0.0040 | −0.0067 / 0.0000 / −0.0054 | 15 / 22 / 14 |
| 3 | `FIXK_10` | +0.0027 | −0.0031 | −0.0094 / +0.0027 / −0.0027 | 9 / 4 / 4 |
| 3 | `FIXK_15` | +0.0040 | −0.0004 | −0.0013 / +0.0013 / −0.0013 | 1 / 3 / 3 |
| 3 | `FIXK_20` (= deployed) | 0.0000 | 0.0000 | 0 / 0 / 0 | 0 |
| — | `THRESH_best` (global recalibration) | +0.0148 | +0.0041 | +0.0081 / +0.0081 / −0.0040 | 24 / 12 / 27 |
| **4** | `F89_T1` class-balanced | — | +0.0000 | −0.0013 / +0.0013 / 0.0000 | 1 / 1 / 0 |
| 4 | `F89_T2a` CSLS | — | −0.0004 | −0.0013 / 0.0000 / 0.0000 | 1 / 0 / 0 |
| 4 | `F89_T2b` whitening | — | +0.0027 | +0.0040 / +0.0027 / +0.0013 | 23 / 22 / 31 |
| 4 | `F89_T4` whiten + class-balanced | — | +0.0054 | +0.0013 / +0.0040 / +0.0108 | 25 / 27 / 28 |

**Three readings, in order of how much they carry.**

1. **Every raw-space *positive* shrinks toward zero or inverts; every raw-space *kill* survives, and
   most get worse.** `FIXK_1/2/3` −0.0430 → −0.0332, `cos_shape` −0.0417 → −0.0322, and F95's
   nominate-and-verify arms go from ~0 in raw (−0.0040 / +0.0054) to **catastrophic** in head space
   (−0.0784 / −0.0641, i.e. **−58 / −48 items on 744**). **The raw arena is optimistic in exactly one
   direction.** A screen that over-reports effects cannot manufacture a false kill; it can only
   manufacture a false lead.
2. **F95's MECHNOV verdict is not merely confirmed in head space — it is confirmed by an order of
   magnitude.** The one cell that cleared F95's own +0.010 bar under the same-session anchor
   (`VSW_PREGATE_RECORD.md` §4.3: HateMM × MLP × mean-top-3, **+0.0107**) reads **−0.0641** in head
   space. The environment-conditioned "0 of 36" count debate (`PREGATE_DETERMINISM_CLAUSE.md` §3) is
   **moot in the space the system deploys in**.
3. **Rung 4 is an independent validation of the fold-head arena itself.** `MECHFIX_PREGATE_2026-07-27.md`
   §4.1 measured the same four operators in the deployed head space on the **test** split with proxy
   heads: HateMM T1 **+0.0000**, T2a **+0.0000**, T2b **−0.0078**, T4 **−0.0046**. This record's
   head-space *train-fold* arena reads **+0.0000 / −0.0004 / +0.0027 / +0.0054**. **T1 and T2a are
   inert in both; T2b and T4 are sub-0.008 and sign-unstable in both. The verdicts agree cell for
   cell**, which is what a usable screen looks like — and note this is a **head→head** comparison,
   which is the comparison that agrees, while the **raw→head** comparisons above are the ones that do
   not.

**Correction to the tasking's premise, carried from §2.10:** F89/MECHFIX was **not** a raw-arena
verdict. `mechfix_ops.py:16-17` names the trained head's fused embedding as its key space and
`MECHFIX_PREGATE_2026-07-27.md` §4.1 reports test-split 3-seed numbers on proxy heads. Its T4 = +0.0067
on MHC-ZH is already a held-out head-space number and has no raw counterpart to transfer.

---

## §5. SECONDARY — MHC-ZH (the §3.7 pre-run amendment). **Every arena property replicates.**

n = 579 train, bank pos-rate 0.3109, encoder `Qwen2.5-VL-7B-Instruct-LoRA_HF`, floor job 13150,
18 mint units at 25.1-36.1 s each (504.7 s for all 18), 3 arena runs. **ZH carries no K-HST-1 verdict** — F105's raw ZH
`VSW_pow` is **−0.0017** (`vsw_main_zh_OUT.json`, re-read), so there is no positive to transfer. (The
`transfer_ratio` field in `headspace_report_zh_OUT.json` divides by the **HateMM** raw effect, is
meaningless for ZH, and **must not be quoted**; it is left in place rather than edited, and flagged
here.)

### 5.1 GATE-FID (ZH) — PASS, and the instrument reproduces an INDEPENDENT re-mint exactly

| seed | floor dev acc @29 (job 13150) | proxy dev acc @29 | Δ | mean \|Δ\| over 30 epochs |
|---|---|---|---|---|
| 0 | 0.8462 | 0.8462 | **0.0000** | 0.0235 |
| 1 | 0.8590 | 0.8333 | **−0.0257** | 0.0201 |
| 2 | 0.8462 | 0.8462 | **0.0000** | 0.0214 |
| **3-seed mean** | **0.8505** | **0.8419** | **−0.0086** | 0.0217 |

`B_fid`(ZH) = **0.0086** = 0.67 dev items (n = 78, one item = 0.0128). STOP rule not triggered.

> **Cross-instrument check, unplanned and strong.** `HEADCOV_PREGATE_RECORD.md` §4.2 reports the ZH dev
> deployed accuracy of the **ERRPAT** re-mint heads (`errpat_zh_remint_v2`, a different script, a
> different group, a different agent) at the final epoch as **0.8462 / 0.8333 / 0.8462**. This record's
> independent re-mint reads **0.8462 / 0.8333 / 0.8462** — **exact agreement at 4 dp on 3 of 3 seeds.**
> Two independent implementations of the same CPU re-mint recipe produce the same head.

### 5.2 ARENA-2, PARITY-λ0 (ZH) — both PASS

| | head s0 | head s1 | head s2 | 3-seed mean | **RAW arena (F105)** |
|---|---|---|---|---|---|
| pooled deployed acc | 0.8929 | 0.8895 | 0.8946 | **0.8923** | **0.8480** |

Bar [0.7091, 0.98]: **PASS 3/3.** The head buys **+0.0443** over raw on the identical items — within
0.002 of the HateMM figure (+0.0426). **PARITY-λ0 54/54.**

### 5.3 K-HST-2 / K-HST-3 (ZH) — both replicate, and K-HST-3 replicates **harder**

| quantity | **HateMM** | **MHC-ZH** |
|---|---|---|
| mean raw ∩ head top-20 overlap | **2.04 / 20** | **2.85 / 20** |
| queries sharing ZERO neighbours | 19.9 % | 12.7 % |
| queries sharing ≥ 10 | 0.31 % | 4.55 % |
| raw-vs-head deployed decision agreement | 0.8965 | 0.8820 |
| **`d_AUC` (verifier − cosine), held-out pairs, HEAD** | **−0.0643** | **−0.1294** |
| `d_AUC`, RAW (F95 banked) | **+0.1572** | **+0.2302** |
| verifier in-sample fitting-pair AUC, HEAD | 0.9999 | 0.9999 |
| cosine in-sample fitting-pair AUC, HEAD | 0.9264 | 0.9173 |

> **On both datasets the verifier's advantage over the cosine inverts sign between the two arenas, and
> the inversion is larger where the raw advantage was larger** (ZH: +0.2302 → −0.1294, a swing of
> 0.3596; HateMM: +0.1572 → −0.0643, a swing of 0.2215). **The raw arena does not merely shrink this
> quantity — it reverses it, on 2 of 2 datasets, on 15 of 15 fold cells each.**

### 5.4 Collapse diagnostics (ZH) — same picture, same magnitudes

| quantity | RAW | HEAD s0 / s1 / s2 |
|---|---|---|
| median within-query cosine spread (rank 1 − rank 20) | **0.02213** | **1.52e-04 / 1.43e-04 / 1.86e-04** (≈ 130× collapse) |
| PCA(256) explained variance | 0.9627 (5/5 folds) | **1.0000 (5/5 folds)** |
| top-20 neighbourhood purity | 0.7175 | 0.8546 / 0.8566 / 0.8604 |
| median \|deployed vote\| | 0.5116 | 0.9523 / 0.9617 / 0.9694 |
| **fraction of items with \|vote\| < 0.10** | **7.94 % (46 items)** | **0.35 % / 0.86 % / 1.21 %** (2 / 5 / 7 items) |
| coverage(20) | 1.0000 | 0.9862 / 0.9914 / 0.9862 |

The head-space spread on ZH (1.43-1.86e-04) again lands on `HEADCOV_PREGATE_RECORD.md` §4.4's
independently measured **1.95e-04**, this time on the same dataset. The re-weightable population
collapses **7-23×**.

### 5.5 ZH ladder and degeneracy (secondary, descriptive)

| arm | RAW Δacc | HEAD Δacc (3-seed) | HEAD per-seed | HEAD changed |
|---|---|---|---|---|
| `VSW_pow` | −0.0017 | **−0.0046** | 0.0000 / −0.0086 / −0.0052 | 0 / 5 / 3 |
| `VSW_exp` | −0.0138 | −0.0017 | 0 / 0 / −0.0052 | 0 / 0 / 3 |
| `VSW_lin` | 0.0000 | 0.0000 | 0 / 0 / 0 | 0 |
| `ORACLE_lambda_pow` (ceiling) | +0.0121 | **+0.0035** | +0.0017 / +0.0052 / +0.0035 | 1 / 3 / 4 |
| `THRESH_best` | −0.0052 | +0.0023 | −0.0035 / +0.0086 / +0.0017 | 16 / 11 / 11 |
| `F95_mlp_max` | −0.0466 | **−0.1738** | −0.1865 / −0.1865 / −0.1485 | 166 / 168 / 148 |
| `F95_mlp_mean3` | −0.0345 | −0.1457 | −0.1399 / −0.1710 / −0.1261 | 149 / 165 / 133 |
| `F95_cos_shape` | −0.0293 | −0.0363 | −0.0449 / −0.0415 / −0.0225 | 50 / 42 / 37 |
| `FIXK_1` | −0.0294 | −0.0369 | | 50 / 42 / 38 |
| `F89_T4` whiten + class-balanced | — | −0.0063 | −0.0086 / −0.0052 / −0.0052 | 27 / 19 / 33 |

DEG-A fires 3/3 (0.9724 / 0.9724 / 0.9758; raw 0.9516 — also fires); DEG-B fires 3/3
(1.0000 @k=15 / 0.9914 @k=15 / 0.9948 @k=20; raw 0.9706 @k=20 — also fires); class balance PASS 3/3.
**On ZH, `VSW_pow` agrees with the deployed rule on 0.9914-1.0000 of items — on seed 0 it IS the
deployed rule** (λ\* = 0 on every fold, 0 decisions changed).

**MECHFIX cross-check (ZH):** `MECHFIX_PREGATE_2026-07-27.md` §4.1's head-space **test** reads are
T1 +0.0000, T2a +0.0000, T2b +0.0000, T4 **+0.0067**; this record's head-space train-fold arena reads
−0.0006 / +0.0000 / −0.0040 / **−0.0063**. T1 and T2a are inert in both; T2b/T4 are sub-0.007 and
**sign-unstable across the two head-space arenas**, which is itself worth recording: the *best number
in the MECHFIX battery* (T4 ZH +0.0067) does not reproduce its sign on a second head-space arena, and
was already reported by its own record as *"inside the ±0.014 seed band"* and *"not a lead"*.

---

## §6. PERMUTATION NULL — the declared 30-draw budget, run per §2.8

Δ was **+0.0009 > 0**, so the §2.8 budget ran (30 draws, seed 0, PRIMARY family, `PERM_SEED = 12345`,
F105's own scheme: fitting-fold **item** labels permuted, bank labels / retrieval / cosines / deployed
floor / gold labels untouched). Δ did **not** clear the +0.010 escalation threshold, so the 200-draw
escalation was **not** run — as declared.

| | **RAW arena (F105, banked, 200 draws)** | **HEAD arena (this record, 30 draws, seed 0)** |
|---|---|---|
| observed Δacc | **+0.0255** | **+0.0027** |
| null mean ± sd | −0.0005 ± 0.0053 | −0.0008 ± 0.0021 |
| **null maximum** | **+0.0134** | **+0.0040** |
| draws ≥ observed | 0 of 200 | **2 of 30** |
| fraction of draws ≥ 0 (null reaches the fallback) | 0.5700 | **0.5333** |
| **p** | **0.0050** | **0.0968** |

> **In the raw arena the observed effect was 1.9× the largest of 200 null draws. In head space the
> observed effect is BELOW the largest of 30 null draws and 2 draws beat it.** The null retains the
> property F105 required of it and F98 lacked — the fallback is reachable, 53 % of draws land at or
> above zero — so `p = 0.0968` is an honest non-significance, not an artefact of an unreachable floor.
> **Resolution is 1/31 = 0.0323, so the smallest p this design can produce is 0.0323; the measured
> 0.0968 is 3× that and is not resolution-limited.**

---

## §7. VERDICT TABLE

| gate | bar | measured | verdict |
|---|---|---|---|
| **GATE-FID** (HateMM) | `B_fid` < +0.0255 | **0.0093** (1 dev item); per-seed max 0.0280 | **PASS on the declared 3-seed statistic**, per-seed band disclosed as wider than the effect |
| **GATE-FID** (ZH) | same | **0.0086**; reproduces the ERRPAT re-mint at 4 dp on 3/3 seeds | **PASS** |
| **ARENA-1** | fold assignment identical to the banked raw arena | **90/90 assertions** (HateMM) + 90/90 (ZH) | **PASS (hard assert)** |
| **ARENA-2** | pooled head deployed acc ∈ [maj+0.02, 0.98] | HateMM **0.8867**, ZH **0.8923** | **PASS 6/6 heads — an unsaturated head-space arena exists** |
| **PARITY-λ0** | λ=0 bit-exact vs `mechfix_ops.deployed_vote` | **54/54** (HateMM) + **54/54** (ZH) | **PASS (exact)** |
| **K-HST-1** | ≥ +0.0128 = TRANSFERS | **+0.0009** (3-seed mean), ratio **0.035**, p = 0.0968, inside `B_fid` | **PARTIAL by the letter — reported as NO MEASURABLE TRANSFER**, cannot carry a positive |
| **K-HST-2** | mean overlap < 10/20 ⇒ membership explains non-transfer | **2.04/20** (HateMM), **2.85/20** (ZH) | **FIRES 2/2 datasets** |
| **K-HST-3** | `d_AUC` ≤ 0 ⇒ verifier uninformative in head space | **−0.0643** (HateMM), **−0.1294** (ZH); in-sample 0.9999 | **FIRES 2/2 datasets, 30/30 fold cells, SIGN-INVERTED vs raw** |
| **DEG-A** | ≥ 0.95 ⇒ fires | 0.9691-0.9839 (HateMM), 0.9724-0.9758 (ZH) | **FIRES 6/6** (raw HateMM 0.9220 did not) |
| **DEG-B** | ≥ 0.95 ⇒ fires | 0.9919-0.9946 @ k=20 (HateMM), 0.9914-1.0000 (ZH) | **FIRES 6/6** (raw HateMM 0.9328 did not) |
| **CLASS BALANCE** | within 0.10 of bank rate | 0.0013-0.0104 | **PASS 6/6** — no number here is a collapse artefact |

**Bottom line.** *The campaign's best raw-arena result does not survive the trip to the deployed head
space.* +0.0255 → **+0.0009** (28× shrink, p = 0.0968, below the null maximum, inside the instrument
band, sign unstable across head seeds). Even the **hindsight-optimal λ** reaches only +0.0072 there.
The reason is measured, not inferred: the two arenas share **2 of 20** retrieved neighbours per query,
the re-weightable population is **4-21× smaller** in head space, and the trained pair verifier —
0.9999 AUC on the pairs it was fitted on — is **worse than the plain cosine** on held-out pairs, by
−0.064 / −0.129, on 30 of 30 fold cells.

---

## §8. THE DIRECT ANSWER: is the raw-space pregate arena trustworthy for screening?

### 8.1 The answer, in the form the evidence supports

> **The raw arena is trustworthy for KILLS and untrustworthy for LEADS. Its error is one-sided.**

Measured, this session, on 2 datasets × 3 head seeds, over 21 operator cells (§4.10, §5.5):

* **Every raw-space POSITIVE shrank toward zero or inverted:** `VSW_pow` +0.0255 → **+0.0009**;
  `VSW_exp` +0.0255 → +0.0013; `VSW_lin` +0.0188 → **0.0000**; `THRESH_best` +0.0148 → +0.0041;
  `CTRL_cos_pow` +0.0067 → −0.0009; `FIXK_15` +0.0040 → −0.0004; `FIXK_10` +0.0027 → −0.0031;
  F95 `mlp_mean3` +0.0054 (recorded) / +0.0107 (same-session anchor) → **−0.0641**; the λ-oracle
  ceiling +0.0349 → +0.0072. **9 of 9 raw positives fail to transfer; the median shrink is >7×; three
  invert sign.**
* **Every raw-space KILL survived, and most deepened:** `FIXK_1/2/3` −0.0430 → −0.0332,
  `cos_shape` −0.0417 → −0.0322, F95 `mlp_max` −0.0040 → **−0.0784** (HateMM) and −0.0466 →
  **−0.1738** (ZH). **0 of 8 kills reversed.**
* **The mechanism is structural, not incidental,** and it has two independent legs, both measured:
  **(i)** the head compresses the decision margin — median |vote| 0.51-0.66 → 0.95-0.99, and the
  population an operator could possibly move (|vote| < 0.10) collapses from 43-46 items to 2-10;
  **(ii)** any *fitted relation score* over head keys memorises the bank (in-sample pair AUC **0.9999**)
  and generalises **worse than the plain cosine** (`d_AUC` −0.064 / −0.129), inverting the raw arena's
  +0.157 / +0.230. Both legs make the raw arena systematically **more permissive** than head space.

**A screen that over-reports effect sizes cannot manufacture a false kill.** So the practical reading
is that **every raw-space closure of the last three days is MORE secure than it was, not less** — and
the campaign's raw-arena leads were the part that needed checking. There was exactly one, and it is
now checked.

**Where the raw arena is NOT one-sidedly conservative, stated against interest.** The one-sidedness is
about *magnitude*, not about *rank*. Two raw-space negatives got **much worse** in head space
(F95 `mlp_max`, `mlp_mean3`), which means the raw arena also mis-estimates how bad a bad operator is.
Nothing in this record establishes that an operator which is *negative* in raw could not be *positive*
in head space — that possibility is untested, it is the same untested direction the F95 module's own
limitation L1 named (`VSW_PREGATE_RECORD.md` §3.1: *"a raw-space null does not logically entail a
head-space null"*), and **this record does not close it.** What it does establish is that the raw
arena's *positives* do not survive, which is the direction the campaign was actually exposed on.

### 8.2 Re-open candidates, ranked by how much their conclusion would change

| # | candidate | what changes | what does NOT change | recommendation |
|---|---|---|---|---|
| **1** | **F105/VSW's `+0.0255` quoted as a NEAR-MISS** (*"1.2× under the +0.030 bar"*, *"permutation-validated p = 0.0050"*, *"the largest honest raw-arena effect the campaign has produced"*) | **Large.** In the deployed head space the same operator on the same items gives **+0.0009**, p = **0.0968**, below the null maximum, ceiling +0.0072. The near-miss framing is a **property of the raw arena, not of the method.** | **F105's VERDICT (KILL) is unchanged and is now over-determined.** K-VSW-1 was missed in raw and is missed by 33× in head space. K-VSW-2's outcome (b) — the aggregation axis closed across the sharpness continuum — holds a fortiori (§4.9). | **RE-SCOPE, do not re-open.** Wherever +0.0255 is quoted, add *"raw train-LOO arena; head-space transfer measured at +0.0009 (F113)"*. No GPU, no test touch, no new run. |
| **2** | **F95's control-1 datum** *"the verifier carries ordering information the cosine does not: +0.1572 / +0.2302 / +0.1785, 5/5 fold signs, 18/18 cells"* — cited in `LITSWEEP6_RELGEN`, F97 and F105 as the reason the relational asset is real | **Large in scope, zero in verdict.** The quantity **inverts sign** in the deployed space on **2 of 2** datasets and **30 of 30** fold cells (−0.0643 / −0.1294), with the verifier at 0.9999 in-sample. The claim *"trained pair relations beat the cosine"* is **true in the raw encoder space and false in the deployed space.** | F95's KILL, F97's K-VGA-3 (a within-session **relative** comparison), and the *"analysis-grade only"* routing of the relational asset all stand — indeed K-VGA-3's conclusion is strengthened, since the verifier features get worse where the deployed system lives. | **RE-SCOPE.** Any paper sentence asserting the relation score is informative must say *"in the raw encoder key space"*. `$0`. |
| **3** | **F89/MECHFIX T4 on MHC-ZH, +0.0067**, described by its own record as *"the best number in the battery"* | **Moderate.** A second, independent head-space arena (train-fold, 3 seeds) gives **−0.0063** — same magnitude, opposite sign. | Its own record already ruled it *"sub-bar positive, not a lead"*, *"inside the ±0.014 seed band"*, *"not 3/3 on mF1"*. The MECHFIX verdict (0 of 15 cells clears) is unchanged. | **ANNOTATE.** Add the second-arena sign disagreement as evidence that the +0.0067 is seed/arena noise. `$0`. |
| **4** | **F94/KSWEEP's residual reading that k=10-15 is marginally better than k=20** (+0.0027 / +0.0040 raw) | **Small.** Head space: −0.0031 / −0.0004. The residual dies. | F94's kill of the k axis is unchanged and now measured in the deployed space too. | **ANNOTATE.** `$0`. |
| **5** | **F96/RESTRANS, F97/VGA, F98/AGGNET, F112/MEMBANK-C4** | **None identified.** All four rest on a kill, on a degeneracy control firing, or on a within-session relative comparison. F98's decisive bar was missed by >2×; the shrink direction measured here can only widen that. | everything | **DO NOT RE-OPEN.** |
| **6** | **F108/STREAMCOMP's raw-arena stream deltas** (ZH +0.0156, EN +0.0310) | **Supported, not overturned.** This record is a third and fourth matched pair in the same direction and supplies the *mechanism* F108 could only observe. | F108's conclusion (the instrument is unvalidated) — now upgraded from "unvalidated" to "characterised, one-sided". | **CITE, no action.** |

**Nothing in this record justifies spending a test touch, and none is recommended.** Every question it
raises is answerable at `$0` on train + dev.

### 8.3 Is one cell enough to generalise? **No — and here is exactly what is and is not established.**

* **Established on 2 datasets × 3 head seeds (6 independently minted heads, 30 fold cells):** the head
  arena is unsaturated and buildable at ~35-40 s/head; the raw and head arenas retrieve **different
  neighbourhoods** (2.0-2.9 of 20 shared); the fitted relation score **inverts** its advantage over the
  cosine; the re-weightable population collapses 4-23×; the degeneracy controls that did **not** fire
  in raw **do** fire in head space. These are arena properties and they replicate cleanly.
* **Established on ONE cell only:** the failure of a raw-space **positive** to transfer. That is not a
  sampling choice — **F105/VSW is the only raw-space positive the campaign ever produced.** There is
  no second one to test. So "one cell" is the population, not a sample of it.
* **NOT established:** that a raw-space **negative** cannot be a head-space positive (§8.1); that any
  of this transfers to the **test** split (all arenas here query train-split items held out from their
  own head, which is closer to deployment than raw but is still not deployment); that the CPU-minted
  proxy head equals the CUDA floor to better than **±0.0093** (3-seed) / **±0.0280** (single seed).

**The next cheapest cell, ranked.**
1. **MHC-EN head-space arena** — completes the 3-dataset replication of K-HST-2/K-HST-3 and ARENA-2.
   Cost: 18 mints × ~33 s + 3 arenas ≈ **15 CPU-minutes, `$0`, zero test touch.** Recommended if the
   arena characterisation is to be quoted as general.
2. **Switch the screening arena.** The genuinely valuable follow-up is not another measurement but a
   change of practice: **every future `$0` pregate should be run in the fold-head arena, which now
   costs ~35-40 s of CPU per fold-head and is a strictly better proxy for deployment than the raw
   arena.** The whole `mechfix_ops` / `vsw_pregate` battery already runs in it unmodified — this record
   is the proof. Marginal cost per new operator after the arena exists: **seconds.**
3. **A dev-query head-space arena** (deployed-configuration head, real dev queries, n = 107/78) as a
   third rung between the fold arena and test. The six deployed-configuration heads minted for GATE-FID
   already contain the keys; only the driver is missing. Cost: **~10 CPU-minutes, `$0`.** Value:
   it is the only remaining zero-test-touch arena that queries with *non-train* items.

---

## §9. LIMITATIONS — stated against the record's own conclusion

1. **The head is a CPU-minted proxy, not the deployed head.** The deployed checkpoints are gone
   (§1.3). The proxy is measured, not assumed: `B_fid` = 0.0093 (HateMM) / 0.0086 (ZH) on the 3-seed
   dev anchor, and **0.0280 on a single seed** — wider than the effect. Every conclusion here is stated
   on 3-seed means for that reason, and **no per-seed head-space number carries a verdict**. The
   headline conclusion is not band-limited: it rests on the λ-oracle ceiling (+0.0072, same head, same
   session, no cross-instrument comparison), on the null (p = 0.0968, same head), and on `d_AUC` and
   membership, which are same-session paired quantities.
2. **The fold heads are trained on 4/5 of the train split, so they are not the deployed head object.**
   That is inherent: a head that has seen an item cannot be queried with it. The raw arena has the
   identical structure, so the *comparison* is matched; the *absolute* head-space numbers are those of
   a 595-item-trained head, not a 744-item-trained one.
3. **The queries are train-split items, not test items.** This arena is strictly closer to deployment
   than raw — it uses the deployed key space and the deployed retrieval — but it is not deployment.
   Everything here is a **screen**, and the record claims only that it is a better screen.
4. **`VSW_lin` never leaves λ = 0 in head space** (0 changed decisions, all seeds, both datasets), so
   its "transfer" is a degenerate 0.0000 rather than a measured shrink. Reported as such.
5. **The permutation null is 30 draws on one seed**, resolution 1/31. It was the declared budget
   (§2.8) and the escalation condition was not met. p = 0.0968 is not resolution-limited (3× the
   floor), but it is a 30-draw p and is quoted as such.
6. **DET-4 applies.** The pair verifier is a 30-epoch Adam MLP, the estimator
   `PREGATE_DETERMINISM_CLAUSE.md` §1.3 measured as the one that amplifies a thread-count perturbation
   above 4 dp. Every run here honoured **DET-1** (`OMP/MKL/OPENBLAS/NUMEXPR_NUM_THREADS = 8` exported
   by the driver before process start, asserted in-process) and **DET-2** (full `runtime` block in
   every output), so the Tier-B parity requirement is met; but the record's own margins are compared
   against the clause's reference band (`auc_mlp` range 0.0014, pooled accuracy excursion ≤ 0.0067)
   and **`VSW_pow`'s head-space +0.0009 sits well inside it.** That is a further, independent reason
   the +0.0009 is reported as *no measurable transfer* rather than as a small positive. The quantities
   the verdict actually rests on — the membership overlap, the deployed floors, the cosine spreads, the
   PCA variance, `FIXK_*`, `cos_shape` — are **closed-form** and Tier-A.
7. **The MHC-ZH `transfer_ratio` field** in `headspace_report_zh_OUT.json` divides by the HateMM raw
   effect and is meaningless; it is flagged in §5 and must not be quoted. Left un-edited so the JSON
   matches the script that produced it.
8. **No claim is made about MHC-EN**, which was not run.

---

## §10. FILE MANIFEST AND COMPLIANCE

**Written by this record (all new):**

| path | role |
|---|---|
| `scripts/analysis/headspace_mint.py` | mints deployed-recipe heads per (seed, fold); extracts the 1024-d head key space; `torch.load` test guard; DET-1 assert; DET-2 block |
| `scripts/analysis/headspace_arena.py` | the per-fold-X emitter + the frozen F105 battery + K-HST-2/K-HST-3 + the F89/F95 ladder |
| `scripts/analysis/headspace_fidelity.py` | GATE-FID; dev-only trainlog reader |
| `scripts/analysis/headspace_report.py` | reporting only |
| `scripts/analysis/headspace_drive.sh` | orchestration + the DET-1 export; changes no arm |
| `scripts/analysis/headspace_arena_{hatemm,zh}_s{0,1,2}_OUT.json` (+ `.log`) | 6 arena outputs |
| `scripts/analysis/headspace_perm_hatemm_s0_OUT.json` | the 30-draw null |
| `scripts/analysis/headspace_fidelity{,_zh}_OUT.json`, `headspace_report{,_zh}_OUT.json` | gates + merged tables |

**Read-only inputs:** `data/CLIP_Embedding/{HateMM,MHC_zh}/{train,dev_seen}_*.pt`;
`scripts/analysis/{mechfix_ops,mechnov_pairverify,vsw_pregate}.py` (sha-asserted at run time);
`scripts/analysis/vsw_ckpt/{hatemm,zh}/f{0..4}.npz`; `scripts/analysis/vsw_main_{hatemm,zh}_OUT.json`;
`scripts/analysis/vsw_perm_hatemm_OUT.json`; the **`Val_Retrieval` lines only** of
`slurm/logs/enc3s_HateMM_…-LoRA-curric_HF_seed{0,1,2}_13241.trainlog` and
`slurm/logs/enc3s_MHC_zh_…-LoRA_HF_seed{0,1,2}_13150.trainlog`; `src/run_rac.py`,
`src/model/{classifier,evaluate_rac}.py`, `src/data_loader/{dataset,rac_dataloader}.py`,
`src/utils/metrics.py` (read for the recipe audit of §1.4).
**Read for context:** `VSW_PREGATE_RECORD.md`, `HEADCOV_PREGATE_RECORD.md`,
`STREAMCOMP_FORENSIC_RECON.md` §5.3, `MECHFIX_PREGATE_2026-07-27.md`, `ERRPAT_HateMM_2026-07-26.md`,
`PREGATE_DETERMINISM_CLAUSE.md`, `AGGNET_PREGATE_RECORD.md`, `LITSWEEP6_RELGEN.md`.
**Scratch (not committed):** the 36 mint `.npz` and 6 arena checkpoint dirs live under the session
scratchpad; they are regenerable in ~30 CPU-minutes from the committed scripts.

**Required statements.** **ZERO GPU, ZERO SLURM, ZERO Modal, ZERO test-split contact, ZERO test-touch
budget consumed.** No `test_seen` cache, no test-split artifact and no test label was opened, read or
computed; the `torch.load` guard and the `Val_Retrieval`-only reader are the code-level enforcement.
No frozen script was modified — `mechfix_ops.py`, `mechnov_pairverify.py` and `vsw_pregate.py` were
imported unmodified with their sha256 asserted at run time on every invocation. No banked verdict was
rewritten, no `state/` prereg, config, `research-wiki/` page or frozen artefact was mutated, no
checkpoint was loaded, and the **1 361** `state_dict` writes the 36 mint units
would otherwise have produced (~46 GB) were suppressed rather than written. **This is a pregate and a diagnostic. It promotes nothing.**
