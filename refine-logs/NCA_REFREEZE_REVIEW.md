# NCA / soft-kNN HEAD-LOSS family — REFREEZE-1 RE-REVIEW (independent 0-context)

**Role:** independent 0-context re-freeze reviewer. CPU-only; NO SLURM/GPU, NO test-set access,
NO push, NO `state/` mutation. Verified every item below MYSELF (own harness / own `sha256sum` /
own `git show`); the fix record's assertions were re-derived, not taken on faith.
**Date:** 2026-07-25 NZST.
**Object under review:** commit `8f08e9f48c9a149410123937c04da8d5aeef124d` (current HEAD), the
REFREEZE-1 A3 mixup dropout-mode fix, against `refine-logs/NCA_PREREG.md` §4.5 (code-fix ⇒ re-freeze
clause), DEV-3, F0.2; `NCA_FREEZE.md` REFREEZE-1; `NCA_SUBMIT_RECORD.md §2.1`; `NCA_REFREEZE_FIX.md`.
**Env:** `/data/jehc223/miniconda3/envs/HateVideo/bin/python` (torch 2.6.0+cu124, faiss OK, numpy 1.26.4).
**Harness (throwaway, not in repo):**
`…/scratchpad/nca_review_harness.py`, pre-fix source `…/scratchpad/loss_prefix.py`
(= `git show 8f08e9f^:src/model/loss.py`, sha `e1244ada…` = the frozen pre-fix A).

---

## R1 — Diff confinement — **PASS**

`git show 8f08e9f --stat` touches **exactly 3 files**: `refine-logs/NCA_FREEZE.md` (+32),
`refine-logs/NCA_REFREEZE_FIX.md` (NEW, +228), `src/model/loss.py` (+18). No other file.
`git diff --numstat 8f08e9f^ 8f08e9f` over the five code/artifact files shows **only** `src/model/loss.py`
changed = `18 insertions / 0 deletions`; `run_rac.py`, `ncafam_family.sbatch`, `classifier.py`,
`retrieval.py` = **unchanged** (absent from numstat). The two loss.py hunk headers are BOTH
`@@ … def _manifold_mixup_bce(…) @@` (hunk 1 `-706,6 +706,22`; hunk 2 `-715,6 +731,8`); on-disk read of
lines 697–743 confirms both inserted blocks sit strictly INSIDE `_manifold_mixup_bce` (function body
708–743; next function `compute_aux_loss` begins line 746). No other function, no other file besides the
two refine-logs records. **Confined.**

## R2 — Fix semantics — **PASS (one non-blocking NOTE: no try/finally)**

The fix (loss.py:721–724, 734–735):
```
_mixup_dropouts   = [m for m in model.modules() if isinstance(m, nn.Dropout)]   # enumerate Dropout only
_mixup_prev_modes = [m.training for m in _mixup_dropouts]                        # snapshot exact prior flags
for _m in _mixup_dropouts: _m.train()                                           # enable train on Dropout only
… img_proj/text_proj forward … mlp/output_layer forward (the mixup forward) …
for _m, _mode in zip(_mixup_dropouts, _mixup_prev_modes): _m.train(_mode)        # restore EXACT prior mode
```
It snapshots each `nn.Dropout` submodule's `.training`, enables train on **Dropout only** (the `isinstance`
filter touches no other module type — zero running-stat side effect), and restores each module's exact prior
flag via `_m.train(_mode)` (not a blanket `.train()`/`.eval()`). Restore is placed immediately after the
last dropout-bearing forward (`model.output_layer(model.mlp(x_mix))`) and before the BCE/label math.
**Exception-safety NOTE (non-blocking):** the enable/restore pair is a plain sequential block, **not** a
`try/finally`. If one of the four submodule forwards (or the `Beta`/`randperm` draws) between enable and
restore raised, the Dropout modules would be left in train mode. Assessed as a NOTE, not a REJECT: (i) each
seed is its own `run_rac` process, and an exception on that path is a fatal training crash that terminates
the process — there is no in-process recovery path that would silently reuse a train-mode-leaked model to
produce a corrupted measurement; (ii) the idiom matches house style — the very bug being fixed
(`retrieval.py:330` `model.eval()` with no restore) is itself non-exception-safe; (iii) restore is
guaranteed on every normal, loss-producing path. Empirically confirmed restore in R3.

## R3 — Bug actually fixed — **PASS** (own CPU harness, deployed config `--batch_norm False`, `num_layers` default 3 ⇒ 6 Dropout)

Built the real `classifier_hateClipper` (image/text 512, proj 1024, map 1024, num_layers 3, align,
dropout 0.2/0.4/0.1, batch_norm False), forced `model.eval()` to simulate the post-mining leaked state,
hooked every `nn.Dropout` to record `.training` DURING the forward, and called `_manifold_mixup_bce`
(mixup_alpha 2.0) on the frozen pre-fix and post-fix modules. Verbatim:
```
[PRE-FIX ] dropout .training observed DURING forward = [False, False, False, False, False, False]  all-train=False
[PRE-FIX ] model.training pre=False post=False restored=True ; dropout modes restored=True
[POST-FIX] dropout .training observed DURING forward = [True, True, True, True, True, True]        all-train=True
[POST-FIX] model.training pre=False post=False restored=True ; dropout modes restored=True
```
(a) with the fix, the mixup forward now sees **all six** Dropout submodules `training=True` (bug reproduced
pre-fix: all False); (b) after the call, `model.training` and every Dropout `.training` are restored to the
pre-call `eval` state. **R3 VERDICT: PASS.**

## R4 — Floor bit-exactness — **PASS**

Own harness ran the FULL floor path `compute_loss(mixup=False, head_loss='triplet', hybrid_loss=True)`
INCLUDING the eval-mode-leaking FAISS mining (`--Faiss_GPU False`, `--metric cos`,
`no_pseudo_gold_positives 1`, synthetic 200-row bank supplied so mining is deterministic), same fixed
`torch.manual_seed`, on the pre-fix vs post-fix modules. All returned values bit-identical:
```
total_loss       bit-exact=True  pre=0.8857722878456116 post=0.8857722878456116
in_batch         bit-exact=True  pre=0.9789290428161621 post=0.9789290428161621
hard             bit-exact=True  pre=0.08353237807750702 post=0.08353237807750702
pseudo           bit-exact=True  pre=0.08432251960039139 post=0.08432251960039139
loss_classifier  bit-exact=True  pre=0.6934058666229248 post=0.6934058666229248
train_feats      bit-exact=True  shape=(200,1024) dtype=float32
train_labels     bit-exact=True  shape=(200,)     dtype=int64
total_loss hex   pre=000000203f58ec3f  post=000000203f58ec3f   (identical to last bit)
```
**A1/A2 unaffected — verified by inspection:** for `head_loss in {'nca','supcon'}` the early branch
(loss.py:43–64) RETURNS the 7-tuple **before** any FAISS mining and before the hybrid/mixup block, so
`_manifold_mixup_bce` (loss.py:585, gated `mixup=True`) is never reached; the diff is textually confined to
that function ⇒ A1 (nca) and A2 (supcon) paths are byte-unaffected. **R4 VERDICT: PASS (bit-exact).**

## R5 — Sha ledger — **PASS** (own `sha256sum` at current HEAD `8f08e9f`)

```
2ae7a73f6df4008186e5200f851e16902f567ec93f2c3681d03743c909dd0c9b  src/model/loss.py                 == REFREEZE-1 new sha A   ✓
b85eb72a690bc8fccc2ff5d5358fd6523359bf6596d2b2a0d6d0701bec9e53e3  src/run_rac.py                    == B (original frozen)    ✓
baf41be8a1264445d6bbc63d2eb54966abed6da33c52176ae62d93bf79c14b94  scripts/slurm/ncafam_family.sbatch == C (original frozen)    ✓
e7b61df485b97eb683279398746090c2d4b3d446fc4c53b5c85e14d366c23378  src/model/classifier.py           == reused-unchanged       ✓
d43e3bc417f775175021283c4bd4aa25c0df98aa4c4b34a90f8c696e195bcf57  src/utils/retrieval.py            == reused-unchanged       ✓
```
loss.py = the REFREEZE-1 sha A; run_rac.py / sbatch = the ORIGINAL frozen B/C byte-for-byte; classifier &
retrieval byte-identical. All match `NCA_FREEZE.md` REFREEZE-1. **Ledger consistent.**

## R6 — Prereg consistency — **PASS**

- **A3 is now a clean "floor + mixup" ablation:** the mixup BCE forward runs with dropout ON (R3),
  matching the floor's train-mode BCE forward (loss.py:32 → dropout ON). The dropout-off confound of
  `NCA_SUBMIT_RECORD.md §2.1` is eliminated; the measured A3 delta = mixup interpolation vs floor, no
  longer conflated with dropout-disabling.
- **DEV-3 restored:** `_manifold_mixup_bce` now reproduces "exactly the deployed align forward" (train
  mode, dropout ON, `fusion_mode=='align'` asserted, mod_dropout off) as DEV-3 §11.3 binds.
- **F0.2 covers the RNG divergence:** the fix adds dropout-RNG consumption within the A3 forward (shifting
  the subsequent `Beta`/`randperm` draws) — still **confined to the A3 arm**; the floor and A1/A2 never
  call the function (R4 bit-exact + early-return), so F0.2's pre-declared "treatment arms diverge, flag-off
  is byte-identical" holds unchanged. A3 had zero prior test-touch, so it has no trajectory to reproduce.
- **Authorization VOID stated correctly:** `NCA_FREEZE.md` REFREEZE-1 reads "AUTHORIZATION IS VOID until
  (1) an independent 0-context re-review approves this fix + re-freeze, AND (2) the mandatory codex gate
  (§4.5) re-runs clean on the patched loss.py" — correct. This review discharges (1); the codex re-gate (2)
  + smoke remain open before submission.

## R7 — No scope creep — **PASS (one non-blocking NOTE)**

Nothing in the commit changes semantics beyond the confined loss.py fix; the other two files are prose
records. I skimmed `NCA_REFREEZE_FIX.md`; its load-bearing claims — 18/0 confined diff, floor/A1/A2 never
enter the function, E1 floor bit-exact, E2 dropout ON + mode restored, E3 no running-stat module (batch_norm
False ⇒ `nn.BatchNorm1d` branch never constructed), sha A `e1244ada…`→`2ae7a73f…`, B/C intact — all match
what I measured. **NOTE (non-blocking, evidence-harness only):** the fix record's E2/E3 report "Dropout: 5 /
Linear: 5 / ReLU: 2", i.e. its equivalence harness built the head at `num_layers=2`. The DEPLOYED frozen
config omits `--num_layers` in both `ncafam_family.sbatch` and the `enc3seed_lora_curric` anchor ⇒ argparse
default **3** ⇒ the real head has **6** Dropout / 6 Linear / 3 ReLU. This does NOT affect the fix: it
enumerates dropouts dynamically (`[m for m in model.modules() if isinstance(m, nn.Dropout)]`), so it is
count-agnostic; my R3 at the true deployed `num_layers=3` (6 dropouts) confirms all six are toggled ON and
restored. The record's E2/E3 merely under-sized the head by one MLP layer; no impact on the diff, the shas,
the floor bit-exactness (E1/R4), or the fix's correctness.

---

## RULING

R1 PASS (diff = loss.py 18/0, both hunks inside `_manifold_mixup_bce`; only 2 prose records besides).
R2 PASS (snapshot→Dropout-only train→exact-prior restore; no-try/finally = non-blocking note).
R3 PASS (post-fix mixup forward sees all 6 Dropout `training=True`; model+dropout restored to pre-call eval).
R4 PASS (floor path bit-exact pre/post across all 5 scalars + bank; A1/A2 early-return before mixup, unaffected).
R5 PASS (loss.py = sha A `2ae7a73f…`; run_rac/sbatch = original B/C; classifier/retrieval unchanged).
R6 PASS (A3 clean floor+mixup ablation; DEV-3 restored; F0.2 covers A3-confined RNG; VOID clause correct).
R7 PASS (no semantic scope creep; E2/E3 num_layers=2 under-count = non-blocking evidence note).

**REFREEZE-1 APPROVED** — authorization restored pending the mandatory codex re-gate (§4.5) + smoke (§4.4).
Two non-blocking notes travel to the codex gate / executor: (a) the mixup mode-restore is not wrapped in
try/finally (harmless — an exception on that path is a fatal crash, no silent reuse); (b) the
`NCA_REFREEZE_FIX.md` E2/E3 evidence harness used `num_layers=2` (5 dropouts) vs the deployed `num_layers=3`
(6 dropouts) — the fix is count-agnostic and verified correct at the true config in R3.

**Review statements:** ZERO GPU/SLURM/Modal spent (CPU-only harness, `sha256sum`, `git show`, seconds). No
test metric read. Prereg `NCA_PREREG.md` NOT modified. `state/` and `autoresearch/…/state/` not touched. No
job submitted. Not pushed.
