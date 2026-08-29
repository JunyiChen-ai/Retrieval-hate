# R12 deviation D1 — demotion-clause key lookup in the two verdict scripts

Filed 2026-08-18, **after** the R12-ANCHOR numbers were read and **before** the R12-IMG analyzer ran.
This memo is written before the corrected R12-ANCHOR verdict is re-run.

## What happened

`R12_FREEZE.md` §2.5 and §3.5 both arm a REAUDIT_NCA demotion clause: an arm that is **dev-negative
with the CI excluding zero while test-positive cannot STAND**. Both `verdict.py` scripts looked the
dev contrast up under the key `"P1/<contrast>"`. `idea-stage/r10_combo/analyze_dev_panel.py`, which
is the frozen dev-panel producer and was reused unchanged, writes its keys with the prefix
`"dev_mf1_P1/"` (alongside `test_mf1_P1/`, `test_mf1_P2/` and `gap_P1/`).

The lookup therefore returned `None` on every arm, and the first R12-ANCHOR verdict recorded
`dev_demotion_fired: {MHC_zh: null, HateMM: null}` instead of a boolean.

## Why it cannot change either verdict

The demotion clause is **strictly a KILL-adder**: it can only remove a STANDS, never create one.

- **R12-ANCHOR.** Both candidates fail clause 1 outright. `AF_PT − CAT` = −0.0003 (CI
  [−0.0048, +0.0041]) on MHC-ZH and −0.0002 (CI [−0.0038, +0.0034]) on HateMM; `AF_A0 − CAT` =
  −0.0007 and −0.0003 with CIs likewise straddling zero. `verdict = KILL` is reached through the
  `not c1` branch, which does not consult the demotion result. Additionally the clause could not
  have fired even if it had been read correctly: the dev contrasts are `AF_PT − CAT` = −0.00030
  (CI [−0.0042, +0.0040], does not exclude zero) on MHC-ZH, so the "CI excluding zero" condition
  fails, and the test contrast is negative, so the "test-positive" condition fails too.
- **R12-IMG.** The defect was corrected before that pilot's analyzer ran, so its verdict is produced
  by the corrected code and the clause is live as frozen.

## What was done

One-line key correction in `idea-stage/r12_anchor/verdict.py` and `idea-stage/r12_img/verdict.py`
(`"P1/%s"` → `"dev_mf1_P1/%s"`). No bar, no clause, no arm definition, no protocol and no seed range
was changed. `idea-stage/r12_anchor/verdict.py` is re-run once on the unchanged
`{zh,hm}_grid.json` and `{zh,hm}_devpanel.json`; the grids themselves are **not** recomputed.

## Standing of the R12-ANCHOR verdict

Unchanged: **KILL** for both `AF_PT` and `AF_A0`. The re-run only replaces two `null`s with the
booleans the frozen rule asks for.
