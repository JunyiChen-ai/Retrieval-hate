# R11-UNION deviation D1 — a partial blindness slip during the analyzer smoke test

Filed **before** the frozen run and committed together with
`idea-stage/R11_UNION_FREEZE.md`. Nothing was changed in response to what was
seen; this memo exists so the reader can discount the result appropriately.

## What happened

`idea-stage/r11_union/analyze_union.py` needs a log directory containing 14 arm
names that do not exist yet. To exercise the code path before the frozen
submission I built a synthetic directory of symlinks into the **R10-COMBO** logs
(`logging/runs/r10_combo/zh/logs`, seeds 600–628), mapping the R10-COMBO arms
onto the R11 arm names. Most of that mapping is nonsense on purpose (`K1` played
`ANCA_l01`, and so on) and carries no information.

**But three of the mappings were identity** — `A0→A0`, `LL→LL`, `CAT→CAT`, with
`CATB` fed by `CAT` at seeds 601–629. The derived arms `AVG`, `WAVG`, `SEL` and
the control `ECTL` are functions of `z_CAT` and `z_LL` only. So for those four,
the smoke run produced **real numbers on the real test split**, on the R10-COMBO
seed range.

## Exactly what was seen

Only the union-accounting block was printed and read (mean over 29 seeds, MHC-ZH,
P1, test): for each arm the union-fix pool size, the retained fraction of the
union, the number of newly broken A0-correct items, and the net errors saved
against A0. Concretely, `AVG`, `SEL` and `ECTL` all showed a larger net-errors-
saved figure than `CAT`, with `ECTL` close behind `AVG`.

**Not seen:** no macro-F1, no contrast, no confidence interval, no P2, no HateMM,
and nothing at all about the anchor family (`ANCA`, `ANCL`, `LBL`), the `MC` arm,
or any dev fit. The seeds are 600–628, disjoint from the R11 judgement range
700–729 / 700–714.

## Why the frozen rule is not compromised

1. **The rule pre-dates the slip.** `verdict.py`, including the `ECTL` control
   clause for the decision-level family and the `LBL` control clause for the
   anchor family, was written and saved before the smoke run. The slip could not
   have shaped it, and in fact the one thing the slip showed — `ECTL` sitting
   close behind `AVG` — is precisely what that clause exists to catch.
2. **No definition was touched afterwards.** `AVG`, `WAVG`, `SEL`, `ECTL`,
   `ANCA`, `ANCL`, `LBL` and `MC` are frozen exactly as they were at the moment
   of the smoke run. No weight grid, bucket count, λ grid, bar or protocol was
   adjusted.
3. **The judgement runs on fresh seeds.** Seeds 700–729 / 700–714 were never
   touched. The leaked statistic is an error count on a different seed range, not
   the judgement metric.

## What it still costs

The test split (149 MHC-ZH items) is the same one the verdict will use, so this
is a genuine, if narrow, peek at test-derived statistics for three of the five
candidates. The honest discount: **the decision-level family (`AVG`, `WAVG`,
`SEL`) on MHC-ZH is one degree less blind than the anchor family.** If a
decision-level arm stands on MHC-ZH and only there, the result document must say
so and treat HateMM — untouched by the slip — as the load-bearing replication.

## Process fix, forward-looking

Analyzer smoke tests must use a log directory whose arm-to-source mapping is a
**derangement** (no identity mappings), or synthetic logs, so no derived arm can
be evaluated on its real parents.
