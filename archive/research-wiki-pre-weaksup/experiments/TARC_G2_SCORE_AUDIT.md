---
title: "TARC G2 sub-gate A — independent macro-F1 re-audit"
auditor: independent reviewer (fresh, zero-context)
date: "2026-07-13"
scope: verify exp-tarc-t0.md §11.1 two numbers (all-videos 0.6137, hate-only 0.6760)
method: from-scratch scorer; scripts/analysis/score_target_pred.py NOT read or reused
verdict: BOTH CONFIRMED to 4 dp — under one load-bearing convention (see §7)
---

# TARC G2 sub-gate A — independent macro-F1 re-audit

## 1. Task
Recompute, with a from-scratch scoring implementation, the effective 3-class
{Blacks, Jews, Other-merged} macro-F1 for the Qwen2.5-VL-7B target predictions on
HateMM, and check the two values recorded in `exp-tarc-t0.md` §11.1:

- all videos: **0.6137**
- hate-only videos: **0.6760**

I wrote my own scorer before looking at any §11 result number and without reading
`scripts/analysis/score_target_pred.py`. The rules were taken only from §6-G2.

## 2. Inputs (verified live 2026-07-13)
| file | role | facts |
|---|---|---|
| `data/gt/HateMM/target_pred_qwen7b.json` | predictions | 1066 ids, fields `primary`/`raw`; no `_meta`. pred `primary` dist: 0(Blacks) 455, -1(None) 251, 1(Jews) 177, 2 99, 3 28, 4 27, 5 18, 6 6, 7 5 |
| `data/gt/HateMM/target_map.json` | GT | 1083 video keys (+`_meta`), fields `targets`/`primary`. `code_dict`: Blacks0 Jews1 Whites2 Others3 LGBTQ4 Muslims5 Sexits6 Asian7; `-1 == no target` |
| `data/gt/HateMM/HateMM_annotation.csv` | hate label | 1083 rows; `label` ∈ {Hate 431, Non Hate 652}; stem = `video_file_name` minus `.mp4` |

All 1066 pred ids are present in GT (0 missing). 17 GT videos have no prediction
(not scored). Scored population = the 1066 predicted ids.

## 3. Scoring rule as read from §6-G2
- Compare **pred `primary` vs GT `primary`**.
- Effective label set = **{Blacks(0), Jews(1), Other-merged}**, where Other-merged =
  codes 2..7 (Whites/Others/LGBTQ/Muslims/Sexits/Asian) collapsed to one class
  "given the skew" (Blacks+Jews ≈ 97% of hate targets).
- Kill number: macro-F1 over these classes **≥ 0.60** "on the effective label set".
- macro-F1 = unweighted mean of the three per-class F1.
- "hate-only" scope = CSV `label == Hate`.
- §6 does **not** spell out how a video with no GT target (`primary = -1`) enters the
  3-class scoring. Per the task instruction I therefore computed **both** conventions
  (see §7); one of them reproduces §11.1 exactly.

## 4. My implementation (independent)
`scratchpad/my_scorer.py` + `scratchpad/variant.py`. Core:
```python
def bucket(code):
    if code == 0:  return "Blacks"
    if code == 1:  return "Jews"
    if code == -1: return "None"      # no-target / model "None" / parse-fail
    return "Other"                    # codes 2..7 merged
# per class c in {Blacks,Jews,Other}:
#   tp = #(gt==c & pred==c); fp = #(gt!=c & pred==c); fn = #(gt==c & pred!=c)
#   P = tp/(tp+fp); R = tp/(tp+fn); F1 = 2PR/(P+R); macro = mean(F1 over 3 classes)
```
Two `-1` handling conventions tested:
- **EXCLUDE-GT-None** — drop videos whose GT `primary == -1` from the population
  (a "no target" GT is not a member of the effective label set, so predicting a
  target on it is not penalised). Pred `primary == -1` on a real-target video is
  still a false negative.
- **INCLUDE-GT-None** — keep GT-None videos as a 4th background category so any
  target prediction on them is a false positive.

## 5. Results

### 5.1 ALL VIDEOS
**EXCLUDE-GT-None (n = 1023):**
| class | P | R | F1 | sup |
|---|---|---|---|---|
| Blacks | 0.7728 | 0.7446 | 0.7585 | 466 |
| Jews | 0.5975 | 0.8962 | 0.7170 | 106 |
| Other | 0.6461 | 0.2550 | 0.3657 | 451 |
| **macro-F1** | | | **0.613704** | |

**INCLUDE-GT-None (n = 1066):** Blacks 0.7626/0.7446/0.7535 · Jews
0.5367/0.8962/0.6714 · Other 0.6284/0.2550/0.3628 → **macro-F1 = 0.595894**

Confusion (rows = GT, cols = pred; full 4×4, from the include run):
```
GT\pred    Blacks    Jews   Other    None
Blacks        347      18      55      46
Jews            1      95       8       2
Other         101      46     115     189
None            6      18       5      14      <- these 43 rows are the only difference
```
EXCLUDE drops the entire `GT=None` row (43 videos): it removes 6 FP from Blacks,
18 FP from Jews, 5 FP from Other. Recall is untouched (it never depends on GT-None).

### 5.2 HATE-ONLY (CSV label == Hate)
**EXCLUDE-GT-None (n = 426):**
| class | P | R | F1 | sup |
|---|---|---|---|---|
| Blacks | 0.9625 | 0.8107 | 0.8801 | 317 |
| Jews | 0.6824 | 0.8657 | 0.7632 | 67 |
| Other | 0.3226 | 0.4762 | 0.3846 | 42 |
| **macro-F1** | | | **0.675970** | |

**INCLUDE-GT-None (n = 427):** Blacks 0.9590/0.8107/0.8786 · Jews unchanged ·
Other unchanged → **macro-F1 = 0.675469**. (Only 1 hate video has GT-None, so the
convention barely moves hate-only.)

## 6. Per-digit comparison to §11.1
| quantity | §11.1 recorded | my EXCLUDE-GT-None | match |
|---|---|---|---|
| all-videos macro-F1 | **0.6137** | 0.613704 → **0.6137** | ✅ exact (4 dp) |
| hate-only macro-F1 | **0.6760** | 0.675970 → **0.6760** | ✅ exact (4 dp) |
| all Blacks P/R/F1 | 0.773 / 0.745 / 0.758 | 0.7728 / 0.7446 / 0.7585 | ✅ |
| all Jews P/R/F1 | 0.597 / 0.896 / 0.717 | 0.5975 / 0.8962 / 0.7170 | ✅ |
| all Other P/R/F1 | 0.646 / 0.255 / 0.366 | 0.6461 / 0.2550 / 0.3657 | ✅ |
| all Blacks/Jews/Other sup | 466 / 106 / 451 | 466 / 106 / 451 | ✅ |
| hate Blacks/Jews/Other F1 | 0.880 / 0.763 / 0.385 | 0.8801 / 0.7632 / 0.3846 | ✅ |
| hate Blacks/Jews/Other sup | 317 / 67 / 42 | 317 / 67 / 42 | ✅ |
| all confusion GT=Blacks row | 347/18/55/46 | 347/18/55/46 | ✅ |
| all confusion GT=Jews row | 1/95/8/2 | 1/95/8/2 | ✅ |
| pred dist (scorer, 1066) | B455 N251 J177 W99 O28 L27 M18 S6 A5 | identical | ✅ |

Every recorded digit reproduces.

## 7. The load-bearing convention (the only ambiguity)
§11.1's two numbers are reproduced **exactly** only under **EXCLUDE-GT-None**:
videos whose GT `primary == -1` (no target) are removed from the scored population,
so a target prediction on a no-target video is **not** a false positive. This is a
defensible reading of "macro-F1 … on the effective label set {Blacks,Jews,Other}"
(None ∉ the label set), and it is consistent with §11.1's own confusion table, which
lists three GT rows (Blacks/Jews/Other) against four pred columns (…/None).

The handling is **asymmetric** and this asymmetry is what makes the numbers work:
`-1` on the **GT** side → video dropped; `-1` on the **pred** side (model says "None"
for a real-target video) → still a false negative. The asymmetry is not stated in §6's
prose; it is inferred from the reproduction.

**Post-computation cross-check (production scorer).** After finishing the above I
opened `scripts/analysis/score_target_pred.py` to confirm intent. It implements exactly
EXCLUDE-GT-None and documents it: line 103 `eff_classes = [0, 1, 2]  # …(GT never -1
here)`; line 163 `continue  # no GT target community; excluded from macro-F1`; the run
line prints `(GT primary=-1 excluded, …)`. So the convention is the **deliberate,
documented** behaviour of the scorer, not an accident — it is only *under-specified in
§6's prose*, not ambiguous in implementation. The pred side uses `primary` directly, so
pred `-1` maps to a `None(pred)` bucket and remains a false negative, matching my run.

**Why it matters for the gate (≥ 0.60):**
| scope | EXCLUDE (as recorded) | INCLUDE (naive background) |
|---|---|---|
| all videos | **0.6137 — PASS** | **0.5959 — FAIL** |
| hate-only | **0.6760 — PASS** | 0.6755 — PASS |

The **all-videos** sub-gate pass is convention-dependent: it clears 0.60 by +0.0137
under EXCLUDE, but falls to 0.5959 (below the bar) if GT-None videos are counted as a
false-positive background. The **hate-only** pass is robust (only 1 GT-None hate video;
0.676 either way). Since §11.1 also reports the hate-only 0.6760, sub-gate A has at
least one population that clears 0.60 under any convention.

## 8. Verdict
- **all-videos macro-F1 = 0.6137 — CONFIRMED** (exact, EXCLUDE-GT-None convention).
- **hate-only macro-F1 = 0.6760 — CONFIRMED** (exact, and robust to the convention).
- Both effective-3-class per-class P/R/F1, supports, predicted distribution, and the
  Blacks/Jews confusion rows all reproduce.
- **One caveat to record in §11.1:** the numbers assume no-target (GT `-1`) videos are
  excluded from the population. State this convention explicitly; note that the
  all-videos figure would be 0.5959 (a gate miss) under the naive "GT-None as
  false-positive background" reading. Hate-only is unaffected.

Files: `scratchpad/my_scorer.py` (4×4 confusion + both scopes),
`scratchpad/variant.py` (EXCLUDE vs INCLUDE side-by-side).
