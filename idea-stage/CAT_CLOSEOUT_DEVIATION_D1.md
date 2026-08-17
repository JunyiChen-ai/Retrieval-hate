# CAT close-out — deviation D1

**Filed:** 2026-08-18, **before any arm metric on seeds 1300–1319 / 1400–1429 / 1500–1524
existed** and before any leg of the frozen run was launched. Freeze: `idea-stage/CAT_CLOSEOUT_FREEZE.md`
(commit `ae286c9`).

## What is wrong

Freeze §4.2 specifies the inner train/dev split of each CV cell as

```
StratifiedShuffleSplit(..., random_state = 1000·(20260818 + r) + f)
```

`1000 · 20260818 = 20_260_818_000`. `sklearn.model_selection.StratifiedShuffleSplit` routes its
`random_state` through `numpy.random.RandomState`, which requires a seed in `[0, 2**32 − 1]`
(4_294_967_295). The frozen expression exceeds that by a factor of ~4.7 and raises
`ValueError: Seed must be between 0 and 2**32 - 1` on the first cell. **The frozen design is
unimplementable as written.** Caught by a pre-run smoke of `build_cv.py`; no cell was built, no
head was trained, no metric was computed.

## The change

```
random_state = (1000·(20260818 + r) + f) mod (2**32 − 1)
```

Nothing else changes: same `StratifiedKFold` seeds `20260818 + r`, same stratification, same
inner-dev fraction `d = |dev_seen| / (|train| + |dev_seen|)`, same cell → head-seed map
`1500 + 5r + f`, same arms, same read-out protocol, same decision rule in §4.3.

## Why this cannot bias any verdict

1. The RNG seed of a stratified shuffle is an **arbitrary label**. No property of the frozen
   design depends on the particular integer; it exists only to make the partition reproducible.
   Any deterministic function of `(r, f)` fixed in advance is equivalent for every purpose the
   freeze states.
2. The replacement is a pure range-fold of the frozen expression — it is the frozen formula, taken
   modulo the library's documented domain — not a re-choice made to hunt for a better partition.
3. It is filed **before** any CV cell was built and before any head ran, so no partition was ever
   observed, let alone selected.
4. Leg C's decision rule (§4.3) is untouched, and Leg C has no gate power over Legs A, B or D.

## Standing

Freeze §4.2 is read with this substitution in every downstream document. The rest of
`CAT_CLOSEOUT_FREEZE.md` stands unchanged.
