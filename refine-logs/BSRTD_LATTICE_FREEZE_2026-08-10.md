# B-SRTD Intervention-Lattice Build — FREEZE

**Frozen**: 2026-08-09T18:44:23Z UTC (before any cell was generated).
**Frozen artefact**: `data/Counterfactual/BSRTD/BUILD_RECORD_2026-08-10.md`
(`data/` is gitignored, so the freeze is recorded here by content hash).

```
sha256  c9028a4fa7a2348acf295204da698fc03e8e442d6b06e7dd5fe97919bdae6f55
        data/Counterfactual/BSRTD/BUILD_RECORD_2026-08-10.md
```

That file, at that hash, contains — decided **before** generation started:

1. the two intervention axes (A = target substitution, B = stance reversal);
2. the 2×2 cell definition (`orig`, `targetsub`, `stancerev`, `both`);
3. the **expected-label rule**: A is label-preserving, B is label-flipping — so
   seed label 1 → (1, 1, 0, 0) and seed label 0 → (0, 0, 1, 1);
4. seed eligibility (train+val splits only, cleaned length ≥ 40 chars, group-referent
   required) and the deterministic `sha256("BSRTD-2026-08-10|"+id)` selection order;
5. the per-(lang, split, label) quotas summing to 304 lattices, above the MVE floor of
   ≥200 train + ≥80 val;
6. the **four-criterion verification rubric** (① axis fidelity, ② fluency/naturalness,
   ③ in-lattice label consistency, ④ minimal-pair property) and the two-pass,
   blind, one-repair-round procedure.

Only §5 "Results" of the frozen file is written after the fact; §1–§4 are not revised.

**Hard red line ① (zero test-set contact)**: no script and no subagent in this build reads
`data/gt/*/test.jsonl`.

**User ruling recorded (2026-08-10)**: per-item generation and per-item verification are done by
Claude subagents *in place of* the human verification pass that `idea-stage/IDEA_REPORT.md` §7.2
named as C4/B-SRTD's revival prerequisite.
