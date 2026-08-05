# Review Request — Gate-0 Reopen 2026-07-31

**You are a fresh independent reviewer.** You have had no exposure to the
adjudicator's reasoning. Do not assume any statement below is true; verify it.

## What you are reviewing

`refine-logs/GATE0_REOPEN_2026-07-31.md` — the adjudication record — together with
how it landed in:

- `TARGET_STATE.json::gate0_reopen_2026_07_31` (new block) and the ten
  `candidate_registry` status changes it made (C05–C14)
- the final bullet of `TARGET_FINDINGS.md`
- the final section of `TARGET_LOOP.md` (`## Gate-0 reopen — 2026-07-31`)

The advisory input being adjudicated is
`refine-logs/C05PLUS_FORENSIC_RECON_2026-07-31.md`. It is **advisory only** and
creates no registry status by itself.

## Review scope — read this carefully, it is narrow

You are **NOT** reviewing the science of any candidate. You are not asked whether
C09 will work, whether C07 deserves to die, or whether the campaign's strategy is
right. **Do not** produce findings of the form "C09's mechanism is unlikely to
succeed" — that is out of scope.

You **ARE** reviewing exactly three things:

1. **Are the strikes faithful to the evidence?** For every candidate recorded as
   `struck_gate0_2026_07_31`, is the stated basis actually supported by the
   primary source it cites, quoted accurately, and applied within that source's
   own written scope? A strike that needs a ban stretched wider than its text is a
   defect.
2. **Are the statuses correctly scoped?** Is each disposition (`struck`, `held_*`,
   `gated_on_*`, `next_active_candidate_post_C04`) the right *kind* of record for
   the evidence behind it? In particular: is anything recorded as a registry-level
   strike that is really a measured kill, or vice versa? Is the reversibility
   language present and correct?
3. **Is anything recorded as measured that wasn't?** Every number in the record
   should trace to a primary source or to a stated re-measurement. Flag any figure
   presented as measured that is actually an inference, a prediction, or a
   transcription from a secondary document.

Also in scope, because it is the record's own headline claim: the record asserts
that it *downgraded* three of the recon's seven recommended strikes to HOLD
(C10, C11, C12) because they needed text read past its scope. **Verify those
downgrades are justified** — if any downgrade is itself wrong (i.e. the ban really
does cover the candidate and the strike should have stood), that is a defect of
equal weight to a stretched strike.

## How to verify

Read primary sources, not the record's summaries of them. Useful locations:

- `autoresearch/goal_mllm_plus3/state/directions_tried.json` — `banned_constraints`
  (11 entries) and `dead[].ban_scope` (76 entries; the relevant ones include the
  entries for F47, F55, F60, F66, F70, F78, F80, F82, F98, F99, F100/EUM,
  F101/BSY, TVB)
- `autoresearch/goal_mllm_plus3/state/findings.jsonl` — F55, F60, F70, F75, F78,
  F80, F88, F98, F99, F106, F107, F112, F113, F114
- `autoresearch/goal_mllm_plus3/state/progress.json`
- `refine-logs/ERRPAT_{HateMM,MHC-EN,MHC-ZH}_2026-07-26.md`
- `refine-logs/LITSWEEP*.md`, `refine-logs/HEADCOV_PREGATE_RECORD.md`,
  `refine-logs/HEADSPACE_TRANSFER_PREGATE.md`, `refine-logs/GRADEDLBL_PREGATE_RECORD.md`,
  `refine-logs/NCA_FORENSIC_RECON.md`
- `artifacts/c01_policy_contrastive/v4/a0/C01-A0-v4/C01_A0_OUT.json`
- `research-wiki/TARGET_GATE0_ITER6_LITERATURE.md`
- `data/gt/{HateMM,MHC_zh,MHC}/{train,val}.jsonl` — you may recompute any census
  claim from these label files

## Hard rules binding you

- **Read-only.** Zero GPU, zero SLURM, zero Modal, no model loads, no `.pt` opens.
- **No test-split access.** Do not open any `test.jsonl` or any `*test*` split file.
- **Write nothing.** Do not edit `TARGET_STATE.json`, `TARGET_LOOP.md`,
  `TARGET_FINDINGS.md`, `TARGET_REVIEW_RAW.md` or any record under review.
- Do not touch anything in the **C04** lineage (`artifacts/c04/`, `configs/c04/`,
  `refine-logs/C04_*`) — a separate worker holds it. Do not modify anything in C02.

## Verdict format required

Return `GO` or `REVISE` plus counts in the exact form `<n> Critical / <n> High /
<n> Important`, then the findings themselves, each with a `file:line` citation and
a concrete required repair. `GO (0C/0H/0I)` means: every strike is faithful and
in-scope, every status is the correct kind of record, every downgrade is
justified, and nothing is recorded as measured that is not.

Be adversarial. A rubber-stamp is worse than useless here — this record moves ten
registry statuses.
