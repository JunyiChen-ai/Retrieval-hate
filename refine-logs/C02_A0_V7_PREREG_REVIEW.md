# C02 A0 v7 — scoped re-check (round 7)

**Reviewer:** the round-4/5/6 reviewer. This is the **scoped re-check I specified at the end
of `refine-logs/C02_A0_V6_PREREG_REVIEW.md` §4** — the eight hashes, the three strings, a
constants sweep, and a stale-identifier scan. I did not re-open anything else, and §3 below
says exactly what that means.
**Date:** 2026-07-30 (Pacific/Auckland)
**Type:** read-only static review. Nothing was executed. See §5.

**Verdict:** `GO (0C/0H/0I)`

---

## 0. Hashes, namespaces, stale identifiers

| path | declared | recomputed | verdict |
|---|---|---|---|
| `configs/c02/c02_a0_v7.json` | `4fac6050…98a94ed1` | `4fac60501de74e8975d3bca0209837ce416c15bdeff00cbcdb3fdd1898a94ed1` | **MATCH** |
| `src/utils/c02_density_views.py` | `6b2107b7…8a5a0740` | `6b2107b7a3a899492e68e735fe1e49c97de8c6214c6c3fa6440dfa268a5a0740` | **MATCH** |
| `src/utils/generate_c02_density_view_text_embedding_HF.py` | `0223e885…b3abce47` | `0223e885cbada6eed5866258c004376cbda966a2f3a0490d30ec7857b3abce47` | **MATCH** |
| `scripts/slurm/c02_density_extract.sbatch` | `a2e12b9b…511e9c27` | `a2e12b9b5370f96a4fd531ef0ffd538ff9859f497ff06f63eae84747511e9c27` | **MATCH** |
| `scripts/analysis/c02_a0_mint.py` | `3696addc…a3833484` | `3696addc260f137f5100761562072f6d93bc00912bc7e744e863f948a3833484` | **MATCH** |
| `scripts/analysis/c02_a0_arena_v7.py` | `1548a7e3…9e16258fd9` | `1548a7e330b3c05557cf86ebea6bbf60368a5ec2383950f066798a9e16258fd9` | **MATCH** |
| `scripts/slurm/c02_a0_cpu_v7.sbatch` | `592bad52…f659ca81` | `592bad52e2b6ee68a45fd1f54de00f7000fd8491fe7123205913e498f659ca81` | **MATCH** |
| `refine-logs/C02_A0_V7_RECORD.md` | `a3b4f144…97797e4` | `a3b4f1440e70e5997a2abc0aeed9c07cdac106c0064ea030fa351a94497797e4` | **MATCH** |

**v6 executables ABSENT** (`ls` → *No such file or directory*): `configs/c02/c02_a0_v6.json`,
`scripts/analysis/c02_a0_arena_v6.py`, `scripts/slurm/c02_a0_cpu_v6.sbatch`. The three
directories hold exactly one C02 A0 config, one arena and one wrapper.
**`artifacts/c02_edq` does not exist**; `find . -name '*c02den*'` returns nothing.

**The start-up trap, checked for the fourth time and clear:**
`generate_c02_density_view_text_embedding_HF.py:64` pins
`FROZEN_VIEWS_SHA256 = 6b2107b7…8a5a0740`, which **equals** the v7 view module's hash;
`:63` still equals the deployed extractor's hash. The four wrapper-pinned frozen modules
were re-hashed **now**, against a tree that other agents write concurrently, and all four
still match `c02_a0_cpu_v7.sbatch:76-79`: `headspace_fidelity.py 72fd8e0a…`,
`mechfix_ops.py 635c1312…`, `mechnov_pairverify.py 77b0defd…`,
`headspace_mint.py cefdf8dc2f4a…`. These also agree with `arena_v7.py:80-87` and
`c02_a0_mint.py:56`.

**Stale identifiers: none.** A `grep` for `v6`/`V6` across the whole v7 set returns hits
only inside the config's `supersedes` audit block (`:332-341`), which is where the v6
identity is supposed to be preserved. Every record pointer in all seven files reads
`C02_A0_V7_RECORD.md` (`views:4`, `extractor:4`, `mint:5`, `arena:4`,
`extract.sbatch:14,24`, `cpu_v7.sbatch:12`, `config:11`), and `run_id`, `a0_namespace`,
both schema versions, `--job-name`, `RUN_ID`, `CFG` and both arena references in the
wrapper (`:59` import, `:113` invocation) are consistently `v7`.

**Stale "merge" language: none.** Every surviving occurrence is correct: the deliberate
contrast "DROPPED, not merged" (`arena:250,258`, `config:153`, self-test assertion message
`arena:628`), the record's description of what F1/F2 fixed (`RECORD:39-40`), and PEFT's
unrelated `model.merge_and_unload()` in the extractor (`:183`). The two v6 sites I flagged
— the halt comment and the halt message — no longer contain the word.

**Pointer-only files** are byte-identical in size to their v5/v6 versions (view module
11 067 B / 268 lines; extractor 13 830 B / 316 lines; mint 10 558 B / 250 lines; extraction
wrapper 2 457 B / 64 lines), consistent with equal-length `V6`→`V7` substitution plus, in
the extractor, the equal-length hash re-pin.

---

## 1. The three strings

### F1 — **CLOSED.**

`configs/c02/c02_a0_v7.json:196` now reads "*…that the degeneracy-matched grouping never
straddles that boundary and **DROPS a singleton class group rather than merging it, and
that the dropped item leaves the donor pool entirely***". That is exactly what
`arena_v7.py:625-630` asserts — `ndrop1 == 1`, `sum(g.size) == 119`, and `3 not in` any
returned group — and it is now consistent with the same config's `arms.SHUFFLE` at `:153`.
The internal contradiction is gone, and the string also correctly names
`c02_a0_arena_v7.py`.

### F2 — **CLOSED.**

`arena_v7.py:303-309`. The comment now reads "*Unreachable by construction: shuffle_groups
emits only groups of size >= 2 and Sattolo is a derangement for every such group*", and the
halt message is now "*derangement left a fixed point: a donor group violated
`shuffle_groups`' size >= 2 rule*". No merge rule is named anywhere. Two further checks:
the unreachability claim is **true** — `out` receives a group only under `g.size >= 2`
(`:268-270`), `covered` is exactly the concatenation of `out` (`:302`), and Sattolo leaves
no fixed point in any group of size ≥ 2 — and the new message is *meaningful* if it ever
fires, since it now names the invariant that would have been violated rather than a deleted
mechanism. The string concatenation across `:308-309` is syntactically fine (the apostrophe
in `shuffle_groups'` sits inside a double-quoted literal). This is the only executable
change in v7, and it is a message string inside an existing `halt(…)` call.

### F3 — **CLOSED, and the restatement is CORRECT, not merely different.**

The correction appears on all four surfaces, in consistent terms: the arena comment
(`:257-267`), the arena's emitted `shuffle_control.grouping` string (`:934-943`), the config
(`arms.SHUFFLE`, `:153`) and the record (`:36`). I checked each clause against the code
rather than against the prose:

1. *"A class group of exactly ONE member is DROPPED, not merged"* — true of
   `shuffle_groups:268-272`.
2. *"that item then keeps its OWN displacement in SHUFFLE"* — true: it never enters `out`,
   so `perm[i] = i` (`:284`) and `build_arms` gives
   `z_i^v = NAT_i + (view_v(i) − NAT_i) = view_v(i)`.
3. **"a lone DEGENERATE item, whose displacement is ZERO by construction"** — this is the
   load-bearing claim you asked me to test against `build_arms` as written, and it is
   **true, bitwise, in both key spaces**, for both members of `degen_mask`:
   * `degen_text` items: the extractor computes one vector per **distinct view string** and
     copies it into the other slots (`generate_c02_density_view_text_embedding_HF.py:244-254`),
     so for an item whose six view strings are all `T` the six cached rows are the *same
     tensor*, bit-identical. The mint then forwards `(img_i, text_i)` through one head per
     view with identical inputs, so `keys[v][i] == keys["NAT"][i]` exactly, and
     `keys[v].astype("float64")[i] − nat_sh[i]` is exactly `0.0`.
   * `degen_zero` items: the zero-guard writes `zero.clone()` into all six slots
     (`extractor:236-238`), and `arena_v7.py:735` asserts every view's zero mask equals the
     banked one — identical inputs again, so the displacement is exactly `0.0`.
   * Consequently `shuffled[v][i] = NAT_i + 0 = NAT_i`, while `FULL`'s row for that item is
     `keys[v][i] = NAT_i`. **Bitwise identical**, so "EXACTLY MATCHED" is literal, not
     approximate. The same holds in the secondary raw space, where
     `raw_keys[v] = l2n(concat(l2n(img), l2n(view_raw[v])))` and `view_raw[v][i]` is
     identical across `v` by the same two mechanisms.
4. *"merging would instead have handed that degenerate item a real displacement it never has
   under FULL and made FULL > SHUFFLE EASIER"* — true, and it is the correct statement of
   where the conservative property actually lives.
5. *"(A lone NON-degenerate item would leave one unshuffled real displacement in the
   control … which can only make the conjunct harder.)"* — true, and correctly separated as
   the other sub-case; it needs a partition of exactly one item and is unreachable at
   `n ≥ 579`.

The record's F3 row (`:36`) states the same thing and, to its credit, says plainly that the
v6 reasoning "is wrong for the case the rule exists to handle" and that "the code is
unchanged; only its description was wrong". That is an accurate account of both the defect
and its scope.

---

## 2. Constants sweep — clean

Config ↔ code, every scalar the design gates on:

| constant | code | config | ✓ |
|---|---|---|---|
| `TOPK` / rank weights | `arena:90`, `M._rank_weights(20)` | `:171` `topk: 20`, `:172-191` `[20…1]` | ✓ |
| accuracy / macro-F1 bar | `BAR_ACC = BAR_MF1 = 0.050` `:93-94` | `decision_rule.PASS :239` | ✓ |
| net-fix bar | `BAR_NETFIX_RATE = 0.030` `:95` | `:239` + `net_fix_clause` | ✓ |
| view support | `VIEW_SUPPORT_MIN = 0.60` `:96` | `gates.VIEW_SUPPORT :206` | ✓ |
| bootstrap | `B = 10000`, seed `20260730` `:97-98` | `statistics.bootstrap :212` | ✓ |
| seeds | `SHUFFLE/NOISE/BOOTSTRAP = 20260730` `:98-100` | `seeds_frozen :223-225` | ✓ |
| head seeds | `SEEDS = (0,1,2)` `:92` | `arena.seeds :107-110` | ✓ |
| alpha / Holm family | `ALPHA = 0.05` `:101`, `"{ds}_acc"/"{ds}_mf1"` | `:214-220` (4 names) | ✓ |
| ARENA-2 band | `0.02` / `0.98` `:102-103` | `gates.ARENA2 :202` | ✓ |
| GATE-EXT | `0.99` `:104` | `gates.GATE_EXT :204` | ✓ |
| tiny norm | `1e-12` `:105` | `gates.ZERO_CONTRACT :205` | ✓ |
| KRR | `ridge 1.0`, `gamma = 1/d` `:106` | `mechanism_diagnostics` | ✓ |
| arms | 9 `ARM_NAMES` `:108-110` | `arms` keys `:117-155` | ✓ |
| views | `VIEW_NAMES`, `K_WINDOWS 4`, `SEP " "`, `L_MAX 12000` (`views:68-72`) | `:17-24, 26, 28, 33` | ✓ |
| halt exit code | `SystemExit(3)` `:1177` | `output.halt_exit_code :278` | ✓ |
| resources | `#SBATCH` 8 CPU / 32 G / no `--gres`; extraction 8 / 1 A100 / 64 G | `execution :249-258`, `det1_threads 8 :263` | ✓ |
| namespaces / schema | `c02_a0_result_v7` `:1076`, `c02_a0_decision_v7` `:1109` | `:271-275` | ✓ |

Every v5/v6 hardening is still in place in the v7 arena: banked-cache tiny-row check
(`:724`), manifest sha comparison (`:793`), `degen_mask = degen_text | degen_zero`
(`:812`), parity-cell count assertion (`:953`), the secondary-arena `try/except Halt`
(`:1030`), and `SystemExit(3)` (`:1177`). The retracted H1 wording returns **zero** matches
in both the config and the arena, and the H3 retraction remains the only occurrence of
"EXCHANGEABLE BY CONSTRUCTION". No threshold, bar, arm, seed, gate, metric or decision-rule
term moved between v6 and v7 — consistent with the record's claim that the only executable
change is the F2 message string.

---

## 3. New findings — none, and what `0I` does and does not mean

I found **no new defect** in the v7 diff. The diff is what it claims to be: three prose
surfaces, one halt-message string, one hash re-pin, and record-pointer updates.

`GO (0C/0H/0I)` means: **within the scope I set for this re-check, nothing is open, and
nothing I raised across rounds 4-6 is unaddressed.** It does not mean the artifact set has
no properties worth knowing at submission time. Three things I recorded earlier as
**ACCEPTED-AS-DECLARED** are still true, are not defects, cannot be closed by code, and are
now declared in the frozen artifacts themselves rather than only in review files:

* the frozen set is not git-tracked, so the sha256 chain is the whole audit mechanism
  (`RECORD §2`, and an iteration-8-wide condition I spot-verified);
* the 4.0 GPU-hour cap is a post-hoc `sacct` measurement under amendment condition (f), not
  an in-job interlock (`config:264`);
* amendment condition (e) is a manual `squeue` check recorded in `RECORD §5`
  (`config:265`).

And two operator preconditions still apply immediately before `sbatch`, because no code can
enforce them: **run the `squeue` check**, and **re-verify the eight sha256 values** — the
tree has concurrent writers, which is exactly why I re-hashed the four frozen modules again
in this pass rather than trusting last round's result.

---

## 4. Verdict

```
GO (0C/0H/0I)
```

F1 CLOSED · F2 CLOSED · F3 CLOSED and correct. Eight hashes match, the v6 executables are
gone, both namespaces are clean, the constants sweep is clean, no stale v6 or merge
language survives, and the one executable change is a halt-message string whose new text is
both accurate and reachable-only-on-invariant-violation.

For the record, since this was the point of v7: you were right to build it. F3 was not a
wording preference — the v6 text asserted a direction of protection ("makes `FULL > SHUFFLE`
harder") that did not exist in the only case the rule can fire in, inside a hash-frozen
preregistration whose whole function is to fix the interpretation of a result before the
result exists. The v7 statement is the true one, and it is stated identically in the code,
in the string the run will emit into `C02_A0_OUT.json`, in the config and in the record.

This set is ready to run as frozen.

---

## 5. What I did and did not execute

**Did:** `sha256sum`, `ls`, `find`, `grep`, `sed`, `tr`, `cut`, `wc`, and file reads —
scoped as agreed: recomputed all 8 v7 hashes plus the 4 wrapper-pinned frozen-module hashes
and the extractor's two pins; read `arena_v7.py:244-313` and `:930-952`, the config's
`arms.SHUFFLE`, `oracle.self_test`, `gates`, `statistics`, `execution` and `output`
sections, `C02_A0_V7_RECORD.md §2`, and the wrapper's changed lines; grepped the whole v7
set for `v6`/`V6`, for `merge`, for the retracted H1/H3 phrases and for the record
pointers; and swept every gated constant config-against-code.

**Did not:** run Python of any kind, import any module, load or open any `.pt` cache,
`.npz`, model, adapter or video; open any `test_seen` cache, `test.jsonl`, or any test label
or metric; run `squeue`, `sacct`, `sbatch`, `bash -n` or any SLURM command; touch a GPU or
Modal; modify, move or delete any reviewed file; re-open the design questions settled in
rounds 4-6; or write anything other than this review.
