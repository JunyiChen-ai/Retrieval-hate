# C02 A0 v8 — confined re-check after a self-reported erratum (round 8)

**Reviewer:** the round-4/5/6/7 reviewer. Confined to the five items in the v8 request.
The centrepiece is **§4, the measured-claims register** — the list of every empirical claim
in the frozen set that is asserted as measured, with what I could and could not verify.
**Date:** 2026-07-30 (Pacific/Auckland)
**Type:** read-only static review. Nothing was executed. See §8.

**Verdict:** `GO (0C/0H/1I)`

**On the erratum itself, before anything else.** The implementer found and self-reported a
false empirical claim that had survived seven freezes and six independent reviews,
including my own `GO (0C/0H/0I)` on v7, *before* spending any GPU time and with no external
pressure to look. That is the behaviour the project's numeric-provenance discipline exists
to produce. My §4 below is the structural answer to "how did six reviewers miss it": the
claim was an assertion standing beside a correct derivation, and nothing in the artifact
set distinguished the two. §6 proposes a fix for that.

---

## 1. Hashes, v7 absence, namespaces

| path | declared | recomputed | verdict |
|---|---|---|---|
| `configs/c02/c02_a0_v8.json` | `280c7b81…24ea1ed7` | `280c7b81e905373579283684e25a961604a5a948940047a71fe8cb3624ea1ed7` | **MATCH** |
| `src/utils/c02_density_views.py` | `b427c100…40a42d8` | `b427c100ae14584dfcbe3b5330bbaa1a7703171378cde7ec467e252b540a42d8` | **MATCH** |
| `src/utils/generate_c02_density_view_text_embedding_HF.py` | `6c5b9fd1…92ee235` | `6c5b9fd15e653ede183a1ad1d15f6f151f2f1dfc3760883d4b7c884e992ee235` | **MATCH** |
| `scripts/slurm/c02_density_extract.sbatch` | `afda59d2…835cb0d` | `afda59d243db9a6e2c1af882f2338cacaf509497e824a50b3b7fbc65c835cb0d` | **MATCH** |
| `scripts/analysis/c02_a0_mint.py` | `33404572…601e85b` | `3340457278124aae715a44980c6c8e1f5bf6ce3ea31b874c4de7bff8b601e85b` | **MATCH** |
| `scripts/analysis/c02_a0_arena_v8.py` | `7f8f491e…15b80349` | `7f8f491e5632775fe465555622c2560ae650b148059a85b91ef2d7c515b80349` | **MATCH** |
| `scripts/slurm/c02_a0_cpu_v8.sbatch` | `85576a24…5a3635aa4` | `85576a244d890ff143ea16abe78ebfa145f990daf8288d7455e45b25a3635aa4` | **MATCH** |
| `refine-logs/C02_A0_V8_RECORD.md` | `4745ff1b…7f68901d` | `4745ff1bad6eb7c1b99391afd57077deaefc2cb1d3dcea6c8802e91b7f68901d` | **MATCH** |

**v7 executables ABSENT**; `configs/c02/` holds one config, `scripts/analysis/` one arena,
`scripts/slurm/` one A0 wrapper. **`artifacts/c02_edq` does not exist**;
`find . -name '*c02den*'` returns nothing.

The start-up trap is clear for the fifth version running:
`generate_c02_density_view_text_embedding_HF.py:64` pins
`FROZEN_VIEWS_SHA256 = b427c100…40a42d8`, which equals the v8 view module's hash. The four
wrapper-pinned frozen modules were re-hashed **in this pass** and all four still match
`c02_a0_cpu_v8.sbatch:76-79`. All seven files carry the `C02_A0_V8_RECORD.md` pointer.

---

## 2. Does the false claim survive anywhere? — **No, except inside its own retraction**

`grep` over the whole v8 set for `1.5e-07`, `1.4901161`, `different code path`,
`NOT bit-equal`, `not bit-equal` and `exhaustive` returns exactly four hits, all of which
are the retraction or its audit trail:

* `configs/c02/c02_a0_v8.json:198` — the new `oracle.search_width_erratum` field, which
  states the retraction;
* `configs/c02/c02_a0_v8.json:351` — the `supersedes.v7.reason` entry recording why v8
  exists;
* `scripts/analysis/c02_a0_arena_v8.py:199-209` — the retraction inside the `orbit_vote`
  docstring.

Critically, **`config:169` (`oracle.search`) has been cleaned**: the clause "*an exhaustive
`k = n_bank` search is NOT bit-equal … measured max |delta sim| 1.5e-07*" that stood in
v1-v7 is gone, and what remains is the exactness argument plus "*For a singleton orbit it
IS the literal deployed k = 20 faiss call*". No file asserts the retracted claim.

---

## 3. Is the replacement justification true?

The exactness argument is **unchanged** (`arena_v8.py:187-197`, `config:170` and
`config:197`), still carries the tie side-condition added in v5, and still holds — I
re-derived it independently in rounds 4 and 5 and have not revisited the conclusion: if
`m_j ≥ τ` then at most 19 items are strictly above `j` in the achieving pair, so `j` is
inside that pair's own top-20, with the documented exception of ≥20 exact float32 ties at
`τ`.

The three retained reasons for `k = topk` (`arena_v8.py:206-209`):

| reason | verdict |
|---|---|
| "provably sufficient for the top-topk (above)" | **Correct.** It is the exactness argument, independently re-derived, with its side condition stated adjacent. |
| "O(topk) rather than O(n_bank) per view pair" | **Correct only under one reading — see the single Info finding below.** A `faiss.IndexFlatIP` computes all `n_bank` inner products per query whatever `k` is; `k` governs the selection heap and the size of the returned `(nq × k)` arrays, not the scan. So the claim is true of the *returned object and the selection*, and false of the *search cost*. |
| "for a singleton orbit it is LITERALLY the deployed call rather than merely equal to it" | **Correct, and it is now the load-bearing one.** `M.deployed_vote` reaches `_flat_ip(b, q, 20)` = `IndexFlatIP.search(q, 20)` (`mechfix_ops.py:45-49, 80-82`); `orbit_vote` with a singleton orbit issues the same call on operands normalised by the same `faiss.normalize_L2` on private float32 copies. The distinction it draws — *literally the deployed call*, not *provably equal to it* — is exactly the right one to lean on after an erratum about a claimed numerical equivalence. |

The retraction's own wording is accurate about the cause: the `1.4901161193847656e-07`
figure is characteristic of a float32 double-normalisation, and I independently verified the
*mechanism* it is now attributed to — `mechfix_ops._norm32` (`:37-42`) does
`np.ascontiguousarray(np.asarray(X, dtype="float32"))`, which returns the **same object**
for an already-float32 C-contiguous input, and `faiss.normalize_L2` works in place, so a
second call on a stored buffer normalises twice. That is a real defect of the frozen helper
and is precisely why the arena's local `_norm32` (`:167-178`) always copies.

---

## 4. Item 5 — the measured-claims register

Every empirical claim I can find in the v8 frozen set that is **asserted as already
measured**, enumerated whether or not I believe it. `[V]` = I verified it from the
artifacts or the frozen data in this review chain. `[D]` = documentary: I confirmed the
cited source exists and says this, but did not reproduce the underlying measurement.
`[U]` = **cannot be verified without execution — an assertion, not a derivation.**

### [V] Verified by me

1. **Identity counts, all four splits** (`config:40-77`). `n` = 744 / 107 / 579 / 78 by
   `wc -l`. Whitespace-only `text`: **39** and **9** on HateMM train/val, **0** and **0** on
   MHC-ZH train/val, by `awk` — this pass re-verified the MHC-ZH pair, which I had asserted
   in round 6 from the HateMM run only. Over-length records (`len > 12000`): **9** and
   **1** on HateMM, **0** and **0** on MHC-ZH; the count is identical at raw-length
   thresholds of 12 000, 12 052 and 12 200 bytes, so no record sits near the boundary and
   the byte/char distinction cannot move it. Hence `full_identity` 48 / 10 / 0 / 0 and
   `view_support` 0.9355 / 0.9065 / 1.0 / 1.0 all reproduce exactly.
2. **`max_chars` 80731 / 12275 / 708 / 343** (`config:49,57,65,73`), all four now measured
   by character count plus escape and scaffolding arithmetic (HateMM train line 540: 80 784
   chars − 38 prefix − 14 suffix − 1 `\"` escape = 80 731; HateMM val line 56: 12 327 − 37 −
   14 − 1 = 12 275; MHC-ZH train line 436: 756 − 32 − 14 − 2 = 708; MHC-ZH val line 26:
   389 − 32 − 14 − 0 = 343). The `max_chars_definition` field's claim that this is
   `len(json['text'])` in Python characters — the quantity `L_MAX` compares against — is
   also correct.
3. **`B_fid` = 0.0093 (HateMM) / 0.0086 (MHC-ZH)**, the F113 figures behind the GATE-FID
   stop rule. Verified this pass **against the banked artifacts themselves**:
   `scripts/analysis/headspace_fidelity_OUT.json` and `…_zh_OUT.json` both contain
   `"B_fid_abs_3seedmean"` equal to those values. Note the v8 set does not even assert them
   — `RECORD:119` correctly lists `B_fid` as *(not measured)*, because the run computes its
   own.
4. **The `_norm32` aliasing mechanism** — verified by reading `mechfix_ops.py:37-42`, as
   above. (Its *magnitude* is [U] item 1.)
5. **The k = topk exactness argument** and its tie side-condition — derivation, re-derived.
6. **"for a singleton orbit it IS the literal deployed call"** — verified by comparing
   `orbit_vote` to `mechfix_ops.deployed_vote`/`_flat_ip`.
7. **"a lone degenerate item's displacement is ZERO by construction"** — verified bitwise
   in round 7 from the extractor's one-forward-per-distinct-string copy rule
   (`extractor:244-254`), the zero-guard's `zero.clone()` into all six slots (`:236-238`),
   and the mint's identical forwards.
8. **Prompt byte-identity to the deployed assembly**, `num_frames = 8`,
   `max_pixels = 360*420`, the English instruction/label/placeholder defaults, and that
   `gen_embed_lora.sbatch` passed none of those flags — verified in round 4 by reading
   `generate_VideoMLLM_embedding_lora_HF.py:63-68,151-170,438-442`.
9. **Asset facts**: both LoRA adapters exist and are *older* than their banked caches; both
   banked caches, both P3 score files and all four gt files exist; 0 broken video symlinks
   in either `All/` tree; scipy 1.17.1 / numpy 1.26.4 / faiss-cpu 1.13.2; 1.8 TB free.
10. **All frozen-module hashes** (5 modules + the extractor's two pins), re-verified this
    pass.

### [D] Documentary — source checked, measurement not reproduced

11. **The sacct budget basis** cited at `config:266` ("largest observed extraction job
    13468 = 02:00:08"). I confirmed `TARGET_STATE.json::iteration_8_…amendment` contains
    that table with that value; I did not run `sacct`.
12. **The C01 zero-contract provenance** — that row 355 / `hate_video_95` is a documented
    video-decode-failure zero consumed by the deployed baseline
    (`C01_ZERO_CONTRACT_PROBE.md`, `PROVENANCE_AUDIT_2026-07-28.md:187-193`). I verified
    those files say it, and that `hate_video_95` is gt line 356 = index 355. The arena
    itself labels criteria 1 and 4 `DOCUMENTARY_CITATION_NOT_COMPUTED`, which is the
    honest tag.
13. **F113's transfer figures** (the 2.3× raw→held-out shrink, the HateMM sign inversion)
    — context only; nothing in the v8 decision reads them.

### [U] Asserted as measured — I cannot verify these from the artifacts

14. **The erratum's own new measurement** (`config:198`, `arena_v8.py:200-205`,
    `RECORD:24`): that an exhaustive `k = n_bank` search is bit-equal to `k = 20`
    (similarities, ids and votes identical, max |Δsim| = 0.0), and that the
    `1.4901161193847656e-07` belonged entirely to the aliasing defect. **I cannot run
    Python, so this is an assertion to me exactly as its predecessor was.** Two things make
    it materially different from the claim it replaces, and both are worth stating plainly:
    (a) it is *plausible on mechanism* — a flat IP index computes the same inner products
    regardless of `k`, which affects only heap size and result width, so bit-equality is
    what one would predict; (b) **nothing in the design depends on it.** v8 retains
    `k = topk` on two non-empirical grounds (provable sufficiency; literally the deployed
    call), so if this new measurement were also wrong, no gate, metric or verdict would
    move. That asymmetry is the right structure for an erratum: it withdraws a load-bearing
    claim and replaces it with reasons that need no measurement.
15. **The KRR synthetic-signal R² figures** (`config:233`): 0.0087 before the fit-fold
    z-scoring repair, 0.8788 after at n = 744 / d = 1024, −0.0163 on a null. Consequence if
    wrong: the declared repair might be unnecessary or insufficient — but the KRR probe is a
    **secondary, ungated diagnostic** whose contraction comparison is explicitly deferred to
    Stage-1, and no term of the decision rule reads it.
16. **The structural-zero census on the caches this run will open**: "1 on HateMM train, 0
    on MHC-ZH" (`arena_v8.py:195`, `config:197`), and the runtime predictions that follow
    from it (`config:77`: view_support 0.9341 / 1.0000). I verified the row's *identity* and
    its *documentary* provenance ([V]9, [D]12) but **not** that the specific LoRA-curric and
    MHC-ZH train caches carry exactly that census — that needs a `.pt` load, which is
    forbidden. Consequence if wrong: none for correctness. `ZERO_CONTRACT` computes the
    masks and halts on any mismatch, `VIEW_SUPPORT` computes the real fraction, and both
    predictions are correctly hedged with "*if the extraction reports …*". Only the
    predicted numbers and the tie-reachability remark would be off.
17. **The GPU-hour projection** (`config:266`): ~8 760 text forwards + 1 508 video decodes
    ⇒ ~1.5-2.5 GPU-h. The forward count is *derivable* and I re-derived ≈8 758 from the
    verified identity counts; the **wall-clock is an estimate, not a measurement**, and is
    correctly labelled "Projected spend". Consequence if wrong: an over-4.0-hour run **voids
    the result** under amendment condition (f), and nothing in-job enforces the cap. This is
    the [U] item with the largest downside.
18. **"the deployed CLI … byte-identical to `enc3seed_lora_curric.sbatch` / as replayed by
    `errpat_zh_remint.py`"** — inherited from `headspace_mint.py:126-153`, which the design
    imports unchanged. I verified the CLI *is* imported unchanged and its hash is pinned and
    enforced; I did **not** diff it against those two banked sbatch files. Consequence if
    wrong: the proxy head is not the deployed recipe — which is exactly what GATE-FID
    exists to bound, and which measured 0.0093 / 0.0086 previously ([V]3).

**Nothing in [U] is load-bearing for the verdict.** 14 is decorative by construction, 15 and
16 touch only secondary or self-computing quantities, 18 is bounded by a gate that runs
before the arena, and 17 is a budget risk that the registry already handles as a post-hoc
protocol check rather than a scientific one.

---

## 5. Item 4 — diff-through from v7

* **Constants sweep: clean.** `arena_v8.py:90-110` is unchanged in every value — `TOPK 20`,
  `BAR_ACC/BAR_MF1 0.050`, `BAR_NETFIX_RATE 0.030`, `VIEW_SUPPORT_MIN 0.60`,
  `BOOTSTRAP_B 10000`, all three seeds `20260730`, `ALPHA 0.05`, `ARENA2 0.02/0.98`,
  `EXT 0.99`, `TINY_NORM 1e-12`, `KRR_RIDGE 1.0`, the same nine `ARM_NAMES` — and each still
  matches its config counterpart (`topk :171`, weights `[20…1]`, PASS rule `:249`, gates
  `:202-209`, `seeds_frozen`, Holm family, resources, `halt_exit_code 3`, namespaces,
  `c02_a0_result_v8` / `c02_a0_decision_v8`).
* **No stale v7 identifiers.** `v7`/`V7` appears only where it must: inside the erratum
  (which names v1-v7) and inside the `supersedes` audit block. `RUN_ID`, `CFG`,
  `--job-name`, the wrapper's self-test import (`:59`) and its arena invocation (`:113`) are
  all `v8` and mutually consistent.
* **Self-test unchanged.** The same six cases in the same order with the same names
  (`:581,592,614,623,639,668`), including the `build_arms`-driven SHUFFLE case and the
  drop-not-merge assertions added in v6/v7.
* **Every earlier hardening intact:** `g.size >= 2` drop rule (`:277`), banked tiny-row
  check (`:732`), manifest sha comparison (`:801`), `degen_mask` union (`:820`),
  parity-cell count (`:961`), secondary-arena `try/except Halt` (`:1038`), `SystemExit(3)`
  (`:1185`).
* **PARITY-NAT still binds, for the reason now stated.** It never depended on the retracted
  claim. It binds because a singleton orbit issues literally the deployed
  `IndexFlatIP.search(q, 20)` on identically-normalised private copies, and
  `parity_native` asserts predictions **and** the sorted top-20 similarity vector bit-equal
  against `M.deployed_vote` on every row of all 15 seed×fold cells, with the cell count
  itself asserted. Withdrawing a false statement about a path the design does not take
  changes nothing numerically.

---

## 6. The finding, and the meta-question

**I1 (Info) — `scripts/analysis/c02_a0_arena_v8.py:207-208`.** "*it is O(topk) rather than
O(n_bank) per view pair*" is unquantified and is false under the natural reading. A flat
inner-product index computes all `n_bank` similarities per query regardless of `k`; what
`k` bounds is the selection heap and the `(nq × k)` result, not the scan. True as written
of the returned object and the selection, false of the search cost. Documentation only:
no number the run produces depends on it, and reasons 1 and 3 over-determine the choice.
**Do not cut a v9 for this** — declare it, or fold it in if a v9 ever happens for a real
reason. Suggested wording: "*it returns a `(nq × topk)` result and selects with a
`topk`-sized heap instead of materialising all `n_bank` similarities per view pair*".

**On your meta-question — yes, and I think it is the single highest-value process change
available here.** A review request that asks "is this argument sound?" gets the argument
checked; a number sitting next to the argument, phrased as already measured, reads as
settled context and is exactly what a static reviewer has no way to test. Six of us checked
the derivation and none of us asked *which of these numbers were produced by running
something*. Concretely, I would put a **MEASURED-CLAIMS REGISTER** in the record — a
standing section, not a one-off — with one row per empirical claim and four columns:
*claim · how obtained (script/artifact/command, or "estimate") · what depends on it ·
re-derivable by a static reviewer?* Then the review request can say "audit the register",
and the reviewer's job on numbers becomes checkable instead of implicit. §4 above is that
register for v8; it took one pass to build, it converted three claims I had previously
taken on trust into verified ones ([V]1's MHC-ZH counts, [V]2's fourth `max_chars`, [V]3's
`B_fid`), and it surfaced that the largest residual risk in the set is not a science claim
at all but the unenforced GPU-hour projection ([U]17).

A second, cheaper habit that would have caught this specific class: when two defects are
live in one dry run, **re-measure every claim attributed to the first after fixing the
second**. The erratum's own account says that is precisely what did not happen.

---

## 7. Verdict

```
GO (0C/0H/1I)
```

Eight hashes match; v7 executables gone; namespaces clean. The retracted claim survives
nowhere but inside its own retraction, and `oracle.search` has been cleaned of it. The
replacement justification is true on two of its three reasons as written and true on the
third under the result-size reading only (I1). The exactness argument is unchanged and
still holds; PARITY-NAT still binds, and never depended on the retracted claim. Constants,
self-test, hardenings and identifiers are all clean through the diff. No new defect.

The single Info is documentation with no numerical consequence, and I recommend declaring
it rather than producing a v9. Two operator preconditions are unchanged and unenforceable
in code: run the `squeue` check for amendment condition (e), and re-verify the eight
sha256 values immediately before `sbatch`. The one risk I would keep visible while the
extraction runs is [U]17 — the 4.0 GPU-hour cap is a post-hoc `sacct` check, and an
overrun voids the result.

---

## 8. What I did and did not execute

**Did:** `sha256sum`, `ls`, `find`, `grep`, `sed`, `awk`, `cut`, `rev`, `tr`, `wc`, and file
reads. New verifications this pass: the MHC-ZH whitespace-only counts (0/0), the
over-length record counts on all four splits (9/1/0/0, stable across three thresholds), the
fourth `max_chars` value (343), and the two banked `B_fid` figures read directly out of
`scripts/analysis/headspace_fidelity{,_zh}_OUT.json`. Plus the confined v8 checks: eight
hashes, the four wrapper-pinned module hashes, the extractor's two pins, the retracted-claim
greps, `orbit_vote`'s docstring, `oracle.search`/`search_width_erratum`, the constants
sweep, the self-test case list and the stale-identifier scan.

**Did not:** run Python of any kind, import any module, load or open any `.pt` cache,
`.npz`, model, adapter or video; open any `test_seen` cache, `test.jsonl`, or any test
label or metric; run `squeue`, `sacct`, `sbatch` or any SLURM command; touch a GPU or
Modal; modify, move or delete any reviewed file; or write anything other than this review.
The two JSON files I read (`headspace_fidelity{,_zh}_OUT.json`) are dev-only GATE-FID gate
outputs containing no test metric — I checked their `gate` block only.
