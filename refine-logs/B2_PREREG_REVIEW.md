# B2 Pre-Registration Review — frozen Qwen2.5-VL-32B encoder SCALE axis (all 3 datasets, 3-seed paired, dual protocol)

**Reviewer:** fresh zero-prior-context pre-registration reviewer (read-only + CPU code
verification; no GPU / SLURM / downloads / commits). **Date:** 2026-07-14.
**Under review:** `research-wiki/experiments/exp-encoder-32b-b2.md`
(status `DRAFT-UNREVIEWED`; sha256 below).
**Method:** re-read the draft against the parent (`exp-encoder-3seed.md`), B1
(`exp-encoder-zh-b1.md` + `refine-logs/B1_VERDICT_REVIEW.md`), and the SAV verdict
(`refine-logs/SAV_F1_VERDICT_REVIEW.md`); independently read the extraction script, the RGCL
head/loader source, the disk_guard script, the SLURM templates, and the reused reference
logs; verified split counts and cache/weight state on disk.

**VERDICT: APPROVED — with 4 mandatory revisions (all trivial text/spec-precision fixes;
none touch the scientific design). Conditional execution authorization appended.**

---

## 0. Prereg hash (recorded)

```
sha256(exp-encoder-32b-b2.md) = d39ea5dc2dc5cab2d2ff267ed8d9365e8e526bac495ab74ce681a85c719a77e5
```

The three Stage-D/E/T sbatch runners are **NOT YET AUTHORED** (verified: `scripts/slurm/`
has no `enc3seed_32b_b2.sbatch`, no 32B download sbatch, no 32B extraction sbatch). Their
sha256 must be recorded at delta-check, once authored + diff-verified.

---

## 1. RULE FIDELITY — PASS

- **Decision rule transcribed VERBATIM.** B2 lines 216-228 (blockquote) reproduce
  `exp-encoder-3seed.md:73-85` word-for-word: per-seed paired Δ = (Qwen−CLIP) on acc AND
  mF1; 3-seed mean±std + sign consistency; paired-t **as effect-size descriptor only**
  (n=3, no significance claim); **pass = mean Δacc ≥ +0.030 AND mean ΔmF1 ≥ +0.030 AND 3/3
  sign-positive**; headline requires the criterion on **≥ 2 datasets under a stated
  protocol**. Confirmed identical to parent and to B1's transcription (`B1_VERDICT_REVIEW.md`
  Task 4).
- **Both protocols, judged independently.** B2 202-207 transcribes the (A) val-selected /
  (B) final-epoch definitions verbatim from `exp-encoder-3seed.md:66-71`; the fixed write-up
  format "final-epoch: pass/fail; val-selected: pass/fail" is carried (B2 209-210, kill rule
  5). No protocol is designated "primary" (correctly — unlike ZH-only B1 — since HateMM/EN
  carry no 78-dev tax; B2 211-214).
- **Goal-relevance pre-declared and unambiguous.** B2 239-247: "**Goal-relevant success …
  a PASS of the 32B-vs-CLIP comparison on ANY of MHC-EN or MHC-ZH**"; "**HateMM-only
  improvement is NOT a goal pass** — merely restates what 7B already banked." This matches
  the checklist's framing exactly (MHC-EN or ZH pass = goal-relevant; HateMM-only = not a
  goal pass since banked at 7B).
- **Primary vs secondary comparison well-formed.** Primary = 32B-vs-**CLIP** (the goal gate,
  same control the parent scored the goal on); secondary = 32B-vs-**7B** (pure scale
  increment, diagnostic only). Faithful elaboration of the parent, not a rule change.

## 2. REFERENCE ARMS — PASS, with one mandatory provenance-precision fix (Rev-1)

- **32B compares against SAME-RUNNER CLIP + 7B** — the runner family is `enc3seed*`, each a
  CONFIGS-only copy of `scripts/slurm/enc3seed.sbatch` whose per-config python command is the
  exact `train_archive_baseline` command (verified: `enc3seed.sbatch:44-62`, archive-OFF via
  `--lambda_seg 0` and no `--archive_feats`; differs across arms only in `--dataset`/
  `--model`/`--seed`). So 32B (B2) and the CLIP/7B references share the runner modulo
  `--model`. ✓
- **HateMM + MHC-EN CLIP/7B references (job 12850) — logs exist, VERIFIED:**
  `enc3s_HateMM_{clip,Qwen}_seed{0,1,2}_12850.trainlog` (6) and
  `enc3s_MHC_openai_clip…_seed{0,1,2}_12850.trainlog` (3) + `enc3s_MHC_Qwen…_seed0_12850.trainlog`
  (1) all present.
- **⚠ Provenance error (Rev-1, MANDATORY).** MHC-EN **7B-Qwen s1/s2 are NOT in job 12850** —
  only seed0 is. They are the reused **arcbase logs 12275/12276**
  (`arcbase_MHC_Qwen2.5-VL-7B-Instruct_HF_seed{1,2}_1227{5,6}.trainlog`, VERIFIED present),
  exactly as the parent documents (`exp-encoder-3seed.md:118,258-259`). B2 lines 108-109 and
  the table header at line 136 wrongly attribute all MHC-EN 7B arms to 12850. Both are still
  same-runner (`train_archive_baseline`, current code — the parent audited 12275 Namespace =
  differs only in `model=`, `exp-encoder-3seed.md:141,145-146`), so the same-runner property
  holds; but the **executor's Stage-T Namespace cross-check for MHC-EN 7B s1/s2 must target
  12275/12276, not 12850.** (Numeric-provenance-discipline: the delta-check diffs against
  specific log filenames — they must be the right ones.)
- **MHC-ZH CLIP/7B references (job 13115) — VERIFIED:** all six
  `enc3s_MHC_zh_{clip,Qwen}_seed{0,1,2}_13115.trainlog` present; produced by
  `enc3seed_zh_b1.sbatch` (CONFIGS-copy). Same-runner. ✓
- **Namespace cross-check mandated at delta-check — PRESENT.** B2 113-114, 118, 191-193,
  326-327, and kill rule 2 (336-338) require the 32B `Namespace` to differ from each reused
  reference arm **only** in `model=`/`exp_comment=`/`output_path=`, HALT otherwise. ✓

## 3. COST / DISK HONESTY — PASS

- **Count-based extraction arithmetic — internally consistent, VERIFIED.** Video counts
  equal the gt-split totals (I verified: HateMM 744+107+215=**1066**, MHC 549+80+161=**790**,
  MHC_zh 579+78+149=**806**). At 7-9 s/video: 1066×(7-9)=2.07-2.66 GPU-h, 790×(7-9)=1.54-1.98,
  806×(7-9)=1.57-2.02 → **5.2-6.7 GPU-h** total. Matches the draft's "~2.1-2.7 / ~1.5-2.0 /
  ~1.6-2.0 → ~5-7 GPU-h" (B2 416-419) to rounding.
- **2-forwards-per-video folded into 7-9 s — consistent.** The extractor runs exactly 2
  frozen forwards/video (img "prefix" span + text "response" span,
  `generate_VideoMLLM_embedding_HF.py:345-359`, `_encode` span logic :290/:304). The draft
  states the 2 forwards are already inside the 7-9 s figure (B2 416-417). Scaling the 7B
  anchor consistently: 7-9 s/video at 32B ≈ **3-3.75×** the stated 2.4 s/video 7B anchor
  (both-forwards) — sub-linear-to-linear against 32B's ~4.6× dense-param ratio, i.e.
  consistent-to-conservative. The draft **honestly flags** the reconciliation with the
  scale-recon's lower "~1-1.5 GPU-h/dataset" figure (which implies ~600-700 videos, below the
  actual 790-1066) and adopts the higher count-based number (B2 420-424). Honest budget. ✓
- **Transient-disk lifecycle — correct.** Download 66G (Stage-D) → extract (Stage-E) →
  **G-dims verify** → **delete weights (Stage-C)** → train (Stage-T reads only the .pt,
  `HF_HUB_OFFLINE=1`, needs no weights). Weights deleted **only after G-dims PASS** (B2
  306-313). Efficient ordering (weights gone before the full train). ✓
- **Quota-grace — logic sound; free-space claim VERIFIED.** `df` shows **409G Avail** on
  `/data` and `du /data/jehc223 = 382G` — matches the draft's 382G-used / 409G-free. Adding
  66G → 448G used, 343G free remaining; far under the 3000G hard cap; the ~24h grace window >
  the ~few-hours transient lifecycle. The soft-quota grace clock is a point-in-time value I
  cannot re-derive read-only, but the reasoning holds regardless. (Executor should re-`df`
  before Stage-D and record it — folded into the authorization.)
- **disk_guard behaviour — VERIFIED, and the .pt-push caveat is correctly covered.**
  `disk_guard.sh` **never removes `models--*`** (hard skip, `disk_guard.sh:378-380`
  "SKIP (models--* protected)"; only `datasets--*`/`.locks` are ever purgeable, and only under
  the non-default `DISK_GUARD_HF_PURGE=1`) — so the staged 66G weights cannot be pruned
  mid-job. Its reclaim allowlist **does include `data/CLIP_Embedding`** (`disk_guard.sh:85`),
  so the draft's caveat that disk_guard **can** prune the fresh 32B `.pt` is correct, and the
  mitigation (the extraction runner ends with `b2_push` of `data/CLIP_Embedding/<ds>`, B2
  386-389; base template `gen_embed_mllm.sbatch` already does this) is appropriate. Note also
  the extraction sbatch itself calls `disk_guard.sh` at startup (`gen_embed_mllm.sbatch:19`),
  which is safe for the just-downloaded weights (models--* protected).

## 4. GATES — PASS; dim-wiring SETTLED affirmatively by code read (no extra pre-check needed)

- **G-repro as first-run sanity-only — LEGITIMATE substitution, clearly labelled.** There is
  genuinely no prior 32B log to reproduce against (VERIFIED: no `*32B*` under
  `data/CLIP_Embedding`, no `models--…-32B-…` in the HF cache, no 32B enc3s trainlog). The
  parent/B1 seed0 bit-for-bit reproduction gate therefore **cannot apply**; the draft
  correctly downgrades G-repro to a sanity check (loads 5120-d caches without dim/wiring
  error; trains 30 epochs; Test acc/F1 in a plausible non-degenerate band) and **states this
  in the record so the absence-of-repro-match is not mistaken for a skipped gate** (B2
  183-188, 316-324). ✓
- **G-dims (HARD) — well-specified.** dim == 5120, row counts == the verified splits
  (744/107/215, 549/80/161, 579/78/149), id lists == the CLIP/7B arms (paired-id audit). ✓
- **Kill rules + single-submit per stage — present** (B2 329-349; Stage-D/E/T/C each a single
  submit, 249-313, 434-447).

### 4a. Dim-inference settlement (my own code read — the draft's "smoke-confirmed-not-verified" flag is now settled AFFIRMATIVELY)

**Finding: the RGCL head/loader infers input dim from the loaded `.pt` tensors at runtime;
there is NO hard-coded 3584/5120 anywhere on the head or loader path. 5120-d auto-wires
end-to-end. A separate cheap pre-check is NOT needed — the code read settles it.**

Full chain (cited):
- **Loader is dim-agnostic.** `src/data_loader/dataset.py:499-503` routes `HateMM`/`MHC`/
  `MHC_zh` to `load_feats_MHC`; `:606-608` builds `{path}/{dataset}/{split}_{model}.pt` and
  `load_feats_split` (:561-582) returns the raw `img_feats`/`text_feats` tensors **unchanged**
  — no dim assumption, no reshape.
- **Head input dims are read live from the train tensors.**
  `src/run_rac.py:1102` `image_feat_dim = list(enumerate(train_dl))[0][1]["image_feats"].shape[1]`
  and `:1103` `text_feat_dim = … ["text_feats"].shape[1]` — the projection input widths are
  taken from `.shape[1]` of the first train batch (printed at `:1104-1105`).
- **Those inferred dims are passed straight to the head.**
  `src/run_rac.py:1117-1120` `classifier_hateClipper(image_feat_dim, text_feat_dim,
  args.num_layers, args.proj_dim, args.map_dim, …)`.
- **The head builds its first projections from those args.**
  `src/model/classifier.py:76` `self.img_proj = nn.Sequential(nn.Linear(image_dim, map_dim),
  …)`, `:77` `self.text_proj = nn.Sequential(nn.Linear(text_dim, map_dim), …)`. Everything
  downstream operates in `map_dim` (=1024), so the 5120 input width only sizes these two
  `nn.Linear`s. (`fusion_mode="align"` → element-wise product of the two map_dim vectors,
  `classifier.py:81-85` — input dim independent.)
- **Extraction side produces exactly 5120-d for 32B.**
  `generate_VideoMLLM_embedding_HF.py:332` `d = model.config.hidden_size` (the trailing
  comment `# 3584 for Qwen2.5-VL-7B` is a *label of the 7B value*, not a hard-code); features
  stacked `[N, d]` at `:378-379`; written to `{outname}_{out_model_tag}.pt` at `:437`. For
  Qwen2.5-VL-32B `config.hidden_size = 5120`, so `img_feats`/`text_feats` are 5120-d — the
  identical mechanism that yielded 3584-d for 7B and 1024/768-d for CLIP.

**Conclusion:** the "5120 auto-wiring" assumption is **code-verified**, not merely
smoke-confirmed. The G-repro sanity gate remains valuable as a *cache-integrity + first-32B-run*
check (does the extracted `.pt` load and train cleanly), but the *dim-wiring* itself is now
settled by this read. The draft should upgrade its wording (Rev-4 is unrelated; this is a
free strengthening — see note under Revisions).

## 5. TEST-TOUCH — PASS

- **Parent reads Test per epoch — VERIFIED.** `enc3seed.sbatch:70-83` parses **every**
  `Test_Retrieval Epoch NN …` line and reports both val-sel (epoch ≥ warmup, val-acc, roc
  tie-break) and final-epoch from those per-epoch Test readouts — i.e. Test is evaluated each
  epoch by construction.
- **B2 follows the parent's accounting explicitly** (B2 352-366): it inherits the encoder
  campaign's precedent (the whole 3-seed dual-protocol comparison = the one pre-declared
  evaluation) rather than inventing a new one, and **says so** ("This differs from the strict
  one-held-out-touch discipline used elsewhere; B2 inherits the encoder campaign's
  precedent"). Accounting is consistent with parent and B1. ✓

## 6. BURN-HISTORY RISKS — mostly covered; 2 mandatory fixes (Rev-2, Rev-3)

- **Download sbatch (NEW) — spec is correct.** Draft Stage-D (B2 251-263): `HF_HUB_OFFLINE=0`
  (must reach the hub ✓), **no `--gres`** (CPU-only ✓), **no `--time`** ✓. These are exactly
  the three burn-avoidance checks for a new download job, and the draft passes all three. (The
  runner is not yet authored — hash at delta-check.)
- **Extraction sbatch needs `PYTORCH_CUDA_ALLOC_CONF` — covered.** Draft mandates
  `export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` (B2 271-272), P10-proven
  (`p10_score_ladder.sbatch:24`, VERIFIED, with the "32B bf16 … fragmentation OOM without
  this" comment; that file also confirms **32B bf16 fits 80G**). The base template
  `gen_embed_mllm.sbatch` does **not** set it, so it must be added — draft covers this. Keep
  `HF_HUB_OFFLINE=1` for Stage-E (weights local from Stage-D) — draft covers this
  (`gen_embed_mllm.sbatch:26` already sets it). bf16 + sdpa + `device_map=None` +
  `model.to(device)` are script defaults (`generate_VideoMLLM_embedding_HF.py:396-402`,
  VERIFIED). ✓
- **⚠ Rev-2 (MANDATORY): extraction sbatch MUST set BOTH `--model` AND `--out_model_tag`.**
  The base template passes **neither** (`gen_embed_mllm.sbatch` calls the script with only
  `--dataset/--num_frames/--device`; the script defaults `--model=Qwen/Qwen2.5-VL-7B-Instruct`
  and `--out_model_tag=Qwen2.5-VL-7B-Instruct_HF`, `generate_VideoMLLM_embedding_HF.py:81,87`).
  If `--out_model_tag` is omitted for the 32B run, the output is written to the **7B cache
  filename** (`{split}_Qwen2.5-VL-7B-Instruct_HF.pt`) — a silent collision/overwrite of the
  existing 7B caches, and Stage-T would then load 7B features under a 32B label. The draft
  mentions adding both flags (B2 268-269) but this must be a **hard diff-verify item**: the
  authored extraction sbatch is required to contain **both** `--model
  Qwen/Qwen2.5-VL-32B-Instruct` **and** `--out_model_tag Qwen2.5-VL-32B-Instruct_HF`, checked
  before submit. (G-dims dim==5120 is a backstop, but the filename collision precedes G-dims.)
- **⚠ Rev-3 (MANDATORY, provenance precision): "P10-proven on 32B bf16" is overstated for the
  extractor.** The *embedding extractor* `generate_VideoMLLM_embedding_HF.py` has only ever
  run **7B**; P10 proved the *scorer* `p10_score_ladder.py` runs 32B bf16 on 80G. The
  extractor reuses the same `Qwen2_5_VLForConditionalGeneration.from_pretrained` bf16/sdpa
  loading path, and its 8-frame `max_pixels=360*420` forward is *lighter* than P10's K30/M120
  windows, so the risk is low and the G-repro sanity gate covers it — but the wording (B2
  provenance line 12; connection line 468) should read "reuses the 32B-proven model-loading
  path; its **first** 32B extraction run is gated by G-repro," not "P10-proven on 32B bf16."
  (Consistent with the project's numeric/provenance-precision discipline.)
- **⚠ Rev-4 (MANDATORY, clarity): specify how G-repro is executed vs the 9-run Stage-T
  single-submit.** The ceremony (B2 441-444) folds "G-repro sanity on HateMM s0" into Stage-E
  verification, then deletes weights (Stage-C), then runs the full 9-run Stage-T — but HateMM
  s0 is *also* config 1 of the 9-run serial `enc3seed_32b_b2.sbatch`, which runs all 9 with no
  inter-run gate. Clarify one of: (a) HateMM-s0 G-repro is a **separate 1-config smoke submit**
  verified before the 9-run Stage-T (cleanest; training needs no weights, so it can run after
  Stage-C deletion), or (b) it is the **first-config output of the 9-run job**, read post-hoc
  as the sanity check. As written it is ambiguous whether G-repro is its own submit (which the
  single-submit-per-stage discipline should name explicitly) or part of Stage-T.

---

## Revisions (mandatory; all trivial text/spec fixes — no design change)

1. **Rev-1 — MHC-EN 7B s1/s2 provenance.** Correct B2 lines 108-109 and the MHC-EN reference
   table header (line 136): MHC-EN 7B-Qwen **seed0 = job 12850**, but **seeds 1/2 = reused
   arcbase jobs 12275/12276** (`arcbase_MHC_Qwen…_seed{1,2}_1227{5,6}.trainlog`; parent
   `exp-encoder-3seed.md:118,258-259`). State that the Stage-T Namespace cross-check for those
   two arms targets 12275/12276 (same runner, current code), not 12850.
2. **Rev-2 — extraction flags.** Elevate to a hard diff-verify item: the authored Stage-E
   sbatch MUST pass **both** `--model Qwen/Qwen2.5-VL-32B-Instruct` **and** `--out_model_tag
   Qwen2.5-VL-32B-Instruct_HF` (omitting the tag silently writes to the 7B cache filename).
3. **Rev-3 — "P10-proven" wording.** Reword B2 line 12 and connection line 468: the *scorer*
   was P10-proven on 32B bf16; the *extractor* reuses that loading path and its first 32B run
   is G-repro-gated.
4. **Rev-4 — G-repro execution mode.** State explicitly whether the HateMM-s0 G-repro sanity
   is a separate smoke submit (verified before the 9-run Stage-T) or the first-config readout
   of the 9-run serial job.

**Free strengthening (recommended, not required for authorization):** update B2 lines 176-181
to cite the dim-inference code read in §4a — the 5120 auto-wiring is now **code-verified**
(`run_rac.py:1102-1103` infer dim from the loaded tensors → `:1117-1120` → `classifier.py:76-77`),
so "should wire in automatically; G-repro/smoke must confirm" can become "dim inferred from
the loaded `.pt` tensor shape (code-verified); G-repro remains a cache-integrity sanity check."

---

## CONDITIONAL EXECUTION AUTHORIZATION

**Status: CONDITIONALLY AUTHORIZED.** The scientific design is sound, the decision rule and
protocols are verbatim from the parent, the gates (incl. the sanity-only G-repro substitution)
are legitimate, the dim-wiring is code-verified, and the disk/cost accounting is honest.
Authorization is **conditional on**, in order:

- **C0. Apply Rev-1…Rev-4** (trivial edits) and record them in the B2 revision history; the
  free dim-inference strengthening is recommended.
- **C1. Author the three runners** — Stage-D download sbatch, Stage-E extraction sbatch
  (`gen_embed_mllm.sbatch` + `--model`/`--out_model_tag`/`expandable_segments`; keep
  `HF_HUB_OFFLINE=1`, `--gres=gpu:a100:1`, no `--time`), Stage-T `enc3seed_32b_b2.sbatch`
  (CONFIGS-only + `QWEN=Qwen2.5-VL-32B-Instruct_HF` copy of `enc3seed.sbatch`). Diff-verify each
  is a minimal delta; **record sha256 of all three**.
- **C2. Reviewer delta-check** of C0+C1 (including that the Stage-D env sets `HF_HUB_OFFLINE=0`
  / no `--gres` / no `--time`; that Stage-E sets both `--model` and `--out_model_tag` and
  `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`; that Stage-T is a byte-identical CONFIGS
  copy).
- **C3. Explicit user/main go** (GPUs shared with the user's own loop; CLAUDE.md — every GPU
  task via SLURM, subagents do the work).

### Staged single-submits (each gated on the prior stage's verification)

1. **Stage-D (download, CPU, no GPU):** single submit. Verify: 32 repo files fetched, cache
   `~/.cache/huggingface/hub/models--Qwen--Qwen2.5-VL-32B-Instruct` present. Record `df`
   before and after. **No resubmit.**
2. **Stage-E (extraction, 1×A100-80G):** single submit, order HateMM → MHC → MHC_zh. Then run
   **G-dims** on all 9 caches (dim==5120; rows 744/107/215, 549/80/161, 579/78/149; id lists ==
   the CLIP/7B arms) and the **HateMM-s0 G-repro sanity** (per Rev-4). Kill rule 1: OOM/crash →
   capture traceback + `nvidia-smi`, record, **HALT — no blind resubmit**.
3. **Stage-C (cleanup):** only **after G-dims PASS**, delete the ~66G weights
   (`rm -rf …/models--Qwen--Qwen2.5-VL-32B-Instruct`) and **document with `df` before/after as
   evidence**. Embeddings retained + B2-pushed.
4. **Stage-T (training, 1×A100):** one serial sbatch (9 runs, 32B arm only). Run the
   **Namespace-diff gate** against 12850 (HateMM + MHC-EN CLIP/EN-7B-s0), **12275/12276**
   (MHC-EN 7B s1/s2, per Rev-1), and 13115 (MHC-ZH CLIP/7B) — HALT on any substantive field
   beyond `model=`/`exp_comment=`/`output_path=`. Then read every number back from the raw
   `enc3s_*_Qwen2.5-VL-32B-Instruct_HF_*` trainlogs (line-numbered provenance) and apply the
   decision rule verbatim (32B-vs-CLIP primary; 32B-vs-7B secondary).

### Executor duties

- **Keep a full execution record** (B2_EXECUTION_RECORD.md style) with raw-log line-numbered
  provenance for every number; **report raw numbers only**, no tabulation-before-transcription.
- **No resubmits** on any stage failure — HALT, capture evidence (traceback + `nvidia-smi` +
  `df`), and escalate; kill rules 1-6 govern.
- **`PENDING (JobHeldUser)` = wait for auto-release**, never force.
- **Weights deletion (Stage-C) only AFTER G-dims PASS, documented with `df` evidence** (before
  + after).
- **Single submit per stage**, each gated on the prior stage's verification; no mid-run
  resubmission of the 9-run Stage-T.
- **No test-set re-runs with tweaked knobs** (test-touch budget = this one 9-run evaluation).

### Hashes recorded

```
prereg  exp-encoder-32b-b2.md  sha256 = d39ea5dc2dc5cab2d2ff267ed8d9365e8e526bac495ab74ce681a85c719a77e5
Stage-D sbatch  = (not yet authored — record at delta-check)
Stage-E sbatch  = (not yet authored — record at delta-check)
Stage-T sbatch  = (not yet authored — record at delta-check)
```

---

## Summary table

| checklist item | finding |
|---|---|
| 1. Rule fidelity | **PASS** — decision rule + both protocols verbatim from parent; goal-relevance (MHC-EN/ZH pass = goal; HateMM-only = not) unambiguous |
| 2. Reference arms | **PASS + Rev-1** — 12850 (HateMM/EN CLIP+7B-s0) & 13115 (ZH) verified same-runner; **MHC-EN 7B s1/s2 = 12275/12276, not 12850** (fix provenance + cross-check target) |
| 3. Cost/disk honesty | **PASS** — count-based arithmetic consistent; 2-forwards folded; lifecycle+quota logic sound (409G free verified); disk_guard models--* protection + .pt-push caveat correct |
| 4. Gates | **PASS** — G-repro sanity-only substitution legitimate & labelled; **dim-wiring code-verified** (see §4a); G-dims well-specified |
| 5. Test-touch | **PASS** — follows parent's per-epoch-Test precedent, stated explicitly |
| 6. Burn history | **Rev-2** (extraction must set both --model AND --out_model_tag), **Rev-3** ("P10-proven" overstated for extractor), **Rev-4** (G-repro execution mode); Stage-D env correct (HF_HUB_OFFLINE=0/no-gres/no-time) |
| **dim-inference** | **SETTLED (code read):** head infers dim from loaded `.pt` shape (`run_rac.py:1102-1103` → `:1117-1120` → `classifier.py:76-77`); no hard-coded 3584/5120; 5120-d auto-wires. No extra pre-check needed. |
| **VERDICT** | **APPROVED — 4 mandatory trivial revisions; conditional execution authorization appended** |

---

# C2 DELTA-CHECK (same reviewer, 2026-07-14) — 1 residual; C2-PASS WITHHELD pending R-1

**Under check:** prereg rev r1 (`exp-encoder-32b-b2.md`, `DRAFT-REV1-AWAITING-DELTA-CHECK`),
the three new runners (`scripts/slurm/b2_stage_{d,e,t}_*.sbatch`), and
`refine-logs/B2_IMPL_NOTES.md`. Every item below was re-verified **in the files directly**,
not from the impl notes' claims.

## Hashes — ALL 4 MATCH the impl-notes record (independently re-hashed)

```
702fd5e6c48156ad178a0412074e3d079713f889158f27b91d8d64a22703e236  scripts/slurm/b2_stage_d_download.sbatch   MATCH
532a8a3458f84862919d625da17b3e7e33d437b465d9bde13e93a475c5a1ff1c  scripts/slurm/b2_stage_e_extract.sbatch    MATCH
9c312da639dba0ee8061b1bb3e22b4a4a074db1812e043763732e666ef04564c  scripts/slurm/b2_stage_t_train.sbatch      MATCH
56588dc1b2f492e002948e9844f5059ba4bab1a156589bc67ca75b082833eb0b  research-wiki/experiments/exp-encoder-32b-b2.md (r1)  MATCH
```

`bash -n`: OK on all three runners (re-run independently).

## Revision landing — ALL FOUR REVISIONS + FREE STRENGTHENING LANDED

- **Rev-1 (12275/12276) — landed at ALL cited sites (7 found, superset of the 6 required):**
  frontmatter provenance (B2:12); Design reference-arms bullet "(Rev-1, provenance-precise)"
  (B2:112-118); same-runner cross-check target list incl. 12275/12276 (B2:119-125); MHC-EN
  reference-table header (B2:147-148); **Namespace-diff gate targets** "12850 / 12275-12276 /
  13115" (B2:206-211); Stage-T section (B2:326-327); ceremony step 6 (B2:495-497); connections
  `controls-against` (B2:522). ✓
- **Rev-2 — landed.** Prereg Stage-E item 1 is now a HARD pre-submit diff-verify item with the
  burn risk stated verbatim (B2:288-299); ceremony step 2 (B2:486-489). **Both flags verified
  IN THE FILE by this reviewer:** `b2_stage_e_extract.sbatch:44` `--model "$MODEL_ID"` and
  `:45` `--out_model_tag "$OUT_TAG"`, with `MODEL_ID="Qwen/Qwen2.5-VL-32B-Instruct"` (:31) and
  `OUT_TAG="Qwen2.5-VL-32B-Instruct_HF"` (:32) — on the single extractor invocation inside
  `extract_one()`. Matches impl-notes §(d) line-for-line. ✓
- **Rev-3 — landed.** Provenance (B2:12); Stage-E "(Rev-3, provenance precision)" bullet
  (B2:305-310); kill rule 1 reworded to "P10 *scorer* proved" (B2:379-384); connections `uses`
  (B2:524). ✓
- **Rev-4 — landed and pinned.** "G-repro execution mode (Rev-4, pinned)" gate bullet
  (B2:364-372): first-config readout of the 9-run Stage-T serial job, config #1 = HateMM s0,
  NO separate smoke submit, NO mid-job intervention, gate applied at verdict processing with
  HALT-tabulation on failure; Stage-E order note fixed (B2:311-314); Stage-T CONFIGS table
  marks config #1 (B2:327,331); ceremony steps 4 ("G-repro is NOT here — Rev-4", B2:491-492)
  and 6 (B2:495-497). **And the authored runner physically implements it:**
  `b2_stage_t_train.sbatch:32` = `"HateMM $QWEN 0"` is config #1. ✓
- **Free strengthening — applied.** Asset-check bullet cites the §4a code read as
  CODE-VERIFIED with the exact chain `run_rac.py:1102-1103` → `:1117-1120` →
  `classifier.py:76-77` (B2:188-197). ✓

## Runner verification (independent)

- **Stage-D (`b2_stage_d_download.sbatch`):** NO `--gres` ✓ (in-file NOTE :8), NO `--time` ✓
  (NOTE :7), `HF_HUB_OFFLINE=0` ✓ (:22), `df` before/after ✓ (:28-29,:34-35), snapshot listing
  + 18-shard count check + `du` totals/per-shard ✓ (:38-54).
  **`HF_HUB_ENABLE_HF_TRANSFER` omission — REVIEWER AGREES, independently verified:**
  `python -c "import hf_transfer"` in the HateVideo env → `ModuleNotFoundError` (re-run by this
  reviewer 2026-07-14; hub 0.29.3); setting the flag without the package makes hub downloads
  fail, so omission is correct. Also verified: `hf` CLI does NOT exist in the env,
  `huggingface-cli` does (`envs/HateVideo/bin/huggingface-cli`) — the authored
  `huggingface-cli download` command (:32) is the right variant and is explicitly permitted by
  the prereg ("or the equivalent", B2:274-275). ✓ (but see **R-1** below)
- **Stage-E (`b2_stage_e_extract.sbatch`):** `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`
  ✓ (:29), `HF_HUB_OFFLINE=1` ✓ (:26), `set -euo pipefail` ✓ (:19, fail-closed between
  datasets), serial order HateMM → MHC → MHC_zh ✓ (:66-68), per-dataset dim/rowcount echo ✓
  (:47-60), per-dataset `b2_push` ✓ (:63), SBATCH gpu/cpu/mem + no `--time` identical to the
  `gen_embed_mllm.sbatch` template ✓, startup `disk_guard.sh || true` retained (safe —
  `models--*` protected) ✓. **Echo-script correctness independently validated against the
  extractor's save contract:** the extractor saves `"ids": [ids]` — ONE sublist of all string
  ids (`generate_VideoMLLM_embedding_HF.py:430-432`) — so the echo's `o["ids"][0]` /
  `len(ids)` is the correct row count; `labels` is saved as a tensor (`:381-384`), so
  `.shape` is valid. ✓
- **Stage-T (`b2_stage_t_train.sbatch`):** independent `diff -u` vs
  `scripts/slurm/enc3seed.sbatch` = **exactly two change regions in one @@ block** — the
  `QWEN=` tag line (7B→32B) and the CONFIGS block (10 mixed rows → the 9 pre-registered 32B
  rows, order = prereg table, config #1 = HateMM s0). Everything else byte-identical,
  including `--force False` (:61), `GROUP_NAME=RAC_video_archive_seeds`, `WARMUP=5`, the full
  `run_one()` python command, the readout parser, and the `--exp_comment "_${MODEL}"`
  mechanism. ✓ Job-name stays `enc3seed` — **B1 precedent confirmed** (verified:
  `enc3seed_zh_b1.sbatch:6` also kept `--job-name=enc3seed`); fresh `$SLURM_JOB_ID`
  disambiguates. The inherited "Runs 10 configs" header-comment nit is pre-flagged in
  impl-notes §(e) — accepted, not an unexplained change. ✓
- **Collision checks re-run:** 0 `*32B*` files under `data/CLIP_Embedding`; 0 `*32B*` dirs
  under `logging/Retrieval/{HateMM,MHC,MHC_zh}/RAC_video_archive_seeds/`; no 32B HF weight
  cache; no `enc3s_*32B*` trainlogs (only the old P10 `dl_qwen25vl_32b.log`). 7B cache
  baseline mtimes recorded (train_*.pt dated Jul 2) as the Rev-2 overwrite tripwire. ✓
- **Runner-name change** (review suggested `enc3seed_32b_b2.sbatch`; authored as
  `b2_stage_t_train.sbatch`): **accepted** — documented amendment, pinned consistently in
  prereg r1 (frontmatter, ceremony step 2, connections, revision history "Runner names
  pinned"). Not a residual.

## Residuals — 1 (numbered)

- **R-1 (trivial, but undocumented prereg↔runner divergence — must be reconciled before
  C2-PASS).** The prereg's pre-registered Stage-D SLURM spec reads `--cpus-per-task=8`,
  `--mem=32G` (`exp-encoder-32b-b2.md:271-272`, unchanged from r0), but the authored runner
  requests `--cpus-per-task=4` / `--mem=16G` (`b2_stage_d_download.sbatch:3-4`), and
  impl-notes §(b) states 4/16G **without flagging the divergence**. Functionally harmless
  (smaller request; download is network-bound; no `--gres`/no `--time` unaffected) — but the
  executed artifact must match the frozen prereg text or the prereg must be amended, per this
  project's ceremony discipline. **Fix (either side, one line):** (a) amend
  `exp-encoder-32b-b2.md:271-272` to 4/16G via a documented REPLACE-in-place r2 note, or
  (b) change `b2_stage_d_download.sbatch:3-4` to 8/32G. Then **re-hash the one amended file**
  and record it. No other item needs re-review; C2-PASS may be appended immediately on
  confirmation of the reconciliation + new hash.

## C2 status

**C2-PASS WITHHELD — solely on R-1.** All four revisions landed at every cited site, all four
hashes match, all three runners are faithful minimal deltas (Stage-E Rev-2 flags verified
in-file at :44-45; Stage-T diff exactly two hunks; Stage-D env trio correct and the
hf_transfer omission independently confirmed correct). Upon R-1 reconciliation (one-line fix
on either side + re-hash of the amended file), C2-PASS is immediate and the conditional
authorization becomes live pending **C3 (explicit user/main go)** only.

---

# C2-PASS (same reviewer, 2026-07-14) — R-1 RESOLVED; authorization LIVE pending C3 only

**R-1 resolution verified in-file by this reviewer:** `b2_stage_d_download.sbatch:3-4` now
reads `#SBATCH --cpus-per-task=8` / `#SBATCH --mem=32G`, matching the frozen prereg Stage-D
spec (`exp-encoder-32b-b2.md:271-272`) exactly. Only the two resource lines changed (rest of
the file identical to the previously-reviewed version); `bash -n` re-run OK. All four files
independently re-hashed:

```
FINAL C2-PASS HASH SET (2026-07-14)
817a951d717be56e7329ccb894c2f6ffb1edeb85e656d91286a57b34bd35284a  scripts/slurm/b2_stage_d_download.sbatch   (R-1 fixed)
532a8a3458f84862919d625da17b3e7e33d437b465d9bde13e93a475c5a1ff1c  scripts/slurm/b2_stage_e_extract.sbatch    (unchanged)
9c312da639dba0ee8061b1bb3e22b4a4a074db1812e043763732e666ef04564c  scripts/slurm/b2_stage_t_train.sbatch      (unchanged)
56588dc1b2f492e002948e9844f5059ba4bab1a156589bc67ca75b082833eb0b  research-wiki/experiments/exp-encoder-32b-b2.md (r1, unchanged)
```

**C2-PASS.** The C2 delta-check is complete with zero open residuals: Rev-1/2/3/4 + free
strengthening landed at every cited site; all three runners are faithful minimal deltas of
their templates; Stage-D env (HF_HUB_OFFLINE=0, no --gres, no --time, hf_transfer omission)
correct; Stage-E carries both Rev-2 flags (:44-45) + expandable_segments + fail-closed serial
order + dim echoes + b2_push; Stage-T diff = exactly two hunks with config #1 = HateMM s0
(Rev-4). **The conditional execution authorization of this review is NOW LIVE, pending only
C3 (explicit user/main go).** Execution binds to the FINAL C2-PASS HASH SET above: if any of
the four files changes after this stamp, the C2-PASS is void and a re-delta-check is
required before submit. Staged single-submits and executor duties as specified in the
authorization section (D → E(+G-dims) → C → T; records, no resubmits, JobHeldUser=wait,
raw-numbers-only reporting, weights deletion only after G-dims PASS with df evidence).
