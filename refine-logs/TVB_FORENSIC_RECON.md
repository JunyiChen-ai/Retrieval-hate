# TVB — TASK-ADAPTIVE VERBALIZATION BOTTLENECK — FORENSIC RECON (zero-GPU)

**Agent:** forensic-recon (2026-07-28 adversarial wave, candidate 4 of 4) · **Date:** 2026-07-28 NZST.
**Discipline honoured.** CPU-only code inspection + read-only counts over `data/gt/*.jsonl` metadata.
**ZERO GPU / SLURM / Modal / training.** **TEST-SPLIT CONTACT: NONE** — the row counts in §1.3 are over the
union of split *manifests* (id/text/label/title keys only); no feature file, no test metric, no model was
touched. No prereg written, no job submitted, no frozen artifact mutated.

**Status of this document.** **Recon-level PRE-CLOSURE**, not a measured KILL — and an unusual one: the
premise is **measured FALSE by code inspection**, so there is no cell to kill. Nothing was run.

---

## 0. THE CELL, AND WHY IT DOES NOT EXIST

**Mechanism statement (as proposed).** The MLLM *verbalizes* the video into text; that verbalization becomes
the retrieval key; therefore the verbalization is a bottleneck that could be **task-adapted** (better
prompts, task-tuned decoding, learned verbalizers) to carry more hate-relevant signal into the key.

**The premise is false.** There is **no MLLM verbalization anywhere in the deployed key path.**

---

## 1. THE DEPLOYED KEY PATH, CODE-CONFIRMED

### 1.1 The text leg is a fixed English instruction + the raw transcript, encoded as hidden states

`src/utils/generate_VideoMLLM_embedding_HF.py:397-401`:

```python
text_prompt = (
    args.text_instruction
    + "\n" + args.title_label   + (title      if title      else args.none_placeholder)
    + "\n" + args.transcript_label + (transcript if transcript else args.none_placeholder)
)
```

with the in-file comment at `:392-396` asserting the assembled prompt "is byte-identical to the deployed
literal", and `title = item.get("title","")`, `transcript = item.get("text","")` at `:390-391`. The
instruction is a **module-level constant**, `:48-52`:

```python
TEXT_INSTRUCTION = (
    "You are analysing a short video for potentially hateful or offensive content. "
    "Considering the frames together with the provided title and transcript, "
    "summarise the targets, symbols, tone, and any harmful intent conveyed."
)
```

immediately under the section header at `:44`: **"# Fixed instructions (never sampled from the model; pure
encoder use)."** The result is fed to `_encode(...)` with `span="response"` (`:402-405`) — i.e. **Qwen hidden
states**, not generated tokens.

### 1.2 Nothing is ever sampled on this path

`grep -rn "do_sample\|\.generate(" src/` returns exactly three hits, none of them on the deployed key path:

| hit | file:line | on the key path? |
|---|---|---|
| `out_ids = model.generate(` with `do_sample=False` | `src/utils/generate_video_archive_HF.py:695,698` | **No** — this is the *archive* side-channel (§2), and it is greedy |
| comment "`predict_with_generate` is OFF in the frozen yaml, so no `.generate()` surface exists" | `src/moka/routed_lora.py:306` | No — MokA, and it says the surface does not exist |
| `do_sample_frames=False` | `src/utils/generate_VideoMLLM_embedding_molmo2_HF.py:241` | No — a **frame**-sampling flag on the Molmo2 extractor, not text sampling |

`generate_VideoMLLM_embedding_HF.py` — the deployed extractor — contains **zero** occurrences of `do_sample`
or `.generate(`. **There is no decoding to adapt.**

### 1.3 The `Title:` field is always `(none)` — verified by count

Counted this session over `data/gt/{HateMM,MHC,MHC_zh}/{train,val,test}.jsonl`, `title` key non-empty:

| dataset | gt rows | rows with a non-empty `title` |
|---|---|---|
| HateMM | 1 066 | **0** |
| MHC (EN) | 790 | **0** |
| MHC_zh | 806 | **0** |

*(Newly counted this session over split manifests only. Independently confirms F74's audit — "titles absent
in ALL splits (the `Title:` field always \"(none)\")" — and `LITSWEEP3_ZH_SPECIFIC.md:39-40`.)*

So the deployed text key is, verbatim and for every row: **one fixed English instruction constant + the
literal string `Title: (none)` + the raw `text` field.** The only per-item variation is the raw transcript
string. There is no verbalizer, no schema, no summary, no sampling.

---

## 2. THE ARCHIVE IS A SEPARATE, DEFAULT-OFF, 768-d CLIP-TEXT SIDE-CHANNEL

The one place the campaign ever *did* verbalize is the MLLM structured archive, and it is not the key path:

- `src/run_rac.py:346-352`: `--archive_feats`, `default=None`, help text ending
  **"None (default) = archive fully OFF, bit-for-bit baseline."**
- The archive is CLIP-**text**, not Qwen hidden states: `src/utils/generate_video_archive_HF.py:29` documents
  the output contract as `{"ids": [ids], "text_feats": [N,768], "labels": ...}`, encoded by
  `CLIPTextModel.from_pretrained("openai/clip-vit-large-patch14-336")` (`:911-913`, `:867`).
- Its generation is **greedy**: `do_sample=False` (`:698`).
- **No HateMM archive exists at all.** `data/Archive/` contains only `MHC/` and `MHC_zh/`; a
  `find data -path "*HateMM*" -name "*archive*"` returns nothing.

So even the archive route (a) is off by default, (b) lives in a different 768-d space from the 3584⊕3584
Qwen key, and (c) does not exist for the anchor dataset.

---

## 3. PRE-CLOSURE TABLE — EVERY REACHABLE VERSION OF TVB

| version of "task-adaptive verbalization" | status | quoted binding text (`file:line`) |
|---|---|---|
| **Steelmanned (a learned/MLLM verbalization *becomes* the key)** | **MEASURED WORSE THAN BLIND TRUNCATION** | P8/P8b/P8c. "**B (summary) HURTS EN under both protocols** (val-sel −0.023, final-epoch **−0.079**, 0/3 seeds positive)" and, decisively, "**The rent test fails in the wrong direction:** on final-epoch B (−0.079) is *worse* than the naive first-70-token control C (−0.058) — **the MLLM summary does not even beat blind truncation once trained**" (`research-wiki/EXP_p8_semantic_compression.md:124-129`; table `:118-121`). The record's own lesson: "P8 had the *strongest* no-head probe of any front … yet the trained retrieval head does WORSE on the compressed text than on the raw chunk-mean. **A passing no-head probe is necessary but not sufficient**" (`:132-135`) |
| **Reachable-a (vary the readout: layer / span / prompt wording)** | **MEASURED DEAD (F70)** | "grid R1-L24/R2-ow-L28/R3-ow-L24 vs R0 … ZH best +0.0128 … HateMM best +0.0093 … one-word readout REGRESSES HateMM (−0.056/−0.065). **ALL cells ≤ +0.020 ⇒ KS-readout-dead fired**" (`findings.jsonl` F70). Ban scope: "Extraction readout variants (hidden layer L24, one-word prompts, last-token span) over the adapted encoders on ZH/HateMM: all inside perm-null" (`directions_tried.json`, F70 entry) |
| **Reachable-b (match the instruction language to the content language)** | **MEASURED DEAD (F80), with an explicit do-not-re-propose** | "Arm-L (LoRA, primary) val-sel mean Δacc −0.0358 (**0/3**) / final −0.0112 (**0/3**); Arm-F (frozen) val-sel −0.0336 (**0/3**) / final −0.0045 (**0/3**) … Chinese prompt actively HURTS val-sel" (`findings.jsonl` F80). Ban scope: "extraction-instruction language variations (any language, any stream, either encoder arm) on MHC_zh … **do NOT re-propose prompt-language matching elsewhere** without new mechanism" (`directions_tried.json`, F80 entry) |
| **Field version (verbalize into structured schema fields and distil them)** | **FORMALLY BANNED** | P4 schema-field distillation: "**FAIL (within-noise)** … **probe PASS** (fields decodable AUC .62–.93, label-informative AUC .74–.78); train EN final −0.001, ZH +0.008 (sub-threshold) … fields real but **redundant** with the direct hateful-label supervision" (`research-wiki/CAMPAIGN_mllm_method_role.md:57`). `banned_constraints[6]` = "**P1-P5 re-proposals**" (`directions_tried.json:461`, 0-indexed) |
| **Only never-run cell: multi-prompt ENSEMBLING** | **NEVER RUN, USER-GATED, RANKED LAST** | "a **multi-PROMPT** ensemble (MetaEOL, 2402.18458) is **flagged as 'not literally covered'** [by the `cross-seed ensembles` ban] and needs a one-line user micro-ruling" (`LITSURVEY_MLLM_EMBEDDING.md:145`); ranked **7 of 7**: "**Multi-prompt ensembling** | never run | **~0 predicted** (D1; F70+F80 dead neighbors) | **Needs GPU re-extract (no banked per-prompt caches) → not worth it**" (`LITSWEEP5_COMPLETENESS.md:120`), and last in the user-gate EV ordering "… door-closers F78>F79>F76 (<3%, paper-value) > **multi-prompt (~0)**" (`:14`) |

**Note the structural point.** There is **no $0 gate** for multi-prompt ensembling, because there are no
banked per-prompt caches — it is the one TVB-adjacent cell that cannot be pre-gated for free, and its
honest prior is ~0.

---

## 4. THE GOLD-CHEATING ORACLES — ALL BELOW BAR

Three independent oracles price "what would a *perfect* verbalization buy?" All are gold-cheating upper
bounds and all sit under +0.030.

| oracle | value | source |
|---|---|---|
| **ZH best-possible transcript** (independent Whisper-large-v3 K4 re-run vs the deployed text; the 2 core errors it could recover both flipped) | **+0.0134** | `ERRPAT_MHC-ZH_2026-07-26.md:378-380`, with its own ruling at `:385`: "**ZH ASR re-channelling is arithmetically capped below the bar at $0. Do not spend GPU.**" |
| **Perfect verbalized target field** — 9-way one-hot gold target at coverage 1.0, conditional-info gate on HateMM train | **Qwen +0.0035** [0.0013, 0.0059] · **CLIP +0.0145** [0.0091, 0.0204] | `refine-logs/C3_G0COND_ORACLE_PROBE_OUT.json`, arms `oracle_target`; null control `shuffled_target` +0.0008 / −0.0005 |
| **Real (non-oracle) verbalized-attribute predictor** — Qwen-7B predicted target community | direct Δacc **−0.0019** (HateMM/CLIP); best **Fano-projected** Δacc **+0.0094** | `refine-logs/C3_REAL_PREDICTOR_PROBE.md:153` ("best real-predictor Δacc is **+0.0094** (Fano, HateMM/CLIP)"), `:103`; raw arm values re-read from `C3_REAL_PREDICTOR_PROBE_OUT.json` |

**Reading.** A *perfect* verbalized field is worth **+0.0035 to +0.0145**; a *real* one is worth **−0.0019
direct / +0.0094 projected**; and a *perfect transcript* on the dataset with the thinnest text is worth
**+0.0134**. Against a **+0.030 on ≥2 datasets** bar, the whole verbalization axis is arithmetically capped
at roughly **half the bar in its best gold-cheating corner**.

---

## 5. TWO LEDGER ERRATA THAT MUST BE PROPAGATED

Both were established by litsweep-3 (F77, commit `d4af64b`) and are re-verified here. They are recorded in
this file because several documents still repeat the superseded versions; the propagation pass is listed in
§5.3.

### 5.1 ERRATUM A — "ZH transcripts median 4 words" is a WHITESPACE ARTEFACT

> "**'ZH transcripts median 4 words' is a WHITESPACE-SPLIT ARTIFACT — false as stated.** Chinese text has no
> inter-word spaces, so `text.split()` is meaningless."  — `LITSWEEP3_ZH_SPECIFIC.md:18-19`

Measured on `data/gt/MHC_zh/{train,val,test}.jsonl` (`LITSWEEP3_ZH_SPECIFIC.md:21-25`):

| split | n | **gt-text CHARS median** (mean / max) | whitespace-"words" median | rows w/ `<em class="keyword">` |
|---|---|---|---|---|
| train | 579 | **106** (134.2 / 708) | 4 | **243/579 = 42.0 %** |
| val | 78 | **108.5** (131.4 / 343) | 3 | 34/78 |
| test | 149 | **105** (129.4 / 361) | 4 | 63/149 |

> "The deployed ZH text stream is **median ~106 Chinese characters (~50–70 words) — content-rich, not
> degenerate.**" — `:27`

**Institutional note:** the campaign's own frozen measurement code already handles this correctly —
`scripts/analysis/restrans_pregate.py:117-128` defines transcript volume as "whitespace tokens for
HateMM/MHC-EN, **characters** of the composed gt text for MHC-ZH". The artefact lives in *prose*, not in the
frozen operators. No measured result is affected.

### 5.2 ERRATUM B — the deployed ZH "transcript" is the Bilibili DESCRIPTION, not Whisper ASR

> "**The deployed ZH 'transcript' is the Bilibili DESCRIPTION/metadata, NOT the Whisper ASR.** … `gt["text"]`
> for ZH is the Bilibili search-result description (with literal `<em class="keyword">…</em>` highlight
> markup around the *un-obfuscated search keyword*, often the slur itself — **present in 42 % of train
> rows**). The genuinely short/noisy Whisper ASR lives in a **separate, non-deployed** file
> (`data/ASR/MHC_zh/*_asrK4_whisper-large-v3.jsonl`), e.g. id `BV1em4y1B7bQ` ASR = `小蜜蜂嗯嗯` while its
> deployed gt-text is a full sentence. The `<em>` keyword highlight is **baked into the current 0.8537
> floor** and inadvertently surfaces the slur." — `LITSWEEP3_ZH_SPECIFIC.md:29-37`

*(Provenance detail: `LITSWEEP3_ZH_SPECIFIC.md:30` cites the extractor at `:349-355`; in the current
revision of `src/utils/generate_VideoMLLM_embedding_HF.py` the same assembly is at **`:397-401`**. Same code,
different line numbers — this recon's citations use the current file.)*

**Why this matters beyond bookkeeping.** It relocates the ZH story twice over: (i) the ZH key is
content-rich, so "improve the ZH text" is not a live lever — the ZH wall is "78-dev val-selection noise
(+0.0246 vs +0.030) plus representation saturation (LoRA text-AUC 0.925 …), **not encoder deficiency**"
(`LITSWEEP3_ZH_SPECIFIC.md:191-194`); and (ii) it makes obfuscation-density arguments about ZH backwards —
the slur is often surfaced un-obfuscated by the markup.

### 5.3 Propagation performed 2026-07-28

| file:line (pre-edit) | claim | action |
|---|---|---|
| `refine-logs/LITSWEEP2_INPUT_FIDELITY.md:64` | table row "MHC_zh train … median **4** … **near-empty**" | row struck; full erratum box inserted beneath the table |
| `refine-logs/LITSWEEP2_INPUT_FIDELITY.md:71` | "(ii) ZH text ≈4 words means the ZH image stream carries *more relative weight*" | struck + withdrawal note |
| `refine-logs/LITSWEEP2_INPUT_FIDELITY.md:199` | "§5: ZH text is only ~4 words median, so ZH's image stream carries far more relative weight" | struck + withdrawal note |
| `refine-logs/RESOLUTION_FORENSIC_RECON.md:124-127` | "ZH transcripts are ~4 words median (`LITSWEEP2_INPUT_FIDELITY.md:64`), so the ZH **image** stream carries more relative weight" | half-bullet struck; erratum box inserted; the separately-measured 13.7× resolution headroom explicitly preserved |
| `refine-logs/ERRPAT_MHC-ZH_2026-07-26.md:372` | "The deployed transcript is already ASR-derived" | erratum note inserted; **the section's measured numbers stand** (the bigram-overlap and length comparisons were measured directly, not inferred from the premise) |
| `research-wiki/DESIGN_iter1.md:285-286` | "MHClip-ZH >0.85 is aspirational (**ASR-bound**)" | erratum note inserted |

**Already correct, no action needed** (both carry the erratum in the house provenance style):
`research-wiki/DRAFT_experiments_chapter.md:787-795` and `research-wiki/PAPER_MASTER_TABLES.md:410-419`.

**Deliberately NOT edited:** `autoresearch/goal_mllm_plus3/state/findings.jsonl` line 75 (F74) still says
"ZH transcripts median only 4 words". `findings.jsonl` is an **append-only ledger of what was found when**;
F77 (line 78) records the correction on the record, so the ledger is self-correcting and rewriting a
historical row would destroy that. Flagged here instead.

---

## 6. VERDICT

> **PRE-CLOSED. The premise is measured FALSE by code inspection: there is no MLLM verbalization in the
> deployed key path.**

- The deployed text key is a **fixed English instruction constant + `Title: (none)` + the raw transcript**,
  encoded as Qwen hidden states, with **no sampling anywhere** (§1).
- The only verbalization in the repo is the archive side-channel: **default OFF**, **768-d CLIP-text**,
  **greedy**, and **absent for HateMM** (§2).
- The steelmanned version (P8) was measured **worse than blind truncation**; the two reachable versions are
  measured dead (F70 ≤ +0.020 everywhere; F80 0/3 on both arms and both protocols, with an explicit
  do-not-re-propose); the field version is formally banned (§3).
- Every gold-cheating oracle sits at **+0.0035 to +0.0145** (perfect field) or **+0.0134** (perfect ZH
  transcript) — roughly **half the bar**, before any conversion loss (§4).
- The one never-run cell, **multi-prompt ensembling**, has **no $0 gate** (no banked per-prompt caches),
  needs a **user micro-ruling**, and is ranked **7 of 7 at ~0** by the campaign's own completeness audit (§3).

**No GPU. No prereg. The direction should be recorded as premise-refuted rather than as a candidate.**

### P(pass) estimates

| bar | estimate | reasoning |
|---|---|---|
| P(≥ +0.030 acc on ≥2 datasets) — any TVB version | **< 1 %** | §4 caps the *perfect* version at ~half the bar on the one dataset where it was priced |
| P(multi-prompt ensembling ≥ +0.030 on ≥1 dataset) | **~1 %** | LITSWEEP-5's own read is "~0 predicted"; F70/F80 are its immediate dead neighbours |
| P(any TVB version is legal without a new user ruling) | **~0 %** | the only never-run cell is explicitly micro-ruling-gated |
| Cost if the micro-ruling came back permissive | **≥ ~1 GPU-h re-extraction** | no banked per-prompt caches (`LITSWEEP5_COMPLETENESS.md:120`) |

---

## PROVENANCE

- Code read directly (read-only): `src/utils/generate_VideoMLLM_embedding_HF.py:44-52,385-405`,
  `src/run_rac.py:345-352`, `src/utils/generate_video_archive_HF.py:29,237,695-698,867,911-913`,
  `src/utils/metrics.py:262-284`, `scripts/analysis/restrans_pregate.py:117-128`.
- Records read directly: `research-wiki/EXP_p8_semantic_compression.md`,
  `research-wiki/CAMPAIGN_mllm_method_role.md:57`, `refine-logs/LITSWEEP3_ZH_SPECIFIC.md`,
  `refine-logs/LITSWEEP5_COMPLETENESS.md:14,120`, `refine-logs/LITSURVEY_MLLM_EMBEDDING.md:55,145`,
  `refine-logs/ERRPAT_MHC-ZH_2026-07-26.md:366-385`, `refine-logs/C3_REAL_PREDICTOR_PROBE.md`,
  `refine-logs/C3_G0COND_ORACLE_PROBE_OUT.json`, `refine-logs/C3_REAL_PREDICTOR_PROBE_OUT.json`.
- Ban ledger: `autoresearch/goal_mllm_plus3/state/directions_tried.json` (F70 and F80 dead entries;
  `banned_constraints:454-463`); `state/findings.jsonl` F70, F74, F77, F80, F81, F82.
- **Counted this session:** the title-coverage table in §1.3, over `data/gt/*/{train,val,test}.jsonl`
  manifests. Recon-grade (inline `python3`), not gate-grade; it is a metadata count, not a measurement
  against a bar.
- **Required statements:** ZERO GPU / SLURM / Modal / training spent by this recon; no feature file and no
  held-out test metric read or produced; no `state/` mutated by this file; no prereg, config, or frozen
  artifact touched.
