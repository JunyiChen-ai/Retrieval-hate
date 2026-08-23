# TERA Gate-0 — REGISTERED DEVIATION D-2

- **Study**: TERA-GATE0
- **Prereg**: `research-wiki/EXP_tera_gate0_prereg.md` (sha256 `f6c1ce6c652bcedd18451d4ee3a490ca2c72c603489e89c6a161855537ed6e98`)
- **Appendix**: `research-wiki/EXP_tera_gate0_impl_appendix.md` v3 (sha256 `ea158b2c23bd0a9ed8cecdbaccdecd21e97621f9a88b3db8a7c2dcbba2c42ffc`)
- **Frozen config**: `research-wiki/tera_gate0_frozen_config.json`, `payload_sha256 = 7ba80eaf697ac46bb90b30161b1726aba7ee238e73001dd832ce30dba8a1dabe`
- **Affected stage**: Gate-C blinded error-coverage audit (prereg §4), Run 1 artefact
  directory `artifacts/tera_gate0/tera-gate0-20260807T000625Z-7ba80eaf/`
- **Registered (UTC date)**: 2026-08-07
- **Registered BEFORE**: any Gate-C label was produced. At the time of writing,
  `logging/runs/gate_c_annotation/` contains no row, `gate_c_audit.jsonl` does not exist in
  the run directory, and no coverage, kappa, or msc quantity has been computed.
- **Authority**: user adjudication, 2026-08-07, comprising the original ruling plus two
  recorded revisions (DUA frame-exposure exemption widened from a single-item exemption to
  a general one; second coder changed from a Qwen model to a second Claude instance).
- **Registration basis**: prereg §12 — registered before the affected stage begins, with the
  reason and the expected directional effect stated.

---

## 1. D-2 — the Gate-C annotators are Claude Opus 5 agents, not humans

### 1.1 What the frozen protocol assumed and what is executed

Prereg §4.1/§4.2 and appendix §11 describe Gate-C as a manual audit ("C: CPU/manual audit",
§13.1) performed by annotators who see the video, the transcript and the official span
overlay. The frozen harness treats the audit as external input: `run_gate0.py:783-787`
reads `--gate-c-audit` as a JSONL file and never constrains who produced it.

**Executed instead:** every Gate-C label is produced by a Claude Opus 5 agent.

| role | `coder_id` | scope |
|---|---|---|
| primary coder | `claude-opus-5-c1` | all 133 audited items (120-cap FN draw + controls, as listed in `gate_c_items_blinded.jsonl`) |
| second coder | `claude-opus-5-c2` | the 27 double-coded items registered in `gate_c_sample.json["double_coded"]` |
| adjudicator | `claude-opus-5-adj` | only those double-coded items whose two `primary_cause` values disagree |

Each item is labelled by a **separate agent instance with no shared context**. The c2
instances receive the identical item material and the identical instruction text as c1 and
are never shown any c1 output, any c1 row, or the existence of a c1 label for that item. The
coordinating process performs no labelling judgement of any kind: it validates fields against
the frozen enumerations, computes the arithmetic field `span_video_duration_ratio` from the
coder's own intervals and the probed duration, maps the neutral handle back to `video_id`,
and concatenates rows.

### 1.2 Reason

User adjudication of 2026-08-07. The prereg does not name a human annotator pool, does not
register annotator identity as a frozen quantity, and no such pool exists for this project.
The alternative to model annotation is that Gate-C cannot be executed at all.

### 1.3 Expected directional effect on registered endpoints

Not neutral, and not claimed to be neutral. Two directional statements are registered here,
before any number exists:

1. **Reliability (`kappa >= 0.60`) is expected to be biased upward** relative to a
   human double-coding design, because the two coders are independent samples from the same
   model rather than two different annotators. See limitation (ii) in §4.
2. **Coverage of the union set is expected to be biased downward** by the audio limitation,
   because non-speech audio evidence cannot be perceived. See limitation (i) in §4.
   A downward bias on the union makes Gate-C **harder** to pass, so a pass under this
   deviation is conservative with respect to the registered 30% / CI-20% / msc-15% thresholds.

No threshold, taxonomy entry, sampling weight, seed, decision rule, or harness byte is
changed by this deviation.

---

## 2. Registered presentation specification

The frozen protocol (`annotation_protocol.json`, sha256
`62d7eecc151181d6737db28a37035a91d596eacb6c016eb4ff08d5d36e469297`) fixes
`blinding.shown = ["video", "transcript", "official span overlay"]`. A text-and-image agent
cannot be given a video stream, so "video" is instantiated as a fixed frame grid. The
instantiation is registered here rather than chosen per item.

Per audited item, and identically for c1, c2 and the adjudicator:

1. **Frames — 30 per video.** One frame per registered temporal window. Window `k`
   (`k = 0..29`) has registered boundaries `[kD/30, (k+1)D/30)` (prereg §3); the presented
   frame is taken at the window **midpoint** `t_k = (k + 0.5) * D / 30`. `D` is the probed
   media duration read from `data/gt/HateMM/hate_spans.json` — the same `duration` field the
   official span list is expressed against, so frames and spans share one time base. Frames
   are decoded from `/home/jehc223/data/HateMM/video/<video_id>.mp4`, downscaled so the
   longest side is at most 512 px, aspect ratio preserved, and presented in temporal order
   labelled by window index and by `t_k` in seconds.
2. **Transcript.** The `text` field of the item's row in `data/gt/HateMM/train.jsonl`
   (keyed by `id`). This is the transcript that is an ordinary model input for the
   deployable arms, which is the condition prereg §4.1 attaches to showing it.
3. **Official hate spans.** The `spans` list from `hate_spans.json` for that video, in
   seconds, plus the probed `duration`. Gold spans are read here under the prereg §3 carve-out
   ("Gold spans may be read only by Gate-C annotation/evaluation").
4. **Span overlay.** For each of the 30 windows, a precomputed boolean marking whether the
   window interval `[kD/30, (k+1)D/30)` intersects any official span, plus the overlapping
   seconds. This is the text-medium instantiation of the "official span overlay" the frozen
   protocol permits.

Nothing else is presented. In particular the agents receive no model score, no correctness
category (false negative / true positive / false positive), no retrieval output, no TERA
output, no arm identity, no fold identity, and no tercile.

### 2.1 Neutral handles — an interface mitigation beyond the frozen protocol

HateMM identifiers are label-bearing strings (`hate_video_403`, `non_hate_video_644`). This
leakage is already on record in the freeze notes; the frozen protocol nonetheless lists
`video_id` as a form field.

Mitigation applied here, **in addition to** the registered blinding and not in place of any
part of it: the audited items are presented under neutral handles `item_001` … `item_133`,
assigned by position in `gate_c_items_blinded.jsonl` (sha256
`a319c1733652466c85d0b7f259e56c0ea25ca792e250cf8ff565d8c238f5a4f2`), whose order is itself a
seeded permutation produced by the frozen harness (`run_gate0.py`, `default_rng(20260807)`).
No agent sees a `video_id`. The handle-to-`video_id` map is held by the coordinating process
and applied when rows are written, so the emitted `gate_c_audit.jsonl` still carries real
`video_id` values as the harness requires.

This does not remove the underlying leakage — the gold span list itself distinguishes
positives (non-empty `spans`) from controls drawn from the negative population (empty
`spans`), and that overlay is mandated by the frozen protocol. It removes only the
identifier channel.

---

## 3. Adjudication and reliability computation

- The 27 double-coded items are exactly `gate_c_sample.json["double_coded"]`
  (sha256 of that file: `d43a22975f1f485ab420dcc1e1cf798baca9aa2bcdbe2415c0dc426f40703bed`),
  read programmatically by key. No other field of that file is read by the annotation
  pipeline.
- For a double-coded item whose two `primary_cause` values **disagree**, a third independent
  Claude instance (`claude-opus-5-adj`) is given the same item material plus both raw coder
  rows verbatim, and emits an additional row with `adjudicated: true`.
- **Original rows are never rewritten, deleted, or marked `superseded`.** Both coder rows
  remain in `gate_c_audit.jsonl` in the order c1, then c2, then any adjudicated row. This
  ordering is load-bearing for the frozen harness: `run_gate0.py:794-800` builds the kappa
  pair for a video from the **first two rows** of that video in file order, and resolves the
  video's mechanism set from the adjudicated row when one exists. c1-then-c2 ordering
  therefore makes the reported kappa the c1/c2 pair, and adjudication overrides the mechanism
  set without disturbing it.
- **No reliability, coverage, or decision quantity is computed on the annotation side.**
  Raw agreement, Cohen's kappa, weighted/unweighted coverage, the bootstrap CI and the
  `msc_subset` are produced only by the frozen Run 2 harness from the submitted
  `gate_c_audit.jsonl`. The coordinating process does not count agreements, does not
  tabulate causes, and reports only the number of adjudicated items.

---

## 4. Registered limitations

Both are registered before any label exists and are reported with the Gate-C result whatever
that result is.

**(i) No audio perception.** The agents receive frames and a transcript. They cannot hear the
audio track. Non-speech audio evidence — music, chants, sirens, laughter, sound effects,
prosody — is therefore invisible except where it is legible in a frame or in the transcript.
Consequences:

- `audio_nonspeech` in `required_modalities` will be under-recorded;
- `cross_modal` as a primary or secondary cause is correspondingly **under-counted**, since a
  visual+non-speech-audio combination will typically be recorded as visual-only;
- `cross_modal` is one of the three members of the registered union set
  (`short_localized`, `multi_segment_complementary`, `cross_modal`), so the union coverage
  estimate is biased **downward**, i.e. against passing Gate-C. The direction is conservative
  and is accepted on that basis.
- The `annotation_ambiguity_or_noise <= 0.25` check may be biased upward (items whose only
  evidence is audio may be recorded as ambiguous), which is also in the fail-safe direction.

**(ii) Kappa measures same-model independent reproduction, not inter-annotator agreement.**
`claude-opus-5-c1` and `claude-opus-5-c2` are independent, context-isolated instances of the
same model reading the same material under the same instructions. Their agreement quantifies
whether that model reproduces its own primary-cause decision across independent runs. It is
**not** an estimate of agreement between two different annotators and must not be reported as
one. The registered `kappa >= 0.60` check is evaluated on this quantity because the prereg
fixes the check on "double-coded primary-cause Cohen's kappa" without specifying annotator
provenance; the reinterpretation is recorded here so that the number is never quoted as
human inter-rater reliability.

**(iii) Scope note.** Consequently, any published claim from Gate-C must state that the error
taxonomy was applied by an LLM annotator, and must not describe the audit as human-coded.

---

## 5. Byte-integrity statements

- `artifacts/tera_gate0/tera-gate0-20260807T000625Z-7ba80eaf/annotation_protocol.json`
  is **not modified**. Verified sha256 at the time of writing:
  `62d7eecc151181d6737db28a37035a91d596eacb6c016eb4ff08d5d36e469297`. This is the value each
  emitted row carries in its `protocol_sha256` field.
- `gate_c_items_blinded.jsonl` and `gate_c_sample.json` are **read only**.
- `scripts/tera_gate0/*.py` are **not modified**; the audit is external input to the frozen
  harness and no code path is added, removed or adapted for it.
- `research-wiki/EXP_tera_gate0_prereg.md` and `…_impl_appendix.md` are **not edited**, for
  the reason given in D-1 §3: their digests are embedded in `payload_sha256` and any edit
  would change the `run_id` and trigger `HALT_CONFIG_HASH_MISMATCH`. The documentary
  back-fill into the prereg's `REGISTERED DEVIATIONS / ERRATA` subsection is deferred to
  campaign close-out. **Until then, this file is the authoritative timestamp for D-2.**

---

## 6. Emitted row contract

Rows conform to `form_version = "tera-gate0-gatec/1"` and to the `form_fields` list in the
frozen protocol: `video_id`, `coder_id`, `primary_cause`, `secondary_causes`,
`minimal_sufficient_intervals`, `required_modalities`, `single_interval_sufficient`,
`span_video_duration_ratio`, `confidence`, `notes`, `protocol_sha256`, `form_version`; plus
`adjudicated: true` on adjudication rows only. `primary_cause` is one of the eight frozen
taxonomy strings; `secondary_causes` is a subset of the same eight excluding the primary;
`required_modalities` is a non-empty subset of the frozen five;
`single_interval_sufficient` is boolean; `confidence` is `high`/`medium`/`low`.
`span_video_duration_ratio` is computed by the coordinating process as the total measure of
the union of the coder's `minimal_sufficient_intervals` divided by the probed duration `D`,
so that it cannot be a free-hand estimate.

Per-coder raw rows are retained at
`logging/runs/gate_c_annotation/claude_c1_rows.jsonl` and
`logging/runs/gate_c_annotation/claude_c2_rows.jsonl`; the assembled audit file submitted to
Run 2 is
`artifacts/tera_gate0/tera-gate0-20260807T000625Z-7ba80eaf/gate_c_audit.jsonl`.
