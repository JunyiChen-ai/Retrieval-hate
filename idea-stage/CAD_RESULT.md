# CAD — counterfactual data augmentation on the HateMM train split: RESULTS

**Verdict: KILL.** Adding 231 counterfactual rows — each a training hate video whose
transcript has been minimally rewritten to remove the identity attack, relabelled 0 —
costs **−0.0507** test macro-F1 against the baseline, on **0 of 3** seeds positive. The
control that adds the *same number* of label-0 rows on the *same* hate-video visuals but
with random benign text loses only **−0.0062**. So the counterfactual structure is not
merely useless here: it is **−0.0444 worse than adding arbitrary negatives**.

Design, prompt, quality gates and the decision rule were frozen in
`idea-stage/CAD_FREEZE.md` (commit `1dc1891`) **before** the frozen generation run, before
any feature was encoded, and before any candidate number existed. The 9 training runs were
a single background submission; no re-run, no tuning after any number was seen.

API spend: **¥0.42** (273,697 input + 99,847 output tokens, qwen-plus list price) against
a ¥5 ceiling. Wall clock: 5.0 min generation, 12.6 min encoding (232 videos, GPU shared
with another tenant), 102 s for all 9 head runs.

---

## 1. Main table — test (215 videos), classifier head, val-selected epoch

Selection rule imported verbatim from `scripts/rgcl_ablation_analyze.py`: best epoch ≥
warmup(5) by (dev head acc, dev head roc); report test macro-F1 at that epoch. Selection
is dev-only; test labels were never read before these 9 runs finished.

| arm | train rows | test macro-F1 (mean ± std) | per-seed | test ROC | val macro-F1 |
|---|---|---|---|---|---|
| **A0** *(baseline)* | 744 | **0.8679 ± 0.0036** | 0.8638 0.8704 0.8694 | 0.9317 | 0.8693 |
| **CAD** *(candidate)* | 975 | 0.8172 ± 0.0283 | 0.8107 0.8482 0.7927 | 0.9278 | 0.8008 |
| **CTRLRAND** *(control)* | 975 | 0.8616 ± 0.0063 | 0.8545 0.8666 0.8638 | 0.9249 | 0.8388 |

**Baseline reproduction.** A0 here is bit-identical to the TEXT_MERGE A0 cache and
reproduces it exactly, seed for seed: 0.8638 / 0.8704 / 0.8694 → 0.8679 ± 0.0036, the same
three numbers as `TEXT_MERGE_RESULT.md` §2. The comparison is therefore clean.

### Paired-by-seed deltas (test macro-F1)

| comparison | mean | per-seed | seeds positive |
|---|---|---|---|
| **CAD − A0** (primary) | **−0.0507** | −0.0531 −0.0222 −0.0767 | **0/3** |
| CTRLRAND − A0 | −0.0062 | −0.0093 −0.0038 −0.0056 | 0/3 |
| CAD − CTRLRAND | −0.0444 | −0.0438 −0.0184 −0.0711 | 0/3 |

### Frozen verdict (CAD_FREEZE.md §5)

| # | clause | requirement | measured | pass? |
|---|---|---|---|---|
| 1 | `mean(CAD − A0) ≥ +0.005` | +0.005 | **−0.0507** | ✗ |
| 2 | `CAD − A0 > 0` on 3/3 seeds | 3/3 | **0/3** | ✗ |
| 3 | `Δ_CTRL < 0.5 · Δ_CAD` | — | both negative | ✗ |

**KILL** on all three clauses.

---

## 2. What the control says

CTRLRAND holds everything constant except the *text* of the added rows: same 231 image
features copied from the same 231 hate videos, same label 0, same count, same donor-free
draw. Its text comes from randomly drawn distinct non-hate train videos instead of the
counterfactual rewrite.

CTRLRAND loses −0.0062 — near noise. CAD loses −0.0507. The counterfactual pairing is
**8× more damaging** than the generic version of the same intervention. The frozen "gain
came from adding negatives, not from structure" branch does not apply: neither arm gains.
What the data shows is the opposite of the hoped-for result — the minimal-pair structure is
the *harmful* ingredient.

---

## 3. Where the loss comes from: precision up, recall collapses

Recomputed offline from each arm's selected checkpoint and cross-checked against the
trainlog's own test macro-F1 (all 9 match to < 5e-4).

| arm | hate precision | hate recall | false negatives (of 86 hate) | false positives (of 129 non-hate) |
|---|---|---|---|---|
| **A0** | 0.836 | 0.849 | 13.0 | 14.3 |
| **CAD** | **0.934** | **0.640** | **31.0** | 4.0 |
| CTRLRAND | 0.904 | 0.760 | 20.7 | 7.0 |

(means over 3 seeds)

The augmented head becomes far more conservative: it almost stops producing false positives
(14.3 → 4.0) and misses 2.4× as many hate videos (13.0 → 31.0). On a balanced macro-F1
that trade is heavily net-negative.

**Test ROC barely moves** (0.9317 → 0.9278, −0.004) while macro-F1 drops 0.051. The
augmented rows do not destroy the representation's ability to *rank* hate above non-hate;
they move the decision boundary. The head is being taught, 231 times, that a
transcript-and-picture combination that looks almost exactly like a known hate video is
label 0, and its only available response is to raise the bar for calling anything hate.

This is a real mechanism, not a val/test transfer artifact: **validation macro-F1 drops
too** (0.8693 → 0.8008), by more than test does. The augmentation hurts everywhere.

---

## 4. Generation and gate accounting

| stage | count |
|---|---|
| HateMM train rows with label 1 | 298 |
| G0 — transcript < 40 non-whitespace chars, skipped before the API | 14 |
| eligible, sent to qwen-plus | 284 |
| G1 — provider input moderation refused | 3 |
| G1 — API error / unparseable response | 0 |
| G2 — length ratio outside [0.40, 1.20] | 0 |
| G3 — rewrite still matched the identity-attack word list | 25 |
| G4 — rewrite identical to the original | 24 |
| G5 — source video's frames failed to decode (`hate_video_95`) | 1 |
| **augmented rows actually trained on** | **231** |

Train split 744 → 975 rows. Accepted rows: length ratio min 0.456 / median 0.981 / max
1.029; self-reported edit spans min 0 / median 7 / max 47.

**One in six eligible rewrites failed a gate for a quality reason (49 of 284 by G3+G4).**
That number is the honest headline about the generator: a cheap text LLM asked to sanitise
a noisy ASR hate transcript leaves a detectable slur in 9% of cases and refuses to edit at
all in another 8%.

---

## 5. Rewrite examples (three, spread over the edit-count distribution)

**`hate_video_13`** (2 edits, ratio 0.99) — a clean minimal edit:

> ORIGINAL: … I want to go go better to you the sorrow that's all around the BJ **hateco**
> just from the Je O town. …
>
> REWRITTEN: … I want to go go better to you the sorrow that's all around the BJ just from
> the Je O town. …

**`hate_video_165`** (7 edits, ratio 1.00) — a **partial** edit, and the failure mode that
matters:

> ORIGINAL: … we're going to have a look at the two **aundins** in a McDonald's having a
> epic **chimp out**. And this is a **chimp out** to end all **chimp outs** … so let's see
> how these **Waekands** act in this. …
>
> REWRITTEN: … we're going to have a look at the two **people** in a McDonald's having a
> epic **chimp out**. And this is a **chimp out** to end all **chimp outs** … so let's see
> how these **people** act in this. …

The explicit slurs are replaced; the dehumanising "chimp out" framing — the actual hate —
survives untouched, and the word list did not catch it. This row entered training labelled
**0**.

**`hate_video_344`** (47 edits, ratio 0.93) — heavy editing that still leaks:

> ORIGINAL: … What's up **Faaggots** seems like **nigger** everything these days … **nigger**
> rooms, **Nigger** bitch, booty, **nigger** bones. **Nickckers, nicker** actors, **Nicker**
> MCs … 
>
> REWRITTEN: … What's up seems like everything these days … rooms, bitch, booty, bones.
> **Nickckers, nicker** actors, **Nicker** MCs …

The correctly spelled slurs are removed; the ASR-garbled spellings ("Nickers", "nicker")
are not. Again labelled **0**.

---

## 6. Why it failed — and the gold-label risk stated in advance

CAD_FREEZE.md §1 flagged one risk before the run: an augmented row keeps the original
video's image feature, so if a video's hate is carried by the picture rather than by speech,
that row's gold label 0 is wrong. The results are consistent with that risk being real and
with a second one the examples make visible:

1. **The rewrite is often incomplete.** 49 of 284 were caught by the gates; §5 shows two
   accepted rows that still carry the attack in a form the word list cannot see. Every such
   row is a training example that says "this hateful content is not hateful". This matches
   the repo's own earlier measurement (see §7): when an MLLM was asked to judge its own
   sanitised rewrites, only ~50% were judged benign.
2. **Even a perfect rewrite only fixes one channel.** The image feature is untouched by
   construction, and OCR-borne on-screen text is inside the picture. A video whose hate is
   in the burned-in caption is still hateful after a perfect transcript rewrite.

Both failure modes push in the same direction — they insert label-0 rows that a correct
model should call hate — and both are consistent with the observed collapse in recall.

The result does **not** license the conclusion "counterfactual augmentation cannot work for
video". It licenses: *transcript-only* counterfactual augmentation, with a cheap text LLM
and a word-list gate, on this frozen-feature grid, loses half a point of macro-F1 and does
so by destroying recall. A version that attributed the hate to a channel first, edited that
channel, and re-encoded the picture accordingly is untested — and is the direction the
novelty search below says is actually unoccupied.

---

## 7. Differentiation / novelty self-check

Search done 2026-08-13 (web search, English only; no citation-graph sweep). Question: has
anyone applied counterfactual data augmentation — minimally edit a hateful example so it
becomes non-hateful, add it as a training row with a flipped label — to hateful **video**
detection?

**Text domain: occupied.**
- Kaushik, Hovy & Lipton, ICLR 2020, *Learning the Difference that Makes a Difference with
  Counterfactually-Augmented Data* — crowdworkers minimally edit sentiment/NLI examples to
  flip the gold label. The canonical CAD definition.
- Sen, Samory, Wagner & Augenstein, NAACL 2022, *Counterfactually Augmented Data and
  Unintended Bias: The Case of Sexism and Hate Speech Detection* — CAD applied directly to
  hate/sexism; finds construct-driven CAD raises false positives on benign uses of identity
  terms. The standing warning for any hate-domain CAD.
- Sen et al., arXiv 2311.01270 (2023), *People Make Better Edits* — measures LLM-generated
  CAD against human edits for harmful-language detection; humans win. The closest precedent
  for "let an LLM do the minimal edit", which is exactly what this experiment did, and its
  finding is consistent with the 49/284 gate-failure rate measured here.
- Vidgen et al. ACL 2021 (*Learning from the Worst*), Gardner et al. 2020 contrast sets, and
  HateCheck (Röttger et al. 2021) are the usual placeholders. HateCheck is a diagnostic test
  suite, not training augmentation, so it is a weaker prior-art match than usually assumed.

**Meme / multimodal: newly occupied.**
- Kiela et al., NeurIPS 2020, Hateful Memes Challenge — benign confounders are literally
  minimal replacements that flip hateful → non-hateful, but hand-built as dataset
  *evaluation* structure, not a generative augmentation method. Partial match.
- Singh, Jaidka & Mukerjee, arXiv 2508.11808 (2025), *Labels or Input? Rethinking
  Augmentation in Multimodal Hate Detection* — an LLM+VLM pipeline isolates the hateful
  modality and rewrites it, producing ~2.5k counterfactually neutral memes added as
  label-flipped training rows. The closest true methodological hit. Memes only; no video.

**Video domain: no hit found.** ImpliHateVid (ACL 2025) uses two-stage contrastive learning;
HateClipSeg (ACM MM 2025) is segment annotation; *Cross-Modal Transfer from Memes to Videos*
(arXiv 2501.15438) shares the data-scarcity motivation but transfers meme data rather than
editing counterfactuals.

**The nearest prior work is inside this repository, and it is also a kill.**
`research-wiki/EXP_p5_counterfactual_negs.md` (2026-07) generated MLLM "sanitized
counterfactual twins" of each train positive on MHC / MHC_zh in CLIP space and used the twin
as one extra **hard negative in the contrastive loss** — never as a labelled training row.
It was killed twice: its quality gate (does the MLLM judge its own sanitised text benign?)
passed on only 0.503 (EN) / 0.337 (ZH) of rewrites, and even the verified-clean subset lost
−0.027 test macro-F1 on EN with 0/3 seeds positive, with a random-pairing control performing
the same. The present experiment is a genuinely different injection point — a labelled row
inside the classification loss and the retrieval bank, rather than a repulsion target in the
triplet loss — on a different dataset and a different feature space, and it reproduces both
of P5's qualitative findings: the sanitisation is unreliable, and the specific counterfactual
pairing buys nothing over a random control. Here it is in fact considerably *worse* than the
random control, which P5 did not observe.

**Honest position.** Counterfactual augmentation is not novel in the abstract — standard in
text, and since 2025 done for memes. The video setting is unoccupied, but this experiment is
not evidence that it is a good place to be. If the direction is revisited, the case has to be
about the multi-channel attribution-then-edit problem (which channel carries the hate, edit
that one, re-encode it), not about porting text-domain CAD to a new dataset, and it has to
answer Sen et al. 2022 on the false-positive/recall trade — which is precisely the trade that
killed this run.

Coverage limits: web search only, English only, no Semantic Scholar / citation-graph sweep,
no forward-citation check on Singh et al. 2025.

---

## 8. Artifacts

| item | path |
|---|---|
| Freeze (design, prompt, gates, decision rule) | `idea-stage/CAD_FREEZE.md` (commit `1dc1891`) |
| Prompt bank (`V2` = frozen) | `idea-stage/cad/prompts_cad.py` |
| Generation | `idea-stage/cad/cadgen.py` → `rewrites_train_hate.jsonl` |
| Quality gates + word list | `idea-stage/cad/gates.py` → `accepted.json` |
| Encoding + arm assembly | `idea-stage/cad/build_cad_feats.py` → `assemble_meta.json` |
| Arm caches | `data/CLIP_Embedding/HateMM/{train,dev_seen,test_seen}_CAD-{A0,CAD,CTRLRAND}.pt` |
| Training grid (9 runs, one submission) | `idea-stage/cad/run_arms.sh` |
| Frozen readout + verdict | `idea-stage/cad/analyze.py` → `idea-stage/cad/results.json` |
| Gate accounting + examples | `idea-stage/cad/report_examples.py` |
| Driver / logs | `idea-stage/cad/run_all.sh`, `logging/runs/cad/run.log`, `run.pid` |

Red lines held: test split is a byte-identical A0 copy in all three arms and its labels were
never read before the grid finished; the decision rule was fixed at commit `1dc1891`; the 9
runs were one submission; the DashScope key appears in no file, log or report.
