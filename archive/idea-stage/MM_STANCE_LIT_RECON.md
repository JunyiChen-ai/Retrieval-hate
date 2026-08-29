# MM_STANCE_LIT_RECON — multimodal stance detection as a field, and what of it we can actually use

**Date** 2026-08-14 · **Type** literature reconnaissance only (read-only + web; no code, no experiments, no GPU)

**Scope.** This file covers **multimodal stance detection itself** (image-text and video), 2022–2026.
It deliberately does **not** re-cover the stance × hate-speech intersection — that is
`idea-stage/STANCE_LIT_RECON.md` (HateCheck F20/F21, `2404.01651` use-vs-mention, `2206.06423`
stance pretraining, `2406.00020` QueerReclaimLex, `2210.00910` hypothesis engineering, …). Where a
finding here contradicts or sharpens something in that file, it is flagged.

**Our situation, for the reader who arrives here cold.** Hateful-video detection; 45.4 % of
detection errors are stance confusion (criticism / quotation / news reportage judged as the author
asserting hate). Three rounds of zero-shot MLLM stance judgment are dead:
direct 5-way classification 0.257, content-masked 0.375, symmetric 2-way comparison 0.469 against a
0.50 chance line (`idea-stage/CONTRAST_STANCE_RESULT.md`). Our stance field is
`{endorses, condemns, quotes_mentions, reports, no_hate_content}`; the model emitted
`quotes_mentions` and `reports` **zero times** in round 1. Diagnosed cause: a safety-training prior
— offensive content on screen ⇒ "the author is asserting it".

---

## 1. One-page conclusion

**Is there anything portable? Yes, but much less than the field's size suggests, and none of it is
a stance *model* we can lift.**

Multimodal stance detection is a real, active, top-venue line: **Findings of ACL 2024** (MMSD +
TMPT), **ACM MM 2024** (MmMtCSD + MLLM-SD), **EMNLP 2025 Main** (T-MAD), **Findings of EMNLP 2025**
(CLIMATEMEMES), **ACL 2026 Main** (MM-StanceDet), plus workshop/preprint video work (MultiClimate,
TikStance, two-stage climate). But three structural facts close the obvious door:

1. **It is a different object.** Every one of these tasks is *target-conditioned*: stance toward a
   **named entity supplied at inference** (Trump, Biden, chloroquine, Tesla, Bitcoin, "climate
   change"), labels `Favor / Against / Neutral`. Ours is the speaker's **relation to the hateful
   material he is presenting** — no named target, and the interesting distinction
   (`quotes_mentions` / `reports` vs `endorses`) does not exist in any of their label sets.
   The `Unrelated` label in MWTWT is topical irrelevance, not mention.
2. **The models do not transfer, and this is measured.** T-MAD (EMNLP 2025 Main, Table 2) reports
   the cross-target ("zero-shot") setting on the same corpora: **TMPT falls from 55.4 / 61.6
   in-target to 31.7 / 32.7 macro-F1** on MTSE-DT/JB, and from 55.7 / 46.8 to 32.2 / 26.5 on MTWQ,
   while zero-shot **GPT-4-Vision holds at 72.7 / 71.3** on the same cells. A supervised multimodal
   stance head does not even survive a change of *target inside the same corpus*; it will not
   survive a change of task, domain and modality to hateful video.
3. **The data is largely not obtainable.** MMSD — the field's main benchmark, 17,544 items — ships
   **tweet IDs only, no images** (Apache-2.0 code, "researchers are only permitted to publish Post
   IDs"); rehydrating X images in 2026 is not a realistic plan. MmMtCSD's public release is
   unverified. TikStance ships no baselines and no verified public release. The exceptions are
   **MultiClimate** (CC-licensed YouTube, frames + transcripts actually in the repo) and
   **CLIMATEMEMES** (1,184 Reddit memes, `github.com/mainlp/ClimateMemes`).

**Therefore: exclude the whole "train a stance model on their data and transfer it" plan.** It
fails on labels, on transfer, and on data availability independently — three separate kills.

**The two things that are worth taking, in priority order.**

> **Path A (recommended, near-zero cost): use MultiClimate as a capability gate on our stance
> reasoner, before building anything.**
> MultiClimate is the only downloadable **video** stance corpus with human labels: 100 CC YouTube
> videos, 4,209 frame-transcript pairs, `Support / Neutral / Oppose`, Cohen's κ = 0.703, and a
> published human-agreement ceiling (annotator accuracy 0.826 / weighted F1 0.823). A 100M-parameter
> BERT+ViT fusion reaches **0.747 / 0.749**; zero-shot Llama3-8B and Gemma2-9B reach **≈ 0.48**.
> That gives us a calibrated, in-modality, out-of-domain test of the *perception* step: can our
> reasoner read a stance off a frame+transcript pair at all, when the stance is overt and the target
> is named? If it cannot clear ~0.75 there, the record it emits about our hateful videos is noise
> and no downstream head will fix it. **Preconditions:** none — one MLLM pass over 4,209 pairs, no
> annotation, no training, no GPU beyond inference. **What it cannot do:** MultiClimate contains no
> use-vs-mention distinction, so passing it proves perception, not the distinction we need. It is a
> necessary-condition gate, not evidence of the mechanism.
>
> **Path B (the only positive wiring result in this literature, and it is in our shape):
> frozen LLM → text records → small trained fusion head.**
> `2509.08024` (preprint, unpublished — flagged) on MultiClimate: Mistral-7B produces a summary of
> the transcript, a captioner produces a *domain-aware* caption of the frame, both go into a
> BERT-style joint encoder alongside a ViT branch, and a linear head classifies. Result **76.2 %
> accuracy / 0.762 F1 vs 73.1 / 0.732 for BERT+ViT** and 71.2 / 0.684 for MemeCLIP — **+3.1 points
> over a strong trained fusion baseline, 202M parameters, 0.043 s/item.** This is the same
> topology we already chose (frozen reasoner emits typed text → trained consumer), now with an
> in-modality (video) precedent at the same +1–3 point magnitude our text analogues predicted.
> **Preconditions:** none new; it strengthens the design we already have rather than replacing it.
> It is also a *cheap mandatory baseline*: the LLM-summary + caption channel with **no stance
> field** is exactly the ablation that tells us whether our gain comes from stance or from merely
> putting LLM text in front of the encoder.

**One correction to our framing that this sweep forces.** We cannot claim "zero-shot LLM stance
judgment is known to be weak" as background. On **text** stance benchmarks it is strong —
ChatGPT reaches macro-F1 78.0 on SemEval-2016 and 79–86 on P-Stance (`2304.03087`; contamination
caveat below), i.e. at or above supervised SOTA. The zero-shot deficit appears only when the input
is **multimodal** (MMSD: GPT-4V 37.6–72.8 vs TMPT 60–77 in-target; MultiClimate: ≈0.48 vs 0.747) or
when the content is **hate-adjacent**. Our 0.257 is therefore *anomalously* low relative to what the
stance literature predicts, and that anomaly is itself the finding: it points at the
offensive-content prior, not at stance being intrinsically hard. Use the multimodal and
hate-specific citations for background; do not use the generic "LLMs are bad at stance" claim.

> **A third, smaller item worth knowing about: PrideMM** (`2409.14703`, EMNLP 2024 Main, public).
> 5,063 memes carrying a **hate label and a stance label on the same item** — the only public
> multimodal corpus where both exist. It is the one place the wiring question "does adding a stance
> channel help a hate head?" can be asked with zero new annotation. Two caveats keep it off the
> main path: its stance is *topical* (support/oppose the LGBTQ+ movement), which correlates with the
> hate label and so makes any gain near-circular; and its four tasks are **parallel heads over the
> same frozen CLIP features**, i.e. stance is never actually fed to the hate head. Cheap, worth one
> run, not evidence for the mechanism.

**One number that should move our own pre-registered gate.** `IDEA_REPORT` requires the reasoner's
stance labels to clear **macro-F1 ≥ 0.80 per corpus** before a head is trained. In this literature,
a purpose-built supervised model with frozen CLIP and trained adapters reaches **62.00 % accuracy /
0.8011 AUROC** on 3-way stance over 5,063 memes (MemeCLIP on PrideMM, EMNLP 2024 Main), and the best
video-stance system reaches **0.747–0.762** against a human-annotator ceiling of 0.826 (MultiClimate).
**Nobody in multimodal stance is anywhere near 0.80 macro-F1 with supervision**, on an easier
task than ours. Our bar is not merely optimistic (as `STANCE_LIT_RECON.md` §5 already suspected on
reliability grounds) — it is above the published state of the art for the perception step. Either
soften it to a graded/soft target or accept that the gate will kill the direction on a bar no
existing system meets.

**On the "is there a corpus we could train a use-vs-mention classifier on" question (§7): one
genuine find, and it is behind a paywall.** The LDC Belief-and-Sentiment corpus (**BeSt**,
`LDC2023T13`) carries a label `ROB` defined as *"the writer reports another source's belief without
revealing her own"* — literally our distinction — with **10,777 English instances** (plus 7,524 ZH
and 8,822 ES). FactBank (`LDC2009T23`) annotates factuality **per (event, source) pair** with an
explicit `AUTHOR` pseudo-source. Both are LDC-paid; **the first thing to check is whether the
university already holds an LDC membership**, because if it does, this is the cheapest real
supervision in either sweep. Free fallbacks, in order: reconstruct ~1,826 use/mention pairs by rule
from the CONAN family we already hold (this is exactly how `2404.01651` built its own set — it
releases **no data**, only notebooks); or train a quote/attribution span tagger on
PolNeAR + DirectQuote + PDNC (~70k free English annotations). **Both fallbacks carry a published
wrong-sign risk** (`2404.01651` Table 7: counterspeech *with* quotation marks is misclassified more
often, 28.57 % vs 7.23 % FPR), which must be argued against before any quote feature is built.
Multimodal use-vs-mention supervision does not exist, confirmed twice independently.

**And the gap stays open.** No paper found — in stance or in hate — runs the controlled
manipulation we ran: fixed offensive content, varied speaker stance, measuring whether the model's
judgment tracks the speaker. `2404.01651` is the nearest and it is text-only, binary
harmful/not-harmful, and not framed as stance. The multimodal stance field does not model
use-vs-mention at all (§4). That slot is unoccupied.

---

## 2. Work list — multimodal stance detection, 2022–2026

Venue column: **verified** unless marked. "MMSD" = the five Liang et al. Twitter datasets.

| # | work | venue (verified) | modality / data | task + labels | method in one line | headline numbers |
|---|---|---|---|---|---|---|
| 1 | **Multi-modal Stance Detection: New Datasets and Model** — Liang, Li, Zhao, Gui, Yang, Yu, Wong, Xu · `2402.14298` | **Findings of ACL 2024** (`2024.findings-acl.736`) | Twitter **text + image**; 5 datasets, **17,544 items** total | target-conditioned stance. MTSE (Trump 1,647 / Biden 1,260), MCCQ (chloroquine 1,355), MWTWT (5 mergers, 899–2,986), MRUC (Russia 1,110 / Ukraine 1,081), MTWQ (China 1,397 / Taiwan 1,928). Labels `Favor/Against/Neutral`; MWTWT `Support/Refute/Comment/Unrelated`. Annotator Cohen κ 0.703 / 0.689 / 0.729 / 0.752 / 0.691 | **TMPT**: frozen BERT-base + frozen ViT-base, 7 learnable visual prompt tokens, per-target textual prompt "The stance on [TARGET] is:", concat → FC+softmax | in-target macro-F1 TMPT 60.84 / 67.67 / 77.59 / 75.76 / 67.59 (paper's own per-dataset figures); zero-shot GPT-4 49.81–70.46, GPT-4V **37.61–72.82** |
| 2 | **Multimodal Multi-turn Conversation Stance Detection (MmMtCSD)** — Niu et al. · `2409.00597` | **ACM MM 2024** (`10.1145/3664647.3681416`) | **Reddit** text + images, conversation trees; **21,340** annotated instances (66 % vision-related), depth 1–6 | stance toward Tesla (6,300) / Bitcoin (8,148) / Post-T (6,892); `Favor 48.4 % / Against 25.8 % / None 25.8 %` | **MLLM-SD**: LLaMA2-chat-7B, frozen ViT @448², GPT-4V captions as text, **LoRA r=64 on q/v**, one-shot CoT prompt | F1-avg 64.89 / 71.28 / 79.40 vs GPT-4V 61.31 / 61.28 / 70.04, TMPT 59.72 / 62.90 / 70.79, LLaMA-70b 60.35 / 61.75 / 64.99 |
| 3 | **T-MAD: Target-driven Multimodal Alignment for Stance Detection** — Zhang, Zhang, Cheng, Xu | **EMNLP 2025 Main** (`2025.emnlp-main.30`) | MMSD + MultiClimate | same as #1/#6, **plus an explicit zero-shot (unseen-target) split** | RoBERTa/BERT + ViT, target-driven alignment, **InfoNCE-estimated mutual information → per-modality relevance weights**, 5 steps of target-queried multi-head attention, FC head; `+CWVF` fuses an MLLM vote by confidence | in-target macro-F1 T-MAD 71.1–86.2; **zero-shot: TMPT 31.7 / 32.7 on MTSE-DT/JB vs GPT-4V 72.7 / 71.3 vs T-MAD 58.1 / 62.8**; MultiClimate zero-shot T-MAD+CWVF 78.0 acc / 80.8 F1 |
| 4 | **MM-StanceDet: Retrieval-Augmented Multi-modal Multi-agent Stance Detection** · `2604.27934` | **ACL 2026 Main** — *author-stated on the arXiv page; not independently confirmed against the ACL 2026 accepted list* | MMSD (all five) | same | **gpt-4o-mini** agents: text-analysis (incl. sarcasm/irony), image-analysis, **modality-conflict**, 3 debaters (support/oppose/neutral), adjudicator with self-reflection; RAG for context | in-target macro-F1 e.g. MTSE 70.12, MWTWT 71.93, MRUC 64.02; claims SOTA over GPT-4V and TMPT (per-cell table not fully extracted) |
| 5 | **MIND: Meta-cognitive Intuitive-reflective Network for Dual-reasoning** · `2511.06057` (v1 was titled *ReMoD: Rethinking Modality Contribution…*) | **preprint** (no venue field) | MMSD | same | dual-process: fast hypothesis from Modality/Semantic "experience pools", then Modality-CoT + Semantic-CoT reflection that rewrites the pools during training | "significantly outperforms most baseline models" — **no extracted numbers; UNVERIFIED** |
| 6 | **MultiClimate: Multimodal Stance Detection on Climate Change Videos** — Wang et al. · `2409.18346` | **EMNLP 2024 workshop NLP4PI** (`2024.nlp4pi-1.27`) — *workshop, not main* | **video**: 100 CC-licensed YouTube videos → **4,209 frame–transcript pairs** (frame every 5 s) | `Support 1,847 / Neutral 1,192 / Oppose 1,170`; 2 annotators, **Cohen κ 0.703**, human accuracy 0.826 / wF1 0.823 | benchmark study: BERT, ResNet50, ViT, BERT+ViT, BERT+ResNet50, CLIP, BLIP, IDEFICS-9B (ZS + FT), Llama3-8B, Gemma2-9B | **BERT+ViT 0.747 acc / 0.749 F1** — beats CLIP, BLIP, 9B IDEFICS; BERT alone 0.705; **zero-shot Llama3 / Gemma2 ≈ 0.48** |
| 7 | **Two Stage Context Learning with LLMs for Multimodal Stance Detection on Climate Change** — Pangtey, Kabde, Dar, Kumar · `2509.08024` | **arXiv preprint only** (no venue) | MultiClimate (3,372/417/420 split) | same as #6 | Mistral-7B summarises transcript; domain-aware image captioner; joint text module (BERT-style, multi-head attn over text+caption) + ViT branch; concat → linear head; 202M params | **76.2 % acc / 0.762 F1** vs BERT+ViT 73.1/0.732, BERT+ResNet50 72.6/0.723, MemeCLIP 71.2/0.684 |
| 8 | **CLIMATEMEMES: What Media Frames Reveal About Stance** — Zhou, Peng, Luebke, Haßler, Haim, Mohammad, Plank · `2505.16592` | **Findings of EMNLP 2025** (`2025.findings-emnlp.286`) | **1,184 memes**, 47 subreddits; public at `github.com/mainlp/ClimateMemes` | stance `convinced 78.0 % / skeptical 17.2 % / neither 4.8 %` (**κ 0.83**) + 8 media frames, multi-label, 2.11 frames/meme (per-frame κ > 0.7) | benchmark of VLMs and text-only LLMs, zero-shot and 1–4-shot | stance F1 (4-shot): **Mistral-7B text-only 60.54 > LLaVA-NeXT 56.68 > Qwen2-7B 53.28 > Molmo-7B-D 49.53** — the text-only model wins |
| 9 | **TikStance** · `2607.15240` | **preprint** | **161 TikTok videos + 13,876 comments**, Trump/Biden/Harris | `Favor/Against/None` for **both** video→target and comment→target; 3 annotators, **Krippendorff α 0.743 / 0.723 / 0.722** | dataset paper only | **no baselines, no zero-shot numbers, no verified public release** |
| 10 | **GET-Tok** · `2402.05882` | preprint (cs.SI) | **43,697 TikTok videos** (Peru 2022 coup), AI-generated transcripts / descriptions / OCR / **stance** | weakly-labelled stance | TikTok Research API + generative-AI enrichment; pipeline code at `github.com/gabbypinto/GET-Tok-Peru` | label-generation model, prompt and human validation **not stated on the abstract page — UNVERIFIED**; largest video-stance label pool found, but weak labels |
| 11 | **Inter-Stance** · `2604.22739` | preprint (cs.CV) | 45 dyads / 90 people, **20 TB**: 2D+3D face, thermal, voice, PPG/EDA/HR/BP/respiration | agreement / disagreement / neutral in face-to-face dyads | corpus paper | **not our object at all** — interpersonal stance from physiology, no content, no text. Listed only so it is not mistaken for a usable resource |
| 12 | **AuDisAgent — multimodal controversy detection** · `2605.02939` | preprint | video + comments | controversy, not stance per se | "Viewing Panel Agent" simulates post-screening discussion among audiences with diverse stances; comment-bootstrapping for cold start | no separation of creator stance from depicted content |
| 13 | **Acquired TASTE** · `2412.03681` | preprint | text + **conversational structure** (not vision) | stance in conversation | transformer embeddings + structural embeddings via gated residual networks | listed for completeness; "multimodal" here means content + social structure |
| 14 | **MemeCLIP / PrideMM** — Bikram et al. · `2409.14703` | **EMNLP 2024 Main** | **5,063 text-embedded images** (LGBTQ+ Pride), public at `github.com/SiddhantBikram/MemeCLIP` | **four tasks on the same items**: hate (No 50.97 / Hate 49.03), hate target (Undirected 31.07 / Individual 10.03 / Community 46.90 / Organization 12.00), **stance (Neutral 28.80 / Support 37.70 / Oppose 33.50)**, humor (67.57 % humorous) | **frozen** CLIP image+text encoders + linear projections to 1024-d + feature adapters with residual α = 0.2 + cosine classifier with semantic-aware init; 430M total params, only the light modules trained | hate acc 76.06 / AUROC 84.52; **stance acc 62.00 / AUROC 80.11**; humor 80.27 / 85.59; zero-shot GPT-4 on hate 70.00 acc / 69.38 F1 vs MemeCLIP 73.00 / 72.54 |

**Adjacent, not occupants:** multimodal sarcasm detection (MMSD2.0/MMSD3.0 line, CVPR 2026
`MMSD3.0`) — same acronym collision, different task (it detects image–text incongruity, not who
holds a position). Multimodal *sentiment/emotion* is a separate literature and `2206.06423` already
told us affect is the wrong auxiliary.

**Chinese/other-language multimodal stance:** CMFC (Chinese, BERT + ResNet-50, macro-F1 85.40) is
cited in the LLM-stance survey `2505.08464` but was **not verified first-hand — UNVERIFIED**.

---

## 3. Method breakdown — how these systems actually fuse and where the MLLM sits

Four distinct architectures, and they map cleanly onto four positions for the MLLM.

**(a) Frozen encoders + prompt tuning (TMPT, `2402.14298`).** BERT-base and ViT-base both **frozen**;
the only trainable parameters are 7 visual prompt tokens inserted at ViT layer 1, the textual prompt
embedding, and a single FC layer. Fusion is plain **vector concatenation**. This is the cheapest
supervised design in the field and the one closest to our own head-level retraining budget
(~52 s/head in our stack). Its weakness is exactly the one that disqualifies it for us: the target
string is part of the prompt, so the model is target-conditioned by construction.

**(b) Frozen encoders + alignment/weighting + iterative refinement (T-MAD, EMNLP 2025).** RoBERTa +
ViT, target embedding as query. Two mechanisms worth remembering independently of the task:
(i) **modality relevance by mutual information** — MI(modality, target) approximated by the negative
InfoNCE loss, exponentiated into a relevance score `r`, and the fused representation is
`(r_I·Ẽ_I + r_S·Ẽ_S)/(r_I+r_S) + λ·E_t` with λ = 0.5. That is a principled per-item answer to "which
modality should I believe here", and it is directly reusable in a multi-channel fusion head like
ours. (ii) **Confidence-weighted voting fusion (CWVF)**: prompt the MLLM 5× and use label frequency
as its confidence, prompt the trained model and use its softmax; take the higher-confidence label,
ties to the trained model. CWVF is what lifts zero-shot MTSE-DT from 58.1 to 77.2 — i.e. **most of
their cross-target robustness comes from the MLLM vote, not from the trained network.**

**(c) Fine-tuned MLLM (MLLM-SD, ACM MM 2024).** LLaMA2-chat-7B + frozen ViT, **LoRA r=64 on query and
value matrices**, GPT-4V captions injected as text, one-shot CoT. Beats zero-shot GPT-4V by 3.6–9.4
F1. **Excluded for us by the stated constraint** (no fine-tuning of large models — training stack
missing) and it is also the design our own `MEMORY` notes rule out.

**(d) Frozen LLM/VLM as an agent ensemble (MM-StanceDet, ACL 2026; MIND).** gpt-4o-mini plays
text-analyst (incl. sarcasm/irony), image-analyst, **modality-conflict analyst**, three stance
debaters and an adjudicator — roughly 7+ LLM calls per item, no training. MIND is the same family
with "experience pools" refreshed during training. These are the field's current frontier and they
are also the design our prior sweep already warned about: unconstrained free-form reasoning lost
10–46 points to constrained typed questions in `2604.24179`. Note that MM-StanceDet's
modality-conflict agent asks whether image and text **agree**, never whether the poster **endorses**
the image.

**(e) Frozen LLM → text artefacts → trained small fusion (`2509.08024`).** The only one in our
shape: Mistral-7B summary + domain-aware caption → joint BERT-style encoder + ViT → linear head,
202M params, **+3.1 accuracy over BERT+ViT** on MultiClimate. This is the one method in the whole
multimodal-stance literature that we could re-implement as-is with our existing stack.

**Classification heads are uniformly trivial.** Every supervised system in the list ends in a
single fully-connected layer + softmax over 3 (or 4) classes. Nobody uses a structured/graded head,
nobody predicts a posterior over stance types plus an explicitness score. Our `IDEA_REPORT` design
(graded typed record, carry the record not the label) is not contradicted by anything here — it is
simply unattempted.

**Cost.** TMPT-class: minutes on one GPU. T-MAD: 5 iterative attention steps + 5 MLLM samples per
item for CWVF. MM-StanceDet: ~7 gpt-4o-mini calls/item (no cost figure published). `2509.08024`:
0.043 s/item inference, 202M params.

---

## 4. Use-vs-mention in multimodal stance: **nobody does it**

Checked directly, paper by paper:

| work | does it separate poster's stance from the stance of the depicted/quoted material? | evidence |
|---|---|---|
| TMPT / MMSD `2402.14298` | **No.** No discussion of irony, quoting, or poster-vs-image-content. Their §6.5 error analysis notes ~70 % of misclassified samples contained stance-related images but attributes it to *image complexity* (text-in-image, memes) | full-text check |
| MmMtCSD / MLLM-SD `2409.00597` | **No.** Context is used to *disambiguate* stance, not to attribute it to a source. The finding is that conversation history helps (deeper threads → better) | full-text check |
| T-MAD `2025.emnlp-main.30` | **No.** "Modality inconsistency" is a fusion-weighting problem, solved by MI-based relevance, not an attribution problem | full text, §3–4 read |
| MM-StanceDet `2604.27934` | **No.** The modality-conflict agent assesses alignment/divergence between image and text; it does not address ironic reuse or whether the poster endorses the image | full-text check |
| MIND `2511.06057` | **Closest wording, still no.** The abstract names "irony or conflict" as inter-modal dynamics that "collectively shape the user's final stance" — but it treats them as noise to reason through, not as an attribution structure with a separate source | abstract |
| CLIMATEMEMES `2505.16592` | **No** — and this is the sharpest miss, because climate memes routinely *depict the opponent's argument in order to ridicule it*. The paper contains no sentence separating what the meme shows from what the poster believes | full-text check |
| MultiClimate `2409.18346` | **Undefined.** The guidelines never state whether the label is the on-screen speaker's stance or the uploader's; annotators were told to prioritise whichever modality "evokes stronger emotions related to the stance" when text and image conflict. For a corpus of news-style climate videos containing interviewees, this is a real latent ambiguity | full-text check |
| **PrideMM / MemeCLIP** `2409.14703` | **No.** Stance and hate are **four parallel classification heads over the same frozen CLIP features**; stance is never an input to the hate head, and "Oppose" means opposing the movement's goals, not opposing the material shown. A meme that displays homophobic content in order to condemn it has no representation in this schema | full-text check |
| TikStance `2607.15240` | **Partially, by accident.** It annotates video→target and comment→target stance **separately**, which is a two-level source structure — but within a video there is no author-vs-material split | abstract |

**The one place the structure exists in stance research at all is the reply/quote distinction in
text.** *Stance in Replies and Quotes* (**SRQ**, `2006.00691`) labels the stance of a response toward
the post it replies to or **quotes**, >5,200 labels, `support / deny / no clear stance`; its
headline finding is that **replies and quotes behave differently enough that pooling them lowers
accuracy** — i.e. the quoting relation is its own thing. That is the nearest published
acknowledgement that quoting changes the stance-inference problem, and it is 2020, text-only, and
not connected to hate.

**Also checked and empty:** no computational work found on "sharing ≠ endorsing" as a detection
problem; the retweet literature (`1411.3555` and descendants) treats retweeting as a *behavioural*
signal about agreement rates, not as an attribution decision about a piece of content. No dataset
found in any modality that marks "this video/meme displays material the author is criticising".

**A neighbouring capability that is measured, and that MLLMs are known to fail: "who said what" in
video.** Separate from stance, there is an audio-visual benchmark line on **speaker-attributed
reasoning**: **M3-SLU** (`2510.19358`) with Speaker-Attributed QA and Speaker Attribution via
Utterance Matching, **AV-SpeakerBench** (`2512.02231`), and **AMUSE** (`2512.16250`) with an
Audio-Visual Speaker Association task mapping each utterance to its visible speaker. The common
reported finding is that current models "still struggle with speaker-attributed reasoning". *All
numbers here are **UNVERIFIED** — abstracts only, via search summaries.* This matters for us because
attributing a proposition to a source is the perceptual prerequisite for use-vs-mention: if the
model cannot reliably say *who* uttered the hateful line in a video, it cannot say whether the
uploader endorses it. Worth one verification pass before it is cited, but it is a plausible
mechanistic co-factor alongside the safety prior — and it suggests a diagnostic we could run cheaply
(does our reasoner correctly attribute the hateful utterance to a speaker at all?).

**Consequence.** The §2.3 verdict in `STANCE_LIT_RECON.md` extends cleanly: not only is stance absent
from hateful-video work, **use-vs-mention is absent from multimodal stance work**. The two
literatures are disjoint and neither contains our object. Nothing here weakens the novelty claim;
it strengthens it, at the cost of removing the possibility of borrowing a ready-made component.

---

## 5. Zero-shot LLM/MLLM performance — the honest summary

**Text stance (for calibration, and it is a correction to our prior framing).**

| model / prompt | benchmark | metric | number | source |
|---|---|---|---|---|
| ChatGPT, DQA prompt | SemEval-2016 | macro-F1 | **78.0** | `2304.03087` (arXiv only, **no verified venue**) |
| ChatGPT, StSQA (1-shot) | SemEval-2016 | macro-F1 | 78.9 | same |
| ChatGPT DQA | P-Stance Trump / Biden / Bernie | macro-F1 | 83.2 / 82.0 / 79.4 | same |
| ChatGPT DQA | VAST | macro-F1 | 62.3 vs supervised WS-BERT 74.5 | same — **UNVERIFIED** (table not opened) |
| "LLM prompting" | P-Stance | macro-F1 | 86.52 | survey `2505.08464` Table 3 |
| GPT-3.5 / Mistral-7B / Llama3-8B / Flan-T5 / Falcon-7B | PStance / SCD / KE-MLM | **weighted** F1 | 0.787/0.685/0.695; 0.804/0.637/0.671; 0.711/0.617/0.639; 0.693/0.591/0.623; 0.477/0.513/0.494 | `2510.20154`, **EMNLP 2025 Main** (`2025.emnlp-main.1605`) |

> **Two caveats that must travel with the strong text numbers.** (i) **Contamination.** Aiyappa et
> al., TrustNLP @ ACL 2023 (`2303.12767`), is a data-contamination case study *specifically* on
> ChatGPT stance detection on SemEval-2016. Any "ChatGPT ≥ supervised on SemEval-2016" claim is
> suspect. (ii) **Retrieval hurts.** Nguyen & Kim, **Findings of ACL 2025**
> (`2025.findings-acl.764` / `2507.01543`): giving LLMs external Wikipedia/web evidence **degrades**
> zero-shot stance by up to **27.9 %**, because the model aligns its answer to the stance of the
> retrieved evidence rather than the text. CoT does not fix it. *This is directly relevant to us: a
> retrieval-augmented hate pipeline can import the stance of the retrieved neighbours.*

**Multimodal stance — this is where zero-shot actually falls behind.**

| setting | zero-shot | supervised | gap |
|---|---|---|---|
| MMSD in-target, macro-F1 (per-dataset, TMPT paper) | GPT-4 49.81–70.46; **GPT-4V 37.61–72.82**; LLaMA2-70b 30.21–62.77; Qwen-VL-7B 27.73–50.51 | TMPT 60.84–77.59 | supervised ahead on 4/5 datasets; **open 7B VLMs collapse** |
| MMSD in-target (T-MAD reproduction, 12 target cells) | GPT-4 41.6–81.5; GPT-4V 37.6–72.8 | TMPT 43.6–81.2; T-MAD 49.3–86.2 | reproduction differs from the original paper's per-dataset figures — treat both as indicative, not exact |
| **MMSD cross-target (unseen target)** | **GPT-4V 72.7 / 71.3 (MTSE-DT/JB)** | **TMPT 31.7 / 32.7**; T-MAD 58.1 / 62.8; T-MAD+CWVF 77.2 / 75.6 | **zero-shot beats the supervised model by 40 points** |
| MmMtCSD, F1-avg | GPT-4V 61.31 / 61.28 / 70.04; gpt-3.5 60.84 / 54.78 / 66.31; GPT-4 56.91 / 66.62 / 61.54; Claude-3 56.61 / 55.99 / 46.33 | MLLM-SD 64.89 / 71.28 / 79.40 | 3.6 / 4.7 / 9.4 |
| MultiClimate (video) | Llama3-8B ≈ 0.48, Gemma2-9B ≈ 0.48, IDEFICS-9B below fusion | BERT+ViT 0.747 / 0.749; `2509.08024` 0.762 | ~0.27 accuracy — **the largest zero-shot deficit in the table, and it is the video setting** |
| CLIMATEMEMES (memes, 4-shot) | LLaVA-NeXT 56.68, Molmo-7B-D 49.53 | (no supervised head reported) | **text-only Mistral-7B 60.54 beats both VLMs** |

**Three patterns worth carrying into our design.**
1. **Vision often makes zero-shot worse.** GPT-4V falls *below* text-only GPT-4 on MWTWT
   (57.90 vs 71.62) and MRUC (37.61 vs 52.69) in the TMPT paper's table; on CLIMATEMEMES the
   text-only 7B beats both VLMs. Consistent with our own finding that our MLLM's video-level stance
   judgment is worse than what a text channel would give.
2. **Small VLMs are unusable.** Qwen-VL-7B 27.7–50.5 macro-F1 on a 3-class task. This corroborates,
   from a second literature, the Davidson (*Nat Hum Behav* 2025) instruction already in
   `STANCE_LIT_RECON.md` §3.6: do not put a 7B-class reasoner in the stance slot.
3. **A ~100M trained fusion model beats a 9B zero-shot VLM in the video setting** (MultiClimate
   0.747 vs ≈0.48). This is the in-modality version of `2507.05455`'s "the consumer must be trained".

**On the specific bias we found (offensive content ⇒ "author asserts it").** Nothing in the
multimodal stance literature reports it. Adjacent published documentation, all outside stance:
- **Dönmez, Vu, Faleńska, EMNLP 2024 Main (`2024.emnlp-main.1019`)** — 16 LLMs on (non-)offensive
  speech identification; documents **over-reliance on profanity** and failure to recognise
  stereotypes, i.e. keying on surface offensive tokens rather than communicative intent. Per-model
  numbers **UNVERIFIED** (PDF unparseable).
- **Zhang, He, Ji, Lu, ACL 2024 Main (`2402.11406`)** — LLMs show "excessive sensitivity towards
  groups or topics … misclassifying benign statements as hate speech", plus collapsed confidence
  calibration. Numbers **UNVERIFIED** (none in abstract).
- **`2509.00673`** (preprint) — censored vs uncensored models on Latent Hatred: 69.0 % vs 64.1 %
  strict accuracy, refusal 12.6 % vs 24.2 %; target-group variance 54.8 points; irony the worst
  category at 64.4 %.
- **`2509.13608`** (preprint) — GPT-4o-mini on 500 Hateful Memes: **144/487 refused**; on the rest
  precision 0.521 / recall 0.904 — the over-flagging signature, but no speaker-stance manipulation.
- **Counter-citation we must pre-empt: Davidson, *Nature Human Behaviour* 2025**
  (`s41562-025-02360-w`) — large MLLMs *can* make context-sensitive hate evaluations aligned with
  n = 1,854 humans; the failures concentrate in small models. A reviewer will use this against a
  blanket "MLLMs can't do stance" claim. Our own result is narrower and survives it: our reasoner
  fails on *speaker attribution* under a *5-way typed* schema, not on context-sensitivity in
  general.

**Pairwise > pointwise** (our 0.469 vs 0.257) is a known effect elsewhere, never shown for stance:
`2403.16950` PairS (venue **UNVERIFIED**) lifts Spearman-vs-human on NewsRoom coherence from
0.32→0.55 (Mistral-7B), **0.02→0.43** (Llama-2-7B-chat), 0.44→0.56 (GPT-3.5), 0.55→0.64 (GPT-4) —
the weakest model gains most, which matches our jump; `2306.17563` PRP (**Findings of NAACL 2024**)
beats pointwise LLM ranking "by double-digit margins". Neither is stance and neither is hate.

---

## 6. Portability — one verdict per method

Constraints applied, as instructed: **no fine-tuning of large models** (training stack absent),
**no large new annotation** (750 human stance judgments were not approved), and stance labels for our
videos do not exist.

| candidate | what it would take | verdict |
|---|---|---|
| **TMPT** (frozen BERT+ViT + prompt tuning) — train on MMSD, transfer to our videos | (a) MMSD images are **not distributed** (tweet IDs only, X hydration infeasible); (b) the label set has no mention/quote class; (c) it is target-conditioned, and T-MAD Table 2 shows it drops to **31.7 macro-F1 on an unseen target within the same corpus** | **EXCLUDE.** Three independent kills. Do not attempt the transfer experiment. |
| **MLLM-SD** (LoRA-tuned LLaMA2-7B + ViT) | LoRA fine-tuning of a 7B MLLM | **EXCLUDE** by the stated constraint (no large-model fine-tuning). |
| **T-MAD full model** | supervised training with a target embedding; needs MMSD | **EXCLUDE** (data + target conditioning). |
| ↳ **T-MAD's MI-based modality relevance weighting** (component only) | `r = exp(MI(modality, target))` estimated by negative InfoNCE; a ~20-line change inside a fusion head we already train | **BORROW, cheap, optional.** Reusable as a per-item "which channel do I believe" weight over our existing channels (transcript / OCR / vision / stance record). Its value for us is **unmeasured** — it is a plausible refinement, not evidence. Try only after a stance channel exists. |
| ↳ **T-MAD's CWVF** (5-sample MLLM vote + confidence fusion with the trained head) | 5 extra MLLM calls per video; no training | **BORROW as a baseline, not a method.** It is the mechanism that produced most of T-MAD's cross-target robustness. But it re-introduces the frozen MLLM's verdict into the decision, and our MLLM's stance verdict is 0.257 — CWVF would import the bias. Worth running once as a control precisely to show that. |
| **MM-StanceDet / MIND** (agentic, gpt-4o-mini, ~7 calls/item) | API cost at ~7 calls × our corpus; no training | **EXCLUDE for the main line.** Free-form multi-agent reasoning is the form `2604.24179` measured at −10/−22/−46 against constrained typed questions, and neither system addresses use-vs-mention. The *only* transferable idea is the **modality-conflict question as a typed field** ("do the image and the speech agree?"), which is one extra question in a schema we are already emitting — near-zero marginal cost. |
| **`2509.08024`** (frozen Mistral summary + domain-aware caption → trained joint encoder + ViT → linear head) | Re-implementable with our existing stack: one frozen LLM pass per video + a small trained head | **ADOPT as an additional channel and as a mandatory ablation.** +3.1 accuracy over BERT+ViT in the video-stance setting, our exact topology, 202M params, 0.043 s/item. Caveat: **preprint, unpublished, single dataset, no significance test** — treat the +3.1 as indicative. Its role for us is the **no-stance control**: LLM text in front of the encoder *without* any stance field. If that control captures the whole gain, our stance typology is not doing the work. |
| **MultiClimate as an out-of-domain capability gate** | Clone repo, run our reasoner over 4,209 frame-transcript pairs, compare against the published 0.747 fusion number and the ≈0.48 zero-shot LLM number and the 0.826 human ceiling | **ADOPT — do this first.** Zero annotation, zero training, inference-only, in-modality (video), with a published human ceiling and a published zero-shot floor. It is the cheapest available answer to "can our reasoner read stance off video at all". **Limitation to state up front:** it tests overt target-conditioned stance, not use-vs-mention; passing is necessary, not sufficient. Also note the corpus does not define whose stance is labelled (§4), so treat scores near the annotator ceiling with suspicion. |
| **CLIMATEMEMES as a second gate** | 1,184 memes, public repo, κ 0.83, published VLM baselines 49.5–60.5 F1 | **OPTIONAL, weaker.** Static images not video, and the published baselines are all ≤ 7B so the bar is low. Use only if a second out-of-domain point is wanted cheaply. |
| **TikStance / GET-Tok as weakly-labelled video stance training data** | TikStance: no verified release, no baselines. GET-Tok: 43,697 videos but **AI-generated** stance labels with unverified prompt/validation | **HOLD.** GET-Tok is the only large video-stance label pool in existence, but training on unvalidated LLM labels imports exactly the bias we are trying to remove. Revisit only if its validation section turns out to contain human agreement figures. |
| **PrideMM / MemeCLIP** — the only public multimodal corpus carrying a **hate label and a stance label on the same item** | 5,063 images, public, frozen-CLIP + adapter recipe, ~430M params of which only the adapters train | **USE, but only as a sanity platform, and read the caveat.** It is the one place where "does adding a stance channel help a hate head?" can be asked with **zero new annotation**. Two hard caveats: (i) *stance here is topical* — support/oppose the LGBTQ+ movement — which is heavily correlated with the hate label, so a gain there is close to circular and does not demonstrate use-vs-mention; (ii) **supervised stance on these memes only reaches acc 62.00 / AUROC 80.11** with a purpose-built model. That number is the single most useful calibration point in this file for our own gate (below). |
| **Training a use-vs-mention classifier on a stance corpus and transferring it** | requires a corpus with the mention/quote distinction — none of the multimodal stance corpora have it | **EXCLUDE from this literature.** The viable candidates all live outside multimodal stance: LDC BeSt `ROB` (10,777 EN, paid — check institutional licence first), rule-reconstructed CONAN use/mention pairs (~1.8k, free, we already hold the inputs), a quote-span tagger on PolNeAR+DirectQuote+PDNC (~70k, free). All text-only, all subject to the wrong-sign warning. See §7. |

**Cross-cutting warning for any retrieval-augmented variant.** `2025.findings-acl.764` reports
that supplying external evidence **degrades** zero-shot LLM stance by up to 27.9 %, because the
model adopts the stance of the retrieved material. Our project is a retrieval-augmented hate
detector. If any stance judgment is ever made *after* retrieval conditioning, this is a named,
published failure mode we would be walking into.

---

## 7. Corpora that carry a use-vs-mention / quotation / attribution signal

The question behind this section: is there any public corpus, in any modality, from which we could
train a **small** classifier for "is the speaker asserting this or quoting/reporting it" and carry it
over to our videos — given that we have no stance labels of our own and no approval to create 750.

**Found inside the stance literature (this sweep).**

| resource | venue | size | schema | modality | availability | usable as transfer supervision? |
|---|---|---|---|---|---|---|
| **SRQ — Stance in Replies and Quotes** `2006.00691` | preprint (2020) | **>5,200** stance labels | `support / deny / no clear stance` of a response toward the post it **replies to or quotes** | text | release **not stated on the abstract page — UNVERIFIED** | **The closest structural match in stance research.** The quote relation is annotated as its own thing, and the paper's own finding is that replies and quotes are different enough that pooling them *lowers* accuracy. But the stance is of a *responder toward a separate post* — the same "different object" problem as the rumour-stance family in `STANCE_LIT_RECON.md` §3.9. Our speaker and our quoted material are inside one video. |
| **PrideMM** `2409.14703` | EMNLP 2024 Main | 5,063 memes | hate + target + **stance (Support/Oppose/Neutral toward the movement)** + humor, all on the same item | image+text | public, `github.com/SiddhantBikram/MemeCLIP` | Only for the "does a stance channel help a hate head" wiring question — the stance is topical, not attributional, and is correlated with the hate label (§6 caveat). |
| **MultiClimate** `2409.18346` | EMNLP 2024 workshop | 4,209 frame–transcript pairs | `Support/Neutral/Oppose` | **video** | public, CC videos, frames in repo | Perception gate only (§6 Path A). No mention/quote class. |
| **CLIMATEMEMES** `2505.16592` | Findings EMNLP 2025 | 1,184 memes | stance + 8 media frames | image+text | public, `github.com/mainlp/ClimateMemes` (an `images/` directory is present; whether it ships the image files or references is **UNVERIFIED**) | Perception gate only. |
| **MMSD** `2402.14298` | Findings ACL 2024 | 17,544 | `Favor/Against/Neutral`, `Unrelated` in MWTWT | image+text | **IDs only, images not distributed** | **No.** |

**Already inventoried in `STANCE_LIT_RECON.md` §3.8 — not re-derived here, but they remain the
strongest candidates for a *text-side* mention classifier:** Kurrek, Saleem & Ruths
(`2020.alw-1.17`, 39.8k Reddit slur usages typed derogatory / appropriative /
non-derogatory-non-appropriative / homonym), CAD (`2021.naacl-main.182`, Counter Speech and
Non-hateful Slurs annotated in thread context, but only 220 counter-speech instances, κ = 0.267),
and DEBAGREEMENT (the agree/neutral/attack corpus that produced the +3 weighted F1 in
`2206.06423`). Nothing found in this sweep displaces them.

**Wider sweep — attribution, quotation and author-commitment corpora (independent agent, ~56 tool
calls).** Headline: **no public corpus in any modality labels "speaker asserts hate" vs "speaker
quotes/condemns hate" at trainable scale.** But two separate families come close from different
sides, and one of them is a genuine, previously-unnoticed match.

**(a) The closest label in existence is `ROB` in the LDC Belief-and-Sentiment corpus — and it is
paywalled.**

| resource | venue | size | the relevant label | availability |
|---|---|---|---|---|
| **BeSt / DEFT ERE Belief & Sentiment** | LREC 2022 pp. 2460–2467; TAC KBP 2016 | EN 157k words + ZH 133k + ES 79k; ~60.3k English belief annotations | `CB / NCB / **ROB** / NA`, where **ROB = "the writer reports another source's belief without revealing her own"**. Counts: **ROB 10,777 EN / 7,524 ZH / 8,822 ES = 27,123** | **LDC2023T13, paid** |
| **FactBank 1.0** | LDC2009T23; *LRE* 43(3) 2009 | 208 docs, ~9,500 events | `factValue ∈ {CT+,PR+,PS+,CT−,PR−,PS−,CTu,Uu}` **per (event, source) pair**, with pseudo-source `AUTHOR`; `fb_sip` records the source-introducing predicate | **LDC, paid.** ~3,659 non-author-source judgments (FBST split per Murzaku et al., Findings ACL 2023) |
| **LDC Committed Belief** | LDC2019T16 / T09 / T03 | EN 1,217 files, 952,723 words, 161,702 annotations | same `CB/NCB/ROB/NA` | LDC, paid; per-label counts **UNVERIFIED** |

`ROB` is, in text, exactly the distinction we need: *this proposition is being reported, and the
author has not committed to it.* 10,777 English instances is enough to train a small classifier.
**Action item, not a plan:** find out whether the university already holds an LDC membership
(most do) — if it does, this is the cheapest real supervision found anywhere in either sweep. If it
does not, do not buy it on this evidence alone.

> **Warning that must travel with this.** `2404.01651` Table 7 (already in `STANCE_LIT_RECON.md`
> §4.2) found that counterspeech **containing quotation marks is misclassified *more* often**
> (gpt-4 hate: 28.57 % FPR with quotes vs 7.23 % without). A quotation/attribution feature therefore
> has a *published wrong sign* in the hate setting. A trained attribution tagger is more than a
> surface cue, but nothing in this sweep shows it flips the sign. **Any quote-detection channel must
> be justified against that result before it is built**, not after.

**(b) Free, downloadable structural quotation supervision (English) — enough to train a quote/source
tagger without LDC.**

| resource | venue | size | schema | availability |
|---|---|---|---|---|
| **PolNeAR** | LREC 2018 (`L18-1524`) | 1,008 political news articles, ~760k words, **~24k attributions** (1 per 32 words) | token-level `source / cue / content / none` — **no direct-vs-indirect type** | free, `github.com/networkdynamics/PolNeAR`; **license not stated** |
| **DirectQuote** | LREC 2022 (`2022.lrec-1.752`) | 19,760 paragraphs, **10,353 direct quotations** from 39,153 articles | CoNLL IOB1 `LeftSpeaker / RightSpeaker / Unknown / Speaker / Out`, speakers linked to Wikidata | free, `github.com/THUNLP-MT/DirectQuote`; license not stated |
| **PDNC** | LREC 2022 (`2022.lrec-1.628`) | **35,978 quotations**, 22 novels | quotation **type: Explicit / Implicit / Anaphoric** + speaker, addressee, referring expression | free, `github.com/Priya22/project-dialogism-novel-corpus`; license not stated |
| **RiQuA** | LREC 2020 (`2020.lrec-1.104`) | **5,963 quotations**, 11 works of 19th-c. literature, doubly annotated | span, speaker, addressee, cue; direct **and indirect** | free tarball (Stuttgart IMS); license not stated |
| **Quotebank** | WSDM 2021 | **235M** attributed quotations, 189.7 GB, 2008–2020 | quote, speaker probabilities, Wikidata QIDs | Zenodo 4277311, **CC-BY-4.0** — but labels are **automatic** (BERT-extracted), silver only |
| **REDEWIEDERGABE** | LREC 2020 (`2020.lrec-1.100`) | 489,459 tokens, **12,123 STWR instances** | **direct / indirect / free-indirect / reported**, plus a **non-factual STWR** attribute | `github.com/redewiedergabe/corpus`, **CC-BY-NC-SA-4.0**; **German only** |
| **FRACAS** | LREC-COLING 2024 (`2024.lrec-main.654`) | 1,676 French texts, **10,965 attribution relations** (Direct 4,437 / Indirect 4,672 / Mixed 1,909) | quote type + speaker type | Zenodo 8353229, **restricted** (needs NIST Reuters authorization); French |
| **PARC 3.0** | LREC 2016 (`L16-1619`) | ~20k attribution relations over WSJ | `source / cue / content` spans | **not downloadable**; requires LDC PTB **and** PDTB licenses + author contact |

*(Verified negatives from that sweep: no annotated corpus for scare quotes; no spoken/audio corpus
annotated for direct-vs-reported speech; no computational sign-language role-shift corpus.)*

**(c) The use-mention paper releases no data — but the recipe is free and we already own the inputs.**
`2404.01651`'s repo (`github.com/kristinagligoric/use-mention`, MIT) contains **only analysis
notebooks and figures — zero annotation files**. Their human validation set is **160 examples**.
Their analysis set of **1,826 (use, mention) pairs is constructed by rule**: in a paired
counterspeech corpus, the original hateful statement is the *use* and the counterspeech is the
*mention*. Sources are Chung et al. 2021, Fanton et al. 2021 (Multi-Target CONAN) and He et al. 2023.
**We already hold the CONAN family**, so ~1.8k use/mention pairs can be reconstructed at zero
licensing cost. This is the cheapest text-side supervision available to us. Its weakness is the one
`STANCE_LIT_RECON.md` §3.1 already recorded: these are expert-written canonical counterspeech, not
naturalistic content, and `2204.04042` showed that fine-tuning on constructed counterspeech cases
costs i.i.d. performance.

**(d) Condemnation-axis corpora, free and larger than what we had catalogued.**

| resource | venue | size | labels | note |
|---|---|---|---|---|
| **Finding Authentic Counterhate Arguments** | **EMNLP 2023** (`2023.emnlp-main.855`) | **54,816** tweet/paragraph pairs | binary: is this paragraph an authentic counterhate argument for this tweet | largest free condemnation-axis resource found |
| **Albanyan & Blanco** | **AAAI 2022** | 5,652 tweet/reply pairs | 4 binary: reply is hate **or counter-hate**; justifies; attacks author; adds hate | `github.com/albanyan/hateful-tweets-replies` |
| **Mathew et al., *Thou shalt not hate*** | **ICWSM 2019** | **13,924 YouTube comments on hate videos**; 6,898 counterspeech / 7,026 not | 8 counterspeech types including **Denouncing hateful speech**, Presenting facts, Hostile language | **The only condemnation corpus that is grounded in hate *video* context** — comments *on* hateful YouTube videos. Closest domain match in the whole inventory. |
| **Measuring Hate Speech corpus** | NLPerspectives @ LREC 2022 | 39,565 comments / 135,556 annotations (YouTube+Reddit+Twitter) | includes an ordinal **`attack_defend`** item — does the speaker attack or defend the target | HF `ucberkeley-dlab/measuring-hate-speech`, **CC-BY-4.0**. Exact counts **UNVERIFIED**. An `attack_defend` axis on real comments is close to our `endorses` vs `condemns` split. |
| **IntentCONAN / v2** | ACL 2023 (`2023.acl-long.318`) / NAACL 2024 (`2024.naacl-long.374`) | 6,831 → **13,952** counterspeeches | intents `POS / INF / QUE / **DEN (denouncing)**` | built on Multi-Target CONAN — overlaps what we hold |
| **DIALOCONAN** | EMNLP 2022 (`2022.emnlp-main.549`) | 3,059 dialogues, 16,625 turns | hater vs NGO turn role, by construction | `github.com/marcoguerini/CONAN`, research-only |
| **HateCheck F20 / F21 / F9** | ACL 2021 (`2021.acl-long.4`) | **F20 = 173, F21 = 141, F9 = 81** (425 total) | denouncement quoting hate / denouncement by direct reference / reclaimed slurs | **CC-BY-4.0. Far too small to train on — but it is precisely our target functionality set and free, so it is the natural zero-cost held-out probe for any quote/endorse discriminator.** |
| **CrowdCounter** | CoNLL 2024 | 3,425 HS-CS pairs | empathy, humor, questioning, warning, shaming, contradiction | ~570/type — too small |
| **MultiPRIDE** | EVALITA 2026 (`2026.evalita-1.18`) | size **UNVERIFIED** | Task A **reclamatory vs non-reclamatory** slur use; Task B adds user-bio context; EN/IT/ES | the shared task behind `2602.12818`, already in `STANCE_LIT_RECON.md` §3.6 |

**(e) Multimodal: verified void, second independent confirmation.** Nothing marks whether an
image / video / meme **shows** versus **endorses** the depicted content. Explicit negative searches:
screenshot-of-a-tweet meme datasets; TikTok stitch/duet stance toward the original; TikTok
counterspeech; YouTube reaction/commentary hate stance; any "endorse vs critique the depicted
content" label. TikTok StitchGraph (`2502.18661`) has the stitch *structure* but no stance labels.
MemeGuard (`2406.05344`) *generates* counterspeech for memes but does not label poster stance.
HateMM, HateClipSeg and MultiHateClip all lack the axis (already verified on disk in
`STANCE_LIT_RECON.md` §2.2).

**Net answer to the commissioned question.** There is no free multimodal supervision for
use-vs-mention and there will not be one; if we want that label on video we make it ourselves.
On the text side there are exactly three zero-cost options, in decreasing order of directness:
**(i)** reconstruct ~1.8k use/mention pairs from CONAN by rule; **(ii)** train a quote/attribution
span tagger on PolNeAR + DirectQuote + PDNC (~70k annotations, all free) — subject to the
wrong-sign warning above; **(iii)** use the `attack_defend` axis of the Measuring Hate Speech corpus
(CC-BY, 39.5k comments) or Mathew's 13.9k YouTube comments on hate videos as a condemnation-axis
training signal. And one paid option that is a much better match than any of them: **LDC BeSt's
`ROB` label, 10,777 English instances** — check for an existing institutional LDC licence before
anything else.

---

## 8. Coverage and limits

**Channels used.** arXiv API (title/abstract conjunctions and full listings), arXiv HTML full text,
ACL Anthology landing pages and PDFs, GitHub repository READMEs, targeted web search. Full text or
full tables read directly for: `2402.14298`, `2409.00597`, `2025.emnlp-main.30` (Tables 1–2 read
from the PDF), `2604.27934`, `2409.18346`, `2509.08024`, `2505.16592`. Abstract-only for
`2511.06057`, `2607.15240`, `2604.22739`, `2605.02939`, `2402.05882`, `2006.00691`.

**Enumeration.** `all:"multimodal stance detection"` over arXiv returns **7 papers** — all
enumerated in §2. `abs:"stance" AND abs:"video"` returns **31 papers**, all listed and triaged;
only 6 detect stance *from* video content. This is consistent with the parallel enumeration recorded
in `STANCE_LIT_RECON.md` §6.

**Verified independently during this sweep** (venue + abstract read from the ACL Anthology):
`2024.emnlp-main.1019` (Dönmez et al., EMNLP 2024 Main — over-reliance on profanity),
`2025.findings-acl.764` (Nguyen & Kim, Findings ACL 2025 — external information degrades LLM stance
by up to 27.9 %), `2025.emnlp-main.30` (T-MAD, tables read from PDF), `2409.14703` (MemeCLIP,
EMNLP 2024 Main). `2304.03087` (ChatGPT CoT stance, the source of the 78.0 SemEval figure) confirmed
as an **arXiv preprint with no venue**.

**Known gaps.** The audio-visual speaker-attribution benchmarks in §4 (`2510.19358`, `2512.02231`,
`2512.16250`) were reached only through search summaries — **no number from them is verified**.
No Google Scholar; no CNKI/Wanfang (CMFC and other Chinese multimodal stance work is
therefore unverified); ACM DL was reached only via the DOI landing page for MmMtCSD; MM-StanceDet's
per-cell result table was not fully extracted; MIND has no extracted numbers. IEEE/Springer journal
venues (where a lot of Chinese multimodal-stance work lands) were not swept systematically.

**Two parallel independent sweeps** fed §5 and §7 (~65 and ~56 tool calls respectively; arXiv,
ACL Anthology, Semantic Scholar citation graph, LDC catalogue pages, GitHub, Zenodo, HuggingFace).
Their own stated gaps: COLA's plain zero-shot per-model rows, MultiClimate's per-model table,
Dönmez et al.'s per-model numbers, `2402.11406`'s per-model numbers, and the EMNLP 2025 T-MAD table
were all blocked by image-only or malformed PDFs on their side (**the T-MAD table was subsequently
read directly from the PDF in this session and is in §2/§5**). Unverified in §7: licences for
PolNeAR / DirectQuote / RiQuA / PDNC (none states one), Multilingual-HateCheck and IntentCONAN-v2
per-class counts, MultiPRIDE size, REDEWIEDERGABE per-type counts, LDC pricing for FactBank/BeSt.

**Provenance note on venues.** MemeCLIP/PrideMM "EMNLP 2024 Main" and MM-StanceDet "ACL 2026 Main"
both come from the **arXiv comments field** (author-stated); neither was cross-checked against the
official accepted-papers list.

**Numbers to re-verify before any of them enters a paper.** The MMSD per-dataset TMPT figures
disagree between the original paper (60.84 / 67.67 / 77.59 / 75.76 / 67.59) and T-MAD's
reproduction (per-target, e.g. MTSE 55.41 / 61.61) — read both tables before quoting either.
The ChatGPT 78.0 SemEval-2016 figure is contamination-suspect. Everything marked UNVERIFIED above.
