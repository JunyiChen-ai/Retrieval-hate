# STREAM COMPOSITION — FORENSIC RECON (zero-GPU, TRAIN split only)

**Agent:** streamcomp forensic-recon · **Date:** 2026-07-28 NZST · **Cost $0** (CPU, `OMP_NUM_THREADS=4`;
zero GPU, zero SLURM, zero Modal, zero training).
**Question:** the deployed retrieval key is the FUSED key (image ⊕ text). If the image stream contributes
no unique information (F86) and drags neighbourhood purity down on ZH/EN (F44, VSW §3.6), **what does the
deployed vote do on a text-only or stream-reweighted retrieval key?**

**Test-split contact: NONE.** The only data files opened are
`data/CLIP_Embedding/{HateMM,MHC_zh,MHC}/train_<model>.pt`. Every test number quoted below is
**transcribed from a banked record, re-read from its source this session** — none was computed here.

**Write scope.** This recon wrote **only** this file plus scratch under the session scratchpad. It did not
write, edit, run or delete `refine-logs/VSW_*`, `scripts/analysis/*vsw*`, `refine-logs/LITSWEEP7_*`,
`refine-logs/LITSWEEP8_*`, `refine-logs/MEMBANK_C4_*`, `scripts/analysis/*membank_c4*`, or
`findings.jsonl`, `refine-logs/MEMBANK_C4_PREGATE_RECORD.md`, `refine-logs/INSTRUMENT_VALIDATION_RECON.md`,
`refine-logs/PROVENANCE_AUDIT_2026-07-28.md`, `refine-logs/HEADCOV_PREGATE_RECORD.md`,
`refine-logs/DISK_FORENSICS_2026-07-28.md` or `refine-logs/PREGATE_DETERMINISM_CLAUSE.md` — all opened
read-only where opened at all. Committed on `main` (this file only), not pushed; the ledger rows
(`findings.jsonl` F106, `directions_tried.json` ban_scope) were appended after this record was frozen.

---

## 0. VERDICT (up front)

**PRE-CLOSED, and the residual sliver is arithmetically dead. NO new candidate. Do not spend GPU.**

1. **The question is already measured, on all three datasets, in both arenas.** F95 banked the deployed
   vote's train-LOO accuracy **and macro-F1** on the fused, text-only and image-only keys under the exact
   5-fold protocol (`scripts/analysis/mechnov_pairverify_{hatemm,zh,en}_OUT.json → spaces.*.pooled`), and
   F88/ERRPAT banked **deployed text-only numbers on TEST for all three datasets**
   (`ERRPAT_HateMM_2026-07-26.md:514-524`, `ERRPAT_MHC-ZH_2026-07-26.md:246-256`,
   `ERRPAT_MHC-EN_2026-07-26.md:241-257`). The paper draft already quotes the HateMM one as a settled
   ablation (`research-wiki/DRAFT_analysis_chapter.md:937-940`). Per the tasking's own stop rule, **A
   settles it**; §4 was run anyway, cheaply, because the reweighted-key cell (`a`-sweep) was genuinely
   untouched in this arena, and it is the cell that would have carried any objection.
2. **On TEST the direction is 0-for-3.** HateMM head-space text-only **+0.0047 val-sel (sign 2/3) /
   +0.0093 final (sign 3/3)** — inside the ±0.014 seed band; MHC-ZH raw text-only **+0.0067 = 1 item**;
   MHC-EN text-only **−0.0109 BELOW** the deployed number. Not one leg is within 3× of the +0.030 bar.
3. **NEW, and this is the load-bearing new fact: the deployable stream-weight meets the goal conjunct on
   exactly ONE dataset of three, in the arena maximally favourable to it.** With `a*` selected per fold on
   the fitting fold's own internal LOO (never touching the held-out fold), raw train-LOO Δ vs fused is
   **HateMM −0.0027 acc / −0.0021 mF1 · MHC-ZH +0.0346 / +0.0376 · MHC-EN +0.0200 / +0.0363** (newly
   computed, `streamcomp_nested_OUT.json`). ZH clears the conjunct; EN misses on acc; HateMM is negative.
   **The goal needs ≥2. It is dead on arithmetic before transfer, protocol or novelty is even argued.**
4. **NEW: the reweight is structurally unrepresentable in the deployed head.** Under `align`/Hadamard on
   pre-L2-normalised projections (`src/model/classifier.py:115-141`), scaling the streams by `a` and
   `1−a` gives `(a·î) ⊙ ((1−a)·t̂) = a(1−a)·(î ⊙ t̂)` — a single **global scalar** on the fused vector, not
   a per-stream reweight. The `a` parameter has no image in the deployed architecture. Realising it
   requires `concat`, which is **F85-KILLED on both datasets and both protocols** and whose scope is
   frozen (`FUSIONCAT_VERDICT_REVIEW.md:305-313, 275-279`).
5. **D7: dead, and named verbatim.** `ENCODER_SWAP_DIAGNOSIS.md:198-200` rules exactly this object:
   *"**'Down-weight/gate the collapsed image stream on MHC'** = modality gating / learned fusion weights =
   textbook, decision-side (Axis A conditional-redundancy), D7-dead."* A positive here would not be usable
   as novelty.
6. **The tasking's fusion-tax prediction is measured and is REFUTED on HateMM.** Text-only lowers break
   exposure on ZH (0.1283 → 0.1040) and EN (0.2126 → 0.1640) but **raises** it on HateMM (0.0653 →
   0.0717), and it lowers **fix supply** on all three. The exposure asymmetry is not repairable by
   dropping the image stream — it is traded, not removed.

---

## 1. PREMISE CHECKS — two corrections before anything is built on them

**1.1 "the text streams are equally pure across all three datasets (~0.85)" — FALSE for EN.**
VSW's own §3.6 table (`VSW_ASYMMETRY_RECON.md:285-287`) reads `purity: fused / text / img` =
`0.80 / 0.85 / 0.75` (HateMM), `0.70 / 0.85 / 0.60` (ZH), **`0.65 / 0.70 / 0.60`** (EN). EN's text stream
is at **0.70**, not 0.85. The "equally pure text everywhere" framing holds for HateMM↔ZH only. F44 says
the same thing from the AUC side: text-only train-LOO AUC `0.888 / 0.847 / 0.851` for
HateMM/ZH/EN-frozen-Qwen (`ENCODER_SWAP_DIAGNOSIS.md:96-102`) — but EN's **fused** ceiling is the lowest
of the three for label reasons F88 separately priced (label-semantics mismatch, 40.9 % of consensus
errors, `research-wiki/DRAFT_analysis_chapter.md:920-926`).

**1.2 "break exposure 0.0127 / 0.0448 / 0.0678 (5.33×)" — correct, but that is the *verifier-conjunct*
statistic and it does not transfer across key spaces.** VSW's exposure counts `correct ∧ cheap ∧ *the
verifier's push hurts*` (`VSW_ASYMMETRY_RECON.md:181-201`). The F95 verifier is fitted **in the fused
space over fused top-20 neighbour sets**; on a text-only key the neighbour sets change (see §4.5:
membership overlap is only 37–50 %), so the verifier term is undefined there. §4.4 therefore reports the
**verifier-free geometric half** — `cheap` alone — which is the half that actually moves with the key.
**Cross-validation that this is VSW's own quantity:** at `a=0.5` my `frac_COR cheap(θ≤0.10)` reads
**0.0653 / 0.1283 / 0.2126**, against VSW's bank-size control table `0.0653` (HateMM, native 595),
`0.1283` (ZH, 463) and `0.2118` (EN, 439) at `VSW_ASYMMETRY_RECON.md:259-263` — **exact on two of three**,
0.0008 apart on the third (their EN row is a seeded resample, mine the native folds). Independent
implementations agree.

**Supersession note (F105 / VSW, commit `e9a17fe`).** The exchange-rate screen `ER ≥ 1.2` is **refuted**:
VSW reached **ER 6.0** on HateMM and still failed. The correct law is
`net = changed × (2·precision − 1)`, with precision decaying monotonically with sharpness, so the binding
screen is **net items** against **22.3 / 17.4 / 16.5** (HateMM / MHC-ZH / MHC-EN = `0.030 × n` for
n = 744 / 579 / 549). This recon cites **no ER bar** — §4.4's supply/exposure decomposition is used as a
*mechanism* read, never as a pass/fail screen — and §4.3 is judged on the accuracy/mF1 conjunct directly.

**1.3 Everything else in the tasking reproduces.** Magnitude-inertness, cone dynamic range, fix
yield 0.250/0.227/0.264 and the 5.33× exposure spread were re-read from `VSW_ASYMMETRY_RECON.md`
§3.3/§5/§6.1 and are used as stated.

---

## 2. TASK A — PRE-CLOSURE AUDIT

### A1 · F85 / fusion-concat + `FUSIONSWAP_FORENSIC_RECON.md` — **PRE-CLOSED for the trained fusion operator; NOT BINDING on a text-only key**

*What was swapped.* One token: `--fusion_mode "align"` → `"concat"`, no code diff
(`FUSIONSWAP_FORENSIC_RECON.md:30, 83`). The head retrains from scratch per arm on cached LoRA features;
`concat`, `align`, `cross` are the only three branches in the repo (`src/model/classifier.py:85-90,
138-143`) and **`align` is the only fusion ever run on video** (`FUSIONSWAP_FORENSIC_RECON.md:28`).

*What was measured.* `FUSIONCAT_VERDICT_REVIEW.md:305-313`, verbatim:
> **ZH … vs floor 13150.** Mean paired Δacc **+0.0067** val-selected / **−0.0045** final-epoch; ΔmF1
> **+0.0097** / **−0.0023**; sign acc 2/3 / 1/3. … **Ruling: FORMAL NEGATIVE both protocols; KS-arm-dead
> KILLED.**
> **HateMM … vs floor 13241.** Mean paired Δacc **−0.0031** on **both** protocols … sign 0/3 positive on
> every leg. … **Ruling: FORMAL NEGATIVE both protocols; KS-arm-dead KILLED.**

*The recorded caveat, verbatim* (`FUSIONCAT_VERDICT_REVIEW.md:295-299`):
> The **F0.6 bundling caveat** is carried into the null as well: the arm that failed was `concat` **with
> 2.0× first-Linear params** (2,098,176 vs 1,049,600), so the honest reading is "the concat-fusion arm,
> capacity bump included, does not beat Hadamard here" — the measured null is **not** attributed to the
> operator in isolation, and it does **not** license a claim that "extra head capacity cannot help" in
> general.

*Ruling.* **PARTIALLY BINDING.** F85 kills the trained *fusion-operator swap*; it did not test a
*stream-selection* key. But it is fully binding on the only route by which a stream weight could be
deployed (§0.4), and its scope clause forecloses a follow-up: *"a **param-matched control** … or a third
dataset/encoder are each a **new** pre-registered family costing a **new** bite — none is authorized by
this verdict, and none may be run as a 'follow-up' to it"* (`FUSIONCAT_VERDICT_REVIEW.md:275-279`).
FUSIONSWAP's own recommendation was **PARK** with `P(goal) ≈ 0.03–0.06`
(`FUSIONSWAP_FORENSIC_RECON.md:112, 120`).

### A2 · F44 / F48 / F50 — the fusion mechanism and the Hadamard erratum — **PRE-CLOSED**

**F44 (`ENCODER_SWAP_DIAGNOSIS.md`).** Per-stream train-LOO kNN AUC, `:96-102`:

| stream | HateMM CLIP→Qwen | MHC-EN CLIP→Qwen | MHC-ZH CLIP→Qwen |
|---|---|---|---|
| image | 0.826 → 0.817 (−0.009) | 0.734 → **0.599 (−0.135)** | 0.718 → 0.721 (+0.003) |
| text | 0.847 → 0.888 (+0.041) | 0.797 → 0.851 (+0.054) | 0.802 → 0.847 (+0.045) |
| concat | 0.867 → 0.883 (+0.016) | 0.801 → 0.825 (+0.023) | 0.764 → 0.840 (+0.076) |

`:116-124`: *"Because RGCL fuses image and text as two **equal-weight L2-normed 1024-d blocks**, the fused
outcome is dominated by whichever stream is worse … **MHC-EN:** collapsed Qwen image **cancels** the +0.054
text gain in the 50/50 concat → fused Δacc ≈ 0 (dev −0.012)."* And `:126-128`: *"The trained head has
*some* capacity to attenuate the collapsed image block, but the banked **test** result (MHC-EN
frozen-Qwen FAIL) shows it does **not** net-recover."*

**The Hadamard erratum (F48).** `FA_GATE_RECORD.md:9-16`, verbatim:
> The wave-4 recon's §0.3 correction showed the deployed head fuses via **`fusion_mode='align'` = a
> parameter-free element-wise **Hadamard** product of two L2-normed projections** … — **not** the
> "equal-weight concat" F44's prose described. In align mode a linear `img_proj` cannot map varying inputs
> to a constant (zero weights => `normalize(0)` NaN), so the head **structurally cannot down-weight the
> collapsed Qwen image factor**; F44's dismissal of a modality-reweighting fusion lever … rests on a
> premise the align head does not satisfy.

and `:118-119`: *"The align (Hadamard) control is the *worst* accuracy of all — consistent with §0.3 (a
collapsed factor corrupts multiplicatively)."*

**F50 (`FA_GATE_RECORD.md`) already ran the exact sweep this tasking proposes.** Arm A1, `:38-41`:
> **A1 Qwen weighted-concat** = `z = [sqrt(w).imghat_Q , sqrt(1-w).texthat_Q]`, `w in {0.00,0.05,…,1.00}`
> (21). Because each block is L2-unit, `||z||=1` and `cos(z_a,z_b) = w.img_cos + (1-w).txt_cos` — a clean
> convex reweight of the two modality cosines. **`w->0` = Qwen-text-only**; `w->1` = Qwen-image-only. This
> is the reweight the align head structurally cannot perform.

That is my §4 `a`-family, in cosine space, with `w ≡ a`. Measured (MHC-EN **dev**, n=80, `:110-114`):
`w=0.00` (text-only) dev acc **0.7375**, Δ vs CLIP-concat **−0.0250**; `w=0.50` 0.7500; A3a Qwen-align
0.7250. All three inferential guards failed on the best cell (`:156-158`): bootstrap CI
**[−0.0625, +0.150]**, selection-null **p = 0.766** against a shuffled-label max-over-`w` mean of
**+0.076**, and the pre-declared kill `d_oracle = +0.025 < +0.03` fired (`:172`).

*Ruling.* **PRE-CLOSED with one disclosed gap.** F50's arms ran on **MHC-EN primary + HateMM control
only** — ZH was never in that grid (`VSW_ASYMMETRY_RECON.md:397-398` records the same limit), and F50 read
**dev**, not train-LOO. The letter of F50's ban is explicit
(`FUSIONSWAP_FORENSIC_RECON.md:54-56`, quoting `directions_tried.json` F50):
> "…do not re-propose **fixed compositions, reweights, or per-modality temperatures** over banked frozen
> features; conversion requires adaptation (F45) or a new information source with alignment>0.663 (F49
> bar)."

A fixed stream reweight is **"fixed compositions, reweights"** verbatim. **BINDING.**

### A3 · F58 / `HATEMM_LORA_STREAM_DECOMP.md` — **NOT BINDING (stream diagnostics only)**

F58 measured **per-stream kNN AUC** on raw features, train-LOO and dev (`:81-85`) — image 0.826/0.817/0.821,
text 0.847/0.888/0.920, concat 0.867/0.883/0.909 across CLIP/frozen/LoRA. It **never ran a text-only
retrieval key through the deployed head and kNN vote.** Its downstream numbers (`:119-124`) are the banked
fused test arms. So: stream-level diagnostic, not a deployed measurement.

Two things F58 *does* contribute. (i) Its own table already shows the fusion tax on **HateMM**: LoRA
text-only train-LOO AUC **0.920 > concat 0.909**, while on **dev** the ordering reverses (text 0.899 <
concat 0.910) — the train↔held-out reversal this recon re-finds in §5. (ii) Its verdict is that the
HateMM pass is **text-carried on a strong swap-neutral image base** (`:52-73`), not image-borne, and the
paper was errata'd accordingly (`research-wiki/DRAFT_analysis_chapter.md:509-511`).

### A4 · F86 / `LSMI_GATE_RECORD.md` — **PARTIALLY BINDING; the tasking's `U1 = 0` shorthand needs the record's own qualifiers**

The VSW recon's caution is correct and I restate it exactly. The **mechanical verdict at `d* = 16` is
`INDETERMINATE`**, per-dataset `{MHC_zh: INDETERMINATE, HateMM: INDETERMINATE, MHC_en: FUSION_CAPPED}`
(`LSMI_GATE_RECORD.md:566-577`). But the record is explicit that the INDETERMINATE is **not** about
synergy (`:583-596`):
> **The synergy question is answered, and the answer is "none", everywhere.** … **There is no image×text
> synergy to fuse.** … The largest atom is **not** redundancy — it is **text uniqueness** `U2` … while
> **image uniqueness `U1` is pinned at exactly 0.0000 on 5 of 6 certified cells**. So `R > U1 + U2` fails
> on ZH and HateMM, *not* because synergy is present but because the pair is **uniqueness-dominated and
> the uniqueness is all on the text side**.

**The cells matter.** From the A7 table (`:488-504`): `U1 = −0.0000 / −0.0000` (ZH crossfit/dev at d*=16),
`−0.0000 / 0.0000` (HateMM), and **`−0.0836` on the MHC-EN d*=16 crossfit cell** — that is the 6th cell,
and it is EN. Saying "`U1 = 0` everywhere" **overstates**; the record's own wording is "pinned at exactly
0.0000 on **5 of 6** certified cells" (`:538, :592-593`).

Its own scope limits (`:610-615`) are binding on this recon too:
> **What this gate does NOT support:** any claim about what a *differently trained encoder* would yield
> (§5.2); any claim about synergy in the 26–47 % of per-stream variance outside the certified subspace;
> and any prediction of the in-flight fusion-concat family's numbers — `concat` has strictly more capacity
> than `align` to exploit `U1`/`U2`, and this gate says `U1 ≈ 0`, so it is *consistent with* a small
> effect there, but that family's verdict is its own and is not adjudicated here.

*Ruling.* F86 **supports** "there is no synergy for a fusion block to capture" and "the uniqueness is
text-side". It does **not** by itself license "therefore delete the image stream" — deleting a **redundant**
stream (`R = 0.069–0.178` on every cell) can still cost accuracy, and §4.4 measures that it costs fix
supply on 3/3.

### A5 · F65 vision-LoRA · F67 frame budget · F91 Molmo2 — **BINDING as a pattern: the image stream moved three times and converted nothing**

* **F65** (`VISION_UNFREEZE_VERDICT_REVIEW.md:268-272`): *"**EN image-MOVED gate:** MOVED →
  EN-HEAD-PROCEEDS (dAUC +0.0320 train-LOO / +0.0065 dev; reproduced bit-for-bit)."* … *"**K-V2
  (add-over-generic, the DECISIVE ViT bar): TIE on both datasets, both protocols — NO ViT
  contribution.**"*
* **F67** (`FRAME16_VERDICT_REVIEW.md:157-159`): *"`HateMM-16f: final-epoch: FAIL (Δacc +0.0015 / ΔmF1
  +0.0020, acc sign 1/3, < +0.030 bar); val-selected: FAIL (Δacc −0.0077 / ΔmF1 −0.0086, acc sign 0/3);
  KS-16f-dead: KILLED`"* — doubling the frame budget does not move the deployed number.
* **F91** (`MOLMO2_PROBE_RECORD.md:104-107`): *"**It bought nothing.** The trained head still lands 0.0217
  *below* the floor. This is the **9th law-I datum**, and the cleanest one yet: the image side moved by
  +0.056 in raw retrieval and the deployed number went **down**."*

**F91 also supplies the single most important control for this recon**, and it is a counterexample to the
inference the tasking is contemplating (`MOLMO2_PROBE_RECORD.md:118-122`):
> **Hadamard degenerates.** Molmo2's raw elementwise product collapses to acc 0.5628 … The deployed fusion
> is Hadamard but in *head* space after learned projections, so this is diagnostic rather than deployed;
> it is nonetheless the most plausible mechanism for why a **better raw concat (0.8186, the best fused raw
> read of all three arms) trains to a *worse* head.**

**A better raw retrieval key has already been observed to produce a worse deployed head, on HateMM, in this
project.** Any raw-space positive in §4 inherits that counterexample.

### A6 · Existing text-only / image-only **DEPLOYED** numbers — **THIS SETTLES IT**

**(a) Train-LOO, raw arena, deployed operator, F95 protocol — already banked, with macro-F1.**
`scripts/analysis/mechnov_pairverify_{hatemm,zh,en}_OUT.json → spaces.{fused,text,img}.pooled`, re-read
this session:

| dataset | fused acc / mF1 | **text-only acc / mF1** | image-only acc / mF1 |
|---|---|---|---|
| HateMM | 0.8441 / 0.8419 | **0.8441 / 0.8422** | 0.7688 / 0.7561 |
| MHC-ZH | 0.8480 / 0.8281 | **0.8636 / 0.8442** | 0.7012 / 0.6083 |
| MHC-EN | 0.7796 / 0.7286 | **0.8106 / 0.7785** | 0.6995 / 0.5561 |

(These are the source of VSW §3.6's `Δ +0.0000 / +0.0156 / +0.0310`. My §4 reproduces all six of these
figures independently to 4 dp.)

**(b) TEST, deployed — banked on all three datasets. This is the decisive block.**

* **HateMM, head space, 3 seeds, both protocols** (`ERRPAT_HateMM_2026-07-26.md:514-524`), verbatim:
  > **3. Text-only arm — PAPER VALUE, not a performance bet. Measured here: +0.0047 acc (val-sel, sign
  > 2/3) / +0.0093 acc (final, sign 3/3), 3-seed.** In the trained head space, kNN over the text
  > projection alone scores 0.8822/0.8853 vs the deployed 0.8775/0.8760, while the image projection alone
  > scores 0.7411-0.7426. Both deltas are inside the ±0.014 seed band and far under the +0.030 bar, so
  > this is **measured-not-promoted** … Note the read here is post-hoc (the `text_proj` sub-space of a
  > head trained under Hadamard fusion), so a paper-grade text-only ablation needs a properly trained
  > text-only arm.

  Re-read from source: `scripts/analysis/errpat_hatemm_ceilings_OUT.json → stream_means` =
  `valsel_txt_acc 0.8822 / valsel_txt_mF1 0.8759 / valsel_deployed_acc 0.8775 / valsel_deployed_mF1
  0.8715 / final_txt_acc 0.8853 / final_txt_mF1 0.8797 / final_deployed_acc 0.8760 / final_deployed_mF1
  0.8699 / valsel_img_acc 0.7426 / final_img_acc 0.7411`.

* **MHC-ZH, raw arena, test** (`ERRPAT_MHC-ZH_2026-07-26.md:246-256`): image-only 0.7047, **text-only
  0.8523**, fused L2-concat **0.8456**; *"the image stream is near-useless on ZH and fusion adds no
  accuracy over text alone (text-only 0.8523 vs fused 0.8456 = **1 item**, and the deployed 3-seed mean is
  also 0.8456)."* Reconfirmed at `:420-422`: *"Same vote operator on the pre-head L2-concat key gives
  0.8456 — exactly the deployed 3-seed mean — and text-only raw gives 0.8523."*

* **MHC-EN, raw arena, test** (`ERRPAT_MHC-EN_2026-07-26.md:241-257`): Qwen image-only 0.7267/mF1 0.5899,
  **Qwen text-only 0.7826 / mF1 0.7448**, *deployed fused (ARM-V, 4 seeds)* **0.7935 ± 0.0205 / 0.7497 ±
  0.0250**. Text-only is **−0.0109 acc / −0.0049 mF1 BELOW** the deployed number, and the record's own
  reading is *"**the entire trained stack buys ~+0.011 acc / +0.005 mF1 over a raw text 20-NN with no head
  at all** (0.7935 vs 0.7826; well inside the 4-seed ±0.0205 band)."*

* **HateMM, raw arena, test, on the deployed encoder** (`MOLMO2_PROBE_RECORD.md:84-95`, bank = train,
  query = test, deployed operator): LoRA-curric **text 0.8233 / 0.8208** vs **concat 0.8140 / 0.8118**,
  against a same-path head floor of **0.8775 / 0.8760** (`MOLMO2_PROBE_RECORD.md:55`).
  **ERRATUM, self-caught: this pair is NOT matched to the deployed fused key and must not be quoted as
  one.** F91's `concat` view is `np.concatenate([img, txt])` with the L2-normalisation applied only to
  the *whole* concatenated vector at scoring time (`scripts/analysis/molmo2_geom_diag.py:71-77`), i.e.
  `l2n(concat(img, txt))` — **without the per-stream normalisation** that the deployed key
  `l2n(concat(l2n(img), l2n(txt)))` and every arm in §4 use. The two differ whenever the streams have
  unequal norms. The `+0.0093` is therefore **indicative only** and is withdrawn from every matched
  comparison in §5.2.4 / §5.3. **What does survive**, because it does not depend on how `concat` is
  built: the best raw key of any kind on HateMM is 0.8233 against a head floor of 0.8760, so **the
  trained head is worth +0.0527 over the best raw key** — 5.7× the raw-space gap being argued about.

* **The paper already banks it** (`research-wiki/DRAFT_analysis_chapter.md:937-940`): *"the campaign's
  honest text-only-versus-fused ablation reads are now measured rather than asserted (HateMM text-only
  0.8822 / 0.8853 against the deployed 0.8775 / 0.8760, both inside the ±0.014 seed band, with the
  apparent image complementarity killed by its own error arithmetic — the image stream fixes 11–14
  deployed errors and breaks 40–43 items the fusion gets right)."*

*Ruling.* **PRE-CLOSED.** Deployed text-only numbers exist on **3/3 datasets on TEST**, and the direction
is **0-for-3** against the +0.030 bar, with EN's sign **negative**.

### A7 · D7 — **is a stream-selection change usable as novelty? NO, and it is ruled by name**

`ENCODER_SWAP_DIAGNOSIS.md:198-200`, verbatim:
> **"Down-weight/gate the collapsed image stream on MHC"** = modality gating / learned fusion weights =
> textbook, decision-side (Axis A conditional-redundancy), D7-dead; the trained head already has
> attenuation capacity and still failed on test.

The standing ruling it invokes (`D7_RULING_DOSSIER.md:42-46`, binding, quoted from
`DECISION_MEMO_pending.md:80-85`):
> **编排解读(binding):D7 = RESOLVED-NEGATIVE。** encoder-class 杠杆——frozen swap、LoRA-adapted swap,
> 及推而广之的通用决策规则校准(如 B5)——**均不满足 goal 的 novelty 子句**;它们保留为**合法的性能 /
> 消融 / 诊断素材**。

F85's prereg says the same thing unconditionally for the fusion axis
(`FUSIONCAT_VERDICT_REVIEW.md:291-295`): *"The fusion operator is a **generic architecture/capacity knob**.
This outcome is a **door-closer for the fusion axis** and yields **NO novelty contribution** — exactly as
it would have yielded none had it passed."*

**Ruling: D7-DEAD, up front.** Even a clean positive would be a performance / ablation / diagnostic row,
never a novelty win. A second, independent objection worth stating: the goal clause is *NOVEL MECHANISM ×
**MLLM-integrated** × ≥+3 acc* (`D7_RULING_DOSSIER.md:45, 51`). A method whose retrieval key **deletes the
MLLM's visual stream** is a weaker MLLM integration, not a novel one.

---

## 3. TASK B — PARITY (asserted before any new number was read)

Frozen module `scripts/analysis/mechfix_ops.py` used **unmodified**, sha256 re-verified this session =
`635c13124e79ba1a299bc13fc1175a03aa11e09924f5413ce51061793c83fc8d` (asserted in-script; the script aborts
on mismatch). Protocol: `StratifiedKFold(5, shuffle=True, random_state=0)` over the train split,
item-disjoint, bank = the 4 fitting folds, queries = the held-out fold, `topk=20`, pooled over 5 folds —
identical to F95 (`scripts/analysis/mechnov_pairverify.py:57, 210, 219, 280`).

| gate | HateMM | MHC-ZH | MHC-EN |
|---|---|---|---|
| pooled deployed train-LOO **acc**, recomputed | **0.8441** | **0.8480** | **0.7796** |
| banked anchor (`MECHNOV_PAIRVERIFY_PREGATE.md:296-300`; VSW §2) | 0.8441 | 0.8480 | 0.7796 |
| pooled deployed train-LOO **mF1**, recomputed | **0.8419** | **0.8281** | **0.7286** |
| banked anchor (`mechnov_pairverify_*_OUT.json → spaces.fused.pooled.mF1_deployed`) | 0.8419 | 0.8281 | 0.7286 |
| **anchor gate** | **PASS** | **PASS** | **PASS** |
| `a=0.5` key vs the deployed fused construction, max abs elementwise deviation | **0.0** | **0.0** | **0.0** |

**Both gates PASS 3/3, and `a = 0.5` reproduces the deployed fused key bit-exactly** (max |Δ| = 0.0 in
float64, on all three datasets — not merely to tolerance). ZH's 0.8480 matches the F95 anchor at 4 dp as
required.

**FROZEN `a`-GRID, declared in the script header before any judged number was produced** (13 points):
`0.00 0.10 0.20 0.25 0.30 0.40 0.50 0.60 0.70 0.75 0.80 0.90 1.00`, with
`K(a) = l2n(concat(a·l2n(img), (1−a)·l2n(txt)))`, 7168-d. `a = 0.00` ≡ text-only, `a = 1.00` ≡ image-only.

---

## 4. TASK B — MEASUREMENTS (every number in this section is **newly computed this session**)

Artifacts: `<scratchpad>/streamcomp_recon.py` → `streamcomp_OUT.json`;
`<scratchpad>/streamcomp_nested.py` → `streamcomp_nested_OUT.json`.
Encoders = the F95/VSW arena map: HateMM `Qwen2.5-VL-7B-Instruct-LoRA-curric_HF`, MHC_zh
`Qwen2.5-VL-7B-Instruct-LoRA_HF`, MHC `Qwen2.5-VL-7B-Instruct_HF`
(`scripts/analysis/mechnov_pairverify.py:79-85`).

### 4.1 Deployed vote on each key — acc and macro-F1 (raw train-LOO)

| dataset | (i) fused `a=0.5` | (ii) **text-only `a=0`** | (iii) image-only `a=1` |
|---|---|---|---|
| HateMM (n=744) | 0.8441 / 0.8419 | **0.8441 / 0.8422** (Δ **+0.0000 / +0.0003**) | 0.7688 / 0.7561 |
| MHC-ZH (n=579) | 0.8480 / 0.8281 | **0.8636 / 0.8442** (Δ **+0.0156 / +0.0161**) | 0.7012 / 0.6083 |
| MHC-EN (n=549) | 0.7796 / 0.7286 | **0.8106 / 0.7785** (Δ **+0.0310 / +0.0499**) | 0.6995 / 0.5561 |

All six banked F95 figures reproduce to 4 dp. **Text-only clears the +0.030 acc **and** mF1 conjunct on
exactly ONE dataset (EN) in raw train space, and is a flat 0.0000 on HateMM.**

### 4.2 (iv) The frozen `a`-sweep — full-hindsight

| `a` | HateMM acc / mF1 | MHC-ZH acc / mF1 | MHC-EN acc / mF1 |
|---|---|---|---|
| 0.00 | 0.8441 / 0.8422 | 0.8636 / 0.8442 | 0.8106 / 0.7785 |
| 0.10 | **0.8481 / 0.8461** | 0.8653 / 0.8460 | **0.8179 / 0.7863** |
| 0.20 | 0.8441 / 0.8422 | 0.8756 / 0.8582 | 0.8087 / 0.7745 |
| 0.25 | 0.8333 / 0.8315 | **0.8808 / 0.8636** | 0.7978 / 0.7616 |
| 0.30 | 0.8401 / 0.8382 | 0.8791 / 0.8625 | 0.7942 / 0.7565 |
| 0.40 | **0.8481 / 0.8463** | 0.8601 / 0.8412 | 0.8051 / 0.7678 |
| **0.50 (deployed)** | 0.8441 / 0.8419 | 0.8480 / 0.8281 | 0.7796 / 0.7286 |
| 0.60 | 0.8320 / 0.8286 | 0.8135 / 0.7849 | 0.7468 / 0.6806 |
| 0.70 | 0.8011 / 0.7950 | 0.7789 / 0.7326 | 0.7304 / 0.6310 |
| 0.75 | 0.7944 / 0.7860 | 0.7513 / 0.6937 | 0.7213 / 0.6114 |
| 0.80 | 0.7876 / 0.7784 | 0.7427 / 0.6711 | 0.7140 / 0.5851 |
| 0.90 | 0.7728 / 0.7605 | 0.7081 / 0.6207 | 0.6976 / 0.5574 |
| 1.00 | 0.7688 / 0.7561 | 0.7012 / 0.6083 | 0.6995 / 0.5561 |

**Hindsight best (`a` chosen on the evaluation data itself):** HateMM `a=0.10`, **+0.0040 / +0.0042**;
ZH `a=0.25`, **+0.0328 / +0.0355**; EN `a=0.10`, **+0.0383 / +0.0577**. Conjunct met on **2 of 3** —
but this is a hindsight ceiling, and HateMM caps at +0.0040 no matter what.

**Two structural readings.** (a) **Text-only is never the argmax.** The optimum sits at a small but
*non-zero* image weight (`a* ∈ {0.10, 0.25, 0.10}`) on all three — so "the image stream is pure poison,
delete it" is measured **false**; the image stream is worth a little, at about a fifth of its deployed
weight. (b) **The curve is jumpy where it matters.** On HateMM adjacent grid points swing
0.8441 → 0.8481 → 0.8441 → 0.8333 → 0.8401 → 0.8481 → 0.8441, i.e. ±0.015 (±11 items) between
near-identical compositions — the exact noise signature F50 flagged on the same object
(`FA_GATE_RECORD.md:125-126`: *"a **jumpy, non-monotonic** function of w (adjacent near-identical
compositions swing acc by ±0.04 = ±3 videos), the noise signature"*), and whose selection-null gave
**p = 0.766** (`:158`).

### 4.3 The deployable read — `a*` selected per fold, never touching the held-out fold

Selection rule declared before running: for each outer fold, `a* = argmax` over the frozen grid of the
**fitting fold's own internal leave-one-out accuracy** (bank = fitting fold, queries = fitting fold,
`exclude_self=True`); ties broken toward `a = 0.50`, then toward smaller `a`. Applied to the held-out fold,
pooled over 5 folds.

| dataset | `a*` per fold | fused acc / mF1 | **nested acc / mF1** | **Δ acc / Δ mF1** | conjunct (+0.030 both)? |
|---|---|---|---|---|---|
| HateMM | 0.40, 0.50, 0.00, 0.50, 0.40 | 0.8441 / 0.8419 | 0.8414 / 0.8398 | **−0.0027 / −0.0021** | **NO** |
| MHC-ZH | 0.30, 0.25, 0.25, 0.25, 0.30 | 0.8480 / 0.8281 | **0.8826 / 0.8657** | **+0.0346 / +0.0376** | **YES** |
| MHC-EN | 0.25, 0.00, 0.00, 0.20, 0.00 | 0.7796 / 0.7286 | 0.7996 / 0.7649 | **+0.0200 / +0.0363** | **NO** (acc) |

**This is the finding.** In the arena maximally favourable to the direction — raw keys, train split,
leave-one-out, no head, no transfer, no protocol penalty, a genuinely deployable selector — the goal
conjunct is met on **exactly one dataset of three**. HateMM is *negative*; EN misses on accuracy by
0.0100. **The ≥2-dataset requirement fails on arithmetic**, before any of the four independent objections
in §5 is applied.

### 4.4 Break exposure and fix supply per stream configuration (verifier-free; see §1.2)

Flip cost = the minimum probability mass that must be moved between the 20 rank weights to drive the
deployed convex score `s = Σ p_j (2·lab_j−1)·cos_j` across 0 (closed-form optimal transport: drain the
current-sign entries, largest |v| first, into the single most extreme opposite-sign entry) — VSW §3.1's
definition, verifier term omitted because the F95 verifier is fused-space-fitted over fused neighbour sets.

| θ = 0.10 | HateMM fused → text | MHC-ZH fused → text | MHC-EN fused → text |
|---|---|---|---|
| **break exposure** = cheap∧correct / n_COR | **0.0653 → 0.0717 (↑ WORSE)** | 0.1283 → **0.1040** | 0.2126 → **0.1640** |
| **fix supply rate** = cheap∧wrong / n_ERR | 0.3103 → **0.2672** (↓) | 0.5000 → **0.3544** (↓) | 0.5041 → **0.3942** (↓) |
| n_ERR | 116 → 116 | 88 → **79** | 121 → **104** |
| median top-20 purity, CORRECT | 0.85 → 0.90 | 0.75 → 0.85 | 0.70 → 0.75 |
| median top-20 purity, WRONG | 0.325 → 0.250 | 0.400 → 0.350 | 0.400 → 0.400 |

| θ = 0.20 | HateMM | MHC-ZH | MHC-EN |
|---|---|---|---|
| break exposure fused → text | 0.1895 → **0.1433** | 0.2974 → **0.2060** | 0.4369 → **0.3438** |

**The fusion-tax prediction is half-right and its HateMM leg is refuted.** Dropping the image stream does
lower break exposure on ZH (−0.024) and EN (−0.049) at θ=0.10, exactly as the account predicts — but on
**HateMM it raises it** (0.0653 → 0.0717), and on all three it lowers fix supply by 4–15 points. The
correct set gets *purer* and the error set gets *more inverted* (HateMM error purity 0.325 → 0.250):
text-only buys a cleaner decision at the price of a less rescuable error population. **Exposure is traded,
not removed** — and the asymmetry that VSW identified is therefore **not repairable by dropping the image
stream**, which is precisely the tasking's hypothesis, measured false.

### 4.5 Top-20 membership overlap — the only channel that matters

| dataset | mean \|top20(text) ∩ top20(fused)\| | as a fraction | prediction agreement with fused | image-only overlap |
|---|---|---|---|---|
| HateMM | 7.425 / 20 (median 7) | **0.3712** | 0.9032 | 11.128 / 20 (0.5564) |
| MHC-ZH | 9.933 / 20 (median 10) | **0.4966** | 0.9085 | 9.971 / 20 (0.4985) |
| MHC-EN | 7.659 / 20 (median 8) | **0.3830** | 0.8379 | 10.362 / 20 (0.5181) |

Text-only replaces **50–63 % of every neighbourhood** yet moves the decision on only 9.2 / 9.2 / 16.2 % of
items. Two consequences. (i) The vote is extraordinarily insensitive to membership: it takes a majority
turnover of the retrieved set to move one item in ten. That is a *stability* result and it is bad news for
any membership-side operator, not just this one. (ii) On HateMM the image-only key overlaps the fused key
**more** than the text-only key does (0.5564 vs 0.3712) while scoring 7.5 points worse — so fused-key
membership on HateMM is image-dominated even though the *decision* is text-carried (F58). That
dissociation is a genuinely new observation and it is a paper sentence, not a lever.

### 4.6 DECLARED IN ADVANCE — this is a raw-space read and does not entail a head-space result

Stated before the numbers, per F97 limitation (1) and F95's own §2.1
(`MECHNOV_PAIRVERIFY_PREGATE.md:164-165`): *"**The raw space is the honest arena for a pregate; the price
is that a raw-space result does not transfer automatically to the deployed head space in either
direction.**"* The deployed head is trained on **fused** inputs and its `img_proj` / `text_proj` are
stream-specific, so a raw-key result is a diagnosis of the banked key space, not of the deployed object.
**The project holds a banked counterexample in the required direction** (F91, §A5): Molmo2's raw concat
0.8186 was the best raw fused read of all three arms and trained to a head **0.0217 below floor**.

**What a head-space verdict would cost, priced honestly:**

* **Text-only arm.** No such branch exists — `src/model/classifier.py:85-90, 138-143` implements only
  `concat` / `align` / `cross`, and the mod-dropout path (`:129-136`) is `self.training`-gated and
  coin-flipped, so it cannot express a permanent single stream. Needs a new `fusion_mode='text_only'`
  branch (`input_shape = map_dim`; `x = text_feats`) plus arg plumbing — ~6–10 lines, i.e. a **code diff**,
  which under house style draws a **mandatory codex-code-review gate + prereg-freeze + independent
  0-context review** (`FUSIONSWAP_FORENSIC_RECON.md:35, 83`).
  Compute: reuses `scripts/slurm/ncafam_family.sbatch` on cached features; the NCA precedent ran **24 head
  runs in 9m28s** (`FUSIONSWAP_FORENSIC_RECON.md:85`), so 2 datasets × 3 seeds = 6 runs ≈ **3 min wall,
  ~0.05 GPU-h**. A $0 CPU alternative exists at 52 s/run (`ERRPAT_HateMM_2026-07-26.md:527-529`) but
  **must be paired against a CPU-trained floor**, never the banked GPU floor (F87 merge-drift discipline,
  `:533-536`).
* **Stream-weighted (`a`) arm — NOT PURCHASABLE AT ANY PRICE under the deployed fusion.** Under
  `align`/Hadamard on pre-L2-normalised projections (`src/model/classifier.py:115-141`),
  `(a·î) ⊙ ((1−a)·t̂) = a(1−a)·(î ⊙ t̂)`: the weight collapses to a **single global scalar** on the fused
  vector — a temperature on the head input, not a per-stream reweight. Hadamard destroys the per-stream
  ratio by construction. The only architecture that can carry `a` is `concat`, which is
  **F85-KILLED on both datasets and both protocols** with **scope explicitly frozen against exactly this
  follow-up** (`FUSIONCAT_VERDICT_REVIEW.md:275-279, 305-313`). So the §4.3 ZH cell has **no deployable
  realisation** inside the current head.

---

## 5. TASK C — VERDICT

### 5.1 Does a text-only or stream-reweighted key beat fused on ≥2 datasets in raw space?

**Text-only: NO** — the conjunct (+0.030 acc AND +0.030 mF1) is met on **1 of 3** (EN only:
+0.0310 / +0.0499). HateMM is +0.0000 / +0.0003.
**Stream-reweighted, full hindsight: YES on 2 of 3** (ZH +0.0328/+0.0355, EN +0.0383/+0.0577; HateMM
+0.0040/+0.0042).
**Stream-reweighted, deployable: NO — 1 of 3** (ZH +0.0346/+0.0376; EN +0.0200 acc misses; HateMM
−0.0027).

The honest headline is the third line, and the gap between the second and third is the whole story: the
2-dataset version is a hindsight ceiling, and the deployable selector — which is *better* than the ceiling
on ZH and *worse* on EN — converts it to 1-of-3. That is F98's ceiling-vs-delivery law showing up again in
a new family.

### 5.2 The objections, priced

1. **D7 — is it novel? NO, ruled by name.** `ENCODER_SWAP_DIAGNOSIS.md:198-200` classes a stream
   down-weight/gate as *"modality gating / learned fusion weights = textbook, decision-side (Axis A
   conditional-redundancy), **D7-dead**"*, under a binding RESOLVED-NEGATIVE ruling
   (`D7_RULING_DOSSIER.md:42-46`). F85's prereg says the fusion axis yields **no** novelty *whether it
   passes or fails* (`FUSIONCAT_VERDICT_REVIEW.md:291-295`). Secondary: deleting the MLLM's visual stream
   weakens, rather than establishes, the goal's *MLLM-integrated* clause.
2. **Head-retrain requirement — the arm doesn't exist and the reweight is unrepresentable.** §4.6: text-only
   needs a code diff + codex gate + prereg + review for ~0.05 GPU-h; the `a`-family is **structurally
   annihilated by Hadamard** and can only be realised through `concat`, which is F85-killed with scope
   frozen. And the raw→head inference has a banked counterexample (F91, image side +0.056 raw, head
   −0.0217).
3. **Is it just re-deriving F44/F85/F50? Substantially yes.** F44 is the mechanism
   (`ENCODER_SWAP_DIAGNOSIS.md:116-124`); F50/FA ran the identical convex `w`-sweep with `w→0` =
   text-only and killed it on all three inferential guards (`FA_GATE_RECORD.md:38-41, 110-114, 156-158,
   172`); F85 killed the only deployable carrier. The genuinely new content of this recon is §4.3 (the
   deployable selector is 1-of-3), §4.4 (the fusion-tax exposure prediction fails on HateMM), §4.5
   (membership turnover of 50–63 % moves ≤ 1 item in 6), and §4.6 (the Hadamard annihilation argument).
   None of those is a lever.
4. **Transfer — the raw train-LOO read does not predict the held-out read, and on HateMM the same
   quantity inverts sign.** Full arena-by-arena accounting, with the matched/unmatched distinction made
   explicitly, is in **§5.3**; it is stated there rather than here because it is a statement about the
   *instrument*, not about this candidate.

### 5.3 INSTRUMENT VALIDITY — raw train-LOO vs held-out, stated as MATCHED and UNMATCHED pairs

This section is load-bearing outside this recon, so the **comparability of each pair is stated before its
number**. The quantity throughout is `Δ = TEXT − FUSED`, where
`FUSED = l2n(concat(l2n(img), l2n(txt)))` and `TEXT = l2n(txt)`, under the deployed top-20 rank-weighted
signed-cosine vote.

**(a) MATCHED PAIR — MHC-ZH, accuracy. Same sign, shrinks 2.3×.**

| side | arena | split | bank | n | protocol | seeds | **Δ (text − fused)** |
|---|---|---|---|---|---|---|---|
| train | raw banked keys | train | 4/5 of train (≈463) | 579 pooled | StratifiedKFold(5, shuffle, rs=0), item-disjoint, pooled over 5 held-out folds | deterministic, no head | **+0.0156** acc / **+0.0161** mF1 |
| held-out | raw banked keys | **test** | full train (579) | 149 | single draw | deterministic (features seed-independent) | **+0.0067** acc (**= 1 item**) |

Encoder `Qwen2.5-VL-7B-Instruct-LoRA_HF` on **both** sides. Key construction verified identical: mine
(§3) vs `scripts/analysis/errpat_zh_taxonomy.py:292-300`
(`fused_concat_l2n = hstack([l2n(tr_img), l2n(tr_txt)])`), scored by a *"Rank-weighted signed-cosine
top-k vote in a raw (pre-head) key space"* (`:94`). **Disclosed differences:** split, bank size
(463 vs 579), n. Nothing else differs.

**(b) MATCHED PAIR — HateMM, AUC. A SIGN INVERSION.**
F58 `HATEMM_LORA_STREAM_DECOMP.md:81-85`, machinery `scripts/analysis/encoder_swap_geometry.py:64`
(`np.concatenate([l2n(img), l2n(txt)])` — matched to the deployed construction), LoRA encoder, k=20 kNN:

| side | split | n | text-only AUC | concat AUC | **Δ (text − concat)** |
|---|---|---|---|---|---|
| train-LOO | train | 744 | **0.920** | 0.909 | **+0.011** |
| held-out | **dev** | 107 | 0.899 | **0.910** | **−0.011** |

**Same object, same construction, same encoder — and the sign flips between train-LOO and held-out.**
This is the cleanest matched instrument-validity datum in the set. Caveats to carry when citing it: it is
**AUC, not accuracy**, and **n_dev = 107**, so ±0.011 is small in absolute terms.

**(c) NOT MATCHED — MHC-EN. The largest raw advantage in the study, and NOT a demonstrated sign inversion
of the same quantity. Do not cite it as one.**
*Train side (raw ↔ raw):* Δ = **+0.0310** acc / **+0.0499** mF1; encoder `Qwen2.5-VL-7B-Instruct_HF`
(frozen Qwen, **no LoRA**), n = 549, 5-fold pooled, deterministic.
*Test side:* the only banked EN comparator is raw Qwen text-only **0.7826 / 0.7448** (n = 161, single
draw, bank = train) measured against **the trained deployed pipeline** — ARM-V = frozen-Qwen → RGCL
`align` head → **archive-kNN α = 0.25** key → top-20 vote, **val-selected, 4 seeds**,
**0.7935 ± 0.0205 / 0.7497 ± 0.0250** (`ERRPAT_MHC-EN_2026-07-26.md:241-246`; arm definition `:80`).
**The two sides compare different objects** — raw-text-vs-raw-fused on one, raw-text-vs-trained-pipeline
on the other — so the −0.0109 is *not* the same quantity as the +0.0310. **No raw-fused MHC-EN test read
exists anywhere in the repo**, and producing one is a test touch, forbidden here.
The honest EN claim is therefore about **deployability, not inversion**: *a raw text-only key that leads
the raw fused key by +0.0310 on train-LOO nonetheless **loses to the deployed system** by −0.0109 acc /
−0.0049 mF1 on test* — with the record's own reading being that the entire trained stack buys only
**~+0.011 acc / +0.005 mF1** over that raw text vote (`:257`), i.e. both sit inside the 4-seed ±0.0205
band.

**(d) NOT MATCHED — HateMM, accuracy.** Train-LOO raw Δ = **+0.0000 / +0.0003**; the test-side figures
(**+0.0047** val-sel sign 2/3 / **+0.0093** final sign 3/3) are measured in the trained head's
**`text_proj` sub-space**, a post-hoc arena the source record itself flags as such
(`ERRPAT_HateMM_2026-07-26.md:523-524`). Different arenas, not a pair. The F91 raw-test HateMM figures
are **withdrawn** from matched use (§A6 erratum: `molmo2_geom_diag.py:71-77` omits the per-stream
L2-norm).

**What the pair-set supports, at exactly the strength the evidence allows.**
Matched pairs exist on **2 of 3** datasets, and on **both** the raw train-LOO Δ **fails to predict the
held-out Δ** — ZH shrinks 2.3× (same sign), HateMM **inverts sign** (in AUC). On the third, EN, the
largest raw advantage in the study coexists with the text-only key **losing to the deployed system** on
test. **The raw train-LOO key-space read is not a calibrated instrument for the held-out deployed read**,
and every §4 number here — including its own §4.3 ZH-only positive — must be consumed with that discount.
*Partial* mechanism, offered as partial: the ZH and HateMM encoders are LoRA-adapted **on their own train
split**, and that adaptation moved the **text** stream specifically
(`B3_ZH_LORA_DECOMPOSITION.md:53, 62`: ZH text **+0.078** train-LOO / **+0.062** dev;
`HATEMM_LORA_STREAM_DECOMP.md:84`: HateMM text **+0.0317** / **+0.0236**), so a train-LOO read that
up-weights text up-weights the stream that saw the split. **EN carries no LoRA and still fails to
transfer**, so adaptation contamination is not the whole story.

### 5.4 If it does not convert — what that closes

It closes the most obvious exploit of the fusion-tax account and it does so with a mechanism rather than a
p-value: **the exposure asymmetry is NOT repairable by dropping the image stream.** Text-only trades
break exposure for fix supply (§4.4), *raises* exposure on HateMM, and the optimum of the whole family
sits at a **non-zero** image weight on 3/3 — so the image stream is not a pure tax. Combined with F86
(`R = 0.069–0.178` shared, `U1 ≈ 0`), the correct sentence is: *the image stream carries no unique
information but does carry redundant information that the vote uses to rescue errors; removing it buys
purity on the correct set and pays for it on the error set, and the exchange rate is under bar on 3/3
datasets on test.*

### 5.5 Verdict, probability, ruling

**VERDICT: PRE-CLOSED (Task A) and independently KILLED (Task B). No candidate is nominated. Zero GPU
recommended.** The direction is closed by five independent walls, any one of which is sufficient: (i) it
is measured on TEST on 3/3 datasets and is 0-for-3, with EN negative; (ii) the deployable raw-space read
meets the conjunct on 1 of 3 and the goal needs ≥2; (iii) the reweight is unrepresentable under the
deployed Hadamard fusion and its only carrier (`concat`) is F85-killed with scope frozen; (iv) F50 ran the
identical sweep and it failed bootstrap CI, selection-null (p = 0.766) and the oracle-threshold kill; (v)
it is D7-dead by name, so a positive would be an ablation row, not a contribution.

**P(this reaches +0.030 acc AND +0.030 mF1 on ≥2 datasets, 3/3 seeds, final-epoch protocol): ~1 %.**
Reasoning, not vibes: the deployable raw ceiling is already 1-of-3 (§4.3), so the conjunct requires a
second dataset that the *most favourable possible* measurement says is not there; the one leg with both
arenas shrinks 2.3× and the biggest one flips sign (§5.2.4); the head-side base rate for this project is
~0-for-22 promoted; and F91 supplies a banked instance of a better raw key producing a worse head. The
residual ~1 % is the ZH-only scenario where a properly trained `a≈0.25` head holds +0.0346 to test — which
would still be **1 dataset**, hence goal-irrelevant.

**Does it need a user ruling? NO.** Nothing here is user-gated. D7 already rules the object dead by name
(`ENCODER_SWAP_DIAGNOSIS.md:198-200`), F85's frozen scope already forbids the follow-up
(`FUSIONCAT_VERDICT_REVIEW.md:275-279`), and no download, no cross-dataset mixing, no OCR and no test
touch is implicated. The orchestrator can close this without escalating. **One optional, non-goal item**
the user may want on the record: F88 flagged that *"a paper-grade text-only ablation needs a properly
trained text-only arm"* (`ERRPAT_HateMM_2026-07-26.md:523-524`) — that is a ~0.05 GPU-h / $0-CPU
door-closer for the paper's ablation table, explicitly **not** a goal bet, and it is the only reason to
ever build the `text_only` branch.

---

## 6. WHAT THIS RECON ADDS TO THE RECORD

1. **The deployable stream-weight selector meets the goal conjunct on exactly 1 of 3 datasets**
   (ZH +0.0346/+0.0376; EN +0.0200/+0.0363; HateMM −0.0027/−0.0021), against a full-hindsight ceiling of
   2 of 3 — a fresh instance of F98's ceiling-vs-delivery law in a new family (§4.3).
2. **The fusion-tax account's exploit is measured and fails**: text-only *raises* break exposure on HateMM
   (0.0653 → 0.0717) and lowers fix supply on 3/3. Exposure is traded, not removed (§4.4).
3. **The optimum of the stream-weight family sits at a non-zero image weight on 3/3** (`a* ∈ {0.10, 0.25,
   0.10}`) — "the image stream is pure poison" is measured false (§4.2).
4. **The stream weight `a` is structurally unrepresentable under the deployed `align`/Hadamard fusion**
   (it collapses to a global scalar), so the only carrier is `concat` = F85-killed, scope frozen (§4.6).
   This is a general result about the deployed architecture, not about this candidate.
5. **A membership-stability datum for future membership-side operators**: replacing 50–63 % of every
   top-20 moves the decision on only 9–16 % of items (§4.5).
6. **Independent 4-dp reproduction** of all six F95 fused/text/image acc+mF1 cells, and exact reproduction
   of VSW §3.6's `frac_COR cheap(θ≤0.10)` (0.0653 / 0.1283 / 0.2126 vs their 0.0653 / 0.1283 / 0.2118) —
   two independent implementations of the flip-cost statistic agree (§1.2, §3).
7. **A train↔test reversal warning with a number**: the raw train-LOO text-only advantage is +0.0156 on ZH
   and +0.0310 on EN; the banked test reads are +0.0067 and **−0.0109**. Raw train-LOO reads on this
   project's key spaces overstate, and can invert (§5.2.4).
8. **Premise correction**: EN's text stream purity is **0.70**, not ~0.85 (`VSW_ASYMMETRY_RECON.md:287`);
   and F86's `U1 = 0.0000` holds on **5 of 6** certified cells, the exception being MHC-EN d\*=16 crossfit
   at **−0.0836** (`LSMI_GATE_RECORD.md:500, 538, 592-593`) (§1.1, §A4).

---

## 7. FILE MANIFEST

| artefact | role |
|---|---|
| `refine-logs/STREAMCOMP_FORENSIC_RECON.md` | this record (the only file written under `refine-logs/`) |
| `<scratchpad>/streamcomp_recon.py` → `streamcomp_OUT.json` | parity gate; 13-point frozen `a`-sweep; acc/mF1; flip-cost exposure & supply; purity; top-20 overlap |
| `<scratchpad>/streamcomp_nested.py` → `streamcomp_nested_OUT.json` | deployable per-fold `a*` selection on fitting-fold internal LOO |

**Read-only inputs:** `scripts/analysis/mechfix_ops.py` (sha256 asserted in-script),
`scripts/analysis/mechnov_pairverify.py` (arena/protocol constants),
`scripts/analysis/mechnov_pairverify_{hatemm,zh,en}_OUT.json`,
`scripts/analysis/errpat_hatemm_ceilings_OUT.json`,
`data/CLIP_Embedding/{HateMM,MHC_zh,MHC}/train_*.pt`, `src/model/classifier.py`, `src/run_rac.py`,
and the records cited inline.
**Required statements:** ZERO GPU / SLURM / Modal / training / test-touch spent. No held-out test metric
was read from a raw source or produced; all test figures are transcriptions from banked records, re-read
from source this session. No `state/`, prereg, config, `research-wiki/`, `findings.jsonl`, or frozen
artifact mutated. Committed on `main` (this path only, staged individually — never `git add -A`), not
pushed.
