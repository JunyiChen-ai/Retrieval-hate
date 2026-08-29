# R15-NT / R15-FS — result

**Date** 2026-08-18 · **Freeze** `idea-stage/R15_NT_FREEZE.md`, commit **`e3740dc`**, committed
before `scripts/r15_nt/run_nt.py` existed · **Cost** ¥0 (no cloud, no API), local RTX 5090 ·
**Test-label contact: zero** — everything below is out-of-fold inside the 237-video train split.

## VERDICT: **KILL**, and the motivating premise is refuted with the sign reversed

The frozen KILL rule fires. Every mechanism arm is **worse** than plain concatenation, and gate G0 —
the premise the entire family was built on — comes out **negative and significant**.

| gate | contrast | Δ wv-AUC [95% CI] | verdict |
|---|---|---|---|
| **G0 premise** | `AUD − ALL` | **−0.0393 [−0.0690, −0.0098]** | **premise refuted** — audio alone is far *worse* than the fusion, not better |
| **P1a** | `AUDCENT − ALL` | −0.0180 [−0.0387, +0.0033] | fail |
| **P1b** | `AUDCENT − AUD` | +0.0213 [−0.0046, +0.0474] | fail (CI contains zero) |
| **P2** | `ALLCENT − ALL` | **−0.0371 [−0.0608, −0.0137]** | fail, significantly negative |
| **P2** | `AUDVIS0 − ALL` | **−0.0214 [−0.0393, −0.0038]** | fail, significantly negative |

δ = +0.010, video-clustered paired bootstrap, 10 000 resamples, seed 4399, n = 193 out-of-fold train
videos with within-video label variation, 5 seeds 4300-4304 × 5 folds per arm.

---

## 1. All seven arms

Out-of-fold, 5-fold video-grouped CV inside train, 5 seeds, no early stopping and no per-fold model
selection in any arm. Identical head, optimiser, epoch budget and grid throughout; only the input
composition differs.

| arm | input | width | **wv-AUC** (sd) | F1@tIoU .3/.5/.7 | window macro-F1 | between-video var share |
|---|---|---|---|---|---|---|
| **`ALL`** | V ⊕ T ⊕ O ⊕ A ⊕ M | 3586 | **0.5901** (0.0021) | 28.2 / 16.0 / 6.7 | 59.9 | 0.451 |
| `AUD` | audio | 1024 | 0.5507 (0.0043) | 28.7 / 13.5 / 4.0 | 56.3 | 0.435 |
| `VIS` | visual | 1024 | 0.5535 (0.0024) | 26.0 / 12.7 / 6.8 | 57.6 | 0.605 |
| `TXT` | ASR ⊕ OCR ⊕ masks | 1538 | 0.5288 (0.0028) | 28.8 / 13.8 / 5.1 | 53.4 | 0.321 |
| `ALLCENT` | all four channels within-video centered | 3586 | 0.5530 (0.0038) | 29.0 / 13.5 / 4.9 | 55.9 | **0.095** |
| `AUDCENT` | raw audio ⊕ centered V/T/O | 3586 | 0.5721 (0.0033) | 29.1 / 14.7 / 5.6 | 56.7 | 0.150 |
| `AUDVIS0` | audio ⊕ ASR ⊕ OCR (visual deleted) | 2562 | 0.5687 (0.0035) | 30.5 / 14.0 / 6.2 | 58.0 | 0.348 |

**The plainest arm wins outright.** `ALL` at 0.5901 reproduces round 12's `A0_B0_C0` anchor of 0.5878
under a different seed block (sd 0.0021), so the protocol is intact. Every other arm is below it, and
three of the six are below it with a CI excluding zero.

---

## 2. What G0 actually killed

**M7's audio-only 0.623 does not survive the matched protocol. It measures 0.5507.**

`R14_WVD_FREEZE.md` §1 M7 read single-channel within-video AUC on the **39-video val split with
val-based epoch selection**: audio 0.623 > visual 0.587 ≈ ASR 0.583 > OCR 0.572, all four 0.671.
Under 5-fold CV inside train with no selection, the same channels give:

| channel | M7 (val, epoch-selected) | matched protocol | change |
|---|---|---|---|
| audio | 0.623 | **0.5507** | −0.072 |
| visual | 0.587 | 0.5535 | −0.034 |
| text (ASR/OCR) | 0.583 / 0.572 | 0.5288 (joint) | — |
| all four | 0.671 | **0.5901** | −0.081 |

Two things fall at once. **The level was inflated** — already known from `R14_WVD_RESULT.md` §4, now
confirmed channel by channel. And **the ordering was an artifact too**: under the matched protocol
visual (0.5535) and audio (0.5507) are indistinguishable (`AUD − VIS` = −0.0027 [−0.0371, +0.0324]),
and audio's apparent lead over the fusion reverses into a −0.0393 deficit with a CI excluding zero.

**The negative-transfer premise is therefore false as stated.** Fusion is not diluting a strong audio
signal; fusion beats every single channel by 0.035 to 0.061 wv-AUC and is doing real work. The
reviewer's conditional — *"if the 0.623 was not produced under the same protocol, it is not
admissible evidence, and the honest answer becomes: no legal family is left"* — is triggered, and it
is triggered in the strongest available form, because the contrast did not merely fail to replicate,
it reversed sign significantly.

**What is *not* refuted, stated precisely.** Round 11's circular-shift control (shuffling audio
within a video costs 3.30 macro-F1, CI excluding zero; shuffling CLIP visual costs 0.28, CI
containing zero) is a different measurement on a different metric and split, and nothing here
contradicts it. A channel can carry genuine moment-level information — so that destroying its
temporal order hurts — while still being a weaker standalone within-video discriminator than the
four-channel fusion. Both are true. What is refuted is only the inference that was drawn from
putting the shift control next to M7: that the fusion is being dragged down by video-identity
channels.

---

## 3. The centering arms, and why their failure is mechanistically clean

Within-video leave-one-out centering did exactly what it was designed to do and the result got
worse.

| arm | between-video share of score variance | wv-AUC |
|---|---|---|
| `ALL` (no centering) | 0.451 | **0.5901** |
| `AUDCENT` (V/T/O centered) | 0.150 | 0.5721 |
| `ALLCENT` (all centered) | **0.095** | 0.5530 |

The mechanism is confirmed to operate — the between-video component of the score variance is driven
from 0.451 down to 0.095 — and **within-video discrimination falls monotonically as it does**. This
is not a null; it is a refutation with a dose-response. Removing the video-identity component does
not free the head to discriminate moments; it removes context the head was using.

`AUDVIS0` — the cheapest possible version, simply deleting the channel the shift control called
temporally uninformative — is also significantly negative (−0.0214). CLIP visual contributes to
within-video discrimination even though shuffling it within a video is free. Those two facts are
compatible: the visual channel can supply a per-video reference frame that improves the *calibration*
of the other channels' contributions without itself varying informatively over time.

Note the honest prior recorded in the freeze held: centered features carry strictly less information
than round 12's factor-C input (absolute ⊕ residual ⊕ rank), which was itself null at −0.0031. The
family was an inductive-bias bet against an information loss, and the information loss won.

---

## 4. R15-FS — two more slate candidates falsified at zero cost

Computed on the out-of-fold seed-averaged per-window scores of arm `ALL` (baseline read-out 0.5915
on n = 193). No fitting, no selection.

### FS-A — evidence/label offset (candidate D10): **dead**

| shift | wv-AUC | Δ [95% CI] | folds sharing sign |
|---|---|---|---|
| −2 | 0.5598 | −0.0327 [−0.0629, −0.0025] | 3/5 |
| −1 | 0.5873 | −0.0055 [−0.0277, +0.0168] | 3/5 |
| **0** | **0.5915** | — | — |
| +1 | 0.5609 | −0.0321 [−0.0591, −0.0055] | 4/5 |
| +2 | 0.5375 | −0.0555 [−0.0861, −0.0245] | 5/5 |

Every shift is negative and degradation is monotone in |shift|. No lag beats zero, so the
evidence-to-label alignment on this grid is **not** misspecified. D10 is closed.

### FS-B — region-pooling ceiling (candidate D4): **dead**

Both are **oracle diagnostics** — they consume gold structure and are ceilings, not methods.

| read-out | wv-AUC | Δ vs unpooled [95% CI] | gate |
|---|---|---|---|
| FS-B1 pool inside gold segments *(oracle)* | 0.5930 | +0.0015 [−0.0075, +0.0105] | — |
| **FS-B2 pool inside maximal same-label runs *(oracle)*** | 0.5976 | **+0.0061 [−0.0275, +0.0392]** | **+0.015 → fail** |
| FS-C running mean width 3 *(label-free, descriptive)* | 0.6050 | +0.0135 [−0.0019, +0.0288] | no gate |

FS-B2 is the most generous ceiling available — it uses the label itself to define the pooling
regions — and it buys +0.006. **No label-free clustering can beat a partition defined by the labels**,
so the SEGPOOL family is closed without a line of clustering code.

One diagnostic worth recording: **FS-C, a plain width-3 running mean, outscores both oracle
partitions** (+0.0135 vs +0.0061). The score curve's error is therefore not "noise around a correct
region mean" — if it were, an oracle partition would dominate a fixed-width smoother. It is
broad-band. That is an independent confirmation of round 12's lag-1 error autocorrelation of 0.334
and of why the decode axis prices at 2-4 F1 points. FS-C is descriptive and carries no gate; per the
freeze, a positive FS result is not claimable as a decode contribution in any case (K1).

---

## 5. What round 13 closes

1. **The temporal-informativeness / within-video nuisance-suppression family is KILLED** — the last
   family the external review left alive. Three arms, all below plain concatenation, two of them
   significantly, with a dose-response in the intended mechanism variable.
2. **The premise it rested on is refuted with the sign reversed.** Audio-only is 0.5507, not 0.623;
   the fusion beats every single channel. M7's channel *ordering*, not just its levels, was a
   val-plus-epoch-selection artifact.
3. **D10 (evidence/label offset) is closed** — no lag beats zero, degradation is monotone.
4. **D4 (region pooling / SEGPOOL) is closed by its own oracle ceiling** — +0.006 against a +0.015
   bar with gold-defined regions.
5. **The per-window score error is broad-band, not region-structured** (a fixed-width smoother beats
   an oracle partition), which is the mechanistic reason both 3 and 4 fail and is consistent with the
   measured 0.334 error autocorrelation.

Under the rule frozen in `R15_NT_FREEZE.md` §3, committed before any number existed, the written
conclusion of this round is therefore the one the freeze pre-committed:

> **No legal mechanism family remains for hateful-video temporal localization on this substrate
> under the project's current constraints**, and the goal is escalated to the user as a scope
> question — not quietly re-attempted.

## 6. Deviations

Single invocation, no crash, total wall **15.1 s** for all 175 head fits plus both bootstrap panels.
Nothing was tuned; no arm, epoch count, learning rate, head, grid or decoder was altered; the val and
test id lists were read only for the disjointness assertion. Guards printed at the top of `run.log`:

```
[guard] split disjoint OK  train=237 val=39 test=119
[guard] pilot touches 237 train videos only; 0 val/test ids in any tensor
[guard] path guard active: any path containing 'test.jsonl' raises
[folds] sizes = [48, 48, 47, 47, 47]
[endpoint] OOF train videos with within-video label variation = 193  (ASSERT 193 OK)
```

**D1 — tie handling in the Part 2 read-out.** Round 12's `wv_auc_per_video` ranks with
`argsort().argsort()`, which breaks exact ties by array index rather than by midrank. This is
harmless for continuous model scores but **wrong for FS-B, where pooling creates exact ties by
construction** — under index tie-breaking a constant pooled score would read 1.0 rather than the
0.500 the freeze states holds by construction. Resolution: **Part 1 uses the frozen function
unchanged**, so the 0.5901 anchor is protocol-identical to round 12; **Part 2 uses midranks**
(`rankdata(method="average")`), which is mathematically identical whenever scores are distinct. The
gap was caught by the pre-run synthetic smoke test, not by any real-data metric. Measured impact on
the `ALL` seed-averaged scores: the two forms differ on **8 of 193 videos**, by at most 0.0486,
giving 0.5909 (frozen) vs 0.5915 (midrank). The ties are real — float32 softmax saturates to exactly
1.0 on 164 windows across 30 videos — not an implementation error. No gate threshold, arm, seed or
design element changed.

  *Carry-forward note, not a correction to any prior result:* the same index tie-breaking is present
  in the round 11 and round 12 numbers. Its measured effect here is +0.0006 wv-AUC, an order of
  magnitude below every δ those rounds used, so no prior verdict is affected. Future work on this
  read-out should use midranks.

**D2 — leave-one-out centering when a masked block has ≤ 1 non-empty window.** `mean_{j≠k}` is
undefined there and the freeze specifies no behaviour. Convention taken: the LOO mean is the zero
vector, so `cent(x)_k = x_k` for a lone valid window and an all-empty video stays at zero. Affects 4
videos for ASR and 14 for OCR (plus 1 and 35 all-empty respectively) out of 395. No other choice is
available without inventing one.

**D3 — no third deviation.** The run was a single submission with no crash, so no
`run_attempt1_crash.log` exists.

## 7. Reproduction

| artifact | path |
|---|---|
| freeze (pre-code, commit `e3740dc`) | `idea-stage/R15_NT_FREEZE.md` |
| candidate slate, hostile scores, occupancy | `idea-stage/R15_CANDIDATES.md` |
| review bundle | `idea-stage/codex_brainstorm_bundle_r15_2026-08-18.md` |
| runner | `scripts/r15_nt/run_nt.py` → `idea-stage/r15_nt/out/results.json` |
| dumped OOF scores | `idea-stage/r15_nt/out/` |
| logs | `logging/runs/r15_nt/run.log` |

The runner asserts train/val/test id disjointness and that no val or test id enters any tensor it
fits or scores, and prints both assertions at the top of `run.log`.
