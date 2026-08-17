# R12-IMG result — the image stream's read-out is insensitive to which positions you pool

Frozen design: `idea-stage/R12_FREEZE.md` §2, committed at **`a9cd557`** before any pilot code
existed. Deviation: `idea-stage/R12_DEVIATION_D1.md` (demotion-clause key lookup, corrected
**before** this pilot's analyzer ran, so this verdict is produced by the corrected code).
Single submission `idea-stage/r12_img/run_all.sh`.

**Cost: ¥0.00 (no API). One extraction pass (1872 videos, 6 splits, 0 failures, 1 zero-vector guard,
28 min) plus 360 head-training runs (240 MHC-ZH + 120 HateMM), 0 failures, 43 min wall on the local
RTX 5090 shared with a concurrent grid. Zero test-label tuning: every span, arm, control and epoch
rule is fixed by the freeze or computed from train only.**

---

## Headline

1. **Frozen verdict: neither candidate stands.** `ISPLIT − I0` = **−0.0031** (MHC-ZH) / **+0.0025**
   (HateMM); `I2M − I0` = **+0.0007** / **−0.0006**. Every judgement CI straddles zero except one,
   and that one is *negative*: `ISPLIT − IRSPLIT` = −0.0057 [−0.0116, −0.0001] on MHC-ZH, i.e. the
   semantic split loses to a **random** positional split with the same block sizes.
2. **The pre-committed conclusion applies**: the image stream's flat prefix mean is not improved by
   a semantic positional split or by a second-moment read-out at this sample size. The last
   never-varied read-out in the substrate is measured and closed.
3. **The mechanism is clean and it is the opposite of the text-side story.** cos(PRE, VIS) =
   **0.999 / 0.998**: the deployed ~1042-position mean *is* the 1023-position vision-block mean,
   because the instruction block is 19 tokens. And the 19-token instruction read-out **on its own**
   scores −0.0011 / −0.0005 against the deployed one, while being geometrically different
   (cos(PRE, INS) = 0.39 / 0.61) and carrying **higher** within-class effective rank (40.9 vs 31.8
   on MHC-ZH). On HateMM `IVIS − I0` = −0.0001 with a CI of [−0.0011, +0.0008] and 1/15 seeds
   positive — the two read-outs are effectively the same classifier.
4. **The premise resolves against the pilot, and it was declared in advance that it could.**
   Freeze §2.2 stated: "low effective rank is consistent with destructive pooling *and* with
   beneficial denoising ... this diagnostic motivates the pilot; it does not predict its sign."
   The result is the second reading. A 19-position random subset (`RB`) carries **roughly double**
   the within-class effective rank of the deployed read-out (70 vs 32-37) and buys nothing.
   **Within-class effective rank does not predict head accuracy on this substrate.**

---

## 1. The table

P1 = test macro-F1 at `argmax_{e≥5}` dev macro-F1; P2 = final epoch. MHC-ZH 30 seeds (800-829),
HateMM 15 seeds (800-814), both fresh ranges. **The text stream is `CAT` in every arm** and is
byte-identical across arms, so it cancels in every contrast; only `img_feats` differ. Every vector
in the table comes from one extraction pass on this machine.

| arm | img_feats | dim | MHC-ZH P1 | HateMM P1 |
|---|---|---|---|---|
| **I0** | `n(PRE)` — the deployed prefix mean | 3584 | 0.8176 ± 0.0096 | 0.8751 ± 0.0114 |
| **ISPLIT** | `[n(VIS) ‖ n(INS)]` — candidate B2 | 7168 | 0.8145 ± 0.0133 | **0.8775 ± 0.0088** |
| **I2M** | `[n(PRE) ‖ n(STD)]` — candidate B1 | 7168 | **0.8183 ± 0.0104** | 0.8745 ± 0.0103 |
| **IRSPLIT** | `[n(RA) ‖ n(RB)]` — random positional split | 7168 | *0.8202 ± 0.0122* | 0.8728 ± 0.0143 |
| **IRW** | `[n(PRE) ‖ n(PRE·R)]` — matched width | 7168 | 0.8167 ± 0.0110 | 0.8747 ± 0.0078 |
| *IVIS* | `n(VIS)` — diagnostic | 3584 | 0.8162 ± 0.0097 | 0.8750 ± 0.0114 |
| *IINS* | `n(INS)` — diagnostic | 3584 | 0.8165 ± 0.0083 | 0.8746 ± 0.0099 |
| *ISTD* | `n(STD)` — diagnostic | 3584 | 0.8151 ± 0.0097 | 0.8745 ± 0.0132 |

The whole grid spans **0.0057 macro-F1 on MHC-ZH and 0.0047 on HateMM** — about one test item.

### 1.1 The judgement contrasts (frozen list)

Paired mean ± paired-bootstrap 95 % CI, B = 20000, bootstrap seed 20260817.

| clause | contrast | MHC-ZH P1 | HateMM P1 | pass |
|---|---|---|---|---|
| 1 | ISPLIT − I0 | −0.0031 [−0.0091, +0.0024] 11/30 | +0.0025 [−0.0051, +0.0107] 8/15 | no |
| 2 | ISPLIT − IRW | −0.0022 [−0.0080, +0.0033] 12/30 | +0.0028 [−0.0041, +0.0096] 7/15 | no |
| 3 | **ISPLIT − IRSPLIT** | **−0.0057 [−0.0116, −0.0001]** 8/30 | +0.0047 [−0.0045, +0.0144] 9/15 | no |
| 1 | I2M − I0 | +0.0007 [−0.0047, +0.0059] 14/30 | −0.0006 [−0.0088, +0.0083] 7/15 | no |
| 2 | I2M − IRW | +0.0016 [−0.0030, +0.0062] 12/30 | −0.0002 [−0.0060, +0.0058] 5/15 | no |
| 3 | I2M − IRSPLIT | −0.0019 [−0.0081, +0.0043] 12/30 | +0.0017 [−0.0064, +0.0104] 7/15 | no |

Controls and diagnostics (no verdict power):

| contrast | MHC-ZH P1 | HateMM P1 |
|---|---|---|
| IRW − I0 (width control) | −0.0009 [−0.0067, +0.0045] | −0.0004 [−0.0046, +0.0045] |
| IRSPLIT − I0 (random split) | +0.0026 [−0.0032, +0.0082] | −0.0023 [−0.0111, +0.0057] |
| *IVIS − I0* | −0.0014 [−0.0041, +0.0012] 6/30 | **−0.0001 [−0.0011, +0.0008] 1/15** |
| *IINS − I0* | −0.0011 [−0.0054, +0.0033] | −0.0005 [−0.0045, +0.0033] |
| *ISTD − I0* | −0.0026 [−0.0074, +0.0025] | −0.0006 [−0.0091, +0.0075] |
| ISPLIT − I2M | −0.0038 [−0.0087, +0.0010] | +0.0030 [−0.0022, +0.0086] |

Mechanical application of the frozen rule: `idea-stage/r12_img/verdict.py` → `verdict.json`.
Both candidates `DOES NOT STAND`; the demotion clause did not fire (every candidate's dev contrast
against `I0` is negative — −0.0040 / −0.0017 for `ISPLIT`, −0.0021 / −0.0011 for `I2M` — but the
test contrast is not positive on both datasets, so the clause's conjunction is not met; it was in
any case never reached, since clause 1 fails).

---

## 2. Mechanism — why the image stream behaves nothing like the text stream

### 2.1 The span geometry (train split only, no test contact)

Mean row cosine between L2-normalised spans, and within-class effective rank (exponential of the
entropy of the covariance eigenspectrum, per class):

| quantity | MHC-ZH | HateMM |
|---|---|---|
| **cos(PRE, VIS)** | **0.999** | **0.998** |
| cos(PRE, INS) | 0.387 | 0.605 |
| cos(VIS, INS) | 0.357 | 0.582 |
| cos(PRE, STD) | −0.082 | −0.346 |
| cos(RA, RB) | 0.952 | 0.950 |
| cos(VIS, RA) | 0.999 | 0.998 |
| *text-side reference* cos(A0, TXT) | *0.452* | *0.439* |
| erank_within(PRE) | 31.8 / 25.8 | 37.4 / 38.7 |
| erank_within(VIS) | 31.7 / 25.8 | 37.3 / 38.6 |
| **erank_within(INS)** | **40.9 / 34.3** | **45.8 / 32.1** |
| erank_within(STD) | 34.1 / 28.3 | 26.8 / 25.9 |
| **erank_within(RB)** | **69.9 / 47.2** | **70.0 / 64.4** |

Span lengths, from the extraction's own statistics: median total sequence 1045 (MHC-ZH) / 961
(HateMM), vision block ends at 1023 / 939, assistant header starts at 1042 / 958 — so the
instruction block is **19 tokens** on both datasets.

### 2.2 What that means

**(a) The image-side split is not the analogue of the text-side split.** `CAT` worked because the
*deployed* text read-out was the **short** block (3 assistant-header tokens) and the transcript
content (124-229 tokens) was pooled by neither stream. Splitting exposed a genuinely unpooled
region. On the image side the deployed read-out is the **long** block: 1023 of 1042 positions, so
`cos(PRE, VIS) = 0.999` and the 19 instruction tokens are already averaged away to nothing. The
short block here is not unpooled — it is *drowned*, and un-drowning it is what `ISPLIT` does. It
buys nothing.

**(b) The short block alone is as good as the whole prefix.** `IINS − I0` = −0.0011 / −0.0005, on a
19-token read-out with cosine 0.39 / 0.61 to the deployed one. Two read-outs that share less than
half their direction give the same classifier. On the text side the same comparison was decisive in
the other direction: standalone `TXT` was −0.019 / −0.105 against `A0`, and only the concatenation
helped.

**(c) Effective rank does not predict accuracy here.** `RB`, the 19-position random complement,
carries roughly **twice** the within-class effective rank of the deployed read-out (70 vs 32-37) and
`IRSPLIT − I0` is +0.0026 / −0.0023 — indistinguishable. `INS` has ~30 % more effective rank and is
neutral. `STD` is nearly orthogonal to `PRE` (cos −0.08 / −0.35), is a genuinely new second moment,
and `ISTD − I0` is −0.0026 / −0.0006. The freeze declared in advance that the rank diagnostic could
not predict the sign; it did not, and the honest reading is that the image stream's low within-class
rank reflects **redundancy across positions**, not information destroyed by averaging.

**(d) The semantic split loses to a random split on the dataset with more seeds.**
`ISPLIT − IRSPLIT` = −0.0057 with the CI excluding zero on MHC-ZH (30 seeds) and +0.0047 with the
CI containing zero on HateMM (15 seeds). The clause was frozen to test whether the split is
*semantic* rather than "any second view of the same forward". It is not.

---

## 3. What this closes

The image stream had exactly one read-out across eleven rounds, and the round-10 effective-rank
diagnostic was the strongest premise the project had ever had for varying it. Four variants of that
read-out — semantic split, second moment, random split, matched-width random projection — plus
three standalone sub-span read-outs now sit inside a 0.005 band of the deployed mean on both
datasets, with every judgement CI straddling zero.

Combined with the token-position result on the text stream (`CAT`, four replications) and the layer
result (`LL`, demoted after two failures to replicate), the read-out axis of this substrate is
now measured on all three of its dimensions — **layer, text token position, image token position** —
and exactly one of the three yields anything.

**No method-paper claim is available here and none was pre-authorised** (freeze §0.1c). Two
independent novelty sweeps had already graded the primitive occupied — DINOv2 `2304.07193`'s frozen
linear eval *is* summary-token ⊕ mean-pooled patch tokens; `2506.10178` (ICLR 2026) benchmarks
thirteen structured read-outs against global average pooling with a +7.9 headline; `2509.24901`
(ICLR 2026) and `2608.00726` state the information-bottleneck framing; HateSieve `2408.05794` and
xDORA `2602.19212` occupy it inside the hate domain. The result is therefore a closed axis, not a
finding to write up.

---

## 4. Scope limits

- Same-machine, same-extraction-pass, head-level only. **No absolute number here is comparable to
  the project's A100-extracted ledger**; only within-table contrasts are results.
- MHC-ZH and HateMM only. MHC-EN has no read-out cache; ImpliHateVid has no raw video.
- Layer 28 only; one prompt (the deployed `IMG_INSTRUCTION`); 8 frames; one head; one
  hyper-parameter set; one fusion mode.
- The spans tested are mean-pool variants and one second moment. Learned pooling (attentive probing,
  latent-attention pooling, prototypical probes) was **not** tested and is not covered by this
  result — but it is also the part of the design space the novelty sweeps graded most heavily
  occupied, and it costs a trainable module the deployed topology does not have.
- Sink-aware read-outs (dropping the highest-norm positions) were candidate B3, scored 2.7 by the
  hostile reviewer and not run. `cos(PRE, VIS) = 0.999` is indirect evidence against them mattering
  much: whatever the sinks do, they are already inside both the deployed read-out and every variant
  measured here.
- +0.005 is ≈ 0.7 test items of 149 (MHC-ZH) and ≈ 1.1 of 215 (HateMM).
- Standing caveat: these test splits have been used by roughly 90 prior candidates; the
  paired-bootstrap intervals are conditional descriptive intervals, not post-selection-valid
  confirmatory ones.

---

## 5. Artefacts

| what | where |
|---|---|
| freeze (`a9cd557`) | `idea-stage/R12_FREEZE.md` §2 |
| deviation D1 | `idea-stage/R12_DEVIATION_D1.md` |
| extractor fork | `idea-stage/r12_img/extract_img.py` |
| extraction driver / log | `logging/runs/r12_extract/{drive.sh,run.log,run.pid}` |
| new caches | `data/CLIP_Embedding/{MHC_zh,HateMM}/{train,dev_seen,test_seen}_*-ip.pt`, `*_R12IM-*.pt` |
| span statistics | `data/CLIP_Embedding/*/IPSTATS_*.json` |
| arm builder + per-arm sha256 | `idea-stage/r12_img/build_img.py`, `build_meta_{MHC_zh,HateMM}.json` |
| single submission | `idea-stage/r12_img/run_all.sh` |
| judgement read-out | `idea-stage/r12_img/{zh,hm}_grid.json` |
| dev panel | `idea-stage/r12_img/{zh,hm}_devpanel.json` |
| mechanical verdict | `idea-stage/r12_img/verdict.py` → `verdict.json` |
| logs | `logging/runs/r12_img/{run.log,run.pid,zh/,hm/}` |

**Belts that passed.** (i) **BELT B1**: on the first 12 videos of every split of both datasets, the
extracted `PRE` span equals the frozen deployed `_pool_span(span="prefix")` computed on the same
forward with **max abs diff exactly 0.0** — 6/6 splits, 12/12 items each. (ii) The fixed Gaussian
`R` used by `IRW` re-hashes to the `r6_readout` manifest exactly. (iii) `-tp` and `-ip` id order and
labels are asserted equal per split before any arm is written. (iv) Zero NaN in any span of any
split; one undecodable HateMM video handled by the same zero-vector guard the banked caches use.
(v) 360/360 runs completed, 0 failures, 30/30 epochs each.

Seeds 800-829 / 800-814 are consumed and disjoint from every previously consumed range.
