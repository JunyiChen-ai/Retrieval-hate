# ERRPAT — MultiHateClip-EN forensic error-pattern analysis

**Executor:** errpat-mhc-en subagent. **Date:** 2026-07-26 NZST. **Status:** forensic diagnostics only.

**Discipline honored.** CPU only (`CUDA_VISIBLE_DEVICES=""`, `OMP_NUM_THREADS=4`, faiss 4 threads).
**ZERO GPU jobs, ZERO SLURM submissions, ZERO training, ZERO Modal.** Total wall time ~4 min inline.
Nothing under `autoresearch/goal_mllm_plus3/state/` was read-modified. No file deleted or moved. No git
commit. New scratch files only, all prefixed `errpat_`:

- `/data/jehc223/RGCL/scripts/analysis/errpat_mhc_en.py` + `errpat_mhc_en_out.json`
- `/data/jehc223/RGCL/scripts/analysis/errpat_mhc_en_b.py` + `errpat_mhc_en_b_out.json`

**TEST-SET READ DISCLOSURE (forensic).** This analysis reads MHC-EN test labels for **diagnosis of
already-banked predictions**. It selects no hyper-parameter, no threshold, no memory subset, and no
config. Every quantity below that uses test labels to define an operator is labelled **ORACLE / UPPER
BOUND** and is *not* a deployable result. The one place where an operator could have been fitted (decision
threshold) was fitted on **dev only** and then reported on test, which is the legal protocol.

**Numeric provenance.** Every number below was re-read from a primary file during report writing; the path
is cited inline. 4 dp unless a source carries more.

---

## 0. BOTTOM LINE

1. **The deployed EN artifacts are recoverable EXACTLY — no proxy was needed.** Per-item test predictions
   for the deployed EN best-stack exist banked for all 4 seeds, and the final-epoch no-key floor was
   recomputed from snapshotted heads and **validated bit-exact to 4 dp against the primary trainlogs on
   all 6 arms** (3 seeds × 2 encoders, test *and* dev).
2. **The residual error is 22 hard items (13.66% of n=161) plus a 20-item noise band.** Per-seed error
   count is 33.25 ± 3.3; 22 items are wrong in **all 4 seeds** (8 FP / 14 FN), 20 items flip with seed,
   119 items are never wrong.
3. **The consensus errors are vote-LOCKED, not threshold-reachable.** Consensus errors retrieve a top-20
   neighbourhood that is on average only **0.2205** correct-class (vs 0.4781 for seed-flipped, **0.8738**
   for always-right). Only **1 of 22** is fixed by the item's own label-oracle global threshold.
4. **The decision-threshold family is now measured DEAD on the deployed EN arm, deployably.** A
   dev-selected threshold *loses* on **0/6 arms** (mean Qwen −0.0083 acc, CLIP −0.0104); the test-label
   oracle threshold gains only +0.0207 / +0.0124. This closes the last unmeasured corner of the B5
   threshold family on EN.
5. **The image stream contributes no positive-class evidence at all** — raw Qwen image-only 20-NN predicts
   positive on 11.80% of items against a 30.43% base rate, with positive recall **0.2449**. This is F86's
   `U1 = 0.0000` and F44's "EN image collapse" made concrete per item, and it deflates the apparent
   stream-selection headroom (see §6.3).
6. **The Offensive-boundary hypothesis is REAL but modest and not a distinct cluster.** Consensus error
   rate is Normal 7.14% / **Offensive 30.56%** / Hateful 23.08%; deployed recall is Hateful 0.7115 vs
   Offensive 0.5833. Offensive carries 11 of 14 consensus FNs — but mostly because it *is* 73.5% of the
   positives. The per-class rate gap Offensive−Hateful is 0.0748 = **under one video at n=13**.
7. **One positive-signed lever, sub-bar:** replaying the 14-id rule-based memory-bank prune across all 4
   deployed seeds gives **+0.0093 acc / +0.0089 mF1, 3/4 seeds positive, 0 items broken** — one third of
   the +0.030 bar and inside the ±0.014 seed band. Method-level side finding: F78's "$0 banked-keys
   premise FALSE" **does not hold for EN** — exact multi-seed deletion-only curation replay is available
   at $0 (§6.5).
8. **Genuinely-open in-box set is EMPTY.** Every cluster maps to LOCKED (F44/F47/F49/F66/F86), VETOED, or
   user-gated. The only autonomous move that moved a number at all is 3× under bar and already
   test-consumed.

---

## 1. WHICH EN CONFIG IS "DEPLOYED" (Step 0 anchor)

Re-read `research-wiki/PAPER_MASTER_TABLES.md:41-42` verbatim:

```
| **MHC-EN** (161) | frozen-Qwen floor(无键) | frozen-Qwen | 0.7702 ± 0.0221 | 0.7010 ± 0.0448 | 0.7888 ± 0.0152 | 0.7488 ± 0.0208 | 4 | exp-archive-knn-seeds Add.3 · `ebc1988` |
| **MHC-EN** (161) | + archive-kNN α0.25(最优栈) | frozen-Qwen | 0.7935 ± 0.0205 | 0.7497 ± 0.0250 | 0.7826 ± 0.0134 | 0.7430 ± 0.0196 | 4 | exp-archive-knn-seeds Add.3 · `ebc1988` |
```

**Resolution — the task's warning was correct and matters.** EN is **frozen-Qwen**, not CLIP and not LoRA:

- LoRA-Qwen on EN **FAILED both protocols** and lands *below both frozen encoders*
  (`PAPER_MASTER_TABLES.md:681`: val-sel Δacc −0.0021 acc 2/3, final-ep +0.0000 acc 1/3, seed-0 anchor
  val-sel −0.0310 acc vs frozen-CLIP). So no adaptation arm is deployed on EN.
- frozen-Qwen beats frozen-CLIP on EN under the final-epoch protocol in my recompute
  (0.7847 vs 0.7785, §2.2) — consistent with the master-table 4-seed final-epoch 0.7888.

**Two protocol arms are therefore both "deployed", and I analysed both:**

| arm | config | protocol | acc | macro-F1 | seeds |
|---|---|---|---|---|---|
| **ARM-V** (primary) | frozen-Qwen → RGCL align head → **archive-kNN α=0.25** key → top-20 vote | val-selected | **0.7935 ± 0.0205** | **0.7497 ± 0.0250** | 4 |
| **ARM-F** | frozen-Qwen → RGCL align head, **no archive key** | final-epoch (e29) | 0.7847 ± 0.0156 | 0.7425 ± 0.0201 | 3 of 4 |
| ARM-F-CLIP | frozen-CLIP, same head recipe | final-epoch (e29) | 0.7785 ± 0.0129 | 0.7203 ± 0.0087 | 3 of 4 |

ARM-V is the master-table headline stack; my recompute of its 4-seed mean from the banked per-item
predictions is **0.7934782608695652 / 0.7497388274389120**, i.e. the master-table row to the last digit
(`scripts/analysis/errpat_mhc_en_out.json:arm_v_mean`). ARM-V is the **primary** object for the taxonomy;
ARM-F is used for stream forensics, the threshold family, and protocol/encoder flips.

Frontier context (unchanged, `refine-logs/LITSWEEP5_HATEMM_EN.md` per F81): EN sits at/above every legal
published method (RAMF 0.740/0.717, GPT-4V 0.63). This analysis is about *why the residual is what it is*,
not about a gap to close against literature.

---

## 2. STEP 1 — PER-ITEM ARTIFACT STATUS

### 2.1 ARM-V: BANKED, EXACT — no proxy required

`scripts/analysis/p2_out/cache_MHC_s{0,1,2,3}.json` (409,262–409,356 B each), produced by
`scripts/analysis/p2_rerank_eval.py --mode collect`. Each file holds all **161** test items with
`id, label, margin, floor_vote, floor_pred, gated`, plus the **top-60** `(neighbor_id, cosine_sim,
neighbor_label)` in retrieval order. Header (re-read at report time):

| seed | ckpt | header `floor.acc` / `macro_f1` | header `logged` |
|---|---|---|---|
| 0 | `best_model_24_0.7875.pt` | 0.8074534161490683 / 0.7625707625707625 | [0.8075, 0.7626] |
| 1 | `best_model_29_0.7875.pt` | 0.7639751552795031 / 0.7145390070921986 | [0.7640, 0.7145] |
| 2 | `best_model_21_0.8125.pt` | 0.7950310559006211 / 0.7505282434145655 | [0.7950, 0.7505] |
| 3 | `best_model_27_0.7875.pt` | 0.8074534161490683 / 0.7713172966781214 | [0.8075, 0.7713] |

Three independent gates asserted in `errpat_mhc_en.py:arm_v` and all PASSED: (i) accuracy recomputed from
the dumped per-item predictions equals the header `floor` to <1e-12; (ii) equals the *logged trainlog*
value at 4 dp; (iii) re-voting the dumped top-20 `(sim, label)` from scratch reproduces `floor_vote`
(asserted in `errpat_mhc_en_b.py:deletion_replay`, `acc_before` identical to `floor.acc` to <1e-12 on all
4 seeds). **The deployed EN vote is therefore reproducible offline from banked data at zero GPU cost.**

Corroborating dump: `scripts/role3/out/gate_MHC_base.json` — 241 items (80 val + 161 test) for seed 0,
carrying `own_card` / neighbour archive cards / raw `text`. Its own repro gate reads
`{'test_acc': 0.8075, 'test_macro_f1': 0.7626, 'gate': 'PASS (bit-identical to training log)'}`. Used for
the content/mechanism forensics in §4.3.

### 2.2 ARM-F: RECOMPUTED, machinery-validated to 4 dp (not "proxy")

F78 is right that the *floor head ckpts under `logging/Retrieval/MHC/*/ckpt/` are deleted* — those dirs are
empty. But 12 final-epoch heads survive in `refine-logs/router_ckpt_snapshot/` (335 MB, uncommitted), of
which `MHC_{CLIP,Qwen}_s{0,1,2}_e29.pt` are the EN arms (provenance `ROUTER_GATE_RECORD.md:54-62`: MHC
Qwen s0 = enc3s job 12850; s1/s2 = reused arcbase 12275/12276). Reloading them with
`cross_channel_router_gate.build_head` + the banked test cache and running the deployed vote
(`metrics.compute_metrics_retrieval` `use_sim=True, arithmetic, topk=20`, memory = train split) reproduces
the primary trainlog line **exactly on all 6 arms**, on test *and* on dev:

| arm | recomputed test acc / mF1 | trainlog anchor | source line |
|---|---|---|---|
| Qwen s0 | 0.801242 / 0.759612 | 0.8012 / 0.7596 | `slurm/logs/enc3s_MHC_Qwen2.5-VL-7B-Instruct_HF_seed0_12850.trainlog:272` |
| Qwen s1 | 0.770186 / 0.720289 | 0.7702 / 0.7203 | `slurm/logs/arcbase_MHC_Qwen2.5-VL-7B-Instruct_HF_seed1_12275.trainlog:273` |
| Qwen s2 | 0.782609 / 0.747547 | 0.7826 / 0.7475 | `slurm/logs/arcbase_MHC_Qwen2.5-VL-7B-Instruct_HF_seed2_12276.trainlog:274` |
| CLIP s0 | 0.763975 / 0.714539 | 0.7640 / 0.7145 | `slurm/logs/enc3s_MHC_openai_clip-vit-large-patch14-336_HF_seed0_12850.trainlog` |
| CLIP s1 | 0.782609 / 0.715935 | 0.7826 / 0.7159 | same, `seed1` |
| CLIP s2 | 0.788820 / 0.730292 | 0.7888 / 0.7303 | same, `seed2` |

Dev anchors also asserted (Qwen 0.7625/0.7875/0.7750, CLIP 0.7375/0.7500/0.7000 — the ckpt-filename
suffixes and `ANCHOR` in `cross_channel_router_gate.py:35-36`). These are hard `assert`s in
`errpat_mhc_en.py:arm_f` and `errpat_mhc_en_b.py:threshold_family`; the scripts abort if any fails.
The 4-seed final-epoch mean of the primary trainlogs (0.8012, 0.7702, 0.7826, 0.8012) is 0.7888 = the
master-table `final-ep acc 0.7888 ± 0.0152`; seed 3's head is not in the snapshot, hence 3 seeds here.

### 2.3 What is genuinely unrecoverable

- **Head-space test embeddings were never persisted** for any EN arm (F78 correct); they are *regenerated*
  above, which is equivalent but not "reloaded".
- **ARM-V's own head ckpt is gone** (`…arc-knn-a0.25/ckpt/best_model_24_0.7875.pt`, job 12210). This is
  why ARM-V analysis works off the banked *neighbour lists* rather than re-projecting — which is exact for
  everything in this report, and exact for deletion-only bank edits (§6.5), but **cannot** support bank
  additions, key-space changes, or re-training.
- **MHC-EN video durations do not exist** anywhere in the repo. Duration was therefore dropped from the
  covariate set; transcript length is used as the content-volume proxy.
- MHC-EN **audio/Whisper caches cover train+val only, no test** (`data/audio/MHC/whisper_*_trainval.pt`,
  `artifacts/sav_f0/extract/MHC/{train,val}`), so no audio covariate is available for test items.

---

## 3. STEP 2a — ERROR INVENTORY

### 3.1 Per seed (ARM-V, deployed val-selected)

| seed | acc | errors | FP | FN |
|---|---|---|---|---|
| 0 | 0.8075 | 31 | 12 | 19 |
| 1 | 0.7640 | 38 | 17 | 21 |
| 2 | 0.7950 | 33 | 14 | 19 |
| 3 | 0.8075 | 31 | 15 | 16 |
| **mean** | **0.7935** | **33.25** | 14.50 | 18.75 |

ARM-F (Qwen final-epoch) errors per seed: 32 / 37 / 35. Test composition: n=161, 49 positive
(30.43%), 112 negative; 3-way `Normal 112 / Offensive 36 / Hateful 13`
(`/data/jehc223/Multihateclip/English/annotation(new).json`, 891 entries; all 161 test ids present, zero
missing). Note **Offensive is 36/49 = 73.47% of positives**, the test-split analogue of F82's train
finding (Offensive = 63–73% of positives).

### 3.2 Seed stability — the error set splits cleanly in three

| bucket | n | % of test | mean margin \|vote\| | mean top-20 correct-class frac | median transcript words | FP / FN |
|---|---|---|---|---|---|---|
| **consensus (wrong in 4/4 seeds)** | **22** | 13.66% | **0.5526** | **0.2205** | 97.0 | 8 / 14 |
| seed-flipped (wrong in 1–3 of 4) | 20 | 12.42% | 0.2711 | 0.4781 | 65.5 | 12 / 8 |
| always right (0/4) | 119 | 73.91% | 0.7369 | 0.8738 | 58.0 | — |

Union of all items wrong in ≥1 seed = 42. The margin and neighbourhood-purity ordering is monotone across
the three buckets and by a wide margin — **the deployed vote is well-calibrated about its own errors at the
population level**, which is exactly why per-item *selection* has repeatedly looked promising and repeatedly
failed to convert (F47/F49).

Within the consensus bucket the two error directions are structurally different:

- **consensus FN (14):** mean margin **0.6614**, mean correct-class neighbour fraction **0.1536** —
  confidently wrong, neighbourhood is overwhelmingly Normal.
- **consensus FP (8):** mean margin **0.3621**, correct-class fraction **0.3375** — borderline.

### 3.3 Reachability (F66 framing)

Of the 22 consensus errors:

- **vote-locked** (<10% of top-20 is the correct class): **5**; 1 item has literally **zero**
  correct-class neighbours in its top-20 (`MvB-lgB8jRo`).
- <25% correct-class: **11**; ≥50%: **0**.
- **threshold-reachable** (fixed in ≥2 of 4 seeds by that seed's own *label-oracle* global threshold):
  **1**.

The label-oracle global-threshold gain on ARM-V is **+0.0140 mean acc** (per seed +0.0124 / +0.0186 /
+0.0186 / +0.0124). So the entire oracle threshold budget on the deployed arm is spent almost exclusively
on the **seed-flipped noise band**, not on the hard 22. §6.2 shows the deployable version of this operator
is negative.

### 3.4 Protocol and encoder flips

Majority-vote consolidation (ARM-V majority of 4 vs ARM-F-Qwen majority of 3):

| | count |
|---|---|
| wrong under both protocols | 27 |
| val-sel wrong, final-epoch right | 5 |
| val-sel right, final-epoch wrong | 9 |
| right under both | 120 |

Encoder flips at final epoch (Qwen majority vs CLIP majority): Qwen-wrong/CLIP-right **18**,
CLIP-wrong/Qwen-right **15**, both wrong **18**, both right **110**. **33 items disagree, near-symmetrically**
— this is F44's "MHC-EN = rotation, not Pareto" reproduced at item granularity, and the 18-item
Qwen-wrong/CLIP-right pool is the +0.1118 ceiling that F47's dev oracle priced at +0.1083.
13 of the 22 consensus errors are *also* wrong under CLIP; 9 are Qwen-specific.

---

## 4. STEP 2b — ERROR FORENSICS

### 4.1 Stream forensics (ties to F86: S≈0, U1 = 0.0000, text-dominant)

Raw single-modality 20-NN signed-cosine votes over the train bank (no head; identical construction to
`cross_channel_router_gate.raw_modality_vote`), test split, seed-independent:

| stream | acc | macro-F1 | predicts-positive rate | pos recall | neg recall | mean \|vote\| |
|---|---|---|---|---|---|---|
| Qwen **image**-only | 0.7267 | **0.5899** | **0.1180** | **0.2449** | 0.9375 | 0.3545 |
| Qwen **text**-only | 0.7826 | 0.7448 | 0.3106 | 0.6531 | 0.8393 | 0.5120 |
| CLIP image-only | 0.7143 | 0.5802 | — | — | — | — |
| CLIP text-only | 0.7143 | 0.6453 | — | — | — | — |
| *deployed fused (ARM-V, 4 seeds)* | *0.7935 ± 0.0205* | *0.7497 ± 0.0250* | — | — | — | — |

Two hard readings:

1. **The image stream is a near-constant negative predictor.** It votes positive on 11.80% of items
   against a 30.43% true positive rate, recovering only 24.49% of positives. macro-F1 0.5899 is what a
   heavily negative-biased predictor scores. This is `U1 = 0.0000` (F86) and the "Qwen IMAGE stream
   collapses to near-chance" of F44 (dev AUC 0.6756 vs text 0.8458, re-read from
   `scripts/analysis/vis_image_moved_MHC_out.json:generic`), stated as an item-level behaviour: *the image
   stream never supplies positive-class evidence*.
2. **The entire trained stack buys ~+0.011 acc / +0.005 mF1 over a raw text 20-NN with no head at all**
   (0.7935 vs 0.7826; well inside the 4-seed ±0.0205 band). On EN, essentially all extractable signal is
   in the text stream and is already linearly retrievable from frozen features.

Cross-tab, all 161 items (fused = ARM-F-Qwen majority of 3 seeds correct):

| img right | txt right | fused right | n |
|---|---|---|---|
| 1 | 1 | 1 | 95 |
| 0 | 1 | 1 | 22 |
| 0 | 0 | 0 | 15 |
| 1 | 0 | 0 | 12 |
| 1 | 0 | 1 | 5 |
| 1 | 1 | 0 | 5 |
| 0 | 1 | 0 | 4 |
| 0 | 0 | 1 | 3 |

On the 22 consensus errors: img-wrong+txt-wrong **12**, img-right+txt-wrong **7**, img-wrong+txt-right
**2**, both-right **1**.

**Honest deflation — this is the key methodological caveat of the whole stream analysis.** 6 of the 7
"image-only right" consensus errors are **FPs** (`TvuSOkN7OrM, ko0Ub2dTh-U, YDEsYXYlB8o, 6IZVj1joK6Q,
WYJ3_pvq0aw, NmMrESRM134`). Since the image stream predicts negative 88.20% of the time, being "right" on
a true negative carries no information. Removing the degenerate-bias cases leaves only **4 of 22**
consensus errors with a non-degenerate single-stream rescue (`dK43yHIUMKA` image-only, `cXRgVEENkPA` and
`msWtrZSVUis` text-only, `Jocr_4Py5-U` both raw streams right while fused is wrong). The naive
stream-selection ceiling of 10/22 = +0.0621 is therefore really **4/22 = +0.0248** (§6.3).

### 4.2 Content covariates — transcript volume, not transcript absence

| transcript-word quartile (cut at 12 / 61 / 125 words) | n | pos | consensus errors | FN | FP | positive recall |
|---|---|---|---|---|---|---|
| Q1 shortest (mean 2.2 w) | 42 | 10 | 4 (9.52%) | 3 | 1 | 0.6250 |
| Q2 (mean 39.1 w) | 39 | 11 | 4 (10.26%) | 2 | 2 | 0.7727 |
| Q3 (mean 93.0 w) | 40 | 16 | 5 (12.50%) | 4 | 1 | 0.6094 |
| **Q4 longest (mean 160.4 w)** | 40 | 12 | **9 (22.50%)** | 5 | 4 | **0.4792** |

Error rate rises monotonically with transcript length and roughly doubles from Q1 to Q4; positive recall in
the longest quartile drops to 0.4792. Consensus errors have median 97 transcript words vs 58 for
always-right items.

**Empty transcripts are NOT an error driver** — 11 test items have an empty transcript and only **1** of
them is a consensus error. This kills the obvious "missing text ⇒ falls back to the dead image stream"
hypothesis: the failure mode is the opposite, **long transcripts diluting a mean-pooled 3584-d text vector**
so the hateful span stops dominating the embedding. (This is consistent with, and an EN-side analogue of,
the pooling-dilution intuition that P3 evidence-density weighting tried and failed to convert — P3 was a
*trained* re-weighting, and its failure does not remove the descriptive fact.)

### 4.3 Content / mechanism forensics from the banked archive cards

Using `scripts/role3/out/gate_MHC_base.json` `own_card` fields (target_groups / mechanism / explicitness,
generated by Qwen2.5-VL-7B over the video, provenance `data/Archive/MHC/*_archive.jsonl`):

**The `target_groups` field is near-degenerate and carries NO error signal.** 152 of 161 test items have
`target_groups == []` (43/49 positives, 109/112 negatives). Consensus-FN rate is 12/43 = 27.91% for
`target_groups == []` positives vs 2/6 = 33.33% for the rest — no separation. I explicitly do **not** build
a cluster on target-emptiness; the field is an extraction artifact (the same degeneracy the A-line hit,
F-A-line "91–93% rows = one literal constant"). Flagging this because the surface reading of the
consensus-error card dump ("12 of 14 FNs have no target group!") is a base-rate trap.

**The `mechanism` field DOES separate on the FP side.** Among the 112 Normal items:

| Normal subset | n | consensus FP | rate |
|---|---|---|---|
| card lists an inflammatory mechanism (slur / insult / dehumanization / threat / mockery / sexual_harassment) | 32 | **6** | **0.1875** |
| card lists none of those | 80 | 2 | 0.0250 |

A 7.5× FP-rate ratio. 4 of the 8 consensus FPs carry `slur` explicitly. Reading the raw titles confirms
it: `WYJ3_pvq0aw` "surprise mother f * cker", `6IZVj1joK6Q` "a bitch", `NmMrESRM134` Will/Jada gossip,
`ko0Ub2dTh-U` "a HOOKER tries to RIZZ me". **Profanity is not hate, and the deployed retrieval vote cannot
tell them apart.** Note what this says about fixability: the archive card *agrees with the model*, so
augmenting the pipeline with archive-card mechanism features would reinforce these FPs, not fix them.

Two consensus FPs are a different animal — **counter-speech / meta-commentary about hate**:
`KDcCiUU8q5E` "Trump & his audience's Misogyny & Cruelty" (a critique of misogyny, gold label Normal) and
`YDEsYXYlB8o` "Should gays be around children?" (gold Normal; its archive card says
`target=['LGBTQ+'] mech=['stereotyping','dehumanization'] expl=explicit`, i.e. the MLLM sides with the
model against the gold label).

### 4.4 Per-class recall of the deployed arm (ARM-V, 4-seed mean)

| original 3-way class | n | deployed recall / specificity | per seed |
|---|---|---|---|
| Hateful | 13 | recall **0.7115** | 0.7692 / 0.6154 / 0.7692 / 0.6923 |
| Offensive | 36 | recall **0.5833** | 0.5556 / 0.5556 / 0.5556 / 0.6667 |
| Normal | 112 | specificity **0.8705** | 0.8929 / 0.8482 / 0.8750 / 0.8661 |

Consensus error rate by class: Normal 8/112 = **0.0714**, Offensive 11/36 = **0.3056**, Hateful 3/13 =
**0.2308**.

**Verdict on the Offensive-boundary question (asked explicitly in the tasking).** The error mass *is*
concentrated on Offensive in absolute terms (11 of 14 consensus FNs, 78.6%), and Offensive recall is 0.128
below Hateful recall. But the honest arithmetic is: Offensive is 73.47% of positives, so 11/14 is close to
what a uniform positive-class error rate would produce; the *rate* gap Offensive−Hateful = 0.0748
corresponds to **0.97 videos at n=13**. Genuine Hateful items are missed at 23% too. **Conclusion: the
Offensive/Hateful boundary is a contributing factor, not the dominant one — the deployed method is weak on
positives generally.** I therefore do *not* nominate an "Offensive-boundary" cluster as the top cluster,
which is the opposite of what a naive count would suggest. §7 still quantifies the protocol option because
the tasking asked for it.

### 4.5 Overlap with the consensus-denoising noisy-label ids

The banked EN positive result is a **train-side memory-bank** edit, so the overlap question is "do the
noisy entries appear in the errors' retrieved neighbourhoods". Ids re-read from
`scripts/analysis/memory_editing_demo.py:76`: `NOISY_IDS = ["XScP1AiMkNM", "QvPp8Q7QhWE"]`.

| | count |
|---|---|
| test items with a human-2 id inside top-20 | 19 |
| test items with a human-2 id inside top-60 | 41 |
| **consensus errors** with a human-2 id in top-20 | **2** |
| test items with a 14-id-rule-list entry in top-20 | 51 |
| consensus errors with a 14-id entry in top-20 | 8 |

So the human-flagged noise contaminates 19 neighbourhoods but sits in only 2 of the 22 hard errors — it is
a **noise-band** contaminant, not a hard-error cause. §6.5 confirms this by exact replay: the items it
flips are two low-margin FPs.

---

## 5. STEP 3 — NAMED CLUSTERS

All counts are over the **22 consensus errors** (wrong in 4/4 deployed seeds) unless noted. Percentages are
of the 22. "acc ceiling" = flipping every item in the cluster, as a fraction of n=161.

| # | cluster | n | % of consensus errors | acc ceiling if all flipped | fix status (§6) |
|---|---|---|---|---|---|
| **C1** | **Vote-locked non-group-harm Offensive misses** | **9** | 40.9% | +0.0559 | LOCKED (F44 label-limited, F66, F86) |
| **C2** | **Borderline Offensive misses** | 2 | 9.1% | +0.0124 | LOCKED (F82 graded-label pregate) |
| **C3** | **Genuine Hateful misses** | 3 | 13.6% | +0.0186 | LOCKED (F44/F65 image, F49) |
| **C4** | **Lexical-surface FPs (profanity ≠ hate)** | 5 | 22.7% | +0.0311 | LOCKED / VETOED (OCR, closed APIs) |
| **C5** | **Counter-speech / meta-commentary FPs** | 2 | 9.1% | +0.0124 | LOCKED (F47 selection, F49) |
| **C6** | **Gold-label-disputed FP** | 1 | 4.5% | +0.0062 | PROTOCOL / annotation question (§7) |
| **C7** | *(cross-cutting)* **long-transcript dilution** | 9 of the 22 sit in Q4 | 40.9% | ≤+0.0559 (overlaps C1–C5) | LOCKED (P3 trained-flat; F66 segment ops) |
| **C8** | *(separate bucket)* **seed-noise band** | 20 items, 11.25 errors/seed | — | +0.0699 variance-only | VETOED (cross-seed ensembles user-banned) |

Cluster membership is by the archive-card mechanism + gold 3-way label + neighbourhood-purity profile;
C1/C2 split at correct-class neighbour fraction 0.20 (C1 below, C2 above); C4/C5 split on whether the card's
mechanism is a surface slur/insult (C4) vs commentary-about-hate framing (C5).

### C1 — Vote-locked non-group-harm Offensive misses (n=9, +0.0559 ceiling)

Positives labelled Offensive whose harm is sexual, abusive, or vulgar but **not directed at a protected
group**; the positive side of the train memory is dominated by group-targeted hate, so their retrieved
neighbourhood is almost entirely Normal.

- `MvB-lgB8jRo` — "How Farmers Sexually Violate Cows For Dairy #shorts"; margin 0.9732, **top-20
  correct-class fraction 0.0000** (zero positive neighbours out of 20), card `mech=['stereotyping']`.
- `Z4xwQ4SRUAs` — "My husband yelled at me for refusing to give him a h…"; margin 0.8614, purity 0.0500,
  card `mech=['sexual_harassment']`, 182 transcript words.
- `B0HFN9lqM6o` — "STORYTIME: I WAS SEXUALLY HARASSED *TWICE* WHILE WOR…"; margin 0.7056, purity 0.1000,
  card `mech=['sexual_harassment']`, 238 transcript words (longest error in the set).

Also in C1: `LMOHDiG5XQg`, `ZI_jbZWzW1A`, `Jocr_4Py5-U`, `03qOelm_dK8`, `4QP3y5JrtlU`, `k9OtaMbK0Ac`.

### C2 — Borderline Offensive misses (n=2, +0.0124)

Same class, but the neighbourhood is mixed (purity 0.25–0.38) so the vote is nearly recoverable:
`msWtrZSVUis` ("Asking couples if they would swap", purity 0.2500, raw text stream votes correctly) and
`lzKJ_AWegCc` ("Boy or girl, maybe ladyboys? Pattaya, Walking street", purity 0.3750, margin 0.2486 —
the smallest FN margin).

### C3 — Genuine Hateful misses (n=3, +0.0186)

- `cXRgVEENkPA` — "Catholic Drag Queens"; card `target=['religious group','sexual orientation']
  mech=['stereotyping','mockery'] expl=explicit`; margin 0.5204, purity 0.2375. **Raw text stream votes
  correctly; both fused Qwen and fused CLIP are wrong on all seeds** — a fusion-cancellation miss, i.e. the
  F44/F50 rotation mechanism costing a true Hateful item.
- `dK43yHIUMKA` — "Is now the era of homosexuality ? Dave Allen"; card `target=['homosexuals']
  mech=['stereotyping','coded_language']`; the **only positive** the image stream uniquely rescues (the other six image-only rescues are all true negatives, i.e. the degenerate bias of §4.1).
- `uPJtlBAOT_U` — "Idiot frauditor ARRESTED and TAZED!"; card `target=[] mech=['threat']` — harassment
  with no group target, so mechanistically it belongs with C1 despite the Hateful gold label. It is also the one consensus error that **frozen-CLIP gets right on all 3 seeds** while frozen-Qwen misses it on all 3 — the sharpest single instance of the encoder rotation.

### C4 — Lexical-surface FPs: profanity ≠ hate (n=5, +0.0311)

Normal videos whose text carries an explicit slur or insult, pulling in hateful neighbours:
`WYJ3_pvq0aw` ("surprise mother f * cker", `mech=['slur']`, margin 0.2703), `6IZVj1joK6Q` ("a bitch",
`mech=['slur']`), `NmMrESRM134` (`mech=['slur','insult','dehumanization']`), `ko0Ub2dTh-U`
(`mech=['slur','stereotyping']`), `TvuSOkN7OrM` ("Smooth Alley Cat", `mech=['insult']`).
Base rate evidence: Normal items with an inflammatory mechanism card FP at 0.1875 vs 0.0250 for the rest.

### C5 — Counter-speech / meta-commentary FPs (n=2, +0.0124)

`KDcCiUU8q5E` "Trump & his audience's Misogyny & Cruelty" (critique of misogyny; 154 transcript words;
fixed by the label-oracle threshold in 1 of 4 seeds) and `pofgIFZpR7c` "YOU NEED to stop MASTURBATING"
(`mech=['stereotyping'] expl=implicit`; lowest consensus margin 0.1727, purity 0.4125, threshold-fixed in
2 of 4 seeds — the single most nearly-recoverable consensus error).

### C6 — Gold-label-disputed FP (n=1, +0.0062)

`YDEsYXYlB8o` "Should gays be around children?" — gold **Normal**, but the banked MLLM archive card reads
`target=['LGBTQ+'] mech=['stereotyping','dehumanization'] expl=explicit`. Model, archive card, and (on the
evidence of the title) a reasonable annotator disagree with the release label. 205 transcript words.
This is the one consensus error where "the method is wrong" is not clearly the right description.

### C8 — Seed-noise band (20 items, 11.25 errors/seed)

Mean margin 0.2711, correct-class neighbour fraction 0.4781, 12 FP / 8 FN. Accounts for **33.8%** of the
average seed's 33.25 errors while being no individual item's stable failure. Contains
`cYQyH7hbNnw` and `xqilG4oMvvI` — the exact two items the human-2 memory deletion flips (§6.5).

---

## 6. STEP 4 — SOLUTION MAPPING, WITH CEILINGS

### 6.1 Summary table

| cluster / lever | max flippable items | acc ceiling | status | authority |
|---|---|---|---|---|
| C1 non-group-harm Offensive | 9 | +0.0559 | **LOCKED** | F44 (EN = label-limited, not representation-limited), F82 (graded-label oracle EN +0.0250 < bar), F86 (no synergy to exploit) |
| C2 borderline Offensive | 2 | +0.0124 | **LOCKED** | F82 (vote-side graded mechanism closed <2%) |
| C3 genuine Hateful misses | 3 | +0.0186 | **LOCKED** | F44 + F65 (vision-LoRA, image MOVED with zero conversion), F49 (alignment ceiling q>0.663) |
| C4 lexical-surface FPs | 5 | +0.0311 | **LOCKED / VETOED** | OCR **user-VETOED**; closed-model APIs VETOED; F50 (cross-encoder fusion raised AUC to 0.898 and still did not convert) |
| C5 counter-speech FPs | 2 | +0.0124 | **LOCKED** | F47 (per-item selection dead at all 3 supervision sources), F49 |
| C6 label-disputed FP | 1 | +0.0062 | **PROTOCOL** (§7) | — |
| C7 long-transcript dilution | ≤9 (overlaps above) | ≤+0.0559 | **LOCKED** | P3 evidence-density pooling trained-flat on all 3 datasets; F66 (segment ops symmetric slice +0.0064 on EN) |
| C8 seed-noise band | 11.25/seed | +0.0699 | **VETOED** | cross-seed ensembles user-banned |
| decision threshold (global) | 1 of 22 | dev-selected **−0.0083** | **DEAD (measured here)** | B5 family; §6.2 |
| per-item stream selection | 4 non-degenerate | +0.0248 | **LOCKED** | F47, F49, F66, F86 |
| per-item encoder selection | 18 | +0.1118 | **LOCKED** | F47 (oracle +0.1083 dev, router +0.0000, dev-CV −0.0458) |
| memory-bank curation (14-id rule) | 2–3/seed | **measured +0.0093** | sub-bar, test-consumed | §6.5; F77 lead, F78 park |
| audio (CLAP/AST) | unknown | — | user-gated, likely zeroed | F81; and **EN test audio features do not exist** (§2.3) |
| Molmo2-8B tower swap | unknown | — | user-gated, F65-weakened | F81 EV ordering |
| MNTP stage-2 | unknown | — | user-gated | F81 (~8–12%) |

### 6.2 Decision threshold — NEW measurement, and it is a clean kill

The error structure practically invites a threshold shift: 8 of 22 consensus errors are FPs at low margin,
and the label-oracle global threshold is worth +0.0140 on ARM-V. So I measured the **deployable** version
on the machinery-validated ARM-F arms: pick the threshold maximising dev accuracy (n=80, tie-broken on dev
macro-F1), apply to test.

| arm | dev-selected τ | Δacc on test | ΔmF1 on test | test-label-oracle Δacc |
|---|---|---|---|---|
| Qwen s0 | +0.0478 | −0.0062 | −0.0124 | +0.0124 |
| Qwen s1 | −0.1714 | −0.0062 | +0.0099 | +0.0186 |
| Qwen s2 | +0.1420 | −0.0124 | −0.0174 | +0.0311 |
| CLIP s0 | −0.7449 | −0.0062 | +0.0042 | +0.0248 |
| CLIP s1 | +0.2985 | −0.0062 | −0.0153 | +0.0062 |
| CLIP s2 | +0.7907 | −0.0186 | −0.0461 | +0.0062 |
| **mean Qwen** | — | **−0.0083** | −0.0066 | +0.0207 |
| **mean CLIP** | — | **−0.0104** | −0.0191 | +0.0124 |

**0 of 6 arms improves.** The dev-selected thresholds are wildly unstable (−0.745 to +0.791) because 80 dev
items cannot resolve a threshold — the deployed cutoff `vote ≥ 0` is already at or better than what dev can
recommend. B5 killed this family on ZH; the FA gate priced the *oracle* side on EN's best cross-encoder
config at +0.025 < +0.030. **This closes the deployable side on the actually-deployed EN arm.** It is a
door-closer, not an opportunity.

### 6.3 Why the stream/encoder headroom is not an opportunity, restated per item

- Naive stream-selection ceiling on consensus errors: 10/22 = +0.0621. **Corrected for the image stream's
  88.2% negative-prediction bias: 4/22 = +0.0248.**
- Encoder-selection ceiling: 18 items = +0.1118, which independently reproduces F47's dev oracle +0.1083.
- Both are per-item *selection* operators. F47 closed per-item channel selection at all three supervision
  sources (unsupervised/feature-conditional, train-supervised — degenerate because the head memorises train
  at LOO 0.998, and dev-supervised — negative at the CV ceiling). F49 turned it into arithmetic: any router
  input needs which-arm-wins alignment q > 0.663 and none is available. F66 made the same statement for
  segment selection (EN: oracle +0.0700 = symmetric +0.0064 + selection +0.0636, i.e. 91% banned-only).
- My per-item numbers add a mechanistic reason those closures were inevitable on EN: **the "second opinion"
  that the oracle exploits is largely the image stream being constantly negative.** A selector cannot learn
  "trust the image stream" without learning "predict Normal", which is the majority class.

### 6.4 Protocol question — quantified, for the user (not a method change)

Scoring the **same deployed predictions** under two alternative label protocols:

| protocol | n | positives | acc | macro-F1 |
|---|---|---|---|---|
| deployed: harmful = Hateful ∪ Offensive | 161 | 49 (30.4%) | 0.7935 ± 0.0205 | 0.7497 ± 0.0250 |
| **Offensive-excluded** (Hateful vs Normal) | 125 | 13 (10.4%) | **0.8540 ± 0.0239** | **0.7096 ± 0.0402** |
| Offensive → negative, same predictions (diagnostic) | 161 | 13 | 0.7562 | 0.5864 |

Per-seed Offensive-excluded accuracies: 0.8800 / 0.8240 / 0.8640 / 0.8480.

**Reading.** Dropping Offensive buys **+0.0605 headline accuracy** and **loses −0.0401 macro-F1**, because
the positive rate falls to 10.4% and accuracy starts being carried by the majority class. It is a
rebalancing artifact, not a capability gain, and it would break comparability with every published
MultiHateClip number (all of which use the 2-class harmful-vs-normal collapse). Row 3 is diagnostic only —
a real `hateful_vs_rest` model would be retrained (`scripts/prep_mhc.py` already implements that label
scheme). **This is a PROTOCOL decision for the user, and my recommendation is not to take it**: it costs
macro-F1, costs external comparability, and shrinks the positive class to 13 test videos where 1 video =
0.0080 accuracy.

### 6.5 Memory-bank curation — the only positive-signed lever, and a correction to F78's premise

Exact replay: for each deployed seed, drop the flagged ids from the memory bank by removing them from the
banked top-60 neighbour list and refilling from rank 21+ (asserted exact — `len(kept) >= 20` holds for
every item, and the un-edited replay reproduces each seed's `floor.acc` to <1e-12).

| deletion set | s0 Δacc | s1 | s2 | s3 | mean Δacc | mean ΔmF1 | seeds + | items fixed / broken |
|---|---|---|---|---|---|---|---|---|
| human 2-entry (`XScP1AiMkNM`, `QvPp8Q7QhWE`) | **+0.0124** | 0 | 0 | 0 | +0.0031 | +0.0031 | 1/4 | 2 / **0** |
| 14-entry rule-hit list | +0.0124 | +0.0186 | +0.0062 | 0 | **+0.0093** | **+0.0089** | **3/4** | 6 / **0** |

Seed-0 reproduces the banked result exactly: 0.8074534161490683 → 0.8198757763975155 and macro-F1
0.7625707625707625 → 0.7748468920287408, matching `research-wiki/DEMO_memory_editing.md:52`
("acc 0.8075→0.8199、macro-F1 0.7626→0.7748"). The two items it flips are named for the first time:
`cYQyH7hbNnw` (vote +0.008649 → −0.018971) and `xqilG4oMvvI` (+0.046778 → −0.027381) — **both low-margin
FPs from the C8 noise band**, neither a hard error.

Three findings, in decreasing strength:

1. **The banked EN positive is single-seed.** The 2-entry deletion moves seed 0 and **exactly nothing on
   seeds 1/2/3** (0 vote flips). Its multi-seed mean is +0.0031. The claim "deleting the two human-flagged
   entries repairs EN" is a **1-of-4-seed effect worth 2 videos**, which the original single-seed demo could
   not see. This should be reflected wherever the result is cited as a pillar-4 positive.
2. **The 14-id rule list is strictly better than the human pair across seeds** — +0.0093 mean, 3/4 seeds
   positive, 6 items fixed, **0 broken on any seed**. `DEMO_memory_editing.md:52` says the extra 12 ids are
   inert ("其余 12 条规则命中条目增删无影响"); that is true **on seed 0 only** — on seeds 1 and 2 they are
   worth +0.0186 and +0.0062. Still 3× under the +0.030 bar and inside the ±0.014 band, and it is now a
   test-set measurement, so it cannot be promoted; a legitimate version needs a dev-side pregate, and at
   dev n=80 one item is 0.0125, so **dev cannot resolve a +0.009 effect**. That arithmetic wall is the
   honest reason this lever is not promotable, independent of the bar.
3. **Method-level correction to F78.** F78 parked curation because "head embeddings never persisted, floor
   head ckpts all 6 deleted", making a faithful multi-seed pregate cost ~0.3 GPU-h. **For EN that premise
   does not hold**: `p2_out/cache_MHC_s{0..3}.json` banks the top-60 neighbour lists *in the deployed
   archive-kNN key space for all 4 deployed seeds*, which supports **exact, $0, multi-seed,
   deletion-only** bank replay — demonstrated above. The limitation is real but narrower than F78 states:
   it applies to bank *additions*, key-space changes, and re-training, not to pruning. If curation is ever
   revisited on EN, it is a $0 CPU job, not a GPU re-mint.

### 6.6 Genuinely open, in-box

**Empty.** Every cluster in §5 lands in LOCKED, VETOED, or user-gated. The one lever that produced a
positive number (§6.5) is 3× under bar, inside the seed band, below the resolution of the selection split,
and already spent on test. I am stating this plainly rather than dressing a sub-bar effect as a lead.

---

## 7. WHAT THE ERROR STRUCTURE SAYS ABOUT THE EN CEILING

Three independent statements, each grounded in the per-item numbers above:

1. **EN's residual is a label-semantics mismatch, not a representation deficit.** The method implements
   "group-targeted hateful video retrieval"; the gold binary label is "Hateful ∪ Offensive vs Normal",
   where Offensive covers sexual harassment, animal cruelty, insults and vulgarity with no group target.
   C1 (9 items, 40.9% of consensus errors) is precisely the intersection, and those items retrieve
   neighbourhoods that are 0–14% correct-class — the memory bank has nothing to match them to. This is
   F44's "MHC-EN = label-limited" and F82's "Offensive = 63–73% of positives, down-weighting them drags the
   majority of positives toward Normal", now visible item by item.
2. **The measurement channel is at the edge of its resolution.** One test video = 0.0062 accuracy; one dev
   video = 0.0125. The whole residual signal (33.25 errors/seed) contains a 20-item, 11.25-error/seed noise
   band; the seed std is ±0.0205. Every remaining candidate effect in §6 is 1–3 videos. This is the same
   arithmetic wall F66 and F82 hit from other directions.
3. **The image modality is not merely weak on EN — it is inert for the positive class.** Positive recall
   0.2449 at an 11.80% positive-prediction rate. Any future proposal that routes to, protects, or
   re-weights the EN image stream should be priced against that number, because F86's `U1 = 0.0000` and
   F65's zero conversion are what it looks like per item.

---

## 8. LIMITATIONS (stated, not buried)

- **ARM-F is a faithful recompute, not a reload.** Head weights are the originals; embeddings are
  regenerated. Validated to 4 dp on test and dev for all 6 arms, and the assertions abort on mismatch —
  but this is not the same as reading a persisted prediction file.
- **ARM-V's head ckpt is gone**; ARM-V analysis is exact only for operations expressible on banked top-60
  neighbour lists (this covers everything in this report plus deletion-only bank edits).
- Seed 3's final-epoch head is not in the snapshot, so ARM-F is 3 seeds where the master table has 4.
- **Cluster boundaries C1/C2 and C4/C5 involve my judgement** on archive-card mechanism plus title
  reading, on n=22. They are descriptive, not measured partitions. The counts move by 1–2 items under
  reasonable alternative cuts; the *ceilings* in §6.1 should be read with that ±1–2 item slack.
- **`target_groups` is a degenerate field** (152/161 empty) and was deliberately excluded from cluster
  definitions after base-rate checking.
- **No duration and no test-split audio covariate exist** for MHC-EN (§2.3).
- The archive cards are Qwen2.5-VL-7B outputs, not ground truth. They are used as *descriptive evidence
  about surface cues*, never as labels, and §4.3 notes explicitly that where they separate FPs they agree
  with the model rather than with the gold label.
- All §5–§6 error-set statistics are computed on the test split with test labels. They characterise a
  frozen prediction set; no operator was chosen with them.

---

## 9. FILE MANIFEST

| path | contents |
|---|---|
| `scripts/analysis/errpat_mhc_en.py` | arms V + F, per-item join, taxonomy, oracle threshold, stream cross-tabs |
| `scripts/analysis/errpat_mhc_en_out.json` | per-item records for all 161 items + aggregates |
| `scripts/analysis/errpat_mhc_en_b.py` | deletion replay, threshold family, protocol arithmetic, quartiles, ceilings |
| `scripts/analysis/errpat_mhc_en_b_out.json` | part-B results incl. per-seed flip lists |
| `refine-logs/ERRPAT_MHC-EN_2026-07-26.md` | this report |

Primary sources re-read at report time: `research-wiki/PAPER_MASTER_TABLES.md:41-42,681`;
`scripts/analysis/p2_out/cache_MHC_s{0,1,2,3}.json` headers; `scripts/role3/out/gate_MHC_base.json`
(`repro`, `config`, `samples`); `slurm/logs/enc3s_MHC_Qwen2.5-VL-7B-Instruct_HF_seed0_12850.trainlog:272`;
`slurm/logs/arcbase_MHC_Qwen2.5-VL-7B-Instruct_HF_seed{1,2}_1227{5,6}.trainlog:{273,274}`;
`slurm/logs/enc3s_MHC_openai_clip-vit-large-patch14-336_HF_seed{0,1,2}_12850.trainlog`;
`research-wiki/DEMO_memory_editing.md:52`; `scripts/analysis/memory_editing_demo.py:76`;
`scripts/analysis/vis_image_moved_MHC_out.json`; `/data/jehc223/Multihateclip/English/annotation(new).json`;
`refine-logs/ROUTER_GATE_RECORD.md:54-62`; `refine-logs/ISR_PREGATE_RECORD.md:113`;
`refine-logs/LP_GATE_RECORD.md`; `refine-logs/B5_VERDICT_REVIEW.md:135-155`;
`autoresearch/goal_mllm_plus3/state/findings.jsonl` (F44, F46, F47, F49, F50, F63, F66, F77, F78, F81, F82,
F85, F86, F87).
