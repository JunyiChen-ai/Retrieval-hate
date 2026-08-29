# TASK A — Adversarial review of the C1 (E2EQ) KILL decision

**Reviewer:** fresh zero-prior-context reviewer. Read-only. No GPU, no submissions, no code edits.
**Date:** 2026-07-13.
**Object under review:** the main loop's decision to KILL C1 (RA-HMD-style two-stage QLoRA of
Qwen2.5-VL-7B under the retrieval-contrastive objective) at gate G0, zero GPU, recorded in
`research-wiki/LITERATURE_mllm_integration_2026-07-13.md` §7 and pre-registered in
`research-wiki/experiments/exp-e2eq-e0.md`.

## VERDICT: **KILL_CONFIRMED** (on expected value), with one honest caveat on the cost framing.

The kill's factual chain checks out against the repo and against the anchor paper's own ablation.
The +3-acc bar is structurally unreachable by this route. The strongest counter-argument I could
build (the untested sequential cell is a real, never-run configuration whose settling experiment is
near-free) is TRUE on the facts but does **not** overturn the kill on expected value — it only
argues the negative should be *measured* rather than *inferred*, at a cost of seconds of GPU, if the
team wants an airtight paper sentence instead of a defensible inference.

---

## 1. Independent verification of the claimed P9 / P9b numbers (repo, file:line)

All numbers cited by the kill were re-read from the primary record
`research-wiki/EXP_p9_lmm_rgcl_video.md`. They reproduce.

### 1.1 P9 (Stage-1 LoRA-SFT, rgcl-OFF), two read-outs vs the frozen-Qwen RGCL floor

| claim in kill (§7) | repo reading | file:line | match |
|---|---|---|---|
| P9 C3-mlp ≈ floor: EN +0.6 / ZH +1.0 / HateMM +0.9 | EN test 0.7909 vs floor 0.7847 (+0.6); ZH matched-protocol +1.0; HateMM 0.8698 vs 0.8605 (+0.9, s0) | `EXP_p9_lmm_rgcl_video.md:134`, `:150`, `:191` | ✅ |
| P9 C3-knn **below floor**: EN −2.7 / ZH −2.2 / HateMM −4.7 | EN 0.7578 vs 0.7847 (−2.7); ZH 0.7964 vs 0.8188 (−2.2); HateMM 0.814 vs 0.8605 (−4.7) | `:135`, `:192` | ✅ |

Two caveats a fair reviewer must record (neither rescues C1):
- **HateMM C3-mlp +0.9 is single-seed (s0 only).** s1/s2 were GPU-blocked at the time
  (`EXP_p9:191,196`). The 3-seed evidence for "Stage-1 LoRA ≈ frozen floor" is on EN/ZH; HateMM is
  one seed. The pattern is nonetheless consistent across all three datasets.
- **The ZH "+4.5" headline shrinks to +1.0** once compared at a protocol-matched LoRA floor
  (`:142–155`): the +4.5-vs-frozen number attributes the *entire* LoRA benefit to the stage; the fair
  floor is a LoRA system at the same no-selection protocol (0.8537±0.012), against which C3-mlp is
  +1.0, within noise. The kill uses the correct (+1.0) number.

### 1.2 P9b (RGCL trained *jointly* inside the LMM loop), "0/12"

| claim in kill (§7) | repo reading | file:line | match |
|---|---|---|---|
| P9b joint RGCL: 0/12 cells beat floor | Wave = {D3, C3′}×{ZH,EN}×3 seeds = 12 runs; verdict FAIL both datasets; "no cell of the wave beats its protocol-matched floor; best cell C3′-ZH-mlp 0.8591 ≈ floor 0.8537 (+0.5, within ±1.2pt band)" | `EXP_p9:340–350`, `:363–364` | ✅ (see nuance) |

**Precision nuance (adversarial honesty):** "0/12" conflates the joint-RGCL arm (D3, 6 cells) with
its rgcl-OFF control (C3′, 6 cells). The accurate statement is: *D3 (joint RGCL) beat floor+1.5 in
0/6 cells; across all 12 wave cells the best is +0.5, within the seed band.* Substance unchanged: the
joint-RGCL arm failed and nothing in the wave cleared the floor.

### 1.3 The mechanism read P9b actually produced (this is the pivot for the counter-argument)

`EXP_p9:352–364`: the **pure rgcl-term effect** (D3 − C3′, same branch, same recipe) on the memory
read-out is **positive**: **ZH +1.8pt** (0.8389 vs 0.8210), EN +0.2pt (within noise). Mirror-image on
the MLP head: ZH −1.8, EN −1.2. Net conclusion (`:361–364`): the RGCL term *redistributes* accuracy
head→memory (an almost exact ±1.8 swap on ZH), buys **no system-level accuracy**, whole system stays
below floor. **So the retrieval-contrastive objective demonstrably moves the kNN read-out in the
intended direction — but nets to zero at the system level in this regime.** Hold this for §4.

---

## 2. Independent re-derivation of the RA-HMD ablation (arXiv 2502.13061, Table 3a)

WebFetch of the HTML full text (`arxiv.org/html/2502.13061v1`, 2026-07-13) returned Table 3(a)
(HatefulMemes, in-domain) **identical** to the numbers in `exp-e2eq-e0.md:128–139`:

| config | AUC | Acc | in-domain increment |
|---|---|---|---|
| full LMM-RGCL | 91.1 | 82.1 | — |
| w/o Stage 1 (drop LoRA-SFT) | 84.4 | 74.2 | Stage 1 buys **+6.7 AUC / +7.9 Acc** |
| w/o Stage 2 (drop RGCL contrastive) | 90.2 | 81.4 | Stage 2 buys only **+0.9 AUC / +0.7 Acc** |

Stage definitions (fetched, paraphrase of the paper): **Stage 1** = LoRA fine-tune the LMM backbone
(trainable) jointly with an MLP + logistic-regression classifier head, loss = L_LM + L_CE. **Stage 2**
= freeze the backbone, **refine the same MLP + LRC heads** with the RGCL contrastive loss + CE, using
FAISS retrieval of pseudo-gold positives + hard negatives from the train set.

### 2.1 The load-bearing question: does "w/o Stage 2" keep a trained head, or drop it?

The fetch could **not** find a verbatim sentence stating the inference mode of the ablated rows (an
LLM-inferred "context indicates it uses the Stage-1 LRC head"). I therefore settle it by an
**internal-consistency proof that does not depend on the paper's prose**:

- "w/o Stage 2" = **81.4 acc / 90.2 AUC** — only **0.7 acc / 0.9 AUC below the full system (82.1/91.1)**.
- A **headless** read-out (raw features / untrained kNN, no classifier) is exactly what P9's C3-knn
  measured on video, and it lands **~4–5 acc pts BELOW** a trained head on the same features
  (HateMM −4.7, `EXP_p9:192`). On memes the "w/o Stage 1" row (RGCL head on *non-adapted* features) is
  74.2 — i.e. even a *trained* head on frozen features sits ~8 pts below full.
- A configuration scoring **81.4** (≈ full) therefore **cannot** be headless/raw. It must retain the
  **Stage-1-trained MLP+LRC classifier head** on the adapted features.

**Conclusion:** "w/o Stage 2" keeps the trained head. Hence **full − w/o-Stage2 = +0.7 acc genuinely
prices the RGCL-contrastive Stage-2 as the marginal refinement added on top of an already-trained
head** — which is *exactly* the increment C1 proposes to add on video beyond P9's C3-mlp. The
ablation **does** price C1's untested increment. **The kill does not overreach on this axis.**

Note the structure makes the pricing even cleaner than the kill states: Stage 2 in RA-HMD is not
"add a new head," it is "contrastively *refine* the existing head + add a kNN vote," and that whole
refinement is worth **+0.7 acc in-domain** on 8,500 memes. Its real value is out-of-domain (Table 3b:
−3.7 AUC / −7.6 Acc cross-dataset if removed) — an OOD/robustness lever, not an in-domain accuracy
lever. Our goal is in-domain +3 per dataset.

> ⚠️ Decimal caveat inherited from the drafts: RA-HMD numbers are HTML/LLM reads. Before any number
> reaches the paper, re-verify Table 1 / 3a / 3b against the PDF. The **qualitative** result (Stage 2
> in-domain ≈ +0.7; "w/o Stage 2" retains a trained head) is robust to ±0.x decimals.

---

## 3. The kill's decomposition, restated and stress-tested

The correct baseline for C1 in *our* pipeline is the **frozen-Qwen RGCL floor**, because the encoder
swap already banked the representation jump (HateMM +5.3 over frozen-CLIP, 3/3 seeds, both protocols;
`exp-encoder-3seed.md:184–198,228`). C1's ceiling over that floor decomposes into two increments:

1. **Stage-1 LoRA-adaptation over frozen-Qwen** — *directly measured on video by P9's C3-mlp*:
   +0.9 HateMM (s0) / +0.6 EN / +1.0 ZH, all within the ±1–2pt noise floor.
2. **Stage-2 RGCL contrastive refinement** — *not measured on video in its faithful sequential form*,
   but priced by RA-HMD's own in-domain ablation at **+0.7 acc** (on 10–15× more data).

Sum ≈ **+1.6 acc** over the frozen-Qwen floor, generously. That is **below the +3 bar and inside the
±1–2pt noise floor** the project has repeatedly hit (D3 in the reflection; TARC false-positive cells).

**Why this decomposition is methodologically strong, not hand-wavy:** the kill does **not** borrow the
tempting meme number — the Stage-1 "+7.9" from the ablation. It uses P9's *direct video measurement*
(+0.9) for the Stage-1 increment (the meme +7.9 does **not** transfer to video: our frozen-Qwen
features are already strong, so LoRA adaptation adds ~flat), and only borrows the anchor paper for the
one increment nobody measured on video (Stage-2, +0.7). Both are small. This is the right way to use a
cross-domain prior: measure what you can locally, borrow the anchor only for the residual, and cap it.

---

## 4. The strongest counter-argument I could build — and whether it survives

**Counter (KILL_OVERREACH steelman):** *The one faithful cell — Stage-1 LoRA → extract adapted
features → **sequentially** train an RGCL contrastive head + kNN on the frozen adapted features
(RA-HMD's exact released order) — was never run on video. P9 ran raw kNN with **no** Stage-2 head;
P9b ran RGCL **jointly** (degenerate at bs=1, 4-frame evidence cut, `EXP_p9:259–272`). P9's kNN went
below floor precisely because Stage-1 SFT reshaped the geometry FOR the MLP head and AGAINST a raw
kNN (`EXP_p9:213`) — and Stage-2's RGCL loss is the objective explicitly designed to reshape the space
back toward the retrieval memory. P9b's mechanism read confirms the direction is right (D3-knn −
C3′-knn = +1.8 ZH, `EXP_p9:353`). So the untested cell is not a re-run — it is the exact fix for P9's
failure mode, and the settling experiment is near-free: the Stage-1 adapted-feature caches ALREADY
exist on disk (`data/CLIP_Embedding/HateMM/{train,dev_seen,test_seen}_p9c3_hatemm_s{0,1,2}.pt`,
verified 2026-07-13), so Stage-2 on cached features is ~seconds/run (enc3seed precedent). Killing at
"zero GPU" spends nothing but also learns nothing on a genuinely open, near-free cell.*

**This is the strongest counter because every clause is TRUE:** the cell is genuinely untested; the
mechanism (Stage-2 RGCL repairs the kNN read-out) is real and independently confirmed by P9b; and the
settling cost is seconds because the adapted features are already cached.

**Why it does not survive the +3 bar (point by point):**

- **(a) The +0.7 ceiling already accounts for the kNN repair.** "w/o Stage 2" = 81.4 (§2.1) is the
  trained-head system *before* RGCL refinement; "full" = 82.1 is *after* the refinement **and** the
  kNN vote. So the +0.7 already includes whatever the contrastive term + kNN buy — precisely the
  "fix" the counter invokes. RA-HMD priced the entire fix at +0.7 in-domain on 8,500 memes. The
  counter cannot claim a fix the anchor paper already measured and found worth +0.7.
- **(b) Repairing the kNN read-out to *match a flat head* yields a flat system.** P9b showed the RGCL
  term moves knn up to ≈ mlp (ZH knn 0.8389 vs mlp 0.8412, `EXP_p9:346–348`) — but the head itself is
  the ceiling, and P9 already measured that head at ≈ frozen-Qwen floor on video (+0.9 HateMM, noise).
  A kNN that climbs to meet a flat head is still a flat system. P9b realized exactly this: net-zero
  redistribution, whole system below the protocol-matched floor (`EXP_p9:361–364`).
- **(c) The sequential-vs-joint distinction cuts the *right* way but not far enough.** Sequential
  frozen-feature Stage-2 does sidestep P9b's bs=1 in-batch degeneracy (`EXP_p9:259–272`) — a real
  methodological improvement — but the anchor paper's +0.7 is *itself* the sequential frozen-feature
  number (it is how RA-HMD ships and ablates). So the clean sequential run's own ceiling is +0.7, not
  more. Removing P9b's confound cannot exceed the confound-free number RA-HMD already reports.
- **(d) Stage-2's demonstrated strength is OOD, and MHC-EN is not an OOD task.** One might hope
  Stage-2's cross-domain lever (−7.6 acc if removed, Table 3b) rescues MHC-EN. But MHC-EN is judged
  in-domain on its own test; the goal is in-domain +3 per dataset. Repurposing an OOD-robustness term
  for an in-domain accuracy bar is speculative and unsupported by the anchor's in-domain +0.7.

**Net:** the counter correctly identifies that C1's cell is *untested and cheap*, but it cannot make
the *expected* gain exceed ≈ +1.6 over the frozen-Qwen floor, which is below the ±1–2pt noise floor
and far below +3. The route's realistic ceiling on HateMM is the frozen result it already has, not
beyond it (`exp-e2eq-e0.md:334–336`, honest-prior section — I concur with the draft's own prior).

---

## 5. The one honest caveat: "zero-GPU kill" slightly over-claims the saved cost

The kill is filed as a **zero-GPU G0 kill**. That framing is *strategically* correct (expected gain <
noise floor ⇒ do not spend GPU) but *factually* imprecise about the savings, because the distinguishing
experiment is not "an expensive GPU campaign avoided" — it is **~seconds of GPU on already-cached
adapted features** plus one localized loader edit (`run_rac_lmm.py` accepting the video LoRA-feature
cache, the single plumbing item exp-e2eq E-G0(d) already names). Verified on disk 2026-07-13:
`train_p9c3_hatemm_s{0,1,2}.pt`, `test_seen_p9c3_hatemm_s{0,1,2}.pt`, and the MHC-EN equivalents all
exist from P9. So the true decision is not "kill to save a GPU campaign" but "kill because the outcome
is pre-determined to be within noise, so even seconds of GPU + a reviewed edit + a test touch are not
worth the ceremony."

That is a **defensible** call under the project's own G0-cond doctrine (do not spend GPU when the
projected gain is structurally below the noise floor and the bar). But it means the resulting paper
sentence is an **inference** ("RA-HMD's ablation + our P9/P9b imply the faithful sequential cell would
land within noise"), not a **measurement**.

### Minimal cheap evidence that would convert the inference into a measurement (if desired)

**One HateMM seed-0 run:** train the Stage-2 frozen-feature RGCL head + kNN on the already-cached
`train_p9c3_hatemm_s0.pt` adapted features; read out DEV acc vs the frozen-Qwen DEV floor. Cost:
seconds of GPU (enc3seed ≈ 25 s/run) + the one loader edit + `codex-code-review` of that edit. **No
test touch needed** — a DEV read-out that lands within ±0.010 of the frozen-Qwen DEV floor (the
overwhelmingly likely outcome) closes the cell as a *measured* negative:
*"RA-HMD's exact sequential two-stage, run faithfully on video, still lands within noise of the
frozen-encoder RGCL floor at 7B"* — a cleaner, confound-free completion of the P9/P9b negative than
either P9 (raw-kNN, no Stage-2 head) or P9b (joint, bs=1-degenerate) could give.

This is genuinely optional. It changes the paper's *framing* (measured vs inferred negative), not the
*decision* (C1 cannot reach +3). If GPU minutes and one review cycle are cheaper than the reviewer's
time to defend an inference, run it; otherwise the kill stands as filed.

---

## 6. Bottom line

- **VERDICT: KILL_CONFIRMED.** The kill's factual chain is verified (P9/P9b numbers reproduce; the
  RA-HMD Table 3a numbers reproduce; the "w/o Stage 2" row provably retains a trained head, so +0.7
  genuinely prices C1's marginal increment). The +3 bar is structurally unreachable: C1 ≈ P9-Stage-1
  (flat on video, directly measured) + RA-HMD-Stage-2 (+0.7 in-domain ceiling) ≈ within noise.
- **Strongest surviving-tension:** the one faithful sequential cell is genuinely untested and the
  settling run is ~seconds (adapted features already cached). This does not change the decision, but it
  means the kill is best recorded as *"expected within-noise; optionally convertible to a measured
  negative at seconds of GPU"* rather than *"nothing to learn here."*
- **If the team wants an airtight paper sentence:** run the single HateMM-s0 sequential Stage-2 DEV
  read-out on the cached `*_p9c3_hatemm_s0.pt` features (no test touch). Otherwise the inferred kill is
  defensible as filed.
- **Do NOT** re-open C1 as a multi-seed, multi-dataset, test-touching campaign — that spends the
  ceremony budget on a route whose ceiling is +1.6, and reproduces P9/P9b.

---

### Provenance index (this review)

- P9 C3-mlp/knn table: `research-wiki/EXP_p9_lmm_rgcl_video.md:134–135`; HateMM `:189–192`; ZH
  reconciliation `:142–155`; floor definition `:118–129`.
- P9b verdict `:340–350`; mechanism read `:352–364`; bs=1 degeneracy `:259–272`.
- RA-HMD Table 3a / Table 1 / stage defs: arXiv 2502.13061v1 (HTML, WebFetch 2026-07-13); mirrored in
  `research-wiki/experiments/exp-e2eq-e0.md:112–139`.
- Frozen-Qwen HateMM floor (val-sel 0.8729 / final-ep 0.8682) and MHC-EN floor (0.7805 / 0.7847):
  `research-wiki/experiments/exp-encoder-3seed.md:154–159,164–170`; encoder-swap HateMM PASS `:184–198,228`.
- Cached P9 adapted features (settling-run cost): `data/CLIP_Embedding/HateMM/*_p9c3_hatemm_s*.pt`,
  `data/CLIP_Embedding/MHC/*_p9c3_*_s*.pt` (verified on the login node 2026-07-13).
- Qwen2.5-VL-7B is the only complete local VL checkpoint (16 GB); 32B/72B/Qwen3-VL absent
  (`exp-e2eq-e0.md:212–220`, cross-checked live).
