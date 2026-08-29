# B3 Forensic Recon — resolving the "LoRA-ZH +5.1 vs +1.0" conflict; scoping the cheapest clean paired test

**Agent:** forensic recon (read-only + CPU; no GPU/SLURM/commits). **Date:** 2026-07-14.
**Mission:** (1) reconcile the arcbase `0.8537` ZH LoRA gap (claim A, +5.1 acc) against the P9
"+1.0 ZH within noise" reading (claim B) from PRIMARY logs; (2) scope the cheapest clean
same-runner paired LoRA-Qwen-vs-CLIP test (B3), incl. the seed-semantics issue and HateMM.

**Bottom line up front:** **There is NO numerical conflict.** The word "floor" is overloaded.
Both claims are simultaneously true and both documents *agree* that `0.8537` is the arcbase
LoRA number. Claim A's +5.1 is `LoRA-encoder (0.8537) − frozen-CLIP (0.8027)`. Claim B's +1.0
is `P9-decision-level-LoRA-SFT (0.8635) − LoRA-encoder (0.8537)` — a *different, orthogonal* gap.
P9 itself explicitly credits the ~+4.5–5 "LoRA benefit we already had" (EXP_p9:143,161). The +5.1
gap is **REAL, multi-seed, and was never same-seed paired-tested** — but it is a **LoRA fine-tuning**
gap, not an MLLM-frozen-encoder gap (the frozen-Qwen swap *loses* on ZH). A same-seed paired
final-epoch verdict is **computable from existing logs with ZERO new GPU** (RECON-PREVIEW below).

---

## 1. CONFLICT RESOLUTION — the two "LoRA" systems and the overloaded "floor"

### 1.1 What claim A (arcbase 12223-12227) actually is

- **Source:** `research-wiki/experiments/exp-archive-knn-seeds.md:52-61` (per-seed val-sel) +
  `:157-165` Addendum 2 (final-epoch); master row `PAPER_MASTER_TABLES.md:43`.
- **System:** the arcbase **LoRA-*encoder*** stack. A LoRA-SFT of the Qwen2.5-VL-7B encoder whose
  features were extracted **once** into a single cache
  (`data/CLIP_Embedding/MHC_zh/{train,dev_seen,test_seen}_Qwen2.5-VL-7B-Instruct-LoRA_HF.pt`,
  mtime **2026-07-02**), fed into **our standard RGCL classifier + kNN read-out** (`Test_Retrieval`,
  `src/run_rac.py`), archive OFF (`archive_feats=None`).
- **Read-out:** our RGCL head + retrieval-kNN vote (`Test_Retrieval`). **Not** an MLP-only head.
- **Seeds:** 5 (12223-12227), **final-epoch** (ep29, no-selection) acc **0.8537 ± 0.0120** /
  F1 0.8259 ± 0.0124 (per-seed acc 0.8456/0.8389/0.8523/0.8658/0.8658).
- **Primary-log verification (this recon, re-read directly):**
  `slurm/logs/arcbase_MHC_zh_Qwen2.5-VL-7B-Instruct-LoRA_HF_seed{0,1,2}_1222{3,4,5}.trainlog`,
  epoch-29 `Test_Retrieval`: seed0 acc 0.8456 / macroF1 0.8181; seed1 0.8389 / 0.8113;
  seed2 0.8523 / 0.8226. Namespace header (seed0/12223) = `model='Qwen2.5-VL-7B-Instruct-LoRA_HF'`,
  `dataset='MHC_zh'`, `topk=20`, `epochs=30`, `batch_size=64`, `lr=0.0001`, `warmup=5`,
  `lambda_seg=0.0`, `archive_feats=None`, `proj_dim=1024`, `map_dim=1024`, `fusion_mode='align'`,
  `loss='triplet'`, `metric='cos'`, `hybrid_loss=True`. **All confirmed.**
- **Claim A "+5.1":** 0.8537 (arcbase LoRA, 5-seed final) − 0.8027 (5-seed consensus **frozen-CLIP**
  final floor, `PAPER_MASTER_TABLES.md:46`; `exp-consensus-zh-seeds.md`) = **+0.0510 acc / +0.0665 F1**.

### 1.2 What claim B (P9 "+1.0 ZH") actually is

- **Source:** `research-wiki/EXP_p9_lmm_rgcl_video.md:142-155` ("ZH floor reconciliation").
- **System:** a **completely different LoRA fine-tune** — the **P9 C3** joint LM + binary
  classifier-head LoRA-SFT via LLAMA-FACTORY `sft_classifier` (`Ver202512`), trained **per-seed**
  (caches `data/CLIP_Embedding/MHC_zh/{split}_p9c3_mhczh_s0/s1/s2.pt`, mtime **2026-07-08**).
- **Read-out for the +1.0:** **C3-mlp = the in-LMM MLP classifier head** (sigmoid ≥ 0.5), test
  0.8635 (per-seed 0.8456/0.8792/0.8658, EXP_p9:134,139-140), **3 seeds**. Explicitly the fork's
  own head, **not** our kNN.
- **What "+1.0" is against:** floor **(c)** in the reconciliation table (EXP_p9:150) =
  **`0.8537±0.012` — the arcbase LoRA number from §1.1** ("LoRA final-epoch, multi-seed, no-selection").
  So **P9's +1.0 = C3-mlp (0.8635) − arcbase-LoRA-encoder (0.8537)**, both LoRA, protocol-matched
  (final-epoch, no-selection). P9's verdict: the decision-level LoRA-SFT **matches, does not beat**,
  the encoder-level LoRA route we already had (EXP_p9:152-155,161).

### 1.3 Why both are true — the resolution

`0.8537` is the **common anchor** in both documents. The two claims measure **orthogonal gaps**:

| claim | comparison | value | what "floor" means | status |
|---|---|---|---|---|
| **A (+5.1)** | arcbase LoRA-encoder 0.8537 vs **frozen-CLIP** 0.8027 | +0.051 acc | frozen-CLIP baseline (no LoRA) | REAL, never same-seed paired |
| **B (+1.0)** | P9 decision-level LoRA-SFT 0.8635 vs **LoRA-encoder** 0.8537 | +0.010 acc | our *own* prior LoRA route | within noise |

P9 is **not** claiming LoRA is only +1.0 over CLIP. P9 *explicitly* credits the big gap as the LoRA
benefit it already possessed: EXP_p9:143 "The '+4.5pt' is vs the **frozen** RGCL floor (a), which
attributes the *entire LoRA benefit* to C3"; EXP_p9:161 "the +4.5 vs frozen is the LoRA benefit we
already had." (P9's row (a) uses frozen-**Qwen** 0.8188, giving +4.5; claim A uses frozen-**CLIP**
0.8027, giving +5.1 — same phenomenon, different frozen baseline.) So both documents concur:
**LoRA beats frozen by ~+4.5–5.1; the new decision-level LoRA-SFT adds essentially nothing on top.**

**This exact resolution was already reached independently** by the B1 pre-registration reviewer:
`refine-logs/B1_PREREG_REVIEW.md:57-75` (Task A.3) — "a real +5 acc / +6.6 F1 multi-seed gap, but
it is a **fine-tuning** result on a *'not-novelty'* lever, cross-runner/cross-experiment, never
same-seed paired, and explicitly barred from same-cell comparison." This recon confirms A.3 from
primary logs and adds the same-runner paired preview A.3 did not compute.

---

## 2. CACHE + COMPARABILITY FINDINGS (incl. the load-bearing seed-semantics issue)

### 2.1 Caches on disk (verified by CPU `torch.load`, HateVideo env)

**MHC-ZH LoRA cache — SINGLE SHARED artifact, NOT seed-specific:**
```
data/CLIP_Embedding/MHC_zh/train_Qwen2.5-VL-7B-Instruct-LoRA_HF.pt   (579, 3584)img (579,3584)txt (579,)lbl  mtime 2026-07-02
data/CLIP_Embedding/MHC_zh/dev_seen_..._LoRA_HF.pt                   ( 78, 3584)/( 78,3584)/( 78,)             2026-07-02
data/CLIP_Embedding/MHC_zh/test_seen_..._LoRA_HF.pt                  (149, 3584)/(149,3584)/(149,)             2026-07-02
```
Keys `['ids','img_feats','text_feats','labels']`; rows = ZH splits 579/78/149; dim 3584 (Qwen
hidden). **One file per split, no seed suffix.** LoRA-vs-CLIP test ids **set- AND order-identical**
(verified: `set-equal=True, order-equal=True` on train and test) → same 149 test videos, so the
read-out is over an identical test set.

**Frozen-CLIP ZH cache** (for the paired arm): `..._openai_clip-vit-large-patch14-336_HF.pt`,
(579/78/149) rows, **1024 img / 768 text**, mtime 2026-07-01. Also single/shared.

**HateMM LoRA caches — seed-specific, DIFFERENT origin:**
```
data/CLIP_Embedding/HateMM/{train,dev_seen,test_seen}_p9c3_hatemm_s0.pt / _s1.pt / _s2.pt   mtime 2026-07-08
```
These are the **P9-style per-seed LLAMA-FACTORY LoRA-SFT** caches (3 encoder draws). **There is NO
arcbase-style single-cache `Qwen2.5-VL-7B-Instruct-LoRA_HF.pt` for HateMM** and **no `arcbase_HateMM*`
LoRA trainlog exists** (`ls slurm/logs/ | grep arcbase.*[Hh]ate` → empty). So HateMM has *only* the
P9 per-seed LoRA, never an arcbase encoder-level LoRA run analogous to 12223-27.

### 2.2 THE SEED-SEMANTICS ISSUE (critical, load-bearing)

**The arcbase ZH "5 seeds" (12223-12227) all read the ONE shared LoRA cache above.** The LoRA
encoder was trained **once**; the extracted features are fixed; the 5 seeds vary **only the
downstream RGCL head training** (init + data shuffling via `--seed`). Corroborated in-project:
`exp-archive-knn-seeds.md:258-259` — "LoRA-only ZH is itself remarkably stable (std 0.014) — the
instability is introduced by the archive-kNN channel, not the encoder."

Consequences:
1. **"5 seeds" = head-seed variance only, on ONE LoRA-encoder draw.** The ±0.012 band does NOT
   include LoRA-SFT-training-seed variance. The single encoder could be a lucky/unlucky draw; the
   reported uncertainty is understated for the *LoRA lever* as a whole.
2. **The paired arm is symmetric at the head level.** The frozen-CLIP arm (13115) also reads a
   single shared cache — encoder fixed, head-seed varies. So pairing seed-s LoRA vs seed-s CLIP is a
   legitimate **head-level** same-seed paired test (same `--seed` ⇒ same shuffle order + head init,
   same code path ⇒ only the input features differ). It is NOT an *encoder-draw* paired test.
3. **Contrast P9's caches:** `p9c3_mhczh_s{0,1,2}` and `p9c3_hatemm_s{0,1,2}` ARE per-seed encoders
   (3 full LoRA-SFT draws) — those DO carry encoder-training variance. This is why P9's C3 numbers
   are noisier (e.g. C3-knn ZH 0.792/0.752/0.846, EXP_p9:135) than arcbase's head-only ±0.012.

### 2.3 Same-runner comparability: 12223-27 (LoRA) vs 13115 (CLIP)

**Both invoke `src/run_rac.py` with a BYTE-IDENTICAL argument vector except `--model`.** Verified by
reading both runners:
- `scripts/slurm/train_archive_baseline.sbatch` (12223-27, defaults EPOCHS=30 BATCH=64 WARMUP=5
  LAMBDA_SEG=0 SEG_MODE=full NUM_SUBCLIPS=4 EM_ROUNDS=2 CONS_TOPK=10 CONS_MARGIN=0.2).
- `scripts/slurm/enc3seed_zh_b1.sbatch` (13115) — its header comment even states each run is "the
  exact `train_archive_baseline.sbatch` python command (differs across configs ONLY in
  --dataset / --model / --seed)."

Shared flags (both): `--batch_size 64 --lr 0.0001 --epochs 30 --topk 20 --dataset MHC_zh
--proj_dim 1024 --map_dim 1024 --dropout 0.2 0.4 0.1 --fusion_mode align --hard_negatives_loss True
--no_hard_negatives 1 --final_eval False --group_name RAC_video_archive_seeds --metric cos
--loss triplet --batch_norm False --hybrid_loss True --warmup 5 --majority_voting arithmetic
--no_pseudo_gold_positives 1 --lambda_seg 0 --seg_mode full --num_subclips 4 --em_rounds 2
--consensus_topk 10 --consensus_margin 0.2 --Faiss_GPU False --force False`. The arcbase seed0
Namespace (§1.1) matches this exactly. **⇒ Same head hyperparameters, same archive-OFF path, same
GROUP; the ONLY difference is the encoder feature cache (LoRA-Qwen 3584-d vs frozen-CLIP 1024/768).**

**Code-version caveat (the one real gap):** 12223-27 ran **2026-07-04**; 13115 CLIP ran
**2026-07-14** (`B1_EXECUTION_RECORD.md:42`). Two reproduction gates show the archive-OFF path is
code-stable across this window: (a) 12223 reproduced job 12149 **bit-for-bit**
(`exp-archive-knn-seeds.md:35-37`); (b) frozen-Qwen s0 in 13115 reproduced old-code job 1151518
**exactly** (val-sel 0.7412/0.7919, final 0.7864/0.8188 — `B1_EXECUTION_RECORD.md:120-121,129`).
So the cross-version confound is low-risk, but not literally same-job.

**Seed-pairing legitimacy:** same `--seed` + same code ⇒ same data order + head init in both arms;
the test is over the identical 149 videos (§2.1). Pairing seed-s LoRA vs seed-s CLIP is legitimate as
a **head-level paired comparison**. Honest label: a paired read is defensible; an unpaired read + note
is the conservative fallback given the 10-day code gap.

---

## 3. RECON-PREVIEW — same-seed paired LoRA-Qwen vs frozen-CLIP, MHC-ZH (existing logs, ZERO GPU)

**Not a verdict — a preview from existing primary logs.** LoRA = 12223/12224/12225 (seeds 0/1/2);
CLIP = 13115 frozen-CLIP arm (`B1_EXECUTION_RECORD.md:109-114`). Final-epoch (ep29), same runner-cmd.

| seed | LoRA acc | CLIP acc | **ΔAcc** | LoRA F1 | CLIP F1 | **ΔF1** |
|---|---|---|---|---|---|---|
| 0 | 0.8456 | 0.8054 | **+0.0402** | 0.8181 | 0.7706 | **+0.0475** |
| 1 | 0.8389 | 0.8054 | **+0.0335** | 0.8113 | 0.7542 | **+0.0571** |
| 2 | 0.8523 | 0.8322 | **+0.0201** | 0.8226 | 0.7913 | **+0.0313** |
| **mean** | **0.8456** | **0.8143** | **+0.0313** | **0.8173** | **0.7720** | **+0.0453** |

- **Acc:** mean **+0.0313**, **3/3 seeds positive**, **2/3 seeds ≥ +0.030** (seed2 = +0.0201).
- **F1:** mean **+0.0453**, **3/3 seeds positive**, **3/3 seeds ≥ +0.030**.
- **vs 5-seed consensus CLIP floor** (unpaired, cross-runner): LoRA 5-seed 0.8537 − 0.8027 =
  **+0.0510 acc / +0.0665 F1** (= claim A).

**Load-bearing decomposition (all from 13115 + 12223-25, final-epoch, same runner, paired):**

| ZH arm (final-ep, seeds 0/1/2 mean) | acc | Δ vs frozen-CLIP |
|---|---|---|
| frozen-CLIP (13115) | 0.8143 | — (baseline) |
| **frozen-Qwen** encoder swap (13115) | 0.8031 | **−0.0113 (FAILS)** |
| **LoRA-Qwen** (12223-25) | 0.8456 | **+0.0313** |

⇒ On ZH the **frozen** MLLM-encoder swap *loses* (−0.011, mirroring the B1 result and
`B1_PREREG_REVIEW.md:66-69`), while the **LoRA fine-tune** wins (+0.031). The entire ZH gap is
attributable to **LoRA task/language adaptation of the encoder, NOT to MLLM-encoder identity.**
LoRA − frozen-Qwen (paired) = +0.0425 acc — the fine-tuning adds on top of the frozen 7B features.

**Interpretation gate (do not over-read):** this is a **head-seed-paired** preview on a **single
fixed LoRA-encoder draw**, LoRA vs CLIP conflates {encoder identity + task fine-tuning + 7B-vs-300M
capacity}, and the two arms are 10 days apart in code. It shows the +5.1 gap is real and reproduces
under the identical head runner as a same-seed paired near-pass (acc marginal at +0.031, F1 clear at
+0.045) — but it is a LoRA-lever result, classified project-wide as "MIXED performance lever, not
novelty" (`query_pack.md`; `B1_PREREG_REVIEW.md:64`).

---

## 4. HATEMM — LoRA-vs-CLIP picture from existing logs

- **No arcbase HateMM LoRA-encoder run exists** (§2.1). The only HateMM LoRA is P9 C3 (per-seed
  LLAMA-FACTORY joint LM+head SFT), `p9c3_hatemm_s{0,1,2}` (mtime 2026-07-08).
- **P9 C3 on HateMM** (EXP_p9:180-196): C3-mlp (in-LMM head) **0.8698** (s0 only; s1/s2 were GPU-blocked)
  vs trained-RGCL floor **0.8605** = **+0.9pt (≈ floor)**; C3-knn (our memory) **0.814**
  (s0/s1/s2 .823/.814/.805) = **−4.7pt BELOW floor**. Frozen best stack = 0.870 (MoRE §3.2,
  `PAPER_MASTER_TABLES.md:36`).
- **Verdict shape:** HateMM LoRA-SFT does **NOT** pass the +0.03/+0.03 test — it lands ≈ floor on the
  MLP head and −4.7 below on our kNN. **HateMM's banked positive is the FROZEN encoder swap**
  (Qwen-vs-CLIP, `exp-encoder-3seed.md:25-34`: +0.053–0.056 acc / +0.056–0.066 F1, 3/3 seeds, both
  protocols — the project's most robust positive).
- **The ZH/HateMM asymmetry (the "unexplained" part):** the two datasets have **opposite lever
  profiles.** HateMM: **frozen-swap wins, LoRA flat**. ZH: **frozen-swap loses (−0.011), LoRA wins
  (+0.031)**. No single lever ("MLLM-as-encoder") passes +0.03/+0.03 on both datasets under the same
  mechanism — HateMM needs frozen-Qwen, ZH needs LoRA. This is precisely why the "encoder positive =
  HateMM only" memory holds for the *frozen* lever, and why ZH's gap has stayed unbanked.

---

## 5. HONEST-PRIOR — will the ZH LoRA gap survive a clean test?

**Why it might be REAL:**
- LoRA-SFT adapts the encoder to the **task + language** (Chinese hate video). ZH is the dataset
  where frozen features are weakest relative to headroom, so task/language adaptation has the most to
  give — consistent with frozen-Qwen *losing* on ZH while LoRA *wins*.
- The preview is **3/3 positive on both metrics**, same-seed paired, same runner-command, same 149
  test videos, F1 mean +0.045 clears the bar cleanly; the LoRA-only arm is intrinsically stable
  (head-seed std 0.012, `exp-archive-knn-seeds.md:258`).
- The two reproduction gates (12223=12149 bit-for-bit; frozen-Qwen s0 = old 1151518 exactly) show the
  archive-OFF path is code-stable, so the 10-day gap is unlikely to move the preview much.

**Why it might NOT survive a clean test / not count:**
- **Single encoder draw.** The 5 arcbase seeds share ONE LoRA cache (§2.2) — the ±0.012 band is
  head-only; a differently-seeded LoRA-SFT could land lower. P9's per-seed LoRA (which *does* re-train
  the encoder) is noisier and its kNN read-out on the SFT'd space *loses* (ZH C3-knn 0.7964,
  EXP_p9:135) — evidence the gap is fragile to how the encoder is fine-tuned/read-out.
- **Wrong lever for the novelty claim.** LoRA is a "MIXED performance lever, not novelty"
  (`B1_PREREG_REVIEW.md:64`; `query_pack.md`). The +5.1 conflates fine-tuning + encoder identity +
  capacity; it does not isolate "MLLM-as-encoder," and the *frozen* swap (which does isolate it)
  fails on ZH.
- **Head-hyperparameter / code drift.** 10-day code gap; a maximally clean paired verdict wants both
  arms produced by the identical job/code.
- **Selection-of-what-got-recorded.** The arcbase 12223-27 were run as a LoRA-*only control* for the
  archive-kNN experiment (`exp-archive-knn-seeds.md:30-34`), not as a LoRA-vs-CLIP encoder test; the
  frozen-CLIP same-runner arm only exists because B1 (13115) happened to produce it 10 days later.
  The pairing is opportunistic, not pre-registered as such.
- **Barred by the project's own accounting.** `PAPER_MASTER_TABLES.md:58` declares the LoRA-Qwen main
  stack and the frozen-CLIP floor "**不可直接同格并比**" (not directly comparable side-by-side).

**Net honest prior:** the gap is a **genuine LoRA-fine-tuning effect** that would very likely
reproduce as a same-seed paired near-pass under a clean same-code run — but it is **not** a
frozen-encoder-swap pass, is measured on a **single encoder draw**, and sits on a lever the project
classifies as non-novelty. A clean B3 verdict would confirm the *magnitude/pairing* but would not
convert it into an "MLLM-as-encoder" banked positive.

---

## 6. WHAT A CLEAN B3 VERDICT NEEDS

**Existing logs SUFFICE for a head-level paired final-epoch read** (§3): LoRA-Qwen vs frozen-CLIP,
MHC-ZH, +0.0313 acc / +0.0453 F1, 3/3 positive, same runner-command, same test set. This is already a
defensible RECON-PREVIEW; the only impurity is the 10-day code gap (gated as low-risk).

**Minimal run to make it a formally clean B3 verdict (cheap):**
- Add **3 rows** to `scripts/slurm/enc3seed_zh_b1.sbatch`'s CONFIGS array:
  `"MHC_zh Qwen2.5-VL-7B-Instruct-LoRA_HF 0/1/2"`. Features are cached ⇒ each run ~20-25 s;
  **~2 min GPU total** (the "~10 min" budget is conservative). This produces the LoRA arm under
  **literally the same job / code / runner** as the 13115 CLIP arm ⇒ removes the code-version gap and
  yields a clean same-code same-seed paired final-epoch verdict (acc + F1, both protocols).
  - Guard: `--group_name RAC_video_archive_seeds` LoRA dirs already exist (2026-07-04 runs), so use a
    fresh GROUP or `FORCE=True` to avoid collision, OR just re-read 12223-25 (the reproduction gates
    make a re-run near-certain to match). New logs: `enc3s_MHC_zh_Qwen2.5-VL-7B-Instruct-LoRA_HF_seed{0,1,2}_<JID>.trainlog`.

**What NO existing log and NO cheap run can give (state as an inherent limitation):**
- **LoRA-encoder-training-seed variance.** The arcbase cache is a single draw; the P9 per-seed caches
  use a different recipe. A test of the LoRA *lever's* encoder-draw stability needs ≥3 fresh LoRA-SFT
  re-trainings + re-extraction (hours of GPU) — a different, larger question, arguably out of B3 scope.
- **A HateMM arcbase-LoRA counterpart** — none exists; would need a HateMM LoRA-encoder SFT + cache
  before any HateMM LoRA-vs-CLIP arcbase-style pairing is possible. (P9 already tested LLAMA-FACTORY
  LoRA on HateMM: ≈ floor / below-floor — §4.)

---

## Provenance index (file:line)
- Claim A: `exp-archive-knn-seeds.md:52-61` (val-sel), `:157-165` (final-epoch Add.2), `:169-180`
  (weight-identity audit), `:35-37` (12223=12149 bit-for-bit); `PAPER_MASTER_TABLES.md:43,46`.
- Claim A primary logs: `slurm/logs/arcbase_MHC_zh_Qwen2.5-VL-7B-Instruct-LoRA_HF_seed{0,1,2}_1222{3,4,5}.trainlog`
  ep29 (re-read this recon).
- Claim B: `EXP_p9_lmm_rgcl_video.md:131-155` (reconciliation, floor (c)=0.8537), `:134,139-140`
  (C3-mlp per-seed), `:143,161` ("LoRA benefit we already had"), `:180-196` (HateMM).
- Independent prior resolution: `B1_PREREG_REVIEW.md:12-75` (Task A / A.3), `:66-69` (frozen-Qwen ZH loses).
- CLIP paired arm: `B1_EXECUTION_RECORD.md:105-125` (13115 per-seed), `:120-121,129` (reproduction gate).
- Runners: `scripts/slurm/enc3seed_zh_b1.sbatch`, `scripts/slurm/train_archive_baseline.sbatch`.
- Caches: `data/CLIP_Embedding/MHC_zh/*LoRA_HF*.pt` (single, 2026-07-02),
  `data/CLIP_Embedding/{MHC_zh,HateMM}/*p9c3*_s{0,1,2}.pt` (per-seed, 2026-07-08); id-alignment +
  dims verified by CPU `torch.load` this recon.
