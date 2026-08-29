# B4 Forensic Recon — has "LoRA-Qwen encoder on MHC-EN, paired 3-seed vs frozen-CLIP" ever been measured, and what would it cost?

**Agent:** B4 forensic recon (read-only; ZERO GPU/SLURM/commits). **Date:** 2026-07-14.
**Sibling doc mirrored:** `refine-logs/B3_FORENSIC_RECON.md` (ZH lineage). **Cell under recon:**
LoRA-SFT-adapted Qwen2.5-VL-7B *encoder* features on **MHC-EN** (dataset code `MHC`), fed to the
standard archive-OFF RGCL head (the `enc3s`/`arcbase` protocol), paired 3-seed vs frozen-CLIP,
dual-protocol (val-selected + final-epoch), decision rule = mean Δacc ≥ +0.030 AND mean ΔmF1 ≥ +0.030
AND sign 3/3 (identical to B3 / `exp-encoder-3seed.md:73-85`).

**Bottom line up front:** The **exact paired 3-seed protocol was never executed** (no `enc3s_MHC_*LoRA*`
or `arcbase_MHC_*LoRA*` trainlog exists — only `MHC_zh` has those). **But the cell is NOT virgin.**
The identical adapter + identical feature cache + identical RGCL+kNN head were already measured on
MHC-EN at **seed 0, both protocols**, and the result is a **banked, wiki-noded, adversarially-verified
NEGATIVE**: LoRA **REGRESSES** below both frozen floors on EN (`exp-lora-sft-encoder.md:21`,
verdict `partial`, 2026-07-02). Everything upstream (adapter + extracted features + the frozen-CLIP
3-seed control) already exists, so **formalizing the 3-seed verdict costs ~2 min GPU** — but it would
**close/formalize a known negative, not explore an open question.** Honest prior: **FAIL both
protocols** (seed0 anchor: final-epoch +0.006 acc, val-selected **−0.031** acc).

---

## (i) HAS THE EN-LoRA CELL BEEN MEASURED BEFORE? — **PARTIALLY YES (seed0, banked NEGATIVE); the 3-seed paired protocol: NO**

**The 3-seed paired object does NOT exist.** Exhaustive log check:
`ls slurm/logs/ | grep _MHC_ | grep -i LoRA | grep -v zh` returns exactly **two** files, **both seed0**:
- `arc_MHC_Qwen2.5-VL-7B-Instruct-LoRA_HF_knn_a0.25_seg0full_12211.trainlog` — archive-**ON** (kNN α=0.25).
- `rgcl_MHC_Qwen2.5-VL-7B-Instruct-LoRA_HF_2723309.trainlog` — archive-**OFF**, old-code (`group_name=RAC_video`).

There is **no** `enc3s_MHC_*-LoRA_HF_seed{0,1,2}_*.trainlog` and **no** `arcbase_MHC_*-LoRA_HF_*`. The
EN arcbase 3-seed runs used the **frozen** Qwen (`arcbase_MHC_Qwen2.5-VL-7B-Instruct_HF_seed{1,2,3}_1227{5,6,7}`,
not `-LoRA_HF`). So the archive-OFF **3-seed LoRA** arm exists **for ZH only** (12223-27 / 13150), **never for EN.**

**But the underlying comparison IS measured (seed0) and is a canonical banked negative.** The dedicated
wiki node `research-wiki/experiments/exp-lora-sft-encoder.md` (date 2026-07-02, `verdict: partial`,
`confidence: high`, "adversarially verified") is *exactly* this cell — "LoRA-SFT of Qwen2.5-VL encoder,
prediction via RGCL contrastive + kNN head":
> `:21` "**MHC-EN 0.6916 M-F1 / 0.7516 acc = REGRESSES below frozen CLIP (0.783 acc) and frozen Qwen
> (0.789 acc).** … helps ZH (best-ever), hurts EN (regresses below both frozen floors)."

**Primary-log confirmation of that seed0 reading (re-read this recon):**
`rgcl_MHC_Qwen2.5-VL-7B-Instruct-LoRA_HF_2723309.trainlog`
- `:250` `Test_Retrieval Epoch 26 macroF1: 0.6916 … acc: 0.7516 roc: 0.8488` (val-selected epoch, selEp26).
- `:275` `Test_Retrieval Epoch 29 macroF1: 0.7302 … acc: 0.7702 roc: 0.8593` (final-epoch).
- Cross-check: archive-ON `arc_...12211.trainlog` final-ep = **0.7702 / 0.7302** (identical final readout).

Corroborated across five independent records: `ITERATION_LOG.md:838,846,861,864,882`; `gap_map.md:9`
(names the jobs: "first LoRA-adapted RGCL runs on disk; jobs **2723309/2794237**"); `query_pack.md:32,42,44`;
`EXP_p9_lmm_rgcl_video.md:62-63,159`; `DESIGN_iter1.md:274,351,357`.

### Seed0 paired preview (what a 3-seed run would be built on) — ZERO GPU, from existing logs

LoRA arm = `2723309` (archive-OFF seed0). CLIP arm = `enc3s_MHC_openai_clip-...-336_HF_seed0_12850`
(= `exp-encoder-3seed.md:95,165`; identical 161-video EN test, see (ii) id-alignment). frozen-Qwen arm
= `exp-encoder-3seed.md:96,166`.

| protocol | LoRA (s0) | frozen-CLIP (s0) | **Δ (LoRA−CLIP)** | frozen-Qwen (s0) | LoRA vs Qwen |
|---|---|---|---|---|---|
| **val-selected** | 0.7516 / 0.6916 | 0.7826 / 0.7113 | **−0.0310 acc / −0.0197 F1** (REGRESS) | 0.7888 / 0.7378 | LoRA below Qwen |
| **final-epoch**  | 0.7702 / 0.7302 | 0.7640 / 0.7145 | **+0.0062 acc / +0.0157 F1** (≪ +0.030) | 0.8012 / 0.7596 | LoRA below Qwen |

On EN the LoRA encoder sits **below both frozen encoders** (matches `ITERATION_LOG.md:861,864`) — the
mirror image of ZH, where LoRA *beat* frozen-Qwen by +0.043 (`B3_FORENSIC_RECON.md:184-190`). The
seed0 anchor is a **clear FAIL**: val-selected is *negative*, final-epoch is +0.006 (one-fifth of the bar).

---

## (ii) ADAPTER / CACHE INVENTORY — the whole upstream chain already exists

| artifact | path | mtime | status |
|---|---|---|---|
| **EN LoRA adapter** | `logging/lora/MHC/adapter_model.safetensors` (+ `adapter_config.json`) | 2026-07-02 | **EXISTS** |
| **EN LoRA feature cache** | `data/CLIP_Embedding/MHC/{train,dev_seen,test_seen}_Qwen2.5-VL-7B-Instruct-LoRA_HF.pt` | 2026-07-02 | **EXISTS** |
| **EN frozen-CLIP cache** | `data/CLIP_Embedding/MHC/{…}_openai_clip-vit-large-patch14-336_HF.pt` | 2026-07-01 | EXISTS |
| **EN frozen-CLIP 3-seed control (enc3s)** | `enc3s_MHC_openai_clip-...-336_HF_seed{0,1,2}_12850.trainlog` | 2026-07-11 | **EXISTS (all 3 seeds)** |
| **EN LoRA head runs** | `rgcl_...LoRA_HF_2723309` (arc-OFF), `arc_...LoRA...a0.25_12211` (arc-ON) | — | EXISTS, **seed0 only** |
| **EN LoRA 3-seed enc3s/arcbase arm** | (would be `enc3s_MHC_...-LoRA_HF_seed{0,1,2}`) | — | **MISSING — the only gap** |

**Adapter provenance & VETO compliance (single-dataset own train split — PASS):**
- `logging/lora/MHC/adapter_config.json`: base `Qwen/Qwen2.5-VL-7B-Instruct`, **r=16, α=32**, `task_type=CAUSAL_LM`,
  targets = all MLP `{gate,up,down}_proj` layers 0-27 + `{q,k,v,o}_proj`.
- `logging/lora/MHC/README.md`: "fine-tuned … on the **mhc_lora_train** dataset" (single dataset).
  `all_results.json`: 2.96 epochs, eval_loss 0.162, train_runtime 8222 s (~2.3 h), 204 steps.
- SFT data is **EN own train split only**: `data/lora_sft/MHC/train_yn.json` = **549 records** (= EN train
  count). **Separate per-dataset adapters** exist (`logging/lora/{MHC, MHC_zh}`, each 161 MB, both 2026-07-02);
  the EN cache was extracted by attaching `logging/lora/MHC` (`lora_embed_12146.out:1` `LORA_DIR=logging/lora/MHC`,
  `splits=train,val,test`, merge_and_unload). ⇒ **No cross-dataset mixing; own-split only; binary hateful/normal
  labels (no gold aux, no OCR) — clears all three standing vetoes.**

**Cache integrity (verified this recon via CPU `torch.load`, HateVideo env):** EN LoRA cache = train
(549, 3584)img/(549,3584)txt/(549)lbl, test (161, 3584)/(161,3584)/(161); keys `['ids','img_feats',
'text_feats','labels']`. **id-alignment LoRA-vs-CLIP: set-equal AND order-equal = True on both train and
test; labels bit-identical** (EN test pos 49/161, train pos 168/549). ⇒ any paired read is over the
identical 161 EN test videos, symmetric to the ZH B3 setup.

---

## (iii) COST TABLE TO MEASURE THE CELL — ~2 min GPU (branch (a): adapter already exists)

Because the adapter and features are already on disk, this is the **(a)** branch of the recon question
(no SFT training, no extraction). It is the **same recipe B3 used for ZH** (`B3_FORENSIC_RECON.md:265-273`).

| step | needed? | status | cost |
|---|---|---|---|
| LoRA-SFT adapter (EN own train split) | prerequisite | **DONE** (`logging/lora/MHC`, 2026-07-02) | 0 (≈2.3 h GPU if ever re-trained) |
| LoRA feature extraction train/dev/test | prerequisite | **DONE** (`…/MHC/*-LoRA_HF.pt`, job 12146) | 0 |
| frozen-CLIP 3-seed control (enc3s) | comparison arm | **DONE** (`enc3s_MHC …CLIP… 12850`, all 3 seeds) | 0 (reuse) or ~1 min if re-run same-code |
| **LoRA arm under enc3s head, seeds 0/1/2** | **THE missing piece** | **NOT RUN** | **~2 min GPU** (cached feats → ~20-25 s/run) |
| G-repro gate | verify | seed0 anchor = `2723309:275` (0.7702/0.7302 final) | folded into the run |

**Concrete recipe (no GPU spent by this recon):** copy `scripts/slurm/enc3seed.sbatch` (the 12850 runner;
`CONFIGS` at lines 31-41 already lists `"MHC $CLIP {0,1,2}"`) and add three rows
`"MHC Qwen2.5-VL-7B-Instruct-LoRA_HF {0,1,2}"`. One serial sbatch, cached features ⇒ seconds each.
Produces `enc3s_MHC_...-LoRA_HF_seed{0,1,2}_<JID>.trainlog`; pair vs the existing 12850 CLIP arm; run the
same dual-protocol decision rule.

**The (b) branch (no adapter) is NOT triggered** — the EN adapter exists, so no LoRA-SFT re-training (which
would be ~2.3 h GPU + extraction) is required.

**One caveat to gate at run time (same as B3):** the seed0 anchor `2723309` is *old-code* `group_name=RAC_video`.
`exp-encoder-3seed.md:126-146` already retired the old-vs-new-code confound **bit-for-bit** for CLIP and
frozen-Qwen seed0 (every post-hoc flag inert at defaults); the LoRA seed0 has not itself been bit-for-bit
re-verified under enc3s code — that reproduction check *is* the B4 G-repro (expected to pass, since only
`--model` differs and the archive-OFF path is flag-gated).

---

## (iv) NON-ISOMORPHISM VERDICT vs P9 — B4 is a RE-MEASURE of a banked cell, NOT a P9 re-proposal

Two distinct LoRA *systems* exist (the B3 decomposition, `B3_FORENSIC_RECON.md:19-72`):

1. **Encoder-level LoRA** (`logging/lora/MHC`, cache `…-LoRA_HF.pt`, 2026-07-02) → **exactly the B4 cell.**
   Banked negative on EN via `exp-lora-sft-encoder.md:21` and `ITERATION_LOG.md:861,864`. **B4 IS isomorphic
   to this** at seed0 — same adapter, same cache, same RGCL+kNN head. B4 = *3-seed formalization of an
   already-banked seed0 negative*, not a virgin cell.
2. **P9 decision-level LoRA-SFT** (LLAMA-FACTORY `sft_classifier`, per-seed caches
   `data/CLIP_Embedding/MHC/{…}_p9c3_mhc_s{0,1,2}.pt`, `_p9c3p_en_s*`, `_p9d3_en_s*`, mtime 2026-07-08) —
   a **different recipe** (joint LM + binary head SFT, per-seed encoders). **B4 is NOT isomorphic to P9 C3/D3.**

**Did P9 measure EN LoRA features vs floor?** Yes — its *own* decision-level variant, and it also FAILED EN:
`EXP_p9:132` EN floor 0.7847; `:212` "C3-knn vs floor: **EN −2.7**, ZH −2.2, HateMM −4.7 (head: EN +0.6…)";
`:159` "(EN low-prior; iter-2 encoder-LoRA also regressed EN, so frozen IS EN's best)". Crucially **P9 itself
cites the B4 cell as already-closed** to justify skipping EN: `:62-63` "MHC-EN is included but **a-priori
UNLIKELY** — the **iter-2 LoRA-SFT of this same Qwen-VL encoder already regressed EN (0.7516/0.6916, below
both frozen floors)** and crossed 0.85 on neither MHClip split."

**Verdict:** B4 is **not** a P9 re-proposal (different mechanism), **but it IS a re-measure of the
`exp-lora-sft-encoder.md` cell** — a cell the project already banks as a verified negative on EN and which
P9 explicitly leaned on as closed. Both LoRA families (encoder-level *and* decision-level) independently fail
EN. A B4 run would upgrade a seed0-anchored banked negative to a formal 3-seed paired verdict; it would not
open new ground.

---

## (v) HONEST PRIOR — falsifiable expectation (no advocacy)

**Prediction: EN LoRA vs frozen-CLIP, paired 3-seed, FAILS both protocols.** Point estimates from the seed0
anchor: **final-epoch mean Δacc ≈ +0.006** (≪ +0.030), **val-selected mean Δacc ≈ −0.031** (negative).
Both metrics on both protocols are expected below the bar; val-selected is expected *negative*.

**Basis (four independent legs):**
1. **The cell's seed0 is already measured and banked as REGRESS** (`exp-lora-sft-encoder.md:21`;
   `2723309:250` val-sel LoRA−CLIP = −0.031 acc / −0.020 F1). This is not a prior over an unknown — it is a
   near-complete measurement missing only seeds 1/2.
2. **Even the frozen-Qwen encoder swap FAILED EN** (`exp-encoder-3seed.md:200-234`: val-sel mean Δacc +0.019
   /2-of-3; final-ep +0.006 /1-of-3), and **LoRA sits BELOW frozen-Qwen on EN** (val-sel 0.7516 < 0.7888;
   final 0.7702 < 0.8012). For LoRA to clear +0.030 vs CLIP it must exceed a frozen-Qwen that itself only
   reaches ≈CLIP+0.006 — and LoRA is *worse* than that frozen-Qwen. Mechanically implausible.
3. **EN is data/label-limited** (SAV verdict; 549 train, EN test 161). LoRA-SFT on 549 EN samples *degrades*
   the encoder (the opposite of ZH, where LoRA relieved the documented handicap of an English-CLIP text tower
   on Chinese — `query_pack.md:32`). This is the accepted explanation for the ZH/EN sign-flip, not a mystery.
4. **P9's independent decision-level EN LoRA-SFT also fails** (C3-knn −2.7 vs floor; `EXP_p9:212`).

**What would falsify it:** seeds 1 and 2 both delivering ≥ +0.06 acc (LoRA−CLIP) while seed0 is +0.006, so
the 3-seed mean crosses +0.030 with 3/3 sign. Given (a) seed0 val-selected is already *negative*, (b) EN
per-seed encoder deltas have run ±0.02-0.05 (`exp-encoder-3seed.md:200-221`), and (c) LoRA underperforms
even frozen-Qwen on EN, this is a **<5% outcome**. The expected value of the run is **formal closure of a
known negative**, not a new positive — the mirror image of ZH B3, where the *same* recipe delivered a
marginal final-epoch PASS (`B3_VERDICT_REVIEW.md:20-24`).

---

## Provenance index (file:line / path)
- Canonical banked cell: `research-wiki/experiments/exp-lora-sft-encoder.md:1-24` (verdict partial, EN regresses, 2026-07-02).
- Seed0 primary logs (re-read): `slurm/logs/rgcl_MHC_Qwen2.5-VL-7B-Instruct-LoRA_HF_2723309.trainlog:250` (val-sel ep26 0.7516/0.6916), `:275` (final ep29 0.7702/0.7302); `arc_MHC_...LoRA...a0.25_seg0full_12211.trainlog` (arc-ON, final 0.7702/0.7302).
- CLIP / frozen-Qwen EN s0-s2 controls: `exp-encoder-3seed.md:95-98,161-170,200-241`; logs `enc3s_MHC_*_12850.trainlog`, `arcbase_MHC_Qwen2.5-VL-7B-Instruct_HF_seed{1,2}_1227{5,6}`.
- Corroboration of the EN regress: `ITERATION_LOG.md:838,846,861,864,882`; `gap_map.md:9` (jobs 2723309/2794237); `query_pack.md:32,42,44`; `DESIGN_iter1.md:274,351,357`; `EXP_p9_lmm_rgcl_video.md:62-63,132,159,212`.
- Adapter + SFT: `logging/lora/MHC/{adapter_config.json,README.md,all_results.json}`; `data/lora_sft/MHC/train_yn.json` (549); extraction `slurm/logs/lora_embed_12146.out:1`.
- Caches / id-alignment: `data/CLIP_Embedding/MHC/*-LoRA_HF.pt` (2026-07-02), `*_openai_clip-...-336_HF.pt` (2026-07-01); torch.load set/order/label equality verified this recon.
- Runner to add the LoRA arm: `scripts/slurm/enc3seed.sbatch:31-41` (CONFIGS). B3 precedent: `B3_FORENSIC_RECON.md:259-283`, `B3_VERDICT_REVIEW.md:14-46`.
- No EN-LoRA 3-seed exists: `ls slurm/logs/ | grep _MHC_ | grep -i LoRA | grep -v zh` = only `2723309` + `12211` (both seed0).
