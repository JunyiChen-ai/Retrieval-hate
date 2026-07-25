# MEMORY-BANK CURATION — $0 FORENSIC RECON (batch-4 LEAD, LITSWEEP3 §4 / shortlist #1)

**Agent:** curation-recon (zero-GPU / zero-SLURM / zero-Modal / zero-test-touch forensic recon).
**Date:** 2026-07-25 NZST. **Deliverable:** this doc + one local commit (no push). `state/` untouched.
**Candidate:** train-label-only curation (prune/re-weight/prototype-select) of the deployed kNN memory bank
via LOO-kNN-influence / Data-OOB / condensed-NN / prototype-select on BANKED caches + a balanced-bank arm
for macro-F1. Templates honored: `ISR_PREGATE_RECORD.md` (`a6e41f8`), `LP_GATE_RECORD.md` (`7be6e3f`, F63),
`LITSWEEP3_DATA_CENTRIC.md` §4/§9 (`8629188`).

**VERDICT (up front): PARK.** Not on prior alone — on a **machinery blocker the LITSWEEP3 §4 costing missed**:
the deployed vote runs over the **trained head embedding**, and the deployed floor heads (13150 ZH-LoRA,
13241 HateMM-curric, 3 seeds each) **are deleted from disk**. Only the *pre-head* feature caches + trainlog
numbers survive. So a **faithful, multi-seed** curation pregate of the DEPLOYED bank is **NOT $0** (needs a
6-head GPU re-mint first), and the only **$0 CPU** object (the raw fused key) is **seed-independent →
single-draw → cannot satisfy the multi-seed discipline the tasking requires** to be immune to the withdrawn
archive-as-key failure class. F63 + Wall-A/C then cap the prior. Details below; a $0 paper-value-only screen
is spec'd in §7 for the orchestrator's option, explicitly non-binding on the floor.

---

## 1. MECHANISM PIN — the deployed vote, and exactly where a curated bank plugs in

**Deployed decision = rank-weighted signed-cosine top-20 kNN over own-train memory, in the trained HEAD
embedding space.** Traced end-to-end (not the probe re-impls; the real training/eval path):

| step | file:line | what |
|---|---|---|
| per-epoch eval call | `src/run_rac.py:813-830` | `retrieve_evaluate_RAC_(train_dl, evaluate_dl, model, largest_retrieval=args.topk=20, threshold=-1, ...)` then `compute_metrics_retrieval(..., majority_voting='arithmetic', topk=20, use_sim=True)` — Val_Retrieval / Test_Retrieval |
| memory build | `src/model/evaluate_rac.py:349-377` | iterate **whole `train_dl`** → `train_feats = model(img,txt,return_embed=True)[1]` (head embed), `train_labels`, `train_ids`. **This is the bank.** |
| query build | `src/model/evaluate_rac.py:385-399` | dev/test → `evaluate_feats` (same head embed) |
| retrieval | `src/model/evaluate_rac.py:412-430` | `faiss.IndexFlatIP(dim)`; L2-normalise train+eval (cosine); `index.add(train_feats)`; `D,I = index.search(evaluate_feats, 20)` |
| neighbour list | `src/model/evaluate_rac.py:435-474` | per query top-20 → `retrieved_scores`=cosine `D`, `retrieved_label`=`train_labels[I]` |
| the vote | `src/utils/metrics.py:262-301` | `use_sim` path: `w=[20..1]` (229-230); `map=(lab*2-1)*sim` (268-270); `vote=Σ(map·w)/Σw` (283-284); predict `1` iff `sigmoid(vote)≥0.5 ⇔ vote≥0` (300) |
| head | `src/model/classifier.py:115-148` | `classifier_hateClipper.forward`: `img_proj`/`text_proj` → L2-norm → **`align` (Hadamard) fuse** → `embed = self.mlp[:-2](x)`; `map_dim=1024, num_layers=3, proj_dim=1024, dropout=[0.2,0.4,0.1]` |

**Where a curated bank plugs in:** a curated bank = a **row subset (prune) and/or per-row weight** of the
train memory assembled at `evaluate_rac.py:349-377`, applied identically to every eval query. Concretely,
index only the retained rows at line 429 (`index.add(train_feats[keep])`, `train_labels[keep]`,
`train_ids[keep]`); an optional per-row weight enters as a multiplicative factor on `retrieved_scores`
before the vote. **Eligible-for-pruning split = TRAIN ONLY.** Dev/test are queries; they are **never** in the
memory (disjoint sets; no self-exclusion code path is used for them). This makes curation a **symmetric,
global, train-label-only** edit — categorically distinct from F47/F66 per-item (per-test-instance) selection.

**Floor's bank = full own-train memory — CONFIRMED.** No sub-selection anywhere in `retrieve_evaluate_RAC_`;
the whole `train_dl` is indexed. ZH-LoRA train **V=579**; HateMM-curric train **V≈1100** (verified 579 for ZH
by loading the cache; HateMM count inherited from banked N4 guards).

---

## 2. CACHE / CKPT INVENTORY (exists / missing, per dataset × seed) — the decisive finding

The vote indexes the **head embedding**, which = `head_ckpt(seed) ∘ raw_features`. Two objects are needed;
only one survives.

### 2.1 Pre-head feature caches (img/text 3584-d + labels) — **seed-INDEPENDENT** (one LoRA extraction)
| floor | split | path (`data/CLIP_Embedding/…`) | sha256 (full) | status |
|---|---|---|---|---|
| ZH-LoRA (13150) | train (V=579) | `MHC_zh/train_Qwen2.5-VL-7B-Instruct-LoRA_HF.pt` | `b2e8e78d19c71d2ca674903586d53ca171c33a539956ee37c1c61f44a5e01f1d` | ✅ |
| ZH-LoRA | dev_seen (V=78) | `MHC_zh/dev_seen_…-LoRA_HF.pt` | `4c07af75098391c999013e1cf6fb7ffe8fac29546d9ce329d51004a37e4f5d3c` | ✅ |
| ZH-LoRA | test_seen | `MHC_zh/test_seen_…-LoRA_HF.pt` | `4e107bf65f58745a5749499a76c89d4d2695c869003582b55f5e1db91a5d2af2` | ✅ (verdict-only) |
| HateMM-curric (13241) | train | `HateMM/train_…-LoRA-curric_HF.pt` | `5e80f39327a743144067857e6f8c9f0c909e3131bdc13bcb063be6abc333e7cf` | ✅ |
| HateMM-curric | dev_seen (V=107) | `HateMM/dev_seen_…-LoRA-curric_HF.pt` | `46ee4fd9fcaec80b7859a5e4c18b76e84b4020fa242ced802f289f790e4d7cb0` | ✅ |
| HateMM-curric | test_seen | `HateMM/test_seen_…-LoRA-curric_HF.pt` | `b50ae4ecb077a8334d13ee9e60147c16d4566ab5f0870dd2228958bf97ca1ccb` | ✅ (verdict-only) |

Cache keys = `{ids, img_feats[V,3584], text_feats[V,3584], labels[V]}`. **One file per split — NOT per seed.**
The LoRA adapter is a single extraction; per-seed variation comes ONLY from the enc3s head.

### 2.2 Deployed floor HEAD ckpts (the object the vote actually indexes) — **MISSING, all 6**
| floor | seed 0 | seed 1 | seed 2 |
|---|---|---|---|
| ZH-LoRA (13150), `logging/Retrieval/MHC_zh/.../…LoRA_HF/ckpt/` | ❌ empty | ❌ empty | ❌ empty |
| HateMM-curric (13241), `logging/Retrieval/HateMM/RAC_video_lora_curric/.../ckpt/` | ❌ empty | ❌ empty | ❌ empty |

- Both floors' `ckpt/` dirs exist but contain **0 files** (`best_model_*`, `last_model_*`, `epoch_model_*` all
  purged in disk cleanup). No saved head-embedding pickle or `*_retrieval_logging_dict.pkl` survives either
  (0 pkls in both group trees).
- **Only trainlog anchors survive** (the would-be parity targets):
  - ZH-LoRA e29 **dev** acc `[0.8462, 0.8590, 0.8462]` (s0/1/2); **test** `[0.8456, 0.8389, 0.8523]`.
  - HateMM-curric e29 **dev** acc `[0.8505, 0.8224, 0.8411]`; **test** `[0.8791, 0.8791, 0.8791]`.
- Head ckpts that DO exist (`refine-logs/router_ckpt_snapshot/`): **frozen** CLIP/Qwen × {HateMM, MHC=EN} ×
  s0/1/2 e29 — **none is a deployed floor** (frozen ≠ LoRA/curric), and **there is NO ZH head ckpt of any
  kind** (snapshot has MHC=EN only). The `RAC_video_ncafam` dirs hold **NCA/mixup ARM** ckpts, not the
  triplet floor.

**LITSWEEP3 §4's load-bearing premise — "~$0 on the already-banked keys … reuse the F63 keys" — is FALSE.**
Only the *raw pre-head features* were banked. The **keys the vote uses (head embeddings) were never persisted,
and the heads are gone.**

---

## 3. THE $0-FAITHFULNESS BLOCKER (why the two obvious paths both fail a duty)

**Path A — faithful (head-space) curation.** Correct object (edits the DEPLOYED bank), and multi-seed
(3 heads/floor). **But NOT $0:** requires re-minting 6 head ckpts (2 floors × 3 seeds, 30-epoch enc3s head on
cached feats) on GPU/SLURM, **bit-exact (G-repro) to the §2.2 anchors**, before any curation. The head is
tiny (~0.3 GPU-h for all 6, = the NCA-family footprint), but it is a training run through SLURM, not a CPU
pregate. After the re-mint, selection + pregate + a formal 3-seed **test-touch** verdict are all CPU.

**Path B — $0 CPU (raw fused key, LP-gate 7168-d `L2(img)⊕L2(txt)` renorm).** Runs today with no GPU. **But
two duties fail:**
1. **Not the deployed decision object.** The head *reshapes* the space (that is what training buys); a prune
   selected by influence on the raw fused key is not the prune that helps the deployed head-embedding vote.
   The parity would be vs a raw-key one-hop baseline computed in-harness, **not** vs a banked floor number —
   so the pregate would not bind the floor. (This is exactly F63's substrate.)
2. **Seed-independent → single-draw.** The raw key has no seed axis (one extraction), so K-CUR-2 sign-across-
   seeds is **unrunnable** and the whole test collapses to one draw — **precisely the failure class that got
   archive-as-key WITHDRAWN** ("multi-seed + sha1 audit ⇒ selection artifacts"). The tasking's explicit
   immunity requirement ("multi-seed discipline, no dev cherry-picking") is **not satisfiable** on the $0
   object.

There is no third object: no per-seed head-embedding cache and no retrieval-logging pkl survive (§2.2).

---

## 4. WALLS PRICED (honest)

- **F63 — the directly-binding headwind (`7be6e3f`).** The deployed one-hop top-20 vote already reads the
  extractable signal at the **1-hop-separable ceiling**, and its permutation-null center is **POSITIVE**:
  graph operations on this bank help *random* labels more than *real* ones. Curation is a **node-removal graph
  edit** on that same graph → same wall. Pruning a 1-hop-separable, ~600–1100-node, label-noisy bank risks
  removing the **clean-hard** memories (Late-Stopping's exact warning) and *lowering* the vote. The **trained
  head space is even more 1-hop-separable** than the raw-fused space F63 measured, so the faithful version is
  *more* wall-bound, not less.
- **W2-E prototype memory — ALREADY BANKED-DEAD (`954b0cb`, NO-GO pre-ceremony).** "Prototype-as-key /
  mode-local kNN over the banked pooled vectors: **zero new information over the flat kNN**, a strict
  *coarsening* of full kNN over ~600 exemplars, performance prior near-zero below the D3 noise floor, fails
  D7." The candidate's **prototype-select arm is this cell** — one of its three named operators is
  pre-refuted.
- **Wall-A / F66 (`a6e41f8`) — price it, but don't overclaim it.** F66's arithmetic (91–98% of oracle
  headroom is per-item-selection-only; symmetric slice +0.001–0.006) was computed on the **CLIP W2-B
  per-segment SELECTION oracle** (sub-clip aggregation on the CLIP caches). Bank curation is a **different
  operator (node removal) in a different (head-embedding, whole-video) space**, so F66 does **not** transfer
  as a *literal numeric cap*. What transfers is the **mechanism lesson** (a symmetric global operator cannot
  reach per-item-selection headroom) — and there F63 is the concrete, directly-measured instance. Net: treat
  the convertible magnitude as "≈F63-small," not "provably ≤+0.006."
- **Wall-C anti-alignment.** ZH test climbs to ep29 while dev plateaus (F45). "Remove the noisy tail"
  curation shrinks the late-epoch memory footprint — anti-aligned with ZH's late climb (the SWA/F62 kill
  shape). Curation cannot add dev items and cannot move the dev-argmax that selects the checkpoint (AUG kill
  F60); its only ZH lever is indirect val-sel *variance reduction* on a fixed head.
- **Withdrawn archive-as-key + the only banked positive.** Bank-as-key accuracy claims were **WITHDRAWN** as
  selection artifacts (multi-seed + sha1). The lone banked positive is the **human 2-entry EN deletion** — a
  hand-audited **micro-edit (2 entries)**, human-in-the-loop only; the **AUTO two-vote MLLM repair was
  NEGATIVE** (AND-rule C−A=0, embedding-only over-deletes; `archive-auto-repair` dead-list). A *learned mass
  prune* is the auto-generalisation those two data points specifically warn against.
- **Noise vs target.** Head-seed noise ±0.014; P(≥+3 on any dataset) ~1–2%. The realistic best case is ZH
  val-sel *stabilisation* or a thin mF1 rebalance — not a new +3.

---

## 5. IS THE FULL CANDIDATE $0 END-TO-END? — NO (as-is); "$0-after-a-small-GPU-recovery" (faithful)

- **As banked today:** **NOT $0** and **not faithfully runnable.** The $0 object (raw key) is unfaithful +
  single-draw (§3B); the faithful object (head space) has **no ckpts** (§2.2).
- **Faithful minimum:** one-time **~0.3 GPU-h SLURM re-mint of 6 heads** (bit-exact to §2.2 anchors — the
  re-mint IS the K-CUR-0 parity), **then** selection + pregate + a formal **3-seed test-touch** verdict are
  **all CPU**. So the candidate could still be *among the cheapest ceremonies* (one tiny GPU step, then
  CPU-only through verdict) — but the tasking's "$0 through formal verdict / cheapest ceremony ever" framing
  rested on the **false premise that the heads/keys were banked.** They were not.

---

## 6. RECOMMENDATION — **PARK** (prior + one-line escalation)

**PARK.** Honest prior (re-priced **down** from LITSWEEP3 §4's 8–12% because the $0 faithful path evaporates
and the $0-runnable path is unfaithful + single-draw + F63-substrate):
- **ZH val-sel stabilisation ≥+1 (stable 3/3): ~5–8%.**
- **+3 on any dataset: ~1%.**
- **Paper-value: unchanged-high** (PV-1 pillar-4 valuation curve) — but achievable via the §7 diagnostic
  without a floor-binding claim.

**Escalation rule (if the orchestrator wants it closed on the record, not parked):** the *only* admissible
faithful route is **head-recovery-first** — a single prereg to re-mint the 6 floor heads (bit-exact G-repro
vs §2.2 anchors) as the parity gate, after which the entire curation ceremony (selection → 3-seed pregate →
test-touch verdict) is CPU-only. Given F63 + W2-E + Wall-C, the recon's expectation for that route is a
KILL; spend the GPU only if the pillar-4 paper sentence ("we auto-generalised the human deletion and it does
/ does not move the deployed vote, 3-seed") is judged worth ~0.3 GPU-h.

---

## 7. OPTIONAL $0 CPU DIAGNOSTIC (paper-value ONLY — **non-binding on the floor**; run only if PV-1 wanted)

If the orchestrator wants the PV-1 valuation curve at zero cost **and accepts it cannot verdict the deployed
floor** (raw-fused key, single-draw), here is the hand-off-able spec. It is **not** the LEAD pregate.

- **Object:** raw fused key `z = L2( L2(img) ⊕ L2(text) )` (7168-d), reusing `lp_gate.py:fused_key` +
  `knn` + `read_score` (rank-weighted signed-cosine top-20, `router_gate:73-79` verbatim). Datasets: ZH-LoRA,
  HateMM-curric (train+dev only; test **never** opened — assert `split ∈ {train,dev_seen}`).
- **Selection (train-label-only, no dev sweep):** per train row compute (a) **LOO-kNN influence** = Δ
  train-LOO top-20 vote acc on its removal; (b) **Data-OOB** (Kwon-Zou ICML'23). Prune the lowest-value rows
  at a **pre-fixed retention grid {95, 90, 80, 70}%**; the mF1 arm additionally **class-balances** the
  retained bank. Retention level is **pre-fixed / chosen by train-LOO**, never by the dev score. Freeze the
  kept-index list (sha256) **before** any dev read.
- **Score (dev, reporting only):** re-run the vote over the pruned bank; report the full grid transparently.
- **Kill-switches (house style):**
  - **K-CUR-0 (machinery parity, $0):** retention=100% must reproduce the intact-bank dev decisions
    **bit-exact** (Δ=0.0000, 0 flips) — the α→0 analogue.
  - **K-CUR-1 (perm-null gate):** best pruned-bank Δ must beat the **random-prune-at-same-retention** null
    (1000×, same retention); if inside the null (F63's positive-center warning) ⇒ **auto-KILL**.
  - **K-CUR-2 (sign consistency):** **UNRUNNABLE on this object (seed-independent)** — this is the load-
    bearing reason the diagnostic cannot verdict the floor; record it as VOID, not PASS.
  - **K-CUR-3 (over-deletion tripwire):** if the best cell needs >25% prune, flag the archive-repair
    over-deletion pattern (C−A).
- **Binding-close:** even a "positive" here is **paper-value only** and must carry the K-CUR-2-VOID +
  raw-key-not-head banner; a floor-binding number requires the §6 head-recovery route.

---

## 8. PROVENANCE & DISCIPLINE

- Mechanism: `src/run_rac.py:813-830`, `src/model/evaluate_rac.py:349-474`, `src/utils/metrics.py:214-320`,
  `src/model/classifier.py:115-148`. Machinery precedents: `scripts/analysis/cross_channel_router_gate.py`
  (deployed-vote reproduction, head path), `scripts/analysis/lp_gate.py` (F63), `scripts/analysis/isr_pregate.py`
  (β-decomposition template).
- Caches/ckpts: sha256 in §2.1; head-ckpt absence verified by `ls .../ckpt/` (0 files) + `find` (0 head `.pt`,
  0 pkl) for both floors; anchors from `slurm/logs/enc3s_MHC_zh_…LoRA_HF_seed{0,1,2}_13150.trainlog` and
  `slurm/logs/enc3s_HateMM_…LoRA-curric_HF_seed{0,1,2}_13241.trainlog`.
- Walls: F63 `LP_GATE_RECORD.md` (`7be6e3f`); F66 `ISR_PREGATE_RECORD.md` (`a6e41f8`); W2-E
  `W2E_FORENSIC_RECON.md` (`954b0cb`); F45 `B3_ZH_LORA_DECOMPOSITION.md`; F60 AUG; F62 SWA; archive-auto-repair
  + human-2-entry-EN positive + banned `kNN-vote-pool expansion` from `state/directions_tried.json`.
- **ZERO GPU / SLURM / Modal spent. No held-out test metric read or produced. No `state/`, prereg, config,
  research-wiki, or frozen artifact mutated. Committed on `main`, not pushed.**
