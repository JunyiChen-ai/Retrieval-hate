# LITSWEEP-5 / S3 — Adversarial Completeness Audit (literature sweep round 5 of 5, LAST)

**Agent:** litsweep-5 S3 (adversarial completeness auditor — the F61 job, redone for round-5-redteam's successor state).
**Date:** 2026-07-25 · **Discipline:** ZERO GPU / SLURM / Modal / training / test-touch. CPU reading + code grep + WebSearch. `autoresearch/state/` untouched. One local commit, no push.
**Charge:** attack "everything in-box has been tried." F61 did this on 2026-07-19, found 6 real gaps; **all 6 were then measured dead** (F62b SWA, F63 LP, F64 audio-Whisper, F65 vision-LoRA, F66 ISR, F67 frame16). Since F61 the campaign also ran F68's 6-cell batch (F69-F73, 0-for-6), litsweep2's 2 (F75 NCA dead, F76 resolution park), and litsweep3's batch-4 (F78 curation park, F79 ELR park, F80 ZH-prompt dead). **17 consecutive door-closers post-F61, all null/park.** This audit asks whether F61's refutation still holds, what remains, and — decisively — whether the 2026-07-25 oracle-queue fallback is viable.

---

## 0. BOTTOM LINE (read first)

1. **F61's refutation still literally holds:** "we measured everything" is still false at the enumeration level — **5 genuinely never-measured in-box cells + 7 measured-under-relaxed-conditions cells + 2 fresh ban-scope letter-overreaches** survive an adversarial read. BUT **every one is D7-novelty-dead as a performance lever, F66-arithmetic-capped near zero, or park-priced <3% for the goal.** Exactly F61's honest framing: this refutes *"space exhausted,"* not *"goal reachable."*
2. **Fresh-2026 delta (since litsweep2's Jul-2026 cutoff) = 0 transplantable levers.** The Jul-2026 arXiv window surfaces only isomorphic-to-dead families (graph-diffusion=F63, kNN-MoE-routing=F47, prototype-head=W2-E, training-free=zero-training kills, localization=P6) and non-transplantable pretraining architectures (LeVLJEPA). Corroborates litsweep2-fresh's "no 2026 paper beats us in-box."
3. **LOAD-BEARING NEW FINDING — the oracle-queue fallback is internally contradicted by its own successor findings.** The 2026-07-25 ruling blesses *"trained selector/reshaper on train labels only (F66 binds only fixed-map φ₀)."* But **both** blessed attack classes are already measured dead by findings that post-date the ruling's evidence base: (i) a trained **selector** on train labels = F47's train-supervised source, degenerate by memorization (kNN LOO 0.998, train-disagreement "Qwen-correct" = 0/109); (ii) a trained **symmetric reshaper** on train labels = F75's object (NCA/SupCon/mixup, 0/8 formal) and F66 caps it at +0.001-0.006. **The oracle-queue's first move is pre-killed unless a NEW GPU-gated channel passes the F49 alignment gate (>0.663) — and no $0 in-box channel remains to feed it.**
4. **The only goal-live upside lives behind user gates, ranked:** ZH val-sel retirement (STRENGTHENED, and it is a *protocol ruling* that makes ZH a 2nd passing dataset, not an operator) ≫ Molmo2-8B (HateMM-only ceiling, WEAKENED by F65) > MNTP stage-2 (highest novelty, ~8-12% post-crater) > door-closers F78>F79>F76 (<3%, paper-value) > multi-prompt (~0).

**Verdict line:** *Enumeration is NOT exhausted (≥7 live never-/under-measured cells + 2 ban letter-overreaches), but NONE carries a defensible ≥+3-on-≥2-datasets prior — all are D7-dead, F66-capped, or <3%; goal upside is user-gated only, and the oracle-queue fallback is contradicted by F47/F75/F77 and needs a new GPU-gated F49 channel it does not have.*

---

## 1. FULL DECISION-SURFACE ENUMERATION × dead/parked/banned cross-tab

Pipeline (code-confirmed): LoRA/frozen Qwen2.5-VL dual-stream (8f, `max_pixels=151200`, img mean-pool / text last-token, fixed English prompts) → RGCL align head (`fusion_mode∈{concat,align,cross}` @ `classifier.py:85-142`, `--loss∈{naive,triplet,contrastive}`+hybrid-BCE @ `run_rac.py:141`, FAISS mining) → top-20 rank-weighted signed-cos kNN vote over own-train memory. Every choice below is cross-tabbed against the dead/parked/banned ledger.

| Decision axis | Current setting | Covering verdict | Status |
|---|---|---|---|
| Frame **count** | 8 | F67 KILL (16f); 32f dominated | dead / (b) 32f |
| Frame **selection policy** | uniform linspace | F67 closed *count* not *policy* | **(a) never-measured** |
| Spatial **resolution** | 151200px cap | F76 PARK (recon only; door-closer unrun) | **(b) parked** |
| **Prompt** structure | fixed English IMG/TEXT | F70 KILL (layer/token/one-word); F80 KILL (Chinese) | dead |
| **Multi-prompt** ensemble | none | never run; user micro-ruling | **(a) never-measured** |
| **Layers** | final | F70 KILL (L24) | dead |
| **Readout/pooling** | mean-pool / last-token | F70 KILL; F72 crater (bidir) | dead |
| Resolution × **detail-preserving readout** (joint) | — | F76 parked resolution *alone*; joint never posed | **(a) never-measured** |
| **Fusion** — fixed composition/reweight | Hadamard align | F50 KILL (fixed comps/reweights/temps) | dead |
| **Fusion** — *trained* mode swap (concat/cross/x-attn) | align only | F50 bans *fixed* comps, not *trained* heads | **(a) never-measured (c-challenge #1)** |
| **Loss** — triplet+BCE vs NCA/SupCon/mixup | triplet+BCE | F75 KILL (family) | dead |
| **Loss** — ArcFace/CosFace/ProxyNCA++ angular | — | arguably outside F75 letter | **(a)/(c-challenge #2)** |
| **Loss** — ELR/noise-robust additive | — | F79 PARK (recon; probe unrun) | **(b) parked** |
| **Mining** — frozen-space hard-neg source | FAISS online | C3geo/R3/C5 dead | dead |
| **Memory** — bank curation (train-label) | full own-train | F78 PARK (recon; multi-seed pregate unrun) | **(b) parked** |
| **Memory** — prototype / pool-expansion | none | W2-E dead; pseudo-label pool-expansion banned | dead/banned |
| **Vote** — multi-hop diffusion | 1-hop top-20 | F63 KILL (LP) | dead |
| **k / weighting / metric** | top-20 rank-wt signed-cos | §0-fact-2 near-flat, direction-inconsistent | priced-down, not formally swept |
| **Protocol** — val-sel vs final-epoch | both reported | user-gated (ZH retirement) | **user-gate** |
| **Encoder** identity | CLIP/Qwen2.5-VL-7B | swap = positive (HateMM); Molmo2/Qwen3-VL download-gated | **user-gate** |
| **LoRA recipe** knobs (DoRA/rsLoRA/PiSSA/rank/epochs) | r16/α32/3ep/7-mod | F51 closes *object* not *recipe*; code-present `finetuning_args.py:99-115` | **(a) never-measured** |
| **Vision-tower** unfreeze | frozen | F65 CLOSED | dead |
| **Audio** stream — Whisper-encoder | none | F64 KILL | dead |
| **Audio** — general-audio AST/BEATs | none | download-gated; prior lowered by F64 | **(b) under-relaxed** |
| **Adapted (LoRA) 32B/72B** scale | 7B only | B2 measured *frozen* only; adapted-scale Law-IV-open | **(b) under-relaxed, download-gated** |
| **OCR** / title channel | none | user-vetoed / absent-from-source | banned/n-a |

### (a) Genuinely never-measured (no binding verdict, no covering ban letter) — **5 cells**
1. **Trained fusion-mode swap** (concat / cross-bmm / small cross-attention head, end-to-end with triplet+BCE). Code-present (`classifier.py:85-142`). D7-DEAD (generic head engineering, no MLLM role); ≈$0 (head-only, cached). *See ban-challenge #1.*
2. **LoRA recipe knobs** (DoRA/rsLoRA/PiSSA/rank/epochs/target-set). Code-present (`finetuning_args.py:99-115`). D7-DEAD; GPU (~3-3.5h × grid, no $0 gate). EN bottleneck (frozen vision tower) untouched by any recipe.
3. **Frame-selection policy** (content-aware/diversity keyframe @ fixed 8f). litsweep2-INPUT priced ~0 (short videos, 8f near-optimal, only 7-22% near-dup slack, query-agnostic encoder). *(S1 sibling may cover the motion/order variant.)*
4. **Multi-prompt single-model embedding ensemble.** Never run; user micro-ruling (grazes cross-seed-veto *spirit*, not letter). Predicted ~0 (D1: reshuffles same frozen rep; readout F70 + prompt-language F80 both dead nearby).
5. **Resolution × detail-preserving-readout JOINT cell.** litsweep2-INPUT flagged that resolution's payoff is *conditional on a detail-preserving readout* (mean-pool attenuates); F76 parked resolution-alone; the joint cell was never posed. GPU re-extraction.

### (b) Measured-but-under-relaxed-conditions (a stronger realization could exceed the measured null) — **7 cells**
1. **General-audio encoders (AST/BEATs).** F64 killed the *Whisper-encoder* realization on all 3 datasets; ban scope explicitly *"does NOT close general-audio encoders (download-gated), prior now lowered."* Under-measured, download-gated, prior lowered.
2. **Independent-segment frame-local Qwen** (F61-GAP-2). F66 measured the symmetric slice on *CLIP* subclip caches + β-decomposition; frame-local *Qwen* per-segment (F35-immune) was never extracted — but F66's β-arithmetic ("symmetric legal slice +0.001-0.006, 91-98% selection-locked") is representation-agnostic and covers it *by arithmetic*, not by extraction. Under-relaxed but arithmetically pre-priced.
3. **Frame budget 32f.** F67 measured 8→16 (KILL); 32f "not run (dominated by 16f null prior)."
4. **Resolution door-closer.** F76 recon measured the *premise* (corrected HateMM 2.71x, not litsweep2's 6.5x) and priced <3%; the actual 409920px re-extract + 3-seed head was never run.
5. **Memory-bank curation.** F78 recon priced ~1% and found the "$0 banked keys" premise false (floor head ckpts deleted); the faithful multi-seed pregate (needs 0.3 GPU-h re-mint) was never run.
6. **ELR noise-robust head.** F79 recon quantified the noise proxy (13-17% upper bound, boundary-dominated) and priced ~1-2%; the actual ELR probe (0.16 GPU-h, expected KILL) was never run.
7. **Adapted (LoRA) 32B/72B scale** (F61-GAP-6). B2 measured *frozen* 32B (scale regresses); Structural-Law-IV (convertibility runs through *adaptation*, not identity) leaves the *adapted*-larger-scale cell formally open. Download-gated (prior downloads failed, F8); HIGH cost.

### (c) Dead-with-ban-scope-arguably-over-reaching — **2 fresh challenges** (§2)
Plus a secondary (F67 count→policy, already listed as (a)#3). **Meta-observation:** F61's own 6 ban-scope challenges (audio, temporal-independent-segment, LP-escapes-F46/47, SWA-vs-veto, ISR, F49-catch-22) are now **all closed** — measured dead (F64/F66/F63/F62b) or doctrinally patched (the F47 $0-gate replaces the F49 arithmetic pre-kill, per REDTEAM GAP-4). The ban-scope-overreach *attack surface itself* is nearly exhausted; the 2 below are the only fresh letter-overreaches I can construct, and both are D7-dead or F66-capped.

---

## 2. BAN-SCOPE CHALLENGES (≥2, F61-style, adversarial toward my own challenges)

### Challenge #1 — F50 fusion ban over-reaches from *fixed compositions* to *trained fusion heads*
**Ban (verbatim, F50):** *"do not re-propose **fixed compositions, reweights, or per-modality temperatures** over banked frozen features; conversion requires adaptation (F45) or a new information source with alignment>0.663."*
**Over-reach:** a **trained** fusion head (`fusion_mode='concat'` or `'cross'`, or a small cross-attention block optimized end-to-end with triplet+BCE) is **none of those three named objects** — it is a nonlinear trained operator, not a fixed composition. F50 measured only (A1) a scalar within-Qwen reweight (rotation at every w) and (A2) a *fixed* cross-encoder CLIP-img+Qwen-text concat scored by kNN/oracle. Neither is a trained fusion-mode swap. Code confirms `concat`/`cross` exist and are unswept on video (`classifier.py:85-142`; no banked sweep in F1-F80).
**Adversarial self-rebuttal (why it is still not a goal-live cell):** (i) **D7-DEAD** — head-architecture engineering carries no MLLM-novelty role; a gain is a performance/ablation row only. (ii) The deployed align-MLP is *already* a nonlinear trained head and defines the floor, so a different nonlinear fusion has thin headroom above it. (iii) F66 caps the symmetric-conversion story regardless of fusion form. **Verdict: genuine letter-overreach, $0 to close, but D7-dead — a door-closer, not a goal cell.**

### Challenge #2 — F75 loss-family ban over-reaches from {vote-consistent, contrastive, mixup} to *angular-margin / proxy* losses
**Ban (verbatim, F75):** *"head-loss swaps of the triplet+BCE hybrid toward **vote-consistent (NCA/soft-kNN), contrastive (SupCon), or mixup-BCE** objectives ... tau/alpha retunes = tactics, banned."*
**Over-reach:** **ArcFace/CosFace angular-margin** (Deng CVPR19 arXiv:1801.07698; Wang CVPR18) is a proxy-based margin-*softmax classification* loss against class centers — not a vote-consistent LOO surrogate (NCA), not a pairwise/log-softmax contrastive (SupCon), not mixup. litsweep2-HEAD §2.5 argues it is "non-isomorphic to triplet (proxy-vs-pair, global-angular-vs-hardest-pair)" and *better geometrically matched to the cosine-weighted vote* than the current cosine-margin triplet. It is arguably outside the F75 letter.
**Adversarial self-rebuttal:** (i) **F66 caps it** — ArcFace is a *symmetric* embedding-geometry operator; the convertible ZH/EN headroom is 91-98% selection-only, so it can recover at most +0.001-0.006. (ii) F75 is explicitly *"the first measured negative for trained-reshaping-unlocks-oracle-headroom"* — its *mechanism* (symmetric reshaping doesn't convert the selection-locked headroom) generalizes past its named-loss letter, and ArcFace is a symmetric reshaper. (iii) Angular-margin gains are documented to shrink toward **binary** classification (2-class = our regime), and its design target is many-class open-set face-ID. **Verdict: genuine letter-overreach, but the F75 *mechanism* + F66 *arithmetic* price it near-zero — closing it is a door-closer, not a goal cell.**

*(Secondary, already counted: F67's ban names "denser frame sampling (16f)"; frame-selection *policy* @8f is uncovered — but litsweep2 priced it ~0. A letter-gap, not a goal cell.)*

---

## 3. FRESH-2026 DELTA (since litsweep2's ~Jul-2026 cutoff) — verified web-surfaced, **0 transplantable levers**

WebSearch 2026-07-25 across: multimodal hate/harm video · frozen-VLM feature classification heads · kNN-augmented classifiers · small-n multimodal fine-tuning. arXiv IDs from result metadata:

| Paper | arXiv | Window | Relevance / transplant verdict |
|---|---|---|---|
| MultiHateGNN — Dual-Stream GNN for hate video | 2509.13515 | Sep-2025 (in litsweep2 window) | Graph-based classification = **isomorphic to F63 LP** (graph-diffusion over kNN memory, KILLED all 3 datasets). Not transplantable. |
| Training-Free Interpretable Hateful Video Detection | 2601.15115 | Jan-2026 (in window) | Training-free = the **zero-training family** we killed (W2-B/S2S/ISR); not a trained-head accuracy lever. Not transplantable. |
| Routing by Analogy — kNN-MoE expert assignment | 2601.02144 | Jan-2026 (in window) | kNN-confidence-driven expert mixing = **F47 verbatim** (vote-margin/purity routing, nulled at 3 supervision sources). Not transplantable. |
| Supervised Classification Heads as Semantic Prototypes | 2605.22484 | May-2026 (in window) | Class-weight-as-prototype = **W2-E prototype family** (dead) + D7-dead. Not transplantable. |
| MemVerse / Memory-Modular Classification | 2512.03627 / 2504.06021 | Dec-2025 / Apr-2025 (already in F68 ledger) | Editable-memory pillar-4 tie-in only; AUTO variant already negative. No new lever. |
| **LeVLJEPA** — end-to-end VL pretraining w/o negatives | **2607.00784** | **Jul-2026 (fresh)** | From-scratch JEPA **pretraining architecture**; not an in-box transplantable head/loss/channel. Download+pretrain-gated at best. |
| Audio-Visual Event Recognition — KD + INT8 quant | **2607.16980** | **Jul-2026 (fresh)** | Efficiency/compression, not an accuracy lever; audio-visual = **F64-dead audio axis**. Not transplantable. |

**Fresh-window count (2607.xxxxx = genuinely since ~Jul-10): 2 papers, 0 transplantable.** Both are architecture/efficiency, and one lives on the dead audio axis. Every *relevant* method surfaced is isomorphic to an existing kill. This is the third independent sweep (F68/F74 prior) to conclude the 2026 literature offers no in-box lever.

---

## 4. ROADS NOT TAKEN

### 4(i) — User-gated levers: CURRENT expected value (re-priced against everything measured since each was gated)

| Rank | Lever | State-change since gating | Current EV | Note |
|---|---|---|---|---|
| **1** | **ZH val-sel protocol retirement** | **STRENGTHENED** (F77/L2): ZH final-epoch already 3/3 PASS +0.0313/+0.0453; wall = 78-dev *selection* noise, NOT representation (LoRA text-AUC 0.925; Qwen native-Chinese SOTA). | **HIGHEST.** Retiring val-sel makes ZH a one-protocol pass → with HateMM (F53, both protocols) that is **2 datasets under a single consistent final-epoch protocol.** This is the single highest-value move in the entire remaining space, and it is a **protocol ruling, not an operator or a GPU spend.** | Whether "≥2 datasets under final-epoch-only" satisfies the user's goal is the user's call; the *evidence* now favors retirement (78-dev underpowered, test climbs past dev-plateau). |
| **2** | **Molmo2-8B encoder download** (SigLIP2 tower) | **WEAKENED** by F65: vision-unfreeze *moved* EN image AUC +0.032 but converted **zero** at the head — so even a better EN image stream (Molmo2's whole selling point) is now empirically shown to convert nothing on the label-limited datasets. | Moderate P(≥+1 **HateMM-only**); **P(≥2-dataset) ≈ 0** (F44 label-limit unchanged; F65 lowered the EN-conversion prior). Only representation-class lever with a non-trivial prior (the class that cleared +3 once), but its ceiling is a HateMM re-nudge. | Worth a cloud-triage frozen probe reading HateMM headroom + MHC-EN image-AUC **iff** user green-lights a download; closes with a B2-shaped null if EN image stays near-chance. |
| **3** | **MNTP stage-2** (LLM2Vec bidirectional recovery) | measured-adjacent (F72 crater): naked mask-flip = -10..-14pt (Llama-pattern), so stage-2 MNTP is the *only* way the bidirectional axis could pay. | **Low-moderate ~8-12%** (post-crater honest prior; LLM2Vec precedent: MNTP recovers AND *can* exceed causal — but our +3-over-causal target sits *above* mere recovery). **Highest D7-novelty** of any live lever; 2-4 GPU-h/ds. | The one lever with both a real novelty story and a non-trivial prior; target above recovery is the risk. |
| **4** | **Door-closer F78 curation** | F78 PARK (~1%) | Low perf; **highest paper-value** (pillar-4 auditable memory; automates the banked human-2-entry-EN positive). $0-after-0.3-GPU-h re-mint. | Value = hypothesis-closure + paper, not goal. |
| **5** | **Door-closer F79 ELR** | F79 PARK (~1-2%) | Low; closes the mined-pair-noise hypothesis on the record. 0.16 GPU-h, expected KILL. | Wall-C anti-aligned (ZH test climbs late; ELR biases early). |
| **6** | **Door-closer F76 resolution** | F76 PARK (<3%, premise corrected 6.5x→2.71x HateMM) | Low; virgin axis but the only clean-converting dataset (HateMM) is already near-native, and mean-pool attenuates. GPU 1h re-extract. | Best run *jointly* with detail-preserving readout ((a)#5) or not at all. |
| **7** | **Multi-prompt ensembling** | never run | ~0 predicted (D1; F70+F80 dead neighbors) | Needs GPU re-extract (no banked per-prompt caches) → not worth it. |
| (8) | Adapted 32B/72B scale (GAP-6) | B2 frozen-regression prior; Law-IV-open | Low-mod, download+HIGH-cost | Prior-justification over-scope, not a silently-closed cell. |

### 4(ii) — Oracle-queue viability after F77/L1, and the F49-channel enumeration

**The oracle-queue ruling (2026-07-25, progress.json):** *"attack directions in descending oracle-headroom order ... Legal attack on selection-locked pools = trained selector/reshaper on train labels only (F66 binds only fixed-map φ₀)."* Ruling was written at lit-round-count 3 — **before** F75/F77/L1 sharpened the walls.

**The contradiction (load-bearing):** the ruling's two blessed attack classes are both already measured dead by post-ruling-evidence findings:
- **Trained SELECTOR on train labels** = F47's train-supervised source. **DEAD:** the deployed kNN vote *memorizes* train (CLIP LOO 0.998), so the train-time "which-operator-errs" target is degenerate — train-disagreement "Qwen-correct" = **0/109, 0/102, 0/92**, and that train base rate is the *inverse* of the ~0.55 test base rate (L1 §0, F47 §3.2). Training labels **cannot** supervise the test-time selection decision in this pipeline — a data-generating-process obstacle upstream of any selector capacity.
- **Trained symmetric RESHAPER on train labels** = F75's object (NCA/soft-kNN/SupCon/mixup, 0/8 formal, 7/8 KS-dead) + F66 caps symmetric reshaping at +0.001-0.006. **DEAD/capped.**

So the oracle-queue's first move is pre-killed **unless** a genuinely-new input channel passes the F49 alignment gate (>0.663 with the oracle routing decision) — per F77/L1 the *only* escape.

**Enumeration of in-box channels for the F49 >0.663 gate (Duty 3-ii core question):**

| Candidate channel | Alignment-tested? | $0 in-box? | Verdict |
|---|---|---|---|
| Vote margins / purity / sub-votes / confidence-diff / transcript stats | **YES** (F47) | yes | nulled, GBM+linear; banked-derivable, cannot re-feed |
| MLLM modality-locus judgment | **YES** (F49) | yes | 0.588 < 0.663; DEAD |
| Resolution-boosted per-image detail signal | **NO** | **no** (F76 GPU re-extract) | new pixels, but mean-pool attenuates; prior <3% |
| TTA-consistency variance (K-view re-encode) | **NO** | **no** (GPU K-view re-extract) | L1 predicts <0.663 (Ashukha redundancy: TTA-var ≈ softmax confidence = the nulled margin) |
| General-audio AST/BEATs per-item confidence | **NO** | **no** (download-gated) | Whisper already null (F64); prior lowered |
| Independent-segment frame-local Qwen confidence | **NO** | **no** (GPU re-extract) | F66 β-decomp: symmetric-only slice; per-item selection still law-III banned |
| Vision-adapted image per-item confidence | **NO** | yes (F65 caches exist) | F65: image MOVED, zero head conversion — routing-alignment untested but conversion already null |

**Conclusion:** **No $0 in-box channel remains to feed the F49 gate.** Every never-alignment-tested candidate is GPU-gated re-extraction (or download-gated), each with a prior <3-15%, and **each additionally hits the F47 train-non-transferability wall** — even a perfectly-aligned new input cannot be *trained into a selector* because the train split is memorized. The one arguably-$0 candidate (vision-adapted confidence, F65 caches on disk) already has its *conversion* measured null. **The oracle-queue, as ruled, has no viable in-box first move; its viability depends entirely on a user relaxation (GPU re-extraction budget for resolution/TTA/general-audio, or a download) — and even then the train-non-transferability obstacle is unresolved.**

---

## 5. VERDICT (adversarial toward BOTH conclusions)

**Steelman "N live cells remain":** 5 genuinely never-measured cells + 7 under-relaxed cells + 2 fresh ban letter-overreaches are real; a pure completeness auditor MUST report them; the F50/F75 bans *do* over-reach on the letter; the trained-fusion and ArcFace cells are $0/cheap and unswept. F61 was right that "measured everything" is false, and it is *still* false.

**Steelman "exhaustion confirmed":** every one of those cells is (i) **D7-novelty-dead** as a performance lever (fusion-head, LoRA-recipe, key-map recipe), (ii) **F66-arithmetic-capped** near zero (ArcFace, any symmetric loss/curation/reshaper — 91-98% of headroom is selection-locked, legal slice +0.001-0.006), or (iii) **park-priced <3%** for the goal (resolution, curation, ELR, frame-policy, multi-prompt). The base rate of a post-F61 door-closer converting to the goal is now **0/17**. The goal (≥+3 acc on ≥2 datasets) is walled by MHC label-limits (EN dead at 5 levels) + F66 arithmetic + the HateMM ceiling — none of which any in-box operator can move, confirmed independently by F68, F74, F77, and now S3.

**Reconciliation (the honest verdict):**

> **Enumeration-level exhaustion is NOT literally confirmed — ≥7 never-/under-measured in-box cells and 2 ban-scope letter-overreaches survive an adversarial read — BUT every surviving cell is D7-novelty-dead, F66-arithmetic-capped, or park-priced <3% for the goal; NO cell carries a defensible ≥+3-on-≥2-datasets prior. The goal's remaining upside lives ONLY behind user gates, ranked: (1) ZH val-sel retirement [STRENGTHENED — a protocol ruling that makes ZH the 2nd passing dataset] ≫ (2) Molmo2-8B [HateMM-only, WEAKENED by F65] > (3) MNTP stage-2 [highest novelty, ~8-12%] > (4-6) door-closers F78>F79>F76 [<3%, paper-value] > (7) multi-prompt [~0]. The 2026-07-25 oracle-queue fallback is internally contradicted by F47/F75/F77 — its blessed train-label selector/reshaper attacks are already measured dead — and has no $0 in-box channel left to feed the F49 gate; it is viable only under a user GPU-re-extraction or download relaxation, and even then the F47 train-non-transferability wall is unresolved.**

---

## 6. PROVENANCE
- Ledger: `autoresearch/goal_mllm_plus3/state/directions_tried.json` (full dead[]/banned_constraints[]/positives_bank[]/user_rulings), `state/findings.jsonl` F61-F80, `state/progress.json` (oracle-queue ruling).
- F61 method + companions: `refine-logs/REDTEAM_UNTESTED_CELLS.md` (adb8bc2), `REDTEAM_BAN_SCOPE_AUDIT.md` (5dd23e4), `REDTEAM_EXTERNAL_FAMILIES.md`.
- Six prior shortlists: `LITSWEEP2_{HEAD_OBJECTIVES,INPUT_FIDELITY,FRESH_2026}.md`, `LITSWEEP3_{DATA_CENTRIC,SELECTOR_CONVERSION,ZH_SPECIFIC}.md`.
- Code facts verified this sweep: `src/model/classifier.py:85-142` (fusion_mode {concat,align,cross}); `src/run_rac.py:141-142` (--loss {naive,triplet,contrastive}); `RA-HMD/LLAMA-FACTORY-Ver202512/src/llamafactory/hparams/finetuning_args.py:99-115,539-543` (DoRA/rsLoRA/PiSSA + vision-freeze flags).
- Fresh-2026 (WebSearch 2026-07-25, verified arXiv IDs): 2509.13515, 2601.15115, 2601.02144, 2605.22484, 2512.03627, 2504.06021, 2607.00784, 2607.16980; angular-margin cites 1801.07698 (ArcFace, Deng CVPR19), Wang CVPR18 (CosFace).
- **Required statements:** ZERO GPU / SLURM / Modal / training / test-touch spent. No held-out test metric read or produced. No `state/`, prereg, config, `research-wiki/`, or frozen artifact mutated. Committed on `main`, not pushed.
