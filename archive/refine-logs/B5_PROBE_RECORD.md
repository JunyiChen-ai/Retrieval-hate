# B5 PROBE RECORD — operating-point conversion of the frozen-Qwen MHC-ZH ranking advantage

**Stage:** G0-cond PROBE STAGE ONLY (authorized `refine-logs/B5_PREREG_REVIEW.md` §5). The formal
single-submit stage is NOT authorized and was NOT run.
**Executor role:** raw numbers only, provenance-lined, **NO pass/fail interpretation** (verdict
processing is an independent fresh agent — review §5 condition 6).
**Date:** 2026-07-14.

---

## 0. Provenance & environment

- **Repo HEAD at execution:** `be30d87cc9a115253aae156feb3b9282f08a98e6`.
- **Working tree state:** the amended prereg + design + probe script + sbatches + this record are
  **untracked / uncommitted** (`git status --porcelain`: `?? research-wiki/experiments/exp-conv-zh-b5.md`,
  `?? refine-logs/B5_PROBE_DESIGN.md`, `?? scripts/analysis/b5_conv_probe.py`,
  `?? scripts/analysis/b5_conv_probe.sbatch`, `?? scripts/analysis/b5_conv_probe_cuda.sbatch`). Batch
  commit is deferred to after independent verdict review (per task instruction: do NOT commit here).
- **Amendments applied (r1, before the probe ran):** blocking A1 (per-protocol AND-eligibility
  kill-switch) + A2 (co-equal DEV-side G-repro anchor); non-blocking A3–A10. Confirmed in
  `research-wiki/experiments/exp-conv-zh-b5.md` §16 amendment log and mirrored in
  `refine-logs/B5_PROBE_DESIGN.md` (A1 §7 / A2 §4 / A3 §5 / A6 §8 / A7 §9 / A9 §1).
- **Conda env:** `HateVideo`. faiss = CPU `IndexFlatIP` (`Faiss_GPU=False`) on both the CPU and the
  cuda-fallback run; the ONLY device-dependent step is the head forward (A9).
- **Probe script:** `scripts/analysis/b5_conv_probe.py` (deterministic, seeded SEED=0; reuses the real
  repo modules — does NOT reimplement the vote; `macro_f1_fast` self-checked == sklearn
  `f1_score(average='macro', zero_division=0)` on 200 random cases). Device-parametric via
  `B5_PROBE_DEVICE` (default `cpu`; the authorized fallback sets `cuda`).
  - sha256 (r1, cpu-only variant, pre-fallback-edit): `57a774da55b128067d014293347e858de18f1c799cccb8293636350d8bcd02f9`
  - (device-parametric edit applied for the fallback; re-hash recorded at §6 below.)
- **Heads:** 11 safekept checkpoints in `refine-logs/b5_ckpt_snapshot/` (sha256-verified vs source in
  `B5_HEADS_SAFEKEEP_MANIFEST.md`; CLIP s0 e29 serves both protocol slots). Re-verified 2/11 this
  session (CLIP_s0_e29 `a8ccc5ee…`, Qwen_s0_e29 `ce2919e6…`) — match manifest.
- **Cached features:** `data/CLIP_Embedding/MHC_zh/{train,dev_seen,test_seen}_{model}.pt`.
- **Splits (loaded, verified):** dev n=78 (28 pos), test n=149 (45 pos); CLIP-vs-Qwen dev/test
  ids+labels identical per seed/protocol (pairing precondition — asserted OK in-run).
- **G-repro anchors:** re-read from the six `slurm/logs/enc3s_MHC_zh_*_13115.trainlog` primary logs this
  session (all 12 test + 12 dev anchors independently reproduced from `Test_Retrieval` / `Val_Retrieval
  Epoch NN macroF1:` lines; match `B5_PROBE_DESIGN.md` §4 and `B5_PREREG_REVIEW.md` §1.1/§2 Item-5).

---

## Execution timeline (STRICT ORDER; every submission logged)

| # | job | device | type | state | outcome |
|---|---|---|---|---|---|
| 1 | 13156 (`b5probe`) | CPU (`CUDA_VISIBLE_DEVICES=""`) | CPU replay | FAILED exit 2 (by-design HALT) | G-repro: acc+mF1 12/12 PASS 4dp; **roc mismatch (≤7e-4) on 5 Qwen slots** → HALT per strict order |
| 2 | 13158 (`b5probeC`) | **cuda A100-80GB confirmed** (`sacct` AllocTRES `gres/gpu=1`; log `device=cuda` + `nvidia-smi`), `Faiss_GPU=False` | AUTHORIZED G-repro fallback (review §5) — single submission, **cuda spend CONSUMED** | FAILED exit 2 (by-design HALT) | G-repro: acc+mF1 12/12 PASS 4dp; **roc mismatch (≤7e-4) on 4 Qwen slots** (a DIFFERENT set than 13156) → HALT per strict order |
| 3 | 13170 (`b5probe`) | CPU (`CUDA_VISIBLE_DEVICES=""`), `Faiss_GPU=False`; script **v4** (A11 gate) | ONE authorized zero-GPU CPU (b)–(e) continuation (ruling §C; team-lead GO) | **COMPLETED exit 0** (elapsed 00:01:16) | **G-repro 12/12 PASS under A11**; strict order (b)–(e) computed — see below |

The CPU→cuda escalation is the pre-authorized fallback (review §5 cond. 3; prereg §6.3; design §6): a
4-dp CPU G-repro mismatch → ONE `device='cuda'`, `Faiss_GPU=False` eval. **The prereg/review premise
that this cuda eval is "bit-exact to 13115" holds for acc/macroF1 but is FALSE for roc** — see the
root-cause diagnosis after §(a). The single cuda authorization is now consumed (job 13158 verifiably
ran on cuda); no further cuda retries are authorized.

---

## (a) G-REPRO GATE — deployed cut (vote ≥ 0), test AND dev, 4 dp

### CPU replay (job 13156) — HALT evidence

acc + macro-F1 reproduced **12/12 to 4 dp** on BOTH test and dev; roc reproduced on all CLIP arms and
Qwen s0-final, and mismatched (float-eps, all |Δ| ≤ 0.0007) on 5 Qwen slots. Deployed vote-sign is
therefore reproduced everywhere (acc/mF1 exact); only the rank-based roc drifts at the 4th dp on the
higher-dim Qwen head forward. Raw:

| arm·seed·proto | test mf1/acc/roc (CPU) | test anchor | dev mf1/acc/roc (CPU) | dev anchor | verdict |
|---|---|---|---|---|---|
| CLIP s0 final  | 0.7706/0.8054/0.8382 | 0.7706/0.8054/0.8382 | 0.7857/0.8077/0.8329 | 0.7857/0.8077/0.8329 | PASS |
| CLIP s0 valsel | 0.7706/0.8054/0.8382 | 0.7706/0.8054/0.8382 | 0.7857/0.8077/0.8329 | 0.7857/0.8077/0.8329 | PASS |
| CLIP s1 final  | 0.7542/0.8054/0.8342 | 0.7542/0.8054/0.8342 | 0.7225/0.7692/0.8879 | 0.7225/0.7692/0.8879 | PASS |
| CLIP s1 valsel | 0.7579/0.8054/0.8346 | 0.7579/0.8054/0.8346 | 0.7471/0.7821/0.8836 | 0.7471/0.7821/0.8836 | PASS |
| CLIP s2 final  | 0.7913/0.8322/0.8444 | 0.7913/0.8322/0.8444 | 0.7645/0.7949/0.8764 | 0.7645/0.7949/0.8764 | PASS |
| CLIP s2 valsel | 0.7742/0.8121/0.8419 | 0.7742/0.8121/0.8419 | 0.7894/0.8205/0.8343 | 0.7894/0.8205/0.8343 | PASS |
| Qwen s0 final  | 0.7864/0.8188/0.8906 | 0.7864/0.8188/0.8906 | 0.7650/0.7821/0.8579 | 0.7650/0.7821/0.8579 | PASS |
| Qwen s0 valsel | 0.7412/0.7919/**0.8840** | 0.7412/0.7919/**0.8838** | 0.7940/0.8205/0.8693 | 0.7940/0.8205/0.8693 | FAIL (test_roc Δ+0.0002) |
| Qwen s1 final  | 0.7759/0.8054/**0.8949** | 0.7759/0.8054/**0.8951** | 0.8050/0.8205/0.8864 | 0.8050/0.8205/0.8864 | FAIL (test_roc Δ−0.0002) |
| Qwen s1 valsel | 0.7871/0.8121/0.8874 | 0.7871/0.8121/0.8874 | 0.8628/0.8718/**0.9300** | 0.8628/0.8718/**0.9307** | FAIL (dev_roc Δ−0.0007) |
| Qwen s2 final  | 0.7514/0.7852/0.8806 | 0.7514/0.7852/0.8806 | 0.7613/0.7821/**0.8443** | 0.7613/0.7821/**0.8436** | FAIL (dev_roc Δ+0.0007) |
| Qwen s2 valsel | 0.7759/0.8054/**0.8938** | 0.7759/0.8054/**0.8940** | 0.8301/0.8462/0.8514 | 0.8301/0.8462/0.8514 | FAIL (test_roc Δ−0.0002) |

→ CPU G-repro FAIL (roc-only, Qwen-only) ⇒ HALT ⇒ authorized cuda fallback (job 13158).

### cuda fallback (job 13158) — head-forward device=cuda CONFIRMED (A100-80GB); Faiss_GPU=False

Proof the cuda path took effect (refuting any "device flag never propagated" reading): `sacct -j 13158`
AllocTRES = `billing=4,cpu=4,gres/gpu=1,mem=32G`; `slurm/logs/b5probeC_13158.out` header prints
`device=cuda` and `NVIDIA A100-SXM4-80GB, 81920 MiB`; the script banner prints
`head-forward device=cuda ... AUTHORIZED G-REPRO FALLBACK`. The roc values also **differ** from the CPU
run (job 13156) — e.g. Qwen s1-final test_roc 0.8949(CPU)→0.8951(cuda), Qwen s2-valsel test_roc
0.8938(CPU, FAIL)→0.8940(cuda, PASS) — which is only possible if the compute path changed.

| arm·seed·proto | test mf1/acc/roc (cuda) | test anchor | dev mf1/acc/roc (cuda) | dev anchor | verdict |
|---|---|---|---|---|---|
| CLIP s0 final  | 0.7706/0.8054/0.8382 | 0.7706/0.8054/0.8382 | 0.7857/0.8077/0.8329 | 0.7857/0.8077/0.8329 | PASS |
| CLIP s0 valsel | 0.7706/0.8054/0.8382 | 0.7706/0.8054/0.8382 | 0.7857/0.8077/0.8329 | 0.7857/0.8077/0.8329 | PASS |
| CLIP s1 final  | 0.7542/0.8054/0.8342 | 0.7542/0.8054/0.8342 | 0.7225/0.7692/0.8879 | 0.7225/0.7692/0.8879 | PASS |
| CLIP s1 valsel | 0.7579/0.8054/0.8346 | 0.7579/0.8054/0.8346 | 0.7471/0.7821/0.8836 | 0.7471/0.7821/0.8836 | PASS |
| CLIP s2 final  | 0.7913/0.8322/0.8444 | 0.7913/0.8322/0.8444 | 0.7645/0.7949/0.8764 | 0.7645/0.7949/0.8764 | PASS |
| CLIP s2 valsel | 0.7742/0.8121/0.8419 | 0.7742/0.8121/0.8419 | 0.7894/0.8205/0.8343 | 0.7894/0.8205/0.8343 | PASS |
| Qwen s0 final  | 0.7864/0.8188/0.8906 | 0.7864/0.8188/0.8906 | 0.7650/0.7821/0.8579 | 0.7650/0.7821/0.8579 | PASS |
| Qwen s0 valsel | 0.7412/0.7919/**0.8833** | 0.7412/0.7919/**0.8838** | 0.7940/0.8205/**0.8700** | 0.7940/0.8205/**0.8693** | FAIL (test_roc Δ−0.0005, dev_roc Δ+0.0007) |
| Qwen s1 final  | 0.7759/0.8054/0.8951 | 0.7759/0.8054/0.8951 | 0.8050/0.8205/**0.8871** | 0.8050/0.8205/**0.8864** | FAIL (dev_roc Δ+0.0007) |
| Qwen s1 valsel | 0.7871/0.8121/**0.8878** | 0.7871/0.8121/**0.8874** | 0.8628/0.8718/0.9307 | 0.8628/0.8718/0.9307 | FAIL (test_roc Δ+0.0004) |
| Qwen s2 final  | 0.7514/0.7852/0.8806 | 0.7514/0.7852/0.8806 | 0.7613/0.7821/**0.8443** | 0.7613/0.7821/**0.8436** | FAIL (dev_roc Δ+0.0007) |
| Qwen s2 valsel | 0.7759/0.8054/0.8940 | 0.7759/0.8054/0.8940 | 0.8301/0.8462/0.8514 | 0.8301/0.8462/0.8514 | PASS |

cuda G-repro **under the ORIGINAL roc-4dp clause**: 8/12 PASS, 4/12 FAIL (roc-only, Qwen-only, all
|Δ| ≤ 0.0007). acc AND macroF1 match **12/12 to 4 dp on both test and dev** (as on CPU). Under the
original gate this residual roc mismatch ⇒ HALT. **RESOLVED by amendment A11 (see A11 RESOLUTION
below): under the amended gate (roc |Δ| ≤ 1e-3) this same 13158 evidence is a G-repro PASS.**

---

## ROOT-CAUSE DIAGNOSIS — the roc-4dp clause is unsatisfiable by replay (NOT a plumbing bug)

1. **acc + macroF1 reproduce 12/12 EXACTLY** (4 dp, test+dev, on BOTH the CPU and cuda runs). The
   deployed operating point (vote ≥ 0) — the quantity the G-repro gate exists to protect and the only
   quantity the calibration uses — is fully reproduced. Every vote **sign** is correct.
2. **roc is the only mismatch, and it is rank-noise.** roc = `roc_auc_score(labels, continuous_vote)`
   (identical function to the anchor). The continuous vote is a weighted sum of neighbour
   `(2·label−1)·sim·w`; float-eps in the head matmul perturbs the vote at ~1e-5, which occasionally
   swaps two near-tied videos in the AUC ranking → a discrete roc shift of ~1/(n_pos·n_neg) ≈ 2e-4 per
   swap. Observed drift ≤ 7e-4 (1–3 swaps), Qwen-only (3584-dim → more accumulation than CLIP 1024/768).
3. **Even cuda-vs-cuda is not bit-reproducible.** The anchor roc came from job 13115's training-time
   forward, produced by a **non-deterministic cuBLAS algorithm draw** (run_rac.py does not set
   deterministic mode). A fresh eval forward — even on the same node/GPU — can select a different
   cuBLAS kernel and round differently at float-eps. The CPU and cuda runs give **different** roc
   drifts (different failing-slot sets), which is direct evidence the drift is compute-path float
   noise, not a code error (a real bug would move acc/mF1 too, and by more than 1e-4).
4. **Therefore the pre-registered roc-to-4dp clause cannot be satisfied by any replay**, and the
   prereg §6.1 / review §5 premise "cuda fallback = bit-exact to 13115" is false **for the rank
   statistic roc** (it is true for acc/macroF1). This is a pre-registration design fact surfaced by
   execution, not a machinery defect and not a device-plumbing bug.

**Executor stance (review §5 cond. 6 — NO pass/fail interpretation, NO gate relaxation by the
executor):** the binding gate as written is not met → HALT. Whether to (i) amend the G-repro gate to
**acc + macroF1 exact-4dp AND roc within a small tolerance (e.g. |Δ| ≤ 1e-3)** — which the evidence
supports and which would let the probe proceed on the acc/mF1-exact votes — or (ii) uphold roc-4dp and
rule the replay unable to certify, is a **pre-registration amendment / independent-verdict decision
above the executor's authority.** It is NOT resolved here, and (b)–(e) are NOT computed.

---

## A11 RESOLUTION — gate amended (roc |Δ| ≤ 1e-3); G-repro = PASS; ONE CPU run authorized for (b)–(e)

An independent, fresh, zero-context amendment reviewer adjudicated the escalated gate question:
`refine-logs/B5_GATE_AMENDMENT_RULING.md` (commit 5295076), **AMEND-APPROVED, amendment A11**, 2026-07-15.

- **Amended gate (REPLACE-in-place in `B5_PROBE_DESIGN.md` §4 and prereg §6.3):** acc AND macroF1 exact
  at 4 dp (test+dev, all 12 slots) **AND** roc within **|Δ| ≤ 1e-3** of anchor. Rationale: roc is a rank
  statistic from a non-deterministic cuBLAS draw in the 13115 training forward → roc-to-4dp is
  unsatisfiable by any replay; acc/macroF1 (the deployed operating point, the only quantities the
  calibration consumes) reproduce exactly. The reviewer independently re-verified the acc/mF1 12/12-exact
  evidence, the swap-granularity arithmetic (0.0002 test / 0.0007 dev), and code-checked that roc is
  **unused downstream** (referenced only in the (a) gate block; `select_tau`/oracle/honest/D3 use only
  acc/mF1), so relaxing roc cannot move any (b)–(e) verdict — releasing the HALT only.
- **G-repro = PASS on existing evidence, NO new GPU.** Job 13158 (and CPU 13156) both satisfy the amended
  gate: acc/mF1 12/12 exact; max roc |Δ| = 0.0007 < 1e-3. The consumed cuda spend stands; no further
  cuda/GPU run is authorized or needed for the gate.
- **In-code realisation (v4):** `scripts/analysis/b5_conv_probe.py` §(a) roc checks widened to |Δ| ≤ 1e-3
  (acc/mF1 stay exact-4dp); vote / `select_tau` / oracle / D3 / strict order / `sys.exit(2)` untouched.
  Diff-verified; new sha256 in §6.
- **Continuation (authorized): ONE zero-GPU CPU SLURM submission** of the frozen v4 probe to compute
  (b)–(e) under the unchanged strict order (freeze dev-τ → A1 oracle kill-switch → honest preview → D3).
  Executor applies NO pass/fail interpretation; independent verdict processing (A7 hand-check + A2 dev
  anchor) validates the calibration machine before any formal stage. **This ruling unblocks the HALT
  only; it authorizes NO formal stage.** Pending the team-lead's diff re-check GO before submission.

---

All (b)–(e) numbers below are the RAW output of job **13170** (CPU, script v4, G-repro 12/12 PASS under
A11), transcribed verbatim from `slurm/logs/b5probe_13170.out` and cross-checked against
`refine-logs/b5_probe_out/b5_conv_probe_results.json`. **No pass/fail interpretation is applied here**
(the KILL-SWITCH / ELIGIBLE / clears / D3-fragile flags are the script's pre-declared mechanical rule
outputs, not the executor's judgement); independent verdict processing is separate.

### (a) G-REPRO GATE under A11 (job 13170, CPU) — 12/12 PASS

All 12 slots PASS: test+dev acc/macroF1 exact-4dp AND roc |Δ| ≤ 1e-3. (The CPU roc drifts match the
13156 pattern, all ≤ 7e-4 < 1e-3; the 13158 cuda evidence independently passes the same amended gate.)
Deployed roc read from the run: Qwen s0-valsel test 0.8840 / dev 0.8693; s1-final test 0.8949; s1-valsel
dev 0.9300; s2-final dev 0.8443; s2-valsel test 0.8938 — all within 1e-3 of anchor. G-REPRO = **PASS**.

## (b) FROZEN dev-selected thresholds τ (argmax dev macro-F1; A3 lower-median plateau; DEV ONLY, before any test eval)

| arm·seed·proto | τ\*(dev-macroF1) | dev macroF1@τ | τ(dev balanced-acc, secondary) |
|---|---|---|---|
| CLIP s0 final  | +0.06163 | 0.8106 | +0.06163 |
| CLIP s0 valsel | +0.06163 | 0.8106 | +0.06163 |
| CLIP s1 final  | −0.66502 | 0.7956 | −0.99868 |
| CLIP s1 valsel | −0.79554 | 0.8126 | −0.79554 |
| CLIP s2 final  | +0.22164 | 0.7970 | −0.70602 |
| CLIP s2 valsel | −0.01476 | 0.7894 | −0.53520 |
| Qwen s0 final  | −0.53315 | 0.8022 | −0.53315 |
| Qwen s0 valsel | −0.13331 | 0.8017 | −0.13331 |
| Qwen s1 final  | −0.18573 | 0.8501 | −0.18573 |
| Qwen s1 valsel | +0.11897 | 0.8756 | +0.11897 |
| Qwen s2 final  | −0.13818 | 0.8079 | −0.69454 |
| Qwen s2 valsel | +0.01900 | 0.8301 | +0.01900 |

## (c) ORACLE kill-switch (amended A1; each arm its OWN test-optimal τ; paired Qwen−CLIP)

**final-epoch:**

| seed | Qacc | Cacc | ΔAcc | QmF1 | CmF1 | ΔmF1 |
|---|---|---|---|---|---|---|
| 0 | 0.8389 | 0.8188 | +0.0201 | 0.8065 | 0.7837 | +0.0228 |
| 1 | 0.8188 | 0.8121 | +0.0067 | 0.8047 | 0.7677 | +0.0370 |
| 2 | 0.8121 | 0.8322 | −0.0201 | 0.7983 | 0.7943 | +0.0040 |

mean paired ΔAcc_oracle = **+0.0022** (2/3 +); mean paired ΔmF1_oracle = **+0.0213** (3/3 +);
**ELIGIBLE (AND ≥ +0.03) = False.**

**val-selected:**

| seed | Qacc | Cacc | ΔAcc | QmF1 | CmF1 | ΔmF1 |
|---|---|---|---|---|---|---|
| 0 | 0.8054 | 0.8188 | −0.0134 | 0.7828 | 0.7837 | −0.0009 |
| 1 | 0.8121 | 0.8188 | −0.0067 | 0.7960 | 0.7778 | +0.0182 |
| 2 | 0.8389 | 0.8188 | +0.0201 | 0.8039 | 0.7808 | +0.0230 |

mean paired ΔAcc_oracle = **−0.0000** (1/3 +); mean paired ΔmF1_oracle = **+0.0134** (2/3 +);
**ELIGIBLE (AND ≥ +0.03) = False.**

**KILL-SWITCH (A1 per-protocol AND-eligibility): `B5 DEAD (neither protocol eligible) = True`** (script
output; oracle numbers are an upper bound, never a result).

## (d) VAL-CALIBRATED honest preview (frozen dev-τ applied to test; paired Qwen−CLIP) — computed regardless of (c), labeled

**final-epoch:**

| seed | Qacc | Cacc | ΔAcc | QmF1 | CmF1 | ΔmF1 |
|---|---|---|---|---|---|---|
| 0 | 0.7517 | 0.8121 | −0.0604 | 0.7380 | 0.7771 | −0.0391 |
| 1 | 0.7987 | 0.7785 | +0.0201 | 0.7764 | 0.7504 | +0.0260 |
| 2 | 0.8054 | 0.8121 | −0.0067 | 0.7807 | 0.7608 | +0.0199 |

mean paired ΔAcc = **−0.0157** (1/3 +); mean paired ΔmF1 = **+0.0023** (2/3 +);
clears +0.03/+0.03 & 3/3 = **False**.

**val-selected:**

| seed | Qacc | Cacc | ΔAcc | QmF1 | CmF1 | ΔmF1 |
|---|---|---|---|---|---|---|
| 0 | 0.7852 | 0.8121 | −0.0268 | 0.7484 | 0.7771 | −0.0287 |
| 1 | 0.7785 | 0.7517 | +0.0268 | 0.7245 | 0.7302 | −0.0058 |
| 2 | 0.7987 | 0.8054 | −0.0067 | 0.7669 | 0.7677 | −0.0008 |

mean paired ΔAcc = **−0.0022** (1/3 +); mean paired ΔmF1 = **−0.0118** (0/3 +);
clears +0.03/+0.03 & 3/3 = **False**.

**Calibration tax (oracle − honest) + secondary balanced-acc arm (sensitivity only):**

| arm·seed·proto | honAcc | honmF1 | orcAcc | orcmF1 | taxAcc | taxmF1 | balAcc | balmF1 |
|---|---|---|---|---|---|---|---|---|
| CLIP s0 final  | 0.8121 | 0.7771 | 0.8188 | 0.7837 | 0.0067 | 0.0066 | 0.8121 | 0.7771 |
| CLIP s0 valsel | 0.8121 | 0.7771 | 0.8188 | 0.7837 | 0.0067 | 0.0066 | 0.8121 | 0.7771 |
| CLIP s1 final  | 0.7785 | 0.7504 | 0.8121 | 0.7677 | 0.0336 | 0.0173 | 0.6711 | 0.6624 |
| CLIP s1 valsel | 0.7517 | 0.7302 | 0.8188 | 0.7778 | 0.0671 | 0.0476 | 0.7517 | 0.7302 |
| CLIP s2 final  | 0.8121 | 0.7608 | 0.8322 | 0.7943 | 0.0201 | 0.0335 | 0.7919 | 0.7605 |
| CLIP s2 valsel | 0.8054 | 0.7677 | 0.8188 | 0.7808 | 0.0134 | 0.0131 | 0.7584 | 0.7263 |
| Qwen s0 final  | 0.7517 | 0.7380 | 0.8389 | 0.8065 | 0.0872 | 0.0685 | 0.7517 | 0.7380 |
| Qwen s0 valsel | 0.7852 | 0.7484 | 0.8054 | 0.7828 | 0.0201 | 0.0344 | 0.7852 | 0.7484 |
| Qwen s1 final  | 0.7987 | 0.7764 | 0.8188 | 0.8047 | 0.0201 | 0.0283 | 0.7987 | 0.7764 |
| Qwen s1 valsel | 0.7785 | 0.7245 | 0.8121 | 0.7960 | 0.0336 | 0.0715 | 0.7785 | 0.7245 |
| Qwen s2 final  | 0.8054 | 0.7807 | 0.8121 | 0.7983 | 0.0067 | 0.0176 | 0.7919 | 0.7819 |
| Qwen s2 valsel | 0.7987 | 0.7669 | 0.8389 | 0.8039 | 0.0403 | 0.0370 | 0.7987 | 0.7669 |

## (e) D3 GUARDS (≥1000 paired bootstrap, common dev-resample index A6; 3-seed-mean paired Δ; τ stability)

**final-epoch** (1000 resamples, common idx):
- ΔAcc 5/50/95 pct = **−0.0291 / +0.0022 / +0.0604**; 5th ≤ 0 (D3-fragile) = **True**.
- ΔmF1 5/50/95 pct = **−0.0108 / +0.0144 / +0.0606**; 5th ≤ 0 (D3-fragile) = **True**.
- τ stability: Qwen τ by seed = [−0.5332, −0.1857, −0.1382] (std 0.1761); CLIP τ = [+0.0616, −0.6650, +0.2216] (std 0.3858).

**val-selected** (1000 resamples, common idx):
- ΔAcc 5/50/95 pct = **−0.0201 / +0.0067 / +0.0403**; 5th ≤ 0 (D3-fragile) = **True**.
- ΔmF1 5/50/95 pct = **−0.0282 / +0.0006 / +0.0358**; 5th ≤ 0 (D3-fragile) = **True**.
- τ stability: Qwen τ by seed = [−0.1333, +0.1190, +0.0190] (std 0.1037); CLIP τ = [+0.0616, −0.7955, −0.0148] (std 0.3873).

## Handoff to independent verdict processing

Raw numbers only above; the executor applies no scientific interpretation. Independent verdict
processing performs the A7 hand-recomputation of one honest cell from the dumped
`refine-logs/b5_probe_out/{arm}_s{seed}_{proto}.npz` (`votes_*`/`labels_*`, 12 files present) together
with the A2 dev anchor, to validate the calibration machine. The A1 oracle kill-switch and §6.5 honest
preview gates are the pre-declared governors of whether any formal stage is ever spent. **No formal
stage is authorized by this record.**

---

## 6. Fallback submission log & hashes

- **cuda fallback:** `scripts/analysis/b5_conv_probe_cuda.sbatch` → job **13158** (single submission;
  `--gres=gpu:a100:1`, `B5_PROBE_DEVICE=cuda`, `Faiss_GPU=False`; FAILED exit 2 = by-design G-repro
  HALT; ran 12 s on A100-80GB). This WAS the ONE authorized cuda fallback and it verifiably ran on cuda,
  so the single cuda spend is **CONSUMED**; it also mismatched (roc-only) ⇒ no further cuda retries are
  authorized. A resubmit would neither help (the mismatch is intrinsic roc rank float-eps, not a
  device/plumbing fault) nor be authorized.
- **Cosmetic script fix (post-13158, no logic/threshold/order change):** the HALT banner previously
  hardcoded "FAILED ON CPU" for every device (the string that seeded a "device flag never took effect"
  misread). It is now device-accurate (`device={dev}`) with a device-specific next-step line. Diff
  touches only two `print(...)` statements + `sys.exit(2)` context; the vote/threshold/oracle/bootstrap
  logic and the strict order are untouched. No re-run was triggered by this cosmetic edit.
- **Probe-script version chain (`scripts/analysis/b5_conv_probe.py`) — for the reviewer's diff audit
  (review item iv):**
  - **v1** `57a774da55b128067d014293347e858de18f1c799cccb8293636350d8bcd02f9` — CPU-only variant. **Ran
    by job 13156.**
  - **v2** `bfa644b20b7738eeb48229dd795e516c250503ae3a0b781b978d36f72432b0ad` — v1 + device-parametric
    edits (`make_args` reads `B5_PROBE_DEVICE` env; `model_obj.to(args.device)`; header print shows
    device). **Ran by job 13158** (`B5_PROBE_DEVICE=cuda`).
  - **v3** `7c88aa03d1241ef50dc29f2d7ae71ad2e7e8654489adc475ecf07b8d80217460` — v2 + the cosmetic
    device-accurate HALT banner (two `print(...)` branches + the pre-existing `sys.exit(2)`).
    **PRINT-ONLY**; no change to grid / `select_tau` / oracle / bootstrap / strict order. **Not run by
    any job** (13156, 13158 are the only submissions to date).
  - **v4** `3d075345c0425d5ef0a19c87267c6178828c9e72b709798154f370f04147cdb0` — current on-disk = v3 +
    **A11 gate-tolerance edit** (§(a) roc checks widened to |Δ| ≤ 1e-3; acc/mF1 stay exact-4dp; §(a)
    header label updated). Gate-block-only; vote / `select_tau` / `grid` / `oracle_max` / D3 / strict
    order / `sys.exit(2)` untouched. **This is the version the ONE authorized CPU run for (b)–(e) will
    execute.**
  - Version→job map: v1→13156 (CPU, HALT under original gate); v2→13158 (cuda, HALT under original gate,
    the adjudicated evidence); v3→no job; v4→the pending CPU (b)–(e) run.
  - Vote/threshold/oracle/D3 compute logic is byte-identical across **v1→v4**; behavioural deltas are
    ONLY: head-forward device (v1=cpu; v2/v3/v4 respect `B5_PROBE_DEVICE`) and the A11 roc gate tolerance
    (v4) — both downstream-inert for (b)–(e).
- cuda sbatch sha256: `65d1dd05984899a03ad5058a8a4081b77d0c4a81e60fe8dfb4fd6bd98df92a87`.
