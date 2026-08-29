# MOLMO2 — $0 CPU-head probe record + VERDICT (2026-07-27)

**VERDICT: KILL.** Frozen Molmo2-8B lands **below** the strongest same-path floor on both
protocols and both metrics. The pre-declared bar (recon §7: Δ ≥ +0.0200 on acc **and** mF1,
3/3 sign-consistent, both protocols) is missed by a wide margin and with the **wrong sign**.

No formal/GPU 3-seed cell was started. This probe selects nothing and promotes nothing.

Companion: `refine-logs/MOLMO2_FORENSIC_RECON.md` (model identity, env, disk, deviation ledger,
and the bars — all fixed **before** extraction, commits `c1d450c` / `997b227`).

---

## 0. PROVENANCE

| item | value |
|---|---|
| encoder | `allenai/Molmo2-8B` (Qwen3-8B LLM + SigLIP2-so400m-patch14-384), apache-2.0 |
| local weights | `/data/jehc223/models/Molmo2-8B-bf16` (fp32→bf16 shard-streamed, 17.32 GB) |
| extraction job | **13648**, COMPLETED, 46:27 elapsed (of which 30:45 was `disk_guard` preamble) |
| probe job | **13653**, COMPLETED, 47:15 elapsed (chained `--dependency=afterok:13648`) |
| GPU cost | **~18 min** (extraction only). Probe = **$0 GPU**, CPU-only. |
| dataset | HateMM (F44 cap), 744 train / 107 val / 215 test |
| features | `data/CLIP_Embedding/HateMM/{train,dev_seen,test_seen}_Molmo2-8B_HF.pt`, Dv=Dt=**4096** |
| md5 | train `710ed9f085140a2a0efe7256d61f40a8` · dev_seen `2dcb179e022317d1837616281dff2998` · test_seen `70d481a27ff40c0e3179051b0c7661a4` |

**Zero-vector guard is matched across all three arms.** `hate_video_95.mp4` is corrupt (decord and
PyAV both fail on NAL-unit splitting) and takes the zero-vector guard in the Molmo2 cache. Both
banked Qwen caches contain exactly **1** zero-vector row, the **same** `hate_video_95`. All arms
carry the identical guard on the identical item; comparability is intact.

### 0.1 Same-path floor validated — arm B reproduces the banked proxy EXACTLY

CPU-trained heads are not bit-exact to the CUDA floor (F87 / ERRPAT §8), so arms B and C were
**re-run in this job** rather than quoted. Arm B reproduces the banked ERRPAT CPU proxy to **4 dp
in all four cells**:

| | val-sel acc / mF1 | final acc / mF1 |
|---|---|---|
| ERRPAT banked CPU proxy | 0.8775 / 0.8715 | 0.8760 / 0.8699 |
| **arm B measured here** | **0.8775 / 0.8715** | **0.8760 / 0.8699** |

So the floor this verdict is measured against is the right floor, on the right path.

---

## 1. HEAD RESULTS — 3 arms × 3 seeds, HateMM

Byte-identical `run_rac.py` command from `scripts/slurm/enc3seed_lora_curric.sbatch`; only
`--model`, `--device cpu`, `--save_embed`, `--group_name`, `--output_path`, `--exp_comment` differ.

| arm | encoder | val-sel acc | val-sel mF1 | final acc | final mF1 |
|---|---|---|---|---|---|
| **A** | **Molmo2-8B (frozen)** | **0.8558** | **0.8466** | **0.8636** | **0.8548** |
| **B** | Qwen2.5-VL-7B LoRA-curric (**floor**) | 0.8775 | 0.8715 | 0.8760 | 0.8699 |
| **C** | Qwen2.5-VL-7B frozen (control) | 0.8620 | 0.8534 | 0.8620 | 0.8531 |

Per-seed (acc): A val-sel 0.8419 / 0.8605 / 0.8651, final 0.8744 / 0.8605 / 0.8558.

### 1.1 Verdict against the pre-declared bar

| comparison | protocol | Δacc | ΔmF1 | per-seed acc signs | bar |
|---|---|---|---|---|---|
| **A − B (the bar)** | val-sel | **−0.0217** | **−0.0249** | − − − | **FAIL** |
| **A − B (the bar)** | final | **−0.0124** | **−0.0151** | + − − | **FAIL** |
| A − C (control) | val-sel | −0.0062 | −0.0068 | − − + | tie |
| A − C (control) | final | +0.0015 | +0.0016 | + − − | tie |

Bar required **≥ +0.0200 on both metrics, 3/3 signs, both protocols**. Measured: negative on
both metrics on both protocols. **KILL — not marginal, not a split.**

Against the frozen-vs-frozen control the swap is a **tie**: |Δ| ≤ 0.0068, and 1 test item on
n=215 = 0.00465, so every A−C cell is within ~1-2 items. **A better video-native encoder is not
a better encoder for this task.**

---

## 2. GEOMETRY — did the representation actually change?

RAW pre-head encoder output (`scripts/analysis/molmo2_geom_diag.py`, bank = train, query = test,
deployed top-20 rank-weighted signed-cosine operator). No head, no training, no selection.

| arm | view | raw-kNN acc | mF1 | top-1 cos | mean top-20 | PR | lead var | ρ(length) |
|---|---|---|---|---|---|---|---|---|
| **A molmo2** | img | **0.7814** | 0.7689 | **0.9960** | 0.9941 | 13.632 | 0.2234 | +0.4856 |
| **A molmo2** | text | 0.8000 | 0.7970 | **0.9881** | 0.9845 | **27.870** | **0.1419** | **+0.9052** |
| **A molmo2** | concat | **0.8186** | 0.8145 | **0.9904** | 0.9876 | **33.607** | **0.1126** | +0.8507 |
| **A molmo2** | hadamard | **0.5628** | 0.5463 | **0.9999** | 0.9997 | **3.069** | 0.4613 | +0.4733 |
| B lora-curric | img | 0.7256 | 0.7112 | 0.9537 | 0.9275 | 12.551 | 0.2329 | +0.4122 |
| B lora-curric | text | 0.8233 | 0.8208 | 0.9554 | 0.9413 | 23.071 | 0.1565 | +0.9432 |
| B lora-curric | concat | 0.8140 | 0.8118 | 0.9439 | 0.9234 | 22.337 | 0.1670 | +0.8617 |
| B lora-curric | hadamard | 0.7535 | 0.7437 | 0.9686 | 0.9478 | 8.635 | 0.2341 | +0.5533 |
| C frozen-qwen | img | 0.7163 | 0.7014 | 0.9536 | 0.9274 | 12.713 | 0.2297 | +0.4275 |
| C frozen-qwen | text | 0.8186 | 0.8159 | 0.9627 | 0.9505 | 20.701 | 0.1661 | +0.9530 |
| C frozen-qwen | concat | 0.8000 | 0.7983 | 0.9477 | 0.9281 | 21.436 | 0.1716 | +0.8524 |
| C frozen-qwen | hadamard | 0.7116 | 0.6972 | 0.9680 | 0.9476 | 8.746 | 0.2283 | +0.5834 |

**The geometry DID change — and the accuracy still did not convert.** That dissociation is the
result worth carrying, not the flat number.

1. **The vision tower genuinely improved.** Molmo2's raw image stream is the best image stream
   measured on HateMM: **0.7814** vs 0.7256 (LoRA-curric, **+0.0558**) and 0.7163 (frozen Qwen,
   **+0.0651**). The SigLIP2 video-native tower does exactly what the literature advertises.
2. **It bought nothing.** The trained head still lands 0.0217 *below* the floor. This is the
   **9th law-I datum**, and the cleanest one yet: the image side moved by +0.056 in raw retrieval
   and the deployed number went **down**. Previous law-I data showed the image stream moving with
   zero conversion; here it moves *and the conversion is negative*.
3. **Cone collapse got WORSE, not better.** Top-1 cosine rises from Qwen's 0.9439-0.9686 to
   **0.9881-0.9999**. The pathology F89 named as the thing that must change moved in the wrong
   direction. Molmo2's keys are more nearly parallel than Qwen's.
4. **The spectrum simultaneously got BROADER.** Participation ratio rises (text 27.87 vs 23.07;
   concat **33.607** vs 22.337) and leading-direction variance share falls (concat **0.1126** vs
   0.1670). So Molmo2 spreads variance over *more* directions while its top-1 neighbours are
   *more* saturated — a different geometry, not a better-conditioned one for retrieval.
5. **The nuisance axis survived intact.** Length-organisation ρ on the text stream is **+0.9052**
   vs +0.9432 / +0.9530. Retrieval is still almost entirely ordered by transcript length. Swapping
   the encoder does not touch the axis F89 identified.
6. **Hadamard degenerates.** Molmo2's raw elementwise product collapses to acc 0.5628 with
   PR 3.069 and top-1 cosine 0.9999 — two near-parallel streams multiplied together annihilate the
   signal. The deployed fusion is Hadamard but in *head* space after learned projections, so this
   is diagnostic rather than deployed; it is nonetheless the most plausible mechanism for why a
   better raw concat (0.8186, the best fused raw read of all three arms) trains to a *worse* head.

---

## 3. WHAT THIS CLOSES

The encoder swap was one of the **two remaining entry points** F89 left open after every
post-representation repair was closed. This probe closes it at the strongest available candidate:
a 2025-generation, video-native, SigLIP2-towered 8B model, on the one dataset (HateMM) where
encoder identity had ever converted before.

It closes it in the informative direction. The swap is not a null that leaves "maybe a better
encoder exists" open — it is a case where the encoder measurably **is** better on the image side
(+0.056 raw kNN) and measurably **does** reorganise the spectrum, and the deployed metric still
declines. The binding constraint is therefore not the quality of the visual representation. It is
the selection/fusion path and the length-organised retrieval geometry, both of which the swap left
untouched (ρ ≈ +0.91 either way).

**Recommendation: park the encoder-swap axis.** Do not re-propose a different VLM tower at 4-8B
on HateMM; the informative version of that experiment has now been run. If the encoder axis is
ever reopened it should be on the *text* side (the dominant stream, and the one Molmo2 made
slightly worse: 0.8000 vs 0.8233), not the vision side.

---

## 4. LIMITATIONS

1. **Frozen vs adapted asymmetry.** Arm A is frozen, arm B is LoRA-adapted. That is why arm C is
   carried; against the like-for-like frozen control the result is a **tie**, not a loss. The
   *promotion* bar is correctly set against the strongest floor, since a new encoder must beat what
   is deployed to matter.
2. **Head-size asymmetry.** Molmo2's 4096-d features give the input projection ~14 % more
   parameters than the 3584-d arms. Inherent to any encoder swap across hidden sizes (the 32B arm
   at 5120-d had the same property); not controllable without crippling one arm.
3. **Three FORCED prompt/resolution deviations** (recon §4): fixed 378×378 with 3×3 pooling
   (648 vision tokens vs Qwen's ~1540), the `<|video|>` placeholder preceding the user turn, and
   injected per-frame timestamps. These are what "video-native" *means*; they cannot be removed
   without defeating the swap. This is an encoder-as-shipped comparison, not a single-variable
   ablation.
4. **fp32 → bf16 cast** of the checkpoint (recon §3), matched to the deployed compute dtype but a
   real difference from running the weights as shipped.
5. **CPU-trained heads**, compared only against CPU-trained floors (§0.1 confirms parity at 4 dp).
   Absolute numbers are proxy-level, not the banked GPU floor.
6. **Raw-space geometry is pre-head** and therefore diagnostic. The deployed pipeline sees these
   features only through learned projections; §2.6 is a mechanism hypothesis, not a measurement of
   the deployed fusion.
7. All test quantities are **single-draw descriptive reads**. Val-selection used val only
   (warmup ≥ 5); the final-epoch protocol selects nothing. No test number selected anything.
