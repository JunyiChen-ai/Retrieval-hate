# MOKA-ZH Pre-Registration — HASH FREEZE (single-submit lock)

**Issued by:** the independent 0-context pre-registration reviewer (see
`refine-logs/MOKA_PREREG_REVIEW.md`). **ZERO GPU / SLURM / Modal spent. NO job submitted. NO test
metric read. NO `state/` or `research-wiki/` mutation. Not pushed.**
**Freeze timestamp (`date -u`):** `Sat Jul 25 13:23:02 UTC 2026`.
**Prereg:** `refine-logs/MOKA_PREREG.md`, commit `7c4a22e`
(`prereg+impl: MokA-ZH routed-LoRA cell — awaiting 0-context review`), 842 lines.
**Ruling:** **APPROVED-WITH-NOTES** — 6 non-blocking notes (N1–N6), 5 of them write-up-binding
conditions; none requires a code change, none alters a threshold, kill bar, gate order, or the
test-touch budget.
**Design authority:** `refine-logs/MOKA_FORENSIC_RECON.md` (`dbf30f1`).
**Credited upstream:** `external/baselines/MokA` @ `b28e83431d057e2b83c8b7f5bd7cde9f33d6393a`
(GeWu-Lab/MokA, NeurIPS 2025) — user ruling 2026-07-26: **ungated, credited**.

## VERDICT: **FROZEN** — all 8 objects re-hashed on disk at freeze time, 8/8 MATCH

---

## 1. Freeze block (prereg §5.3, row **P** filled here)

```
FROZEN dc3f1078a89fc2e1de30c870103c2b7f2986fd419698d6c49b5b9ec0966c53f8  refine-logs/MOKA_PREREG.md
A      9b0fc502193b0521f1359978f85b35b6ce98034d1371f174315c8f1a219a8386  src/moka/routed_lora.py
B      fae40487263fd7f65e2d0566205e57d3a2caf1b9d1477c693c7cdfdc891c9749  src/moka/train_moka.py
C      75bb8156705bff3c9bbce97542b90135c8f206f5bac30455f6987b0c48612399  src/utils/generate_VideoMLLM_embedding_lora_HF.py
D      843dace46611d3a2f5e6942a5f0e780fe8b2fb623a3c969949c142bd559d7793  scripts/analysis/moka_smoke.py
E      df3c9a6a8721f5d97935e4b87137732726329877c14ee8635902976eaf70e38b  scripts/slurm/lora_sft_moka.sbatch
F      fd1b7f295cdb7106e0e64629cb1a2391355f9daad4ab0f2ad773292f48b31bde  scripts/slurm/moka_extract_head.sbatch
G      51b883e9f0a78c26d9b4af185b54a4703a250a3cab4c947756782c6c8fe49764  RA-HMD/LLAMA-FACTORY-Ver202512/my_configs/hatevideo/mhc_zh_qwen25vl_lora_moka_sft.yaml
```

| # | path | prereg §5.3 | on-disk at freeze | match |
|---|---|---|---|---|
| P | `refine-logs/MOKA_PREREG.md` | *(reviewer fills)* | `dc3f1078…0966c53f8` | ✔ **now pinned** |
| A | `src/moka/routed_lora.py` (372 L) | `9b0fc502…19a8386` | `9b0fc502…19a8386` | ✔ |
| B | `src/moka/train_moka.py` (52 L) | `fae40487…891c9749` | `fae40487…891c9749` | ✔ |
| C | `src/utils/generate_VideoMLLM_embedding_lora_HF.py` (+33/−2) | `75bb8156…c48612399` | `75bb8156…c48612399` | ✔ |
| D | `scripts/analysis/moka_smoke.py` (324 L) | `843dace4…d559d7793` | `843dace4…d559d7793` | ✔ |
| E | `scripts/slurm/lora_sft_moka.sbatch` (113 L) | `df3c9a6a…eaf70e38b` | `df3c9a6a…eaf70e38b` | ✔ |
| F | `scripts/slurm/moka_extract_head.sbatch` (162 L) | `fd1b7f29…2f48b31bde` | `fd1b7f29…2f48b31bde` | ✔ |
| G | `RA-HMD/…/mhc_zh_qwen25vl_lora_moka_sft.yaml` (48 L, **gitlink-resident**) | `51b883e9…c8f6e49764` | `51b883e9…c8f6e49764` | ✔ |

**Row P note.** The prereg's own sha256 is recorded **here** rather than written into
`MOKA_PREREG.md`, because inserting a file's own digest into that file is self-invalidating. This is
the house pattern (`CAND2_FREEZE.md:19,33`, `LORA_HATEMM_FREEZE.md:17,34`): the freeze document
carries the prereg sha. **`MOKA_PREREG.md` was NOT edited by the reviewer.**

**Row G note (gitlink).** `RA-HMD/LLAMA-FACTORY-Ver202512` is a git **gitlink** (`git ls-files -s`
mode `160000`) pinned at `a912747c408b3c661b4029ecf1d88b9d91c7f1a8` — **unchanged at `HEAD`,
`HEAD~1` and `HEAD~2`**, i.e. commit `7c4a22e` did not move it. Artifact **G** lives inside that
gitlink and is therefore **hash-frozen, not committed** (CAND2 / LORA_HATEMM precedent). Inside the
submodule: **zero modified tracked files**, `git diff --stat HEAD` empty; the only untracked entries
are **G** itself and the pre-existing `.cuda_home_shim/`. **ZERO vendored lines edited.**

---

## 2. Reused-unchanged machinery (prereg §5.2) — verify-only, all MATCH at freeze

```
b85eb72a690bc8fccc2ff5d5358fd6523359bf6596d2b2a0d6d0701bec9e53e3  src/run_rac.py
4379224671defe7dafb638c4f0c8b69295a27d11646b685912a249e2385e29ad  scripts/slurm/enc3seed_zh_b3.sbatch
e767eba0ca6ff40679857e5efb759d72aa985629a9ece6584ea424ac2baba62f  scripts/slurm/lora_sft.sbatch
2f2429fbd8f6b0b82fba173a4efc5f12ae75cf5e5bce791c3819663c5f1439ea  RA-HMD/LLAMA-FACTORY-Ver202512/my_configs/hatevideo/mhc_zh_qwen25vl_lora_sft.yaml
286a9e44953ff2b2f17af3821f3ed3e254569cb68893fefe6b451b04d6ab9101  (run_one block: enc3seed_zh_b3.sbatch:42-83 == moka_extract_head.sbatch:112-153)
```

- `src/run_rac.py` = `b85eb72a…` — **identical to the sha frozen in `ZHPROMPT_PREREG.md §5.2:462`**
  (grep-verified, not assumed).
- `git status --porcelain src/run_rac.py src/model/loss.py src/model/classifier.py
  src/utils/retrieval.py` → **empty (CLEAN)**.
- **`run_one()` byte-identity:** the two 42-line blocks `diff` empty and both hash to
  `286a9e44953ff2b2f17af3821f3ed3e254569cb68893fefe6b451b04d6ab9101`. (Prereg §4.1(d) cites
  `4379224671…` next to "lines 42-83"; that value is the correct **whole-file** pin for
  `enc3seed_zh_b3.sbatch`. Both pins are recorded here — the executor may check either.)
- Banked inputs are **read-only** and this family writes none of them: `logging/lora/MHC_zh`
  (392 tensors, exactly **40,370,176** params — F0.7's figure, re-summed at freeze),
  `data/CLIP_Embedding/MHC_zh/{train,dev_seen,test_seen}_Qwen2.5-VL-7B-Instruct-LoRA_HF.pt`, and the
  three `13150` trainlogs.

---

## 3. Collision surfaces — re-verified ABSENT at freeze

`logging/lora/MHC_zh_moka` ✗ · `logging/Retrieval/MHC_zh/RAC_video_moka*` ✗ ·
`logging/_smoke_moka` ✗ · `data/CLIP_Embedding/MHC_zh/*moka*` and `*-um*` ✗ ·
`slurm/logs/*moka*` ✗. Disk: **518 G avail / 97 % used** (job 1 preflights ≥ 25 G, `exit 3`
otherwise; `disk_guard.sh` threshold 250 G). Job 1 aborts `exit 4` on a pre-existing MokA adapter;
job 1 hard-refuses `HateMM` with `exit 2`; job 2 aborts `exit 2` on a missing input adapter.

---

## 4. VOID-ON-EDIT

**Any byte change to P or A–G voids this authorization.** At submit time the executor MUST re-run
`sha256sum` on **P and A–G** and additionally confirm:

1. `src/run_rac.py` = `b85eb72a690bc8fccc2ff5d5358fd6523359bf6596d2b2a0d6d0701bec9e53e3`;
2. `src/model/loss.py`, `src/model/classifier.py`, `src/utils/retrieval.py` **git-clean**;
3. the `RA-HMD/LLAMA-FACTORY-Ver202512` gitlink still at `a912747c408b3c661b4029ecf1d88b9d91c7f1a8`
   with **zero** modified tracked files inside;
4. `run_one()` in `moka_extract_head.sbatch:112-153` still byte-identical to
   `enc3seed_zh_b3.sbatch:42-83` (block sha `286a9e44…`);
5. the §4.7 collision surfaces still absent, and `logging/_smoke_moka/` deleted after the GPU smoke.

**Any mismatch = AUTHORIZATION VOID.** **Prereg §4.6 stands unmodified:** if the codex gate (§4.5),
the CPU smoke (§4.3) or the GPU smoke (§4.4) forces a code fix, the affected shas change, **this
freeze block MUST be re-issued and a new independent 0-context review re-run against the amended
files before submit.** No code edit lands silently post-freeze.

---

## 5. Conditions carried forward from the review (write-up-binding; NO code change required)

1. **N1** — `KS-MOKA-2` (median `‖A_v − A_t‖_F/‖A_t‖_F < 0.05`) is **structurally unable to fire**:
   measured at freeze, two independent Kaiming draws at the deployed `A` shape (16 × 3584) sit at
   **1.4136** relative-Frobenius, while a trained `lora_A`'s **total** displacement over the real
   3-epoch deployed ZH SFT (`checkpoint-73` → final) is median **0.0506** / max **0.1267**. Report
   `KS-MOKA-2` as a **non-degeneracy floor**, **never** as "routing is real"; base any
   routing-is-active claim on `fallback_calls == 0` under `MOKA_STRICT=1` and on `KS-MOKA-3`.
2. **N2** — restate **F0.2** (single encoder draw; encoder-seed noise **not separable** from the
   routing effect) and **F0.6** (94.6 % vision tokens vs MokA's own 98.4 % text regime) at verdict
   time, whatever the outcome — as §7's template already mandates.
3. **N3** — DEV-1's gather-residual rationale is **shape-specific**: reproduced exactly at the
   smoke's `bsz 2 × seq 11` (5.96e-08 / 1.19e-07), but at the deployed `bsz 1 × seq ≈ 2823` **both**
   formulations are bit-exact at every width tested. Dense remains the conservative freeze; carry the
   shape-dependence if DEV-1 is restated.
4. **N4** — state which parse of `KS-MOKA-1`'s "on BOTH protocols … OR … 3/3" quantifier was applied
   (no false PASS is reachable under either).
5. **N5** — if **both** streams move under `KS-MOKA-3`, say so explicitly instead of forcing one of
   the three enumerated readings; §2.3's bundled "both protocols" FORMAL phrasing is stricter than
   §3.1/§7's per-protocol reporting — report per protocol, require both for the goal conjunct.
6. **N6** — F0.6's 579-row quantile tuple is not bit-reproducible (the "text token" counting
   convention is undefined; the 94.6 % headline reproduces to within 0.5 pt under every convention
   tried). State the convention when F0.6 is restated.

**Documentation-only, non-blocking:** `src/moka/routed_lora.py:34-35` still says "FLOPs are
IDENTICAL", contradicting §F0.7/DEV-1's own amendment (rank identical, compute ≈ +1 %). The prereg
text — which binds the write-up — is correct. Fixing the comment edits artifact **A** and therefore
**fires §4.6**.

---

## 6. Verified-at-freeze machinery summary (reviewer's own runs, CPU only)

- `python scripts/analysis/moka_smoke.py` → **ALL SMOKE CHECKS PASS**, S1–S8, exit 0; **S2 identity
  control `max|Δ| = 0.000e+00` in all 6 cells**; S7 `40,370,176 → 58,490,880 = 1.448864×`; S8 on the
  real ZH record `2688 == 2688`, `seq 2823 = 2688 + 135`.
- Independent re-derivation at **deployed** Qwen2.5-VL-7B projection dims (3584/512/18944) × 4 mask
  patterns incl. the real 94.6 % layout: **dense vs upstream PEFT = 0.000e+00 in 16/16 cells.**
- Monkey-patch: `llamafactory.model.adapter.get_peft_model` rebound, **`peft.get_peft_model` and
  `peft.mapping.get_peft_model` UNPATCHED** (no global leak), hot path `adapter._setup_lora_tuning`
  confirmed, original preserved.
- Floors 13150 re-parsed from raw trainlogs with the `enc3seed_zh_b3` embedded parser: **bit-match
  §2.1**; FORMAL = floor + 0.030 exactly (0.8622 / 0.8315 / 0.8756 / 0.8473).
- Budget 0.2 + 0.6 + 3.1 + 0.7 + 0.05 = **4.65 ≤ 4.7**, every basis re-read from `sacct`
  (12143 `02:39:49` 16 CPU/120 G; 13150 `00:02:46`; `lora_embed` `00:26:17`–`00:37:25`) and from
  `all_results.json` (`train_runtime 8635.9986`).
- `bash -n` both sbatch **SYNTAX_OK**; `py_compile` all four Python artifacts **PASS**; extractor
  defaults `moka=False`, `no_merge=False`.

---

**FREEZE OUTCOME: PASS.** Authorization to proceed to the §3.11 gate order stands, subject to §4.6
and to the §5 conditions above. No mismatch found. No GPU / SLURM / Modal spent at review or freeze.
Not pushed.
