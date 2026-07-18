# CAND-2 DRAW-2 REPLICATION — Independent 0-Context Verdict Review (HateMM only)

**Reviewer:** independent 0-context verdict reviewer (no prior context by design). **Date:** 2026-07-18.
**Mode:** render the binding verdict strictly against the frozen pre-registration `refine-logs/CAND2_REP2_PREREG.md`
(commit `2d15ffb`). Zero interaction, no push, modified nothing except this review file. Every comparison number
below was **re-derived by the reviewer** from raw trainlogs with a from-scratch parser that reproduces the frozen
banked arms bit-exact (validation in §4).

**D7 novelty + goal satisfaction are OUT OF SCOPE** — they are the USER's rulings (prereg F-R0.4 / §8). This
review decides only whether the F56 HateMM val-sel add-over-generic effect **replicates / weakly-hardens /
downgrades / is ruled draw-noise.**

---

## 0. Gate checks (both must pass before any number is read)

### 0.1 Prereg hash freeze — VERIFIED FIRST
`sha256sum refine-logs/CAND2_REP2_PREREG.md` =
`365511e91f56577df388266f13d5f8f5d963cf03fc6928be0fd9d576c54a2636` — **MATCHES** the mandated target. Proceed.

Frozen artifacts re-hashed at verdict time (all MATCH the freeze block, prereg §5.3 / `CAND2_REP2_FREEZE.md`):

| # | artifact | sha256 | status |
|---|---|---|---|
| P | `CAND2_REP2_PREREG.md` | `365511e9…c2636` | MATCH |
| A | `hatemm_qwen25vl_lora_curric_sft_rep2.yaml` | `d645de31…54c6` | MATCH |
| B | `lora_sft_curric_rep2.sbatch` | `265f3e73…c3c1e` | MATCH |
| C | `enc3seed_lora_curric_rep2.sbatch` | `a32fd3bb…baac` | MATCH |
| — | `train_curric.json` (HateMM, on disk) | `73307ef2…82b` | MATCH (frozen draw-1 curriculum) |

Config `A` line 50 pins `seed: 1` (the single manipulated variable).

### 0.2 MUST-CHECK — effective SFT seed of the draw-2 run == 1  ✅ **PASS (NOT 42)**
The stdout log carries no numeric `seed=` echo, so the reviewer read the **effective runtime seed from the SFT
artifact** as instructed. Loading `logging/lora/HateMM_curric_rep2/training_args.bin` (the pickled
`Seq2SeqTrainingArguments`, under `HateVideo` with `llamafactory` on `PYTHONPATH`):

```
seed = 1
data_seed = None
output_dir = /data/jehc223/RGCL/logging/lora/HateMM_curric_rep2
num_train_epochs = 3.0 ; learning_rate = 0.0001 ; per_device_train_batch_size = 1 ; gradient_accumulation_steps = 8
```

**Effective seed = 1**, not 42. The single manipulated variable took effect; recipe (3 ep, lr 1e-4, eff-bs 8)
and output_dir match F-R0.3. **No VIOLATION — the run is valid and the performance verdict proceeds.**

---

## 1. Raw draw-2 head reads (re-derived from job 13246 trainlogs; val-sel protocol = warmup≥5, max Val acc, roc tie-break)

Group `RAC_video_lora_curric_rep2`, model `Qwen2.5-VL-7B-Instruct-LoRA-curric-rep2_HF`, seeds 0/1/2.

| seed | sel epoch | val acc / roc @ sel | **VAL-SEL** test acc / mF1 | **FINAL-EP** (ep29) test acc / mF1 |
|---|---|---|---|---|
| 0 | 18 | 0.8598 / 0.9193 | 0.8744 / 0.8678 | 0.8837 / 0.8771 |
| 1 | 16 | 0.8692 / 0.9095 | 0.8651 / 0.8574 | 0.8884 / 0.8823 |
| 2 | 29 | 0.8411 / 0.9190 | 0.8791 / 0.8745 | 0.8791 / 0.8745 |
| **mean** | — | — | **0.8729 / 0.8666** | **0.8837 / 0.8780** |

Selection audit (seed2 was a large val-acc tie broken by roc, per frozen protocol): seed0 max val acc 0.8598 at
{15,18}→ep18 higher roc; seed1 unique max 0.8692 @ep16; seed2 val acc 0.8411 tied across {14,19,21–24,26,28,29}→
ep29 highest roc (0.9190). This is the frozen `exp-encoder-3seed` rule applied verbatim.

## 2. Banked comparison arms — RE-DERIVED (not taken from prereg §2), bit-exact reproduction

| arm | protocol | s0 acc/mF1 | s1 acc/mF1 | s2 acc/mF1 | mean acc/mF1 |
|---|---|---|---|---|---|
| generic-LoRA (13235) | val-sel | 0.8605/0.8521 | 0.8698/0.8620 | 0.8558/0.8495 | **0.8620/0.8545** |
| generic-LoRA (13235) | final-ep | 0.8651/0.8580 | 0.8744/0.8660 | 0.8698/0.8613 | **0.8698/0.8618** |
| draw-1 curric (13241) | val-sel | 0.8791/0.8730 | 0.8744/0.8678 | 0.8791/0.8724 | **0.8775/0.8711** |
| draw-1 curric (13241) | final-ep | 0.8791/0.8730 | 0.8791/0.8724 | 0.8791/0.8724 | **0.8791/0.8726** |

Every cell equals the prereg §2 banked table to 4dp (see §4). CLIP floor (12850) taken as banked (prereg §2):
val-sel mean 0.8202/0.8085, final-ep mean 0.8124/0.7936.

---

## 3. Pre-declared bars — rendered VERBATIM against the frozen §3

### 3.1 K-REP-1 (PRIMARY, BINDING, val-sel) — **does NOT PASS**

draw-2 curric-rep2 − generic-LoRA (val-sel), per seed:

| seed | rep2 acc/mF1 | generic acc/mF1 | Δacc | ΔmF1 |
|---|---|---|---|---|
| 0 | 0.8744/0.8678 | 0.8605/0.8521 | **+0.0139** | +0.0157 |
| 1 | 0.8651/0.8574 | 0.8698/0.8620 | **−0.0047** | −0.0046 |
| 2 | 0.8791/0.8745 | 0.8558/0.8495 | **+0.0233** | +0.0250 |
| **mean** | 0.8729/0.8666 | 0.8620/0.8545 | **+0.0108** | **+0.0120** |

PASS rule (frozen §3.1): mean Δacc **≥ +0.010** AND per-seed sign **3/3** AND mean ΔmF1 **≥ 0**.
- mean Δacc = **+0.0108 ≥ +0.010** ✓ (clears by +0.0008)
- per-seed sign = **2/3** positive (seed1 = −0.0047) ✗ ← **the failing condition**
- mean ΔmF1 = **+0.0120 ≥ 0** ✓

⇒ **K-REP-1 does NOT PASS.** The mean cleared the acc bar, but the **3/3 sign gate failed on seed1**
(val-selection landed seed1's rep2 draw at ep16 → test acc 0.8651, below generic seed1's 0.8698). The F56 val-sel
signal is **directionally present but not seed-robust** on the binding protocol under the second draw.

### 3.2 KS-REP (RETIREMENT KILL) — **does NOT fire**
FIRES ⇔ draw-2 val-sel mean Δacc **≤ −0.014**. Observed mean = **+0.0108** (positive, far from −0.014).
⇒ **KS-REP NOT fired.** The effect did not reverse; F56 is **not** ruled draw-noise / not retired.

### 3.3 K-REP-2 (SECONDARY, POOLED 6-pt, val-sel) — **HARDENED**

| draw | s0 Δacc | s1 Δacc | s2 Δacc | draw sign |
|---|---|---|---|---|
| draw-1 (re-derived) | +0.0186 | +0.0046 | +0.0233 | 3/3 |
| draw-2 (measured) | +0.0139 | −0.0047 | +0.0233 | 2/3 |

Pooled arithmetic: sum = 0.0186+0.0046+0.0233+0.0139−0.0047+0.0233 = **+0.0790**; pooled mean = 0.0790/6 =
**+0.01317**; pooled sign = **5/6** positive (only draw-2 seed1 negative).

HARDENED rule (frozen §3.2): pooled mean Δacc **≥ +0.010** AND **≥ 5/6** sign.
- pooled mean **+0.01317 ≥ +0.010** ✓
- sign **5/6 ≥ 5/6** ✓

⇒ **K-REP-2 = HARDENED.** (This is *not* a "wash": a null draw-2 would have given ≤4/6 sign and pooled ≈+0.0078;
the observed 5/6 with pooled +0.0132 clears the pre-declared "stop cherry-picking one draw" read.)

### 3.4 Non-binding reads (reported, not decision-bearing)
- **Final-epoch add-over-generic (F-R0.7: NON-binding).** draw-2 − generic (final-ep): per-seed Δacc
  [+0.0186, +0.0140, +0.0093] → **mean +0.0140, sign 3/3**, mean ΔmF1 +0.0162. (Interesting inversion: draw-2 is
  cleaner on final-ep than val-sel, but draw-1's final-ep was a +0.0093 TIE, so this leg is not the effect under
  replication and carries no decision weight.)
- **draw-2 curric − CLIP (regime sanity).** val-sel per-seed Δacc [+0.0465, +0.0372, +0.0744], mean **+0.0527**,
  3/3, well above the generic−0.014 hold and above the CLIP floor (rep2 mean 0.8729 ≫ 0.8202 val-sel; 0.8837 ≫
  0.8124 final-ep). draw-2 **holds the inherited HateMM pass**; **not** KS-below-floor. Run is healthy.
- **SFT loss sanity (§4a).** `all_results.json` eval_loss = 0.1345 (train_loss 0.1027) — inside the recipe band
  (draw-1 HateMM generic ≈0.108, MHC anchor ≈0.162); finite, not exploding/flat.

---

## 4. Parser validation (why the re-derived numbers are trustworthy)

The reviewer's from-scratch parser (val-sel = warmup≥5 argmax Val acc, roc tie-break → that epoch's Test macro
metrics; final-ep = epoch 29 Test) reproduces **every** frozen banked cell **to 4dp**: generic-LoRA val-sel mean
0.8620/0.8545 and final-ep 0.8698/0.8618; draw-1 curric val-sel 0.8775/0.8711 and final-ep 0.8791/0.8726; and the
draw-1 K-C2-2 per-seed val-sel Δacc [+0.0186,+0.0046,+0.0233] (mean +0.0155, 3/3), matching prereg §2/§3
exactly. Bit-exact reproduction of the banked arms confirms the selection logic **is** the frozen protocol, so the
draw-2 reads are apples-to-apples with the controls. (Draw-1 mean ΔmF1 re-derives to +0.0165 vs the prereg's
+0.0166 — a 0.0001 rounding of the banked draw-1 number, immaterial, non-binding, and not a draw-2 quantity.)

## 5. Compliance (all satisfied)

- **Seed = single manipulated variable = 1** (§0.2); draw-1 was implicit HF default 42. ✔
- **Curriculum bit-exactness (STEP-1b, §4b).** On-disk `train_curric.json` sha = `73307ef2…82b` == frozen draw-1
  curriculum; the 13244 SFT log L1320 re-emits the identical sha ⇒ draw-2 trained the **identical multiset**; the
  only difference from draw-1 is the SFT seed. ✔
- **Same-code pairing (§4c).** `run_one … PY` block of `enc3seed_lora_curric_rep2.sbatch` is **byte-identical**
  (`diff` empty) to both `enc3seed_lora_curric.sbatch` (draw-1) and `enc3seed.sbatch` (anchor); all 3 head seeds
  ran `fusion_mode=align, topk=20, loss=triplet, archive OFF`, differing from controls only by `--model
  …-curric-rep2_HF` and `--group_name RAC_video_lora_curric_rep2`. ✔
- **Single draw-2 attempt (binding; `CAND2_REP2_FREEZE.md` / `…SUBMIT_RECORD.md`).** Exactly one rep2 SFT dir
  (`HateMM_curric_rep2`) and exactly three rep2 head trainlogs (job 13246) exist — no re-draws, no seed-shopping. ✔
- **Single test-touch.** The three 13246 head reads are the only budgeted HateMM-rep2-curriculum-encoder test
  evaluations; test touched only at this verdict. ✔

---

## 6. FINAL VERDICT BLOCK

Decision tree (frozen §3.4) — the reviewer lands **branch 2**: *"K-REP-1 not-pass AND not KS-REP … Adjudicate by
K-REP-2: pooled **HARDENED** ⇒ weakly hardened (draw-1 carried it, draw-2 agreed in direction but under-bar)."*
K-REP-1 does not pass (sign 2/3), KS-REP did not fire (mean +0.0108 > −0.014), and K-REP-2 is HARDENED (pooled
+0.01317, 5/6). **This is NOT the "replicates (hardened)" branch (that requires K-REP-1 PASS) and NOT the
"downgraded" branch (that requires K-REP-2 NOT hardened).**

```
HateMM draw-2: K-REP-1 (val-sel add-over-generic): NOT-PASS (mean +0.0108 acc, sign 2/3, ΔmF1 +0.0120).
               K-REP-2 (pooled 6-pt): HARDENED (pooled mean +0.01317 acc, sign 5/6).
               KS-REP: NOT fired.  final-ep add-over-generic (non-binding): mean +0.0140 acc, sign 3/3.
VERDICT: F56 HateMM val-sel add-over-generic = WEAKLY-HARDENED.
(D7 novelty + goal satisfaction remain the USER's — not decided here.)
```

**Plain-language:** The one live novelty-bearing positive (F56 HateMM curriculum add-over-generic) **did not fully
replicate** on the binding val-selected protocol — the second independent SFT draw missed the strict 3/3 sign gate
because seed1 flipped negative (−0.0047), even though the draw-2 mean (+0.0108) cleared the acc bar. It also **did
not reverse** (KS-REP quiet; run healthy, still well above the CLIP floor and holding the inherited HateMM pass).
The pooled 2-draw read is HARDENED (5/6, +0.0132), so draw-2 **agreed in direction** and the effect is not a
single-draw cherry-pick — but it is **weaker than a clean replication**. Per F-R0.9 this remains a **2-draw**
estimate ("survived one independent replication under-bar," not "proven stable"). The F56 positive stands, softened
from "single-draw PASS" to **"weakly-hardened."**
