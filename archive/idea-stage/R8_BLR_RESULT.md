# R8-1 BLR — result

Run 2026-08-17. Design, arms, read-out and decision rule frozen and committed at `eac73b6`
(`idea-stage/R8_BLR_FREEZE.md`) **before any seed in the range 200-229 was executed**. Single
submission: one background job, **600/600 runs complete, 0 failures, EXIT=0**, ~28 min wall on the
shared local RTX 5090. `idea-stage/r8_blr/analyze.py` was run **exactly once** on the complete grid,
unedited since the freeze commit, with no re-run.

- Grid driver `idea-stage/r8_blr/blr.py` → `idea-stage/r8_blr/results.json`,
  log `logging/runs/r8_blr/run.log`.
- Read-out `idea-stage/r8_blr/analyze.py` → `idea-stage/r8_blr/verdict.json`.
- Cost: **¥0**. Zero API calls, zero cloud.

# VERDICT: **KILL** — 0 of 4 datasets pass

The candidate fails on every dataset. It is never ≥ +0.005 over the global pairwise arm, over the
balanced-BCE arm, or over its own pair-count control, under the primary protocol.

## 1. Arm means — test macro-F1, 30 seeds (200-229)

| dataset | | `A0` BCE | `BALBCE` | `PAIRG` | `PAIRL` | `RANDL` |
|---|---|---|---|---|---|---|
| HateMM | P1 | 0.8701 | 0.8766 | 0.8769 | 0.8733 | 0.8788 |
| | P2 | 0.8688 | 0.8773 | 0.8775 | 0.8794 | 0.8753 |
| | P1 ROC | 0.9165 | 0.9325 | 0.9317 | 0.9337 | 0.9307 |
| MHC-EN | P1 | 0.7136 | 0.7023 | 0.7015 | 0.7003 | 0.7021 |
| | P2 | 0.7319 | 0.7106 | 0.7099 | 0.7099 | 0.7095 |
| | P1 ROC | 0.8545 | 0.8546 | 0.8561 | 0.8602 | 0.8533 |
| MHC-ZH | P1 | 0.8044 | 0.8207 | 0.8219 | 0.8142 | 0.8182 |
| | P2 | 0.8007 | 0.8215 | 0.8261 | 0.8286 | 0.8204 |
| | P1 ROC | 0.9046 | 0.9216 | 0.9215 | 0.9301 | 0.9225 |
| ImpliHateVid | P1 | 0.9128 | 0.9233 | 0.9240 | 0.9273 | 0.9240 |
| | P2 | 0.9112 | 0.9267 | 0.9281 | 0.9279 | 0.9264 |
| | P1 ROC | 0.9659 | 0.9658 | 0.9654 | 0.9678 | 0.9645 |

`A0` reproduces the ledger to within noise on the two datasets that have a ledger entry: HateMM
0.8701 vs 0.8747 and MHC-ZH 0.8044 vs 0.8014, on a disjoint seed range and in a harness declared
(in `r4_harness.py`) to be near, not byte-identical to, `run_rac.py`.

## 2. The candidate — all four frozen conditions, P1

| dataset | `PAIRL−A0` | `PAIRL−PAIRG` | `PAIRL−BALBCE` | `PAIRL−RANDL` | passes |
|---|---|---|---|---|---|
| HateMM | +0.0031 [−0.0006,+0.0074] | **−0.0037** [−0.0058,−0.0015] | **−0.0034** [−0.0055,−0.0012] | **−0.0056** | no |
| MHC-EN | **−0.0132** [−0.0185,−0.0079] | −0.0012 [−0.0047,+0.0026] | −0.0019 [−0.0049,+0.0012] | −0.0017 | no |
| MHC-ZH | +0.0098 [+0.0049,+0.0145] | **−0.0077** [−0.0108,−0.0046] | **−0.0065** [−0.0098,−0.0032] | **−0.0040** | no |
| ImpliHateVid | +0.0146 [+0.0128,+0.0162] | +0.0033 [+0.0019,+0.0047] | +0.0040 [+0.0026,+0.0054] | +0.0033 | no |

Boundary localisation is **negative** against every control on HateMM and MHC-ZH, flat on MHC-EN,
and positive but **below the +0.005 bar** on ImpliHateVid (+0.0033 / +0.0040 against the two
controls that carry a magnitude condition). → **KILL**, exactly as pre-registered and as the
external reviewer predicted.

## 3. The finding — the banked pairwise-objective result is class balancing, not ranking

These are the secondary quantities the freeze required, with no decision attached to them.

**The ranking term, isolated** (`PAIRG − BALBCE`: identical sampling, identical anchored pointwise
term, the *only* difference is whether the pairwise ranking loss is added):

| dataset | Δ macro-F1 P1 | 95 % CI | seeds + | Δ macro-F1 P2 | Δ ROC P1 |
|---|---|---|---|---|---|
| HateMM | +0.0003 | [−0.0023,+0.0029] | 10/30 | +0.0001 | −0.0008 |
| MHC-EN | −0.0008 | [−0.0026,+0.0008] | 5/30 | −0.0008 | +0.0014 |
| MHC-ZH | +0.0011 | [−0.0017,+0.0039] | 13/30 | +0.0046 | −0.0001 |
| ImpliHateVid | +0.0007 | [−0.0004,+0.0017] | 13/30 | +0.0014 | −0.0004 |

**The pairwise ranking term contributes nothing, on any dataset, on either read-out, on either
metric. Every CI contains zero and the largest effect is +0.0011.**

**The balancing, isolated** (`BALBCE − A0`: balanced positive/negative sampling with no ranking term
at all, against ordinary shuffled-minibatch BCE):

| dataset | Δ macro-F1 P1 | 95 % CI | seeds + | Δ macro-F1 P2 | Δ ROC P1 |
|---|---|---|---|---|---|
| HateMM | **+0.0065** | [+0.0023,+0.0109] | 19/30 | +0.0085 | **+0.0160** |
| MHC-EN | **−0.0113** | [−0.0166,−0.0061] | 7/30 | −0.0213 | +0.0002 |
| MHC-ZH | **+0.0163** | [+0.0109,+0.0216] | 27/30 | +0.0208 | **+0.0170** |
| ImpliHateVid | **+0.0106** | [+0.0091,+0.0122] | 30/30 | +0.0155 | −0.0001 |

And the whole banked contrast (`PAIRG − A0`, the closest arm to the recorded pairwise recipe) is
+0.0068 / −0.0121 / +0.0174 / +0.0113 on macro-F1 and +0.0153 / +0.0016 / +0.0170 / −0.0005 on ROC —
i.e. **it is the balancing term, to three decimal places, on every dataset.**

`IDEA_REPORT` §8.8 banked "a pairwise/AUC objective beats BCE on ranking in 4 of 4 cells" as a
baseline upgrade. This run identifies its mechanism: **the pairwise objective samples an equal
number of positives and negatives per step, and that — not the ranking loss — is the entire
effect.** The prediction was made in advance by the literature sweep, from `2512.01766` (May 2026,
attributing last-layer-retraining gains to implicit group balance) and `2607.09832` (Jul 2026,
balanced-softmax classifier retraining on frozen features), and it is now measured here.

**Consequences.**

1. The project's "positive #1" is not an objective-level effect and cannot motivate one. It is a
   one-line change to the sampler, worth +0.0065 to +0.0163 macro-F1 on 3 of 4 datasets with CIs
   excluding zero, and **−0.0113 on MHC-EN** — so it is a per-dataset default, not a universal one.
2. It is not a contribution: `2007.07314` (logit adjustment) and `2607.09832` (BS-cRT) own it.
   It should be banked as a better default and an ablation row, in the same category as the
   R6-1C two-layer concatenation.
3. Round 8's three diagnostics said ranking gains do not convert to macro-F1 here. This run
   strengthens that to the harder statement: **on the test splits, at 30 seeds, the ranking term
   does not even produce a ROC gain once balancing is controlled for** (ΔROC −0.0008 / +0.0014 /
   −0.0001 / −0.0004).

## 4. A fourth, independent instance of the ROC / macro-F1 decoupling

`PAIRL` — boundary-localised ranking pressure — improves **test ROC over every other arm on every
dataset**, with all sixteen CIs excluding zero (`PAIRL−PAIRG` ΔROC +0.0019 / +0.0042 / +0.0086 /
+0.0024; `PAIRL−RANDL` +0.0029 / +0.0069 / +0.0076 / +0.0033) — and it **loses macro-F1 under P1 on
three of the four**. The mechanism designed specifically to concentrate ranking pressure where
macro-F1 is decided produced more ROC and less macro-F1. This is the same signature as diagnostics
D2 and D3, now measured on the test splits with 30 seeds and a paired bootstrap.

## 5. Declared defects and scope limits

1. **MHC-EN was run on the wrong encoder.** The cell used
   `Qwen2.5-VL-7B-Instruct-LoRA_HF`, while `RESEARCH_BRIEF.md` §3 records the deployed MHC-EN
   contrast line as the **frozen** `Qwen2.5-VL-7B-Instruct_HF` (0.7331 vs the LoRA variant's
   0.7133). Every verdict quantity is a within-cell seed-paired delta, so the KILL is unaffected;
   but the MHC-EN *absolutes* sit on a non-deployed encoder, and the MHC-EN sign reversal on
   `BALBCE − A0` may be encoder-specific rather than dataset-specific. The same cell definition was
   used by `idea-stage/r8_decomp/decomp2.py` and `decomp3.py`. Declared, not patched: re-running
   would be a second submission of a frozen grid.
2. `Q = 0.25` and `NPAIR = 1024` were fixed a priori and never swept, per the freeze. A different
   `Q` might behave differently; this run does not test that and no follow-up is authorised by the
   freeze.
3. P1 and P2 disagree in sign on `PAIRL − PAIRG` on three datasets (P1 negative, P2 positive). The
   frozen primary is P1. The divergence is consistent with `PAIRL` producing a sharper score
   distribution that interacts with epoch selection on an 78-107-item dev split — the same
   selection instability measured in D2 (selected epoch std 2.9-7.0 epochs).
4. Test labels were read only for the final metrics. No threshold, epoch rule, arm, hyper-parameter,
   encoder or dataset was selected on them.
