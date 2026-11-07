# Rule-6 code review, review round 2 (E1b), 2026-11-08

Reviewer: one subagent (Claude Fable 5.1), read-only, synthetic checks only. Scope: `baseline_infer.py --hidden`, `e1_fusion.py` variant `hid`, `launch/run_round2_cpu.sh`.

**PASS.** No bug that would crash the runs or change the E1b conclusion. Checked:

1. Hidden features are the input of each baseline's final per-second layer, from the same forward pass as the score, row-aligned with it: Fed-WSVAD hook on `model.temporal` (called once per forward; `permute(1,0,2).reshape(-1,D)` gives the same row order as `score_ids`; an assert turns a double call into a crash); DSANet hook on the input of `classifier` (`visual_features + mlp2(visual_features)`, called once; `reshape(-1,D)[0:n]` matches `logits1`); MACIL-SD unpack order `(..., av_logits, v_out, a_out)` matches `avce_network.py`, `Att_MMIL.fc` reads a_out and v_out, crop mean and index map applied as in `macilsd/infer.py`.
2. `load_hidden`: rows cut or last-row padded to T exactly as `fit_len`; mean and standard deviation from the training split only (synthetic check); missing videos become zeros and are counted.
3. Variant `hid`: alpha fixed at 0, so the loss and the validation objective are on the fused output; test scored once after selection. `nofeat` / `feat` unchanged (same parameters under the same seed, same input widths, nofeat head does not read z).
4. Launcher: paths match round 1; skip logic per output root; failures caught by `wait` and the FAILED grep; summary presence checked.
5. Memory, DeHate (4679 / 559 / 1151 videos, 621k seconds): about 4–6 GB per process, 15–18 GB for 3 processes; uoa-lab1 has about 55 GB available.

Notes: commit and pull before launching on uoa-lab3 / uoa-lab1; the launcher is not for uoa-lab2 (the `results` link is not set up there). E1b feeds the baseline's in-sample features on the training videos by the reviewer's design, so it measures the round-1 F1 concern (over-trust of the baseline) instead of assuming it.

## Variant `hid_val` (same day)

Reviewer: one subagent (Claude Fable 5.1), read-only, synthetic run of `fit_heldout` (23 videos, 5 folds, 3 epochs). **PASS.** Selection sees only validation videos and labels; test scored once with the refit head, alpha 0; every (learning rate, epoch) key of the out-of-fold scores covers each validation video exactly once; the refit uses the chosen learning rate and stops at the chosen epoch; nofeat / feat / hid unchanged. Note: the launcher must run where the `*_hidden.npz` files are (uoa-lab3, uoa-lab1); uoa-lab2 has only the copied `hidden_check.json`. Design note: the cross-validation objective pools out-of-fold scores of five heads, which affects selection only.
