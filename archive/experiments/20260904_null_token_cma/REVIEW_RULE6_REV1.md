# Rule-6 code review: candidate 4 revision 1 (null-token keys inside candidate 1's training)

Reviewed 2026-09-05. Scope: only bugs that would change what the experiment observes or concludes (mechanism present in forward/loss/final score, leakage, alignment, hyperparameter and checkpoint chain, shared evaluator). Files: `experiments/20260903_hier_evidence_mil/train.py` (diff against the previous commit), `src/null_token_cma.py`, `experiments/20260903_hier_evidence_mil/search.py`, `experiments/20260904_null_token_cma/launch/run_rev1_*.sh`, `experiments/20260904_null_token_cma/README.md` section 8, with `scripts/reproduction_baselines/macilsd/{Transformer,avce_network,train}.py` and `src/hier_evidence_common.py` as the unchanged substrate. Besides reading, I ran a synthetic CPU forward of the wrapped MACIL-SD layer (B=2, T=7, seq_len 7/4; and the test-time shape B=5, seq_len=None) and one `distil_step` call; no training was run.

## Verdict: PASS (no blocker)

## 1. Pre-existing arms unchanged

The diff to `train.py` adds an import, three names in `ABLATIONS`, the `NULL_TOKEN_ARMS` map, one guarded block in `Candidate.forward` (`if getattr(self.av.cma, "needs_context", False):`, train.py lines 194-206) and one guarded replacement after `partner` is built (lines 368-374). `needs_context` exists only on `NullTokenKeys`; `CrossAttentionBlock`, `_SelfAttnBoth`, `_NoAttn`, `_UnsharedCMA` do not have it, so for `full` and every other pre-existing arm the forward path, the RNG consumption at model construction, the loss, the EMA and the scoring path are the same as before. The import of `src/null_token_cma.py` has no module-level side effects. The candidate-1 record stays valid.

## 2. Mechanism present in training and at test time

- Both cross-modal directions get the extended key/value sequence: `NullTokenKeys.forward` (null_token_cma.py lines 66-76) calls the shared `TransformerLayer` twice, with `cat([n, audio])` as keys and values for the video query and `cat([n, video])` for the audio query; synthetic check: both layer calls see q (2,7,128), k = v (2,8,128), mask (2,8) with column 0 True.
- Mask: `Candidate.forward` builds `mask = arange(T) < seq_len` (True = valid row) on the truncated `keep` length used by the batch; the wrapper prepends a True column, giving (B, T+1), which is exactly the shape `Transformer.attention` checks against `scores.shape[-1]`. Synthetic check: attention mass on padded keys is 0.0 in all three arms; the token receives non-zero mass (about .14-.21 at initialisation).
- Test time: `score_split` calls `model(f_a, f_v, seq_len=None)` with the five crops as the batch (B=5, full untruncated sequence). `Candidate.forward` then builds an all-True mask (5, T) and the context from all rows, calls `set_context`, and the wrapper prepends the token, so the token is in the keys at inference exactly as in training (the train/test consistency that motivated the candidate). Verified with B=5, T=11.
- `set_context` is called on every path into `model.av` (the only caller of `AVCE_Model.forward` is `Candidate.forward`, which always sets it when `needs_context`); the wrapper clears the context after each forward and asserts on a missing one, so a stale context cannot be reused silently (verified: a second forward without `set_context` raises).
- Context source: `c` is the masked mean of `f_a_in[..., SCAF_OFFSET:SCAF_OFFSET+4]` taken after the bookkeeping columns are zeroed, after `ell/ELL_SCALE`, after `hide_input` and `hide_cols` zeroing, i.e. the same four columns and the same scaling the backbone's `fc_a` receives; padded rows (zeros from `process_feat`) are excluded by the mask. This matches the candidate-4 reference (`model.py` lines 190-196).
- The token is inserted after the `fc_a` / `fc_v` projection (hid-dim space), as in the reference; the layer applies its LayerNorm to the query only, so the token enters the key/value projections unnormalised in both implementations.

## 3. Optimiser and EMA chain

- `model.av.cma` is replaced (train.py line 372) before `opt_av = optim.Adam(model.parameters(), ...)` (line 376). Synthetic check: `cma.base`, `cma.cond.weight`, `cma.cond.bias` are in the optimiser's parameter set and receive non-zero gradients through the MIL bag loss.
- `distil_step` matches `model.av` parameters to partner parameters by name. The wrapper keeps the attribute name `layer`, so the shared layer's names (`cma.layer.self_attn.linears.*`, `cma.layer.feed_forward.*`, `cma.layer.sublayer.*`) are unchanged and still EMA-matched to the partner's `cma.layer.*`; the token parameters (`cma.base`, `cma.cond.*`) have no partner counterpart and are left untouched. Synthetic `distil_step` at epoch 0 (m = .91) changed exactly the same parameter set as in the `full` arm plus nothing from the token. (`sublayer` LayerNorms did not appear in the changed list only because the partner's LayerNorms are at the same initial values; they are matched by name as before.)
- `best_state = copy.deepcopy(model.state_dict())` and `load_state_dict` include the token parameters; the wrapper's `context`/`mask` are plain attributes, not buffers, so nothing stale is saved. Test scoring happens in-process after loading, no external checkpoint loading is involved.

## 4. Search and launch chain

- `search.py` passes `--ablation args.ablation` to `train.py` (line 100), writes `<out_root>/<corpus>/seed<seed>/trial<k>/hparams.json` (line 94) and `study_summary.json` with `best.number` (lines 156, 165). The launch scripts read `$RR/$CORPUS/seed$SEED/study_summary.json` -> `best.number` and `trial$BEST/hparams.json`, which is the same layout. `hparams.json` carries `fine_tag=qwen`; the candidate-1 record's trials have no `fine_tag`, which resolves to the same default `qwen` in `train.py`, so the verdict cache is the same.
- The arm loop runs `null_token_const`, `masked_no_token`, `full` with `--config $CFG` = the null_token best trial's hparams and `--seed $SEED`, into `$RR/ablations/$CORPUS/seed$SEED/$ARM`; all three arms and the search share `DEFAULTS` (EMA m .91 / 50 epochs, CMAL weights from the config, five-crop inference). This is the pre-registered comparison of section 8.
- `rev1/` is a fresh output root (no `optuna.db` from candidate 4's own search can be resumed by mistake: candidate 4's studies live under `runs/20260904_null_token_cma/<corpus>/`, revision 1 under `runs/20260904_null_token_cma/rev1/<corpus>/`). Neither remote had a `rev1/` directory at review time.

## 5. Leakage and evaluator

- Splits, HMM fitting on train labels only, checkpoint selection on val, and test scoring through the single shared evaluator (`scripts/reproduction_baselines/eval_baseline_scores.py` via `run_evaluator`) are untouched. The new video-level context uses only the VLM verdict columns already available at test time; no label enters it.
- Test-set use in the Optuna objective is the pre-existing candidate-1 protocol (rule 7), not a change of this revision.

## Non-blocking observations (recorded, no action required)

- The README's claim that all shared parameters are initialised bit-identically to the `full` run of the same seed is correct. However, `nn.Linear(4, hid)` in the wrapper consumes the global torch RNG after `partner` is built, and the training `DataLoader` has no explicit generator, so the epoch shuffle order (and the dropout draws) of the null-token arms differ from the `full` run of the same seed. This is the usual arm-to-arm noise and does not affect the pre-registered comparison; it only means that the arms are not "same data order, token added".
- `masked_no_token` is identical to `full` at inference (all-True mask, no token); it differs only during training, which is the intended candidate-3 setting.
- Operational, outside rule 6: the launch scripts start with `git pull origin main`; at review time both remotes (`uoa-lab1`, `uoa-lab3`) were at an older commit and the revision-1 commit had not been pushed, so pushing must precede launching.
