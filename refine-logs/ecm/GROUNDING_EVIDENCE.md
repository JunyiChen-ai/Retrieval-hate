# ECM-RGCL Grounding Evidence

## Local evidence frozen before proposal

- `TARGET_LOOP.md`, `TARGET_STATE.json`, `TARGET_FINDINGS.md`: final target and supervision contract; SSR/EDCM/CTE stopped under their frozen gates; target unmet.
- `research-wiki/TARGET_GATE0_ITER3_LITERATURE.md`: ECM was the third reserve; its only defensible delta is strict-OOF MLLM semantic failure diagnosis connected to final-retrieval constrained optimization.
- `refine-logs/EXPERIMENT_RESULTS.md`: SSR's optimistic selected-edge universe touched only 2/7 EN and 3/15 ZH errors depending on family.
- `refine-logs/edcm/EXPERIMENT_RESULTS.md`: top-64/two-swap reachability was `+0.0273/+0.0394` EN and `+0.0380/+0.0444` ZH, below the frozen screen.
- `refine-logs/cte/EXPERIMENT_RESULTS.md`: CTE stopped at exact numerical/cost parity with zero teacher calls; this is not a performance impossibility result.
- `refine-logs/sq/EXPERIMENT_TRACKER.md`: SQ remains `PLAN_ONLY_NOT_RUN` at formal S0/S1. It must not be called a performance failure.
- `src/model/classifier.py`, `src/model/loss.py`, `src/run_rac.py`, `src/model/evaluate_rac.py`: the final embedding is produced by the existing `img_proj`, `text_proj`, align fusion and MLP; ordinary kNN is the endpoint; the current optimizer is AdamW over model parameters and can be intercepted between `backward()` and `step()` without adding an inference module.

## Closest method families and frozen novelty boundary

- **GroupDRO** (Sagawa et al., ICLR 2020): minimizes worst predefined-group risk; regularization/early stopping is essential. ECM may not claim minimax/group robustness as new.
- **PG-DRO** (Ghosal & Li, AAAI 2023): already supports probabilistic group membership. A soft MLLM posterior plus DRO is therefore a required baseline, not ECM novelty.
- **JTT** (Liu et al., ICML 2021): a first model identifies misclassified training examples and the second upweights them. ECM must not expose correctness to the teacher or gate/upweight error examples.
- **EIIL** (Creager et al., ICML 2021): infers environments from a reference classifier for invariant learning. ECM modes must be semantic failure mechanisms, not merely inferred difficulty/environment partitions, and EIIL+GroupDRO is binding.
- **PCGrad** (Yu et al., NeurIPS 2020), **MGDA** (Sener & Koltun, NeurIPS 2018) and **CAGrad** (Liu et al., NeurIPS 2021): projection/common-descent/conflict-averse optimization is prior art. ECM cannot claim gradient projection itself; same-mode PCGrad/CAGrad controls are binding.
- **DISC** (Wu et al., ICML 2023): concept discovery plus intervention already addresses interpretable spurious correlation. ECM is not first concept-aware debiasing.
- Prior-art caution: stochastic gradient manipulation may not converge to a Pareto solution when weights depend only on instantaneous gradients (Zhou et al., NeurIPS 2022). ECM therefore uses EMA-normalized mode gradients, a base-descent constraint and a deterministic fallback.

## Narrow claim allowed

The only candidate novelty is the complete interface: a frozen MLLM assigns confidence-bearing **whole-video failure-mechanism posteriors to every strict-OOF train prediction without seeing correctness or gold**, and those posteriors define balanced semantic mode objectives whose common-descent constraints directly edit the shared embedding optimizer used by the unchanged final RGCL kNN. The optimization primitive alone is not new.

## Primary sources

- GroupDRO: https://arxiv.org/abs/1911.08731
- PG-DRO: https://ojs.aaai.org/index.php/AAAI/article/view/26394
- JTT: https://arxiv.org/abs/2107.09044
- EIIL: https://proceedings.mlr.press/v139/creager21a.html
- PCGrad: https://papers.nips.cc/paper/2020/file/3fe78a8acf5fda99de95303940a2420c-Paper.pdf
- MGDA for deep MTL: https://proceedings.neurips.cc/paper/2018/hash/432aca3a1e345e339f35a30c8f65edce-Abstract.html
- CAGrad: https://proceedings.neurips.cc/paper/2021/hash/9d27fdf2477ffbff837d73ef7ae23db9-Abstract.html
- DISC: https://proceedings.mlr.press/v202/wu23w.html
- Stochastic gradient-manipulation caveat: https://proceedings.neurips.cc/paper_files/paper/2022/hash/f91bd64a3620aad8e70a27ad9cb3ca57-Abstract-Conference.html
