# Relation-V9 minimal pilot

The same code is used per corpus. Each manifest must add `train_scores` for
every evidence stream. Dependence weights and frozen marginal calibration are
fit only on that corpus's train split. Training uses video labels only; each
epoch is evaluated on validation Frame AP/ROC and the checkpoint is selected
by maximum validation AP with ROC tie-break. Test is not opened by training or
selection.

The preregistered pilot is one seed (234), 5 epochs, one architecture, expert
dropout 0.2, Gaussian score noise 0.01, and no hyperparameter sweep. Run the
synthetic tests first, then one corpus per SLURM task. HCS currently has complete
VADCLIP/Fed dense train scores and can use a dedicated two-stream pilot manifest;
VERA is not silently imputed. The other three corpora must generate train scores
for their declared pools before a valid pilot.

HateMM has a fail-closed three-stream manifest at
`results/reproduction/relation_v9/manifests/hatemm_macil_vera.json`.  Its two
MACIL branches share each seed's one AV checkpoint inference; fixed VERA still
requires a real HateMM train inference.  Generate the missing evidence only on
SLURM, then require the CPU preflight before the pilot:

```bash
bash scripts/reproduction_baselines/relation_v9/submit_hatemm_pipeline.sh
```

The wrapper submits MACIL and VERA evidence independently, then an `afterok`
preflight dependency, then an `afterok` pilot dependency.  Evidence jobs never
run the global completeness preflight themselves.

```bash
sbatch scripts/reproduction_baselines/relation_v9/relation_v9_smoke.sbatch
```
