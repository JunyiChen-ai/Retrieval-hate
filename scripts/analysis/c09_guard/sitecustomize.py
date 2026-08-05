"""Installed by PYTHONPATH in scripts/slurm/c09_a0_cpu.sbatch so that EVERY python
process of the C09 A0 job (36 mints + 2 fidelity runs + the arena) carries the
test-split guard from its first bytecode, without editing any hash-frozen module."""
try:
    import c09guard
    c09guard.install()
except Exception as _e:      # never break the interpreter over the guard
    import sys
    sys.stderr.write("[c09guard] install failed: {!r}\n".format(_e))
