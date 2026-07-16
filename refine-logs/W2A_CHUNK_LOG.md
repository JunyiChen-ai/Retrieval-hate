# W2-A probe — chunked-resume ledger (path 2)

Raw operational ledger for the chunked K9 run. Root cause: Modal clamps the effective
function timeout to ~3600s server-side (Starter-plan cap) despite a correctly-passed
`MODAL_PROBE_TIMEOUT=43200` (verified: env reaches child, computes 43200, no stray 3600);
both single-container attempts died at ~62 min. Path 2 = resumable `--ci_ckpt` chunk loop
under a soft budget < cap, committing the checkpoint to the volume each chunk.

**Integrity / determinism:** every chunk shares one hash-pinned checkpoint
`/w2a_ci_ckpt.json` whose `_meta` pins `probe_sha=af4a2f9f…` (frozen probe) and
`grd_sha` = the extraction c013884 grounded-cache sha256s
(HateMM `1cae1f83…+41bda7de…`, MHC `9f8da7a1…+7c1a1a4f…`, both verified MATCH). Each perm
seed is `np.random.default_rng(CI_PERM_BASE+si)`, container-independent; point-arms cached
once. The accumulated 150-perm null is therefore equivalent to one uninterrupted run.

**Pre-loop lineage (cancellations + resume):**
- Run A `ap-93KHpJNP9yDSui6fhIlOLs` (first detached, launch ~02:36Z) — CANCELLED ~03:38:35Z (~62 min, server timeout). Committed HateMM Z_best point-arms + partial perms.
- Run B `ap-qNF5v5HekPTrGvwjPWGIp0` (distinct name rgcl-w2a-probe-e, launch ~04:51Z) — CANCELLED 05:53:41Z (~62 min, server timeout).
- Volume `modal volume rm /w2a_ci_ckpt.json` issued ~06:00Z but RACED with Modal volume eventual-consistency — the checkpoint persisted; the validation chunk RESUMED it (so this is a valid deterministic resume of the cancelled lineage, not a clean-from-scratch run — corrected provenance).
- Validation chunk C `ap-nkulMcYIzCYyVHmuLGY4pL` (budget 1200s, launch ~06:1xZ) — proved the soft-budget→periodic-commit→resume mechanism; checkpoint advanced HateMM Z_best 40→60 perms, point-arms cached.

## Chunk ledger

Driver history: the client-side chain driver (chunk 1) was REAPED at ~07:31Z (login=compute
node reaps non-SLURM background processes — the recurring failure mode), so chunk 2 never
chained. Fix (team-lead directive 07:45Z): the chunk loop now runs inside a CPU-ONLY SLURM
job (SLURM job 13212, non-detached modal client, no GPU) which is not reaped. Probe compute
still on Modal. Checkpoint safe throughout (chunk 1's progress committed at its soft-terminate).

| chunk | app id | launch ts (UTC) | perms-done-after (sum across cells) |
|---|---|---|---|
| val (C) | ap-nkulMcYIzCYyVHmuLGY4pL | ~2026-07-16T06:1xZ | 60 (HateMM Z_best) |
| 1 (reaped driver) | ap-R8EPtV3V1aQsBgmj3TElRl | ~2026-07-16T06:3xZ | 140 (HateMM Z_best); committed before driver reap ~07:31Z |
| loop under SLURM 13212 (rows appended by the job below) | | | |
| 1 | ap-DHOcJQWtHAVEU2g54M1xyq | 2026-07-16T08:49:58Z | 230 |
| 2 | ap-7SspMdpMBRJjtfzSWqvnUu | 2026-07-16T09:43:38Z | 310 |
| 3 | ap-RPjOTBQsELsdyyltlbGTxp | 2026-07-16T10:37:19Z | COMPLETE |
