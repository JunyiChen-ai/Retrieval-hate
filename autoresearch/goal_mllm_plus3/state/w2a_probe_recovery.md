# W2-A probe real-run recovery provenance (orchestrator copy)

**PATH-2 CHUNK LOOP (authoritative since 06:1xZ, after F40):** Modal Starter clamps function timeout server-side at ~3600s → probe runs as a resumable chunk loop: soft budget 3200s/chunk, MODAL_PROBE_TIMEOUT=3500, periodic+on-terminate volume commits, probe --ci_ckpt resume (deterministic: point-arms cached, per-seed perm indices rng(CI_PERM_BASE+si), completed arms pinned). ~13-15 chunks ≈ 10-12h wall (450 total perms across 3 variants @ ~72s/perm). Validation chunk: app `ap-nkulMcYIzCYyVHmuLGY4pL` (1200s, progress reused). **RESUME COMMAND if executor dies** (repeat until /W2A_PROBE_RESULTS.md appears on volume rgcl-features; checkpoint /w2a_ci_ckpt.json):
`MODAL_PROBE_APP_NAME=rgcl-w2a-probe-e MODAL_PROBE_TIMEOUT=3500 conda run --no-capture-output -n HateVideo modal run --detach scripts/cloud/modal_probe_runner.py::run --script scripts/analysis/w2a_probe_cloud_adapter.py --chunk-budget 3200`

**INCIDENT 2 (F40): app `ap-qNF5v5HekPTrGvwjPWGIp0` ALSO CANCELLED, 05:53:41Z — ~62 min after launch, same as incident 1. ROOT CAUSE = systematic ~3600s Modal function timeout (12h intent never reached the function decorator for detached runs); F38 cross-agent-sweep hypothesis RETRACTED. Third relaunch authorized 06:07Z with verified timeout plumbing (effective timeout must be PRINTED at launch + banner) or <50min resumable --ci_ckpt chunk loop. Wall-time reality: full K9 ≈ 3.9h/dataset CI ×2 — 3600s could never finish.**

PREVIOUS RUN (dead): app `ap-qNF5v5HekPTrGvwjPWGIp0`, app NAME `rgcl-w2a-probe-e`, launched ~2026-07-16T04:51Z, detached, intended 12h cap (NOT effective), CI_NSEED=150 + frozen probe af4a2f9f confirmed in-container. PROVENANCE CORRECTION (executor, 06:2xZ): the intended "clean relaunch / ckpt deleted" did NOT happen — `modal volume rm` raced eventual consistency, checkpoint persisted, and all subsequent chunks RESUME the original cancelled lineage. Accepted as VALID DETERMINISTIC RESUME: checkpoint _meta hash-pinned (probe af4a2f9f; grd_sha == extraction c013884 manifest exactly), per-seed perm rng(CI_PERM_BASE+si) container-independent, point-arms cached ⇒ final 150-perm null ≡ one uninterrupted run (float drift immaterial, cloud triage-only). Validation chunk `ap-nkulMcYIzCYyVHmuLGY4pL` PROVED periodic-commit + resume (40→60 perms). Full lineage in refine-logs/W2A_CHUNK_LOG.md; throughput ~48s/perm → ETA may beat 16-18Z.

**INCIDENT 2026-07-16T03:38:35Z (discovered 04:47Z): app `ap-93KHpJNP9yDSui6fhIlOLs` CANCELLED mid-HateMM** (cancellation signal in modal-client logs; app stopped, 0 tasks; suspected cross-agent same-name cleanup sweep — F38). No numbers produced/read → clean relaunch authorized 04:48Z.

Original launch: ~2026-07-16T02:36Z, CI_NSEED=150 confirmed in-container. Executor: w2a-implementer-e.

- Modal app id: `ap-93KHpJNP9yDSui6fhIlOLs` — **DEAD, see incident above** (was: rgcl-probe, CPU, --detach, 12h timeout). URL: https://modal.com/apps/jehc223/main/ap-93KHpJNP9yDSui6fhIlOLs
- Status check: `modal app logs <app-id>` — orchestrator now liveness-peeks every tick (detached ≠ safe, F38)
- SUPERSEDED: earlier non-detached app `ap-cPC0wvAsC7rilsT88XNjjU` was STOPPED and relaunched detached; ignore it.
- Volume `rgcl-features` output paths: `/W2A_PROBE_RESULTS.md`, `/w2a_probe_results.json`, `/w2a_ci_ckpt.json` (resumable K9 checkpoint)
- Recovery (if executor reaped): `modal volume get rgcl-features /w2a_probe_results.json <local>` (+ `/W2A_PROBE_RESULTS.md`). Volumes persist only on CLEAN container exit — mid-run these paths are not yet committed.
- Frozen probe `scripts/analysis/w2a_probe.py` sha256 `af4a2f9f5b35461173fd82c176bd52c6fc84bf8fc0d09736f938d38d8f6fe06d` UNCHANGED (byte-identical, run via adapter shim).
- Adapter shim `scripts/analysis/w2a_probe_cloud_adapter.py` sha256 `fb609d4be12dd96162634a8c463fdb68134215f530e8297de550b2f4ee2bccdd`; runner timeout edit +10/-2 (post-edit runner sha256 `fe49d86a7b27feff97fdabede89a22443e934551c68246a5648683637f5ce045`). Plumbing-only, approved by orchestrator 02:31Z.
- Dry-run CI_NSEED=5: plumbing validation only (frozen-hash match, symlinks, HateMM N=851 fail-closed PASS); numbers DISCARDED, never read as evidence.
- Pending on completion: W2A_PROBE_RECORD.md (raw-only) + commit; then independent verdict review (must cross-check K2/K3 LIVE from extraction record c013884 before honoring any K9 PROCEED).
- Cloud numbers = triage-only (~1.4pt drift); never mixed into local tables.
