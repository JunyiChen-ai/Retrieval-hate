# CVoI deviation D1 — permanent retirement of old K30 caches

- Timestamp: `2026-08-08T10:55:00+12:00`
- Scope: C5 asset provenance only; no candidate metric was opened.
- Original tolerance: `5e-5`, unchanged.
- Train replay: full `744/744`, `max_abs=1.0418891906738281e-4`, 24 failures, exit 1.
- Decision: `old_cache_comparability=FAIL`; do not run val old-cache replay.
- Retired train SHA256: `8b4a706cec51d106151e57109b24850232239168d5e0ca363341ee76493d7fb7`.
- Retired dev-seen SHA256: `a2ae105e61478b86193267fe67263d1c26436f0881620222f0aa1544fa380778`.
- Required negative evidence: train replay artifact SHA256
  `100db1aaba83546f13a1c7251895f52e6518f1378a48f13d1e069f4c74e6fb4f` and HALT audit SHA256
  `ee05c855c7da77cf5f319cae8e2fc73ec5fe0ee9d048419d73fcae1a9a77f555`.

C5 can pass only through independent validation of the new interior-timestamp train and val
dense4 assets. Passing C5 must never restore an old-cache comparability claim.
