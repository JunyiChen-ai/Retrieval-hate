# Weak-supervised migration archive

- Correct repository: `/home/jehc223/Retrieval-hate`
- Source repository: `/home/jehc223/Hate-follow-up` (label-free only)
- Migrated: weakly supervised method code, baseline ports, protocols, tables, and shared evaluation/feature infrastructure.
- Kept in the source repository: VERA/LAVAD/Vad-R1/EventVAD and explicitly label-free inference utilities.
- Shared infrastructure is copied, not removed, when the label-free pipeline still depends on it.
- Historical `results/` artifacts were not moved. A few frozen-provenance modules intentionally still point to their archived artifacts in `Hate-follow-up`.
- Machine-readable file inventory and SHA256 values: `WEAK_SUPERVISED_MIGRATION_MANIFEST.json`.

No commit or push was performed as part of this migration.
