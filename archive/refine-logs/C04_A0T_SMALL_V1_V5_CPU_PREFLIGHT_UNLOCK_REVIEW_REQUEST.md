# C04-A0T-SMALL-v1 v5 CPU-Preflight Unlock Review Request

Date: 2026-07-30  
Requested verdict: `GO/REVISE (Critical/High/Important)`  
Execution state: **NOT SUBMITTED**

## Exact review surface

- Authority manifest:
  `85a2ddc140ee523fdbdcd6764a736bdbd6b8c1731b7b76439207498d3d74d5a4`
- Authority closure:
  `b97bc9ad533f01e8dd4a9eee2b047117e68c9cefeee869e96bf6993e63640d3a`
- CPU-preflight-authorized config:
  `5228c08fefcd41c354202dfd7a750b46e1a5d4d66a8b3562b736c9d943f526a6`
- Authority context:
  `1263db666eb9fbae70e4a4609f4378e4e76d1c2db1d81dfed4fcf96584bdbcf1`
- Unlock record:
  `6ca50074f1740e667013456f5af423fd63edf52984c89ba6f018179e7f494db0`
- Code/resource review transcription:
  `17d6433f09718f0eca198d0a67afdf0c62c2e3fb8bc19e7d26e6d1cb8115dc4b`

TARGET snapshot:

- `TARGET_STATE.json`:
  `1c5217bc87b4bd6636952ba816d600e8d8b7f3835a12a3a657b85ff65a7588bd`
- `TARGET_LOOP.md`:
  `651c45c9fe07d045cae745fd1905ecd59e02d82f2888bb0183d66a1514e29de3`
- `TARGET_FINDINGS.md`:
  `c33557adb2c4c96aefccb9282e91055a8d01924510cae822400037cbdba912e9`
- `TARGET_REVIEW_RAW.md`:
  `52a8cbb794e63da9dca3e19b388c8c309784386c64c78090443707ae52da28c3`

## Review question

Confirm that the exact snapshot unlocks only CPU preflight and that:

1. the manifest/config contract, exact manifest pin, closure hash and all
   implementation/design/source/model/processor bindings are consistent;
2. only implementation and preflight materialization are true;
3. prompt/map hashes and payload/GPU/reconciliation reviews remain pending;
4. the absent namespace and `NO_PREFLIGHT_PAYLOAD_YET` semantics prevent
   artifact replay;
5. no teacher, GPU/Slurm-GPU, small tranche, reconciliation, dev/test, OCR,
   API/network, cross-dataset, label, chain/release/resubmit action is unlocked.

No CPU preflight or other job may be submitted before a fresh GO.
