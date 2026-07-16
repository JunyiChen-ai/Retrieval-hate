# LoRA-HateMM Pre-Registration — HASH FREEZE (single-submit lock)

**Executor:** submit executor (freeze → collision re-check → smoke → single-submit → verify-start → report). ZERO user interaction. No verdict produced; no test metric read.
**Freeze timestamp (`date -u`):** `Thu Jul 16 16:19:51 UTC 2026`
**Prereg commit:** `3ebd880` (`Pre-register encoder-level LoRA-HateMM measurement (+ bundled B4-EN closure)`)
**Review commit:** `2e41332` (`Independent 0-context review of LORA_HATEMM_PREREG (APPROVED-WITH-NOTES; floors re-parsed, hashes verified)`) — repo HEAD at freeze time.

## Verdict: FROZEN — all artifacts locked, byte-for-byte re-hash PASS

Every sha256 below was recomputed on disk at freeze time and matched, byte-for-byte, the values pinned in
`LORA_HATEMM_PREREG.md` §6.1 / §6.2 / §1.1 and independently reproduced in `LORA_HATEMM_PREREG_REVIEW.md`
(freeze block, lines 164–167; data + reused-machinery shas throughout). No mismatch. Authorization intact.

### Re-hash outputs (verbatim `sha256sum`, freeze time)

```
da1759a4b18fd7689c9086d5343a0e6805dedd24b76c3f2933bdc83a1941cd4b  refine-logs/LORA_HATEMM_PREREG.md
d2f415cd93fa6f7b439fd4b4573a536baf48ad42186dc8bd50f3fab20553e36a  RA-HMD/LLAMA-FACTORY-Ver202512/my_configs/hatevideo/hatemm_qwen25vl_lora_sft.yaml
e767eba0ca6ff40679857e5efb759d72aa985629a9ece6584ea424ac2baba62f  scripts/slurm/lora_sft.sbatch
19c76b177f7dc883a9e03524450ad2e6cb302cdd0a6704d69da68a62188a06fc  scripts/slurm/enc3seed_lora_hatemm.sbatch
c76bb42240feaa300c8b89cdb1fdba1c2d0dbb7360b0ffe53d32fc260a46f386  scripts/slurm/gen_embed_lora.sbatch
dbe3fb81800897cb7bac56d71f5d881d54d46421fdbda214df00d4deb0815c3d  scripts/slurm/enc3seed.sbatch
4379224671defe7dafb638c4f0c8b69295a27d11646b685912a249e2385e29ad  scripts/slurm/enc3seed_zh_b3.sbatch
db371c18f306c5a3a00eeef8550964c3ddacf9e20400439324009ef2e69b1b52  RA-HMD/LLAMA-FACTORY-Ver202512/my_configs/hatevideo/mhc_qwen25vl_lora_sft.yaml
93c6d3d1bffbca22b2dd8beba57a33575a48d8ca61d8d56e3148fecdbb93973a  data/lora_sft/HateMM/train.json
9e103ed35a014af81eb3aa6af0d51a28707efd66a606c5bf0459db570a9cc9ef  data/lora_sft/HateMM/val.json
c12ad356aa2917ed80ef17ba93e7854cd36751f770f05a3b19956cfbfdce8462  data/lora_sft/HateMM/test.json
ebf14b472744b0ca2007695033026b9dde4538aa37ccf019b9482a1ab07681b5  RA-HMD/LLAMA-FACTORY-Ver202512/data/dataset_info.json
```

### Frozen freeze-block (matches prereg §6.4 template + review lines 164–167)

```
FROZEN da1759a4b18fd7689c9086d5343a0e6805dedd24b76c3f2933bdc83a1941cd4b   LORA_HATEMM_PREREG.md
A      d2f415cd93fa6f7b439fd4b4573a536baf48ad42186dc8bd50f3fab20553e36a   hatemm_qwen25vl_lora_sft.yaml
B      e767eba0ca6ff40679857e5efb759d72aa985629a9ece6584ea424ac2baba62f   lora_sft.sbatch
C      19c76b177f7dc883a9e03524450ad2e6cb302cdd0a6704d69da68a62188a06fc   enc3seed_lora_hatemm.sbatch
```

### Data + reused-machinery shas (prereg §1.1 / §6.2) — all matched

| file | sha256 | source of truth |
|---|---|---|
| `data/lora_sft/HateMM/train.json` | `93c6d3d1…93973a` | prereg §1.1 (743 rows) |
| `data/lora_sft/HateMM/val.json` | `9e103ed3…cc9ef` | prereg §1.1 (107 rows) |
| `data/lora_sft/HateMM/test.json` | `c12ad356…dfce8462` | prereg §1.1 (215 rows) |
| `RA-HMD/…/data/dataset_info.json` | `ebf14b47…7681b5` | prereg §1.1 (`hatemm_lora_{train,val,test}` keys) |
| `scripts/slurm/gen_embed_lora.sbatch` | `c76bb422…46f386` | prereg §6.2 (extraction, no edit) |
| `scripts/slurm/enc3seed.sbatch` | `dbe3fb81…0815c3d` | prereg §6.2 (12850 same-code anchor) |
| `scripts/slurm/enc3seed_zh_b3.sbatch` | `43792246…2385e29ad` | prereg §6.2 (B3 template) |
| `RA-HMD/…/mhc_qwen25vl_lora_sft.yaml` | `db371c18…2e69b1b52` | prereg §6.2 (copy source) |

## Transparency note (non-blocking)

The submit-executor task message transcribed artifact A's expected hash with a character that renders close to
`…b459…`; the binding prereg §6.1 (line 279) and the reviewer's independently-recomputed freeze block (review
line 165) both specify `…b439…`, and the file on disk hashes to `…b439…` — matching both binding documents
exactly. Per review Note 1, the freeze mechanism is "the sha256 table + submit-time re-hash," which is the
authority here; disk agrees with the prereg and the review with no contradiction. Freeze stands.

## Required statements

- Freeze is a read-only re-hash + this record + git commit; ZERO GPU/SLURM/Modal spent in Stage 1.
- No held-out test metric read; no verdict rendered.
- Not pushed to any remote.
