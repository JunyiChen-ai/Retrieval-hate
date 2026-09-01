# 淘汰：Policy-gated semantic recurrence premise probe

截至 2026-08-31。validation-only、零训练，不读 test。检查 POWA typed primitive
similarity 是否能成为 visual semantic-neighbor propagation 的 load-bearing gate。

固定规则：以同一 concat local score 为 seed；visual-only control 使用 CLIP cosine；
candidate affinity 是 visual cosine 与 POWA 六维 primitive-probability cosine 的等权
平均；各自取 top-15%、softmax temperature 10，再用相同固定 Gaussian smoothing。
全部只作为 frozen POWA multiset ordering upper bound。

只有 policy-gated rule 在 HMM/HCS 都比 visual-only within ROC 高 `.010` 且比 POWA
高 `.020`，才进入 novelty review；否则归档，不能把 graph propagation 包装成新机制。

结果 `STOP_BEFORE_NOVELTY`：HMM policy-gated `.62624`，但 shuffled-primitive
`.64754` 更高；HCS visual-only `.55867`，policy-gated 降到 `.55167`，shuffled
`.55547`。typed primitive gate 非 load-bearing，未训练、未读 test。
