# Independent novelty review A

截至 2026-08-31。裁定 `STOP 4.2/10`。

- Gate 1：PASS；HodgeRank 与 swap consistency 可 adaptation。
- Gate 2：窄 PASS；未检出 Hodge decomposition/cycle-selective pair inference 用于 hateful-video detection/localization，但 training-free VLM hate localization 已被 LELA/TANDEM占用。
- Gate 3：FAIL；当前是 LELA式双窗 absolute classification、order-swap filter与标准图最小二乘的组件拼接。

致命代数是每条一致 label 同时给 unary 与 edge，且所有 label 都满足 `y_ij=u_i^(ij)-u_j^(ij)`。若同一 window 的 unary 跨边一致，`z_i=u_i` 同时令 edge/unary loss为零；若不一致，解只是对重复 unary 做图平滑。唯一性主要来自 absolute unary anchor，而不是 HodgeRank。

稳定 all-BOTH/all-NEITHER 分别产生整段 `+1/-1`：AB/BA一致、triangle curl为零、图全连通、unary覆盖100%，pooled可很好而positive-video within严格为`.5`。一个非零 triangle 也不能识别哪条 edge 错；删除全部参与 edge会偏向 ties并制造断图。Swap一致性只检稳定性，不检正确性。

若重开，必须改为不由 absolute label确定性派生的独立 cardinal pair preference，使用独立固定 anchor，采用 robust Hodge regression而非整三角删边，并加入 full-unary-only/all-tie/zero-curl-wrong controls后重新审查。
