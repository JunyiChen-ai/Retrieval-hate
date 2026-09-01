# Independent novelty review B

截至 2026-08-31。裁定 `STOP 4.8/10`。

Gate 1 PASS；Gate 2窄 PASS；Gate 3 FAIL。在理想difference model `y_ab=f(a)-f(b)` 下，common reference给`z_i=f(i)-f(r)`；换任意reference只给全部target score加同一常数，pooled AP/ROC与within ROC均不变。因此negative-train medoid在rank指标下只是gauge，不是load-bearing语义。Fixed-reference LLM ranking已有RefRank近邻。

同一模型下`y_ij=y_ir-y_jr`，within query没有新信息；完整图只在噪声下做标准robust aggregation。若far/blank reference改变结果，来自reference identity/context bias而非Hodge理论。Topic/style可让每视频所有target相对reference得到同一`c_v`，within edges全零，pooled很好但within=`.5`。

普通Huber唯一性、choice-token归一化也未定义完整。即使修复，它最多成为工程baseline，不能恢复当前novelty；若让negative labels进入多reference threshold/risk约束则已是新候选，且必须重新审查。
