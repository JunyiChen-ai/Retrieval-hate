# Independent novelty review A

截至 2026-08-31。裁定 `CONDITIONAL GO 6.7/10`。

Gate 1 PASS；Gate 2窄 PASS；Gate 3对task-specific graph construction给窄 PASS。Reference edge与independently queried within edge不再由同一absolute category确定性派生；negative-video全窗benign certificate提供公共reference，within edges提供local order，二者通过`r-i-j` cycles与同一potential耦合。

实现前blockers：普通Huber不保证唯一potential，应换处处严格凸pseudo-Huber或确定性tie-break；五级选择必须固定model/tokenizer/真正single-token choices，并在五选项内重新归一概率；增加`y_ij^derived=y_ir-y_jr` control、second-medoid premise、零方差fail-closed和triangle innovation报告。

最强反例是VLM输出curl-free但由profanity/topic驱动的错误utility；Hodge只检可积性，不验证hate语义。
