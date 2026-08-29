# CLAUDE_STANCE_GATE — 偏离 D1

**日期**: 2026-08-17,标注进行中(batch 1-2 已完成,batch 3+ 尚未开始)
**记录时点**: 在计算任何指标之前。判定线、样本、输入、金标一律未动。

## 事实

`CLAUDE_STANCE_GATE_FREEZE.md` §5.5 规定:99 项分 6 批投喂,**同一标注 agent 用 SendMessage 续批**,
上下文连续。

batch 1、batch 2 按此执行,三个标注 agent 各完成 17+17 = 34 项。
向标注者 A 发送 batch 3 时,SendMessage 返回:

```
Agent "a64aea78657a6e60e" could not be resumed: No transcript found for agent ID: ...
```

即该 agent 的会话记录在两次完成之后不再可恢复,续批机制失效。
同时观察到两个 agent 在 34 项后累计上下文已达 ~133k tokens,按此斜率跑满 99 项会触发自动压缩,
压缩会丢掉第一条消息里的任务定义(Q1/Q2 的类别定义只在首条消息里出现过)。

## 处置

剩余 65 项(item_035 … item_099)改为:**每个标注者由新起的 agent 承接**,分两块投喂:
- chunk B = item_035 … item_067(33 项)
- chunk C = item_068 … item_099(32 项)

每块的 prompt **逐字包含 FREEZE §6 的完整标注 prompt 终稿**(任务说明 + Q1 定义 + Q1 校准段 +
Q2 六类定义 + 规则 + 输出格式),因此每个 agent 都是自足的,不依赖任何先前上下文。
三个标注者之间仍然零通信;chunk B 与 chunk C 的 item 互不重叠,可并行。

输出文件名改为 `annot_<tag>/chunk_b.jsonl`、`chunk_c.jsonl`(batch_1/batch_2 保持原名)。
`score_gate.py` 的 `load_rater` 相应从「固定 6 个 batch 文件名」改为「glob 该标注者目录下所有
`*.jsonl`」。除此之外评分脚本一字未改。

## 影响评估

1. **判定线、样本、金标、输入、主指标 32 行denominator:全部未动。**
2. 标注者的定义从「一个跨全部 99 项上下文连续的 agent」变成「三次独立会话、每次拿到完全相同的
   冻结 prompt」。对本轮要测的东西(Claude 在给定 prompt 下的立场判断能力、以及三个独立判断之间
   的一致性)这不是削弱:上下文连续反而会引入跨 item 的相互参照(FREEZE §6 的规则本来就禁止
   「Do not compare items to each other」),分块只会让这条规则更容易被遵守。
3. κ 的解释相应变为「三次独立标注 pass 之间的一致性」,而不是「三个长期一致的标注者之间」。
   这正是 F3 监督设计要用的量(实际用起来也是一次一批地调用),因此更贴近下游用途。
4. 前 34 项(batch 1-2)是上下文连续的,后 65 项不是。这一不均质性据实记录;
   结果文件会分别报出 item_001-034 与 item_035-099 两段的准确率与一致率,供检查是否有断点。
   该分段读数是**事后诊断**,不改判定,也不得被提升为主指标。
5. 单次执行红线保持:每个 item 每个标注者仍然只标一次,没有任何 item 被重标。
