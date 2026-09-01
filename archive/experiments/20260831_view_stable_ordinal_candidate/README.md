# 淘汰：view-deletion-stable ordinal distillation

截至 2026-08-31。仅完成独立 proposal novelty review，未实现、未训练、未读 test。
查新结论 `STOP`（约 3/10），见 `NOVELTY_SCOUT.md`。

四个完整二元 view 下，“删除任一 view 后多数方向不变”等价于普通 3-of-4
majority，不提供新 certification；2011 年 multiview semi-supervised ranking 已直接用
跨-view 一致 pair 产生 ordinal pseudo-label。3/4 pair relation 还可能形成 cycle，
scalar student 无法全部满足。项目 upper bound 中 HMM/HCS 的 all-view 虽过 feasibility，
但分别不胜 audio 和 VERA/concat+VERA 强单源/少视图，不能主张多数机制更可靠。
