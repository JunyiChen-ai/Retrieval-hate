# Independent novelty review B

截至 2026-08-31。裁定 `STOP 5.0/10`。

Gate 1 PASS；Gate 2 窄 PASS，未检出 Hodge/curl inference进入目标任务；Gate 3 FAIL。逐边恒有 `y_ij=u_i-u_j`，所以 occupancy unary已完全决定 edge flow，Hodge部分没有新增独立局部观察。consistent时它是冗余，inconsistent时只是标准 least-squares smoothing。

最强反例是所有边稳定输出 BOTH：交换一致率1、curl为零、retained unary与connected coverage均100%，唯一解却是positive video整段常数，within为`.5`。交替 BOTH/NEITHER 时所有 edge仍为零，curl看不到 absolute occupancy在共享节点上的冲突。

Hard cycle deletion还会把一个坏 edge同三角中的正确 edge一起删掉，产生 tie-selection bias与不唯一断图。它不是ensemble或calibration，但属于 output-dependent selective edge gating，不能声称完全无routing。

若只做诊断，必须补全 fail-closed coverage与 all-tie controls；若争取 novelty，relational observation必须独立于 unary，并证明固定 unary 时 edge-only information能改变正确时序。
