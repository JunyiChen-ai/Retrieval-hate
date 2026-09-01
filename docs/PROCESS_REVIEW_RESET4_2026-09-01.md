# Process Review RESET4

截至 2026-09-01。审查只读取流程、状态、post-test analysis与RESET3三轮正式结果，不审代码。

## Verdict

**RESET**。RESET3错误地在同一个 marked-splat/duration-field family 内消耗整个窗口。关闭duration-field主方向，下一epoch回到MultiHateLoc已有四语料共同的modality selection/fusion failure；不做premise，除非新方法依赖现有artifact完全未覆盖的新信息源。

## Diagnosis

首版HMM within `.727709`中，leave-one-video-out position-only已达`.709890`，去共同位置轮廓后只剩`.591156`。HCS raw/position-only为`.534081/.530324`，去位置后splat/point仅`.512890/.510991`。Mass conservation使HMM within降`.041247`且HCS三项全降；完整validation选择只改善HMM pooled，HCS仍降；dense-negative对HMM pooled仅`+.000568/+.004705`且损失within，HCS pooled明显下降。证据足以关闭通过renderer、kernel、top-K、regularization、negative loss或配置修补duration-field的路线。

RESET3另一个计数错误是把validation-selected original当独立方法失败。它是必要的配置/基线校正和正式test记录，但没有result-relevant mechanism，不占candidate failure窗口。正确口径：RESET3内result-relevant failures为mass-conserving与dense-negative两次；累计连续method performance failure为`5`。本次review既已触发并完成，不撤销停机，直接开启RESET4窗口`0/3`。

## Mandatory corrections

1. STOP marked-splat/duration-field主方向及其renderer、boundary、kernel、top-K、regularization、dense-negative、margin和配置变体。
2. 回到默认starting architecture MultiHateLoc；下一failure brief只引用已有四语料test error artifact：DMS权重与test-GT最佳单模态匹配率`.216/.333/.375/.323`，fused胜全部单模态比例`.345/.159/.042/.154`，best-branch oracle相对fused within缺口`.106/.171/.211/.106`。
3. 不新增premise；已有test evidence足以定义共同failure。只有全新信息源且现有artifact完全不能判断局部信息时才允许一次廉价premise。
4. 固定顺序：已有test failure → 一次novelty三门 → 最小实现 → 一次technical review → 独立训练HMM/HCS → 立即test。通过后才扩EN/ZH与多seed。
5. 同一mechanism family一次正式失败后最多允许一个由test error analysis直接支持的corrective iteration；再次失败则本epoch关闭。
6. 只有result-relevant method iteration计入failure窗口；validation配置补选、checkpoint复评、matched control和重复评测只记录，不冒充candidate。

## Direction decision

- **STOP**：marked-splat/duration-field及同family续调。
- **CONTINUE**：整体研究，MultiHateLoc作为默认trainable starting architecture。
- **PAUSE**：新teacher、raw statistic、producer及与已知共同failure无关的跨任务adaptation。

最终单一裁定：**RESET**。
