# Distributionally stable rationale: identifiability check

截至 2026-08-31。本检查只审查 README 所定义的多 replacement sufficiency + necessity + lexicographic minimal mask，不重复文献查新，不实现、不训练。

## 裁定

**STOP。** 多 replacement 使 selector 对 benign donor 身份更稳健，但没有使 mask 对真实 hate span 可识别。对常见 frozen classifier，这个问题分别退化为：

- additive classifier：按 frozen classifier 的贡献做稀疏 covering/knapsack，通常是 top-contribution 点集；
- max-pool classifier：选中所有超过 necessity margin 的峰，只有一个峰时严格退化为 top-1；
- topic/broadcast classifier：选择接近全视频、全视频或任意位置的等质量大 mask；
- replacement 未被 frozen classifier 判到 normal margin：可行域可能为空。

因此，它比 V26 增加的是 **joint/set-valued intervention 和 donor robustness**，不是新的 localization identifiability。若 `F` 把 topic、片头、字幕或单个视觉伪相关点当作证据，当前优化会更稳定地复制该错误，而不会纠正它。OOF 防止同视频训练泄漏，但不改变这个结论；student 只会把 selector 的解蒸馏成最终输出。

## 统一写法

先考虑 binary mask。令 frozen classifier 的 positive margin 为 `F(z)`，原视频 margin 为 `M=F(x)`，sufficiency 阈值为 `S`（README 中是原 prediction 的固定 fraction），normal margin 为 `N`。每个 replacement `r` 要同时满足：

`F(x_keep(m,r)) >= S`,

`F(x_remove(m,r)) <= N`。

只要 `F(x)>=S` 且每个完整 replacement 都有 `F(r)<=N`，全一 mask 总是可行：kept input 是 `x`，removed input 是 `r`。所以 lexicographic 最小化不是在“是否存在 hate witness”与“没有 witness”之间识别，而是在一个保证包含全视频的可行集里寻找 frozen classifier 的最小证书。

反之，negative-train **video label** 不保证 `F(r)<=N`。若某个 replacement 是 classifier false positive，并且 `F` 对原正证据单调，则全一 mask 的 removed input 已违反 necessity，减少 mask 只会重新加入原正证据，整个可行域为空。故必须区分“label-negative donor”与“frozen-margin-certified replacement”；README 当前只保证前者。

## 反例一：additive classifier

设 replacement 为零贡献，classifier margin为

`F(x)=sum_t a_t`, 其中 `a_t>=0`。

则两个约束等价为

`sum_t m_t a_t >= S`,

`sum_t m_t a_t >= M-N`。

所以 selector 只是寻找最小质量 mask，使所选贡献达到

`C=max(S, M-N)`。

### 单峰：严格 top-1

取 `T=10, M=.99, S=.792, N=.10`，贡献为

`a_1=.90`, 其余九帧各 `.01`。

此时 `C=.89`，唯一最小解是 `{1}`。它同时 sufficient 且 necessary，但只证明 frozen classifier 的最大贡献帧控制 bag logit；不证明这一秒覆盖真实事件。对连续 mask 的 `L1` 最小化，该问题是 fractional knapsack，会优先把质量放到贡献/单位 mask 最大的帧，因而比 binary 版本更直接地产生单点或“一个整帧加一个 fractional 边界帧”。

### 分布式 topic：接近全视频

取 `T=10` 且每帧 `a_t=.10`，于是 `M=1, S=.8, N=.1`。任何 9 帧都是最小 binary 解：sufficiency 要至少 8 帧，necessity 要至少 9 帧。真实 signal 即使只是全局 topic，selector 也会返回 90% 视频；十个等价解由 tie-break 任意决定位置，而不是由 hate span 决定。

### 非连续多峰：不是 span

取第 2、9 帧贡献各 `.46`，其余八帧各 `.01`，仍有 `M=1, S=.8, N=.1`。阈值 `C=.9`，最小解首先选 `{2,9}`。这是两个不相邻点；约束没有连续性或完整事件覆盖语义。若 tie 后还需少量质量，它只会再取任一高贡献帧，不会填满两峰之间的 span。

对一般 additive replacement，记 `d_t^r=a(x_t)-a(r_t)`、`B_r=F(r)`，每个 `r` 只产生一个线性 covering 约束：

`sum_t m_t d_t^r >= max(S-B_r, M-N)`。

多 replacement 因而把单一 knapsack 变成 robust covering LP/ILP。它可能选择能同时覆盖多个 donor 的若干点，但仍不提供 temporal correctness；连续 `L1` 解倾向于 LP 的稀疏极点，而非完整 span。

## 反例二：max-pool classifier

设 `F(x)=max_t a_t`，replacement score 为零。necessity 要求每个未选原帧均满足 `a_t<=N`，所以所有 `a_t>N` 的帧都必须进入 mask；sufficiency另要求所选集合至少包含一个 `a_t>=S` 的帧。

- 若仅一个帧超过 `N` 且其 score 超过 `S`，唯一最小解就是该帧：严格 top-1。
- 若多个分散峰超过 `N`，最小解是这些峰的非连续集合，不是事件边界。
- 若 topic 被广播到所有帧且每帧 `a_t>N`，necessity 强迫选择全视频。

所以“sufficiency + necessity”并不会一般性避免 top-1；它只把 top-1 扩展成“所有能让 max-pool 保持 positive 的冗余峰”。这仍完全由 `F` 的 pooling geometry 决定。

## 反例三：topic 与 broadcast classifier

### 平均 topic

设每个原帧携带相同 topic value `q=1`，replacement 为零，`F` 是时间平均。取 `T=10, S=.8, N=.1`：

- kept margin 是 `k/10`，所以 `k>=8`；
- removed margin 是 `(10-k)/10`，所以 `k>=9`。

最小 mask 是任意 9 帧。稳定 replacement 无法区分这 10 个位置；固定 tie-break只会制造可复现的位置偏置。

### OR/max broadcast

若同一 topic 在每帧都足以触发 positive，`F(x)=max_t a_t` 且所有 `a_t>N`，necessity要求删除每一个原帧，唯一最小解是全视频。该解完美满足多 replacement 稳定性，却没有任何 localization 信息。

### 真正的 video-level shortcut

若 `F` 的决定由经过全序列编码后广播到各 timestep 的全局 latent 决定，对局部 replacement 的响应可以是：任意保留片段仍 positive，而任意残留片段也使 removed input positive。此时只有全视频 mask可行；若完整 benign replacement 仍被 `F` 判正，则连全视频 mask也不可行。selector不能从这种 classifier 中创造原本不存在的时间分解。

## 多 replacement 没有修复的歧义

1. **共同伪相关仍稳定。** 如果所有 benign donors 都移除了“新闻演播室”“字幕模板”或某个说话人，而 `F` 依赖这些特征，该伪相关点会对所有 replacements 同时 sufficient/necessary，恰好获得最高 stability。
2. **donor support 不是 intervention validity。** 从不同 negative videos 取 replacement 只能改变 donor distribution；跨视频 hybrid 的时序、身份、音色与语义断裂仍可能成为 `F` 的证据。
3. **冗余真实 witness 不可唯一识别。** 两个片段各自都能使 bag positive时，necessity会要求二者都被删除；得到的是 classifier 的所有冗余触发器，不是 minimal causal event。若阈值较松，又可能只选其中一个。
4. **单一共享 mask 与 disagreement tie-break 定义冲突。** README 先要求同一个 `m` 对所有 `r` 满足约束，随后又最小化“replacement之间的mask disagreement”。若只有一个联合 mask，disagreement 恒为零；若实际是每个 replacement 各求一个 mask，则需要重新定义如何形成最终 membership，以及第一阶段究竟最小化各自质量还是联合质量。当前 lexicographic problem 数学上未完整定义。
5. **mask artifact 仍可被利用。** 对非线性 `F`，kept/removed hybrid 的边界或缺失模式可能主导 margin。两个互补 forward 只证明该 mask 控制模型输出，不证明控制的是语义 evidence。

## 与 V26 的识别性比较

V26 的单帧 deletion effect 问“替换这一秒会使 frozen output 改变多少”；当前候选问“哪一个最小帧集合能在多个 replacement 下同时保留并消除 frozen output”。后者确实能表达：

- 多帧的联合/冗余效应；
- 同一 mask 对 donor 变化的稳健性；
- 可重算的 kept/removed bag-level constraints。

但在上述 additive 和 max-pool 情形，它分别严格化简为 contribution ranking/covering 与 thresholded peak collection；在 topic/broadcast 情形则化简为近全视频或全视频。因此它没有增加“选中的帧比 V26 更接近真实 hate span”的识别条件，只把 V26 的局部敏感性扩展成 frozen classifier 的最小集合证书。若 V26 失败来自 classifier evidence 与真实 temporal ownership 错配，本候选没有切断这条失败路径。

## 决策含义

当前候选不应进入 premise 实现。README 已要求 rationale mass 不得单点或近全视频，但那是 **事后退化过滤器**，不是从 video labels 推导出的识别条件；它还会排除上述模型忠实但不定位的解，却不能使剩余解变成正确 span。

只有在新增独立于 `F` attribution 的 train-only 可检验结构假设，并能解析排除以下三类解时，才值得重新送审：

1. 单个最大贡献帧满足全部约束；
2. broadcast/topic 导致近全或全视频 mask；
3. 对所有 replacements 稳定的共同伪相关点。

单纯增加 replacement 数、margin、mask temperature、连续性正则或 student loss都不能提供该识别性，也不会把本次 **STOP** 改成 conditional go。
