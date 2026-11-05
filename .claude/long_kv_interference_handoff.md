# 交接：长 KV request 对同 batch 短 request 的 decode 干扰实验

## 目标与范围

实现一个客户端驱动的微实验：固定 8 个短上下文 request，另外加入 8 个 stressor，逐步增加 stressor 的 context length，观察短 request 的 TPOT 和完成时间如何变化。

核心问题：在 request 数量相同的情况下，长 KV 是否会拖慢同一 instance 内短 request 的 decode，以及影响有多大。

直接使用现有 inference server、streaming API 和 prefix cache。**不修改 engine 内部，不实现 scheduler barrier。将完整 prefix cache 命中视为已基本省掉 prefill，接受少量残余处理和调度误差。** 本任务只实现实验，不实现 session 分 pool 或跨 instance KV 迁移。

## 实验矩阵

只有两类实验，最短 stressor 点就是混跑参考点，不额外增加第三类实验。

| 实验 | 被测短 request | Stressor |
| --- | --- | --- |
| A：独跑 | 8 个，固定短 context | 无 |
| C(L)：混跑 | 同样 8 个，同样 context | 8 个，context length 为 L |

建议默认值，均需支持命令行配置：

- 短 request 数量：8。
- Stressor 数量：8。
- 短 request context length：512 tokens。
- 短 request 输出长度：256 tokens。
- Stressor context sweep：128、512、2048、8192、16384、32768；按模型和显存能力调整。
- 重复次数：5。
- Streaming 开启；固定 decoding 参数，尽量使用 greedy decoding。
- 禁止因 EOS 提前结束；使用当前 server 支持的 ignore_eos 或等价设置。
- 配置足够的 context 上限和并发上限，允许 16 个 request 同时 decode。

混跑中，stressor 应一直运行到短 request 全部完成。其输出 token budget 应明显大于短 request，建议从 2048 开始；若仍提前结束，增大预算。短 request 结束后取消 stressor。

## Prompt 构造与 prefix cache

1. 用被测模型的 tokenizer 构造目标 token 长度的 prompt，优先使用 completion 接口或 token-ID 输入，减少 chat template 的额外长度差异。
2. 8 个短 request 使用不同 prompt，但在所有 sweep 点和重复实验中复用同一组内容。
3. 8 个 stressor 也使用不同 prompt；避免大段共享前缀导致物理 KV 被共享，从而改变预期的显存占用。
4. 精确记录最终输入 token 数。若接口自动加模板，记录实际长度即可，不必为几个 token 的偏差改 engine。
5. 预热短 request：提交与正式测量**完全相同的输入**，生成 1 token，等待完成。正式请求仍提交原始输入，不附加预热生成的 token。
6. 启用 prefix cache。若现有日志或响应方便提供 cached-token 数就记录；没有该字段也不阻塞实验。完整 prefix hit 足以作为近似纯 decode 的实现。

## 每次 trial 的执行流程

### A：短 request 独跑

1. 准备短 request 的 prefix cache。
2. 并发提交 8 个短 request，持续读取 streaming 输出。
3. 等待全部完成，保存每个 request 的时间戳和指标。

### C(L)：与 stressor 混跑

1. 在没有其他活跃请求时，准备 8 个短 request 的 prefix cache。
2. 并发启动 8 个长度为 L 的 stressor。
3. 等待每个 stressor 都已经返回至少 4 个输出 token，确认它们已经进入 decode；不再额外长时间等待。
4. 并发提交 8 个短 request，持续读取两组请求的 streaming 输出。
5. 等待短 request 全部完成，然后取消并清理 stressor。
6. 保存原始时间戳、指标、实际输入长度，以及每个 stressor 在短 request 开始和结束附近已生成的 token 数。

这样 stressor 的长 prefill 发生在短 request 加入之前，短 request 本身通过 prefix cache 省掉主要 prefill 工作。无需严格同步每一个 decode step。

每个 trial 前确保上轮活跃请求已结束。为避免不同点残留的 cache 状态影响实验，有现成 cache-clear 接口就使用，否则可重启 server；不为此新增 engine 功能。清理后重新准备短 request cache。

先做一次端到端 smoke run，再执行完整 sweep。重复实验可随机化 L 的顺序，并保存随机种子。

## 时间戳与指标

客户端统一使用 monotonic clock，记录提交时间、首个输出时间、每个非空输出事件时间、最后输出时间。忽略 role-only、空文本等非 token 内容事件。

每个短 request 至少输出：

- TTFT：首个输出时间减提交时间。
- TPOT：`(最后输出时间 - 首个输出时间) / (实际输出 token 数 - 1)`。
- E2E latency：最后输出时间减提交时间。这是本实验中 prefix 已缓存条件下的 E2E。
- Decode duration：最后输出时间减首个输出时间。
- 实际输入、输出 token 数。

Streaming 事件可能合并多个 token，不要直接把事件数当作 token 数。优先使用 server 返回的 token ID、token 数或 usage；否则用 tokenizer 统计完整输出。只有接口确实提供逐 token 时间戳时，才计算逐 token 间隔的 P95；不把 chunk 间隔标成 token 间隔。

每个 trial 先对 8 个短 request 求平均 TPOT 和平均 E2E，再在重复 trials 之间计算均值与标准差。保留 request 级原始结果，便于观察加入时刻差异。

## 分析方式

记 A 的短 request 平均 TPOT 为 T_A，C(L) 为 T_C(L)，最短 sweep 点为 L_min。

- 相对独跑的总干扰：`T_C(L) - T_A`。
- 总 slowdown：`T_C(L) / T_A`。
- 固定 request 数后，KV 增长的额外干扰：`T_C(L) - T_C(L_min)`。
- 相对最短 stressor 点的 slowdown：`T_C(L) / T_C(L_min)`。

E2E 同样报告独跑值和 sweep 曲线。不要把总混跑开销全部归因于长 KV；相对 L_min 的变化才用于描述 stressor context 增长的影响。

注意 L 是 stressor 的初始输入长度，它在生成期间还会继续增长。记录实际生成量即可，不需要实现固定 KV 长度的特殊 kernel。

## 运行检查

- 确认 server 的配置允许这 16 个请求一起 decode；若已有 scheduler 日志能显示 running request 数，顺手保存。
- 整个 sweep 固定模型、GPU 数、TP、KV dtype、graph 和 attention backend 配置。
- 保存 GPU 型号、server 版本、启动参数与实验参数。
- 正式有效点不应发生 OOM、KV eviction、preemption 或 recomputation。遇到容量不足的点，标注无效及原因，继续其他点；不把异常点混进干扰曲线。
- 若 stressor 在短 request 完成前退出，将 trial 标为无效并增加其输出预算重跑。
- 每个 trial 保存两组请求的活跃时间区间，检查测量期间确实重叠。

## 交付内容

在现有项目中选择合适位置实现，提供：

1. 可通过 CLI 运行的 benchmark，支持 endpoint、model/tokenizer、两组请求数、context sweep、输出长度、重复次数、输出目录等参数。
2. README：server 启动配置、smoke run 和完整 sweep 命令。
3. 原始事件 JSONL、request 级 CSV、trial 级和跨 trial 汇总 CSV。
4. 图一：横轴 stressor context length，纵轴短 request 平均 TPOT；C(L) 曲线带 trial 间误差条，A 用水平参考线。
5. 图二：同样布局，纵轴为短 request 平均 E2E latency。
6. 简短结果说明，报告相对 A 和相对最短 stressor 点的 slowdown；若尚无可用 GPU/server，则交付实现与可执行命令，明确哪些结果尚未实测，不生成虚构数据。

实现优先保持简单：客户端编排、prefix-cache 预热、streaming 时间记录和结果分析足够完成第一版。
