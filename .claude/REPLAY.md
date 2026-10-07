# Session 回放（replay）规则

适用于 `dataset_csv/{openhands,hermes,openclaw}.csv`，列的定义见 `README.md`。

目标是在服务端重现录制时的负载形态：
- 每轮 prefill 的新增 token 数、decode 的 token 数与录制一致；
- 前缀结构与录制一致：每轮只 prefill 新增部分，前面的上下文都命中缓存；
- 轮与轮之间的等待时间与录制一致。

模型输出的**内容**无关紧要，因为回放不评测任务质量。

## 0. 核心思路

```
第 1 轮：ctx = 初始 prompt 的 token
        生成 dummy 输出，强制恰好 decode_num_tokens 个 token
        ctx += dummy 输出（作为 assistant 回复留在历史里）
        等待 turn_toolcall_latency_ms
第 k 轮：ctx += 第 k 行的 prefill_token_ids（通常是上一轮的 tool 结果）
        生成 dummy 输出 → ...
```

与 context-gc 的回放方式相比：
- context-gc 每轮发送录制的完整请求体，历史里放的是**录制的回复**，模型这次生成的输出被丢弃。所以每轮都要把录制回复再 prefill 一遍。
- 本方案让 dummy 输出本身留在历史里。它的 KV 在 decode 时就已经算好，下一轮可以直接命中，每轮只需 prefill 真正新增的部分。

## 1. 用到的列

| 列 | 用途 |
|---|---|
| `prefill_token_ids` | 本轮 decode 之前要追加到上下文里的 token。已离线算好（Qwen3.8 的 chat template 加上 SGLang 的 `/tokenize`），回放时直接使用，客户端不需要再分词 |
| `decode_num_tokens` | 本轮强制生成的 token 数 |
| `prompt_reset` | true：`prefill_token_ids` 是完整的上下文，回放时丢弃之前的 ctx |
| `prev_response_in_history` | false：上一轮的回复没有进入历史，回放时丢弃上一轮的 dummy 输出 |
| `turn_toolcall_latency_ms` | 本轮回复结束到下一轮请求开始的等待时间；最后一行为空 |
| `prompt_available` | false：没有原文，见 2.2 |
| `decode_token_ids` | 录制回复的 token。dummy 回放不用它，只作参考或用于对照实验 |

切分点在 token 层面的含义（以 Hermes 的 dr_005 为例）：

```
第1行 prefill_token_ids（1922 个）:
  <|im_start|>system\n…<|im_end|>\n<|im_start|>user\n…Task id: dr_005<|im_end|>\n<|im_start|>assistant\n<think>
第1行 decode（48 个，回放时为 dummy）:
  \n\n</think>\n\nLet me start by reading …<tool_call>…</tool_call><|im_end|>
第2行 prefill_token_ids（136 个）:
  \n<|im_start|>user\n<tool_response>\n{…PROMPT.md 的内容…}\n</tool_response><|im_end|>\n<|im_start|>assistant\n<think>
```

规则如下：
- 每一行的 prefill **以 `<think>` 结尾**，decode **以 `<|im_end|>` 结尾**；
- 下一行的 prefill **以 `\n` 开头**（也就是上一轮 `<|im_end|>` 之后的换行）。

两个切分点都紧挨着特殊 token（`<think>` 和 `<|im_end|>`），特殊 token 两侧的文本是独立分词的，所以各段首尾相接之后，与整体分词的结果完全一致。这一点已对三个文件的全部行逐 token 校验过，见 README。

## 2. session 筛选

### 2.1 默认使用的"干净" session

| agent | 干净 session | 请求数 | 排除的 session |
|---|---|---:|---|
| OpenHands | 53（DR 20、SWE 20、TB 13） | 1656 | 7 个 TB session 没有请求记录 |
| Hermes | 36（DR 20、TB 16） | 906 | SWE 的 20 个 session 没有 transcript；TB 中 3 个压缩 session 缺 parent；含 `none` 行的 session；`854fa26a…`（transcript 与录制不符） |
| OpenClaw | 56（DR 17、SWE 20、TB 19） | 1663 | `92c17b1c`、`8f434c61`、`b1b270e3`、`27ff97d2` 计时不全、请求重复 |

判定条件：session 内所有行 `prompt_available = true`，除最后一行外所有行 `turn_toolcall_latency_ms` 都非空，并且不在上表的排除名单里。

### 2.2 要使用缺原文的 session 时（可选）

按 context-gc 的 "shape corpus" 方式回放：
- 只用 token 数，每轮新增长度 = `recorded_context_tokens[k] − recorded_context_tokens[k−1] − recorded_decode_tokens[k−1]`；
- 用该 session 固定的随机 token 填充。

这样前缀结构、负载和等待时间都是对的，只是内容是合成的。主要用于 Hermes 的 SWE session，因为它有 token 计数。

## 3. 每个回放副本的准备

### 3.1 salt（必需）

到达轨迹会反复抽到同一个 session。真实场景里不同任务不会共享对话前缀，所以同一轨迹的多个副本不能互相命中缓存。

**规则**：每个副本有一个唯一的 salt（例如 `replica=<id>` 分词后的若干 token）。在每个 reset 行的 `prefill_token_ids` 里，第一条 user 消息的开头（紧跟在第一个 `<|im_start|>user\n` 之后）插入这串 salt token。

- 之后的非 reset 行都接在这个上下文后面，自然继承 salt；
- 各副本之间仍然共享 system 段（以及 tools）前缀，和真实场景一致；
- 插入 salt 后，上下文会比录制长几个 token，可以忽略，在报告中说明即可。

常用 token id（Qwen3.6 和 Qwen3.8 相同）：`<|im_start|>` = 248045，`<|im_end|>` = 248046，`<think>` = 248068，`<|im_start|>user\n` = `[248045, 846, 198]`。

### 3.2 tools schema（可选）

- OpenHands 的 `prefill_token_ids` 里已经包含真实的 tools schema；
- Hermes 和 OpenClaw 没有 schema 原文，所以不含。录制的 `recorded_context_tokens` 比数据集的上下文长度多一个常数：Hermes 3,536（DR 运行为 3,431），OpenClaw 约 3,009。

如果希望共享前缀的长度与真实情况一致，可以在 reset 行的 `<|im_start|>system\n` 之后插入一串**该 agent 所有 session 共用**的、长度等于这个常数的占位 token。不插也可以，只是共享前缀会短约 3K token，需要在报告中说明。

## 4. 回放主循环（token 层面）

客户端为每个 session 维护两份上下文：
- `ctx_after`：上一轮 dummy 输出之后的上下文，也就是 ctx + output + `<|im_end|>`；
- `ctx_before`：上一轮 decode 之前的上下文。

每一行的处理：

```
1. 确定基准：
     prompt_reset = true                     → ctx = salt(prefill_token_ids)
     prev_response_in_history = true         → ctx = ctx_after  + prefill_token_ids
     prev_response_in_history = false        → ctx = ctx_before + prefill_token_ids
2. 发送请求：POST /generate
     input_ids      = ctx
     max_new_tokens = decode_num_tokens
     ignore_eos = true, 不设 stop, temperature = 0（或任意值）, stream = true（用于测 TTFT）
3. ctx_before = ctx
   ctx_after  = ctx + output_ids + [<|im_end|>]      # <|im_end|> = 248046
4. 如果不是最后一行：sleep(turn_toolcall_latency_ms)，然后处理下一行
```

说明：
- **必须用 SGLang 原生的 `/generate`，并传入 `input_ids`**。不能走 chat API：chat API 会把 dummy 文本重新渲染、重新分词，结果和生成时的 id 不一致，只要错一个 token，后面的前缀缓存就全部失效。
- **dummy 输出后面要手动补一个 `<|im_end|>`**，因为 `ignore_eos` 生成的结果不会以它结尾。下一行的 prefill 从紧跟其后的 `\n` 开始。这个 `<|im_end|>` 会在下一次请求时 prefill，只多 1 个 token。
- dummy 输出中间出现 `<|im_end|>` 或其他特殊 token 也没关系：它们只是历史里的普通 token，不影响前缀结构。
- `decode_num_tokens` 与录制值一致。由于计数口径不同，它和 `len(decode_token_ids)` 可能相差 0–3 个 token，这里以 `decode_num_tokens` 为准。

## 5. 例外情况

### 5.1 `prompt_reset = true`

出现在每个 session 的第一行，以及历史被压缩或改写的轮次：OpenHands 34 次压缩加 31 次插入提醒消息，Hermes 3 次，OpenClaw 42 次。

处理：丢弃之前的 ctx，改为 `ctx = salt(prefill_token_ids)`。

这时**不需要**手动控制缓存：SGLang 的 radix cache 会自动复用新旧上下文的最长公共前缀（通常是 system 段加 tools，再加上未被改写的那部分历史），只有其余部分需要重新 prefill。这正是压缩或改写在缓存上的真实代价。

### 5.2 `prev_response_in_history = false`

上一轮的回复没有进入历史，回退到 `ctx_before` 再拼接。分两种情况：
- **OpenHands 有 175 行 `prefill_token_ids` 为空**：上一轮回复为空，OpenHands 把它丢掉，原样重发同一个 prompt。回放时就重发 `ctx_before`，几乎全部命中缓存。
- **OpenClaw 有 31 行**：OpenClaw 把一个经过轻微改写的回复放进了历史，改写后的版本就在本行的 prefill 里。回退之后拼接，等价于重新 prefill 这段回复，与录制时一致。

（OpenHands 插入提醒消息的 31 行已经按 token 级 reset 处理，归入 5.1。）

### 5.3 最后一行

- 生成完成后 session 结束，没有等待，KV 可以释放。
- OpenHands 的最后一行常常是 `finish` 这个 tool call（53 个 session 中有 39 个）。录制中它之后没有下一次请求，所以回放时也不再等待。

### 5.4 deadline

如果要报告"按时完成的吞吐"，给每个 session 设一个预算，例如任务的 timeout，从 session 到达时开始计时。超时就立刻停止：断开正在进行的请求，释放 KV，记为未完成。**不能**跳过超时后的轮次继续计时。

只画 JCT 时可以不设。但在超过容量（越过 wall）之后，JCT 会随测量窗口一起增长，所以 wall 之后的点不应当作稳态数值引用（见 `eviction_ladder/README.md`）。

## 6. 等待时间：用 `turn_toolcall_latency_ms`，不要用单个 tool 耗时相加

`turn_toolcall_latency_ms` = 本轮回复结束到下一轮请求开始的时间。不管有几个 tool call、并行还是串行，harness 自身的开销也都包含在内，这正是服务端看到的 session 空闲时间。

**单个 tool 的耗时不能直接相加**，各 harness 的执行方式不同：
- **Hermes**：
  - 只读 tool（`read_file`、`search_files`）**并行**执行，例如 `0c21c0ee…` 第 14 次请求的两个 `read_file`，分别是 0.000 → 0.827 s 和 0.001 → 0.828 s；
  - 只要有 `terminal` 或 `execute_code`，就**串行**执行，相邻两次之间固定间隔 1.0 s，例如 `128d95e6…` 第 3 次请求的 4 个 curl，等待共 9.4 s；
  - 单个 tool 时，`turn_toolcall_latency_ms` 正好等于这个 tool 的耗时。
- **OpenHands**：tool 串行执行，另加约 50 ms 的 harness 开销（p99 为 330 ms）。
- **OpenClaw**：同一轮的 tool 并行执行，`turn_toolcall_latency_ms` 约等于最长那个 tool 的耗时。

`turn_toolcall_latency_ms` 缺失的行（OpenClaw 的 19 行）已经用 tool 耗时估算并填好，`turn_toolcall_latency_source = estimated_from_tools`。

**数据修正**：Hermes 在 Terminal-Bench 上的这批运行，记录的请求开始时间实际上是上一次请求的结束时间（753 次中有 683 次）。CSV 已改为用上一轮最后一个 tool 结束的时间作为开始时间。`analysis/profiling_audit/raw/runs.jsonl` 中这些 run 的 `llm_spans` 仍是旧值，所以 profiling 报告里 Hermes TB 的 LLM 时间偏高。

## 7. 开跑前必须做的检查

1. **混合架构的前缀缓存**：Qwen3.6 和 Qwen3.8 都是线性注意力加全注意力的混合架构，SGLang 必须在匹配边界上保存了 mamba 状态才能命中前缀（我们之前因此加了 `--mamba-scheduler-strategy extra_buffer`；在 0.5.21 里这个参数叫 `--mamba-radix-cache-strategy extra_buffer`）。

   要验证的是：**dummy 输出进入历史之后，下一轮的请求能否在 ctx + output 处命中**。方法是取一个 session，顺序回放 3–5 轮，检查每轮返回的 `cached_tokens` 是否约等于 `len(ctx_after)`。如果 SGLang 不在输出末尾保存状态，每轮都会退回到最近的保存点重新 prefill，"只 prefill 新增部分"的前提就不成立了。
2. **逐请求的保真检查**：回放时服务端返回的 `prompt_tokens` 应当等于客户端发送的 `len(input_ids)`，`completion_tokens` 应当等于 `decode_num_tokens`。超过 1% 的请求不符时，这一组结果作废。
3. **关闭投机解码**（MTP、EAGLE）：接受率取决于内容，dummy 文本会让 decode 速度失真。
4. **每次运行都用新启动的服务器**，同一组对比使用同一镜像和同一类节点。

## 8. 伪代码

```python
IM_END = 248046

def replay_session(rows, replica_id, t_arrival):
    ctx_before = ctx_after = None
    for i, row in enumerate(rows):                       # 按 turn_index 排序
        if row.prompt_reset:
            ctx = insert_salt(row.prefill_token_ids, replica_id)          # 3.1 / 5.1
        elif row.prev_response_in_history:
            ctx = ctx_after + row.prefill_token_ids
        else:
            ctx = ctx_before + row.prefill_token_ids                      # 5.2
        if deadline_passed(t_arrival): return CUT                          # 5.4
        out = generate(input_ids=ctx, max_new_tokens=row.decode_num_tokens,
                       ignore_eos=True)                                    # 4
        check(out.prompt_tokens == len(ctx), out.completion_tokens == row.decode_num_tokens)   # 7.2
        ctx_before, ctx_after = ctx, ctx + out.output_ids + [IM_END]
        if i + 1 < len(rows):
            sleep(row.turn_toolcall_latency_ms / 1000)                                  # 6
    return DONE
```

## 9. 方案能保证什么、不能保证什么

| | 与录制一致？ |
|---|---|
| 每轮新增 prefill 的 token 数 | 一致，多出的只有 salt 和每轮补的 1 个 `<|im_end|>` |
| 每轮 decode 的 token 数 | Hermes / OpenClaw 是录制的精确值；OpenHands 由回复文本计数得到，**不含 reasoning，偏短** |
| 轮与轮之间的等待 | 一致（`turn_toolcall_latency_ms`） |
| 前缀缓存的结构 | 与"理想 harness"一致，见下文说明 |
| LLM 本身的耗时 | 不要求一致：它取决于回放时的负载，正是要测的量 |
| 任务质量、agent 对变慢的反应 | 无法反映：teacher forcing 假设轨迹与服务快慢无关 |

几点说明：
- **前缀缓存偏乐观**：录制时，harness 把回复放回历史时不一定原样保留（例如 OpenClaw 改写 id 和参数，模板也会重新渲染），所以回复那一段不一定能命中缓存；而本方案总能命中。数据里已知的改写（`prev_response_in_history = false` 和 token 级 reset）已经照录制处理；剩下的偏差对所有对比策略相同，不影响策略之间的比较。
- **dummy 的内容不能被任何组件读取**：
  - 按内容做决策的 eviction 策略，以及我们的 tool latency 预测器，都必须从 CSV 读取**录制的** `response`；
  - 如果预测器的 suffix 接在 agent 上下文之后，那么前缀是 dummy、suffix 引用的是录制的 tool call，在报告中说明即可。
- **模板**：token 是用 Qwen3.8 的模板生成的。它与录制所用的 Qwen3.6 分词器相同，模板只有细微差别（例如 system 段多一句约 38 token 的 reasoning-effort 说明），这里忽略。
