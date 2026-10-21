# 交接：Tool call 耗时预测（TokenCake 基线 vs. 我们的 LLM 方法）

读者：要在**新数据集**上复现并比较这两种方法的 coding agent。

本文自成一体。代码位于 `analysis/latency_pred/`。原数据集上的完整结论见 `REPORT_route1.md` 和 `research_log.md`。

---

## 1. 任务定义

agent 发出一个 tool call 之后、tool 开始执行之前，预测这次执行会花多长时间。预测结果用于决定该 session 的 KV cache 在 tool 执行期间是保留还是换出。

- **预测时刻**：模型已经生成完带 tool call 的 assistant 回复，tool 尚未开始执行。
- **输出**：13 个绝对时长档位（bucket）之一，用字母表示：

  ```
  A <0.1s | B 0.1–0.2s | C 0.2–0.4s | D 0.4–0.8s | E 0.8–1.6s | F 1.6–3.2s
  G 3.2–6.4s | H 6.4–12.8s | I 12.8–25.6s | J 25.6–51s | K 51–102s | L 102–205s | M ≥205s
  ```

  档位边界为 `100·2^k ms`，k = 0..11（`scripts/run_openhands.py::bucket_of`）。
- **两个目标**：
  1. 总体精确准确率（预测档位 = 真实档位）**略胜** TokenCake；
  2. 长尾识别（真实耗时 ≥1.6 s，即 ≥F；以及 ≥6.4 s，即 ≥H）的召回率**显著高于** TokenCake，同时精度不低于它。

### 1.1 预测目标：单个 tool call 还是整个 turn（新数据集请按 turn 做）

原实验的标签是**单个 tool call 的耗时**。后来确认，KV 决策真正需要的是**整个 turn 的 tool 阶段时长**，也就是从回复结束到下一个 LLM 请求开始的时间（`dataset_csv/*.csv` 中的 `turn_toolcall_latency_ms`）。两者在多 tool 的 turn 上会有差别：

| harness | 同一 turn 内多个 tool 的执行方式 | 单个 tool 的档位与 turn 档位不一致的比例 |
|---|---|---|
| OpenClaw | 并行；turn 时长约等于其中最长的那个 | 2.6% |
| Hermes | 只读 tool（read_file、search_files）并行；只要有 terminal 或 execute_code 就串行，且相邻调用之间固定间隔 1 s | 9.2% |
| OpenHands | 串行，另加约 50 ms 的 harness 开销 | 42.8%（主要是 50 ms 开销让 0.2–0.4 s 的 file_editor 越过了 0.4 s 的边界） |

**建议**：
- 新数据集上，**以 turn 为单位，每个 turn 只问一次 LLM**，标签用 turn 的 tool 阶段时长。
- 最后一个 turn 之后没有 tool 阶段，不参与评估。
- 下文描述的是原有的逐调用（per-call）实现，改成逐 turn 的方法见第 6 节。

---

## 2. 基线：TokenCake

实现见 `scripts/tokencake_baseline.py`，逐条结果见 `results/tokencake_baseline.jsonl`。

- **类型划分**（来自 Continuum，arXiv 2511.02230 §5.1）：
  - 普通 tool 的类型就是函数名；
  - shell 类 tool（原数据集中名为 `terminal`、`bash`、`execute_bash`）的类型是命令字符串的第一个词，例如 `curl`、`python3`、`cd`。
  - **新数据集要把它的 shell tool 名加进这个列表**，比如 OpenClaw 的 `exec`、Hermes 的 `terminal`。
- **预测**（来自 TokenCake，arXiv 2510.18586 §4.1）：
  - 每种类型维护一个 EWMA：每观测到一次执行时间 `t_obs`，就更新 `t ← 0.5·t_obs + 0.5·t`（α = 0.5，取自论文的参数表）；
  - 预测的档位 = 当前 EWMA 值所在的档位。
- **冷启动**：该类型还没有历史时，用所有已结束调用的系统级 EWMA（同样 α = 0.5）；连系统级历史都没有时，用 1 s。
- **在线且因果**：
  - 所有 session 的调用按真实墙钟时间合成一条事件流；
  - 每个调用在**开始时**做预测，只能用**在此之前已经结束**的调用来更新历史（包括其他 session 的调用）；
  - 同一时刻既有结束又有开始时，先处理结束。
- 这个基线故意做得简单：它就是一个"弱但现实"的基线。

---

## 3. 我们的方法：LLM suffix 查询（`route1`）

### 3.1 查询机制

实现见 `scripts/run_openhands.py::chat` 和 `research/variants.py`。

- **消息结构**：
  - 前缀 = agent 当时的完整上下文：system + 所有历史消息 + 本 turn 带 tool_calls 的 assistant 消息。这部分已经在 SGLang 的前缀缓存里，不需要重新计算；
  - 后面追加**一条 user 消息（suffix）**，内容是本次的问题和证据，格式见 3.3。
- **请求参数**：`/v1/chat/completions`，`max_tokens = 1`，`temperature = 0`，正则约束 `regex = "[A-M]"`，`logprobs = true`，`top_logprobs = 20`，`chat_template_kwargs = {"enable_thinking": false}`。
- **输出**：
  - argmax 的字母作为预测档位；
  - 再从 top-20 的 logprob 中取 A–M 这 13 个字母，重新归一化，得到 13 个档位上的概率分布，用于长尾判定（见第 4 节）。
- **必须写 "Do not call any tool."**：在 agent 上下文中，模型更倾向于生成新的 `<tool_call>`，不写这句话的话，正则约束下的字母分布几乎是随机的。
- **不分类、不推理**：
  - 让 LLM 先输出类别、再输出档位的"两步法"（v0）效果更差；
  - 先推理几句再回答（think1）会让分布过尖，无法用阈值来调节长尾；
  - 问"会不会超过 X 秒"这类 Y/N 问题（binq）效果也更差。

### 3.2 历史证据：与 TokenCake 拥有相同的信息

实现见 `research/harness.py::Evidence.for_call`。所有证据都是**在线且因果**的：只用在当前调用开始之前已经结束的调用（任何 session 都可以）。

| 证据块 | 内容 |
|---|---|
| 同类历史 | 同类型调用的数量 n、档位分布的百分比、EWMA 及其档位（与 TokenCake 完全相同），以及最近 4 次同类调用 |
| 本 session 历史 | 本 session 最近 4 次调用（任意类型）；或者只取同类型的最近 4 次（hist5 使用后者） |
| 相似调用 | 在最近 3000 个已结束的调用中，按 `difflib.SequenceMatcher` 的字符串相似度检索，取 6 个去重后的最相似调用。**不使用 LLM 或 embedding** |
| timeout | 如果参数里带有 timeout，就给出上限，写作 "so at most \<bucket\>" |

每条历史调用写成 `<规范化的调用文本> → <档位> (<秒数>)`。

**命令规范化**（`norm_cmd`，作用很大）：
- file_editor 写成 `{command} {path}`；
- shell 命令去掉 `-H/--header/-A/--user-agent` 及其参数，去掉 `2>/dev/null`、`2>&1`、`>/dev/null`，再截断到 110 字符。

不做规范化的话，curl 的 URL 会被 header 挤出截断范围，长尾召回会明显下降。

### 3.3 按 tool 分流的 suffix（`route1`）

- **路由规则**：稳定类 tool（原数据集中为 `file_editor`、`task_tracker`、`think`、`finish`）使用 **hist5**，其余（shell）使用 **hist3**。
  - 路由只看 tool 名，不需要 LLM 分类。
  - **新数据集要重新确定哪些 tool 属于"稳定类"**。建议按开发集上各类型耗时的离散程度来定（例如 ≥95% 的调用落在同一个档位）。
- **hist3**（shell 类，模型自由判断）：

  ```
  How long will this tool call take to run?
  Tool call: <tool> <原始参数 JSON，超过 600 字符截断>

  <档位表>

  Earlier calls of the same kind (`curl`) in this environment: n=30
    bucket distribution: B 10% · D 40% · E 43% · F 7%
    recent average (EWMA): 0.521 s → D
    most recent:
      <规范化调用> → D (0.714 s)   ×4
  This session, most recent calls first:
      <规范化调用> → B (0.194 s)   ×4
  Most similar earlier calls:
      <规范化调用> → D (0.696 s)   ×6
  This call's timeout: 20 s (so at most I).

  Do not call any tool. Answer with the letter of the bucket only.
  ```
- **hist5**（稳定类，按历史判断）：
  - 同类历史放在最前面，并明确写出"X% 落在某个档位"；
  - 本 session 历史只保留同类型的调用；
  - 结尾加一句：`Follow the history of the same kind unless this call itself gives a clear reason to differ (different target, scale, flags or timeout).`
- **为什么要分流**：
  - 只用 hist3 时，file_editor 的准确率只有 80%（TokenCake 是 98%），模型会被同 session 里其他类型的调用干扰；
  - 只用 hist5 时，"按历史判断"这句话又压低了 shell 命令的长尾概率；
  - 分流同时修好了这两个问题。

两个真实的 suffix 示例和所有变体的说明见 `prompts/suffix_template.md`。

---

## 4. 评估协议

实现见 `research/evaluate.py` 和 `research/eval_split.py`。

- **划分**：session 按 benchmark 分层，用固定随机种子 0 对半分成开发集和测试集（`harness.split_sessions`）。
  - 所有 prompt 迭代只看开发集；
  - 测试集只评估最终版本；
  - 评估时可以跑全部 session，再拆成两半分别报告。
- **指标**：
  - 精确准确率、±1 档准确率；
  - 长尾 ≥F 和 ≥H 两档各自的召回率和精度。
- **长尾判定规则**：P(档位 ≥ 阈值档位) ≥ θ，概率来自 3.1 中的 13 档分布。报告两种口径：
  1. θ 在开发集上选定：取使精度 ≥ TokenCake 精度的最小 θ，再原样用于测试集；
  2. 固定 θ = 0.25：与 TokenCake 无关，更干净。
- **TokenCake 的长尾判定**：预测档位 ≥ 阈值档位，也就是它的 argmax。
- **显著性**：配对 bootstrap，2000 次，种子 0，报告 LLM − TokenCake 之差的 95% 置信区间。
- **与 TokenCake 在完全相同的调用集合上比较**：只保留延迟记录可靠的调用；参数 JSON 被截断导致 HTTP 400 的调用直接跳过（原数据集中有 8 个）。
- **记录开销**：TTFT。做法是顺序发送请求，每个 turn 先用 `base + "."` 预热一次，让上下文进入缓存，再测量 suffix 查询的 TTFT。

---

## 5. 原数据集上的结果

数据：OpenHands 的 60 个 session（DeepResearch、SWE-Bench Pro、Terminal-Bench 各 20 个），共 2010 次调用，逐调用标签。

| | TokenCake | route1 + Qwen3.6-35B-A3B | route1 + Qwen3.8-27B |
|---|---:|---:|---:|
| 精确准确率 | 70.5% | 72.8%（Δ 95% CI [+0.9, +3.8]） | **78.8%**（[+6.9, +9.8]） |
| ±1 档准确率 | 86.0% | — | 92.3% |
| ≥1.6 s 召回 / 精度（θ = 0.25） | 62.0% / 50.6% | 67.4% / 51.4% | **78.7% / 55.1%** |
| ≥6.4 s 召回 / 精度（θ = 0.25） | 48.0% / 36.4% | 72.0% / 35.5% | **92.0% / 53.5%** |
| suffix 查询 TTFT 中位数 / P90 | — | 213 / 376 ms（SGLang 0.5.9） | 281 / 585 ms（SGLang 0.5.21） |

- Qwen3.8 在开发集和测试集上分别都达成了两个目标。
- Qwen3.6 在测试集上的长尾精度比 TokenCake 低 2–4 个百分点。
- 两个模型的 TTFT 是在不同 SGLang 版本上测的，不能直接比较。

---

## 6. 迁移到新数据集的步骤

1. **准备数据**：每个 tool call（或 turn）需要以下字段：
   - session id；
   - tool 名和参数；
   - 开始和结束时间戳，以及延迟是否可靠的标志；
   - **调用发生时 agent 的完整消息上下文**，供 LLM 查询时作为前缀。

   可以直接参考 `dataset_csv/` 的格式和 `export_csv.py`：它包含 OpenHands、Hermes、OpenClaw 三种 harness 的解析逻辑，以及 turn 级的 `turn_toolcall_latency_ms`。
2. **调整类型划分**：把新数据集的 shell tool 名加入 `ctype`、`norm_cmd` 和 `call_text` 的判断条件。
3. **先跑 TokenCake 基线**（便宜，没有 GPU 也能跑）。确认在线因果的事件流是正确的：每个调用的预测只依赖在它开始之前已经结束的调用。
4. **重新确定稳定类 tool**：在开发集上，看各类型耗时档位的集中程度。
5. **改为逐 turn 预测**（推荐）：
   - 每个 turn 只查询一次，suffix 的 "Tool call:" 部分列出本 turn 的全部 tool call；
   - 证据改为 turn 级：同类历史以本 turn 中"最慢的类型"或"类型组合"为键，历史条目写成 `<组合> → <turn 档位>`；
   - 路由规则：本 turn 全是稳定类 tool 时用 hist5，否则用 hist3；
   - TokenCake 的逐 turn 版本：先对每个 tool 用 EWMA 预测，再按该 harness 的执行方式组合，并行取 max，串行取 sum；Hermes 的串行还要加上 1 s ×（n − 1）。

   这一步尚未在原数据集上验证过，结果需要和逐调用版本对照报告。
6. **先在开发集上迭代**，最后对测试集只跑一次，并按第 4 节报告两种 θ 口径。

---

## 7. 服务端配置与已知的坑

- **SGLang 版本**：
  - Qwen3.8-27B **必须使用 SGLang ≥ 0.5.19**。0.5.9 能加载这个模型，但输出是坏的：复制不出 `/workspace` 这样的 token，85% 的预测都是 "I"，准确率只有 6.2%，而且不会报任何错误。
  - 本机使用的是 0.5.21，装在独立的 venv `/mnt/vdb/lrq/sgl521` 里（`uv pip install --prerelease allow sglang==0.5.21 decord`）。
- **启动参数**（0.5.21）：

  ```
  python -m sglang.launch_server --model-path <snapshot> --served-model-name <name> --tp-size 2 \
    --context-length 262144 --mem-fraction-static 0.85 --trust-remote-code --attention-backend fa3 \
    --mamba-radix-cache-strategy extra_buffer --enable-cache-report --disable-custom-all-reduce
  # 环境变量：SGLANG_ENABLE_JIT_DEEPGEMM=0（本机 nvcc 为 12.8，而 DeepGEMM 要求 ≥ 12.9）
  ```

  - **`extra_buffer` 是必需的**。Qwen3.6 和 Qwen3.8 都是线性注意力与全注意力的混合架构，默认的 `no_buffer` 只能在 8192 token 的分块边界上命中前缀缓存，每次查询都要重算最多约 8K token。0.5.9 中这个参数叫 `--mamba-scheduler-strategy`。
  - 加载这两个模型都需要安装 `decord` 包。
- **必须验证前缀缓存确实生效**：检查每次 suffix 查询返回的 `cached_tokens`（需要开启 `--enable-cache-report`），原实验中命中率的中位数约为 92%。
- **档位必须用字母，不能用数字**：Qwen 会把数字拆成单个字符的 token。如果还要做分类，类别和档位不能共用同一套字母，否则模型会直接照抄。
- **参数 JSON 被截断的调用**：发送请求会得到 HTTP 400，直接跳过并单独计数。

---

## 8. 文件索引（`analysis/latency_pred/`）

| 文件 | 内容 |
|---|---|
| `scripts/tokencake_baseline.py` | TokenCake 基线 |
| `scripts/run_openhands.py` | 档位定义、`chat()` 查询、数据加载（`load_samples`）。模型和地址通过环境变量 `LP_MODEL` / `LP_URL` 指定 |
| `research/harness.py` | 数据划分、在线证据（`Evidence`）、命令规范化（`norm_cmd`）、运行器。用法：`harness.py route1 --split all --workers 1 --ttft --tag <tag>` |
| `research/variants.py` | 所有 suffix 变体，包括 `hist3`、`hist5`、`route1` |
| `research/evaluate.py`、`research/eval_split.py` | 评估 |
| `prompts/suffix_template.md` | suffix 积木块、所有变体及其开发集结果、真实示例 |
| `REPORT_route1.md`、`research_log.md`、`qwen38/eval.txt` | 原数据集上的结果和研究过程 |
| `../../dataset_csv/` | 三种 harness 的 turn 级数据集，以及导出脚本 |
