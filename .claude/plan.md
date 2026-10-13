# 计划：外部可控的前缀 KV 钉住（pin / unpin）

## 目标

给 SGLang 加两个由外部 controller 调用的接口，按 request id 钉住 / 释放一个已结束请求的缓存前缀（KV 路径 + 该节点的 mamba 状态），用于在 agentic workload 的 tool call 间隙保留 session 的上下文。服务端**只提供机制**（钉住、释放、兜底），钉不钉、何时释放由外部 controller（之后的 oracle / 真实策略）决定。

非目标：租期（lease）、controller 崩溃恢复、服务端内置的钉住策略、HiCache 卸载。

## 1. 接口设计

### 1.1 HTTP / IPC

| 接口 | 请求体 | 返回 |
|---|---|---|
| `POST /pin_prefix` | `target_rid`；可选 `expected_return_s`（预计多久后回来，供安全阀排序） | `status` ∈ {`pending`（请求未结束，结束时生效）, `already_pinned`, `already_finished`（请求已结束，拒绝）}；结束时的结果（`pinned` / `nothing_cached` / `over_budget`）只在日志里记录 |
| `POST /unpin_prefix` | `target_rid` | `status` ∈ {`unpinned`, `not_pinned`（幂等）, `force_released`（之前已被安全阀释放）} |
| `GET /prefix_pins` | — | 当前所有 pin 及总占用（调试用） |

`/generate` 的请求体加两个可选字段（避免时序竞争，controller 可二选一使用）：

- `pin_on_finish: bool`（加上可选的 `pin_expected_return_s`）：请求结束时立即钉住，不经过外部往返；等价于在结束前发了 `/pin_prefix`。
- `release_pin_of: str`：本请求被接纳时，自动释放该 rid 的 pin（见 3.3 交接）。

命名避开已有的 `unpin(req)`、`inc_full_pin`/`dec_full_pin`、embedding cache 的 `pin()`、`/ping`。io_struct 里 `BaseReq` 已有 `rid` 字段，所以新字段叫 `target_rid`。

### 1.2 配置（`arg_groups/fields/` 的 memory 命名空间，经 `get_memory()` 读取）

- `--enable-prefix-pin`（默认关）：关闭时接口返回错误，所有挂钩点不生效，行为与现状完全一致。
- `--prefix-pin-max-kv-fraction`（默认 0.5）：钉住的 KV token 总量上限，占 KV 池容量的比例。
- `--prefix-pin-max-mamba-fraction`（默认 0.5）：钉住的 mamba 状态数上限，占 mamba 池的比例。
- `--prefix-pin-release-order`：安全阀的释放顺序，`latest_return`（默认；按 `expected_return_s` 最晚者先放，未提供的视为无穷远）/ `oldest` / `largest`。

### 1.3 实现时的偏离（已实现版本）

- **不支持请求结束后才发 pin**：返回 `already_finished`。结束后的节点可能已被驱逐，或者 KV 还在但 mamba 状态已被驱逐，现有接口无法可靠判断。controller 需要在结束前表达意图（请求运行时发 `/pin_prefix`，或用 `pin_on_finish`）。
- **接纳侧安全阀只在"队头卡住且没有任何请求在运行"时触发**：有请求在运行时，它们总会结束并释放显存，这只是等待而不是死锁；等不等由 controller 权衡。
- **mamba 余量**：每次构建 prefill batch 前，保证空闲 + 可驱逐的 mamba 状态 ≥ 6（`MAMBA_STATES_PER_ADMISSION`：extra_buffer 每个请求 5 个，加上匹配时复制的 1 个）。

## 2. 实现结构

遵循 `large-class-style`：业务逻辑全部放进独立的协作类，Scheduler / TokenizerManager 里只做构造、分发和挂钩调用。

### 2.1 新模块 `mem_cache/prefix_pin.py`：`PrefixPinRegistry`

状态：

- `pins: rid -> PinEntry(node_id, dec_params, kv_tokens, mamba_states, pinned_at, expected_return_s)`
- `pending: rid -> PinIntent`（请求未结束时收到的 pin，或带 `pin_on_finish` 的请求）
- `recent_finished: rid -> node_id`，有界 LRU（比如最近 4096 个），用于"请求结束后才收到 pin"的情况
- `force_released: rid` 的有界集合，用于让 `unpin` 返回 `force_released`
- 总占用计数：钉住的 KV token 数、mamba 状态数

方法（全部只读写自身状态 + 调用 tree cache 的 `inc_lock_ref` / `dec_lock_ref`）：

- `request_pin(target_rid, expected_return_s, *, is_running)`
- `on_request_cached(req, node_id)`：请求结束、前缀插入后调用；记录 `recent_finished`，若有 pending 意图则立即加锁。
- `unpin(target_rid)`
- `handover(req)`：请求被接纳时调用，处理 `release_pin_of`。
- `release_for(need_kv_tokens, need_mamba_states)`：安全阀，按配置的顺序释放 pin 直到满足需求，返回释放了多少。
- `clear()`：`flush_cache` / tree reset 时调用。
- `snapshot()`：给 `/prefix_pins`。

### 2.2 挂钩点

| 位置 | 改动 |
|---|---|
| `mem_cache/unified_radix_cache.py` `insert_req` | 把插入结果的 `last_device_node` 作为返回值（新增返回值，原调用方不受影响）；未插入（`rotation_tail_declined`、`is_insert=False`、`skip_radix_cache_insert`）时返回 None |
| `mem_cache/common.py` `release_kv_cache` | `insert_req` 之后、`tree_cache.unpin(req)` 之前，若 registry 存在则调用 `on_request_cached(req, node_id)`。顺序很重要：先加 pin 锁再释放请求自己的锁，中间不会出现可被驱逐的空档 |
| `managers/io_struct.py` | `PinPrefixReqInput/Output`、`UnpinPrefixReqInput/Output`、`GetPrefixPinsReqInput/Output`；`GenerateReqInput` 加 `pin_on_finish`、`release_pin_of` |
| `managers/tokenizer_control_mixin.py` | 三个 communicator（`_COMMUNICATOR_SPECS`）+ 对应 async 方法；DP>1 时按 FanOut 合并（只有持有该 rid 的 rank 返回非 `not_found`） |
| `entrypoints/http_server.py` | 三个路由 |
| `managers/scheduler.py` | `init_prefix_pin_registry()`（`maybe_` 形式，按 `--enable-prefix-pin` 决定）；`init_request_dispatcher` 注册三个处理函数，函数体只委托给 registry；其余挂钩见第 3 节 |
| `flush_cache`（scheduler） | reset 之前调用 `registry.clear()` |
| `UnifiedRadixCache.sanity_check` | 存在 pin 时跳过"空闲时不应有锁"的检查（和现有 `session.any_holding_kv()` 的处理方式一样） |

## 3. 死锁与资源耗尽的规避

不做租期，靠以下四层保证"任何请求都不会因为 pin 而永远等下去"，最坏情况退化为不钉住（现状的 LRU）：

### 3.1 总量上限（源头控制）

`request_pin` 时检查加上本次后是否超过 `max_kv_fraction` / `max_mamba_fraction`，超过则返回 `over_budget`，不加锁。保证始终有一部分 KV 和 mamba 状态留给运行中和新进来的请求。

### 3.2 安全阀：资源不够时强制释放 pin

在所有"因资源不够而卡住或会崩溃"的地方，先尝试释放 pin，再走原有逻辑：

1. **接纳失败**：`_get_new_batch_prefill_raw` 中 `add_one_req` 返回 `NO_TOKEN`、且本轮没有接纳任何请求（队头卡住）时，调用 `release_for(队头所需)`；若释放了东西，**重建 PrefillAdder 后重试一次**（Adder 在构造时快照了预算，包括 `rem_mamba_slots`）。
2. **decode 显存不够**：`update_running_batch` 中 `check_decode_mem()` 失败时，先 `release_for(本步所需)` 再检查一次，仍不够才走 `retract_decode`。
3. **mamba 槽位分配**：命中前缀后复制 mamba 状态（match 时的 COW，目前是 assert）、`alloc_req_slots`、`cache_unfinished_req` 的捐赠，这几处槽位不够会直接崩溃。处理方式：**在构建 prefill batch 之前**，检查空闲 + 可驱逐的 mamba 槽位是否至少够"队头请求 COW + 新请求槽位"，不够就先 `release_for`。这一处需要实现时仔细核对 match 发生的时机（`init_next_round_input` 在接纳检查之前），是本计划中风险最高的一点。

被安全阀释放的 pin 记入 `force_released`，controller 下次 `unpin` 时能知道。日志里记录每次强制释放的原因和数量。

### 3.3 命中即交接（打破循环等待）

请求在 `/generate` 中带 `release_pin_of=<上一轮的 rid>`，在 `PrefillAdder._commit_prefill_admission` 加锁之后（请求已持有自己对该前缀的锁），调用 `registry.handover(req)` 释放那个 pin。这样"上一轮的 pin → 下一轮被接纳"的交接不依赖 controller 的再次往返，也不会出现同一前缀被锁两次后忘记释放的情况。

不采用"自动检测请求是否命中了被钉住的节点"：Rust tree core 没有向上找父节点的接口，按节点 id 也判断不了"命中了它的后代"，靠 controller 明确告诉服务端更可靠。

### 3.4 幂等与显式失败

- `unpin` 对不存在、已释放、已被强制释放的 pin 都返回成功语义的状态，不报错。
- `pin` 的所有失败都返回明确的 `status`，controller 可据此改用别的动作。
- 请求还在运行时收到 `pin` → 记为 pending，结束时生效；被 abort 或撤回（不插入缓存）的请求，pending 意图直接作废并在日志中记录。

## 4. 测试

1. **单元测试**（`test/registered/unit/`，按 `write-sglang-test` 的约定）：用小模型或 fake tree 测 `PrefixPinRegistry`：
   - pin / unpin / 幂等 / pending → 结束时生效
   - 总量上限拒绝
   - `release_for` 的三种释放顺序
   - handover
   - `clear`
2. **集成测试**（单 GPU，小的纯 attention 模型 + 一个混合 mamba 模型各一个）：
   - 钉住后制造缓存压力（大量无关请求），确认被钉住的前缀下一轮命中长度不变
   - 钉满上限后，新请求仍能被接纳（安全阀生效、不死锁、不崩溃）
   - decode 显存紧张时先释放 pin 而不是撤回请求
   - `flush_cache` 后 registry 为空
3. **回放实验**：loadgen 增加 controller 模式，按 trace 已知的 tool 时长决定 pin / unpin（先做"全部钉住 + 下一轮交接"的最简策略），在 r=4–5 上与 LRU 对比截断 JCT 和是否崩溃。

## 5. 步骤

1. `PrefixPinRegistry` + 单元测试（不接入调度器）
2. `insert_req` 返回节点、`release_kv_cache` 挂钩、pin/unpin 控制消息全链路（HTTP → tokenizer → scheduler），总量上限
3. 安全阀 3.2 的三处挂钩 + 交接 3.3 + `pin_on_finish` / `release_pin_of`
4. 集成测试
5. loadgen controller 模式 + 回放对比实验

## 6. 待确认

1. 安全阀默认的释放顺序：`latest_return`（需要 controller 在 pin 时提供 `expected_return_s`）是否作为默认？
2. 两个总量上限的默认值 0.5 / 0.5 是否合适？这直接决定 oracle 策略能钉住多少。
3. 是否需要支持 DP attention（dp_size>1）？当前实验都是 dp_size=1；支持的话主要是 FanOut 结果合并，工作量不大。
