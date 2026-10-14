# Online queueing-latency predictor：交接文档

写于 2026-10-14，基于 2×H100 NVL 上的实验。设计背景见 `.claude/queuing_predictor.md`（本文档是它之后的实现与结论）。下一站：H200 节点。

## 1. 一句话结论

在线预测器已经端到端跑通：SGLang 暴露调度器快照，sidecar 维护 decode 长度 CDF pool、用步级模拟器做蒙特卡洛，loadgen 在发送每个请求时记录预测值。r=4.5、30 分钟 run 上，对排队 ≥ 1 秒的请求：

| | 中位相对误差 | Spearman | 60 秒以上那档 |
|---|---|---|---|
| 线上发送时预测（loadgen 列） | 12.6–14.2% | 0.86–0.89 | 偏短 30–60% |
| 离线回放，真实 decode 长度（oracle） | **7.0%** | **0.96** | **−0.3%** |
| 离线回放，影子预测 + oracle | **3.7%** | **0.98** | −0.4% |

模拟器本身已经和 SGLang 对齐（oracle 几乎完美，包括长尾）。线上剩余误差主要来自 **decode 长度的不确定性**，这是下一步要攻的方向，不是模拟器。

## 2. 实现了什么

### SGLang（`python/sglang/srt/`）
- `GET /scheduler_snapshot?finished_since=N&max_waiting=K`：每个 DP rank 一份快照（结构见 `managers/scheduler_snapshot.py`）。逻辑在 `managers/scheduler_components/snapshot_inquirer.py`，Scheduler 只做构造和转发。
  - running（含正在做最后一个 prefill chunk 的请求，`prefilling=true`）、chunked、waiting（FCFS 顺序）、正在跑的 step、显存（available / evictable）、new_token_ratio、配置。
  - 每个 waiting 请求带 `prefix_len` 和 `prefix_path = [[边界处的 prefix 长度, 驱逐排位], ...]`，驱逐排位为 None 表示被 lock、不会被驱逐。
  - 每个 running 请求带 `shared_prefix_tokens`（路径上 `lock_ref > 1` 的 token 数，它结束时不会释放）。
  - finished ring：每个请求结束时记录一次（output_streamer 里的 hook），`finished_since=N` 增量拉取，`finished_seq_tail` 用来检测漏拉。
  - 快照构建在这个负载下 ≤ 2 ms（`build_ms` 字段）。
- `POST /match_prefix {"input_ids": [...]}`：对还没发出的请求做只读 prefix 匹配，返回 `matched_tokens` 和 `prefix_path`。
- tree core（`mem_cache/unified_cache/unified_tree_core.py`）新增三个**只读**方法：`peek_prefix_path`（和真实 `match_prefix` 结果一致，但不 split 节点、不刷新 LRU）、`eviction_ranks`（按"LRU + 只能驱逐叶子"给每个可驱逐节点算排位）、`shared_locked_tokens`。只支持 `UnifiedRadixCache`，其他 cache 返回 None。
- 单测：`test/registered/unit/managers/test_snapshot_inquirer.py`；`test/registered/unit/mem_cache/test_unified_radix_cache_unittest.py::TestUnifiedRadixPeekPrefixLen`（需要 GPU）——包括"排位 ≥ 实际驱逐量的边界恰好存活"、"释放量 = 上下文 − 共享部分"这两条用真实 `evict()`/`dec_lock_ref()` 验证的性质。

### Sidecar（`controller/queuing_predictor/`，只依赖 aiohttp + numpy + msgspec，不 import sglang）
| 文件 | 作用 |
|---|---|
| `server.py` | `POST /predict`、`GET /health`、`GET /pool`；后台每 0.25 秒轮询；`--shadow-log`、`--dump-snapshots`、`--log-predicts` |
| `client.py` | 轮询快照、finished 游标、快照历史（`snapshot_before(t_send)`）、调 `/match_prefix` |
| `pool.py` | 全局 decode 长度 pool（最近 20000 个，≥ 500 才预测，之前返回 `ready:false`）；`sample_remaining` 按 D > g 条件采样 |
| `simulator.py` | 步级模拟器（从 `loadgen/qp_sim2.py` 移植），见第 4 节的建模要点 |
| `predictor.py` | P 次蒙特卡洛（默认 8），输出 mean / p50 / p90 |
| `costs.py` / `profile_costs.py` | step cost 模型和离线 profile（第 3 节） |
| `eval_shadow.py` | 影子预测按 rid 和 loadgen.csv 对比 |
| `replay.py` | 离线回放：pool 采样 vs oracle（第 5 节） |
| `env_h100.sh` | 只适用于 H100 那台机器的环境脚本，H200 上不要用 |

### 其他
- `loadgen/loadgen.py --predictor URL`：每个 turn 发出时并发调用 `/predict`（不推迟发送），写入 `turn_predicted_queuing_latency` 列。
- `loadgen/dry_run.sh`：设置 `PREDICTOR_COSTS=<文件>` 即开启预测器；文件不存在时先在第一个（空闲的）server 上 profile。新增 `MEM_FRACTION_STATIC`、`PREDICTOR_PORT`（默认 32000；**不要用 31000+**，那是各 server 的 NCCL 端口）。
- `vistool`："Queue prediction" 一节：实际 / 预测排队时间的 CDF 和时间序列，相对误差（只算实际 ≥ 1 秒）的 CDF 和时间序列。

## 3. 在 H200 上：offline profile 流程

step cost 是按 GPU 拟合的，H100 的参数（`costs/qwen3.8-27b_h100nvl.json`）**不能**用在 H200 上。

### 方案 A：让 dry_run 自动 profile（推荐）
```bash
# 在 H200 计算节点上；dry_run 会在缺 CUDA_HOME 时自己 module load cuda/13.0
CORPUS=loadgen/hermes_clean.jsonl RATE_PER_MIN=4.5 DURATION=1800 \
RUN_NAME=qp_r4.5_h200 \
PREDICTOR_COSTS=$PWD/controller/queuing_predictor/costs/qwen3.8-27b_h200.json \
bash loadgen/dry_run.sh
```
- 文件不存在时，dry_run 先在第一个 server 上跑 `profile_costs`（H100 上约 22 分钟，H200 应该更快），写好 cost 文件，再启动 sidecar 和 loadgen。之后的 run 直接复用这个文件。
- `MEM_FRACTION_STATIC` 在 H200 上用默认的 0.85（cookbook 的 H200 单卡配置）。0.80 只是 H100 的折中。
- profile 日志：`logs/<run>/profile_costs.log`；原始数据点：`<cost 文件>.points.json`。

### 方案 B：单独 profile
```bash
bash loadgen/launch_servers.sh            # 起好 server，保持空闲
python -m controller.queuing_predictor.profile_costs --url http://127.0.0.1:30000 \
    --model-tag Qwen3.8-27B --output controller/queuing_predictor/costs/qwen3.8-27b_h200.json
```
- profile 要求 server **空闲**（有请求在跑会直接退出）。
- prefill：在 (extend, prefix) 网格上发单个请求，用 server 端的 `prefill_finished_time − forward_entry_time`，按 chunk 拆分特征拟合 `p0 + Σ(p1·E + p2·E·(P+E/2)/1e6)`。
- decode：B 个并发流式请求，取"所有请求都在 decode"的时间窗内每 token 间隔，拟合 `d0 + d1·B + d2·ctx_Mtok`。网格会按 `max_total_num_tokens` 和 `max_running_requests` 自动裁剪；H200 的 `max_running` 是 52，比 H100 的 17 大很多，decode 网格会多几个点。
- 中途崩溃时可以用打印出来的点重拟合，不用重跑：`--from-log <profile_costs.log> --chunk 32768`。
- H100 上的拟合质量：prefill 中位误差 3.7%，decode 0.2%。负载下实测 decode 每步与模型之比中位 1.01；prefill 实测偏大 1.1–1.4×（部分是 overlap 模式下 `prefill_finished_time` 晚记一步的测量假象）。

### H200 上要检查的
1. 用 cookbook 页 `docs/cookbook/autoregressive/Qwen/Qwen3.8-27B.mdx` 的 H200 配置（`loadgen/launch_servers.sh` 已经按它写好，偏离项在脚本头部注释里）。
2. 先 `curl /scheduler_snapshot` 看 `config.max_running_requests`、`memory`，和 H200 之前的数字（KV 约 957k，max_running 52）对一下。
3. r=4.5 在 2×H200 上是"拐点"负载，排队比 H100 少很多；若排队样本太少，可以提高 `RATE_PER_MIN`。

## 4. 建模要点（按发现顺序，每条都由数据驱动）

1. **prefix 不能用命中率估计。** 排队请求在 admission 时的真实命中率只有约 0.14，没排队的请求约 0.89。用"最近结束请求的平均命中率"会严重高估命中、低估 prefill 量（实际 extend 是估计的 2 倍）。
2. **排队期间 prefix 会被 LRU 驱逐。** 只看快照时刻的匹配（静态假设），extend 仍被低估 1.6 倍；53% 的排队请求在等待中丢了（部分）prefix。模拟器跟踪"快照后超出 available 的分配量 V"，请求第一次被尝试 admit 时，prefix 截到排位 ≥ V 的最深边界。之后 admission 时的命中率预测 0.137，实际 0.139。
3. **只有被显存卡住的队头受保护。** 有空槽但显存不够时，scheduler 每轮都会对队头做真实匹配（刷新 LRU），所以队头 0% 丢失；被槽位上限卡住时 scheduler 在匹配前就返回，队头不受保护。模拟器在"第一次真正尝试 admit"时冻结 prefix。
4. **共享 prefix 不会随请求结束释放。** system prompt（384/1600/1664 token）被其他运行中请求 lock 着，请求结束只释放"上下文 − 共享部分"；反过来 admit 一个 prefix 已被别人 lock 的请求也不消耗显存。这个偏差平时两头抵消，但在"很多请求结束 + 一个巨大的队头"的场景下会让队头提前 45 秒进去。修复后 oracle 长尾从 −41% 变为 −0.4%。

## 5. 评估方法

- **发送时预测**（loadgen 列）：router 实际可用的预测——请求还没进 SGLang，用发送前最近一次快照 + `/match_prefix`。
- **影子预测**（`predictions.csv`）：请求第一次出现在 waiting 队列时、用 SGLang 内部真实队列做的预测，衡量"信息最完整时模拟器能多准"。
- 二者差距 = 发送时信息不完整（快照延迟、快照后才到的请求看不到）的代价。
- 离线回放（需要 run 时开了 `--dump-snapshots` 和 `--log-predicts`，dry_run 默认已开）：
  ```bash
  python -m controller.queuing_predictor.replay logs/<run> --costs <cost 文件>
  python -m controller.queuing_predictor.eval_shadow logs/<run>
  ```
  `replay` 对每个目标算 P=8 / P=64 / oracle（真实 decode 长度来自 loadgen.csv），结果写 `replay.csv`、`replay_eval.json`。回放的 P=8 和线上原值一致（12.7% 对 12.6%），可以信任。

各轮 run（都在 H100 机器的 `logs/` 下，对应 `/mnt/network_drive/lrq/sglang_logs/`）：

| run | 改动 | 发送时中位误差 / 偏差 / Spearman |
|---|---|---|
| `qp_r4.5_h100_20261014_082938` | 第一版，命中率估计 prefix | 21.5% / −21% / 0.90（影子） |
| `qp_r4.5_h100_prefixpeek_20261014_092858` | 静态 peek prefix | 23.3% / −18% / 0.92 |
| `qp_r4.5_h100_evictrank_20261014_122151` | LRU 驱逐排位 | 12.7% / −3% / 0.91 |
| `qp_r4.5_h100_dump_20261014_132101` | 加快照存储；首次 oracle 回放 | 12.6% / −3% / 0.87；oracle 8.6% |
| `qp_r4.5_h100_shared_20261014_140508` | 共享 prefix 记账 | 14.2% / −3% / 0.89；oracle 7.0% / 0.96 |

## 6. 已知局限与下一步

1. **decode 长度预测（最大的杠杆）。** 线上 vs oracle 差约 7 个百分点；长尾偏短的主因是"一个占大量上下文的请求刚开始 decode，被采样成很快结束"。P 从 8 到 64 只改善约 1 个百分点，p90 覆盖率 68–80%（应为 90%）。方向：用同一 session 前几轮的 decode 长度做条件（离线 `DecodeModel` 做过分组）。**注意**：replay 里同一条轨迹被重放约 3.2 次、长度完全相同，只能用同一 replica 之前的轮次，否则等于泄露答案。
2. **1–10 秒的短等待**相对误差大（发送时 30–45%），主要受当前 step 剩余时间和调度粒度影响。
3. **未建模**：admit 时 lock 住的 prefix 会被驱逐前沿跳过（排位会下移）；mamba 池自己的 LRU 驱逐（H100 上 mamba 使用率不高，没成为瓶颈；H200 上 `max_running` 更大，值得检查）；retraction。
4. **不支持的配置**（`/health` 会报）：非 fcfs、`enable_mixed_chunk`、spec decoding、PD 分离、LoRA。多 DP rank 的代码路径写了但没测过。
5. 预测延迟：单次约 3 ms（P=8），sidecar 不在请求数据路径上。

## 7. 坑

- `--mem-fraction-static 0.85` 在 H100 NVL 上会在 prefill CUDA graph 捕获时 OOM；H200 上没问题。
- H100 机器的 venv 需要把 nvcc/nvvm/crt 钉在 CUDA 13.0（和 torch cu130 的 runtime 一致），否则 FlashInfer JIT 失败；H200 节点用 `module load cuda/13.0`，不受影响。
- sidecar 的轮询在长 prefill step 期间可能 5 秒超时（快照只在两个 scheduler step 之间处理），日志里偶尔的 `poll ... TimeoutError` 无害。
- `loadgen` 的 `max_new_tokens` 等于真实 decode 长度（`ignore_eos`），所以 sidecar 的 `--cap-by-max-new` 在 replay 下等于作弊，默认关闭。
- arrivals 用固定种子，同一个大请求（87k token 的 `r21-t14`）每次 run 都会在 server 1 上造成一次队头阻塞，是检验长尾的好用例。
