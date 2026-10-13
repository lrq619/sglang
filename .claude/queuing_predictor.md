# Online queuing-latency predictor: implementation plan

Status: design, not implemented. Written 2026-10-13 for the coding agent that implements it.

## 0. Goal and background

Predict, for a request about to be sent to an SGLang instance, how long it will
wait from send until its first forward step (queueing latency), plus its prefill
time. Consumers: a router that picks the instance minimizing queue + prefill, and
a KV pin/evict policy for agentic sessions.

The method is validated offline; see `work_dirs/queuing_latency.md` and the
reference code in `loadgen/`:

| File | What to reuse |
|---|---|
| `loadgen/qp_sim2.py` | `simulate_wait_v2` (step-level admission simulator), `split_state`, `StepCosts` (step cost model), `fit_step_costs` |
| `loadgen/qp_online.py` | `OnlinePool` (decode-length pool of finished turns), online evaluation protocol |
| `loadgen/qp_predict.py` | `DecodeModel.sample_remaining` (sample D - g given D > g from a sorted pool) |
| `loadgen/qp_psweep.py` | Monte Carlo over P samples vs point estimates |

Method in one paragraph: take a snapshot of the instance (running requests with
generated-so-far g, waiting queue in FCFS order, the step executing now, free
memory). Draw P scenarios; in each, every running and queued request gets a
remaining decode length sampled from the empirical distribution of finished
turns conditioned on D > g. Run the step-level simulator (prefill-first, chunked
prefill, memory + slot admission) once per scenario and return the mean wait
(also p50/p90). Offline result on the knee load regime: 11.6% median relative
error, vs 37.9% for a fluid model and 69.7% for a single simulation with
conditional-mean lengths. P = 1 is already close to converged; use P = 8.

Why SGLang must expose state: exact input and output token counts, cached-prefix
lengths, the step in flight, and free/evictable memory only exist inside the
scheduler. A proxy would have to re-tokenize and guess cache state.

## 1. Architecture

```
            +---------------------- predictor sidecar (new, separate process) ----------------------+
router ---> | POST /predict                                                                          |
            |   1. per candidate instance: GET /scheduler_snapshot   (on demand, or cached <= 50 ms)  |
            |   2. POST /match_prefix {input_ids}  (target request's cached prefix on that instance)  |
            |   3. Monte Carlo: P x simulate_wait  -> mean / p50 / p90 queue + prefill               |
            | background: GET /scheduler_snapshot?finished_since=<seq> every 1 s -> decode-length pool |
            +-----------------------------------------------------------------------------------------+
                       |  HTTP (read-only)
                 SGLang instances (new: /scheduler_snapshot, /match_prefix)
```

The sidecar never sits in the request data path; it only reads SGLang.

## 2. SGLang side

Read these before touching code (repo rules):
- `.claude/skills/large-class-style/SKILL.md` (Scheduler and TokenizerManager are large classes: logic lives in a collaborator, the class only constructs/wires/delegates).
- `.claude/rules/no-dataclasses.md` (new data containers are `msgspec.Struct`), `.claude/rules/no-getattr-defensive.md`, `.claude/rules/comment-style.md`, `.claude/rules/general-code-style.md`.
- `.claude/skills/write-sglang-test/SKILL.md` for tests.

### 2.1 Endpoint `GET /scheduler_snapshot`

Follow the existing `/server_info` -> `get_internal_state` path exactly:

| Layer | Existing precedent | New code |
|---|---|---|
| HTTP | `entrypoints/http_server.py:843` `server_info()` | `@app.get("/scheduler_snapshot")`, query params `finished_since: int = -1`, `max_waiting: int = 1024` |
| io_struct | `managers/io_struct.py:2284` `GetInternalStateReq` / `...Output` (msgspec `BaseReq`) | `GetSchedulerSnapshotReq(finished_since, max_waiting)`, `GetSchedulerSnapshotReqOutput(snapshot: SchedulerSnapshot)` |
| TokenizerManager | `managers/tokenizer_control_mixin.py:144` `_COMMUNICATOR_SPECS` entry, `:1016` `get_internal_state` | spec `("get_scheduler_snapshot", GetSchedulerSnapshotReqOutput, "watching")`; thin method returning one snapshot per DP rank. Size the fan-out like `get_internal_state` (`update_control_communicator_fan_out`, `:226`) |
| Scheduler dispatch | `managers/scheduler.py:1851` `(GetInternalStateReq, self.get_internal_state)` | one dispatcher entry delegating to the collaborator; no logic in `Scheduler` |
| Collaborator | `managers/scheduler_components/load_inquirer.py` `SchedulerLoadInquirer` (lambda getters, built in `Scheduler.init_load_inquirer`, `scheduler.py:2464`) | new `managers/scheduler_components/snapshot_inquirer.py` with `SchedulerSnapshotInquirer`, built by a new `Scheduler.init_snapshot_inquirer()` called from `__init__` after the pool-stats observer (`scheduler.py:715`) |

`load_inquirer.py` is a grandfathered `@dataclass`; the new collaborator follows
`no-dataclasses.md` (frozen `msgspec.Struct` or a plain class with keyword-only
`__init__`). Give it narrow getters, not the Scheduler:
`get_running_batch`, `get_waiting_queue`, `get_chunked_req`, `get_cur_batch`
(`self.cur_batch_for_debug`), `get_last_batch`, the `SchedulerPoolStatsObserver`,
the `NewTokenRatioTracker`, the finished-request ring (2.3), and static config values.

The handler runs from `process_input_requests`, i.e. between scheduler loop
iterations, never during a forward. That is fine; record times so the client can
age the snapshot (`snapshot_monotonic`, `snapshot_unix`).

Cost budget: build + serialize in < 2 ms with 64 running and 1024 waiting
requests; no per-request tensor ops (use `len()` on `prefix_indices`). Cap
`waiting` at `max_waiting` entries from the head of the queue and report the true
length in `num_waiting`.

DP: one snapshot per DP rank, each carrying `dp_rank` (replies are unordered;
see how `LoadSnapshot.dp_rank` is used in `load_inquirer.py:242`). Only the
attn_tp rank 0 of each DP group replies (`ipc_channels.py`), same as `get_internal_state`.

### 2.2 Endpoint `POST /match_prefix`

Body `{"input_ids": [int, ...]}` (or `{"text": "..."}`, tokenized by the TokenizerManager
exactly like a generate request, chat template not applied). Reply per DP rank:
`{"dp_rank": 0, "matched_tokens": 12288, "device_tokens": 12288, "host_tokens": 0}`.

Must be read-only: call the tree cache's match without taking a lock reference
and without touching LRU timestamps or creating nodes. Check whether
`match_prefix_for_req` (`managers/schedule_policy.py:157`) or the unified radix
cache's match path mutates state (access time update, node split on a partial
match); if it does, add a side-effect-free match method to the cache instead.

Phase 2 (optional): accept a list of candidate inputs in one call.

### 2.3 Finished-request ring

The predictor needs the exact decode length of every finished turn. Add a bounded
ring (e.g. last 65536 entries) owned by the collaborator or a tiny helper, appended
where a request finishes in the scheduler result processing (locate the point
where `req.finished()` first becomes true and the request is released; in the
overlap loop this is in `process_batch_result`). Each entry gets a monotonically
increasing `seq`. `/scheduler_snapshot?finished_since=N` returns entries with
`seq > N`, oldest first, plus `finished_seq_head` (largest seq) so the client can
detect a gap if it fell behind the ring.

Record only requests that ran normally to completion (finish reason length or
stop); report aborted ones with `finish_reason="abort"` so the client can drop them.

### 2.4 Snapshot schema

All lengths are tokens. Times: `*_age_s` are seconds before `snapshot_monotonic`
measured with the same clock as the source field (time stats use
`time.perf_counter()`, `batch.launch_ts` uses `time.monotonic()`; convert each
to an age at build time so the client never mixes clocks).

```jsonc
{
  "dp_rank": 0,
  "snapshot_unix": 1791500000.123,
  "snapshot_monotonic": 12345.678,
  "forward_ct": 918273,

  "config": {                       // static; client may cache per instance
    "max_total_num_tokens": 957105,
    "max_running_requests": 52,     // effective value (after the mamba-slot cap)
    "chunked_prefill_size": 32768,  // null if chunked prefill disabled
    "page_size": 1,
    "schedule_policy": "fcfs",
    "enable_mixed_chunk": false,
    "clip_max_new_tokens": 4096     // SGLANG_CLIP_MAX_NEW_TOKENS_ESTIMATION
  },

  "new_token_ratio": {"current": 0.098, "init": 0.7, "min": 0.098, "decay": 0.001003},

  "memory": {                       // from SchedulerPoolStatsObserver.get_pool_stats()
    "kv_available": 1532, "kv_evictable": 210000, "kv_used": 745573,
    "mamba_available": 12, "mamba_evictable": 40, "mamba_used": 211   // null if not hybrid-SSM
  },

  "step": {                         // last launched batch (cur_batch_for_debug)
    "forward_iter": 918273,
    "mode": "extend",               // "extend" | "decode" | "mixed" | "idle"
    "launch_age_s": 0.84,           // now - batch.launch_ts; null if idle
    "result_processed": false,      // overlap loop: whether its result was consumed
    "num_reqs": 3,
    "extend_tokens": [32768, 0, 0], // per request in the batch for extend/mixed
    "prefix_tokens": [12000, 0, 0]  // prefix length before each extend chunk
  },

  "running": [                      // running_batch.reqs (decoding)
    {"rid": "r42-t7", "input_len": 20311, "output_len": 812, "max_new_tokens": 4096,
     "cached_tokens": 19800, "forward_entry_age_s": 31.2}
  ],
  "chunked": {                      // self.chunked_req mid chunked prefill, else null
    "rid": "r50-t2", "input_len": 70498, "prefilled_len": 32768,
    "max_new_tokens": 4096, "prefix_tokens": 0
  },
  "num_waiting": 1,
  "waiting": [                      // waiting_queue in scheduling order, head first
    {"rid": "r51-t0", "input_len": 2339, "max_new_tokens": 4096,
     "prefix_len": 1800,            // matched cached prefix if already computed this round, else null
     "wait_queue_entry_age_s": 4.1, "is_retracted": false}
  ],
  "num_paused": 0, "num_retracted": 0,

  "finished_seq_head": 551230,
  "finished": [                     // only when finished_since >= 0
    {"seq": 551229, "rid": "r40-t9", "input_len": 15100, "output_len": 173,
     "cached_tokens": 14900, "finish_reason": "stop", "finish_age_s": 0.4}
  ]
}
```

Field sources (verify each; line numbers as of commit 8167efc):
- `input_len` = `len(req.origin_input_ids)`; `output_len` = `len(req.output_ids)` (`schedule_batch.py:1058/1068`).
- `max_new_tokens` = `req.sampling_params.max_new_tokens`.
- `cached_tokens` = `req.cached_tokens` (`:1317`); waiting `prefix_len` = `len(req.prefix_indices)` when the policy has matched it this round (`:1183`), else null.
- `prefilled_len` for the chunked request: from `req.extend_range` / `kv_committed_len` (`ReqKvInfo`, `:928`); pick the one that equals tokens already written to KV and document it.
- Ages: `req.time_stats.wait_queue_entry_time`, `forward_entry_time` (`observability/req_time_stats.py:615`, perf_counter).
- Step: `batch.forward_mode`, `batch.launch_ts`, `batch.extend_lens` (`schedule_batch.py:2400`, `:2532`).
- Memory: `SchedulerPoolStatsObserver.get_pool_stats()` (`scheduler_components/pool_stats_observer.py:198`).
- Ratio: `NewTokenRatioTracker` (`scheduler_components/new_token_ratio_tracker.py`).

Payload types: `msgspec.Struct` classes (`SchedulerSnapshot`, `RunningReqSnapshot`,
`WaitingReqSnapshot`, `StepSnapshot`, `FinishedReqRecord`) in a new
`managers/scheduler_snapshot.py`; they must be msgpack-native for the IPC hop.

### 2.5 Tests (SGLang side)

- Unit test the collaborator with fake getters: field mapping, `max_waiting` cap, ring cursor semantics (gap detection, `finished_since=-1` returns none).
- One server test (small model): send a burst, poll `/scheduler_snapshot`, check `len(running) + num_waiting` against `/metrics` gauges within one scheduler step, and that every finished rid appears once in the ring with `output_len == completion_tokens` of its response.
- `/match_prefix`: after a request completes, matching its own input returns > 0 and repeating the call does not change `/metrics` cache-hit counters or eviction order.

## 3. Predictor sidecar

New package `queue_predictor/` at the repo root (next to `loadgen/`), standalone
Python (aiohttp + numpy), no SGLang import.

### 3.1 Modules

| Module | Responsibility | Port from |
|---|---|---|
| `client.py` | Async SGLang client: snapshot (with a per-instance cache, TTL 50 ms), `match_prefix`, `/tokenize`, finished-ring cursor | new |
| `pool.py` | Decode-length pool per model: sorted lengths of finished turns, sliding cap (last 20000), cold-start prior loaded from a JSON file until `min_pool` (500) entries | `OnlinePool` |
| `costs.py` | `StepCosts` load/save JSON; decode step = `d0 + d1*batch + d2*ctx_Mtok`, prefill step = `p0 + sum(p1*E + p2*E*(P+E/2)/1e6)` | `qp_sim2.StepCosts` |
| `simulator.py` | Step-level simulator (2.4 schema in, wait seconds out) | `simulate_wait_v2`, `split_state` |
| `predictor.py` | Monte Carlo: P scenarios, mean/p50/p90; prefill time of the target | `predict_mc_online` |
| `server.py` | HTTP API (3.3), background ring poller | new |
| `profile_costs.py` | Fit `StepCosts` from a loadgen run directory | `fit_step_costs` |

### 3.2 Simulator changes vs the offline version

1. Initial state from the snapshot instead of reconstruction:
   - running: `ctx = input_len + output_len`, `g = output_len`.
   - chunked: in-flight prefill with `extend_left = input_len - prefilled_len`.
   - waiting: FCFS order as given; `extend = input_len - (prefix_len or 0)`.
   - busy time of the step in flight: predicted cost of `step` (prefill cost from `extend_tokens`/`prefix_tokens`, or decode cost from batch size and context) minus `launch_age_s`, floored at 0. This replaces the offline interpolation, which caused the large error on 1-10 s waits.
2. Admission must use the real `max_new_tokens` for reservations, exactly like `PrefillAdder`: reserve `min(max_new_tokens - output_len, clip) * new_token_ratio.current` per running request and `ctx + min(max_new_tokens, clip)` for the candidate. Sampled decode lengths only decide when requests finish. (Offline, `max_new_tokens` equalled the true length, which hid this.)
3. Free memory starts from `memory.kv_available + memory.kv_evictable` minus reservations; a finishing request returns its context (it becomes evictable cache). Slot limit: `config.max_running_requests`.
4. Age the snapshot: shift all ages by the time between `snapshot_monotonic` and prediction time.

### 3.3 Predictor API

`POST /predict`
```jsonc
// request
{"instances": ["http://10.0.0.1:30000", "http://10.0.0.2:30000"],
 "input_ids": [...],            // or "text": "...", tokenized via the first instance's /tokenize
 "max_new_tokens": 4096,
 "samples": 8}                  // P, default 8
// response
{"predictions": [
  {"instance": "http://10.0.0.1:30000", "dp_rank": 0,
   "queue_mean_s": 3.2, "queue_p50_s": 2.9, "queue_p90_s": 6.1,
   "prefix_tokens": 12288, "prefill_s": 0.41,
   "total_mean_s": 3.61,        // queue_mean_s + prefill_s, what the router minimizes
   "snapshot_age_s": 0.02, "pool_size": 18342}
 ]}
```
`GET /health` returns pool sizes and last snapshot time per instance.
`GET /pool` returns pool quantiles (debugging and drift checks).

Latency budget: < 20 ms per instance at P = 8 with 50 running and 100 waiting
(the offline Python version runs ~2-5 ms per simulation; vectorize the decode
segments with numpy if needed).

### 3.4 Cold start and profiling

- `profile_costs.py <loadgen run dir>` fits `StepCosts` and writes `costs/<model>_<gpu>.json`. The current fit for Qwen3.8-27B on 2xH200 (ratio 1.3) can be regenerated from `logs/sweep_20261009_*_r*` with `fit_step_costs`.
- Prior decode-length distribution: a JSON list of decode lengths (e.g. from `data/hermes_clean.jsonl`), used until the live pool has `min_pool` entries.

## 4. Validation

1. Replay `loadgen` (hermes, `data/arrivals_r4.5_d180.jsonl`, 3 h) with the sidecar running; have loadgen call `/predict` right before each send and log the prediction next to the actual wait (`t_admit - t_send`) in `loadgen.csv`.
2. Acceptance on queued requests (wait > 1 s) in the knee regime: median relative error <= 15% and Spearman >= 0.95 (offline reference: 11.6%, 0.977).
3. Overhead: polling `/scheduler_snapshot` at 10 Hz per instance plus one call per request must not lower throughput by more than 1% vs a run without the sidecar (compare `analyze_jct.py` output and turns completed).
4. Compare against the offline numbers in `work_dirs/queuing_latency.md` and record results in `work_dirs/`.

## 5. Out of scope / caveats

- Not supported in v1: speculative decoding, priority scheduling, non-fcfs `schedule_policy`, `enable_mixed_chunk=True`, PD disaggregation, LoRA. The predictor should refuse (`/health` reports unsupported config) rather than return wrong numbers.
- Overlap scheduler: the snapshot is taken between iterations while the GPU may still run the launched batch; `step.result_processed=false` marks that case. The busy-time formula in 3.2 assumes the launched batch is the one occupying the GPU.
- Retraction: `new_token_ratio.current` jumps after a retract; the snapshot carries the current value, the simulator keeps it constant over the horizon.
- Hybrid models: the mamba slot cap is folded into `max_running_requests`; mamba free slots are reported for diagnostics only.
- Clock alignment across machines is not needed: every time in the snapshot is an age.

## 6. Work items (suggested order)

1. SGLang: snapshot payload structs + collaborator + dispatcher entry + `/scheduler_snapshot` (without ring and match_prefix). Unit tests.
2. SGLang: finished-request ring + `finished_since`. Tests.
3. SGLang: side-effect-free `/match_prefix`. Tests.
4. Sidecar: port simulator/pool/costs from `loadgen/`, unit tests reproducing `loadgen/qp_sim2.py` outputs on a recorded snapshot.
5. Sidecar: HTTP API, background poller, cold start.
6. loadgen hook + validation run (section 4).
