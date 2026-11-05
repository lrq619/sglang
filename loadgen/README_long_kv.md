# Long-KV decode interference micro-benchmark

Question: with the number of running requests held fixed, does the context length
of co-running requests slow down the decode of short requests on the same instance,
and by how much?

- **A**: 8 short requests (512-token context, 128 output tokens) run alone.
- **C(L)**: the same 8 short requests join after 8 stressors of context length L
  have finished prefill and are decoding.

Short prompts are prefix-cached before every trial (identical input, 1 output
token), so their prefill is mostly skipped and their stream time is dominated by decode.

## Server configuration

One H200 running Qwen3.8-27B with the `loadgen/launch_servers.sh` defaults:
TP1, fp8 KV, flashinfer attention, CUDA graphs on, radix cache on,
`--mamba-full-memory-ratio 1.3`, `--mem-fraction-static 0.85`. This gives:

| | |
|---|---|
| KV pool (full-attention layers) | 957,109 tokens |
| mamba state pool | 263 slots, 5 per request, so `max_running_requests` = 52 |
| context length | 262,144 |

## Sweep range

The scheduler admits a request only if its input plus `min(max_new_tokens, 4096)`
fits, so a trial stays free of eviction and retraction only while

    8 * (L + stress_max_new) + 8 * (512 + 128) <= 957,109

With `stress_max_new = 2048` the largest admissible L is ~115K. The sweep stops at
96K (98,304 tokens), which reserves at worst about 84% of the pool:

    L = 128, 512, 2K, 8K, 16K, 32K, 64K, 96K

The short requests stop after 128 tokens, so the stressors never use more
than a small part of their 2048-token budget; they are aborted once the shorts finish.

## Validity checks (per trial)

- every short returned exactly `--short-out` tokens;
- no stressor finished or failed before the last short finished;
- `sglang:num_retracted_requests_total` and `sglang:evicted_tokens_total` did not
  change between the moment the shorts were submitted and the moment they finished.

A trial that fails any check has `valid=0` and an `invalid_reason` in `trials.csv`,
and is excluded from `summary.csv` and from the plots.

## Running

On the login node, `long_kv.sh` is a PBS script: it starts one server, runs the
benchmark, plots, and shuts the server down. Each run writes one flat directory
`~/scratch/logs/<RUN_NAME>_<timestamp>/`.

```bash
# smoke run: A plus three stressor lengths, one repetition
qsub -V -v RUN_NAME=longkv_smoke,BENCH_ARGS="--reps 1 --stress-lens 128 8192 98304 --no-shuffle" \
    -l walltime=01:00:00 loadgen/long_kv.sh

# full sweep: A plus 8 lengths, 5 repetitions, order shuffled within each repetition (seed 0)
qsub -V -v RUN_NAME=longkv loadgen/long_kv.sh

# against an already running server
.venv/bin/python loadgen/bench_long_kv.py --url http://127.0.0.1:30000 --outdir logs/longkv_manual
~/scratch/mplenv/bin/python loadgen/plot_long_kv.py logs/longkv_manual
```

## Outputs

| file | content |
|---|---|
| `events.jsonl` | one record per request (stressors and shorts): submit time and every `(t, completion_tokens)` stream event, relative to trial start |
| `requests.csv` | one row per short request: TTFT, TPOT, E2E, decode duration, prompt, cached and output tokens |
| `trials.csv` | one row per trial: means over the shorts, stressor tokens at short start and end, retraction and eviction deltas, validity |
| `summary.csv` | per point: mean and std across trials, slowdowns vs A and vs C(L_min) |
| `tpot_vs_L.png`, `e2e_vs_L.png` | short-request mean TPOT and E2E vs L, with A as a dashed reference line |
| `run_info.txt`, `servers/`, `config.json` | host, GPU, git commit, server log, benchmark arguments |

TPOT is `(t_last - t_first) / (n_out - 1)`, computed from stream event times and
`completion_tokens`. It counts tokens, not stream events.

## Results

Run `~/scratch/logs/longkv_20261105_121019` (H200, Qwen3.8-27B, launch_servers.sh
defaults, 5 repetitions, all 45 trials valid, 0 retractions and 0 evicted tokens).
Each value is the mean over trials, +- the std across trials, of the per-trial mean over the 8 shorts.

| point | TPOT (ms) | vs A | vs C(128) | E2E (ms) | vs A | vs C(128) |
|---|---|---|---|---|---|---|
| A | 15.96 +- 0.04 | 1.000 | - | 2073 +- 6 | 1.000 | - |
| C(128) | 16.61 +- 0.01 | 1.041 | 1.000 | 2171 +- 2 | 1.047 | 1.000 |
| C(512) | 16.70 +- 0.02 | 1.047 | 1.006 | 2182 +- 3 | 1.052 | 1.005 |
| C(2K) | 16.82 +- 0.00 | 1.054 | 1.013 | 2197 +- 0 | 1.060 | 1.012 |
| C(8K) | 17.47 +- 0.03 | 1.095 | 1.052 | 2281 +- 4 | 1.100 | 1.051 |
| C(16K) | 18.27 +- 0.06 | 1.145 | 1.100 | 2393 +- 26 | 1.154 | 1.102 |
| C(32K) | 19.96 +- 0.02 | 1.251 | 1.202 | 2601 +- 2 | 1.255 | 1.198 |
| C(64K) | 23.41 +- 0.01 | 1.467 | 1.409 | 3044 +- 4 | 1.468 | 1.402 |
| C(96K) | 26.82 +- 0.03 | 1.681 | 1.615 | 3482 +- 4 | 1.680 | 1.604 |

- Adding 8 nearly empty stressors (C(128)) costs +4% TPOT compared with running
  the shorts alone (batch size 8 -> 16).
- Beyond that, short-request TPOT grows linearly with the total KV in the batch, by
  about 13 us per 1K stressor KV tokens. At L = 96K the batch holds about 790K
  KV tokens and short TPOT is 1.6x that of C(128).
- E2E follows TPOT because the shorts are prefix-cached: TTFT stays at 46-81 ms
  (it rises with L by about 15 ms) against a decode time of about 2 s.

Comparing with A gives the total co-location overhead; only the comparison with
C(L_min) isolates the effect of stressor KV length.

## Mix sweep: number of stressors at fixed concurrency

```bash
qsub -V -v RUN_NAME=longkv_mix,BENCH_ARGS="--stress-lens 98304 --stress-counts 0 1 2 3 4 5 6 7 8 --concurrency 16" \
    -l walltime=02:00:00 loadgen/long_kv.sh
```

Run `~/scratch/logs/longkv_mix_20261105_132805`: 16 - k shorts and k stressors of 96K,
5 repetitions, all 45 trials valid, 0 retractions and 0 evicted tokens.
Plots: `tpot_vs_mix.png`, `e2e_vs_mix.png`.

| short:stressor | TPOT (ms) | vs 16:0 | E2E (ms) | vs 16:0 |
|---|---|---|---|---|
| 16:0 | 16.77 +- 0.07 | 1.000 | 2225 +- 30 | 1.000 |
| 15:1 | 18.13 +- 0.09 | 1.081 | 2380 +- 25 | 1.069 |
| 14:2 | 19.41 +- 0.02 | 1.158 | 2554 +- 26 | 1.148 |
| 13:3 | 20.83 +- 0.03 | 1.242 | 2719 +- 16 | 1.222 |
| 12:4 | 22.12 +- 0.01 | 1.319 | 2878 +- 2 | 1.294 |
| 11:5 | 23.11 +- 0.02 | 1.378 | 3008 +- 4 | 1.352 |
| 10:6 | 24.56 +- 0.03 | 1.465 | 3192 +- 4 | 1.435 |
| 9:7 | 25.62 +- 0.02 | 1.528 | 3331 +- 2 | 1.497 |
| 8:8 | 26.83 +- 0.05 | 1.600 | 3485 +- 6 | 1.566 |

- Short-request TPOT grows linearly with the number of 96K stressors, by about
  1.26 ms (+7.5%) per stressor.
- This matches the L sweep: one 96K stressor adds about 98K KV tokens, and at
  about 13 us per 1K KV tokens that predicts +1.28 ms. The 8:8 point, 26.83 ms,
  reproduces C(96K) from the L sweep, 26.82 ms.
