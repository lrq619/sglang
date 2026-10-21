## Paper Summary

This paper tries to correct a false assumption in the author’s prior (unpublished) work about autoscaling in agentic serving. Through hardware measurements on vLLM and a calibrated trace-driven simulation (Azure traces with synthetically constructed multi-turn agent programs), the author claims the following two insights:

- Most KV cache loss is not induced by scale-in, but by eviction under neighbor allocation pressure.
- Parked prefix state is valuable but invisible to native autoscaling signals.

The author then claims that the first-order lever is not autoscaling but admission control. To quantify the cost and benefit of protecting parked prefixes, the author evaluates a minimal fixed-cap admission policy.

## Strengths

- Identifying the root cause of KV cache loss in elastic agentic serving is an important and timely problem.
- The author releases scripts, raw measurements, and the simulator.

# Weaknesses

- Poor writing. The narrative is organized around correcting the author’s own unpublished work rather than around the problem, and the abstract and introduction emphasize autoscaling signals while the main conclusion is about admission control.
- The major results (e.g., the 96.53%/3.47% attribution) come from simulation over synthetically constructed agent programs, not from real agentic workloads or real cluster runs.
- The insights are limited to a specific setting: a single GPU with small KV capacity (~75k tokens), and an engine whose cache semantics are LRU-based with no pinning (native vLLM).

Questions & Comments for the Author

- Please do not frame the paper around correcting assumptions in your unpublished prior work, which readers cannot access. ParkAware can simply be presented as a parked-count baseline. Similarly, Figure 2(a) shows that the three native signals drop to zero during tool calls, but this is only loosely connected to the main takeaway: that scale-in avoidance is not the first-order lever and admission control is. Please clarify how this observation supports the main conclusion, or restructure the paper so the two parts form a coherent argument.
- The main insight, that KV cache loss comes mainly from neighbor allocation pressure rather than scale-in, is derived from a simulated workload with low KV capacity. This setting inherently favors pressure eviction. Moreover, the result depends on vLLM’s no-pinning LRU semantics; with TTL- or pin-based retention (e.g., Continuum), the attribution may change significantly. Please provide real experimental results under more setups (larger KV capacity, different models/GPUs, real agent traces, and different cache retention policies).
- Admission control is proposed as the solution, yet only a single fixed-cap admission policy is evaluated. Please compare against existing admission control and cache-retention approaches (e.g., CONCUR, PEEK, Continuum) to show how the conclusions hold across policies.