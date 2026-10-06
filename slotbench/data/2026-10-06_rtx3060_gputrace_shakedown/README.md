# gputrace shakedown on an RTX 3060 (vast.ai, 2026-10-06, commit 5c016cc)

First full campaign; raw records were lost to the log transport (fixed afterwards), analysis JSONs kept.
Host: vast.ai verified host, Romania, driver 595.71, 4 vCPU. GPU: RTX 3060, 28 SMs, %globaltimer tick 1024 ns.
Clock mapping: edge method feasible in all 15 runs, bound 311–370 ns, rate −24 ppm.

| Angle | Result |
|---|---|
| launch call → first instruction | 4.1 µs p50, 7.1 µs p99 (±0.36 µs); after 100 µs host sleep 11.6 µs; after 2 ms / 50 ms sleep 26.4 µs, with the launch call itself 28 µs |
| kernel end → host | mapped flag 0.5 µs, cudaEventQuery 1.0 µs, cudaStreamSynchronize 1.1 µs p50 |
| inter-kernel gap, 8 queued 10 µs kernels | 2.0 µs p50 (two ticks) |
| dispatch | 28–224 blocks start within one 1024 ns tick (rate unresolvable at this tick); 6 blocks/SM at 256 threads, 8 at 64 threads; kernel span = waves × block length exactly |
| timer-read gaps inside a block | 1 tick at 1–2 blocks/SM; up to 54 µs at 6 blocks/SM (48 spinning warps): warp issue starvation, not preemption |
| second stream, first kernel has 2 ms blocks | waits 1.50 ms for its first SM, with or without stream priority: no preemption of running blocks |
| time-slicing, two processes | quanta 2.09 ms (us) / 2.14 ms (hog), 198 switches/s, not running 42 % of the time; hog's 2 ms blocks never interrupted (fit a quantum) |
| memcpy 8 B / 64 KB / 1 MB | H2D 4.0 / 14.7 / 166 µs, D2H 4.4 / 14.0 / 163 µs end to end |
| GPU dependent read of host memory | 0.49 µs p50, 0.61 µs p99 |
| SM clock | 1.93 GHz throughout (the sampler kept the GPU awake, so no idle ramp was observed; `ramp` strategy added) |

Known defects of this run: graph launches matched off by one (negative latency; fixed), 48 KB shared-memory
variant failed to launch (fixed: 32 KB + launch error checks), "preempted" counters were really warp-starvation
counters (renamed).
