# gputrace on an A100 SXM4 40 GB (vast.ai, 2026-10-06, commit e679b15)

27 strategy runs plus the time-slice hog; raw block records and host events for every run (the hog's 173 016 block
records are summarised in `timeslice.analysis.json` only). Host: vast.ai verified host (New York), 6 vCPU, driver
580.173, persistence mode on. GPU: A100-SXM4-40GB, 108 SMs, %globaltimer tick 1024 ns. Clock mapping: edge method
feasible in all 28 runs, bound 0.52–0.79 µs. RTX 3060 numbers from `../2026-10-06_rtx3060_gputrace` in brackets.

## Findings (p50 unless stated; host-vs-GPU numbers carry the run's clock bound)

**Launch call → first instruction**: 4.4 µs p50, 5.6 µs p99 [3060: 2.9 / 3.9]; graph launch 3.9 µs; after the host
sleeps 50 ms: 11.3 µs with the launch call itself 9.0 µs [3060: 26 µs]. Sleeping 2 ms vs spinning 2 ms: 5.4 vs
5.0 µs, so this server CPU wakes faster than the desktop one. **Back-to-back kernels in one stream: 3.1 µs between
the end of one and the start of the next** [3060: 1.0 µs]: at the GPU front end, not the host.

**Kernel end → host**: mapped flag 1.6 µs, cudaEventQuery 3.5 µs, cudaStreamSynchronize 3.4 µs [3060: 0.9 / 1.4 / 1.4].

**SM clock**: 1.06–1.41 GHz across runs (the card boosts variably); no drop after 50 ms idle (first sample 1.06 vs
1.09 GHz steady, recovered within 20 µs). Persistence mode keeps the card ready; the idle penalty is host-side.

**Dispatch**: 108–3456 blocks start within 1–2 ticks (≥ 840 blocks/µs lower bound); 8 blocks/SM at 256 threads, 4 at
32 KB shared memory; kernel span = waves × block length within 2 % except for 4-wave kernels (885 vs 807 µs ideal:
warp starvation, below).

**Warp-issue starvation under busy spinning** (`dispatch`, `dispatch_timer1`): at 8 blocks × 8 warps per SM, a warp
that spins on `clock64` with no stalls can go the entire 200 µs block without its next timer read (p50 gap 202 µs);
when each iteration reads `%globaltimer` (a stalling instruction) the p50 gap is 13 µs, p99 120 µs. At ≤ 2 blocks/SM
the largest gap is one tick. A stall-free warp monopolises its scheduler's issue slot (greedy-then-oldest
behaviour); a busy-wait is therefore the wrong probe of residency, and the next version yields with `__nanosleep`.

**Two streams** (A = 864 blocks of 0.2 / 0.5 / 2 / 10 ms, B = 108 blocks launched 100 µs later): B's first block
starts when A's first wave ends: wait = A block length − 100 µs (0.105 / 0.404 / 1.905 / 9.904 ms; max − median
≤ 3 µs), **with or without stream priority**. Same as the RTX 3060: running blocks are never preempted for a
higher-priority stream.

**Two processes** (hog = continuous 5 ms kernels): our quantum 2.089 ms p50 (p90 2.091), their gap 2.430 ms
(p99 2.449), 228 switches/s, our thread not running 37 % of the window; every hog block was suspended mid-block for
2.468 ms. Gap − quantum = 0.34 ms per round trip: about 170 µs per context switch, consistent with the ~300 µs
round-trip cost inferred earlier from the gating runs [3060: ~80–100 µs per switch].

**Copies**: 8 B / 4 KB / 64 KB / 1 MB H2D 7.2 / 7.9 / 12.9 / 85.6 µs, D2H 7.3 / 7.5 / 11.2 / 85.8 µs. GPU dependent read
of host memory 1.01 µs p50, 1.07 µs p99 [3060: 0.56 / 0.59].
