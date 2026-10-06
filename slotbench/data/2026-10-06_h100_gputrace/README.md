# gputrace on an H100 PCIe (vast.ai, 2026-10-06, commit ad668cd)

35 strategy runs; raw block records and host events for every run (hog records summarised in the analysis JSON).
Host: vast.ai verified host (US), 32 vCPU, driver 595.71, CUDA 13.2. GPU: H100 PCIe, 114 SMs, **%globaltimer tick
64 ns** (A100 and RTX 3060: 1024 ns). Clock mapping: edge method feasible in all 35 runs, bound 0.54–0.66 µs.

## Findings (p50 unless stated)

**Dispatch, now resolvable**: 114 blocks (one per SM) start within 0.10–0.32 µs of each other; 228 within 0.13 µs;
912 (8 per SM) within 0.19–0.26 µs: **3 500–5 700 blocks/µs**. SM order is not sequential (e.g. 95, 10, 8, 52, 27 ...).
8 blocks/SM at 256 threads, 6 at 32 KB shared memory. Kernel span = waves × block length within 0.3 %. With the
yielding wait, the largest timer gap inside any of 24 515 blocks was 0.8 µs.

**Busy-wait starvation, reproduced**: stall-free spinning warps at 8 blocks/SM: p50 gap 56–68 µs, max 201 µs;
timer-read spinning: p50 0.8 µs but p99 93 µs. Same mechanism as on the A100.

**Launch call → first instruction**: 4.0 µs p50, 5.0 µs p99; graph 3.8 µs; under MPS 4.1 µs. After the host sleeps
2 ms: 6.6 µs; after 50 ms: **39.6 µs, with the launch call itself 39.5 µs** (this server's CPU sleeps deep); after
spinning 2 ms: 5.8 µs. Bursts of 8 kernels: 2.0 µs between consecutive kernels for stream and graph alike, first
kernel 4.2 / 3.6 µs after the call. The 2 µs inter-kernel gap is therefore real on Hopper too (tick 64 ns), not a
tick artifact of the A100 measurement.

**Kernel end → host**: mapped flag 0.87 µs, cudaEventQuery 2.5 µs, cudaStreamSynchronize 2.5 µs.

**Two streams**: B's first block starts when A's wave ends: wait = A block length − 100 µs = 0.1029–0.1034 /
0.4008–0.4034 / 1.9028–1.9034 / 9.9020–9.9032 ms over 5 reps each, priority or not. Third GPU, same result:
no preemption of running blocks for a higher-priority stream.

**Two processes**: time-slice quantum 2.090 ms (us), gap 2.426 ms (them), 226 switches/s, not running 36 %; the hog's
5 ms blocks suspended 2.436 ms each; ≈ 170 µs per context switch. Under MPS: 11 gaps of 0.41–0.44 ms over 7.5 s,
hog on 56 of 114 SMs at 50 %, its blocks never interrupted.

**SM clock**: 1.75–1.77 GHz, no drop after 50 ms idle. **Copies**: 8 B 5.3 / 6.5 µs, 1 MB 39.8 / 26.8 µs (H2D / D2H;
PCIe 5.0). GPU dependent read of host memory 0.98 µs p50, 1.11 µs p99.
