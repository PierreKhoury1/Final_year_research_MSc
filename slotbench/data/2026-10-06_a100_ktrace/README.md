# Kernel timeline, A100 SXM4 (2026-10-06, commit 64b1a31, corrected build)

`gputrace characterize --profile ktrace` on a rented A100-SXM4-40GB (108 SMs, 1024 ns timer tick): `ktrace`
(B = 108 and 432 blocks of 256 threads, 256-FFMA compute phase, 5 launches each), `ktrace_long` (B = 432, 4096
FFMA, 3 launches), `ktrace_idle` (B = 108 after 50 ms of host idle, 5 launches). Every warp stamps
(%globaltimer, clock64, %globaltimer) at eleven checkpoints (`gputrace/ktrace.cuh`); `ktrace_phases.json` is the
SASS between consecutive checkpoints (sm_80 build, `tools/sass_phases.py`). Analysis: `analysis/ktrace.py`;
notebook: `notebooks/kernel_timeline.ipynb`. Cost $0.25 (slow log transfer on this host).

Per warp, cycles (median over all warps and launches): checkpoint 14; two dependent coalesced loads 479 (1 block
per SM) / 742 (4 per SM); first barrier wait for the last warp 21 (p99 174) / 55 (p99 1 972), release after the
last arrival 34 / 60; 256 dependent FFMA 1 238 / 1 846; store + `__threadfence` 644 / 873; atomic ticket 33.
Launch call to first warp 5.9 µs; block starts spread 0.4 µs (1 per SM) / 6.2 µs (4 per SM: 3 resident, the 4th
starts when one finishes); flag write to the host seeing it 1.3 µs; last exit to `cudaStreamSynchronize`
returning 4.3 µs. Block lines: half-width 198 ns median, 560 ns max; timer guard up to 82 ns per launch; every
launch: 0 warps released before the last arrival, 0 ticket-order violations.
