# Kernel timeline, A100 SXM4 (2026-10-06, commit 64b1a31, corrected build)

`gputrace characterize --profile ktrace` on a rented A100-SXM4-40GB (108 SMs, 1024 ns timer tick): `ktrace`
(B = 108 and 432 blocks of 256 threads, 256-FFMA compute phase, 5 launches each), `ktrace_long` (B = 432, 4096
FFMA, 3 launches), `ktrace_idle` (B = 108 after 50 ms of host idle, 5 launches). Every warp stamps
(%globaltimer, clock64, %globaltimer) at eleven checkpoints (`gputrace/ktrace.cuh`); `ktrace_phases.json` is the
SASS between consecutive checkpoints (sm_80 build, `tools/sass_phases.py`). Analysis: `analysis/ktrace.py`;
notebook: `notebooks/kernel_timeline.ipynb`. Cost $0.25 (slow log transfer on this host).

Per warp, cycles (median over all warps and launches; each phase subtracts the 14–16 cy entry checkpoint, ~3 cy
more than the in-phase cost): two dependent `.nc` loads with their S2R/address arithmetic and consuming shared
store 479 (1 block per SM) / 742 (4 per SM); first barrier wait for the last warp 39 (p99 212) / 138 (p99 2 564),
last arriver excluded; release after the last arrival 33 / 82; 256 dependent FFMA in a 16×-unrolled loop
1 238 / 1 846 (~200 cy of loop control inside); store + `__threadfence` 644 / 873; warp 0's atomic with return
298 / 434 (p90 432 / 1 094), the other warps branch around it in ~33 cy. Launch call to first warp 6.1–6.6 µs
on warm launches (36 µs on the run's first); block starts spread 0.5 µs (1 per SM) / 6.1 µs (4 per SM: 3
resident, the 4th starts when one finishes); flag write to the host seeing it 0.4–1.4 µs (the write's GPU window
and the run's ±0.76 µs host bound); last exit to `cudaStreamSynchronize` returning 4.0–4.4 µs. Block bound
(largest half-width over a block's stamps, median over blocks): 53 ns (B = 108) / 38 ns (B = 432), worst block
254 / 285 ns; common-rate interval per launch 1.329–1.337 GHz (B = 108, median), rate inconsistency 0 ns (B =
108), ≤ 14 ns (B = 432), 22–36 ns (`ktrace_long`, 59 µs). Every launch: 0 warps released before the last
arrival, 0 ticket-order violations (plain and round-trip); the ticket check's floor is 250–285 ns. The earlier
README's "timer guard up to 82 ns" was an artefact of a ±5 % rate clip in the fit and is retracted.
