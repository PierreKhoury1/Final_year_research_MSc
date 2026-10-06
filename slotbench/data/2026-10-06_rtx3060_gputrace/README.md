# gputrace on an RTX 3060 (vast.ai, 2026-10-06, commit 904e535)

26 strategy runs plus the time-slice hog, complete raw data (`*.gpu.bin` block records, `*.host.bin` host events,
`*.json` meta, `*.analysis.json` results). Clock-sync samples (`*.pre/post.{classic,up,down}.bin`) kept for five
representative runs. Host: vast.ai verified host (Romania), 4 vCPU, driver 595.71; GPU RTX 3060, 28 SMs,
%globaltimer tick 1024 ns. Clock mapping: edge method feasible in all 27 runs, bound 0.36–0.74 µs, rate −24 ppm.
Reproduce: `python3 analysis/gputrace.py <prefix>`.

## Findings (p50 unless stated; host-vs-GPU numbers carry the run's clock bound)

**Launch call → first instruction** (`launch*`): 2.9 µs p50, 3.9 µs p99 when the host loop never sleeps.
After the host *sleeps* 100 µs: 6.3 µs; after 2 ms or 50 ms: 26 µs p50, 40 µs p99, and the launch call itself takes
27 µs. After the host *spins* 2 ms (`launch_spin2000`): 4.5 µs. The idle penalty is the sleeping CPU, not the GPU.
Graph launch: 2.9 µs, call 1.6 µs. Eight queued 10 µs kernels: 1.0 µs (one tick) between consecutive kernels.

**Kernel end → host** (`notify`): mapped flag polled by the host 0.87 µs; cudaEventQuery 1.36 µs;
cudaStreamSynchronize 1.40 µs.

**SM clock after idle** (`ramp_idle*`, 3 ms blocks sampled every 20 µs): first sample 1.79 GHz vs 1.84 GHz steady
after 50 ms idle; at 95 % within 20 µs. The GPU barely slows down on this host.

**Dispatch** (`dispatch*`): 28–896 blocks start within one or two 1024 ns ticks (rate not resolvable on this tick);
6 blocks/SM at 256 threads, 8 at 64 threads, 3 at 32 KB shared memory; kernel span = waves × block length within one
tick. **Timer-read starvation**: when ≥ 24–48 warps per SM spin reading `%globaltimer`, a warp can go 53–54 µs
between two reads (p50 at 48 warps/SM); at ≤ 16 warps/SM the largest gap is one tick. Whether this is the timer read
path or warp-issue starvation is tested in the next run (`--timer-every`).

**Two streams** (`concurrency_a{200,500,2000,10000}[_prio]`, A = 224 blocks of the given length, B = 28 blocks
launched 100 µs later): B's first block starts when A's first wave ends: wait = A block length − 100 µs, exactly
(0.102 / 0.401 / 1.902 / 9.902 ms, max − median ≤ 1 µs), **with or without stream priority**. Running blocks are never
preempted by a higher-priority stream on this GPU. Priority does change what happens once B runs: its blocks are no
longer starved by A's second-wave warps (B span 201 µs vs 300 µs).

**Two processes** (`timeslice`, hog = continuous 5 ms kernels): our resident thread stops running for 2.245 ms every
switch (p50; p99 2.269), our quantum 2.08 ms p50, 167 switches/s, not running 26 % of the window. Every one of the
hog's 32 536 five-millisecond blocks was suspended mid-block for 2.275 ms: cross-process time-slicing does preempt
running blocks. Our quantum 2.08 ms vs their gap 2.25 ms: about 80–100 µs per switch is lost to the switch itself.

**Copies** (`copy`): 8 B / 4 KB / 64 KB / 1 MB H2D 4.6 / 5.8 / 10.5 / 89.6 µs, D2H 5.0 / 5.2 / 9.7 / 84.4 µs
(call to completion seen by the host). GPU dependent read of host memory: 0.56 µs p50, 0.59 µs p99 (clock64-timed).
