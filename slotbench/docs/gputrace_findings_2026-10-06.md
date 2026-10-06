# gputrace: what one day of measurements established (2026-10-06)

Four GPU models on seven hosts, 10 campaigns, ≈ $3.6 of rented time. Every host↔GPU number carries a hard bound from the
tick-edge clock sync run before and after it; the bound was feasible on all 153 runs (0.3–0.9 µs).
Data and per-run detail: `data/2026-10-06_{rtx3060,a100,a100_v3,h100,a100x2}_gputrace/`.

## 1. The GPU's hardware timer is per device and drifts (2×A100 host)
Two A100s in one server: `%globaltimer` readings differ by **1.742 s** at the same instant and drift apart at
**0.63 µs/s**. Any tracer that compares timestamps across GPUs needs a per-GPU mapping to one clock, refreshed about
every second to hold ±1 µs. gputrace gets both mappings over PCIe from one host thread with bounds of 0.8 µs each.

## 1b. NCCL across two GPUs on one bounded axis (2×A100, PCIe P2P; `data/2026-10-06_a100x2_nccl`)
With every GPU mapped to the host clock (bounds 0.79 / 0.88 µs) and the collectives released together by one host
store, the two GPUs start an all-reduce within −0.3 µs (p50) of each other and, up to 1 MB, finish within 0.7 µs:
inside the ±1.67 µs skew bound. From 16 MB GPU 1 finishes 14 µs before GPU 0, a real asymmetry outside the bound.
8 B all-reduce: 13.3 µs on the GPU. Enqueueing it for two GPUs costs the host 12–16 µs, which doubled the measured
span (23.6 µs) and created a 3.6 µs start skew when the stamps were not gated: the tool distinguishes the two.
0 causality violations in 2 000 collectives. Caveat: no NVLink between the two rented GPUs.

## 1c. Eight A100s (`data/2026-10-06_a100x8_nccl`)
All eight GPU timers mapped to the host clock with bounds 0.74–0.93 µs. Relative to GPU 0 they read −13.6 to −34.7 s
and drift +4 to +10.4 ppm. Released together, all eight start an all-reduce within 1.4 µs (inside the bound); an 8 B
all-reduce finishes in ring order, ~1 µs per GPU, GPU 7 last at +8.4 µs; from 64 KB the two CPU sockets split the
finish times by up to 4 ms (no P2P in the container: NCCL went through host memory). 8-GPU enqueue: 46–54 µs of host time.
0 causality violations across 67 200 GPU-pair checks.

## 2. A higher-priority stream never preempts running blocks (3 GPUs, 40 reps)
Second kernel's first block starts when the running kernel's wave finishes: wait = block length − offset, exactly,
for 0.2, 0.5, 2 and 10 ms blocks, with and without stream priority, on RTX 3060, A100 (two hosts) and H100
(H100: 0.1029–0.1034 ms over 5 reps for 0.2 ms blocks). Priority only orders blocks that have not started.
Consequence for a slot-deadline workload sharing a GPU through streams: the worst-case wait is the co-tenant's
longest block, whatever the priorities say.

## 3. Cross-process time-slicing preempts mid-block, at a cost (3 GPUs)
Quantum 2.089 ms for the measured process and 2.43 ms for the other (RTX 3060: 2.25 ms), 226–228 switches/s;
every 5 ms block of the other process was suspended mid-execution for 2.43–2.47 ms. Gap − quantum gives the switch
cost: ≈ 170 µs per context switch on A100 and H100, ≈ 90 µs on the RTX 3060. Under MPS the measured thread lost
0.0 % of its time (11 gaps of 0.35–0.44 ms in 7.5 s instead of ≈ 1 700 gaps of 2.4 ms) and the co-tenant's SM set
was exactly its percentage (54 of 108, 56 of 114).

## 3b. Cross-process validation of the clock mappings (A100, H100)
In each time-slicing run, our process and the other process map the same GPU's timer to the host clock
independently (each with its own tick-edge fit). The other process can only finish a kernel while it holds the GPU,
so each of its kernel ends must fall inside one of our not-running intervals. It does, every time: **801 of 801**
(A100) and **800 of 800** (H100), within the summed bounds of 1.05 / 1.14 µs. The trace also shows the turn pattern
the summary statistics hid: our not-running intervals alternate between ≈ 0.79 ms (the other process's 5 ms
kernel finishes about 80 % into its turn and it yields) and ≈ 2.44 ms (a full turn ending in mid-block preemption).

## 3c. Memory latency under a co-tenant (A100, RTX 3060; `data/2026-10-06_*_memory_v*`)
One warp pointer-chasing 4 KB–128 MB with each PTX modifier, cycles and ns from the same record. Alone, A100:
L1 39 cy (≤ 128 KB), L2 211 cy, far L2 half 428 cy (24–32 MB), DRAM 571 cy (405 ns); RTX 3060: 41 / 215 / 515 cy
with knees at 128 KB and 4 MB. **Beside a streaming co-tenant block on the same SM, every load costs ~945 cycles at
every tier, L1-resident sets included** (4.5× at L2, 24× at L1), with the SM clock unchanged. A kernel from another
stream starts within 6–8 µs even while a 50 ms kernel occupies half the SMs or one block per SM (`coexist_*`).
**Sharing modes** (`data/2026-10-06_a100_memory_v3_sharing`): with the co-tenant as a separate process, time-slicing
turns L2 hits into DRAM misses (565–571 cy) with a 14 000-cycle p99 and triples DRAM-tier latency; MPS at 50 % keeps
L1 at 39 cycles and the p99 within 4 % of p50, but L2-resident sets still cost 700 cycles (3.3×) because L2 is shared.

## 4. Launch and completion latencies (p50; p99 in the datasets)
| | RTX 3060 | A100 (host 1 / 2) | H100 |
|---|---|---|---|
| launch call → first instruction | 2.9 µs | 4.2 / 5.9 µs | 4.0 µs |
| graph launch → first instruction | 2.9 µs | 4.1 / 5.0 µs | 3.8 µs |
| after the host slept 50 ms | 26 µs | 11 µs | 40 µs |
| after the host spun 2 ms | 4.5 µs | 5.0 µs | 5.8 µs |
| consecutive kernels in one stream, or in one graph | 1.0 µs | 2.0 µs | 2.0 µs |
| kernel end → mapped flag seen by host | 0.9 µs | 1.0 / 1.1 µs | 0.9 µs |
| kernel end → cudaEventQuery / cudaStreamSynchronize | 1.4 / 1.4 µs | 3.0 / 3.0 µs | 2.5 / 2.5 µs |
| 8 B cudaMemcpyAsync H2D, call → done | 4.6 µs | 7.2 / 8.5 µs | 5.3 µs |
| GPU dependent read of host memory | 0.56 µs | 1.0 µs | 0.98 µs |

The idle penalty is the host CPU's sleep state (spinning removes it); the GPU clock does not drop after 50 ms idle on
any of the three (persistence mode on the datacenter cards). Graphs do not shrink the 2 µs inter-kernel gap.

## 5. Dispatch (H100, 64 ns tick)
A full wave of 912 blocks (8 per SM × 114) is dispatched in 0.19–0.26 µs (3 500–5 700 blocks/µs); 114 blocks in
0.10–0.32 µs. SM order is not sequential. Kernel span = waves × block length within 0.3 %.

## 6. A measurement lesson that is also a scheduler fact
A warp that busy-waits with no stalls monopolises its scheduler's issue slot: at 8 blocks/SM the other warps go up
to a whole 200 µs block without being issued (A100 and H100; 54 µs on the RTX 3060 at 6 blocks/SM). A busy-wait is
therefore the wrong residency probe; gputrace's wait yields with `__nanosleep` (then 0 gaps > 5 µs in 70 000 blocks).

## What is not yet done
Several hosts (PTP on the host side, then the same per-GPU mapping); NCCL over NVLink (the rented pairs were PCIe);
MIG; an application trace (cuPHY slots) on this timeline. Transport note: one raw part of 26 was lost on the 2×A100
host to ssh-proxy lines in the container log; the summary block carried all results.
