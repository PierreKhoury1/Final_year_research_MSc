# gputrace: what one day of measurements established (2026-10-06)

Four GPUs on four hosts, 7 campaigns, ≈ $2.6 of rented time. Every host↔GPU number carries a hard bound from the
tick-edge clock sync run before and after it; the bound was feasible on all 153 runs (0.3–0.9 µs).
Data and per-run detail: `data/2026-10-06_{rtx3060,a100,a100_v3,h100,a100x2}_gputrace/`.

## 1. The GPU's hardware timer is per device and drifts (2×A100 host)
Two A100s in one server: `%globaltimer` readings differ by **1.742 s** at the same instant and drift apart at
**0.63 µs/s**. Any tracer that compares timestamps across GPUs needs a per-GPU mapping to one clock, refreshed about
every second to hold ±1 µs. gputrace gets both mappings over PCIe from one host thread with bounds of 0.8 µs each.

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
Several hosts (PTP on the host side, then the same per-GPU mapping); NCCL collective spans on the 2-GPU host;
MIG; an application trace (cuPHY slots) on this timeline. Transport note: one raw part of 26 was lost on the 2×A100
host to ssh-proxy lines in the container log; the summary block carried all results.
