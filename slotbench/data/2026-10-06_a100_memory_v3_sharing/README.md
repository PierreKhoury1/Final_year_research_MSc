# Memory latency while sharing an A100: time-sliced vs MPS (vast.ai, 2026-10-06, commit 1a27625)

Same pointer chase as `../2026-10-06_a100_memory_v2` (one warp, `.ca/.cg/.cs`, 4 KB – 128 MB, 3 reps). Co-tenant:
a **separate process** (`gputrace --role hog --hog-kind stream`) streaming 512 MB read+write on every SM in 50 ms
kernels, (a) with no MPS, so the two processes time-slice, (b) under MPS with the hog limited to 50 % of the SMs.
`timeslice_mps50` repeats the time-slice probe under MPS (11 gaps of 0.4 ms in 7.5 s, hog on 54/108 SMs). Cost $0.08.

## Cycles per `.ca` load, p50 (p99), A100

| tier | alone (v2) | same-SM streaming blocks (v2) | separate process, time-sliced | separate process, MPS 50 % |
|---|---|---|---|---|
| L1 (≤ 128 KB) | 39 | 945 | 39 (p99 39–565) | **39** (39) |
| L2 (256 KB – 16 MB) | 211 | 945 | 565–571 (p99 3 956–14 104) | 697–700 (713–728) |
| L2 far half (24–32 MB) | 428 | 952 | 570 (2 264–2 827) | 701–703 (711–712) |
| DRAM (48–128 MB) | 571–576 | 953 | 1 418–1 697 (1 422–1 703) | 707–720 (721–725) |

- **MPS partitioning removes the same-SM penalty completely**: L1-resident loads return to 39 cycles (945 with a
  co-tenant block on the same SM). It also removes the time-slice tail (p99 within 4 % of p50).
- **MPS does not protect L2**: the co-tenant's streaming shares the 40 MB L2, so working sets that fit L2 alone cost
  697–700 cycles, 3.3× the idle 211 and above the idle DRAM latency; DRAM-tier loads cost 25 % more from bandwidth
  contention.
- **Time-slicing is worst at the tail**: L2-tier loads cost DRAM latency (the other process evicts L2 during its
  2 ms turns) and the p99 is 14 000 cycles (a switch landing inside a 256-load batch); DRAM-tier loads cost 2.5–3×
  (1 418–1 697), consistent with cold TLB and L2 after each switch (not separated here).

For a memory-bound kernel sharing a GPU with a streaming tenant: time-slicing costs it its L2 and a 10 µs tail;
MPS keeps its L1 and its latency distribution but still costs it its L2. The scheduling-level result of the gating
study (MPS meets the 5G deadline, time-slicing does not) has its memory-level explanation here.
