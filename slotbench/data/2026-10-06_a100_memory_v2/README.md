# Memory hierarchy alone and beside a co-tenant, A100 SXM4 (vast.ai, 2026-10-06, commit 0a8e3eb) - v2

Pointer chase as in `../2026-10-06_a100_memory_v1` (one warp, 128-byte lines, 17 working sets, `.ca/.cg/.cs`, 3 reps,
adaptive batch so each kernel yields ~256 records of (clock64, %globaltimer)). Co-tenant: `k_stream`, one 256-thread
block per SM reading and writing 512 MB continuously, as back-to-back 50 ms kernels on another stream of the same
process (149 256 blocks over 70 s). Cost $0.11.

## Latency per load, cycles (ns), p50

| working set | .ca alone | .ca with co-tenant | .cg alone | .cg with co-tenant |
|---|---|---|---|---|
| 16 KB – 128 KB (L1) | 39 (28–36 ns) | **942–948 (668 ns)** | 284 | 946 |
| 256 KB – 16 MB (L2 near) | 210–211 (148 ns) | 941–952 (668–676 ns) | 276–284 | 946–948 |
| 24–32 MB (L2 far) | 428 (304 ns) | 952 (675 ns) | 280–313 | 948 |
| 48–128 MB (DRAM) | 571–576 (405 ns) | 953–954 (676 ns) | 565–569 | 949 |

With a bandwidth-bound block on the same SM, a latency-bound load costs ~945 cycles **at every tier**, L1-resident
working sets included (p99 960–983); the SM clock is unchanged (1.41 GHz in both runs). This is not L2 pollution,
which would cap at the DRAM figure: the co-tenant's traffic occupies the SM's load path (L1 miss handling, the
queues toward L2 and DRAM) and every request waits behind it. 4.5× at L2, 24× at L1.

## Can a kernel from another stream start while a long kernel is running? (`coexist_*`)
Yes, immediately: with a 50 ms kernel holding half the SMs (54 blocks) or one block on every SM (108 blocks of
32 threads), a 1-block kernel launched 1 ms later on another stream started **5.6–8.3 µs** after its launch call
(3 reps each). The v1 co-tenant failure was therefore not a placement rule; the likely cause is the synchronous
`cudaMemcpy` issued while the 120 s kernel ran (unverified). The v2 design relaunches the co-tenant every 50 ms and
uploads working sets before the chase kernels are queued.

Next (`memory_coproc`, `memory_mps50` in the campaign): the same co-tenant as a separate process, time-sliced and
under MPS at 50 % of the SMs, to test the prediction that SM partitioning removes the same-SM penalty.
