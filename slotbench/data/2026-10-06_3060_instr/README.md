# The instruction table, RTX 3060: alone, beside a same-SM co-tenant, under MPS (2026-10-06, commit b97ce4c)

Produced by the command-line tool on a rented RTX 3060 (driver 580.173, 28 SMs, 1024 ns timer tick):
`gputrace characterize --profile instr` → `report.md`, `summary.json`, `timeline.html`, per-run analysis and
raw records; `cloud/onstart_cli.sh` shipped the output directory. Cost $0.02.

Method (`gputrace/instr.cuh`, strategy `instr`): a chain of N dependent instructions of one kind between two
`%clock64` reads, N = 1, 2, 4, 8, 16, 32 (and 128 for the non-load kinds), 256 samples each; the chain is tied
to the opening read by a volatile seed load and to the closing read by a volatile store of its result. A
least-squares fit of the per-N medians, cycles = a + b·N, gives the latency b and the bracket overhead a
(`latency_cycles_p10` is the same fit through the 10th percentiles). `sass_check.json`: every one of the 91
bracket kernels in the compiled binary holds exactly N target opcodes between its two clock reads. Loads chase a
random permutation of 128-byte lines in a working set per tier (16 KB L1, 1 MB L2, 24 MB, 128 MB DRAM).
Conditions: alone; one streaming block on every SM in the same process, kernels of 50 ms queued four deep so
there is no gap (`instr_cotenant`; the analysis keeps only the samples a co-tenant block covered on the probe's
SM, 35 888 of 40 704); a streaming process under MPS with 50 % of the SMs (`instr_mps50`, all samples: the
other process holds other SMs, what is shared is the L2 and the fabric).

## Latency per instruction, cycles (slope b)

| instruction | alone | same-SM co-tenant | MPS 50 % |
|---|---|---|---|
| LDG.ca, 16 KB (L1) | 38.9 | 1 306 | 38.8 |
| LDG.ca, 1 MB (L2) | 242 (p10 213) | 1 286 | 723 |
| LDG.ca, 24 MB / 128 MB (DRAM; 3 MB L2) | 410 / 443 | 1 281 / 1 285 | 577 / 725 |
| LDG.cg (L2 only), 16 KB / 1 MB / 128 MB | 214 / 213 / 440 | 1 304 / 1 278 / – | 725 / 724 / 740 |
| LDG.cs, 16 KB / 1 MB / 128 MB | 38.9 / 226 / 522 | 1 310 / 1 285 / 1 280 | 38.8 / 734 / 728 |
| LDG.nc, 16 KB / 1 MB / 128 MB | 38.9 / 214 / 439 | 1 302 / 1 268 / 1 287 | 38.9 / 520 / 728 |
| LDS (shared, dependent) | 23.0 | 23.4 | 23.0 |
| FADD / FFMA / IMAD (dependent) | 4.0 / 4.0 / 4.0 | 4.0 / 4.0 / 4.0 | 4.0 / 4.0 / 4.0 |
| SHFL.IDX (dependent, 32 lanes) | 25.9 | 26.2 | 25.9 |
| ATOM.ADD with return, global | 242 | 321 | 271 |
| RED (no return), per op + fence | 8.1 + 499 | 8.3 + 535 | 8.1 + 500 |
| STG per op + fence | 7.7 + 499 | 7.9 + 536 | 7.8 + 498 |
| `__syncthreads`, 256 threads | 37.0 | 37.0 | 37.0 |

- The same-SM streaming co-tenant (one block of 256 threads doing float4 loads/stores) makes every global load
  cost ~1 290 cycles whatever its tier, L1 hits included: the load/store unit and the L1 are the shared resource,
  not the DRAM. Everything that does not go through the LSU (ALU chains, shuffles, shared memory, the barrier) is
  unchanged to the cycle; the fence costs 7 % more, the global atomic 33 % more.
- Under MPS the other process has its own SMs: L1 hits and the ALU are exactly the alone values; L2-tier loads
  cost 720 cycles (3.4× alone) because the 3 MB L2 is shared; the atomic 12 % more.
- SHFL: the first builds measured 0 cycles, because ptxas proved a shuffle of a warp-uniform value to be the
  identity and dropped the chain on the converged path (a CALL fallback remained for the divergent one). The
  chain now starts from a lane-dependent value and each link reads the next lane; the SASS holds SHFL.IDX × N.
- The DRAM row (443 cy at 128 MB; p10 216) shows the 3060's L2 (3 MB) does not hold the 24 MB set either.
- `timeline.html`: the three runs' kernels on the host axis, bound 0.50–0.65 µs.
