# The instruction table, A100 SXM4: per-instruction brackets alone, beside a co-tenant, under MPS (2026-10-06, commit 3d84898)

Method (`gputrace/instr.cuh`, strategy `instr`): a chain of N dependent instructions of one kind between two
`%clock64` reads, N = 1, 2, 4, 8, 16, 32, 256 samples each; the chain is tied to the opening read by a volatile seed
load and to the closing read by a volatile store of its result, so `ptxas` cannot move it. A least-squares fit of
the per-N medians, cycles = a + b·N, gives the latency b and the bracket overhead a. `tools/sass_check.py` lists,
for every bracket kernel in the compiled binary (`gputrace.sass.gz`), the opcodes between the two clock reads and
checks that exactly N target opcodes are there (`sass_check.json`). Loads chase a random permutation of 128-byte
lines in a working set chosen for each tier (16 KB L1, 1 MB L2, 24 MB far L2, 128 MB DRAM).
Conditions: alone; a streaming kernel on every SM in the same process (`instr_cotenant`); a streaming process under
MPS with 50 % of the SMs (`instr_mps50`). Cost $0.06.

## Latency per instruction, cycles (slope b); the N=1 bracket in brackets

| instruction | alone | same-SM co-tenant | MPS 50 % |
|---|---|---|---|
| LDG.ca, 16 KB (L1) | 38.8 (408) | 1 000 | **34.3** |
| LDG.ca, 1 MB (L2) | 232.7 | 951 | 713 |
| LDG.ca, 24 MB (far L2) | 248.8 | 952 | 698 |
| LDG.ca, 128 MB (DRAM) | 402.2 = 272 ns | 953 | 722 |
| LDG.cg (L2 only), 16 KB / 1 MB / 128 MB | 285 / 283 / 497 | 945–980 | 279 / 705 / 714 |
| LDG.cs, 16 KB / 1 MB / 128 MB | 38.8 / 211 / 485 | 950–1 028 | 36.7 / 700 / 718 |
| LDS (shared, dependent) | 23.0 (384) | 17 ± noise | 23.0 |
| FADD / FFMA / IMAD (dependent) | 3.9 / 3.9 / 4.0 | not resolvable (see below) | 3.9 / 3.9 / 3.9 |
| ATOM.ADD with return, global | 386.8 = 288 ns | 899 | 417 |
| RED (no return), per op + fence | 7.8 + 776 | 26 + 926 | 7.8 + 806 |
| STG per op + fence | 7.6 + 776 | 21 + 1 083 | 7.6 + 806 |
| `__syncthreads`, 256 threads | 25.8 | 21 | 25.3 |

- SASS: 48 of 78 bracket kernels verified exactly in this build; the LDG rows were flagged only because the checker
  counted the bracket's own seed load (fixed afterwards: the seed and result accesses are `STRONG.SYS` and are now
  excluded). The SHFL row is **invalid** in this run: 31 of the 32 lanes had exited before the chain and a shuffle
  with exited lanes completes trivially (slope −0.1); fixed afterwards (all lanes run the chain).
- Under the same-SM co-tenant, chains of ≤ 32 ALU instructions (≤ 130 cycles) are swamped by the ~300–700 cycles of
  issue interference, so their slopes are not resolvable with N ≤ 32; N = 128 was added afterwards.
- MPS restores L1 loads (34 cy), ALU and barrier latencies exactly; L2-tier loads stay at ~700 cycles because the L2
  is shared with the streaming process; the global atomic costs 8 % more.
- ns per instruction is reported only for brackets longer than 8 timer ticks (DRAM loads, atomics), from the same
  record's `%globaltimer` span: the DRAM load is 272 ns at the 1.41–1.48 GHz the record itself implies.
