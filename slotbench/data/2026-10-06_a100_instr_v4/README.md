# The instruction table, A100 SXM4 (host 3): alone, beside a same-SM co-tenant, under MPS (2026-10-06, commit b97ce4c)

Produced by the command-line tool on a rented A100-SXM4-40GB (driver 570.133, 108 SMs, 1024 ns timer tick):
`gputrace characterize --profile instr` → `report.md`, `summary.json`, `timeline.html`, per-run analysis and
raw records; `cloud/onstart_cli.sh` shipped the output directory. Cost $0.12. This is a third A100 host (the
v3 table and the first runs of this build, `cli_a100b` in the session notes, were on the $0.54/h host with a
bracket overhead of 360 cycles and a 388-cycle atomic; this host has 217 and 240; the ALU, L1, L2, SHFL and
barrier numbers agree to the cycle across the two).

Method as in `data/2026-10-06_3060_instr/README.md` (chains of N dependent instructions between two `%clock64`
reads, N = 1..32 and 128, 256 samples each, fit cycles = a + b·N; `sass_check.json`: 91 of 91 bracket kernels hold
exactly N target opcodes between the clock reads). Conditions: alone; one streaming block on every SM in the
same process with its 50 ms kernels queued four deep (`instr_cotenant`; fit on the 40 176 of 40 704 samples a
co-tenant block covered on the probe's SM); a streaming process under MPS with 50 % of the SMs (`instr_mps50`).

## Latency per instruction, cycles (slope b)

| instruction | alone | same-SM co-tenant | MPS 50 % |
|---|---|---|---|
| LDG.ca, 16 KB (L1) | 38.8 | 951 | 39.7 |
| LDG.ca, 1 MB (L2) | 235 (p10 210) | 950 | 237 |
| LDG.ca, 24 MB (far L2) | 368 (p10 208) | 953 | 714 |
| LDG.ca, 128 MB (DRAM) | 479 = 336 ns at 1.44 GHz (p10 214) | 960 | 720 |
| LDG.cg (L2 only), 16 KB / 1 MB / 128 MB | 286 / 280 / 515 | 949 / 943 / 949 | 281 / 706 / 717 |
| LDG.cs, 16 KB / 1 MB / 128 MB | 38.9 / 211 / 474 | 954 / 953 / 956 | 38.8 / 282 / 578 |
| LDG.nc, 16 KB / 1 MB / 128 MB | 38.9 / 211 / 475 | 950 / 945 / 957 | 38.9 / 707 / 724 |
| LDS (shared, dependent) | 23.0 | 22.9 | 23.0 |
| FADD / FFMA / IMAD (dependent) | 4.0 / 4.0 / 4.0 | 3.9 / 3.6 / 3.9 | 4.0 / 4.2 / 4.0 |
| SHFL.IDX (dependent, 32 lanes) | 26.0 | 26.4 | 26.0 |
| ATOM.ADD with return, global | 240 = 168 ns | 846 | 420 |
| RED (no return), per op + fence | 8.1 + 564 | 8.2 + 1 547 | 8.1 + 793 |
| STG per op + fence | 7.7 + 505 | 7.8 + 1 548 | 7.8 + 794 |
| `__syncthreads`, 256 threads | 36.0 | 33.9 | 34.9 |

- Same-SM streaming co-tenant: every global load ~950 cycles at every tier (L1 hits included), the atomic 3.5×,
  the fence 3×; ALU, shuffle, shared memory and the barrier unchanged to the cycle. The bracket overhead itself
  (seed load + result store) goes from 217 to ~1 070 cycles.
- MPS 50 %: L1 hits and the ALU exact; L2-tier loads 700–720 cycles (the L2 is shared with the streaming
  process; .ca at 1 MB stayed at 237 in this run, the 24 MB and 128 MB sets did not); the atomic 1.75×, the
  fence 1.5×.
- The median DRAM row (479) is above the p10 (214) because the 128 MB chase occasionally hits lines the warm
  walk left in L2; the p10 is the hit path, the median the mix.
